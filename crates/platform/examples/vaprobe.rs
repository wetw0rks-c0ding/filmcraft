//! Diagnostics for the Linux backend: what does this machine's VA-API driver offer, why does it take
//! or decline a stream, and where does a decode spend its time?
//!
//! ```text
//! vacaps                              the vendor string, the libva version and every profile
//! vaprobe [--time] [--pixels] <file>  decode a real file and report the pixels that come out
//! ```
//!
//! `--time` decodes every sample and prints the driver decode, GPU-to-CPU readback and plane
//! conversion costs per picture. `--pixels` prints the first and last decoded picture's luma samples
//! and a checksum of the whole frame, which is what says a hardware decode produced the stream's
//! picture rather than a blank surface.
#[cfg(target_os = "linux")]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let has = |flag: &str| args.iter().any(|a| a == flag);
    if has("--caps") || args.is_empty() {
        print!("{}", filmcraft_platform::vaapi::capabilities());
    }
    if args.is_empty() {
        return;
    }
    let time = has("--time");
    let pixels = has("--pixels");
    for path in args.iter().filter(|a| !a.starts_with("--")) {
        probe(path, time, pixels);
    }
}

/// One file: decode it, and say what came out.
#[cfg(target_os = "linux")]
fn probe(path: &str, time: bool, pixels: bool) {
    use filmcraft_codecs::VideoDecoder as _;
    let Ok(bytes) = std::fs::read(path) else {
        println!("{path}: unreadable");
        return;
    };
    let Ok(file) = filmcraft_isobmff::open(bytes.clone()) else {
        println!("{path}: not an MP4");
        return;
    };
    let Some(t) = file.track_of_kind(filmcraft_isobmff::TrackKind::Video) else {
        println!("{path}: no video");
        return;
    };
    let Some(entry) = file.tracks[t].entries.first() else { return };
    let info = filmcraft_codecs::hw::NalStreamInfo::from_entry(entry);
    let Some(Ok(info)) = info else {
        println!("{path}: not an avcC / hvcC stream");
        return;
    };
    println!(
        "{path}: {:?} profile {} {}x{} {} bits, decoded picture buffer declares {} pictures",
        info.codec,
        info.profile_idc,
        info.coded.0,
        info.coded.1,
        info.bit_depth_luma,
        filmcraft_platform::vaapi::surfaces_for(&info).map_or_else(|e| format!("({e})"), |n| n.to_string())
    );
    let mut d = match filmcraft_platform::vaapi::VaDecoder::new(info) {
        Ok(d) => d,
        Err(e) => {
            println!("  declined: {e}");
            return;
        }
    };
    println!("  decoder: {}", d.name());
    let (mut n, t0, mut shown) = (0usize, std::time::Instant::now(), 0usize);
    let mut first: Option<(i64, filmcraft_frame::VideoFrame)> = None;
    for i in 0..file.tracks[t].samples.len() {
        let Ok(s) = file.read_sample(&bytes, t, i) else { break };
        let pts = file.tracks[t].samples[i].pts;
        match d.decode(&s, pts) {
            Ok(frames) => {
                for f in frames {
                    n += 1;
                    if pixels && first.is_none() {
                        first = Some((f.pts, f.frame.clone()));
                    }
                    if pixels && (shown < 3 || i + 1 == file.tracks[t].samples.len()) {
                        shown += 1;
                        println!("  sample {i} pts {}: {}", f.pts, describe(&f.frame));
                    }
                }
            }
            Err(e) => {
                println!("  error at sample {i}: {e}");
                break;
            }
        }
    }
    n += d.flush().len();
    let wall = t0.elapsed();
    if time {
        let t = d.timings();
        let per = |x: std::time::Duration| x.as_secs_f64() * 1000.0 / t.frames.max(1) as f64;
        println!(
            "  {n} pictures in {:.0} ms ({:.1} fps): driver decode {:.2} ms/picture, GPU to CPU readback {:.2} ms, plane conversion {:.2} ms",
            wall.as_secs_f64() * 1000.0,
            n as f64 / wall.as_secs_f64(),
            per(t.decode),
            per(t.readback),
            per(t.convert)
        );
    } else {
        println!("  {n} pictures in {:.0} ms", wall.as_secs_f64() * 1000.0);
    }
    if let Some((pts, data)) = first {
        println!("  first picture at pts {pts}: {}", describe(&data));
    }
}

/// A frame as a row of samples and a checksum: enough to see that it is the stream's picture and
/// not a blank surface, without dumping a whole frame.
///
/// The checksum is FNV-1a/64 over **the luma plane's bytes** — one byte per 8-bit sample, two per
/// 10-bit one — so it can be compared against any other byte-level hash of the same picture (a
/// software decode, ffmpeg's VA-API decode, a reference yuv). An earlier version folded each sample
/// into four little-endian bytes whatever its depth, which made the number match nothing and sent
/// me looking for a parity that was never there.
#[cfg(target_os = "linux")]
fn describe(frame: &filmcraft_frame::VideoFrame) -> String {
    let bits: u32 = match &frame.data {
        filmcraft_frame::PixelData::Yuv8 { .. } => 8,
        filmcraft_frame::PixelData::Yuv16 { bits, .. } => *bits,
        _ => return "a format this backend does not produce".to_string(),
    };
    // The luma plane as the bytes it is on the wire, so the hash is over the same bytes a reference
    // decode of the same picture would produce.
    let bytes: Vec<u8> = match &frame.data {
        filmcraft_frame::PixelData::Yuv8 { planes, .. } => planes[0].as_ref().clone(),
        filmcraft_frame::PixelData::Yuv16 { planes, .. } => planes[0].iter().flat_map(|v| v.to_le_bytes()).collect(),
        _ => Vec::new(),
    };
    if bytes.is_empty() {
        return "an empty plane".to_string();
    }
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in &bytes {
        hash = (hash ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
    }
    // A real row of the real picture, sampled across its actual width. The stride is the frame's own
    // width, not the sample count divided by a constant: the first version of this divided the plane
    // length by a hard-coded 640, which for a 640-wide frame stepped a whole row per tap and so
    // printed one sample per *column* under a `row0` label.
    let width = usize::try_from(frame.width).unwrap_or(bytes.len()).max(1);
    // Bytes per luma sample: a 10-bit sample is a 16-bit word, which is how `biplanar::to_frame`
    // writes it. `(bits / 8).max(1)` said 1 for 10 bits, so a 10-bit plane reported twice its
    // samples and every index into it was odd — hence `luma[last]=2` and a byte-shifted row.
    let sample_bytes = if bits > 8 { 2 } else { 1 };
    let samples = bytes.len() / sample_bytes;
    let at = |i: usize| -> u32 {
        let start = i.min(samples.saturating_sub(1)) * sample_bytes;
        let lo = bytes.get(start).copied().unwrap_or(0);
        if sample_bytes == 2 { u32::from(u16::from_le_bytes([lo, bytes.get(start + 1).copied().unwrap_or(0)])) } else { u32::from(lo) }
    };
    let row: Vec<u32> = (0..12).map(|k| at(k * width / 12)).collect();
    let sum: u64 = (0..samples).map(|i| u64::from(at(i))).sum();
    format!(
        "{bits} bits, {}x{}, {samples} luma samples, row0 {row:?}, luma[centre]={} luma[last]={} mean={:.1} fnv1a={hash:016x}",
        frame.width,
        frame.height,
        at(samples / 2),
        at(samples.saturating_sub(1)),
        sum as f64 / samples as f64,
    )
}
#[cfg(not(target_os = "linux"))]
fn main() {}
