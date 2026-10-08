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
        "{path}: {:?} profile {} {}x{} {} bits, {} surfaces needed",
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
    let mut first: Option<(i64, filmcraft_frame::PixelData)> = None;
    for i in 0..file.tracks[t].samples.len() {
        let Ok(s) = file.read_sample(&bytes, t, i) else { break };
        let pts = file.tracks[t].samples[i].pts;
        match d.decode(&s, pts) {
            Ok(frames) => {
                for f in frames {
                    n += 1;
                    if pixels && first.is_none() {
                        first = Some((f.pts, f.frame.data.clone()));
                    }
                    if pixels && (shown < 3 || i + 1 == file.tracks[t].samples.len()) {
                        shown += 1;
                        println!("  sample {i} pts {}: {}", f.pts, describe(&f.frame.data));
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

/// A frame as its corner samples and a checksum: enough to see that it is the stream's picture and
/// not a black surface, without dumping a whole frame.
#[cfg(target_os = "linux")]
fn describe(data: &filmcraft_frame::PixelData) -> String {
    let bits: u32 = match data {
        filmcraft_frame::PixelData::Yuv8 { .. } => 8,
        filmcraft_frame::PixelData::Yuv16 { bits, .. } => *bits,
        _ => return "a format this backend does not produce".to_string(),
    };
    let luma: Vec<u32> = match data {
        filmcraft_frame::PixelData::Yuv8 { planes, .. } => planes[0].iter().map(|&b| u32::from(b)).collect(),
        filmcraft_frame::PixelData::Yuv16 { planes, .. } => planes[0].iter().map(|&v| u32::from(v)).collect(),
        _ => Vec::new(),
    };
    if luma.is_empty() {
        return "an empty plane".to_string();
    }
    // FNV-1a over the luma plane: a hash, so "the whole picture" is one number rather than a page.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &v in &luma {
        for b in v.to_le_bytes() {
            hash = (hash ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }
    let corner = |at: usize| luma[at.min(luma.len() - 1)];
    // Twelve samples spread across the first row: enough to see the picture's own structure rather
    // than a uniform surface, which is what a decode that never rendered looks like (a driver that
    // accepts the records and renders nothing leaves every sample at NV12's neutral 128).
    let row: Vec<u32> = (0..12).map(|k| corner(k * (luma.len() / 640))).collect();
    format!(
        "{bits} bits, {} luma samples, row0 {row:?}, luma[centre]={} luma[last]={} mean={:.1} fnv1a={hash:016x}",
        luma.len(),
        corner(luma.len() / 2),
        corner(luma.len() - 1),
        luma.iter().map(|&v| f64::from(v)).sum::<f64>() / luma.len() as f64,
    )
}
#[cfg(not(target_os = "linux"))]
fn main() {}
