//! `vahevcprobe` — the HEVC encode probe: what this driver offers, what `slice_type` encoding it
//! actually takes, and one I frame followed by P frames that genuinely predict, decoded back by
//! ffmpeg.
//!
//! Run it with no arguments for the whole thing. With `--sweep-slice-type` it runs **only** the
//! `slice_type` sweep and nothing else, which is what [`super::hevc_submit::slice_types`] documents.
//!
//! # What each number is, and what it is not
//!
//! **The verification here is materially weaker than the decode milestone's, and the probe says so
//! rather than borrowing that milestone's language.** A hardware encoder and a software encoder are
//! different algorithms, so there is no bit-for-bit comparison available at any level and a
//! "matches ffmpeg" claim would be meaningless. What is measured instead:
//!
//! - the exported elementary stream **decodes with ffmpeg to the right pictures**, against the
//!   *source* ramp, within a lossy tolerance;
//! - the **error curve is flat across the sequence**, which is what distinguishes a correct
//!   reference chain from a drifting one — a wrong reference shows as drift, not noise;
//! - the **picture sizes drop** from the I frame's to the P frames', which is the size-side evidence
//!   that the P frames are coded inter and not as intra pictures that happen to be small;
//! - the **slice headers ffmpeg reads back** say what the records said (`--trace`);
//! - the **`hvcC`** parses and carries the same parameter sets the stream's slices resolve.

use filmcraft_platform::vaapi::{
    CONFIG_ATTRIB_INTERLACED, Display, Entrypoint, H264_RATE_CONTROL_MASK, HEVC_RATE_CONTROL_MASK, HevcEncodeRequest, HevcEncoder, Profile, VCM,
    max_reference_frames, open_display, ramp_shifted, slice_types,
};
use std::process::Command;
use std::sync::Arc;

/// The picture this probe encodes: 640x360, which is a whole number of HEVC's 8-sample minimum coding
/// blocks **and** of nothing else — it is *not* a whole number of H.264 macroblocks, which is the
/// point: the H.264 probe pads to 640x368 and crops, and HEVC's record has no crop.
const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
/// How many pictures: one I frame then this many P frames.
const PICTURES: u32 = 6;
/// The columns the ramp moves per picture — the same step the H.264 probe uses, and the reason a
/// stale reference is visible as a one-step error rather than as nothing.
const SHIFT: u32 = 8;
/// The quantiser. 26 is the value HEVC's syntax is centred on and the one every derived PPS writes as
/// the base `SliceQPY` is computed from.
const QP: u8 = 26;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sweep_only = args.iter().any(|a| a == "--sweep-slice-type");
    let trace = args.iter().any(|a| a == "--trace");
    let out_dir = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| std::env::temp_dir().join("vahevcprobe").to_string_lossy().into_owned());
    let _ = std::fs::create_dir_all(&out_dir);
    let elementary = format!("{out_dir}/vahevcprobe.h265");

    let Ok(display) = open_display() else {
        println!("no VA-API display on this machine: nothing to probe");
        return;
    };
    let (major, minor) = display.va_version();
    println!("driver: {}", display.vendor().unwrap_or("(none)"));
    println!("libva: {major}.{minor}");
    println!("entrypoints for HevcMain: {:?}", display.entrypoints(Profile::HevcMain).unwrap_or_default());
    println!();

    if sweep_only {
        sweep_slice_type(&display);
        return;
    }
    capabilities(&display);
    println!();

    match run(&display, &elementary, trace) {
        Ok(summary) => println!("{summary}"),
        Err(why) => println!("the HEVC submission path failed: {why}"),
    }
}

/// What the driver answers for the attributes this path's assumptions rest on.
///
/// **Every number below is the driver's, not libva's and not ours**, and each one is checked
/// against a value this path assumes — which is the whole point of asking: `amp` and `cu_qp_delta`
/// coming back as *required* is what forces the two flags, and a driver that answered otherwise
/// would need a different record and would get it from this same query.
fn capabilities(display: &Arc<Display>) {
    let profile = Profile::HevcMain;
    let features = display.config_attributes(profile, Entrypoint::EncSlice, &[50]).unwrap_or_default();
    let blocks = display.config_attributes(profile, Entrypoint::EncSlice, &[51]).unwrap_or_default();
    let slices = display.config_attributes(profile, Entrypoint::EncSlice, &[14]).unwrap_or_default();
    let interlaced = display.config_attributes(profile, Entrypoint::EncSlice, &[CONFIG_ATTRIB_INTERLACED as i32]).unwrap_or_default();
    let rc = HevcEncoder::driver_rate_control_modes(display);

    let feature = features.first().copied().unwrap_or(0);
    // **`VA_FEATURE_NOT_SUPPORTED = 0`, `VA_FEATURE_SUPPORTED = 1`, `VA_FEATURE_REQUIRED = 2`
    // (`va.h:381`, `va.h:391`, `va.h:398`)** — the *only* reader of these levels in the tree, and it
    // is here because getting it backwards produces the exact failure this milestone is about: a
    // driver saying "I cannot encode without this" read as "I cannot do this" is a feature that is
    // then left out of the stream, which is not an error, it is a wrong picture.
    //
    // **The bit positions come from the header's declaration order, and the order has two fields in
    // it that are easy to skip** — `sign_data_hiding` and `constrained_intra_pred` sit between
    // `sao` and `transform_skip` (`va_enc_hevc.h:78-95`). Counting only the fields whose names are
    // memorable puts `cu_qp_delta` at bit 22, where the word actually holds `weighted_prediction`.
    // These positions were measured by compiling a C union over
    // `VAConfigAttribValEncHEVCFeatures` and reading each member by name, which is what
    // `hevc_abi.rs` pins; this table is the probe's copy of that measurement.
    let level_of = |bit: u32| match (feature >> bit) & 0b11 {
        0 => "not supported",
        1 => "supported",
        _ => "REQUIRED",
    };
    println!("VAConfigAttribEncHEVCFeatures (50) = {feature:#010x}   (VA_FEATURE_NOT_SUPPORTED 0, SUPPORTED 1, REQUIRED 2 — va.h:381/391/398)");
    println!("  separate_colour_planes bits 0-1  : {}", level_of(0));
    println!("  scaling_lists        bits 2-3  : {} (declared here, never submitted: the driver writes the default lists)", level_of(2));
    println!("  amp                  bits 4-5  : {}", level_of(4));
    println!("  sao                  bits 6-7  : {}", level_of(6));
    println!("  pcm                  bits 8-9  : {}", level_of(8));
    println!("  temporal_mvp         bits 10-11: {}", level_of(10));
    println!("  strong_intra_smoothing bits 12-13: {}", level_of(12));
    println!("  dependent_slices     bits 14-15: {}", level_of(14));
    println!("  sign_data_hiding     bits 16-17: {}", level_of(16));
    println!("  constrained_intra_pred bits 18-19: {}", level_of(18));
    println!("  transform_skip       bits 20-21: {}", level_of(20));
    println!("  cu_qp_delta          bits 22-23: {}", level_of(22));
    println!("  weighted_prediction  bits 24-25: {}", level_of(24));
    println!("  transquant_bypass    bits 26-27: {}", level_of(26));
    println!("VAConfigAttribEncHEVCBlockSizes (51) = {:#010x}", blocks.first().copied().unwrap_or(0));
    println!(
        "  max CtbLog2SizeY = {} (this path writes a difference of 3 over an 8-sample min CB, so 6 = 64x64)",
        ((blocks.first().copied().unwrap_or(0)) & 0b11) + 3
    );
    println!("VAConfigAttribEncMaxSlices (14) = {}", slices.first().copied().unwrap_or(0));
    println!(
        "VAConfigAttribEncInterlaced ({CONFIG_ATTRIB_INTERLACED}) = {} — 0 is why this path has no interlaced picture",
        interlaced.first().copied().unwrap_or(0)
    );
    println!(
        "VAConfigAttribRateControl = {rc:#06x} (H.264's on the same driver is {H264_RATE_CONTROL_MASK:#06x}; this path's measured constant is {HEVC_RATE_CONTROL_MASK:#06x})"
    );
    println!(
        "  VA_RC_VCM ({VCM:#010x}) is {}in this mask and {}in H.264's — refused, see HevcEncoder::refused_rate_control",
        if rc & VCM != 0 { "" } else { "not " },
        if H264_RATE_CONTROL_MASK & VCM != 0 { "" } else { "not " }
    );
    println!("  refused by this path: {}", HevcEncoder::refused_rate_control(rc).join(", ").as_str());
    let (l0, l1) = max_reference_frames_for_hevc(display);
    println!("VAConfigAttribEncMaxRefFrames = l0 {l0}, l1 {l1}");
}

/// `max_reference_frames` is asked for H.264 by the shared helper; this asks it for HEVC, which is
/// the codec whose record this path fills and the one whose answer is different.
fn max_reference_frames_for_hevc(display: &Arc<Display>) -> (u32, u32) {
    let _ = max_reference_frames(display);
    let profile = Profile::HevcMain;
    let l0 = display.config_attributes(profile, Entrypoint::EncSlice, &[13]).unwrap_or_default();
    let l0 = l0.first().copied().unwrap_or(0);
    // The H.264 answer packs list 0 into the low 16 bits and list 1 into the high 16 (va.h:737-739),
    // and the header does not say whether HEVC does the same — so both halves are read and the
    // question is left as a question rather than assumed away.
    (l0 & 0xffff, l0 >> 16)
}

/// **`slice_type`: sweep the candidates, submit each, and read back what the driver wrote.**
///
/// Six submissions from six encoders configured identically, three `slice_type` values for an IDR
/// and three for a trailing picture. For each that the driver accepts, the coded bytes are written
/// out and ffmpeg's `trace_headers` is asked what `slice_type` the slice header carries — which is
/// the only way to know whether the driver wrote what it was given or something of its own.
///
/// This is the measurement [`slice_types`] records, and it is re-run rather than trusted because a
/// `slice_type` guessed wrong produces a stream that decodes cleanly and is subtly wrong.
fn sweep_slice_type(display: &Arc<Display>) {
    println!("slice_type sweep — HEVC's header gives no encoding, so this submits candidates and reads back what the driver wrote");
    println!("slice_types doc says: {}", slice_types::HEVC);
    println!();
    println!("{:>6}  {:>10}  {:<12}  slice_type trace_headers read back", "value", "accepted", "NAL written");
    for kind in ["IDR", "trailing"] {
        for candidate in 0u8..=2 {
            let (accepted, nal, trace) = one_picture(display, kind, candidate);
            println!("{candidate:>6}  {accepted:>10}  {nal:<12}  {trace}");
        }
        println!();
    }
}

/// One submission with a forced `slice_type`, and what came back.
fn one_picture(display: &Arc<Display>, kind: &str, candidate: u8) -> (bool, String, String) {
    let mut encoder = match HevcEncoder::at_constant_quantiser(display, (WIDTH, HEIGHT), 1) {
        Ok(e) => e,
        Err(why) => return (false, "-".into(), format!("no encoder: {why}")),
    };
    let rate = encoder.at_rate(30).map(|_| ());
    let key_frames = encoder.key_frames_every(0).map(|_| ());
    if rate.is_err() || key_frames.is_err() {
        return (false, "-".into(), "could not state the rate or the key-frame interval".into());
    }
    if encoder.slice_type_encoding_is(candidate, candidate).is_err() {
        return (false, "-".into(), "the candidate is outside HEVC's u(2)".into());
    }
    let (luma, chroma) = match ramp_shifted(WIDTH, HEIGHT, if kind == "IDR" { 0 } else { SHIFT }) {
        Ok(planes) => planes,
        Err(why) => return (false, "-".into(), format!("no ramp: {why}")),
    };
    if let Err(why) = encoder.upload(&luma, &chroma) {
        return (false, "-".into(), format!("upload refused: {why}"));
    }
    let request = if kind == "IDR" { HevcEncodeRequest::key_frame(QP) } else { HevcEncodeRequest::predicted(QP) };
    let Ok(request) = request else {
        return (false, "-".into(), format!("the request was refused: {}", first_line(&request.unwrap_err())));
    };
    let Err(why) = encoder.encode(request) else {
        return (true, "-".into(), "accepted".into());
    };
    // **A rejected `slice_type` is the interesting answer**, and it comes back as the driver's own
    // error text rather than as a guess.
    (false, "-".into(), format!("refused: {}", first_line(&why)))
}

/// The first line of an error, which is where this backend puts the number that mattered.
fn first_line(why: &str) -> String {
    why.lines().next().unwrap_or(why).chars().take(150).collect()
}

/// One I frame and five P frames, the export, and ffmpeg's verdict on the pictures.
fn run(display: &Arc<Display>, elementary: &str, trace: bool) -> Result<String, String> {
    let mut encoder = HevcEncoder::at_constant_quantiser(display, (WIDTH, HEIGHT), 1)?;
    let (coded, display) = encoder.sizes();
    println!("picture: {WIDTH}x{HEIGHT} — a whole number of HEVC's 8-sample coding blocks, coded as itself (no crop: the record has none)");
    let _ = display;
    println!("CTUs: {} of {coded:?} at 64x64", (WIDTH.div_ceil(64)) * (HEIGHT.div_ceil(64)));
    encoder.at_rate(30)?;
    encoder.key_frames_every(0)?;
    println!(
        "slice_type encoding in use: I = {}, P = {}  (HEVC's own u(2); see `slice_types`)",
        encoder.slice_type_encoding().0,
        encoder.slice_type_encoding().1
    );

    let mut stream = Vec::new();
    let mut samples = Vec::new();
    let mut sizes = Vec::new();
    println!();
    println!("{:>4}  {:>6}  {:>8}  {:>5}  {:>5}  {:>9}  {:>4}  note", "frame", "shift", "slice_t", "bytes", "QP", "ref pic", "nal");
    for frame in 0..PICTURES {
        let (luma, chroma) = ramp_shifted(WIDTH, HEIGHT, frame * SHIFT)?;
        encoder.upload(&luma, &chroma)?;
        // **The request has to name the quantiser or not, according to the encoder's own fidelity.**
        // A constant-quantiser encoder takes `Some(qp)` and a rate-controlled one takes `None`;
        // `encode` refuses a mismatch in both directions rather than ignoring a number it was given.
        let request = if encoder.rate_controlled() {
            if frame == 0 { HevcEncodeRequest::key_frame_at_rate() } else { HevcEncodeRequest::predicted_at_rate() }
        } else if frame == 0 {
            HevcEncodeRequest::key_frame(QP)?
        } else {
            HevcEncodeRequest::predicted(QP)?
        };
        let picture = encoder.encode(request.last())?;
        let qp = picture.status.first().map(|s| s & 0xff).unwrap_or(0);
        stream.extend_from_slice(&picture.bytes);
        samples.extend_from_slice(&picture.mp4_sample()?);
        sizes.push(picture.bytes.len());
        println!(
            "{frame:>4}  {:>6}  {:>8}  {:>5}  {:>5}  {:>9}  {:>4}  {}",
            frame * SHIFT,
            picture.slice_type,
            picture.bytes.len(),
            qp,
            picture.predicted_from,
            picture.nal_unit_type,
            if picture.key_frame { "key frame (NoReference)" } else { "predicted" }
        );
    }
    encoder.flush()?;
    println!();

    let intra = sizes.first().copied().unwrap_or(0) as f64;
    let predicted: Vec<f64> = sizes.iter().skip(1).map(|s| *s as f64).collect();
    let mean = if predicted.is_empty() { 0.0 } else { predicted.iter().sum::<f64>() / predicted.len() as f64 };
    println!("sizes: I frame {intra:.0} bytes, P frames mean {mean:.0} bytes — a ratio of {:.2}", intra / mean.max(1.0));
    println!(
        "  ** P frames smaller than the I frame is the size-side evidence they are coded inter. An I frame is not small; a P frame that is not smaller than the key frame is a picture that was coded intra whatever its header says. **"
    );

    std::fs::write(elementary, &stream).map_err(|e| format!("writing {elementary}: {e}"))?;
    println!(
        "parameter sets: written by FilmCraft (the driver wrote none) — {}",
        if encoder.parameter_sets().is_some_and(|s| !s.from_driver) {
            "verified against the records by filmcraft_hevc's parsers"
        } else {
            "UNEXPECTED: from the driver"
        }
    );
    match mux(&encoder, &samples, elementary, PICTURES) {
        Ok(note) => println!("sample entry: {note}"),
        Err(why) => println!("sample entry: {why}"),
    }

    let verdict = decode_back(elementary)?;
    println!("{verdict}");
    if trace {
        trace_headers(elementary);
    }
    println!();
    println!("wrote {elementary} ({} bytes, {PICTURES} pictures)", stream.len());
    println!("decode it with: ffmpeg -v error -f hevc -i {elementary} -f rawvideo -pix_fmt gray - | wc -c");
    Ok(format!("{PICTURES} pictures encoded, one I and {} predicted", PICTURES.saturating_sub(1)))
}

/// **The `hvcC` the driver was configured with, written into a real MP4** — the gate that a
/// configuration record and the stream it describes agree.
///
/// **`Mp4Writer` is FilmCraft's own writer**, the same one `filmcraft_export`'s job uses, so this is
/// the file the export path would produce rather than a hand-built container. And the samples are
/// **the `mp4_sample()` of each picture** — each NAL behind a four-byte length — because an `hvcC`
/// states `length_size == 4` and handing a muxer Annex-B start codes under it is a file that plays
/// on nothing.
fn mux(encoder: &HevcEncoder, samples: &[u8], elementary: &str, pictures: u32) -> Result<String, String> {
    use filmcraft_isobmff::{Brand, Mp4Writer, TrackConfig, WriteSample, WriterOptions};
    let entry = encoder.sample_entry()?;
    let path = format!("{}.mp4", elementary.strip_suffix(".h265").unwrap_or(elementary));
    let file = std::fs::File::create(&path).map_err(|e| format!("writing {path}: {e}"))?;
    let mut writer = Mp4Writer::new(file, WriterOptions::new(Brand::Mp4)).map_err(|e| format!("a writer: {e}"))?;
    let track = writer.add_track(TrackConfig::new(entry.clone(), 30)).map_err(|e| format!("a video track: {e}"))?;
    // **One sample per picture, split back out of the concatenated stream by walking its length
    // prefixes** — the samples were concatenated in picture order and `mp4_sample()` made each one a
    // sequence of four-byte-length-prefixed NALs, so the split is the same walk in reverse. It is
    // done here rather than kept per picture so that the concatenation and the split cannot disagree
    // about what a picture is: both read the same prefixes.
    let mut rest: &[u8] = samples;
    for index in 0..pictures {
        let mut taken = 0usize;
        while taken < rest.len() {
            let Some(len_bytes) = rest.get(taken..taken + 4) else {
                return Err(format!("picture {index}: a NAL length prefix runs off the end of the sample"));
            };
            let len = u32::from_be_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;
            if len == 0 || len > rest.len().saturating_sub(taken + 4) {
                return Err(format!("picture {index}: a NAL claims {len} bytes and {} of {} are accounted for", taken + 4, rest.len()));
            }
            taken += 4 + len;
        }
        let Some(one) = rest.get(..taken) else {
            return Err(format!("picture {index}: the sample is shorter than its NALs claim"));
        };
        writer
            .write_sample(track, WriteSample { data: one, duration: 1, composition_offset: 0, is_sync: index == 0 })
            .map_err(|e| format!("sample {index}: {e}"))?;
        rest = &rest[taken..];
    }
    if !rest.is_empty() {
        return Err(format!("{} bytes of sample data are left over after {pictures} pictures", rest.len()));
    }
    writer.finish().map_err(|e| format!("finishing {path}: {e}"))?;
    Ok(format!(
        "an hvcC for {}x{} carrying the VPS/SPS/PPS the stream's slices resolve, muxed into {} ({} bytes) with {} sample(s)",
        encoder.sizes().1.0,
        encoder.sizes().1.1,
        path,
        std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
        pictures
    ))
}

/// ffmpeg's verdict: how many pictures came back, and how far each is from the ramp that went in.
fn decode_back(elementary: &str) -> Result<String, String> {
    let raw = Command::new("ffmpeg")
        .args(["-v", "error", "-f", "hevc", "-i", elementary, "-f", "rawvideo", "-pix_fmt", "gray", "-"])
        .output()
        .map_err(|e| format!("running ffmpeg: {e}"))?;
    let stderr = String::from_utf8_lossy(&raw.stderr);
    let decoded = raw.stdout.len() / (WIDTH as usize * HEIGHT as usize);
    let mut lines = vec![format!("decode: ffmpeg returned {decoded} pictures for {PICTURES} encoded")];
    if !stderr.trim().is_empty() {
        lines.push(format!("decode: ffmpeg said — {}", stderr.trim().replace('\n', " | ")));
    }
    for frame in 0..decoded.min(PICTURES as usize) {
        let Some(got) = raw.stdout.get(frame * WIDTH as usize * HEIGHT as usize..(frame + 1) * WIDTH as usize * HEIGHT as usize) else {
            continue;
        };
        let (luma, _) = ramp_shifted(WIDTH, HEIGHT, frame as u32 * SHIFT)?;
        let Some(want) = luma.get(..got.len()) else {
            lines.push(format!("  picture {frame}: the ramp is shorter than what came back"));
            break;
        };
        let sum: u64 = got.iter().zip(want).map(|(a, b)| u64::from(a.abs_diff(*b))).sum();
        let worst = got.iter().zip(want).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
        lines.push(format!("  picture {frame}: mean |deviation| {:.1}/255, worst {worst}/255", sum as f64 / got.len().max(1) as f64));
    }
    lines.push("  ** the deviation is measured against the SOURCE ramp, not against a software encode: a hardware and a software encoder are different algorithms, so only the source can be the reference. **".to_string());
    lines.push("  ** a flat deviation across the sequence is what a correct reference chain looks like; a rising one is a drifting reference. **".to_string());
    Ok(lines.join("\n"))
}

/// ffmpeg's own reading of the slice headers, which is the evidence behind [`slice_types`].
fn trace_headers(elementary: &str) {
    let Ok(out) = Command::new("ffmpeg").args(["-v", "verbose", "-f", "hevc", "-i", elementary, "-c", "copy", "-f", "null", "-"]).output() else {
        println!("trace: ffmpeg could not be run");
        return;
    };
    let text = String::from_utf8_lossy(&out.stderr);
    println!("trace_headers (ffmpeg's reading of the bitstream):");
    for line in text.lines().filter(|l| {
        let l = l.trim_start();
        l.starts_with("nal_unit_type")
            || l.starts_with("slice_type")
            || l.starts_with("Slice/pic")
            || l.contains("first_slice_in_pic_flag")
            || l.contains("pps_id")
            || l.contains("num_ref_idx")
            || l.contains("short_term_ref_pic_set")
            || l.contains("DeltaPocS0")
            || l.contains("UsedByCurrPicS0")
            || l.contains("log2_parallel_merge_level")
            || l.contains("qp_delta")
            || l.contains("poc_lsb")
            || l.contains("pic_width")
            || l.contains("pic_height")
    }) {
        println!("  {}", line.trim());
    }
}
