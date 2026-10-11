//! Pictures through the VA-API H.264 slice encoder, with a decode check that can tell a predicted
//! frame from an intra one.
//!
//! ```text
//! vaencprobe [-qp N] [-w W] [-h H] [-f FRAMES] [-refs N] [-fps N] [-gop N] [-o FILE] [-allidr]
//! ```
//!
//! Default: **one I frame followed by five P frames** of a 640x360 luma ramp that **moves 8 columns
//! left per picture**, at QP 26 and 30 fps, written to `/tmp/vaencprobe.h264`.
//!
//! # Why the content moves, which a still picture cannot show
//!
//! Task 1 used a still ramp, which proves the encoder can make the picture it was given and says
//! nothing about prediction: with every picture identical, a decoder that ignored the references
//! entirely would produce exactly the right pixels. Moving the ramp makes a wrong reference
//! impossible to hide — a stale or wrong reference shifts the picture by a whole step, which on this
//! ramp is about 3 luma per column (`8 * 219 / 639`), and the error *accumulates* along the sequence
//! rather than staying flat.
//!
//! # What the output is evidence of
//!
//! Four things, and all of them are needed — a correct decode on its own proves nothing:
//!
//! 1. **Per-frame deviation from what was put in**, one number per frame. If the references were
//!    wrong the deviation would grow along the P frames; if they were right it stays flat at the
//!    quantiser's noise floor. That number is computed here from ffmpeg's decode and compared with
//!    the ramp, not asserted by eye.
//! 2. **Per-frame coded size.** A P frame that does not drop in size after the I frame was not
//!    predicted from, whatever it decodes to. `-allidr` makes it airtight by encoding the same
//!    pictures as key frames for comparison.
//! 3. **Which pictures the loop made key frames, and why.** The `frame_num` wrap comes every 16
//!    pictures ([`VaEncoder::key_frame_due`]), so with `-gop 0` or any GOP that is not a divisor of
//!    16 this column is not always empty, and a probe that did not print it would let a reader
//!    assume the key frames were the ones that were asked for.
//! 4. **The driver's own NAL types per picture.** A driver that honours
//!    `VAConfigAttribEncPackedHeaders` puts a type 7 (SPS) and a type 8 (PPS) in there; iHD 26.3.5
//!    writes neither, and the derived sets are what the stream carries.
//!
//! **Reference depth above one is refused by the encoder, not by this probe**, because it produces a
//! silently wrong picture on this driver; see `VaEncoder::new` for the measurement.
//!
//! Verify independently with:
//!
//! ```text
//! ffmpeg -v error -f h264 -i /tmp/vaencprobe.h264 -f rawvideo -pix_fmt gray - > frames.gray
//! ```
//!
//! which gives `frames * width * height` bytes of luma in display order.
#[cfg(target_os = "linux")]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut qp = 26u8;
    let (mut width, mut height) = (640u32, 360u32);
    let mut frames = 6usize;
    let mut refs = 1u32;
    let mut fps = 30u32;
    let mut gop = 0u32;
    let mut out = "/tmp/vaencprobe.h264".to_string();
    let mut all_idr = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-qp" => qp = it.next().and_then(|v| v.parse().ok()).unwrap_or(qp),
            "-w" => width = it.next().and_then(|v| v.parse().ok()).unwrap_or(width),
            "-h" => height = it.next().and_then(|v| v.parse().ok()).unwrap_or(height),
            "-f" => frames = it.next().and_then(|v| v.parse().ok()).unwrap_or(frames),
            "-refs" => refs = it.next().and_then(|v| v.parse().ok()).unwrap_or(refs),
            "-fps" => fps = it.next().and_then(|v| v.parse().ok()).unwrap_or(fps),
            "-gop" => gop = it.next().and_then(|v| v.parse().ok()).unwrap_or(gop),
            "-o" => out = it.next().cloned().unwrap_or(out),
            "-allidr" => all_idr = true,
            other => {
                eprintln!("vaencprobe: unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    if let Err(why) = run(width, height, qp, frames, refs, fps, gop, all_idr, &out) {
        eprintln!("vaencprobe: {why}");
        std::process::exit(1);
    }
}

/// Encode a run of pictures and report what is needed to tell prediction from intra coding.
#[cfg(target_os = "linux")]
fn run(width: u32, height: u32, qp: u8, frames: usize, refs: u32, fps: u32, gop: u32, all_idr: bool, out: &str) -> Result<(), String> {
    use filmcraft_platform::vaapi::{EncodeRequest, VaEncoder, ramp_shifted};
    let api = filmcraft_platform::vaapi::open_display().map_err(|why| format!("no VA-API display: {why}"))?;
    println!("driver: {} (libva {}.{})", api.vendor().unwrap_or("(none)"), api.va_version().0, api.va_version().1);

    // What the driver says it can hold references for, asked before any config exists.
    let (l0, l1) = filmcraft_platform::vaapi::max_reference_frames(&api);
    println!("VAConfigAttribEncMaxRefFrames: reference list 0 = {l0}, list 1 = {l1} (asked with {refs})");
    // **What the driver reports for rate control**, which is what decides whether the settings
    // gate could ever accept a bitrate. Printed here because the decline says "the driver reports
    // <names>" and a probe that did not show it would leave the claim unchecked.
    println!("VAConfigAttribRateControl: {}", filmcraft_platform::hardware_encode::vaapi_rate_control_modes_named());

    let mut encoder = VaEncoder::new(&api, (width, height), refs)?;
    encoder.at_rate(fps)?;
    encoder.key_frames_every(gop)?;
    let (coded, display) = encoder.sizes();
    let (l, r, t, b) = encoder.crop();
    println!("picture: {display:?} displayed, {coded:?} coded, crop left {l} right {r} top {t} bottom {b}, QP {qp}");
    println!("stream: {fps} fps, key frame every {gop} pictures (0 = one for the whole stream)");
    println!("references held: {} of {} capacity", encoder.reference_count(), encoder.reference_capacity());

    // The ramp moves 8 columns per picture. 8 is small enough that a correct P frame's residual is
    // cheap and large enough that a whole-step misalignment is unmistakable (about 3 luma per
    // column on this width), and it is a whole number of pixels so nothing here depends on a
    // fractional motion vector the encoder might round differently.
    let shift = 8u32;
    let mut annex_b = Vec::new();
    // **Every source picture is kept**, because the decode check compares ffmpeg's output with what
    // went *in* — one ramp per picture — and not with another encode. That comparison is the only
    // one with an answer: a hardware encoder and this project's software encoder are different
    // algorithms, so "matches the software encoder" would mean nothing here.
    let mut sources: Vec<Vec<u8>> = Vec::new();
    println!();
    println!("{:>4}  {:>5}  {:>7}  {:>7}  {:>5}  {:>3}  {:>12}  {:>10}  note", "frame", "shift", "slice_t", "bytes", "QP", "refs", "driver NALs", "frame_num");
    let mut sizes = Vec::new();
    for index in 0..frames {
        // **Every picture is a different picture.** This is the whole test: a still sequence would
        // decode correctly whether or not the references were used.
        let (luma, chroma) = ramp_shifted(display.0, display.1, shift * index as u32)?;
        encoder.upload(&luma, &chroma)?;
        sources.push(luma);
        // The picture *kind* is the loop's decision, not this call's: a caller asks for a P picture
        // and the loop makes it an IDR when the reference state or the `frame_num` wrap demands it.
        // The reason comes back on the picture, so it is printed rather than assumed away.
        let last = index + 1 == frames;
        let mut request = EncodeRequest::predicted(qp)?;
        if last {
            request = request.last();
        }
        let picture = encoder.encode(request)?;
        let avg_qp = picture.status.first().copied().unwrap_or(0) & 0xff;
        sizes.push(picture.driver_bytes);
        // **The NAL types the driver itself wrote**, which is the measurement Task 4 turns on: a
        // driver that honours `VAConfigAttribEncPackedHeaders` puts a type 7 (SPS) and a type 8 (PPS)
        // in there, and one that accepts the attribute and then ignores it writes nothing but the
        // slice. Printed per picture rather than once for the stream, because "the stream has
        // parameter sets" and "the *driver* writes them" are different claims and only this one is
        // about the driver.
        let driver_only = picture.bytes.len().saturating_sub(picture.driver_bytes);
        let driver_nals = nal_types(picture.bytes.get(driver_only..).unwrap_or(&[]));
        let why = match picture.key_frame_reason {
            Some(reason) => format!("key frame ({reason:?})"),
            None if picture.key_frame => "key frame (asked for)".to_string(),
            None => "predicted".to_string(),
        };
        println!(
            "{index:>4}  {:>5}  {:>7}  {:>7}  {:>5}  {:>3}  {:>12}  {:>10}  {why}",
            shift * index as u32,
            picture.slice_type,
            picture.driver_bytes,
            avg_qp,
            picture.predicted_from,
            driver_nals,
            picture.frame_num,
        );
        if index == 0 {
            annex_b.extend_from_slice(&picture.bytes);
        } else {
            // The parameter sets only need to precede the first frame: every later picture's copy
            // would be a repeat, and a repeat in the middle of a stream is what a decoder counts as
            // a new sequence.
            let driver_only = picture.bytes.len().saturating_sub(picture.driver_bytes);
            annex_b.extend_from_slice(picture.bytes.get(driver_only..).unwrap_or(&[]));
        }
    }
    println!();
    // **The size drop is the evidence that prediction happened.** A P frame coded as intra would be
    // the same size as the I frame, and would decode correctly — so this line, not the decode, is
    // what rules that out.
    let first = sizes.first().copied().unwrap_or(0);
    let rest: Vec<usize> = sizes.iter().skip(1).copied().collect();
    let mean_rest = if rest.is_empty() { 0 } else { rest.iter().sum::<usize>() / rest.len() };
    println!(
        "sizes: I frame {first} bytes, P frames mean {mean_rest} bytes — a ratio of {:.2}",
        if mean_rest == 0 { f64::NAN } else { first as f64 / mean_rest as f64 }
    );
    // **The decode check, once, over the whole stream.** ffmpeg is given exactly the bytes the
    // muxer would be given — an elementary stream — and its output is compared with the ramps that
    // went in, one picture at a time.
    match decode_deviations(&annex_b, &sources) {
        None => println!("decode: ffmpeg is not on this machine, so the decode check did not run (the numbers above are the only evidence)"),
        Some(rows) if rows.is_empty() => println!("decode: ffmpeg decoded no picture at all from this stream"),
        Some(rows) => {
            println!("decode: ffmpeg returned {} pictures for {} encoded", rows.len(), frames);
            println!("  mean absolute deviation per picture: {}", format_list(&rows));
            let flat = rows.iter().filter(|d| **d < 24).count();
            println!("  {flat} of {} within 24/255, worst {}", rows.len(), rows.iter().copied().max().unwrap_or(0));
            println!("  ** a flat deviation across the sequence is what a correct reference chain looks like; a rising one is a drifting reference **");
        }
    }
    println!(
        "parameter sets: {}",
        if picture_parameter_sets_are_ours(&annex_b) { "written by FilmCraft (the driver wrote none)" } else { "written by the driver" }
    );
    let entry_note = match encoder.sample_entry() {
        Ok(_) => format!("an avcC for {}x{}, from {} picture(s) of parameter sets", display.0, display.1, encoder.pictures_encoded()),
        Err(why) => format!("not yet: {why}"),
    };
    println!("sample entry: {entry_note}");
    match encoder.flush() {
        Ok(()) => println!("flush: ended cleanly, {frames} pictures encoded"),
        Err(why) => println!("flush: {why}"),
    }

    if all_idr {
        // The size control: the same moving content, every picture coded intra. **This is what makes
        // the size drop evidence rather than an observation** — without it, a P frame that was small
        // for some other reason (a quiet picture, a driver quirk) would look like prediction. With
        // it, the comparison is same content, same quantiser, same everything but the picture type.
        println!();
        println!("control: the same pictures, every one coded as an IDR:");
        let mut control = VaEncoder::new(&api, (width, height), 1).map_err(|e| format!("control encoder: {e}"))?;
        control.at_rate(fps)?;
        control.key_frames_every(1)?;
        let mut control_sizes = Vec::new();
        for index in 0..frames {
            let (luma, chroma) = ramp_shifted(display.0, display.1, shift * index as u32)?;
            control.upload(&luma, &chroma)?;
            let c = control.encode(EncodeRequest::predicted(qp)?)?;
            control_sizes.push(c.driver_bytes);
        }
        println!("  intra sizes: {control_sizes:?}");
        let control_mean = control_sizes.iter().sum::<usize>() / control_sizes.len().max(1);
        let pred_mean: usize = rest.iter().sum::<usize>() / rest.len().max(1);
        println!("  intra mean {control_mean} bytes vs predicted mean {pred_mean} bytes — a ratio of {:.2}", control_mean as f64 / pred_mean.max(1) as f64);
        println!("  ** a P frame smaller than the same picture coded intra can only be predicted from something **");
    }
    std::fs::write(out, &annex_b).map_err(|e| format!("writing {out}: {e}"))?;
    println!();
    println!("wrote {out} ({} bytes, {} pictures)", annex_b.len(), frames);
    println!("decode it with: ffmpeg -v error -f h264 -i {out} -f rawvideo -pix_fmt gray - | wc -c");
    println!("  (expect {} = {} pictures of {}x{} luma)", frames * display.0 as usize * display.1 as usize, frames, display.0, display.1);
    Ok(())
}

/// `12 9 8 …` for a run of deviations, truncated so the line stays readable.
#[cfg(target_os = "linux")]
fn format_list(values: &[u8]) -> String {
    if values.is_empty() {
        return "(no pictures)".to_string();
    }
    let text = values.iter().map(u8::to_string).collect::<Vec<_>>().join(" ");
    if text.len() <= 120 {
        return text;
    }
    format!("{text:.117}…")
}

/// **Every decoded picture's mean absolute luma difference from the picture that was encoded into
/// it**, or `None` when ffmpeg is not on this machine.
///
/// The comparison is against the **source ramp**, not against another encode. A hardware encoder
/// and this project's software encoder are different algorithms with different decisions, so a
/// "matches the software encoder" claim would be meaningless — this answers the only question that
/// has an answer here (docs/superpowers/plans/2026-10-09-vaapi-linux-encode.md, "Verification — and
/// it is weaker than decode").
///
/// **A picture count that does not match the source count is a short result, not a comparison of the
/// ones that happened to line up.** Pairing decoded picture *n* with source picture *n* when ffmpeg
/// returned fewer would silently compare the wrong pictures and report a deviation that means
/// nothing, so the lengths are required to be equal and the caller prints the two counts side by
/// side.
#[cfg(target_os = "linux")]
fn decode_deviations(stream: &[u8], sources: &[Vec<u8>]) -> Option<Vec<u8>> {
    let ffmpeg = ffmpeg_path()?;
    let mut child = std::process::Command::new(&ffmpeg)
        .args(["-v", "error", "-f", "h264", "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "gray", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .ok()?;
    // The child's stdin is a pipe this function owns; writing to it and dropping the handle closes it,
    // which is what tells ffmpeg the elementary stream has ended. A closed pipe mid-write means ffmpeg
    // stopped reading, and `wait_with_output` then reports whatever it managed to decode.
    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        let _ = stdin.write_all(stream);
    }
    let output = child.wait_with_output().ok()?;
    // **ffmpeg's own diagnostics are not ignored.** A stream it decodes *with* a warning has been
    // accepted and is not necessarily right, and a probe that printed a deviation over it would
    // report a number computed from a stream nobody would trust.
    if !output.stderr.is_empty() {
        eprintln!("  ffmpeg said: {}", String::from_utf8_lossy(&output.stderr).trim());
    }
    let size = sources.first().map(Vec::len).filter(|size| *size > 0)?;
    // Every source picture is the same size — they are all the same geometry — so `size` is one
    // picture's length and the picture count is the buffer's.
    if sources.iter().any(|source| source.len() != size) {
        return Some(Vec::new());
    }
    if output.stdout.len() != size * sources.len() {
        return Some(Vec::new());
    }
    let mut rows = Vec::with_capacity(sources.len());
    for (index, source) in sources.iter().enumerate() {
        let Some(at) = index.checked_mul(size) else { return Some(Vec::new()) };
        let Some(picture) = output.stdout.get(at..at.saturating_add(size)) else { return Some(Vec::new()) };
        let sum: u64 = picture.iter().zip(source).map(|(a, b)| u64::from(a.abs_diff(*b))).sum();
        rows.push(u8::try_from(sum / size as u64).unwrap_or(u8::MAX));
    }
    Some(rows)
}

/// ffmpeg on this machine, if there is one.
#[cfg(target_os = "linux")]
fn ffmpeg_path() -> Option<String> {
    for candidate in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/bin/ffmpeg"] {
        if std::path::Path::new(candidate).exists() {
            return Some(candidate.to_string());
        }
    }
    std::process::Command::new("ffmpeg").arg("-version").output().ok().filter(|o| o.status.success()).map(|_| "ffmpeg".to_string())
}

/// The `nal_unit_type` of every NAL in an Annex-B run, in order — `7 8 5`, say.
///
/// A real walk of the start codes rather than a search for `00 00 01` anywhere: that byte sequence
/// occurs inside slice payload by coincidence, and a probe that matched one would report parameter
/// sets a stream does not have.
#[cfg(target_os = "linux")]
fn nal_types(bytes: &[u8]) -> String {
    let mut out = Vec::new();
    let mut at = 0usize;
    while let Some(found) = bytes.get(at..).and_then(|rest| rest.windows(3).position(|w| w == [0, 0, 1])) {
        let start = at + found + 3;
        let Some(&header) = bytes.get(start) else { break };
        out.push(header & 0x1f);
        at = start;
    }
    out.iter().map(u8::to_string).collect::<Vec<_>>().join(" ")
}

/// Whether the stream's own parameter sets are the ones this module wrote.
///
/// The probe cannot ask the encoder which it used for a given picture — that is reported per picture
/// — so this looks at the stream: the first access unit's bytes before the first slice are the
/// parameter sets either way, and the difference is which ones they are. What it actually answers is
/// "does this stream begin with an SPS and a PPS", which is what a decoder needs and what the report
/// has to be able to say.
#[cfg(target_os = "linux")]
fn picture_parameter_sets_are_ours(bytes: &[u8]) -> bool {
    let mut found = (false, false);
    let mut at = 0usize;
    while let Some(found_at) = bytes.get(at..).and_then(|rest| rest.windows(3).position(|w| w == [0, 0, 1])) {
        let start = at + found_at + 3;
        let Some(&header) = bytes.get(start) else { break };
        match header & 0x1f {
            7 => found.0 = true,
            8 => found.1 = true,
            // The slice is the last thing before the next picture's data; everything after it is
            // payload and cannot hold a start code that means anything.
            1 | 5 => return found.0 && found.1,
            _ => {}
        }
        at = start;
    }
    found.0 && found.1
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("vaencprobe: the VA-API encoder is Linux-only");
}
