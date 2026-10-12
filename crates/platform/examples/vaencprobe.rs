//! Pictures through the VA-API H.264 slice encoder, with a decode check that can tell a predicted
//! frame from an intra one.
//!
//! ```text
//! vaencprobe [-qp N] [-w W] [-h H] [-f FRAMES] [-refs N] [-fps N] [-gop N] [-o FILE] [-allidr]
//!            [-br KBPS] [-maxbr KBPS] [-vbr] [-tex AMPLITUDE]
//! ```
//!
//! Default: **one I frame followed by five P frames** of a 640x360 luma ramp that **moves 8 columns
//! left per picture**, at QP 26 and 30 fps, written to `/tmp/vaencprobe.h264`.
//!
//! # The two fidelity modes, and what each one shows
//!
//! `-br KBPS` builds a **CBR** session at that many kilobits per second instead of a constant-quantiser
//! one; `-vbr` with `-br` builds a **VBR** session with `-maxbr` (or 1.5x the target) as its ceiling.
//! With neither, the session is constant quantiser at `-qp`.
//!
//! **The table's QP column is the evidence for the difference.** Under constant quantiser it reads
//! `-qp` on every picture — the driver's own statement of what it coded, agreeing with the request. Under
//! a bitrate it moves, because the rate controller is choosing per picture, and the sizes move with it;
//! that is the mode an export actually runs in, since FilmCraft's settings state a bitrate and not a
//! quantiser.
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
//! ffmpeg -v error -f h264 -i /tmp/vaencprobe.h264 -f rawvideo -pix_fmt yuv420p - > frames.yuv
//! ```
//!
//! which gives `frames * (width * height + 2 * ceil(width/2) * ceil(height/2))` bytes in display
//! order — 4:2:0, so **the luma plane is the first `width * height` bytes of each frame**, and that
//! is the plane the deviation numbers above are computed from.
//!
//! **Do not ask for `-pix_fmt gray` to check this file.** `gray` is a *full-range* pixel format, so
//! `swscale` expands this backend's limited-range luma while converting to it: an uploaded `16` comes
//! back as `0`, an uploaded `235` as `255`, and an uploaded `117` as `118`. Measured on this machine,
//! on the probe's own default run, over the ramp's 16..=235 range that expansion is a mean absolute
//! offset of **9 levels** — which is the whole of the "mean absolute deviation per picture: 9" the
//! probe used to print, and every bit of it was a colour-range conversion rather than encoder loss.
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
    let mut kbps = 0u32;
    let mut max_kbps = 0u32;
    let mut vbr = false;
    let mut tex = 0u8;
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
            "-br" => kbps = it.next().and_then(|v| v.parse().ok()).unwrap_or(kbps),
            "-maxbr" => max_kbps = it.next().and_then(|v| v.parse().ok()).unwrap_or(max_kbps),
            "-vbr" => vbr = true,
            "-tex" => tex = it.next().and_then(|v| v.parse().ok()).unwrap_or(tex),
            other => {
                eprintln!("vaencprobe: unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    if let Err(why) = run(width, height, qp, frames, refs, fps, gop, all_idr, &out, kbps, max_kbps, vbr, tex) {
        eprintln!("vaencprobe: {why}");
        std::process::exit(1);
    }
}

/// The picture a run encodes: the moving ramp, or the ramp with detail added when `-tex` asks for it.
#[cfg(target_os = "linux")]
fn source(display: &(u32, u32), shift: u32, tex: u8) -> Result<(Vec<u8>, Vec<u8>), String> {
    match tex {
        0 => filmcraft_platform::vaapi::ramp_shifted(display.0, display.1, shift),
        amplitude => textured_ramp(display.0, display.1, shift, amplitude),
    }
}

/// **A ramp with a deterministic high-frequency pattern added, and the reason this function exists.**
///
/// A pure ramp has a **near-zero residual**: intra prediction reconstructs it almost exactly, so at any
/// quantiser above about 40 every coefficient rounds to zero and the encoded picture is *just the
/// prediction*. Measured on this driver: a 200 kbit/s CBR frame and a constant-quantiser QP-45 frame
/// of the same ramp have **byte-identical decodes** and `slice_qp_delta = 0` in both, because neither
/// has a coefficient to quantise.
///
/// **That makes the ramp blind to exactly the question a rate-control run has to answer** — whether the
/// quantiser the driver used is the one the bitstream says. A picture with detail has a residual at
/// every quantiser, so the decode it produces depends on the QP, and a stream whose headers disagree
/// with what was encoded shows up as a picture that is wrong rather than one that is indistinguishable.
///
/// The pattern is a **fixed integer hash of the coordinates**, not a generator: the same flag on the
/// same machine produces the same picture, so two runs are comparable, and no LCG state has to be
/// carried between pictures for the sequence to stay a function of the picture index alone. The shift
/// still applies, so the reference chain is as testable here as on the plain ramp.
///
/// It wraps into 16..=235 so a sample can never leave H.264's legal luma range and have its deviation
/// blamed on the encoder clipping input rather than on the quantiser.
#[cfg(target_os = "linux")]
fn textured_ramp(width: u32, height: u32, shift: u32, amplitude: u8) -> Result<(Vec<u8>, Vec<u8>), String> {
    let (mut luma, chroma) = filmcraft_platform::vaapi::ramp_shifted(width, height, shift)?;
    let columns = width.max(1) as usize;
    for (at, sample) in luma.iter_mut().enumerate() {
        let x = (at % columns) as u32;
        let y = (at / columns) as u32;
        // A 3-term integer hash of the coordinates: cheap, deterministic, and spread enough that a
        // 4x4 intra block's four coefficients differ, which is what makes the residual non-zero.
        let hash = x.wrapping_mul(2_654_435_761).wrapping_add(y.wrapping_mul(40_503)).wrapping_add(shift.wrapping_mul(12_345));
        // Fold the hash's high byte into -amplitude..=amplitude around the ramp's own value.
        let stepped = ((i32::from(((hash >> 16) & 0xff) as u8)) - 128) * i32::from(amplitude) / 128;
        let wanted = i32::from(*sample) + stepped;
        *sample = u8::try_from(wanted).unwrap_or(*sample).clamp(16, 235);
    }
    Ok((luma, chroma))
}

/// Which fidelity statement this run makes, resolved from the flags.
///
/// **The `-qp` value is deliberately dropped in a bitrate mode rather than ignored silently.** Under
/// CBR/VBR the quantiser is the rate controller's, and submitting `-qp` would be a second statement
/// about a picture's fidelity that the driver does not honour — which is exactly what
/// `VaEncoder::encode` refuses, so the probe refuses it here and says why rather than passing a
/// request the encoder would reject several lines later.
#[cfg(target_os = "linux")]
#[derive(Clone, Copy)]
enum Mode {
    /// Constant quantiser at `qp`.
    ConstantQuantiser(u8),
    /// Constant bit rate, in bits per second.
    Cbr(u32),
    /// Variable bit rate between a target and a ceiling, in bits per second.
    Vbr {
        /// The rate the controller is asked to aim at.
        target: u32,
        /// The highest rate it may reach.
        max: u32,
    },
}

#[cfg(target_os = "linux")]
impl Mode {
    fn name(&self) -> String {
        match self {
            Self::ConstantQuantiser(qp) => format!("constant quantiser at QP {qp}"),
            Self::Cbr(bps) => format!("CBR at {} kbit/s", bps / 1000),
            Self::Vbr { target, max } => format!("VBR at {} kbit/s, ceiling {} kbit/s", target / 1000, max / 1000),
        }
    }
}

/// The mode the flags describe, or why they cannot describe one.
#[cfg(target_os = "linux")]
fn resolve_mode(qp: u8, kbps: u32, max_kbps: u32, vbr: bool) -> Result<Mode, String> {
    if kbps == 0 {
        if vbr || max_kbps != 0 {
            return Err("-vbr and -maxbr need a -br to aim at".into());
        }
        return Ok(Mode::ConstantQuantiser(qp));
    }
    let Some(bps) = kbps.checked_mul(1000) else {
        return Err(format!("{kbps} kbit/s does not fit the record's 32-bit bits-per-second field"));
    };
    if !vbr {
        if max_kbps != 0 {
            return Err("-maxbr only means something with -vbr; CBR's target is its ceiling".into());
        }
        return Ok(Mode::Cbr(bps));
    }
    // The same 1.5x ceiling `ExportSettings::resolve` and this crate's export gate use, so a probe
    // run and an export of the same settings ask the driver for the same thing.
    let max_kbps = if max_kbps >= kbps { max_kbps } else { kbps.saturating_mul(3) / 2 };
    let max = max_kbps.checked_mul(1000).ok_or_else(|| format!("a ceiling of {max_kbps} kbit/s does not fit the record's 32-bit bits-per-second field"))?;
    Ok(Mode::Vbr { target: bps, max })
}

/// The encoder this run drives, in the mode the flags describe.
///
/// **The mode is a constructor argument rather than a call after construction**, and the reason is
/// libva's: `VAConfigAttribRateControl` is read by `vaCreateConfig` and nothing changes it
/// afterwards. The two bitrate modes go through `VaEncoder`'s named constructors rather than a
/// `RateControl` value, so this file does not have to name a type it only passes along.
#[cfg(target_os = "linux")]
fn build(
    api: &std::sync::Arc<filmcraft_platform::vaapi::Display>,
    size: (u32, u32),
    refs: u32,
    mode: Mode,
) -> Result<filmcraft_platform::vaapi::VaEncoder, String> {
    use filmcraft_platform::vaapi::VaEncoder;
    match mode {
        Mode::ConstantQuantiser(_) => VaEncoder::at_constant_quantiser(api, size, refs),
        Mode::Cbr(bps) => VaEncoder::at_constant_bitrate(api, size, refs, bps),
        Mode::Vbr { target, max } => VaEncoder::at_variable_bitrate(api, size, refs, target, max),
    }
}

/// Encode a run of pictures and report what is needed to tell prediction from intra coding.
#[cfg(target_os = "linux")]
fn run(
    width: u32,
    height: u32,
    qp: u8,
    frames: usize,
    refs: u32,
    fps: u32,
    gop: u32,
    all_idr: bool,
    out: &str,
    kbps: u32,
    max_kbps: u32,
    vbr: bool,
    tex: u8,
) -> Result<(), String> {
    use filmcraft_platform::vaapi::{EncodeRequest, VaEncoder};
    let api = filmcraft_platform::vaapi::open_display().map_err(|why| format!("no VA-API display: {why}"))?;
    println!("driver: {} (libva {}.{})", api.vendor().unwrap_or("(none)"), api.va_version().0, api.va_version().1);

    // What the driver says it can hold references for, asked before any config exists.
    let (l0, l1) = filmcraft_platform::vaapi::max_reference_frames(&api);
    println!("VAConfigAttribEncMaxRefFrames: reference list 0 = {l0}, list 1 = {l1} (asked with {refs})");
    // **What the driver reports for rate control.** It was printed because the settings gate's
    // bitrate *decline* said "the driver reports <names>"; it is printed now for the opposite
    // reason — the gate *submits* a mode, and this is the answer to "could it have submitted
    // anything else", which a reader of the probe is entitled to.
    println!("VAConfigAttribRateControl: {}", filmcraft_platform::hardware_encode::vaapi_rate_control_modes_named());

    let mode = resolve_mode(qp, kbps, max_kbps, vbr)?;
    let mut encoder = build(&api, (width, height), refs, mode)?;
    encoder.at_rate(fps)?;
    encoder.key_frames_every(gop)?;
    let (coded, display) = encoder.sizes();
    let (l, r, t, b) = encoder.crop();
    println!("picture: {display:?} displayed, {coded:?} coded, crop left {l} right {r} top {t} bottom {b}");
    // **Which fidelity statement this run makes, printed before anything is encoded** — and in a
    // bitrate mode `-qp` is named as *not applied*, because a quantiser sitting in the command line
    // next to a bitrate and having no effect is the kind of thing a probe report should say out loud.
    match mode {
        Mode::ConstantQuantiser(_) => println!("fidelity: {}", mode.name()),
        other => println!("fidelity: {} (the -qp value {qp} does not apply in this mode)", other.name()),
    }
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
        let (luma, chroma) = source(&display, shift * index as u32, tex)?;
        encoder.upload(&luma, &chroma)?;
        sources.push(luma);
        // The picture *kind* is the loop's decision, not this call's: a caller asks for a P picture
        // and the loop makes it an IDR when the reference state or the `frame_num` wrap demands it.
        // The reason comes back on the picture, so it is printed rather than assumed away.
        let last = index + 1 == frames;
        // **The request names a quantiser only in constant-quantiser mode.** In a bitrate mode the
        // rate controller chooses and the request says so with `None`, which is the pairing
        // `VaEncoder::encode` checks; passing `-qp` through regardless would be refused there with a
        // message about modes rather than about pictures, which is a worse place to learn it.
        let mut request = match mode {
            Mode::ConstantQuantiser(qp) => EncodeRequest::predicted(qp)?,
            _ => EncodeRequest::predicted_at_rate()?,
        };
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
    // **The achieved bit rate, in a bitrate mode — the number that says the rate control is running.**
    //
    // It is the coded bytes over the pictures' duration at the rate the encoder was told, which is the
    // definition of a bitrate and nothing else. **A short run under-reports** (the controller has not
    // converged and the first key frame dominates), so the line says how many pictures it is over
    // rather than presenting the number bare; the export-level oracle in `tests/vaenc.rs` is where a
    // size claim is actually asserted.
    if fps > 0 && frames > 1 {
        let total: usize = sizes.iter().sum();
        let seconds = frames as f64 / f64::from(fps);
        let achieved = total as f64 * 8.0 / seconds / 1000.0;
        match mode {
            Mode::ConstantQuantiser(_) => {
                println!("bit rate: {achieved:.0} kbit/s over {frames} pictures — no target; this mode states a quantiser, not a rate")
            }
            Mode::Cbr(bps) => println!(
                "bit rate: {achieved:.0} kbit/s against a {target:.0} kbit/s CBR target ({ratio:.0}% of it)",
                target = bps as f64 / 1000.0,
                ratio = achieved / (bps as f64 / 1000.0) * 100.0
            ),
            Mode::Vbr { target, max } => println!(
                "bit rate: {achieved:.0} kbit/s against a {target_k:.0} kbit/s VBR target and a {max_k:.0} kbit/s ceiling",
                target_k = target as f64 / 1000.0,
                max_k = max as f64 / 1000.0
            ),
        }
    }
    // **The decode check, once, over the whole stream.** ffmpeg is given exactly the bytes the
    // muxer would be given — an elementary stream — and its output is compared with the ramps that
    // went in, one picture at a time.
    match decode_deviations(&annex_b, &sources, display) {
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
        // it, the comparison is same content, same everything but the picture type.
        //
        // **The control is always constant quantiser**, whatever mode the run itself used, because the
        // comparison it exists for is "predicted against intra at the same quality" and a bitrate mode
        // has no single quality to hold still. A run in a bitrate mode therefore compares a predicted
        // stream at a moving quantiser against an intra stream at `-qp`, which is a weaker statement —
        // and it says so, rather than printing a ratio that reads as if it were the same comparison.
        println!();
        println!(
            "control: the same pictures, every one coded as an IDR at QP {qp}{}",
            match mode {
                Mode::ConstantQuantiser(_) => "",
                _ => " (the run above is at a bit rate, so this compares across qualities — the decode check above is the stronger evidence)",
            }
        );
        let mut control = VaEncoder::at_constant_quantiser(&api, (width, height), 1).map_err(|e| format!("control encoder: {e}"))?;
        control.at_rate(fps)?;
        control.key_frames_every(1)?;
        let mut control_sizes = Vec::new();
        for index in 0..frames {
            let (luma, chroma) = source(&display, shift * index as u32, tex)?;
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
    // **The instruction names `yuv420p`, and says why `gray` is wrong, because a reader who follows
    // it with `gray` gets a number nine levels high and believes the encoder lost that much.** The
    // expected byte count is therefore the *4:2:0* frame, not the luma plane.
    let (pw, ph) = (display.0 as usize, display.1 as usize);
    let luma = pw.saturating_mul(ph);
    let frame_bytes = luma.saturating_add(pw.div_ceil(2).saturating_mul(ph.div_ceil(2)).saturating_mul(2));
    println!("decode it with: ffmpeg -v error -f h264 -i {out} -f rawvideo -pix_fmt yuv420p - | wc -c");
    println!(
        "  (expect {} = {} pictures of {}x{} as 4:2:0; the luma plane is the first {} bytes of each)",
        frame_bytes.saturating_mul(frames),
        frames,
        display.0,
        display.1,
        luma
    );
    println!("  (do not use -pix_fmt gray to check this file: it is a full-range format and expands the limited-range luma, so it reports ~9 too high)");
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
///
/// **The reference is the plane that was uploaded, which here is the source luma itself.** The
/// probe builds its pictures as planar 4:2:0 luma and hands that same `Vec` to
/// [`VaEncoder::upload`], so `sources` *is* the encoded plane and no BT.709 conversion is in the
/// path — unlike the export gate, which starts from RGBA and must run `rgba_to_yuv420_8` to learn
/// what the encoder was given. Comparing against anything else here (a generated ramp, a scaled
/// picture) would subtract a colour conversion from the encoder's loss, which is the defect
/// `-pix_fmt gray` already was on the decode side.
#[cfg(target_os = "linux")]
fn decode_deviations(stream: &[u8], sources: &[Vec<u8>], display: (u32, u32)) -> Option<Vec<u8>> {
    let ffmpeg = ffmpeg_path()?;
    let mut child = std::process::Command::new(&ffmpeg)
        // **`-pix_fmt yuv420p`, and the luma plane is sliced out of it — not `-pix_fmt gray`.**
        //
        // `gray` is a **full-range** pixel format, so `swscale` expands the limited-range luma this
        // backend codes while converting to it: an uploaded `16` comes back as `0`, an uploaded `235`
        // as `255`, and an uploaded `117` as `118`. Measured on this machine, on a flat picture
        // uploaded at 117 and reconstructed exactly: `-pix_fmt gray` reads 118 and `-pix_fmt yuv420p`
        // reads 117. Over this ramp's own 16..=235 range the expansion is a mean absolute offset of
        // **9 levels**, and a probe that subtracts a decoded full-range plane from an uploaded
        // limited-range plane measures **a colour-range conversion, not an encoder** — no tolerance
        // makes that number honest. `yuv420p` is passed through untouched, and the luma plane is the
        // plane that was uploaded, so like is compared with like.
        .args(["-v", "error", "-f", "h264", "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .ok()?;
    // **The output is drained on its own thread while the input is being written, and that is not an
    // optimisation — it is the difference between working and hanging forever.**
    //
    // Both ends are pipes, and each has about 64 KiB of buffer. Writing the stream first and reading
    // the decode afterwards deadlocks the moment the decoded luma passes the buffer: ffmpeg blocks
    // writing pictures nobody is reading, stops reading the elementary stream, and this process blocks
    // writing a stream nobody is reading. **Measured on this machine at 320x180:** 60 pictures (3.4 MB
    // of gray) complete, 72 pictures (4.1 MB) hang until killed — `vaencprobe` parked in
    // `anon_pipe_write` with ffmpeg parked writing its own stdout. The hang is a function of decoded
    // size, not of the encoder, which had finished and printed its sizes and bit rate long before.
    //
    // So the reader starts first and takes the stdout and stderr handles; `wait_with_output` below no
    // longer has pipes to drain, which is exactly what makes this correct rather than merely lucky.
    use std::io::Read as _;
    let mut stdout = child.stdout.take()?;
    let mut stderr = child.stderr.take()?;
    let reader = std::thread::spawn(move || {
        let mut out = Vec::new();
        let mut err = Vec::new();
        // Both are read to completion: a decode that writes more than the pipe holds only makes
        // progress while something is draining it, and a half-read pipe is what turns a hang into a
        // short answer. `read_to_end` returns the bytes it got, and a read error ends the drain with
        // what arrived — ffmpeg dying mid-stream is reported by the length check below, not by a panic.
        let _ = stdout.read_to_end(&mut out);
        let _ = stderr.read_to_end(&mut err);
        (out, err)
    });

    // The child's stdin is a pipe this function owns; writing to it and dropping the handle closes it,
    // which is what tells ffmpeg the elementary stream has ended. A closed pipe mid-write means ffmpeg
    // stopped reading, and the join below then reports whatever it managed to decode.
    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        let _ = stdin.write_all(stream);
    }
    let status = child.wait().ok()?;
    let (stdout, stderr) = reader.join().ok()?;
    let output = std::process::Output { status, stdout, stderr };
    // **ffmpeg's own diagnostics are not ignored.** A stream it decodes *with* a warning has been
    // accepted and is not necessarily right, and a probe that printed a deviation over it would
    // report a number computed from a stream nobody would trust.
    if !output.stderr.is_empty() {
        eprintln!("  ffmpeg said: {}", String::from_utf8_lossy(&output.stderr).trim());
    }
    // **The luma plane is the first `width * height` bytes of each 4:2:0 frame, and the stride from
    // one picture to the next is the whole frame** — luma plus both chroma planes. Comparing at the
    // luma stride over a `yuv420p` buffer would pair every picture with a chroma plane, which is a
    // comparison that cannot fail to be wrong and is not obviously so.
    //
    // `ceil`, not `/ 2`: the encoder refuses an odd size upstream, but the arithmetic is written the
    // way the picture actually is so it stays correct for any size that reaches it, and every step is
    // `checked_*` because the geometry arrives from the command line.
    let (w, h) = (display.0 as usize, display.1 as usize);
    let size = w.checked_mul(h).filter(|size| *size > 0)?;
    let chroma = w.div_ceil(2).checked_mul(h.div_ceil(2))?;
    let frame = size.checked_add(chroma.saturating_mul(2))?;
    // Every source picture is the same size — they are all the same geometry — so `size` is one
    // picture's luma length and the picture count is the buffer's.
    if sources.iter().any(|source| source.len() != size) {
        return Some(Vec::new());
    }
    if output.stdout.len() != frame.checked_mul(sources.len())? {
        return Some(Vec::new());
    }
    let mut rows = Vec::with_capacity(sources.len());
    for (index, source) in sources.iter().enumerate() {
        let Some(at) = index.checked_mul(frame) else { return Some(Vec::new()) };
        let Some(picture) = output.stdout.get(at..at.saturating_add(size)) else { return Some(Vec::new()) };
        // Every pixel of the plane, paired with the same raster position in the plane that was
        // uploaded, and divided by the pixels compared — so the number this returns *is* the mean
        // absolute deviation, and it reaches 0 on a stream that lost nothing.
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
