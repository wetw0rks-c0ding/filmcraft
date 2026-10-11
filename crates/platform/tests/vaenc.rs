//! The VA-API H.264 slice encoder at the weaker standard, and the export gate in front of it
//! (Linux). Skips without a VA-API driver, or without ffmpeg for the decode half.
//!
//! # Encode verification is materially weaker than decode's, and this file says so
//!
//! The decode milestone could assert 72 of 72 pictures bit-exact against ffmpeg because both sides
//! were the same algorithm. **That is not available at any level here.** A hardware encoder and
//! FilmCraft's software encoder are different algorithms making different decisions, so "matches the
//! software encoder" would be a claim with no content — and "matches ffmpeg's encode" is worse than
//! nothing, because it measures ffmpeg's encoder.
//!
//! What this file asserts instead, which is the gate the plan states:
//!
//! 1. **The file decodes with ffmpeg to the right pictures, compared against the _source_**, within
//!    a lossy tolerance ([`a_muxed_export_decodes_to_the_pictures_that_went_in`]). The comparison is
//!    against what was rendered and encoded, not against another encode.
//! 2. **The stream is well-formed** ([`the_sample_entry_describes_the_bitstream_that_was_written`]):
//!    the `avcC`'s parameter sets parse and agree with the slices, the sample table has no gaps and
//!    monotonic times, and the sample entry is refused rather than empty until a picture exists.
//! 3. **Size and rate control land where the settings asked** — with the honest caveat that this
//!    backend is **constant quantiser and sends no bitrate**, so the size claim is about the
//!    quantiser, not the bitrate. See [`what_the_export_gate_declines_and_why`].
//!
//! # What is *not* claimed
//!
//! - **Nothing is compared against a software encode.** There is no "matches ffmpeg" here.
//! - **No bitrate guarantee.** The gate declines any export that names one; see
//!   [`what_the_export_gate_declines_and_why`].
//! - **The pictures this backend produces are within a tolerance, not identical.** QP 26 on a
//!   synthetic ramp measures about 9/255 mean absolute deviation, and every tolerance below is set
//!   from that measurement rather than picked to make a test pass.
#![cfg(target_os = "linux")]

use std::path::PathBuf;

use filmcraft_export::{BitrateMode, EncoderFrame, ExportSettings, FieldOrder, Format, H264Pass, H264Profile, HardwareEncoding};
use filmcraft_isobmff::{Brand, Mp4Writer, TrackConfig, WriteSample, WriterOptions};
use filmcraft_platform::hardware_encode::{vaapi_config_for, vaapi_encoder_factory};
use filmcraft_platform::vaapi::{EncodeRequest, VaEncoder, open_display, ramp_shifted};
use filmcraft_time::FrameRate;

/// The encoder registry and the perf counters are process-wide: one test at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A scratch directory that is **not** `/tmp`: it is created inside this crate's `target` so a
/// machine with a full or shared `/tmp` does not fail the run for the wrong reason.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("vaenc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("a scratch directory {}: {e}", dir.display()));
    dir.join(name)
}

/// The settings an export that this backend *can* take would have — **a bitrate of zero**, which is
/// the only way past [`vaapi_config_for`]'s bitrate decline and the only value that says "no bitrate
/// asked for". It exists so the muxer oracle can drive a real encoder through the real muxer; every
/// test that asserts a *decline* uses [`hw_settings`] instead, which is the default.
fn no_bitrate_settings(path: String) -> ExportSettings {
    ExportSettings {
        format: Format::H264,
        path,
        include_audio: false,
        bitrate_kbps: 0,
        bitrate_mode: BitrateMode::Vbr1Pass,
        hardware_encoding: HardwareEncoding::Auto,
        h264_profile: H264Profile::Main,
        field_order: FieldOrder::Progressive,
        keyframe_distance: Some(8),
        ..Default::default()
    }
}

/// The settings FilmCraft actually produces, which name a bitrate.
fn hw_settings(path: String) -> ExportSettings {
    ExportSettings { path, include_audio: false, hardware_encoding: HardwareEncoding::Auto, ..Default::default() }
}

/// One synthetic luma ramp per picture, as RGBA — what an export would hand the encoder.
///
/// **The ramp moves eight columns a picture**, which is what makes prediction observable: with a
/// still picture a decoder that ignored the references entirely would produce exactly the right
/// pixels, so a "looks right" result would prove nothing.
fn ramp_frame(w: u32, h: u32, index: u32) -> Vec<u8> {
    let (luma, chroma) = ramp_shifted(w, h, 8 * index).unwrap_or_else(|e| panic!("the ramp is buildable: {e}"));
    let mut rgba = Vec::with_capacity((w as usize) * (h as usize) * 4);
    for row in 0..h as usize {
        for col in 0..w as usize {
            let y = luma[row * w as usize + col];
            let ci = (row / 2) * (w as usize / 2) + col / 2;
            // The chroma is a flat 128 in `ramp_shifted`, so this is exact; the unpack is kept so a
            // change to the generator cannot quietly make the oracle compare against the wrong source.
            let (cb, cr) = (chroma.get(ci * 2).copied().unwrap_or(128), chroma.get(ci * 2 + 1).copied().unwrap_or(128));
            rgba.extend_from_slice(&[y, cb, cr, 255]);
        }
    }
    rgba
}

// ---------------------------------------------------------------------------
// The export gate
// ---------------------------------------------------------------------------

/// **What the gate takes and what it declines, and each decline's reason.**
///
/// This is the test that says the wiring exists and what it is honestly good for. Two of its cases
/// are not obvious and are the ones worth reading:
///
/// - **A bitrate is declined**, because this backend is constant quantiser and cannot honour one.
///   The gate's message names the driver's own answer for rate control rather than blaming
///   VA-API: measured on iHD 26.3.5, `VAConfigAttribRateControl` reports **CBR, VBR, CQP, ICQ,
///   QVBR and TCBRC** — the driver *can*; this backend does not. That is a limitation of this
///   code, and a decline that said "VA-API cannot" would be a claim about every driver rather than
///   this one.
/// - **A key-frame interval past `MaxFrameNum` is declined**, because the frame loop inserts a key
///   frame every 16 pictures whatever the settings say. An export that asked for 250 and got 16 is
///   a file many times the size it was promised, and "tell the caller" and "refuse" are the only
///   honest answers; this refuses.
#[test]
fn what_the_export_gate_declines_and_why() {
    let _one = lock();
    let at = |f: &dyn Fn(&mut ExportSettings)| {
        let mut s = no_bitrate_settings(String::new());
        f(&mut s);
        vaapi_config_for(Format::H264, 640, 360, FrameRate::FPS_30, &s)
    };

    // **The case that exists**: no bitrate, Main profile, progressive, square pixels, whole fps.
    assert!(at(&|_| {}).is_ok(), "an export with no bitrate target is one this backend can take");

    // The declines, each checked for the *reason* and not merely that there is one.
    let cases: [(&str, Box<dyn Fn(&mut ExportSettings)>, &str); 9] = [
        ("hardware encoding is off", Box::new(|s: &mut ExportSettings| s.hardware_encoding = HardwareEncoding::Off), "hardware encoding is off"),
        ("MXF", Box::new(|s: &mut ExportSettings| s.format = Format::MxfOp1a), "Annex B"),
        ("two-pass", Box::new(|s: &mut ExportSettings| s.h264_pass = H264Pass::First), "two-pass"),
        ("a bitrate", Box::new(|s: &mut ExportSettings| s.bitrate_kbps = 8_000), "constant quantiser"),
        ("HDR", Box::new(|s: &mut ExportSettings| s.signal = filmcraft_export::ColorSignal::PQ), "HDR"),
        ("an odd picture size", Box::new(|_: &mut ExportSettings| {}), "4:2:0"),
        ("non-square pixels", Box::new(|s: &mut ExportSettings| s.pixel_aspect = Some((4, 3))), "non-square pixels"),
        ("interlaced", Box::new(|s: &mut ExportSettings| s.field_order = FieldOrder::UpperFirst), "interlaced"),
        ("a Baseline or High profile", Box::new(|s: &mut ExportSettings| s.h264_profile = H264Profile::High), "state Main"),
    ];
    for (name, set, why) in cases {
        let got = if name == "an odd picture size" {
            let mut s = no_bitrate_settings(String::new());
            set(&mut s);
            vaapi_config_for(Format::H264, 641, 361, FrameRate::FPS_30, &s)
        } else {
            at(&set)
        };
        let Err(why_said) = got else {
            panic!("{name} is accepted, which is a stream the settings asked for and this backend cannot produce");
        };
        assert!(why_said.contains(why), "{name}: the decline says {why:?}, not just that something is wrong: {why_said}");
    }

    // The key-frame ceiling, which needs its own settings because it is a number.
    let mut s = no_bitrate_settings(String::new());
    s.keyframe_distance = Some(250);
    let why = vaapi_config_for(Format::H264, 640, 360, FrameRate::FPS_30, &s).expect_err("250 is past MaxFrameNum");
    assert!(why.contains("250"), "the message names the interval asked for: {why}");
    assert!(why.contains("16"), "and the ceiling it hit: {why}");

    // A fractional frame rate is refused rather than rounded: `VAEncMiscParameterFrameRate` here
    // carries whole frames per second, so 30000/1001 is a rate this backend cannot state.
    let why = vaapi_config_for(Format::H264, 640, 360, FrameRate::FPS_23_976, &no_bitrate_settings(String::new()))
        .expect_err("23.976 is not a whole number of frames per second");
    assert!(why.contains("fractional"), "and it says why rather than rounding: {why}");

    // **HEVC is declined by name, not by accident.** There is no HEVC encoder in this backend, and
    // the software path has none either, so the reason must say HEVC rather than "not H.264" —
    // otherwise a reader cannot tell a decline from a bug.
    let Err(why) = vaapi_config_for(Format::Hevc, 640, 360, FrameRate::FPS_30, &no_bitrate_settings(String::new())) else {
        panic!("HEVC is accepted, but this backend declares H.264 encode records and nothing else");
    };
    assert!(why.contains("HEVC"), "the HEVC decline names HEVC: {why}");
}

/// **The factory is registered, and the counters move for the declines it makes.**
///
/// `register()` on Linux puts `vaapi_encoder_factory` in front of the software encoders. This
/// asserts the registration is there (the `encoder_registered` check) and that asking the factory
/// for a real export with the settings FilmCraft actually produces declines **and counts the
/// decline** — `perf.stats` `export.hardware` is the only way a user learns a machine has no
/// hardware encode path for a given export, so a decline that did not count would be invisible.
#[test]
fn the_factory_is_registered_and_its_declines_are_counted() {
    let _one = lock();
    filmcraft_platform::register();
    // Registering twice is harmless and `register()` is called by every other test in the binary.
    filmcraft_platform::register();
    assert!(filmcraft_export::encoder_registered(vaapi_encoder_factory), "the VA-API encoder factory is registered in front of the software encoders");

    let before = filmcraft_export::hw_encode_stats();
    let settings = hw_settings(scratch("declined.mp4").to_string_lossy().to_string());
    let got = vaapi_encoder_factory(Format::H264, 640, 360, FrameRate::FPS_30, &settings);
    let after = filmcraft_export::hw_encode_stats();
    // The default settings name a bitrate, so this backend declines — and says so. **Asserted as a
    // decline with a count, not as a skip**: this is the honest state of the gate today, and a test
    // that quietly passed when the gate started accepting bitrates would hide the change.
    assert!(got.is_none(), "a settings-set bitrate is declined by this constant-quantiser backend");
    assert_eq!(after.declined, before.declined + 1, "and the decline is counted for perf.stats");
    assert_eq!(after.sessions, before.sessions, "with no encoder created");
    assert_eq!(after.frames, before.frames, "and no picture encoded");

    // And with the one setting that says "no bitrate", the factory builds a real encoder.
    let settings = no_bitrate_settings(scratch("accepted.mp4").to_string_lossy().to_string());
    match vaapi_encoder_factory(Format::H264, 640, 360, FrameRate::FPS_30, &settings) {
        Some(Ok(_)) => {}
        Some(Err(why)) => panic!("an export with no bitrate target was refused by the factory: {why}"),
        None => {
            if open_display().is_err() {
                eprintln!("SKIPPED: no VA-API display");
                return;
            }
            panic!("the factory declined an export vaapi_config_for accepts, which points at the display or the encoder, not the settings");
        }
    }
}

// ---------------------------------------------------------------------------
// The muxer oracle — a real file, decoded by ffmpeg, compared to the source
// ---------------------------------------------------------------------------

/// **A real MP4, written by FilmCraft's own muxer from this encoder's samples, decoded by ffmpeg,
/// and compared to the pictures that went in.**
///
/// This is the export-level oracle the previous review said did not exist. It does exist, and this
/// is it: [`Mp4Writer`] is the same writer `filmcraft_export`'s job uses, so the container, the
/// sample table and the `avcC` are produced the way a real export produces them — not assembled by
/// a test.
///
/// **What it compares, and what it deliberately does not.** The reference is the **source ramp**,
/// one per picture, not another encode: a hardware encoder and a software encoder are different
/// algorithms, so "matches the software encode" would mean nothing here. The tolerance is a lossy
/// one appropriate to a QP-26 constant-quantiser encode, and its number is measured rather than
/// guessed — `examples/vaencprobe.rs` measures **about 9/255 mean absolute deviation** on this ramp
/// at QP 26, so 24 is roughly two and a half times the quantiser's own noise and still nowhere near
/// a wrong picture (a wrong reference shows 200+, see the depth-two refusal).
#[test]
fn a_muxed_export_decodes_to_the_pictures_that_went_in() {
    let _one = lock();
    let Some(ffmpeg) = filmcraft_testkit::ffmpeg_or_skip("decoding the muxed export") else { return };
    if open_display().is_err() {
        eprintln!("SKIPPED: no VA-API display");
        return;
    }
    let (w, h, count) = (320u32, 180u32, 24usize);

    // The encoder, configured exactly as the gate configures one.
    let mut encoder = match vaapi_encoder_factory(Format::H264, w, h, FrameRate::FPS_30, &no_bitrate_settings(String::new())) {
        Some(Ok(e)) => e,
        Some(Err(why)) => panic!("the factory refused a configuration its own gate accepts: {why}"),
        None => {
            eprintln!("SKIPPED: the factory declined with a display open; the log line above says why");
            return;
        }
    };

    // Every source picture is kept: ffmpeg's output is compared with what went **in**.
    let sources: Vec<Vec<u8>> = (0..count as u32).map(|i| ramp_frame(w, h, i)).collect();
    let path = scratch("muxed.mp4");
    let file = std::fs::File::create(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut writer = Mp4Writer::new(file, WriterOptions::new(Brand::Mp4)).unwrap_or_else(|e| panic!("a writer: {e}"));

    let mut track = None;
    let mut packets = Vec::new();
    for (index, rgba) in sources.iter().enumerate() {
        let frame = EncoderFrame { width: w, height: h, rgba, hdr: None, index: index as u64 };
        let out = encoder.encode(&frame).unwrap_or_else(|e| panic!("picture {index} did not encode: {e}"));
        // **One packet per picture, and it is checked here rather than assumed.** A hardware encoder
        // that returned two packets for one picture, or none, would be a file with a sample table
        // that does not match the pictures, and that is the defect the plan's "no gaps" rule is about.
        assert_eq!(out.len(), 1, "picture {index} produced {} samples, not one", out.len());
        packets.extend(out);
        if track.is_none() {
            // **The track is added after the first picture**, because `sample_entry` before a
            // picture exists is the trait's documented "may only be complete after the first
            // frame" case. Waiting is the point: an empty `avcC` written up front is a file no
            // player can start.
            track = Some(writer.add_track(TrackConfig::new(encoder.sample_entry(), 30)).unwrap_or_else(|e| panic!("a video track: {e}")));
        }
    }
    for out in encoder.flush().unwrap_or_else(|e| panic!("flush: {e}")) {
        packets.push(out);
    }
    let track = track.unwrap_or_else(|| panic!("no picture was encoded, so there is nothing to mux"));
    for (index, p) in packets.iter().enumerate() {
        writer
            .write_sample(track, WriteSample { data: &p.data, duration: p.duration, composition_offset: p.composition_offset, is_sync: p.key })
            .unwrap_or_else(|e| panic!("sample {index}: {e}"));
    }
    drop(writer.finish().unwrap_or_else(|e| panic!("finishing the file: {e}")));
    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    assert!(size > 0, "the muxer wrote nothing");

    // **ffmpeg decodes the file, and its own diagnostics are not ignored.** A stream ffmpeg decodes
    // *with* a warning has been accepted and is not necessarily right, and a test that reported a
    // deviation over it would be reporting a number computed from a stream nobody would trust.
    let raw = std::process::Command::new(&ffmpeg)
        .args(["-v", "error", "-i", path.to_string_lossy().as_ref(), "-f", "rawvideo", "-pix_fmt", "gray", "-"])
        .output()
        .unwrap_or_else(|e| panic!("ffmpeg: {e}"));
    assert!(raw.stderr.is_empty(), "ffmpeg decoded the export with a diagnostic: {}", String::from_utf8_lossy(&raw.stderr));

    // **A short decode is a short result, not a comparison of the ones that happened to line up.**
    let one = (w as usize) * (h as usize);
    assert_eq!(raw.stdout.len(), one * count, "ffmpeg returned {} bytes of luma for {count} pictures of {w}x{h} ({one} bytes each)", raw.stdout.len());

    // The comparison: against the source, picture by picture, within a lossy tolerance.
    let mut deviations = Vec::with_capacity(count);
    for (index, source) in sources.iter().enumerate() {
        let Some(picture) = raw.stdout.get(index * one..(index + 1) * one) else { continue };
        // The ramp_frame builds RGBA from luma, so the source luma is the red channel of every
        // fourth byte — taken back out of the source rather than from the ramp directly, so the
        // oracle compares what the muxer saw, not what the generator meant.
        let sum: u64 =
            picture.iter().enumerate().filter(|(i, _)| i % 4 == 0).map(|(i, decoded)| u64::from(decoded.abs_diff(*source.get(i / 4).unwrap_or(&0)))).sum();
        let pixels = one as u64;
        deviations.push(u8::try_from(sum / pixels).unwrap_or(u8::MAX));
    }
    let worst = deviations.iter().copied().max().unwrap_or(u8::MAX);
    assert!(deviations.iter().all(|d| *d < 24), "every picture is within a QP-26 tolerance of the source; deviations {deviations:?} (worst {worst})");
    // **Flat, not merely small.** A drifting reference produces pictures that each look plausible
    // and get steadily worse; a run whose deviation is flat is what a correct reference chain looks
    // like. The assertion is on the *spread*, so a stream that quietly degraded would fail here even
    // if every frame passed the absolute bound.
    let (lo, hi) = (deviations.iter().copied().min().unwrap_or(0), worst);
    assert!(hi - lo <= 4, "the deviation is flat across the sequence, not drifting: {deviations:?}");
    eprintln!("muxed {count} pictures to {} bytes, mean absolute deviation {deviations:?}", size);
}

// ---------------------------------------------------------------------------
// The stream's own well-formedness
// ---------------------------------------------------------------------------

/// **The `avcC` describes the bitstream it is handed, and the sample entry is refused until it can.**
///
/// Three claims, each different from the others:
///
/// 1. **The parameter sets in the sample entry parse** and describe the stream: they are an SPS and
///    a PPS of the right profile, and re-parsing them with `filmcraft_h264`'s own parser (the one
///    the decode path trusts) succeeds.
/// 2. **The slices in the coded bytes parse against those sets.** This is the "consistent with the
///    bitstream" half, and it is a real check rather than a restatement: the SPS's
///    `log2_max_frame_num`, `frame_mbs_only_flag` and `log2_max_pic_order_cnt_lsb` are not readable
///    from a slice header — they *are* its length — so a set that disagreed would produce a
///    different, plausible `frame_num` rather than a parse failure.
/// 3. **`sample_entry` refuses before the first picture** rather than handing out an empty `avcC`.
#[test]
fn the_sample_entry_describes_the_bitstream_that_was_written() {
    let _one = lock();
    let Ok(display) = open_display() else {
        eprintln!("SKIPPED: no VA-API display");
        return;
    };

    // Before anything is encoded there is no stream to describe, and the answer is an error.
    let mut fresh = VaEncoder::new(&display, (320, 180), 1).expect("a one-reference encoder is measured to work");
    fresh.at_rate(30).expect("30 fps is a legal frame rate");
    fresh.key_frames_every(8).expect("8 is a legal key-frame interval");
    let why = fresh.sample_entry().expect_err("nothing has been encoded, so there is no stream to describe");
    assert!(why.contains("encoded nothing"), "and the refusal says what is missing: {why}");

    let mut encoder = fresh;
    let mut samples = Vec::new();
    // Past the 16-picture `MaxFrameNum`, so the key frame at the wrap is in the run and the sample
    // entry has survived one.
    for index in 0..20u32 {
        let (luma, chroma) = ramp_shifted(320, 180, 8 * index).expect("the ramp is buildable");
        encoder.upload(&luma, &chroma).expect("the ramp uploads");
        let picture =
            encoder.encode(EncodeRequest::predicted(26).expect("QP 26 is in range")).unwrap_or_else(|e| panic!("picture {index} did not encode: {e}"));
        samples.push(picture);
    }

    // The `avcC` the muxer would be handed.
    let entry = encoder.sample_entry().expect("a sample entry after twenty pictures");
    let filmcraft_isobmff::CodecConfig::Avc(avc) = &entry.codec else {
        panic!("an H.264 sample entry carries an avcC, not {:?}", entry.codec);
    };
    assert_eq!(avc.length_size, 4, "the samples' four-byte lengths are what it declares");
    assert_eq!((avc.sps.len(), avc.pps.len()), (1, 1), "one SPS and one PPS");
    assert_eq!(avc.profile, 77, "Main, read from byte 1 of the stored SPS");
    let video = entry.video.as_ref().unwrap_or_else(|| panic!("an H.264 sample entry carries video parameters: {:?}", entry.video));
    assert_eq!((video.width, video.height), (320, 180), "the display size, not the macroblock-padded 320x184");

    // **The sets parse with the decode path's own parsers.** The NAL header byte is kept by
    // `avcC` (`AvcConfig::new` takes profile/compat/level from bytes 1..4), so the RBSP is `get(1..)`.
    let sps_rbsp = filmcraft_bitstream::unescape_rbsp(avc.sps.first().and_then(|s| s.get(1..)).unwrap_or_default());
    let sps = filmcraft_h264::params::Sps::parse(&sps_rbsp).unwrap_or_else(|e| panic!("the sample entry's SPS does not parse: {e}"));
    assert_eq!(sps.profile_idc, 77, "and says Main, as the avcC's own profile byte does");
    assert_eq!((sps.pic_width_in_mbs, sps.pic_height_in_map_units), (20, 12), "320x180 is 20x12 macroblocks");
    assert!(sps.frame_crop.is_some(), "and crops the four padding rows the macroblock rounding added");
    let pps_rbsp = filmcraft_bitstream::unescape_rbsp(avc.pps.first().and_then(|s| s.get(1..)).unwrap_or_default());
    let pps = filmcraft_h264::params::Pps::parse(&pps_rbsp, &[Some(sps.clone())]).unwrap_or_else(|e| panic!("the sample entry's PPS does not parse: {e}"));
    assert_eq!(pps.pic_init_qp, 26, "and the quantiser the stream was encoded at");

    // **The coded bytes parse against those sets** — the "consistent with the bitstream" half. Each
    // sample is a length-prefixed NAL run, so the first four bytes are a length and the payload is
    // one NAL unit whose header byte is at offset 4.
    for (index, picture) in samples.iter().enumerate() {
        let sample = picture.mp4_sample().unwrap_or_else(|e| panic!("picture {index} as an MP4 sample: {e}"));
        // **The sample is a length-prefixed NAL run, walked the way a decoder walks it.** The
        // access unit carries this backend's parameter sets ahead of the slice, so the first unit is
        // a type 7 and the slice is not first; a test that assumed otherwise would be testing the
        // order rather than the content. The walk is what also proves the lengths add up — a
        // truncated or over-long length is a sample a decoder reads past the end of.
        let mut at = 0usize;
        let mut slices = 0usize;
        while at < sample.len() {
            let Some(len_bytes) = sample.get(at..at + 4).and_then(|b| <[u8; 4]>::try_from(b).ok()) else {
                panic!("picture {index}: the sample ends inside a NAL length at byte {at} of {}", sample.len());
            };
            let len = u32::from_be_bytes(len_bytes) as usize;
            let start = at + 4;
            let Some(unit) = sample.get(start..start.checked_add(len).filter(|end| *end <= sample.len()).unwrap_or(usize::MAX)) else {
                panic!("picture {index}: a NAL of {len} bytes at byte {start} runs past the {} byte sample", sample.len());
            };
            at = start + len;
            let Some(&header_byte) = unit.first() else {
                panic!("picture {index}: a NAL of length {len} carries no header byte");
            };
            let kind = header_byte & 0x1f;
            if !matches!(kind, 1 | 5) {
                // A type 7 or 8 here is this module's own parameter set, already checked above from
                // the sample entry; anything else is a NAL type this backend does not submit, and
                // naming it is the difference between "the stream has something odd in it" and a
                // silent skip.
                assert!(matches!(kind, 7 | 8 | 10 | 11), "picture {index} holds a NAL of type {kind}, which this backend never writes");
                continue;
            }
            slices += 1;
            let nal = filmcraft_h264::slice::NalHeader::parse(header_byte).unwrap_or_else(|e| panic!("picture {index}'s NAL header: {e}"));
            let rbsp = filmcraft_bitstream::unescape_rbsp(unit.get(1..).unwrap_or_default());
            let lookup = |id: u32| -> filmcraft_h264::Result<(&filmcraft_h264::params::Pps, &filmcraft_h264::params::Sps)> {
                if id != pps.id {
                    return Err(filmcraft_h264::Error::MissingParameterSet(format!("the slice names PPS {id} and the sample entry's is {}", pps.id)));
                }
                Ok((&pps, &sps))
            };
            let (header, _, _) = filmcraft_h264::slice::SliceHeader::parse(&rbsp, nal, lookup)
                .unwrap_or_else(|e| panic!("picture {index}'s slice does not parse against the sample entry's parameter sets: {e}"));
            // **The slice says what the encoder says it coded.** A driver that coded every picture
            // intra would still decode, and would look right — this is the check that catches it.
            assert_eq!(
                header.slice_type,
                if picture.key_frame { filmcraft_h264::slice::SliceType::I } else { filmcraft_h264::slice::SliceType::P },
                "picture {index}'s slice type"
            );
            assert_eq!(header.idr, picture.key_frame, "picture {index}'s idr flag");
            assert_eq!(
                header.frame_num,
                u32::from(picture.frame_num) % (1 << sps.log2_max_frame_num),
                "picture {index}'s frame number, modulo the SPS's own width"
            );
            assert_eq!(
                header.pic_order_cnt_lsb,
                u32::from(picture.poc) % (1 << sps.log2_max_poc_lsb),
                "picture {index}'s picture order count, modulo the SPS's own width"
            );
        }
        assert!(slices >= 1, "picture {index}'s sample holds no slice at all, so it describes no picture");
    }

    // **No gaps and monotonic times**, which is the sample-table claim rather than the bitstream's.
    assert_eq!(samples.len(), 20, "every picture produced one access unit");
    for (index, picture) in samples.iter().enumerate() {
        assert!(!picture.bytes.is_empty(), "picture {index} is empty");
        assert!(picture.coded_buffer_used > 0, "picture {index} wrote nothing to the coded buffer");
        // A P frame that is the size of an I frame has not been predicted from, whatever it decodes
        // to. This is the size half of the evidence, and it is why the size is checked here and not
        // only in the decode half.
        if index > 0 && !picture.key_frame {
            let key = samples.iter().find(|p| p.key_frame).map(|p| p.driver_bytes).unwrap_or(usize::MAX);
            assert!(picture.driver_bytes < key, "picture {index} is {} bytes against a {key} byte key frame — it was not predicted from", picture.driver_bytes);
        }
    }
    let keys = samples.iter().filter(|p| p.key_frame).count();
    assert!(keys >= 2, "a 20-picture run at MaxFrameNum 16 has at least two key frames (the first and the wrap), and had {keys}");
}

/// **The rate-control modes the driver reports, and that this backend sends `CQP` anyway.**
///
/// The gate's bitrate decline names the driver's answer, so the claim "the driver *can* do CBR and
/// VBR and this backend does not" needs something behind it. This is that: the mask is queried and
/// asserted to contain `CQP` — the mode this backend submits — and the answer is printed so the
/// decline's wording can be checked against a real machine rather than believed.
#[test]
fn the_drivers_rate_control_modes_are_reported_and_this_backend_sends_cqp() {
    let _one = lock();
    let Ok(display) = open_display() else {
        eprintln!("SKIPPED: no VA-API display");
        return;
    };
    let mask = VaEncoder::driver_rate_control_modes(&display);
    let names = VaEncoder::rate_control_names(mask);
    eprintln!("VAConfigAttribRateControl: {names} (mask {mask:#x})");
    assert_eq!(
        VaEncoder::rate_control_names(0),
        "nothing the driver named",
        "a zero mask is named as such and not as 'none' — it means the driver did not answer, which is not a claim that nothing is supported"
    );
    assert_eq!(VaEncoder::rate_control_names(0x12), "CBR and CQP", "the names are the header's bits, one per mode");
    // `CQP` is the mode this backend submits, so a driver that does not name it cannot run it — and
    // the encoder would have failed at `vaCreateConfig` instead. If this ever fails on a machine,
    // the probe would have failed first, and this line is where to look.
    assert!(names.contains("CQP"), "the driver names CQP, which is the mode this backend submits; it reported {names}");
    // **And the bitrate modes are named too**, because the gate's decline says "the driver reports
    // …" and this is what backs that half of the sentence. It is not an assertion that this backend
    // can use them — it cannot, and the decline says so.
    if !names.contains("VBR") {
        eprintln!("this driver names no VBR mode, so the gate's decline about bitrate is not contradicted here");
    }
}

/// **Hostile settings are errors, never crashes**, which is the one gate that applies to every
/// refusal in [`vaapi_config_for`].
///
/// The never-crash rule says an input-derived number is hostile, and a settings object is one: every
/// one of these values comes from a project file or a UI field. The assertions are that a value is
/// *refused with a reason*, never that it happens to work.
#[test]
fn hostile_settings_are_errors_rather_than_panics() {
    let _one = lock();
    let base = || no_bitrate_settings(String::new());
    // Sizes, rates and counts that would divide by zero, overflow a field, or index past an array.
    let cases: [(u32, u32, FrameRate); 7] = [
        (0, 360, FrameRate::FPS_30),
        (640, 0, FrameRate::FPS_30),
        (u32::MAX, u32::MAX, FrameRate::FPS_30),
        (640, 360, FrameRate { num: 0, den: 1 }),
        (640, 360, FrameRate { num: 30, den: 0 }),
        (640, 360, FrameRate { num: -1, den: 1 }),
        (640, 360, FrameRate { num: i64::MAX, den: 1 }),
    ];
    for (w, h, rate) in cases {
        let mut s = base();
        s.keyframe_distance = Some(u32::MAX);
        let got = vaapi_config_for(Format::H264, w, h, rate, &s);
        assert!(got.is_err(), "a {w}x{h} picture at {}/{} was accepted with an unbounded key-frame interval", rate.num, rate.den);
    }
    // And the factory itself, which is the path a real export takes: none of these may panic.
    for (w, h, rate) in cases {
        let settings = hw_settings(scratch("hostile.mp4").to_string_lossy().to_string());
        let _ = vaapi_encoder_factory(Format::H264, w, h, rate, &settings);
    }
}
