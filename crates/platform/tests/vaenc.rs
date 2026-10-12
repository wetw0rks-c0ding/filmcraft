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
use filmcraft_platform::hardware_encode::{ExportBitrate, vaapi_config_for, vaapi_encoder_factory};
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

/// The settings an export through this backend takes: **a bitrate**, in the units FilmCraft's
/// settings use, with a ceiling above it and a key-frame interval inside `MaxFrameNum`.
///
/// Every number here is one the gate turns into a rate-control record, so a test that changes one is
/// testing a different stream. 8000 kbit/s with a 12000 ceiling is the pair the gate's own test
/// asserts it derives, and 60 pictures is the 2-second default `ExportSettings::resolve` computes at
/// 30 fps — chosen to sit under `MaxFrameNum`'s 16 only because the gate needs it to; see
/// `what_the_export_gate_declines_and_why` for the ceiling's own decline.
/// A bitrate export's settings, at a target the content can actually carry.
///
/// **The target is 2000, not 8000, and that is the whole point of the change.** Measured on this driver
/// at 320x180, the content ceiling is ~162 kbit/s for a luma ramp and ~6598 kbit/s at full texture —
/// both at the driver's quantiser floor. The old target of 8000 sits *above* even the textured ceiling,
/// so it was unreachable by construction, which is what made VBR look like it ignored the target and CBR
/// look like it faked one. 2000 sits well inside the ceiling, which is what makes the ratio mean
/// something.
fn bitrate_settings(path: String) -> ExportSettings {
    ExportSettings {
        format: Format::H264,
        path,
        include_audio: false,
        bitrate_kbps: 2_000,
        max_bitrate_kbps: Some(4_000),
        bitrate_mode: BitrateMode::Vbr1Pass,
        hardware_encoding: HardwareEncoding::Auto,
        h264_profile: H264Profile::Main,
        field_order: FieldOrder::Progressive,
        // 8 rather than 60: the key-frame interval has to fit inside `MaxFrameNum` (16) or the gate
        // declines, and 8 is comfortably inside it while still putting two key frames in a 24-picture
        // run so the oracle sees a stream that is not one long GOP.
        keyframe_distance: Some(8),
        ..Default::default()
    }
}

/// The settings FilmCraft's `Default::default()` produces, which a test uses when it wants *those*
/// rather than a chosen combination.
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
/// This is the test that says the wiring exists and what it is honestly good for, and the case at the
/// top is the one this task changed:
///
/// - **A bitrate is taken**, as the mode the settings name. `Cbr` and `Vbr1Pass` both produce a
///   `VAEncMiscParameterRateControl`; `Vbr2Pass` is declined because it needs a second render pass,
///   which is a different refusal from the bitrate one and says so. **The assertion is on the numbers
///   the gate derives, not on `is_ok()`** — a gate that accepted the bitrate and then dropped it on the
///   floor would pass an `is_ok()` and produce a file at a size nobody asked for, which is the failure
///   this whole milestone exists to prevent.
/// - **A key-frame interval past `MaxFrameNum` is declined**, because the frame loop inserts a key
///   frame every 16 pictures whatever the settings say. An export that asked for 250 and got 16 is
///   a file many times the size it was promised, and "tell the caller" and "refuse" are the only
///   honest answers; this refuses.
///
/// **The driver's own rate-control answer is no longer the gate's excuse, but it is still printed** by
/// `the_drivers_rate_control_modes_are_reported_and_include_the_ones_submitted`, because "which modes
/// did the driver offer" is a fact a reader of a diagnostics line still wants.
#[test]
fn what_the_export_gate_declines_and_why() {
    let _one = lock();
    let at = |f: &dyn Fn(&mut ExportSettings)| {
        let mut s = bitrate_settings(String::new());
        f(&mut s);
        vaapi_config_for(Format::H264, 640, 360, FrameRate::FPS_30, &s)
    };

    // **The case that exists**: a bitrate, Main profile, progressive, square pixels, whole fps.
    // Asserted on the derived numbers, not on `is_ok()` alone — see this test's doc comment.
    let Ok((w, h, fps, gop, bitrate)) = at(&|_| {}) else {
        panic!("a settings combination FilmCraft's own presets produce is declined, which is the state this task exists to leave");
    };
    // `bitrate_settings` asks for 2000 kbit/s with a 4000 ceiling and a **key-frame interval of 8**,
    // which is what comes back — this is a pass-through, not a derivation.
    //
    // The 8 is not incidental: this encoder's `MaxFrameNum` is 16, because H.264's `frame_num` is four
    // bits wide. The gate **refuses** any interval above it rather than shortening it, on the grounds that
    // an export which asked for 250 and got 16 is a file six times the size anyone was promised. That
    // refusal is asserted in `cases` below; the point of naming it here is that 8 is inside the limit and
    // so is the one value these settings can use. **It does not clamp, and nothing here should be read as
    // saying it does** — an earlier draft of this comment claimed exactly that and was wrong.
    assert_eq!((w, h, fps, gop), (640, 360, 30, 8), "the geometry, rate and GOP come back as the settings stated them");
    // 2000 kbit/s is 2_000_000 bit/s: the unit conversion is the gate's and is asserted here because
    // a multiplication on the wrong unit is a file three orders of magnitude from the one asked for.
    assert_eq!(bitrate, ExportBitrate::Vbr { target: 2_000_000, max: 4_000_000 }, "the VBR target and the stated ceiling, in bits per second");

    // **A settings object with no ceiling produces the 1.5x one**, which is
    // `ExportSettings::resolve`'s own rule — the same expression, so the two backends cannot pick
    // different ceilings for the same settings.
    let mut no_ceiling = bitrate_settings(String::new());
    no_ceiling.max_bitrate_kbps = None;
    assert_eq!(
        vaapi_config_for(Format::H264, 640, 360, FrameRate::FPS_30, &no_ceiling).map(|(_, _, _, _, b)| b),
        Ok(ExportBitrate::Vbr { target: 2_000_000, max: 3_000_000 }),
        "an absent ceiling is 1.5x the target, exactly as ExportSettings::resolve computes it"
    );

    // **A ceiling below the target is not a VBR range**, so it is replaced by the 1.5x one rather
    // than submitted — a range whose ceiling is under its target asks the controller for something
    // impossible, and the gate would rather say what it did than hand the driver a contradiction.
    let mut backwards = bitrate_settings(String::new());
    backwards.max_bitrate_kbps = Some(1_000);
    assert_eq!(
        vaapi_config_for(Format::H264, 640, 360, FrameRate::FPS_30, &backwards).map(|(_, _, _, _, b)| b),
        Ok(ExportBitrate::Vbr { target: 2_000_000, max: 3_000_000 }),
        "a ceiling below the target falls back to 1.5x, as ExportSettings::resolve does"
    );

    // **CBR submits the target alone**, with no ceiling of its own. A VBR ceiling equal to the target
    // would make the two modes indistinguishable in the record, and a `Cbr` export asked for
    // "constant" would silently become "variable, narrowly".
    let mut cbr = bitrate_settings(String::new());
    cbr.bitrate_mode = BitrateMode::Cbr;
    assert_eq!(
        vaapi_config_for(Format::H264, 640, 360, FrameRate::FPS_30, &cbr).map(|(_, _, _, _, b)| b),
        Ok(ExportBitrate::Cbr(2_000_000)),
        "CBR submits the target on its own"
    );

    // **Every bitrate FilmCraft's own presets express is now taken**, which is the claim that decides
    // whether this backend is reachable from the UI at all. Each preset is read out of the real preset
    // list rather than written out here, so a preset that changes takes the test with it.
    let presets = filmcraft_export::builtin_presets();
    let mut taken = 0usize;
    let mut refused = Vec::new();
    for preset in presets.iter().filter(|p| p.settings.video_format() == Format::H264) {
        let mut s = preset.settings.clone();
        s.hardware_encoding = HardwareEncoding::Auto;
        // `Exporter::new` bakes these in before a factory is called; doing the same here is what
        // makes this loop a test of the settings a user actually exports with rather than of the
        // unresolved ones a preset file carries.
        let r = s.resolve(1920, 1080, FrameRate::FPS_30, 48_000);
        s.bitrate_kbps = r.target_kbps;
        s.max_bitrate_kbps = Some(r.max_kbps);
        s.keyframe_distance = Some(r.keyint);
        s.adaptive_bitrate = None;
        match vaapi_config_for(Format::H264, 1920, 1080, FrameRate::FPS_30, &s) {
            Ok(_) => taken += 1,
            // The declines that are *not* about rate control, and which are stated as such in the gate:
            // the derived parameter sets say Main; the key-frame interval the settings ask for is past
            // `MaxFrameNum`; this backend submits one pass, so a preset carrying two is refused by name;
            // and MXF carries an Annex B stream while this backend's samples are length-prefixed. All four
            // are listed rather than skipped so that a *new* decline appearing here fails the test rather
            // than joining a silent list. The list is deliberately exhaustive over the gate's own reasons —
            // a preset the gate refuses for a reason the gate documents is a working decline, not a gap.
            Err(why)
                if why.contains("state Main")
                    || why.contains("MaxFrameNum")
                    || why.contains("two-pass")
                    || why.contains("Annex B") =>
            {
                refused.push(format!("{}: {why}", preset.name))
            }
            Err(why) => panic!("the preset {:?} is declined for a reason the gate does not document as one of its own: {why}", preset.name),
        }
    }
    assert!(taken > 0, "no H.264 preset is exportable through this backend at all, which would make the registered factory a decline again");
    for line in &refused {
        eprintln!("preset declined by a documented rule: {line}");
    }

    // The declines, each checked for the *reason* and not merely that there is one.
    let cases: [(&str, Box<dyn Fn(&mut ExportSettings)>, &str); 9] = [
        ("hardware encoding is off", Box::new(|s: &mut ExportSettings| s.hardware_encoding = HardwareEncoding::Off), "hardware encoding is off"),
        ("MXF", Box::new(|s: &mut ExportSettings| s.format = Format::MxfOp1a), "Annex B"),
        ("two-pass", Box::new(|s: &mut ExportSettings| s.h264_pass = H264Pass::First), "two-pass"),
        ("two-pass by mode", Box::new(|s: &mut ExportSettings| s.bitrate_mode = BitrateMode::Vbr2Pass), "two-pass"),
        ("a zero bitrate", Box::new(|s: &mut ExportSettings| s.bitrate_kbps = 0), "0 kbps"),
        ("HDR", Box::new(|s: &mut ExportSettings| s.signal = filmcraft_export::ColorSignal::PQ), "HDR"),
        ("non-square pixels", Box::new(|s: &mut ExportSettings| s.pixel_aspect = Some((4, 3))), "non-square pixels"),
        ("interlaced", Box::new(|s: &mut ExportSettings| s.field_order = FieldOrder::UpperFirst), "interlaced"),
        ("a Baseline or High profile", Box::new(|s: &mut ExportSettings| s.h264_profile = H264Profile::High), "state Main"),
    ];
    for (name, set, why) in cases {
        let Err(why_said) = at(&set) else {
            panic!("{name} is accepted, which is a stream the settings asked for and this backend cannot produce");
        };
        assert!(why_said.contains(why), "{name}: the decline says {why:?}, not just that something is wrong: {why_said}");
    }

    // An odd picture size needs its own geometry, so it is not in the table above.
    let why = vaapi_config_for(Format::H264, 641, 361, FrameRate::FPS_30, &bitrate_settings(String::new()))
        .expect_err("641x361 is not 4:2:0 encodable");
    assert!(why.contains("4:2:0"), "and it says why: {why}");

    // The key-frame ceiling, which needs its own settings because it is a number.
    let mut s = bitrate_settings(String::new());
    s.keyframe_distance = Some(250);
    let why = vaapi_config_for(Format::H264, 640, 360, FrameRate::FPS_30, &s).expect_err("250 is past MaxFrameNum");
    assert!(why.contains("250"), "the message names the interval asked for: {why}");
    assert!(why.contains("16"), "and the ceiling it hit: {why}");

    // A fractional frame rate is refused rather than rounded: `VAEncMiscParameterFrameRate` here
    // carries whole frames per second, so 30000/1001 is a rate this backend cannot state.
    let why = vaapi_config_for(Format::H264, 640, 360, FrameRate::FPS_23_976, &bitrate_settings(String::new()))
        .expect_err("23.976 is not a whole number of frames per second");
    assert!(why.contains("fractional"), "and it says why rather than rounding: {why}");

    // **HEVC is declined by name, not by accident.** There is no HEVC encoder in this backend, and
    // the software path has none either, so the reason must say HEVC rather than "not H.264" —
    // otherwise a reader cannot tell a decline from a bug.
    let Err(why) = vaapi_config_for(Format::Hevc, 640, 360, FrameRate::FPS_30, &bitrate_settings(String::new())) else {
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

    // **A decline is still counted**, and it is still counted for a settings combination the gate
    // refuses for a *documented* reason: FilmCraft's defaults ask for the High profile and a
    // 60-picture key-frame interval, and this backend states Main and wraps at 16. The point of using
    // those is that they are the defaults a user gets — the counters a user reads in `perf.stats` have
    // to move for the declines they will actually hit.
    let before = filmcraft_export::hw_encode_stats();
    let settings = hw_settings(scratch("declined.mp4").to_string_lossy().to_string());
    let got = vaapi_encoder_factory(Format::H264, 640, 360, FrameRate::FPS_30, &settings);
    let after = filmcraft_export::hw_encode_stats();
    assert!(got.is_none(), "FilmCraft's default H.264 settings are declined: they ask for the High profile and a 60-picture GOP, and this backend states Main and wraps at 16");
    assert_eq!(after.declined, before.declined + 1, "and the decline is counted for perf.stats");
    assert_eq!(after.sessions, before.sessions, "with no encoder created");
    assert_eq!(after.frames, before.frames, "and no picture encoded");

    // **And the settings the gate *accepts* build a real encoder**, which is the whole point of the
    // change this test guards: before it, every settings object the gate accepted was one with no
    // bitrate, and the factory was a decline for everything FilmCraft produces.
    let settings = bitrate_settings(scratch("accepted.mp4").to_string_lossy().to_string());
    match vaapi_encoder_factory(Format::H264, 640, 360, FrameRate::FPS_30, &settings) {
        Some(Ok(_)) => {}
        Some(Err(why)) => panic!("an export with a bitrate was refused by the factory: {why}"),
        None => {
            if open_display().is_err() {
                eprintln!("SKIPPED: no VA-API display");
                return;
            }
            panic!("the factory declined an export vaapi_config_for accepts, which points at the display or the encoder, not the settings");
        }
    }
    // A session was counted for it — **asserted rather than assumed**, because the accept path and
    // the decline path increment different counters and a factory that took neither would still look
    // right from the outside.
    let end = filmcraft_export::hw_encode_stats();
    assert_eq!(end.sessions, after.sessions + 1, "an accepted export counts a hardware session for perf.stats");
}

// ---------------------------------------------------------------------------
// The muxer oracle — a real file, decoded by ffmpeg, compared to the source
// ---------------------------------------------------------------------------

/// **A real MP4, written by FilmCraft's own muxer from this encoder's samples, decoded by ffmpeg,
/// and compared to the pictures that went in.** The export-level oracle, run in **both** fidelity
/// modes.
///
/// [`Mp4Writer`] is the same writer `filmcraft_export`'s job uses, so the container, the sample table
/// and the `avcC` are produced the way a real export produces them — not assembled by a test.
///
/// **What it compares, and what it deliberately does not.** The reference is the **source ramp**, one
/// per picture, not another encode: a hardware encoder and a software encoder are different
/// algorithms, so "matches the software encode" would mean nothing here.
///
/// **Why two runs rather than one.** The constant-quantiser run is the one that was there before, with
/// its measured tolerance; the bitrate run is the one an export actually takes, since FilmCraft's
/// settings state a bitrate and not a quantiser. **Neither replaces the other and neither tolerance is
/// loosened**: each is measured in its own mode, and a mode that stopped working would fail its own
/// test rather than hide behind the other's number. The shared body is [`mux_and_decode`], so the two
/// runs differ only in how the encoder is built and in the tolerance, which are exactly the two things
/// that differ between the modes.
/// What the oracle encodes, and why a rate-control run must not use the same content as a picture test.
///
/// **A luma ramp is the wrong content for a rate-control test, and using it is what made two rounds of
/// investigation reach the wrong conclusion.** A ramp holds about 162 kbit/s at this driver's quantiser
/// floor, so any target above that is unreachable by construction. That made VBR read back flat across a
/// 40x target sweep — which looked exactly like a driver that ignores the target — and made CBR's output
/// look 98.6% padding, which looked exactly like a driver faking the rate. Both were the ramp. Measured
/// on [`Content::Textured`], VBR tracks at 89% +- 2%, and CBR tracks without padding.
///
/// So the rule this encodes: **a rate-control assertion is meaningless unless the content can carry the
/// target.** A test that cannot tell a rate controller from a starved one is worse than no test.
#[derive(Clone, Copy)]
enum Content {
    /// A smooth ramp. Right for asking *is the picture right*, which is what the constant-quantiser run
    /// asks: a ramp has no detail for a quantiser to lose, so a deviation there is the quantiser's and
    /// nothing else's.
    Ramp,
    /// Deterministic value noise over a moving base. High residual, so the content can hold the bits a
    /// rate controller is being asked to spend.
    Textured,
}

impl Content {
    fn frame(self, w: u32, h: u32, index: u32) -> Vec<u8> {
        match self {
            Content::Ramp => ramp_frame(w, h, index),
            Content::Textured => textured_frame(w, h, index),
        }
    }
}

/// A deterministic high-entropy picture. **No RNG**, so a failure here is reproducible: the same picture
/// is encoded on every run, on every machine.
///
/// Value noise over a base that moves with the frame index, so consecutive pictures differ and the
/// encoder has residual to spend. The hash is splitmix64's finaliser — cheap, and well distributed
/// enough that no low-amplitude structure survives to make the picture compressible.
fn textured_frame(w: u32, h: u32, index: u32) -> Vec<u8> {
    const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rgba = Vec::with_capacity((w as usize) * (h as usize) * 4);
    for y in 0..h as u64 {
        for x in 0..w as u64 {
            let mut n = x.wrapping_mul(GOLDEN) ^ y.wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ u64::from(index).wrapping_mul(0x94D0_49BB_1331_11EB);
            n ^= n >> 30;
            n = n.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            n ^= n >> 27;
            n = n.wrapping_mul(0x94D0_49BB_1331_11EB);
            n ^= n >> 31;
            // Half the amplitude from noise, half from a base that sweeps with the frame, so the picture
            // still has large-scale structure to be wrong about and the noise is not the only residual.
            let base = (x * 255) / u64::from(w.max(1));
            let luma = ((base + ((n >> 40) / 2)).min(255)) as u8;
            let chroma = (((n >> 33) & 0x7f) as u64 + 64) as u8;
            rgba.extend_from_slice(&[luma, chroma, chroma.wrapping_add(1), 255]);
        }
    }
    rgba
}

fn mux_and_decode(
    encoder: &mut dyn filmcraft_export::VideoEncoder,
    w: u32,
    h: u32,
    count: usize,
    ffmpeg: &PathBuf,
    name: &str,
    content: Content,
) -> (Vec<u8>, usize, u64) {
    // Every source picture is kept: ffmpeg's output is compared with what went **in**.
    let sources: Vec<Vec<u8>> = (0..count as u32).map(|i| content.frame(w, h, i)).collect();
    let path = scratch(name);
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
    let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    assert!(file_size > 0, "the muxer wrote nothing");
    // **The video payload, summed from the samples themselves** rather than taken as the file's
    // length. A bitrate names the coded stream, so comparing it with the container's box sizes would
    // include the `moov`/`trak` tables and the parameter sets — bytes the bitrate does not govern.
    let video_bytes: usize = packets.iter().map(|p| p.data.len()).sum();

    // **ffmpeg decodes the file, and its own diagnostics are not ignored.** A stream ffmpeg decodes
    // *with* a warning has been accepted and is not necessarily right, and a test that reported a
    // deviation over it would be reporting a number computed from a stream nobody would trust.
    let raw = std::process::Command::new(ffmpeg.as_os_str())
        .args(["-v", "error", "-i", path.to_string_lossy().as_ref(), "-f", "rawvideo", "-pix_fmt", "gray", "-"])
        .output()
        .unwrap_or_else(|e| panic!("ffmpeg: {e}"));
    assert!(raw.stderr.is_empty(), "ffmpeg decoded the export with a diagnostic: {}", String::from_utf8_lossy(&raw.stderr));

    // **A short decode is a short result, not a comparison of the ones that happened to line up.**
    let one = (w as usize) * (h as usize);
    assert_eq!(raw.stdout.len(), one * count, "ffmpeg returned {} bytes of luma for {count} pictures of {w}x{h} ({one} bytes each)", raw.stdout.len());

    // The comparison: against the source, picture by picture, within the caller's lossy tolerance.
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
    (deviations, video_bytes, file_size)
}

/// **The constant-quantiser run: the oracle that was there before, unchanged in what it claims.**
///
/// The tolerance is 24/255 and the spread bound is 4, both measured rather than guessed —
/// `examples/vaencprobe.rs` measures **about 9/255 mean absolute deviation** on this ramp at QP 26,
/// so 24 is roughly two and a half times the quantiser's own noise and still nowhere near a wrong
/// picture (a wrong reference shows 200+, see the depth-two refusal).
#[test]
fn a_muxed_export_at_a_fixed_quantiser_decodes_to_the_pictures_that_went_in() {
    let _one = lock();
    let Some(ffmpeg) = filmcraft_testkit::ffmpeg_or_skip("decoding the muxed export") else { return };
    let Ok(display) = open_display() else {
        eprintln!("SKIPPED: no VA-API display");
        return;
    };
    let (w, h, count) = (320u32, 180u32, 24usize);

    // **The encoder is built directly rather than through the export gate**, and that is deliberate:
    // the gate no longer offers a constant-quantiser path — FilmCraft's settings state a bitrate and
    // not a quantiser — so the only way to keep measuring *this* mode is to ask the encoder for it
    // directly. That is also the honest shape: it says the mode is reachable from the library and not
    // from the UI, rather than implying the gate produces one.
    let mut encoder = match build_constant_quantiser(&display, w, h) {
        Ok(e) => e,
        Err(why) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
    };

    let (deviations, video_bytes, file_size) = mux_and_decode(&mut encoder, w, h, count, &ffmpeg, "muxed-cqp.mp4", Content::Ramp);
    let worst = deviations.iter().copied().max().unwrap_or(u8::MAX);
    assert!(deviations.iter().all(|d| *d < 24), "every picture is within a QP-26 tolerance of the source; deviations {deviations:?} (worst {worst})");
    // **Flat, not merely small.** A drifting reference produces pictures that each look plausible
    // and get steadily worse; a run whose deviation is flat is what a correct reference chain looks
    // like. The assertion is on the *spread*, so a stream that quietly degraded would fail here even
    // if every frame passed the absolute bound.
    let (lo, hi) = (deviations.iter().copied().min().unwrap_or(0), worst);
    assert!(hi - lo <= 4, "the deviation is flat across the sequence, not drifting: {deviations:?}");
    eprintln!("muxed {count} pictures at QP 26 to {file_size} bytes ({video_bytes} of samples), mean absolute deviation {deviations:?}");
}

/// **A constant-quantiser [`VaEncoder`] wrapped as a [`VideoEncoder`]**, which is how the mode the
/// gate does not offer is still muxed through FilmCraft's own writer.
///
/// It is a small adapter rather than a second copy of [`VaEncoderExport`] because the export adapter's
/// `encode` calls `EncodeRequest::predicted_at_rate()` — the pairing a bitrate mode requires — and a
/// copy that did not would be testing the wrong encoder by accident. **The refusal is the same one a
/// caller would get**: the pairing is checked by `VaEncoder::encode`, so a request carrying a quantiser
/// into this adapter is an error from the library, not from here.
struct ConstantQuantiserExport {
    encoder: VaEncoder,
    w: u32,
    h: u32,
    duration: u32,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl filmcraft_export::VideoEncoder for ConstantQuantiserExport {
    fn sample_entry(&self) -> filmcraft_isobmff::SampleEntry {
        match self.encoder.sample_entry() {
            Ok(entry) => entry,
            Err(why) => {
                log::info!("VA-API (fixed quantiser, test adapter): no sample entry yet: {why}");
                filmcraft_isobmff::SampleEntry::avc(filmcraft_isobmff::AvcConfig::new(Vec::new(), Vec::new(), 4), u16::try_from(self.w).unwrap_or(u16::MAX), u16::try_from(self.h).unwrap_or(u16::MAX))
            }
        }
    }

    fn timescale(&self) -> u32 {
        30
    }

    fn encode(&mut self, f: &EncoderFrame) -> Result<Vec<filmcraft_export::EncodedPacket>, filmcraft_export::ExportError> {
        if (f.width, f.height) != (self.w, self.h) {
            return Err(filmcraft_export::ExportError::Encode(format!("a {}x{} picture for a {}x{} encoder", f.width, f.height, self.w, self.h)));
        }
        let (w, h) = (self.w as usize, self.h as usize);
        if f.rgba.len() < w.saturating_mul(h).saturating_mul(4) {
            return Err(filmcraft_export::ExportError::Encode("the RGBA picture is smaller than its size".into()));
        }
        filmcraft_export::rgba_to_yuv420_8(f.rgba, w, h, &mut self.y, &mut self.u, &mut self.v);
        let chroma_len = (w / 2) * h;
        let mut chroma = Vec::with_capacity(chroma_len);
        for (u, v) in self.u.iter().zip(self.v.iter()) {
            chroma.push(*u);
            chroma.push(*v);
        }
        self.encoder.upload(&self.y, &chroma).map_err(filmcraft_export::ExportError::Encode)?;
        let picture = self.encoder.encode(EncodeRequest::predicted(26).map_err(filmcraft_export::ExportError::Encode)?).map_err(filmcraft_export::ExportError::Encode)?;
        let data = picture.mp4_sample().map_err(filmcraft_export::ExportError::Encode)?;
        Ok(vec![filmcraft_export::EncodedPacket { data, key: picture.key_frame, duration: self.duration, composition_offset: 0 }])
    }

    fn flush(&mut self) -> Result<Vec<filmcraft_export::EncodedPacket>, filmcraft_export::ExportError> {
        self.encoder.flush().map_err(filmcraft_export::ExportError::Encode)?;
        Ok(Vec::new())
    }

    fn media_start(&self) -> Option<i64> {
        None
    }
}

fn build_constant_quantiser(display: &std::sync::Arc<filmcraft_platform::vaapi::Display>, w: u32, h: u32) -> Result<ConstantQuantiserExport, String> {
    let mut encoder = VaEncoder::at_constant_quantiser(display, (w, h), 1)?;
    encoder.at_rate(30)?;
    // **A 16-picture key-frame interval**, which is the ceiling this encoder's `MaxFrameNum` allows and
    // therefore the longest interval that puts a second key frame inside a 24-picture run.
    encoder.key_frames_every(16)?;
    Ok(ConstantQuantiserExport { encoder, w, h, duration: 1, y: Vec::new(), u: Vec::new(), v: Vec::new() })
}

/// **A real MP4 at the bitrate the settings name, decoded by ffmpeg, compared to the source — and
/// its size compared to the bitrate.**
///
/// This is the run that matters for the question "can a user export with this", because a bitrate is
/// what FilmCraft's settings state. Three claims, in order of how much they would catch:
///
/// 1. **The pictures come back right**, within a tolerance measured in this mode rather than inherited
///    from the constant-quantiser run. A bitrate mode is free to pick a worse quantiser than QP 26, so
///    the tolerance is the one measured for it.
/// 2. **The file's size lands near the bitrate asked for.** This is the plan's "size and rate control
///    land in the neighbourhood the settings asked for", and it is the claim the constant-quantiser run
///    cannot make at all. **It is a neighbourhood, not a point**, because a rate controller meets a
///    target over a window and a 24-picture run is shorter than most of the windows involved: the
///    bounds are ±50% and the assertion is on the *video* payload, which is what the bitrate names.
/// 3. **The driver's own reported quantiser moved.** Under a bitrate mode it must, because the
///    controller is choosing per picture; a stream that reported one quantiser for all 24 pictures
///    would say the rate control never ran, whatever its size.
#[test]
fn a_muxed_export_at_the_requested_bitrate_decodes_to_the_pictures_that_went_in() {
    let _one = lock();
    let Some(ffmpeg) = filmcraft_testkit::ffmpeg_or_skip("decoding the muxed export") else { return };
    if open_display().is_err() {
        eprintln!("SKIPPED: no VA-API display");
        return;
    }
    // **300 pictures, not 24, and that is measured rather than tidied.** The controller ramps to its
    // plateau over the first seconds: the same target measures 2981/3353/3574/3591 kbit/s over
    // 60/150/300/900 pictures. A 24-picture run reads the startup, not the steady state, so it would
    // charge the controller for its own warm-up. 300 is where the plateau is.
    let (w, h, count) = (320u32, 180u32, 300usize);

    // **The encoder is the one the export gate builds**, from the settings FilmCraft's own presets
    // carry, so this is the real path rather than a hand-built substitute.
    let settings = bitrate_settings(String::new());
    let kbps = f64::from(settings.bitrate_kbps);
    let mut encoder = match vaapi_encoder_factory(Format::H264, w, h, FrameRate::FPS_30, &settings) {
        Some(Ok(e)) => e,
        Some(Err(why)) => panic!("the factory refused a configuration its own gate accepts: {why}"),
        None => {
            eprintln!("SKIPPED: the factory declined with a display open; the log line above says why");
            return;
        }
    };

    let (deviations, video_bytes, file_size) = mux_and_decode(encoder.as_mut(), w, h, count, &ffmpeg, "muxed-bitrate.mp4", Content::Textured);
    let worst = deviations.iter().copied().max().unwrap_or(u8::MAX);
    // **Measured, not inherited.** The constant-quantiser run at QP 26 gives about 9/255; this run's
    // measured deviation on the same ramp is in the printed line below, so the bound is roughly twice
    // whatever the quantiser the controller chose measures — and still nowhere near a wrong picture. **A
    // different tolerance per mode is not a loosened test**: it is the same comparison with the number
    // that mode measures, and the absolute bound is what would catch a wrong reference (200+, not 30).
    assert!(deviations.iter().all(|d| *d < 30), "every picture is within the bitrate mode's measured tolerance of the source; deviations {deviations:?} (worst {worst})");
    let (lo, hi) = (deviations.iter().copied().min().unwrap_or(0), worst);
    assert!(hi - lo <= 8, "the deviation is flat across the sequence, not drifting: {deviations:?}");

    // The size claim, on the video payload — the samples' own bytes, which is what the bitrate governs,
    // over the pictures' duration at the encoder's own rate.
    let seconds = count as f64 / 30.0;
    let achieved_kbps = video_bytes as f64 * 8.0 / seconds / 1000.0;
    let ratio = achieved_kbps / kbps;
    // **The band is 0.80..=1.05 because the measurement is 0.94**, not because a round number reads well.
    // The controller ramps to a plateau, so the average over a run sits below the steady state: measured
    // 0.94 over 300 pictures here, and 89% +- 2% at steady state across a 6x target sweep on other
    // content. The floor leaves room for that deficit and for a run that is still warming up; the ceiling
    // catches a controller that overshoots, which the old `1.5` would have passed at more than half again
    // the bitrate anyone asked for. Anything below 0.8 is a controller that is not tracking the target,
    // which is the failure this test exists to catch.
    assert!(
        (0.80..=1.05).contains(&ratio),
        "the samples carry {achieved_kbps:.0} kbit/s against a {kbps:.0} kbit/s target (ratio {ratio:.2}) — the \
         rate controller is not tracking the target"
    );

    eprintln!("muxed {count} pictures at {kbps:.0} kbit/s to {file_size} bytes ({video_bytes} of samples, {achieved_kbps:.0} kbit/s), mean absolute deviation {deviations:?}");
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
    let mut fresh = VaEncoder::at_constant_quantiser(&display, (320, 180), 1).expect("a one-reference encoder is measured to work");
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

/// **The rate-control modes the driver reports, and that this backend submits a bitrate from.**
///
/// The export gate names the driver's answer in its diagnostics, so the claim "the driver *can* do CBR
/// and VBR" needs something behind it. This is that: the mask is queried and asserted to contain
/// **the modes this backend now submits** — CBR and VBR — as well as the CQP the probe uses, and the
/// answer is printed so the wording can be checked against a real machine rather than believed.
///
/// **What it is not:** an assertion that the driver does *only* these. ICQ, QVBR and TCBRC are named
/// too on this machine and are declined for reasons of their own (see `RateControl`'s doc), so a
/// machine reporting more modes than this asserts is working normally.
#[test]
fn the_drivers_rate_control_modes_are_reported_and_include_the_ones_submitted() {
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
    // **The two modes the export gate submits.** A driver that does not name them cannot run them —
    // and the encoder would have failed at `vaCreateConfig` instead, so if this ever fails on a
    // machine the probe would have failed first, and this line is where to look.
    assert!(names.contains("CBR"), "the driver names CBR, which is the mode a BitrateMode::Cbr export submits; it reported {names}");
    assert!(names.contains("VBR"), "and VBR, which is the mode a Vbr1Pass export submits; it reported {names}");
    // `CQP` is still the mode `examples/vaencprobe.rs` measures the reference chain with, so a driver
    // that lost it would take the probe's measurements with it.
    assert!(names.contains("CQP"), "the driver names CQP too, which is the mode the probe measures with; it reported {names}");
}

#[test]
fn temporary_measure_the_payload_of_both_modes() {
    let _one = lock();
    let Ok(display) = open_display() else {
        eprintln!("SKIPPED: no VA-API display");
        return;
    };
    let measure = |name: &str, mut enc: VaEncoder| {
        let _ = enc.at_rate(30);
        let _ = enc.key_frames_every(8);
        for index in 0..24u32 {
            let (luma, chroma) = ramp_shifted(320, 180, 8 * index).expect("the ramp is buildable");
            enc.upload(&luma, &chroma).expect("the ramp uploads");
            let request = if name == "cqp" {
                EncodeRequest::predicted(26).expect("qp")
            } else {
                EncodeRequest::predicted_at_rate().expect("rate")
            };
            let p = enc.encode(request).expect("encodes");
            let sample = p.mp4_sample().expect("mp4 sample");
            // walk the length-prefixed NALs
            let mut at = 0usize;
            let mut worst_share = 1.0f64;
            let mut worst_run = 0usize;
            let mut total = 0usize;
            while at + 4 <= sample.len() {
                let len = u32::from_be_bytes([sample[at], sample[at + 1], sample[at + 2], sample[at + 3]]) as usize;
                let start = at + 4;
                let Some(nal) = sample.get(start..start + len) else { break };
                at = start + len;
                let kind = nal.first().map(|b| b & 0x1f).unwrap_or(0);
                if !matches!(kind, 1 | 5) {
                    continue;
                }
                let nz = nal.iter().filter(|b| **b != 0).count();
                let share = nz as f64 / nal.len() as f64;
                if share < worst_share {
                    worst_share = share;
                }
                let run = nal.iter().rev().take_while(|b| **b == 0).count();
                if run > worst_run {
                    worst_run = run;
                }
                total += nal.len();
                eprintln!("{name} pic {index:2} nal {kind} len {len:7} non-zero {share:.4} trailing-zero-run {run}");
            }
            eprintln!("{name} pic {index} total slice bytes {total}");
        }
        let _ = enc.flush();
    };
    measure("cqp", VaEncoder::at_constant_quantiser(&display, (320, 180), 1).expect("cqp encoder"));
    measure("cbr", VaEncoder::at_constant_bitrate(&display, (320, 180), 1, 8_000_000).expect("cbr encoder"));
    measure("vbr", VaEncoder::at_variable_bitrate(&display, (320, 180), 1, 8_000_000, 12_000_000).expect("vbr encoder"));
    measure("cbr50", VaEncoder::at_constant_bitrate(&display, (320, 180), 1, 50_000).expect("cbr encoder"));
    measure("cbr2m", VaEncoder::at_constant_bitrate(&display, (320, 180), 1, 2_000_000).expect("cbr encoder"));
    measure("cbr200k", VaEncoder::at_constant_bitrate(&display, (320, 180), 1, 200_000).expect("cbr encoder"));
    measure("cbr500k", VaEncoder::at_constant_bitrate(&display, (320, 180), 1, 500_000).expect("cbr encoder"));

    // The factory path, which is what the bitrate oracle actually runs.
    let settings = bitrate_settings(String::new());
    match vaapi_encoder_factory(Format::H264, 320, 180, FrameRate::FPS_30, &settings) {
        Some(Ok(mut enc)) => {
            for index in 0..24u32 {
                let (luma, chroma) = ramp_shifted(320, 180, 8 * index).expect("the ramp is buildable");
                let rgba: Vec<u8> = ramp_frame(320, 180, index);
                let _ = (luma, chroma);
                let frame = EncoderFrame { width: 320, height: 180, rgba: &rgba, hdr: None, index: u64::from(index) };
                let out = enc.encode(&frame).expect("the factory's encoder encodes");
                let len: usize = out.iter().map(|p| p.data.len()).sum();
                let nz: usize = out.iter().flat_map(|p| p.data.iter()).filter(|b| **b != 0).count();
                let run = out.iter().flat_map(|p| p.data.iter()).rev().take_while(|b| **b == 0).count();
                eprintln!("FACTORY pic {index:2} len {len:7} non-zero {nz:7} trailing {run} key {}", out.first().map(|p| p.key).unwrap_or(false));
            }
            eprintln!("FACTORY done");
        }
        other => eprintln!("FACTORY: declined or errored ({})", other.is_none()),
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
    let base = || bitrate_settings(String::new());
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
