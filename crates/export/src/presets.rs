//! Built-in export presets. These are FilmCraft's own functional definitions (names, sizes and
//! bitrates chosen by us for common delivery targets); no preset files from any other product are
//! used. User presets live in the engine (`<data dir>/export-presets.json`).

use filmcraft_time::FrameRate;
use serde::{Deserialize, Serialize};

use crate::settings::{AudioCodec, AudioSettings, BitrateMode, MxfVideoCodec, Scaling};
use crate::{ExportSettings, Format};

/// A named set of export settings (`path` and `range` are left empty).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportPreset {
    pub name: String,
    /// Browser group: "Match Source", "Web & Social", "Broadcast", "Image Sequence", "Animated
    /// GIF", "Audio Only" (user presets default to "User").
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub description: String,
    pub settings: ExportSettings,
    /// Built in (not saved, cannot be deleted).
    #[serde(skip)]
    pub builtin: bool,
}

/// Name comparison key: case- and punctuation-insensitive (so `Match Source - Adaptive High
/// Bitrate` finds `Match Source – Adaptive High Bitrate`).
pub fn preset_key(name: &str) -> String {
    name.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

/// The default preset (Quick Export, a fresh Export mode).
pub const DEFAULT_PRESET: &str = "Match Source – Adaptive High Bitrate";

/// The settings every H.264 preset starts from.
///
/// **It names no profile, deliberately.** `h264_profile` is left at the default,
/// [`crate::settings::H264Profile::Auto`], so a preset inherits "whatever this machine can
/// actually produce" rather than a profile this file would have to keep true on every platform
/// (see [`crate::ExportSettings::resolved_h264_profile`]). A preset that promised High and got a
/// Main file would be a shipped description that is false, which is worse than a description that
/// says nothing about the profile at all.
fn h264(size: Option<(u32, u32)>, kbps: u32, max_kbps: u32) -> ExportSettings {
    ExportSettings {
        format: Format::H264,
        frame_size: size,
        bitrate_kbps: kbps,
        max_bitrate_kbps: Some(max_kbps),
        bitrate_mode: BitrateMode::Vbr1Pass,
        audio: AudioSettings { codec: AudioCodec::Aac, sample_rate: Some(48_000), channels: 2, bitrate_kbps: 320, bits: 16 },
        ..Default::default()
    }
}

fn adaptive(bpp: f32) -> ExportSettings {
    ExportSettings { adaptive_bitrate: Some(bpp), max_bitrate_kbps: None, ..h264(None, 10_000, 0) }
}

/// The settings every HEVC preset starts from.
///
/// **It names no profile, for the same reason [`h264`] names none** — except that there is nothing
/// here to delegate to. HEVC's profile is the encoder's own: this backend's derived parameter sets
/// write `general_profile_idc = 1` (Main) and FilmCraft has no HEVC profile setting to be false
/// about, so the file's `profile` field is whatever the machine produced.
///
/// # The three H.264 defaults that do not carry over, and why each is refused rather than faked
///
/// - **`BitrateMode::Vbr1Pass`, and never constant quality.** `ConstantQuality` states a quantiser
///   and no rate; every fidelity the HEVC gate can submit is a rate, so both hardware gates decline
///   that mode by name. **H.264 survives the decline** — the built-in encoder takes it and codes the
///   quantiser asked for — **while HEVC has nothing to fall back to and the export becomes "encoder
///   not available"**, which is a worse answer than a wrong file but still a worse answer than a
///   working preset. So this is a rate, and no HEVC preset is built on a quantiser.
/// - **One pass.** A two-pass export renders the sequence once to measure it and again to encode it,
///   and this backend is handed each picture exactly once. `ExportSettings::validate` refuses
///   HEVC + `Vbr2Pass` before an encoder is ever asked, so a preset naming it would fail at the
///   settings layer rather than at the gate.
/// - **The sequence's own frame rate, unrounded.** The gate refuses a fractional rate (23.976,
///   29.97, 59.94) rather than rounding it to a stream nobody chose, and a preset cannot fix that:
///   naming a whole rate here would export a 23.976 sequence at 24 fps, which is a file that runs
///   fast. A whole-rate sequence is a **precondition of these presets, not a setting of them**.
///
/// # What is still true of every export through this path, and is the reason the preset is conditional
///
/// **The sequence's rate has to be one the stream can state.** The HEVC path writes the frame rate
/// into the SPS's VUI, and it writes it from a fixed 30 fps (`DEFAULT_FRAME_RATE_FOR_LEVEL` in
/// `crates/platform/src/vaapi/hevc_submit.rs`) rather than from the rate the caller stated, while the
/// `hvcC` carries the rate the caller did state. The cross-check that compares the two
/// (`check_hvcc_against_the_picture`) therefore **refuses every rate but 30 fps, by name**, rather
/// than writing a container that contradicts its own bitstream. Measured on this machine: 30 fps
/// exports (1920×1080, ffprobe `codec_name=hevc`, `nb_read_frames=30`, ffmpeg silent), and 24, 25
/// and 60 fps all fail with `the hvcC states 60 fps … and the VUI in the SPS it carries states 30`.
/// That is a defect in the encoder, not a property of HEVC, and it applies to every route to the
/// format — the preset does not create it and cannot hide it.
fn hevc(size: (u32, u32), kbps: u32, max_kbps: u32) -> ExportSettings {
    ExportSettings {
        format: Format::Hevc,
        frame_size: Some(size),
        bitrate_kbps: kbps,
        max_bitrate_kbps: Some(max_kbps),
        bitrate_mode: BitrateMode::Vbr1Pass,
        audio: AudioSettings { codec: AudioCodec::Aac, sample_rate: Some(48_000), channels: 2, bitrate_kbps: 320, bits: 16 },
        ..Default::default()
    }
}

/// The HEVC presets — **and the machine, not this file, decides whether there are any**.
///
/// # Why the list is conditional
///
/// [`crate::available`] is the same question the format list asks before it offers `Format::Hevc` at
/// all: a probe registered by `filmcraft_platform::register` that creates a real HEVC session and
/// asks the driver. **FilmCraft has no built-in HEVC encoder**, so on a machine with no hardware one
/// every use of these presets ends in `spawn_export`'s refusal — "H.265 export is not available (no
/// encoder registered)" — before a frame is rendered. Offering a preset whose every use is that error
/// would be a lie about what the preset does; **not offering it is the format list's own rule,
/// applied one step earlier.** On this machine the probe passes, the preset is listed, and it is the
/// same `Web & Social` entry the H.264 hardware presets are, so it is reachable from the same
/// Quick Export and Export-mode paths.
///
/// **What it does not do is soften anything for machines that cannot.** The probe asks "can this
/// machine encode HEVC at all", which is a weaker question than "can it encode *this* export"; a
/// sequence this preset cannot serve (a 23.976 rate) still fails, with the reason, rather than being
/// quietly dropped from the list.
fn hevc_presets() -> Vec<ExportPreset> {
    if !crate::available(Format::Hevc) {
        return Vec::new();
    }
    vec![p(
        "HEVC 1080p High Quality",
        "Web & Social",
        "H.265 1920×1080, VBR 1 pass, 10 Mbps target, AAC 320 kbps. Encoded by the machine's HEVC \
         encoder — offered only where one exists, and there is no built-in HEVC encoder to fall back \
         to. Needs a sequence at 30 fps: the hardware path states its frame rate in the stream, so \
         another rate is refused rather than rounded.",
        hevc((1920, 1080), 10_000, 15_000),
    )]
}

fn mov(format: Format, profile: &str) -> ExportSettings {
    let mut s = ExportSettings {
        format,
        audio: AudioSettings { codec: AudioCodec::Pcm, sample_rate: Some(48_000), channels: 2, bitrate_kbps: 320, bits: 24 },
        render_at_max_depth: true,
        ..Default::default()
    };
    match format {
        Format::ProRes => s.prores_profile = profile.into(),
        Format::Apv => s.apv_profile = profile.into(),
        _ => s.dnx_profile = profile.into(),
    }
    s
}

fn mxf(format: Format, codec: MxfVideoCodec, profile: &str) -> ExportSettings {
    let mut s = ExportSettings { mxf_video_codec: codec, ..mov(format, profile) };
    match codec {
        MxfVideoCodec::ProRes => s.prores_profile = profile.into(),
        MxfVideoCodec::Dnxhr => s.dnx_profile = profile.into(),
        MxfVideoCodec::H264 => {
            s.bitrate_kbps = 50_000;
            s.adaptive_bitrate = Some(0.3);
            s.bitrate_mode = BitrateMode::Vbr1Pass;
        }
    }
    s
}

fn audio_only(format: Format, bits: u16) -> ExportSettings {
    ExportSettings {
        format,
        audio: AudioSettings { codec: AudioCodec::Pcm, sample_rate: Some(48_000), channels: 2, bitrate_kbps: 0, bits },
        ..Default::default()
    }
}

fn p(name: &str, category: &str, description: &str, settings: ExportSettings) -> ExportPreset {
    ExportPreset { name: name.into(), category: category.into(), description: description.into(), settings, builtin: true }
}

/// Every built-in preset, in browser order.
///
/// # Which of these a hardware encoder takes, and which it declines by name
///
/// **Measured on this machine, not inferred.** `crates/platform/tests/vaenc.rs` reads this very list
/// and asks the H.264 gate (`vaapi_export::config_for`) about every preset whose
/// [`ExportSettings::video_format`] is [`Format::H264`], and `tests/hevenc.rs` asks the HEVC gate
/// (`vaapi::hevc_config_for`) the same question of every HEVC preset. The full table, with every
/// refusal quoted, is `docs/vaapi-progress.md` §3.10. The short form:
///
/// - **8 of the 11 presets that reach the H.264 gate are taken.** The three that are not are the two
///   `Vimeo` presets (**`BitrateMode::Vbr2Pass`**, refused with `two-pass encoding`) and
///   `MXF OP1a H.264` (**`MXF carries an Annex B stream, and this backend's samples are
///   length-prefixed`**). Each carries its own comment below saying the refusal is deliberate.
/// - **The HEVC preset is taken at 30 fps only**, for the encoder reason `hevc()` documents, and it is
///   *conditional on this machine having an HEVC encoder* — see `hevc_presets`.
/// - **The other nineteen presets are not declined by a gate at all, because no gate is asked.**
///   ProRes, DNxHR, APV, the MXF containers that carry them, the image sequences, GIF and the audio
///   formats have no VA-API encoder in FilmCraft. The registered factory returns `None` for them at
///   its routing line *before* either gate runs — `if !hevc && (format != Format::H264 ||
///   s.hardware_encoding != HardwareEncoding::Auto) { return None; }` — so there is no refusal to
///   quote and nothing to weaken. Asking the H.264 gate anyway would answer `ProRes is not H.264: the
///   H.264 gate takes H.264, and Format::Hevc goes to the HEVC gate`, and the HEVC gate `ProRes is
///   not HEVC`; those are routing messages, not capability findings, and the built-in encoders behind
///   those formats are the point.
///
/// **`MXF OP1a H.264` is the one non-H.264 *container* that does reach a gate**, because
/// [`ExportSettings::video_format`] reports the MXF *video codec* rather than the container: the
/// export code calls the factory with `video_format()` (`job.rs`), so that preset arrives as
/// `Format::H264` with `format` still `Format::MxfOp1a`, and the H.264 gate reads `s.format.is_mxf()`
/// and refuses. That is why its refusal below is the gate's and not the routing line's.
pub fn builtin_presets() -> Vec<ExportPreset> {
    let ms = "Match Source";
    let web = "Web & Social";
    let bc = "Broadcast";
    let vertical = ExportSettings { scaling: Scaling::ScaleToFill, ..h264(Some((1080, 1920)), 12_000, 16_000) };
    let mut out = vec![
        p(DEFAULT_PRESET, ms, "H.264 at the sequence's frame size and rate; bitrate adapts to the frame size (0.2 bits per pixel)", adaptive(0.2)),
        p("Match Source – Adaptive Medium Bitrate", ms, "H.264 at the sequence's frame size and rate; 0.1 bits per pixel", adaptive(0.1)),
        p("Match Source – Adaptive Low Bitrate", ms, "H.264 at the sequence's frame size and rate; 0.05 bits per pixel", adaptive(0.05)),
        p("High Quality 1080p HD", web, "H.264 1920×1080, VBR 1 pass, 20 Mbps target, AAC 320 kbps", h264(Some((1920, 1080)), 20_000, 24_000)),
        p("High Quality 2160p 4K", web, "H.264 3840×2160, VBR 1 pass, 60 Mbps target, AAC 320 kbps", h264(Some((3840, 2160)), 60_000, 72_000)),
        p("YouTube 1080p Full HD", web, "H.264 1920×1080 for video sharing sites, VBR 1 pass, 16 Mbps", h264(Some((1920, 1080)), 16_000, 20_000)),
        p("YouTube 2160p 4K Ultra HD", web, "H.264 3840×2160 for video sharing sites, VBR 1 pass, 45 Mbps", h264(Some((3840, 2160)), 45_000, 54_000)),
        p(
            "Vimeo 1080p Full HD",
            web,
            "H.264 1920×1080, VBR 2 pass, 18 Mbps",
            // **This preset is declined by the hardware encoders, on purpose.** Two-pass VBR renders
            // the sequence twice — once to measure it, once to encode it — and both hardware gates
            // refuse it before any bitrate is read: "two-pass encoding" in the VA-API H.264 gate
            // (`hardware_encode::vaapi_export::config_for`), and the same sentence with the reason
            // spelled out in the HEVC gate. The backend is handed each picture once and has no
            // `Pass` concept at all, so there is nothing to submit the measured rate control to:
            // encoding it anyway would be a file at a rate nobody chose, which is exactly what the
            // refusal is for.
            //
            // **So it exports, through FilmCraft's own encoder**, which takes the two passes and is
            // the only thing that does. `filmcraft_engine::export_tests` probes this file's `h264`
            // codec out of the MXF/H.264 formats and `h264_presets_hit_their_bitrates` checks this
            // preset's measured rate — the preset works; it just does not work through the GPU.
            ExportSettings { bitrate_mode: BitrateMode::Vbr2Pass, ..h264(Some((1920, 1080)), 18_000, 24_000) },
        ),
        p(
            "Vimeo 2160p 4K Ultra HD",
            web,
            "H.264 3840×2160, VBR 2 pass, 50 Mbps",
            // **Declined by the hardware encoders for the same reason as the 1080p Vimeo preset
            // above**, and for the same reason: two passes, one picture at a time, no `Pass`.
            ExportSettings { bitrate_mode: BitrateMode::Vbr2Pass, ..h264(Some((3840, 2160)), 50_000, 65_000) },
        ),
        p("Social Vertical 1080×1920", web, "H.264 9:16 portrait for short-form social video, scaled to fill, 12 Mbps", vertical),
    ];
    // **HEVC lands at the end of "Web & Social"** rather than at the end of the list, because the
    // browser groups by category and this preset belongs beside the delivery presets it sits next to.
    out.extend(hevc_presets());
    out.extend(vec![
        p("Apple ProRes 422 HQ", bc, "QuickTime, ProRes 422 HQ, 24-bit 48 kHz PCM", mov(Format::ProRes, "hq")),
        p("Apple ProRes 422", bc, "QuickTime, ProRes 422, 24-bit 48 kHz PCM", mov(Format::ProRes, "standard")),
        p("Apple ProRes 422 LT", bc, "QuickTime, ProRes 422 LT, 24-bit 48 kHz PCM", mov(Format::ProRes, "lt")),
        p("Apple ProRes 422 Proxy", bc, "QuickTime, ProRes 422 Proxy, 24-bit 48 kHz PCM", mov(Format::ProRes, "proxy")),
        p("Avid DNxHR HQ", bc, "QuickTime, DNxHR HQ (8-bit 4:2:2), 24-bit 48 kHz PCM", mov(Format::DnxHr, "hq")),
        p("Avid DNxHR SQ", bc, "QuickTime, DNxHR SQ (8-bit 4:2:2), 24-bit 48 kHz PCM", mov(Format::DnxHr, "sq")),
        p("Avid DNxHR LB", bc, "QuickTime, DNxHR LB (8-bit 4:2:2), 24-bit 48 kHz PCM", mov(Format::DnxHr, "lb")),
        p("APV 422-10", bc, "QuickTime, APV 422-10 (10-bit 4:2:2), 24-bit 48 kHz PCM", mov(Format::Apv, "422-10")),
        p("APV 422-12", bc, "QuickTime, APV 422-12 (12-bit 4:2:2), 24-bit 48 kHz PCM", mov(Format::Apv, "422-12")),
        p("MXF OP1a DNxHR HQ", bc, "MXF OP1a, DNxHR HQ (8-bit 4:2:2), 24-bit 48 kHz PCM, start timecode", mxf(Format::MxfOp1a, MxfVideoCodec::Dnxhr, "hq")),
        p("MXF OP1a ProRes 422 HQ", bc, "MXF OP1a, ProRes 422 HQ, 24-bit 48 kHz PCM, start timecode", mxf(Format::MxfOp1a, MxfVideoCodec::ProRes, "hq")),
        // **This preset is declined by the hardware encoder, on purpose.** MXF carries the H.264
        // **Annex B** byte stream with in-band parameter sets and access units (ST 381-3), which is
        // why the built-in encoder switches to `PacketFormat::AnnexB` and turns the AUDs on for
        // exactly this container. The VA-API path has no such switch — the samples it hands the muxer
        // are four-byte length-prefixed, which is the MP4 form — so the gate refuses by name: "MXF
        // carries an Annex B stream, and this backend's samples are length-prefixed".
        //
        // **That is a property of the container, not of the codec**, so no choice in this file makes
        // it export through the GPU: the only "fix" available here would be to stop shipping MXF
        // H.264, which is a real broadcast deliverable and works today through FilmCraft's own
        // encoder. The engine's preset test probes the `h264` codec inside an `mxf` container out of
        // this very preset, so it is not a preset that stopped working.
        p("MXF OP1a H.264", bc, "MXF OP1a, H.264 High long GOP (0.3 bits per pixel), 24-bit 48 kHz PCM", mxf(Format::MxfOp1a, MxfVideoCodec::H264, "")),
        p(
            "MXF OP-Atom DNxHR (Avid)",
            bc,
            "Avid-style OP-Atom: DNxHR HQ picture file plus one 24-bit 48 kHz PCM file per channel (_A1, _A2…)",
            mxf(Format::MxfOpAtom, MxfVideoCodec::Dnxhr, "hq"),
        ),
        p(
            "PNG Sequence",
            "Image Sequence",
            "Numbered PNG stills at the sequence's frame size",
            ExportSettings { format: Format::PngSequence, ..Default::default() },
        ),
        p("TIFF Sequence", "Image Sequence", "Numbered TIFF stills (RGB, 8-bit)", ExportSettings { format: Format::TiffSequence, ..Default::default() }),
        p("BMP Sequence", "Image Sequence", "Numbered BMP stills (24-bit)", ExportSettings { format: Format::BmpSequence, ..Default::default() }),
        p(
            "Animated GIF 640×360",
            "Animated GIF",
            "Looping GIF, 640×360 at 15 fps",
            ExportSettings { format: Format::Gif, frame_size: Some((640, 360)), frame_rate: Some(FrameRate::new(15, 1)), ..Default::default() },
        ),
        p("Waveform Audio 48 kHz 16-bit", "Audio Only", "WAV, stereo 48 kHz 16-bit PCM", audio_only(Format::Wav, 16)),
        p("Waveform Audio 48 kHz 24-bit", "Audio Only", "WAV, stereo 48 kHz 24-bit PCM", audio_only(Format::Wav, 24)),
        p("AIFF 48 kHz 16-bit", "Audio Only", "AIFF, stereo 48 kHz 16-bit PCM", audio_only(Format::Aiff, 16)),
    ]);
    out
}

/// A built-in preset by name ([`preset_key`] comparison).
pub fn find_builtin(name: &str) -> Option<ExportPreset> {
    let k = preset_key(name);
    builtin_presets().into_iter().find(|p| preset_key(&p.name) == k)
}

/// What `docs/vaapi-progress.md` §3.10 asserts, checked against the list that assertion is about.
///
/// **The table is a claim about this file, and nothing else in the tree pins it.** The hardware gates
/// ask their own question in `crates/platform/tests/` (which needs `filmcraft-platform`, a crate
/// above this one), so a preset added, renamed or re-routed here would change every row of that table
/// with no test in this crate failing. These four pin the partition it rests on: which presets reach
/// a gate at all, which name a pass count the gates refuse, which put H.264 in an MXF container, and
/// that the HEVC preset is absent unless a probe says this machine has an HEVC encoder.
///
/// **Pinned by name rather than by count**, because a count survives a rename or a re-routed
/// container and would then be an assertion that cannot fail.
#[cfg(test)]
mod reachability_tests {
    use super::{BitrateMode, Format, builtin_presets};

    /// Every preset whose [`crate::ExportSettings::video_format`] is H.264, in browser order: **the
    /// set the H.264 gate is asked about**, and the only presets whose hardware-export status is a
    /// gate's decision rather than the factory's routing.
    const AT_THE_H264_GATE: [&str; 11] = [
        super::DEFAULT_PRESET,
        "Match Source – Adaptive Medium Bitrate",
        "Match Source – Adaptive Low Bitrate",
        "High Quality 1080p HD",
        "High Quality 2160p 4K",
        "YouTube 1080p Full HD",
        "YouTube 2160p 4K Ultra HD",
        "Vimeo 1080p Full HD",
        "Vimeo 2160p 4K Ultra HD",
        "Social Vertical 1080×1920",
        "MXF OP1a H.264",
    ];

    fn names_matching(f: impl Fn(&crate::ExportPreset) -> bool) -> Vec<String> {
        builtin_presets().iter().filter(|p| f(p)).map(|p| p.name.clone()).collect()
    }

    #[test]
    fn the_presets_that_reach_the_h264_gate_are_the_ones_the_table_names() {
        let at_gate = names_matching(|p| p.settings.video_format() == Format::H264);
        assert_eq!(
            at_gate,
            AT_THE_H264_GATE.map(String::from).to_vec(),
            "a preset added, renamed or moved to another container changes which rows of docs/vaapi-progress.md \
             §3.10 are true; the HEVC preset is conditional and must not appear in this list"
        );
    }

    #[test]
    fn the_two_presets_that_name_a_second_pass_are_the_two_the_hardware_gates_refuse() {
        let two_pass = names_matching(|p| p.settings.bitrate_mode == BitrateMode::Vbr2Pass);
        assert_eq!(
            two_pass,
            ["Vimeo 1080p Full HD", "Vimeo 2160p 4K Ultra HD"].map(String::from).to_vec(),
            "both hardware gates refuse Vbr2Pass as 'two-pass encoding' before any bitrate is read, so a third \
             preset naming it is a third declined export and the table must say so"
        );
    }

    #[test]
    fn one_preset_puts_h264_in_an_mxf_container_and_the_gate_refuses_it_as_annex_b() {
        let mxf_h264 = names_matching(|p| p.settings.format.is_mxf() && p.settings.video_format() == Format::H264);
        assert_eq!(
            mxf_h264,
            ["MXF OP1a H.264"].map(String::from).to_vec(),
            "MXF carries an Annex B stream and the hardware backend's samples are length-prefixed, so this is \
             the one non-H.264 container that reaches a gate rather than being declined by the routing line"
        );
    }

    #[test]
    fn no_hevc_preset_is_offered_without_a_probe_saying_this_machine_has_an_encoder() {
        // **`filmcraft-platform` is not a dependency of this crate, so no probe can be registered
        // here** — which is exactly the state that assertion is about, and the state the platform
        // tests are in when they report "FilmCraft ships 0 HEVC preset(s)". If a probe ever *is*
        // registered in this test binary the assertion below fails rather than silently describing a
        // machine that has one.
        assert!(
            !crate::available(Format::Hevc),
            "a format probe for HEVC is registered in this test binary, so the preset list is conditional on a \
             machine state this test can no longer state: re-check the HEVC row of docs/vaapi-progress.md §3.10"
        );
        let hevc = names_matching(|p| p.settings.video_format() == Format::Hevc);
        assert!(
            hevc.is_empty(),
            "the HEVC preset is gated on crate::available(Format::Hevc), so with no probe it must not be listed, \
             but these are listed: {hevc:?}"
        );
    }
}
