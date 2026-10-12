//! Export settings beyond the basics: the Video / Audio / Multiplexer / Effects / Metadata
//! sections of Export mode, as plain serde data (every field has a default so older settings and
//! presets keep loading), plus the derived values (output size, rate, bitrate) and the estimated
//! file size.

use filmcraft_time::{FrameRate, Tick};
use serde::{Deserialize, Serialize};

use crate::{ExportSettings, Format};

/// How the sequence picture is fitted into a different output frame size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Scaling {
    /// Whole picture visible, black bars where the aspect ratios differ.
    #[default]
    ScaleToFit,
    /// Frame filled, the picture cropped where the aspect ratios differ.
    ScaleToFill,
    /// Frame filled, the picture distorted.
    StretchToFill,
}

/// Field order of the encoded video. FilmCraft's encoders write progressive frames.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FieldOrder {
    #[default]
    Progressive,
    UpperFirst,
    LowerFirst,
}

impl FieldOrder {
    pub fn label(self) -> &'static str {
        match self {
            FieldOrder::Progressive => "Progressive",
            FieldOrder::UpperFirst => "Upper First",
            FieldOrder::LowerFirst => "Lower First",
        }
    }
}

/// H.264 profile, or [`H264Profile::Auto`] — "whatever this machine can actually produce".
///
/// **`Auto` is a delegation, and it is the default**, which is what makes it the honest value for
/// a preset to inherit: the built-in presets name no profile at all, and a preset that names
/// nothing must not leave the file's profile to whatever the backend happens to do.
///
/// A named profile is a promise — *this file is a High profile* — and a promise the platform has to
/// be able to keep. On Linux it cannot always be kept, and that is measured, not suspected:
/// VA-API's derived parameter sets write `profile_idc = 77` (Main) because the sequence record
/// carries no profile field to derive from, and a driver offered `VA_PROFILE_H264_HIGH` answers
/// with a **byte-identical Main bitstream** (`entropy_coding_mode_flag = 0`, no CABAC, no 8x8).
/// Before [`Auto`] existed, the only ways to export H.264 through that backend were to pick Main by
/// hand or to be declined — and picking Main silently changed a preset that had asked for nothing,
/// which is the one outcome this enum exists to rule out.
///
/// The rule that replaces that guess is [`ExportSettings::resolved_h264_profile`], and the summary
/// prints what it resolved to ([`ExportSettings::summary`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum H264Profile {
    /// **What the encoder that will run on this machine can actually produce** for this export:
    /// see [`ExportSettings::resolved_h264_profile`]. The default, because a settings file that
    /// names no profile should not name one by accident.
    #[default]
    Auto,
    Baseline,
    Main,
    High,
}

impl H264Profile {
    /// Every profile, in the order a settings panel lists them: the delegation first, then the
    /// three named profiles from weakest to strongest.
    pub const ALL: [H264Profile; 4] = [H264Profile::Auto, H264Profile::Baseline, H264Profile::Main, H264Profile::High];

    pub fn label(self) -> &'static str {
        match self {
            H264Profile::Auto => "Auto",
            H264Profile::Baseline => "Baseline",
            H264Profile::Main => "Main",
            H264Profile::High => "High",
        }
    }
}

/// Export ▸ Hardware encoding: whether H.264 may be encoded by the system's hardware video encoder
/// (VideoToolbox on macOS, NVENC on NVIDIA GPUs on Windows) instead of FilmCraft's own encoder. Off
/// by default: hardware encoders make different streams, and exports are otherwise byte-identical
/// from run to run and machine to machine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum HardwareEncoding {
    #[default]
    Off,
    /// Use the hardware encoder when the system has one and it takes the settings; otherwise the
    /// software encoder.
    Auto,
}

/// **How the video encoder is asked to spend its bits** — a rate, or a fixed quantiser.
///
/// The name is historical: `Cbr` / `Vbr1Pass` / `Vbr2Pass` are all rates, and the mode that names no
/// rate at all arrived later. `ConstantQuality` is in this enum rather than beside it because it is
/// the same choice the other three make — *what the export promises about fidelity* — and a settings
/// file that carries both a rate and a quality has two answers to one question.
///
/// **It is not a Linux-only mode.** [`ExportSettings`] is one structure read by every backend, so a
/// mode added here is offered on macOS and Windows too, and each backend either takes it or declines
/// it by name. That is why [`Self::ALL`] exists: a backend's list of modes is a list of *its* modes,
/// and the settings surface lists all five.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BitrateMode {
    /// Constant bitrate.
    Cbr,
    /// Variable bitrate, one pass.
    #[default]
    Vbr1Pass,
    /// Variable bitrate, two passes (the picture is rendered and analysed first).
    Vbr2Pass,
    /// **Constant quality**: every picture is coded at the quantiser in
    /// [`ExportSettings::constant_qp`] and the file comes out at whatever size that takes. It states
    /// no rate at all, so `bitrate_kbps` and `max_bitrate_kbps` are not read and the size estimate
    /// has nothing to compute from — which is a fact about the mode, reported as one, rather than a
    /// number carried over from the last rate the user typed.
    ///
    /// [`ExportSettings::constant_quantiser`] is the checked way to read the quantiser, and every
    /// encoder that takes this mode refuses a value outside H.264's 0..=51 rather than clamping it.
    ConstantQuality,
    /// **Video conference mode** — `VA_RC_VCM`, `0x00000008`, `va.h:1114`:
    ///
    /// ```c
    /// /** \brief Variable bitrate. */
    /// #define VA_RC_VBR                       0x00000004
    /// /** \brief Video conference mode. */
    /// #define VA_RC_VCM                       0x00000008
    /// /** \brief Constant QP. */
    /// #define VA_RC_CQP                       0x00000010
    /// ```
    ///
    /// **It is here because the driver advertises it and FilmCraft had no way to name it**, which is
    /// the reason every other variant here is. A driver's rate-control attribute is a *set*, and
    /// this machine's H.265 encoder reports `0x14de` for it — eight modes, and exactly this bit is
    /// the difference from the `0x14d6` the same driver reports for H.264, so H.264's set has no
    /// name for it anywhere in FilmCraft and H.265's does
    /// (`crates/platform/src/vaapi/hevc_abi.rs:703`, a measurement of this driver, pinned). A
    /// capability nobody can spell is a capability that either never reaches a driver or reaches it
    /// unnamed.
    ///
    /// **What it states: a bitrate, and a cap on any one picture.** `va.h:3894` gives
    /// `VA_CODED_BUF_STATUS_BITRATE_HIGH` the note *"(VCM only)"* — a frame closing on its maximum
    /// size — beside `VA_CODED_BUF_STATUS_BITRATE_OVERFLOW` at `va.h:3900`, *"The peak bitrate was
    /// exceeded for this frame"*. So `bitrate_kbps` and `max_bitrate_kbps` are read by this mode,
    /// the size estimate has something to compute from, and the summary prints a rate for it.
    /// **It states no quantiser**: `constant_qp` means nothing here, and unlike
    /// [`BitrateMode::ConstantQuality`] this mode is not the case of a mode that names no rate.
    ///
    /// **It is one pass** — nothing about the mode asks for a second rendering of the sequence.
    ///
    /// **A note on the placement, for the next reader.** VCM is a *latency* concept filed among
    /// rates: what it buys is a bounded delay, pictures not being held back to hit an average, and
    /// that is a property of how the encoder spends time rather than of the bitrate schedule
    /// itself. It sits here because it answers the same question the others answer — what this
    /// export promises about fidelity — and a settings file carrying a latency choice beside a rate
    /// choice would have two answers to one question. `Realtime` and `LowLatency` were considered
    /// and rejected as the name for exactly the same reason: they describe what the mode *buys*
    /// rather than what it *is*, and would later read as a preset or as a render mode.
    VideoConference,
}

impl BitrateMode {
    /// Every mode, in the order a settings panel lists them.
    pub const ALL: [BitrateMode; 5] =
        [BitrateMode::Cbr, BitrateMode::Vbr1Pass, BitrateMode::Vbr2Pass, BitrateMode::ConstantQuality, BitrateMode::VideoConference];

    pub fn label(self) -> &'static str {
        match self {
            BitrateMode::Cbr => "CBR",
            BitrateMode::Vbr1Pass => "VBR, 1 pass",
            BitrateMode::Vbr2Pass => "VBR, 2 pass",
            BitrateMode::ConstantQuality => "Constant quality",
            BitrateMode::VideoConference => "Video conference (VCM)",
        }
    }

    /// **Whether this mode names a rate**, which is what `bitrate_kbps` and `max_bitrate_kbps` are
    /// for. `false` for [`BitrateMode::ConstantQuality`] alone.
    ///
    /// It exists so that the two statements about a file's size — the estimated bytes and the
    /// settings the estimate came from — are made by the same branch. A summary that printed
    /// "Target 20.00 Mbps" beside a constant-quality export would be quoting a number the export
    /// does not use.
    ///
    /// **The arms are spelled out rather than written as "everything but `ConstantQuality`".** The
    /// catch-all is the wrong default in this one place: a new mode would silently answer `true`,
    /// the summary would quote a bitrate it does not submit, and nothing would fail. Written this
    /// way, the sixth mode does not compile until someone has decided what it states.
    ///
    /// **[`BitrateMode::VideoConference`] answers `true`, and it is worth being explicit about why
    /// that is not the [`BitrateMode::ConstantQuality`] answer.** Both sound like "a mode that names
    /// no rate", and they are opposites. `ConstantQuality` states a quantiser and no rate, which is
    /// why both hardware gates refuse it rather than encoding it at a rate nobody chose.
    /// `VA_RC_VCM` states a rate *and* adds a per-picture ceiling to it — `va.h:3894` marks
    /// `VA_CODED_BUF_STATUS_BITRATE_HIGH` "(VCM only)" and `va.h:3900` describes the peak bitrate
    /// being exceeded — so it is driven by the very numbers this predicate asks about, and saying
    /// `false` here would print `QP {constant_qp}` beside an export that submits no quantiser at
    /// all. The reason to name it separately is that the two modes are refused or accepted for
    /// **opposite** reasons; they are not two flavours of the same gap.
    pub fn names_a_bitrate(self) -> bool {
        match self {
            BitrateMode::Cbr | BitrateMode::Vbr1Pass | BitrateMode::Vbr2Pass | BitrateMode::VideoConference => true,
            BitrateMode::ConstantQuality => false,
        }
    }

    /// Whether the mode needs the sequence rendered twice before it is encoded.
    ///
    /// Unlike [`Self::names_a_bitrate`] this one keeps its catch-all, and that is deliberate: a
    /// mode that does not ask to be named `Vbr2Pass` is single-pass, so landing in a new variant is
    /// the conservative answer rather than the aggressive one. VCM is among them — it changes what
    /// the encoder does with a picture, not how many times the sequence is rendered.
    pub fn is_two_pass(self) -> bool {
        matches!(self, BitrateMode::Vbr2Pass)
    }
}

/// Container of H.264 exports.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Multiplexer {
    #[default]
    Mp4,
    /// QuickTime.
    Mov,
}

/// Video codec of an MXF export.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MxfVideoCodec {
    /// Avid DNxHR (VC-3, SMPTE ST 2019-4 mapping).
    #[default]
    Dnxhr,
    /// Apple ProRes (RDD 44 mapping).
    ProRes,
    /// H.264 long GOP (ST 381-3 byte-stream mapping).
    H264,
}

impl MxfVideoCodec {
    pub const ALL: [MxfVideoCodec; 3] = [MxfVideoCodec::Dnxhr, MxfVideoCodec::ProRes, MxfVideoCodec::H264];
    pub fn label(self) -> &'static str {
        match self {
            MxfVideoCodec::Dnxhr => "Avid DNxHR",
            MxfVideoCodec::ProRes => "Apple ProRes",
            MxfVideoCodec::H264 => "H.264",
        }
    }
    /// The export format whose encoder this codec uses.
    pub fn encoder_format(self) -> Format {
        match self {
            MxfVideoCodec::Dnxhr => Format::DnxHr,
            MxfVideoCodec::ProRes => Format::ProRes,
            MxfVideoCodec::H264 => Format::H264,
        }
    }
}

/// Audio codec. `Auto` = AAC in MP4, PCM in QuickTime / WAV / AIFF.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AudioCodec {
    #[default]
    Auto,
    Aac,
    Pcm,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioSettings {
    pub codec: AudioCodec,
    /// Output sample rate (None = the sequence's).
    pub sample_rate: Option<u32>,
    /// 1 (mono), 2 (stereo) or 6 (5.1: L, R, C, LFE, Ls, Rs).
    pub channels: u32,
    /// AAC bitrate.
    pub bitrate_kbps: u32,
    /// PCM sample size: 16 or 24.
    pub bits: u16,
}

impl Default for AudioSettings {
    fn default() -> Self {
        AudioSettings { codec: AudioCodec::Auto, sample_rate: None, channels: 2, bitrate_kbps: 320, bits: 16 }
    }
}

/// Where an overlay sits in the frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Placement {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    #[default]
    BottomCenter,
    BottomRight,
}

impl Placement {
    pub const ALL: [Placement; 9] = [
        Placement::TopLeft,
        Placement::TopCenter,
        Placement::TopRight,
        Placement::CenterLeft,
        Placement::Center,
        Placement::CenterRight,
        Placement::BottomLeft,
        Placement::BottomCenter,
        Placement::BottomRight,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Placement::TopLeft => "Top Left",
            Placement::TopCenter => "Top Center",
            Placement::TopRight => "Top Right",
            Placement::CenterLeft => "Center Left",
            Placement::Center => "Center",
            Placement::CenterRight => "Center Right",
            Placement::BottomLeft => "Bottom Left",
            Placement::BottomCenter => "Bottom Center",
            Placement::BottomRight => "Bottom Right",
        }
    }
    /// (x, y) anchor fractions: 0 = left/top, 0.5 = centre, 1 = right/bottom.
    pub fn anchor(self) -> (f32, f32) {
        let i = Placement::ALL.iter().position(|p| *p == self).unwrap_or(7);
        ((i % 3) as f32 * 0.5, (i / 3) as f32 * 0.5)
    }
}

/// Export ▸ Effects ▸ Image Overlay.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageOverlay {
    pub enabled: bool,
    /// PNG / JPEG file.
    pub path: String,
    pub placement: Placement,
    /// Offset in output pixels.
    pub offset: (f32, f32),
    /// Width as a percentage of the output width (the image keeps its aspect ratio).
    pub size_percent: f32,
    /// 0–100.
    pub opacity: f32,
}

impl Default for ImageOverlay {
    fn default() -> Self {
        ImageOverlay { enabled: false, path: String::new(), placement: Placement::TopRight, offset: (0.0, 0.0), size_percent: 15.0, opacity: 100.0 }
    }
}

/// Export ▸ Effects ▸ Name Overlay / Timecode Overlay.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TextOverlay {
    pub enabled: bool,
    /// Name overlay: the text (prefix + suffix in one). Timecode overlay: an optional prefix.
    pub text: String,
    pub placement: Placement,
    pub offset: (f32, f32),
    /// Text height as a percentage of the output height.
    pub size_percent: f32,
    /// 0–100 (the text and its backing box).
    pub opacity: f32,
}

impl Default for TextOverlay {
    fn default() -> Self {
        TextOverlay { enabled: false, text: String::new(), placement: Placement::BottomCenter, offset: (0.0, 0.0), size_percent: 5.0, opacity: 100.0 }
    }
}

/// Export ▸ Effects ▸ Video Limiter: keeps luma inside a legal range.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VideoLimiter {
    pub enabled: bool,
    /// Luma floor and ceiling in percent of the video range (0 % = black, 100 % = white).
    pub min_percent: f32,
    pub max_percent: f32,
}

impl Default for VideoLimiter {
    fn default() -> Self {
        VideoLimiter { enabled: false, min_percent: 0.0, max_percent: 100.0 }
    }
}

/// Export ▸ Effects ▸ Loudness Normalization (ITU-R BS.1770-4 integrated loudness).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LoudnessNormalization {
    pub enabled: bool,
    /// Target integrated loudness (LUFS).
    pub target_lufs: f64,
    /// True-peak ceiling (dBTP) enforced by a look-ahead limiter.
    pub true_peak_dbtp: f64,
}

impl Default for LoudnessNormalization {
    fn default() -> Self {
        LoudnessNormalization { enabled: false, target_lufs: -23.0, true_peak_dbtp: -2.0 }
    }
}

/// Export ▸ Effects.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExportEffects {
    pub image_overlay: ImageOverlay,
    pub name_overlay: TextOverlay,
    pub timecode_overlay: TextOverlay,
    pub video_limiter: VideoLimiter,
    pub loudness: LoudnessNormalization,
}

impl ExportEffects {
    pub fn any_video(&self) -> bool {
        self.image_overlay.enabled || self.name_overlay.enabled || self.timecode_overlay.enabled || self.video_limiter.enabled
    }
}

/// Export ▸ Metadata (written to the MP4 / QuickTime `udta`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExportMetadata {
    pub title: String,
    pub creator: String,
    pub copyright: String,
    pub description: String,
    pub comment: String,
}

impl ExportMetadata {
    /// `udta` items (QuickTime `©xxx` keys) plus the encoder name.
    pub fn udta(&self) -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> =
            [("©nam", &self.title), ("©ART", &self.creator), ("©cpy", &self.copyright), ("©des", &self.description), ("©cmt", &self.comment)]
                .into_iter()
                .filter(|(_, s)| !s.trim().is_empty())
                .map(|(k, s)| (k.to_string(), s.trim().to_string()))
                .collect();
        v.push(("©too".into(), format!("FilmCraft {}", env!("CARGO_PKG_VERSION"))));
        v
    }
}

/// Values derived from settings + sequence at export time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Resolved {
    /// Output frame size (even for chroma-subsampled codecs).
    pub width: u32,
    pub height: u32,
    pub rate: FrameRate,
    pub sample_rate: u32,
    pub channels: u32,
    /// Video target / max bitrate (kbps) for bitrate-driven codecs.
    pub target_kbps: u32,
    pub max_kbps: u32,
    /// Keyframe distance (frames).
    pub keyint: u32,
}

impl ExportSettings {
    /// The format whose video encoder runs: the MXF video codec for MXF exports, else the format.
    pub fn video_format(&self) -> Format {
        if self.format.is_mxf() { self.mxf_video_codec.encoder_format() } else { self.format }
    }

    /// **The H.264 quantiser a [`BitrateMode::ConstantQuality`] export is encoded at**, or why the
    /// settings cannot state one.
    ///
    /// **Refused outside 0..=51, never clamped.** `constant_qp` arrives from a project file and a UI
    /// field, both untrusted, and a clamp would encode at the nearest quality instead of the one
    /// asked for — a wrong picture that looks right, which is the defect class this whole file exists
    /// to refuse (see [`crate::VideoEncoder`]'s contract).
    ///
    /// It is checked here rather than at each encoder because every backend that takes this mode
    /// wants the same number in the same units, and one checked accessor is the only way the
    /// software encoder, VideoToolbox, NVENC and VA-API cannot disagree about what "QP 40" means.
    pub fn constant_quantiser(&self) -> crate::Result<u8> {
        let qp = self.constant_qp;
        if qp > 51 {
            return Err(crate::ExportError::Unsupported(format!("a quantiser of {qp}: H.264 numbers them 0 (best) to 51 (worst)")));
        }
        Ok(qp)
    }

    /// Whether the output is a numbered image sequence.
    pub fn is_image_sequence(&self) -> bool {
        matches!(self.format, Format::PngSequence | Format::TiffSequence | Format::BmpSequence)
    }

    /// Output size, rate, audio format and bitrate for a sequence of `seq_w`×`seq_h` at `seq_rate`.
    pub fn resolve(&self, seq_w: u32, seq_h: u32, seq_rate: FrameRate, seq_sr: u32) -> Resolved {
        let (mut w, mut h) = match self.frame_size {
            Some((w, h)) => (w.max(2), h.max(2)),
            None => (((seq_w as f32 * self.scale).round() as u32).max(2), ((seq_h as f32 * self.scale).round() as u32).max(2)),
        };
        if !self.is_image_sequence() && self.format != Format::Gif {
            w &= !1;
            h &= !1;
        }
        let rate = self.frame_rate.filter(|r| r.num > 0 && r.den > 0).unwrap_or(seq_rate);
        let fps = rate.num as f64 / rate.den as f64;
        let target = match self.adaptive_bitrate {
            Some(bpp) if bpp > 0.0 => ((w as f64 * h as f64 * fps * bpp as f64) / 1000.0).round() as u32,
            _ => self.bitrate_kbps,
        }
        .max(100);
        // Only `Cbr` pins the rate exactly; every other mode gets a ceiling above the target, which
        // is what `VideoConference` wants too — it caps a rate rather than fixing one. Written as a
        // catch-all because the default (a variable rate with a headroom ceiling) is the right one
        // for a new mode, unlike `names_a_bitrate`'s.
        let max = match self.bitrate_mode {
            BitrateMode::Cbr => target,
            _ => self.max_bitrate_kbps.filter(|m| *m >= target).unwrap_or(target * 3 / 2),
        };
        let keyint = self.keyframe_distance.filter(|k| *k > 0).unwrap_or_else(|| (fps * 2.0).round().max(1.0) as u32);
        let channels = match self.audio.channels {
            0 | 1 => 1,
            6.. => 6,
            _ => 2,
        };
        Resolved {
            width: w,
            height: h,
            rate,
            sample_rate: self.audio.sample_rate.filter(|r| (8000..=192_000).contains(r)).unwrap_or(seq_sr),
            channels,
            target_kbps: target,
            max_kbps: max,
            keyint,
        }
    }

    /// **The H.264 profile this export is actually encoded at**, which is `self.h264_profile`
    /// unless that is [`H264Profile::Auto`].
    ///
    /// # The rule, and why it lives here rather than in `crates/platform`
    ///
    /// **`Auto` asks what the *backend* can produce, not what the user typed**, and the answer
    /// depends on which encoder the export will actually reach:
    ///
    /// - **The built-in encoder will run** — hardware encoding off (the default), or the OS encoder
    ///   declines this export outright — and it writes any of the three profiles, so `Auto` is
    ///   [`H264Profile::High`], the strongest. Nothing is lost by asking.
    /// - **The OS encoder may take it** — hardware encoding on, and a container its samples fit —
    ///   so `Auto` is the best profile *every* encoder that could run here can produce:
    ///   - **macOS (VideoToolbox) and Windows (NVENC)** pass the profile through to the driver and
    ///     write it into the stream, so `Auto` is [`H264Profile::High`] — unchanged from before
    ///     this variant existed.
    ///   - **Linux (VA-API)** writes `profile_idc = 77` in its derived SPS whatever profile it is
    ///     offered, and answers an offered High with a byte-identical Main bitstream, so `Auto` is
    ///     [`H264Profile::Main`]. That costs quality on a machine that could have had High, and it
    ///     is paid on purpose: a preset that names nothing must not hand the driver a profile the
    ///     file will not carry, and must not have to be edited to work on a third platform.
    ///
    /// **Every encoder is given *this* value, not the one that was stored** — the built-in encoder
    /// (`filmcraft_export::h264_factory`), the MXF picture record and the OS backends all resolve
    /// it. That is what keeps the invariant: *whatever ends up encoding, and whatever the hardware
    /// path declines to take, the summary names the profile in the file* — the one thing an `Auto`
    /// has to be if it is to be worth having.
    ///
    /// **A user who wants High on Linux says so, and gets it.** A named [`H264Profile::High`] is
    /// passed through untouched: the VA-API gate declines it by name
    /// (`hardware_encode::vaapi_config_for`), the export falls back to the built-in encoder, and
    /// both the summary and the file say High. `Auto` gives that up for a preset that needs no
    /// editing anywhere; it never substitutes for asking.
    ///
    /// **It is here, and not in the platform crate**, because the summary is computed from the
    /// settings — before an encoder is chosen, and by a crate that cannot depend on
    /// `crates/platform`. A rule the summary could not call would be a rule no user ever sees.
    pub fn resolved_h264_profile(&self) -> H264Profile {
        if self.h264_profile != H264Profile::Auto {
            return self.h264_profile;
        }
        // The three conditions are the VA-API gate's own, one for one: hardware encoding on (it is
        // the only encoder on this platform that may take the export), a container it can carry
        // (MXF is an Annex B stream, which this backend's length-prefixed samples are not — so the
        // gate declines it by name whatever the profile, and the built-in encoder will run), and
        // the platform whose encoder writes Main.
        if self.hardware_encoding == HardwareEncoding::Auto && cfg!(target_os = "linux") && !self.format.is_mxf() {
            H264Profile::Main
        } else {
            H264Profile::High
        }
    }

    /// The audio codec actually used.
    pub fn audio_codec(&self) -> AudioCodec {
        match (self.format, self.audio.codec) {
            (Format::Wav | Format::Aiff | Format::MxfOp1a | Format::MxfOpAtom, _) => AudioCodec::Pcm,
            (f, _) if f.is_h26x() && self.multiplexer == Multiplexer::Mp4 => AudioCodec::Aac,
            (_, AudioCodec::Auto) if self.format.is_h26x() => AudioCodec::Aac,
            (_, AudioCodec::Auto) => AudioCodec::Pcm,
            (_, c) => c,
        }
    }

    /// Whether the format carries audio at all.
    pub fn has_audio(&self) -> bool {
        self.include_audio && !self.is_image_sequence() && self.format != Format::Gif
    }

    /// Whether the format carries video.
    pub fn has_video(&self) -> bool {
        !matches!(self.format, Format::Wav | Format::Aiff)
    }

    /// File extension of the output.
    pub fn extension(&self) -> &'static str {
        if self.format.is_h26x() && self.multiplexer == Multiplexer::Mov { "mov" } else { self.format.extension() }
    }

    /// Estimated output size in bytes for `duration` of a sequence (`seq_w`×`seq_h` at `seq_rate`).
    ///
    /// **Zero for a constant-quality H.264 or HEVC export, and that is an answer rather than a
    /// gap.** The size of a constant-quantiser file is whatever the pictures quantise to, which is
    /// the one number in this function that cannot be computed before the pictures are encoded —
    /// carrying the last typed bitrate over would print a figure for a rate the export never uses.
    /// [`Summary::estimated_size`] says so in words where a number would otherwise be.
    pub fn estimate_bytes(&self, seq_w: u32, seq_h: u32, seq_rate: FrameRate, seq_sr: u32, duration: Tick) -> u64 {
        let r = self.resolve(seq_w, seq_h, seq_rate, seq_sr);
        let secs = duration.seconds().max(0.0);
        let fps = r.rate.num as f64 / r.rate.den as f64;
        let px = r.width as f64 * r.height as f64;
        let video_bps = match self.video_format() {
            Format::H264 | Format::Hevc if !self.bitrate_mode.names_a_bitrate() => 0.0,
            Format::H264 | Format::Hevc => r.target_kbps as f64 * 1000.0,
            Format::ProRes => crate::prores_profile(&self.prores_profile).nominal_mbps_1080p30() * 1e6 * px / (1920.0 * 1080.0) * fps / 29.97,
            Format::DnxHr => {
                // nominal 1080p29.97 data rates of the DNxHR profiles (Mb/s)
                let mbps = match self.dnx_profile.to_ascii_lowercase().as_str() {
                    "lb" => 45.0,
                    "sq" => 145.0,
                    "hqx" => 220.0,
                    _ => 220.0,
                };
                mbps * 1e6 * px / (1920.0 * 1080.0) * fps / 29.97
            }
            Format::Apv => {
                let bpp = match crate::apv_profile(&self.apv_profile) {
                    filmcraft_apv::Profile::P422_12 => 3.6,
                    filmcraft_apv::Profile::P444_10 => 4.2,
                    filmcraft_apv::Profile::P444_12 | filmcraft_apv::Profile::P4444_10 | filmcraft_apv::Profile::P4444_12 => 5.0,
                    _ => 3.0,
                };
                px * fps * bpp
            }
            Format::Mjpeg => px * 8.0 * (0.4 + self.quality as f64 / 100.0 * 2.0) * fps / 8.0,
            Format::PngSequence => px * 4.0 * 0.45 * 8.0 * fps,
            Format::TiffSequence | Format::BmpSequence => px * if self.format == Format::BmpSequence { 3.0 } else { 4.0 } * 8.0 * fps,
            Format::Gif => px * 0.6 * 8.0 * fps,
            Format::Wav | Format::Aiff | Format::MxfOp1a | Format::MxfOpAtom => 0.0,
        };
        let audio_bps = if self.has_audio() || matches!(self.format, Format::Wav | Format::Aiff) {
            match self.audio_codec() {
                AudioCodec::Aac => self.audio.bitrate_kbps as f64 * 1000.0,
                _ => r.sample_rate as f64 * r.channels as f64 * if self.audio.bits >= 24 { 24.0 } else { 16.0 },
            }
        } else {
            0.0
        };
        ((video_bps + audio_bps) * secs / 8.0).round() as u64
    }
}

/// The Export-mode Summary panel: one line each for the output video, audio and format.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Summary {
    pub format: String,
    pub video: String,
    pub audio: String,
    pub estimated_bytes: u64,
    pub estimated_size: String,
}

impl ExportSettings {
    /// Summary lines for a sequence of `seq_w`×`seq_h` at `seq_rate` / `seq_sr` exporting `duration`.
    pub fn summary(&self, seq_w: u32, seq_h: u32, seq_rate: FrameRate, seq_sr: u32, duration: Tick) -> Summary {
        let r = self.resolve(seq_w, seq_h, seq_rate, seq_sr);
        let mbps = |k: u32| format!("{:.2} Mbps", k as f64 / 1000.0);
        let video = if !self.has_video() {
            "No video".to_string()
        } else {
            let par = self.pixel_aspect.map(|(n, d)| format!("{:.2}", n.max(1) as f64 / d.max(1) as f64)).unwrap_or_else(|| "1.0".into());
            let mut v = format!("{}x{} ({par}), {} fps, {}", r.width, r.height, r.rate.label(), self.field_order.label());
            match self.video_format() {
                Format::H264 => {
                    // **The resolved profile, not the one that was stored**: `H264Profile::Auto`
                    // means "what this machine can produce", and a summary that printed "Auto"
                    // would describe the setting rather than the file. The same mechanism
                    // `audio_codec()` gives `AudioCodec::Auto` a few lines below, and the same
                    // reason it exists: the user has to be able to read off the panel what the
                    // export will write.
                    v += &format!(
                        ", {} {}, {}",
                        self.resolved_h264_profile().label(),
                        self.h264_level.map(|l| format!("L{}.{}", l / 10, l % 10)).unwrap_or_else(|| "auto level".into()),
                        self.bitrate_mode.label()
                    );
                    if self.bitrate_mode.names_a_bitrate() {
                        v += &match self.bitrate_mode {
                            BitrateMode::Cbr => format!(", {}", mbps(r.target_kbps)),
                            _ => format!(", Target {}, Max {}", mbps(r.target_kbps), mbps(r.max_kbps)),
                        };
                    } else {
                        // The quantiser, and the fact that it is what makes the size unknowable here.
                        // `ConstantQuality` is the only mode that gets here — `names_a_bitrate` is an
                        // exhaustive match, so `VideoConference` cannot reach a line that would quote
                        // `constant_qp` for a mode that states no quantiser.
                        v += &format!(", QP {}", self.constant_qp);
                    }
                    v += &format!(", keyframe every {} frames", r.keyint);
                }
                Format::Hevc => {
                    v += &format!(", HEVC Main (hardware encoder), {}", self.bitrate_mode.label());
                    if self.bitrate_mode.names_a_bitrate() {
                        v += &match self.bitrate_mode {
                            BitrateMode::Cbr => format!(", {}", mbps(r.target_kbps)),
                            _ => format!(", Target {}, Max {}", mbps(r.target_kbps), mbps(r.max_kbps)),
                        };
                    } else {
                        v += &format!(", QP {}", self.constant_qp);
                    }
                    v += &format!(", keyframe every {} frames", r.keyint);
                }
                Format::ProRes => {
                    use filmcraft_prores::Profile;
                    v += match crate::prores_profile(&self.prores_profile) {
                        Profile::Proxy => ", ProRes 422 Proxy",
                        Profile::Lt => ", ProRes 422 LT",
                        Profile::Standard => ", ProRes 422",
                        _ => ", ProRes 422 HQ",
                    }
                }
                Format::DnxHr => v += &format!(", DNxHR {}", if self.dnx_profile.is_empty() { "HQ".into() } else { self.dnx_profile.to_ascii_uppercase() }),
                Format::Apv => v += &format!(", {}", crate::apv_profile(&self.apv_profile).name()),
                Format::Mjpeg => v += &format!(", quality {}", self.quality),
                _ => {}
            }
            v
        };
        let audio = if self.has_audio() || matches!(self.format, Format::Wav | Format::Aiff) {
            let ch = match r.channels {
                1 => "Mono",
                6 => "5.1",
                _ => "Stereo",
            };
            match self.audio_codec() {
                AudioCodec::Aac => format!("AAC, {} kbps, {} Hz, {ch}", self.audio.bitrate_kbps, r.sample_rate),
                _ => format!("Uncompressed {}-bit PCM, {} Hz, {ch}", if self.audio.bits >= 24 { 24 } else { 16 }, r.sample_rate),
            }
        } else {
            "No audio".to_string()
        };
        let container = match self.format {
            Format::H264 | Format::Hevc if self.multiplexer == Multiplexer::Mov => "QuickTime",
            Format::H264 | Format::Hevc => "MP4",
            Format::ProRes | Format::DnxHr | Format::Apv | Format::Mjpeg => "QuickTime",
            Format::MxfOp1a | Format::MxfOpAtom => self.mxf_video_codec.label(),
            Format::PngSequence | Format::TiffSequence | Format::BmpSequence => "Image sequence",
            _ => "",
        };
        let format = if container.is_empty() { self.format.label().to_string() } else { format!("{} ({container})", self.format.label()) };
        let estimated_bytes = self.estimate_bytes(seq_w, seq_h, seq_rate, seq_sr, duration);
        // **A constant-quality H.264 / HEVC export has no size estimate, and says so in words.**
        // `estimated_bytes` is zero there, which `format_bytes` would happily round up to "1 KB" —
        // a number for a file whose size is the one thing the settings did not state.
        let estimated_size = if estimated_bytes == 0 && !self.bitrate_mode.names_a_bitrate() && matches!(self.video_format(), Format::H264 | Format::Hevc) {
            "not known in constant quality".to_string()
        } else {
            format_bytes(estimated_bytes)
        };
        Summary { format, video, audio, estimated_bytes, estimated_size }
    }
}

/// A time left, rounded up to the second: `45 s`, `2:05`, `1:02:05`.
pub fn format_eta(d: std::time::Duration) -> String {
    let s = d.as_secs().saturating_add(u64::from(d.subsec_nanos() > 0));
    match s {
        0..=59 => format!("{s} s"),
        60..=3599 => format!("{}:{:02}", s / 60, s % 60),
        _ => format!("{}:{:02}:{:02}", s / 3600, s % 3600 / 60, s % 60),
    }
}

/// `1.2 GB`, `350 MB`, `12 KB`.
pub fn format_bytes(b: u64) -> String {
    let b = b as f64;
    if b >= 1e9 {
        format!("{:.1} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.0} MB", b / 1e6)
    } else {
        format!("{:.0} KB", (b / 1e3).max(1.0))
    }
}
