//! The hardware encoders behind `filmcraft_export::VideoEncoder`: VideoToolbox's H.264 and HEVC on
//! macOS (safe code around [`crate::videotoolbox_encode`]) and VA-API's H.264 slice encoder on Linux
//! (safe code around [`crate::vaapi`]).
//!
//! H.264 hardware encoding is opt-in per export (`ExportSettings::hardware_encoding`, `Auto`): its output
//! depends on the machine, while the built-in encoder's is byte-identical everywhere. The factory
//! declines, so the built-in encoder is used, for everything the hardware path does not take: other
//! formats, MXF (Annex B), two-pass VBR, HDR, odd picture sizes, non-square pixels, and any
//! configuration the hardware encoder cannot create a session for. A hardware encoder that fails in
//! the middle of an export reports an error (the export stops); it cannot hand the stream over to
//! another encoder.
//!
//! HEVC (`Format::Hevc`) has no built-in encoder, so choosing the format is the opt-in, the same
//! configurations are declined, and a declined export fails with "encoder not available" instead of
//! falling back; [`hevc_available`] tells the format list whether this machine can do it at all.
//!
//! **Linux has an HEVC encoder too, and a `Format::Hevc` export here is not declined.** This used to
//! say the opposite, and it was true while it said it: [`crate::vaapi::VaEncoder`] declared the H.264
//! encode records and nothing else, so a declined HEVC export failed with "encoder not available"
//! rather than writing a file whose container described an HEVC stream and whose samples were H.264.
//! **The second reason is now the first**: [`crate::vaapi::HevcEncoder`] encodes HEVC, and the two
//! gates — [`vaapi_export::config_for`] for H.264 and [`crate::vaapi::hevc_config_for`] for HEVC —
//! decline different things, each for a reason its own records can state.
//!
//! **AV1 is the third case, and it is the one a format list gets wrong most easily.** A VA-API driver
//! can advertise `VAEntrypointEncSlice` for `VAProfileAV1Profile0`, accept `vaCreateConfig`,
//! `vaCreateSurfaces`, `vaCreateContext` and `vaRenderPicture`, and still write no AV1 byte at all —
//! which is what this machine's does, measured ([`crate::vaapi::driver_outcomes`]). So
//! "the driver advertises an AV1 encoder" is not a question with an answer, because two different
//! drivers give the same advertisement and one of them cannot encode. [`vaapi_export::av1_available`]
//! asks the question that does have an answer — **one key frame, rendered, synced, and mapped back
//! to a coded buffer with bytes in it** — and it is **not registered**, because
//! [`filmcraft_export::Format`] has no `Av1` variant to register it against. That is deliberate and
//! it is the whole of AV1's status in the format list today: nothing, with the probe that would
//! decide it already written and measured.
//!
//! # The two backends do not accept the same exports, and the differences are measured
//!
//! [`config_for`] (VideoToolbox) and [`vaapi_config_for`] (VA-API) share most of their decline list
//! and differ in two places, each for a reason the backend can state rather than a preference:
//!
//! - **Both take a bitrate.** VideoToolbox through `VTCompressionPropertyKey_AverageBitRate`, and
//!   VA-API through a `VA_RC_CBR` / `VA_RC_VBR` session with a `VAEncMiscParameterRateControl`. The
//!   difference is in the modes they can express: VideoToolbox has no constant-quantiser entry point
//!   here, while VA-API's encoder can also run constant quantiser — which neither gate asks for.
//!   **The mode that states one is declined by both, with the same words**, because it names no rate
//!   and every record either backend builds is a rate: mapping a quantiser the settings state onto
//!   `VTCompressionPropertyKey_Quality` or onto `VAPictureH264::pic_init_qp` is a piece of work in
//!   its own right, and until it is done the honest answer to a mode this gate cannot express is a
//!   decline rather than a substitution. See [`vaapi_config_for`] for what each gate accepts and why.
//! - **VA-API is one slice per picture and one reference deep**, both measured; the VideoToolbox path
//!   has no such limit. The slice limit is not expressible in the settings and so is not a decline
//!   here — it is a property of the encoder, stated in [`vaapi::VaEncoder`]'s own documentation.

#[cfg(target_os = "macos")]
use std::sync::OnceLock;

#[cfg(target_os = "macos")]
use filmcraft_export::{BitrateMode, H264Pass, H264Profile};
#[cfg(target_os = "macos")]
use filmcraft_export::{EncodedPacket, EncoderFrame, ExportError, ExportSettings, Format, HardwareEncoding, VideoEncoder};
#[cfg(target_os = "macos")]
use filmcraft_isobmff::{AvcConfig, HevcConfig, SampleEntry};
#[cfg(target_os = "macos")]
use filmcraft_time::FrameRate;

#[cfg(target_os = "macos")]
use crate::videotoolbox_encode::{ParamSets, VtCodec, VtConfig, VtEncoder, VtPacket, VtProfile, VtRate};

/// The hardware encoder's configuration for an export, or why it is not used for it.
#[cfg(target_os = "macos")]
pub fn config_for(w: u32, h: u32, rate: FrameRate, s: &ExportSettings) -> Result<VtConfig, String> {
    let hevc = s.format == Format::Hevc;
    if !hevc && s.hardware_encoding != HardwareEncoding::Auto {
        return Err("hardware encoding is off".into());
    }
    if s.format.is_mxf() {
        return Err("MXF carries an Annex B stream".into());
    }
    if s.bitrate_mode == BitrateMode::Vbr2Pass || !matches!(s.h264_pass, H264Pass::Single) {
        return Err("two-pass encoding".into());
    }
    // **A constant-quality export is refused rather than given a bitrate**, with the same words and
    // for the same reason as the VA-API gates. `BitrateMode::ConstantQuality` states a quantiser and
    // no rate, both [`VtRate`] variants are rates, and the `_ =>` arm below once turned this mode
    // into a VBR and dropped the quantiser — a file at a rate nobody chose beside a summary reading
    // `QP n`. This gate covers HEVC as well as H.264 (`hevc` above), so on this side of the refusal
    // the consequence differs by format and is worth stating: H.264 has the built-in encoder behind
    // it and the export falls back to it at the quantiser asked for, while **HEVC has no built-in
    // encoder, so a constant-quality HEVC export becomes "encoder not available" rather than a wrong
    // file** — the same trade the Linux HEVC gate already made. The refusal is not a claim about the
    // API: `VTCompressionPropertyKey_Quality` can be set, but nothing here maps FilmCraft's H.264
    // quantiser onto it, and a mapping this gate invented would be a table sitting next to a rate the
    // user did state.
    if s.bitrate_mode == BitrateMode::ConstantQuality {
        return Err(format!(
            "a constant-quality export: it states a quantiser ({}) and no bitrate, and this gate submits the bitrate. \
             FilmCraft's software encoders take that mode; this backend is declined for it rather than encoded at a rate nobody chose",
            s.constant_qp
        ));
    }
    if s.signal.is_hdr() {
        return Err("HDR".into());
    }
    if !w.is_multiple_of(2) || !h.is_multiple_of(2) {
        return Err("odd picture size".into());
    }
    if s.pixel_aspect.is_some_and(|(n, d)| n != d) {
        return Err("non-square pixels".into());
    }
    let timescale = u32::try_from(rate.num).ok().filter(|n| *n > 0).ok_or("invalid frame rate")?;
    let frame_duration = u32::try_from(rate.den).ok().filter(|d| *d > 0).ok_or("invalid frame rate")?;
    let kbps = s.bitrate_kbps.max(100);
    let max_kbps = s.max_bitrate_kbps.filter(|m| *m >= kbps).unwrap_or(kbps.saturating_mul(3) / 2);
    let fps = f64::from(timescale) / f64::from(frame_duration);
    Ok(VtConfig {
        width: w,
        height: h,
        timescale,
        frame_duration,
        keyframe_interval: s.keyframe_distance.filter(|k| *k > 0).unwrap_or_else(|| (fps * 2.0).round().max(1.0) as u32),
        profile: match (hevc, s.resolved_h264_profile()) {
            (true, _) => VtProfile::HevcMain,
            (false, H264Profile::Baseline) => VtProfile::Baseline,
            (false, H264Profile::Main) => VtProfile::Main,
            // **`Auto` is High here, and it is High for a reason rather than by default of the
            // match**: VideoToolbox takes the profile and writes it into the stream it returns, so
            // the strongest profile is the one this machine can actually produce — and on macOS
            // that is what `resolved_h264_profile` resolves `Auto` to as well.
            (false, H264Profile::Auto) | (false, H264Profile::High) => VtProfile::High,
        },
        rate: match s.bitrate_mode {
            BitrateMode::Cbr => VtRate::Cbr { kbps },
            _ => VtRate::Vbr { target_kbps: kbps, max_kbps },
        },
    })
}

/// Whether this machine has a hardware HEVC encoder (asked once, by creating a small session): what
/// makes `Format::Hevc` available.
#[cfg(target_os = "macos")]
pub fn hevc_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let probe = VtConfig {
            width: 640,
            height: 360,
            timescale: 30,
            frame_duration: 1,
            keyframe_interval: 30,
            profile: VtProfile::HevcMain,
            rate: VtRate::Vbr { target_kbps: 2000, max_kbps: 3000 },
        };
        let result = VtEncoder::new(probe);
        if let Err(why) = &result {
            log::info!("no hardware HEVC encoder: {why}");
        }
        result.is_ok()
    })
}

/// The encoder factory `register` puts in front of the built-in ones: a hardware encoder when the
/// export asks for one (H.264 with `hardware_encoding`; HEVC always) and the OS takes the
/// configuration, `None` (the built-in encoder, if there is one) otherwise.
#[cfg(target_os = "macos")]
pub fn videotoolbox_encoder_factory(
    format: Format,
    w: u32,
    h: u32,
    rate: FrameRate,
    s: &ExportSettings,
) -> Option<filmcraft_export::Result<Box<dyn VideoEncoder>>> {
    let name = match format {
        Format::H264 if s.hardware_encoding == HardwareEncoding::Auto => "H.264",
        Format::Hevc => "HEVC",
        _ => return None,
    };
    let config = match config_for(w, h, rate, s) {
        Ok(c) => c,
        Err(why) => {
            log::info!("hardware {name} encoding declined: {why}");
            return None;
        }
    };
    match VtEncoder::new(config) {
        Ok(vt) => Some(Ok(Box::new(HardwareEncoder { vt, y: Vec::new(), u: Vec::new(), v: Vec::new(), decoded: 0 }))),
        Err(why) => {
            log::info!("hardware {name} encoding declined: {why}");
            None
        }
    }
}

/// The `hvcC` configuration of an HEVC stream: VideoToolbox's own record when it wrote one.
#[cfg(target_os = "macos")]
fn hevc_config(p: &ParamSets) -> Result<HevcConfig, String> {
    let record = p.hvcc.as_deref().ok_or("VideoToolbox wrote no hvcC record for the stream")?;
    let cfg = HevcConfig::parse(record).map_err(|e| format!("unreadable hvcC record: {e}"))?;
    if cfg.vps().is_empty() || cfg.sps().is_empty() || cfg.pps().is_empty() || cfg.length_size != 4 {
        return Err("the hvcC record lacks a parameter set or does not use 4-byte NAL lengths".into());
    }
    Ok(cfg)
}

/// [`VtEncoder`] as a `VideoEncoder`: RGBA pictures in, MP4 samples out.
#[cfg(target_os = "macos")]
pub struct HardwareEncoder {
    vt: VtEncoder,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    /// Compressed frames returned so far.
    decoded: i64,
}

#[cfg(target_os = "macos")]
impl HardwareEncoder {
    /// MP4 samples from compressed frames. Without frame reordering every frame is presented at
    /// its decode time, so composition offsets are 0; a frame whose time is not `n × duration`
    /// after the first (a dropped or reordered frame) is an error, never a file with wrong times.
    fn samples(&mut self, packets: Vec<VtPacket>) -> filmcraft_export::Result<Vec<EncodedPacket>> {
        let duration = self.vt.config().frame_duration;
        let mut out = Vec::with_capacity(packets.len());
        for p in packets {
            let n = self.decoded;
            self.decoded += 1;
            if n.checked_mul(i64::from(duration)) != Some(p.pts) || p.pts != p.dts {
                return Err(ExportError::Encode(format!(
                    "VideoToolbox returned frame {n} at time {} (decode time {}), expected {}",
                    p.pts,
                    p.dts,
                    n.saturating_mul(i64::from(duration))
                )));
            }
            out.push(EncodedPacket { data: p.data, key: p.key, duration, composition_offset: 0 });
        }
        Ok(out)
    }
}

#[cfg(target_os = "macos")]
impl VideoEncoder for HardwareEncoder {
    fn sample_entry(&self) -> SampleEntry {
        let c = self.vt.config();
        let (w, h) = (c.width.min(u32::from(u16::MAX)) as u16, c.height.min(u32::from(u16::MAX)) as u16);
        match (c.profile.codec(), self.vt.parameter_sets()) {
            (VtCodec::H264, p) => {
                let (sps, pps) = p.map(|p| (p.sps, p.pps)).unwrap_or_default();
                SampleEntry::avc(AvcConfig::new(sps, pps, 4), w, h)
            }
            // `encode` refuses to go on without a usable record, so a stream that got this far has one
            (VtCodec::Hevc, p) => SampleEntry::hevc(p.as_ref().and_then(|p| hevc_config(p).ok()).unwrap_or_default(), w, h),
        }
    }

    fn timescale(&self) -> u32 {
        self.vt.config().timescale
    }

    fn encode(&mut self, f: &EncoderFrame) -> filmcraft_export::Result<Vec<EncodedPacket>> {
        let c = self.vt.config();
        let codec = c.profile.codec();
        if f.hdr.is_some() {
            return Err(ExportError::Unsupported("hardware encoding does not take HDR pictures".into()));
        }
        if (f.width, f.height) != (c.width, c.height) {
            return Err(ExportError::Encode(format!("picture is {}x{}, the encoder was created for {}x{}", f.width, f.height, c.width, c.height)));
        }
        let (w, h) = (f.width as usize, f.height as usize);
        if f.rgba.len() < w.saturating_mul(h).saturating_mul(4) {
            return Err(ExportError::Encode("the RGBA picture is smaller than its size".into()));
        }
        filmcraft_export::rgba_to_yuv420_8(f.rgba, w, h, &mut self.y, &mut self.u, &mut self.v);
        let packets = self.vt.encode(f.index, &self.y, &self.u, &self.v).map_err(ExportError::Encode)?;
        // the muxer needs the parameter sets after the first group of pictures
        if f.index == 0 {
            let params = self.vt.parameter_sets().ok_or_else(|| ExportError::Encode("VideoToolbox returned no parameter sets for the first frame".into()))?;
            if codec == VtCodec::Hevc {
                hevc_config(&params).map_err(ExportError::Encode)?;
            }
        }
        self.samples(packets)
    }

    fn flush(&mut self) -> filmcraft_export::Result<Vec<EncodedPacket>> {
        let packets = self.vt.flush().map_err(ExportError::Encode)?;
        self.samples(packets)
    }
}

// ---------------------------------------------------------------------------
// Linux: the VA-API H.264 slice encoder as an export encoder
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
mod vaapi_export {
    use crate::vaapi::{EncodeRequest, EncodedPicture, VaEncoder};
    use filmcraft_export::{BitrateMode, H264Pass};
    use filmcraft_export::{EncoderFrame, ExportError, ExportSettings, Format, HardwareEncoding, VideoEncoder};
    use filmcraft_isobmff::SampleEntry;
    use filmcraft_time::FrameRate;
    use std::sync::OnceLock;

    /// **The bitrate an export asks of this backend, in bits per second.**
    ///
    /// This is [`crate::vaapi::RateControl`] spelled in the terms [`ExportSettings`] uses, and the
    /// reason it is a separate type rather than the `vaapi` one is reachability rather than taste:
    /// `vaapi/mod.rs` re-exports an explicit list of names, and a type only the export gate and the
    /// gate's tests need does not belong in a list the encode milestone maintains. [`build`] is the
    /// one place the two are translated, so the translation is one function rather than a rule spread
    /// over every caller.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum ExportBitrate {
        /// A constant bit rate: the same number every second, and what `BitrateMode::Cbr` means.
        Cbr(u32),
        /// A variable bit rate between a target and a ceiling, both in bits per second — which is
        /// what FilmCraft's `target_kbps` and `max_kbps` mean together.
        Vbr {
            /// The rate the driver is asked to aim at.
            target: u32,
            /// The highest rate it may reach.
            max: u32,
        },
    }

    impl ExportBitrate {
        /// What [`crate::vaapi::VaEncoder`] is built with, or why it cannot be built.
        fn build(&self, display: &std::sync::Arc<crate::vaapi::Display>, size: (u32, u32), references: u32) -> Result<VaEncoder, String> {
            match *self {
                Self::Cbr(bps) => VaEncoder::at_constant_bitrate(display, size, references, bps),
                Self::Vbr { target, max } => VaEncoder::at_variable_bitrate(display, size, references, target, max),
            }
        }
    }

    /// What [`crate::vaapi::VaEncoder`] needs that [`ExportSettings`] states, or why this
    /// export is not one it can do.
    ///
    /// # The bitrate: submitted, not declined
    ///
    /// **A settings-set bitrate is honoured.** [`crate::vaapi::RateControl::Cbr`] carries
    /// `s.bitrate_kbps` and [`crate::vaapi::RateControl::Vbr`] carries the `target`/`max_kbps` pair
    /// FilmCraft's own settings already mean, in the fields `VAEncMiscParameterRateControl` has for
    /// exactly those two numbers. That is the whole of what changed when this gate was opened: before,
    /// every mode `BitrateMode` can name was refused and the registered factory was a decline rather
    /// than an encoder; now `Cbr` and `Vbr1Pass` are taken and the other two are refused, each with
    /// its own reason — `Vbr2Pass` needs an analysis pass over the pictures before the real one, and
    /// `ConstantQuality` names no rate at all.
    ///
    /// # The two-pass refusal, and why it is not folded into the bitrate one
    ///
    /// `Vbr2Pass` is refused **before** any bitrate is read, because it is a different refusal: a
    /// two-pass export renders the sequence once to measure it and again to encode it, and this
    /// backend is handed pictures one at a time by a pipeline that has no way to encode them twice.
    /// Submitting the second pass's bitrate to a single-pass driver would produce a file at a rate
    /// nobody chose with nothing wrong anywhere in it, so it is an error naming the pass. The
    /// constant-quality refusal below is refused in the same place for the same structural reason —
    /// both are read *before* `s.bitrate_kbps`, which is the field whose value they must never reach.
    ///
    /// # What is still declined, and it is a short list
    ///
    /// ## The two refusals that are *not* about capability, named as such
    ///
    /// - **A bitrate of zero is refused rather than turned into something.** The settings layer floors
    ///   a target at 100 kbps before a factory is called, so a zero here is a caller that skipped it —
    ///   and the only honest answer to "encode at no bits per second" is an error, not a substituted
    ///   number.
    /// - **A constant-quality export is refused rather than given a bitrate.**
    ///   [`BitrateMode::ConstantQuality`] states a quantiser in `ExportSettings::constant_qp` and no
    ///   rate at all, and every [`ExportBitrate`] variant is a rate — so the `_ =>` arm below, which
    ///   once turned this mode into a VBR and dropped the quantiser, encoded it at a rate nobody chose
    ///   while `ExportSettings::summary` printed `QP n` beside it. The decline carries the quantiser,
    ///   which is what makes it this export's error and not a passing remark: H.264 has a software
    ///   encoder behind this one, so the file the user gets is the software encoder's, at the
    ///   quantiser they asked for. [`crate::vaapi::hevc_config_for`] refuses the same mode in the same
    ///   words, and the refusal is not a claim about the driver: `VA_RC_CQP` exists and is measured
    ///   (see the check below).
    ///
    /// ## The refusals that *are* about what the records can express
    ///
    /// - **Baseline and High profiles**, because the derived parameter sets say Main. The sequence
    ///   record carries no profile field to derive from and an SPS claiming High while the slices are
    ///   CAVLC at 4x4 would describe a different stream from the one encoded. **`Auto` is taken**, and
    ///   resolves to Main: a preset that names no profile is exactly the case this backend can serve
    ///   honestly (see `ExportSettings::resolved_h264_profile`).
    /// - **Interlaced**, because the sequence record writes `frame_mbs_only_flag = 1` and a field
    ///   sequence cannot be expressed at all.
    /// - **Fractional frame rates**, because `VAEncMiscParameterFrameRate`'s numerator here is whole
    ///   frames per second and rounding 23.976 to 24 is a stream at a rate nobody chose.
    ///
    /// # The driver's own answer, kept in the message
    ///
    /// The bitrate refusal this replaced named `VAConfigAttribRateControl`'s answer so a reader could
    /// tell "this driver cannot" from "this backend did not". That distinction still matters — the
    /// modes *not* taken (ICQ, the temporal-layer ones) are declined for reasons of their own — so
    /// [`rate_control_modes_named`] is still what a diagnostic prints, and the driver is still asked
    /// before a config is made.
    pub fn config_for(format: Format, w: u32, h: u32, rate: FrameRate, s: &ExportSettings) -> Result<(u32, u32, u32, u32, ExportBitrate), String> {
        if format != Format::H264 {
            // **HEVC is not declined here any more, and this message is what says so.** It used to
            // read "HEVC: the Linux backend has no HEVC encoder", which was true then and is false
            // now: `Format::Hevc` is served by [`crate::vaapi::hevc_config_for`], which has HEVC's own
            // decline list (a picture size that is a whole number of 8-sample coding blocks, no
            // H.264 `MaxFrameNum` ceiling, no H.264 profile). **The factory routes the two formats
            // to the two gates**, so this one declines a format it has no records for and says which
            // gate does — rather than reporting a machine that cannot encode HEVC at all.
            //
            // Naming the other gate matters: a reader who saw only "not H.264" could not tell a
            // decline from a bug, and could not tell either that HEVC is taken by the *other* gate.
            return Err(if format == Format::Hevc {
                "HEVC: this is the H.264 gate, and a Format::Hevc export is served by the HEVC gate (crate::vaapi::hevc_config_for) \
                 with HEVC's own decline list"
                    .into()
            } else {
                format!("{format:?} is not H.264: the H.264 gate takes H.264, and Format::Hevc goes to the HEVC gate")
            });
        }
        if s.hardware_encoding != HardwareEncoding::Auto {
            return Err("hardware encoding is off".into());
        }
        if s.format.is_mxf() {
            return Err("MXF carries an Annex B stream, and this backend's samples are length-prefixed".into());
        }
        // **Two-pass is refused before the bitrate is read**, and for its own reason: a two-pass
        // export renders the sequence twice and this backend is handed each picture once. The bitrate
        // gate below would otherwise accept it and the file would come out at a rate nobody chose.
        if !matches!(s.h264_pass, H264Pass::Single) {
            return Err("two-pass encoding".into());
        }
        if s.bitrate_mode == BitrateMode::Vbr2Pass {
            return Err("two-pass encoding".into());
        }
        // **A constant-quality export is refused, and it is refused before the bitrate is read** —
        // for the same reason and in the same words as [`crate::vaapi::hevc_config_for`], which had
        // to add the same check to its own `_ =>` arm. The mode states a quantiser in
        // [`ExportSettings::constant_qp`] and no rate at all, so the `bitrate_kbps` below is a
        // number left over from another mode: the settings layer still carries a target (20 000 kbps
        // by default) precisely because the field has always been there, and `resolve` still floors
        // and ceilings it for a mode that names no rate. The `_ =>` arm this replaces turned that
        // leftover into a `VAEncMiscParameterRateControl` — a 20.00 Mbps VBR under a 1.5x ceiling,
        // for an export whose summary read `QP 26`. Nothing in the file would have said otherwise:
        // every `ExportBitrate` variant is a rate, so the quantiser was simply gone.
        //
        // **Declining is the honest answer and not a gap.** The record that would carry the
        // quantiser is `VAPictureH264::pic_init_qp` (`pic_init_qp_minus26 + 26`,
        // `/usr/include/va/va_enc_h264.h:342`) under `VA_RC_CQP` (`nCQP = 0x00000010`,
        // `/usr/include/va/va.h`), and this backend has both — `RateControl::ConstantQuantiser` is
        // measured on this driver and `examples/vaencprobe.rs` exercises it. **Submitting it here is a
        // separate piece of work, not a branch of this match**: it changes what the gate sends, so
        // until it is measured end to end the answer to a mode this gate cannot yet express is the
        // decline the HEVC gate already gives, not a substitution. On this side of the fallback the
        // decline is also the better export — the software encoder takes `ConstantQuality` as
        // `RateControl::Qp` and codes the quantiser the user asked for.
        if s.bitrate_mode == BitrateMode::ConstantQuality {
            return Err(format!(
                "a constant-quality export: it states a quantiser ({}) and no bitrate, and this gate submits the bitrate. \
                 FilmCraft's software encoders take that mode; this backend is declined for it rather than encoded at a rate nobody chose",
                s.constant_qp
            ));
        }
        // **The bitrate, as the mode FilmCraft's own settings mean.** `bitrate_kbps` is the target;
        // `max_bitrate_kbps` is the ceiling and, exactly as `ExportSettings::resolve` computes it
        // (and as the software H.264 encoder's own factory does at `lib.rs:1159`), a ceiling below
        // the target is not a VBR range and becomes 1.5x.
        //
        // **A bitrate of zero is refused rather than turned into something.** The settings layer
        // floors `target_kbps` at 100 kbps before an encoder factory is called (`resolve`), so a
        // zero here means the caller bypassed that — and the only honest answer to "encode at no
        // bits per second" is an error, not a substituted number.
        let kbps = s.bitrate_kbps;
        if kbps == 0 {
            return Err(format!(
                "a bitrate target of 0 kbps: this backend submits the bitrate it is given rather than choosing one, and there is \
                 nothing here to convert. The settings layer floors a target at 100 kbps (ExportSettings::resolve), so a zero here \
                 is a caller that skipped it. The driver reports {} for rate control",
                super::vaapi_export::rate_control_modes_named()
            ));
        }
        // **kbit/s → bit/s, checked.** `saturating_mul` would turn an overflow into a bitrate three
        // orders of magnitude too small — a file at 0.1% of the requested size with no error anywhere,
        // which is the whole class of defect this gate exists to refuse. `u32::MAX / 1000` is about
        // 4.29 Gbit/s, past any H.264 export.
        let bps = kbps.checked_mul(1000).ok_or_else(|| format!("a bitrate target of {kbps} kbps, which does not fit the record's bits-per-second field"))?;
        let bitrate = match s.bitrate_mode {
            BitrateMode::Cbr => ExportBitrate::Cbr(bps),
            // The ceiling, in the same checked units. `max` at 1.5x is `ExportSettings::resolve`'s own
            // rule, applied to a number it has already floored — the same expression, so the two
            // backends cannot pick different ceilings for the same settings.
            _ => {
                let max_kbps = s.max_bitrate_kbps.filter(|m| *m >= kbps).unwrap_or(kbps.saturating_mul(3) / 2);
                let max_bps = max_kbps
                    .checked_mul(1000)
                    .ok_or_else(|| format!("a VBR ceiling of {max_kbps} kbps, which does not fit the record's bits-per-second field"))?;
                ExportBitrate::Vbr { target: bps, max: max_bps }
            }
        };
        if s.signal.is_hdr() {
            return Err("HDR: this backend encodes 4:2:0 8-bit and writes no VUI or `colr`".into());
        }
        if !w.is_multiple_of(2) || !h.is_multiple_of(2) {
            return Err("odd picture size: 4:2:0 chroma is subsampled by two".into());
        }
        if s.pixel_aspect.is_some_and(|(n, d)| n != d) {
            return Err("non-square pixels: the derived SPS carries no VUI aspect-ratio signalling".into());
        }
        // **Interlaced is refused rather than silently progressive.** `sequence_record` writes
        // `frame_mbs_only_flag = 1`, so a field sequence cannot be expressed at all, and an export
        // that asked for fields and got progressive pictures would be a file that does not say
        // what it is. The VideoToolbox gate does not need this check because VideoToolbox takes
        // the setting; this backend cannot.
        if s.field_order != filmcraft_export::FieldOrder::Progressive {
            return Err("interlaced output: this encoder writes frame_mbs_only = 1 and cannot express a field sequence".into());
        }
        // **Baseline and High are refused because the derived SPS says Main.** `params::sps`
        // writes `profile_idc = 77` unconditionally — the sequence record carries no profile field
        // to derive it from, and an SPS claiming High while the slice headers are CAVLC and the
        // transform is 4x4 would describe a different stream from the one encoded. A profile this
        // backend cannot state faithfully is a decline, not a nearest match.
        //
        // **`H264Profile::Auto` is not one of those profiles: it is the delegation this backend
        // *can* honour.** `ExportSettings::resolved_h264_profile` resolves it to Main on Linux (and
        // the hardware-encoding check above has already established that the hardware path is the
        // one in play), so a preset that names no profile passes this gate and comes out a Main
        // file that says so — which is the whole reason the variant exists. The test is written
        // against the *resolved* profile rather than against a list that also accepts `Auto`, so
        // that the two places which know what this encoder writes cannot drift apart.
        if s.resolved_h264_profile() != filmcraft_export::H264Profile::Main {
            return Err(format!(
                "the {} H.264 profile: this backend's derived parameter sets state Main. Choose Auto or Main to export through it",
                s.h264_profile.label()
            ));
        }
        // The frame rate, as the whole number of pictures a second the encoder is told.
        //
        // **A fractional rate is refused rather than rounded**, and the check is done on the
        // integers rather than on a float: `rate.num / rate.den` as an `f64` cannot tell 24 from
        // 23.976 reliably at the sizes involved, and the answer here has to be exactly right
        // because it becomes `VAEncMiscParameterFrameRate`'s numerator. `num % den == 0` is the
        // definition of a whole number of frames per second and is not a floating-point
        // comparison at all.
        if rate.num <= 0 || rate.den <= 0 {
            return Err(format!("an invalid frame rate of {}/{}", rate.num, rate.den));
        }
        if rate.num % rate.den != 0 {
            return Err(format!("a fractional frame rate of {}/{}: this backend submits whole frames per second", rate.num, rate.den));
        }
        let fps = u32::try_from(rate.num / rate.den).map_err(|_| format!("a frame rate of {} fps", rate.num / rate.den))?;
        if fps == 0 {
            return Err("a frame rate of 0 fps".into());
        }
        // The key-frame interval, and **the ceiling the frame loop imposes**. `MaxFrameNum` is 256
        // here (see `vaapi::MAX_FRAME_NUM`), so the loop inserts a key frame every 256 pictures
        // whatever this is. That is not something to pass through silently: an export that asked
        // for 300 and gets 256 has a file a sixth larger than it was promised, and the only honest
        // answers are to tell the caller or to refuse. This refuses, with the number in the
        // message; the loop's own `key_frame_reason` is the safety net for the runs that do pass.
        //
        // **The refusal is still a refusal; the ceiling it names is now 256 rather than 16.** Every
        // preset FilmCraft ships asks for at most 250, so this no longer declines an interval the
        // settings state outright — what it still catches is an interval past what `frame_num` can
        // number at all, which is a real limit and not one this gate invented.
        let gop = s.keyframe_distance.filter(|k| *k > 0).unwrap_or_else(|| (f64::from(fps) * 2.0).round().max(1.0) as u32);
        if u64::from(gop) > u64::from(VaEncoder::max_frame_num()) {
            return Err(format!(
                "a key frame every {gop} pictures: this encoder's MaxFrameNum is {} (frame_num is eight bits wide), so it cannot \
                 reach that far without a key frame of its own",
                VaEncoder::max_frame_num()
            ));
        }
        Ok((w, h, fps, gop, bitrate))
    }

    /// What the encoder needs from the settings, resolved to numbers.
    pub struct VaEncoderExport {
        encoder: VaEncoder,
        w: u32,
        h: u32,
        duration: u32,
        rate: FrameRate,
        y: Vec<u8>,
        u: Vec<u8>,
        v: Vec<u8>,
        /// Pictures handed to [`VideoEncoder::encode`] so far, which is what the timestamp check
        /// counts against.
        done: u64,
    }

    impl VaEncoderExport {
        /// One sample per picture, at its own time.
        ///
        /// **`composition_offset` is 0 and the picture count is checked rather than assumed.** This
        /// backend codes no picture out of display order (measured — see `vaapi::VaEncoder`), so a
        /// packet's presentation time is `n × duration` and its composition offset its own time, so
        /// a non-zero offset would be a picture the bitstream does not have. A caller asking for a
        /// key frame explicitly is honoured: `VaEncoder::encode` takes the picture kind from the
        /// request, and the loop's own forced key frames (at the `frame_num` wrap) come back on
        /// `EncodedPicture::key_frame` and are passed through, so the sample table says where the
        /// stream's random-access points really are rather than where the settings asked for them.
        fn sample(&mut self, picture: EncodedPicture) -> filmcraft_export::Result<Vec<filmcraft_export::EncodedPacket>> {
            let data = picture.mp4_sample().map_err(ExportError::Encode)?;
            self.done = self.done.saturating_add(1);
            Ok(vec![filmcraft_export::EncodedPacket { data, key: picture.key_frame, duration: self.duration, composition_offset: 0 }])
        }
    }

    impl VideoEncoder for VaEncoderExport {
        /// The `avcC` the encoder has built — which is an **`Err` until a picture has been
        /// encoded**, and `VideoEncoder::sample_entry` cannot return an `Err`. So the refusal is
        /// carried as an *empty* configuration record, which is what the VideoToolbox path does
        /// for the same reason, and the muxer reads the parameter sets again off the first sample.
        ///
        /// That is a real limitation and it is the `VideoEncoder` trait's, not this backend's: the
        /// trait says a sample entry "may only be complete after the first frame" and has no way to
        /// say "there is not one yet". `VaEncoder::sample_entry` answers properly with a `Result`;
        /// this method cannot.
        fn sample_entry(&self) -> SampleEntry {
            match self.encoder.sample_entry() {
                Ok(entry) => entry,
                Err(why) => {
                    log::info!("VA-API: no sample entry yet, so the muxer takes the parameter sets off the first sample: {why}");
                    SampleEntry::avc(
                        filmcraft_isobmff::AvcConfig::new(Vec::new(), Vec::new(), 4),
                        u16::try_from(self.w).unwrap_or(u16::MAX),
                        u16::try_from(self.h).unwrap_or(u16::MAX),
                    )
                }
            }
        }

        fn timescale(&self) -> u32 {
            u32::try_from(self.rate.num).unwrap_or(u32::MAX)
        }

        fn encode(&mut self, f: &EncoderFrame) -> filmcraft_export::Result<Vec<filmcraft_export::EncodedPacket>> {
            if f.hdr.is_some() {
                return Err(ExportError::Unsupported("VA-API hardware encoding does not take HDR pictures".into()));
            }
            if (f.width, f.height) != (self.w, self.h) {
                return Err(ExportError::Encode(format!("VA-API: a {}x{} picture for a {}x{} encoder", f.width, f.height, self.w, self.h)));
            }
            let (w, h) = (self.w as usize, self.h as usize);
            if f.rgba.len() < w.saturating_mul(h).saturating_mul(4) {
                return Err(ExportError::Encode("the RGBA picture is smaller than its size".into()));
            }
            filmcraft_export::rgba_to_yuv420_8(f.rgba, w, h, &mut self.y, &mut self.u, &mut self.v);
            // 4:2:0 interleaved chroma, which is what `VaEncoder::upload` takes: the chroma
            // planes are interleaved into one buffer of half the luma's area.
            let chroma_len = (w / 2) * h;
            let mut chroma = Vec::with_capacity(chroma_len);
            for (u, v) in self.u.iter().zip(self.v.iter()) {
                chroma.push(*u);
                chroma.push(*v);
            }
            self.encoder
                .upload(&self.y, &chroma)
                .map_err(|e| ExportError::Encode(format!("VA-API: {e} (turn Export ▸ Hardware encoding off to use the software encoder)")))?;
            // **No quantiser is named, and that is the whole point.** The export gate built this
            // encoder under a [`RateControl`] derived from `s.bitrate_mode` and `s.bitrate_kbps`, so
            // the picture's fidelity is that bitrate and there is no honest QP for this function to
            // pass — `EncodeRequest::rate_controlled` is the constructor that says so, and
            // `VaEncoder::encode` refuses the other pairing rather than quietly honouring one half.
            //
            // **`s.quality` (0-100) is still not used**, and now for a stronger reason than before: it
            // was ignored because the backend was constant quantiser and the software encoder ignores
            // it too; it is ignored now because a percentage mapped onto a QP would be a table this
            // backend would be inventing, sitting next to a bitrate the user did state. Two
            // statements about the same fidelity, one of them invented.
            let request = EncodeRequest::predicted_at_rate().map_err(ExportError::Encode)?;
            let picture = self
                .encoder
                .encode(request)
                .map_err(|e| ExportError::Encode(format!("VA-API: {e} (turn Export ▸ Hardware encoding off to use the software encoder)")))?;
            filmcraft_export::note_hw_encode_frame();
            self.sample(picture)
        }

        fn flush(&mut self) -> filmcraft_export::Result<Vec<filmcraft_export::EncodedPacket>> {
            // Every picture was read back inside `encode`, so there is nothing queued: flush ends
            // the stream and checks the last picture was marked, which is a fact about the stream
            // rather than a guess (see `VaEncoder::flush`).
            self.encoder.flush().map_err(ExportError::Encode)?;
            Ok(Vec::new())
        }

        /// **Always `None`, and that is a measurement rather than a default.** `media_start` is the
        /// B-frame delay an edit list needs, and this backend codes no picture out of display
        /// order: measured over a 40-picture run, every slice the driver wrote was a P or an I and
        /// every `pic_order_cnt_lsb` was in display order (`vaapi::VaEncoder`'s module note). A
        /// delay invented here would be an edit list for a delay that does not exist.
        fn media_start(&self) -> Option<i64> {
            None
        }
    }

    /// The Export encoder factory for the VA-API encoders — **one factory, two gates**.
    ///
    /// # The shape, which is VideoToolbox's shape rather than a third one
    ///
    /// `let hevc = ...` is the same line the macOS side runs before its own decline list, and it is
    /// here for the same reason: **HEVC has no built-in encoder in FilmCraft, so choosing the format
    /// is the opt-in**, exactly as on macOS — while H.264 always has the software encoder behind it
    /// and therefore needs the `Export ▸ Hardware encoding` flag as well. A factory that read the
    /// flag for both would decline every HEVC export on a machine where the user had left the flag
    /// off, which is the opposite of what choosing the format means.
    ///
    /// # What is shared and what is not
    ///
    /// **Shared, deliberately**: this function, the `declined` closure (the `perf.stats` counter),
    /// the display, the one session counter, and the fact that a decline is `None` rather than an
    /// `Err` — because a declined export is *not* a failed one: there is a software encoder behind
    /// H.264 and no encoder at all behind HEVC, and the latter is what the export code reports as
    /// "encoder not available".
    ///
    /// **Not shared**: the gate (`config_for` against [`crate::vaapi::hevc_config_for`]) and the
    /// adapter ([`VaEncoderExport`] against [`crate::vaapi::HevcEncoderExport`]). HEVC's decline list
    /// and H.264's are genuinely different documents — HEVC has no `frame_num` to run out of and no
    /// crop to express a padded size, and both of those are H.264 gate rules.
    pub fn factory(format: Format, w: u32, h: u32, rate: FrameRate, s: &ExportSettings) -> Option<filmcraft_export::Result<Box<dyn VideoEncoder>>> {
        let hevc = format == Format::Hevc;
        if !hevc && (format != Format::H264 || s.hardware_encoding != HardwareEncoding::Auto) {
            return None;
        }
        let declined = |why: &str| {
            log::info!("VA-API hardware {} encoding declined: {why}", if hevc { "HEVC" } else { "H.264" });
            filmcraft_export::note_hw_encode_declined();
            None
        };
        let display = match crate::vaapi::open_display() {
            Ok(d) => d,
            Err(why) => return declined(&format!("no VA-API display: {why}")),
        };
        // **HEVC first, and its own gate and adapter.** One reference picture on both sides, which is
        // a measurement on each: `HevcEncoder::new` refuses a deeper one with HEVC's own numbers.
        if hevc {
            let config = match crate::vaapi::hevc_config_for(format, w, h, rate, s) {
                Ok(c) => c,
                Err(why) => return declined(&why),
            };
            let encoder = match crate::vaapi::HevcEncoderExport::build(config, display, rate) {
                Ok(e) => e,
                Err(why) => return declined(&why),
            };
            filmcraft_export::note_hw_encode_session();
            return Some(Ok(Box::new(encoder)));
        }
        let (ew, eh, fps, gop, bitrate) = match config_for(format, w, h, rate, s) {
            Ok(c) => c,
            Err(why) => return declined(&why),
        };
        // One reference: the only depth measured to produce correct pictures, and the ceiling this
        // backend accepts (`VaEncoder::new` refuses deeper with the measurement in the message).
        let mut encoder = match bitrate.build(&display, (ew, eh), 1) {
            Ok(e) => e,
            Err(why) => return declined(&why),
        };
        if let Err(why) = encoder.at_rate(fps).and_then(|e| e.key_frames_every(gop)) {
            return declined(&why);
        }
        filmcraft_export::note_hw_encode_session();
        let duration = u32::try_from(rate.den).unwrap_or(u32::MAX);
        Some(Ok(Box::new(VaEncoderExport { encoder, w: ew, h: eh, duration, rate, y: Vec::new(), u: Vec::new(), v: Vec::new(), done: 0 })))
    }

    /// What the driver says it supports for rate control, for diagnostics and for the decline
    /// message. Not a decision: a mask of 0 means "the driver did not say".
    pub fn rate_control_modes() -> u32 {
        crate::vaapi::open_display().map(|d| VaEncoder::driver_rate_control_modes(&d)).unwrap_or(0)
    }

    /// The same answer, named — what a decline message puts in it, and what
    /// `examples/vaencprobe.rs` prints under "rate control".
    pub fn rate_control_modes_named() -> String {
        VaEncoder::rate_control_names(rate_control_modes())
    }

    // -----------------------------------------------------------------------
    // The AV1 availability probe
    // -----------------------------------------------------------------------

    /// The picture size the AV1 probe encodes: **320x240**, the size the AV1 and HEVC measurements
    /// were taken at, and a whole number of AV1's 8x8 superblocks in both directions.
    pub(super) const AV1_PROBE_SIZE: (u32, u32) = (320, 240);
    /// The rate the probe's [`crate::vaapi::Av1Plan`] states — **30 fps**, whole, which is what
    /// `VAEncMiscParameterFrameRate` takes.
    pub(super) const AV1_PROBE_FPS: u32 = 30;
    /// The quantiser: **100**, the value [`crate::vaapi::Av1Encoder::at_constant_quantiser`] is
    /// built with and mid-range on `base_q_idx`'s 0..=255.
    pub(super) const AV1_PROBE_QINDEX: u8 = 100;

    /// **Whether this machine's VA-API driver can produce an AV1 picture FilmCraft can use**, asked
    /// once, through [`OnceLock`]: the Linux counterpart of [`crate::vaapi::hevc_available`] and
    /// the same shape as it.
    ///
    /// # The criterion, and why it is not the cheap one
    ///
    /// **A VA-API AV1 encoder is available if and only if one key frame, submitted through
    /// [`crate::vaapi::Av1Encoder::encode`], renders, syncs, and maps back to a coded buffer holding
    /// at least one byte.** That is the whole criterion, and the last clause is it: *the driver
    /// produced a bitstream*, not *the driver took our records*.
    ///
    /// **The cheap question has a measured false positive here, which is the entire reason this
    /// probe is more expensive than the HEVC one.** [`crate::vaapi::hevc_available`] deliberately
    /// stops at "a session was created", on the grounds that this is the cheapest honest question. For
    /// AV1 that question is answered **yes by a driver that writes no AV1 byte at all**, measured on
    /// this machine's iHD 26.3.5 / libva 1.24 through the shipped submission path
    /// ([`crate::vaapi::driver_outcomes`]):
    ///
    /// ```text
    /// vaQueryConfigEntrypoints(VAProfileAV1Profile0) -> [1 VLD, 6 EncSlice]    va.h:553, va.h:558
    /// vaCreateConfig / vaCreateSurfaces / vaCreateContext -> VA_STATUS_SUCCESS
    /// vaRenderPicture([sequence record])                 -> VA_STATUS_SUCCESS
    /// vaEndPicture                                       -> VA_STATUS_ERROR_ENCODING_ERROR (24)  va.h:288
    /// vaRenderPicture([sequence record, picture record]) -> VA_STATUS_ERROR_INVALID_BUFFER (7)    va.h:271
    /// vaMapBuffer(coded)                                 -> the uninitialised buffer: no AV1 bytes
    /// ```
    ///
    /// So both of the cheap answers — "it advertises `VAEntrypointEncSlice`" and "it created a
    /// config, a context and a coded buffer" — are true here, and **neither distinguishes this
    /// driver from one that would encode**. Only asking the question that produces a bitstream
    /// does. A probe built on the advertisement would answer `true` on a machine where every AV1
    /// export fails at its first picture, and `register_format_probe` would then put AV1 in the
    /// format list on that machine — which is a lie told by the list rather than by the export.
    ///
    /// # What this deliberately does not claim
    ///
    /// - **Not bit-equality with any other encoder.** Encode verification is materially weaker than
    ///   decode's: a hardware encoder and FilmCraft's software encoder are different algorithms
    ///   making different decisions, so "matches the software encoder" has no content, and "matches
    ///   ffmpeg" would be measuring ffmpeg's encoder. **Nothing here says either.**
    /// - **Not that the bytes are a decodable AV1 bitstream.** "Bytes came back" is a weaker claim
    ///   than "AV1 came back", and it is the weaker claim deliberately: there is no AV1 bitstream on
    ///   this machine, so a stronger criterion could not be *evaluated* here, and a criterion that
    ///   cannot be evaluated is not a criterion. Parsing the first OBU with `filmcraft_av1` is the
    ///   obvious next strengthening, for the machine that eventually produces one.
    /// - **Not that an export at the user's picture size will work.** 320x240 is what this asks
    ///   about, exactly as it is for HEVC.
    ///
    /// # `Err` is the answer, not a failure of the probe
    ///
    /// Every failure — no display, no `libva`, a driver that refuses the config, and the refusal
    /// this machine gives at `vaRenderPicture` — is a `false`, with the reason logged. There is no
    /// error to propagate because nothing here is an export: a caller asking "can this machine
    /// export AV1" gets yes or no.
    ///
    /// # Why it is not registered today
    ///
    /// **It cannot be, and that is a fact about [`filmcraft_export::Format`] rather than about this
    /// probe.** There is no `Format::Av1` variant, so there is nothing to
    /// `filmcraft_export::register_format_probe` it against. Adding the variant *without* this
    /// probe would be worse than adding nothing: the format list would claim AV1 is available while
    /// every AV1 export failed. Adding it *with* this probe would be honest on this machine — the
    /// probe answers `false` here — but is still the wrong order of work, because a probe that says
    /// `false` is not a reason to ship a format list entry that never works. **The order is the
    /// format, the encoder, then the probe.**
    ///
    /// # What a `Format::Av1` would need, and none of which is here
    ///
    /// - **The variant**, in `filmcraft_export`: `Format::Av1` with its `from_name` / `id` /
    ///   `extension` (`mp4`) / `label`, a slot in `Format::ALL` (14 → 15),
    ///   `has_builtin_encoder` answering `false` for it, and `is_h26x` left alone — AV1 is neither
    ///   H.264 nor H.265.
    /// - **The registration**, `register_format_probe(Format::Av1, vaapi_av1_available)` beside the
    ///   HEVC one in `platform::register`. That is the single line this probe is waiting for.
    /// - **The gate and the adapter**: an `av1_config_for` decline list and an `Av1EncoderExport`
    ///   implementing `VideoEncoder`, routed from `vaapi_export::factory`, which is an
    ///   `hevc || h264` match today and would become a three-way one.
    /// - **A sample entry**: `SampleEntry::av1(Av1Config, w, h)` and an `av01` four-CC. The muxer
    ///   **already writes `av1C`** (`isobmff/src/codec.rs:1372`) and `Av1Config` already round-trips
    ///   through its own parser (`:1613`), but `SampleEntry` has only `avc` and `hevc` constructors
    ///   (`codec.rs:1271`, `:1275`) — so there is no way to hand the muxer an `av01` entry yet, and
    ///   an export that got one from elsewhere would be a file nobody has tested.
    /// - **A preset**, gated on `available(Format::Av1)` exactly as the HEVC ones are gated in
    ///   `export/src/presets.rs`.
    /// - **A driver that takes the picture record** — which is the part that does not exist on this
    ///   machine, and the only one no amount of code here can supply.
    pub fn av1_available() -> bool {
        static AVAILABLE: OnceLock<bool> = OnceLock::new();
        *AVAILABLE.get_or_init(|| match av1_probe() {
            Ok(bytes) => {
                log::info!("VA-API AV1: the driver produced a {bytes}-byte AV1 key frame, so this machine can export AV1");
                true
            }
            Err(why) => {
                log::info!("no usable VA-API AV1 encoder: {why}");
                false
            }
        })
    }

    /// The one question [`av1_available`] asks, as a value: **the number of bytes of AV1 the driver
    /// wrote for one key frame**, or why there are none.
    ///
    /// Every step is one FilmCraft's shipped path already takes, and none of them is the interesting
    /// one: open the render node (`O_RDWR`, `vaapi/ffi.rs:2903` — with `O_RDONLY`, `vaCreateContext`
    /// fails `VA_STATUS_ERROR_ALLOCATION_FAILED` for **every** codec and a working machine reads as
    /// having no VA-API at all), build the encoder, upload a ramp, encode one key frame. **The step
    /// that decides the answer is the last one**: `Av1Encoder::encode` renders, syncs and reads the
    /// coded buffer back, and refuses a picture it cannot show bytes for
    /// ([`crate::vaapi::DRIVER_REFUSES_THE_AV1_PICTURE_RECORD`]).
    ///
    /// # What it asks libva, and how each answer is judged
    ///
    /// - **No `vaQueryConfigAttributes` anywhere on this path.** That call segfaults inside
    ///   `iHD_drv_video.so` on libva 1.24, so nothing here may make it — and nothing does, because
    ///   the probe's only question is whether a picture comes back.
    /// - **`vaCreateConfig` is judged on its returned `VAStatus` and never on a value libva writes
    ///   back into the attribute array.** `Encoder::new` sets three attributes (render format, rate
    ///   control, packed headers) and reads only `st != OK`.
    /// - **`vaCreateBuffer` and `vaSyncBuffer` carry the arguments libva 1.24 gives them**
    ///   (`num_elements`, `timeout_ns`), through `Encoder`'s own wrapper, which is where those
    ///   arguments live — so a probe written here cannot omit them by omission.
    ///
    /// **The empty-bytes check below is not a duplicate of that refusal.** `encode` already refuses
    /// an empty buffer, so on this driver the `?` above returns first and this arm is not reached.
    /// It is here because it is the criterion: an `encode` that ever returns `Ok` with no picture is
    /// the one failure this probe exists to catch, and a check that only ever exercises its other
    /// branch is not a check of the criterion.
    pub(super) fn av1_probe() -> Result<usize, String> {
        let display = crate::vaapi::open_display()?;
        let (luma, chroma) = crate::vaapi::ramp_picture(AV1_PROBE_SIZE.0, AV1_PROBE_SIZE.1)?;
        let plan = crate::vaapi::Av1Plan::new(AV1_PROBE_SIZE.0, AV1_PROBE_SIZE.1, AV1_PROBE_FPS, AV1_PROBE_QINDEX)?;
        let mut encoder = crate::vaapi::Av1Encoder::at_constant_quantiser(&display, plan)?;
        encoder.upload(&luma, &chroma)?;
        let picture = encoder.encode(crate::vaapi::Av1EncodeRequest::key_frame(0))?;
        if picture.bytes.is_empty() {
            return Err("the driver accepted an AV1 picture record and wrote no bytes into the coded buffer, so there is no encoder here".into());
        }
        Ok(picture.bytes.len())
    }
}

#[cfg(target_os = "linux")]
pub use vaapi_export::factory as vaapi_encoder_factory;

#[cfg(target_os = "linux")]
pub use vaapi_export::{
    ExportBitrate, VaEncoderExport, av1_available as vaapi_av1_available, config_for as vaapi_config_for, rate_control_modes as vaapi_rate_control_modes,
    rate_control_modes_named as vaapi_rate_control_modes_named,
};

// ---------------------------------------------------------------------------
// The gate against the summary: the two statements a user reads about one file
// ---------------------------------------------------------------------------

/// The H.264 VA-API gate and the summary line describing the same export.
///
/// **`config_for` and `ExportSettings::summary` are in different crates and were written at different
/// times**, so nothing links them: each can be right on its own while the pair lies. These are the
/// tests that hold the two together, and they are unit tests in this file rather than in `tests/`
/// because the thing under test is this file's gate.
#[cfg(all(test, target_os = "linux"))]
mod gate_against_summary {
    use super::vaapi_export::{ExportBitrate, config_for};
    use filmcraft_export::{BitrateMode, ExportSettings, Format, H264Profile, HardwareEncoding};
    use filmcraft_time::{FrameRate, Tick};

    const W: u32 = 640;
    const H: u32 = 360;
    const FPS: FrameRate = FrameRate::FPS_30;
    const QP: u8 = 30;

    /// An H.264 export this gate otherwise takes: Main profile, progressive, square pixels, SDR, a
    /// whole frame rate, one pass, a key-frame interval inside `MaxFrameNum`, and a target rate
    /// **set to a value no preset uses** — so the numbers below are this test's, not the defaults'.
    fn settings(mode: BitrateMode) -> ExportSettings {
        ExportSettings {
            format: Format::H264,
            hardware_encoding: HardwareEncoding::Auto,
            h264_profile: H264Profile::Main,
            bitrate_mode: mode,
            bitrate_kbps: 2_000,
            max_bitrate_kbps: Some(4_000),
            constant_qp: QP,
            keyframe_distance: Some(8),
            ..Default::default()
        }
    }

    /// The gate's answer for `mode`, as the record it hands the driver.
    fn bitrate(mode: BitrateMode) -> Result<ExportBitrate, String> {
        config_for(Format::H264, W, H, FPS, &settings(mode)).map(|(_, _, _, _, b)| b)
    }

    /// What the summary tells the user this export's video fidelity is.
    fn summary_video(mode: BitrateMode) -> String {
        settings(mode).summary(W, H, FPS, 48_000, Tick::from_units(300, FrameRate::FPS_30.den)).video
    }

    /// **A constant-quality export is declined, and the decline names the quantiser it stated.**
    ///
    /// Before the fix this returned `Ok(Vbr { target: 2_000_000, max: 4_000_000 })`: the mode fell
    /// into the `_ =>` arm of the bitrate match, so the settings' quantiser was dropped on the floor
    /// and a rate nobody chose was submitted instead, while the summary beside it said `QP 30`.
    #[test]
    fn a_constant_quality_export_is_declined_with_its_quantiser() {
        let Err(why) = bitrate(BitrateMode::ConstantQuality) else {
            panic!("a constant-quality export was accepted by a gate that submits a bitrate: it states a quantiser and no rate");
        };
        // The number, not just the mode: matching the mode would pass on a fixed string printed
        // beside any export, and the claim being tested is that the gate *read the settings*.
        assert!(why.contains(&QP.to_string()), "the decline names the quantiser the settings stated, not just the mode: {why}");
        // **The same settings with the one word changed produce the record this mode used to get.**
        // That is the counterpart, and it is why the refusal is the fix rather than the substitution:
        // the VBR below is the file a user asking for QP 30 would have received and been told was QP 30.
        assert_eq!(bitrate(BitrateMode::Vbr1Pass), Ok(ExportBitrate::Vbr { target: 2_000_000, max: 4_000_000 }));
        // **And the number a real export would have carried**, because `ExportSettings::default()`
        // names no rate at all in constant-quality mode: its 20 000 kbps target (a figure left over
        // from the rate-based modes) became a 20.00 Mbps VBR under a 1.5x ceiling, beside a summary
        // reading `QP 26`. Asserted through the gate's own output rather than quoted, so the number
        // is checked rather than carried along in a comment.
        let mut defaults = ExportSettings { bitrate_mode: BitrateMode::Vbr1Pass, ..Default::default() };
        defaults.hardware_encoding = HardwareEncoding::Auto;
        defaults.h264_profile = H264Profile::Main;
        assert_eq!(config_for(Format::H264, W, H, FPS, &defaults).map(|(_, _, _, _, b)| b), Ok(ExportBitrate::Vbr { target: 20_000_000, max: 30_000_000 }));
    }

    /// **The gate's answer for every mode, checked against the mode and against the summary.**
    ///
    /// `ExportBitrate` has two variants and every one of them is a rate, so a mode the gate accepts
    /// is encoded at a rate — and a summary saying `QP n` is a claim about a quantiser this backend
    /// never submits. The two are contradictions, whichever of them arrives first, so each mode is
    /// checked both ways: **the mode the gate takes is spelled out**, and **the summary's claim about
    /// fidelity is derived from the settings rather than quoted**, so a mode added to `BitrateMode`
    /// later has no arm to fall into and fails here instead of silently inheriting the `_ =>`.
    ///
    /// **Written over `BitrateMode::ALL`, so a fifth mode is covered by construction** — which is the
    /// point: this defect was a `_ =>` arm, and an arm is by definition the one a new variant lands in.
    #[test]
    fn the_gate_never_contradicts_what_the_summary_says_the_fidelity_is() {
        for mode in BitrateMode::ALL {
            let gate = bitrate(mode);
            let says = summary_video(mode);
            let says_qp = says.contains("QP ");
            assert!(
                !(gate.is_ok() && says_qp),
                "{mode:?} is taken by the gate, which submits a bitrate, while the summary quotes a quantiser ({says}): {gate:?}"
            );
            if gate.is_ok() {
                // The converse, from the other side: what the gate takes is a mode the summary
                // describes with a rate. Asserted against the settings' own predicate rather than a
                // copy of it, so the two sides of the comparison are computed independently.
                assert!(mode.names_a_bitrate(), "{mode:?} is taken by the gate, but the summary describes it without a rate");
                assert!(says.contains("Mbps"), "{mode:?} is taken at a rate, so the summary must quote one: {says}");
            }
        }
        // **The table itself**, so "every mode was considered" is a fact rather than a claim about
        // the loop above: what this gate submits, or the words it declines with. A mode whose answer
        // changes has to change this line too, which is the review that should accompany it.
        assert_eq!(bitrate(BitrateMode::Cbr), Ok(ExportBitrate::Cbr(2_000_000)), "CBR submits the target alone");
        assert_eq!(bitrate(BitrateMode::Vbr1Pass), Ok(ExportBitrate::Vbr { target: 2_000_000, max: 4_000_000 }), "one-pass VBR submits the pair");
        assert!(bitrate(BitrateMode::Vbr2Pass).is_err_and(|why| why.contains("two-pass")), "two-pass is declined as two-pass, not as a bitrate");
        assert!(
            bitrate(BitrateMode::ConstantQuality).is_err_and(|why| why.contains("constant-quality")),
            "constant quality is declined by name, not mapped onto a rate"
        );
    }

    /// **The summary and the decline both follow the settings, and agree with each other.**
    ///
    /// The file either says it was encoded at a quantiser or the export says why it cannot be, and
    /// the number in both statements has to be the number the user typed: the summary prints
    /// `self.constant_qp` (not a resolved or clamped value) and the decline prints the same field.
    /// **Three quantisers rather than one**, including both ends of H.264's range — "Range: 0 to 51,
    /// inclusive" for the macroblock quantiser parameter, `/usr/include/va/va_enc_h264.h:570` — so a
    /// fixed 26 or a fixed 30 printed beside any export would fail here rather than pass on the
    /// default.
    #[test]
    fn the_summary_and_the_decline_quote_the_same_quantiser() {
        for qp in [QP, 0, 51] {
            let mut s = settings(BitrateMode::ConstantQuality);
            s.constant_qp = qp;
            let said = s.summary(W, H, FPS, 48_000, Tick::from_units(300, FrameRate::FPS_30.den)).video;
            assert!(said.contains(&format!("QP {qp}")), "QP {qp}: the summary quotes the quantiser the settings state, not a fixed one: {said}");
            let Err(why) = config_for(Format::H264, W, H, FPS, &s).map(|(_, _, _, _, b)| b) else {
                panic!("a constant-quality export was accepted by a gate that submits a bitrate");
            };
            assert!(why.contains(&format!("quantiser ({qp})")), "QP {qp}: the decline quotes the same number the summary did: {why}");
        }
    }
}

// ---------------------------------------------------------------------------
// The AV1 probe: what it answers, and the control that makes the answer evidence
// ---------------------------------------------------------------------------

/// **The AV1 probe on this machine, and the H.264 control that gives its answer meaning.**
///
/// # Why a control is required, and what it must be
///
/// **A probe that answers `false` on a machine with no VA-API at all is worth nothing**: a missing
/// render node, an unresolvable `libva.so.2`, a dead DRM client — every one of those produces the
/// same `false`, and each is a machine fact rather than an AV1 fact. So the H.264 encoder is run
/// through **the same four steps in this process**: open the render node, build a slice encoder,
/// upload a ramp, render one picture and map the coded buffer back. Only the codec differs, so an
/// H.264 picture coming back is evidence that *this machine, through this code, right now* can do
/// it — which is what leaves the AV1 answer as a statement about AV1.
///
/// **H.264 is the right control for a second, measured reason**: it is the codec this driver
/// advertises an encoder for *and* actually has one for. `examples/vaprobe.rs --caps` on this
/// machine answers `H264Main VLD: true EncSlice: true` — the advertisement — and the control below
/// is what establishes the picture, which the advertisement never does (for AV1, on this very
/// driver, it demonstrably does not).
///
/// **The steps are not the interesting part; the control is.** Every step both probes take is one
/// FilmCraft's shipped path already takes, and a probe that failed at `open_display` or at
/// `vaCreateConfig` would be saying something about the machine rather than about AV1. The test
/// below pins that: the refusal has to name `vaRenderPicture`.
///
/// # What none of this is
///
/// **Encode verification is materially weaker than decode's and nothing here pretends otherwise.** A
/// hardware encoder and a software encoder are different algorithms making different decisions, so
/// there is no bit-equality available at any level, and **"matches ffmpeg" is never a claim about
/// an encode** — it would be a measurement of ffmpeg's encoder. There is no AV1 bitstream on this
/// machine at all, so there is nothing to decode and nothing at all to say about fidelity: what is
/// measured here is **whether a driver produces bytes**, not whether they are good ones.
#[cfg(all(test, target_os = "linux"))]
mod av1_probe_evidence {
    use super::vaapi_export::{AV1_PROBE_FPS, AV1_PROBE_QINDEX, AV1_PROBE_SIZE, av1_available, av1_probe};
    use crate::vaapi::{Av1Encoder, Av1Plan, DRIVER_REFUSES_THE_AV1_PICTURE_RECORD, EncodeRequest, Entrypoint, Profile, VaEncoder, open_display, ramp_picture};

    /// **The libva session is process-wide on this driver**, so the tests below take turns. The
    /// mutex is poisoned deliberately: a test that panicked while holding it has not broken the
    /// session, and `AGENTS.md` §0.6 says lock poisoning is not fatal.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        SERIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// **The H.264 control: [`av1_probe`]'s four steps, in this process, on the codec this machine
    /// does encode.** [`VaEncoder`] is the submission path [`super::VaEncoderExport`] wraps, so
    /// every libva call in here is the same call the H.264 export path makes — the driver, the
    /// render node, `vaRenderPicture`, `vaSyncBuffer`, `vaMapBuffer`.
    ///
    /// **One picture, a key frame**, because a predicted picture would need a reference and the
    /// point is the session, not the frame loop. The bitrate is a rate nobody is choosing for a
    /// probe; the ramp compresses to almost nothing at any rate, which is what makes a picture
    /// coming back unambiguous.
    fn h264_control() -> Result<usize, String> {
        let display = open_display()?;
        let (luma, chroma) = ramp_picture(AV1_PROBE_SIZE.0, AV1_PROBE_SIZE.1)?;
        let mut encoder = VaEncoder::at_constant_bitrate(&display, AV1_PROBE_SIZE, 1, 500_000)?;
        // Both stream-shaped settings have to be stated before `encode` will run, and neither is
        // defaulted: a rate or a GOP invented by a probe is a stream nobody asked for, which is the
        // whole argument for `hevc_available` not encoding a picture at all.
        encoder.at_rate(AV1_PROBE_FPS)?.key_frames_every(AV1_PROBE_FPS)?;
        encoder.upload(&luma, &chroma)?;
        let picture = encoder.encode(EncodeRequest::key_frame_at_rate()?)?;
        if picture.bytes.is_empty() {
            return Err("the H.264 control rendered and synced but wrote no bytes into the coded buffer".into());
        }
        Ok(picture.bytes.len())
    }

    /// **AV1 encode is not available on this machine, and it is not available for the reason the
    /// criterion names** — the driver refuses the picture record — **which an H.264 picture in this
    /// same process rules out as a fact about the machine.**
    ///
    /// The deliberate breaks, one per assertion: a probe that stopped at `open_display` (its refusal
    /// would name the display rather than `vaRenderPicture`); a probe that stopped at `vaCreateConfig`
    /// (the negation below); a probe whose `OnceLock` cached something other than its own result; and
    /// a driver that began taking the AV1 picture record (the `Ok` arm, which is a red test and not a
    /// pass, because the recorded measurement would then be stale).
    #[test]
    fn the_probe_refuses_at_the_picture_record_while_an_h264_picture_comes_back() {
        let _guard = lock();
        // **The control runs first and it gates everything else.** Without a working H.264 encoder in
        // this process, "AV1 is unavailable" is indistinguishable from "this machine cannot do
        // anything through VA-API", and a green tick would mean nothing — so the skip is loud and
        // names what could not be established.
        let control = match h264_control() {
            Ok(bytes) => bytes,
            Err(why) => {
                eprintln!("SKIPPED: no working H.264 control in this process ({why}), so the AV1 probe's answer is not evidence here");
                return;
            }
        };
        eprintln!("the H.264 control: {control} bytes of coded picture came back in this process");

        let probed = av1_probe();
        // **The cached answer is this probe**, so the assertions below are about the function the
        // format list would call. A probe that cached something else would make them claims about the
        // wrong function — which is a real possibility and not a hypothetical one, because the cache
        // is a `OnceLock` written in one place and read in another.
        assert_eq!(av1_available(), probed.is_ok(), "av1_available() is the probe's own result, memoised: the probe returned {probed:?}");
        match &probed {
            // **A red test and not a pass**: a driver that began taking the AV1 picture record makes
            // `vaapi::driver_outcomes::SUMMARY` stale, and the recorded refusal and this probe's
            // criterion then have to be revisited together rather than one of them quietly winning.
            Ok(bytes) => {
                panic!("FilmCraft's own AV1 submission path produced {bytes} bytes on this driver, which contradicts {DRIVER_REFUSES_THE_AV1_PICTURE_RECORD}")
            }
            Err(refusal) => {
                // **Where it failed is the claim.** "The probe said no" would pass on a machine with
                // no GPU; "the driver took the records and refused the picture record" cannot.
                assert!(refusal.contains("vaRenderPicture"), "the probe got as far as rendering and was refused there, which is the measured state: {refusal}");
                assert!(
                    !refusal.contains("vaCreateConfig"),
                    "and it did not fail while creating the session, or this would be a fact about the machine rather than about AV1: {refusal}"
                );
                // **The control and the probe in one assertion**, because neither half means anything
                // alone.
                assert!(
                    !av1_available(),
                    "an H.264 picture came back ({control} bytes) from this driver in this process, so VA-API works here, and AV1 was refused at the picture record: {refusal}"
                );
                // **Printed, because the numbers are the evidence** and a reader running this test
                // should not have to take the two counts above on trust.
                eprintln!("the AV1 probe's own answer: {refusal}");
                eprintln!("{DRIVER_REFUSES_THE_AV1_PICTURE_RECORD}");
            }
        }
    }

    /// **The advertisement and the session are both yes while the probe is no** — which is the whole
    /// reason this probe is not built on either of them.
    ///
    /// Every cheap question a capability probe would ask has a `true` here:
    ///
    /// - `vaQueryConfigEntrypoints(VAProfileAV1Profile0)` answers `[1 VLD, 6 EncSlice]`, so
    ///   `EncSlice` is offered (`/usr/include/va/va.h:553`, `:558`);
    /// - `vaCreateConfig` / `vaCreateSurfaces` / `vaCreateContext` all return `VA_STATUS_SUCCESS`,
    ///   which [`Av1Encoder::at_constant_quantiser`] would refuse if they did not.
    ///
    /// **A probe built on either answer would put AV1 in the format list on a machine where every AV1
    /// export fails at its first picture.** This test is what holds the difference between
    /// "advertised" and "produces output", and the two are compared as two separate facts about the
    /// same display rather than asserted about each other, so it fails if either the driver or the
    /// criterion moves.
    #[test]
    fn the_driver_advertises_an_av1_encoder_and_creates_a_session_and_writes_no_picture() {
        let _guard = lock();
        let display = match open_display() {
            Ok(display) => display,
            Err(why) => {
                eprintln!("SKIPPED: no VA-API display: {why}");
                return;
            }
        };
        let entrypoints = display
            .entrypoints(Profile::Av1Main)
            .unwrap_or_else(|why| panic!("vaQueryConfigEntrypoints(VAProfileAV1Profile0), which the rest of this test is about: {why}"));
        assert!(
            entrypoints.contains(&Entrypoint::EncSlice),
            "this driver advertises VAEntrypointEncSlice for VAProfileAV1Profile0, which is what the *cheap* probe would answer and what this one must not"
        );
        let plan = Av1Plan::new(AV1_PROBE_SIZE.0, AV1_PROBE_SIZE.1, AV1_PROBE_FPS, AV1_PROBE_QINDEX)
            .unwrap_or_else(|why| panic!("a {AV1_PROBE_SIZE:?} AV1 plan at {AV1_PROBE_FPS} fps: {why}"));
        assert!(
            Av1Encoder::at_constant_quantiser(&display, plan).is_ok(),
            "vaCreateConfig / vaCreateSurfaces / vaCreateContext succeed for VAProfileAV1Profile0 here, which is the other cheap answer. If this fails, this driver is no longer the one {DRIVER_REFUSES_THE_AV1_PICTURE_RECORD} was measured on, and this test's premise is gone"
        );
        // **The contradiction this test exists for**: everything above is true, and the probe is false.
        assert!(
            !av1_available(),
            "the driver advertises VAEntrypointEncSlice for AV1 and creates an AV1 encoder session, and still no AV1 picture comes out of it -- so an availability probe must ask for a picture, not for an advertisement"
        );
    }
}
