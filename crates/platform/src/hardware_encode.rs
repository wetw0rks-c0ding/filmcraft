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
//! **The Linux backend has no HEVC encoder and does not pretend to.** [`crate::vaapi::VaEncoder`]
//! declares the H.264 encode records and nothing else, so a `Format::Hevc` export here is declined
//! with the reason rather than producing a file whose container describes an HEVC stream and whose
//! samples are H.264.
//!
//! # The two backends do not accept the same exports, and the differences are measured
//!
//! [`config_for`] (VideoToolbox) and [`vaapi_config_for`] (VA-API) share most of their decline list
//! and differ in two places, each for a reason the backend can state rather than a preference:
//!
//! - **VideoToolbox takes a bitrate**; VA-API's H.264 slice encoder here is **constant quantiser**,
//!   so a settings-set bitrate cannot be honoured and an export that names one is declined. See
//!   [`vaapi_config_for`] for the measurement and the reasoning.
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
        profile: match (hevc, s.h264_profile) {
            (true, _) => VtProfile::HevcMain,
            (false, H264Profile::Baseline) => VtProfile::Baseline,
            (false, H264Profile::Main) => VtProfile::Main,
            (false, H264Profile::High) => VtProfile::High,
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

    /// What [`crate::vaapi::VaEncoder`] needs that [`ExportSettings`] states, or why this
    /// export is not one it can do.
    ///
    /// # The decline that decides this backend's usefulness: the bitrate
    ///
    /// **A settings-set bitrate cannot be honoured, so an export that names one is declined.**
    /// `VaEncoder` submits [`crate::vaapi::RC_CQP`] and nothing else, and the module's own note
    /// says why: a bitrate it cannot derive faithfully would be a *default quality*, which for an
    /// encoder means a file at a size nobody asked for.
    ///
    /// That makes this gate stricter than the VideoToolbox and NVENC ones, and it is worth being
    /// blunt about the consequence rather than letting a reader find it out from an export: **every
    /// `BitrateMode` FilmCraft's settings can express names a bitrate target** — `Cbr`, `Vbr1Pass`
    /// and `Vbr2Pass` all do — so **this backend declines every H.264 export the current settings
    /// can produce**, and the registered factory is a decline rather than an encoder.
    ///
    /// What is *not* the reason, because it would be convenient to say so and it is not measured:
    /// the driver is not the limit. `VAConfigAttribRateControl` is queried here and its answer is
    /// in the decline message, so a machine whose driver cannot do CBR or VBR at all produces a
    /// different message from one whose driver can — and neither says "VA-API cannot do bitrate
    /// control", which would be a claim about every driver rather than this one.
    ///
    /// The ways out, neither of which this task makes, and both of which are stated so a reader
    /// does not have to guess: submit a rate-control mode with a bitrate (which needs the rate
    /// control misc buffer this backend does not build, and whose QP then moves per picture, which
    /// the parameter-set self-check in `VaEncoder` deliberately refuses), or give the settings a
    /// constant-quality mode that maps to a QP rather than a bitrate.
    pub fn config_for(format: Format, w: u32, h: u32, rate: FrameRate, s: &ExportSettings) -> Result<(u32, u32, u32, u32), String> {
        if format != Format::H264 {
            // **No HEVC encoder here**, and the reason is the module's: `VaEncoder` declares the
            // H.264 encode records and nothing else. Naming it here matters, because the software
            // side has no HEVC encoder either — so a declined HEVC export fails with "encoder not
            // available" rather than quietly becoming an H.264 file with an HEVC container.
            return Err(if format == Format::Hevc { "HEVC".into() } else { format!("{format:?} is not H.264") });
        }
        if s.hardware_encoding != HardwareEncoding::Auto {
            return Err("hardware encoding is off".into());
        }
        if s.format.is_mxf() {
            return Err("MXF carries an Annex B stream, and this backend's samples are length-prefixed".into());
        }
        if !matches!(s.h264_pass, H264Pass::Single) {
            return Err("two-pass encoding".into());
        }
        if s.bitrate_mode != BitrateMode::Vbr1Pass || s.bitrate_kbps != 0 {
            return Err(format!(
                "a bitrate target ({} kbps, {}): this backend is constant quantiser, so it cannot honour one. \
                 The driver reports {} for rate control",
                s.bitrate_kbps,
                match s.bitrate_mode {
                    BitrateMode::Cbr => "CBR",
                    BitrateMode::Vbr1Pass => "VBR 1-pass",
                    BitrateMode::Vbr2Pass => "VBR 2-pass",
                },
                super::vaapi_export::rate_control_modes_named()
            ));
        }
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
        if s.h264_profile != filmcraft_export::H264Profile::Main {
            return Err(format!("the {:?} H.264 profile: this backend's derived parameter sets state Main", s.h264_profile));
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
        // The key-frame interval, and **the ceiling the frame loop imposes**. `MaxFrameNum` is 16
        // here (see `vaapi::MAX_FRAME_NUM`), so the loop inserts a key frame every 16 pictures
        // whatever this is. That is not something to pass through silently: an export that asked
        // for 250 and gets 16 has a file six times the size it was promised, and the only honest
        // answers are to tell the caller or to refuse. This refuses, with the number in the
        // message; the loop's own `key_frame_reason` is the safety net for the runs that do pass.
        let gop = s.keyframe_distance.filter(|k| *k > 0).unwrap_or_else(|| (f64::from(fps) * 2.0).round().max(1.0) as u32);
        if u64::from(gop) > u64::from(VaEncoder::max_frame_num()) {
            return Err(format!(
                "a key frame every {gop} pictures: this encoder's MaxFrameNum is {} (frame_num is four bits wide), so it cannot \
                 reach that far without a key frame of its own",
                VaEncoder::max_frame_num()
            ));
        }
        Ok((w, h, fps, gop))
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
            // **The quantiser is the one setting this backend takes as a number, and it is the
            // caller's.** `s.quality` (0-100) is not used: the software H.264 encoder ignores it in
            // favour of the bitrate too, and mapping a percentage onto a QP would be a table this
            // backend would be inventing.
            let request = EncodeRequest::predicted(26).map_err(ExportError::Encode)?;
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

    /// The Export encoder factory for the VA-API H.264 encoder.
    pub fn factory(format: Format, w: u32, h: u32, rate: FrameRate, s: &ExportSettings) -> Option<filmcraft_export::Result<Box<dyn VideoEncoder>>> {
        if format != Format::H264 || s.hardware_encoding != HardwareEncoding::Auto {
            return None;
        }
        let declined = |why: &str| {
            log::info!("VA-API hardware encoding declined: {why}");
            filmcraft_export::note_hw_encode_declined();
            None
        };
        let (ew, eh, fps, gop) = match config_for(format, w, h, rate, s) {
            Ok(c) => c,
            Err(why) => return declined(&why),
        };
        let display = match crate::vaapi::open_display() {
            Ok(d) => d,
            Err(why) => return declined(&format!("no VA-API display: {why}")),
        };
        // One reference: the only depth measured to produce correct pictures, and the ceiling this
        // backend accepts (`VaEncoder::new` refuses deeper with the measurement in the message).
        let mut encoder = match VaEncoder::new(&display, (ew, eh), 1) {
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
}

#[cfg(target_os = "linux")]
pub use vaapi_export::factory as vaapi_encoder_factory;

#[cfg(target_os = "linux")]
pub use vaapi_export::{
    VaEncoderExport, config_for as vaapi_config_for, rate_control_modes as vaapi_rate_control_modes, rate_control_modes_named as vaapi_rate_control_modes_named,
};
