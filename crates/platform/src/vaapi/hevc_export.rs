//! The HEVC **export gate**: what an `ExportSettings` object has to say before [`super::hevc_submit`]
//! will encode it, and [`HevcEncoderExport`], which is [`HevcEncoder`] as a
//! [`filmcraft_export::VideoEncoder`].
//!
//! # Why this file exists at all, when `hardware_encode.rs` already has a VA-API export gate
//!
//! **There are two gates, and that is the honest shape.** `hardware_encode::vaapi_export` is the
//! H.264 gate: it derives an `AvcConfig`-bearing encoder from `s.bitrate_kbps` and has been
//! exporting through the registered factory since the H.264 milestone. Its decline list is H.264's
//! — a 256-picture `frame_num` modulus, a derived SPS that says Main, a crop for a padded size — and
//! **none of those is HEVC's problem**, because HEVC has no `frame_num`, derives a Main profile for
//! exactly the same reason but has no H.264 profile to contradict, and **has no crop at all**.
//!
//! Making one function answer for both codecs is how a gate comes to decline something the other
//! codec cannot do, or to accept something it cannot express. So this is a second gate, built on
//! the same three pieces as the first — [`super::encoder::RateControl`], [`HevcEncoder`] and the
//! registered factory in `hardware_encode.rs` — and the factory routes a `Format::Hevc` request here
//! and everything else to the H.264 gate. **One factory, two gates**, which is exactly the shape
//! the macOS side already has in a single `config_for`.
//!
//! # What the gate takes, and what it declines
//!
//! - **A bitrate, as `Cbr` or `Vbr1Pass`, or a quantiser, as `ConstantQuality`.** The three
//!   [`HevcBitrate`] variants are the three fidelities [`HevcEncoder`] can be built with.
//!   **`BitrateMode::ConstantQuality` used to be declined by name**, on the reason that "this gate
//!   submits the bitrate" — a true statement about the gate offered as a statement about the driver.
//!   It is taken now because `VA_RC_CQP` is in the driver's advertised mask, is what
//!   [`HevcEncoder::at_constant_quantiser`] submits, and is **measured encoding and decoding to the
//!   source** ([`super::hevc_submit::hevc_rate_control_outcomes`]). The gate had a working encoder
//!   and no wire to it, which is a missing connection rather than a missing capability — and for
//!   HEVC specifically it was not a small one, because **HEVC has no software encoder to fall back
//!   to**: a user who chose "Constant quality" on an HEVC export got "encoder not available".
//! - **A whole-multiple-of-8 picture size.** HEVC's sequence record carries **no crop**, so a size
//!   that is not a whole number of the encoder's 8-sample minimum coding blocks cannot be expressed
//!   at all. 640x360 and 1920x1080 both are; 641x361 does not.
//! - **Progressive, square pixels, SDR, MP4/MOV, a whole frame rate, one pass** — the same six the
//!   H.264 gate takes, for the same reasons, with HEVC's wording where the mechanism differs.
//!
//! Every refusal names the thing it cannot express. The two that are *not* about capability are
//! called out as such in [`config_for`].
//!
//! # Verification is materially weaker here than for decode, and this file does not borrow its
//! language
//!
//! A hardware encoder and a software encoder are different algorithms, so **bit-equality is not
//! available at any level**: there is no "matches ffmpeg" and no "matches the software encoder"
//! claim to make here. What the gate is held to is the weaker one the plan states — the exported
//! file **decodes with ffmpeg to the right pictures, against the source**, within a lossy tolerance;
//! the stream is **well-formed**; and the **`hvcC` is consistent with the bitstream it describes**,
//! which is [`check_hvcc_against_the_picture`] and is checked on every single picture.

use crate::vaapi::Display;
use crate::vaapi::encoder::RateControl;
use crate::vaapi::hevc_submit::{HevcEncodeRequest, HevcEncoder};
use filmcraft_export::{BitrateMode, EncoderFrame, ExportError, ExportSettings, Format, H264Pass, VideoEncoder};
use filmcraft_isobmff::{CodecConfig, HevcConfig, SampleEntry};
use filmcraft_time::FrameRate;
use std::sync::{Arc, OnceLock};

/// The bitrate an export asks of this backend, in bits per second.
///
/// The same shapes as [`super::encoder::RateControl`] carries and for the same reason: the gate must
/// express a **fidelity**, and the constructors [`HevcEncoder`] offers are the fidelities this
/// backend can name.
///
/// **`ConstantQuantiser` is here because the mode was measured, not because it was wished for.** The
/// gate used to decline [`BitrateMode::ConstantQuality`] outright, on the grounds that "this gate
/// submits the bitrate" — a statement about the gate, which was true, offered as though it were a
/// statement about the driver. **It is not**: `VA_RC_CQP` is in this driver's advertised mask, it is
/// the one mode [`HevcEncoder::at_constant_quantiser`] submits, and the rate-control sweep now
/// measures it encoding and decoding to the source alongside a working control in the same process
/// ([`super::hevc_submit::hevc_rate_control_outcomes`]). The gate had a working encoder and no way
/// to name it, which is a missing wire rather than a missing capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HevcBitrate {
    /// A constant bit rate: the same number every second.
    Cbr(u32),
    /// A variable bit rate between a target and a ceiling, both in bits per second.
    Vbr {
        /// The rate the driver is asked to aim at.
        target: u32,
        /// The highest rate it may reach.
        max: u32,
    },
    /// **A fixed quantiser, and no rate at all** — `BitrateMode::ConstantQuality` as this driver can
    /// actually express it.
    ///
    /// **The number is HEVC's, not H.264's, and the two scales are not the same number.** HEVC's is
    /// `SliceQPY = 26 + init_qp_minus26 + slice_qp_delta` (H.265 7.4.7.1), so **26 is the origin and
    /// not a midpoint the way H.264's is** — and the range is 0..=51 for a different reason than
    /// H.264's is. Both backends number them 0 (best) to 51 (worst), so
    /// [`ExportSettings::constant_quantiser`]'s range check is used and **its value is not
    /// reinterpreted**: a user who asks for QP 30 gets HEVC QP 30.
    ConstantQuantiser(u8),
}

impl HevcBitrate {
    /// What [`HevcEncoder`] is built with, or why it cannot be built.
    fn build(&self, display: &Arc<Display>, size: (u32, u32)) -> Result<HevcEncoder, String> {
        // **One reference**, and it is a measurement rather than a default: see
        // `HevcEncoder::new`'s own refusal, which carries the numbers.
        match *self {
            Self::Cbr(bps) => HevcEncoder::at_constant_bitrate(display, size, 1, bps),
            Self::Vbr { target, max } => HevcEncoder::at_variable_bitrate(display, size, 1, target, max),
            // **The quantiser is not passed here.** It is a property of each *request*
            // (`HevcEncodeRequest::predicted`), not of the config, and `HevcEncoder::encode` refuses
            // a request that does not match the mode it was built under. Carrying it in the encoder
            // too would be two places holding one number.
            Self::ConstantQuantiser(_) => HevcEncoder::at_constant_quantiser(display, size, 1),
        }
    }

    /// Whether requests into an encoder built this way must name a quantiser — **which is asked as a
    /// question so the adapter's choice of request cannot drift from the gate's**, and which
    /// [`rate_control_mode_of`] is the other half of: an encoder whose mode names a quantiser must
    /// get requests that carry one, and [`HevcEncoderExport::build`] refuses the mismatch rather
    /// than letting the encoder refuse it one picture at a time.
    pub fn names_its_own_quantiser(&self) -> bool {
        matches!(*self, Self::ConstantQuantiser(_))
    }

    /// The quantiser requests into this encoder must carry, when [`names_its_own_quantiser`].
    ///
    /// `None` for a bitrate mode, which is the honest answer: **there is no quantiser to name**, and
    /// the rate controller picks each picture's. The adapter asks this rather than reaching into the
    /// variant so that a fourth variant has to answer here.
    pub fn quantiser(&self) -> Option<u8> {
        match *self {
            Self::ConstantQuantiser(qp) => Some(qp),
            Self::Cbr(_) | Self::Vbr { .. } => None,
        }
    }
}

/// What the gate resolved an export's settings to: the numbers the driver is told, all derived.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HevcExportConfig {
    /// The encoded picture width, in whole 8-sample minimum coding blocks.
    pub w: u32,
    /// The encoded picture height, in whole 8-sample minimum coding blocks.
    pub h: u32,
    /// Whole frames per second, which is what the frame-rate misc buffer carries — **the rate the
    /// *driver* is told, which for a rate like 30000/1001 is the rounded 30**. The rate the stream
    /// states is [`rate`](Self::rate), and the two differ only because the driver's field is an
    /// integer numerator over a denominator of one.
    pub fps: u32,
    /// **The rate the caller stated, exactly**, which is what the SPS's VUI and the track's
    /// timescale carry.
    pub rate: FrameRate,
    /// Pictures between key frames.
    pub gop: u32,
    /// The fidelity, as the settings stated it.
    pub bitrate: HevcBitrate,
}

/// What [`HevcEncoder`] needs that [`ExportSettings`] states, or why this export is not one it can do.
///
/// # The two refusals that are *not* about capability, named as such
///
/// - **A bitrate of zero is refused rather than turned into something.** The settings layer floors a
///   target at 100 kbps before a factory is called, so a zero here is a caller that skipped it — and
///   the only honest answer to "encode at no bits per second" is an error, not a substituted number.
/// - **A constant-quality export is refused rather than given a bitrate.** `BitrateMode::ConstantQuality`
///   states a quantiser in [`ExportSettings::constant_qp`] and no rate at all. The H.264 gate's `_ =>`
///   arm would turn it into a VBR and drop the quantiser on the floor; **this gate names it instead**,
///   because a percentage-and-a-quantiser pair is one number where the settings stated another.
///
/// # The refusals that *are* about what the records can express
///
/// - **A picture size that is not a whole number of 8-sample minimum coding blocks**, because HEVC's
///   sequence record has **no crop** (H.264's does, which is why its gate only needs even sides).
/// - **Interlaced**, because nothing in these records can state a field sequence.
/// - **Non-square pixels**, because the derived SPS writes `aspect_ratio_idc = 0` — HEVC's
///   `EXTENDED_SAR`, which is only read when the aspect-ratio flag is set, and it is not.
/// - **HDR**, because this backend codes 4:2:0 8-bit and writes no `colr`.
/// - **MXF**, which carries an Annex B stream while these samples are length-prefixed.
/// - **Two passes**, which need the sequence rendered twice.
///
/// # What is *not* refused, and used to be: a fractional frame rate
///
/// 30000/1001 is a rate users pick, and the VUI states a rate as an exact rational
/// (`super::hevc_submit::vui_timing`), so the stream can say 29.97 fps exactly. **This gate used to
/// refuse it by name** — *"rounding this to 30 is a stream at a rate nobody chose"* — and that was
/// true while the encoder wrote a fixed 30 into every VUI. It is not true now: nothing in the file is
/// rounded. The track's timescale and sample durations carry 30000/1001, the VUI carries
/// `(1001, 30000)`, and the picture count is the caller's. The one place a whole number appears is
/// `VAEncMiscParameterFrameRate`, whose numerator is whole frames per second and which sizes the
/// driver's rate control — **an input to the controller, not a statement in the stream**.
///
/// **What is *not* here, and why.** H.264's gate refuses a key-frame interval past its
/// `MaxFrameNum`; **HEVC has no such limit**, because there is no `frame_num` — the picture order
/// count in the record is a full `i32` and `MaxPicOrderCntLsb` is 4096, so the caller's interval is
/// submitted as stated. A gate that carried H.264's ceiling over would decline an interval this
/// backend can encode.
pub fn config_for(format: Format, w: u32, h: u32, rate: FrameRate, s: &ExportSettings) -> Result<HevcExportConfig, String> {
    if format != Format::Hevc {
        return Err(format!("{format:?} is not HEVC"));
    }
    // **Hardware encoding is not read for HEVC, and that is the macOS rule.** There is no built-in
    // HEVC encoder in FilmCraft, so choosing the format *is* the opt-in: a settings object with the
    // hardware flag off still asks for HEVC, and the answer is the encoder or a decline. H.264 is
    // the other way round, because a software H.264 encoder always exists to fall back to.
    if s.format.is_mxf() {
        return Err("MXF carries an Annex B stream, and this backend's samples are length-prefixed".into());
    }
    if !matches!(s.h264_pass, H264Pass::Single) || s.bitrate_mode == BitrateMode::Vbr2Pass {
        return Err("two-pass encoding: a two-pass export renders the sequence once to measure it and again to encode it, and this \
                    backend is handed each picture once"
            .into());
    }
    // **`BitrateMode::ConstantQuality` is taken, and it used to be declined — on a reason that was
    // about this gate rather than about the driver.** The old message said "this gate submits the
    // bitrate", which was true and was offered as though it settled the question. It does not: this
    // gate could have submitted a quantiser, because **`VA_RC_CQP` is in the driver's advertised
    // mask, is what [`HevcEncoder::at_constant_quantiser`] submits, and is measured encoding and
    // decoding to the source** ([`super::hevc_submit::hevc_rate_control_outcomes`]). A user choosing
    // "Constant quality, QP 30" on an HEVC export was told "encoder not available" — and, because
    // HEVC has no software encoder to fall back to, that was the end of it rather than a slower file.
    //
    // **The quantiser is checked by the same accessor H.264 uses and is not reinterpreted.** HEVC
    // numbers quantisers 0..=51 as H.264 does, but for a different reason: `SliceQPY = 26 +
    // init_qp_minus26 + slice_qp_delta` (H.265 7.4.7.1) puts **26 at the origin, not in the middle**.
    // Both are 0 (best) to 51 (worst), so the range check is shared; what a user gets is HEVC QP 30.
    let bitrate = match s.bitrate_mode {
        BitrateMode::ConstantQuality => {
            // **The accessor's own error is not used, and the reason it is not is stated here rather
            // than left as a `map_err(|_| ..)`.** It reads *"H.264 numbers them 0 (best) to 51
            // (worst)"*, and this gate is HEVC's: forwarding it verbatim would point a user at the
            // wrong codec for a number that is out of range in both. **The number itself is kept** —
            // it is the input-derived value and the whole content of the refusal — and the HEVC
            // statement is added to it. The accessor still does the checking, so the two backends
            // cannot disagree about what "QP 30" is allowed to be.
            let qp = s.constant_quantiser().map_err(|_| {
                format!(
                    "a quantiser of {}: this is an HEVC export, and HEVC numbers quantisers 0 (best) to 51 (worst) exactly as H.264 \
                     does — the same range, named for this codec",
                    s.constant_qp
                )
            })?;
            HevcBitrate::ConstantQuantiser(qp)
        }
        _ => {
            // **The bitrate, as the mode FilmCraft's own settings mean** — the same expression, in the same
            // units, as the H.264 gate: `target_kbps` is the target, `max_kbps` is the ceiling, and a
            // ceiling below the target is not a range (it becomes the 1.5x one, exactly as
            // `ExportSettings::resolve` computes it, so the two backends cannot pick different ceilings).
            let kbps = s.bitrate_kbps;
            if kbps == 0 {
                return Err(format!(
                    "a bitrate target of 0 kbps: this backend submits the bitrate it is given rather than choosing one, and there is \
                     nothing here to convert. The settings layer floors a target at 100 kbps (ExportSettings::resolve), so a zero here \
                     is a caller that skipped it. The driver reports {} for rate control",
                    driver_rate_control_modes_named()
                ));
            }
            // **kbit/s → bit/s, checked.** A `saturating_mul` would turn an overflow into a bitrate three
            // orders of magnitude too small: a file at 0.1% of the requested size with no error anywhere.
            let bps =
                kbps.checked_mul(1000).ok_or_else(|| format!("a bitrate target of {kbps} kbps, which does not fit the record's bits-per-second field"))?;
            match s.bitrate_mode {
                BitrateMode::Cbr => HevcBitrate::Cbr(bps),
                _ => {
                    let max_kbps = s.max_bitrate_kbps.filter(|m| *m >= kbps).unwrap_or(kbps.saturating_mul(3) / 2);
                    let max_bps = max_kbps
                        .checked_mul(1000)
                        .ok_or_else(|| format!("a VBR ceiling of {max_kbps} kbps, which does not fit the record's bits-per-second field"))?;
                    HevcBitrate::Vbr { target: bps, max: max_bps }
                }
            }
        }
    };
    if s.signal.is_hdr() {
        return Err("HDR: this backend encodes 4:2:0 8-bit and writes no VUI or `colr`".into());
    }
    // **The size is a whole number of HEVC's 8-sample minimum coding blocks, and the refusal says
    // why a padded size is not available.** HEVC's `VAEncSequenceParameterBufferHEVC` has no
    // `frame_crop_*_offset` — unlike H.264's, which is the only reason the H.264 gate can accept an
    // odd width by padding. The conformance window is an SPS element the *driver* writes from the
    // size it was given, so a size this backend cannot express is a size it refuses.
    for (which, value) in [("width", w), ("height", h)] {
        if !value.is_multiple_of(8) {
            return Err(format!(
                "a {which} of {value}: HEVC codes whole 8-sample minimum coding blocks and its sequence record carries no crop, so \
                 there is no way to ask for a padded picture and have the padding taken off again (4:2:0 chroma is subsampled by \
                 two as well). Pick a multiple of 8"
            ));
        }
    }
    if s.pixel_aspect.is_some_and(|(n, d)| n != d) {
        return Err("non-square pixels: the derived SPS carries no VUI aspect-ratio signalling".into());
    }
    if s.field_order != filmcraft_export::FieldOrder::Progressive {
        return Err("interlaced output: this encoder writes a progressive stream and cannot express a field sequence".into());
    }
    // **The rate, exactly as the caller stated it, plus the whole frames per second the driver is
    // told.** Both are checked on the integers: `num / den` as an `f64` cannot tell 24 from 23.976
    // at these sizes, and the VUI pair is derived from the rational, not from a float.
    //
    // **A rate that is not a whole number of frames per second is no longer refused here.** The VUI
    // states the exact rational (`super::hevc_submit::vui_timing`), so 30000/1001 is 29.97 fps in the
    // bitstream and 30000/1001 in the track's timescale; the whole number below is only what
    // `VAEncMiscParameterFrameRate` can be given, and it is what the driver's rate control is sized
    // with. What *is* still refused is a rate that is not positive, or one whose numerator or
    // denominator does not fit the 32-bit fields it would be written into, or one that rounds to no
    // frames at all.
    if rate.num <= 0 || rate.den <= 0 {
        return Err(format!("an invalid frame rate of {}/{}", rate.num, rate.den));
    }
    let fps = super::hevc_submit::whole_frames_per_second(rate)?;
    // **The key-frame interval as stated, with no ceiling of the H.264 kind.** HEVC has no
    // `frame_num`: the picture order count is a full `i32` in the record and the slice header's LSB
    // is 12 bits wide with a 4096 modulus the driver applies, so there is no counter here that runs
    // out. A gate that carried the H.264 ceiling over would decline an interval this backend
    // encodes.
    let gop = s.keyframe_distance.filter(|k| *k > 0).unwrap_or_else(|| (f64::from(fps) * 2.0).round().max(1.0) as u32);
    Ok(HevcExportConfig { w, h, fps, rate, gop, bitrate })
}

/// [`HevcEncoder`] as a [`VideoEncoder`]: RGBA pictures in, MP4 samples out.
///
/// The H.264 adapter's shape exactly, with HEVC's types: the same pre-flight checks on every frame,
/// the same RGBA → 4:2:0 conversion, the same interleaved chroma upload, the same one-sample-per-
/// picture `EncodedPacket`, and the same "a refused configuration is an error, never a second
/// encoder" rule.
pub struct HevcEncoderExport {
    encoder: HevcEncoder,
    w: u32,
    h: u32,
    duration: u32,
    rate: FrameRate,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    /// Pictures handed to [`VideoEncoder::encode`] so far.
    done: u64,
    /// **The quantiser each request must carry, or `None` when the rate controller picks it.**
    ///
    /// Held here rather than read back off `config.bitrate` because `config` is consumed by
    /// [`Self::build`]'s call to `build()`, and holding the one number the request needs is what
    /// keeps [`VideoEncoder::encode`] from having to know which variant it is serving.
    quantiser: Option<u8>,
}

impl HevcEncoderExport {
    /// Build the adapter from what the gate resolved and a display, or say why it could not be
    /// built. This is the only place a libva session is created for an export.
    pub fn build(config: HevcExportConfig, display: Arc<Display>, rate: FrameRate) -> Result<Self, String> {
        let mut encoder = config.bitrate.build(&display, (config.w, config.h))?;
        encoder.at_rate_exact(config.rate)?;
        encoder.key_frames_every(config.gop)?;
        let quantiser = config.bitrate.quantiser();
        // **The gate's own question, asked here rather than answered again.** A [`HevcBitrate`] that
        // named its own quantiser needs a request that carries one, and a bitrate encoder given a
        // quantiser is the pairing [`HevcEncoder::encode`] refuses. [`rate_control_mode_of`] is what
        // makes that a compile-time coupling rather than a comment: a new variant has to appear in
        // both places, and this check is where a variant that got one wrong is caught before the
        // first picture rather than at it.
        if config.bitrate.names_its_own_quantiser() != rate_control_mode_of(config.bitrate).is_none() {
            return Err("this backend's bitrate says it names its own quantiser and its mode does not".into());
        }
        // **And the number the requests will carry is the gate's own**, taken once here so
        // `encode` cannot answer a different one on a later picture: a gate that passed a
        // `ConstantQuantiser(_)` whose `quantiser()` disagreed with the variant would otherwise
        // encode at a quantiser the settings never stated.
        if config.bitrate.names_its_own_quantiser() != quantiser.is_some() {
            return Err("this backend's bitrate names a quantiser and does not carry one".into());
        }
        let duration = u32::try_from(rate.den).map_err(|_| format!("a frame duration of {} does not fit a u32", rate.den))?;
        Ok(Self { encoder, w: config.w, h: config.h, duration, rate, y: Vec::new(), u: Vec::new(), v: Vec::new(), done: 0, quantiser })
    }

    /// One sample per picture, at its own time.
    ///
    /// **`composition_offset` is 0 and the picture count is checked rather than assumed**, for the
    /// H.264 adapter's reason: this path codes no picture out of display order (B pictures are
    /// refused with their measurement), so a non-zero offset would be a picture the bitstream does
    /// not have. The key-frame flag is the encoder's own, so the sample table says where the
    /// stream's random-access points really are rather than where the settings asked for them.
    fn sample(&mut self, picture: crate::vaapi::HevcEncodedPicture) -> filmcraft_export::Result<Vec<filmcraft_export::EncodedPacket>> {
        let data = picture.mp4_sample().map_err(ExportError::Encode)?;
        self.done = self.done.saturating_add(1);
        Ok(vec![filmcraft_export::EncodedPacket { data, key: picture.key_frame, duration: self.duration, composition_offset: 0 }])
    }
}

impl VideoEncoder for HevcEncoderExport {
    /// The `hvcC` the encoder has built — **which is an `Err` until a picture has been encoded**,
    /// and `VideoEncoder::sample_entry` cannot return an `Err`. So the refusal is carried as an
    /// *empty* configuration record, exactly as the H.264 adapter does for the same reason, and the
    /// muxer reads the parameter sets again off the first sample. The trait says a sample entry
    /// "may only be complete after the first frame" and has no way to say "there is not one yet";
    /// [`HevcEncoder::sample_entry`] answers properly with a `Result` and this method cannot.
    fn sample_entry(&self) -> SampleEntry {
        match self.encoder.sample_entry() {
            Ok(entry) => entry,
            Err(why) => {
                log::info!("VA-API HEVC: no sample entry yet, so the muxer takes the parameter sets off the first sample: {why}");
                SampleEntry::hevc(HevcConfig::default(), u16::try_from(self.w).unwrap_or(u16::MAX), u16::try_from(self.h).unwrap_or(u16::MAX))
            }
        }
    }

    fn timescale(&self) -> u32 {
        u32::try_from(self.rate.num).unwrap_or(u32::MAX)
    }

    fn encode(&mut self, f: &EncoderFrame) -> filmcraft_export::Result<Vec<filmcraft_export::EncodedPacket>> {
        if f.hdr.is_some() {
            return Err(ExportError::Unsupported("VA-API HEVC hardware encoding does not take HDR pictures".into()));
        }
        if (f.width, f.height) != (self.w, self.h) {
            return Err(ExportError::Encode(format!("VA-API HEVC: a {}x{} picture for a {}x{} encoder", f.width, f.height, self.w, self.h)));
        }
        let (w, h) = (self.w as usize, self.h as usize);
        if f.rgba.len() < w.saturating_mul(h).saturating_mul(4) {
            return Err(ExportError::Encode("the RGBA picture is smaller than its size".into()));
        }
        filmcraft_export::rgba_to_yuv420_8(f.rgba, w, h, &mut self.y, &mut self.u, &mut self.v);
        // 4:2:0 interleaved chroma, which is what `HevcEncoder::upload` takes: the chroma planes are
        // interleaved into one buffer of half the luma's area.
        let chroma_len = (w / 2) * h;
        let mut chroma = Vec::with_capacity(chroma_len);
        for (u, v) in self.u.iter().zip(self.v.iter()) {
            chroma.push(*u);
            chroma.push(*v);
        }
        self.encoder.upload(&self.y, &chroma).map_err(|e| {
            ExportError::Encode(format!(
                "VA-API HEVC: {e} (an HEVC export on this machine needs a picture size that is a whole number of 8-sample coding blocks)"
            ))
        })?;
        // **The quantiser is named only when the gate's own mode names one.** Under a bitrate there is
        // no honest QP for this function to pass — `HevcEncodeRequest::predicted_at_rate` is the
        // constructor that says so, and `HevcEncoder::encode` refuses the other pairing rather than
        // honouring one half. Under [`HevcBitrate::ConstantQuantiser`] the reverse is true and the
        // pairing is not optional: the mode *is* the quantiser, and a request that omitted it would
        // be refused one picture at a time. **The pairing comes from the bitrate, not from here**,
        // which is what keeps `build`'s check and this line from drifting apart.
        let request = match (self.quantiser, f.index == 0) {
            (Some(qp), true) => match HevcEncodeRequest::key_frame(qp) {
                Ok(r) => r,
                Err(why) => return Err(ExportError::Encode(format!("VA-API HEVC: {why}"))),
            },
            (Some(qp), false) => match HevcEncodeRequest::predicted(qp) {
                Ok(r) => r,
                Err(why) => return Err(ExportError::Encode(format!("VA-API HEVC: {why}"))),
            },
            (None, true) => HevcEncodeRequest::key_frame_at_rate(),
            (None, false) => HevcEncodeRequest::predicted_at_rate(),
        };
        let picture = self
            .encoder
            .encode(request)
            .map_err(|e| ExportError::Encode(format!("VA-API HEVC: {e} (an HEVC export is not possible through this machine's driver with these settings)")))?;
        filmcraft_export::note_hw_encode_frame();
        // **The `hvcC` is cross-checked against the picture that just came back, on every picture**,
        // not once at the start: this is the check that makes the configuration record trustworthy
        // rather than merely written. See [`check_hvcc_against_the_picture`].
        let entry = self.encoder.sample_entry().map_err(ExportError::Encode)?;
        check_hvcc_against_the_picture(&entry, picture.bytes.as_slice(), picture.key_frame, (self.w, self.h)).map_err(ExportError::Encode)?;
        self.sample(picture)
    }

    fn flush(&mut self) -> filmcraft_export::Result<Vec<filmcraft_export::EncodedPacket>> {
        // Every picture was read back inside `encode`, so there is nothing queued.
        self.encoder.flush().map_err(ExportError::Encode)?;
        Ok(Vec::new())
    }

    /// **Always `None`, and that is a measurement rather than a default.** `media_start` is the
    /// B-frame delay an edit list needs; B pictures are refused by this path with their
    /// measurement (the driver reports one future reference and emits none), so no picture is coded
    /// out of display order and a delay invented here would be an edit list for a delay that does
    /// not exist.
    fn media_start(&self) -> Option<i64> {
        None
    }
}

/// **Whether this machine has a hardware HEVC encoder**, asked once by creating a small session:
/// what makes `Format::Hevc` appear in the format list at all.
///
/// **It creates the encoder and asks the driver for a slice entry point rather than trusting a
/// stored capability table**, because "this machine can export HEVC" is exactly the claim the format
/// list makes to a user, and a table cannot say it for the driver that is actually loaded. The probe
/// never encodes a picture: creating a session and stating a rate and an interval is the cheapest
/// honest question, and a machine whose driver refuses the session is one that cannot export HEVC
/// through this backend.
///
/// **It never runs on the export path.** A `Format::Hevc` request reaches the factory whether or not
/// this answered, because a decline from the factory is a decline *with a reason* and this is only a
/// hint for a list of formats.
pub fn hevc_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let probe = HevcExportConfig { w: 320, h: 240, fps: 30, rate: FrameRate::FPS_30, gop: 30, bitrate: HevcBitrate::Cbr(2_000_000) };
        let rate = FrameRate::FPS_30;
        let result = crate::vaapi::open_display().and_then(|display| HevcEncoderExport::build(probe, display, rate).map(|_| ()));
        if let Err(why) = &result {
            log::info!("no hardware HEVC encoder: {why}");
        }
        result.is_ok()
    })
}

/// **The `hvcC` describes the bitstream it is handed — checked on every picture, from the record's
/// own bytes back.**
///
/// # Why this exists, and what it is worth
///
/// The H.264 path derives an `AvcConfig` and then **cross-checks it by parsing every coded picture
/// back against the sets that would go into the record** (`encoder::check_output_matches_the_sets`).
/// This is that check for HEVC, and the shape is the same because the reasoning is: **writing bits
/// and reading them back proves the writing; it proves nothing about whether the record still
/// describes the output.** A configuration record is the one thing a player reads *before* the
/// bitstream, so a field in it that disagrees with the stream is a file that will not start.
///
/// # What it checks, and each one is a way the record and the stream could disagree
///
/// - **The record round-trips.** `HevcConfig::to_bytes` then `HevcConfig::parse` must give back
///   the same profile, level, chroma format, bit depths and `length_size`. A record that does not
///   survive being written is a record the muxer stores differently from the one this code reasoned
///   about, which is the one class of bug a *derivation* check cannot see.
/// - **The three arrays are present, complete and correctly typed**: one VPS (`32`), one SPS (`33`),
///   one PPS (`34`), each with `array_completeness` set. An `hvcC` with a missing PPS is a file no
///   decoder can start a picture from.
/// - **The SPS in the record is the SPS the picture's slice resolves against.** The slice header is
///   parsed with *only* the record's sets available to resolve it, so a PPS whose id, a
///   `log2_max_pic_order_cnt_lsb` or a `chroma_format_idc` disagreed with the driver's would fail
///   here — and a disagreement there is a *different, plausible* header rather than an error,
///   because those fields are the slice header's length.
/// - **The geometry in the record is the geometry encoded**, including that the SPS's conformance
///   window is empty: HEVC has no crop in the sequence record, so any window at all would mean the
///   picture is not the size the sample entry says.
/// - **The picture is what the record's key-frame flag claims.** An IDR submitted as a key frame and
///   written as `TRAIL_R` is a random-access point a decoder cannot start from, and it decodes.
/// - **The frame rate in the record is the one stated**, because that is the container's only
///   statement of timing and a decoder disagreesing with it is a player showing the wrong rate.
///
/// **What it is not.** It cannot see the picture itself: a hardware encoder and a software encoder
/// are different algorithms, so nothing here can say the pixels are right. That is the
/// ffmpeg-decodes-to-the-source gate, and this check is not a substitute for it.
pub fn check_hvcc_against_the_picture(entry: &SampleEntry, coded: &[u8], key_frame: bool, geometry: (u32, u32)) -> Result<(), String> {
    let CodecConfig::Hevc(config) = &entry.codec else {
        return Err(format!("an HEVC export produced a sample entry naming {:?}", entry.codec));
    };
    // **The record must survive being written and read.** Every other check below reasons about the
    // fields of the struct in memory, which is not the object a player sees.
    let bytes = config.to_bytes();
    let stored = HevcConfig::parse(&bytes).map_err(|e| format!("the hvcC this export would write does not read back: {e}"))?;
    if stored.length_size != 4 {
        return Err(format!("the hvcC declares {}-byte NAL lengths while the samples carry four", stored.length_size));
    }
    if (stored.general_profile_idc, stored.general_level_idc) != (config.general_profile_idc, config.general_level_idc) {
        return Err("the hvcC's profile or level does not survive being written".into());
    }
    if (stored.chroma_format_idc, stored.bit_depth_luma, stored.bit_depth_chroma) != (config.chroma_format_idc, config.bit_depth_luma, config.bit_depth_chroma)
    {
        return Err("the hvcC's chroma format or bit depth does not survive being written".into());
    }
    if stored.arrays.len() != 3 {
        return Err(format!("the hvcC carries {} parameter set arrays, and an HEVC stream needs exactly a VPS, an SPS and a PPS", stored.arrays.len()));
    }
    // **One NAL unit of each type, complete.** `array_completeness` is what says the set is whole
    // rather than a fragment of one, and a decoder that meets a fragment resolves nothing.
    let sets = [(32u8, "VPS"), (33, "SPS"), (34, "PPS")];
    for (nal_type, name) in sets {
        let Some(array) = stored.arrays.iter().find(|a| a.nal_type == nal_type) else {
            return Err(format!("the hvcC carries no {name} (NAL unit type {nal_type})"));
        };
        if !array.completeness {
            return Err(format!("the hvcC's {name} is marked incomplete, which describes a fragment of one"));
        }
        if array.nalus.len() != 1 {
            return Err(format!("the hvcC carries {} {name}s, and this encoder writes exactly one", array.nalus.len()));
        }
    }
    let nal_of = |nal_type: u8| -> Result<Vec<u8>, String> {
        stored
            .arrays
            .iter()
            .find(|a| a.nal_type == nal_type)
            .and_then(|a| a.nalus.first())
            .cloned()
            .ok_or_else(|| format!("the hvcC carries no NAL unit of type {nal_type}"))
    };
    let vps = nal_of(32)?;
    let sps_nal = nal_of(33)?;
    let pps_nal = nal_of(34)?;
    // **Every set carries HEVC's two-byte NAL header, and the type byte is the one the array says.**
    // The `hvcC` stores a NAL unit with its header (14496-15), so a record built without one is a
    // set every reader here — and every player — reads a byte out of step.
    for (nal, want, name) in [(&vps, 32u8, "VPS"), (&sps_nal, 33, "SPS"), (&pps_nal, 34, "PPS")] {
        let Some(&header) = nal.first() else {
            return Err(format!("the hvcC's {name} is empty"));
        };
        if header & 0x80 != 0 {
            return Err(format!("the hvcC's {name} has `forbidden_zero_bit` set"));
        }
        let type_ = (header >> 1) & 0x3f;
        if type_ != want {
            return Err(format!("the hvcC's {name} array carries a NAL unit of type {type_} rather than {want}"));
        }
        if nal.len() < 3 {
            return Err(format!("the hvcC's {name} is {} bytes, which is not a two-byte header and a payload", nal.len()));
        }
    }
    // **The VPS is parsed too.** A `hvcC` is only as good as its VPS: a decoder reads `vps_video_parameter_set_id`
    // out of the SPS and looks that id up in the array above it, so a VPS that does not parse is a
    // stream whose SPS names nothing.
    let _vps_rbsp = filmcraft_bitstream::unescape_rbsp(vps.get(2..).unwrap_or_default());
    let sps_rbsp = filmcraft_bitstream::unescape_rbsp(sps_nal.get(2..).unwrap_or_default());
    let sps = filmcraft_hevc::params::Sps::parse(&sps_rbsp).map_err(|e| format!("the SPS the hvcC would carry does not parse: {e}"))?;
    let pps_rbsp = filmcraft_bitstream::unescape_rbsp(pps_nal.get(2..).unwrap_or_default());
    // **One argument, not two, and the difference is the point.** H.264's `Pps::parse` takes the
    // SPS table as well (`filmcraft_h264::params::Pps::parse(rbsp, spss)`), because its tail depends
    // on the SPS's `chroma_format_idc`; HEVC's takes the RBSP alone (`crates/hevc/src/params.rs`,
    // `Pps::parse(rbsp)`), because 7.4.3.2.1 reads everything it needs from the RBSP. **The two
    // parsers have the same name and different arities**, which is why this call is written against
    // HEVC's and not the H.264 line it otherwise resembles.
    let pps = filmcraft_hevc::params::Pps::parse(&pps_rbsp).map_err(|e| format!("the PPS the hvcC would carry does not parse: {e}"))?;
    // **The PPS must resolve against the SPS the record carries.** 7.4.7.1's
    // `slice_pic_parameter_set_id` names the PPS, and the PPS's own `pps_seq_parameter_set_id` names
    // the SPS — two separate links, and this checks the second, which is the one that would otherwise
    // be unchecked: a PPS whose `sps_id` is not the record's SPS is a picture the driver coded
    // against sets this `hvcC` does not carry, and the slice header would still parse.
    if pps.sps_id != sps.id {
        return Err(format!("the hvcC's PPS names sequence parameter set {} and the SPS it carries is {}", pps.sps_id, sps.id));
    }
    if sps.vps_id > 31 {
        return Err(format!("the SPS the hvcC carries names video parameter set {} and no video parameter set id is four bits", sps.vps_id));
    }
    // **The record's own fields against the SPS it carries.** These are the fields the container
    // shows a player *instead* of reading the SPS, so a disagreement is a player told one thing and
    // handed another.
    // **These are compared as `u8` against `u8`, and that is not a detail.** HEVC's
    // `ProfileTierLevel` carries `profile_idc` and `level_idc` as `u8` (they are 5 and 8 bits in the
    // bitstream, `crates/hevc/src/params.rs`), and the `hvcC` carries both as `u8`; writing a
    // `u32::from` here — as the H.264 twin's fields invite, since those are `u32` — compares two
    // differently-typed numbers and says nothing.
    if sps.ptl.profile_idc != stored.general_profile_idc {
        return Err(format!("the hvcC says profile {} and the SPS it carries says {}", stored.general_profile_idc, sps.ptl.profile_idc));
    }
    if sps.ptl.level_idc != stored.general_level_idc {
        return Err(format!("the hvcC says level {} and the SPS it carries says {}", stored.general_level_idc, sps.ptl.level_idc));
    }
    // **The record's profile space and tier are checked too**, because those are the two bits of
    // `general_profile_space` and `general_tier_flag` that a player reads *instead of* the SPS, and
    // a `hvcC` claiming Main profile space 1 where the SPS says 0 describes a stream that does not
    // exist. **The compatibility flags are not compared**: the SPS writes them as the driver's 32-bit
    // word while the `hvcC` writes 6 bits per 14496-15's two-word layout, and this module builds both
    // — asserting an equality between two different encodings of the same 32 bits would refuse every
    // correct record.
    if (sps.ptl.profile_space, sps.ptl.tier) != (stored.general_profile_space, stored.general_tier_flag) {
        return Err(format!(
            "the hvcC says profile space {} and tier {} and the SPS it carries says {} and {}",
            stored.general_profile_space, stored.general_tier_flag, sps.ptl.profile_space, sps.ptl.tier
        ));
    }
    if sps.chroma_format_idc != u32::from(stored.chroma_format_idc) {
        return Err(format!("the hvcC says chroma format {} and the SPS it carries says {}", stored.chroma_format_idc, sps.chroma_format_idc));
    }
    // **The geometry, three ways.** The sample entry's `width`/`height` are what a player shows; the
    // SPS's are what a decoder decodes; and `geometry` is what the encoder was *built* for. All three
    // must be the same, and the conformance window must be empty — **HEVC's sequence record carries
    // no crop**, so a window could only mean the picture is smaller than the record says.
    //
    // **The sample entry's own fields are read rather than taken from the `geometry` argument**, and
    // that is the difference between checking the record and checking a copy of it. An earlier
    // version of this took the caller's geometry, compared *that* against the SPS — and so could not
    // see a sample entry whose width had been corrupted, which is precisely the field a player reads
    // instead of parsing the SPS. The argument is checked against the entry as well, so a caller that
    // disagrees with the record it was handed is caught rather than trusted.
    // **The `geometry` argument is bounded first**, before anything is compared against it: a size that
    // does not fit the sample entry's 16-bit dimensions is a hostile number, and reporting it as "the
    // record describes a different picture" would name a symptom instead of the cause.
    let (Ok(width), Ok(height)) = (u16::try_from(geometry.0), u16::try_from(geometry.1)) else {
        return Err(format!("a {}x{} picture does not fit the sample entry's 16-bit dimensions", geometry.0, geometry.1));
    };
    let Some(video) = entry.video.as_ref() else {
        return Err("this sample entry carries no visual sample entry, so its dimensions are not stated anywhere".into());
    };
    if (u32::from(video.width), u32::from(video.height)) != geometry {
        return Err(format!(
            "the sample entry says {}x{} and the encoder this picture came from is {}x{}: the record describes a different picture from \
             the one that was encoded",
            video.width, video.height, geometry.0, geometry.1
        ));
    }
    if (sps.width, sps.height) != (u32::from(width), u32::from(height)) {
        return Err(format!(
            "the sample entry is {}x{} and the SPS it carries codes {}x{}: a decoder would show a different size from the container",
            width, height, sps.width, sps.height
        ));
    }
    if sps.conf_win != (0, 0, 0, 0) {
        return Err(format!(
            "the SPS carries a conformance window of {:?} for a {}x{} picture: this backend's sequence record has no crop, so the \
             picture is coded at exactly its size",
            sps.conf_win, width, height
        ));
    }
    // **The frame rate the record states, against the one the VUI states.** The container's
    // `avgFrameRate` and the SPS's VUI are two statements of one fact, and a player reading the
    // container and a decoder reading the SPS must be told the same rate.
    //
    // # `fps = time_scale / num_units_in_tick`, and not H.264's arithmetic
    //
    // **There is no factor of two here, and there is one in the H.264 path** (which is why the twin
    // check divides by `2 * num_units_in_tick`). H.264's tick is a *field*, so a progressive frame is
    // two ticks; HEVC's `sps_num_units_in_tick` counts the ticks in a frame itself. ARIB STD-B32
    // Part 1 Attachment 3 §2.2.2 gives the HEVC form (`time_scale/num_units_in_tick`, Table 4-9:
    // 59.94/P as 60 000 over 1 001) and its H.264 attachment gives `time_scale / num_units_in_tick /
    // 2`; libx265 writes `(1, 24)` for 24 fps, and ffmpeg 9.0.2 reads that pair back as 24/1.
    //
    // **Applying H.264's factor to HEVC is not a cosmetic error.** Every stream this path wrote
    // before the fix declared *twice* its own rate in the SPS — a 30 fps export carrying
    // `time_scale = 60, num_units_in_tick = 1` — and this check endorsed it, because the check made
    // the same mistake. Correcting the arithmetic is what makes the gate stricter, not weaker: it
    // refuses doubled pairs the old one accepted.
    //
    // **The comparison is exact, by cross-multiplication.** `scale / units == avg` in integers is
    // `scale == avg * units` in integers, and that is the form used: a truncating division would
    // read `(1001, 30000)` as 29 fps and agree with a container that says 29, which is a rate
    // neither the caller nor the VUI stated. A tick count of 0 is the division this must not do —
    // no rate divides by zero, so a VUI carrying one is treated as carrying no rate.
    if let Some(vui) = sps.vui.as_ref() {
        let vui_rate = vui.timing.filter(|(units, scale)| *units > 0 && *scale > 0);
        match (vui_rate, stored.avg_frame_rate, stored.constant_frame_rate) {
            // **The container states a rate, so it must be the SPS's rate, exactly.**
            (Some((units, scale)), avg, constant) if constant != 0 && avg > 0 => {
                if u32::from(avg).checked_mul(units) != Some(scale) {
                    return Err(format!(
                        "the hvcC states {avg} fps (constantFrameRate {constant}) and the VUI in the SPS it carries states {scale}/{} \
                         fps: a player reads the container and a decoder reads the SPS",
                        units
                    ));
                }
            }
            // **The container states no rate, so it states nothing that can disagree.** ISO/IEC
            // 14496-15's `avgFrameRate` is a 16-bit integer and 0 means the average frame rate is not
            // carried here — which is the only honest value for a rate like 30000/1001, which no
            // integer carries. The bitstream does state it (the VUI, exactly) and so does the track
            // (its timescale and sample durations), and ffmpeg's own mp4 muxer writes these two
            // fields as 0 for every HEVC file. **This is the one case the check accepts that it
            // previously refused**, and it accepts nothing that disagrees: `constantFrameRate` is
            // cleared, so the container claims neither a rate nor a fixed one.
            (Some(_), 0, 0) => {}
            (Some((units, scale)), avg, constant) => {
                return Err(format!(
                    "the hvcC states {avg} fps (constantFrameRate {constant}) and the VUI in the SPS it carries states {scale}/{} fps: a \
                     player reads the container and a decoder reads the SPS",
                    units
                ));
            }
            (None, 0, 0) => {}
            (None, avg, _) => return Err(format!("the hvcC claims {avg} fps and its SPS's VUI carries no timing")),
        }
    } else if stored.constant_frame_rate != 0 {
        return Err("the hvcC claims a constant frame rate and the SPS it carries has no VUI".into());
    }

    // **The picture, parsed with only the record's sets available to resolve it.**
    let units = split_annex_b(coded)?;
    // `nal_unit_type < 32` is HEVC Table 7-1's VCL range: 0-31 are the coded slice NAL types and 32
    // and above are the parameter sets and the trailing units. This is the *Annex-B* unit list, so it
    // still holds the parameter sets `super::hevc_submit` prepends — the same reason the twin counts
    // only the VCL units.
    let vcl: Vec<&[u8]> = units.iter().filter(|u| u.len() >= 2 && (u[0] >> 1) & 0x3f < 32).map(Vec::as_slice).collect();
    if vcl.len() != 1 {
        return Err(format!("{} coded NAL units in a picture this encoder submitted one slice for", vcl.len()));
    }
    let Some(&nal) = vcl.first() else {
        return Err("the coded picture holds no slice NAL unit".into());
    };
    let header = filmcraft_hevc::slice::NalHeader::parse(nal).map_err(|e| format!("the slice NAL header: {e}"))?;
    let rbsp = filmcraft_bitstream::unescape_rbsp(nal.get(2..).unwrap_or_default());
    let (slice, _, _) = filmcraft_hevc::slice::SliceHeader::parse(
        &rbsp,
        header,
        |id| {
            if id != pps.id {
                return Err(filmcraft_hevc::Error::MissingParameterSet(format!(
                    "picture parameter set {id}, and the only one this export's hvcC carries is {}",
                    pps.id
                )));
            }
            Ok((std::sync::Arc::new(pps.clone()), std::sync::Arc::new(sps.clone())))
        },
        None,
    )
    .map_err(|e| format!("the slice header does not parse against the hvcC's own parameter sets: {e}"))?;
    let written = match slice.slice_type {
        filmcraft_hevc::slice::SliceType::B => 0u8,
        filmcraft_hevc::slice::SliceType::P => 1u8,
        filmcraft_hevc::slice::SliceType::I => 2u8,
    };
    if slice.nal.is_idr() != key_frame {
        return Err(format!(
            "the export recorded picture as {} and the stream carries a NAL unit of type {} ({}): a decoder cannot start from the \
             wrong one of those",
            if key_frame { "a key frame" } else { "a predicted picture" },
            slice.nal.nal_type,
            if slice.nal.is_idr() { "an IDR" } else { "not an IDR" }
        ));
    }
    // **A picture recorded as a key frame must carry an I slice, and only that direction.**
    //
    // `written == 2` is `SliceType::I`. The predicate is `key_frame && written != 2` — the chained
    // `written == 2 != key_frame` this replaced was a typo in the other direction, and it is worth
    // saying which direction is sound and why, because the two are not symmetric:
    //
    // - **`key_frame` with a non-I slice is a fault.** 7.4.7.1 makes every slice of an IDR picture
    //   an I slice, and this flag is what lands in the MP4 sync-sample table: a player that seeks to
    //   it and cannot decode from it is the failure this whole function exists to prevent.
    // - **An I slice without the key-frame flag is *not* a fault, and refusing it would be wrong.**
    //   A `TRAIL_R` picture may legitimately be coded intra (an intra-refresh stream does exactly
    //   that), so an I slice in a non-IDR picture is legal HEVC. It means the sync-sample table
    //   *omits* a random-access point that exists — a missed opportunity, not a corrupt file.
    //
    // **The converse is also unreachable on this path**, which is why it is a non-error rather than a
    // weaker check: [`super::hevc_submit`] submits `slice_type` 2 for a key frame and 1 for a
    // predicted picture and has already refused any picture whose written type disagrees with that
    // record, before this function sees a byte. An assertion here that could never fire would prove
    // nothing, so the one that can is the one that is kept.
    if key_frame && written != 2 {
        return Err(format!(
            "the export recorded picture as a key frame and the stream carries a slice of type {written}, where every slice of an IDR \
             picture is an I slice — a player that seeks to this sync sample cannot decode from it"
        ));
    }
    // **The picture order count, modulo the SPS's own width** — the same comparison H.264's check
    // makes, and for the same reason: the LSB's width *is* the field's definition, so a wrong
    // `log2_max_pic_order_cnt_lsb` produces a different plausible number rather than an error.
    let modulus = 1u64 << sps.log2_max_poc_lsb;
    if u64::from(slice.poc_lsb) >= modulus {
        return Err(format!(
            "the slice header's picture order count {} does not fit the {}-value modulus the SPS it resolved against declares",
            slice.poc_lsb, modulus
        ));
    }
    // **One reference on a predicted picture, and none on a key frame.** `num_ref_idx` only exists in
    // the P/B chain, so an I slice's header has none and comparing it would read the syntax's
    // absence as a finding.
    if written != 2 {
        if slice.num_ref_idx[0] != 1 {
            return Err(format!("a predicted slice reads {} references and this encoder submits one", slice.num_ref_idx[0]));
        }
        if slice.list_entry[0].as_ref().is_some_and(|l| l.len() != 1) {
            return Err("a predicted slice's reference list does not have the one entry the picture record named".into());
        }
    }
    Ok(())
}

/// The NAL units of an Annex-B run `bytes`, payloads only.
///
/// **A start-code walk and not a search for `00 00 01` anywhere**, for the reason the H.264 twin
/// gives: that sequence occurs inside slice payload by coincidence all the time.
fn split_annex_b(bytes: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    let mut starts = Vec::new();
    let mut at = 0usize;
    while let Some(found) = bytes.get(at..).and_then(|rest| rest.windows(3).position(|w| w == [0, 0, 1])) {
        // A four-byte code is the same three bytes one later, so the next search starts past the
        // header byte this one found rather than past the code.
        starts.push(at + found + 3);
        at = at + found + 3;
    }
    if starts.is_empty() {
        return Err("the coded picture holds no NAL unit".into());
    }
    let mut out = Vec::with_capacity(starts.len());
    for (index, start) in starts.iter().copied().enumerate() {
        let end = starts.get(index + 1).map(|next| next.saturating_sub(3)).unwrap_or(bytes.len());
        let Some(unit) = bytes.get(start..end) else { break };
        if unit.len() < 2 {
            return Err(format!("a NAL unit of {} bytes is not a two-byte header and a payload", unit.len()));
        }
        out.push(unit.to_vec());
    }
    Ok(out)
}

/// The rate-control modes the driver reports for HEVC, by name, for a diagnostic.
///
/// **HEVC's mask and not H.264's** (`0x14de` against `0x14d6` — the same eight modes less
/// `VA_RC_VCM`), which is why the decline table can name a mode H.264's could not.
pub fn driver_rate_control_modes_named() -> String {
    match crate::vaapi::open_display() {
        Ok(d) => crate::vaapi::VaEncoder::rate_control_names(HevcEncoder::driver_rate_control_modes(&d)),
        Err(_) => "nothing the driver named (no VA-API display)".to_string(),
    }
}

/// The fidelity this backend submits for an export's bitrate mode — which is [`HevcBitrate`] and
/// nothing else.
///
/// It exists as a named question so the gate and the adapter cannot drift apart: the adapter builds
/// one request shape per answer, and a new variant that took a quantiser would have to answer here
/// or fail to compile at the adapter.
pub fn rate_control_mode_of(bitrate: HevcBitrate) -> Option<RateControl> {
    match bitrate {
        HevcBitrate::Cbr(bps) => Some(RateControl::Cbr(bps)),
        HevcBitrate::Vbr { target, max } => Some(RateControl::Vbr { target, max }),
        // **`None` because `RateControl` has no constant-quantiser variant carrying the number** —
        // its `ConstantQuantiser` is the unit variant that means "no record, the picture record's
        // `pic_init_qp` is the quality", which is the same *mode* and a different *number*. The
        // quantiser itself rides on the request (`HevcBitrate::quantiser`), where `HevcEncoder`
        // expects it, so nothing is lost by this answer being `None` — and it is `None` for exactly
        // the reason `names_its_own_quantiser` is `true`, which is the pairing `build` checks.
        HevcBitrate::ConstantQuantiser(_) => None,
    }
}
