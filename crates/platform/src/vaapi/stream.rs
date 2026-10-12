//! What differs between the codecs the Linux decoder takes: which streams the GPU's VA-API
//! decoders can do, how a config is asked for them, and how a container sample becomes libva's
//! input. H.264 / HEVC (`avcC` / `hvcC`) go in as Annex B; VP9 frames and AV1 temporal units go in
//! as they are in the container. Safe code, with the unit tests of what is declined.

use filmcraft_codecs::hw::{FrameCodec, FrameStreamInfo, NalCodec, NalStreamInfo, StreamInfo};

use super::ffi::{Display, Entrypoint, Profile, RtFormat};
use crate::annexb::to_annex_b;
use crate::biplanar::Geometry;

/// Largest picture the backend takes (as the Media Foundation one).
pub const MAX_SIDE: u32 = 8192;

/// A stream the decoder takes.
#[derive(Clone, Debug)]
pub enum Stream {
    /// H.264 / HEVC.
    Nal(NalStreamInfo),
    /// VP9 / AV1.
    Frame(FrameStreamInfo),
}

/// What the GPU's video engine must decode for a stream. libva configures a decoder with a profile,
/// an entry point and a render format rather than with a codec name, so this is what a config is
/// created from; [`supports`] is what says the driver really offers it.
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    /// The profile the config is created with.
    pub profile: Profile,
    /// The entry point, which is [`Entrypoint::Vld`] for every decode: a VA-API hardware decoder is
    /// the video decoder, and the encoder entry points are never a decode path.
    pub entrypoint: Entrypoint,
    /// The render format a decoded picture is created in: 8-bit or 10-bit 4:2:0.
    pub format: RtFormat,
    /// Size the surfaces are created at (the coded size, before the crop).
    pub size: (u32, u32),
}

impl From<StreamInfo> for Stream {
    fn from(i: StreamInfo) -> Self {
        match i {
            StreamInfo::Nal(n) => Self::Nal(n),
            StreamInfo::Frame(f) => Self::Frame(f),
        }
    }
}

impl From<NalStreamInfo> for Stream {
    fn from(i: NalStreamInfo) -> Self {
        Self::Nal(i)
    }
}

impl From<FrameStreamInfo> for Stream {
    fn from(i: FrameStreamInfo) -> Self {
        Self::Frame(i)
    }
}

/// The stream info of a sample entry this backend could take (`avcC` / `hvcC` / `vpcC` / `av1C`), or
/// `None` for other codecs and unreadable configuration records.
///
/// The same helper the Windows backend has, for the same two callers (the factory in front of our
/// decoders, and the diagnostics that ask what this machine can decode): a frame-coded entry is
/// recognised here whether or not `plan_frame` takes it, so that the recognition and the plan cannot
/// drift apart.
pub fn stream_info(entry: &filmcraft_isobmff::SampleEntry) -> Option<StreamInfo> {
    match NalStreamInfo::from_entry(entry) {
        Some(r) => r.ok().map(StreamInfo::Nal),
        None => FrameStreamInfo::from_entry(entry)?.ok().map(StreamInfo::Frame),
    }
}

/// What the driver needs for an `avcC` / `hvcC` stream, or why it is declined: 4:2:0 progressive,
/// 8 or 10 bits — H.264 at 8 bits in `profile_idc` 66 (Baseline), 77 (Main) or 100 (High), and HEVC
/// at 8 bits in 1 (Main) or 3 (Main Still Picture), or at 10 bits in 2 (Main 10). Nothing else, and
/// the profile arms below are what say why: 10-bit H.264 included.
pub fn plan_nal(info: &NalStreamInfo) -> Result<Plan, String> {
    if info.interlaced {
        return Err("field-coded video".into());
    }
    if info.chroma_format_idc != 1 {
        return Err(format!("chroma_format_idc {}", info.chroma_format_idc));
    }
    if info.bit_depth_luma != info.bit_depth_chroma || !matches!(info.bit_depth_luma, 8 | 10) {
        return Err(format!("{}-bit luma / {}-bit chroma", info.bit_depth_luma, info.bit_depth_chroma));
    }
    let (cx, cy, w, h) = info.crop;
    if w == 0 || h == 0 || w > MAX_SIDE || h > MAX_SIDE {
        return Err(format!("picture size {w}x{h}"));
    }
    // The surfaces are created at the coded size, so that is what `MAX_SIDE` bounds as well: VA's
    // surface attributes are `uint16_t` wide to begin with, and a size a driver cannot be asked
    // for is a bitstream FilmCraft does not hand to one.
    let (cw, ch) = info.coded;
    if cw == 0 || ch == 0 || cw > MAX_SIDE || ch > MAX_SIDE {
        return Err(format!("coded picture size {cw}x{ch}"));
    }
    // A crop that does not fit the coded picture is a bitstream that says something the surfaces
    // would not hold; it is declined rather than trimmed to what does fit.
    if cx.saturating_add(w) > cw || cy.saturating_add(h) > ch {
        return Err(format!("crop {cx},{cy} {w}x{h} is outside the {cw}x{ch} coded picture"));
    }
    let ten = info.bit_depth_luma == 10;
    let format = if ten { RtFormat::Yuv420_10 } else { RtFormat::Nv12 };
    let profile = match (info.codec, info.profile_idc, ten) {
        // Baseline, Main and High at 8 bits: the profiles a VA-API driver lists (libva's own
        // `VAProfileH264Baseline` is 5 but deprecated, so Constrained Baseline is what is asked for).
        (NalCodec::H264, 66, false) => Profile::H264ConstrainedBaseline,
        (NalCodec::H264, 77, false) => Profile::H264Main,
        (NalCodec::H264, 100, false) => Profile::H264High,
        // 10-bit H.264 is declined rather than planned into the 8-bit profile above: libva names it
        // `VAProfileH264High10` (36, va.h:542), which `Profile` does not carry, and asking for High
        // with a 10-bit render format would pass `supports` (every driver lists High with VLD) and
        // then be refused at `vaCreateConfig` — a capability claimed and not verified. Taken when
        // there is a 10-bit fixture to prove a hardware path for and the profile to name it with.
        (NalCodec::H264, _, true) => return Err(format!("10-bit H.264 (profile {})", info.profile_idc)),
        (NalCodec::H264, p, _) => return Err(format!("H.264 profile {p} at {} bits", info.bit_depth_luma)),
        // Main (1) and Main Still Picture (3) at 8 bits, Main 10 (2) at 10 bits. H.265 has no Main 10
        // Still Picture, so a profile 1 or 3 at 10 bits is a self-contradictory sequence parameter
        // set and is declined rather than asked for as Main 10.
        (NalCodec::Hevc, 1 | 3, false) => Profile::HevcMain,
        (NalCodec::Hevc, 2, true) => Profile::HevcMain10,
        (NalCodec::Hevc, p, _) => return Err(format!("HEVC profile {p} at {} bits", info.bit_depth_luma)),
    };
    Ok(Plan { profile, entrypoint: Entrypoint::Vld, format, size: info.coded })
}

/// What the driver needs for a VP9 / AV1 stream, or why it is declined.
///
/// **Declining here rather than at the first frame is the point**: a stream that cannot be described
/// is refused while it is still a stream description, so the caller sees "this machine cannot take
/// that stream" and takes the software decoder, rather than a mid-stream failure once a context
/// exists.
///
/// **VP9 and AV1 are both planned here, and for the same reason.** Each codec's record and the state
/// it is built against both exist — `super::picture::Vp9State` with `vp9_picture_record`, and
/// `super::av1::Av1State` with `av1_picture_record` — and what this function has to say about that
/// state is **nothing**: a `FrameStreamInfo` describes the stream, and the reference surfaces are the
/// pictures decoded out of it, which is a run and not a description. So the missing state is not a
/// reason to decline here, and the records are submitted by the one caller that threads a state
/// (`VaDecoder::picture_frame`). The profile table below is each record builder's own table, called
/// from here so the two cannot disagree about which VP9 or which AV1 this backend takes.
///
/// **What changed, and when it became true.** This function used to decline AV1 while saying *"the
/// uncompressed frame header is exposed, but no picture-parameter record is built from it yet"*. That
/// was accurate from `51f5b77` (when `filmcraft_av1::FrameHeader::parse` became reachable) until the
/// record was built — and it was deliberately **not** flipped earlier, because a plan here is a claim
/// that the stream can be decoded, and `VaDecoder` had no AV1 decode to back it: a plan a decoder
/// cannot honour is a capability claimed and not verified, which is the failure this backend keeps
/// trying to kill. It is flipped now because `VaDecoder::picture_frame` dispatches AV1 to
/// `Av1State::frame_at` and the parity suite decodes AV1 fixtures picture for picture.
pub fn plan_frame(info: &FrameStreamInfo) -> Result<Plan, String> {
    if info.mono || info.subsampling != (1, 1) {
        return Err(format!("chroma subsampling {:?}{}", info.subsampling, if info.mono { " (monochrome)" } else { "" }));
    }
    // The one size a frame-coded stream has: it is both the surface size `Plan.size` carries and the
    // picture `first_geometry` describes, so bounding it here bounds both.
    let (w, h) = info.size;
    if w == 0 || h == 0 || w > MAX_SIDE || h > MAX_SIDE {
        return Err(format!("picture size {w}x{h}"));
    }
    // The profile and format this stream would be planned into. Resolved before the decline so that
    // a profile or depth this backend would not take is still reported as the reason, and so that the
    // decline names what was on offer.
    let (profile, format) = match (info.codec, info.profile, info.bit_depth) {
        // VP9's own table, from the record builder: profiles 0 and 2 at the depths 4:2:0 has a
        // surface for, and a decline with the codec's reason for every other pair. Called rather than
        // re-listed so that a profile the record builder would refuse is never planned here.
        (FrameCodec::Vp9, p, b) => {
            if let Some(why) = super::picture::vp9_decline(p, b) {
                return Err(why);
            }
            (Profile::Vp9(p), if b == 10 { RtFormat::Yuv420_10 } else { RtFormat::Nv12 })
        }
        // AV1's own table, for the same reason and in the same shape: a profile or depth this backend
        // cannot put a surface for is declined as *that*, with the codec's reason, before the fact that
        // no record is built yet can be mistaken for the reason. `va_dec_av1.h:44` — "This AV1 decoding
        // API supports 8-bit/10bit 420 format only" — is where profile 1 (4:4:4), profile 2 (4:2:2 at
        // 12 bits) and 12-bit come from, and `va_dec_av1.h:44` is a header line rather than this
        // backend's opinion.
        (FrameCodec::Av1, p, b) => {
            if let Some(why) = super::picture::av1_decline(p, b) {
                return Err(why);
            }
            (Profile::Av1Main, if b == 10 { RtFormat::Yuv420_10 } else { RtFormat::Nv12 })
        }
    };
    // One codec is planned and one is declined, and each arm says what is actually true — see the
    // note above. The VP9 arm above has already refused every profile and depth this backend does
    // not decode, so reaching `Ok` here means a record the backend can build for every frame.
    match info.codec {
        FrameCodec::Vp9 | FrameCodec::Av1 => Ok(Plan { profile, entrypoint: Entrypoint::Vld, format, size: info.size }),
    }
}

/// Whether this driver really decodes `plan`: it lists the profile, and it lists the video decoder
/// for that profile. This is the "hardware is verified, not assumed" check — a machine whose
/// driver names a profile it cannot create a context for declines the stream here rather than
/// half-way through the first GOP.
pub fn supports(display: &Display, plan: &Plan) -> Result<(), String> {
    if !display.profiles()?.contains(&plan.profile) {
        return Err(format!("this driver does not offer the {:?} profile", plan.profile));
    }
    if !display.entrypoints(plan.profile)?.contains(&plan.entrypoint) {
        return Err(format!("this driver offers {:?} no {:?} entry point", plan.profile, plan.entrypoint));
    }
    Ok(())
}

impl Stream {
    /// What the GPU would decode this stream with, or why it is declined.
    pub fn plan(&self) -> Result<Plan, String> {
        match self {
            Self::Nal(i) => plan_nal(i),
            Self::Frame(i) => plan_frame(i),
        }
    }

    /// The hybrid decoder's view of the stream.
    pub fn info(&self) -> StreamInfo {
        match self {
            Self::Nal(i) => i.clone().into(),
            Self::Frame(i) => i.clone().into(),
        }
    }

    /// The backend's name for the stream, as the counters, the logs and the diagnostics show it.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Nal(i) if i.codec == NalCodec::H264 => "VA-API H.264",
            Self::Nal(_) => "VA-API HEVC",
            Self::Frame(i) if i.codec == FrameCodec::Vp9 => "VA-API VP9",
            Self::Frame(_) => "VA-API AV1",
        }
    }

    /// The geometry of the first picture: the stream's own (H.264 / HEVC), or the expected one
    /// until the first VP9 key frame says (colour and size come from its header). `bits` is what
    /// the plan's render format holds, 8 or 10.
    pub fn first_geometry(&self, bits: u32) -> Geometry {
        match self {
            Self::Nal(i) => Geometry { crop: i.crop, bits, color: i.color, par: i.par },
            Self::Frame(i) => Geometry { crop: (0, 0, i.size.0, i.size.1), bits, color: i.color.unwrap_or(filmcraft_color::ColorInfo::REC709), par: (1, 1) },
        }
    }

    /// Whether the decoder can start (again) at `sample` after a drain: an IDR picture, a key frame.
    pub fn is_restart(&self, sample: &[u8]) -> bool {
        match self {
            Self::Nal(info) => info.nal_types(sample).iter().any(|&t| match info.codec {
                NalCodec::H264 => t == 5,
                NalCodec::Hevc => matches!(t, 19 | 20),
            }),
            Self::Frame(i) => i.is_random_access(sample),
        }
    }

    /// `sample` as the decoder's input: Annex B for a NAL codec, the container's own bytes for a
    /// frame codec. With `headers` the stream's parameter sets / sequence header go first (unless
    /// the sample carries them): a decoder starting at this sample has what it needs.
    pub fn input(&self, sample: &[u8], headers: bool) -> Result<Vec<u8>, String> {
        match self {
            Self::Nal(info) => to_annex_b(info, sample, headers),
            Self::Frame(i) => {
                if sample.is_empty() {
                    return Err("empty sample".into());
                }
                if headers && !i.config_obus.is_empty() && !i.carries_sequence_header(sample) {
                    let mut v = Vec::with_capacity(i.config_obus.len() + sample.len());
                    v.extend_from_slice(&i.config_obus);
                    v.extend_from_slice(sample);
                    return Ok(v);
                }
                Ok(sample.to_vec())
            }
        }
    }

    /// Picture format a VP9 key frame declares (size and colour); `None` for other samples and codecs.
    pub fn declared_picture(&self, sample: &[u8]) -> Option<Geometry> {
        let Self::Frame(i) = self else { return None };
        if i.codec != FrameCodec::Vp9 {
            return None;
        }
        let p = i.picture_params(sample)?;
        Some(Geometry { crop: (0, 0, p.size.0, p.size.1), bits: p.bit_depth, color: p.color, par: (1, 1) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vaapi::ffi::load;
    use filmcraft_codecs::hw::{FrameCodec, FrameStreamInfo, NalCodec, NalStreamInfo};

    // copied from media_foundation/stream.rs:238 — the crate's hand-built stream description
    fn nal(codec: NalCodec, profile_idc: u8, bits: u32, chroma_format_idc: u32) -> NalStreamInfo {
        NalStreamInfo {
            codec,
            length_size: 4,
            highest_tid: None,
            parameter_sets: Vec::new(),
            coded: (1920, 1088),
            crop: (0, 0, 1920, 1080),
            chroma_format_idc,
            bit_depth_luma: bits,
            bit_depth_chroma: bits,
            interlaced: false,
            profile_idc,
            color: filmcraft_color::ColorInfo::REC709,
            par: (1, 1),
            reorder: 2,
        }
    }

    fn frame(codec: FrameCodec, profile: u8, bits: u32, subsampling: (u8, u8)) -> FrameStreamInfo {
        FrameStreamInfo::for_tests(codec, profile, bits, subsampling, (1920, 1080))
    }

    #[test]
    fn declines_what_vaapi_does_not_hardware_decode() {
        // interlaced
        let mut i = nal(NalCodec::H264, 100, 8, 1);
        i.interlaced = true;
        assert!(plan_nal(&i).is_err(), "field-coded H.264");
        // chroma that is not 4:2:0
        assert!(plan_nal(&nal(NalCodec::H264, 100, 8, 2)).is_err(), "4:2:2");
        assert!(plan_nal(&nal(NalCodec::H264, 100, 8, 0)).is_err(), "monochrome");
        // bit depths outside 8 and 10
        assert!(plan_nal(&nal(NalCodec::H264, 100, 12, 1)).is_err(), "12-bit");
        // luma and chroma depths differing
        let mut i = nal(NalCodec::H264, 100, 10, 1);
        i.bit_depth_chroma = 8;
        assert!(plan_nal(&i).is_err(), "mixed bit depth");
    }

    #[test]
    fn takes_the_two_nal_codecs_at_8_and_10_bit() {
        for (codec, profile, ten) in [(NalCodec::H264, 66u8, false), (NalCodec::H264, 100, false), (NalCodec::Hevc, 1, false), (NalCodec::Hevc, 2, true)] {
            let p = plan_nal(&nal(codec, profile, if ten { 10 } else { 8 }, 1)).unwrap_or_else(|e| panic!("{codec:?} {profile} {ten}: {e}"));
            assert_eq!(p.entrypoint, Entrypoint::Vld);
            assert_eq!(p.format, if ten { RtFormat::Yuv420_10 } else { RtFormat::Nv12 });
        }
        // 10-bit H.264 is the one depth this backend declines (see `plan_nal`): libva names it
        // `VAProfileH264High10`, which `Profile` does not carry, so there is nothing to verify.
        assert!(plan_nal(&nal(NalCodec::H264, 100, 10, 1)).is_err(), "10-bit H.264 High");
    }

    /// Which frame codecs are **planned**, and which are declined while still a description rather
    /// than at their first frame — and the two answers are the same for both codecs.
    ///
    /// **Both are planned.** Each per-frame header is exposed, each record is built, and the state
    /// that record needs (the eight reference slots and the parse state they are read against)
    /// belongs to a *run* — so it is not a reason to refuse a *description*. What is still refused is
    /// what this backend cannot put a surface for: VP9 profiles 1, 3 and 4 and 12-bit, and AV1
    /// profiles 1 and 2 and 12-bit, which the profile tables below decide. Asserting a decline for a
    /// VP9 or AV1 profile this backend *does* decode would assert that the decoder cannot decode it,
    /// which is false for both — and was false for AV1 the moment its record existed.
    #[test]
    fn the_frame_codecs_are_planned_or_declined_for_their_own_reason() {
        for (codec, profile, bits, want) in [
            (FrameCodec::Vp9, 0u8, 8u32, Profile::Vp9(0)),
            (FrameCodec::Vp9, 2, 10, Profile::Vp9(2)),
            (FrameCodec::Av1, 0, 8, Profile::Av1Main),
            (FrameCodec::Av1, 0, 10, Profile::Av1Main),
        ] {
            let info = FrameStreamInfo::for_tests(codec, profile, bits, (1, 1), (1920, 1080));
            let plan = plan_frame(&info).unwrap_or_else(|e| panic!("{codec:?} profile {profile} at {bits} is planned: {e}"));
            // The plan is the codec's own profile and depth, at the surface format for that depth — the
            // same mapping the record's own `profile` and `bit_depth` bytes carry, so the config a driver
            // is asked about is the one the records describe.
            assert_eq!(plan.profile, want, "{codec:?} profile {profile} at {bits}");
            assert_eq!(plan.format, if bits == 10 { RtFormat::Yuv420_10 } else { RtFormat::Nv12 }, "{codec:?} profile {profile} at {bits}");
            assert_eq!(plan.entrypoint, Entrypoint::Vld, "a hardware decoder is the video decoder");
            assert_eq!(plan.size, (1920, 1080), "the surfaces are created at the stream's own size");
            // and the same stream through `Stream::plan`, which is what a decoder actually calls
            assert!(Stream::from(info).plan().is_ok(), "{codec:?} profile {profile} at {bits} through Stream::plan");
        }
        // VP9 profiles this backend has no surface for are still declined, and as *that*: a 4:2:2 stream
        // is not told it lacks a parser, which it does not.
        for (profile, bits, why) in [(1u8, 8u32, "4:2:2"), (3, 8, "4:2:2"), (4, 8, "reserves"), (0, 12, "12-bit")] {
            let info = FrameStreamInfo::for_tests(FrameCodec::Vp9, profile, bits, (1, 1), (1920, 1080));
            let e = plan_frame(&info).expect_err("VP9 profile {profile} at {bits} bits");
            assert!(e.contains(why), "VP9 profile {profile} at {bits} bits: {e}");
            assert!(Stream::from(info).plan().is_err(), "VP9 profile {profile} at {bits} bits through Stream::plan");
        }
    }

    /// The VP9 profile table is **one** table, and this is what makes it one: every pair it declines,
    /// `plan_frame` and the record builder decline with the same sentence — and every pair it *takes*,
    /// `plan_frame` plans and the record builder builds.
    ///
    /// The failure this guards against is not hypothetical — the H.264 milestone's `[3, 3]` index
    /// collision was a picture record and a slice record disagreeing about the same frame, and a
    /// decline that only one of the two sites knows about is the same kind of split, one step earlier.
    /// Both sites call `picture::vp9_decline`, so this test is really pinning that both sites still
    /// call it: it walks the whole table, including every depth, and compares the two answers.
    #[test]
    fn the_vp9_decline_is_the_same_sentence_wherever_it_is_decided() {
        for profile in 0u8..=4 {
            for bits in [8u32, 10, 12] {
                let info = FrameStreamInfo::for_tests(FrameCodec::Vp9, profile, bits, (1, 1), (1920, 1080));
                let from_plan = plan_frame(&info);
                // The record builder's own table, reached the way `vp9_picture_record` reaches it:
                // with a `FrameHeader` carrying the same profile and depth. It does not have to be a
                // parseable frame here — `vp9_decline` reads nothing else.
                let header = filmcraft_vp9::header::FrameHeader {
                    profile,
                    color: filmcraft_vp9::header::ColorConfig { bit_depth: bits as u8, ..Default::default() },
                    ..Default::default()
                };
                let from_record = crate::vaapi::picture::vp9_decline(header.profile, u32::from(header.color.bit_depth));
                match (from_plan, from_record) {
                    // A pair the backend does not take: both refuse, and `plan_frame` refuses with
                    // the table's own sentence — the profile reason, word for word.
                    (Err(planned), Some(from_table)) => assert_eq!(planned, from_table, "profile {profile} at {bits} bits"),
                    // A pair both take: `plan_frame` *plans* it, as the profile and depth the record
                    // itself carries. This is the half that catches a table entry added to
                    // `vp9_decline` without `plan_frame` declining it — the opposite drift — and the
                    // other: a pair the table takes that `plan_frame` still refuses, which would be a
                    // stream this backend can describe and will not decode.
                    (Ok(planned), None) => {
                        assert_eq!(planned.profile, Profile::Vp9(profile), "profile {profile} at {bits} bits is planned as its own profile");
                        assert_eq!(planned.format, if bits == 10 { RtFormat::Yuv420_10 } else { RtFormat::Nv12 }, "profile {profile} at {bits} bits");
                    }
                    (planned, from_table) => {
                        panic!("profile {profile} at {bits} bits: plan_frame says {planned:?}, the record builder's table says {from_table:?}")
                    }
                }
            }
        }
    }

    /// A stream this backend would not take anyway is declined as *that*, not with a record-missing
    /// reason: the table that applies to the records too stays live under the plan, so a 4:4:4 or a
    /// profile 2 stream is told the truth about itself.
    #[test]
    fn declines_frame_codecs_vaapi_would_not_take_anyway() {
        for (codec, profile, bits, why) in [
            (FrameCodec::Av1, 1u8, 8u32, "AV1 profile 1"),
            (FrameCodec::Av1, 2, 10, "AV1 profile 2"),
            (FrameCodec::Av1, 0, 12, "AV1 profile 0 at 12 bits"),
            // VP9's reasons come from its own table now, and each says *why* rather than restating the
            // numbers — the numbers are already in the sentence's prefix, so a bare "VP9 profile 1 at
            // 8 bits" no longer tells a user what was wrong with it.
            (FrameCodec::Vp9, 1, 8, "VP9 profile 1 at 8 bits: 4:2:2"),
            (FrameCodec::Vp9, 3, 10, "VP9 profile 3 at 10 bits: 4:2:2"),
            (FrameCodec::Vp9, 0, 12, "VP9 profile 0 at 12 bits: 12-bit"),
            // Profile 4 is reserved by the specification: refused, not parsed and guessed at.
            (FrameCodec::Vp9, 4, 8, "VP9 profile 4 at 8 bits: profile 4, which the VP9 specification reserves"),
        ] {
            let info = FrameStreamInfo::for_tests(codec, profile, bits, (1, 1), (1920, 1080));
            let e = plan_frame(&info).expect_err("{why}");
            assert!(e.starts_with(why), "{why}: {e}");
        }
        // and the chroma formats below 4:2:0, which are refused before the profile is looked at
        for (codec, subsampling) in [(FrameCodec::Av1, (0u8, 0u8)), (FrameCodec::Vp9, (1, 0))] {
            let info = FrameStreamInfo::for_tests(codec, 0, 8, subsampling, (1920, 1080));
            assert!(plan_frame(&info).expect_err("{subsampling:?}").starts_with("chroma subsampling"), "{subsampling:?}");
        }
    }

    #[test]
    fn declines_oversized_and_out_of_range_crops() {
        // the model's crop table: empty, oversized, off the coded edge, and a crop origin that
        // overflows when added to its width (`saturating_add` is the line that has to hold here)
        for crop in [
            (0, 0, 9000, 100),
            (0, 0, 0, 1080),
            (0, 0, 1920, 0),
            (0, 0, 1080, 9000),
            (8, 0, 1920, 1080),
            (0, 16, 1920, 1080),
            (u32::MAX, 0, 16, 16),
            (u32::MAX, u32::MAX, 1, 1),
        ] {
            let mut i = nal(NalCodec::H264, 100, 8, 1);
            i.crop = crop;
            assert!(plan_nal(&i).is_err(), "{crop:?}");
        }
        // a crop that leaves the coded picture it was read from
        let mut i = nal(NalCodec::H264, 100, 8, 1);
        i.crop = (100, 0, 640, 100);
        i.coded = (640, 360);
        assert!(plan_nal(&i).is_err(), "crop outside the coded picture");
    }

    /// The coded size is what the surfaces are created at, so it is bounded like the crop: VA's
    /// surface attributes are `uint16_t` to begin with, and `Plan.size` carries it.
    #[test]
    fn declines_an_unbounded_coded_picture() {
        for coded in [(100_000, 100_000), (0, 1080), (1920, 0), (9000, 1080), (u32::MAX, u32::MAX)] {
            let mut i = nal(NalCodec::H264, 100, 8, 1);
            i.coded = coded;
            assert!(plan_nal(&i).is_err(), "coded {coded:?}");
        }
        let mut big = nal(NalCodec::H264, 100, 8, 1);
        big.coded = (MAX_SIDE, MAX_SIDE);
        assert_eq!(plan_nal(&big).unwrap().size, (MAX_SIDE, MAX_SIDE), "a coded picture of exactly MAX_SIDE is taken");
    }

    /// The whole table, not only what it accepts: every profile the mapping takes and the depth it
    /// takes it at, and the depth each one is refused at. libva names more profiles than a codec has
    /// `profile_idc` values (`VAProfileH264High10` is one we do not carry), so this table is what
    /// this backend asks a driver about, not what libva has.
    #[test]
    fn the_nal_profiles_map_to_the_ones_a_vaapi_driver_lists() {
        for (p, profile) in [(66u8, Profile::H264ConstrainedBaseline), (77, Profile::H264Main), (100, Profile::H264High)] {
            let plan = plan_nal(&nal(NalCodec::H264, p, 8, 1)).unwrap();
            assert_eq!(
                (plan.profile, plan.format, plan.entrypoint, plan.size),
                (profile, RtFormat::Nv12, Entrypoint::Vld, (1920, 1088)),
                "H.264 profile {p} at 8 bits"
            );
            // every H.264 profile this backend takes is 8-bit only (see `plan_nal`)
            assert!(plan_nal(&nal(NalCodec::H264, p, 10, 1)).is_err(), "H.264 profile {p} at 10 bits");
        }
        // HEVC Main and Main Still Picture at 8 bits; neither is a 10-bit profile (H.265 has no
        // Main 10 Still Picture), so both are declined above 8 bits rather than planned as Main 10
        for p in [1u8, 3] {
            assert_eq!(plan_nal(&nal(NalCodec::Hevc, p, 8, 1)).unwrap().profile, Profile::HevcMain, "HEVC profile {p} at 8 bits");
            assert!(plan_nal(&nal(NalCodec::Hevc, p, 10, 1)).is_err(), "HEVC profile {p} at 10 bits");
        }
        let main10 = plan_nal(&nal(NalCodec::Hevc, 2, 10, 1)).unwrap();
        assert_eq!((main10.profile, main10.format), (Profile::HevcMain10, RtFormat::Yuv420_10));
        assert!(plan_nal(&nal(NalCodec::Hevc, 2, 8, 1)).is_err(), "Main 10 at 8 bits is not a stream");
        for (codec, p) in [(NalCodec::H264, 110u8), (NalCodec::H264, 118), (NalCodec::H264, 244), (NalCodec::Hevc, 4), (NalCodec::Hevc, 5)] {
            assert!(plan_nal(&nal(codec, p, 8, 1)).is_err(), "{codec:?} profile {p} at 8 bits");
        }
    }

    /// The profile and format each frame profile is planned as — for VP9 and for AV1 alike — plus the
    /// size bound that applies to a frame stream whatever its profile: a VA-API surface is allocated
    /// at that size and cropped on readback, so `MAX_SIDE` bounds it.
    #[test]
    fn the_frame_profile_table_survives_the_decline() {
        for (codec, p, bits, profile, format) in [
            (FrameCodec::Vp9, 0u8, 8u32, Profile::Vp9(0), RtFormat::Nv12),
            (FrameCodec::Vp9, 2, 10, Profile::Vp9(2), RtFormat::Yuv420_10),
            (FrameCodec::Av1, 0, 8, Profile::Av1Main, RtFormat::Nv12),
            (FrameCodec::Av1, 0, 10, Profile::Av1Main, RtFormat::Yuv420_10),
        ] {
            let plan = plan_frame(&frame(codec, p, bits, (1, 1))).unwrap_or_else(|why| panic!("{codec:?} profile {p} at {bits} is planned: {why}"));
            assert_eq!(plan.profile, profile, "{codec:?} profile {p} at {bits}");
            assert_eq!(plan.format, format, "{codec:?} profile {p} at {bits}");
            // The plan's size is the surface size, and it is the stream's own picture size — which
            // is what bounds both the allocation and the readback crop.
            assert_eq!(plan.size, (1920, 1080), "{codec:?} profile {p} at {bits}: surfaces are created at this size");
        }
        // and the size bound applies to a frame stream that *is* planned: a stream of exactly
        // `MAX_SIDE` is within it and is taken, which is what makes the next line a bound and not a
        // blanket refusal.
        let mut big = frame(FrameCodec::Vp9, 0, 8, (1, 1));
        big.size = (MAX_SIDE, MAX_SIDE);
        assert_eq!(plan_frame(&big).expect("MAX_SIDE is within the bound").size, (MAX_SIDE, MAX_SIDE));
        let mut huge = frame(FrameCodec::Vp9, 0, 8, (1, 1));
        huge.size = (MAX_SIDE + 1, MAX_SIDE);
        assert!(plan_frame(&huge).expect_err("an oversized stream").starts_with("picture size"), "an oversized frame stream is still refused on its size");
    }

    #[test]
    fn a_restart_is_an_idr_picture_and_the_first_geometry_is_the_streams_own() {
        let h = Stream::from(nal(NalCodec::H264, 100, 8, 1));
        assert!(h.is_restart(&[0, 0, 0, 1, 0x65, 0]), "H.264 IDR");
        assert!(!h.is_restart(&[0, 0, 0, 1, 0x41, 0]), "H.264 non-IDR");
        let hevc = Stream::from(nal(NalCodec::Hevc, 1, 8, 1));
        assert!(hevc.is_restart(&[0, 0, 0, 1, 0x26, 0]), "HEVC IDR_W_RADL");
        assert!(hevc.is_restart(&[0, 0, 0, 1, 0x29, 0]), "HEVC IDR_N_LP");
        assert!(!hevc.is_restart(&[0, 0, 0, 1, 0x02, 0]), "HEVC trail picture");
        assert_eq!(h.first_geometry(10).crop, (0, 0, 1920, 1080), "the crop, at the planned depth");
        assert_eq!(Stream::from(frame(FrameCodec::Vp9, 0, 8, (1, 1))).first_geometry(8).crop, (0, 0, 1920, 1080));
    }

    #[test]
    fn declines_monochrome_and_absurd_frame_pictures() {
        let mut mono = frame(FrameCodec::Av1, 0, 8, (1, 1));
        mono.mono = true;
        assert!(plan_frame(&mono).expect_err("monochrome").starts_with("chroma subsampling"), "monochrome");
        for size in [(0, 1080), (1920, 0), (9000, 1080), (1920, 9000)] {
            let mut i = frame(FrameCodec::Vp9, 0, 8, (1, 1));
            i.size = size;
            assert!(plan_frame(&i).expect_err("{size:?}").starts_with("picture size"), "{size:?}");
        }
    }

    #[test]
    fn names_the_four_codecs_the_way_the_logs_and_the_ui_show_them() {
        for (stream, name) in [
            (Stream::from(nal(NalCodec::H264, 100, 8, 1)), "VA-API H.264"),
            (Stream::from(nal(NalCodec::Hevc, 1, 8, 1)), "VA-API HEVC"),
            (Stream::from(frame(FrameCodec::Vp9, 0, 8, (1, 1))), "VA-API VP9"),
            (Stream::from(frame(FrameCodec::Av1, 0, 8, (1, 1))), "VA-API AV1"),
        ] {
            assert_eq!(stream.name(), name);
            // the hybrid decoder's view of the stream is the same description
            assert!(
                matches!((&stream, stream.info()), (Stream::Nal(_), StreamInfo::Nal(_)) | (Stream::Frame(_), StreamInfo::Frame(_))),
                "{name} keeps its description"
            );
        }
    }

    #[test]
    fn input_is_annex_b_for_a_nal_codec_and_the_containers_own_bytes_otherwise() {
        let h = Stream::from(nal(NalCodec::H264, 100, 8, 1));
        assert_eq!(h.input(&[0, 0, 0, 2, 0x65, 9], false).unwrap(), [0, 0, 0, 1, 0x65, 9]);
        assert!(h.input(&[0, 0, 0, 9, 0x65, 9], false).is_err(), "a length that runs past the sample");

        let mut av1 = frame(FrameCodec::Av1, 0, 8, (1, 1));
        av1.config_obus = vec![0x0a, 0x02, 0x00, 0x00];
        let a = Stream::from(av1);
        let unit = [0x32, 0x01, 0x02];
        assert_eq!(a.input(&unit, false).unwrap(), unit, "without headers the sample is what it is");
        assert_eq!(a.input(&unit, true).unwrap(), [0x0a, 0x02, 0x00, 0x00, 0x32, 0x01, 0x02], "the sequence header goes first");
        let own = [0x0a, 0x02, 0x00, 0x00];
        assert_eq!(a.input(&own, true).unwrap(), own, "a sample that carries the sequence header is passed as it is");
        assert!(a.input(&[], true).is_err(), "an empty sample");
    }

    /// What the driver itself says it offers: `supports` approves a plan exactly when the driver
    /// lists the profile with the video decoder for it, and declines what the driver does not. A
    /// machine with no libva, no display, or a driver naming none of the profiles this backend
    /// plans has nothing to check and skips with the reason printed.
    #[test]
    fn the_driver_is_asked_whether_it_offers_the_plan() {
        let Ok(api) = load() else {
            eprintln!("SKIPPED: no libva");
            return;
        };
        let Ok(d) = Display::new(&api) else {
            eprintln!("SKIPPED: no VA display");
            return;
        };
        // Every profile this backend can plan (`plan_nal`, and `plan_frame` for VP9), so the check is
        // asked about each one the driver names rather than about one arbitrary profile.
        let ours =
            [Profile::H264ConstrainedBaseline, Profile::H264Main, Profile::H264High, Profile::HevcMain, Profile::HevcMain10, Profile::Vp9(0), Profile::Vp9(2)];
        let reported = d.profiles().expect("vaQueryConfigProfiles");
        let planned: Vec<Profile> = ours.iter().copied().filter(|p| reported.contains(p)).collect();
        if planned.is_empty() {
            eprintln!("SKIPPED: this driver names none of the profiles this backend plans: {reported:?}");
            return;
        }
        for profile in planned {
            let vld = d.entrypoints(profile).expect("vaQueryConfigEntrypoints").contains(&Entrypoint::Vld);
            let plan = Plan { profile, entrypoint: Entrypoint::Vld, format: RtFormat::Nv12, size: (1920, 1080) };
            assert_eq!(vld, supports(&d, &plan).is_ok(), "{profile:?} through VLD: the driver says {vld}");
        }
        let plan = Plan { profile: Profile::H264High, entrypoint: Entrypoint::Vld, format: RtFormat::Nv12, size: (1920, 1080) };
        // an entry point no driver offers for any profile: not a decode path, so never supported
        assert!(supports(&d, &Plan { entrypoint: Entrypoint::Other(4242), ..plan }).is_err(), "an entry point that is not VLD");
        // and a profile libva has no number for is declined, even for a codec this backend takes
        let unknown = Profile::Other(4241);
        assert!(!reported.contains(&unknown), "libva has no profile {unknown:?}");
        assert!(supports(&d, &Plan { profile: unknown, ..plan }).is_err(), "a profile this driver does not list");
    }
}
