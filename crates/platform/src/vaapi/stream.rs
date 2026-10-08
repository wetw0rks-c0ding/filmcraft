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

/// What the driver needs for an `avcC` / `hvcC` stream, or why it is declined: 4:2:0 progressive at
/// 8 or 10 bits, in one of the profiles a VA-API decoder is configured for.
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
    // A crop that does not fit the coded picture is a bitstream that says something the surfaces
    // would not hold; it is declined rather than trimmed to what does fit.
    if cx.saturating_add(w) > info.coded.0 || cy.saturating_add(h) > info.coded.1 {
        return Err(format!("crop {cx},{cy} {w}x{h} is outside the {}x{} coded picture", info.coded.0, info.coded.1));
    }
    let ten = info.bit_depth_luma == 10;
    let format = if ten { RtFormat::Yuv420_10 } else { RtFormat::Nv12 };
    // The depth is the render format here rather than a profile of its own: libva names one
    // profile per profile_idc, and a driver that cannot decode one of the two depths for it
    // refuses the config it is asked for, which leaves the stream to FilmCraft's own decoder.
    let profile = match (info.codec, info.profile_idc, ten) {
        // Baseline, Main and High: the H.264 profiles a VA-API driver lists.
        (NalCodec::H264, 66, _) => Profile::H264ConstrainedBaseline,
        (NalCodec::H264, 77, _) => Profile::H264Main,
        (NalCodec::H264, 100, _) => Profile::H264High,
        (NalCodec::H264, p, _) => return Err(format!("H.264 profile {p} at {} bits", info.bit_depth_luma)),
        // Main, Main 10 and Main Still Picture (Main 10 at 8 bits is not a thing, and is declined
        // with the profile that asked for it).
        (NalCodec::Hevc, 1..=3, true) => Profile::HevcMain10,
        (NalCodec::Hevc, 1 | 3, false) => Profile::HevcMain,
        (NalCodec::Hevc, p, _) => return Err(format!("HEVC profile {p} at {} bits", info.bit_depth_luma)),
    };
    Ok(Plan { profile, entrypoint: Entrypoint::Vld, format, size: info.coded })
}

/// What the driver needs for a VP9 / AV1 stream, or why it is declined: 4:2:0 only, 8-bit or
/// 10-bit, VP9 profiles 0 and 2 (at their own depth) and AV1 main profile.
pub fn plan_frame(info: &FrameStreamInfo) -> Result<Plan, String> {
    if info.mono || info.subsampling != (1, 1) {
        return Err(format!("chroma subsampling {:?}{}", info.subsampling, if info.mono { " (monochrome)" } else { "" }));
    }
    let (w, h) = info.size;
    if w == 0 || h == 0 || w > MAX_SIDE || h > MAX_SIDE {
        return Err(format!("picture size {w}x{h}"));
    }
    let (profile, format) = match (info.codec, info.profile, info.bit_depth) {
        // the two VP9 profiles a VA-API driver decodes, each at its own depth
        (FrameCodec::Vp9, 0, 8) => (Profile::Vp9(0), RtFormat::Nv12),
        (FrameCodec::Vp9, 2, 10) => (Profile::Vp9(2), RtFormat::Yuv420_10),
        (FrameCodec::Vp9, p, b) => return Err(format!("VP9 profile {p} at {b} bits")),
        // AV1 main profile, at either depth (`VAProfileAV1Profile0`)
        (FrameCodec::Av1, 0, 8) => (Profile::Av1Main, RtFormat::Nv12),
        (FrameCodec::Av1, 0, 10) => (Profile::Av1Main, RtFormat::Yuv420_10),
        (FrameCodec::Av1, p, b) => return Err(format!("AV1 profile {p} at {b} bits")),
    };
    Ok(Plan { profile, entrypoint: Entrypoint::Vld, format, size: info.size })
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
    fn takes_the_four_codecs_at_8_and_10_bit() {
        for (codec, profile, ten) in
            [(NalCodec::H264, 66u8, false), (NalCodec::H264, 100, false), (NalCodec::H264, 100, true), (NalCodec::Hevc, 1, false), (NalCodec::Hevc, 2, true)]
        {
            let p = plan_nal(&nal(codec, profile, if ten { 10 } else { 8 }, 1)).unwrap_or_else(|e| panic!("{codec:?} {profile} {ten}: {e}"));
            assert_eq!(p.entrypoint, Entrypoint::Vld);
            assert_eq!(p.format, if ten { RtFormat::Yuv420_10 } else { RtFormat::Nv12 });
        }
        // `for_tests` is the crate's own hand-built description (hw_frame.rs:195)
        for (profile, bits) in [(0u8, 8u32), (2, 10)] {
            assert!(plan_frame(&FrameStreamInfo::for_tests(FrameCodec::Vp9, profile, bits, (1, 1), (1920, 1080))).is_ok(), "VP9 profile {profile} at {bits}");
        }
        for bits in [8u32, 10] {
            assert!(plan_frame(&FrameStreamInfo::for_tests(FrameCodec::Av1, 0, bits, (1, 1), (1920, 1080))).is_ok(), "AV1 profile 0 at {bits}");
        }
    }

    #[test]
    fn declines_frame_codecs_vaapi_does_not_take() {
        // 4:2:0 is taken; the profiles and chroma formats below it are not (monochrome is in the next test)
        assert!(plan_frame(&FrameStreamInfo::for_tests(FrameCodec::Vp9, 0, 8, (1, 1), (1920, 1080))).is_ok(), "4:2:0 VP9 is taken");
        assert!(plan_frame(&FrameStreamInfo::for_tests(FrameCodec::Av1, 1, 8, (1, 1), (1920, 1080))).is_err(), "AV1 profile 1");
        assert!(plan_frame(&FrameStreamInfo::for_tests(FrameCodec::Vp9, 1, 8, (1, 1), (1920, 1080))).is_err(), "VP9 profile 1 at 8 bits");
        assert!(plan_frame(&FrameStreamInfo::for_tests(FrameCodec::Vp9, 0, 12, (1, 1), (1920, 1080))).is_err(), "VP9 profile 0 at 12 bits");
        assert!(plan_frame(&FrameStreamInfo::for_tests(FrameCodec::Av1, 0, 8, (0, 0), (1920, 1080))).is_err(), "4:4:4");
    }

    #[test]
    fn declines_oversized_and_out_of_range_crops() {
        let mut i = nal(NalCodec::H264, 100, 8, 1);
        i.crop = (0, 0, 9000, 100);
        assert!(plan_nal(&i).is_err(), "wider than 8192");
        // a crop that leaves the coded picture
        let mut i = nal(NalCodec::H264, 100, 8, 1);
        i.crop = (100, 0, 640, 100);
        i.coded = (640, 360);
        assert!(plan_nal(&i).is_err(), "crop outside the coded picture");
        // a zero-size crop
        let mut i = nal(NalCodec::H264, 100, 8, 1);
        i.crop = (0, 0, 0, 100);
        assert!(plan_nal(&i).is_err(), "empty crop");
    }

    /// Every profile the mapping accepts, and the depth it takes it at: libva names one profile
    /// per codec profile, so this is the table a driver is asked against.
    #[test]
    fn the_nal_profiles_map_to_the_ones_a_vaapi_driver_lists() {
        for (p, profile) in [(66u8, Profile::H264ConstrainedBaseline), (77, Profile::H264Main), (100, Profile::H264High)] {
            for bits in [8, 10] {
                let plan = plan_nal(&nal(NalCodec::H264, p, bits, 1)).unwrap();
                assert_eq!((plan.profile, plan.entrypoint, plan.size), (profile, Entrypoint::Vld, (1920, 1088)), "H.264 profile {p} at {bits} bits");
                assert_eq!(plan.format, if bits == 10 { RtFormat::Yuv420_10 } else { RtFormat::Nv12 });
            }
        }
        // HEVC Main and Main Still Picture at either depth, Main 10 only at 10 bits
        for p in [1u8, 3] {
            assert_eq!(plan_nal(&nal(NalCodec::Hevc, p, 8, 1)).unwrap().profile, Profile::HevcMain, "HEVC profile {p} at 8 bits");
            assert_eq!(plan_nal(&nal(NalCodec::Hevc, p, 10, 1)).unwrap().profile, Profile::HevcMain10, "HEVC profile {p} at 10 bits");
        }
        assert_eq!(plan_nal(&nal(NalCodec::Hevc, 2, 10, 1)).unwrap().profile, Profile::HevcMain10);
        assert!(plan_nal(&nal(NalCodec::Hevc, 2, 8, 1)).is_err(), "Main 10 at 8 bits is not a stream");
        for (codec, p) in [(NalCodec::H264, 110u8), (NalCodec::H264, 118), (NalCodec::H264, 244), (NalCodec::Hevc, 4)] {
            assert!(plan_nal(&nal(codec, p, 8, 1)).is_err(), "{codec:?} profile {p}");
        }
    }

    /// Every profile a frame plan is for, the format the picture is created in, and the size: a
    /// VA-API surface is allocated at the coded size and cropped on readback.
    #[test]
    fn the_frame_profiles_map_to_the_ones_a_vaapi_driver_lists() {
        for (p, bits, profile, format) in [(0u8, 8u32, Profile::Vp9(0), RtFormat::Nv12), (2, 10, Profile::Vp9(2), RtFormat::Yuv420_10)] {
            let plan = plan_frame(&frame(FrameCodec::Vp9, p, bits, (1, 1))).unwrap();
            assert_eq!(
                (plan.profile, plan.format, plan.entrypoint, plan.size),
                (profile, format, Entrypoint::Vld, (1920, 1080)),
                "VP9 profile {p} at {bits} bits"
            );
        }
        for (bits, format) in [(8u32, RtFormat::Nv12), (10, RtFormat::Yuv420_10)] {
            let plan = plan_frame(&frame(FrameCodec::Av1, 0, bits, (1, 1))).unwrap();
            assert_eq!((plan.profile, plan.format, plan.entrypoint), (Profile::Av1Main, format, Entrypoint::Vld), "AV1 profile 0 at {bits} bits");
        }
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
        for ss in [(1, 0), (0, 0)] {
            assert!(plan_frame(&frame(FrameCodec::Vp9, 0, 8, ss)).is_err(), "4:2:2 or 4:4:4 VP9 is {ss:?}");
        }
        let mut mono = frame(FrameCodec::Av1, 0, 8, (1, 1));
        mono.mono = true;
        assert!(plan_frame(&mono).is_err(), "monochrome");
        for size in [(0, 1080), (1920, 0), (9000, 1080), (1920, 9000)] {
            let mut i = frame(FrameCodec::Vp9, 0, 8, (1, 1));
            i.size = size;
            assert!(plan_frame(&i).is_err(), "{size:?}");
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

    /// What the driver itself says it offers: the plan is supported exactly when the driver lists
    /// the profile with the video decoder for it. A machine without libva or a display skips.
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
        let plan = Plan { profile: Profile::H264High, entrypoint: Entrypoint::Vld, format: RtFormat::Nv12, size: (1920, 1080) };
        let offered = d.profiles().expect("vaQueryConfigProfiles").contains(&Profile::H264High)
            && d.entrypoints(Profile::H264High).expect("vaQueryConfigEntrypoints").contains(&Entrypoint::Vld);
        assert_eq!(offered, supports(&d, &plan).is_ok(), "the driver offers H.264 High through VLD: {offered}");
        // an entry point no driver offers for any profile: not a decode path, so never supported
        assert!(supports(&d, &Plan { entrypoint: Entrypoint::Other(4242), ..plan }).is_err(), "an entry point that is not VLD");
        // and a profile the driver does not list is declined, even for a codec we do take
        let unknown = Profile::Other(4241);
        assert!(!d.profiles().expect("vaQueryConfigProfiles").contains(&unknown), "libva has no profile {unknown:?}");
        assert!(supports(&d, &Plan { profile: unknown, ..plan }).is_err(), "a profile this driver does not list");
    }
}
