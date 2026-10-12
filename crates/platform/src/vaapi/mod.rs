//! VA-API hardware video decoding on Linux (libva 2.x over a DRM render node).
//!
//! The GPU's video engine is reached through the driver's own media stack: libva opens the render
//! node, the driver says which profiles its decoder takes and what a decoded picture looks like in
//! memory, and the backend drives one picture at a time the way the Windows and macOS backends
//! drive their own. Pictures are copied back out of the driver and handed to
//! [`crate::HybridDecoder`], which keeps FilmCraft's own decoder as the fallback, so a stream the
//! driver declines (or fails on part-way) still plays.
//!
//! libva is LGPL and is therefore never linked: its symbols are resolved at run time out of the
//! system's own library, so nothing LGPL enters the tree or the binary, and a machine without
//! VA-API reports that it has no hardware decoder instead of failing to start
//! (docs/adr/0001-platform-ffi.md).
//!
//! `ffi.rs` is this backend's FFI module: the two libraries opened with `dlopen`, the `VA*`
//! structures and the entry points behind [`Session`], which owns a config, a ring of surfaces and
//! a decode context and is the only thing that holds a libva handle after [`Display`]. This module
//! is safe code.

#[allow(unsafe_code)]
mod ffi;
// How the encode readback gets at a libva buffer object's own storage: the descriptor FilmCraft
// already holds the display open on, asked DRM to map the object libva named. A second FFI module
// because it is a second ABI — plain DRM rather than libva — and the rule that `unsafe` lives only
// in FFI modules is what keeps that boundary where it is.
#[allow(unsafe_code)]
mod drm;
// Safe code with nothing OS-specific in it, so its tables could be tested on any machine. The module
// it lives in is Linux-only today (`ffi` needs `dlopen`), so that is where these tests run, and
// compiling them on macOS and Windows is tracked follow-up work: the allow below is what keeps that
// from needing changes in this file when the gate moves.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
mod stream;
// What a picture is submitted as, read out of FilmCraft's own parsers. Safe code with nothing
// OS-specific in it either, for the same reason as `stream` above.
mod picture;
// AV1: the reference slots 7.20 refreshes, the picture-parameter record, and the **per-tile** slice
// records. Its own file because `picture.rs` is where VP9's equivalent state lives and this is a
// second, materially different one: AV1's reference slot is read by the frame *header parser*, not
// only by the record, and its slice record is one per tile where VP9's is one per frame.
mod av1;
// The SPS and PPS for a picture this module encoded, derived from the records it submitted and
// verified by FilmCraft's own H.264 parser. It exists because iHD 26.3.5 on the machine this
// milestone was written for accepts the packed-header mode and then writes no parameter sets at all
// (see the module's own note, which has the measured evidence).
mod params;
// The decoder itself — the surfaces, the decode loop and the readback. Part of this module, split
// into its own file because the module is large; it is safe code like everything here but `ffi`.
mod decoder;
// The encoder's other half: the records that describe what to encode, and the drive of one
// picture through it. Encode inverts decode's data flow (the app hands the driver a picture and the
// driver writes the slice headers), so there is nothing to parse here and this module builds
// records instead. Safe code, like everything here but `ffi`.
mod encoder;
// HEVC encode records, declared and pinned but **not yet submitted** — the sequence, picture and
// slice records from `va_enc_hevc.h`. Its own file because a second writer on `ffi.rs` (which the
// H.264 rate-control change is editing) and on this one is what corrupted the tree earlier in this
// session, and because declaring a record needs no `unsafe` and so does not belong in the FFI
// module on its merits either. `hevc_abi`'s pins live inside it rather than in the shared
// `abi_guard.rs`, for the same reason.
mod hevc_enc;
// AV1 encode records, declared and pinned but **not yet submitted** — the six records of
// `va_enc_av1.h`, measured against the installed header. Its own file for the same reasons as
// `hevc_enc`: zero overlap with `ffi.rs` and with the shared `abi_guard.rs`, and declaring a record
// needs no `unsafe`.
mod av1_enc;
// **No VP9 encode records**, because a `dlopen` probe of this driver found no VP9 encoder: all four
// `VAProfileVP9Profile0`..`3` advertise `VAEntrypointVLD` and nothing else, and every
// `vaGetConfigAttributes` read against `VAEntrypointEncSlice`/`EncSliceLP` returns
// `VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT`. `va_enc_vp9.h` and `VAConfigAttribEncVP9` (58) *do*
// exist, so this is the driver declining a codec libva can express. This module holds that answer
// and its evidence rather than a record nobody could submit.
mod vp9_enc;
// The HEVC **submission** path: the frame loop, the records it builds per picture, the readback and
// the check that what the driver wrote agrees with what it was told. Beside `hevc_enc` for the same
// reason that one is beside `ffi.rs`: a second writer on one file is how the tree got corrupted
// earlier in this session, and `hevc_params` is a third file for the same reason.
mod hevc_params;
mod hevc_submit;

pub use decoder::{Timings, VaDecoder};
// VP9: the state one decode carries across its frames, and the record one frame is submitted as.
// Exported from here so that a VP9 decode is reachable without naming a `VA*` structure — `mod.rs`
// never names one. `Vp9State` is the entry point: it threads the one `HeaderState` the record is
// built from and owns the eight reference slots, which `build_frame` cannot be given (see
// `picture::build_frame`).
pub use encoder::{Current, EncodeRequest, EncodedPicture, MAX_REFERENCES, Reference, VaEncoder, macroblocks_of, ramp_picture, ramp_shifted};
// The HEVC submission path. `HevcEncoder` is the entry point — the counterpart of `VaEncoder` — and
// `HevcEncodeRequest` / `HevcEncodedPicture` are its own types rather than the H.264 ones, because
// HEVC has no `frame_num` to carry, no crop to report, and a `slice_type` whose encoding is not
// H.264's. `h264_slice_type_sweep_is_not_an_hevc_answer` in `tests/hevenc.rs` is what keeps them
// from drifting into each other.
// `PictureKind` is re-exported because `HevcEncodeRequest::new` takes one: **HEVC's own request type
// naming H.264's enum would be the drift this very comment warns about**, so the enum is shared and
// shared *on purpose* — it is `encoder::PictureKind`, the same one `EncodeRequest` takes, and the
// refusal of `B` lives in `HevcEncoder::encode` rather than in the type, so it stays reachable.
pub use hevc_submit::{
    HevcEncodeRequest, HevcEncodedPicture, HevcEncoder, HevcReference, PictureKind, REFUSES_THE_QUANTIZATION_MISC_BUFFER, hevc_rate_control_outcomes,
    slice_types,
};
// The constants the probe and the oracle test ask the driver about. Re-exported because
// `hevc_submit` is a private module, and a `pub` item in a private module is dead code the rest of
// the crate never sees; these four are the numbers a caller needs to check a driver against what
// this path assumes, and `examples/vahevcprobe.rs` prints all of them.
pub use hevc_submit::{CONFIG_ATTRIB_INTERLACED, H264_RATE_CONTROL_MASK, HEVC_RATE_CONTROL_MASK, VCM};
// **The packed-header constants, re-exported because they are the two numbers this milestone turned
// on and they are easy to confuse with each other**: `VA_ENC_PACKED_HEADER_SLICE` (4) is the config
// attribute that says slice headers will be *supplied*, and `VAEncPackedHeaderSlice` (3) is the type
// a packed-header *buffer* carries. `SEQUENCE | PICTURE` is 3 in both families, which is why the
// wrong flag reads as a plausible number and produces no picture at all.
pub use hevc_enc::{PACKED_HEADER_SLICE as PACKED_HEADER_SLICE_BUFFER_TYPE, PACKED_SLICE_FLAG as PACKED_HEADER_SLICE_TYPE};
pub use hevc_submit::PACKED_HEADERS_FOR_HEVC;
pub use picture::{Vp9Frame, Vp9Reference, Vp9ReferenceSet, Vp9State, vp9_decline, vp9_is_new_frame, vp9_picture_record, vp9_slice_record};
// AV1, the same three shapes for the same three reasons: `Av1State` is the per-decode reference
// state (`refresh_frame_flags` has no member in the AV1 *decode* record — see `picture::av1`'s
// module note for the header lines that is checked against), `Av1ReferenceSet` is the eight slots the
// record's `ref_frame_map` is, and `av1_picture_record` is the record itself. `Av1State::picture_record`
// is the entry point rather than the free function, because the surface the record names is only known
// once the frame has been read — a re-show codes no picture and needs none.
pub use av1::{AV1_FILM_GRAIN, AV1_LARGE_SCALE_TILE, Av1Frame, Av1ReferenceSet, Av1State, av1_entry_disagreement, av1_picture_record, av1_surface_pool};
// What a caller outside this module needs to build a decoder: [`Display`] is the initialised libva
// connection a session is created on; [`stream_info`] reads a sample entry into the codec-agnostic
// `StreamInfo` the rest of the workspace speaks, and [`Stream`] is this backend's own view of a
// stream — a `StreamInfo` converts into it (`impl From<StreamInfo> for Stream`), so [`VaDecoder::new`]
// takes either; and [`build_frame`] builds the picture-parameter record for a frame-coded sample,
// which every frame codec is declined before it can reach, and which is public so that the decline
// and the builder cannot drift apart (see `picture::build_frame`).
pub use ffi::Display;
// `Profile` and `Entrypoint` are what the encoder's capability queries are asked about, and
// `Entrypoint::EncSlice` in particular is the question the whole encode path gates on.
pub use ffi::status;
pub use ffi::{Entrypoint, Profile};
pub use picture::build_frame;
pub use stream::Stream;
pub use stream::stream_info;

use ffi::VaImage;

/// Whether libva and a DRM render node exist on this system, and so whether there is a hardware
/// decoder here at all. A machine without them reports that it has none rather than failing to start.
pub fn available() -> bool {
    ffi::load().and_then(|api| Display::new(&api)).is_ok()
}

/// What the driver says it can hold references for, as `(reference list 0, reference list 1)`.
///
/// The header's attribute is **`VAConfigAttribEncMaxRefFrames`** (`va.h:734-745`) — there is no
/// `VAConfigAttribEncMaxReferenceFrames`, and the H.264 answer packs list 0 into the low 16 bits
/// and list 1 into the high 16 (`va.h:737-739`).
///
/// It is asked of the *profile and entry point* through `vaGetConfigAttributes` rather than read off
/// a config, because it is a read-only attribute and libva gives no way to read one off a config that
/// did not set it. `(0, 0)` means "unknown" — either the driver said zero or libva cannot answer —
/// and it is not a claim that the driver supports nothing: see [`VaEncoder::new`], which treats a
/// zero as "cannot tell" and works at the depth the caller asked for.
///
/// On the machine this milestone was written for (Intel Arc B580, iHD 26.3.5) this answers
/// **(1, 1)**: one reference in each list, which is exactly enough for the P chain
/// `examples/vaencprobe.rs` measures.
pub fn max_reference_frames(display: &Display) -> (u32, u32) {
    ffi::Encoder::driver_max_reference_frames(display, ffi::Profile::H264Main)
}

/// An initialised VA-API display on this machine's DRM render node, or why there is none.
///
/// The same call [`VaDecoder::new`] makes internally. It is public because the encode path needs a
/// display of its own — the decoder's is created and owned inside the decoder — and a caller
/// driving [`VaEncoder`] directly (the encode milestone's probe, `examples/vaencprobe.rs`) needs one
/// to hand it.
///
/// It returns the whole [`Display`] rather than the `Api` behind it, because a `Display` is the safe
/// wrapper: no libva handle leaves [`ffi`].
pub fn open_display() -> Result<std::sync::Arc<Display>, String> {
    let api = ffi::load()?;
    Display::new(&api)
}

/// How many pictures a stream's sequence parameter set says its decoded picture buffer holds, or why
/// that cannot be counted.
///
/// **This is not the number of surfaces the decode loop allocates.** The loop allocates
/// [`picture::surface_pool`], which comes from the record's own reference-slot count rather than from
/// the declaration, because the declaration is not a bound the buffer respects — HEVC decides what
/// stays a reference from each picture's own set, and consecutive sets can name more pictures between
/// them than `sps_max_dec_pic_buffering_minus1` counts. A decoder's pool is therefore normally
/// larger than what this reports, and `examples/vaprobe.rs` prints this number under "decoded picture
/// buffer declares", which is about the stream and not about the decoder.
///
/// It is exposed on its own because it is answerable from a stream description alone, before any
/// display is opened, so a caller can ask on a machine with no GPU at all.
pub fn surfaces_for(info: &filmcraft_codecs::hw::NalStreamInfo) -> Result<usize, String> {
    picture::reference_pictures(info)
}

/// What this driver's decoder offers, for diagnostics (`examples/vaprobe.rs --caps`): the vendor
/// string, the libva version, and every profile the driver names with a yes or no for whether it
/// offers the video decoder for it.
///
/// The whole profile list is reported, including the ones this backend does not decode: a user asking
/// why a clip falls back to software wants to know whether the machine can do it at all, and a
/// profile this milestone has no records for (`VP9`, `AV1`) is a different answer from one no driver
/// offers.
pub fn capabilities() -> String {
    let (api, display) = match ffi::load().and_then(|api| Display::new(&api).map(|d| (api, d))) {
        Ok(pair) => pair,
        Err(why) => return format!("no VA-API: {why}"),
    };
    let _ = &api;
    let (major, minor) = display.va_version();
    let mut out = format!("vendor: {}\nlibva: {major}.{minor}\n", display.vendor().unwrap_or("(none)"));
    let profiles = match display.profiles() {
        Ok(profiles) => profiles,
        Err(why) => {
            out += &format!("profiles: {why}\n");
            return out;
        }
    };
    for profile in profiles {
        // A profile whose entry points cannot be queried is reported as offering nothing rather than
        // skipped: a driver that answers the profile list and then refuses the entry point list is a
        // driver whose capability is unknown, and "unknown" belongs in the report.
        let entrypoints = display.entrypoints(profile);
        let has = |want: ffi::Entrypoint| entrypoints.as_ref().map(|e| e.contains(&want)).unwrap_or(false);
        // `EncSlice` is reported next to `Vld` because the encode backend gates on exactly this
        // question -- whether the driver offers a slice encoder for the profile -- and a capability
        // report that cannot answer it is not one the encode path can decide against.
        out += &format!("{profile:?} VLD: {} EncSlice: {}\n", has(ffi::Entrypoint::Vld), has(ffi::Entrypoint::EncSlice));
    }
    out
}

/// The surfaces a decoder decodes into, handed out one at a time.
///
/// A picture is decoded into a surface and read back out of it before the next picture is decoded,
/// but a surface a later picture still refers to may not be decoded into: the driver would read the
/// new picture as that reference. So the ring does not hand surfaces out on a bare rotation — it
/// hands out one that is not in the decoded picture buffer, and the caller passes that buffer in
/// because only the picture-record builder knows what it holds.
///
/// The rotation is still what decides *which* of the free surfaces: starting from where the last one
/// left off means consecutive pictures land in different surfaces, and it means the search is bounded
/// by the number of surfaces rather than by anything the stream says.
struct Ring {
    surfaces: Vec<u32>,
    next: usize,
}

impl Ring {
    /// A ring over the ids `vaCreateSurfaces` filled in — which is the only source of surface ids
    /// that is guaranteed right. libva does not promise they are contiguous, and a surface id this
    /// code invented rather than read would name a surface the driver never created.
    fn from_surfaces(surfaces: &[u32]) -> Self {
        Self { surfaces: surfaces.to_vec(), next: 0 }
    }

    /// The next surface that is not one of `live` — the surfaces the decoded picture buffer still
    /// refers to. `None` when every surface is referenced, which is the one thing that cannot be
    /// decoded into anything: the caller turns it into an error rather than overwriting a reference.
    ///
    /// The search starts at the cursor and wraps, so it visits each surface once and gives up after
    /// `len()` steps whatever the stream says.
    fn take_unreferenced(&mut self, live: &[u32]) -> Option<u32> {
        let count = self.surfaces.len();
        for step in 0..count {
            let at = (self.next + step) % count;
            let surface = self.surfaces.get(at).copied()?;
            if !live.contains(&surface) {
                self.next = (at + 1) % count;
                return Some(surface);
            }
        }
        None
    }

    /// How many surfaces the ring holds.
    fn len(&self) -> usize {
        self.surfaces.len()
    }
}

/// The one gate every readback goes through: whether `img` describes a picture FilmCraft can read
/// out of the buffer it says the pixels are in.
///
/// Everything in a `VaImage` was written by the driver, so every number in it is hostile and none of
/// it is indexed, added or multiplied without a bound. A `VaImage` that fails here is refused before
/// a single row is read out of the mapped buffer: the alternative is a row read that runs off the
/// end of memory the driver sized, which is the one thing in this backend that cannot be caught
/// afterwards.
///
/// What is checked, and why each of them:
///
/// - `num_planes == 2`: 4:2:0 is biplanar. Three planes is planar 4:2:0 and one is greyscale, and
///   [`crate::biplanar::to_frame`] reads a luma plane followed by one interleaved chroma plane, so
///   neither is something this readback path can hand it.
/// - `format.bits_per_pixel` is [`NV12_BPP`] or [`P010_BPP`]: the two values libva reports for
///   4:2:0 at 8 and 10 bits (see the constant for why those and not 8 and 10). This is also what
///   says how wide a sample is, and so how wide a row has to be to hold one.
/// - `width` and `height` are not zero: a picture with no rows has no picture in it.
/// - `offsets[0] == 0`: the luma plane is the front of the buffer. A driver that says otherwise is
///   describing a layout this path does not model, and reading it as if the first plane began at
///   zero would read whatever happens to be there.
/// - each plane's pitch is at least one row of that plane: a pitch narrower than a row is the
///   Review Focus #1 defect, where reading a row of the picture walks off the end of the plane.
/// - each plane ends inside `data_size`: a `vaMapBuffer` that succeeded says the buffer is mapped,
///   not that the driver's geometry fits inside it, so a bogus `data_size` or an offset that
///   overflows is the Review Focus #2 defect. `checked_mul` / `checked_add` throughout: a driver
///   that overflows an offset is not to be trusted into a wraparound.
fn validate_image(img: &VaImage) -> Result<(), String> {
    if img.num_planes != 2 {
        return Err(format!("{} planes, not the 2 of a biplanar 4:2:0 picture", img.num_planes));
    }
    // Bytes per sample: NV12 holds one byte per sample, P010 a 16-bit word with the significant
    // bits in the high half. The chroma plane covers every second column and row, so its row is
    // half as many samples again over the same pitch as the luma row.
    let sample_bytes = match img.format.bits_per_pixel {
        ffi::NV12_BPP => 1usize,
        ffi::P010_BPP => 2usize,
        other => return Err(format!("{other} bits per pixel is not 4:2:0 8/10-bit")),
    };
    let (width, height) = (usize::from(img.width), usize::from(img.height));
    if width == 0 || height == 0 {
        return Err(format!("{width}x{height} picture"));
    }
    let data_size = img.data_size as usize;
    if img.offsets[0] != 0 {
        return Err(format!("the luma plane starts at offset {} of the buffer", img.offsets[0]));
    }
    // The two planes are pitched the same. libva does not promise it, but `biplanar::Biplanar` — which
    // is what the copy is made through — carries a single stride for both planes, so an image whose
    // planes are pitched differently cannot be read at all rather than read wrongly. `with_mapped`
    // checks this again where it indexes, so the refusal holds even if a caller skips this gate.
    if img.pitches[0] != img.pitches[1] {
        return Err(format!("luma pitch {} and chroma pitch {} differ", img.pitches[0], img.pitches[1]));
    }
    for plane in 0..2usize {
        let (Some(&pitch), Some(&offset)) = (img.pitches.get(plane), img.offsets.get(plane)) else {
            return Err(format!("plane {plane} has no pitch or offset"));
        };
        let (rows, cols) = if plane == 0 { (height, width) } else { (height.div_ceil(2), width.div_ceil(2)) };
        let row_bytes = cols.checked_mul(sample_bytes).ok_or_else(|| format!("plane {plane} row width overflows"))?;
        if (pitch as usize) < row_bytes {
            return Err(format!("plane {plane} pitch {pitch} is under the {row_bytes} bytes a row of {cols} samples needs"));
        }
        let plane_bytes = (pitch as usize).checked_mul(rows).ok_or_else(|| format!("plane {plane} of {rows} rows overflows"))?;
        let end = (offset as usize).checked_add(plane_bytes).ok_or_else(|| format!("plane {plane} ends past the end of the buffer"))?;
        if end > data_size {
            return Err(format!("plane {plane} ends at byte {end} of a {data_size} byte buffer"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffi::{NV12_BPP, P010_BPP, VaImageFormat};

    /// A `VaImage` that is internally consistent for a 4:2:0 picture of `w` x `h` with the given
    /// luma pitch: two planes, chroma interleaved so it uses the same pitch over half as many rows,
    /// and `data_size` covering both.
    fn image(w: u16, h: u16, pitch_luma: u32) -> VaImage {
        let luma_bytes = pitch_luma.checked_mul(u32::from(h)).unwrap_or(0);
        let chroma_bytes = pitch_luma.checked_mul(u32::from(h) / 2).unwrap_or(0);
        VaImage {
            image_id: 0,
            format: VaImageFormat {
                fourcc: 0,
                byte_order: 0,
                bits_per_pixel: NV12_BPP,
                depth: 0,
                red_mask: 0,
                green_mask: 0,
                blue_mask: 0,
                alpha_mask: 0,
                va_reserved: [0; ffi::VA_PADDING_LOW],
            },
            buf: 0,
            width: w,
            height: h,
            data_size: luma_bytes.saturating_add(chroma_bytes),
            num_planes: 2,
            pitches: [pitch_luma, pitch_luma, 0],
            offsets: [0, luma_bytes, 0],
            num_palette_entries: 0,
            entry_bytes: 0,
            component_order: [0; 4],
            va_reserved: [0; ffi::VA_PADDING_LOW],
        }
    }

    #[test]
    fn the_surface_ring_skips_the_surfaces_the_picture_buffer_still_holds() {
        // The round-robin is what picks among the free ones; what it must never do is hand back a
        // surface the driver is going to read as a reference. This is the defect the ring existed to
        // invite: a bare rotation on a 6-surface ring with 5 live references eventually lands on a
        // live one, and the driver decodes the picture into the surface it is reading as a reference.
        let mut ring = Ring::from_surfaces(&[10, 11, 12, 13]);
        assert_eq!(ring.take_unreferenced(&[]), Some(10));
        assert_eq!(ring.take_unreferenced(&[]), Some(11));
        // 12 and 13 are free, so the rotation reaches them...
        assert_eq!(ring.take_unreferenced(&[]), Some(12));
        assert_eq!(ring.take_unreferenced(&[]), Some(13));
        // ...and comes round to 10, which is still free.
        assert_eq!(ring.take_unreferenced(&[]), Some(10));
        // But with 10 and 11 live, neither is handed out again.
        assert_eq!(ring.take_unreferenced(&[10, 11]), Some(12));
        assert_eq!(ring.take_unreferenced(&[10, 11]), Some(13));
        // And with every surface live there is nothing to decode into, which is an error at the
        // call site rather than a reference being overwritten.
        assert_eq!(ring.take_unreferenced(&[10, 11, 12, 13]), None);
    }

    /// The ring is built from the ids the driver filled in, not from an arithmetic guess: libva does
    /// not promise surface ids are contiguous, and an invented id names a surface that does not
    /// exist.
    #[test]
    fn the_ring_holds_exactly_the_ids_the_driver_returned() {
        let mut ring = Ring::from_surfaces(&[7, 900, 3, 41]);
        let mut got = Vec::new();
        // Exactly `len()` hand-outs: with nothing live every surface is free, so a `while let` here
        // would never end.
        for _ in 0..ring.len() {
            got.push(ring.take_unreferenced(&[]).unwrap_or(0));
        }
        assert_eq!(got, vec![7, 900, 3, 41], "the driver's own ids, in the order it gave them");
        assert_eq!(ring.len(), 4);
        assert_eq!(Ring::from_surfaces(&[]).take_unreferenced(&[]), None, "an empty ring has nothing");
    }

    #[test]
    fn a_pitch_narrower_than_a_row_is_refused() {
        // Review Focus #1: a driver pitch below one row would make row reads run off the plane.
        assert!(validate_image(&image(640, 360, 8)).is_err(), "pitch 8 < 640");
        assert!(validate_image(&image(640, 360, 639)).is_err(), "pitch 639 < 640");
        assert!(validate_image(&image(640, 360, 640)).is_ok());
        assert!(validate_image(&image(640, 360, 1280)).is_ok(), "a driver-aligned wider pitch");
    }

    #[test]
    fn an_image_whose_planes_overrun_its_buffer_is_refused() {
        // Review Focus #2: vaMapBuffer succeeded but the driver reported a bogus data_size.
        let mut img = image(640, 360, 640);
        img.data_size = 16;
        assert!(validate_image(&img).is_err(), "planes beyond data_size");
        // a plane offset past the end
        let mut img = image(640, 360, 640);
        img.offsets[1] = u32::MAX;
        assert!(validate_image(&img).is_err(), "offset out of range");
        // an offset that overflows when added to a pitch
        let mut img = image(640, 360, 640);
        img.offsets[1] = u32::MAX - 8;
        assert!(validate_image(&img).is_err(), "offset + pitch overflows");
    }

    #[test]
    fn a_surface_that_is_not_two_planes_or_the_wrong_depth_is_refused() {
        let mut img = image(640, 360, 640);
        img.num_planes = 0;
        assert!(validate_image(&img).is_err(), "no planes");
        let mut img = image(640, 360, 640);
        img.num_planes = 3;
        assert!(validate_image(&img).is_err(), "three planes");
        let mut img = image(640, 360, 640);
        img.format.bits_per_pixel = 16;
        assert!(validate_image(&img).is_err(), "16 bits per pixel is not 4:2:0 8/10-bit");
        let mut img = image(640, 360, 640);
        img.format.bits_per_pixel = P010_BPP;
        assert!(validate_image(&img).is_err(), "a 10-bit image in a surface the picture fills 8-bit rows of");
        assert!(validate_image(&image(0, 360, 640)).is_err(), "zero width");
        assert!(validate_image(&image(640, 0, 640)).is_err(), "zero height");
        // The two planes are read through one stride (`biplanar::Biplanar` has a single `stride`),
        // so an image whose planes are pitched differently cannot be read at all rather than read at
        // the wrong stride. `with_mapped` re-checks this where it indexes.
        let mut img = image(640, 360, 640);
        img.pitches[1] = 1280;
        img.data_size = 640 * 360 + 1280 * 180;
        assert!(validate_image(&img).is_err(), "a chroma pitch the copy cannot express");
        // The other way round is refused too: it is the pair that has to agree, not one order.
        let mut img = image(640, 360, 640);
        img.pitches[0] = 1280;
        img.pitches[1] = 640;
        img.data_size = 1280 * 360 + 640 * 180;
        assert!(validate_image(&img).is_err(), "and a luma pitch the chroma does not share");
    }

    /// **Every item an AV1 picture-parameter record is built out of is reachable from outside
    /// `filmcraft-av1`**, and this test is what holds that.
    ///
    /// The failure it guards against is not a wrong number but a *missing name*: the record builder
    /// cannot be written until the uncompressed frame header is, and `FrameHeader::parse` used to be
    /// `pub(crate)` over a `pub(crate)` `BitReader`, so the only way to reach a frame header was to
    /// reach into the crate. Every item below is named *here*, from another crate, so a change to
    /// `crates/av1`'s visibility fails to **compile in this file** rather than surfacing later as a
    /// confusing error part way through the builder.
    ///
    /// Each item is named for the record field it fills, which is why this list exists at all: an item
    /// that is reachable but not named by any caller is surface nobody has asked for, and nothing in
    /// the build can tell the difference.
    #[test]
    fn the_av1_frame_header_parser_is_reachable() {
        // `BitReader` reads the uncompressed header out of the frame's own bytes, so the record is
        // derived from the frame rather than from a parameter set.
        let mut r = filmcraft_av1::BitReader::new(&[0x80, 0x00, 0x00, 0x00]);
        assert_eq!(r.f(1), Ok(1), "the first bit of a key frame's uncompressed header");
        assert_eq!(r.position(), 1);
        // `FrameHeader::parse` is the parse the software decoder runs; the record reads its result.
        // The signature is named rather than called, because calling it needs a parsed sequence header.
        let _parse: fn(&mut filmcraft_av1::BitReader, &mut filmcraft_av1::header::HeaderState, u32, u32) -> filmcraft_av1::Result<filmcraft_av1::FrameHeader> =
            filmcraft_av1::FrameHeader::parse;
        // `HeaderState::new` seeds the cross-frame state exactly as `filmcraft_av1::Decoder::new`
        // does, so the record is built against the state the software decode runs.
        let mut refs: [filmcraft_av1::header::RefInfo; filmcraft_av1::NUM_REF_FRAMES] = std::array::from_fn(|_| filmcraft_av1::header::RefInfo::default());
        let seq = filmcraft_av1::SequenceHeader::parse(&[0x0a, 0x00, 0x00, 0x00, 0x24, 0xcf, 0x7f, 0x0d, 0xbf, 0xff, 0x30, 0x08])
            .unwrap_or_else(|e| panic!("a sequence header: {e}"));
        let st = filmcraft_av1::header::HeaderState::new(&seq, &mut refs);
        assert_eq!(st.current_frame_id, 0, "a fresh state has decoded no frame");
        assert_eq!(st.prev_gm_params[0][2], 1 << filmcraft_av1::WARPEDMODEL_PREC_BITS, "global motion starts at the identity translation scale");
        // The array bounds the record's own arrays, so they are named rather than written twice.
        assert_eq!([(); filmcraft_av1::REFS_PER_FRAME].len(), 7, "ref_frame_idx[]");
        assert_eq!([(); filmcraft_av1::NUM_REF_FRAMES].len(), 8, "ref_frame_map[]");
        assert_eq!([(); filmcraft_av1::MAX_SEGMENTS].len(), 8, "feature_data[][]");
        assert_eq!([(); filmcraft_av1::SEG_LVL_MAX].len(), 8, "a feature row");
        // `primary_ref_frame` is 7 when there is no primary reference; a record that wrote 0 there
        // would load propagated info from the LAST slot instead of loading none.
        assert_eq!(filmcraft_av1::PRIMARY_REF_NONE, 7, "PRIMARY_REF_NONE");
        // The frame types the record's `frame_type` field names, in the header's own order
        // (`va_dec_av1.h:416-419`), so a packed bit can be read back as a name.
        assert_eq!([filmcraft_av1::KEY_FRAME, filmcraft_av1::INTER_FRAME, filmcraft_av1::INTRA_ONLY_FRAME, filmcraft_av1::SWITCH_FRAME], [0, 1, 2, 3]);
    }
}
