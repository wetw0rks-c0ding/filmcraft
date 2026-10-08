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
// Safe code with nothing OS-specific in it, so its tables could be tested on any machine. The module
// it lives in is Linux-only today (`ffi` needs `dlopen`), so that is where these tests run, and
// compiling them on macOS and Windows is tracked follow-up work: the allow below is what keeps that
// from needing changes in this file when the gate moves.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
mod stream;
// What a picture is submitted as, read out of FilmCraft's own parsers. Safe code with nothing
// OS-specific in it either, for the same reason as `stream` above.
mod picture;
// The decoder itself — the surfaces, the decode loop and the readback. Part of this module, split
// into its own file because the module is large; it is safe code like everything here but `ffi`.
mod decoder;

pub use decoder::{Timings, VaDecoder};
// What a caller outside this module needs to build a decoder and to ask what this machine has:
// the display (the initialised libva connection), the stream description (which converts straight
// into [`VaDecoder::new`]), the two diagnostics functions, and the record builder for a frame-coded
// sample — which every frame codec is declined before it can reach, and which is public so that the
// decline and the builder cannot drift apart (see `picture::build_frame`).
pub use ffi::Display;
pub use ffi::status;
pub use picture::build_frame;
pub use stream::Stream;

use ffi::VaImage;

/// Whether libva and a DRM render node exist on this system, and so whether there is a hardware
/// decoder here at all. A machine without them reports that it has none rather than failing to start.
pub fn available() -> bool {
    ffi::load().and_then(|api| Display::new(&api)).is_ok()
}

/// How many surfaces a stream needs, or why they cannot be counted: the size of the decoded picture
/// buffer its own sequence parameter set declares, which is what the decode loop allocates. This is
/// [`decoder::VaDecoder::surfaces`]'s input, exposed on its own because it is answerable from a
/// stream description alone — before any display is opened, so a caller can ask on a machine with no
/// GPU at all (`examples/vaprobe.rs` does).
pub fn surfaces_for(info: &filmcraft_codecs::hw::NalStreamInfo) -> Result<usize, String> {
    picture::reference_pictures(info)
}

/// What this driver's decoder offers, for diagnostics (`examples/vacaps.rs`): the vendor string, the
/// libva version, and every profile the driver names with a yes or no for whether it offers the video
/// decoder for it.
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
        let vld = display.entrypoints(profile).map(|e| e.contains(&ffi::Entrypoint::Vld)).unwrap_or(false);
        out += &format!("{profile:?} VLD: {vld}\n");
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
}
