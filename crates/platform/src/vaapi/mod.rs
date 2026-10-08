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
//! `ffi.rs` is this backend's FFI module: the two libraries opened with `dlopen` and the `VA*`
//! structures. This module is safe code.

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

// What the rest of the backend needs; no libva type crosses this line. `OK` and the buffer-type
// constants are part of it too — the decode loop submits those buffers and compares statuses — as
// are the render formats a surface is created in. The padding widths stay for the same reason:
// they decide the size of every `VA*` structure, the picture-parameter records among them. The raw
// profile and entry point numbers do not: those are what `Profile`, `Entrypoint` and `RtFormat` are
// compared as.
pub use ffi::{
    Api, Display, Entrypoint, IMAGE_BUFFER, OK, PICTURE_PARAMETER_BUFFER, Profile, RtFormat, SLICE_DATA_BUFFER, SLICE_PARAMETER_BUFFER, VA_PADDING_LOW,
    VA_PADDING_MEDIUM, YUV420, YUV420_10, status,
};
// What the decoder itself drives: the stream it takes, what the driver would decode it with, the
// check that the driver really offers that, and the records one access unit is submitted as.
pub use picture::{NalPoc, Picture, SliceParam, build_frame, build_nal};
pub use stream::{MAX_SIDE, Plan, Stream, plan_frame, plan_nal, supports};
