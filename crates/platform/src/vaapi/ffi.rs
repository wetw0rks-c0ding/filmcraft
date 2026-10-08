//! The VA-API FFI layer: the symbols of libva and of its DRM helper, resolved at run time, and the
//! `VA*` structures, constants and status codes the decoder passes across.
//!
//! FFI module (docs/adr/0001-platform-ffi.md): every `unsafe` block has a `// SAFETY:` comment, the
//! public items are safe and return `Result<_, String>`, and no libva type leaves this module —
//! `VADisplay`, the `VA*ID` handles and the function pointers are implementation details of this
//! file.
//!
//! libva is never linked (it is LGPL): the system's own library is opened with `dlopen` when the
//! backend first asks for it, exactly as the Windows backend opens `mfplat.dll` and NVENC's
//! driver, so a machine without VA-API still starts and decodes in software.
//!
//! Two libraries are involved, and the split matters: everything except `vaGetDisplayDRM` is in
//! `libva.so.2`, and `vaGetDisplayDRM` — which turns a DRM file descriptor into a display — is in
//! `libva-drm.so.2`. `dlclose` runs only when the last [`Api`] is gone, which is what keeps a
//! function pointer from outliving the code it points at; [`Display`] holds an `Arc<Api>` for the
//! same reason.
//!
//! `vaInitialize` is mandatory before anything else is asked of a display (the library segfaults
//! otherwise) and its status is checked: [`Display::new`] is where that happens.
//!
//! The structures and constants are hand-declared from `/usr/include/va/` (libva 2.x) rather than
//! generated, so the tests at the bottom of this file check them twice over: the constants and the
//! layout are pinned against the header, and the layout is compared with the library that is
//! actually installed ([`Display::probe_image_layout`]) — libva has no `sizeof` / `offsetof` of its
//! own to ask, so a behavioural check is the only way to hold the declaration to the driver.

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::mem::size_of;
use std::sync::{Arc, MutexGuard, OnceLock};

// ---------------------------------------------------------------------------------------------
// libva's own types, none of which leaves this module.
// ---------------------------------------------------------------------------------------------

/// `VADisplay` (`va.h:260`): an opaque, window-system dependent display handle, `void *` to us.
type VADisplay = *mut c_void;

/// `VAGenericID` (`va.h:1580`): the `unsigned int` behind every libva handle. The five handles are
/// named apart below because the C API does — a `VABufferID` in a call out is not a `VAConfigID` —
/// and the three the decode loop has not called yet are `allow(dead_code)` rather than unnamed.
type VAGenericID = u32;
#[allow(dead_code)]
type VAConfigID = VAGenericID;
#[allow(dead_code)]
type VAContextID = VAGenericID;
#[allow(dead_code)]
type VASurfaceID = VAGenericID;
type VABufferID = VAGenericID;
type VAImageID = VAGenericID;

/// `VAStatus` (`va.h:262`): what every libva entry point returns.
type VAStatus = c_int;
/// The two libva enumerations the decoder asks about, which C declares as plain `int`.
type VAProfile = c_int;
type VAEntrypoint = c_int;

/// `VAConfigAttrib` (`va.h:1072`): one configuration attribute — a type and its value. Eight bytes.
/// This backend passes none: `vaCreateConfig` takes a null list with a zero count for the defaults,
/// which is what a decoder wants (the profile, the entry point and the render format are the three
/// things it does choose, and those are arguments rather than attributes).
#[repr(C)]
#[derive(Default)]
pub struct VaConfigAttrib {
    type_: VAStatus,
    value: c_uint,
}

/// `VASurfaceAttrib` (`va.h:1742`): one surface attribute — a type, the flags saying whether it can
/// be read or written, and a value whose meaning the type decides. 16 bytes, laid out as the union
/// `VAGenericValue` the header ends the struct with.
///
/// This backend passes none, either: `vaCreateSurfaces` takes a null list with a zero count, and the
/// driver allocates surfaces in the default (`VA_SURFACE_ATTRIB_MEM_TYPE_VA`) memory type, which is
/// the only type a mapped readback path can rely on. It is declared because
/// `vaCreateSurfaces` takes the array's type, so a caller that passes nothing still has to hand the
/// library a pointer to *something*.
#[repr(C)]
#[derive(Default)]
pub struct VaSurfaceAttrib {
    type_: VAStatus,
    flags: c_uint,
    value: VaGenericValue,
}

/// `VAGenericValue` (`va.h:245`): the eight-byte value a `VASurfaceAttrib` carries, as the union
/// the header declares it (a 64-bit integer, or a pointer of the same width). Only its size is ever
/// used here, and only as the tail of a `VASurfaceAttrib` a null list points past.
#[repr(C)]
union VaGenericValue {
    integer: u64,
    pointer: *mut c_void,
}

impl Default for VaGenericValue {
    fn default() -> Self {
        Self { integer: 0 }
    }
}

// ---------------------------------------------------------------------------------------------
// Constants, all read out of `/usr/include/va/` on the target machine (libva 2.24).
// ---------------------------------------------------------------------------------------------

/// `VA_STATUS_SUCCESS` (`va.h:264`): the status every entry point returns on success.
pub const OK: i32 = 0;

/// `VA_PADDING_LOW` / `VA_PADDING_MEDIUM` (`va.h:360-361`): the reserved-field widths libva
/// pads its structures with, which is what makes them a stable size.
pub const VA_PADDING_LOW: usize = 4;
/// See [`VA_PADDING_LOW`].
pub const VA_PADDING_MEDIUM: usize = 8;

/// `VA_RT_FORMAT_YUV420` (`va.h:1083`): 8-bit 4:2:0.
pub const YUV420: u32 = 0x0000_0001;
/// `VA_RT_FORMAT_YUV420_10` (`va.h:1088`): 10-bit 4:2:0. `0x100`, *not* a packed bit count: the
/// low byte is the chroma subsampling and the byte above it the bit depth. Easy to get wrong by
/// orders of magnitude, so it is pinned by `the_format_constants_are_the_ones_libva_uses`.
pub const YUV420_10: u32 = 0x0000_0100;

/// `VA_FOURCC_NV12` (`va.h:4425`) and `VA_LSB_FIRST` (`va.h:4714`): the image format the ABI
/// probe asks the driver for.
const FOURCC_NV12: u32 = 0x3231_564e;
const BYTE_ORDER_LSB_FIRST: u32 = 1;

/// `VA_FOURCC_NV12` (`va.h:4425`): the four-character code of an 8-bit biplanar 4:2:0 image. The
/// readback path asks for this by name (and for [`FOURCC_P010`] at 10 bits) so a driver that
/// derives something else — NV21, say, which is the same picture with U and V swapped — is
/// refused rather than read as if the components were in the order FilmCraft expects.
pub const NV12_FOURCC: u32 = FOURCC_NV12;
/// `VA_FOURCC_P010` (`va.h:4478`): the 10-bit biplanar 4:2:0 image, each sample a 16-bit word with
/// the 10 significant bits in the high half.
pub const P010_FOURCC: u32 = 0x3031_3050;

/// `VAImageFormat.bits_per_pixel` as libva reports it for a 4:2:0 image: **12** for NV12 and
/// **24** for P010.
///
/// Not "8" and "10": the field counts the bits a *luma and chroma sample together* take across the
/// two planes, which for 4:2:0 is twice the sample depth (8 + 4 + 4, or 16 + 4 + 4). Every driver
/// agrees on this — an Intel iHD driver at libva 1.24 reports exactly these two for NV12 and P010
/// — so they are the two values [`crate::vaapi::validate_image`] takes, and anything else is a
/// picture this backend does not know how to read.
pub const NV12_BPP: u32 = 12;
/// See [`NV12_BPP`].
pub const P010_BPP: u32 = 24;

/// `VA_STATUS_ERROR_OPERATION_FAILED` (`va.h:266`): the status `vaDeriveImage` is documented to
/// return when a surface cannot be represented as a `VAImage`, which is the signal to read the
/// picture the indirect way instead (`vaCreateImage` + `vaPutImage`). See [`Session::read`].
pub const OPERATION_FAILED: i32 = 0x1;
/// `VA_STATUS_ERROR_HW_BUSY` (`va.h:291`): the video engine was resetting. `vaSyncSurface` reports
/// it while the reset runs and succeeds on a later call, so [`Session::sync`] asks again rather
/// than reporting a decode failure the stream may not have had.
pub const HW_BUSY: i32 = 0x22;

/// `VASurfaceRendering` / `VASurfaceDisplaying` / `VASurfaceReady` (`va.h:4296-4299`): what
/// `vaQuerySurfaceStatus` says about a surface. `READY` is the state a surface has to be in before
/// its pixels can be read.
pub const SURFACE_READY: u32 = 0x4;

/// `VA_PROGRESSIVE` (`va.h:1920`): the only frame-pictures flag a decoder passes to
/// `vaCreateContext`. Every stream this backend takes is progressive (`plan_nal` declines the rest),
/// so this is the whole flag word.
pub const PROGRESSIVE: c_int = 0x1;

/// `VABufferType` (`va.h:2042`): the buffer kinds a decode loop submits, in the order libva
/// numbers them.
pub const PICTURE_PARAMETER_BUFFER: VAStatus = 0;
/// `VAIQMatrixBufferType` (`va.h:2043`): the H.264 scaling lists (`VAIQMatrixBufferH264`, `va.h`).
/// **A decode that does not submit one of these does not decode.** The iHD driver takes a
/// submission without it, reports success at every call, and leaves the render target holding the
/// cleared surface — a uniform mid-grey picture, which is the plausible-looking wrong answer this
/// backend must never hand back. See [`crate::vaapi::picture`]'s `h264_iq_matrix`, which builds it
/// from the sequence parameter set's own lists.
pub const IQ_MATRIX_BUFFER: VAStatus = 1;
/// See [`PICTURE_PARAMETER_BUFFER`].
pub const SLICE_PARAMETER_BUFFER: VAStatus = 4;
/// See [`PICTURE_PARAMETER_BUFFER`].
pub const SLICE_DATA_BUFFER: VAStatus = 5;

/// `VA_SLICE_DATA_FLAG_ALL` (`va.h:3080`): the whole slice is in the buffer. Every slice record
/// this backend builds carries its slice whole, so this is the flag they all get; the other
/// `VA_SLICE_DATA_FLAG_*` values (`BEGIN`, `MIDDLE`, `END`) describe a slice submitted in
/// pieces, which a single-frame-at-a-time decoder never does.
pub const SLICE_DATA_FLAG_ALL: u32 = 0x0;

/// `VAPictureH264::flags` bits (`va.h:3583-3592`).
pub const PICTURE_H264_SHORT_TERM_REFERENCE: u32 = 0x0000_0008;
/// See [`PICTURE_H264_SHORT_TERM_REFERENCE`].
pub const PICTURE_H264_LONG_TERM_REFERENCE: u32 = 0x0000_0010;
/// `VA_PICTURE_H264_INVALID` (`va.h:3583`): the entry is not a picture the driver may read. It is
/// what an unused `ReferenceFrames` slot or reference list entry carries, which is a defined value
/// rather than a surface id that might name a real but wrong surface.
pub const PICTURE_H264_INVALID: u32 = 0x0000_0001;

/// `VA_PICTURE_HEVC_INVALID` (`va.h:5310`): see [`PICTURE_H264_INVALID`].
pub const PICTURE_HEVC_INVALID: u32 = 0x0000_0001;
/// `VA_PICTURE_HEVC_LONG_TERM_REFERENCE` (`va.h:5316`): the `ReferenceFrames` entry is a long-term
/// reference of the stream (`va.h:5319`).
pub const PICTURE_HEVC_LONG_TERM_REFERENCE: u32 = 0x0000_0008;
/// `VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE` (`va.h:5325`): the entry is one of the
/// `NumPocStCurrBefore` pictures the current picture's short-term reference picture set refers to.
pub const PICTURE_HEVC_RPS_ST_CURR_BEFORE: u32 = 0x0000_0010;
/// `VA_PICTURE_HEVC_RPS_ST_CURR_AFTER` (`va.h:5331`): see
/// [`PICTURE_HEVC_RPS_ST_CURR_BEFORE`], for `NumPocStCurrAfter`.
pub const PICTURE_HEVC_RPS_ST_CURR_AFTER: u32 = 0x0000_0020;
/// `VA_PICTURE_HEVC_RPS_LT_CURR` (`va.h:5337`): see [`PICTURE_HEVC_RPS_ST_CURR_BEFORE`], for
/// `NumPocLtCurr`. The three are exclusive of one another and of
/// [`PICTURE_HEVC_LONG_TERM_REFERENCE`] (`va.h:5326`), so one entry carries one of the four.
pub const PICTURE_HEVC_RPS_LT_CURR: u32 = 0x0000_0040;

/// `VA_INVALID_SURFACE` (`va.h:1648`), which is `VA_INVALID_ID`: the picture id an invalid entry
/// carries.
pub const INVALID_SURFACE: u32 = 0xffff_ffff;

/// `VAEntrypoint` (`va.h:553`): `VAEntrypointVLD` is the hardware video decoder, the entry point
/// every decode goes through.
pub const VLD: VAEntrypoint = 1;
/// `VAEntrypointEncSlice` (`va.h:558`): the slice-level encoder. Not used for decoding; listed so
/// a driver's entry points can be reported whole.
pub const ENC_SLICE: VAEntrypoint = 6;

/// `VAProfile` (`va.h:511-537`): the decoding profiles this backend can drive. A driver profile
/// outside this list still reaches [`Profile::Other`], so nothing is silently dropped.
pub const H264_MAIN: VAProfile = 6;
/// See [`H264_MAIN`].
pub const H264_HIGH: VAProfile = 7;
/// See [`H264_MAIN`].
pub const H264_CONSTRAINED_BASELINE: VAProfile = 13;
/// See [`H264_MAIN`].
pub const HEVC_MAIN: VAProfile = 17;
/// See [`H264_MAIN`].
pub const HEVC_MAIN_10: VAProfile = 18;
/// `VAProfileVP9Profile0` (`va.h:524`): the first of the four VP9 profiles.
pub const VP9_PROFILE_0: VAProfile = 19;
/// See [`VP9_PROFILE_0`].
pub const VP9_PROFILE_1: VAProfile = 20;
/// See [`VP9_PROFILE_0`].
pub const VP9_PROFILE_2: VAProfile = 21;
/// See [`VP9_PROFILE_0`].
pub const VP9_PROFILE_3: VAProfile = 22;
/// `VAProfileAV1Profile0` (`va.h:537`).
pub const AV1_MAIN: VAProfile = 32;

/// `RTLD_NOW | RTLD_LOCAL`: resolve every symbol at open time (a missing one is then an error
/// here rather than a jump through a null pointer later) and keep the library's symbols out of the
/// global namespace. Both are the Linux/glibc values, which is all `dlopen` needs; the workspace
/// has no `libc` to declare them from (`<dlfcn.h>`).
const RTLD_NOW: c_int = 0x2;
const RTLD_LOCAL: c_int = 0x0;

/// `O_RDWR` from `<fcntl.h>`: a DRM render node is opened read/write by the DRM client API.
const O_RDWR: c_int = 0x2;

/// The libraries to open, and where the symbols live (`libva.so.2` and `libva-drm.so.2`; the dash
/// is the distribution's, and the soname the only one to load).
const LIBVA: &[u8] = b"libva.so.2\0";
const LIBVA_DRM: &[u8] = b"libva-drm.so.2\0";

/// The DRM render node to open when the environment does not name one.
const DRM_NODE: &str = "/dev/dri/renderD128";
/// Names the DRM render node to open instead of [`DRM_NODE`]. A diagnostic escape hatch for a
/// machine whose render node is numbered differently (a second GPU), documented here because it is
/// the one thing in this file that changes behaviour from outside the program.
const DRM_NODE_ENV: &str = "FILMCRAFT_VAAPI_DRM_NODE";

/// How many entries a driver-reported list may hold: no driver has anything like this many
/// profiles or entry points, and the answer is what sizes the array we hand it.
const MAX_DRIVER_ENTRIES: usize = 4096;

/// The most surfaces one decode session allocates. A decoded picture buffer holds at most 16
/// pictures (15 for HEVC) and the decoder reads each picture back before decoding the next, so this
/// is far more than any conforming stream needs; it is here so that a stream description claiming an
/// absurd decoded picture buffer size costs a clamp rather than an allocation.
pub const MAX_SURFACES: usize = 32;

/// How many times `vaSyncSurface` is asked again while it reports the video engine busy. The engine
/// resets in a fraction of a frame; a driver still answering after this many is one that has stopped
/// answering, and the decode is reported as failed so the stream goes to the software decoder.
const MAX_SYNC_ATTEMPTS: u32 = 64;

/// The longest driver string copied out of libva (a vendor name is a few dozen bytes), so a
/// driver that hands back a huge or unterminated string cannot drive a large allocation.
const MAX_DRIVER_STRING: usize = 256;

/// The size the ABI probe image is created at: big enough for every driver to have a real pitch
/// and plane offsets, small enough to be free.
const PROBE_WIDTH: u16 = 64;
const PROBE_HEIGHT: u16 = 64;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
    fn dlclose(handle: *mut c_void) -> c_int;
    /// C's `open` is variadic; only the mode argument follows the flags, and only for `O_CREAT`,
    /// which is never used here. It is declared variadic so the call sets up the register the
    /// SysV ABI requires for a variadic call.
    fn open(path: *const c_char, flags: c_int, ...) -> c_int;
    fn close(fd: c_int) -> c_int;
}

// ---------------------------------------------------------------------------------------------
// The structures. Sizes and offsets follow libva 2.24's headers; see the tests at the bottom.
// ---------------------------------------------------------------------------------------------

/// `VAImageFormat` (`va.h:4717`): what a buffer of image data is — a four-character code, the bit
/// depth, and for packed formats the component masks. 48 bytes.
#[repr(C)]
#[derive(Default)]
pub struct VaImageFormat {
    pub fourcc: u32,
    /// `VA_LSB_FIRST` or `VA_MSB_FIRST` (`va.h:4714-4715`).
    pub byte_order: u32,
    pub bits_per_pixel: u32,
    /// Significant bits per pixel.
    pub depth: u32,
    pub red_mask: u32,
    pub green_mask: u32,
    pub blue_mask: u32,
    pub alpha_mask: u32,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAImage` (`va.h:4734`): the description libva hands back for a decoded picture — its size, how
/// many planes it has, and where each plane starts and how it is pitched. Every readback path
/// copies out of this, so a field declared at the wrong offset or width reads its neighbour's
/// value instead of its own; [`Display::probe_image_layout`] is the check against the library that
/// is installed, and `va_image_layout_matches_the_header` pins the offsets against the header.
/// 120 bytes.
///
/// The fields are `pub(crate)` rather than public: they are libva's memory layout rather than
/// FilmCraft's data, but [`crate::vaapi::validate_image`] is the single gate every readback goes
/// through and has to read the plane geometry the driver wrote — which is exactly what is unsafe to
/// read from anywhere else. Nothing outside this backend's one module tree reads them.
#[repr(C)]
#[derive(Default)]
pub struct VaImage {
    pub image_id: VAImageID,
    pub format: VaImageFormat,
    /// The buffer (`VAImageBufferType`) the image data lives in, to be mapped and unmapped.
    pub buf: VABufferID,
    pub width: u16,
    pub height: u16,
    /// Bytes the driver allocated for all planes together.
    pub data_size: u32,
    /// 1..=3 for the planar formats; whatever else a driver reports is not read.
    pub num_planes: u32,
    /// Bytes per row of each plane.
    pub pitches: [u32; 3],
    /// Byte offset of each plane within the mapped buffer.
    pub offsets: [u32; 3],
    /// Palette bookkeeping; zero for the YUV formats this backend takes.
    pub num_palette_entries: i32,
    pub entry_bytes: i32,
    pub component_order: [i8; 4],
    pub va_reserved: [u32; VA_PADDING_LOW],
}

// ---------------------------------------------------------------------------------------------
// The picture- and slice-parameter records: what a decode submits per access unit. Unlike the
// image above, these are built by [`crate::vaapi::picture`] and read back by its tests, so their
// fields are `pub`; the records themselves stay in this module's hands (`Picture` carries opaque
// owned bytes) and nothing here is handed to libva until the decode loop copies it into a buffer.
//
// The C declarations carry bitfields (`seq_fields`, `pic_fields`, `slice_parsing_fields`,
// `LongSliceFlags`), which Rust has no syntax for. Each is declared here as the union's `u32`
// `value` member, and `picture.rs` packs the syntax elements into it with the bit order
// `/usr/include/va/` writes them in — the same order a C compiler would, from bit 0 up. Reading a
// packed word back is the same operation the other way, which is what the record readers below
// and `picture.rs`'s tests do; each bit position is pinned there against the parsed value.
//
// Every field keeps the C name, including the deprecated ones the headers still declare
// (`num_slice_groups_minus1`, `tg_start`, …), because a renamed field is one the next reader has
// to look up and one a declaration fix has to match against the header.
//
// Where the C layout has a hole between two fields — alignment padding a C compiler inserts — the
// Rust declaration names it `padding_*` and gives it a width, which is the same thing libva does with
// its own `va_reserved` fields. Naming every byte matters here: `#[derive(Default)]` does not
// initialise implicit padding, so a record with unnamed padding would have indeterminate bytes and
// `record_bytes` could not hand them to a driver as a `[u8]`. With the holes named, every byte of a
// record is a field, and every field is initialised. The widths are pinned with the rest of the
// layout by `the_picture_parameter_records_match_the_header`.
//
// Sizes and offsets are libva 2.24's own (`va.h:3572` and `va_dec_hevc.h:57`), pinned by
// `the_picture_parameter_records_match_the_header`.
// ---------------------------------------------------------------------------------------------

/// `VAPictureH264` (`va.h:3572`): one picture's identity — which surface it decodes into, which
/// slot it keeps in the DPB and its picture order counts. 36 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaPictureH264 {
    /// The surface the picture is decoded into (the render target).
    pub picture_id: VABufferID,
    /// `frame_num` of the slice header, which is how the DPB slot is referred to.
    pub frame_idx: u32,
    /// `VA_PICTURE_H264_*` bits ([`PICTURE_H264_SHORT_TERM_REFERENCE`] and
    /// [`PICTURE_H264_LONG_TERM_REFERENCE`]).
    pub flags: u32,
    pub top_field_order_cnt: i32,
    pub bottom_field_order_cnt: i32,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAPictureParameterBufferH264` (`va.h:3599`): the picture-level record, one per access unit,
/// submitted before its slices. 672 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaPictureParameterBufferH264 {
    /// The picture this record describes.
    pub curr_pic: VaPictureH264,
    /// The DPB, as surfaces. This backend leaves them to the decode loop, which owns the surface
    /// ring; see `crate::vaapi::picture`.
    pub reference_frames: [VaPictureH264; 16],
    pub picture_width_in_mbs_minus1: u16,
    pub picture_height_in_mbs_minus1: u16,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    /// How many entries of `reference_frames` are valid.
    pub num_ref_frames: u8,
    /// The C layout's hole at offset 619, before `seq_fields`.
    pub padding_after_num_ref_frames: [u8; 1],
    /// `seq_fields`: the sequence parameter set's flags, packed into the union's `value` member.
    pub seq_fields: u32,
    /// FMO is not supported, so these stay at what the picture parameter set says and the
    /// decode is declined if it asks for more than one slice group.
    pub num_slice_groups_minus1: u8,
    pub slice_group_map_type: u8,
    pub slice_group_change_rate_minus1: u16,
    pub pic_init_qp_minus26: i8,
    pub pic_init_qs_minus26: i8,
    pub chroma_qp_index_offset: i8,
    pub second_chroma_qp_index_offset: i8,
    /// `pic_fields`: the picture parameter set's flags plus what this slice says, packed into the
    /// union's `value` member.
    pub pic_fields: u32,
    /// `frame_num` of the first slice of the picture.
    pub frame_num: u16,
    /// The C layout's hole at offsets 638-639, before `va_reserved`.
    pub padding_after_frame_num: [u8; 2],
    pub va_reserved: [u32; VA_PADDING_MEDIUM],
}

/// `VASliceParameterBufferH264` (`va.h:3664`): one slice's record, submitted per slice. 3128
/// bytes, most of it the two reference picture lists and the prediction weights.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaSliceParameterBufferH264 {
    /// Bytes of this slice in the slice data buffer.
    pub slice_data_size: u32,
    /// Byte offset of this slice's NAL unit in the slice data buffer.
    pub slice_data_offset: u32,
    /// `VA_SLICE_DATA_FLAG_*` ([`SLICE_DATA_FLAG_ALL`] for a whole slice).
    pub slice_data_flag: u32,
    /// Bits from the NAL header to the start of `slice_data()`, emulation prevention bytes
    /// removed — which is where FilmCraft's own parser leaves the bit position.
    pub slice_data_bit_offset: u16,
    pub first_mb_in_slice: u16,
    pub slice_type: u8,
    pub direct_spatial_mv_pred_flag: u8,
    pub num_ref_idx_l0_active_minus1: u8,
    pub num_ref_idx_l1_active_minus1: u8,
    pub cabac_init_idc: u8,
    pub slice_qp_delta: i8,
    pub disable_deblocking_filter_idc: u8,
    pub slice_alpha_c0_offset_div2: i8,
    pub slice_beta_offset_div2: i8,
    /// The C layout's hole at offsets 25-27, before `RefPicList0[]`.
    pub padding_before_ref_pic_list0: [u8; 3],
    pub ref_pic_list0: [VaPictureH264; 32],
    pub ref_pic_list1: [VaPictureH264; 32],
    pub luma_log2_weight_denom: u8,
    pub chroma_log2_weight_denom: u8,
    pub luma_weight_l0_flag: u8,
    /// The C layout's hole at offset 2335, before `luma_weight_l0[]`.
    pub padding_before_luma_weight_l0: [u8; 1],
    pub luma_weight_l0: [i16; 32],
    pub luma_offset_l0: [i16; 32],
    pub chroma_weight_l0_flag: u8,
    /// The C layout's hole at offset 2465, before `chroma_weight_l0[][]`.
    pub padding_before_chroma_weight_l0: [u8; 1],
    pub chroma_weight_l0: [[i16; 2]; 32],
    pub chroma_offset_l0: [[i16; 2]; 32],
    pub luma_weight_l1_flag: u8,
    /// The C layout's hole at offset 2723, before `luma_weight_l1[]`.
    pub padding_before_luma_weight_l1: [u8; 1],
    pub luma_weight_l1: [i16; 32],
    pub luma_offset_l1: [i16; 32],
    pub chroma_weight_l1_flag: u8,
    /// The C layout's hole at offset 2853, before `chroma_weight_l1[][]`.
    pub padding_before_chroma_weight_l1: [u8; 1],
    pub chroma_weight_l1: [[i16; 2]; 32],
    pub chroma_offset_l1: [[i16; 2]; 32],
    /// The C layout's hole at offsets 3110-3111, before `va_reserved`.
    pub padding_before_va_reserved: [u8; 2],
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAIQMatrixBufferH264` (`va.h`): the quantizer and inverse-quantizer scaling lists a picture is
/// decoded with. 240 bytes — six 4x4 lists, six 8x8 lists, then the reserved tail.
///
/// Both arrays are in **raster scan order**, which is the order FilmCraft's own parser keeps its
/// `ScalingMatrices` in (`params.rs`: `zz4_to_raster`), so filling this record from the sequence
/// parameter set is a copy rather than a re-scan. A stream whose sequence parameter set says
/// `seq_scaling_matrix_present_flag == 0` gets flat lists of 16 (`ScalingMatrices::flat`), which
/// is what 7.4.2.1.1 specifies as the default and what every encoder uses when it has no
/// quantizer matrices to send.
///
/// This is a buffer a decode cannot be without: see [`IQ_MATRIX_BUFFER`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VaIqMatrixBufferH264 {
    /// `[Intra Y, Intra Cb, Intra Cr, Inter Y, Inter Cb, Inter Cr]`, raster order.
    pub scaling_list4x4: [[u8; 16]; 6],
    /// `[Intra Y, Inter Y]`, raster order — the 8x8 arrays are `[2][64]`, so the record holds the
    /// luma pair and nothing for chroma: 8x8 chroma lists only exist at `chroma_format_idc == 3`
    /// (4:4:4), which [`super::plan_nal`] declines.
    pub scaling_list8x8: [[u8; 64]; 2],
    pub va_reserved: [u32; VA_PADDING_LOW],
}

impl Default for VaIqMatrixBufferH264 {
    /// Flat lists of 16 — the quantizer every stream without custom matrices decodes with
    /// (7.4.2.1.1's default), so the empty record is the right one to hand a driver rather than
    /// zeroes, which would name a quantizer nothing in H.264 allows.
    fn default() -> Self {
        Self { scaling_list4x4: [[16; 16]; 6], scaling_list8x8: [[16; 64]; 2], va_reserved: [0; VA_PADDING_LOW] }
    }
}

/// `VAPictureHEVC` (`va.h:5292`): the HEVC counterpart of [`VaPictureH264`]. 28 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaPictureHevc {
    /// The surface the picture is decoded into.
    pub picture_id: VABufferID,
    /// Picture order count, which is how the DPB slot is referred to.
    pub pic_order_cnt: i32,
    /// `VA_PICTURE_HEVC_*` bits.
    pub flags: u32,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAPictureParameterBufferHEVC` (`va_dec_hevc.h:57`): the HEVC picture-level record. 604 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaPictureParameterBufferHevc {
    pub curr_pic: VaPictureHevc,
    /// The DPB, as surfaces; the decode loop's to fill, like `reference_frames` above.
    pub reference_frames: [VaPictureHevc; 15],
    pub pic_width_in_luma_samples: u16,
    pub pic_height_in_luma_samples: u16,
    /// `pic_fields`: the sequence and picture parameter sets' flags, packed into the union's
    /// `value` member.
    pub pic_fields: u32,
    pub sps_max_dec_pic_buffering_minus1: u8,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub pcm_sample_bit_depth_luma_minus1: u8,
    pub pcm_sample_bit_depth_chroma_minus1: u8,
    pub log2_min_luma_coding_block_size_minus3: u8,
    pub log2_diff_max_min_luma_coding_block_size: u8,
    pub log2_min_transform_block_size_minus2: u8,
    pub log2_diff_max_min_transform_block_size: u8,
    pub log2_min_pcm_luma_coding_block_size_minus3: u8,
    pub log2_diff_max_min_pcm_luma_coding_block_size: u8,
    pub max_transform_hierarchy_depth_intra: u8,
    pub max_transform_hierarchy_depth_inter: u8,
    pub init_qp_minus26: i8,
    pub diff_cu_qp_delta_depth: u8,
    pub pps_cb_qp_offset: i8,
    pub pps_cr_qp_offset: i8,
    pub log2_parallel_merge_level_minus2: u8,
    pub num_tile_columns_minus1: u8,
    pub num_tile_rows_minus1: u8,
    pub column_width_minus1: [u16; 19],
    pub row_height_minus1: [u16; 21],
    /// `slice_parsing_fields`: what a driver needs to parse this slice header, packed into the
    /// union's `value` member.
    pub slice_parsing_fields: u32,
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    pub num_short_term_ref_pic_sets: u8,
    pub num_long_term_ref_pic_sps: u8,
    pub num_ref_idx_l0_default_active_minus1: u8,
    pub num_ref_idx_l1_default_active_minus1: u8,
    pub pps_beta_offset_div2: i8,
    pub pps_tc_offset_div2: i8,
    pub num_extra_slice_header_bits: u8,
    /// Bits `short_term_ref_pic_set()` takes in this slice header, so a driver can skip it.
    pub st_rps_bits: u32,
    pub va_reserved: [u32; VA_PADDING_MEDIUM],
}

/// `VASliceParameterBufferHEVC` (`va_dec_hevc.h:358`): one slice segment's record. 264 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaSliceParameterBufferHevc {
    pub slice_data_size: u32,
    pub slice_data_offset: u32,
    /// `VA_SLICE_DATA_FLAG_*` ([`SLICE_DATA_FLAG_ALL`] for a whole segment).
    pub slice_data_flag: u32,
    /// Bytes from the NAL header to the start of `slice_segment_data()`.
    pub slice_data_byte_offset: u32,
    pub slice_segment_address: u32,
    /// Indices into `reference_frames`; `0xFF` marks an entry the picture does not use.
    pub ref_pic_list: [[u8; 15]; 2],
    /// The C layout's hole at offsets 50-51, before `LongSliceFlags`.
    pub padding_before_long_slice_flags: [u8; 2],
    /// `LongSliceFlags`: the slice header's own syntax elements, packed into the union's `value`
    /// member.
    pub long_slice_flags: u32,
    /// Index of the collocated picture, `0xFF` when temporal MVP is off.
    pub collocated_ref_idx: u8,
    pub num_ref_idx_l0_active_minus1: u8,
    pub num_ref_idx_l1_active_minus1: u8,
    pub slice_qp_delta: i8,
    pub slice_cb_qp_offset: i8,
    pub slice_cr_qp_offset: i8,
    pub slice_beta_offset_div2: i8,
    pub slice_tc_offset_div2: i8,
    pub luma_log2_weight_denom: u8,
    pub delta_chroma_log2_weight_denom: i8,
    pub delta_luma_weight_l0: [i8; 15],
    pub luma_offset_l0: [i8; 15],
    pub delta_chroma_weight_l0: [[i8; 2]; 15],
    pub chroma_offset_l0: [[i8; 2]; 15],
    pub delta_luma_weight_l1: [i8; 15],
    pub luma_offset_l1: [i8; 15],
    pub delta_chroma_weight_l1: [[i8; 2]; 15],
    pub chroma_offset_l1: [[i8; 2]; 15],
    pub five_minus_max_num_merge_cand: u8,
    /// The C layout's hole at offset 247, before `num_entry_point_offsets`.
    pub padding_before_num_entry_point_offsets: [u8; 1],
    pub num_entry_point_offsets: u16,
    pub entry_offset_to_subset_array: u16,
    /// Emulation prevention bytes in this slice header, which is what the entry point offsets and
    /// `slice_data_byte_offset` are counted without.
    pub slice_data_num_emu_prevn_bytes: u16,
    /// The C layout's hole at offsets 254-255, before `va_reserved`.
    pub padding_before_va_reserved: [u8; 2],
    pub va_reserved: [u32; VA_PADDING_LOW - 2],
}

// ---------------------------------------------------------------------------------------------
// Reading the records back out of the opaque bytes `Picture` carries. The tests of
// `picture.rs` are what reads them today, so the `cfg_attr` keeps them out of the dead-code lint
// without letting them go unchecked when the decode loop does read them.
// ---------------------------------------------------------------------------------------------

/// The record of type `T` in `bytes`, or `None` when the buffer is too short for one.
///
/// # Safety
/// `T` must be a `#[repr(C)]` record this module declares, so that any `size_of::<T>()` bytes are a
/// valid `T`. Every caller passes one of the structs below and nothing else; a short buffer is
/// refused rather than read past.
#[cfg_attr(not(test), allow(dead_code))]
unsafe fn record_of<T: Default + Copy>(bytes: &[u8]) -> Option<T> {
    if bytes.len() < size_of::<T>() {
        return None;
    }
    let mut record = T::default();
    // SAFETY: `bytes` is at least `size_of::<T>()` long (checked above) and `record` is a live,
    // initialised `T`, so this copies exactly one `T`'s worth of bytes into it.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), std::ptr::from_mut(&mut record).cast::<u8>(), size_of::<T>()) };
    Some(record)
}

/// The bytes of a record: exactly the bytes the driver reads, at the same addresses they land at
/// when a buffer is made from them.
///
/// Safe because a record is a plain `#[repr(C)]` value whose every byte is a named field: the holes
/// the C layout has between fields are declared as `padding_*` fields here (see the note above this
/// file's structures), so `Default` initialises all of them and a record of one has no indeterminate
/// byte anywhere. `record` is borrowed for the copy and nothing is read back through a pointer.
pub fn record_bytes<T: Copy>(record: &T) -> Vec<u8> {
    // SAFETY: `record` is a live `T` borrowed for the length of this function and is at least
    // `size_of::<T>()` bytes of readable memory, so the slice stays inside it and outlives nothing;
    // `T: Copy` means the value owns nothing the bytes could outlive; and every byte of a `T` is an
    // initialised field — the records' alignment holes are named `padding_*` fields, which
    // `#[derive(Default)]` zeroes — so the `&[u8]` never covers uninitialised memory.
    unsafe { std::slice::from_raw_parts(std::ptr::from_ref(record).cast::<u8>(), size_of::<T>()) }.to_vec()
}

/// The H.264 picture parameter record `bytes` carries, or `None` if it is not one long enough.
#[cfg_attr(not(test), allow(dead_code))]
pub fn picture_params_h264(bytes: &[u8]) -> Option<VaPictureParameterBufferH264> {
    // SAFETY: `VaPictureParameterBufferH264` is a `#[repr(C)]` struct declared above.
    unsafe { record_of::<VaPictureParameterBufferH264>(bytes) }
}

/// The H.264 slice parameter record `bytes` carries, or `None` if it is not one long enough.
#[cfg_attr(not(test), allow(dead_code))]
pub fn slice_params_h264(bytes: &[u8]) -> Option<VaSliceParameterBufferH264> {
    // SAFETY: `VaSliceParameterBufferH264` is a `#[repr(C)]` struct declared above.
    unsafe { record_of::<VaSliceParameterBufferH264>(bytes) }
}

/// The H.264 IQ matrix record `bytes` carries, or `None` if it is not one long enough.
#[cfg_attr(not(test), allow(dead_code))]
pub fn iq_matrix_h264(bytes: &[u8]) -> Option<VaIqMatrixBufferH264> {
    // SAFETY: `VaIqMatrixBufferH264` is a `#[repr(C)]` struct declared above.
    unsafe { record_of::<VaIqMatrixBufferH264>(bytes) }
}

/// The HEVC picture parameter record `bytes` carries, or `None` if it is not one long enough.
#[cfg_attr(not(test), allow(dead_code))]
pub fn picture_params_hevc(bytes: &[u8]) -> Option<VaPictureParameterBufferHevc> {
    // SAFETY: `VaPictureParameterBufferHevc` is a `#[repr(C)]` struct declared above.
    unsafe { record_of::<VaPictureParameterBufferHevc>(bytes) }
}

/// The HEVC slice parameter record `bytes` carries, or `None` if it is not one long enough.
#[cfg_attr(not(test), allow(dead_code))]
pub fn slice_params_hevc(bytes: &[u8]) -> Option<VaSliceParameterBufferHevc> {
    // SAFETY: `VaSliceParameterBufferHevc` is a `#[repr(C)]` struct declared above.
    unsafe { record_of::<VaSliceParameterBufferHevc>(bytes) }
}

// ---------------------------------------------------------------------------------------------
// What the driver reports, as types the rest of the backend can compare.
// ---------------------------------------------------------------------------------------------

/// A codec profile the driver's decoder supports, as something FilmCraft can name and compare. A
/// profile this backend does not decode is kept as [`Profile::Other`] rather than dropped, so a
/// driver's capability list is reported in full.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    H264ConstrainedBaseline,
    H264Main,
    H264High,
    HevcMain,
    HevcMain10,
    /// VP9 profile 0..=3 (`VAProfileVP9Profile0`..`3`), which differ in bit depth and colour space
    /// handling, so the number is kept.
    Vp9(u8),
    Av1Main,
    /// A profile libva or the driver knows and this backend does not name.
    Other(i32),
}

impl Profile {
    /// The profile as the `VAProfile` libva takes, which is what `entrypoints` asks about and what
    /// a config is created with.
    pub fn bits(self) -> VAProfile {
        match self {
            Self::H264ConstrainedBaseline => H264_CONSTRAINED_BASELINE,
            Self::H264Main => H264_MAIN,
            Self::H264High => H264_HIGH,
            Self::HevcMain => HEVC_MAIN,
            Self::HevcMain10 => HEVC_MAIN_10,
            // Beyond 3 libva numbers no VP9 profile, so the number stays a number rather
            // than silently naming a different profile's bits.
            Self::Vp9(n) => VP9_PROFILE_0 + i32::from(n),
            Self::Av1Main => AV1_MAIN,
            Self::Other(bits) => bits,
        }
    }

    /// The profile libva reported.
    fn from_bits(bits: VAProfile) -> Self {
        match bits {
            H264_CONSTRAINED_BASELINE => Self::H264ConstrainedBaseline,
            H264_MAIN => Self::H264Main,
            H264_HIGH => Self::H264High,
            HEVC_MAIN => Self::HevcMain,
            HEVC_MAIN_10 => Self::HevcMain10,
            VP9_PROFILE_0 => Self::Vp9(0),
            VP9_PROFILE_1 => Self::Vp9(1),
            VP9_PROFILE_2 => Self::Vp9(2),
            VP9_PROFILE_3 => Self::Vp9(3),
            AV1_MAIN => Self::Av1Main,
            other => Self::Other(other),
        }
    }
}

/// An entry point the driver offers for a profile. Only [`Entrypoint::Vld`] decodes; the others are
/// reported so a driver's entry points are not filtered on the way in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entrypoint {
    /// `VAEntrypointVLD`: the hardware video decoder.
    Vld,
    /// `VAEntrypointEncSlice`: the slice-level encoder. Listed, not used for decoding.
    EncSlice,
    /// An entry point this backend does not name.
    Other(i32),
}

impl Entrypoint {
    /// The entry point as the `VAEntrypoint` libva takes.
    pub fn bits(self) -> VAEntrypoint {
        match self {
            Self::Vld => VLD,
            Self::EncSlice => ENC_SLICE,
            Self::Other(bits) => bits,
        }
    }

    /// The entry point libva reported.
    fn from_bits(bits: VAEntrypoint) -> Self {
        match bits {
            VLD => Self::Vld,
            ENC_SLICE => Self::EncSlice,
            other => Self::Other(other),
        }
    }
}

/// A render format (`VA_RT_FORMAT_*`) a surface can be created in. This backend takes 8- and
/// 10-bit 4:2:0; anything else is [`RtFormat::Other`], which the decoder declines.
#[allow(non_camel_case_types)] // `Yuv420_10` is libva's `VA_RT_FORMAT_YUV420_10`, not a typo.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RtFormat {
    Nv12,
    Yuv420_10,
    /// A format this backend does not take.
    Other(u32),
}

impl RtFormat {
    /// The format as the `VA_RT_FORMAT_*` libva takes, which is what a surface is created in.
    pub fn bits(self) -> u32 {
        match self {
            Self::Nv12 => YUV420,
            Self::Yuv420_10 => YUV420_10,
            Self::Other(bits) => bits,
        }
    }
}

impl From<u32> for RtFormat {
    fn from(bits: u32) -> Self {
        match bits {
            YUV420 => Self::Nv12,
            YUV420_10 => Self::Yuv420_10,
            other => Self::Other(other),
        }
    }
}

/// libva's own words for a status code, or the number when libva is not loaded or does not know the
/// code: error messages name the status, and this is where the text comes from.
///
/// This opens libva if it is not open yet. Code that already has an [`Api`] in hand — everything on
/// a [`Display`] error path — uses [`Api::status_text`] instead, so that decorating an error can
/// never be the thing that loads a library (and so can never re-enter the one-time load).
pub fn status(s: i32) -> String {
    match load() {
        Ok(api) => api.status_text(s),
        Err(_) => format!("VA-API status {s}"),
    }
}

// ---------------------------------------------------------------------------------------------
// The two open libraries and the entry points in them.
// ---------------------------------------------------------------------------------------------

/// The two `dlopen` handles, shared by every [`Api`] that was made from them.
struct Libraries {
    va: *mut c_void,
    va_drm: *mut c_void,
}

impl Drop for Libraries {
    fn drop(&mut self) {
        // SAFETY: both handles came from `dlopen` in `load_uncached` and this is their only owner,
        // so each is closed exactly once. Every function pointer resolved out of them belongs to an
        // `Api`, which holds a reference to this `Libraries` and is therefore dropped before it:
        // nothing can jump into a closed library.
        unsafe {
            dlclose(self.va_drm);
            dlclose(self.va);
        }
    }
}

// SAFETY: two `dlopen` handles and nothing else. They are opened before any `Api` exists and
// closed only in `Drop`, on whichever thread happens to drop the last reference, which the loader
// allows.
unsafe impl Send for Libraries {}
// SAFETY: see above.
unsafe impl Sync for Libraries {}

/// The libva entry points this backend needs, resolved once and kept for as long as the process
/// decodes with VA-API.
#[derive(Clone)]
pub struct Api {
    /// Read only to keep the libraries open: this is what outlives every function pointer below.
    _libraries: Arc<Libraries>,
    initialize: InitializeFn,
    terminate: TerminateFn,
    get_display_drm: GetDisplayDrmFn,
    query_vendor_string: QueryVendorStringFn,
    max_profiles: MaxProfilesFn,
    query_profiles: QueryProfilesFn,
    max_entrypoints: MaxEntrypointsFn,
    query_entrypoints: QueryEntrypointsFn,
    /// Optional: `vaErrorStr` is libva 1.0-era and only decorates error messages.
    error_str: Option<ErrorStrFn>,
    /// Optional: the image calls are the ABI layout probe and the indirect readback path (see
    /// [`Session::read`]). Both are needed together, so a libva with neither is still a working
    /// decoder on any driver whose surfaces can be derived.
    create_image: Option<CreateImageFn>,
    destroy_image: Option<DestroyImageFn>,
    put_image: Option<PutImageFn>,
    create_config: CreateConfigFn,
    destroy_config: DestroyConfigFn,
    create_surfaces: CreateSurfacesFn,
    destroy_surfaces: DestroySurfacesFn,
    create_context: CreateContextFn,
    destroy_context: DestroyContextFn,
    create_buffer: CreateBufferFn,
    destroy_buffer: DestroyBufferFn,
    derive_image: DeriveImageFn,
    begin_picture: BeginPictureFn,
    render_picture: RenderPictureFn,
    end_picture: EndPictureFn,
    sync_surface: SyncSurfaceFn,
    query_surface_status: QuerySurfaceStatusFn,
    map_buffer: MapBufferFn,
    unmap_buffer: UnmapBufferFn,
}

// SAFETY: the function pointers are resolved once and are not mutated afterwards, and the
// libraries they point into stay open for as long as any `Api` exists (see `Libraries`).
unsafe impl Send for Api {}
// SAFETY: see above.
unsafe impl Sync for Api {}

/// The libva library and its DRM helper, opened and resolved once per process; a machine without
/// them gets the error once and never again.
static API: OnceLock<Result<Arc<Api>, String>> = OnceLock::new();

/// The libva entry points, opening `libva.so.2` and `libva-drm.so.2` on first use.
///
/// The `Arc` is kept for as long as the process uses VA-API, which is what keeps the libraries
/// open: [`Display`] takes a reference, so a function pointer can never outlive its library. A
/// failure is remembered rather than retried (probing a missing library on every clip would open
/// and close it thousands of times).
pub fn load() -> Result<Arc<Api>, String> {
    API.get_or_init(|| load_uncached().map(Arc::new)).clone()
}

fn load_uncached() -> Result<Api, String> {
    // SAFETY: opens the named library; a null handle is the documented failure and is reported.
    let va = unsafe { dlopen(LIBVA.as_ptr().cast(), RTLD_NOW | RTLD_LOCAL) };
    if va.is_null() {
        return Err(format!("libva.so.2 is not installed ({})", dl_error()));
    }
    // SAFETY: as above, for the DRM helper.
    let va_drm = unsafe { dlopen(LIBVA_DRM.as_ptr().cast(), RTLD_NOW | RTLD_LOCAL) };
    if va_drm.is_null() {
        // SAFETY: `va` was just opened here and is closed exactly once, on this error path.
        unsafe { dlclose(va) };
        return Err(format!("libva-drm.so.2 is not installed ({})", dl_error()));
    }
    let libraries = Arc::new(Libraries { va, va_drm });
    match resolve(va, va_drm, libraries) {
        Ok(api) => Ok(api),
        Err(why) => {
            log::info!("VA-API: {why}");
            Err(why)
        }
    }
}

/// Every symbol this backend calls, with the signature its header declares. The entry points the
/// decoder cannot work without are required; the rest are optional, and a missing optional symbol
/// is a capability this build simply does not have.
fn resolve(va: *mut c_void, va_drm: *mut c_void, libraries: Arc<Libraries>) -> Result<Api, String> {
    // SAFETY: every symbol below is looked up in a handle `load_uncached` just opened, with the
    // signature `/usr/include/va/va.h` (or `va_drm.h`) declares for it. All of them are part of
    // libva 2.x, so a missing one means the library is not what it claims to be.
    unsafe {
        Ok(Api {
            _libraries: libraries,
            initialize: symbol(va, c"vaInitialize")?,
            terminate: symbol(va, c"vaTerminate")?,
            get_display_drm: symbol(va_drm, c"vaGetDisplayDRM")?,
            query_vendor_string: symbol(va, c"vaQueryVendorString")?,
            max_profiles: symbol(va, c"vaMaxNumProfiles")?,
            query_profiles: symbol(va, c"vaQueryConfigProfiles")?,
            max_entrypoints: symbol(va, c"vaMaxNumEntrypoints")?,
            query_entrypoints: symbol(va, c"vaQueryConfigEntrypoints")?,
            error_str: optional_symbol(va, c"vaErrorStr"),
            create_image: optional_symbol(va, c"vaCreateImage"),
            destroy_image: optional_symbol(va, c"vaDestroyImage"),
            put_image: optional_symbol(va, c"vaPutImage"),
            create_config: symbol(va, c"vaCreateConfig")?,
            destroy_config: symbol(va, c"vaDestroyConfig")?,
            create_surfaces: symbol(va, c"vaCreateSurfaces")?,
            destroy_surfaces: symbol(va, c"vaDestroySurfaces")?,
            create_context: symbol(va, c"vaCreateContext")?,
            destroy_context: symbol(va, c"vaDestroyContext")?,
            create_buffer: symbol(va, c"vaCreateBuffer")?,
            destroy_buffer: symbol(va, c"vaDestroyBuffer")?,
            derive_image: symbol(va, c"vaDeriveImage")?,
            begin_picture: symbol(va, c"vaBeginPicture")?,
            render_picture: symbol(va, c"vaRenderPicture")?,
            end_picture: symbol(va, c"vaEndPicture")?,
            sync_surface: symbol(va, c"vaSyncSurface")?,
            query_surface_status: symbol(va, c"vaQuerySurfaceStatus")?,
            map_buffer: symbol(va, c"vaMapBuffer")?,
            unmap_buffer: symbol(va, c"vaUnmapBuffer")?,
        })
    }
}

type InitializeFn = unsafe extern "C" fn(VADisplay, *mut c_int, *mut c_int) -> VAStatus;
type TerminateFn = unsafe extern "C" fn(VADisplay) -> VAStatus;
type GetDisplayDrmFn = unsafe extern "C" fn(c_int) -> VADisplay;
type QueryVendorStringFn = unsafe extern "C" fn(VADisplay) -> *const c_char;
type MaxProfilesFn = unsafe extern "C" fn(VADisplay) -> VAStatus;
type MaxEntrypointsFn = unsafe extern "C" fn(VADisplay) -> VAStatus;
type QueryProfilesFn = unsafe extern "C" fn(VADisplay, *mut VAProfile, *mut c_int) -> VAStatus;
type QueryEntrypointsFn = unsafe extern "C" fn(VADisplay, VAProfile, *mut VAEntrypoint, *mut c_int) -> VAStatus;
type ErrorStrFn = unsafe extern "C" fn(VAStatus) -> *const c_char;
type CreateImageFn = unsafe extern "C" fn(VADisplay, *mut VaImageFormat, c_int, c_int, *mut VaImage) -> VAStatus;
type DestroyImageFn = unsafe extern "C" fn(VADisplay, VAImageID) -> VAStatus;
type PutImageFn = unsafe extern "C" fn(VADisplay, VASurfaceID, VAImageID, c_int, c_int, c_uint, c_uint, c_int, c_int, c_uint, c_uint) -> VAStatus;
type CreateConfigFn = unsafe extern "C" fn(VADisplay, VAProfile, VAEntrypoint, *mut VaConfigAttrib, c_int, *mut VAConfigID) -> VAStatus;
type DestroyConfigFn = unsafe extern "C" fn(VADisplay, VAConfigID) -> VAStatus;
type CreateSurfacesFn = unsafe extern "C" fn(VADisplay, c_uint, c_uint, c_uint, *mut VASurfaceID, c_uint, *mut VaSurfaceAttrib, c_uint) -> VAStatus;
type DestroySurfacesFn = unsafe extern "C" fn(VADisplay, *mut VASurfaceID, c_int) -> VAStatus;
type CreateContextFn = unsafe extern "C" fn(VADisplay, VAConfigID, c_int, c_int, c_int, *mut VASurfaceID, c_int, *mut VAContextID) -> VAStatus;
type DestroyContextFn = unsafe extern "C" fn(VADisplay, VAContextID) -> VAStatus;
type CreateBufferFn = unsafe extern "C" fn(VADisplay, VAContextID, VAStatus, c_uint, c_uint, *mut c_void, *mut VABufferID) -> VAStatus;
type DestroyBufferFn = unsafe extern "C" fn(VADisplay, VABufferID) -> VAStatus;
type DeriveImageFn = unsafe extern "C" fn(VADisplay, VASurfaceID, *mut VaImage) -> VAStatus;
type BeginPictureFn = unsafe extern "C" fn(VADisplay, VAContextID, VASurfaceID) -> VAStatus;
type RenderPictureFn = unsafe extern "C" fn(VADisplay, VAContextID, *mut VABufferID, c_int) -> VAStatus;
type EndPictureFn = unsafe extern "C" fn(VADisplay, VAContextID) -> VAStatus;
type SyncSurfaceFn = unsafe extern "C" fn(VADisplay, VASurfaceID) -> VAStatus;
type QuerySurfaceStatusFn = unsafe extern "C" fn(VADisplay, VASurfaceID, *mut c_uint) -> VAStatus;
type MapBufferFn = unsafe extern "C" fn(VADisplay, VABufferID, *mut *mut c_void) -> VAStatus;
type UnmapBufferFn = unsafe extern "C" fn(VADisplay, VABufferID) -> VAStatus;

/// The exported symbol `name` of the open library `handle` as a function pointer of type `F`.
///
/// # Safety
/// `F` must be the `extern "C" fn` type `name` really has in that library.
unsafe fn symbol<F: Copy>(handle: *mut c_void, name: &CStr) -> Result<F, String> {
    if size_of::<F>() != size_of::<usize>() {
        return Err("not a function pointer type".into());
    }
    // SAFETY: `handle` is an open `dlopen` handle and `name` a NUL-terminated symbol name.
    let p = unsafe { dlsym(handle, name.as_ptr()) };
    if p.is_null() {
        return Err(format!("libva does not export {}", name.to_string_lossy()));
    }
    // SAFETY: `F` is pointer-sized (checked above) and, by this function's contract, the type of
    // the function `p` points to.
    Ok(unsafe { std::mem::transmute_copy::<_, F>(&p) })
}

/// [`symbol`], for a symbol a driver may legitimately not have.
unsafe fn optional_symbol<F: Copy>(handle: *mut c_void, name: &CStr) -> Option<F> {
    // SAFETY: the caller of this function upholds the contract of `symbol` for the same `F`.
    unsafe { symbol(handle, name) }.ok()
}

/// What the last `dl*` call that failed said, for an error message.
fn dl_error() -> String {
    // SAFETY: `dlerror` returns either NULL or a NUL-terminated string owned by the loader that is
    // valid until the next `dlerror` call, and copies it out before making another.
    let p = unsafe { dlerror() };
    if p.is_null() { "unknown error".to_string() } else { string_from(p) }
}

/// A NUL-terminated C string as owned Rust text, at most [`MAX_DRIVER_STRING`] bytes: a driver
/// string that is not terminated would make the length unbounded, and one that is enormous has
/// nothing useful in it.
fn string_from(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: libva documents the return of `vaQueryVendorString` and `vaErrorStr` as a
    // NUL-terminated string (`va.h:403,486`), owned by the driver, and only reads from it here.
    let bytes = unsafe { CStr::from_ptr(p) }.to_bytes();
    // Lossy, so a truncated or non-UTF-8 driver string is still a valid `str` and never panics.
    String::from_utf8_lossy(bytes.get(..MAX_DRIVER_STRING).unwrap_or(bytes)).into_owned()
}

impl Api {
    /// libva's words for a status code, or the number when it has none for it.
    ///
    /// This is the [`status`] work with the library already in hand, which is what a caller that
    /// has an `Api` should use: it cannot load anything, so it cannot re-enter the one-time load.
    pub(crate) fn status_text(&self, s: i32) -> String {
        let Some(error_str) = self.error_str else { return format!("VA-API status {s}") };
        // SAFETY: `error_str` is `vaErrorStr` (`va.h:403`), which takes a status and returns a
        // NUL-terminated string (or null for a code it does not know) and does nothing else.
        let text = string_from(unsafe { error_str(s) });
        if text.is_empty() { format!("VA-API status {s}") } else { text }
    }
}

// ---------------------------------------------------------------------------------------------
// The display: an initialised VA-API connection to the GPU's video engine.
// ---------------------------------------------------------------------------------------------

/// An initialised VA-API display — what this backend decodes with, and the only place a libva
/// handle exists. Dropping it terminates the connection and closes the DRM node.
pub struct Display {
    api: Arc<Api>,
    /// The DRM render node, closed in `Drop` (`None` once it is).
    drm: Option<c_int>,
    dpy: VADisplay,
    version: (c_int, c_int),
    /// The driver's self-description, copied out of libva once when the display is opened.
    vendor: Option<&'static str>,
    /// Serialises our use of the display: libva does not document its entry points as safe to call
    /// concurrently on one display, and a decode drives them in sequence anyway.
    gate: std::sync::Mutex<()>,
}

// SAFETY: a `Display` is a libva handle, the DRM descriptor libva keeps using behind it, and the
// version and vendor string we read at construction. Sending one to the decoding thread is the
// point; concurrent calls into libva are serialised by `gate`, and `Drop` (which runs `vaTerminate`
// and closes the descriptor) can only run once the last `Arc` is gone, so no other reference is
// still in use.
unsafe impl Send for Display {}
// SAFETY: see above.
unsafe impl Sync for Display {}

impl Display {
    /// Open the DRM render node and initialise libva on it.
    ///
    /// `vaInitialize` is mandatory before anything else is asked of a display, so it happens here
    /// and its status is checked: a driver that refuses to initialise leaves FilmCraft with its own
    /// decoders rather than a half-open display. Every failure says which step it was.
    pub fn new(api: &Api) -> Result<Arc<Display>, String> {
        let fd = open_drm_node()?;
        // SAFETY: `fd` is an open DRM descriptor, and `get_display_drm` is `vaGetDisplayDRM`
        // (`va_drm.h:52`), which returns the display derived from the descriptor, or null on
        // failure. libva keeps using that descriptor for the life of the display, so ours stays
        // open until after `vaTerminate` in `Drop` — it is closed on every path that does not reach
        // a `Display`.
        let dpy = unsafe { (api.get_display_drm)(fd) };
        if dpy.is_null() {
            close_fd(fd);
            return Err("vaGetDisplayDRM returned no display".to_string());
        }
        let (mut major, mut minor) = (0 as c_int, 0 as c_int);
        // SAFETY: `dpy` came from `vaGetDisplayDRM` above and both out-pointers are live locals.
        // No other libva entry point has been called on `dpy`, which is what this call requires.
        let st = unsafe { (api.initialize)(dpy, &mut major, &mut minor) };
        if st != OK {
            // Nothing to terminate: the display was never initialised.
            close_fd(fd);
            let why = format!("vaInitialize failed: {}", api.status_text(st));
            log::info!("VA-API: {why}");
            return Err(why);
        }
        // libva returns the driver's own string, which lives as long as the library; it is copied
        // out here and leaked, so `vendor` can name it for as long as the display exists. A driver
        // with no vendor string returns null, which is `None` rather than an empty name. The leak
        // is a few dozen bytes once per display (there is one display per process).
        // SAFETY: `query_vendor_string` is `vaQueryVendorString` (`va.h:486`), which returns a
        // NUL-terminated string owned by the driver or null, and only reads from it here.
        let vendor = string_from(unsafe { (api.query_vendor_string)(dpy) });
        Ok(Arc::new(Display {
            // The display keeps its own handle on the libraries, so a function pointer below can
            // never outlive them.
            api: Arc::new(api.clone()),
            drm: Some(fd),
            dpy,
            version: (major, minor),
            vendor: (!vendor.is_empty()).then(|| vendor.leak() as &'static str),
            gate: std::sync::Mutex::new(()),
        }))
    }

    /// Our use of the display, serialised. A poisoned lock is not a failure here — the lock
    /// guards no state libva could have torn — so it is recovered rather than propagated.
    fn gate(&self) -> MutexGuard<'_, ()> {
        self.gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The libva version this driver implements, as `(major, minor)` from `vaInitialize`.
    pub fn va_version(&self) -> (i32, i32) {
        self.version
    }

    /// The driver's own description of itself ("Intel iHD"), or `None` when it has none.
    pub fn vendor(&self) -> Option<&'static str> {
        self.vendor
    }

    /// Every profile the driver's decoder can be configured for. Unknown profiles are reported as
    /// [`Profile::Other`] rather than dropped.
    pub fn profiles(&self) -> Result<Vec<Profile>, String> {
        let _gate = self.gate();
        // SAFETY: the display is initialised; the out-pointer refers to the array just allocated,
        // which is `capacity()` entries long as libva's own `vaMaxNumProfiles` asked.
        let capacity = capacity(unsafe { (self.api.max_profiles)(self.dpy) });
        let mut profiles = vec![0 as VAProfile; capacity];
        let mut reported: c_int = 0;
        // SAFETY: as above, plus the count out-pointer, which is a live local.
        let st = unsafe { (self.api.query_profiles)(self.dpy, profiles.as_mut_ptr(), &mut reported) };
        if st != OK {
            return Err(format!("vaQueryConfigProfiles failed: {}", self.api.status_text(st)));
        }
        // The count is the driver's to write, so it is bounded by what was allocated before it is
        // used as a length.
        let count = usize::try_from(reported).unwrap_or(0).min(profiles.len());
        Ok(profiles.into_iter().take(count).map(Profile::from_bits).collect())
    }

    /// Every entry point the driver offers for `p` — [`Entrypoint::Vld`] is the one a decode uses.
    pub fn entrypoints(&self, p: Profile) -> Result<Vec<Entrypoint>, String> {
        let _gate = self.gate();
        // SAFETY: the display is initialised, and the array and count out-pointers below refer to
        // live memory `capacity()` long as libva's own `vaMaxNumEntrypoints` asked.
        let capacity = capacity(unsafe { (self.api.max_entrypoints)(self.dpy) });
        let mut entrypoints = vec![0 as VAEntrypoint; capacity];
        let mut reported: c_int = 0;
        // SAFETY: as above, plus the profile, which is a plain integer libva compares.
        let st = unsafe { (self.api.query_entrypoints)(self.dpy, p.bits(), entrypoints.as_mut_ptr(), &mut reported) };
        if st != OK {
            return Err(format!("vaQueryConfigEntrypoints failed: {}", self.api.status_text(st)));
        }
        let count = usize::try_from(reported).unwrap_or(0).min(entrypoints.len());
        Ok(entrypoints.into_iter().take(count).map(Entrypoint::from_bits).collect())
    }

    /// Whether the installed driver fills a `VAImage` the way this module declares it: libva has
    /// no `sizeof` / `offsetof` to ask, so the check is behavioural. The driver is asked to create
    /// a real NV12 image, and the plane geometry it writes back is read out of *this* struct: if a
    /// field were declared at the wrong offset, it would read a neighbour's value and the plane
    /// arithmetic would stop adding up. `false` means the probe could not be run (a driver that
    /// will not create an NV12 image, or a libva without `vaCreateImage`).
    pub fn probe_image_layout(&self) -> Result<bool, String> {
        let (Some(create_image), Some(destroy_image)) = (self.api.create_image, self.api.destroy_image) else {
            return Ok(false);
        };
        let _gate = self.gate();
        let (width, height) = (i32::from(PROBE_WIDTH), i32::from(PROBE_HEIGHT));
        let mut format = VaImageFormat { fourcc: FOURCC_NV12, byte_order: BYTE_ORDER_LSB_FIRST, bits_per_pixel: 12, ..Default::default() };
        let mut image = VaImage::default();
        // SAFETY: the display is initialised, `format` and `image` are live locals, and the sizes
        // are the ones libva is asked to allocate for.
        let st = unsafe { create_image(self.dpy, &mut format, width, height, &mut image) };
        if st != OK {
            return Err(format!("vaCreateImage failed for the ABI probe: {}", self.api.status_text(st)));
        }
        // SAFETY: `image` is the image libva just created, and this is its matching destroy; the
        // id is the one the driver wrote above.
        unsafe { destroy_image(self.dpy, image.image_id) };
        Ok(plane_geometry_is_consistent(&image, usize::from(PROBE_WIDTH), usize::from(PROBE_HEIGHT)))
    }
}

impl Drop for Display {
    fn drop(&mut self) {
        // SAFETY: `dpy` came from `vaGetDisplayDRM` and `vaInitialize` succeeded on it, so this is
        // the matching `vaTerminate`, before the descriptor it was opened from goes away.
        unsafe { (self.api.terminate)(self.dpy) };
        if let Some(fd) = self.drm.take() {
            close_fd(fd);
        }
    }
}

/// Whether a driver-filled `VAImage` describes an NV12 image of `width` x `height` in the planes it
/// claims to have. Every number in it is the driver's, so nothing is indexed or added without a
/// bound: an image whose planes do not add up to its `data_size` is one whose fields were read from
/// somewhere other than where libva wrote them.
fn plane_geometry_is_consistent(image: &VaImage, width: usize, height: usize) -> bool {
    if usize::from(image.width) != width || usize::from(image.height) != height {
        return false;
    }
    let reported = image.num_planes;
    if !(1..=3).contains(&reported) {
        return false;
    }
    let planes = usize::try_from(reported).unwrap_or(0);
    (0..planes).all(|plane| {
        let (Some(&pitch), Some(&offset)) = (image.pitches.get(plane), image.offsets.get(plane)) else { return false };
        let Some(row_bytes) = (pitch as usize).checked_mul(height) else { return false };
        // A plane is at least a row per pixel row, and ends inside the buffer the driver sized.
        pitch as usize >= width && (offset as usize).checked_add(row_bytes).is_some_and(|end| end <= image.data_size as usize)
    })
}

/// How many entries to allocate for a driver query, from the `vaMaxNum*` answer: bounded, because
/// a driver's answer sizes the array we hand it, and never zero, so there is always somewhere for
/// the driver to write.
fn capacity(reported: VAStatus) -> usize {
    usize::try_from(reported).unwrap_or(MAX_DRIVER_ENTRIES).clamp(1, MAX_DRIVER_ENTRIES)
}

/// The DRM render node to open: `FILMCRAFT_VAAPI_DRM_NODE` when it is set (see
/// [`DRM_NODE_ENV`]), else [`DRM_NODE`].
fn drm_node() -> String {
    match std::env::var(DRM_NODE_ENV) {
        Ok(path) if !path.is_empty() => path,
        _ => DRM_NODE.to_string(),
    }
}

/// Open the DRM render node. A node that is not there, or cannot be opened, is a normal, silent
/// "no hardware here" and not an error worth logging.
fn open_drm_node() -> Result<c_int, String> {
    let path = drm_node();
    let Ok(c_path) = CString::new(path.as_str()) else { return Err(format!("{DRM_NODE_ENV} is not a usable path: {path:?}")) };
    // SAFETY: `c_path` is a NUL-terminated path that outlives the call, and `open` copies it.
    let fd = unsafe { open(c_path.as_ptr(), O_RDWR) };
    if fd < 0 {
        log::debug!("VA-API: no DRM render node at {path}");
        return Err(format!("no DRM render node at {path}"));
    }
    Ok(fd)
}

/// Close a descriptor from [`open_drm_node`], ignoring the result: the caller is already reporting
/// a failure, and a failed `close` must not replace it with a different one.
fn close_fd(fd: c_int) {
    // SAFETY: `fd` is an open descriptor this module opened, and this closes it exactly once.
    unsafe { close(fd) };
}

// ---------------------------------------------------------------------------------------------
// The decode session: a config, a ring of surfaces, a decode context, and the picture readback
// that gets pixels back out of a surface. Everything that needs a libva handle lives behind these
// six types, so the decoder itself never names one and every pairing that has to happen on every
// path — buffer and image destroyed, buffer unmapped — is a `Drop`, not a line someone has to
// remember on the error return.
// ---------------------------------------------------------------------------------------------

/// A configured decode session: what [`crate::vaapi::VaDecoder`] decodes one stream with.
///
/// It owns the three libva objects a decoder needs, in the order libva requires them to be torn
/// down: the config, the surfaces (which cannot be destroyed while a context uses them) and the
/// context. Dropping it releases all three, so a decode that fails half-way cannot leave a context
/// or a surface behind.
///
/// `vaSyncSurface` is mandatory before a surface's pixels are read, and [`Session::sync`] is it:
/// rendering is asynchronous, and a readback that does not sync reads whatever the driver had
/// finished writing — which is a plausible-looking black picture rather than an error.
pub struct Session {
    display: Arc<Display>,
    config: VAConfigID,
    surfaces: Vec<u32>,
    /// `None` after [`Session::clear_context`], which is what a seek does: no reference picture
    /// survives it, so the next picture makes a fresh context.
    context: Option<VAContextID>,
    /// The four-character code and bit depth a decoded picture is read back as.
    fourcc: u32,
    bpp: u32,
    size: (u32, u32),
    /// Whether this session's surfaces are always read back the indirect way, remembered so that a
    /// driver that cannot derive reports it once rather than on every picture.
    indirect: bool,
}

impl Session {
    /// Configure `display` for `profile` at `entrypoint`, create `count` surfaces of `format` at
    /// `size`, and make a decode context for them.
    ///
    /// Every failure names the step and the driver's own status text, because "hardware decoding
    /// declined" is only actionable if it says which of the three steps the driver refused. `size`
    /// is the *coded* picture size (the surfaces hold the padding too) and `count` is the number of
    /// surfaces the stream's decoded picture buffer needs; both come from the plan and the stream
    /// description, which bound them.
    pub fn new(display: &Arc<Display>, profile: Profile, entrypoint: Entrypoint, format: RtFormat, size: (u32, u32), count: usize) -> Result<Self, String> {
        let count = count.clamp(1, MAX_SURFACES);
        let count_u = c_uint::try_from(count).map_err(|_| format!("{count} surfaces"))?;
        let width = c_uint::try_from(size.0).map_err(|_| format!("surface width {}", size.0))?;
        let height = c_uint::try_from(size.1).map_err(|_| format!("surface height {}", size.1))?;
        let context_width = c_int::try_from(size.0).map_err(|_| format!("picture width {}", size.0))?;
        let context_height = c_int::try_from(size.1).map_err(|_| format!("picture height {}", size.1))?;
        let (fourcc, bpp) = match format {
            RtFormat::Nv12 => (NV12_FOURCC, NV12_BPP),
            RtFormat::Yuv420_10 => (P010_FOURCC, P010_BPP),
            RtFormat::Other(bits) => return Err(format!("render format {bits:#x} is not 4:2:0 8- or 10-bit")),
        };
        let api = Arc::clone(&display.api);
        let mut session = Self { display: Arc::clone(display), config: 0, surfaces: Vec::new(), context: None, fourcc, bpp, size, indirect: false };

        let mut surfaces = vec![0 as VASurfaceID; count];
        {
            let _gate = display.gate();
            let mut config: VAConfigID = 0;
            // SAFETY: the display is initialised; no attributes are passed (null, zero), which
            // libva reads as "every attribute at its default"; and `config` is a live local the
            // call writes the config id into.
            let st = unsafe { (api.create_config)(display.dpy, profile.bits(), entrypoint.bits(), std::ptr::null_mut(), 0, &mut config) };
            if st != OK {
                return Err(format!("vaCreateConfig failed for {:?}/{:?}: {}", profile, entrypoint, api.status_text(st)));
            }
            session.config = config;
            // SAFETY: as above for the surfaces: `surfaces` is a live array `count_u` long, which is
            // what the driver is told to fill, and no attribute list is passed. A `Session` that is
            // dropped from here on destroys the config it holds, so this needs no manual unwind.
            let st = unsafe { (api.create_surfaces)(display.dpy, format.bits(), width, height, surfaces.as_mut_ptr(), count_u, std::ptr::null_mut(), 0) };
            if st != OK {
                return Err(format!("vaCreateSurfaces failed for {} surfaces at {}x{}: {}", count, size.0, size.1, api.status_text(st)));
            }
            session.surfaces = std::mem::take(&mut surfaces);
            let Ok(contexts) = c_int::try_from(session.surfaces.len()) else {
                return Err(format!("{count} surfaces does not fit a context's render target list"));
            };
            let mut context: VAContextID = 0;
            // SAFETY: `config` is the config created above, `surfaces` the array just filled and
            // `contexts` its length, and `context` is a live local for the id to be written into.
            let st = unsafe {
                (api.create_context)(display.dpy, config, context_width, context_height, PROGRESSIVE, session.surfaces.as_mut_ptr(), contexts, &mut context)
            };
            if st != OK {
                return Err(format!("vaCreateContext failed at {}x{}: {}", size.0, size.1, api.status_text(st)));
            }
            session.context = Some(context);
        }
        Ok(session)
    }

    /// The surfaces pictures are decoded into, in the order the driver gave them.
    pub fn surfaces(&self) -> &[u32] {
        &self.surfaces
    }

    /// The coded picture size the surfaces and the context were created at.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Whether this session's pictures have to be read back the indirect way.
    pub fn reads_indirectly(&self) -> bool {
        self.indirect
    }

    /// Drop the decode context, so that the next picture makes a new one. A seek does this: no
    /// reference picture survives it, and a context that still held the previous run's references
    /// would decode the new run against surfaces that hold something else.
    pub fn clear_context(&mut self) {
        if let Some(context) = self.context.take() {
            let api = &self.display.api;
            // SAFETY: `context` is this session's own context, taken out of the option so no other
            // reference to it exists, and this is its matching destroy.
            unsafe { (api.destroy_context)(self.display.dpy, context) };
        }
    }

    /// Make a decode context again, for the surfaces this session already owns.
    fn make_context(&mut self) -> Result<VAContextID, String> {
        let api = &self.display.api;
        let width = c_int::try_from(self.size.0).map_err(|_| format!("picture width {}", self.size.0))?;
        let height = c_int::try_from(self.size.1).map_err(|_| format!("picture height {}", self.size.1))?;
        let Ok(count) = c_int::try_from(self.surfaces.len()) else { return Err(format!("{} surfaces", self.surfaces.len())) };
        let mut context: VAContextID = 0;
        // SAFETY: `config` is this session's own config, `surfaces` the array this session owns and
        // `count` its length, and `context` is a live local for the id to be written into.
        let st = unsafe { (api.create_context)(self.display.dpy, self.config, width, height, PROGRESSIVE, self.surfaces.as_mut_ptr(), count, &mut context) };
        if st != OK {
            return Err(format!("vaCreateContext failed at {}x{}: {}", self.size.0, self.size.1, api.status_text(st)));
        }
        Ok(context)
    }

    /// A buffer holding `data` of kind `kind` (a [`PICTURE_PARAMETER_BUFFER`],
    /// [`SLICE_PARAMETER_BUFFER`] or [`SLICE_DATA_BUFFER`]), destroyed when it is dropped.
    ///
    /// `size` is the byte length and `num_elements` is 1. That is what libva's own decode examples
    /// pass for all three kinds (`va.h:3838`: "size" and "num_elements", which for a decoder's
    /// parameter buffers is a single record rather than an array of them), and it is what the
    /// drivers read back with `vaBufferGetNumElements` — a driver asked for one element per byte
    /// refuses the buffer with `VA_STATUS_ERROR_INVALID_PARAMETER`, which is what an Intel iHD driver
    /// does for a 672-byte picture parameter buffer.
    ///
    /// libva copies the data in (`va.h:3826`), so the buffer owns its own copy of the record and the
    /// caller's bytes do not have to outlive the call.
    pub fn buffer(&mut self, kind: VAStatus, data: &[u8]) -> Result<Buffer, String> {
        let Ok(size) = c_uint::try_from(data.len()) else { return Err(format!("{} byte buffer", data.len())) };
        let mut id: VABufferID = 0;
        // SAFETY: the context exists (a buffer is only made for a decode in flight) and `data` is
        // `size` bytes of readable memory, which is the count passed below, so libva copies exactly
        // that many bytes from it. `id` is a live local for the buffer id.
        let st = unsafe { (self.display.api.create_buffer)(self.display.dpy, self.context_id()?, kind, size, 1, data.as_ptr().cast_mut().cast(), &mut id) };
        if st != OK {
            return Err(format!("vaCreateBuffer failed for {size} bytes of type {kind}: {}", self.display.api.status_text(st)));
        }
        Ok(Buffer { display: Arc::clone(&self.display), id })
    }

    /// The id of the decode context, making one if a seek dropped it.
    fn context_id(&mut self) -> Result<VAContextID, String> {
        if let Some(context) = self.context {
            return Ok(context);
        }
        let context = self.make_context()?;
        self.context = Some(context);
        Ok(context)
    }

    /// Decode into `surface`: begin the picture, submit `buffers`, end it.
    ///
    /// `vaEndPicture` is called on every path that got as far as `vaBeginPicture`, so a
    /// `vaRenderPicture` that fails cannot leave the context mid-picture — which libva would refuse
    /// every later call on.
    pub fn render(&mut self, surface: u32, buffers: &[u32]) -> Result<(), String> {
        let api = Arc::clone(&self.display.api);
        let context = self.context_id()?;
        if buffers.is_empty() {
            return Err("a picture with no buffers to render".into());
        }
        let Ok(count) = c_int::try_from(buffers.len()) else { return Err(format!("{} buffers", buffers.len())) };
        let _gate = self.display.gate();
        // SAFETY: `context` is this session's own context and `surface` one of its own surfaces;
        // both were created together and neither has been destroyed.
        let begun = unsafe { (api.begin_picture)(self.display.dpy, context, surface) };
        if begun != OK {
            return Err(format!("vaBeginPicture failed on surface {surface}: {}", api.status_text(begun)));
        }
        // SAFETY: as above, plus `buffers` is a live slice of `count` ids, all of them buffers this
        // session created and not yet destroyed.
        let st = unsafe { (api.render_picture)(self.display.dpy, context, buffers.as_ptr().cast_mut(), count) };
        // SAFETY: as above. `vaEndPicture` is the matching call for the `vaBeginPicture` above, and
        // is made whether or not `vaRenderPicture` succeeded: libva requires the pair.
        let ended = unsafe { (api.end_picture)(self.display.dpy, context) };
        if st != OK {
            return Err(format!("vaRenderPicture failed for {} buffers: {}", buffers.len(), api.status_text(st)));
        }
        if ended != OK {
            return Err(format!("vaEndPicture failed: {}", api.status_text(ended)));
        }
        Ok(())
    }

    /// Wait for the rendering on `surface` to finish. **Required before its pixels are read.**
    ///
    /// `VA_STATUS_ERROR_HW_BUSY` means the video engine was resetting; libva documents that the
    /// call succeeds on a later one, so this asks again rather than reporting a decode failure the
    /// stream may not have had. The retry count is bounded because a driver that keeps answering
    /// `HW_BUSY` is a driver that is not going to answer at all.
    pub fn sync(&self, surface: u32) -> Result<(), String> {
        let api = &self.display.api;
        let _gate = self.display.gate();
        for _ in 0..MAX_SYNC_ATTEMPTS {
            // SAFETY: `surface` is one of this session's own surfaces.
            let st = unsafe { (api.sync_surface)(self.display.dpy, surface) };
            match st {
                OK => return Ok(()),
                HW_BUSY => continue,
                // `VA_STATUS_ERROR_DECODING_ERROR` (`va.h:283`) is the important one: the driver ran
                // the decode and the picture in the surface is not decodable. It is reported like any
                // other failure, and `status_text` says which it was.
                other => return Err(format!("vaSyncSurface failed on surface {surface}: {}", api.status_text(other))),
            }
        }
        Err(format!("vaSyncSurface on surface {surface} still reports the engine busy"))
    }

    /// Whether the driver says `surface` is ready to be read — that is, still being rendered into,
    /// or being displayed. `Err` if the query itself failed.
    pub fn surface_ready(&self, surface: u32) -> Result<bool, String> {
        let api = &self.display.api;
        let _gate = self.display.gate();
        let mut status: c_uint = 0;
        // SAFETY: `surface` is one of this session's own surfaces and `status` is a live local for
        // the flag word libva writes.
        let st = unsafe { (api.query_surface_status)(self.display.dpy, surface, &mut status) };
        if st != OK {
            return Err(format!("vaQuerySurfaceStatus failed on surface {surface}: {}", api.status_text(st)));
        }
        Ok(status & SURFACE_READY != 0)
    }

    /// The pixels of `surface`, mapped and ready to read.
    ///
    /// Two paths, in the order libva documents. `vaDeriveImage` is the direct one: the surface's
    /// own memory is handed back as a `VAImage`, and reading it is one copy. When it returns
    /// `VA_STATUS_ERROR_OPERATION_FAILED` — the documented answer for a surface that cannot be
    /// represented as an image, which is most often a driver whose surfaces are tiled — the picture
    /// is read the indirect way instead: an image of its own is created and the surface copied into
    /// it. That is a real copy through the driver and it is slower, which is why whether a session
    /// needs it is remembered and reported rather than discovered once per picture.
    ///
    /// The caller checks [`VaImage`] against its own gate ([`crate::vaapi::validate_image`]) before
    /// reading a byte, and borrows the mapped bytes through [`Readback::with_bytes`]; the map is
    /// paired with its unmap and the image with its destroy in [`Readback`]'s `Drop`, so a failure
    /// between here and the read cannot leak either.
    pub fn read(&mut self, surface: u32) -> Result<Readback, String> {
        if self.indirect {
            return self.read_indirect(surface);
        }
        let mut image = VaImage::default();
        let status = {
            let _gate = self.display.gate();
            // SAFETY: `surface` is one of this session's own surfaces and `image` is a live local
            // libva fills in with the image's own description.
            unsafe { (self.display.api.derive_image)(self.display.dpy, surface, &mut image) }
        };
        if status == OPERATION_FAILED {
            self.indirect = true;
            log::info!("VA-API: this driver's surfaces cannot be derived into an image; reading pictures back through a copy");
            return self.read_indirect(surface);
        }
        if status != OK {
            return Err(format!("vaDeriveImage failed on surface {surface}: {}", self.display.api.status_text(status)));
        }
        let display = Arc::clone(&self.display);
        let read = Readback::map(display, image, true)?;
        if read.image().format.fourcc != self.fourcc {
            return Err(format!(
                "the driver derived surface {surface} as fourcc {:#x}, not the {:#x} this session decodes into",
                read.image().format.fourcc,
                self.fourcc
            ));
        }
        Ok(read)
    }

    /// The indirect readback: `vaCreateImage` + `vaPutImage` (`va.h:4892`), the fallback `vaDeriveImage`
    /// documents. An error on either step destroys the image, so a failure here leaves nothing
    /// behind.
    fn read_indirect(&self, surface: u32) -> Result<Readback, String> {
        let (Some(create_image), Some(destroy_image), Some(put_image)) =
            (self.display.api.create_image, self.display.api.destroy_image, self.display.api.put_image)
        else {
            return Err("this libva has no vaCreateImage / vaPutImage, so a surface that cannot be derived cannot be read".into());
        };
        let api = &self.display.api;
        let mut format = VaImageFormat { fourcc: self.fourcc, byte_order: BYTE_ORDER_LSB_FIRST, bits_per_pixel: self.bpp, ..Default::default() };
        let mut image = VaImage::default();
        let (width, height) = (
            c_int::try_from(self.size.0).map_err(|_| format!("picture width {}", self.size.0))?,
            c_int::try_from(self.size.1).map_err(|_| format!("picture height {}", self.size.1))?,
        );
        let st = {
            let _gate = self.display.gate();
            // SAFETY: the display is initialised, `format` and `image` are live locals, and the sizes
            // are the ones the image is allocated at.
            unsafe { create_image(self.display.dpy, &mut format, width, height, &mut image) }
        };
        if st != OK {
            return Err(format!("vaCreateImage failed for the indirect readback: {}", api.status_text(st)));
        }
        // From here the image exists, so every exit has to destroy it. `Readback::map` takes that
        // over as soon as the image is mapped; before that the two calls below do it themselves.
        let extent = |s: u32| c_uint::try_from(s).unwrap_or(0);
        let st = {
            let _gate = self.display.gate();
            // SAFETY: `image` is the image just created, and `surface` is one of this session's own
            // surfaces; the rectangle is the whole of both.
            unsafe {
                put_image(
                    self.display.dpy,
                    surface,
                    image.image_id,
                    0,
                    0,
                    extent(self.size.0),
                    extent(self.size.1),
                    0,
                    0,
                    extent(self.size.0),
                    extent(self.size.1),
                )
            }
        };
        if st != OK {
            // SAFETY: the image this function created, destroyed once, on its way out.
            unsafe { destroy_image(self.display.dpy, image.image_id) };
            return Err(format!("vaPutImage failed on surface {surface}: {}", api.status_text(st)));
        }
        let display = Arc::clone(&self.display);
        // `vaCreateImage` made this image, so this function owns it until a `Readback` does — and
        // `Readback::map` only takes ownership on success. A map that fails therefore has to destroy
        // it here, which is what `derived == false` says: `Readback::map` destroys a *derived* image
        // itself (nothing else can tell libva the id is unwanted) but leaves a created one to the
        // caller, because a created image's id is the caller's to release.
        let image_id = image.image_id;
        match Readback::map(display, image, false) {
            Ok(read) => Ok(read),
            Err(why) => {
                let _gate = self.display.gate();
                // SAFETY: `image_id` is the image `vaCreateImage` filled in above, it was never mapped
                // (so never unmapped), and nothing else holds it — this is its one destroy.
                unsafe { destroy_image(self.display.dpy, image_id) };
                Err(why)
            }
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let api = Arc::clone(&self.display.api);
        let _gate = self.display.gate();
        // libva's own order: surfaces may only be destroyed after the contexts that use them.
        if let Some(context) = self.context.take() {
            // SAFETY: `context` is this session's own context, taken out of the option so no other
            // reference to it exists, and this is its matching destroy.
            unsafe { (api.destroy_context)(self.display.dpy, context) };
        }
        if !self.surfaces.is_empty() {
            let count = c_int::try_from(self.surfaces.len()).unwrap_or(0);
            // SAFETY: `self.surfaces` is the array of surface ids this session created, `count` is its
            // length (zero on the conversion that cannot represent it, which destroys none), and this
            // is their one destroy.
            unsafe { (api.destroy_surfaces)(self.display.dpy, self.surfaces.as_mut_ptr(), count) };
        }
        if self.config != 0 {
            // SAFETY: `config` is this session's own config id, created in `Session::new` and not
            // destroyed on any earlier path, and this is its one destroy.
            unsafe { (api.destroy_config)(self.display.dpy, self.config) };
        }
    }
}

/// A buffer submitted to a decode, destroyed when it is dropped.
///
/// Destroying a buffer is how libva is told a decode is finished with it, and a buffer that outlives
/// its decode holds driver memory. `Drop` rather than a call the decode loop has to remember, so
/// that an error between `vaCreateBuffer` and `vaRenderPicture` cannot leak one.
pub struct Buffer {
    display: Arc<Display>,
    id: VABufferID,
}

impl Buffer {
    /// The buffer id `vaRenderPicture` takes.
    pub fn id(&self) -> u32 {
        self.id
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let _gate = self.display.gate();
        // SAFETY: `id` is the buffer this value owns and was created by the same display, which it
        // holds a reference to; this is its one destroy.
        unsafe { (self.display.api.destroy_buffer)(self.display.dpy, self.id) };
    }
}

/// A decoded picture's pixels, mapped and readable — what a readback returns.
///
/// The mapped region belongs to the driver and is only valid until this is dropped, which is why it
/// is lent rather than handed out: [`Readback::with_bytes`] borrows it for the duration of a
/// closure, so nothing can hold a slice past the unmap. Dropping unmaps the buffer and destroys the
/// image, on every path: a decode that fails between mapping and reading leaks neither.
pub struct Readback {
    /// Kept alive so the display — and through it the libva handle the mapping belongs to — outlives
    /// the unmap in `Drop`.
    display: Arc<Display>,
    /// The driver's own description of the image, which the readback gate reads.
    image: VaImage,
    /// The mapped buffer, or null when `vaMapBuffer` has not been called (it cannot have, or `map`
    /// would have failed).
    data: *mut c_void,
    mapped: bool,
}

impl Readback {
    /// Map `image`'s buffer. `derived` says whether the image came from `vaDeriveImage` or from
    /// `vaCreateImage`, which is what [`destroy_unread_image`] and `Drop` have to match.
    fn map(display: Arc<Display>, image: VaImage, derived: bool) -> Result<Self, String> {
        let mut data: *mut c_void = std::ptr::null_mut();
        let status = {
            let _gate = display.gate();
            // SAFETY: `image.buf` is the buffer of an image libva created on this display, and
            // `data` is a live local for the address libva writes into it.
            unsafe { (display.api.map_buffer)(display.dpy, image.buf, &mut data) }
        };
        if status != OK {
            let api = &display.api;
            destroy_unread_image(&display, &image, derived);
            return Err(format!("vaMapBuffer failed: {}", api.status_text(status)));
        }
        if data.is_null() {
            // A successful map with no address is a driver that did not do what it said. There is
            // nothing to unmap, but the image still has to be destroyed — see `destroy_unread_image`
            // for why that is only for a derived one.
            destroy_unread_image(&display, &image, derived);
            return Err("vaMapBuffer succeeded but returned no address".into());
        }
        Ok(Self { display, image, data, mapped: true })
    }

    /// The image as the driver described it: its size, planes, pitches and offsets. This is what the
    /// readback gate reads before a byte is read out of the mapped buffer.
    pub fn image(&self) -> &VaImage {
        &self.image
    }

    /// Borrow the mapped buffer's two planes for the duration of `f`, as
    /// `(luma, chroma, stride, rows)`: a luma plane of `rows` rows of `stride` bytes, and a chroma
    /// plane of `rows / 2` rows of the same stride, which is what a biplanar 4:2:0 image is.
    ///
    /// This is the one place a driver-reported offset becomes an address a pixel read happens at, so
    /// every number in it is bounded against `data_size` — the only length libva reports for a mapped
    /// buffer — and both planes are required to be inside it. The closure cannot keep either slice: it
    /// returns nothing, so the borrow ends when it does and the unmap in `Drop` cannot pull the
    /// ground out from under a caller.
    ///
    /// `data_size` is what the caller checked against the picture geometry (see
    /// [`crate::vaapi::validate_image`]); this is the second gate, and it does not rely on that check
    /// having been made, because this is the one that indexes.
    pub fn with_mapped(&self, f: impl FnOnce(&[u8], &[u8], usize, usize)) {
        // SAFETY: `data` is what `vaMapBuffer` returned for this image's buffer and is still mapped
        // (only `Drop` unmaps it, and only once, with this value borrowed throughout), so the slice
        // below is `data_size` readable bytes; both slices handed to `f` are subranges of it, which is
        // what the `get` calls check, and neither outlives the borrow of `self`.
        let bytes = unsafe { std::slice::from_raw_parts(self.data.cast::<u8>(), self.image.data_size as usize) };
        let (Some(&luma_pitch), Some(&chroma_pitch), Some(&luma_at), Some(&chroma_at)) =
            (self.image.pitches.first(), self.image.pitches.get(1), self.image.offsets.first(), self.image.offsets.get(1))
        else {
            return f(&[], &[], 0, 0);
        };
        let (stride, chroma_stride, rows) = (luma_pitch as usize, chroma_pitch as usize, self.image.height as usize);
        let chroma_rows = rows.div_ceil(2);
        // Each plane's own pitch, not the luma's: reading the chroma plane at the luma pitch would
        // walk off it. But `biplanar::Biplanar` has a *single* stride for both planes, so a driver
        // that pitches them differently cannot be read through it at all — `validate_image` refuses
        // that image before this is reached, and this second check is what makes the refusal hold even
        // if a caller skipped it (an empty pair makes `to_frame` fail, which is an error rather than
        // a picture read at the wrong stride).
        if chroma_stride != stride {
            return f(&[], &[], 0, 0);
        }
        let (Some(luma_len), Some(chroma_len)) = (stride.checked_mul(rows), chroma_stride.checked_mul(chroma_rows)) else {
            return f(&[], &[], 0, 0);
        };
        let (Some(luma), Some(chroma)) =
            (bytes.get(luma_at as usize..).and_then(|b| b.get(..luma_len)), bytes.get(chroma_at as usize..).and_then(|b| b.get(..chroma_len)))
        else {
            return f(&[], &[], 0, 0);
        };
        f(luma, chroma, stride, rows)
    }
}

impl Drop for Readback {
    fn drop(&mut self) {
        let _gate = self.display.gate();
        if self.mapped {
            // SAFETY: `image.buf` is the buffer mapped by `Readback::map` and has not been unmapped
            // since (only this `Drop` unmaps it, and only once, with the value borrowed throughout), so
            // this is its one unmap.
            unsafe { (self.display.api.unmap_buffer)(self.display.dpy, self.image.buf) };
            self.mapped = false;
        }
        if let Some(destroy) = self.display.api.destroy_image {
            // SAFETY: `image.image_id` is the id libva wrote for this image, which this value owns:
            // either `vaDeriveImage` filled it in or `vaCreateImage` created it, and nothing else holds
            // it. This is its one destroy.
            unsafe { destroy(self.display.dpy, self.image.image_id) };
        }
    }
}

/// Destroy an image that was created but never read: the paths out of [`Readback::map`] where the map
/// failed, so that a failed readback leaks neither the image nor its buffer.
///
/// Only for a *derived* image. An image from `vaCreateImage` whose map failed is the caller's to
/// destroy — [`Session::read_indirect`] does, on its own error path — because libva has no way to
/// tell a caller that its own `vaCreateImage` id is no longer wanted, and destroying it here would be
/// a double destroy.
fn destroy_unread_image(display: &Arc<Display>, image: &VaImage, derived: bool) {
    if !derived {
        return;
    }
    let _gate = display.gate();
    if let Some(destroy) = display.api.destroy_image {
        // SAFETY: `image.image_id` is the id `vaDeriveImage` wrote into `image`, the image was never
        // mapped and so never unmapped, and nothing else holds the id, so this is its one destroy.
        unsafe { destroy(display.dpy, image.image_id) };
    }
}

// ---------------------------------------------------------------------------------------------
// Tests: the hand-declared ABI against `/usr/include/va/`, and against the library that is
// installed. Mirrors `nvenc/abi_tests.rs`, except that the header is present on the machine that
// runs this, so the layout can be compared with libva itself.
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// The loader and the mandatory `vaInitialize` work on a machine that has them: both libraries
    /// open, every required symbol resolves, and the display reports a version and a profile list.
    /// What a profile then supports is a separate question, answered by the decoder's own tests.
    #[test]
    fn the_libraries_load_and_the_display_initializes() {
        let Ok(api) = load() else {
            eprintln!("SKIPPED: no libva");
            return;
        };
        let Ok(d) = Display::new(&api) else {
            eprintln!("SKIPPED: no VA display");
            return;
        };
        // The loader opened both libraries, found every required symbol and initialised the
        // display, which is what this test is for. Beyond that, only what this backend promises:
        // a display reports its version and answers a profile query. Whether any profile offers a
        // video decoder is a capability question, not a load question — a driver that offers only
        // encode entry points is a machine with no hardware decoder, which is a legitimate answer
        // (see the module doc), not a failure.
        let (major, _) = d.va_version();
        assert!(major >= 1, "vaInitialize reported no libva version");
        assert!(!d.profiles().expect("vaQueryConfigProfiles").is_empty(), "the driver supports no profile at all");
    }

    /// `OK` is success, libva has words for it, and the load-free path a [`Display`] error uses
    /// says the same thing as the public `status` — otherwise decorating an error there would
    /// quietly produce a different message than everything else.
    #[test]
    fn va_status_zero_is_success() {
        assert_eq!(OK, 0);
        assert!(!status(OK).is_empty());
        let Ok(api) = load() else {
            eprintln!("SKIPPED: no libva");
            return;
        };
        assert_eq!(api.status_text(OK), status(OK), "status() and Api::status_text disagree");
        assert!(!api.status_text(OK).is_empty());
        // A code libva does not know still names itself rather than coming back empty.
        assert!(!api.status_text(-999).is_empty(), "an unknown status is not described");
    }

    /// Every libva 2.24 constant this module carries, pinned so a wrong one fails on every machine
    /// rather than only where a driver happens to be installed. The profile numbers are the ones
    /// that matter most: a wrong one does not fail a call, it configures the wrong codec, and the
    /// symptom is "this stream decodes in software" rather than a test failure.
    #[test]
    fn the_format_constants_are_the_ones_libva_uses() {
        assert_eq!(YUV420, 0x0000_0001);
        assert_eq!(YUV420_10, 0x0000_0100, "VA_RT_FORMAT_YUV420_10 is 0x100");
        assert_eq!(VA_PADDING_LOW, 4);
        assert_eq!(VA_PADDING_MEDIUM, 8);
        assert_eq!(PICTURE_PARAMETER_BUFFER, 0);
        assert_eq!(IQ_MATRIX_BUFFER, 1, "VAIQMatrixBufferType");
        assert_eq!(SLICE_PARAMETER_BUFFER, 4);
        assert_eq!(SLICE_DATA_BUFFER, 5);
        assert_eq!(VLD, 1);
        assert_eq!(ENC_SLICE, 6, "VAEntrypointEncSlice");
        assert_eq!(H264_MAIN, 6, "VAProfileH264Main");
        assert_eq!(H264_HIGH, 7, "VAProfileH264High");
        assert_eq!(H264_CONSTRAINED_BASELINE, 13, "VAProfileH264ConstrainedBaseline");
        assert_eq!(HEVC_MAIN, 17, "VAProfileHEVCMain");
        assert_eq!(HEVC_MAIN_10, 18, "VAProfileHEVCMain10");
        assert_eq!(VP9_PROFILE_0, 19, "VAProfileVP9Profile0");
        assert_eq!(VP9_PROFILE_1, 20, "VAProfileVP9Profile1");
        assert_eq!(VP9_PROFILE_2, 21, "VAProfileVP9Profile2");
        assert_eq!(VP9_PROFILE_3, 22, "VAProfileVP9Profile3");
        assert_eq!(AV1_MAIN, 32, "VAProfileAV1Profile0");
        assert_eq!(OPERATION_FAILED, 0x1, "VA_STATUS_ERROR_OPERATION_FAILED");
        assert_eq!(HW_BUSY, 0x22, "VA_STATUS_ERROR_HW_BUSY");
        assert_eq!(SURFACE_READY, 0x4, "VASurfaceReady");
        assert_eq!(PROGRESSIVE, 0x1, "VA_PROGRESSIVE");
        // The IQ matrix record's shape, which is what `h264_iq_matrix` fills: 96 bytes of 4x4
        // lists, 128 of 8x8, then the reserved tail — 240 in total, and the driver reads a
        // submission of any other length as something else entirely.
        assert_eq!(size_of::<VaIqMatrixBufferH264>(), 240, "VAIQMatrixBufferH264 is 240 bytes");
        assert_eq!(offset_of!(VaIqMatrixBufferH264, scaling_list4x4), 0);
        assert_eq!(offset_of!(VaIqMatrixBufferH264, scaling_list8x8), 96, "after the six 4x4 lists");
        assert_eq!(offset_of!(VaIqMatrixBufferH264, va_reserved), 224, "and the two 8x8 luma lists are 128 bytes");
        assert_eq!(NV12_BPP, 12, "libva reports 12 bits per pixel for NV12, not 8");
        assert_eq!(P010_BPP, 24, "and 24 for P010, not 10");
        assert_eq!(NV12_FOURCC, 0x3231_564e, "VA_FOURCC_NV12");
        assert_eq!(P010_FOURCC, 0x3031_3050, "VA_FOURCC_P010");
        assert_eq!(PICTURE_HEVC_LONG_TERM_REFERENCE, 0x8, "VA_PICTURE_HEVC_LONG_TERM_REFERENCE");
        assert_eq!(PICTURE_HEVC_RPS_ST_CURR_BEFORE, 0x10);
        assert_eq!(PICTURE_HEVC_RPS_ST_CURR_AFTER, 0x20);
        assert_eq!(PICTURE_HEVC_RPS_LT_CURR, 0x40);
    }

    /// The layouts of the two attribute structures a decoder passes a null array of: libva takes
    /// `(attrib_list, num_attribs)` and the array's type is in the signature, so a caller that passes
    /// nothing still needs the struct to exist and be the right size.
    #[test]
    fn the_attribute_records_match_the_header() {
        assert_eq!(size_of::<VaConfigAttrib>(), 8, "VAConfigAttrib (va.h:1072)");
        assert_eq!(offset_of!(VaConfigAttrib, value), 4);
        assert_eq!(size_of::<VaSurfaceAttrib>(), 16, "VASurfaceAttrib (va.h:1742)");
        assert_eq!(offset_of!(VaSurfaceAttrib, flags), 4);
        assert_eq!(offset_of!(VaSurfaceAttrib, value), 8, "the VAGenericValue tail");
        assert_eq!(size_of::<VaGenericValue>(), 8, "VAGenericValue (va.h:245)");
    }

    /// A driver that will not decode into a derived image says so with
    /// `VA_STATUS_ERROR_OPERATION_FAILED`, and that is the only status this backend reads a surface
    /// back the indirect way for. A driver that fails for any other reason is an error, not a reason
    /// to copy the picture a second way: the distinction is what keeps a broken driver from being
    /// quietly worked around.
    #[test]
    fn only_operation_failed_is_treated_as_a_derive_failure() {
        assert_eq!(OPERATION_FAILED, 1);
        assert_ne!(OPERATION_FAILED, OK);
        assert_ne!(OPERATION_FAILED, HW_BUSY);
    }

    /// The picture- and slice-parameter records, as libva 2.24's headers declare them, so a field
    /// declared with the wrong type or width — which the driver would read as a neighbouring field's
    /// value — fails here rather than as corruption on screen. The sizes come from the headers on
    /// this machine; the offsets are the ones the C declarations imply, and every struct's trailing
    /// reserved field is checked too, since that is what makes each record a stable size.
    #[test]
    fn the_picture_parameter_records_match_the_header() {
        // H.264 (va.h:3572-3721)
        assert_eq!(size_of::<VaPictureH264>(), 36, "VAPictureH264 (va.h:3572)");
        assert_eq!(offset_of!(VaPictureH264, frame_idx), 4);
        assert_eq!(offset_of!(VaPictureH264, flags), 8);
        assert_eq!(offset_of!(VaPictureH264, top_field_order_cnt), 12);
        assert_eq!(offset_of!(VaPictureH264, bottom_field_order_cnt), 16);
        assert_eq!(offset_of!(VaPictureH264, va_reserved), 20);
        assert_eq!(size_of::<VaPictureParameterBufferH264>(), 672, "VAPictureParameterBufferH264 (va.h:3599)");
        assert_eq!(offset_of!(VaPictureParameterBufferH264, curr_pic), 0);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, reference_frames), 36);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, picture_width_in_mbs_minus1), 612);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, picture_height_in_mbs_minus1), 614);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, bit_depth_luma_minus8), 616);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, num_ref_frames), 618);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, padding_after_num_ref_frames), 619, "the C layout's hole before seq_fields");
        assert_eq!(offset_of!(VaPictureParameterBufferH264, seq_fields), 620);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, num_slice_groups_minus1), 624);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, slice_group_change_rate_minus1), 626);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, pic_init_qp_minus26), 628);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, second_chroma_qp_index_offset), 631);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, pic_fields), 632);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, frame_num), 636);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, padding_after_frame_num), 638, "the C layout's hole before va_reserved");
        assert_eq!(offset_of!(VaPictureParameterBufferH264, va_reserved), 640);
        assert_eq!(size_of::<VaSliceParameterBufferH264>(), 3128, "VASliceParameterBufferH264 (va.h:3664)");
        assert_eq!(offset_of!(VaSliceParameterBufferH264, slice_data_bit_offset), 12);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, first_mb_in_slice), 14);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, slice_type), 16);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, slice_beta_offset_div2), 24);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, padding_before_ref_pic_list0), 25, "the C layout's hole before RefPicList0[]");
        assert_eq!(offset_of!(VaSliceParameterBufferH264, ref_pic_list0), 28);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, ref_pic_list1), 1180);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, luma_log2_weight_denom), 2332);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, padding_before_luma_weight_l0), 2335, "the hole before luma_weight_l0[]");
        assert_eq!(offset_of!(VaSliceParameterBufferH264, luma_offset_l0), 2400);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, chroma_weight_l0_flag), 2464);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, padding_before_chroma_weight_l0), 2465, "the hole before chroma_weight_l0[][]");
        assert_eq!(offset_of!(VaSliceParameterBufferH264, chroma_offset_l0), 2594);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, luma_weight_l1_flag), 2722);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, padding_before_luma_weight_l1), 2723, "the hole before luma_weight_l1[]");
        assert_eq!(offset_of!(VaSliceParameterBufferH264, chroma_offset_l1), 2982);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, padding_before_va_reserved), 3110, "the hole before va_reserved");
        assert_eq!(offset_of!(VaSliceParameterBufferH264, va_reserved), 3112);
        // HEVC (va.h:5292, va_dec_hevc.h:57-479)
        assert_eq!(size_of::<VaPictureHevc>(), 28, "VAPictureHEVC (va.h:5292)");
        assert_eq!(offset_of!(VaPictureHevc, pic_order_cnt), 4);
        assert_eq!(offset_of!(VaPictureHevc, flags), 8);
        assert_eq!(offset_of!(VaPictureHevc, va_reserved), 12);
        assert_eq!(size_of::<VaPictureParameterBufferHevc>(), 604, "VAPictureParameterBufferHEVC (va_dec_hevc.h:57)");
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, reference_frames), 28);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, pic_width_in_luma_samples), 448);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, pic_height_in_luma_samples), 450);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, pic_fields), 452);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, sps_max_dec_pic_buffering_minus1), 456);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, log2_min_luma_coding_block_size_minus3), 461);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, init_qp_minus26), 469);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, log2_parallel_merge_level_minus2), 473);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, column_width_minus1), 476);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, row_height_minus1), 514);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, slice_parsing_fields), 556);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, log2_max_pic_order_cnt_lsb_minus4), 560);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, num_extra_slice_header_bits), 567);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, st_rps_bits), 568);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, va_reserved), 572);
        assert_eq!(size_of::<VaSliceParameterBufferHevc>(), 264, "VASliceParameterBufferHEVC (va_dec_hevc.h:358)");
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, slice_data_byte_offset), 12);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, slice_segment_address), 16);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, ref_pic_list), 20);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, padding_before_long_slice_flags), 50, "the C layout's hole before LongSliceFlags");
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, long_slice_flags), 52);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, collocated_ref_idx), 56);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, slice_qp_delta), 59);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, slice_beta_offset_div2), 62);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, luma_log2_weight_denom), 64);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, chroma_offset_l1), 216);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, five_minus_max_num_merge_cand), 246);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, padding_before_num_entry_point_offsets), 247, "the hole before num_entry_point_offsets");
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, slice_data_num_emu_prevn_bytes), 252);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, padding_before_va_reserved), 254, "the hole before va_reserved");
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, va_reserved), 256);
    }

    /// Every byte of a record is a named field, so no byte of a default record is indeterminate —
    /// which is what lets `record_bytes` hand them to a driver as a `&[u8]`. The C layout's
    /// alignment holes are declared as `padding_*` fields; this checks that every default record is
    /// all zeroes, so a field added without a value, or a padding field that stopped being one,
    /// shows up here.
    #[test]
    fn a_default_record_is_every_byte_zero() {
        for bytes in [
            record_bytes(&VaPictureH264::default()),
            record_bytes(&VaPictureParameterBufferH264::default()),
            record_bytes(&VaSliceParameterBufferH264::default()),
            record_bytes(&VaPictureHevc::default()),
            record_bytes(&VaPictureParameterBufferHevc::default()),
            record_bytes(&VaSliceParameterBufferHevc::default()),
        ] {
            assert!(!bytes.is_empty());
            assert!(
                bytes.iter().all(|&b| b == 0),
                "a default record has a non-zero byte: {} of {}",
                bytes.iter().position(|&b| b != 0).unwrap_or(0),
                bytes.len()
            );
        }
    }

    /// A record read back out of the bytes `picture.rs` built is the same record it built, and a
    /// buffer too short for one is refused rather than read past. The two sides only agree if
    /// `record_bytes` and `record_of` describe the same layout, which is what makes them a pair.
    #[test]
    fn a_record_round_trips_through_its_bytes() {
        let picture = VaPictureParameterBufferH264 {
            curr_pic: VaPictureH264 {
                picture_id: 7,
                frame_idx: 3,
                flags: PICTURE_H264_SHORT_TERM_REFERENCE,
                top_field_order_cnt: 11,
                bottom_field_order_cnt: 22,
                ..Default::default()
            },
            picture_width_in_mbs_minus1: 39,
            picture_height_in_mbs_minus1: 22,
            num_ref_frames: 4,
            seq_fields: 0x1234_5678,
            pic_fields: 0x9abc_def0,
            frame_num: 3,
            ..Default::default()
        };
        let bytes = record_bytes(&picture);
        assert_eq!(bytes.len(), size_of::<VaPictureParameterBufferH264>());
        assert_eq!(picture_params_h264(&bytes), Some(picture), "the record reads back as the one built");

        // A short buffer is `None`, and a longer one is still the record: a driver reads a prefix.
        assert_eq!(picture_params_h264(&bytes[..size_of::<VaPictureParameterBufferH264>() - 1]), None, "a byte short");
        assert_eq!(picture_params_h264(&[]), None);
        assert_eq!(picture_params_h264(&bytes[..10]), None, "an empty slice of it");
        assert_eq!(slice_params_h264(&bytes), None, "an H.264 picture record read as a slice record");

        // The four records are four different sizes. A buffer shorter than the record is refused; a longer
        // one still reads as the record at its front, which is why the caller has to know which
        // record it built rather than letting a reader guess.
        assert_eq!(record_bytes(&VaPictureParameterBufferH264::default()).len(), size_of::<VaPictureParameterBufferH264>());
        assert_eq!(record_bytes(&VaSliceParameterBufferH264::default()).len(), size_of::<VaSliceParameterBufferH264>());
        assert_eq!(record_bytes(&VaPictureParameterBufferHevc::default()).len(), size_of::<VaPictureParameterBufferHevc>());
        assert_eq!(record_bytes(&VaSliceParameterBufferHevc::default()).len(), size_of::<VaSliceParameterBufferHevc>());
        assert_eq!(slice_params_h264(&bytes), None, "an H.264 picture record is not long enough to be a slice record");
        let extra = [bytes.as_slice(), &[0u8; 8]].concat();
        assert_eq!(picture_params_h264(&extra).expect("a record at the front of a longer buffer"), picture, "trailing bytes are ignored");
    }

    /// The layout as `/usr/include/va/va.h` has it, so a field added or moved here is a deliberate,
    /// reviewed change rather than a silent ABI change.
    #[test]
    fn va_image_layout_matches_the_header() {
        assert_eq!(size_of::<VaImage>(), 120, "VAImage size (va.h:4734)");
        assert_eq!(size_of::<VaImageFormat>(), 48, "VAImageFormat size (va.h:4717)");
        assert_eq!(offset_of!(VaImage, format), 4, "VAImage.format");
        assert_eq!(offset_of!(VaImage, buf), 52, "VAImage.buf");
        assert_eq!(offset_of!(VaImage, width), 56, "VAImage.width");
        assert_eq!(offset_of!(VaImage, height), 58, "VAImage.height");
        assert_eq!(offset_of!(VaImage, data_size), 60, "VAImage.data_size");
        assert_eq!(offset_of!(VaImage, num_planes), 64, "VAImage.num_planes");
        assert_eq!(offset_of!(VaImage, pitches), 68, "VAImage.pitches");
        assert_eq!(offset_of!(VaImage, offsets), 80, "VAImage.offsets");
        assert_eq!(offset_of!(VaImage, num_palette_entries), 92, "VAImage.num_palette_entries");
        assert_eq!(offset_of!(VaImage, entry_bytes), 96, "VAImage.entry_bytes");
        assert_eq!(offset_of!(VaImage, component_order), 100, "VAImage.component_order");
    }

    /// The layout against the library itself: the driver fills an image and the plane geometry it
    /// writes has to read back as NV12 planes inside the size it reported.
    #[test]
    fn va_image_plane_geometry_comes_from_the_driver() {
        let Ok(api) = load() else {
            eprintln!("SKIPPED: no libva");
            return;
        };
        let Ok(d) = Display::new(&api) else {
            eprintln!("SKIPPED: no VA display");
            return;
        };
        let Ok(ran) = d.probe_image_layout() else {
            eprintln!("SKIPPED: the driver will not create a probe image");
            return;
        };
        assert!(ran, "the plane geometry libva wrote is not where this module reads it");
    }

    /// The profile and entry point mapping, including the values the mapping must not lose.
    #[test]
    fn profiles_and_entrypoints_round_trip() {
        for p in [
            Profile::H264ConstrainedBaseline,
            Profile::H264Main,
            Profile::H264High,
            Profile::HevcMain,
            Profile::HevcMain10,
            Profile::Vp9(0),
            Profile::Vp9(3),
            Profile::Av1Main,
            Profile::Other(1234),
        ] {
            assert_eq!(Profile::from_bits(p.bits()), p, "{p:?}");
        }
        assert_eq!(Profile::Vp9(0).bits(), 19, "VAProfileVP9Profile0");
        assert_eq!(Profile::Vp9(3).bits(), 22, "VAProfileVP9Profile3");
        assert_eq!(Profile::from_bits(45), Profile::Other(45), "a profile we do not name is kept");
        for e in [Entrypoint::Vld, Entrypoint::EncSlice, Entrypoint::Other(9)] {
            assert_eq!(Entrypoint::from_bits(e.bits()), e, "{e:?}");
        }
        for f in [RtFormat::Nv12, RtFormat::Yuv420_10, RtFormat::Other(0x1234)] {
            assert_eq!(RtFormat::from(f.bits()), f, "{f:?}");
        }
        assert_eq!(RtFormat::Nv12.bits(), YUV420);
        assert_eq!(RtFormat::Yuv420_10.bits(), YUV420_10);
    }

    /// A driver count is what sizes the array handed to it, so both ends are bounded.
    #[test]
    fn driver_counts_are_bounded() {
        assert_eq!(capacity(-1), MAX_DRIVER_ENTRIES, "a nonsense maximum still gets an array");
        assert_eq!(capacity(0), 1, "zero still leaves the driver somewhere to write");
        assert_eq!(capacity(12), 12);
        assert_eq!(capacity(c_int::MAX), MAX_DRIVER_ENTRIES, "an absurd maximum is capped");
    }

    /// Plane geometry that does not add up is refused: this is what a wrong field offset looks
    /// like, and what a hostile `data_size` would look like.
    #[test]
    fn plane_geometry_is_checked_not_trusted() {
        let mut image = VaImage { width: 64, height: 64, data_size: 64 * 64 * 3 / 2, num_planes: 1, pitches: [64, 0, 0], ..Default::default() };
        assert!(plane_geometry_is_consistent(&image, 64, 64));

        image.pitches = [64, 0, 0];
        image.offsets = [image.data_size - 64, 0, 0];
        assert!(!plane_geometry_is_consistent(&image, 64, 64), "a plane that ends outside data_size");

        image.offsets = [0, 0, 0];
        image.pitches = [u32::MAX, 0, 0];
        assert!(!plane_geometry_is_consistent(&image, 64, 64), "a pitch that overflows");

        image.pitches = [64, 0, 0];
        image.num_planes = 4;
        assert!(!plane_geometry_is_consistent(&image, 64, 64), "more planes than VAImage has");

        image.num_planes = 1;
        image.width = 32;
        assert!(!plane_geometry_is_consistent(&image, 64, 64), "a size we did not ask for");
    }
}
