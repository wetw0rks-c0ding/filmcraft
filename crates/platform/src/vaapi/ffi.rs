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

use std::ffi::{CStr, CString, c_char, c_int, c_void};
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

/// `VABufferType` (`va.h:2042`): the buffer kinds a decode loop submits, in the order libva
/// numbers them.
pub const PICTURE_PARAMETER_BUFFER: VAStatus = 0;
/// See [`PICTURE_PARAMETER_BUFFER`].
pub const SLICE_PARAMETER_BUFFER: VAStatus = 4;
/// See [`PICTURE_PARAMETER_BUFFER`].
pub const SLICE_DATA_BUFFER: VAStatus = 5;
/// See [`PICTURE_PARAMETER_BUFFER`].
pub const IMAGE_BUFFER: VAStatus = 9;

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
    fourcc: u32,
    /// `VA_LSB_FIRST` or `VA_MSB_FIRST` (`va.h:4714-4715`).
    byte_order: u32,
    bits_per_pixel: u32,
    /// Significant bits per pixel.
    depth: u32,
    red_mask: u32,
    green_mask: u32,
    blue_mask: u32,
    alpha_mask: u32,
    va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAImage` (`va.h:4734`): the description libva hands back for a decoded picture — its size, how
/// many planes it has, and where each plane starts and how it is pitched. Every readback path
/// copies out of this, so a field declared at the wrong offset or width reads its neighbour's
/// value instead of its own; [`Display::probe_image_layout`] is the check against the library that
/// is installed, and `va_image_layout_matches_the_header` pins the offsets against the header.
/// 120 bytes.
///
/// The fields are private to this file: they are libva's memory layout rather than FilmCraft's
/// data, so a decoder asks this module for plane geometry instead of reading them itself.
#[repr(C)]
#[derive(Default)]
pub struct VaImage {
    image_id: VAImageID,
    format: VaImageFormat,
    /// The buffer (`VAImageBufferType`) the image data lives in, to be mapped and unmapped.
    buf: VABufferID,
    width: u16,
    height: u16,
    /// Bytes the driver allocated for all planes together.
    data_size: u32,
    /// 1..=3 for the planar formats; whatever else a driver reports is not read.
    num_planes: u32,
    /// Bytes per row of each plane.
    pitches: [u32; 3],
    /// Byte offset of each plane within the mapped buffer.
    offsets: [u32; 3],
    /// Palette bookkeeping; zero for the YUV formats this backend takes.
    num_palette_entries: i32,
    entry_bytes: i32,
    component_order: [i8; 4],
    va_reserved: [u32; VA_PADDING_LOW],
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
    /// Optional: the image calls are only the ABI layout probe (and, later, readback).
    create_image: Option<CreateImageFn>,
    destroy_image: Option<DestroyImageFn>,
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
        assert_eq!(SLICE_PARAMETER_BUFFER, 4);
        assert_eq!(SLICE_DATA_BUFFER, 5);
        assert_eq!(IMAGE_BUFFER, 9);
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
