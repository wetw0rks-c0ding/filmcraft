// The declarations in this file are transcribed from libva's `va/va.h`, `va/va_dec_hevc.h`, `va/va_enc_hevc.h` and `va/va_drm.h`
// (VA-API 1.23, libva 2.23; the HEVC encode records measured against the installed 1.24.0 headers, which re-measure every
// record already pinned here identically), which carry this notice:
//
// Copyright (c) 2007-2009 Intel Corporation. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sub license, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice (including the
// next paragraph) shall be included in all copies or substantial portions
// of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
// IN NO EVENT SHALL INTEL AND/OR ITS SUPPLIERS BE LIABLE FOR
// ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! VA-API (`va.h`, `va_dec_hevc.h`, `va_enc_hevc.h`, `va_drm.h`; MIT-licensed headers by Intel) as `libva.so.2` /
//! `libva-drm.so.2` expect it: the H.264 and HEVC decode buffers this backend fills in, the HEVC *encode* records a
//! follow-up session will fill in, the image structures it reads back through, and the entry points it calls. Written
//! from the public header; sizes, field offsets and the compiler's own padding are checked against a C compiler's by
//! `abi_tests.rs`.
//!
//! The structures are plain data and compile on every target, so the code that fills them in
//! (`vaapi::h264`) is tested everywhere; only `vaapi::va` loads the library (Linux).
//!
//! FFI module (docs/adr/0001-platform-ffi.md): plain `repr(C)` data, no logic.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals, dead_code)]

use std::ffi::{c_char, c_int, c_uint, c_void};

pub type VADisplay = *mut c_void;
pub type VAStatus = c_int;
pub type VAGenericID = c_uint;
pub type VAConfigID = VAGenericID;
pub type VAContextID = VAGenericID;
pub type VASurfaceID = VAGenericID;
pub type VABufferID = VAGenericID;
pub type VAImageID = VAGenericID;
/// `VAProfile` (a C enum: `int`).
pub type VAProfile = c_int;
/// `VAEntrypoint` (a C enum: `int`).
pub type VAEntrypoint = c_int;
/// `VABufferType` (a C enum: `int`).
pub type VABufferType = c_int;
/// `VAConfigAttribType` (a C enum: `int`).
pub type VAConfigAttribType = c_int;
pub type VAMessageCallback = Option<unsafe extern "C" fn(user_context: *mut c_void, message: *const c_char)>;

pub const VA_STATUS_SUCCESS: VAStatus = 0;
pub const VA_STATUS_ERROR_UNSUPPORTED_PROFILE: VAStatus = 0x0c;
pub const VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT: VAStatus = 0x0d;
pub const VA_STATUS_ERROR_UNSUPPORTED_RT_FORMAT: VAStatus = 0x0e;
pub const VA_STATUS_ERROR_RESOLUTION_NOT_SUPPORTED: VAStatus = 0x13;
pub const VA_STATUS_ERROR_DECODING_ERROR: VAStatus = 0x17;

pub const VA_INVALID_ID: VAGenericID = 0xffff_ffff;
pub const VA_INVALID_SURFACE: VASurfaceID = VA_INVALID_ID;

pub const VAProfileH264Main: VAProfile = 6;
pub const VAProfileH264High: VAProfile = 7;
pub const VAProfileH264ConstrainedBaseline: VAProfile = 13;
pub const VAProfileHEVCMain: VAProfile = 17;
pub const VAProfileHEVCMain10: VAProfile = 18;
pub const VAEntrypointVLD: VAEntrypoint = 1;

pub const VAConfigAttribRTFormat: VAConfigAttribType = 0;
pub const VA_RT_FORMAT_YUV420: c_uint = 0x0000_0001;
pub const VA_RT_FORMAT_YUV420_10: c_uint = 0x0000_0100;

/// `vaCreateContext` flag: progressive pictures only.
pub const VA_PROGRESSIVE: c_int = 0x1;

pub const VAPictureParameterBufferType: VABufferType = 0;
pub const VAIQMatrixBufferType: VABufferType = 1;
pub const VASliceParameterBufferType: VABufferType = 4;
pub const VASliceDataBufferType: VABufferType = 5;

/// `VASliceParameterBuffer*::slice_data_flag`: the whole slice is in the buffer.
pub const VA_SLICE_DATA_FLAG_ALL: u32 = 0x00;

pub const VA_PICTURE_H264_INVALID: u32 = 0x0000_0001;
pub const VA_PICTURE_H264_SHORT_TERM_REFERENCE: u32 = 0x0000_0008;
pub const VA_PICTURE_H264_LONG_TERM_REFERENCE: u32 = 0x0000_0010;

pub const VA_FOURCC_NV12: u32 = 0x3231_564E;
pub const VA_FOURCC_P010: u32 = 0x3031_3050;

pub const VA_PICTURE_HEVC_INVALID: u32 = 0x0000_0001;
pub const VA_PICTURE_HEVC_LONG_TERM_REFERENCE: u32 = 0x0000_0008;
pub const VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE: u32 = 0x0000_0010;
pub const VA_PICTURE_HEVC_RPS_ST_CURR_AFTER: u32 = 0x0000_0020;
pub const VA_PICTURE_HEVC_RPS_LT_CURR: u32 = 0x0000_0040;

pub const VA_PADDING_LOW: usize = 4;
pub const VA_PADDING_MEDIUM: usize = 8;
pub const VA_PADDING_HIGH: usize = 16;

// ---- HEVC encoding: `va_enc_hevc.h` and the packed-header enum of `va.h`.
//
// Declarations only. Nothing here, or anywhere else in this crate, encodes an HEVC picture yet;
// #1256 adds the H.264 encode session and this branch only supplies the HEVC records a follow-up
// session (see the notes on `VAEncSliceParameterBufferHEVC` below) needs, with their ABI pinned.
// The constants come from `va.h` / `va_enc_hevc.h`; #1256 already declares `VAEntrypointEncSlice`,
// `VA_RC_CBR` / `VA_RC_VBR`, `VAEncCodedBufferType`, `VAEncSequenceParameterBufferType`,
// `VAEncPictureParameterBufferType`, `VAEncSliceParameterBufferType` and
// `VAEncMiscParameterBufferType`, so none of them is repeated here.

/// `VAEncPictureParameterBufferHEVC::last_picture`: the driver appends an `end_of_seq()` NAL unit
/// to this (last) picture, so the coded buffer is finished.
pub const HEVC_LAST_PICTURE_EOSEQ: u8 = 0x01;
/// `..last_picture`: the driver appends an `end_of_stream()` NAL unit to this picture.
pub const HEVC_LAST_PICTURE_EOSTREAM: u8 = 0x02;

/// The `VAEncPackedHeaderType` of `va.h`, the type a packed-header buffer's parameter names.
pub const VAEncPackedHeaderSequence: u32 = 1;
pub const VAEncPackedHeaderPicture: u32 = 2;
pub const VAEncPackedHeaderSlice: u32 = 3;

/// `VAEncPackedHeaderTypeHEVC` (a C enum). A driver is told *which* of the two sequence headers a
/// `VAEncPackedHeaderSequence` buffer holds by the `nal_unit_type` inside it (32 for the VPS, 33 for
/// the SPS, per `va_enc_hevc.h`), not by this number, which is the same for both.
pub type VAEncPackedHeaderTypeHEVC = c_uint;

/// Packed video parameter set (`nal_unit_type` 32).
pub const VAEncPackedHeaderHEVC_VPS: VAEncPackedHeaderTypeHEVC = VAEncPackedHeaderSequence;
/// Packed sequence parameter set (`nal_unit_type` 33).
pub const VAEncPackedHeaderHEVC_SPS: VAEncPackedHeaderTypeHEVC = VAEncPackedHeaderSequence;
/// Packed picture parameter set (`nal_unit_type` 34).
pub const VAEncPackedHeaderHEVC_PPS: VAEncPackedHeaderTypeHEVC = VAEncPackedHeaderPicture;
/// Packed slice segment header: the `slice_header()` syntax element only, no `slice_data()` and no
/// `rbsp_slice_trailing_bits()`.
pub const VAEncPackedHeaderHEVC_Slice: VAEncPackedHeaderTypeHEVC = VAEncPackedHeaderSlice;
/// Packed SEI. Deprecated by the header (`va_deprecated_enum`): pass the bytes as
/// `VAEncPackedHeaderRawData` instead. `0x80000001` does not fit `int`, so the enum's compatible
/// type is `unsigned int` and so is this.
pub const VAEncPackedHeaderHEVC_SEI: VAEncPackedHeaderTypeHEVC = 0x8000_0001;

// #1256 declares buffer types 21 to 24 and 27 but not 25 and 26, the pair a session sends a packed
// header in: a `VAEncPackedHeaderParameterBuffer` of the `VAEncPackedHeaderTypeHEVC` above, and the
// header bytes themselves.
pub const VAEncPackedHeaderParameterBufferType: VABufferType = 25;
pub const VAEncPackedHeaderDataBufferType: VABufferType = 26;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VAConfigAttrib {
    pub type_: VAConfigAttribType,
    pub value: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VAPictureH264 {
    pub picture_id: VASurfaceID,
    pub frame_idx: u32,
    pub flags: u32,
    pub TopFieldOrderCnt: i32,
    pub BottomFieldOrderCnt: i32,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

impl VAPictureH264 {
    /// An unused entry (`VA_INVALID_SURFACE`, `VA_PICTURE_H264_INVALID`).
    pub const INVALID: Self = Self {
        picture_id: VA_INVALID_SURFACE,
        frame_idx: 0,
        flags: VA_PICTURE_H264_INVALID,
        TopFieldOrderCnt: 0,
        BottomFieldOrderCnt: 0,
        va_reserved: [0; VA_PADDING_LOW],
    };
}

/// `VAPictureParameterBufferH264`. `seq_fields` and `pic_fields` are C bit-field unions, stored
/// here as their `value` word (bits allocated from the least significant end, as GCC and Clang do
/// on the little-endian targets libva runs on).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VAPictureParameterBufferH264 {
    pub CurrPic: VAPictureH264,
    pub ReferenceFrames: [VAPictureH264; 16],
    pub picture_width_in_mbs_minus1: u16,
    pub picture_height_in_mbs_minus1: u16,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub num_ref_frames: u8,
    pub seq_fields: u32,
    pub num_slice_groups_minus1: u8,
    pub slice_group_map_type: u8,
    pub slice_group_change_rate_minus1: u16,
    pub pic_init_qp_minus26: i8,
    pub pic_init_qs_minus26: i8,
    pub chroma_qp_index_offset: i8,
    pub second_chroma_qp_index_offset: i8,
    pub pic_fields: u32,
    pub frame_num: u16,
    pub va_reserved: [u32; VA_PADDING_MEDIUM],
}

/// `seq_fields` bit positions (`chroma_format_idc` 2 bits, the log2 fields 4, `pic_order_cnt_type` 2).
pub mod seq_bits {
    pub const CHROMA_FORMAT_IDC: u32 = 0;
    pub const RESIDUAL_COLOUR_TRANSFORM: u32 = 2;
    pub const GAPS_IN_FRAME_NUM_ALLOWED: u32 = 3;
    pub const FRAME_MBS_ONLY: u32 = 4;
    pub const MB_ADAPTIVE_FRAME_FIELD: u32 = 5;
    pub const DIRECT_8X8_INFERENCE: u32 = 6;
    pub const MIN_LUMA_BI_PRED_SIZE_8X8: u32 = 7;
    pub const LOG2_MAX_FRAME_NUM_MINUS4: u32 = 8;
    pub const PIC_ORDER_CNT_TYPE: u32 = 12;
    pub const LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4: u32 = 14;
    pub const DELTA_PIC_ORDER_ALWAYS_ZERO: u32 = 18;
}

/// `pic_fields` bit positions (`weighted_bipred_idc` 2 bits).
pub mod pic_bits {
    pub const ENTROPY_CODING_MODE: u32 = 0;
    pub const WEIGHTED_PRED: u32 = 1;
    pub const WEIGHTED_BIPRED_IDC: u32 = 2;
    pub const TRANSFORM_8X8_MODE: u32 = 4;
    pub const FIELD_PIC: u32 = 5;
    pub const CONSTRAINED_INTRA_PRED: u32 = 6;
    pub const PIC_ORDER_PRESENT: u32 = 7;
    pub const DEBLOCKING_FILTER_CONTROL_PRESENT: u32 = 8;
    pub const REDUNDANT_PIC_CNT_PRESENT: u32 = 9;
    pub const REFERENCE_PIC: u32 = 10;
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VAIQMatrixBufferH264 {
    /// 4x4 scaling lists in raster scan order: Intra Y, Cb, Cr, Inter Y, Cb, Cr.
    pub ScalingList4x4: [[u8; 16]; 6],
    /// 8x8 scaling lists in raster scan order: Intra Y, Inter Y.
    pub ScalingList8x8: [[u8; 64]; 2],
    pub va_reserved: [u32; VA_PADDING_LOW],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VASliceParameterBufferH264 {
    pub slice_data_size: u32,
    pub slice_data_offset: u32,
    pub slice_data_flag: u32,
    /// Bits from the start of the NAL unit (its header byte included) to `slice_data()`, counted
    /// without emulation prevention bytes; the data buffer itself keeps them.
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
    pub RefPicList0: [VAPictureH264; 32],
    pub RefPicList1: [VAPictureH264; 32],
    pub luma_log2_weight_denom: u8,
    pub chroma_log2_weight_denom: u8,
    pub luma_weight_l0_flag: u8,
    pub luma_weight_l0: [i16; 32],
    pub luma_offset_l0: [i16; 32],
    pub chroma_weight_l0_flag: u8,
    pub chroma_weight_l0: [[i16; 2]; 32],
    pub chroma_offset_l0: [[i16; 2]; 32],
    pub luma_weight_l1_flag: u8,
    pub luma_weight_l1: [i16; 32],
    pub luma_offset_l1: [i16; 32],
    pub chroma_weight_l1_flag: u8,
    pub chroma_weight_l1: [[i16; 2]; 32],
    pub chroma_offset_l1: [[i16; 2]; 32],
    pub va_reserved: [u32; VA_PADDING_LOW],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VAPictureHEVC {
    pub picture_id: VASurfaceID,
    pub pic_order_cnt: i32,
    pub flags: u32,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

impl VAPictureHEVC {
    /// An unused entry (`VA_INVALID_SURFACE`, `VA_PICTURE_HEVC_INVALID`).
    pub const INVALID: Self = Self { picture_id: VA_INVALID_SURFACE, pic_order_cnt: 0, flags: VA_PICTURE_HEVC_INVALID, va_reserved: [0; VA_PADDING_LOW] };
}

/// `VAPictureParameterBufferHEVC`; `pic_fields` and `slice_parsing_fields` are C bit-field unions
/// stored as their `value` word (see [`VAPictureParameterBufferH264`]).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VAPictureParameterBufferHEVC {
    pub CurrPic: VAPictureHEVC,
    pub ReferenceFrames: [VAPictureHEVC; 15],
    pub pic_width_in_luma_samples: u16,
    pub pic_height_in_luma_samples: u16,
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
    pub slice_parsing_fields: u32,
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    pub num_short_term_ref_pic_sets: u8,
    pub num_long_term_ref_pic_sps: u8,
    pub num_ref_idx_l0_default_active_minus1: u8,
    pub num_ref_idx_l1_default_active_minus1: u8,
    pub pps_beta_offset_div2: i8,
    pub pps_tc_offset_div2: i8,
    pub num_extra_slice_header_bits: u8,
    /// Bits of `short_term_ref_pic_set()` in the slice header (0 when the SPS's set is used),
    /// counted without emulation prevention bytes.
    pub st_rps_bits: u32,
    pub va_reserved: [u32; VA_PADDING_MEDIUM],
}

/// `VAPictureParameterBufferHEVC::pic_fields` bit positions (`chroma_format_idc` 2 bits).
pub mod hevc_pic_bits {
    pub const CHROMA_FORMAT_IDC: u32 = 0;
    pub const SEPARATE_COLOUR_PLANE: u32 = 2;
    pub const PCM_ENABLED: u32 = 3;
    pub const SCALING_LIST_ENABLED: u32 = 4;
    pub const TRANSFORM_SKIP_ENABLED: u32 = 5;
    pub const AMP_ENABLED: u32 = 6;
    pub const STRONG_INTRA_SMOOTHING_ENABLED: u32 = 7;
    pub const SIGN_DATA_HIDING_ENABLED: u32 = 8;
    pub const CONSTRAINED_INTRA_PRED: u32 = 9;
    pub const CU_QP_DELTA_ENABLED: u32 = 10;
    pub const WEIGHTED_PRED: u32 = 11;
    pub const WEIGHTED_BIPRED: u32 = 12;
    pub const TRANSQUANT_BYPASS_ENABLED: u32 = 13;
    pub const TILES_ENABLED: u32 = 14;
    pub const ENTROPY_CODING_SYNC_ENABLED: u32 = 15;
    pub const PPS_LOOP_FILTER_ACROSS_SLICES_ENABLED: u32 = 16;
    pub const LOOP_FILTER_ACROSS_TILES_ENABLED: u32 = 17;
    pub const PCM_LOOP_FILTER_DISABLED: u32 = 18;
    pub const NO_PIC_REORDERING: u32 = 19;
    pub const NO_BI_PRED: u32 = 20;
}

/// `VAPictureParameterBufferHEVC::slice_parsing_fields` bit positions.
pub mod hevc_slice_parsing_bits {
    pub const LISTS_MODIFICATION_PRESENT: u32 = 0;
    pub const LONG_TERM_REF_PICS_PRESENT: u32 = 1;
    pub const SPS_TEMPORAL_MVP_ENABLED: u32 = 2;
    pub const CABAC_INIT_PRESENT: u32 = 3;
    pub const OUTPUT_FLAG_PRESENT: u32 = 4;
    pub const DEPENDENT_SLICE_SEGMENTS_ENABLED: u32 = 5;
    pub const PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT: u32 = 6;
    pub const SAMPLE_ADAPTIVE_OFFSET_ENABLED: u32 = 7;
    pub const DEBLOCKING_FILTER_OVERRIDE_ENABLED: u32 = 8;
    pub const PPS_DISABLE_DEBLOCKING_FILTER: u32 = 9;
    pub const SLICE_SEGMENT_HEADER_EXTENSION_PRESENT: u32 = 10;
    pub const RAP_PIC: u32 = 11;
    pub const IDR_PIC: u32 = 12;
    pub const INTRA_PIC: u32 = 13;
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VASliceParameterBufferHEVC {
    pub slice_data_size: u32,
    pub slice_data_offset: u32,
    pub slice_data_flag: u32,
    /// Bytes from the start of the NAL unit (its two-byte header included) to
    /// `slice_segment_data()`, counted without emulation prevention bytes; the data buffer itself
    /// keeps them.
    pub slice_data_byte_offset: u32,
    pub slice_segment_address: u32,
    /// Indices into `ReferenceFrames` (0xFF: none).
    pub RefPicList: [[u8; 15]; 2],
    pub LongSliceFlags: u32,
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
    pub ChromaOffsetL0: [[i8; 2]; 15],
    pub delta_luma_weight_l1: [i8; 15],
    pub luma_offset_l1: [i8; 15],
    pub delta_chroma_weight_l1: [[i8; 2]; 15],
    pub ChromaOffsetL1: [[i8; 2]; 15],
    pub five_minus_max_num_merge_cand: u8,
    pub num_entry_point_offsets: u16,
    pub entry_offset_to_subset_array: u16,
    /// Emulation prevention bytes in the slice segment header.
    pub slice_data_num_emu_prevn_bytes: u16,
    pub va_reserved: [u32; VA_PADDING_LOW - 2],
}

/// `VASliceParameterBufferHEVC::LongSliceFlags` bit positions (`slice_type` and `color_plane_id` 2 bits).
pub mod hevc_slice_bits {
    pub const LAST_SLICE_OF_PIC: u32 = 0;
    pub const DEPENDENT_SLICE_SEGMENT: u32 = 1;
    pub const SLICE_TYPE: u32 = 2;
    pub const COLOR_PLANE_ID: u32 = 4;
    pub const SLICE_SAO_LUMA: u32 = 6;
    pub const SLICE_SAO_CHROMA: u32 = 7;
    pub const MVD_L1_ZERO: u32 = 8;
    pub const CABAC_INIT: u32 = 9;
    pub const SLICE_TEMPORAL_MVP_ENABLED: u32 = 10;
    pub const SLICE_DEBLOCKING_FILTER_DISABLED: u32 = 11;
    pub const COLLOCATED_FROM_L0: u32 = 12;
    pub const SLICE_LOOP_FILTER_ACROSS_SLICES_ENABLED: u32 = 13;
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VAIQMatrixBufferHEVC {
    /// `ScalingList[sizeId][matrixId][j]` of the spec for sizeId 0..3, raster order.
    pub ScalingList4x4: [[u8; 16]; 6],
    pub ScalingList8x8: [[u8; 64]; 6],
    pub ScalingList16x16: [[u8; 64]; 6],
    pub ScalingList32x32: [[u8; 64]; 2],
    pub ScalingListDC16x16: [u8; 6],
    pub ScalingListDC32x32: [u8; 2],
    pub va_reserved: [u32; VA_PADDING_LOW],
}

// ---- HEVC encoding (`va_enc_hevc.h`)
//
// Declarations, not an encoder: no session, no `vaRenderPicture`, no buffer here is written by
// anything in this crate. Sizes, field offsets and the compiler's padding are pinned against a C
// compiler's view of the header by `abi_tests.rs`, so a future session can be written on top of
// these records without measuring anything again.

/// `VAEncSequenceParameterBufferHEVC`, the `seq_parameter_set_data()` the driver writes its SPS from.
/// `seq_fields`, `vui_fields` and `scc_fields` are C bit-field unions, stored here as their `value`
/// word (bits allocated from the least significant end, as GCC and Clang do on the little-endian
/// targets libva runs on; see [`VAPictureParameterBufferH264`]).
///
/// Four gaps of compiler padding (1, 2, 3 and 3 bytes) sit inside it, after `general_tier_flag`,
/// `max_transform_hierarchy_depth_intra`, `vui_parameters_present_flag` and `aspect_ratio_idc`: they
/// make no field's size any different, so `abi_tests.rs` pins each of them as a measured gap.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VAEncSequenceParameterBufferHEVC {
    pub general_profile_idc: u8,
    pub general_level_idc: u8,
    pub general_tier_flag: u8,
    pub intra_period: u32,
    pub intra_idr_period: u32,
    pub ip_period: u32,
    /// The initial bitrate, when rate control is CBR or VBR.
    pub bits_per_second: u32,
    pub pic_width_in_luma_samples: u16,
    pub pic_height_in_luma_samples: u16,
    pub seq_fields: u32,
    pub log2_min_luma_coding_block_size_minus3: u8,
    pub log2_diff_max_min_luma_coding_block_size: u8,
    pub log2_min_transform_block_size_minus2: u8,
    pub log2_diff_max_min_transform_block_size: u8,
    pub max_transform_hierarchy_depth_inter: u8,
    pub max_transform_hierarchy_depth_intra: u8,
    pub pcm_sample_bit_depth_luma_minus1: u32,
    pub pcm_sample_bit_depth_chroma_minus1: u32,
    pub log2_min_pcm_luma_coding_block_size_minus3: u32,
    pub log2_max_pcm_luma_coding_block_size_minus3: u32,
    pub vui_parameters_present_flag: u8,
    pub vui_fields: u32,
    pub aspect_ratio_idc: u8,
    pub sar_width: u32,
    pub sar_height: u32,
    pub vui_num_units_in_tick: u32,
    pub vui_time_scale: u32,
    pub min_spatial_segmentation_idc: u16,
    pub max_bytes_per_pic_denom: u8,
    pub max_bits_per_min_cu_denom: u8,
    /// Screen-content coding (palette mode): unused in Main / Main 10.
    pub scc_fields: u32,
    pub va_reserved: [u32; VA_PADDING_MEDIUM - 1],
}

/// `VAEncSequenceParameterBufferHEVC::seq_fields` bit positions (`chroma_format_idc` 2 bits).
pub mod enc_seq_bits {
    pub const CHROMA_FORMAT_IDC: u32 = 0;
    pub const SEPARATE_COLOUR_PLANE_FLAG: u32 = 2;
    pub const BIT_DEPTH_LUMA_MINUS8: u32 = 3;
    pub const BIT_DEPTH_CHROMA_MINUS8: u32 = 6;
    pub const SCALING_LIST_ENABLED_FLAG: u32 = 9;
    pub const STRONG_INTRA_SMOOTHING_ENABLED_FLAG: u32 = 10;
    pub const AMP_ENABLED_FLAG: u32 = 11;
    pub const SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG: u32 = 12;
    pub const PCM_ENABLED_FLAG: u32 = 13;
    pub const PCM_LOOP_FILTER_DISABLED_FLAG: u32 = 14;
    pub const SPS_TEMPORAL_MVP_ENABLED_FLAG: u32 = 15;
    /// 0 = a normal sequence, 1 = low delay: no random access B, coding types I, P and LDB only.
    pub const LOW_DELAY_SEQ: u32 = 16;
    /// Dyadic hierarchical GOP structure (then the encoder modulates QP); 0 is the flat default.
    pub const HIERACHICAL_FLAG: u32 = 17;
}

/// `VAEncSequenceParameterBufferHEVC::vui_fields` bit positions (the two motion-vector ranges 5 bits).
pub mod enc_vui_bits {
    pub const ASPECT_RATIO_INFO_PRESENT_FLAG: u32 = 0;
    pub const NEUTRAL_CHROMA_INDICATION_FLAG: u32 = 1;
    pub const FIELD_SEQ_FLAG: u32 = 2;
    pub const VUI_TIMING_INFO_PRESENT_FLAG: u32 = 3;
    pub const BITSTREAM_RESTRICTION_FLAG: u32 = 4;
    pub const TILES_FIXED_STRUCTURE_FLAG: u32 = 5;
    pub const MOTION_VECTORS_OVER_PIC_BOUNDARIES_FLAG: u32 = 6;
    pub const RESTRICTED_REF_PIC_LISTS_FLAG: u32 = 7;
    pub const LOG2_MAX_MV_LENGTH_HORIZONTAL: u32 = 8;
    pub const LOG2_MAX_MV_LENGTH_VERTICAL: u32 = 13;
}

/// `VAEncSequenceParameterBufferHEVC::scc_fields` bit positions (a 32-bit union here; the picture
/// record's is 16-bit).
pub mod enc_seq_scc_bits {
    pub const PALETTE_MODE_ENABLED_FLAG: u32 = 0;
}

/// `VAEncPictureParameterBufferHEVC`, the `pic_parameter_set_rbsp()` the driver writes its PPS from
/// plus the DPB and the buffer the coded picture is written to. `pic_fields` and `scc_fields` are C
/// bit-field unions stored as their `value` word (`scc_fields` is a **16-bit** union here, against a
/// 32-bit one in the sequence record; see `abi_tests.rs`). Two bytes of compiler padding sit between
/// `nal_unit_type` and `pic_fields`.
///
/// Differences from the decode [`VAPictureParameterBufferHEVC`], which the same names do not carry:
/// the current picture is `decoded_curr_pic` (the DPB slot the reconstruction goes to), the QP is
/// `pic_init_qp` where the decode record stores `init_qp_minus26`, the collocated reference is
/// `collocated_ref_pic_index` (an index into `reference_frames`, not the slice's `collocated_ref_idx`
/// syntax element), and `column_width_minus1` / `row_height_minus1` are bytes, not `u16`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VAEncPictureParameterBufferHEVC {
    /// The reconstructed picture; a scratch surface the application hands the driver.
    pub decoded_curr_pic: VAPictureHEVC,
    /// The DPB, in the order the picture's reference indices and `collocated_ref_pic_index` count.
    pub reference_frames: [VAPictureHEVC; 15],
    /// The coded bitstream buffer (of type `VAEncCodedBufferType`).
    pub coded_buf: VABufferID,
    /// `0xFF` when the slice's temporal MVP is off, else an index into `reference_frames`.
    pub collocated_ref_pic_index: u8,
    /// [`HEVC_LAST_PICTURE_EOSEQ`] and / or [`HEVC_LAST_PICTURE_EOSTREAM`], or 0.
    pub last_picture: u8,
    /// `init_qp_minus26 + 26`.
    pub pic_init_qp: u8,
    pub diff_cu_qp_delta_depth: u8,
    pub pps_cb_qp_offset: i8,
    pub pps_cr_qp_offset: i8,
    pub num_tile_columns_minus1: u8,
    pub num_tile_rows_minus1: u8,
    pub column_width_minus1: [u8; 19],
    pub row_height_minus1: [u8; 21],
    pub log2_parallel_merge_level_minus2: u8,
    /// The CTU bit-size limit (spec A.3.2); 0 means none is checked.
    pub ctu_max_bitsize_allowed: u8,
    pub num_ref_idx_l0_default_active_minus1: u8,
    pub num_ref_idx_l1_default_active_minus1: u8,
    /// PPS id, used by the driver when it writes new slice headers for slice size control.
    pub slice_pic_parameter_set_id: u8,
    /// NAL unit type, likewise for the driver's own slice headers.
    pub nal_unit_type: u8,
    pub pic_fields: u32,
    /// 0: the level comes from `pic_fields::coding_type`. > 0: `hierarchical_level_plus1 - 1` is it.
    pub hierarchical_level_plus1: u8,
    pub va_byte_reserved: u8,
    /// Screen-content coding (current-picture reference): unused in Main / Main 10. A 16-bit union.
    pub scc_fields: u16,
    pub va_reserved: [u32; VA_PADDING_HIGH - 1],
}

/// `VAEncPictureParameterBufferHEVC::pic_fields` bit positions (`coding_type` 3 bits: I 1, P 2, B 3,
/// hierarchical B1 4 and B2 5).
pub mod enc_pic_bits {
    pub const IDR_PIC_FLAG: u32 = 0;
    pub const CODING_TYPE: u32 = 1;
    pub const REFERENCE_PIC_FLAG: u32 = 4;
    pub const DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG: u32 = 5;
    pub const SIGN_DATA_HIDING_ENABLED_FLAG: u32 = 6;
    pub const CONSTRAINED_INTRA_PRED_FLAG: u32 = 7;
    pub const TRANSFORM_SKIP_ENABLED_FLAG: u32 = 8;
    pub const CU_QP_DELTA_ENABLED_FLAG: u32 = 9;
    pub const WEIGHTED_PRED_FLAG: u32 = 10;
    pub const WEIGHTED_BIPRED_FLAG: u32 = 11;
    pub const TRANSQUANT_BYPASS_ENABLED_FLAG: u32 = 12;
    pub const TILES_ENABLED_FLAG: u32 = 13;
    pub const ENTROPY_CODING_SYNC_ENABLED_FLAG: u32 = 14;
    pub const LOOP_FILTER_ACROSS_TILES_ENABLED_FLAG: u32 = 15;
    pub const PPS_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG: u32 = 16;
    /// Only meaningful with `scaling_list_enabled_flag`; asks for the driver's
    /// `VAQMatrixBufferHEVC` alongside.
    pub const SCALING_LIST_DATA_PRESENT_FLAG: u32 = 17;
    pub const SCREEN_CONTENT_FLAG: u32 = 18;
    /// Let the driver pick the weighted-prediction factors, which then overrides the slice's.
    pub const ENABLE_GPU_WEIGHTED_PREDICTION: u32 = 19;
    pub const NO_OUTPUT_OF_PRIOR_PICS_FLAG: u32 = 20;
}

/// `VAEncPictureParameterBufferHEVC::scc_fields` bit positions (a 16-bit union, unlike the sequence
/// record's 32-bit one).
pub mod enc_pic_scc_bits {
    pub const PPS_CURR_PIC_REF_ENABLED_FLAG: u32 = 0;
}

/// `VAEncSliceParameterBufferHEVC`, the slice's parameters: where it starts, how many CTUs it holds,
/// its reference lists and its `pred_weight_table()`. No hole in it (every field starts where the last
/// one ended).
///
/// The record carries no `slice_data_offset`, so it cannot say where the slice data starts inside the
/// slice data buffer: the application writes the slice segment header there itself (the NAL unit
/// header and `slice_header()`, emulation prevention included) and the driver appends
/// `slice_segment_data()`. That is what #1256's H.264 session already does, and it is the work a
/// HEVC session still has ahead of it. Compared with the decode [`VASliceParameterBufferHEVC`]:
/// `max_num_merge_cand` is `MaxNumMergeCand` itself where the decode record stores
/// `five_minus_max_num_merge_cand`, `slice_fields` adds `num_ref_idx_active_override_flag`, the
/// reference lists are whole [`VAPictureHEVC`] pictures rather than indices, and there is no
/// `slice_data_byte_offset`, `RefPicList` or `slice_data_num_emu_prevn_bytes` to tell the driver
/// about emulation prevention the application inserted.
///
/// `pred_weight_table_bit_offset` and `pred_weight_table_bit_length` are in bits from the start of
/// the packed slice header, and only mean something with
/// [`enc_pic_bits::ENABLE_GPU_WEIGHTED_PREDICTION`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VAEncSliceParameterBufferHEVC {
    pub slice_segment_address: u32,
    pub num_ctu_in_slice: u32,
    pub slice_type: u8,
    pub slice_pic_parameter_set_id: u8,
    pub num_ref_idx_l0_active_minus1: u8,
    pub num_ref_idx_l1_active_minus1: u8,
    /// Reference picture list 0 (P slices), as reconstructed surface ids.
    pub ref_pic_list0: [VAPictureHEVC; 15],
    /// Reference picture list 1 (B slices), likewise.
    pub ref_pic_list1: [VAPictureHEVC; 15],
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
    /// `MaxNumMergeCand` itself, 1 to 5.
    pub max_num_merge_cand: u8,
    pub slice_qp_delta: i8,
    pub slice_cb_qp_offset: i8,
    pub slice_cr_qp_offset: i8,
    pub slice_beta_offset_div2: i8,
    pub slice_tc_offset_div2: i8,
    pub slice_fields: u32,
    pub pred_weight_table_bit_offset: u32,
    pub pred_weight_table_bit_length: u32,
    pub va_reserved: [u32; VA_PADDING_MEDIUM - 2],
}

/// `VAEncSliceParameterBufferHEVC::slice_fields` bit positions (`colour_plane_id` and
/// `slice_deblocking_filter_disabled_flag` 2 bits each). Unlike the decode record's
/// `LongSliceFlags`, the driver does not write this header: the application packs these into the
/// slice header itself.
pub mod enc_slice_bits {
    pub const LAST_SLICE_OF_PIC_FLAG: u32 = 0;
    pub const DEPENDENT_SLICE_SEGMENT_FLAG: u32 = 1;
    pub const COLOUR_PLANE_ID: u32 = 2;
    pub const SLICE_TEMPORAL_MVP_ENABLED_FLAG: u32 = 4;
    pub const SLICE_SAO_LUMA_FLAG: u32 = 5;
    pub const SLICE_SAO_CHROMA_FLAG: u32 = 6;
    /// 0: the picture's `num_ref_idx_l*_default_active_minus1` also apply to this slice.
    pub const NUM_REF_IDX_ACTIVE_OVERRIDE_FLAG: u32 = 7;
    pub const MVD_L1_ZERO_FLAG: u32 = 8;
    pub const CABAC_INIT_FLAG: u32 = 9;
    pub const SLICE_DEBLOCKING_FILTER_DISABLED_FLAG: u32 = 10;
    pub const SLICE_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG: u32 = 12;
    pub const COLLOCATED_FROM_L0_FLAG: u32 = 13;
}

// What an HEVC encode *session* still has to bring, measured against the same headers so it does not
// have to be measured again (nothing here is declared: it belongs to the session, not to the records).
//
// * `VAEncPackedHeaderParameterBuffer` (`va.h`): `type` at 0, `bit_length` at 4, `has_emulation_bytes`
//   at 8, `va_reserved[VA_PADDING_LOW]` at 12, 28 bytes in all. `has_emulation_bytes` is new in
//   1.24 and absent in the 1.23 headers the decode records were measured against; a record declaring
//   it has to be measured against whichever headers that PR pins against.
// * The packed-header flags a `vaCreateConfig` asks for (`VAConfigAttribEncPackedHeaders`, 10):
//   NONE 0, SEQUENCE 1, PICTURE 2, SLICE 4, MISC 8 (deprecated), RAW_DATA 16.
// * `VAQMatrixBufferHEVC` (`va_enc_hevc.h`), the encode IQ matrix: 1016 bytes, and its
//   `scaling_lists_4x4` / `_8x8` / `_16x16` / `_32x32` / `scaling_list_dc_16x16` / `_32x32` /
//   `va_reserved` sit at 0 / 96 / 480 / 864 / 992 / 998 / 1000 — the same layout, to the byte, as the
//   decode [`VAIQMatrixBufferHEVC`] above, so that record is already the right bytes under another
//   name. Only needed with `enc_seq_bits::SCALING_LIST_ENABLED_FLAG`.
// * `VAConfigAttribValEncHEVCFeatures` and `VAConfigAttribValEncHEVCBlockSizes`, both 16 two-bit
//   fields (plus 2 and 10 reserved bits) of a `u32` word read from `VAConfigAttribEncHEVCFeatures` /
//   `VAConfigAttribEncHEVCBlockSizes` to learn what the driver's encoder can do.
// * The slice segment header itself: the driver writes none. See [`VAEncSliceParameterBufferHEVC`].

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VAImageFormat {
    pub fourcc: u32,
    pub byte_order: u32,
    pub bits_per_pixel: u32,
    pub depth: u32,
    pub red_mask: u32,
    pub green_mask: u32,
    pub blue_mask: u32,
    pub alpha_mask: u32,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VAImage {
    pub image_id: VAImageID,
    pub format: VAImageFormat,
    pub buf: VABufferID,
    pub width: u16,
    pub height: u16,
    pub data_size: u32,
    pub num_planes: u32,
    pub pitches: [u32; 3],
    pub offsets: [u32; 3],
    pub num_palette_entries: i32,
    pub entry_bytes: i32,
    pub component_order: [i8; 4],
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// The `libva.so.2` entry points used (all `VAStatus`-returning unless noted).
pub type vaInitialize = unsafe extern "C" fn(dpy: VADisplay, major: *mut c_int, minor: *mut c_int) -> VAStatus;
pub type vaTerminate = unsafe extern "C" fn(dpy: VADisplay) -> VAStatus;
pub type vaErrorStr = unsafe extern "C" fn(status: VAStatus) -> *const c_char;
pub type vaQueryVendorString = unsafe extern "C" fn(dpy: VADisplay) -> *const c_char;
pub type vaSetInfoCallback = unsafe extern "C" fn(dpy: VADisplay, callback: VAMessageCallback, user_context: *mut c_void) -> VAMessageCallback;
pub type vaSetErrorCallback = unsafe extern "C" fn(dpy: VADisplay, callback: VAMessageCallback, user_context: *mut c_void) -> VAMessageCallback;
pub type vaGetConfigAttributes =
    unsafe extern "C" fn(dpy: VADisplay, profile: VAProfile, entrypoint: VAEntrypoint, attrib_list: *mut VAConfigAttrib, num_attribs: c_int) -> VAStatus;
pub type vaCreateConfig = unsafe extern "C" fn(
    dpy: VADisplay,
    profile: VAProfile,
    entrypoint: VAEntrypoint,
    attrib_list: *mut VAConfigAttrib,
    num_attribs: c_int,
    config_id: *mut VAConfigID,
) -> VAStatus;
pub type vaDestroyConfig = unsafe extern "C" fn(dpy: VADisplay, config_id: VAConfigID) -> VAStatus;
/// The last two arguments are `VASurfaceAttrib *attrib_list, unsigned int num_attribs`; this
/// backend passes none, so the attribute structure is not declared.
pub type vaCreateSurfaces = unsafe extern "C" fn(
    dpy: VADisplay,
    format: c_uint,
    width: c_uint,
    height: c_uint,
    surfaces: *mut VASurfaceID,
    num_surfaces: c_uint,
    attrib_list: *mut c_void,
    num_attribs: c_uint,
) -> VAStatus;
pub type vaDestroySurfaces = unsafe extern "C" fn(dpy: VADisplay, surfaces: *mut VASurfaceID, num_surfaces: c_int) -> VAStatus;
pub type vaCreateContext = unsafe extern "C" fn(
    dpy: VADisplay,
    config_id: VAConfigID,
    picture_width: c_int,
    picture_height: c_int,
    flag: c_int,
    render_targets: *mut VASurfaceID,
    num_render_targets: c_int,
    context: *mut VAContextID,
) -> VAStatus;
pub type vaDestroyContext = unsafe extern "C" fn(dpy: VADisplay, context: VAContextID) -> VAStatus;
pub type vaCreateBuffer = unsafe extern "C" fn(
    dpy: VADisplay,
    context: VAContextID,
    type_: VABufferType,
    size: c_uint,
    num_elements: c_uint,
    data: *mut c_void,
    buf_id: *mut VABufferID,
) -> VAStatus;
pub type vaDestroyBuffer = unsafe extern "C" fn(dpy: VADisplay, buffer_id: VABufferID) -> VAStatus;
pub type vaBeginPicture = unsafe extern "C" fn(dpy: VADisplay, context: VAContextID, render_target: VASurfaceID) -> VAStatus;
pub type vaRenderPicture = unsafe extern "C" fn(dpy: VADisplay, context: VAContextID, buffers: *mut VABufferID, num_buffers: c_int) -> VAStatus;
pub type vaEndPicture = unsafe extern "C" fn(dpy: VADisplay, context: VAContextID) -> VAStatus;
pub type vaSyncSurface = unsafe extern "C" fn(dpy: VADisplay, render_target: VASurfaceID) -> VAStatus;
pub type vaCreateImage = unsafe extern "C" fn(dpy: VADisplay, format: *mut VAImageFormat, width: c_int, height: c_int, image: *mut VAImage) -> VAStatus;
pub type vaDestroyImage = unsafe extern "C" fn(dpy: VADisplay, image: VAImageID) -> VAStatus;
pub type vaGetImage =
    unsafe extern "C" fn(dpy: VADisplay, surface: VASurfaceID, x: c_int, y: c_int, width: c_uint, height: c_uint, image: VAImageID) -> VAStatus;
pub type vaPutImage = unsafe extern "C" fn(
    dpy: VADisplay,
    surface: VASurfaceID,
    image: VAImageID,
    src_x: c_int,
    src_y: c_int,
    src_width: c_uint,
    src_height: c_uint,
    dest_x: c_int,
    dest_y: c_int,
    dest_width: c_uint,
    dest_height: c_uint,
) -> VAStatus;
pub type vaMapBuffer = unsafe extern "C" fn(dpy: VADisplay, buf_id: VABufferID, pbuf: *mut *mut c_void) -> VAStatus;
pub type vaUnmapBuffer = unsafe extern "C" fn(dpy: VADisplay, buf_id: VABufferID) -> VAStatus;
/// `libva-drm.so.2`: the display of a DRM device (render node) file descriptor.
pub type vaGetDisplayDRM = unsafe extern "C" fn(fd: c_int) -> VADisplay;
