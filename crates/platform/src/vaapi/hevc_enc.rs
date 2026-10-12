//! HEVC **encode** records: what to hand a VA-API driver so it encodes an HEVC/H.265 picture.
//!
//! # Why this file exists and is empty of behaviour
//!
//! libva declares three HEVC encode records in `va_enc_hevc.h` and this is where they are
//! declared here. **Nothing submits them yet.** The HEVC submission path is a later task and
//! this is its foundation: a record that is declared, measured against the installed header and
//! pinned is a record the next task can build a submission out of, where a record that is *both*
//! declared and used is a record whose every field is load-bearing the day it is written. The
//! H.264 encode records in [`super::ffi`] are the reference for **shape and conventions** — a
//! record here is thin, named, and its holes are named — not for values.
//!
//! So what is deliberately *absent* is the point: there is no `encode()` here, no rate control, no
//! submission and no settings wiring. Rate control is the dependency — a wrong QP or frame type is
//! a silently wrong picture, the same failure class as the H.264 weight bug — and it is being
//! fixed in its own change before this one is built on.
//!
//! # Why a separate file rather than more of `ffi.rs`
//!
//! Zero overlap with the H.264 encode work is deliberate. `ffi.rs` is the FFI module (the only
//! place `unsafe` lives) and is being edited concurrently for rate control; a second writer on one
//! file is how a tree gets corrupted. These records are **safe, plain data** with no `unsafe`
//! anywhere in this file and none needed, so they do not belong in the FFI module on their merits
//! either — see "the `unsafe` posture" below.
//!
//! # What is here, and what is measured
//!
//! Three records, every one measured with a C compiler against `/usr/include/va/va_enc_hevc.h`
//! rather than transcribed from it:
//!
//! | record | measured `sizeof` | holes |
//! |---|---|---|
//! | [`VaEncSequenceParameterBufferHevc`] | 116 | 4 |
//! | [`VaEncPictureParameterBufferHevc`] | 576 | 1 |
//! | [`VaEncSliceParameterBufferHevc`] | 1076 | 0 |
//!
//! **The five holes are named `padding_*` fields of the C-measured width.** `sizeof` is blind to a
//! hole — padding changes no size — which is how `VASegmentParameterVP9` agreed with a C compiler
//! on every size and every offset for five commits and handed the driver two uninitialised bytes
//! per record. Each hole here is declared as a field, and [`hevc_abi`] asks the C compiler for its
//! width **as a gap** (the difference between two `offsetof`s minus the member's own width),
//! because C gives padding no name and no `sizeof` can see one.
//!
//! The five unions in these records are declared as their `value` member — a `u32`, or a `u16` for
//! the picture record's `scc_fields` — for the same reason H.264's are: a union has one `offsetof`,
//! so a bit packed one position too high agrees with C on the record's size, on every field offset
//! and on every hole, and produces **a picture** rather than an error. [`hevc_abi`] therefore
//! measures all **88** bit positions by *running* a C program that sets one member to all ones and
//! reads `__builtin_ctz` / `31 - __builtin_clz` back, rather than trusting declaration order.
//!
//! # The `unsafe` posture
//!
//! **This file needs none and has none.** libva is LGPL and is never linked — its symbols are
//! resolved at run time out of the system's own library, and that is what ADR 0001 rests on
//! ([docs/adr/0001-platform-ffi.md]). Declaring a record needs no symbol and no `extern "C"` call:
//! it is `#[repr(C)]` data. There is not a single `unsafe` block here, so there is no
//! `#[allow(unsafe_code)]` here either — a file that carried the allowance without needing it
//! would be a file whose next edit could use it.
//!
//! # Open questions this file records rather than answers
//!
//! Three things the header does not state, checked and recorded instead of guessed:
//!
//! 1. **HEVC's `slice_type` encoding is not documented.** `va_enc_hevc.h:776-779` says only
//!    "Slice type. Corresponds to HEVC syntax element of the same name." — it does not say what
//!    the numbers are, and HEVC's `slice_type` is *not* H.264's offset-by-five encoding
//!    (`ffi::SLICE_TYPE_*`, which the H.264 header also does not state). `grep -rn
//!    VA_ENC_SLICE_TYPE /usr/include/va/` returns nothing: no enum exists to ask. So no
//!    `SLICE_TYPE_*` constant is declared for HEVC, and [`VaEncSliceParameterBufferHevc::slice_type`]
//!    says so. Guessing here is exactly the "confident claim nothing checks" this branch's
//!    documented failure mode is.
//! 2. **There are no `VAEncMiscParameter*` HEVC variants.** Every `VAEncMiscParameterBuffer*` is in
//!    `va.h`, not `va_enc_hevc.h`, and all three FilmCraft needs (frame rate, rate control,
//!    quantization) are already declared in [`super::ffi`]. Nothing to add, and nothing to guess.
//! 3. **`VAEncPictureAttribute*` / `VAEncSliceAttribute*` do not exist.** `grep -rn "PictureAttribute\|SliceAttribute"
//!    /usr/include/va/` returns nothing across all 37 headers of libva 1.24.0. The HEVC-specific
//!    *config* attributes that do exist are [`CONFIG_ATTRIB_HEVC_FEATURES`] (50) and
//!    [`CONFIG_ATTRIB_HEVC_BLOCK_SIZES`] (51), and both are declared and pinned here. If that
//!    pair of enum families is meant to come from a later libva than this machine has, that is a
//!    "check a newer header" question and not something to invent here.
//!
//! `VAQMatrixBufferHEVC` (`va_enc_hevc.h:908`) is also **not** declared, for a stated reason: it is
//! submitted only when `seq_fields.scaling_list_enabled_flag` is 1 (`va_enc_hevc.h:896-904`), and
//! an encoder that never enables scaling lists never sends it. It is the next record to want when
//! a scaling-list encode is ever asked for.

// Nothing here has a caller until the HEVC submission path lands (the module note above), so the
// dead-code lint has nothing useful to say about it — except inside the tests below, which name
// every item. Kept here rather than on the `mod hevc_enc;` line so this file is the whole change.
#![cfg_attr(not(test), allow(dead_code))]

use super::ffi::{VA_PADDING_MEDIUM, VaPictureHevc};

/// `VABufferID` (`va.h:2042`), written as the `u32` it is.
///
/// libva spells it `typedef VAGenericID VABufferID;` over `typedef uint32_t VAGenericID;`
/// (`va.h:1645`, `:2038`). [`super::ffi`] keeps both aliases to itself — they are private to it, and
/// making them public would mean editing the file the H.264 rate-control change is editing — so the
/// width is spelled here and **pinned against the header's own typedef** in [`hevc_abi`]
/// (`_Static_assert(sizeof(VABufferID) == 4)`), so it cannot drift into another width by being
/// written down twice.
type BufferId = u32;

/// `VA_PADDING_HIGH` (`va.h:362`): 16.
///
/// [`super::ffi`] declares `VA_PADDING_LOW` and `VA_PADDING_MEDIUM`; `VA_PADDING_HIGH` is declared
/// here because only the HEVC *encode* picture record uses it (`va_enc_hevc.h:750`), and adding it
/// next door to the other two would mean editing a file another change is already editing.
const VA_PADDING_HIGH: usize = 16;

/// `VAEncSequenceParameterBufferHEVC` (`va_enc_hevc.h:346`): what to encode — the sequence
/// parameter set's own syntax elements, the periods, the picture geometry and the VUI.
/// **116 bytes**, alignof 4.
///
/// ```text
/// $ cc -w -o hevcenc_measure hevcenc_measure.c && ./hevcenc_measure
/// == VAEncSequenceParameterBufferHEVC sizeof=116 alignof=4
/// ```
///
/// One of these is submitted with **every** picture, not once per sequence, exactly as H.264's is
/// (`ffi::VaEncSequenceParameterBufferH264`).
///
/// **Four holes, all C-measured:** byte 3 (after `general_tier_flag`), 34-35 (after
/// `max_transform_hierarchy_depth_intra`), 53-55 (after `vui_parameters_present_flag`) and 61-63
/// (after `aspect_ratio_idc`). Each is declared as a `padding_*` field and pinned as a gap.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaEncSequenceParameterBufferHevc {
    /// `general_profile_idc`, 1..=2 for Main and Main 10 (`va_enc_hevc.h:347-350`).
    pub general_profile_idc: u8,
    /// `general_level_idc` is **30 times the level number** (`va_enc_hevc.h:351-354`): level 4.0 is
    /// 120, not 4. The same trap as H.264's `level_idc` and the same reason this is a `u8` here.
    pub general_level_idc: u8,
    /// `general_tier_flag`, and **0 is the only value allowed below level 4** (`va_enc_hevc.h:356-359`).
    pub general_tier_flag: u8,
    /// The C layout's hole at byte 3, before `intra_period`.
    pub padding_after_general_tier_flag: [u8; 1],
    /// Period between I frames; 0 for none.
    pub intra_period: u32,
    /// Period between IDR frames; 0 for none.
    pub intra_idr_period: u32,
    /// Period between I/P frames; 0 for none.
    pub ip_period: u32,
    /// The initial bitrate, used only under CBR or VBR (`va_enc_hevc.h:366-377`). Rate control is
    /// a later task; nothing reads this yet.
    pub bits_per_second: u32,
    /// **Pixels, not coding units** — 1920x1080 is 1920x1080 here. Must be a multiple of the
    /// minimum CU size (`va_enc_hevc.h:378-380`).
    pub pic_width_in_luma_samples: u16,
    /// The height, with the same multiple-of-min-CU-size requirement.
    pub pic_height_in_luma_samples: u16,
    /// `seq_fields`: the sequence parameter set's flags, packed into the union's `value` member.
    /// **Bit positions are measured, not inferred** — see the module note and [`hevc_abi`]:
    /// `chroma_format_idc` is bits 0-1, `bit_depth_luma_minus8` bits 3-5, `scaling_list_enabled_flag`
    /// bit 9, `sps_temporal_mvp_enabled_flag` bit 15, `reserved_bits` 18-31.
    pub seq_fields: u32,
    pub log2_min_luma_coding_block_size_minus3: u8,
    pub log2_diff_max_min_luma_coding_block_size: u8,
    pub log2_min_transform_block_size_minus2: u8,
    pub log2_diff_max_min_transform_block_size: u8,
    pub max_transform_hierarchy_depth_inter: u8,
    pub max_transform_hierarchy_depth_intra: u8,
    /// The C layout's hole at bytes 34-35, before `pcm_sample_bit_depth_luma_minus1`.
    pub padding_after_max_transform_hierarchy_depth_intra: [u8; 2],
    pub pcm_sample_bit_depth_luma_minus1: u32,
    pub pcm_sample_bit_depth_chroma_minus1: u32,
    pub log2_min_pcm_luma_coding_block_size_minus3: u32,
    pub log2_max_pcm_luma_coding_block_size_minus3: u32,
    /// Whether the VUI below is sent at all. Zero here means the whole VUI block is ignored, which
    /// is the honest state for an encode that states no aspect ratio.
    pub vui_parameters_present_flag: u8,
    /// The C layout's hole at bytes 53-55, before `vui_fields`.
    pub padding_after_vui_parameters_present_flag: [u8; 3],
    /// `vui_fields`: the VUI's flags, packed like `seq_fields`. Measured: the eight single-bit
    /// flags are bits 0-7 in header order, `log2_max_mv_length_horizontal` is 8-12 and
    /// `log2_max_mv_length_vertical` is 13-17.
    pub vui_fields: u32,
    /// The Extended SAR aspect ratio idc (`va_enc_hevc.h:500-501`).
    pub aspect_ratio_idc: u8,
    /// The C layout's hole at bytes 61-63, before `sar_width`.
    pub padding_after_aspect_ratio_idc: [u8; 3],
    pub sar_width: u32,
    pub sar_height: u32,
    pub vui_num_units_in_tick: u32,
    pub vui_time_scale: u32,
    pub min_spatial_segmentation_idc: u16,
    pub max_bytes_per_pic_denom: u8,
    pub max_bits_per_min_cu_denom: u8,
    /// `scc_fields`: screen-content-coding flags, packed into the union's `value` member. Measured:
    /// `palette_mode_enabled_flag` is bit 0 and `reserved` is bits 1-31. Main and Main 10 have no
    /// SCC, so every bit clear is the state those profiles mean.
    pub scc_fields: u32,
    /// The header's own `va_reserved[VA_PADDING_MEDIUM - 1]` — seven words, not eight, because the
    /// record reserves *one* less than the padding width (`va_enc_hevc.h:528`). It is the last
    /// member and the record is 116 bytes, so it ends at 116 with no trailing hole.
    pub va_reserved: [u32; VA_PADDING_MEDIUM - 1],
}

/// `VAEncPictureParameterBufferHEVC` (`va_enc_hevc.h:555`): one picture's identity, which surface
/// it is reconstructed into, its reference frames, and which buffer the bitstream goes to.
/// **576 bytes**, alignof 4.
///
/// ```text
/// $ cc -w -o hevcenc_measure hevcenc_measure.c && ./hevcenc_measure
/// == VAEncPictureParameterBufferHEVC sizeof=576 alignof=4
/// ```
///
/// **One hole, C-measured:** bytes 506-507, between `nal_unit_type` and `pic_fields`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaEncPictureParameterBufferHevc {
    /// **`decoded_curr_pic.picture_id` is the reconstructed (decoded) picture, not the input**
    /// (`va_enc_hevc.h:556-564`): the driver is handed a *scratch* surface to reconstruct into, and
    /// `vaBeginPicture`'s surface is the input it reads. Putting the input surface here is the
    /// mistake the header's own comment exists to warn about. The header also says the long-term
    /// reference and RPS fields are to be set to 0 and ignored — this record has none.
    pub decoded_curr_pic: VaPictureHevc,
    /// The decoded picture buffer the P and B slices predict from: 15 reconstructed pictures, by
    /// surface, at offset 28.
    pub reference_frames: [VaPictureHevc; 15],
    /// The `ENC_CODED_BUFFER` the compressed picture is written into — a [`BufferId`].
    pub coded_buf: BufferId,
    /// Index **into `reference_frames`**, and *not* the HEVC `collocated_ref_idx` syntax element —
    /// the header is explicit that they differ (`va_enc_hevc.h:584-589`). **0xFF when
    /// `slice_fields.slice_temporal_mvp_enabled_flag` is 0**, so "no collocated picture" is 255 and
    /// not 0, which names reference slot 0 instead.
    pub collocated_ref_pic_index: u8,
    /// [`LAST_PICTURE_EOSTREAM`], or 0 for a picture in the middle. See the constant for why the
    /// sibling end-of-sequence flag is deliberately not declared.
    pub last_picture: u8,
    /// `init_qp_minus26 + 26` (`va_enc_hevc.h:601`), so this number *is* the picture's quantiser.
    pub pic_init_qp: u8,
    pub diff_cu_qp_delta_depth: u8,
    pub pps_cb_qp_offset: i8,
    pub pps_cr_qp_offset: i8,
    /// Tiles are off unless `pic_fields.tiles_enabled_flag` says otherwise, and the header's
    /// column and row width arrays below are only read when they are.
    pub num_tile_columns_minus1: u8,
    pub num_tile_rows_minus1: u8,
    pub column_width_minus1: [u8; 19],
    pub row_height_minus1: [u8; 21],
    pub log2_parallel_merge_level_minus2: u8,
    /// The CTU bit size limit, or 0 for no limit (`va_enc_hevc.h:628-631`).
    pub ctu_max_bitsize_allowed: u8,
    pub num_ref_idx_l0_default_active_minus1: u8,
    pub num_ref_idx_l1_default_active_minus1: u8,
    pub slice_pic_parameter_set_id: u8,
    /// The NAL unit type, 0..=63 (`va_enc_hevc.h:650-654`). "Used by GPU to generate new slice
    /// headers in slice size control" — this is an input to the driver's slice headers, not an
    /// output read back.
    pub nal_unit_type: u8,
    /// The C layout's hole at bytes 506-507, before `pic_fields`.
    pub padding_after_nal_unit_type: [u8; 2],
    /// `pic_fields`: the picture's own flags, packed into the union's `value` member. **Measured,
    /// not inferred** — see [`hevc_abi`]. The two that decide what the picture is: `idr_pic_flag`
    /// is bit 0 (leaving it clear on the first picture gives a stream no decoder can start from)
    /// and `coding_type` is bits 1-3, where 1 = I, 2 = P, 3 = B, 4 = B1 and 5 = B2
    /// (`va_enc_hevc.h:660-668`).
    pub pic_fields: u32,
    /// Hierarchical B level + 1 (`va_enc_hevc.h:731-735`). 0 means no hierarchical level, which is
    /// the honest value while B-frames are out of scope.
    pub hierarchical_level_plus1: u8,
    /// The header's own one-byte reserved field, which it spells `va_byte_reserved` rather than
    /// folding into the trailing array (`va_enc_hevc.h:737-738`).
    pub va_byte_reserved: u8,
    /// `scc_fields`: screen-content flags, packed into the union's `value` member — **a `u16` here,
    /// where the sequence record's union of the same name is a `u32`**. Measured:
    /// `pps_curr_pic_ref_enabled_flag` is bit 0 and `reserved` is bits 1-15.
    pub scc_fields: u16,
    /// The header's own `va_reserved[VA_PADDING_HIGH - 1]` — **fifteen** words, not eight
    /// (`va_enc_hevc.h:750`). HEVC's picture record is the one encode record on this backend with
    /// the high padding; getting that width wrong would cost 32 bytes per picture.
    pub va_reserved: [u32; VA_PADDING_HIGH - 1],
}

/// `VAEncSliceParameterBufferHEVC` (`va_enc_hevc.h:770`): one slice segment's record.
/// **1076 bytes**, alignof 4.
///
/// ```text
/// $ cc -w -o hevcenc_measure hevcenc_measure.c && ./hevcenc_measure
/// == VAEncSliceParameterBufferHEVC sizeof=1076 alignof=4
/// ```
///
/// **No holes at all, and that is a measurement rather than an absence of one.** Its neighbours are
/// full of them: H.264's encode slice record has six `padding_*` fields and this record has none.
/// [`hevc_abi`] asserts a C-measured gap of **zero** between every pair of members and after the
/// last, so "no hole" is asked rather than assumed — the `VASegmentParameterVP9` failure was a
/// record that agreed with C on every size and offset and still had an undeclared hole.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaEncSliceParameterBufferHevc {
    /// Starting CTU address for this slice.
    pub slice_segment_address: u32,
    /// Number of CTUs in this slice — the whole picture is one slice in this backend's style, as
    /// it is for H.264.
    pub num_ctu_in_slice: u32,
    /// Slice type. **What the numbers mean is not in the header**, and HEVC's is not H.264's
    /// offset-by-five encoding — see the module note's first open question. No `SLICE_TYPE_*`
    /// constant is declared for it for that reason.
    pub slice_type: u8,
    pub slice_pic_parameter_set_id: u8,
    pub num_ref_idx_l0_active_minus1: u8,
    pub num_ref_idx_l1_active_minus1: u8,
    /// Reference picture list 0 (P slices), by surface. **Fifteen entries of a 28-byte
    /// `VAPictureHEVC` at offset 12** — unlike H.264's encode slice record, whose lists are 32
    /// entries at offset 36. The same header list, a different length, in a different place.
    pub ref_pic_list0: [VaPictureHevc; 15],
    /// Reference picture list 1 (B slices), at offset 432.
    pub ref_pic_list1: [VaPictureHevc; 15],
    // The `pred_weight_table()` block below: nine members from offset 852 to 1033, all `i8` or
    // arrays of it, with no hole anywhere in it.
    /// The `pred_weight_table()` syntax elements, as spelled in H.265. **All of them are ignored**
    /// when `pic_fields.enable_gpu_weighted_prediction` is set (`va_enc_hevc.h:709-721`), which is
    /// the mode an encoder that does not compute its own prediction weights wants.
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
    /// `MaxNumMergeCand`, 1..=5 (`va_enc_hevc.h:821-824`).
    pub max_num_merge_cand: u8,
    /// Added to `pic_init_qp` to give this slice's quantiser. **0 means "the picture's QP"**, which
    /// is what a per-slice override must say.
    pub slice_qp_delta: i8,
    pub slice_cb_qp_offset: i8,
    pub slice_cr_qp_offset: i8,
    pub slice_beta_offset_div2: i8,
    pub slice_tc_offset_div2: i8,
    /// `slice_fields`: the slice segment header's flags, packed into the union's `value` member.
    /// **Measured, not inferred** — see [`hevc_abi`]: `last_slice_of_pic_flag` is bit 0,
    /// `colour_plane_id` bits 2-3, `slice_deblocking_filter_disabled_flag` bits 10-11 and
    /// `collocated_from_l0_flag` bit 13.
    pub slice_fields: u32,
    /// Where `pred_weight_table()` starts in the packed slice header, and how long it is
    /// (`va_enc_hevc.h:876-890`). Only read when the encoder prefers to override the weighted
    /// prediction parameters passed in from the application — see
    /// `enable_gpu_weighted_prediction`.
    pub pred_weight_table_bit_offset: u32,
    pub pred_weight_table_bit_length: u32,
    /// The header's own `va_reserved[VA_PADDING_MEDIUM - 2]` — **six** words, not eight
    /// (`va_enc_hevc.h:892`). The record is 1076 bytes and this ends at 1076, so there is no
    /// trailing hole either.
    pub va_reserved: [u32; VA_PADDING_MEDIUM - 2],
}

/// `HEVC_LAST_PICTURE_EOSTREAM` (`va_enc_hevc.h:253`): the picture record's `last_picture` value
/// that tells the driver this is the last picture of the stream, so it appends an `end_of_stream()`
/// NAL.
///
/// **Its sibling `HEVC_LAST_PICTURE_EOSEQ` (`va_enc_hevc.h:247`) is deliberately not declared**, for
/// the same reason H.264's is not: it marks the end of a *sequence* within a stream, a distinction
/// this backend's encoder has no way to express — its picture stream is one sequence from end to
/// end — and declaring a constant nothing could set honestly is how a constant gets set anyway.
pub const LAST_PICTURE_EOSTREAM: u8 = 0x02;

/// `VAEncPackedHeaderHEVC_VPS` / `SPS` (`va_enc_hevc.h:277`, `:286`): the packed-header type that
/// carries the video and sequence parameter sets.
///
/// **Both enumerators are the same value, 1** — measured, and the compiler agrees: the header
/// assigns both `= VAEncPackedHeaderSequence`, so `VPS == SPS == VAEncPackedHeaderSequence`. A
/// packed-header buffer's type therefore does *not* say whether it is a VPS or an SPS; the driver
/// takes the sequence and the SPS as the first and second such buffers of the sequence. This is
/// recorded because the obvious mistake — reading a difference into two constants that have none —
/// produces a stream missing its parameter sets and no error anywhere.
pub const PACKED_HEADER_VPS_OR_SPS: u32 = 1;

/// `VAEncPackedHeaderHEVC_PPS` (`va_enc_hevc.h:295`): the packed-header type for the picture
/// parameter set, which is `VAEncPackedHeaderPicture`.
pub const PACKED_HEADER_PPS: u32 = 2;

/// `VAEncPackedHeaderHEVC_Slice` (`va_enc_hevc.h:308`): the packed-header type for a slice header,
/// which is `VAEncPackedHeaderSlice`.
///
/// Two things about the packed-header types that are recorded here rather than left for the next
/// task to discover:
///
/// - **`VAEncPackedHeaderHEVC_SEI` (`va_enc_hevc.h:323`) is deliberately not declared.** The header
///   marks it `@deprecated` and says every application can pass the same data through
///   `VA_ENC_PACKED_HEADER_RAW_DATA` instead. Its value, measured, is `0x80000001` — inside
///   `VAEncPackedHeaderMiscMask` — so it cannot be confused with a real type if it is ever written
///   by accident.
/// - **Which buffers a sequence record must be accompanied by, and when**, is the header's own note
///   at `va_enc_hevc.h:333-344`: in packed-sequence mode the driver expects a
///   `VAEncPackedHeaderParameterBuffer` and a `VAEncPackedHeaderDataBuffer` in the *same*
///   `vaRenderPicture` call. That call is a later task.
pub const PACKED_HEADER_SLICE: u32 = 3;

/// `VAConfigAttribEncHEVCFeatures` (`va.h:1002`): the HEVC encoder's feature set, read through
/// `vaGetConfigAttributes` and answered with a [`HevcFeatures`] word.
///
/// This is one of the two attributes HEVC has that the codec-generic ones do not cover; the other
/// is [`CONFIG_ATTRIB_HEVC_BLOCK_SIZES`]. The generic encoder attributes — packed headers
/// (`ffi::CONFIG_ATTRIB_ENC_PACKED_HEADERS`), maximum reference frames
/// (`ffi::CONFIG_ATTRIB_ENC_MAX_REF_FRAMES`) and rate control
/// (`ffi::CONFIG_ATTRIB_RATE_CONTROL`) — are already declared in [`super::ffi`] and are not
/// repeated here.
pub const CONFIG_ATTRIB_HEVC_FEATURES: u32 = 50;

/// `VAConfigAttribEncHEVCBlockSizes` (`va.h:1012`): the coding-tree and transform-block sizes the
/// HEVC encoder supports, answered with a [`HevcBlockSizes`] word.
pub const CONFIG_ATTRIB_HEVC_BLOCK_SIZES: u32 = 51;

/// `VAConfigAttribEncMaxSlices` (`va.h:721`): the most slices the driver will encode in one frame.
/// Read-only. An encoder that has not asked cannot know whether one slice per picture is legal,
/// and the answer is a driver-specific number — 2 on some hardware, which would mean the
/// single-slice H.264 spike's shape does not transfer.
///
/// `VAConfigAttribEncInterlaced` (`va.h:703`) is **not declared**, for the same reason
/// [`LAST_PICTURE_EOSTREAM`]'s sibling is: it is a constant for an encoder that encodes interlaced
/// content, and FilmCraft's does not, so there is no honest value to write and a declared constant
/// is one a later edit can set.
pub const CONFIG_ATTRIB_MAX_SLICES: u32 = 14;

/// `VAConfigAttribValEncHEVCFeatures` (`va_enc_hevc.h:56`): the 32-bit word
/// [`CONFIG_ATTRIB_HEVC_FEATURES`] is answered in. Fifteen 2-bit fields; see [`HevcFeatures`].
pub type HevcFeatures = u32;

/// `VAConfigAttribValEncHEVCBlockSizes` (`va_enc_hevc.h:154`): the 32-bit word
/// [`CONFIG_ATTRIB_HEVC_BLOCK_SIZES`] is answered in. Eleven 2-bit fields and a 10-bit `reserved`;
/// see [`HevcBlockSizes`].
pub type HevcBlockSizes = u32;

/// The bit a field of [`HevcFeatures`] occupies, **measured by running C** and pinned in
/// [`hevc_abi`] rather than read off the declaration order.
///
/// The feature words are `VA_FEATURE_*` values, so a field is *not* a 0/1 flag but a support level
/// — and there is no `VA_FEATURE_SUPPORTED` constant in these headers to compare it against. Reading
/// one of these as a boolean is the kind of thing that works on one driver and silently means
/// "unsupported" on another.
pub mod features {
    /// 2 bits, 0-1: separate colour planes (`scaling_list`-independent; allows
    /// `separate_colour_plane_flag` in the SPS).
    pub const SEPARATE_COLOUR_PLANES: u32 = 0;
    /// 2 bits, 2-3: scaling lists — and note the header's extra sentence (`va_enc_hevc.h:63-69`):
    /// enabling this also obliges the application to supply a `VAQMatrixBufferHEVC`, which this
    /// file does not declare. A driver that supports scaling lists is therefore **not** something
    /// this encoder can use yet.
    pub const SCALING_LISTS: u32 = 2;
    /// 2 bits, 4-5: asymmetric motion partitions.
    pub const AMP: u32 = 4;
    /// 2 bits, 6-7: sample adaptive offset.
    pub const SAO: u32 = 6;
    /// 2 bits, 8-9: PCM sample blocks.
    pub const PCM: u32 = 8;
    /// 2 bits, 10-11: temporal motion vector prediction.
    pub const TEMPORAL_MVP: u32 = 10;
    /// 2 bits, 12-13: strong intra smoothing.
    pub const STRONG_INTRA_SMOOTHING: u32 = 12;
    /// 2 bits, 14-15: dependent slices.
    pub const DEPENDENT_SLICES: u32 = 14;
    /// 2 bits, 16-17: sign data hiding.
    pub const SIGN_DATA_HIDING: u32 = 16;
    /// 2 bits, 18-19: constrained intra prediction.
    pub const CONSTRAINED_INTRA_PRED: u32 = 18;
    /// 2 bits, 20-21: transform skipping.
    pub const TRANSFORM_SKIP: u32 = 20;
    /// 2 bits, 22-23: QP delta within coding units.
    pub const CU_QP_DELTA: u32 = 22;
    /// 2 bits, 24-25: weighted prediction.
    pub const WEIGHTED_PREDICTION: u32 = 24;
    /// 2 bits, 26-27: transform and quantisation bypass.
    pub const TRANSQUANT_BYPASS: u32 = 26;
    /// 2 bits, 28-29: deblocking filter disable.
    pub const DEBLOCKING_FILTER_DISABLE: u32 = 28;
    /// 2 bits, 30-31: `reserved`, which the header says must not be used because it would collide
    /// with `VA_ATTRIB_NOT_SUPPORTED` (`va_enc_hevc.h:143-144`).
    pub const RESERVED: u32 = 30;

    /// The 2-bit mask a feature field occupies, from its bit offset.
    pub const fn mask(bit: u32) -> u32 {
        0b11 << bit
    }
}

/// The bit a field of [`HevcBlockSizes`] occupies, **measured by running C** and pinned in
/// [`hevc_abi`].
///
/// These are `log2` sizes *minus the offset named in the field itself* — `CtbLog2SizeY` is
/// `log2_max_coding_tree_block_size_minus3 + 3` — so a block size of 64 CTUs is the value 3 in bits
/// 0-1. Reading one of these as a block size is off by three and produces a picture the driver
/// cannot place.
pub mod block_sizes {
    /// 2 bits, 0-1.
    pub const LOG2_MAX_CODING_TREE_BLOCK_SIZE_MINUS3: u32 = 0;
    /// 2 bits, 2-3.
    pub const LOG2_MIN_CODING_TREE_BLOCK_SIZE_MINUS3: u32 = 2;
    /// 2 bits, 4-5.
    pub const LOG2_MIN_LUMA_CODING_BLOCK_SIZE_MINUS3: u32 = 4;
    /// 2 bits, 6-7.
    pub const LOG2_MAX_LUMA_TRANSFORM_BLOCK_SIZE_MINUS2: u32 = 6;
    /// 2 bits, 8-9.
    pub const LOG2_MIN_LUMA_TRANSFORM_BLOCK_SIZE_MINUS2: u32 = 8;
    /// 2 bits, 10-11.
    pub const MAX_MAX_TRANSFORM_HIERARCHY_DEPTH_INTER: u32 = 10;
    /// 2 bits, 12-13.
    pub const MIN_MAX_TRANSFORM_HIERARCHY_DEPTH_INTER: u32 = 12;
    /// 2 bits, 14-15.
    pub const MAX_MAX_TRANSFORM_HIERARCHY_DEPTH_INTRA: u32 = 14;
    /// 2 bits, 16-17.
    pub const MIN_MAX_TRANSFORM_HIERARCHY_DEPTH_INTRA: u32 = 16;
    /// 2 bits, 18-19.
    pub const LOG2_MAX_PCM_CODING_BLOCK_SIZE_MINUS3: u32 = 18;
    /// 2 bits, 20-21.
    pub const LOG2_MIN_PCM_CODING_BLOCK_SIZE_MINUS3: u32 = 20;
    /// 10 bits, 22-31: reserved for future use.
    pub const RESERVED: u32 = 22;
    /// The 10-bit mask the `reserved` field occupies, from its bit offset.
    pub const fn reserved_mask(bit: u32) -> u32 {
        0x3FF << bit
    }
}

#[cfg(test)]
#[path = "hevc_abi.rs"]
mod hevc_abi;
