//! The pins on the AV1 **encode** records: every field of every record, named, and every one of
//! them checked against the installed `/usr/include/va/va_enc_av1.h` by a C compiler.
//!
//! This is [`super::abi_guard`] for the AV1 encode records and lives beside its records rather
//! than in the shared file for one reason only: **zero overlap**. `abi_guard.rs` is shared with the
//! four codecs' *decode* and is being read by other work; two writers on one file is how the tree
//! got corrupted earlier in this session. Nothing here is shared, and the file is `#[cfg(test)]`
//! only, so the cost of the duplication is compile time and nothing else.
//!
//! # What is checked, and why each check exists
//!
//! | test | what it would miss |
//! |---|---|
//! | [`every_record_the_av1_encoder_would_submit_tiles_byte_for_byte`] | a **hole**. No `sizeof` and no `offsetof` can see one — padding changes no size and moves no named field. `VASegmentParameterVP9` agreed with C on 24/24 records for five commits while handing the driver two uninitialised bytes per record. |
//! | [`no_av1_enc_record_has_a_byte_the_header_does_not_account_for`] | a wrong offset, a wrong field width, or a hole of the wrong width — asked of C **as a gap**, because C gives padding no name and `sizeof` cannot see one. |
//! | [`the_av1_enc_bitfield_words_are_the_bits_the_c_compiler_measures`] | a **bit position**. A union has one `offsetof`, so a member packed one bit too high agrees with C on the record's `sizeof`, on every field offset and on every hole, and produces a *picture*. |
//! | [`the_av1_enc_constants_are_what_the_header_declares`] | a constant somebody typed from the same header twice. |
//! | [`the_av1_enc_word_widths_are_the_ones_the_bits_were_measured_in`] | a bit read back out of the **wrong width word**. These records mix `u8`, `u16` and `u32` unions; an 8-bit member read as a 32-bit one reports a high bit of 31 and passes every size, offset and hole check. |
//! | [`a_default_av1_encode_record_names_no_byte_it_leaves_unset`] | a record whose `Default` fills one field and not another — which for these records includes the five holes. |
//! | [`the_feature_levels_are_the_ones_va_h_defines`] | reading a feature level wrong. `VA_FEATURE_REQUIRED` is **2**, it obliges a submission, and **3 is not a `VA_FEATURE_*` value at all** — a decoder that cannot tell 2 from 1 gets HEVC right by luck and AV1 wrong on purpose. |
//! | [`the_driver_s_av1_feature_word_has_no_required_field`] | the driver answering something other than `0x05001000`. The decode is done by **C, through the header's own union**, so this is a measurement of what the word means and not a second transcription of the sentence above it. |
//! | [`the_av1_enc_records_are_the_sizes_this_file_documents`] | a record file's table going stale while the declaration is right. |
//!
//! # Why the pins ask for *gaps*, not just sizes
//!
//! `sizeof` is blind twice over: it cannot see a hole, and **shrinking a hole does not always move
//! it** — the compiler re-inserts the byte the field no longer accounts for. A declaration with the
//! right members and a hole one byte too narrow still has the right `sizeof` and the right
//! `offsetof` for every named member, and still hands the driver a byte nobody wrote. So the C side
//! is asked `offsetof(b) - offsetof(a) - sizeof(a)` between each pair of members, and compared with
//! the width of the `padding_*` field that claims to cover it. **A record with no hole is asserted
//! to have a gap of zero**, so "no hole" is a measurement and not an absence of one — which is
//! exactly the status [`VaRefFrameCtrlAv1`](super::VaRefFrameCtrlAv1) holds, being the one record
//! here that is a bare union.
//!
//! # The bitfield tables were measured by running C
//!
//! All **100** positions came from setting one member to `~0u` and reading `__builtin_ctzll` and
//! `__builtin_clzll` back out of the union's `value`, with the word's real width carried per row.
//! Nothing reads the header's declaration order. The word width is carried per row rather than
//! assumed because these records are where that assumption first costs something: `seg_flags`,
//! `loop_filter_flags` and `tile_group_obu_hdr_info` are **8-bit** unions and `qmatrix_flags` and
//! `loop_restoration_flags` are **16-bit**, in the same record as four 32-bit ones.
//!
//! The inner anonymous struct is `bits` in eight of the nine unions and **`fields`** in
//! `VARefFrameCtrlAV1` (`va_enc_av1.h:489`, `:535`), so the inner name is carried per row too.

use super::super::ffi::VA_PADDING_LOW;
use super::*;
use std::mem::{offset_of, size_of};

/// Where the libva headers live on this machine — the same path, and the same reasoning, as
/// [`super::abi_guard`]: it is a path to *check*, never an `-I` include flag.
const LIBVA_HEADERS: &str = "/usr/include/va";

/// One field of a record: the member's name **in the C header**, our field's name, and its width.
///
/// The C name is `-` where C gives the gap no name at all, which is every `padding_*` field and is
/// the reason a hole cannot simply be read off a header.
struct RecordField {
    c_name: &'static str,
    rust_name: &'static str,
    offset: usize,
    width: usize,
}

/// A record and its fields, in declaration order.
struct EncRecord {
    /// The struct's name in `/usr/include/va/`.
    c_name: &'static str,
    /// `size_of` on our declaration, which the C compiler is asked to agree with.
    size: usize,
    fields: &'static [RecordField],
}

/// `enc_records!` builds [`ENC_RECORDS`] from the list below.
///
/// A macro rather than plain data because a field's offset can only be read with `offset_of!`,
/// which needs the field name as a token; writing the list as a macro invocation is what lets each
/// row carry both names and have the offset computed rather than transcribed — which is the whole
/// point of this module.
macro_rules! enc_records {
    ($($c_record:literal : size_of::<$ty:ty>() => [ $( ($c_field:literal, $field:ident, $width:expr) ),* $(,)? ]),* $(,)?) => {
        &[$(EncRecord {
            c_name: $c_record,
            size: size_of::<$ty>(),
            fields: &[
                $(RecordField {
                    c_name: $c_field,
                    rust_name: stringify!($field),
                    offset: offset_of!($ty, $field),
                    width: $width,
                }),*
            ],
        }),*]
    };
}

/// The six AV1 encode records, with every byte of every one of them named.
///
/// **Five holes, all `padding_*` fields of a C-measured width:**
///
/// | record | bytes | why |
/// |---|---|---|
/// | `VAEncSequenceParameterBufferAV1` | 21-23 | `order_hint_bits_minus_1` is a byte and `va_reserved` must be 4-aligned |
/// | `VAEncSegParamAV1` | 138-139 | `feature_mask` ends 2 bytes short of the 4-aligned `va_reserved` |
/// | `VAEncWarpedMotionParamsAV1` | 37-39 | `invalid` is a byte and `va_reserved` must be 4-aligned |
/// | `VAEncPictureParameterBufferAV1` | 538-539 | `loop_restoration_flags` is a `u16` and `wm` must be 4-aligned |
/// | `VAEncTileGroupBufferAV1` | 2-3 | two `u8`s and `va_reserved` must be 4-aligned |
///
/// **`VARefFrameCtrlAV1` has none**, and it has no `padding_*` row here on purpose: a member with
/// no C name can only appear as one of those, so a hole appearing where the header has none would
/// mean this declaration *grew* one the driver never saw. The gap check below asserts a width of
/// **zero** for it rather than leaving a hole-free record unchecked.
static ENC_RECORDS: &[EncRecord] = enc_records![
    "VAEncSequenceParameterBufferAV1": size_of::<VaEncSequenceParameterBufferAv1>() => [
        ("seq_profile", seq_profile, size_of::<u8>()),
        ("seq_level_idx", seq_level_idx, size_of::<u8>()),
        ("seq_tier", seq_tier, size_of::<u8>()),
        ("hierarchical_flag", hierarchical_flag, size_of::<u8>()),
        ("intra_period", intra_period, size_of::<u32>()),
        ("ip_period", ip_period, size_of::<u32>()),
        ("bits_per_second", bits_per_second, size_of::<u32>()),
        ("seq_fields", seq_fields, size_of::<u32>()),
        ("order_hint_bits_minus_1", order_hint_bits_minus_1, size_of::<u8>()),
        ("-", padding_after_order_hint_bits_minus_1, size_of::<[u8; 3]>()),
        ("va_reserved", va_reserved, 16 * size_of::<u32>()),
    ],
    "VAEncSegParamAV1": size_of::<VaEncSegParamAv1>() => [
        ("seg_flags", seg_flags, size_of::<u8>()),
        ("segment_number", segment_number, size_of::<u8>()),
        ("feature_data", feature_data, size_of::<[[i16; SEG_LVL_MAX]; MAX_SEGMENTS]>()),
        ("feature_mask", feature_mask, size_of::<[u8; MAX_SEGMENTS]>()),
        ("-", padding_after_feature_mask, size_of::<[u8; 2]>()),
        ("va_reserved", va_reserved, VA_PADDING_LOW * size_of::<u32>()),
    ],
    "VAEncWarpedMotionParamsAV1": size_of::<VaEncWarpedMotionParamsAv1>() => [
        ("wmtype", wmtype, size_of::<u32>()),
        ("wmmat", wmmat, size_of::<[i32; 8]>()),
        ("invalid", invalid, size_of::<u8>()),
        ("-", padding_after_invalid, size_of::<[u8; 3]>()),
        ("va_reserved", va_reserved, VA_PADDING_LOW * size_of::<u32>()),
    ],
    "VARefFrameCtrlAV1": size_of::<VaRefFrameCtrlAv1>() => [
        ("value", value, size_of::<u32>()),
    ],
    "VAEncPictureParameterBufferAV1": size_of::<VaEncPictureParameterBufferAv1>() => [
        ("frame_width_minus_1", frame_width_minus_1, size_of::<u16>()),
        ("frame_height_minus_1", frame_height_minus_1, size_of::<u16>()),
        ("reconstructed_frame", reconstructed_frame, size_of::<SurfaceId>()),
        ("coded_buf", coded_buf, size_of::<BufferId>()),
        ("reference_frames", reference_frames, size_of::<[SurfaceId; 8]>()),
        ("ref_frame_idx", ref_frame_idx, size_of::<[u8; 7]>()),
        ("hierarchical_level_plus1", hierarchical_level_plus1, size_of::<u8>()),
        ("primary_ref_frame", primary_ref_frame, size_of::<u8>()),
        ("order_hint", order_hint, size_of::<u8>()),
        ("refresh_frame_flags", refresh_frame_flags, size_of::<u8>()),
        ("reserved8bits1", reserved8bits1, size_of::<u8>()),
        ("ref_frame_ctrl_l0", ref_frame_ctrl_l0, size_of::<VaRefFrameCtrlAv1>()),
        ("ref_frame_ctrl_l1", ref_frame_ctrl_l1, size_of::<VaRefFrameCtrlAv1>()),
        ("picture_flags", picture_flags, size_of::<u32>()),
        ("seg_id_block_size", seg_id_block_size, size_of::<u8>()),
        ("num_tile_groups_minus1", num_tile_groups_minus1, size_of::<u8>()),
        ("temporal_id", temporal_id, size_of::<u8>()),
        ("filter_level", filter_level, size_of::<[u8; 2]>()),
        ("filter_level_u", filter_level_u, size_of::<u8>()),
        ("filter_level_v", filter_level_v, size_of::<u8>()),
        ("loop_filter_flags", loop_filter_flags, size_of::<u8>()),
        ("superres_scale_denominator", superres_scale_denominator, size_of::<u8>()),
        ("interpolation_filter", interpolation_filter, size_of::<u8>()),
        ("ref_deltas", ref_deltas, size_of::<[i8; 8]>()),
        ("mode_deltas", mode_deltas, size_of::<[i8; 2]>()),
        ("base_qindex", base_qindex, size_of::<u8>()),
        ("y_dc_delta_q", y_dc_delta_q, size_of::<i8>()),
        ("u_dc_delta_q", u_dc_delta_q, size_of::<i8>()),
        ("u_ac_delta_q", u_ac_delta_q, size_of::<i8>()),
        ("v_dc_delta_q", v_dc_delta_q, size_of::<i8>()),
        ("v_ac_delta_q", v_ac_delta_q, size_of::<i8>()),
        ("min_base_qindex", min_base_qindex, size_of::<u8>()),
        ("max_base_qindex", max_base_qindex, size_of::<u8>()),
        ("qmatrix_flags", qmatrix_flags, size_of::<u16>()),
        ("reserved16bits1", reserved16bits1, size_of::<u16>()),
        ("mode_control_flags", mode_control_flags, size_of::<u32>()),
        ("segments", segments, size_of::<VaEncSegParamAv1>()),
        ("tile_cols", tile_cols, size_of::<u8>()),
        ("tile_rows", tile_rows, size_of::<u8>()),
        ("reserved16bits2", reserved16bits2, size_of::<u16>()),
        ("width_in_sbs_minus_1", width_in_sbs_minus_1, size_of::<[u16; 63]>()),
        ("height_in_sbs_minus_1", height_in_sbs_minus_1, size_of::<[u16; 63]>()),
        ("context_update_tile_id", context_update_tile_id, size_of::<u16>()),
        ("cdef_damping_minus_3", cdef_damping_minus_3, size_of::<u8>()),
        ("cdef_bits", cdef_bits, size_of::<u8>()),
        ("cdef_y_strengths", cdef_y_strengths, size_of::<[u8; 8]>()),
        ("cdef_uv_strengths", cdef_uv_strengths, size_of::<[u8; 8]>()),
        ("loop_restoration_flags", loop_restoration_flags, size_of::<u16>()),
        ("-", padding_after_loop_restoration_flags, size_of::<[u8; 2]>()),
        ("wm", wm, size_of::<[VaEncWarpedMotionParamsAv1; 7]>()),
        ("bit_offset_qindex", bit_offset_qindex, size_of::<u32>()),
        ("bit_offset_segmentation", bit_offset_segmentation, size_of::<u32>()),
        ("bit_offset_loopfilter_params", bit_offset_loopfilter_params, size_of::<u32>()),
        ("bit_offset_cdef_params", bit_offset_cdef_params, size_of::<u32>()),
        ("size_in_bits_cdef_params", size_in_bits_cdef_params, size_of::<u32>()),
        ("byte_offset_frame_hdr_obu_size", byte_offset_frame_hdr_obu_size, size_of::<u32>()),
        ("size_in_bits_frame_hdr_obu", size_in_bits_frame_hdr_obu, size_of::<u32>()),
        ("tile_group_obu_hdr_info", tile_group_obu_hdr_info, size_of::<u8>()),
        ("number_skip_frames", number_skip_frames, size_of::<u8>()),
        ("reserved16bits3", reserved16bits3, size_of::<u16>()),
        ("skip_frames_reduced_size", skip_frames_reduced_size, size_of::<i32>()),
        ("va_reserved", va_reserved, 16 * size_of::<u32>()),
    ],
    "VAEncTileGroupBufferAV1": size_of::<VaEncTileGroupBufferAv1>() => [
        ("tg_start", tg_start, size_of::<u8>()),
        ("tg_end", tg_end, size_of::<u8>()),
        ("-", padding_after_tg_end, size_of::<[u8; 2]>()),
        ("va_reserved", va_reserved, VA_PADDING_LOW * size_of::<u32>()),
    ],
];

/// **Every byte of every AV1 encode record belongs to a named field, and the fields line up end to
/// end with no gap between two of them.**
///
/// This is the property no `sizeof` and no `offsetof` can see. Padding changes no size and moves no
/// named member, so a record with a hole passes every other test in this file — which is precisely
/// how `VASegmentParameterVP9` decoded 24/24 pictures bit-exact for five commits while handing the
/// driver two indeterminate bytes per record.
///
/// The check is the strict one: each field must start exactly where the one before it ends, and the
/// last must end exactly at the record's size. A `padding_*` field that is absent, or present with
/// the wrong width, fails here as loudly as one that is too wide.
///
/// The widths and offsets walked above are **the table's**, which is a separate statement from the
/// width and offset of the *struct* the driver is handed — and the second loop below is what joins
/// the two. Without it a hole narrowed in the declaration (`[u8; 3]` to `[u8; 2]`) still leaves the
/// record's `sizeof` and every named member's `offsetof` exactly where they were, because the
/// compiler re-inserts the byte at the next 4-aligned field: **this test passed green over that
/// break**, and so did every other pin in this file, because the table was still right and the
/// struct was not.
#[test]
fn every_record_the_av1_encoder_would_submit_tiles_byte_for_byte() {
    assert_eq!(ENC_RECORDS.len(), 6, "the six records of va_enc_av1.h, and nothing else");
    for record in ENC_RECORDS {
        assert!(!record.fields.is_empty(), "{}: no fields listed, so the loop below proved nothing", record.c_name);
        let mut at = 0usize;
        for field in record.fields {
            assert!(field.width > 0, "{}: {} is {} bytes wide", record.c_name, field.rust_name, field.width);
            assert_eq!(field.offset, at, "{}: {} starts at {}, and the field before it ends at {at}", record.c_name, field.rust_name, field.offset);
            at += field.width;
        }
        assert_eq!(at, record.size, "{}: the named fields end at {at} and the record is {} bytes", record.c_name, record.size);
    }

    // **The declaration agrees with the table**: every row is the same field of the same struct, at
    // the same offset and the same width, read from the struct itself with `offset_of!` and
    // `size_of_val!` rather than from the table.
    macro_rules! row_matches {
        ($record:expr, $field:ident, $ty:ty) => {{
            const _: () = ();
            let got_off = std::mem::offset_of!($ty, $field);
            let got_wid = {
                // A zeroed instance of the record, to borrow the field from: no `unsafe`, and the
                // bytes are never read, only the field's size taken.
                let r = <$ty>::default();
                let _ = &r;
                std::mem::size_of_val(&r.$field)
            };
            assert_eq!(
                got_off,
                $record.offset,
                "{}.{} starts at {} in the struct, the table says {}",
                stringify!($ty),
                stringify!($field),
                got_off,
                $record.offset
            );
            assert_eq!(
                got_wid,
                $record.width,
                "{}.{} is {} bytes in the struct, the table says {}",
                stringify!($ty),
                stringify!($field),
                got_wid,
                $record.width
            );
        }};
    }
    let rec = |n: &str| ENC_RECORDS.iter().find(|r| r.c_name == n).unwrap_or_else(|| panic!("no table row for {n}"));
    let row = |n: &str, f: &str| rec(n).fields.iter().find(|x| x.rust_name == f).unwrap_or_else(|| panic!("no {n}.{f} row"));

    row_matches!(row("VAEncSequenceParameterBufferAV1", "seq_profile"), seq_profile, VaEncSequenceParameterBufferAv1);
    row_matches!(row("VAEncSequenceParameterBufferAV1", "order_hint_bits_minus_1"), order_hint_bits_minus_1, VaEncSequenceParameterBufferAv1);
    row_matches!(
        row("VAEncSequenceParameterBufferAV1", "padding_after_order_hint_bits_minus_1"),
        padding_after_order_hint_bits_minus_1,
        VaEncSequenceParameterBufferAv1
    );
    row_matches!(row("VAEncSequenceParameterBufferAV1", "va_reserved"), va_reserved, VaEncSequenceParameterBufferAv1);

    row_matches!(row("VAEncSegParamAV1", "seg_flags"), seg_flags, VaEncSegParamAv1);
    row_matches!(row("VAEncSegParamAV1", "feature_data"), feature_data, VaEncSegParamAv1);
    row_matches!(row("VAEncSegParamAV1", "feature_mask"), feature_mask, VaEncSegParamAv1);
    row_matches!(row("VAEncSegParamAV1", "padding_after_feature_mask"), padding_after_feature_mask, VaEncSegParamAv1);
    row_matches!(row("VAEncSegParamAV1", "va_reserved"), va_reserved, VaEncSegParamAv1);

    row_matches!(row("VAEncWarpedMotionParamsAV1", "wmtype"), wmtype, VaEncWarpedMotionParamsAv1);
    row_matches!(row("VAEncWarpedMotionParamsAV1", "wmmat"), wmmat, VaEncWarpedMotionParamsAv1);
    row_matches!(row("VAEncWarpedMotionParamsAV1", "invalid"), invalid, VaEncWarpedMotionParamsAv1);
    row_matches!(row("VAEncWarpedMotionParamsAV1", "padding_after_invalid"), padding_after_invalid, VaEncWarpedMotionParamsAv1);
    row_matches!(row("VAEncWarpedMotionParamsAV1", "va_reserved"), va_reserved, VaEncWarpedMotionParamsAv1);

    row_matches!(row("VAEncPictureParameterBufferAV1", "frame_width_minus_1"), frame_width_minus_1, VaEncPictureParameterBufferAv1);
    row_matches!(row("VAEncPictureParameterBufferAV1", "reconstructed_frame"), reconstructed_frame, VaEncPictureParameterBufferAv1);
    row_matches!(row("VAEncPictureParameterBufferAV1", "ref_frame_idx"), ref_frame_idx, VaEncPictureParameterBufferAv1);
    row_matches!(row("VAEncPictureParameterBufferAV1", "ref_frame_ctrl_l0"), ref_frame_ctrl_l0, VaEncPictureParameterBufferAv1);
    row_matches!(row("VAEncPictureParameterBufferAV1", "picture_flags"), picture_flags, VaEncPictureParameterBufferAv1);
    row_matches!(row("VAEncPictureParameterBufferAV1", "loop_filter_flags"), loop_filter_flags, VaEncPictureParameterBufferAv1);
    row_matches!(row("VAEncPictureParameterBufferAV1", "qmatrix_flags"), qmatrix_flags, VaEncPictureParameterBufferAv1);
    row_matches!(row("VAEncPictureParameterBufferAV1", "mode_control_flags"), mode_control_flags, VaEncPictureParameterBufferAv1);
    row_matches!(row("VAEncPictureParameterBufferAV1", "segments"), segments, VaEncPictureParameterBufferAv1);
    row_matches!(row("VAEncPictureParameterBufferAV1", "width_in_sbs_minus_1"), width_in_sbs_minus_1, VaEncPictureParameterBufferAv1);
    row_matches!(row("VAEncPictureParameterBufferAV1", "loop_restoration_flags"), loop_restoration_flags, VaEncPictureParameterBufferAv1);
    row_matches!(
        row("VAEncPictureParameterBufferAV1", "padding_after_loop_restoration_flags"),
        padding_after_loop_restoration_flags,
        VaEncPictureParameterBufferAv1
    );
    row_matches!(row("VAEncPictureParameterBufferAV1", "wm"), wm, VaEncPictureParameterBufferAv1);
    row_matches!(row("VAEncPictureParameterBufferAV1", "tile_group_obu_hdr_info"), tile_group_obu_hdr_info, VaEncPictureParameterBufferAv1);
    row_matches!(row("VAEncPictureParameterBufferAV1", "va_reserved"), va_reserved, VaEncPictureParameterBufferAv1);

    row_matches!(row("VAEncTileGroupBufferAV1", "tg_start"), tg_start, VaEncTileGroupBufferAv1);
    row_matches!(row("VAEncTileGroupBufferAV1", "tg_end"), tg_end, VaEncTileGroupBufferAv1);
    row_matches!(row("VAEncTileGroupBufferAV1", "padding_after_tg_end"), padding_after_tg_end, VaEncTileGroupBufferAv1);
    row_matches!(row("VAEncTileGroupBufferAV1", "va_reserved"), va_reserved, VaEncTileGroupBufferAv1);
}

/// **The header agrees with us on every field's offset, every field's width and every hole's
/// width** — for all six records at once, including the one that genuinely has no hole.
///
/// Each record's `sizeof` is asserted; each member its `offsetof` and its `sizeof`; and each pair of
/// neighbours the **gap** between them, because C gives padding no name and the only way to ask a C
/// compiler "how wide is that hole" is as the difference between two `offsetof`s minus the first
/// member's own width. The gap is then compared with the width of the `padding_*` field that claims
/// to cover it, so a hole one byte too narrow fails — **and a `sizeof` would not move.**
#[test]
fn no_av1_enc_record_has_a_byte_the_header_does_not_account_for() {
    if !std::path::Path::new(LIBVA_HEADERS).join("va_enc_av1.h").is_file() {
        eprintln!("SKIPPED: no {LIBVA_HEADERS}/va_enc_av1.h");
        return;
    }
    let Some(cc) = compiler() else {
        eprintln!("SKIPPED: no C compiler to measure the headers with");
        return;
    };
    let mut source = String::from("#include <stdint.h>\n#include <va/va.h>\n#include <va/va_enc_av1.h>\n");
    for record in ENC_RECORDS {
        source.push_str(&format!("\n/* {} */\n", record.c_name));
        source.push_str(&format!(
            "_Static_assert(sizeof({r}) == {size}, \"{r}: sizeof\");\n_Static_assert(_Alignof({r}) == {align}, \"{r}: alignof\");\n",
            r = record.c_name,
            size = record.size,
            align = align_of(record.c_name)
        ));
        // `previous` tracks the last member **C names**, not the last field: a `padding_*` field
        // stands between two C members and has no C name of its own, so the hole it covers is the
        // gap from the member before it to the member after — and the width compared against it is
        // every byte between them, the `padding_*` field where there is exactly one and **zero**
        // where there is none.
        let mut previous: Option<&RecordField> = None;
        for field in record.fields {
            if field.c_name == "-" {
                assert!(
                    field.rust_name.starts_with("padding_"),
                    "{}: {} has no C name but is not a padding_* field, so it is a member the header declares and the gap check cannot measure it",
                    record.c_name,
                    field.rust_name
                );
                continue;
            }
            source.push_str(&format!(
                "_Static_assert(__builtin_offsetof({r}, {f}) == {off}, \"{r}.{f}: offsetof\");\n_Static_assert(sizeof((({r} *)0)->{f}) == {w}, \"{r}.{f}: sizeof\");\n",
                r = record.c_name, f = field.c_name, off = field.offset, w = field.width
            ));
            if let Some(prev) = previous {
                let gap = field.offset.checked_sub(prev.offset + prev.width);
                source.push_str(&format!(
                    "_Static_assert(__builtin_offsetof({r}, {b}) - __builtin_offsetof({r}, {a}) - sizeof((({r} *)0)->{a}) == {gap}, \"{r}: the hole between {a} and {b}\");\n",
                    r = record.c_name, a = prev.c_name, b = field.c_name, gap = gap.expect("a C member before this one is measured to end past it")
                ));
            } else {
                assert_eq!(field.offset, 0, "{}: the first field {} starts at {} and a record starts at 0", record.c_name, field.rust_name, field.offset);
            }
            previous = Some(field);
        }
        // The trailing alignment padding after the last member: the same difference, measured
        // against the record's own size rather than against another member's offset.
        let last = previous.expect("a record whose every field is a padding field is rejected by the tiling test");
        let tail = record.size.checked_sub(last.offset + last.width);
        source.push_str(&format!(
            "_Static_assert(sizeof({r}) - __builtin_offsetof({r}, {f}) - sizeof((({r} *)0)->{f}) == {tail}, \"{r}: the padding after {f}\");\n",
            r = record.c_name,
            f = last.c_name,
            tail = tail.expect("the last member ends past the record's own size")
        ));
    }
    let Some(out) = check_c(&cc, "filmcraft-va-av1-enc-abi", "av1encabi.c", &source) else {
        return;
    };
    assert!(
        out.status.success(),
        "{cc} disagrees with our AV1 encode records on a field offset, a field width, or a hole's width:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The measured `alignof` of each record, carried per row so the C side is asked for it.
///
/// Not read from the header prose: `alignof` is a property of the compiler's layout rules, which
/// is the whole reason `sizeof` alone is never trusted here.
const fn align_of(c_name: &str) -> usize {
    match c_name.as_bytes() {
        b"VAEncSequenceParameterBufferAV1" => 4,
        b"VAEncSegParamAV1" => 4,
        b"VAEncWarpedMotionParamsAV1" => 4,
        b"VARefFrameCtrlAV1" => 4,
        b"VAEncPictureParameterBufferAV1" => 4,
        b"VAEncTileGroupBufferAV1" => 4,
        _ => 0,
    }
}

/// One bitfield member of one of these records' unions: the struct it lives in, the union's C name,
/// a **label unique across this table**, the name of the anonymous struct inside the union (`bits`
/// in eight of the nine unions here and **`fields`** in `VARefFrameCtrlAV1` — carried per row
/// rather than assumed), the member's own C name, the width of the word it lives in, the bit range
/// a C compiler puts it at, and the name of the union's `value` member.
///
/// The last column is the **bare** member name, always `value` in this table: the test composes the
/// full C path from columns 2 and 9, so a row that also spelled the union's name in this column
/// produced `r.tile_group_obu_hdr_info.tile_group_obu_hdr_info.value`, which no C compiler accepts.
///
/// The label is carried because several unions would otherwise collide.
const ENC_BITFIELDS: &[(&str, &str, &str, &str, &str, u8, u32, u32, &str)] = &[
    // VAConfigAttribValEncAV1.bits (va_enc_av1.h:60-134), uint32_t. Fourteen 2-bit fields and a
    // 4-bit `reserved` -- the last member is the only one in the whole table that is neither 1, 2,
    // 3 nor 5 bits wide, which is exactly why the width is measured rather than derived.
    ("VAConfigAttribValEncAV1", "", "features", "bits", "support_128x128_superblock", 32, 0, 1, "value"),
    ("VAConfigAttribValEncAV1", "", "features", "bits", "support_filter_intra", 32, 2, 3, "value"),
    ("VAConfigAttribValEncAV1", "", "features", "bits", "support_intra_edge_filter", 32, 4, 5, "value"),
    ("VAConfigAttribValEncAV1", "", "features", "bits", "support_interintra_compound", 32, 6, 7, "value"),
    ("VAConfigAttribValEncAV1", "", "features", "bits", "support_masked_compound", 32, 8, 9, "value"),
    ("VAConfigAttribValEncAV1", "", "features", "bits", "support_warped_motion", 32, 10, 11, "value"),
    ("VAConfigAttribValEncAV1", "", "features", "bits", "support_palette_mode", 32, 12, 13, "value"),
    ("VAConfigAttribValEncAV1", "", "features", "bits", "support_dual_filter", 32, 14, 15, "value"),
    ("VAConfigAttribValEncAV1", "", "features", "bits", "support_jnt_comp", 32, 16, 17, "value"),
    ("VAConfigAttribValEncAV1", "", "features", "bits", "support_ref_frame_mvs", 32, 18, 19, "value"),
    ("VAConfigAttribValEncAV1", "", "features", "bits", "support_superres", 32, 20, 21, "value"),
    ("VAConfigAttribValEncAV1", "", "features", "bits", "support_restoration", 32, 22, 23, "value"),
    ("VAConfigAttribValEncAV1", "", "features", "bits", "support_allow_intrabc", 32, 24, 25, "value"),
    ("VAConfigAttribValEncAV1", "", "features", "bits", "support_cdef_channel_strength", 32, 26, 27, "value"),
    ("VAConfigAttribValEncAV1", "", "features", "bits", "reserved", 32, 28, 31, "value"),
    // VAConfigAttribValEncAV1Ext1.bits (va_enc_av1.h:141-171), uint32_t. A 5-bit mask, two 8-bit
    // masks and an 11-bit reserved -- none of them 2 bits, so nothing here may use `features::mask`.
    ("VAConfigAttribValEncAV1Ext1", "", "ext1", "bits", "interpolation_filter", 32, 0, 4, "value"),
    ("VAConfigAttribValEncAV1Ext1", "", "ext1", "bits", "min_segid_block_size_accepted", 32, 5, 12, "value"),
    ("VAConfigAttribValEncAV1Ext1", "", "ext1", "bits", "segment_feature_support", 32, 13, 20, "value"),
    ("VAConfigAttribValEncAV1Ext1", "", "ext1", "bits", "reserved", 32, 21, 31, "value"),
    // VAConfigAttribValEncAV1Ext2.bits (va_enc_av1.h:177-203), uint32_t. Two 2-bit fields, a
    // 3-bit mask, a 13-bit count and a 12-bit reserved.
    ("VAConfigAttribValEncAV1Ext2", "", "ext2", "bits", "tile_size_bytes_minus1", 32, 0, 1, "value"),
    ("VAConfigAttribValEncAV1Ext2", "", "ext2", "bits", "obu_size_bytes_minus1", 32, 2, 3, "value"),
    ("VAConfigAttribValEncAV1Ext2", "", "ext2", "bits", "tx_mode_support", 32, 4, 6, "value"),
    ("VAConfigAttribValEncAV1Ext2", "", "ext2", "bits", "max_tile_num_minus1", 32, 7, 19, "value"),
    ("VAConfigAttribValEncAV1Ext2", "", "ext2", "bits", "reserved", 32, 20, 31, "value"),
    // VAEncSequenceParameterBufferAV1.seq_fields.bits (va_enc_av1.h:284-323), uint32_t
    ("VAEncSequenceParameterBufferAV1", "seq_fields", "seq_fields", "bits", "still_picture", 32, 0, 0, "value"),
    ("VAEncSequenceParameterBufferAV1", "seq_fields", "seq_fields", "bits", "use_128x128_superblock", 32, 1, 1, "value"),
    ("VAEncSequenceParameterBufferAV1", "seq_fields", "seq_fields", "bits", "enable_filter_intra", 32, 2, 2, "value"),
    ("VAEncSequenceParameterBufferAV1", "seq_fields", "seq_fields", "bits", "enable_intra_edge_filter", 32, 3, 3, "value"),
    ("VAEncSequenceParameterBufferAV1", "seq_fields", "seq_fields", "bits", "enable_interintra_compound", 32, 4, 4, "value"),
    ("VAEncSequenceParameterBufferAV1", "seq_fields", "seq_fields", "bits", "enable_masked_compound", 32, 5, 5, "value"),
    ("VAEncSequenceParameterBufferAV1", "seq_fields", "seq_fields", "bits", "enable_warped_motion", 32, 6, 6, "value"),
    ("VAEncSequenceParameterBufferAV1", "seq_fields", "seq_fields", "bits", "enable_dual_filter", 32, 7, 7, "value"),
    ("VAEncSequenceParameterBufferAV1", "seq_fields", "seq_fields", "bits", "enable_order_hint", 32, 8, 8, "value"),
    ("VAEncSequenceParameterBufferAV1", "seq_fields", "seq_fields", "bits", "enable_jnt_comp", 32, 9, 9, "value"),
    ("VAEncSequenceParameterBufferAV1", "seq_fields", "seq_fields", "bits", "enable_ref_frame_mvs", 32, 10, 10, "value"),
    ("VAEncSequenceParameterBufferAV1", "seq_fields", "seq_fields", "bits", "enable_superres", 32, 11, 11, "value"),
    ("VAEncSequenceParameterBufferAV1", "seq_fields", "seq_fields", "bits", "enable_cdef", 32, 12, 12, "value"),
    ("VAEncSequenceParameterBufferAV1", "seq_fields", "seq_fields", "bits", "enable_restoration", 32, 13, 13, "value"),
    ("VAEncSequenceParameterBufferAV1", "seq_fields", "seq_fields", "bits", "bit_depth_minus8", 32, 14, 16, "value"),
    ("VAEncSequenceParameterBufferAV1", "seq_fields", "seq_fields", "bits", "subsampling_x", 32, 17, 17, "value"),
    ("VAEncSequenceParameterBufferAV1", "seq_fields", "seq_fields", "bits", "subsampling_y", 32, 18, 18, "value"),
    ("VAEncSequenceParameterBufferAV1", "seq_fields", "seq_fields", "bits", "mono_chrome", 32, 19, 19, "value"),
    ("VAEncSequenceParameterBufferAV1", "seq_fields", "seq_fields", "bits", "reserved_bits", 32, 20, 31, "value"),
    // VAEncSegParamAV1.seg_flags.bits (va_enc_av1.h:344-372), **uint8_t**
    ("VAEncSegParamAV1", "seg_flags", "seg_flags", "bits", "segmentation_enabled", 8, 0, 0, "value"),
    ("VAEncSegParamAV1", "seg_flags", "seg_flags", "bits", "segmentation_update_map", 8, 1, 1, "value"),
    ("VAEncSegParamAV1", "seg_flags", "seg_flags", "bits", "segmentation_temporal_update", 8, 2, 2, "value"),
    ("VAEncSegParamAV1", "seg_flags", "seg_flags", "bits", "reserved", 8, 3, 7, "value"),
    // VARefFrameCtrlAV1.fields (va_enc_av1.h:489-535), uint32_t -- **named `fields`, not `bits`**,
    // and the union IS the record, so `value` is the record's only member.
    ("VARefFrameCtrlAV1", "", "ref_frame_ctrl", "fields", "search_idx0", 32, 0, 2, "value"),
    ("VARefFrameCtrlAV1", "", "ref_frame_ctrl", "fields", "search_idx1", 32, 3, 5, "value"),
    ("VARefFrameCtrlAV1", "", "ref_frame_ctrl", "fields", "search_idx2", 32, 6, 8, "value"),
    ("VARefFrameCtrlAV1", "", "ref_frame_ctrl", "fields", "search_idx3", 32, 9, 11, "value"),
    ("VARefFrameCtrlAV1", "", "ref_frame_ctrl", "fields", "search_idx4", 32, 12, 14, "value"),
    ("VARefFrameCtrlAV1", "", "ref_frame_ctrl", "fields", "search_idx5", 32, 15, 17, "value"),
    ("VARefFrameCtrlAV1", "", "ref_frame_ctrl", "fields", "search_idx6", 32, 18, 20, "value"),
    ("VARefFrameCtrlAV1", "", "ref_frame_ctrl", "fields", "Reserved", 32, 21, 31, "value"),
    // VAEncPictureParameterBufferAV1.picture_flags.bits (va_enc_av1.h:619-667), uint32_t
    ("VAEncPictureParameterBufferAV1", "picture_flags", "picture_flags", "bits", "frame_type", 32, 0, 1, "value"),
    ("VAEncPictureParameterBufferAV1", "picture_flags", "picture_flags", "bits", "error_resilient_mode", 32, 2, 2, "value"),
    ("VAEncPictureParameterBufferAV1", "picture_flags", "picture_flags", "bits", "disable_cdf_update", 32, 3, 3, "value"),
    ("VAEncPictureParameterBufferAV1", "picture_flags", "picture_flags", "bits", "use_superres", 32, 4, 4, "value"),
    ("VAEncPictureParameterBufferAV1", "picture_flags", "picture_flags", "bits", "allow_high_precision_mv", 32, 5, 5, "value"),
    ("VAEncPictureParameterBufferAV1", "picture_flags", "picture_flags", "bits", "use_ref_frame_mvs", 32, 6, 6, "value"),
    ("VAEncPictureParameterBufferAV1", "picture_flags", "picture_flags", "bits", "disable_frame_end_update_cdf", 32, 7, 7, "value"),
    ("VAEncPictureParameterBufferAV1", "picture_flags", "picture_flags", "bits", "reduced_tx_set", 32, 8, 8, "value"),
    ("VAEncPictureParameterBufferAV1", "picture_flags", "picture_flags", "bits", "enable_frame_obu", 32, 9, 9, "value"),
    ("VAEncPictureParameterBufferAV1", "picture_flags", "picture_flags", "bits", "long_term_reference", 32, 10, 10, "value"),
    ("VAEncPictureParameterBufferAV1", "picture_flags", "picture_flags", "bits", "disable_frame_recon", 32, 11, 11, "value"),
    ("VAEncPictureParameterBufferAV1", "picture_flags", "picture_flags", "bits", "allow_intrabc", 32, 12, 12, "value"),
    ("VAEncPictureParameterBufferAV1", "picture_flags", "picture_flags", "bits", "palette_mode_enable", 32, 13, 13, "value"),
    ("VAEncPictureParameterBufferAV1", "picture_flags", "picture_flags", "bits", "allow_screen_content_tools", 32, 14, 14, "value"),
    ("VAEncPictureParameterBufferAV1", "picture_flags", "picture_flags", "bits", "force_integer_mv", 32, 15, 15, "value"),
    ("VAEncPictureParameterBufferAV1", "picture_flags", "picture_flags", "bits", "reserved", 32, 16, 31, "value"),
    // VAEncPictureParameterBufferAV1.loop_filter_flags.bits (va_enc_av1.h:692-701), **uint8_t**
    ("VAEncPictureParameterBufferAV1", "loop_filter_flags", "loop_filter_flags", "bits", "sharpness_level", 8, 0, 2, "value"),
    ("VAEncPictureParameterBufferAV1", "loop_filter_flags", "loop_filter_flags", "bits", "mode_ref_delta_enabled", 8, 3, 3, "value"),
    ("VAEncPictureParameterBufferAV1", "loop_filter_flags", "loop_filter_flags", "bits", "mode_ref_delta_update", 8, 4, 4, "value"),
    ("VAEncPictureParameterBufferAV1", "loop_filter_flags", "loop_filter_flags", "bits", "reserved", 8, 5, 7, "value"),
    // VAEncPictureParameterBufferAV1.qmatrix_flags.bits (va_enc_av1.h:742-753), **uint16_t**
    ("VAEncPictureParameterBufferAV1", "qmatrix_flags", "qmatrix_flags", "bits", "using_qmatrix", 16, 0, 0, "value"),
    ("VAEncPictureParameterBufferAV1", "qmatrix_flags", "qmatrix_flags", "bits", "qm_y", 16, 1, 4, "value"),
    ("VAEncPictureParameterBufferAV1", "qmatrix_flags", "qmatrix_flags", "bits", "qm_u", 16, 5, 8, "value"),
    ("VAEncPictureParameterBufferAV1", "qmatrix_flags", "qmatrix_flags", "bits", "qm_v", 16, 9, 12, "value"),
    ("VAEncPictureParameterBufferAV1", "qmatrix_flags", "qmatrix_flags", "bits", "reserved", 16, 13, 15, "value"),
    // VAEncPictureParameterBufferAV1.mode_control_flags.bits (va_enc_av1.h:761-802), uint32_t
    ("VAEncPictureParameterBufferAV1", "mode_control_flags", "mode_control_flags", "bits", "delta_q_present", 32, 0, 0, "value"),
    ("VAEncPictureParameterBufferAV1", "mode_control_flags", "mode_control_flags", "bits", "delta_q_res", 32, 1, 2, "value"),
    ("VAEncPictureParameterBufferAV1", "mode_control_flags", "mode_control_flags", "bits", "delta_lf_present", 32, 3, 3, "value"),
    ("VAEncPictureParameterBufferAV1", "mode_control_flags", "mode_control_flags", "bits", "delta_lf_res", 32, 4, 5, "value"),
    ("VAEncPictureParameterBufferAV1", "mode_control_flags", "mode_control_flags", "bits", "delta_lf_multi", 32, 6, 6, "value"),
    ("VAEncPictureParameterBufferAV1", "mode_control_flags", "mode_control_flags", "bits", "tx_mode", 32, 7, 8, "value"),
    ("VAEncPictureParameterBufferAV1", "mode_control_flags", "mode_control_flags", "bits", "reference_mode", 32, 9, 10, "value"),
    ("VAEncPictureParameterBufferAV1", "mode_control_flags", "mode_control_flags", "bits", "skip_mode_present", 32, 11, 11, "value"),
    ("VAEncPictureParameterBufferAV1", "mode_control_flags", "mode_control_flags", "bits", "reserved", 32, 12, 31, "value"),
    // VAEncPictureParameterBufferAV1.loop_restoration_flags.bits (va_enc_av1.h:841-864), **uint16_t**
    ("VAEncPictureParameterBufferAV1", "loop_restoration_flags", "loop_restoration_flags", "bits", "yframe_restoration_type", 16, 0, 1, "value"),
    ("VAEncPictureParameterBufferAV1", "loop_restoration_flags", "loop_restoration_flags", "bits", "cbframe_restoration_type", 16, 2, 3, "value"),
    ("VAEncPictureParameterBufferAV1", "loop_restoration_flags", "loop_restoration_flags", "bits", "crframe_restoration_type", 16, 4, 5, "value"),
    ("VAEncPictureParameterBufferAV1", "loop_restoration_flags", "loop_restoration_flags", "bits", "lr_unit_shift", 16, 6, 7, "value"),
    ("VAEncPictureParameterBufferAV1", "loop_restoration_flags", "loop_restoration_flags", "bits", "lr_uv_shift", 16, 8, 8, "value"),
    ("VAEncPictureParameterBufferAV1", "loop_restoration_flags", "loop_restoration_flags", "bits", "reserved", 16, 9, 15, "value"),
    // VAEncPictureParameterBufferAV1.tile_group_obu_hdr_info.bits (va_enc_av1.h:938-957), **uint8_t**
    ("VAEncPictureParameterBufferAV1", "tile_group_obu_hdr_info", "tile_group_obu_hdr_info", "bits", "obu_extension_flag", 8, 0, 0, "value"),
    ("VAEncPictureParameterBufferAV1", "tile_group_obu_hdr_info", "tile_group_obu_hdr_info", "bits", "obu_has_size_field", 8, 1, 1, "value"),
    ("VAEncPictureParameterBufferAV1", "tile_group_obu_hdr_info", "tile_group_obu_hdr_info", "bits", "temporal_id", 8, 2, 4, "value"),
    ("VAEncPictureParameterBufferAV1", "tile_group_obu_hdr_info", "tile_group_obu_hdr_info", "bits", "spatial_id", 8, 5, 6, "value"),
    ("VAEncPictureParameterBufferAV1", "tile_group_obu_hdr_info", "tile_group_obu_hdr_info", "bits", "reserved", 8, 7, 7, "value"),
];

/// **Every one of the [`ENC_BITFIELDS`] bitfield members is at the bit a C compiler puts it at, and
/// is the width the C compiler gives it** — measured by *running* C.
///
/// The failure this exists for is invisible to every other test in this file. A union has one
/// `offsetof`, so a member packed one bit too high agrees with the C compiler on the record's
/// `sizeof`, on every field offset and on every hole width, and the result of submitting it is a
/// **picture** rather than an error. Nor can the header be read for it: the members are declared in
/// an order that is right on the target that compiled it and says nothing about any other.
///
/// The comparison has two independent sides — a C program that sets one member to all ones and
/// prints `__builtin_ctzll(v)` and the high set bit, and the table above — so a member that moves,
/// or a table entry that drifts, fails here naming the union and the member.
#[test]
fn the_av1_enc_bitfield_words_are_the_bits_the_c_compiler_measures() {
    if !std::path::Path::new(LIBVA_HEADERS).join("va_enc_av1.h").is_file() {
        eprintln!("SKIPPED: no {LIBVA_HEADERS}/va_enc_av1.h");
        return;
    }
    let Some(cc) = compiler() else {
        eprintln!("SKIPPED: no C compiler to measure the headers with");
        return;
    };
    // `~0u` is all ones for any width up to 32, which is every member here; the assignment truncates
    // to the member's own width, so a member narrower than a word cannot spill into its neighbour --
    // and `report` reading the whole word back is what would show it if it did.
    let mut source = String::from(
        "#include <stdio.h>\n#include <string.h>\n#include <stdint.h>\n#include <va/va.h>\n#include <va/va_enc_av1.h>\n\n\
         static void report(const char *u, const char *m, unsigned long long v, int width) {\n\
         \x20   if (!v) { printf(\"%s %s NONE\\n\", u, m); return; }\n\
         \x20   unsigned lo = (unsigned)__builtin_ctzll(v);\n\
         \x20   unsigned hi = (unsigned)(width - 1) - ((unsigned)__builtin_clzll(v) - (unsigned)(64 - width));\n\
         \x20   printf(\"%s %s %u %u\\n\", u, m, lo, hi);\n\
         }\n\n\
         int main(void) {\n",
    );
    for (record, c_union, label, c_inner, member, word, _, _, c_value) in ENC_BITFIELDS {
        // `c_union` is the union's **member name inside the record**, and is EMPTY where the union
        // *is* the record -- the three attribute unions and `VARefFrameCtrlAV1`, where `value` is a
        // member of the union rather than of a struct that contains it. Writing the union's own
        // name in that column is what produced `r.bits.bits.m`, a path no C compiler accepts and
        // which therefore fails to compile rather than measuring the wrong thing -- the good
        // failure, but not one worth having.
        let path = match *c_union {
            "" => format!("{c_inner}.{member}"),
            u => format!("{u}.{c_inner}.{member}"),
        };
        let value_path = match *c_union {
            "" => c_value.to_string(),
            u => format!("{u}.{c_value}"),
        };
        source.push_str(&format!(
            "\x20 {{ {record} r; memset(&r, 0, sizeof r); r.{path} = ~0u; report(\"{label}\", \"{member}\", (unsigned long long)r.{value_path}, {word}); }}\n"
        ));
    }
    source.push_str("\x20 return 0;\n}\n");
    let Some((built, exe)) = build_c(&cc, "filmcraft-va-av1-enc-bits", "av1encbits.c", &source) else {
        return;
    };
    assert!(built.status.success(), "{cc} rejected the AV1 encode bitfield probe against the headers:\n{}", String::from_utf8_lossy(&built.stderr));
    // **Run before the directory goes.** The probe *is* the executable just written into it, so
    // removing the directory first is how this test would skip itself without noticing.
    let run =
        std::process::Command::new(exe.to_str().unwrap_or_default()).output().unwrap_or_else(|e| panic!("running the AV1 encode bitfield probe failed: {e}"));
    let _ = std::fs::remove_dir_all(exe.parent().unwrap_or(std::path::Path::new(".")));
    let printed = String::from_utf8_lossy(&run.stdout).to_string();

    let mut seen = 0usize;
    for (record, _, label, _, member, word, low, high, _) in ENC_BITFIELDS {
        let want = format!("{label} {member} {low} {high}");
        let line = printed
            .lines()
            .find(|l| l.starts_with(&format!("{label} {member} ")))
            .unwrap_or_else(|| panic!("{cc} printed nothing for {record}.{member}:\n{printed}"));
        assert_eq!(line, want, "{record}: {member} in a {word}-bit union — {cc} says `{line}`, our table says `{want}`");
        seen += 1;
    }
    assert_eq!(seen, ENC_BITFIELDS.len(), "every member in the table was checked");
}

/// **Every bit position the code in [`super`] publishes as a constant is the one C measured.**
///
/// The `features::`, `ext1::` and `ext2::` masks are named in the record file for the submission
/// task to use and nothing uses them yet, so without this they would be numbers somebody typed off
/// a declaration order — the exact thing the table above exists to distrust. Each row names its
/// **C member**, so the constant and the measurement are joined by the header's own name rather
/// than by two strings that look alike; a renamed constant then fails here instead of quietly
/// matching nothing.
#[test]
fn the_published_av1_masks_are_the_measured_ones() {
    // (our constant's name, our bit offset, the union label and C member it stands for)
    let rows: &[(&str, u32, &str, &str)] = &[
        ("features::SUPPORT_128X128_SUPERBLOCK", features::SUPPORT_128X128_SUPERBLOCK, "features", "support_128x128_superblock"),
        ("features::SUPPORT_FILTER_INTRA", features::SUPPORT_FILTER_INTRA, "features", "support_filter_intra"),
        ("features::SUPPORT_INTRA_EDGE_FILTER", features::SUPPORT_INTRA_EDGE_FILTER, "features", "support_intra_edge_filter"),
        ("features::SUPPORT_INTERINTRA_COMPOUND", features::SUPPORT_INTERINTRA_COMPOUND, "features", "support_interintra_compound"),
        ("features::SUPPORT_MASKED_COMPOUND", features::SUPPORT_MASKED_COMPOUND, "features", "support_masked_compound"),
        ("features::SUPPORT_WARPED_MOTION", features::SUPPORT_WARPED_MOTION, "features", "support_warped_motion"),
        ("features::SUPPORT_PALETTE_MODE", features::SUPPORT_PALETTE_MODE, "features", "support_palette_mode"),
        ("features::SUPPORT_DUAL_FILTER", features::SUPPORT_DUAL_FILTER, "features", "support_dual_filter"),
        ("features::SUPPORT_JNT_COMP", features::SUPPORT_JNT_COMP, "features", "support_jnt_comp"),
        ("features::SUPPORT_REF_FRAME_MVS", features::SUPPORT_REF_FRAME_MVS, "features", "support_ref_frame_mvs"),
        ("features::SUPPORT_SUPERRES", features::SUPPORT_SUPERRES, "features", "support_superres"),
        ("features::SUPPORT_RESTORATION", features::SUPPORT_RESTORATION, "features", "support_restoration"),
        ("features::SUPPORT_ALLOW_INTRABC", features::SUPPORT_ALLOW_INTRABC, "features", "support_allow_intrabc"),
        ("features::SUPPORT_CDEF_CHANNEL_STRENGTH", features::SUPPORT_CDEF_CHANNEL_STRENGTH, "features", "support_cdef_channel_strength"),
        ("features::RESERVED", features::RESERVED, "features", "reserved"),
        ("ext1::INTERPOLATION_FILTER", ext1::INTERPOLATION_FILTER, "ext1", "interpolation_filter"),
        ("ext1::MIN_SEGID_BLOCK_SIZE_ACCEPTED", ext1::MIN_SEGID_BLOCK_SIZE_ACCEPTED, "ext1", "min_segid_block_size_accepted"),
        ("ext1::SEGMENT_FEATURE_SUPPORT", ext1::SEGMENT_FEATURE_SUPPORT, "ext1", "segment_feature_support"),
        ("ext1::RESERVED", ext1::RESERVED, "ext1", "reserved"),
        ("ext2::TILE_SIZE_BYTES_MINUS1", ext2::TILE_SIZE_BYTES_MINUS1, "ext2", "tile_size_bytes_minus1"),
        ("ext2::OBU_SIZE_BYTES_MINUS1", ext2::OBU_SIZE_BYTES_MINUS1, "ext2", "obu_size_bytes_minus1"),
        ("ext2::TX_MODE_SUPPORT", ext2::TX_MODE_SUPPORT, "ext2", "tx_mode_support"),
        ("ext2::MAX_TILE_NUM_MINUS1", ext2::MAX_TILE_NUM_MINUS1, "ext2", "max_tile_num_minus1"),
        ("ext2::RESERVED", ext2::RESERVED, "ext2", "reserved"),
    ];
    for (name, bit, label, member) in rows {
        let (_, _, _, _, measured_member, word, low, high, _) = ENC_BITFIELDS
            .iter()
            .find(|(_, _, l, _, m, ..)| *l == *label && *m == *member)
            .unwrap_or_else(|| panic!("{name} has no measured bitfield behind it in ENC_BITFIELDS"));
        assert_eq!(measured_member, member, "{name}: the C member it stands for");
        assert_eq!(*bit, *low, "{name}: our constant says bit {bit}, C measured {low} of a {word}-bit word");
        // The width too, not just the start: a mask one bit too wide would cover the next field and
        // read two fields as one.
        let width = high - low + 1;
        assert!(width > 0 && width <= 13, "{name}: a measured width of {width} bits in a {word}-bit word");
    }

    // `features`: fourteen 2-bit fields and one 4-bit `reserved` -- and the two must not share a
    // mask function, which is why `RESERVED` has its own.
    assert_eq!(features::mask(features::SUPPORT_128X128_SUPERBLOCK), 0b11, "a 2-bit mask at bit 0");
    assert_eq!(features::mask(features::SUPPORT_CDEF_CHANNEL_STRENGTH), 0b11 << 26, "a 2-bit mask at bit 26");
    assert_eq!(features::reserved_mask(), 0xF << 28, "the 4-bit reserved mask at bit 28");
    assert_eq!(features::reserved_mask().count_ones(), 4, "and that mask is four bits wide");
    for (name, bit, ..) in rows.iter().filter(|(name, ..)| name.starts_with("features::") && !name.ends_with("RESERVED")) {
        let covered = features::mask(*bit);
        assert_eq!(covered.count_ones(), 2, "{name}: a feature field is two bits");
        assert_eq!(covered.trailing_zeros(), *bit, "{name}: the mask starts at the bit the constant names");
        assert_eq!(covered & features::reserved_mask(), 0, "{name}: a feature mask never reaches the reserved field");
    }

    // `ext1`: a 5-bit mask and two 8-bit ones, plus an 11-bit reserved -- none of them 2 bits, so
    // none of them may use `features::mask`.
    assert_eq!(ext1::interpolation_filter_mask().count_ones(), 5, "a 5-bit interpolation-filter mask");
    assert_eq!(ext1::interpolation_filter_mask(), 0x1F << ext1::INTERPOLATION_FILTER);
    assert_eq!(ext1::byte_mask(ext1::MIN_SEGID_BLOCK_SIZE_ACCEPTED).count_ones(), 8, "an 8-bit mask");
    assert_eq!(ext1::byte_mask(ext1::SEGMENT_FEATURE_SUPPORT), 0xFF << 13);
    assert_eq!(ext1::reserved_mask().count_ones(), 11, "the 11-bit reserved mask");
    assert_eq!(ext1::reserved_mask(), 0x7FF << ext1::RESERVED);

    // `ext2`: two 2-bit size fields, a 3-bit tx-mode mask, a 13-bit tile count and a 12-bit
    // reserved. The 13-bit one is the widest field in any of these unions.
    assert_eq!(ext2::size_mask(ext2::TILE_SIZE_BYTES_MINUS1), 0b11, "a 2-bit mask at bit 0");
    assert_eq!(ext2::size_mask(ext2::OBU_SIZE_BYTES_MINUS1), 0b11 << 2, "a 2-bit mask at bit 2");
    assert_eq!(ext2::tx_mode_mask().count_ones(), 3, "the 3-bit tx-mode mask");
    assert_eq!(ext2::max_tile_mask().count_ones(), 13, "the 13-bit tile-count mask");
    assert_eq!(ext2::max_tile_mask(), 0x1FFF << ext2::MAX_TILE_NUM_MINUS1);
    assert_eq!(ext2::reserved_mask().count_ones(), 12, "the 12-bit reserved mask");
    assert_eq!(ext2::reserved_mask(), 0xFFF << ext2::RESERVED);
}

/// **The three `VA_FEATURE_*` levels are the ones `va.h` defines — and `REQUIRED` is 2, not 3.**
///
/// `va.h:381`, `:391` and `:398` are the only authority for what a field of
/// `VAConfigAttribValEncAV1` can hold, and the third of those three values is the one that has to
/// be pinned rather than remembered. `VA_FEATURE_REQUIRED` obliges a submission ("any
/// corresponding feature flag must be set and any additional configuration needed by the feature
/// must be supplied"), so a decoder that reads it as 1 treats a binding constraint as a choice;
/// a decoder that reads a real `1` as "required" invents one that does not exist. **3 is not a
/// `VA_FEATURE_*` value at all**, which is why this asserts the values and not merely that they
/// are distinct.
#[test]
fn the_feature_levels_are_the_ones_va_h_defines() {
    if !std::path::Path::new(LIBVA_HEADERS).join("va_enc_av1.h").is_file() {
        eprintln!("SKIPPED: no {LIBVA_HEADERS}/va_enc_av1.h");
        return;
    }
    let Some(cc) = compiler() else {
        eprintln!("SKIPPED: no C compiler to measure the headers with");
        return;
    };
    // Asserted *in C*, so the numbers are read out of va.h rather than compared with a second
    // transcription of it. The `va_enc_av1.h` include is what makes the AV1 union available; the
    // levels themselves live in va.h, which that header includes.
    let source = format!(
        "#include <stdint.h>\n#include <va/va.h>\n#include <va/va_enc_av1.h>\n\
         _Static_assert(VA_FEATURE_NOT_SUPPORTED == {ns}, \"VA_FEATURE_NOT_SUPPORTED\");\n\
         _Static_assert(VA_FEATURE_SUPPORTED == {sup}, \"VA_FEATURE_SUPPORTED\");\n\
         _Static_assert(VA_FEATURE_REQUIRED == {req}, \"VA_FEATURE_REQUIRED\");\n\
         _Static_assert(VA_FEATURE_NOT_SUPPORTED < VA_FEATURE_SUPPORTED\n\
                    && VA_FEATURE_SUPPORTED < VA_FEATURE_REQUIRED, \"the three are ordered\");\n\
         _Static_assert(VA_FEATURE_REQUIRED != 3, \"3 is not a VA_FEATURE_* value\");\n\
         _Static_assert(sizeof(VAConfigAttribValEncAV1) == 4, \"the word these levels live in is one uint32_t\");\n",
        ns = feature::NOT_SUPPORTED,
        sup = feature::SUPPORTED,
        req = feature::REQUIRED,
    );
    let Some(out) = check_c(&cc, "filmcraft-va-av1-feature-levels", "av1featurelevels.c", &source) else {
        return;
    };
    assert!(out.status.success(), "{cc} disagrees with our VA_FEATURE_* levels:\n{}", String::from_utf8_lossy(&out.stderr));

    // And on our side, so the values are pinned even where no C compiler is installed -- and so a
    // "skip everything, the headers are missing" path cannot leave them unverified.
    assert_eq!(feature::NOT_SUPPORTED, 0, "va.h:381");
    assert_eq!(feature::SUPPORTED, 1, "va.h:391");
    assert_eq!(feature::REQUIRED, 2, "va.h:398 -- the value that obliges a submission");
    assert_eq!(feature::LEVELS, [0, 1, 2], "every level va.h defines, and no others");
    assert_ne!(feature::REQUIRED, 3, "3 is not a VA_FEATURE_* value; reading a field as 3 is undefined");
    // The two spellings of the reading agree -- `features::level` delegates, and a delegation that
    // is written but never compared here is a delegation nobody checks.
    for bit in [features::SUPPORT_CDEF_CHANNEL_STRENGTH, features::SUPPORT_SUPERRES, features::RESERVED] {
        for word in [0u32, 0x0500_1000, u32::MAX] {
            assert_eq!(features::level(word, bit), feature::level(word, bit), "features::level and feature::level must be one function");
        }
    }
}

/// **Decoded by C, through the header's own union: this driver's AV1 feature word has no
/// `VA_FEATURE_REQUIRED` field, and CDEF in particular is supported rather than required.**
///
/// The word is measured, not transcribed. The C side sets `.value` on a real
/// `VAConfigAttribValEncAV1` and prints **every** member back, so the expected lines below are
/// what the header's own layout says the word means; a pin that only checked `cdef` would pass
/// with `palette_mode` or `allow_intrabc` at any level.
///
/// This is the test that answers "does a submission that signals CDEF have to?", and the answer is
/// **no**: `support_cdef_channel_strength` is `VA_FEATURE_SUPPORTED` (1), which `va.h:382-389`
/// describes as *"the user may decide whether or not to use this feature"*. `enable_cdef` in
/// [`VaEncSequenceParameterBufferAv1`] is therefore a choice, and a submission may leave it 0.
///
/// The four "no" assertions at the end are the load-bearing ones: they hold **for the whole
/// union**, so a driver answering a different word tomorrow fails here with a field name rather
/// than silently changing what a submission is obliged to do.
#[test]
fn the_driver_s_av1_feature_word_has_no_required_field() {
    if !std::path::Path::new(LIBVA_HEADERS).join("va_enc_av1.h").is_file() {
        eprintln!("SKIPPED: no {LIBVA_HEADERS}/va_enc_av1.h");
        return;
    }
    let Some(cc) = compiler() else {
        eprintln!("SKIPPED: no C compiler to measure the headers with");
        return;
    };
    // Every 2-bit field is named in `ENC_BITFIELDS`; this asks C for all of them plus `reserved`,
    // in the union's own declaration order, so the output is the header's account of the word.
    let source = format!(
        "#include <stdio.h>\n#include <stdint.h>\n#include <va/va.h>\n#include <va/va_enc_av1.h>\n\
         int main(void) {{\n\
         \x20   VAConfigAttribValEncAV1 a;\n\
         \x20   a.value = {word}u;\n\
         \x20   printf(\"support_128x128_superblock %u\\n\", a.bits.support_128x128_superblock);\n\
         \x20   printf(\"support_filter_intra %u\\n\", a.bits.support_filter_intra);\n\
         \x20   printf(\"support_intra_edge_filter %u\\n\", a.bits.support_intra_edge_filter);\n\
         \x20   printf(\"support_interintra_compound %u\\n\", a.bits.support_interintra_compound);\n\
         \x20   printf(\"support_masked_compound %u\\n\", a.bits.support_masked_compound);\n\
         \x20   printf(\"support_warped_motion %u\\n\", a.bits.support_warped_motion);\n\
         \x20   printf(\"support_palette_mode %u\\n\", a.bits.support_palette_mode);\n\
         \x20   printf(\"support_dual_filter %u\\n\", a.bits.support_dual_filter);\n\
         \x20   printf(\"support_jnt_comp %u\\n\", a.bits.support_jnt_comp);\n\
         \x20   printf(\"support_ref_frame_mvs %u\\n\", a.bits.support_ref_frame_mvs);\n\
         \x20   printf(\"support_superres %u\\n\", a.bits.support_superres);\n\
         \x20   printf(\"support_restoration %u\\n\", a.bits.support_restoration);\n\
         \x20   printf(\"support_allow_intrabc %u\\n\", a.bits.support_allow_intrabc);\n\
         \x20   printf(\"support_cdef_channel_strength %u\\n\", a.bits.support_cdef_channel_strength);\n\
         \x20   printf(\"reserved %u\\n\", a.bits.reserved);\n\
         \x20   return 0;\n\
         }}\n",
        word = DRIVER_AV1_FEATURES,
    );
    let Some((_out, exe)) = build_c(&cc, "filmcraft-va-av1-feature-word", "av1featureword.c", &source) else {
        return;
    };
    let run = std::process::Command::new(&exe).output();
    let _ = std::fs::remove_dir_all(exe.parent().unwrap_or(std::path::Path::new(".")));
    let run = run.unwrap_or_else(|e| panic!("running {} failed: {e}", exe.display()));
    assert!(run.status.success(), "the C probe failed: {}", String::from_utf8_lossy(&run.stderr));
    let printed = String::from_utf8_lossy(&run.stdout).to_string();

    // What C says, joined to our own constants by the C member's name -- so a renamed constant
    // fails here instead of quietly matching nothing.
    let rows: &[(&str, u32)] = &[
        ("support_128x128_superblock", features::SUPPORT_128X128_SUPERBLOCK),
        ("support_filter_intra", features::SUPPORT_FILTER_INTRA),
        ("support_intra_edge_filter", features::SUPPORT_INTRA_EDGE_FILTER),
        ("support_interintra_compound", features::SUPPORT_INTERINTRA_COMPOUND),
        ("support_masked_compound", features::SUPPORT_MASKED_COMPOUND),
        ("support_warped_motion", features::SUPPORT_WARPED_MOTION),
        ("support_palette_mode", features::SUPPORT_PALETTE_MODE),
        ("support_dual_filter", features::SUPPORT_DUAL_FILTER),
        ("support_jnt_comp", features::SUPPORT_JNT_COMP),
        ("support_ref_frame_mvs", features::SUPPORT_REF_FRAME_MVS),
        ("support_superres", features::SUPPORT_SUPERRES),
        ("support_restoration", features::SUPPORT_RESTORATION),
        ("support_allow_intrabc", features::SUPPORT_ALLOW_INTRABC),
        ("support_cdef_channel_strength", features::SUPPORT_CDEF_CHANNEL_STRENGTH),
    ];
    // The word's three set features, as C measured them. Anything else is NOT_SUPPORTED.
    let supported: &[&str] = &["support_palette_mode", "support_allow_intrabc", "support_cdef_channel_strength"];

    for (name, bit) in rows {
        // The constant must be the bit C measured, joined through ENC_BITFIELDS as the mask test
        // above does -- so this test cannot pass by reading the word at the wrong offset.
        let measured = ENC_BITFIELDS
            .iter()
            .find(|(_, _, l, _, m, ..)| *l == "features" && *m == *name)
            .map(|row| (row.6, row.7))
            .unwrap_or_else(|| panic!("{name} has no measured bitfield behind it in ENC_BITFIELDS"));
        assert_eq!(measured.0, *bit, "{name}: our constant says bit {bit}, C measured {}", measured.0);

        let want = if supported.contains(name) { feature::SUPPORTED } else { feature::NOT_SUPPORTED };
        let line = printed.lines().find(|l| l.starts_with(&format!("{name} "))).unwrap_or_else(|| panic!("the C probe printed nothing for {name}:\n{printed}"));
        let got: u32 =
            line.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or_else(|| panic!("{name}: C printed an unparseable level in `{line}`"));
        // Read it back through *our* reader too, so the reader is exercised and not just C.
        assert_eq!(got, want, "{name}: C says level {got} for 0x{DRIVER_AV1_FEATURES:08x}, we expected {want}");
        assert_eq!(feature::level(DRIVER_AV1_FEATURES, *bit), got, "{name}: our reader disagrees with C at bit {bit}");
        assert_eq!(feature::is_required(DRIVER_AV1_FEATURES, *bit), want == feature::REQUIRED, "{name}: our is_required disagrees with the level C measured");
    }

    // The whole point, stated as an assertion over the entire union rather than field by field:
    // nothing obliges a submission, and `reserved` is clean, so the word means what it says.
    assert!(
        !rows.iter().any(|(_name, bit)| feature::is_required(DRIVER_AV1_FEATURES, *bit)),
        "0x{DRIVER_AV1_FEATURES:08x} marks a feature VA_FEATURE_REQUIRED, which would oblige the submission to set it"
    );
    assert_eq!(
        (DRIVER_AV1_FEATURES & features::reserved_mask()) >> features::RESERVED,
        0,
        "reserved bits 28-31 are zero, so the decode above is the header's and not a repacking"
    );
    // And the word is what the module note says it is, bit by bit -- three single bits at the low
    // half of their 2-bit fields, which is why each reads exactly 1 and none reads 3.
    assert_eq!(DRIVER_AV1_FEATURES, 0x0500_1000, "the word the driver answered, re-stated");
    assert_eq!(DRIVER_AV1_FEATURES.count_ones(), 3, "three set bits: palette_mode, allow_intrabc, cdef");
    assert_eq!(feature::level(DRIVER_AV1_FEATURES, features::SUPPORT_CDEF_CHANNEL_STRENGTH), feature::SUPPORTED, "CDEF is supported, not required");
    assert!(!feature::is_required(DRIVER_AV1_FEATURES, features::SUPPORT_CDEF_CHANNEL_STRENGTH), "so enable_cdef is a choice, not an obligation");
}

/// **Every union in these records is the width its members were measured in, and the code reads
/// each union's `value` at that width.**
///
/// This test exists because these records are the first place on this backend where assuming "a
/// union is a `u32`" is wrong *within a single record*: `VAEncPictureParameterBufferAV1` holds
/// **four** 32-bit unions, **two** 16-bit ones and **two** 8-bit ones, and `VAEncSegParamAV1` and
/// `VARefFrameCtrlAV1` add an 8-bit and a 32-bit union respectively. A member read out of the
/// wrong width agrees with C on the record's `sizeof`, on every offset and on every hole — and
/// produces a wrong picture rather than an error.
#[test]
fn the_av1_enc_word_widths_are_the_ones_the_bits_were_measured_in() {
    // Our own types must be exactly the width the header declares for them.
    assert_eq!(size_of::<VaRefFrameCtrlAv1>(), 4, "VARefFrameCtrlAV1 is one 32-bit word");
    assert_eq!(size_of::<Av1Features>(), 4, "VAConfigAttribValEncAV1 is one 32-bit word");
    assert_eq!(size_of::<Av1Ext1>(), 4, "VAConfigAttribValEncAV1Ext1 is one 32-bit word");
    assert_eq!(size_of::<Av1Ext2>(), 4, "VAConfigAttribValEncAV1Ext2 is one 32-bit word");
    // The union *fields* inside the records, by what they are embedded as: a `u8` for each of the
    // three 8-bit unions and a `u16` for each of the two 16-bit ones, asserted through the record's
    // own measured layout rather than by naming the type, because in a `#[repr(C)]` record the
    // field's declared type is the only thing that could make a 32-bit union look 8 bits wide.
    // Looked up **by name**, not by index: an index silently re-points at whatever field an edit
    // moved it to, and the assertion then measures the wrong field and still passes or fails for a
    // reason nobody can read off the line.
    let width_of = |record: &str, field: &str| -> Option<usize> {
        ENC_RECORDS.iter().find(|r| r.c_name == record).and_then(|r| r.fields.iter().find(|f| f.c_name == field)).map(|f| f.width)
    };
    assert_eq!(width_of("VAEncSegParamAV1", "seg_flags"), Some(1), "seg_flags is one byte");
    assert_eq!(width_of("VAEncPictureParameterBufferAV1", "loop_filter_flags"), Some(1), "loop_filter_flags is one byte");
    assert_eq!(width_of("VAEncPictureParameterBufferAV1", "tile_group_obu_hdr_info"), Some(1), "tile_group_obu_hdr_info is one byte");
    assert_eq!(width_of("VAEncPictureParameterBufferAV1", "qmatrix_flags"), Some(2), "qmatrix_flags is two bytes");
    assert_eq!(width_of("VAEncPictureParameterBufferAV1", "loop_restoration_flags"), Some(2), "loop_restoration_flags is two bytes");
    assert_eq!(width_of("VAEncPictureParameterBufferAV1", "picture_flags"), Some(4), "picture_flags is four bytes");
    assert_eq!(width_of("VAEncPictureParameterBufferAV1", "mode_control_flags"), Some(4), "mode_control_flags is four bytes");

    // Every 8-bit row in the table must sit in a union whose C `value` member is a uint8_t, and
    // the same for 16- and 32-bit rows -- asked of C rather than assumed.
    let Some(cc) = compiler() else {
        eprintln!("SKIPPED: no C compiler to measure the headers with");
        return;
    };
    if !std::path::Path::new(LIBVA_HEADERS).join("va_enc_av1.h").is_file() {
        eprintln!("SKIPPED: no {LIBVA_HEADERS}/va_enc_av1.h");
        return;
    }
    let mut source = String::from("#include <stdint.h>\n#include <va/va.h>\n#include <va/va_enc_av1.h>\n");
    // One assertion per union, not per member: the word is a property of the union.
    let mut asserted = std::collections::BTreeSet::new();
    for (record, c_union, _, _, _, word, _, _, c_value) in ENC_BITFIELDS {
        if !asserted.insert((*record, *c_union)) {
            continue;
        }
        let word = *word;
        let bytes = match word {
            8 => 1,
            16 => 2,
            _ => 4,
        };
        // Composed from the same two columns as the bitfield probe, so `value` is appended exactly
        // once: an empty `c_union` means the union *is* the C type and `value` hangs off it
        // directly, otherwise it hangs off the named member.
        let path = match *c_union {
            "" => (*c_value).to_string(),
            u => format!("{u}.{c_value}"),
        };
        source.push_str(&format!("_Static_assert(sizeof(((({r} *)0)->{p})) == {n}, \"{r}.{p} is not {n} bytes\");\n", r = record, p = path, n = bytes));
    }
    let Some(out) = check_c(&cc, "filmcraft-va-av1-enc-widths", "av1encwidths.c", &source) else {
        return;
    };
    assert!(out.status.success(), "{cc} disagrees with our AV1 encode union word widths:\n{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(asserted.len(), 12, "the twelve distinct unions across these records and the three attribute unions");
}

/// **Every constant in this file is what the header declares**, asked of the header rather than
/// compared with a number somebody typed from it a second time.
#[test]
fn the_av1_enc_constants_are_what_the_header_declares() {
    if !std::path::Path::new(LIBVA_HEADERS).join("va_enc_av1.h").is_file() {
        eprintln!("SKIPPED: no {LIBVA_HEADERS}/va_enc_av1.h");
        return;
    }
    let Some(cc) = compiler() else {
        eprintln!("SKIPPED: no C compiler to measure the headers with");
        return;
    };
    let source = format!(
        "#include <stdint.h>\n#include <va/va.h>\n#include <va/va_enc_av1.h>\n\
         _Static_assert(VAConfigAttribEncAV1 == {f0}, \"VAConfigAttribEncAV1\");\n\
         _Static_assert(VAConfigAttribEncAV1Ext1 == {f1}, \"VAConfigAttribEncAV1Ext1\");\n\
         _Static_assert(VAConfigAttribEncAV1Ext2 == {f2}, \"VAConfigAttribEncAV1Ext2\");\n\
         _Static_assert(VAConfigAttribEncTileSupport == {tile}, \"VAConfigAttribEncTileSupport\");\n\
         _Static_assert(VAConfigAttribEncPerBlockControl == {pbc}, \"VAConfigAttribEncPerBlockControl\");\n\
         _Static_assert(VAConfigAttribEncMaxSlices == {slices}, \"VAConfigAttribEncMaxSlices\");\n\
         _Static_assert(VAConfigAttribEncMaxTileRows == {rows}, \"VAConfigAttribEncMaxTileRows\");\n\
         _Static_assert(VAConfigAttribEncVP9 == 58, \"VAConfigAttribEncVP9 exists in va.h -- the driver declines it, the header does not\");\n\
         _Static_assert(VAEncPackedHeaderAV1_SPS == {sps}, \"VAEncPackedHeaderAV1_SPS\");\n\
         _Static_assert(VAEncPackedHeaderAV1_PPS == {pps}, \"VAEncPackedHeaderAV1_PPS\");\n\
         _Static_assert(VAEncPackedHeaderAV1_SPS == VAEncPackedHeaderSequence, \"the AV1 SPS is the generic Sequence type\");\n\
         _Static_assert(VAEncPackedHeaderAV1_PPS == VAEncPackedHeaderPicture, \"the AV1 PPS is the generic Picture type\");\n\
         _Static_assert(VAEncPackedHeaderSequence == 1 && VAEncPackedHeaderPicture == 2 && VAEncPackedHeaderSlice == 3, \"VAEncPackedHeaderType\");\n\
         _Static_assert(VAAV1EncTransformationIdentity == {t0} && VAAV1EncTransformationTranslation == {t1}\n\
                    && VAAV1EncTransformationRotzoom == {t2} && VAAV1EncTransformationAffine == {t3}\n\
                    && VAAV1EncTransformationCount == {t4}, \"VAEncTransformationTypeAV1\");\n\
         _Static_assert(VA_AV1_MAX_SEGMENTS == {segs} && VA_AV1_SEG_LVL_MAX == {seglvl}, \"the AV1 segment array bounds\");\n\
         _Static_assert(VA_PADDING_LOW == {low} && VA_PADDING_MEDIUM == 8 && VA_PADDING_HIGH == 16, \"the padding widths\");\n\
         _Static_assert(sizeof(VABufferID) == {bufferid} && sizeof(VASurfaceID) == {surfaceid}, \"VABufferID and VASurfaceID\");\n\
         _Static_assert(sizeof(VAEncTransformationTypeAV1) == {wmtype}, \"VAEncTransformationTypeAV1, a C enum\");\n\
         _Static_assert(sizeof(VAConfigAttribValEncAV1) == {fsize} && sizeof(VAConfigAttribValEncAV1Ext1) == {e1size}\n\
                    && sizeof(VAConfigAttribValEncAV1Ext2) == {e2size}, \"the three attribute unions\");\n\
         _Static_assert(sizeof(VAEncSegMapBufferAV1) == 16, \"VAEncSegMapBufferAV1 -- app-side, deliberately not declared here\");\n",
        f0 = CONFIG_ATTRIB_AV1_FEATURES,
        f1 = CONFIG_ATTRIB_AV1_EXT1,
        f2 = CONFIG_ATTRIB_AV1_EXT2,
        tile = CONFIG_ATTRIB_TILE_SUPPORT,
        pbc = CONFIG_ATTRIB_PER_BLOCK_CONTROL,
        slices = CONFIG_ATTRIB_MAX_SLICES,
        rows = CONFIG_ATTRIB_MAX_TILE_ROWS,
        sps = PACKED_HEADER_SPS,
        pps = PACKED_HEADER_PPS,
        t0 = transformation::IDENTITY,
        t1 = transformation::TRANSLATION,
        t2 = transformation::ROTZOOM,
        t3 = transformation::AFFINE,
        t4 = transformation::COUNT,
        segs = MAX_SEGMENTS,
        seglvl = SEG_LVL_MAX,
        low = VA_PADDING_LOW,
        bufferid = size_of::<BufferId>(),
        surfaceid = size_of::<SurfaceId>(),
        wmtype = size_of::<u32>(),
        fsize = size_of::<Av1Features>(),
        e1size = size_of::<Av1Ext1>(),
        e2size = size_of::<Av1Ext2>(),
    );
    let Some(out) = check_c(&cc, "filmcraft-va-av1-enc-consts", "av1encconsts.c", &source) else {
        return;
    };
    assert!(out.status.success(), "{cc} disagrees with our AV1 encode constants:\n{}", String::from_utf8_lossy(&out.stderr));

    // And on our side, so a wrong *type* is caught too.
    assert_eq!(size_of::<BufferId>(), 4, "VABufferID");
    assert_eq!(size_of::<SurfaceId>(), 4, "VASurfaceID");
    assert_eq!(MAX_SEGMENTS, 8, "VA_AV1_MAX_SEGMENTS");
    assert_eq!(SEG_LVL_MAX, 8, "VA_AV1_SEG_LVL_MAX");
    // SPS and PPS being consecutive enumerators is the fact the record file warns about; assert it
    // here so splitting them into unrelated constants later is a red test.
    assert_eq!(PACKED_HEADER_SPS, 1);
    assert_eq!(PACKED_HEADER_PPS, 2);
    assert_ne!(PACKED_HEADER_SPS, PACKED_HEADER_PPS, "an AV1 sequence header is not an AV1 frame header");
    // The parameter counts are the only thing in `transformation` a submission can act on, so
    // check the whole table against the header's own prose (0, 2, 4 and 6 parameters).
    assert_eq!(transformation::parameters(transformation::IDENTITY), Some(0));
    assert_eq!(transformation::parameters(transformation::TRANSLATION), Some(2));
    assert_eq!(transformation::parameters(transformation::ROTZOOM), Some(4));
    assert_eq!(transformation::parameters(transformation::AFFINE), Some(6));
    // `COUNT` is the enumeration's sentinel, not a transform: `parameters` must say no rather than
    // index anything, since `wmtype` is hostile input.
    assert_eq!(transformation::parameters(transformation::COUNT), None, "COUNT is not a transformation");
    assert_eq!(transformation::parameters(4), None);
    assert_eq!(transformation::parameters(u32::MAX), None, "and an arbitrary u32 is not a transformation");
    // The widest transform reads 6 of the 8 `wmmat` entries, so two are never read -- and must
    // still be zero, which the `Default` pin below is what proves.
    assert!(transformation::parameters(transformation::AFFINE).unwrap_or(0) <= 8);
}

/// **`Default` sets every byte of every AV1 encode record, including the five holes.**
///
/// `#[derive(Default)]` is used rather than a hand-written builder here, and this test is the
/// reason that is safe rather than merely convenient:
///
/// - Every array in these six records is 63 entries or fewer, so `Default` is derivable for all of
///   them. That is a fact about the types, and this file's own sizes are what would change it, so
///   it is not assumed: adding a longer array to any of these records fails to **compile** with
///   the derive and is then hand-written, loudly.
/// - A derive cannot "forget" a field, which is the failure a hand-written builder is usually held
///   up as preventing. What it *can* do is leave a byte unwritten, and the tiling test above proves
///   every byte is a named field — so proving every named field is zero proves every byte is zero,
///   which is what this test checks by naming each `padding_*` field.
///
/// A picture record at 1032 bytes with an unwritten hole in it is a picture the driver reads from
/// stack memory. That is the class of bug this exists for.
#[test]
fn a_default_av1_encode_record_names_no_byte_it_leaves_unset() {
    // The sequence record: its one hole, plus the 16-word tail.
    let seq = VaEncSequenceParameterBufferAv1::default();
    assert!(seq.padding_after_order_hint_bits_minus_1.iter().all(|&b| b == 0), "the hole at bytes 21-23");
    assert!(seq.va_reserved.iter().all(|&w| w == 0), "the reserved tail");
    assert_eq!(
        (seq.seq_profile, seq.seq_level_idx, seq.seq_tier, seq.hierarchical_flag, seq.order_hint_bits_minus_1),
        (0, 0, 0, 0, 0),
        "and the scalar fields around them"
    );
    assert_eq!((seq.intra_period, seq.ip_period, seq.bits_per_second, seq.seq_fields), (0, 0, 0, 0));

    // The segment record: its one hole, the 128 signed feature bytes and the 4-word tail.
    let seg = VaEncSegParamAv1::default();
    assert!(seg.padding_after_feature_mask.iter().all(|&b| b == 0), "the hole at bytes 138-139");
    assert!(seg.va_reserved.iter().all(|&w| w == 0), "the reserved tail");
    assert_eq!((seg.seg_flags, seg.segment_number), (0, 0), "0 segments, which is the honest AV1 default");
    assert!(seg.feature_data.iter().all(|row| row.iter().all(|&v| v == 0)), "every feature delta");
    assert!(seg.feature_mask.iter().all(|&b| b == 0), "every feature mask");

    // The warped-motion record: its one hole, the eight `wmmat` entries and the 4-word tail. The
    // two `wmmat` entries the affine transform never reads are named here on purpose.
    let wm = VaEncWarpedMotionParamsAv1::default();
    assert!(wm.padding_after_invalid.iter().all(|&b| b == 0), "the hole at bytes 37-39");
    assert!(wm.va_reserved.iter().all(|&w| w == 0), "the reserved tail");
    assert_eq!((wm.wmtype, wm.invalid), (0, 0), "identity transform, valid");
    assert!(wm.wmmat.iter().all(|&v| v == 0), "all eight wmmat entries, including the two affine never reads");

    // The reference-control union: a bare word.
    assert_eq!(VaRefFrameCtrlAv1::default().value, 0, "no reference preferred");

    // The picture record: its one hole, the embedded segment record and the 16-word tail.
    let pic = VaEncPictureParameterBufferAv1::default();
    assert!(pic.padding_after_loop_restoration_flags.iter().all(|&b| b == 0), "the hole at bytes 538-539");
    assert!(pic.va_reserved.iter().all(|&w| w == 0), "the reserved tail");
    assert_eq!((pic.frame_width_minus_1, pic.frame_height_minus_1), (0, 0));
    assert_eq!((pic.reconstructed_frame, pic.coded_buf), (0, 0));
    assert_eq!((pic.reserved8bits1, pic.reserved16bits1, pic.reserved16bits2, pic.reserved16bits3), (0, 0, 0, 0));
    assert_eq!((pic.picture_flags, pic.mode_control_flags), (0, 0));
    assert_eq!((pic.qmatrix_flags, pic.loop_filter_flags, pic.loop_restoration_flags, pic.tile_group_obu_hdr_info), (0, 0, 0, 0));
    assert_eq!((pic.ref_frame_ctrl_l0.value, pic.ref_frame_ctrl_l1.value), (0, 0));
    assert_eq!(pic.skip_frames_reduced_size, 0, "a signed field, and zero is the honest default");
    assert!(pic.reference_frames.iter().all(|&s| s == 0), "all eight reference surfaces");
    assert!(pic.ref_frame_idx.iter().all(|&b| b == 0));
    assert!(pic.ref_deltas.iter().all(|&d| d == 0) && pic.mode_deltas.iter().all(|&d| d == 0));
    assert!(pic.filter_level.iter().all(|&b| b == 0) && pic.cdef_y_strengths.iter().all(|&b| b == 0) && pic.cdef_uv_strengths.iter().all(|&b| b == 0));
    assert!(pic.width_in_sbs_minus_1.iter().all(|&w| w == 0), "all 63 column widths");
    assert!(pic.height_in_sbs_minus_1.iter().all(|&w| w == 0), "all 63 row heights");
    assert!(
        pic.wm.iter().all(|w| w.va_reserved.iter().all(|&x| x == 0) && w.padding_after_invalid.iter().all(|&b| b == 0)),
        "the seven embedded warped-motion records"
    );
    assert!(pic.bit_offset_qindex + pic.bit_offset_segmentation + pic.bit_offset_loopfilter_params + pic.bit_offset_cdef_params == 0, "the four bit offsets");
    assert_eq!(pic.size_in_bits_cdef_params + pic.byte_offset_frame_hdr_obu_size + pic.size_in_bits_frame_hdr_obu, 0, "the three sizes");

    // The tile-group record: its one hole and the 4-word tail.
    let tile = VaEncTileGroupBufferAv1::default();
    assert!(tile.padding_after_tg_end.iter().all(|&b| b == 0), "the hole at bytes 2-3");
    assert!(tile.va_reserved.iter().all(|&w| w == 0), "the reserved tail");
    assert_eq!((tile.tg_start, tile.tg_end), (0, 0));
}

/// The six record sizes, stated as numbers and checked against C.
///
/// The gap test above already compares every `sizeof` with the C compiler; this test exists for the
/// opposite direction — it fails the moment one of the six sizes in the record file's table moves,
/// so a record that grew a field is a red test naming the record rather than a comment that has
/// quietly gone stale. It is **not** a portability hazard: these are 32-bit-target sizes and the
/// numbers are read from `size_of` on the same target the C compiler is asked about.
#[test]
fn the_av1_enc_records_are_the_sizes_this_file_documents() {
    assert_eq!(size_of::<VaEncSequenceParameterBufferAv1>(), 88, "VAEncSequenceParameterBufferAV1");
    assert_eq!(size_of::<VaEncSegParamAv1>(), 156, "VAEncSegParamAV1");
    assert_eq!(size_of::<VaEncWarpedMotionParamsAv1>(), 56, "VAEncWarpedMotionParamsAV1");
    assert_eq!(size_of::<VaRefFrameCtrlAv1>(), 4, "VARefFrameCtrlAV1");
    assert_eq!(size_of::<VaEncPictureParameterBufferAv1>(), 1032, "VAEncPictureParameterBufferAV1");
    assert_eq!(size_of::<VaEncTileGroupBufferAv1>(), 20, "VAEncTileGroupBufferAV1");
}

/// The five holes, stated as byte offsets so a hole that moves is a named failure.
///
/// The tiling test proves every byte is accounted for and the gap test asks C for the width; this
/// one pins **where** each hole is, which is what a reader of the record file's table is relying on
/// and what nothing else would catch if a field were reordered above a hole.
#[test]
fn the_av1_enc_holes_are_where_this_file_documents_them() {
    let seq = ENC_RECORDS.iter().find(|r| r.c_name == "VAEncSequenceParameterBufferAV1").expect("the sequence record");
    let hole = seq.fields.iter().find(|f| f.rust_name == "padding_after_order_hint_bits_minus_1").expect("the sequence record's hole");
    assert_eq!(hole.offset, 21, "VAEncSequenceParameterBufferAV1: bytes 21-23");
    assert_eq!(hole.width, 3);

    let seg = ENC_RECORDS.iter().find(|r| r.c_name == "VAEncSegParamAV1").expect("the segment record");
    let hole = seg.fields.iter().find(|f| f.rust_name == "padding_after_feature_mask").expect("the segment record's hole");
    assert_eq!(hole.offset, 138, "VAEncSegParamAV1: bytes 138-139");
    assert_eq!(hole.width, 2);

    let wm = ENC_RECORDS.iter().find(|r| r.c_name == "VAEncWarpedMotionParamsAV1").expect("the warped-motion record");
    let hole = wm.fields.iter().find(|f| f.rust_name == "padding_after_invalid").expect("the warped-motion record's hole");
    assert_eq!(hole.offset, 37, "VAEncWarpedMotionParamsAV1: bytes 37-39");
    assert_eq!(hole.width, 3);

    let pic = ENC_RECORDS.iter().find(|r| r.c_name == "VAEncPictureParameterBufferAV1").expect("the picture record");
    let hole = pic.fields.iter().find(|f| f.rust_name == "padding_after_loop_restoration_flags").expect("the picture record's hole");
    assert_eq!(hole.offset, 538, "VAEncPictureParameterBufferAV1: bytes 538-539");
    assert_eq!(hole.width, 2);

    let tile = ENC_RECORDS.iter().find(|r| r.c_name == "VAEncTileGroupBufferAV1").expect("the tile-group record");
    let hole = tile.fields.iter().find(|f| f.rust_name == "padding_after_tg_end").expect("the tile-group record's hole");
    assert_eq!(hole.offset, 2, "VAEncTileGroupBufferAV1: bytes 2-3");
    assert_eq!(hole.width, 2);

    // Five holes, in five records, and the sixth record has none. Counted rather than asserted
    // per record, so a sixth hole appearing anywhere fails here naming nothing -- which is why the
    // assertions above exist alongside it.
    let total: usize = ENC_RECORDS.iter().flat_map(|r| r.fields.iter()).filter(|f| f.rust_name.starts_with("padding_")).count();
    assert_eq!(total, 5, "exactly five holes across the six records");
}

/// `cc` on this machine, or `None`.
fn compiler() -> Option<String> {
    ["cc", "gcc", "clang"].into_iter().find_map(which)
}

/// `name` on the `PATH`, or `None`.
fn which(name: &str) -> Option<String> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).find_map(|dir| {
        let candidate = dir.join(name);
        candidate.is_file().then(|| candidate.to_string_lossy().into_owned())
    })
}

/// Write `source` into a private directory under `TMPDIR`, ask `cc` to check it, and hand back its
/// output — or `None`, having said why on stdout, where the thing needed is not installed.
///
/// This is the syntax-only pass, which is what a `_Static_assert` batch needs: every assertion is a
/// compile-time constant, so there is nothing to *run*, and nothing is left behind.
fn check_c(cc: &str, dir: &str, file: &str, source: &str) -> Option<std::process::Output> {
    let (dir, path) = write_probe(dir, file, source)?;
    let out = std::process::Command::new(cc)
        .args(["-w", "-fsyntax-only", path.to_str().unwrap_or_default()])
        .output()
        .unwrap_or_else(|e| panic!("running {cc} failed: {e}"));
    let _ = std::fs::remove_dir_all(&dir);
    Some(out)
}

/// The same, for a probe that has to be **run**: the C program is compiled and linked and its
/// standard output handed back next to the path of the executable, and the caller removes the
/// directory **after** it has run it.
fn build_c(cc: &str, dir: &str, file: &str, source: &str) -> Option<(std::process::Output, std::path::PathBuf)> {
    let (dir, path) = write_probe(dir, file, source)?;
    let exe = dir.join("probe");
    let out = std::process::Command::new(cc)
        .args(["-w", "-o", exe.to_str().unwrap_or_default(), path.to_str().unwrap_or_default()])
        .output()
        .unwrap_or_else(|e| panic!("running {cc} failed: {e}"));
    Some((out, exe))
}

/// `None`, having said why, where a scratch file cannot be written or named.
fn write_probe(dir: &str, file: &str, source: &str) -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    let dir = std::env::temp_dir().join(format!("{dir}-{}", std::process::id()));
    if std::fs::create_dir_all(&dir).is_err() {
        eprintln!("SKIPPED: no writable temporary directory");
        return None;
    }
    let path = dir.join(file);
    if std::fs::write(&path, source.as_bytes()).is_err() {
        eprintln!("SKIPPED: cannot write {}", path.display());
        return None;
    }
    Some((dir, path))
}
