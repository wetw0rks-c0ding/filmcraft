//! The pins on the HEVC **encode** records: every field of every record, named, and every one of
//! them checked against the installed `/usr/include/va/va_enc_hevc.h` by a C compiler.
//!
//! This is [`super::abi_guard`] for the HEVC encode records and lives beside its records rather
//! than in the shared file for one reason only: **zero overlap**. `abi_guard.rs` is being edited
//! concurrently for the H.264 rate-control work, and two agents writing one file is how the tree
//! got corrupted earlier in this session. Nothing here is shared, and the file is `#[cfg(test)]`
//! only, so the cost of the duplication is compile time and nothing else.
//!
//! # What is checked, and why each check exists
//!
//! | test | what it would miss |
//! |---|---|
//! | [`every_record_the_hevc_encoder_would_submit_tiles_byte_for_byte`] | a **hole**. No `sizeof` and no `offsetof` can see one — padding changes no size and moves no named field. `VASegmentParameterVP9` agreed with C on 24/24 records for five commits while handing the driver two uninitialised bytes per record. |
//! | [`no_hevc_enc_record_has_a_byte_the_header_does_not_account_for`] | a wrong offset, a wrong field width, or a hole of the wrong width — asked of C **as a gap**, because C gives padding no name and `sizeof` cannot see one. |
//! | [`the_hevc_enc_bitfield_words_are_the_bits_the_c_compiler_measures`] | a **bit position**. A union has one `offsetof`, so a member packed one bit too high agrees with C on the record's size, on every field offset and on every hole, and produces a *picture*. |
//! | [`the_hevc_enc_constants_are_what_the_header_declares`] | a constant somebody typed from the same header twice. |
//! | [`a_default_hevc_encode_record_names_no_byte_it_leaves_unset`] | a record whose `Default` fills one field and not another. |
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
//! exactly the status [`VAEncSliceParameterBufferHevc`](super::VaEncSliceParameterBufferHevc)
//! holds, while its H.264 counterpart has six of them.
//!
//! # The bitfield tables were measured by running C, and one of them caught the probe
//!
//! All 88 positions came from setting a member to all ones and reading `__builtin_ctz` /
//! `31 - __builtin_clz` back out of the union's `value`. Nothing reads the header's declaration
//! order. The first version of the 16-bit probe set its member to `~0xffffu`, which truncates to
//! **zero** in a 1-bit field, and printed `NONE` — which is what saved it, and is the reason the
//! word width is carried per row rather than assumed. `~0u` truncates to each member's own width,
//! so a member narrower than a word cannot spill into its neighbour; `report` reading the whole
//! word back is what would show it if it did.

use super::super::ffi::{VA_PADDING_LOW, VA_PADDING_MEDIUM, VaPictureHevc};
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
/// A macro rather than plain data because a field's offset can only be read with `offset_of!`, which
/// needs the field name as a token; writing the list as a macro invocation is what lets each row
/// carry both names and have the offset computed rather than transcribed — which is the whole point
/// of this module.
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

/// The three HEVC encode records, with every byte of every one of them named.
///
/// **Five holes, all `padding_*` fields of a C-measured width:**
///
/// | record | bytes | why |
/// |---|---|---|
/// | `VAEncSequenceParameterBufferHEVC` | 3 | after `general_tier_flag`, before the first `uint32_t` |
/// | `VAEncSequenceParameterBufferHEVC` | 34-35 | after `max_transform_hierarchy_depth_intra`, before the first `uint32_t` |
/// | `VAEncSequenceParameterBufferHEVC` | 53-55 | after `vui_parameters_present_flag`, before `vui_fields` |
/// | `VAEncSequenceParameterBufferHEVC` | 61-63 | after `aspect_ratio_idc`, before `sar_width` |
/// | `VAEncPictureParameterBufferHEVC` | 506-507 | after `nal_unit_type`, before `pic_fields` |
///
/// **`VAEncSliceParameterBufferHEVC` has none**, and it has no `padding_*` row here on purpose: a
/// member with no C name can only appear as one of those, so a hole appearing where the header has
/// none would mean this declaration *grew* one the driver never saw. The gap check below asserts a
/// width of **zero** for every pair of neighbours in it rather than leaving a hole-free record
/// unchecked.
static ENC_RECORDS: &[EncRecord] = enc_records![
    "VAEncSequenceParameterBufferHEVC": size_of::<VaEncSequenceParameterBufferHevc>() => [
        ("general_profile_idc", general_profile_idc, size_of::<u8>()),
        ("general_level_idc", general_level_idc, size_of::<u8>()),
        ("general_tier_flag", general_tier_flag, size_of::<u8>()),
        ("-", padding_after_general_tier_flag, size_of::<[u8; 1]>()),
        ("intra_period", intra_period, size_of::<u32>()),
        ("intra_idr_period", intra_idr_period, size_of::<u32>()),
        ("ip_period", ip_period, size_of::<u32>()),
        ("bits_per_second", bits_per_second, size_of::<u32>()),
        ("pic_width_in_luma_samples", pic_width_in_luma_samples, size_of::<u16>()),
        ("pic_height_in_luma_samples", pic_height_in_luma_samples, size_of::<u16>()),
        ("seq_fields", seq_fields, size_of::<u32>()),
        ("log2_min_luma_coding_block_size_minus3", log2_min_luma_coding_block_size_minus3, size_of::<u8>()),
        ("log2_diff_max_min_luma_coding_block_size", log2_diff_max_min_luma_coding_block_size, size_of::<u8>()),
        ("log2_min_transform_block_size_minus2", log2_min_transform_block_size_minus2, size_of::<u8>()),
        ("log2_diff_max_min_transform_block_size", log2_diff_max_min_transform_block_size, size_of::<u8>()),
        ("max_transform_hierarchy_depth_inter", max_transform_hierarchy_depth_inter, size_of::<u8>()),
        ("max_transform_hierarchy_depth_intra", max_transform_hierarchy_depth_intra, size_of::<u8>()),
        ("-", padding_after_max_transform_hierarchy_depth_intra, size_of::<[u8; 2]>()),
        ("pcm_sample_bit_depth_luma_minus1", pcm_sample_bit_depth_luma_minus1, size_of::<u32>()),
        ("pcm_sample_bit_depth_chroma_minus1", pcm_sample_bit_depth_chroma_minus1, size_of::<u32>()),
        ("log2_min_pcm_luma_coding_block_size_minus3", log2_min_pcm_luma_coding_block_size_minus3, size_of::<u32>()),
        ("log2_max_pcm_luma_coding_block_size_minus3", log2_max_pcm_luma_coding_block_size_minus3, size_of::<u32>()),
        ("vui_parameters_present_flag", vui_parameters_present_flag, size_of::<u8>()),
        ("-", padding_after_vui_parameters_present_flag, size_of::<[u8; 3]>()),
        ("vui_fields", vui_fields, size_of::<u32>()),
        ("aspect_ratio_idc", aspect_ratio_idc, size_of::<u8>()),
        ("-", padding_after_aspect_ratio_idc, size_of::<[u8; 3]>()),
        ("sar_width", sar_width, size_of::<u32>()),
        ("sar_height", sar_height, size_of::<u32>()),
        ("vui_num_units_in_tick", vui_num_units_in_tick, size_of::<u32>()),
        ("vui_time_scale", vui_time_scale, size_of::<u32>()),
        ("min_spatial_segmentation_idc", min_spatial_segmentation_idc, size_of::<u16>()),
        ("max_bytes_per_pic_denom", max_bytes_per_pic_denom, size_of::<u8>()),
        ("max_bits_per_min_cu_denom", max_bits_per_min_cu_denom, size_of::<u8>()),
        ("scc_fields", scc_fields, size_of::<u32>()),
        ("va_reserved", va_reserved, (VA_PADDING_MEDIUM - 1) * size_of::<u32>()),
    ],
    "VAEncPictureParameterBufferHEVC": size_of::<VaEncPictureParameterBufferHevc>() => [
        ("decoded_curr_pic", decoded_curr_pic, size_of::<VaPictureHevc>()),
        ("reference_frames", reference_frames, size_of::<[VaPictureHevc; 15]>()),
        ("coded_buf", coded_buf, size_of::<BufferId>()),
        ("collocated_ref_pic_index", collocated_ref_pic_index, size_of::<u8>()),
        ("last_picture", last_picture, size_of::<u8>()),
        ("pic_init_qp", pic_init_qp, size_of::<u8>()),
        ("diff_cu_qp_delta_depth", diff_cu_qp_delta_depth, size_of::<u8>()),
        ("pps_cb_qp_offset", pps_cb_qp_offset, size_of::<i8>()),
        ("pps_cr_qp_offset", pps_cr_qp_offset, size_of::<i8>()),
        ("num_tile_columns_minus1", num_tile_columns_minus1, size_of::<u8>()),
        ("num_tile_rows_minus1", num_tile_rows_minus1, size_of::<u8>()),
        ("column_width_minus1", column_width_minus1, size_of::<[u8; 19]>()),
        ("row_height_minus1", row_height_minus1, size_of::<[u8; 21]>()),
        ("log2_parallel_merge_level_minus2", log2_parallel_merge_level_minus2, size_of::<u8>()),
        ("ctu_max_bitsize_allowed", ctu_max_bitsize_allowed, size_of::<u8>()),
        ("num_ref_idx_l0_default_active_minus1", num_ref_idx_l0_default_active_minus1, size_of::<u8>()),
        ("num_ref_idx_l1_default_active_minus1", num_ref_idx_l1_default_active_minus1, size_of::<u8>()),
        ("slice_pic_parameter_set_id", slice_pic_parameter_set_id, size_of::<u8>()),
        ("nal_unit_type", nal_unit_type, size_of::<u8>()),
        ("-", padding_after_nal_unit_type, size_of::<[u8; 2]>()),
        ("pic_fields", pic_fields, size_of::<u32>()),
        ("hierarchical_level_plus1", hierarchical_level_plus1, size_of::<u8>()),
        ("va_byte_reserved", va_byte_reserved, size_of::<u8>()),
        ("scc_fields", scc_fields, size_of::<u16>()),
        ("va_reserved", va_reserved, (VA_PADDING_HIGH - 1) * size_of::<u32>()),
    ],
    "VAEncSliceParameterBufferHEVC": size_of::<VaEncSliceParameterBufferHevc>() => [
        ("slice_segment_address", slice_segment_address, size_of::<u32>()),
        ("num_ctu_in_slice", num_ctu_in_slice, size_of::<u32>()),
        ("slice_type", slice_type, size_of::<u8>()),
        ("slice_pic_parameter_set_id", slice_pic_parameter_set_id, size_of::<u8>()),
        ("num_ref_idx_l0_active_minus1", num_ref_idx_l0_active_minus1, size_of::<u8>()),
        ("num_ref_idx_l1_active_minus1", num_ref_idx_l1_active_minus1, size_of::<u8>()),
        ("ref_pic_list0", ref_pic_list0, size_of::<[VaPictureHevc; 15]>()),
        ("ref_pic_list1", ref_pic_list1, size_of::<[VaPictureHevc; 15]>()),
        ("luma_log2_weight_denom", luma_log2_weight_denom, size_of::<u8>()),
        ("delta_chroma_log2_weight_denom", delta_chroma_log2_weight_denom, size_of::<i8>()),
        ("delta_luma_weight_l0", delta_luma_weight_l0, size_of::<[i8; 15]>()),
        ("luma_offset_l0", luma_offset_l0, size_of::<[i8; 15]>()),
        ("delta_chroma_weight_l0", delta_chroma_weight_l0, size_of::<[[i8; 2]; 15]>()),
        ("chroma_offset_l0", chroma_offset_l0, size_of::<[[i8; 2]; 15]>()),
        ("delta_luma_weight_l1", delta_luma_weight_l1, size_of::<[i8; 15]>()),
        ("luma_offset_l1", luma_offset_l1, size_of::<[i8; 15]>()),
        ("delta_chroma_weight_l1", delta_chroma_weight_l1, size_of::<[[i8; 2]; 15]>()),
        ("chroma_offset_l1", chroma_offset_l1, size_of::<[[i8; 2]; 15]>()),
        ("max_num_merge_cand", max_num_merge_cand, size_of::<u8>()),
        ("slice_qp_delta", slice_qp_delta, size_of::<i8>()),
        ("slice_cb_qp_offset", slice_cb_qp_offset, size_of::<i8>()),
        ("slice_cr_qp_offset", slice_cr_qp_offset, size_of::<i8>()),
        ("slice_beta_offset_div2", slice_beta_offset_div2, size_of::<i8>()),
        ("slice_tc_offset_div2", slice_tc_offset_div2, size_of::<i8>()),
        ("slice_fields", slice_fields, size_of::<u32>()),
        ("pred_weight_table_bit_offset", pred_weight_table_bit_offset, size_of::<u32>()),
        ("pred_weight_table_bit_length", pred_weight_table_bit_length, size_of::<u32>()),
        ("va_reserved", va_reserved, (VA_PADDING_MEDIUM - 2) * size_of::<u32>()),
    ],
];

/// **Every byte of every HEVC encode record belongs to a named field, and the fields line up end to
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
#[test]
fn every_record_the_hevc_encoder_would_submit_tiles_byte_for_byte() {
    assert_eq!(ENC_RECORDS.len(), 3, "the sequence, picture and slice records, and nothing else");
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
}

/// **The header agrees with us on every field's offset, every field's width and every hole's
/// width** — for all three records at once, including the one that genuinely has no hole.
///
/// Each record's `sizeof` is asserted; each member its `offsetof` and its `sizeof`; and each pair of
/// neighbours the **gap** between them, because C gives padding no name and the only way to ask a C
/// compiler "how wide is that hole" is as the difference between two `offsetof`s minus the first
/// member's own width. The gap is then compared with the width of the `padding_*` field that claims
/// to cover it, so a hole one byte too narrow fails — **and a `sizeof` would not move.**
///
/// The numbers come from `offset_of!` and `size_of` on our own declaration, so the two sides are
/// compared directly rather than both against a number somebody typed twice.
#[test]
fn no_hevc_enc_record_has_a_byte_the_header_does_not_account_for() {
    if !std::path::Path::new(LIBVA_HEADERS).join("va_enc_hevc.h").is_file() {
        eprintln!("SKIPPED: no {LIBVA_HEADERS}/va_enc_hevc.h");
        return;
    }
    let Some(cc) = compiler() else {
        eprintln!("SKIPPED: no C compiler to measure the headers with");
        return;
    };
    let mut source = String::from("#include <stdint.h>\n#include <va/va.h>\n#include <va/va_enc_hevc.h>\n");
    for record in ENC_RECORDS {
        source.push_str(&format!("\n/* {} */\n", record.c_name));
        source.push_str(&format!("_Static_assert(sizeof({r}) == {size}, \"{r}: sizeof\");\n", r = record.c_name, size = record.size));
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
    let Some(out) = check_c(&cc, "filmcraft-va-hevc-enc-abi", "hevcencabi.c", &source) else {
        return;
    };
    assert!(
        out.status.success(),
        "{cc} disagrees with our HEVC encode records on a field offset, a field width, or a hole's width:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// One bitfield member of one of these records' unions: the struct it lives in, the union's C name,
/// a **label unique across this table**, the name of the anonymous struct inside the union (libva
/// calls it `bits` in seven of the eight unions here and `fields` in none — so it is carried per
/// row rather than assumed, which is the mistake the AV1 table records once), the member's own C
/// name, the width of the word it lives in, the bit range a C compiler puts it at, and the C path to
/// the union's `value` member — which is not `{union}.value` for the two attribute unions, whose
/// bitfields sit in the union itself.
///
/// The label is carried because **`scc_fields` appears twice** — 32 bits in the sequence record and
/// 16 in the picture record — and a table keyed on the C name alone would compare the two against
/// each other.
const ENC_BITFIELDS: &[(&str, &str, &str, &str, &str, u8, u32, u32, &str)] = &[
    // VAEncSequenceParameterBufferHEVC.seq_fields (va_enc_hevc.h:387-426), uint32_t
    ("VAEncSequenceParameterBufferHEVC", "seq_fields", "seq_fields", "bits", "chroma_format_idc", 32, 0, 1, "seq_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "seq_fields", "seq_fields", "bits", "separate_colour_plane_flag", 32, 2, 2, "seq_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "seq_fields", "seq_fields", "bits", "bit_depth_luma_minus8", 32, 3, 5, "seq_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "seq_fields", "seq_fields", "bits", "bit_depth_chroma_minus8", 32, 6, 8, "seq_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "seq_fields", "seq_fields", "bits", "scaling_list_enabled_flag", 32, 9, 9, "seq_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "seq_fields", "seq_fields", "bits", "strong_intra_smoothing_enabled_flag", 32, 10, 10, "seq_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "seq_fields", "seq_fields", "bits", "amp_enabled_flag", 32, 11, 11, "seq_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "seq_fields", "seq_fields", "bits", "sample_adaptive_offset_enabled_flag", 32, 12, 12, "seq_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "seq_fields", "seq_fields", "bits", "pcm_enabled_flag", 32, 13, 13, "seq_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "seq_fields", "seq_fields", "bits", "pcm_loop_filter_disabled_flag", 32, 14, 14, "seq_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "seq_fields", "seq_fields", "bits", "sps_temporal_mvp_enabled_flag", 32, 15, 15, "seq_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "seq_fields", "seq_fields", "bits", "low_delay_seq", 32, 16, 16, "seq_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "seq_fields", "seq_fields", "bits", "hierachical_flag", 32, 17, 17, "seq_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "seq_fields", "seq_fields", "bits", "reserved_bits", 32, 18, 31, "seq_fields.value"),
    // VAEncSequenceParameterBufferHEVC.vui_fields (va_enc_hevc.h:475-499), uint32_t
    ("VAEncSequenceParameterBufferHEVC", "vui_fields", "vui_fields", "bits", "aspect_ratio_info_present_flag", 32, 0, 0, "vui_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "vui_fields", "vui_fields", "bits", "neutral_chroma_indication_flag", 32, 1, 1, "vui_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "vui_fields", "vui_fields", "bits", "field_seq_flag", 32, 2, 2, "vui_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "vui_fields", "vui_fields", "bits", "vui_timing_info_present_flag", 32, 3, 3, "vui_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "vui_fields", "vui_fields", "bits", "bitstream_restriction_flag", 32, 4, 4, "vui_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "vui_fields", "vui_fields", "bits", "tiles_fixed_structure_flag", 32, 5, 5, "vui_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "vui_fields", "vui_fields", "bits", "motion_vectors_over_pic_boundaries_flag", 32, 6, 6, "vui_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "vui_fields", "vui_fields", "bits", "restricted_ref_pic_lists_flag", 32, 7, 7, "vui_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "vui_fields", "vui_fields", "bits", "log2_max_mv_length_horizontal", 32, 8, 12, "vui_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "vui_fields", "vui_fields", "bits", "log2_max_mv_length_vertical", 32, 13, 17, "vui_fields.value"),
    // VAEncSequenceParameterBufferHEVC.scc_fields (va_enc_hevc.h:518-526), uint32_t
    ("VAEncSequenceParameterBufferHEVC", "scc_fields", "seq_scc_fields", "bits", "palette_mode_enabled_flag", 32, 0, 0, "scc_fields.value"),
    ("VAEncSequenceParameterBufferHEVC", "scc_fields", "seq_scc_fields", "bits", "reserved", 32, 1, 31, "scc_fields.value"),
    // VAEncPictureParameterBufferHEVC.pic_fields (va_enc_hevc.h:656-729), uint32_t
    ("VAEncPictureParameterBufferHEVC", "pic_fields", "pic_fields", "bits", "idr_pic_flag", 32, 0, 0, "pic_fields.value"),
    ("VAEncPictureParameterBufferHEVC", "pic_fields", "pic_fields", "bits", "coding_type", 32, 1, 3, "pic_fields.value"),
    ("VAEncPictureParameterBufferHEVC", "pic_fields", "pic_fields", "bits", "reference_pic_flag", 32, 4, 4, "pic_fields.value"),
    ("VAEncPictureParameterBufferHEVC", "pic_fields", "pic_fields", "bits", "dependent_slice_segments_enabled_flag", 32, 5, 5, "pic_fields.value"),
    ("VAEncPictureParameterBufferHEVC", "pic_fields", "pic_fields", "bits", "sign_data_hiding_enabled_flag", 32, 6, 6, "pic_fields.value"),
    ("VAEncPictureParameterBufferHEVC", "pic_fields", "pic_fields", "bits", "constrained_intra_pred_flag", 32, 7, 7, "pic_fields.value"),
    ("VAEncPictureParameterBufferHEVC", "pic_fields", "pic_fields", "bits", "transform_skip_enabled_flag", 32, 8, 8, "pic_fields.value"),
    ("VAEncPictureParameterBufferHEVC", "pic_fields", "pic_fields", "bits", "cu_qp_delta_enabled_flag", 32, 9, 9, "pic_fields.value"),
    ("VAEncPictureParameterBufferHEVC", "pic_fields", "pic_fields", "bits", "weighted_pred_flag", 32, 10, 10, "pic_fields.value"),
    ("VAEncPictureParameterBufferHEVC", "pic_fields", "pic_fields", "bits", "weighted_bipred_flag", 32, 11, 11, "pic_fields.value"),
    ("VAEncPictureParameterBufferHEVC", "pic_fields", "pic_fields", "bits", "transquant_bypass_enabled_flag", 32, 12, 12, "pic_fields.value"),
    ("VAEncPictureParameterBufferHEVC", "pic_fields", "pic_fields", "bits", "tiles_enabled_flag", 32, 13, 13, "pic_fields.value"),
    ("VAEncPictureParameterBufferHEVC", "pic_fields", "pic_fields", "bits", "entropy_coding_sync_enabled_flag", 32, 14, 14, "pic_fields.value"),
    ("VAEncPictureParameterBufferHEVC", "pic_fields", "pic_fields", "bits", "loop_filter_across_tiles_enabled_flag", 32, 15, 15, "pic_fields.value"),
    ("VAEncPictureParameterBufferHEVC", "pic_fields", "pic_fields", "bits", "pps_loop_filter_across_slices_enabled_flag", 32, 16, 16, "pic_fields.value"),
    ("VAEncPictureParameterBufferHEVC", "pic_fields", "pic_fields", "bits", "scaling_list_data_present_flag", 32, 17, 17, "pic_fields.value"),
    ("VAEncPictureParameterBufferHEVC", "pic_fields", "pic_fields", "bits", "screen_content_flag", 32, 18, 18, "pic_fields.value"),
    ("VAEncPictureParameterBufferHEVC", "pic_fields", "pic_fields", "bits", "enable_gpu_weighted_prediction", 32, 19, 19, "pic_fields.value"),
    ("VAEncPictureParameterBufferHEVC", "pic_fields", "pic_fields", "bits", "no_output_of_prior_pics_flag", 32, 20, 20, "pic_fields.value"),
    ("VAEncPictureParameterBufferHEVC", "pic_fields", "pic_fields", "bits", "reserved", 32, 21, 31, "pic_fields.value"),
    // VAEncPictureParameterBufferHEVC.scc_fields (va_enc_hevc.h:740-748), uint16_t — and the second
    // union named `scc_fields` in these records, a *narrower* one.
    ("VAEncPictureParameterBufferHEVC", "scc_fields", "pic_scc_fields", "bits", "pps_curr_pic_ref_enabled_flag", 16, 0, 0, "scc_fields.value"),
    ("VAEncPictureParameterBufferHEVC", "scc_fields", "pic_scc_fields", "bits", "reserved", 16, 1, 15, "scc_fields.value"),
    // VAEncSliceParameterBufferHEVC.slice_fields (va_enc_hevc.h:841-873), uint32_t
    ("VAEncSliceParameterBufferHEVC", "slice_fields", "slice_fields", "bits", "last_slice_of_pic_flag", 32, 0, 0, "slice_fields.value"),
    ("VAEncSliceParameterBufferHEVC", "slice_fields", "slice_fields", "bits", "dependent_slice_segment_flag", 32, 1, 1, "slice_fields.value"),
    ("VAEncSliceParameterBufferHEVC", "slice_fields", "slice_fields", "bits", "colour_plane_id", 32, 2, 3, "slice_fields.value"),
    ("VAEncSliceParameterBufferHEVC", "slice_fields", "slice_fields", "bits", "slice_temporal_mvp_enabled_flag", 32, 4, 4, "slice_fields.value"),
    ("VAEncSliceParameterBufferHEVC", "slice_fields", "slice_fields", "bits", "slice_sao_luma_flag", 32, 5, 5, "slice_fields.value"),
    ("VAEncSliceParameterBufferHEVC", "slice_fields", "slice_fields", "bits", "slice_sao_chroma_flag", 32, 6, 6, "slice_fields.value"),
    ("VAEncSliceParameterBufferHEVC", "slice_fields", "slice_fields", "bits", "num_ref_idx_active_override_flag", 32, 7, 7, "slice_fields.value"),
    ("VAEncSliceParameterBufferHEVC", "slice_fields", "slice_fields", "bits", "mvd_l1_zero_flag", 32, 8, 8, "slice_fields.value"),
    ("VAEncSliceParameterBufferHEVC", "slice_fields", "slice_fields", "bits", "cabac_init_flag", 32, 9, 9, "slice_fields.value"),
    ("VAEncSliceParameterBufferHEVC", "slice_fields", "slice_fields", "bits", "slice_deblocking_filter_disabled_flag", 32, 10, 11, "slice_fields.value"),
    ("VAEncSliceParameterBufferHEVC", "slice_fields", "slice_fields", "bits", "slice_loop_filter_across_slices_enabled_flag", 32, 12, 12, "slice_fields.value"),
    ("VAEncSliceParameterBufferHEVC", "slice_fields", "slice_fields", "bits", "collocated_from_l0_flag", 32, 13, 13, "slice_fields.value"),
    // VAConfigAttribValEncHEVCFeatures.bits (va_enc_hevc.h:57-145), uint32_t. Fifteen 2-bit fields,
    // so their `bits` member is the union itself and there is no inner anonymous struct to name.
    ("VAConfigAttribValEncHEVCFeatures", "bits", "features", "", "separate_colour_planes", 32, 0, 1, "value"),
    ("VAConfigAttribValEncHEVCFeatures", "bits", "features", "", "scaling_lists", 32, 2, 3, "value"),
    ("VAConfigAttribValEncHEVCFeatures", "bits", "features", "", "amp", 32, 4, 5, "value"),
    ("VAConfigAttribValEncHEVCFeatures", "bits", "features", "", "sao", 32, 6, 7, "value"),
    ("VAConfigAttribValEncHEVCFeatures", "bits", "features", "", "pcm", 32, 8, 9, "value"),
    ("VAConfigAttribValEncHEVCFeatures", "bits", "features", "", "temporal_mvp", 32, 10, 11, "value"),
    ("VAConfigAttribValEncHEVCFeatures", "bits", "features", "", "strong_intra_smoothing", 32, 12, 13, "value"),
    ("VAConfigAttribValEncHEVCFeatures", "bits", "features", "", "dependent_slices", 32, 14, 15, "value"),
    ("VAConfigAttribValEncHEVCFeatures", "bits", "features", "", "sign_data_hiding", 32, 16, 17, "value"),
    ("VAConfigAttribValEncHEVCFeatures", "bits", "features", "", "constrained_intra_pred", 32, 18, 19, "value"),
    ("VAConfigAttribValEncHEVCFeatures", "bits", "features", "", "transform_skip", 32, 20, 21, "value"),
    ("VAConfigAttribValEncHEVCFeatures", "bits", "features", "", "cu_qp_delta", 32, 22, 23, "value"),
    ("VAConfigAttribValEncHEVCFeatures", "bits", "features", "", "weighted_prediction", 32, 24, 25, "value"),
    ("VAConfigAttribValEncHEVCFeatures", "bits", "features", "", "transquant_bypass", 32, 26, 27, "value"),
    ("VAConfigAttribValEncHEVCFeatures", "bits", "features", "", "deblocking_filter_disable", 32, 28, 29, "value"),
    ("VAConfigAttribValEncHEVCFeatures", "bits", "features", "", "reserved", 32, 30, 31, "value"),
    // VAConfigAttribValEncHEVCBlockSizes.bits (va_enc_hevc.h:155-230), uint32_t. Eleven 2-bit
    // fields and one 10-bit `reserved`, so the last member is **10 bits wide** — the only member
    // in this table that is not 1, 2, 3 or 5, and the reason the width is measured, not derived.
    ("VAConfigAttribValEncHEVCBlockSizes", "bits", "block_sizes", "", "log2_max_coding_tree_block_size_minus3", 32, 0, 1, "value"),
    ("VAConfigAttribValEncHEVCBlockSizes", "bits", "block_sizes", "", "log2_min_coding_tree_block_size_minus3", 32, 2, 3, "value"),
    ("VAConfigAttribValEncHEVCBlockSizes", "bits", "block_sizes", "", "log2_min_luma_coding_block_size_minus3", 32, 4, 5, "value"),
    ("VAConfigAttribValEncHEVCBlockSizes", "bits", "block_sizes", "", "log2_max_luma_transform_block_size_minus2", 32, 6, 7, "value"),
    ("VAConfigAttribValEncHEVCBlockSizes", "bits", "block_sizes", "", "log2_min_luma_transform_block_size_minus2", 32, 8, 9, "value"),
    ("VAConfigAttribValEncHEVCBlockSizes", "bits", "block_sizes", "", "max_max_transform_hierarchy_depth_inter", 32, 10, 11, "value"),
    ("VAConfigAttribValEncHEVCBlockSizes", "bits", "block_sizes", "", "min_max_transform_hierarchy_depth_inter", 32, 12, 13, "value"),
    ("VAConfigAttribValEncHEVCBlockSizes", "bits", "block_sizes", "", "max_max_transform_hierarchy_depth_intra", 32, 14, 15, "value"),
    ("VAConfigAttribValEncHEVCBlockSizes", "bits", "block_sizes", "", "min_max_transform_hierarchy_depth_intra", 32, 16, 17, "value"),
    ("VAConfigAttribValEncHEVCBlockSizes", "bits", "block_sizes", "", "log2_max_pcm_coding_block_size_minus3", 32, 18, 19, "value"),
    ("VAConfigAttribValEncHEVCBlockSizes", "bits", "block_sizes", "", "log2_min_pcm_coding_block_size_minus3", 32, 20, 21, "value"),
    ("VAConfigAttribValEncHEVCBlockSizes", "bits", "block_sizes", "", "reserved", 32, 22, 31, "value"),
];

/// **Every one of the 88 bitfield members in these records and attribute unions is at the bit a C
/// compiler puts it at, and is the width the C compiler gives it** — measured by *running* C.
///
/// The failure this exists for is invisible to every other test in this file. A union has one
/// `offsetof`, so a member packed one bit too high agrees with the C compiler on the record's
/// `sizeof`, on every field offset and on every hole width, and the result of submitting it is a
/// **picture** rather than an error. Nor can the header be read for it: the members are declared in
/// an order that is right on the target that compiled it and says nothing about any other.
///
/// The comparison has two independent sides — a C program that sets one member to all ones and
/// prints `__builtin_ctz(v)` and the high set bit, and the table above — so a member that moves, or
/// a table entry that drifts, fails here naming the union and the member.
#[test]
fn the_hevc_enc_bitfield_words_are_the_bits_the_c_compiler_measures() {
    if !std::path::Path::new(LIBVA_HEADERS).join("va_enc_hevc.h").is_file() {
        eprintln!("SKIPPED: no {LIBVA_HEADERS}/va_enc_hevc.h");
        return;
    }
    let Some(cc) = compiler() else {
        eprintln!("SKIPPED: no C compiler to measure the headers with");
        return;
    };
    // `~0u` is all ones for any width up to 32, which is every member here; the assignment truncates
    // to the member's own width, so a member narrower than a word cannot spill into its neighbour —
    // and `report` reading the whole word back is what would show it if it did.
    let mut source = String::from(
        "#include <stdio.h>\n#include <string.h>\n#include <stdint.h>\n#include <va/va.h>\n#include <va/va_enc_hevc.h>\n\n\
         static void report(const char *u, const char *m, unsigned v, int width) {\n\
         \x20   if (!v) { printf(\"%s %s NONE\\n\", u, m); return; }\n\
         \x20   unsigned lo = (unsigned)__builtin_ctz(v);\n\
         \x20   unsigned hi = (width == 32) ? 31u - (unsigned)__builtin_clz(v)\n\
         \x20                              : 15u - ((unsigned)__builtin_clz(v) - 16u);\n\
         \x20   printf(\"%s %s %u %u\\n\", u, m, lo, hi);\n\
         }\n\n\
         int main(void) {\n",
    );
    for (record, c_union, label, c_inner, member, word, _, _, c_value) in ENC_BITFIELDS {
        let path = if c_inner.is_empty() { format!("{c_union}.{member}") } else { format!("{c_union}.{c_inner}.{member}") };
        source.push_str(&format!(
            "\x20 {{ {record} r; memset(&r, 0, sizeof r); r.{path} = ~0u; report(\"{label}\", \"{member}\", (unsigned)r.{c_value}, {word}); }}\n"
        ));
    }
    source.push_str("\x20 return 0;\n}\n");
    let Some((built, exe)) = build_c(&cc, "filmcraft-va-hevc-enc-bits", "hevcencbits.c", &source) else {
        return;
    };
    assert!(built.status.success(), "{cc} rejected the HEVC encode bitfield probe against the headers:\n{}", String::from_utf8_lossy(&built.stderr));
    // **Run before the directory goes.** The probe *is* the executable just written into it, so
    // removing the directory first is how this test would skip itself without noticing.
    let run =
        std::process::Command::new(exe.to_str().unwrap_or_default()).output().unwrap_or_else(|e| panic!("running the HEVC encode bitfield probe failed: {e}"));
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
/// The `features::` and `block_sizes::` masks are named in the record file for the next task to use
/// and nothing uses them yet, so without this they would be numbers somebody typed off a declaration
/// order — the exact thing the table above exists to distrust. Each row names its **C member**, so
/// the constant and the measurement are joined by the header's own name rather than by two strings
/// that look alike; a renamed constant then fails here instead of quietly matching nothing.
#[test]
fn the_published_hevc_feature_and_block_size_masks_are_the_measured_ones() {
    // (our constant's name, our bit offset, the union label and C member it stands for)
    let rows: &[(&str, u32, &str, &str)] = &[
        ("features::SEPARATE_COLOUR_PLANES", features::SEPARATE_COLOUR_PLANES, "features", "separate_colour_planes"),
        ("features::SCALING_LISTS", features::SCALING_LISTS, "features", "scaling_lists"),
        ("features::AMP", features::AMP, "features", "amp"),
        ("features::SAO", features::SAO, "features", "sao"),
        ("features::PCM", features::PCM, "features", "pcm"),
        ("features::TEMPORAL_MVP", features::TEMPORAL_MVP, "features", "temporal_mvp"),
        ("features::STRONG_INTRA_SMOOTHING", features::STRONG_INTRA_SMOOTHING, "features", "strong_intra_smoothing"),
        ("features::DEPENDENT_SLICES", features::DEPENDENT_SLICES, "features", "dependent_slices"),
        ("features::SIGN_DATA_HIDING", features::SIGN_DATA_HIDING, "features", "sign_data_hiding"),
        ("features::CONSTRAINED_INTRA_PRED", features::CONSTRAINED_INTRA_PRED, "features", "constrained_intra_pred"),
        ("features::TRANSFORM_SKIP", features::TRANSFORM_SKIP, "features", "transform_skip"),
        ("features::CU_QP_DELTA", features::CU_QP_DELTA, "features", "cu_qp_delta"),
        ("features::WEIGHTED_PREDICTION", features::WEIGHTED_PREDICTION, "features", "weighted_prediction"),
        ("features::TRANSQUANT_BYPASS", features::TRANSQUANT_BYPASS, "features", "transquant_bypass"),
        ("features::DEBLOCKING_FILTER_DISABLE", features::DEBLOCKING_FILTER_DISABLE, "features", "deblocking_filter_disable"),
        ("features::RESERVED", features::RESERVED, "features", "reserved"),
        (
            "block_sizes::LOG2_MAX_CODING_TREE_BLOCK_SIZE_MINUS3",
            block_sizes::LOG2_MAX_CODING_TREE_BLOCK_SIZE_MINUS3,
            "block_sizes",
            "log2_max_coding_tree_block_size_minus3",
        ),
        (
            "block_sizes::LOG2_MIN_CODING_TREE_BLOCK_SIZE_MINUS3",
            block_sizes::LOG2_MIN_CODING_TREE_BLOCK_SIZE_MINUS3,
            "block_sizes",
            "log2_min_coding_tree_block_size_minus3",
        ),
        (
            "block_sizes::LOG2_MIN_LUMA_CODING_BLOCK_SIZE_MINUS3",
            block_sizes::LOG2_MIN_LUMA_CODING_BLOCK_SIZE_MINUS3,
            "block_sizes",
            "log2_min_luma_coding_block_size_minus3",
        ),
        (
            "block_sizes::LOG2_MAX_LUMA_TRANSFORM_BLOCK_SIZE_MINUS2",
            block_sizes::LOG2_MAX_LUMA_TRANSFORM_BLOCK_SIZE_MINUS2,
            "block_sizes",
            "log2_max_luma_transform_block_size_minus2",
        ),
        (
            "block_sizes::LOG2_MIN_LUMA_TRANSFORM_BLOCK_SIZE_MINUS2",
            block_sizes::LOG2_MIN_LUMA_TRANSFORM_BLOCK_SIZE_MINUS2,
            "block_sizes",
            "log2_min_luma_transform_block_size_minus2",
        ),
        (
            "block_sizes::MAX_MAX_TRANSFORM_HIERARCHY_DEPTH_INTER",
            block_sizes::MAX_MAX_TRANSFORM_HIERARCHY_DEPTH_INTER,
            "block_sizes",
            "max_max_transform_hierarchy_depth_inter",
        ),
        (
            "block_sizes::MIN_MAX_TRANSFORM_HIERARCHY_DEPTH_INTER",
            block_sizes::MIN_MAX_TRANSFORM_HIERARCHY_DEPTH_INTER,
            "block_sizes",
            "min_max_transform_hierarchy_depth_inter",
        ),
        (
            "block_sizes::MAX_MAX_TRANSFORM_HIERARCHY_DEPTH_INTRA",
            block_sizes::MAX_MAX_TRANSFORM_HIERARCHY_DEPTH_INTRA,
            "block_sizes",
            "max_max_transform_hierarchy_depth_intra",
        ),
        (
            "block_sizes::MIN_MAX_TRANSFORM_HIERARCHY_DEPTH_INTRA",
            block_sizes::MIN_MAX_TRANSFORM_HIERARCHY_DEPTH_INTRA,
            "block_sizes",
            "min_max_transform_hierarchy_depth_intra",
        ),
        (
            "block_sizes::LOG2_MAX_PCM_CODING_BLOCK_SIZE_MINUS3",
            block_sizes::LOG2_MAX_PCM_CODING_BLOCK_SIZE_MINUS3,
            "block_sizes",
            "log2_max_pcm_coding_block_size_minus3",
        ),
        (
            "block_sizes::LOG2_MIN_PCM_CODING_BLOCK_SIZE_MINUS3",
            block_sizes::LOG2_MIN_PCM_CODING_BLOCK_SIZE_MINUS3,
            "block_sizes",
            "log2_min_pcm_coding_block_size_minus3",
        ),
        ("block_sizes::RESERVED", block_sizes::RESERVED, "block_sizes", "reserved"),
    ];
    for (name, bit, label, member) in rows {
        let (_, _, _, _, measured_member, word, low, _high, _) = ENC_BITFIELDS
            .iter()
            .find(|(_, _, l, _, m, ..)| *l == *label && *m == *member)
            .unwrap_or_else(|| panic!("{name} has no measured bitfield behind it in ENC_BITFIELDS"));
        assert_eq!(measured_member, member, "{name}: the C member it stands for");
        assert_eq!(*bit, *low, "{name}: our constant says bit {bit}, C measured {low} of a {word}-bit word");
    }
    // Every attribute field is 2 bits wide, so `mask` covers exactly {bit, bit + 1}: checked by the
    // count, by the position, and by the absence of overlap between neighbours — a mask one bit too
    // wide would spill into the next field and read two fields as one.
    for (name, bit, ..) in rows.iter().filter(|(name, ..)| name.starts_with("features::")) {
        let covered = features::mask(*bit);
        assert_eq!(covered.count_ones(), 2, "{name}: a feature field is two bits");
        assert_eq!(covered.trailing_zeros(), *bit, "{name}: the mask starts at the bit the constant names");
    }
    assert_eq!(features::mask(features::SEPARATE_COLOUR_PLANES), 0b11, "a 2-bit mask at bit 0");
    assert_eq!(features::mask(features::DEBLOCKING_FILTER_DISABLE), 0b11 << 28, "a 2-bit mask at bit 28");
    assert_eq!(block_sizes::reserved_mask(block_sizes::RESERVED), 0x3FF << 22, "the 10-bit reserved mask at bit 22");
    // The one field in these unions that is **not** two bits, so the one mask that is not `mask()`.
    let reserved_width = ENC_BITFIELDS.iter().find(|(_, _, l, _, m, ..)| *l == "block_sizes" && *m == "reserved").map(|row| row.7 - row.6 + 1).unwrap_or(0);
    assert_eq!(reserved_width, 10, "block_sizes.reserved is ten bits, which is why it has its own mask");
    assert_eq!(block_sizes::reserved_mask(0).count_ones(), 10, "and that mask is ten bits wide");
}

/// **Every constant in this file is what the header declares**, asked of the header rather than
/// compared with a number somebody typed from it a second time.
///
/// `VAEncPackedHeaderHEVC_VPS` and `VAEncPackedHeaderHEVC_SPS` are the row that matters: the header
/// assigns both the same enumerator, and asserting both against *different* Rust constants is what
/// makes "these two are the same number" a checked fact instead of a comment nobody re-reads.
#[test]
fn the_hevc_enc_constants_are_what_the_header_declares() {
    if !std::path::Path::new(LIBVA_HEADERS).join("va_enc_hevc.h").is_file() {
        eprintln!("SKIPPED: no {LIBVA_HEADERS}/va_enc_hevc.h");
        return;
    }
    let Some(cc) = compiler() else {
        eprintln!("SKIPPED: no C compiler to measure the headers with");
        return;
    };
    let source = format!(
        "#include <stdint.h>\n#include <va/va.h>\n#include <va/va_enc_hevc.h>\n\
         _Static_assert(HEVC_LAST_PICTURE_EOSEQ == 1, \"HEVC_LAST_PICTURE_EOSEQ\");\n\
         _Static_assert(HEVC_LAST_PICTURE_EOSTREAM == {eostream}, \"HEVC_LAST_PICTURE_EOSTREAM\");\n\
         _Static_assert(VAEncPackedHeaderHEVC_VPS == {vps}, \"VAEncPackedHeaderHEVC_VPS\");\n\
         _Static_assert(VAEncPackedHeaderHEVC_SPS == 1, \"VAEncPackedHeaderHEVC_SPS is the same enumerator as VPS\");\n\
         _Static_assert(VAEncPackedHeaderHEVC_PPS == {pps}, \"VAEncPackedHeaderHEVC_PPS\");\n\
         _Static_assert(VAEncPackedHeaderHEVC_Slice == {slice}, \"VAEncPackedHeaderHEVC_Slice\");\n\
         _Static_assert(VAEncPackedHeaderHEVC_SEI == 0x80000001, \"VAEncPackedHeaderHEVC_SEI, deprecated and not declared here\");\n\
         _Static_assert(VAEncPackedHeaderSequence == 1 && VAEncPackedHeaderPicture == 2 && VAEncPackedHeaderSlice == 3, \"VAEncPackedHeaderType\");\n\
         _Static_assert(VAConfigAttribEncHEVCFeatures == {feat}, \"VAConfigAttribEncHEVCFeatures\");\n\
         _Static_assert(VAConfigAttribEncHEVCBlockSizes == {blocks}, \"VAConfigAttribEncHEVCBlockSizes\");\n\
         _Static_assert(VAConfigAttribEncMaxSlices == {slices}, \"VAConfigAttribEncMaxSlices\");\n\
         _Static_assert(VAConfigAttribEncInterlaced == 11, \"VAConfigAttribEncInterlaced, deliberately not declared here\");\n\
         _Static_assert(VA_PADDING_LOW == {low} && VA_PADDING_MEDIUM == {medium} && VA_PADDING_HIGH == {high}, \"the padding widths\");\n\
         _Static_assert(sizeof(VAPictureHEVC) == {picture}, \"VAPictureHEVC\");\n\
         _Static_assert(sizeof(VABufferID) == {bufferid} && sizeof(VASurfaceID) == {surfaceid}, \"VABufferID and VASurfaceID\");\n\
         _Static_assert(sizeof(VAConfigAttribValEncHEVCFeatures) == {featsize}, \"VAConfigAttribValEncHEVCFeatures\");\n\
         _Static_assert(sizeof(VAConfigAttribValEncHEVCBlockSizes) == {blocksize}, \"VAConfigAttribValEncHEVCBlockSizes\");\n",
        eostream = LAST_PICTURE_EOSTREAM,
        vps = PACKED_HEADER_VPS_OR_SPS,
        pps = PACKED_HEADER_PPS,
        slice = PACKED_HEADER_SLICE,
        feat = CONFIG_ATTRIB_HEVC_FEATURES,
        blocks = CONFIG_ATTRIB_HEVC_BLOCK_SIZES,
        slices = CONFIG_ATTRIB_MAX_SLICES,
        low = VA_PADDING_LOW,
        medium = VA_PADDING_MEDIUM,
        high = VA_PADDING_HIGH,
        picture = size_of::<VaPictureHevc>(),
        bufferid = size_of::<BufferId>(),
        surfaceid = size_of::<BufferId>(),
        featsize = size_of::<HevcFeatures>(),
        blocksize = size_of::<HevcBlockSizes>(),
    );
    let Some(out) = check_c(&cc, "filmcraft-va-hevc-enc-consts", "hevcencconsts.c", &source) else {
        return;
    };
    assert!(out.status.success(), "{cc} disagrees with our HEVC encode constants:\n{}", String::from_utf8_lossy(&out.stderr));

    // And on our side, so a wrong *type* is caught too: both attribute words are a `u32` and
    // `VAPictureHEVC` is the record the picture and slice records embed fifteen times each.
    assert_eq!(size_of::<HevcFeatures>(), 4, "VAConfigAttribValEncHEVCFeatures is one 32-bit word");
    assert_eq!(size_of::<HevcBlockSizes>(), 4, "VAConfigAttribValEncHEVCBlockSizes is one 32-bit word");
    assert_eq!(size_of::<VaPictureHevc>(), 28, "VAPictureHEVC (va.h:5292), fifteen of them per record");
    assert_eq!(VA_PADDING_HIGH, 16, "va.h:362 — and the picture record's tail is 15 of them, not 8");
    // VPS and SPS being one enumerator is the fact the record file warns about; assert it here so
    // splitting them into two constants later is a red test and not a silent behaviour change.
    assert_eq!(PACKED_HEADER_VPS_OR_SPS, 1);
    assert_eq!(PACKED_HEADER_PPS, 2);
    assert_ne!(PACKED_HEADER_VPS_OR_SPS, PACKED_HEADER_PPS, "a VPS/SPS packed header is not a PPS packed header");
}

/// **`Default` sets every byte of every HEVC encode record, including the holes.**
///
/// `#[derive(Default)]` is used rather than a hand-written builder here, and this test is the
/// reason that is safe rather than merely convenient:
///
/// - The AV1 decode picture record **cannot** derive it — `[u16; 63]` has no `Default` — so that one
///   is hand-written. Every array in these three records is 21 entries or fewer, so `Default` is
///   derivable for all of them. That is a fact about the types, and this file's own sizes are what
///   would change it, so it is not assumed: adding a longer array to any of these records fails to
///   **compile** with the derive and is then hand-written, loudly.
/// - A derive cannot "forget" a field, which is the failure a hand-written builder is usually held
///   up as preventing. What it *can* do is leave a byte unwritten, and the tiling test above proves
///   every byte is a named field — so proving every named field is zero proves every byte is zero,
///   which is what this test checks by naming each `padding_*` field.
///
/// A record at 576 bytes with an unwritten hole in it is a picture the driver reads from stack
/// memory. That is the class of bug this exists for.
#[test]
fn a_default_hevc_encode_record_names_no_byte_it_leaves_unset() {
    // The sequence record: four holes, plus the tail.
    let seq = VaEncSequenceParameterBufferHevc::default();
    assert!(seq.padding_after_general_tier_flag.iter().all(|&b| b == 0), "the hole at byte 3");
    assert!(seq.padding_after_max_transform_hierarchy_depth_intra.iter().all(|&b| b == 0), "the hole at bytes 34-35");
    assert!(seq.padding_after_vui_parameters_present_flag.iter().all(|&b| b == 0), "the hole at bytes 53-55");
    assert!(seq.padding_after_aspect_ratio_idc.iter().all(|&b| b == 0), "the hole at bytes 61-63");
    assert!(seq.va_reserved.iter().all(|&w| w == 0), "the reserved tail");
    assert_eq!(
        (seq.general_profile_idc, seq.general_level_idc, seq.general_tier_flag, seq.intra_period, seq.bits_per_second),
        (0, 0, 0, 0, 0),
        "and the scalar fields around them"
    );
    assert_eq!((seq.pic_width_in_luma_samples, seq.pic_height_in_luma_samples, seq.min_spatial_segmentation_idc), (0, 0, 0));

    // The picture record: one hole, plus a 15-word tail.
    let pic = VaEncPictureParameterBufferHevc::default();
    assert!(pic.padding_after_nal_unit_type.iter().all(|&b| b == 0), "the hole at bytes 506-507");
    assert!(pic.va_reserved.iter().all(|&w| w == 0), "the reserved tail");
    assert_eq!((pic.last_picture, pic.pic_init_qp, pic.nal_unit_type, pic.va_byte_reserved), (0, 0, 0, 0));
    assert_eq!(pic.scc_fields, 0, "a u16 union, unlike the sequence record's u32 one of the same name");
    assert_eq!(pic.coded_buf, 0);
    assert_eq!(pic.collocated_ref_pic_index, 0, "zero, and 0xFF is what a *temporal MVP off* picture wants — see the field's note");
    assert!(pic.reference_frames.iter().all(|p| p.picture_id == 0 && p.pic_order_cnt == 0 && p.flags == 0 && p.va_reserved.iter().all(|&w| w == 0)));
    assert!(pic.column_width_minus1.iter().all(|&b| b == 0));
    assert!(pic.row_height_minus1.iter().all(|&b| b == 0));

    // The slice record: no hole to name, so this one is about the weights and the reserved tail.
    let sli = VaEncSliceParameterBufferHevc::default();
    assert!(sli.va_reserved.iter().all(|&w| w == 0), "the reserved tail");
    assert_eq!((sli.slice_type, sli.max_num_merge_cand, sli.luma_log2_weight_denom), (0, 0, 0));
    assert_eq!((sli.slice_qp_delta, sli.slice_cb_qp_offset, sli.slice_tc_offset_div2), (0, 0, 0));
    assert!(sli.delta_luma_weight_l0.iter().all(|&w| w == 0));
    assert!(sli.chroma_offset_l1.iter().all(|row| row.iter().all(|&w| w == 0)));
    assert!(sli.ref_pic_list0.iter().all(|p| p.picture_id == 0));
    assert!(sli.ref_pic_list1.iter().all(|p| p.picture_id == 0));
}

/// The three record sizes, stated as numbers and checked against C.
///
/// The gap test above already compares every `sizeof` with the C compiler; this test exists for the
/// opposite direction — it fails the moment one of the three sizes in the record file's table moves,
/// so a record that grew a field is a red test naming the record rather than a comment that has
/// quietly gone stale. It is **not** a portability hazard: these are 32-bit-target sizes and the
/// numbers are read from `size_of` on the same target the C compiler is asked about.
#[test]
fn the_hevc_enc_records_are_the_sizes_this_file_documents() {
    assert_eq!(size_of::<VaEncSequenceParameterBufferHevc>(), 116, "VAEncSequenceParameterBufferHEVC");
    assert_eq!(size_of::<VaEncPictureParameterBufferHevc>(), 576, "VAEncPictureParameterBufferHEVC");
    assert_eq!(size_of::<VaEncSliceParameterBufferHevc>(), 1076, "VAEncSliceParameterBufferHEVC");
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
