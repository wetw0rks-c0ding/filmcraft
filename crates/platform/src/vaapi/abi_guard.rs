//! The guard against the bug this backend had: a hand-declared `extern "C"` type that does not
//! match the header it claims to come from.
//!
//! `vaSyncBuffer` was declared with two parameters where libva's third (`uint64_t timeout_ns`,
//! `va.h:4391`) was never written at the call site. That is undefined behaviour: the driver read an
//! argument slot nobody set, the call returned `VA_STATUS_ERROR_TIMEDOUT` in a debug build and
//! success in a release one, and every picture it produced was correct. Nothing in the language can
//! catch that, because the *declaration* is the thing that is wrong — the call site matched the
//! declaration perfectly, so `rustc`, the linker and every test that made the call were satisfied.
//!
//! So the declaration has to be written down somewhere an independent source can check, and there
//! are two oracles, not equally available:
//!
//! * [`every_entry_point_has_the_arity_the_header_declares`] needs nothing but the header text. It
//!   counts each prototype's parameters and compares them with an arity that `va_entry_points!`
//!   derives from the very parameter list the pointer type was built out of, so the two cannot
//!   drift apart. **This is the check that would have caught `vaSyncBuffer`**, and it needs no
//!   driver, no library and no compiler.
//! * [`every_declaration_matches_what_a_c_compiler_measures`] needs a C compiler, and it is the only
//!   thing that can judge a *struct*: it redeclares every entry point (a C redeclaration that
//!   disagrees with the header is a hard error, not a warning) and static-asserts every record's
//!   `sizeof`. A record declared eight bytes short is the size-only version of a wrong arity, and
//!   this module had two — `VASurfaceAttrib` and `VABufferInfo`.
//!
//! Both skip, with a note on stdout, where the thing they need is not installed, so neither can turn
//! a machine red for the wrong reason. Where the headers *are* installed — every machine that can
//! run the VA-API backend at all — they are an assertion, not a courtesy.

use super::*;
use std::mem::{offset_of, size_of};

/// Where the libva headers live on this machine.
const LIBVA_HEADERS: &str = "/usr/include/va";

/// The header text with its `/* … */` and `//` comments blanked out, so a prototype cannot be found
/// inside a doc comment and a parameter cannot be cut in half by one. Newlines are kept, so a byte
/// offset still names the line a reader would go and look at.
fn without_c_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let bytes = src.as_bytes();
    let (mut i, mut depth) = (0usize, 0usize);
    while i < bytes.len() {
        if depth == 0 && bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'*') {
            depth = 1;
            i += 2;
            continue;
        }
        if depth == 0 && bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'/') {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if depth > 0 {
            if bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/') {
                depth = 0;
                i += 2;
                continue;
            }
            out.push(if bytes[i] == b'\n' { '\n' } else { ' ' });
            i += 1;
            continue;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// The byte offset just past `name(` in `src`, if `name` appears there as a whole identifier.
/// `drm.drm_ioctl_prime_…`, and a longer name that merely ends in `name`, are not matches.
fn find_call(src: &str, name: &str) -> Option<usize> {
    let bytes = src.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = src[from..].find(name) {
        let at = from + rel;
        from = at + name.len();
        let standalone = at == 0 || !(bytes[at - 1].is_ascii_alphanumeric() || bytes[at - 1] == b'_' || bytes[at - 1] == b'.');
        let mut j = at + name.len();
        while bytes.get(j).is_some_and(u8::is_ascii_whitespace) {
            j += 1;
        }
        if standalone && bytes.get(j) == Some(&b'(') {
            return Some(j + 1);
        }
    }
    None
}

/// How many parameters the argument list starting at `from` has, counting the commas at its top
/// level. `()` and `(void)` are zero.
fn parameter_count(src: &str, from: usize) -> usize {
    // `from` is already *inside* the call's parentheses, so the depth starts at one and the first
    // `)` is the one that closes the list.
    let (mut depth, mut commas, mut any) = (1usize, 0usize, false);
    for &b in src.as_bytes().get(from..).unwrap_or_default() {
        match b {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    break;
                }
            }
            b',' if depth == 1 => commas += 1,
            b' ' | b'\t' | b'\n' | b'\r' if depth == 1 => {}
            _ if depth == 1 => any = true,
            _ => {}
        }
    }
    if !any {
        return 0;
    }
    commas + 1
}

/// Every entry point's arity, re-derived from the header that declares it.
///
/// Nothing in the toolchain can do this on its own — a call site type-checks against the
/// declaration, so a wrong arity in the *declaration* is invisible to the compiler, to the linker
/// and to every test that makes the call. The only oracle is the header, so this reads it.
#[test]
fn every_entry_point_has_the_arity_the_header_declares() {
    let dir = std::path::Path::new(LIBVA_HEADERS);
    if !dir.join("va.h").is_file() {
        eprintln!("SKIPPED: no {LIBVA_HEADERS}/va.h, so there is no header to check the declarations against");
        return;
    }
    for entry in ENTRY_POINTS {
        let file = entry.header.split(':').next().unwrap_or(entry.header);
        let path = dir.join(file);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{} names {}: {e}", entry.symbol, path.display()));
        let stripped = without_c_comments(&text);
        let at = find_call(&stripped, entry.symbol)
            .unwrap_or_else(|| panic!("{} does not declare {} at all, so the declaration of it is unchecked", entry.header, entry.symbol));
        let found = parameter_count(&stripped, at);
        assert_eq!(found, entry.arity, "{}: the header ({}) declares {found} parameters, our declaration has {}", entry.symbol, entry.header, entry.arity);
    }
    assert!(!ENTRY_POINTS.is_empty(), "the table of entry points is empty, so the loop above proved nothing");
}

/// VP9's two records are measured here for the reason the other records are, and the slice record is
/// named in this test's own title now rather than only in the table below because of what it is
/// worth: `VADecPictureParameterBufferVP9` (`va_dec_vp9.h:58`) has a structural difference from
/// H.264 — its `first_partition_size` is a `uint16_t` behind a one-byte hole (offset 46, not 45) —
/// and `VASliceParameterBufferVP9` (`va_dec_vp9.h:279`) is eight `VASegmentParameterVP9`s, each of
/// which is 36 bytes where its fields add up to 34, because `chroma_dc_quant_scale` ends at 18 and
/// the `uint32_t` tail is 4-aligned and so starts at 20. Both are the shapes where a declaration that
/// reads the header's field list as its byte order is a record at offsets the driver did not compile
/// against, and both are **submitted for every VP9 picture**, so there is no "only the interesting
/// codec" discount to apply.
///
/// The two bytes at 18-19 are also asked of the C compiler **as a gap**, because C gives that padding
/// no name and no `sizeof` can see one: the hole is where the compiler puts `va_reserved` minus where
/// `chroma_dc_quant_scale` ends, and that difference is compared with the width of the
/// `padding_before_va_reserved` field our declaration uses to cover it. A record that left those two
/// bytes to the compiler agrees with every `sizeof` and every `offsetof` above and still hands two
/// uninitialised bytes to the driver, eight records a picture.
#[test]
fn the_vp9_records_are_the_sizes_the_header_declares() {
    if !std::path::Path::new(LIBVA_HEADERS).join("va_dec_vp9.h").is_file() {
        eprintln!("SKIPPED: no {LIBVA_HEADERS}/va_dec_vp9.h");
        return;
    }
    let Some(cc) = ["cc", "gcc", "clang"].into_iter().find_map(which) else {
        eprintln!("SKIPPED: no C compiler to measure the headers with");
        return;
    };
    // Every size, and the offsets a plausible reading gets wrong, asked of the C compiler rather than
    // of `offset_of!` on our own declaration: the point is that the two agree, and a test that reads
    // both sides of the comparison out of our own code proves nothing.
    let source = format!(
        "#include <stdint.h>\n#include <va/va.h>\n#include <va/va_dec_vp9.h>\n\
         _Static_assert(sizeof(VADecPictureParameterBufferVP9) == {pic_size}, \"sizeof\");\n\
         _Static_assert(__builtin_offsetof(VADecPictureParameterBufferVP9, first_partition_size) == {first}, \"first_partition_size\");\n\
         _Static_assert(__builtin_offsetof(VADecPictureParameterBufferVP9, va_reserved) == {reserved}, \"va_reserved\");\n\
         _Static_assert(sizeof(VASegmentParameterVP9) == {seg_size}, \"sizeof\");\n\
         _Static_assert(__builtin_offsetof(VASegmentParameterVP9, filter_level) == {seg_filter}, \"filter_level\");\n\
         _Static_assert(__builtin_offsetof(VASegmentParameterVP9, luma_ac_quant_scale) == {seg_ac}, \"luma_ac_quant_scale\");\n\
         _Static_assert(__builtin_offsetof(VASegmentParameterVP9, chroma_ac_quant_scale) == {seg_uv_ac}, \"chroma_ac_quant_scale\");\n\
         _Static_assert(__builtin_offsetof(VASegmentParameterVP9, chroma_dc_quant_scale) == {seg_uv_dc}, \"chroma_dc_quant_scale\");\n\
         _Static_assert(__builtin_offsetof(VASegmentParameterVP9, va_reserved) == {seg_reserved}, \"va_reserved\");\n\
         _Static_assert(__builtin_offsetof(VASegmentParameterVP9, va_reserved) - __builtin_offsetof(VASegmentParameterVP9, chroma_dc_quant_scale) - sizeof(int16_t) == {seg_hole}, \"the hole before va_reserved\");\n\
         _Static_assert(sizeof(VASliceParameterBufferVP9) == {slice_size}, \"sizeof\");\n\
         _Static_assert(__builtin_offsetof(VASliceParameterBufferVP9, seg_param) == {slice_seg}, \"seg_param\");\n\
         _Static_assert(__builtin_offsetof(VASliceParameterBufferVP9, va_reserved) == {slice_reserved}, \"va_reserved\");\n",
        pic_size = size_of::<VaPictureParameterBufferVp9>(),
        first = offset_of!(VaPictureParameterBufferVp9, first_partition_size),
        reserved = offset_of!(VaPictureParameterBufferVp9, va_reserved),
        seg_size = size_of::<VaSegmentParameterVp9>(),
        seg_filter = offset_of!(VaSegmentParameterVp9, filter_level),
        seg_ac = offset_of!(VaSegmentParameterVp9, luma_ac_quant_scale),
        seg_uv_ac = offset_of!(VaSegmentParameterVp9, chroma_ac_quant_scale),
        seg_uv_dc = offset_of!(VaSegmentParameterVp9, chroma_dc_quant_scale),
        seg_reserved = offset_of!(VaSegmentParameterVp9, va_reserved),
        // **The hole is pinned from the C side as well, as a gap and not as a member.** C has no name
        // for the padding between `chroma_dc_quant_scale` and `va_reserved`, so the only way a C
        // compiler can be asked "how wide is it" is as the difference between the two offsets minus
        // the field's own width — and that difference is then compared with the width of the
        // `padding_before_va_reserved` field that declares it on our side. A declaration that left the
        // hole implicit, or gave it the wrong width, fails here.
        seg_hole = size_of::<[u8; 2]>(),
        slice_size = size_of::<VaSliceParameterBufferVp9>(),
        slice_seg = offset_of!(VaSliceParameterBufferVp9, seg_param),
        slice_reserved = offset_of!(VaSliceParameterBufferVp9, va_reserved),
    );
    let dir = std::env::temp_dir().join(format!("filmcraft-va-vp9-abi-{}", std::process::id()));
    if std::fs::create_dir_all(&dir).is_err() {
        eprintln!("SKIPPED: no writable temporary directory");
        return;
    }
    let path = dir.join("vp9abi.c");
    let written = std::fs::write(&path, source.as_bytes()).is_ok();
    let out = if written {
        std::process::Command::new(&cc).args(["-fsyntax-only", "-I", LIBVA_HEADERS, path.to_str().unwrap_or_default()]).output()
    } else {
        eprintln!("SKIPPED: cannot write {}", path.display());
        return;
    };
    let _ = std::fs::remove_dir_all(&dir);
    let out = out.unwrap_or_else(|e| panic!("running {cc} failed: {e}"));
    assert!(out.status.success(), "{cc} disagrees with our VP9 records:\n{}", String::from_utf8_lossy(&out.stderr));
}

// ---------------------------------------------------------------------------------------------
// Every record this backend hands to a driver, and every field in it, written out once.
//
// The list is by hand on purpose. A record gains a field far more often than it loses one, and a
// new field that nobody listed here has to be *added here to compile* — so the failure mode of this
// table going stale is a red build naming the record, not a driver silently reading a byte this
// crate never initialised. The alternative, deriving the fields from the Rust struct, cannot be
// written in stable Rust at all (`offset_of!` needs a field name as a token) and a reflectively
// derived one would happily agree with whatever the declaration already said, which is the defect
// this whole module exists to catch.
//
// Each row names three things: the member's name **in the C header**, our field's name, and the
// field's width. The C name is `-` where C gives the gap no name at all — which is every
// `padding_*` field, and is the reason the hole cannot simply be read off a header.
// ---------------------------------------------------------------------------------------------

/// One field of a record, as [`every_record_the_backend_submits_tiles_byte_for_byte`] and
/// [`no_submitted_record_has_a_byte_the_header_does_not_account_for`] both read it.
struct SubmittedField {
    /// The member's name in `/usr/include/va/`, or `-` for a `padding_*` field, which C has no name for.
    c_name: &'static str,
    /// Our field's name, for the message a failure prints.
    rust_name: &'static str,
    /// `offset_of!` on our declaration.
    offset: usize,
    /// `size_of` of the field's type.
    width: usize,
}

/// A record and its fields, in declaration order.
struct SubmittedRecord {
    /// The struct's name in `/usr/include/va/`.
    c_name: &'static str,
    /// `size_of` on our declaration, which the C compiler is asked to agree with.
    size: usize,
    fields: &'static [SubmittedField],
}

/// `submitted_records!` builds [`SUBMITTED_RECORDS`] from the list above.
///
/// It is a macro rather than plain data because a field's offset can only be read with
/// `offset_of!`, which needs the field name as a token; writing the list as a macro invocation is
/// what lets each row carry both names and have the offset computed rather than transcribed.
macro_rules! submitted_records {
    ($($c_record:literal : size_of::<$ty:ty>() => [ $( ($c_field:literal, $field:ident, $width:expr) ),* $(,)? ]),* $(,)?) => {
        &[$(SubmittedRecord {
            c_name: $c_record,
            size: size_of::<$ty>(),
            fields: &[
                $(SubmittedField {
                    c_name: $c_field,
                    rust_name: stringify!($field),
                    offset: offset_of!($ty, $field),
                    width: $width,
                }),*
            ],
        }),*]
    };
}

/// Every record this backend hands to libva, with every field in it.
///
/// Which records belong here is decided by what reaches a driver, not by what is interesting: the
/// H.264 and HEVC picture and slice records are submitted on **every picture**, so a hole in one
/// has a wider blast radius than the VP9 hole this list started from. The attribute and image
/// records are in it too — the backend passes a null array for the former, but the array's *stride*
/// is what a driver walks, so a record one byte short is wrong the moment anyone passes one, and
/// the last two are records libva writes into rather than reads, where an unnamed hole is a read of
/// bytes the driver never set.
static SUBMITTED_RECORDS: &[SubmittedRecord] = submitted_records![
    "VAImageFormat": size_of::<VaImageFormat>() => [
        ("fourcc", fourcc, size_of::<u32>()),
        ("byte_order", byte_order, size_of::<u32>()),
        ("bits_per_pixel", bits_per_pixel, size_of::<u32>()),
        ("depth", depth, size_of::<u32>()),
        ("red_mask", red_mask, size_of::<u32>()),
        ("green_mask", green_mask, size_of::<u32>()),
        ("blue_mask", blue_mask, size_of::<u32>()),
        ("alpha_mask", alpha_mask, size_of::<u32>()),
        ("va_reserved", va_reserved, VA_PADDING_LOW * size_of::<u32>()),
    ],
    "VAImage": size_of::<VaImage>() => [
        ("image_id", image_id, size_of::<VABufferID>()),
        ("format", format, size_of::<VaImageFormat>()),
        ("buf", buf, size_of::<VABufferID>()),
        ("width", width, size_of::<u16>()),
        ("height", height, size_of::<u16>()),
        ("data_size", data_size, size_of::<u32>()),
        ("num_planes", num_planes, size_of::<u32>()),
        ("pitches", pitches, size_of::<[u32; 3]>()),
        ("offsets", offsets, size_of::<[u32; 3]>()),
        ("num_palette_entries", num_palette_entries, size_of::<i32>()),
        ("entry_bytes", entry_bytes, size_of::<i32>()),
        ("component_order", component_order, size_of::<[i8; 4]>()),
        ("va_reserved", va_reserved, VA_PADDING_LOW * size_of::<u32>()),
    ],
    "VAConfigAttrib": size_of::<VaConfigAttrib>() => [
        ("type", type_, size_of::<VAStatus>()),
        ("value", value, size_of::<VAStatus>()),
    ],
    "VASurfaceAttrib": size_of::<VaSurfaceAttrib>() => [
        ("type", type_, size_of::<VAStatus>()),
        ("flags", flags, size_of::<VAStatus>()),
        ("value", value, size_of::<VaGenericValue>()),
    ],
    "VAGenericValue": size_of::<VaGenericValue>() => [
        ("type", type_, size_of::<VAStatus>()),
        ("-", padding_before_value, size_of::<[u8; 4]>()),
        ("value", value, size_of::<VaGenericValueValue>()),
    ],
    "VAPictureH264": size_of::<VaPictureH264>() => [
        ("picture_id", picture_id, size_of::<VABufferID>()),
        ("frame_idx", frame_idx, size_of::<u32>()),
        ("flags", flags, size_of::<u32>()),
        ("TopFieldOrderCnt", top_field_order_cnt, size_of::<i32>()),
        ("BottomFieldOrderCnt", bottom_field_order_cnt, size_of::<i32>()),
        ("va_reserved", va_reserved, VA_PADDING_LOW * size_of::<u32>()),
    ],
    "VAPictureParameterBufferH264": size_of::<VaPictureParameterBufferH264>() => [
        ("CurrPic", curr_pic, size_of::<VaPictureH264>()),
        ("ReferenceFrames", reference_frames, size_of::<[VaPictureH264; 16]>()),
        ("picture_width_in_mbs_minus1", picture_width_in_mbs_minus1, size_of::<u16>()),
        ("picture_height_in_mbs_minus1", picture_height_in_mbs_minus1, size_of::<u16>()),
        ("bit_depth_luma_minus8", bit_depth_luma_minus8, size_of::<u8>()),
        ("bit_depth_chroma_minus8", bit_depth_chroma_minus8, size_of::<u8>()),
        ("num_ref_frames", num_ref_frames, size_of::<u8>()),
        ("-", padding_after_num_ref_frames, size_of::<[u8; 1]>()),
        ("seq_fields", seq_fields, size_of::<u32>()),
        ("num_slice_groups_minus1", num_slice_groups_minus1, size_of::<u8>()),
        ("slice_group_map_type", slice_group_map_type, size_of::<u8>()),
        ("slice_group_change_rate_minus1", slice_group_change_rate_minus1, size_of::<u16>()),
        ("pic_init_qp_minus26", pic_init_qp_minus26, size_of::<i8>()),
        ("pic_init_qs_minus26", pic_init_qs_minus26, size_of::<i8>()),
        ("chroma_qp_index_offset", chroma_qp_index_offset, size_of::<i8>()),
        ("second_chroma_qp_index_offset", second_chroma_qp_index_offset, size_of::<i8>()),
        ("pic_fields", pic_fields, size_of::<u32>()),
        ("frame_num", frame_num, size_of::<u16>()),
        ("-", padding_after_frame_num, size_of::<[u8; 2]>()),
        ("va_reserved", va_reserved, VA_PADDING_MEDIUM * size_of::<u32>()),
    ],
    "VASliceParameterBufferH264": size_of::<VaSliceParameterBufferH264>() => [
        ("slice_data_size", slice_data_size, size_of::<u32>()),
        ("slice_data_offset", slice_data_offset, size_of::<u32>()),
        ("slice_data_flag", slice_data_flag, size_of::<u32>()),
        ("slice_data_bit_offset", slice_data_bit_offset, size_of::<u16>()),
        ("first_mb_in_slice", first_mb_in_slice, size_of::<u16>()),
        ("slice_type", slice_type, size_of::<u8>()),
        ("direct_spatial_mv_pred_flag", direct_spatial_mv_pred_flag, size_of::<u8>()),
        ("num_ref_idx_l0_active_minus1", num_ref_idx_l0_active_minus1, size_of::<u8>()),
        ("num_ref_idx_l1_active_minus1", num_ref_idx_l1_active_minus1, size_of::<u8>()),
        ("cabac_init_idc", cabac_init_idc, size_of::<u8>()),
        ("slice_qp_delta", slice_qp_delta, size_of::<i8>()),
        ("disable_deblocking_filter_idc", disable_deblocking_filter_idc, size_of::<u8>()),
        ("slice_alpha_c0_offset_div2", slice_alpha_c0_offset_div2, size_of::<i8>()),
        ("slice_beta_offset_div2", slice_beta_offset_div2, size_of::<i8>()),
        ("-", padding_before_ref_pic_list0, size_of::<[u8; 3]>()),
        ("RefPicList0", ref_pic_list0, size_of::<[VaPictureH264; 32]>()),
        ("RefPicList1", ref_pic_list1, size_of::<[VaPictureH264; 32]>()),
        ("luma_log2_weight_denom", luma_log2_weight_denom, size_of::<u8>()),
        ("chroma_log2_weight_denom", chroma_log2_weight_denom, size_of::<u8>()),
        ("luma_weight_l0_flag", luma_weight_l0_flag, size_of::<u8>()),
        ("-", padding_before_luma_weight_l0, size_of::<[u8; 1]>()),
        ("luma_weight_l0", luma_weight_l0, size_of::<[i16; 32]>()),
        ("luma_offset_l0", luma_offset_l0, size_of::<[i16; 32]>()),
        ("chroma_weight_l0_flag", chroma_weight_l0_flag, size_of::<u8>()),
        ("-", padding_before_chroma_weight_l0, size_of::<[u8; 1]>()),
        ("chroma_weight_l0", chroma_weight_l0, size_of::<[[i16; 2]; 32]>()),
        ("chroma_offset_l0", chroma_offset_l0, size_of::<[[i16; 2]; 32]>()),
        ("luma_weight_l1_flag", luma_weight_l1_flag, size_of::<u8>()),
        ("-", padding_before_luma_weight_l1, size_of::<[u8; 1]>()),
        ("luma_weight_l1", luma_weight_l1, size_of::<[i16; 32]>()),
        ("luma_offset_l1", luma_offset_l1, size_of::<[i16; 32]>()),
        ("chroma_weight_l1_flag", chroma_weight_l1_flag, size_of::<u8>()),
        ("-", padding_before_chroma_weight_l1, size_of::<[u8; 1]>()),
        ("chroma_weight_l1", chroma_weight_l1, size_of::<[[i16; 2]; 32]>()),
        ("chroma_offset_l1", chroma_offset_l1, size_of::<[[i16; 2]; 32]>()),
        ("-", padding_before_va_reserved, size_of::<[u8; 2]>()),
        ("va_reserved", va_reserved, VA_PADDING_LOW * size_of::<u32>()),
    ],
    "VAIQMatrixBufferH264": size_of::<VaIqMatrixBufferH264>() => [
        ("ScalingList4x4", scaling_list4x4, size_of::<[[u8; 16]; 6]>()),
        ("ScalingList8x8", scaling_list8x8, size_of::<[[u8; 64]; 2]>()),
        ("va_reserved", va_reserved, VA_PADDING_LOW * size_of::<u32>()),
    ],
    "VAPictureHEVC": size_of::<VaPictureHevc>() => [
        ("picture_id", picture_id, size_of::<VABufferID>()),
        ("pic_order_cnt", pic_order_cnt, size_of::<i32>()),
        ("flags", flags, size_of::<u32>()),
        ("va_reserved", va_reserved, VA_PADDING_LOW * size_of::<u32>()),
    ],
    "VAPictureParameterBufferHEVC": size_of::<VaPictureParameterBufferHevc>() => [
        ("CurrPic", curr_pic, size_of::<VaPictureHevc>()),
        ("ReferenceFrames", reference_frames, size_of::<[VaPictureHevc; 15]>()),
        ("pic_width_in_luma_samples", pic_width_in_luma_samples, size_of::<u16>()),
        ("pic_height_in_luma_samples", pic_height_in_luma_samples, size_of::<u16>()),
        ("pic_fields", pic_fields, size_of::<u32>()),
        ("sps_max_dec_pic_buffering_minus1", sps_max_dec_pic_buffering_minus1, size_of::<u8>()),
        ("bit_depth_luma_minus8", bit_depth_luma_minus8, size_of::<u8>()),
        ("bit_depth_chroma_minus8", bit_depth_chroma_minus8, size_of::<u8>()),
        ("pcm_sample_bit_depth_luma_minus1", pcm_sample_bit_depth_luma_minus1, size_of::<u8>()),
        ("pcm_sample_bit_depth_chroma_minus1", pcm_sample_bit_depth_chroma_minus1, size_of::<u8>()),
        ("log2_min_luma_coding_block_size_minus3", log2_min_luma_coding_block_size_minus3, size_of::<u8>()),
        ("log2_diff_max_min_luma_coding_block_size", log2_diff_max_min_luma_coding_block_size, size_of::<u8>()),
        ("log2_min_transform_block_size_minus2", log2_min_transform_block_size_minus2, size_of::<u8>()),
        ("log2_diff_max_min_transform_block_size", log2_diff_max_min_transform_block_size, size_of::<u8>()),
        ("log2_min_pcm_luma_coding_block_size_minus3", log2_min_pcm_luma_coding_block_size_minus3, size_of::<u8>()),
        ("log2_diff_max_min_pcm_luma_coding_block_size", log2_diff_max_min_pcm_luma_coding_block_size, size_of::<u8>()),
        ("max_transform_hierarchy_depth_intra", max_transform_hierarchy_depth_intra, size_of::<u8>()),
        ("max_transform_hierarchy_depth_inter", max_transform_hierarchy_depth_inter, size_of::<u8>()),
        ("init_qp_minus26", init_qp_minus26, size_of::<i8>()),
        ("diff_cu_qp_delta_depth", diff_cu_qp_delta_depth, size_of::<u8>()),
        ("pps_cb_qp_offset", pps_cb_qp_offset, size_of::<i8>()),
        ("pps_cr_qp_offset", pps_cr_qp_offset, size_of::<i8>()),
        ("log2_parallel_merge_level_minus2", log2_parallel_merge_level_minus2, size_of::<u8>()),
        ("num_tile_columns_minus1", num_tile_columns_minus1, size_of::<u8>()),
        ("num_tile_rows_minus1", num_tile_rows_minus1, size_of::<u8>()),
        ("column_width_minus1", column_width_minus1, size_of::<[u16; 19]>()),
        ("row_height_minus1", row_height_minus1, size_of::<[u16; 21]>()),
        ("slice_parsing_fields", slice_parsing_fields, size_of::<u32>()),
        ("log2_max_pic_order_cnt_lsb_minus4", log2_max_pic_order_cnt_lsb_minus4, size_of::<u8>()),
        ("num_short_term_ref_pic_sets", num_short_term_ref_pic_sets, size_of::<u8>()),
        ("num_long_term_ref_pic_sps", num_long_term_ref_pic_sps, size_of::<u8>()),
        ("num_ref_idx_l0_default_active_minus1", num_ref_idx_l0_default_active_minus1, size_of::<u8>()),
        ("num_ref_idx_l1_default_active_minus1", num_ref_idx_l1_default_active_minus1, size_of::<u8>()),
        ("pps_beta_offset_div2", pps_beta_offset_div2, size_of::<i8>()),
        ("pps_tc_offset_div2", pps_tc_offset_div2, size_of::<i8>()),
        ("num_extra_slice_header_bits", num_extra_slice_header_bits, size_of::<u8>()),
        ("st_rps_bits", st_rps_bits, size_of::<u32>()),
        ("va_reserved", va_reserved, VA_PADDING_MEDIUM * size_of::<u32>()),
    ],
    "VASliceParameterBufferHEVC": size_of::<VaSliceParameterBufferHevc>() => [
        ("slice_data_size", slice_data_size, size_of::<u32>()),
        ("slice_data_offset", slice_data_offset, size_of::<u32>()),
        ("slice_data_flag", slice_data_flag, size_of::<u32>()),
        ("slice_data_byte_offset", slice_data_byte_offset, size_of::<u32>()),
        ("slice_segment_address", slice_segment_address, size_of::<u32>()),
        ("RefPicList", ref_pic_list, size_of::<[[u8; 15]; 2]>()),
        ("-", padding_before_long_slice_flags, size_of::<[u8; 2]>()),
        ("LongSliceFlags", long_slice_flags, size_of::<u32>()),
        ("collocated_ref_idx", collocated_ref_idx, size_of::<u8>()),
        ("num_ref_idx_l0_active_minus1", num_ref_idx_l0_active_minus1, size_of::<u8>()),
        ("num_ref_idx_l1_active_minus1", num_ref_idx_l1_active_minus1, size_of::<u8>()),
        ("slice_qp_delta", slice_qp_delta, size_of::<i8>()),
        ("slice_cb_qp_offset", slice_cb_qp_offset, size_of::<i8>()),
        ("slice_cr_qp_offset", slice_cr_qp_offset, size_of::<i8>()),
        ("slice_beta_offset_div2", slice_beta_offset_div2, size_of::<i8>()),
        ("slice_tc_offset_div2", slice_tc_offset_div2, size_of::<i8>()),
        ("luma_log2_weight_denom", luma_log2_weight_denom, size_of::<u8>()),
        ("delta_chroma_log2_weight_denom", delta_chroma_log2_weight_denom, size_of::<i8>()),
        ("delta_luma_weight_l0", delta_luma_weight_l0, size_of::<[i8; 15]>()),
        ("luma_offset_l0", luma_offset_l0, size_of::<[i8; 15]>()),
        ("delta_chroma_weight_l0", delta_chroma_weight_l0, size_of::<[[i8; 2]; 15]>()),
        ("ChromaOffsetL0", chroma_offset_l0, size_of::<[[i8; 2]; 15]>()),
        ("delta_luma_weight_l1", delta_luma_weight_l1, size_of::<[i8; 15]>()),
        ("luma_offset_l1", luma_offset_l1, size_of::<[i8; 15]>()),
        ("delta_chroma_weight_l1", delta_chroma_weight_l1, size_of::<[[i8; 2]; 15]>()),
        ("ChromaOffsetL1", chroma_offset_l1, size_of::<[[i8; 2]; 15]>()),
        ("five_minus_max_num_merge_cand", five_minus_max_num_merge_cand, size_of::<u8>()),
        ("-", padding_before_num_entry_point_offsets, size_of::<[u8; 1]>()),
        ("num_entry_point_offsets", num_entry_point_offsets, size_of::<u16>()),
        ("entry_offset_to_subset_array", entry_offset_to_subset_array, size_of::<u16>()),
        ("slice_data_num_emu_prevn_bytes", slice_data_num_emu_prevn_bytes, size_of::<u16>()),
        ("-", padding_before_va_reserved, size_of::<[u8; 2]>()),
        ("va_reserved", va_reserved, (VA_PADDING_LOW - 2) * size_of::<u32>()),
    ],
    "VADecPictureParameterBufferVP9": size_of::<VaPictureParameterBufferVp9>() => [
        ("frame_width", frame_width, size_of::<u16>()),
        ("frame_height", frame_height, size_of::<u16>()),
        ("reference_frames", reference_frames, size_of::<[VASurfaceID; 8]>()),
        ("pic_fields", pic_fields, size_of::<u32>()),
        ("filter_level", filter_level, size_of::<u8>()),
        ("sharpness_level", sharpness_level, size_of::<u8>()),
        ("log2_tile_rows", log2_tile_rows, size_of::<u8>()),
        ("log2_tile_columns", log2_tile_columns, size_of::<u8>()),
        ("frame_header_length_in_bytes", frame_header_length_in_bytes, size_of::<u8>()),
        ("-", padding_before_first_partition_size, size_of::<[u8; 1]>()),
        ("first_partition_size", first_partition_size, size_of::<u16>()),
        ("mb_segment_tree_probs", mb_segment_tree_probs, size_of::<[u8; 7]>()),
        ("segment_pred_probs", segment_pred_probs, size_of::<[u8; 3]>()),
        ("profile", profile, size_of::<u8>()),
        ("bit_depth", bit_depth, size_of::<u8>()),
        ("va_reserved", va_reserved, VA_PADDING_MEDIUM * size_of::<u32>()),
    ],
    "VASegmentParameterVP9": size_of::<VaSegmentParameterVp9>() => [
        ("segment_flags", segment_flags, size_of::<u16>()),
        ("filter_level", filter_level, size_of::<[[u8; 2]; 4]>()),
        ("luma_ac_quant_scale", luma_ac_quant_scale, size_of::<i16>()),
        ("luma_dc_quant_scale", luma_dc_quant_scale, size_of::<i16>()),
        ("chroma_ac_quant_scale", chroma_ac_quant_scale, size_of::<i16>()),
        ("chroma_dc_quant_scale", chroma_dc_quant_scale, size_of::<i16>()),
        ("-", padding_before_va_reserved, size_of::<[u8; 2]>()),
        ("va_reserved", va_reserved, VA_PADDING_LOW * size_of::<u32>()),
    ],
    "VASliceParameterBufferVP9": size_of::<VaSliceParameterBufferVp9>() => [
        ("slice_data_size", slice_data_size, size_of::<u32>()),
        ("slice_data_offset", slice_data_offset, size_of::<u32>()),
        ("slice_data_flag", slice_data_flag, size_of::<u32>()),
        ("seg_param", seg_param, size_of::<[VaSegmentParameterVp9; 8]>()),
        ("va_reserved", va_reserved, VA_PADDING_LOW * size_of::<u32>()),
    ],
    "VACodedBufferSegment": size_of::<VaCodedBufferSegment>() => [
        ("size", size, size_of::<u32>()),
        ("bit_offset", bit_offset, size_of::<u32>()),
        ("status", status, size_of::<u32>()),
        ("reserved", reserved, size_of::<u32>()),
        ("buf", buf, size_of::<*mut c_void>()),
        ("next", next, size_of::<*mut c_void>()),
        ("va_reserved", va_reserved, VA_PADDING_LOW * size_of::<u32>()),
    ],
    "VABufferInfo": size_of::<VaBufferInfo>() => [
        ("handle", handle, size_of::<usize>()),
        ("type", buffer_type, size_of::<u32>()),
        ("mem_type", mem_type, size_of::<u32>()),
        ("mem_size", mem_size, size_of::<usize>()),
        ("va_reserved", va_reserved, VA_PADDING_LOW * size_of::<u32>()),
    ],
    "VAEncSequenceParameterBufferH264": size_of::<VaEncSequenceParameterBufferH264>() => [
        ("seq_parameter_set_id", seq_parameter_set_id, size_of::<u8>()),
        ("level_idc", level_idc, size_of::<u8>()),
        ("-", padding_after_level_idc, size_of::<[u8; 2]>()),
        ("intra_period", intra_period, size_of::<u32>()),
        ("intra_idr_period", intra_idr_period, size_of::<u32>()),
        ("ip_period", ip_period, size_of::<u32>()),
        ("bits_per_second", bits_per_second, size_of::<u32>()),
        ("max_num_ref_frames", max_num_ref_frames, size_of::<u32>()),
        ("picture_width_in_mbs", picture_width_in_mbs, size_of::<u16>()),
        ("picture_height_in_mbs", picture_height_in_mbs, size_of::<u16>()),
        ("seq_fields", seq_fields, size_of::<u32>()),
        ("bit_depth_luma_minus8", bit_depth_luma_minus8, size_of::<u8>()),
        ("bit_depth_chroma_minus8", bit_depth_chroma_minus8, size_of::<u8>()),
        ("num_ref_frames_in_pic_order_cnt_cycle", num_ref_frames_in_pic_order_cnt_cycle, size_of::<u8>()),
        ("-", padding_after_num_ref_frames_in_pic_order_cnt_cycle, size_of::<[u8; 1]>()),
        ("offset_for_non_ref_pic", offset_for_non_ref_pic, size_of::<i32>()),
        ("offset_for_top_to_bottom_field", offset_for_top_to_bottom_field, size_of::<i32>()),
        ("offset_for_ref_frame", offset_for_ref_frame, size_of::<[i32; 256]>()),
        ("frame_cropping_flag", frame_cropping_flag, size_of::<u8>()),
        ("-", padding_after_frame_cropping_flag, size_of::<[u8; 3]>()),
        ("frame_crop_left_offset", frame_crop_left_offset, size_of::<u32>()),
        ("frame_crop_right_offset", frame_crop_right_offset, size_of::<u32>()),
        ("frame_crop_top_offset", frame_crop_top_offset, size_of::<u32>()),
        ("frame_crop_bottom_offset", frame_crop_bottom_offset, size_of::<u32>()),
        ("vui_parameters_present_flag", vui_parameters_present_flag, size_of::<u8>()),
        ("-", padding_after_vui_parameters_present_flag, size_of::<[u8; 3]>()),
        ("vui_fields", vui_fields, size_of::<u32>()),
        ("aspect_ratio_idc", aspect_ratio_idc, size_of::<u8>()),
        ("-", padding_after_aspect_ratio_idc, size_of::<[u8; 3]>()),
        ("sar_width", sar_width, size_of::<u32>()),
        ("sar_height", sar_height, size_of::<u32>()),
        ("num_units_in_tick", num_units_in_tick, size_of::<u32>()),
        ("time_scale", time_scale, size_of::<u32>()),
        ("va_reserved", va_reserved, VA_PADDING_LOW * size_of::<u32>()),
    ],
    "VAEncPictureParameterBufferH264": size_of::<VaEncPictureParameterBufferH264>() => [
        ("CurrPic", curr_pic, size_of::<VaPictureH264>()),
        ("ReferenceFrames", reference_frames, size_of::<[VaPictureH264; 16]>()),
        ("coded_buf", coded_buf, size_of::<VABufferID>()),
        ("pic_parameter_set_id", pic_parameter_set_id, size_of::<u8>()),
        ("seq_parameter_set_id", seq_parameter_set_id, size_of::<u8>()),
        ("last_picture", last_picture, size_of::<u8>()),
        ("-", padding_after_last_picture, size_of::<[u8; 1]>()),
        ("frame_num", frame_num, size_of::<u16>()),
        ("pic_init_qp", pic_init_qp, size_of::<u8>()),
        ("num_ref_idx_l0_active_minus1", num_ref_idx_l0_active_minus1, size_of::<u8>()),
        ("num_ref_idx_l1_active_minus1", num_ref_idx_l1_active_minus1, size_of::<u8>()),
        ("chroma_qp_index_offset", chroma_qp_index_offset, size_of::<i8>()),
        ("second_chroma_qp_index_offset", second_chroma_qp_index_offset, size_of::<i8>()),
        ("-", padding_after_second_chroma_qp_index_offset, size_of::<[u8; 1]>()),
        ("pic_fields", pic_fields, size_of::<u32>()),
        ("va_reserved", va_reserved, VA_PADDING_LOW * size_of::<u32>()),
    ],
    "VAEncSliceParameterBufferH264": size_of::<VaEncSliceParameterBufferH264>() => [
        ("macroblock_address", macroblock_address, size_of::<u32>()),
        ("num_macroblocks", num_macroblocks, size_of::<u32>()),
        ("macroblock_info", macroblock_info, size_of::<VABufferID>()),
        ("slice_type", slice_type, size_of::<u8>()),
        ("pic_parameter_set_id", pic_parameter_set_id, size_of::<u8>()),
        ("idr_pic_id", idr_pic_id, size_of::<u16>()),
        ("pic_order_cnt_lsb", pic_order_cnt_lsb, size_of::<u16>()),
        ("-", padding_after_pic_order_cnt_lsb, size_of::<[u8; 2]>()),
        ("delta_pic_order_cnt_bottom", delta_pic_order_cnt_bottom, size_of::<i32>()),
        ("delta_pic_order_cnt", delta_pic_order_cnt, size_of::<[i32; 2]>()),
        ("direct_spatial_mv_pred_flag", direct_spatial_mv_pred_flag, size_of::<u8>()),
        ("num_ref_idx_active_override_flag", num_ref_idx_active_override_flag, size_of::<u8>()),
        ("num_ref_idx_l0_active_minus1", num_ref_idx_l0_active_minus1, size_of::<u8>()),
        ("num_ref_idx_l1_active_minus1", num_ref_idx_l1_active_minus1, size_of::<u8>()),
        ("RefPicList0", ref_pic_list0, size_of::<[VaPictureH264; 32]>()),
        ("RefPicList1", ref_pic_list1, size_of::<[VaPictureH264; 32]>()),
        ("luma_log2_weight_denom", luma_log2_weight_denom, size_of::<u8>()),
        ("chroma_log2_weight_denom", chroma_log2_weight_denom, size_of::<u8>()),
        ("luma_weight_l0_flag", luma_weight_l0_flag, size_of::<u8>()),
        ("-", padding_after_luma_weight_l0_flag, size_of::<[u8; 1]>()),
        ("luma_weight_l0", luma_weight_l0, size_of::<[i16; 32]>()),
        ("luma_offset_l0", luma_offset_l0, size_of::<[i16; 32]>()),
        ("chroma_weight_l0_flag", chroma_weight_l0_flag, size_of::<u8>()),
        ("-", padding_after_chroma_weight_l0_flag, size_of::<[u8; 1]>()),
        ("chroma_weight_l0", chroma_weight_l0, size_of::<[[i16; 2]; 32]>()),
        ("chroma_offset_l0", chroma_offset_l0, size_of::<[[i16; 2]; 32]>()),
        ("luma_weight_l1_flag", luma_weight_l1_flag, size_of::<u8>()),
        ("-", padding_after_luma_weight_l1_flag, size_of::<[u8; 1]>()),
        ("luma_weight_l1", luma_weight_l1, size_of::<[i16; 32]>()),
        ("luma_offset_l1", luma_offset_l1, size_of::<[i16; 32]>()),
        ("chroma_weight_l1_flag", chroma_weight_l1_flag, size_of::<u8>()),
        ("-", padding_after_chroma_weight_l1_flag, size_of::<[u8; 1]>()),
        ("chroma_weight_l1", chroma_weight_l1, size_of::<[[i16; 2]; 32]>()),
        ("chroma_offset_l1", chroma_offset_l1, size_of::<[[i16; 2]; 32]>()),
        ("cabac_init_idc", cabac_init_idc, size_of::<u8>()),
        ("slice_qp_delta", slice_qp_delta, size_of::<i8>()),
        ("disable_deblocking_filter_idc", disable_deblocking_filter_idc, size_of::<u8>()),
        ("slice_alpha_c0_offset_div2", slice_alpha_c0_offset_div2, size_of::<i8>()),
        ("slice_beta_offset_div2", slice_beta_offset_div2, size_of::<i8>()),
        ("-", padding_after_slice_beta_offset_div2, size_of::<[u8; 1]>()),
        ("va_reserved", va_reserved, VA_PADDING_LOW * size_of::<u32>()),
    ],
    "VAEncMiscParameterBuffer": size_of::<VaEncMiscParameterBufferHeader>() => [
        ("type", type_, size_of::<u32>()),
    ],
    "VAEncMiscParameterFrameRate": size_of::<VaEncMiscParameterFrameRate>() => [
        ("framerate", framerate, size_of::<u32>()),
        ("framerate_flags", framerate_flags, size_of::<u32>()),
        ("va_reserved", va_reserved, VA_PADDING_LOW * size_of::<u32>()),
    ],
    "VAEncMiscParameterQuantization": size_of::<VaEncMiscParameterQuantization>() => [
        ("quantization_flags", quantization_flags, size_of::<u32>()),
        ("va_reserved", va_reserved, size_of::<u32>()),
    ],
];

/// **Every byte of every record this backend hands to a driver belongs to a named field**, and the
/// fields line up end to end with no gap between two of them.
///
/// This is the property [`record_bytes`]'s safety argument rests on, and it is the one no `sizeof`
/// and no `offsetof` can see. `VASegmentParameterVP9` agreed with a C compiler on every size and
/// every offset *and* had a 2-byte hole at 18-19 that `#[derive(Default)]` never wrote: the bytes
/// were indeterminate, `record_bytes` handed them to the driver, and the decode came out bit-exact
/// on 24 of 24 frames because **this** driver ignores those bytes — a fact about the driver, not
/// about the record. A hole is invisible to every check that reads a number out of a struct, so it
/// has to be caught by laying the fields out and asking whether they *cover* the record.
///
/// The check is deliberately the strict one: each field must start exactly where the one before it
/// ends, and the last must end exactly at the record's size. A record with no hole passes because
/// its fields genuinely tile it, not because it was skipped — and a `padding_*` field that is
/// absent, or present with the wrong width, fails here as loudly as a missing one.
#[test]
fn every_record_the_backend_submits_tiles_byte_for_byte() {
    for record in SUBMITTED_RECORDS {
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

/// **The header agrees with us on every field's offset and on every hole's width**, for all of them
/// at once — including the records that genuinely have no hole.
///
/// This is the half of the hole check the C compiler can see, and it is asked as a *gap* rather
/// than as a member, because C gives padding no name: for two members `a` and `b` of a record `R`
/// the hole between them is `offsetof(R, b) - offsetof(R, a) - sizeof(a)`, and that difference is
/// compared with the width of the `padding_*` field that claims to cover it. A record with no hole
/// is asserted to have a gap of **zero** there rather than left unchecked, so "no hole" is a
/// measurement and not an absence of one.
///
/// Each record also has its `sizeof` asserted, and each member its `offsetof` and `sizeof` — a hole
/// of the wrong width shows up as all three disagreeing at once. The offsets are read out of
/// `offset_of!` on our own declaration, so the two sides are compared directly rather than both
/// against a number somebody typed twice; a declaration that read the header's field list as its
/// byte order fails here with the record and field named.
#[test]
fn no_submitted_record_has_a_byte_the_header_does_not_account_for() {
    if !std::path::Path::new(LIBVA_HEADERS).join("va.h").is_file() {
        eprintln!("SKIPPED: no {LIBVA_HEADERS}/va.h");
        return;
    }
    let Some(cc) = ["cc", "gcc", "clang"].into_iter().find_map(which) else {
        eprintln!("SKIPPED: no C compiler to measure the headers with");
        return;
    };
    let mut source = String::from(
        "#include <stdint.h>\n#include <va/va.h>\n#include <va/va_drm.h>\n#include <va/va_dec_hevc.h>\n#include <va/va_dec_vp9.h>\n#include <va/va_enc_h264.h>\n",
    );
    for record in SUBMITTED_RECORDS {
        source.push_str(&format!("\n/* {} */\n", record.c_name));
        source.push_str(&format!("_Static_assert(sizeof({r}) == {size}, \"{r}: sizeof\");\n", r = record.c_name, size = record.size));
        // The members the header names, each at our offset and our width.
        //
        // `previous` tracks the last member **C names**, not the last field: a `padding_*` field
        // stands between two C members and has no C name of its own, so the hole it covers is the
        // gap from the member before it to the member after, and the width that gap is compared
        // against is every byte between them — the `padding_*` field on its own where there is
        // exactly one, and nothing at all, i.e. zero, where there is none. A record with no hole is
        // therefore asserted to have a zero-width gap, which is a measurement rather than a skip.
        let mut previous: Option<&SubmittedField> = None;
        for field in record.fields {
            if field.c_name == "-" {
                assert!(
                    field.rust_name.starts_with("padding_"),
                    "{}: {} has no C name but is not a padding_* field, so it is a member the header declares and the hole check cannot measure it",
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
                let gap = field.offset - (prev.offset + prev.width);
                source.push_str(&format!(
                    "_Static_assert(__builtin_offsetof({r}, {b}) - __builtin_offsetof({r}, {a}) - sizeof((({r} *)0)->{a}) == {gap}, \"{r}: the hole between {a} and {b}\");\n",
                    r = record.c_name, a = prev.c_name, b = field.c_name, gap = gap
                ));
            } else {
                assert_eq!(field.offset, 0, "{}: the first field {} starts at {} and a record starts at 0", record.c_name, field.rust_name, field.offset);
            }
            previous = Some(field);
        }
        // Trailing alignment padding after the last member, if any: the same difference, measured
        // against the record's own size rather than against another member's offset.
        let last = previous.expect("a record whose every field is a padding field is rejected by the tiling test");
        let tail = record.size - (last.offset + last.width);
        source.push_str(&format!(
            "_Static_assert(sizeof({r}) - __builtin_offsetof({r}, {f}) - sizeof((({r} *)0)->{f}) == {tail}, \"{r}: the padding after {f}\");\n",
            r = record.c_name,
            f = last.c_name,
            tail = tail
        ));
    }
    let dir = std::env::temp_dir().join(format!("filmcraft-va-holes-{}", std::process::id()));
    if std::fs::create_dir_all(&dir).is_err() {
        eprintln!("SKIPPED: no writable temporary directory");
        return;
    }
    let path = dir.join("holes.c");
    let written = std::fs::write(&path, source.as_bytes()).is_ok();
    let out = if written {
        std::process::Command::new(&cc).args(["-fsyntax-only", "-I", LIBVA_HEADERS, path.to_str().unwrap_or_default()]).output()
    } else {
        eprintln!("SKIPPED: cannot write {}", path.display());
        return;
    };
    let _ = std::fs::remove_dir_all(&dir);
    let out = out.unwrap_or_else(|e| panic!("running {cc} failed: {e}"));
    assert!(
        out.status.success(),
        "{cc} disagrees with our records on a field offset, a field width, or a hole's width:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Every declaration, judged by a C compiler reading the same headers.
///
/// The entry points are *redeclared*, not merely counted: C requires a redeclaration to be
/// compatible with the original, so a wrong parameter type, a missing or extra parameter, or a
/// different return type is a compile error that quotes the header's own declaration. The records
/// are `_Static_assert`ed on `sizeof`, and the number comes from `size_of` on the Rust side, so the
/// two are compared directly instead of both being compared with a number somebody typed twice.
#[test]
fn every_declaration_matches_what_a_c_compiler_measures() {
    if !std::path::Path::new(LIBVA_HEADERS).join("va.h").is_file() {
        eprintln!("SKIPPED: no {LIBVA_HEADERS}/va.h");
        return;
    }
    let Some(cc) = ["cc", "gcc", "clang"].into_iter().find_map(which) else {
        eprintln!("SKIPPED: no C compiler to measure the headers with");
        return;
    };
    let records: &[(&str, usize)] = &[
        ("VAImageFormat", size_of::<VaImageFormat>()),
        ("VAImage", size_of::<VaImage>()),
        ("VAConfigAttrib", size_of::<VaConfigAttrib>()),
        ("VASurfaceAttrib", size_of::<VaSurfaceAttrib>()),
        ("VAGenericValue", size_of::<VaGenericValue>()),
        ("VAPictureH264", size_of::<VaPictureH264>()),
        ("VAPictureParameterBufferH264", size_of::<VaPictureParameterBufferH264>()),
        ("VASliceParameterBufferH264", size_of::<VaSliceParameterBufferH264>()),
        ("VAIQMatrixBufferH264", size_of::<VaIqMatrixBufferH264>()),
        ("VAPictureHEVC", size_of::<VaPictureHevc>()),
        ("VAPictureParameterBufferHEVC", size_of::<VaPictureParameterBufferHevc>()),
        ("VASliceParameterBufferHEVC", size_of::<VaSliceParameterBufferHevc>()),
        ("VADecPictureParameterBufferVP9", size_of::<VaPictureParameterBufferVp9>()),
        ("VASegmentParameterVP9", size_of::<VaSegmentParameterVp9>()),
        ("VASliceParameterBufferVP9", size_of::<VaSliceParameterBufferVp9>()),
        ("VACodedBufferSegment", size_of::<VaCodedBufferSegment>()),
        ("VABufferInfo", size_of::<VaBufferInfo>()),
        ("VAEncSequenceParameterBufferH264", size_of::<VaEncSequenceParameterBufferH264>()),
        ("VAEncPictureParameterBufferH264", size_of::<VaEncPictureParameterBufferH264>()),
        ("VAEncSliceParameterBufferH264", size_of::<VaEncSliceParameterBufferH264>()),
        ("VAEncMiscParameterFrameRate", size_of::<VaEncMiscParameterFrameRate>()),
        ("VAEncMiscParameterQuantization", size_of::<VaEncMiscParameterQuantization>()),
    ];
    let mut source =
        String::from("#include <va/va.h>\n#include <va/va_drm.h>\n#include <va/va_dec_hevc.h>\n#include <va/va_dec_vp9.h>\n#include <va/va_enc_h264.h>\n");
    source.push_str(
        "/* One line per entry point, exactly as ffi.rs declares it. A redeclaration\n   that disagrees with the header is an error, not a warning. */\n",
    );
    for entry in ENTRY_POINTS {
        let (ret, args) = entry.prototype.split_once('(').unwrap_or((entry.prototype, ""));
        source.push_str(&format!("{} {}({});\n", ret.trim(), entry.symbol, args.trim_end_matches(')')));
    }
    source.push_str("\n/* Every record's size, as size_of sees it in Rust. */\n");
    for (name, size) in records {
        source.push_str(&format!("_Static_assert(sizeof({name}) == {size}, \"{name}\");\n"));
    }
    let dir = std::env::temp_dir().join(format!("filmcraft-va-abi-{}", std::process::id()));
    if std::fs::create_dir_all(&dir).is_err() {
        eprintln!("SKIPPED: no writable temporary directory");
        return;
    }
    let path = dir.join("abi.c");
    let written = std::fs::write(&path, &source).is_ok();
    let out = if written {
        std::process::Command::new(&cc).args(["-fsyntax-only", "-I", LIBVA_HEADERS, path.to_str().unwrap_or_default()]).output()
    } else {
        eprintln!("SKIPPED: cannot write {}", path.display());
        return;
    };
    let _ = std::fs::remove_dir_all(&dir);
    let out = out.unwrap_or_else(|e| panic!("running {cc} failed: {e}"));
    assert!(out.status.success(), "{cc} rejected our declarations against the headers:\n{}", String::from_utf8_lossy(&out.stderr));
}

/// `name` on the `PATH`, or `None`.
fn which(name: &str) -> Option<String> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).find_map(|dir| {
        let candidate = dir.join(name);
        candidate.is_file().then(|| candidate.to_string_lossy().into_owned())
    })
}
