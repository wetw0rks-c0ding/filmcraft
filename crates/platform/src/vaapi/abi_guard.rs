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
//!
//! **A note on stdout is not loud**, though: `cargo test` captures a passing test's output, so on a
//! machine without the headers every check above returns early and the suite is green while nothing
//! was verified. [`the_abi_guard_ran_on_this_machine_or_announced_that_it_did_not`] at the end of
//! this file is the one test here that fails in that state instead, and it is what a reader of a
//! green run should look for.

use super::*;
use std::mem::{offset_of, size_of};

/// Where the libva headers live on this machine.
///
/// **This is a path to check, never an include flag.** The C sources below say `#include <va/va.h>`,
/// so compiling them with `-I /usr/include/va` asks the compiler to resolve that as
/// `/usr/include/va/va/va.h`; it only worked because `/usr/include` is searched as a fallback after
/// every `-I` has missed, which is exactly the kind of "right today, wrong the moment anyone relies
/// on the flag" this project keeps writing down. The default system include path already resolves
/// `<va/…>` and `<drm/…>`, so no `-I` is passed at all.
const LIBVA_HEADERS: &str = "/usr/include/va";

/// Where the DRM headers live, for the one record that is not a libva record.
const DRM_HEADERS: &str = "/usr/include/drm";

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
        std::process::Command::new(&cc).args(["-fsyntax-only", path.to_str().unwrap_or_default()]).output()
    } else {
        eprintln!("SKIPPED: cannot write {}", path.display());
        return;
    };
    let _ = std::fs::remove_dir_all(&dir);
    let out = out.unwrap_or_else(|e| panic!("running {cc} failed: {e}"));
    assert!(out.status.success(), "{cc} disagrees with our VP9 records:\n{}", String::from_utf8_lossy(&out.stderr));
}

// ---------------------------------------------------------------------------------------------
// The AV1 bitfield unions, measured by *running* a C program rather than by reading the header.
//
// Every other check in this module is a `_Static_assert`, which works because `sizeof` and
// `offsetof` are compile-time constants. A bitfield's position is not: nothing in C can ask "where
// does member `x` of this union start" without setting it and reading the word back, which is a
// runtime operation. So this one compiles and **runs**, and compares what the C compiler produced
// with a table written here — two independent sides, either of which can be wrong.
//
// Why it is not a table compared with itself: the header declares each union's members in an order
// that happens to be right on the target that compiled it and says nothing about any other, so
// "I read the order off the header and wrote it down" is not a measurement. Every entry below was
// produced by `printf("%u %u", __builtin_ctz(v), 31 - __builtin_clz(v))` after setting that member to
// all ones — the low set bit is where the member starts and the high set bit is where it ends.
//
// This matters because a bit position in these records is invisible to every other gate: a union has
// one `offsetof`, so a member packed one bit too high agrees with the C compiler on the record's
// size, on every field offset and on every hole, and produces a *picture*.
// ---------------------------------------------------------------------------------------------

/// One union member of an AV1 record: which struct, which union, **which member of the anonymous
/// struct inside that union** (libva calls it `bits` in seven of the eight and `fields` in
/// `seq_info_fields`, `va_dec_av1.h:289` — so it is carried per row rather than assumed), the
/// member's own C name, and the bit range a C compiler puts it at.
const AV1_BITFIELDS: &[(&str, &str, &str, &str, u32, u32)] = &[
    // VADecPictureParameterBufferAV1.seq_info_fields (va_dec_av1.h:266-291), uint32_t
    ("VADecPictureParameterBufferAV1", "seq_info_fields", "fields", "still_picture", 0, 0),
    ("VADecPictureParameterBufferAV1", "seq_info_fields", "fields", "use_128x128_superblock", 1, 1),
    ("VADecPictureParameterBufferAV1", "seq_info_fields", "fields", "enable_filter_intra", 2, 2),
    ("VADecPictureParameterBufferAV1", "seq_info_fields", "fields", "enable_intra_edge_filter", 3, 3),
    ("VADecPictureParameterBufferAV1", "seq_info_fields", "fields", "enable_interintra_compound", 4, 4),
    ("VADecPictureParameterBufferAV1", "seq_info_fields", "fields", "enable_masked_compound", 5, 5),
    ("VADecPictureParameterBufferAV1", "seq_info_fields", "fields", "enable_dual_filter", 6, 6),
    ("VADecPictureParameterBufferAV1", "seq_info_fields", "fields", "enable_order_hint", 7, 7),
    ("VADecPictureParameterBufferAV1", "seq_info_fields", "fields", "enable_jnt_comp", 8, 8),
    ("VADecPictureParameterBufferAV1", "seq_info_fields", "fields", "enable_cdef", 9, 9),
    ("VADecPictureParameterBufferAV1", "seq_info_fields", "fields", "mono_chrome", 10, 10),
    ("VADecPictureParameterBufferAV1", "seq_info_fields", "fields", "color_range", 11, 11),
    ("VADecPictureParameterBufferAV1", "seq_info_fields", "fields", "subsampling_x", 12, 12),
    ("VADecPictureParameterBufferAV1", "seq_info_fields", "fields", "subsampling_y", 13, 13),
    ("VADecPictureParameterBufferAV1", "seq_info_fields", "fields", "chroma_sample_position", 14, 14),
    ("VADecPictureParameterBufferAV1", "seq_info_fields", "fields", "film_grain_params_present", 15, 15),
    ("VADecPictureParameterBufferAV1", "seq_info_fields", "fields", "reserved", 16, 31),
    // VADecPictureParameterBufferAV1.pic_info_fields (va_dec_av1.h:409-447), uint32_t
    ("VADecPictureParameterBufferAV1", "pic_info_fields", "bits", "frame_type", 0, 1),
    ("VADecPictureParameterBufferAV1", "pic_info_fields", "bits", "show_frame", 2, 2),
    ("VADecPictureParameterBufferAV1", "pic_info_fields", "bits", "showable_frame", 3, 3),
    ("VADecPictureParameterBufferAV1", "pic_info_fields", "bits", "error_resilient_mode", 4, 4),
    ("VADecPictureParameterBufferAV1", "pic_info_fields", "bits", "disable_cdf_update", 5, 5),
    ("VADecPictureParameterBufferAV1", "pic_info_fields", "bits", "allow_screen_content_tools", 6, 6),
    ("VADecPictureParameterBufferAV1", "pic_info_fields", "bits", "force_integer_mv", 7, 7),
    ("VADecPictureParameterBufferAV1", "pic_info_fields", "bits", "allow_intrabc", 8, 8),
    ("VADecPictureParameterBufferAV1", "pic_info_fields", "bits", "use_superres", 9, 9),
    ("VADecPictureParameterBufferAV1", "pic_info_fields", "bits", "allow_high_precision_mv", 10, 10),
    ("VADecPictureParameterBufferAV1", "pic_info_fields", "bits", "is_motion_mode_switchable", 11, 11),
    ("VADecPictureParameterBufferAV1", "pic_info_fields", "bits", "use_ref_frame_mvs", 12, 12),
    ("VADecPictureParameterBufferAV1", "pic_info_fields", "bits", "disable_frame_end_update_cdf", 13, 13),
    ("VADecPictureParameterBufferAV1", "pic_info_fields", "bits", "uniform_tile_spacing_flag", 14, 14),
    ("VADecPictureParameterBufferAV1", "pic_info_fields", "bits", "allow_warped_motion", 15, 15),
    ("VADecPictureParameterBufferAV1", "pic_info_fields", "bits", "large_scale_tile", 16, 16),
    ("VADecPictureParameterBufferAV1", "pic_info_fields", "bits", "reserved", 17, 31),
    // VADecPictureParameterBufferAV1.mode_control_fields (va_dec_av1.h:544-576), uint32_t
    ("VADecPictureParameterBufferAV1", "mode_control_fields", "bits", "delta_q_present_flag", 0, 0),
    ("VADecPictureParameterBufferAV1", "mode_control_fields", "bits", "log2_delta_q_res", 1, 2),
    ("VADecPictureParameterBufferAV1", "mode_control_fields", "bits", "delta_lf_present_flag", 3, 3),
    ("VADecPictureParameterBufferAV1", "mode_control_fields", "bits", "log2_delta_lf_res", 4, 5),
    ("VADecPictureParameterBufferAV1", "mode_control_fields", "bits", "delta_lf_multi", 6, 6),
    ("VADecPictureParameterBufferAV1", "mode_control_fields", "bits", "tx_mode", 7, 8),
    ("VADecPictureParameterBufferAV1", "mode_control_fields", "bits", "reference_select", 9, 9),
    ("VADecPictureParameterBufferAV1", "mode_control_fields", "bits", "reduced_tx_set_used", 10, 10),
    ("VADecPictureParameterBufferAV1", "mode_control_fields", "bits", "skip_mode_present", 11, 11),
    ("VADecPictureParameterBufferAV1", "mode_control_fields", "bits", "reserved", 12, 31),
    // VADecPictureParameterBufferAV1.qmatrix_fields (va_dec_av1.h:527-542), uint16_t
    ("VADecPictureParameterBufferAV1", "qmatrix_fields", "bits", "using_qmatrix", 0, 0),
    ("VADecPictureParameterBufferAV1", "qmatrix_fields", "bits", "qm_y", 1, 4),
    ("VADecPictureParameterBufferAV1", "qmatrix_fields", "bits", "qm_u", 5, 8),
    ("VADecPictureParameterBufferAV1", "qmatrix_fields", "bits", "qm_v", 9, 12),
    ("VADecPictureParameterBufferAV1", "qmatrix_fields", "bits", "reserved", 13, 15),
    // VADecPictureParameterBufferAV1.loop_restoration_fields (va_dec_av1.h:606-618), uint16_t
    ("VADecPictureParameterBufferAV1", "loop_restoration_fields", "bits", "yframe_restoration_type", 0, 1),
    ("VADecPictureParameterBufferAV1", "loop_restoration_fields", "bits", "cbframe_restoration_type", 2, 3),
    ("VADecPictureParameterBufferAV1", "loop_restoration_fields", "bits", "crframe_restoration_type", 4, 5),
    ("VADecPictureParameterBufferAV1", "loop_restoration_fields", "bits", "lr_unit_shift", 6, 7),
    ("VADecPictureParameterBufferAV1", "loop_restoration_fields", "bits", "lr_uv_shift", 8, 8),
    ("VADecPictureParameterBufferAV1", "loop_restoration_fields", "bits", "reserved", 9, 15),
    // VADecPictureParameterBufferAV1.loop_filter_info_fields (va_dec_av1.h:471-484), uint8_t
    ("VADecPictureParameterBufferAV1", "loop_filter_info_fields", "bits", "sharpness_level", 0, 2),
    ("VADecPictureParameterBufferAV1", "loop_filter_info_fields", "bits", "mode_ref_delta_enabled", 3, 3),
    ("VADecPictureParameterBufferAV1", "loop_filter_info_fields", "bits", "mode_ref_delta_update", 4, 4),
    ("VADecPictureParameterBufferAV1", "loop_filter_info_fields", "bits", "reserved", 5, 7),
    // VASegmentationStructAV1.segment_info_fields (va_dec_av1.h:81-114), uint32_t
    ("VASegmentationStructAV1", "segment_info_fields", "bits", "enabled", 0, 0),
    ("VASegmentationStructAV1", "segment_info_fields", "bits", "update_map", 1, 1),
    ("VASegmentationStructAV1", "segment_info_fields", "bits", "temporal_update", 2, 2),
    ("VASegmentationStructAV1", "segment_info_fields", "bits", "update_data", 3, 3),
    ("VASegmentationStructAV1", "segment_info_fields", "bits", "reserved", 4, 31),
    // VAFilmGrainStructAV1.film_grain_info_fields (va_dec_av1.h:155-172), uint32_t
    ("VAFilmGrainStructAV1", "film_grain_info_fields", "bits", "apply_grain", 0, 0),
    ("VAFilmGrainStructAV1", "film_grain_info_fields", "bits", "chroma_scaling_from_luma", 1, 1),
    ("VAFilmGrainStructAV1", "film_grain_info_fields", "bits", "grain_scaling_minus_8", 2, 3),
    ("VAFilmGrainStructAV1", "film_grain_info_fields", "bits", "ar_coeff_lag", 4, 5),
    ("VAFilmGrainStructAV1", "film_grain_info_fields", "bits", "ar_coeff_shift_minus_6", 6, 7),
    ("VAFilmGrainStructAV1", "film_grain_info_fields", "bits", "grain_scale_shift", 8, 9),
    ("VAFilmGrainStructAV1", "film_grain_info_fields", "bits", "overlap_flag", 10, 10),
    ("VAFilmGrainStructAV1", "film_grain_info_fields", "bits", "clip_to_restricted_range", 11, 11),
    ("VAFilmGrainStructAV1", "film_grain_info_fields", "bits", "reserved", 12, 31),
];

/// **Every AV1 bitfield union member is at the bit a C compiler puts it at, and its width is the
/// width the C compiler gives it** — measured by *running* the C, not by reading the header.
///
/// The failure this exists for is not visible to any other check in this module. A union has one
/// `offsetof`, so a member packed one bit too high agrees with the C compiler on the record's
/// `sizeof`, on every field offset and on every hole width, and the result of submitting it is a
/// **picture** rather than an error. Nor can the header be read for it: the members are declared in
/// an order that is right on the target that compiled it and carries no information about any other.
///
/// The comparison has two independent sides — a C program that sets one member to all ones and prints
/// `__builtin_ctz(value)` and `31 - __builtin_clz(value)`, and [`AV1_BITFIELDS`] above — so a member
/// that moves, or a table entry that drifts, fails here naming the union and the member.
#[test]
fn the_av1_bitfield_words_are_the_bits_the_c_compiler_measures() {
    if !std::path::Path::new(LIBVA_HEADERS).join("va_dec_av1.h").is_file() {
        eprintln!("SKIPPED: no {LIBVA_HEADERS}/va_dec_av1.h");
        return;
    }
    let Some(cc) = ["cc", "gcc", "clang"].into_iter().find_map(which) else {
        eprintln!("SKIPPED: no C compiler to measure the headers with");
        return;
    };
    // Every member is set to all ones of its width. `__builtin_ctz` then names the bit it starts at
    // and `31 - __builtin_clz` the bit it ends at. The two are computed by the compiler that lays
    // the union out, which is the whole point: nothing here reads the header's text.
    let mut source = String::from(
        "#include <stdio.h>\n#include <string.h>\n#include <stdint.h>\n#include <va/va.h>\n#include <va/va_dec_av1.h>\n\n\
         static void report(const char *u, const char *m, unsigned v) {\n\
         \x20   if (v == 0) { printf(\"%s %s NONE\\n\", u, m); return; }\n\
         \x20   printf(\"%s %s %u %u\\n\", u, m, (unsigned)__builtin_ctz(v), 31u - (unsigned)__builtin_clz(v));\n\
         }\n\n\
         int main(void) {\n",
    );
    for (record, union, sfield, member, _, _) in AV1_BITFIELDS {
        // `~0u` is all ones for any width up to 32, which is every member here; the assignment
        // truncates to the member's own width, so a member narrower than a word cannot spill into its
        // neighbour — and `report` reading the whole word back is what would show it if it did.
        source.push_str(&format!(
            "\x20 {{ {record} r; memset(&r, 0, sizeof r); r.{union}.{sfield}.{member} = ~0u; report(\"{union}\", \"{member}\", (unsigned)r.{union}.value); }}\n"
        ));
    }
    source.push_str("\x20 return 0;\n}\n");
    let dir = std::env::temp_dir().join(format!("filmcraft-va-av1-bits-{}", std::process::id()));
    if std::fs::create_dir_all(&dir).is_err() {
        eprintln!("SKIPPED: no writable temporary directory");
        return;
    }
    let path = dir.join("av1bits.c");
    let written = std::fs::write(&path, source.as_bytes()).is_ok();
    let exe = dir.join("av1bits");
    let built = if written {
        std::process::Command::new(&cc).args(["-w", "-o", exe.to_str().unwrap_or_default(), path.to_str().unwrap_or_default()]).output()
    } else {
        eprintln!("SKIPPED: cannot write {}", path.display());
        return;
    };
    let built = built.unwrap_or_else(|e| panic!("running {cc} failed: {e}"));
    assert!(built.status.success(), "{cc} rejected the AV1 bitfield probe against the headers:\n{}", String::from_utf8_lossy(&built.stderr));
    // **Run before the directory goes.** The probe *is* the executable just written into it, so
    // removing the directory first is how this test would skip itself without noticing: every
    // comparison below would then be against an empty string and the `find` would panic anyway —
    // but with a message about a missing file rather than about a wrong bit.
    let run = std::process::Command::new(exe.to_str().unwrap_or_default()).output().unwrap_or_else(|e| panic!("running the AV1 bitfield probe failed: {e}"));
    let _ = std::fs::remove_dir_all(&dir);
    let printed = String::from_utf8_lossy(&run.stdout).to_string();

    let mut seen = 0usize;
    for (record, union, _, member, low, high) in AV1_BITFIELDS {
        let want = format!("{union} {member} {low} {high}");
        let line = printed
            .lines()
            .find(|l| l.starts_with(&format!("{union} {member} ")))
            .unwrap_or_else(|| panic!("{cc} printed nothing for {record}.{union}.{member}:\n{printed}"));
        assert_eq!(line, want, "{record}: {union}.{member} — {cc} says `{line}`, our table says `{want}`");
        seen += 1;
    }
    assert_eq!(seen, AV1_BITFIELDS.len(), "every member in the table was checked");
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
    // AV1 (va_dec_av1.h). **This is the record with seven holes**, and each one is written here as
    // the `padding_*` field that covers it rather than left to the compiler, so that
    // `every_record_the_backend_submits_tiles_byte_for_byte` proves the fields cover the record and
    // `no_submitted_record_has_a_byte_the_header_does_not_account_for` can ask C for each gap's width
    // as a difference between two `offsetof`s. Six interior gaps plus a trailing pad:
    //
    //   17-23  after anchor_frames_num      7   the VASurfaceID * is 8-aligned
    //   81-83  after order_hint              3   VASegmentationStructAV1 is 4-aligned
    //   674-675 after context_update_tile_id 2   pic_info_fields' value is a uint32_t
    //   703    after v_ac_delta_q            1   qmatrix_fields' value is a uint16_t
    //   706-707 after qmatrix_fields         2   mode_control_fields' value is a uint32_t
    //   1156-1159 after va_reserved          4   the struct's alignment is 8, for that pointer
    //
    // And one place where a hole was *claimed* and is not there: `VASegmentationStructAV1` has none.
    // The plan for this milestone records a 4-byte hole at 148-152 before its `va_reserved`, and
    // `feature_mask` ends at 140 with `va_reserved` a `uint32_t[4]` at 140 — measured, gap zero. It
    // is listed with no `padding_*` row so that "no hole" is asserted as a gap of zero rather than
    // left unchecked, which is what would happen if the record were simply absent from this table.
    "VADecPictureParameterBufferAV1": size_of::<VaPictureParameterBufferAv1>() => [
        ("profile", profile, size_of::<u8>()),
        ("order_hint_bits_minus_1", order_hint_bits_minus_1, size_of::<u8>()),
        ("bit_depth_idx", bit_depth_idx, size_of::<u8>()),
        ("matrix_coefficients", matrix_coefficients, size_of::<u8>()),
        ("seq_info_fields", seq_info_fields, size_of::<u32>()),
        ("current_frame", current_frame, size_of::<VASurfaceID>()),
        ("current_display_picture", current_display_picture, size_of::<VASurfaceID>()),
        ("anchor_frames_num", anchor_frames_num, size_of::<u8>()),
        ("-", padding_after_anchor_frames_num, size_of::<[u8; 7]>()),
        ("anchor_frames_list", anchor_frames_list, size_of::<u64>()),
        ("frame_width_minus1", frame_width_minus1, size_of::<u16>()),
        ("frame_height_minus1", frame_height_minus1, size_of::<u16>()),
        ("output_frame_width_in_tiles_minus_1", output_frame_width_in_tiles_minus_1, size_of::<u16>()),
        ("output_frame_height_in_tiles_minus_1", output_frame_height_in_tiles_minus_1, size_of::<u16>()),
        ("ref_frame_map", ref_frame_map, size_of::<[VASurfaceID; 8]>()),
        ("ref_frame_idx", ref_frame_idx, size_of::<[u8; 7]>()),
        ("primary_ref_frame", primary_ref_frame, size_of::<u8>()),
        ("order_hint", order_hint, size_of::<u8>()),
        ("-", padding_after_order_hint, size_of::<[u8; 3]>()),
        ("seg_info", seg_info, size_of::<VaSegmentationStructAv1>()),
        ("film_grain_info", film_grain_info, size_of::<VaFilmGrainStructAv1>()),
        ("tile_cols", tile_cols, size_of::<u8>()),
        ("tile_rows", tile_rows, size_of::<u8>()),
        ("width_in_sbs_minus_1", width_in_sbs_minus_1, size_of::<[u16; 63]>()),
        ("height_in_sbs_minus_1", height_in_sbs_minus_1, size_of::<[u16; 63]>()),
        ("tile_count_minus_1", tile_count_minus_1, size_of::<u16>()),
        ("context_update_tile_id", context_update_tile_id, size_of::<u16>()),
        ("-", padding_after_context_update_tile_id, size_of::<[u8; 2]>()),
        ("pic_info_fields", pic_info_fields, size_of::<u32>()),
        ("superres_scale_denominator", superres_scale_denominator, size_of::<u8>()),
        ("interp_filter", interp_filter, size_of::<u8>()),
        ("filter_level", filter_level, size_of::<[u8; 2]>()),
        ("filter_level_u", filter_level_u, size_of::<u8>()),
        ("filter_level_v", filter_level_v, size_of::<u8>()),
        ("loop_filter_info_fields", loop_filter_info_fields, size_of::<u8>()),
        ("ref_deltas", ref_deltas, size_of::<[i8; 8]>()),
        ("mode_deltas", mode_deltas, size_of::<[i8; 2]>()),
        ("base_qindex", base_qindex, size_of::<u8>()),
        ("y_dc_delta_q", y_dc_delta_q, size_of::<i8>()),
        ("u_dc_delta_q", u_dc_delta_q, size_of::<i8>()),
        ("u_ac_delta_q", u_ac_delta_q, size_of::<i8>()),
        ("v_dc_delta_q", v_dc_delta_q, size_of::<i8>()),
        ("v_ac_delta_q", v_ac_delta_q, size_of::<i8>()),
        ("-", padding_after_v_ac_delta_q, size_of::<[u8; 1]>()),
        ("qmatrix_fields", qmatrix_fields, size_of::<u16>()),
        ("-", padding_after_qmatrix_fields, size_of::<[u8; 2]>()),
        ("mode_control_fields", mode_control_fields, size_of::<u32>()),
        ("cdef_damping_minus_3", cdef_damping_minus_3, size_of::<u8>()),
        ("cdef_bits", cdef_bits, size_of::<u8>()),
        ("cdef_y_strengths", cdef_y_strengths, size_of::<[u8; 8]>()),
        ("cdef_uv_strengths", cdef_uv_strengths, size_of::<[u8; 8]>()),
        ("loop_restoration_fields", loop_restoration_fields, size_of::<u16>()),
        ("wm", wm, size_of::<[VaWarpedMotionParamsAv1; 7]>()),
        ("va_reserved", va_reserved, VA_PADDING_MEDIUM * size_of::<u32>()),
        ("-", padding_after_va_reserved, size_of::<[u8; 4]>()),
    ],
    "VASegmentationStructAV1": size_of::<VaSegmentationStructAv1>() => [
        ("segment_info_fields", segment_info_fields, size_of::<u32>()),
        ("feature_data", feature_data, size_of::<[[i16; 8]; 8]>()),
        ("feature_mask", feature_mask, size_of::<[u8; 8]>()),
        ("va_reserved", va_reserved, VA_PADDING_LOW * size_of::<u32>()),
    ],
    "VAFilmGrainStructAV1": size_of::<VaFilmGrainStructAv1>() => [
        ("film_grain_info_fields", film_grain_info_fields, size_of::<u32>()),
        ("grain_seed", grain_seed, size_of::<u16>()),
        ("num_y_points", num_y_points, size_of::<u8>()),
        ("point_y_value", point_y_value, size_of::<[u8; 14]>()),
        ("point_y_scaling", point_y_scaling, size_of::<[u8; 14]>()),
        ("num_cb_points", num_cb_points, size_of::<u8>()),
        ("point_cb_value", point_cb_value, size_of::<[u8; 10]>()),
        ("point_cb_scaling", point_cb_scaling, size_of::<[u8; 10]>()),
        ("num_cr_points", num_cr_points, size_of::<u8>()),
        ("point_cr_value", point_cr_value, size_of::<[u8; 10]>()),
        ("point_cr_scaling", point_cr_scaling, size_of::<[u8; 10]>()),
        ("ar_coeffs_y", ar_coeffs_y, size_of::<[i8; 24]>()),
        ("ar_coeffs_cb", ar_coeffs_cb, size_of::<[i8; 25]>()),
        ("ar_coeffs_cr", ar_coeffs_cr, size_of::<[i8; 25]>()),
        ("cb_mult", cb_mult, size_of::<u8>()),
        ("cb_luma_mult", cb_luma_mult, size_of::<u8>()),
        ("-", padding_before_cb_offset, size_of::<[u8; 1]>()),
        ("cb_offset", cb_offset, size_of::<u16>()),
        ("cr_mult", cr_mult, size_of::<u8>()),
        ("cr_luma_mult", cr_luma_mult, size_of::<u8>()),
        ("cr_offset", cr_offset, size_of::<u16>()),
        ("va_reserved", va_reserved, VA_PADDING_LOW * size_of::<u32>()),
    ],
    "VAWarpedMotionParamsAV1": size_of::<VaWarpedMotionParamsAv1>() => [
        ("wmtype", wmtype, size_of::<u32>()),
        ("wmmat", wmmat, size_of::<[i32; 8]>()),
        ("invalid", invalid, size_of::<u8>()),
        ("-", padding_before_va_reserved, size_of::<[u8; 3]>()),
        ("va_reserved", va_reserved, VA_PADDING_LOW * size_of::<u32>()),
    ],
    "VASliceParameterBufferAV1": size_of::<VaSliceParameterBufferAv1>() => [
        ("slice_data_size", slice_data_size, size_of::<u32>()),
        ("slice_data_offset", slice_data_offset, size_of::<u32>()),
        ("slice_data_flag", slice_data_flag, size_of::<u32>()),
        ("tile_row", tile_row, size_of::<u16>()),
        ("tile_column", tile_column, size_of::<u16>()),
        ("tg_start", tg_start, size_of::<u16>()),
        ("tg_end", tg_end, size_of::<u16>()),
        ("anchor_frame_idx", anchor_frame_idx, size_of::<u8>()),
        ("-", padding_before_tile_idx_in_tile_list, size_of::<[u8; 1]>()),
        ("tile_idx_in_tile_list", tile_idx_in_tile_list, size_of::<u16>()),
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
    // The rate-control record, and the row that is here for a reason beyond the usual: **it has no
    // hole at all**, so it is the one submitted record where the tiling test proves nothing on its
    // own and the C gap check has to carry it. Every member is a `uint32_t` or a union of them, from
    // offset 0 to 44, and `va_reserved[VA_PADDING_LOW]` takes it to 60 — twelve members in a row with
    // no padding between any two of them and none after the last.
    //
    // **There is no `padding_*` row here, and that is the assertion.** A record added to this list
    // is supposed to name every byte it contains; a member with no C name can only appear as one of
    // those, and one appearing where the header has none would mean this declaration grew a hole the
    // header does not have — the `VASegmentParameterVP9` failure, which agreed with C on every size
    // and offset and handed the driver two indeterminate bytes per record. The gap check asks the
    // compiler for `offsetof(b) - offsetof(a) - sizeof(a)` between each pair of members and asserts
    // it equals **zero** here rather than leaving a hole-free record unchecked, so "no hole" is a
    // measurement rather than an absence of one.
    "VAEncMiscParameterRateControl": size_of::<VaEncMiscParameterRateControl>() => [
        ("bits_per_second", bits_per_second, size_of::<u32>()),
        ("target_percentage", target_percentage, size_of::<u32>()),
        ("window_size", window_size, size_of::<u32>()),
        ("initial_qp", initial_qp, size_of::<u32>()),
        ("min_qp", min_qp, size_of::<u32>()),
        ("basic_unit_size", basic_unit_size, size_of::<u32>()),
        // The union's `value` member. C sizes it as the union, which is `sizeof(uint32_t)` — 4 — and
        // that is what the `sizeof` static assertion below asks the compiler.
        ("rc_flags", rc_flags, size_of::<u32>()),
        ("ICQ_quality_factor", icq_quality_factor, size_of::<u32>()),
        ("max_qp", max_qp, size_of::<u32>()),
        ("quality_factor", quality_factor, size_of::<u32>()),
        ("target_frame_size", target_frame_size, size_of::<u32>()),
        ("va_reserved", va_reserved, VA_PADDING_LOW * size_of::<u32>()),
    ],
    // **The one record here that is not a libva record**, and the one whose verification read as
    // rigorous and was wrong: `struct drm_prime_handle` is handed to the kernel by
    // `DRM_IOCTL_PRIME_HANDLE_TO_FD`, it was declared eight bytes short, and the request number that
    // follows from its size was therefore a request number no kernel defines — which the kernel
    // dispatched anyway, writing its answer past the end of the struct. One dma-buf descriptor leaked
    // per encoded picture, and the descriptor libva owns was closed twice.
    //
    // Until this row existed it was the only record in the backend no oracle could reach: not in
    // this list, and the C sources above did not include `<drm/drm.h>` at all. Both `sizeof` and every
    // `offsetof` are now asked of the installed header, and `every_declaration_matches_what_a_c_compiler_measures`
    // additionally asserts the *request number* against `DRM_IOCTL_PRIME_HANDLE_TO_FD` itself rather
    // than against a literal somebody typed from the same header twice.
    // Named with the C spelling `struct drm_prime_handle` rather than the bare tag, because that is
    // what every generated `sizeof` / `offsetof` needs — a C struct has no typedef. The other rows
    // are `typedef struct … Name;` in libva and drm, which is why they need no `struct`.
    "struct drm_prime_handle": size_of::<drm::DrmPrimeHandle>() => [
        ("handle", handle, size_of::<u32>()),
        ("flags", flags, size_of::<u32>()),
        ("fd", fd, size_of::<c_int>()),
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
    if !std::path::Path::new(LIBVA_HEADERS).join("va.h").is_file() || !drm_headers_present() {
        eprintln!("SKIPPED: no {LIBVA_HEADERS}/va.h or no {DRM_HEADERS}/drm.h");
        return;
    }
    let Some(cc) = ["cc", "gcc", "clang"].into_iter().find_map(which) else {
        eprintln!("SKIPPED: no C compiler to measure the headers with");
        return;
    };
    // `<drm/drm.h>` is in the list because `struct drm_prime_handle` is a record this backend hands
    // to the kernel — see the `drm_prime_handle` row of the list above. It is installed on every
    // machine that can run this backend at all (libva's own DRM backend includes it), so its absence
    // is a machine with no DRM headers rather than a machine where the check could not be asked.
    let mut source = String::from(
        "#include <stdint.h>\n#include <va/va.h>\n#include <va/va_drm.h>\n#include <va/va_dec_av1.h>\n#include <va/va_dec_hevc.h>\n#include <va/va_dec_vp9.h>\n#include <va/va_enc_h264.h>\n#include <drm/drm.h>\n",
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
        std::process::Command::new(&cc).args(["-fsyntax-only", path.to_str().unwrap_or_default()]).output()
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
    if !std::path::Path::new(LIBVA_HEADERS).join("va.h").is_file() || !drm_headers_present() {
        eprintln!("SKIPPED: no {LIBVA_HEADERS}/va.h or no {DRM_HEADERS}/drm.h");
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
        ("VADecPictureParameterBufferAV1", size_of::<VaPictureParameterBufferAv1>()),
        ("VASegmentationStructAV1", size_of::<VaSegmentationStructAv1>()),
        ("VAFilmGrainStructAV1", size_of::<VaFilmGrainStructAv1>()),
        ("VAWarpedMotionParamsAV1", size_of::<VaWarpedMotionParamsAv1>()),
        ("VASliceParameterBufferAV1", size_of::<VaSliceParameterBufferAv1>()),
        ("VACodedBufferSegment", size_of::<VaCodedBufferSegment>()),
        ("VABufferInfo", size_of::<VaBufferInfo>()),
        ("VAEncSequenceParameterBufferH264", size_of::<VaEncSequenceParameterBufferH264>()),
        ("VAEncPictureParameterBufferH264", size_of::<VaEncPictureParameterBufferH264>()),
        ("VAEncSliceParameterBufferH264", size_of::<VaEncSliceParameterBufferH264>()),
        ("VAEncMiscParameterFrameRate", size_of::<VaEncMiscParameterFrameRate>()),
        ("VAEncMiscParameterQuantization", size_of::<VaEncMiscParameterQuantization>()),
        ("VAEncMiscParameterRateControl", size_of::<VaEncMiscParameterRateControl>()),
        ("struct drm_prime_handle", size_of::<drm::DrmPrimeHandle>()),
    ];
    let mut source = String::from(
        "#include <va/va.h>\n#include <va/va_drm.h>\n#include <va/va_dec_av1.h>\n#include <va/va_dec_hevc.h>\n#include <va/va_dec_vp9.h>\n#include <va/va_enc_h264.h>\n#include <drm/drm.h>\n",
    );
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
    source.push_str(&format!(
        "/* The request number, asked of the header's own macro rather than of a literal\n   somebody typed from that header: this is the value the drm_prime_handle struct\n   produced, and it is the value the kernel dispatches on. */\n_Static_assert({name} == {value:#x}, \"{name}\");\n",
        name = drm::PRIME_HANDLE_TO_FD_ABI_NAME,
        value = drm::PRIME_HANDLE_TO_FD,
    ));
    let dir = std::env::temp_dir().join(format!("filmcraft-va-abi-{}", std::process::id()));
    if std::fs::create_dir_all(&dir).is_err() {
        eprintln!("SKIPPED: no writable temporary directory");
        return;
    }
    let path = dir.join("abi.c");
    let written = std::fs::write(&path, &source).is_ok();
    let out = if written {
        std::process::Command::new(&cc).args(["-fsyntax-only", path.to_str().unwrap_or_default()]).output()
    } else {
        eprintln!("SKIPPED: cannot write {}", path.display());
        return;
    };
    let _ = std::fs::remove_dir_all(&dir);
    let out = out.unwrap_or_else(|e| panic!("running {cc} failed: {e}"));
    assert!(out.status.success(), "{cc} rejected our declarations against the headers:\n{}", String::from_utf8_lossy(&out.stderr));
}

/// Whether the DRM headers are installed, which they are on every machine that can run this backend
/// at all — libva's DRM backend includes them, so "no `drm.h`" means a machine with no DRM
/// development files rather than a stack this backend could use. The guard skips there, with a note,
/// so that adding the record cannot turn a machine red for a reason that has nothing to do with it.
fn drm_headers_present() -> bool {
    std::path::Path::new(DRM_HEADERS).join("drm.h").is_file()
}

/// `name` on the `PATH`, or `None`.
fn which(name: &str) -> Option<String> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).find_map(|dir| {
        let candidate = dir.join(name);
        candidate.is_file().then(|| candidate.to_string_lossy().into_owned())
    })
}

// ---------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------
// Width, signedness and shape — the properties of a declaration that change no byte of a record.
//
// Everything above asks the C compiler for a **number**: a `sizeof`, an `offsetof`, the width of a
// hole, a bit position. All of those are blind to the difference between a `u32` and an `i32`,
// because the two are the same four bytes at the same offset in a struct of the same size. They
// compile, they link, they pass every check in this file — and then the driver writes a value above
// `INT32_MAX` and the crate reads a negative number, or the reverse, with nothing to notice and
// nothing but a wrong picture to show for it.
//
// This section is **every record in [`SUBMITTED_RECORDS`]**, not a sample of two, and it is
// additive: it touches neither that list nor the hole checks nor
// [`every_record_the_backend_submits_tiles_byte_for_byte`]. The sample it started as covered the
// shapes a signedness mistake is possible in; the rest turned out to need three more shapes before
// it could be checked at all, and each of those is a shape where a *wrong* answer was not
// detectable by the two numbers above:
//
//   * **a flag union** — the C declarations carry bitfields (`seq_fields`, `pic_fields`,
//     `slice_parsing_fields`, `LongSliceFlags`, the AV1 `*_fields` …), which Rust has no syntax
//     for, so `ffi.rs` declares each as the union's own `value` word. `sizeof` is 4 for the union
//     and for the word, so a Rust `u32` for a union whose `value` is a `uint16_t` is invisible to
//     every other check in this file.
//   * **an enumerated type** — C11 6.7.2.2p4 leaves the underlying type to the implementation, so
//     there is a *shape* to check and no *signedness* to compare. `_Generic`'s `default:` bucket
//     used to lump these in with unions and pointers, which is why more rows alone would not do it.
//   * **a pointer** — `VACodedBufferSegment::buf` and `::next` are `void *` and nothing else, and a
//     `u64` for either is the same eight bytes as a `uintptr_t`.
//
// The two sides are independent, which is what makes the comparison worth anything:
//
// * **C** — `_Generic` over the member, folded with `sizeof` into `sign * 256 + width`, plus
//   `__builtin_classify_type(__typeof__(member))` for the shape. This reads the header. Editing
//   this file cannot change it.
// * **Rust** — [`CKindOf`], read off the type spelled in [`TYPED_RECORDS`]. That spelling is a
//   transcription and is **not** trusted on its own: the tiling test already pins each field's width
//   against `offset_of!` on the real struct, so a wrong width cannot survive it, and this section
//   re-asserts the two tables' widths agree for every field. What nothing else pinned is the
//   **signedness** and the **shape**, and those are what the C compiler is being asked about.
//
// ### The one thing that is deliberately *not* checked
//
// `VADecPictureParameterBufferAV1::anchor_frames_list` is a `VASurfaceID *` in the header
// (`va_dec_av1.h:330`) and a `u64` in `ffi.rs` — a real shape disagreement, settled deliberately on
// the other side of the boundary and pinned by `ffi.rs`'s own
// `the_pointer_shaped_field_is_an_address_and_not_a_pointer`. It is the one member of the 385 here
// that this check does not cover, and it is named in [`NOT_CHECKED`] with the reason rather than
// passed over.
// ---------------------------------------------------------------------------------------------

/// A C compiler's [`__builtin_classify_type`] answers, as spelled in [`CK_PROLOGUE`].
///
/// **The `__typeof__` is not decoration.** Given an *expression*, `__builtin_classify_type` reports
/// an enumerated type as [`CK_INTEGER`] — the expression's type has already been converted to the
/// underlying integer type by the conversions a builtin applies to an argument. Given the *type*,
/// it reports [`CK_ENUMERAL`]. Both were measured on this target (GCC 16.2, clang 23.1), and the
/// difference is the whole reason an enum can be told from an integer at all.
const CK_INTEGER: i32 = 1;
const CK_ENUMERAL: i32 = 3;
const CK_POINTER: i32 = 5;
const CK_RECORD: i32 = 12;
const CK_UNION: i32 = 13;

/// Which way round a C integer reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Sign {
    Signed,
    Unsigned,
}

impl Sign {
    /// The number the C side is compared with for an integer of this signedness and `width`.
    ///
    /// In one place, so "the same signedness" is one number rather than a literal repeated in a
    /// generated string, and so flipping one bit of it is visibly a different value.
    const fn code(self, width: usize) -> i32 {
        match self {
            Sign::Unsigned => width as i32,
            Sign::Signed => 256 + width as i32,
        }
    }

    /// How a message names this signedness.
    const fn spelling(self) -> &'static str {
        match self {
            Sign::Unsigned => "unsigned",
            Sign::Signed => "signed",
        }
    }
}

/// What the C compiler must say about the header's member, given what our declaration says.
///
/// One enum rather than a bare signedness because the shapes are part of the answer: a declaration
/// that stores a flag union's `value` word is *not* the same declaration as one that stores a
/// plain `uint32_t`, and before [`CK_CLASS`] existed nothing here could tell them apart.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CKind {
    /// The header's member is an integer of this signedness and this width.
    Integer { sign: Sign, width: usize },
    /// The header's member is a nested record. Its width is [`SubmittedField::width`]'s business;
    /// there is no signedness to disagree about.
    Record,
    /// The header's member is a **union**, and the word this declaration stores inside it is this
    /// integer. Both halves are asserted: the class on the member, the width and signedness on the
    /// named word — which is why [`CKindOf::UNION_WORD`] carries a name.
    UnionValue { sign: Sign, width: usize },
    /// The header's member is an **enumerated type**. C11 6.7.2.2p4 leaves the underlying type to
    /// the implementation, so there is no signedness to compare and none is claimed; what is
    /// asserted is the shape, and the width by the cross-check against [`SUBMITTED_RECORDS`].
    Enum,
    /// The header's member is a pointer.
    Pointer,
}

impl CKind {
    /// The `__builtin_classify_type` class the header's member has to answer with.
    const fn expected_class(self) -> i32 {
        match self {
            CKind::Integer { .. } => CK_INTEGER,
            CKind::Record => CK_RECORD,
            CKind::UnionValue { .. } => CK_UNION,
            CKind::Enum => CK_ENUMERAL,
            CKind::Pointer => CK_POINTER,
        }
    }

    /// The number the header's member has to agree with, for the kinds that have one.
    const fn code(self) -> Option<i32> {
        match self {
            CKind::Integer { sign, width } | CKind::UnionValue { sign, width } => Some(sign.code(width)),
            CKind::Record | CKind::Enum | CKind::Pointer => None,
        }
    }

    /// What our declaration says, in words, for the message a failure prints.
    fn describe(self) -> String {
        match self {
            CKind::Integer { sign, width } => format!("a {} {width}-byte integer", sign.spelling()),
            CKind::UnionValue { sign, width } => format!("the `value` word of a union, {} {width} bytes", sign.spelling()),
            CKind::Record => "a nested record".to_string(),
            CKind::Enum => "an enumerated type".to_string(),
            CKind::Pointer => "a pointer".to_string(),
        }
    }
}

/// A Rust field type, read for what C will say about the C type it stands for.
///
/// The C **spelling** deliberately does not matter. `VAStatus`, `VAGenericID`, `int32_t` and
/// `unsigned int` are all four bytes; only the last of them is a different answer to the question
/// this asks, and `_Generic` matches a typedef against its underlying type, so none of the others
/// can be confused with it either.
trait CKindOf {
    /// The shape, and where there is one the width and signedness, C has to agree with.
    const KIND: CKind;
    /// How many levels of array the type is nested in, so the C side can peel the same way.
    ///
    /// Peeling is needed because `_Generic` on an array **decays it to a pointer** (C11 6.3.2.1p3),
    /// which would silently answer "pointer" for every array member here — including
    /// `int8_t component_order[4]`, where the answer that matters is `int8_t`. A two-dimensional
    /// array needs it twice: `int16_t chroma_weight_l0[32][2]` peeled once is still an array, and
    /// `__builtin_classify_type` decays *that* to `int16_t *` as well, so a second `[0]` is the
    /// only thing that reaches the `int16_t` we actually declared.
    const DEPTH: usize;
    /// The name the header gives the union word this declaration stores, or `""` for a type that is
    /// not a union. Every flag union in these records spells it `value` except
    /// `VAGenericValue::value`, whose union holds `i`, `f`, `p` and `fn` (`va.h:1666-1675`).
    const UNION_WORD: &'static str = "";
}

macro_rules! c_kind {
    ($($ty:ty => $kind:expr),* $(,)?) => {$(
        impl CKindOf for $ty {
            const KIND: CKind = $kind;
            const DEPTH: usize = 0;
        }
    )*};
}
c_kind!(
    u8 => CKind::Integer { sign: Sign::Unsigned, width: 1 },
    i8 => CKind::Integer { sign: Sign::Signed, width: 1 },
    u16 => CKind::Integer { sign: Sign::Unsigned, width: 2 },
    i16 => CKind::Integer { sign: Sign::Signed, width: 2 },
    // `c_int`, `VAStatus` and `VAProfile`/`VAEntrypoint` all *are* `i32`, so one impl covers them.
    // That collapse is also why the Rust type alone cannot tell a member the header declares as an
    // enum from one it declares as an `int`: both spell `i32` here. [`HeaderEnum`] and [`FlagWord`]
    // are how those two shapes are spelled apart.
    u32 => CKind::Integer { sign: Sign::Unsigned, width: 4 },
    i32 => CKind::Integer { sign: Sign::Signed, width: 4 },
    // `VABufferID`, `VAImageID`, `VASurfaceID`, `c_uint` and `VAConfigAttrib::value` are all `u32`,
    // for the same reason and with the same consequence.
    u64 => CKind::Integer { sign: Sign::Unsigned, width: 8 },
    i64 => CKind::Integer { sign: Sign::Signed, width: 8 },
    // `size_t` and `uintptr_t`, which is what `VABufferInfo::handle` and `::mem_size` are.
    usize => CKind::Integer { sign: Sign::Unsigned, width: 8 },
    // `void *`, which is what `VACodedBufferSegment::buf` and `::next` are.
    *mut c_void => CKind::Pointer,
);
/// An array of anything: the kind is the element's and the depth grows, because that is what the C
/// side reaches once it has indexed its way in one level at a time.
impl<T: CKindOf, const N: usize> CKindOf for [T; N] {
    const KIND: CKind = T::KIND;
    const DEPTH: usize = T::DEPTH + 1;
}

/// The nested records these fields hold. A record added to [`TYPED_RECORDS`] with a field type that
/// is neither an integer, nor one of these, nor [`FlagWord`], nor [`HeaderEnum`] **does not compile**,
/// and the error names the field — so the list cannot fall behind the table without the build noticing.
macro_rules! c_record {
    ($($ty:ty),* $(,)?) => {$(
        impl CKindOf for $ty {
            const KIND: CKind = CKind::Record;
            const DEPTH: usize = 0;
        }
    )*};
}
c_record!(
    VaImageFormat,
    VaGenericValue,
    VaPictureH264,
    VaPictureHevc,
    VaSegmentationStructAv1,
    VaFilmGrainStructAv1,
    VaWarpedMotionParamsAv1,
    VaPictureParameterBufferVp9,
    VaSegmentParameterVp9,
    VaSliceParameterBufferVp9
);

/// A Rust type `ffi.rs` writes **in place of a C flag union**, saying which word of that union it
/// holds and how wide that word is.
///
/// This is a trait on the *declared* type rather than a marker type with two type parameters on
/// purpose. The two things a flag-union field needs are independent: the **width** the declaration
/// occupies, which is `size_of` of what `ffi.rs` actually wrote, and the **word** inside the
/// header's union that we read, whose signedness and width are what `CKIND` has to agree with. Those
/// are the same thing for twenty-one of the twenty-two unions — `ffi.rs` declares the `value` word
/// itself, so the field is a `uint32_t` and the word is a `uint32_t` — and **not** the same thing for
/// `VAGenericValue::value`, where the declaration is the whole eight-byte union (`ffi.rs`'s
/// `VaGenericValueValue`, which holds a `void *`) and the word read out of it is a four-byte `int32_t`
/// named `i` (`va.h:1666-1675`). Two type parameters on a marker could not say both, because the
/// marker itself has to be as wide as the declaration for the width cross-check to mean anything; two
/// associated consts can, and it keeps the twenty-one ordinary rows in [`TYPED_RECORDS`] reading as
/// the plain integers they are.
trait FlagWordOf {
    /// The word this declaration stores, as the header has to describe it.
    const WORD: CKind;
    /// The name the header gives that word inside the union.
    const NAME: &'static str;
}

macro_rules! flag_word {
    ($($ty:ty => $word:expr, $name:literal),* $(,)?) => {$(
        impl FlagWordOf for $ty {
            const WORD: CKind = $word;
            const NAME: &'static str = $name;
        }
    )*};
}
flag_word!(
    u32 => CKind::UnionValue { sign: Sign::Unsigned, width: 4 }, "value",
    u16 => CKind::UnionValue { sign: Sign::Unsigned, width: 2 }, "value",
    u8 => CKind::UnionValue { sign: Sign::Unsigned, width: 1 }, "value",
    // The one union in these records whose integer word is not called `value`, and the one whose
    // declared field is wider than the word: `ffi.rs` declares the whole union.
    VaGenericValueValue => CKind::UnionValue { sign: Sign::Signed, width: 4 }, "i",
);

/// A field spelled `FlagWord<T>` in [`TYPED_RECORDS`]: `ffi.rs` stores a C flag union's word, because
/// Rust has no bitfield syntax, so the packed word is stored on its own and `picture.rs` sets the bits.
///
/// `#[repr(transparent)]` is what keeps this honest rather than convenient: the type is exactly as
/// wide as `T`, so the width cross-check against [`SUBMITTED_RECORDS`] reads the width of the type
/// `ffi.rs` actually wrote down and `size_of::<FlagWord<u32>>() == 4` is a fact about `u32` rather
/// than about this marker. Nothing is ever read out of it — the records are written through `ffi.rs`'s
/// own field names — so the field is dead by construction.
#[repr(transparent)]
#[allow(dead_code, reason = "never read; it is here only so that repr(transparent) makes its width T's")]
struct FlagWord<T: FlagWordOf>(T);

impl<T: FlagWordOf> CKindOf for FlagWord<T> {
    const KIND: CKind = T::WORD;
    const DEPTH: usize = 0;
    const UNION_WORD: &'static str = T::NAME;
}

/// A field the header declares as an **enumerated type** and this declaration spells as a plain
/// integer. Five members are like that — see [`TYPED_RECORDS`] for the row each one is.
///
/// `#[repr(transparent)]` for the same reason as [`FlagWord`]: the width cross-check has to read the
/// width of the type `ffi.rs` wrote down, and [`CKind::Enum`] has to say "C will not tell us the
/// signedness of this one" instead of pretending to know it.
#[repr(transparent)]
#[allow(dead_code, reason = "never read; it is here for its size, which is what repr(transparent) makes equal to T's")]
struct HeaderEnum<T: CKindOf>(T);

impl<T: CKindOf> CKindOf for HeaderEnum<T> {
    const KIND: CKind = CKind::Enum;
    const DEPTH: usize = 0;
}

/// One field of one record, with the type our declaration gives it.
struct TypedField {
    /// The member's name in `/usr/include/va/`.
    c_name: &'static str,
    /// `size_of` of the field's type, cross-checked against [`SUBMITTED_RECORDS`] below.
    width: usize,
    /// What our field's type says its shape, width and signedness are.
    kind: CKind,
    /// How far into the field's type the thing C has to measure is.
    depth: usize,
    /// The union word [`CKind::UnionValue`] measures, or `""` when there is none.
    union_word: &'static str,
}

/// A record and its fields, in declaration order.
struct TypedRecord {
    /// The struct's name in `/usr/include/va/`.
    c_name: &'static str,
    fields: &'static [TypedField],
}

macro_rules! typed_records {
    ($($c_record:literal => [ $( ($c_field:literal, $field:ident : $fty:ty) ),* $(,)? ]),* $(,)?) => {
        &[$(TypedRecord {
            c_name: $c_record,
            fields: &[
                $(TypedField {
                    c_name: $c_field,
                    width: size_of::<$fty>(),
                    kind: <$fty as CKindOf>::KIND,
                    depth: <$fty as CKindOf>::DEPTH,
                    union_word: <$fty as CKindOf>::UNION_WORD,
                }),*
            ],
        }),*]
    };
}

/// Every record [`SUBMITTED_RECORDS`] holds, every member of each, with the Rust type the
/// declaration gives it.
///
/// Nothing here restates an offset or a record's size — those belong to [`SUBMITTED_RECORDS`], and
/// this table is cross-checked against it member by member — because the only thing it adds is the
/// **type**. The rows are written out rather than generated because a row added here is the only
/// way a new member becomes visible to this check, and a table generated from the header could not
/// disagree with it.
///
/// `rust_name` is no longer spelled: [`TYPED_RECORDS`] is keyed on the C member name and
/// [`SubmittedField::c_name`], which is what the cross-check below already compared, so the second
/// transcription of each field's Rust name is one fewer place for the two tables to drift.
static TYPED_RECORDS: &[TypedRecord] = typed_records![
    "VAImageFormat" => [
        ("fourcc", fourcc: u32),
        ("byte_order", byte_order: u32),
        ("bits_per_pixel", bits_per_pixel: u32),
        ("depth", depth: u32),
        ("red_mask", red_mask: u32),
        ("green_mask", green_mask: u32),
        ("blue_mask", blue_mask: u32),
        ("alpha_mask", alpha_mask: u32),
        ("va_reserved", va_reserved: [u32; VA_PADDING_LOW]),
    ],
    "VAImage" => [
        ("image_id", image_id: VAImageID),
        ("format", format: VaImageFormat),
        ("buf", buf: VABufferID),
        ("width", width: u16),
        ("height", height: u16),
        ("data_size", data_size: u32),
        ("num_planes", num_planes: u32),
        ("pitches", pitches: [u32; 3]),
        ("offsets", offsets: [u32; 3]),
        ("num_palette_entries", num_palette_entries: i32),
        ("entry_bytes", entry_bytes: i32),
        ("component_order", component_order: [i8; 4]),
        ("va_reserved", va_reserved: [u32; VA_PADDING_LOW]),
    ],
    // The three records a driver reads back, and the one with the most shapes in it: an enumerated
    // type, a nested record and a union, all within sixteen bytes.
    "VAConfigAttrib" => [
        ("type", type_: HeaderEnum<VAStatus>),
        ("value", value: c_uint),
    ],
    "VASurfaceAttrib" => [
        ("type", type_: HeaderEnum<VAStatus>),
        ("flags", flags: c_uint),
        ("value", value: VaGenericValue),
    ],
    "VAGenericValue" => [
        ("type", type_: HeaderEnum<VAStatus>),
        ("value", value: FlagWord<VaGenericValueValue>),
    ],
    "VAPictureH264" => [
        ("picture_id", picture_id: VABufferID),
        ("frame_idx", frame_idx: u32),
        ("flags", flags: u32),
        ("TopFieldOrderCnt", top_field_order_cnt: i32),
        ("BottomFieldOrderCnt", bottom_field_order_cnt: i32),
        ("va_reserved", va_reserved: [u32; VA_PADDING_LOW]),
    ],
    "VAPictureParameterBufferH264" => [
        ("CurrPic", curr_pic: VaPictureH264),
        ("ReferenceFrames", reference_frames: [VaPictureH264; 16]),
        ("picture_width_in_mbs_minus1", picture_width_in_mbs_minus1: u16),
        ("picture_height_in_mbs_minus1", picture_height_in_mbs_minus1: u16),
        ("bit_depth_luma_minus8", bit_depth_luma_minus8: u8),
        ("bit_depth_chroma_minus8", bit_depth_chroma_minus8: u8),
        ("num_ref_frames", num_ref_frames: u8),
        ("seq_fields", seq_fields: FlagWord<u32>),
        ("num_slice_groups_minus1", num_slice_groups_minus1: u8),
        ("slice_group_map_type", slice_group_map_type: u8),
        ("slice_group_change_rate_minus1", slice_group_change_rate_minus1: u16),
        ("pic_init_qp_minus26", pic_init_qp_minus26: i8),
        ("pic_init_qs_minus26", pic_init_qs_minus26: i8),
        ("chroma_qp_index_offset", chroma_qp_index_offset: i8),
        ("second_chroma_qp_index_offset", second_chroma_qp_index_offset: i8),
        ("pic_fields", pic_fields: FlagWord<u32>),
        ("frame_num", frame_num: u16),
        ("va_reserved", va_reserved: [u32; VA_PADDING_MEDIUM]),
    ],
    // Two-dimensional arrays at their widest: `int16_t chroma_weight_l0[32][2]` needs peeling twice
    // before `_Generic` sees the `int16_t` it has to judge, and `[[i16; 2]; 32]` is how that reads
    // on our side.
    "VASliceParameterBufferH264" => [
        ("slice_data_size", slice_data_size: u32),
        ("slice_data_offset", slice_data_offset: u32),
        ("slice_data_flag", slice_data_flag: u32),
        ("slice_data_bit_offset", slice_data_bit_offset: u16),
        ("first_mb_in_slice", first_mb_in_slice: u16),
        ("slice_type", slice_type: u8),
        ("direct_spatial_mv_pred_flag", direct_spatial_mv_pred_flag: u8),
        ("num_ref_idx_l0_active_minus1", num_ref_idx_l0_active_minus1: u8),
        ("num_ref_idx_l1_active_minus1", num_ref_idx_l1_active_minus1: u8),
        ("cabac_init_idc", cabac_init_idc: u8),
        ("slice_qp_delta", slice_qp_delta: i8),
        ("disable_deblocking_filter_idc", disable_deblocking_filter_idc: u8),
        ("slice_alpha_c0_offset_div2", slice_alpha_c0_offset_div2: i8),
        ("slice_beta_offset_div2", slice_beta_offset_div2: i8),
        ("RefPicList0", ref_pic_list0: [VaPictureH264; 32]),
        ("RefPicList1", ref_pic_list1: [VaPictureH264; 32]),
        ("luma_log2_weight_denom", luma_log2_weight_denom: u8),
        ("chroma_log2_weight_denom", chroma_log2_weight_denom: u8),
        ("luma_weight_l0_flag", luma_weight_l0_flag: u8),
        ("luma_weight_l0", luma_weight_l0: [i16; 32]),
        ("luma_offset_l0", luma_offset_l0: [i16; 32]),
        ("chroma_weight_l0_flag", chroma_weight_l0_flag: u8),
        ("chroma_weight_l0", chroma_weight_l0: [[i16; 2]; 32]),
        ("chroma_offset_l0", chroma_offset_l0: [[i16; 2]; 32]),
        ("luma_weight_l1_flag", luma_weight_l1_flag: u8),
        ("luma_weight_l1", luma_weight_l1: [i16; 32]),
        ("luma_offset_l1", luma_offset_l1: [i16; 32]),
        ("chroma_weight_l1_flag", chroma_weight_l1_flag: u8),
        ("chroma_weight_l1", chroma_weight_l1: [[i16; 2]; 32]),
        ("chroma_offset_l1", chroma_offset_l1: [[i16; 2]; 32]),
        ("va_reserved", va_reserved: [u32; VA_PADDING_LOW]),
    ],
    "VAIQMatrixBufferH264" => [
        ("ScalingList4x4", scaling_list4x4: [[u8; 16]; 6]),
        ("ScalingList8x8", scaling_list8x8: [[u8; 64]; 2]),
        ("va_reserved", va_reserved: [u32; VA_PADDING_LOW]),
    ],
    "VAPictureHEVC" => [
        ("picture_id", picture_id: VABufferID),
        ("pic_order_cnt", pic_order_cnt: i32),
        ("flags", flags: u32),
        ("va_reserved", va_reserved: [u32; VA_PADDING_LOW]),
    ],
    "VAPictureParameterBufferHEVC" => [
        ("CurrPic", curr_pic: VaPictureHevc),
        ("ReferenceFrames", reference_frames: [VaPictureHevc; 15]),
        ("pic_width_in_luma_samples", pic_width_in_luma_samples: u16),
        ("pic_height_in_luma_samples", pic_height_in_luma_samples: u16),
        ("pic_fields", pic_fields: FlagWord<u32>),
        ("sps_max_dec_pic_buffering_minus1", sps_max_dec_pic_buffering_minus1: u8),
        ("bit_depth_luma_minus8", bit_depth_luma_minus8: u8),
        ("bit_depth_chroma_minus8", bit_depth_chroma_minus8: u8),
        ("pcm_sample_bit_depth_luma_minus1", pcm_sample_bit_depth_luma_minus1: u8),
        ("pcm_sample_bit_depth_chroma_minus1", pcm_sample_bit_depth_chroma_minus1: u8),
        ("log2_min_luma_coding_block_size_minus3", log2_min_luma_coding_block_size_minus3: u8),
        ("log2_diff_max_min_luma_coding_block_size", log2_diff_max_min_luma_coding_block_size: u8),
        ("log2_min_transform_block_size_minus2", log2_min_transform_block_size_minus2: u8),
        ("log2_diff_max_min_transform_block_size", log2_diff_max_min_transform_block_size: u8),
        ("log2_min_pcm_luma_coding_block_size_minus3", log2_min_pcm_luma_coding_block_size_minus3: u8),
        ("log2_diff_max_min_pcm_luma_coding_block_size", log2_diff_max_min_pcm_luma_coding_block_size: u8),
        ("max_transform_hierarchy_depth_intra", max_transform_hierarchy_depth_intra: u8),
        ("max_transform_hierarchy_depth_inter", max_transform_hierarchy_depth_inter: u8),
        ("init_qp_minus26", init_qp_minus26: i8),
        ("diff_cu_qp_delta_depth", diff_cu_qp_delta_depth: u8),
        ("pps_cb_qp_offset", pps_cb_qp_offset: i8),
        ("pps_cr_qp_offset", pps_cr_qp_offset: i8),
        ("log2_parallel_merge_level_minus2", log2_parallel_merge_level_minus2: u8),
        ("num_tile_columns_minus1", num_tile_columns_minus1: u8),
        ("num_tile_rows_minus1", num_tile_rows_minus1: u8),
        ("column_width_minus1", column_width_minus1: [u16; 19]),
        ("row_height_minus1", row_height_minus1: [u16; 21]),
        ("slice_parsing_fields", slice_parsing_fields: FlagWord<u32>),
        ("log2_max_pic_order_cnt_lsb_minus4", log2_max_pic_order_cnt_lsb_minus4: u8),
        ("num_short_term_ref_pic_sets", num_short_term_ref_pic_sets: u8),
        ("num_long_term_ref_pic_sps", num_long_term_ref_pic_sps: u8),
        ("num_ref_idx_l0_default_active_minus1", num_ref_idx_l0_default_active_minus1: u8),
        ("num_ref_idx_l1_default_active_minus1", num_ref_idx_l1_default_active_minus1: u8),
        ("pps_beta_offset_div2", pps_beta_offset_div2: i8),
        ("pps_tc_offset_div2", pps_tc_offset_div2: i8),
        ("num_extra_slice_header_bits", num_extra_slice_header_bits: u8),
        ("st_rps_bits", st_rps_bits: u32),
        ("va_reserved", va_reserved: [u32; VA_PADDING_MEDIUM]),
    ],
    "VASliceParameterBufferHEVC" => [
        ("slice_data_size", slice_data_size: u32),
        ("slice_data_offset", slice_data_offset: u32),
        ("slice_data_flag", slice_data_flag: u32),
        ("slice_data_byte_offset", slice_data_byte_offset: u32),
        ("slice_segment_address", slice_segment_address: u32),
        ("RefPicList", ref_pic_list: [[u8; 15]; 2]),
        ("LongSliceFlags", long_slice_flags: FlagWord<u32>),
        ("collocated_ref_idx", collocated_ref_idx: u8),
        ("num_ref_idx_l0_active_minus1", num_ref_idx_l0_active_minus1: u8),
        ("num_ref_idx_l1_active_minus1", num_ref_idx_l1_active_minus1: u8),
        ("slice_qp_delta", slice_qp_delta: i8),
        ("slice_cb_qp_offset", slice_cb_qp_offset: i8),
        ("slice_cr_qp_offset", slice_cr_qp_offset: i8),
        ("slice_beta_offset_div2", slice_beta_offset_div2: i8),
        ("slice_tc_offset_div2", slice_tc_offset_div2: i8),
        ("luma_log2_weight_denom", luma_log2_weight_denom: u8),
        ("delta_chroma_log2_weight_denom", delta_chroma_log2_weight_denom: i8),
        ("delta_luma_weight_l0", delta_luma_weight_l0: [i8; 15]),
        ("luma_offset_l0", luma_offset_l0: [i8; 15]),
        ("delta_chroma_weight_l0", delta_chroma_weight_l0: [[i8; 2]; 15]),
        ("ChromaOffsetL0", chroma_offset_l0: [[i8; 2]; 15]),
        ("delta_luma_weight_l1", delta_luma_weight_l1: [i8; 15]),
        ("luma_offset_l1", luma_offset_l1: [i8; 15]),
        ("delta_chroma_weight_l1", delta_chroma_weight_l1: [[i8; 2]; 15]),
        ("ChromaOffsetL1", chroma_offset_l1: [[i8; 2]; 15]),
        ("five_minus_max_num_merge_cand", five_minus_max_num_merge_cand: u8),
        ("num_entry_point_offsets", num_entry_point_offsets: u16),
        ("entry_offset_to_subset_array", entry_offset_to_subset_array: u16),
        ("slice_data_num_emu_prevn_bytes", slice_data_num_emu_prevn_bytes: u16),
        ("va_reserved", va_reserved: [u32; VA_PADDING_LOW - 2]),
    ],
    "VADecPictureParameterBufferVP9" => [
        ("frame_width", frame_width: u16),
        ("frame_height", frame_height: u16),
        ("reference_frames", reference_frames: [VASurfaceID; 8]),
        ("pic_fields", pic_fields: FlagWord<u32>),
        ("filter_level", filter_level: u8),
        ("sharpness_level", sharpness_level: u8),
        ("log2_tile_rows", log2_tile_rows: u8),
        ("log2_tile_columns", log2_tile_columns: u8),
        ("frame_header_length_in_bytes", frame_header_length_in_bytes: u8),
        ("first_partition_size", first_partition_size: u16),
        ("mb_segment_tree_probs", mb_segment_tree_probs: [u8; 7]),
        ("segment_pred_probs", segment_pred_probs: [u8; 3]),
        ("profile", profile: u8),
        ("bit_depth", bit_depth: u8),
        ("va_reserved", va_reserved: [u32; VA_PADDING_MEDIUM]),
    ],
    // `segment_flags` is the narrowest flag union in the records — a `uint16_t`'s worth of bits in a
    // union whose `value` word is a `uint16_t` — which is why it is measured rather than assumed to
    // be the `u32` the other twenty-one are.
    "VASegmentParameterVP9" => [
        ("segment_flags", segment_flags: FlagWord<u16>),
        ("filter_level", filter_level: [[u8; 2]; 4]),
        ("luma_ac_quant_scale", luma_ac_quant_scale: i16),
        ("luma_dc_quant_scale", luma_dc_quant_scale: i16),
        ("chroma_ac_quant_scale", chroma_ac_quant_scale: i16),
        ("chroma_dc_quant_scale", chroma_dc_quant_scale: i16),
        ("va_reserved", va_reserved: [u32; VA_PADDING_LOW]),
    ],
    "VASliceParameterBufferVP9" => [
        ("slice_data_size", slice_data_size: u32),
        ("slice_data_offset", slice_data_offset: u32),
        ("slice_data_flag", slice_data_flag: u32),
        ("seg_param", seg_param: [VaSegmentParameterVp9; 8]),
        ("va_reserved", va_reserved: [u32; VA_PADDING_LOW]),
    ],
    // The AV1 picture record: seven unions in one member list, one nested record each of two kinds,
    // an eight-element surface array — and the one member this check does not cover, which is
    // `anchor_frames_list`; see [`NOT_CHECKED`].
    "VADecPictureParameterBufferAV1" => [
        ("profile", profile: u8),
        ("order_hint_bits_minus_1", order_hint_bits_minus_1: u8),
        ("bit_depth_idx", bit_depth_idx: u8),
        ("matrix_coefficients", matrix_coefficients: u8),
        ("seq_info_fields", seq_info_fields: FlagWord<u32>),
        ("current_frame", current_frame: VASurfaceID),
        ("current_display_picture", current_display_picture: VASurfaceID),
        ("anchor_frames_num", anchor_frames_num: u8),
        ("frame_width_minus1", frame_width_minus1: u16),
        ("frame_height_minus1", frame_height_minus1: u16),
        ("output_frame_width_in_tiles_minus_1", output_frame_width_in_tiles_minus_1: u16),
        ("output_frame_height_in_tiles_minus_1", output_frame_height_in_tiles_minus_1: u16),
        ("ref_frame_map", ref_frame_map: [VASurfaceID; 8]),
        ("ref_frame_idx", ref_frame_idx: [u8; 7]),
        ("primary_ref_frame", primary_ref_frame: u8),
        ("order_hint", order_hint: u8),
        ("seg_info", seg_info: VaSegmentationStructAv1),
        ("film_grain_info", film_grain_info: VaFilmGrainStructAv1),
        ("tile_cols", tile_cols: u8),
        ("tile_rows", tile_rows: u8),
        ("width_in_sbs_minus_1", width_in_sbs_minus_1: [u16; 63]),
        ("height_in_sbs_minus_1", height_in_sbs_minus_1: [u16; 63]),
        ("tile_count_minus_1", tile_count_minus_1: u16),
        ("context_update_tile_id", context_update_tile_id: u16),
        ("pic_info_fields", pic_info_fields: FlagWord<u32>),
        ("superres_scale_denominator", superres_scale_denominator: u8),
        ("interp_filter", interp_filter: u8),
        ("filter_level", filter_level: [u8; 2]),
        ("filter_level_u", filter_level_u: u8),
        ("filter_level_v", filter_level_v: u8),
        ("loop_filter_info_fields", loop_filter_info_fields: FlagWord<u8>),
        ("ref_deltas", ref_deltas: [i8; 8]),
        ("mode_deltas", mode_deltas: [i8; 2]),
        ("base_qindex", base_qindex: u8),
        ("y_dc_delta_q", y_dc_delta_q: i8),
        ("u_dc_delta_q", u_dc_delta_q: i8),
        ("u_ac_delta_q", u_ac_delta_q: i8),
        ("v_dc_delta_q", v_dc_delta_q: i8),
        ("v_ac_delta_q", v_ac_delta_q: i8),
        ("qmatrix_fields", qmatrix_fields: FlagWord<u16>),
        ("mode_control_fields", mode_control_fields: FlagWord<u32>),
        ("cdef_damping_minus_3", cdef_damping_minus_3: u8),
        ("cdef_bits", cdef_bits: u8),
        ("cdef_y_strengths", cdef_y_strengths: [u8; 8]),
        ("cdef_uv_strengths", cdef_uv_strengths: [u8; 8]),
        ("loop_restoration_fields", loop_restoration_fields: FlagWord<u16>),
        ("wm", wm: [VaWarpedMotionParamsAv1; 7]),
        ("va_reserved", va_reserved: [u32; VA_PADDING_MEDIUM]),
    ],
    "VASegmentationStructAV1" => [
        ("segment_info_fields", segment_info_fields: FlagWord<u32>),
        ("feature_data", feature_data: [[i16; 8]; 8]),
        ("feature_mask", feature_mask: [u8; 8]),
        ("va_reserved", va_reserved: [u32; VA_PADDING_LOW]),
    ],
    "VAFilmGrainStructAV1" => [
        ("film_grain_info_fields", film_grain_info_fields: FlagWord<u32>),
        ("grain_seed", grain_seed: u16),
        ("num_y_points", num_y_points: u8),
        ("point_y_value", point_y_value: [u8; 14]),
        ("point_y_scaling", point_y_scaling: [u8; 14]),
        ("num_cb_points", num_cb_points: u8),
        ("point_cb_value", point_cb_value: [u8; 10]),
        ("point_cb_scaling", point_cb_scaling: [u8; 10]),
        ("num_cr_points", num_cr_points: u8),
        ("point_cr_value", point_cr_value: [u8; 10]),
        ("point_cr_scaling", point_cr_scaling: [u8; 10]),
        ("ar_coeffs_y", ar_coeffs_y: [i8; 24]),
        ("ar_coeffs_cb", ar_coeffs_cb: [i8; 25]),
        ("ar_coeffs_cr", ar_coeffs_cr: [i8; 25]),
        ("cb_mult", cb_mult: u8),
        ("cb_luma_mult", cb_luma_mult: u8),
        ("cb_offset", cb_offset: u16),
        ("cr_mult", cr_mult: u8),
        ("cr_luma_mult", cr_luma_mult: u8),
        ("cr_offset", cr_offset: u16),
        ("va_reserved", va_reserved: [u32; VA_PADDING_LOW]),
    ],
    // The only record whose enumerated member is a field of its own rather than a discriminator.
    "VAWarpedMotionParamsAV1" => [
        ("wmtype", wmtype: HeaderEnum<u32>),
        ("wmmat", wmmat: [i32; 8]),
        ("invalid", invalid: u8),
        ("va_reserved", va_reserved: [u32; VA_PADDING_LOW]),
    ],
    "VASliceParameterBufferAV1" => [
        ("slice_data_size", slice_data_size: u32),
        ("slice_data_offset", slice_data_offset: u32),
        ("slice_data_flag", slice_data_flag: u32),
        ("tile_row", tile_row: u16),
        ("tile_column", tile_column: u16),
        ("tg_start", tg_start: u16),
        ("tg_end", tg_end: u16),
        ("anchor_frame_idx", anchor_frame_idx: u8),
        ("tile_idx_in_tile_list", tile_idx_in_tile_list: u16),
        ("va_reserved", va_reserved: [u32; VA_PADDING_LOW]),
    ],
    // The record libva writes a chain of into: the two pointers are the only `void *` in any record
    // the backend submits, and they are the only members here whose C class is a pointer.
    "VACodedBufferSegment" => [
        ("size", size: u32),
        ("bit_offset", bit_offset: u32),
        ("status", status: u32),
        ("reserved", reserved: u32),
        ("buf", buf: *mut c_void),
        ("next", next: *mut c_void),
        ("va_reserved", va_reserved: [u32; VA_PADDING_LOW]),
    ],
    "VABufferInfo" => [
        ("handle", handle: usize),
        ("type", buffer_type: u32),
        ("mem_type", mem_type: u32),
        ("mem_size", mem_size: usize),
        ("va_reserved", va_reserved: [u32; VA_PADDING_LOW]),
    ],
    "VAEncSequenceParameterBufferH264" => [
        ("seq_parameter_set_id", seq_parameter_set_id: u8),
        ("level_idc", level_idc: u8),
        ("intra_period", intra_period: u32),
        ("intra_idr_period", intra_idr_period: u32),
        ("ip_period", ip_period: u32),
        ("bits_per_second", bits_per_second: u32),
        ("max_num_ref_frames", max_num_ref_frames: u32),
        ("picture_width_in_mbs", picture_width_in_mbs: u16),
        ("picture_height_in_mbs", picture_height_in_mbs: u16),
        ("seq_fields", seq_fields: FlagWord<u32>),
        ("bit_depth_luma_minus8", bit_depth_luma_minus8: u8),
        ("bit_depth_chroma_minus8", bit_depth_chroma_minus8: u8),
        ("num_ref_frames_in_pic_order_cnt_cycle", num_ref_frames_in_pic_order_cnt_cycle: u8),
        ("offset_for_non_ref_pic", offset_for_non_ref_pic: i32),
        ("offset_for_top_to_bottom_field", offset_for_top_to_bottom_field: i32),
        ("offset_for_ref_frame", offset_for_ref_frame: [i32; 256]),
        ("frame_cropping_flag", frame_cropping_flag: u8),
        ("frame_crop_left_offset", frame_crop_left_offset: u32),
        ("frame_crop_right_offset", frame_crop_right_offset: u32),
        ("frame_crop_top_offset", frame_crop_top_offset: u32),
        ("frame_crop_bottom_offset", frame_crop_bottom_offset: u32),
        ("vui_parameters_present_flag", vui_parameters_present_flag: u8),
        ("vui_fields", vui_fields: FlagWord<u32>),
        ("aspect_ratio_idc", aspect_ratio_idc: u8),
        ("sar_width", sar_width: u32),
        ("sar_height", sar_height: u32),
        ("num_units_in_tick", num_units_in_tick: u32),
        ("time_scale", time_scale: u32),
        ("va_reserved", va_reserved: [u32; VA_PADDING_LOW]),
    ],
    "VAEncPictureParameterBufferH264" => [
        ("CurrPic", curr_pic: VaPictureH264),
        ("ReferenceFrames", reference_frames: [VaPictureH264; 16]),
        ("coded_buf", coded_buf: VABufferID),
        ("pic_parameter_set_id", pic_parameter_set_id: u8),
        ("seq_parameter_set_id", seq_parameter_set_id: u8),
        ("last_picture", last_picture: u8),
        ("frame_num", frame_num: u16),
        ("pic_init_qp", pic_init_qp: u8),
        ("num_ref_idx_l0_active_minus1", num_ref_idx_l0_active_minus1: u8),
        ("num_ref_idx_l1_active_minus1", num_ref_idx_l1_active_minus1: u8),
        ("chroma_qp_index_offset", chroma_qp_index_offset: i8),
        ("second_chroma_qp_index_offset", second_chroma_qp_index_offset: i8),
        ("pic_fields", pic_fields: FlagWord<u32>),
        ("va_reserved", va_reserved: [u32; VA_PADDING_LOW]),
    ],
    "VAEncSliceParameterBufferH264" => [
        ("macroblock_address", macroblock_address: u32),
        ("num_macroblocks", num_macroblocks: u32),
        ("macroblock_info", macroblock_info: VABufferID),
        ("slice_type", slice_type: u8),
        ("pic_parameter_set_id", pic_parameter_set_id: u8),
        ("idr_pic_id", idr_pic_id: u16),
        ("pic_order_cnt_lsb", pic_order_cnt_lsb: u16),
        ("delta_pic_order_cnt_bottom", delta_pic_order_cnt_bottom: i32),
        ("delta_pic_order_cnt", delta_pic_order_cnt: [i32; 2]),
        ("direct_spatial_mv_pred_flag", direct_spatial_mv_pred_flag: u8),
        ("num_ref_idx_active_override_flag", num_ref_idx_active_override_flag: u8),
        ("num_ref_idx_l0_active_minus1", num_ref_idx_l0_active_minus1: u8),
        ("num_ref_idx_l1_active_minus1", num_ref_idx_l1_active_minus1: u8),
        ("RefPicList0", ref_pic_list0: [VaPictureH264; 32]),
        ("RefPicList1", ref_pic_list1: [VaPictureH264; 32]),
        ("luma_log2_weight_denom", luma_log2_weight_denom: u8),
        ("chroma_log2_weight_denom", chroma_log2_weight_denom: u8),
        ("luma_weight_l0_flag", luma_weight_l0_flag: u8),
        ("luma_weight_l0", luma_weight_l0: [i16; 32]),
        ("luma_offset_l0", luma_offset_l0: [i16; 32]),
        ("chroma_weight_l0_flag", chroma_weight_l0_flag: u8),
        ("chroma_weight_l0", chroma_weight_l0: [[i16; 2]; 32]),
        ("chroma_offset_l0", chroma_offset_l0: [[i16; 2]; 32]),
        ("luma_weight_l1_flag", luma_weight_l1_flag: u8),
        ("luma_weight_l1", luma_weight_l1: [i16; 32]),
        ("luma_offset_l1", luma_offset_l1: [i16; 32]),
        ("chroma_weight_l1_flag", chroma_weight_l1_flag: u8),
        ("chroma_weight_l1", chroma_weight_l1: [[i16; 2]; 32]),
        ("chroma_offset_l1", chroma_offset_l1: [[i16; 2]; 32]),
        ("cabac_init_idc", cabac_init_idc: u8),
        ("slice_qp_delta", slice_qp_delta: i8),
        ("disable_deblocking_filter_idc", disable_deblocking_filter_idc: u8),
        ("slice_alpha_c0_offset_div2", slice_alpha_c0_offset_div2: i8),
        ("slice_beta_offset_div2", slice_beta_offset_div2: i8),
        ("va_reserved", va_reserved: [u32; VA_PADDING_LOW]),
    ],
    // The header's record is a struct with a `VAEncMiscParameterType` discriminator and a flexible
    // `uint32_t data[]` tail (`va.h:2475-2478`), so `type` is the whole of it that has a name. Our
    // declaration is the four-byte header on its own, which is what a `vaCreateBuffer` of this size
    // asks for.
    "VAEncMiscParameterBuffer" => [
        ("type", type_: HeaderEnum<u32>),
    ],
    "VAEncMiscParameterFrameRate" => [
        ("framerate", framerate: u32),
        ("framerate_flags", framerate_flags: FlagWord<u32>),
        ("va_reserved", va_reserved: [u32; VA_PADDING_LOW]),
    ],
    "VAEncMiscParameterQuantization" => [
        ("quantization_flags", quantization_flags: FlagWord<u32>),
        ("va_reserved", va_reserved: u32),
    ],
    // The record with no hole, so the tiling test proves nothing on it and this one has to carry it.
    // Four of its twelve members are `uint32_t` and one is a flag union's `value` word: the two are
    // the same four bytes, which is exactly why the shape assertion is not optional here.
    "VAEncMiscParameterRateControl" => [
        ("bits_per_second", bits_per_second: u32),
        ("target_percentage", target_percentage: u32),
        ("window_size", window_size: u32),
        ("initial_qp", initial_qp: u32),
        ("min_qp", min_qp: u32),
        ("basic_unit_size", basic_unit_size: u32),
        ("rc_flags", rc_flags: FlagWord<u32>),
        ("ICQ_quality_factor", icq_quality_factor: u32),
        ("max_qp", max_qp: u32),
        ("quality_factor", quality_factor: u32),
        ("target_frame_size", target_frame_size: u32),
        ("va_reserved", va_reserved: [u32; VA_PADDING_LOW]),
    ],
    // The one record here that is not a libva record, and the only one whose C name needs `struct`
    // spelled in front of it — which is why the probe declares its instance rather than writing
    // `((R *)0)->m`: a C struct has no typedef, so there is no `R` to write.
    "struct drm_prime_handle" => [
        ("handle", handle: u32),
        ("flags", flags: u32),
        ("fd", fd: c_int),
    ],
];

/// Members of [`SUBMITTED_RECORDS`] that this check deliberately does **not** cover, each with the
/// reason it is not settled here.
///
/// This is a named list rather than a silent omission because a check that quietly stops covering a
/// field reads exactly like one that never covered it: [`every_submitted_record_agrees_with_the_header_on_width_and_signedness`]
/// asserts that the set of uncovered members is precisely the `padding_*` fields — which C has no
/// name for and so no type to compare — plus these. Adding a row here is therefore a claim somebody
/// has to read, and removing a member from [`SUBMITTED_RECORDS`] while leaving its row here fails.
const NOT_CHECKED: &[(&str, &str, &str)] = &[(
    "VADecPictureParameterBufferAV1",
    "anchor_frames_list",
    "the header declares a `VASurfaceID *` (`va_dec_av1.h:330`) and `ffi.rs` declares a `u64`, on \
     purpose: the field is an address into a list the application owns, large-scale tile is declined, \
     and `record_bytes` flattens the record before any address in it could escape. So this is a \
     genuine shape disagreement between the header and the declaration, and the check that can see it \
     is the one that would have to call the declaration wrong. It is settled in `ffi.rs` instead — \
     `the_pointer_shaped_field_is_an_address_and_not_a_pointer` there pins the field's width with \
     `size_of_val` against `size_of::<*mut c_void>()`, and the record's `Send + Sync` is what fails to \
     compile if the field ever becomes a raw pointer — and repeating the same claim in a second file \
     would be a second thing to keep in step, not a second guard. Add it here as `FlagWord` of a \
     pointer type if `ffi.rs` ever changes its mind.",
)];

/// The C half of the shape, width and signedness check, as macros the generated source uses.
///
/// `CKIND(x)` is the one number both sides agree or disagree on. `_Generic` produces the sign — it
/// matches the controlling expression's type against a list of *distinct* C types, so a typedef
/// answers as its underlying type without its spelling appearing anywhere below — and `sizeof`
/// produces the width. `default: 2` is "not an integer at all", and it is only ever reached for a
/// `float`, a `double` or a type that is not a number, because [`CK_CLASS`] is what tells the five
/// non-integer shapes apart.
///
/// **`CK_CLASS` takes a `__typeof__` and that is load-bearing.** `__builtin_classify_type(member)`
/// reports an enumerated member as an integer, because the argument is converted to the enum's
/// underlying type first; `__builtin_classify_type(__typeof__(member))` reports it as
/// [`CK_ENUMERAL`]. The same applies to an array, which decays to a pointer as an argument and stays
/// an array as a type — so the class assertion is also how a member that stopped being an array is
/// caught.
///
/// `char` is listed apart from `signed char` and gets its **own** code (3) on purpose: plain
/// `char`'s signedness is implementation-defined and is signed only on this target. No member here
/// declares one, so no field lands on 3; if one ever did, the comparison would fail rather than
/// quietly assume a signedness the header never promised.
const CK_PROLOGUE: &str = "\
#define CK_SIGN(x) _Generic((x), \
    _Bool: 0, \
    char: 3, \
    signed char: 1, unsigned char: 0, \
    short: 1, unsigned short: 0, \
    int: 1, unsigned int: 0, \
    long: 1, unsigned long: 0, \
    long long: 1, unsigned long long: 0, \
    float: 3, double: 3, \
    default: 2)
#define CKIND(x) (CK_SIGN(x) * 256 + (int)sizeof(x))
#define CK_CLASS(x) __builtin_classify_type(__typeof__(x))
";

/// The C source for the shape, width and signedness check: one assertion pair per member, over
/// [`TYPED_RECORDS`].
///
/// Every record gets a real **instance** rather than the `((R *)0)->member` this used to spell, and
/// it is not a style choice. The peel an array needs — `m[0]`, and `m[0][0]` for the two-dimensional
/// ones — is an address computation off a null pointer, and clang's `-Wnull-dereference` fires on it
/// even inside `_Static_assert`. A named object of the record's own type has the same types, the
/// same offsets and no warning.
///
/// `rewrite` is handed `(record, member, kind)` and returns the kind the C side is compared with.
/// The real check passes [`CKindOf::KIND`] through; the test that proves the check can fail passes a
/// function that returns a deliberately wrong kind for six named members, and asserts the compiler
/// rejects the result. **A check that cannot fail is worse than no check**, so that failure is a
/// committed test rather than something a reviewer has to take on trust.
fn typed_kind_source(rewrite: impl Fn(&str, &str, CKind) -> CKind) -> String {
    let mut source = String::from(
        "#include <stdint.h>\n#include <va/va.h>\n#include <va/va_drm.h>\n#include <va/va_dec_av1.h>\n\
         #include <va/va_dec_hevc.h>\n#include <va/va_dec_vp9.h>\n#include <va/va_enc_h264.h>\n#include <drm/drm.h>\n",
    );
    source.push_str(CK_PROLOGUE);
    for (index, record) in TYPED_RECORDS.iter().enumerate() {
        source.push_str(&format!("static {r} ck_rec_{index};\n", r = record.c_name));
    }
    for (index, record) in TYPED_RECORDS.iter().enumerate() {
        source.push_str(&format!("\n/* {} */\n", record.c_name));
        for field in record.fields {
            // The member itself, then one `[0]` per level of the Rust type — see `CKindOf::DEPTH`.
            let mut peeled = format!("ck_rec_{index}.{}", field.c_name);
            for _ in 0..field.depth {
                peeled.push_str("[0]");
            }
            let label = format!("{}.{}", record.c_name, field.c_name);
            let kind = rewrite(record.c_name, field.c_name, field.kind);
            source.push_str(&format!(
                "_Static_assert(CK_CLASS({peeled}) == {class}, \"{label}: we declare {ours}, and the header declares a member of a different shape\");\n",
                class = kind.expected_class(),
                ours = kind.describe(),
            ));
            let Some(code) = kind.code() else { continue };
            // For a union the number is about the *word inside it*, which is the one thing this
            // declaration actually stores; for everything else it is about the member itself.
            let measured = match kind {
                CKind::UnionValue { .. } => {
                    assert!(!field.union_word.is_empty(), "{label}: spelled as a flag union, but its type names no word of that union to measure");
                    format!("{peeled}.{}", field.union_word)
                }
                _ => peeled.clone(),
            };
            source.push_str(&format!(
                "_Static_assert(CKIND({measured}) == {code}, \"{label}: we declare {ours}, and the header declares a different width or signedness\");\n",
                ours = kind.describe(),
            ));
        }
    }
    source
}

/// Every record the backend submits agrees with `/usr/include/va/` on every member's **shape, width
/// and signedness**, and with [`SUBMITTED_RECORDS`] on every member's width.
///
/// The width half of the second claim is what keeps this section honest. The type spelled in
/// [`TYPED_RECORDS`] is a transcription, so a wrong one would otherwise be able to agree with
/// itself: `size_of` of a mistyped type is the size of the type that was written down, not of the
/// field. Comparing each width with the row [`SUBMITTED_RECORDS`] already pinned ties it to
/// `offset_of!` on the real struct — which is what the tiling test uses to prove the fields cover
/// the record — so the two tables cannot drift apart on width, and the signedness and the shape are
/// left with nothing to hide behind. Both markers — [`FlagWord`] and [`HeaderEnum`] — are
/// `#[repr(transparent)]`, so a marked field's width *is* the width of the type underneath it and
/// the cross-check reads the declaration rather than the marker.
///
/// The coverage assertion is the other half of the claim: every member [`SUBMITTED_RECORDS`] names
/// that is not a `padding_*` field — which C gives no name to, and so no type — is either in
/// [`TYPED_RECORDS`] or listed in [`NOT_CHECKED`] with a reason. A new record added to
/// [`SUBMITTED_RECORDS`] therefore cannot join the list without this test naming it.
#[test]
fn every_submitted_record_agrees_with_the_header_on_width_and_signedness() {
    let Some(cc) = require_headers_and_compiler("every_submitted_record_agrees_with_the_header_on_width_and_signedness") else {
        return;
    };
    let mut checked = 0usize;
    for record in TYPED_RECORDS {
        assert!(!record.fields.is_empty(), "{}: no members listed, so the loop below proved nothing", record.c_name);
        let submitted = SUBMITTED_RECORDS
            .iter()
            .find(|r| r.c_name == record.c_name)
            .unwrap_or_else(|| panic!("{} is in TYPED_RECORDS but not in SUBMITTED_RECORDS, so it is not a record this backend submits", record.c_name));
        for field in record.fields {
            let row = submitted
                .fields
                .iter()
                .find(|f| f.c_name == field.c_name)
                .unwrap_or_else(|| panic!("{}.{} is not a member the header names in SUBMITTED_RECORDS", record.c_name, field.c_name));
            assert_eq!(
                field.width, row.width,
                "{}.{}: the type written in TYPED_RECORDS is {} bytes wide and the type the SUBMITTED_RECORDS row pinned is {} — one of the two tables is stale",
                record.c_name, field.c_name, field.width, row.width
            );
            checked += 1;
        }
    }
    // The coverage claim, in both directions. A member that stopped being checked without being named
    // is the failure mode this is here to prevent, and an excuse outliving the member it excuses is
    // the other one.
    let mut uncovered = Vec::new();
    for record in SUBMITTED_RECORDS {
        for field in record.fields {
            if field.c_name == "-" {
                continue; // a `padding_*` field: C gives the gap no name, so there is no type to compare
            }
            let covered = TYPED_RECORDS.iter().find(|r| r.c_name == record.c_name).is_some_and(|r| r.fields.iter().any(|f| f.c_name == field.c_name));
            let excused = NOT_CHECKED.iter().any(|(r, f, _)| *r == record.c_name && *f == field.c_name);
            if !covered && !excused {
                uncovered.push(format!("{}.{}", record.c_name, field.c_name));
            }
        }
    }
    assert!(
        uncovered.is_empty(),
        "{} member(s) of SUBMITTED_RECORDS are neither in TYPED_RECORDS nor in NOT_CHECKED, so this check silently stopped covering {}: {}",
        uncovered.len(),
        if uncovered.len() == 1 { "it" } else { "them" },
        uncovered.join(", ")
    );
    for (record, field, reason) in NOT_CHECKED {
        assert!(!reason.trim().is_empty(), "{record}.{field}: listed as not checked with no reason given");
        assert!(
            SUBMITTED_RECORDS.iter().any(|r| r.c_name == *record && r.fields.iter().any(|f| f.c_name == *field)),
            "{record}.{field} is listed in NOT_CHECKED but is not a member of that record any more, so the excuse outlived what it excuses"
        );
    }
    assert_eq!(checked, 384, "every member of every record was cross-checked against SUBMITTED_RECORDS");
    assert_eq!(TYPED_RECORDS.len(), 30, "every record this backend submits is checked");
    let out = compile_probe(&cc, "va-typed-kind", &typed_kind_source(|_, _, kind| kind));
    assert!(
        out.status.success(),
        "{cc} disagrees with our declarations about a member's shape, its width or its signedness:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// **The check is proved able to fail, in each of the three directions it now exists for**, by
/// running the real generator and the real compiler against declarations that are wrong on purpose.
///
/// A check extended from nineteen assertions to seven hundred without re-proving it red is an
/// untested extension, so every direction this section added is exercised here with a deliberately
/// wrong answer: the member asked about is a **newly covered** one each time, not one of the two this
/// started with, and each is the *correct* answer with one thing changed, so the source compiled here
/// differs from the one the test above compiles in exactly that and nowhere else.
///
/// * `VASliceParameterBufferHEVC.slice_qp_delta` — we declare `i8`, the header declares `int8_t`,
///   and the number compared against is the one an `unsigned char` produces. **A Rust `i8` for a C
///   signed 8-bit field read with the wrong sign** — the same byte, a different answer for every
///   negative quantiser offset;
/// * `VASegmentParameterVP9.luma_ac_quant_scale` — the signedness is right and the width is not: a
///   four-byte number compared against the header's two-byte `int16_t`. The tiling test catches a
///   width that is wrong on its own, because the next member moves; what it cannot catch is a width
///   wrong *and* the next one compensating, which is exactly what a hand-written width table invites;
/// * `VADecPictureParameterBufferAV1.seq_info_fields` — the **flag-union shape**, which is new: the
///   source claims the header declares a plain `uint32_t` where it declares the sequence-parameter
///   union, and **both** assertions fire — the class wants 1 where the compiler says 13, and the
///   number wants 4 where the compiler measures `CK_SIGN` on a union and gets its `default: 2`
///   bucket;
/// * `VAWarpedMotionParamsAV1.wmtype` — the **enumerated shape**, also new, the same way round: the
///   source claims a plain `uint32_t` where the header declares `VAWarpedMotionType`, and only the
///   class assertion fires, because there is no signedness to compare for an enum and claiming one
///   would be the bug;
/// * `VAPictureH264.TopFieldOrderCnt` and `VAImage.num_palette_entries`, the two this check started
///   with, kept so that the original proof is not lost when the rest of the table moves.
///
/// The compiler must **reject** it, and its output must name all six — a failure a reader cannot act
/// on is not much better than no failure.
#[test]
fn the_shape_width_and_signedness_check_rejects_a_declaration_that_is_wrong() {
    let Some(cc) = require_headers_and_compiler("the_shape_width_and_signedness_check_rejects_a_declaration_that_is_wrong") else {
        return;
    };
    let source = typed_kind_source(|record, field, kind| match (record, field) {
        // signed 1 -> unsigned 1: the same byte, read with the wrong sign.
        ("VAPictureH264", "TopFieldOrderCnt") | ("VAImage", "num_palette_entries") | ("VASliceParameterBufferHEVC", "slice_qp_delta") => {
            CKind::Integer { sign: Sign::Unsigned, width: 1 }
        }
        // signed 2 -> signed 4: the right sign, twice the width.
        ("VASegmentParameterVP9", "luma_ac_quant_scale") => CKind::Integer { sign: Sign::Signed, width: 4 },
        // A flag union read as a plain integer: the shape is wrong, and so is the number.
        ("VADecPictureParameterBufferAV1", "seq_info_fields") => CKind::Integer { sign: Sign::Unsigned, width: 4 },
        // An enumerated type read as a plain integer: the shape is wrong.
        ("VAWarpedMotionParamsAV1", "wmtype") => CKind::Integer { sign: Sign::Unsigned, width: 4 },
        _ => kind,
    });
    let out = compile_probe(&cc, "va-typed-kind-wrong", &source);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        !out.status.success(),
        "{cc} accepted six members compared against the wrong shape, width or signedness, so the check is not checking anything:\n{source}"
    );
    for field in [
        "VAPictureH264.TopFieldOrderCnt",
        "VAImage.num_palette_entries",
        "VASliceParameterBufferHEVC.slice_qp_delta",
        "VASegmentParameterVP9.luma_ac_quant_scale",
        "VADecPictureParameterBufferAV1.seq_info_fields",
        "VAWarpedMotionParamsAV1.wmtype",
    ] {
        assert!(
            stderr.contains(field),
            "{cc} rejected the wrong source but did not name {field}, so the failure would not tell a reader which declaration to fix:\n{stderr}"
        );
    }
}

/// **This machine actually ran the guard, or said out loud that it did not.**
///
/// Every other check in this module opens with `if the header is not there { eprintln!("SKIPPED");
/// return }`, which makes it **pass while checking nothing**. `cargo test` captures a passing test's
/// output, so the note reaches nobody: the run is green, this file reads as an assertion, and a
/// runner without `libva-dev` reports that a hand-declared FFI layer is verified when nothing ran. A
/// guard that does that is worse than no guard, because it reads as assurance it never gave.
///
/// This test cannot do anything about those `eprintln!`s without changing checks that currently
/// work, so it does the next best thing and makes **itself** the visible one: on a machine with the
/// headers it compiles and runs a probe, so "available" means "working", not merely "a file is
/// there"; on a machine without them it **fails**, naming everything that went unchecked — unless
/// [`ACKNOWLEDGE_ENV`] says `skip`, which is the one way past it and is a claim that the person
/// setting it knows the guard did not run.
///
/// **The honest answer about CI is that it does not run there at all.** This repository's workflows
/// are `packaging-lint.yml`, `release.yml` and `windows-arm64.yml`; none of them runs `cargo test`,
/// and none installs `libva-dev`. So there is currently no GitHub runner for this guard to skip on,
/// and wiring one up (a workflow step that installs `libva-dev` and runs
/// `cargo test -p filmcraft-platform`) is the change that would make the rest of this module worth
/// having on every commit. That is outside this file; this test is the part that can live here.
#[test]
fn the_abi_guard_ran_on_this_machine_or_announced_that_it_did_not() {
    let mut missing: Vec<String> = Vec::new();
    for file in ["va.h", "va_drm.h", "va_dec_av1.h", "va_dec_hevc.h", "va_dec_vp9.h", "va_enc_h264.h"] {
        if !std::path::Path::new(LIBVA_HEADERS).join(file).is_file() {
            missing.push(format!("{LIBVA_HEADERS}/{file}"));
        }
    }
    if !drm_headers_present() {
        missing.push(format!("{DRM_HEADERS}/drm.h"));
    }
    let cc = ["cc", "gcc", "clang"].into_iter().find_map(which);
    if cc.is_none() {
        missing.push("a C compiler (cc, gcc or clang)".to_string());
    }
    if !missing.is_empty() {
        if std::env::var(ACKNOWLEDGE_ENV).as_deref() == Ok("skip") {
            eprintln!(
                "{ACKNOWLEDGE_ENV}=skip: this machine cannot run the ABI guard — {} missing. Nothing about the hand-declared ABI was verified here.",
                missing.join(", ")
            );
            return;
        }
        panic!(
            "the ABI guard did not run on this machine: {} missing.\n\
             Every other test in this module passed **without checking anything**, because each of them skips when this is true.\n\
             Install the headers (`libva-dev` and `libdrm-dev` on Debian/Ubuntu) and a C compiler, or set {ACKNOWLEDGE_ENV}=skip to record, in writing, that this machine cannot check the declarations.",
            missing.join(", ")
        );
    }
    // "The headers are installed" is not "the guard works". Compile and run something, so the pass
    // below means a C compiler read them.
    let cc = cc.unwrap_or_else(|| panic!("the compiler disappeared between the check and the use of it"));
    let out = compile_probe(
        &cc,
        "va-guard-alive",
        &format!("#include <va/va.h>\n_Static_assert(sizeof(VAPictureH264) == {}, \"VAPictureH264\");\n", size_of::<VaPictureH264>()),
    );
    assert!(out.status.success(), "{cc} is installed but rejected the headers:\n{}", String::from_utf8_lossy(&out.stderr));
    let src = typed_kind_source(|_, _, kind| kind);
    let out = compile_probe(&cc, "va-guard-alive-kind", &src);
    assert!(
        out.status.success(),
        "{cc} is installed but rejected the shape-width-and-signedness source, so the guard would have been failing rather than skipping:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Set to `skip` to let this module's tests pass **without checking anything**, on a machine that
/// has no libva headers. Setting it is a claim that whoever set it knows the guard did not run.
const ACKNOWLEDGE_ENV: &str = "FILMCRAFT_ABI_GUARD";

/// `None` only ever comes back on the acknowledged path, so the caller can `return` and the test
/// still reads as a decision somebody made rather than as a silent skip.
fn require_headers_and_compiler(what: &str) -> Option<String> {
    let cc = ["cc", "gcc", "clang"].into_iter().find_map(which);
    let mut absent: Vec<String> = Vec::new();
    if !std::path::Path::new(LIBVA_HEADERS).join("va.h").is_file() {
        absent.push(format!("{LIBVA_HEADERS}/va.h"));
    }
    if !drm_headers_present() {
        absent.push(format!("{DRM_HEADERS}/drm.h"));
    }
    if cc.is_none() {
        absent.push("a C compiler (cc, gcc or clang)".to_string());
    }
    if absent.is_empty() {
        return cc;
    }
    if std::env::var(ACKNOWLEDGE_ENV).as_deref() == Ok("skip") {
        eprintln!("{ACKNOWLEDGE_ENV}=skip: `{what}` did NOT run — {} missing.", absent.join(", "));
        return None;
    }
    panic!(
        "`{what}` cannot run here, and a guard that does not run is not a guard: {}.\n\
         Install `libva-dev` and `libdrm-dev`, or set {ACKNOWLEDGE_ENV}=skip to record in writing that this machine cannot check the declarations.",
        absent.join(", ")
    );
}

/// Compile `source` with `-fsyntax-only` and hand the raw result back, compiling or not.
///
/// The caller asserts on `status`, so a *failing* compile is a usable value rather than an error:
/// [`the_shape_width_and_signedness_check_rejects_a_declaration_that_is_wrong`] needs one.
fn compile_probe(cc: &str, stem: &str, source: &str) -> std::process::Output {
    let dir = std::env::temp_dir().join(format!("filmcraft-{stem}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{cc}: cannot create {}: {e}", dir.display()));
    let path = dir.join("probe.c");
    std::fs::write(&path, source.as_bytes()).unwrap_or_else(|e| panic!("{cc}: cannot write {}: {e}", path.display()));
    let out = std::process::Command::new(cc)
        .args(["-fsyntax-only", path.to_str().unwrap_or_default()])
        .output()
        .unwrap_or_else(|e| panic!("running {cc} on {} failed: {e}", path.display()));
    let _ = std::fs::remove_dir_all(&dir);
    out
}
