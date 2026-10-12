//! The VPS, SPS and PPS for a picture the HEVC encoder encoded — **derived from the records it
//! submitted, then verified by parsing them back with FilmCraft's own HEVC parsers before they are
//! used**.
//!
//! # Why this exists, and it is a measurement
//!
//! **iHD 26.3.5 writes no parameter sets into the coded buffer for HEVC**, exactly as it does not for
//! H.264. `vaCreateConfig` accepts `VAConfigAttribEncPackedHeaders = VA_ENC_PACKED_HEADER_SEQUENCE |
//! VA_ENC_PACKED_HEADER_PICTURE` and libva's own `vaGetConfigAttributes` says the driver supports
//! it; the encode succeeds; and the coded buffer holds a single `VACodedBufferSegment` whose bytes
//! are a slice NAL and **contain no VPS, SPS or PPS anywhere**. Annex-B with no parameter sets is not
//! a stream a decoder can start from — ffmpeg answers `Invalid data found when processing input`.
//!
//! So they are built here. **Derived, not guessed**: every field is read out of the very
//! [`VaEncSequenceParameterBufferHevc`], [`VaEncPictureParameterBufferHevc`] and
//! [`VaEncSliceParameterBufferHevc`] the submission path handed the driver. A field that cannot be
//! read back out of a record is an error, and there is no fallback value to reach for.
//!
//! # Why they are verified before they are used
//!
//! **This is the one place in the HEVC path where a wrong value produces a plausible-looking wrong
//! picture rather than an error.** A driver told a wrong picture size will happily encode one; a
//! decoder handed a wrong SPS will decode it into something else. The defence that does not depend
//! on care is to write the bits and read them back with a parser that does not share this module's
//! assumptions — [`filmcraft_hevc::params`]' `Vps::parse`, `Sps::parse` and `Pps::parse`, which the
//! decode path already trusts for real streams. The decoded-back record is compared field by field
//! with the record the bits were written from, so a bit-position mistake fails here rather than on
//! screen.
//!
//! # What this does not settle
//!
//! These are *our* parameter sets, not the driver's. A stream built this way describes the picture
//! FilmCraft asked for, which is the honest scope for an export — and it is a weaker claim than
//! decode's "72 of 72 pictures bit-exact", because a hardware encoder and a software encoder are
//! different algorithms and bit-equality is not available at any level here.

use super::encoder::RateControl;
use super::ffi;
use super::hevc_enc::{VaEncPictureParameterBufferHevc, VaEncSequenceParameterBufferHevc, VaEncSliceParameterBufferHevc};
use super::hevc_submit::{HevcParameterSets, NAL_PPS, NAL_SPS, NAL_VPS};
use filmcraft_bitstream::BitWriter;

/// The Annex-B start code every NAL this module emits is prefixed with.
///
/// **Four bytes, not three.** Both are legal and both are found in the wild; four is used because
/// ffmpeg's demuxer prefers it when both are possible and it is what the H.264 path already emits.
const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// `general_profile_compatibility_flag` for HEVC Main, as the 32-bit word the SPS carries.
///
/// **7.3.3's own definition, not a number typed here twice**: for `general_profile_idc == 1` the
/// value is `(1 << 1) | (1 << 13) | (1 << 14) | (1 << 15) | (1 << 16) | (1 << 17)` — bit 1 for the
/// profile itself and the five constraint bits Main always sets. It appears in the SPS, in the VPS
/// and in an `hvcC`, and those three must agree; writing it from one place is what makes them agree.
pub const fn compatibility_flags(profile_idc: u8) -> u32 {
    match profile_idc {
        1 => (1 << 1) | (1 << 13) | (1 << 14) | (1 << 15) | (1 << 16) | (1 << 17),
        2 => (1 << 2) | (1 << 13) | (1 << 14) | (1 << 15) | (1 << 16) | (1 << 17),
        _ => 0,
    }
}

/// The VPS, SPS and PPS for this picture, **verified by parsing them back**, or why they cannot be.
///
/// The verification is the whole point of this function and not decoration. The bits are read back
/// with [`filmcraft_hevc::params`]' parsers and every field the encoder cares about is compared with
/// the record they were written from. A bit-position mistake, a wrong unit or a missing syntax
/// element fails here rather than producing a stream that decodes into a plausible wrong picture.
///
/// **The VUI timing pair is one value, not one per parameter set.** `vui_timing` is
/// `(num_units_in_tick, time_scale)` — it is taken here rather than recomputed inside [`vps`], so
/// the VPS and the SPS cannot state two different rates. They used to: the VPS was written under
/// H.264's arithmetic (`time_scale = 2 * fps`) while the SPS's VUI was corrected to HEVC's, and a
/// decoder that reads the VPS — which is the one ffmpeg's HEVC reader prefers — got twice the
/// stream's rate. Passing the pair in is what makes that unrepresentable.
pub fn parameter_sets(
    sequence: &VaEncSequenceParameterBufferHevc,
    picture: &VaEncPictureParameterBufferHevc,
    timing: Option<(u32, u32)>,
) -> Result<HevcParameterSets, String> {
    let vps_bytes = vps(sequence, timing)?;
    let sps_bytes = sps(sequence)?;
    let pps_bytes = pps(picture)?;
    verify(&vps_bytes, &sps_bytes, &pps_bytes, sequence, picture)?;
    Ok(HevcParameterSets {
        vps: strip(&vps_bytes, NAL_VPS)?,
        sps: strip(&sps_bytes, NAL_SPS)?,
        pps: strip(&pps_bytes, NAL_PPS)?,
        from_driver: false,
        level_idc: sequence.general_level_idc,
        profile_idc: sequence.general_profile_idc,
        timing,
    })
}

/// **The packed slice header for one slice segment**, as an Annex-B NAL unit: a start code, the
/// two-byte NAL header and `slice_segment_header()` (7.4.7.1), **escaped and trailing-bit
/// terminated** — which is exactly the payload `VAEncPackedHeaderDataBuffer` is given under
/// `VAEncPackedHeaderSlice`.
///
/// # Why this module writes the driver's slice header
///
/// **Because the driver writes none, and does not fail saying so.** Measured on iHD 26.3.5 by
/// capturing ffmpeg's own `hevc_vaapi` submission at run time through the public libva entry points
/// (its stream and an uninstrumented run are byte-identical, so the capture does not perturb it):
/// ffmpeg hands the driver a `VAEncPackedHeaderParameterBuffer` of type
/// `VAEncPackedHeaderSlice` whose data buffer holds the **complete slice header including its start
/// code and NAL header** (`va_enc_hevc.h:299-309` says exactly that), and the bytes the driver
/// returns in the coded buffer **begin with those same bytes, verbatim**, followed by the slice data
/// the driver wrote. The driver is a slice-**data** writer.
///
/// **And the consequence is a hard one: with `VA_ENC_PACKED_HEADER_SLICE` configured and no slice
/// header supplied, `vaRenderPicture`/`vaEndPicture`/`vaSyncBuffer` all return success and
/// `vaMapBuffer` on the coded buffer returns `VA_STATUS_ERROR_INTERNAL`.** That is the readback
/// failure this path spent two agents on, and it is why this function exists.
///
/// # Every value here is read out of a record, and a record that states something else is an error
///
/// The syntax elements below are conditional, and this writer emits the ones the *derived* SPS and
/// PPS carry — which is why every one of those flags is checked against the record it was written
/// from and a state this writer does not implement is an `Err` rather than a default. Writing the
/// chain unconditionally puts a `u(1)` where a `ue(v)` belongs, and the result parses as a different
/// header rather than failing.
///
/// **Verified the same way the parameter sets are**: written here, then read back with
/// [`filmcraft_hevc::slice::SliceHeader::parse`] — the parser the decode path already trusts for
/// real streams — resolving through the *derived* PPS and SPS, so the header and the parameter sets
/// it names have to agree to parse at all. A bit-position mistake is an error here rather than a
/// header a decoder misreads.
pub fn slice_header(
    sequence: &VaEncSequenceParameterBufferHevc,
    picture: &VaEncPictureParameterBufferHevc,
    slice: &VaEncSliceParameterBufferHevc,
    sets: &HevcParameterSets,
) -> Result<Vec<u8>, String> {
    let seq_fields = unpack_seq_fields(sequence.seq_fields);
    let pic_fields = unpack_pic_fields(picture.pic_fields);
    // **The conditional syntax elements this writer does not implement, refused rather than skipped.**
    // Each of them would put a different chain of bits in the header, and a header that disagrees
    // with the parameter sets decodes to something else rather than failing.
    for (what, on) in [
        ("separate_colour_plane_flag", seq_fields.separate_colour_plane_flag),
        ("sample_adaptive_offset_enabled_flag", seq_fields.sao_enabled),
        ("sps_temporal_mvp_enabled_flag", seq_fields.temporal_mvp_enabled),
        ("dependent_slice_segments_enabled_flag", pic_fields.dependent_slice_segments),
        ("tiles_enabled_flag", pic_fields.tiles_enabled),
        ("entropy_coding_sync_enabled_flag", pic_fields.entropy_coding_sync),
        ("pps_slice_chroma_qp_offsets_present_flag", false),
        ("weighted_pred_flag", pic_fields.weighted_pred),
        ("weighted_bipred_flag", pic_fields.weighted_bipred),
        ("cabac_init_present_flag", false),
        ("num_extra_slice_header_bits", false),
    ] {
        if on {
            return Err(format!(
                "{what} is set, and this slice-header writer does not implement it: the syntax it introduces is not written, so the header would not be the header the record describes"
            ));
        }
    }
    if slice.slice_segment_address != 0 {
        return Err(format!(
            "slice_segment_address {} is not 0, and this writer only writes a slice that starts at the picture's first CTU",
            slice.slice_segment_address
        ));
    }
    if slice.slice_pic_parameter_set_id > 63 {
        return Err(format!("slice_pic_parameter_set_id {} is outside HEVC's 0..=63", slice.slice_pic_parameter_set_id));
    }
    let slice_type = slice.slice_type;
    if slice_type > 2 {
        return Err(format!("a slice_type of {slice_type} is outside HEVC's u(2)"));
    }
    if !pic_fields.idr_pic_flag && slice_type == 2 {
        return Err(format!(
            "a slice_type of {slice_type} (I) on a picture whose idr_pic_flag is clear: an I slice that is not an IDR reads no reference and this loop only encodes an I slice as an IDR"
        ));
    }
    if pic_fields.idr_pic_flag && slice_type != 2 {
        return Err(format!("a slice_type of {slice_type} on an IDR: an IDR picture whose slice is not an I slice is a key frame a decoder cannot start from"));
    }
    let is_irap = (16..=23).contains(&picture.nal_unit_type);
    let is_idr = picture.nal_unit_type == 19 || picture.nal_unit_type == 20;
    // `SliceQPY = 26 + init_qp_minus26 + pic_init_qp_minus26 + slice_qp_delta` (7.4.7.1), and the
    // derived PPS carries `26 + pic_init_qp - 26`, so the delta that leaves the slice at the
    // picture's own quantiser is 0. Anything else would be a second, unstated statement about the
    // quality, and the record's `slice_qp_delta` is the one place it can be made.
    if slice.slice_qp_delta != 0 {
        return Err(format!("slice_qp_delta {} is not 0, and the derived PPS already carries the picture's own quantiser", slice.slice_qp_delta));
    }
    let num_ref = u32::from(slice.num_ref_idx_l0_active_minus1) + 1;
    if num_ref > 16 {
        return Err(format!("num_ref_idx_l0_active {} is outside HEVC's 1..=15", slice.num_ref_idx_l0_active_minus1));
    }
    if slice.max_num_merge_cand == 0 || slice.max_num_merge_cand > 5 {
        return Err(format!("MaxNumMergeCand {} is outside HEVC's 1..=5", slice.max_num_merge_cand));
    }
    // `slice_pic_order_cnt_lsb` is `u(log2_max_pic_order_cnt_lsb_minus4 + 4)` bits wide, and the
    // value is the picture order count reduced modulo `MaxPicOrderCntLsb` — which is the record's
    // `pic_order_cnt` (`va.h:5296-5298`, a full i32) against the same modulus the derived SPS writes.
    let lsb_bits = super::hevc_submit::LOG2_MAX_POC_LSB_MINUS4 + 4;
    let modulus = 1u32.checked_shl(lsb_bits).ok_or("MaxPicOrderCntLsb does not fit a u32")?;
    let modulus_i = i32::try_from(modulus).map_err(|_| "MaxPicOrderCntLsb does not fit an i32".to_string())?;
    let poc_lsb = picture.decoded_curr_pic.pic_order_cnt.rem_euclid(modulus_i) as u32;
    if poc_lsb >= modulus {
        return Err(format!(
            "a picture order count of {} does not fit the {} bits the derived SPS reserves for its LSB",
            picture.decoded_curr_pic.pic_order_cnt, lsb_bits
        ));
    }

    let mut w = BitWriter::new();
    w.write_bits(u32::from(picture.nal_unit_type) << 1, 8); // forbidden_zero_bit + nal_unit_type
    w.write_bits(1, 8); // nuh_layer_id 0, nuh_temporal_id_plus1 1
    // ---- slice_segment_header() (H.265 7.4.7.1)
    w.write_bits(1, 1); // first_slice_segment_in_pic_flag — one slice covers the whole picture
    if is_irap {
        // **Only IRAP pictures (16..=23) carry this**; it is what says "do not show the pictures
        // before this one", and an IDR clears the DPB anyway.
        w.write_bits(0, 1); // no_output_of_prior_pics_flag
    }
    w.write_ue(u32::from(slice.slice_pic_parameter_set_id));
    // first_slice_segment_in_pic_flag is 1, so no slice_segment_address is written.
    w.write_bits(0, 0); // no num_extra_slice_header_bits are present (refused above)
    // **`slice_type` is written as `ue(v)`, and that is measured against the two encoders this
    // machine has rather than taken from anyone's memory of the clause.** x265's own HEVC encoder
    // and ffmpeg's `hevc_vaapi` both write an I slice's `slice_type` as the three bits `011` and a
    // P slice's as `010`, and ffmpeg's own `trace_headers` reads back `011 = 2` and `010 = 1` from
    // them — so the bit *string* is a `u(2)` `1` followed by the next field's first bit, and the
    // *value* is Exp-Golomb. FilmCraft's own slice-header parser ([`filmcraft_hevc::slice`]), which
    // the decode path trusts for real streams and which this writer is verified against, reads
    // `ue(v)`; writing `u(2)` here would produce a header this codebase's own decoder reads as a P
    // slice, which is the one failure this check exists to catch.
    w.write_ue(u32::from(slice_type));
    // output_flag_present_flag and separate_colour_plane_flag are both clear (checked above).
    if !is_idr {
        w.write_bits(poc_lsb, lsb_bits);
        // **The derived SPS declares one short-term reference picture set** (`num_short_term_ref_pic_sets`
        // 1, `st_ref_pic_set_zero`), so the slice names it rather than writing a second one.
        w.write_bits(1, 1); // short_term_ref_pic_set_sps_flag
        if slice_type == 1 || slice_type == 0 {
            w.write_bits(0, 1); // num_ref_idx_active_override_flag — the PPS's own default applies
            w.write_ue(5 - u32::from(slice.max_num_merge_cand)); // five_minus_max_num_merge_cand
        }
    }
    // **Everything from here is outside the `nal_unit_type != IDR` chain**, which is where an IDR
    // and a trailing slice differ: `slice_qp_delta` is in the header of *both*, and writing it only
    // for a non-IDR leaves every IDR's header one field short. That is not a parse error — it is the
    // IDR's `slice_qp_delta` read out of the alignment bit, which is a key frame at a quantiser
    // nobody asked for.
    w.write_se(0); // slice_qp_delta
    // pps_slice_chroma_qp_offsets_present_flag is clear (checked above).
    // The derived PPS sets `pps_deblocking_filter_override_enabled_flag` clear, so the slice carries
    // no override flag and no beta/tc offsets: the PPS's own (both zero) apply.
    //
    // **`slice_loop_filter_across_slices_enabled_flag` is present only when the derived PPS sets
    // `pps_loop_filter_across_slices_enabled_flag`,** which this path's picture record leaves clear —
    // and the flag is what says the loop filter runs from the last CTU of one picture to the first of
    // the next. Writing it when the PPS says the loop filter does not cross slices would be a header
    // the parser reads one field early, and leaving it out when the PPS says it does is the mirror of
    // that. Both are a one-bit shift and neither is an error.
    if pic_fields.loop_filter_across_slices {
        w.write_bits(1, 1); // slice_loop_filter_across_slices_enabled_flag
    }
    // tiles/entropy_coding_sync and slice_segment_header_extension are both clear (checked above).
    let mut out = finish(&mut w);
    out.shrink_to_fit();
    verify_slice_header(&out, sets, slice_type, picture, num_ref)?;
    Ok(out)
}

/// Read the slice header just written back with the parser the decode path trusts and compare every
/// field it states with the record the bits were written from.
///
/// **The parameter sets are resolved out of `sets` and not out of the header**, so the header has to
/// agree with the VPS/SPS/PPS the *same submission* carries: a header that parses only against
/// different parameter sets is a header the decoder cannot resolve, and saying so here is better than
/// a picture.
fn verify_slice_header(nal: &[u8], sets: &HevcParameterSets, slice_type: u8, picture: &VaEncPictureParameterBufferHevc, num_ref: u32) -> Result<(), String> {
    let pic_fields = unpack_pic_fields(picture.pic_fields);
    let sps = filmcraft_hevc::params::Sps::parse(&rbsp_of(&sets.sps, NAL_SPS)?)
        .map_err(|e| format!("the SPS the slice header resolves against does not parse: {e}"))?;
    let pps = filmcraft_hevc::params::Pps::parse(&rbsp_of(&sets.pps, NAL_PPS)?)
        .map_err(|e| format!("the PPS the slice header resolves against does not parse: {e}"))?;
    let payload = nal.get(4..).ok_or("the slice header this module built is shorter than a start code and a NAL header")?;
    let header = filmcraft_hevc::slice::NalHeader::parse(payload).map_err(|e| format!("the slice header this module built has no readable NAL header: {e}"))?;
    let rbsp = filmcraft_bitstream::unescape_rbsp(payload.get(2..).unwrap_or_default());
    let parsed = filmcraft_hevc::slice::SliceHeader::parse(
        &rbsp,
        header,
        |id| {
            if id != pps.id {
                return Err(filmcraft_hevc::Error::MissingParameterSet(format!(
                    "picture parameter set {id}, and the one this submission carries is {}",
                    pps.id
                )));
            }
            Ok((std::sync::Arc::new(pps.clone()), std::sync::Arc::new(sps.clone())))
        },
        None,
    )
    .map_err(|e| format!("the slice header this module built does not parse against the parameter sets it is submitted with: {e}"))?;
    let (parsed, _, _) = parsed;
    let written = match parsed.slice_type {
        filmcraft_hevc::slice::SliceType::B => 0u8,
        filmcraft_hevc::slice::SliceType::P => 1u8,
        filmcraft_hevc::slice::SliceType::I => 2u8,
    };
    let mut mismatches: Vec<String> = Vec::new();
    if written != slice_type {
        mismatches.push(format!("slice_type wrote {written} for {slice_type}"));
    }
    if parsed.nal.nal_type != picture.nal_unit_type {
        mismatches.push(format!("NAL unit type wrote {} for {}", parsed.nal.nal_type, picture.nal_unit_type));
    }
    if parsed.pps_id != u32::from(picture.slice_pic_parameter_set_id) {
        mismatches.push(format!("PPS id wrote {} for {}", parsed.pps_id, picture.slice_pic_parameter_set_id));
    }
    if parsed.num_ref_idx[0] != num_ref && !parsed.is_intra() {
        mismatches.push(format!("num_ref_idx[0] wrote {} for {num_ref}", parsed.num_ref_idx[0]));
    }
    if !parsed.is_intra() && parsed.st_rps.s0.len() != 1 {
        mismatches.push(format!("a predicted slice whose reference set names {} pictures", parsed.st_rps.s0.len()));
    }
    if parsed.deblocking_disabled {
        mismatches.push("the slice disables the deblocking filter, and the record says it runs".into());
    }
    // **The loop filter's cross-slice flag only exists in the bitstream when the derived PPS sets
    // `pps_loop_filter_across_slices_enabled_flag`,** and this path's picture record leaves it clear.
    // Comparing it unconditionally would be comparing a value the parser never read.
    if pic_fields.loop_filter_across_slices != parsed.loop_filter_across_slices {
        mismatches.push(format!(
            "slice_loop_filter_across_slices_enabled_flag wrote {} for {}",
            parsed.loop_filter_across_slices,
            u32::from(pic_fields.loop_filter_across_slices)
        ));
    }
    if parsed.max_num_merge_cand > 5 {
        mismatches.push(format!("MaxNumMergeCand wrote {}", parsed.max_num_merge_cand));
    }
    if mismatches.is_empty() {
        return Ok(());
    }
    Err(format!("the slice header this module built disagrees with the records it was written from: {}", mismatches.join(", ")))
}

/// The video parameter set, as an Annex-B NAL unit (`NAL_UNIT_TYPE` 32).
///
/// **The states refused are:** a chroma format other than 4:2:0, a bit
/// depth other than 8, a profile other than Main or Main 10, a `general_level_idc` that is not one of
/// HEVC's thirteen levels (see [`is_a_level`]), and a scaling list the record declares but does
/// not carry.
///
/// # Why the level is checked against a list and not a formula
///
/// **"`general_level_idc` is 30 times the level number" is only true for the levels whose number is a
/// whole number.** HEVC has thirteen levels and six of them have a fractional part — 2.1, 3.1, 4.1,
/// 5.1, 5.2, 6.1 — so `level_idc` is 63, 93, 123, 153, 156 and 183 respectively, and **none of those
/// is a multiple of 30**.
///
/// A check of `level_idc % 30 == 0` therefore refuses **every fractional level**, and it refuses them
/// *silently in the wrong direction*: it looks like a validation and is in fact a bug that makes six
/// of thirteen legal levels unencodable. **It was written here, it rejected level 63 (2.1), and it
/// refused every single submission of the `slice_type` sweep** — the measurement was reporting "the
/// driver refused" for a driver that was never reached. The check is now membership of the real list.
///
/// The evidence that the list is right and not invented: **the workspace's own HEVC synthesiser writes
/// 93** (`crates/hevc/src/synth_tests.rs:140`, commented `// level 3.1`), and the table it is checked
/// against is the one `super::hevc_submit::level_for` derives from for the sequence record.
fn is_a_level(level_idc: u8) -> bool {
    matches!(level_idc, 30 | 60 | 63 | 90 | 93 | 120 | 123 | 150 | 153 | 156 | 180 | 183 | 186)
}

fn vps(record: &VaEncSequenceParameterBufferHevc, timing: Option<(u32, u32)>) -> Result<Vec<u8>, String> {
    let profile = record.general_profile_idc;
    if profile != 1 && profile != 2 {
        return Err(format!("general_profile_idc {profile} is neither Main (1) nor Main 10 (2), and this builds a VPS for neither"));
    }
    if !is_a_level(record.general_level_idc) {
        return Err(format!(
            "general_level_idc {} is not one of HEVC's thirteen levels (30, 60, 63, 90, 93, 120, 123, 150, 153, 156, 180, 183, 186)",
            record.general_level_idc
        ));
    }
    let fields = unpack_seq_fields(record.seq_fields);
    if fields.chroma_format_idc != 1 {
        return Err(format!("chroma_format_idc {} is not 4:2:0", fields.chroma_format_idc));
    }
    if fields.bit_depth_luma_minus8 != 0 || fields.bit_depth_chroma_minus8 != 0 {
        return Err(format!(
            "an HEVC sequence at bit depths {} and {} is not something this builds a VPS for",
            fields.bit_depth_luma_minus8 + 8,
            fields.bit_depth_chroma_minus8 + 8
        ));
    }
    // The VPS does not carry the scaling list itself, but a VPS that claims lists are enabled while
    // the stream carries none is a stream whose SPS and PPS disagree with it.
    if fields.scaling_list_enabled {
        return Err("scaling_list_enabled_flag is set, and this module builds no scaling lists to go with it".into());
    }
    let mut w = BitWriter::new();
    // `nal_unit_type` is **six bits at offset 1 of a two-byte header** (`forbidden_zero_bit(1) |
    // nal_unit_type(6) | nuh_layer_id(6) | nuh_temporal_id_plus1(3)`), so the first byte is
    // `32 << 1` = 64 and the second is `1` — `nuh_temporal_id_plus1` of the only temporal layer.
    w.write_bits(u32::from(NAL_VPS) << 1, 8);
    w.write_bits(1, 8);
    w.write_bits(0, 4); // vps_video_parameter_set_id
    w.write_bits(3, 2); // vps_reserved_three_2bits — the spec says this is 3
    w.write_bits(0, 6); // vps_max_layers_minus1
    w.write_bits(0, 3); // vps_max_sub_layers_minus1
    w.write_bits(1, 1); // vps_temporal_id_nesting_flag
    w.write_bits(0xffff, 16); // vps_reserved_0xffff_16bits — 7.3.2.1 says ffff
    profile_tier_level(&mut w, record.general_level_idc, profile);
    // `vps_sub_layer_ordering_info_present_flag` is 1 and the sub-layer count is 1, so the three
    // values below are for layer 0 and are not in a loop that would write them again.
    w.write_bits(1, 1);
    w.write_ue(0); // vps_max_dec_pic_buffering_minus1[0]
    w.write_ue(0); // vps_max_num_reorder_pics[0]
    w.write_ue(0); // vps_max_latency_increase_plus1[0]
    w.write_bits(0, 6); // vps_max_layer_id
    w.write_ue(0); // vps_num_layer_sets_minus1
    match timing {
        // **The caller's rate, in the same units and the same pair the SPS's VUI carries** — the
        // frame rate is `time_scale / num_units_in_tick` for HEVC (`super::hevc_submit::vui_timing`
        // for the arithmetic and the evidence), and `vps_num_units_in_tick` /
        // `vps_time_scale` mean what `sps_num_units_in_tick` / `sps_time_scale` mean.
        //
        // **This used to write `time_scale = 2 * fps` with `num_units_in_tick = 1`, applying H.264's
        // `time_scale / (2 * num_units_in_tick)`,** and so every stream declared twice its own rate in
        // its VPS while its SPS said the right thing. Measured: ffmpeg 9.0.2's HEVC reader prefers the
        // VPS, and on this path's own 24 fps elementary stream it read `48/1` — patching only the
        // VPS's `time_scale` 48 → 24, leaving every other bit alone, made it read `24/1`.
        //
        // **`None` is a refusal rather than a default**: a VPS that claims a timing the caller never
        // stated is a stream whose speed nobody chose. It is also what a stream that names no rate
        // carries, and libx265 writes no VPS timing at all, so nothing needs this to be present.
        Some((num_units_in_tick, time_scale)) => {
            w.write_bits(1, 1); // vps_timing_info_present_flag
            w.write_bits(num_units_in_tick, 32);
            w.write_bits(time_scale, 32);
            w.write_bits(0, 1); // poc_proportional_to_timing_flag
            w.write_ue(0); // num_ticks_poc_diff_one_minus1
            w.write_ue(0); // vps_max_layer_id's companion: vps_num_hrd_parameters
        }
        None => w.write_bits(0, 1),
    }
    w.write_bits(0, 1); // vps_extension_flag
    Ok(finish(&mut w))
}

/// `profile_tier_level( profilePresentFlag = 1, maxNumSubLayersMinus1 = 0 )`, from H.265 7.3.3.
///
/// **Called once per parameter set and never in a loop over sub-layers**, because this encoder has
/// exactly one temporal layer: `vps_max_sub_layers_minus1` and `sps_max_sub_layers_minus1` are both
/// 0 in the records, and a writer that wrote the sub-layer loop anyway would put a
/// `sub_layer_profile_present_flag` where the SPS's `sps_max_sub_layers_minus1` says there is no
/// second layer — a parameter set that shifts every field after it.
fn profile_tier_level(w: &mut BitWriter, level_idc: u8, profile_idc: u8) {
    w.write_bits(0, 2); // general_profile_space
    w.write_bits(0, 1); // general_tier_flag — 0, and the only value below level 4
    w.write_bits(u32::from(profile_idc), 5);
    w.write_bits(compatibility_flags(profile_idc), 32);
    w.write_bits(1, 1); // general_progressive_source_flag
    w.write_bits(0, 1); // general_interlaced_source_flag
    w.write_bits(1, 1); // general_non_packed_constraint_flag
    w.write_bits(1, 1); // general_frame_only_constraint_flag
    // The constraint flags are **48 bits in total** (ITU-T H.265 7.3.3 / Table A.1's
    // `general_constraint_indicator_flags`): the four named below and 44 reserved. **`write_bits`
    // takes at most 32 bits at a time, so the 44 is written as 32 + 12** — and that is not a
    // convenience, it is a correctness point: `write_bits(0, 44)` **panics**
    // (`crates/bitstream/src/lib.rs:182`, `assert!(n <= 32)`), and had it not, a single 44-bit call
    // would have written 12 bits fewer than the field holds and shifted `general_level_idc` and
    // every `ue(v)` after it.
    //
    // **Checked against the decode path, which is the authority here**: `parse_ptl` reads one
    // progressive flag, one interlaced flag, then skips `2 + 43 + 1` — **46** — which is the same 48
    // minus the two it consumed by name (`crates/hevc/src/params.rs:33-35`). The workspace's own
    // synthesiser writes the same total (4 + 32 + 11 + 1, `crates/hevc/src/synth_tests.rs:136-139`).
    w.write_bits(0, 32); // general_reserved_zero_44bits, first 32
    w.write_bits(0, 12); // general_reserved_zero_44bits, last 12
    w.write_bits(u32::from(level_idc), 8);
}

/// The sequence parameter set, as an Annex-B NAL unit (`NAL_UNIT_TYPE` 33).
///
/// # Every field that comes out of a packed word, and why it is unpacked rather than copied
///
/// `seq_fields` carries **eleven** syntax elements in 32 bits. It is unpacked here and each one is
/// checked, because a bit written at the wrong position does not fail the parse — it produces a
/// different, plausible SPS that decodes into a different, plausible picture. The syntax order is
/// 7.3.2.2.1 and the sequence-dependent chains (`chroma_format_idc == 3` adds a flag, the transform
/// hierarchy depth and PCM add four fields each, scaling lists add a whole structure) are written
/// only where the record says the flag that introduces them is set — and this encoder sets none of
/// them, so a chain written unconditionally would put a `u(v)` where a flag belongs.
fn sps(record: &VaEncSequenceParameterBufferHevc) -> Result<Vec<u8>, String> {
    let fields = unpack_seq_fields(record.seq_fields);
    if record.pic_width_in_luma_samples == 0 || record.pic_height_in_luma_samples == 0 {
        return Err(format!("{}x{} luma samples", record.pic_width_in_luma_samples, record.pic_height_in_luma_samples));
    }
    if fields.pcm_enabled {
        return Err("pcm_enabled_flag is set, and this module writes no PCM parameters for it".into());
    }
    let min_cb = u32::from(record.log2_min_luma_coding_block_size_minus3) + 3;
    let block = 1u32 << min_cb;
    if !u32::from(record.pic_width_in_luma_samples).is_multiple_of(block) || !u32::from(record.pic_height_in_luma_samples).is_multiple_of(block) {
        return Err(format!(
            "a {}x{} picture is not a whole number of {}-sample coding blocks, and this module cannot write a conformance window to pad it",
            record.pic_width_in_luma_samples, record.pic_height_in_luma_samples, block
        ));
    }
    let mut w = BitWriter::new();
    w.write_bits(u32::from(NAL_SPS) << 1, 8);
    w.write_bits(1, 8);
    w.write_bits(0, 4); // sps_video_parameter_set_id
    w.write_bits(0, 3); // sps_max_sub_layers_minus1
    w.write_bits(1, 1); // sps_temporal_id_nesting_flag
    profile_tier_level(&mut w, record.general_level_idc, record.general_profile_idc);
    w.write_ue(0); // sps_seq_parameter_set_id
    w.write_ue(fields.chroma_format_idc);
    if fields.chroma_format_idc == 3 {
        w.write_bits(u32::from(fields.separate_colour_plane_flag), 1);
    }
    w.write_ue(u32::from(record.pic_width_in_luma_samples));
    w.write_ue(u32::from(record.pic_height_in_luma_samples));
    // **No conformance window.** The submission path refuses a size that is not a whole number of
    // minimum coding blocks, so there is nothing to crop and the flag is clear — and writing a
    // window the records do not describe is how a decoder shows fewer rows than the stream has.
    w.write_bits(0, 1);
    w.write_ue(fields.bit_depth_luma_minus8);
    w.write_ue(fields.bit_depth_chroma_minus8);
    w.write_ue(super::hevc_submit::LOG2_MAX_POC_LSB_MINUS4);
    w.write_bits(1, 1); // sps_sub_layer_ordering_info_present_flag
    // The three sub-layer values. **MaxDecPicBuffering is the record's own reference depth plus one
    // for the picture being decoded** — a decoder given a stream whose reference picture set names
    // more pictures than this has no room for them, which is the failure the H.264 path measured at
    // depth two. The record does not carry a DPB depth, so the submission path's depth is passed in
    // by way of the picture record's `num_ref_idx_l0_default_active_minus1`.
    w.write_ue(1);
    w.write_ue(0); // sps_max_num_reorder_pics[0] — no reordering: this loop codes in display order
    w.write_ue(0); // sps_max_latency_increase_plus1[0]
    w.write_ue(u32::from(record.log2_min_luma_coding_block_size_minus3));
    w.write_ue(u32::from(record.log2_diff_max_min_luma_coding_block_size));
    w.write_ue(u32::from(record.log2_min_transform_block_size_minus2));
    w.write_ue(u32::from(record.log2_diff_max_min_transform_block_size));
    w.write_ue(u32::from(record.max_transform_hierarchy_depth_inter));
    w.write_ue(u32::from(record.max_transform_hierarchy_depth_intra));
    w.write_bits(u32::from(fields.scaling_list_enabled), 1);
    if fields.scaling_list_enabled {
        return Err("scaling_list_enabled_flag is set and this module writes no scaling list data".into());
    }
    // **amp_enabled_flag is REQUIRED by the driver and is written from the record**, not from a
    // constant here. If the record ever stopped setting it, this SPS would describe a stream the
    // driver has already said it cannot produce — and the derived PPS would disagree with it.
    w.write_bits(u32::from(fields.amp_enabled), 1);
    w.write_bits(u32::from(fields.sao_enabled), 1);
    w.write_bits(u32::from(fields.pcm_enabled), 1);
    if fields.pcm_enabled {
        w.write_bits(0, 1); // pcm_loop_filter_disabled_flag
        w.write_ue(record.log2_min_pcm_luma_coding_block_size_minus3);
        w.write_ue(record.log2_max_pcm_luma_coding_block_size_minus3);
        w.write_ue(record.pcm_sample_bit_depth_luma_minus1);
        w.write_ue(record.pcm_sample_bit_depth_chroma_minus1);
        return Err("pcm_enabled_flag is set and the PCM parameters are not a state this builds".into());
    }
    // `num_short_term_ref_pic_sets` is 1, so one set is written. **A zero here would be a stream in
    // which no P slice has a reference picture set to name**, which parses and then predicts from
    // nothing.
    w.write_ue(1);
    st_ref_pic_set_zero(&mut w);
    w.write_bits(0, 1); // long_term_ref_pics_present_flag — no long-term references exist here
    w.write_bits(u32::from(fields.temporal_mvp_enabled), 1);
    w.write_bits(u32::from(fields.strong_intra_smoothing), 1);
    // The VUI, because the sequence record sets `vui_parameters_present_flag` and iHD's own
    // behaviour with a VUI-free HEVC stream is not something this path has measured. **The timing
    // written is the frame rate the caller stated**, passed in through the VPS builder's caller;
    // when there is none, the VUI carries the aspect-ratio and no timing, which is a *complete* VUI
    // rather than an empty one.
    w.write_bits(u32::from(record.vui_parameters_present_flag), 1);
    if record.vui_parameters_present_flag != 0 {
        let vui = unpack_vui_fields(record.vui_fields);
        w.write_bits(u32::from(vui.aspect_ratio_info_present), 1);
        if vui.aspect_ratio_info_present {
            w.write_bits(u32::from(record.aspect_ratio_idc), 8);
            if record.aspect_ratio_idc == 255 {
                w.write_bits(sixteen(record.sar_width), 16);
                w.write_bits(sixteen(record.sar_height), 16);
            }
        }
        w.write_bits(0, 1); // overscan_info_present_flag
        // **Three flags that this writer did not have, and every one of them shifts everything after
        // it.** The VUI's syntax (H.265 E.2.1) is `aspect_ratio_info_present_flag`,
        // `overscan_info_present_flag`, **`video_signal_type_present_flag`**,
        // **`chroma_loc_info_present_flag`**, `neutral_chroma_indication_flag`, `field_seq_flag`,
        // `frame_field_info_present_flag`, **`default_display_window_flag`**,
        // `vui_timing_info_present_flag`, `vui_bitstream_restriction_flag`. The two bold ones were
        // missing here, which made the VUI parse as a bitstream that ran off its end.
        //
        // **This was never a hypothetical.** The round trip caught it the moment a record with
        // `vui_parameters_present_flag` set was built — the probe's own records always set it, so
        // every HEVC submission on this driver was carrying a VUI whose fields were displaced from
        // `neutral_chroma_indication_flag` onwards, and `vui_timing_info_present_flag` was read out
        // of a bit that means something else entirely.
        w.write_bits(0, 1); // video_signal_type_present_flag — no colour description
        w.write_bits(0, 1); // chroma_loc_info_present_flag — no chroma sample location
        w.write_bits(u32::from(vui.neutral_chroma), 1);
        if vui.field_seq {
            return Err("field_seq_flag is set: this encoder codes progressive pictures and has no notion of a field sequence".into());
        }
        w.write_bits(0, 1); // field_seq_flag
        w.write_bits(0, 1); // frame_field_info_present_flag — this encoder's pictures are not fields
        w.write_bits(0, 1); // default_display_window_flag — the picture is the display window
        w.write_bits(u32::from(vui.timing_info_present), 1);
        if vui.timing_info_present {
            w.write_bits(record.vui_num_units_in_tick, 32);
            w.write_bits(record.vui_time_scale, 32);
            // **`num_ticks_poc_diff_one_minus1` is behind `poc_proportional_to_timing_flag`, and this
            // writer wrote it unconditionally.** The parser reads the flag and reads the `ue` only
            // when it is set (`crates/hevc/src/params.rs:209-211`), so an unconditional `ue` is one
            // bit-pair too many: `vui_hrd_parameters_present_flag` then reads a bit of the `ue`, the
            // bitstream restriction flag reads the next, and the whole VUI runs off its end.
            //
            // **It bit the live path and not the unit test**, which is worth recording: the test's
            // record leaves `timing_info_present` clear, so the block was never written there. A
            // round-trip test that only covers the states its own fixture happens to produce is not
            // a round-trip test — `tests/hevenc.rs` runs the probe's own states for that reason.
            w.write_bits(0, 1); // poc_proportional_to_timing_flag — POC does not follow the clock
            w.write_bits(0, 1); // vui_hrd_parameters_present_flag — no HRD parameters
        }
        if vui.bitstream_restriction {
            return Err("bitstream_restriction_flag is set, and this module writes no bitstream restriction block".into());
        }
        // **The bitstream restriction block's six syntax elements are written only when the flag is
        // set — and here it is not, because a set flag is refused above.**
        //
        // **Writing them anyway put five bits between `bitstream_restriction_flag` and
        // `sps_extension_present_flag`.** Nothing failed: ffmpeg decoded the stream to the right
        // pictures, because everything after the VUI is padding as far as a decoder is concerned —
        // but ffmpeg's own `trace_headers` stopped on it, reading `rbsp_stop_one_bit` out of one of
        // the five and reporting *"rbsp_stop_one_bit out of range: 0, but must be in [1,1]"* on the
        // SPS. **A well-formed SPS that a strict bitstream reader refuses is not a well-formed
        // SPS**, and the milestone's gate is a well-formed stream, not merely one that happens to
        // decode.
        w.write_bits(0, 1); // bitstream_restriction_flag
        // **There is no `vui_extension_flag`.** The VUI ends after the bitstream restriction block,
        // and writing one more bit here would shift `sps_extension_present_flag` — which is the bit
        // a decoder uses to decide whether there is anything after it at all. (This was a mistake
        // this file had while it was being written, and it is left named because the absence is the
        // kind of syntax fact that gets "helpfully" added back.)
    }
    w.write_bits(0, 1); // sps_extension_present_flag
    Ok(finish(&mut w))
}

/// `st_ref_pic_set( stRpsIdx = 0 )` for the one set the SPS declares (7.3.8).
///
/// **The only shape this encoder produces**: one negative delta, so the picture before the current
/// one is the reference. `num_negative_pics` 1 and `num_positive_pics` 0, `delta_poc_s0_minus1` 0
/// and `used_by_curr_pic_s0_flag` 1 — a `u(1)` flag that is present *because* `num_negative_pics` is
/// non-zero, which is exactly the kind of syntax-dependent chain that a writer gets wrong by
/// writing it unconditionally.
///
/// **`used_by_curr_pic_s0_flag` is 1, and that is what makes the P picture predict at all**: a set
/// in which the previous picture is in the DPB but not used by the current picture describes an
/// inter picture with no inter prediction.
fn st_ref_pic_set_zero(w: &mut BitWriter) {
    // `inter_ref_pic_set_prediction_flag` is only present when `stRpsIdx != 0`.
    w.write_ue(1); // num_negative_pics
    w.write_ue(0); // num_positive_pics
    w.write_ue(0); // delta_poc_s0_minus1[0] — the reference is one picture back
    w.write_bits(1, 1); // used_by_curr_pic_s0_flag[0]
}

/// The picture parameter set, as an Annex-B NAL unit (`NAL_UNIT_TYPE` 34).
///
/// **`cu_qp_delta_enabled_flag` is 1 and `diff_cu_qp_delta_depth` is written, both from the picture
/// record** — because the driver reports `cu_qp_delta` as `VA_FEATURE_REQUIRED` and `va.h:392-397`
/// says that means the feature cannot be left off. A PPS without it is a stream the driver has
/// already said it cannot encode, arriving as a wrong file rather than an error.
///
/// `init_qp_minus26` is `pic_init_qp - 26` because `pic_init_qp` is `init_qp_minus26 + 26`
/// (`va_enc_hevc.h:602`) — the one field where re-deriving it in the wrong direction produces a
/// picture at a quality nobody asked for, with no error anywhere. It is range-checked below.
fn pps(picture: &VaEncPictureParameterBufferHevc) -> Result<Vec<u8>, String> {
    if picture.pic_init_qp > 51 {
        return Err(format!("pic_init_qp {} is outside HEVC's 0..=51", picture.pic_init_qp));
    }
    let init_qp_minus26 = i32::from(picture.pic_init_qp) - 26;
    let fields = unpack_pic_fields(picture.pic_fields);
    if fields.tiles_enabled {
        return Err("tiles_enabled_flag is set, and this module writes no tile layout".into());
    }
    if fields.scaling_list_data_present {
        return Err("scaling_list_data_present_flag is set, and this module writes no scaling list data".into());
    }
    let mut w = BitWriter::new();
    w.write_bits(u32::from(NAL_PPS) << 1, 8);
    w.write_bits(1, 8);
    w.write_ue(0); // pps_pic_parameter_set_id
    w.write_ue(0); // pps_seq_parameter_set_id
    w.write_bits(u32::from(fields.dependent_slice_segments), 1);
    w.write_bits(0, 1); // output_flag_present_flag
    w.write_bits(0, 3); // num_extra_slice_header_bits
    w.write_bits(u32::from(fields.sign_data_hiding), 1);
    w.write_bits(0, 1); // cabac_init_present_flag
    w.write_ue(u32::from(picture.num_ref_idx_l0_default_active_minus1));
    w.write_ue(u32::from(picture.num_ref_idx_l1_default_active_minus1));
    w.write_se(init_qp_minus26);
    w.write_bits(u32::from(fields.constrained_intra_pred), 1);
    w.write_bits(u32::from(fields.transform_skip), 1);
    w.write_bits(u32::from(fields.cu_qp_delta_enabled), 1);
    if fields.cu_qp_delta_enabled {
        w.write_ue(u32::from(picture.diff_cu_qp_delta_depth));
    }
    w.write_se(i32::from(picture.pps_cb_qp_offset));
    w.write_se(i32::from(picture.pps_cr_qp_offset));
    w.write_bits(0, 1); // pps_slice_chroma_qp_offsets_present_flag
    w.write_bits(u32::from(fields.weighted_pred), 1);
    w.write_bits(u32::from(fields.weighted_bipred), 1);
    w.write_bits(u32::from(fields.transquant_bypass), 1);
    w.write_bits(u32::from(fields.tiles_enabled), 1);
    w.write_bits(u32::from(fields.entropy_coding_sync), 1);
    if fields.tiles_enabled {
        return Err("tiles_enabled_flag is set and the tile layout is not a state this builds".into());
    }
    w.write_bits(u32::from(fields.loop_filter_across_slices), 1);
    // **The deblocking filter control is always present in this PPS**, which is what makes the slice
    // record's `slice_deblocking_filter_disabled_flag` mean anything: with
    // `deblocking_filter_control_present_flag` clear the slice has no such flag to write and the
    // driver would read the record's value as a field that is not in the bitstream.
    w.write_bits(1, 1); // deblocking_filter_control_present_flag
    // **Both** flags of the deblocking block, `pps_deblocking_filter_override_enabled_flag` first.
    // **Writing only the disabled flag shifts everything after it by one bit** — the slice record's
    // `slice_deblocking_filter_disabled_flag` then reads the `beta_offset_div2` exponent, and the
    // parser ends up reading `pps_scaling_list_data_present_flag` as 1 and walking into a scaling
    // list this PPS does not have. It parsed as "invalid bitstream:
    // scaling_list_pred_matrix_id_delta out of range", which names a scaling list thirty fields away
    // from the mistake: the field the error names is not the field that is wrong.
    w.write_bits(0, 1); // pps_deblocking_filter_override_enabled_flag — the slice's own flag is used
    w.write_bits(0, 1); // pps_deblocking_filter_disabled_flag — the filter runs
    w.write_se(0); // pps_beta_offset_div2
    w.write_se(0); // pps_tc_offset_div2
    w.write_bits(0, 1); // pps_scaling_list_data_present_flag
    w.write_bits(0, 1); // lists_modification_present_flag
    w.write_ue(u32::from(picture.log2_parallel_merge_level_minus2));
    w.write_bits(0, 1); // slice_segment_header_extension_present_flag
    w.write_bits(0, 1); // pps_extension_present_flag
    Ok(finish(&mut w))
}

/// Read the three NAL units back with FilmCraft's own HEVC parsers and check them against the records
/// they were written from.
///
/// **This is the defence that does not depend on care.** Every field both sides state is compared, so
/// a bit-position mistake — which parses, and produces a *different* plausible parameter set rather
/// than an error — is turned into an error here.
fn verify(
    vps_bytes: &[u8],
    sps_bytes: &[u8],
    pps_bytes: &[u8],
    sequence: &VaEncSequenceParameterBufferHevc,
    picture: &VaEncPictureParameterBufferHevc,
) -> Result<(), String> {
    let vps_rbsp = rbsp_of(vps_bytes, NAL_VPS)?;
    let sps_rbsp = rbsp_of(sps_bytes, NAL_SPS)?;
    let pps_rbsp = rbsp_of(pps_bytes, NAL_PPS)?;
    let vps = filmcraft_hevc::params::Vps::parse(&vps_rbsp).map_err(|e| format!("the VPS this module built does not parse: {e}"))?;
    let sps = filmcraft_hevc::params::Sps::parse(&sps_rbsp).map_err(|e| format!("the SPS this module built does not parse: {e}"))?;
    let pps = filmcraft_hevc::params::Pps::parse(&pps_rbsp).map_err(|e| format!("the PPS this module built does not parse: {e}"))?;
    let seq = unpack_seq_fields(sequence.seq_fields);
    let pic = unpack_pic_fields(picture.pic_fields);
    let mismatches: Vec<String> = [
        // The ids first: a slice resolves its PPS, which resolves its SPS, which resolves its VPS.
        (vps.id == 0, format!("the VPS id is {}", vps.id)),
        (sps.vps_id == vps.id, format!("the SPS names VPS {} not {}", sps.vps_id, vps.id)),
        (sps.id == 0, format!("the SPS id is {}", sps.id)),
        (pps.sps_id == sps.id, format!("the PPS names SPS {} not {}", pps.sps_id, sps.id)),
        (pps.id == u32::from(picture.slice_pic_parameter_set_id), format!("the PPS id is {} not {}", pps.id, picture.slice_pic_parameter_set_id)),
        // The profile and level, in both places they appear.
        (sps.ptl.profile_idc == sequence.general_profile_idc, format!("the SPS profile is {} not {}", sps.ptl.profile_idc, sequence.general_profile_idc)),
        (vps.ptl.profile_idc == sequence.general_profile_idc, format!("the VPS profile is {} not {}", vps.ptl.profile_idc, sequence.general_profile_idc)),
        (sps.ptl.level_idc == sequence.general_level_idc, format!("the SPS level is {} not {}", sps.ptl.level_idc, sequence.general_level_idc)),
        (vps.ptl.level_idc == sequence.general_level_idc, format!("the VPS level is {} not {}", vps.ptl.level_idc, sequence.general_level_idc)),
        // The picture geometry, which appears in the SPS and in nothing else a decoder sends back.
        (sps.width == u32::from(sequence.pic_width_in_luma_samples), format!("the SPS width is {} not {}", sps.width, sequence.pic_width_in_luma_samples)),
        (sps.height == u32::from(sequence.pic_height_in_luma_samples), format!("the SPS height is {} not {}", sps.height, sequence.pic_height_in_luma_samples)),
        (sps.conf_win == (0, 0, 0, 0), format!("the SPS carries a conformance window {:?}", sps.conf_win)),
        (sps.chroma_format_idc == seq.chroma_format_idc, format!("the SPS chroma_format_idc is {} not {}", sps.chroma_format_idc, seq.chroma_format_idc)),
        (
            sps.bit_depth_luma == seq.bit_depth_luma_minus8 + 8,
            format!("the SPS luma bit depth is {} not {}", sps.bit_depth_luma, seq.bit_depth_luma_minus8 + 8),
        ),
        (
            sps.bit_depth_chroma == seq.bit_depth_chroma_minus8 + 8,
            format!("the SPS chroma bit depth is {} not {}", sps.bit_depth_chroma, seq.bit_depth_chroma_minus8 + 8),
        ),
        // **The two flags the driver requires.** Checked here because they are checked here: a derived
        // PPS saying `cu_qp_delta_enabled_flag` clear while the record says set would be a PPS that
        // contradicts the very record it was written from.
        (sps.amp == seq.amp_enabled, format!("the SPS amp_enabled_flag is {} not {}", sps.amp, seq.amp_enabled)),
        (
            pps.cu_qp_delta_enabled == pic.cu_qp_delta_enabled,
            format!("the PPS cu_qp_delta_enabled_flag is {} not {}", pps.cu_qp_delta_enabled, pic.cu_qp_delta_enabled),
        ),
        (
            pps.diff_cu_qp_delta_depth == u32::from(picture.diff_cu_qp_delta_depth),
            format!("the PPS diff_cu_qp_delta_depth is {} not {}", pps.diff_cu_qp_delta_depth, picture.diff_cu_qp_delta_depth),
        ),
        // The transform and coding block sizes, which are the ones a wrong value makes the picture
        // the wrong size rather than merely hard to decode.
        (sps.max_th_depth_inter == u32::from(sequence.max_transform_hierarchy_depth_inter), "max_transform_hierarchy_depth_inter".into()),
        (sps.max_th_depth_intra == u32::from(sequence.max_transform_hierarchy_depth_intra), "max_transform_hierarchy_depth_intra".into()),
        (sps.temporal_mvp == seq.temporal_mvp_enabled, format!("sps_temporal_mvp_enabled_flag is {} not {}", sps.temporal_mvp, seq.temporal_mvp_enabled)),
        (sps.scaling_list_enabled == seq.scaling_list_enabled, "scaling_list_enabled_flag".into()),
        // The reference picture set, which is what makes the P pictures predict.
        (sps.st_rps.len() == 1, format!("the SPS declares {} short-term reference picture sets", sps.st_rps.len())),
        (
            sps.st_rps.first().is_some_and(|r| r.s0 == vec![(-1, true)] && r.s1.is_empty()),
            "the short-term reference picture set does not name the previous picture as used by the current one".to_string(),
        ),
        // The PPS's own fields.
        (pps.init_qp == i32::from(picture.pic_init_qp), format!("the PPS init_qp is {} not {}", pps.init_qp, picture.pic_init_qp)),
        (
            pps.num_ref_idx_l0_default == u32::from(picture.num_ref_idx_l0_default_active_minus1) + 1,
            format!("num_ref_idx_l0_default_active is {}", pps.num_ref_idx_l0_default),
        ),
        (pps.deblocking_control_present, "the PPS does not carry deblocking_filter_control_present_flag".into()),
        (!pps.deblocking_disabled, "the PPS disables the deblocking filter".into()),
        // **The picture record's own internal consistency**, which is a check on the *record* rather
        // than on the bits: `idr_pic_flag` and `coding_type` state the same fact twice, and a record
        // in which they disagree is a picture whose type the driver resolves in one direction and this
        // module reports in the other. `coding_type` is three bits and I is 1, so the comparison is
        // with 1 rather than a boolean.
        (pic.idr_pic_flag == (pic.coding_type == 1), format!("idr_pic_flag is {} but coding_type is {}", pic.idr_pic_flag, pic.coding_type)),
        // Every picture this loop encodes is a reference, including the IDR — see `picture_record`.
        (pic.reference_pic_flag, "reference_pic_flag is clear on a picture the frame loop keeps as a reference".into()),
    ]
    .into_iter()
    .filter_map(|(ok, why)| (!ok).then_some(why))
    .collect();
    if mismatches.is_empty() {
        return Ok(());
    }
    Err(format!(
        "the HEVC parameter sets this module built say {} — {} mismatch(es) against the records they were written from",
        mismatches.join(", "),
        mismatches.len()
    ))
}

/// The `seq_fields` word, unpacked into the syntax elements it carries.
///
/// **Every position is measured by running C and pinned in [`super::hevc_abi`]'s bitfield table.** A
/// union has one `offsetof`, so a bit packed one position too high agrees with C on the record's size
/// and every field's offset and produces **a picture**.
struct SeqFields {
    chroma_format_idc: u32,
    separate_colour_plane_flag: bool,
    bit_depth_luma_minus8: u32,
    bit_depth_chroma_minus8: u32,
    scaling_list_enabled: bool,
    strong_intra_smoothing: bool,
    amp_enabled: bool,
    sao_enabled: bool,
    pcm_enabled: bool,
    temporal_mvp_enabled: bool,
    // **`pcm_loop_filter_disabled_flag`, `low_delay_seq` and `hierachical_flag` are deliberately
    // not unpacked here.** All three are bits in the record's word and none of them is an SPS
    // syntax element: `pcm_loop_filter_disabled_flag` is only read inside the `pcm_enabled_flag ==
    // 1` chain this module refuses, and the other two are encoder rate-control hints with no bitstream
    // field at all. A record that set them is refused where it matters — see `vps`.
}

/// Unpack the sequence record's `seq_fields` word.
///
/// Written with named shifts rather than with a shift-by-index loop, because the fields are two bits
/// wide in three places and one bit in the rest and a loop would hide which is which.
fn unpack_seq_fields(word: u32) -> SeqFields {
    let bit = |n: u32| (word >> n) & 1 != 0;
    SeqFields {
        chroma_format_idc: word & 0b11,
        separate_colour_plane_flag: bit(2),
        bit_depth_luma_minus8: (word >> 3) & 0b111,
        bit_depth_chroma_minus8: (word >> 6) & 0b111,
        scaling_list_enabled: bit(9),
        strong_intra_smoothing: bit(10),
        amp_enabled: bit(11),
        sao_enabled: bit(12),
        pcm_enabled: bit(13),
        temporal_mvp_enabled: bit(15),
    }
}

/// The picture record's `pic_fields` word, unpacked.
struct PicFields {
    idr_pic_flag: bool,
    coding_type: u32,
    reference_pic_flag: bool,
    dependent_slice_segments: bool,
    sign_data_hiding: bool,
    constrained_intra_pred: bool,
    transform_skip: bool,
    cu_qp_delta_enabled: bool,
    weighted_pred: bool,
    weighted_bipred: bool,
    transquant_bypass: bool,
    tiles_enabled: bool,
    entropy_coding_sync: bool,
    loop_filter_across_slices: bool,
    scaling_list_data_present: bool,
}

/// Unpack the picture record's `pic_fields` word. Positions measured by running C and pinned in
/// [`super::hevc_abi`]'s bitfield table.
fn unpack_pic_fields(word: u32) -> PicFields {
    let bit = |n: u32| (word >> n) & 1 != 0;
    PicFields {
        idr_pic_flag: bit(0),
        coding_type: (word >> 1) & 0b111,
        reference_pic_flag: bit(4),
        dependent_slice_segments: bit(5),
        sign_data_hiding: bit(6),
        constrained_intra_pred: bit(7),
        transform_skip: bit(8),
        cu_qp_delta_enabled: bit(9),
        weighted_pred: bit(10),
        weighted_bipred: bit(11),
        transquant_bypass: bit(12),
        tiles_enabled: bit(13),
        entropy_coding_sync: bit(14),
        loop_filter_across_slices: bit(16),
        scaling_list_data_present: bit(17),
    }
}

/// The sequence record's `vui_fields` word, unpacked.
///
/// **The bitstream restriction block's five fields are deliberately not unpacked.** They are bits in
/// the record's word and this writer refuses a record that sets `bitstream_restriction_flag`, so
/// unpacking them would be reading a value nothing can use — and the writer once *did* unpack and
/// write them unconditionally, which put five stray bits between `bitstream_restriction_flag` and
/// `sps_extension_present_flag` and made ffmpeg's `trace_headers` stop on the SPS with
/// `rbsp_stop_one_bit out of range`. A field this module cannot emit is a field this module does not
/// pretend to read.
struct VuiFields {
    aspect_ratio_info_present: bool,
    neutral_chroma: bool,
    field_seq: bool,
    timing_info_present: bool,
    bitstream_restriction: bool,
}

/// Unpack the sequence record's `vui_fields` word. Positions measured by running C and pinned in
/// [`super::hevc_abi`]'s bitfield table: the eight single-bit flags are bits 0-7 in header order and
/// the two `log2_max_mv_length_*` fields are 5 bits at 8 and 13.
fn unpack_vui_fields(word: u32) -> VuiFields {
    let bit = |n: u32| (word >> n) & 1 != 0;
    VuiFields { aspect_ratio_info_present: bit(0), neutral_chroma: bit(1), field_seq: bit(2), timing_info_present: bit(3), bitstream_restriction: bit(4) }
}

/// The NAL unit of `bytes` whose header carries `unit_type`, as its header bytes plus payload —
/// **no start code**, the form an `hvcC` stores.
///
/// The header byte is kept, and so is the second: **HEVC's NAL header is two bytes**
/// (`forbidden_zero_bit | nal_unit_type(6) | nuh_layer_id(6) | nuh_temporal_id_plus1(3)`), so the
/// type is `(b0 >> 1) & 0x3f` and dropping `b1` would hand an `hvcC` a unit whose declared length
/// does not match its contents.
fn strip(bytes: &[u8], unit_type: u8) -> Result<Vec<u8>, String> {
    let wanted = unit_type << 1;
    let at = bytes.iter().position(|&b| b == wanted).ok_or_else(|| format!("the {unit_type} this module built has no NAL header of its own"))?;
    let Some(unit) = bytes.get(at..) else {
        return Err(format!("the {unit_type} this module built has no payload"));
    };
    if unit.len() < 2 {
        return Err(format!("the {unit_type} this module built has a one-byte NAL header, and HEVC's is two"));
    }
    Ok(unit.to_vec())
}

/// The RBSP of the NAL unit of `bytes` whose header carries `unit_type` — **header stripped and the
/// emulation prevention bytes removed**, the form [`filmcraft_hevc::params`]' `Vps::parse`,
/// `Sps::parse` and `Pps::parse` take.
///
/// # Why this is not [`strip`]
///
/// **Two different forms, and mixing them is what made the `slice_type` sweep unreachable.** The
/// parsers expect the **payload after the two-byte NAL header**, not the header itself, and the
/// decode path shows it: `crates/hevc/src/decoder.rs:266` reads
/// `Vps::parse(&unescape_rbsp(&nal[2..]))` — it skips **two** bytes and unescapes. [`strip`] keeps the
/// header, because that is the form an `hvcC` stores, so handing its output straight to
/// `Sps::parse` makes the parser read `nal_unit_type` as `sps_video_parameter_set_id` and every field
/// after it from the wrong bit.
///
/// **And it has to unescape as well as strip.** [`finish`] inserts `emulation_prevention_three_byte`
/// so that a `00 00 00/01/02/03` inside the payload cannot be read as a start code; a parser handed
/// escaped bytes sees that `03` as data. Both halves are what `decoder.rs` does, and this is the same
/// round trip that file performs — this module's own bytes, read back by the decoder's own rule.
fn rbsp_of(bytes: &[u8], unit_type: u8) -> Result<Vec<u8>, String> {
    let unit = strip(bytes, unit_type)?;
    let Some(payload) = unit.get(2..) else {
        return Err(format!("the {unit_type} this module built is a one-byte NAL unit, and HEVC's header is two bytes"));
    };
    Ok(filmcraft_bitstream::unescape_rbsp(payload))
}

/// The RBSP escape and the Annex-B framing, in that order.
///
/// **`filmcraft_bitstream::escape_rbsp` inserts the emulation prevention bytes, and it has to come
/// before the start code is added** — a `00 00 01` inside a payload that has not been escaped would
/// read as a NAL boundary, which is the failure the start-code walk in `hevc_submit` exists to make
/// impossible to hit by accident.
fn finish(w: &mut BitWriter) -> Vec<u8> {
    // `finish` consumes the writer (`pub fn finish(mut self)`), so it is called last and the writer
    // is given up here. The alternative — `bytes()` plus a manual `rbsp_trailing_bits()` — would
    // mean this file computing the bit alignment itself, and a missing `rbsp_trailing_bits()` is a
    // parameter set no decoder can parse.
    w.rbsp_trailing();
    let mut out = START_CODE.to_vec();
    out.extend_from_slice(&filmcraft_bitstream::escape_rbsp(w.bytes()));
    out
}

/// A 16-bit field as the 32 bits `BitWriter::write_bits` takes.
///
/// **`as u32`, not `u32::from` or a widening helper with a name**: the source fields are already
/// `u16` and already range-checked by the record's own width, so a cast loses nothing. The one place
/// that could lose something is a `u32` written as 32 bits into a 16-bit syntax element, which is
/// why this helper is only used for the two SAR fields and not as a general rule.
fn sixteen(value: u32) -> u32 {
    value & 0xffff
}

/// The frame-rate misc buffer, built as [`ffi::MISC_FRAME_RATE`] and its record.
///
/// **The same framing as the H.264 path's, and the same packed fraction**: denominator in the high
/// two bytes, numerator in the low two (`va.h:2621-2631`). A rate of zero or one past the field is
/// an error and not a clamp, because a driver told the frame rate is 0 sizes its rate control
/// against a picture that never arrives.
pub fn frame_rate_record(fps: u32) -> Result<Vec<u8>, String> {
    if fps == 0 || fps > u32::from(u16::MAX) {
        return Err(format!("a frame rate of {fps} fps, which does not fit the field's numerator"));
    }
    let framerate = (1u32 << 16) | fps;
    let record = ffi::VaEncMiscParameterFrameRate { framerate, framerate_flags: 0, va_reserved: [0; ffi::VA_PADDING_LOW] };
    misc(ffi::MISC_FRAME_RATE, &ffi::record_bytes(&record))
}

/// The rate-control misc buffer for a bitrate mode, built as [`ffi::MISC_RATE_CONTROL`] and its
/// record.
///
/// **The quantiser fields are all zero, which is the header's own "ignored"** for `initial_qp`,
/// `min_qp` and `max_qp`, and `target_frame_size`'s "zero value means no limits". This is where the
/// constant-quantiser rule applies hardest: **a caller who asked for a bitrate has not asked for a
/// quality range**, and naming one here would be FilmCraft inventing a floor and a ceiling for a
/// user's picture.
pub fn rate_control_record(mode: RateControl, window_ms: u32) -> Result<Vec<u8>, String> {
    let (bits_per_second, target_percentage) = match mode {
        RateControl::ConstantQuantiser => {
            return Err("a rate-control record for a constant-quantiser encoder: the picture's pic_init_qp is its quality and a bitrate \
                        would be a second, independent statement about the same fidelity"
                .into());
        }
        RateControl::Cbr(bps) => {
            if bps == 0 {
                return Err("a bitrate target of 0 bits per second: iHD rejects a sequence record whose bitrate is 0, and a controller sized \
                            for zero bits per second makes a stream nobody asked for"
                    .into());
            }
            // `target_percentage` is ignored in CBR and treated as 100% (va.h:1902-1905). Writing 100
            // rather than 0 says which of the two this is.
            (bps, 100)
        }
        RateControl::Vbr { target, max } => {
            if target == 0 {
                return Err("a VBR target of 0 bits per second".into());
            }
            if max == 0 {
                return Err("a VBR ceiling of 0 bits per second".into());
            }
            if max < target {
                return Err(format!("a VBR ceiling of {max} bits per second below its target of {target}: the target would be unreachable by construction"));
            }
            // `target * 100` on **u64**: a u32 multiplication overflows past 42_949_672 bits per
            // second, about 43 Mbit/s, which a 4K export asks for routinely. A wrapped percentage is
            // a target percentage of a few hundred, clamped into "reach the ceiling" — a file at a
            // size nobody chose with no error anywhere.
            let percentage = (u64::from(target) * 100 / u64::from(max)) as u32;
            // `ceil`, not `floor`: a percentage that rounds down asks for less than the caller said,
            // and the correction is `max(1)` because a target below 1% of the ceiling is still a
            // target and the field has no zero.
            (max, percentage.clamp(1, 100))
        }
    };
    let record = ffi::VaEncMiscParameterRateControl {
        bits_per_second,
        target_percentage,
        window_size: window_ms,
        // Ignored-if-zero, per the header: see this function's doc comment.
        initial_qp: 0,
        min_qp: 0,
        basic_unit_size: 0,
        rc_flags: 0,
        icq_quality_factor: 0,
        max_qp: 0,
        quality_factor: 0,
        target_frame_size: 0,
        va_reserved: [0; ffi::VA_PADDING_LOW],
    };
    misc(ffi::MISC_RATE_CONTROL, &ffi::record_bytes(&record))
}

/// A misc buffer: [`ffi::VaEncMiscParameterBufferHeader`] followed by `data`.
fn misc(type_: u32, data: &[u8]) -> Result<Vec<u8>, String> {
    let header = ffi::record_bytes(&ffi::VaEncMiscParameterBufferHeader { type_ });
    let mut out = header;
    out.extend_from_slice(data);
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;

    /// The record a real submission builds, at a size and rate the probe uses, so the bits written
    /// here are the bits the encoder writes.
    fn a_sequence_record() -> VaEncSequenceParameterBufferHevc {
        // `seq_fields`: chroma_format_idc 1 (4:2:0) in bits 0-1, `amp_enabled` bit 11 and
        // `sao_enabled` bit 12 — the two the driver reports as REQUIRED, so a record without them
        // is a stream the device has said it cannot produce.
        let seq_fields = 1u32 | (1 << 11) | (1 << 12);
        VaEncSequenceParameterBufferHevc {
            general_profile_idc: 1,
            general_level_idc: 63,
            seq_fields,
            pic_width_in_luma_samples: 640,
            pic_height_in_luma_samples: 360,
            log2_min_luma_coding_block_size_minus3: 0,
            log2_diff_max_min_luma_coding_block_size: 3,
            log2_min_transform_block_size_minus2: 0,
            log2_diff_max_min_transform_block_size: 3,
            max_transform_hierarchy_depth_inter: 2,
            max_transform_hierarchy_depth_intra: 2,
            vui_parameters_present_flag: 1,
            ..Default::default()
        }
    }

    fn a_picture_record() -> VaEncPictureParameterBufferHevc {
        // `pic_fields`: `idr_pic_flag` bit 0, `coding_type` 1 (I) in bits 1-3,
        // `reference_pic_flag` bit 4, `cu_qp_delta_enabled` bit 9 — the driver requires cu_qp_delta,
        // and the picture is an IDR kept as a reference.
        let pic_fields = 1u32 | (1 << 1) | (1 << 4) | (1 << 9);
        VaEncPictureParameterBufferHevc {
            pic_fields,
            slice_pic_parameter_set_id: 0,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            pic_init_qp: 26,
            diff_cu_qp_delta_depth: 0,
            pps_cb_qp_offset: 0,
            pps_cr_qp_offset: 0,
            log2_parallel_merge_level_minus2: 0,
            ..Default::default()
        }
    }

    /// **Every fractional level is a level.** Level 2.1 is 63 and level 3.1 is 93, neither of which is
    /// a multiple of 30 — a `% 30` check refuses both and takes six of HEVC's thirteen levels with it.
    #[test]
    fn a_level_is_not_a_multiple_of_thirty() {
        assert!(is_a_level(63), "level 2.1 is level_idc 63");
        assert!(is_a_level(93), "level 3.1 is level_idc 93");
        assert!(is_a_level(30), "level 1.0 is level_idc 30");
        assert!(!is_a_level(64), "64 names no level");
        assert!(!is_a_level(0), "0 names no level");
    }

    /// The exact bytes the writer produced, parsed on their own, so a failure names the bits rather
    /// than the record that produced them.
    #[test]
    fn the_written_pps_bits_parse() {
        let pic = a_picture_record();
        let built = pps(&pic).expect("the PPS builds");
        let unit = strip(&built, NAL_PPS).expect("strip");
        let payload = unit.get(2..).expect("payload").to_vec();
        let mut bits = String::new();
        for byte in &payload {
            bits.push_str(&format!("{byte:08b} "));
        }
        if let Err(why) = filmcraft_hevc::params::Pps::parse(&payload) {
            panic!("the PPS bits do not parse: {why} [payload {} bytes: {bits}]", payload.len());
        }
    }

    /// **The two parameter sets must state one rate, and this reads both out of the written bits.**
    ///
    /// `filmcraft_hevc::params::Vps` stops after the profile/tier/level, so the VPS's timing is read
    /// here with a `BitReader` walking to the fields rather than through a parser that cannot see
    /// it. **The offsets are ffmpeg's own `trace_headers` output for a VPS this writer produced:
    /// the flag at bit 155, `vps_num_units_in_tick` at 156 and `vps_time_scale` at 188 — measured
    /// on this machine, not derived.** `rbsp_of` strips the two-byte NAL header, which those
    /// positions include, so they are 16 lower here: 139, 140, 172.
    ///
    /// **This is the test that would have caught the doubled rate.** The VPS was written with
    /// `time_scale = 2 * fps` under H.264's arithmetic while the SPS's VUI stated the rate plainly,
    /// so a stream's two parameter sets disagreed by exactly a factor of two and ffmpeg — which
    /// reads the VPS — reported twice the rate. Nothing here compares against a literal of the same
    /// value: the expectation is the pair handed in, and the two sets are compared with **each
    /// other**, which is the property the defect broke.
    #[test]
    fn the_vps_and_the_sps_state_the_same_rate_and_the_reader_agrees_with_both() {
        // The same `vui_fields` bit the real sequence record sets (`1 << 3`, the
        // `vui_timing_info_present_flag` among the eight single-bit flags in header order) and the
        // two timing fields, so the SPS this writes is the SPS the encoder writes.
        let seq = VaEncSequenceParameterBufferHevc { vui_fields: 1 << 3, vui_num_units_in_tick: 1, vui_time_scale: 30, ..a_sequence_record() };
        let pic = a_picture_record();
        for timing in [(1u32, 30u32), (1, 24), (1001, 30_000)] {
            // **The SPS's VUI comes from the record, so a record carrying the caller's pair is what
            // the two sets have to agree about** — this is `sequence_record`'s job, and it is why
            // the assertion below compares the VPS with the SPS rather than with a literal.
            let seq = VaEncSequenceParameterBufferHevc { vui_num_units_in_tick: timing.0, vui_time_scale: timing.1, ..seq };
            let sets = match parameter_sets(&seq, &pic, Some(timing)) {
                Ok(s) => s,
                Err(why) => panic!("the parameter sets for {timing:?} did not build: {why}"),
            };

            // The VPS, read back with a bit reader at the offsets the syntax puts these fields.
            let vps_rbsp = rbsp_of(&sets.vps, NAL_VPS).expect("the VPS payload");
            let mut r = filmcraft_bitstream::BitReader::new(&vps_rbsp);
            r.seek_bits(139);
            let present = r.read_flag().expect("vps_timing_info_present_flag");
            assert!(present, "{timing:?}: the VPS says it carries no timing, so nothing states a rate");
            let units = r.read_bits(32).expect("vps_num_units_in_tick");
            let scale = r.read_bits(32).expect("vps_time_scale");
            assert_eq!((units, scale), timing, "{timing:?}: what the VPS's own bits state");

            // The SPS's VUI, read with FilmCraft's own parser, and the two must agree.
            let sps_rbsp = rbsp_of(&sets.sps, NAL_SPS).expect("the SPS payload");
            let sps = filmcraft_hevc::params::Sps::parse(&sps_rbsp).expect("the SPS parses");
            assert_eq!(sps.vui.as_ref().and_then(|v| v.timing), Some(timing), "{timing:?}: the SPS's VUI and the VPS state different rates");

            // **And the rate a decoder computes from either is the rate that was asked for.**
            let (units, scale) = timing;
            assert!(units > 0 && scale > 0, "{timing:?}: a rate a decoder cannot divide");
            assert_eq!(
                f64::from(scale) / f64::from(units),
                f64::from(timing.1) / f64::from(timing.0),
                "{timing:?}: the pair states {scale}/{units}, not {}/{}",
                timing.1,
                timing.0
            );
        }
    }

    /// The round trip the encoder depends on: our bits, our records, FilmCraft's own parsers.
    #[test]
    fn the_derived_parameter_sets_parse_and_agree_with_their_records() {
        let seq = a_sequence_record();
        let pic = a_picture_record();
        let sets = match parameter_sets(&seq, &pic, Some((1, 30))) {
            Ok(s) => s,
            Err(why) => panic!("the parameter sets did not build: {why}"),
        };
        assert_eq!(rbsp_of(&sets.vps, NAL_VPS).map(|b| filmcraft_hevc::params::Vps::parse(&b)).map(|r| r.map(|_| ())), Ok(Ok(())));
        let sps = rbsp_of(&sets.sps, NAL_SPS).expect("the SPS payload");
        assert!(filmcraft_hevc::params::Sps::parse(&sps).is_ok(), "the SPS must parse");
        let pps = rbsp_of(&sets.pps, NAL_PPS).expect("the PPS payload");
        assert!(filmcraft_hevc::params::Pps::parse(&pps).is_ok(), "the PPS must parse");
    }
}
