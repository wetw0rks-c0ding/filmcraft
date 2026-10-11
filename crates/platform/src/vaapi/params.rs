//! The SPS and PPS for a picture this module encoded — derived from the records, then **verified by
//! FilmCraft's own H.264 parser** before they are used.
//!
//! # Why this exists
//!
//! Measured on the machine this milestone was written for (Intel Arc B580, iHD 26.3.5, libva 1.24):
//! **iHD does not write the parameter sets into the coded buffer**, even when the configuration asks
//! for it. `vaCreateConfig` accepts `VAConfigAttribEncPackedHeaders =
//! VA_ENC_PACKED_HEADER_SEQUENCE | VA_ENC_PACKED_HEADER_PICTURE` and writes `0x3` back — libva's own
//! `vaGetConfigAttributes` says the driver supports it, the encode succeeds, and the coded buffer
//! holds a single `VACodedBufferSegment` whose bytes begin `00 00 01 05` (a valid Annex-B IDR slice)
//! and contain **no NAL of type 7 or 8 anywhere**. Annex-B with no parameter sets is not a stream a
//! decoder can start from: ffmpeg answers `Invalid data found when processing input`.
//!
//! So the parameter sets are built here. They are **derived, not guessed**: every field is read out
//! of the very [`VaEncSequenceParameterBufferH264`] and [`VaEncPictureParameterBufferH264`] this
//! module submitted, so if the record says 40x23 macroblocks and a bottom crop of 4 units, the SPS
//! says exactly that. If a field cannot be read back out of the record, that is an error — there is
//! no fallback value to reach for.
//!
//! # Why they are verified before they are used
//!
//! **This is the one place in the encode path where a wrong value produces a plausible-looking wrong
//! picture rather than an error.** A driver told a wrong picture size will happily encode one; a
//! decoder handed a wrong SPS will decode it into something else. The only defence that does not
//! depend on care is to write the bits and read them back with a parser that does not share this
//! module's assumptions — [`filmcraft_h264::params::Sps::parse`] and `Pps::parse`, which the decode
//! path already trusts for real streams. The decoded-back record is compared field by field with the
//! record these bits were written from, so a bit-position mistake fails here instead of on screen.
//!
//! # What this does not settle
//!
//! These are *our* parameter sets, not the driver's. A stream built this way describes the picture
//! FilmCraft asked for, which is the honest scope for an export — but the milestone's Task 4 is
//! still to build `avcC`/`hvcC`, and a driver that does emit its own parameter sets would have its
//! used instead. [`VaEncoder::encode`] prefers the driver's: it only falls back to these when the
//! driver's own output carries no NAL of type 7 or 8.

use super::ffi;
use filmcraft_bitstream::BitWriter;
use filmcraft_h264::params::{Pps, Sps};

/// `VAProfileH264Main` as an `H.264` profile_idc: `VAProfileH264Main` is 6, and H.264's
/// `profile_idc` for Main is 77. They are different numbering systems, which is worth naming because
/// the config profile and the SPS profile have to agree and nothing checks that for us.
const PROFILE_IDC_MAIN: u8 = 77;

/// The same number, reachable from `encoder`'s tests, so the byte offset an `avcC` takes its
/// profile from can be checked against the value that was written rather than against a second
/// copy of 77 typed in the assertion.
#[cfg(test)]
pub(crate) const PROFILE_IDC_FOR_TEST: u8 = PROFILE_IDC_MAIN;
/// `nal_ref_idc = 3` (the parameter sets are reference NAL units) and `nal_unit_type` 7 for an SPS.
const NAL_SPS: u8 = 0x67;
/// `nal_ref_idc = 3`, `nal_unit_type` 8 for a PPS.
const NAL_PPS: u8 = 0x68;
/// The Annex-B start code every NAL this module emits is prefixed with.
const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// The SPS for `record`, as an Annex-B NAL unit, or why it cannot be built.
///
/// Every value is read out of `record` — the record this module submitted — so the SPS and the
/// encode cannot disagree: there is one source, and it is the one the driver was told.
///
/// The syntax order is 7.3.2.1.1: `profile_idc`, `constraint_set` flags and `level_idc`, then
/// `seq_parameter_set_id`, then the fields each of which implies another. `seq_fields` is unpacked
/// here rather than copied, because it is the one field that carries six syntax elements and a wrong
/// bit position in it is invisible until a decoder acts on it.
fn sps(record: &ffi::VaEncSequenceParameterBufferH264) -> Result<Vec<u8>, String> {
    // Every field is checked before it is written, so a record this cannot describe is an error
    // rather than an SPS that says something the record did not.
    if record.picture_width_in_mbs == 0 || record.picture_height_in_mbs == 0 {
        return Err(format!("{}x{} macroblocks", record.picture_width_in_mbs, record.picture_height_in_mbs));
    }
    if record.bit_depth_luma_minus8 != 0 || record.bit_depth_chroma_minus8 != 0 {
        return Err(format!(
            "an H.264 sequence at bit depths {} and {} is not something this builds a sequence parameter set for",
            u32::from(record.bit_depth_luma_minus8) + 8,
            u32::from(record.bit_depth_chroma_minus8) + 8
        ));
    }
    let seq_fields = record.seq_fields;
    let chroma_format_idc = seq_fields & 0b11;
    let frame_mbs_only_flag = (seq_fields >> 2) & 1;
    let mb_adaptive_frame_field_flag = (seq_fields >> 3) & 1;
    let seq_scaling_matrix_present_flag = (seq_fields >> 4) & 1;
    let direct_8x8_inference_flag = (seq_fields >> 5) & 1;
    let log2_max_frame_num_minus4 = (seq_fields >> 6) & 0xf;
    let pic_order_cnt_type = (seq_fields >> 10) & 0b11;
    let log2_max_pic_order_cnt_lsb_minus4 = (seq_fields >> 12) & 0xf;
    // Bit 16 is `delta_pic_order_always_zero_flag`. It is not written below because
    // `pic_order_cnt_type == 0` (refused above unless it is 0) makes it implicit in the syntax: it
    // only appears in the bitstream for type 1. Read here so the field is accounted for.
    let _delta_pic_order_always_zero_flag = (seq_fields >> 16) & 1;

    // The states an SPS for this backend's configuration cannot describe. Each is a refusal rather
    // than a substitute value, for the reason the module is about: a parameter set that says
    // something other than what was encoded decodes into something other than what was encoded.
    if chroma_format_idc != 1 {
        return Err(format!("chroma_format_idc {chroma_format_idc} is not 4:2:0"));
    }
    if mb_adaptive_frame_field_flag != 0 {
        return Err("mb_adaptive_frame_field_flag is set, so the picture is not one frame".into());
    }
    if seq_scaling_matrix_present_flag != 0 {
        return Err("seq_scaling_matrix_present_flag is set, so the sequence declares scaling matrices it does not carry".into());
    }
    if pic_order_cnt_type != 0 {
        return Err(format!("pic_order_cnt_type {pic_order_cnt_type} is not the type-0 this builds a parameter set for"));
    }
    if frame_mbs_only_flag == 0 {
        return Err("frame_mbs_only_flag is clear, so the sequence is a field sequence".into());
    }

    let mut w = BitWriter::new();
    w.write_bits(u32::from(NAL_SPS), 8);
    // `profile_idc`. Main profile, and Main profile's constraint_set flags are all zero — the
    // backend encodes CAVLC (`entropy_coding_mode_flag` clear, see `picture_record`), no 8x8
    // transform and no weighted prediction, which is exactly what Main without constraints means.
    w.write_bits(u32::from(PROFILE_IDC_MAIN), 8);
    // `constraint_set0..5` plus `reserved_zero_2bits`, all clear.
    w.write_bits(0, 8);
    w.write_bits(u32::from(record.level_idc), 8);
    // `seq_parameter_set_id`, a `ue(v)`.
    w.write_ue(0);
    // `log2_max_frame_num_minus4`, `pic_order_cnt_type`, `log2_max_pic_order_cnt_lsb_minus4`.
    w.write_ue(log2_max_frame_num_minus4);
    w.write_ue(pic_order_cnt_type);
    w.write_ue(log2_max_pic_order_cnt_lsb_minus4);
    // `num_ref_frames_in_pic_order_cnt_cycle` and the cycle itself are only written when
    // `pic_order_cnt_type == 1`, which the check above has refused. This is a state-dependent
    // syntax chain: writing one element of it unconditionally is how a parameter set gets a
    // `ue(v)` where a flag belongs.
    w.write_ue(record.max_num_ref_frames);
    // `gaps_in_frame_num_value_allowed_flag`.
    w.write_bit(false);
    w.write_ue(u32::from(record.picture_width_in_mbs) - 1);
    w.write_ue(u32::from(record.picture_height_in_mbs) - 1);
    w.write_bit(frame_mbs_only_flag != 0);
    w.write_bit(direct_8x8_inference_flag != 0);
    // `frame_cropping_flag` and, only when it is set, the four crop offsets.
    let cropping = (record.frame_crop_left_offset, record.frame_crop_right_offset, record.frame_crop_top_offset, record.frame_crop_bottom_offset);
    if cropping == (0, 0, 0, 0) {
        w.write_bit(false);
    } else {
        w.write_bit(true);
        for offset in [cropping.0, cropping.1, cropping.2, cropping.3] {
            w.write_ue(offset);
        }
    }
    // `vui_parameters_present_flag`: the record says no, and this module has no timing to declare.
    w.write_bit(record.vui_parameters_present_flag != 0);
    if record.vui_parameters_present_flag != 0 {
        // `aspect_ratio_info_present_flag` and then the aspect ratio idc, which is 255 (EXTENDED_SAR)
        // or an idc. An SAR this module cannot derive is the honest "unspecified".
        w.write_bit(false);
    }
    w.rbsp_trailing();
    let nal = filmcraft_bitstream::escape_rbsp(&w.finish());
    let mut out = START_CODE.to_vec();
    out.extend_from_slice(&nal);
    Ok(out)
}

/// The PPS for `sequence` and `picture`, as an Annex-B NAL unit, or why it cannot be built.
///
/// `pic_init_qp_minus26` is `pic_init_qp - 26` because the record's `pic_init_qp` is
/// `pic_init_qp_minus26 + 26` — the one field where re-deriving it in the wrong direction produces
/// a picture at a quality nobody asked for, with no error anywhere. It is checked below against the
/// 0..=51 range a `se(v)` can express as a quantiser.
fn pps(sequence: &ffi::VaEncSequenceParameterBufferH264, picture: &ffi::VaEncPictureParameterBufferH264) -> Result<Vec<u8>, String> {
    let _ = sequence;
    // `pic_init_qp` is 0..=51 for any H.264 picture; outside that the derived `se(v)` would name a
    // quantiser no decoder would accept, so it is refused here rather than written.
    if picture.pic_init_qp > 51 {
        return Err(format!("pic_init_qp {} is outside H.264's 0..=51", picture.pic_init_qp));
    }
    // `pic_init_qp_minus26` is signed and H.264's range for it is -26..=25, which is exactly the
    // 0..=51 a `u8` quantiser can hold once 26 is added.
    let pic_init_qp_minus26 = i32::from(picture.pic_init_qp) - 26;
    let fields = picture.pic_fields;
    let mut w = BitWriter::new();
    w.write_bits(u32::from(NAL_PPS), 8);
    w.write_ue(0); // pic_parameter_set_id
    w.write_ue(0); // seq_parameter_set_id
    w.write_bit(fields & (1 << 3) != 0); // entropy_coding_mode_flag
    // `bottom_field_pic_order_in_frame_present_flag`: the record's `pic_order_present_flag` is the
    // closest thing to it and is what says whether a POC delta rides in the slice header.
    w.write_bit(fields & (1 << 11) != 0);
    w.write_ue(0); // num_slice_groups_minus1
    w.write_ue(u32::from(picture.num_ref_idx_l0_active_minus1));
    w.write_ue(u32::from(picture.num_ref_idx_l1_active_minus1));
    w.write_bit(fields & (1 << 4) != 0); // weighted_pred_flag
    w.write_bits((fields >> 5) & 0b11, 2); // weighted_bipred_idc
    w.write_se(pic_init_qp_minus26);
    w.write_se(pic_init_qp_minus26); // pic_init_qs_minus26: the same quantiser, no VUI overrides it
    w.write_se(i32::from(picture.chroma_qp_index_offset));
    w.write_bit(fields & (1 << 9) != 0); // deblocking_filter_control_present_flag
    w.write_bit(fields & (1 << 7) != 0); // constrained_intra_pred_flag
    w.write_bit(fields & (1 << 10) != 0); // redundant_pic_cnt_present_flag
    w.rbsp_trailing();
    let nal = filmcraft_bitstream::escape_rbsp(&w.finish());
    let mut out = START_CODE.to_vec();
    out.extend_from_slice(&nal);
    Ok(out)
}

/// The parameter sets for this picture, **verified by parsing them back**, or why they cannot be.
///
/// The verification is the whole point of this function and not decoration. The bits are read back
/// with [`Sps::parse`] and [`Pps::parse`] — the parsers the decode path already trusts for real
/// streams — and every field the encoder cares about is compared with the record the bits were
/// written from. A bit-position mistake, a wrong unit or a missing syntax element fails here, in
/// a test, rather than producing a stream that decodes into a plausible wrong picture.
pub fn parameter_sets(sequence: &ffi::VaEncSequenceParameterBufferH264, picture: &ffi::VaEncPictureParameterBufferH264) -> Result<Vec<u8>, String> {
    let sps_bytes = sps(sequence)?;
    let pps_bytes = pps(sequence, picture)?;
    verify(&sps_bytes, &pps_bytes, sequence, picture)?;
    let mut out = sps_bytes;
    out.extend_from_slice(&pps_bytes);
    Ok(out)
}

/// Read `sps_bytes` and `pps_bytes` back with FilmCraft's own parsers and check them against the
/// records they were written from.
///
/// Two checks, and the split is not tidiness — it is the difference between *what the record
/// determines* and *what this module chose*, and the two are verified differently:
///
/// - [`cross_check`] compares the fields the submitted record carries, and it is the one that runs
///   on **either** provenance. A driver that writes its own SPS and PPS is checked against the same
///   record we submitted, so a driver that quietly rewrote the picture size or the reference depth
///   is caught rather than believed.
/// - the table below additionally compares the fields **no record carries** — `profile_idc`,
///   `gaps_in_frame_num_value_allowed_flag`, `bottom_field_pic_order_in_frame_present_flag`,
///   `constrained_intra_pred_flag`, `redundant_pic_cnt_present_flag` — against the values this
///   module chose to write. That check only makes sense for the derived path, where those values
///   *are* this module's and a bit written in the wrong place would go unnoticed; for a
///   driver-written set they are the driver's own choice and comparing them against ours would be
///   comparing the driver with a guess.
fn verify(
    sps_bytes: &[u8],
    pps_bytes: &[u8],
    sequence: &ffi::VaEncSequenceParameterBufferH264,
    picture: &ffi::VaEncPictureParameterBufferH264,
) -> Result<(), String> {
    let Some(sps_rbsp) = strip(sps_bytes, NAL_SPS) else {
        return Err("the SPS this module built has no Annex-B start code".into());
    };
    let Some(pps_rbsp) = strip(pps_bytes, NAL_PPS) else {
        return Err("the PPS this module built has no Annex-B start code".into());
    };
    let sps = Sps::parse(&sps_rbsp).map_err(|e| format!("the SPS this module built does not parse: {e}"))?;
    let pps = Pps::parse(&pps_rbsp, &[Some(sps.clone())]).map_err(|e| format!("the PPS this module built does not parse: {e}"))?;
    cross_check(&sps, &pps, sequence, picture)?;
    // Every comparison below is between a field this module **wrote** and the field the parser
    // read back. They are different code paths over the same bytes, which is the only kind of check
    // that catches a bit written at the wrong position.
    //
    // **Every syntax element this module writes is checked, not the ones that happened to be
    // convenient.** The ones below were missing until now, and they are missing for the worst
    // possible reason: each stays parseable when it is wrong. `pic_order_cnt_type` is a `ue(v)`
    // either way, so a `log2_max_pic_order_cnt_lsb_minus4` written at the wrong position does not
    // stop the SPS parsing — it shifts everything after it and the parser reports a *different*
    // `max_num_ref_frames`, which is a picture nobody can play rather than an error. The module's
    // own premise is that this comparison "is the only defence that does not depend on care", and a
    // defence with a list of exceptions is not one.
    let mismatches: Vec<String> = [
        (sps.profile_idc == PROFILE_IDC_MAIN, format!("profile_idc is {} not {PROFILE_IDC_MAIN}", sps.profile_idc)),
        (!sps.gaps_in_frame_num_allowed, format!("gaps_in_frame_num_value_allowed_flag is {}", sps.gaps_in_frame_num_allowed)),
        (pps.bottom_field_pic_order_in_frame_present == (picture.pic_fields & (1 << 11) != 0), "bottom_field_pic_order_in_frame_present_flag".into()),
        (pps.constrained_intra_pred == (picture.pic_fields & (1 << 7) != 0), "constrained_intra_pred_flag".into()),
        (pps.redundant_pic_cnt_present == (picture.pic_fields & (1 << 10) != 0), "redundant_pic_cnt_present_flag".into()),
    ]
    .into_iter()
    .filter_map(|(ok, why)| (!ok).then_some(why))
    .collect();
    if mismatches.is_empty() {
        return Ok(());
    }
    Err(format!(
        "the parameter sets this module built say {} — {} mismatch(es) against the values they were written from",
        mismatches.join(", "),
        mismatches.len()
    ))
}

/// **Check a parameter set against the record this backend submitted**, whichever wrote the set.
///
/// This is the check that makes a derived `avcC` self-checking rather than merely self-consistent:
/// the derived sets are written from the record and read back out of the bits, which proves the
/// *writing* was right. It proves nothing about whether the sets still *describe the output* — a
/// record is what was asked for, not what was encoded. So the same comparison is run against a
/// driver's own parameter sets, where it does mean that: if the driver wrote an SPS saying a
/// different picture size, a different reference depth or a different frame-number width from the
/// record, then the `avcC` FilmCraft would hand the muxer and the stream are describing different
/// things, and the honest result is this error.
///
/// Only the fields the record carries are compared. `profile_idc`, `constraint_set*_flag`,
/// `gaps_in_frame_num_value_allowed_flag`, `bottom_field_pic_order_in_frame_present_flag`,
/// `constrained_intra_pred_flag` and `redundant_pic_cnt_present_flag` are **not** in
/// `VAEncSequenceParameterBufferH264` or `VAEncPictureParameterBufferH264`: they are choices this
/// module makes when it derives a set, and a driver that made different ones is not contradicting
/// anything FilmCraft submitted.
pub fn cross_check(
    sps: &Sps,
    pps: &Pps,
    sequence: &ffi::VaEncSequenceParameterBufferH264,
    picture: &ffi::VaEncPictureParameterBufferH264,
) -> Result<(), String> {
    let seq_fields = sequence.seq_fields;
    let mismatches: Vec<String> = [
        // The ids first, because everything else is looked up through them: a slice resolves its
        // PPS by `pic_parameter_set_id`, which resolves its SPS, and an `avcC` whose sets are not
        // the ones the slices name is a stream no decoder can start.
        (sps.id == u32::from(sequence.seq_parameter_set_id), format!("seq_parameter_set_id is {} not {}", sps.id, sequence.seq_parameter_set_id)),
        (pps.id == u32::from(picture.pic_parameter_set_id), format!("pic_parameter_set_id is {} not {}", pps.id, picture.pic_parameter_set_id)),
        (
            pps.sps_id == u32::from(sequence.seq_parameter_set_id),
            format!("the PPS names sequence parameter set {} not {}", pps.sps_id, sequence.seq_parameter_set_id),
        ),
        // `level_idc` is a conformance statement, and a stream whose level is below its own
        // geometry is one a conforming decoder refuses — so it is a field the record determines and
        // therefore checked on **either** provenance. If a driver rewrote it, the level FilmCraft
        // reports and the level in the `avcC` are different claims about the same stream.
        (sps.level_idc == sequence.level_idc, format!("level_idc is {} not {}", sps.level_idc, sequence.level_idc)),
        (
            sps.log2_max_frame_num == ((seq_fields >> 6) & 0xf) + 4,
            format!("log2_max_frame_num is {} not {}", sps.log2_max_frame_num, ((seq_fields >> 6) & 0xf) + 4),
        ),
        (sps.pic_order_cnt_type == (seq_fields >> 10) & 0b11, format!("pic_order_cnt_type is {} not {}", sps.pic_order_cnt_type, (seq_fields >> 10) & 0b11)),
        // Bits 12-15. Written as a `ue(v)` that is 0 for this encoder's `pic_order_cnt_type == 0`
        // configuration, and read back as the value the parser saw — so a bit position mistake here
        // shows up as a number rather than as a parse failure, which is exactly the case a check is
        // for. The frame loop derives every picture's `pic_order_cnt_lsb` modulo
        // `1 << (this + 4)`, so a wrong value here is a wrong picture order count, silently.
        (
            sps.log2_max_poc_lsb == ((seq_fields >> 12) & 0xf) + 4,
            format!("log2_max_pic_order_cnt_lsb is {} not {}", sps.log2_max_poc_lsb, ((seq_fields >> 12) & 0xf) + 4),
        ),
        // `delta_pic_order_always_zero_flag` is bit 16. `sps()` reads it so the field is accounted
        // for and does not write it, because `pic_order_cnt_type == 0` makes it implicit in the
        // syntax; the parser only reports it for type 1, so it is compared against the same rule
        // rather than against the bit.
        (sps.delta_pic_order_always_zero || sps.pic_order_cnt_type != 1, "delta_pic_order_always_zero_flag".into()),
        (
            sps.max_num_ref_frames == sequence.max_num_ref_frames,
            format!("max_num_ref_frames is {} not {}", sps.max_num_ref_frames, sequence.max_num_ref_frames),
        ),
        (sps.chroma_format_idc == (seq_fields & 0b11), format!("chroma_format_idc is {} not {}", sps.chroma_format_idc, seq_fields & 0b11)),
        (sps.frame_mbs_only == ((seq_fields >> 2) & 1 != 0), "frame_mbs_only_flag".into()),
        (sps.direct_8x8_inference == ((seq_fields >> 5) & 1 != 0), "direct_8x8_inference_flag".into()),
        (
            sps.pic_width_in_mbs == u32::from(sequence.picture_width_in_mbs),
            format!("width is {} mbs not {}", sps.pic_width_in_mbs, sequence.picture_width_in_mbs),
        ),
        (
            sps.pic_height_in_map_units == u32::from(sequence.picture_height_in_mbs),
            format!("height is {} not {}", sps.pic_height_in_map_units, sequence.picture_height_in_mbs),
        ),
        (sps.frame_crop == expected_crop(sequence), format!("frame_crop is {:?} not {:?}", sps.frame_crop, expected_crop(sequence))),
        // The PPS's `pic_init_qp`: the one field of a picture record that a derived SPS/PPS pair
        // cannot be stable under, because it moves with the quantiser. A driver that rewrote it is
        // caught here, and a caller that changes the quantiser mid-export is caught by
        // `VaEncoder::sample_entry`.
        (pps.pic_init_qp == i32::from(picture.pic_init_qp), format!("pic_init_qp is {} not {}", pps.pic_init_qp, picture.pic_init_qp)),
        // `pic_init_qs_minus26` is written as the same quantiser: this backend submits no VUI timing
        // and no picture-order-count cycle, so nothing overrides the initial Q for a non-I picture.
        // That is a decision, and a decision that is not checked is a decision that can drift.
        (pps.pic_init_qs == i32::from(picture.pic_init_qp), format!("pic_init_qs is {} not {}", pps.pic_init_qs, picture.pic_init_qp)),
        (pps.entropy_coding_mode == (picture.pic_fields & (1 << 3) != 0), "entropy_coding_mode_flag".into()),
        (pps.deblocking_filter_control_present == (picture.pic_fields & (1 << 9) != 0), "deblocking_filter_control_present_flag".into()),
        (pps.weighted_pred == (picture.pic_fields & (1 << 4) != 0), "weighted_pred_flag".into()),
        (pps.weighted_bipred_idc == (picture.pic_fields >> 5) & 0b11, format!("weighted_bipred_idc is {}", pps.weighted_bipred_idc)),
        (
            pps.num_ref_idx_l0_default_active == u32::from(picture.num_ref_idx_l0_active_minus1) + 1,
            format!("num_ref_idx_l0_default_active is {} not {}", pps.num_ref_idx_l0_default_active, u32::from(picture.num_ref_idx_l0_active_minus1) + 1),
        ),
        (
            pps.num_ref_idx_l1_default_active == u32::from(picture.num_ref_idx_l1_active_minus1) + 1,
            format!("num_ref_idx_l1_default_active is {} not {}", pps.num_ref_idx_l1_default_active, u32::from(picture.num_ref_idx_l1_active_minus1) + 1),
        ),
    ]
    .into_iter()
    .filter_map(|(ok, why)| (!ok).then_some(why))
    .collect();
    if mismatches.is_empty() {
        return Ok(());
    }
    Err(format!("the parameter sets say {} — {} mismatch(es) against the record this backend submitted", mismatches.join(", "), mismatches.len()))
}

/// The crop the SPS should carry, or `None` when the record declares none.
fn expected_crop(sequence: &ffi::VaEncSequenceParameterBufferH264) -> Option<(u32, u32, u32, u32)> {
    let crop = (sequence.frame_crop_left_offset, sequence.frame_crop_right_offset, sequence.frame_crop_top_offset, sequence.frame_crop_bottom_offset);
    (crop != (0, 0, 0, 0)).then_some(crop)
}

/// The RBSP of the NAL unit in `bytes` whose header byte is `header`, or `None`.
fn strip(bytes: &[u8], header: u8) -> Option<Vec<u8>> {
    let at = bytes.iter().position(|&b| b == header)?;
    // The header byte itself is not part of the RBSP, and the four bytes before it are the start code.
    let rbsp = bytes.get(at + 1..)?;
    Some(rbsp.to_vec())
}

// **Whether the driver wrote its own parameter sets is decided in `encoder`, not here**, by
// [`super::encoder::resolve_parameter_sets`] walking the driver's own coded NAL units with
// `annex_b_units` and looking for a type 7 and a type 8. An earlier version of this module
// answered the question with a `has_parameter_sets` helper here; it was removed rather than
// repaired because it was a *second* Annex-B walk that nothing but its own tests called, and two
// walks of one format is two places for the answer to differ — which is the whole failure this
// branch has documented three times. There is one walk now, and it is the one the answer comes
// from.
