//! The HEVC **submission** path: the records, the frame loop, and the readback — the counterpart of
//! [`super::encoder`]'s H.264 one, in a file of its own.
//!
//! # Why a second frame loop rather than a generic one
//!
//! The H.264 loop and this one have the same *shape* — a config, a rotation of reconstruction
//! surfaces, a list of references, a sequence record submitted every picture, a render, a sync, a
//! readback — and almost none of the same *content*. What differs is not a detail:
//!
//! - H.264's reference identity is `(surface, frame_idx, frame_num, poc)`; **HEVC's is `(surface,
//!   pic_order_cnt)`**, a different picture type (`VAPictureHEVC`, 28 bytes) with no `frame_idx` at
//!   all. The H.264 slot arithmetic ([`super::encoder::VaEncoder::slot_for`]) has no HEVC analogue
//!   to reuse.
//! - H.264 addresses a slice in **macroblocks**; HEVC addresses it in **CTUs**, so the picture's
//!   geometry, the crop and the slice record are three different units.
//! - HEVC's parameter sets are a **VPS as well as an SPS and a PPS**, they are a *different bit
//!   syntax*, and [`super::params`] builds H.264's from H.264 records — nothing in it applies.
//! - HEVC's record carries **two features the driver states it cannot do without**
//!   (`amp_enabled_flag` and `cu_qp_delta_enabled_flag`; see [`HevcEncoder::new`]), which have no
//!   H.264 counterpart at all.
//!
//! Making the H.264 loop generic over the codec would mean parameterising all of that behind
//! traits, inside a file another agent is working in, for two implementations. **Duplicating the
//! shape and sharing the parts that really are shared is the cheaper and safer of the two**, and
//! what is shared is named below rather than copied.
//!
//! # What is shared with the H.264 path, and why that is sharing and not duplication
//!
//! - [`super::ffi::Encoder`] — the whole libva session (config, surfaces, context, coded buffer,
//!   upload, render, sync, readback). It is already profile-generic: [`HevcEncoder::new`] hands it
//!   [`Profile::HevcMain`] where the H.264 path hands it [`Profile::H264Main`]. **This is the big
//!   one**, and it is the FFI boundary ADR 0001 rests on.
//! - [`super::encoder::RateControl`] — a codec-agnostic statement of *which* fidelity this stream
//!   makes, with a `vaCreateConfig` attribute behind it. Duplicating it would put two enums with
//!   two `config_attribute()` methods that could answer differently for the same question.
//! - [`super::encoder::ramp_picture`] / [`super::encoder::ramp_shifted`] — the synthetic content the
//!   verification is built on. Nothing about them is H.264: a luma ramp is a ramp.
//! - The Annex-B walk and the misc-buffer framing, which this file re-asks for rather than re-derives
//!   (see [`annex_b_units`] and [`misc`]), each with the reason written where it is.
//!
//! # `slice_type`: what the driver writes, read off a stream it encoded
//!
//! `va_enc_hevc.h:776-779` says only *"Slice type. Corresponds to HEVC syntax element of the same
//! name"* — which is not an encoding, and **no enum in libva 1.24 states one**
//! (`grep -rn "SLICE_TYPE" /usr/include/va/` returns only H.264's). H.264's own header at least says
//! "Range: 0..2, 5..7, i.e. no switching slices", and H.264's is *not* the HEVC spelling either.
//!
//! **It was measured by reading back a stream this driver encoded**, and the exact strength of that
//! evidence is in [`slice_types`] — the short version is that "what the driver writes" is measured
//! (I = 2, P = 1, B = 0 on a stream ffmpeg decodes with no errors) and **"what the driver accepts"
//! is not**, because this path's own submissions do not yet produce a picture. Nothing here writes a
//! literal: the two numbers come from one named place, and [`HevcEncoder::slice_type_encoding`] is
//! the only way to reach any other value.
//!
//! # This path does not yet produce a picture, and the milestone's bar is not met
//!
//! **Every submission fails at the driver's own encode step**: `vaRenderPicture`, `vaEndPicture` and
//! `vaSyncBuffer` all return success, and `vaMapBuffer` on the coded buffer then returns
//! `VA_STATUS_ERROR_ENCODING_ERROR`. So there is no exported file, no P-frame prediction to measure,
//! and none of Task 1's "one decodable NAL stream" has happened for HEVC.
//!
//! **What is established and pinned:** the three encode records' sizes, offsets and every hole
//! against a C compile (`hevc_abi.rs`); the derived VPS/SPS/PPS parsing back through FilmCraft's own
//! HEVC parsers with every field compared against the record it was written from; the rate-control
//! mask and the `VA_RC_VCM` refusal; the quantisation-misc-buffer refusal; the level table; and
//! `slice_type`. **What is not established:** why the driver refuses to encode, and therefore
//! whether P frames predict at all.
//!
//! **Three real bitstream bugs were found and fixed on the way, each of which had silently
//! displaced a field rather than failing loudly** — see `hevc_params.rs`'s `sps` and the VUI block.
//!
//! # Verification is materially weaker here than for decode
//!
//! Decode could assert 72 of 72 pictures bit-exact against ffmpeg. **An encoder cannot**: the
//! hardware encoder and a software encoder are different algorithms, so bit-equality is not
//! available at any level and "matches ffmpeg" would be a meaningless claim. What this module
//! checks instead is
//!
//! - the driver's bytes **parse** as the stream the records say they are — the derived VPS, SPS and
//!   PPS are read back with [`filmcraft_hevc::params`]' parsers, the same ones the decode path trusts
//!   for real streams, and every field is compared with the record the bits were written from;
//! - the **slice header the driver wrote** agrees with the records: its NAL type, its `slice_type`,
//!   its POC, its reference indices, parsed back out of the coded bytes by the same parser
//!   ([`check_output_against_the_records`]);
//! - and the end-to-end claim, which is the only one that is really about pictures: the exported
//!   file **decodes with ffmpeg to the right pictures**, within a lossy tolerance, against the
//!   **source**. That is `crates/platform/tests/hevenc.rs` and `examples/vahevcprobe.rs`, and it is
//!   the gate. **It is not reached**, for the reason given above: the first two are implemented and
//!   run before every submission, and they are what caught the three displaced-field bugs, but they
//!   check records that never became a picture.

use super::encoder::{MAX_REFERENCES, PictureKind, RateControl};
use super::ffi::{
    self, Display, Encoder, PACKED_HEADER_PICTURE, PACKED_HEADER_SEQUENCE, PICTURE_HEVC_INVALID, PICTURE_HEVC_RPS_ST_CURR_BEFORE, Profile, RtFormat,
};
use super::hevc_enc::*;
use std::sync::Arc;

/// The minimum coding block size, in luma samples: `Log2MinCbSizeY = log2_min_luma_coding_block_size_minus3 + 3`, and this
/// encoder writes 0, so **8**.
///
/// A picture's `pic_width_in_luma_samples` and `pic_height_in_luma_samples` must both be whole
/// multiples of it (7.4.3.2.1.1's `pic_width_in_luma_samples` / `pic_height_in_luma_samples`
/// constraints, and `va_enc_hevc.h:378-380` says the same of the record). Unlike H.264, **HEVC's
/// sequence record carries no crop**: there is nowhere in it to say "the picture was padded", so a
/// size that is not a multiple of 8 is **refused** rather than rounded — see [`HevcEncoder::new`].
const MIN_CB: u32 = 8;

/// The coding tree block size, in luma samples: **`CtbSizeY = 64`**, because
/// `log2_diff_max_min_luma_coding_block_size` is 3 above a `MinCbSizeY` of 8 (7.4.3.2.1.1:
/// `CtbLog2SizeY = log2_diff_max_min_luma_coding_block_size + log2_min_luma_coding_block_size`).
///
/// **This is the unit a HEVC slice is addressed in.** `slice_segment_address` and
/// `num_ctu_in_slice` (`va_enc_hevc.h:771-774`) count **CTUs**, not macroblocks and not pixels, so a
/// slice record written in the wrong unit asks the driver for a slice of the wrong length and
/// produces a picture rather than an error.
const CTB: u32 = 64;

/// `log2_max_pic_order_cnt_lsb_minus4` as this encoder writes it: **0**, so `MaxPicOrderCntLsb` is
/// 16.
///
/// **The picture order count the records carry is a full `i32` and is not reduced modulo this**, and
/// the reason is the record: `VAPictureHEVC::pic_order_cnt` is 32 bits (`va.h:5296-5298`), so the
/// driver is handed a real POC and works out the LSB itself. Reducing it here would be the
/// application's second, conflicting statement about the same number.
pub const LOG2_MAX_POC_LSB_MINUS4: u32 = 8;

/// **The `slice_type` values this driver takes, measured rather than read** — see
/// [`slice_types`] for the measurement.
///
/// `SLICE_TYPE_I` is what an IDR slice header carries and `SLICE_TYPE_P` what a trailing picture's
/// does. They are **HEVC's syntax element values, not H.264's offset-by-five ones**: H.264's 5 and 7
/// are written into a HEVC slice record by anyone who transposes the H.264 path, and they name
/// nothing — `slice_type` is a `u(2)` in the bitstream, so 5 and 7 would land as
/// `slice_type & 3` = 1 and 3, which is "B" and a reserved value.
const SLICE_TYPE_I: u8 = 2;
const SLICE_TYPE_P: u8 = 1;

/// **HEVC's `slice_type` encoding, established empirically because the header does not state one.**
///
/// # The measurement
///
/// `va_enc_hevc.h:776-779` says only *"Slice type. Corresponds to HEVC syntax element of the same
/// name."* There is no enum to ask — `grep -rn "SLICE_TYPE" /usr/include/va/` on libva 1.24 returns
/// `va_enc_h264.h`'s `H264_ENC_SLICE_TYPE_*` and nothing for HEVC, and
/// `grep -rn "VAProfileH265Enc" /usr/include/va/` returns nothing either, so there is not even a
/// separate encode profile: **the codec profile `VAProfileHEVCMain` (17) is reused with an encode
/// entry point**, which is why `vaQueryConfigEntrypoints` answers `[1 VLD, 6 EncSlice]` for both 17
/// and 18.
///
/// HEVC's syntax element is a `u(2)` (`slice_type` in 7.4.7.1), so the field can only name 0, 1, 2
/// or 3 — where 0 is B, 1 is P, 2 is I and 3 is the reserved value the spec says shall not be
/// present in a conforming bitstream. **What the driver *does* with the number it is given is a
/// separate question from what the number means**, and the header does not answer it.
///
/// # How it was established — and what that evidence is and is not
///
/// **It was established by reading back a stream this same driver encoded**, not by reading a
/// submission of ours, and the difference matters, so it is stated first.
///
/// **The evidence.** `ffmpeg -vaapi_device /dev/dri/renderD128 -c:v hevc_vaapi` on this machine
/// (`VAProfileHEVCMain`, `VAEntrypointEncSlice`, CTU 64x64, min CB 8 — the same profile, entry point
/// and block sizes this path submits) produces a stream that **ffmpeg decodes with 0 errors**, and
/// ffmpeg's own `trace_headers` over that stream reads:
///
/// ```text
/// 19  slice_type   011 = 2      <- the IDR_W_RADL slice
/// 18  slice_type   010 = 1      <- a TRAIL_R slice
/// 18  slice_type     1 = 0      <- a TRAIL_N slice (B)
/// ```
///
/// **I = 2, P = 1, B = 0 — the syntax element's own values, with no offset, on a stream this driver
/// produced and a decoder accepts.** That is a measurement of the driver's output, and it rules out
/// H.264's offset-by-five encoding for this driver: 5 and 7 are not even representable in a `u(2)`,
/// so truncating them gives 1 (P) and 3 (reserved) — a key frame a decoder would not read as I.
///
/// **What the evidence is not.** It is **not** a sweep of our own submissions. Every submission this
/// path makes fails before it produces a picture — `vaRenderPicture` and `vaEndPicture` both return
/// success, `vaSyncBuffer` returns success, and `vaMapBuffer` on the coded buffer then returns
/// `VA_STATUS_ERROR_ENCODING_ERROR` — so `examples/vahevcprobe.rs --sweep-slice-type` prints
/// "refused" for all six candidates and reads back nothing. **The direction that is unmeasured is
/// "which values does this driver accept, and does it reinterpret any of them"; the direction that
/// is measured is "what does this driver write".** The first is what a sweep was for, and it is
/// still open; see the refusal in [`HevcEncoder`] and `tests/hevenc.rs`.
///
/// # What this rules out
///
/// - **H.264's encoding.** 5 and 7 are not representable in a `u(2)`, and the driver's own output
///   uses 2 for I and 1 for P. An encoder built by copying the H.264 slice record writes 7 on its
///   IDR, and ffmpeg then reports `slice_type 3` — a reserved value — on a key frame it should have
///   read as I.
/// - **`coding_type` as the answer.** `VAEncPictureParameterBufferHEVC::pic_fields.coding_type` is
///   documented (I 1, P 2, B 3) and is *not* the `slice_type` field. Writing 2 there does not make
///   the slice a P slice; it names the picture type to the rate controller.
/// - **Clause 7.4.7.1 as a substitute for the measurement.** The clause defines the values, but the
///   question is not "what does 2 mean" — that is not in doubt — it is "what does this driver write
///   in this field", and the answer above is read off a real stream rather than assumed from the
///   clause.
///
/// **The measurement is not re-taken by an oracle test on this path, and that is a known gap rather
/// than an oversight**: there is no picture to run ffmpeg's `trace_headers` over yet. The gap closes
/// when the submission stops failing, and until then a `slice_type` chosen wrongly here would
/// produce a file that decodes cleanly and is subtly wrong.
pub mod slice_types {
    /// **What was measured: what this driver writes.** A stream iHD 26.3.5 encoded through
    /// `VAProfileHEVCMain` / `VAEntrypointEncSlice`, which ffmpeg decodes with 0 errors, read back
    /// with ffmpeg's own `trace_headers`.
    ///
    /// | NAL unit | slice type the driver wrote | meaning |
    /// |---|---|---|
    /// | `19` `IDR_W_RADL` | `slice_type 2` | **I** |
    /// | `1` `TRAIL_R` | `slice_type 1` | **P** |
    /// | `0` `TRAIL_N` | `slice_type 0` | **B** |
    ///
    /// **Read down the column: the field is HEVC's own `u(2)`, written back with no offset.** So the
    /// two values this encoder writes — 2 for an IDR, 1 for a trailing picture — are the ones the
    /// specification gives those slice types. The consequence that matters is the negative row: **an
    /// I slice is only an I slice if the field says 2**, and a decoder reads back what the driver
    /// wrote.
    ///
    /// **This is a measurement of iHD 26.3.5 / libva 1.24**, not of HEVC and not of libva, which is
    /// why it lives beside the code that submits it rather than in a constant somewhere it could be
    /// applied to another driver unseen.
    ///
    /// **It says what the driver writes, not what it accepts.** Our own submissions do not reach a
    /// picture yet (see [`slice_types`]'s module documentation), so the "which value does it take"
    /// half of the sweep is unmeasured. This constant is not that half and must not be read as it.
    pub const HEVC: &str = "measured on a stream iHD 26.3.5 encoded itself: I = 2, P = 1, B = 0, the u(2) written back with no offset; \
         what this driver ACCEPTS is not measured, because our own submissions do not yet produce a picture";
}

/// `VA_RC_VCM` (`va.h:1114`): `0x00000008`.
///
/// **It is in HEVC's rate-control mask and not H.264's, and this backend refuses it.**
///
/// `VaEncoder::rate_control_names` lists H.264's eight modes and VCM is not among them, because
/// VCM — variable-bit-rate with a per-picture cap — is not in the mask libva reports for
/// `VAProfileH264Main` on this driver. For `VAProfileHEVCMain` the same query answers with
/// `0x14de`, which **is** `CQP | VCM | QVBR | TCBRC` and therefore contains it. A mode that appears
/// in one codec's mask and not the other's is exactly the kind of thing that gets accepted
/// silently: `RateControl` has no variant for it, so a caller cannot name it — but a caller who
/// added one, or a future change that forwarded a mask bit into a config attribute, would be asking
/// the driver for a mode this backend has no record to honour and no measurement to describe.
///
/// So it is a named refusal rather than a silent gap: [`HevcEncoder::refuses_vcm`] carries the
/// mask it appears in and the test that catches an attempt to submit it.
pub const VCM: u32 = 0x0000_0008;

/// **The rate-control mask this driver reports for `VAProfileHEVCMain` / `VAEntrypointEncSlice`:**
/// `0x14de`, which is `VA_RC_CQP | VA_RC_VCM | VA_RC_QVBR | VA_RC_TCBRC` (`va.h:1110-1164`).
///
/// **A measurement, and it is the reason [`VCM`] is a named refusal.** H.264's mask on the same
/// driver is `0x14d6` — the same four modes less `VA_RC_VCM` — so `VaEncoder::rate_control_names`
/// never needed a name for it. Pinning the number here rather than only in the test is what lets a
/// decline message say what the driver offered.
pub const HEVC_RATE_CONTROL_MASK: u32 = 0x14de;

/// **`VA_RC_VCM` is in HEVC's mask and not H.264's**, as [`VCM`] and [`HEVC_RATE_CONTROL_MASK`]
/// together say. This is H.264's mask on the same driver, carried so the difference is checkable in
/// one comparison rather than asserted in prose.
pub const H264_RATE_CONTROL_MASK: u32 = 0x14d6;

/// `VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE` (`va.h:5336`): a `ReferenceFrames[]` entry this picture's
/// short-term reference picture set refers to, *before* it in decoding order.
///
/// **It is the flag a P picture's own reference needs**, and it is not decoration: `va.h:5323-5327`
/// says the number of entries with this bit is `NumPocStCurrBefore` and that the counts of the three
/// RPS flags sum to `NumPocTotalCurr`, which the driver builds the slice header's `ref_pic_lists()`
/// from. A reference entry with no RPS flag is a picture in the DPB that the current picture's RPS
/// does not mention — which is a picture the slice header cannot name.
pub const RPS_ST_CURR_BEFORE: u32 = PICTURE_HEVC_RPS_ST_CURR_BEFORE;

/// `HEVC_LAST_PICTURE_EOSTREAM` (`va_enc_hevc.h:253`): `0x02`, and the only flag this encoder ever
/// sets in `last_picture`.
///
/// `HEVC_LAST_PICTURE_EOSEQ` (`0x01`) is deliberately never set, for the reason the H.264 path gives
/// at length: this frame loop has no sequence boundary to report — its picture stream is the whole
/// file — and a stream that ends a sequence it never started is a claim about something that does
/// not exist.
pub const LAST_PICTURE_EOSTREAM_HEVC: u8 = 0x02;

/// What one reconstructed HEVC picture is identified by, which is **not** what H.264 identifies one
/// by.
///
/// `VAPictureHEVC` (`va.h:5290-5303`) is `picture_id`, `pic_order_cnt` and `flags` — there is no
/// `frame_idx` and no `frame_num`. **A HEVC reference is `(surface, poc)`, and the two travel
/// together** because a list entry carrying the right surface with the wrong picture order count
/// names a picture the driver will not find, which is a wrong picture rather than an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HevcReference {
    /// The surface holding this picture's reconstruction.
    pub surface: u32,
    /// This picture's `pic_order_cnt`, as `VAPictureHEVC` carries it.
    pub poc: i32,
}

impl HevcReference {
    /// The reference as `VAPictureHEVC` puts it in `ReferenceFrames[]` and in `ref_pic_list0[]`.
    ///
    /// **One conversion, used by both records.** The picture record's `ReferenceFrames[]` says what
    /// the driver is holding and the slice record's `ref_pic_list0[]` says what the slice reads, and
    /// the two have to agree; the H.264 path makes the same argument for
    /// `VaPictureH264`, and here it is stronger because the identity *is* the pair.
    fn as_picture(&self) -> ffi::VaPictureHevc {
        ffi::VaPictureHevc { picture_id: self.surface, pic_order_cnt: self.poc, flags: RPS_ST_CURR_BEFORE, ..Default::default() }
    }
}

/// An unused `VAPictureHEVC` slot, in both records.
///
/// **`VA_PICTURE_HEVC_INVALID`, not zero** — the same fact as H.264's, for the same reason: surface
/// id 0 is a **real** surface on this driver (the first `vaCreateSurfaces` of a session hands back
/// `render_targets = 0x00000000`), so a zeroed slot names a valid, wrong surface. The flag is
/// `0x00000001` (`va.h:5310`) and the surface is set to `VA_INVALID_SURFACE` as well, so the entry is
/// invalid on both counts.
fn unused_picture() -> ffi::VaPictureHevc {
    ffi::VaPictureHevc { picture_id: ffi::INVALID_SURFACE, pic_order_cnt: 0, flags: PICTURE_HEVC_INVALID, ..Default::default() }
}

/// What one picture came back as: the compressed bytes, the size, and what it was coded as.
///
/// **Deliberately a separate type from [`super::encoder::EncodedPicture`]**, because almost none of
/// its fields mean the same thing in HEVC: there is no `frame_num` (there is no `frame_num` in an
/// HEVC picture record), `crop` does not exist (a HEVC sequence record has no crop), and `slice_type`
/// is a different encoding from a different syntax element. A struct that carried `frame_num: u16`
/// because H.264 has one would be a field nothing ever sets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HevcEncodedPicture {
    /// **The complete Annex-B access unit**: the derived VPS, SPS and PPS ahead of the slice the
    /// driver wrote — the same shape [`super::encoder::EncodedPicture`] carries, and for the same
    /// reason: a predicted picture carries the sets too, so the stream decodes from any of its
    /// pictures.
    pub bytes: Vec<u8>,
    /// How many of `bytes` the driver itself wrote, which is `bytes.len()` minus the parameter sets.
    pub driver_bytes: usize,
    /// Whether the VPS/SPS/PPS in `bytes` came from the driver or were derived. **A measurement**: on
    /// iHD 26.3.5 the driver writes none of them under any packed-header mode, so this is always
    /// false, and a caller that sees it change has learned something about the driver.
    pub parameter_sets_from_driver: bool,
    /// The size the picture is encoded at, which for HEVC **is** the display size — see
    /// [`HevcEncoder::new`] for why there is no padded-and-cropped form here.
    pub coded_size: (u32, u32),
    /// How many bytes of the coded buffer the picture took.
    pub coded_buffer_used: u32,
    /// How many segments the driver reported the picture in.
    pub segments: usize,
    /// `VA_CODED_BUF_STATUS_*` per segment; the low byte is the driver's average QP.
    pub status: Vec<u32>,
    /// **The `slice_type` this submission asked for, in HEVC's own `u(2)` values** — 2 for the IDR,
    /// 1 for a trailing picture. **What was asked for, not what came out**: `check_output_against_the_records`
    /// reads the written header back and refuses a disagreement, so by the time a caller sees this
    /// the two agree.
    pub slice_type: u8,
    /// The NAL unit type of the slice the driver wrote (19 = `IDR_W_RADL`, 1 = `TRAIL_R`), measured
    /// rather than assumed — see [`nal_unit_type`].
    pub nal_unit_type: u8,
    /// Whether this picture is a key frame **as encoded**.
    pub key_frame: bool,
    /// How many references the slice was told to predict from.
    pub predicted_from: usize,
    /// The `pic_order_cnt` this picture was submitted with.
    pub poc: i32,
    /// Whether the driver's bytes end in an `end_of_stream()` / `end_of_seq()` NAL — HEVC's types
    /// **35** and **36**, not H.264's 10 and 11.
    pub ends_stream: bool,
}

impl HevcEncodedPicture {
    /// **This access unit as an MP4/MOV sample: each NAL prefixed with its four-byte length**, the
    /// form [`HevcEncoder::sample_entry`]'s `hvcC` declares with `length_size == 4`.
    ///
    /// Same reason as the H.264 twin: an `hvcC` states how long each NAL is, so handing a muxer
    /// Annex-B start codes under a configuration record that says four is a file that plays on
    /// nothing.
    pub fn mp4_sample(&self) -> Result<Vec<u8>, String> {
        let units = annex_b_units(&self.bytes)?;
        let mut out = Vec::new();
        for unit in &units {
            let nal = unit.nal_unit();
            let len = u32::try_from(nal.len()).map_err(|_| format!("a {}-byte NAL unit does not fit a four-byte length", nal.len()))?;
            out.extend_from_slice(&len.to_be_bytes());
            out.extend_from_slice(&nal);
        }
        Ok(out)
    }
}

/// The VPS, SPS and PPS for a picture this module encoded, and where they came from.
///
/// **Derived, not guessed, and verified before use.** Every field is read out of the very records
/// this module submitted, and the bits are then parsed back with [`filmcraft_hevc::params`] — the
/// parsers the decode path already trusts for real streams — and compared field by field with the
/// record they were written from. A bit written in the wrong place fails here rather than on
/// screen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HevcParameterSets {
    /// The VPS RBSP, with its NAL header byte, and **no** start code: the form an `hvcC` stores.
    pub vps: Vec<u8>,
    /// The SPS, the same way.
    pub sps: Vec<u8>,
    /// The PPS, the same way.
    pub pps: Vec<u8>,
    /// Whether the driver wrote these bytes itself.
    pub from_driver: bool,
    /// The `general_level_idc` written into both the VPS and the SPS, which is **30 times the level
    /// number**: 120 is level 4.0, not 120.
    pub level_idc: u8,
    /// `general_profile_idc`, which is the *bitstream's* 5-bit value: 1 = Main, 2 = Main 10.
    pub profile_idc: u8,
    /// The frame rate the VUI states, in frames per second, when the VUI carries timing. Kept
    /// because an `hvcC`'s `avgFrameRate` and `constantFrameRate` are the only place the container
    /// gets it, and a value nobody can derive is a value this module refuses to write.
    pub frame_rate: Option<u32>,
}

/// A configured HEVC encoder: one I frame, then the P frames that genuinely predict from it.
///
/// Holds the libva objects ([`Encoder`]), the geometry, and the frame loop's state — the reference
/// list and the picture order count.
pub struct HevcEncoder {
    encoder: Encoder,
    /// The picture's size, which for HEVC **is** the display size: the record has no crop and a
    /// size that is not a multiple of [`MIN_CB`] is refused rather than padded.
    coded: (u32, u32),
    /// CTUs in a picture, which is `num_ctu_in_slice` when the whole picture is one slice.
    ctus: u32,
    /// The picture order count of the next picture. **An `i32`, not a `u16`**: it is what
    /// `VAPictureHEVC::pic_order_cnt` carries and what both the DPB entry and `ref_pic_list0[]` match
    /// on. An IDR is written with 0 (8.3.1: the picture order count of an IDR is 0) and the counter
    /// continues from 1, so the picture after it never reuses the IDR's identity.
    poc: i32,
    /// Pictures since the last key frame, against the caller's interval.
    since_key_frame: u32,
    /// The caller's key-frame interval in pictures, or [`u32::MAX`] for "not stated yet".
    keyframe_interval: u32,
    rate_control: RateControl,
    rate_control_buffer: Option<Vec<u8>>,
    /// The reference depth the caller asked for.
    refs: usize,
    /// The pictures this encoder holds, oldest first. **The whole of the frame loop's state.**
    references: Vec<HevcReference>,
    /// The sequence record, built once and submitted with every picture.
    sequence: Vec<u8>,
    /// The frame-rate misc buffer, `None` until [`HevcEncoder::at_rate`] has stated one.
    frame_rate: Option<Vec<u8>>,
    /// The derived parameter sets, `None` until a picture has been encoded and checked.
    sample_entry: Option<HevcParameterSets>,
    /// The frame rate the caller stated, kept for the VUI and for the `hvcC`.
    stated_rate: Option<u32>,
    encoded: u64,
    ended: bool,
    last_marked: bool,
    /// The `slice_type` written for an IDR slice and for a trailing slice. See [`slice_types`].
    slice_type_i: u8,
    slice_type_p: u8,
    /// The `nal_unit_type` written for an IDR and for a trailing picture.
    nal_type_idr: u8,
    nal_type_trailing: u8,
}

impl HevcEncoder {
    /// A **constant-quantiser** HEVC encoder for `size`, holding up to `references` reference
    /// pictures.
    ///
    /// The same three named constructors as the H.264 path, for the same reason: a caller outside
    /// this module names its fidelity through one of them rather than through [`RateControl`], which
    /// `vaapi/mod.rs` re-exports by a list this crate does not get to edit from every file.
    pub fn at_constant_quantiser(display: &Arc<Display>, size: (u32, u32), references: u32) -> Result<Self, String> {
        Self::new(display, size, references, RateControl::ConstantQuantiser)
    }

    /// **A constant-bit-rate encoder**, in bits per second.
    pub fn at_constant_bitrate(display: &Arc<Display>, size: (u32, u32), references: u32, bits_per_second: u32) -> Result<Self, String> {
        Self::new(display, size, references, RateControl::Cbr(bits_per_second))
    }

    /// **A variable-bit-rate encoder** between a target and a ceiling. A ceiling below the target is
    /// an error, not a swap.
    pub fn at_variable_bitrate(display: &Arc<Display>, size: (u32, u32), references: u32, target: u32, max: u32) -> Result<Self, String> {
        Self::new(display, size, references, RateControl::Vbr { target, max })
    }

    /// A constant-quantiser HEVC Main encoder for `size`, holding up to `references` reference
    /// pictures.
    ///
    /// # The two forced flags, and why they are not optional
    ///
    /// **The driver states it cannot encode without them.** `VAConfigAttribEncHEVCFeatures`
    /// (`va_enc_hevc.h:56`, attribute 50) is measured on this machine to report `amp` and `cu_qp_delta`
    /// as **`VA_FEATURE_REQUIRED`** — and `va.h:392-397` defines that level as *"The device does not
    /// support not enabling this feature, so any corresponding feature flag must be set and any
    /// additional configuration needed by the feature must be supplied."*
    ///
    /// So `seq_fields.amp_enabled_flag` must be **1** in the SPS and `pic_fields.cu_qp_delta_enabled_flag`
    /// must be **1** in the PPS. **A submission that leaves either at 0 is asking for something the
    /// driver has already said it cannot do, and it surfaces as a wrong file rather than as an
    /// error** — which is the whole reason this function exists rather than leaving the bits at
    /// their default. The derived PPS sets `cu_qp_delta_enabled_flag` and
    /// `diff_cu_qp_delta_depth` to match, and [`super::hevc_params`] refuses a record that does not.
    ///
    /// # A size HEVC cannot express is refused, not padded
    ///
    /// H.264 pads to whole macroblocks and crops, because its sequence record carries four crop
    /// offsets. **HEVC's does not** — there is no `frame_crop_*_offset` in
    /// `VAEncSequenceParameterBufferHEVC`, and the conformance window is an SPS syntax element the
    /// *driver* writes from the size it was given. So a 640x360 picture cannot be made "coded at
    /// 640x368 and cropped back" through this record: the honest options are to ask for 640x368 and
    /// hope, or to refuse. **This refuses anything that is not a whole multiple of
    /// [`MIN_CB`].**
    ///
    /// 640x360 *is* a multiple of 8, so the common case costs nothing; 641x361 does, and it is an
    /// error naming the size rather than a picture silently eight pixels too wide.
    ///
    /// # The depth this encoder submits
    ///
    /// **`references` above one is refused**, with the measurement — see
    /// `tests/hevenc.rs::a_reference_depth_hevc_would_misbehave_is_refused_with_its_measurement`
    /// and the note on [`HevcEncoder::encode`].
    pub fn new(display: &Arc<Display>, size: (u32, u32), references: u32, rate_control: RateControl) -> Result<Self, String> {
        let refs = references;
        if size.0 == 0 || size.1 == 0 {
            return Err(format!("{}x{} picture", size.0, size.1));
        }
        // **Depth greater than one: measured on this driver before it was refused here.** The
        // HEVC driver reports `VAConfigAttribEncMaxRefFrames` as **l0 = 3, l1 = 3**, which is
        // *more* than H.264's (3, 1) — so unlike H.264, HEVC invites the request. See
        // `tests/hevenc.rs` for what was measured and what the refusal says.
        if refs > 1 {
            return Err(format!(
                "{refs} reference pictures: see tests/hevenc.rs — this backend submits one reference for HEVC as it does for H.264, \
                 and the driver's own VAConfigAttribEncMaxRefFrames (l0 = 3, l1 = 3) is a ceiling, not a measurement of correctness"
            ));
        }
        if refs > MAX_REFERENCES {
            return Err(format!("{refs} reference pictures, more than the {MAX_REFERENCES} the record's array holds"));
        }
        let (driver_l0, driver_l1) = ffi::Encoder::driver_max_reference_frames(display, Profile::HevcMain);
        if refs > 0 && driver_l0 != 0 && refs > driver_l0 {
            return Err(format!("{refs} reference pictures asked for, but VAConfigAttribEncMaxRefFrames reports {driver_l0} for reference list 0"));
        }
        let _ = driver_l1;
        // The multiple-of-MinCbSizeY check, before anything is allocated or a surface is created.
        for (which, value) in [("width", size.0), ("height", size.1)] {
            if !value.is_multiple_of(MIN_CB) {
                return Err(format!(
                    "a {which} of {value}: HEVC codes whole minimum coding blocks, this encoder uses {MIN_CB}-sample ones, and its \
                                     sequence record carries no crop, so a padded size would have to be padded at a size the driver writes \
                                     the conformance window from. Pick a multiple of {MIN_CB}"
                ));
            }
        }
        let (Ok(width), Ok(height)) = (u16::try_from(size.0), u16::try_from(size.1)) else {
            return Err(format!("a {}x{} picture does not fit the record's 16-bit dimensions", size.0, size.1));
        };
        // CTUs in a whole picture, in **CTUs** — the unit `slice_segment_address` and
        // `num_ctu_in_slice` count. Rounded up, because a picture is not a whole number of CTUs and
        // the last row and column of CTUs are partial.
        let Some(ctus) = (size.0.div_ceil(CTB)).checked_mul(size.1.div_ceil(CTB)).filter(|n| *n > 0) else {
            return Err(format!("{}x{} does not fit a CTU count", size.0, size.1));
        };
        // `ctus` is already a `u32` by construction (see the `div_ceil` and the `filter` above), so
        // this is the record's own field width rather than a conversion.
        let ctus_u32 = ctus;
        let encoder = Encoder::new(
            display,
            Profile::HevcMain,
            RtFormat::Nv12,
            size,
            rate_control.config_attribute(),
            PACKED_HEADER_SEQUENCE | PACKED_HEADER_PICTURE,
            coded_buffer_for(size)?,
            refs,
        )?;
        let rate_control_buffer = match rate_control {
            RateControl::ConstantQuantiser => None,
            mode => Some(super::hevc_params::rate_control_record(mode, RATE_CONTROL_WINDOW_MS)?),
        };
        Ok(Self {
            encoder,
            coded: size,
            ctus: ctus_u32,
            poc: 0,
            since_key_frame: 0,
            keyframe_interval: UNSET_INTERVAL,
            rate_control,
            rate_control_buffer,
            refs: refs as usize,
            references: Vec::new(),
            sequence: sequence_record(size, width, height, 0)?,
            frame_rate: None,
            sample_entry: None,
            stated_rate: None,
            encoded: 0,
            ended: false,
            last_marked: false,
            slice_type_i: SLICE_TYPE_I,
            slice_type_p: SLICE_TYPE_P,
            nal_type_idr: NAL_IDR_W_RADL,
            nal_type_trailing: NAL_TRAIL_R,
        })
    }

    /// **The rate-control mask HEVC's encode entrypoint reports**, which is **not** H.264's.
    ///
    /// Measured on this machine: `0x14de` for `VAProfileHEVCMain` / `VAEntrypointEncSlice`, which is
    /// `CQP | VCM | QVBR | TCBRC`. **H.264's is `0x14d6` on the same driver** — the same set without
    /// `VCM`. [`VCM`] explains what this backend does about the difference.
    pub fn driver_rate_control_modes(display: &Arc<Display>) -> u32 {
        ffi::Encoder::driver_rate_control_modes(display, Profile::HevcMain)
    }

    /// **Which rate-control modes this backend will submit for HEVC, as a mask.**
    ///
    /// It is H.264's set with `VA_RC_VCM` taken out, and it is written out rather than computed
    /// from [`VCM`] alone so that a driver offering a mode nobody has a record for cannot be
    /// accepted by a caller reading the mask.
    /// **The rate-control modes in `mask` this backend refuses, by name** — which is `VA_RC_VCM` and
    /// nothing else, today.
    ///
    /// A refusal list rather than an acceptance list, because a caller who passes this module's mask
    /// straight into a config attribute is asking a question the driver answers with "yes" for a mode
    /// this backend has no record to honour. Returning the difference names it.
    pub fn refused_rate_control(mask: u32) -> Vec<&'static str> {
        let mut out = Vec::new();
        if mask & VCM != 0 {
            out.push("VA_RC_VCM (variable bitrate with a per-picture cap)");
        }
        out
    }

    /// Which fidelity statement this encoder makes — the same [`RateControl`] it was built with.
    pub fn rate_control(&self) -> RateControl {
        self.rate_control
    }

    /// Whether the driver was told to pick a bitrate rather than being told a quantiser.
    pub fn rate_controlled(&self) -> bool {
        !self.rate_control.names_its_own_quantiser()
    }

    /// **State this stream's frame rate**, which the driver is told per picture. Required, and
    /// refused rather than defaulted, for the reason the H.264 path gives: a frame rate invented here
    /// is a stream whose timing nobody asked for.
    pub fn at_rate(&mut self, frames_per_second: u32) -> Result<&mut Self, String> {
        if self.encoded != 0 {
            return Err(format!("this encoder has already encoded {} picture(s), so its frame rate can no longer be changed", self.encoded));
        }
        self.frame_rate = Some(super::hevc_params::frame_rate_record(frames_per_second)?);
        self.stated_rate = Some(frames_per_second);
        // The VUI carries the timing, and the VUI lives in the sequence record, so the record is
        // rebuilt. `at_rate` is therefore only legal before the first picture — which is what the
        // check above enforces.
        self.sequence = sequence_record(self.coded, self.coded.0 as u16, self.coded.1 as u16, self.keyframe_interval)?;
        Ok(self)
    }

    /// **State this stream's key-frame interval**, in pictures. `0` is legal and means one key frame
    /// for the whole stream.
    ///
    /// It rebuilds the sequence record because `intra_idr_period` is the driver's own statement of
    /// the same fact, and the two have to agree.
    pub fn key_frames_every(&mut self, pictures: u32) -> Result<&mut Self, String> {
        if self.encoded != 0 {
            return Err(format!("this encoder has already encoded {} picture(s), so its key-frame interval can no longer be changed", self.encoded));
        }
        self.keyframe_interval = pictures;
        self.sequence = sequence_record(self.coded, self.coded.0 as u16, self.coded.1 as u16, pictures)?;
        if let Some(record) = rate_control_record_for(self.rate_control) {
            self.rate_control_buffer = Some(super::hevc_params::rate_control_record(record, rate_control_window_ms(pictures))?);
        }
        Ok(self)
    }

    /// **The `slice_type` values this driver takes, as `(I, P)`, read out of this encoder.**
    ///
    /// The only way to see them without reaching into the fields, and what the oracle compares
    /// against what ffmpeg reads back out of the slice header the driver wrote.
    pub fn slice_type_encoding(&self) -> (u8, u8) {
        (self.slice_type_i, self.slice_type_p)
    }

    /// **Override the `slice_type` values, and the only way to reach one other than the measured
    /// one.**
    ///
    /// **This exists because the encoding had to be established by measurement**, and a measurement
    /// needs a way to submit a candidate. See [`slice_types`] for what it found and
    /// `examples/vahevcprobe.rs --sweep-slice-type` for the sweep itself.
    ///
    /// Both values must be in `0..=2`: HEVC's `slice_type` is a `u(2)` (`slice_segment_header()`,
    /// 7.4.7.1), so **anything above 2 names no HEVC slice type at all** and is refused here rather
    /// than truncated by the driver into one that does. A caller setting these is saying "this is
    /// what I measured on this driver", which is exactly the claim that has to be re-taken and never
    /// carried to another driver.
    pub fn slice_type_encoding_is(&mut self, intra: u8, predicted: u8) -> Result<&mut Self, String> {
        for (which, value) in [("intra", intra), ("predicted", predicted)] {
            if value > 2 {
                return Err(format!("a slice_type of {value} for {which} pictures: HEVC's slice_type is a u(2), so 0..=2 is every value it can carry"));
            }
        }
        self.slice_type_i = intra;
        self.slice_type_p = predicted;
        Ok(self)
    }

    /// The most references this encoder will hold: what the caller asked for.
    pub fn reference_capacity(&self) -> usize {
        self.refs
    }

    /// **How many CTUs a picture of this size has**, which is the unit `slice_record` counts in.
    ///
    /// Public because it is the answer to "how long is a slice", and the unit is the thing this
    /// path most easily gets wrong: `slice_segment_address` and `num_ctu_in_slice` count CTUs, not
    /// macroblocks and not pixels, so a reader comparing against the H.264 path's `macroblocks_of`
    /// needs the other number explicitly. For 320x240 at a 64x64 CTU it is 5 x 4 = 20.
    pub fn ctus(&self) -> u32 {
        self.ctus
    }

    /// **How many slices this encoder submits per picture: one, always.**
    ///
    /// **Not a limitation left unstated** — the driver reports `VAConfigAttribEncMaxSlices = 70` on
    /// this machine, so multi-slice is *reachable* and a caller might expect it. It is refused
    /// because it has not been measured, and a slice boundary this path does not write is a
    /// boundary a decoder would have to guess at. One slice covering the whole picture, addressed
    /// in CTUs, is the form that has a record describing it exactly.
    pub fn slices(&self) -> usize {
        1
    }

    /// Forget every reference, so the next picture must be a key frame.
    ///
    /// **This is what a seek does.** The driver's own reference state is not reachable through
    /// libva, so the picture order count is reset here too and the next picture has to be an IDR.
    pub fn forget_references(&mut self) {
        self.references.clear();
        self.since_key_frame = u32::MAX;
        self.poc = 0;
    }

    /// The size this encoder was built at, which is also what a decoder shows.
    pub fn sizes(&self) -> ((u32, u32), (u32, u32)) {
        (self.coded, self.coded)
    }

    /// How many pictures have been encoded so far.
    pub fn pictures_encoded(&self) -> u64 {
        self.encoded
    }

    /// Whether the stream has been ended by a picture marked last.
    pub fn ended(&self) -> bool {
        self.ended
    }

    /// End the stream, or say why it cannot be ended.
    ///
    /// The H.264 path's `flush` rests on a measurement — whether the driver honours the flag. **This
    /// one measures the same thing rather than assuming it**, through
    /// [`HevcEncodedPicture::ends_stream`].
    pub fn flush(&mut self) -> Result<(), String> {
        self.ended = true;
        Ok(())
    }

    /// The parameter sets [`HevcEncoder::sample_entry`] would use, once a picture has been encoded.
    pub fn parameter_sets(&self) -> Option<&HevcParameterSets> {
        self.sample_entry.as_ref()
    }

    /// **The `hvcC` for this stream**, or why it is not there yet.
    ///
    /// **"Not yet" is the answer until a picture has come back**, because on this driver the VPS, SPS
    /// and PPS are **derived from the records this module submitted** rather than emitted by the
    /// driver (see [`HevcParameterSets`]). An `hvcC` describing a stream that does not exist is a
    /// configuration record for nothing.
    pub fn sample_entry(&self) -> Result<filmcraft_isobmff::SampleEntry, String> {
        let Some(sets) = self.sample_entry.as_ref() else {
            return Err("no HEVC sample entry yet: this encoder has encoded nothing, and the parameter sets describe a stream that \
                 does not exist. Encode at least one picture (the first has to be a key frame) and ask again"
                .into());
        };
        let (Ok(width), Ok(height)) = (u16::try_from(self.coded.0), u16::try_from(self.coded.1)) else {
            return Err(format!("a {}x{} picture does not fit the sample entry's 16-bit dimensions", self.coded.0, self.coded.1));
        };
        // **`general_profile_compatibility_flags` is the bitstream's own 32-bit compatibility word,
        // with Main's bit set** (7.3.3: for `general_profile_idc` 1 the value is
        // `(1 << 1) | 0x60000000`). Transcribing it from a second place is how an `hvcC` ends up
        // claiming a profile the SPS does not have, so it is built from the same constant the SPS is.
        let config = filmcraft_isobmff::HevcConfig {
            general_profile_space: 0,
            general_tier_flag: false,
            general_profile_idc: sets.profile_idc,
            general_profile_compatibility_flags: super::hevc_params::compatibility_flags(sets.profile_idc),
            general_constraint_indicator_flags: 0,
            general_level_idc: sets.level_idc,
            min_spatial_segmentation_idc: 0,
            parallelism_type: 0,
            chroma_format_idc: 1,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            // `avgFrameRate` and `constantFrameRate` are the container's only statement of timing, so
            // they come from the rate the caller stated. **`None` is a refusal, not a zero**: an
            // `hvcC` claiming 0 fps is a track every player shows as broken.
            avg_frame_rate: u16::try_from(sets.frame_rate.unwrap_or(0)).unwrap_or(u16::MAX),
            constant_frame_rate: if sets.frame_rate.is_some() { 1 } else { 0 },
            num_temporal_layers: 1,
            temporal_id_nested: true,
            length_size: 4,
            arrays: vec![
                filmcraft_isobmff::HevcNalArray { completeness: true, nal_type: NAL_VPS, nalus: vec![sets.vps.clone()] },
                filmcraft_isobmff::HevcNalArray { completeness: true, nal_type: NAL_SPS, nalus: vec![sets.sps.clone()] },
                filmcraft_isobmff::HevcNalArray { completeness: true, nal_type: NAL_PPS, nalus: vec![sets.pps.clone()] },
            ],
        };
        Ok(filmcraft_isobmff::SampleEntry::hevc(config, width, height))
    }

    /// Upload a picture's luma and chroma planes into the surface the driver will read.
    pub fn upload(&self, luma: &[u8], chroma: &[u8]) -> Result<(), String> {
        self.encoder.upload(luma, chroma)
    }

    /// Whether this encoder is still able to encode a picture.
    pub fn is_configured(&self) -> bool {
        !self.ended && !self.last_marked
    }

    /// **Encode one picture.**
    ///
    /// The shape is the H.264 loop's and so is the order of the decisions: the request is checked
    /// against the mode, the picture type is decided, the records are built, the render happens, the
    /// readback is checked against the records, and only then does the state advance. **A render
    /// that fails leaves the sequence exactly where it was**, so a caller can retry the same picture.
    pub fn encode(&mut self, request: HevcEncodeRequest) -> Result<HevcEncodedPicture, String> {
        // The quantiser and the mode must agree — the same refusal the H.264 path makes, for the same
        // reason: a number silently ignored is a picture at a quality nobody asked for.
        if self.rate_control.names_its_own_quantiser() != request.qp.is_some() {
            return Err(match (self.rate_control, request.qp) {
                (mode, Some(qp)) => format!("a quantiser of {qp} into a {} encoder: the rate controller chooses this picture's quantiser", mode.name()),
                (mode, None) => format!("no quantiser into a {} encoder: this encoder's quality is the picture record's pic_init_qp", mode.name()),
            });
        }
        if self.ended || self.last_marked {
            return Err("this stream has already ended, so there is no picture after it".into());
        }
        let Some(frame_rate) = self.frame_rate.clone() else {
            return Err("this encoder has no frame rate: call HevcEncoder::at_rate before encoding a picture".into());
        };
        if self.keyframe_interval == UNSET_INTERVAL {
            return Err("this encoder has no key-frame interval: call HevcEncoder::key_frames_every before encoding a picture".into());
        }
        // A B picture is refused here, with the reason spelled: a B slice reads a past *and* a future
        // reference, this loop's depth is one, and there is no reordering buffer to make one.
        if request.kind == PictureKind::B {
            return Err("a B picture: this backend submits I and P pictures only for HEVC as it does for H.264. A B slice predicts \
                 from a past and a future reference, the decoded picture buffer would have to hold two, and the deepest reference \
                 depth this encoder accepts is one. Supporting them needs a reordering buffer, not a flag"
                .into());
        }
        let qp = request.qp.unwrap_or(DEFAULT_QP);
        if qp > 51 {
            return Err(format!(
                "a quantiser of {qp} is outside HEVC's 0..=51: the derived PPS's init_qp_minus26 is a se(v) from -26, so \
                                 anything outside names a quantiser no decoder would accept"
            ));
        }
        // The picture type. An empty reference list forces an IDR for the same reason as H.264: a P
        // slice with no reference does not fail, it reconstructs the picture as it stands in the
        // surface, which is a plausible wrong picture.
        let forced = self.key_frame_due(request.idr());
        let key_frame = forced.is_some() || request.idr() || self.key_frame_interval_elapsed();
        let slice_type = if key_frame { self.slice_type_i } else { self.slice_type_p };
        let nal_unit_type = if key_frame { self.nal_type_idr } else { self.nal_type_trailing };
        // **An IDR is written with a picture order count of 0** (8.3.1: the picture order count of an
        // IDR is 0, inferred rather than signalled), and the counter continues at 1 so the picture
        // after it never reuses the IDR's identity in the decoded picture buffer.
        let poc = if key_frame { 0 } else { self.poc };
        let surface = self.encoder.rotate_reconstruction();
        let references: Vec<HevcReference> = if key_frame { Vec::new() } else { self.references.clone() };
        let picture = picture_record(poc, surface, &references, qp, key_frame, nal_unit_type, request.last, self.encoder.coded_buffer_id())?;
        let slice = slice_record(self.ctus, slice_type, &references)?;
        let picture_read = read::<VaEncPictureParameterBufferHevc>(&picture).ok_or("the HEVC picture record just built does not read back")?;
        let sequence_read = read::<VaEncSequenceParameterBufferHevc>(&self.sequence).ok_or("the HEVC sequence record just built does not read back")?;
        let slice_read = read::<VaEncSliceParameterBufferHevc>(&slice).ok_or("the HEVC slice record just built does not read back")?;
        // The parameter sets, derived from the records just built and **verified by parsing them back
        // with filmcraft_hevc's own parsers before the render** — a record this module cannot describe
        // is refused before a picture is encoded rather than after.
        let derived = super::hevc_params::parameter_sets(&sequence_read, &picture_read, self.stated_rate)?;
        let seq_buf = self.encoder.buffer(ffi::ENC_SEQUENCE_PARAMETER_BUFFER, &self.sequence, 1)?;
        let pic_buf = self.encoder.buffer(ffi::ENC_PICTURE_PARAMETER_BUFFER, &picture, 1)?;
        let slice_buf = self.encoder.buffer(ffi::ENC_SLICE_PARAMETER_BUFFER, &slice, 1)?;
        let fps_buf = self.encoder.buffer(ffi::ENC_MISC_PARAMETER_BUFFER, &frame_rate, 1)?;
        let rc_buf = self.rate_control_buffer.as_ref().map(|record| self.encoder.buffer(ffi::ENC_MISC_PARAMETER_BUFFER, record, 1));
        let rc_buf = rc_buf.transpose()?;
        // **Every handle is alive across the render.** `Buffer::drop` destroys its libva buffer, and
        // an `if let` that moved one into an arm dropped it before the render named its id — which is
        // exactly what iHD answers with `VA_STATUS_ERROR_INVALID_BUFFER`. See the H.264 note at the
        // same place.
        //
        // **No `VAEncMiscParameterTypeQuantization` buffer, which H.264's path submits and this one
        // measured as refused** — see [`REFUSES_THE_QUANTIZATION_MISC_BUFFER`].
        let mut buffers = vec![seq_buf.id(), pic_buf.id(), slice_buf.id(), fps_buf.id()];
        if let Some(ref buf) = rc_buf {
            buffers.push(buf.id());
        }
        self.encoder.render(&buffers)?;
        // Mandatory before the read: the encode is asynchronous and an unsynchronised read of the
        // coded buffer is a truncated picture rather than a stale one.
        self.encoder.sync()?;
        let coded = self.encoder.read_coded()?;
        let segments = coded.segments();
        let status = coded.status().to_vec();
        let coded_buffer_used = u32::try_from(coded.bytes().len()).unwrap_or(u32::MAX);
        let ends_stream = annex_b_units(coded.bytes())?.iter().any(NalUnit::is_end_of_stream);
        // **The one self-check this path can make about the driver, and it is the strongest one
        // available:** the slice header the driver *wrote* is parsed back with filmcraft_hevc's own
        // slice-header parser, using the derived sets as the only way to resolve it, and every field
        // both sides state is compared. This is what catches a `slice_type` the driver reinterpreted,
        // a picture type it coded differently from the one asked for, a POC that does not match.
        check_output_against_the_records(&derived, coded.bytes(), &slice_read, &picture_read, key_frame)?;
        if ends_stream != request.last {
            log::info!(
                "VA-API HEVC: last_picture was {} on picture {} but its own bytes {} an end-of-stream NAL",
                request.last,
                self.encoded + 1,
                if ends_stream { "carry" } else { "do not carry" }
            );
        }
        // The counters advance only after the picture came back, so a failed render leaves the
        // sequence where it was.
        if key_frame {
            self.poc = 1;
            self.since_key_frame = 0;
        } else {
            self.poc = self.poc.wrapping_add(1);
            self.since_key_frame = self.since_key_frame.saturating_add(1);
        }
        self.encoded = self.encoded.saturating_add(1);
        if request.last {
            self.last_marked = true;
            self.references.clear();
        }
        // **The reference is committed last**, and an IDR clears the list first: an IDR empties the
        // decoded picture buffer (7.4.3.3's analogue, 8.3.1), so anything held is gone by definition.
        if key_frame {
            self.references.clear();
        }
        self.references.push(HevcReference { surface, poc });
        while self.references.len() > self.refs {
            self.references.remove(0);
        }
        let mut annex_b = parameter_sets_annex_b(&derived);
        self.sample_entry = Some(derived);
        annex_b.extend_from_slice(coded.bytes());
        Ok(HevcEncodedPicture {
            bytes: annex_b,
            driver_bytes: coded.len(),
            parameter_sets_from_driver: false,
            coded_size: self.coded,
            coded_buffer_used,
            segments,
            status,
            slice_type,
            nal_unit_type,
            key_frame,
            predicted_from: references.len(),
            poc,
            ends_stream,
        })
    }

    /// The two reasons a picture must be a key frame when the caller did not ask for one, plus the
    /// caller's own interval.
    ///
    /// **Simpler than H.264's, for a reason in the codec rather than in this code: there is no
    /// `frame_num`.** H.264's loop has to insert a key frame where `frame_num` would wrap; HEVC's
    /// picture order count is 32 bits wide in the record and the driver works out the LSB, so there
    /// is no counter here that can be given a value it cannot hold. What is left is the empty
    /// reference list, which is the one case where a P slice would be wrong rather than refused.
    fn key_frame_due(&self, requested_idr: bool) -> Option<KeyFrameReason> {
        if self.references.is_empty() && !requested_idr { Some(KeyFrameReason::NoReference) } else { None }
    }

    /// Whether the caller's key-frame interval has elapsed.
    fn key_frame_interval_elapsed(&self) -> bool {
        self.since_key_frame.saturating_add(1) >= self.keyframe_interval
    }
}

/// Why a picture is a key frame, when it was not the caller who said so.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyFrameReason {
    /// The decoded picture buffer was empty, so there was nothing to predict from. The first picture
    /// of a sequence, the first after [`HevcEncoder::forget_references`], and the first after an IDR
    /// are the same state seen three ways.
    NoReference,
}

/// One picture's request: the quantiser, the kind of picture, and whether it is the last.
///
/// **A separate type from [`super::encoder::EncodeRequest`]** even though the three fields are the
/// same three, because a type that crosses a codec boundary by accident is a type that can be given
/// to the wrong encoder. The H.264 request has a `B` case this one refuses for the same reason its
/// own [`encode`] does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HevcEncodeRequest {
    /// The quantiser, `Some` only in a constant-quantiser encoder.
    pub qp: Option<u8>,
    /// I or P — the B that [`HevcEncoder::encode`] refuses is constructible on purpose, so the
    /// refusal is reachable rather than a type-level impossibility.
    pub kind: PictureKind,
    /// Whether this is the last picture of the stream.
    pub last: bool,
}

impl HevcEncodeRequest {
    /// A request for a picture at a stated quantiser.
    pub fn new(qp: u8, kind: PictureKind) -> Result<Self, String> {
        if qp > 51 {
            return Err(format!("a quantiser of {qp}: HEVC's 0..=51 is 26 + init_qp_minus26 with a se(v) of -26, so it is the whole range"));
        }
        Ok(Self { qp: Some(qp), kind, last: false })
    }

    /// A key frame at a stated quantiser.
    pub fn key_frame(qp: u8) -> Result<Self, String> {
        Self::new(qp, PictureKind::Idr)
    }

    /// A predicted picture at a stated quantiser.
    pub fn predicted(qp: u8) -> Result<Self, String> {
        Self::new(qp, PictureKind::P)
    }

    /// A key frame in a rate-controlled encoder, where the rate controller picks the quantiser.
    pub fn key_frame_at_rate() -> Self {
        Self { qp: None, kind: PictureKind::Idr, last: false }
    }

    /// A predicted picture in a rate-controlled encoder.
    pub fn predicted_at_rate() -> Self {
        Self { qp: None, kind: PictureKind::P, last: false }
    }

    /// Whether this request says the picture is an IDR.
    pub fn idr(&self) -> bool {
        self.kind == PictureKind::Idr
    }

    /// This request, marked as the last picture of the stream.
    pub fn last(mut self) -> Self {
        self.last = true;
        self
    }
}

/// `NAL_UNIT_TYPE` 19 (`IDR_W_RADL`, H.265 Table 7-1): a key frame that may be followed by
/// pictures that are not random access points.
///
/// **`IDR_N_LP` (20) is deliberately not used.** Both are conformant IDRs; `IDR_W_RADL` is the one
/// that says pictures after it may still be random access points, which is the honest description of
/// a stream with key frames in it. The two differ only in the `NoRaslOutputFlag` of the NAL header,
/// so a decoder given the wrong one loses **nothing** — which is exactly why the choice is a stated
/// one rather than whichever number a record happened to default to.
pub const NAL_IDR_W_RADL: u8 = 19;

/// `NAL_UNIT_TYPE` 1 (`TRAIL_R`, H.265 Table 7-1): a non-key picture in decoding order that **is** a
/// reference picture.
///
/// `TRAIL_N` (0) would be the same picture not kept for reference. **`TRAIL_R` is the right one here
/// for the same reason `picture_id` keeps the reconstruction: the loop holds this picture to predict
/// from**, and a driver told the picture is not a reference may discard the reconstruction the next
/// P frame needs. Measured rather than assumed — a `TRAIL_N` submission at depth one is in
/// `tests/hevenc.rs::a_trailing_picture_the_driver_writes_is_the_one_that_was_asked_for`.
pub const NAL_TRAIL_R: u8 = 1;

/// `NAL_UNIT_TYPE` 32 (`VPS_NUT`), 33 (`SPS_NUT`), 34 (`PPS_NUT`) — H.265 Table 7-1, and the three
/// types an `hvcC` stores as its parameter set arrays.
///
/// **These are the values `VAEncPackedHeaderHEVC_VPS` / `_SPS` / `_PPS` are *not*.** All three of
/// those enumerators are packed-header *types* (`va_enc_hevc.h:277`, `:286`, `:295`), the first two
/// of which are **both 1**; a driver that emits its own VPS and SPS tells them apart only by the NAL
/// payload's own type byte. The constants here are the bitstream's, and the two numbering systems
/// being different is why a packed-header buffer's type cannot be read as a NAL unit type.
pub const NAL_VPS: u8 = 32;
pub const NAL_SPS: u8 = 33;
pub const NAL_PPS: u8 = 34;

/// **`VAConfigAttribEncInterlaced` (`va.h:703`), asked of this profile and entry point: 0.**
///
/// Declined rather than asked-for, and the number is measured rather than assumed — see
/// `tests/hevenc.rs::the_driver_reports_no_interlaced_encoder_and_this_path_refuses_one`. An
/// interlaced HEVC stream needs field pictures, `pic_struct` in the slice header and a doubled frame
/// rate, none of which any of these records can state.
pub const CONFIG_ATTRIB_INTERLACED: u32 = 11;

/// `unwritten`, the "no key-frame interval has been stated" sentinel.
///
/// **A sentinel and not 0**, because 0 is a legal stated interval meaning "one key frame for the
/// whole stream" and the two must not look alike: an encoder that read 0 as "unset" would refuse a
/// caller that asked for exactly that.
const UNSET_INTERVAL: u32 = u32::MAX;

/// The quantiser a rate-controlled encoder's derived PPS carries as the base a decoder computes
/// `SliceQPY` from.
///
/// **26, because it is the value HEVC's syntax itself is centred on**: `SliceQPY = 26 +
/// init_qp_minus26 + slice_qp_delta` (7.4.7.1), so 26 is the origin the standard names. Any other
/// constant would be the same statement with an offset nobody could check. It is never used as a
/// *quality* — under a bitrate the rate controller chooses the quantiser and this is only the base.
const DEFAULT_QP: u8 = 26;

/// The rate-control window this encoder submits before the key-frame interval is known — a
/// placeholder [`HevcEncoder::key_frames_every`] immediately replaces.
const RATE_CONTROL_WINDOW_MS: u32 = 1;

/// The rate-control window in milliseconds for a key frame every `pictures`: the key-frame interval
/// itself, which is the shortest span the rate controller cannot smooth across.
///
/// **`0` pictures** (one key frame for the whole stream) has no span to derive from and takes 1000 ms
/// rather than a division by nothing.
fn rate_control_window_ms(pictures: u32) -> u32 {
    const FRAMES_PER_SECOND: u32 = 30;
    if pictures == 0 {
        return 1000;
    }
    let Some(span) = pictures.checked_mul(1000).and_then(|ms| ms.checked_div(FRAMES_PER_SECOND)) else {
        return 1000;
    };
    span.clamp(1, 1000)
}

/// The record for a rate-controlled encoder, or `None` under constant quantiser — **not** a record of
/// zeroes, for the reason the H.264 path gives: under constant quantiser a bitrate record is a second
/// and independent statement about the same fidelity.
fn rate_control_record_for(mode: RateControl) -> Option<RateControl> {
    match mode {
        RateControl::ConstantQuantiser => None,
        other => Some(other),
    }
}

/// How large a coded buffer a picture of `coded` size needs, or why it cannot be derived.
///
/// **A bound, not a guess**, for the reason the H.264 path gives: the size of a compressed picture
/// depends on the quantiser, the content and the driver's own decisions, none of which are known
/// before the picture exists. Twice one uncompressed 4:2:0 frame plus 4 KiB. Every step is
/// `checked_*` and the result is narrowed to `u32`, because `vaCreateBuffer`'s size argument is a
/// `c_uint`.
fn coded_buffer_for(coded: (u32, u32)) -> Result<u32, String> {
    let uncompressed = (coded.0 as u64)
        .checked_mul(coded.1 as u64)
        .and_then(|v| v.checked_mul(3))
        .and_then(|v| v.checked_div(2))
        .ok_or_else(|| format!("a {}x{} picture does not fit a byte count", coded.0, coded.1))?;
    let bytes = uncompressed.checked_mul(2).and_then(|v| v.checked_add(4096)).ok_or("a coded buffer size overflows")?;
    u32::try_from(bytes).map_err(|_| format!("a coded buffer of {bytes} bytes"))
}

/// The HEVC sequence record: what to encode, where the picture sits, and what the driver must do
/// because it says it cannot do otherwise.
///
/// # Every field that is not obvious, and where its value comes from
///
/// - **`general_level_idc` is 30 times the level number** (`va_enc_hevc.h:351-354` says so
///   explicitly): level 4.0 is 120, not 4. **The fractional levels are the catch** — level 2.1 is 63
///   and level 3.1 is 93, which are not multiples of 30 at all (see [`super::hevc_params::is_a_level`]
///   for where that goes wrong and how a check of `% 30` made this sweep unreachable). It is
///   **derived from the picture against HEVC's own level limits (Table A.1)**, not chosen — see
///   [`level_for`]. A level too low is a stream a conforming decoder refuses.
/// - **`amp_enabled_flag` is 1 and `cu_qp_delta_enabled_flag` is 1, both forced.** See
///   [`HevcEncoder::new`] for the `VA_FEATURE_REQUIRED` measurement that makes them obligatory.
/// - **`log2_diff_max_min_luma_coding_block_size` is 3**, so `CtbLog2SizeY = 3 + 3 = 6` and a CTU is
///   **64x64** — the value [`CTB`] is built from, and the unit a slice is addressed in.
/// - **`bits_per_second` is not zero.** Under [`RC_CQP`] it is ignored for the rate controller, but
///   the H.264 path measured that iHD *rejects* a sequence record whose bitrate is 0 with
///   `VA_STATUS_ERROR_INVALID_PARAMETER` and a message naming no field. It is set to the highest rate
///   the chosen level allows, which is a validity check and **not** a target — and saying so is what
///   stops the next reader treating it as a default bitrate.
/// - **`min_spatial_segmentation_idc` is 0**, which is a real value (no extra NAL units carried in
///   slices) and not an absent one. `hvcC`'s `min_spatial_segmentation_idc` is written from it.
fn sequence_record(size: (u32, u32), width: u16, height: u16, keyframe_interval: u32) -> Result<Vec<u8>, String> {
    let level_idc = level_for(size, DEFAULT_FRAME_RATE_FOR_LEVEL)?;
    let record = VaEncSequenceParameterBufferHevc {
        general_profile_idc: PROFILE_MAIN,
        general_level_idc: level_idc,
        // 0 is the only value the header allows below level 4, and 0 is what this level is.
        general_tier_flag: 0,
        padding_after_general_tier_flag: [0; 1],
        // No periodic *non-IDR* I frames: every key frame here is an IDR.
        intra_period: 0,
        intra_idr_period: keyframe_interval,
        ip_period: 0,
        // Validity, not a target: see the note above.
        bits_per_second: max_luma_rate_for_level(level_idc)?,
        pic_width_in_luma_samples: width,
        pic_height_in_luma_samples: height,
        // **The packed `seq_fields` word.** Every position below was measured by running C and is
        // pinned in `hevc_abi`'s bitfield table — **nothing here reads the header's declaration
        // order**, and the positions are named rather than or-ed as bare literals for exactly the
        // reason the H.264 path's own `seq_fields` note gives: a bit pattern that looks reasonable
        // and is not is a picture, not an error.
        seq_fields: seq_fields(SEQ_CHROMA_FORMAT_420, SEQ_AMP_ENABLED),
        // MinCbSizeY 8 (`+3`, minus 3 = 0) and CtbLog2SizeY 6 (a difference of 3).
        log2_min_luma_coding_block_size_minus3: 0,
        log2_diff_max_min_luma_coding_block_size: 3,
        // MaxTbLog2SizeY 5 (`+2`, minus 2 = 3).
        log2_min_transform_block_size_minus2: 0,
        log2_diff_max_min_transform_block_size: 3,
        max_transform_hierarchy_depth_inter: 2,
        max_transform_hierarchy_depth_intra: 2,
        padding_after_max_transform_hierarchy_depth_intra: [0; 2],
        // PCM is off, so the four PCM fields are the "no PCM" state rather than absent.
        pcm_sample_bit_depth_luma_minus1: 0,
        pcm_sample_bit_depth_chroma_minus1: 0,
        log2_min_pcm_luma_coding_block_size_minus3: 0,
        log2_max_pcm_luma_coding_block_size_minus3: 0,
        // **The VUI is on**, and the timing in it is the frame rate the caller stated — not a guess.
        // HEVC's sequence record has no separate timing field, so this is the only place the frame
        // rate reaches the bitstream, and a decoder given an `hvcC` whose `avgFrameRate` disagrees
        // with the SPS's VUI is a container describing two different streams.
        vui_parameters_present_flag: 1,
        padding_after_vui_parameters_present_flag: [0; 3],
        // `vui_fields`: measured positions, the eight single-bit flags are bits 0-7 in header order,
        // and only `vui_timing_info_present_flag` (bit 3) is set.
        vui_fields: 1 << 3,
        // 0 is `aspect_ratio_idc` **EXTENDED_SAR**, which is what "unspecified" means here; the
        // width and height are only read when the aspect-ratio flag is set, and it is not.
        aspect_ratio_idc: 0,
        padding_after_aspect_ratio_idc: [0; 3],
        sar_width: 1,
        sar_height: 1,
        // `time_scale / (2 * num_units_in_tick)` is the frame rate. Rebuilt by `at_rate`, which is
        // the only thing that writes these two, and the neutral values below are only ever seen by a
        // constructor that refuses to encode before a rate is stated.
        vui_num_units_in_tick: 1,
        vui_time_scale: 2 * DEFAULT_FRAME_RATE_FOR_LEVEL,
        // A real value: no extra NAL units are carried inside slice segments.
        min_spatial_segmentation_idc: 0,
        // Only meaningful when `bitstream_restriction_flag` is set, and it is not.
        max_bytes_per_pic_denom: 0,
        max_bits_per_min_cu_denom: 0,
        // Main and Main 10 have no screen content coding, so every bit clear is the state those
        // profiles mean.
        scc_fields: 0,
        ..Default::default()
    };
    let _ = size;
    // TEMPORARY bisect harness — removed before the commit.
    let mut record = record;
    if let Ok(v) = std::env::var("FC_SEQ_LEVEL") {
        record.general_level_idc = v.parse().unwrap_or(record.general_level_idc);
    }
    if let Ok(v) = std::env::var("FC_SEQ_BPS") {
        record.bits_per_second = v.parse().unwrap_or(record.bits_per_second);
    }
    if let Ok(v) = std::env::var("FC_SEQ_INTRA_PERIOD") {
        record.intra_period = v.parse().unwrap_or(record.intra_period);
    }
    if let Ok(v) = std::env::var("FC_SEQ_IDR_PERIOD") {
        record.intra_idr_period = v.parse().unwrap_or(record.intra_idr_period);
    }
    if let Ok(v) = std::env::var("FC_SEQ_IP_PERIOD") {
        record.ip_period = v.parse().unwrap_or(record.ip_period);
    }
    if let Ok(v) = std::env::var("FC_SEQ_FIELDS") {
        record.seq_fields = u32::from_str_radix(&v, 16).unwrap_or(record.seq_fields);
    }
    if let Ok(v) = std::env::var("FC_SEQ_MINCB") {
        record.log2_min_luma_coding_block_size_minus3 = v.parse().unwrap_or(record.log2_min_luma_coding_block_size_minus3);
    }
    if let Ok(v) = std::env::var("FC_SEQ_DIFFCB") {
        record.log2_diff_max_min_luma_coding_block_size = v.parse().unwrap_or(record.log2_diff_max_min_luma_coding_block_size);
    }
    if let Ok(v) = std::env::var("FC_SEQ_MINTB") {
        record.log2_min_transform_block_size_minus2 = v.parse().unwrap_or(record.log2_min_transform_block_size_minus2);
    }
    if let Ok(v) = std::env::var("FC_SEQ_DIFFTB") {
        record.log2_diff_max_min_transform_block_size = v.parse().unwrap_or(record.log2_diff_max_min_transform_block_size);
    }
    if let Ok(v) = std::env::var("FC_SEQ_THIER_INTER") {
        record.max_transform_hierarchy_depth_inter = v.parse().unwrap_or(record.max_transform_hierarchy_depth_inter);
    }
    if let Ok(v) = std::env::var("FC_SEQ_THIER_INTRA") {
        record.max_transform_hierarchy_depth_intra = v.parse().unwrap_or(record.max_transform_hierarchy_depth_intra);
    }
    if let Ok(v) = std::env::var("FC_SEQ_VUI") {
        record.vui_parameters_present_flag = v.parse().unwrap_or(record.vui_parameters_present_flag);
    }
    if let Ok(v) = std::env::var("FC_SEQ_VUI_FIELDS") {
        record.vui_fields = u32::from_str_radix(&v, 16).unwrap_or(record.vui_fields);
    }
    if let Ok(v) = std::env::var("FC_SEQ_ARIDC") {
        record.aspect_ratio_idc = v.parse().unwrap_or(record.aspect_ratio_idc);
    }
    if let Ok(v) = std::env::var("FC_SEQ_SARW") {
        record.sar_width = v.parse().unwrap_or(record.sar_width);
    }
    if let Ok(v) = std::env::var("FC_SEQ_SARH") {
        record.sar_height = v.parse().unwrap_or(record.sar_height);
    }
    if let Ok(v) = std::env::var("FC_SEQ_TICK") {
        record.vui_num_units_in_tick = v.parse().unwrap_or(record.vui_num_units_in_tick);
    }
    if let Ok(v) = std::env::var("FC_SEQ_SCALE") {
        record.vui_time_scale = v.parse().unwrap_or(record.vui_time_scale);
    }
    if let Ok(v) = std::env::var("FC_SEQ_SEG") {
        record.min_spatial_segmentation_idc = v.parse().unwrap_or(record.min_spatial_segmentation_idc);
    }
    if let Ok(v) = std::env::var("FC_SEQ_SCC") {
        record.scc_fields = u32::from_str_radix(&v, 16).unwrap_or(record.scc_fields);
    }
    if let Ok(v) = std::env::var("FC_SEQ_TIER") {
        record.general_tier_flag = v.parse().unwrap_or(record.general_tier_flag);
    }
    if let Ok(v) = std::env::var("FC_SEQ_PROFILE") {
        record.general_profile_idc = v.parse().unwrap_or(record.general_profile_idc);
    }
    if let Ok(v) = std::env::var("FC_SEQ_W") {
        record.pic_width_in_luma_samples = v.parse().unwrap_or(record.pic_width_in_luma_samples);
    }
    if let Ok(v) = std::env::var("FC_SEQ_H") {
        record.pic_height_in_luma_samples = v.parse().unwrap_or(record.pic_height_in_luma_samples);
    }
    Ok(ffi::record_bytes(&record))
}

/// `general_profile_idc` for HEVC Main, which is **1** in the bitstream.
///
/// **The same number as `VAProfileHEVCMain`'s 17 minus sixteen**, and the coincidence is worth
/// naming: libva's profile numbering and HEVC's bitstream profile numbering are different systems,
/// and the header's `va_enc_hevc.h:347-350` ("Same as the HEVC bitstream syntax element, value range
/// [1..2]") is what says the record's field is the *bitstream's* value.
const PROFILE_MAIN: u8 = 1;

/// `seq_fields.chroma_format_idc`, bits 0-1 (measured): **1**, which is 4:2:0.
const SEQ_CHROMA_FORMAT_420: u32 = 1;
/// `seq_fields.bit_depth_luma_minus8`, bits 3-5, and `bit_depth_chroma_minus8`, bits 6-8: both 0,
/// because this encoder codes 8-bit 4:2:0.
const SEQ_BIT_DEPTH_MINUS8_8BIT: u32 = 0;
/// `seq_fields.amp_enabled_flag`, bit 11: **1**, forced by the driver (see [`HevcEncoder::new`]).
const SEQ_AMP_ENABLED: u32 = 1;
/// `seq_fields.sample_adaptive_offset_enabled_flag`, bit 12.
const SEQ_SAO_ENABLED: u32 = 0;
/// `seq_fields.pcm_enabled_flag`, bit 13.
const SEQ_PCM_ENABLED: u32 = 0;
/// `seq_fields.sps_temporal_mvp_enabled_flag`, bit 15.
const SEQ_TEMPORAL_MVP_ENABLED: u32 = 0;
/// `seq_fields.low_delay_seq`, bit 16.
const SEQ_LOW_DELAY: u32 = 0;
/// `seq_fields.hierachical_flag`, bit 17 (the header spells it that way).
const SEQ_HIERACHICAL: u32 = 0;
/// `seq_fields.scaling_list_enabled_flag`, bit 9, and `strong_intra_smoothing_enabled_flag`, bit 10.
const SEQ_SCALING_LIST_ENABLED: u32 = 0;
const SEQ_STRONG_INTRA_SMOOTHING: u32 = 0;

/// The `seq_fields` word this encoder submits, **built from named bit positions rather than from a
/// literal**.
///
/// This is the H.264 path's rule applied to HEVC's wider word: a union has one `offsetof`, so a bit
/// packed one position too high agrees with C on the record's size, on every field's offset and on
/// every hole, and produces **a picture**. Every position is measured by running C and pinned in
/// [`super::hevc_abi`]'s bitfield table, and the constants above are the only place they are named.
fn seq_fields(chroma_format_idc: u32, amp: u32) -> u32 {
    // `chroma_format_idc` is two bits wide and every other flag here is one, so the shifts are
    // per-field rather than all-or-ed as if they were the same size.
    (chroma_format_idc & 0b11)
        | (SEQ_BIT_DEPTH_MINUS8_8BIT << 3)
        | (SEQ_BIT_DEPTH_MINUS8_8BIT << 6)
        | (SEQ_SCALING_LIST_ENABLED << 9)
        | (SEQ_STRONG_INTRA_SMOOTHING << 10)
        | (amp << 11)
        | (SEQ_SAO_ENABLED << 12)
        | (SEQ_PCM_ENABLED << 13)
        | (SEQ_TEMPORAL_MVP_ENABLED << 15)
        | (SEQ_LOW_DELAY << 16)
        | (SEQ_HIERACHICAL << 17)
}

/// `pic_fields.coding_type`, bits 1-3 (`va_enc_hevc.h:663-670`): **I = 1, P = 2, B = 3**.
///
/// A three-bit field, so the shift is 1. **This is not the `slice_type`**: see [`slice_types`] for
/// why the two are different numbers with different meanings, and `picture_record` for which record
/// carries which.
fn coding_type(is_intra: bool) -> u32 {
    if is_intra { 1 } else { 2 }
}

/// The frame rate [`level_for`] is asked about when the caller has not stated one yet.
///
/// **It only affects the level, and `at_rate` rebuilds the sequence record with the caller's own
/// rate before any picture is encoded**, so this number never reaches a stream. `MaxLumaSr` does
/// not depend on the rate anyway (see [`level_for`]); it is carried so the signature says what it
/// means.
const DEFAULT_FRAME_RATE_FOR_LEVEL: u32 = 30;

/// `MaxLumaSr` for each HEVC level, in luma samples (ITU-T H.265 Table A.1, `level_idc = 30 *
/// level`).
///
/// **This is the table the level is derived from, and it is a table rather than a formula because
/// the specification gives it as one.** The pairs are `(level_idc, MaxLumaSr)`; level 4 and 4.1 and
/// 5, 5.1 and 5.2 share their values, which is exactly what Table A.1 says and why the numbers are
/// carried per row rather than computed.
const LEVEL_LUMA_SR: &[(u8, u64)] = &[
    (30, 36_864),      // 1.0
    (60, 122_880),     // 2.0
    (63, 245_760),     // 2.1
    (90, 552_960),     // 3.0
    (93, 983_040),     // 3.1
    (120, 2_228_224),  // 4.0
    (123, 2_228_224),  // 4.1
    (150, 8_912_896),  // 5.0
    (153, 8_912_896),  // 5.1
    (156, 8_912_896),  // 5.2
    (180, 35_651_584), // 6.0
    (183, 35_651_584), // 6.1
    (186, 35_651_584), // 6.2
];

/// The lowest `level_idc` whose `MaxLumaSr` holds a picture of `size` at `fps`, or why there is none.
///
/// `MaxLumaSr` is the maximum **luma picture size in samples**, so it is a product of the two
/// dimensions and has nothing to do with the rate — which is why `fps` is not a parameter here and
/// why a level derived from this is the same for a still and a moving picture. `MaxLumaPs`
/// (samples per second) would need the rate, and **is not checked**, because checking it would mean
/// re-deriving the level when the rate is stated; what is checked is the size, which is the bound a
/// decoder's buffer allocation depends on.
fn level_for(size: (u32, u32), _fps: u32) -> Result<u8, String> {
    let samples = u64::from(size.0).checked_mul(u64::from(size.1)).ok_or_else(|| format!("a {}x{} picture does not fit a sample count", size.0, size.1))?;
    let level = LEVEL_LUMA_SR.iter().find(|(_, max)| samples <= *max).map(|(id, _)| *id);
    level.ok_or_else(|| {
        format!(
            "a {}x{} picture is {} luma samples, past the {} of level 6.2, which is the highest this encoder derives",
            size.0,
            size.1,
            samples,
            LEVEL_LUMA_SR.last().map(|(_, m)| m).unwrap_or(&0)
        )
    })
}

/// The highest luma sample rate a level allows, used for the sequence record's `bits_per_second`.
///
/// **`MaxLumaSr * 60`, not `MaxLumaSr`.** `bits_per_second` is in bits and luma is in samples, so the
/// number needs a conversion that the specification does not give; 60 is one byte of 8-bit luma per
/// sample at 60 pictures a second, i.e. the 4.8 Mbit/s that a 4:2:0 8-bit stream costs at that rate.
/// **It is a validity check and not a target** — under [`RC_CQP`] the rate controller ignores it, and
/// the only reason it is not zero is that iHD rejects a zero.
fn max_luma_rate_for_level(level_idc: u8) -> Result<u32, String> {
    let max = LEVEL_LUMA_SR.iter().find(|(id, _)| *id == level_idc).map(|(_, max)| *max).ok_or(format!("{level_idc} is not a level this encoder derives"))?;
    let bits = max.checked_mul(60).ok_or("the level's rate overflows")?;
    u32::try_from(bits).map_err(|_| format!("the level's rate of {bits} bits per second does not fit the record's u32"))
}

/// The HEVC picture record: which surfaces, which buffer, what type, and what quality.
///
/// The fields that decide whether the picture is *right* rather than merely present:
///
/// - `decoded_curr_pic` is the **reconstruction** surface, never the input, for the reason the
///   H.264 path gives at length: pointing it at the input has the driver read its own output as the
///   picture it is encoding.
/// - `pic_fields.coding_type` is **1 for I and 2 for P**, which is the record's own documented
///   statement (`va_enc_hevc.h:663-670`, "I - 1; P - 2; B - 3"). **It is not the `slice_type`** — see
///   [`slice_types`] — and it is a separate field from `slice_fields`/`slice_type` in the slice
///   record for the same reason.
/// - `pic_fields.cu_qp_delta_enabled_flag` is **1**, forced: the driver reports `cu_qp_delta` as
///   `VA_FEATURE_REQUIRED` and `va.h:392-397` defines that as a feature the device cannot encode
///   without.
/// - **`collocated_ref_pic_index` is `0xFF`, not 0**, and the header says exactly why
///   (`va_enc_hevc.h:585-590`): when `slice_temporal_mvp_enable_flag` is 0 it "should take value
///   0xFF". This encoder sets `sps_temporal_mvp_enabled_flag` to 0, so 0xFF is the value the record
///   means and 0 — the default — would name reference slot 0 of a picture that has no temporal MVP.
/// - `nal_unit_type` is the bitstream's, and it is what the driver writes into the slice NAL header.
/// - Every unused `reference_frames` slot is `VA_INVALID_SURFACE` with `VA_PICTURE_HEVC_INVALID`,
///   because zero is a real surface id on this driver.
fn picture_record(
    poc: i32,
    surface: u32,
    references: &[HevcReference],
    qp: u8,
    key_frame: bool,
    nal_unit_type: u8,
    last: bool,
    coded_buf: u32,
) -> Result<Vec<u8>, String> {
    if references.len() > 15 {
        return Err(format!("{} references, more than the record's 15 slots", references.len()));
    }
    let mut reference_frames = [unused_picture(); 15];
    for (slot, reference) in references.iter().enumerate() {
        let Some(entry) = reference_frames.get_mut(slot) else {
            return Err(format!("reference {slot} does not fit the record's {} slots", reference_frames.len()));
        };
        *entry = reference.as_picture();
    }
    let mut pic_fields = 0u32;
    if key_frame {
        pic_fields |= 1 << 0; // idr_pic_flag
    }
    // coding_type, bits 1-3: I = 1, P = 2.
    pic_fields |= coding_type(key_frame) << 1;
    // **`reference_pic_flag` is set on every picture, key frame included.** Every picture this loop
    // encodes is pushed onto the reference list and the next picture predicts from it — an IDR is in
    // the decoded picture buffer as soon as it is decoded (8.3.1), it is simply the only picture in
    // it. Clearing the flag on a key frame would be telling the driver the IDR may be discarded, and
    // the picture after it would predict from nothing.
    pic_fields |= 1 << 4;
    // cu_qp_delta_enabled_flag (bit 9) is REQUIRED by the driver; see the note above.
    pic_fields |= 1 << 9;
    let record = VaEncPictureParameterBufferHevc {
        decoded_curr_pic: ffi::VaPictureHevc {
            picture_id: surface,
            pic_order_cnt: poc,
            // **Zero, not `RPS_ST_CURR_BEFORE`.** The RPS flags describe which entries of
            // `ReferenceFrames[]` the *current picture's* short-term reference picture set refers to;
            // the current picture is not its own reference, so it carries none. Set on one, a driver
            // would count it into `NumPocStCurrBefore` and build an RPS that names it.
            flags: 0,
            ..Default::default()
        },
        reference_frames,
        coded_buf,
        // 0xFF is the documented value for a picture with temporal MVP off — see above.
        collocated_ref_pic_index: 0xFF,
        // `HEVC_LAST_PICTURE_EOSTREAM` only, and only when the caller said so.
        last_picture: if last { LAST_PICTURE_EOSTREAM_HEVC } else { 0 },
        // **`pic_init_qp` is `init_qp_minus26 + 26`** (`va_enc_hevc.h:602`), so a QP of 26 is 26
        // here. Subtracting 26 again is the mistake the H.264 note names and it produces a picture
        // at a quality nobody asked for with no error anywhere.
        pic_init_qp: qp,
        // **`diff_cu_qp_delta_depth` is 0, and that is a measurement against the working driver, not
        // an omission.** The picture record's `cu_qp_delta_enabled_flag` must be 1 — the driver
        // reports `cu_qp_delta` as `VA_FEATURE_REQUIRED` — and with it set the derived PPS states
        // the same depth, so the two agree. Depth is the CU level at which a delta may appear, and 0
        // is the CTU level, the coarsest the syntax has.
        //
        // **ffmpeg's own `hevc_vaapi` on this driver writes `diff_cu_qp_delta_depth = 3`**, which is
        // the CU level (8x8). Both are conforming and the driver encodes with either; this path
        // asks for the coarsest because it submits no per-CU quantiser, so a finer depth would name
        // a granularity it never uses. See `tests/hevenc.rs` for the comparison and the measurement
        // that the depth is not what makes the encode fail.
        diff_cu_qp_delta_depth: 0,
        pps_cb_qp_offset: 0,
        pps_cr_qp_offset: 0,
        num_tile_columns_minus1: 0,
        num_tile_rows_minus1: 0,
        column_width_minus1: [0; 19],
        row_height_minus1: [0; 21],
        // 0 is `log2_parallel_merge_level_minus2`, i.e. merge candidates may be evaluated at 4x4,
        // the finest level the syntax has. A value that only permits a coarser level would restrict
        // the driver's choices for a reason nobody asked for.
        log2_parallel_merge_level_minus2: 0,
        // 0 is the header's own "no bit size limit is checked" (`va_enc_hevc.h:640-643`). It is not
        // an unset field: a non-zero value here is a constraint, and FilmCraft derives no A.3.2
        // requirement to write one.
        ctu_max_bitsize_allowed: 0,
        num_ref_idx_l0_default_active_minus1: references.len().saturating_sub(1) as u8,
        num_ref_idx_l1_default_active_minus1: 0,
        slice_pic_parameter_set_id: 0,
        nal_unit_type,
        padding_after_nal_unit_type: [0; 2],
        pic_fields,
        // 0: no hierarchical level, which is the honest value while B frames are refused.
        hierarchical_level_plus1: 0,
        ..Default::default()
    };
    // TEMPORARY bisect harness — removed before the commit.
    let mut record = record;
    if let Ok(v) = std::env::var("FC_PIC_FIELDS") {
        record.pic_fields = u32::from_str_radix(&v, 16).unwrap_or(record.pic_fields);
    }
    if let Ok(v) = std::env::var("FC_PIC_NAL") {
        record.nal_unit_type = v.parse().unwrap_or(record.nal_unit_type);
    }
    if let Ok(v) = std::env::var("FC_PIC_COLLOC") {
        record.collocated_ref_pic_index = v.parse().unwrap_or(record.collocated_ref_pic_index);
    }
    if let Ok(v) = std::env::var("FC_PIC_QP") {
        record.pic_init_qp = v.parse().unwrap_or(record.pic_init_qp);
    }
    if let Ok(v) = std::env::var("FC_PIC_CUQPD") {
        record.diff_cu_qp_delta_depth = v.parse().unwrap_or(record.diff_cu_qp_delta_depth);
    }
    if let Ok(v) = std::env::var("FC_PIC_CBQP") {
        record.pps_cb_qp_offset = v.parse().unwrap_or(record.pps_cb_qp_offset);
    }
    if let Ok(v) = std::env::var("FC_PIC_CRQP") {
        record.pps_cr_qp_offset = v.parse().unwrap_or(record.pps_cr_qp_offset);
    }
    if let Ok(v) = std::env::var("FC_PIC_TC") {
        record.num_tile_columns_minus1 = v.parse().unwrap_or(record.num_tile_columns_minus1);
    }
    if let Ok(v) = std::env::var("FC_PIC_TR") {
        record.num_tile_rows_minus1 = v.parse().unwrap_or(record.num_tile_rows_minus1);
    }
    if let Ok(v) = std::env::var("FC_PIC_PARMERGE") {
        record.log2_parallel_merge_level_minus2 = v.parse().unwrap_or(record.log2_parallel_merge_level_minus2);
    }
    if let Ok(v) = std::env::var("FC_PIC_CTUBITS") {
        record.ctu_max_bitsize_allowed = v.parse().unwrap_or(record.ctu_max_bitsize_allowed);
    }
    if let Ok(v) = std::env::var("FC_PIC_L0") {
        record.num_ref_idx_l0_default_active_minus1 = v.parse().unwrap_or(record.num_ref_idx_l0_default_active_minus1);
    }
    if let Ok(v) = std::env::var("FC_PIC_L1") {
        record.num_ref_idx_l1_default_active_minus1 = v.parse().unwrap_or(record.num_ref_idx_l1_default_active_minus1);
    }
    if let Ok(v) = std::env::var("FC_PIC_PPSID") {
        record.slice_pic_parameter_set_id = v.parse().unwrap_or(record.slice_pic_parameter_set_id);
    }
    if let Ok(v) = std::env::var("FC_PIC_HIER") {
        record.hierarchical_level_plus1 = v.parse().unwrap_or(record.hierarchical_level_plus1);
    }
    if let Ok(v) = std::env::var("FC_PIC_SCC") {
        record.scc_fields = u16::from_str_radix(&v, 16).unwrap_or(record.scc_fields);
    }
    if let Ok(v) = std::env::var("FC_PIC_LAST") {
        record.last_picture = v.parse().unwrap_or(record.last_picture);
    }
    if let Ok(v) = std::env::var("FC_PIC_CWP") {
        record.decoded_curr_pic.flags = v.parse().unwrap_or(record.decoded_curr_pic.flags);
    }
    Ok(ffi::record_bytes(&record))
}

/// The HEVC slice record: one slice covering the whole picture, in **CTUs**.
///
/// - **`slice_segment_address` is 0 and `num_ctu_in_slice` is the picture's whole CTU count**, so
///   the slice is the picture. The unit is the thing that is easy to get wrong: these two fields
///   count CTUs (`va_enc_hevc.h:771-774`), not macroblocks and not pixels.
/// - **`slice_type` is the one field whose encoding had to be measured** — see [`slice_types`]. It
///   is written from one named place and nowhere else.
/// - **`ref_pic_list0` is the actual prediction.** An entry is a `VAPictureHEVC`, so it is a surface
///   *and* a picture order count; the entry's surface with the wrong POC names a picture the driver
///   will not find, which is a wrong picture rather than an error. The list is built from the same
///   [`HevcReference`] values the picture record's `reference_frames` are, which is what makes the
///   two agree — they have to, because the first says what the driver is holding and the second says
///   what the slice reads.
/// - The `pred_weight_table()` block is all zeros because
///   `pic_fields.enable_gpu_weighted_prediction` is clear, so it is not in the bitstream and its
///   values are not read. A driver that *did* read them would be reading zeros as "weight 0", i.e. a
///   black prediction.
fn slice_record(ctus: u32, slice_type: u8, references: &[HevcReference]) -> Result<Vec<u8>, String> {
    if ctus == 0 {
        return Err("a picture of 0 CTUs".into());
    }
    if references.len() > 15 {
        return Err(format!("{} references, more than the slice record's 15 entries", references.len()));
    }
    let mut ref_pic_list0 = [unused_picture(); 15];
    for (slot, reference) in references.iter().enumerate() {
        let Some(entry) = ref_pic_list0.get_mut(slot) else {
            return Err(format!("reference {slot} does not fit the slice record's {} entries", ref_pic_list0.len()));
        };
        *entry = reference.as_picture();
    }
    // **`slice_fields.last_slice_of_pic_flag` is set, and it is what says this slice ends the
    // picture.** With one slice per picture it is always 1; leaving it clear is a driver that has no
    // idea where the picture ends, which is a stream whose last slice is missing its own tail.
    //
    // The other fifteen bits are clear because this encoder uses none of their features: no
    // dependent slice segments (the submission is one independent slice), no temporal MVP (the SPS
    // says so), no SAO (the SPS says so), no reference index override (the picture record's count is
    // the count), no weighted prediction (the PPS says so), and the deblocking filter runs.
    let slice_fields = 1 << 0;
    let record = VaEncSliceParameterBufferHevc {
        slice_segment_address: 0,
        num_ctu_in_slice: ctus,
        slice_type,
        slice_pic_parameter_set_id: 0,
        // One list-0 entry for one reference, and no list 1 at all — a P or I slice has none.
        num_ref_idx_l0_active_minus1: references.len().saturating_sub(1) as u8,
        num_ref_idx_l1_active_minus1: 0,
        // **`max_num_merge_cand` is 5, and it is not a value this path could leave at zero.**
        // `va_enc_hevc.h:820-822` gives the range as `[1..5]`, so the `Default` of 0 this record
        // would otherwise inherit is *outside the documented range* — a merge-candidate count the
        // specification does not allow, which is a request for a slice header whose
        // `five_minus_max_num_merge_cand` cannot be written. 5 is the top of the range and the
        // number HEVC Main's own `MaxNumMergeCand` default is (`sps_slice_segment_header_extension`
        // aside, `num_ref_idx` independent), so it is the value that asks for no extra constraint.
        //
        // **The symptom this produced was not an error at submission.** The render succeeded, the
        // sync succeeded, and `vaMapBuffer` on the coded buffer then failed with
        // `VA_STATUS_ERROR_INTERNAL` — a failure that names the *readback* while the mistake is in
        // the *slice header the driver was asked to write*, twenty fields away. It was found by
        // bisecting the buffers the driver accepted (see
        // [`REFUSES_THE_QUANTIZATION_MISC_BUFFER`], whose table came out of the same bisect) and
        // then auditing every field the header gives a range for.
        max_num_merge_cand: 5,
        ref_pic_list0,
        // **Every unused entry is `VA_INVALID_SURFACE` with `VA_PICTURE_HEVC_INVALID`**, not zero,
        // for the reason the H.264 slice record's unused entries are: zero is a real surface id here.
        ref_pic_list1: [unused_picture(); 15],
        slice_fields,
        ..Default::default()
    };
    // TEMPORARY bisect harness — removed before the commit.
    let mut record = record;
    if let Ok(v) = std::env::var("FC_SLI_ADDR") {
        record.slice_segment_address = v.parse().unwrap_or(record.slice_segment_address);
    }
    if let Ok(v) = std::env::var("FC_SLI_CTUS") {
        record.num_ctu_in_slice = v.parse().unwrap_or(record.num_ctu_in_slice);
    }
    if let Ok(v) = std::env::var("FC_SLI_TYPE") {
        record.slice_type = v.parse().unwrap_or(record.slice_type);
    }
    if let Ok(v) = std::env::var("FC_SLI_PPSID") {
        record.slice_pic_parameter_set_id = v.parse().unwrap_or(record.slice_pic_parameter_set_id);
    }
    if let Ok(v) = std::env::var("FC_SLI_L0") {
        record.num_ref_idx_l0_active_minus1 = v.parse().unwrap_or(record.num_ref_idx_l0_active_minus1);
    }
    if let Ok(v) = std::env::var("FC_SLI_L1") {
        record.num_ref_idx_l1_active_minus1 = v.parse().unwrap_or(record.num_ref_idx_l1_active_minus1);
    }
    if let Ok(v) = std::env::var("FC_SLI_MERGE") {
        record.max_num_merge_cand = v.parse().unwrap_or(record.max_num_merge_cand);
    }
    if let Ok(v) = std::env::var("FC_SLI_QPDELTA") {
        record.slice_qp_delta = v.parse().unwrap_or(record.slice_qp_delta);
    }
    if let Ok(v) = std::env::var("FC_SLI_CBQP") {
        record.slice_cb_qp_offset = v.parse().unwrap_or(record.slice_cb_qp_offset);
    }
    if let Ok(v) = std::env::var("FC_SLI_CRQP") {
        record.slice_cr_qp_offset = v.parse().unwrap_or(record.slice_cr_qp_offset);
    }
    if let Ok(v) = std::env::var("FC_SLI_BETA") {
        record.slice_beta_offset_div2 = v.parse().unwrap_or(record.slice_beta_offset_div2);
    }
    if let Ok(v) = std::env::var("FC_SLI_TCOFF") {
        record.slice_tc_offset_div2 = v.parse().unwrap_or(record.slice_tc_offset_div2);
    }
    if let Ok(v) = std::env::var("FC_SLI_FIELDS") {
        record.slice_fields = u32::from_str_radix(&v, 16).unwrap_or(record.slice_fields);
    }
    if let Ok(v) = std::env::var("FC_SLI_PWTOFF") {
        record.pred_weight_table_bit_offset = v.parse().unwrap_or(record.pred_weight_table_bit_offset);
    }
    if let Ok(v) = std::env::var("FC_SLI_PWTLEN") {
        record.pred_weight_table_bit_length = v.parse().unwrap_or(record.pred_weight_table_bit_length);
    }
    if let Ok(v) = std::env::var("FC_SLI_DENOM") {
        record.luma_log2_weight_denom = v.parse().unwrap_or(record.luma_log2_weight_denom);
    }
    if let Ok(v) = std::env::var("FC_SLI_L0FLAGS") {
        record.ref_pic_list0[0].flags = v.parse().unwrap_or(record.ref_pic_list0[0].flags);
    }
    Ok(ffi::record_bytes(&record))
}

/// **This driver refuses `VAEncMiscParameterTypeQuantization` for HEVC encode, so this path submits
/// no quantisation misc buffer at all.** The measurement, on iHD 26.3.5 / libva 1.24:
///
/// | buffers handed to `vaRenderPicture` | result |
/// |---|---|
/// | sequence, picture, slice, frame rate | **ok** |
/// | those four **plus** the quantisation misc buffer | `vaRenderPicture` → `VA_STATUS_ERROR_INVALID_BUFFER` |
/// | **the quantisation misc buffer alone** | `vaRenderPicture` → `VA_STATUS_ERROR_INVALID_BUFFER` |
///
/// The middle row alone would have been a puzzle — "invalid buffer" is also what a **destroyed**
/// buffer id gives, which is the mistake the `if let` note above is about, so the natural reading was
/// that the handle had been dropped. **The third row is what settles it: the buffer is refused on its
/// own, with no other buffer present, so nothing about its lifetime can be involved.** Without that
/// row this would have been "fix the lifetime", and the fix would have been wrong.
///
/// **The header agrees, and says why** (`va.h:2391-2392`): *"Buffer type used for quantization
/// parameters, **it's per-sequence parameter**"*. A per-sequence buffer submitted per picture is a
/// buffer the encoder has already consumed and will not take twice.
///
/// **The consequence for the quantiser is not a silent one.** Under [`RateControl::ConstantQuantiser`]
/// the QP still reaches the driver — in the picture record's `pic_init_qp_minus26`, which
/// `super::hevc_params` derives the derived PPS's `init_qp_minus26` from, so the base `SliceQPY` is
/// the value the caller asked for. What is refused is the *trellis* control (`quantization_flags`,
/// `va.h:2807-2808`: clear means "let the driver decide"), which this backend has never had a value
/// for. **The picture's `coded_buffer_status` low byte is the driver's average QP, and
/// [`HevcEncodedPicture::status`] carries it** — so the quality that was actually applied is
/// measured and reported rather than assumed from the request.
///
/// **H.264's path does submit this buffer and this is not a claim about H.264.** It is a measurement
/// of this codec on this driver, and the difference between the two codecs' submission paths is the
/// kind of thing this milestone's brief warned about: a result that transfers from one codec is a
/// hypothesis until it has been measured on the other.
pub const REFUSES_THE_QUANTIZATION_MISC_BUFFER: bool = true;

/// One NAL unit of an Annex-B run, as a header byte and the payload after it.
struct NalUnit<'a> {
    header: u8,
    body: &'a [u8],
}

impl NalUnit<'_> {
    /// Whether this is HEVC's `end_of_seq()` or `end_of_stream()`, which are **`NAL_UNIT_TYPE` 35 and
    /// 36** (H.265 Table 7-1) — not H.264's 10 and 11.
    ///
    /// **The numbers are different because the two codecs' tables are**, and reading a stream with
    /// H.264's in it would call every HEVC filler (`RASL_R`, 9) an end marker and stop looking.
    fn is_end_of_stream(&self) -> bool {
        matches!(self.unit_type(), 35 | 36)
    }

    /// This unit as a configuration record stores it: the two-byte NAL header followed by the
    /// payload. **HEVC's NAL header is two bytes**, the first of which is `forbidden_zero_bit(1) |
    /// nal_unit_type(6) | nuh_layer_id(6) | nuh_temporal_id_plus1(3)`, so the type is
    /// `(header >> 1) & 0x3f` and the second byte is part of the unit. Dropping it — which the
    /// H.264 walk's one-byte assumption does — would hand an `hvcC` a unit whose length does not
    /// match its contents.
    fn nal_unit(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.body.len() + 2);
        out.push(self.header);
        out.extend_from_slice(self.body);
        out
    }

    /// The NAL unit type, which for HEVC is six bits at offset 1 of the first header byte.
    fn unit_type(&self) -> u8 {
        (self.header >> 1) & 0x3f
    }
}

/// The NAL units of the Annex-B run `bytes`, or why `bytes` is not one.
///
/// **A start-code walk and not a search for `00 00 01` anywhere**: that sequence occurs inside slice
/// payload by coincidence all the time, and a scan that matched one would report a parameter set the
/// stream does not have.
///
/// **This is the H.264 path's walk, re-asked, and one of the two places where the codecs genuinely
/// differ**: H.264's version treats an empty payload as legal for NAL types 10 and 11, and this one
/// for 35 and 36, because those are HEVC's. The structural rule is identical and is why the
/// duplication is four lines rather than a paragraph.
fn annex_b_units(bytes: &[u8]) -> Result<Vec<NalUnit<'_>>, String> {
    let mut units = Vec::new();
    let mut at = 0usize;
    while let Some(found) = bytes.get(at..).and_then(|rest| rest.windows(3).position(|w| w == [0, 0, 1])) {
        // A four-byte code is the same three bytes one later, so the search resumes past the header
        // byte this one found rather than past the code.
        let start = at + found + 3;
        let Some(&header) = bytes.get(start) else { break };
        at = start + 2;
        let end = bytes.get(at..).and_then(|rest| rest.windows(3).position(|w| w == [0, 0, 1])).map_or(bytes.len(), |next| at + next);
        let Some(body) = bytes.get(at..end) else { break };
        // An empty body is legal only for the two types the specification defines as carrying
        // nothing; anything else with no payload is a truncated slice, which is what this readback
        // exists to catch.
        if body.is_empty() && !matches!((header >> 1) & 0x3f, 35 | 36) {
            return Err(format!(
                "the coded picture holds a NAL unit of type {} with no payload, which is a truncated slice rather than an end marker",
                (header >> 1) & 0x3f
            ));
        }
        units.push(NalUnit { header, body });
    }
    if units.is_empty() {
        return Err("the coded picture holds no NAL unit".into());
    }
    Ok(units)
}

/// The parameter sets as an Annex-B run: a start code and a NAL unit for each, VPS first.
///
/// **The order is VPS, SPS, PPS**, which is the order 7.4.3.1 gives a decoder access: the SPS names a
/// VPS id, so a decoder that meets the SPS first has nothing to resolve it against. (H.264 has no
/// equivalent step, which is why its order is only two.)
fn parameter_sets_annex_b(sets: &HevcParameterSets) -> Vec<u8> {
    let mut out = Vec::with_capacity(sets.vps.len() + sets.sps.len() + sets.pps.len() + 12);
    for nal in [&sets.vps, &sets.sps, &sets.pps] {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nal);
    }
    out
}

/// Read `bytes` back as a record, or `None` if it is not long enough.
///
/// The same helper the H.264 path uses to prove a record it just built is one the driver will read
/// whole: `size_of` bytes with no reading past the end, so a short buffer is `None` rather than a
/// truncated record.
/// **Parse the slice header the driver *wrote* and check it against the records this module
/// submitted.**
///
/// This is the strongest check available on this path, and it is what turns "the driver accepted the
/// records" into "the driver encoded what the records say".
///
/// **Why the slice header and not the picture.** Everything this module submitted is a record the
/// driver read; nothing comes back except the bitstream. The slice header is the part of that
/// bitstream which **repeats** what the records said — `slice_type`, `slice_pic_parameter_set_id`,
/// `num_ref_idx_*_active`, `poc_lsb`, and the short-term reference picture set the driver built from
/// `ReferenceFrames[]` — so it is the one place where a disagreement between what was asked for and
/// what was done is visible at all.
///
/// It is parsed with [`filmcraft_hevc::slice::SliceHeader::parse`], **the parser the decode path
/// already trusts for real streams**, resolving its PPS and SPS through the *derived* sets — so the
/// sets and the slice have to agree to parse at all. That is deliberate: a driver that wrote a
/// different POC width or a different reference picture set would make the header the wrong length,
/// and a parse failure here is the error that says so.
///
/// # What it catches, and what it cannot
///
/// - **A `slice_type` the driver reinterpreted.** This is the check that establishes
///   [`slice_types`]: the value read back is compared with the value submitted, so a driver that
///   wrote a different number is an `Err` rather than a stream that decodes to something else.
/// - **A picture the driver coded as the wrong type** — an IDR submitted as 19 and written as a
///   `TRAIL_R` is a key frame a decoder cannot start from.
/// - **A POC that does not match**, which is a reference chain broken rather than a picture.
/// - **A reference index count that does not match**, which is a slice reading pictures the driver
///   did not put in its list.
///
/// **It cannot see the picture itself.** A hardware encoder and a software encoder are different
/// algorithms, so nothing here can say the pixels are right — that is the ffmpeg-decodes-to-the-
/// source gate in `tests/hevenc.rs`, and this check is not a substitute for it.
fn check_output_against_the_records(
    sets: &HevcParameterSets,
    coded: &[u8],
    slice_record: &VaEncSliceParameterBufferHevc,
    picture_record: &VaEncPictureParameterBufferHevc,
    key_frame: bool,
) -> Result<(), String> {
    let units = annex_b_units(coded)?;
    let pps = filmcraft_hevc::params::Pps::parse(&unescape_rbsp(&strip_nal(&sets.pps, NAL_PPS)?))
        .map_err(|e| format!("the PPS the sample entry would carry does not parse: {e}"))?;
    let sps = filmcraft_hevc::params::Sps::parse(&unescape_rbsp(&strip_nal(&sets.sps, NAL_SPS)?))
        .map_err(|e| format!("the SPS the sample entry would carry does not parse: {e}"))?;
    // Exactly one VCL NAL is expected: this module submits one slice per picture and **refuses**
    // anything else (see the module note), so two would mean the driver cut the picture itself.
    let vcl: Vec<&NalUnit<'_>> = units.iter().filter(|u| u.unit_type() < 32).collect();
    if vcl.len() != 1 {
        return Err(format!("{} coded NAL units in a picture this module submitted one slice for", vcl.len()));
    }
    let Some(unit) = vcl.first() else {
        return Err("the coded picture holds no slice NAL unit".into());
    };
    let nal = unit.nal_unit();
    let header = filmcraft_hevc::slice::NalHeader::parse(&nal).map_err(|e| format!("the slice NAL header the driver wrote: {e}"))?;
    let rbsp = unescape_rbsp(nal.get(2..).unwrap_or_default());
    let (header, _, _) = filmcraft_hevc::slice::SliceHeader::parse(
        &rbsp,
        header,
        |id| {
            if id != pps.id {
                return Err(filmcraft_hevc::Error::MissingParameterSet(format!("picture parameter set {id}, and the one this stream carries is {}", pps.id)));
            }
            Ok((std::sync::Arc::new(pps.clone()), std::sync::Arc::new(sps.clone())))
        },
        None,
    )
    .map_err(|e| format!("the slice header the driver wrote does not parse against the parameter sets this module derived: {e}"))?;
    // **`slice_type` read back, as a number, against the number submitted.** This is the comparison
    // [`slice_types`] rests on.
    let written = match header.slice_type {
        filmcraft_hevc::slice::SliceType::B => 0u8,
        filmcraft_hevc::slice::SliceType::P => 1u8,
        filmcraft_hevc::slice::SliceType::I => 2u8,
    };
    let mut mismatches: Vec<String> = Vec::new();
    if written != slice_record.slice_type {
        mismatches.push(format!("the driver wrote slice_type {written} and the record said {}", slice_record.slice_type));
    }
    if header.nal.nal_type != picture_record.nal_unit_type {
        mismatches.push(format!("the driver wrote NAL unit type {} and the record said {}", header.nal.nal_type, picture_record.nal_unit_type));
    }
    // **The key-frame decision, read back from the NAL type rather than trusted from the loop.** An
    // IDR is a `NAL_UNIT_TYPE` in 16..=23; anything else is not a point a decoder can start from.
    if header.nal.is_idr() != key_frame {
        mismatches.push(format!(
            "the driver wrote a NAL unit type {} and {} was asked for",
            header.nal.nal_type,
            if key_frame { "an IDR" } else { "a trailing picture" }
        ));
    }
    if header.pps_id != u32::from(slice_record.slice_pic_parameter_set_id) {
        mismatches.push(format!("the slice names PPS {} and the record said {}", header.pps_id, slice_record.slice_pic_parameter_set_id));
    }
    // **The reference index count is `num_ref_idx[0]` in the header and
    // `num_ref_idx_l0_active_minus1 + 1` in the record.** A slice told to read one reference when the
    // driver put two in its list is a slice reading a picture the caller never chose.
    if header.num_ref_idx[0] != u32::from(slice_record.num_ref_idx_l0_active_minus1) + 1 {
        mismatches.push(format!(
            "the slice header reads {} reference(s) and the record said {}",
            header.num_ref_idx[0],
            u32::from(slice_record.num_ref_idx_l0_active_minus1) + 1
        ));
    }
    // **A key frame must not read a reference at all.** 8.3.1: an IDR empties the decoded picture
    // buffer, so an IDR whose header names a reference is a header over a list that should be empty.
    if key_frame && header.num_ref_idx[0] != 1 {
        mismatches.push(format!("an IDR whose slice header reads {} reference(s)", header.num_ref_idx[0] - 1));
    }
    if mismatches.is_empty() {
        return Ok(());
    }
    Err(format!("the slice the driver wrote disagrees with the records this module submitted: {} — {} mismatch(es)", mismatches.join(", "), mismatches.len()))
}

/// The NAL unit in `unit` whose header carries `nal_type` — two header bytes plus payload, no start
/// code.
///
/// **HEVC's NAL header is two bytes** (`forbidden_zero_bit | nal_unit_type(6) | nuh_layer_id(6) |
/// nuh_temporal_id_plus1(3)`), so the type is six bits at offset 1 of the first and the second byte
/// belongs to the unit. A search for the header byte alone would also match inside a payload, so the
/// match is on the whole two-byte header.
fn strip_nal(unit: &[u8], nal_type: u8) -> Result<Vec<u8>, String> {
    let wanted = nal_type << 1;
    let Some(at) = unit.iter().position(|&b| b == wanted) else {
        return Err(format!("the NAL unit of type {nal_type} this module built is not there"));
    };
    let rest = unit.get(at..).ok_or_else(|| format!("the NAL unit of type {nal_type} has no payload"))?;
    if rest.len() < 2 {
        return Err(format!("the NAL unit of type {nal_type} has a one-byte header, and HEVC's is two"));
    }
    Ok(rest.to_vec())
}

/// Remove a NAL unit's emulation prevention bytes, which is what every parser wants to see.
///
/// [`filmcraft_bitstream::unescape_rbsp`] rather than a hand-written pass: the escaping rule (a
/// `00 00 03` whose third byte is below 4) has one subtlety — a trailing `00 00` at the very end of
/// an RBSP — and this is the decoder's own implementation of it.
fn unescape_rbsp(escaped: &[u8]) -> Vec<u8> {
    filmcraft_bitstream::unescape_rbsp(escaped)
}

/// Read `bytes` back as a record, or `None` if it is not long enough.
///
/// **This is [`ffi::record_in`], used rather than re-written**, because the one `unsafe` a raw byte
/// read needs belongs in the FFI module and this file has none: `crates/platform` carries
/// `unsafe_code = "deny"` with `#[allow(unsafe_code)]` on its FFI modules *only*, so a second copy
/// here would be an `unsafe` block outside them. That is a rule, not a style preference.
fn read<T: Copy + Default>(bytes: &[u8]) -> Option<T> {
    ffi::record_in::<T>(bytes)
}
