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
//! # This path produces a picture, and the cause was the packed **slice header**, not a record
//!
//! **One I frame followed by P frames that genuinely predict, and ffmpeg decodes the stream to the
//! right pictures** (320x240, `VA_RC_CQP`, QP 26: I 184 bytes, P 43-55, mean 48; every picture
//! within a mean absolute deviation of 9.1/255 of the **source** ramp, worst 21).
//!
//! **The driver writes no slice header of its own.** It writes the bytes the application supplies and
//! then the slice *data*, which is why the submission path has to supply one — and why, not
//! supplying it, every submission failed at `vaMapBuffer` with `VA_STATUS_ERROR_INTERNAL` while
//! `vaRenderPicture`, `vaEndPicture` and `vaSyncBuffer` all returned success. **`va_enc_hevc.h:299-309`
//! says so**, and so does a capture of ffmpeg's own `hevc_vaapi` submission: its coded bytes begin
//! with the packed slice header it submitted, verbatim.
//!
//! **What was *not* the cause**, and this is the correction to what two agents recorded: not any
//! record (see below), not the rate control, and **not `VAConfigAttribEncPackedHeaders`** — the
//! attribute reads 0x0, 0x2, 0x3 and 0x5 alike on this driver once the buffer is there. The table
//! is on [`PACKED_HEADERS_FOR_HEVC`].
//!
//! # What the two earlier searches established, and why they were right to stop where they did
//!
//! 1. **Every field of all three records was swept**, over the range `va_enc_hevc.h` states for it.
//!    **The failure moved for none of them.** That is a real result and it still holds.
//! 2. **The records were then made byte-for-byte identical to a submission this same driver
//!    accepts.** **The failure did not move.** That step is still sound — *with one correction*:
//!    re-capturing ffmpeg's `hevc_vaapi` submission through the public libva entry points and
//!    comparing field by field shows the two record sets were never equal in the first place (ffmpeg
//!    sets `sao_enabled_flag`, `sps_temporal_mvp_enabled_flag`, `transform_skip_enabled_flag` and
//!    `pps_loop_filter_across_slices_enabled_flag` where this path does not, and leaves
//!    `cu_qp_delta_enabled_flag` clear where the driver reports it `VA_FEATURE_REQUIRED`). **So the
//!    conclusion "the records are not the cause" was right and the evidence for it was weaker than
//!    claimed.** The corrected claim is the weaker, better one: *no difference between these records
//!    accounted for the failure*, which is what re-running ffmpeg's own records through this
//!    submission showed — they fail here exactly as ours do.
//! 3. **The rate-control attribute was swept**, and this is the result:
//!
//! | `VAConfigAttribRateControl` | mode | outcome, with **no packed slice header supplied** |
//! |---|---|---|
//! | `0x02` | `VA_RC_CBR` | `vaMapBuffer` → `VA_STATUS_ERROR_INTERNAL` |
//! | `0x04` | `VA_RC_VBR` | `vaMapBuffer` → `VA_STATUS_ERROR_INTERNAL` |
//! | `0x08` | `VA_RC_VCM` | `vaMapBuffer` → `VA_STATUS_ERROR_INTERNAL` |
//! | **`0x10`** | **`VA_RC_CQP`** — the one [`RateControl::ConstantQuantiser`] names | **`vaMapBuffer` → `VA_STATUS_ERROR_INTERNAL`** |
//! | `0x40` | `VA_RC_ICQ` | `vaEndPicture` → `VA_STATUS_ERROR_INTERNAL` |
//! | `0x80` | `VA_RC_MB` | `vaMapBuffer` → `VA_STATUS_ERROR_INTERNAL` |
//! | `0x400` | `VA_RC_QVBR` | `vaEndPicture` → `VA_STATUS_ERROR_INTERNAL` |
//! | `0x1000` | `VA_RC_TCBRC` | `vaMapBuffer` → `VA_STATUS_ERROR_INTERNAL` |
//! | `0x01` / `0x20` / `0x100` / `0x200` / `0x800` | not in the driver's `0x14de` mask | `vaCreateConfig` refuses them |
//!
//! **That table was real, and it is why the rate control was ruled out**: it was swept *with* a
//! buffer set the driver refuses, so every mode failed for the same reason and the sweep could not
//! have distinguished a rate-control problem from the packed-header one. **With the slice header
//! supplied, `VA_RC_CQP` — the mode this path names — encodes**, and the table's other seven modes
//! are untested rather than refuted. [`hevc_rate_control_outcomes`] says so.
//!
//! **What this rules out, so it is not re-tried:** a single out-of-range record field; the record set
//! as a whole (ffmpeg's own records submitted through this path fail identically); the surface pool
//! and the context's render targets; the picture size (ffmpeg encodes 320x240 and 320x256 on this
//! driver); the rate-control attribute; **and route (b)** — configuring
//! `VA_ENC_PACKED_HEADER_NONE` and letting the driver be asked for nothing it must supply also
//! produces no picture.
//!
//! **Two earlier measurements were re-taken here and both still hold**: the quantisation misc buffer
//! is refused **on its own**, with nothing else present, so it is not a buffer-lifetime bug
//! ([`REFUSES_THE_QUANTIZATION_MISC_BUFFER`]); and `VA_RC_VCM` is in HEVC's mask and not H.264's
//! ([`VCM`]).
//!
//! **What is established and pinned:** the three encode records' sizes, offsets and every hole
//! against a C compile (`hevc_abi.rs`); the derived VPS/SPS/PPS and slice header parsing back through
//! FilmCraft's own HEVC parsers with every field compared against the record it was written from;
//! the packed-header configuration and its measurement; the rate-control mask and the `VA_RC_VCM`
//! refusal; the quantisation-misc-buffer refusal; the level table; and `slice_type`.
//!
//! # Four real bitstream bugs were found on the way, none of which failed loudly
//!
//! Every one of these is a mistake that parses, and three of the four produced a stream that
//! **decoded to the right pictures anyway** — which is the argument for parsing the output rather
//! than trusting it:
//!
//! - **the SPS wrote the VUI's bitstream restriction block's five fields outside its flag** (see
//!   `hevc_params.rs`'s `sps`), which put five stray bits before `sps_extension_present_flag` and
//!   made ffmpeg's `trace_headers` stop on the SPS;
//! - **the Annex-B walk kept one of HEVC's two NAL-header bytes** ([`NalUnit`]), which read the
//!   slice header one byte out of step and named a PPS id three fields away;
//! - **`key_frame_interval_elapsed` did not exclude the interval 0**, so every picture after the
//!   first was forced to be an IDR ([`HevcEncoder::key_frame_interval_elapsed`]);
//! - and the slice header this path now writes was itself wrong twice — `slice_qp_delta` inside the
//!   non-IDR chain, and `slice_loop_filter_across_slices_enabled_flag` present when the derived PPS
//!   does not carry it. Both were caught by writing the header and reading it back with the
//!   decoder's own parser before submitting it.
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
//! - the **slice header in the coded picture** agrees with the records: its NAL type, its
//!   `slice_type`, its PPS id, its reference indices, parsed back out of the coded bytes by the same
//!   parser ([`check_output_against_the_records`]). **What this is worth is narrower than it looks**:
//!   the driver does not write that header, it passes the one this module supplied through verbatim,
//!   so this checks FilmCraft's own header against FilmCraft's own records, and ffmpeg's
//!   `trace_headers` over the stream is the independent read;
//! - and the end-to-end claim, which is the only one that is really about pictures: the exported
//!   file **decodes with ffmpeg to the right pictures**, within a lossy tolerance, against the
//!   **source**. That is `crates/platform/tests/hevenc.rs` and `examples/vahevcprobe.rs`, and it is
//!   the gate.

/// The picture kinds an [`HevcEncodeRequest`] can name. Re-exported from [`super::encoder`] rather
/// than redeclared, because HEVC's request type naming a *different* enum is exactly the drift the
/// module note warns about — and the `B` case stays constructible so [`HevcEncoder::encode`]'s
/// refusal of it is reachable rather than a type-level impossibility.
pub use super::encoder::PictureKind;
use super::encoder::{MAX_REFERENCES, RateControl};
use super::ffi::{self, Display, Encoder, PICTURE_HEVC_INVALID, PICTURE_HEVC_RPS_ST_CURR_BEFORE, Profile, RtFormat};
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

/// **`VAConfigAttribEncPackedHeaders` for this path: `VA_ENC_PACKED_HEADER_SEQUENCE |
/// VA_ENC_PACKED_HEADER_SLICE` (0x5) — chosen for what it states, and explicitly *not* the thing
/// that fixed this path's readback failure.**
///
/// # What the measurement says, because the first version of this note said the opposite
///
/// | `VAConfigAttribEncPackedHeaders` | packed sequence + slice header buffers | outcome on iHD 26.3.5 |
/// |---|---|---|
/// | `0x5` `SEQUENCE \| SLICE` | **submitted** | **a picture comes back** |
/// | `0x3` `SEQUENCE \| PICTURE` | submitted | **a picture comes back** |
/// | `0x2` `PICTURE` | submitted | **a picture comes back** |
/// | `0x0` `VA_ENC_PACKED_HEADER_NONE` | submitted | **a picture comes back** |
/// | `0x5` `SEQUENCE \| SLICE` | **not** submitted | `vaMapBuffer` → `VA_STATUS_ERROR_INTERNAL` |
/// | `0x3` `SEQUENCE \| PICTURE` | not submitted | `vaMapBuffer` → `VA_STATUS_ERROR_INTERNAL` |
///
/// **The driver does not read this attribute on this path. What it refuses to encode without is the
/// packed slice header buffer**, and every row of the top half encodes identically (184 / 51 / 45 /
/// 43 / 55 / 48 bytes) whatever the attribute says. So the value is here because it must *state*
/// what the application supplies — and an attribute that lies about its own buffer set is a
/// different kind of wrong from one that happens not to be enforced.
///
/// # Why `0x5` and not `0x3`, given that `0x3` also encodes
///
/// **Because `0x3` says this path supplies a packed *picture* header and it does not.** The PPS
/// goes to the driver inside the packed *sequence* buffer with the VPS and SPS, which is what
/// ffmpeg's own `hevc_vaapi` does on this driver (one `VAEncPackedHeaderSequence` data buffer
/// carrying all three). Declaring `VA_ENC_PACKED_HEADER_PICTURE` claims a buffer nobody supplies,
/// and **`VA_ENC_PACKED_HEADER_NONE` claims there are none.** `0x5` is the two flags that match the
/// two pairs actually submitted.
///
/// ffmpeg's own `vaCreateConfig` for `VAProfileHEVCMain` / `VAEntrypointEncSlice` on this machine
/// passes `0xd` — `SEQUENCE | SLICE | MISC`, the deprecated `VA_ENC_PACKED_HEADER_MISC` included.
/// **`0x5` is that value less `MISC`**, and `MISC` is deprecated in favour of
/// `VA_ENC_PACKED_HEADER_RAW_DATA`, which this path does not use.
///
/// **H.264's path configures `0x3` and needs no packed headers at all** — measured, on the same
/// driver, with 8 of 11 presets exporting. A sibling that manages without them is exactly why the
/// HEVC case had to be measured rather than copied from either direction.
pub const PACKED_HEADERS_FOR_HEVC: u32 = super::ffi::PACKED_HEADER_SEQUENCE | super::hevc_enc::PACKED_SLICE_FLAG;

/// `log2_max_pic_order_cnt_lsb_minus4` as this encoder writes it: **8**, so `MaxPicOrderCntLsb` is
/// `2^(8+4)` = **4096** and `slice_pic_order_cnt_lsb` is a **12-bit** field.
///
/// **The value was 8 and the note said 0.** A reader who believed the note would expect a 4-bit
/// field and a modulus of 16, and the slice header written by
/// [`super::hevc_params::slice_header`] takes its width and its modulus from this constant — so the
/// note was not a typo in a comment but a description of a different encoder, and ffmpeg's
/// `trace_headers` over the stream this path produced reads
/// `slice_pic_order_cnt_lsb 000000000001 = 1` and then `short_term_ref_pic_set_sps_flag` one field
/// later, which is what a 12-bit field is.
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
/// **It is in HEVC's rate-control mask and not H.264's, and the export gate does not offer it — but
/// the reason has moved, because the first sweep's reason was falsified.**
///
/// `VaEncoder::rate_control_names` lists H.264's eight modes and VCM is not among them, because VCM
/// is not in the mask libva reports for `VAProfileH264Main` on this driver. For `VAProfileHEVCMain`
/// the same query answers with `0x14de`, which **contains** `VA_RC_VCM` and therefore names it (the
/// full expansion is on [`HEVC_RATE_CONTROL_MASK`]).
///
/// **What the first sweep concluded, and why it was wrong.** It read VCM's failure as the mode being
/// unrunnable and recorded it as a refusal. **The re-taken sweep measures VCM *encoding*** — mean
/// 9.1/255 from the source, ffmpeg silent, both with a rate-control record and with none. **So the
/// driver runs it and the old reason is false.**
///
/// **What the gate declines now, and why that is still the right answer.** The decline is **not**
/// "this driver cannot do VCM" — it can. It is that **nothing above this backend can ask for it**:
/// [`super::encoder::RateControl`] has no variant for it, FilmCraft's `BitrateMode` has no mode that
/// means "a variable bit rate with a per-picture cap", and `ExportSettings` has no field for the cap.
/// A mode that appears in one codec's mask and not the other's, and that no setting above it can
/// name, is exactly the kind of thing that gets accepted silently the moment a mask bit is forwarded
/// into a config attribute — so it stays a **named refusal with a stated reason**, and the reason is
/// now the absence of a way to ask for it rather than the driver's inability to answer.
pub const VCM: u32 = 0x0000_0008;

/// `VA_RC_MB` (`va.h:1125`): **`0x00000080`**, the mode that moves quantiser between macroblocks.
///
/// **Declared here rather than in [`super::ffi`] because `ffi.rs` is the FFI module and this
/// constant exists only for the re-sweep**: the H.264 path has never submitted it, `RateControl` has
/// no variant for it, and nothing outside this module names it. It is here because the re-sweep
/// submits **every mode the driver advertises** — including the ones nobody here has a record for —
/// and a sweep that could only name four of the eight would be reporting the modes it can spell.
pub const MB: u32 = 0x0000_0080;

/// **What each rate-control mode this driver advertises did**, measured by submitting the same
/// records under each one — **re-taken, because the first sweep's precondition no longer holds.**
///
/// # Why this table was taken twice
///
/// The first sweep was taken **without a packed slice header**, which this driver refuses to encode
/// without. **So it answered "which mode encodes when the slice header is missing", not "which mode
/// encodes"** — it could not have told a rate-control problem from a missing buffer — and it recorded
/// that limitation, concluding that `VA_RC_ICQ` and `VA_RC_QVBR` were **untested, not refuted**.
///
/// **A sweep whose precondition has changed is not carried forward.** The table below was re-taken
/// with the same records, the same buffer set and the same buffer order a CBR export uses, changing
/// **only** the `VAConfigAttribRateControl` value, and **each mode was submitted twice — once with a
/// rate-control record and once with none — which is what separates a mode that fails from a record
/// that fails.** Measured on iHD 26.3.5, six 320x240 pictures at 30 fps:
///
/// | attribute | mode | with CBR's record | with no record | what it is |
/// |---|---|---|---|---|
/// | `0x02` | `VA_RC_CBR` | encodes, mean 9.1/255 | encodes, mean 9.2/255 | works — what the export gate submits |
/// | `0x04` | `VA_RC_VBR` | encodes, mean 9.1/255 | encodes, mean 9.1/255 | works — what the export gate submits |
/// | `0x08` | `VA_RC_VCM` | **encodes**, mean 9.1/255 | encodes, mean 9.1/255 | **the driver runs it** |
/// | **`0x10`** | **`VA_RC_CQP`** — what [`RateControl::ConstantQuantiser`] names | **not swept** | **not swept** | refused by [`HevcEncoder::at_swept_rate_control`]: it is submitted through the constructor that carries the quantiser's contract |
/// | `0x40` | `VA_RC_ICQ` | **`vaEndPicture` → `VA_STATUS_ERROR_INTERNAL`** | **the same** | **refuted — the mode, not the record** |
/// | `0x80` | `VA_RC_MB` | encodes, mean 9.1/255 | encodes, mean 9.1/255 | works |
/// | `0x400` | `VA_RC_QVBR` | **`vaEndPicture` → `VA_STATUS_ERROR_INTERNAL`** | **the same** | **refuted — the mode, not the record** |
/// | `0x1000` | `VA_RC_TCBRC` | **`vaRenderPicture` → invalid VABufferID for 9 encode buffers** | **encodes**, mean 9.2/255 | **fails on the record, not on the mode** |
/// | `0x01`, `0x20`, `0x100`, `0x200`, `0x800` | not in the driver's `0x14de` mask | `vaCreateConfig` refuses with `VA_STATUS_ERROR_INVALID_PARAMETER` | — | not offered |
///
/// **Three rows say something the first sweep could not.**
///
/// - **`VA_RC_ICQ` and `VA_RC_QVBR` are refuted, not untested.** They fail **both ways**, and that
///   pair is the whole argument: removing the rate-control record the mode objected to changed
///   nothing, so the failure is the **mode**. The first sweep's "untested, not refuted" was correct
///   about its own evidence and has now been replaced by evidence.
/// - **`VA_RC_TCBRC` fails on the *record*, not on the mode** — it encodes with no record at all and
///   fails at `vaRenderPicture` with CBR's, which is the signature of a record this backend cannot
///   build rather than a mode it cannot run.
/// - **`VA_RC_VCM` encodes.** The old note called it a refusal because `RateControl` has no variant
///   for it; **that reason is falsified** — the driver runs it. What is still true is that
///   FilmCraft's settings and [`RateControl`] have no way to *name* it, so the export gate declines it
///   as **unreachable rather than broken**. [`VCM`] keeps the constant and the mask measurement;
///   **the refusal reason has moved.**
///
/// # What this still rules out, so it is not tried again
///
/// A single out-of-range record field; every zero-from-`Default` candidate including
/// `max_num_merge_cand`; the surface pool and the context's render targets (both swept, and a
/// working submission's shape replicated); the picture size (ffmpeg encodes 320x240 **and** 320x256
/// here); **the rate-control attribute**; **and the whole record set** — ffmpeg's own records
/// submitted through this path fail identically.
pub mod hevc_rate_control_outcomes {
    /// **What the table adds up to, in one line**, so a reader does not have to re-derive it.
    ///
    /// **This is the re-taken sweep.** The first one was taken without a packed slice header — which
    /// this driver refuses to encode without — so every mode failed for one reason that had nothing
    /// to do with the mode, and it could only say the other seven were *untested, not refuted*.
    /// **That precondition no longer holds and the table was re-taken**, each mode submitted both
    /// with a rate-control record and with none so that a mode that fails could be told from a record
    /// that fails.
    pub const SUMMARY: &str = "re-measured on iHD 26.3.5 with the packed slice header supplied, which the first sweep omitted and \
         which is why it could only call seven modes untested: VA_RC_CBR, VA_RC_VBR, VA_RC_MB and VA_RC_VCM all ENCODE (mean 9.1/255 \
         from the source, ffmpeg silent); VA_RC_CQP is not swept because it is submitted through the constructor that carries the \
         quantiser; VA_RC_ICQ and VA_RC_QVBR are now REFUTED rather than untested -- both fail at vaEndPicture with \
         VA_STATUS_ERROR_INTERNAL WITH AND WITHOUT a rate-control record, so the failure is the MODE and not the record; and \
         VA_RC_TCBRC fails on the RECORD rather than the mode (vaRenderPicture, invalid VABufferID for 9 encode buffers with \
         CBR's record, and it encodes with none)";
}

/// **The rate-control mask this driver reports for `VAProfileHEVCMain` / `VAEntrypointEncSlice`:**
/// `0x14de`, which is **`VA_RC_CBR | VA_RC_VBR | VA_RC_VCM | VA_RC_CQP | VA_RC_ICQ | VA_RC_MB |
/// VA_RC_QVBR | VA_RC_TCBRC`** (`va.h:1110-1164`).
///
/// **The expansion is eight modes, not four.** It was previously written `CQP | VCM | QVBR | TCBRC`,
/// which is `0x1418` — a number this module does not have and the reader has no way to know that.
/// `0x14de` is `0x14d6 | VA_RC_VCM`, and the eight bits are `CBR 0x02`, `VBR 0x04`, `VCM 0x08`,
/// `CQP 0x10`, `ICQ 0x40`, `MB 0x80`, `QVBR 0x400`, `TCBRC 0x1000`.
///
/// **A measurement, and it is the reason [`VCM`] is a named refusal.** H.264's mask on the same
/// driver is `0x14d6` — the same eight modes less `VA_RC_VCM` — so `VaEncoder::rate_control_names`
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
    ///
    /// # **The parameter sets are left out of the sample, and that is measured**
    ///
    /// The driver's coded buffer for the **first** picture carries the VPS, SPS and PPS — this path
    /// handed them to it as a packed sequence header and it passes them through. **An `hvcC` already
    /// carries those three NAL units**, so a sample that repeats them is a sample carrying its
    /// configuration record inline, and ffmpeg answers with *"Two slices reporting being the first
    /// in the same frame"* and **decodes one frame out of six**. The elementary stream is unaffected
    /// — it is Annex-B and wants them — which is why this shows up only in the muxed file and why
    /// the two access-unit forms are not the same bytes.
    ///
    /// **Only the VCL units go in**, which is what 7.4.2.4.1 defines an access unit as for this
    /// purpose, and the parameter sets reach the decoder through the `hvcC` — which is also the
    /// only way a decoder starting at an arbitrary sample can resolve them.
    pub fn mp4_sample(&self) -> Result<Vec<u8>, String> {
        let units = annex_b_units(&self.bytes)?;
        let mut out = Vec::new();
        for unit in &units {
            // `nal_unit_type < 32` is the VCL range (HEVC Table 7-1): 0-31 are the coded slice NAL
            // types, 32 and above are the parameter sets and the trailing units.
            if !unit.is_vcl() {
                continue;
            }
            let nal = unit.nal_unit();
            let len = u32::try_from(nal.len()).map_err(|_| format!("a {}-byte NAL unit does not fit a four-byte length", nal.len()))?;
            out.extend_from_slice(&len.to_be_bytes());
            out.extend_from_slice(&nal);
        }
        if out.is_empty() {
            return Err("this picture holds no coded slice NAL, so it is not an access unit".into());
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
    /// The record the encoder submits, `None` when it submits none — see
    /// [`HevcEncoder::new_with_attribute`].
    rate_control: Option<RateControl>,
    /// **The `VAConfigAttribRateControl` value submitted to `vaCreateConfig`**, kept so a caller can
    /// read back exactly what was asked of the driver. It is the same number as
    /// `rate_control.config_attribute()` for every mode [`RateControl`] names, and the only thing
    /// that names the mode for a swept one.
    rate_control_attribute: u32,
    rate_control_buffer: Option<Vec<u8>>,
    /// The reference depth the caller asked for.
    refs: usize,
    /// The pictures this encoder holds, oldest first. **The whole of the frame loop's state.**
    references: Vec<HevcReference>,
    /// The sequence record, built once and submitted with every picture.
    sequence: Vec<u8>,
    /// **The parameter sets last handed to the driver as a packed sequence header**, so the next
    /// picture submits them only when they have actually changed. See
    /// [`HevcEncoder::new_picture_has_new_parameter_sets`].
    sent_sets: Option<HevcParameterSets>,
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
    /// **`references` above one is refused, and for HEVC that refusal now rests on HEVC's own
    /// measurement rather than on H.264's** — see `tests/hevenc.rs` for the numbers. Briefly: with
    /// the depth-2 refusal lifted, a depth of 2 **renders, syncs, and decodes in ffmpeg with no
    /// error at all**, and every picture from the third on is **wrong** — the mean absolute
    /// deviation from the source ramp goes 9.1, 9.1, **16.8, 26.6, 36.9, 37.1**, and the third
    /// picture is the first one that reads the second reference, at 137 bytes against the 43-55 of
    /// the frames that do not. HEVC *invites* the request (the driver reports l0 = 3, l1 = 3), which
    /// is exactly why it has to be measured rather than assumed.
    pub fn new(display: &Arc<Display>, size: (u32, u32), references: u32, rate_control: RateControl) -> Result<Self, String> {
        Self::new_with_attribute(display, size, references, rate_control.config_attribute(), Some(rate_control))
    }

    /// **An encoder built under an arbitrary `VAConfigAttribRateControl` value — the re-sweep's
    /// only entrance, and the reason the recorded sweep could be re-taken at all.**
    ///
    /// # Why this exists, and what it is careful about
    ///
    /// [`RateControl`] has three variants, so a caller cannot name `VA_RC_ICQ`, `VA_RC_QVBR`,
    /// `VA_RC_MB`, `VA_RC_TCBRC` or `VA_RC_VCM` through it at all — which is what made the earlier
    /// sweep a table of outcomes for modes nothing could submit. This takes the raw attribute
    /// instead, so a mode the driver advertises can be **submitted and measured** instead of being
    /// described.
    ///
    /// **The rate-control record it submits is CBR's**, taken from `bits_per_second`, and that is
    /// stated here because it is the one thing a reader must know to read the result: the sweep
    /// varies **the config attribute** and holds everything else — the records, the buffer set, the
    /// order — identical to a CBR export, so a mode that works and one that does not differ in the
    /// mode and nothing else. A driver that wants a different record for one of its modes (ICQ's
    /// `icq_quality_factor`, say) is therefore measured against a record it did not ask for, and a
    /// mode that fails here may be failing on the record rather than on the mode. **That is the
    /// weaker claim and it is the one this sweep makes**; a mode that works here is a mode this
    /// submission works, and a mode that fails is only "does not work with CBR's record".
    /// [`HevcEncoder::at_swept_rate_control_without_a_record`] separates the two.
    ///
    /// **`VA_RC_CQP` is refused here.** It is the mode [`HevcEncoder::at_constant_quantiser`]
    /// submits, through the constructor that carries the quantiser's own contract, and an encoder
    /// built here would refuse every request that named one — which is exactly what a constant
    /// quantiser needs.
    pub fn at_swept_rate_control(display: &Arc<Display>, size: (u32, u32), references: u32, attribute: u32, bits_per_second: u32) -> Result<Self, String> {
        if attribute == super::ffi::RC_CQP {
            return Err("VA_RC_CQP is not swept: it is what HevcEncoder::at_constant_quantiser submits, and this entry point builds an \
                        encoder whose requests cannot name the quantiser it needs"
                .into());
        }
        Self::new_with_attribute(display, size, references, attribute, Some(RateControl::Cbr(bits_per_second)))
    }

    /// **A swept mode submitted with no rate-control record at all** — the control for
    /// [`HevcEncoder::at_swept_rate_control`], and what tells a mode apart from a record.
    ///
    /// A mode that fails with a record and encodes without one is **failing on the record**, and the
    /// honest refusal names that; a mode that fails both ways is failing on the mode. **Nothing else
    /// about the submission changes**, so this isolates the one variable the two entry points differ
    /// in — which is the whole reason it is a second function rather than a flag read three frames
    /// away.
    pub fn at_swept_rate_control_without_a_record(display: &Arc<Display>, size: (u32, u32), references: u32, attribute: u32) -> Result<Self, String> {
        if attribute == super::ffi::RC_CQP {
            return Err("VA_RC_CQP is not swept: it is what HevcEncoder::at_constant_quantiser submits".into());
        }
        Self::new_with_attribute(display, size, references, attribute, None)
    }

    /// **The encoder this module builds, under a `VAConfigAttribRateControl` of its own.**
    ///
    /// `rate_control` is the *record* the encoder submits ([`super::hevc_params::rate_control_record`]
    /// reads it) and the fidelity it reports; `attribute` is the value handed to `vaCreateConfig`.
    /// For every mode [`RateControl`] names the two are the same number, and they differ only where
    /// this constructor is called directly with a swept mode — which is why
    /// [`HevcEncoder::config_attribute`] exists and why `rate_control` is an `Option`.
    ///
    /// **`None` submits no record**, which is the shape a constant-quantiser submission has: the
    /// picture record's `pic_init_qp` is then the only statement of the picture's quality, and under
    /// a swept mode that statement is one the caller did not choose — which is why the re-sweep reads
    /// such a result as "this mode runs on the picture record alone" and nothing more.
    fn new_with_attribute(
        display: &Arc<Display>,
        size: (u32, u32),
        references: u32,
        attribute: u32,
        rate_control: Option<RateControl>,
    ) -> Result<Self, String> {
        let refs = references;
        if size.0 == 0 || size.1 == 0 {
            return Err(format!("{}x{} picture", size.0, size.1));
        }
        // **Depth greater than one: measured on this driver before it was refused here.** The HEVC
        // driver reports `VAConfigAttribEncMaxRefFrames` as **l0 = 3, l1 = 3**, which is *more* than
        // H.264's (3, 1) — so unlike H.264, HEVC invites the request. **It was measured with the
        // refusal lifted once a picture could be produced at all**: at depth 2 every call succeeds,
        // ffmpeg decodes without a word of complaint, and every picture from the third on is wrong
        // (mean absolute deviation from the source 9.1, 9.1, 16.8, 26.6, 36.9, 37.1). See
        // `tests/hevenc.rs`.
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
        let encoder = Encoder::new(display, Profile::HevcMain, RtFormat::Nv12, size, attribute, PACKED_HEADERS_FOR_HEVC, coded_buffer_for(size)?, refs)?;
        // **No record under `None` and none under `VA_RC_CQP`**, and the two are different reasons:
        // the first was asked for, the second is the header's own (the picture record's `pic_init_qp`
        // *is* the quality, so a bitrate beside it would be a second statement about one thing).
        let rate_control_buffer = match rate_control {
            None | Some(RateControl::ConstantQuantiser) => None,
            Some(mode) => Some(super::hevc_params::rate_control_record(mode, RATE_CONTROL_WINDOW_MS)?),
        };
        Ok(Self {
            encoder,
            coded: size,
            ctus: ctus_u32,
            poc: 0,
            since_key_frame: 0,
            keyframe_interval: UNSET_INTERVAL,
            rate_control,
            rate_control_attribute: attribute,
            rate_control_buffer,
            refs: refs as usize,
            references: Vec::new(),
            sequence: sequence_record(size, width, height, 0)?,
            sent_sets: None,
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
    /// eight modes (`CBR | VBR | VCM | CQP | ICQ | MB | QVBR | TCBRC` — see
    /// [`HEVC_RATE_CONTROL_MASK`]). **H.264's is `0x14d6` on the same driver** — the same set without
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
    ///
    /// **This is a refusal of `VA_RC_VCM` alone and is not a statement that the rest work** — it is
    /// a refusal of *naming*, and [`hevc_rate_control_outcomes`] measures the modes themselves: the
    /// re-taken sweep has **CBR, VBR, MB and VCM all encoding**, so VCM is here because nothing above
    /// this backend can ask for it rather than because the driver will not run it.
    pub fn refused_rate_control(mask: u32) -> Vec<&'static str> {
        let mut out = Vec::new();
        if mask & VCM != 0 {
            out.push("VA_RC_VCM (variable bitrate with a per-picture cap)");
        }
        out
    }

    /// The record this encoder submits, `None` when it submits none.
    ///
    /// **This is the record, not the mode** — the two are the same number for every mode
    /// [`RateControl`] names and differ for a swept one, so a caller that wants to know what the
    /// driver was asked for reads [`HevcEncoder::config_attribute`] instead.
    pub fn rate_control(&self) -> Option<RateControl> {
        self.rate_control
    }

    /// **The `VAConfigAttribRateControl` value this encoder was created with**, which is what the
    /// driver was asked for and what the re-sweep varies.
    pub fn config_attribute(&self) -> u32 {
        self.rate_control_attribute
    }

    /// **One `VAConfigAttribRateControl` value by name**, for a refusal message and for the re-sweep's
    /// output — the single-value form of [`super::encoder::VaEncoder::rate_control_names`], which
    /// answers for a *mask*.
    ///
    /// A value in neither list is named by its number rather than guessed at, because the honest
    /// answer for an unknown bit is that this tree has no name for it — which is also the answer a
    /// driver offering a mode nobody here can read deserves.
    pub fn rate_control_name(attribute: u32) -> &'static str {
        match attribute {
            super::ffi::RC_CQP => "VA_RC_CQP",
            super::ffi::RC_CBR => "VA_RC_CBR",
            super::ffi::RC_VBR => "VA_RC_VBR",
            super::ffi::RC_ICQ => "VA_RC_ICQ",
            super::ffi::RC_QVBR => "VA_RC_QVBR",
            super::ffi::RC_TCBRC => "VA_RC_TCBRC",
            VCM => "VA_RC_VCM",
            MB => "VA_RC_MB",
            _ => "a rate control",
        }
    }

    /// Whether the driver was told to pick a bitrate rather than being told a quantiser.
    ///
    /// **The attribute decides, not the record.** For every mode [`RateControl`] names the two
    /// agree — only `VA_RC_CQP` names its own quantiser — and the attribute is the one that stays
    /// true for a swept mode, where [`HevcEncoder::rate_control`] reports a record this path
    /// submitted under a mode the record does not describe.
    pub fn rate_controlled(&self) -> bool {
        self.rate_control_attribute != super::ffi::RC_CQP
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
        // **Rebuilt from the same record the encoder was built with**, so a window that changed
        // reaches the driver for every bitrate mode and no constant-quantiser one. `None` and
        // `Some(ConstantQuantiser)` both mean "no record", and the reason they are different states
        // is at [`HevcEncoder::new_with_attribute`].
        self.rate_control_buffer = match self.rate_control {
            None | Some(RateControl::ConstantQuantiser) => None,
            Some(mode) => Some(super::hevc_params::rate_control_record(mode, rate_control_window_ms(pictures))?),
        };
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

    /// **Whether this picture has to carry the packed sequence headers**, decided by comparing the
    /// sets about to be submitted with the ones submitted last.
    ///
    /// **The driver passes them through**: on this driver the VPS, SPS and PPS handed to a
    /// `VAEncPackedHeaderDataBuffer` come back in the coded buffer of the picture they were submitted
    /// with (measured — ffmpeg's stream contains its own 97-byte packed sequence buffer verbatim). So
    /// submitting them with every picture would put three parameter sets in every access unit, and a
    /// stream with three parameter sets per picture is a stream whose second decoder starts at a
    /// parameter set the first already used.
    ///
    /// **The test is the bytes, not a flag**, so a sequence record rebuilt by `at_rate` or
    /// `key_frames_every` — the only two things that can change the sets, and both of which are
    /// refused after the first picture — is caught whichever way it happens.
    fn new_picture_has_new_parameter_sets(&mut self, sets: &HevcParameterSets) -> bool {
        let same = self.sent_sets.as_ref().is_some_and(|sent| sent.vps == sets.vps && sent.sps == sets.sps && sent.pps == sets.pps);
        if same {
            return false;
        }
        self.sent_sets = Some(sets.clone());
        true
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
        // reason: a number silently ignored is a picture at a quality nobody asked for. **The mode
        // named in the message is the attribute's**, which is what was asked of the driver and is
        // right for a swept mode where the record names something else.
        if self.rate_controlled() == request.qp.is_some() {
            let mode = Self::rate_control_name(self.rate_control_attribute);
            return Err(match request.qp {
                Some(qp) => format!("a quantiser of {qp} into a {mode} encoder: the rate controller chooses this picture's quantiser"),
                None => format!("no quantiser into a {mode} encoder: this encoder's quality is the picture record's pic_init_qp"),
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
        // **The slice header, written from the same three records and verified against the parameter
        // sets it is submitted with.** This is not a nicety: under `VA_ENC_PACKED_HEADER_SLICE` the
        // driver writes no slice header of its own — it writes the bytes it was given and then the
        // slice *data* — so a header this module does not write is a stream whose slices name nothing.
        let header = super::hevc_params::slice_header(&sequence_read, &picture_read, &slice_read, &derived)?;
        let seq_buf = self.encoder.buffer(ffi::ENC_SEQUENCE_PARAMETER_BUFFER, &self.sequence, 1)?;
        // **The packed sequence header pair**, which `va_enc_hevc.h:333-340` says the driver expects
        // in the *same* `vaRenderPicture` call as the sequence record whenever the pipeline was
        // configured with `VA_ENC_PACKED_HEADER_SEQUENCE`.
        //
        // **Submitted with the first picture and again only if the sets change**, which is what
        // ffmpeg's own `hevc_vaapi` does on this driver and what the driver's own pass-through
        // implies: the VPS, SPS and PPS come back in the coded buffer of the picture they were
        // submitted with, so submitting them with every picture would put three parameter sets in
        // every picture. What "change" means is decided by the bytes, not by a flag: see
        // [`HevcEncoder::new_picture_has_new_parameter_sets`].
        let sets_sent = self.new_picture_has_new_parameter_sets(&derived);
        let ph_seq_data = match sets_sent {
            true => Some(self.encoder.buffer(PACKED_HEADER_DATA_BUFFER_TYPE as _, &parameter_sets_annex_b(&derived), 1)?),
            false => None,
        };
        let ph_seq_param_buf = match sets_sent {
            true => Some(self.encoder.buffer(
                PACKED_HEADER_PARAMETER_BUFFER_TYPE as _,
                &packed_header_parameter(super::hevc_enc::PACKED_HEADER_VPS_OR_SPS, &parameter_sets_annex_b(&derived)),
                1,
            )?),
            false => None,
        };
        let pic_buf = self.encoder.buffer(ffi::ENC_PICTURE_PARAMETER_BUFFER, &picture, 1)?;
        let slice_buf = self.encoder.buffer(ffi::ENC_SLICE_PARAMETER_BUFFER, &slice, 1)?;
        let fps_buf = self.encoder.buffer(ffi::ENC_MISC_PARAMETER_BUFFER, &frame_rate, 1)?;
        // **The packed slice header pair**, with every slice, for the same reason and the same
        // header's instruction (`va_enc_hevc.h:299-309`).
        let ph_slice_data = self.encoder.buffer(PACKED_HEADER_DATA_BUFFER_TYPE as _, &header, 1)?;
        let ph_slice_param = self.encoder.buffer(PACKED_HEADER_PARAMETER_BUFFER_TYPE as _, &packed_header_parameter(PACKED_HEADER_SLICE, &header), 1)?;
        let rc_buf = self.rate_control_buffer.as_ref().map(|record| self.encoder.buffer(ffi::ENC_MISC_PARAMETER_BUFFER, record, 1));
        let rc_buf = rc_buf.transpose()?;
        // **Every handle is alive across the render.** `Buffer::drop` destroys its libva buffer, and
        // an `if let` that moved one into an arm dropped it before the render named its id — which is
        // exactly what iHD answers with `VA_STATUS_ERROR_INVALID_BUFFER`. See the H.264 note at the
        // same place.
        //
        // **No `VAEncMiscParameterTypeQuantization` buffer, which H.264's path submits and this one
        // measured as refused** — see [`REFUSES_THE_QUANTIZATION_MISC_BUFFER`].
        //
        // **The refusal is a branch rather than an absence**, and that is deliberate: a buffer this
        // driver refuses is one whose submission would turn every encode into
        // `vaRenderPicture → VA_STATUS_ERROR_INVALID_BUFFER`, so "this path does not submit one" and
        // "this path has measured that it must not" are different claims and only the second one is
        // worth anything. Reading the decision from the constant means a change to the measurement
        // changes the submission, and the oracle test can see it.
        // **The order ffmpeg's own `hevc_vaapi` uses on this driver**, measured by capturing its
        // `vaRenderPicture` calls: `[sequence, frame rate, picture, packed sequence parameter,
        // packed sequence data, packed slice parameter, packed slice data, slice]`, dropping the two
        // packed-sequence buffers on every picture after the first. The header does not state an
        // order; this is what a submission this driver accepts looks like, and it is recorded as a
        // measurement rather than a preference.
        let mut buffers = vec![seq_buf.id(), fps_buf.id(), pic_buf.id()];
        if let (Some(param_buf), Some(data)) = (&ph_seq_param_buf, &ph_seq_data) {
            buffers.push(param_buf.id());
            buffers.push(data.id());
        }
        buffers.push(ph_slice_param.id());
        buffers.push(ph_slice_data.id());
        buffers.push(slice_buf.id());
        if let Some(ref buf) = rc_buf {
            buffers.push(buf.id());
        }
        if submits_the_quantization_misc_buffer() {
            // **Unreachable on any driver measured so far**, and the `Err` rather than a `panic!` or
            // a silent skip is the point: if a driver ever accepts the buffer, the submission must
            // be *built* for it, and this says so rather than half-implementing a fourth record.
            return Err("this driver was measured to accept VAEncMiscParameterTypeQuantization for HEVC, so the \
                        submission path has to build and submit it. That is a new measurement and a new buffer, not a flag"
                .into());
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
        // **The coded bytes are the whole access unit, as the driver wrote it.** It passed our packed
        // sequence header through verbatim into the first picture, so the VPS, SPS and PPS are already
        // the first three NAL units of that picture — **prepending them again would put six
        // parameter sets in a six-picture stream and a decoder would resolve its slices against the
        // second set**, which is this path's own sets again and so would still decode, which is
        // exactly why nothing would notice.
        self.sample_entry = Some(derived);
        Ok(HevcEncodedPicture {
            bytes: coded.bytes().to_vec(),
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
    ///
    /// **`> 0` is the load-bearing part, and it was missing.** An interval of 0 is the legal value
    /// for "one key frame for the whole stream" that `key_frames_every` documents, and
    /// `since_key_frame + 1 >= 0` is true for every picture — so every picture after the first was
    /// forced to be an IDR. **This could not be seen until a picture came back**: with no picture
    /// there was no stream to look at, and the six-frame probe reported six key frames where it was
    /// asked for one I and five P. It is the H.264 path's test, unchanged, because that one has an
    /// exported stream behind it.
    fn key_frame_interval_elapsed(&self) -> bool {
        self.keyframe_interval != UNSET_INTERVAL && self.keyframe_interval > 0 && self.since_key_frame >= self.keyframe_interval
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
        // **The VUI is on**, and the timing in it is the frame rate the caller stated. **Where it is
        // not** — the constructor path below, before `at_rate` has been called — it is
        // `DEFAULT_FRAME_RATE_FOR_LEVEL`, which is why `at_rate` rebuilds this record before any picture
        // is encoded. `check_hvcc_against_the_picture` refuses the mismatch rather than letting it ship.
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

/// The frame rate used for the VUI and for [`level_for`] **until `at_rate` supplies the caller's own**.
///
/// **This number does reach a stream.** The sequence record's `vui_time_scale` is written as
/// `2 * DEFAULT_FRAME_RATE_FOR_LEVEL`, so a record built before `at_rate` states a rate carries 30 fps in
/// its VUI whatever the caller asked for. `at_rate` rebuilds the record with the caller's rate before any
/// picture is encoded, and `check_hvcc_against_the_picture` refuses a sequence whose `hvcC` rate and SPS VUI
/// rate disagree — so the defect is caught rather than shipped, but it is a defect, not a neutral default.
/// Measured: 30 fps exports; 24, 25 and 60 all fail by name. Fixing it means threading the rate through
/// `sequence_record`, not widening a check.
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

/// Whether this submission will carry a `VAEncMiscParameterTypeQuantization` buffer — **which is
/// [`REFUSES_THE_QUANTIZATION_MISC_BUFFER`] saying no, read through the value the submission path
/// itself branches on** rather than a constant quoted in a comment.
///
/// It exists so the refusal is a *value the code consults* and not only a note: a caller that wants
/// to know whether the trellis-control buffer is in play asks this, and a change that started
/// submitting one would change the answer and be caught by the oracle test rather than by a reader
/// who happens to scroll to the right paragraph.
///
/// **It is a constant and not a query**, because the measurement is of *this driver*, not of the
/// record: re-asking it would mean a `vaRenderPicture` per picture, which is not a measurement but a
/// cost. The honest shape is the one H.264's path has — the decision is made once, named, and
/// documented — and this is that decision with a getter.
pub const fn submits_the_quantization_misc_buffer() -> bool {
    !REFUSES_THE_QUANTIZATION_MISC_BUFFER
}

/// One NAL unit of an Annex-B run, as its **two-byte** HEVC NAL header and the payload after it.
///
/// **Two bytes, and the second one is the whole point.** `nal_unit()` is what an `hvcC` stores and
/// what the slice-header parser is handed, and it reads `(nal[0] >> 1) & 0x3f` for the type and then
/// starts the RBSP at `nal[2]`. A walk that keeps only `nal[0]` therefore hands the parser a RBSP
/// that begins one byte early — at the second header byte — so every syntax element after
/// `nuh_temporal_id_plus1` is read one byte out of step.
///
/// **It read as `missing parameter set: picture parameter set 1` on a stream whose PPS is id 0**,
/// which is the kind of error that names something three fields away from the mistake. It could not
/// fire before this path produced a picture: `annex_b_units` is only ever called on a coded buffer,
/// and no submission of this path had produced one.
struct NalUnit<'a> {
    header: [u8; 2],
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
    /// `(header[0] >> 1) & 0x3f` and the second byte belongs to the unit.
    fn nal_unit(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.body.len() + 2);
        out.extend_from_slice(&self.header);
        out.extend_from_slice(self.body);
        out
    }

    /// The NAL unit type, which for HEVC is six bits at offset 1 of the first header byte.
    fn unit_type(&self) -> u8 {
        (self.header[0] >> 1) & 0x3f
    }

    /// Whether this is a **coded slice** NAL — `NAL_UNIT_TYPE` 0 to 31, the VCL range of HEVC's
    /// Table 7-1. Everything from 32 up is a parameter set, a prefix SEI, an access-unit delimiter or
    /// one of the trailing units, none of which is picture data.
    fn is_vcl(&self) -> bool {
        self.unit_type() < 32
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
        // **HEVC's NAL header is two bytes**, so both are taken: the second carries the low five bits
        // of `nuh_layer_id` and `nuh_temporal_id_plus1`, and the payload begins after it.
        let (Some(&first), Some(&second)) = (bytes.get(start), bytes.get(start + 1)) else { break };
        at = start + 2;
        let end = bytes.get(at..).and_then(|rest| rest.windows(3).position(|w| w == [0, 0, 1])).map_or(bytes.len(), |next| at + next);
        let Some(body) = bytes.get(at..end) else { break };
        // An empty body is legal only for the two types the specification defines as carrying
        // nothing; anything else with no payload is a truncated slice, which is what this readback
        // exists to catch.
        if body.is_empty() && !matches!((first >> 1) & 0x3f, 35 | 36) {
            return Err(format!(
                "the coded picture holds a NAL unit of type {} with no payload, which is a truncated slice rather than an end marker",
                (first >> 1) & 0x3f
            ));
        }
        units.push(NalUnit { header: [first, second], body });
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
///
/// **This is the *packed header* the driver is given, not an addition to the output.** The driver
/// passes it through into the coded buffer verbatim (measured), so prepending it again would put a
/// second copy of every parameter set in the first picture.
/// The `VAEncPackedHeaderParameterBuffer` that names a packed header of `header_type` and says how
/// long its data buffer is.
///
/// **`bit_length` is bits, not bytes** (`va.h:2450`), and ffmpeg's own submission on this driver
/// measures it: a 97-byte VPS+SPS+PPS buffer is declared as 776 bits, and an 8-byte slice header as
/// 64. A record carrying the byte count instead is a header the driver is told is eight times too
/// long.
///
/// **`has_emulation_bytes` is 1**, because these headers are Annex-B with start codes and the
/// payload has already been escaped by [`super::hevc_params`]' writers.
fn packed_header_parameter(header_type: u32, data: &[u8]) -> Vec<u8> {
    let bits = u32::try_from(data.len().saturating_mul(8)).unwrap_or(u32::MAX);
    let record =
        VaEncPackedHeaderParameterBuffer { header_type, bit_length: bits, has_emulation_bytes: 1, padding_and_reserved: [0; 3 + ffi::VA_PADDING_LOW * 4] };
    ffi::record_bytes(&record)
}

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
/// **Parse the slice header in the coded picture and check it against the records this module
/// submitted.**
///
/// **What this is worth is narrower than it looks, and saying so is the point.** The driver does not
/// write this slice header: it writes the bytes this module supplied as a packed slice header and
/// then the slice data. **So this checks FilmCraft's own header against FilmCraft's own records**,
/// which is worth doing — it is a round trip through the decoder's parser on every single picture,
/// and it is what catches a header whose bits do not say what the record says — but **it is not
/// evidence about the driver**, and the independent read is ffmpeg's `trace_headers` over the
/// exported stream (`examples/vahevcprobe.rs --trace`).
///
/// **It is still the strongest check available on this path**, because everything else this module
/// submitted is a record the driver read and nothing comes back except the bitstream.
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
    let pps =
        filmcraft_hevc::params::Pps::parse(&rbsp_of(&sets.pps, NAL_PPS)?).map_err(|e| format!("the PPS the sample entry would carry does not parse: {e}"))?;
    let sps =
        filmcraft_hevc::params::Sps::parse(&rbsp_of(&sets.sps, NAL_SPS)?).map_err(|e| format!("the SPS the sample entry would carry does not parse: {e}"))?;
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
    // `num_ref_idx_l0_active_minus1 + 1` in the record**, and it is **only comparable on a predicted
    // picture**: `num_ref_idx` is inside the `slice_type == P || slice_type == B` chain, so an I
    // slice's header has none at all and the parser leaves it at 0. Comparing it anyway reads that 0
    // as "a key frame that reads no reference" — which is not a finding, it is the syntax the record
    // asked for.
    //
    // **And a key frame must not read a reference at all** (8.3.1: an IDR empties the decoded picture
    // buffer), which is stated by refusing a non-I slice on an IDR when the header is built rather
    // than by counting references in a header that cannot carry them.
    if written != 2 {
        if header.num_ref_idx[0] != u32::from(slice_record.num_ref_idx_l0_active_minus1) + 1 {
            mismatches.push(format!(
                "the slice header reads {} reference(s) and the record said {}",
                header.num_ref_idx[0],
                u32::from(slice_record.num_ref_idx_l0_active_minus1) + 1
            ));
        }
        if key_frame {
            mismatches.push("an IDR whose slice header carries a reference index".into());
        }
    }
    if mismatches.is_empty() {
        return Ok(());
    }
    Err(format!("the slice the driver wrote disagrees with the records this module submitted: {} — {} mismatch(es)", mismatches.join(", "), mismatches.len()))
}

/// **The RBSP of the NAL unit of type `nal_type` inside `unit`: header stripped, emulation prevention
/// bytes removed.** What [`filmcraft_hevc::params`]' parsers take, and *not* what
/// [`unescape_rbsp`] takes.
///
/// **The two are different and the difference is a two-byte shift of every syntax element.**
/// `strip_nal` returns the unit *including* its two-byte NAL header, which is what a writer needs and
/// what an Annex-B parser wants; feeding that straight to a parser reads `nal_unit_type` as the first
/// `ue(v)` and every field after it off by sixteen bits. **It read as `chroma QP offset out of
/// range`, thirty-odd fields from the mistake** — and it could not fire before, because this
/// function only runs on a coded picture and no submission of this path had ever produced one.
fn rbsp_of(unit: &[u8], nal_type: u8) -> Result<Vec<u8>, String> {
    let full = strip_nal(unit, nal_type)?;
    let Some(payload) = full.get(2..) else {
        return Err(format!("the {nal_type} NAL unit is a one-byte header, and HEVC's is two"));
    };
    Ok(unescape_rbsp(payload))
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
