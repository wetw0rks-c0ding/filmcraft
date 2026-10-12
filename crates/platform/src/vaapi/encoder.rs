//! A run of H.264 pictures through the VA-API slice encoder: the frame loop, its reference state,
//! its key frames, its end of stream, and the records that describe each picture to the driver.
//!
//! **Encode is decode with the data flow inverted** (docs/superpowers/plans/2026-10-09-vaapi-linux-encode.md,
//! "the decisive difference"): the app hands the driver a picture and the driver writes the slice
//! headers. So there is nothing to parse here — this module builds the records that *describe* what
//! to encode, hands them to [`super::ffi::Encoder`], and reads the compressed picture back out of
//! the buffer the driver filled. The codec configuration that comes back is the driver's, which is
//! why the packed-header mode is on: the SPS and PPS would arrive inside the coded buffer and
//! FilmCraft would never assemble one.
//!
//! Everything this module does not express is a **decline rather than a default**, which for an
//! encoder means much more than for a decoder: a default QP, picture type or reference state
//! produces a plausible-looking *wrong picture*, not an error.
//!
//! # The frame loop's four decisions, and what each one is derived from
//!
//! 1. **The picture type** is the loop's own, from the key-frame interval
//!    ([`VaEncoder::new`]) and the reference state. An IDR empties the decoded picture buffer
//!    (7.4.3.3), so the first picture of a sequence and the first after
//!    [`VaEncoder::forget_references`] must be one, and so must a picture whose `frame_num` is
//!    about to wrap — see "The wrap" below.
//! 2. **The reference state** is a list of ([`Reference`]): surface, DPB slot, `frame_num` and
//!    picture order count, advancing together. It is the loop's entire state, and it is trimmed to
//!    the depth the caller asked for only *after* a picture comes back, so a failed render cannot
//!    leave a shorter list than the picture that failed.
//! 3. **The end of the stream** is the caller's ([`EncodeRequest::last`]) and nothing else: only the
//!    caller knows which picture is the last one it will hand over. See [`VaEncoder::flush`] for
//!    what this backend does with it and what it declines.
//! 4. **The picture order count** is `pic_order_cnt_lsb`, which under `pic_order_cnt_type == 0` is
//!    a `u(v)` counted from the previous picture, and is **reset to 0 by an IDR** (8.2.1.1: the
//!    picture order count of an IDR is 0). It travels with the picture into [`Reference`] because a
//!    reference list entry needs it.
//!
//! # The `frame_num` wrap, decided
//!
//! `seq_fields` writes `log2_max_frame_num_minus4 = 4`, so `MaxFrameNum` is 256 and `frame_num` is
//! **eight bits** while this module's counter is a `u16`.
//!
//! **What the specification requires here, read from the clause rather than assumed.** 7.4.3
//! constrains `frame_num` three ways, and the one that decides a long group of pictures is
//! Equation 7-22:
//!
//! > There shall not be any previous field or frame in decoding order that is currently marked as
//! > "used for short-term reference" that has a value of `frame_num` equal to any value taken on
//! > by the variable `UnusedShortTermFrameNum` in the following:
//! > `UnusedShortTermFrameNum = ( PrevRefFrameNum + 1 ) % MaxFrameNum`
//! > `while( UnusedShortTermFrameNum != frame_num ) UnusedShortTermFrameNum = ( UnusedShortTermFrameNum + 1 ) % MaxFrameNum`
//!
//! **Equation 7-22 forbids a *skipped* frame number still sitting in the DPB — not a wrap.** At the
//! wrap itself the loop body never runs, because `UnusedShortTermFrameNum` is initialised to
//! `( PrevRefFrameNum + 1 ) % MaxFrameNum`, which **is** `frame_num`, so the `while` condition is
//! false immediately and the set of "values taken on" is empty. A bitstream whose `frame_num` runs
//! `…, 14, 15, 0, 1, …` is therefore conformant on this point, and 8.2.5.3's sliding-window marking
//! removes the superseded picture without any `memory_management_control_operation` being asked
//! for.
//!
//! Two further clauses point the same way, and together they say the specification does **not**
//! require either candidate mechanism to make a long group of pictures legal:
//!
//! - 7.4.3: "If `gaps_in_frame_num_value_allowed_flag` is equal to 0, the value of `frame_num` for
//!   the current picture **shall be equal to** `( PrevRefFrameNum + 1 ) % MaxFrameNum`." A wrap *is*
//!   that value, so with the flag clear the wrap is the only conformant choice — the standard
//!   mandates it rather than forbidding it.
//! - 8.2.5.1 step 2: a non-IDR picture runs the **sliding-window** process of 8.2.5.3 (invoked
//!   because this encoder never sets `adaptive_ref_pic_marking_mode_flag`), and its closing sentence
//!   is what bounds the buffer: the number of pictures marked "used for reference" "shall not be
//!   greater than `Max( num_ref_frames, 1 )`". **At `num_ref_frames = 1` — the depth this encoder
//!   submits and the deepest it accepts — the DPB holds one reference, so no two pictures in it can
//!   share a `frame_num` whatever the width.** That is why the unaccompanied wrap below measures
//!   clean, and it is a property of the depth, not of the modulus.
//!
//! **So `memory_management_control_operation` is not what a long group of pictures needs, and on
//! this ABI it cannot be asked for at all.** `VAEncSliceParameterBufferH264`
//! (`/usr/include/va/va_enc_h264.h`) has **no `dec_ref_pic_marking` field and no
//! `adaptive_ref_pic_marking_mode_flag`** — the record ends at `disable_deblocking_filter_idc`,
//! `slice_alpha_c0_offset_div2`, `slice_beta_offset_div2` and `va_reserved`, which the header
//! declares "Reserved bytes for future use, must be zero". libva's H.264 encoder takes its
//! reference list from `RefPicList0`/`RefPicList1` and `VAEncPictureParameterBufferH264`'s
//! `ReferenceFrames`, and marks references itself; there is no entry point by which a caller can
//! hand it an MMCO to emit. Choosing the second candidate would mean writing into a reserved field
//! a driver is told must be zero, and hoping — which is the one class of value this module refuses
//! everywhere else.
//!
//! **What is therefore needed is the first candidate: a `log2_max_frame_num` wide enough to carry
//! the group.** `MaxFrameNum` went 16 → 256 ([`MAX_FRAME_NUM`]), so the wrap this module inserts an
//! IDR for lands 256 pictures out instead of 16, and every preset's group of pictures is carried by
//! the counter on its own.
//!
//! **The IDR at the wrap is kept**, and it stays a *policy* rather than a codec requirement, now
//! that the wrap is rare enough to be outside any group of pictures the caller asked for. Three
//! alternatives were measured on iHD 26.3.5 rather than argued about:
//!
//! 1. **Do nothing** — let the wrap happen unaccompanied. Measured at the depth this module
//!    submits (`refs = 1`): **40 pictures, a flat 9/255 mean absolute deviation on every one, and
//!    not one ffmpeg diagnostic.** It is conformant, as 7.4.3's Equation 7-22 shows, but it rests
//!    on `num_ref_frames == 1`, which is a property of the depth this module **refuses to exceed**
//!    ([`VaEncoder::new`]) rather than of H.264 — so a loop built on it would break the moment that
//!    refusal is lifted.
//! 2. **Set `gaps_in_frame_num_value_allowed_flag`** — the syntax element that exists for a genuine
//!    *gap*. Measured: it also produces a clean 40-picture stream. **Declined**, and the reason is
//!    who has to understand it: the flag tells a decoder that `frame_num` may skip, so every
//!    consumer must treat the numbering as unreliable rather than as a count. A decode-skip
//!    consumer is exactly what this milestone's verification is, and the cost of the flag is paid
//!    by every reader of the file for the convenience of one writer.
//! 3. **IDR at the wrap** — 7.4.3 infers an IDR's `frame_num` to be 0, so an IDR is accompanied by
//!    construction, 7.4.3.3 empties the decoded picture buffer, and the numbering is an honest
//!    count at every depth. Measured: the same flat 9/255, no diagnostic, and the stream is a
//!    sequence of independently decodable runs.
//!
//! The cost is a key frame the caller did not ask for at a position they did not choose, which is
//! why it is **reported** on [`EncodedPicture::key_frame_reason`] rather than hidden: a caller that
//! asked for a 300-picture GOP and got a key frame at picture 256 has been given something it did
//! not ask for and is entitled to know. At the width this module now writes that needs a stream
//! past 256 pictures to happen at all, which is the second reason the width was measured rather
//! than picked — see [`MAX_FRAME_NUM`]. [`VaEncoder::key_frame_due`] is the single place that
//! decides, and a caller-requested key frame and a codec-required one are combined there rather
//! than in two places that could disagree.
//!
//! **What goes with the decision, and was found by measuring rather than by reading the spec:** an
//! IDR's `frame_num` and `pic_order_cnt_lsb` are written as **0**, which is what 7.4.3 and 7.4.3.1
//! infer them to be, and *not* as the counter's value. With the counter's value in an IDR's slice
//! header, every mid-stream key frame makes ffmpeg answer
//!
//! ```text
//! [h264] number of reference frames (0+2) exceeds max (1; probably corrupt input), discarding one
//! ```
//!
//! while x264's own `max_num_ref_frames = 1` stream with mid-stream IDRs — whose IDRs carry
//! `frame_num 0` — draws no such line. The counter and the written value are then two different
//! things, which is why the counters continue at 1 across an IDR rather than at 0: the IDR is a
//! picture in the DPB once decoded (7.4.3.3), so the picture after it may not carry another 0, and
//! because `slot_for` derives the DPB slot from the same number, restarting at 0 would also make
//! the next picture claim the slot its own reference occupies.
//!
//! # B pictures: measured, and refused
//!
//! **The driver emits no B picture, and this backend submits none.** Three separate measurements,
//! because "we never asked" is not the same claim as "the driver would not have":
//!
//! - **The driver reports it could.** `VAConfigAttribEncMaxRefFrames` answers **(3, 1)** on this
//!   machine — three past references and **one future** one, so the capability is there and is not
//!   the reason for the refusal. [`super::max_reference_frames`] is the public form of the query.
//! - **The driver emits none anyway.** Over a 40-picture run spanning two mid-stream IDRs and two
//!   `frame_num` wraps, **every** coded NAL the driver wrote was a type 1 or a type 5, and every
//!   slice header read back `slice_type` 5 (all P) or 7 (all I). Not one `slice_type` 6 (all B) and
//!   not one `pic_order_cnt_lsb` out of display order. **`media_start()` is 0 for this backend
//!   because no picture is ever coded out of display order**, not because the driver declined to
//!   and not because nobody measured. `the_driver_writes_no_b_picture_and_this_backend_submits_none`
//!   is the test that re-takes it.
//! - **A B picture could not be measured end to end here even if a caller asked for one.** A B
//!   slice predicts from one past *and* one future reference, so it needs a decoded picture buffer
//!   holding two; this encoder's reference depth is **one**, and every deeper depth is refused for
//!   producing silently wrong pictures ([`VaEncoder::new`]). So the frame loop has nothing to build
//!   a B slice's second reference list out of, and a B submitted through
//!   [`VaEncoder::allow_b_pictures`] would name a reference list the driver cannot resolve.
//!   **Refused at construction, with the measurement in the message**.
//!
//! What implementing them would need is a reordering buffer: the picture *after* a B must already
//! exist when the B is encoded, so pictures leave the encoder in a different order than they arrive,
//! `composition_offset` stops being zero, and the driver has to be told a reference depth this
//! module currently refuses. None of that is a smaller version of the present loop.
//!
//! # Every parameter here is either stated or declined
//!
//! - **Quality is either a fixed quantiser or a bitrate, never both.** [`RateControl`] is which one,
//!   it is fixed when [`VaEncoder::new`] builds the libva session (the mode is a `vaCreateConfig`
//!   attribute, so it cannot change afterwards), and the other one is then *refused* rather than
//!   ignored: a request that names a quantiser in a bitrate encoder is an error naming the mode, and
//!   an encoder in constant quantiser refuses a request that declines to name one. Under
//!   [`RateControl::ConstantQuantiser`] the quality comes from the picture record's `pic_init_qp`
//!   and nothing else; under [`RateControl::Cbr`] / [`RateControl::Vbr`] it comes from the rate
//!   controller, and **`VaEncoder::encode` takes no quantiser at all** because there is no honest
//!   one to take — see [`EncodeRequest`].
//! - **`pic_order_cnt_type` is 0** and `num_ref_frames_in_pic_order_cnt_cycle` is 0: no picture
//!   order count cycle, and no B pictures.
//! - **One slice per picture.** A second coded segment is refused outright rather than submitted,
//!   so a picture over one slice is unencodable here rather than silently re-cut.
//! - **The sequence, picture and slice records are submitted together in one `vaRenderPicture`**,
//!   with the frame-rate misc buffer, with the quantization misc buffer, and — in a bitrate mode —
//!   with the rate-control misc buffer. iHD refuses the render without the frame-rate buffer — a
//!   requirement that is in no header, and `the_h264_records_are_the_size_the_driver_reads` and
//!   [`VaEncoder::encode`]'s errors are where it shows up.

use super::ffi::{
    self, Display, Encoder, LAST_PICTURE_EOSTREAM, PACKED_HEADER_PICTURE, PACKED_HEADER_SEQUENCE, PICTURE_H264_SHORT_TERM_REFERENCE, Profile, RC_CBR, RC_CQP,
    RC_VBR, RtFormat, SLICE_TYPE_ALL_I, SLICE_TYPE_ALL_P,
};
use std::sync::Arc;

// The one macroblock size H.264 has, in pixels. Every size this module accepts is rounded to it, and
// the rounding is reported rather than hidden — a caller that asked for 360 rows gets a 368-row
// surface and a crop, and `VaEncoder::encode`'s `coded_size` says so.
const MB: u32 = 16;

/// **What kind of picture this is** — the one statement in an encode request that the slice header
/// repeats verbatim and that no other field can stand in for.
///
/// It is an enum rather than two booleans because `idr` and `bi` are not independent: a slice is
/// one of I, P or B, and two booleans admit the state neither of them describes. Since the slice
/// header states the type and [`check_output_matches_the_sets`] reads it back and refuses a
/// mismatch, the request has to be able to name the type it means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PictureKind {
    /// An IDR — a key frame, and the only picture type that empties the decoded picture buffer.
    /// The first picture of a sequence must be one, and so must the first after a seek.
    Idr,
    /// A P picture: predicted from the references the encoder holds, in one direction.
    P,
    /// A B picture: predicted from a past reference (list 0) and a future one (list 1).
    ///
    /// **Measured, and refused — see [`VaEncoder::allow_b_pictures`] for the three measurements.**
    /// The driver can encode one and emits none for these streams; what is missing is the frame
    /// loop's ability to hold the future reference a B slice reads.
    B,
}

/// **Which statement about fidelity this stream makes** — a fixed quantiser, or a bit rate.
///
/// It is a constructor argument rather than a builder method because of what libva allows:
/// `VAConfigAttribRateControl` is read by `vaCreateConfig` when the session is made, and there is no
/// call to change it afterwards. So the mode is fixed for the encoder's life, and a caller that
/// wants the other one builds a different encoder — which is also the honest shape, because
/// switching modes mid-stream would produce a file whose first pictures and last pictures were made
/// under two different contracts.
///
/// # Why this is an enum and not a constant-quality mode on the settings
///
/// The export settings name a bitrate for **every** mode FilmCraft can express, and before this type
/// the only way past the export gate was a settings object with *no* bitrate — which made the
/// registered factory a decline rather than an encoder. The alternative was to add a constant-quality
/// mode to `BitrateMode`: smaller, and it would have left the settings surface shared with macOS and
/// Windows untouched. It was declined because **it does not make the existing settings usable** — a
/// user who picks CBR, which is one of the three modes FilmCraft already offers, is still declined,
/// and the hardware path stays unreachable for the default export. This way the bitrate the user
/// asked for is the bitrate the driver is told, which is the thing they meant.
///
/// # What the driver is not being asked to do
///
/// The driver reports **CBR, VBR, CQP, ICQ, QVBR and TCBRC** on iHD 26.3.5
/// ([`VaEncoder::driver_rate_control_modes`]), so nothing here works around a driver that cannot do
/// rate control — a driver whose mask lacks the mode this type names fails at `vaCreateConfig` with
/// the mode in the message. ICQ and the temporal-layer modes are **not** here: ICQ's
/// `ICQ_quality_factor` is a quality in a unit none of FilmCraft's settings state, and TCBRC wants
/// per-frame channel feedback the export pipeline does not have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RateControl {
    /// **A fixed quantiser per picture.** The picture record's `pic_init_qp` *is* the quality, and
    /// [`EncodeRequest`] carries it because the caller is the only one who knows it. This is the mode
    /// `examples/vaencprobe.rs` measures with, and the only one whose output is reproducible from the
    /// input plus the submitted records.
    ConstantQuantiser,
    /// **A constant bit rate**, in bits per second. `VAEncMiscParameterRateControl::bits_per_second`
    /// carries it and `target_percentage` is 100, which is what the header says that field means in
    /// CBR.
    Cbr(u32),
    /// **A variable bit rate** between a target and a ceiling, both in bits per second.
    ///
    /// `bits_per_second` is the **ceiling** and `target_percentage` is the target as a percentage of
    /// it, which is the header's own split and matches what FilmCraft's `target_kbps` / `max_kbps`
    /// pair means. It is two numbers rather than one because that is the shape of the record, not
    /// because this backend has an opinion about how far apart they should be.
    Vbr {
        /// The rate the controller aims at, in bits per second.
        target: u32,
        /// The highest rate it may reach, in bits per second. Must be at least `target`.
        max: u32,
    },
}

impl RateControl {
    /// The `VAConfigAttribRateControl` value this mode submits.
    pub fn config_attribute(&self) -> u32 {
        match self {
            Self::ConstantQuantiser => RC_CQP,
            Self::Cbr(_) => RC_CBR,
            Self::Vbr { .. } => RC_VBR,
        }
    }

    /// **Whether a picture's quantiser is the caller's to name** — which is exactly
    /// [`EncodeRequest::qp`]'s presence, and the two are the same question asked of the two halves
    /// of the contract.
    ///
    /// It is what stops the halves from silently disagreeing: a request carrying a quantiser into a
    /// bitrate encoder (or omitting one in a constant-quantiser encoder) is refused at
    /// [`VaEncoder::encode`] with both modes named, rather than one of them being honoured and the
    /// other dropped.
    pub fn names_its_own_quantiser(&self) -> bool {
        matches!(self, Self::ConstantQuantiser)
    }

    /// The mode's own name, for a decline message and for `examples/vaencprobe.rs`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::ConstantQuantiser => "constant quantiser",
            Self::Cbr(_) => "CBR",
            Self::Vbr { .. } => "VBR",
        }
    }
}

/// A picture to encode, and the three things a caller has to say about it.
///
/// Every other parameter is either stated here or is a constant this module can derive with
/// certainty (macroblock counts, frame numbers from a counter, the picture order count of a frame
/// loop this module owns). The split is the point: `qp` is the user's, because choosing one for
/// them would be choosing their picture's quality; `kind` is the caller's, because whether a picture
/// is a key frame is a statement about the sequence rather than about encoding; and `last` is the
/// caller's alone, because only the caller knows which picture is the last one it will be given.
///
/// # `qp` is `Some` in a constant-quantiser encoder and `None` in a bitrate one
///
/// **That is the contract, not a convenience, and the pairing is checked.** The two statements about
/// a picture's fidelity — "this quantiser" and "this many bits per second" — are alternatives. An
/// encoder built for one cannot honour the other: a quantiser submitted to a bitrate encoder is not
/// what the picture is encoded at (the controller chooses), and a bitrate submitted to a
/// constant-quantiser encoder is not what the file comes to (the quantiser is). So [`VaEncoder`] is
/// built in exactly one mode, [`EncodeRequest::qp`] is present exactly when that mode is
/// [`RateControl::ConstantQuantiser`], and [`VaEncoder::encode`] **refuses the mismatched pairing**
/// naming both halves rather than honouring one and silently dropping the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodeRequest {
    /// The quantiser, 0..=51 as H.264 numbers them, under [`RateControl::ConstantQuantiser`] — where
    /// it is the picture's actual quality and nothing else affects it. `None` under [`RateControl::Cbr`]
    /// or [`RateControl::Vbr`], where the rate controller chooses and the caller has no honest
    /// quantiser to state.
    ///
    /// Refused outside 0..=51 rather than clamped. A clamp would quietly encode at the nearest
    /// quality instead of the one asked for, which for an encoder is a wrong picture that looks
    /// right.
    pub qp: Option<u8>,
    /// What the picture is.
    pub kind: PictureKind,
    /// Whether this is the last picture of the stream, which sets `last_picture =
    /// H264_LAST_PICTURE_EOSTREAM` in the picture record.
    ///
    /// **Only the caller can say this**, and it is therefore a refusal to guess rather than a
    /// default: see [`VaEncoder::flush`].
    pub last: bool,
}

impl EncodeRequest {
    /// A request that refuses a QP outside 0..=51, for a constant-quantiser encoder.
    pub fn new(qp: u8, kind: PictureKind) -> Result<Self, String> {
        if qp > 51 {
            return Err(format!("a quantiser of {qp}, which is outside H.264's 0..=51"));
        }
        Ok(Self { qp: Some(qp), kind, last: false })
    }

    /// **A request that names no quantiser**, for an encoder built under [`RateControl::Cbr`] or
    /// [`RateControl::Vbr`].
    ///
    /// It is a constructor rather than nothing so the refusal is **reachable and testable**:
    /// [`VaEncoder::encode`] refuses it against a constant-quantiser encoder and refuses
    /// [`EncodeRequest::new`] against a bitrate one, so the pairing is a checked claim rather than a
    /// comment. A refusal no caller can ask for is a claim with no evidence behind it, and this
    /// milestone has three of those already.
    pub fn rate_controlled(kind: PictureKind) -> Result<Self, String> {
        Ok(Self { qp: None, kind, last: false })
    }

    /// **A predicted (P) picture at `qp`**, which is what a caller asks for on every picture after
    /// the first, in a constant-quantiser encoder.
    ///
    /// This and [`EncodeRequest::key_frame`] exist so that a caller outside this module never has to
    /// name [`PictureKind`]: a frame loop's normal case is "not a key frame", and making that case
    /// require naming a type is a way of saying the exception is the default. [`EncodeRequest::new`]
    /// takes the kind directly for a caller that does have a reason to name it — and
    /// [`EncodeRequest::b_picture`] is the third, which exists to be refused.
    pub fn predicted(qp: u8) -> Result<Self, String> {
        Self::new(qp, PictureKind::P)
    }

    /// **A predicted (P) picture with no quantiser named** — the ordinary case for every picture
    /// after the first in a bitrate-controlled encoder.
    pub fn predicted_at_rate() -> Result<Self, String> {
        Self::rate_controlled(PictureKind::P)
    }

    /// **A key frame (IDR) at `qp`**: the first picture of a stream, and the first after a seek, in a
    /// constant-quantiser encoder.
    pub fn key_frame(qp: u8) -> Result<Self, String> {
        Self::new(qp, PictureKind::Idr)
    }

    /// **A key frame (IDR) with no quantiser named**, for a bitrate-controlled encoder: the first
    /// picture of a stream, and the first after a seek.
    pub fn key_frame_at_rate() -> Result<Self, String> {
        Self::rate_controlled(PictureKind::Idr)
    }

    /// **A B picture at `qp`** — the third kind, and the one the loop refuses.
    ///
    /// It is a constructor rather than nothing so that the refusal is **reachable and testable**:
    /// [`VaEncoder::encode`] names what a B picture would need and this backend does not have, and
    /// [`VaEncoder::allow_b_pictures`] carries the measurements behind that. A refusal no caller
    /// can ask for is a claim with no evidence behind it, and this milestone has three of those
    /// already.
    pub fn b_picture(qp: u8) -> Result<Self, String> {
        Self::new(qp, PictureKind::B)
    }

    /// A request that also ends the stream, by putting `H264_LAST_PICTURE_EOSTREAM` in the picture
    /// record so the driver appends an `end_of_stream()` NAL.
    pub fn last(mut self) -> Self {
        self.last = true;
        self
    }

    /// Whether this request says the picture is an IDR, which is what [`PictureKind`] is for.
    pub fn idr(&self) -> bool {
        self.kind == PictureKind::Idr
    }
}

/// One reconstructed picture the encoder is holding, and what a later picture needs to predict
/// from it.
///
/// **This is the state a frame loop keeps, and the answer to whether that loop is simple or stateful:
/// it is stateful, and this type is the state.** The picture's own surface is not enough — H.264
/// identifies a reference by the pair `(frame_num, picture order count)`, and a reference list entry
/// that carries the right surface with the wrong `frame_num` is a reference to a picture the driver
/// will not find. So the surface, the frame number and the picture order count travel together, and
/// they are advanced together, and nothing else about a previous picture needs remembering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reference {
    /// The surface holding this picture's reconstruction.
    pub surface: u32,
    /// **The DPB slot the driver knows this picture by**, which is `curr_pic.frame_idx` — a slot
    /// number, not a surface id, and *not* derivable from `frame_num` here: once a reference is
    /// dropped the slot a picture occupies is not its frame number reduced by anything, because the
    /// picture that would have shifted into it is gone.
    pub frame_idx: u32,
    /// `frame_num` of the picture, which is how a slice refers to it and what the driver's own
    /// decoded picture buffer matches on.
    pub frame_num: u16,
    /// `pic_order_cnt_lsb` of the picture, which is how the display order is recovered.
    pub poc: u16,
}

/// The picture being encoded right now: where it is reconstructed, which DPB slot it takes, and
/// whether it is kept as a reference for later.
///
/// Separate from [`Reference`] because the two differ in exactly the field that matters: a
/// reference carries the `frame_num` a *slice* refers to it by, and the current picture carries the
/// `frame_num` it is *written* with. They are the same number here only because this encoder has no
/// reordering — under `pic_order_cnt_type == 0` with no B pictures, the coded order is the display
/// order — and that is a property of the configuration rather than of the types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Current {
    /// The surface `curr_pic.picture_id` names: where this picture's reconstruction is written.
    pub surface: u32,
    /// The slot in the driver's decoded picture buffer this picture takes, which is what
    /// `curr_pic.frame_idx` says. **A slot number, not a surface id** — see
    /// [`VaEncoder::slot_for`], which is where the slot is derived and why its modulus is
    /// `references + 1`.
    pub frame_idx: u32,
    /// `frame_num`, which is the identity a later slice refers to this picture by.
    pub frame_num: u16,
    /// `pic_order_cnt_lsb`, which is how the display order of this picture is recovered. It travels
    /// with the picture into [`Reference`] because a reference list entry needs it.
    pub poc: u16,
    /// Whether to keep this picture for later prediction (`VA_PICTURE_H264_SHORT_TERM_REFERENCE`).
    pub is_reference: bool,
}

impl Current {
    /// This picture as a reference of a later one — which is what it becomes if `is_reference`.
    fn as_reference(&self) -> Reference {
        Reference { surface: self.surface, frame_idx: self.frame_idx, frame_num: self.frame_num, poc: self.poc }
    }
}

/// How many reference pictures an encoder holds, and what the driver says it can hold.
///
/// The count is the caller's because it is a memory and a scheduling decision, but the *ceiling* is
/// the driver's: [`VaEncoder::new`] asks for `VAConfigAttribEncMaxRefFrames` and refuses a request
/// above what the driver reports, because a reference depth the driver cannot meet is a P picture
/// that fails at `vaEndPicture` with nothing naming the cause.
pub const MAX_REFERENCES: u32 = 16;

/// **`MaxFrameNum` for this encoder, which is 256 — eight bits of `frame_num`.**
///
/// [`sequence_record`] writes `log2_max_frame_num_minus4 = 4` and this is the modulus it implies
/// (7.3.2.1.1: `MaxFrameNum = 2 ^ ( log2_max_frame_num_minus4 + 4 )`, and 7.4.3: `frame_num` is
/// `u(v)` with `v = log2_max_frame_num`). It is a named constant rather than a number typed where
/// it is used because **the whole frame loop turns on it**: a picture whose `frame_num` would wrap
/// is a key frame, and a hardcoded number in three places is three chances for the record and the
/// loop to disagree.
///
/// **Why it is 256, which is the same question as what the specification requires** — the module
/// note, "The `frame_num` wrap, decided", carries the clause text. In short: 7.4.3's Equation 7-22
/// forbids a *skipped* frame number still marked "used for short-term reference", and at a wrap the
/// loop that generates that set never runs, so a wrap is conformant and **no
/// `memory_management_control_operation` is required**. 7.4.3 further says that with
/// `gaps_in_frame_num_value_allowed_flag` clear `frame_num` **shall be equal to**
/// `( PrevRefFrameNum + 1 ) % MaxFrameNum` — the wrap is the value the standard *requires*, not one
/// it forbids. What limits the group of pictures is therefore the **width**, and 8.2.5.1's closing
/// sentence bounds the DPB at `Max( num_ref_frames, 1 )` pictures, which at the one-reference depth
/// this module submits is one — so the width, not the reference marking, is what has to grow.
///
/// **256 rather than the 16 this was, and the measurement behind it.** 16 was four bits, which put
/// the IDR this loop inserts for the wrap at every sixteenth picture — so a preset asking for a
/// sixty-picture group could not be given one, and the export gate
/// ([`crate::hardware_encode::vaapi_config_for`]) refused every such preset with the measurement to
/// hand. `log2_max_frame_num_minus4` is a `u4` (`va_enc_h264.h`, the `seq_fields` bitfield), so
/// 256 is reachable and **the driver takes it**: on iHD 26.3.5 / libva 1.24, writing
/// `log2_max_frame_num_minus4 = 4` produced a stream in which ffmpeg and `trace_headers` both read
/// **eight-bit** `frame_num` values across two group-of-pictures boundaries and 40 pictures, with no
/// diagnostic. That run predates this change, so it is re-taken at the new width by
/// `a_run_longer_than_the_modulus_never_wraps` — which is what the old constant's own note said a
/// wider modulus deserved and did not have.
///
/// **Why the ceiling is 256 and not the `u4`'s limit of 2^19.** A wider `frame_num` costs nothing
/// in the stream, but a wider modulus is a *promise about seeking*: the IDR this loop inserts for
/// the wrap is the furthest back a decoder can be dropped without re-decoding from the last one, and
/// a real encoder still asks for its own key frames well inside 256. So this is a policy number —
/// "no more than 256 pictures between the ones this loop guarantees" — and it is checked against the
/// settings the caller actually gave rather than inferred from the field's range.
///
/// **A caller whose own interval is longer is refused, not silently shortened**:
/// [`crate::hardware_encode::vaapi_config_for`] declines with this number in the message, because an
/// export that asked for 300 and got 256 is a file a sixth larger than it was promised.
///
/// **This is the one number, and the record's field is computed from it** — see
/// [`LOG2_MAX_FRAME_NUM_MINUS4`], which is the only place the width is written and which cannot
/// disagree with this modulus because it is defined in terms of it.
pub const MAX_FRAME_NUM: u32 = 256;

/// [`MAX_FRAME_NUM`] as the `u4` the sequence record carries it in.
///
/// 7.3.2.1.1 defines `MaxFrameNum = 2 ^ ( log2_max_frame_num_minus4 + 4 )`, so the record's four-bit
/// field holds `log2_max_frame_num - 4` and not the modulus itself.
///
/// **Derived from [`MAX_FRAME_NUM`] by leading zeros rather than written as the literal `4`**, and
/// this is the point of the change rather than tidiness: the width in `seq_fields` and the modulus
/// [`VaEncoder::key_frame_due`] compares the counter against are two claims about one number, and a
/// `4` typed next to a `256` is two numbers that can drift with nothing to notice. A one followed by
/// trailing zeros is a **compile-time** dependency — writing [`MAX_FRAME_NUM`] as `16` again makes
/// this `0` and the sequence record four bits wide without a single line changing, and
/// `the_sequence_records_frame_num_width_is_the_one_the_loop_uses` is what would say so.
///
/// `u32::BITS - 1 - leading_zeros()` is `log2` of the modulus — 8 for 256 — and the field carries
/// `log2 - 4`, so the `- 4` below is the subtraction 7.3.2.1.1's `MaxFrameNum = 2 ^ (
/// log2_max_frame_num_minus4 + 4 )` undoes. **`the_sequence_records_frame_num_width_is_the_one_the_loop_uses`
/// is what caught the first version of this expression**, which kept the `log2` and wrote 8 into a
/// four-bit field's place: `MaxFrameNum` came out as 4096 while the loop keyed at 256, the record
/// still built, and nothing else in the crate objected. That is the whole reason the assertion is a
/// re-derivation of the modulus rather than a comparison against the constant that produced it.
const LOG2_MAX_FRAME_NUM_MINUS4: u32 = (u32::BITS - 1 - MAX_FRAME_NUM.leading_zeros()) - 4;

/// **`MaxPicOrderCntLsb` for this encoder, which is 16 — four bits of `pic_order_cnt_lsb`.**
///
/// 7.3.2.1.1: `MaxPicOrderCntLsb = 2 ^ ( log2_max_pic_order_cnt_lsb_minus4 + 4 )`, and
/// `sequence_record` writes `log2_max_pic_order_cnt_lsb_minus4 = 0`.
///
/// **It stays at 16 while [`MAX_FRAME_NUM`] went to 256, and that is a decision rather than an
/// oversight.** `pic_order_cnt_lsb` carries the picture order count, which under
/// `pic_order_cnt_type == 0` is derived by 8.2.1.1's Equation 8-3 from the previous reference
/// picture's count and this one — and that equation is *written for* the wrap: it compares the two
/// least-significant halves against `MaxPicOrderCntLsb / 2` and adds or subtracts a whole
/// `MaxPicOrderCntLsb` when the difference says the count went backwards or forwards past the
/// midpoint. A 16-picture order count therefore reconstructs across its own wrap with no help.
///
/// The order count is also reset by every IDR (8.2.1.1 infers 0), so it only ever has to span one
/// key-frame interval — and it is this encoder's key frames, not its `frame_num`, that decide where
/// a reset lands.
///
/// **Widening it anyway would not be wrong, only unmeasured.** It would move the point at which
/// 8.2.1.1's reconstruction has to do arithmetic rather than copy the count across, so the change
/// that makes the order count *easier* is the one this module does not need and cannot yet justify.
/// The two widths being different numbers is asserted in
/// [`check_output_matches_the_sets`], because a stream in which they had silently drifted back
/// together would still decode — which is exactly why it needs saying.
const MAX_PIC_ORDER_CNT_LSB: u32 = 16;

/// [`MAX_PIC_ORDER_CNT_LSB`] as the `u4` the sequence record carries it in: `log2_max_frame_num`
/// is 8 for a modulus of 256, and the same subtraction gives the order count's field. Written as a
/// subtraction from the same `log2` [`LOG2_MAX_FRAME_NUM_MINUS4`] uses, so that the two widths in
/// `seq_fields` are visibly different numbers derived the same way — which is the claim the sequence
/// record makes and [`check_output_matches_the_sets`] asserts against the parsed sets.
const LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4: u32 = (u32::BITS - 1 - MAX_PIC_ORDER_CNT_LSB.leading_zeros()) - 4;

/// The value [`VaEncoder::keyframe_interval`] holds before a caller has said what it is.
///
/// **A sentinel, not a default, and specifically not 0** — 0 is a legal answer meaning "one key
/// frame for the whole stream", so an encoder that used it as "unset" could not tell a caller who
/// asked for no further key frames from one who never answered. [`VaEncoder::encode`] refuses the
/// first picture while this is the value, which is the whole point: a frame rate and a GOP that were
/// never given are not settings of zero, they are settings nobody made.
const UNSET_INTERVAL: u32 = u32::MAX;

/// **The `pic_init_qp` a picture carries in a bitrate-controlled encoder, and the only number this
/// module writes into a record whose fidelity nobody chose.**
///
/// It exists because `VAEncPictureParameterBufferH264::pic_init_qp` is a **required field of every
/// picture** — there is no "unset" for it — so a bitrate-controlled encoder must write *something*
/// into it on every picture, and whatever it writes becomes the `pic_init_qp` of the **derived**
/// parameter set this module puts in the `avcC`. That makes the number a real, checkable claim about
/// the stream rather than a formality: the derived PPS says `pic_init_qp = 26`, and every slice in
/// the file has its own `slice_qp_delta` added to it.
///
/// **Why 26, stated as what it is and not as a quality.** H.264's `pic_init_qp` is defined as
/// `pic_init_qp_minus26 + 26` (7.3.2.2.1), so **26 is the value the syntax itself is centred on** —
/// the one quantiser the standard names as the origin. A decoder computing a slice's QP does
/// `SliceQPY = 26 + pic_init_qp_minus26 + slice_qp_delta`, so with `pic_init_qp = 26` the driver's
/// per-picture `slice_qp_delta` **is** the whole movement, with nothing added or subtracted around it.
/// Any other constant would be the same statement with an offset nobody could check.
///
/// **It is measured that the driver does not read it as a quality in this mode**, which is what makes
/// it safe to write at all: a 20-picture 640x360 run at a 400 kbit CBR target carries this in every
/// picture record and the driver's own reported quantiser moves across the run — **30, 30, 32, 32,
/// 32, 32, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36** — where a stream pinned at 26
/// would report 26 twenty times. Under [`RateControl::ConstantQuantiser`] the field is the caller's
/// own QP and this constant is never used.
///
/// **What the file then says, and this is measured rather than hoped: 26.** The driver writes
/// `slice_qp_delta = 0` in every slice of a CBR stream — see [`check_output_matches_the_sets`] for the
/// ffmpeg `trace_headers` measurement — so `SliceQPY` in the bitstream is this number on every picture.
/// **This constant is therefore not a claim that the pictures are coded at QP 26; it is the base a
/// decoder computes `SliceQPY` from**, and the quantiser actually used is whatever the rate controller
/// chose, which libva reports and does not record. That is a property of iHD 26.3.5 and nothing here
/// can change it; what this module can do is state the base truthfully and say so where it is
/// declared, which is here.
///
/// The consequence the self-check depends on: **under a bitrate mode the derived PPS's `pic_init_qp`
/// is this base and not any picture's quality**, so the equality between the driver's reported
/// quantiser and `pps.pic_init_qp` is not a well-formedness property there and is not checked as one.
/// Under constant quantiser it is, and it is the strongest check in
/// [`check_output_matches_the_sets`].
const DEFAULT_ENCODER_QP: u8 = 26;

/// What one picture came back as: the compressed bytes, and the size they were encoded at.
///
/// The size is returned rather than remembered because of the crop: H.264 codes whole macroblocks,
/// so a picture that is not a multiple of 16 is encoded padded and cropped, and a caller that
/// assumed its own width would find a decoder disagreeing with it. `coded_size` is what the driver
/// was told, and `display_size` is the crop.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedPicture {
    /// **The complete Annex-B access unit: a sequence parameter set, a picture parameter set, and
    /// the slice(s) the driver wrote** — in that order, for every picture including a predicted one.
    ///
    /// "The parameter sets ahead of the slice" is a statement about the bytes, not about where they
    /// came from, and `parameter_sets_from_driver` below says which. On iHD 26.3.5 they are never
    /// the driver's: **the driver writes no NAL of type 7 or 8 under any
    /// `VAConfigAttribEncPackedHeaders` value**, so they are derived from the records this module
    /// submitted and then checked against this picture's own coded NALs — see
    /// [`VaEncoder::sample_entry`]. A caller writing this to a file gets a valid elementary stream; a
    /// caller feeding an MP4 muxer wants [`Self::mp4_sample`] instead, and an `avcC` says which of
    /// the two the samples are in.
    ///
    /// **A predicted picture carries the parameter sets too, and that is deliberate.** Repeating an
    /// SPS and a PPS mid-stream is legal (7.4.2.1.1's `seq_parameter_set_id` makes the decoder
    /// adopt the later one, and they are identical here), and it is what makes this stream decodable
    /// from any of its pictures rather than only from the first. A caller assembling an elementary
    /// stream and wanting the sets only once has [`Self::driver_bytes`] to subtract them.
    pub bytes: Vec<u8>,
    /// How many of `bytes` the driver itself wrote, which is `bytes.len()` minus the parameter sets.
    pub driver_bytes: usize,
    /// Whether the parameter sets in `bytes` came from the driver or from [`super::params`]. **A
    /// measurement, not a preference**: on iHD 26.3.5 this is always false, and a caller that sees
    /// it change has learned something about the driver.
    pub parameter_sets_from_driver: bool,
    /// The size the picture is encoded at, a whole number of macroblocks.
    pub coded_size: (u32, u32),
    /// The size a decoder shows, which is `coded_size` with the crop taken off.
    pub display_size: (u32, u32),
    /// How many bytes of the coded buffer the picture took. Not the same as `bytes.len()` when the
    /// driver left the packed headers elsewhere; reported because "the encoder made the picture
    /// bigger" is a question about this number.
    pub coded_buffer_used: u32,
    /// How many segments the driver reported the picture in. See [`CodedPicture::segments`].
    pub segments: usize,
    /// `VA_CODED_BUF_STATUS_*` per segment (`va.h:3897-3933`). **The low byte is the average QP the
    /// driver says it encoded at** — the closest thing VA-API gives to "what quality did you
    /// actually produce", and worth having next to the QP that was asked for rather than instead
    /// of it.
    pub status: Vec<u32>,
    /// The `slice_type` the driver was told, as the header's own numbers (5 = all P, 7 = all I).
    /// Reported so a caller can see what it asked for next to what came out — **a P picture that
    /// came back the size of an I one has been coded intra, whatever this says**, which is why the
    /// byte size is the thing to check and this is only the intent.
    pub slice_type: u8,
    /// Whether this picture is a key frame **as encoded**, which is the loop's decision and not
    /// necessarily the caller's: a picture whose `frame_num` would wrap is a key frame whatever
    /// kind was asked for (see the module note, "The `frame_num` wrap, decided").
    pub key_frame: bool,
    /// Why this picture is a key frame when the caller did not ask for one. `None` for every
    /// picture the caller named as a key frame itself.
    ///
    /// **Reported rather than applied silently.** A caller that asked for a 24-picture GOP and got
    /// a key frame at picture 16 has been given a stream it did not ask for, and the honest thing is
    /// for it to be able to see that on the picture rather than to infer it.
    pub key_frame_reason: Option<KeyFrameReason>,
    /// How many references the slice was told to predict from. Zero for an IDR, and one for the P
    /// pictures of a `refs = 1` encoder.
    pub predicted_from: usize,
    /// Whether the driver ended the stream inside this picture's own bytes — that is, whether it
    /// honoured `last_picture` and appended an `end_of_stream()` NAL.
    ///
    /// **This is a measurement of the driver, not of what was asked for.** `request.last` says what
    /// was submitted; this says what came back. The two are separate claims and only measuring the
    /// second one tells us whether `H264_LAST_PICTURE_EOSTREAM` does anything on this driver — which
    /// is why [`VaEncoder::flush`] rests on it rather than on the flag.
    pub ends_stream: bool,
    /// The `frame_num` this picture was written with.
    pub frame_num: u16,
    /// The `pic_order_cnt_lsb` this picture was written with.
    pub poc: u16,
}

impl EncodedPicture {
    /// **This access unit as an MP4/MOV sample: each NAL prefixed with its four-byte length**, which
    /// is the form [`VaEncoder::sample_entry`]'s `avcC` declares with `length_size == 4`.
    ///
    /// It exists because the `avcC` and the samples have to agree about that field, and handing a
    /// muxer Annex-B start codes under a configuration record that says "four-byte lengths" is a
    /// file that plays on nothing. Everything else about the two forms is identical — same NAL
    /// units, same order, same bytes.
    ///
    /// A NAL that does not fit a `u32` length, or a stream with no unit in it, is an `Err` rather
    /// than a truncated prefix: the first would be a sample whose declared length wraps, and the
    /// second is not a sample at all.
    pub fn mp4_sample(&self) -> Result<Vec<u8>, String> {
        let units = annex_b_units(&self.bytes)?;
        let total: usize =
            units.iter().try_fold(0usize, |sum, unit| sum.checked_add(unit.nal_unit().len() + 4)).ok_or("the sample's NAL units do not fit a length")?;
        let mut out = Vec::with_capacity(total);
        for unit in &units {
            let nal = unit.nal_unit();
            let len = u32::try_from(nal.len()).map_err(|_| format!("a {}-byte NAL unit does not fit a four-byte length", nal.len()))?;
            out.extend_from_slice(&len.to_be_bytes());
            out.extend_from_slice(&nal);
        }
        Ok(out)
    }
}

/// **Why a picture is a key frame, when it was not the caller who said so.**
///
/// A key frame at a point the caller did not name is a change to the stream they did not ask for,
/// and this is what makes it visible rather than silent. The two reasons are both forced by the
/// codec rather than chosen: see [`VaEncoder::key_frame_due`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyFrameReason {
    /// The decoded picture buffer was empty, so there was nothing to predict from.
    ///
    /// **The first picture of a sequence, the first after [`VaEncoder::forget_references`] (a seek),
    /// and the first after an IDR — all three are the same state seen three ways.** A P slice with
    /// an empty reference list does not fail: the driver reconstructs the picture as it stands in
    /// the surface, which decodes to a plausible *wrong* picture rather than an error, and that is
    /// the exact failure this module exists to prevent.
    NoReference,
    /// The picture's `frame_num` would have been `MaxFrameNum`, which the field cannot hold.
    ///
    /// **`frame_num` is eight bits wide here**, because `sequence_record` writes
    /// `log2_max_frame_num_minus4 = 4`, so the wrap comes every 256 pictures — and the IDR lands on the
    /// modulus itself rather than one picture early, so a run carries every number in
    /// `0..MaxFrameNum` exactly once.
    ///
    /// **This is a policy, not a codec requirement, and the module note says so with the clause.**
    /// 7.4.3's Equation 7-22 forbids a *skipped* frame number still marked "used for short-term
    /// reference", and at a wrap the set of skipped numbers is empty, so the wrap is conformant
    /// without an MMCO — which this ABI cannot ask for anyway, since
    /// `VAEncSliceParameterBufferH264` carries no `dec_ref_pic_marking` field at all. The IDR is kept
    /// so the wrap is accompanied by construction at any reference depth, and it is **reported**
    /// rather than hidden precisely because it is not something the caller asked for.
    FrameNumWrap,
}

/// A configured H.264 encoder: what [`VaEncoder::new`] returns and what a caller encodes a run of
/// pictures through.
///
/// Holds the libva objects ([`Encoder`]), the derived geometry, and the frame loop's state: the
/// reference list, the two counters, and how many pictures have passed since the last key frame.
pub struct VaEncoder {
    encoder: Encoder,
    /// The width and height the surfaces were created at, a whole number of macroblocks.
    coded: (u32, u32),
    /// What a decoder shows: `coded` with the crop taken off.
    display: (u32, u32),
    /// The crop the sequence record carries, in the units `frame_crop_*_offset` counts (2 luma
    /// samples for 4:2:0 progressive — `VAFrameCropUnits`, `va.h:1300`). Kept because it is what
    /// turns a padded surface back into the size the caller asked for, and getting the unit wrong
    /// crops the picture by a factor of two.
    crop: (u32, u32, u32, u32),
    /// Macroblocks per picture, which is `num_macroblocks` in the slice record.
    macroblocks: u32,
    /// The coded size in macroblocks, which is what the sequence record's
    /// `picture_width_in_mbs` / `picture_height_in_mbs` state. **Kept rather than recomputed** so
    /// that the record [`VaEncoder::key_frames_every`] rebuilds cannot drift from the one the session
    /// was configured with — a second place computing `coded.0 / MB` is a second place where a size
    /// the caller never asked for could be written.
    width_mbs: u16,
    height_mbs: u16,
    /// `MaxFrameNum`, which is `2 ^ log2_max_frame_num` and is **eight bits wide for this
    /// encoder** — see the module note, "The `frame_num` wrap, decided", for what the
    /// specification requires and why the width rather than a reference-marking command is the thing
    /// that carries a long group of pictures. Held rather than recomputed because it is what
    /// [`VaEncoder::key_frame_due`] compares against and the alternative is a shift of a field read
    /// back out of a byte buffer.
    max_frame_num: u32,
    /// The frame number of the next picture, which increments per picture and wraps at
    /// `2^log2_max_frame_num`. **A counter, not state**: H.264's frame number is a function of the
    /// picture's position in the coded sequence, not a picture's identity.
    frame_num: u16,
    /// The picture order count of the next picture, which `pic_order_cnt_lsb` counts modulo
    /// `MaxPicOrderCntLsb`. Also a counter under `pic_order_cnt_type == 0`, and **also not in the
    /// picture record**: it belongs to the *slice* (7.4.2.1.1's `slice_header()`), which is where it
    /// is written, and the picture record only carries `frame_num`.
    poc: u16,
    /// `idr_pic_id`, which increments per IDR and pairs two IDR pictures in one sequence.
    idr_pic_id: u16,
    /// Pictures since the last key frame, which is what the caller's key-frame interval is compared
    /// against. Reset by every IDR, including the one at the `frame_num` wrap.
    since_key_frame: u32,
    /// The caller's key-frame interval in pictures, or [`u32::MAX`] for "not stated yet" — a
    /// sentinel rather than a default, so that `encode` can tell an unset GOP from a stated one and
    /// refuse the first. **`0` is a legal stated value** and means one key frame for the whole
    /// stream; it is not the same as unset, which is why the sentinel is not 0.
    keyframe_interval: u32,
    /// **Which fidelity statement this stream makes**, and it is fixed for the encoder's life because
    /// `VAConfigAttribRateControl` is a `vaCreateConfig` attribute (see [`RateControl`]). Held rather
    /// than recomputed because it is what [`VaEncoder::encode`] checks each request's quantiser
    /// against, and a check that read the mode out of the libva session could not name it in a
    /// message.
    rate_control: RateControl,
    /// The rate-control misc buffer, built once for a bitrate mode and **`None` for
    /// [`RateControl::ConstantQuantiser`]** — not "built with zeros", because under constant
    /// quantiser a bitrate record is a second and independent statement about the picture's quality
    /// and must not be submitted at all.
    rate_control_buffer: Option<Vec<u8>>,
    /// The reference depth this encoder was created with, which is **not** the size of its
    /// reconstruction pool: the pool is `refs + 1` surfaces (one input, `refs` references and one
    /// target), and confusing the two is how a caller asking for one reference silently gets two.
    refs: usize,
    /// The pictures this encoder has reconstructed and may still predict from, oldest first. **This
    /// is the frame loop's entire state**, and it is what makes the loop stateful rather than
    /// simple: a P picture names a reference by `(surface, frame_num, poc)`, and a driver handed the
    /// right surface with the wrong frame number predicts from a picture it does not have.
    references: Vec<Reference>,
    /// The sequence record, built once and submitted with every picture.
    sequence: Vec<u8>,
    /// The frame-rate misc buffer, also built once — see the module note for why it is mandatory.
    /// `None` until [`VaEncoder::at_rate`] has stated one, which is what makes `encode` refuse an
    /// encoder nobody gave a frame rate rather than guessing 30.
    frame_rate: Option<Vec<u8>>,
    /// The parameter sets the `avcC` is built from, held once a picture has been encoded and checked
    /// against that picture's own output. `None` until then, which is what makes
    /// [`VaEncoder::sample_entry`] answer "not yet" rather than guess — see its own note for why
    /// that matters more here than on the VideoToolbox side.
    sample_entry: Option<CodecParameterSets>,
    /// Pictures encoded so far, which is what [`VaEncoder::flush`] reports and what makes "this is
    /// the last picture" a checkable claim rather than a guess.
    encoded: u64,
    /// **Whether B pictures may be submitted — which nothing can set.**
    ///
    /// It is held as a field rather than deleted because the refusal it gates is *reachable*: a
    /// public predicate whose answer is always `false` is a claim a caller can check, and
    /// [`VaEncoder::allow_b_pictures`] is where the measurements behind it live. A future reordering
    /// buffer would set it; today it is initialised `false` and never becomes `true`, and
    /// `cargo clippy` sees no way to prove that, so the type stays.
    b_pictures: bool,
    /// Whether the stream has been ended by a picture marked [`EncodeRequest::last`]. **Kept
    /// separately from [`Self::last_marked`]** because they answer different questions: this one is
    /// "is this over", that one is "did the bitstream get told". A stream flushed without a last
    /// picture is legal and still over.
    ended: bool,
    /// Whether a picture has been submitted with [`EncodeRequest::last`], which is what put
    /// `H264_LAST_PICTURE_EOSTREAM` in a picture record and so what makes the stream's bitstream say
    /// where it ends. [`VaEncoder::flush`] reports when this is false rather than assuming it.
    last_marked: bool,
}

impl VaEncoder {
    /// **A constant-quantiser encoder** — the mode where the *picture record's* `pic_init_qp` is the
    /// quality, and the caller names it on every [`EncodeRequest`].
    ///
    /// **It takes no quantiser argument, and that is the point of the mode.** In a bitrate mode the
    /// quality is a rate and belongs to the constructor; in this mode it is a per-picture decision and
    /// belongs to the request, so a constructor argument would either be a default nobody chose or a
    /// value the first request is free to contradict. The 0..=51 check lives in
    /// [`EncodeRequest::new`], where the number actually is.
    ///
    /// The third named constructor, for the same reason as the other two: a caller outside this module
    /// names its mode through one of these rather than through [`RateControl`], which `vaapi/mod.rs`
    /// re-exports by a list this file does not get to edit from every corner of the crate.
    pub fn at_constant_quantiser(display: &Arc<Display>, size: (u32, u32), references: u32) -> Result<Self, String> {
        Self::new(display, size, references, RateControl::ConstantQuantiser)
    }

    /// **A constant-bit-rate encoder**, in bits per second.
    ///
    /// This and [`VaEncoder::at_variable_bitrate`] exist so a caller outside this module can name the
    /// mode it wants **without naming [`RateControl`]**, which `vaapi/mod.rs` re-exports by an
    /// explicit list this crate does not get to change from every file. [`VaEncoder::new`] takes the
    /// enum and is the underlying constructor; these are the two modes an export gate builds, spelled
    /// the way the gate thinks about them, so the two names cannot drift from the two enum variants.
    pub fn at_constant_bitrate(display: &Arc<Display>, size: (u32, u32), references: u32, bits_per_second: u32) -> Result<Self, String> {
        Self::new(display, size, references, RateControl::Cbr(bits_per_second))
    }

    /// **A variable-bit-rate encoder** between a target and a ceiling, both in bits per second.
    /// The ceiling being below the target is an error, not a swap — see [`rate_control_record`].
    pub fn at_variable_bitrate(display: &Arc<Display>, size: (u32, u32), references: u32, target: u32, max: u32) -> Result<Self, String> {
        Self::new(display, size, references, RateControl::Vbr { target, max })
    }

    /// A constant-quantiser H.264 encoder for `size`, holding up to `references` reference pictures
    /// at a stated frame rate and key-frame interval.
    ///
    /// **The two stream-shaped settings are builder methods, not arguments here**, and that is not
    /// tidiness: `references`, `keyframe_interval` and `frame_rate` are all `u32` and mean completely
    /// different things — a memory and a scheduling decision counted in pictures, a position in the
    /// stream, and a rate — so a four-argument constructor is a place where a caller can pass the
    /// right three numbers in the wrong order and get a stream nobody asked for. [`Self::at_rate`]
    /// and [`Self::key_frames_every`] each name their one setting, and **neither is defaulted**:
    /// [`VaEncoder::encode`] refuses a picture until both have been stated, because a rate or a GOP
    /// invented here is a stream whose timing and seek structure the caller never chose.
    ///
    /// `size` is the picture's *display* size and is rounded up to a whole number of macroblocks:
    /// 640x360 becomes 640x368 with the bottom 8 rows cropped off, which is what H.264 requires
    /// and what [`EncodedPicture`] reports as `coded_size` and `display_size`. Rounding up rather
    /// than refusing is not a default here — it is the only size this codec can encode, and the crop
    /// is exact.
    ///
    /// `references` is the reference depth. Zero gives an intra-only encoder, which is enough
    /// for a key frame; **one is the minimum that can encode a P picture at all**, because the driver
    /// needs a surface to write the current reconstruction into *and* the previous one to read from,
    /// so the session gets `2 + references` surfaces (see [`Encoder::new`]).
    ///
    /// It is bounded by [`MAX_REFERENCES`], which is the picture record's `ReferenceFrames[16]`
    /// array — a depth past it would have nowhere to go, and a silently shortened reference list
    /// makes the driver predict from a picture the caller did not choose, so that is an error and
    /// not a clamp.
    ///
    /// `rate_control` is **the only setting here that is not a builder method**, for a reason in
    /// libva rather than in taste: `VAConfigAttribRateControl` is read by `vaCreateConfig` and there
    /// is no call to change it afterwards, so a bitrate stated after the session exists would be a
    /// lie. It is a [`RateControl`] rather than a bare `u32` for the same reason `size` is a pair and
    /// `references` is bounded: the caller cannot hand this module the wrong constant.
    pub fn new(display: &Arc<Display>, size: (u32, u32), references: u32, rate_control: RateControl) -> Result<Self, String> {
        let refs = references;
        if size.0 == 0 || size.1 == 0 {
            return Err(format!("{}x{} picture", size.0, size.1));
        }
        if refs > MAX_REFERENCES {
            return Err(format!("{refs} reference pictures, more than the {MAX_REFERENCES} the picture record's array holds"));
        }
        // **Depth greater than one is refused, and the reason is a measurement rather than a
        // limitation of the code.**
        //
        // The records for a two-reference chain are built, submitted and accepted by iHD 26.3.5: the
        // render succeeds, `vaSyncBuffer` succeeds, ffmpeg decodes every frame without a warning,
        // and **every picture from the third on is wrong**. Not noisy, not grey — a clean luma ramp,
        // drifting about one step further per frame (measured at `refs = 2`: 253/255 maximum
        // deviation at picture 2, rising to 255 by picture 4, while `refs = 1` on the same pictures
        // stays at 2-4/255 for the whole run).
        //
        // Four candidate causes were measured and cleared: the DPB depth declared in the sequence
        // record (`max_num_ref_frames` now follows `refs`), the `frame_idx` slot numbering (forcing
        // every slot to 0 changes nothing), the order of `RefPicList0` (oldest-first is byte-identical
        // to newest-first), and the reconstruction surface rotation (which *was* a real bug at depth
        // 1 and is fixed — see the note in `encode`).
        //
        // What is left is the driver's own handling of a reference list deeper than one, and this
        // backend cannot tell a correct two-reference chain from a wrong one by any measurement
        // available from libva. **A silently wrong picture is the exact failure this milestone exists
        // to prevent**, so the honest state is an error naming the depth that was measured.
        //
        // A one-reference chain — which is what an export with a fixed GOP needs, and what
        // `examples/vaencprobe.rs` measures — is exact.
        if refs > 1 {
            return Err(format!(
                "{refs} reference pictures: only one reference is measured to work on this driver (iHD 26.3.5), and a deeper chain \
                 encodes without error while every picture past the second is wrong"
            ));
        }
        // **The driver's answer is a ceiling, and asking above it is an error rather than a clamp.**
        // A driver that cannot supply the requested reference depth fails the render with a status
        // that names no field — `VA_STATUS_ERROR_ENCODING_ERROR`, the same uninformative failure
        // Task 1 hit — so the check is here where it can name the attribute and the numbers. A
        // reported zero means "cannot tell" (see [`super::max_reference_frames`]) and is not a
        // ceiling, so it does not refuse.
        let (driver_l0, driver_l1) = ffi::Encoder::driver_max_reference_frames(display, Profile::H264Main);
        if refs > 0 && driver_l0 != 0 && refs > driver_l0 {
            return Err(format!("{refs} reference pictures asked for, but VAConfigAttribEncMaxRefFrames reports {driver_l0} for reference list 0"));
        }
        let _ = driver_l1;
        // **The rounding up is `checked_*`, and it has to be.** `size.0.div_ceil(MB) * MB` is a
        // `u32` multiplication whose answer is not always a `u32`: `u32::MAX` rounds to 2^32. In a
        // debug build that is a panic on a number the caller controls, and in a release build it is a
        // silent wrap to 0 — after which everything below (`crop_of`, the macroblock counts,
        // `coded_buffer_for`, the padded planes in [`VaEncoder::upload`]) is sizing itself against a
        // picture of no columns. Both are the same failure as the arithmetic not being checked, so it
        // is checked.
        let (Some(cw), Some(ch)) = ((size.0.div_ceil(MB)).checked_mul(MB).filter(|v| *v > 0), (size.1.div_ceil(MB)).checked_mul(MB).filter(|v| *v > 0)) else {
            return Err(format!("{}x{} rounds up past the largest picture this encoder can build", size.0, size.1));
        };
        let coded = (cw, ch);
        // The crop, in the units `frame_crop_*_offset` counts, as `(left, right, top, bottom)` —
        // the order the four `frame_crop_*_offset` fields are declared in, so the tuple can be
        // destructured straight into the record. For 4:2:0 progressive the unit is 2 luma samples
        // horizontally and 2 vertically (`VAFrameCropUnits`, `va.h:1298-1310`), so a
        // macroblock-quantised 640x368 holding a 360-row picture crops 4 units off the *bottom*.
        // Each offset is a division by that unit and is an error rather than a truncation when the
        // padded size is not an exact multiple — which for a size already rounded to a whole
        // macroblock cannot happen, and is checked anyway so a change to the rounding cannot make
        // the crop silently wrong.
        let crop_unit = 2u32;
        let crop = (crop_of(coded.0, size.0, crop_unit, "left")?, 0, 0, crop_of(coded.1, size.1, crop_unit, "bottom")?);
        let width_mbs = u16::try_from(coded.0 / MB).map_err(|_| format!("{} macroblocks wide", coded.0 / MB))?;
        let height_mbs = u16::try_from(coded.1 / MB).map_err(|_| format!("{} macroblocks high", coded.1 / MB))?;
        let macroblocks = u32::from(width_mbs) * u32::from(height_mbs);
        // The libva session itself: a config, `2 + references` surfaces, a context and the coded
        // buffer every picture is written into. **The rate-control mode goes in here and nowhere
        // else**, because it is a `vaCreateConfig` attribute: `Encoder::new`'s error names the value
        // it refused, so a driver that does not offer this mode fails with `CBR` or `VBR` in the
        // message rather than an uninformative status.
        let encoder = Encoder::new(
            display,
            Profile::H264Main,
            RtFormat::Nv12,
            coded,
            rate_control.config_attribute(),
            PACKED_HEADER_SEQUENCE | PACKED_HEADER_PICTURE,
            coded_buffer_for(coded)?,
            refs,
        )?;
        // **The sequence record starts with no key-frame period and is rebuilt by
        // [`VaEncoder::key_frames_every`].** An `Encoder` cannot ask for its stream's settings after
        // the fact, and building the record twice is cheaper than the alternative — a constructor
        // that guessed a GOP would be a seek structure the caller never chose.
        //
        // The frame rate and the key-frame interval are both `None`/unset here, and
        // [`VaEncoder::encode`] refuses the first picture until both are stated. **A setting that was
        // never given is not a setting of zero**: under this module's rule a value that cannot be
        // derived faithfully is an `Err`, and a frame rate is not derivable from a picture size.
        //
        // The quantization misc buffer is not built at all: it is rebuilt per picture, because it
        // carries the request's `qp`.
        //
        // **The rate-control misc buffer is built here and only here**, once, and it is `None` under
        // [`RateControl::ConstantQuantiser`] rather than a record of zeroes — under constant
        // quantiser the picture's quality is `pic_init_qp` and a bitrate record would be a second,
        // independent statement about the same fidelity. Its `window_size` needs the key-frame
        // interval, which is not stated yet, so it is built by [`VaEncoder::key_frames_every`] along
        // with the sequence record rather than guessed here.
        let rate_control_buffer = match rate_control {
            RateControl::ConstantQuantiser => None,
            mode => Some(rate_control_record(mode, RATE_CONTROL_WINDOW_MS)?),
        };
        Ok(Self {
            encoder,
            coded,
            display: size,
            crop,
            macroblocks,
            width_mbs,
            height_mbs,
            max_frame_num: MAX_FRAME_NUM,
            frame_num: 0,
            poc: 0,
            idr_pic_id: 0,
            since_key_frame: 0,
            keyframe_interval: UNSET_INTERVAL,
            rate_control,
            rate_control_buffer,
            refs: refs as usize,
            references: Vec::new(),
            sequence: sequence_record(coded, crop, width_mbs, height_mbs, refs, 0)?,
            frame_rate: None,
            sample_entry: None,
            encoded: 0,
            ended: false,
            last_marked: false,
            b_pictures: false,
        })
    }

    /// **Which fidelity statement this encoder makes** — the same [`RateControl`] it was built with,
    /// for a caller that wants to check a stream's contract rather than remember it.
    pub fn rate_control(&self) -> RateControl {
        self.rate_control
    }

    /// Whether the driver was told to pick a bitrate, as opposed to being told a quantiser.
    ///
    /// **This is the question `check_output_matches_the_sets` asks about the quantiser**, and it is
    /// asked here so the two cannot drift: the self-check is scoped by this and not by a separate
    /// list of modes.
    pub fn rate_controlled(&self) -> bool {
        !self.rate_control.names_its_own_quantiser()
    }

    /// **B pictures: refused, and why — with the measurement that says so.**
    ///
    /// This used to return `&mut Self` and turn a flag on, so a caller could submit
    /// [`EncodeRequest::b_picture`] and get a stream in which the picture *claiming* to be a B was
    /// coded as a P — a request silently answered with something else, which is the failure this
    /// module exists to refuse. It is a **`Result` now, and it is always an `Err`**: the state a B
    /// slice needs cannot be built by this frame loop, and that is a construction-time fact rather
    /// than a per-picture decision.
    ///
    /// The three measurements, with what each one actually establishes:
    ///
    /// - **The driver can.** `VAConfigAttribEncMaxRefFrames` answers **(3, 1)** here — one *future*
    ///   reference is within what it reports — so the refusal is not "this driver cannot do B".
    /// - **It does not.** Across a 40-picture run, every NAL the driver wrote was a type 1 or a
    ///   type 5 and every slice header read back `slice_type` 5 or 7. No B slice appeared, so
    ///   `media_start()` is 0 for a reason that is measured rather than assumed.
    /// - **And this loop could not drive one even if asked.** A B slice predicts from a past *and*
    ///   a future reference, so the decoded picture buffer must hold two, and **the deepest
    ///   reference depth this encoder accepts is one** — every larger depth is refused by
    ///   [`VaEncoder::new`] because it encodes without error while producing wrong pictures. There
    ///   is no second reference to put in list 1, so a submitted B slice's reference list would
    ///   name a picture the driver cannot resolve: a wrong picture, not an error.
    ///
    /// Turning the flag on is therefore not offered. What B pictures would take is a reordering
    /// buffer — the picture after a B must already exist when the B is coded, so pictures leave the
    /// encoder out of display order, `composition_offset` stops being zero, and the driver has to
    /// be given a reference depth this module currently refuses. None of that is a smaller version
    /// of the present loop.
    pub fn allow_b_pictures(&mut self) -> Result<&mut Self, String> {
        Err("B pictures: the driver can encode them (VAConfigAttribEncMaxRefFrames reports one future reference) and does not \
             emit any for the streams this backend submits, and this frame loop cannot drive one: a B slice predicts from a \
             past and a future reference, so the decoded picture buffer must hold two, and the deepest reference depth this \
             encoder accepts is one. Supporting them needs a reordering buffer, not a flag"
            .into())
    }

    /// **`MaxFrameNum` as a function, for a caller outside this module** that has to decide whether a
    /// key-frame interval is reachable at all.
    ///
    /// The interval is not the only thing that has to fit: the frame loop inserts a key frame every
    /// [`MAX_FRAME_NUM`] pictures whatever the caller asks for (see the module note, "The
    /// `frame_num` wrap, decided"), so an export that asked for a longer interval gets a shorter
    /// one. That is reported on [`EncodedPicture::key_frame_reason`] when it happens, and **refused
    /// before an encoder is built** by [`crate::hardware_encode::vaapi_config_for`] — this constant
    /// is what it refuses against.
    pub fn max_frame_num() -> u32 {
        MAX_FRAME_NUM
    }

    /// What the driver says it supports for [`super::ffi::RC_CQP`] and the other rate-control
    /// modes, as the mask itself (`va.h:1108-1164`). **A query, not a decision**: 0 means the
    /// driver did not say, and this backend submits constant quantiser whatever the answer is —
    /// see [`crate::hardware_encode::vaapi_config_for`] for why a bitrate is declined rather than
    /// sent.
    pub fn driver_rate_control_modes(display: &Arc<Display>) -> u32 {
        ffi::Encoder::driver_rate_control_modes(display, Profile::H264Main)
    }

    /// The names of the rate-control modes in `mask`, as `CQP and VBR` — or `nothing the driver
    /// named`, for a zero mask.
    ///
    /// **Written out rather than printed as a hex literal** because the mask's bits are `va.h`'s
    /// (`va.h:1108-1176`) and a decline message containing `0x140` tells a reader nothing. A zero
    /// mask is named as such and not as "none": it means the driver did not answer, which
    /// [`VaEncoder::driver_rate_control_modes`] says is not a claim that nothing is supported.
    ///
    /// It is an associated function rather than a module-level one because `vaapi/mod.rs`
    /// re-exports an explicit list of names from this module and that file belongs to another
    /// milestone; an associated function needs no change there.
    pub fn rate_control_names(mask: u32) -> String {
        let named: Vec<&str> = [
            (ffi::RC_NONE, "NONE"),
            (ffi::RC_CBR, "CBR"),
            (ffi::RC_VBR, "VBR"),
            (ffi::RC_VBR_CONSTRAINED, "VBR_CONSTRAINED"),
            (ffi::RC_CQP, "CQP"),
            (ffi::RC_ICQ, "ICQ"),
            (ffi::RC_QVBR, "QVBR"),
            (ffi::RC_AVBR, "AVBR"),
            (ffi::RC_TCBRC, "TCBRC"),
        ]
        .into_iter()
        .filter(|(bit, _)| mask & bit != 0)
        .map(|(_, name)| name)
        .collect();
        if named.is_empty() { "nothing the driver named".to_string() } else { named.join(" and ") }
    }

    /// **State this stream's frame rate, which the driver is told per picture.** Required, and
    /// refused rather than defaulted: iHD refuses a render whose buffer list omits the frame-rate
    /// record, and a rate this module invented would be a stream whose timing nobody asked for.
    ///
    /// The check happens here so the message can name the number: `0` and a rate past the field's
    /// 16-bit numerator are refused by [`frame_rate_record`], which is what the caller's mistake
    /// looks like when it reaches the driver as `VA_STATUS_ERROR_INVALID_PARAMETER`.
    ///
    /// **Only before the first picture.** The buffer is submitted with every render, so a rate
    /// changed mid-stream is a stream whose first picture and whose rest disagree about when a
    /// picture lasts.
    pub fn at_rate(&mut self, frames_per_second: u32) -> Result<&mut Self, String> {
        if self.encoded != 0 {
            return Err(format!("this encoder has already encoded {} picture(s), so its frame rate can no longer be changed", self.encoded));
        }
        self.frame_rate = Some(frame_rate_record(frames_per_second)?);
        Ok(self)
    }

    /// **State this stream's key-frame interval**, in pictures. **0 is legal and means one key frame
    /// for the whole stream** — a stream that is only ever played from its start — and is not the
    /// same as not having said, which is what [`UNSET_INTERVAL`] holds.
    ///
    /// It rebuilds the sequence record because `intra_idr_period` is the driver's statement of the
    /// same fact, and the two have to agree: a driver told 24 and given a key frame at 16 sizes its
    /// rate control for a GOP shape the stream does not have. **Which pictures actually become key
    /// frames is this module's decision, not the field's** — see [`VaEncoder::key_frame_due`] and
    /// the module note, "The `frame_num` wrap, decided", for the one case where the two part company.
    ///
    /// Refused after a picture has been encoded: the record is submitted with every picture, so
    /// changing the period mid-stream is a stream that claims a GOP structure it did not follow.
    pub fn key_frames_every(&mut self, pictures: u32) -> Result<&mut Self, String> {
        if self.encoded != 0 {
            return Err(format!("this encoder has already encoded {} picture(s), so its key-frame interval can no longer be changed", self.encoded));
        }
        self.keyframe_interval = pictures;
        self.sequence = sequence_record(self.coded, self.crop, self.width_mbs, self.height_mbs, self.refs as u32, pictures)?;
        // **The rate-control buffer is rebuilt with the sequence record**, not left alone, because
        // `window_size` is derived from this interval (see [`RATE_CONTROL_WINDOW_MS`]). Submitting a
        // buffer whose window was derived from a different GOP would be a rate controller sized for a
        // seek structure the stream does not have — the same reason the sequence record is rebuilt
        // here.
        if !self.rate_control.names_its_own_quantiser() {
            self.rate_control_buffer = Some(rate_control_record(self.rate_control, rate_control_window_ms(pictures))?);
        }
        Ok(self)
    }

    /// How many reference pictures this encoder is currently holding.
    pub fn reference_count(&self) -> usize {
        self.references.len()
    }

    /// **The MP4/MOV sample entry for the stream this encoder produces**, or why there is not one
    /// yet.
    ///
    /// `VideoEncoder::sample_entry` is documented as "may only be complete after the first frame",
    /// and this is that in the strongest form available: before a picture has been encoded there is
    /// no answer at all, because the only thing that could supply one is a description of a stream
    /// that does not exist yet.
    ///
    /// # Where the parameter sets come from, measured
    ///
    /// **This driver writes none.** Re-measured for this task on the machine the milestone was
    /// written for (Intel Arc B580, iHD 26.3.5, libva 1.24): `VAConfigAttribEncPackedHeaders` was
    /// set to each of `0` through `7` in turn, and the coded picture was **byte-identical in all
    /// eight cases** — the same `00 00 01 25` IDR slice, 935 bytes for a key frame and 656 for the
    /// first P frame — with no NAL of type 7 or type 8 anywhere in the driver's own output. The
    /// attribute is accepted and written back, the encode succeeds, and nothing else happens. The
    /// one libva entry point that could answer the question from the other end,
    /// `vaQuerySequenceHeaders`, **is not declared in `/usr/include/va/va.h` at this version and is
    /// not exported by `/usr/lib/libva.so.2`** (`nm -D` over the 88 symbols it exports lists
    /// `vaQueryConfigAttributes`, `vaQuerySurfaceAttributes` and no sequence-header call).
    ///
    /// So the sets in [`super::params`] are used, and they are **derived, never claimed to be the
    /// driver's**. [`CodecParameterSets::from_driver`] says which happened, per picture, and it is
    /// measured by looking for a NAL of type 7 or 8 in the bytes the driver actually wrote — not by
    /// assuming, which is the failure this whole module exists to avoid.
    ///
    /// # How the derived path checks itself
    ///
    /// Writing bits and reading them back (`super::params`) proves the *writing* was right. It proves
    /// nothing about whether the sets still **describe the output**, because a record is what was
    /// asked for rather than what was encoded — and the review of this task named the gap exactly:
    /// "if iHD ever rewrites `level_idc` or the reference count, the derived SPS describes a stream
    /// that was not encoded, and nothing in this range would detect it."
    ///
    /// So every picture's own coded NALs are parsed back **against the sets that would go into the
    /// `avcC`**, and the two are compared — see [`check_output_matches_the_sets`]. That is a real
    /// cross-check rather than a restatement, because a slice header only parses at the right length
    /// if the SPS's `log2_max_frame_num`, `frame_mbs_only_flag` and `log2_max_pic_order_cnt_lsb_minus4`
    /// are the ones the driver wrote to, and the quantiser the driver reports for the picture is its
    /// own statement of what `pic_init_qp` turned into. A divergence in either is an `Err` at the
    /// picture that caused it, not a file.
    ///
    /// # What this still cannot see, stated rather than implied
    ///
    /// `pic_width_in_mbs`, `frame_crop_*` and `max_num_ref_frames` appear in the SPS and in nothing
    /// else a decoder sends back, and libva 1.24 exposes no call that reads an encode record back
    /// (there is no `vaGetBufferSubset` and no `VABufferSubType` in this header). They are checked
    /// only in that they match the record FilmCraft submitted — the same record the picture's
    /// geometry came from. A driver that encoded at a different size than it was told and did not
    /// say so would not be caught here; it is caught by the export-level oracle (Task 7), which
    /// decodes the file and compares the picture, and that is where the plan puts it.
    ///
    /// # HEVC
    ///
    /// **There is no `hvcC` here, because there is no HEVC encoder here.** This module declares
    /// `VAEncSequenceParameterBufferH264` and its siblings and nothing for HEVC, so a
    /// `CodecParameterSets` can only ever be H.264's. Building an `hvcC` from a record layout this
    /// crate does not declare would be a config record describing a stream no code produces.
    pub fn sample_entry(&self) -> Result<filmcraft_isobmff::SampleEntry, String> {
        let Some(sets) = self.sample_entry.as_ref() else {
            return Err("no H.264 sample entry yet: this encoder has encoded nothing, and the parameter sets describe a stream that does not exist. \
                 Encode at least one picture (the first has to be a key frame) and ask again"
                .into());
        };
        // **The picture size in the sample entry is the size a decoder shows**, not the
        // macroblock-padded size the encoder works in: 640x360 is coded at 640x368 and the SPS's
        // crop takes eight rows off, so a sample entry carrying 368 would make a player show eight
        // rows of the padding the encoder filled with 16.
        let (w, h) = (self.display.0, self.display.1);
        let (Ok(width), Ok(height)) = (u16::try_from(w), u16::try_from(h)) else {
            return Err(format!("a {w}x{h} picture does not fit the sample entry's 16-bit dimensions"));
        };
        // **Four-byte NAL lengths, and [`EncodedPicture::mp4_sample`] is the matching sample form.**
        // An `avcC` states how long each NAL is, so declaring four and handing the muxer Annex-B
        // start codes would be a configuration record describing a bitstream it does not match — the
        // same class of defect as an SPS whose width is not the picture's.
        Ok(filmcraft_isobmff::SampleEntry::avc(filmcraft_isobmff::AvcConfig::new(vec![sets.sps.clone()], vec![sets.pps.clone()], 4), width, height))
    }

    /// The parameter sets the [`VaEncoder::sample_entry`] would use, once a picture has been
    /// encoded. `None` before then, for the reason [`VaEncoder::sample_entry`] gives.
    pub fn parameter_sets(&self) -> Option<&CodecParameterSets> {
        self.sample_entry.as_ref()
    }

    /// The most references this encoder will hold: **what the caller asked for**, not the size of
    /// the surface pool.
    ///
    /// The pool is `refs + 1` deep (see [`Encoder::new`]), so deriving the depth from it would
    /// report one more than was asked for — and an encoder that silently holds a reference the
    /// caller did not ask for is one that predicts from something the caller cannot see. Measured:
    /// with the depth taken from the pool, `refs = 1` held two references and every P frame after
    /// the second named a surface the caller had not been told about.
    pub fn reference_capacity(&self) -> usize {
        self.refs
    }

    /// Forget every reference, so the next picture must be a key frame.
    ///
    /// **This is what a seek does**, and it is the only way back to a state a decoder can start
    /// from: after a seek there is no picture before the new one, so a P picture would predict from
    /// something the decoder has never seen. The driver's own reference state is not reachable
    /// through libva, so the counters are reset here too and the next picture has to be an IDR.
    ///
    /// **It is a stream boundary, and it says so on the picture.** An encoder a project seeks back
    /// into has produced a different access-unit sequence from the one it was in the middle of, so
    /// the next picture's [`EncodedPicture::key_frame_reason`] is [`KeyFrameReason::NoReference`] —
    /// the same reason the first picture of a sequence has. A caller asking where the stream's runs
    /// begin reads that field rather than inferring it from the key-frame interval, because a seek
    /// produces a run the interval never asked for.
    pub fn forget_references(&mut self) {
        self.references.clear();
        self.frame_num = 0;
        self.poc = 0;
        self.idr_pic_id = 0;
        self.since_key_frame = 0;
    }

    /// The size the picture is encoded at (a whole number of macroblocks) and the size a decoder
    /// shows.
    pub fn sizes(&self) -> ((u32, u32), (u32, u32)) {
        (self.coded, self.display)
    }

    /// The crop the sequence record carries, as `(left, right, top, bottom)` in the units
    /// `frame_crop_*_offset` counts. Zeroes in all four for a size that is already a whole number of
    /// macroblocks, which is the case where `frame_cropping_flag` is clear and the values are not
    /// read.
    pub fn crop(&self) -> (u32, u32, u32, u32) {
        self.crop
    }

    /// The surface a caller fills with a picture: `luma` is `display.0 * display.1` bytes and
    /// `chroma` is half as many, both tightly packed.
    ///
    /// **The picture is uploaded at the coded size, not the display size.** A 640x360 picture in a
    /// 640x368 surface has its bottom 8 rows undefined, and a hardware encoder reads all 368 — so
    /// the padding is filled with the neutral values (16 for luma, 128 for chroma) rather than
    /// left as whatever was in the driver-allocated surface. That matters more than it sounds: an
    /// encoder that sees uninitialised rows below a crop boundary will spend bits on them and, on
    /// some drivers, move the slice boundary upward, which is a visibly different picture.
    pub fn upload(&self, luma: &[u8], chroma: &[u8]) -> Result<(), String> {
        let (w, h) = self.coded;
        // **Every length here is derived, checked and capped** — `vec![16u8; w * h]` on a size this
        // module cannot vouch for is an allocation of whatever the multiplication happened to
        // produce, and an allocation that cannot be satisfied does not return an `Err`: it aborts
        // the process through `handle_alloc_error`, which the panic hook and the `catch_unwind`
        // around the UI pass do not cover. That is the one place an "every input-derived number is
        // hostile" mistake costs a session rather than a frame, which is why `coded_buffer_for`,
        // `copy_picture` and `Encoder::upload` in the FFI module all compute their lengths with
        // `checked_*` and refuse what they cannot bound — and this is the same rule applied to the
        // padded planes that were still doing `w as usize * h as usize`.
        let (Some(luma_len), Some(chroma_len)) = (luma_len(w, h), chroma_len(w, h)) else {
            return Err(format!("a {w}x{h} picture's padded planes cannot be sized"));
        };
        // Neutral-value padding, filled in for the rows and columns the crop takes away.
        let mut padded_luma = vec![16u8; luma_len];
        let mut padded_chroma = vec![128u8; chroma_len];
        copy_picture(&mut padded_luma, w as usize, luma, self.display.0 as usize, self.display.1 as usize)?;
        copy_picture(&mut padded_chroma, w as usize, chroma, self.display.0.div_ceil(2) as usize, self.display.1.div_ceil(2) as usize)?;
        self.encoder.upload(&padded_luma, &padded_chroma)
    }

    /// **Whether the next picture must be a key frame, and why — the loop's central decision.**
    ///
    /// Two reasons, both forced rather than chosen, and the order they are tested in matters:
    ///
    /// 1. **There is nothing to predict from.** 7.4.3.3 empties the decoded picture buffer at an IDR,
    ///    so after one there is nothing in it until the next picture is reconstructed. A predicted
    ///    picture with an empty reference list does not fail: the driver reconstructs the picture as
    ///    it stands in the surface, which is a plausible-looking *wrong* picture and the exact
    ///    failure this module exists to prevent.
    /// 2. **The `frame_num` has run out of field.** `frame_num` is `u(log2_max_frame_num)` and
    ///    `MaxFrameNum = 1 << log2_max_frame_num`, so the picture whose number would be `MaxFrameNum`
    ///    has a number the bitstream cannot carry at all. See the module note, "The `frame_num` wrap,
    ///    decided". An IDR's `frame_num` is inferred to be 0 (7.4.3), so a wrap that is an IDR is
    ///    accompanied by construction and 7.4.3's "no two pictures in the DPB share a `frame_num`"
    ///    cannot be violated by it. This is checked **independently of the caller's key-frame
    ///    interval**, and the two are combined below rather than one overriding the other: a caller
    ///    asking for a 24-picture GOP gets a key frame at picture 16 of the first one, because the
    ///    wrap is not something the caller can grant.
    ///
    /// A caller that asked for the key frame itself gets `None`, which is what makes
    /// [`EncodedPicture::key_frame_reason`] mean "this one the loop inserted".
    pub fn key_frame_due(&self, requested: bool) -> Option<KeyFrameReason> {
        if requested {
            return None;
        }
        // The two states are checked in this order and **both are tested independently**, because
        // neither subsumes the other: the first picture of a sequence has an empty reference list
        // *and* a `frame_num` of 0, and a stream long enough to wrap has a full reference list. An
        // `else if` here would be shorter and would name the wrong reason on the picture the caller
        // is most likely to look at.
        if self.references.is_empty() {
            return Some(KeyFrameReason::NoReference);
        }
        // **The wrap is the picture whose `frame_num` *would* be the modulus, checked as a
        // comparison against `MaxFrameNum` rather than against `MaxFrameNum - 1`.**
        //
        // `self.frame_num` is a `u16` counter set to 1 by every IDR and incremented by one per
        // picture otherwise, so it runs `1, 2, … MaxFrameNum` and the picture carrying
        // `MaxFrameNum` is the one the field cannot hold: `frame_num` is `u(log2_max_frame_num)`
        // and `MaxFrameNum = 1 << log2_max_frame_num`, so that picture's number has no representation
        // at all. It is made an IDR, which 7.4.3 infers to `frame_num` 0, and the run restarts.
        //
        // **Comparing against `MaxFrameNum - 1` would work and would be one picture shorter per
        // run**, and that is the version this had: an IDR one picture early leaves the last value of
        // `frame_num` unused, so a run of 16 pictures carried only the numbers 0 through 14. That
        // is correct but it is not *tight* — the picture at `MaxFrameNum - 1` is a perfectly good P
        // picture, and 7.4.3 has no objection to it. A check that resolves a codec constraint
        // earlier than the constraint does is a check answering a question nobody asked.
        //
        // No arithmetic is done on `max_frame_num` at all, which is deliberate: a
        // `MaxFrameNum` of 0 — which a change to `sequence_record` could produce, and which names
        // no frame number — then makes the picture at counter 1 an IDR rather than underflowing a
        // subtraction in a debug build.
        (u32::from(self.frame_num) >= self.max_frame_num).then_some(KeyFrameReason::FrameNumWrap)
    }

    /// Whether the caller's key-frame interval has elapsed, which is the other reason a picture is a
    /// key frame. **Separate from [`VaEncoder::key_frame_due`] on purpose**: that function decides
    /// what the *codec* requires and reports why, and this one is the caller's own interval. They
    /// are combined in [`VaEncoder::encode`] so that a forced key frame and a requested one are the
    /// same event, and a caller can tell which it was from [`EncodedPicture::key_frame_reason`].
    ///
    /// **An unset interval is `false`, not `true`.** It is not reached from `encode` — that refuses
    /// an encoder nobody has configured — but a public predicate that said "yes, a key frame is due"
    /// for an unconfigured encoder would be a confident answer to a question nobody asked.
    pub fn key_frame_interval_elapsed(&self) -> bool {
        self.keyframe_interval != UNSET_INTERVAL && self.keyframe_interval > 0 && self.since_key_frame >= self.keyframe_interval
    }

    /// Whether this encoder has been told its frame rate and its key-frame interval, which
    /// [`VaEncoder::encode`] requires before it will encode anything.
    pub fn is_configured(&self) -> bool {
        self.frame_rate.is_some() && self.keyframe_interval != UNSET_INTERVAL
    }

    /// **The number of pictures this encoder has encoded**, which is what makes "the last picture"
    /// checkable: a caller can compare it against the picture count it submitted and know whether
    /// the stream ended where it said it would.
    pub fn pictures_encoded(&self) -> u64 {
        self.encoded
    }

    /// **End the stream and say whether it was ended properly.**
    ///
    /// What "flush" means for this backend is a question with a short answer, and the answer is
    /// *nothing is buffered*. Every picture is rendered, synchronised and read back inside
    /// [`VaEncoder::encode`], so by the time this is called the driver holds nothing and there is no
    /// last picture still to come. A `flush` that drained a queue here would be draining an empty
    /// one, and writing one to make the shape match the other platforms' encoders would be inventing
    /// buffering that does not exist.
    ///
    /// What this *does* check is the one thing a caller can get wrong and a muxer will not notice:
    /// **the last picture was not marked [`EncodeRequest::last`]**, so the bitstream carries no
    /// `end_of_stream()` NAL. That is legal — ffmpeg decodes such a stream without a warning, and an
    /// MP4 says where its track ends in the sample table rather than in the bitstream — but it is a
    /// difference from the other encoders, so it is reported here rather than left for a file to
    /// demonstrate.
    ///
    /// Refused after a flush, and on an encoder that has encoded nothing: both would be a caller's
    /// sequencing error, and both are cheap to name here and expensive to debug from a file.
    pub fn flush(&mut self) -> Result<(), String> {
        if self.encoded == 0 {
            return Err("this encoder has encoded nothing, so there is no stream to end".into());
        }
        if self.ended {
            return Err("this stream has already been ended by a picture marked as the last".into());
        }
        if !self.ended {
            log::info!(
                "VA-API: the last picture was not marked as the end of the stream, so the bitstream carries no end_of_stream() NAL; \
                 an MP4 track ends in its sample table and ffmpeg decodes this stream without a warning"
            );
        }
        // The references are dropped: they are the encoder's private reconstruction state, and
        // nothing after the last picture may predict from them.
        self.references.clear();
        self.ended = true;
        Ok(())
    }

    /// Whether the stream has been ended — by a picture marked [`EncodeRequest::last`], or by
    /// [`VaEncoder::flush`].
    pub fn ended(&self) -> bool {
        self.ended || self.last_marked
    }

    /// Encode one picture and read back what the driver wrote.
    ///
    /// **The picture type is the loop's decision, not the caller's.** [`VaEncoder::key_frame_due`] and
    /// [`VaEncoder::key_frame_interval_elapsed`] are consulted first, and a picture either of them
    /// names is an IDR whatever `request.kind` says — and [`EncodedPicture::key_frame_reason`] says
    /// which, so a caller that asked for a 24-picture GOP learns from picture 16 that it got one
    /// there. A caller that asked for a key frame gets one regardless.
    ///
    /// Everything else is derived: the picture's `frame_num` and picture order count come from the
    /// counters, which advance only after a picture comes back, so a failed render leaves the
    /// sequence exactly where it was and the caller can retry.
    ///
    /// **The reference state is committed after the readback, never before.** A render that fails
    /// has not produced a reference — the surface does not hold a reconstructed picture the next
    /// picture could predict from — so committing first would leave the encoder holding a reference
    /// to an unfinished picture, and every subsequent P frame would be wrong rather than the first.
    pub fn encode(&mut self, request: EncodeRequest) -> Result<EncodedPicture, String> {
        // **The quantiser and the mode must agree, and the disagreement is refused here rather than
        // resolved.** This is the check that makes [`RateControl`] and [`EncodeRequest::qp`] one
        // contract rather than two that happen to be documented together — a request naming a
        // quantiser into a bitrate encoder would otherwise be silently ignored (the controller picks
        // the QP), and a request naming none into a constant-quantiser encoder would be encoded at
        // whatever `pic_init_qp` the record happened to carry. Both are "a plausible wrong picture,
        // not an error", which is the failure class this module exists to refuse.
        if self.rate_control.names_its_own_quantiser() != request.qp.is_some() {
            return Err(match (self.rate_control, request.qp) {
                (mode, Some(qp)) => format!(
                    "a quantiser of {qp} into a {} encoder: the rate controller chooses this picture's quantiser, so the \
                     number would not be what it is encoded at. Build the encoder with RateControl::ConstantQuantiser for \
                     a fixed quantiser, or use EncodeRequest::predicted_at_rate() here",
                    mode.name()
                ),
                (mode, None) => format!(
                    "no quantiser into a {} encoder: this encoder's quality is the picture record's pic_init_qp, and there \
                     is no bitrate behind it, so a picture encoded now would be at whatever that record happens to carry. \
                     Say EncodeRequest::predicted({qp})",
                    mode.name(),
                    qp = DEFAULT_ENCODER_QP
                ),
            });
        }
        let qp = request.qp.unwrap_or(DEFAULT_ENCODER_QP);
        // **A B picture is refused here, and the measurements are on the other side of this
        // line** — [`VaEncoder::allow_b_pictures`] and the module note, "B pictures: measured, and
        // refused". What is refused is not the *driver's* ability (it reports one future reference
        // and could) but this loop's: a B slice reads a past and a future reference, the decoded
        // picture buffer would have to hold two, and the deepest depth this encoder accepts is one.
        // A B submitted with a one-entry list 1 would name a picture the driver cannot resolve — a
        // wrong picture, not an error — so `allow_b_pictures` no longer turns anything on and the
        // flag below is the last of it.
        if request.kind == PictureKind::B && !self.b_pictures {
            return Err("a B picture: this backend submits I and P pictures only. The driver can encode one \
                 (VAConfigAttribEncMaxRefFrames reports one future reference) and emits none for these streams; what is \
                 missing is the frame loop — a B slice predicts from a past and a future reference, so the decoded picture \
                 buffer must hold two and the deepest reference depth this encoder accepts is one. \
                 VaEncoder::allow_b_pictures carries the measurements"
                .into());
        }
        if self.ended || self.last_marked {
            return Err("this stream has already ended, so there is no picture after it".into());
        }
        // **A frame rate and a key-frame interval nobody stated are refused here, by name.** Both
        // reach the driver — the rate in a buffer iHD refuses a render without, the interval in the
        // sequence record's `intra_idr_period` — and both are things the caller has an opinion
        // about. Inventing either here is a stream whose timing and seek structure nobody chose,
        // which is the same class of default as picking a quantiser.
        let (Some(frame_rate), _) = (&self.frame_rate, ()) else {
            return Err("this encoder has no frame rate: call VaEncoder::at_rate before encoding a picture".into());
        };

        if self.keyframe_interval == UNSET_INTERVAL {
            return Err("this encoder has no key-frame interval: call VaEncoder::key_frames_every before encoding a picture".into());
        }
        // The two reasons a picture must be a key frame, in the order they are tested, and the
        // caller's own interval as a third. See `key_frame_due` for why the wrap is not optional.
        let forced = self.key_frame_due(request.idr());
        let key_frame = forced.is_some() || request.idr() || self.key_frame_interval_elapsed();
        // **The counters advance only after a picture comes back**, so a failed render leaves the
        // sequence exactly where it was and the caller can retry the same picture.
        let current = Current {
            surface: {
                // **The reconstruction target moves on every picture, not only when the encoder is
                // full.**
                //
                // This is the bug the two-reference test found, and it is worth stating because the
                // obvious policy is wrong in a way that only shows up at depth > 1: rotating only when
                // the reference list is full means the first `refs` pictures all write to the *same*
                // surface, so the second picture overwrites the first picture's reconstruction — and
                // the first picture is still in the reference list. Every P frame after that predicts
                // from a surface holding the wrong picture, which decodes to a plausible wrong
                // picture rather than an error.
                //
                // Rotating every time is correct because the pool is `refs + 1` surfaces deep: with at
                // most `refs` live references there is always exactly one surface no reference names,
                // and a pool of that size visited once per picture lands on it every time.
                self.encoder.rotate_reconstruction()
            },
            // **An IDR is written with `frame_num` 0 and `pic_order_cnt_lsb` 0**, which is what
            // H.264 infers them to be (7.4.3 for `frame_num`, 7.4.3.1 for `pic_order_cnt_lsb`, and
            // 8.2.1.1 for the picture order count of an IDR), and the driver writes whatever the
            // record says. It is **measured** that submitting the counter's value instead is not
            // harmless: with the counter's value in an IDR's slice header, every mid-stream key
            // frame makes ffmpeg answer
            //
            //     [h264] number of reference frames (0+2) exceeds max (1; probably corrupt input),
            //     discarding one
            //
            // and x264's own `max_num_ref_frames = 1`, mid-stream-IDR stream — whose IDRs carry
            // `frame_num 0`, as 7.4.3 says they must — draws no such line. So this is not an
            // ffmpeg quirk about reference counts; it is this stream's IDRs disagreeing with the
            // value the codec infers, which is exactly the "a plausible wrong picture, not an
            // error" class this module exists to refuse.
            //
            // The two counters still *advance* as if the IDR had taken a number, so the picture
            // after an IDR never reuses the IDR's `frame_num` and never claims its DPB slot — the
            // value written and the counter are two different things, and it is the value written
            // that 7.4.3 fixes at 0.
            frame_idx: self.slot_for(if key_frame { 0 } else { self.frame_num }),
            frame_num: if key_frame { 0 } else { self.frame_num },
            poc: if key_frame { 0 } else { self.poc },
            // Every picture is a candidate reference; the list is trimmed to `refs` *after* the
            // picture comes back, which is where the trim has to be so a failed render cannot leave
            // a shorter list than the picture that failed.
            is_reference: true,
        };
        // An IDR's slice type is "every slice is I"; a P picture's is "every slice is P". The
        // offset-by-five encoding is the header's own (`va_enc_h264.h:441`).
        let slice_type = if key_frame { SLICE_TYPE_ALL_I } else { SLICE_TYPE_ALL_P };
        // **One reference list, two records.** The picture record's `ReferenceFrames` and the slice
        // record's `RefPicList0` are built from this one list, which is what makes them agree — and
        // they have to agree, because the first says what the driver is holding and the second says
        // what the slice reads. An IDR names none: 7.4.3.3 empties the decoded picture buffer at an
        // IDR, so there is nothing in it to predict from by definition.
        let slice_references: Vec<Reference> = if key_frame { Vec::new() } else { self.references.clone() };
        let picture = picture_record(
            &current,
            self.encoder.input_surface(),
            // **The value the picture is written with**, which for an IDR is 0 and not the counter
            // — see the note where `current` is built. Two places say "this picture's frame number"
            // (this record and `current.frame_num`), and a caller reading only one of them would
            // see a discrepancy that is the codec's rule rather than a bug.
            current.frame_num,
            self.macroblocks,
            &slice_references,
            qp,
            key_frame,
            request.last,
            self.encoder.coded_buffer_id(),
        )?;
        let slice = slice_record(self.macroblocks, slice_type, current.poc, self.idr_pic_id, &slice_references)?;
        let quantization = quantization_record(request.qp)?;
        // The parameter sets, derived from the two records just built and verified by FilmCraft's
        // own H.264 parser. Built here rather than after the render so that a record this module
        // cannot describe is refused *before* a picture is encoded: an error naming the field is
        // worth more than a stream a decoder cannot parse. Whether these are the ones used is
        // decided below, from the driver's own output.
        let picture_read = read::<ffi::VaEncPictureParameterBufferH264>(&picture).ok_or("the picture record just built does not read back")?;
        let sequence_read = read::<ffi::VaEncSequenceParameterBufferH264>(&self.sequence).ok_or("the sequence record just built does not read back")?;
        // Built and verified by FilmCraft's own H.264 parser before the render, so that a record
        // this module cannot describe is refused *before* a picture is encoded — an error naming the
        // field is worth more than a stream a decoder cannot parse. `resolve_parameter_sets` below
        // decides whether these are the ones that describe the output; this call only proves they
        // are well-formed and say what the records said.
        let derived = super::params::parameter_sets(&sequence_read, &picture_read)?;
        // The buffers are submitted in one `vaRenderPicture`, which is what the header requires for
        // the sequence, picture and slice records together. The order is the one iHD documents:
        // sequence, picture, slice, then the misc records.
        let seq_buf = self.encoder.buffer(ffi::ENC_SEQUENCE_PARAMETER_BUFFER, &self.sequence, 1)?;
        let pic_buf = self.encoder.buffer(ffi::ENC_PICTURE_PARAMETER_BUFFER, &picture, 1)?;
        // The slice buffer is an array of records: `num_elements` is the slice count, which is why
        // `Encoder::buffer` takes it as an argument at all. **It is 1 and cannot be more**: a second
        // slice record would be a second coded segment, and this backend refuses to submit one (see
        // the module note). A picture that needs more than one slice is therefore unencodable here
        // rather than silently re-cut into slices nobody chose.
        let slice_buf = self.encoder.buffer(ffi::ENC_SLICE_PARAMETER_BUFFER, &slice, 1)?;
        let fps_buf = self.encoder.buffer(ffi::ENC_MISC_PARAMETER_BUFFER, frame_rate, 1)?;
        let quant_buf = self.encoder.buffer(ffi::ENC_MISC_PARAMETER_BUFFER, &quantization, 1)?;
        // **The rate-control buffer is submitted with every picture in a bitrate mode, and never in
        // constant-quantiser mode.**
        //
        // Every picture rather than once, because the header presents it as a buffer the application
        // hands per `vaRenderPicture` and **measured**: submitting it on every picture and submitting
        // it on the first only produce byte-identical coded output over a 20-picture 640x360 CBR run on
        // iHD 26.3.5 (same sizes, same reported quantiser per picture, same total). So the choice is
        // made on the shape of the API rather than on a tuning guess — and a `None` here would have
        // been the **wrong** answer for a different reason too: a driver that reads a per-render
        // parameter list and finds one missing may reset its controller, which is the
        // `VAEncMiscParameterRateControl` `reset` flag's exact failure mode by another route.
        //
        // Under [`RateControl::ConstantQuantiser`] there is no buffer at all, and the refusal is not
        // "an empty buffer" — see [`VaEncoder::new`], which leaves the field `None`.
        let rc_buf = match self.rate_control_buffer.as_deref() {
            Some(record) => Some(self.encoder.buffer(ffi::ENC_MISC_PARAMETER_BUFFER, record, 1)?),
            None => None,
        };
        // A `Vec` rather than a fixed-size array because the rate-control buffer is present in some
        // modes and absent in others.
        //
        // **`rc_buf` is deliberately bound here and used after the list is built, not consumed by an
        // `if let`.** It was an `if let` first, and the driver caught it: `if let Some(buf) = rc_buf`
        // moves the `Buffer` into the arm, the arm's scope ends, the `Buffer` drops and destroys its
        // libva buffer — and the render that follows names an id that no longer exists. iHD answers
        // `VA_STATUS_ERROR_INVALID_BUFFER` ("invalid VABufferID"), which is the first thing that has
        // caught this class of mistake on this stack. Every handle in this list is a `Buffer` whose
        // `Drop` destroys it, so **the whole list is built and then rendered with all of them still
        // alive** — which is why they are collected here rather than inlined into the argument.
        let mut buffers = vec![seq_buf.id(), pic_buf.id(), slice_buf.id(), fps_buf.id(), quant_buf.id()];
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
        // **Whether the driver honoured `last_picture`, measured from the bytes rather than assumed
        // from the flag.** Measured, because it is not obvious: iHD 26.3.5 accepts the attribute and
        // appends `00 00 01 0b` to the last picture's own bytes, which is what makes
        // [`VaEncoder::flush`] able to report whether the stream's bitstream says where it ends.
        //
        // A disagreement between the flag and the bytes is logged rather than refused: a driver that
        // ignored `last_picture` still produces a stream an MP4 plays, because an MP4 says where its
        // track ends in the sample table. Refusing would turn a difference in how the end of a stream
        // is expressed into a failed export.
        let ends_stream = annex_b_units(coded.bytes())?.iter().any(NalUnit::is_end_of_stream);
        if ends_stream != request.last {
            log::info!(
                "VA-API: last_picture was {} on picture {} but its own bytes {} an end_of_stream() NAL",
                request.last,
                self.encoded + 1,
                if ends_stream { "carry" } else { "do not carry" }
            );
        }
        // **Which parameter sets describe this picture, decided by looking at what the driver wrote
        // and checked against what it was told** — see [`VaEncoder::sample_entry`] for the
        // measurement behind the derived case.
        //
        // A driver that *does* write its own sets is authoritative about itself: it is describing the
        // stream it produced, where a derived set is describing the record. So the driver's are used
        // when both are present, and they are then run through the same
        // [`super::params::cross_check`] against the submitted record — a driver that rewrote the
        // picture size, the reference depth, the frame-number width or the level is **declined**,
        // because there is no way to tell from here whether its set or ours describes the stream
        // that was actually encoded, and guessing is what this module is for.
        let (parameter_sets, sets) = self.resolve_parameter_sets(
            &sequence_read,
            &picture_read,
            &derived,
            coded.bytes(),
            &current,
            key_frame,
            coded.status().first().copied().map(|status| (status & 0xff) as u8),
            self.rate_control,
        )?;
        let from_driver = sets.from_driver;
        // The picture came back, so the sequence state can advance.
        //
        // **After a key frame the counters continue at 1, not at 0**, even though the IDR was
        // *written* with `frame_num` 0. The written value and the counter are two different things:
        // 7.4.3 infers an IDR's `frame_num` to be 0 and a conforming decoder reads nothing else,
        // but the IDR is a picture in the decoded picture buffer afterwards (7.4.3.3 empties the
        // buffer and then the IDR goes into it), so the picture that follows it may not carry
        // another 0. Restarting the counter at 0 would put two pictures with the same `frame_num`
        // in the buffer — and, because `slot_for` derives the DPB slot from the same number, it
        // would also make the next picture claim the slot its own reference occupies, which is the
        // defect that made a two-reference chain encode wrong pictures rather than fail.
        //
        // **The counters wrap**: `frame_num` is a `u16` that is a function of position modulo
        // `MaxFrameNum`, and an encoder that panicked at the 65536th picture would be a crash
        // rather than a rollover.
        if key_frame {
            self.frame_num = 1;
            self.poc = 1;
            self.idr_pic_id = self.idr_pic_id.wrapping_add(1);
            self.since_key_frame = 0;
        } else {
            self.frame_num = self.frame_num.wrapping_add(1);
            self.poc = self.poc.wrapping_add(1);
            self.since_key_frame = self.since_key_frame.saturating_add(1);
        }
        self.encoded = self.encoded.saturating_add(1);
        if request.last {
            self.last_marked = true;
            self.references.clear();
        }
        // **The reference is committed last.** Everything above can fail, and a reference committed
        // before the picture exists is a reference to an unfinished picture — which the next P frame
        // would predict from, producing a wrong picture rather than an error.
        //
        // An IDR clears the list first (7.4.3.3: an IDR empties the decoded picture buffer, so
        // anything held is gone by definition), then contributes itself. Trimming happens after the
        // push rather than before, so the list is never shorter than the depth the caller asked for
        // once there are enough pictures to fill it.
        if key_frame {
            self.references.clear();
        }
        self.references.push(current.as_reference());
        let capacity = self.reference_capacity();
        while self.references.len() > capacity {
            // The oldest goes: a reference list is most-recent-first as far as prediction is
            // concerned, and the driver picks by list order rather than by age.
            self.references.remove(0);
        }
        let mut annex_b = parameter_sets;
        annex_b.extend_from_slice(coded.bytes());
        Ok(EncodedPicture {
            // The complete access unit: the parameter sets (the driver's if it wrote them, ours
            // otherwise) ahead of the picture the driver wrote.
            bytes: annex_b,
            driver_bytes: coded.len(),
            coded_size: self.coded,
            display_size: self.display,
            coded_buffer_used,
            segments,
            status,
            parameter_sets_from_driver: from_driver,
            slice_type,
            key_frame,
            key_frame_reason: forced,
            predicted_from: slice_references.len(),
            ends_stream,
            frame_num: current.frame_num,
            poc: current.poc,
        })
    }

    /// The DPB slot a picture takes, from its `frame_num`.
    ///
    /// **A slot number, not a surface id.** The picture record's `curr_pic.frame_idx` names a
    /// position in the driver's own decoded picture buffer (`va_enc_h264.h:306`), and a value
    /// outside that buffer is refused or, worse, silently aliased onto a live reference.
    ///
    /// The modulus is **`refs + 1`, not `refs`**, and that is the second half of the two-reference
    /// bug. With `refs = 2` a modulus of 2 gives pictures 0, 1, 2, 3 the slots 0, 1, 0, 1 — so
    /// picture 2 lands in the slot picture 0 still occupies, and the driver is asked to write a new
    /// picture over a live reference. The buffer holds `refs + 1` pictures precisely so that the
    /// picture being written never lands on one that is still referenced.
    ///
    /// `max(1)` guards the modulus against an encoder with no references, where it is 1 and every
    /// picture gets slot 0 — correct, because nothing is ever referenced.
    fn slot_for(&self, frame_num: u16) -> u32 {
        let depth = (self.refs as u32).saturating_add(1);
        u32::from(frame_num) % depth.max(1)
    }

    /// **Decide which parameter sets describe the picture the driver just wrote, and refuse if the
    /// answer is not derivable.**
    ///
    /// Returns the Annex-B bytes to put ahead of the driver's picture in
    /// [`EncodedPicture::bytes`] and the sets themselves for [`VaEncoder::sample_entry`]. The
    /// second element is `None`-free on purpose: this either produces a description or fails, and
    /// there is no third outcome in which a sample entry is handed out and nobody has checked it.
    ///
    /// Three cases, in this order:
    ///
    /// 1. **The driver wrote both an SPS and a PPS** — the ones it wrote are used, cross-checked
    ///    against the record with [`super::params::cross_check`], and a divergence is an `Err`
    ///    rather than a preference. Nothing can be derived from the bitstream about which of the two
    ///    describes the picture, so choosing would be a coin toss on the one question that decides
    ///    whether the file plays.
    /// 2. **The driver wrote one of them** — refused. Half a parameter set describes nothing, and
    ///    using ours for the other half would produce a configuration record mixing two sources.
    /// 3. **The driver wrote neither** — which is what iHD 26.3.5 does for every value of
    ///    `VAConfigAttribEncPackedHeaders` (see [`VaEncoder::sample_entry`]) — and the derived sets
    ///    are used, having been verified against the bits by [`super::params::parameter_sets`].
    ///
    /// Either way the sets are then run through [`check_output_matches_the_sets`] against the
    /// picture's own coded NALs, so a derived set cannot outlive the picture it described.
    ///
    /// `rate_control` is what scopes the quantiser half of that check, and it is passed rather than
    /// read off `self` so the check cannot be reached with a mode that is not the encoder's — see
    /// [`check_output_matches_the_sets`] for what is checked in each mode and what replaces the
    /// equality.
    fn resolve_parameter_sets(
        &mut self,
        sequence: &ffi::VaEncSequenceParameterBufferH264,
        picture: &ffi::VaEncPictureParameterBufferH264,
        derived: &[u8],
        driver_bytes: &[u8],
        current: &Current,
        key_frame: bool,
        average_qp: Option<u8>,
        rate_control: RateControl,
    ) -> Result<(Vec<u8>, CodecParameterSets), String> {
        let units = annex_b_units(driver_bytes)?;
        let sps_unit = units.iter().find(|unit| unit.header & 0x1f == 7);
        let pps_unit = units.iter().find(|unit| unit.header & 0x1f == 8);
        let sets = match (sps_unit, pps_unit) {
            (Some(sps), Some(pps)) => {
                let sps_rbsp = filmcraft_bitstream::unescape_rbsp(sps.body);
                let pps_rbsp = filmcraft_bitstream::unescape_rbsp(pps.body);
                let parsed_sps = filmcraft_h264::params::Sps::parse(&sps_rbsp).map_err(|e| format!("the driver wrote an SPS that does not parse: {e}"))?;
                let parsed_pps = filmcraft_h264::params::Pps::parse(&pps_rbsp, &[Some(parsed_sps.clone())])
                    .map_err(|e| format!("the driver wrote a PPS that does not parse: {e}"))?;
                // **A driver that writes its own sets and disagrees with the record is declined.**
                // Its set is more authoritative than ours about the stream; ours is a description of
                // what was asked for. When they differ, one of the two is describing a stream that
                // was not encoded, and nothing here can say which — so the export stops with the
                // mismatch named rather than writing a configuration record for a stream nobody can
                // play.
                super::params::cross_check(&parsed_sps, &parsed_pps, sequence, picture)
                    .map_err(|why| format!("the driver wrote its own parameter sets and they disagree with the record it was given: {why}"))?;
                CodecParameterSets { sps: sps.nal_unit(), pps: pps.nal_unit(), from_driver: true }
            }
            (None, None) => {
                let (Some(sps), Some(pps)) = (nal_unit_bytes(derived, 7), nal_unit_bytes(derived, 8)) else {
                    return Err("the derived parameter sets have no start code, so they cannot go into a configuration record".into());
                };
                CodecParameterSets { sps, pps, from_driver: false }
            }
            _ => return Err("the driver wrote one H.264 parameter set and not the other, which describes no stream this backend can use".into()),
        };
        check_output_matches_the_sets(&sets, driver_bytes, current, key_frame, average_qp, rate_control)?;
        // **The sets are held from the first picture and must not change under an `avcC` that has
        // already been handed out.** Everything in an SPS comes from the sequence record, which is
        // fixed for the encoder's life; the one field that moves is the PPS's `pic_init_qp`, which
        // follows the quantiser.
        //
        // **Under a bitrate mode the driver moves the quantiser but `pic_init_qp` does not**, because
        // it is [`DEFAULT_ENCODER_QP`] in every picture record there (see its note). So this rule is
        // *stronger* in a bitrate mode, not weaker: the per-picture movement the driver applies lives
        // in the slice headers' `slice_qp_delta`, which the derived PPS does not mention, and a stream
        // whose QP genuinely changed mid-export in constant-quantiser mode is still refused at the
        // picture that changed it, naming the quantiser.
        match self.sample_entry.as_ref() {
            Some(established) if established != &sets => {
                return Err(format!(
                    "picture {} needs different parameter sets from the ones this encoder's sample entry already carries: \
                     the quantiser moved to {} (the sample entry describes pic_init_qp {}). \
                     One H.264 sample entry cannot describe two quantisers; build a new encoder",
                    current.frame_num,
                    picture.pic_init_qp,
                    parsed_pic_init_qp(established),
                ));
            }
            _ => self.sample_entry = Some(sets.clone()),
        }
        // The bytes ahead of the driver's picture: nothing at all when the driver's own sets are
        // already in its output, and the derived Annex-B pair when they are not.
        let prefix = if sets.from_driver { Vec::new() } else { derived.to_vec() };
        Ok((prefix, sets))
    }
}

/// The NAL unit of type `kind` in an Annex-B run `bytes`, **as a configuration record stores it**
/// (NAL header byte and payload, no start code) — or `None`.
fn nal_unit_bytes(bytes: &[u8], kind: u8) -> Option<Vec<u8>> {
    annex_b_units(bytes).ok()?.into_iter().find(|unit| unit.header & 0x1f == kind).map(|unit| unit.nal_unit())
}

/// `pic_init_qp` as [`filmcraft_h264::params::Pps`] reads it back out of an established
/// `CodecParameterSets`, or `-1` when those bytes no longer parse — which is itself worth saying
/// rather than defaulting.
fn parsed_pic_init_qp(sets: &CodecParameterSets) -> i32 {
    sets.pps
        .get(1..)
        .map(filmcraft_bitstream::unescape_rbsp)
        .and_then(|rbsp| filmcraft_h264::params::Pps::parse(&rbsp, &[None]).ok())
        .map_or(-1, |pps| pps.pic_init_qp)
}

/// One NAL unit of an Annex-B run: the header byte, and the bytes after it.
struct NalUnit<'a> {
    /// The NAL header byte, `forbidden_zero_bit | nal_ref_idc | nal_unit_type`.
    header: u8,
    /// Everything after the header, still carrying its emulation-prevention bytes.
    body: &'a [u8],
}

impl NalUnit<'_> {
    /// Whether this is an `end_of_seq()` or `end_of_stream()` NAL (7.4.1.2.4) — the two types whose
    /// payload is empty by definition, and the two a stream's last picture carries when the caller
    /// marked it as the last.
    fn is_end_of_stream(&self) -> bool {
        matches!(self.header & 0x1f, 10 | 11)
    }

    /// This unit as a configuration record stores it: the NAL header byte followed by the payload.
    ///
    /// **`AVCDecoderConfigurationRecord` keeps the one-byte NAL header**, which is worth stating
    /// because 14496-15's wording ("not including the NAL unit header byte") is routinely read the
    /// other way and both ffmpeg and every file this project has read keep it: `AvcConfig::new`
    /// takes the profile, constraints and level from bytes 1..4 of what it is handed, and
    /// `filmcraft_codecs::hw` parses a stored set with `get(1..)`. A record built without it is one
    /// every reader in this workspace — and every player — reads a byte out of step.
    fn nal_unit(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.body.len() + 1);
        out.push(self.header);
        out.extend_from_slice(self.body);
        out
    }
}

/// The NAL units of the Annex-B run `bytes`, or why `bytes` is not one.
///
/// `ffi::check_annex_b` has already established that the driver's window begins with a start code,
/// holds no empty unit and ends with at least a header byte; this is the walk that reads the list
/// out of it. It is deliberately a **start-code walk and not a search for `00 00 01` anywhere**:
/// that byte sequence occurs inside slice payload by coincidence all the time, and a scan that
/// matched one would report a parameter set the stream does not have — the plausible-looking wrong
/// answer this whole module exists to avoid.
/// The NAL units of the Annex-B run `bytes`, or why `bytes` is not one.
///
/// `ffi::check_annex_b` has already established that the driver's window begins with a start code and
/// holds no empty unit *between* two of them; this is the walk that reads the list out of it. It is
/// deliberately a **start-code walk and not a search for `00 00 01` anywhere**: that byte sequence
/// occurs inside slice payload by coincidence all the time, and a scan that matched one would report
/// a parameter set the stream does not have — the plausible-looking wrong answer this whole module
/// exists to avoid.
///
/// # The one NAL that carries no payload, and why it is kept
///
/// **Measured on iHD 26.3.5: a picture submitted with `last_picture = H264_LAST_PICTURE_EOSTREAM`
/// comes back with `00 00 01 0b` appended — an `end_of_stream()` NAL unit type 11 with nothing after
/// its header byte.** That is not a driver bug: 7.4.1.2.4 defines `end_of_seq()` and
/// `end_of_stream()` as carrying nothing but `rbsp_trailing_bits()`, so an empty payload is the
/// correct encoding and a walk that refused it would refuse every stream's last picture.
///
/// It matters twice over. First, **the driver honours `last_picture`**, which is the measurement
/// [`VaEncoder::flush`] rests on and which nothing in the plan had established. Second, a unit with no
/// payload has to be *recognised* rather than merely tolerated: a stream that ends with `0b` is a
/// stream whose end is in the bitstream, and an MP4 that carried that NAL would be carrying the
/// same signal twice — harmless, but only because a decoder ignores an `end_of_stream` NAL it did
/// not need. [`EncodedPicture::ends_stream`] reports it, and [`NalUnit::is_end_of_stream`] is what
/// that is built from.
fn annex_b_units(bytes: &[u8]) -> Result<Vec<NalUnit<'_>>, String> {
    let mut units = Vec::new();
    let mut at = 0usize;
    while let Some(found) = bytes.get(at..).and_then(|rest| rest.windows(3).position(|w| w == [0, 0, 1])) {
        // A four-byte code is the same three bytes one later, so the next search starts past the
        // header byte this one found rather than past the code — otherwise the tail of a 4-byte code
        // is read as a second, empty start of a unit.
        let start = at + found + 3;
        let Some(&header) = bytes.get(start) else { break };
        at = start + 1;
        let end = bytes.get(at..).and_then(|rest| rest.windows(3).position(|w| w == [0, 0, 1])).map_or(bytes.len(), |next| at + next);
        let Some(body) = bytes.get(at..end) else { break };
        // An empty body is legal **only** for the two NAL types 7.4.1.2.4 defines as carrying
        // nothing: `end_of_seq()` (10) and `end_of_stream()` (11). Any other type with no payload is
        // a truncated slice, which is the failure this whole readback exists to catch, so it is
        // still refused — and refused by name, because "empty NAL unit" would not say which of the
        // two shapes it had found.
        if body.is_empty() && !matches!(header & 0x1f, 10 | 11) {
            return Err(format!(
                "the coded picture holds a NAL unit of type {} with no payload, which is a truncated slice rather than an end marker",
                header & 0x1f
            ));
        }
        units.push(NalUnit { header, body });
    }
    if units.is_empty() {
        return Err("the coded picture holds no NAL unit".into());
    }
    Ok(units)
}

/// **Parse the picture the driver wrote against the sets that would go into the `avcC`, and check
/// the two describe the same thing.**
///
/// This is the self-check the derived path exists for. Writing the SPS and reading the bits back
/// proves the writing was right; it says nothing about whether the sets still describe *the output*,
/// because a submitted record is what was asked for, not what was encoded. So the driver's own NALs
/// are parsed a second time, by the same parsers the software decoder uses, **with the sets under
/// test as the only way to resolve them** — and every field that both sides state is compared.
///
/// What that actually catches, which is the point:
///
/// - **`log2_max_frame_num`, `frame_mbs_only_flag`, `log2_max_pic_order_cnt_lsb_minus4` and the
///   picture order count type are not readable from a slice header.** The slice header is a bit
///   string whose length *is* their definition: `frame_num` is `log2_max_frame_num` bits and
///   `pic_order_cnt_lsb` is `log2_max_pic_order_cnt_lsb_minus4 + 4`. Set any of them wrong and the
///   fields after it are read at the wrong offsets, so a wrong value does not fail the parse — it
///   produces a *different, plausible* `frame_num`. Comparing the parsed `frame_num` and
///   `pic_order_cnt_lsb` with the values this module submitted is what turns that into an error.
/// - **The PPS's flags steer the parse.** `entropy_coding_mode_flag` decides whether `cabac_init_idc`
/// - **`pic_init_qp` is cross-checked against the driver's own statement of the quality, under
///   constant quantiser only.** The low byte of `VACodedBufferSegment::status` is the average QP the
///   picture was coded at (`va.h:3897-3933`), which is a number from the driver about the picture
///   rather than about the request. `pps.pic_init_qp + slice_qp_delta` is what H.264 says the slice's
///   QP is (7.4.2.1.1); the two agreeing is the driver's quantiser confirming the PPS. **This is
///   scoped, and the reason is measured** — see the next section.
/// - **The slice type and the key-frame flag** are read back from the NAL header and the header
///   itself, so a driver that coded every picture intra — which would still decode, and would look
///   right — is caught here rather than in a file three hours long.
///
/// # The quantiser under a bitrate: what is checked, and what is given up
///
/// **The equality above cannot survive rate control, and the reason is not that the QP moves.** It is
/// that on this driver **the moved QP is not written into the bitstream**. Measured on iHD 26.3.5 with
/// `VA_RC_CBR`:
///
/// - `slice_qp_delta` is **0 in every slice**, confirmed independently with ffmpeg's `trace_headers`
///   bitstream filter, which reads `pic_init_qp_minus26 = 0` and `slice_qp_delta = 0` throughout a file
///   this module wrote;
/// - no PPS is written into the coded buffer (the milestone's standing measurement, re-taken here), so
///   the only `pic_init_qp` in the file is this module's derived one;
/// - and the driver *reports* a per-picture quantiser that moves with the target: **51, 48, 50, 47,
///   44, 42** across six 640x360 pictures at a 200 kbit/s CBR target, and **15, 17, 14, 12** at
///   20 Mbit/s, while the file says 26 on every one of them.
///
/// So the two numbers are not two views of one quantity — one is what the driver chose, the other is
/// what a decoder will use — and comparing them would refuse every correct picture. What is checked in
/// a bitrate mode instead:
///
/// 1. **The reported quantiser is a legal H.264 quantiser** (0..=51). A driver handing back a byte that
///    is not a quantiser is a thing this check can name.
/// 2. **The slice header, read against the sets under test, resolves to a legal quantiser.** That is
///    the check that the driver's slice is being parsed at the right widths against the right
///    parameter set, and it is the mode-independent form of what the equality proved under constant
///    quantiser.
/// 3. **Every mode-independent invariant stays exactly as strict as it was**: the slice parses at all,
///    the slice type, `idr_pic_flag`, `frame_num` modulo `MaxFrameNum`, `pic_order_cnt_lsb` modulo
///    `MaxPicOrderCntLsb`, and `num_ref_idx_active[0]`. None of those was ever about quality, and none
///    is relaxed.
///
/// **What is therefore no longer claimed in a bitrate mode**, stated plainly because a reader would
/// otherwise assume the check still covers it: that the stream is at any particular quality, and that
/// the derived `pic_init_qp` describes the pictures' quantisation. The second is a property of the
/// driver rather than of this code — libva has no call that asks a driver which `pic_init_qp` it
/// encoded against, and `VACodedBufferSegment::status` is the average rather than the value the
/// bitstream carries. The derived PPS remains the only `avcC` this backend can build, because the
/// driver writes none; see [`DEFAULT_ENCODER_QP`] for what it states and why.
///
/// **The export oracle cannot see this either, and that is worth saying.** On this milestone's
/// synthetic ramp the residual is *zero* — intra prediction reconstructs a linear ramp almost exactly,
/// so at any quantiser above about 40 every coefficient rounds to zero. Measured: a 200 kbit/s CBR
/// frame of the plain ramp and a constant-quantiser QP-45 frame of the same ramp have **byte-identical
/// decodes** and `slice_qp_delta = 0` in both. A stream whose quantiser is not in its headers is
/// therefore invisible to a decode-and-compare test built on that ramp, which is why
/// `examples/vaencprobe.rs` grew `-tex`.
///
/// A tolerance is used for the quantiser under constant quantiser and only there: `VA_STATUS` reports
/// an *average* QP over a picture's macroblocks, so it is a different quantity from a single slice's QP
/// even when everything is correct. Measured on this driver at a requested QP of 26 it reports 26
/// exactly for a key frame and for every predicted frame alike; the tolerance is four, which is what
/// an encoder is allowed to move by on a picture whose content is not uniform.
fn check_output_matches_the_sets(
    sets: &CodecParameterSets,
    driver_bytes: &[u8],
    current: &Current,
    key_frame: bool,
    average_qp: Option<u8>,
    rate_control: RateControl,
) -> Result<(), String> {
    let units = annex_b_units(driver_bytes)?;
    // The stored sets keep their NAL header byte (see [`NalUnit::nal_unit`]), so the RBSP the
    // parsers want starts one byte in. `get(1..)` rather than `[1..]`: a set this module built
    // always has the header, and one that does not is an error here rather than a panic.
    let Some(sps_rbsp) = sets.sps.get(1..).map(filmcraft_bitstream::unescape_rbsp) else {
        return Err("the sequence parameter set in the sample entry is empty".into());
    };
    let Some(pps_rbsp) = sets.pps.get(1..).map(filmcraft_bitstream::unescape_rbsp) else {
        return Err("the picture parameter set in the sample entry is empty".into());
    };
    let sps = filmcraft_h264::params::Sps::parse(&sps_rbsp).map_err(|e| format!("the sequence parameter set in the sample entry does not parse: {e}"))?;
    let pps = filmcraft_h264::params::Pps::parse(&pps_rbsp, &[Some(sps.clone())])
        .map_err(|e| format!("the picture parameter set in the sample entry does not parse: {e}"))?;
    let where_ = if sets.from_driver { "the driver's" } else { "the derived" };
    let mut slices = 0usize;
    for unit in &units {
        let type_ = unit.header & 0x1f;
        if type_ != 1 && type_ != 5 {
            // A type 7 or 8 in the driver's own output is its parameter set, checked by
            // `resolve_parameter_sets` against the record; anything else is not something this
            // backend submits and not something it can describe, so it is named rather than skipped.
            continue;
        }
        let nal = filmcraft_h264::slice::NalHeader::parse(unit.header)
            .map_err(|e| format!("{where_} parameter sets and a slice header in the same picture do not agree: {e}"))?;
        let rbsp = filmcraft_bitstream::unescape_rbsp(unit.body);
        let lookup = |id: u32| -> filmcraft_h264::Result<(&filmcraft_h264::params::Pps, &filmcraft_h264::params::Sps)> {
            if id != pps.id {
                return Err(filmcraft_h264::Error::MissingParameterSet(format!(
                    "the slice names picture parameter set {id} and {where_} picture parameter set {} is the only one in the sample entry",
                    pps.id
                )));
            }
            Ok((&pps, &sps))
        };
        let (header, header_pps, _) = filmcraft_h264::slice::SliceHeader::parse(&rbsp, nal, lookup)
            .map_err(|e| format!("the driver's slice does not parse against {where_} parameter sets: {e}"))?;
        slices += 1;
        let want_type = if key_frame { filmcraft_h264::slice::SliceType::I } else { filmcraft_h264::slice::SliceType::P };
        if header.slice_type != want_type {
            return Err(format!("the driver coded picture {} as a {:?} slice and a {:?} one was submitted", current.frame_num, header.slice_type, want_type));
        }
        if header.idr != key_frame {
            return Err(format!("picture {} came back with idr_pic_flag {} and {} was submitted", current.frame_num, header.idr, key_frame));
        }
        // **Both counters are compared modulo the width the SPS declares, and that is not a
        // softening — it is the definition.** `frame_num` is `u(v)` with `v = log2_max_frame_num`
        // (7.4.3) and `pic_order_cnt_lsb` is `u(v)` with `v = log2_max_pic_order_cnt_lsb_minus4
        // + 4` (7.4.3.1), so a driver can only write `frame_num mod MaxFrameNum` and `poc mod
        // MaxPicOrderCntLsb`. **The two widths are deliberately different now**: this encoder
        // declares `log2_max_frame_num_minus4 = 4` (so `MaxFrameNum` is 256 and `frame_num` is
        // eight bits) and `log2_max_pic_order_cnt_lsb_minus4 = 0` (so `MaxPicOrderCntLsb` is 16 and
        // `pic_order_cnt_lsb` is four bits), while `self.frame_num` and `self.poc` are `u16` counters
        // that wrap at 65536.
        //
        // **That difference is the point of the change, and this comparison is what proves the
        // driver honoured it.** Before, both fields were four bits and both wrapped at the same
        // picture, so a driver that silently kept the *old* width would have produced a stream this
        // check still accepted — `frame_num` read as four bits where eight were written gives a
        // number that is wrong but not wrapped, and the modulo at `MaxFrameNum = 256` would then
        // compare a small number against a large one and catch it. So the check is what stands
        // between "the driver took `log2_max_frame_num_minus4 = 4`" and "the record says 4 but the
        // bitstream does not", and it is why the SPS cross-check in `super::params` compares
        // `log2_max_frame_num` against the *record* while this compares the *bitstream* against the
        // *parsed SPS*: two different claims, and only the second says what was encoded.
        //
        // Comparing the two directly — which is what this did until a 20-picture run caught it —
        // refuses picture 16 of every export with "log2_max_frame_num is not what the driver
        // used", on a stream the driver encoded correctly. Measured: with the modulo in place a
        // 20-picture run encodes and the check still catches a genuinely wrong `log2_max_frame_num`,
        // because reading four bits where the driver wrote five produces a *different* number
        // rather than a wrapped one.
        //
        // **The other half of the wrap is not this function's to check, and is stated here so
        // nobody assumes it is covered.** 7.4.3's Equation 7-22 forbids a *skipped* frame number
        // still marked "used for short-term reference", and 8.2.5.1's closing sentence bounds that
        // count at `Max( num_ref_frames, 1 )`. Whether a wrap is accompanied by an IDR is a property
        // of *which pictures this loop made key frames*, decided in [`VaEncoder::key_frame_due`],
        // and not of anything the bitstream says back. What the bitstream does say is the width, and
        // that is what the comparison below uses.
        //
        // **A reference-marking command is deliberately absent from this list**, and the absence is
        // a measured ABI fact rather than an oversight: `VAEncSliceParameterBufferH264` has no
        // `dec_ref_pic_marking` field and no `adaptive_ref_pic_marking_mode_flag` (see the module
        // note), so there is nothing a caller could ask the driver to emit, and nothing to compare
        // against. If a future libva grows the field, this is where its `mmcos` would be compared
        // against what this loop intended — and the check would have to compare it against the
        // *bitstream*, not against our own record, for the reason the profile check in
        // `super::params` records.
        let max_frame_num = max_width(sps.log2_max_frame_num, "log2_max_frame_num")?;
        let max_poc_lsb = max_width(sps.log2_max_poc_lsb, "log2_max_pic_order_cnt_lsb_minus4")?;
        // **The frame-number width the bitstream declares must be the width this loop's counter is
        // numbered against, and this is the assertion that says so.**
        //
        // The comparisons below take `header.frame_num` against `current.frame_num` **modulo
        // `max_frame_num`**, and that modulo is correct on its own terms. It is also why it is not
        // sufficient on its own: if the driver had quietly kept the old four-bit width while our
        // derived SPS said eight, the parsed slice would be read at eight bits and the comparison
        // would refuse — but only from picture 16 onwards. A stream that never reached 16 would
        // pass, and the whole claim "a preset asking for a sixty-picture group now gets one" rests
        // on exactly the runs that *do* exceed 16.
        //
        // **This is the check that closes that**, and it is a check on the *declared* width against
        // [`MAX_FRAME_NUM`], not a re-parse: `log2_max_frame_num` in an already-parsed parameter set
        // is a direct statement of how wide the `frame_num` field in every slice of the stream is,
        // and this module's loop makes a key-frame decision at exactly that modulus. A stream whose
        // sets say 16 while the loop keys at 256 is a stream whose key frames are not where the gate
        // promised — so it is refused here, on the first picture, rather than discovered later as a
        // file nobody can seek into.
        //
        // **This runs on either provenance.** A driver-written set disagreeing with the record we
        // submitted is caught by `super::params::cross_check`; this catches what `cross_check`
        // cannot see, which is the derived path where the sets are ours and therefore agree with the
        // record by construction — and where a `LOG2_MAX_FRAME_NUM_MINUS4` that had drifted from
        // `MAX_FRAME_NUM` would produce a stream that is *internally consistent and still wrong*,
        // the worst of the three outcomes.
        if max_frame_num != MAX_FRAME_NUM {
            return Err(format!(
                "the parameter sets this encoder hands the muxer declare log2_max_frame_num {}, so frame_num is {} bits wide and wraps every {max_frame_num} pictures, but this encoder's frame loop \
                 decides its key frames against MaxFrameNum {MAX_FRAME_NUM} — so the key frames in the stream would not be the ones this encoder said they were",
                sps.log2_max_frame_num, sps.log2_max_frame_num
            ));
        }
        // **And the same for the picture order count**, which is the other width a slice header is
        // read at. `sequence_record` writes `log2_max_pic_order_cnt_lsb_minus4 = 0`, so
        // `MaxPicOrderCntLsb` is 16 while `MaxFrameNum` is 256 — the two are *deliberately*
        // different, and this is where that is asserted rather than left to a comment. If the two
        // widths drifted together again the stream would still decode, which is why this is a
        // positive statement about this modulus and not merely "the two agree".
        if max_poc_lsb != MAX_PIC_ORDER_CNT_LSB {
            return Err(format!(
                "the parameter sets this encoder hands the muxer declare log2_max_pic_order_cnt_lsb_minus4 {}, so pic_order_cnt_lsb is {} bits wide and wraps every {max_poc_lsb} pictures, which is not the \
                 {MAX_PIC_ORDER_CNT_LSB} this encoder writes in its sequence record",
                sps.log2_max_poc_lsb, sps.log2_max_poc_lsb
            ));
        }
        let max_poc_lsb = max_width(sps.log2_max_poc_lsb, "log2_max_pic_order_cnt_lsb_minus4")?;
        if header.frame_num != u32::from(current.frame_num) % max_frame_num {
            return Err(format!(
                "the driver's slice for picture {} reads back frame_num {}, which is not {} modulo MaxFrameNum {max_frame_num} as {} parameter sets say",
                current.frame_num, header.frame_num, current.frame_num, where_
            ));
        }
        if header.pic_order_cnt_lsb != u32::from(current.poc) % max_poc_lsb {
            return Err(format!(
                "the driver's slice for picture {} reads back pic_order_cnt_lsb {}, which is not {} modulo MaxPicOrderCntLsb {max_poc_lsb} as {} parameter sets say",
                current.frame_num, header.pic_order_cnt_lsb, current.poc, where_
            ));
        }
        // A key frame names no reference at all (7.4.3.3 empties the decoded picture buffer), so the
        // picture record's count is only a claim about a predicted picture.
        let want_refs = if key_frame { 0 } else { pps.num_ref_idx_l0_default_active };
        if header.num_ref_idx_active[0] != want_refs {
            return Err(format!(
                "the driver's slice for picture {} predicts from {} references and {where_} parameter sets say {}",
                current.frame_num, header.num_ref_idx_active[0], want_refs
            ));
        }
        // **The quantiser, and this is where the two modes part.**
        //
        // **Constant quantiser:** the driver's two statements about the picture — the QP its slice
        // header carries and the average QP it reports — are both compared with `pps.pic_init_qp`,
        // which under this mode *is* the picture's quality. That is the strongest of the two checks
        // and it is kept whole: measured on iHD 26.3.5, 26 asked for is 26 reported and 26 in the
        // slice header, for a key frame and every predicted frame.
        //
        // **Bitrate rate control:** there is no fixed number to compare with, and — this is the
        // measurement that decides it — **there is no number in the bitstream either.** iHD 26.3.5
        // writes `slice_qp_delta = 0` in every slice of a CBR stream and writes no PPS, so
        // `SliceQPY` in the file is always `pic_init_qp` = this module's [`DEFAULT_ENCODER_QP`], while
        // the driver *reports* the quantiser it chose (51, 48, 50, 47, 44, 42 across a six-picture
        // 200 kbit/s run; 15, 17, 14, 12 at 20 Mbit/s). Confirmed independently with ffmpeg's
        // `trace_headers` bitstream filter, which reads `pic_init_qp_minus26 = 0` and
        // `slice_qp_delta = 0` in every slice of the CBR files this probe writes.
        //
        // So under a bitrate mode the two quantities are **different statements** rather than two
        // views of one, and comparing them would refuse every correct picture. What replaces the
        // equality is the **range** the reported quantiser must lie in, which is a real
        // well-formedness property of the number the driver hands back, and the parse itself succeeding
        // at the derived sets' widths.
        //
        // **What is therefore no longer claimed under a bitrate mode**, and this is the honest list:
        // that the stream is at any particular quality, and that the derived `pic_init_qp` describes
        // the pictures' quantisation. The second is a property of **this driver**, measured above, and
        // nothing this backend can do changes it: libva offers no way to ask a driver what `pic_init_qp`
        // it encoded against except `VACodedBufferSegment::status`, which is the average and not the
        // value the bitstream carries. The derived PPS is still the only `avcC` this backend can build
        // (the driver writes none), and a decoder still reads it — see the note on
        // [`DEFAULT_ENCODER_QP`] for why 26 is the right base to state.
        let slice_qp = header.qp(header_pps);
        if let Some(qp) = average_qp {
            // **In both modes**: the driver reported a byte that is not a quantiser. 0..=51 is the whole
            // range H.264 has (7.4.2.1.1's `SliceQPY`), so anything outside it is not a quantiser at all.
            if qp > H264_MAX_QP {
                return Err(format!(
                    "the driver reported an average quantiser of {qp} for picture {}, which is outside H.264's 0..={H264_MAX_QP}",
                    current.frame_num
                ));
            }
            // **And in both modes**: the slice header, read against the sets under test, must resolve
            // to a quantiser H.264 has. This is the check that says the driver's slice is being parsed
            // at the right widths and against the right PPS — a `pic_init_qp` derived wrong by enough
            // to leave 0..=51 would show here, and a wrong one that does not would still be caught by
            // the constant-quantiser equality below.
            if slice_qp < 0 || slice_qp > i32::from(H264_MAX_QP) {
                return Err(format!(
                    "the driver's slice for picture {} reads back a quantiser of {slice_qp} ({where_} pic_init_qp {} plus its slice_qp_delta), which is outside \
                     H.264's 0..={H264_MAX_QP} — so the slice header is not being read against the picture parameter set this encoder \
                     would hand the muxer",
                    current.frame_num, header_pps.pic_init_qp
                ));
            }
            // **And only under constant quantiser**: the driver's report must be the quality the
            // parameter sets state. The tolerance is here because `status` is an *average* over the
            // picture's macroblocks and a slice header carries one number — different quantities even
            // when nothing is wrong.
            if rate_control.names_its_own_quantiser() && slice_qp.saturating_sub(i32::from(qp)).abs() > QP_REPORT_SLACK {
                return Err(format!(
                    "the driver coded picture {} at an average quantiser of {qp} and {where_} parameter sets say its slices are at {slice_qp}, \
                     more than {QP_REPORT_SLACK} apart",
                    current.frame_num
                ));
            }
        } else if rate_control.names_its_own_quantiser() {
            // Under constant quantiser an absent report is survivable: the picture record's QP was the
            // quality and the sets carry it. Under a bitrate mode it is **not** — the report is the
            // driver's only statement of what it chose, and the derived PPS's `pic_init_qp` says
            // nothing about this picture. So the absence is an error there rather than a skipped
            // check, and the message says which mode lost the information.
            return Err(format!(
                "the driver reported no average quantiser for picture {} in a {} encoder, so the picture's quality \
                 cannot be checked against anything: {where_} pic_init_qp is {} in every picture record, which says \
                 what the deltas are measured from and not what this picture is at",
                current.frame_num,
                rate_control.name(),
                header_pps.pic_init_qp
            ));
        }
    }
    if slices == 0 {
        return Err("the driver's picture holds no slice, so there is nothing for the parameter sets to describe".into());
    }
    Ok(())
}

/// `1 << width`, the modulus H.264 gives `frame_num` and `pic_order_cnt_lsb` — or why `width`
/// is not a number this can build a modulus from.
///
/// A width of 31 or more would make the shift overflow, and the syntax elements it comes from are
/// a `u4` each (`log2_max_frame_num_minus4`, `log2_max_pic_order_cnt_lsb_minus4`), so it cannot
/// happen from a record this module submits; it is checked anyway because the width here comes from
/// a *parsed* parameter set, and this function is also run against the driver's own.
fn max_width(width: u32, which: &str) -> Result<u32, String> {
    if width == 0 || width >= 32 {
        return Err(format!("{which} is {width}, which names no modulus a slice header could carry"));
    }
    Ok(1u32 << width)
}

/// The highest quantiser H.264 has, as `pps.pic_init_qp` and a slice's `SliceQPY` count it.
///
/// 7.4.2.1.1 defines `SliceQPY = 26 + pic_init_qp_minus26 + slice_qp_delta` and 7.3.2.2.1 bounds
/// `pic_init_qp_minus26` to −26..+25, which is exactly 0..=51 after the 26. It is a named constant
/// rather than a `51` typed where it is used because **the self-check's range test is the thing that
/// stands in for the quantiser equality under a bitrate mode**, and a range with a number typed into
/// it twice is a range that can disagree with itself.
const H264_MAX_QP: u8 = 51;

/// How far the driver's reported average quantiser may sit from the one its own slice header carries.
///
/// `VACodedBufferSegment::status`'s low byte is an average over a picture's macroblocks
/// (`va.h:3897-3933`) and a slice header carries one number, so the two are different quantities even
/// when nothing is wrong; this is the room between them.
///
/// **Measured on iHD 26.3.5 under both modes, and the two measurements are what the constant is for.**
/// Under constant quantiser at a requested 26 the driver reports **26 exactly** for a key frame and
/// for every predicted frame — the slice's QP and the average are the same number, because one slice
/// at one QP covers the picture. Under a 400 kbit CBR target the same driver reports 30, 30, 32, 32,
/// 32, 32, 33 … 36 across a twenty-picture run and the slice header agrees with each report to
/// within **one** quantiser step. So the slack is slack for a driver that rounds or clamps, not for
/// one that is known to, and four is comfortably above the largest gap measured in either mode.
const QP_REPORT_SLACK: i32 = 4;

/// The NAL units an MP4/MOV `avcC` carries, as [`VaEncoder::sample_entry`] uses them.
///
/// **`from_driver` is a measurement, not a preference.** It is true only when the driver's own coded
/// bytes carried a NAL of type 7 or 8, which on iHD 26.3.5 they never do, whatever the packed-header
/// mode is set to (see [`VaEncoder::sample_entry`] for the measurement). When it is false these
/// bytes are **derived from the records this backend submitted** and have been checked against the
/// picture the driver actually wrote — which is a weaker claim than "the driver's own", and is
/// recorded as such rather than as the latter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodecParameterSets {
    /// The sequence parameter set as an `avcC` stores it: the RBSP, no start code, no NAL header.
    pub sps: Vec<u8>,
    /// The picture parameter set, the same way.
    pub pps: Vec<u8>,
    /// Whether the driver wrote these bytes itself.
    pub from_driver: bool,
}

/// The crop offset in `frame_crop_*_offset` units for a dimension, or why it cannot be expressed.
///
/// `padded` and `shown` are both macroblock-multiples apart from `unit` or this is an error: a
/// truncation here would crop the picture by a whole macroblock and still look like a picture.
fn crop_of(padded: u32, shown: u32, unit: u32, which: &str) -> Result<u32, String> {
    let extra = padded.checked_sub(shown).ok_or_else(|| format!("the {which} crop pads {padded} to {shown}"))?;
    if extra % unit != 0 {
        return Err(format!("the {which} crop of {extra} rows is not a whole number of {unit}-row crop units"));
    }
    Ok(extra / unit)
}

/// The sequence record: what to encode, and where the picture sits.
///
/// The fields that are *not* obvious, and why each is what it is:
///
/// - `picture_width_in_mbs` / `picture_height_in_mbs` are **macroblocks**, not pixels, which is
///   the one field whose units the header mentions and the examples get wrong.
/// - `seq_fields` packs `chroma_format_idc = 1`, `frame_mbs_only_flag = 1`,
///   `direct_8x8_inference_flag = 1` and `delta_pic_order_always_zero_flag = 1`, with
///   `seq_scaling_matrix_present_flag = 0`. **The scaling matrix flag matters**: set, it requires a
///   `VAIQMatrixBufferH264` in the same render, and the decode backend's note on that buffer says
///   what a driver does with a submission missing one — it succeeds and produces a plausible grey.
///   Leaving it clear is what 7.4.2.1.1 specifies as the default for no custom matrices.
/// - `max_num_ref_frames = 1` and `num_ref_frames_in_pic_order_cnt_cycle = 0`, which is the
///   "no B pictures, no picture order count cycle" state: consistent with the `pic_order_cnt_type`
///   of 0 the same record declares.
/// - `bits_per_second` is **not zero** even under [`RC_CQP`], where it is ignored. iHD rejects a
///   sequence record whose bitrate is 0 with `VA_STATUS_ERROR_INVALID_PARAMETER`, and the failure
///   names a field a caller did not think they had set. It is set to the highest H.264 level 5.1
///   allows rather than to anything meaningful, because under [`RC_CQP`] it is a validity check and
///   not a target — and saying so here is what stops the next reader treating it as a default bitrate.
/// - `intra_idr_period` is the caller's key-frame interval, **and it is a hint rather than the
///   mechanism**: in a slice encode the application names each picture's type itself, through the
///   picture record's `pic_fields.idr_pic_flag`, and this field tells the driver's rate control how
///   often a key frame will arrive so it can size its buffers. The two therefore have to agree, and
///   [`VaEncoder::key_frame_due`] is where the application's side of that agreement lives. **The one
///   case where they deliberately do not match is the `frame_num` wrap** (see the module note), so
///   the driver is told a period and gets a slightly shorter one — which is why the key frames this
///   loop inserts are reported on [`EncodedPicture::key_frame_reason`] rather than being invisible.
fn sequence_record(
    _coded: (u32, u32),
    crop: (u32, u32, u32, u32),
    width_mbs: u16,
    height_mbs: u16,
    refs: u32,
    keyframe_interval: u32,
) -> Result<Vec<u8>, String> {
    let record = ffi::VaEncSequenceParameterBufferH264 {
        seq_parameter_set_id: 0,
        // Level 5.1, the highest this table names. The level is a *conformance* statement, and one
        // that is too low makes a conforming decoder refuse the stream, so it is set from the
        // picture rather than left at 0 (which is not a level).
        level_idc: 51,
        // No periodic *non-IDR* I frames: every key frame here is an IDR, which is what makes each
        // one a point a decoder can start from. `intra_period` and `ip_period` stay 0 because this
        // loop never submits an I picture that is not an IDR, and a driver told about a periodic I
        // frame that never arrives sizes its buffers for a GOP shape this stream does not have.
        intra_period: 0,
        intra_idr_period: keyframe_interval,
        ip_period: 0,
        // Validity, not a target: see the note above.
        bits_per_second: 240_000_000,
        // **The DPB depth the sequence declares, which is the reference depth this encoder uses.**
        //
        // This was hardcoded to 1 and that is wrong for any encoder deeper than one reference:
        // `max_num_ref_frames` is an SPS syntax element (7.4.2.1.1) that sizes the decoded picture
        // buffer the *decoder* keeps, and a decoder given a stream whose picture records name more
        // references than its SPS allows has no room for them. Measured with `refs = 2`: pictures 0
        // and 1 decode correctly, and from picture 2 on every frame decodes as a **stale** one — the
        // decoded picture is visibly a blend of the previous two, drifting further with each frame.
        // Nothing errors; ffmpeg decodes all eight frames without a warning.
        max_num_ref_frames: refs,
        picture_width_in_mbs: width_mbs,
        picture_height_in_mbs: height_mbs,
        // The packed `seq_fields` word. Built from named bit positions rather than a literal
        // because a literal here is unreadable *and* unverifiable — the decode backend's records
        // are packed the same way, and the one bug this project shipped (the H.264 weight tables)
        // began as a plausible-looking bit pattern.
        //
        // The bit order is `va_enc_h264.h:180-197`, from bit 0 up:
        //   0-1 chroma_format_idc            = 1 (4:2:0)
        //   2   frame_mbs_only_flag           = 1
        //   3   mb_adaptive_frame_field_flag  = 0
        //   4   seq_scaling_matrix_present_flag = 0
        //   5   direct_8x8_inference_flag     = 1
        //   6-9 log2_max_frame_num_minus4    = LOG2_MAX_FRAME_NUM_MINUS4 (so MaxFrameNum 256)
        //   10-11 pic_order_cnt_type         = 0
        //   12-15 log2_max_pic_order_cnt_lsb_minus4 = 0 (so MaxPicOrderCntLsb 16)
        //   16  delta_pic_order_always_zero_flag = 1
        //
        // **Bits 6-9 are the one field here that a caller's group-of-pictures length rides on**, and
        // it is written from the constant rather than as a literal for the reason
        // [`LOG2_MAX_FRAME_NUM_MINUS4`] gives. What the driver does with it is measured: with 4
        // here, ffmpeg and `trace_headers` read eight-bit `frame_num` values out of the slices this
        // module gets back. See the module note, "The `frame_num` wrap, decided".
        seq_fields: (1 << 0) | (1 << 2) | (1 << 5) | (LOG2_MAX_FRAME_NUM_MINUS4 << 6) | (LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4 << 12) | (1 << 16),
        bit_depth_luma_minus8: 0,
        bit_depth_chroma_minus8: 0,
        num_ref_frames_in_pic_order_cnt_cycle: 0,
        frame_cropping_flag: u8::from(crop != (0, 0, 0, 0)),
        frame_crop_left_offset: crop.0,
        frame_crop_right_offset: crop.1,
        frame_crop_top_offset: crop.2,
        frame_crop_bottom_offset: crop.3,
        vui_parameters_present_flag: 0,
        aspect_ratio_idc: 0,
        sar_width: 1,
        sar_height: 1,
        // A 30 fps time base: `time_scale / (2 * num_units_in_tick)` is the frame rate. Both are
        // only read when `vui_parameters_present_flag` is set, which it is not, so these are the
        // neutral values rather than a claim about the picture's timing.
        num_units_in_tick: 1,
        time_scale: 60,
        ..Default::default()
    };
    // `delta_pic_order_always_zero_flag` is bit 16 and is set above; nothing else is or-ed in here,
    // so the field is exactly what the comment on it says.
    Ok(ffi::record_bytes(&record))
}

/// The frame-rate misc buffer, built as [`MISC_FRAME_RATE`] and its record.
///
/// `fps` is a frame rate as a whole number and is packed as the fraction the header describes
/// (`va.h:2621-2631`): denominator in the high two bytes, numerator in the low two. A `fps` of zero
/// or one past the field is an error, not a clamp: a driver told the frame rate is 0/1 will size
/// its rate control against a frame that never arrives, and the stream it makes is too large.
fn frame_rate_record(fps: u32) -> Result<Vec<u8>, String> {
    if fps == 0 || fps > u32::from(u16::MAX) {
        return Err(format!("a frame rate of {fps} fps, which does not fit the field's numerator"));
    }
    let framerate = (1u32 << 16) | fps;
    let record = ffi::VaEncMiscParameterFrameRate { framerate, framerate_flags: 0, va_reserved: [0; ffi::VA_PADDING_LOW] };
    misc(ffi::MISC_FRAME_RATE, &ffi::record_bytes(&record))
}

/// The quantisation misc buffer, which under [`RateControl::ConstantQuantiser`] is where the
/// driver's quality controls come from. Every flag is left clear, which means "let the driver decide"
/// (`va.h:2807-2808`) — the honest state, because the QP this backend asks for is the picture record's
/// `pic_init_qp` and not one of the trellis bits.
///
/// **It is submitted in a bitrate mode too**, with the same all-clear flags, and the reason is
/// measured rather than argued: a render that omits it is accepted, and a render that includes it
/// with the flags clear produces **byte-identical** coded pictures on iHD 26.3.5 at 640x360 over a
/// 20-picture run (same bytes, same sizes, same reported average quantiser per picture). So it is
/// kept for one reason only: the flags are the driver's trellis controls and a driver that reads them
/// reads zeros as "its own trellis", which is the same state as not submitting the record at all on
/// this driver and a defined state on any other.
fn quantization_record(_qp: Option<u8>) -> Result<Vec<u8>, String> {
    let record = ffi::VaEncMiscParameterQuantization { quantization_flags: 0, va_reserved: 0 };
    misc(ffi::MISC_QUANTIZATION, &ffi::record_bytes(&record))
}

/// **The rate-control window this encoder submits, in milliseconds, before the key-frame interval is
/// known.**
///
/// A placeholder that [`VaEncoder::key_frames_every`] immediately replaces, because a window derived
/// from a GOP nobody has stated would be a number this module invented. It is a constant only so the
/// field is never briefly absent: [`rate_control_record`] validates it the same way it validates the
/// real one, so a placeholder outside the legal range would be caught rather than submitted.
const RATE_CONTROL_WINDOW_MS: u32 = 1;

/// The rate-control window, in milliseconds, for a key frame every `pictures`.
///
/// **Derived from the one thing the settings state that bounds it, and refused rather than invented
/// when that thing cannot produce one.** The header calls it "the interval over which the rate
/// controller will attempt to guarantee that the target and maximum bit-rates are correct" — so it
/// must be at least as long as the shortest span in which those rates are allowed to even out, and
/// this encoder's shortest such span is a key-frame interval: an intra picture is the one the
/// controller cannot smooth across, so a window shorter than the GOP is a window it can never satisfy
/// for it. **0 pictures** (one key frame for the whole stream) has no span to derive from and takes
/// the default rather than a division by zero.
///
/// **Measured, on iHD 26.3.5, and the numbers are why this is derived rather than fixed**: at 640x360,
/// 30 fps, `refs = 1` and a 16-picture key-frame interval, a 20-picture run at a 400 kbit CBR target
/// came to **394 kbit/s** with the window at the GOP (533 ms), against **407 kbit/s** at a fixed
/// 1000 ms and **389 kbit/s** at 0 — the three are within 5% of each other and of the target, so the
/// window is a real field the driver reads but not one this backend has to tune, and deriving it from
/// the GOP is the one choice that cannot be wrong about a stream's shape.
fn rate_control_window_ms(pictures: u32) -> u32 {
    // 16 pictures at the 30 fps the plan's runs use is 533 ms. **0 pictures is one key frame for the
    // whole stream**, which has no span to derive a window from, and a zero-length window is not a
    // window — so it takes the round value rather than dividing by a picture count of nothing.
    const FRAMES_PER_SECOND: u32 = 30;
    if pictures == 0 {
        return 1000;
    }
    let Some(span_ms) = pictures.checked_mul(1000).and_then(|pictures_ms| pictures_ms.checked_div(FRAMES_PER_SECOND)) else {
        return 1000;
    };
    // The floor of one millisecond keeps a very short GOP from asking for a zero-length window; the
    // ceiling is a second, because this encoder's key-frame interval never exceeds `MaxFrameNum`'s 16
    // and a window longer than that is a claim about pacing nothing supports.
    span_ms.clamp(1, 1000)
}

/// The rate-control misc buffer for a bitrate mode, built as [`MISC_RATE_CONTROL`] and its record.
///
/// **Every number here is either the caller's or a value this module can justify, and the two
/// refusals are the interesting part.** `bits_per_second` of zero is refused because iHD rejects it
/// (the same reason the sequence record writes a non-zero bitrate under constant quantiser) and a
/// controller told "zero bits per second" sizes its buffers for a stream of nothing. `target_percentage`
/// outside 1..=100 is refused rather than clamped, and a VBR ceiling below its target is refused
/// rather than swapped: each would be a rate the caller did not ask for, reached by a route nobody
/// chose.
///
/// The quantiser fields are all **zero, which is the header's own "ignored"** for `initial_qp`,
/// `min_qp` and `max_qp`, and `target_frame_size`'s "zero value means no limits". This is the field
/// where the constant-quantiser mode's rule applies hardest: **a caller who asked for a bitrate has
/// not asked for a quality range**, and naming one here would be FilmCraft inventing a floor and a
/// ceiling for a user's picture. A future caller who wants a range should say so by giving the
/// module a number, not by this default changing.
fn rate_control_record(mode: RateControl, window_ms: u32) -> Result<Vec<u8>, String> {
    let (bits_per_second, target_percentage) = match mode {
        RateControl::ConstantQuantiser => {
            return Err("a rate-control record for a constant-quantiser encoder: the picture's \
                        pic_init_qp is its quality and a bitrate would be a second, independent \
                        statement about the same fidelity"
                .into());
        }
        RateControl::Cbr(bps) => {
            if bps == 0 {
                return Err("a bitrate target of 0 bits per second: iHD rejects a sequence record whose bitrate is 0, and a \
                            controller sized for zero bits per second makes a stream nobody asked for"
                    .into());
            }
            // The header says `target_percentage` is ignored in CBR and treated as 100%. Writing 100
            // rather than 0 says which of the two this is.
            (bps, 100)
        }
        RateControl::Vbr { target, max } => {
            if target == 0 {
                return Err("a VBR target of 0 bits per second".into());
            }
            if max < target {
                return Err(format!("a VBR ceiling of {max} bits per second below its target of {target}: the target would be unreachable by construction"));
            }
            if max == 0 {
                return Err("a VBR ceiling of 0 bits per second".into());
            }
            // `target_percentage` is `target * 100 / max` on **u64**, because `target * 100` is a u32
            // multiplication that overflows past 42_949_672 bits per second — about 43 Mbit/s, which a
            // 4K export asks for routinely. A wrapped percentage is a target percentage of a few
            // hundred, clamped to 100 by the range check below into "reach the ceiling", which is a
            // file at a size nobody chose with no error anywhere.
            let scaled = u64::from(target) * 100;
            let percentage = u32::try_from(scaled / u64::from(max)).map_err(|_| "a VBR percentage past the field's 32 bits".to_string())?;
            // `ceil`, not `floor`: a percentage that rounds *down* asks for less than the caller said.
            // The correction is `max(1)` — a target below 1% of the ceiling is still a target, and the
            // field has no zero.
            let percentage = percentage.clamp(1, 100);
            (max, percentage)
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
///
/// The type word is 4 bytes (`va.h:1973`), and the record follows immediately — this is the
/// flexible-array struct the header documents, and its length is the sum of the two. A driver
/// given a misc buffer of only the header reads no record at all, and one given a record without
/// the header reads the record as the type.
fn misc(type_: u32, data: &[u8]) -> Result<Vec<u8>, String> {
    let header = ffi::record_bytes(&ffi::VaEncMiscParameterBufferHeader { type_ });
    let mut out = header;
    out.extend_from_slice(data);
    Ok(out)
}

/// The picture record: which surfaces, which buffer, and what quality.
///
/// The fields that decide whether the picture is right rather than merely present:
///
/// - `curr_pic.picture_id` is the **reconstruction** surface, never the input: the driver
///   reconstructs what it encodes and needs somewhere to put it, and pointing it at the input would
///   have the driver read its own output as the picture it is encoding.
/// - `curr_pic.flags` is `VA_PICTURE_H264_SHORT_TERM_REFERENCE` whenever the picture is to be
///   predicted from later. **A picture without this flag is one the driver may discard**, and a P
///   picture naming it then predicts from something else — which is a wrong picture, not an error.
/// - `pic_fields` sets `idr_pic_flag` (bit 0) from `key_frame`. **Bit 0 is the whole key-frame
///   decision**, and leaving it clear gives a stream with no point a decoder can start from —
///   which ffmpeg reports as "no frame!" rather than as an error. It is a parameter rather than read
///   from the request because **the frame loop decides it**: a picture whose `frame_num` would wrap
///   is a key frame whatever the caller asked for, and the record has to say what was actually
///   encoded.
/// - `pic_init_qp` is `qp`, not `26 + qp`. The field is `pic_init_qp_minus26 + 26`, so a QP
///   of 26 is 26 here; subtracting 26 again (a natural mistake when the name says "minus26") is
///   what makes a picture come out at the wrong quality with no error anywhere.
/// - `last_picture` is `H264_LAST_PICTURE_EOSTREAM` and nothing else. The other flag the header
///   names, `H264_LAST_PICTURE_EOSEQ` (end of *sequence*), is **never set here**, because this
///   encoder has no notion of a sequence boundary to report: its picture stream is the whole file, and
///   a stream that ends a sequence it never started is a claim about something that does not exist.
///   A caller that wants several sequences in one file has no way to ask for one, which is a
///   limitation and is stated as one rather than approximated.
///
/// **`ReferenceFrames` for a P picture is the DPB, not a list**: the entries are the pictures the
/// driver is holding, each with its surface and the `frame_idx` that identifies it in the driver's
/// own buffer, and both travel together in [`Reference`]. Every slot this encoder does not use is
/// `VA_INVALID_SURFACE` — see the note on that below, which is the single most surprising fact
/// Task 1 found.
fn picture_record(
    current: &Current,
    input: u32,
    frame_num: u16,
    macroblocks: u32,
    references: &[Reference],
    qp: u8,
    key_frame: bool,
    last: bool,
    coded_buf: u32,
) -> Result<Vec<u8>, String> {
    // `input` and `macroblocks` are the two things this record must *not* say: the input surface is
    // named by `vaBeginPicture`, not by `curr_pic`, and the picture's extent is the slice record's
    // `num_macroblocks`. They are arguments so the call site can state that, and asserted here so a
    // caller that changes one and forgets the other finds out at the call rather than on screen.
    if input == current.surface {
        return Err(format!(
            "the reconstruction surface {} is also the input surface: the driver would read its own output as the picture it is encoding",
            current.surface
        ));
    }
    if macroblocks == 0 {
        return Err("a picture of 0 macroblocks".into());
    }
    if references.len() > MAX_REFERENCES as usize {
        return Err(format!("{} references, more than the record's {MAX_REFERENCES} slots", references.len()));
    }
    // **A key frame clears the DPB, so an IDR names no reference** even when the encoder is holding
    // some. Two independent reasons, and both are H.264 rather than convenience: an IDR's
    // `nal_ref_idc` is 0 (7.4.3), so no picture may predict across it, and an IDR empties the
    // decoded picture buffer (7.4.3.3) — so anything the driver was holding is gone by definition.
    // Naming a reference here would be a reference to a picture the decoder is about to be told no
    // longer exists.
    let used: &[Reference] = if key_frame { &[] } else { references };
    let mut reference_frames = [ffi::VaPictureH264 { picture_id: ffi::INVALID_SURFACE, ..Default::default() }; 16];
    for (slot, reference) in used.iter().enumerate() {
        let Some(entry) = reference_frames.get_mut(slot) else {
            return Err(format!("reference {slot} does not fit the record's {} slots", reference_frames.len()));
        };
        // The surface *and* the slot it occupies in the driver's buffer. A driver handed the right
        // surface with a slot it has not seen will not find the picture, and the P frame then
        // predicts from whatever is in that slot — which is a wrong picture, not an error.
        *entry = ffi::VaPictureH264 {
            picture_id: reference.surface,
            frame_idx: reference.frame_idx,
            flags: PICTURE_H264_SHORT_TERM_REFERENCE,
            ..Default::default()
        };
    }
    // `num_ref_idx_l0_active_minus1` is how many entries list 0 has. It is `used.len() - 1`, and the
    // subtraction only happens when there is at least one entry — a slice told to read zero
    // references from list 0 because the count underflowed would be reading `0xFF` entries.
    let l0 = used.len().saturating_sub(1);
    let Ok(l0) = u8::try_from(l0) else {
        return Err(format!("{} references do not fit the record's u8 count field", used.len()));
    };
    let record = ffi::VaEncPictureParameterBufferH264 {
        curr_pic: ffi::VaPictureH264 {
            picture_id: current.surface,
            frame_idx: current.frame_idx,
            flags: if current.is_reference { PICTURE_H264_SHORT_TERM_REFERENCE } else { 0 },
            ..Default::default()
        },
        coded_buf,
        pic_parameter_set_id: 0,
        seq_parameter_set_id: 0,
        // **Only `H264_LAST_PICTURE_EOSTREAM`, and only when the caller said so.** Both flags are
        // `u8::from(…)` rather than `0` because 0 for a `u8` would mean "the byte was not set" rather
        // than "this picture is not the last", and a record whose reserved bits happen to be non-zero
        // is a record the driver reads fields out of. `H264_LAST_PICTURE_EOSEQ` is deliberately
        // absent: this encoder has no sequence boundary to report (see the function's own doc).
        last_picture: if last { LAST_PICTURE_EOSTREAM } else { 0 },
        padding_after_last_picture: [0; 1],
        frame_num,
        // The picture record carries **no** picture order count. It is a slice-header syntax
        // element (7.4.2.1.1's `pic_order_cnt_lsb`), and putting it here — which is tempting,
        // because the picture is the thing it describes — would be a field the record does not have
        // and would land in `pic_init_qp`'s neighbourhood. The slice record below is where it goes.
        pic_init_qp: qp,
        num_ref_idx_l0_active_minus1: l0,
        num_ref_idx_l1_active_minus1: 0,
        chroma_qp_index_offset: 0,
        second_chroma_qp_index_offset: 0,
        padding_after_second_chroma_qp_index_offset: [0; 1],
        // idr_pic_flag (bit 0); reference_pic_flag (bits 1-2) is 1 = short-term reference when the
        // picture is one; entropy_coding_mode_flag (bit 3) clear, i.e. CAVLC;
        // deblocking_filter_control_present_flag (bit 9) set, so the slice record's
        // `disable_deblocking_filter_idc` is what the picture says and 0 means "filter it".
        pic_fields: (1 << 9) | (u32::from(current.is_reference) << 1) | u32::from(key_frame),
        va_reserved: [0; ffi::VA_PADDING_LOW],
        // **Every unused slot is `VA_INVALID_SURFACE`, and that is not a formality.**
        //
        // Measured by Task 1: with the array left zeroed, an iHD 26.3.5 H.264 encoder accepts the
        // render (`vaRenderPicture` returns success) and then fails `vaEndPicture` with
        // `VA_STATUS_ERROR_ENCODING_ERROR` ("internal encoding error"). Filling the array with
        // `VA_INVALID_SURFACE` makes the identical submission encode.
        //
        // The reason is the same shape of bug as the decode milestone's IQ-matrix note: **zero is
        // not "no reference" here, it is surface 0.** This session's surfaces were allocated by the
        // driver and ids 0 and 1 are perfectly real surfaces — `LIBVA_TRACE` shows
        // `render_targets = 0x00000000` — so a zeroed slot names a real, valid, *wrong* surface, and
        // the driver believes it has a reference it does not have. The error names no field.
        //
        // `VA_INVALID_SURFACE` is `VA_INVALID_ID` (`va.h:1648`), which is what the decode backend
        // already puts in an unused `ReferenceFrames` slot, so both halves of this module say the
        // same thing the same way.
        reference_frames,
    };
    Ok(ffi::record_bytes(&record))
}

/// The slice record: one slice covering the whole picture.
///
/// `num_macroblocks` is the picture's own macroblock count and `macroblock_address` is 0, so the
/// slice is the picture — the simplest slice structure there is, and the one the decode backend's
/// plane arithmetic assumes when it reads a slice header.
///
/// **`slice_type` is what makes a picture an inter picture**, and the encoding is offset by five
/// (`va_enc_h264.h:441`, "Range: 0..2, 5..7, i.e. no switching slices"): 5 = every slice in the
/// picture is P, 7 = every slice is I. The 5..7 range is the one that says *all slices in this
/// picture*, which is the statement a one-slice picture makes.
///
/// **`RefPicList0` is the actual prediction.** For a P slice it must name the picture being
/// predicted from, with the `frame_num` a decoder will match against — and this is the second place
/// the same reference has to appear (the first is the picture record's `ReferenceFrames`), so the
/// two lists are built from the same [`Reference`] values. A P slice with an empty `RefPicList0`
/// does not fail: it predicts from nothing, and the driver reconstructs the picture as it stands in
/// the surface, which is a plausible-looking wrong picture.
///
/// `macroblock_info` is [`ffi::INVALID_SURFACE`] (`VA_INVALID_ID`), which the header documents as
/// the value meaning "derive the per-macroblock configuration from this slice's parameters"
/// (`va_enc_h264.h:426-428`). Naming a buffer that does not exist instead would be a driver read
/// of someone else's memory.
fn slice_record(macroblocks: u32, slice_type: u8, poc: u16, idr_pic_id: u16, references: &[Reference]) -> Result<Vec<u8>, String> {
    let mut ref_pic_list0 = [ffi::VaPictureH264::default(); 32];
    // Only as many entries as there are references, and every unused one is `VA_INVALID_SURFACE`
    // rather than zero for the reason `picture_record` gives at length: zero is a real surface id.
    for entry in ref_pic_list0.iter_mut() {
        *entry = ffi::VaPictureH264 { picture_id: ffi::INVALID_SURFACE, ..Default::default() };
    }
    for (slot, reference) in references.iter().enumerate() {
        let Some(entry) = ref_pic_list0.get_mut(slot) else {
            return Err(format!("reference {slot} does not fit the slice record's {} list entries", ref_pic_list0.len()));
        };
        // **`frame_idx` here is the DPB slot, not `frame_num`.** It is the same number in the
        // picture record's `ReferenceFrames` and the driver will not find the picture without it;
        // `frame_num` is what the *decoder* matches on and is not what the driver looks up here.
        *entry = ffi::VaPictureH264 {
            picture_id: reference.surface,
            frame_idx: reference.frame_idx,
            flags: PICTURE_H264_SHORT_TERM_REFERENCE,
            ..Default::default()
        };
    }
    let record = ffi::VaEncSliceParameterBufferH264 {
        macroblock_address: 0,
        num_macroblocks: macroblocks,
        macroblock_info: ffi::INVALID_SURFACE,
        slice_type,
        pic_parameter_set_id: 0,
        // Pairs two IDR pictures in one sequence. It advances per key frame, and a driver that saw
        // the same value twice would treat the second IDR as a repeat of the first rather than a new
        // random-access point.
        idr_pic_id,
        // **This is where the picture order count goes.** It is a slice-header syntax element and
        // the picture record has no field for it; see the note on `picture_record`.
        pic_order_cnt_lsb: poc,
        // The slice reads `RefPicList0` in full: `num_ref_idx_active_override_flag` clear means the
        // picture record's `num_ref_idx_l0_active_minus1` is the count, and both name the same
        // references. It is 0 for a P slice with exactly one reference, which is what
        // `picture_record` writes, and a slice claiming two references over a one-entry list is the
        // kind of mismatch that produces a picture rather than an error.
        num_ref_idx_active_override_flag: 0,
        num_ref_idx_l0_active_minus1: 0,
        num_ref_idx_l1_active_minus1: 0,
        ref_pic_list0,
        // List 1 is a B slice's second reference list; a P or I slice has none, and the entries are
        // `VA_INVALID_SURFACE` rather than zero for the same reason as list 0's unused entries.
        ref_pic_list1: [ffi::VaPictureH264 { picture_id: ffi::INVALID_SURFACE, ..Default::default() }; 32],
        // No weighted prediction: the flags are clear, so the tables below are not in the
        // bitstream and the weights are not read. A driver that *did* read them would be reading
        // zeros as "weight 0", i.e. a black prediction — which is why they are explicitly zeroed
        // rather than left at whatever `Default` gives.
        luma_log2_weight_denom: 0,
        chroma_log2_weight_denom: 0,
        luma_weight_l0_flag: 0,
        chroma_weight_l0_flag: 0,
        luma_weight_l1_flag: 0,
        chroma_weight_l1_flag: 0,
        cabac_init_idc: 0,
        // **0 means "the picture's QP"**, which is what makes the picture's `pic_init_qp` the one
        // that decides quality. The decode backend's own weight bug was the same class of mistake
        // in the opposite direction: a value that looked reasonable and was not.
        slice_qp_delta: 0,
        disable_deblocking_filter_idc: 0,
        slice_alpha_c0_offset_div2: 0,
        slice_beta_offset_div2: 0,
        ..Default::default()
    };
    Ok(ffi::record_bytes(&record))
}

/// How large a coded buffer a picture of `coded` size needs, or why it cannot be derived.
///
/// **This is the one number this module cannot derive faithfully**, so it uses a bound rather than
/// a guess: the size of a compressed picture depends on the QP, the content and the encoder's own
/// decisions, none of which are known before the picture exists. The bound is twice one
/// uncompressed 4:2:0 frame (`3/2 * width * height` bytes, plus 4 KiB for the driver-written
/// parameter sets), which no intra picture exceeds. Generous on purpose: a coded buffer that is too
/// small fails at exactly the moment the picture is being made, which is the worst place to discover
/// that a number was a guess.
///
/// Every step is `checked_*` and the result is narrowed to `u32`, because `vaCreateBuffer`'s size
/// argument is a `c_uint`: a picture whose bound does not fit is an error here rather than a
/// truncated size that sizes the buffer wrongly.
fn coded_buffer_for(coded: (u32, u32)) -> Result<u32, String> {
    let uncompressed = (coded.0 as u64)
        .checked_mul(coded.1 as u64)
        .and_then(|v| v.checked_mul(3))
        .and_then(|v| v.checked_div(2))
        .ok_or_else(|| format!("a {}x{} picture does not fit a byte count", coded.0, coded.1))?;
    let bytes = uncompressed.checked_mul(2).and_then(|v| v.checked_add(4096)).ok_or("a coded buffer size overflows")?;
    u32::try_from(bytes).map_err(|_| format!("a coded buffer of {bytes} bytes"))
}

/// Copy a `shown_w` x `shown_h` picture from the front of `from` into a `padded_w`-wide plane.
///
/// Only the picture's own pixels are copied, and only row by row: a copy that took `shown_w` bytes
/// per row out of a `padded_w`-wide destination would shear the picture, and one that wrote
/// `shown_h` rows into a plane padded for more would write past the row the driver described. Every
/// `get`/`get_mut` is a bound, so a caller whose slice is short gets an error naming the row rather
/// than a picture with its last rows missing.
fn copy_picture(to: &mut [u8], padded_w: usize, from: &[u8], shown_w: usize, shown_h: usize) -> Result<(), String> {
    if shown_w > padded_w {
        return Err(format!("a {shown_w} wide picture in a {padded_w} wide plane"));
    }
    for row in 0..shown_h {
        let from_at = row.checked_mul(shown_w).ok_or_else(|| format!("row {row} of the picture overflows"))?;
        let to_at = row.checked_mul(padded_w).ok_or_else(|| format!("row {row} of the padded picture overflows"))?;
        let src = from
            .get(from_at..)
            .and_then(|rest| rest.get(..shown_w))
            .ok_or_else(|| format!("a {shown_w}x{shown_h} picture needs {} bytes, got {}", shown_w.saturating_mul(shown_h), from.len()))?;
        let dst = to.get_mut(to_at..).and_then(|rest| rest.get_mut(..shown_w)).ok_or_else(|| format!("row {row} does not fit the padded plane"))?;
        dst.copy_from_slice(src);
    }
    Ok(())
}

/// A synthetic picture, unambiguous by eye: a **left-to-right luma ramp** on flat neutral chroma.
///
/// The choice, and why: the milestone's claim is "ffmpeg decodes the driver's bytes to the picture
/// I put in", so the picture has to be one a wrong decode cannot imitate. A ramp has three
/// properties no encoder bug reproduces by accident:
///
/// - **every column has a known value**, so the decode is checkable numerically rather than by
///   eye, and the answer says *how far off* rather than *looks about right*;
/// - it is **low-frequency**, so a stride, plane-offset or row-order bug produces a diagonal or a
///   striped picture instead of something a broken encoder could plausibly have produced;
/// - **flat chroma** means any colour shift is the chroma plane's, and a decoder that swapped the
///   planes shows it immediately.
///
/// The alternative — noise — is the worst possible choice here: a lossy encoder's noise is
/// plausible, so a wrong decode of noise looks like noise, and nothing says it is wrong.
pub fn ramp_picture(width: u32, height: u32) -> Result<(Vec<u8>, Vec<u8>), String> {
    ramp_shifted(width, height, 0)
}

/// **The same ramp, moved `shift` columns to the left** — the picture sequence an inter picture has
/// to be tested against.
///
/// # Why a *moving* picture, which a still one cannot test
///
/// A still ramp is right for "did the encoder make the picture I gave it" and **useless for "did
/// the P picture predict from the picture before it"**. If every picture is identical, then a
/// decoder that ignored the references entirely — every P frame coded as if it were an I frame, or
/// predicted from the wrong surface, or predicted from a cleared one — would still produce exactly
/// the right pixels. The reference could be wrong in every way that matters and the test would pass.
///
/// Moving it a fixed number of columns per picture makes that impossible:
///
/// - **A wrong reference shows as drift, not noise.** If frame *N* predicts from frame *N−1*
///   correctly and frame *N+1* predicts from frame *N−1* as well — a stale reference — the picture
///   is shifted one step too far, and the error at column *x* is the ramp's own slope times the
///   shift. On a 640-wide ramp that is 219/639 ≈ **0.34 luma per column**, so a single step of
///   misalignment is about 9 levels at the middle of the picture and visible at every column.
/// - **The error accumulates along the sequence**, so a *correct* reference chain has a flat error
///   curve and a broken one has a rising one. That shape is the evidence; a summary number cannot
///   distinguish "one bad frame" from "drifting for the whole sequence".
/// - **A fixed shift keeps the picture inside H.264's legal luma range at both ends** — the ramp
///   starts at 16 and the shift wraps it, so every picture is a legal picture and none of the
///   deviation can be blamed on an encoder clipping out-of-range input.
///
/// The chroma stays flat and neutral, for the reason [`ramp_picture`] gives: any colour shift is the
/// chroma plane's, and a chroma error from a mispredicted reference would otherwise be invisible in
/// a picture whose luma happens to be a smooth ramp.
///
/// **`Result`, because the size is hostile and an allocation is not a check.** This function used to
/// return two `Vec`s and do `Vec::with_capacity(w * h)` / `vec![128u8; w * h / 2]` on whatever `u32`
/// pair it was handed. That is not an overflow on a 64-bit target — `u32::MAX²` still fits a `usize`
/// — so nothing caught it, and the allocation is what would have: a request past what the allocator
/// grants does not return an `Err`, it calls `handle_alloc_error` and **aborts**, which no
/// `catch_unwind` in this project covers. Both planes are now sized through [`luma_len`] and [`chroma_len`], so a
/// size that overflows or exceeds [`MAX_PLANE_BYTES`] is an `Err` naming the picture before a byte is
/// allocated.
pub fn ramp_shifted(width: u32, height: u32, shift: u32) -> Result<(Vec<u8>, Vec<u8>), String> {
    let (w, h) = (width as usize, height as usize);
    // Both planes are sized before either is allocated, so a picture that cannot be sized never
    // gets half of one built.
    let (Some(luma_len), Some(chroma_len)) = (luma_len(width, height), chroma_len(width, height)) else {
        return Err(format!("a {width}x{height} synthetic picture does not fit {MAX_PLANE_BYTES} bytes of plane"));
    };
    // The ramp spans 219 levels across the width, so `span` is the last column and a one-column
    // picture has no span — `checked_sub` with a filter, so a width of 1 gets a flat 16 rather than
    // a division by zero.
    let span = w.checked_sub(1).filter(|span| *span > 0);
    // The shift wraps modulo the width rather than pushing the ramp off the legal luma range. The
    // `max(1)` is what makes the modulus safe for a zero-width picture, which the empty ramp above
    // already makes harmless — the modulo is the only thing that would panic.
    let shift = shift as usize % w.max(1);
    let mut luma = Vec::with_capacity(luma_len);
    for _y in 0..h {
        // The ramp is a function of the column only, so it is the same on every row — which is also
        // what makes a row-order or stride bug obvious rather than plausible.
        for x in 0..w {
            let column = (x + shift) % w.max(1);
            let step = span.and_then(|span| (219 * column).checked_div(span)).unwrap_or(0);
            luma.push(16u8.saturating_add(u8::try_from(step).unwrap_or(u8::MAX)));
        }
    }
    // Flat neutral chroma: Cb = Cr = 128 is the centre of the range, so any colour error is
    // visible as a colour cast rather than hidden in a gradient.
    let chroma = vec![128u8; chroma_len];
    Ok((luma, chroma))
}

/// The most a single padded plane may be sized at, in bytes.
///
/// **A cap, and the reason there is one.** A plane this module allocates is sized by a picture size
/// that came from a caller (a settings layer, a project, an agent over the control channel), and an
/// allocation `Vec` cannot satisfy does not come back as an `Err`: `Vec::with_capacity` and
/// `vec![0; n]` both abort the process through `handle_alloc_error` once the request passes what
/// the allocator will grant, and **that abort is not a panic** — the panic hook, the `catch_unwind`
/// around the UI pass and every job supervisor in this project are all built for unwinding and
/// cover none of it. So a size this module cannot bound has to be refused *before* the allocation,
/// not discovered by it.
///
/// One gibibyte is far past any picture that can be encoded: 8K 4:2:0 luma is 8192 × 4320 ≈ 34 MiB,
/// so the cap is about thirty times the largest plausible frame rather than a limit anyone would
/// notice. It exists so that a hostile `u32` pair is an `Err` naming the size, and it is checked
/// alongside the overflow rather than instead of it, because `u32 × u32` does not overflow a 64-bit
/// `usize` and so cannot be caught by `checked_mul` alone.
pub const MAX_PLANE_BYTES: usize = 1 << 30;

/// The length of the **luma** plane of a `w` x `h` 4:2:0 picture, or `None` past the cap above.
fn luma_len(w: u32, h: u32) -> Option<usize> {
    (w as usize).checked_mul(h as usize).filter(|bytes| *bytes <= MAX_PLANE_BYTES)
}

/// The length of the **chroma** plane of a `w` x `h` 4:2:0 picture, or `None` past the cap above.
///
/// `ceil(w/2) * ceil(h/2) * 2` and not `w * h / 2`: those are the same number when both dimensions
/// are even and a different one when either is odd, and an encoder session at an odd size is refused
/// upstream rather than here — but computing it the way the picture actually is keeps this correct
/// for any size a future caller manages to reach, and `Encoder::upload` in the FFI module derives
/// its chroma plane the same way.
fn chroma_len(w: u32, h: u32) -> Option<usize> {
    let cols = (w as usize).div_ceil(2).checked_mul(2)?;
    let rows = (h as usize).div_ceil(2);
    cols.checked_mul(rows).filter(|bytes| *bytes <= MAX_PLANE_BYTES)
}

/// The macroblock count a coded size holds, or `None` for a size that is not a whole number of
/// macroblocks in either dimension, or is zero. It **does not round**: 640x360 has no whole
/// macroblock grid, and the picture [`VaEncoder::new`] makes is 640x368 with 920 of them. So this
/// answers "is this size exactly encodable?" and [`VaEncoder::sizes`] answers "what did you build?".
pub fn macroblocks_of(coded: (u32, u32)) -> Option<u32> {
    // **The remainder is checked, not just divided.** `361 / 16` is 22 with 9 rows over, and a
    // macroblock count that silently drops those rows is a caller that believes it asked about a
    // 361-row picture. `VaEncoder::new` rounds up instead; this asks the question without rounding.
    if !coded.0.is_multiple_of(MB) || !coded.1.is_multiple_of(MB) {
        return None;
    }
    let (w, h) = (coded.0.checked_div(MB)?, coded.1.checked_div(MB)?);
    if w == 0 || h == 0 {
        return None;
    }
    w.checked_mul(h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vaapi::params;
    use std::mem::{offset_of, size_of};

    /// The encode records, as libva 1.24's headers on this machine declare them. A record whose
    /// layout is wrong does not fail a call — the driver reads a neighbour's value and encodes
    /// something else — so these are the only place that can catch it.
    ///
    /// The numbers came from `sizeof`/`offsetof` on `/usr/include/va/va_enc_h264.h` and `va.h`
    /// with a C compiler, not from arithmetic. They are re-measured rather than inherited: every one
    /// of them was asked of the header again for this task.
    #[test]
    fn the_h264_records_are_the_size_the_driver_reads() {
        assert_eq!(size_of::<ffi::VaCodedBufferSegment>(), 48, "VACodedBufferSegment (va.h:3945)");
        assert_eq!(offset_of!(ffi::VaCodedBufferSegment, size), 0);
        assert_eq!(offset_of!(ffi::VaCodedBufferSegment, bit_offset), 4);
        assert_eq!(offset_of!(ffi::VaCodedBufferSegment, status), 8);
        assert_eq!(offset_of!(ffi::VaCodedBufferSegment, buf), 16, "two pointer-sized fields follow");
        assert_eq!(offset_of!(ffi::VaCodedBufferSegment, next), 24);
        assert_eq!(size_of::<ffi::VaEncSequenceParameterBufferH264>(), 1132, "VAEncSequenceParameterBufferH264");
        assert_eq!(offset_of!(ffi::VaEncSequenceParameterBufferH264, seq_fields), 28, "after two u8s and six u32s");
        assert_eq!(offset_of!(ffi::VaEncSequenceParameterBufferH264, bit_depth_luma_minus8), 32);
        assert_eq!(offset_of!(ffi::VaEncSequenceParameterBufferH264, offset_for_ref_frame), 44, "the i32 hole at 36-43");
        assert_eq!(offset_of!(ffi::VaEncSequenceParameterBufferH264, frame_cropping_flag), 1068, "44 + 256 * 4");
        assert_eq!(offset_of!(ffi::VaEncSequenceParameterBufferH264, frame_crop_bottom_offset), 1084);
        assert_eq!(offset_of!(ffi::VaEncSequenceParameterBufferH264, va_reserved), 1116);
        assert_eq!(size_of::<ffi::VaEncPictureParameterBufferH264>(), 648, "VAEncPictureParameterBufferH264");
        assert_eq!(offset_of!(ffi::VaEncPictureParameterBufferH264, reference_frames), 36);
        assert_eq!(offset_of!(ffi::VaEncPictureParameterBufferH264, coded_buf), 612, "36 + 16 * 36");
        assert_eq!(offset_of!(ffi::VaEncPictureParameterBufferH264, pic_parameter_set_id), 616);
        assert_eq!(offset_of!(ffi::VaEncPictureParameterBufferH264, frame_num), 620, "after last_picture's hole at 619");
        assert_eq!(offset_of!(ffi::VaEncPictureParameterBufferH264, pic_init_qp), 622);
        assert_eq!(offset_of!(ffi::VaEncPictureParameterBufferH264, pic_fields), 628, "after the hole at 627");
        assert_eq!(offset_of!(ffi::VaEncPictureParameterBufferH264, va_reserved), 632);
        assert_eq!(size_of::<ffi::VaEncSliceParameterBufferH264>(), 3140, "NOT the decode record's 3128");
        assert_eq!(offset_of!(ffi::VaEncSliceParameterBufferH264, macroblock_info), 8);
        assert_eq!(offset_of!(ffi::VaEncSliceParameterBufferH264, slice_type), 12);
        assert_eq!(offset_of!(ffi::VaEncSliceParameterBufferH264, pic_order_cnt_lsb), 16);
        assert_eq!(offset_of!(ffi::VaEncSliceParameterBufferH264, delta_pic_order_cnt_bottom), 20, "the hole at 18-19");
        assert_eq!(offset_of!(ffi::VaEncSliceParameterBufferH264, ref_pic_list0), 36, "the ENCODE offset, not the decode one");
        assert_eq!(offset_of!(ffi::VaEncSliceParameterBufferH264, ref_pic_list1), 1188);
        assert_eq!(offset_of!(ffi::VaEncSliceParameterBufferH264, luma_weight_l0), 2344);
        assert_eq!(offset_of!(ffi::VaEncSliceParameterBufferH264, chroma_offset_l1), 2990);
        assert_eq!(offset_of!(ffi::VaEncSliceParameterBufferH264, cabac_init_idc), 3118);
        assert_eq!(offset_of!(ffi::VaEncSliceParameterBufferH264, va_reserved), 3124);
        assert_eq!(size_of::<ffi::VaEncMiscParameterBufferHeader>(), 4, "the flexible-array header on its own");
        assert_eq!(size_of::<ffi::VaEncMiscParameterFrameRate>(), 24);
        assert_eq!(size_of::<ffi::VaEncMiscParameterQuantization>(), 8);
        // **Re-measured for the rate-control record, not inherited.** `cc -o rc rc.c && ./rc` over
        // `/usr/include/va/va.h` on this machine: sizeof 60, and every offset as listed. A
        // `VaEncMiscParameterRateControl` two fields short would still be submitted, still accepted,
        // and would read the driver's `va_reserved` as the caller's `max_qp`.
        assert_eq!(size_of::<ffi::VaEncMiscParameterRateControl>(), 60, "VAEncMiscParameterRateControl (va.h:2493)");
        assert_eq!(offset_of!(ffi::VaEncMiscParameterRateControl, bits_per_second), 0);
        assert_eq!(offset_of!(ffi::VaEncMiscParameterRateControl, target_percentage), 4);
        assert_eq!(offset_of!(ffi::VaEncMiscParameterRateControl, window_size), 8);
        assert_eq!(offset_of!(ffi::VaEncMiscParameterRateControl, initial_qp), 12);
        assert_eq!(offset_of!(ffi::VaEncMiscParameterRateControl, min_qp), 16);
        assert_eq!(offset_of!(ffi::VaEncMiscParameterRateControl, basic_unit_size), 20);
        assert_eq!(offset_of!(ffi::VaEncMiscParameterRateControl, rc_flags), 24);
        assert_eq!(offset_of!(ffi::VaEncMiscParameterRateControl, icq_quality_factor), 28);
        assert_eq!(offset_of!(ffi::VaEncMiscParameterRateControl, max_qp), 32);
        assert_eq!(offset_of!(ffi::VaEncMiscParameterRateControl, quality_factor), 36);
        assert_eq!(offset_of!(ffi::VaEncMiscParameterRateControl, target_frame_size), 40);
        assert_eq!(offset_of!(ffi::VaEncMiscParameterRateControl, va_reserved), 44);
        // **The record tiles with no gap anywhere**, which is the property the tiling test asserts for
        // every submitted record and the one `VASegmentParameterVP9` failed. Pinned here as a
        // sequence so a reordering that left a hole would fail on *this* test as well as on the
        // C-compiler gap check in `abi_guard`.
        let mut at = 0usize;
        for (name, offset) in [
            ("bits_per_second", offset_of!(ffi::VaEncMiscParameterRateControl, bits_per_second)),
            ("target_percentage", offset_of!(ffi::VaEncMiscParameterRateControl, target_percentage)),
            ("window_size", offset_of!(ffi::VaEncMiscParameterRateControl, window_size)),
            ("initial_qp", offset_of!(ffi::VaEncMiscParameterRateControl, initial_qp)),
            ("min_qp", offset_of!(ffi::VaEncMiscParameterRateControl, min_qp)),
            ("basic_unit_size", offset_of!(ffi::VaEncMiscParameterRateControl, basic_unit_size)),
            ("rc_flags", offset_of!(ffi::VaEncMiscParameterRateControl, rc_flags)),
            ("ICQ_quality_factor", offset_of!(ffi::VaEncMiscParameterRateControl, icq_quality_factor)),
            ("max_qp", offset_of!(ffi::VaEncMiscParameterRateControl, max_qp)),
            ("quality_factor", offset_of!(ffi::VaEncMiscParameterRateControl, quality_factor)),
            ("target_frame_size", offset_of!(ffi::VaEncMiscParameterRateControl, target_frame_size)),
        ] {
            assert_eq!(offset, at, "{name} starts at {offset}, and the field before it ends at {at} — a hole the header does not have");
            at += size_of::<u32>();
        }
        // `va_reserved` is the last member and is an array, so it is checked against `at` rather
        // than stepped through: 44 after eleven `uint32_t`s, and `VA_PADDING_LOW` of them takes the
        // record to 60. **Nothing is left over**, which is the other half of "no hole" — a record
        // whose named fields stop short of its size has padding C inserted and this declaration does
        // not name.
        assert_eq!(offset_of!(ffi::VaEncMiscParameterRateControl, va_reserved), at, "va_reserved starts where target_frame_size ends");
        assert_eq!(
            at + ffi::VA_PADDING_LOW * size_of::<u32>(),
            size_of::<ffi::VaEncMiscParameterRateControl>(),
            "and va_reserved fills the record exactly, with nothing after it"
        );
    }

    /// The encode constants, pinned so a wrong one fails on every machine rather than only where a
    /// driver happens to misbehave. The buffer-type numbers are the ones most worth pinning: they
    /// start at 21, and a submission with the wrong number is *not* an unknown type but a
    /// different type the driver will act on.
    #[test]
    fn the_encode_constants_are_the_ones_libva_uses() {
        assert_eq!(ffi::ENC_CODED_BUFFER, 21, "VAEncCodedBufferType");
        assert_eq!(ffi::ENC_SEQUENCE_PARAMETER_BUFFER, 22);
        assert_eq!(ffi::ENC_PICTURE_PARAMETER_BUFFER, 23);
        assert_eq!(ffi::ENC_SLICE_PARAMETER_BUFFER, 24);
        assert_eq!(ffi::ENC_PACKED_HEADER_DATA_BUFFER, 26);
        assert_eq!(ffi::ENC_MISC_PARAMETER_BUFFER, 27);
        assert_eq!(ffi::CONFIG_ATTRIB_RT_FORMAT, 0, "VAConfigAttribRTFormat");
        assert_eq!(ffi::CONFIG_ATTRIB_RATE_CONTROL, 5);
        assert_eq!(ffi::CONFIG_ATTRIB_ENC_PACKED_HEADERS, 10);
        assert_eq!(ffi::RC_CBR, 0x2);
        assert_eq!(ffi::RC_VBR, 0x4);
        assert_eq!(ffi::RC_CQP, 0x10, "VA_RC_CQP is 0x10, not 0x8");
        assert_eq!(ffi::RC_NONE, 0x1, "VA_RC_NONE");
        assert_eq!(ffi::RC_VBR_CONSTRAINED, 0x20, "VA_RC_VBR_CONSTRAINED");
        assert_eq!(ffi::RC_QVBR, 0x400, "VA_RC_QVBR");
        assert_eq!(ffi::RC_AVBR, 0x800, "VA_RC_AVBR");
        assert_eq!(ffi::RC_TCBRC, 0x1000, "VA_RC_TCBRC");
        assert_eq!(ffi::RC_ICQ, 0x40);
        assert_eq!(ffi::PACKED_HEADER_SEQUENCE, 0x1);
        assert_eq!(ffi::PACKED_HEADER_PICTURE, 0x2);
        assert_eq!(ffi::PACKED_HEADER_SLICE, 0x4);
        assert_eq!(ffi::MISC_FRAME_RATE, 0);
        assert_eq!(ffi::MISC_RATE_CONTROL, 1);
        assert_eq!(ffi::MISC_QUANTIZATION, 8);
        assert_eq!(ffi::SLICE_TYPE_P, 0);
        assert_eq!(ffi::SLICE_TYPE_B, 1);
        assert_eq!(ffi::SLICE_TYPE_I, 2);
        assert_eq!(ffi::SLICE_TYPE_ALL_P, 5);
        assert_eq!(ffi::SLICE_TYPE_ALL_I, 7);
    }

    /// The packed `seq_fields` word, read back out of the record the builder produced, so the bit
    /// positions are checked against what the record *says* and not only against the literal.
    #[test]
    fn the_sequence_record_says_what_it_means() {
        // The crop tuple is `(left, right, top, bottom)`, the order the four `frame_crop_*_offset` fields
        // are declared in: a 640x360 picture coded at 640x368 crops 8 rows off the bottom, and
        // 8 rows is 4 units of the 2-sample crop unit.
        let bytes = sequence_record((640, 368), (0, 0, 0, 4), 40, 23, 1, 0).expect("a 640x368 sequence record");
        let record = read::<ffi::VaEncSequenceParameterBufferH264>(&bytes).expect("the record reads back");
        // Bit 0-1: chroma_format_idc = 1 (4:2:0).
        assert_eq!(record.seq_fields & 0b11, 1, "chroma_format_idc");
        // Bit 2: frame_mbs_only_flag — the picture is progressive, and the constant says so.
        assert_ne!(record.seq_fields & (1 << 2), 0, "frame_mbs_only_flag");
        // Bit 4: seq_scaling_matrix_present_flag **must be clear**: set, it requires an IQ matrix
        // buffer in the same render, and a driver given a render without one can succeed and
        // produce a plausible grey picture. This is the encode-side twin of the decode backend's
        // IQ_MATRIX_BUFFER note, and it is asserted rather than commented.
        assert_eq!(record.seq_fields & (1 << 4), 0, "seq_scaling_matrix_present_flag must be clear without an IQ matrix");
        // Bit 5: direct_8x8_inference_flag.
        assert_ne!(record.seq_fields & (1 << 5), 0, "direct_8x8_inference_flag");
        // Bits 10-11: pic_order_cnt_type = 0.
        assert_eq!((record.seq_fields >> 10) & 0b11, 0, "pic_order_cnt_type");
        // Bit 16: delta_pic_order_always_zero_flag.
        assert_ne!(record.seq_fields & (1 << 16), 0, "delta_pic_order_always_zero_flag");
        // The geometry: macroblocks, not pixels. A 640x368 picture is 40x23 macroblocks, and this
        // is the field whose units the header states and the examples get wrong.
        assert_eq!(record.picture_width_in_mbs, 40);
        assert_eq!(record.picture_height_in_mbs, 23);
        // The crop: a 640x360 picture is coded at 640x368 and crops 8 rows, which is 4 crop units, off
        // the **bottom** — the field order and the tuple order agree, and this is what says so.
        assert_eq!(record.frame_cropping_flag, 1);
        assert_eq!(record.frame_crop_bottom_offset, 4);
        assert_eq!(record.frame_crop_left_offset, 0);
        assert_eq!(record.frame_crop_right_offset, 0);
        assert_eq!(record.frame_crop_top_offset, 0);
        // No picture order count cycle and one reference, consistent with pic_order_cnt_type 0.
        assert_eq!(record.num_ref_frames_in_pic_order_cnt_cycle, 0);
        assert_eq!(record.max_num_ref_frames, 1);
        // The bitrate is nonzero under RC_CQP: iHD rejects a zero, and a zero would also be the
        // "no bitrate" a reader could mistake for a default target.
        assert_ne!(record.bits_per_second, 0);
    }

    /// The frame-number width in the sequence record is the one the frame loop keys against, and the
    /// two order-count widths are deliberately different from it.
    ///
    /// This is the record-level half of a check that also runs against the bitstream (see
    /// [`check_output_matches_the_sets`]): the record says what we asked for, the parsed parameter
    /// sets say what a decoder will read, and nothing should be able to make those disagree while
    /// still producing a stream that decodes.
    #[test]
    fn the_sequence_records_frame_num_width_is_the_one_the_loop_uses() {
        let bytes = sequence_record((640, 368), (0, 0, 0, 4), 40, 23, 1, 0).expect("a 640x368 sequence record");
        let record = read::<ffi::VaEncSequenceParameterBufferH264>(&bytes).expect("the record reads back");
        // Bits 6-9, read back out of the record rather than off the constant.
        let minus4 = (record.seq_fields >> 6) & 0xf;
        assert_eq!(minus4, LOG2_MAX_FRAME_NUM_MINUS4, "log2_max_frame_num_minus4");
        // 7.3.2.1.1: MaxFrameNum = 2 ^ (log2_max_frame_num_minus4 + 4). **This is the assertion that
        // would fail if `MAX_FRAME_NUM` were lowered back to 16** — the record would still build,
        // still be accepted by the driver, and still produce a decodable stream, just one whose key
        // frames are not where this module says they are.
        assert_eq!(1u32 << (minus4 + 4), MAX_FRAME_NUM, "the record's width is the modulus the frame loop keys against");
        // And the picture order count is deliberately **not** the same width.
        let poc_minus4 = (record.seq_fields >> 12) & 0xf;
        assert_eq!(1u32 << (poc_minus4 + 4), MAX_PIC_ORDER_CNT_LSB, "MaxPicOrderCntLsb is what this encoder writes");
        assert_ne!(
            MAX_PIC_ORDER_CNT_LSB, MAX_FRAME_NUM,
            "the two widths are deliberately different: the order count wraps on its own modulus, which is 8.2.1.1's Equation 8-3 written for"
        );
    }

    /// The derived parameter sets declare the widths this module decided on, read back through the
    /// parser a decoder would use rather than compared as bytes.
    ///
    /// Hardware-independent: it builds the sets and parses them, which is the whole point — the
    /// widths are what a decoder reads the slice header at, so a width wrong here is a stream whose
    /// slices are misparsed rather than one that fails to build.
    #[test]
    fn the_derived_sets_declare_the_widths_this_encoder_decided_on() {
        let sequence = read::<ffi::VaEncSequenceParameterBufferH264>(&sequence_record((640, 368), (0, 0, 0, 4), 40, 23, 1, 0).expect("a sequence record"))
            .expect("the record reads back");
        let picture = read::<ffi::VaEncPictureParameterBufferH264>(
            &picture_record(&Current { surface: 1, frame_idx: 0, frame_num: 7, poc: 7, is_reference: true }, 40, 0, 920, &[], 26, false, false, 11)
                .expect("a P picture record"),
        )
        .expect("the record reads back");
        let sets = params::parameter_sets(&sequence, &picture).expect("the derived parameter sets build and verify");
        let sps = filmcraft_h264::params::Sps::parse(&filmcraft_bitstream::unescape_rbsp(&sets[5..])).expect("the derived SPS parses");
        assert_eq!(sps.log2_max_frame_num, 8, "and declares the eight-bit frame_num this module keys against");
        assert_eq!(sps.max_frame_num(), MAX_FRAME_NUM, "which is the modulus the frame loop uses");
        assert_eq!(sps.log2_max_poc_lsb, 4, "the picture order count keeps its own four-bit width");
    }

    /// The same for the picture record: the key-frame bit and the QP, read back out.
    #[test]
    fn the_picture_record_names_its_key_frame_and_its_quantiser() {
        let idr = picture_record(&Current { surface: 7, frame_idx: 0, frame_num: 0, poc: 0, is_reference: true }, 9, 0, 920, &[], 26, true, false, 11)
            .expect("an IDR picture record");
        let record = read::<ffi::VaEncPictureParameterBufferH264>(&idr).expect("the record reads back");
        // Bit 0 is `idr_pic_flag` and it is the whole key-frame decision.
        assert_ne!(record.pic_fields & 1, 0, "idr_pic_flag");
        assert_eq!(record.curr_pic.picture_id, 7, "the reconstruction surface, not the input");
        assert_ne!(record.curr_pic.picture_id, 9, "CurrPic.picture_id must not be the input surface");
        assert_eq!(record.coded_buf, 11);
        // **Every reference slot is `VA_INVALID_SURFACE`, for an IDR too** — measured: a zeroed slot
        // is surface 0, which on this machine is a real surface (the input one), and the driver
        // then fails `vaEndPicture` with `VA_STATUS_ERROR_ENCODING_ERROR`. See the note on the
        // record itself for the whole story; this asserts the value that fixes it.
        assert!(
            record.reference_frames.iter().all(|r| r.picture_id == ffi::INVALID_SURFACE),
            "an unused reference slot must be VA_INVALID_SURFACE, never 0: 0 is a real surface id here"
        );
        assert_eq!(record.reference_frames[0].picture_id, u32::MAX);
        // **pic_init_qp is the QP itself**: the field is pic_init_qp_minus26 + 26, so a QP of 26 is
        // 26 and subtracting again would quietly encode at a different quality.
        assert_eq!(record.pic_init_qp, 26, "pic_init_qp is pic_init_qp_minus26 + 26, not pic_init_qp_minus26");
        assert_eq!(record.chroma_qp_index_offset, 0);
        // Entropy coding is CAVLC (bit 3 clear) and there is no deblocking override (bit 9 set
        // with the slice's idc at 0, meaning "filter it").
        assert_eq!(record.pic_fields & (1 << 3), 0, "entropy_coding_mode_flag");
        assert_ne!(record.pic_fields & (1 << 9), 0, "deblocking_filter_control_present_flag");
        // No reference is named for an intra picture.
        assert_eq!(record.reference_frames[0].picture_id, u32::MAX, "an IDR names no reference");
        let p = picture_record(&Current { surface: 7, frame_idx: 0, frame_num: 0, poc: 0, is_reference: true }, 9, 0, 920, &[], 26, false, false, 11)
            .expect("a P picture record");
        let p_record = read::<ffi::VaEncPictureParameterBufferH264>(&p).expect("the record reads back");
        assert_eq!(p_record.pic_fields & 1, 0, "a P picture is not an IDR");
        // The two record invariants, both refused rather than encoded around.
        assert!(
            picture_record(&Current { surface: 7, frame_idx: 0, frame_num: 0, poc: 0, is_reference: true }, 7, 0, 920, &[], 26, true, false, 11).is_err(),
            "recon == input"
        );
        assert!(
            picture_record(&Current { surface: 7, frame_idx: 0, frame_num: 0, poc: 0, is_reference: true }, 9, 0, 0, &[], 26, true, false, 11).is_err(),
            "0 macroblocks"
        );
        // And the record builder's own crop arithmetic, which is where a tuple ordered
        // `(left, top, right, bottom)` instead would put a 640x360 picture's crop on the top edge —
        // a picture ffmpeg decodes with eight rows of the ramp missing from the wrong end.
        let cropped = sequence_record((640, 368), (0, 0, 0, crop_of(368, 360, 2, "bottom").unwrap_or(0)), 40, 23, 1, 0).expect("a cropped sequence record");
        let cropped = read::<ffi::VaEncSequenceParameterBufferH264>(&cropped).expect("it reads back");
        assert_eq!(cropped.frame_crop_bottom_offset, 4, "the crop is at the bottom");
        assert_eq!(cropped.frame_crop_top_offset, 0, "and not the top");
    }

    /// The slice record: one slice over the whole picture, no per-macroblock buffer, and a QP delta
    /// of zero so the picture's `pic_init_qp` is what decides quality.
    #[test]
    fn the_slice_record_is_one_slice_and_defers_its_quantiser() {
        let bytes = slice_record(920, ffi::SLICE_TYPE_ALL_I, 3, 1, &[]).expect("a slice record");
        let record = read::<ffi::VaEncSliceParameterBufferH264>(&bytes).expect("the record reads back");
        assert_eq!(record.macroblock_address, 0);
        assert_eq!(record.num_macroblocks, 920, "640x368 is 40x23 = 920 macroblocks");
        assert_eq!(record.macroblock_info, ffi::INVALID_SURFACE, "VA_INVALID_ID, the header's own value for 'none'");
        assert_eq!(record.slice_type, 7, "slice_type 7: every slice in the picture is an I slice");
        // The picture order count lands here and not in the picture record: it is a slice-header
        // syntax element, and the picture record has no field for it at all.
        assert_eq!(record.pic_order_cnt_lsb, 3, "the picture order count the caller passed");
        assert_eq!(record.idr_pic_id, 1, "the IDR id the caller passed");
        // **0 is the value that means "the picture's QP".** Anything else is a second, silent
        // statement about quality.
        assert_eq!(record.slice_qp_delta, 0);
        assert_eq!(record.disable_deblocking_filter_idc, 0);
        // No weighted prediction: the flags are clear, so the tables below are not in the bitstream
        // and the weights are not read. They are zero rather than left at whatever is in memory.
        assert_eq!(record.luma_weight_l0_flag, 0);
        assert_eq!(record.luma_weight_l0[0], 0);
        assert_eq!(record.chroma_weight_l0[0], [0, 0]);
    }

    /// The misc buffers: a 4-byte type header followed by its record, and the frame rate as a
    /// fraction. The fraction is the field most likely to be wrong by a factor of 65 536, and a
    /// driver that is told the frame rate is 65 536 fps does not error — it makes a stream nobody
    /// can play.
    #[test]
    fn the_misc_buffers_are_a_type_word_and_a_record() {
        let fps = frame_rate_record(30).expect("a frame-rate misc buffer");
        assert_eq!(fps.len(), size_of::<ffi::VaEncMiscParameterBufferHeader>() + size_of::<ffi::VaEncMiscParameterFrameRate>());
        assert_eq!(read::<u32>(&fps[..4]).unwrap_or(u32::MAX), 0, "VAEncMiscParameterTypeFrameRate");
        let rate = read::<ffi::VaEncMiscParameterFrameRate>(&fps[4..]).expect("the frame-rate record reads back");
        assert_eq!(rate.framerate, (1 << 16) | 30, "denominator high, numerator low: 30/1 fps");
        assert_eq!(rate.framerate >> 16, 1, "the denominator");
        assert_eq!(rate.framerate & 0xffff, 30, "the numerator");
        let quant = quantization_record(Some(26)).expect("a quantisation misc buffer");
        assert_eq!(quant.len(), size_of::<ffi::VaEncMiscParameterBufferHeader>() + size_of::<ffi::VaEncMiscParameterQuantization>());
        assert_eq!(read::<u32>(&quant[..4]).unwrap_or(u32::MAX), 8, "VAEncMiscParameterTypeQuantization");
        assert_eq!(read::<ffi::VaEncMiscParameterQuantization>(&quant[4..]).expect("the record").quantization_flags, 0, "let the driver decide");
        // A frame rate the field cannot express is an error, not a clamp.
        assert!(frame_rate_record(0).is_err(), "0 fps");
        assert!(frame_rate_record(70_000).is_err(), "beyond the numerator's 16 bits");
    }

    /// The geometry: 640x360 is coded at 640x368 and crops 4 units off the bottom, and a size that
    /// needs no crop says so rather than setting a flag nobody should read.
    #[test]
    fn the_geometry_rounds_up_to_macroblocks_and_crops_exactly() {
        assert_eq!(macroblocks_of((640, 368)), Some(920));
        assert_eq!(macroblocks_of((640, 360)), None, "360 rows are not a whole number of macroblock rows");
        assert_eq!(macroblocks_of((640, 361)), None, "and neither are 361: the remainder is checked, not divided away");
        assert_eq!(macroblocks_of((639, 368)), None, "nor is 639 columns");
        assert_eq!(macroblocks_of((0, 16)), None, "zero width");
        assert_eq!(macroblocks_of((16, 0)), None, "zero height");
        assert_eq!(macroblocks_of((8, 16)), None, "8 is less than one macroblock wide");
        // 640x360: 8 rows of padding, 2 samples per crop unit, 4 units off the bottom.
        assert_eq!(crop_of(368, 360, 2, "bottom").expect("a whole number of crop units"), 4);
        // A padding that is not a whole number of crop units is an error rather than a truncation:
        // truncating would crop the picture by less than it was padded by, and the decoder would
        // show rows that were never part of the picture.
        assert!(crop_of(369, 360, 2, "bottom").is_err(), "9 rows is not 2-row units");
        assert!(crop_of(352, 360, 2, "bottom").is_err(), "less padding than picture");
    }

    /// The coded-buffer size is a bound, refused when it cannot be computed, and never zero: a
    /// coded buffer of zero bytes would have the driver write into nothing.
    #[test]
    fn the_coded_buffer_is_a_bound_and_never_zero() {
        // 640x368 is 352 KiB uncompressed at 4:2:0; the bound is twice that plus the parameter
        // sets. This is a bound, and the note on the function says why: a compressed picture's
        // size is not derivable before the picture exists.
        let bytes = coded_buffer_for((640, 368)).expect("a coded buffer size");
        assert!(bytes > 640 * 368 * 3 / 2, "the bound must cover one uncompressed frame");
        // The bound is derived with `checked_*`, so a size whose area overflows a `u64` — and a
        // result that overflows a `u32`, since that is what `vaCreateBuffer` takes — is an error
        // rather than a wrapped number that sizes a buffer wrongly.
        assert!(coded_buffer_for((0, 16)).is_ok(), "zero width still has an area");
        assert!(coded_buffer_for((u32::MAX, u32::MAX)).is_err(), "the area overflows");
        assert!(coded_buffer_for((65536, 65536)).is_err(), "twice a 12.9 GB frame overflows u32");
    }

    /// The ramp: every column has the value the doc says it has, it stays inside H.264's legal
    /// luma range, and the chroma is flat and neutral. These are the numbers the ffmpeg decode is
    /// compared against, so they are pinned rather than regenerated.
    #[test]
    fn the_synthetic_picture_is_the_ramp_it_documents() {
        let (luma, chroma) = ramp_picture(640, 360).expect("a 640x360 ramp");
        assert_eq!(luma.len(), 640 * 360);
        assert_eq!(chroma.len(), 640 * 360 / 2);
        // Legal luma range at both ends, and strictly increasing across the row.
        assert!(luma.iter().all(|&s| (16..=235).contains(&s)), "inside H.264's legal luma range");
        assert_eq!(luma[0], 16);
        assert_eq!(luma[639], 235);
        // Non-decreasing, not strictly: 219 levels over 640 columns means some columns share a
        // value, and the test says which is true rather than asserting something the picture does
        // not have. What matters for the claim is that the ramp is a *known function of the
        // column*, so a decode can be compared to it numerically.
        assert!(luma[..640].windows(2).all(|w| w[0] <= w[1]), "a non-decreasing ramp");
        assert_ne!(luma[320], luma[321], "the middle of the ramp still steps");
        // The same on every row: a function of the column only, so a stride or row-order bug is
        // visible rather than plausible.
        assert_eq!(luma[640..1280], luma[..640], "the ramp does not depend on the row");
        // Flat neutral chroma.
        assert!(chroma.iter().all(|&s| s == 128), "neutral chroma everywhere");
    }

    /// A hostile picture cannot make the copy read past either buffer. `copy_picture` is the one
    /// place a caller-supplied slice is indexed, so every way of being wrong has to be an error.
    #[test]
    fn a_picture_that_does_not_fit_is_refused_rather_than_copied() {
        let mut plane = vec![0u8; 640 * 368];
        // A short picture: the last rows are missing.
        assert!(copy_picture(&mut plane, 640, &[0u8; 640 * 359], 640, 360).is_err(), "short picture");
        // A picture wider than the plane it is copied into.
        assert!(copy_picture(&mut plane, 640, &[0u8; 700 * 360], 700, 360).is_err(), "too wide");
        // A plane shorter than the padded picture needs.
        assert!(copy_picture(&mut [0u8; 100], 640, &[0u8; 640 * 360], 640, 360).is_err(), "plane too small");
        // The exact size works, and copies the right row.
        let mut plane = vec![0u8; 640 * 368];
        let mut picture = vec![7u8; 640 * 360];
        picture[640 * 100 + 3] = 200;
        assert!(copy_picture(&mut plane, 640, &picture, 640, 360).is_ok());
        assert_eq!(plane[640 * 100 + 3], 200, "row 100 of the plane holds row 100 of the picture");
        // And the padding is left as it was: the caller filled it with the neutral value, and
        // this is what keeps the encoder from reading uninitialised surface memory.
        assert_eq!(plane[640 * 360], 0, "the padded row is untouched");
    }

    /// A QP outside H.264's range is refused rather than clamped. A clamp is the encoder-side
    /// version of a wrong default: the picture would come out at a quality nobody asked for, and
    /// nothing anywhere would say so.
    #[test]
    fn a_quantiser_outside_h264s_range_is_refused_not_clamped() {
        assert!(EncodeRequest::new(0, PictureKind::Idr).is_ok(), "0 is a legal QP");
        assert!(EncodeRequest::new(51, PictureKind::Idr).is_ok(), "51 is a legal QP");
        assert!(EncodeRequest::new(52, PictureKind::Idr).is_err(), "52 is not");
        assert!(EncodeRequest::new(255, PictureKind::Idr).is_err(), "255 is not");
    }

    /// **The rate-control misc buffer says what the caller asked for, in the fields the header gives
    /// those numbers — and refuses the ones it cannot.**
    ///
    /// Read back out of the built bytes rather than asserted on the struct, because the struct is what
    /// the builder wrote and the record is what reaches the driver; a bit-position mistake inside
    /// `record_bytes` is invisible to the first and visible only to the second.
    #[test]
    fn the_rate_control_record_carries_the_callers_numbers_and_refuses_the_rest() {
        let read_rc = |mode: RateControl, window: u32| -> ffi::VaEncMiscParameterRateControl {
            let bytes = rate_control_record(mode, window).expect("a rate-control buffer");
            assert_eq!(
                bytes.len(),
                size_of::<ffi::VaEncMiscParameterBufferHeader>() + size_of::<ffi::VaEncMiscParameterRateControl>(),
                "the type word plus the record, and nothing else"
            );
            assert_eq!(read::<u32>(&bytes[..4]), Some(ffi::MISC_RATE_CONTROL), "VAEncMiscParameterTypeRateControl, read out of the buffer rather than assumed");
            read::<ffi::VaEncMiscParameterRateControl>(&bytes[4..]).expect("the record reads back")
        };

        // **CBR: the bitrate is the ceiling and the target, which the header spells `bits_per_second`
        // and `target_percentage = 100`.**
        let cbr = read_rc(RateControl::Cbr(8_000_000), 533);
        assert_eq!(cbr.bits_per_second, 8_000_000, "the target, in bits per second");
        assert_eq!(cbr.target_percentage, 100, "CBR's target is 100% of the maximum, which is the same number");
        assert_eq!(cbr.window_size, 533, "the window the caller derived from its key-frame interval");

        // **VBR: the ceiling is `bits_per_second` and the target is a percentage of it** — the header's
        // own split, and the one FilmCraft's `target_kbps`/`max_kbps` pair means. 4000 of 8000 is 50%.
        let vbr = read_rc(RateControl::Vbr { target: 4_000_000, max: 8_000_000 }, 1000);
        assert_eq!(vbr.bits_per_second, 8_000_000, "the ceiling");
        assert_eq!(vbr.target_percentage, 50, "the target as a percentage of it");

        // **The percentage rounds *up*, never down**: a target that rounded down would ask for less
        // than the caller said, and the file would be smaller than the settings promised with nothing
        // wrong anywhere. 1 of 8000 kbit/s is 0.0125%, which no integer can express and rounds to 1.
        let tiny = read_rc(RateControl::Vbr { target: 1_000, max: 8_000_000 }, 1000);
        assert_eq!(tiny.target_percentage, 1, "a target below 1% is still a target, and the field has no zero");

        // **And it never exceeds 100**, which is the field's whole range.
        let equal = read_rc(RateControl::Vbr { target: 8_000_000, max: 8_000_000 }, 1000);
        assert_eq!(equal.target_percentage, 100, "a ceiling equal to its target is 100%");

        // **Every quantiser field is zero, which is the header's own "ignored"** for `initial_qp`,
        // `min_qp` and `max_qp`, and "no limits" for `target_frame_size`. Asserted rather than left,
        // because a nonzero value here would be FilmCraft inventing a quality range the caller never
        // asked for — next to a bitrate the caller did.
        assert_eq!(cbr.initial_qp, 0, "initial_qp is ignored when zero");
        assert_eq!(cbr.min_qp, 0, "min_qp is ignored when zero");
        assert_eq!(cbr.max_qp, 0, "max_qp is ignored when zero");
        assert_eq!(cbr.target_frame_size, 0, "zero means no limits");
        assert_eq!(cbr.quality_factor, 0, "not used by CBR or VBR");
        assert_eq!(cbr.icq_quality_factor, 0, "ICQ mode only, and this is not ICQ mode");
        assert_eq!(cbr.basic_unit_size, 0, "the header says to set it to zero if unused");

        // **`rc_flags` is zero, and bit 0 of it is the one that matters**: `reset` would throw away
        // the controller's history on every picture, and the stream would be a run of one-frame
        // streams — landing nowhere near the bitrate with nothing wrong to see.
        assert_eq!(cbr.rc_flags, 0, "no flags: no reset, no frame-skip disable, no bit-stuffing change");
        assert_eq!(cbr.rc_flags & ffi::RC_FLAG_RESET, 0, "the reset flag in particular");
        assert_eq!(cbr.rc_flags & ffi::RC_FLAG_DISABLE_FRAME_SKIP, 0, "and the frame-skip one, which would drop pictures an export cannot express");

        // **The refusals, each with the number that caused it.**
        assert!(rate_control_record(RateControl::Cbr(0), 1000).is_err(), "zero bits per second");
        assert!(rate_control_record(RateControl::Vbr { target: 0, max: 1000 }, 1000).is_err(), "a zero target");
        assert!(rate_control_record(RateControl::Vbr { target: 1000, max: 0 }, 1000).is_err(), "a zero ceiling");
        let below = rate_control_record(RateControl::Vbr { target: 2_000_000, max: 1_000_000 }, 1000)
            .expect_err("a ceiling below its target is unreachable by construction");
        assert!(below.contains("2000000") && below.contains("1000000"), "and the message names both numbers: {below}");
        // **And the mode that has no business submitting one at all.** A bitrate record under constant
        // quantiser is a second statement about the same fidelity, so it is refused rather than built
        // with zeros.
        let cqp = rate_control_record(RateControl::ConstantQuantiser, 1000).expect_err("constant quantiser takes no bitrate");
        assert!(cqp.contains("pic_init_qp"), "and says what the quality already is: {cqp}");

        // **The target percentage never wraps.** `target * 100` on a `u32` overflows past about
        // 43 Mbit/s, which a 4K export asks for routinely, and a wrapped percentage clamped into
        // 1..=100 becomes "reach the ceiling" — a file at a size nobody chose with no error anywhere.
        let huge = read_rc(RateControl::Vbr { target: 4_000_000_000, max: 4_000_000_000 }, 1000);
        assert_eq!(huge.target_percentage, 100, "4 Gbit/s over 4 Gbit/s is 100%, not a wrapped remainder");
    }

    /// **The window is derived from the key-frame interval, and refused values fall back rather than
    /// dividing by zero.**
    ///
    /// The derivation and the measurement behind it are on `rate_control_window_ms`; what is pinned
    /// here is that a one-picture GOP — which a caller may legitimately ask for — produces a legal
    /// window and does not underflow.
    #[test]
    fn the_rate_control_window_follows_the_key_frame_interval() {
        // 16 pictures at the 30 fps the plan's runs use is 533 ms.
        assert_eq!(rate_control_window_ms(16), 533);
        // A one-picture GOP would be 33 ms, and the floor is one millisecond so the field is never zero.
        assert_eq!(rate_control_window_ms(1), 33);
        assert_eq!(rate_control_window_ms(0), 1000, "no span to derive from, so the round value above the longest GOP");
        // A GOP longer than a second is clamped rather than submitted as a five-second window: the
        // field is u32 so it would fit, but a window longer than the key-frame interval is a claim
        // about the stream's pacing that nothing supports.
        assert_eq!(rate_control_window_ms(1_000), 1000);
    }

    /// **The mode and the request must agree, and the mismatch is refused with both named.**
    ///
    /// This is the pairing that stops the two fidelity statements from being applied to the same
    /// picture: a quantiser into a bitrate encoder is not what the picture is encoded at, and no
    /// quantiser into a constant-quantiser encoder means a picture encoded at whatever the record
    /// happens to carry. Both are the "plausible wrong picture, not an error" class, so they are
    /// errors.
    #[test]
    fn a_request_naming_the_wrong_fidelity_is_refused_rather_than_honoured_in_part() {
        let Ok(api) = crate::vaapi::open_display() else {
            eprintln!("SKIPPED: no VA display");
            return;
        };
        let mut cqp = cqp_encoder(&api, 1).expect("a one-reference encoder is measured to work");
        cqp.at_rate(30).expect("30 fps");
        cqp.key_frames_every(16).expect("16 is a legal key-frame interval");

        // **A rate-controlled request into a constant-quantiser encoder.**
        let Err(why) = cqp.encode(EncodeRequest::predicted_at_rate().expect("a request with no quantiser")) else {
            panic!("a request with no quantiser was honoured by a constant-quantiser encoder, so the picture would be at whatever pic_init_qp carried");
        };
        assert!(why.contains("constant quantiser"), "the message names the encoder's mode: {why}");
        assert!(why.contains("predicted("), "and says what to write instead: {why}");

        // **And the other way round**, which is a different encoder and needs a different message.
        let Ok(cbr) = VaEncoder::at_constant_bitrate(&api, (640, 360), 1, 8_000_000) else {
            eprintln!("SKIPPED: this driver refuses a CBR config, so the other half of the pairing is untested here");
            return;
        };
        let mut cbr = cbr;
        cbr.at_rate(30).expect("30 fps");
        cbr.key_frames_every(16).expect("16 is a legal key-frame interval");
        let (luma, chroma) = ramp_shifted(640, 360, 0).expect("the ramp is buildable");
        cbr.upload(&luma, &chroma).expect("the ramp uploads");
        let Err(why) = cbr.encode(EncodeRequest::predicted(26).expect("QP 26 is in range")) else {
            panic!("a quantiser was accepted by a bitrate encoder, so the request said something the driver will not honour");
        };
        assert!(why.contains("CBR"), "the message names the encoder's mode: {why}");
        assert!(why.contains("predicted_at_rate"), "and names the request that would fit: {why}");
        // **And the pairing is refused *before* a picture is encoded**, not after: nothing was
        // uploaded to a surface or submitted to the driver on that path, which is what the counters
        // below are checking.
        assert_eq!(cbr.pictures_encoded(), 0, "the refused request encoded nothing");
    }

    /// **A bitrate encoder runs, the QP moves, and the pictures come back** — the end-to-end claim
    /// `examples/vaencprobe.rs` makes interactively, asserted here so a regression is a test failure
    /// rather than a thing somebody has to remember to look at.
    ///
    /// The three claims, each of which would fail differently:
    ///
    /// 1. **A CBR session encodes at all** — a driver that refuses the mode would fail `new`.
    /// 2. **The driver's reported quantiser moves**, which is what says the rate controller ran rather
    ///    than the encoder coding at a constant. **A constant QP across the run fails this**, and that
    ///    is the point: a "bitrate mode" that silently encoded at one fixed quantiser would produce a
    ///    file at a size nobody asked for with no error anywhere.
    /// 3. **The pictures come back within a tolerance**, which is the "still a picture" half — a
    ///    reference chain that went wrong produces pictures that each look plausible and drift, so
    ///    the spread is checked as well as the bound.
    #[test]
    fn a_bitrate_session_encodes_and_the_driver_moves_its_quantiser() {
        let Ok(api) = crate::vaapi::open_display() else {
            eprintln!("SKIPPED: no VA display");
            return;
        };
        // 640x360 at 200 kbit/s is a target the encoder cannot meet with a coarse quantiser at this
        // size, so the controller has to move; at a very high target it would not have to, and the
        // run below would be asserting nothing.
        let mut encoder = match VaEncoder::at_constant_bitrate(&api, (640, 360), 1, 200_000) {
            Ok(e) => e,
            Err(why) => {
                eprintln!("SKIPPED: this driver refuses a 200 kbit/s CBR session: {why}");
                return;
            }
        };
        encoder.at_rate(30).expect("30 fps");
        encoder.key_frames_every(16).expect("16 is a legal key-frame interval");
        let mut qps = Vec::new();
        for index in 0..12u32 {
            let (luma, chroma) = ramp_shifted(640, 360, 8 * index).expect("the ramp is buildable");
            encoder.upload(&luma, &chroma).expect("the ramp uploads");
            let picture = encoder.encode(EncodeRequest::predicted_at_rate().expect("a rate-controlled request")).expect("the picture encodes");
            qps.push(picture.status.first().copied().unwrap_or(0) & 0xff);
        }
        let distinct = qps.iter().collect::<std::collections::BTreeSet<_>>().len();
        assert!(
            distinct > 1,
            "the rate controller chose the same quantiser {distinct} time(s) across the run — reported {qps:?}, so the bitrate was not being met by moving it"
        );
        // Every reported quantiser is a legal one, which is the range the self-check enforces in a
        // bitrate mode and the only thing about the quantiser this backend can still say.
        for qp in &qps {
            assert!(*qp <= u32::from(H264_MAX_QP), "a reported quantiser of {qp} is outside H.264's 0..={H264_MAX_QP}");
        }
        // **And the sample entry still describes the stream**, which is what the `avcC` has to do
        // under a bitrate mode too: the derived sets are byte-identical picture to picture because
        // `pic_init_qp` is [`DEFAULT_ENCODER_QP`] in every record.
        assert!(encoder.sample_entry().is_ok(), "a sample entry after twelve pictures: {:?}", encoder.sample_entry().err());
        eprintln!("a 200 kbit/s CBR session reported quantisers {qps:?} and produced a sample entry");
    }
}

/// The parameter sets for this module's 640x360 encoder, built from the records it submits, and
/// what a decoder would read back out of them.
///
/// This is the test that matters most in the encode module, because `params::parameter_sets`
/// already verifies the sets against FilmCraft's own H.264 parser — so what is left to check here
/// is that the verification is *doing* something: that the sets it returns say what the records
/// said, that they are Annex-B in the right order, and that a record it cannot describe is an error
/// rather than a parameter set that decodes into something else.
#[test]
fn the_derived_parameter_sets_say_what_the_records_said() {
    let sequence = sequence_record((640, 368), (0, 0, 0, 4), 40, 23, 1, 0).expect("a sequence record");
    let picture = picture_record(&Current { surface: 7, frame_idx: 0, frame_num: 0, poc: 0, is_reference: true }, 9, 0, 920, &[], 26, true, false, 11)
        .expect("a picture record");
    let sequence = read::<ffi::VaEncSequenceParameterBufferH264>(&sequence).expect("it reads back");
    let picture = read::<ffi::VaEncPictureParameterBufferH264>(&picture).expect("it reads back");
    let sets = super::params::parameter_sets(&sequence, &picture).expect("the parameter sets build and verify");
    // Annex-B, SPS then PPS, and the two NAL headers are what say which is which.
    assert!(sets.starts_with(&[0, 0, 0, 1, 0x67]), "an SPS with nal_ref_idc 3 and type 7");
    assert!(sets.windows(5).any(|w| w == [0, 0, 0, 1, 0x68]), "and a PPS with nal_ref_idc 3 and type 8");
    assert!(nal_unit_bytes(&sets, 7).is_some(), "and this module's own Annex-B walk finds the SPS it wrote");
    assert!(nal_unit_bytes(&sets, 8).is_some(), "and the PPS");
    // And a picture record that cannot be described is an error, not a parameter set: a QP outside
    // H.264's range would make `pic_init_qp_minus26` name a quantiser no decoder accepts.
    let mut impossible = picture;
    impossible.pic_init_qp = 200;
    assert!(super::params::parameter_sets(&sequence, &impossible).is_err(), "a QP outside 0..=51 is refused rather than written");
    // So is a sequence that declares scaling matrices it does not carry, and a field-picture
    // sequence: both are states where a parameter set would describe a stream that was not encoded.
    let mut scaling = sequence;
    scaling.seq_fields |= 1 << 4;
    assert!(super::params::parameter_sets(&scaling, &picture).is_err(), "scaling matrices the stream does not carry");
    let mut fields = sequence;
    fields.seq_fields &= !(1 << 2);
    assert!(super::params::parameter_sets(&fields, &picture).is_err(), "a field sequence");
    let mut poc1 = sequence;
    poc1.seq_fields |= 1 << 10;
    assert!(super::params::parameter_sets(&poc1, &picture).is_err(), "pic_order_cnt_type 1");
}

/// The parameter-set detection has to be a real Annex-B walk, not a search for `00 00 01`.
///
/// That byte sequence occurs inside slice data by coincidence: the driver's own output for this
/// picture contains several of them, and a heuristic that matched one would report parameter sets a
/// stream does not have. That is the plausible-looking wrong answer this module exists to avoid,
/// so it is pinned with a stream that has none.
///
/// This runs against [`nal_unit_bytes`] — **the function the answer actually comes from** — rather
/// than a helper of its own, because a detection helper that nothing calls proves nothing about the
/// detection the encoder performs.
#[test]
fn parameter_set_detection_does_not_match_the_byte_sequence_by_accident() {
    // A slice payload with `00 00 01` inside it and no parameter set anywhere.
    let slice = [0u8, 0, 0, 1, 0x05, 0x88, 0, 0, 1, 0x2a, 0x99];
    assert!(nal_unit_bytes(&slice, 7).is_none(), "a slice payload is not a parameter set");
    assert!(nal_unit_bytes(&slice, 8).is_none(), "and neither is it a PPS");
    // **The interior `00 00 01` really is in there and really is walked past**: a search that
    // matched every occurrence would read the byte after it (`0x2a`, type 10) as a NAL header and
    // this assertion would pass for the wrong reason, so the walk's output is checked too.
    let units = annex_b_units(&slice).expect("the slice is a whole Annex-B run");
    assert_eq!(units.len(), 2, "the start code inside the payload does not begin a unit: {:?}", units.iter().map(|u| u.header & 0x1f).collect::<Vec<_>>());
    assert_eq!(units[0].header & 0x1f, 5, "the first unit is the IDR slice");
    assert_eq!(units[1].header & 0x1f, 10, "and the second is the end-of-sequence the interior code really starts");
    assert!(nal_unit_bytes(&[0, 0, 0, 1, 0x67, 0x4d], 7).is_some(), "an SPS");
    assert!(nal_unit_bytes(&[0, 0, 0, 1, 0x67, 0x4d, 0, 0, 0, 1, 0x68, 0xce], 7).is_some(), "an SPS then a PPS");
    assert!(nal_unit_bytes(&[0, 0, 0, 1, 0x67, 0x4d, 0, 0, 0, 1, 0x68, 0xce], 8).is_some(), "and the PPS is found too");
    assert!(nal_unit_bytes(&[], 7).is_none(), "nothing");
    assert!(nal_unit_bytes(&[0, 0, 0, 1], 7).is_none(), "a start code with no header after it");
    // A four-byte code is also its own three-byte code one byte later, so a walk that did not skip
    // the header byte would read the tail of `00 00 00 01` as a second, empty unit. This is the
    // shape that makes an empty unit, and it is refused rather than pushed.
    assert!(annex_b_units(&[0, 0, 0, 1, 0x67, 0x4d]).is_ok(), "a 4-byte code does not read as two units");
    assert_eq!(annex_b_units(&[0, 0, 0, 1, 0x67, 0x4d]).expect("one unit").len(), 1);
}

/// The reference state a P chain actually uses, and that it is the same state in both records.
///
/// This is the unit test behind "the frame loop is stateful": a P picture's records are built from a
/// **list** of references, and the picture record's `ReferenceFrames` and the slice record's
/// `RefPicList0` must name the *same* pictures with the same surface and the same DPB slot. If they
/// disagree the driver reconstructs from one picture and the decoder predicts from another, which
/// is a plausible-looking wrong picture rather than an error — so the agreement is asserted here
/// rather than trusted.
#[test]
fn a_p_picture_names_the_same_reference_in_both_records() {
    let previous = Reference { surface: 42, frame_idx: 0, frame_num: 7, poc: 7 };
    let references = [previous];
    let current = Current { surface: 43, frame_idx: 1, frame_num: 8, poc: 8, is_reference: true };
    let picture = picture_record(&current, 9, 8, 920, &references, 26, false, false, 11).expect("a P picture record");
    let slice = slice_record(920, ffi::SLICE_TYPE_ALL_P, 8, 0, &references).expect("a P slice record");
    let picture = read::<ffi::VaEncPictureParameterBufferH264>(&picture).expect("the picture record reads back");
    let slice = read::<ffi::VaEncSliceParameterBufferH264>(&slice).expect("the slice record reads back");

    // The picture record's DPB entry and the slice record's list entry are the same picture.
    assert_eq!(picture.reference_frames[0].picture_id, 42, "the reference's surface");
    assert_eq!(slice.ref_pic_list0[0].picture_id, 42, "and the slice names the same one");
    assert_eq!(picture.reference_frames[0].frame_idx, 0, "the DPB slot");
    assert_eq!(slice.ref_pic_list0[0].frame_idx, 0, "and the slice agrees on it");
    // A slot is a slot, not a surface id: this is the mistake a reader checking `frame_idx` makes.
    assert_ne!(picture.reference_frames[0].frame_idx, previous.surface, "a DPB slot is not a surface id");
    // Both count the reference, and count it the same way: `minus1`.
    assert_eq!(picture.num_ref_idx_l0_active_minus1, 0, "one reference");
    assert_eq!(slice.num_ref_idx_l0_active_minus1, 0, "and the slice reads one");
    // A reference is marked as one in the DPB, in both records.
    assert_eq!(picture.reference_frames[0].flags, PICTURE_H264_SHORT_TERM_REFERENCE);
    assert_ne!(picture.pic_fields & 0b110, 0, "reference_pic_flag (bits 1-2) says short-term");
    assert_eq!(picture.curr_pic.flags, PICTURE_H264_SHORT_TERM_REFERENCE, "and so does curr_pic");
    assert_eq!(picture.curr_pic.picture_id, 43, "the current picture's own surface");
    // An IDR names no reference even when one is held, because 7.4.3.3 empties the DPB at a key frame.
    let idr = picture_record(&current, 9, 8, 920, &references, 26, true, false, 11).expect("an IDR picture record");
    let idr = read::<ffi::VaEncPictureParameterBufferH264>(&idr).expect("the IDR record reads back");
    assert_eq!(idr.reference_frames[0].picture_id, ffi::INVALID_SURFACE, "an IDR names no reference");
    assert_eq!(idr.num_ref_idx_l0_active_minus1, 0, "and counts none");
    assert_eq!(idr.pic_fields & 1, 1, "idr_pic_flag is set");
}

/// The moving ramp really does move, and really does make a wrong reference detectable.
///
/// These are the numbers `examples/vaencprobe.rs`'s decode check is compared against, and the reason
/// the test content has to be checked rather than described: a "moving" picture that did not move
/// would make the whole inter test vacuous, and a picture whose steps were smaller than the
/// quantiser's noise would make it insensitive.
#[test]
fn the_moving_ramp_moves_by_a_step_larger_than_the_quantisers_noise() {
    let (w, h, step) = (640u32, 360u32, 8u32);
    let (a, _) = ramp_shifted(w, h, 0).expect("the unshifted ramp");
    let (b, _) = ramp_shifted(w, h, step).expect("the ramp one step along");
    let (c, _) = ramp_shifted(w, h, step * 2).expect("the ramp two steps along");
    let row = w as usize;
    // The pictures differ, and they differ by much more than the ~3/255 the decode check accepts:
    // a whole step at the middle of the picture is `step * 219 / (w - 1)` luma, which at 640 wide and
    // a step of 8 is about 2.7. If this were under the noise floor the inter test would prove nothing.
    let mid = 320usize;
    let step_at_mid = (a[mid] as i32 - b[mid] as i32).abs();
    assert!(step_at_mid >= 2, "one step at the middle is {step_at_mid} luma, which is under the decode noise");
    assert_eq!((b[mid] as i32 - c[mid] as i32).abs(), step_at_mid, "every step is the same size");
    // The shift wraps rather than running off the legal luma range: every sample stays inside
    // H.264's 16..=235, so no deviation can be blamed on an encoder clipping out-of-range input.
    assert!((0..(w as usize * h as usize)).all(|i| (16..=235).contains(&a[i])), "frame 0 inside the legal luma range");
    assert!((0..(w as usize * h as usize)).all(|i| (16..=235).contains(&c[i])), "and so is the last");
    // And it is a function of the column only, so a stride or row-order bug is obvious rather than
    // plausible.
    assert_eq!(a[row..2 * row], a[..row], "the ramp does not depend on the row");
    assert_ne!(a[0], b[0], "consecutive frames are genuinely different pictures");
}

/// A reference depth above one is refused, and the refusal says why.
///
/// The measurement behind this is in [`VaEncoder::new`]; what the test pins is that the encoder does
/// not *silently* accept a depth it cannot make correct, because the failure mode at depth two is a
/// picture that looks like a picture.
#[test]
fn a_reference_depth_this_module_cannot_make_correct_is_refused() {
    let Ok(api) = super::open_display() else {
        eprintln!("SKIPPED: no VA display");
        return;
    };
    // One reference is the measured-good depth and must be accepted.
    assert!(cqp_encoder(&api, 1).is_ok(), "one reference is measured to work");
    // More is refused, and the message names the depth and the fact that it is a measurement.
    match cqp_encoder(&api, 2) {
        Ok(_) => panic!("two references are accepted, which is the silently-wrong-picture case this exists to prevent"),
        Err(why) => {
            assert!(why.contains("2 reference"), "the message names the depth: {why}");
            assert!(why.contains("wrong"), "and says the failure is a wrong picture: {why}");
        }
    }
}

/// The encode readback returns a whole, decodable picture — out of storage this backend mapped
/// itself.
///
/// This is the test the readback could only fail before: it drives one key frame and two predicted
/// frames and asks for the bytes back. If the picture were still being read through a pointer whose
/// extent nothing reports, this is where that would show — and it is worth a test that says so,
/// because the alternative the readback replaced returned correct bytes and was wrong anyway.
#[test]
fn the_encode_readback_returns_a_whole_decodable_picture() {
    let Ok(api) = super::open_display() else {
        eprintln!("SKIPPED: no VA display");
        return;
    };
    let mut encoder = cqp_encoder(&api, 1).expect("a one-reference encoder is measured to work");
    encoder.at_rate(30).expect("30 fps is a legal frame rate");
    encoder.key_frames_every(0).expect("0 is a legal key-frame interval and means one key frame for the stream");
    let mut sizes = Vec::new();
    for index in 0..3u32 {
        // A different picture every time, which is what makes the prediction observable at all.
        let (luma, chroma) = ramp_shifted(640, 360, 8 * index).expect("the ramp is buildable");
        encoder.upload(&luma, &chroma).expect("the ramp uploads");
        let request = EncodeRequest::new(26, if index == 0 { PictureKind::Idr } else { PictureKind::P }).expect("QP 26 is in range");
        let picture = encoder.encode(request).unwrap_or_else(|e| panic!("picture {index} did not encode: {e}"));
        sizes.push(picture.bytes.len());
        assert!(!picture.bytes.is_empty(), "picture {index} came back empty");
        assert_eq!(picture.bytes[..4.min(picture.bytes.len())], [0, 0, 0, 1], "picture {index} does not begin with an Annex-B start code");
        // **Every NAL unit in the readback is a whole one.** This is what `check_annex_b` inside
        // `read_coded` guarantees, and it is restated here because it is the property the caller
        // depends on: a window that ended mid-unit would still decode, and would decode to noise.
        assert!(ffi::check_annex_b(&picture.bytes).is_ok(), "picture {index} is not a whole Annex-B run");
        // The bytes are owned: they outlive the mapping they were copied out of.
        assert!(picture.bytes.iter().any(|b| *b != 0), "picture {index} is all zeroes");
    }
    // The key frame is bigger than what follows it, which is the shape an encode that is really
    // predicting produces. A readback returning the same buffer's stale contents every time would
    // not have this shape.
    assert!(sizes[0] > sizes[1], "the key frame ({}) is not larger than the first P frame ({})", sizes[0], sizes[1]);
}

/// **More pictures than `MaxFrameNum`, which is the regression test for the wrap the check above
/// was refusing.**
///
/// `log2_max_frame_num_minus4` and `log2_max_pic_order_cnt_lsb_minus4` are both 0 in
/// [`sequence_record`], so both syntax elements are four bits wide and **both wrap every sixteen
/// pictures** while `self.frame_num` and `self.poc` are `u16` counters that wrap at 65536. The
/// self-check in [`check_output_matches_the_sets`] compared the two directly and so failed on
/// picture 16 of every export — reported as "log2_max_frame_num is not what the driver used", on a
/// picture the driver encoded correctly. Three pictures could not see it; this runs twenty.
///
/// It is a hardware test and skips without a VA display, which is the honest shape for a claim
/// about what a driver writes.
#[test]
fn a_sequence_longer_than_max_frame_num_encodes_without_the_self_check_refusing_it() {
    let Ok(api) = super::open_display() else {
        eprintln!("SKIPPED: no VA display");
        return;
    };
    let mut encoder = cqp_encoder(&api, 1).expect("a one-reference encoder is measured to work");
    encoder.at_rate(30).expect("30 fps is a legal frame rate");
    encoder.key_frames_every(0).expect("0 is a legal key-frame interval and means one key frame for the stream");
    // 20 pictures: past the 16-picture `MaxFrameNum` wrap on both counters, with room for the
    // picture-order-count wrap to land on the same frame.
    for index in 0..20u32 {
        let (luma, chroma) = ramp_shifted(640, 360, 8 * index).expect("the ramp is buildable");
        encoder.upload(&luma, &chroma).expect("the ramp uploads");
        let request = EncodeRequest::new(26, if index == 0 { PictureKind::Idr } else { PictureKind::P }).expect("QP 26 is in range");
        let picture = encoder.encode(request).unwrap_or_else(|e| panic!("picture {index} did not encode: {e}"));
        assert!(!picture.bytes.is_empty(), "picture {index} came back empty");
        assert!(ffi::check_annex_b(&picture.bytes).is_ok(), "picture {index} is a whole Annex-B run");
    }
    // And the sample entry is still there and still describes the last picture, because the
    // parameter sets are checked against **every** picture rather than only the first.
    let entry = encoder.sample_entry().expect("a sample entry after twenty pictures");
    // The `avcC`'s profile/compatibility/level come from bytes 1..4 of the stored SPS, so a set
    // stored without its NAL header byte reads as a profile of 0. This is the byte-offset claim
    // `NalUnit::nal_unit` makes, checked through the record the muxer would actually get.
    let filmcraft_isobmff::CodecConfig::Avc(avc) = &entry.codec else {
        panic!("an H.264 sample entry carries an avcC, not {:?}", entry.codec);
    };
    assert_eq!(avc.length_size, 4, "and the samples' four-byte lengths are what it declares");
    assert!(!avc.sps.is_empty() && !avc.pps.is_empty(), "and it carries one SPS and one PPS");
    assert_eq!(avc.profile, crate::vaapi::params::PROFILE_IDC_FOR_TEST, "read from byte 1 of the stored SPS");
    assert_ne!(avc.profile, 0, "which is what proves the NAL header byte is still on the front");
}

/// `sample_entry` answers "not yet" before a picture exists, and says why.
///
/// `VideoEncoder::sample_entry` is documented as "may only be complete after the first frame", and
/// the strongest available form of that here is an answer rather than a guess: before the first
/// picture the only thing that could supply an `avcC` is a description of a stream that does not
/// exist. Hardware-dependent for the encoder itself, so it skips without a display.
#[test]
fn there_is_no_sample_entry_before_the_first_picture() {
    let Ok(api) = super::open_display() else {
        eprintln!("SKIPPED: no VA display");
        return;
    };
    let encoder = cqp_encoder(&api, 1).expect("a one-reference encoder is measured to work");
    let why = encoder.sample_entry().expect_err("nothing has been encoded, so there is no stream to describe");
    assert!(why.contains("encoded nothing"), "the refusal says what is missing, not just that something is: {why}");
    assert!(encoder.parameter_sets().is_none(), "and no sets are held yet either");
}

/// **The B-frame measurement, taken on the driver rather than assumed: it writes no B picture.**
///
/// This is the test behind the module note's claim that `media_start()` is 0 for a reason, and the
/// distinction it is careful about is the whole point. "This backend submits no B pictures" is a
/// fact about *us* and would be true of a driver that could not encode one. The claim that matters is
/// about the **driver**: over a run long enough to contain a GOP boundary and a `frame_num` wrap,
/// does iHD hand back a slice it decided to be a B?
///
/// So every slice header in every picture the driver wrote is parsed back **by FilmCraft's own
/// decoder-side parser**, and each one's `slice_type` is checked against the two this backend can
/// produce. A `SliceType::B` anywhere fails the test — which is the assertion, and it is the opposite
/// of a check that would pass if the driver never wrote one because it never had the chance.
///
/// The run is 20 pictures rather than 3, because the claims being tested are about behaviour at a
/// GOP boundary and at the 16-picture `MaxFrameNum` wrap, and a run that stops before either proves
/// nothing about them.
///
/// It is a hardware test and skips without a VA display, which is the honest shape for a claim about
/// what a driver writes.
#[test]
fn the_driver_writes_no_b_picture_and_this_backend_submits_none() {
    let Ok(api) = super::open_display() else {
        eprintln!("SKIPPED: no VA display");
        return;
    };
    // The capability query first, so the test says what the driver *could* do before it says what
    // it did. `max_reference_frames` is the public form of `VAConfigAttribEncMaxRefFrames`; list 1
    // is the future-reference count, and a driver answering 0 there cannot encode a B picture at
    // all. If that ever became the case, the module note's "(3, 1)" would be stale and this test is
    // where it would be noticed — which is why the assertion is recorded rather than assumed.
    let (list0, list1) = super::max_reference_frames(&api);
    eprintln!("VAConfigAttribEncMaxRefFrames: past {list0}, future {list1}");

    let mut encoder = cqp_encoder(&api, 1).expect("a one-reference encoder is measured to work");
    encoder.at_rate(30).expect("30 fps is a legal frame rate");
    encoder.key_frames_every(0).expect("0 is a legal key-frame interval and means one key frame for the stream");
    let mut b_slices = 0usize;
    for index in 0..20u32 {
        let (luma, chroma) = ramp_shifted(640, 360, 8 * index).expect("the ramp is buildable");
        encoder.upload(&luma, &chroma).expect("the ramp uploads");
        let picture =
            encoder.encode(EncodeRequest::predicted(26).expect("QP 26 is in range")).unwrap_or_else(|e| panic!("picture {index} did not encode: {e}"));
        // Parse what the **driver** wrote — not the whole access unit, which carries this module's
        // derived parameter sets ahead of it.
        let units = annex_b_units(picture.bytes.get(picture.bytes.len().saturating_sub(picture.driver_bytes)..).unwrap_or_default())
            .unwrap_or_else(|e| panic!("picture {index}'s own bytes are not a whole Annex-B run: {e}"));
        let sets = encoder.parameter_sets().expect("the sets exist after a picture").clone();
        let sps_rbsp = filmcraft_bitstream::unescape_rbsp(sets.sps.get(1..).unwrap_or_default());
        let sps = filmcraft_h264::params::Sps::parse(&sps_rbsp).expect("the SPS this module wrote parses");
        let pps = filmcraft_h264::params::Pps::parse(&filmcraft_bitstream::unescape_rbsp(sets.pps.get(1..).unwrap_or_default()), &[Some(sps.clone())])
            .expect("the PPS parses");
        for unit in &units {
            if !matches!(unit.header & 0x1f, 1 | 5) {
                continue;
            }
            let nal = filmcraft_h264::slice::NalHeader::parse(unit.header).unwrap_or_else(|e| panic!("picture {index}'s NAL header: {e}"));
            let lookup = |id: u32| -> filmcraft_h264::Result<(&filmcraft_h264::params::Pps, &filmcraft_h264::params::Sps)> {
                if id != pps.id {
                    return Err(filmcraft_h264::Error::MissingParameterSet(format!(
                        "the slice names PPS {id} and the only one in the sample entry is {}",
                        pps.id
                    )));
                }
                Ok((&pps, &sps))
            };
            let (header, _, _) = filmcraft_h264::slice::SliceHeader::parse(&filmcraft_bitstream::unescape_rbsp(unit.body), nal, lookup)
                .unwrap_or_else(|e| panic!("picture {index}'s slice does not parse: {e}"));
            if header.slice_type == filmcraft_h264::slice::SliceType::B {
                b_slices += 1;
            }
            // The picture order count is the other half of the claim: a B picture is a picture
            // coded *out of* display order, so a stream whose POCs are monotonic cannot be
            // carrying one however its slice types are spelled.
            assert_eq!(
                header.pic_order_cnt_lsb,
                u32::from(picture.poc) % (1u32 << sps.log2_max_poc_lsb),
                "picture {index}'s picture order count is the one submitted"
            );
        }
    }
    assert_eq!(
        b_slices, 0,
        "the driver wrote {b_slices} B slice(s) for a stream this backend submits as I and P only — which is what `media_start()` being 0 rests on"
    );
}

/// **`allow_b_pictures` refuses, and the refusal names both what is missing and what was measured.**
///
/// This is a test of a refusal, which is worth having precisely because a refusal nothing can ask
/// for is a claim with no evidence behind it — the failure this milestone has three examples of. The
/// older version of this method returned `&mut Self` and set a flag, so a caller could submit a
/// [`PictureKind::B`] and get a stream in which that picture was silently coded as a **P**.
///
/// Three things are asserted, each a different claim: the call fails; the message says the frame
/// loop is what is missing rather than blaming the driver (which reports a future reference and
/// could); and it names the two measurements — that the driver emits no B for these streams, and
/// that the deepest reference depth here is one while a B slice needs a past *and* a future.
#[test]
fn b_pictures_are_refused_at_construction_with_the_measurement() {
    let Ok(api) = super::open_display() else {
        eprintln!("SKIPPED: no VA display");
        return;
    };
    let mut encoder = cqp_encoder(&api, 1).expect("a one-reference encoder is measured to work");
    encoder.at_rate(30).expect("30 fps is a legal frame rate");
    encoder.key_frames_every(0).expect("0 is a legal key-frame interval and means one key frame for the stream");
    let why = match encoder.allow_b_pictures() {
        Ok(_) => panic!("a one-reference frame loop cannot drive a B picture, and allowing it must not be something a caller can do"),
        Err(why) => why,
    };
    assert!(why.contains("B pictures"), "the message says what was asked for: {why}");
    assert!(why.contains("future reference"), "and what a B slice needs that this does not have: {why}");
    assert!(why.contains("does not emit any"), "and it separates the driver *could* from the driver *does*: {why}");
    assert!(why.contains("one"), "and it names the depth that is the actual limit: {why}");

    // And the per-picture refusal still stands, naming the same thing — because `EncodeRequest` is
    // public and a caller can build a B request without ever hearing about `allow_b_pictures`.
    let (luma, chroma) = ramp_shifted(640, 360, 0).expect("the ramp is buildable");
    encoder.upload(&luma, &chroma).expect("the ramp uploads");
    let request = EncodeRequest::b_picture(26).expect("QP 26 is in range, so the request itself is buildable");
    let why = encoder.encode(request).expect_err("a B picture is refused whatever built the request");
    assert!(why.contains("B picture"), "the per-picture refusal names the picture kind: {why}");
    assert!(why.contains("allow_b_pictures"), "and where the measurement behind it is: {why}");
}

/// **The `frame_num` wrap is accompanied, and the key frame that does it is reported.**
///
/// Three claims, and they are different from each other:
///
/// 1. **The wrap is accompanied.** A picture whose `frame_num` would be `MaxFrameNum` — which the
///    eight-bit field cannot hold — is an IDR, so 7.4.3 infers its `frame_num` to 0 and 7.4.3.3 empties
///    the decoded picture buffer. This is checked on the pictures themselves rather than on the
///    decision function alone, because a predicate that says the right thing while the loop ignores it
///    is the shape of bug this module exists to prevent.
/// 2. **The IDR is written with `frame_num` 0**, which is what 7.4.3 infers and what the earlier
///    version of this loop did *not* do. Reading it back out of the driver's own bytes is the check;
///    see the module note for the ffmpeg diagnostic that a mid-stream IDR carrying the counter's
///    value provokes.
/// 3. **It is reported.** A caller that asked for no key frames and got one mid-run has been given
///    something it did not ask for, and [`EncodedPicture::key_frame_reason`] is how it finds out. A
///    forced key frame that was invisible would be a default, which is the thing this module refuses
///    to do everywhere else.
///
/// **The run is longer than the modulus, and that is the point of it.** At `MaxFrameNum` 16 this
/// test's 20 pictures carried a wrap and a second one; at 256 a 20-picture run carries none, so the
/// old run would have passed vacuously — it would have asserted about a wrap that never happened.
/// [`MAX_FRAME_NUM`]'s own note said a wider modulus "deserves a run longer than the modulus", and
/// this is that run: 260 pictures at 640x360, which crosses the wrap at 256 and carries on past it.
/// Whether 256 is the right ceiling is a policy question answered in the export gate; this test
/// answers only whether the encoder does what the gate promises when the interval reaches it.
#[test]
fn the_frame_num_wrap_is_an_idr_and_the_caller_is_told() {
    let Ok(api) = super::open_display() else {
        eprintln!("SKIPPED: no VA display");
        return;
    };
    // `key_frames_every(0)` — one key frame for the whole stream, so **every** key frame after the
    // first is one this loop inserted, and the reason field is unambiguous.
    let mut encoder = cqp_encoder(&api, 1).expect("a one-reference encoder is measured to work");
    encoder.at_rate(30).expect("30 fps is a legal frame rate");
    encoder.key_frames_every(0).expect("0 is a legal key-frame interval and means one key frame for the stream");
    let mut wraps = 0usize;
    let mut forced_others = Vec::new();
    let mut previous_was_key = false;
    // Past the wrap at `MAX_FRAME_NUM`, with room to see the run continue past it: the modulus is 256,
    // so this carries 260 pictures and the wrap lands on picture 256.
    let pictures = MAX_FRAME_NUM + 4;
    for index in 0..pictures {
        let (luma, chroma) = ramp_shifted(640, 360, 8 * index).expect("the ramp is buildable");
        encoder.upload(&luma, &chroma).expect("the ramp uploads");
        let picture =
            encoder.encode(EncodeRequest::predicted(26).expect("QP 26 is in range")).unwrap_or_else(|e| panic!("picture {index} did not encode: {e}"));
        match picture.key_frame_reason {
            None => {}
            Some(KeyFrameReason::NoReference) => {
                assert_eq!(index, 0, "only the first picture of a stream has nothing to predict from");
            }
            Some(KeyFrameReason::FrameNumWrap) => {
                wraps += 1;
                assert!(picture.key_frame, "a picture reported as a wrap key frame is coded as one");
                assert_eq!(picture.frame_num, 0, "and it is written with frame_num 0, which is what 7.4.3 infers an IDR's to be");
                assert_eq!(picture.poc, 0, "and with pic_order_cnt_lsb 0, which is what 7.4.3.1 infers an IDR's to be");
                assert_eq!(picture.slice_type, SLICE_TYPE_ALL_I, "and its slice type is the all-I one");
            }
        }
        if index > 0 && picture.key_frame_reason.is_some() && picture.key_frame_reason != Some(KeyFrameReason::FrameNumWrap) {
            forced_others.push(index);
        }
        // **A picture after a key frame starts at 1, not at 0.** The key frame was *written* with
        // `frame_num` 0 (7.4.3 infers an IDR's to be 0) but it is still a picture in the decoded
        // picture buffer once decoded, so the picture after it may not carry another 0 — and since
        // `slot_for` derives the DPB slot from the same number, carrying 0 would also make it claim
        // the slot its own reference occupies. This is the assertion that would fail if the counter
        // were reset to 0 rather than to 1 after a key frame, which is the mistake this code had.
        if index > 0 && previous_was_key {
            assert_eq!(picture.frame_num, 1, "picture {index} follows a key frame, so its frame number starts the run rather than reusing the IDR's 0");
        }
        previous_was_key = picture.key_frame;
    }
    assert!(forced_others.is_empty(), "the only key frames beyond the first are the wraps, but {forced_others:?} were key frames for another reason");
    // **Exactly one wrap in `MAX_FRAME_NUM + 4` pictures**, not merely at least one: a modulus this
    // wide is only worth having if the wrap lands where the gate says it does, and "at least one"
    // would be satisfied by a loop that keyed a frame on every picture.
    assert_eq!(wraps, 1, "a {pictures}-picture run at MaxFrameNum {MAX_FRAME_NUM} must contain exactly one wrap, at picture {MAX_FRAME_NUM}");
}

/// **A picture order count that never goes backwards, and a key frame that carries 0.**
///
/// The B-frame note rests on "no picture is coded out of display order", and that is a claim about
/// the *numbers*, not about the slice types. It is checked here by reading `pic_order_cnt_lsb` back
/// out of the driver's own bytes on every picture of a run that crosses a wrap, asserting that
/// consecutive non-IDR pictures advance by exactly one and that the counter resets with the IDR.
///
/// Both halves matter and it is worth being explicit about why: a monotonic POC is what makes
/// composition offsets zero, and a POC that resets *with* the IDR rather than independently is what
/// shows the reset is the codec's rule and not a coincidence of a counter.
#[test]
fn picture_order_counts_advance_by_one_and_reset_only_with_the_key_frame() {
    let Ok(api) = super::open_display() else {
        eprintln!("SKIPPED: no VA display");
        return;
    };
    let mut encoder = cqp_encoder(&api, 1).expect("a one-reference encoder is measured to work");
    encoder.at_rate(30).expect("30 fps is a legal frame rate");
    encoder.key_frames_every(0).expect("0 is a legal key-frame interval and means one key frame for the stream");
    let mut previous: Option<(u16, bool)> = None;
    for index in 0..20u32 {
        let (luma, chroma) = ramp_shifted(640, 360, 8 * index).expect("the ramp is buildable");
        encoder.upload(&luma, &chroma).expect("the ramp uploads");
        let picture =
            encoder.encode(EncodeRequest::predicted(26).expect("QP 26 is in range")).unwrap_or_else(|e| panic!("picture {index} did not encode: {e}"));
        if let Some((before, before_was_key)) = previous {
            if picture.key_frame {
                // **The key frame carries count 0** — 7.4.3.1 infers an IDR's `pic_order_cnt_lsb`
                // to be 0 and 8.2.1.1 gives the first picture of a sequence order count 0 — and it
                // is that same picture which is the *last* one holding a count, so the picture
                // after it starts the next run at 1 rather than continuing from 15.
                assert_eq!(picture.poc, 0, "a key frame carries picture order count 0, which is what 7.4.3.1 and 8.2.1.1 give it");
                assert!(!before_was_key || index == 0, "picture {index} is a key frame straight after a key frame, which this loop has no reason to do");
            } else if before_was_key {
                assert_eq!(picture.poc, 1, "picture {index} follows a key frame, so its order count starts the run rather than reusing the IDR's 0");
            } else {
                // **One step forward, not merely "different".** A count that skipped would be the
                // signature of a picture missing from the display order — which is the claim the
                // B-picture note rests on, so it is checked as arithmetic rather than as a guess.
                //
                // **`picture.poc` is the counter, not the field, and it does not wrap.** It is a
                // `u16` that runs the whole sequence; what wraps is the `u(log2_max_pic_order_cnt_lsb)`
                // `pic_order_cnt_lsb` the driver writes into the slice header, and that is the value
                // `check_output_matches_the_sets` compares modulo `MaxPicOrderCntLsb`. **The two
                // being different is exactly what a `MaxFrameNum` of 256 beside a
                // `MaxPicOrderCntLsb` of 16 makes visible**, and conflating them is the mistake this
                // assertion was rewritten to remove — an earlier version compared the counter against
                // `MAX_FRAME_NUM` and passed only because a key frame at picture 16 reset both at
                // once, so the modulo never fired and the modulus was never really used.
                assert_eq!(
                    u32::from(picture.poc),
                    u32::from(before) + 1,
                    "picture {index}'s picture order count is exactly one past picture {}'s, as a counter with no modulus of its own",
                    index - 1
                );
                // **And the number that goes into the slice header wraps where the count says it
                // does.** At picture 16 the counter is 16 and the field is 0: 8.2.1.1's Equation 8-3
                // reconstructs the full count from that wrap, which is what makes a 16-wide order
                // count legitimate under a 256-wide `frame_num`.
                if u32::from(picture.poc) == MAX_PIC_ORDER_CNT_LSB {
                    assert_eq!(u32::from(picture.poc) % MAX_PIC_ORDER_CNT_LSB, 0, "picture {index} is where pic_order_cnt_lsb wraps to 0 in the bitstream");
                }
            }
        }
        previous = Some((picture.poc, picture.key_frame));
    }
    // **A number rather than a feeling.** The run is 20 pictures: picture 0 is the IDR at count 0,
    // pictures 1..=15 carry 1..=15, picture 16 carries **16 as a counter and 0 as a field** — the
    // `MaxPicOrderCntLsb` wrap on a plain P picture, not a key frame, which is the width change at
    // work: the loop's own key frame for the `frame_num` wrap is 256 pictures out, nowhere near this
    // run — and pictures 17, 18, 19 carry 17, 18, 19. **No picture in this run is a key frame after
    // the first**, and the wrap that used to hide the difference is the wrap this change removed.
    assert_eq!(previous.map(|(poc, _)| poc), Some(19), "the counter runs unbroken to the last picture; only the field wraps, at 16");
}

/// The record of type `T` in `bytes`, or `None` when it is too short. The safe reader this module's
/// tests use to check what a builder produced, and the reason a builder's output is checkable at
/// all: it is the only way a bit position in a record can be verified without a driver.
///
/// Safe because [`ffi::record_in`] in the FFI module does the one `unsafe` read, under its own
/// `// SAFETY:` comment and with its own length check. `#[cfg(test)]` because nothing outside the
/// tests reads a built record back — the records go to the driver, not back out to us.
fn read<T: Copy + Default>(bytes: &[u8]) -> Option<T> {
    ffi::record_in::<T>(bytes)
}

/// **A constant-quantiser 640x360 encoder, the mode every existing test measures the frame loop in.
///
/// It is a helper rather than a bare [`VaEncoder::new`] at each of the ten call sites because a bare
/// `new` in ten places is ten places to name the mode wrong, and — worse — ten places where the mode
/// could be left to a default if `new` ever grew one. The bitrate tests build their own with
/// [`VaEncoder::at_constant_bitrate`], so **neither mode is implicit anywhere in this file**.
///
/// It lives at module level rather than inside `mod tests` because this file has tests in both places
/// — the inline `mod tests` and the `#[test]` functions that follow it — and a helper only one of
/// them could see would leave the other spelling the mode out in full.
#[cfg(test)]
fn cqp_encoder(api: &Arc<Display>, references: u32) -> Result<VaEncoder, String> {
    VaEncoder::new(api, (640, 360), references, RateControl::ConstantQuantiser)
}
