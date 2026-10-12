//! AV1 **bitstream assembly** for the encode submission path: the `sequence_header_obu()` and
//! `frame_header_obu()` that a libva AV1 encoder is handed as packed headers.
//!
//! # Why this file exists at all
//!
//! HEVC's path needed the same file for the same reason, and the reason is in
//! `va_enc_hevc.h:299-309`: **the driver writes no slice header of its own.** It writes the bytes
//! the application supplies and then the slice *data*. AV1 is the same shape one level up:
//! `va_enc_av1.h:213-230` declares `VAEncPackedHeaderAV1_SPS` ("shall contain the complete
//! `sequence_header_obu()` syntax element") and `VAEncPackedHeaderAV1_PPS` ("shall contain the
//! complete `frame_header_obu()` syntax element"), and `va_enc_av1.h:923-934` says the frame-header
//! OBU "contains an obu header, a 4-byte long obu_size field, `frame_header_obu()` syntax chain,
//! and a trailing bit". **Nothing in these records lets a driver invent that syntax.** So a
//! submission has to own it, exactly as HEVC's had to own its slice header, and this is that file.
//!
//! # Built from the specification, and which specification
//!
//! Every bit here comes from the **AV1 Bitstream & Decoding Process Specification, version 1.0.0
//! with Errata 1** (AOMediaCodec/av1-spec, commit `5e04f3f75e73a5898d7616c47c52f032144b8f80`) —
//! the same pinned revision `crates/av1/README.md` names and `crates/av1/tools/extract_tables.py`
//! downloads. `AGENTS.md` §2 permits implementing from "the VP9 and AV1 bitstream specs", so this
//! is a permitted source and not a third-party decoder's implementation.
//!
//! **This corrects a premise carried in the brief and in [`super::av1_enc`]'s module note**, both
//! of which record the specification as unavailable to this tree. It is available and it is
//! permitted: `crates/av1` is built entirely from it and ships a tool that fetches it. What is
//! *not* permitted, and is not used, is reading a third-party decoder's or encoder's source —
//! which is the only thing the note's `wmmat` paragraph actually needed, and which it did not
//! have.
//!
//! # The `wmmat` scale, resolved — and how
//!
//! [`super::av1_enc`]'s first open question is "the `wmmat` scale is not stated anywhere in the
//! header". **That is true of the header and false of the problem**, and the specification settles
//! it without any implementation source:
//!
//! - `03.symbols.md:124` — `WARPEDMODEL_PREC_BITS` = **16**, "Internal precision of warped motion
//!   models".
//! - `06.bitstream.syntax.md`, `global_motion_params()` — every `gm_params[ref][i]` is initialised to
//!   `(i % 3 == 2) ? 1 << WARPEDMODEL_PREC_BITS : 0`, i.e. **`[0, 0, 65536, 0, 0, 65536]`**.
//!
//! So the identity warp's six parameters are specified, not invented, and they are exactly what
//! `filmcraft_av1`'s own `default_gm_params()` already contained before this file existed.
//!
//! **The stronger point is that a submission of this shape never needs the scale at all.**
//! `global_motion_params()` reads nothing at all when `is_global` is 0, and
//! [`super::av1_enc::transformation::parameters`] says identity reads **0** of the eight `wmmat`
//! entries. A key frame returns from `global_motion_params()` before reading anything; a predicted
//! frame that names identity global motion for its one reference writes one `is_global = 0` bit per
//! reference frame and no parameters. **The unstated scale therefore cannot affect a submission
//! this module builds**, and the record's `wm[].wmmat` is left zero — which is the only value that
//! is correct for a transform that reads no parameters.
//!
//! # What this file does not do
//!
//! It does not call libva and it does not submit anything; [`super::submit`] does that. Every
//! function here is pure, and **every submission is checked by parsing the bytes back with
//! `filmcraft_av1`'s own parser** — `check_sequence_header` and `check_frame_header` below run inside
//! [`super::submit::build_submission`], so a header this module cannot describe is refused before any
//! display is touched. The tests in `submit` exercise both, for a key frame and for a predicted one.
//! That is the same round-trip discipline `hevc_params.rs` uses for its derived VPS/SPS/PPS, and the
//! reason to prefer FilmCraft's parsers over a second transcription of the syntax is the same: **a bit
//! written in the wrong place parses as something else.** Five such bugs were found this way while
//! this file was written — an OBU-header bit read the wrong way round, a `max_frame_*_minus_1` written
//! in 15 bits instead of 16, `frame_type` built from a bool cast (KEY_FRAME is 0, not 1),
//! `force_integer_mv` and `use_ref_frame_mvs` written where the specification infers them, and
//! `showable_frame` written where it is inferred — **none of which any size or offset check would have
//! seen.**
//!
//! **What a round trip is worth, stated plainly.** `filmcraft-av1`'s frame-header parser is
//! bit-exact against libdav1d on real streams, so a header that parses back to the values it was
//! written from is a header in the bitstream. It is **not** evidence about the driver: everything
//! else submitted is a record the driver reads and nothing comes back except the bitstream.

/// The specification constants this file needs, named here with their sources.
///
/// **`filmcraft_av1` keeps most of these in its private `spec_tables`**, which is the right place for
/// a decoder's full table and the wrong place for a submission path to reach into — a submission
/// cannot ask another crate for a `pub` it does not export. Every value below is a **named constant
/// of the AV1 specification**, not a table lookup, and `crates/av1/src/spec_tables.rs` already holds
/// the same numbers extracted from the same pinned revision; this file states them so a reader of
/// *this* file can check them against the specification without opening another crate.
pub mod consts {
    /// `CP_BT_709` (spec 6.4.2): 1.
    pub const CP_BT_709: u32 = 1;
    /// `TC_BT_709` (spec 6.4.2): 1.
    pub const TC_BT_709: u32 = 1;
    /// `MC_BT_709` (spec 6.4.2): 1.
    pub const MC_BT_709: u32 = 1;
    /// `CSP_UNKNOWN` (spec 6.4.2): 0, the chroma sample position a 4:2:0 picture with no stated
    /// position carries.
    pub const CSP_UNKNOWN: u32 = 0;
    /// `SELECT_SCREEN_CONTENT_TOOLS` (spec 6.10.1): 2 — "the bitstream contains a syntax element
    /// selecting whether to use screen content tools".
    pub const SELECT_SCREEN_CONTENT_TOOLS: u32 = 2;
    /// `SELECT_INTEGER_MV` (spec 6.10.1): 2.
    pub const SELECT_INTEGER_MV: u32 = 2;
    /// `TX_MODE_SELECT` (spec 6.10.3): 2, against `ONLY_4X4` 0 and `TX_MODE_LARGEST` 1.
    pub const TX_MODE_SELECT: u32 = 2;
}

/// `IDENTITY` in the specification's global-motion enumeration (spec 6.10), spelled here rather than
/// taken from `filmcraft_av1` because `header` does not re-export it and the value is this file's own
/// claim about the bitstream.
const IDENTITY: u8 = 0;

/// `OBU_SEQUENCE_HEADER` (spec 6.2.2, value 1) — the OBU a `sequence_header_obu()` is wrapped in.
const OBU_SEQUENCE_HEADER: u8 = 1;
/// `OBU_FRAME_HEADER` (spec 6.2.2, value 3).
const OBU_FRAME_HEADER: u8 = 3;

/// `WARPEDMODEL_PREC_BITS` (`03.symbols.md:124`): **16**, "Internal precision of warped motion
/// models". Re-declared from the specification here rather than taken from `filmcraft_av1`, because
/// this file is the one that asserts the `wmmat` scale and a constant that came from the same crate
/// as the verifier proves nothing to a reader of this file.
pub const WARPEDMODEL_PREC_BITS: u32 = 16;

/// **`1 << WARPEDMODEL_PREC_BITS`**: the value the specification writes at `gm_params[ref][2]` and
/// `[5]` for the identity warp (`global_motion_params()`'s initialiser).
pub const IDENTITY_WARP_DIAGONAL: i32 = 1 << WARPEDMODEL_PREC_BITS;

/// The identity warp's six parameters, as `global_motion_params()` initialises them:
/// `[0, 0, 1 << WARPEDMODEL_PREC_BITS, 0, 0, 1 << WARPEDMODEL_PREC_BITS]`.
///
/// **Exposed as one constant rather than recomputed**, because `filmcraft_av1`'s decoder and this
/// writer must agree and a reader has to be able to check them against each other.
pub const IDENTITY_WARP: [i32; 6] = [0, 0, IDENTITY_WARP_DIAGONAL, 0, 0, IDENTITY_WARP_DIAGONAL];

/// `num_units_in_display_tick` this path writes: **1000**, so `time_scale` is the rate in
/// milli-fps and an integral rate states itself without a fraction.
///
/// **A convention rather than a fact**, and one that has to be written down because a reader of the
/// timing info cannot recover it: the pair (1000, rate x 1000) says "rate frames per second" only
/// if you know this constant.
pub const DISPLAY_TICK_HZ: u32 = 1000;

/// The AV1 level table (Annex A, "Levels"): `(seq_level_idx, MaxPicSize, MaxHSize, MaxVSize,
/// MaxDisplayRate)` for every level the annex defines.
///
/// **Rows the annex leaves blank are absent, not zero** — the annex's note says "the missing
/// entries in these tables (for example level 2.2 and 7.0) represent levels that are not yet
/// defined", and a row of zeros would be a level that permits nothing, which is a different
/// claim from "this level is not defined". `seq_level_idx` 2 (2.2), 3 (2.3), 6 (3.2), 7 (3.3),
/// 20 (7.0), 21 (7.1), 22 (7.2) and 23 (7.3) are therefore missing from this table, and
/// [`level_for`] skips over them by walking `seq_level_idx` rather than the rows.
static LEVELS: [(u8, u64, u32, u32, u64); 15] = [
    (0, 147_456, 2048, 1152, 4_423_680),
    (1, 278_784, 2816, 1584, 8_363_520),
    (4, 665_856, 4352, 2448, 19_975_680),
    (5, 1_065_024, 5504, 3096, 31_950_720),
    (8, 2_359_296, 6144, 3456, 70_778_880),
    (9, 2_359_296, 6144, 3456, 141_557_760),
    (12, 8_912_896, 8192, 4352, 267_386_880),
    (13, 8_912_896, 8192, 4352, 534_773_760),
    (14, 8_912_896, 8192, 4352, 1_069_547_520),
    (15, 8_912_896, 8192, 4352, 1_069_547_520),
    (16, 35_651_584, 16384, 8704, 1_069_547_520),
    (17, 35_651_584, 16384, 8704, 2_139_095_040),
    (18, 35_651_584, 16384, 8704, 4_278_190_080),
    (19, 35_651_584, 16384, 8704, 4_278_190_080),
    (24, 35_651_584, 16384, 8704, 4_278_190_080),
];

/// The lowest `seq_level_idx` whose Annex A limits contain a `width` x `height` picture at
/// `frames_per_second`, or why there is none.
///
/// **Derived, not chosen**, for the reason HEVC's is: a level too low is a stream a conforming
/// decoder refuses. Annex A's `MaxPicSize`, `MaxHSize`, `MaxVSize` and `MaxDisplayRate` are the four
/// limits that a picture and a rate can violate, and all four are checked.
pub fn level_for(width: u32, height: u32, frames_per_second: u32) -> Result<u8, String> {
    let samples = (width as u64).checked_mul(height as u64).ok_or("a picture size that overflows a sample count")?;
    let rate = samples.checked_mul(frames_per_second as u64).ok_or("a frame rate that overflows a sample rate")?;
    for (idx, max_pic, max_h, max_v, max_rate) in LEVELS {
        if samples <= max_pic && width <= max_h && height <= max_v && rate <= max_rate {
            return Ok(idx);
        }
    }
    Err(format!("no AV1 level in the annex's table holds a {width}x{height} picture at {frames_per_second} fps"))
}

/// MSB-first bit writer, which is what every AV1 syntax element in this file uses.
///
/// **Not `filmcraft_bitstream::BitWriter`**, for one reason: AV1's `f(n)` is MSB-first like every
/// other codec here, but it also needs `su(n)`, `ns(n)` and `uvlc()`, which that writer does not
/// have, and reaching for it and then adding the three missing syntaxes to it would put AV1's
/// `trailing_bits` (one 1 then zeros — *not* `rbsp_trailing`) on someone else's type.
#[derive(Clone, Debug, Default)]
struct Bits {
    /// One entry per bit. A `Vec<u8>` of 0/1 rather than packed bytes, because every write here is
    /// at most 32 bits and the code is never on a hot path; the packing is in [`Bits::bytes`].
    bits: Vec<u8>,
}

impl Bits {
    fn new() -> Self {
        Self::default()
    }

    /// `f(n)`: the `n` bits of `value`, most significant first. `n` above 32 is refused rather than
    /// truncated, because AV1's widest `f()` in this file is `f(32)` for the timing info and a
    /// silent truncation would be a wrong number in the stream.
    fn f(&mut self, value: u32, n: u32) -> Result<(), String> {
        if n > 32 {
            return Err(format!("f({n}): AV1 writes at most 32 bits in one go"));
        }
        for i in (0..n).rev() {
            self.bits.push(((value >> i) & 1) as u8);
        }
        Ok(())
    }

    /// A single bit.
    fn flag(&mut self, value: bool) {
        self.bits.push(u8::from(value));
    }

    /// `uvlc()` (spec 4.10.3). AV1's only use in this file is `num_ticks_per_picture_minus_1`.
    fn uvlc(&mut self, value: u32) -> Result<(), String> {
        if value == 0 {
            self.flag(true);
            return Ok(());
        }
        // `leadingZeros` is the count of zero bits before the `leadingZeros` payload bits.
        let mut scratch = value;
        let mut leading = 0u32;
        while scratch > 0 {
            leading = leading.saturating_add(1);
            scratch >>= 1;
        }
        for _ in 0..leading {
            self.flag(false);
        }
        self.f(value, leading)
    }

    /// `trailing_bits()` (spec 5.3.1): a `1` then zeros until the byte is aligned.
    fn trailing_bits(&mut self) {
        self.bits.push(1);
        while !self.bits.len().is_multiple_of(8) {
            self.bits.push(0);
        }
    }

    /// The bytes written so far. The caller has already appended `trailing_bits()`, so this is
    /// whole bytes by construction — and it is a `Result` rather than a panic because
    /// `trailing_bits()` is what makes it true, and a writer that could be finished early would be
    /// a writer that can produce a partial byte.
    fn bytes(&self) -> Result<Vec<u8>, String> {
        if !self.bits.len().is_multiple_of(8) {
            return Err(format!("{} bits written, which is not a whole number of bytes", self.bits.len()));
        }
        let mut out = Vec::with_capacity(self.bits.len() / 8);
        for chunk in self.bits.chunks(8) {
            let mut byte = 0u8;
            for bit in chunk {
                byte = (byte << 1) | (*bit & 1);
            }
            out.push(byte);
        }
        Ok(out)
    }
}

/// `leb128()` (spec 4.10.6): seven bits per byte, little end first, high bit set on every byte but
/// the last.
fn leb128(value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    let mut left = value;
    loop {
        let byte = (left & 0x7f) as u8;
        left >>= 7;
        if left == 0 {
            out.push(byte);
            break;
        }
        out.push(byte | 0x80);
    }
    out
}

/// `leb128()` written into exactly four bytes, little end first.
///
/// **This is what the libva AV1 frame-header OBU carries**, and it is the one place this file does
/// something the plain `leb128()` would not: `va_enc_av1.h:923-934` says the frame-header OBU bit
/// stream "contains an obu header, **a 4-byte long obu_size field**, `frame_header_obu()` syntax
/// chain, and a trailing bit". A decoder reads the value, not the width, and AV1's `leb128()` allows
/// up to eight bytes, so a fixed-width four-byte field is a legal encoding of the same number — but
/// it is the encoding the header names, so it is the one written.
fn leb128_fixed4(value: u32) -> Vec<u8> {
    let bytes = value.to_le_bytes();
    let significant = leb128(value as u64).len().min(4);
    // The value is written little-end first; the `leb128` continuation bits live in the top bit of
    // every byte but the last, so a width-4 field is the little-end bytes with the continuation
    // bits set on all but the highest-index byte that still carries value.
    let mut out = [0u8; 4];
    out[..significant].copy_from_slice(&bytes[..significant]);
    for byte in out.iter_mut().take(significant.saturating_sub(1)) {
        *byte |= 0x80;
    }
    out.to_vec()
}

/// An OBU header (spec 6.2.1) with a size field, as a single byte.
///
/// `obu_has_size_field` is 1 and `obu_extension_flag` is 0, which is the form this path submits:
/// `tile_group_obu_hdr_info.obu_extension_flag` is the record's own statement of the extension flag
/// and `obu_has_size_field` its statement of the size field, so both are one place and both are
/// written from the record's value rather than from here.
fn obu_header(obu_type: u8, extension: bool, has_size: bool) -> Result<u8, String> {
    if obu_type > 15 {
        return Err(format!("an OBU type of {obu_type}: obu_type is f(4)"));
    }
    let mut bits = Bits::new();
    bits.flag(false); // obu_forbidden_bit
    bits.f(obu_type as u32, 4)?;
    bits.flag(extension);
    bits.flag(has_size);
    bits.flag(false); // obu_reserved_1bit
    bits.bytes()?.first().copied().ok_or_else(|| "an OBU header is one byte and none was written".to_string())
}

/// What a picture is coded as, and — for a predicted picture — what it predicts from.
///
/// **Two cases, not three.** `INTRA_ONLY_FRAME` and `SWITCH_FRAME` exist in the syntax but are
/// refused here for the reasons [`super::submit`] gives: an intra-only frame needs a reference for
/// the propagated segment ids and context tables, and a switch frame needs an error-resilient
/// header this module does not write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    /// A key frame: `frame_type = KEY_FRAME`, no references, `primary_ref_frame = PRIMARY_REF_NONE`.
    Key {
        /// `order_hint` for this picture, `OrderHintBits` bits wide.
        order_hint: u32,
    },
    /// A predicted frame: `frame_type = INTER_FRAME`, one reference, `reference_mode = SINGLE_REFERENCE`.
    Inter {
        /// `order_hint` for this picture, `OrderHintBits` bits wide.
        order_hint: u32,
        /// The `reference_frames[]` index the reference picture's surface occupies.
        reference: u8,
    },
}

/// Everything both OBUs derive from: the geometry, the rate, the level and the quantiser.
///
/// **One struct rather than two sets of arguments**, because the sequence header and the frame
/// header must agree about the size and the rate or the stream is not decodable, and a pair of
/// independent argument lists is how they stop agreeing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Av1Plan {
    /// The coded picture width, in luma samples.
    pub width: u32,
    /// The coded picture height, in luma samples.
    pub height: u32,
    /// The frame rate the timing info states, in frames per second.
    pub frames_per_second: u32,
    /// `OrderHintBits` — 1..=8. `order_hint` is this wide in every frame header.
    pub order_hint_bits: u32,
    /// The `seq_level_idx` this picture and rate need, from [`level_for`].
    pub level_idx: u8,
    /// `base_q_idx` for every picture this plan writes.
    pub base_qindex: u8,
}

impl Av1Plan {
    /// A plan for a `width` x `height` picture at `frames_per_second`, with `OrderHintBits = 7`.
    ///
    /// **No range check on `base_qindex`, because there is none to make.** It is a `u8`, and
    /// `base_q_idx` is `f(8)` — a check that `u8` is at most 255 compares a literal with its own
    /// type and cannot fail, which is the kind of assertion this tree's notes call out twice.
    ///
    /// **A size that cannot be expressed is refused, not rounded.** AV1 has no crop in the
    /// sequence header — `frame_size()` derives the size from `max_frame_*_minus_1` and a frame's
    /// own `frame_size_override_flag` — so there is nowhere to say "the picture was padded", and a
    /// driver given a padded size writes the padded conformance window.
    pub fn new(width: u32, height: u32, frames_per_second: u32, base_qindex: u8) -> Result<Self, String> {
        if width == 0 || height == 0 {
            return Err(format!("a {width}x{height} picture"));
        }
        if frames_per_second == 0 {
            return Err("a frame rate of 0 fps: the timing info would divide by it".to_string());
        }
        // `max_frame_width_minus_1` is `frame_width_bits` wide and `frame_width_bits_minus_1` is
        // f(4), so the width field is at most 16 bits and the picture at most 65536 samples.
        if width >= (1 << FRAME_WIDTH_BITS) || height >= (1 << FRAME_WIDTH_BITS) {
            return Err(format!(
                "a {width}x{height} picture: frame_width_bits is f(4)+1, so a dimension above {} cannot be stated",
                (1u32 << FRAME_WIDTH_BITS) - 1
            ));
        }
        Ok(Self { width, height, frames_per_second, order_hint_bits: ORDER_HINT_BITS, level_idx: level_for(width, height, frames_per_second)?, base_qindex })
    }
}

/// `OrderHintBits` this path writes: **7**, so `order_hint` is a 7-bit field.
///
/// **Not derived from the picture.** The reason it is a constant rather than a computed minimum is
/// that the value is a *contract between the sequence header and every frame header in the stream*:
/// changing it after the first picture would re-width `order_hint` in the middle of a coded
/// sequence. Seven is the value [`super::av1_enc::VaEncSequenceParameterBufferAv1::order_hint_bits_minus_1`]
/// carries alongside it (`order_hint_bits_minus_1` = 6), and it holds 128 pictures of distance,
/// which is four times any key-frame interval FilmCraft's settings offer.
pub const ORDER_HINT_BITS: u32 = 7;

/// The `sequence_header_obu()` of spec 5.5, in the OBU a libva AV1 encoder is handed.
///
/// **The whole OBU, not the payload**: `va_enc_av1.h:213-220` says the packed-header data buffer
/// "shall contain the complete `sequence_header_obu()` syntax element", and the syntax element is
/// `open_bitstream_unit()` — header, `obu_size`, payload, trailing bits.
pub fn sequence_header_obu(plan: &Av1Plan) -> Result<Vec<u8>, String> {
    let mut b = Bits::new();
    // `sequence_header_obu()` (spec 5.5.2), read against `filmcraft_av1`'s parser field by field.
    b.f(0, 3)?; // seq_profile: only 0 is advertised for AV1 on this driver
    b.flag(false); // still_picture
    b.flag(false); // reduced_still_picture_header
    b.flag(true); // timing_info_present_flag: the rate is part of the contract with the caller
    // `timing_info()` (spec 5.5.3). `num_units_in_display_tick` is the time unit and `time_scale` is
    // how many of them a second is, so the two together state the rate and neither alone does.
    b.f(DISPLAY_TICK_HZ, 32)?; // num_units_in_display_tick
    let Some(time_scale) = plan.frames_per_second.checked_mul(DISPLAY_TICK_HZ) else {
        return Err(format!("a frame rate of {} does not fit the timing info's 32-bit time_scale", plan.frames_per_second));
    };
    b.f(time_scale, 32)?;
    b.flag(true); // equal_picture_interval
    b.uvlc(0)?; // num_ticks_per_picture_minus_1
    b.flag(false); // decoder_model_info_present_flag
    b.flag(false); // initial_display_delay_present_flag
    b.f(0, 5)?; // operating_points_cnt_minus_1
    b.f(0, 12)?; // operating_point_idc[0]
    b.f(u32::from(plan.level_idx), 5)?; // seq_level_idx[0]
    if plan.level_idx > 7 {
        b.flag(false); // seq_tier[0]
    }
    b.f(FRAME_WIDTH_BITS_MINUS_1, 4)?; // frame_width_bits_minus_1: 15, so frame_width_bits is 16
    b.f(FRAME_WIDTH_BITS_MINUS_1, 4)?; // frame_height_bits_minus_1
    b.f(plan.width - 1, FRAME_WIDTH_BITS)?; // max_frame_width_minus_1, frame_width_bits wide
    b.f(plan.height - 1, FRAME_WIDTH_BITS)?; // max_frame_height_minus_1, frame_height_bits wide
    b.flag(false); // frame_id_numbers_present_flag
    b.flag(false); // use_128x128_superblock: support_128x128_superblock is NOT_SUPPORTED on this driver
    b.flag(false); // enable_filter_intra
    b.flag(false); // enable_intra_edge_filter
    b.flag(false); // enable_interintra_compound
    b.flag(false); // enable_masked_compound
    b.flag(false); // enable_warped_motion
    b.flag(false); // enable_dual_filter
    b.flag(true); // enable_order_hint: the record carries order_hint and order_hint_bits_minus_1
    b.flag(false); // enable_jnt_comp
    b.flag(false); // enable_ref_frame_mvs
    b.flag(true); // seq_choose_screen_content_tools -> SELECT_SCREEN_CONTENT_TOOLS
    b.flag(true); // seq_choose_integer_mv -> SELECT_INTEGER_MV
    b.f(plan.order_hint_bits - 1, 3)?; // order_hint_bits_minus_1
    b.flag(false); // enable_superres
    b.flag(true); // enable_cdef: support_cdef_channel_strength is VA_FEATURE_SUPPORTED
    b.flag(false); // enable_restoration: support_restoration is VA_FEATURE_NOT_SUPPORTED
    // `color_config()` (spec 5.5.2), for 8-bit 4:2:0 BT.709 limited range.
    b.flag(false); // high_bitdepth
    b.flag(false); // mono_chrome
    b.flag(true); // color_description_present_flag
    b.f(consts::CP_BT_709, 8)?;
    b.f(consts::TC_BT_709, 8)?;
    b.f(consts::MC_BT_709, 8)?;
    b.flag(false); // color_range: studio swing, which is what the NV12 upload carries
    // seq_profile == 0 fixes subsampling_x = subsampling_y = 1, so chroma_sample_position is read.
    b.f(consts::CSP_UNKNOWN, 2)?;
    b.flag(false); // separate_uv_delta_q
    b.flag(false); // film_grain_params_present
    b.trailing_bits();
    let payload = b.bytes()?;

    let mut out = Vec::with_capacity(payload.len() + 8);
    out.push(obu_header(OBU_SEQUENCE_HEADER, false, true)?);
    out.extend_from_slice(&leb128(payload.len() as u64));
    out.extend_from_slice(&payload);
    Ok(out)
}

/// The `frame_width_bits_minus_1` this path writes: **15**, so `frame_width_bits` is 16.
///
/// **The widest the field can be, on purpose.** The choice is between a width field wide enough for
/// the largest picture this module will ever be asked for and a narrow one that would refuse 1920
/// as "too wide". Sixteen costs three bits of header per stream and makes the only limit on size the
/// level table, which is where a limit on size belongs.
///
/// **Two constants, not one, and the second is the first plus one** — `frame_width_bits_minus_1` is
/// `f(4)`, so the *value* written is 15 and the *width* of every `max_frame_*_minus_1` field is 16.
/// Writing 15 in a 16-bit field (or 319 in a 15-bit field) does not fail: it costs two bits, the
/// parser reads one pixel too wide, and every later field in the header shifts. That is exactly what
/// the round-trip check caught the first time it ran.
const FRAME_WIDTH_BITS_MINUS_1: u32 = 15;
/// `frame_width_bits`: **16**, the width of `max_frame_width_minus_1` and `max_frame_height_minus_1`.
const FRAME_WIDTH_BITS: u32 = FRAME_WIDTH_BITS_MINUS_1 + 1;

/// The `frame_header_obu()` of spec 6.2.5, in the OBU a libva AV1 encoder is handed.
///
/// **The whole OBU, including the fixed-width `obu_size` field**, for the reason
/// [`leb128_fixed4`] gives.
pub fn frame_header_obu(plan: &Av1Plan, kind: FrameKind) -> Result<Vec<u8>, String> {
    let mut b = Bits::new();
    // `uncompressed_header()` (spec 5.9.2), read against `filmcraft_av1`'s parser field by field.
    let key_frame = matches!(kind, FrameKind::Key { .. });
    b.flag(false); // show_existing_frame
    // `frame_type` is `f(2)` and the specification's own values are used rather than a bool cast:
    // **KEY_FRAME is 0 and INTER_FRAME is 1**, so `u32::from(is_key_frame)` is exactly the wrong way
    // round — and it is a mistake that parses, because a "key frame" written as an inter frame reads
    // back as an inter frame that then has to resolve references the key frame never named. The
    // round-trip check caught this one too.
    b.f(if key_frame { filmcraft_av1::KEY_FRAME as u32 } else { filmcraft_av1::INTER_FRAME as u32 }, 2)?;
    b.flag(true); // show_frame
    // **`showable_frame` is NOT written, and this is the specification's own condition**:
    // `if ( show_frame ) { showable_frame = frame_type != KEY_FRAME } else { showable_frame =
    // read_bits(1) }`. This path always shows its pictures, so the flag is **inferred for every frame
    // type** and no bit is read. Writing it anyway costs one bit — and one bit here is not a one-bit
    // mistake: the bit lands in `error_resilient_mode`, which then makes the decoder read a
    // `ref_order_hint[8]` loop it has no reason to read, and the header runs off into fields this
    // path never wrote. The round trip reported it as "reference to an empty slot", a symptom fifty
    // bits away from the cause.
    if !key_frame {
        b.flag(false); // error_resilient_mode
    }
    // For a key frame `error_resilient_mode` is inferred as 1 and no bit is written.
    b.flag(false); // disable_cdf_update
    b.flag(false); // allow_screen_content_tools: the sequence named SELECT, so the flag is read
    // **`force_integer_mv` is NOT written, and that is the specification's condition rather than an
    // omission**: the flag is read only `if ( allow_screen_content_tools )`, and this path writes
    // that 0, so the decoder infers `force_integer_mv` as 0 and reads no bit. Writing it anyway
    // costs exactly one bit and shifts every field after it -- which is what the round trip caught,
    // and which no size or offset check would have.
    // (For an intra frame `force_integer_mv` is inferred as 1 on top of that.)
    b.flag(false); // frame_size_override_flag: the sequence header states the size
    let (order_hint, reference) = match kind {
        FrameKind::Key { order_hint } => (order_hint, None),
        FrameKind::Inter { order_hint, reference } => (order_hint, Some(reference)),
    };
    b.f(order_hint, plan.order_hint_bits)?;
    // `primary_ref_frame` is in the common part, before the intra/inter split: it is inferred as
    // `PRIMARY_REF_NONE` for an intra or error-resilient frame and read as `f(3)` otherwise.
    if !key_frame {
        b.f(0, 3)?; // primary_ref_frame: the reference named below
        // **`refresh_frame_flags` is in the common part, before the intra/inter split**, not inside
        // the inter branch: the specification reads it right after `primary_ref_frame`, because the
        // `ref_order_hint` loop that follows it is guarded on it. Writing it later costs eight bits
        // of shift and turns every `ref_frame_idx[]` into a different slot — which is what the round
        // trip caught as "reference to an empty slot", a message that names the symptom rather than
        // the cause.
        b.f(REFRESH_LAST_ONLY, 8)?; // refresh_frame_flags: keep the one reference, drop the rest
    }
    if key_frame {
        // `frame_size()` with `frame_size_override_flag` clear takes the size from the sequence
        // header and reads nothing; `superres_params()` reads nothing because `enable_superres` is 0.
        b.flag(false); // render_size(): render_and_frame_size_different
    // `allow_intrabc` is read only when `allow_screen_content_tools` is 1, and this path writes
    // it 0, so nothing is written here. **Every one of these "nothing is written" lines is a
    // condition in the specification, not an omission** — the round trip is what says so.
    } else {
        b.flag(false); // frame_refs_short_signaling: the references are written explicitly
        // The reference slot is an index into `reference_frames[]`, which is `f(3)` and 0..=7, so a
        // value above 7 names no entry and is refused rather than truncated into one that does.
        let slot = reference.ok_or("an inter frame with no reference")?;
        if slot > 7 {
            return Err(format!("a reference_frames[] index of {slot}: ref_frame_idx is f(3), so 0..=7 is every index it can name"));
        }
        for _ in 0..filmcraft_av1::REFS_PER_FRAME {
            b.f(u32::from(slot), 3)?; // ref_frame_idx[i]
        }
        // **`render_size()` is read in this branch too, not only in the intra branch**: the
        // specification reaches `frame_size(); render_size();` on the same path a predicted frame
        // takes once `frame_size_override_flag` is clear. Writing it only for a key frame costs one
        // bit and shifts `cdef_params()` — where `cdef_bits` then reads as 2 instead of 0, the
        // strength loop runs four times instead of once, and the header runs off its own bytes. The
        // round trip named the symptom ("reference to an empty slot") and not the cause, which is
        // why it is written down here.
        b.flag(false); // render_size(): render_and_frame_size_different
        // `frame_size_with_refs()` is reached only when `frame_size_override_flag` is set, so with it
        // clear the size comes from the sequence header and nothing is read.
        b.flag(false); // allow_high_precision_mv
        b.flag(false); // is_filter_switchable
        b.f(0, 2)?; // interpolation_filter: 0 = EIGHTTAP, which the driver reports supported
        b.flag(false); // is_motion_mode_switchable
        // `use_ref_frame_mvs` is **not read**: the sequence header has `enable_ref_frame_mvs` 0, and
        // the specification infers it rather than reading it in that case. Writing it here is the
        // one-bit mistake this path made before the round trip caught it.
    }
    b.flag(false); // disable_frame_end_update_cdf
    tile_info(&mut b, plan)?;
    quantization_params(&mut b, plan)?;
    b.flag(false); // segmentation_enabled
    // `delta_q_params()`: base_q_idx is 0 only for lossless, which this path does not submit.
    if plan.base_qindex > 0 {
        b.flag(false); // delta_q_present
    }
    // `delta_lf_params()` is empty when delta_q_present is 0.
    loop_filter_params(&mut b)?;
    cdef_params(&mut b)?;
    // `lr_params()` reads nothing: `enable_restoration` is 0 in the sequence header.
    if plan.base_qindex > 0 {
        b.flag(true); // tx_mode_select -> TX_MODE_SELECT, the only mode the driver advertises
    }
    if !key_frame {
        b.flag(false); // reference_select: SINGLE_REFERENCE, which is the record's reference_mode 0
    }
    // `skip_mode_params()` reads nothing: `reference_select` is 0, so skip mode is not allowed.
    // `allow_warped_motion` reads nothing: `enable_warped_motion` is 0 in the sequence header.
    b.flag(false); // reduced_tx_set
    if !key_frame {
        // `global_motion_params()`: `is_global` per reference frame, and 0 for all of them is
        // identity, which reads no parameters at all. See the module note on `wmmat`.
        for _ in filmcraft_av1::LAST_FRAME..=(filmcraft_av1::LAST_FRAME + filmcraft_av1::REFS_PER_FRAME - 1) {
            b.flag(false); // is_global
        }
    }
    // `film_grain_params()` reads nothing: `film_grain_params_present` is 0 in the sequence header.
    b.trailing_bits();
    let payload = b.bytes()?;

    let mut out = Vec::with_capacity(payload.len() + 8);
    out.push(obu_header(OBU_FRAME_HEADER, false, true)?);
    out.extend_from_slice(&leb128_fixed4(payload.len() as u32));
    out.extend_from_slice(&payload);
    Ok(out)
}

/// `REFRESH_FRAME_FLAGS` that keeps exactly one reference slot: the LAST slot, bit 0.
///
/// **One reference, and the reason is the measurement the brief carries**: a reference depth above
/// one renders, decodes cleanly and is *wrong from the third picture* on this driver, for both
/// H.264 and HEVC. AV1's refresh flags are what choose which decoded pictures survive to be
/// referred to, so this is where an AV1 submission says "keep the one".
pub const REFRESH_LAST_ONLY: u32 = 1;

/// `tile_info()` (spec 5.9.15) for one tile covering the whole picture.
///
/// **Uniform spacing with one column and one row**, which is the only case where the syntax says the
/// layout outright: with `TileColsLog2 == 0` and `TileRowsLog2 == 0` the decoder derives both counts
/// from the superblock grid, so there is nothing left for the application to get wrong.
///
/// The two `increment_*_log2` bits below are the whole of the variable-length part, and both are
/// written 0 — "stop here" — because `minLog2TileCols` is already 0 for any picture whose width is
/// under `MAX_TILE_WIDTH >> sbSize`.
fn tile_info(b: &mut Bits, plan: &Av1Plan) -> Result<(), String> {
    b.flag(true); // uniform_tile_spacing_flag
    b.flag(false); // increment_tile_cols_log2
    b.flag(false); // increment_tile_rows_log2
    let _ = plan;
    Ok(())
}

/// `quantization_params()` (spec 5.9.17) for one base quantiser and no delta.
///
/// **No `using_qmatrix`**, which is not a style choice: `VaEncPictureParameterBufferAv1::qmatrix_flags`
/// indexes a `VAQMatrixBufferAV1` that libva 1.24.0 does not declare at all, so an application
/// cannot honestly set `using_qmatrix` to 1 on this header — see [`super::av1_enc`]'s fourth open
/// question, which this is the same answer to.
fn quantization_params(b: &mut Bits, plan: &Av1Plan) -> Result<(), String> {
    b.f(u32::from(plan.base_qindex), 8)?; // base_q_idx
    b.flag(false); // DeltaQYDc: delta_coded
    b.flag(false); // DeltaQUDc: delta_coded
    b.flag(false); // DeltaQUAc: delta_coded
    b.flag(false); // using_qmatrix
    Ok(())
}

/// The deblocking levels this path writes, `loop_filter_level[0..4]`.
///
/// **8, 8, 8, 8** — the specification's own resolution scale is 0..63 and 8 is the middle of the
/// range, chosen once here so the picture record's `filter_level[2]`, `filter_level_u` and
/// `filter_level_v` and the bitstream cannot disagree about it. `sharpness` is 0 and the reference
/// and mode deltas are not written at all: `loop_filter_delta_enabled` is 0, which means the decoder
/// uses the specification's defaults and the record's `ref_deltas`/`mode_deltas` are not consulted.
pub const LOOP_FILTER_LEVEL: [u8; 4] = [8, 8, 8, 8];

fn loop_filter_params(b: &mut Bits) -> Result<(), String> {
    for level in LOOP_FILTER_LEVEL {
        b.f(u32::from(level), 6)?;
    }
    b.f(0, 3)?; // loop_filter_sharpness
    b.flag(false); // loop_filter_delta_enabled
    Ok(())
}

/// `cdef_params()` (spec 5.9.18) for damping 3 and no strength at any index.
///
/// **CDEF on with every strength zero**, which is a choice and not an obligation: the driver answers
/// `support_cdef_channel_strength = VA_FEATURE_SUPPORTED` (bit 26 of `VAConfigAttribEncAV1`), so
/// `enable_cdef` is 1 by this path's choice — and the picture record's `cdef_damping_minus_3` is 0,
/// matching the damping this writes.
fn cdef_params(b: &mut Bits) -> Result<(), String> {
    b.f(0, 2)?; // cdef_damping_minus_3 -> CdefDamping = 3
    b.f(0, 2)?; // cdef_bits -> one index
    b.f(0, 4)?; // cdef_y_pri_strength[0]
    b.f(0, 2)?; // cdef_y_sec_strength[0]
    b.f(0, 4)?; // cdef_uv_pri_strength[0]
    b.f(0, 2)?; // cdef_uv_sec_strength[0]
    Ok(())
}

/// Bytes to skip before the `frame_header_obu()` payload: the OBU header byte and the **fixed-width
/// four-byte** `obu_size` field `va_enc_av1.h:925` names.
///
/// **A constant is safe here and not for the sequence header** because this OBU's size field is
/// written four bytes wide by [`leb128_fixed4`]; the sequence header's is a plain `leb128` and is
/// walked instead.
const FRAME_HEADER_OBU_OFFSET: usize = 5;

/// Read the `sequence_header_obu()` this module wrote back with **FilmCraft's own AV1 parser**,
/// and check every field against the plan it was written from.
///
/// This is the whole verification of this file, and what it is worth is stated rather than implied:
/// `filmcraft-av1` parses AV1 bit-exact against libdav1d, so a header that reads back to the
/// values it was written from is a header in the bitstream. **It is not evidence about the driver** —
/// nothing here has been submitted anywhere.
pub fn check_sequence_header(plan: &Av1Plan, obu: &[u8]) -> Result<(), String> {
    // The OBU header byte: forbidden bit clear, type 1, no extension, a size field, reserved clear.
    let Some(&header) = obu.first() else { return Err("a sequence-header OBU with no header byte".into()) };
    if header & 0b1000_0000 != 0 {
        return Err(format!("sequence-header OBU header {header:#010b}: obu_forbidden_bit is set"));
    }
    let obu_type = (header >> 3) & 0xf;
    if obu_type != OBU_SEQUENCE_HEADER {
        return Err(format!("sequence-header OBU header {header:#010b}: obu_type is {obu_type}, not {OBU_SEQUENCE_HEADER}"));
    }
    // **The OBU header's bit positions: forbidden is bit 7, `obu_type` is bits 6-3, the extension flag
    // is bit 2, the size flag is bit 1 and the reserved bit is 0** — `obu_header()` is `f(1), f(4),
    // f(1), f(1), f(1)` written MSB-first, so the two flags are `0b0000_0100` and `0b0000_0010`.
    // Getting these two the wrong way round is a check that passes for the wrong reason: an OBU with
    // *both* flags clear would satisfy the wrong pair.
    if header & 0b0000_0100 != 0 {
        return Err(format!("sequence-header OBU header {header:#010b}: obu_extension_flag is set, so the payload is not where this says it is"));
    }
    if header & 0b0000_0010 == 0 {
        return Err(format!("sequence-header OBU header {header:#010b}: obu_has_size_field is clear, so there is no size to skip"));
    }
    if header & 0b0000_0001 != 0 {
        return Err(format!("sequence-header OBU header {header:#010b}: obu_reserved_1bit is set, and the specification says it must be 0"));
    }
    // **The payload starts after the header and the `leb128` size field, and the size field is
    // variable-width**, so the offset is read rather than written down. A fixed offset here would
    // be a constant that is right for a 1-byte size and wrong for every OBU over 127 bytes — and
    // wrong in the way that hands the parser the size byte as if it were the first payload byte,
    // which parses into a plausible small frame width rather than an error. That is exactly what
    // this check caught the first time it ran.
    let mut at = 1usize;
    let mut size_bytes = 0usize;
    let declared = loop {
        let byte = *obu.get(at).ok_or_else(|| format!("a {}-byte sequence-header OBU ends inside its own size field", obu.len()))?;
        size_bytes += 1;
        if size_bytes > 8 {
            return Err(format!("{} bytes of leb128 with no terminating byte in a sequence-header OBU", size_bytes));
        }
        at += 1;
        if byte & 0x80 == 0 {
            break leb128_value(&obu[1..at])?;
        }
    };
    let payload = obu.get(at..).ok_or_else(|| format!("a {}-byte sequence-header OBU ends inside its own size field", obu.len()))?;
    if declared as usize != payload.len() {
        return Err(format!("the sequence-header OBU's obu_size says {declared} bytes and it carries {}", payload.len()));
    }
    let parsed = filmcraft_av1::header::SequenceHeader::parse(payload).map_err(|e| format!("the sequence header this module wrote does not parse: {e}"))?;

    for (what, got, want) in [
        ("seq_profile", u32::from(parsed.profile), 0u32),
        ("max_frame_width", parsed.max_frame_width, plan.width),
        ("max_frame_height", parsed.max_frame_height, plan.height),
        ("seq_level_idx[0]", u32::from(parsed.seq_level_idx[0]), u32::from(plan.level_idx)),
        ("order_hint_bits", parsed.order_hint_bits, plan.order_hint_bits),
        ("enable_order_hint", u32::from(parsed.enable_order_hint), 1),
        ("enable_cdef", u32::from(parsed.enable_cdef), 1),
        ("enable_restoration", u32::from(parsed.enable_restoration), 0),
        ("enable_superres", u32::from(parsed.enable_superres), 0),
        ("use_128x128_superblock", u32::from(parsed.use_128x128_superblock), 0),
        ("color bit_depth", u32::from(parsed.color.bit_depth), 8),
        ("color num_planes", parsed.color.num_planes as u32, 3),
        ("color color_primaries", u32::from(parsed.color.color_primaries), consts::CP_BT_709),
        ("color transfer_characteristics", u32::from(parsed.color.transfer_characteristics), consts::TC_BT_709),
        ("color matrix_coefficients", u32::from(parsed.color.matrix_coefficients), consts::MC_BT_709),
        ("color subsampling_x", u32::from(parsed.color.subsampling_x), 1),
        ("color subsampling_y", u32::from(parsed.color.subsampling_y), 1),
        ("timing_info_present", u32::from(parsed.timing_info_present), 1),
        ("equal_picture_interval", u32::from(parsed.equal_picture_interval), 1),
        ("film_grain_params_present", u32::from(parsed.film_grain_params_present), 0),
    ] {
        if got != want {
            return Err(format!("the sequence header reads back {what} = {got}, and it was written {want}"));
        }
    }
    if parsed.still_picture || parsed.reduced_still_picture_header {
        return Err("the sequence header reads back as a still-picture or reduced header, and it was written as neither".into());
    }
    if parsed.seq_force_screen_content_tools != consts::SELECT_SCREEN_CONTENT_TOOLS {
        return Err(format!(
            "the sequence header reads back seq_force_screen_content_tools = {}, and it was written {}",
            parsed.seq_force_screen_content_tools,
            consts::SELECT_SCREEN_CONTENT_TOOLS
        ));
    }
    if parsed.seq_force_integer_mv != consts::SELECT_INTEGER_MV {
        return Err(format!(
            "the sequence header reads back seq_force_integer_mv = {}, and it was written {}",
            parsed.seq_force_integer_mv,
            consts::SELECT_INTEGER_MV
        ));
    }
    Ok(())
}

/// The `sequence_header_obu()`'s payload: the bytes after the OBU header and its `leb128` size.
///
/// **A named function rather than an offset**, because the size field is variable-width and a
/// constant offset is a constant that is right for short OBUs and wrong for the rest — see
/// [`check_sequence_header`], which is where that went wrong once.
fn sequence_header_payload(obu: &[u8]) -> Result<&[u8], String> {
    let mut at = 1usize;
    let mut size_bytes = 0usize;
    loop {
        let byte = *obu.get(at).ok_or_else(|| format!("a {}-byte sequence-header OBU ends inside its own size field", obu.len()))?;
        size_bytes += 1;
        if size_bytes > 8 {
            return Err(format!("{size_bytes} bytes of leb128 with no terminating byte in a sequence-header OBU"));
        }
        at += 1;
        if byte & 0x80 == 0 {
            break;
        }
    }
    obu.get(at..).ok_or_else(|| format!("a {}-byte sequence-header OBU ends inside its own size field", obu.len()))
}

/// Read the `frame_header_obu()` this module wrote back with FilmCraft's own AV1 parser, and check
/// every field the records state.
///
/// `reference_state` is what the decoder already holds when the picture is decoded — the eight
/// `RefInfo` slots — and it is an argument rather than something built here because the parse is
/// stateful: `FrameHeader::parse` resolves `ref_frame_idx[]` against it and **refuses a reference
/// to an empty slot**, which is the same check a real decoder makes and therefore the same check
/// worth making here.
pub fn check_frame_header(
    plan: &Av1Plan,
    kind: FrameKind,
    obu: &[u8],
    reference_state: &[filmcraft_av1::header::RefInfo; filmcraft_av1::NUM_REF_FRAMES],
) -> Result<(), String> {
    let payload = obu.get(FRAME_HEADER_OBU_OFFSET..).ok_or_else(|| format!("a {}-byte frame-header OBU is shorter than its own header and size", obu.len()))?;
    let Some(&header) = obu.first() else { return Err("a frame-header OBU with no header byte".into()) };
    if header & 0b1000_0000 != 0 {
        return Err(format!("frame-header OBU header {header:#010b}: obu_forbidden_bit is set"));
    }
    // The same bit positions as the sequence-header check above: extension is bit 2 and the size
    // field is bit 1. See that check for why writing these two down matters.
    if header & 0b0000_0100 != 0 {
        return Err(format!("frame-header OBU header {header:#010b}: obu_extension_flag is set, and this path writes an unextended OBU"));
    }
    if header & 0b0000_0010 == 0 {
        return Err(format!(
            "frame-header OBU header {header:#010b}: obu_has_size_field is clear, and va_enc_av1.h:925-927 says this OBU carries a 4-byte obu_size field"
        ));
    }
    if header & 0b0000_0001 != 0 {
        return Err(format!("frame-header OBU header {header:#010b}: obu_reserved_1bit is set, and the specification says it must be 0"));
    }
    let obu_type = (header >> 3) & 0xf;
    if obu_type != OBU_FRAME_HEADER {
        return Err(format!("frame-header OBU header {header:#010b}: obu_type is {obu_type}, not {OBU_FRAME_HEADER}"));
    }
    // The four-byte obu_size field the libva header names, read back as a `leb128`.
    let size_bytes = obu.get(1..FRAME_HEADER_OBU_OFFSET).ok_or("a frame-header OBU with no size field".to_string())?;
    let size = leb128_value(size_bytes)?;
    if usize::try_from(size).ok() != Some(payload.len()) {
        return Err(format!("the frame-header OBU's obu_size says {size} bytes and it carries {}", payload.len()));
    }
    // The sequence header this frame header was written against, parsed through the **same** check a
    // submission runs, so the frame header is resolved against a sequence header that has itself
    // been read back and compared.
    let sequence_obu = sequence_header_obu(plan)?;
    check_sequence_header(plan, &sequence_obu)?;
    let sequence = filmcraft_av1::header::SequenceHeader::parse(sequence_header_payload(&sequence_obu)?)
        .map_err(|e| format!("the sequence header this frame header was written against does not parse: {e}"))?;

    let owned = reference_state.to_vec();
    let mut refs: [filmcraft_av1::header::RefInfo; filmcraft_av1::NUM_REF_FRAMES] = match owned.try_into() {
        Ok(array) => array,
        Err(_) => return Err("the reference state does not have NUM_REF_FRAMES slots".into()),
    };
    let mut state = filmcraft_av1::header::HeaderState::new(&sequence, &mut refs);
    let mut reader = filmcraft_av1::BitReader::new(payload);
    let parsed = filmcraft_av1::header::FrameHeader::parse(&mut reader, &mut state, 0, 0)
        .map_err(|e| format!("the frame header this module wrote does not parse: {e}"))?;

    match kind {
        FrameKind::Key { order_hint } => {
            if parsed.frame_type != filmcraft_av1::KEY_FRAME as u8 {
                return Err(format!("a key frame-header reads back frame_type = {}, and it was written KEY_FRAME", parsed.frame_type));
            }
            if !parsed.frame_is_intra {
                return Err("a key frame-header reads back as not intra, and a key frame is intra by definition".into());
            }
            if !parsed.error_resilient_mode {
                return Err("a key frame-header reads back error_resilient_mode = false; the syntax infers it as 1 for a shown key frame".into());
            }
            if !parsed.show_frame {
                return Err("a key frame-header reads back show_frame = false, and it was written shown".into());
            }
            if parsed.refresh_frame_flags != (1u32 << filmcraft_av1::NUM_REF_FRAMES) - 1 {
                return Err(format!(
                    "a key frame-header reads back refresh_frame_flags = {:#x}, and the syntax infers allFrames for a shown key frame",
                    parsed.refresh_frame_flags
                ));
            }
            if parsed.order_hint != order_hint {
                return Err(format!("a key frame-header reads back order_hint = {}, and it was written {order_hint}", parsed.order_hint));
            }
        }
        FrameKind::Inter { order_hint, reference } => {
            if parsed.frame_type != filmcraft_av1::INTER_FRAME as u8 {
                return Err(format!("an inter frame-header reads back frame_type = {}, and it was written INTER_FRAME", parsed.frame_type));
            }
            if parsed.order_hint != order_hint {
                return Err(format!("an inter frame-header reads back order_hint = {}, and it was written {order_hint}", parsed.order_hint));
            }
            for (i, got) in parsed.ref_frame_idx.iter().enumerate() {
                if *got != usize::from(reference) {
                    return Err(format!("an inter frame-header reads back ref_frame_idx[{i}] = {got}, and every reference was written as {reference}"));
                }
            }
            if parsed.reference_select {
                return Err("an inter frame-header reads back reference_select = true, and it was written SINGLE_REFERENCE".into());
            }
            if parsed.refresh_frame_flags != REFRESH_LAST_ONLY {
                return Err(format!(
                    "an inter frame-header reads back refresh_frame_flags = {:#x}, and it was written {REFRESH_LAST_ONLY:#x}",
                    parsed.refresh_frame_flags
                ));
            }
            if parsed.frame_width != plan.width || parsed.frame_height != plan.height {
                return Err(format!(
                    "an inter frame-header reads back {}x{} and the picture is {}x{}",
                    parsed.frame_width, parsed.frame_height, plan.width, plan.height
                ));
            }
        }
    }
    for (what, got, want) in [
        ("base_q_idx", parsed.quant.base_q_idx, u32::from(plan.base_qindex)),
        ("loop_filter_level[0]", parsed.lf.level[0], u32::from(LOOP_FILTER_LEVEL[0])),
        ("loop_filter_level[1]", parsed.lf.level[1], u32::from(LOOP_FILTER_LEVEL[1])),
        ("loop_filter_sharpness", parsed.lf.sharpness, 0),
        ("cdef_damping", parsed.cdef.damping, 3),
        ("cdef_bits", parsed.cdef.bits, 0),
        ("tx_mode", u32::from(parsed.tx_mode), consts::TX_MODE_SELECT),
        ("tile cols", parsed.tile_info.cols as u32, 1),
        ("tile rows", parsed.tile_info.rows as u32, 1),
        ("upscaled_width", parsed.upscaled_width, plan.width),
        ("frame_height", parsed.frame_height, plan.height),
    ] {
        if got != want {
            return Err(format!("the frame header reads back {what} = {got}, and it was written {want}"));
        }
    }
    if parsed.quant.using_qmatrix {
        return Err("the frame header reads back using_qmatrix = true, and it was written false: libva 1.24.0 declares no VAQMatrixBufferAV1 for the indexes to point into".into());
    }
    if parsed.seg.enabled {
        return Err("the frame header reads back segmentation_enabled = true, and it was written false".into());
    }
    if parsed.lf.delta_enabled {
        return Err("the frame header reads back loop_filter_delta_enabled = true, and it was written false".into());
    }
    if parsed.allow_warped_motion {
        return Err("the frame header reads back allow_warped_motion = true; the sequence header has enable_warped_motion = 0, so it is inferred false".into());
    }
    if parsed.skip_mode_present {
        return Err("the frame header reads back skip_mode_present = true, and reference_select = 0 forbids it".into());
    }
    // The identity warp: every reference frame's global motion is identity, and the specification
    // gives identity those six parameters. This is the assertion that makes the `wmmat` scale a
    // measured fact rather than a note.
    for reference_frame in filmcraft_av1::LAST_FRAME..(filmcraft_av1::LAST_FRAME + filmcraft_av1::REFS_PER_FRAME) {
        let typ = *parsed.gm_type.get(reference_frame).ok_or("the parsed header has no such reference frame")?;
        if typ != IDENTITY {
            return Err(format!("reference frame {reference_frame} reads back global motion type {typ}, and every one was written identity"));
        }
        let params = *parsed.gm_params.get(reference_frame).ok_or("the parsed header has no such reference frame")?;
        if params != IDENTITY_WARP {
            return Err(format!(
                "reference frame {reference_frame} reads back global motion parameters {params:?}, and the identity warp is {IDENTITY_WARP:?}"
            ));
        }
    }
    // And the record's own end of it: the transform reads no parameters, so the eight `wmmat`
    // entries stay zero.
    let identity_parameters = super::transformation::parameters(super::transformation::IDENTITY);
    if identity_parameters != Some(0) {
        return Err(format!("the identity transform reads {identity_parameters:?} wmmat entries, and this module says it reads none"));
    }
    Ok(())
}

/// The `leb128` value in `bytes`, or why it is not one.
///
/// **Bounded by eight bytes**, which is AV1's own limit on `leb128()` (`03.symbols.md`), so a
/// buffer that is not a `leb128` is refused rather than read past.
fn leb128_value(bytes: &[u8]) -> Result<u32, String> {
    let mut value: u64 = 0;
    for (index, byte) in bytes.iter().copied().enumerate().take(8) {
        value |= u64::from(byte & 0x7f) << (index * 7);
        if byte & 0x80 == 0 {
            return u32::try_from(value).map_err(|_| format!("a leb128 of {value} does not fit a frame header's u32 size field"));
        }
    }
    Err(format!("{} bytes of leb128 with no terminating byte", bytes.len()))
}

/// Whether this tree may assert the `wmmat` scale — the one sentence in this file that is a claim
/// about the specification rather than about the driver, and the reason [`IDENTITY_WARP`] is a
/// constant rather than an arithmetic expression nobody can check.
///
/// [`IDENTITY_WARP`] is not merely equal to the specification's initialiser: it is *the
/// specification's initialiser*, written out. `03.symbols.md:124` gives `WARPEDMODEL_PREC_BITS` as
/// 16 and `global_motion_params()` writes `(i % 3 == 2) ? 1 << WARPEDMODEL_PREC_BITS : 0`, which
/// is `[0, 0, 65536, 0, 0, 65536]` — and that is what `IDENTITY_WARP` says.
pub const WMMAT_SCALE_IS_SPECIFIED: &str = "WARPEDMODEL_PREC_BITS = 16 (av1-spec 03.symbols.md:124) and global_motion_params() \
     initialises gm_params[ref][i] = (i % 3 == 2) ? 1 << WARPEDMODEL_PREC_BITS : 0, i.e. the identity warp is \
     [0, 0, 65536, 0, 0, 65536]. A submission whose global motion is identity reads no wmmat entry at all \
     (is_global = 0 reads nothing), so the scale cannot affect anything this module writes";
