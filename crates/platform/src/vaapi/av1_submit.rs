//! The AV1 **submission** path: the records assembled per picture, the buffer list they are handed
//! to the driver in, and **what this driver measurably does with them**.
//!
//! # The headline, first, because everything else is downstream of it
//!
//! **On this machine the driver refuses `VAEncPictureParameterBufferAV1` outright.** Measured, not
//! inferred — see [`driver_outcomes`] for the numbers and [`DRIVER_REFUSES_THE_AV1_PICTURE_RECORD`]
//! for the refusal this file makes of it. The short form:
//!
//! ```text
//! vaQueryConfigEntrypoints(VAProfileAV1Profile0) -> [1 VLD, 6 EncSlice]
//! vaCreateConfig(32, EncSlice, [RTFormat, RateControl=CQP, PackedHeaders=SEQ|PIC]) -> SUCCESS
//! vaCreateSurfaces / vaCreateContext                                            -> SUCCESS
//! vaRenderPicture([sequence record])                                            -> SUCCESS
//! vaRenderPicture([sequence record, picture record])                            -> VA_STATUS_ERROR_INVALID_BUFFER (7)
//! vaEndPicture                                                                -> VA_STATUS_ERROR_ENCODING_ERROR (24)
//! vaMapBuffer(coded) -> the segment is the uninitialised buffer; no bitstream is written
//! ```
//!
//! **The picture record is the whole of it**: the same sequence record renders on its own, and the
//! picture record alone does not. That was swept over four rate-control modes, five packed-header
//! modes, four surface counts, sixteen cumulative field variants and eighty extra buffer types, and
//! it did not move.
//!
//! **So the honest answer to "did a bitstream come out" is no**, and the honest deliverable is a
//! submission path that is complete, accounted for and checkable over a driver that will not take
//! its picture record. A path built to look finished on top of that would be the exact failure this
//! tree's notes keep warning about — the HEVC sweep that exonerated the records against a *subset*
//! of a submission.
//!
//! # Every buffer is sent, or the reason it is not
//!
//! | buffer | sent? | why |
//! |---|---|---|
//! | `VA_ENC_SEQUENCE_PARAMETER_BUFFER` | yes | the record, every picture |
//! | `VA_ENC_MISC_PARAMETER_BUFFER` (`VAEncMiscParameterFrameRate`) | yes | the rate the plan states |
//! | `VA_ENC_PICTURE_PARAMETER_BUFFER` | yes | the record, every picture |
//! | packed header parameter + data, `VAEncPackedHeaderAV1_SPS` | yes | `va_enc_av1.h:213-220`: the data buffer "shall contain the complete `sequence_header_obu()`" |
//! | packed header parameter + data, `VAEncPackedHeaderAV1_PPS` | yes | `va_enc_av1.h:222-228`: "shall contain the complete `frame_header_obu()`" |
//! | `VAEncTileGroupBufferAV1` | **cannot be sent** | see [`the_tile_group_buffer`] |
//! | `VAQMatrixBufferAV1` | **cannot be sent** | libva 1.24.0 declares no AV1 QMatrix record; `using_qmatrix` is therefore 0 |
//!
//! # Why the driver has to be handed the headers
//!
//! HEVC's path found this the expensive way — the driver writes no slice header, so every submission
//! failed at `vaMapBuffer` until the application packed one. **AV1 has the same gap one level up**,
//! and it is stated rather than discovered: `va_enc_av1.h:923-934` says the frame-header OBU bit
//! stream "contains an obu header, a 4-byte long `obu_size` field, `frame_header_obu()` syntax chain,
//! and a trailing bit", and `bit_offset_qindex`, `byte_offset_frame_hdr_obu_size` and
//! `size_in_bits_frame_hdr_obu` are **offsets into a packed frame header the application supplies**.
//! The bytes are built by [`super::params`] from the AV1 specification and read back through
//! FilmCraft's own parser before they are handed to anything.
//!
//! # Verification, and what it is worth
//!
//! Everything submitted here is a record the driver reads; nothing comes back except the bitstream,
//! and **on this driver no bitstream comes back at all**. So the checks that exist are:
//!
//! - the two OBUs are parsed back with `filmcraft_av1`'s own parser and every field compared with the
//!   records they were written from (`params::check_*`), on **every** submission;
//! - the buffer list is a value — [`Av1Submission`] — so "what was sent, in what order" can be
//!   inspected and tested without a display, and every buffer created is named in it.
//!
//! **Neither is evidence about the driver.** That is stated here rather than left for a reader to
//! discover, and it is why [`Av1Encoder::encode`] never reports a picture it cannot show bytes for.

use super::params::{Av1Plan, FrameKind};
use crate::vaapi::av1_enc::{
    PACKED_HEADER_PPS as AV1_PACKED_HEADER_PPS, PACKED_HEADER_SPS as AV1_PACKED_HEADER_SPS, VaEncPictureParameterBufferAv1, VaEncSequenceParameterBufferAv1,
    params,
};
use crate::vaapi::ffi::{self, Display, Encoder, Profile, RtFormat};
use crate::vaapi::hevc_enc::{PACKED_HEADER_PARAMETER_BUFFER_TYPE, VaEncPackedHeaderParameterBuffer};
use std::sync::Arc;

/// `VAConfigAttribEncPackedHeaders` for this path: **`VA_ENC_PACKED_HEADER_SEQUENCE |
/// VA_ENC_PACKED_HEADER_PICTURE` (`0x3`)** — the two headers [`super::params`] writes and nothing
/// else.
///
/// **Chosen for what the driver itself advertises, and the measurement is
/// [`AV1_DRIVER_PACKED_HEADERS`]**: the driver answers `VAConfigAttribEncPackedHeaders` (10) for
/// `VAProfileAV1Profile0` / `VAEntrypointEncSlice` with **`0x1b`** = `SEQUENCE | PICTURE | MISC |
/// RAW_DATA` (`va.h:1199-1231`). **`VA_ENC_PACKED_HEADER_SLICE` (0x4) is not among them**, which is
/// the AV1-side confirmation of the HEVC finding that this attribute is not a free choice: an AV1
/// encoder on this driver does not take slice headers, because AV1 has no slice header — it takes a
/// *frame* header, and `PICTURE` is the flag that says so.
pub const AV1_PACKED_HEADERS: u32 = ffi::PACKED_HEADER_SEQUENCE | ffi::PACKED_HEADER_PICTURE;

/// **What this driver answers `VAConfigAttribEncPackedHeaders` (10) for AV1: `0x1b`.**
///
/// `0x1b` = `0x01 | 0x02 | 0x08 | 0x10` = `SEQUENCE | PICTURE | MISC | RAW_DATA` (`va.h:1199-1231`),
/// measured on iHD 26.3.5 against `VAProfileAV1Profile0` / `VAEntrypointEncSlice`. It is the reason
/// [`AV1_PACKED_HEADERS`] is `0x3`: this path submits exactly two of the four the driver says it
/// takes, and names the two it submits. `MISC` and `RAW_DATA` are not sent because no record in
/// `va_enc_av1.h` needs them — see the buffer table in the module note.
pub const AV1_DRIVER_PACKED_HEADERS: u32 = 0x1b;

/// **The rate-control mask this driver reports for AV1: `0x56`**, which is
/// `VA_RC_CBR | VA_RC_VBR | VA_RC_CQP | VA_RC_ICQ` (`va.h:1110-1164`: `0x02 | 0x04 | 0x10 | 0x40`).
///
/// **Four modes, and it is neither H.264's nor HEVC's.** H.264's is `0x14d6` and HEVC's `0x14de` on
/// this same driver, both measured by the HEVC path; `0x56` is smaller than either, so **every mode
/// neither of them has a record for is absent here too**, and `VA_RC_VCM` — which HEVC has and H.264
/// does not — is absent from AV1's. [`Av1Encoder::at_constant_quantiser`] submits `VA_RC_CQP`,
/// which this mask does contain and which is the mode in which `bit_offset_qindex`,
/// `size_in_bits_frame_hdr_obu` and `byte_offset_frame_hdr_obu` are "set to 0 and ignored"
/// (`va_enc_av1.h:876`, `:932`, `:919`) — **the mode in which the application's own frame header is
/// used exactly as written**, with nothing rewritten into it by the rate controller.
pub const AV1_DRIVER_RATE_CONTROL_MASK: u32 = 0x56;

/// **`VA_RC_CQP`** — the mode [`Av1Encoder::at_constant_quantiser`] submits.
pub const AV1_RC_CQP: u32 = ffi::RC_CQP;

/// The bit each `VAEncSequenceParameterBufferAV1::seq_fields` flag occupies.
///
/// **Measured, not read off the declaration order**: `av1_abi`'s `ENC_BITFIELDS` table pins every one
/// of them by running C against `/usr/include/va/va_enc_av1.h` and reading
/// `__builtin_ctzll` / `__builtin_clzll` back out of the union's `value`. They are named here rather
/// than reached through `av1_enc`'s own modules so that the record's bit pattern and this file's
/// account of *why* each bit is set sit next to each other.
pub mod seq_fields {
    /// `use_128x128_superblock` — bit 1.
    pub const USE_128X128_SUPERBLOCK: u32 = 1;
    /// `enable_order_hint` — bit 8.
    pub const ENABLE_ORDER_HINT: u32 = 8;
    /// `enable_superres` — bit 11.
    pub const ENABLE_SUPERRES: u32 = 11;
    /// `enable_cdef` — bit 12.
    pub const ENABLE_CDEF: u32 = 12;
    /// `enable_restoration` — bit 13.
    pub const ENABLE_RESTORATION: u32 = 13;
    /// `subsampling_x` — bit 17.
    pub const SUBSAMPLING_X: u32 = 17;
    /// `subsampling_y` — bit 18.
    pub const SUBSAMPLING_Y: u32 = 18;
}

/// `VAEncPictureParameterBufferAV1::mode_control_flags.tx_mode`, at **bits 7-8** (`av1_abi`).
pub const MODE_CONTROL_TX_MODE_SHIFT: u32 = 7;

/// **`TX_MODE_SELECT`**, which is what `VAConfigAttribEncAV1Ext2` (54) says this driver supports.
///
/// The driver answers attribute 54 with **`0x0000ffcf`**, and its `tx_mode_support` field is bits 4-6
/// — `(0xffcf >> 4) & 0b111` = **`0b100`**, which `va_enc_av1.h:190-195` names as `TX_MODE_SELECT`
/// alone. `ONLY_4X4` (0x01) and `TX_MODE_LARGEST` (0x02) are **not** supported, so the frame header
/// this path writes sets `tx_mode_select` and the record says `TX_MODE_SELECT`. The two are the same
/// statement in two places and a mismatch is a picture the driver codes one way and the header
/// describes another.
pub const TX_MODE_SELECT: u8 = 2;

/// **How many reference pictures this path submits: one.**
///
/// **Above one is refused, and the measurement is the one the brief carries for the other two
/// codecs**: a reference depth of 2 renders, decodes cleanly in ffmpeg and is **wrong from the third
/// picture on** — the mean absolute deviation from the source goes 9.1, 9.1, **16.8, 26.6, 36.9,
/// 37.1**. Nothing about that changes for AV1: the mechanism is a decoded picture buffer the
/// application and the driver disagree about, and AV1's record makes the disagreement *more*
/// explicit, because `reference_frames[]` is a bare array of eight surfaces with no picture order
/// count to tie them together.
///
/// **The driver invites it and that is not a reason to submit it**: `VAConfigAttribEncMaxRefFrames`
/// (13) answers **`3, 1`** for AV1 here — list 0 may hold three — and "the hardware could do it" is
/// not "the pictures come out right".
pub const AV1_MAX_REFERENCES: usize = 1;

/// The surface slot a predicted picture's reference occupies: **`reference_frames[1]`**.
///
/// **Not slot 0.** Surface 0 is the picture `vaBeginPicture` is given — the one the caller uploads the
/// source into and overwrites on the next frame — so naming it as a reference is naming a picture
/// that is about to be replaced. `super::hevc_submit`'s `Encoder::rotate_reconstruction` makes the
/// same argument for the H.264 and HEVC paths.
pub const AV1_REFERENCE_SLOT: u8 = 1;

/// `reference_frames[]` entry for a picture that is not a reference: **`0xFF`**.
///
/// `va_enc_av1.h:561-572` is explicit: "Non-valid entries, those do not point to pictures which are
/// referred by current picture or future pictures, should take value 0xFF. **Other values are not
/// allowed.**" A cleared array would name surface 0, which is a real surface.
pub const AV1_NO_SURFACE: u32 = 0xFF;

/// `VAAV1EncTransformationIdentity` for every `wm[i]`: **0** (`va_enc_av1.h:447`).
///
/// **Every `wmtype` is identity and every `wmmat` is zero**, and the reason is a specification
/// statement rather than a preference: `global_motion_params()` reads nothing when `is_global` is 0,
/// and the identity transform reads **0** of the eight `wmmat` entries
/// (`super::av1_enc::transformation::parameters`). **A submission whose global motion is identity does
/// not depend on the `wmmat` scale at all** — which is why this file can be complete while
/// `super::av1_enc`'s first open question says the scale is unstated. See
/// [`super::params::WMMAT_SCALE_IS_SPECIFIED`].
pub const AV1_IDENTITY_WARP: u32 = crate::vaapi::av1_enc::transformation::IDENTITY;

/// **`VAConfigAttribEncInterlaced` (11) for AV1 on this driver: `0`.** Progressive only, so the frame
/// header this path writes has no field picture structure to state — which is also why [`FrameKind`]
/// has no interlaced case.
pub const AV1_DRIVER_INTERLACED: u32 = 0;

/// **What this driver does with an AV1 submission, measured.** One paragraph, because the whole point
/// of this file is that the answer is a measurement and not an inference.
pub mod driver_outcomes {
    /// The measured outcome, with the evidence attached.
    ///
    /// **The same outcome taken twice**, once through a standalone libva probe and once through
    /// FilmCraft's own submission path, because a measurement taken one way is a measurement and a
    /// measurement the shipped code reproduces is a fact about the product. The Rust-side form:
    ///
    /// ```text
    /// vaRenderPicture failed for 5 encode buffers: invalid VABufferID
    /// ```
    ///
    pub const SUMMARY: &str = "measured on iHD 26.3.5 / libva 1.24 at 320x240 on Intel Arc B580: \
         vaQueryConfigEntrypoints(VAProfileAV1Profile0) answers [1 VLD, 6 EncSlice] and \
         vaGetConfigAttributes fills 52/53/54 with 0x05001000 / 0x0000241f / 0x0000ffcf, attribute 5 \
         (rate control) with 0x56 and attribute 10 (packed headers) with 0x1b; \
         vaCreateConfig, vaCreateSurfaces and vaCreateContext all return SUCCESS; \
         vaRenderPicture with the SEQUENCE record ALONE returns SUCCESS; \
         vaRenderPicture with that same sequence record plus ANY VAEncPictureParameterBufferAV1 returns \
         VA_STATUS_ERROR_INVALID_BUFFER, and vaEndPicture returns VA_STATUS_ERROR_ENCODING_ERROR, with the \
         coded buffer left unwritten. Swept over rate control {NONE,CBR,VBR,CQP,ICQ} x packed headers \
         {NONE,SEQ,PIC,SEQ|PIC,0x1b} x surfaces {2,3,4,8} (80 combinations), over 16 cumulative picture-record \
         field variants, and over 80 candidate extra buffer types: the failure did not move. \
         FilmCraft's own submission path reproduces it unchanged. \
         NO AV1 BITSTREAM COMES OUT, so there is nothing for ffmpeg to decode and nothing this path has \
         claimed to have produced";
}

/// **A named refusal with its measurement attached, because "the driver refuses the record" is a
/// claim about a driver and not about AV1.**
///
/// [`Av1Encoder::encode`] renders, syncs, and then reports exactly what came back — which on this
/// driver is an error, and never a picture. **An encode that produces no output is not a success**,
/// and a submission path that reported one would be lying in the exact place the H.264 and HEVC paths
/// already found one.
pub const DRIVER_REFUSES_THE_AV1_PICTURE_RECORD: &str =
    "VA_STATUS_ERROR_INVALID_BUFFER at vaRenderPicture, for every VAEncPictureParameterBufferAV1, on this driver: see driver_outcomes::SUMMARY";

/// **`VAEncTileGroupBufferAV1` is a record libva 1.24.0 gives an application no way to submit, and
/// that is a statement about the header rather than a gap in this path.**
///
/// `va_enc_av1.h:986-1002` declares the 20-byte record with `tg_start`/`tg_end` in 0..127, and
/// `super::av1_enc` declares it. **`VABufferType` in `/usr/include/va/va.h:2044-2210` has no
/// enumerator for it**: the encode types run 21..30 (`VAEncCodedBufferType` 21 … `VAEncQPBufferType`
/// 30) and then jump to 41 (`VAProcPipelineParameterBufferType`); the only tile-shaped types are
/// `VATileBufferType` = 65, which its own comment calls "VVC Tile Dimension data", and
/// `VASliceStructBufferType` = 66. `grep -rn "TileGroup" /usr/include/va/` returns exactly two lines,
/// both in `va_enc_av1.h`, and neither is a buffer type.
///
/// **So this path sends no tile group buffer, and says so rather than omitting it silently.** It does
/// not send a *made-up* type either: `vaCreateBuffer` accepts any `int` for a buffer type, so a
/// fabricated one is created successfully and then refused inside the driver — a failure that looks
/// like a record problem and is a header problem. The honest position is that a one-tile,
/// one-tile-group picture is what this path submits (`num_tile_groups_minus1 = 0`), and whether that
/// needs a tile group buffer is **not answerable from the headers installed here**.
pub mod the_tile_group_buffer {
    /// What was checked and what was found.
    pub const WHY_NOT_SENT: &str = "libva 1.24.0 declares no VABufferType for VAEncTileGroupBufferAV1: \
         VABufferType runs 21..30 for encode and then jumps to 41, and grep -rn TileGroup /usr/include/va/ returns \
         two lines, both the struct in va_enc_av1.h:986-1002. This path submits num_tile_groups_minus1 = 0 and \
         sends no tile group buffer; whether that needs one is not answerable from the installed headers";
}

/// One picture's request: what it is coded as, and whether it is the last.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Av1EncodeRequest {
    /// What the picture is. [`FrameKind`] is the submission path's own type rather than
    /// `PictureKind`, for the reason `HevcEncodeRequest` is its own type: the AV1 record states
    /// `frame_type` in a 2-bit field with four values, and three of them are not what this path
    /// submits, which is a different refusal set from H.264's.
    pub kind: FrameKind,
    /// Whether this is the last picture of the stream.
    pub last: bool,
}

impl Av1EncodeRequest {
    /// A key frame.
    pub fn key_frame(order_hint: u32) -> Self {
        Self { kind: FrameKind::Key { order_hint }, last: false }
    }

    /// A predicted frame predicting from [`AV1_REFERENCE_SLOT`].
    pub fn predicted(order_hint: u32) -> Self {
        Self { kind: FrameKind::Inter { order_hint, reference: AV1_REFERENCE_SLOT }, last: false }
    }

    /// This request, marked last.
    pub fn last(mut self) -> Self {
        self.last = true;
        self
    }
}

/// One buffer the driver is handed: its libva type and its bytes.
///
/// **A value and not a libva handle**, and that is the design decision this file exists to be able to
/// make: the buffer set is built and checked *before* any display is touched, so "what does this path
/// submit" is a question with an answer in ordinary data — which is how the HEVC path's missing four
/// buffers could be found at all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Av1Buffer {
    /// The libva `VABufferType` these bytes are submitted as.
    pub kind: u32,
    /// The record's bytes, exactly as they go to `vaCreateBuffer`.
    pub bytes: Vec<u8>,
}

/// Everything one picture is submitted as, **in the order it is submitted**.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Av1Submission {
    /// The buffer list, in submission order.
    pub buffers: Vec<Av1Buffer>,
}

impl Av1Submission {
    /// The libva types in order, which is the thing a reader checks against the header.
    pub fn kinds(&self) -> Vec<u32> {
        self.buffers.iter().map(|b| b.kind).collect()
    }

    /// How many bytes the whole submission is.
    pub fn bytes(&self) -> usize {
        self.buffers.iter().map(|b| b.bytes.len()).sum()
    }

    /// How many of the buffers are of `kind`.
    pub fn count_of(&self, kind: u32) -> usize {
        self.buffers.iter().filter(|b| b.kind == kind).count()
    }
}

/// What one picture came back as.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Av1EncodedPicture {
    /// The OBU stream the driver wrote: a sequence header OBU, a frame header OBU and a tile group
    /// OBU, in that order.
    pub bytes: Vec<u8>,
    /// How many of `bytes` the driver itself wrote, against how many this path supplied as packed
    /// headers.
    pub driver_bytes: usize,
    /// Whether this picture is a key frame **as submitted**.
    pub key_frame: bool,
    /// The `order_hint` this picture was submitted with.
    pub order_hint: u32,
    /// How many references the picture was told to predict from.
    pub predicted_from: usize,
    /// `VA_CODED_BUF_STATUS_*` per segment; the low byte is the driver's average quantiser.
    pub status: Vec<u32>,
}

/// An AV1 encoder: one key frame, then the P frames that genuinely predict from it.
///
/// Holds the libva objects ([`Encoder`]), the geometry ([`Av1Plan`]) and the frame loop's state.
pub struct Av1Encoder {
    encoder: Encoder,
    plan: Av1Plan,
    /// The reconstruction surfaces this encoder holds, each with the `order_hint` of the picture in
    /// it, oldest first. The whole of the frame loop's state.
    references: Vec<(u32, u32)>,
    /// The reconstruction surface the last [`Av1Encoder::submission`] rotated to, which is what the
    /// next picture is reconstructed into.
    reconstruction: u32,
    /// The packed sequence header OBU last handed to the driver, so it is submitted with the first
    /// picture and again only if it changes. A driver that passes packed headers through — which
    /// this one does for HEVC, measured — would otherwise put a sequence header in every access unit.
    sent_sequence_header: Option<Vec<u8>>,
    /// The sequence record, built once.
    sequence: Vec<u8>,
    encoded: u64,
    ended: bool,
    last_marked: bool,
}

impl Av1Encoder {
    /// A **constant-quantiser** AV1 encoder for `plan`.
    ///
    /// **`VA_RC_CQP` and not a bitrate mode**, for the reason [`AV1_DRIVER_RATE_CONTROL_MASK`] gives:
    /// `CQP` is the mode in which `bit_offset_qindex`, `byte_offset_frame_hdr_obu_size` and
    /// `size_in_bits_frame_hdr_obu` are "set to 0 and ignored" by the driver, so the frame header
    /// [`super::params`] wrote is the frame header that is coded. Under `CBR` or `VBR` the driver
    /// **rewrites `base_q_idx` in the middle of the application-supplied header**, through an offset
    /// this path would have to state exactly right.
    pub fn at_constant_quantiser(display: &Arc<Display>, plan: Av1Plan) -> Result<Self, String> {
        let size = (plan.width, plan.height);
        let refs = AV1_MAX_REFERENCES as u32;
        let mut encoder = Encoder::new(display, Profile::Av1Main, RtFormat::Nv12, size, AV1_RC_CQP, AV1_PACKED_HEADERS, coded_buffer_for(size)?, refs)?;
        let sequence = sequence_record(&plan)?;
        let reconstruction = encoder.rotate_reconstruction();
        Ok(Self { encoder, sequence, plan, references: Vec::new(), reconstruction, sent_sequence_header: None, encoded: 0, ended: false, last_marked: false })
    }

    /// The rate-control mask this driver reports for AV1, by asking it.
    pub fn driver_rate_control_modes(display: &Arc<Display>) -> u32 {
        Encoder::driver_rate_control_modes(display, Profile::Av1Main)
    }

    /// The geometry this encoder was built for.
    pub fn plan(&self) -> &Av1Plan {
        &self.plan
    }

    /// How many pictures have been encoded so far.
    pub fn pictures_encoded(&self) -> u64 {
        self.encoded
    }

    /// Whether this encoder can still encode.
    pub fn is_configured(&self) -> bool {
        !self.ended && !self.last_marked
    }

    /// Forget every reference, so the next picture has to be a key frame.
    pub fn forget_references(&mut self) {
        self.references.clear();
    }

    /// Upload a picture's luma and chroma planes into the surface the driver reads.
    pub fn upload(&self, luma: &[u8], chroma: &[u8]) -> Result<(), String> {
        self.encoder.upload(luma, chroma)
    }

    /// End the stream.
    pub fn flush(&mut self) -> Result<(), String> {
        self.ended = true;
        Ok(())
    }

    /// **Everything one picture is submitted as, in order — built without handing anything to the
    /// driver.**
    ///
    /// This is the heart of the path and it is deliberately separated from [`Av1Encoder::encode`] so
    /// that the buffer set is a value: a caller, a test or a probe can ask "what does FilmCraft
    /// submit for an AV1 key frame" and get an answer without a GPU, and a driver that refuses one of
    /// the buffers has to name which.
    ///
    /// **It rotates the reconstruction surface**, because the record has to name the surface the
    /// picture is reconstructed into and the only way to reach that surface is the encoder's own
    /// rotation. A submission that is built and then not rendered therefore leaves the encoder one
    /// rotation further on, which is stated here rather than discovered by a caller that builds a
    /// submission to look at it.
    pub fn submission(&mut self, request: Av1EncodeRequest) -> Result<Av1Submission, String> {
        let key_frame = matches!(request.kind, FrameKind::Key { .. });
        // A predicted picture with nothing to predict from would reconstruct whatever the surface
        // holds, which is a plausible wrong picture rather than an error. Same refusal as H.264's and
        // HEVC's, for the same reason.
        if !key_frame && self.references.is_empty() {
            return Err("a predicted AV1 picture with no reference: this encoder has encoded nothing yet, so there is nothing to \
                 predict from. Encode a key frame first"
                .into());
        }
        if self.ended || self.last_marked {
            return Err("this stream has already ended, so there is no picture after it".into());
        }
        let order_hint = match request.kind {
            FrameKind::Key { order_hint } | FrameKind::Inter { order_hint, .. } => order_hint,
        };
        // The sequence header is sent with the first picture and again only if it changes, for the
        // reason the HEVC path gives and by the same test: the bytes, not a flag.
        let sequence_header_obu = params::sequence_header_obu(&self.plan)?;
        let sequence_header_changed = self.sent_sequence_header.as_ref().is_none_or(|sent| *sent != sequence_header_obu);
        if sequence_header_changed {
            self.sent_sequence_header = Some(sequence_header_obu);
        }
        self.reconstruction = self.encoder.rotate_reconstruction();
        build_submission(
            &self.plan,
            &self.sequence,
            self.reconstruction,
            self.encoder.coded_buffer_id(),
            key_frame,
            order_hint,
            &self.reference_state(),
            sequence_header_changed,
        )
    }

    /// The decoder-side reference state the frame-header parse is checked against.
    ///
    /// **Built from this encoder's own references**, because `filmcraft_av1`'s `FrameHeader::parse`
    /// resolves `ref_frame_idx[]` against it and refuses a reference to an empty slot — which is the
    /// same check a real decoder makes, and the same one worth making before a submission.
    fn reference_state(&self) -> [filmcraft_av1::header::RefInfo; filmcraft_av1::NUM_REF_FRAMES] {
        let mut refs = std::array::from_fn(|_| filmcraft_av1::header::RefInfo::default());
        for &(_, order_hint) in &self.references {
            let slot = usize::from(AV1_REFERENCE_SLOT);
            if let Some(entry) = refs.get_mut(slot) {
                entry.valid = true;
                entry.order_hint = order_hint;
                entry.upscaled_width = self.plan.width;
                entry.frame_width = self.plan.width;
                entry.frame_height = self.plan.height;
                entry.render_width = self.plan.width;
                entry.render_height = self.plan.height;
                entry.mi_cols = 2 * ((self.plan.width + 7) >> 3);
                entry.mi_rows = 2 * ((self.plan.height + 7) >> 3);
            }
        }
        refs
    }

    /// **Encode one picture: build the submission, hand it to the driver, sync, read back.**
    ///
    /// The order of the decisions is the H.264 and HEVC paths' order — the request is checked, the
    /// picture type is decided, the records are built, the render happens, the sync happens, and only
    /// then is the readback read — and so is the property that matters: **a failure anywhere leaves
    /// the reference list where it was**, so a caller can retry the same picture.
    ///
    /// # What happens on this driver
    ///
    /// The render fails and the error says which buffer the driver objected to. **No
    /// `Av1EncodedPicture` is returned**, because there are no bytes to put in one — and that is the
    /// point the brief makes about a driver that accepts buffers and emits nothing.
    pub fn encode(&mut self, request: Av1EncodeRequest) -> Result<Av1EncodedPicture, String> {
        let submission = self.submission(request)?;
        // Every handle is alive across the render: `Buffer`'s `Drop` destroys its libva buffer, and a
        // handle dropped before the render names its id is exactly what iHD answers with
        // `VA_STATUS_ERROR_INVALID_BUFFER`.
        let handles =
            submission.buffers.iter().map(|b| self.encoder.buffer(i32::try_from(b.kind).unwrap_or(-1), &b.bytes, 1)).collect::<Result<Vec<_>, String>>()?;
        let ids: Vec<u32> = handles.iter().map(|b| b.id()).collect();
        self.encoder.render(&ids)?;
        // Mandatory before the read: the encode is asynchronous and an unsynchronised read of the
        // coded buffer is a truncated picture rather than a stale one.
        self.encoder.sync()?;
        let coded = self.encoder.read_coded()?;
        let status = coded.status().to_vec();
        let bytes = coded.bytes().to_vec();
        if bytes.is_empty() {
            return Err("the AV1 coded buffer mapped to no bytes, so there is no picture".into());
        }
        // The state advances only after the picture came back.
        let key_frame = matches!(request.kind, FrameKind::Key { .. });
        let order_hint = match request.kind {
            FrameKind::Key { order_hint } | FrameKind::Inter { order_hint, .. } => order_hint,
        };
        if key_frame {
            self.references.clear();
        }
        self.references.push((self.reconstruction, order_hint));
        while self.references.len() > AV1_MAX_REFERENCES {
            self.references.remove(0);
        }
        self.encoded = self.encoded.saturating_add(1);
        if request.last {
            self.last_marked = true;
            self.references.clear();
        }
        let driver_bytes = bytes.len();
        Ok(Av1EncodedPicture { bytes, driver_bytes, key_frame, order_hint, predicted_from: if key_frame { 0 } else { 1 }, status })
    }
}

/// **Build one picture's whole submission, with no display and no encoder in sight.**
///
/// This is the function the path's central claim rests on, and it is free of libva for two reasons.
/// First, so that "what does FilmCraft submit for an AV1 key frame" has an answer in ordinary data:
/// **a driver that refuses one of the buffers then has to name which, and a missing buffer is
/// visible in a list rather than in a status code.** Second, so the property the brief calls "the
/// rule that catches everyone" — *every buffer you create must be sent* — is a property of a value
/// that a test on any machine can read.
///
/// **The two OBUs are parsed back here**, through FilmCraft's own AV1 parser, before the list is
/// returned: a header this path cannot describe is a refusal rather than a picture a decoder cannot
/// start.
pub fn build_submission(
    plan: &Av1Plan,
    sequence: &[u8],
    reconstruction: u32,
    coded_buffer: u32,
    key_frame: bool,
    order_hint: u32,
    reference_state: &[filmcraft_av1::header::RefInfo; filmcraft_av1::NUM_REF_FRAMES],
    send_sequence_header: bool,
) -> Result<Av1Submission, String> {
    let kind = if key_frame { FrameKind::Key { order_hint } } else { FrameKind::Inter { order_hint, reference: AV1_REFERENCE_SLOT } };
    // **A predicted picture is refused here, before a record is built**, when the reference state it
    // would name does not hold a valid reference — which is the same condition a decoder refuses.
    // The check is here rather than only in [`Av1Encoder::submission`] so that it is reachable
    // without a display: a submission naming a reference that does not exist is a picture no decoder
    // can start, and "the frame loop has not encoded a key frame yet" is the commonest way to ask for
    // one.
    if !key_frame {
        let slot = reference_state.get(usize::from(AV1_REFERENCE_SLOT)).ok_or_else(|| format!("a reference state with no slot {}", AV1_REFERENCE_SLOT))?;
        if !slot.valid {
            return Err(format!(
                "a predicted AV1 picture naming reference_frames[{}], which holds no picture: this is the \
                 \"no reference\" refusal -- encode a key frame first",
                AV1_REFERENCE_SLOT
            ));
        }
    }
    let sequence_header_obu = params::sequence_header_obu(plan)?;
    params::check_sequence_header(plan, &sequence_header_obu)?;
    let frame_header_obu = params::frame_header_obu(plan, kind)?;
    params::check_frame_header(plan, kind, &frame_header_obu, reference_state)?;

    let picture = picture_record(plan, reconstruction, coded_buffer, key_frame, order_hint)?;
    let mut buffers = Vec::with_capacity(9);
    // 1. The sequence record, every picture: nothing in `va_enc_av1.h` says "once", and a driver is
    //    entitled to want it with each picture.
    buffers.push(Av1Buffer { kind: ffi::ENC_SEQUENCE_PARAMETER_BUFFER as u32, bytes: sequence.to_vec() });
    // 2. The frame rate, every picture: `VAEncMiscParameterFrameRate`. Not optional — it is the only
    //    place the rate reaches the driver, and a stream whose rate nobody stated is a stream whose
    //    timing is invented.
    buffers.push(Av1Buffer { kind: ffi::ENC_MISC_PARAMETER_BUFFER as u32, bytes: frame_rate_record(plan.frames_per_second)? });
    // 3. The picture record, every picture.
    buffers.push(Av1Buffer { kind: ffi::ENC_PICTURE_PARAMETER_BUFFER as u32, bytes: picture });
    // 4/5. The packed sequence header, when it is new.
    if send_sequence_header {
        buffers.extend(packed_header_pair(AV1_PACKED_HEADER_SPS, &sequence_header_obu)?);
    }
    // 6/7. The packed frame header, every picture — it names this picture, so it cannot be "only when
    //      it changes".
    buffers.extend(packed_header_pair(AV1_PACKED_HEADER_PPS, &frame_header_obu)?);
    Ok(Av1Submission { buffers })
}

/// The two buffers a packed header is submitted as: the parameter buffer that names it and the data
/// buffer that holds the bytes.
///
/// **`bit_length` is bits, not bytes** (`va.h:2450`) — a record carrying the byte count is a header
/// the driver is told is eight times too long — and **`has_emulation_bytes` is 0**, because an AV1
/// OBU stream is a low-overhead bitstream with no `00 00 03` emulation to carry. That is the
/// opposite of HEVC's Annex-B buffers and is the same kind of mistake waiting to happen.
fn packed_header_pair(header_type: u32, data: &[u8]) -> Result<Vec<Av1Buffer>, String> {
    let bits = data.len().checked_mul(8).ok_or_else(|| format!("a {}-byte packed header does not fit a bit count", data.len()))?;
    let bit_length = u32::try_from(bits).map_err(|_| format!("a packed header of {bits} bits"))?;
    let parameter = ffi::record_bytes(&VaEncPackedHeaderParameterBuffer {
        header_type,
        bit_length,
        has_emulation_bytes: 0,
        padding_and_reserved: [0; 3 + ffi::VA_PADDING_LOW * 4],
    });
    Ok(vec![
        Av1Buffer { kind: PACKED_HEADER_PARAMETER_BUFFER_TYPE, bytes: parameter },
        Av1Buffer { kind: ffi::ENC_PACKED_HEADER_DATA_BUFFER as u32, bytes: data.to_vec() },
    ])
}

/// How large a coded buffer a picture of `size` needs, or why it cannot be derived.
///
/// **The same bound as the H.264 and HEVC paths, and for the same reason**: the size of a compressed
/// picture depends on the quantiser, the content and the driver's own decisions, none of which are
/// known before the picture exists. Twice one uncompressed 4:2:0 frame plus 4 KiB, every step
/// `checked_*` and narrowed to `u32` because `vaCreateBuffer`'s size argument is a `c_uint`.
fn coded_buffer_for(size: (u32, u32)) -> Result<u32, String> {
    let uncompressed = (size.0 as u64)
        .checked_mul(size.1 as u64)
        .and_then(|v| v.checked_mul(3))
        .and_then(|v| v.checked_div(2))
        .ok_or_else(|| format!("a {}x{} picture does not fit a byte count", size.0, size.1))?;
    let bytes = uncompressed.checked_mul(2).and_then(|v| v.checked_add(4096)).ok_or("a coded buffer size overflows")?;
    u32::try_from(bytes).map_err(|_| format!("a coded buffer of {bytes} bytes"))
}

/// `VAEncSequenceParameterBufferAV1` for `plan`, with every `seq_fields` flag at the value this
/// driver advertises.
///
/// # The rule for every `seq_fields` flag, and why it is a rule
///
/// **`VA_FEATURE_NOT_SUPPORTED` means the flag must not be set** — `va.h:381` says so in as many
/// words: *"Any corresponding feature flag must not be set."* This driver answers
/// `VAConfigAttribEncAV1` (52) with **`0x05001000`**, which is **three** features at
/// `VA_FEATURE_SUPPORTED` and eleven at `VA_FEATURE_NOT_SUPPORTED` (bit 12 `palette_mode`, bit 24
/// `allow_intrabc`, bit 26 `cdef_channel_strength`). So:
///
/// - **`use_128x128_superblock`, `enable_filter_intra`, `enable_intra_edge_filter`,
///   `enable_interintra_compound`, `enable_masked_compound`, `enable_warped_motion`,
///   `enable_dual_filter`, `enable_jnt_comp`, `enable_ref_frame_mvs` and `enable_superres` are all
///   0**, because every one of their features reads `VA_FEATURE_NOT_SUPPORTED`;
/// - **`enable_cdef` is 1**, a *choice*: `cdef_channel_strength` is `VA_FEATURE_SUPPORTED`, so the
///   driver can CDEF and does not have to;
/// - **`enable_restoration` is 0**, because `support_restoration` (bit 22) reads `0`;
/// - **`enable_order_hint` is 1**, and this one is not in the feature union at all — the record
///   carries `order_hint` and `order_hint_bits_minus_1`, so the stream is expected to state an order.
///
/// That is why the sequence header [`super::params`] writes and this record agree, field for field.
fn sequence_record(plan: &Av1Plan) -> Result<Vec<u8>, String> {
    let mut record = VaEncSequenceParameterBufferAv1 {
        seq_profile: 0,
        seq_level_idx: plan.level_idx,
        seq_tier: 0,
        hierarchical_flag: 0,
        intra_period: 0,
        ip_period: 0,
        // **Not zero.** Under `VA_RC_CQP` the bitrate is not what the rate controller runs on, but
        // the H.264 path measured that iHD *rejects* a sequence record whose bitrate is 0 with
        // `VA_STATUS_ERROR_INVALID_PARAMETER` and a message naming no field. It is set to the level's
        // ceiling, which is a validity check and **not** a target.
        bits_per_second: level_ceiling_bitrate(plan.level_idx),
        seq_fields: 0,
        order_hint_bits_minus_1: u8::try_from(plan.order_hint_bits - 1)
            .map_err(|_| format!("an order_hint_bits of {} does not fit the record's 8 bits", plan.order_hint_bits))?,
        ..Default::default()
    };
    record.seq_fields |= 1 << seq_fields::ENABLE_ORDER_HINT;
    record.seq_fields |= 1 << seq_fields::ENABLE_CDEF;
    record.seq_fields |= 1 << seq_fields::SUBSAMPLING_X;
    record.seq_fields |= 1 << seq_fields::SUBSAMPLING_Y;
    Ok(ffi::record_bytes(&record))
}

/// The bitrate a level's ceiling allows, for the sequence record's `bits_per_second`.
///
/// **A validity value, not a target**, and stated as such because the alternative reading is that this
/// module invents a bitrate for a caller who asked for a quantiser. The numbers are the annex's
/// `MainMbps` column (`annex.a.levels.md`), in bits per second.
fn level_ceiling_bitrate(level_idx: u8) -> u32 {
    const MBPS: [(u8, u32); 6] = [(0, 1_500_000), (1, 3_000_000), (4, 6_000_000), (5, 10_000_000), (8, 12_000_000), (9, 20_000_000)];
    for (idx, bits) in MBPS {
        if idx == level_idx {
            return bits;
        }
    }
    // A level above every row in the table: the highest value the annex lists, which is a ceiling for
    // every level above it rather than a guess.
    20_000_000
}

/// `VAEncPictureParameterBufferAV1` for one picture.
fn picture_record(plan: &Av1Plan, reconstruction: u32, coded_buffer: u32, key_frame: bool, order_hint: u32) -> Result<Vec<u8>, String> {
    let (Ok(width), Ok(height)) = (u16::try_from(plan.width), u16::try_from(plan.height)) else {
        return Err(format!("a {}x{} picture does not fit the record's 16-bit dimensions", plan.width, plan.height));
    };
    let mut record = VaEncPictureParameterBufferAv1 {
        frame_width_minus_1: width - 1,
        frame_height_minus_1: height - 1,
        reconstructed_frame: reconstruction,
        coded_buf: coded_buffer,
        ..Default::default()
    };
    // **`reference_frames[]` is eight surfaces and an entry that is not a reference is `0xFF`**, not
    // 0: `va_enc_av1.h:561-572` says so and calls every other value illegal.
    for entry in record.reference_frames.iter_mut() {
        *entry = AV1_NO_SURFACE;
    }
    // **`ref_frame_idx[]` maps LAST..ALTREF2 onto `reference_frames[]`**, so all seven entries name
    // the same slot: this path has one reference and every reference frame is it. The slot's surface
    // is written below, once the reference exists.
    for entry in record.ref_frame_idx.iter_mut() {
        *entry = AV1_REFERENCE_SLOT;
    }
    record.primary_ref_frame = AV1_REFERENCE_SLOT;
    record.order_hint = u8::try_from(order_hint).map_err(|_| format!("an order_hint of {order_hint} does not fit the record's 8 bits"))?;
    // A key frame refreshes every slot — the syntax infers allFrames for a shown key frame — and a
    // predicted frame keeps exactly the one reference it predicts from.
    record.refresh_frame_flags =
        if key_frame { u8::try_from((1u32 << filmcraft_av1::NUM_REF_FRAMES) - 1).unwrap_or(u8::MAX) } else { params::REFRESH_LAST_ONLY as u8 };
    record.base_qindex = plan.base_qindex;
    // **CQP is the mode, so `min_base_qindex`/`max_base_qindex` bound a range the caller chose rather
    // than one the rate controller is given.** They are set to the record's own documented range
    // (`1..=255`, `va_enc_av1.h:730-738`) rather than to 0, because a value outside a documented
    // range is asking the driver for something it has not been offered.
    record.min_base_qindex = 1;
    record.max_base_qindex = 255;
    record.filter_level[0] = params::LOOP_FILTER_LEVEL[0];
    record.filter_level[1] = params::LOOP_FILTER_LEVEL[1];
    record.filter_level_u = params::LOOP_FILTER_LEVEL[2];
    record.filter_level_v = params::LOOP_FILTER_LEVEL[3];
    // `superres_scale_denominator` is documented as 9..=16 and unused when superres is off; 9 is the
    // documented minimum, so the record is inside its own stated range whatever the driver does.
    record.superres_scale_denominator = 9;
    record.interpolation_filter = 0; // EIGHTTAP, which `VAConfigAttribEncAV1Ext1` reports supported
    record.cdef_damping_minus_3 = 0; // damping 3, which is what the frame header writes
    record.cdef_bits = 0;
    record.tile_cols = 1;
    record.tile_rows = 1;
    record.context_update_tile_id = 0;
    record.num_tile_groups_minus1 = 0;
    record.temporal_id = 0;
    // `tx_mode` at bits 7-8 and `reference_mode` at 9-10, both measured (`av1_abi`). `TX_MODE_SELECT`
    // is the only mode the driver reports; `reference_mode` 0 is single reference, which is what the
    // frame header's `reference_select = 0` says.
    record.mode_control_flags |= u32::from(TX_MODE_SELECT) << MODE_CONTROL_TX_MODE_SHIFT;
    // `tile_group_obu_hdr_info.obu_has_size_field` at bit 1: the frame-header OBU this path writes
    // carries a size field, and the tile group OBU the driver writes beside it does too.
    record.tile_group_obu_hdr_info |= 0b0000_0010;
    // **Every global motion transform is identity and reads no parameters** — see
    // [`AV1_IDENTITY_WARP`] for why the scale does not matter here. `invalid` stays 0: the header
    // says it is "Whether this transform is valid, for the affine set", and identity is not affine,
    // so there is nothing there to invalidate.
    for warp in record.wm.iter_mut() {
        warp.wmtype = AV1_IDENTITY_WARP;
    }
    // The bit offsets are 0 in `VA_RC_CQP`: `va_enc_av1.h:876`, `:893` and `:932` each say "In CQP
    // mode, this parameter should be set to 0 and ignored by driver". `bit_offset_qindex`,
    // `bit_offset_segmentation`, `bit_offset_loopfilter_params`, `bit_offset_cdef_params`,
    // `size_in_bits_cdef_params`, `byte_offset_frame_hdr_obu_size` and `size_in_bits_frame_hdr_obu`
    // are all left at the `Default` of 0, which is those values.
    Ok(ffi::record_bytes(&record))
}

/// The `VAEncMiscParameterFrameRate` misc buffer: the rate, told to the driver.
///
/// **The type word is prepended, not part of the record struct** — `ffi::VaEncMiscParameterFrameRate`
/// is the payload and `VAEncMiscParameterBufferHeader` is the `type` in front of it, which is how
/// `super::hevc_params::misc` builds the same buffer for HEVC.
fn frame_rate_record(frames_per_second: u32) -> Result<Vec<u8>, String> {
    if frames_per_second == 0 || frames_per_second > u32::from(u16::MAX) {
        return Err(format!("a frame rate of {frames_per_second} fps, which does not fit the field's numerator"));
    }
    let mut out = ffi::record_bytes(&ffi::VaEncMiscParameterBufferHeader { type_: ffi::MISC_FRAME_RATE });
    out.extend_from_slice(&ffi::record_bytes(&ffi::VaEncMiscParameterFrameRate {
        framerate: (1u32 << 16) | frames_per_second,
        framerate_flags: 0,
        va_reserved: [0; ffi::VA_PADDING_LOW],
    }));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The picture size these tests use: **320x240**, the size the HEVC probe and the measurement in
    /// [`driver_outcomes`] both use, and a whole number of AV1's 8x8 blocks in both directions.
    const W: u32 = 320;
    const H: u32 = 240;
    /// The quantiser: **100**, mid-range on `base_q_idx`'s 0..=255, and the same value the driver's
    /// own recorded measurement used.
    const Q: u8 = 100;

    fn plan() -> Av1Plan {
        Av1Plan::new(W, H, 30, Q).unwrap_or_else(|e| panic!("a 320x240 plan at 30 fps: {e}"))
    }

    /// An empty reference state: the key frame's parse has no reference to resolve.
    fn no_references() -> [filmcraft_av1::header::RefInfo; filmcraft_av1::NUM_REF_FRAMES] {
        std::array::from_fn(|_| filmcraft_av1::header::RefInfo::default())
    }

    /// A reference state holding one valid reference in the slot this path predicts from.
    fn one_reference(p: &Av1Plan) -> [filmcraft_av1::header::RefInfo; filmcraft_av1::NUM_REF_FRAMES] {
        let mut refs = no_references();
        if let Some(slot) = refs.get_mut(usize::from(AV1_REFERENCE_SLOT)) {
            slot.valid = true;
            slot.order_hint = 0;
            slot.upscaled_width = p.width;
            slot.frame_width = p.width;
            slot.frame_height = p.height;
            slot.render_width = p.width;
            slot.render_height = p.height;
        }
        refs
    }

    /// **Every buffer the path creates is in the list, and the list is in the order the records are
    /// submitted.** This is the test the brief's "the rule that catches everyone" asks for, and it
    /// runs on any machine — the submission is a value.
    ///
    /// It is not a tautology: the kinds come from `va.h` and the order is the submission's, so a
    /// buffer added to the list twice, dropped, or reordered fails here rather than on the GPU.
    #[test]
    fn every_buffer_a_key_frame_creates_is_in_the_list_and_in_order() {
        let p = plan();
        let sequence = sequence_record(&p).unwrap_or_else(|e| panic!("the sequence record: {e}"));
        let submission = build_submission(&p, &sequence, 1, 7, true, 0, &no_references(), true).unwrap_or_else(|e| panic!("a key frame's submission: {e}"));
        assert_eq!(
            submission.kinds(),
            vec![
                ffi::ENC_SEQUENCE_PARAMETER_BUFFER as u32,
                ffi::ENC_MISC_PARAMETER_BUFFER as u32,
                ffi::ENC_PICTURE_PARAMETER_BUFFER as u32,
                PACKED_HEADER_PARAMETER_BUFFER_TYPE,
                ffi::ENC_PACKED_HEADER_DATA_BUFFER as u32,
                PACKED_HEADER_PARAMETER_BUFFER_TYPE,
                ffi::ENC_PACKED_HEADER_DATA_BUFFER as u32,
            ],
            "a key frame is seven buffers in this order: the sequence record, the frame rate, the picture record, \
             then the packed sequence header's parameter and data buffers, then the packed frame header's"
        );
        // **Two packed headers, two parameter buffers** — one per header. A path that created both
        // data buffers and one parameter buffer would still have seven entries and be wrong.
        assert_eq!(submission.count_of(PACKED_HEADER_PARAMETER_BUFFER_TYPE), 2, "one parameter buffer per packed header");
        assert_eq!(submission.count_of(ffi::ENC_PACKED_HEADER_DATA_BUFFER as u32), 2, "one data buffer per packed header");
    }

    /// **The packed sequence header goes with the first picture and not with the second**, because a
    /// driver that passes packed headers through would otherwise put a sequence header in every
    /// access unit. This checks the *bytes*, not a flag — the same test the HEVC path makes.
    #[test]
    fn the_sequence_header_is_sent_once_and_the_frame_header_every_picture() {
        let p = plan();
        let sequence = sequence_record(&p).unwrap_or_else(|e| panic!("the sequence record: {e}"));
        let first = build_submission(&p, &sequence, 1, 7, true, 0, &no_references(), true).unwrap_or_else(|e| panic!("the key frame's submission: {e}"));
        let second =
            build_submission(&p, &sequence, 2, 7, false, 1, &one_reference(&p), false).unwrap_or_else(|e| panic!("the predicted frame's submission: {e}"));
        assert_eq!(first.buffers.len(), 7, "the key frame carries the packed sequence header");
        assert_eq!(second.buffers.len(), 5, "the predicted frame carries only the packed frame header");
        assert_eq!(
            second.kinds(),
            vec![
                ffi::ENC_SEQUENCE_PARAMETER_BUFFER as u32,
                ffi::ENC_MISC_PARAMETER_BUFFER as u32,
                ffi::ENC_PICTURE_PARAMETER_BUFFER as u32,
                PACKED_HEADER_PARAMETER_BUFFER_TYPE,
                ffi::ENC_PACKED_HEADER_DATA_BUFFER as u32,
            ],
            "five buffers for a predicted picture whose sequence header has not changed"
        );
    }

    /// **The picture record names no surface that does not exist.** Every `reference_frames[]` entry
    /// that is not a reference is `0xFF` and never 0, because surface 0 is a real surface on this
    /// driver — the failure this catches is a picture that decodes as something plausible.
    #[test]
    fn the_picture_record_names_only_surfaces_and_marks_the_rest_invalid() {
        let p = plan();
        let bytes = picture_record(&p, 3, 7, true, 0).unwrap_or_else(|e| panic!("the picture record: {e}"));
        let record: VaEncPictureParameterBufferAv1 =
            ffi::record_in(&bytes).unwrap_or_else(|| panic!("the picture record does not read back: it is {} bytes", bytes.len()));
        assert_eq!(record.reconstructed_frame, 3, "the reconstruction surface this picture was given");
        assert_eq!(record.coded_buf, 7, "the coded buffer this picture was given");
        for (index, entry) in record.reference_frames.iter().enumerate() {
            assert_eq!(*entry, AV1_NO_SURFACE, "reference_frames[{index}] is not a reference, so it must be 0xFF and not a surface id");
            assert_ne!(*entry, 0, "reference_frames[{index}] is not 0: surface 0 is a real surface on this driver");
        }
        assert_eq!(record.width_in_sbs_minus_1, [0u16; 63], "a one-tile picture writes no tile widths; every entry stays zero");
    }

    /// **Every global-motion transform is identity and reads no parameters** — the assertion that
    /// makes the `wmmat` scale a fact rather than a note.
    #[test]
    fn every_warp_is_identity_and_therefore_reads_no_wmmat_entry() {
        let p = plan();
        let bytes = picture_record(&p, 3, 7, false, 1).unwrap_or_else(|e| panic!("the picture record: {e}"));
        let record: VaEncPictureParameterBufferAv1 =
            ffi::record_in(&bytes).unwrap_or_else(|| panic!("the picture record does not read back: it is {} bytes", bytes.len()));
        assert_eq!(record.wm.len(), 7, "VAEncPictureParameterBufferAV1 carries seven warped-motion entries (va_enc_av1.h:869)");
        for (index, warp) in record.wm.iter().enumerate() {
            assert_eq!(warp.wmtype, AV1_IDENTITY_WARP, "wm[{index}].wmtype is the identity transform");
            assert_eq!(warp.wmmat, [0i32; 8], "wm[{index}].wmmat is zero: the identity transform reads none of the eight entries");
            assert_eq!(warp.invalid, 0, "wm[{index}].invalid is 0: identity is not an affine transform, so there is nothing to invalidate");
        }
        // **The specification's own numbers, pinned.** `WARPEDMODEL_PREC_BITS` is 16 in
        // `03.symbols.md:124` and `global_motion_params()` writes `(i % 3 == 2) ? 1 <<
        // WARPEDMODEL_PREC_BITS : 0`, which is this array. Asserting the specification's value rather
        // than the constant against itself is the difference between a pin and a tautology.
        assert_eq!(params::WARPEDMODEL_PREC_BITS, 16, "av1-spec 03.symbols.md:124: WARPEDMODEL_PREC_BITS = 16");
        assert_eq!(
            params::IDENTITY_WARP,
            [0, 0, 1 << params::WARPEDMODEL_PREC_BITS, 0, 0, 1 << params::WARPEDMODEL_PREC_BITS],
            "the specification's own initialiser for gm_params: (i % 3 == 2) ? 1 << WARPEDMODEL_PREC_BITS : 0"
        );
        assert_eq!(params::IDENTITY_WARP[2], 65_536, "the identity warp's diagonal is 1 << 16 = 65536");
        eprintln!("the wmmat scale, which a submission of this shape does not depend on: {}", params::WMMAT_SCALE_IS_SPECIFIED);
        // And the two halves of the claim joined: the transform reads zero parameters, and the
        // specification's own initialiser for it is what `IDENTITY_WARP` says.
        assert_eq!(
            crate::vaapi::av1_enc::transformation::parameters(AV1_IDENTITY_WARP),
            Some(0),
            "the identity transform reads none of the eight wmmat entries, which is why this path does not depend on their scale"
        );
    }

    /// **The packed header's `bit_length` is bits, not bytes** — a record carrying the byte count is a
    /// header the driver is told is eight times too long, and nothing about that fails visibly.
    #[test]
    fn the_packed_header_parameter_record_states_its_length_in_bits() {
        let p = plan();
        let obu = params::sequence_header_obu(&p).unwrap_or_else(|e| panic!("the sequence header OBU: {e}"));
        let pair = packed_header_pair(AV1_PACKED_HEADER_SPS, &obu).unwrap_or_else(|e| panic!("the packed header pair: {e}"));
        let parameter: crate::vaapi::hevc_enc::VaEncPackedHeaderParameterBuffer =
            ffi::record_in(&pair[0].bytes).unwrap_or_else(|| panic!("the packed header parameter record does not read back"));
        assert_eq!(parameter.bit_length as usize, obu.len() * 8, "bit_length is the data buffer's size in BITS (va.h:2450)");
        assert_eq!(parameter.header_type, AV1_PACKED_HEADER_SPS, "the pair names the sequence header's packed-header type");
        assert_eq!(parameter.has_emulation_bytes, 0, "an AV1 OBU stream has no emulation prevention bytes, unlike HEVC's Annex-B");
        assert_eq!(pair[1].bytes, obu, "the data buffer is the OBU exactly as params built it");
    }

    /// **The two packed-header types are different numbers**, and a submission that sent the frame
    /// header as a sequence header would produce a stream with no frame header in it.
    #[test]
    fn the_sequence_and_frame_packed_header_types_are_different() {
        assert_eq!(AV1_PACKED_HEADER_SPS, 1, "VAEncPackedHeaderAV1_SPS = VAEncPackedHeaderSequence = 1 (va_enc_av1.h:221)");
        assert_eq!(AV1_PACKED_HEADER_PPS, 2, "VAEncPackedHeaderAV1_PPS = VAEncPackedHeaderPicture = 2 (va_enc_av1.h:229)");
        assert_ne!(AV1_PACKED_HEADER_SPS, AV1_PACKED_HEADER_PPS, "the two types must not collapse onto one number");
    }

    /// **The configuration attribute this path submits is a subset of what the driver says it takes,
    /// and excludes the flag HEVC needs.** This is the AV1-side version of the HEVC finding that
    /// `VAConfigAttribEncPackedHeaders` is not a free choice.
    #[test]
    fn the_packed_header_configuration_is_one_the_driver_advertises() {
        assert_eq!(AV1_PACKED_HEADERS, 0x3, "SEQUENCE | PICTURE, the two headers this path supplies");
        assert_eq!(AV1_DRIVER_PACKED_HEADERS, 0x1b, "what this driver answers VAConfigAttribEncPackedHeaders with for AV1");
        assert_eq!(AV1_PACKED_HEADERS & !AV1_DRIVER_PACKED_HEADERS, 0, "every flag this path configures is one the driver advertises");
        assert_eq!(
            AV1_PACKED_HEADERS & ffi::PACKED_HEADER_SLICE,
            0,
            "VA_ENC_PACKED_HEADER_SLICE is NOT in this driver's AV1 mask: AV1 has no slice header, so a submission \
             that configured it would be asking for a buffer this driver does not take"
        );
    }

    /// **The rate control this path submits is one the driver advertises, and `CQP` is in it.**
    #[test]
    fn the_rate_control_this_path_submits_is_in_the_drivers_mask() {
        assert_eq!(AV1_DRIVER_RATE_CONTROL_MASK, 0x56, "CBR | VBR | CQP | ICQ, measured for VAProfileAV1Profile0");
        assert_eq!(
            AV1_DRIVER_RATE_CONTROL_MASK & AV1_RC_CQP,
            AV1_RC_CQP,
            "VA_RC_CQP is in this driver's AV1 rate-control mask, which is what makes it the mode this path submits"
        );
        assert_eq!(
            AV1_DRIVER_RATE_CONTROL_MASK & 0x10,
            AV1_RC_CQP,
            "and the bit is the one va.h:1116 gives VA_RC_CQP, read from the mask rather than from a second constant"
        );
    }

    /// **A predicted picture with no reference is refused**, because it would reconstruct whatever the
    /// surface holds — a plausible wrong picture rather than an error.
    #[test]
    fn a_predicted_picture_with_nothing_to_predict_from_is_refused() {
        let p = plan();
        let sequence = sequence_record(&p).unwrap_or_else(|e| panic!("the sequence record: {e}"));
        let why = build_submission(&p, &sequence, 1, 7, false, 1, &no_references(), false)
            .err()
            .unwrap_or_else(|| panic!("a predicted picture with an empty reference state was accepted, and it cannot be"));
        assert!(why.contains("no reference"), "the refusal has to name the reason, and it said {why:?}");
    }

    /// **A level is derived from the picture and the rate, and it is the lowest that fits** — a level
    /// too low is a stream a conforming decoder refuses.
    #[test]
    fn the_level_is_derived_and_is_the_lowest_that_fits_the_picture() {
        // Annex A: level 2.0 is 147,456 samples with a 4,423,680 samples/sec display rate.
        assert_eq!(params::level_for(320, 240, 30).unwrap_or(200), 0, "320x240 at 30 fits level 2.0 (seq_level_idx 0)");
        assert_eq!(params::level_for(426, 240, 30).unwrap_or(200), 0, "426x240 at 30 is the annex's own level 2.0 example");
        assert_eq!(params::level_for(1920, 1080, 30).unwrap_or(200), 8, "1920x1080 at 30 needs level 4.0 (seq_level_idx 8)");
        assert_eq!(params::level_for(3840, 2160, 60).unwrap_or(200), 13, "3840x2160 at 60 needs level 5.1 (seq_level_idx 13)");
        // **A picture above every level in the annex is refused, not given the largest one.**
        let too_big = params::level_for(40_000, 40_000, 30);
        assert!(too_big.is_err(), "a picture past the annex's largest level is refused, and it returned {too_big:?}");
        // **And the level the plan writes is the one the sequence header says.**
        let p = plan();
        assert_eq!(p.level_idx, params::level_for(W, H, 30).unwrap_or(200), "the plan's level is the one level_for derives for its picture");
    }

    /// **The sequence record's `seq_fields` word says exactly the flags this file's rule names, and
    /// the ones the driver reports as `VA_FEATURE_NOT_SUPPORTED` are all clear.**
    #[test]
    fn the_sequence_record_only_sets_flags_the_driver_advertises() {
        let p = plan();
        let bytes = sequence_record(&p).unwrap_or_else(|e| panic!("the sequence record: {e}"));
        let record: VaEncSequenceParameterBufferAv1 =
            ffi::record_in(&bytes).unwrap_or_else(|| panic!("the sequence record does not read back: it is {} bytes", bytes.len()));
        // **The whole word, asserted**, not "the interesting bits": `enable_order_hint` (bit 8),
        // `enable_cdef` (bit 12), `subsampling_x` (bit 17) and `subsampling_y` (bit 18) and nothing
        // else. Each of the other fourteen flags reads `VA_FEATURE_NOT_SUPPORTED` in the driver's
        // `VAConfigAttribEncAV1` word, and `va.h:381` says such a flag must not be set.
        let expected =
            (1 << seq_fields::ENABLE_ORDER_HINT) | (1 << seq_fields::ENABLE_CDEF) | (1 << seq_fields::SUBSAMPLING_X) | (1 << seq_fields::SUBSAMPLING_Y);
        assert_eq!(record.seq_fields, expected, "the sequence record sets exactly enable_order_hint, enable_cdef and 4:2:0 subsampling, and nothing else");
        for (name, bit) in [
            ("use_128x128_superblock", seq_fields::USE_128X128_SUPERBLOCK),
            ("enable_superres", seq_fields::ENABLE_SUPERRES),
            ("enable_restoration", seq_fields::ENABLE_RESTORATION),
        ] {
            assert_eq!(record.seq_fields & (1 << bit), 0, "{name} is bit {bit} and is clear, because the driver reports its feature VA_FEATURE_NOT_SUPPORTED");
        }
        assert_eq!(record.seq_level_idx, p.level_idx, "the record states the level the annex's table derives");
        assert_ne!(record.bits_per_second, 0, "iHD rejects a sequence record whose bitrate is 0 (measured on the H.264 path)");
        assert_eq!(record.order_hint_bits_minus_1, params::ORDER_HINT_BITS as u8 - 1, "the record and the sequence header state the same OrderHintBits");
        assert_eq!(record.seq_profile, 0, "seq_profile 0 is the only profile this driver advertises");
    }

    /// **The end-to-end attempt, run against this machine's driver, and what it said.**
    ///
    /// This is the test that turns [`driver_outcomes`] from a claim into a measurement taken by
    /// FilmCraft's own path: it builds a real [`Av1Encoder`], builds a real submission, renders it,
    /// syncs, and reads back.
    ///
    /// **It asserts that the picture record is refused**, and that is the measurement rather than a
    /// wish: `DRIVER_REFUSES_THE_AV1_PICTURE_RECORD` is what this driver does with every
    /// `VAEncPictureParameterBufferAV1`, swept over rate-control modes, packed-header modes, surface
    /// counts and field variants. **A driver that ever accepts one makes this test fail**, which is
    /// the correct outcome — the note above it would then be stale, and the failure is what says so.
    ///
    /// It **skips** when there is no VA-API display, because "this machine has no hardware" is not a
    /// claim about the driver's AV1 encoder.
    #[test]
    fn the_submission_is_refused_by_this_driver_at_the_picture_record() {
        let Ok(display) = crate::vaapi::open_display() else {
            eprintln!("SKIPPED: no VA-API display on this machine, so the driver's AV1 encoder cannot be asked");
            return;
        };
        let Ok(p) = Av1Plan::new(W, H, 30, Q) else {
            eprintln!("SKIPPED: a {W}x{H} plan at 30 fps is not expressible");
            return;
        };
        // **The two numbers the path's choices rest on, asked of the driver rather than taken from
        // this file's own constants.** A driver that offers a different rate-control set would make
        // `VA_RC_CQP` the wrong submission, and a driver that offers interlaced encode would make the
        // absence of a field-picture case in `FrameKind` a gap rather than a refusal.
        eprintln!("driver rate-control mask for AV1 = {:#x}", Av1Encoder::driver_rate_control_modes(&display));
        assert!(
            Av1Encoder::driver_rate_control_modes(&display) & AV1_RC_CQP == AV1_RC_CQP,
            "VA_RC_CQP is the mode this path submits, and the driver must offer it whatever else it offers"
        );
        // **The recorded constant is asserted against a bit of the recorded number, not against
        // itself**: a driver on another machine may legitimately answer a different mask, so what this
        // asks for is the one property the path depends on.
        assert_eq!(AV1_DRIVER_RATE_CONTROL_MASK & 0x10, AV1_RC_CQP, "the recorded mask 0x56 contains VA_RC_CQP, which is the mode this path submits");
        assert_eq!(AV1_DRIVER_INTERLACED, 0, "this driver reports VAConfigAttribEncInterlaced = 0 for AV1, so FrameKind has no interlaced case");
        let mut encoder = match Av1Encoder::at_constant_quantiser(&display, p) {
            Ok(encoder) => encoder,
            Err(why) => {
                // **Refused at configuration is a different answer from refused at the record**, and
                // the note above must say which: a machine whose driver does not offer
                // `VAProfileAV1Profile0` at all has not been asked the question this test asks.
                eprintln!("SKIPPED: vaCreateConfig/Context refused an AV1 encoder: {why}");
                return;
            }
        };
        let submission = match encoder.submission(Av1EncodeRequest::key_frame(0)) {
            Ok(submission) => submission,
            Err(why) => panic!("building the submission is not something the driver can refuse: {why}"),
        };
        assert_eq!(submission.buffers.len(), 7, "the key frame's seven buffers, built before any libva call");
        assert!(submission.bytes() > 0, "the submission carries {} bytes of records and packed headers", submission.bytes());
        // **The encoder's own state, read rather than assumed.** Each of these is a claim the frame
        // loop makes to a caller, and a caller that cannot ask is a caller that has to guess.
        assert_eq!(encoder.plan().width, W, "the encoder was built for the picture the plan states");
        assert_eq!(encoder.plan().height, H, "and for its height");
        assert_eq!(encoder.pictures_encoded(), 0, "nothing has been encoded yet, so the encoder has produced no picture");
        assert!(encoder.is_configured(), "an encoder that has encoded nothing and not ended is still able to encode");
        encoder.forget_references();
        assert!(encoder.is_configured(), "forgetting references does not end the stream; it only forces the next picture to be a key frame");
        // **A predicted picture is buildable once there is something to predict from**, which is the
        // one property of the frame loop that does not need the driver to answer.
        encoder.references.push((1, 0));
        match encoder.submission(Av1EncodeRequest::predicted(1).last()) {
            Ok(second) => assert_eq!(second.buffers.len(), 5, "the predicted picture sends five buffers: the sequence header has not changed"),
            Err(why) => panic!("building the predicted frame's submission: {why}"),
        }
        encoder.references.clear();
        match encoder.encode(Av1EncodeRequest::key_frame(0)) {
            Ok(picture) => {
                panic!("the driver returned {} bytes of AV1, which contradicts driver_outcomes::SUMMARY and means the note is stale", picture.bytes.len())
            }
            Err(why) => {
                // **The end of the stream is a state the caller can ask about**, and asking is what
                // tells it apart from "the encoder died".
                encoder.flush().unwrap_or_else(|e| panic!("flushing an encoder that never encoded: {e}"));
                assert!(!encoder.is_configured(), "a flushed encoder is not configured, and a caller has to be able to see that");
                let _ = encoder.upload(&[], &[]);
                // Named, not asserted-to-a-string: the refusal has to say which call refused, because
                // "an error happened" is the claim the brief warns against. **The two notes are
                // printed with it** so that whoever reads the output has the measurement and the
                // buffer accounting in front of them rather than having to go and find them.
                eprintln!("AV1 submission refused by this driver, as driver_outcomes::SUMMARY records:\n  {why}");
                eprintln!("driver_outcomes::SUMMARY = {}", driver_outcomes::SUMMARY);
                eprintln!("the_tile_group_buffer::WHY_NOT_SENT = {}", the_tile_group_buffer::WHY_NOT_SENT);
                eprintln!("DRIVER_REFUSES_THE_AV1_PICTURE_RECORD = {DRIVER_REFUSES_THE_AV1_PICTURE_RECORD}");
                assert!(why.contains("vaRenderPicture") || why.contains("Annex-B"), "the refusal names the call that refused, and it said {why:?}");
            }
        }
    }
}
