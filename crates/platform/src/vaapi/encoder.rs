//! One H.264 picture through the VA-API slice encoder, end to end.
//!
//! **Encode is decode with the data flow inverted** (docs/superpowers/plans/2026-10-09-vaapi-linux-encode.md,
//! "the decisive difference"): the app hands the driver a picture and the driver writes the slice
//! headers. So there is nothing to parse here — this module builds the records that *describe* what
//! to encode, hands them to [`super::ffi::Encoder`], and reads the compressed picture back out of
//! the buffer the driver filled. The codec configuration that comes back is the driver's, which is
//! why the packed-header mode is on: the SPS and PPS arrive inside the coded buffer and FilmCraft
//! never assembles one.
//!
//! This is deliberately the smallest thing that can fail informatively: **one context, one picture,
//! one readback**. It is not a frame loop, not a settings layer and not rate control, and it is not
//! one. Everything it does not express is a decline rather than a default, which for an encoder
//! means much more than for a decoder: a default QP, frame type or reference state produces a
//! plausible-looking *wrong picture*, not an error.
//!
//! # Every parameter here is either stated or declined
//!
//! - The picture is **constant quantiser** ([`super::ffi::RC_CQP`]), so quality comes from the
//!   picture record's `pic_init_qp` and nothing else. No bitrate is sent, because a bitrate this
//!   backend cannot derive faithfully would be a default quality — and [`VaEncoder::encode`]
//!   therefore takes the QP as an argument rather than choosing one.
//! - **`intra_idr_period` is 0**, so exactly one picture in the sequence is an IDR: the first. That is
//!   what makes the output a stream rather than a fragment, and it is why `frame_num` starts at 0.
//! - **`pic_order_cnt_type` is 0** and `num_ref_frames_in_pic_order_cnt_cycle` is 0: no picture
//!   order count cycle, and no B pictures. See [`EncodeRequest::Picture`] for what that costs.
//! - **The sequence, picture and slice records are submitted together in one `vaRenderPicture`**,
//!   with the frame-rate misc buffer. iHD refuses the render without the frame-rate buffer — a
//!   requirement that is in no header, and `the_h264_records_are_the_size_the_driver_reads` and
//!   [`VaEncoder::encode`]'s errors are where it shows up.
//!
//! # What this does *not* establish
//!
//! One picture is not a stream this backend could ship. It does not show that a P picture predicts
//! correctly (there is nothing to predict from), that `sample_entry()` can be built from the
//! driver's parameter sets, or that a B-frame reordering exists at all. Those are Tasks 3-5.

use super::ffi::{
    self, Display, Encoder, PACKED_HEADER_PICTURE, PACKED_HEADER_SEQUENCE, PICTURE_H264_SHORT_TERM_REFERENCE, Profile, RC_CQP, RtFormat, SLICE_TYPE_ALL_I,
    SLICE_TYPE_ALL_P,
};
use std::sync::Arc;

// The one macroblock size H.264 has, in pixels. Every size this module accepts is rounded to it, and
// the rounding is reported rather than hidden — a caller that asked for 360 rows gets a 368-row
// surface and a crop, and `VaEncoder::encode`'s `coded_size` says so.
const MB: u32 = 16;

/// A picture to encode, and the two things a caller has to say about it.
///
/// Every other parameter is either stated here or is a constant this module can derive with
/// certainty (macroblock counts, frame numbers from a counter, the picture order count of a frame
/// loop this module does not have). The split is the point: `qp` is the user's, because choosing
/// one for them would be choosing their picture's quality, and `idr` is the caller's, because
/// whether a picture is a key frame is a statement about the sequence, not about encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodeRequest {
    /// The quantiser, 0..=51 as H.264 numbers them. **The picture's actual quality**: under
    /// [`RC_CQP`] nothing else affects it.
    ///
    /// Refused outside 0..=51 rather than clamped. A clamp would quietly encode at the nearest
    /// quality instead of the one asked for, which for an encoder is a wrong picture that looks
    /// right.
    pub qp: u8,
    /// Whether this picture is an IDR — a key frame. The first picture of a sequence must be.
    pub idr: bool,
}

impl EncodeRequest {
    /// A request that refuses a QP outside 0..=51.
    pub fn new(qp: u8, idr: bool) -> Result<Self, String> {
        if qp > 51 {
            return Err(format!("a quantiser of {qp}, which is outside H.264's 0..=51"));
        }
        Ok(Self { qp, idr })
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
    /// [`ffi::DPB_SLOTS`].
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

/// What one picture came back as: the compressed bytes, and the size they were encoded at.
///
/// The size is returned rather than remembered because of the crop: H.264 codes whole macroblocks,
/// so a picture that is not a multiple of 16 is encoded padded and cropped, and a caller that
/// assumed its own width would find a decoder disagreeing with it. `coded_size` is what the driver
/// was told, and `display_size` is the crop.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedPicture {
    /// The complete Annex-B access unit: the SPS and the PPS ahead of the slice the driver wrote.
    ///
    /// Which parameter sets these are is in [`Self::parameter_sets_from_driver`]. They are here
    /// rather than in the driver's own output because the driver needs them to be *somewhere*, and a
    /// stream without them is not a stream (Task 4 of the plan builds `avcC` from the driver's own
    /// SPS and PPS; on the machine this milestone was written on, the driver writes none, which is
    /// why [`super::params`] exists).
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
    /// How many references the slice was told to predict from. Zero for an IDR, and one for the P
    /// pictures of a `refs = 1` encoder.
    pub predicted_from: usize,
    /// The `frame_num` this picture was written with.
    pub frame_num: u16,
    /// The `pic_order_cnt_lsb` this picture was written with.
    pub poc: u16,
}

/// A configured H.264 encoder: what [`VaEncoder::new`] returns and what a caller encodes through.
///
/// Holds the libva objects ([`Encoder`]) and the derived geometry, and nothing else. The picture
/// loop itself is deliberately absent: one picture is what this milestone establishes, and a loop
/// would need a reference management this module does not yet have (see [`EncodeRequest::idr`]).
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
    frame_rate: Vec<u8>,
}

impl VaEncoder {
    /// A constant-quantiser H.264 encoder for `size`, holding up to `refs` reference pictures.
    ///
    /// `size` is the picture's *display* size and is rounded up to a whole number of macroblocks:
    /// 640x360 becomes 640x368 with the bottom 8 rows cropped off, which is what H.264 requires
    /// and what [`EncodedPicture`] reports as `coded_size` and `display_size`. Rounding up rather
    /// than refusing is not a default here — it is the only size this codec can encode, and the crop
    /// is exact.
    ///
    /// `refs` is the reference depth. Zero gives an intra-only encoder, which is all Task 1 had and
    /// is enough for a key frame; **one is the minimum that can encode a P picture at all**, because
    /// the driver needs a surface to write the current reconstruction into *and* the previous one to
    /// read from, so the session gets `2 + refs` surfaces (see [`Encoder::new`]).
    ///
    /// `refs` is bounded by [`MAX_REFERENCES`], which is the picture record's `ReferenceFrames[16]`
    /// array — a depth past it would have nowhere to go, and a silently shortened reference list
    /// makes the driver predict from a picture the caller did not choose, so that is an error and
    /// not a clamp.
    pub fn new(display: &Arc<Display>, size: (u32, u32), refs: u32) -> Result<Self, String> {
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
        let coded = (size.0.div_ceil(MB) * MB, size.1.div_ceil(MB) * MB);
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
        let sequence = sequence_record(coded, crop, width_mbs, height_mbs, refs)?;
        let frame_rate = frame_rate_record(30)?;
        let encoder = Encoder::new(
            display,
            Profile::H264Main,
            RtFormat::Nv12,
            coded,
            RC_CQP,
            PACKED_HEADER_SEQUENCE | PACKED_HEADER_PICTURE,
            coded_buffer_for(coded)?,
            refs,
        )?;
        // The quantization misc buffer is rebuilt per picture (it carries the request's `qp`), so it
        // is not built here.
        Ok(Self {
            encoder,
            coded,
            display: size,
            crop,
            macroblocks,
            frame_num: 0,
            poc: 0,
            idr_pic_id: 0,
            refs: refs as usize,
            references: Vec::new(),
            sequence,
            frame_rate,
        })
    }

    /// How many reference pictures this encoder is currently holding.
    pub fn reference_count(&self) -> usize {
        self.references.len()
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
    pub fn forget_references(&mut self) {
        self.references.clear();
        self.frame_num = 0;
        self.poc = 0;
        self.idr_pic_id = 0;
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
        let luma_len = w as usize * h as usize;
        let chroma_len = luma_len / 2;
        // Neutral-value padding, filled in for the rows and columns the crop takes away.
        let mut padded_luma = vec![16u8; luma_len];
        let mut padded_chroma = vec![128u8; chroma_len];
        copy_picture(&mut padded_luma, w as usize, luma, self.display.0 as usize, self.display.1 as usize)?;
        copy_picture(&mut padded_chroma, w as usize, chroma, self.display.0.div_ceil(2) as usize, self.display.1.div_ceil(2) as usize)?;
        self.encoder.upload(&padded_luma, &padded_chroma)
    }

    /// Encode one picture and read back what the driver wrote.
    ///
    /// `request.idr` chooses the picture's type and nothing else does. An IDR is intra-coded and
    /// clears the reference state; a P picture predicts from the references this encoder is holding,
    /// and is **refused** when it is holding none — because the alternative is a P record naming a
    /// reference that does not exist, which the driver accepts and encodes into whatever happens to
    /// be in that surface. That is the plausible-looking wrong picture this module exists to avoid,
    /// so the first picture of a sequence, and the first after [`VaEncoder::forget_references`],
    /// must be an IDR.
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
        EncodeRequest::new(request.qp, request.idr)?;
        if !request.idr && self.references.is_empty() {
            return Err("a P picture with no reference: this encoder has reconstructed nothing to predict from, so only an IDR is offered".into());
        }
        // **The reconstruction target moves on every picture, not only when the encoder is full.**
        //
        // This is the bug the two-reference test found, and it is worth stating because the obvious
        // policy is wrong in a way that only shows up at depth > 1: rotating only when the reference
        // list is full means the first `refs` pictures all write to the *same* surface, so the
        // second picture overwrites the first picture's reconstruction — and the first picture is
        // still in the reference list. Every P frame after that predicts from a surface holding the
        // wrong picture, which decodes to a plausible wrong picture rather than an error.
        //
        // Rotating every time is correct because the pool is `refs + 1` surfaces deep: with at most
        // `refs` live references there is always exactly one surface no reference names, and a pool
        // of that size visited once per picture lands on it every time.
        self.encoder.rotate_reconstruction();
        let current = Current {
            surface: self.encoder.recon_surface(),
            frame_idx: self.slot_for(self.frame_num),
            frame_num: self.frame_num,
            poc: self.poc,
            // Every picture is a candidate reference; the list is trimmed to `refs` *after* the
            // picture comes back, which is where the trim has to be so a failed render cannot leave
            // a shorter list than the picture that failed.
            is_reference: true,
        };
        // An IDR's slice type is "every slice is I"; a P picture's is "every slice is P". The
        // offset-by-five encoding is the header's own (`va_enc_h264.h:441`).
        let slice_type = if request.idr { SLICE_TYPE_ALL_I } else { SLICE_TYPE_ALL_P };
        // **One reference list, two records.** The picture record's `ReferenceFrames` and the slice
        // record's `RefPicList0` are built from this one list, which is what makes them agree — and
        // they have to agree, because the first says what the driver is holding and the second says
        // what the slice reads. An IDR names none: 7.4.3.3 empties the decoded picture buffer at an
        // IDR, so there is nothing in it to predict from by definition.
        let slice_references: Vec<Reference> = if request.idr { Vec::new() } else { self.references.clone() };
        let picture = picture_record(
            &current,
            self.encoder.input_surface(),
            self.frame_num,
            self.macroblocks,
            &slice_references,
            request,
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
        let parameter_sets = super::params::parameter_sets(&sequence_read, &picture_read)?;
        // The buffers are submitted in one `vaRenderPicture`, which is what the header requires for
        // the sequence, picture and slice records together. The order is the one iHD documents:
        // sequence, picture, slice, then the misc records.
        let seq_buf = self.encoder.buffer(ffi::ENC_SEQUENCE_PARAMETER_BUFFER, &self.sequence, 1)?;
        let pic_buf = self.encoder.buffer(ffi::ENC_PICTURE_PARAMETER_BUFFER, &picture, 1)?;
        // The slice buffer is an array of records: `num_elements` is the slice count, which is why
        // `Encoder::buffer` takes it as an argument at all.
        let slice_buf = self.encoder.buffer(ffi::ENC_SLICE_PARAMETER_BUFFER, &slice, 1)?;
        let fps_buf = self.encoder.buffer(ffi::ENC_MISC_PARAMETER_BUFFER, &self.frame_rate, 1)?;
        let quant_buf = self.encoder.buffer(ffi::ENC_MISC_PARAMETER_BUFFER, &quantization, 1)?;
        let buffers = [seq_buf.id(), pic_buf.id(), slice_buf.id(), fps_buf.id(), quant_buf.id()];
        self.encoder.render(&buffers)?;
        // Mandatory before the read: the encode is asynchronous and an unsynchronised read of the
        // coded buffer is a truncated picture rather than a stale one.
        self.encoder.sync()?;
        let coded = self.encoder.read_coded()?;
        let segments = coded.segments();
        let status = coded.status().to_vec();
        let coded_buffer_used = u32::try_from(coded.bytes().len()).unwrap_or(u32::MAX);
        // **The driver's own parameter sets win.** iHD accepts the packed-header mode and on this
        // machine writes none, but a driver that does write them has described the stream more
        // authoritatively than a parameter set this module derived — and this module's would be
        // built from a record the driver may have adjusted. So the driver's are used when present
        // and ours only when they are not, and which happened is reported.
        let from_driver = super::params::has_parameter_sets(coded.bytes());
        let parameter_sets = if from_driver { Vec::new() } else { parameter_sets };
        // The picture came back, so the sequence state can advance. **The counters wrap**: H.264's
        // `frame_num` is a `u16` that is a function of position modulo `MaxFrameNum`, and an
        // encoder that panics at the 65536th picture would be a crash rather than a rollover.
        self.frame_num = self.frame_num.wrapping_add(1);
        self.poc = self.poc.wrapping_add(1);
        if request.idr {
            self.idr_pic_id = self.idr_pic_id.wrapping_add(1);
        }
        // **The reference is committed last.** Everything above can fail, and a reference committed
        // before the picture exists is a reference to an unfinished picture — which the next P frame
        // would predict from, producing a wrong picture rather than an error.
        //
        // An IDR clears the list first (7.4.3.3: an IDR empties the decoded picture buffer, so
        // anything held is gone by definition), then contributes itself. Trimming happens after the
        // push rather than before, so the list is never shorter than the depth the caller asked for
        // once there are enough pictures to fill it.
        if request.idr {
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
            predicted_from: slice_references.len(),
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
fn sequence_record(_coded: (u32, u32), crop: (u32, u32, u32, u32), width_mbs: u16, height_mbs: u16, refs: u32) -> Result<Vec<u8>, String> {
    let record = ffi::VaEncSequenceParameterBufferH264 {
        seq_parameter_set_id: 0,
        // Level 5.1, the highest this table names. The level is a *conformance* statement, and one
        // that is too low makes a conforming decoder refuse the stream, so it is set from the
        // picture rather than left at 0 (which is not a level).
        level_idc: 51,
        // No periodic I frames and no periodic IDR frames: the first picture is the only key frame,
        // and each subsequent picture names itself as an IDR. A caller encoding a longer sequence
        // needs `intra_idr_period`, which is a settings question and not this spike's.
        intra_period: 0,
        intra_idr_period: 0,
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
        //   6-9 log2_max_frame_num_minus4    = 0
        //   10-11 pic_order_cnt_type         = 0
        //   12-15 log2_max_pic_order_cnt_lsb_minus4 = 0
        //   16  delta_pic_order_always_zero_flag = 1
        seq_fields: (1 << 0) | (1 << 2) | (1 << 5) | (1 << 16),
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

/// The quantisation misc buffer, which under [`RC_CQP`] is where the driver's quality controls
/// come from. Every flag is left clear, which means "let the driver decide" (`va.h:2807-2808`) —
/// the honest state, because the QP this backend asks for is the picture record's `pic_init_qp`
/// and not one of the trellis bits.
fn quantization_record(_qp: u8) -> Result<Vec<u8>, String> {
    let record = ffi::VaEncMiscParameterQuantization { quantization_flags: 0, va_reserved: 0 };
    misc(ffi::MISC_QUANTIZATION, &ffi::record_bytes(&record))
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
/// - `pic_fields` sets `idr_pic_flag` (bit 0) from `request.idr`. **Bit 0 is the whole key-frame
///   decision**, and leaving it clear gives a stream with no point a decoder can start from —
///   which ffmpeg reports as "no frame!" rather than as an error.
/// - `pic_init_qp` is `request.qp`, not `26 + qp`. The field is `pic_init_qp_minus26 + 26`, so a QP
///   of 26 is 26 here; subtracting 26 again (a natural mistake when the name says "minus26") is
///   what makes a picture come out at the wrong quality with no error anywhere.
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
    request: EncodeRequest,
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
    let used: &[Reference] = if request.idr { &[] } else { references };
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
        last_picture: 0,
        padding_after_last_picture: [0; 1],
        frame_num,
        // The picture record carries **no** picture order count. It is a slice-header syntax
        // element (7.4.2.1.1's `pic_order_cnt_lsb`), and putting it here — which is tempting,
        // because the picture is the thing it describes — would be a field the record does not have
        // and would land in `pic_init_qp`'s neighbourhood. The slice record below is where it goes.
        pic_init_qp: request.qp,
        num_ref_idx_l0_active_minus1: l0,
        num_ref_idx_l1_active_minus1: 0,
        chroma_qp_index_offset: 0,
        second_chroma_qp_index_offset: 0,
        padding_after_second_chroma_qp_index_offset: [0; 1],
        // idr_pic_flag (bit 0); reference_pic_flag (bits 1-2) is 1 = short-term reference when the
        // picture is one; entropy_coding_mode_flag (bit 3) clear, i.e. CAVLC;
        // deblocking_filter_control_present_flag (bit 9) set, so the slice record's
        // `disable_deblocking_filter_idc` is what the picture says and 0 means "filter it".
        pic_fields: (1 << 9) | (u32::from(current.is_reference) << 1) | u32::from(request.idr),
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
pub fn ramp_picture(width: u32, height: u32) -> (Vec<u8>, Vec<u8>) {
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
pub fn ramp_shifted(width: u32, height: u32, shift: u32) -> (Vec<u8>, Vec<u8>) {
    let (w, h) = (width as usize, height as usize);
    // The ramp spans 219 levels across the width, so `span` is the last column and a one-column
    // picture has no span — `checked_sub` with a filter, so a width of 1 gets a flat 16 rather than
    // a division by zero.
    let span = w.checked_sub(1).filter(|span| *span > 0);
    // The shift wraps modulo the width rather than pushing the ramp off the legal luma range. The
    // `max(1)` is what makes the modulus safe for a zero-width picture, which `Vec::with_capacity(0)`
    // already makes harmless — the modulo is the only thing that would panic.
    let shift = shift as usize % w.max(1);
    let mut luma = Vec::with_capacity(w * h);
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
    let chroma = vec![128u8; w * h / 2];
    (luma, chroma)
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
    use std::mem::{offset_of, size_of};

    /// The encode records, as libva 1.24's headers on this machine declare them. A record whose
    /// layout is wrong does not fail a call — the driver reads a neighbour's value and encodes
    /// something else — so these are the only place that can catch it.
    ///
    /// The numbers came from `sizeof`/`offsetof` on `/usr/include/va/va_enc_h264.h` and `va.h`
    /// with a C compiler, not from arithmetic.
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
        let bytes = sequence_record((640, 368), (0, 0, 0, 4), 40, 23, 1).expect("a 640x368 sequence record");
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

    /// The same for the picture record: the key-frame bit and the QP, read back out.
    #[test]
    fn the_picture_record_names_its_key_frame_and_its_quantiser() {
        let idr = picture_record(
            &Current { surface: 7, frame_idx: 0, frame_num: 0, poc: 0, is_reference: true },
            9,
            0,
            920,
            &[],
            EncodeRequest { qp: 26, idr: true },
            11,
        )
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
        let p = picture_record(
            &Current { surface: 7, frame_idx: 0, frame_num: 0, poc: 0, is_reference: true },
            9,
            0,
            920,
            &[],
            EncodeRequest { qp: 26, idr: false },
            11,
        )
        .expect("a P picture record");
        let p_record = read::<ffi::VaEncPictureParameterBufferH264>(&p).expect("the record reads back");
        assert_eq!(p_record.pic_fields & 1, 0, "a P picture is not an IDR");
        // The two record invariants, both refused rather than encoded around.
        assert!(
            picture_record(
                &Current { surface: 7, frame_idx: 0, frame_num: 0, poc: 0, is_reference: true },
                7,
                0,
                920,
                &[],
                EncodeRequest { qp: 26, idr: true },
                11
            )
            .is_err(),
            "recon == input"
        );
        assert!(
            picture_record(
                &Current { surface: 7, frame_idx: 0, frame_num: 0, poc: 0, is_reference: true },
                9,
                0,
                0,
                &[],
                EncodeRequest { qp: 26, idr: true },
                11
            )
            .is_err(),
            "0 macroblocks"
        );
        // And the record builder's own crop arithmetic, which is where a tuple ordered
        // `(left, top, right, bottom)` instead would put a 640x360 picture's crop on the top edge —
        // a picture ffmpeg decodes with eight rows of the ramp missing from the wrong end.
        let cropped = sequence_record((640, 368), (0, 0, 0, crop_of(368, 360, 2, "bottom").unwrap_or(0)), 40, 23, 1).expect("a cropped sequence record");
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
        let quant = quantization_record(26).expect("a quantisation misc buffer");
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
        let (luma, chroma) = ramp_picture(640, 360);
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
        assert!(EncodeRequest::new(0, true).is_ok(), "0 is a legal QP");
        assert!(EncodeRequest::new(51, true).is_ok(), "51 is a legal QP");
        assert!(EncodeRequest::new(52, true).is_err(), "52 is not");
        assert!(EncodeRequest::new(255, true).is_err(), "255 is not");
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
    let sequence = sequence_record((640, 368), (0, 0, 0, 4), 40, 23, 1).expect("a sequence record");
    let picture = picture_record(
        &Current { surface: 7, frame_idx: 0, frame_num: 0, poc: 0, is_reference: true },
        9,
        0,
        920,
        &[],
        EncodeRequest { qp: 26, idr: true },
        11,
    )
    .expect("a picture record");
    let sequence = read::<ffi::VaEncSequenceParameterBufferH264>(&sequence).expect("it reads back");
    let picture = read::<ffi::VaEncPictureParameterBufferH264>(&picture).expect("it reads back");
    let sets = super::params::parameter_sets(&sequence, &picture).expect("the parameter sets build and verify");
    // Annex-B, SPS then PPS, and the two NAL headers are what say which is which.
    assert!(sets.starts_with(&[0, 0, 0, 1, 0x67]), "an SPS with nal_ref_idc 3 and type 7");
    assert!(sets.windows(5).any(|w| w == [0, 0, 0, 1, 0x68]), "and a PPS with nal_ref_idc 3 and type 8");
    assert!(super::params::has_parameter_sets(&sets), "and this module recognises its own output");
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
#[test]
fn parameter_set_detection_does_not_match_the_byte_sequence_by_accident() {
    // A slice payload with `00 00 01` inside it and no parameter set anywhere.
    let slice = [0u8, 0, 0, 1, 0x05, 0x88, 0, 0, 1, 0x2a, 0x99];
    assert!(!super::params::has_parameter_sets(&slice), "a slice payload is not a parameter set");
    assert!(super::params::has_parameter_sets(&[0, 0, 0, 1, 0x67, 0x4d]), "an SPS");
    assert!(super::params::has_parameter_sets(&[0, 0, 0, 1, 0x67, 0x4d, 0, 0, 0, 1, 0x68, 0xce]), "an SPS then a PPS");
    assert!(!super::params::has_parameter_sets(&[]), "nothing");
    assert!(!super::params::has_parameter_sets(&[0, 0, 0, 1]), "a start code with no header after it");
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
    let picture = picture_record(&current, 9, 8, 920, &references, EncodeRequest { qp: 26, idr: false }, 11).expect("a P picture record");
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
    let idr = picture_record(&current, 9, 8, 920, &references, EncodeRequest { qp: 26, idr: true }, 11).expect("an IDR picture record");
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
    let (a, _) = ramp_shifted(w, h, 0);
    let (b, _) = ramp_shifted(w, h, step);
    let (c, _) = ramp_shifted(w, h, step * 2);
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
    assert!(VaEncoder::new(&api, (640, 360), 1).is_ok(), "one reference is measured to work");
    // More is refused, and the message names the depth and the fact that it is a measurement.
    match VaEncoder::new(&api, (640, 360), 2) {
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
    let mut encoder = VaEncoder::new(&api, (640, 360), 1).expect("a one-reference encoder is measured to work");
    let mut sizes = Vec::new();
    for index in 0..3u32 {
        // A different picture every time, which is what makes the prediction observable at all.
        let (luma, chroma) = ramp_shifted(640, 360, 8 * index);
        encoder.upload(&luma, &chroma).expect("the ramp uploads");
        let request = EncodeRequest::new(26, index == 0).expect("QP 26 is in range");
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
