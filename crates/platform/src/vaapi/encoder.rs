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
    /// The frame number, which increments per picture and wraps at `2^log2_max_frame_num`.
    frame_num: u16,
    //// The picture order count, which `pic_order_cnt_lsb` counts modulo `MaxPicOrderCntLsb`. **Not
    /// in the picture record**: it belongs to the *slice* (7.4.2.1.1's `slice_header()`), which is
    /// where it is written, and the picture record only carries `frame_num`.
    poc: u16,
    /// `idr_pic_id`, which increments per IDR and pairs two IDR pictures in one sequence.
    idr_pic_id: u16,
    /// The sequence record, built once and submitted with every picture.
    sequence: Vec<u8>,
    /// The frame-rate misc buffer, also built once — see the module note for why it is mandatory.
    frame_rate: Vec<u8>,
}

impl VaEncoder {
    /// A constant-quantiser H.264 encoder for `size`, with the driver writing the parameter sets.
    ///
    /// `size` is the picture's *display* size and is rounded up to a whole number of macroblocks:
    /// 640x360 becomes 640x368 with the bottom 8 rows cropped off, which is what H.264 requires
    /// and what [`EncodedPicture`] reports as `coded_size` and `display_size`. Rounding up rather
    /// than refusing is not a default here — it is the only size this codec can encode, and the crop
    /// is exact.
    ///
    /// `coded_bytes` is how large the coded buffer is. It is the caller's because only they know
    /// what an uncompressed picture of this size costs; a fixed fraction of the picture area would
    /// be a guess, and an encoder whose buffer is too small fails at the one moment the picture is
    /// being made. `size_of::<()>()`-sized guesses are refused by [`coded_buffer_for`].
    pub fn new(display: &Arc<Display>, size: (u32, u32)) -> Result<Self, String> {
        if size.0 == 0 || size.1 == 0 {
            return Err(format!("{}x{} picture", size.0, size.1));
        }
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
        let sequence = sequence_record(coded, crop, width_mbs, height_mbs)?;
        let frame_rate = frame_rate_record(30)?;
        let encoder =
            Encoder::new(display, Profile::H264Main, RtFormat::Nv12, coded, RC_CQP, PACKED_HEADER_SEQUENCE | PACKED_HEADER_PICTURE, coded_buffer_for(coded)?)?;
        // The quantization misc buffer is rebuilt per picture (it carries the request's `qp`), so it
        // is not built here.
        Ok(Self { encoder, coded, display: size, crop, macroblocks, frame_num: 0, poc: 0, idr_pic_id: 0, sequence, frame_rate })
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
    /// Everything is built from `request` and this encoder's own counters, and each record is a
    /// value that can be derived: the picture's `frame_num` and picture order count come from the
    /// counters, the IDR flag from `request.idr`, and the quantiser from `request.qp`. There is
    /// nothing here that is *assumed*, which is the property that makes an encoder's output
    /// trustworthy: the alternative is a plausible picture at a quality nobody chose.
    ///
    /// **A P picture is refused.** With `pic_order_cnt_type == 0`, one reference frame and no B
    /// pictures, the first picture has to be an IDR and any other would predict from a reference
    /// this encoder does not yet manage. Rather than submit a P record naming a reference that does
    /// not exist — which the driver would accept and encode from a cleared surface — the second
    /// picture is an error. That is the honest boundary: what is *measured* is one IDR picture, and
    /// the frame loop is Task 5.
    pub fn encode(&mut self, request: EncodeRequest) -> Result<EncodedPicture, String> {
        EncodeRequest::new(request.qp, request.idr)?;
        if !request.idr {
            return Err("a P picture: this encoder has one picture and no reference frame to predict from, so only an IDR is offered".into());
        }
        let picture = picture_record(
            self.encoder.recon_surface(),
            self.encoder.input_surface(),
            self.frame_num,
            self.macroblocks,
            request,
            self.encoder.coded_buffer_id(),
        )?;
        let slice = slice_record(self.macroblocks, SLICE_TYPE_ALL_I, self.poc, self.idr_pic_id, request)?;
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
        let coded = self.encoder.read_coded_from_driver_pointer()?;
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
        // The counters advance only once the picture came back, so a failed render leaves the
        // encoder's sequence state exactly where it was and the caller can retry.
        self.frame_num = self.frame_num.wrapping_add(1);
        self.poc = self.poc.wrapping_add(1);
        self.idr_pic_id = self.idr_pic_id.wrapping_add(1);
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
        })
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
fn sequence_record(_coded: (u32, u32), crop: (u32, u32, u32, u32), width_mbs: u16, height_mbs: u16) -> Result<Vec<u8>, String> {
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
        max_num_ref_frames: 1,
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
/// - `curr_pic.picture_id` is the **reconstruction** surface and `ReferenceFrames` is empty, so the
///   picture is intra-coded and nothing is predicted. This is the encoder's requirement, not a
///   simplification: the driver reconstructs what it encodes and needs somewhere to put it.
/// - `pic_fields` sets `idr_pic_flag` (bit 0) from `request.idr`. **Bit 0 is the whole key-frame
///   decision**, and leaving it clear gives a stream with no point a decoder can start from —
///   which ffmpeg reports as "no frame!" rather than as an error.
/// - `pic_init_qp` is `request.qp`, not `26 + qp`. The field is `pic_init_qp_minus26 + 26`, so a QP
///   of 26 is 26 here; subtracting 26 again (a natural mistake when the name says "minus26") is
///   what makes a picture come out at the wrong quality with no error anywhere.
fn picture_record(recon: u32, input: u32, frame_num: u16, macroblocks: u32, request: EncodeRequest, coded_buf: u32) -> Result<Vec<u8>, String> {
    // `input` and `macroblocks` are the two things this record must *not* say: the input surface is
    // named by `vaBeginPicture`, not by `curr_pic`, and the picture's extent is the slice record's
    // `num_macroblocks`. They are arguments so the call site can state that, and asserted here so a
    // caller that changes one and forgets the other finds out at the call rather than on screen.
    if input == recon {
        return Err(format!(
            "the reconstruction surface {recon} is also the input surface: the driver would read its own output as the picture it is encoding"
        ));
    }
    if macroblocks == 0 {
        return Err("a picture of 0 macroblocks".into());
    }
    // A key frame is not a reference for this module's purposes (nothing follows it), and an IDR
    // clears the DPB by definition. `frame_idx` names the DPB slot, which is the surface's own
    // index here because there is exactly one picture in the buffer.
    //
    // The `flags` bit is defensive: `VaEncoder::encode` refuses a non-IDR picture before reaching
    // here, so it is never set. It is spelled out because a P picture's flag is *this* bit and a
    // reader checking the record will look for it.
    let curr_pic =
        ffi::VaPictureH264 { picture_id: recon, frame_idx: 0, flags: if request.idr { 0 } else { PICTURE_H264_SHORT_TERM_REFERENCE }, ..Default::default() };
    let record = ffi::VaEncPictureParameterBufferH264 {
        curr_pic,
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
        num_ref_idx_l0_active_minus1: 0,
        num_ref_idx_l1_active_minus1: 0,
        chroma_qp_index_offset: 0,
        second_chroma_qp_index_offset: 0,
        padding_after_second_chroma_qp_index_offset: [0; 1],
        // idr_pic_flag (bit 0); entropy_coding_mode_flag (bit 3) clear, i.e. CAVLC;
        // deblocking_filter_control_present_flag (bit 9) set, so the slice record's
        // `disable_deblocking_filter_idc` is what the picture says and 0 means "filter it".
        pic_fields: (1 << 9) | u32::from(request.idr),
        va_reserved: [0; ffi::VA_PADDING_LOW],
        // **Every reference slot is `VA_INVALID_SURFACE`, always — including for an IDR.**
        //
        // Measured: with the array left zeroed, an iHD 26.3.5 H.264 encoder accepts the render
        // (`vaRenderPicture` returns success) and then fails `vaEndPicture` with
        // `VA_STATUS_ERROR_ENCODING_ERROR` ("internal encoding error"). Filling the array with
        // `VA_INVALID_SURFACE` makes the same submission encode.
        //
        // The reason is the same shape of bug as the decode milestone's IQ-matrix note, and it is
        // worth stating exactly: **zero is not "no reference" here, it is surface 0.** This session's
        // surfaces were allocated by the driver and ids 0 and 1 are perfectly real surfaces (the
        // trace shows `render_targets = 0x00000000`, so the input surface is one of them). A zeroed
        // slot therefore names a real, valid, wrong surface, the driver believes it has a reference,
        // and the encode fails somewhere with no field name in the error.
        //
        // `VA_INVALID_SURFACE` is `VA_INVALID_ID` (`va.h:1648`), which is what the decode backend
        // already uses for an unused `ReferenceFrames` slot, so the two halves of this module say
        // the same thing the same way.
        reference_frames: [ffi::VaPictureH264 { picture_id: ffi::INVALID_SURFACE, ..Default::default() }; 16],
    };
    Ok(ffi::record_bytes(&record))
}

/// The slice record: one slice covering the whole picture.
///
/// `num_macroblocks` is the picture's own macroblock count and `macroblock_address` is 0, so the
/// slice is the picture — the simplest slice structure there is, and the one the decode backend's
/// plane arithmetic assumes when it reads a slice header.
///
/// `slice_type` is [`SLICE_TYPE_ALL_I`] (7) rather than 2: the header says "0..2, 5..7, i.e. no
/// switching slices" (`va_enc_h264.h:441`), and the 5..7 range is the one that says every slice in
/// the picture is of this type — which is the statement this slice makes. A driver that is handed
/// a 2 here for a one-slice picture encodes correctly on iHD, and the distinction is exactly the
/// kind of thing a later multi-slice task has to get right.
///
/// `macroblock_info` is [`ffi::INVALID_SURFACE`] (`VA_INVALID_ID`), which the header documents as
/// the value meaning "derive the per-macroblock configuration from this slice's parameters"
/// (`va_enc_h264.h:426-428`). Naming a buffer that does not exist instead would be a driver read
/// of someone else's memory.
fn slice_record(macroblocks: u32, slice_type: u8, poc: u16, idr_pic_id: u16, request: EncodeRequest) -> Result<Vec<u8>, String> {
    let _ = request;
    let record = ffi::VaEncSliceParameterBufferH264 {
        macroblock_address: 0,
        num_macroblocks: macroblocks,
        macroblock_info: ffi::INVALID_SURFACE,
        slice_type,
        pic_parameter_set_id: 0,
        // Pairs two IDR pictures in one sequence. Both counters advance per key frame in
        // `VaEncoder::encode`, and a driver that saw the same value twice would treat the second
        // IDR as a repeat of the first rather than a new random-access point.
        idr_pic_id,
        // **This is where the picture order count goes.** It is a slice-header syntax element and
        // the picture record has no field for it; see the note on `picture_record`.
        pic_order_cnt_lsb: poc,
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
/// The choice, and why: the spike's whole claim is "ffmpeg decodes the driver's bytes to the picture
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
    let (w, h) = (width as usize, height as usize);
    let mut luma = Vec::with_capacity(w * h);
    for _y in 0..h {
        // The ramp is a function of the column only, so it is the same on every row. 16..=235 is
        // H.264's legal luma range (7.4.2.1.1), and starting at 16 rather than 0 means the picture
        // is fully inside it: an encoder that clips outside-range input would otherwise produce a
        // ramp with flattened ends and a decode that could not be told apart from a QP error.
        // `checked_div` rather than a bare `/`: a one-column picture has no ramp to span, and 16 —
        // the bottom of H.264's legal luma range — is the right value for it rather than a
        // division by zero avoided by a guard elsewhere.
        let span = w.checked_sub(1).filter(|span| *span > 0);
        for x in 0..w {
            let step = span.and_then(|span| (219 * x).checked_div(span)).unwrap_or(0);
            luma.push(16u8.saturating_add(u8::try_from(step).unwrap_or(u8::MAX)));
        }
    }
    // Flat neutral chroma: Cb = Cr = 128 is the centre of the range, so any chroma error is
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
        let bytes = sequence_record((640, 368), (0, 0, 0, 4), 40, 23).expect("a 640x368 sequence record");
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
        let idr = picture_record(7, 9, 0, 920, EncodeRequest { qp: 26, idr: true }, 11).expect("an IDR picture record");
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
        let p = picture_record(7, 9, 0, 920, EncodeRequest { qp: 26, idr: false }, 11).expect("a P picture record");
        let p_record = read::<ffi::VaEncPictureParameterBufferH264>(&p).expect("the record reads back");
        assert_eq!(p_record.pic_fields & 1, 0, "a P picture is not an IDR");
        // The two record invariants, both refused rather than encoded around.
        assert!(picture_record(7, 7, 0, 920, EncodeRequest { qp: 26, idr: true }, 11).is_err(), "recon == input");
        assert!(picture_record(7, 9, 0, 0, EncodeRequest { qp: 26, idr: true }, 11).is_err(), "0 macroblocks");
        // And the record builder's own crop arithmetic, which is where a tuple ordered
        // `(left, top, right, bottom)` instead would put a 640x360 picture's crop on the top edge —
        // a picture ffmpeg decodes with eight rows of the ramp missing from the wrong end.
        let cropped = sequence_record((640, 368), (0, 0, 0, crop_of(368, 360, 2, "bottom").unwrap_or(0)), 40, 23).expect("a cropped sequence record");
        let cropped = read::<ffi::VaEncSequenceParameterBufferH264>(&cropped).expect("it reads back");
        assert_eq!(cropped.frame_crop_bottom_offset, 4, "the crop is at the bottom");
        assert_eq!(cropped.frame_crop_top_offset, 0, "and not the top");
    }

    /// The slice record: one slice over the whole picture, no per-macroblock buffer, and a QP delta
    /// of zero so the picture's `pic_init_qp` is what decides quality.
    #[test]
    fn the_slice_record_is_one_slice_and_defers_its_quantiser() {
        let bytes = slice_record(920, ffi::SLICE_TYPE_ALL_I, 3, 1, EncodeRequest { qp: 26, idr: true }).expect("a slice record");
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
    let sequence = sequence_record((640, 368), (0, 0, 0, 4), 40, 23).expect("a sequence record");
    let picture = picture_record(7, 9, 0, 920, EncodeRequest { qp: 26, idr: true }, 11).expect("a picture record");
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
