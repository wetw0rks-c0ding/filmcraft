//! The VA-API **AV1** encoder reached from outside the module that wrote it, and what it refuses
//! (Linux). Skips without a VA-API driver — and says so when it does, because a silent skip is a
//! green tick that means nothing.
//!
//! # Why this file exists
//!
//! `Av1Encoder` was declared with a `#[path]` attribute from inside `av1_enc.rs` rather than as a
//! module of `vaapi`, which left it a `pub` item in a private module inside a private module: dead
//! code the rest of the crate never saw. **Nothing in `tests/` or `examples/` could name it**, and
//! the only test of the submission path lived inside the submission path. Every name below is written
//! `filmcraft_platform::vaapi::…`, so an edit that puts the encoder back out of reach fails to
//! **compile** here rather than quietly leaving the path untested again.
//!
//! # The headline, and it is not good news
//!
//! **No AV1 bytes come out of this encoder.** Measured on iHD 26.3.5 / libva 1.24 at 320x240 on an
//! Intel Arc B580, through the shipped code:
//!
//! ```text
//! vaQueryConfigEntrypoints(VAProfileAV1Profile0)      -> [1 VLD, 6 EncSlice]   <- encode is advertised
//! vaCreateConfig / vaCreateSurfaces / vaCreateContext -> SUCCESS
//! vaRenderPicture([sequence record])                  -> SUCCESS
//! vaEndPicture                                        -> VA_STATUS_ERROR_ENCODING_ERROR (24)
//! vaRenderPicture([sequence, picture])                -> VA_STATUS_ERROR_INVALID_BUFFER (7)
//! ```
//!
//! Swept over rate control {NONE,CBR,VBR,CQP,ICQ} x packed headers {NONE,SEQ,PIC,SEQ|PIC,0x1b} x
//! surfaces {2,3,4,8} (80 combinations), over 16 cumulative picture-record field variants and over 80
//! candidate extra buffer types: the failure did not move. ffmpeg's own `av1_vaapi` declines the same
//! profile independently. **So what this milestone delivers is a path that is complete, accounted for
//! and checkable over a driver that will not take its picture record** — not an encoder.
//!
//! # Encode verification is materially weaker than decode's, and this file says so
//!
//! The decode milestone could assert pictures bit-exact against ffmpeg because both sides were the
//! same algorithm. **That is not available at any level here.** A hardware encoder and FilmCraft's
//! software encoder are different algorithms making different decisions, so "matches the software
//! encoder" would be a claim with no content — and "matches ffmpeg's encode" is worse than nothing,
//! because it would be measuring ffmpeg's encoder. **Nothing in this file says either.** There is no
//! bitstream to decode, so there is no fidelity claim here at all. What the file *can* say, and does,
//! is that the bytes FilmCraft submits are real AV1 that FilmCraft's own bit-exact parser reads back
//! to the values they were written from.
//!
//! # What is asserted, and the deliberate break each assertion catches
//!
//! - [`the_render_call_accepts_the_sequence_record_alone_and_refuses_the_picture_record`] — **the
//!   discriminating one, and the reason this file is worth having.** One session, one config, one
//!   sequence record: one render submits the sequence record alone and another submits the very same
//!   record *plus the picture record*, and the two answers differ. **"It fails" would pass on any
//!   breakage at all** — no libva, a dead DRM node, the wrong profile, a session that never
//!   configured. This cannot: a driver that starts refusing the sequence record fails the first half,
//!   and a driver that starts accepting the picture record fails the second.
//! - [`an_av1_encoder_is_configured_and_produces_no_picture`] — the driver's own `EncSlice`
//!   advertisement, `Av1Encoder`'s state read back rather than assumed, and the refusal `encode`
//!   reports. Breaks on a driver that stops advertising AV1 encode, or on a path that returns `Ok`
//!   with bytes the driver never wrote.
//! - [`a_key_frames_submission_carries_an_av1_bitstream_any_machine_can_parse`] — **runs with no GPU
//!   at all**, because `build_submission` is a pure function. Breaks on an OBU of the wrong type, a
//!   length field that disagrees with its content, a dropped header, a field in the wrong place.
//! - [`a_predicted_picture_naming_an_empty_reference_slot_is_refused`] — the frame loop's first-frame
//!   refusal, which is a decoder's refusal too. Breaks on a path that clears the array instead, which
//!   would name surface 0, which is a real surface.
//! - [`the_recorded_driver_numbers_are_what_this_path_assumes`] — the constants the path's choices
//!   rest on, each compared against something that could disagree with it, and the one that has no
//!   counterpart recorded in prose rather than given a fake assertion.

#![cfg(target_os = "linux")]

use filmcraft_av1::header::{FrameHeader, HeaderState, RefInfo, SequenceHeader};
use filmcraft_av1::{BitReader, KEY_FRAME, NUM_REF_FRAMES};
use filmcraft_platform::vaapi::{
    AV1_DRIVER_INTERLACED, AV1_DRIVER_PACKED_HEADERS, AV1_DRIVER_RATE_CONTROL_MASK, AV1_MAX_REFERENCES, AV1_PACKED_HEADERS, AV1_RC_CQP, Av1Buffer,
    Av1EncodeRequest, Av1Encoder, Av1Plan, DRIVER_REFUSES_THE_AV1_PICTURE_RECORD, Display, Entrypoint, Profile, RtFormat, SliceEncoder, build_submission,
    driver_outcomes, open_display, the_tile_group_buffer,
};

/// The libva session is process-wide on this driver, so one test at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The display, or `None` with the reason — **and a skip is not a pass, so the reason is printed.**
fn display() -> Option<std::sync::Arc<Display>> {
    match open_display() {
        Ok(d) => Some(d),
        Err(why) => {
            eprintln!("SKIPPED: no VA-API display: {why}");
            None
        }
    }
}

/// A display on a driver that **advertises an AV1 encoder and answers the recorded numbers** — the
/// precondition for asserting the recorded refusal — or `None`, with the reason printed.
///
/// Three different questions, which is why they are not collapsed into one `if`: there is no
/// hardware at all; the driver does not offer AV1 encode (a different answer from "it offers it and
/// then refuses the record"); or the driver's answers differ from the ones the measurement was taken
/// on, in which case the recorded refusal is not a statement about this machine and asserting it
/// would be asserting somebody else's GPU.
fn recorded_driver() -> Option<std::sync::Arc<Display>> {
    let display = display()?;
    let entrypoints = match display.entrypoints(Profile::Av1Main) {
        Ok(e) => e,
        Err(why) => {
            eprintln!("SKIPPED: vaQueryConfigEntrypoints(VAProfileAV1Profile0) could not be asked: {why}");
            return None;
        }
    };
    if !entrypoints.contains(&Entrypoint::EncSlice) {
        eprintln!(
            "SKIPPED: this driver advertises {entrypoints:?} for VAProfileAV1Profile0 and not VAEntrypointEncSlice, so it has no AV1 encoder to refuse a record with"
        );
        return None;
    }
    let modes = Av1Encoder::driver_rate_control_modes(&display);
    if modes != AV1_DRIVER_RATE_CONTROL_MASK {
        eprintln!(
            "SKIPPED: this driver answers VAConfigAttribEncRateControl for AV1 with {modes:#x}; driver_outcomes::SUMMARY records {AV1_DRIVER_RATE_CONTROL_MASK:#x}. \
             The recorded refusal is a fact about a driver, not about AV1, so it is not asserted here."
        );
        return None;
    }
    Some(display)
}

/// The picture size: **320x240**, the size the module's own measurements and the HEVC probe use, and
/// a whole number of AV1's 8x8 superblocks in both directions.
const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
/// The quantiser: **100**, mid-range on `base_q_idx`'s 0..=255, and the value the recorded
/// measurement was taken at.
const QINDEX: u8 = 100;
/// The coded buffer the raw session below is created with: twice an uncompressed 4:2:0 frame plus
/// 4 KiB, the same bound the module uses.
///
/// **Not an assertion.** No picture comes out of either render, so nothing here depends on the size
/// beyond `vaCreateBuffer` accepting it; a driver that refused it would make `SliceEncoder::new` say
/// so and the test would skip loudly rather than quietly measure something else.
const CODED_BYTES: u32 = ((WIDTH as u64) * (HEIGHT as u64) * 3 / 2 * 2) as u32 + 4096;

fn plan() -> Av1Plan {
    Av1Plan::new(WIDTH, HEIGHT, 30, QINDEX).unwrap_or_else(|e| panic!("a {WIDTH}x{HEIGHT} AV1 plan at 30 fps: {e}"))
}

/// An empty reference state: a key frame's header has no reference to resolve.
fn no_references() -> [RefInfo; NUM_REF_FRAMES] {
    std::array::from_fn(|_| RefInfo::default())
}

/// The `obu_type` in an OBU header byte (`(header >> 3) & 0xf`), or `None` for a buffer that is not
/// an unextended OBU.
///
/// Classifying by the header byte rather than by position in the buffer list is deliberate: what is
/// being checked is *what FilmCraft submits*, and reading it back off the OBU type means a dropped
/// or substituted header is caught as the wrong bytes rather than as the wrong index.
fn obu_type(bytes: &[u8]) -> Option<u8> {
    let header = *bytes.first()?;
    if header & 0b1000_0000 != 0 {
        return None; // obu_forbidden_bit: not a header this path writes, and not one to parse.
    }
    Some((header >> 3) & 0xf)
}

/// The payload of a **sequence-header** OBU — everything after the one-byte header and the `leb128`
/// `obu_size` — with the size field itself checked against what follows it.
///
/// **The check is the point**: an OBU whose declared length disagrees with its content is not
/// decodable, and no size or offset check elsewhere in this submission would notice. The size field
/// is read as a minimal `leb128`, bounded at eight bytes, which is the AV1 specification's own limit
/// on `leb128()` (`03.symbols.md`), so a buffer that is not a `leb128` is refused rather than read
/// past.
fn sequence_obu_payload(obu: &[u8]) -> &[u8] {
    let mut at = 1usize;
    let mut size_bytes = 0usize;
    let mut size = 0usize;
    loop {
        let byte = *obu.get(at).unwrap_or_else(|| panic!("the sequence-header OBU is {} bytes and ends inside its own size field", obu.len()));
        size_bytes += 1;
        assert!(size_bytes <= 8, "the sequence-header OBU has {size_bytes} bytes of leb128 with no terminating byte");
        size |= usize::from(byte & 0x7f) << (7 * (size_bytes - 1));
        at += 1;
        if byte & 0x80 == 0 {
            break;
        }
    }
    let payload = obu.get(at..).unwrap_or_else(|| panic!("the sequence-header OBU is {} bytes and has no payload after its size field", obu.len()));
    assert_eq!(size, payload.len(), "the sequence-header OBU's obu_size says {size} bytes and it carries {}", payload.len());
    payload
}

/// The payload of a **frame-header** OBU, and the reason it is not [`sequence_obu_payload`].
///
/// **The frame header's `obu_size` is a fixed four-byte field, not a minimal `leb128`.**
/// `va_enc_av1.h:925-927` says the frame-header OBU "contains an obu header, a **4-byte** long
/// `obu_size` field, `frame_header_obu()` syntax chain, and a trailing bit". Reading those four bytes
/// as a minimal `leb128` stops at the first byte and reports a length three bytes short, so a test
/// that shares one reader between the two OBUs disagrees with the module about the frame header — and
/// would be right about the sequence header, which *is* a minimal `leb128`.
fn frame_obu_payload(obu: &[u8]) -> &[u8] {
    const SIZE_FIELD: usize = 4;
    let at = 1 + SIZE_FIELD;
    let field = obu.get(1..at).unwrap_or_else(|| panic!("the frame-header OBU is {} bytes, too short for its obu header and its 4-byte obu_size", obu.len()));
    let mut size = 0usize;
    for (byte, &value) in field.iter().enumerate() {
        size |= usize::from(value & 0x7f) << (7 * byte);
    }
    let payload = obu.get(at..).unwrap_or_else(|| panic!("the frame-header OBU is {} bytes and has no payload after its size field", obu.len()));
    assert_eq!(size, payload.len(), "the frame-header OBU's 4-byte obu_size says {size} bytes and it carries {}", payload.len());
    payload
}

/// FilmCraft's **own** sequence record and picture record, for `session`.
///
/// The sequence record's bytes come from [`Av1Encoder::submission`], which is the only public source
/// of them, and that is deliberate: a test that transcribed the record layout would be testing its
/// own transcription. The picture record is then rebuilt against this session's coded buffer and
/// reconstruction surface by [`build_submission`], because **both records have to name buffers that
/// exist in the session being rendered into** — a record carrying another session's ids would be
/// refused for a reason that has nothing to do with the measurement.
fn filmcraft_records(display: &std::sync::Arc<Display>, plan: &Av1Plan, session: &mut SliceEncoder) -> Result<(Av1Buffer, Av1Buffer), String> {
    let mut encoder = Av1Encoder::at_constant_quantiser(display, *plan)?;
    let submission = encoder.submission(Av1EncodeRequest::key_frame(0))?;
    let sequence = submission.buffers.first().ok_or("a submission with no sequence record")?.clone();
    assert_eq!(submission.buffers.len(), 7, "a key frame is seven buffers, and the sequence record is the first of them");
    let full = build_submission(plan, &sequence.bytes, session.rotate_reconstruction(), session.coded_buffer_id(), true, 0, &no_references(), true)?;
    let picture = full.buffers.get(2).ok_or("a rebuilt submission with no picture record")?.clone();
    Ok((sequence, picture))
}

/// **`vaRenderPicture` accepts the sequence record alone and refuses the picture record.** This is the
/// discriminating measurement, and it is the reason [`SliceEncoder`] is exported at all.
///
/// Two renders, **one session, one config, one sequence record**: the first submits the sequence
/// record by itself, the second submits that very record plus the picture record. The two answers
/// differ, and the only difference between them is the picture record.
///
/// # Why this discriminates, where "it fails" would not
///
/// Asserting only that the encode fails would pass on **any** breakage: no libva, a dead DRM node,
/// the wrong profile, a session that never configured. This cannot, because each half names the libva
/// call it is about:
///
/// - the sequence-record render must **not** fail at `vaRenderPicture` — the driver took the record;
/// - the picture-record render must fail **at `vaRenderPicture`**, naming the call.
///
/// A driver that began refusing the sequence record fails the first half. A driver that began
/// accepting the picture record fails the second. Neither is a change to a string in this file.
///
/// # The sequence-only render is not an encode, and this test does not pretend otherwise
///
/// `vaRenderPicture` accepting a record with nothing to encode leaves the driver to fail at
/// `vaEndPicture` (measured: `VA_STATUS_ERROR_ENCODING_ERROR`, 24) with the coded buffer unwritten.
/// **A driver that accepts your buffers and emits nothing is not a success**, and this file says so
/// by making the sequence-only half assert *only* about `vaRenderPicture`, and by showing at the end
/// that the readback reports no picture either way.
#[test]
fn the_render_call_accepts_the_sequence_record_alone_and_refuses_the_picture_record() {
    let _guard = lock();
    let Some(display) = recorded_driver() else { return };
    let p = plan();

    let mut session = match SliceEncoder::new(
        &display,
        Profile::Av1Main,
        RtFormat::Nv12,
        (WIDTH, HEIGHT),
        AV1_RC_CQP,
        AV1_PACKED_HEADERS,
        CODED_BYTES,
        AV1_MAX_REFERENCES as u32,
    ) {
        Ok(session) => session,
        Err(why) => {
            eprintln!("SKIPPED: this driver would not create an AV1 slice-encoder session, so it was never asked about the picture record: {why}");
            return;
        }
    };
    let (sequence, picture) = match filmcraft_records(&display, &p, &mut session) {
        Ok(pair) => pair,
        Err(why) => {
            eprintln!("SKIPPED: this driver would not configure an AV1 encoder, so it was never asked about the picture record: {why}");
            return;
        }
    };

    // Both handles stay alive across both renders. A handle dropped before the render names an id
    // that no longer exists, and iHD answers that with `VA_STATUS_ERROR_INVALID_BUFFER` — the very
    // error this test localises — so a dropped handle would make the second render fail for the
    // wrong reason and the first render's acceptance meaningless.
    let sequence_buffer = session
        .buffer(i32::try_from(sequence.kind).unwrap_or(-1), &sequence.bytes, 1)
        .unwrap_or_else(|e| panic!("vaCreateBuffer for the sequence record ({} bytes): {e}", sequence.bytes.len()));
    let picture_buffer = session
        .buffer(i32::try_from(picture.kind).unwrap_or(-1), &picture.bytes, 1)
        .unwrap_or_else(|e| panic!("vaCreateBuffer for the picture record ({} bytes): {e}", picture.bytes.len()));

    // ---- Render one: the sequence record alone. `vaRenderPicture` must take it.
    let alone = session.render(&[sequence_buffer.id()]);
    match &alone {
        Ok(()) => eprintln!("the sequence record alone rendered, and vaEndPicture did not object"),
        Err(why) => {
            assert!(
                !why.contains("vaRenderPicture"),
                "the driver refused the SEQUENCE record itself: {why}. The recorded finding is that this record renders on its own, so either that has \
                 changed or this session differs in some way the note does not describe."
            );
            assert!(why.contains("vaEndPicture"), "with the sequence record alone the failure is expected at vaEndPicture, and it said {why:?}");
            eprintln!("the sequence record alone: vaRenderPicture took it, and vaEndPicture then said: {why}");
        }
    }

    // ---- Render two: the same sequence record plus the picture record. `vaRenderPicture` must refuse.
    let Err(refusal) = session.render(&[sequence_buffer.id(), picture_buffer.id()]) else {
        panic!(
            "the driver accepted VAEncPictureParameterBufferAv1, which contradicts {DRIVER_REFUSES_THE_AV1_PICTURE_RECORD}: driver_outcomes::SUMMARY is stale and this file's premise has changed"
        );
    };
    assert!(refusal.contains("vaRenderPicture"), "the refusal names the call that refused, and it said {refusal:?}");
    eprintln!("the same sequence record plus the picture record: {refusal}");

    // ---- The localisation, stated as an inequality rather than as two separate hopes.
    //
    // One render carries one buffer and the other carries two, and **they are not the same answer**.
    // Written this way because "the second failed" is only evidence if the first did not fail the
    // same way — which is exactly the part a blanket failure assertion leaves out.
    assert!(
        !alone.as_ref().err().is_some_and(|why| why.contains("vaRenderPicture")),
        "adding one buffer changed the outcome at vaRenderPicture, and that is the localisation: {alone:?} then {refusal}"
    );

    // **Nothing is reported as a picture**, whichever way each render went. The readback refuses the
    // buffer because what is in it is not a bitstream FilmCraft can read, and `Av1Encoder::encode`
    // is what turns that into an error rather than into an empty `Ok`.
    let synced = session.sync();
    assert!(synced.is_ok(), "vaSyncBuffer on a buffer the driver left alone: {synced:?}");
    match session.read_coded() {
        Ok(picture) => panic!("the driver reported {} bytes of coded picture from a session in which no picture was ever encoded", picture.len()),
        Err(why) => eprintln!("no coded picture came back, as expected: {why}"),
    }
    eprintln!("{DRIVER_REFUSES_THE_AV1_PICTURE_RECORD}");
    eprintln!("driver_outcomes::SUMMARY = {}", driver_outcomes::SUMMARY);
    eprintln!("the tile group's absence is a header fact: {}", the_tile_group_buffer::WHY_NOT_SENT);
}

/// **An AV1 encoder is configured, advertises what the path needs, and produces no picture.**
///
/// This is the reachability assertion in its plainest form: `Av1Encoder` is named from outside the
/// module that wrote it, driven once, and asked to account for itself. Nothing here is a claim about
/// fidelity — there is no bitstream — so the claims are the ones that can be made: the driver
/// advertises `VAEntrypointEncSlice` for AV1, a session configures, the frame loop's state is
/// readable, and `encode` reports an error rather than a picture.
#[test]
fn an_av1_encoder_is_configured_and_produces_no_picture() {
    let _guard = lock();
    let Some(display) = recorded_driver() else { return };
    let p = plan();

    // **The advertisement, asked of the driver rather than read off a note.** `vaQueryConfigProfiles`
    // naming AV1 is not the question; `EncSlice` for it is, and that is the whole encode gate.
    let entrypoints =
        display.entrypoints(Profile::Av1Main).unwrap_or_else(|e| panic!("the AV1 entry points, having just been asked for by recorded_driver(): {e}"));
    assert!(entrypoints.contains(&Entrypoint::Vld), "this driver decodes VAProfileAV1Profile0, which is what a decode-capable driver reports alongside encode");
    assert!(entrypoints.contains(&Entrypoint::EncSlice), "and it advertises VAEntrypointEncSlice for it, which is the question the whole encode path gates on");

    let mut encoder = match Av1Encoder::at_constant_quantiser(&display, p) {
        Ok(encoder) => encoder,
        Err(why) => {
            eprintln!("SKIPPED: vaCreateConfig/vaCreateContext refused an AV1 encoder, so this driver was never asked about the picture record: {why}");
            return;
        }
    };
    // **Every one of these is a claim the frame loop makes to a caller, and a caller that cannot ask
    // has to guess.** They are read here rather than assumed, which is part of why this file is worth
    // having: these methods were unreachable before this milestone.
    assert_eq!(encoder.plan().width, WIDTH, "the encoder was built for the picture the plan states");
    assert_eq!(encoder.plan().height, HEIGHT, "and for its height");
    assert_eq!(encoder.plan().frames_per_second, 30, "and at the rate the plan states");
    assert_eq!(encoder.plan().base_qindex, QINDEX, "and at the quantiser, which is what VA_RC_CQP takes from the picture record");
    assert_eq!(encoder.pictures_encoded(), 0, "nothing has been encoded yet, so the encoder has produced no picture");
    assert!(encoder.is_configured(), "an encoder that has encoded nothing and not ended is still able to encode");
    encoder.forget_references();
    assert!(encoder.is_configured(), "forgetting references does not end the stream; it only forces the next picture to be a key frame");

    // The submission is built before anything is handed over, so its shape is a statement about what
    // *would* be sent rather than about what survived the driver. The order is checked by what each
    // buffer **carries**, not by transcribing libva's buffer-type numbers into this file.
    let submission = encoder
        .submission(Av1EncodeRequest::key_frame(0))
        .unwrap_or_else(|e| panic!("building the key frame's submission is not something the driver can refuse: {e}"));
    let kinds = submission.kinds();
    assert_eq!(
        kinds.len(),
        7,
        "a key frame is seven buffers: sequence record, frame rate, picture record, then two packed headers as a parameter/data pair each"
    );
    assert_eq!(
        obu_type(&submission.buffers[4].bytes),
        Some(1),
        "the first packed header's data buffer is the OBU_SEQUENCE_HEADER, which va_enc_av1.h:213-220 says shall contain the complete sequence_header_obu()"
    );
    assert_eq!(
        obu_type(&submission.buffers[6].bytes),
        Some(3),
        "and the second's is the OBU_FRAME_HEADER, which va_enc_av1.h:222-228 says shall contain the complete frame_header_obu()"
    );
    assert_eq!(
        kinds[3], kinds[5],
        "one packed-header parameter buffer per header, not one for both: a path that built both data buffers and a single parameter buffer would still have seven entries and be wrong"
    );
    assert_eq!(kinds[4], kinds[6], "and one packed-header data buffer per header");
    assert_ne!(
        kinds[3], kinds[4],
        "a packed-header parameter buffer is not the data buffer that holds the bytes, and a submission that merged them would still have seven entries"
    );
    assert!(submission.bytes() > 0, "the submission carries {} bytes of records and packed headers", submission.bytes());

    match encoder.encode(Av1EncodeRequest::key_frame(0)) {
        // **The refusal below names five buffers, not seven**, and that is not a discrepancy: the
        // `submission` call above already sent the packed sequence header, so `encode` builds the
        // next picture's submission without it. A reader seeing "5" against a seven-buffer list above
        // should find it explained here rather than have to work it out.
        Ok(picture) => panic!(
            "the driver returned {} bytes of AV1, which contradicts {DRIVER_REFUSES_THE_AV1_PICTURE_RECORD}: driver_outcomes::SUMMARY is stale",
            picture.bytes.len()
        ),
        Err(why) => {
            assert!(why.contains("vaRenderPicture"), "the refusal names the call that refused, and it said {why:?}");
            // **A failure leaves the frame loop where it was**, which is what lets a caller retry the
            // same picture. Observable only because `pictures_encoded` advances after the bytes come
            // back and not before — so this is a real assertion rather than a restatement of the error.
            assert_eq!(encoder.pictures_encoded(), 0, "a refused picture is not an encoded picture, and the encoder must not count it");
            assert!(encoder.is_configured(), "and a refused picture does not end the stream");
            // The end of the stream is a state a caller can ask about, which is what tells it apart
            // from "the encoder died".
            encoder.flush().unwrap_or_else(|e| panic!("flushing an encoder that never encoded: {e}"));
            assert!(!encoder.is_configured(), "a flushed encoder is not configured, and a caller has to be able to see that");
            eprintln!("the refusal, as driver_outcomes::SUMMARY records it:\n  {why}");
        }
    }
}

/// **The submission carries a real AV1 bitstream, and FilmCraft's own bit-exact parser reads it back
/// to what the records state.** This is the half that runs with no GPU, because `build_submission` is
/// a pure function — which is the whole reason it is one.
///
/// The deliberate breaks, one per assertion: an OBU of the wrong type (the type is read back out of
/// the header byte), a length field that disagrees with its content, a dropped header (exactly one of
/// each type must be present), a geometry written in the wrong place, and a frame header that is not
/// the key frame the request asked for.
#[test]
fn a_key_frames_submission_carries_an_av1_bitstream_any_machine_can_parse() {
    let p = plan();
    // The sequence record is the record module's business and is not checked here; what is checked is
    // the two OBUs, because those are the bytes the driver is documented to need written for it and
    // the only part of a submission that has to be real AV1 rather than a record the driver reads.
    let submission = build_submission(&p, &[], 1, 7, true, 0, &no_references(), true).unwrap_or_else(|e| panic!("a key frame's submission: {e}"));
    assert_eq!(submission.buffers.len(), 7, "a key frame is seven buffers");

    let sequences: Vec<&[u8]> = submission.buffers.iter().map(|b| b.bytes.as_slice()).filter(|b| obu_type(b) == Some(1)).collect();
    let frames: Vec<&[u8]> = submission.buffers.iter().map(|b| b.bytes.as_slice()).filter(|b| obu_type(b) == Some(3)).collect();
    assert_eq!(sequences.len(), 1, "exactly one OBU_SEQUENCE_HEADER (obu_type 1) in the submission, not {}", sequences.len());
    assert_eq!(frames.len(), 1, "exactly one OBU_FRAME_HEADER (obu_type 3) in the submission, not {}", frames.len());

    let sequence =
        SequenceHeader::parse(sequence_obu_payload(sequences[0])).unwrap_or_else(|e| panic!("the packed sequence header does not parse as AV1: {e}"));
    assert_eq!(sequence.max_frame_width, WIDTH, "the sequence header states the picture's width");
    assert_eq!(sequence.max_frame_height, HEIGHT, "and its height");
    assert_eq!(sequence.profile, 0, "seq_profile 0 (Main) is the only profile this driver advertises for AV1");
    assert_eq!(sequence.op_idc, 0, "operating point 0");
    assert_eq!(sequence.order_hint_bits, p.order_hint_bits, "and the OrderHintBits the plan states, which every frame header's order_hint is read in");

    // The frame header is resolved against that sequence header — the same resolution a decoder
    // makes, which is why a reference to an empty slot is caught here rather than on the GPU.
    let mut refs: [RefInfo; NUM_REF_FRAMES] = no_references();
    let mut state = HeaderState::new(&sequence, &mut refs);
    let mut reader = BitReader::new(frame_obu_payload(frames[0]));
    let parsed = FrameHeader::parse(&mut reader, &mut state, 0, 0).unwrap_or_else(|e| panic!("the packed frame header does not parse as AV1: {e}"));
    assert_eq!(parsed.frame_type as u32, KEY_FRAME as u32, "the frame header is the key frame the submission asked for");
    assert!(parsed.frame_is_intra, "a key frame is intra by definition");
    assert_eq!(parsed.upscaled_width, WIDTH, "the frame header states the picture's width");
    assert_eq!(parsed.frame_height, HEIGHT, "and its height");
    assert_eq!(parsed.quant.base_q_idx, u32::from(QINDEX), "and the quantiser the plan states");
    assert_eq!(parsed.order_hint, 0, "and the order hint the request carried");
}

/// **A predicted picture naming an empty reference slot is refused, and the refusal names the slot.**
///
/// This is the frame loop's first-frame refusal and it is a decoder's refusal too: a picture that
/// predicts from a slot holding no picture is not decodable, and encoding it anyway would produce a
/// plausible wrong picture rather than an error. Checked through `build_submission` rather than only
/// through `Av1Encoder::submission` because that is where a caller has to be able to reach it — while
/// writing an export, before a display is opened.
///
/// The deliberate break: a path that cleared the whole array instead of refusing would name surface
/// 0, which is a real surface on this driver, and this would return `Ok`.
#[test]
fn a_predicted_picture_naming_an_empty_reference_slot_is_refused() {
    let p = plan();
    let Err(why) = build_submission(&p, &[], 1, 7, false, 1, &no_references(), true) else {
        panic!("a predicted picture with no reference was submitted: it would reconstruct whatever the surface happens to hold");
    };
    assert!(why.contains("reference_frames[1]"), "the refusal names the slot it refused, and it said {why:?}");
    assert!(why.contains("no reference"), "and it says what the condition is, and it said {why:?}");
}

/// **The recorded numbers are what this path's choices rest on, each checked against something that
/// could disagree with it.**
///
/// Every assertion here compares two things that could differ. `assert_eq!(0x1b, 0x1b)` would pass
/// forever and prove nothing; these ask whether a flag the path *sets* is one the driver
/// *advertises*, or whether a recorded mask *contains* the mode the path submits.
///
/// **`AV1_MAX_REFERENCES` is the one value here with no counterpart to compare against**, so it is not
/// given a fake assertion. It is a decision, not a measurement: a reference depth above one is
/// refused because depth 2 **silently diverges** on this driver rather than erroring — measured on
/// H.264 and HEVC at mean absolute deviation 9.1, 9.1, 16.8, 26.6, 36.9, 37.1 against the source
/// ramp, flat for two pictures and wrong from the third. The driver *invites* it
/// (`VAConfigAttribEncMaxRefFrames` answers 3, 1 for AV1 here), and that is not a reason to submit
/// it. Pinning `== 1` holds the decision; nothing in this file can re-derive it.
#[test]
fn the_recorded_driver_numbers_are_what_this_path_assumes() {
    // **`VA_RC_CQP` is libva's value** (`va.h:1116`, 0x10), not this path's transcription of it: a
    // drift there would make the mask assertion below pass for the wrong reason, which is why the
    // mask is asked to *contain* the mode rather than being compared with it.
    assert_eq!(AV1_RC_CQP, 0x10, "VA_RC_CQP is 0x0000_0010 (va.h:1116), the constant-quantiser mode this path submits");
    assert_eq!(
        AV1_DRIVER_RATE_CONTROL_MASK & AV1_RC_CQP,
        AV1_RC_CQP,
        "the recorded mask {AV1_DRIVER_RATE_CONTROL_MASK:#x} CONTAINS VA_RC_CQP, which is the mode this path submits; a mask lacking it would make vaCreateConfig refuse"
    );
    // The converse is not required and not asserted: the driver also offers CBR, VBR and ICQ, and
    // this path submits none of them.
    assert_eq!(
        AV1_DRIVER_RATE_CONTROL_MASK & 0x40,
        0x40,
        "the recorded mask also carries VA_RC_ICQ (0x40), so 'contains CQP' is a real containment and not the whole mask"
    );
    // **Every packed-header flag this path configures is one the driver advertises.**
    assert_eq!(AV1_PACKED_HEADERS, 0x3, "VA_ENC_PACKED_HEADER_SEQUENCE (0x1) | VA_ENC_PACKED_HEADER_PICTURE (0x2), va.h:1199-1231");
    assert_eq!(
        AV1_DRIVER_PACKED_HEADERS & AV1_PACKED_HEADERS,
        AV1_PACKED_HEADERS,
        "the recorded driver mask 0x1b admits both flags this path sets, which is why 0x3 is the submitted value"
    );
    assert_eq!(AV1_PACKED_HEADERS & !AV1_DRIVER_PACKED_HEADERS, 0, "and no flag this path configures is one the driver did not advertise");
    // **VA_ENC_PACKED_HEADER_SLICE (0x4) is not among them**, which is the AV1-side confirmation that
    // `VAConfigAttribEncPackedHeaders` is not a free choice: an AV1 encoder takes a *frame* header,
    // and PICTURE is the flag that says so.
    assert_eq!(AV1_DRIVER_PACKED_HEADERS & 0x4, 0, "the driver does not advertise VA_ENC_PACKED_HEADER_SLICE for AV1, because AV1 has no slice header");
    assert_eq!(
        AV1_DRIVER_PACKED_HEADERS & 0x18,
        0x18,
        "it does advertise MISC and RAW_DATA, which this path declines to send because no record in va_enc_av1.h needs them"
    );
    assert_eq!(
        AV1_DRIVER_INTERLACED, 0,
        "the driver reports VAConfigAttribEncInterlaced = 0 for AV1, which is why there is no interlaced case in the frame kind and why sending one would be a guess"
    );
    // **The refusal is a named thing carrying both the status and the call**, so a rewrite that keeps
    // the wording but loses the status — or the reverse — fails here.
    assert!(
        DRIVER_REFUSES_THE_AV1_PICTURE_RECORD.contains("VA_STATUS_ERROR_INVALID_BUFFER"),
        "the named refusal carries the status the driver answered with: {DRIVER_REFUSES_THE_AV1_PICTURE_RECORD}"
    );
    assert!(
        DRIVER_REFUSES_THE_AV1_PICTURE_RECORD.contains("vaRenderPicture"),
        "and carries the call it was refused at: {DRIVER_REFUSES_THE_AV1_PICTURE_RECORD}"
    );
    // **The recorded outcome says in words that nothing came out**, because "no bytes" is the finding
    // and must not be paraphrased away into a sentence about records.
    assert!(driver_outcomes::SUMMARY.contains("NO AV1 BITSTREAM COMES OUT"), "the recorded outcome says in words that nothing came out");
    // ... and it records **both halves** of the measurement, because either alone is a weaker claim:
    // "the driver refuses a record" without "the same record alone renders" is not a localisation.
    assert!(driver_outcomes::SUMMARY.contains("ALONE returns SUCCESS"), "the recorded outcome says the sequence record alone renders");
    assert!(driver_outcomes::SUMMARY.contains("VA_STATUS_ERROR_INVALID_BUFFER"), "and that adding the picture record is refused");
    // **The tile group's absence is a header fact and says so**, rather than being an omission a
    // reader has to notice.
    assert!(the_tile_group_buffer::WHY_NOT_SENT.contains("VAEncTileGroupBufferAV1"), "the tile group's absence names the record that cannot be submitted");
    assert!(the_tile_group_buffer::WHY_NOT_SENT.contains("VABufferType"), "and names the reason: libva declares no buffer type for it");
}
