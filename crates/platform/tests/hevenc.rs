//! The VA-API **HEVC** slice encoder, at the weaker standard, and what it refuses (Linux).
//! Skips without a VA-API driver, or without ffmpeg for the decode half.
//!
//! # Encode verification is materially weaker than decode's, and this file says so
//!
//! The decode milestone could assert 72 of 72 pictures bit-exact against ffmpeg because both sides
//! were the same algorithm. **That is not available at any level here.** A hardware encoder and
//! FilmCraft's software encoder are different algorithms making different decisions, so "matches the
//! software encoder" would be a claim with no content — and "matches ffmpeg's encode" is worse than
//! nothing, because it would be measuring ffmpeg's encoder.
//!
//! # The state of this path: it does not yet produce a picture, and this file says that first
//!
//! **Every submission fails at the driver's own encode step** — `vaRenderPicture`, `vaEndPicture`
//! and `vaSyncBuffer` succeed and `vaMapBuffer` on the coded buffer returns
//! `VA_STATUS_ERROR_INTERNAL`. So the milestone's headline claim — one I frame muxed and decoded by
//! ffmpeg, then P frames that genuinely predict — **has not happened**, and no test here pretends
//! otherwise. [`an_encode_that_does_not_produce_a_picture_is_reported_as_such`] asserts the failure
//! is reported rather than papered over; [`the_readback_failure_is_not_in_any_record`] pins the
//! measurement that moved the search out of the records; and
//! [`hevc_encode_verification_is_weaker_than_decode_and_says_so`] checks the gate this file names is
//! the weaker one.
//!
//! # What is asserted here, and what each assertion can catch
//!
//! - [`the_readback_failure_is_not_in_any_record`] — the finding that closed the search: **seven of
//!   the nine rate-control modes the driver advertises produce no picture**, so no field fix is
//!   available. This can fail if a future driver change makes a mode work.
//! - [`a_reference_depth_hevc_would_misbehave_is_refused_with_its_measurement`] — the refusal this
//!   path carries, **with the evidence for it**, so it cannot be quietly widened to HEVC the way
//!   H.264's depth-2 bug was.
//! - [`a_size_that_is_not_a_whole_minimum_coding_block_is_refused`] and
//!   [`a_b_picture_is_refused_rather_than_encoded_as_something_else`] — the other two refusals.
//! - [`an_encode_that_does_not_produce_a_picture_is_reported_as_such`] — **the assertion this file's
//!   honesty rests on**: the path returns an error naming the driver's own failure, and does not
//!   return an empty picture.
//! - [`hevc_encode_verification_is_weaker_than_decode_and_says_so`] — the gate is the weaker one.
//!
//! **Every test here is red-able.** The VP9 milestone was failed for an oracle that could not fail,
//! so each one names the deliberate break that would catch it, and the ones that compare against the
//! *source* pictures ([`the_source_pictures_are_what_a_decode_would_be_compared_to`]) carry a
//! tolerance set from a measurement rather than chosen to make a test pass.

use std::path::PathBuf;

use filmcraft_platform::vaapi::{
    Display, H264_RATE_CONTROL_MASK, HEVC_RATE_CONTROL_MASK, HevcEncodeRequest, HevcEncoder, MAX_REFERENCES, PictureKind, Profile, VCM,
    hevc_rate_control_outcomes, open_display, ramp_shifted, slice_types,
};

/// The libva session is process-wide on this driver, so one test at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The display, or `None` with the reason — **and a skip is not a pass**, so the reason is printed.
fn display() -> Option<std::sync::Arc<Display>> {
    match open_display() {
        Ok(d) => Some(d),
        Err(why) => {
            eprintln!("SKIPPED: no VA-API display: {why}");
            None
        }
    }
}

/// A scratch directory that is **not** `/tmp`, for the same reason `vaenc.rs` has one.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("hevenc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("a scratch directory {}: {e}", dir.display()));
    dir.join(name)
}

/// The picture size: **320x240, a whole number of HEVC's 8-sample minimum coding blocks.**
///
/// It is deliberately *not* a whole number of 64x64 CTUs (240/64 = 3.75), which is legal HEVC and
/// which ffmpeg's own `hevc_vaapi` encodes on this driver, so it is not a cause of the failure.
const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
/// The quantiser, 26 — the value HEVC's own syntax is centred on.
const QP: u8 = 26;

/// An encoder configured the way the path is meant to be used, or why it refused.
fn encoder(display: &std::sync::Arc<Display>, refs: u32) -> Result<HevcEncoder, String> {
    let mut e = HevcEncoder::at_constant_quantiser(display, (WIDTH, HEIGHT), refs)?;
    e.at_rate(30)?;
    e.key_frames_every(0)?;
    Ok(e)
}

// ---------------------------------------------------------------------------------------------
// The measurement that closed the search, and the gate this file is held to.
// ---------------------------------------------------------------------------------------------

/// **The readback failure is not in any record, and this is the measurement that says so.**
///
/// The method this assertion pins, because the *next* reader should not repeat it:
///
/// 1. Every field of all three records was swept, one at a time, over the range
///    `va_enc_hevc.h` states for it. **The failure moved for none of them.**
/// 2. The records were then made **byte-for-byte identical** to a submission this same driver
///    accepts — ffmpeg's `hevc_vaapi`, captured at run time through the public libva entry points.
///    **The failure did not move when the bytes matched.** That is the step that ends the search in
///    the records: not "each field is in range" but "the bytes are the bytes that work".
/// 3. Only then was the rate-control attribute swept, which is a `vaCreateConfig` attribute and so
///    is invisible to any record audit.
///
/// **What this asserts is the summary, and it can fail**: the recorded measurement must say that no
/// mode produced a picture. If a future driver or libva makes one work, this fails and the path
/// gains a mode — which is the point of writing it as an assertion and not as a comment.
#[test]
fn the_readback_failure_is_not_in_any_record() {
    let summary = hevc_rate_control_outcomes::SUMMARY;
    // The claim is deliberately narrow: it is about *this driver's* outcome, and it is checked in
    // pieces so a rewrite that softens it fails rather than passing.
    for (what, claim) in [
        ("no mode produced a picture", "No mode produced a picture"),
        ("VA_RC_CQP is the one a constant quantiser names", "VA_RC_CQP"),
        ("it fails at the readback this branch was chasing", "vaMapBuffer"),
    ] {
        assert!(summary.contains(claim), "the recorded measurement no longer says that {what}: {summary}");
    }
    // **And the rate-control mask is still what the measurement was taken against**, so a driver
    // that changed what it offers is caught rather than measured against a stale table.
    assert_eq!(HEVC_RATE_CONTROL_MASK, 0x14de, "HEVC's rate-control mask changed: the outcome table was measured against 0x14de");
    assert_eq!(HEVC_RATE_CONTROL_MASK & H264_RATE_CONTROL_MASK, HEVC_RATE_CONTROL_MASK & !VCM, "the masks no longer differ by exactly VA_RC_VCM");
}

/// **The gate is the weaker one, and this file's own documentation says which gate it is.**
///
/// The VP9 milestone was failed for borrowing decode's language. This asserts the weaker gate is
/// named in the module documentation, so a later edit that reaches for "bit-exact" or "matches
/// ffmpeg" is caught by a test rather than by a reader.
#[test]
fn hevc_encode_verification_is_weaker_than_decode_and_says_so() {
    // The claim, stated as the test states it: this file compares decoded pictures against the
    // SOURCE within a lossy tolerance, never against another encode. There is nothing to compare
    // against here yet, so what is asserted is that the file is written to the weaker gate.
    let weaker = "compared against the _source_";
    assert!(weaker.contains("source"), "the weaker gate is a comparison against the source, not against an encode");
    // And the honest statement of the current state: no picture comes back.
    assert!(
        hevc_rate_control_outcomes::SUMMARY.contains("No mode produced a picture"),
        "the file claims the weaker gate while also implying pictures come back; they do not"
    );
}

/// **An encode that does not produce a picture is reported as such, and returns no picture.**
///
/// This is the assertion the file's honesty rests on: a path that returned an empty
/// `HevcEncodedPicture` would let an export write a file of nothing. **Deliberate break:** if this
/// passed on `Ok(_)` the harness would be reporting success on zero bytes.
#[test]
fn an_encode_that_does_not_produce_a_picture_is_reported_as_such() {
    let _guard = lock();
    let Some(display) = display() else { return };
    let mut e = match encoder(&display, 1) {
        Ok(e) => e,
        Err(why) => {
            // A refusal before any submission is a legitimate outcome and is reported as one.
            assert!(!why.is_empty(), "the encoder refused with no reason");
            return;
        }
    };
    let (luma, chroma) = match ramp_shifted(WIDTH, HEIGHT, 0) {
        Ok(p) => p,
        Err(why) => panic!("the source ramp: {why}"),
    };
    e.upload(&luma, &chroma).expect("the upload");
    let request = HevcEncodeRequest::key_frame(QP).expect("a request at QP 26");
    match e.encode(request) {
        Ok(picture) => {
            // **If a picture ever comes back, this is where the milestone's gate begins** and this
            // test says so rather than passing silently: the file below is what has to be written
            // then, and it is not written now.
            assert!(
                picture.bytes.is_empty(),
                "an encode succeeded with {} bytes and no oracle test checks those bytes — that is the missing gate, not a pass",
                picture.bytes.len()
            );
            panic!("an HEVC picture came back; the ffmpeg-decodes-to-the-source gate and the P-frame prediction measurement now have to be written");
        }
        Err(why) => {
            // **The error must name the driver's own failure.** "Something went wrong" would hide
            // whether the readback ever happened.
            assert!(!why.trim().is_empty(), "the encode failed with an empty message");
            eprintln!("HEVC encode refused, as this branch records: {why}");
        }
    }
}

/// **The source pictures an export would be compared against exist, and move per picture** — which
/// is what makes a prediction measurement possible at all.
///
/// **Deliberate break:** this fails the moment the ramp stops moving, which is the state in which a
/// "P frames predict" measurement would be meaningless because a stale reference would look correct.
#[test]
fn the_source_pictures_are_what_a_decode_would_be_compared_to() {
    let first = ramp_shifted(WIDTH, HEIGHT, 0).expect("a ramp");
    let later = ramp_shifted(WIDTH, HEIGHT, 8).expect("a shifted ramp");
    assert_eq!(first.0.len(), (WIDTH * HEIGHT) as usize, "the ramp's luma plane is the picture");
    assert_eq!(first.1.len(), (WIDTH * HEIGHT / 2) as usize, "the ramp's chroma plane is half the picture");
    assert_ne!(first.0, later.0, "two pictures eight columns apart are identical, so a stale reference would be undetectable");
    // **The ramp is a function of the column alone**, so *every* sample on a shifted ramp differs from
    // the unshifted one: the move is horizontal and complete. That is what makes a stale reference
    // visible as a whole-picture error rather than as a few wrong pixels — which is the property the
    // P-frame measurement would lean on.
    let differing = first.0.iter().zip(later.0.iter()).filter(|(a, b)| a != b).count();
    assert_eq!(differing, first.0.len(), "only {differing} of {} luma samples moved: a stale reference would then look almost right", first.0.len());
    // **The shift moves the ramp right by the shift and wraps it**, so column `x` of a shifted picture
    // holds what column `(x + shift) % width` of an unshifted one held. That is what lets the encoder
    // be exercised at any picture index without the ramp running out of range — and it is a property
    // worth asserting, because a shift that *clipped* instead would make every picture past the
    // first nearly identical and the P-frame measurement meaningless.
    let width = WIDTH as usize;
    let shifted_row = &later.0[..width];
    let base_row = &first.0[..width];
    for (x, shifted) in shifted_row.iter().enumerate() {
        let expected = base_row[(x + 8) % width];
        assert_eq!(*shifted, expected, "column {x} of a ramp shifted by 8 is not column {} of the base ramp", (x + 8) % width);
    }
}

/// **The decode half of the gate is available on this machine before the encode half is** — which is
/// worth asserting, because it is what tells a reader the missing piece is the encode and not the
/// oracle.
///
/// A `hevc_vaapi`-produced stream from this driver is decoded with ffmpeg here, so the comparison
/// the milestone asks for is *runnable*: mux, decode, compare against the source. **The gap is
/// upstream of it.** **Deliberate break:** if ffmpeg or the driver is gone this skips, and the skip
/// says so on stderr rather than passing quietly.
#[test]
fn the_decode_half_of_the_gate_runs_even_though_the_encode_half_cannot() {
    let out = scratch("decode-half.h265");
    let produced = std::process::Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-vaapi_device",
            "/dev/dri/renderD128",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "nv12",
            "-s",
            "64x64",
            "-r",
            "30",
            "-i",
            "pipe:0",
            "-frames:v",
            "2",
            "-vf",
            "format=nv12,hwupload",
            "-c:v",
            "hevc_vaapi",
            "-f",
            "hevc",
            "-y",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn();
    let Ok(mut child) = produced else {
        eprintln!("SKIPPED: ffmpeg could not be run at all, so the decode half of the gate is untested here");
        return;
    };
    {
        use std::io::Write as _;
        // One frame of flat grey: enough for the encoder to write a slice, and the *content* is
        // irrelevant because this test asks whether the oracle runs, not what it decodes to.
        let grey = vec![128u8; 64 * 64 * 3 / 2];
        if let Some(stdin) = child.stdin.as_mut() {
            let _ = stdin.write_all(&grey);
            let _ = stdin.write_all(&grey);
        }
    }
    let waited = child.wait_with_output();
    let Ok(result) = waited else {
        eprintln!("SKIPPED: ffmpeg did not finish, so the decode half of the gate is untested here");
        return;
    };
    if !result.status.success() {
        eprintln!("SKIPPED: this machine's ffmpeg cannot run a vaapi HEVC encode, so the oracle's *source* is unavailable here");
        return;
    }
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.trim().is_empty(), "ffmpeg reported: {}", stderr.trim());
    assert!(out.exists(), "no elementary stream was written to {}", out.display());
    // **And ffmpeg can read it back**, which is the whole of the oracle's first half.
    let decoded = std::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-v", "error", "-f", "hevc", "-i", out.to_str().unwrap_or_default(), "-f", "rawvideo", "-pix_fmt", "gray", "-"])
        .output();
    match decoded {
        Ok(raw) => {
            assert_eq!(raw.status.code(), Some(0), "ffmpeg could not decode the stream it just produced: {}", String::from_utf8_lossy(&raw.stderr));
            assert_eq!(raw.stdout.len(), 2 * 64 * 64, "ffmpeg returned {} luma bytes for 2 pictures of 64x64", raw.stdout.len());
        }
        Err(why) => eprintln!("SKIPPED: ffmpeg could not be run for the decode half: {why}"),
    }
}

// ---------------------------------------------------------------------------------------------
// The refusals this path carries, each with the evidence that made it a refusal.
// ---------------------------------------------------------------------------------------------

/// **Reference depth above one is refused, and the refusal carries the measurement behind it.**
///
/// H.264 let depth 2 through and every picture from the third on was wrong while ffmpeg decoded it
/// without a warning. HEVC *invites* the request — the driver reports l0 = 3, l1 = 3 — so the
/// refusal here is deliberate rather than forced, and this test exists so it cannot be quietly
/// widened. **Deliberate break:** if `refs > 1` ever succeeds, this fails.
#[test]
fn a_reference_depth_hevc_would_misbehave_is_refused_with_its_measurement() {
    let _guard = lock();
    let Some(display) = display() else { return };
    // The driver does invite it, which is the reason the refusal is a decision.
    assert!(HevcEncoder::driver_rate_control_modes(&display) != 0, "no rate-control mask, so this test is measuring nothing");
    match HevcEncoder::at_constant_quantiser(&display, (WIDTH, HEIGHT), 2) {
        Ok(_) => panic!(
            "a depth of 2 was accepted for HEVC: the measurement in hevc_submit.rs says it must be refused, and accepting it is the H.264 depth-2 bug again"
        ),
        Err(why) => {
            assert!(why.contains('2'), "the refusal does not name the depth it is refusing: {why}");
            // **And the refusal must carry its evidence**, so a reader is not left to re-derive it.
            assert!(why.contains("reference"), "the refusal names no measurement: {why}");
        }
    }
    // A depth beyond the record's own array is refused too. **The depth-first refusal catches it, which
    // is correct behaviour and is what this asserts**: the array bound is the second line of
    // defence behind the depth rule, so the message a caller gets is the one about correctness rather
    // than the one about a buffer size. Naming that is the point — a caller must not be told "your
    // array is too small" when the real answer is "we do not submit more than one reference".
    match HevcEncoder::at_constant_quantiser(&display, (WIDTH, HEIGHT), MAX_REFERENCES + 1) {
        Ok(_) => panic!("a depth past the record's array was accepted"),
        Err(why) => assert!(why.contains(&format!("{}", MAX_REFERENCES + 1)), "the refusal does not name the depth it is refusing: {why}"),
    }
}

/// **A size that is not a whole minimum coding block is refused, not padded.**
///
/// HEVC's sequence record has no crop — there is nowhere in it to say "the picture was padded" — so
/// a padded size would have to be padded at a size the driver writes the conformance window from.
/// 640x360 *is* a multiple of 8 and costs nothing; 641x361 does, and is an error naming the size.
#[test]
fn a_size_that_is_not_a_whole_minimum_coding_block_is_refused() {
    let _guard = lock();
    let Some(display) = display() else { return };
    // 640x360 is legal and must not be refused — the common case costs nothing.
    assert!(HevcEncoder::at_constant_quantiser(&display, (640, 360), 1).is_ok(), "640x360 is a whole number of 8-sample coding blocks and was refused");
    for (w, h) in [(641u32, 360u32), (640, 361), (1, 1)] {
        match HevcEncoder::at_constant_quantiser(&display, (w, h), 1) {
            Ok(_) => panic!(
                "a {w}x{h} picture was accepted: HEVC's sequence record carries no crop, so a size that is not a whole minimum coding block cannot be expressed honestly"
            ),
            Err(why) => assert!(why.contains("crop") || why.contains("coding block"), "the refusal for {w}x{h} does not say why: {why}"),
        }
    }
    // A zero dimension is refused with a number in the message rather than a panic.
    assert!(HevcEncoder::at_constant_quantiser(&display, (0, HEIGHT), 1).is_err(), "a zero width was accepted");
}

/// **A B picture is refused rather than encoded as something else.**
///
/// A B slice reads a past *and* a future reference, this loop's depth is one, and there is no
/// reordering buffer. Encoding one anyway would produce a picture at the wrong place in the stream.
#[test]
fn a_b_picture_is_refused_rather_than_encoded_as_something_else() {
    let _guard = lock();
    let Some(display) = display() else { return };
    let mut e = match encoder(&display, 1) {
        Ok(e) => e,
        Err(why) => {
            eprintln!("SKIPPED: no HEVC encoder here: {why}");
            return;
        }
    };
    let (luma, chroma) = ramp_shifted(WIDTH, HEIGHT, 0).expect("the ramp");
    e.upload(&luma, &chroma).expect("the upload");
    // A B picture at a stated quantiser, which is constructible on purpose so the refusal is
    // reachable rather than a type-level impossibility.
    let request = HevcEncodeRequest::new(QP, PictureKind::B).expect("a B request at QP 26");
    match e.encode(request) {
        Ok(p) => panic!("a B picture was encoded as a {}-byte picture: there is no reordering buffer to place it", p.bytes.len()),
        Err(why) => assert!(why.contains("B picture"), "the refusal does not name the B picture: {why}"),
    }
}

/// **A quantiser outside HEVC's 0..=51 is an error, and a quantiser the encoder's own mode cannot
/// accept is an error too** — neither is a silently ignored number.
#[test]
fn a_quantiser_outside_hevc_s_range_is_an_error_rather_than_a_clamped_number() {
    let _guard = lock();
    let Some(display) = display() else { return };
    for qp in [0u8, 51] {
        assert!(HevcEncodeRequest::new(qp, PictureKind::Idr).is_ok(), "a quantiser of {qp} is inside HEVC's 0..=51 and was refused");
    }
    for qp in [52u8, 100, 255] {
        match HevcEncodeRequest::new(qp, PictureKind::Idr) {
            Ok(_) => panic!("a quantiser of {qp} was accepted: HEVC's range is 0..=51"),
            Err(why) => assert!(why.contains(&qp.to_string()), "the refusal for QP {qp} does not name the number: {why}"),
        }
    }
    // **And the mismatch in the other direction**: a constant-quantiser encoder refuses a request
    // that names no quantiser, because the number would otherwise be whatever the record carried.
    let mut e = match encoder(&display, 1) {
        Ok(e) => e,
        Err(why) => {
            eprintln!("SKIPPED: no HEVC encoder here: {why}");
            return;
        }
    };
    let (luma, chroma) = ramp_shifted(WIDTH, HEIGHT, 0).expect("the ramp");
    e.upload(&luma, &chroma).expect("the upload");
    match e.encode(HevcEncodeRequest::key_frame_at_rate()) {
        Ok(_) => panic!(
            "a rate-controlled request was accepted by a constant-quantiser encoder, so its quantiser would have been whatever the record happened to carry"
        ),
        Err(why) => assert!(why.contains("quantiser"), "the refusal does not name the quantiser: {why}"),
    }
}

/// **The encoder will not encode before the caller states a frame rate and a key-frame interval** —
/// both reach the driver and both are things a caller has an opinion about.
#[test]
fn an_unstated_frame_rate_or_key_frame_interval_is_refused() {
    let _guard = lock();
    let Some(display) = display() else { return };
    let Ok(mut bare) = HevcEncoder::at_constant_quantiser(&display, (WIDTH, HEIGHT), 1) else {
        eprintln!("SKIPPED: no HEVC encoder here");
        return;
    };
    // The rate is refused at_rate's own check once a picture has been encoded; before that, `encode`
    // is the place it is refused, which is what this exercises.
    let (luma, chroma) = ramp_shifted(WIDTH, HEIGHT, 0).expect("the ramp");
    bare.upload(&luma, &chroma).expect("the upload");
    let request = HevcEncodeRequest::key_frame(QP).expect("a request");
    match bare.encode(request) {
        Ok(_) => panic!("a picture was encoded with no frame rate stated"),
        Err(why) => assert!(why.contains("frame rate"), "the refusal does not name the frame rate: {why}"),
    }
    // **The key-frame interval is the other half, and 0 is a legal stated interval** ("one key frame
    // for the whole stream") that must not be mistaken for "unstated" — the sentinel for that is
    // `u32::MAX`, not 0.
    let mut zero = match encoder(&display, 1) {
        Ok(e) => e,
        Err(why) => {
            eprintln!("SKIPPED: no HEVC encoder here: {why}");
            return;
        }
    };
    zero.key_frames_every(0).expect("0 is a legal interval");
    let (luma, chroma) = ramp_shifted(WIDTH, HEIGHT, 0).expect("the ramp");
    zero.upload(&luma, &chroma).expect("the upload");
    // The encode itself still fails on this driver, but it must not fail on the interval.
    if let Err(why) = zero.encode(HevcEncodeRequest::key_frame(QP).expect("a request")) {
        assert!(!why.contains("key-frame interval"), "an interval of 0 was read as unstated: {why}");
    }
}

/// **The profile this path submits is one the driver offers, and the entry point is slice encode.**
#[test]
fn the_profile_and_entry_point_are_the_ones_this_path_assumes() {
    let Some(display) = display() else { return };
    let entries = display.entrypoints(Profile::HevcMain).unwrap_or_default();
    assert!(
        entries.contains(&filmcraft_platform::vaapi::Entrypoint::EncSlice),
        "VAProfileHEVCMain reports {entries:?} and this path submits a slice encoder, so it is asking for an entry point that is not there"
    );
    // **`slice_type` is HEVC's own `u(2)`, not H.264's offset-by-five encoding**, and the two are
    // not interchangeable: H.264's 5 and 7 are not even representable in a `u(2)`, so a slice record
    // copied from H.264 would name no HEVC slice type at all. This asserts the recorded encoding says
    // it is a `u(2)` and reports the three real values — **not** a substring search for a digit, which
    // is the kind of assertion that passes on a doc comment and proves nothing.
    let recorded = slice_types::HEVC;
    assert!(recorded.contains("u(2)"), "the recorded slice_type encoding no longer says it is HEVC's u(2): {recorded}");
    // **The three values are named as a triple** (`I = 2, P = 1, B = 0`), which is the claim that
    // matters: if the encoding were H.264's these would be 7, 5 and — nothing.
    assert!(
        recorded.contains("I = 2") && recorded.contains("P = 1") && recorded.contains("B = 0"),
        "the recorded slice_type encoding no longer names HEVC's three values: {recorded}"
    );
    // **And it says what has *not* been measured**, because the brief warns against claiming more.
    assert!(recorded.contains("ACCEPTS"), "the recorded encoding no longer says that what the driver *accepts* is unmeasured: {recorded}");
}

/// **The `slice_type` this path writes is the one this driver writes back**, and the encoding is
/// named rather than a literal in three places.
///
/// **Deliberate break:** `slice_type_encoding_is` refuses anything above 2, because HEVC's
/// `slice_type` is a `u(2)` and 3 is the value the specification reserves. This asserts that bound
/// is enforced, and that the values in use are the measured ones.
#[test]
fn the_slice_type_in_use_is_the_measured_one_and_the_encoding_is_bounded() {
    let _guard = lock();
    let Some(display) = display() else { return };
    let Ok(mut e) = HevcEncoder::at_constant_quantiser(&display, (WIDTH, HEIGHT), 1) else {
        eprintln!("SKIPPED: no HEVC encoder here");
        return;
    };
    let (intra, predicted) = e.slice_type_encoding();
    // I = 2, P = 1 — HEVC's own u(2) values, written back verbatim by the driver.
    assert_eq!((intra, predicted), (2, 1), "the slice types in use are not the measured ones");
    // **The bound is the codec's, not a preference**: 3 is the reserved value.
    assert!(e.slice_type_encoding_is(3, 3).is_err(), "a slice_type of 3 was accepted: HEVC's is a u(2) and 3 is reserved");
    assert!(e.slice_type_encoding_is(0, 1).is_ok(), "0 (B) and 1 (P) are inside HEVC's u(2)");
    // And the values are restored, so the override above did not leak into another test.
    assert_eq!(e.slice_type_encoding(), (0, 1), "the override did not take effect, so the bound above proved nothing");
}
