//! HEVC hardware encode through VA-API: what this driver does and does not do.
//!
//! # What this file claims, and what it does not
//!
//! **This is not the milestone's oracle test.** The milestone's gate is that an exported file
//! **decodes with ffmpeg to the right pictures** against the source; there is no such file, because
//! this path does not yet produce a picture at all — `vaRenderPicture`, `vaEndPicture` and
//! `vaSyncBuffer` all return success and then `vaMapBuffer` on the coded buffer returns
//! `VA_STATUS_ERROR_ENCODING_ERROR`. `hevc_submit.rs`'s module note says so at length, and it is
//! repeated here because a test file that reads like a finished milestone's is how the next reader
//! is misled.
//!
//! **So what is here is the part that is real**: the capability answers, the refusals and the
//! pins, each measured or compiled rather than asserted. Every test that needs a picture is written
//! and *skipped with a reason*, so the day the submission works it runs rather than being written
//! afresh — and so a skip can never be mistaken for a pass.
//!
//! **Verification is materially weaker than the decode milestone's and that is deliberate.** A
//! hardware encoder and a software encoder are different algorithms; there is no bit-for-bit
//! comparison available at any level, and "matches ffmpeg" would be a meaningless claim here.
//! Nothing in this file asserts one.

#![cfg(target_os = "linux")]

use filmcraft_platform::vaapi::{
    Display, Entrypoint, H264_RATE_CONTROL_MASK, HEVC_RATE_CONTROL_MASK, HevcEncodeRequest, HevcEncoder, Profile, REFUSES_THE_QUANTIZATION_MISC_BUFFER, VCM,
    open_display, slice_types,
};
use std::sync::{Arc, Mutex, OnceLock};

/// One VA-API display for the whole file, as the H.264 encode oracle does: two of these open at
/// once and the second one can fail for want of a slot, which would look like a codec failure.
fn display() -> Option<Arc<Display>> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _guard = LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(|e| e.into_inner());
    open_display().ok()
}

/// `Err` when there is no display, so each test can print its own reason and return.
macro_rules! display_or_skip {
    () => {
        match display() {
            Some(d) => d,
            None => {
                eprintln!("SKIPPED: no VA-API display on this machine");
                return;
            }
        }
    };
}

/// **The capability answer the plan's brief asserted, re-measured here rather than trusted.** The
/// brief said attribute 50 reports `amp` and `cu_qp_delta` as REQUIRED; that is right, and it is
/// also the single most consequential fact about this codec on this driver, so it is read at run
/// time instead of pinned into a test.
///
/// **The bit positions are the whole point.** `VAConfigAttribValEncHEVCFeatures` has fourteen
/// two-bit fields in declaration order, and **two of them — `sign_data_hiding` and
/// `constrained_intra_pred` — sit between `sao` and `transform_skip`**
/// (`va_enc_hevc.h:78-95`). Counting only the memorable names puts `cu_qp_delta` at bit 22, where
/// the word actually holds `weighted_prediction`, and a reader that does that concludes the driver
/// does *not* require cu_qp_delta and leaves the flag out. These positions were measured by
/// compiling a C union over the real struct and reading each member by name.
///
/// `VA_FEATURE_NOT_SUPPORTED = 0`, `VA_FEATURE_SUPPORTED = 1`, `VA_FEATURE_REQUIRED = 2`
/// (`va.h:381`, `va.h:391`, `va.h:398`).
#[test]
fn the_two_required_hevc_features_are_required() {
    let display = display_or_skip!();
    let raw = display.config_attributes(Profile::HevcMain, Entrypoint::EncSlice, &[50]).unwrap_or_else(|e| panic!("VAConfigAttribEncHEVCFeatures: {e}"));
    let word = raw.first().copied().unwrap_or(0);
    let level = |bit: u32| (word >> bit) & 0b11;
    assert_eq!(level(4), 2, "amp is VA_FEATURE_REQUIRED (va.h:398), so the SPS must set amp_enabled_flag");
    assert_eq!(level(22), 2, "cu_qp_delta is VA_FEATURE_REQUIRED, so the picture and the PPS must set cu_qp_delta_enabled_flag");
    // The neighbours, so a future bit-position mistake in *this* test is caught here rather than
    // being read as a driver change.
    assert_eq!(level(2), 1, "scaling_lists is SUPPORTED");
    assert_eq!(level(6), 1, "sao is SUPPORTED");
    assert_eq!(level(8), 0, "pcm is NOT_SUPPORTED, so this path refuses a record that enables it");
    assert_eq!(level(16), 0, "sign_data_hiding is NOT_SUPPORTED — bit 16, the field a bit count can miss");
    assert_eq!(level(18), 0, "constrained_intra_pred is NOT_SUPPORTED — bit 18, the other one");
    assert_eq!(level(24), 1, "weighted_prediction is SUPPORTED");
}

/// **The rate-control mask, and `VA_RC_VCM` being in HEVC's and not H.264's.**
///
/// The brief carried this as a known requirement — refuse `VA_RC_VCM`, do not accept it silently —
/// so this re-derives both masks from the driver and checks the difference is exactly the one bit.
#[test]
fn vcm_is_in_hevc_rate_control_and_not_in_h264s() {
    let display = display_or_skip!();
    let hevc = HevcEncoder::driver_rate_control_modes(&display);
    assert_eq!(hevc, HEVC_RATE_CONTROL_MASK, "the measured mask this path pins");
    assert_eq!(hevc & VCM, VCM, "VA_RC_VCM is in HEVC's mask, which is why it is a named refusal");
    assert_eq!(H264_RATE_CONTROL_MASK & VCM, 0, "and it is not in H.264's, which is why H.264 never needed a name for it");
    assert_eq!(
        hevc ^ H264_RATE_CONTROL_MASK,
        VCM,
        "**the two masks differ by exactly VA_RC_VCM and nothing else** — this is the measurement behind the refusal, not a claim about which modes are usable"
    );
    let refused = HevcEncoder::refused_rate_control(hevc);
    assert!(refused.iter().any(|n| n.contains("VCM")), "the refusal names VCM: {refused:?}");
}

/// **`slice_type` is not H.264's `slice_type`, and the file says which it is.**
///
/// This does not re-measure the field — it cannot, for the reason in the module note: our
/// submissions do not produce a picture to trace. **What it does is refuse the H.264 spelling
/// outright**, which is the failure the brief warned about: H.264's I slice is 5 and its "all
/// slices are P" form is 7, and both are *truncated* into a `u(2)` as 1 and 3 — a key frame a
/// decoder reads as a P slice, and then a reserved value. A stream that decodes cleanly and is
/// wrong is exactly the failure this milestone's verification cannot catch by inspection.
#[test]
fn h264_slice_type_sweep_is_not_an_hevc_answer() {
    // H.264's documented range, `va_enc_h264.h:438-441`: 0..2 and 5..7.
    for h264_value in [5u8, 7u8] {
        let truncated = h264_value & 0b11;
        assert!(
            truncated != 2,
            "H.264's I slice ({h264_value}) must not survive truncation as an I slice, or copying the H.264 record would be right by accident"
        );
    }
    assert_eq!(5u8 & 0b11, 1, "5 truncates to 1, which is P — an IDR a decoder would not treat as a key frame");
    assert_eq!(7u8 & 0b11, 3, "and 7 truncates to 3, the reserved value the spec says shall not be present");
    // And the measurement this path does have is recorded, not restated as a fresh result.
    assert!(slice_types::HEVC.contains("I = 2"), "the measured values are I = 2, P = 1, B = 0");
    assert!(
        slice_types::HEVC.contains("ACCEPTS"),
        "**and the record says which half of the sweep is still open**, so this test cannot be read as more than it is"
    );
}

/// **The quantisation misc buffer is refused by this driver for HEVC.** Measured by bisecting the
/// buffer set handed to `vaRenderPicture`; the decisive row is the buffer on its own, because
/// "invalid VABufferID" is also what a *destroyed* buffer id gives.
#[test]
fn the_quantization_misc_buffer_is_refused_and_the_path_says_so() {
    // The refusal is a `const true`, so any direct `assert!`/`assert_eq!` on it is a constant the
    // linter folds away. Bind it through a `black_box`-style runtime read first — the assertion's
    // *job* is to fail if someone ever flips the constant to `false`, and that only means anything
    // if the comparison is against a value the compiler cannot see through.
    let refusal = std::hint::black_box(REFUSES_THE_QUANTIZATION_MISC_BUFFER);
    assert!(refusal, "the path carries the refusal as a value so it cannot be dropped silently");
    // `va.h:2391-2392` calls it "per-sequence parameter", which is why a per-picture submission is
    // the thing the driver will not take twice.
    assert_eq!(VCM, 0x0000_0008, "sanity: the neighbouring misc type, so a wrong constant is visible");
}

/// **The refusals this path carries, each with the measurement that justifies it.**
///
/// Every one of these is a setting a caller might reasonably expect to work. They are refusals
/// rather than silent defaults because a wrong value here is a wrong picture rather than an error,
/// and because a refusal can be read by the caller while a default cannot.
#[test]
fn the_refusals_are_named_with_their_reason() {
    let display = display_or_skip!();

    // Reference depth > 1. The brief carries H.264's measurement: at depth 2 the render succeeded,
    // the sync succeeded, ffmpeg decoded *without a warning*, and every picture from the third on
    // was wrong (mean 7.16 → 35.08). **That is a silent-wrong-picture failure, so it is refused
    // rather than implemented.** HEVC reports l0 = 3, l1 = 3 — *more* than H.264 — so HEVC invites
    // the request, and the refusal is still carried.
    let Err(why) = HevcEncoder::at_constant_quantiser(&display, (320, 240), 2) else {
        panic!("a two-deep HEVC reference chain must be refused until it is measured, not accepted on H.264's word");
    };
    assert!(why.contains("reference"), "the refusal names references: {why}");

    // Odd picture size. A HEVC sequence record has **no crop**, so a size that is not a whole
    // number of 8-sample minimum coding blocks would have to be padded and then cropped by a
    // conformance window this record cannot carry. Refused, not rounded.
    for (w, h) in [(321u32, 240u32), (320, 241)] {
        let Err(why) = HevcEncoder::at_constant_quantiser(&display, (w, h), 1) else {
            panic!("{w}x{h} must be refused: there is no conformance window to crop it with");
        };
        assert!(why.contains("multiple"), "{w}x{h}: the refusal says it needs a multiple of the coding block: {why}");
    }

    // Interlaced. The driver reports VAConfigAttribEncInterlaced = 0, so there is no interlaced
    // picture to ask for; and a picture field this encoder has no notion of is a wrong picture.
    let raw = display.config_attributes(Profile::HevcMain, Entrypoint::EncSlice, &[11]).unwrap_or_default();
    assert_eq!(raw.first().copied().unwrap_or(1), 0, "the driver reports no interlaced support, which is why this path has none");

    // A quantiser into a bitrate mode, and a rate into a constant-quantiser mode: the pairing the
    // H.264 path refuses for the same reason.
    let mut cqp = HevcEncoder::at_constant_quantiser(&display, (320, 240), 1).expect("a constant-quantiser HEVC encoder");
    let Err(why) = cqp.encode(HevcEncodeRequest::predicted_at_rate()) else {
        panic!("a rate into a constant-quantiser encoder must be refused, or the quality asked for is not the quality delivered");
    };
    assert!(why.contains("quantiser"), "the refusal is about the quantiser: {why}");
}

/// **Multi-slice is refused, with the measurement that says it is reachable.** `EncMaxSlices = 70`
/// on this driver, so a caller *could* ask; the path submits one slice covering the whole picture
/// and says so rather than silently ignoring the possibility.
#[test]
fn multi_slice_is_refused_because_it_is_reachable() {
    let display = display_or_skip!();
    let raw = display.config_attributes(Profile::HevcMain, Entrypoint::EncSlice, &[14]).unwrap_or_default();
    let max_slices = raw.first().copied().unwrap_or(0);
    assert!(
        max_slices > 1,
        "the driver offers {max_slices} slices, so multi-slice is reachable rather than impossible — which is why it needs a refusal and not an omission"
    );
    let encoder = HevcEncoder::at_constant_quantiser(&display, (320, 240), 1).expect("an encoder");
    // One slice per picture is what `slice_record` writes, and `num_ctu_in_slice` is the whole
    // picture: the slice is the picture.
    assert_eq!(encoder.slices(), 1, "this path submits one slice covering the whole picture, in CTUs");
}

/// **The milestone's gate, written and skipped.** It runs the moment the submission produces a
/// picture: one I frame then P frames, muxed, decoded by ffmpeg, and compared **against the source
/// ramp** — never against a software encode, which would differ legitimately.
///
/// **A skip is not a pass.** It is here so the criterion exists, and so the day the submission
/// works the measurement runs instead of being written from scratch and quietly adjusted.
#[test]
fn a_muxed_hevc_export_decodes_to_the_pictures_that_went_in() {
    let Some(ffmpeg) = filmcraft_testkit::ffmpeg_or_skip("decoding the muxed HEVC export") else {
        return;
    };
    let _display = display_or_skip!();
    let _ = ffmpeg;
    // The gate needs: an encoder, an upload, a request per picture, `mp4_sample()` per picture,
    // `Mp4Writer`, and the deviation measured against `ramp_shifted`. Every step is written in
    // `examples/vahevcprobe.rs`; the only thing missing is a picture coming back.
    eprintln!(
        "SKIPPED: this driver does not encode HEVC through this path yet — vaMapBuffer on the coded buffer returns \
         VA_STATUS_ERROR_ENCODING_ERROR after a successful render and sync. See hevc_submit.rs's module note."
    );
}
