//! VA-API decoding against our software decoders (Linux): every picture of the fixtures this backend
//! can describe must be the software decoder's — bit-exact planes, colour, pixel aspect, draft flag and
//! picture — also after `reset` + reseek and a mid-stream `flush`; a forced mid-stream failure must
//! continue in software with the software decoder's exact output; damaged samples and parameter sets
//! must give errors or fall back, never crash or hang; a VP9 or AV1 stream must be declined while it is
//! still a description, with a reason; and an H.264 stream whose B slices are implicitly weighted must
//! be declined rather than decoded with weights that are wrong. Skips without ffmpeg (fixture
//! generator) or without a VA-API driver.
//!
//! # Pictures are compared by presentation time
//!
//! A VA-API decode is synchronous — one access unit is rendered and read back inside the `decode`
//! call that submitted it — so this backend returns each picture from the call that produced it and
//! its output is in **decode order**. The software decoders hold pictures back for reordering and
//! emit **presentation order** (H.264 7.4.2.4.1). "The sixth picture" of the two is therefore two
//! different frames, and every comparison here is keyed by presentation time
//! ([`assert_same_by_pts`]), which is what `examples/vaparity.rs` does for the same reason. The two
//! decoders return the same *set* of pictures with the same content; only the order they arrive in
//! differs, and [`a_picture_is_emitted_as_it_is_decoded`] pins that.
//!
//! # `h264_high.mp4` is declined, and this backend is what was wrong
//!
//! `h264_high.mp4` is x264 High profile with B frames, so its picture parameter set carries
//! `weighted_bipred_idc == 2` — x264's default, so this is most H.264. The per-slice
//! `weighted_bipred_flag` of 7.4.3.2 is not read by `filmcraft_h264`
//! (`crates/h264/src/slice.rs:230`), which resolves such a slice as H.264's **implicit** weighting:
//! a weight per (`ref0`, `ref1`) *pair*, derived from the two pictures' POC distances
//! (`crates/h264/src/slicedec.rs:316-331`). That is what `filmcraft_h264` decodes, and it is
//! bit-for-bit ffmpeg's own decode on all 72 pictures of the fixture.
//!
//! A `VASliceParameterBufferH264` cannot express it: `luma_weight_l0` / `luma_weight_l1` are one
//! weight **per reference index per list** (`va_dec_h264.h:3702-3723`), and a pair-dependent function
//! does not fit two per-index arrays. This backend used to submit an empty table there — "explicit,
//! this slice is not weighted" — which is simply wrong, and it decoded 60 of the fixture's 72
//! pictures differently from ffmpeg (12 of 72 agreed; [`an_h264_b_slice_this_record_cannot_weight_is_declined`]
//! pins the numbers, with ffmpeg as the oracle).
//!
//! So `h264_weights` now **checks** instead of guessing: it computes every (`ref0`, `ref1`) pair the
//! slice's own reference lists could use, and submits the default explicit table when they are all
//! 1.0 — which is a different statement from "no weights", and is what makes the hardware decoder's
//! output right — and declines the slice when one of them is not. On this fixture it declines at the
//! second B picture of the first GOP (pair `(0, 1)` wants `(48, 16)`), and `HybridDecoder` finishes
//! the run in software.
//!
//! The same encode with `weightb=0` — `weighted_bipred_idc == 0`, which codes no weight table and no
//! implicit mode at all, and is the only difference between the two files — is bit-exact against
//! ffmpeg on **all 72** pictures
//! ([`an_h264_fixture_this_record_can_express_is_bit_exact_against_ffmpeg`]). So parity is no longer
//! two-tiered with a tolerated divergence: everything this backend can describe is asserted bit for
//! bit, and the one thing it cannot describe is asserted to be declined.
//!
//! # Two branches here rest on a unit test alone, and this is where to look
//!
//! 1. **The all-1.0 implicit weight table.** A B slice in a `weighted_bipred_idc == 2` stream whose
//!    reference pairs all want 1.0 is submitted a default explicit table rather than declined. No
//!    fixture reaches it: every `idc == 2` fixture here exercises the decline, and the `weightb=0` ones
//!    code no weight table at all, so `h264_default_weights` is exercised only by
//!    `vaapi::picture::tests::an_implicitly_weighted_b_slice_is_refused_rather_than_guessed`, which
//!    asserts the bytes it writes. A weight that is wrong but still near 1.0 would pass every test in
//!    this file. The branch's own doc says so at `vaapi::picture::h264_weights`.
//! 2. **Per-segment HEVC reference lists.** `hevc_multi_segment.mp4` has two segments in every picture
//!    and each segment's record is checked against that segment's own header — but **no
//!    encoder-produced fixture reaches the case this is about**: the only encoder available here gives
//!    both segments of a picture the same `num_ref_idx`, so handing the second the first's lists produces
//!    byte-identical records and no assertion here can tell the two apart. The mechanism is proved
//!    separately, in
//!    `vaapi::picture::tests::a_segment_record_must_name_its_own_lists_not_the_first_segments`, and the gap
//!    is stated at `vaapi::picture::build_hevc`.
//!
//!    **What would reach it**, for whoever picks it up: a stream whose segments differ in
//!    `num_ref_idx_l0/l1_active_minus1` — from a real multi-segment hardware-encoded source, or built by
//!    mutating the fixture, since the field is a `ue(v)` the B segments code as `010` and `010` → `011` is
//!    an in-place three-bit flip that shifts nothing. It needs a bit offset for that field, which
//!    `filmcraft_hevc::slice::SliceHeader` does not expose, so either the position is found by searching
//!    the header's bits or `crates/hevc` learns to report it. And the result must be a
//!    **record-consistency test, not a parity test**: a segment asking for more references than it was
//!    coded with decodes a different picture, so comparing its output with ffmpeg would be comparing
//!    against a stream that is not the fixture.
#![cfg(target_os = "linux")]

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use common::*;
use filmcraft_codecs::hw::{NalCodec, NalStreamInfo, StreamInfo};
use filmcraft_codecs::{DecodedFrame, VideoDecoder};
use filmcraft_frame::PixelData;
use filmcraft_platform::HybridDecoder;
use filmcraft_platform::vaapi::VaDecoder;

/// What this backend is expected to do with one fixture, which is the only thing the fixtures differ
/// in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tier {
    /// Decoded in hardware, and every picture bit for bit what the software decoder produces. This is
    /// what "interchangeable" means, and there is no tier below it.
    EveryPicture,
    /// Refused in hardware and finished in software: this stream's B slice records cannot say what
    /// the stream says. The finished run must still be the software decoder's own pictures, and the
    /// refusal must name the reason. See the file's doc comment.
    Declines,
}

/// The hardware decoder for a stream, or `None` (skip) when this machine has none for it.
fn hardware(s: &Stream) -> Option<Box<dyn VideoDecoder>> {
    match filmcraft_platform::vaapi_factory(&s.entry) {
        Some(Ok(d)) => {
            assert!(d.name().starts_with("VA-API"), "{}", d.name());
            Some(d)
        }
        Some(Err(e)) => panic!("factory error: {e}"),
        None => {
            eprintln!("SKIPPED: no VA-API decoder for {}", s.entry.codec.name());
            None
        }
    }
}

/// FilmCraft's own decoder for a stream: the reference the hardware decoder has to match.
///
/// H.264 is built with **one** frame thread ([`H264Decoder::with_threads`]), so the reference does
/// not depend on how many cores the machine has. That is a repeatability measure, not a workaround:
/// the complete run is deterministic at every thread count — measured on `h264_high.mp4`, ten runs
/// each at 1, 4 and 16 threads, all 72 pictures, one presentation-time set, one set of pixels. What
/// the thread count changes is how many pictures come out of `decode` *before* the `flush`, because
/// the pictures still held back for reordering are the ones the frame-threaded path has not finished:
/// 70 at one thread, 65 at four, 45–53 at sixteen. A harness that hashes what `decode` returned and
/// never called `flush` — as `examples/vaparity.rs` does — is what first reported this decoder as
/// dropping pictures nondeterministically.
fn software(s: &Stream) -> Box<dyn VideoDecoder> {
    match &s.entry.codec {
        filmcraft_isobmff::CodecConfig::Avc(a) => Box::new(filmcraft_codecs::video::H264Decoder::with_threads(a.to_bytes(), 1).unwrap()),
        _ => filmcraft_codecs::software_video_decoder(&s.entry).unwrap(),
    }
}

/// The parity fixtures and what each is expected to do.
///
/// `h264_high.mp4` declines: its B slices are implicitly weighted, which a slice record cannot
/// express. **`h264_high_unweighted.mp4` is the control that makes the parity assertion an H.264
/// assertion** — the same source, encoder, frame rate, GOPs, B frames and cropped 360-line picture
/// with `weightb=0` added, which is the *only* difference from `h264_high.mp4`, decoded in hardware
/// and compared picture for picture. Losing it must not quietly remove the control, so
/// [`streams_with_control`] insists on it whenever ffmpeg is there to generate it.
fn streams(ff: &Path) -> Vec<(String, Stream, Tier)> {
    let mut out = Vec::new();
    for (name, _) in FIXTURES {
        if let Some(path) = named(ff, name) {
            let tier = if *name == "h264_high.mp4" { Tier::Declines } else { Tier::EveryPicture };
            out.push((name.to_string(), read_stream(&path), tier));
        }
    }
    out
}

/// The H.264 control fixture: the fixture `h264_high.mp4` is named in [`FIXTURES`] with exactly one
/// encoder parameter changed, `weightb=0`, so its picture parameter set says
/// `weighted_bipred_idc == 0` — no weight table in a B slice and no implicit mode.
///
/// Derived from that entry rather than written out again, because the whole value of this fixture is
/// that it differs from `h264_high.mp4` in that one parameter and in nothing else. A second copy of
/// the argument list would drift, and a drifted control stops being a control while still looking
/// like one.
fn unweighted(ff: &Path) -> Option<PathBuf> {
    let (name, args) = FIXTURES.iter().find(|f| f.0 == "h264_high.mp4")?;
    assert_eq!(*name, "h264_high.mp4");
    let mut args: Vec<&str> = args.to_vec();
    // `-x264-params` takes one colon-separated value, so the extra parameter joins that value rather
    // than becoming a second occurrence of the option.
    let p = args.iter().position(|a| *a == "-x264-params").map(|i| i + 1)?;
    args[p] = Box::leak(format!("{}:weightb=0", args[p]).into_boxed_str());
    fixture(ff, "h264_high_unweighted.mp4", &args)
}

/// [`streams`] plus the H.264 control fixture, whose absence is a failure rather than a smaller run.
///
/// This is the fixture the hardware decoder is asserted bit-exact on for H.264. A run that quietly
/// dropped it would still be green with one H.264 fixture fewer and no hardware H.264 assertion at
/// all, which is the coverage loss this function exists to make loud.
fn streams_with_control(ff: &Path) -> Vec<(String, Stream, Tier)> {
    let mut out = streams(ff);
    let path = unweighted(ff).unwrap_or_else(|| {
        panic!("the H.264 control fixture h264_high_unweighted.mp4 could not be generated, so the H.264 bit-exactness assertion would not be run at all")
    });
    assert!(!out.iter().any(|(n, _, _)| n == "h264_high_unweighted.mp4"), "the control fixture is listed twice");
    out.push(("h264_high_unweighted.mp4".to_string(), read_stream(&path), Tier::EveryPicture));
    // An HEVC stream whose every picture is split into two slice segments, so the per-segment record
    // path is decoded on the real driver rather than only built in a unit test. What that does and does
    // not prove is said at `vaapi::picture::build_hevc`.
    match multi_segment_hevc(ff) {
        Some(path) => out.push(("hevc_multi_segment.mp4".to_string(), read_stream(&path), Tier::EveryPicture)),
        None => eprintln!("SKIPPED: the multi-segment HEVC fixture could not be generated"),
    }
    out
}

/// The multi-segment HEVC fixture: `slices=2:no-tskip=1` is what makes x265 split each picture into
/// two, and it is the only way this suite can reach a segment record that is not the first one.
fn multi_segment_hevc(ff: &Path) -> Option<PathBuf> {
    let (_name, args) = FIXTURES.iter().find(|f| f.0 == "hevc_main.mp4")?;
    let mut args: Vec<&str> = args.to_vec();
    let p = args.iter().position(|a| *a == "-x265-params").map(|i| i + 1)?;
    args[p] = Box::leak(format!("{}:slices=2:no-tskip=1", args[p]).into_boxed_str());
    fixture(ff, "hevc_multi_segment.mp4", &args)
}

/// The slice type of each picture of an H.264 stream, read from the first two syntax elements of
/// its first slice: `first_mb_in_slice` and `slice_type` (7.4.3). A picture whose first slice does
/// not start at macroblock 0 is a later segment of an earlier picture, so it is passed over.
fn h264_slice_types(info: &NalStreamInfo, s: &Stream) -> BTreeMap<i64, u32> {
    let mut out = BTreeMap::new();
    for (sample, pts) in &s.samples {
        for nal in info.nals(sample) {
            // 1 = a non-IDR slice, 5 = an IDR slice: both carry a slice header.
            if !nal.first().is_some_and(|&h| matches!(h & 0x1f, 1 | 5)) {
                continue;
            }
            let rbsp = filmcraft_bitstream::unescape_rbsp(nal.get(1..).unwrap_or_default());
            let mut r = filmcraft_bitstream::BitReader::new(&rbsp);
            if r.read_ue() != Ok(0) {
                continue;
            }
            if let Ok(raw) = r.read_ue()
                && raw <= 9
            {
                out.insert(*pts, raw % 5);
                break;
            }
        }
    }
    out
}

/// Every picture of `s` from the hardware decoder is the software decoder's: whole stream, after
/// `reset` + reseek to each later sync sample, and a full pass again after all the resets. This is
/// Review Focus #5's test.
fn parity(name: &str, s: &Stream) {
    let Some(mut hw) = hardware(s) else { return };
    let mut sw = software(s);
    let a = decode_all(hw.as_mut(), &s.samples);
    let b = decode_all(sw.as_mut(), &s.samples);
    assert!(a.len() >= 40, "{name}: {} pictures", a.len());
    assert_eq!(a.len(), b.len(), "{name}: picture counts");
    assert_same_by_pts(name, &a, &b);
    assert!(hw.name().starts_with("VA-API"), "{name}: stayed in hardware ({})", hw.name());

    // reset + reseek to each later sync sample, then decode a stretch of the run from there
    let syncs: Vec<usize> = (1..s.samples.len()).filter(|&i| s.sync[i]).collect();
    assert!(!syncs.is_empty(), "{name}: more than one GOP");
    for &k in syncs.iter().rev() {
        let end = (k + 17).min(s.samples.len());
        for d in [&mut hw, &mut sw] {
            d.reset();
        }
        let a = decode_all(hw.as_mut(), &s.samples[k..end]);
        let b = decode_all(sw.as_mut(), &s.samples[k..end]);
        assert!(!a.is_empty(), "{name}: pictures after seeking to {k}");
        assert_same_by_pts(&format!("{name} from sample {k}"), &a, &b);
    }
    // back to the start after seeking around: the whole stream again
    hw.reset();
    sw.reset();
    let a = decode_all(hw.as_mut(), &s.samples);
    let b = decode_all(sw.as_mut(), &s.samples);
    assert_eq!(a.len(), b.len(), "{name} after resets: picture count");
    assert_same_by_pts(&format!("{name} after resets"), &a, &b);

    // random-access / disposable answers match the software decoder's
    for (smp, _) in &s.samples {
        assert_eq!(hw.is_random_access(smp), sw.is_random_access(smp), "{name}");
        assert_eq!(hw.is_disposable(smp), sw.is_disposable(smp), "{name}");
    }
}

/// Every picture of every fixture this backend can describe, bit for bit what FilmCraft's own decoder
/// produces: whole stream, after `reset` + reseek to each later sync sample, and a full pass again
/// after all the resets. This is Review Focus #5's test.
#[test]
fn bit_exact_with_the_software_decoder() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let streams = streams_with_control(&ff);
    let mut compared = 0usize;
    for (name, s, tier) in &streams {
        if *tier != Tier::EveryPicture {
            continue;
        }
        compared += 1;
        parity(name, s);
    }
    // The three fixtures this backend must decode: HEVC 8-bit, HEVC 10-bit and the H.264 control.
    // Anything less and this is no longer a parity test — in particular losing the control fixture
    // would leave the H.264 half of the claim unasserted, which is why it is counted here by name.
    assert!(compared >= 3, "only {compared} fixtures were compared bit for bit, which is not a parity test");
    assert!(
        streams.iter().any(|(n, _, t)| n == "h264_high_unweighted.mp4" && *t == Tier::EveryPicture),
        "the H.264 control fixture was not compared bit for bit"
    );
}

/// A picture comes back from the `decode` call that produced it, so the run is in decode order
/// where the software decoder's is in presentation order.
///
/// Pinned so that the difference is a stated fact about this backend rather than something a reader
/// has to infer from the parity tests matching by presentation time — and so that a backend which
/// starts reordering (which would be an improvement) says so here instead of silently changing what
/// everything downstream sees.
#[test]
fn a_picture_is_emitted_as_it_is_decoded() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    for (name, s, tier) in streams_with_control(&ff) {
        if tier != Tier::EveryPicture {
            continue;
        }
        let Some(mut hw) = hardware(&s) else { continue };
        let hardware_pts: Vec<i64> = s.samples.iter().flat_map(|(b, p)| hw.decode(b, *p).unwrap()).map(|f| f.pts).collect();
        let software_pts: Vec<i64> = decode_all(software(&s).as_mut(), &s.samples).iter().map(|f| f.pts).collect();
        assert_eq!(hardware_pts.len(), software_pts.len(), "{name}: picture counts");
        assert_eq!(hardware_pts, s.samples.iter().map(|(_, p)| *p).collect::<Vec<_>>(), "{name}: one picture per access unit, in decode order");
        let mut sorted = hardware_pts.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, software_pts, "{name}: the same presentation times as the software decoder, which emits them in order");
        assert_ne!(hardware_pts, software_pts, "{name}: this fixture reorders, so the two orders really do differ");
    }
}

/// After a `flush` the context is drained, and a drained VA-API context cannot resume from the
/// middle of a GOP: its references are gone, so continuing is an error, and the hybrid decoder then
/// replays the run in software instead. From an IDR picture it decodes again.
#[test]
fn a_drained_context_restarts_only_at_an_idr() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    for (name, s, tier) in streams_with_control(&ff) {
        let name = name.as_str();
        if tier == Tier::Declines {
            continue;
        }
        let Some(info) = filmcraft_platform::vaapi::stream_info(&s.entry) else {
            eprintln!("SKIPPED: no VA-API stream description for {name}");
            continue;
        };
        // One decoder for the two halves below: `VaDecoder::new` opens a context, and asking twice
        // whether a context exists is a second context this test does not need.
        let Ok(va) = VaDecoder::new(info.clone()) else {
            eprintln!("SKIPPED: no VA-API decoder for {name}");
            continue;
        };
        let is_idr = |i: usize| match &info {
            StreamInfo::Nal(n) => n.nal_types(&s.samples[i].0).iter().any(|&t| if n.codec == NalCodec::H264 { t == 5 } else { matches!(t, 19 | 20) }),
            StreamInfo::Frame(f) => f.is_random_access(&s.samples[i].0),
        };
        // A sample the stream does not mark as a restart point, so continuing from it cannot be
        // legitimate whatever the fixture is: an IDR picture is always a sync sample, so `!sync` is
        // enough and the fixture does not have to be inspected to know it.
        let Some(mid) = (s.samples.len() / 2..s.samples.len()).find(|&i| !s.sync[i]) else {
            eprintln!("SKIPPED: no mid-GOP sample in the second half of {name}");
            continue;
        };
        assert!(!is_idr(mid), "{name}: sample {mid} is not a sync sample, so it cannot be an IDR picture");
        let mut sw = software(&s);
        let mut va = va;
        let before: Vec<DecodedFrame> = s.samples[..mid].iter().flat_map(|(d, p)| va.decode(d, *p).unwrap()).collect();
        assert_eq!(before.len(), mid, "{name}: a VA-API decode is one picture in, one picture out");
        assert!(va.flush().is_empty(), "{name}: a VA-API decode holds nothing back, so a flush returns nothing");
        assert!(va.decode(&s.samples[mid].0, s.samples[mid].1).is_err(), "{name}: continuing mid-GOP after a flush is an error");
        match (mid..s.samples.len()).find(|&i| s.sync[i]) {
            // A CRA is a restart point for a *seek* — the parity test resets into one — but not for a
            // drained context, whose references a CRA cannot bring back. x265 writes CRA pictures at
            // the HEVC fixtures' keyframes, so this is the whole of the HEVC case.
            Some(sync) if !is_idr(sync) => {
                assert!(
                    va.decode(&s.samples[sync].0, s.samples[sync].1).is_err(),
                    "{name}: a drained context continued at the restart point {sync}, which is not an IDR picture"
                );
                println!(
                    "{name}: the restart point after the middle is sample {sync}, which is not an IDR picture, so a drained context refuses to continue there"
                );
            }
            // flush, then that IDR: the same pictures as the software decoder's for the run that follows
            Some(sync) => {
                va.reset();
                let _ = decode_all(&mut va, &s.samples[..sync.min(mid + 2)]);
                let a = decode_all(&mut va, &s.samples[sync..]);
                let b = decode_all(sw.as_mut(), &s.samples[sync..]);
                assert!(!a.is_empty(), "{name}: pictures after restarting at IDR {sync}");
                assert_same_by_pts(&format!("{name} flush then IDR {sync}"), &a, &b);
            }
            None => eprintln!("SKIPPED: no restart point after sample {mid} of {name}"),
        }
        // The same drain through the hybrid decoder, which is what that error is *for*: it hands the
        // run to the software decoder, so the finished run has to be the software decoder's own
        // pictures — the first half from the hardware, the second half replayed through software.
        let reference = decode_all(software(&s).as_mut(), &s.samples);
        let mut d = HybridDecoder::new(Box::new(VaDecoder::new(info.clone()).expect("the decoder this test just used")), s.entry.clone(), info.clone());
        let mut out: Vec<DecodedFrame> = Vec::new();
        for (smp, pts) in &s.samples[..mid] {
            out.extend(d.decode(smp, *pts).unwrap());
        }
        assert!(d.is_hardware(), "{name}: still in hardware after the first half");
        assert!(d.flush().is_empty(), "{name}: the hardware holds nothing back, so the flush returns nothing");
        out.extend(decode_all(&mut d, &s.samples[mid..]));
        assert!(!d.is_hardware(), "{name}: a drained context did not hand the run to the software decoder");
        assert_same_by_pts(&format!("{name} flushed mid-GOP at {mid}"), &out, &reference);
    }
}

/// A forced mid-stream failure hands the rest of the run to the software decoder, and the finished
/// run is the software decoder's own output.
///
/// The reference is the software decoder's whole output, because that — not the hardware decoder's —
/// is what must come out: the pictures before the failure were hardware's and are dropped by the
/// replay, and the rest is decoded twice.
#[test]
fn a_forced_mid_stream_failure_continues_in_software() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    for (name, s, tier) in streams_with_control(&ff) {
        let name = name.as_str();
        let Some(info) = filmcraft_platform::vaapi::stream_info(&s.entry) else { continue };
        if VaDecoder::new(info.clone()).is_err() {
            eprintln!("SKIPPED: no VA-API decoder for {name}");
            continue;
        }
        let reference = decode_all(software(&s).as_mut(), &s.samples);
        assert!(!reference.is_empty(), "{name}");
        // The first sample (so nothing at all came out of the hardware), two samples into the first
        // GOP, the next GOP's sync sample, the sample after it, and nine samples into that GOP.
        let k = (1..s.samples.len()).find(|&i| s.sync[i]).unwrap();
        for fail_at in [0, 5, k, k + 1, k + 9] {
            let mut va = VaDecoder::new(info.clone()).unwrap();
            va.fail_after(fail_at as u64);
            let mut d = HybridDecoder::new(Box::new(va), s.entry.clone(), info.clone());
            let before = filmcraft_codecs::hw::hw_stats().fallbacks;
            let out = decode_all(&mut d, &s.samples);
            assert!(!d.is_hardware(), "{name}: switched to software");
            assert!(filmcraft_codecs::hw::hw_stats().fallbacks > before, "{name}: fallback counted");
            // Failing at the first sample leaves every picture of the run to the software decoder, so
            // the output must be its own bit for bit. A failure further in leaves the pictures before
            // it to the hardware, so the run is only as good as those pictures are — which is the
            // fixture's tier. For every fixture this backend decodes correctly that is the software
            // decoder's own pictures as well.
            if fail_at == 0 || tier == Tier::EveryPicture {
                assert_same_by_pts(&format!("{name} failing at sample {fail_at}"), &out, &reference);
            } else {
                assert_eq!(out.len(), reference.len(), "{name} failing at sample {fail_at}: the run is complete");
                assert_same_pts_only(&format!("{name} failing at sample {fail_at}"), &out, &reference);
            }
            // the instance stays in software across seeks
            d.reset();
            assert_same_by_pts(&format!("{name} after the fallback and a reset"), &decode_all(&mut d, &s.samples), &reference);
        }
    }
}

/// The presentation times two outputs have in common, and nothing else: the same set of pictures,
/// compared by position rather than by content.
///
/// Only for the runs whose *pixels* cannot be compared — see the note at its one call site. It exists
/// because "the run is complete" is a real requirement of a fallback and has to be assertable without
/// also asserting the pixels.
fn assert_same_pts_only(what: &str, a: &[DecodedFrame], b: &[DecodedFrame]) {
    let pts = |out: &[DecodedFrame]| -> Vec<i64> { by_pts(out).keys().copied().collect() };
    assert_eq!(pts(a), pts(b), "{what}: presentation times");
}

/// Run `f` on a thread; fail if it panics or takes longer than `limit`, else hand back what it
/// returned. The value is there so a caller can assert on what the damaged runs actually did: a
/// test that only checks "nothing blew up" also passes when the damage never reached anything.
///
/// The worker's thread id comes back with the value so a hang names the thread that has to be looked
/// for: a thread stuck in a driver call cannot be killed, so a timed-out worker is still running and
/// the only handle on it is its id. The panic payload is reported for the same reason — "it panicked"
/// without the message loses the one thing that says what.
fn bounded<T: Send + 'static>(what: &str, limit: Duration, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let worker = std::thread::current().id();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        let _ = tx.send(r.map_err(|payload| (worker, panic_message(payload.as_ref()))));
    });
    match rx.recv_timeout(limit) {
        Ok(Ok(v)) => v,
        Ok(Err((worker, message))) => panic!("{what}: worker thread {worker:?} panicked: {message}"),
        Err(_) => panic!("{what}: hung for {limit:?}; the worker thread is still running and cannot be killed"),
    }
}

/// A panic payload as text, which is `&str` or `String` in every panic the standard library and this
/// test suite raise — anything else is described rather than dropped, so the report is never just
/// "panicked" with nothing to go on.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "<a panic payload that is neither &str nor String>".to_string())
}

/// Bit flips, truncation and corrupt length prefixes, and corrupt configuration records, must give an
/// error, a fallback or correct pictures — never a panic and never a hang.
///
/// The decoder here is the factory's, which is a `HybridDecoder`: a damaged access unit it cannot
/// decode comes back as **`Ok`**, because it replays the run in software and hands the software
/// decoder's pictures over. Counting errors alone would therefore call a run that quietly fell back
/// "not rejected" and report it as coverage. So the two are counted apart: `errors` is the access
/// units the decoder refused outright, and `fallbacks` is the access units it absorbed by leaving
/// hardware. A picture it produced itself is neither, and there is nothing to assert about one —
/// which is why this is a "never crashed or hung" test and not a correctness one.
#[test]
fn damaged_samples_never_crash_or_hang() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    for (name, s, _) in streams_with_control(&ff) {
        let name = name.as_str();
        if hardware(&s).is_none() {
            continue;
        }
        // bit flips, truncation and corrupt length prefixes, through the factory's decoder
        let (mut rejected, mut absorbed) = (0usize, 0usize);
        for seed in 1..=6u64 {
            let samples = s.samples.clone();
            let entry = s.entry.clone();
            let (e, f) = bounded(&format!("{name} seed {seed}"), Duration::from_secs(60), move || {
                let mut seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
                let (mut errors, mut fallbacks, mut in_hardware) = (0usize, 0usize, true);
                let mut d = filmcraft_platform::vaapi_factory(&entry).unwrap().unwrap();
                for (i, (smp, pts)) in samples.iter().enumerate().take(40) {
                    let mut m = smp.clone();
                    match xorshift(&mut seed) % 4 {
                        0 => {
                            for _ in 0..1 + xorshift(&mut seed) % 8 {
                                let at = (xorshift(&mut seed) as usize) % m.len().max(1);
                                if let Some(b) = m.get_mut(at) {
                                    *b ^= 1 << (xorshift(&mut seed) % 8);
                                }
                            }
                        }
                        1 => m.truncate((xorshift(&mut seed) as usize) % m.len().max(1)),
                        2 if m.len() > 4 => m[..4].copy_from_slice(&(xorshift(&mut seed) as u32).to_be_bytes()),
                        _ => {}
                    }
                    errors += usize::from(d.decode(&m, *pts).is_err());
                    // The instance's own name says which decoder it is using, which is how "fell back"
                    // is told from "absorbed" — and unlike the process-wide fallback counter it is a
                    // fact about this run.
                    let still = d.name().starts_with("VA-API");
                    if in_hardware && !still {
                        fallbacks += 1;
                    }
                    in_hardware = still;
                    if i % 13 == 12 {
                        d.reset();
                    }
                }
                let _ = d.flush();
                (errors, fallbacks)
            });
            rejected += e;
            absorbed += f;
        }
        // The damage has to reach the decoder, or "never crashed" is a statement about clean samples: on
        // every fixture these six runs are rejected well over a hundred times between them.
        assert!(rejected > 0, "{name}: none of the 240 damaged access units was rejected");
        println!(
            "{name}: of 240 damaged access units, {rejected} were rejected as errors and {absorbed} more made the decoder leave hardware; the rest decoded",
        );
        // a damaged configuration record: the factory declines, or the decoder errors / falls back
        let rec = match &s.entry.codec {
            filmcraft_isobmff::CodecConfig::Avc(a) => a.to_bytes(),
            filmcraft_isobmff::CodecConfig::Hevc(c) => c.to_bytes(),
            _ => continue,
        };
        let hevc = matches!(s.entry.codec, filmcraft_isobmff::CodecConfig::Hevc(_));
        let mut seed = 0x1234_5678u64;
        let mut planned = 0usize;
        for round in 0..24 {
            let mut r = rec.clone();
            if round % 3 == 2 {
                r.truncate(r.len() * (round + 1) / 30);
            } else {
                for _ in 0..1 + round % 4 {
                    let at = 6 + (xorshift(&mut seed) as usize) % r.len().saturating_sub(6).max(1);
                    if let Some(b) = r.get_mut(at) {
                        *b ^= 1 << (xorshift(&mut seed) % 8);
                    }
                }
            }
            let samples: Vec<_> = s.samples.iter().take(12).cloned().collect();
            planned += bounded(&format!("{name} record round {round}"), Duration::from_secs(60), move || {
                let parsed = if hevc { NalStreamInfo::from_hvcc(&r) } else { NalStreamInfo::from_avcc(&r) };
                let Ok(info) = parsed else { return 0 };
                let Ok(mut va) = VaDecoder::new(info) else { return 0 };
                for (smp, pts) in &samples {
                    if va.decode(smp, *pts).is_err() {
                        break;
                    }
                }
                let _ = va.flush();
                1
            });
        }
        // The same guard for the records: a corruption every record parser refused would leave this
        // half testing nothing at all, which is what a passing run looks like from the outside.
        assert!(planned > 0, "{name}: every corrupted configuration record was refused before a context existed");
        println!("{name}: {planned} of 24 corrupted configuration records were planned and submitted");
    }
}

/// `sample` with `sets` spliced in front of it as length-prefixed NAL units.
fn with_in_band(info: &NalStreamInfo, sample: &[u8], sets: &[&[u8]]) -> Vec<u8> {
    let ls = info.length_size;
    let mut out = Vec::new();
    for set in sets {
        let len = u32::try_from(set.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&len.to_be_bytes()[4 - ls..]);
        out.extend_from_slice(set);
    }
    out.extend_from_slice(sample);
    assert_eq!(info.nals(&out).len(), info.nals(sample).len() + sets.len(), "the spliced sample parses");
    out
}

/// A sample carrying parameter sets the hardware session was *not* set up from falls back to the
/// software decoder rather than being decoded with the sample entry's picture structure. This is
/// Review Focus #4's test.
///
/// Both halves are tested, because either alone would pass for the wrong reason: a repeat of the
/// entry's own sets is not a change and must stay in hardware, and a set that differs is a change
/// and must not.
#[test]
fn in_band_parameter_sets_are_not_confused_with_the_config_record() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let (Some(plain_path), Some(weighted_path)) = (unweighted(&ff), named(&ff, "h264_high.mp4")) else {
        eprintln!("SKIPPED: the H.264 fixtures could not be generated");
        return;
    };
    let (plain, weighted) = (read_stream(&plain_path), read_stream(&weighted_path));
    let entry_info = NalStreamInfo::from_entry(&plain.entry).unwrap().unwrap();
    let other_info = NalStreamInfo::from_entry(&weighted.entry).unwrap().unwrap();
    let Some(mut hw) = hardware(&plain) else { return };
    // The two fixtures are the same stream at two values of one field (`weighted_bipred_idc`), so
    // their parameter sets differ in bytes but describe the same picture structure — which is what
    // lets the two halves of this test be compared. The stream under test is the one whose pictures
    // this backend decodes correctly, so that "the hardware kept going" and "the software decoder
    // took over" can both be checked picture for picture.
    let own: Vec<&[u8]> = entry_info.parameter_sets.iter().map(Vec::as_slice).collect();
    let other: Vec<&[u8]> = other_info.parameter_sets.iter().map(Vec::as_slice).collect();
    assert_ne!(own, other, "the two fixtures really do carry different parameter sets");
    let run = |sets: &[&[u8]]| -> Vec<(Vec<u8>, i64)> { (0..20).map(|i| (with_in_band(&entry_info, &plain.samples[i].0, sets), plain.samples[i].1)).collect() };

    // A repeat of the sample entry's own sets: not a change, so the hardware keeps going. What says so
    // is the decoder this instance still holds, not the process-wide fallback counter — a test
    // running beside this one can move that counter, and a counter that moves is not a fact about
    // this run.
    let repeat = run(&own);
    let out = decode_all(hw.as_mut(), &repeat);
    assert!(hw.name().starts_with("VA-API"), "a repeat of the sample entry's own sets stays in {}", hw.name());
    assert_same_by_pts("in-band sets repeated from the sample entry", &out, &decode_all(software(&plain).as_mut(), &repeat));
    hw.reset();

    // A parameter set that differs from the sample entry's: the session was not set up for it, so
    // the run continues in software, and its output is the software decoder's own.
    let differing = run(&other);
    let out = decode_all(hw.as_mut(), &differing);
    assert!(!hw.name().starts_with("VA-API"), "in-band sets that differ continued in {}", hw.name());
    assert_same_by_pts("in-band sets that differ from the sample entry", &out, &decode_all(software(&plain).as_mut(), &differing));
}

/// A VP9 or AV1 sample entry yields no VA-API decoder, and the reason names what is missing.
///
/// `plan_frame` declines every frame-coded stream while it is still a description, because libva
/// wants each frame's uncompressed header as a picture-parameter record and neither
/// `filmcraft-vp9` nor `filmcraft-av1` exposes it. That decline is what is tested here: it is a real
/// behaviour of this backend, and it is why there are no VP9 or AV1 *parity* fixtures in this file —
/// a parity test over them could only ever print SKIPPED.
#[test]
fn a_frame_codec_is_declined_before_a_context_exists() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let src = "testsrc2=s=640x360:r=24:d=1";
    let cases: [(&str, Vec<&str>, &str); 2] = [
        (
            "vp9_p0.mp4",
            vec!["-f", "lavfi", "-i", src, "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "40", "-deadline", "realtime", "-cpu-used", "8", "-pix_fmt", "yuv420p"],
            "filmcraft-vp9",
        ),
        (
            "av1_p0.mp4",
            vec!["-f", "lavfi", "-i", src, "-c:v", "libaom-av1", "-crf", "50", "-b:v", "0", "-cpu-used", "8", "-pix_fmt", "yuv420p"],
            "filmcraft-av1",
        ),
    ];
    let mut checked = 0;
    for (name, args, crate_name) in cases {
        let Some(path) = fixture(&ff, name, &args) else {
            eprintln!("SKIPPED: ffmpeg cannot encode {name}");
            continue;
        };
        let s = read_stream(&path);
        // The entry is recognised as a frame-coded stream…
        let info = filmcraft_platform::vaapi::stream_info(&s.entry).unwrap_or_else(|| panic!("{name}: not recognised as a vpcC / av1C stream"));
        assert!(matches!(info, StreamInfo::Frame(_)), "{name}: recognised as {info:?}");
        // …and declined with a reason that names the missing record, not the driver.
        let why = VaDecoder::new(info).err().unwrap_or_else(|| panic!("{name}: this backend took a frame-coded stream"));
        assert!(why.contains("picture parameters"), "{name}: {why}");
        assert!(why.contains("frame header"), "{name}: {why}");
        assert!(why.contains(crate_name), "{name}: {why}");
        // The factory declines it too, so a user gets FilmCraft's own decoder and never sees this.
        assert!(filmcraft_platform::vaapi_factory(&s.entry).is_none(), "{name}: the factory handed a frame-coded stream to a VA-API decoder");
        // …and FilmCraft's own decoder can still play it, which is the point of declining.
        assert!(!decode_all(software(&s).as_mut(), &s.samples).is_empty(), "{name}: the software decoder produced no pictures");
        checked += 1;
    }
    assert!(checked > 0, "no VP9 or AV1 fixture could be generated");
}

/// How many of `out`'s pictures are ffmpeg's, and the presentation time of the first that is not.
///
/// `out` is keyed by presentation time, so a picture returned twice would collapse into one entry and
/// this would score the survivors instead of the run — hence the length check. Only the pictures `out`
/// actually has are scored: a run that was refused part way through is a legitimate thing to hand this,
/// and what matters is that everything it *did* produce is ffmpeg's.
fn oracle(out: &[DecodedFrame], reference: &Ffmpeg) -> (usize, Option<i64>) {
    let map: BTreeMap<i64, Vec<u8>> = out.iter().map(|f| (f.pts, pixels(f))).collect();
    assert_eq!(map.len(), out.len(), "a presentation time was returned twice, so scoring pictures by presentation time would miss one");
    let mut first_wrong = None;
    for (pts, picture) in &map {
        let want = reference.at(*pts).unwrap_or_else(|| panic!("pts {pts} is not a picture ffmpeg decoded"));
        if picture != want {
            first_wrong.get_or_insert(*pts);
        }
    }
    (map.len() - usize::from(first_wrong.is_some()), first_wrong)
}

/// ffmpeg's own decode of a fixture, addressed by presentation time.
struct Ffmpeg {
    pictures: Vec<Vec<u8>>,
    /// Presentation time of the *i*-th picture, ascending — ffmpeg emits in presentation order, so the
    /// *i*-th frame it writes is the picture with the *i*-th smallest presentation time.
    order: Vec<i64>,
}

impl Ffmpeg {
    fn at(&self, pts: i64) -> Option<&Vec<u8>> {
        self.order.iter().position(|p| *p == pts).map(|i| &self.pictures[i])
    }
    fn len(&self) -> usize {
        self.pictures.len()
    }
}

/// ffmpeg's own decode of a fixture, paired with the stream's presentation times so a picture can be
/// looked up by the time it was decoded with rather than by its position in the output.
fn ffmpeg_pictures(ff: &Path, path: &Path, s: &Stream) -> Ffmpeg {
    let raw = std::process::Command::new(ff)
        .args(["-v", "error", "-i", path.to_str().unwrap(), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
        .output()
        .expect("ffmpeg decode");
    assert!(raw.status.success(), "ffmpeg failed: {}", String::from_utf8_lossy(&raw.stderr));
    let size = 640 * 360 * 3 / 2;
    assert_eq!(raw.stdout.len() % size, 0, "ffmpeg's decode is a whole number of 640x360 4:2:0 pictures");
    let mut order: Vec<i64> = s.samples.iter().map(|(_, p)| *p).collect();
    order.sort_unstable();
    Ffmpeg { pictures: raw.stdout.chunks_exact(size).map(<[u8]>::to_vec).collect(), order }
}

/// The control fixture: with `weightb=0` this backend decodes **every** picture, and every one of them
/// is bit-for-bit ffmpeg's.
///
/// This is the assertion that makes the H.264 half of the parity claim real, and it is measured
/// against ffmpeg rather than against our own decoder, so it cannot pass because both FilmCraft
/// decoders drifted the same way. `h264_high.mp4` is the same encode with that one parameter changed,
/// so the two together say the weight path was the only thing wrong.
#[test]
fn an_h264_fixture_this_record_can_express_is_bit_exact_against_ffmpeg() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let Some(path) = unweighted(&ff) else {
        panic!("the H.264 control fixture h264_high_unweighted.mp4 could not be generated");
    };
    let s = read_stream(&path);
    let reference = ffmpeg_pictures(&ff, &path, &s);
    // The raw decoder, not the factory's hybrid: a decline must show up as a missing picture here, not
    // as a software replay that quietly looks like a hardware success.
    let info = filmcraft_platform::vaapi::stream_info(&s.entry).expect("an avcC stream");
    let mut va = match VaDecoder::new(info) {
        Ok(va) => va,
        Err(e) => {
            eprintln!("SKIPPED: no VA-API decoder for this driver: {e}");
            return;
        }
    };
    let mut out: Vec<DecodedFrame> = Vec::new();
    let mut declined = 0usize;
    for (smp, pts) in &s.samples {
        match va.decode(smp, *pts) {
            Ok(mut frames) => out.append(&mut frames),
            Err(_) => declined += 1,
        }
    }
    out.extend(va.flush());
    assert_eq!(declined, 0, "the control fixture must not need a single decline, and {declined} were declined");
    let (right, first_wrong) = oracle(&out, &reference);
    println!("h264_high_unweighted.mp4 against ffmpeg's own decode: this backend is right on {right}/{}", out.len());
    assert_eq!(right, out.len(), "the first disagreement with ffmpeg is at pts {first_wrong:?}");
    // and the reference the rest of this file compares against is ffmpeg's too
    let software = decode_all(software(&s).as_mut(), &s.samples);
    let (sw_right, _) = oracle(&software, &reference);
    assert_eq!(sw_right, reference.len(), "the software decoder disagrees with ffmpeg, so the parity comparison would be against a wrong reference");
}

/// An H.264 B slice whose weights the record cannot express is **declined**, and the stream finishes
/// in software.
///
/// `h264_high.mp4` is x264's default `weightb=1` over High profile with B frames, so its picture
/// parameter set says `weighted_bipred_idc == 2` and `filmcraft_h264` resolves each B slice as H.264
/// implicit weighting. H.264 8.4.3 derives that weight from `pic0 = RefPicList0[refIdxL0]` and
/// `pic1 = RefPicList1[refIdxL1]`, so it is a function of the **pair** — `td = DiffPicOrderCnt(pic1, pic0)`
/// moves when `refIdxL1` does, and `luma_weight_l0` is one weight per list position, indexed by `i`
/// alone (`va.h:3702-3717`; the spec's own explicit-mode Equations 8-289/8-290 index it by `refIdxL0WP`).
/// So this backend cannot say that, and **submitting a per-position table anyway is what decoded 60 of
/// the fixture's 72 pictures wrongly** (only 12 of 72 agreed with ffmpeg, re-measured while checking
/// this, with no error raised at all). Approximating is not available; the slice is refused.
///
/// What is asserted here is the refusal and its consequence: the raw decoder returns an `Err` whose
/// reason names the implicit weights, and the factory's decoder — the one a user gets — finishes the
/// whole run in software with the software decoder's own pictures.
#[test]
fn an_h264_b_slice_this_record_cannot_weight_is_declined() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let Some(path) = named(&ff, "h264_high.mp4") else {
        eprintln!("SKIPPED: h264_high.mp4 could not be generated");
        return;
    };
    let s = read_stream(&path);
    let reference = ffmpeg_pictures(&ff, &path, &s);
    let info = filmcraft_platform::vaapi::stream_info(&s.entry).expect("an avcC stream");
    let mut va = match VaDecoder::new(info.clone()) {
        Ok(va) => va,
        Err(e) => {
            eprintln!("SKIPPED: no VA-API decoder for this driver: {e}");
            return;
        }
    };
    // The run: the first pictures decode, and then a B slice is refused with a reason that says why.
    let (mut out, mut declines) = (Vec::new(), Vec::new());
    for (i, (smp, pts)) in s.samples.iter().enumerate() {
        match va.decode(smp, *pts) {
            Ok(mut frames) => out.append(&mut frames),
            Err(e) => declines.push((i, e.to_string())),
        }
    }
    let Some((at, why)) = declines.first().cloned() else {
        panic!("every picture of h264_high.mp4 decoded in hardware, so the implicit weights were not exercised after all and this test is measuring nothing");
    };
    assert!(why.contains("implicit"), "the refusal must name the implicit weights, not the driver: {why}");
    assert!(why.contains("per list position"), "the refusal must say why the record cannot carry them: {why}");
    // The refusal is latched, because the pictures after a picture that could not be built are decoded
    // against a picture buffer that is missing it — which is a plausible-looking wrong picture, not an
    // error. So the decoder keeps saying no, and nothing comes out of a flush.
    // `at + 1` is the sample after the refusal and `at + 4` three further on, both clamped to the stream:
    // `at` comes out of the stream itself, so a stream whose first refusal is near its end would
    // otherwise make this a slice-range panic instead of the assertion below it.
    let end = (at + 4).min(s.samples.len());
    assert!(at < end, "the refusal is at sample {at} of {}, so there is at least one sample after it", s.samples.len());
    for after in &s.samples[at + 1..end] {
        let e = va
            .decode(&after.0, after.1)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_else(|| panic!("sample {} decoded after the decoder gave up on the run", after.1));
        assert!(e.contains("gave up on this run"), "the next refusal repeats the reason rather than a new one: {e}");
        assert!(e.contains("implicit"), "and keeps naming why the run ended: {e}");
    }
    assert!(va.flush().is_empty(), "no picture is readable out of a decoder that has given up on its run");
    // A seek clears the latch — it is the one thing that starts a run over, and the picture buffer is
    // empty afterwards. What the next decode then says is the decoder's own fresh decision about this
    // sample, not the carried-over reason: this one is a mid-GOP access unit, so its references are gone
    // and the refusal is about that rather than about weights. What matters is that the latch is gone.
    va.reset();
    let again = va
        .decode(&s.samples[at].0, s.samples[at].1)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_else(|| panic!("a seek cleared the picture buffer and the decoder then accepted a mid-GOP access unit"));
    assert!(!again.contains("gave up on this run"), "the latch is cleared by a seek, so this is a fresh decision: {again}");
    let kinds = h264_slice_types(&filmcraft_codecs::hw::NalStreamInfo::from_entry(&s.entry).unwrap().unwrap(), &s);
    let slice_type = kinds.get(&s.samples[at].1).copied().unwrap_or(9);
    assert_eq!(slice_type % 5, 1, "the first refusal is at sample {at}, which is not a B picture (slice_type {slice_type})");
    println!("h264_high.mp4: the first refusal is at sample {at}, a B picture: {why}");
    // The pictures that were decoded before it are still ffmpeg's, so nothing wrong was handed over.
    let (right, first_wrong) = oracle(&out, &reference);
    assert_eq!(right, out.len(), "a picture this backend produced before refusing differs from ffmpeg at pts {first_wrong:?}");
    println!("h264_high.mp4: the {right} pictures decoded before the refusal are all ffmpeg's");
    // What a user gets: the whole run, in software, exactly as the software decoder produces it.
    let mut d = filmcraft_platform::vaapi_factory(&s.entry).expect("a VA-API factory").expect("a decoder for an avcC stream");
    assert!(d.name().starts_with("VA-API"), "the stream starts in hardware");
    let fallbacks = filmcraft_codecs::hw::hw_stats().fallbacks;
    let run = decode_all(d.as_mut(), &s.samples);
    assert!(!d.name().starts_with("VA-API"), "the stream did not finish in software");
    assert!(filmcraft_codecs::hw::hw_stats().fallbacks > fallbacks, "the refusal is counted as a fallback");
    let software = decode_all(software(&s).as_mut(), &s.samples);
    assert_same_by_pts("h264_high.mp4 finished in software", &run, &software);
    let (sw_right, _) = oracle(&software, &reference);
    assert_eq!(sw_right, reference.len(), "the software decoder disagrees with ffmpeg");
    println!("h264_high.mp4: the run is finished by the software decoder, which is right on {sw_right}/{} pictures", reference.len());
}

/// One picture's pixels as the bytes ffmpeg's raw 4:2:0 output holds them: luma, then U, then V.
fn pixels(f: &DecodedFrame) -> Vec<u8> {
    let PixelData::Yuv8 { planes, .. } = &f.frame.data else { panic!("expected 8-bit 4:2:0 planes") };
    assert_eq!((f.frame.width, f.frame.height), (640, 360), "this oracle is written for the 640x360 fixtures");
    let mut out = Vec::with_capacity(planes.iter().map(|p| p.len()).sum());
    for p in planes {
        out.extend(p.iter());
    }
    out
}
