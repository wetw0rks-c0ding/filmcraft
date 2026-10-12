//! VA-API decoding against our software decoders (Linux): every picture of the fixtures this backend
//! can describe must be the software decoder's — bit-exact planes, colour, pixel aspect, draft flag and
//! picture — also after `reset` + reseek and a mid-stream `flush`; a forced mid-stream failure must
//! continue in software with the software decoder's exact output; damaged samples and parameter sets
//! must give errors or fall back, never crash or hang; a VP9 stream this backend can decode must **not**
//! be declined while it is still a description and one it cannot must be, with a reason naming the
//! chroma format or the reserved profile; an AV1 stream must still be declined, with the reason that it
//! has no reachable per-frame header; and an H.264 stream whose B slices are implicitly weighted must
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
//! weight **per reference index per list** (`va.h:3702-3723`, inside `VASliceParameterBufferH264` —
//! there is no `va_dec_h264.h` in libva; `/usr/include/va/` has `va_enc_h264.h` and `va_fei_h264.h`
//! but the decode records are all in `va.h`), and a pair-dependent function
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
///
/// **VP9 is `EveryPicture` in both depths.** `vp9_p0_parity.mp4` (profile 0, 8-bit),
/// `vp9_p2_parity.mp4` (profile 2, 10-bit) and `vp9_p0_segmented.mp4` (`SEG_LVL_ALT_Q` on) all get the
/// whole run — every picture, a `reset` + reseek to each later sync sample, and a full pass again after
/// all the resets — which is the standard every other codec here already held and VP9 did not. They are
/// listed in [`FIXTURES`] rather than built by a test body so `cargo xtask fixtures platform`
/// regenerates them; the profile-2 entry is what turns "VP9 profiles 0 and 2, 8- and 10-bit" in
/// `README.md` / `ROADMAP.md` / `docs/architecture.md` / `docs/adr/0001-platform-ffi.md` from a claim
/// into something a committed test checks, and
/// [`a_vp9_fixture_that_really_uses_segmentation`] is what stops the third of them from quietly
/// becoming a fourth copy of the first.
///
/// **AV1 is `EveryPicture` in all five of its fixtures**, for the same reason and one more: AV1 has
/// **no `refresh_frame_flags` in its decode record** (`grep -in refresh /usr/include/va/va_dec_av1.h`
/// returns lines 421 and 436, both inside comments), so the reference state a picture is built against
/// is entirely this backend's, and a picture-for-picture comparison is the only thing that says it is
/// the stream's. The five entries exist because five different arms need five different encodes — see
/// the note on [`FIXTURES`] in `tests/common/mod.rs` for what each one reaches that the others do not.
/// `streams()` assigns `EveryPicture` to every non-`h264_high.mp4` entry, so each is wired by being
/// named here at all.
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

/// Pre-generate every fixture this file's tests use (`cargo xtask fixtures platform`).
///
/// **This test is why the fixture table is a table.** Before it existed, `cargo xtask fixtures`
/// rejected the argument `platform` outright (`no fixture generator for ["platform"]`), so nothing in
/// this crate's matrix — including every fixture the VP9 bit-exactness claims rest on — could be
/// regenerated by the one command the repository documents for exactly that. A fresh clone got
/// whichever fixtures happened to be cached and every other machine got a skip.
///
/// It walks [`FIXTURES`] and **the two fixtures derived from it** ([`unweighted`],
/// [`multi_segment_hevc`]), because those are named in assertions by name and a suite that dropped
/// them would be green with less coverage. The VP9 superframe fixture is *not* here: it is assembled
/// from a table entry's own encoded frames inside the test that uses it, which is stated at
/// [`vp9_superframe_fixture`].
#[test]
#[ignore]
fn generate_fixtures() {
    let Some(ff) = filmcraft_testkit::ffmpeg() else {
        eprintln!("SKIPPED: ffmpeg not found");
        return;
    };
    let dir = dir();
    for (name, args) in FIXTURES {
        let out = dir.join(name);
        filmcraft_testkit::fixtures::generate_and_report(&format!("platform/{name}"), &[out], || fixture(&ff, name, args));
    }
    for (what, made) in [("h264_high_unweighted.mp4", unweighted(&ff)), ("hevc_multi_segment.mp4", multi_segment_hevc(&ff))] {
        let out = dir.join(what);
        filmcraft_testkit::fixtures::generate_and_report(&format!("platform/{what}"), &[out], || made.map(|p| p.exists()));
    }
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
///
/// Returns how many pictures were compared, or `None` when this machine has no hardware decoder for
/// the stream — **so the caller can print and count it.** Every claim the documents make about this
/// backend is "N/N pictures on fixture F", and until this returned a number the only way to get one
/// was to add a print by hand: the numbers in `docs/vaapi-progress.md` came from a **gitignored**
/// report, which is precisely how a claim ends up in five documents with nothing to regenerate it.
fn parity(name: &str, s: &Stream) -> Option<usize> {
    let mut hw = hardware(s)?;
    let mut sw = software(s);
    let a = decode_all(hw.as_mut(), &s.samples);
    let b = decode_all(sw.as_mut(), &s.samples);
    assert!(a.len() >= 40, "{name}: {} pictures", a.len());
    assert_eq!(a.len(), b.len(), "{name}: picture counts");
    assert_same_by_pts(name, &a, &b);
    assert!(hw.name().starts_with("VA-API"), "{name}: stayed in hardware ({})", hw.name());
    // Kept before the loop below shadows `a`, so the number this returns is the whole run's.
    let pictures = a.len();

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
    Some(pictures)
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
            println!("{name}: {:?} — not a bit-exactness fixture", tier);
            continue;
        }
        compared += 1;
        // **One line per fixture, with the tier named and the picture count**, so the run's own
        // output is the evidence `docs/vaapi-progress.md` §1 quotes. `None` means this machine has no
        // hardware decoder for the stream — printed as such rather than passing silently, because
        // "every VP9 assertion passed" and "every VP9 assertion was skipped" read the same in a log.
        match parity(name, s) {
            Some(n) => println!("{name}: EveryPicture — {n}/{n} pictures bit for bit"),
            None => println!("{name}: EveryPicture — SKIPPED, this machine has no hardware decoder for it"),
        }
    }
    // The three fixtures this backend must decode: HEVC 8-bit, HEVC 10-bit and the H.264 control.
    // Anything less and this is no longer a parity test — in particular losing the control fixture
    // would leave the H.264 half of the claim unasserted, which is why it is counted here by name.
    assert!(compared >= 3, "only {compared} fixtures were compared bit for bit, which is not a parity test");
    assert!(
        streams.iter().any(|(n, _, t)| n == "h264_high_unweighted.mp4" && *t == Tier::EveryPicture),
        "the H.264 control fixture was not compared bit for bit"
    );
    // **VP9, by name, at both depths and on the segmented fixture.** These three are what the
    // repository's documents claim ("VP9 profiles 0 and 2 … 8- and 10-bit" in `README.md`,
    // `ROADMAP.md`, `docs/architecture.md`, `docs/adr/0001-platform-ffi.md`), so they are counted
    // rather than inferred from `compared`. Without this the claim is checkable only by reading the
    // fixture table, and dropping one VP9 entry from [`FIXTURES`] would leave every document's
    // sentence exactly as true and the proof one fixture smaller — which is how five documents came
    // to assert a 10-bit decode no committed test could check.
    for name in ["vp9_p0_parity.mp4", "vp9_p2_parity.mp4", "vp9_p0_segmented.mp4"] {
        assert!(
            streams.iter().any(|(n, _, t)| n == name && *t == Tier::EveryPicture),
            "{name} was not compared bit for bit, so the VP9 claim the documents make is unproven again"
        );
    }
    // **AV1, by name, at both depths and on the three fixtures that exist for one arm each.** These
    // five are what the repository's documents claim ("AV1 Main (profile 0) at 4:2:0, 8- and 10-bit"
    // in `README.md` / `ROADMAP.md` / `docs/architecture.md` / `docs/adr/0001-platform-ffi.md`), so
    // they are counted rather than inferred from `compared`. Each entry earns its place by reaching
    // an arm the others do not — the tiles and the `show_existing_frame` and the four-frame temporal
    // unit (`av1_tiled.mp4`), the `PRIMARY_REF_NONE` inherited path and CDEF-off
    // (`av1_intra_only.mp4`), the restoration-type encoding (`av1_loop_restoration.mp4`) — and a run
    // that silently dropped any of them would leave the documents' sentence exactly as true and the
    // proof smaller, which is how five documents came to assert a 10-bit decode no committed test
    // could check.
    for name in ["av1_p0_8bit.mp4", "av1_p0_10bit.mp4", "av1_tiled.mp4", "av1_intra_only.mp4", "av1_loop_restoration.mp4"] {
        assert!(
            streams.iter().any(|(n, _, t)| n == name && *t == Tier::EveryPicture),
            "{name} was not compared bit for bit, so the AV1 claim the documents make is unproven again"
        );
    }
}

/// A picture comes back from the `decode` call that produced it, so the run is in decode order
/// where the software decoder's is in presentation order.
///
/// Pinned so that the difference is a stated fact about this backend rather than something a reader
/// has to infer from the parity tests matching by presentation time — and so that a backend which
/// starts reordering (which would be an improvement) says so here instead of silently changing what
/// everything downstream sees.
///
/// **Which fixtures reorder is a fact about the container, read here rather than assumed, and the
/// difference is asserted to hold exactly when it does.** The older version of this test asserted
/// `hardware_pts != software_pts` for every fixture, on the reasoning that "this fixture reorders, so
/// the two orders really do differ" — true of H.264 and HEVC, whose containers interleave B pictures,
/// and **false of VP9, which is not a limitation of this fixture but of the codec**: VP9 has no
/// B-frames, so it hides an alt-ref frame and shows it again with `show_existing_frame` rather than
/// emitting pictures out of order, and its access units are already in presentation order. Asserting
/// the difference unconditionally would therefore have meant either excluding VP9 from a standard
/// every other codec holds, or quietly dropping the assertion to let it in.
///
/// So the property is stated once and checked in both directions: the hardware's order **is** the
/// container's decode order, the software decoder's **is** presentation order, and the two differ
/// **exactly when the container reorders**. A backend that started reordering would fail
/// `hardware_pts == container_pts`; a fixture that stopped reordering would fail the `assert_eq!` the
/// other way. [`reordering_fixtures`] is counted so the distinction cannot stop being exercised
/// altogether — the assertion is still reached for a codec that reorders, and that is asserted here.
#[test]
fn a_picture_is_emitted_as_it_is_decoded() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let mut compared = 0usize;
    let mut reordering = 0usize;
    for (name, s, tier) in streams_with_control(&ff) {
        if tier != Tier::EveryPicture {
            continue;
        }
        let Some(mut hw) = hardware(&s) else { continue };
        compared += 1;
        // What the container itself says, read from the sample table: decode order, and whether that
        // is presentation order. This is the fixture's own property and the test states it rather
        // than carrying an assumption about which codecs have it.
        let container_order: Vec<i64> = s.samples.iter().map(|(_, p)| *p).collect();
        let mut presentation_order = container_order.clone();
        presentation_order.sort_unstable();
        let reorders = container_order != presentation_order;
        reordering += usize::from(reorders);

        let hardware_pts: Vec<i64> = s.samples.iter().flat_map(|(b, p)| hw.decode(b, *p).unwrap()).map(|f| f.pts).collect();
        let software_pts: Vec<i64> = decode_all(software(&s).as_mut(), &s.samples).iter().map(|f| f.pts).collect();
        assert_eq!(hardware_pts.len(), software_pts.len(), "{name}: picture counts");
        assert_eq!(hardware_pts, container_order, "{name}: one picture per access unit, in the container's decode order");
        assert_eq!(software_pts, presentation_order, "{name}: the software decoder emits them in presentation order");
        // The whole of the distinction, in both directions, rather than an assumption about it.
        assert_eq!(hardware_pts != software_pts, reorders, "{name}: the two orders differ exactly when the container reorders (this one does: {reorders})");
        if reorders {
            println!(
                "{name}: the container reorders, so decode order ({:?}…) and presentation order really do differ",
                &hardware_pts[..hardware_pts.len().min(4)]
            );
        } else {
            println!("{name}: no reordering — {} has no B-frames, so its access units are already in presentation order", s.entry.codec.name());
        }
    }
    assert!(compared > 0, "no fixture reached the decoder-order check at all");
    // **The assertion above is only reached for a stream that reorders**, so the count is asserted
    // rather than left: if every fixture here stopped reordering, `hardware_pts != software_pts`
    // would never be evaluated and the test would still be green while proving nothing about order.
    assert!(reordering >= 3, "only {reordering} of {compared} fixtures reorder, so the decode-order / presentation-order distinction is barely exercised");
    // **The fixture list is walked a second time for its names**, so a VP9 entry that silently stops
    // generating leaves the no-reordering arm unasserted rather than passing it vacuously.
    // And a codec with no reordering is genuinely in the list rather than skipped around.
    let names: Vec<String> = streams_with_control(&ff).into_iter().map(|(n, _, _)| n).collect();
    assert!(names.iter().any(|n| n.starts_with("vp9_")), "no VP9 fixture is in the list, so the no-reordering arm is unasserted: {names:?}");
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
    // The keys are (presentation time, index among the pictures at it) — see `by_pts`, which says why
    // the index is part of the key. Collecting the pairs keeps a stream that shows two pictures under
    // one timestamp comparable here too, which is the whole point of the key.
    let pts = |out: &[DecodedFrame]| -> Vec<(i64, usize)> { by_pts(out).keys().copied().collect() };
    assert_eq!(pts(a), pts(b), "{what}: pictures, matched by presentation time");
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

/// The superframe index VP9 Annex B defines, built over `frames`: the frames' bytes, then a marker
/// byte, then each frame's length as a little-endian `u32`, then the marker again.
///
/// **This is a real superframe, assembled rather than encoded, and the reason is measured.** Every
/// combination of `libvpx-vp9` arguments available here — `-auto-alt-ref 1` with
/// `-lag-in-frames` 16 / 19 / 25, `-frame-parallel 0`, `-error-resilient 0`, `-deadline` good / best /
/// realtime, `-cpu-used` 0 through 8, at one and at two GOPs — was swept for a stream carrying one, and
/// **none produced a single superframe or a single hidden frame** (16 combinations, 0 and 0). libvpx
/// packs alt-ref frames into superframes only in configurations this build does not take, so asking
/// ffmpeg harder does not reach the case.
///
/// The marker is **not** an encoder behaviour, though: it is the index the *format* defines, and the
/// codec's own parser reads it (`crates/vp9/src/header.rs:602`, which is what the decode loop now
/// calls). So the fixture is built here over **real encoded frames** — each member is a frame libvpx
/// actually produced, not a synthetic byte string — and the test below asserts the result really does
/// split, which is what stops this from being a fixture that would pass vacuously.
fn superframe_index(frames: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for f in frames {
        out.extend_from_slice(f);
    }
    // `0xc0` in the top three bits is the marker; bits 3..4 hold the **size-field width minus one**
    // and the low three hold the frame count minus one (`crates/vp9/src/header.rs:604-606`). The
    // lengths are written as `u32`, so the width must say 4: `3 << 3`. Getting this wrong is a
    // silent no-op rather than an error — the reader checks that the byte where it expects the index
    // to start *is* the marker, finds something else, and returns the whole sample as one frame, so a
    // fixture built with a mismatched width looks like a stream that simply has no superframes. That
    // is why `a_vp9_superframe_fixture_really_carries_superframes` asserts the split rather than the
    // bytes, and why it runs before anything relies on this fixture.
    let marker = 0xc0u8 | (3 << 3) | (frames.len() as u8 - 1);
    out.push(marker);
    for f in frames {
        out.extend_from_slice(&(f.len() as u32).to_le_bytes());
    }
    out.push(marker);
    out
}

/// A VP9 fixture whose samples **carry real superframes**, or `None` without ffmpeg.
///
/// Built by re-packing a normally encoded fixture's own frames: two consecutive inter frames become
/// one sample, with the index the format defines. The frames are libvpx output and the index is the
/// specification's, so the container sample is a superframe in exactly the sense that matters to a
/// decoder — `split_superframe` returns the members, and the members decode.
///
/// **Why the frames are re-packed rather than re-encoded:** no `libvpx-vp9` invocation available here
/// emits a superframe at all (see [`superframe_index`]), so an encoder argument cannot produce this
/// fixture, and a fixture that quietly had none would make every assertion below pass without testing
/// anything. That is why [`a_vp9_superframe_fixture_really_carries_superframes`] exists and why it runs
/// before anything relies on this fixture.
fn vp9_superframe_fixture(ff: &Path) -> Option<Stream> {
    // Ordinary profile 0 at 8 bits: the profile this backend decodes.
    let path = fixture(
        ff,
        "vp9_sf_members.mp4",
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x240:r=24:d=2,noise=alls=12:allf=t",
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "0",
            "-crf",
            "32",
            "-deadline",
            "good",
            "-cpu-used",
            "5",
            "-pix_fmt",
            "yuv420p",
        ],
    )?;
    let plain = read_stream(&path);
    // Two consecutive inter frames, taken from past the key frame so each predicts from pictures the
    // run has already produced rather than from a reference that is not in the stream.
    let (Some(first), Some(second)) = (plain.samples.get(3).cloned(), plain.samples.get(4).cloned()) else {
        return None;
    };
    let packed = superframe_index(&[first.0.clone(), second.0]);
    // **The packed superframe REPLACES both samples rather than being added after them.** A VP9 frame
    // is stateful — it predicts from the references the frames before it left — so decoding sample 3
    // on its own and then again inside the superframe would decode the same frame twice and desync
    // every reference after it. A container that packs two frames into one sample has *one* sample
    // there, and that is what this builds.
    //
    // One presentation time for it, the *first* frame's: that is what a container that packs them
    // records, and it is what our own software decoder attributes to the pictures a superframe shows
    // (one chunk, one timestamp, however many frames it holds).
    let mut samples: Vec<(Vec<u8>, i64)> = plain.samples[..3].to_vec();
    samples.push((packed, first.1));
    samples.extend(plain.samples[5..].iter().cloned());
    // `sync` is a parallel array; the key frame at index 0 is the only sync sample here.
    let mut sync = vec![false; samples.len()];
    if let Some(first_sync) = sync.first_mut() {
        *first_sync = true;
    }
    Some(Stream { entry: plain.entry, samples, sync })
}

/// **The fixture really carries superframes.** Every other claim about superframe handling rests on
/// this one, and it is stated as a separate test for the reason the branch has been bitten by three
/// times: a fixture that silently contains none makes every assertion downstream pass without proving
/// anything.
///
/// It asserts the *split* rather than a byte pattern, because the split is what the decode loop
/// depends on, and it asserts the **per-frame** length is what a record must be built from.
#[test]
fn a_vp9_superframe_fixture_really_carries_superframes() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let Some(s) = vp9_superframe_fixture(&ff) else {
        eprintln!("SKIPPED: ffmpeg cannot encode the libvpx-vp9 members this fixture is built from");
        return;
    };
    // At least one sample splits into more than one frame…
    let split: Vec<usize> = (0..s.samples.len()).map(|i| filmcraft_vp9::header::split_superframe(&s.samples[i].0).len()).collect();
    let superframes = split.iter().filter(|&&n| n > 1).count();
    assert!(superframes > 0, "the fixture carries no superframe at all, so every assertion about them is vacuous: {split:?}");
    // …and the members are the frames that went in, byte for byte.
    let i = split.iter().position(|&n| n > 1).expect("asserted above");
    let parts = filmcraft_vp9::header::split_superframe(&s.samples[i].0);
    assert!(parts.iter().all(|p| !p.is_empty()), "every member of the index is a frame: {parts:?}");
    // **The claim the record depends on:** each member's own length differs from the sample's, and the
    // compressed-header extent is measured from the member. A frame whose uncompressed header is `n`
    // bytes gives `member.len() - n`, and using the sample's length instead is the defect this fixture
    // exists to pin — so the two numbers must be different here, or the test proves nothing.
    let refs: [Option<filmcraft_vp9::header::RefInfo>; 8] = std::array::from_fn(|_| Some(filmcraft_vp9::header::RefInfo { width: 320, height: 240 }));
    for (j, part) in parts.iter().enumerate() {
        let mut st = filmcraft_vp9::header::HeaderState::default();
        let h = filmcraft_vp9::header::parse_uncompressed(part, &mut st, &refs)
            .unwrap_or_else(|e| panic!("member {j} of the superframe at sample {i} is not a frame this parser reads: {e}"));
        let per_frame = part.len() - h.uncompressed_size;
        let whole_sample = s.samples[i].0.len() - h.uncompressed_size;
        assert_ne!(per_frame, whole_sample, "member {j}: the per-frame and whole-sample extents agree, so this fixture cannot catch the defect it exists for");
        assert!(per_frame <= u16::MAX as usize, "member {j}: its own first_partition_size ({per_frame}) fits the record's field");
    }
    // And the whole-sample reading — what the code did before the fix — is larger, which is the wrong
    // direction: it points into the next frame's bytes.
    let mut st = filmcraft_vp9::header::HeaderState::default();
    let h = filmcraft_vp9::header::parse_uncompressed(parts[0], &mut st, &refs).expect("member 0");
    assert!(
        s.samples[i].0.len() - h.uncompressed_size > parts[0].len() - h.uncompressed_size,
        "the pre-fix whole-sample value is the larger one, which is why it is wrong: a driver reading a compressed header \
         that long reads past the end of its own frame"
    );
}

/// **A superframe is one sample holding several frames, and both decoders must treat it that way.**
///
/// This is the end-to-end statement of the fix, made of three claims a wrong implementation cannot
/// satisfy at once:
///
/// 1. **Every frame in it is decoded.** The hidden alt-ref frame inside a superframe is the reference
///    the shown frame after it predicts from. If it were skipped — because it is hidden, or because
///    only the first frame of the sample was parsed — the shown frame's references would resolve
///    against a buffer that does not hold them.
/// 2. **Exactly one picture comes out per shown frame**, never one per *sample* member. A hidden frame
///    is decoded and produces no picture, which is the same rule a single-frame hidden picture follows
///    and what the software decoder does; handing back one picture per frame would show the alt-ref
///    frame, which the stream explicitly says is not for display.
/// 3. **The pictures are the software decoder's, bit for bit** — for both decoders, so the split did
///    not merely survive but produced the same pictures our own VP9 decoder does from the same samples.
///
/// The run is compared whole, not only at the superframe, because a wrong reference map shows up in the
/// pictures *after* it.
///
/// **The software half is the one that runs everywhere** and it is what makes this test meaningful on a
/// machine whose driver cannot decode VP9 at all. The hardware half is skipped, loudly, only when the
/// driver refuses the *stream* — no libva, no render node, no VP9 profile 0 — which is a fact about
/// the machine. A driver that takes the stream and then fails a picture is a failure here.
#[test]
fn a_vp9_superframe_decodes_every_frame_and_shows_only_what_the_stream_shows() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let Some(s) = vp9_superframe_fixture(&ff) else {
        eprintln!("SKIPPED: ffmpeg cannot encode the libvpx-vp9 members this fixture is built from");
        return;
    };
    let i = (0..s.samples.len())
        .find(|&i| filmcraft_vp9::header::split_superframe(&s.samples[i].0).len() > 1)
        .expect("a_vp9_superframe_fixture_really_carries_superframes guarantees this");

    // (1)+(2) for the software decoder, which needs no driver: the run decodes to completion rather
    // than refusing on the second frame's references, and the number of pictures is **one per shown
    // frame** — which for *this* fixture is two, because libvpx packed two inter frames and both are
    // shown. Asserted as a count against the frames the index actually names, so the claim is about
    // the rule and not about a number this particular stream happens to have.
    //
    // (The hidden-alt-ref case, where the count is one *fewer* than the frame count, is what the unit
    // test `a_hidden_frame_produces_no_picture_but_still_moves_its_slots` covers with a frame the
    // parser reads and this encoder cannot be made to emit.)
    let members = filmcraft_vp9::header::split_superframe(&s.samples[i].0).len();
    let shown = {
        let refs: [Option<filmcraft_vp9::header::RefInfo>; 8] = std::array::from_fn(|_| Some(filmcraft_vp9::header::RefInfo { width: 320, height: 240 }));
        filmcraft_vp9::header::split_superframe(&s.samples[i].0)
            .iter()
            .filter(|f| {
                let mut st = filmcraft_vp9::header::HeaderState::default();
                filmcraft_vp9::header::parse_uncompressed(f, &mut st, &refs).is_ok_and(|h| h.show_frame)
            })
            .count()
    };
    let reference = decode_all(software(&s).as_mut(), &s.samples);
    assert!(!reference.is_empty(), "the software decoder produced no pictures from the superframe fixture");
    assert_eq!(
        reference.iter().filter(|f| f.pts == s.samples[i].1).count(),
        shown,
        "a superframe of {members} frame(s) yields one picture per **shown** frame ({shown}), not one per sample"
    );

    let Some(info) = filmcraft_platform::vaapi::stream_info(&s.entry) else {
        eprintln!("SKIPPED: not a vpcC stream");
        return;
    };
    let mut hw = match filmcraft_platform::vaapi::VaDecoder::new(info) {
        Ok(d) => d,
        Err(why) => {
            eprintln!("SKIPPED: this driver does not take a VP9 profile 0 stream: {why}");
            return;
        }
    };
    // **The hardware half is a real assertion now.** It used to report a decode failure as a skip,
    // on the grounds that "this driver cannot decode a VP9 picture at all" was a separate open
    // problem from superframe handling. That problem was this backend's, not the machine's: the
    // submission carried a picture-parameter record and slice data but **no
    // `VASliceParameterBufferVP9`** — the one buffer `va_dec_vp9.h:279` declares and
    // `va_dec_vp9.h:265-268` says is sent once per frame — so `vaEndPicture` answered
    // `VA_STATUS_ERROR_INVALID_PARAMETER` after every other call had reported success. With the
    // record submitted the driver decodes, so a failure here is a failure of this backend and is
    // reported as one.
    let mut out = Vec::new();
    for (sample, pts) in &s.samples {
        match hw.decode(sample, *pts) {
            Ok(frames) => out.extend(frames),
            Err(e) => panic!("the hardware decoder failed on a VP9 stream this driver accepted: {e}"),
        }
    }
    out.extend(hw.flush());
    assert_eq!(out.iter().filter(|f| f.pts == s.samples[i].1).count(), shown, "the hardware decoder yields one picture per shown frame, like the software one");
    assert_eq!(out.len(), reference.len(), "the hardware and software decoders return the same number of pictures");
    assert_same_by_pts("a VP9 superframe", &out, &reference);
    assert!(hw.name().starts_with("VA-API"), "stayed in hardware ({})", hw.name());
}

/// **A VP9 sample entry is no longer declined for being a frame codec**, and the two halves of that
/// are asserted separately because they are different claims.
///
/// - **Profile 0 at 8 bits is planned.** `plan_frame` returns a `Plan` for it, so the only reasons
///   `VaDecoder::new` can still decline are the machine's: no libva, no render node, or a driver that
///   does not offer VP9 8-bit 4:2:0. The test asserts that whatever happens is one of those and
///   **not** a "picture parameters" decline — because that sentence used to be the answer here and
///   would now be a false claim about this backend. A machine that *does* take it is checked for
///   real against ffmpeg by Task 5's parity fixture; what this test owns is that the decline is gone.
/// - **Profiles 1, 3 and 4 are still declined**, and as *that*: a 4:2:2 / 4:4:4 / reserved stream is
///   refused for its chroma format or its reserved profile number, before any driver is asked. This
///   is the half that would otherwise be lost by editing the old test's expectations.
///
/// **AV1 keeps its own decline**, in its own test below, because its reason is a different fact: its
/// per-frame header is not exposed, so there is no record to build.
#[test]
fn a_vp9_stream_this_backend_can_decode_is_not_declined_and_one_it_cannot_is_declined_as_such() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let src = "testsrc2=s=640x360:r=24:d=1";
    // Profile 0 at 8 bits: 4:2:0, which is the whole of what this backend decodes.
    let Some(path) = fixture(
        &ff,
        "vp9_p0.mp4",
        &["-f", "lavfi", "-i", src, "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "40", "-deadline", "realtime", "-cpu-used", "8", "-pix_fmt", "yuv420p"],
    ) else {
        eprintln!("SKIPPED: ffmpeg cannot encode libvpx-vp9");
        return;
    };
    let s = read_stream(&path);
    let info = filmcraft_platform::vaapi::stream_info(&s.entry).unwrap_or_else(|| panic!("vp9_p0.mp4: not recognised as a vpcC stream"));
    match VaDecoder::new(info) {
        // This machine has a VA-API driver that takes VP9: the factory hands it over, and the
        // fixture is then this backend's problem (Task 5 compares it with ffmpeg).
        Ok(_) => assert!(filmcraft_platform::vaapi_factory(&s.entry).is_some(), "the decoder was built, so the factory hands it over too"),
        // It was declined — which is fine on a machine with no VP9 decoder, as long as the reason is
        // the machine and not the codec.
        Err(why) => {
            assert!(!why.contains("picture parameters"), "VP9 profile 0 is planned, so a picture-parameters decline would be false: {why}");
            assert!(
                !why.contains("filmcraft-vp9") || why.contains("does not offer"),
                "the old 'the parser is unreachable' reason is no longer true of VP9: {why}"
            );
            assert!(filmcraft_platform::vaapi_factory(&s.entry).is_none(), "and the factory declines it, so a user gets FilmCraft's own decoder");
        }
    }
    // …and FilmCraft's own decoder can play it whatever this machine could, which is the point of
    // declining rather than failing.
    assert!(!decode_all(software(&s).as_mut(), &s.samples).is_empty(), "vp9_p0.mp4: the software decoder produced no pictures");

    // The profiles this backend has no surface for, each still declined *as itself*.
    //
    // **Two wordings are accepted for the same fact, and both name the chroma format.** A `vpcC` that
    // says 4:2:2 or 4:4:4 is caught by `plan_frame`'s `chroma subsampling` check, which runs before
    // the profile table — so in practice that is the message a real 4:2:2 stream gets, and it is the
    // more specific one (the container states the format outright). The profile table's own `4:2:2`
    // sentence is what a stream whose *frames* disagree with its container gets. Asserting one
    // wording exactly would pin an ordering that is a choice rather than a promise, so what is
    // asserted is the fact: the decline is about the chroma format, and never about the parser.
    // The pixel formats are the codec's own, not a guess: per VP9 6.2's `color_config`, profile 1 is
    // 8-bit 4:2:2 / 4:4:0 and profile 3 is 10-bit 4:4:4, and libvpx refuses the other pairings
    // outright ("Codec bit-depth 8 not supported in profile > 1", "high bit-depth not supported in
    // profile < 2") — so a fixture generated with the wrong one is an ffmpeg error, not a decline.
    for (profile, pix_fmt) in [(1u32, "yuv422p"), (3, "yuv444p10le")] {
        let name = format!("vp9_p{profile}.mp4");
        let Some(path) = fixture(
            &ff,
            &name,
            &[
                "-f",
                "lavfi",
                "-i",
                src,
                "-c:v",
                "libvpx-vp9",
                "-profile:v",
                &profile.to_string(),
                "-b:v",
                "0",
                "-crf",
                "40",
                "-deadline",
                "realtime",
                "-cpu-used",
                "8",
                "-pix_fmt",
                pix_fmt,
            ],
        ) else {
            eprintln!("SKIPPED: ffmpeg cannot encode a profile {profile} libvpx-vp9");
            continue;
        };
        let s = read_stream(&path);
        let info = filmcraft_platform::vaapi::stream_info(&s.entry).unwrap_or_else(|| panic!("{name}: not recognised as a vpcC stream"));
        assert!(matches!(info, StreamInfo::Frame(_)), "{name}: recognised as {info:?}");
        let why_declined = VaDecoder::new(info).err().unwrap_or_else(|| panic!("{name}: this backend took a {pix_fmt} VP9 stream"));
        assert!(
            why_declined.contains("chroma subsampling") || why_declined.contains("4:2:2"),
            "{name}: refused for its chroma format, not for something else: {why_declined}"
        );
        assert!(!why_declined.contains("filmcraft-vp9"), "{name}: VP9's header is reachable, so the parser is not the reason: {why_declined}");
        assert!(filmcraft_platform::vaapi_factory(&s.entry).is_none(), "{name}: the factory handed a {pix_fmt} VP9 stream to a VA-API decoder");
    }
    // **Profile 4 is reserved by the specification**, so no encoder can be asked for one and this
    // half of the claim is a unit test rather than a fixture: `vp9_decline_reason` is what
    // `plan_frame` and the record builder both call, and
    // `vaapi::stream::tests::the_frame_codecs_are_planned_or_declined_for_their_own_reason` walks it.
}

/// **The segmented fixture really uses segmentation, verified with this repo's own parser.**
///
/// This is the instrument the branch should have had before finding 3, and it is a *separate* test for
/// the reason this file keeps repeating: a fixture that silently contains none of the thing it exists
/// to exercise makes every assertion downstream pass without proving anything. A libvpx build without
/// the AQ feature, or a future one that renumbers its segmentation features, would leave
/// `vp9_p0_segmented.mp4` a fourth copy of `vp9_p0_parity.mp4` and this suite would still be green on
/// 96 bit-exact pictures — every one of them decoded down the `count == 1` path.
///
/// **So the parse is this project's, not ffmpeg's or a side-data dump**: `filmcraft_vp9::header`
/// is what builds the record, and a disagreement between what this test reads and what the driver is
/// handed would be a real bug rather than a difference of tools. The numbers below are what this
/// machine's libvpx produces for `-aq-mode 2` on this source, and the deltas are asserted individually
/// rather than as a shape, because *distinct* deltas per segment is the property that makes the
/// eight-entry arm worth testing at all: equal deltas would let a record that wrote one segment's
/// quantizer into all eight pass.
///
/// `SEG_LVL_ALT_Q` is the one feature this build turns on. It is the one that matters most for this
/// task — the quantizer is what has nowhere else to go in the record
/// (`VADecPictureParameterBufferVP9` has no `base_q_idx`, see `vaapi::picture::vp9_slice_record`) — and
/// it is what `vp9_segment_param` resolves per segment. The other three (`SEG_LVL_ALT_L`,
/// `SEG_LVL_REF_FRAME`, `SEG_LVL_SKIP`) are **not** reachable from any libvpx invocation available
/// here, which is stated rather than glossed: the `segment_flags` and per-segment `filter_level`
/// assertions in [`the_vp9_slice_record_carries_every_segment_it_should`] therefore run with those
/// features clear on both arms, and their active path is covered by the unit tests at
/// `vaapi::picture` that build the header directly.
/// Every frame of `vp9_p0_segmented.mp4` on which 6.2.11's `segmentation_update_data` rewrites
/// `FeatureData[][]`, with the `SEG_LVL_ALT_Q` deltas that frame codes, as `(sample, deltas)`.
///
/// **Where these numbers come from, and why they are a table rather than one constant.** They were
/// read off this fixture with an **independent** VP9 header parser — GStreamer's `GstVp9Parser`
/// (`gst-plugins-bad`), fed the same frames — and then this repository's parser was asserted to
/// agree on every one of them, which is what this test does. GStreamer's own numbers, on the frames
/// where `update_data == 1`, are:
///
/// ```text
/// frame  0 (key)  update_data=1  altq = -38, -25, -10, [off], +14
/// frame 10         update_data=1  altq = -66, -39, -15, [off], +14
/// frame 17         update_data=1  altq = -66, -39, -15, [off], +14   (rewrites the same values)
/// frame 24 (key)   update_data=1  altq = -38, -25, -10, [off], +14
/// frame 34, 41     update_data=1  altq = -66, -39, -15, [off], +14   (41 rewrites the same values)
/// frame 48 (key)   update_data=1  altq = -38, -25, -10, [off], +14
/// frame 58, 65     update_data=1  altq = -66, -39, -15, [off], +14   (65 rewrites the same values)
/// frame 72 (key)   update_data=1  altq = -38, -25, -10, [off], +14
/// frame 82, 88     update_data=1  altq = -66, -39, -15, [off], +14   (88 rewrites the same values)
/// ```
///
/// (every other frame: `seg_enabled = 1`, `update_data = 0`, so the values above stand). ffmpeg's
/// own VP9 decoder no longer exposes segmentation as frame side data —
/// `AV_FRAME_DATA_SEGMENTATION_PARAMS` is gone from `libavutil/frame.h` — so GStreamer is the
/// reference implementation available here, and it is a *different* implementation of 6.2.11 from
/// `filmcraft_vp9`, which is what makes it an oracle rather than a mirror. The two agree on every
/// frame.
///
/// The table below lists the eight frames on which the values **change**. Four more frames set
/// `segmentation_update_data` and re-code the values they already had (17, 41, 65, 88); libvpx's
/// adaptive-quantizer search re-decides the same quantizers, and 6.2.11 writes them again. They are
/// not in the table because a frame that rewrites the same numbers says nothing new about the
/// parse, and pinning them would be pinning noise.
///
/// The earlier version of this test pinned the key frame's row (`-38, -25, -10, 0, 14, 0, 0, 0`) and
/// asserted it on all ninety-six frames. It was wrong on the sixty frames that carry `-66`, `-39`,
/// `-15`: the value is a fact about the frame that *coded* it, not about the fixture, and 6.2.11's
/// placement of the `FeatureData` loop inside `if (segmentation_update_data)` is what makes the
/// difference visible.
const SEGMENTED_ALT_Q_UPDATES: &[(usize, [i16; 8])] = &[
    (0, [-38, -25, -10, 0, 14, 0, 0, 0]),
    (10, [-66, -39, -15, 0, 14, 0, 0, 0]),
    (24, [-38, -25, -10, 0, 14, 0, 0, 0]),
    (34, [-66, -39, -15, 0, 14, 0, 0, 0]),
    (48, [-38, -25, -10, 0, 14, 0, 0, 0]),
    (58, [-66, -39, -15, 0, 14, 0, 0, 0]),
    (72, [-38, -25, -10, 0, 14, 0, 0, 0]),
    (82, [-66, -39, -15, 0, 14, 0, 0, 0]),
];

#[test]
fn a_vp9_fixture_that_really_uses_segmentation() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let Some(path) = named(&ff, "vp9_p0_segmented.mp4") else {
        eprintln!("SKIPPED: ffmpeg cannot encode libvpx-vp9");
        return;
    };
    let s = read_stream(&path);
    let refs: [Option<filmcraft_vp9::header::RefInfo>; 8] = std::array::from_fn(|_| Some(filmcraft_vp9::header::RefInfo { width: 640, height: 360 }));
    let mut st = filmcraft_vp9::header::HeaderState::default();
    let (mut frames, mut on) = (0usize, 0usize);
    // **The deltas are pinned per *update* frame, not once for the run.** 6.2.11 puts the whole
    // `FeatureData` loop inside `if (segmentation_update_data)`, so a frame that does not update
    // leaves the array as it was and the next frame's deltas depend on the frame before it. This
    // fixture updates twelve times in ninety-six frames, at the samples listed in
    // [`SEGMENTED_ALT_Q_UPDATES`], and the two value sets are *not* interchangeable: the key frames
    // carry `-38, -25, -10, +14` and every inter-frame update carries `-66, -39, -15, +14`. A single
    // pinned constant asserted on every frame is therefore wrong on nine frames in ten — which is
    // exactly the defect class this branch keeps paying for, and it is what an earlier version of
    // this test did (it pinned the key frame's values and asserted them on all 96).
    let mut seen_updates = 0usize;
    let mut previous: Option<[i16; 8]> = None;
    for (i, (bytes, _)) in s.samples.iter().enumerate() {
        for part in filmcraft_vp9::header::split_superframe(bytes) {
            let h = filmcraft_vp9::header::parse_uncompressed(part, &mut st, &refs).unwrap_or_else(|e| panic!("sample {i} is a frame this parser reads: {e}"));
            frames += 1;
            if !st.seg.enabled {
                continue;
            }
            on += 1;
            // `SEG_LVL_ALT_Q` active on four segments, with four different deltas.
            let active: Vec<usize> = (0..8).filter(|&seg| st.seg.feature_enabled[seg][filmcraft_vp9::SEG_LVL_ALT_Q]).collect();
            assert_eq!(active, vec![0, 1, 2, 4], "sample {i}: the segments carrying SEG_LVL_ALT_Q");
            let d: [i16; 8] = std::array::from_fn(|seg| st.seg.feature_data[seg][filmcraft_vp9::SEG_LVL_ALT_Q]);
            // Only a frame that *changes* the data is pinned, because only such a frame is a fact
            // about the bitstream; every other frame carries the previous one forward by 6.2.11's
            // placement of the loop inside the `segmentation_update_data` branch.
            if previous != Some(d) {
                let (want_sample, want_deltas) = SEGMENTED_ALT_Q_UPDATES
                    .get(seen_updates)
                    .copied()
                    .unwrap_or_else(|| panic!("sample {i}: a thirteenth segmentation update at {d:?}; the table below has {seen_updates}"));
                assert_eq!(i, want_sample, "update {seen_updates} happens at sample {i}, not the pinned {want_sample}");
                assert_eq!(d, want_deltas, "sample {i}: the four segments' quantizer deltas");
                // **8.6.1 `get_qindex`, derived rather than transcribed**: the record's quantizer for
                // a segment is `Clip3(0, 255, base_q_idx + delta)` while
                // `segmentation_abs_or_delta_update == 0` (which is what 6.2.11's
                // `segmentation_abs_or_delta_update` reads as on every frame of this fixture), and
                // `base_q_idx` unchanged for a segment the feature is off on. Computing it here is
                // what makes the pinned deltas *mean* something: it is the number `SegParam[]` is
                // built from, so a sign error in the parser cannot pass by matching a constant.
                let q: Vec<i32> = (0..8)
                    .map(|seg| if active.contains(&seg) { i32::from(d[seg]) + i32::from(h.base_q_idx) } else { i32::from(h.base_q_idx) })
                    .map(|v| v.clamp(0, 255))
                    .collect();
                assert_ne!(q[0], q[1], "sample {i}: segments 0 and 1 resolve to the same qindex {q:?}");
                assert_ne!(q[0], q[2], "sample {i}: segments 0 and 2 resolve to the same qindex {q:?}");
                assert_eq!(q[3], i32::from(h.base_q_idx), "sample {i}: segment 3 carries no ALT_Q, so 8.6.1 leaves its qindex at base_q_idx");
                seen_updates += 1;
            }
            // **Distinct per active segment** — the property that makes the eight-entry arm worth
            // having. Four segments carrying one value would let a record that computed `SegParam[0]`
            // once and copied it into the other seven pass this fixture.
            let distinct: std::collections::BTreeSet<i16> = active.iter().map(|&seg| d[seg]).collect();
            assert_eq!(
                distinct.len(),
                4,
                "sample {i}: the active segments' deltas are not distinct, so this fixture cannot tell the eight entries apart: {distinct:?}"
            );
            // The other three features are off on every segment, which is the honest measurement:
            // no libvpx invocation available here sets them.
            for (name, f) in [
                ("SEG_LVL_ALT_L", filmcraft_vp9::SEG_LVL_ALT_L),
                ("SEG_LVL_REF_FRAME", filmcraft_vp9::SEG_LVL_REF_FRAME),
                ("SEG_LVL_SKIP", filmcraft_vp9::SEG_LVL_SKIP),
            ] {
                assert!(
                    (0..8).all(|seg| !st.seg.feature_enabled[seg][f]),
                    "sample {i}: {name} is active on some segment — no libvpx invocation here sets it, so this fixture's encoder changed; the assertion below would no longer describe it"
                );
            }
            previous = Some(d);
        }
    }
    assert!(frames > 0, "the fixture has no frames this parser reads");
    assert_eq!(seen_updates, SEGMENTED_ALT_Q_UPDATES.len(), "{seen_updates} of {} pinned segmentation updates were reached", SEGMENTED_ALT_Q_UPDATES.len());
    // **Every** frame, not one: a fixture whose segmentation came and went would leave the
    // `SegParam[1..7] = 0` arm asserted on some pictures and the eight-entry arm on others without
    // anything in the test saying so.
    assert_eq!(on, frames, "{on} of {frames} frames carry segmentation_enabled — this fixture must use it throughout");
    println!(
        "vp9_p0_segmented.mp4: all {frames} frames carry segmentation_enabled, SEG_LVL_ALT_Q on segments 0, 1, 2, 4, {seen_updates} data updates at samples {:?}",
        SEGMENTED_ALT_Q_UPDATES.iter().map(|(i, _)| *i).collect::<Vec<_>>()
    );

    // And the **other** arm is a real fixture too: `vp9_p0_parity.mp4` is the same encode without
    // `-aq-mode 2`, and it must *not* use segmentation — otherwise the eight-entry branch would be
    // the only one any test reaches, which is the state this finding was sent to fix.
    let Some(plain) = named(&ff, "vp9_p0_parity.mp4") else { return };
    let plain = read_stream(&plain);
    let mut st2 = filmcraft_vp9::header::HeaderState::default();
    let (mut frames2, mut on2) = (0usize, 0usize);
    for (i, (bytes, _)) in plain.samples.iter().enumerate() {
        for part in filmcraft_vp9::header::split_superframe(bytes) {
            filmcraft_vp9::header::parse_uncompressed(part, &mut st2, &refs).unwrap_or_else(|e| panic!("vp9_p0_parity.mp4 sample {i}: {e}"));
            frames2 += 1;
            on2 += usize::from(st2.seg.enabled);
        }
    }
    assert_eq!(on2, 0, "vp9_p0_parity.mp4: {on2} of {frames2} frames carry segmentation_enabled, so it is not the segmentation-off arm this suite needs");
    println!("vp9_p0_parity.mp4: none of its {frames2} frames carry segmentation_enabled — the segmentation-off arm");
}

/// **An AV1 stream this backend cannot describe is still declined before a context exists, and the
/// reason names what is wrong with *it*.**
///
/// **This test used to assert that every AV1 stream is declined, and that was true until the record
/// was built.** The decodable half is now a parity fixture instead (`av1_p0_8bit.mp4` and its four
/// siblings in `FIXTURES`, all at `Tier::EveryPicture`), exactly as the VP9 milestone moved its half
/// out; what stayed behind is the *feature* decline — a profile or a depth this backend has no surface
/// for — which is a different fact from "no record exists" and would have been lost with it.
///
/// The fixture is generated here rather than taken from the table because it is the only one whose
/// **encoder arguments differ from every parity fixture**: `libaom-av1 -pix_fmt yuv444p` is AV1
/// profile 1, 4:4:4, which `va_dec_av1.h:44` rules out of this API ("This AV1 decoding API supports
/// 8-bit/10bit 420 format only") and which `plan_frame` therefore refuses as *that profile*. A
/// profile-0 stream is not what this test is about, and adding one to the parity table to prove the
/// opposite of what this test says would be the confusing way to spend a fixture.
///
/// What it also pins, which is a claim that used to be false: **the factory hands the software
/// decoder an AV1 stream this backend cannot take, and takes one it can.** The factory arm below is
/// the user's-visible half; the second assertion is the one that would have failed before the record
/// existed, because `vaapi_factory` returned `None` for *every* AV1 stream.
#[test]
fn an_av1_stream_this_backend_cannot_describe_is_declined_as_such() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let src = "testsrc2=s=640x360:r=24:d=1";
    let Some(path) =
        fixture(&ff, "av1_p1_444.mp4", &["-f", "lavfi", "-i", src, "-c:v", "libaom-av1", "-crf", "50", "-b:v", "0", "-cpu-used", "8", "-pix_fmt", "yuv444p"])
    else {
        eprintln!("SKIPPED: ffmpeg cannot encode libaom-av1");
        return;
    };
    let s = read_stream(&path);
    // The entry is recognised as a frame-coded stream…
    let info = filmcraft_platform::vaapi::stream_info(&s.entry).unwrap_or_else(|| panic!("av1_p1_444.mp4: not recognised as a vpcC / av1C stream"));
    assert!(matches!(info, StreamInfo::Frame(_)), "recognised as {info:?}");
    // …and declined — but **as the chroma it is, and not as the profile**, because that is what it
    // actually is declined as. `plan_frame` checks subsampling before it resolves a profile
    // (`stream.rs:152-154`), and 4:4:4 is a chroma this backend has no surface for whatever the
    // profile says. **This test used to assert the profile's reason here and was wrong**, which is
    // the same shape of error the plan warns about: a test that asserts a decline reason the code
    // does not produce is a test that passes only until someone reads what it is really checking.
    // The profile table is reached by the 12-bit case below, which is 4:2:0 and so passes the chroma
    // check first.
    let why = VaDecoder::new(info).err().unwrap_or_else(|| panic!("av1_p1_444.mp4: this backend took a 4:4:4 AV1 stream"));
    assert!(why.contains("chroma subsampling"), "av1_p1_444.mp4: 4:4:4 is declined for its chroma, which is the check that fires first: {why}");
    assert!(!why.contains("filmcraft-av1"), "the record EXISTS now, so a decline must never blame the parser: {why}");
    // The factory declines it too, so a user gets FilmCraft's own decoder and never sees this.
    assert!(filmcraft_platform::vaapi_factory(&s.entry).is_none(), "av1_p1_444.mp4: the factory handed a 4:4:4 AV1 stream to a VA-API decoder");
    // …and FilmCraft's own decoder can still play it, which is the point of declining.
    assert!(!decode_all(software(&s).as_mut(), &s.samples).is_empty(), "av1_p1_444.mp4: the software decoder produced no pictures");

    // **The depth decline, which is the AV1 table's own and the one the plan's Q3 and scope rest on.**
    // 12-bit 4:2:0 is *still* profile 0 — so it clears the chroma check and reaches
    // `av1_decline(0, 12)`, which is the arm that cites `va_dec_av1.h:44`. Without this half the
    // test would cover only a check shared with H.264 and HEVC and would never touch AV1's table.
    let Some(deep) = fixture(
        &ff,
        "av1_p0_12bit.mp4",
        &["-f", "lavfi", "-i", src, "-c:v", "libaom-av1", "-crf", "50", "-b:v", "0", "-cpu-used", "8", "-pix_fmt", "yuv420p12le"],
    ) else {
        eprintln!("SKIPPED: ffmpeg cannot encode a 12-bit libaom-av1 stream, so AV1's own decline table is untested here");
        return;
    };
    let d = read_stream(&deep);
    let info = filmcraft_platform::vaapi::stream_info(&d.entry).unwrap_or_else(|| panic!("av1_p0_12bit.mp4: not recognised as an av1C stream"));
    let why = VaDecoder::new(info).err().unwrap_or_else(|| panic!("av1_p0_12bit.mp4: this backend took a 12-bit AV1 stream"));
    assert!(why.contains("12"), "av1_p0_12bit.mp4: the decline names the depth: {why}");
    assert!(why.contains("va_dec_av1.h:44"), "av1_p0_12bit.mp4: the header line is what makes the claim checkable: {why}");
    assert!(filmcraft_platform::vaapi_factory(&d.entry).is_none(), "av1_p0_12bit.mp4: the factory handed a 12-bit AV1 stream to a VA-API decoder");
    assert!(!decode_all(software(&d).as_mut(), &d.samples).is_empty(), "av1_p0_12bit.mp4: the software decoder produced no pictures");

    // And the converse, on a profile-0 stream the *table* names: the factory takes it, so a user gets
    // the hardware decoder and the parity suite is what holds it to bit-exactness. This assertion is
    // the one that fails if the record is quietly removed again.
    let Some(p0) = named(&ff, "av1_p0_8bit.mp4") else {
        eprintln!("SKIPPED: av1_p0_8bit.mp4 could not be generated");
        return;
    };
    let s0 = read_stream(&p0);
    let info0 = filmcraft_platform::vaapi::stream_info(&s0.entry).unwrap_or_else(|| panic!("av1_p0_8bit.mp4: not recognised as an av1C stream"));
    match filmcraft_platform::vaapi_factory(&s0.entry) {
        None => eprintln!("SKIPPED: no VA-API decoder on this machine for av1_p0_8bit.mp4 — the take-it half is untested here"),
        Some(Err(why)) => panic!("av1_p0_8bit.mp4: the factory declined a profile-0 8-bit AV1 stream this backend decodes: {why}"),
        Some(Ok(d)) => {
            assert!(d.name().starts_with("VA-API"), "av1_p0_8bit.mp4: taken by {}", d.name());
            let plan = filmcraft_platform::vaapi::Stream::from(info0).plan().unwrap_or_else(|e| panic!("av1_p0_8bit.mp4: planned: {e}"));
            assert_eq!(format!("{:?}", plan.profile), "Av1Main", "planned as AV1 Main");
        }
    }
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
///
/// The picture size and depth are the **stream's own**, from its configuration record, not this file's
/// former 640x360 constant: the weighted HEVC fixtures are 352x288 and one of them is 10-bit, and an
/// oracle that quietly read the wrong number of bytes per picture would compare a fixture's picture
/// 3 against its picture 4 — a run that looks like a run of wrong pictures rather than a broken
/// oracle. 10-bit samples come out of ffmpeg as two little-endian bytes each, which is what
/// [`pixels`] writes so the two sides agree byte for byte.
fn ffmpeg_pictures(ff: &Path, path: &Path, s: &Stream) -> Ffmpeg {
    let info = filmcraft_codecs::hw::NalStreamInfo::from_entry(&s.entry).expect("a NAL codec").expect("a described stream");
    let (w, h) = (info.crop.2, info.crop.3);
    assert!(w > 0 && h > 0, "{path:?}: the stream says a picture of {w}x{h}");
    let ten = info.bit_depth_luma == 10;
    let sample_bytes = if ten { 2 } else { 1 };
    let raw = std::process::Command::new(ff)
        .args(["-v", "error", "-i", path.to_str().unwrap(), "-f", "rawvideo", "-pix_fmt", if ten { "yuv420p10le" } else { "yuv420p" }, "-"])
        .output()
        .expect("ffmpeg decode");
    assert!(raw.status.success(), "ffmpeg failed: {}", String::from_utf8_lossy(&raw.stderr));
    let size = w as usize * h as usize * 3 / 2 * sample_bytes;
    assert_eq!(raw.stdout.len() % size, 0, "ffmpeg's decode is a whole number of {w}x{h} {}-bit 4:2:0 pictures", info.bit_depth_luma);
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

/// The HEVC weighted-prediction fixtures, decoded in hardware, are bit for bit **ffmpeg's** decode —
/// and the reason this is its own test is that `bit_exact_with_the_software_decoder` compares against
/// *our* decoder, so it cannot tell a record that is right from two FilmCraft decoders that are wrong
/// in the same way.
///
/// Before this, every HEVC fixture the suite used coded a `pred_weight_table` whose
/// `luma_weight_lX_flag` and `chroma_weight_lX_flag` were **all 0** — weights of 1.0 — so
/// `vaapi_slice_record`'s weight submission had only ever run with a delta of zero on hardware: the
/// `coded − 1 << luma_log2_weight_denom` recovery, the signed `delta_chroma_log2_weight_denom`, and
/// `ChromaOffsetLX` were all checked only in `picture.rs`'s unit tests, against the record's own
/// struct. A delta that was wrong by one would have decoded a subtly different picture on a user's
/// timeline with no error anywhere. `hevc_weightp.mp4` and `hevc_weightp_main10.mp4` are the fixtures
/// that close that, and this is the assertion that they do.
///
/// The same non-vacuity argument the unit tests make applies here, so it is measured rather than
/// assumed: [`hevc_weights_of`] counts, over the stream's own parsed slice headers, how many entries
/// really carry a non-default weight and a non-zero offset, and how many distinct denominators the
/// slices code between them. A fixture that stopped producing weights would fail the count rather than
/// pass quietly on pictures the weighting never touched.
///
/// The raw decoder, not the factory's hybrid, for the same reason as the H.264 oracle above: a decline
/// has to show up as a missing picture here, not as a software replay that looks like a hardware
/// success.
#[test]
fn an_hevc_weighted_fixture_this_record_can_express_is_bit_exact_against_ffmpeg() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let mut checked = 0;
    for name in ["hevc_weightp.mp4", "hevc_weightp_main10.mp4"] {
        let Some(path) = named(&ff, name) else {
            eprintln!("SKIPPED: {name} could not be generated");
            continue;
        };
        let s = read_stream(&path);
        let census = hevc_weights_of(&s);
        println!("{name}: {}", census.describe());
        // The fixture has to be one that *can* go vacuous, so the counts that say it is not are
        // asserted rather than printed. `crates/hevc` measures the same stream against ffmpeg and
        // agrees with these; see its `weightp` conformance test.
        assert!(census.tables >= 20, "{name}: {} slice segments code a prediction weight table, which is too few to be the weight path", census.tables);
        assert!(census.non_default_luma >= 15, "{name}: only {} of {} entries ask for a luma weight other than 1.0", census.non_default_luma, census.entries);
        assert!(census.non_zero_luma_offset >= 10, "{name}: only {} entries carry a non-zero luma offset", census.non_zero_luma_offset);
        assert!(
            census.denominators.len() > 1,
            "{name}: every slice codes the same luma_log2_weight_denom ({:?}), so a recovery that read the wrong one would still agree",
            census.denominators
        );
        assert!(
            census.denominators.iter().any(|&d| d != 7),
            "{name}: every luma_log2_weight_denom is 7, so `coded - (1 << denom)` is `coded - 128` everywhere and a wrong denominator is indistinguishable"
        );
        assert!(
            census.chroma_denominators.iter().any(|&d| d != 0),
            "{name}: no slice lowers ChromaLog2WeightDenom, so the signed delta_chroma_log2_weight_denom is only ever 0 or positive here: {:?}",
            census.chroma_denominators
        );
        assert!(census.b_slices >= 10, "{name}: only {} B pictures, so the L1 weight table is barely exercised", census.b_slices);

        let reference = ffmpeg_pictures(&ff, &path, &s);
        let info = filmcraft_platform::vaapi::stream_info(&s.entry).expect("an hvcC stream");
        let mut va = match VaDecoder::new(info) {
            Ok(va) => va,
            Err(e) => {
                eprintln!("SKIPPED: no VA-API decoder for this driver: {e}");
                continue;
            }
        };
        let (mut out, mut declined) = (Vec::new(), Vec::new());
        for (smp, pts) in &s.samples {
            match va.decode(smp, *pts) {
                Ok(mut frames) => out.append(&mut frames),
                Err(e) => declined.push(e.to_string()),
            }
        }
        out.extend(va.flush());
        assert_eq!(
            declined.len(),
            0,
            "{name}: a weighted HEVC slice this record can carry must not be declined, and {} were: {}",
            declined.len(),
            declined.first().map(String::as_str).unwrap_or("")
        );
        let (right, first_wrong) = oracle(&out, &reference);
        println!("{name} against ffmpeg's own decode: this backend is right on {right}/{} pictures", out.len());
        assert_eq!(right, out.len(), "{name}: the first disagreement with ffmpeg is at pts {first_wrong:?}");
        // The run stayed in hardware: `hw.name()` is what this instance is using, not a counter another
        // test could move.
        assert!(va.name().starts_with("VA-API"), "{name}: stayed in hardware ({})", va.name());
        // and the software decoder is ffmpeg's too, so the comparison the rest of this file makes is
        // against a right reference
        let software = decode_all(software(&s).as_mut(), &s.samples);
        let (sw_right, _) = oracle(&software, &reference);
        assert_eq!(sw_right, reference.len(), "{name}: the software decoder disagrees with ffmpeg");
        checked += 1;
    }
    assert!(checked > 0, "no HEVC weighted-prediction fixture could be generated");
}

/// What an HEVC stream's own slice headers say about their prediction weights.
///
/// Read with `filmcraft_hevc`, the parser the record itself is built from, over the fixture's samples:
/// this is the census that says a fixture is one the weight path is really reached on, and it is
/// deliberately *not* read out of the record — a census of the record would agree with a record that
/// dropped the weights. What it counts is the **coded** stream, so it is a statement about the fixture
/// and not about this backend.
#[derive(Default)]
struct WeightCensus {
    /// Slice segments that code a `pred_weight_table` at all.
    tables: usize,
    /// `num_ref_idx_l0/l1` entries those tables carry, over both lists.
    entries: usize,
    /// Entries asking for a luma weight other than `1 << luma_log2_weight_denom` — that is, entries
    /// whose `delta_luma_weight_lX` is not 0, which is the whole point of the fixture.
    non_default_luma: usize,
    /// Entries carrying a non-zero `luma_offset_lX`, the field the record's `luma_offset_lX` holds.
    non_zero_luma_offset: usize,
    /// Entries carrying a non-zero `ChromaOffsetLX`.
    non_zero_chroma_offset: usize,
    /// Every `LumaLog2WeightDenom` a slice codes, deduplicated and sorted.
    denominators: Vec<u32>,
    /// Every `ChromaLog2WeightDenom` a slice codes, deduplicated and sorted.
    chroma_denominators: Vec<u32>,
    /// B pictures, whose slices carry the list-1 half of the table.
    b_slices: usize,
}

impl WeightCensus {
    fn describe(&self) -> String {
        format!(
            "{} slice segments code a pred_weight_table over {} entries; {} ask for a luma weight other than 1.0, {} carry a non-zero luma offset, {} a non-zero chroma offset; \
             LumaLog2WeightDenom {:?}, ChromaLog2WeightDenom {:?}; {} B pictures",
            self.tables,
            self.entries,
            self.non_default_luma,
            self.non_zero_luma_offset,
            self.non_zero_chroma_offset,
            self.denominators,
            self.chroma_denominators,
            self.b_slices
        )
    }
}

/// [`WeightCensus`] over a stream, parsing every slice segment header with `filmcraft_hevc`.
///
/// The parameter sets come out of the sample entry, the same way the decoder's own
/// `HevcSets::parse` reads them, and a segment whose header will not parse is counted as nothing
/// rather than aborting the census: what is being asked is "how many of this stream's slices weight",
/// and a header this test cannot read is not one whose weights were counted either way.
fn hevc_weights_of(s: &Stream) -> WeightCensus {
    let info = filmcraft_codecs::hw::NalStreamInfo::from_entry(&s.entry).expect("an hvcC stream").expect("a described stream");
    let mut spss: Vec<Option<std::sync::Arc<filmcraft_hevc::params::Sps>>> = vec![None; 16];
    let mut ppss: Vec<Option<std::sync::Arc<filmcraft_hevc::params::Pps>>> = vec![None; 64];
    for nal in &info.parameter_sets {
        let Some(rbsp) = nal.get(2..) else { continue };
        let rbsp = filmcraft_bitstream::unescape_rbsp(rbsp);
        match nal.first().map(|&h| info.nal_type(h)) {
            Some(33) => {
                if let Ok(sps) = filmcraft_hevc::params::Sps::parse(&rbsp)
                    && let Some(slot) = spss.get_mut(sps.id as usize)
                {
                    *slot = Some(std::sync::Arc::new(sps));
                }
            }
            Some(34) => {
                if let Ok(pps) = filmcraft_hevc::params::Pps::parse(&rbsp)
                    && let Some(slot) = ppss.get_mut(pps.id as usize)
                {
                    *slot = Some(std::sync::Arc::new(pps));
                }
            }
            _ => {}
        }
    }
    let lookup = |id: u32| -> Result<(std::sync::Arc<filmcraft_hevc::params::Pps>, std::sync::Arc<filmcraft_hevc::params::Sps>), filmcraft_hevc::Error> {
        let pps = ppss.get(id as usize).and_then(|p| p.as_ref()).cloned().ok_or_else(|| filmcraft_hevc::Error::MissingParameterSet(format!("PPS {id}")))?;
        let sps = spss
            .get(pps.sps_id as usize)
            .and_then(|s| s.as_ref())
            .cloned()
            .ok_or_else(|| filmcraft_hevc::Error::MissingParameterSet(format!("SPS {}", pps.sps_id)))?;
        Ok((pps, sps))
    };

    let mut c = WeightCensus::default();
    for (sample, _) in &s.samples {
        for nal in info.nals(sample) {
            if !matches!(info.nal_type(nal[0]), 0..=31) {
                continue;
            }
            let Ok(hdr) = filmcraft_hevc::slice::NalHeader::parse(nal) else { continue };
            let rbsp = filmcraft_bitstream::unescape_rbsp(nal.get(2..).unwrap_or_default());
            let Ok((sh, _, _)) = filmcraft_hevc::slice::SliceHeader::parse(&rbsp, hdr, lookup, None) else { continue };
            if sh.is_b() {
                c.b_slices += 1;
            }
            let Some(t) = sh.pwt.as_ref() else { continue };
            c.tables += 1;
            push_unique(&mut c.denominators, t.luma_log2_denom);
            push_unique(&mut c.chroma_denominators, t.chroma_log2_denom);
            let luma_one = 1i32.checked_shl(t.luma_log2_denom).unwrap_or(0);
            for list in 0..2 {
                for e in t.l.get(list).map(Vec::as_slice).unwrap_or_default() {
                    c.entries += 1;
                    if e.luma.0 != luma_one {
                        c.non_default_luma += 1;
                    }
                    if e.luma.1 != 0 {
                        c.non_zero_luma_offset += 1;
                    }
                    if e.chroma.iter().any(|(_, off)| *off != 0) {
                        c.non_zero_chroma_offset += 1;
                    }
                }
            }
        }
    }
    c
}

/// Add `v` to a sorted, deduplicated list. `sort_unstable` on a short list of small integers is
/// cheaper than anything cleverer, and the list stays short (the denominators are 0..=7).
fn push_unique(list: &mut Vec<u32>, v: u32) {
    if list.binary_search(&v).is_err() {
        list.push(v);
        list.sort_unstable();
    }
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
///
/// 16-bit planes go out little-endian, which is how `ffmpeg -pix_fmt yuv420p10le` writes them, so the
/// comparison stays byte for byte at 10 bits rather than comparing a re-encoding of the samples.
fn pixels(f: &DecodedFrame) -> Vec<u8> {
    let luma = f.frame.width as usize * f.frame.height as usize;
    // `Yuv8` and `Yuv16` carry differently typed planes, so each arm flattens its own: unifying them
    // in one `let` would need a cast that hides which depth is being compared.
    match &f.frame.data {
        PixelData::Yuv8 { planes, .. } => {
            let sizes: Vec<usize> = planes.iter().map(|p| p.len()).collect();
            assert_eq!(sizes, vec![luma, luma / 4, luma / 4], "the planes are one 4:2:0 picture of {}x{}", f.frame.width, f.frame.height);
            let mut out = Vec::with_capacity(sizes.iter().sum());
            for p in planes {
                out.extend(p.iter().copied());
            }
            out
        }
        PixelData::Yuv16 { planes, .. } => {
            let sizes: Vec<usize> = planes.iter().map(|p| p.len()).collect();
            assert_eq!(sizes, vec![luma, luma / 4, luma / 4], "the planes are one 4:2:0 picture of {}x{}", f.frame.width, f.frame.height);
            let mut out = Vec::with_capacity(sizes.iter().sum::<usize>() * 2);
            for p in planes {
                for s in p.iter() {
                    out.extend(s.to_le_bytes());
                }
            }
            out
        }
        other => panic!("expected 4:2:0 planes, got {other:?}"),
    }
}
