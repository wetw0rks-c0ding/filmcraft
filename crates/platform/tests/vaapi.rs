//! VA-API decoding against our software decoders (Linux): every picture of the HEVC fixtures, and of
//! the H.264 fixture that does not code an explicit weighted bipred, must be the software
//! decoder's — bit-exact planes, colour, pixel aspect and picture — also after `reset` + reseek and
//! a mid-stream `flush`; a forced mid-stream failure must continue in software with the software
//! decoder's exact output; damaged samples and parameter sets must give errors or fall back, never
//! crash or hang; a VP9 or AV1 stream must be declined while it is still a description, with a
//! reason. One fixture is not bit-exact, and the section below says which, why, and which way round
//! the two decoders are wrong. Skips without ffmpeg (fixture generator) or without a VA-API driver.
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
//! # One fixture cannot be bit-exact, and it is not the software decoder that is wrong
//!
//! `h264_high.mp4` is x264 High profile with B-frames, so its picture parameter set carries
//! `weighted_bipred_idc == 2` — "explicit", in which *each B slice* codes a `weighted_bipred_flag`
//! and, when that flag is set, a prediction weight table (H.264 7.4.3.2). `filmcraft_h264` does not
//! read `weighted_bipred_flag` (`crates/h264/src/slice.rs:229`), so its `pred_weight_table` is
//! always `None` in this mode; this backend needs the same flag and gets it from the same parser, so
//! a B slice the stream really does weight is submitted with an **empty** weight table
//! (`crates/platform/src/vaapi/picture.rs`, `h264_weights`), which libva's own encoding reads as
//! "explicit mode, this slice not weighted".
//!
//! That is a guess, and it is wrong here, so the two decoders disagree: on this machine, 60 of the
//! fixture's 72 pictures — 48 of its 51 B pictures and 12 of its 18 P pictures; all 3 of its I
//! pictures match. ffmpeg settles which side is right: ffmpeg's own decode of the fixture is
//! bit-for-bit `filmcraft_h264`'s on all 72 pictures and agrees with this backend on 12
//! ([`the_weighted_bipred_divergence_is_this_backends_not_the_software_decoders`] pins that, with a
//! negative control so "matches ffmpeg" cannot pass vacuously). **This must not be "fixed" by
//! making the hardware path reproduce whatever the software decoder does**: that would replace 60
//! correct pictures with 60 incorrect ones. The hardware path has to read `weighted_bipred_flag`;
//! the fix belongs in `filmcraft-h264`, which would have to expose the flag.
//!
//! The P pictures are worth a word, because the divergence is not confined to B slices:
//! `h264_high_unweighted.mp4` is the same source, encoder, frame rate, GOPs and B frames with
//! `weightb=0` added, so its `weighted_bipred_idc` is 0 while its `weighted_pred_flag` is still 1 —
//! the P weight tables are still coded and still submitted — and it comes out of this backend
//! bit-exact on all 72 pictures. So the P weights this backend sends are right; the 12 P pictures
//! that differ are the ones whose references reach a B picture that was decoded wrong, and a
//! prediction cannot come out right from a reference that is wrong. One encoder parameter is what
//! separates the two fixtures, so it is what the divergence follows.
//!
//! So the parity assertion is two-tiered: [`Agreement::EveryPicture`] for the streams that do not
//! hit this, which is every picture bit for bit and is what interchangeability means, and
//! [`Agreement::StreamOnly`] for the one that does, which pins the stream (same pictures, same
//! geometry, colour and pixel aspect) and every intra-coded picture bit for bit, and reports how
//! many inter-coded pictures differ. There is no blanket allowance: that would absorb a real parity
//! regression.
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

/// What the two decoders must agree on, picture for picture, for one fixture.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Agreement {
    /// Every picture, bit for bit. This is what "interchangeable" means.
    EveryPicture,
    /// The stream, not the pixels: the same pictures with the same geometry, colour and pixel
    /// aspect, and every intra-coded picture bit for bit. Only for the fixture whose
    /// `weighted_bipred_flag` this backend cannot read — see the file's doc comment.
    StreamOnly,
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
/// not depend on how many cores the machine has. Measured on `h264_high.mp4`, ten runs each at 1, 4
/// and 16 threads: every run — one thread or many — ended in a `flush` with all 72 pictures, one
/// presentation-time set and one set of pixels, so this reference is deterministic and pinning is
/// belt and braces rather than a workaround. What the thread count does change is how many pictures
/// come out of `decode` *before* the `flush`, because the pictures still held back for reordering
/// are the ones the frame-threaded path has not finished: 70 at one thread, 65 at four, 45–53 at
/// sixteen. A harness that hashes what `decode` returns and never calls `flush` — as
/// `examples/vaparity.rs` does — is what reported "63 or 65 pictures" for this fixture.
fn software(s: &Stream) -> Box<dyn VideoDecoder> {
    match &s.entry.codec {
        filmcraft_isobmff::CodecConfig::Avc(a) => Box::new(filmcraft_codecs::video::H264Decoder::with_threads(a.to_bytes(), 1).unwrap()),
        _ => filmcraft_codecs::software_video_decoder(&s.entry).unwrap(),
    }
}

/// The parity fixtures and what each must agree on.
///
/// `h264_high_unweighted.mp4` is `h264_high.mp4` with `weightb=0`, so its picture parameter set
/// says `weighted_bipred_idc == 0`: the same encoder, the same B-frames, the same two GOPs, the
/// same cropped 360-line picture, without the one syntax element this backend cannot read. It is
/// what gives an H.264 stream the full bit-exact parity assertion, so that assertion is a real
/// H.264 assertion and not only HEVC.
fn streams(ff: &Path) -> Vec<(String, Stream, Agreement)> {
    let mut out = Vec::new();
    for (name, _) in FIXTURES {
        if let Some(path) = named(ff, name) {
            let agreement = if *name == "h264_high.mp4" { Agreement::StreamOnly } else { Agreement::EveryPicture };
            out.push((name.to_string(), read_stream(&path), agreement));
        }
    }
    if let Some(path) = unweighted(ff) {
        out.push(("h264_high_unweighted.mp4".to_string(), read_stream(&path), Agreement::EveryPicture));
    }
    out
}

/// The H.264 fixture that does not code an explicit `weighted_bipred_flag`: `h264_high.mp4` with
/// `weightb=0` added.
fn unweighted(ff: &Path) -> Option<PathBuf> {
    let args = [
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=640x360:r=24:d=3,noise=alls=12:allf=t",
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-preset",
        "fast",
        "-x264-params",
        "bframes=3:b-pyramid=normal:keyint=24:min-keyint=24:scenecut=0:weightb=0",
        "-pix_fmt",
        "yuv420p",
    ];
    fixture(ff, "h264_high_unweighted.mp4", &args)
}

/// A picture's geometry, colour, pixel aspect, chroma format and depth — everything about it but
/// its pixels.
fn description(f: &DecodedFrame) -> String {
    let v = &f.frame;
    let planes = match &v.data {
        PixelData::Yuv8 { chroma, .. } => format!("8-bit {chroma:?}"),
        PixelData::Yuv16 { chroma, bits, .. } => format!("{bits}-bit {chroma:?}"),
        _ => "another pixel format".to_string(),
    };
    format!("{}x{} {:?} {:?} {planes}", v.width, v.height, v.par, v.color)
}

/// The two outputs matched by presentation time, and the checks [`assert_same_by_pts`] makes.
fn compare(what: &str, a: &[DecodedFrame], b: &[DecodedFrame], agreement: Agreement) {
    if agreement == Agreement::EveryPicture {
        assert_same_by_pts(what, a, b);
        return;
    }
    let (pa, pb) = (by_pts(a), by_pts(b));
    assert_eq!(pa.keys().collect::<Vec<_>>(), pb.keys().collect::<Vec<_>>(), "{what}: presentation times");
    for (pts, x) in &pa {
        assert_eq!(description(x), description(pb[pts]), "{what}: pts {pts} geometry / colour");
        assert_eq!(x.draft, pb[pts].draft, "{what}: pts {pts} draft flag");
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

/// Assert every intra-coded picture of the two outputs is bit for bit the same, and report how many
/// of the inter-coded ones are not.
///
/// This is what the `StreamOnly` tier really asserts: an intra-coded picture depends on nothing but
/// its own slices, so the missing `weighted_bipred_flag` cannot reach it, and it must still come out
/// of the hardware exactly as the software decoder produced it. The count that follows is the
/// divergence the file's doc comment describes; it is printed, not bounded.
fn intra_coded_pictures_are_bit_exact(name: &str, info: &NalStreamInfo, s: &Stream, a: &[DecodedFrame], b: &[DecodedFrame]) {
    let kinds = h264_slice_types(info, s);
    let (pa, pb) = (by_pts(a), by_pts(b));
    let mut intra = 0usize;
    let (mut inter_same, mut inter_differ) = (0usize, 0usize);
    for (pts, x) in &pa {
        let raw = *kinds.get(pts).unwrap_or_else(|| panic!("{name}: no slice type read for pts {pts}"));
        let y = pb[pts];
        let same = match (&x.frame.data, &y.frame.data) {
            (PixelData::Yuv8 { planes: p, chroma: c, .. }, PixelData::Yuv8 { planes: q, chroma: d, .. }) => c == d && (0..3).all(|i| p[i] == q[i]),
            (PixelData::Yuv16 { planes: p, chroma: c, bits: bx, .. }, PixelData::Yuv16 { planes: q, chroma: d, bits: by, .. }) => {
                (c, bx) == (d, by) && (0..3).all(|i| p[i] == q[i])
            }
            _ => panic!("{name}: pts {pts}: pixel formats differ"),
        };
        match raw {
            2 => {
                assert!(same, "{name}: intra-coded pts {pts} differs from the software decoder");
                intra += 1;
            }
            _ if same => inter_same += 1,
            _ => inter_differ += 1,
        }
    }
    assert!(intra > 0, "{name}: no intra-coded picture in the fixture");
    // If nothing diverges any more the `StreamOnly` tier is no longer needed for this fixture, and
    // saying so here beats leaving a weakened assertion in place after the fix lands.
    assert!(inter_differ > 0, "{name}: no inter-coded picture diverges, so this fixture belongs in the every-picture tier");
    println!(
        "{name}: {intra} intra-coded pictures bit-exact; {inter_same} of {} inter-coded pictures agree and {inter_differ} differ (see this file's doc comment)",
        inter_same + inter_differ
    );
}

/// Every picture of `s` from the hardware decoder is the software decoder's: whole stream, after
/// `reset` + reseek to each later sync sample, and a full pass again after all the resets. This is
/// Review Focus #5's test.
fn parity(name: &str, s: &Stream, agreement: Agreement) {
    let Some(mut hw) = hardware(s) else { return };
    let mut sw = software(s);
    let a = decode_all(hw.as_mut(), &s.samples);
    let b = decode_all(sw.as_mut(), &s.samples);
    assert!(a.len() >= 40, "{name}: {} pictures", a.len());
    assert_eq!(a.len(), b.len(), "{name}: picture counts");
    compare(name, &a, &b, agreement);
    if agreement == Agreement::StreamOnly {
        let info = NalStreamInfo::from_entry(&s.entry).unwrap().unwrap();
        intra_coded_pictures_are_bit_exact(name, &info, s, &a, &b);
    }
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
        compare(&format!("{name} from sample {k}"), &a, &b, agreement);
    }
    // back to the start after seeking around: the whole stream again
    hw.reset();
    sw.reset();
    let a = decode_all(hw.as_mut(), &s.samples);
    let b = decode_all(sw.as_mut(), &s.samples);
    assert_eq!(a.len(), b.len(), "{name} after resets: picture count");
    compare(&format!("{name} after resets"), &a, &b, agreement);

    // random-access / disposable answers match the software decoder's
    for (smp, _) in &s.samples {
        assert_eq!(hw.is_random_access(smp), sw.is_random_access(smp), "{name}");
        assert_eq!(hw.is_disposable(smp), sw.is_disposable(smp), "{name}");
    }
}

#[test]
fn bit_exact_with_the_software_decoder() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let streams = streams(&ff);
    assert!(!streams.is_empty(), "no parity fixture could be generated");
    for (name, s, agreement) in &streams {
        parity(name, s, *agreement);
    }
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
    for (name, s, _) in streams(&ff) {
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
    for (name, s, agreement) in streams(&ff) {
        let name = name.as_str();
        let Some(info) = filmcraft_platform::vaapi::stream_info(&s.entry) else {
            eprintln!("SKIPPED: no VA-API stream description for {name}");
            continue;
        };
        if VaDecoder::new(info.clone()).is_err() {
            eprintln!("SKIPPED: no VA-API decoder for {name}");
            continue;
        }
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
        let mut va = VaDecoder::new(info.clone()).unwrap();
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
                compare(&format!("{name} flush then IDR {sync}"), &a, &b, agreement);
            }
            None => eprintln!("SKIPPED: no restart point after sample {mid} of {name}"),
        }
        // The same drain through the hybrid decoder, which is what that error is *for*: it hands the
        // run to the software decoder, so the finished run has to be the software decoder's own
        // pictures — the first half from the hardware, the second half replayed through software.
        let reference = decode_all(software(&s).as_mut(), &s.samples);
        let mut d = HybridDecoder::new(Box::new(VaDecoder::new(info.clone()).unwrap()), s.entry.clone(), info.clone());
        let mut out: Vec<DecodedFrame> = Vec::new();
        for (smp, pts) in &s.samples[..mid] {
            out.extend(d.decode(smp, *pts).unwrap());
        }
        assert!(d.is_hardware(), "{name}: still in hardware after the first half");
        assert!(d.flush().is_empty(), "{name}: the hardware holds nothing back, so the flush returns nothing");
        let fallbacks = filmcraft_codecs::hw::hw_stats().fallbacks;
        out.extend(decode_all(&mut d, &s.samples[mid..]));
        assert!(!d.is_hardware(), "{name}: a drained context did not hand the run to the software decoder");
        assert!(filmcraft_codecs::hw::hw_stats().fallbacks > fallbacks, "{name}: the fallback from a drained context is counted");
        compare(&format!("{name} flushed mid-GOP at {mid}"), &out, &reference, agreement);
    }
}

#[test]
fn a_forced_mid_stream_failure_continues_in_software() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    for (name, s, agreement) in streams(&ff) {
        let name = name.as_str();
        let Some(info) = filmcraft_platform::vaapi::stream_info(&s.entry) else { continue };
        if VaDecoder::new(info.clone()).is_err() {
            eprintln!("SKIPPED: no VA-API decoder for {name}");
            continue;
        }
        // The reference is the *software* decoder's whole output: after the forced failure the run
        // is replayed through it, so that — not the hardware decoder's — is what must come out.
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
            // the output must be the software decoder's own bit for bit — including on the fixture
            // whose pictures this backend decodes differently. Once the failure is further in, the
            // pictures before it came out of the hardware, so that fixture can only be compared at
            // the tier it earns; the streams this backend decodes correctly are compared in full.
            let earned = if fail_at == 0 { Agreement::EveryPicture } else { agreement };
            compare(&format!("{name} failing at sample {fail_at}"), &out, &reference, earned);
            // the instance stays in software across seeks
            d.reset();
            compare(&format!("{name} after the fallback and a reset"), &decode_all(&mut d, &s.samples), &reference, earned);
        }
    }
}

/// Run `f` on a thread; fail if it panics or takes longer than `limit`, else hand back what it
/// returned. The value is there so a caller can assert on what the damaged runs actually did: a
/// test that only checks "nothing blew up" also passes when the damage never reached anything.
fn bounded<T: Send + 'static>(what: &str, limit: Duration, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        let _ = tx.send(r.ok());
    });
    match rx.recv_timeout(limit) {
        Ok(Some(v)) => v,
        Ok(None) => panic!("{what}: panicked"),
        Err(_) => panic!("{what}: hung"),
    }
}

#[test]
fn damaged_samples_never_crash_or_hang() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    for (name, s, _) in streams(&ff) {
        let name = name.as_str();
        if hardware(&s).is_none() {
            continue;
        }
        // bit flips, truncation and corrupt length prefixes, through the factory's decoder
        let mut rejected = 0usize;
        for seed in 1..=6u64 {
            let samples = s.samples.clone();
            let entry = s.entry.clone();
            rejected += bounded(&format!("{name} seed {seed}"), Duration::from_secs(60), move || {
                let mut seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
                let mut errors = 0usize;
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
                    if i % 13 == 12 {
                        d.reset();
                    }
                }
                let _ = d.flush();
                errors
            });
        }
        // The damage has to reach the decoder, or "never crashed" is a statement about clean samples: on
        // every fixture these six runs are rejected well over a hundred times between them.
        assert!(rejected > 0, "{name}: none of the 240 damaged access units was rejected");
        println!("{name}: {} of 240 damaged access units were rejected as errors, the rest decoded or were dropped", rejected);
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

/// The weighted-bipred divergence is this backend's, not the software decoder's: against ffmpeg's own
/// decode of `h264_high.mp4` the software decoder is right on every picture and this backend on
/// only some.
///
/// Without this, the `StreamOnly` tier above would be a gap with nothing saying which side is wrong,
/// and the obvious "fix" would be to make the hardware path reproduce the software decoder's output
/// — which would replace 60 correct pictures with 60 incorrect ones.
#[test]
fn the_weighted_bipred_divergence_is_this_backends_not_the_software_decoders() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let Some(path) = named(&ff, "h264_high.mp4") else {
        eprintln!("SKIPPED: h264_high.mp4 could not be generated");
        return;
    };
    let s = read_stream(&path);
    let Some(mut hw) = hardware(&s) else { return };
    // ffmpeg as the oracle: its own decode of the fixture, in presentation order.
    let raw = std::process::Command::new(&ff)
        .args(["-v", "error", "-i", path.to_str().unwrap(), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
        .output()
        .expect("ffmpeg decode");
    assert!(raw.status.success(), "ffmpeg failed: {}", String::from_utf8_lossy(&raw.stderr));
    let size = 640 * 360 * 3 / 2;
    assert_eq!(raw.stdout.len() % size, 0, "ffmpeg's decode is a whole number of 640x360 4:2:0 pictures");

    let software_out = decode_all(software(&s).as_mut(), &s.samples);
    let hardware_out = decode_all(hw.as_mut(), &s.samples);
    assert_eq!(software_out.len(), hardware_out.len(), "picture counts");
    assert_eq!(software_out.len(), raw.stdout.len() / size, "ffmpeg decoded as many pictures as FilmCraft");
    // ffmpeg emits in presentation order, so a picture is paired with ffmpeg's by sorted presentation
    // time — the same matching every other comparison in this file uses.
    let by_pts = |out: &[DecodedFrame]| out.iter().map(|f| (f.pts, pixels(f))).collect::<BTreeMap<_, _>>();
    let (mine, theirs) = (by_pts(&hardware_out), by_pts(&software_out));
    // Both maps are keyed by presentation time, so a picture returned twice would collapse into one
    // entry and this comparison would score the survivors instead of the run.
    assert_eq!(mine.len(), hardware_out.len(), "this backend returned a presentation time twice");
    assert_eq!(theirs.len(), software_out.len(), "the software decoder returned a presentation time twice");
    let mut order: Vec<i64> = theirs.keys().copied().collect();
    order.sort_unstable();
    let (mut software_right, mut hardware_right) = (0usize, 0usize);
    for (i, pts) in order.iter().enumerate() {
        let want = raw.stdout.get(i * size..(i + 1) * size).expect("a whole picture");
        if theirs[pts] == want {
            software_right += 1;
        }
        if mine[pts] == want {
            hardware_right += 1;
        }
        // The negative control: a picture that matched its own index must not also match its
        // neighbour's, or "matches ffmpeg" would be measuring nothing.
        if let Some(next) = raw.stdout.get((i + 1) * size..(i + 2) * size) {
            assert_ne!(theirs[pts], next, "picture {i} matches ffmpeg's {i} and its successor");
        }
    }
    println!(
        "h264_high.mp4 against ffmpeg's own decode: the software decoder is right on {software_right}/{}, this backend on {hardware_right}/{}",
        software_out.len(),
        software_out.len()
    );
    assert_eq!(software_right, software_out.len(), "the software decoder agrees with ffmpeg on every picture");
    assert!(
        hardware_right < software_right,
        "this backend agrees with ffmpeg on fewer pictures ({hardware_right}) than the software decoder ({software_right})"
    );
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
