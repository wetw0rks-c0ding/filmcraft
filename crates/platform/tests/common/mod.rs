//! Shared helpers for the platform tests: ffmpeg-made fixtures (external generator only, never
//! linked), MP4 sample reading, and decoding a run of samples to compare decoders.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use filmcraft_codecs::{DecodedFrame, VideoDecoder};
use filmcraft_frame::PixelData;
use filmcraft_isobmff::SampleEntry;

/// The parity fixtures: (file, ffmpeg arguments). B-frames everywhere (reordering), a 360-line
/// picture (coded 368: cropping), two GOPs or more (seeks), and an open-GOP HEVC stream (CRA with
/// RASL pictures).
///
/// `hevc_weightp.mp4` and `hevc_weightp_main10.mp4` are the two **weighted-prediction** fixtures, and
/// they are the only ones that reach `vaapi_slice_record`'s weight submission with a delta that is not
/// zero: every other HEVC fixture here codes a `pred_weight_table` whose `luma_weight_lX_flag` and
/// `chroma_weight_lX_flag` are all 0, so every weight is 1.0 and `delta_luma_weight_lX` is 0 by
/// construction. Their encoder arguments are `crates/hevc`'s `weightp` / `weightp_main10` entries
/// (`crates/hevc/tests/common/mod.rs:54-55`), the same source, size and x265 parameters, so the two
/// crates measure the same weighting; `crates/platform/tests/vaapi.rs` names that relationship so a
/// change to either side is visible. Only the length and GOP size are this suite's: every fixture
/// here has to be long enough for a seek and for `parity`'s picture floor.
pub const FIXTURES: &[(&str, &[&str])] = &[
    (
        "h264_high.mp4",
        &[
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
            "bframes=3:b-pyramid=normal:keyint=24:min-keyint=24:scenecut=0",
            "-pix_fmt",
            "yuv420p",
        ],
    ),
    (
        "hevc_main.mp4",
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=640x360:r=24:d=3,noise=alls=12:allf=t",
            "-c:v",
            "libx265",
            "-preset",
            "fast",
            "-tag:v",
            "hvc1",
            "-x265-params",
            "log-level=error:bframes=4:keyint=24:min-keyint=24:scenecut=0:open-gop=1",
            "-pix_fmt",
            "yuv420p",
        ],
    ),
    (
        "hevc_main10.mp4",
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=640x360:r=24:d=3,noise=alls=12:allf=t",
            "-c:v",
            "libx265",
            "-preset",
            "fast",
            "-profile:v",
            "main10",
            "-tag:v",
            "hvc1",
            "-x265-params",
            "log-level=error:bframes=4:keyint=24:min-keyint=24:scenecut=0",
            "-pix_fmt",
            "yuv420p10le",
        ],
    ),
    // Weighted prediction, at both depths: `weightp` puts `weighted_pred_flag` in the picture
    // parameter set and `weightb` puts `weighted_bipred_flag`, and `fade=in` is the ingredient that
    // makes x265 derive a weight at all — `weightp=1` on a plain `testsrc2` codes a table whose
    // every entry is at its default, which is exactly the case the other three fixtures here already
    // reach. These are `crates/hevc`'s `weightp` / `weightp_main10` encodes, lengthened to the frame
    // count and GOP this suite needs; see the note on `FIXTURES`.
    (
        "hevc_weightp.mp4",
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=352x288:r=25,format=yuv420p,fade=in:0:25",
            "-frames:v",
            "50",
            "-c:v",
            "libx265",
            "-tag:v",
            "hvc1",
            "-x265-params",
            "log-level=error:weightp=1:weightb=1:bframes=3:keyint=12:min-keyint=12:scenecut=0",
            "-pix_fmt",
            "yuv420p",
        ],
    ),
    (
        "hevc_weightp_main10.mp4",
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=352x288:r=25,format=yuv420p10le,fade=in:0:18",
            "-frames:v",
            "50",
            "-c:v",
            "libx265",
            "-profile:v",
            "main10",
            "-tag:v",
            "hvc1",
            "-x265-params",
            "log-level=error:weightp=1:weightb=1:keyint=12:min-keyint=12:scenecut=0",
            "-pix_fmt",
            "yuv420p10le",
        ],
    ),
];

pub fn dir() -> PathBuf {
    filmcraft_testkit::fixtures_dir("platform")
}

/// Generate `name` with `ffmpeg -y -v error <args> <out>` (cached).
pub fn fixture(ff: &Path, name: &str, args: &[&str]) -> Option<PathBuf> {
    let out = dir().join(name);
    filmcraft_testkit::fixtures::generate(&out, |tmp| {
        let st = Command::new(ff).args(["-y", "-v", "error"]).args(args).arg(tmp).stdin(Stdio::null()).status();
        match st {
            Ok(s) if s.success() => true,
            other => {
                eprintln!("fixture {name}: ffmpeg failed: {other:?}");
                false
            }
        }
    })
}

/// The fixture by name.
pub fn named(ff: &Path, name: &str) -> Option<PathBuf> {
    let (_, args) = FIXTURES.iter().find(|f| f.0 == name)?;
    fixture(ff, name, args)
}

/// A video track's sample entry, samples (decode order) with pts, and sync flags.
pub struct Stream {
    pub entry: SampleEntry,
    pub samples: Vec<(Vec<u8>, i64)>,
    pub sync: Vec<bool>,
}

pub fn read_stream(path: &Path) -> Stream {
    let bytes = std::fs::read(path).unwrap();
    let file = filmcraft_isobmff::open(bytes.clone()).unwrap();
    let t = file.track_of_kind(filmcraft_isobmff::TrackKind::Video).unwrap();
    let track = &file.tracks[t];
    let samples = (0..track.samples.len()).map(|i| (file.read_sample(&bytes, t, i).unwrap(), track.samples[i].pts)).collect();
    Stream { entry: track.entries[0].clone(), samples, sync: track.samples.iter().map(|s| s.is_sync).collect() }
}

/// Feed `samples` then flush.
pub fn decode_all(d: &mut dyn VideoDecoder, samples: &[(Vec<u8>, i64)]) -> Vec<DecodedFrame> {
    let mut out = Vec::new();
    for (s, p) in samples {
        out.extend(d.decode(s, *p).unwrap_or_else(|e| panic!("{}: {e}", d.name())));
    }
    out.extend(d.flush());
    out
}

/// Assert two decoders' outputs are the same pictures (pts order, size, format, colour, aspect
/// and every sample).
pub fn assert_same(what: &str, a: &[DecodedFrame], b: &[DecodedFrame]) {
    let pa: Vec<i64> = a.iter().map(|f| f.pts).collect();
    let pb: Vec<i64> = b.iter().map(|f| f.pts).collect();
    assert_eq!(pa, pb, "{what}: pts sequence");
    for (x, y) in a.iter().zip(b) {
        let (fx, fy) = (&x.frame, &y.frame);
        assert_eq!((fx.width, fx.height, fx.par, fx.color), (fy.width, fy.height, fy.par, fy.color), "{what}: pts {} geometry / colour", x.pts);
        match (&fx.data, &fy.data) {
            (PixelData::Yuv8 { planes: p, chroma: c, .. }, PixelData::Yuv8 { planes: q, chroma: d, .. }) => {
                assert_eq!(c, d, "{what}: chroma");
                for i in 0..3 {
                    assert!(p[i] == q[i], "{what}: pts {} plane {i} differs (max diff {})", x.pts, max_diff8(&p[i], &q[i]));
                }
            }
            (PixelData::Yuv16 { planes: p, chroma: c, bits: bx, .. }, PixelData::Yuv16 { planes: q, chroma: d, bits: by, .. }) => {
                assert_eq!((c, bx), (d, by), "{what}: chroma / bits");
                for i in 0..3 {
                    assert!(p[i] == q[i], "{what}: pts {} plane {i} differs (max diff {})", x.pts, max_diff16(&p[i], &q[i]));
                }
            }
            _ => panic!("{what}: pts {}: pixel formats differ", x.pts),
        }
    }
}

/// Assert two decoders' outputs are the same pictures, matched by presentation time rather than by
/// position.
///
/// [`assert_same`] compares position by position, which is only right when both decoders emit in the
/// same order. A backend that decodes one picture synchronously returns each picture from the
/// `decode` call that produced it, so its output is in *decode* order, while a decoder that holds
/// pictures back for reordering emits *presentation* order (H.264 7.4.2.4.1). Comparing "the sixth
/// picture" of each then compares two different frames. This compares the same checks
/// picture-for-picture, keyed by pts, which is what `examples/vaparity.rs` does for the same reason.
///
/// The draft flag is compared too: a backend that flagged every picture `draft` — approximate
/// samples, a decoder that skips the deblocking filter — would otherwise pass a comparison that looks
/// only at planes, colour and size.
///
/// **Where a presentation time repeats, the pictures at it are compared in the order they came out**
/// — see [`by_pts`]. A VP9 superframe is several frames under one container sample and this
/// project's own software VP9 decoder shows every one of them, so "one timestamp, one picture" is not
/// a property of the format and refusing to compare such a stream would have left the whole VP9
/// comparison untested.
pub fn assert_same_by_pts(what: &str, a: &[DecodedFrame], b: &[DecodedFrame]) {
    let (pa, pb) = (by_pts(a), by_pts(b));
    assert_eq!(pa.keys().collect::<Vec<_>>(), pb.keys().collect::<Vec<_>>(), "{what}: pictures, matched by presentation time");
    for (key, x) in &pa {
        let (pts, nth) = key;
        let y = pb[key];
        let (fx, fy) = (&x.frame, &y.frame);
        assert_eq!(x.draft, y.draft, "{what}: pts {pts} draft flag ({} vs {})", x.draft, y.draft);
        assert_eq!((fx.width, fx.height, fx.par, fx.color), (fy.width, fy.height, fy.par, fy.color), "{what}: pts {pts} geometry / colour");
        let nth = if *nth == 0 { String::new() } else { format!(" picture {nth} of that timestamp") };
        match (&fx.data, &fy.data) {
            (PixelData::Yuv8 { planes: p, chroma: c, .. }, PixelData::Yuv8 { planes: q, chroma: d, .. }) => {
                assert_eq!(c, d, "{what}: pts {pts}{nth} chroma");
                for i in 0..3 {
                    assert!(p[i] == q[i], "{what}: pts {pts}{nth} plane {i} differs (max diff {})", max_diff8(&p[i], &q[i]));
                }
            }
            (PixelData::Yuv16 { planes: p, chroma: c, bits: bx, .. }, PixelData::Yuv16 { planes: q, chroma: d, bits: by, .. }) => {
                assert_eq!((c, bx), (d, by), "{what}: pts {pts}{nth} chroma / bits");
                for i in 0..3 {
                    assert!(p[i] == q[i], "{what}: pts {pts}{nth} plane {i} differs (max diff {})", max_diff16(&p[i], &q[i]));
                }
            }
            _ => panic!("{what}: pts {pts}{nth}: pixel formats differ"),
        }
    }
}

/// `out` keyed by presentation time **and, where a time repeats, by the order the pictures came out
/// at it**.
///
/// Keying a map by pts alone is what hides a picture: two pictures with the same presentation time
/// collapse into one entry and a comparison then never sees either of them. So the occurrence index
/// is part of the key, which is what lets a stream that *legitimately* shows two pictures under one
/// container timestamp — a VP9 superframe, which is several frames under one sample — be compared
/// picture for picture instead of refused.
///
/// **Refusing was the right answer when it was the only one available and it stayed too long.** The
/// first version of this function asserted that every presentation time appeared once, on the grounds
/// that "a decoder that returns a picture twice fails instead of quietly matching". That is true of
/// every codec then tested and false of VP9: FilmCraft's own software VP9 decoder emits one picture
/// per *shown frame* of a superframe (`crates/vp9/src/decoder.rs:248-255`), so the harness refused to
/// compare the very picture it was written to check. Nothing noticed, because the hardware VP9 path
/// decoded nothing and every VP9 comparison was a skip. The property the guard was protecting is kept:
/// a decoder that returns a *different* number of pictures at a presentation time now fails on the
/// key set, and one that returns the same two pictures in a different order fails on the pixels.
pub fn by_pts(out: &[DecodedFrame]) -> std::collections::BTreeMap<(i64, usize), &DecodedFrame> {
    let mut counts: std::collections::BTreeMap<i64, usize> = std::collections::BTreeMap::new();
    let mut map: std::collections::BTreeMap<(i64, usize), &DecodedFrame> = std::collections::BTreeMap::new();
    for f in out {
        let nth = counts.entry(f.pts).or_default();
        map.insert((f.pts, *nth), f);
        *nth += 1;
    }
    // Still said out loud, because a repeated presentation time is the one thing a reader of a
    // failure here would otherwise have to work out from a mismatch: which stream, which timestamp,
    // and how many pictures share it.
    let repeated: Vec<String> = counts.iter().filter(|(_, c)| **c > 1).map(|(p, c)| format!("{p} ({c} times)")).collect();
    if !repeated.is_empty() {
        eprintln!(
            "NOTE: {} pictures over {} distinct presentation times — {} is shown more than once, so they are compared in the order they came out",
            out.len(),
            counts.len(),
            repeated.join(", ")
        );
    }
    map
}

pub fn max_diff8(a: &[u8], b: &[u8]) -> String {
    if a.len() != b.len() {
        return format!("sizes {} vs {}", a.len(), b.len());
    }
    a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0).to_string()
}

pub fn max_diff16(a: &[u16], b: &[u16]) -> String {
    if a.len() != b.len() {
        return format!("sizes {} vs {}", a.len(), b.len());
    }
    a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0).to_string()
}

pub fn xorshift(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}
