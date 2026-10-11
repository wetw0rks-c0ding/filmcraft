//! Does the VA-API decode agree with FilmCraft's own decoder, picture for picture?
//!
//! ```text
//! vaparity [--hw] <file>   compare the two decoders' luma hashes, picture by picture
//! ```
//!
//! **Both stream shapes go through it.** The hardware decoder is asked for its stream description
//! with `vaapi::stream_info`, which covers `avcC` / `hvcC` and the frame-coded sample entries
//! (`vp09` + `vpcC`) alike; asking the NAL parser alone reported a VP9 file as "not an avcC / hvcC
//! stream" and silently ran the software half alone.
//!
//! Pictures are matched by presentation time, not by position: a decoder that reorders emits its
//! pictures in a different order from one that does not, so comparing "the sixth picture" of each
//! compares two different frames. Every sample is fed to *both* decoders whatever the other one did
//! with it — a decoder that is skipped a sample falls behind and then reports a missing reference,
//! which is a fact about the harness and not about the backend.
//!
//! Both decoders are flushed before they are compared, and the comparison walks the **union** of the
//! two presentation-time sets. Either half matters: without the flush a reordering decoder's last
//! pictures are simply not there (a 72-picture stream reads as 65, which looks like pictures going
//! missing), and without the union a picture only one decoder produced is skipped instead of counted
//! — which is how a stream this backend refuses in hardware read as a match on the few pictures it
//! decoded before the refusal.
//!
//! Off-tree on purpose: it needs a real driver, and a test that only runs when one does not belong
//! in the crate's test suite. `vaprobe` is the shipped diagnostic; this is the parity harness I use
//! to check `vaprobe`'s numbers against something independent.
#[cfg(target_os = "linux")]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let hw = args.iter().any(|a| a == "--hw");
    for path in args.iter().filter(|a| !a.starts_with("--")) {
        parity(path, hw);
    }
}

#[cfg(target_os = "linux")]
fn parity(path: &str, hw: bool) {
    use filmcraft_codecs::VideoDecoder as _;
    let Ok(bytes) = std::fs::read(path) else {
        println!("{path}: unreadable");
        return;
    };
    let Ok(file) = filmcraft_isobmff::open(bytes.clone()) else {
        println!("{path}: not an MP4");
        return;
    };
    let Some(t) = file.track_of_kind(filmcraft_isobmff::TrackKind::Video) else {
        println!("{path}: no video");
        return;
    };
    let Some(entry) = file.tracks[t].entries.first() else { return };
    let Some(software) = software_decoder(entry) else {
        println!("{path}: no software decoder for this stream");
        return;
    };
    let Ok(mut software) = software else {
        println!("{path}: the software decoder could not be made for this stream");
        return;
    };

    // The stream description the hardware backend needs, or why it is declined.
    //
    // **`vaapi::stream_info`, not `NalStreamInfo::from_entry`.** The latter is the NAL path and says
    // `None` for anything that is not an `avcC` / `hvcC` sample entry — which read "this entry is not
    // an avcC / hvcC stream" for a VP9 file and made the hardware half of a VP9 oracle run report a
    // refusal for a stream the backend decodes. `stream_info` is the backend's own front door and
    // covers both shapes (see `vaapi/stream.rs`), so asking it means this harness exercises the same
    // decision `VaDecoder::new` would be given by a caller.
    let described = filmcraft_platform::vaapi::stream_info(entry);
    let declined = described.is_none();
    let mut hardware: Option<filmcraft_platform::vaapi::VaDecoder> = match (hw, described) {
        (true, Some(info)) => filmcraft_platform::vaapi::VaDecoder::new(info).ok(),
        _ => None,
    };
    if hw && hardware.is_none() {
        let why = if declined { "the backend does not describe this sample entry".to_string() } else { "VaDecoder::new declined it".to_string() };
        println!("{path}: no hardware decoder: {why}");
    }

    let mut sw_by_pts: std::collections::HashMap<i64, Picture> = std::collections::HashMap::new();
    let mut hw_by_pts: std::collections::HashMap<i64, Picture> = std::collections::HashMap::new();
    // Pictures **returned**, alongside the maps. Two pictures at one presentation time collapse into one
    // map entry, so the map's length alone would read as pictures going missing — the very confusion
    // this file's doc exists to remove — and nothing in a `HashMap` records that it happened. The two
    // counts are compared instead, and a difference is reported rather than claimed away.
    let (mut sw_pictures, mut hw_pictures) = (0usize, 0usize);
    let mut hw_errors: Vec<(usize, String)> = Vec::new();
    let mut sw_errors: Vec<(usize, String)> = Vec::new();

    for i in 0..file.tracks[t].samples.len() {
        let Ok(s) = file.read_sample(&bytes, t, i) else { break };
        let pts = file.tracks[t].samples[i].pts;
        // Every sample goes to both decoders. `continue` past one of them would leave the other a
        // picture behind, and the reference list for the next sample would be missing an entry.
        match software.decode(&s, pts) {
            Ok(frames) => {
                sw_pictures += frames.len();
                for f in frames {
                    sw_by_pts.insert(f.pts, measure(&f.frame));
                }
            }
            Err(e) => sw_errors.push((i, e.to_string())),
        }
        if let Some(d) = hardware.as_mut() {
            match d.decode(&s, pts) {
                Ok(frames) => {
                    hw_pictures += frames.len();
                    for f in frames {
                        hw_by_pts.insert(f.pts, measure(&f.frame));
                    }
                }
                Err(e) => hw_errors.push((i, e.to_string())),
            }
        }
    }
    // Both decoders are flushed, and the pictures the flush returns are the run: a decoder that
    // reorders holds the last pictures of a run back, and counting only what `decode` returned is how
    // a 72-picture stream reads as 65 — which looks like pictures going missing and is not.
    for f in software.flush() {
        sw_pictures += 1;
        sw_by_pts.insert(f.pts, measure(&f.frame));
    }
    if let Some(d) = hardware.as_mut() {
        for f in d.flush() {
            hw_pictures += 1;
            hw_by_pts.insert(f.pts, measure(&f.frame));
        }
    }

    if args_dump() {
        let mut pts: Vec<i64> = hw_by_pts.keys().copied().collect();
        pts.sort_unstable();
        for p in pts {
            println!("HW {p} {:016x}", hw_by_pts[&p].hash);
        }
        let mut spts: Vec<i64> = sw_by_pts.keys().copied().collect();
        spts.sort_unstable();
        for p in spts {
            println!("SW {p} {:016x}", sw_by_pts[&p].hash);
        }
        return;
    }
    if !hw {
        println!("{path}: {sw_pictures} pictures from the software decoder over {} distinct presentation times", sw_by_pts.len());
        if sw_pictures != sw_by_pts.len() {
            println!("  {} of them share a presentation time with another, which is why the two counts differ", sw_pictures - sw_by_pts.len());
        }
        return;
    }
    let (mut agree, mut differ) = (0usize, 0usize);
    let mut first_diff = None;
    // The **union** of both sets, not the software one's: a picture only the hardware decoder produced is
    // a difference, and iterating the software set alone would skip it — which is what made a stream
    // this backend refuses in hardware (every picture after the refusal comes from software) read as a
    // match on the few the hardware had before it.
    let mut all: Vec<i64> = sw_by_pts.keys().chain(hw_by_pts.keys()).copied().collect();
    all.sort_unstable();
    all.dedup();
    let mut only_hardware = 0usize;
    let mut only_software = 0usize;
    // **Per-frame numbers, for every picture that differs.** A hash says *that* two pictures differ
    // and nothing about how, and "how" is what distinguishes a blank surface from a picture built
    // from the wrong references from a picture off by one quantizer step. So each differing
    // presentation time is reported on its own line with the share of sampled luma values that
    // differ, the largest single difference and the mean absolute difference — enough to tell a
    // cleared render target (everything wrong, wildly) from a near miss (a few taps, by one or two
    // levels) without dumping frames to disk.
    let mut per_frame: Vec<String> = Vec::new();
    for p in &all {
        let (a, b) = (sw_by_pts.get(p), hw_by_pts.get(p));
        // `p` came out of the union, so exactly one side has it once the both-sides case is out of the
        // way — which is what makes the `else` below "only the hardware decoder's" rather than a
        // catch-all that would mislabel a picture neither has. Nothing here can panic: an example that
        // reports on a decoder has no business unwinding, and the examples/ target is not covered by the
        // lib crate's `deny`, so a panic path there would not be caught by the build.
        if let (Some(a), Some(b)) = (a, b) {
            if a.hash == b.hash {
                agree += 1;
            } else {
                differ += 1;
                per_frame.push(format!("  pts {p}: {}", a.against(b)));
                if first_diff.is_none() {
                    first_diff = Some(format!("pts {p}: software {:016x} != hardware {:016x}", a.hash, b.hash));
                }
            }
            continue;
        }
        differ += 1;
        if a.is_some() {
            only_software += 1;
            if first_diff.is_none() {
                first_diff = Some(format!("pts {p}: only the software decoder produced it"));
            }
        } else {
            only_hardware += 1;
            if first_diff.is_none() {
                first_diff = Some(format!("pts {p}: only the hardware decoder produced it"));
            }
        }
    }
    let total = all.len();
    let verdict = if differ == 0 && agree > 0 { "MATCH" } else { "MISMATCH" };
    println!("{verdict} {path}: {agree} of {total} pictures identical, {differ} differ");
    println!(
        "  {sw_pictures} pictures from the software decoder over {} distinct presentation times, {hw_pictures} from the hardware decoder over {}",
        sw_by_pts.len(),
        hw_by_pts.len()
    );
    for (what, pictures, distinct) in [("software", sw_pictures, sw_by_pts.len()), ("hardware", hw_pictures, hw_by_pts.len())] {
        if pictures != distinct {
            println!("  {} of the {what} decoder's pictures share a presentation time with another, so the two counts differ", pictures - distinct);
        }
    }
    if only_hardware > 0 || only_software > 0 {
        println!("  {only_hardware} of them only the hardware decoder produced, {only_software} only the software decoder's");
    }
    if let Some(why) = first_diff {
        println!("  first difference: {why}");
    }
    if let Some((i, e)) = hw_errors.first() {
        println!("  first hardware error at sample {i}: {e} ({} errors)", hw_errors.len());
    }
    if let Some((i, e)) = sw_errors.first() {
        println!("  first software error at sample {i}: {e} ({} errors)", sw_errors.len());
    }
    for line in &per_frame {
        println!("{line}");
    }
}

/// What this harness keeps of one picture: its luma hash, and an evenly spaced sample of its luma
/// values to measure a difference with.
///
/// **The sample is bounded on purpose.** Keeping every plane of every picture would make the harness
/// scale with the file — a 4K run would hold gigabytes — and the numbers a mismatch needs are the
/// *shape* of the difference, not a byte-exact diff of two large buffers. [`SAMPLES`] values spread
/// across the plane answer "how much of it is wrong, by how much, and does it look like a cleared
/// surface or a near miss", which is what tells a wrong reference apart from a wrong quantizer.
struct Picture {
    /// FNV-1a/64 over the whole luma plane's bytes, one byte per 8-bit sample and two per 10-bit one
    /// — **the same measure `vaprobe` prints**, so the two agree and so a number from either is
    /// comparable with a number from the other.
    hash: u64,
    /// `SAMPLES` luma values spread across the plane, as `u32` so an 8- and a 10-bit picture compare
    /// the same way.
    samples: Vec<u32>,
    /// The mean of the whole luma plane, which is what says whether one picture is *brighter* than
    /// the other — the difference between a colour-range or matrix mistake and a picture that is
    /// simply the wrong picture.
    mean: f64,
}

/// How many luma values one picture is reduced to for measurement.
const SAMPLES: usize = 1024;

/// [`Picture::against`]: the sentence a parity report prints for one picture that differs.
impl Picture {
    fn against(&self, other: &Picture) -> String {
        if self.samples.len() != other.samples.len() {
            return format!("the two pictures are different sizes ({} and {} sampled luma values)", self.samples.len(), other.samples.len());
        }
        let mut differing = 0usize;
        let mut worst = 0i64;
        let mut total = 0i64;
        let mut first_at = None;
        for (i, (a, b)) in self.samples.iter().zip(&other.samples).enumerate() {
            let d = i64::from(*a) - i64::from(*b);
            if d != 0 {
                differing += 1;
                worst = worst.max(d.abs());
                total += d.abs();
                first_at.get_or_insert(i);
            }
        }
        let n = self.samples.len().max(1);
        format!(
            "{differing} of {} sampled luma values differ ({}%), largest {worst}, mean {:.2}, first at index {}; \
             luma mean software {:.1}, hardware {:.1}; software starts {:?}, hardware starts {:?}",
            n,
            differing * 100 / n,
            total as f64 / n as f64,
            first_at.map_or_else(|| "-".to_string(), |i| i.to_string()),
            self.mean,
            other.mean,
            &self.samples[..self.samples.len().min(6)],
            &other.samples[..other.samples.len().min(6)]
        )
    }
}

/// One picture, reduced to what the comparison above needs.
///
/// The luma plane is walked once for the hash and once for the sample, so the sample is taken at a
/// fixed stride rather than from the head of the plane — a stride over the head would only ever
/// measure the first rows, and the top row is the same whatever went wrong below it.
#[cfg(target_os = "linux")]
fn measure(frame: &filmcraft_frame::VideoFrame) -> Picture {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let fold = |h: &mut u64, b: u8| *h = (*h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
    let values: Vec<u32> = match &frame.data {
        filmcraft_frame::PixelData::Yuv8 { planes, .. } => planes[0].iter().map(|&v| u32::from(v)).collect(),
        filmcraft_frame::PixelData::Yuv16 { planes, .. } => planes[0].iter().map(|&v| u32::from(v)).collect(),
        // A format this backend does not produce compares as empty rather than as equal: two empty
        // pictures hashing the same is the one result that would be a false "identical".
        _ => Vec::new(),
    };
    match &frame.data {
        filmcraft_frame::PixelData::Yuv8 { .. } => {
            for &v in &values {
                fold(&mut hash, v as u8);
            }
        }
        filmcraft_frame::PixelData::Yuv16 { .. } => {
            for &v in &values {
                for b in v.to_le_bytes() {
                    fold(&mut hash, b);
                }
            }
        }
        _ => {}
    }
    let stride = (values.len() / SAMPLES).max(1);
    let samples = values.iter().step_by(stride).take(SAMPLES).copied().collect();
    // `u64` for the sum so a 4K plane of 8-bit samples cannot overflow the accumulator.
    let total: u64 = values.iter().map(|&v| u64::from(v)).sum();
    let mean = total as f64 / values.len().max(1) as f64;
    Picture { hash, samples, mean }
}

#[cfg(target_os = "linux")]
fn args_dump() -> bool {
    std::env::args().any(|a| a == "--dump")
}

/// The software decoder FilmCraft would use for this entry, which is the reference either way.
#[cfg(target_os = "linux")]
fn software_decoder(entry: &filmcraft_isobmff::SampleEntry) -> Option<filmcraft_codecs::Result<Box<dyn filmcraft_codecs::VideoDecoder>>> {
    use filmcraft_codecs::video::{av1_factory, h264_factory, hevc_factory, prores_factory, vp9_factory};
    let factories: [fn(&filmcraft_isobmff::SampleEntry) -> Option<filmcraft_codecs::Result<Box<dyn filmcraft_codecs::VideoDecoder>>>; 5] =
        [h264_factory, hevc_factory, vp9_factory, av1_factory, prores_factory];
    factories.into_iter().find_map(|f| f(entry))
}

#[cfg(not(target_os = "linux"))]
fn main() {}
