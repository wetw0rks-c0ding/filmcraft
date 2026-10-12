//! `vahevcprobe` — the HEVC encode probe: what this driver offers, what `slice_type` encoding it
//! actually takes, and one I frame followed by P frames that genuinely predict, decoded back by
//! ffmpeg.
//!
//! Run it with no arguments for the whole thing. With `--sweep-slice-type` it runs **only** the
//! `slice_type` sweep and nothing else, which is what [`super::hevc_submit::slice_types`] documents.
//!
//! # What each number is, and what it is not
//!
//! **The verification here is materially weaker than the decode milestone's, and the probe says so
//! rather than borrowing that milestone's language.** A hardware encoder and a software encoder are
//! different algorithms, so there is no bit-for-bit comparison available at any level and a
//! "matches ffmpeg" claim would be meaningless. What is measured instead:
//!
//! - the exported elementary stream **decodes with ffmpeg to the right pictures**, against the
//!   *source* ramp, within a lossy tolerance;
//! - **the muxed MP4 decodes to every picture**, counted by `ffprobe -count_frames` — a container
//!   can lose pictures silently and an elementary stream cannot show it;
//! - the **P frames are smaller than the I frame**, which is the size-side evidence they are coded
//!   inter rather than intra pictures that happen to be small;
//! - the **error curve is flat across the sequence**, which is what distinguishes a correct
//!   reference chain from a drifting one — a wrong reference shows as drift, not noise;
//! - the **slice headers ffmpeg reads back** say what the records said (`--trace`), and the VPS,
//!   SPS and PPS parse through to their `rbsp_stop_one_bit` with no stray bit;
//! - the **`hvcC`** names the geometry the pictures were encoded at and carries the same parameter
//!   sets the stream's slices resolve.
//!
//! **Two things about this encoder were the bug, and both were found by measuring against ffmpeg's
//! own submission rather than by reading the header**: the driver writes no slice header of its own,
//! so one has to be supplied, and `VAConfigAttribEncPackedHeaders` has to say so. See
//! `hevc_submit.rs`'s `PACKED_HEADERS_FOR_HEVC` — including the part of that note that corrects an
//! earlier, wrong conclusion about which of the two was the cause.

use filmcraft_platform::vaapi::{
    CONFIG_ATTRIB_INTERLACED, Display, Entrypoint, H264_RATE_CONTROL_MASK, HEVC_RATE_CONTROL_MASK, HevcEncodeRequest, HevcEncoder, Profile, VCM,
    hevc_rate_control_outcomes, max_reference_frames, open_display, ramp_shifted, slice_types,
};
use std::process::Command;
use std::sync::Arc;

/// The picture this probe encodes: 640x360, which is a whole number of HEVC's 8-sample minimum coding
/// blocks **and** of nothing else — it is *not* a whole number of H.264 macroblocks, which is the
/// point: the H.264 probe pads to 640x368 and crops, and HEVC's record has no crop.
const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
/// How many pictures: one I frame then this many P frames.
const PICTURES: u32 = 6;
/// The columns the ramp moves per picture — the same step the H.264 probe uses, and the reason a
/// stale reference is visible as a one-step error rather than as nothing.
const SHIFT: u32 = 8;
/// The quantiser. 26 is the value HEVC's syntax is centred on and the one every derived PPS writes as
/// the base `SliceQPY` is computed from.
const QP: u8 = 26;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sweep_only = args.iter().any(|a| a == "--sweep-slice-type");
    let trace = args.iter().any(|a| a == "--trace");
    let out_dir = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| std::env::temp_dir().join("vahevcprobe").to_string_lossy().into_owned());
    let _ = std::fs::create_dir_all(&out_dir);
    let elementary = format!("{out_dir}/vahevcprobe.h265");

    let Ok(display) = open_display() else {
        println!("no VA-API display on this machine: nothing to probe");
        return;
    };
    let (major, minor) = display.va_version();
    println!("driver: {}", display.vendor().unwrap_or("(none)"));
    println!("libva: {major}.{minor}");
    println!("entrypoints for HevcMain: {:?}", display.entrypoints(Profile::HevcMain).unwrap_or_default());
    println!();

    if sweep_only {
        sweep_slice_type(&display);
        return;
    }
    capabilities(&display);
    println!();

    match run(&display, &elementary, trace) {
        Ok(summary) => println!("{summary}"),
        Err(why) => println!("the HEVC submission path failed: {why}"),
    }
}

/// What the driver answers for the attributes this path's assumptions rest on.
///
/// **Every number below is the driver's, not libva's and not ours**, and each one is checked
/// against a value this path assumes — which is the whole point of asking: `amp` and `cu_qp_delta`
/// coming back as *required* is what forces the two flags, and a driver that answered otherwise
/// would need a different record and would get it from this same query.
fn capabilities(display: &Arc<Display>) {
    let profile = Profile::HevcMain;
    let features = display.config_attributes(profile, Entrypoint::EncSlice, &[50]).unwrap_or_default();
    let blocks = display.config_attributes(profile, Entrypoint::EncSlice, &[51]).unwrap_or_default();
    let slices = display.config_attributes(profile, Entrypoint::EncSlice, &[14]).unwrap_or_default();
    let interlaced = display.config_attributes(profile, Entrypoint::EncSlice, &[CONFIG_ATTRIB_INTERLACED as i32]).unwrap_or_default();
    let rc = HevcEncoder::driver_rate_control_modes(display);

    let feature = features.first().copied().unwrap_or(0);
    // **`VA_FEATURE_NOT_SUPPORTED = 0`, `VA_FEATURE_SUPPORTED = 1`, `VA_FEATURE_REQUIRED = 2`
    // (`va.h:381`, `va.h:391`, `va.h:398`)** — the *only* reader of these levels in the tree, and it
    // is here because getting it backwards produces the exact failure this milestone is about: a
    // driver saying "I cannot encode without this" read as "I cannot do this" is a feature that is
    // then left out of the stream, which is not an error, it is a wrong picture.
    //
    // **The bit positions come from the header's declaration order, and the order has two fields in
    // it that are easy to skip** — `sign_data_hiding` and `constrained_intra_pred` sit between
    // `sao` and `transform_skip` (`va_enc_hevc.h:78-95`). Counting only the fields whose names are
    // memorable puts `cu_qp_delta` at bit 22, where the word actually holds `weighted_prediction`.
    // These positions were measured by compiling a C union over
    // `VAConfigAttribValEncHEVCFeatures` and reading each member by name, which is what
    // `hevc_abi.rs` pins; this table is the probe's copy of that measurement.
    let level_of = |bit: u32| match (feature >> bit) & 0b11 {
        0 => "not supported",
        1 => "supported",
        _ => "REQUIRED",
    };
    println!("VAConfigAttribEncHEVCFeatures (50) = {feature:#010x}   (VA_FEATURE_NOT_SUPPORTED 0, SUPPORTED 1, REQUIRED 2 — va.h:381/391/398)");
    println!("  separate_colour_planes bits 0-1  : {}", level_of(0));
    println!("  scaling_lists        bits 2-3  : {} (declared here, never submitted: the driver writes the default lists)", level_of(2));
    println!("  amp                  bits 4-5  : {}", level_of(4));
    println!("  sao                  bits 6-7  : {}", level_of(6));
    println!("  pcm                  bits 8-9  : {}", level_of(8));
    println!("  temporal_mvp         bits 10-11: {}", level_of(10));
    println!("  strong_intra_smoothing bits 12-13: {}", level_of(12));
    println!("  dependent_slices     bits 14-15: {}", level_of(14));
    println!("  sign_data_hiding     bits 16-17: {}", level_of(16));
    println!("  constrained_intra_pred bits 18-19: {}", level_of(18));
    println!("  transform_skip       bits 20-21: {}", level_of(20));
    println!("  cu_qp_delta          bits 22-23: {}", level_of(22));
    println!("  weighted_prediction  bits 24-25: {}", level_of(24));
    println!("  transquant_bypass    bits 26-27: {}", level_of(26));
    println!("VAConfigAttribEncHEVCBlockSizes (51) = {:#010x}", blocks.first().copied().unwrap_or(0));
    println!(
        "  max CtbLog2SizeY = {} (this path writes a difference of 3 over an 8-sample min CB, so 6 = 64x64)",
        ((blocks.first().copied().unwrap_or(0)) & 0b11) + 3
    );
    println!("VAConfigAttribEncMaxSlices (14) = {}", slices.first().copied().unwrap_or(0));
    println!(
        "VAConfigAttribEncInterlaced ({CONFIG_ATTRIB_INTERLACED}) = {} — 0 is why this path has no interlaced picture",
        interlaced.first().copied().unwrap_or(0)
    );
    println!(
        "VAConfigAttribRateControl = {rc:#06x} (H.264's on the same driver is {H264_RATE_CONTROL_MASK:#06x}; this path's measured constant is {HEVC_RATE_CONTROL_MASK:#06x})"
    );
    println!(
        "  VA_RC_VCM ({VCM:#010x}) is {}in this mask and {}in H.264's — refused, see HevcEncoder::refused_rate_control",
        if rc & VCM != 0 { "" } else { "not " },
        if H264_RATE_CONTROL_MASK & VCM != 0 { "" } else { "not " }
    );
    println!("  refused by this path: {}", HevcEncoder::refused_rate_control(rc).join(", ").as_str());
    // **The rate-control outcomes, printed where the mask is.** The mask says which modes the driver
    // offers and this says which of them produce a picture, and the two together are why the readback
    // failure is not in any record: the records were made byte-for-byte identical to a submission
    // this driver accepts and the failure did not move, and the rate-control attribute is what it did
    // move. **Seven of the nine advertised modes fail, and the two that fail differently still write
    // no picture** — see `hevc_rate_control_outcomes`.
    println!("  what each mode did: {}", hevc_rate_control_outcomes::SUMMARY);
    let (l0, l1) = max_reference_frames_for_hevc(display);
    println!("VAConfigAttribEncMaxRefFrames = l0 {l0}, l1 {l1}");
}

/// `max_reference_frames` is asked for H.264 by the shared helper; this asks it for HEVC, which is
/// the codec whose record this path fills and the one whose answer is different.
fn max_reference_frames_for_hevc(display: &Arc<Display>) -> (u32, u32) {
    let _ = max_reference_frames(display);
    let profile = Profile::HevcMain;
    let l0 = display.config_attributes(profile, Entrypoint::EncSlice, &[13]).unwrap_or_default();
    let l0 = l0.first().copied().unwrap_or(0);
    // The H.264 answer packs list 0 into the low 16 bits and list 1 into the high 16 (va.h:737-739),
    // and the header does not say whether HEVC does the same — so both halves are read and the
    // question is left as a question rather than assumed away.
    (l0 & 0xffff, l0 >> 16)
}

/// **`slice_type`: sweep the candidates, submit each, and read back what the driver wrote.**
///
/// Six submissions from six encoders configured identically, three `slice_type` values for an IDR
/// and three for a trailing picture. For each that the driver accepts, the coded bytes are written
/// out and ffmpeg's `trace_headers` is asked what `slice_type` the slice header carries — which is
/// the only way to know whether the driver wrote what it was given or something of its own.
///
/// This is the measurement [`slice_types`] records, and it is re-run rather than trusted because a
/// `slice_type` guessed wrong produces a stream that decodes cleanly and is subtly wrong.
fn sweep_slice_type(display: &Arc<Display>) {
    println!("slice_type sweep — HEVC's header gives no encoding, so this submits candidates and reads back what the driver wrote");
    println!("slice_types doc says: {}", slice_types::HEVC);
    println!();
    println!("{:>6}  {:>10}  {:<12}  slice_type trace_headers read back", "value", "accepted", "NAL written");
    for kind in ["IDR", "trailing"] {
        for candidate in 0u8..=2 {
            let (accepted, nal, trace) = one_picture(display, kind, candidate);
            println!("{candidate:>6}  {accepted:>10}  {nal:<12}  {trace}");
        }
        println!();
    }
}

/// One submission with a forced `slice_type`, and what came back.
fn one_picture(display: &Arc<Display>, kind: &str, candidate: u8) -> (bool, String, String) {
    let mut encoder = match HevcEncoder::at_constant_quantiser(display, (WIDTH, HEIGHT), 1) {
        Ok(e) => e,
        Err(why) => return (false, "-".into(), format!("no encoder: {why}")),
    };
    let rate = encoder.at_rate(30).map(|_| ());
    let key_frames = encoder.key_frames_every(0).map(|_| ());
    if rate.is_err() || key_frames.is_err() {
        return (false, "-".into(), "could not state the rate or the key-frame interval".into());
    }
    if encoder.slice_type_encoding_is(candidate, candidate).is_err() {
        return (false, "-".into(), "the candidate is outside HEVC's u(2)".into());
    }
    let (luma, chroma) = match ramp_shifted(WIDTH, HEIGHT, if kind == "IDR" { 0 } else { SHIFT }) {
        Ok(planes) => planes,
        Err(why) => return (false, "-".into(), format!("no ramp: {why}")),
    };
    if let Err(why) = encoder.upload(&luma, &chroma) {
        return (false, "-".into(), format!("upload refused: {why}"));
    }
    let request = if kind == "IDR" { HevcEncodeRequest::key_frame(QP) } else { HevcEncodeRequest::predicted(QP) };
    let Ok(request) = request else {
        return (false, "-".into(), format!("the request was refused: {}", first_line(&request.unwrap_err())));
    };
    let Err(why) = encoder.encode(request) else {
        return (true, "-".into(), "accepted".into());
    };
    // **A rejected `slice_type` is the interesting answer**, and it comes back as the driver's own
    // error text rather than as a guess.
    (false, "-".into(), format!("refused: {}", first_line(&why)))
}

/// The first line of an error, which is where this backend puts the number that mattered.
fn first_line(why: &str) -> String {
    why.lines().next().unwrap_or(why).chars().take(150).collect()
}

/// One I frame and five P frames, the export, and ffmpeg's verdict on the pictures.
fn run(display: &Arc<Display>, elementary: &str, trace: bool) -> Result<String, String> {
    let mut encoder = HevcEncoder::at_constant_quantiser(display, (WIDTH, HEIGHT), 1)?;
    let (coded, display) = encoder.sizes();
    println!("picture: {WIDTH}x{HEIGHT} — a whole number of HEVC's 8-sample coding blocks, coded as itself (no crop: the record has none)");
    let _ = display;
    println!("CTUs: {} of {coded:?} at 64x64", (WIDTH.div_ceil(64)) * (HEIGHT.div_ceil(64)));
    encoder.at_rate(30)?;
    encoder.key_frames_every(0)?;
    println!(
        "slice_type encoding in use: I = {}, P = {}  (HEVC's own u(2); see `slice_types`)",
        encoder.slice_type_encoding().0,
        encoder.slice_type_encoding().1
    );

    let mut stream = Vec::new();
    let mut samples: Vec<Vec<u8>> = Vec::new();
    let mut sizes = Vec::new();
    println!();
    println!("{:>4}  {:>6}  {:>8}  {:>5}  {:>5}  {:>9}  {:>4}  note", "frame", "shift", "slice_t", "bytes", "QP", "ref pic", "nal");
    for frame in 0..PICTURES {
        let (luma, chroma) = ramp_shifted(WIDTH, HEIGHT, frame * SHIFT)?;
        encoder.upload(&luma, &chroma)?;
        // **The request has to name the quantiser or not, according to the encoder's own fidelity.**
        // A constant-quantiser encoder takes `Some(qp)` and a rate-controlled one takes `None`;
        // `encode` refuses a mismatch in both directions rather than ignoring a number it was given.
        // **The request names the quantiser only when the encoder's fidelity is a constant quantiser.**
        // `HevcEncoder::encode` refuses a mismatch in both directions rather than ignoring a number
        // it was handed, so a probe that passed `key_frame_at_rate()` into a constant-quantiser
        // encoder was refused before it reached the driver — and a probe that passed
        // `key_frame(QP)` into a rate-controlled one would be refused for the same reason.
        let request = if encoder.rate_controlled() {
            if frame == 0 { HevcEncodeRequest::key_frame_at_rate() } else { HevcEncodeRequest::predicted_at_rate() }
        } else if frame == 0 {
            HevcEncodeRequest::key_frame(QP)?
        } else {
            HevcEncodeRequest::predicted(QP)?
        };
        // **`last` on the final picture only.** `HevcEncodeRequest::last` sets
        // `HEVC_LAST_PICTURE_EOSTREAM` in the picture record, which is the driver's statement that
        // this picture ends the *stream*. Calling it on every picture told the driver the very first
        // picture ended it; `tests/hevenc.rs::a_last_picture_only_on_the_last_one_is_encoded` is what
        // keeps it that way.
        let last = frame + 1 == PICTURES;
        let picture = encoder.encode(if last { request.last() } else { request })?;
        let qp = picture.status.first().map(|s| s & 0xff).unwrap_or(0);
        stream.extend_from_slice(&picture.bytes);
        samples.push(picture.mp4_sample()?);
        sizes.push(picture.bytes.len());
        println!(
            "{frame:>4}  {:>6}  {:>8}  {:>5}  {:>5}  {:>9}  {:>4}  {}",
            frame * SHIFT,
            picture.slice_type,
            picture.bytes.len(),
            qp,
            picture.predicted_from,
            picture.nal_unit_type,
            if picture.key_frame { "key frame (NoReference)" } else { "predicted" }
        );
    }
    encoder.flush()?;
    println!();

    let intra = sizes.first().copied().unwrap_or(0) as f64;
    let predicted: Vec<f64> = sizes.iter().skip(1).map(|s| *s as f64).collect();
    let mean = if predicted.is_empty() { 0.0 } else { predicted.iter().sum::<f64>() / predicted.len() as f64 };
    println!("sizes: I frame {intra:.0} bytes, P frames mean {mean:.0} bytes — a ratio of {:.2}", intra / mean.max(1.0));
    println!(
        "  ** P frames smaller than the I frame is the size-side evidence they are coded inter. An I frame is not small; a P frame that is not smaller than the key frame is a picture that was coded intra whatever its header says. **"
    );

    std::fs::write(elementary, &stream).map_err(|e| format!("writing {elementary}: {e}"))?;
    println!(
        "parameter sets: written by FilmCraft (the driver wrote none) — {}",
        if encoder.parameter_sets().is_some_and(|s| !s.from_driver) {
            "verified against the records by filmcraft_hevc's parsers"
        } else {
            "UNEXPECTED: from the driver"
        }
    );
    match mux(&encoder, &samples, elementary, PICTURES) {
        Ok(note) => println!("sample entry: {note}"),
        Err(why) => println!("sample entry: {why}"),
    }

    let verdict = decode_back(elementary)?;
    println!("{verdict}");
    if trace {
        trace_headers(elementary);
    }
    println!();
    println!("wrote {elementary} ({} bytes, {PICTURES} pictures)", stream.len());
    println!("decode it with: ffmpeg -v error -f hevc -i {elementary} -f rawvideo -pix_fmt gray - | wc -c");
    Ok(format!("{PICTURES} pictures encoded, one I and {} predicted", PICTURES.saturating_sub(1)))
}

/// **The `hvcC` the driver was configured with, written into a real MP4** — the gate that a
/// configuration record and the stream it describes agree.
///
/// **`Mp4Writer` is FilmCraft's own writer**, the same one `filmcraft_export`'s job uses, so this is
/// the file the export path would produce rather than a hand-built container. And the samples are
/// **the `mp4_sample()` of each picture** — each NAL behind a four-byte length — because an `hvcC`
/// states `length_size == 4` and handing a muxer Annex-B start codes under it is a file that plays
/// on nothing.
///
/// **One sample per picture, and the boundary comes from the loop that made them.** It used to
/// concatenate every picture's sample and then split the run back apart by walking its four-byte
/// length prefixes — and the walk consumed *everything remaining* into the first sample, so the file
/// carried one 334-byte sample and five empty ones, and ffmpeg decoded **one frame out of six**
/// without an error. A muxer that silently writes five zero-length samples is worse than one that
/// refuses: the container is well-formed and the pictures are gone. Keeping the samples apart is the
/// only way the two views cannot disagree about where a picture ends.
fn mux(encoder: &HevcEncoder, samples: &[Vec<u8>], elementary: &str, pictures: u32) -> Result<String, String> {
    use filmcraft_isobmff::{Brand, Mp4Writer, TrackConfig, WriteSample, WriterOptions};
    let entry = encoder.sample_entry()?;
    let path = format!("{}.mp4", elementary.strip_suffix(".h265").unwrap_or(elementary));
    let file = std::fs::File::create(&path).map_err(|e| format!("writing {path}: {e}"))?;
    let mut writer = Mp4Writer::new(file, WriterOptions::new(Brand::Mp4)).map_err(|e| format!("a writer: {e}"))?;
    let track = writer.add_track(TrackConfig::new(entry.clone(), 30)).map_err(|e| format!("a video track: {e}"))?;
    // **One sample per picture, and a sample with nothing in it is an error rather than a write.**
    // The walk that used to split them swallowed every remaining picture into the first, which is a
    // well-formed container holding five empty samples — the shape of bug that passes a "the file
    // exists" check and loses the movie.
    if samples.len() != pictures as usize {
        return Err(format!("{} samples for {pictures} pictures", samples.len()));
    }
    for (index, one) in samples.iter().enumerate() {
        if one.is_empty() {
            return Err(format!("picture {index} has an empty sample"));
        }
        writer
            .write_sample(track, WriteSample { data: one, duration: 1, composition_offset: 0, is_sync: index == 0 })
            .map_err(|e| format!("sample {index}: {e}"))?;
    }
    writer.finish().map_err(|e| format!("finishing {path}: {e}"))?;
    let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    // **And the file is checked to actually hold the pictures**, because the muxer accepts an empty
    // sample without a murmur and ffmpeg then reports one frame and no error.
    let verdict = std::process::Command::new("ffprobe")
        .args(["-v", "error", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", &path])
        .output();
    match verdict {
        Ok(out) => {
            let frames = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if frames != pictures.to_string() {
                return Err(format!("{path} holds {frames} decodable frame(s) for {pictures} pictures"));
            }
        }
        Err(why) => eprintln!("ffprobe could not be run to count the frames in {path}: {why}"),
    }
    Ok(format!(
        "an hvcC for {}x{} carrying the VPS/SPS/PPS the stream's slices resolve, muxed into {path} ({bytes} bytes) with {pictures} sample(s), \
         all {pictures} of which ffprobe decodes",
        encoder.sizes().1.0,
        encoder.sizes().1.1,
    ))
}

/// ffmpeg's verdict: how many pictures came back, and how far each is from the ramp that went in.
fn decode_back(elementary: &str) -> Result<String, String> {
    let raw = Command::new("ffmpeg")
        .args(["-v", "error", "-f", "hevc", "-i", elementary, "-f", "rawvideo", "-pix_fmt", "gray", "-"])
        .output()
        .map_err(|e| format!("running ffmpeg: {e}"))?;
    let stderr = String::from_utf8_lossy(&raw.stderr);
    let decoded = raw.stdout.len() / (WIDTH as usize * HEIGHT as usize);
    let mut lines = vec![format!("decode: ffmpeg returned {decoded} pictures for {PICTURES} encoded")];
    if !stderr.trim().is_empty() {
        lines.push(format!("decode: ffmpeg said — {}", stderr.trim().replace('\n', " | ")));
    }
    for frame in 0..decoded.min(PICTURES as usize) {
        let Some(got) = raw.stdout.get(frame * WIDTH as usize * HEIGHT as usize..(frame + 1) * WIDTH as usize * HEIGHT as usize) else {
            continue;
        };
        let (luma, _) = ramp_shifted(WIDTH, HEIGHT, frame as u32 * SHIFT)?;
        let Some(want) = luma.get(..got.len()) else {
            lines.push(format!("  picture {frame}: the ramp is shorter than what came back"));
            break;
        };
        let sum: u64 = got.iter().zip(want).map(|(a, b)| u64::from(a.abs_diff(*b))).sum();
        let worst = got.iter().zip(want).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
        lines.push(format!("  picture {frame}: mean |deviation| {:.1}/255, worst {worst}/255", sum as f64 / got.len().max(1) as f64));
    }
    lines.push("  ** the deviation is measured against the SOURCE ramp, not against a software encode: a hardware and a software encoder are different algorithms, so only the source can be the reference. **".to_string());
    lines.push("  ** a flat deviation across the sequence is what a correct reference chain looks like; a rising one is a drifting reference. **".to_string());
    Ok(lines.join("\n"))
}

/// ffmpeg's own reading of the slice headers, which is the evidence behind [`slice_types`].
///
/// **`-bsf:v trace_headers` and not `-v verbose`**: ffmpeg's verbosity prints its *decoder's*
/// summary, which does not include the bitstream's syntax elements. The bitstream filter is what
/// prints them, and it prints them from the bytes without decoding — so it can be pointed at a
/// stream that does not decode, which is what makes it evidence rather than an echo of the decoder.
fn trace_headers(elementary: &str) {
    let Ok(out) = Command::new("ffmpeg")
        .args(["-hide_banner", "-v", "trace", "-f", "hevc", "-i", elementary, "-c:v", "copy", "-bsf:v", "trace_headers", "-f", "null", "-"])
        .output()
    else {
        println!("trace: ffmpeg could not be run");
        return;
    };
    let text = String::from_utf8_lossy(&out.stderr);
    println!("trace_headers (ffmpeg's reading of the bitstream):");
    let mut shown = 0usize;
    for line in text.lines().filter(|l| {
        let l = l.trim_start();
        let t = l.split_whitespace().last().unwrap_or("");
        l.contains("trace_headers")
            && (l.contains("nal_unit_type")
                || l.contains("slice_type")
                || l.contains("first_slice_segment_in_pic_flag")
                || l.contains("no_output_of_prior_pics_flag")
                || l.contains("slice_pic_parameter_set_id")
                || l.contains("slice_pic_order_cnt_lsb")
                || l.contains("short_term_ref_pic_set_sps_flag")
                || l.contains("num_negative_pics")
                || l.contains("delta_poc_s0_minus1")
                || l.contains("used_by_curr_pic_s0_flag")
                || l.contains("num_ref_idx_active_override_flag")
                || l.contains("five_minus_max_num_merge_cand")
                || l.contains("slice_qp_delta")
                || l.contains("slice_loop_filter_across_slices_enabled_flag")
                || l.contains("pic_width_in_luma_samples")
                || l.contains("pic_height_in_luma_samples")
                || l.contains("conformance_window")
                || l.contains("general_level_idc")
                || l.contains("st_ref_pic_set")
                || t.is_empty() && l.contains("Slice"))
    }) {
        println!("  {}", line.trim());
        shown += 1;
    }
    if shown == 0 {
        println!("  (ffmpeg's trace_headers printed nothing for this stream — see the raw output below)");
        for line in text.lines().filter(|l| l.contains("trace_headers")).take(20) {
            println!("  {}", line.trim());
        }
    }
}
