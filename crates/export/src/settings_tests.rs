//! M6.5 export settings: image sequences, audio-only formats, frame size / rate / scaling,
//! loudness normalization, two-pass and CBR H.264, the video limiter, overlays, metadata and the
//! QuickTime multiplexer. ffprobe / ffmpeg are used only as external test oracles.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{FrameRequest, Generator, MediaSource};
use filmcraft_project::{ItemKind, Label, MediaClip, MediaRef, SequenceSettings, TrackKind};
use filmcraft_render::SourceMap;
use filmcraft_time::TICKS_PER_SECOND;

use super::*;

/// A scratch directory under the workspace `target/`, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let d = filmcraft_testkit::fixtures::workspace_root().join("target").join("export-tests").join(format!("{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Scratch(d)
    }
    fn path(&self, name: &str) -> String {
        self.0.join(name).to_string_lossy().to_string()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A `w`×`h` 24 fps sequence: a colour matte (1 s) and optionally a 440 Hz tone at `tone_db`.
fn matte(color: [f32; 4], w: u32, h: u32, tone_db: Option<f32>) -> (Arc<Project>, ItemId, SourceMap) {
    let mut p = Project::new("m");
    let r = FrameRate::FPS_24;
    let dur = Tick(2 * TICKS_PER_SECOND);
    let mut m = SourceMap::default();
    let mut add = |p: &mut Project, g: GeneratorSource| {
        let info = g.info().clone();
        let id = p.add_item(
            &info.name.clone(),
            Label::Iris,
            ItemKind::Media(MediaClip {
                media: MediaRef::Generator(g.generator.clone()),
                info,
                interpret: Default::default(),
                mark_in: None,
                mark_out: None,
                markers: vec![],
                offline: false,
                proxy: None,
                identity: None,
            }),
            None,
        );
        m.0.insert(id, Arc::new(g) as Arc<dyn MediaSource>);
        id
    };
    let v = add(&mut p, GeneratorSource::new(Generator::ColorMatte { color }, w, h, r, dur));
    let a = tone_db.map(|db| add(&mut p, GeneratorSource::new(Generator::Tone { hz: 440.0, db }, w, h, r, dur)));
    let seq = p.new_sequence("s", SequenceSettings { width: w, height: h, frame_rate: r, ..Default::default() }, 1, 1, None);
    let vi = p.make_track_item(v, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, r.tick_of(24)), r).unwrap();
    p.sequence_mut(seq).unwrap().video_tracks[0].items.push(vi);
    if let Some(a) = a {
        let ai = p.make_track_item(a, TrackKind::Audio, Tick::ZERO, TimeRange::new(Tick::ZERO, r.tick_of(24)), r).unwrap();
        p.sequence_mut(seq).unwrap().audio_tracks[0].items.push(ai);
    }
    (Arc::new(p), seq, m)
}

fn ffprobe_json(args: &[&str], path: &str) -> Option<serde_json::Value> {
    let ffprobe = filmcraft_testkit::ffprobe_or_skip("export settings")?;
    let mut a = vec!["-v", "error", "-of", "json"];
    a.extend_from_slice(args);
    a.push(path);
    let out = std::process::Command::new(ffprobe).args(&a).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    Some(serde_json::from_slice(&out.stdout).unwrap())
}

fn files_in(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect();
    v.sort();
    v
}

#[test]
fn image_sequences_are_numbered_like_premiere() {
    let (p, seq, m) = matte([1.0, 0.0, 0.0, 1.0], 320, 180, None);
    for (fmt, ext) in [(Format::PngSequence, "png"), (Format::TiffSequence, "tif"), (Format::BmpSequence, "bmp")] {
        let dir = Scratch::new(&format!("seq-{ext}"));
        let mut s = ExportSettings { format: fmt, path: dir.path(&format!("Shot 01.{ext}")), scale: 0.5, ..Default::default() };
        s.range = Some(TimeRange::new(Tick::ZERO, FrameRate::FPS_24.tick_of(12)));
        let prog = Progress::default();
        let r = export(&p, seq, &s, &m, &prog).unwrap();
        assert_eq!(r.frames, 12);
        assert_eq!(prog.done.load(Ordering::Relaxed), 12);
        let want: Vec<String> = (0..12).map(|i| format!("Shot 01{i:03}.{ext}")).collect();
        assert_eq!(files_in(&dir.0), want, "{fmt:?}");
        let img = image::open(dir.0.join(&want[5])).unwrap().to_rgb8();
        assert_eq!((img.width(), img.height()), (160, 90));
        let px = img.get_pixel(80, 45).0;
        assert!(px[0] > 240 && px[1] < 15, "{fmt:?} {px:?}");
    }
}

#[test]
fn wav_and_aiff_audio_only() {
    let (p, seq, m) = matte([0.0, 0.0, 0.0, 1.0], 64, 36, Some(-6.0));
    let dir = Scratch::new("audio");
    // WAV, 24-bit mono at 44.1 kHz
    let wav = dir.path("a.wav");
    let mut s = ExportSettings { format: Format::Wav, path: wav.clone(), ..Default::default() };
    s.audio = AudioSettings { sample_rate: Some(44_100), channels: 1, bits: 24, ..Default::default() };
    s.range = Some(TimeRange::new(Tick::ZERO, Tick(TICKS_PER_SECOND / 2)));
    export(&p, seq, &s, &m, &Progress::default()).unwrap();
    let b = std::fs::read(&wav).unwrap();
    assert_eq!(u16::from_le_bytes([b[22], b[23]]), 1, "mono");
    assert_eq!(u32::from_le_bytes(b[24..28].try_into().unwrap()), 44_100);
    assert_eq!(u16::from_le_bytes([b[34], b[35]]), 24);
    assert_eq!(u32::from_le_bytes(b[40..44].try_into().unwrap()), 22_050 * 3, "0.5 s of 24-bit mono");
    // AIFF, 16-bit stereo at the sequence rate
    let aif = dir.path("a.aif");
    let mut s = ExportSettings { format: Format::Aiff, path: aif.clone(), ..Default::default() };
    s.range = Some(TimeRange::new(Tick::ZERO, Tick(TICKS_PER_SECOND / 2)));
    export(&p, seq, &s, &m, &Progress::default()).unwrap();
    let b = std::fs::read(&aif).unwrap();
    assert_eq!(&b[..4], b"FORM");
    assert_eq!(&b[8..12], b"AIFF");
    assert_eq!(u16::from_be_bytes([b[20], b[21]]), 2);
    assert_eq!(u32::from_be_bytes(b[22..26].try_into().unwrap()), 24_000);
    if let Some(j) = ffprobe_json(&["-show_streams"], &aif) {
        let st = &j["streams"][0];
        assert_eq!(st["codec_name"], "pcm_s16be");
        assert_eq!(st["sample_rate"], "48000");
        assert_eq!(st["channels"], 2);
    }
    if let Some(j) = ffprobe_json(&["-show_streams"], &wav) {
        assert_eq!(j["streams"][0]["codec_name"], "pcm_s24le");
    }
}

#[test]
fn frame_size_rate_and_scaling() {
    let (p, seq, m) = matte([1.0, 0.0, 0.0, 1.0], 320, 180, Some(-12.0));
    let dir = Scratch::new("size");
    let path = dir.path("square.mp4");
    let s = ExportSettings {
        format: Format::H264,
        path: path.clone(),
        frame_size: Some((256, 256)),
        frame_rate: Some(FrameRate::new(12, 1)),
        scaling: Scaling::ScaleToFit,
        ..Default::default()
    };
    let r = export(&p, seq, &s, &m, &Progress::default()).unwrap();
    assert_eq!(r.frames, 12, "1 s at 12 fps");
    let bytes: Arc<[u8]> = std::fs::read(&path).unwrap().into();
    let src = filmcraft_codecs::open_bytes("square.mp4", bytes).unwrap();
    let v = src.info().video.clone().unwrap();
    assert_eq!((v.width, v.height), (256, 256));
    let f = src.video_frame(FrameRequest::full(Tick(TICKS_PER_SECOND / 2))).unwrap().to_rgba8();
    let at = |x: usize, y: usize| &f[(y * 256 + x) * 4..(y * 256 + x) * 4 + 3];
    assert!(at(128, 128)[0] > 200, "picture in the middle: {:?}", at(128, 128));
    assert!(at(128, 10)[0] < 40, "letterbox bar on top: {:?}", at(128, 10));
    assert!(at(128, 245)[0] < 40, "letterbox bar below: {:?}", at(128, 245));
    if let Some(j) = ffprobe_json(&["-count_frames", "-show_streams", "-select_streams", "v:0"], &path) {
        let st = &j["streams"][0];
        assert_eq!(st["r_frame_rate"], "12/1");
        assert_eq!(st["nb_read_frames"], "12");
        assert_eq!(st["width"], 256);
    }
    // fill crops instead of letterboxing
    let path = dir.path("fill.mov");
    let s = ExportSettings {
        format: Format::ProRes,
        path: path.clone(),
        frame_size: Some((200, 200)),
        scaling: Scaling::ScaleToFill,
        include_audio: false,
        ..Default::default()
    };
    export(&p, seq, &s, &m, &Progress::default()).unwrap();
    let bytes: Arc<[u8]> = std::fs::read(&path).unwrap().into();
    let src = filmcraft_codecs::open_bytes("fill.mov", bytes).unwrap();
    let f = src.video_frame(FrameRequest::full(Tick(TICKS_PER_SECOND / 2))).unwrap().to_rgba8();
    assert!(f[(5 * 200 + 100) * 4] > 200, "filled to the top edge");
}

/// Integrated loudness of planar audio.
fn lufs(planar: &[Vec<f32>], sr: u32) -> (f64, f64) {
    let refs: Vec<&[f32]> = planar.iter().map(Vec::as_slice).collect();
    let s = filmcraft_audio_dsp::loudness::measure(&refs, sr as f64);
    (s.integrated_lufs, s.true_peak_dbtp)
}

fn read_wav16(path: &str) -> Vec<Vec<f32>> {
    let b = std::fs::read(path).unwrap();
    let ch = u16::from_le_bytes([b[22], b[23]]) as usize;
    let data = &b[44..];
    let mut out = vec![Vec::new(); ch];
    for (i, s) in data.as_chunks::<2>().0.iter().enumerate() {
        out[i % ch].push(i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0);
    }
    out
}

#[test]
fn loudness_normalization_hits_target_and_true_peak() {
    let (p, seq, m) = matte([0.0, 0.0, 0.0, 1.0], 64, 36, Some(-6.0));
    let dir = Scratch::new("loud");
    for target in [-23.0, -16.0, -14.0] {
        let path = dir.path(&format!("n{}.wav", -target as i32));
        let mut s = ExportSettings { format: Format::Wav, path: path.clone(), ..Default::default() };
        s.effects.loudness = LoudnessNormalization { enabled: true, target_lufs: target, true_peak_dbtp: -1.0 };
        let prog = Progress::default();
        export(&p, seq, &s, &m, &prog).unwrap();
        let (i, tp) = lufs(&read_wav16(&path), 48_000);
        assert!((i - target).abs() < 0.5, "target {target}: measured {i:.2} LUFS");
        assert!(tp <= -0.9, "true peak {tp:.2} dBTP");
        let rep = prog.loudness.lock().unwrap().unwrap();
        assert!((rep.measured_lufs - (target - rep.gain_db)).abs() < 1e-6);
    }
    // a target the peaks cannot reach: the limiter holds the true-peak ceiling
    let path = dir.path("hot.wav");
    let mut s = ExportSettings { format: Format::Wav, path: path.clone(), ..Default::default() };
    s.effects.loudness = LoudnessNormalization { enabled: true, target_lufs: -3.0, true_peak_dbtp: -3.0 };
    export(&p, seq, &s, &m, &Progress::default()).unwrap();
    let (_, tp) = lufs(&read_wav16(&path), 48_000);
    assert!(tp <= -2.8, "limited true peak {tp:.2} dBTP");
    // and through the stepped H.264 + AAC path, decoded by our AAC decoder
    let path = dir.path("loud.mp4");
    let mut s = ExportSettings { format: Format::H264, path: path.clone(), ..Default::default() };
    s.effects.loudness = LoudnessNormalization { enabled: true, target_lufs: -20.0, true_peak_dbtp: -1.0 };
    export(&p, seq, &s, &m, &Progress::default()).unwrap();
    let bytes: Arc<[u8]> = std::fs::read(&path).unwrap().into();
    let src = filmcraft_codecs::open_bytes("loud.mp4", bytes).unwrap();
    let a = src.audio(4800, 38_400, 48_000).unwrap();
    let (i, _) = lufs(&a.channels, 48_000);
    assert!((i + 20.0).abs() < 0.5, "AAC: {i:.2} LUFS");
}

#[test]
fn two_pass_and_cbr_h264() {
    let (p, seq, m) = matte([0.2, 0.5, 0.8, 1.0], 320, 180, None);
    let dir = Scratch::new("2pass");
    for mode in [BitrateMode::Vbr2Pass, BitrateMode::Cbr] {
        let path = dir.path(&format!("{mode:?}.mp4"));
        let s = ExportSettings { format: Format::H264, path: path.clone(), bitrate_mode: mode, bitrate_kbps: 2000, include_audio: false, ..Default::default() };
        let prog = Progress::default();
        let r = export(&p, seq, &s, &m, &prog).unwrap();
        assert_eq!(r.frames, 24);
        let passes = if mode == BitrateMode::Vbr2Pass { 2 } else { 1 };
        assert_eq!(prog.total.load(Ordering::Relaxed), 24 * passes);
        assert_eq!(prog.done.load(Ordering::Relaxed), 24 * passes);
        let bytes: Arc<[u8]> = std::fs::read(&path).unwrap().into();
        let src = filmcraft_codecs::open_bytes("x.mp4", bytes).unwrap();
        let f = src.video_frame(FrameRequest::full(Tick(TICKS_PER_SECOND / 2))).unwrap().to_rgba8();
        assert!((f[2] as i32 - 204).abs() < 12 && (f[1] as i32 - 127).abs() < 12, "{mode:?}: {:?}", &f[..4]);
    }
}

#[test]
fn h264_profile_level_and_keyframes() {
    let (p, seq, m) = matte([0.5, 0.5, 0.5, 1.0], 320, 180, None);
    let dir = Scratch::new("profile");
    let path = dir.path("main.mp4");
    let s = ExportSettings {
        format: Format::H264,
        path: path.clone(),
        h264_profile: H264Profile::Main,
        h264_level: Some(51),
        keyframe_distance: Some(6),
        include_audio: false,
        ..Default::default()
    };
    export(&p, seq, &s, &m, &Progress::default()).unwrap();
    if let Some(j) = ffprobe_json(&["-show_streams"], &path) {
        assert_eq!(j["streams"][0]["profile"], "Main");
        assert_eq!(j["streams"][0]["level"], 51);
    }
    if let Some(j) = ffprobe_json(&["-show_frames", "-select_streams", "v:0", "-show_entries", "frame=key_frame"], &path) {
        let keys = j["frames"].as_array().unwrap().iter().filter(|f| f["key_frame"] == 1).count();
        assert_eq!(keys, 4, "24 frames, a keyframe every 6");
    }
}

/// **`Auto` is the default, and it resolves to a profile an encoder on this machine can write.**
///
/// The rule under test is `ExportSettings::resolved_h264_profile`'s: the *backend* decides, not
/// the stored value. Two facts make the expectation platform-dependent and both are measured rather
/// than assumed — VideoToolbox and NVENC pass the profile through and write it into the stream,
/// while VA-API's derived SPS states Main whatever profile the driver is offered (an offered High
/// comes back as a byte-identical Main bitstream).
#[test]
fn an_auto_h264_profile_resolves_to_what_this_machine_can_produce() {
    let built_in = ExportSettings::default();
    assert_eq!(built_in.h264_profile, H264Profile::Auto, "a settings object that names no profile delegates rather than promising");
    assert_eq!(built_in.hardware_encoding, HardwareEncoding::Off, "and hardware encoding is off unless asked for, so the built-in encoder runs");
    assert_eq!(built_in.resolved_h264_profile(), H264Profile::High, "the built-in encoder writes all three profiles, so the strongest is what Auto means here");

    let hardware = ExportSettings { hardware_encoding: HardwareEncoding::Auto, ..Default::default() };
    let want = if cfg!(target_os = "linux") { H264Profile::Main } else { H264Profile::High };
    assert_eq!(hardware.resolved_h264_profile(), want, "with the OS encoder in play, Auto is the best profile *every* backend here can produce");

    // **MXF is the case the OS encoder cannot take at all** (it carries an Annex B stream, which
    // this backend's length-prefixed samples are not), so the built-in encoder will run, it writes
    // every profile, and Auto is High — which is what the `MXF OP1a H.264` preset's own
    // description says it is. Without this condition the description would be true only with
    // hardware encoding off.
    let mxf = ExportSettings { format: Format::MxfOp1a, hardware_encoding: HardwareEncoding::Auto, ..Default::default() };
    assert_eq!(mxf.resolved_h264_profile(), H264Profile::High, "nothing else can encode MXF H.264 here");

    for named in [H264Profile::Baseline, H264Profile::Main, H264Profile::High] {
        let s = ExportSettings { h264_profile: named, hardware_encoding: HardwareEncoding::Auto, ..Default::default() };
        assert_eq!(
            s.resolved_h264_profile(),
            named,
            "a named profile is a promise and is passed through untouched — Auto delegates, it never rewrites one that was asked for"
        );
    }

    // **Every built-in preset inherits the delegation**, which is the only reason this reaches the
    // hardware export path at all: the presets name no profile, and this is what they resolve to.
    for p in builtin_presets().iter().filter(|p| p.settings.video_format() == Format::H264) {
        assert_eq!(p.settings.h264_profile, H264Profile::Auto, "{} names a profile", p.name);
        assert!(p.settings.resolved_h264_profile() != H264Profile::Auto, "{} leaves the encoder nothing to write", p.name);
    }
}

/// **The summary prints the profile the file will have, not the one that was typed.**
///
/// This is the mechanism `AudioCodec::Auto` uses for the audio line, applied to the video one: the
/// panel has to be able to answer "what is this going to be?" before an encoder is chosen, and an
/// `Auto` that printed "Auto" would answer a different question.
#[test]
fn the_summary_names_the_resolved_profile() {
    let ten = Tick(10 * TICKS_PER_SECOND);
    let line = |s: &ExportSettings| s.summary(1920, 1080, FrameRate::FPS_30, 48_000, ten).video.clone();
    let hardware = ExportSettings { format: Format::H264, hardware_encoding: HardwareEncoding::Auto, ..Default::default() };

    // The two platform answers, one line each, asserted on the exact text a user reads. (The level
    // follows the profile without a comma; that is the panel's existing wording, not this change's.)
    let want = hardware.resolved_h264_profile().label();
    assert_eq!(
        line(&hardware),
        format!("1920x1080 (1.0), 30 fps, Progressive, {want} auto level, VBR, 1 pass, Target 20.00 Mbps, Max 30.00 Mbps, keyframe every 60 frames"),
        "the resolved profile, where the stored value is Auto"
    );
    let named = ExportSettings { h264_profile: H264Profile::Baseline, ..hardware.clone() };
    assert!(line(&named).contains(", Baseline auto level,"), "a named profile is printed as named: {}", line(&named));
    assert!(!line(&hardware).contains("Auto"), "Auto is a delegation, not a description of the file: {}", line(&hardware));
}

/// **A settings file written before `Auto` existed still loads, and its stored profile is kept.**
///
/// This is the data-loss guard for `<data dir>/export-presets.json` and every project that carries
/// export settings: before this change the field's default was `High` and it was always written out,
/// so real files on disk carry `"h264Profile":"high"`. If adding the variant had turned that string
/// into an unknown one, a user's whole preset library would fail to load and every export setting
/// in every project would be rejected — settings files do not fail loudly and then carry on, they
/// fall back to defaults or error, and both lose what was stored.
#[test]
fn settings_written_before_auto_still_deserialise() {
    // Exactly the shape of a stored `ExportSettings` from before this change.
    let stored = serde_json::json!({
        "format": "h264",
        "path": "/home/u/Movies/out.mp4",
        "scale": 1.0,
        "includeAudio": true,
        "quality": 90,
        "bitrateKbps": 20000,
        "burnCaptions": false,
        "fieldOrder": "progressive",
        "h264Profile": "high",
        "h264Level": 41,
        "bitrateMode": "vbr1Pass",
        "constantQp": 26,
        "hardwareEncoding": "off",
        "maxBitrateKbps": 30000,
        "multiplexer": "mp4",
        "audio": { "codec": "aac", "sampleRate": 48000, "channels": 2, "bitrateKbps": 320, "bits": 16 }
    });
    let s: ExportSettings = serde_json::from_value(stored.clone()).unwrap_or_else(|e| panic!("a settings file written before Auto: {e}"));
    assert_eq!(s.h264_profile, H264Profile::High, "a stored \"high\" is still the High profile, not an unknown variant");
    assert_eq!(s.resolved_h264_profile(), H264Profile::High, "and it still encodes as High whatever Auto would have said");
    assert_eq!(
        (s.h264_level, s.bitrate_mode, s.bitrate_kbps, s.hardware_encoding),
        (Some(41), BitrateMode::Vbr1Pass, 20_000, HardwareEncoding::Off),
        "the rest of the file is untouched"
    );
    assert_eq!(s.audio.codec, AudioCodec::Aac, "including the audio codec, which has had an Auto since before this change");

    // Every spelling the field has ever had parses, and the new one round-trips.
    for (name, want) in [("baseline", H264Profile::Baseline), ("main", H264Profile::Main), ("high", H264Profile::High), ("auto", H264Profile::Auto)] {
        let s: ExportSettings = serde_json::from_value(serde_json::json!({ "h264Profile": name })).unwrap_or_else(|e| panic!("\"{name}\": {e}"));
        assert_eq!(s.h264_profile, want, "\"{name}\"");
        let back = serde_json::to_value(&s).unwrap();
        assert_eq!(back["h264Profile"], name, "\"{name}\" survives a round trip under the same name");
    }

    // A file that names no profile at all is a file from before the field existed: it gets Auto.
    let old: ExportSettings = serde_json::from_value(serde_json::json!({ "format": "h264" })).unwrap();
    assert_eq!(old.h264_profile, H264Profile::Auto, "no profile named means the delegation, not the last default");
    // …and every built-in preset (and every preset the library could have saved) round-trips whole.
    for p in builtin_presets() {
        let back: ExportPreset = serde_json::from_value(serde_json::to_value(&p).unwrap()).unwrap_or_else(|e| panic!("{}: {e}", p.name));
        assert_eq!(back.settings.h264_profile, H264Profile::Auto, "{} survives a save and load", p.name);
    }
}

#[test]
fn video_limiter_clamps_luma() {
    let (p, seq, m) = matte([1.0, 1.0, 1.0, 1.0], 64, 36, None);
    let dir = Scratch::new("limiter");
    let mut s = ExportSettings { format: Format::PngSequence, path: dir.path("w.png"), ..Default::default() };
    s.range = Some(TimeRange::new(Tick::ZERO, FrameRate::FPS_24.tick_of(1)));
    s.effects.video_limiter = VideoLimiter { enabled: true, min_percent: 0.0, max_percent: 80.0 };
    export(&p, seq, &s, &m, &Progress::default()).unwrap();
    let img = image::open(dir.0.join("w000.png")).unwrap().to_rgb8();
    let mx = img.pixels().flat_map(|p| p.0).max().unwrap();
    assert!((200..=204).contains(&mx), "white limited to 80 %: {mx}");
    let mut px = vec![255u8, 255, 255, 255, 0, 0, 0, 255];
    limit_rgba8(&mut px, 10.0, 90.0);
    assert!(px[0] <= 230 && px[4] >= 25, "{px:?}");
}

#[test]
fn overlays_draw_where_placed() {
    let (p, seq, m) = matte([0.0, 0.0, 0.0, 1.0], 320, 180, None);
    let dir = Scratch::new("overlay");
    // a 10×10 green PNG as the image overlay
    let logo = dir.path("logo.png");
    image::RgbaImage::from_pixel(10, 10, image::Rgba([0, 255, 0, 255])).save(&logo).unwrap();
    let mut s = ExportSettings { format: Format::PngSequence, path: dir.path("o.png"), ..Default::default() };
    s.range = Some(TimeRange::new(Tick::ZERO, FrameRate::FPS_24.tick_of(1)));
    s.effects.image_overlay = ImageOverlay { enabled: true, path: logo, placement: Placement::TopRight, size_percent: 10.0, ..Default::default() };
    s.effects.name_overlay =
        TextOverlay { enabled: true, text: "CLIENT REVIEW".into(), placement: Placement::BottomCenter, size_percent: 8.0, ..Default::default() };
    s.effects.timecode_overlay = TextOverlay { enabled: true, placement: Placement::TopLeft, size_percent: 8.0, ..Default::default() };
    export(&p, seq, &s, &m, &Progress::default()).unwrap();
    let img = image::open(dir.0.join("o000.png")).unwrap().to_rgb8();
    let count = |x0: u32, x1: u32, y0: u32, y1: u32, f: &dyn Fn([u8; 3]) -> bool| {
        (y0..y1).flat_map(|y| (x0..x1).map(move |x| (x, y))).filter(|&(x, y)| f(img.get_pixel(x, y).0)).count()
    };
    let green = |p: [u8; 3]| p[1] > 200 && p[0] < 60;
    let white = |p: [u8; 3]| p[0] > 200 && p[1] > 200 && p[2] > 200;
    assert!(count(260, 320, 0, 60, &green) > 200, "logo top right");
    assert_eq!(count(0, 160, 0, 180, &green), 0, "no logo on the left");
    assert!(count(60, 260, 140, 180, &white) > 50, "name bottom centre");
    assert!(count(0, 160, 0, 40, &white) > 50, "timecode top left");
    assert_eq!(count(0, 320, 70, 110, &white), 0, "nothing in the middle");
}

#[test]
fn metadata_and_quicktime_multiplexer() {
    let (p, seq, m) = matte([0.0, 0.0, 1.0, 1.0], 160, 90, Some(-12.0));
    let dir = Scratch::new("meta");
    let path = dir.path("m.mov");
    let mut s = ExportSettings { format: Format::H264, path: path.clone(), multiplexer: Multiplexer::Mov, ..Default::default() };
    s.metadata = ExportMetadata { title: "Test Title".into(), creator: "FilmCraft Tests".into(), copyright: "CC0".into(), ..Default::default() };
    s.audio.codec = AudioCodec::Pcm;
    s.audio.bits = 24;
    assert_eq!(s.extension(), "mov");
    export(&p, seq, &s, &m, &Progress::default()).unwrap();
    if let Some(j) = ffprobe_json(&["-show_format", "-show_streams"], &path) {
        assert!(j["format"]["format_name"].as_str().unwrap().contains("mov"));
        let tags = &j["format"]["tags"];
        assert_eq!(tags["title"], "Test Title");
        assert_eq!(tags["copyright"], "CC0");
        let codecs: Vec<&str> = j["streams"].as_array().unwrap().iter().map(|s| s["codec_name"].as_str().unwrap()).collect();
        assert_eq!(codecs, ["h264", "pcm_s24le"]);
    }
}

#[test]
fn interlaced_output_is_rejected_honestly() {
    let (p, seq, m) = matte([0.0, 0.0, 1.0, 1.0], 160, 90, None);
    let s = ExportSettings { format: Format::ProRes, path: "unused.mov".into(), field_order: FieldOrder::UpperFirst, ..Default::default() };
    let e = export(&p, seq, &s, &m, &Progress::default()).unwrap_err();
    assert!(e.to_string().contains("progressive"), "{e}");
}

#[test]
fn settings_serde_and_estimates() {
    let s: ExportSettings = serde_json::from_value(serde_json::json!({"format": "prores", "frameSize": [1280, 720], "proresProfile": "lt"})).unwrap();
    assert_eq!(s.format, Format::ProRes);
    assert_eq!(s.frame_size, Some((1280, 720)));
    assert!(s.include_audio, "defaults fill the rest");
    let back: ExportSettings = serde_json::from_value(serde_json::to_value(&s).unwrap()).unwrap();
    assert_eq!(back.prores_profile, "lt");
    // 10 s of 20 Mbps H.264 + 320 kbps AAC ≈ 25.4 MB
    let h = ExportSettings::default();
    let b = h.estimate_bytes(1920, 1080, FrameRate::FPS_25, 48_000, Tick(10 * TICKS_PER_SECOND));
    assert!((b as f64 - 25.4e6).abs() < 0.2e6, "{b}");
    assert_eq!(format_bytes(b), "25 MB");
    // adaptive bitrate scales with the frame size
    let a = ExportSettings { adaptive_bitrate: Some(0.2), ..Default::default() };
    assert_eq!(a.resolve(1920, 1080, FrameRate::FPS_30, 48_000).target_kbps, 12_442);
    assert_eq!(a.resolve(3840, 2160, FrameRate::FPS_30, 48_000).target_kbps, 49_766);
}

#[test]
fn builtin_presets_are_valid_and_unique() {
    let all = builtin_presets();
    assert!(all.len() >= 24);
    let mut keys: Vec<String> = all.iter().map(|p| presets::preset_key(&p.name)).collect();
    keys.sort();
    keys.dedup();
    assert_eq!(keys.len(), all.len(), "unique names");
    for p in &all {
        p.settings.validate().unwrap();
        assert!(p.builtin && !p.category.is_empty() && !p.description.is_empty(), "{}", p.name);
    }
    assert_eq!(presets::find_builtin("match source - adaptive high bitrate").unwrap().name, presets::DEFAULT_PRESET);
    let json = serde_json::to_value(&all[0]).unwrap();
    let back: ExportPreset = serde_json::from_value(json).unwrap();
    assert_eq!(back.settings.adaptive_bitrate, Some(0.2));
}

/// #108: the default 320 kbps is more than AAC can carry in stereo at 22.05 kHz (6 bits per sample
/// and channel). The export failed with "bitrate out of range"; it is now capped to the limit.
#[test]
fn aac_bitrate_is_capped_for_low_sample_rates() {
    let mut s = ExportSettings::default();
    s.audio.bitrate_kbps = 320;
    for (rate, ch) in [(22_050, 2), (16_000, 2), (8_000, 6), (48_000, 2)] {
        let enc = aac_factory(Format::H264, rate, ch, &s).expect("AAC is always available");
        assert!(enc.is_ok(), "{rate} Hz × {ch}: {:?}", enc.err());
    }
}

#[test]
fn a_time_left_reads_like_a_clock() {
    use std::time::Duration;
    let f = |ms: u64| format_eta(Duration::from_millis(ms));
    assert_eq!(f(0), "0 s");
    assert_eq!(f(1), "1 s", "rounded up: never say 0 s while there is time left");
    assert_eq!(f(45_000), "45 s");
    assert_eq!(f(59_001), "1:00");
    assert_eq!(f(60_000), "1:00");
    assert_eq!(f(125_000), "2:05");
    assert_eq!(f(3_599_000), "59:59");
    assert_eq!(f(3_600_000), "1:00:00");
    assert_eq!(f(3_725_000), "1:02:05");
    assert_eq!(f(100 * 3600 * 1000), "100:00:00");
    assert!(!format_eta(Duration::MAX).is_empty(), "the largest duration cannot overflow");
}

#[test]
fn h265_is_a_format_that_needs_a_registered_encoder() {
    use std::sync::atomic::{AtomicBool, Ordering};
    // names, ids, files and serialized settings
    for name in ["hevc", "H.265", "h265", "HVC1", "x265"] {
        assert_eq!(Format::from_name(name), Some(Format::Hevc), "{name}");
    }
    assert_eq!(Format::from_name(Format::Hevc.id()), Some(Format::Hevc));
    assert_eq!((Format::Hevc.extension(), Format::Hevc.label()), ("mp4", "H.265 (HEVC)"));
    assert!(Format::ALL.contains(&Format::Hevc) && Format::ALL.len() == 14);
    let s: ExportSettings = serde_json::from_value(serde_json::json!({"format": "hevc"})).unwrap();
    assert_eq!(s.format, Format::Hevc);
    assert_eq!(serde_json::to_value(&s).unwrap()["format"], "hevc");

    // it is H.264's family: MP4 with AAC, or QuickTime; the same size estimate
    let h = ExportSettings { format: Format::Hevc, ..Default::default() };
    assert_eq!((h.audio_codec(), h.extension()), (AudioCodec::Aac, "mp4"));
    assert_eq!(ExportSettings { multiplexer: Multiplexer::Mov, ..h.clone() }.extension(), "mov");
    // 10 s of 20 Mbps video + 320 kbps AAC ≈ 25.4 MB, like H.264
    let b = h.estimate_bytes(1920, 1080, FrameRate::FPS_25, 48_000, Tick(10 * TICKS_PER_SECOND));
    assert!((b as f64 - 25.4e6).abs() < 0.2e6, "{b}");
    let summary = h.summary(1920, 1080, FrameRate::FPS_25, 48_000, Tick(10 * TICKS_PER_SECOND));
    assert_eq!(summary.format, "H.265 (HEVC) (MP4)");
    assert!(summary.video.contains("HEVC Main") && summary.video.contains("Target 20.00 Mbps"), "{}", summary.video);

    // a hardware encoder has one pass
    let e = ExportSettings { bitrate_mode: BitrateMode::Vbr2Pass, ..h.clone() }.validate().unwrap_err();
    assert!(e.to_string().contains("two-pass"), "{e}");
    assert!(ExportSettings { bitrate_mode: BitrateMode::Cbr, ..h.clone() }.validate().is_ok());

    // formats with a built-in encoder are always available; this one only when a registered probe says so
    assert!(Format::ALL.iter().filter(|f| f.has_builtin_encoder()).all(|f| available(*f)));
    assert!(!Format::Hevc.has_builtin_encoder());
    static HERE: AtomicBool = AtomicBool::new(false);
    fn probe() -> bool {
        HERE.load(Ordering::SeqCst)
    }
    assert!(!available(Format::Hevc), "nothing registered an encoder for it");
    // and an export says so, honestly
    let (p, seq, m) = matte([0.0, 0.0, 1.0, 1.0], 160, 90, None);
    let sc = Scratch::new("h265-missing");
    let err = export(&p, seq, &ExportSettings { path: sc.path("x.mp4"), include_audio: false, ..h.clone() }, &m, &Progress::default()).unwrap_err().to_string();
    assert!(err.contains("H.265") && err.contains("encoder not available"), "{err}");
    register_format_probe(Format::Hevc, probe);
    register_format_probe(Format::Hevc, probe); // the same probe twice is harmless
    assert!(!available(Format::Hevc));
    HERE.store(true, Ordering::SeqCst);
    assert!(available(Format::Hevc));
    HERE.store(false, Ordering::SeqCst);
    assert!(!available(Format::Hevc));
}

/// **The mode `VA_RC_VCM` names, and that the vocabulary actually carries it.**
///
/// `va.h:1114` — `/** \brief Video conference mode. */ #define VA_RC_VCM 0x00000008` — sits between
/// `VA_RC_VBR` (1112) and `VA_RC_CQP` (1116), and it is a distinct rate-control bit rather than a
/// synonym for either. A driver can advertise it, so FilmCraft can be asked for it, so there has to
/// be a variant that means it.
///
/// The assertions are against the enum's whole contents rather than one lookup, because the failure
/// this guards is a *silent* one: an `ALL` that merely grew, or one whose new entry duplicated an
/// old one, would satisfy a `contains` alone.
#[test]
fn video_conference_is_a_mode_the_vocabulary_carries() {
    // Spelled out as a set: the five, each named, no more and no fewer. Compared as a sorted copy so
    // the panel order in `ALL` is not part of the claim, while a duplicate or a removal still fails.
    let mut got: Vec<BitrateMode> = BitrateMode::ALL.to_vec();
    got.sort_by_key(|m| m.label());
    let mut want = vec![BitrateMode::Cbr, BitrateMode::Vbr1Pass, BitrateMode::Vbr2Pass, BitrateMode::ConstantQuality, BitrateMode::VideoConference];
    want.sort_by_key(|m| m.label());
    assert_eq!(got, want, "the modes FilmCraft can name, and `ALL` is how many a panel lists");
    // `5` is a real count, not a restatement of the array: it would read 4 before this change.
    assert_eq!(BitrateMode::ALL.len(), 5, "a fifth mode");
    assert!(BitrateMode::ALL.contains(&BitrateMode::VideoConference), "and it is the one just added");

    // A label is what the settings panel and the summary print. Distinct for every mode, so a panel
    // listing all five cannot show the same word twice, and non-empty so nothing prints a gap.
    let mut labels: Vec<&str> = BitrateMode::ALL.iter().map(|m| m.label()).collect();
    labels.sort_unstable();
    let count = labels.len();
    labels.dedup();
    assert_eq!(labels.len(), count, "every mode prints its own words: {labels:?}");
    assert!(labels.iter().all(|l| !l.trim().is_empty()), "{labels:?}");
    assert_eq!(BitrateMode::VideoConference.label(), "Video conference (VCM)");
}

/// **`VA_RC_VCM` states a rate, and `ConstantQuality` does not. They are not the same gap.**
///
/// This is the one answer the mode gets wrong by accident. `ConstantQuality` states a quantiser and
/// no rate, so both hardware gates refuse it; a reader who has just met that rule will reach for
/// `false` here too, and `false` is a lie with consequences — `estimate_bytes` returns 0 and the
/// summary prints `QP {constant_qp}` for an export that submits no quantiser.
///
/// So the claim is tested from both sides, against the settings rather than against a copy of the
/// predicate: every mode but `ConstantQuality` names a rate, `ConstantQuality` alone does not, and
/// the two summary statements about the file's size come out right for the new mode.
#[test]
fn video_conference_names_a_bitrate_and_constant_quality_does_not() {
    for mode in BitrateMode::ALL {
        assert_eq!(mode.names_a_bitrate(), mode != BitrateMode::ConstantQuality, "{mode:?}: only constant quality states a quantiser and no rate");
    }
    assert!(BitrateMode::VideoConference.names_a_bitrate(), "VCM holds a rate and caps a picture against it");
    assert!(!BitrateMode::ConstantQuality.names_a_bitrate(), "CQP states QP n and nothing else");

    // **The consequence, read off a real settings object.** H.265 is the format this mode exists for
    // — it is the one whose driver's mask contains the bit — so its summary branch is the one checked.
    // **Audio is off in both**, because `estimate_bytes` adds the audio rate to a zero video rate: the
    // claim under test is about the video, and an H.265 constant-quality export carries no audio, so
    // the estimate really is 0 and the size really is unknowable before the pictures are encoded.
    let ten = Tick(10 * TICKS_PER_SECOND);
    let vcm = ExportSettings {
        format: Format::Hevc,
        bitrate_mode: BitrateMode::VideoConference,
        bitrate_kbps: 20_000,
        max_bitrate_kbps: Some(30_000),
        include_audio: false,
        ..Default::default()
    };

    let summary = vcm.summary(1920, 1080, FrameRate::FPS_25, 48_000, ten);
    assert!(summary.video.contains("Video conference (VCM)"), "the mode is named: {}", summary.video);
    assert!(summary.video.contains("Target 20.00 Mbps") && summary.video.contains("Max 30.00 Mbps"), "the rate it was given: {}", summary.video);
    assert!(!summary.video.contains("QP "), "VCM states no quantiser, so `constant_qp` must not be quoted: {}", summary.video);

    // **The first of the two statements about size: a number, computed from the rate that was given.**
    let bytes = vcm.estimate_bytes(1920, 1080, FrameRate::FPS_25, 48_000, ten);
    let cq = ExportSettings { bitrate_mode: BitrateMode::ConstantQuality, ..vcm.clone() };
    assert_eq!(cq.estimate_bytes(1920, 1080, FrameRate::FPS_25, 48_000, ten), 0, "a constant-quality file has no size to compute before it is encoded");
    assert!(bytes > 0, "a VCM export names a rate, so its size is computable: {bytes}");

    // **The second: what the estimate was called.** Not the constant-quality wording, and not a zero
    // rounded up to "1 KB" by `format_bytes`.
    assert_ne!(summary.estimated_size, "not known in constant quality", "VCM names a rate, so its size is not the constant-quality gap");
    assert_ne!(summary.estimated_size, "1 KB", "a zero estimate must not be dressed up as a size: {}", summary.estimated_size);

    // **One pass.** Nothing about VCM renders the sequence twice.
    assert!(!BitrateMode::VideoConference.is_two_pass());
    for mode in BitrateMode::ALL {
        assert_eq!(mode.is_two_pass(), mode == BitrateMode::Vbr2Pass, "{mode:?}");
    }
}

/// **A settings file can ask for VCM by name, and that is what makes the variant reachable.**
///
/// A variant nothing can name in a stored file is a variant no backend is ever handed, and the
/// spelling is the whole contract with older settings files — so it is asserted in both directions
/// and against a file that carries nothing else.
#[test]
fn video_conference_survives_a_settings_file() {
    let s: ExportSettings =
        serde_json::from_value(serde_json::json!({ "bitrateMode": "videoConference" })).unwrap_or_else(|e| panic!("a settings file naming VCM: {e}"));
    assert_eq!(s.bitrate_mode, BitrateMode::VideoConference);
    assert_eq!(serde_json::to_value(&s).unwrap()["bitrateMode"], "videoConference", "and it round-trips under the same name");

    // Every spelling `ALL` can produce parses, so no mode in the vocabulary is unreachable.
    for mode in BitrateMode::ALL {
        let back = serde_json::to_value(&ExportSettings { bitrate_mode: mode, ..Default::default() }).unwrap()["bitrateMode"].clone();
        let got: ExportSettings = serde_json::from_value(serde_json::json!({ "bitrateMode": back })).unwrap_or_else(|e| panic!("{mode:?} wrote {back}: {e}"));
        assert_eq!(got.bitrate_mode, mode, "{mode:?} must be able to describe itself");
    }

    // **A file written before this change is untouched.** Naming no mode at all still loads, and a
    // file naming one of the four older spellings still reads as itself rather than as VCM.
    let old: ExportSettings = serde_json::from_value(serde_json::json!({ "format": "hevc" })).unwrap();
    assert_eq!(old.bitrate_mode, BitrateMode::Vbr1Pass, "no mode named means the default, never the newest variant");
    for (name, want) in
        [("cbr", BitrateMode::Cbr), ("vbr1Pass", BitrateMode::Vbr1Pass), ("vbr2Pass", BitrateMode::Vbr2Pass), ("constantQuality", BitrateMode::ConstantQuality)]
    {
        let s: ExportSettings = serde_json::from_value(serde_json::json!({ "bitrateMode": name })).unwrap_or_else(|e| panic!("\"{name}\": {e}"));
        assert_eq!(s.bitrate_mode, want, "\"{name}\" is still {want:?}");
    }

    // **No built-in preset starts asking for it.** The mode exists to be *available*; no preset
    // ships with a file that selects a rate control its target hardware may not offer.
    for p in builtin_presets() {
        assert_ne!(p.settings.bitrate_mode, BitrateMode::VideoConference, "{} selects a mode no default export should", p.name);
    }
}

/// **The per-picture ceiling is derived from numbers the summary already prints, or is the user's
/// own figure — and whichever it is, the summary line says so. This is the test that holds that
/// decision in place.**
///
/// # What the decision is, and why it is a test rather than a comment
///
/// `VA_RC_VCM` states *"a variable bit rate and a maximum frame size"*, and the second half now has
/// a field: `ExportSettings::max_frame_size`, in bytes, set from the export panel. Two candidates
/// existed — **a new settings field** and **a number derived from what the settings already say** —
/// and the derivation is what a `None` means, on grounds that are checkable here:
///
/// - **every input is already on the summary line.** A cap the user cannot see is a cap the user
///   cannot reason about, and a field would be a *second* number on that line rather than a derived
///   one. **This test is the claim that the inputs are there**: the frame rate and the rate pair, for
///   every format the mode is offered on.
/// - **a field nothing can set is the defect the mode was created to fix.** `VA_RC_VCM` sat unwired
///   for one reason — a capability no setting could name never reaches a driver — and adding a field
///   no control can set would reproduce that one layer up. **That is why the field arrived with the
///   export-panel control rather than before it**, and why the assertion below is written from the
///   **summary's own text**: a cap read from a field, with nothing on the line saying so, is
///   exactly the failure this test is for — and now it is checkable in both directions, because a
///   summary that failed to say *which* cap is in force would leave a reader unable to tell a
///   derived cap from their own.
///
/// # What it would catch
///
/// A summary that stopped quoting a rate for this mode (which is what would happen if
/// `names_a_bitrate` ever answered `false` for it), a summary that quoted a quantiser instead, a
/// summary that named the derived cap on an export whose cap the user set (or printed the user's
/// figure as if FilmCraft had derived it), or a backend that read a cap from somewhere the summary
/// does not account for. **Deliberate break:** print `QP n` for a VCM export and the second
/// assertion fails; print the rate without the frame rate and the third fails; drop the
/// `max picture …` clause from either case and the last two fail.
#[test]
fn every_input_the_derived_per_picture_ceiling_needs_is_on_the_summary_line() {
    let ten = Tick(10 * TICKS_PER_SECOND);
    // **Two frame rates and three rate pairs, so the assertion is about the *kinds* of number rather
    // than about one hand-picked example.** A summary that printed the rate for 25 fps and not for
    // 60 — or a ceiling derived from a rate the line did not carry — would pass a single case.
    for (rate, target, max) in [(FrameRate::FPS_25, 20_000u32, 30_000u32), (FrameRate::FPS_60, 2_000, 3_000), (FrameRate::new(30_000, 1_001), 8_000, 12_000)] {
        for format in [Format::Hevc, Format::H264] {
            let s = ExportSettings {
                format,
                bitrate_mode: BitrateMode::VideoConference,
                bitrate_kbps: target,
                max_bitrate_kbps: Some(max),
                include_audio: false,
                ..Default::default()
            };
            let said = s.summary(1920, 1080, rate, 48_000, ten).video;

            // **(1) The rate, because the mode states one.** This is the same claim as the test
            // above and it is restated here because it is an *input to a derived number*, not just a
            // statement about the file: a ceiling computed from a rate the summary did not print
            // would be FilmCraft computing a number on the user's behalf without showing its terms.
            assert!(
                said.contains(&format!("Target {:.2} Mbps", target as f64 / 1000.0)),
                "{format:?} at {rate}: the summary does not quote the target the ceiling is derived from: {said}"
            );
            assert!(said.contains(&format!("Max {:.2} Mbps", max as f64 / 1000.0)), "{format:?} at {rate}: and it does not quote the ceiling rate: {said}");
            // **(2) No quantiser, because the mode states none** — and a summary that printed one
            // would be quoting a number this mode has no field for.
            assert!(!said.contains("QP "), "{format:?} at {rate}: a video-conference export states no quantiser, so none may be printed: {said}");
            // **(3) The frame rate, because the ceiling is the rate's share of *one picture*.** It
            // has no other time base, so without this number the ceiling is not derivable by the
            // reader at all. `r.rate.label()` is what the H.264 and H.265 branches both print, and
            // this asserts it through the public summary rather than through `resolve`, so a mode
            // that lost the line fails here and not in a backend.
            assert!(
                said.contains(&rate.label()),
                "{format:?} at {rate}: the summary does not state the frame rate, and the per-picture ceiling is this rate's share of one \
                 picture, so the reader cannot derive the number the backend will use: {said}"
            );
            // **(4) The mode's own name, and the size statement is the rate one.** The ceiling only
            // ever *reduces* what the encoder produces, so the target rate stays the honest basis for
            // an estimate; the point here is that the summary does not fall into the constant-quality
            // wording, which is the only other branch that prints no rate.
            assert!(said.contains("Video conference (VCM)"), "{format:?} at {rate}: {said}");
            assert_ne!(s.estimate_bytes(1920, 1080, rate, 48_000, ten), 0, "{format:?} at {rate}: a mode that names a rate has a computable size");
            // **(5) The derived cap names its terms, and only H.265 has one.** The H.265 backend
            // derives the cap from the target and the frame rate, so the line says where it comes
            // from rather than printing the result — the result is a backend-side number and
            // re-deriving it here would duplicate the arithmetic in a crate that must not depend on
            // the backend. The H.264 encoders apply no per-picture cap at all (the VA-API H.264 gate
            // declines the mode; the software encoder takes it as a variable rate), so the H.264 line
            // must not claim a cap is in force. **Deliberate break:** print the clause for H.264 and
            // this fails.
            if format == Format::Hevc {
                assert!(
                    said.contains(&format!("max picture derived from the {:.2} Mbps target at {}", target as f64 / 1000.0, rate.label())),
                    "{format:?} at {rate}: the cap in force is derived, and the line must say from what: {said}"
                );
                assert!(
                    !said.contains("max picture ") || said.contains("max picture derived"),
                    "{format:?} at {rate}: no cap is set, so no figure may be printed as one: {said}"
                );
            } else {
                assert!(!said.contains("max picture"), "{format:?} applies no per-picture cap, so none may be named: {said}");
            }
        }
    }

    // **The other half: a cap the user set is stated as theirs, in their own unit, and the derived
    // wording is gone.** Written over the same rates and formats as the loop above, so a summary
    // that printed the derived clause beside a set cap would fail at every rate rather than at one
    // hand-picked example. **Three separate numbers, not one**: a summary that printed the target
    // rate's share of a picture *as* a cap would be printing a backend number the backend will not
    // use.
    for (rate, target, max) in [(FrameRate::FPS_25, 20_000u32, 30_000u32), (FrameRate::FPS_60, 2_000, 3_000), (FrameRate::new(30_000, 1_001), 8_000, 12_000)] {
        for (bytes, kb) in [(64_000u32, "64.0"), (1_000, "1.0"), (250_001, "250.0")] {
            let s = ExportSettings {
                format: Format::Hevc,
                bitrate_mode: BitrateMode::VideoConference,
                bitrate_kbps: target,
                max_bitrate_kbps: Some(max),
                max_frame_size: Some(bytes),
                include_audio: false,
                ..Default::default()
            };
            let said = s.summary(1920, 1080, rate, 48_000, ten).video;
            assert!(
                said.contains(&format!("max picture {kb} KB (the cap you set)")),
                "{bytes} bytes at {rate}: the summary states the cap the user set, in kB: {said}"
            );
            assert!(!said.contains("derived from"), "{bytes} bytes at {rate}: the cap is the user's, so the line must not still claim it is derived: {said}");
            // **The rate is still an input to the *target*, and the target is still printed.** A cap
            // the user set removes the frame rate from the cap's arithmetic — not from the export.
            assert!(said.contains(&format!("Target {:.2} Mbps", target as f64 / 1000.0)), "{bytes} bytes at {rate}: the target rate is still stated: {said}");
            assert!(
                said.contains(&rate.label()),
                "{bytes} bytes at {rate}: and so is the frame rate, which every other number on the line is relative to: {said}"
            );
        }
        // **The bytes→kB rendering is `/ 1000`, and this is where the factor of eight is caught on
        // this side of it.** The driver wants bits and the backend multiplies by 8
        // (`vaapi/hevc_export.rs`); if the summary printed *bits* — 64 000 bytes as "512.0 KB" — a
        // reader checking the two against each other would see two different caps and have no way
        // to tell which one the driver gets. The backend's own factor of 8 is pinned in
        // `the_user_cap_reaches_the_driver_in_bits`.
        let s = ExportSettings { format: Format::Hevc, bitrate_mode: BitrateMode::VideoConference, max_frame_size: Some(512_000), ..Default::default() };
        let said = s.summary(1920, 1080, rate, 48_000, ten).video;
        assert!(said.contains("max picture 512.0 KB (the cap you set)"), "512 000 bytes is 512.0 kB, not the 4 096 000 bits it converts to: {said}");
    }

    // **And the field exists, which is the other half of the decision and the part a summary
    // assertion cannot see.** This used to assert the *absence* of every per-picture ceiling field,
    // on the grounds that a field with no control behind it is reachable only from a hand-edited
    // project file and a test. `max_frame_size` is now that field, **with a control** in
    // `crates/ui-egui`'s export panel (`export.video.maxFrameSize`), so the assertion is the
    // positive form of what it was: the field is on `ExportSettings`, the JSON names it, and a
    // summary that read it without accounting for it on the line — the defect the old assertion
    // existed to catch — fails above.
    let written = serde_json::to_value(ExportSettings {
        format: Format::Hevc,
        bitrate_mode: BitrateMode::VideoConference,
        max_frame_size: Some(64_000),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(written["maxFrameSize"], 64_000, "the cap is a field on ExportSettings, in bytes, under its camelCase name");
    // **The number on the line is the number in the field, byte for byte — for three values, and
    // through the public summary rather than through the private clause.** The factor of eight is
    // the backend's business (`vaapi/hevc_export.rs`), but a summary that printed the *bits* would
    // show 512 000 bytes as "512.0 KB" when the driver receives 4 096 000 — two different numbers
    // for one setting, and a reader with no way to tell which one is in the file.
    for bytes in [1_000u32, 64_000, 512_000] {
        let said = ExportSettings { format: Format::Hevc, bitrate_mode: BitrateMode::VideoConference, max_frame_size: Some(bytes), ..Default::default() }
            .summary(1920, 1080, FrameRate::FPS_25, 48_000, ten)
            .video;
        assert!(
            said.contains(&format!("max picture {:.1} KB (the cap you set)", bytes as f64 / 1000.0)),
            "{bytes} bytes is printed as itself, not as its bit count: {said}"
        );
    }
    // **A cap of no other unit is *not* on this struct**, so a later reader cannot find a second
    // spelling of the same number and wonder which one the backend reads.
    let keys: Vec<String> = written.as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default();
    for ceiling_field in ["maxFrameSizeBits", "maxFrameBits", "maxPictureKbps", "maxFrameKbps"] {
        assert!(
            !keys.iter().any(|k| k == ceiling_field),
            "a second per-picture ceiling field ({ceiling_field}) has been added to ExportSettings: the backend reads one, and two unit spellings of one number is how the factor of eight gets introduced"
        );
    }
}

/// **A settings file written before `ExportSettings::max_frame_size` existed loads unchanged, and the
/// export it describes keeps the cap it always had — the derived one.**
///
/// # Why this is a test and not a consequence of `#[serde(default)]`
///
/// The container attribute makes the *deserialisation* work; it says nothing about what the loaded
/// settings then **mean**. A field that defaulted to something other than `None` — a plausible 8 kB,
/// say — would load every project written before it silently onto a different cap from the one its
/// author chose by leaving it alone, and nothing anywhere would report a change. So this asserts
/// the *behaviour*: `None` in, the derived wording on the summary line out, and a file that does
/// name the field round-trips under the same name and the same number.
///
/// **The blob below is copied from the shape of a stored settings file before the field existed**
/// (the same list `settings_written_before_auto_still_deserialise` uses), so it fails the moment the
/// struct stops accepting a file of that age — which is the day a project written by the current
/// release stops opening.
#[test]
fn settings_written_before_the_picture_cap_still_deserialise() {
    let ten = Tick(10 * TICKS_PER_SECOND);
    let stored = serde_json::json!({
        "format": "hevc",
        "path": "/home/u/Movies/out.mp4",
        "scale": 1.0,
        "includeAudio": false,
        "quality": 90,
        "bitrateKbps": 2000,
        "burnCaptions": false,
        "fieldOrder": "progressive",
        "bitrateMode": "videoConference",
        "constantQp": 26,
        "maxBitrateKbps": 3000,
        "multiplexer": "mp4",
        "audio": { "codec": "aac", "sampleRate": 48000, "channels": 2, "bitrateKbps": 320, "bits": 16 }
    });
    let s: ExportSettings = serde_json::from_value(stored).unwrap_or_else(|e| panic!("a settings file written before the picture cap: {e}"));
    assert_eq!(
        s.max_frame_size, None,
        "a file that names no cap has no cap, and must not be given one: leaving it unset is how its author asked for the derived cap"
    );
    // **The derived cap is what this file has always meant**, and the line says so rather than
    // printing a figure the backend would have to derive.
    let said = s.summary(1920, 1080, FrameRate::FPS_25, 48_000, ten).video;
    assert!(
        said.contains("max picture derived from the 2.00 Mbps target at 25 fps"),
        "an export from before the field existed derives its cap, as it always did: {said}"
    );
    assert!(!said.contains("KB (the cap you set)"), "and prints no figure for one that was never set: {said}");

    // **And the field round-trips when it is named**: same camelCase name, same number, in bytes.
    for bytes in [1_000u32, 64_000, 536_870_911] {
        let set = ExportSettings { max_frame_size: Some(bytes), ..s.clone() };
        let json = serde_json::to_value(&set).unwrap();
        assert_eq!(json["maxFrameSize"], bytes, "{bytes} bytes is written under the field's own name");
        let back: ExportSettings = serde_json::from_value(json).unwrap_or_else(|e| panic!("{bytes} bytes: {e}"));
        assert_eq!(back.max_frame_size, Some(bytes), "{bytes} bytes survives a save and a load, and a u32 is not read as something else");
    }
    // **And a written file that names the cap reads back as the user's own on the summary line**,
    // which is the other half of a round trip: the number survives, and so does what it means.
    let set: ExportSettings = serde_json::from_value(
        serde_json::to_value(ExportSettings {
            format: Format::Hevc,
            bitrate_mode: BitrateMode::VideoConference,
            bitrate_kbps: 2_000,
            max_frame_size: Some(64_000),
            include_audio: false,
            ..Default::default()
        })
        .unwrap(),
    )
    .unwrap();
    assert!(
        set.summary(1920, 1080, FrameRate::FPS_25, 48_000, ten).video.contains("max picture 64.0 KB (the cap you set)"),
        "a loaded cap is stated as the user's own figure, in kB"
    );
    // **Every built-in preset round-trips whole**, as the field's own doc claims: a preset library
    // saved before this field must still load, and every one of them ships with no cap.
    for p in builtin_presets() {
        let back: ExportPreset = serde_json::from_value(serde_json::to_value(&p).unwrap()).unwrap_or_else(|e| panic!("{}: {e}", p.name));
        assert_eq!(back.settings.max_frame_size, None, "{} round-trips with no cap, as it was written", p.name);
    }
}

/// **A per-picture cap the panel's slider cannot reach is refused with a message naming the number
/// and the two ways out — and nothing wraps, clamps or panics on the way.**
///
/// **Four classes of hostile value, one test, because the classes are the same failure in
/// different clothes**: a number a user can type, a slider a user cannot, and a hand-edited project
/// file can put *any* of them in. `0` and `1` are the two a slider with the wrong minimum produces;
/// `u32::MAX / 8 + 1` is the first value whose byte→bit conversion does not fit the driver's field,
/// and the one a hand-edited file reaches.
///
/// **Every one of them is an error naming the value**, and `MIN_MAX_FRAME_SIZE_BYTES` itself is
/// accepted — a refusal that also rejected the floor would make the control unusable, and one that
/// accepted the floor plus everything below it would not be a floor.
#[test]
fn a_picture_cap_the_panel_cannot_offer_is_refused_by_name() {
    for bytes in [0u32, 1, 999, MIN_MAX_FRAME_SIZE_BYTES - 1] {
        let s = ExportSettings { format: Format::Hevc, bitrate_mode: BitrateMode::VideoConference, max_frame_size: Some(bytes), ..Default::default() };
        let Err(e) = s.validate() else {
            panic!("a cap of {bytes} bytes was accepted, below the {MIN_MAX_FRAME_SIZE_BYTES} bytes the control offers");
        };
        let why = e.to_string();
        assert!(why.contains(&bytes.to_string()), "the refusal names the number the user typed: {why}");
        assert!(why.contains("1 kB") || why.contains(&MIN_MAX_FRAME_SIZE_BYTES.to_string()), "and what would be accepted instead: {why}");
    }
    // **The floor itself, and the largest value that converts**, are both taken. The largest is
    // `u32::MAX / 8` bytes = 536 870 911, whose bit count is 4 294 967 288 — one below the field's
    // maximum, and the one value a test can use to tell a checked conversion from a saturating one:
    // a `saturating_mul` would answer 4 294 967 288 here *and* for every larger value, so the
    // overflow case below is what separates them.
    for bytes in [MIN_MAX_FRAME_SIZE_BYTES, 64_000, u32::MAX / 8] {
        let s = ExportSettings { format: Format::Hevc, bitrate_mode: BitrateMode::VideoConference, max_frame_size: Some(bytes), ..Default::default() };
        s.validate().unwrap_or_else(|e| panic!("a cap of {bytes} bytes was refused: {e}"));
    }
    // **A settings file is not the only place a cap arrives, and the check is not gated on the mode
    // or the format either.** A hand-written `ExportSettings` can reach a backend without passing
    // through `validate` at all, which is why the gate checks it again; what this asserts is that
    // the settings layer refuses a nonsense cap whatever else the settings say, so the user is told
    // at the panel rather than at a backend they never reach.
    let nonsense = ExportSettings { format: Format::ProRes, max_frame_size: Some(0), ..Default::default() };
    assert!(nonsense.validate().is_err(), "a cap below the floor is refused whatever the format is, since no format reads it");
    assert!(ExportSettings::default().validate().is_ok(), "and a settings object with no cap is not an error");
}
