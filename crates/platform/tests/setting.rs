//! Settings ▸ Playback ▸ Hardware decoding through the decoder registry and the media stack:
//! Off gives the software decoder (and no hardware frames), Auto the hardware one where the
//! system has it. Its own test binary: the switch is process-wide.

mod common;

use common::*;
use filmcraft_media::FrameRequest;

/// The Hardware decoding switch is process-wide, so the tests take turns with it.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn off_uses_the_software_decoder() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let _turn = serial();
    let Some(path) = named(&ff, "hevc_main.mp4") else { return };
    let s = read_stream(&path);
    let available = filmcraft_platform::register();
    assert_eq!(filmcraft_platform::register(), available, "registering twice is harmless");

    filmcraft_codecs::hw::set_hardware_decoding(false);
    let d = filmcraft_codecs::make_video_decoder(&s.entry).unwrap();
    assert_eq!(d.name(), "FilmCraft HEVC");
    let bytes: std::sync::Arc<[u8]> = std::fs::read(&path).unwrap().into();
    let (g0, h0) = (filmcraft_codecs::gop_stats(), filmcraft_codecs::hw::hw_stats());
    let src = filmcraft_codecs::open_bytes("hevc_main.mp4", bytes.clone()).unwrap();
    let rate = src.info().frame_rate();
    for i in 0..30 {
        src.video_frame(FrameRequest::full(rate.tick_of(i))).unwrap();
    }
    assert!(filmcraft_codecs::gop_stats().frames > g0.frames, "frames counted");
    assert_eq!(filmcraft_codecs::hw::hw_stats().frames, h0.frames, "Off: no hardware frames");

    filmcraft_codecs::hw::set_hardware_decoding(true);
    let d = filmcraft_codecs::make_video_decoder(&s.entry).unwrap();
    let hw = filmcraft_platform::hardware_decoder_for(&s.entry);
    if !hw {
        eprintln!("SKIPPED (Auto half): no hardware decoder ({available:?})");
        assert_eq!(d.name(), "FilmCraft HEVC");
        return;
    }
    // The backend `register()` put in front of our decoders and the name its decoder reports for
    // this stream, one arm per system that has one: a system with no backend returned above.
    let (backend, hw_name) = if cfg!(target_os = "macos") {
        ("VideoToolbox", "VideoToolbox HEVC")
    } else if cfg!(target_os = "windows") {
        ("Media Foundation", "Media Foundation HEVC")
    } else {
        ("VA-API", "VA-API HEVC")
    };
    assert_eq!(d.name(), hw_name);
    assert_eq!(filmcraft_codecs::hw::hw_backend(), Some(backend), "perf.stats reports the backend");
    assert!(filmcraft_platform::registered(), "register() put the factory in the registry");
    let h1 = filmcraft_codecs::hw::hw_stats();
    let src = filmcraft_codecs::open_bytes("hevc_main.mp4", bytes).unwrap();
    for i in 0..30 {
        src.video_frame(FrameRequest::full(rate.tick_of(i))).unwrap();
    }
    let h2 = filmcraft_codecs::hw::hw_stats();
    assert!(h2.frames > h1.frames && h2.sessions > h1.sessions, "Auto: hardware frames {h1:?} → {h2:?}");
}

/// The Linux factory on its own: with Hardware decoding Auto it is the GPU's decoder for a stream
/// libva takes, and with it Off it declines so FilmCraft's own decoder is used.
#[test]
#[cfg(target_os = "linux")]
fn auto_uses_the_vaapi_decoder() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let _turn = serial();
    let Some(path) = named(&ff, "h264_high.mp4") else { return };
    let s = read_stream(&path);
    filmcraft_platform::register();
    filmcraft_codecs::hw::set_hardware_decoding(true);
    let Some(Ok(d)) = filmcraft_platform::vaapi_factory(&s.entry) else {
        eprintln!("SKIPPED: no VA-API decoder for {}", s.entry.codec.name());
        return;
    };
    assert!(d.name().starts_with("VA-API"), "{}", d.name());

    // and Off still gives FilmCraft's own decoder
    filmcraft_codecs::hw::set_hardware_decoding(false);
    assert!(filmcraft_platform::vaapi_factory(&s.entry).is_none(), "Off declines");
}
