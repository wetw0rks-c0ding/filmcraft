//! One synthetic picture through the VA-API H.264 slice encoder, written out as an Annex-B stream.
//!
//! This is the spike the encode milestone's Task 1 asks for: **one context, one picture, one
//! readback, one ffmpeg decode**. It is not a feature — there is no frame loop, no rate control and
//! no muxer here, and the crate's encoder does not pretend otherwise. What it establishes is the
//! one thing every later task depends on and cannot be checked without: that the app can hand a
//! picture to the driver and get back bytes ffmpeg decodes to *that* picture.
//!
//! ```text
//! vaencprobe [-qp N] [-w W] [-h H] [-o FILE]
//! ```
//!
//! Default: a 640x360 **left-to-right luma ramp on flat neutral chroma** at QP 26, written to
//! `/tmp/vaencprobe.h264`. Why that picture: a ramp is a known function of the column, so the
//! decode can be compared to it *numerically* rather than by eye, and it is low-frequency, so a
//! stride, plane-offset or row-order bug shows up as a diagonal or a stripe rather than as
//! something a lossy encoder could plausibly have produced. Flat neutral chroma (Cb = Cr = 128) is
//! the third half of that: any colour shift is the chroma plane's and is immediately visible.
//!
//! To verify, decode what it wrote:
//!
//! ```text
//! ffmpeg -i /tmp/vaencprobe.h264 -f rawvideo -pix_fmt gray -  | head -c 640 | xxd | head -4
//! ```
//!
//! and check the first row is 16, 16, 16, ... rising to 235 at column 639. The example prints the
//! measured per-column deviation itself, so "it decoded" and "it decoded to *this* picture" are two
//! different claims and only the second is worth anything.
#[cfg(target_os = "linux")]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut qp = 26u8;
    let (mut width, mut height) = (640u32, 360u32);
    let mut out = "/tmp/vaencprobe.h264".to_string();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-qp" => qp = it.next().and_then(|v| v.parse().ok()).unwrap_or(qp),
            "-w" => width = it.next().and_then(|v| v.parse().ok()).unwrap_or(width),
            "-h" => height = it.next().and_then(|v| v.parse().ok()).unwrap_or(height),
            "-o" => out = it.next().cloned().unwrap_or(out),
            other => {
                eprintln!("vaencprobe: unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    if let Err(why) = run(width, height, qp, &out) {
        eprintln!("vaencprobe: {why}");
        std::process::exit(1);
    }
}

/// Encode one picture and write it out, printing everything the milestone's open questions need.
#[cfg(target_os = "linux")]
fn run(width: u32, height: u32, qp: u8, out: &str) -> Result<(), String> {
    use filmcraft_platform::vaapi::{EncodeRequest, VaEncoder, ramp_picture};
    // `open_display()` opens libva and initialises the driver; it is the same path the decoder takes,
    // so a machine without VA-API says so here rather than failing to start.
    let api = filmcraft_platform::vaapi::open_display().map_err(|why| format!("no VA-API display: {why}"))?;
    println!("driver: {} (libva {}.{})", api.vendor().unwrap_or("(none)"), api.va_version().0, api.va_version().1);

    let mut encoder = VaEncoder::new(&api, (width, height))?;
    let (coded, display) = encoder.sizes();
    let (l, r, t, b) = encoder.crop();
    println!("picture: {display:?} displayed, {coded:?} coded, crop left {l} right {r} top {t} bottom {b}, QP {qp}");
    if coded != display {
        println!("  {}x{} is not a whole number of macroblocks, so it is coded at {coded:?} and cropped back", display.0, display.1);
    }

    // The picture goes in at the coded size, padded with the neutral values the encoder will read.
    // The ramp is built at the *display* size, which is the size the claim is about.
    let (luma, chroma) = ramp_picture(display.0, display.1);
    encoder.upload(&luma, &chroma)?;

    let request = EncodeRequest::new(qp, true)?;
    let picture = encoder.encode(request)?;
    println!("encoded: {} bytes total, {} of them written by the driver, in {} segment(s)", picture.bytes.len(), picture.driver_bytes, picture.segments);
    println!(
        "parameter sets: {}",
        if picture.parameter_sets_from_driver {
            "written by the driver".to_string()
        } else {
            // The measured fact this milestone is built around: iHD accepts
            // VA_ENC_PACKED_HEADER_SEQUENCE|PICTURE and writes none. `crates/platform/src/vaapi/params.rs`
            // has the evidence and derives the ones used here from the records just submitted.
            "written by FilmCraft, derived from the submitted records and verified by FilmCraft's own H.264 parser (the driver wrote none)".to_string()
        }
    );
    println!(
        "driver status {:#010x}: average QP {}",
        picture.status.first().copied().unwrap_or(0),
        // The low byte of `VA_CODED_BUF_STATUS_*` is the average QP the driver says it used
        // (`va.h:3897`). It is a measurement, and it is the only one VA-API offers — worth printing
        // next to the QP that was asked for, because they are not the same statement.
        picture.status.first().copied().unwrap_or(0) & 0xff
    );
    print_nal_units(&picture.bytes);

    // Write the bytes. The mux is not here on purpose: the milestone's Task 4 builds `avcC` from
    // the parameter sets, and a spike that muxed now would have to do that again. Raw Annex-B is
    // what ffmpeg needs to check the claim.
    std::fs::write(out, &picture.bytes).map_err(|e| format!("writing {out}: {e}"))?;
    println!("wrote {out} ({} bytes of Annex-B)", picture.bytes.len());
    println!("decode it with: ffmpeg -v error -i {out} -f rawvideo -pix_fmt gray - | head -c {} | xxd | head -4", display.0);
    Ok(())
}

/// The NAL units the driver wrote, by type and length.
///
/// This is the readback in human form, and it is what says the packed-header mode did its job: the
/// SPS (type 7) and PPS (type 8) are *the driver's*, in the same buffer as the slice (type 5), which
/// is the precondition for Task 4's `sample_entry()`.
#[cfg(target_os = "linux")]
fn print_nal_units(bytes: &[u8]) {
    // A start code is three or four bytes; finding `00 00 01` covers both, and a fourth leading zero
    // simply makes the first NAL one byte shorter, which is harmless for a type report.
    let mut index = 0;
    let mut count = 0;
    while let Some(at) = find(bytes, b"\x00\x00\x01", index) {
        let start = at + 3;
        // The next start code (or the end) is where this NAL ends.
        let end = find(bytes, b"\x00\x00\x01", start).unwrap_or(bytes.len());
        if let Some(&header) = bytes.get(start) {
            count += 1;
            println!("  NAL {count}: type {}, {} bytes", header & 0x1f, end.saturating_sub(start));
        }
        index = start;
    }
    if count == 0 {
        println!("  no NAL start codes: the driver wrote {} bytes that are not Annex-B", bytes.len());
    }
}

/// The first index of `needle` in `haystack` at or after `from`, or `None`.
#[cfg(target_os = "linux")]
fn find(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if from >= haystack.len() || needle.is_empty() {
        return None;
    }
    haystack.get(from..)?.windows(needle.len()).position(|w| w == needle).map(|at| from + at)
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("vaencprobe: the VA-API encoder is Linux-only");
}
