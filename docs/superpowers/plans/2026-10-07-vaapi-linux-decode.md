# VA-API hardware video decoding on Linux — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Linux hardware video decoding for H.264, HEVC, VP9 and AV1 at 4:2:0 8/10-bit, through the same factory, fallback and parity-test seams the macOS and Windows backends already use.

**Architecture:** A new `crates/platform/src/vaapi/` module mirroring `media_foundation/`: an `unsafe` FFI module that `dlopen`s libva at run time and exposes no raw pointers, and a safe module holding all decoder logic. Per picture, FilmCraft builds the `VAPictureParameterBuffer` itself from its own already-public slice-header parsers, decodes into a driver surface, and reads it back to CPU with `vaDeriveImage` + `vaMapBuffer`, feeding the existing `biplanar::to_frame`. `HybridDecoder` wraps it, so a mid-stream failure falls back to the software decoder transparently.

**Tech Stack:** Rust (edition pinned by the workspace), libva 2.x resolved via `dlopen`/`dlsym`, the existing `filmcraft-codecs` / `filmcraft-h264` / `filmcraft-hevc` / `filmcraft-vp9` / `filmcraft-av1` parsers, ffmpeg as a test-fixture generator only.

**Spec:** `docs/superpowers/specs/2026-10-07-vaapi-linux-decode-design.md`

## Global Constraints

- **libva 2.x only.** `vaOpenDisplay` and `vaMapSurface` were removed in libva 2.24 and are not used. There is no version dispatch and no 1.x support; systems on libva 1.x fall back to the software decoder (today's behaviour).
- **No new dependency.** `dlopen`/`dlsym` are called directly; the workspace gains no `libloading`. `crates/platform/Cargo.toml` gains `filmcraft-h264`, `filmcraft-hevc`, `filmcraft-vp9`, `filmcraft-av1` (workspace deps) and nothing else.
- **The DRM library is `libva-drm.so.2`** (dash, not underscore). `vaGetDisplayDRM` lives there; every other symbol is in `libva.so.2`.
- **`vaInitialize(dpy, &maj, &min)` is mandatory** before any other entry point. Calling anything else first segfaults.
- **Scope: 4:2:0 at 8 and 10 bit.** Chroma other than 4:2:0, bit depths outside 8/10, interlaced and field-coded streams, and pictures over 8192 per side are declined. 4:2:2 / 4:4:4 are out of scope.
- **`unsafe` lives only in `vaapi/ffi.rs`**, which carries `#[allow(unsafe_code)]`; every `unsafe` block needs a `// SAFETY:` comment (`clippy::undocumented_unsafe_blocks = "deny"`). No `unsafe` in `vaapi/mod.rs`, `vaapi/stream.rs` or `vaapi/picture.rs`.
- **Safe public API:** every public item returns `Result<T, String>`; no raw pointers or FFI types in public signatures; no `pub unsafe fn`.
- **Never-crash lints** stay on: no `unwrap`, `expect`, `panic!`, `unreachable!`, `todo!` or `unimplemented!` outside `#[cfg(test)]`. The crate-level `deny` attribute in `crates/platform/src/lib.rs` is not to be weakened.
- **Hostile input rules:** every number derived from the bitstream or reported by the driver is validated before use. Driver-reported pitches and plane offsets are checked against the mapped buffer size before any row read; surface indices, slice counts and macroblock counts are bounded; size conversions use `try_from` / `checked_*`.
- **No callbacks into Rust.** The backend is driven synchronously from `decode()`, like the Windows MFT backend, so nothing can unwind across the FFI boundary.
- **Quality gate before every commit:** `cargo xtask ci` (fmt, clippy `-D warnings`, full test suite, layers, assets, wasm).
- **Copy style:** module docs use the project's voice — what the module is, the ASCII pipeline diagram, the "hardware is verified, not assumed" argument, and what is declined. Match `crates/platform/src/media_foundation/mod.rs`.

## Review Focus

Inputs and conditions the spec implies but which no task's tests would otherwise exercise, most likely to bite first:

1. **A driver-reported pitch smaller than one row of the picture** (Intel/AMD align pitches, but a hostile or odd-sized picture can under-align). Row reads would walk out of bounds. Owned by Task 4.
2. **A driver that reports `VAImage` plane offsets/pitches that overrun `data_size`** — `vaMapBuffer` succeeds, then the copy reads past the buffer. Owned by Task 4.
3. **A decoder that returns zero pictures after `vaEndPicture` reports success** (my own probe did exactly this on a malformed picture; a driver can also silently skip a picture it cannot parse). Would show as a blank timeline rather than an error. Owned by Task 4.
4. **A stream whose in-band SPS/PPS differ from the configuration record mid-stream** (a resolution change). Owned by Task 3, via `StreamInfo::parameters_changed` and `HybridDecoder`.
5. **`flush()` mid-GOP followed by more samples** — a drained VA-API context cannot resume from the middle of a run; the decoder must error so `HybridDecoder` replays in software rather than emitting garbage. Owned by Task 4 and Task 6.
6. **A 10-bit stream where luma and chroma depths differ** — must decline, not silently truncate. Owned by Task 2.

---

## File Structure

**Created:**

| Path | Responsibility |
|---|---|
| `crates/platform/src/vaapi/mod.rs` | Safe: `VaDecoder` implementing `VideoDecoder`, surface ring, decode loop, readback, `available()`, `capabilities()` |
| `crates/platform/src/vaapi/ffi.rs` | `unsafe`: `dlopen`/`dlsym` symbol table, the `VA*` C structs and constants we call, safe `Display`/`Session` wrappers |
| `crates/platform/src/vaapi/stream.rs` | Safe: `Stream`, `Plan` (VA profile + entrypoint), decline tables, `input()` for all four codecs. Compiled and tested on every target. |
| `crates/platform/src/vaapi/picture.rs` | Safe: building the per-codec picture parameter buffers (`VAPictureParameterBufferH264`, `VAPictureParameterBufferHEVC`, `VADecPictureParameterBufferVP9`, `VADecPictureParameterBufferAV1`) from parsed slice headers and `FrameStreamInfo`. Compiled and tested on every target. |
| `crates/platform/tests/vaapi.rs` | Linux-only parity, seek, flush, forced-failure and mutation tests. |
| `crates/platform/examples/vacaps.rs` | Diagnostics: what the local driver offers (mirrors `examples/mfcaps.rs`). |

**Modified:**

| Path | Change |
|---|---|
| `crates/platform/src/lib.rs` | Linux arms in `register()`, `registered()`, `hardware_decoder_for()`; `pub mod vaapi;` |
| `crates/platform/Cargo.toml` | Four codec-crate dependencies |
| `apps/filmcraft/src/main.rs:278` | Include Linux in the `registered()` assertion |
| `apps/filmcraft-cli/src/main.rs:391` | Same |
| `docs/adr/0001-platform-ffi.md` | Name `vaapi::ffi` in the `#[allow(unsafe_code)]` list; record the LGPL stance |
| `README.md:187`, `README.md:289` | Linux is no longer a gap |
| `ROADMAP.md:51`, `ROADMAP.md:157` | Mark hardware decode on Linux landed |
| `crates/platform/README.md:73`, `:188-193` | Same |
| `docs/architecture.md:75`, `:232-245` | Same |

**Layer check:** `platform` is L5 and `h264`/`hevc`/`vp9`/`av1` are L0, so the four new dependencies point downward and pass `cargo xtask layers`.

---

### Task 1: FFI layer — symbol loading and ABI structs

The foundation everything else sits on. `media_foundation/mft.rs:63-94` is the precedent for runtime symbol resolution.

**Files:**
- Create: `crates/platform/src/vaapi/ffi.rs`
- Modify: `crates/platform/src/lib.rs` (add `#[cfg(target_os = "linux")] pub mod vaapi;` and the module declaration inside `vaapi/mod.rs`, which Task 1 creates as a stub)

**Interfaces:**
- Consumes: nothing (first task)
- Produces:
  - `pub struct Api` — resolved function pointers, `pub fn load() -> Result<Arc<Api>, String>` (called once, cached by the caller)
  - `pub struct Display` — `pub fn new(api: &Api) -> Result<Arc<Display>, String>`, `pub fn va_version(&self) -> (i32, i32)`, `pub fn vendor(&self) -> Option<&'static str>`
  - `pub fn profiles(&self) -> Result<Vec<Profile>, String>`, `pub fn entrypoints(&self, p: Profile) -> Result<Vec<Entrypoint>, String>`
  - `pub enum Profile { H264ConstrainedBaseline, H264Main, H264High, HevcMain, HevcMain10, Vp9(u8), Av1Main, Other(i32) }`
  - `pub enum Entrypoint { Vld, EncSlice, Other(i32) }`
  - `pub enum RtFormat { Nv12, Yuv420_10, Other(u32) }`
  - `pub fn status(s: i32) -> String` — the `vaErrorStr` text, or the numeric status if the symbol is missing
  - `pub const OK: i32 = 0;`

  All of the above take and return only safe Rust types. Raw `VADisplay`, `VAConfigID`, `VASurfaceID`, `VAContextID`, `VABufferID`, `VAImageID` stay private to `ffi.rs`.

- [ ] **Step 1: Write the failing ABI-layout tests**

In `ffi.rs`, `#[cfg(test)]`, mirror `nvenc/abi_tests.rs`: the `VA*` structs are re-declared by hand, so check sizes and field offsets against the real library when it is present, and skip when it is not.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn va_image_layout_matches_the_library() {
        let Ok(api) = load() else { eprintln!("SKIPPED: no libva"); return };
        let Ok(d) = Display::new(&api) else { eprintln!("SKIPPED: no VA display"); return };
        // VAImage is the struct every readback path copies out of; a wrong layout reads garbage.
        assert_eq!(size_of::<VaImage>(), d.size_of_va_image(), "VAImage size");
        assert_eq!(offset_of!(VaImage, pitches), d.offset_of_va_image_pitches(), "VAImage.pitches offset");
        assert_eq!(offset_of!(VaImage, offsets), d.offset_of_va_image_offsets(), "VAImage.offsets offset");
        assert_eq!(offset_of!(VaImage, num_planes), d.offset_of_va_image_num_planes(), "VAImage.num_planes offset");
    }

    #[test]
    fn va_status_zero_is_success() {
        assert_eq!(OK, 0);
        assert!(!status(OK).is_empty());
    }
}
```

`Display` therefore also needs `pub fn size_of_va_image(&self) -> usize` and the three `offset_of_va_image_*` accessors, which read the real sizes out of the loaded library via a small probe function resolved from it.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p filmcraft-platform --lib ffi`
Expected: FAIL — `load`, `Display`, `VaImage` do not exist yet.

- [ ] **Step 3: Create `vaapi/mod.rs` as a stub and declare the module**

In `crates/platform/src/lib.rs`, next to the existing `#[cfg(target_os = "macos")] pub mod videotoolbox;`:

```rust
#[cfg(target_os = "linux")]
pub mod vaapi;
```

`crates/platform/src/vaapi/mod.rs` starts with the crate's module doc (a shortened version of the one in the spec, in the project's voice) and:

```rust
#[allow(unsafe_code)]
mod ffi;

pub use ffi::{status, Api, Display, Entrypoint, Profile, RtFormat};
```

- [ ] **Step 4: Implement the `VA*` structs and constants in `ffi.rs`**

Declare, with the field types and widths libva 2.24 uses:

- `#[repr(C)] pub struct VaImage { image_id: u32, format: VaImageFormat, buf: u32, width: u16, height: u16, data_size: u32, num_planes: u32, pitches: [u32; 3], offsets: [u32; 3], num_palette_entries: i32, entry_bytes: [u32; VA_PADDING_MEDIUM] }` — verified `size_of == 120`, `offset_of!(num_planes) == 64`, `offset_of!(data_size) == 60`, `offset_of!(pitches) == 68`, `offset_of!(offsets) == 80`
- `#[repr(C)] pub struct VaImageFormat { fourcc: u32, byte_order: u32, bits_per_pixel: u32, depth: u32, red_mask: u32, green_mask: u32, blue_mask: u32, alpha_mask: u32, va_reserved: [u32; VA_PADDING_LOW] }`
- `VaSurfaceAttrib`, `VaContextParameterUpdateBuffer`-adjacent structs only as far as this milestone needs them
- Constants, all verified against `/usr/include/va/va.h` on the target machine with `sizeof`/`offsetof` (the values below are what that probe reported):

  | Constant | Value |
  |---|---|
  | `OK` (`VA_STATUS_SUCCESS`) | `0` |
  | `VA_PADDING_LOW` / `VA_PADDING_MEDIUM` | `4` / `8` |
  | `VA_RT_FORMAT_YUV420` | `0x0000_0001` |
  | `VA_RT_FORMAT_YUV420_10` | `0x0000_0100` |
  | `VAPictureParameterBufferType` | `0` |
  | `VASliceParameterBufferType` | `4` |
  | `VASliceDataBufferType` | `5` |
  | `VAImageBufferType` | `9` |
  | `VAEntrypointVLD` / `VAEntrypointEncSlice` | `1` / `6` |
  | `VAProfileH264Main` / `VAProfileH264High` / `VAProfileH264ConstrainedBaseline` | `6` / `7` / `13` |
  | `VAProfileHEVCMain` / `VAProfileHEVCMain10` | `17` / `18` |
  | `VAProfileAV1Profile0` | `32` |
  | `VAProfileVP9Profile0` .. `VP9Profile3` | `19` / `20` / `21` / `22` |

  `VA_RT_FORMAT_YUV420_10` is `0x100`, not a large packed constant — easy to get wrong, so it is
  pinned by the layout test's sibling assertion in Step 8.

- [ ] **Step 5: Implement `Api::load` with `dlopen`/`dlsym`**

```rust
pub fn load() -> Result<Arc<Api>, String>
```

- Open `libva.so.2` and `libva-drm.so.2` with `libc::dlopen`-equivalent declarations. The workspace has no `libc` dependency in `filmcraft-platform`, so declare the four externs by hand: `dlopen`, `dlsym`, `dlerror`, `dlclose`. Use `RTLD_NOW | RTLD_LOCAL`, which are `0x2` and `0x0` on Linux.
- Resolve every symbol the backend needs into a struct of typed `extern "C" fn` pointers. A symbol that is missing is `Option`, not a load failure, except the ones Task 4 actually calls, which must be present.
- **Keep both handles** in the `Api` so `dlclose` never happens while function pointers are live: `Drop` for `Api` is the only place `dlclose` is called, and `Display` holds an `Arc<Api>`.
- Every `unsafe` block gets a `// SAFETY:` comment naming why the pointer is valid.
- The DRM file descriptor is opened in `Display::new` with `open("/dev/dri/renderD128", O_RDWR)` (`extern` declarations for `open` and `close`), closed in `Drop for Display`. If no render node exists, return `Err` — that is a normal, silent "no hardware here".

The render node path must be overridable for testing: read `FILMCRAFT_VAAPI_DRM_NODE`, defaulting to `/dev/dri/renderD128`. This is a diagnostic escape hatch, documented in the module doc.

- [ ] **Step 6: Implement `Display::new` with the mandatory `vaInitialize`**

```rust
pub fn new(api: &Api) -> Result<Arc<Display>, String>
```

Open the DRM node, call `vaGetDisplayDRM(fd)`, then **`vaInitialize(dpy, &mut maj, &mut min)` and check its status**. Log a clear message when any step fails, naming which one. `va_terminate` is called in `Drop`.

- [ ] **Step 7: Implement the query functions**

`profiles()` maps `vaQueryConfigProfiles` output into `Profile`, including `Vp9(n)` for profiles 0–3 and `Other(i32)` for anything else, so a future driver profile surfaces in `capabilities()` instead of being silently dropped. `entrypoints(p)` maps `vaQueryConfigEntrypoints`. `vendor()` wraps `vaQueryVendorString`, returning `None` rather than an empty string (that is what a driver without a vendor string returns).

Each of these takes the driver-supplied count and array size as hostile: bound them, and never index past what the driver reported.

- [ ] **Step 8: Pin the format constants and run the tests**

Add to the `ffi.rs` test module, so a wrong constant fails on every platform rather than only on a machine with the right driver:

```rust
#[test]
fn the_format_constants_are_the_ones_libva_uses() {
    assert_eq!(YUV420, 0x0000_0001);
    assert_eq!(YUV420_10, 0x0000_0100, "VA_RT_FORMAT_YUV420_10 is 0x100");
    assert_eq!(VA_PADDING_LOW, 4);
    assert_eq!(VA_PADDING_MEDIUM, 8);
    assert_eq!(PICTURE_PARAMETER_BUFFER, 0);
    assert_eq!(SLICE_PARAMETER_BUFFER, 4);
    assert_eq!(SLICE_DATA_BUFFER, 5);
    assert_eq!(IMAGE_BUFFER, 9);
    assert_eq!(VLD, 1);
}
```

Run: `cargo test -p filmcraft-platform --lib ffi`
Expected: PASS — `va_status_zero_is_success` and `the_format_constants_are_the_ones_libva_uses` always; `va_image_layout_matches_the_library` PASSES on this machine (libva 2.24 is present) and prints SKIPPED elsewhere.

- [ ] **Step 9: Commit**

```bash
git add crates/platform/src/vaapi/ffi.rs crates/platform/src/vaapi/mod.rs crates/platform/src/lib.rs
git commit -m "M-HW: VA-API FFI layer for Linux (libva 2.x, symbols resolved at run time)"
```

---

### Task 2: Stream planning and decline tables

Safe code, compiled and tested on **every** target so macOS and Windows CI cover it too. Mirrors `media_foundation/stream.rs`, whose `plan_nal`/`plan_frame` and their decline tables (`stream.rs:233-344`) are the model.

**Files:**
- Create: `crates/platform/src/vaapi/stream.rs`
- Modify: `crates/platform/src/vaapi/mod.rs` (add `mod stream;` and re-exports)

**Interfaces:**
- Consumes: `ffi::{Profile, Entrypoint, RtFormat, Display}` from Task 1; `filmcraft_codecs::hw::{NalCodec, NalStreamInfo, FrameCodec, FrameStreamInfo, StreamInfo}`; `crate::annexb::to_annex_b`
- Produces:
  - `pub enum Stream { Nal(NalStreamInfo), Frame(FrameStreamInfo) }` with `From` impls from each of `StreamInfo`, `NalStreamInfo`, `FrameStreamInfo`
  - `pub struct Plan { pub profile: Profile, pub entrypoint: Entrypoint, pub format: RtFormat, pub size: (u32, u32) }`
  - `pub fn plan_nal(info: &NalStreamInfo) -> Result<Plan, String>`
  - `pub fn plan_frame(info: &FrameStreamInfo) -> Result<Plan, String>`
  - `impl Stream { pub fn plan(&self) -> Result<Plan, String>; pub fn info(&self) -> StreamInfo; pub fn name(&self) -> &'static str; pub fn first_geometry(&self, bits: u32) -> Geometry; pub fn input(&self, sample: &[u8], headers: bool) -> Result<Vec<u8>, String>; pub fn is_restart(&self, sample: &[u8]) -> bool; pub fn declared_picture(&self, sample: &[u8]) -> Option<Geometry> }`
  - `pub fn supports(display: &Display, plan: &Plan) -> Result<(), String>` — asks the driver whether it really offers `plan.profile` / `plan.entrypoint`

  `Stream::name` returns `"VA-API H.264"`, `"VA-API HEVC"`, `"VA-API VP9"`, `"VA-API AV1"`. Tests in Task 6 assert on this prefix.

- [ ] **Step 1: Write the failing decline-table tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_codecs::hw::{FrameCodec, FrameStreamInfo, NalCodec, NalStreamInfo};

    // copied from media_foundation/stream.rs:238 — the crate's hand-built stream description
    fn nal(codec: NalCodec, profile_idc: u8, bits: u32, chroma_format_idc: u32) -> NalStreamInfo {
        NalStreamInfo {
            codec,
            length_size: 4,
            highest_tid: None,
            parameter_sets: Vec::new(),
            coded: (1920, 1088),
            crop: (0, 0, 1920, 1080),
            chroma_format_idc,
            bit_depth_luma: bits,
            bit_depth_chroma: bits,
            interlaced: false,
            profile_idc,
            color: filmcraft_color::ColorInfo::REC709,
            par: (1, 1),
            reorder: 2,
        }
    }

    #[test]
    fn declines_what_vaapi_does_not_hardware_decode() {
        // interlaced
        let mut i = nal(NalCodec::H264, 100, 8, 1);
        i.interlaced = true;
        assert!(plan_nal(&i).is_err(), "field-coded H.264");
        // chroma that is not 4:2:0
        assert!(plan_nal(&nal(NalCodec::H264, 100, 8, 2)).is_err(), "4:2:2");
        assert!(plan_nal(&nal(NalCodec::H264, 100, 8, 0)).is_err(), "monochrome");
        // bit depths outside 8 and 10
        assert!(plan_nal(&nal(NalCodec::H264, 100, 12, 1)).is_err(), "12-bit");
        // luma and chroma depths differing
        let mut i = nal(NalCodec::H264, 100, 10, 1);
        i.bit_depth_chroma = 8;
        assert!(plan_nal(&i).is_err(), "mixed bit depth");
    }

    #[test]
    fn takes_the_four_codecs_at_8_and_10_bit() {
        for (codec, profile, ten) in [(NalCodec::H264, 66u8, false), (NalCodec::H264, 100, false), (NalCodec::H264, 100, true),
                                     (NalCodec::Hevc, 1, false), (NalCodec::Hevc, 2, true)] {
            let p = plan_nal(&nal(codec, profile, if ten { 10 } else { 8 }, 1)).unwrap_or_else(|e| panic!("{codec:?} {profile} {ten}: {e}"));
            assert_eq!(p.entrypoint, Entrypoint::Vld);
            assert_eq!(p.format, if ten { RtFormat::Yuv420_10 } else { RtFormat::Nv12 });
        }
        // `for_tests` is the crate's own hand-built description (hw_frame.rs:195)
        for (profile, bits) in [(0u8, 8u32), (2, 10)] {
            assert!(plan_frame(&FrameStreamInfo::for_tests(FrameCodec::Vp9, profile, bits, (1, 1), (1920, 1080))).is_ok(), "VP9 profile {profile} at {bits}");
        }
        for bits in [8u32, 10] {
            assert!(plan_frame(&FrameStreamInfo::for_tests(FrameCodec::Av1, 0, bits, (1, 1), (1920, 1080))).is_ok(), "AV1 profile 0 at {bits}");
        }
    }

    #[test]
    fn declines_frame_codecs_vaapi_does_not_take() {
        // monochrome and non-4:2:0
        assert!(plan_frame(&FrameStreamInfo::for_tests(FrameCodec::Vp9, 0, 8, (1, 1), (1920, 1080))).is_ok(), "4:2:0 VP9 is taken");
        assert!(plan_frame(&FrameStreamInfo::for_tests(FrameCodec::Av1, 1, 8, (1, 1), (1920, 1080))).is_err(), "AV1 profile 1");
        assert!(plan_frame(&FrameStreamInfo::for_tests(FrameCodec::Vp9, 1, 8, (1, 1), (1920, 1080))).is_err(), "VP9 profile 1 at 8 bits");
        assert!(plan_frame(&FrameStreamInfo::for_tests(FrameCodec::Vp9, 0, 12, (1, 1), (1920, 1080))).is_err(), "VP9 profile 0 at 12 bits");
        assert!(plan_frame(&FrameStreamInfo::for_tests(FrameCodec::Av1, 0, 8, (0, 0), (1920, 1080))).is_err(), "4:4:4");
    }

    #[test]
    fn declines_oversized_and_out_of_range_crops() {
        let mut i = nal(NalCodec::H264, 100, 8, 1);
        i.crop = (0, 0, 9000, 100);
        assert!(plan_nal(&i).is_err(), "wider than 8192");
        // a crop that leaves the coded picture
        let mut i = nal(NalCodec::H264, 100, 8, 1);
        i.crop = (100, 0, 640, 100);
        i.coded = (640, 360);
        assert!(plan_nal(&i).is_err(), "crop outside the coded picture");
        // a zero-size crop
        let mut i = nal(NalCodec::H264, 100, 8, 1);
        i.crop = (0, 0, 0, 100);
        assert!(plan_nal(&i).is_err(), "empty crop");
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p filmcraft-platform --lib vaapi::stream`
Expected: FAIL — `stream` module not declared.

- [ ] **Step 3: Implement `plan_nal` and `plan_frame`**

`plan_nal` declines, in this order, with a distinct message each: interlaced; `chroma_format_idc != 1`; `bit_depth_luma != bit_depth_chroma` or not in {8, 10}; empty or oversized crop; crop leaving the coded picture. Then maps to a profile:

- H.264 `profile_idc` 66 → `Profile::H264ConstrainedBaseline`, 77 → `H264Main`, 100 → `H264High`; anything else declines
- HEVC `profile_idc` 1 or 3 → `HevcMain` (8-bit) or `HevcMain10` (10-bit); anything else declines

`plan_frame` declines: `mono`, `subsampling != (1, 1)`, empty or oversized size, then maps VP9 profile 0 (8-bit) and 2 (10-bit) to `Profile::Vp9(0)` / `Vp9(2)`, and AV1 profile 0 (8- or 10-bit) to `Profile::Av1Main`. Everything else declines with the profile and depth in the message.

`MAX_SIDE` is `8192`, the same value as `media_foundation/stream.rs`.

- [ ] **Step 4: Implement the `Stream` methods**

`input`, `first_geometry`, `is_restart` and `declared_picture` are the same logic as `media_foundation/stream.rs:135-230`, with the same doc comments adapted. `input` calls `crate::annexb::to_annex_b` for NAL codecs (already target-independent) and prepends `config_obus` for a frame codec that needs its sequence header. Add `#[cfg_attr(not(target_os = "linux"), allow(dead_code))]` on the module if CI reports dead code on other targets, matching how `annexb` and `biplanar` are handled in `lib.rs`.

- [ ] **Step 5: Implement `supports`**

Ask the driver: is `plan.profile` in `profiles()` and `plan.entrypoint` in `entrypoints(profile)`? Return `Err` with the profile and entrypoint names when not. This is the "hardware is verified, not assumed" check, and the reason a machine whose driver lists the profile but cannot create a context still declines up front.

- [ ] **Step 6: Run the tests**

Run: `cargo test -p filmcraft-platform --lib vaapi::stream`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/platform/src/vaapi/stream.rs crates/platform/src/vaapi/mod.rs
git commit -m "M-HW: VA-API stream planning and decline tables"
```

---

### Task 3: Picture parameter buffers

The substantive difference from VideoToolbox. VA-API does not work out picture structure itself, so FilmCraft supplies it from its own parsers. All of this is safe code, tested on every target.

**Files:**
- Create: `crates/platform/src/vaapi/picture.rs`
- Modify: `crates/platform/src/vaapi/mod.rs`

**Interfaces:**
- Consumes: Task 2's `Stream`; `ffi`'s buffer-type constants and `Va*` picture-parameter structs (add `VaPictureParameterBufferH264`, `VaPictureParameterBufferH264`-equivalent HEVC struct, `VaSliceParameterBufferH264`, `VaPictureH264`, `VaPictureHevc`, `VaSliceParameterBufferHevc` to `ffi.rs` in this task); `filmcraft_h264::slice::{SliceHeader, PocState, NalHeader}`, `filmcraft_h264::params::{Sps, Pps}`, `filmcraft_bitstream::unescape_rbsp`
- Produces:
  - `pub struct Picture { pub params: Vec<u8>, pub slices: Vec<SliceParam>, pub slice_offsets: Vec<usize> }` — the bytes to hand the driver for one access unit, as opaque owned data so `mod.rs` never sees an FFI struct
  - `pub struct SliceParam { pub offset: usize, pub length: usize }`
  - `pub fn build_nal(info: &NalStreamInfo, sample: &[u8], poc: &mut NalPoc, surface: u32, width_mbs: u32, height_mbs: u32) -> Result<Picture, String>`
  - `pub struct NalPoc` — POC state that persists across pictures (`pub fn reset(&mut self)`)
  - `pub fn build_frame(info: &FrameStreamInfo, sample: &[u8]) -> Result<Picture, String>` — VP9 and AV1, whose picture parameters come from the sample's own header

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_slice_record_per_slice_in_the_access_unit() {
        let info = h264_info();                     // avcC-derived NalStreamInfo, 2 slices per picture
        let sample = h264_access_unit(info.length_size, 2);
        let mut poc = NalPoc::default();
        let p = build_nal(&info, &sample, &mut poc, 7, 40, 23).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(p.slices.len(), 2, "a slice record per slice");
        assert_eq!(p.slice_offsets, vec![0, 1], "records point at their slices");
    }

    #[test]
    fn picture_parameters_carry_the_parsed_slice_header() {
        let info = h264_info();
        let sample = h264_access_unit(info.length_size, 1);
        let mut poc = NalPoc::default();
        let p = build_nal(&info, &sample, &mut poc, 7, 40, 23).unwrap();
        // 40x23 macroblocks for a 640x368 coded picture; the record is opaque bytes, so read it back
        // through the ffi struct and check the fields FilmCraft's own parser produced.
        let pp = unsafe { read_picture_params::<VaPictureParameterBufferH264>(&p.params) };
        assert_eq!(pp.picture_width_in_mbs_minus1, 39);
        assert_eq!(pp.picture_height_in_mbs_minus1, 22);
        assert_eq!(pp.CurrPic.picture_id, 7, "the render target is the surface we passed");
    }

    #[test]
    fn an_unparsable_slice_is_an_error_not_a_panic() {
        let mut info = h264_info();
        let mut sample = h264_access_unit(info.length_size, 1);
        *sample.last_mut() = 0xff;                  // a slice header full of ones
        let mut poc = NalPoc::default();
        assert!(build_nal(&info, &sample, &mut poc, 1, 40, 23).is_err());
        info.length_size = 4;
    }

    #[test]
    fn a_malformed_length_prefix_is_declined() {
        let info = h264_info();
        let mut sample = vec![0, 0, 0, 200, 0x65];  // claims 200 bytes, has 1
        sample.resize(5, 0);
        let mut poc = NalPoc::default();
        assert!(build_nal(&info, &sample, &mut poc, 1, 40, 23).is_err(), "truncated length prefix");
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p filmcraft-platform --lib vaapi::picture`
Expected: FAIL — module not declared.

- [ ] **Step 3: Add the picture-parameter structs to `ffi.rs`**

Declare these six, matching the real headers. Note the VP9 and AV1 types are **not** in `va.h` and are
**not** named `VPPictureParameterBuffer*` — they are `VADecPictureParameterBuffer*` in
`va_dec_vp9.h` and `va_dec_av1.h`:

| Rust name | C type | Header | `size_of` |
|---|---|---|---|
| `VaPictureH264` | `VAPictureH264` | `va.h:3572` | 36 |
| `VaPictureParameterBufferH264` | `VAPictureParameterBufferH264` | `va.h:3599` | 672 |
| `VaSliceParameterBufferH264` | `VASliceParameterBufferH264` | `va.h` | 3128 |
| `VaPictureHevc` | `VAPictureHEVC` | `va_dec_hevc.h` | 28 |
| `VaPictureParameterBufferHevc` | `VAPictureParameterBufferHEVC` | `va_dec_hevc.h` | 604 |
| `VaSliceParameterBufferHevc` | `VASliceParameterBufferHEVC` | `va_dec_hevc.h` | 264 |
| `VaDecPictureParameterBufferVp9` | `VADecPictureParameterBufferVP9` | `va_dec_vp9.h:58` | 92 |
| `VaSliceParameterBufferVp9` | `VASliceParameterBufferVP9` | `va_dec_vp9.h:279` | 316 |
| `VaDecPictureParameterBufferAv1` | `VADecPictureParameterBufferAV1` | `va_dec_av1.h:243` | 1160 |
| `VaSliceParameterBufferAv1` | `VASliceParameterBufferAV1` | `va_dec_av1.h:647` | 40 |

(`size_of` values are from the target machine's `sizeof` probe; the values are advisory — the ABI
check in Task 1 is what enforces them. `VASliceParameterBufferH264` at 3128 bytes is large because
it carries prediction-weight tables.)

Extend the Task 1 layout test to cover these sizes so a struct declared with the wrong fields fails
loudly. Add a `#[cfg(test)] unsafe fn read_picture_params<T: Copy>(bytes: &[u8]) -> T` helper used by
the tests above, which is sound only because the tests build the bytes from a `Picture` the code
itself produced.

The H.264 struct has `seq_fields` / `pic_fields` unions and a trailing
`va_reserved[VA_PADDING_MEDIUM]`; keep both, since the driver may write the reserved words.

- [ ] **Step 4: Implement `build_nal` for H.264**

For each access unit: split into NAL units with `info.nals(sample)`, unescape each RBSP, and parse VCL slice headers with `filmcraft_h264::slice::SliceHeader::parse`, looking PPS/SPS up by id out of `info.parameter_sets`. Build:

- one `VaPictureParameterBufferH264` from the first slice's header: `CurrPic.picture_id = surface`, `CurrPic.flags` from `nal_ref_idc` / slice type, `frame_num`, POC from `PocState::compute`, `picture_width_in_mbs_minus1` / `picture_height_in_mbs_minus1` from the stream's coded size divided by 16 (validated: the coded size must be macroblock-aligned, or decline), `bit_depth_luma_minus8` / `bit_depth_chroma_minus8`, `seq_fields` and `pic_fields` copied from the parsed SPS, `pic_init_qp_minus26` and friends from the PPS, `num_ref_frames` bounded by the parsed DPB size and by `MAX_REF_FRAMES = 16`.
- one `VaSliceParameterBufferH264` per slice: `slice_data_size`, `slice_data_offset`, `first_mb_in_slice`, `slice_data_flag = VA_SLICE_DATA_FLAG_NEW_PICTURE` for the first slice of a picture and `VA_SLICE_DATA_FLAG_NONE` otherwise.

POC state advances once per access unit via `PocState::update`. `nal_ref_idc == 0` pictures are marked `VA_PICTURE_H264_NON_EXISTING`-free and get `CurrPic.flags` without the reference bits.

- [ ] **Step 5: Implement `build_nal` for HEVC**

Same shape against `filmcraft_hevc::slice::SliceHeader::parse` (`:159`), `filmcraft_hevc::params::{Vps, Sps, Pps}`, and `filmcraft_hevc::dpb::PocState::compute` (`:268`). Set `sps_valid`, `pps_valid`, `num_short_term_ref_pic_sets`, the slice segment address, `slice_type`, `pic_output_flag`, `colour_plane_id`, `poc_lsb`, `num_ref_idx_l0_active_minus1` / `l1`, and the RPS from `SliceHeader::st_rps`.

- [ ] **Step 6: Implement `build_frame` for VP9 and AV1**

VP9: fill `VaDecPictureParameterBufferVp9` — `picture_width`, `picture_height`, `render_flags`,
`refresh_gp` / `loop_filter` and the frame context — from `FrameStreamInfo::picture_params(sample)`
plus the sample's uncompressed header. The first picture is a key frame; thereafter they are inter.
One `VaSliceParameterBufferVp9` per spatial segment.

AV1: fill `VaDecPictureParameterBufferAv1` from what `FrameStreamInfo::picture_params` gives
(`crates/codecs/src/hw_frame.rs:234`) plus the frame header bits — `frame_type`, `show_frame`,
`show_existing_frame` and the tile / loop-filter fields the entrypoint requires — and one
`VaSliceParameterBufferAv1` per tile.

The AV1 record is 1160 bytes and version-dependent; if the stream's bitstream version is not one
this milestone builds, return `Err` naming it rather than sending a zeroed record that the driver
would read as a valid picture.

- [ ] **Step 7: Run the tests**

Run: `cargo test -p filmcraft-platform --lib vaapi::picture`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add crates/platform/src/vaapi/picture.rs crates/platform/src/vaapi/ffi.rs crates/platform/src/vaapi/mod.rs
git commit -m "M-HW: VA-API picture parameters from FilmCraft's own slice-header parsers"
```

---

### Task 4: The decoder — surfaces, decode loop, readback

The largest task. Safe code over Task 1's FFI.

**Files:**
- Create: the body of `crates/platform/src/vaapi/mod.rs`
- Modify: `crates/platform/src/vaapi/ffi.rs` (add the config/context/surface/buffer calls)

**Interfaces:**
- Consumes: Tasks 1–3 in full
- Produces:
  - `pub struct VaDecoder { /* as MfDecoder: stream, info, geometry, surfaces, context, ring, in_flight, need_headers, skip_rasl, first, drained, size_checked, timings */ }`
  - `pub fn new(stream: impl Into<Stream>) -> Result<Self, String>`
  - `pub fn available() -> bool`
  - `pub fn capabilities() -> String`
  - `pub fn fail_after(&mut self, n: u64)` (test hook, same shape as `MfDecoder::fail_after`)
  - `pub fn timings(&self) -> Timings` with `pub struct Timings { pub frames: u64, pub decode: Duration, pub readback: Duration, pub convert: Duration }`
  - `impl VideoDecoder for VaDecoder`
  - `struct Ring` and `fn validate_image` — private to `vaapi::mod`, but named here because the
    task's own unit tests call them. `Ring::new(count: usize, first_id: u32) -> Self`,
    `Ring::take(&mut self) -> Option<u32>`, `Ring::len(&self) -> usize`,
    `validate_image(img: &VaImage) -> Result<(), String>`

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// A `VaImage` that is internally consistent for a 4:2:0 picture of `w` x `h` with the given
    /// luma pitch: two planes, chroma interleaved so it uses the same pitch over half as many rows,
    /// and `data_size` covering both.
    fn image(w: u16, h: u16, pitch_luma: u32) -> VaImage {
        let luma_bytes = pitch_luma.checked_mul(u32::from(h)).unwrap_or(0);
        let chroma_bytes = pitch_luma.checked_mul(u32::from(h) / 2).unwrap_or(0);
        VaImage {
            image_id: 0,
            format: VaImageFormat { fourcc: YUV420, byte_order: 0, bits_per_pixel: 12, depth: 8, red_mask: 0, green_mask: 0, blue_mask: 0, alpha_mask: 0, va_reserved: [0; VA_PADDING_LOW] },
            buf: 0,
            width: w,
            height: h,
            data_size: luma_bytes.saturating_add(chroma_bytes),
            num_planes: 2,
            pitches: [pitch_luma, pitch_luma, 0],
            offsets: [0, luma_bytes, 0],
            num_palette_entries: 0,
            entry_bytes: [0; VA_PADDING_MEDIUM],
        }
    }

    #[test]
    fn the_surface_ring_hands_out_every_surface_once_before_reusing_one() {
        let mut ring = Ring::new(4, 4);
        let first: Vec<u32> = (0..4).map(|_| ring.take().expect("a surface")).collect();
        assert_eq!(first, vec![4, 5, 6, 7]);
        assert_eq!(ring.take(), Some(4), "the first surface comes round again");
    }

    #[test]
    fn a_pitch_narrower_than_a_row_is_refused() {
        // Review Focus #1: a driver pitch below one row would make row reads run off the plane.
        assert!(validate_image(&image(640, 360, 8)).is_err(), "pitch 8 < 640");
        assert!(validate_image(&image(640, 360, 639)).is_err(), "pitch 639 < 640");
        assert!(validate_image(&image(640, 360, 640)).is_ok());
        assert!(validate_image(&image(640, 360, 1280)).is_ok(), "a driver-aligned wider pitch");
    }

    #[test]
    fn an_image_whose_planes_overrun_its_buffer_is_refused() {
        // Review Focus #2: vaMapBuffer succeeded but the driver reported a bogus data_size.
        let mut img = image(640, 360, 640);
        img.data_size = 16;
        assert!(validate_image(&img).is_err(), "planes beyond data_size");
        // a plane offset past the end
        let mut img = image(640, 360, 640);
        img.offsets[1] = u32::MAX;
        assert!(validate_image(&img).is_err(), "offset out of range");
        // an offset that overflows when added to a pitch
        let mut img = image(640, 360, 640);
        img.offsets[1] = u32::MAX - 8;
        assert!(validate_image(&img).is_err(), "offset + pitch overflows");
    }

    #[test]
    fn a_surface_that_is_not_two_planes_or_the_wrong_depth_is_refused() {
        let mut img = image(640, 360, 640);
        img.num_planes = 0;
        assert!(validate_image(&img).is_err(), "no planes");
        let mut img = image(640, 360, 640);
        img.num_planes = 3;
        assert!(validate_image(&img).is_err(), "three planes");
        let mut img = image(640, 360, 640);
        img.format.bits_per_pixel = 16;
        assert!(validate_image(&img).is_err(), "16 bits per pixel is not 4:2:0 8/10-bit");
        let mut img = image(0, 360, 640);
        assert!(validate_image(&img).is_err(), "zero width");
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p filmcraft-platform --lib vaapi::`
Expected: FAIL — `Ring`, `validate_image` do not exist.

- [ ] **Step 3: Implement `validate_image`**

`fn validate_image(img: &VaImage) -> Result<(), String>` — the single gate every readback goes through:

- `num_planes` must be 2 (4:2:0 biplanar); anything else declines with the count
- `format.bits_per_pixel` must be 8 or 10
- `width`/`height` must be non-zero
- for each plane, `offsets[i] <= data_size` and `offsets[i] + pitches[i] * required_rows <= data_size`, all with `checked_mul` / `checked_add`
- `pitches[i]` must be at least the picture's width in bytes for the plane (luma: width; chroma: width for interleaved 4:2:0), which is Review Focus #1
- `offsets[0]` must be 0

Use checked arithmetic throughout; a driver that overflows an offset is not to be trusted into a wraparound.

- [ ] **Step 4: Implement `Ring` and the surface pool**

```rust
struct Ring { surfaces: Vec<u32>, next: usize }
impl Ring {
    fn new(count: usize, first_id: u32) -> Self
    fn take(&mut self) -> Option<u32>
    fn len(&self) -> usize
}
```

The surface count comes from the stream's DPB size (`info.reorder` plus a margin), clamped to `[2, MAX_SURFACES = 32]`. A stream declaring a DPB larger than that declines in Task 2's planning rather than allocating without bound.

- [ ] **Step 5: Add the FFI calls to `ffi.rs`**

`vaCreateConfig`, `vaDestroyConfig`, `vaCreateSurfaces`, `vaDestroySurfaces`, `vaCreateContext`, `vaDestroyContext`, `vaCreateBuffer`, `vaDestroyBuffer`, `vaBeginPicture`, `vaRenderPicture`, `vaEndPicture`, `vaSyncSurface`, `vaQuerySurfaceStatus`, `vaDeriveImage`, `vaDestroyImage`, `vaMapBuffer`, `vaUnmapBuffer`, and the indirect fallback pair `vaCreateImage` / `vaPutImage` / `vaDestroyImage`.

`vaSyncSurface` on the render target is required before reading: my hand-written probe read all-zero
pixels until a sync was added, which is exactly the failure mode a driver that renders asynchronously
would cause here.

Each is a safe method on a private wrapper type, so `mod.rs` sees `Result<T, String>` only. **Map/unmap must be exception-safe**: pair `vaDeriveImage` with `vaDestroyImage` and `vaMapBuffer` with `vaUnmapBuffer` on every path, including early returns, so a decode failure never leaks a mapped buffer or a derived image.

- [ ] **Step 6: Implement `VaDecoder::new`**

In order: `stream.plan()?` (Task 2), `Api::load()`, `Display::new()`, `supports(&display, &plan)?`, then `vaCreateConfig`, `vaCreateSurfaces`, `vaCreateContext`. Every failure returns `Err` naming the step and the driver's status text. Log `hardware decoding: {name} on {vendor}` the way `media_foundation/mod.rs:156` does.

`new` takes `impl Into<Stream>` so `StreamInfo` converts directly, matching `MfDecoder::new`.

- [ ] **Step 7: Implement the decode loop**

Per `decode(sample, pts)`:

1. Honour the `fail_after` test hook.
2. If `drained`, accept only a restart point (`stream.is_restart(sample)`) and otherwise return `CodecError::Decode` so `HybridDecoder` replays the run in software — Review Focus #5.
3. Skip HEVC RASL pictures after a CRA at the start of a run, as `MfDecoder::decode` does (`media_foundation/mod.rs:275-290`).
4. For a frame codec, refresh `geometry` from `stream.declared_picture(sample)`.
5. `stream.input(sample, self.need_headers)` for the byte stream.
6. `build_nal` / `build_frame` (Task 3) for the driver records.
7. `vaCreateBuffer` each record, `vaBeginPicture(ctx, surface)`, `vaRenderPicture`, `vaEndPicture`, `vaSyncSurface`.
8. Read back: `vaDeriveImage` → `validate_image` → `vaMapBuffer` → copy the crop out with `biplanar::to_frame` → `vaUnmapBuffer` / `vaDestroyImage`.
   If `vaDeriveImage` returns `VA_STATUS_ERROR_OPERATION_FAILED` — the documented signal that the
   surface cannot be represented as a `VAImage` — fall back once to the indirect path
   (`vaCreateImage` + `vaPutImage` + `vaMapBuffer`). Count the fallback with
   `log::info!` so a machine that always needs it is visible. If the indirect path also fails,
   return an error rather than pushing an empty frame.

**Review Focus #3:** after `vaEndPicture` reports success, if `vaDeriveImage` fails or the surface carries no decodable picture, return an error rather than pushing an empty frame list. A silently blank timeline is the worst outcome; an error hands the stream to `HybridDecoder`.

Track `in_flight: Vec<i64>` exactly as `MfDecoder` does, and error above `MAX_IN_FLIGHT = 64`.

- [ ] **Step 8: Implement `flush`, `reset`, `name`, `is_random_access`, `is_disposable`**

`flush` collects what the driver has, clears `in_flight`, sets `need_headers = true` and `drained = true`, and logs rather than propagating errors (as `MfDecoder::flush` does — `flush` returns no `Result`). `reset` drops the context so the next sample makes a new one with the parameter sets.

- [ ] **Step 9: Implement `available()` and `capabilities()`**

`available()` = `Api::load().and_then(Display::new).is_ok()`. `capabilities()` prints the vendor string, the libva version, and every profile with a yes/no for the VLD entrypoint, mirroring `media_foundation::capabilities`.

- [ ] **Step 10: Run the tests**

Run: `cargo test -p filmcraft-platform --lib vaapi::`
Expected: PASS. `validate_image` and `Ring` tests pass everywhere; nothing here needs a GPU.

- [ ] **Step 11: Commit**

```bash
git add crates/platform/src/vaapi/mod.rs crates/platform/src/vaapi/ffi.rs
git commit -m "M-HW: VA-API decoder with derive/map readback into biplanar::to_frame"
```

---

### Task 5: Registration wiring

**Files:**
- Modify: `crates/platform/src/lib.rs`, `apps/filmcraft/src/main.rs:278`, `apps/filmcraft-cli/src/main.rs:391`

**Interfaces:**
- Consumes: `vaapi::VaDecoder`
- Produces: `pub fn vaapi_factory(entry: &filmcraft_isobmff::SampleEntry) -> Option<filmcraft_codecs::Result<Box<dyn filmcraft_codecs::VideoDecoder>>>` — public, mirroring `videotoolbox_factory` and `media_foundation_factory`, so the Linux integration test and the examples can call it

- [ ] **Step 1: Write the failing test**

`crates/platform/tests/setting.rs` has no `cfg` gate — it runs on every OS, and its switch is
process-wide, so add the Linux assertion as its own `#[cfg(target_os = "linux")]` test in the same
file (following the macOS / Windows arms it already carries):

```rust
#[test]
#[cfg(target_os = "linux")]
fn auto_uses_the_vaapi_decoder() {
    let ff = filmcraft_testkit::require_ffmpeg!();
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
```

`require_ffmpeg!()` expands to a bare `return` when ffmpeg is missing, so no `let Some(ff) = ... else`
wrapper is needed — match the existing tests in that file.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p filmcraft-platform --test setting`
Expected: FAIL — `vaapi_factory` does not exist.

- [ ] **Step 3: Implement `vaapi_factory` in `vaapi/mod.rs`**

Exactly the shape of `media_foundation_factory` (`lib.rs:142-155`): return `None` when `filmcraft_codecs::hw::hardware_decoding()` is off; build the `StreamInfo`; construct `VaDecoder`; wrap in `HybridDecoder::new`; on refusal log the reason and call `hw::note_hw_declined()`.

- [ ] **Step 4: Wire the three functions in `lib.rs`**

- `register()`: a `#[cfg(target_os = "linux")]` arm that calls `filmcraft_codecs::register_video_decoder(vaapi_factory)`, sets `filmcraft_codecs::hw::set_hw_backend("VA-API")` and returns `Availability::Available("VA-API")`. Relax the existing `#[cfg(not(any(target_os = "macos", target_os = "windows")))]` arm to `#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]`.
- `registered()`: add the Linux arm returning `filmcraft_codecs::video_decoder_registered(vaapi_factory)`, and relax the same `cfg`.
- `hardware_decoder_for()`: add a Linux arm building the `StreamInfo` and asking `VaDecoder::new` whether it succeeds, and relax the same `cfg`.

Update the crate-level module doc in `lib.rs` and in `crates/platform/README.md`, which currently say other systems get `Availability::Unavailable`.

- [ ] **Step 5: Relax the app assertions**

`apps/filmcraft/src/main.rs:278` and `apps/filmcraft-cli/src/main.rs:391` both assert `filmcraft_platform::registered() == cfg!(any(target_os = "macos", target_os = "windows"))`. Change both to include `target_os = "linux"`.

- [ ] **Step 6: Run the test**

Run: `cargo test -p filmcraft-platform --test setting`
Expected: PASS — on this machine it runs and asserts the VA-API name; on a machine without a driver it prints SKIPPED.

- [ ] **Step 7: Run the gates**

Run: `cargo xtask ci`
Expected: PASS — including `layers: ok` with the four new L0 dependencies.

- [ ] **Step 8: Commit**

```bash
git add crates/platform/src/lib.rs crates/platform/src/vaapi/mod.rs crates/platform/Cargo.toml apps/filmcraft/src/main.rs apps/filmcraft-cli/src/main.rs crates/platform/tests/setting.rs
git commit -m "M-HW: register the VA-API decoder factory on Linux"
```

---

### Task 6: Parity, seek, flush, failure and mutation tests

The tests that prove the backend is interchangeable with the software decoder, which is the whole claim of the milestone.

**Files:**
- Create: `crates/platform/tests/vaapi.rs`
- Modify: `crates/platform/tests/common/mod.rs` (add VP9 and AV1 fixtures)

**Interfaces:**
- Consumes: `filmcraft_platform::vaapi_factory`, `filmcraft_platform::HybridDecoder`, the existing `tests/common` helpers
- Produces: nothing consumed by later tasks; this task's deliverable is confidence

- [ ] **Step 1: Add the VP9 and AV1 fixtures**

Append to `FIXTURES` in `crates/platform/tests/common/mod.rs`. They must match the shape of the existing ones — B-frames, a 360-line picture so the coded height is cropped, two or more GOPs, and a 10-bit case:

```rust
(
    "vp9_profile0.mp4",
    &["-f", "lavfi", "-i", "testsrc2=s=640x360:r=24:d=3,noise=alls=12:allf=t",
      "-c:v", "libvpx-vp9", "-b:v", "2M", "-profile:v", "0", "-pix_fmt", "yuv420p"],
),
(
    "vp9_profile2_10bit.mp4",
    &["-f", "lavfi", "-i", "testsrc2=s=640x360:r=24:d=3,noise=alls=12:allf=t",
      "-c:v", "libvpx-vp9", "-b:v", "4M", "-profile:v", "2", "-pix_fmt", "yuv420p10le"],
),
(
    "av1_profile0.mp4",
    &["-f", "lavfi", "-i", "testsrc2=s=640x360:r=24:d=3,noise=alls=12:allf=t",
      "-c:v", "libaom-av1", "-cpu-used", "8", "-profile:v", "0", "-pix_fmt", "yuv420p"],
),
```

Generate them once and confirm each plays before relying on them:

Run: `cargo test -p filmcraft-platform --test vaapi` (the fixture step below will generate them)
Expected: fixtures appear under `target/fixtures/platform/` and ffmpeg reports no error.

If a machine's ffmpeg lacks `libvpx-vp9` or `libaom-av1`, the existing `fixture` helper already returns `None` and the test skips — that path is already in use, so no new handling is needed.

- [ ] **Step 2: Write the test file**

`crates/platform/tests/vaapi.rs`, `#![cfg(target_os = "linux")]`, with the same shape as `tests/media_foundation.rs`:

```rust
//! VA-API decoding against our software decoders (Linux): every picture of the H.264 High, HEVC Main,
//! HEVC Main 10, VP9 and AV1 fixtures must be identical (bit-exact planes, colour, pixel aspect, pts
//! and presentation order), also after `reset` + reseek and a mid-stream `flush`; a forced
//! mid-stream failure must continue in software with the software decoder's exact output; damaged
//! samples and parameter sets must give errors or fall back, never crash or hang. Skips without
//! ffmpeg (fixture generator) or without a VA-API driver.
#![cfg(target_os = "linux")]

mod common;

use std::time::Duration;

use common::*;
use filmcraft_codecs::VideoDecoder;
use filmcraft_platform::HybridDecoder;

/// The hardware decoder for a stream, or `None` (skip) when this machine has none for it.
fn hardware(s: &Stream) -> Option<Box<dyn VideoDecoder>> {
    match filmcraft_platform::vaapi_factory(&s.entry) {
        Some(Ok(d)) => { assert!(d.name().starts_with("VA-API"), "{}", d.name()); Some(d) }
        Some(Err(e)) => panic!("factory error: {e}"),
        None => { eprintln!("SKIPPED: no VA-API decoder for {}", s.entry.codec.name()); None }
    }
}

fn software(s: &Stream) -> Box<dyn VideoDecoder> {
    filmcraft_codecs::software_video_decoder(&s.entry).unwrap()
}
```

Then the four tests, copied in structure from `tests/media_foundation.rs`:

- `parity_matches_the_software_decoder(name, s)` — whole stream, then `reset` + reseek to each later sync sample, then a mid-stream `flush`, then a full pass after all the resets, with `assert!` on picture count and `assert_same` on every comparison. This is Review Focus #5's test.
- `a_forced_mid_stream_failure_falls_back_to_software` — drive the inner `VaDecoder` to `fail_after(n)` for several `n`, wrap it in `HybridDecoder`, and require the output to equal the software decoder's exactly.
- `damaged_samples_give_errors_or_fall_back` — flip bits in the slice data with `xorshift`, require only errors, fallback, or correct pictures, never a panic or hang (bound each with the same `Duration` timeout `media_foundation.rs` uses).
- `in_band_parameter_sets_are_not_confused_with_the_config_record` — a sample carrying different SPS/PPS must fall back rather than decode with the wrong picture structure. This is Review Focus #4's test.

Drive all of it from `#[test]` wrappers that iterate over the six fixture names.

- [ ] **Step 3: Run the tests**

Run: `cargo test -p filmcraft-platform --test vaapi -- --nocapture`
Expected: PASS on this machine for the fixtures its driver supports; the rest print `SKIPPED`. On this Arc B580, expect H.264, HEVC Main, HEVC Main 10, VP9 profile 0, VP9 profile 2 and AV1 profile 0 to run, since the driver advertises VLD for all of them.

If a parity test fails with a plane mismatch, read the reported `max diff` before touching the code: a small consistent difference points at a colour-range or shift mistake in the picture record, while a large one points at the surface or pitch handling.

- [ ] **Step 4: Add the diagnostics example**

`crates/platform/examples/vacaps.rs`, mirroring `examples/mfcaps.rs`: print `vaapi::capabilities()`.
No `Cargo.toml` change is needed — `crates/platform/examples/` is picked up automatically (the crate
declares no `[[example]]` sections, and `mfcaps.rs` / `mfprobe.rs` are not registered either).

- [ ] **Step 5: Run the gates**

Run: `cargo xtask ci`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/platform/tests/vaapi.rs crates/platform/tests/common/mod.rs crates/platform/examples/vacaps.rs
git commit -m "M-HW: VA-API parity, seek, flush, fallback and mutation tests"
```

---

### Task 7: Documentation and the ADR

**Files:**
- Modify: `docs/adr/0001-platform-ffi.md`, `README.md`, `ROADMAP.md`, `crates/platform/README.md`, `docs/architecture.md`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: nothing consumed by earlier tasks; this task's deliverable is an auditable record

- [ ] **Step 1: Amend ADR 0001**

Two edits to `docs/adr/0001-platform-ffi.md`:

- In the containment rules (`:28`), add `vaapi::ffi` to the list of modules permitted `#[allow(unsafe_code)]`: change `videotoolbox`, `media_foundation::gpu` / `media_foundation::mft`, and `nvenc::ffi` / `nvenc::session` to also include `vaapi::ffi` on Linux.
- Add a new subsection, **"Calling libva at run time"**, recording the licence stance: `libva` is LGPL-2.1+, FilmCraft resolves its symbols with `dlopen`/`dlsym` at run time rather than linking it, no LGPL code is vendored, copied or shipped, and the application calls only the ABI of the system's own driver. State plainly that this is a reading of `AGENTS.md` section 2 — that the rule governs code entering the repository, not libraries called at run time — and that a stricter reading would rule out VA-API entirely, so the decision is recorded here for review rather than left implicit.

Also note in the same subsection that this backend targets libva **2.x only**: `vaOpenDisplay` and `vaMapSurface` were removed in libva 2.24, and systems on libva 1.x fall back to the software decoder.

- [ ] **Step 2: Update `crates/platform/src/lib.rs` module doc**

The crate doc says hardware decoding covers VideoToolbox and Media Foundation and that other systems register nothing. Add VA-API on Linux (libva 2.x, H.264 / HEVC / VP9 / AV1 at 4:2:0 8/10-bit) to that list.

- [ ] **Step 3: Update `crates/platform/README.md`**

`:73` says other systems get `Availability::Unavailable`; change it to name Linux. Remove `VA-API (Linux) decoders` from the "Not yet" list (`:188-193`) and, while there, leave the zero-copy item in place since it is still true.

- [ ] **Step 4: Update `README.md`**

`:187`: hardware decoding works on macOS, Windows and Linux. `:289`: Linux is no longer listed as having no hardware path; keep the honest note that export rendering still runs on the CPU (hardware encode is the follow-on).

- [ ] **Step 5: Update `ROADMAP.md`**

`:51`: strike hardware decode on Linux (VA-API) from the gaps, leaving the zero-copy and GPU-export items. `:157`: add the landed VA-API decode milestone to the "Running now" hardware-acceleration log, and make "Next" say hardware encoding on Linux (`h264_vaapi` / `hevc_vaapi`, which the driver probe confirmed) alongside zero-copy upload and Lumetri.

- [ ] **Step 6: Update `docs/architecture.md`**

`:75`: the crate table row for `platform` lists the two backends; add Linux. `:232-245`: the Hardware decoding section says the factory "elsewhere does nothing"; add the Linux arm and note that it is 4:2:0 8/10-bit with CPU readback, so a zero-copy note stays accurate.

- [ ] **Step 7: Run the gates**

Run: `cargo xtask ci`
Expected: PASS. `cargo xtask assets` must still pass — no new asset files were added, and this task adds none.

- [ ] **Step 8: Commit**

```bash
git add docs/adr/0001-platform-ffi.md README.md ROADMAP.md crates/platform/README.md docs/architecture.md crates/platform/src/lib.rs
git commit -m "docs: record the VA-API backend, its licence stance and libva 2.x requirement"
```

---

## Follow-on (not in this plan)

Hardware **encoding** on Linux, the milestone this one defers to: an H.264/HEVC `VideoEncoder`
registered with `filmcraft_export::register_encoder`, declining unless
`ExportSettings::hardware_encoding == HardwareEncoding::Auto`, mirroring
`crates/platform/src/nvenc/export.rs`. The driver probe on this machine confirmed `h264_vaapi` and
`hevc_vaapi` encode successfully. Beyond that: zero-copy dmabuf into wgpu, and 4:2:2 / 4:4:4 decode.
