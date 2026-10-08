# VA-API hardware video decoding on Linux — design

Status: draft for review
Date: 2026-10-07
Milestone: 1 of 2 (decode first, hardware encode is a separate follow-on)

## Why

FilmCraft decodes with its own clean-room Rust codecs; it does not shell out to FFmpeg. Hardware
decoding exists today through the operating system's media APIs: VideoToolbox on macOS
(`crates/platform/src/videotoolbox.rs`) and Media Foundation / D3D11 on Windows
(`crates/platform/src/media_foundation/`). Linux has no such path, so on Linux every picture is
decoded on the CPU and long timelines are slow. `docs/adr/0001-platform-ffi.md:13` already names
"VA-API on Linux later" as the intended path, and `ROADMAP.md:51` lists it as the top hardware gap.

The goal of this milestone is that a Linux machine with a VA-API driver gets hardware decoding for
the same streams the macOS and Windows backends take, through the same seams, with the same
guarantee: hardware decoding never makes a file undecodable.

Out of scope for this milestone: hardware encoding (the `HardwareEncoding::Auto` export path),
zero-copy upload of driver surfaces into wgpu, and 4:2:2 / 4:4:4 chroma.

## What is on the target machine (measured, not assumed)

Verified against this workstation before writing the design:

- GPU: Intel Arc B580 (Battlemage, `xe` kernel driver), `/dev/dri/renderD128`
- `libva 2.24.1`, `intel-media-driver 26.3.5`, `mesa 26.2.4`
- Decode entrypoint `VAEntrypointVLD` is offered for: H.264 ConstrainedBaseline / Main / High,
  HEVC Main / Main10, VP9 profiles 0-3, AV1 Main
- Encode entrypoint `VAEntrypointEncSlice` is offered for H.264, HEVC and AV1 (milestone 2)
- Hardware decode + CPU readback proven end-to-end: a solid `#8040C0` 640x360 frame reads back as
  uniform luma `100` (`0x64`), the exact expected value

Four libva 2.x API facts dominate the implementation. Anything written against pre-2.x libva
(including most tutorials and older projects) is wrong on all four:

| Concern | Pre-2.x libva | libva 2.24 |
|---|---|---|
| Display | `vaOpenDisplay(path)` | **`vaOpenDisplay` removed**; open `/dev/dri/renderD128` and call `vaGetDisplayDRM(fd)` |
| Libraries | `libva.so.2` | `libva.so.2` **plus `libva-drm.so.2`** (dash, not underscore) |
| Initialisation | none | **`vaInitialize(dpy, &maj, &min)` is mandatory**; calling other entry points first segfaults |
| Coded buffer | `VABufferTypePicture` | **removed**; use `VASliceDataBufferType` + `VAPictureParameterBufferType` (+ `VASliceParameterBufferType` for NAL codecs) |
| Surface readback | `vaMapSurface` | **removed**; use `vaDeriveImage` + `vaMapBuffer`, with `vaCreateImage` + `vaPutImage` as the indirect fallback |

`vaCreateConfig` also gained `attrib_list` / `num_attribs` arguments, and
`VASurfaceAttribFormat` is now `VASurfaceAttribPixelFormat`.

## Approach

The backend mirrors `crates/platform/src/media_foundation/`: a `#[allow(unsafe_code)]` FFI module
that resolves symbols at run time and exposes no raw pointers, wrapped by a safe module holding all
the decoder logic. `crates/platform/src/lib.rs` gains a Linux arm that registers the factory and
sets the backend name, exactly as the macOS and Windows arms already do.

### Module layout

```
crates/platform/src/vaapi/
  mod.rs        safe: VaDecoder (the VideoDecoder impl), capability checks, decline reasons
  ffi.rs        unsafe: dlopen + symbol table for libva.so.2 / libva-drm.so.2, VA* C structs
```

`ffi.rs` carries `#[allow(unsafe_code)]`; `mod.rs` contains none. `docs/adr/0001-platform-ffi.md:28`
lists the modules permitted `#[allow(unsafe_code)]` and must be amended to name `vaapi::ffi`.

### Symbol loading

`media_foundation/mft.rs:63-94` and `nvenc/session.rs:33-65` already establish the precedent:
`LoadLibraryExW` + `GetProcAddress` so a machine without the vendor library still starts. The Linux
equivalent is `dlopen` / `dlsym`, used directly — the workspace has no `libloading`, and adding a
dependency is not needed for a dozen symbols. Resolution is lazy and cached in a `OnceLock`
(`Option<VaApi>`); `None` means "no VA-API here", which is a normal, silent condition.

`vaGetDisplayDRM` lives in `libva-drm.so.2`, everything else in `libva.so.2`, so both are opened.
The DRM file descriptor is owned by the `VaApi` handle and closed when it drops.

**Licence stance.** `libva` is LGPL-2.1+, and `AGENTS.md` section 2 requires permissive dependency
licences. Resolving symbols at run time against the system library keeps FilmCraft free of any LGPL
code: nothing is vendored, nothing is linked at build time, and no LGPL text ships in the binary.
The tree stays free of an LGPL dependency; the application only calls the ABI of the system's own
driver, the same way a dynamically linked application calls `libc`. This reasoning is recorded in
ADR 0001 so the decision is auditable rather than implicit.

This is the one place where the design bends a written project rule, so it is stated plainly: the
reading is that AGENTS.md section 2 constrains code that enters this repository (vendored, linked or
copied), not libraries a program calls at run time through the operating system's own media stack.
A stricter reading — that the rule covers any library the process ever calls — would rule out VA-API
entirely and close the top item on ROADMAP.md:51. **This needs an explicit yes or no from a
maintainer, recorded in the ADR.**

### Per-frame picture parameters

This is the substantive difference from VideoToolbox, which takes the bitstream and works out
picture structure itself. VA-API requires the application to supply, per picture, a
`VAPictureParameterBuffer` describing frame number, POC, the reference list and macroblock
dimensions.

FilmCraft already has every field, already parsed, already public:

- H.264 / HEVC: `filmcraft_h264::slice::{SliceHeader, PocState, Poc}` (`frame_num`,
  `first_mb_in_slice`, `idr`, `ref_pic_list_mod`, `field_pic`), plus `filmcraft_h264::params::{Sps, Pps}`
  and the HEVC equivalents
- VP9 / AV1: `FrameStreamInfo::picture_params` (`crates/codecs/src/hw_frame.rs:222`) already returns
  size, bit depth, subsampling and colour per sample

So no new bitstream parsing is introduced. FilmCraft's own parsers build the picture record, which
has the useful property that the software and hardware decoders necessarily agree on picture
structure.

### Decode loop

Per sample:

1. Pick the next free surface from a small ring (VA-API needs several for reordering; the count
   comes from the DPB size the sequence header implies)
2. Build the picture parameter buffer from the parsed slice header
3. `vaCreateBuffer` for `VAPictureParameterBufferType`, `VASliceParameterBufferType` (NAL codecs,
   one per slice) and `VASliceDataBufferType`, using `annexb::to_annex_b` (`crates/platform/src/annexb.rs`,
   already target-independent and unit-tested on every platform)
4. `vaBeginPicture` / `vaRenderPicture` / `vaEndPicture`, then `vaSyncSurface`
5. `vaDeriveImage` + `vaMapBuffer` to read the surface, and hand the two planes to
   `biplanar::to_frame` (`crates/platform/src/biplanar.rs:40`), which already converts NV12 / P010
   byte layouts into `PixelData::Yuv8` / `Yuv16`

Step 5 is why the scope is 4:2:0 at 8 and 10 bit. `vaDeriveImage` returns exactly the two-plane
layout `biplanar::to_frame` already consumes, so readback needs no new format-handling code at all.
4:2:2 and 4:4:4 would require a new chroma-deinterleaving routine and are left for later.

### Capability negotiation and decline

`vaInitialize` reports the driver's version; `vaQueryConfigProfiles` and
`vaQueryConfigEntrypoints` say what it can actually take. The factory declines, with a logged
reason and `hw::note_hw_declined()`, for:

- hardware decoding switched off in Settings (checked first, like the other factories)
- no `libva.so.2` / `libva-drm.so.2`, no DRM render node, or `vaGetDisplayDRM` / `vaInitialize` failing
- a profile or entrypoint the driver does not offer
- interlaced / field-coded streams, chroma other than 4:2:0, bit depths outside 8 and 10,
  sizes above 8192, or a crop outside the coded size
- a config or context the driver refuses to create

`VaDecoder` is wrapped in `HybridDecoder` (`crates/platform/src/hybrid.rs`) exactly as the other
two backends are, so a mid-stream failure falls back to the software decoder transparently and a
mid-stream parameter-set change restarts the hardware session. That reuses the existing
`StreamInfo::parameters_changed` logic rather than reimplementing it.

### Integration points

- `crates/platform/src/lib.rs`: Linux arm in `register()`, `registered()` and
  `hardware_decoder_for()`; `#[cfg(target_os = "linux")] pub mod vaapi;`; `hw::set_hw_backend("VA-API")`
- `crates/platform/Cargo.toml`: no new dependency (symbols are resolved by hand)
- `apps/filmcraft/src/main.rs:278` and `apps/filmcraft-cli/src/main.rs:391`: the
  `cfg!(any(target_os = "macos", target_os = "windows"))` assertions must include Linux
- Documentation that currently says Linux has no hardware path:
  `README.md:187` and `:289`, `ROADMAP.md:51` and `:157`, `crates/platform/README.md:73` and
  `:188-193`, `docs/architecture.md:75` and `:232-245`

## Error handling and safety

Everything the project already requires still applies, and the `crates/platform` containment rules
are what make this crate the right home:

- Every FFI failure is a `Result<T, String>`; the public API has no raw pointers and no `unsafe fn`
- Every `unsafe` block carries a `// SAFETY:` comment (required by
  `clippy::undocumented_unsafe_blocks = "deny"`)
- No panic can cross the FFI boundary. This backend is driven synchronously from `decode()`, like
  the Windows MFT backend and NVENC, so there are no callbacks into Rust at all — nothing can unwind
  into libva
- The crate keeps its `deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, ...)` attribute
- No `unsafe` outside `ffi.rs`; the decoder logic in `mod.rs` is ordinary safe Rust
- Hostile input rules apply to everything derived from the bitstream: slice counts, macroblock
  counts and surface indices are bounded by the sequence header, planes are copied with `get()`,
  pitches come from the driver and are validated against the buffer size before any row read, and
  every size conversion is `try_from` / `checked_*`

Device loss is treated as a decode failure, which is what `HybridDecoder` already knows how to
recover from; VA-API has no explicit device-lost status to poll the way D3D11 does.

## Testing

Following the existing split: logic that does not touch the driver is compiled and unit-tested on
every platform, so macOS and Windows CI keep covering it; FFI-dependent tests are `cfg`-gated to
Linux and skip at runtime when the factory declines.

- `crates/platform/src/vaapi/mod.rs` inline tests: decline tables (what the factory refuses),
  picture-parameter construction from a parsed slice header, surface-ring rotation, bounds checks on
  driver-reported pitches and plane offsets
- `crates/platform/src/vaapi/ffi.rs` layout tests for the `VA*` structs, mirroring
  `nvenc/abi_tests.rs` — the struct layouts are re-declared by hand, so they need checking against
  the real library
- `crates/platform/tests/vaapi.rs`, `#![cfg(target_os = "linux")]`: bit-exact plane parity against
  the software decoder using the existing `tests/common/mod.rs` helpers (`assert_same`), the same
  shape as `tests/media_foundation.rs`; seeks, flush, forced failures, and seeded mutation
- `tests/fallback.rs` and `tests/setting.rs` already run everywhere and need no change, since the
  Linux backend goes through `HybridDecoder` and the shared hardware-decoding switch

Fixtures come from ffmpeg through `filmcraft_testkit::require_ffmpeg!()`, as today. The tests
generate a VP9 and an AV1 fixture for the frame-codec path in addition to the existing H.264 and
HEVC ones.

## Estimated shape

Roughly 1,500-2,000 lines in `crates/platform/src/vaapi/`, plus tests and doc updates — larger than
`videotoolbox.rs` (571), comparable to `media_foundation/` (1,433) and smaller than `nvenc/`
(1,879). Four codecs rather than two is the main reason to expect the upper end of that range. The
encode follow-on would mirror `nvenc/export.rs` against `h264_vaapi` / `hevc_vaapi`, which were
confirmed available on this machine.

## Open questions for review

1. Is the run-time `dlopen` reading of AGENTS.md section 2 acceptable, given libva is LGPL-2.1+?
   This is the one place where the design bends a written project rule, so it should be an explicit
   decision rather than an implementation detail.
2. **libva version support (decided: 2.x only).** `vaOpenDisplay` and `vaMapSurface` no longer exist
   in libva 2.24, so this backend targets the 2.x API only. Older distributions ship libva 1.x,
   where `vaInitialize` does not exist and `vaMapSurface` is the only readback; supporting both would
   roughly double the FFI surface and the test matrix. Systems on libva 1.x fall back to the CPU
   decoder, which is today's behaviour anyway, so nothing regresses. If a real user reports an
   older base, 1.x support is a separately-justified follow-on rather than part of this milestone.
