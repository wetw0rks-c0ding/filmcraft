# ADR 0001: `unsafe` OS media FFI in `crates/platform`, and nowhere else

- **Status:** accepted (2026-10-05, approved by the project owner; recorded in
  [AGENTS.md](../../AGENTS.md) §0 item 3 and the craftrules never-crash standard)
- **Issue:** #30 (hardware acceleration)

## Context

FilmCraft is pure Rust and the workspace sets `unsafe_code = "forbid"`. Every codec has a
clean-room software decoder, and those remain the reference. But decoding 4K H.264 / HEVC in
software costs 85–120 ms of CPU per frame on an M4 Pro, so 4K playback drops frames under load,
while every current computer has a hardware video decoder the operating system exposes
(VideoToolbox on macOS; Media Foundation / D3D11 on Windows; VA-API on Linux).

Using it needs FFI into C / Objective-C APIs, which is `unsafe` Rust. The bindings we use (the
`objc2-*` crates, MIT / Apache-2.0 / Zlib) are generated declarations; nothing GPL/LGPL is linked.

## Decision

Allow `unsafe` in exactly one crate, `crates/platform` (`filmcraft-platform`, layer L5), for one
job: OS media FFI (hardware decoding, and hardware encoding through NVENC). Audio output, dialogs, menus
and other OS integration do not go in it.

Containment rules:

1. The crate does not use `lints.workspace = true`. Its own `[lints]` table copies the workspace
   lints except `unsafe_code = "deny"` (not `forbid`), and adds
   `clippy::undocumented_unsafe_blocks = "deny"`. Only the FFI modules (`videotoolbox`, `media_foundation::gpu` / `media_foundation::mft`, and
   `nvenc::ffi` / `nvenc::session` on Windows, `vaapi::ffi` on Linux) carry `#[allow(unsafe_code)]`;
   the rest of the crate (the fallback logic in `hybrid`, the decoder logic in `media_foundation`
   and in `vaapi::mod` / `vaapi::stream` / `vaapi::picture` / `vaapi::decoder`, `annexb`,
   `biplanar`, the encoder logic in `nvenc`) has no `unsafe`.
2. Every `unsafe` block has a `// SAFETY:` comment saying why it is sound.
3. The public API is safe: no `pub unsafe fn`, no raw pointers or FFI types in public signatures;
   failures are `Result`s. The crate keeps the never-crash `deny(clippy::unwrap_used, …)`
   attribute of every clean crate.
4. No panic crosses the FFI boundary: the VideoToolbox output callback runs under `catch_unwind`
   and reports a panic as a failed frame. The Windows backend has no callbacks into Rust (it drives
   a synchronous decoder MFT with `ProcessInput` / `ProcessOutput`), so nothing can unwind into
   COM. NVENC loads the driver library at run time and has no callbacks into Rust either, and the
   Linux VA-API backend is driven synchronously from `decode()` for the same reason.
5. The crate compiles on every target. OS bindings are target-specific dependencies (the
   `objc2-*` crates on macOS, the official `windows` crate on Windows: MIT OR Apache-2.0, already
   in the lockfile through wgpu / winit, with only the Win32 features the backend uses); elsewhere
   `register()` reports `Unavailable` and does nothing. It is not part of the wasm check (L5).

## The fallback guarantee

Registering a hardware decoder never makes a file undecodable, and never changes a picture:

- The factory declines (the software decoder is used) when Settings ▸ Playback ▸ Hardware
  decoding is Off, for formats the hardware path does not take (field-coded H.264, bit depths
  other than 8 / 10, 4:4:4, mismatched luma / chroma depth), and when the OS cannot create a
  hardware session for the stream (hardware decoding is *required*, so the OS's own software
  decoder is never used instead of ours). On Windows that means the decoder MFT must be
  Direct3D-aware, the GPU's DXVA decoder must list the stream, and every picture must come back as
  a Direct3D 11 texture: Microsoft's decoder MFTs quietly decode in software when DXVA is not
  available, and a system-memory picture is treated as a failure.
- A hardware decoder that fails mid-stream (decode error, session invalidated by a GPU change or
  sleep, in-band parameter sets that differ from the sample entry) becomes the software decoder
  for that stream without the caller noticing: `HybridDecoder` replays the samples since the last
  restart point through it and drops pictures already returned. It is logged and counted
  (`perf.stats` `decode.hardware.fallbacks`).
- Pictures are interchangeable with the software decoder's: same planes (bit-exact on the H.264
  High, HEVC Main and Main 10 parity fixtures, including after seeks and flushes), colour, pixel
  aspect, pts and presentation order. Tests compare the two on every run on macOS, and the
  fallback logic is tested on every platform with a stand-in hardware decoder. On Linux the
  comparison is against ffmpeg as an external oracle, picture by picture in presentation order,
  and there is nothing tolerated: everything the VA-API backend can describe is asserted bit for
  bit, while the one H.264 configuration a slice record cannot express — B slices with H.264's
  *implicit* weighted prediction (`weighted_bipred_idc == 2`, which is x264's default for High
  profile with B frames) — is asserted to be declined, so those streams play through the software
  decoder instead. The reason is structural rather than cautious: 8.4.3 derives the implicit weight
  from `pic0 = RefPicList0[refIdxL0]` and `pic1 = RefPicList1[refIdxL1]`, making it a function of
  the **pair**, while `VASliceParameterBufferH264` holds one weight per **list position**. Submitting
  a per-position table anyway was measured to decode 60 of `h264_high.mp4`'s 72 pictures wrongly with
  no error raised, so a decline is the only correct answer here.

## Consequences

- 4K H.264 / HEVC decode costs ~4 ms of CPU per frame instead of 85–120 ms
  ([performance.md](../performance.md)).
- Better patent posture for the most encumbered formats: H.264 / HEVC decoding is done by the
  operating system, which carries the licences, wherever hardware decoding is used.
- A new review burden: changes to `crates/platform` need the same scrutiny as any FFI code.
  Anything else that wants `unsafe` needs a new decision, not an extension of this one.

## Addendum (2026-10-06): hardware H.264 encoding

The crate's second use of the FFI is the one this decision named, hardware encoding: a VideoToolbox
H.264 encoder behind `filmcraft_export::VideoEncoder` (`videotoolbox_encode`, with the same
containment rules; `hardware_encode` is safe code). It differs from decoding in two ways, both
deliberate:

- **It is opt-in per export** (`ExportSettings::hardware_encoding`, `Off` by default). Exports from
  the built-in encoder are byte-identical across machines; a hardware encoder's output depends on the
  machine, so it must be asked for.
- **There is no mid-stream fallback.** A decoder can replay samples through another decoder; an
  encoder cannot hand a half-written stream to another one. The factory declines up front (the
  built-in encoder is used) for everything the hardware path does not take, and for any machine
  where the OS cannot create a hardware session; a hardware encoder that fails during an export
  stops it with an error.

Everything else in this record applies unchanged: `unsafe` stays in `crates/platform`, every block
has a `// SAFETY:` comment, no panic crosses the FFI boundary, the public API is safe, and the crate
compiles everywhere (`register()` is a no-op off macOS).

## Addendum (2026-10-07): hardware H.265 (HEVC) encoding

The same VideoToolbox session wrapper also creates HEVC sessions (`VtProfile::HevcMain`). The rules
above hold; HEVC adds one difference. The crate has no software HEVC encoder to fall back to, and
there will not be one (pure Rust, clean-room: x265 is GPL), so `Format::Hevc` exists only where the
OS has a hardware encoder. Choosing the format is the opt-in, `filmcraft_export::available` asks a
probe the platform crate registers, and what the hardware path does not take (two-pass, odd sizes,
a machine without the encoder) is an error naming the reason instead of a different encoder.

## Addendum (2026-10-09): VP9 through VA-API on Linux

VP9 is the one codec on Linux whose records FilmCraft builds by parsing the bitstream itself, and it
is where this record's "never invent a record" rule is load-bearing rather than merely stated.
`VADecPictureParameterBufferVP9`'s `reference_frames[8]` is **the whole decoded picture buffer
addressed by slot**, not a per-slice list: a wrong slot map is not a crash and not an error, it is a
driver reading some other picture as a reference and producing a plausible wrong one. `Vp9State`
(`vaapi::picture`) therefore carries one `HeaderState` and the eight slots across a run, and a frame's
`refresh_frame_flags` is applied only after the driver has rendered its picture — so a record always
describes the buffer *before* its own frame. VP9 has no weighted prediction in either mode, so the
class of defect that cost the H.264 milestone its budget cannot occur here.

Scope is profiles 0 and 2 at 4:2:0, 8- and 10-bit. Profiles 1 and 3 (4:2:2 / 4:4:0 / 4:4:4) and
profile 4 (reserved by the specification, so undefined rather than merely unimplemented) are declined,
as is 12-bit: every surface this backend creates is 4:2:0 at 8 or 10 bits. **AV1 remains declined**,
because `filmcraft_av1` does not expose its per-frame header and an invented record is precisely the
failure this record exists to prevent.

**That scope is measured, at both depths.** `vp9_p0_parity.mp4` (profile 0, 8-bit) and
`vp9_p2_parity.mp4` (profile 2, 10-bit) are `tests/common::FIXTURES` entries at `Tier::EveryPicture`,
compared picture for picture — 96/96 each — by `bit_exact_with_the_software_decoder`, and regenerated
from the committed table by `cargo xtask fixtures platform`. A third fixture,
`vp9_p0_segmented.mp4`, is the same encode plus `-aq-mode 2` and is the only one whose frames carry
`segmentation_enabled`; without it the eight-entry branch of `va_dec_vp9.h:270-272` would never be
reached on hardware and a record that copied `SegParam[0]` into all eight entries would pass every
other test in the tree.

Two VP9 cases have no H.264 / HEVC analogue and are stated rather than left to fall out of the code:
a frame that re-shows an earlier one (`show_existing_frame`) is **decoded** — VP9 8.2.2 has the
decoder output the picture that slot holds and decode nothing, so it takes no surface from the ring
and is a readback with no submission — and a frame with `show_frame` clear is rendered (it may refresh
references) but produces no picture, which is what FilmCraft's own software VP9 decoder does too.

## Addendum (2026-10-08): VA-API hardware decoding on Linux

Linux reaches the same GPU video engine through the driver's own media stack, libva. The rules
above apply unchanged: `unsafe` is in `vaapi::ffi` only — the hand-written `dlopen` / `dlsym` /
`dlerror` / `dlclose` externs, the hand-declared `VA*` structures and constants, and the `Display`
and `Session` wrappers that own the libva handles — every block has a `// SAFETY:` comment, no
libva type leaves that module, the public API is safe and `Result`-returning, and `register()` is a
no-op off Linux.

libva is **2.x only**: `vaOpenDisplay` and `vaMapSurface` no longer exist in 2.24, so there is no
version dispatch, and a system on libva 1.x declines the stream and decodes in software (today's
behaviour). Two libraries are opened, because `vaGetDisplayDRM` is in `libva-drm.so.2` and
everything else is in `libva.so.2`; `dlclose` runs when the last handle drops, so no function
pointer outlives the library it points at. A machine with no libva, no DRM render node or no driver
reports that it has no hardware decoder instead of failing to start — the same shape as
`mfplat.dll` and NVENC's driver. Decoded pictures come back to the CPU through `vaDeriveImage` +
`vaMapBuffer` (or `vaCreateImage` + `vaPutImage`, when a driver cannot derive a surface into an
image), so there is no zero-copy path on Linux yet — the same gap the other two platforms have.

VP9 is decoded through the same rules (addendum below): profiles 0 and 2 at 4:2:0 8/10-bit, its
picture-parameter record built by `vaapi::picture` from `filmcraft_vp9`'s own uncompressed-header
parser, so libva's requirement that the application parse the frame headers is met by FilmCraft's
own code rather than by a third-party one. AV1 is not, because `filmcraft_av1` does not expose its
per-frame header; declining it is the same "never invent a record" rule the rest of this record
applies.

One caveat about this record's own claims: the safe decline tables in `vaapi::stream` and
`vaapi::picture` have nothing OS-specific in them and would compile anywhere, but the module is
Linux-gated today because `ffi` needs `dlopen`, so those tests run on Linux only. Making them
compile on every target is tracked follow-up work, not a settled state.

### Calling libva at run time — licence position **approved**

libva is **LGPL-2.1+**, and `AGENTS.md` §2 requires permissive dependency licences (MIT /
Apache-2.0 / BSD / ISC / Zlib / Unicode / CC0 / BSL-1.0, or MPL-2.0 used unmodified). This backend
was the one place in the project where that rule came into tension with what was being built. **A
maintainer ruled on it on 2026-10-08: the `dlopen` path is approved.** It is recorded here, with the
reasoning, so the decision can be re-examined rather than rediscovered.

The settled mechanism: `libva.so.2` and `libva-drm.so.2` are opened with `dlopen` and their symbols
resolved with `dlsym` at run time, never linked, so there is no build-time dependency on libva and the
workspace gains no dependency at all. Nothing LGPL is vendored, copied, patched or shipped in the
binary, and the application calls only the ABI of the system's own driver — the way any dynamically
linked program calls `libc`.

That is the reading §2 is being applied with: it governs code entering this repository (vendored,
linked or copied), not libraries a program calls through the operating system's own media stack. The
stricter alternative — that §2 covers any library the process ever calls — was considered and **not**
adopted; it would have ruled VA-API out entirely and closed the top hardware item in `ROADMAP.md`. The
distinguishing fact is that libva is loaded and called by *name* from the user's own system, at the
user's choice, with nothing added to the build: the same posture as VideoToolbox and Media Foundation,
which §0 already permits.

Two consequences worth stating plainly. `libva` and `libva-drm` must stay behind `dlopen` — linking
them would reintroduce a build-time LGPL dependency and void this ruling. And nothing here is
libva-specific enough to be worth keeping private: the backend may be offered to the upstream project
on request, as this one may.
