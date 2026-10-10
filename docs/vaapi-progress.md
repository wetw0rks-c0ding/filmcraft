# Linux GPU video support: progress

Status of the VA-API work on this branch, written so a reader does not have to reconstruct it from 36
commits. Everything here has been measured on the development machine (Intel Arc B580, `/dev/dri/renderD128`,
libva 1.24, iHD 26.3.5); nothing is carried over from another platform unverified.

Two milestones are in flight: **decode**, essentially complete, and **encode**, mid-way through its
feasibility work. A third, **VP9 decode**, is wired into the decode path; the picture-for-picture
oracle tests are the next step.

---

## 1. Hardware decode — working

A decode backend in `crates/platform/src/vaapi/`. H.264 and HEVC at 4:2:0, 8- and 10-bit, plus VP9
profiles 0 and 2 at the same depths (§4).

**Reached through the application, not only a test harness.** `apps/filmcraft/src/main.rs:247` calls
`filmcraft_platform::register()` at startup; `register()` puts the VA-API factory in front of the software
decoders and reports `Available("VA-API")`. MP4 playback reaches it via `Mp4Source::make_decoder` →
`make_video_decoder`, which walks that registry. `crates/platform/tests/setting.rs::auto_uses_the_vaapi_decoder`
plays 30 frames through the real media stack and asserts the hardware frame count advances, with a control
that asserts it does *not* when hardware decoding is switched off.

**Bit-exact against ffmpeg, picture for picture:**

| fixture | result |
|---|---|
| `h264_high_unweighted.mp4` | 72/72 |
| `hevc_main.mp4` | 72/72 |
| `hevc_main10.mp4` | 72/72 |
| `hevc_multi_segment.mp4` | 72/72 |
| `hevc_weightp.mp4` | 50/50 (weighted prediction, hardware path) |
| `hevc_weightp_main10.mp4` | 50/50 (weighted prediction, 10-bit) |

The last two exist because every earlier HEVC fixture coded a prediction weight table whose every flag was
0 — so the weight record's delta recovery had only ever run with delta 0, *on hardware*. The new fixtures
drive 35 and 28 non-default luma weights, 35 and 29 non-zero luma offsets and denominators spanning 0–7
through the record. (`fade=in` is what makes x265 emit a non-default weight at all; plain `testsrc2` does
not, at any encoder effort setting.)

**Measured acceleration**, from `cargo xtask bench --sections decode`:

| fixture | fps | hardware frames | fallbacks |
|---|---|---|---|
| `dec_hevc_1080` | 214.1 | 360 | 0 |
| `dec_hevc_2160` | 63.4 | 216 | 0 |
| `dec_hevc10_2160` | 42.3 | 216 | 0 |
| `dec_h264_1080` | 89.8 | 9 | 3 |
| `dec_h264_2160` | 23.4 | 9 | 3 |

**Readback is to the CPU** (`vaDeriveImage` + `vaMapBuffer`), not zero-copy. At 640×368 the driver decode is
0.55 ms/picture and the GPU→CPU copy is 0.77 ms, so on small pictures the copy costs more than the decode.
This is a deliberate choice, not an oversight.

---

## 2. Hardware decode — declined, with the reason

A decline is always a clean fall back to the software decoder, never a guessed record. Each carries the
reason.

| declined | why |
|---|---|
| H.264 with implicit weighted biprediction | See below. `weighted_bipred_idc == 2` is x264's default for High profile with B frames, so this is **most H.264**. |
| AV1 | A repo visibility boundary, not a capability one: `filmcraft_av1` does not expose the per-frame header libva wants as a record. See §5. |
| VP9 outside profiles 0 / 2 | 4:2:2 / 4:4:0 / 4:4:4 (profiles 1, 3), the reserved profile 4, and 12-bit. Every surface this backend creates is 4:2:0 at 8 or 10 bits, so there is nowhere to put them. |
| 10-bit H.264 | Matches the Windows backend's own decision (`media_foundation/stream.rs:279`); no fixture would prove a hardware path here. |
| HEVC with `scaling_list_enabled_flag` | Needs `VAIQMatrixBufferHEVC`, which this backend does not build. `va_dec_hevc.h` sends that buffer only when the flag is 1, so a driver default would decode a picture the stream does not contain. |
| 4:2:2, 4:4:4 | Out of scope by decision. |

### The H.264 implicit-weight decline

Verified against **ITU-T H.264 (V16) 06/2026**, clause **8.4.3 "Derivation process for prediction weights"**,
cross-checked against the 08/2024, 08/2021, 03/2010, 03/2005 and 05/2003 editions.

Under `implicitModeFlag == 1` there is no array and no `refIdx*WP` at all. The weights come from equations
8-197, 8-198, 8-201, 8-202, 8-282 and 8-283, evaluated **per macroblock** from `refIdxL0` and `refIdxL1`. The
array lookups (`w0C = luma_weight_l0[refIdxL0WP]`, 8-289) exist **only** under `explicitModeFlag == 1`.

So the implicit weight is irreducibly a function of the `(refIdxL0, refIdxL1)` pair — n₀×n₁ entries is the
minimum — and `VASliceParameterBufferH264` has no slot for it. `RefPicList0[32]`/`RefPicList1[32]` are
parallel to `luma_weight_l0[32]`/`luma_weight_l1[32]`, which is per *list position*.

Submitting a per-position approximation was **measured, not argued**: it decodes `h264_high.mp4` to 60 of 72
pictures differing from ffmpeg with no error raised at all. Leaving the `*_weight_lX_flag` fields at 0 so
the driver might derive them was also measured: all 72 pictures decode, 12 match. The driver uses the table
it is given.

**This is unresolved in one direction.** ffmpeg hardware-decodes the same fixture correctly on this machine —
verified with `-hwaccel_output_format vaapi` reporting `pix_fmt: vaapi` and no fallback, output byte-identical
to its own software path. Per the above the driver is handed a pair-dependent weight it cannot store, so
either ffmpeg does something the API does not describe, or iHD derives implicit weights from the slice
header under a condition not yet found. Recorded rather than resolved.

---

## 3. Hardware encode — in progress

Plan: `docs/superpowers/plans/2026-10-09-vaapi-linux-encode.md`. Encode inverts the decode data flow: the
application hands the driver a *picture* and **the driver writes the slice headers**, so the bitstream is read
back out and the codec configuration is the driver's. Verification is necessarily **weaker** than decode's —
a hardware encoder and the software encoder are different algorithms, so bit-equality is unavailable.

Feasibility is established: the capability probe reports `EncSlice: true` for H.264 Main/High/ConstrainedBaseline
and HEVC Main/Main10 on this machine. VP9 reports false, correctly — there is no hardware VP9 encoder.

Three spikes, each measured rather than assumed:

1. **One I frame.** Encodes and decodes to the right picture (luma ramp, max deviation 3/255, driver average
   QP as requested).
2. **P frames predict from the picture before them.** Six frames, ramp shifted 8 columns per picture:
   935 bytes for the I frame then 656/307/322/196/208. Two controls make it falsifiable — coded as intra the
   same content averages 941 bytes (2.79×), and a stale reference would give a 253/255 maximum against a
   measured ~0.3 mean, a 60× separation. **Reference depth > 1 is refused, not attempted**: at depth 2 the
   render succeeds, sync succeeds, ffmpeg decodes every frame *without a warning*, and every picture from the
   third on is wrong (mean 7.16 rising to 35.08). Four candidate causes were measured and cleared first.
3. **The coded-output readback**, which was the blocker. `vaMapBuffer` returns a staging area, not the
   picture. Solved by `vaAcquireBufferHandle` → `DRM_IOCTL_PRIME_HANDLE_TO_FD` → `mmap`: every byte now comes
   from a mapping this backend issued. Output byte-for-byte identical to the previous mechanism.

`pic_order_cnt_type = 2` was tried as a lead for depth > 1 and is **refuted** — the driver reads it, but the
pictures are byte-identical and the refusal stands.

### The FFI audit

A wrong-arity `vaSyncBuffer` (libva's third parameter is `uint64_t timeout_ns`) was found producing correct
pictures in release and `VA_STATUS_ERROR_TIMEDOUT` in debug. An audit of all 32 entry points against the
headers found **three further mismatches, all the same shape — a declaration smaller than the header's**:

- `VaSurfaceAttrib` declared 16, actually 24.
- `VaBufferInfo` declared 24, actually 40 — **live**, passed by pointer to a stack local; this driver
  ignores the difference, so it cost nothing here and is a stack overflow away from firing.
- `struct drm_prime_handle` missing its third field, so the ioctl request number was wrong. The kernel
  dispatches on the number alone so the call succeeded, the returned descriptor was never read and never
  closed — **one leaked dma-buf per encoded picture**, measured — and a descriptor libva owns was double-closed.

Entry points are now declared once in a macro that derives arity mechanically, with tests that check it
against the installed headers (one counts the header's parameters, one has the C compiler static-assert every
record's `sizeof`). Both were verified to fail on the exact historical bugs. `cargo xtask ci` now runs the
platform tests unoptimised as well, so a profile divergence cannot hide a bug again.

---

## 4. VP9 decode — wired, not yet proven against hardware

Plan: `docs/superpowers/plans/2026-10-09-vaapi-linux-vp9-decode.md`.

**The enabling change was one line.** `crates/vp9/src/header.rs` already held
`VADecPictureParameterBufferVP9`'s inputs field for field, and `parse_uncompressed` already takes
`refs: &[Option<RefInfo>; 8]` — the shape of the record's `reference_frames[8]`. Every field is already `pub`.
Only `crates/vp9/src/lib.rs:27`, `mod header;`, was in the way. `crates/vp9` itself is **unchanged** by
this work: it is green and bit-exact with ffmpeg, and the software path depends on it.

**What the record needs beyond the frame header** is the part the first draft of the plan got wrong.
It is not only the fields: the record also needs `frame_header_length_in_bytes` (the *uncompressed*
header's length), `first_partition_size` (the compressed header's), and the two segmentation
probability arrays — which the header itself says "should be parsed directly from bitstream by
application" (`va_dec_vp9.h:172-177`), so they are this backend's job rather than the driver's. The
92-byte record is pinned three ways against a C compile, including a byte-for-byte literal.

**The state is the risky part, and it is named.** `reference_frames[8]` is **the whole decoded picture
buffer addressed by slot**, not a per-slice list: build the record against the wrong slot map and the
driver reads some other picture as a reference, without complaint. `Vp9State` threads one `HeaderState`
across a run and owns the eight slots, and a frame's `refresh_frame_flags` is applied only after the
driver has rendered it, so a record describes the buffer *before* its own frame. Two cases have no
H.264 / HEVC analogue and are decided explicitly rather than left to fall out of the code:
`show_existing_frame` is **decoded** (VP9 8.2.2: output the picture that slot holds, decode nothing),
and `show_frame` clear is rendered but produces no picture.

VP9 is the cheap codec here for three reasons that are properties of the codec: no slice headers,
references in a fixed eight-slot array rather than a per-slice list, and **no weighted prediction in
either mode** — so the class of bug that cost the H.264 milestone most of its budget cannot occur. Scope
is profiles 0 and 2; profiles 1 and 3 decline on chroma, profile 4 is reserved.

**Not yet proven on hardware.** The wiring, the decline tables and the record are in place and unit-
tested; the picture-for-picture comparison against ffmpeg is the next task, and it is the task that
would catch a record that is valid but wrong.

---

## 5. Why AV1 is not decoded yet

Not a hardware limit — the driver advertises `VLD: true` for AV1 Main here.

`crates/av1/src/header.rs:478` declares `pub(crate) fn parse` over private types, so the per-frame
header libva wants as a `VADecPictureParameterBufferAV1` is unreachable. The parser is already written
and bit-exact with ffmpeg in software; it is simply not exposed — and exposing it is a change in
`crates/av1` with its own review, which is why it is a separate milestone from VP9's rather than part
of it. VP9 had the identical wall (`mod header;` in `crates/vp9/src/lib.rs`) and clearing it is what
this milestone did.

The architectural asymmetry is worth understanding: the Windows backend decodes VP9 and AV1 today by
mapping them to fourcc and DXGI profiles (`MFVideoFormat_VP90`, `D3D11_DECODER_PROFILE_AV1_VLD_*`) and
**never parses the bitstream**. Media Foundation and VideoToolbox delegate header parsing to the OS.
VA-API requires the application to parse headers and build every picture record, which is why this
project needs FilmCraft's own parsers for VA-API at all — and why it meets their visibility.

---

## 6. Findings recorded because they were wrong here first

Four conclusions on this branch were wrong and were caught by measurement rather than review. They are listed
because each was plausible, and each cost real time:

- **A clause that does not exist.** An H.264 "8.2.3.2.1" deriving separable per-position weights was proposed
  and acted on. Zero occurrences of that clause number — or of `lumaWeightL0`, `weightA`, `weightB`,
  `PocDiffB`, `w1`/`w2` — across six ITU editions. A fabricated clause reads exactly like a real one.
  **Specificity is not evidence.**
- **A grep that missed fields.** `RefPicList0`/`RefPicList1` are CamelCase; searching for `ref_pic_list`
  found nothing and produced a confident, wrong conclusion that the record had no reference lists at all.
- **A desync bug that was not there.** `luma_offset_lX` was suspected absent from HEVC's `pred_weight_table`.
  It is present — clause 7.3.6.3, in the same `for(i)` body as `delta_luma_weight`, and present since v1.
- **A struct miscounted while verifying.** `struct drm_prime_handle` was confirmed as 8 bytes from five
  lines of a header; line 932 holds a third field. The verification read as rigorous and was wrong.

Also worth recording: a per-task review by fresh subagents **missed** the MMCO 3/6 spec deviation and the
HEVC reference-lists-built-once-per-picture bug, both of which a single CodeRabbit pass found. The
second silently produced wrong pictures on multi-segment HEVC and survived four review rounds.

---

## 7. Not ours: `crates/gpu` failures

`cargo xtask ci` is **red** for reasons predating this branch. Proven unrelated: `git diff main...HEAD --
crates/gpu crates/render` is empty, `filmcraft-gpu`'s dependency closure (`color`, `frame`, `geom`, `render`,
`media`, `project`, `time`) contains nothing this branch changes, `cargo tree` confirms `codecs`, `platform`,
`engine` and `hevc` are absent from its tree, and `Cargo.lock` has zero version bumps. Almost certainly the
new Arc B580 and its driver.

- `fx_tests::gpu_effects_match_cpu_exactly` — `invert` with `channel=Choice(4)` (alpha) fails on *flips*, a
  2 %+ relative difference. Not a tolerance problem and deliberately not absorbed by one: the Rust CPU
  reference and the GPU arm in `gpufx.rs` are byte-identical including the `k = na/a` guard at `a > 1e-6`, so
  the divergence is in the shader, and that ratio is unbounded as alpha approaches zero.
- The suite **segfaults 3 of 3 threaded and never single-threaded** — a concurrency defect, likely multiple
  GPU contexts created at once, not flakiness.

Fixed here: a per-op tolerance for the effects whose GPU path is a power function (`gamma_correction`,
`levels` via `powf(g)`; `tint`, `extract` via `linear_to_srgb`). A GPU evaluates `pow(x, g)` as
`exp2(g·log2(x))` with its own transfer approximation — a different algorithm, not a different rounding — and
a gamma of 5.0 amplifies the relative error. Those four get 1e-3; everything else keeps 1e-4. This took the
test from three failures to one.

---

## 8. Open

- **ffmpeg's H.264 implicit weighting** — unexplained; §2.
- **VP9 on real hardware** — wired and unit-tested; the picture-for-picture oracle is the next task; §4.
- **AV1 hardware decode on Linux** — its own milestone; §5.
- **Encode** — `VideoEncoder`/`EncoderFactory` implementation, `sample_entry` from the driver's parameter
  sets, the frame loop driven by settings, and the docs. Also: reference depth > 1, multi-slice pictures
  (a second coded segment is refused), and five syscalls per encoded frame.
- **The FFI guard does not check parameter width or signedness** — arity is bound mechanically, but the
  Rust↔C type mapping is still a reviewer's job, and both header checks are skipped on GitHub runners where
  the headers are absent.
- **iHD writes no packed parameter sets** — it accepts the mode, returns `0x3`, emits no NAL 7/8. Whether
  that is this driver or this submission is open.
- **`cargo xtask ci` red on `crates/gpu`** — §7. Needs a decision: investigate, or record and skip.
- **libva licence** — approved 2026-10-08 for the run-time `dlopen` (ADR 0001). The ruling rests on libva
  staying behind `dlopen`; linking it would void it.