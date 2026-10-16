# Linux GPU video support: progress

Status of the VA-API work on this branch, written so a reader does not have to reconstruct it from the
branch's commits. Everything here has been measured on the development machine (Intel Arc B580, `/dev/dri/renderD128`,
libva 1.24, iHD 26.3.5); nothing is carried over from another platform unverified.

Two milestones are in flight: **decode**, essentially complete, and **encode**, which is working for H.264,
proven for HEVC and declined by the driver for VP9. Two more are wired into the decode path *and* measured
against our own decoder's pictures, one for one — **VP9 decode** (§4) and **AV1 decode** (§5).

Every number below is produced by a committed test, and each one names the fixture it came from:
`cargo test -p filmcraft-platform --test vaapi -- --nocapture` prints one line per fixture with its
tier and its picture count, and `cargo run -p filmcraft-platform --example vaparity` prints the
same numbers for a single file. **Nothing here came from a report file that is not in the
repository**, because a claim a fresh clone cannot regenerate is a claim nobody can check.

---

## 1. Hardware decode — working

A decode backend in `crates/platform/src/vaapi/`. H.264 and HEVC at 4:2:0, 8- and 10-bit, plus VP9
profiles 0 and 2 at the same depths (§4) and AV1 Main profile 0 at the same depths (§5).

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
| `vp9_p0_parity.mp4` | 96/96 (profile 0, 8-bit) |
| `vp9_p2_parity.mp4` | 96/96 (profile 2, **10-bit**) |
| `vp9_p0_segmented.mp4` | 96/96 (`SEG_LVL_ALT_Q` on all 96 frames) |
| `av1_p0_8bit.mp4` | 96/96 (AV1 Main, profile 0, 8-bit) |
| `av1_p0_10bit.mp4` | 96/96 (AV1 Main, profile 0, **10-bit**) |
| `av1_tiled.mp4` | 48/48 (8×4 tile grid, 640×500, `show_existing_frame`, four-frame units) |
| `av1_intra_only.mp4` | 96/96 (every frame a key frame, CDEF off) |
| `av1_loop_restoration.mp4` | 96/96 (WIENER and SGRPROJ both present) |

Each row is one `tests/common::FIXTURES` entry at `Tier::EveryPicture`, so
`cargo xtask fixtures platform` regenerates the file and `bit_exact_with_the_software_decoder` decodes
it, after a `reset` and reseek to every later sync sample and again after all the resets. The VP9 and
AV1 fixtures are additionally asserted **by name** in that test, so dropping one from the table fails
the suite rather than quietly leaving every document's sentence as true and the proof smaller.

The last two exist because every earlier HEVC fixture coded a prediction weight table whose every flag was
0 — so the weight record's delta recovery had only ever run with delta 0, *on hardware*. The new fixtures
drive 35 and 28 non-default luma weights, 35 and 29 non-zero luma offsets and denominators spanning 0–7
through the record. (`fade=in` is what makes x265 emit a non-default weight at all; plain `testsrc2` does
not, at any encoder effort setting.)

`vp9_p0_segmented.mp4` exists because of the same shape of gap: it is `vp9_p0_parity.mp4` plus
`-aq-mode 2` and nothing else, and it is the only fixture in this repository whose frames carry
`segmentation_enabled`. On every other VP9 fixture `SegParam[1..7]` are zero by rule, so the eight-entry
branch that `va_dec_vp9.h:270-272` describes had never been taken on hardware — a regression that copied
`SegParam[0]` into all eight entries would have passed every other test in the branch.

The five AV1 fixtures earn their place one arm each, and the reason each exists is the thing the other four
cannot reach: `av1_p0_8bit` / `av1_p0_10bit` are the two depths; `av1_tiled` is the only one that reaches
the **per-tile** slice record (`va_dec_av1.h:635`, "should be sent once per tile"), the only one carrying
`show_existing_frame`, and the only one whose sample 1 holds **four** frames; `av1_intra_only` is
`-usage allintra`, so `primary_ref_frame` is `PRIMARY_REF_NONE` on every frame and 7.20's
`setup_past_independence` runs each time, and its sequence header has CDEF off; `av1_loop_restoration` is
the only one that can tell AV1's two restoration numberings apart (§5). §5 has the rest.

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
| AV1 at 4:4:4, 4:2:2 or monochrome | `plan_frame` checks subsampling before it resolves a profile (`vaapi/stream.rs`), so a 4:4:4 stream is declined *as its chroma* even when it is profile 1. Every surface this backend creates is 4:2:0. Asserted by `an_av1_stream_this_backend_cannot_describe_is_declined_as_such` on a `yuv444p` fixture. |
| AV1 profiles 1 and 2, and 12-bit | `va_dec_av1.h:44` in the header's own words: "This AV1 decoding API supports 8-bit/10bit 420 format only." AV1 5.1.1 fixes profile 1 to 4:4:4 and profile 2 to 4:2:2 at 12 bits, so neither is reachable on its own; 12-bit is separately refused because the record's `bit_depth_idx` would say 2 and no surface here can hold it. The 12-bit arm is what reaches `av1_decline`'s own table in that test. |
| AV1 with **applied** film grain | This backend copies pictures out through `vaDeriveImage` + `vaMapBuffer` and never synthesises grain, so a driver rendering grain into `current_display_picture` would disagree with FilmCraft's own decoder on exactly the frames grain touches (`vaapi/av1.rs`, `AV1_FILM_GRAIN`). |
| AV1 large-scale tile | **A scope limitation, not a gate**: `grep -rn "large_scale_tile\|enable_large_scale_tile" crates/av1/src` returns nothing, so there is no flag to read the feature from and no anchor frames to submit. No encoder this suite can drive produces such a stream (`ffmpeg -h encoder=libaom-av1` has no LST option, checked), so it is recorded as a limit rather than as a decline that does not exist (`AV1_LARGE_SCALE_TILE`). |
| AV1 with an in-band sequence header that differs | The surfaces and the `avcC`-equivalent `av1C` were created from one sequence header; a frame that re-declares itself with a different one is an error rather than a re-seed, because the surfaces do not match what it says. |
| VP9 outside profiles 0 / 2 | 4:2:2 / 4:4:0 / 4:4:4 (profiles 1, 3), the reserved profile 4, and 12-bit. Every surface this backend creates is 4:2:0 at 8 or 10 bits, so there is nowhere to put them. |
| 10-bit H.264 | Matches the Windows backend's own decision (`media_foundation/stream.rs:279`); no fixture would prove a hardware path here. |
| HEVC with `scaling_list_enabled_flag` | Needs `VAIQMatrixBufferHEVC`, which this backend does not build. `va_dec_hevc.h` sends that buffer only when the flag is 1, so a driver default would decode a picture the stream does not contain. |
| 4:2:2, 4:4:4 (H.264 / HEVC) | Out of scope by decision. |

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

## 3. Hardware encode — working for H.264, proven for HEVC, VP9 declined by the driver

Plan: `docs/superpowers/plans/2026-10-09-vaapi-linux-encode.md`. Encode inverts the decode data flow: the
application hands the driver a *picture* and **the driver writes the slice headers**, so the bitstream is read
back out and the codec configuration is the driver's.

### The verification standard, which is weaker, and is described that way everywhere

Decode can assert 72 of 72 pictures bit-exact against ffmpeg. **Encode cannot, at any level**: a hardware
encoder and a software encoder are different algorithms, so bit-equality is unavailable — and "matches
ffmpeg" would be a meaningless claim for an encode, not a weaker one. The gate is therefore:

- the exported file **decodes with ffmpeg to the right pictures**, compared against the **source** within a
  lossy tolerance appropriate to the bitrate — never against another encode;
- the stream is **well-formed**: the `avcC` / `hvcC` parses, describes the bitstream it is handed, and
  `sample_entry()` refuses before the first picture rather than handing out an empty config record;
- size and rate control land in the neighbourhood the settings asked for;
- ffmpeg's own stderr is empty — a stream it decodes with a warning has been accepted and is not
  necessarily right.

Nothing below borrows decode's language. Where a figure is "within N/255 of the source", that is the whole
of the claim.

### What the capability probe says, and what it does not

`EncSlice: true` for H.264 Main / High / ConstrainedBaseline and for HEVC Main / Main10 on this machine,
and `EncSlice: true` for **AV1 profile 0**. VP9 reports false, and that is measured rather than inferred
(§3.6).

**No export of VP9 is declined today, because none can be asked for.** `filmcraft_export::Format` has no
`Vp9` variant — its variants are H264, Hevc, ProRes, DnxHr, Apv, Mjpeg, the image and audio sequences and
the two MXF containers — and `vaapi_export::factory` returns `None` for every format that is not H.264 or
HEVC. So there is no user-reachable path that could decline VP9 silently, and adding a VP9 decline to the
gate would mean **adding a `Format::Vp9` variant first**: a decline branch for a format the enum cannot
express is unreachable code, and a `Format::Vp9` with no encoder behind it would fail an export as
"encoder not available" while the format list claimed it was available. Both are features, not declines, and
neither is this section's claim. The measured absence above is what such a feature would be built against,
and on *this* machine it would have nothing to submit to.

### 3.0 The three spikes that established feasibility, each measured rather than assumed

1. **One I frame.** Encodes and decodes to the right picture (luma ramp, max deviation 3/255, driver average
   QP as requested).
2. **P frames predict from the picture before them.** Six frames, ramp shifted 8 columns per picture:
   **935 bytes** for the I frame then **656/307/322/196/208**. Two controls make it falsifiable — coded as
   intra the same content averages 941 bytes (2.79×), and a stale reference would give a 253/255 maximum
   against a measured ~0.3 mean, a 60× separation. The depth>1 result out of this spike is §3.4.
3. **The coded-output readback**, which was the blocker. `vaMapBuffer` returns a staging area, not the
   picture. Solved by `vaAcquireBufferHandle` → `DRM_IOCTL_PRIME_HANDLE_TO_FD` → `mmap`: every byte now comes
   from a mapping this backend issued. Output byte-for-byte identical to the previous mechanism.

### 3.1 Rate control works, and the measurement is content-dependent — which is the lesson

Measured on the oracle's own content at 320×180: **0.94 against a 2000 kbit/s target, bit-identical across
runs**, over 300 pictures. The band in the test is `0.80..=1.05` *because the measurement is 0.94*, and
`89% ± 2%` at steady state across a 6× target sweep on other content.

**What made that number legible was the content, and getting the content wrong cost two rounds of
investigation that reached the wrong conclusion.** The oracle originally encoded a **luma ramp**, whose
ceiling on this driver is **~162 kbit/s** at the quantiser floor, and asked for 8000 kbit/s. That target is
unreachable *by construction*, so:

- VBR looked as if it ignored the target — flat across a 40× sweep;
- CBR looked as if it faked one — 98.6 % trailing zeros.

**Both were the ramp.** On content that can carry the bits, VBR tracks at 89–94 % and CBR does not pad. The
full-texture ceiling on the same driver is **~6598 kbit/s**, and 2000 sits well inside it. Reverting to the
ramp fails the assertion at **ratio 0.16** — the gate was verified red before it was trusted.

A second measurement is in the test for a reason: the controller **ramps to a plateau** (2981 / 3353 / 3574 /
3591 kbit/s over 60 / 150 / 300 / 900 frames), so a 24-picture run reads the warm-up and charges the
controller for it. The run is 300 pictures because of that number and not for any other reason.

### 3.2 A key-frame interval above 16 works, by widening `frame_num` — and **8 of 11 shipped presets now export**

`vaapi_config_for` used to refuse **eight of the eleven H.264 presets FilmCraft ships**, all with the same
message: a key frame every 60 pictures, this encoder's `MaxFrameNum` is 16. The fix is
`log2_max_frame_num_minus4` **0 → 4**, so `MaxFrameNum` is **256** and the caller's interval is carried by
the counter on its own. The IDR the loop still writes at the wrap stays, but it is a *policy* reported on
`key_frame_reason`, not a codec requirement.

**MMCO is not an alternative, because it is not expressible.** `VAEncSliceParameterBufferH264` carries **no
`dec_ref_pic_marking` field and no `adaptive_ref_pic_marking_mode_flag`** — the record ends at the deblocking
fields and a `va_reserved` the header says "must be zero". libva's H.264 encoder marks references itself from
the reference lists it is given, so choosing MMCO would mean writing into a reserved field and hoping, which
is the one class of value this backend refuses everywhere else.

**And a wrap is conformant anyway.** H.264 7.4.3's Equation 7-22 forbids a *skipped* frame number still marked
used for short-term reference, and at the wrap the `UnusedShortTermFrameNum` loop's initialiser is already
`frame_num`, so the body never runs and the set is empty. With
`gaps_in_frame_num_value_allowed_flag` clear, 7.4.3 goes further and *requires* `frame_num == (PrevRefFrameNum
+ 1) % MaxFrameNum` — which is the wrap. The standard mandates it rather than forbidding it.

Measured, not argued: **200 pictures at 640×360 and 240 at 1280×720**, both a 60-picture interval, both
decoded by ffmpeg with **no warning at all** and a flat 9/255 mean absolute deviation on every picture.
`trace_headers` reads `log2_max_frame_num_minus4 = 4` and writes `frame_num` as `00010000 = 16` — **eight
bits**, which the old four-bit field could not reach — with IDRs at pictures 0, 61, 122, 183 and **none** at
16, 32 or 48. `MAX_FRAME_NUM` is 256 and the wrap run is 260 pictures.

**The gate's refusal stays a refusal**, now against 256: **250 is taken** and asserted taken (the value the
old ceiling declined), **300 still declines** and names both numbers in the message.

### 3.3 `H264Profile::Auto` exists, and it is the default

Before `Auto`, every H.264 export on Linux was a lie in one direction or the other: pick **Main** for Linux
and the preset you did not touch quietly stopped being High; pick **High** and the VA-API gate declines,
because its derived parameter sets write `profile_idc = 77` and an offered `VA_PROFILE_H264_HIGH` comes back
as a **byte-identical Main bitstream**. `Auto` is the delegation, resolved by one accessor
(`ExportSettings::resolved_h264_profile`) that the export summary calls, so **the summary prints the profile
that is in the file, not the one that was stored**: Main when the OS encoder may take the export and that
encoder writes Main (Linux + hardware encoding on + a container it can carry), High otherwise. Every encoder
is handed the resolved value — the built-in one, the MXF picture record, VideoToolbox, NVENC — which is what
makes `Auto` worth having. A named `High` is still honoured: the gate declines it, the export falls back to
the built-in encoder, and both the summary and the file say High.

**Deriving High parameter sets is not feasible here**, and that is why substituting silently was rejected:
the sequence record **carries no profile field to derive one from**, so `params::sps` writes
`profile_idc = 77` unconditionally; the derived PPS writes `entropy_coding_mode_flag = 0`, so the stream is
CAVLC with no CABAC and no 8×8 transform; and the driver answers an offered `VA_PROFILE_H264_HIGH` with a
**byte-identical Main bitstream**. An SPS claiming High over CAVLC slices and a 4×4 transform would describe
a different stream from the one encoded, and substituting Main for High without saying so would have left
the summary asserting High over a Main file — which is the defect `Auto` exists to remove.

### 3.4 Reference depth > 1 is refused, measured on both codecs — and **it does not error**

H.264 at depth 2: the render succeeds, the sync succeeds, **ffmpeg stays silent**, and every picture from the
third on is wrong — mean **7.16** rising to **35.08**. Four candidate causes were measured and cleared first,
and `pic_order_cnt_type = 2` was tried as a lead and refuted (the driver reads it; the pictures are
byte-identical).

**HEVC shows the same signature, measured on its own path rather than borrowed**: with the refusal lifted, a
depth of 2 **renders, syncs, and decodes in ffmpeg with no diagnostic**, and the mean absolute deviation
from the source ramp goes **9.1, 9.1, 16.8, 26.6, 36.9, 37.1** — the third picture, the first that reads the
second reference, is the one at 137 bytes against the 43–55 of the frames that do not. Both are refused, and
each refusal carries its own codec's measurement.

**Two agents were misled by this**, and the reason is worth naming: it is the one failure in this milestone
that produces a *valid-looking* stream. Nothing logs, nothing returns a status, and the file decodes.

### 3.5 HEVC encode works — the blocker was four packed-header buffers the path configured for and never sent

**One I frame then P frames that genuinely predict, decoded clean by ffmpeg.** Measured at **320×240** under
`VA_RC_CQP` QP 26: **I 184 bytes, then P 51 / 45 / 43 / 55 / 48** (mean 48), every picture within a mean
absolute deviation of **9.1/255** of the **source** ramp, worst 21, flat across the sequence, with the MP4
holding all six samples.

Two agents failed on this before, and both recorded the failure as *the records are wrong*. It was neither
the records nor the config attribute: **`va_enc_hevc.h:299-309` says the packed slice header's data buffer
"shall contain the `slice_header()` syntax element only, along with any start code prefix and NAL unit type
preceeding it", and the driver writes no slice header of its own** — it writes the bytes the application
supplies, then the slice data. A capture of
ffmpeg's own `hevc_vaapi` on this machine shows those bytes passed through verbatim. With none supplied, the
six-buffer submission answers `vaMapBuffer` → `VA_STATUS_ERROR_INTERNAL` at `0x5`, `0x3`, `0x2` and `0x0`
alike, while `vaRenderPicture`, `vaEndPicture` and `vaSyncBuffer` all return success. Adding the packed
sequence and slice header pairs is what produced the table above. `VAConfigAttribEncPackedHeaders` is
`SEQUENCE | SLICE` (0x5) because that is what this path supplies — **not** because it is what fixed it; the
attribute reads 0x0, 0x2, 0x3 and 0x5 alike on this driver once the buffer is there.

`slice_type` is **measured, not typed**: HEVC's header states no encoding and libva has no enum for one
(`grep -rn "SLICE_TYPE" /usr/include/va/` returns only H.264's), so I = 2, P = 1, B = 0 was established by
`trace_headers` over a stream this driver encoded and ffmpeg decodes with 0 errors. **What the driver
*accepts* is not measured**, because the submissions that would answer it did not exist when the question was
open, and the code says which half is which.

### 3.6 VP9 encode: the driver declines it, and the evidence is a file rather than a stub

**No hardware VP9 encoder on this driver.** All four `VAProfileVP9Profile0..3` are advertised, and **every one
offers `VAEntrypointVLD` and nothing else**. Asking anyway — `vaGetConfigAttributes` for every (profile,
encode entrypoint) pair, across **every attribute id the driver knows** (`vaMaxNumConfigAttributes` = 59, and
`va.h:1610-1612` requires an `attribut_list` holding *at least* that many entries, so the ids in range are
**`0..=58`, 59 of them**) — returns `VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT` for **all 12 pairs** (4 profiles ×
all **three** encode entrypoints: `EncSlice` 6, `EncPicture` 7, `EncSliceLP` **8**), with the poison left intact.

**The sweep is exhaustive and its control is in the same pass: `asked=885 refused=708 status=0=177`, and
none of the 177 is a VP9 one.** A `VA_STATUS_SUCCESS` that left the `0xDEADBEEF` poison in place happened
**zero** times in 885 reads, so this is not "the driver answered some VP9 attributes and declined others" —
it declines every id it knows while the next read in the same loop, for another codec, returns a populated
word. **177 is a status count, not a count of real answers**: only 62 of those reads carried a populated
word and the other 115 were `VA_ATTRIB_NOT_SUPPORTED`. The three controls are H.264 High, HEVC Main and
AV1 profile 0, which between them answer **21, 20 and 21 of their 59 ids**: a codec this driver really
encodes answers the subset that applies to it and
`VA_ATTRIB_NOT_SUPPORTED` for the rest, and VP9's answer is none of them. `vaCreateConfig` — the first call a
submission path would make — likewise refuses every VP9 pair while succeeding for all three controls.

**An off-by-one, and where it came from.** An earlier revision of this section swept **ids 1–58**, read
`vaMaxNumConfigAttributes` = 59 as "the last id is 58", and reported 20/19/20 "of 58". Those three counts are
right for the range swept; the range is the defect, because 59 is a *buffer size* and the ids it puts in
range are `0..=58`. The dropped id is **0**, `VAConfigAttribRTFormat` (`va.h:620`), and every control
answers it — so each control loses exactly one. Re-measured over the full range the controls answer 21, 20
and 21, and the totals move from 870 to 885. The VP9 answer is unaffected either way: **0 of 708**.

`va_enc_vp9.h` and `VAConfigAttribEncVP9` (**58**) **do** exist in the installed headers, so this is **the
driver declining a codec libva can express**, not an API gap.

**How far that generalises, which is the part a reader planning this needs:**

| Claim | Status |
|---|---|
| This driver advertises no VP9 encoder | **Measured** — 12 pairs, 59 attribute ids, `vaCreateConfig`, three working controls |
| libva fully expresses VP9 encode | **Established** from the installed headers and `nm -D` on `libva.so.2` |
| Some other VA-API driver accepts VP9 encode | **Unknown — not determinable here** |

The third row is a real *cannot-determine*, not a hedge: one driver cannot answer a question about other
drivers, and nothing in `/usr/include/va` records what drivers implement. **"VP9 hardware encode is not
available on Linux" would be false** — the API it would rest on is present and specified, `va.h:115` lists
VP9 in the encoder module index and the `va_enc_vp9.h` record set is complete. The honest scope is *this
driver*. (libva exports no per-codec symbols at all — none of its 88 text symbols mentions any codec — so
"is there a VP9 encode symbol" is the wrong question; the codec rides on the profile and entrypoint passed
to `vaCreateConfig`.)

**No new code is needed for a future driver to be picked up, because the capability check already
exists.** `vaapi::stream::supports` (`crates/platform/src/vaapi/stream.rs:202`) is the
"hardware is verified, not assumed" gate, and it is parameterised by profile **and entry point**
rather than by codec — so "does this driver offer VP9 encode" is already answered by
`display.entrypoints(Profile::Vp9(0))?.contains(&Entrypoint::EncSlice)` over the public `Display`
API (`ffi.rs:2765`), which is a live `vaQueryConfigEntrypoints` rather than a table compiled into
FilmCraft. It returns `false` for all four VP9 profiles on this machine (the `Vp9(0..3) … EncSlice:
false` rows above), and it would return `true` on a driver that advertises one. `vaprobe --caps` is
that same question printed for every profile the driver names.

**What a VP9 encode path would still need**, if another driver advertised `EncSlice`: three records
(`VAEncSequenceParameterBufferVP9`, `VAEncPictureParameterBufferVP9`,
`VAEncMiscParameterTypeVP9PerSegmantParam`), the `vpcC` sample entry, and the submission machinery — which
FilmCraft already runs end to end for H.264 and HEVC, readback included. Because encode inverts the data
flow (the driver writes the bitstream), **no VP9 encoder is needed**; what is needed is the frame header's
meaning, and the bit-exact VP9 *decode* path already supplies it: `parse_uncompressed` → `FrameHeader`,
`HeaderState` + `ReferenceState`, `Vp9State`'s eight reference slots, the pinned 316-byte
`VASliceParameterBufferVP9` with all eight segments wired and tested on hardware, and `split_superframe`.
What is left is policy — choosing qindices, a segmentation map, loop-filter levels and GOP structure, and
honouring the `bit_offset_*` handshake where rate control rewrites the driver's header.

The answer lives in `vaapi/vp9_enc.rs` rather than in ~1000 lines of correctly-measured, C-pinned records for
a codec this machine refuses.

### 3.7 AV1 encode: the records are submitted, and the driver refuses the picture record

The driver advertises `EncSlice` for `VAProfileAV1Profile0` **only** — profiles 1 and 2 answer
`VA_STATUS_ERROR_UNSUPPORTED_PROFILE` — and `vaGetConfigAttributes` returns three populated words for it
(`VAConfigAttribEncAV1` 52 → `0x05001000`, Ext1 53 → `0x0000241f`, Ext2 54 → `0x0000ffcf`) against the H.264
control's `VA_ATTRIB_NOT_SUPPORTED` for the same three.

**Six records declared and pinned against a C compile**, every one measured rather than transcribed, with
**five compiler holes** — one in every record that has one, each named as a `padding_*` field of the
C-measured width:

| record | measured `sizeof` | holes |
|---|---|---|
| `VAEncSequenceParameterBufferAV1` | 88 | 1 (bytes 21-23) |
| `VAEncSegParamAV1` | 156 | 1 (bytes 138-139) |
| `VAEncWarpedMotionParamsAV1` | 56 | 1 (bytes 37-39) |
| `VARefFrameCtrlAV1` | 4 | **0**, asserted as a C-measured gap of zero |
| `VAEncPictureParameterBufferAV1` | 1032 | 1 (bytes 538-539) |
| `VAEncTileGroupBufferAV1` | 20 | 1 (bytes 2-3) |

Plus **100 bit positions — 68 across the eight record unions above and 32 across the four
  attribute-value records `VAConfigAttribValEncAV1` (15), `…Ext1` (4), `…Ext2` (5) and
  `VARefFrameCtrlAV1` (8), all measured by running C**, and a
struct↔table join check that exists because shrinking a hole in the *struct* left `sizeof` and every
`offsetof` unchanged so all the pins passed green — the compiler re-inserts the byte. Twelve deliberate
breaks were made; eleven were caught on the first attempt, and the escape is what the join check was added
for.

**The submission path now exists, and the driver refuses the picture record.** It reaches `vaRenderPicture`
and stops. **The localisation is a one-buffer inequality, and it is narrower than "the sequence record
renders":** `vaRenderPicture` **accepts** the sequence record alone, and `vaEndPicture` then fails with
`VA_STATUS_ERROR_ENCODING_ERROR` (24) — so that call is `Err`, not `Ok`. Adding any picture record moves the
failure *earlier*, to `vaRenderPicture` itself, with `VA_STATUS_ERROR_INVALID_BUFFER` (7). **`vaRenderPicture`
is therefore not the call that refuses the picture record; the picture record is what moves the failure onto
it.** Measured two ways that agree — FilmCraft's own path (`vaRenderPicture failed for 5 encode buffers: invalid
VABufferID`) and, independently through the public API, ffmpeg's own `av1_vaapi`, which declines the same
profile on this machine: *"No usable encoding entrypoint found for profile VAProfileAV1Profile0 (32)"*.
This is asserted from outside the module by `crates/platform/tests/av1enc.rs`, which mutates in both
directions to prove it can fail.

The failure was localised by sweeping rate control {NONE,CBR,VBR,CQP,ICQ} × packed headers
{NONE,SEQ,PIC,SEQ|PIC,0x1b} × surfaces {2,3,4,8} — 80 combinations — plus 16 cumulative picture-record field
variants and 80 candidate extra buffer types. It did not move. So this is **the picture record, and nothing
else**, on a driver that advertises `EncSlice` and then refuses it.

**No AV1 bytes exist**, so there is nothing to decode and nothing to claim about picture fidelity. What the
round trip *did* produce is five bitstream bugs that no size or offset check could have seen, one of them
worth naming: `showable_frame` is *inferred* when `show_frame` is 1, and writing it cost one bit which
landed in `error_resilient_mode` and sent the decoder into a `ref_order_hint[8]` loop the specification has
no reason to read — surfacing 50 bits away as *"reference to an empty slot"*. Also `frame_type` built from
a bool cast (KEY_FRAME is 0, not 1), `max_frame_*_minus_1` written 15 bits wide instead of 16, and
`force_integer_mv` / `use_ref_frame_mvs` written where the specification infers them.

**The `wmmat` scale question this section used to call unanswerable is answered, and then made moot.**
`AGENTS.md` §2 permits the AV1 bitstream specification, and `crates/av1` is built from it: `WARPEDMODEL_PREC_BITS
= 16` (`crates/av1/src/spec_tables.rs:423`), so `default_gm_params` (`crates/av1/src/header.rs:1360`) makes
identity `[0, 0, 65536, 0, 0, 65536]`. **The scale is never needed** — `is_global = 0` reads no parameters,
and identity reads 0 of the eight `wmmat` entries. Nothing was invented; a question this file carried as open
was open because a brief said the specification was out of bounds, which it is not (§6).

**Two buffers genuinely cannot be sent, and were not faked**: `VAEncTileGroupBufferAV1` and
`VAQMatrixBufferAV1` have **no `VABufferType` in libva 1.24.0** — encode types run 21..30 then jump to 41.
Sending an invented type number yields a buffer that creates successfully and then fails inside the driver,
which reads as a record bug. Whether a one-tile-group picture needs one is **not answerable from the
installed headers**, and is recorded as unknown rather than guessed.

**Still absent**: `Format::Av1`, export wiring, and a line in `vaapi/mod.rs` — so `Av1Encoder` is
crate-internal and nothing in `tests/` or `examples/` can reach it yet. Declaring AV1 encode records is not
the same claim as encoding AV1, and this section still does not make the second claim.

### 3.8 Three libva landmines, each of which cost real time

1. **`vaQueryConfigAttributes` segfaults inside `iHD_drv_video.so`** on this libva **1.24** (gdb-confirmed on
   this branch). Use **`vaGetConfigAttributes`**, which is what every probe here resolves.
2. **A probe with no working control is not evidence.** The attribute sweep that exonerated the HEVC records
   made them byte-for-byte identical to a submission this same driver accepts and reported that the failure
   did not move. Re-capturing ffmpeg's own `hevc_vaapi` submission through the public entry points and
   comparing field by field showed **the two record sets were never equal in the first place** — so the
   conclusion "the records are not the cause" was right and the evidence for it was weaker than claimed. The
   corrected claim is the weaker, better one: *no difference between these records accounted for the
   failure*, which is what re-running ffmpeg's own records through this submission shows — they fail here
   exactly as ours do. **A comparison over a subset cannot distinguish the thing under test from the thing
   you left out of it.**
3. **An unsupported profile/entrypoint pair leaves the attribute value poisoned rather than zeroed.** libva
   returns the status and does not write the caller's `VAConfigAttrib` at all, so a `0x00000000` means "nothing
   was written here", not "nothing is set". **Gate on the returned status and poison the value you check**
   (`0xDEADBEEF` here) so "the driver never wrote this" and "the driver wrote zero" stay distinguishable.

The rate-control sweep has the same shape as landmine 2 and 3, and now **states its own precondition twice over** —
because it was wrong twice, in two different ways, and the second failure is the one worth generalising from.

**The first sweep was taken with no packed slice header**, which this driver refuses to encode without, so it
measured the **buffer set** rather than the modes. Of the **eight** modes this driver advertises for HEVC Main
(`0x14de`, expanded in the code as CBR, VBR, VCM, CQP, ICQ, MB, QVBR, TCBRC), **six failed at `vaMapBuffer`** and
two (`VA_RC_ICQ`, `VA_RC_QVBR`) **at `vaEndPicture`** — and those two failing earlier is the signature of a buffer
the driver wants and is not getting, not of a mode it dislikes. With the slice header supplied **`VA_RC_CQP` —
the mode this path names — encodes**, and the other modes were re-swept, because the earlier sweep's precondition
no longer held.

**The second sweep fixed the buffer set and still had no control.** It refused `VA_RC_CQP` by name in both columns,
so the known-good mode appeared in neither and **the sweep had no passing row at all**. That is a defect in the
*sweep*, not a property of the driver, and it is the more instructive of the two failures: **a table in which
every mode fails cannot refute anything**, because "the modes this driver refuses" and "this submission is broken"
and "this driver refuses rate control altogether" all produce the same table. A sweep needs a mode that is known
to work, measured in the same process against the same buffer set, or its failures mean nothing.

**The third sweep is the one recorded**, with `VA_RC_CQP` swept and **asserted first** as the control. Every mode
was submitted **twice** — with a rate-control record and without — which is what separates a mode failure from a
record failure. On iHD 26.3.5 / libva 1.24, six 320x240 pictures each, changing only `VAConfigAttribRateControl`:

| mode | with CBR's record | with no record | bytes mapped | ffmpeg | verdict |
|---|---|---|---|---|---|
| `VA_RC_CBR` (0x02) | encodes, mean 9.1/255 | encodes, mean 9.2/255 | yes | silent, 6/6 | works — the gate submits it |
| `VA_RC_VBR` (0x04) | encodes, mean 9.1/255 | encodes, mean 9.1/255 | yes | silent, 6/6 | works — the gate submits it |
| `VA_RC_VCM` (0x08) | encodes, mean 9.1/255 | encodes, mean 9.1/255 | yes | silent, 6/6 | **the driver runs it — and the gate now submits it**, see below |
| **`VA_RC_CQP` (0x10) — THE CONTROL** | **encodes, mean 9.1/255**, sizes `[184, 51, 45, 43, 55, 48]` | **encodes, mean 9.1/255**, same sizes | yes | **silent, 6/6** | **works — and is what the gate now submits for a constant-quality export** |
| `VA_RC_ICQ` (0x40) | `vaEndPicture` → `VA_STATUS_ERROR_INTERNAL` | the same | no | — | **refuted — the mode, not the record** |
| `VA_RC_MB` (0x80) | encodes, mean 9.1/255 | encodes, mean 9.1/255 | yes | silent, 6/6 | accepted, but see below |
| `VA_RC_QVBR` (0x400) | `vaEndPicture` → `VA_STATUS_ERROR_INTERNAL` | the same | no | — | **refuted — the mode, not the record** |
| `VA_RC_TCBRC` (0x1000) | `vaRenderPicture` → invalid `VABufferID` for 9 encode buffers | encodes, mean 9.2/255 | with: no; without: yes | without: silent, 6/6 | **fails on the record, not the mode** |

"Encodes" here is the end-to-end claim, not "a picture came back": bytes mapped out of the coded buffer, ffmpeg
**silent**, **every** picture present, every picture within the tolerance this encoder's own CQP runs measure
(mean 9.1/255, worst 20–21). **Five of the eight encode and decode.**

**ICQ and QVBR fail with and without a record**, so the failure is the mode and both are **refuted** — and the
refutation now carries weight it did not have before, because the same `vaEndPicture` call succeeded for five
other modes seconds earlier. TCBRC fails only with the record.

**`VA_RC_VCM` encodes**, at mean 9.1/255 with ffmpeg silent — which **falsifies the reason it was originally
refused**. It is **wired now**, and the second reason it carried in between is false as well: `9c75de0e` added
`BitrateMode::VideoConference`, so the settings surface *can* name it, and the per-picture cap it states is derived
rather than absent. §3.10 has the wiring and the measurement of the cap reaching the driver.

**The reason that was substituted for the original one was the useful part.** "The driver runs it but nothing above
the backend can name it" is a statement about *this tree's* plumbing, dressed in the clothes of a statement about the
driver — the same mistake `VA_RC_CQP` was refused for ("this gate submits the bitrate", true of the gate and
silent about the driver) and the same mistake the H.264 gate now makes correctly **about** the driver when it
declines VCM by name. **The distinction is between the two and it is checkable**: one is a number in a mask
(`0x14d6` for H.264 against `0x14de` for H.265, the same driver and the same query), the other is a missing
variant. A refusal must say which.

**`VA_RC_MB` encodes, and its sizes are `VA_RC_CQP`'s byte for byte** (`[184, 51, 45, 43, 55, 48]` under both a
record and none). That is a finding rather than a duplicate row: `va.h:1122-1125` says `VA_RC_MB` *"may be
combined with other RC modes, except CQP"* — a **modifier, not a base mode** — and a modifier with nothing to
modify produces the driver's unmodified output. So "works" for MB means **the driver accepts the submission**, not
that per-macroblock control happened. The combination (`VA_RC_CBR | VA_RC_MB`) is **named untested rather than
claimed**.

**`VAConfigAttribEncPackedHeaders` (10) answers `0x1f` for HEVC** — all five flags, `SEQUENCE | PICTURE | SLICE |
MISC | RAW_DATA`, measured with `vaGetConfigAttributes`. **`VA_ENC_PACKED_HEADER_SLICE` (0x4) is among them**,
which is the flag this path depends on. **AV1's answer for the same attribute on the same driver is `0x1b`, which
excludes it** — because AV1 has no slice header, it takes a frame header, and `PICTURE` is the flag that says so.
**The two answers are the evidence that this attribute is not a free choice**: same driver, same attribute,
answering differently per codec for a structural reason. Note that advertising and enforcing are separate: every
value from `0x0` to `0x5` encodes identically on this path, and what it refuses to encode without is the packed
slice header **buffer**.

### 3.9 Reachability from the UI, and what is in progress

The H.264 path is **reachable**: `register()` puts `vaapi_encoder_factory` in front of the software encoders
on Linux, behind a `Once` the way the Windows arm does it, and every decline goes through
`note_hw_encode_declined` so `perf.stats` `export.hardware` can tell a user their machine has no hardware
encode path. It is opt-in per export (`Export ▸ Hardware encoding`, off by default), because a hardware
stream differs from ours and exports are otherwise byte-identical across machines. There is **no mid-stream
fallback**: a decoder can replay samples through another decoder, an encoder cannot hand a half-written stream
to one, so a failure during an export stops it with an error.

**HEVC is reachable too**, through the same factory and the same `Once` (`11b351e`). A `Format::Hevc` export
reaches the driver on a machine whose driver offers `VA_RC_CBR`/`VBR` and whose picture is a whole multiple of
8 — 320x240 and 1920x1080 both are. Taken: a bitrate (`Cbr`, `Vbr1Pass`), **a quantiser (`ConstantQuality`)**,
one pass, progressive, square pixels, SDR, MP4/MOV. Declined, each with its reason: MXF (Annex B), two-pass,
0 kbps, HDR, non-square pixels, interlaced, fractional rate, non-8-multiple size. `Format::Hevc` reaches the
format list through a probe that creates a real session, so a machine without the encoder does not advertise it.

**`BitrateMode::ConstantQuality` is taken for HEVC, and used to be declined.** The refusal said "this gate submits
the bitrate" — a true statement about the gate, offered as though it settled a question about the driver, which it
did not. `VA_RC_CQP` is in the advertised mask, is what `HevcEncoder::at_constant_quantiser` submits, and is now
measured encoding and decoding to the source. **For HEVC this was not a small gap**: there is no software HEVC
encoder behind this one, so a user who picked "Constant quality" on an HEVC export got *encoder not available*
rather than a slower file. The export test asserts the quantiser **reaches the pictures** rather than merely
that an encoder appeared — the same six pictures exported at QP 26 and QP 40 must come out at different sizes,
which is the one directional prediction a lossy encoder's output size makes. HEVC's QP scale is its own
(`SliceQPY = 26 + init_qp_minus26 + slice_qp_delta`, H.265 7.4.7.1, so **26 is the origin, not H.264's
midpoint**); both backends number 0 (best) to 51 (worst), so the range check is shared and the value is not
reinterpreted.

**A shipped HEVC preset now exists** (`29affc43`), so this subsection's earlier statement that FilmCraft
shipped none is superseded by **§3.10**, which is the current per-preset table. The preset is conditional on
`available(Format::Hevc)` and **exports at the sequence's own rate**: the encoder writes the caller's exact
rational into the VUI of **both** parameter sets (§3.10, "#12"), so 24, 25, 30, 60 and 30000/1001 all
export at the rate asked for. That was not true at any point on this branch — the rate was fixed at 30 and
every other rate was refused by name.
`hevc_is_offered_on_this_machine_and_which_presets_it_reaches` reads the real preset list, so a preset added
later is caught rather than silently missing.

**The `hvcC` is cross-checked on every picture**, which is what makes the record trustworthy rather than merely
written: it re-reads the record's own bytes and refuses an `hvcC` that does not round-trip, lacks one complete
VPS/SPS/PPS, disagrees with its own SPS on profile, level, chroma or geometry, states a frame rate its VUI
contradicts, or whose picture's `slice_type` contradicts the sample table's sync-sample flag. The check found
two real defects in itself — it had taken the caller's geometry rather than the sample entry's own dimensions,
and its key-frame predicate was `written == 2 != key_frame` where the sound form is `key_frame && written != 2`
(an I slice in a TRAIL picture is legal HEVC; an IDR with non-I slices is an unstartable sync sample).

#### 3.9a `VA_RC_VCM` is wired for HEVC, and refused for H.264 **as a fact about the driver**

`BitrateMode::VideoConference` was named by `9c75de0e` and both gates let it fall through their `_ =>` arm to a
plain variable bit rate — so the summary printed *"Video conference (VCM), Target 20.00 Mbps, Max 30.00 Mbps"*
beside a file with **none of the guarantee the name promises**. That is the same defect class as the
`ConstantQuality` substitution fixed earlier, and it is invisible for VCM in a way it is not for a quantiser:
**VCM *does* state a rate, so the substituted file is perfectly plausible.** The cap is a ceiling on *one
picture* (`VAEncMiscParameterBufferMaxFrameSize`, `va.h:2753-2762`, in **bits**) and it is a **second misc
buffer**, not a field of `VAEncMiscParameterRateControl` — which is why a mode can be wired and still be wired
with its distinguishing control missing.

**The two gates differ, and the reason is a number rather than a preference.** This driver reports
`VAConfigAttribRateControl = 0x14de` for `VAProfileHEVCMain` and `0x14d6` for `VAProfileH264Main` — the same
driver, the same query, one bit apart, and that bit is `VA_RC_VCM`. So:

- **HEVC takes it.** `RateControl::VideoConference { target, max, max_frame_bits }` submits `0x08` plus the
  `max_frame_size` record, and `HevcBitrate::VideoConference` carries the same three numbers out of
  `hevc_config_for`. **`HevcBitrate::build` handing the ceiling to `HevcEncoder` has its own test**, because
  that is the one place the gate's fidelity and the encoder meet, and the export path cannot observe it
  (the registered factory returns a `Box<dyn VideoEncoder>`).
- **H.264 refuses it by name, and says it is the driver.** The message quotes what the mode states, both mask
  numbers, and where the mode *is* available. **This is the distinction the module history keeps getting wrong**:
  `VA_RC_CQP` was refused here on the reason that "this gate submits the bitrate" — true of the gate, silent
  about the driver — and `VA_RC_VCM` was for a while refused on "nothing above the backend can name it", which
  is a claim about this tree. A refusal must say which of the two it is, and `VaEncoder::new` refuses the
  mode for H.264 on the same measured fact, so a caller reaching past the gate is told the same thing.

**Where the cap lives, and why it is derived.** `ExportSettings` has no field for it. The ceiling is
*the target rate's share of one picture* — `target_bps * rate.den / rate.num` bits, rounded up — computed in
`hevc_export::per_picture_ceiling`. Three reasons, and the third is the one that would change the decision:

1. **Every input is already on the summary line.** The mode names a bitrate, so the summary prints the target,
   the ceiling and the frame rate, and all three are terms of the derived number. A cap the user cannot see is
   a cap the user cannot reason about.
2. **A field with no control behind it is the defect the mode was created to fix.** `VA_RC_VCM` sat unwired
   because a capability no setting could name never reaches a driver; a field reachable only from a
   hand-edited project file and a test reproduces that one layer up.
   `every_input_the_derived_per_picture_ceiling_needs_is_on_the_summary_line` asserts the *absence* of such a
   field from the serialized settings, so adding one becomes a deliberate act.
3. **Making it a real control is a `crates/ui-egui` change**, and a real one. Until that exists the derived
   number is the one FilmCraft can justify from what it was given. The cost is stated rather than hidden: a
   user cannot choose a ceiling other than the rate's own share.

**The cap reaching the driver is measured, and the measurement is red-able.**
`the_vcm_picture_cap_reaches_the_driver` holds the rate, the ceiling, the frame rate, the key-frame interval and
the six pictures fixed and changes one number. The first version compared total coded size, and **the measurement
refused that assertion**: at 320×240 the ramp is small enough that a cap changes the controller's *path* and not
only its ceiling, and the totals run 443 bytes at a 200-bit cap, 528 at 2000 and 472 at 20000 — a *tighter* cap
produced a larger stream in the middle of the range. So the assertion is on **the driver's own average
quantiser**, the low byte of each segment's status word (`VA_CODED_BUF_STATUS_PICTURE_AVE_QP_MASK`, `va.h:3897`),
which has a reason behind it that a size does not: **meeting a small ceiling needs a coarser quantiser**. Six
320×240 pictures, 2 Mbit/s target, 30 fps:

| cap (bits) | average QP per picture | coded sizes |
|---|---|---|
| 200 | 50, 51, 50, 44, 41, 42 | 148, 68, 74, 88, 28, 37 |
| 800 | 50, 44, 49, 41, 40, 36 | 148, 89, 50, 84, 37, 77 |
| 66 667 (the derived value) | 11, 12, 10, 10, 10, 10 | 264, 38, 39, 42, 43, 46 |
| 2 000 000 | 11, 12, 10, 10, 10, 10 | 264, 38, 39, 42, 43, 46 |
| 100 000 000 | 11, 12, 10, 10, 10, 10 | 264, 38, 39, 42, 43, 46 |

**The control is a plain-VBR run in the same process, asserted first**, and the three loose caps agreeing byte for
byte is what makes the tight row evidence rather than noise: the driver is deterministic for a given submission,
and a cap that does not bind does not move a byte. Removing the submission from `HevcEncoder::encode` fails the
quantiser comparison; sending the cap in the *bytes* of `VAEncMiscParameterBufferMultiPassFrameSize` instead of
the bits of this one fails the submission check. **This is not "matches ffmpeg"** and it is not a comparison with
any other encode — it is a direction the driver's own output must take, checked against the same encoder's run
with a cap it does not have to obey.

### 3.10 Every built-in preset, against both hardware gates

The table the previous sessions lost. **One row per preset in `builtin_presets()`, all 31**, and for
each one which gate is asked, what it says, and whether the refusal is right.

**How each row was obtained.** The gates already ask this question about the real preset list, so this
table reads their answers rather than re-deriving them: `crates/platform/tests/vaenc.rs`
(`what_the_export_gate_declines_and_why`) filters `builtin_presets()` by
`video_format() == Format::H264` and calls the H.264 gate `vaapi_export::config_for` on each, printing
every refusal with `eprintln!`; `crates/platform/tests/hevenc.rs`
(`hevc_is_offered_on_this_machine_and_which_presets_it_reaches`) does the same against
`vaapi::hevc_config_for`. Both were run on this machine:

```
test what_the_export_gate_declines_and_why ... preset declined by a documented rule: Vimeo 1080p Full HD: two-pass encoding
preset declined by a documented rule: Vimeo 2160p 4K Ultra HD: two-pass encoding
preset declined by a documented rule: MXF OP1a H.264: MXF carries an Annex B stream, and this backend's samples are length-prefixed
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

**8 of the 11 H.264 presets export** — 11 reaching the gate (pinned by name, not by count, in
`presets::reachability_tests`), minus the 3 refusals printed above. **Encode verification stays the
weaker standard of §3's preamble throughout: no row here claims an encoded stream matches ffmpeg,
because bit-equality is unavailable between a hardware and a software encoder.**

#### The routing rule, which decides 19 of the 31 rows before either gate runs

`job.rs` calls the registered factory with **`settings.video_format()`** — the *encoder's* format, not
the container's — and `vaapi_export::factory` answers with:

```rust
let hevc = format == Format::Hevc;
if !hevc && (format != Format::H264 || s.hardware_encoding != HardwareEncoding::Auto) {
    return None;
}
```

A `None` here is not a refusal, it is the factory never asking. So for every format with **no VA-API
encoder in FilmCraft**, neither gate is consulted and **there is no refusal string to quote**. Asked
anyway, the two gates say only that they are the wrong gate: the H.264 gate answers
`ProRes is not H.264: the H.264 gate takes H.264, and Format::Hevc goes to the HEVC gate` and the HEVC
gate answers `ProRes is not HEVC`. Those are routing messages, not capability findings, and the built-in
encoders behind those formats are exactly what those presets are for. **Nothing here is a gate waiting
to be weakened.**

One row is not settled by routing: **`MXF OP1a H.264` reaches a gate**, because
`ExportSettings::video_format()` reports the MXF *video codec* (`MxfVideoCodec::H264 →
Format::H264`), so the factory is handed `Format::H264` with `format` still `Format::MxfOp1a`, and the
H.264 gate reads `s.format.is_mxf()` and refuses. That is its real refusal, not the routing line's.

#### The table

"Gate" is `H264` = `vaapi_export::config_for`, `HEVC` = `vaapi::hevc_config_for`. Bitrates are the
target/ceiling pair the gate is handed in kbps after `ExportSettings::resolve`; the three `Match
Source` presets name a bits-per-pixel figure instead, so their numbers are what `resolve` computes at
**1920×1080 / 30 fps** (`w*h*fps*bpp/1000`, floored at 100, ceiling `target*3/2`) — they inherit the
sequence's size and rate, so those numbers move with the sequence.

| # | Preset | Format asked | Gate | What the gate says | Exports? |
|---|--------|--------------|------|--------------------|----------|
| 1 | Match Source – Adaptive High Bitrate | `H264` | H264 | — (taken) | **yes** |
| 2 | Match Source – Adaptive Medium Bitrate | `H264` | H264 | — (taken) | **yes** |
| 3 | Match Source – Adaptive Low Bitrate | `H264` | H264 | — (taken) | **yes** |
| 4 | High Quality 1080p HD | `H264` | H264 | — (taken) | **yes** |
| 5 | High Quality 2160p 4K | `H264` | H264 | — (taken) | **yes** |
| 6 | YouTube 1080p Full HD | `H264` | H264 | — (taken) | **yes** |
| 7 | YouTube 2160p 4K Ultra HD | `H264` | H264 | — (taken) | **yes** |
| 8 | Vimeo 1080p Full HD | `H264` | H264 | `two-pass encoding` | no |
| 9 | Vimeo 2160p 4K Ultra HD | `H264` | H264 | `two-pass encoding` | no |
| 10 | Social Vertical 1080×1920 | `H264` | H264 | — (taken) | **yes** |
| 11 | MXF OP1a H.264 | `H264` | H264 | `MXF carries an Annex B stream, and this backend's samples are length-prefixed` | no |
| 12 | HEVC 1080p High Quality | `Hevc` | HEVC | — (taken) | **yes, at the sequence's own rate** |
| 13 | Apple ProRes 422 HQ | `ProRes` | none | routing `None`; H264 gate would say `ProRes is not H.264: …` | no (built-in) |
| 14 | Apple ProRes 422 | `ProRes` | none | routing `None` | no (built-in) |
| 15 | Apple ProRes 422 LT | `ProRes` | none | routing `None` | no (built-in) |
| 16 | Apple ProRes 422 Proxy | `ProRes` | none | routing `None` | no (built-in) |
| 17 | Avid DNxHR HQ | `DnxHr` | none | routing `None` | no (built-in) |
| 18 | Avid DNxHR SQ | `DnxHr` | none | routing `None` | no (built-in) |
| 19 | Avid DNxHR LB | `DnxHr` | none | routing `None` | no (built-in) |
| 20 | APV 422-10 | `Apv` | none | routing `None` | no (built-in) |
| 21 | APV 422-12 | `Apv` | none | routing `None` | no (built-in) |
| 22 | MXF OP1a DNxHR HQ | `DnxHr` | none | routing `None` | no (built-in) |
| 23 | MXF OP1a ProRes 422 HQ | `ProRes` | none | routing `None` | no (built-in) |
| 24 | MXF OP-Atom DNxHR (Avid) | `DnxHr` | none | routing `None` | no (built-in) |
| 25 | PNG Sequence | `PngSequence` | none | routing `None` | no (built-in) |
| 26 | TIFF Sequence | `TiffSequence` | none | routing `None` | no (built-in) |
| 27 | BMP Sequence | `BmpSequence` | none | routing `None` | no (built-in) |
| 28 | Animated GIF 640×360 | `Gif` | none | routing `None` | no (built-in) |
| 29 | Waveform Audio 48 kHz 16-bit | `Wav` | none | routing `None` | no (built-in) |
| 30 | Waveform Audio 48 kHz 24-bit | `Wav` | none | routing `None` | no (built-in) |
| 31 | AIFF 48 kHz 16-bit | `Aiff` | none | routing `None` | no (built-in) |

**The numbers behind the "taken" rows.** Every H.264 preset in the list is `Vbr1Pass`, `Auto` profile
(resolved to Main, which is what the derived SPS writes), progressive, SDR, square pixels, and even
sided (`resolve` forces `w &= !1; h &= !1`), so the gate's own profile, field-order, pixel-aspect and
size checks are all passed by construction. The ceiling is taken as stated — 20 000/24 000, 60 000/72 000,
16 000/20 000, 45 000/54 000, 12 000/16 000 kbps — and the three adaptive presets resolve to
**12 442/18 663**, **6 221/9 331** and **3 110/4 665** kbps at 1920×1080/30. The key-frame interval is
`fps*2` (`resolve`), which is **60** at 30 fps against the encoder's `MaxFrameNum` of **256** (§3.2).

**One conditional the table does not show.** Every H.264 row is also conditional on the user's
`Export ▸ Hardware encoding` opt-in: `HardwareEncoding::default()` is `Off`, and the gate's first check
after the format is `if s.hardware_encoding != HardwareEncoding::Auto { return Err("hardware encoding is
off") }`. The preset list does not set it, so "exports" here means *under the opt-in*, which is how
§3.9 describes it and how both test suites set it (`s.hardware_encoding = HardwareEncoding::Auto`).

#### The verdicts

| # | Preset | Verdict | Why |
|---|--------|---------|-----|
| 8, 9 | Vimeo 1080p Full HD, Vimeo 2160p 4K Ultra HD | **Refused for a real reason** | See below. |
| 11 | MXF OP1a H.264 | **Refused for a real reason** | See below. |
| 12 | HEVC 1080p High Quality | **Exports, at the sequence's own rate** | The rate was a fixed 30 and every other rate was refused by name; it is now the caller's rational, written into both parameter sets. See below. |
| 13–31 | all nineteen remaining presets | **Refused for a real reason** (by routing, not by a gate) | No VA-API encoder exists for these formats in FilmCraft. Nothing is declining a capability; there is no encoder to decline with. The built-in encoders are the deliverable. |
| — | *no preset on this list* | **Ambiguous** | There are none. Every row resolved to one of the two verdicts above. |

**#8/#9, the two-pass refusals — real, and quoted from the gate.** The H.264 gate refuses before it
reads any bitrate, in `hardware_encode.rs`:

```rust
if !matches!(s.h264_pass, H264Pass::Single) {
    return Err("two-pass encoding".into());
}
if s.bitrate_mode == BitrateMode::Vbr2Pass {
    return Err("two-pass encoding".into());
}
```

and the HEVC gate spells the same refusal out: `"two-pass encoding: a two-pass export renders the
sequence once to measure it and again to encode it, and this backend is handed each picture once"`. The
backend has **no `Pass` concept at all** and is handed each picture once, so there is nothing to submit
a measured rate control to; taking the second pass's bitrate and encoding in one pass would produce a
file at a rate nobody chose with nothing wrong anywhere in it. **Left refused.** Both presets already
carry a comment saying so, and both still export — through FilmCraft's own encoder, which does take the
two passes.

**#11, the MXF H.264 refusal — real, and a property of the container.** Quoted above. MXF carries the
H.264 **Annex B** stream with in-band parameter sets (ST 381-3); the VA-API path has no switch to that
form and hands the muxer four-byte length-prefixed samples, which is the MP4 form. **No choice in
`presets.rs` can fix this**, because it is the container and not the codec; the only "fix" available
would be to stop shipping MXF H.264, which is a real broadcast deliverable that works today through
FilmCraft's own encoder. Left refused, already commented at the preset.

**#12, the HEVC rate — fixed, and the fix turned out to be two defects of the same arithmetic.**

The *gate* always accepted any rate; the refusal happened later, in the encoder, and there were **two**
places writing a fixed 30 and each applied **H.264's** convention, which HEVC does not have:

- `sequence_record(size, width, height, keyframe_interval)` took **no frame-rate argument** and wrote
  `vui_num_units_in_tick: 1` with `vui_time_scale: 2 * DEFAULT_FRAME_RATE_FOR_LEVEL` where the constant is
  30 — **60**, so `time_scale / (2 * num_units_in_tick)`, which is **H.264's** arithmetic. HEVC's is
  `time_scale / num_units_in_tick`: a 30 fps stream so written declares **60 fps in its own SPS**.
  `at_rate(fps)` rebuilt the record without passing a rate, so the rebuild kept the fixed 60.
- **The same doubling was in the VPS**, which nothing had mentioned. `hevc_params::vps` took a whole
  `frame_rate` and wrote `time_scale = 2 * fps` for the same reason — so the SPS said 30 and the VPS said
  60, and **the two parameter sets of one stream disagreed by a factor of two.**
  This is the half that reaches a reader: ffmpeg's HEVC parser prefers the VPS, and on this path's own
  24 fps elementary stream it read **`48/1`**. Patching only the VPS's `time_scale` 48 → 24 — every other
  bit untouched — made it read **`24/1`**, which is what makes this a measured finding and not an
  inference. (`filmcraft_hevc::params::Vps` stops after profile/tier/level and cannot see the timing, so
  the VPS's bits are read directly; the offsets are ffmpeg's own `trace_headers` output for a VPS this
  path wrote — flag at 155, `num_units_in_tick` at 156, `time_scale` at 188 — and the tests pin them.)
- The `hvcC` carried the caller's rate and `check_hvcc_against_the_picture` compared the two. **It made
  the same mistake as the writer**, so it endorsed the doubling: a 30 fps export passed with an SPS
  declaring 60.

**What landed.** `vui_timing(rate)` computes `(num_units_in_tick, time_scale)` once, from the caller's
rational, and **both parameter sets are given that one pair** — a single argument threaded from
`at_rate_exact` through `parameter_sets`, so the VPS and the SPS cannot state two rates. The cross-check
uses HEVC's arithmetic and cross-multiplies (`scale == avg * units`) rather than dividing, which makes it
**stricter**: it refuses doubled pairs the old check accepted. `hvcC` states no rate at all for a
fractional one (30000/1001), because `avgFrameRate` is a 16-bit integer and 0 is ISO/IEC 14496-15's own
"not carried here" — the bitstream and the track's timescale carry it exactly.

**Verified red before trusted.** Re-introducing `time_scale = 2 * fps` in the VPS fails the unit test
(`left: (1, 60), right: (1, 30)`) and the integration test (`24 fps: the VPS states (1, 48) and the SPS's
VUI states (1, 24)`), and dropping `hevc_mp4toannexb` from the test's own extraction fails it again with
`the elementary stream carries NAL types [9], which has no VPS (32) or SPS (33)` — see below.

#### A test that was measuring a guess, and is now measuring a bitstream

`every_rate_the_gate_takes_states_that_rate_in_the_bitstream_and_in_the_container` extracted the
elementary stream with `-c copy -f hevc`. **The parameter sets live in the `hvcC` and in no sample**, so
that produced a **525-byte file holding one access-unit delimiter and nothing else** — ffprobe read it
with a diagnostic (*"missing picture in access unit with size 525"*) and an `r_frame_rate` of
**`1200000/1`**, which is not this path's rate doubled or mangled but what a reader falls back to for a
stream that states nothing. `-bsf:v hevc_mp4toannexb` is what prepends the extradata; with it the file
carries VPS, SPS, PPS and the slices, and ffprobe reads the rate the test meant to assert. **The
assertion itself is unchanged** — it still says the bitstream states the rate the caller asked for — and
what changed is that there is now a bitstream to say it.

#### One thing this table could not measure, now that it can

- **The HEVC row is now measured, not read off the gate's checks.** `hevc_presets()` returns
  empty unless `available(Format::Hevc)`, which is `vaapi::hevc_available` registered by
  `filmcraft_platform::register()`, so **`tests/hevenc.rs` calls `register()` first** — without it the
  preset is not in the list the test reads and it reports `0 of 0`. `hevc_is_offered_on_this_machine_and_
  which_presets_it_reaches` asserts the list is non-empty for exactly that reason.
- **§3.9's "No shipped preset changed status" is now stale** for the same reason: it says FilmCraft
  "ships no HEVC preset at all", and `29affc43` added one. The sentence above is the corrected claim.

#### Where the table is pinned, so it cannot rot

`presets.rs` carries four tests (`presets::reachability_tests`) that assert this partition **by name**,
because a count survives a rename or a re-routed container and would then be an assertion that cannot
fail: which 11 presets reach the H.264 gate, which two name a second pass, which one puts H.264 in an
MXF container, and that no HEVC preset is listed without a probe. They are checked against the real
`builtin_presets()` output, and were confirmed able to fail (a one-character change to an expected name
turns the first red with the two lists printed side by side).

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
record's `sizeof`). `cargo xtask ci` now runs the
platform tests unoptimised as well, so a profile divergence cannot hide a bug again.

**What that sentence got wrong, and how it stands now.** When this was first written the guard covered two of the
three historical ABI defects. `struct drm_prime_handle` — the one whose missing third field made the ioctl number
wrong, so the call *succeeded* while leaking a dma-buf per encoded picture and double-closing a descriptor libva
owns — was checked only by two hand-typed literals, and §6 of this file is the record of that style of verification
reading as rigorous and being wrong. It is now a `_Static_assert` against a real compile of `<drm/drm.h>`, like every
other record, and its four failure modes are demonstrated rather than asserted: widening the record, reordering
fields at the correct size (invisible to the old literals), a wrong request number, and a missing field.

**The guard has since grown a blind spot of its own, which is worth stating because `sizeof` cannot see it.** A
compiler hole changes no size, so `sizeof` passes over one. `VASegmentParameterVP9` decoded 24/24 bit-exact for five
commits — that was the `vaparity` run on the old 24-picture `vp9_p0.mp4`, before the 96-picture fixtures of §4
existed — while handing the driver two uninitialised bytes per record, and it was found by construction rather than
by symptom. Every record submitted to a driver is now also checked for unnamed holes, pinned as *gaps* rather than
sizes, each named as a `padding_*` field of the C-measured width. Shrinking a gap does not always move `sizeof` —
the compiler re-inserts the byte — which is why the pins ask for the gap and not the size, and why the AV1 encode
records (§3.7) carry a struct↔table join check in addition: twelve deliberate breaks were made and the one that
slipped through was a shrunk hole in the *struct* that left every `sizeof` and every `offsetof` unchanged.

---

## 4. VP9 decode — wired, and measured against hardware

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

**Proven on hardware, picture for picture, at both depths.** Three fixtures, all
`Tier::EveryPicture`, all bit-exact against FilmCraft's own decoder, which is itself bit-exact against
ffmpeg: `vp9_p0_parity.mp4` (profile 0, 8-bit) 96/96, `vp9_p2_parity.mp4` (profile 2, **10-bit**) 96/96
and `vp9_p0_segmented.mp4` 96/96. Each is a `tests/common::FIXTURES` entry, so
`cargo xtask fixtures platform` regenerates it from the committed table and
`bit_exact_with_the_software_decoder` decodes it — whole run, a `reset` and reseek to every later sync
sample, and a full pass again after all the resets. **The 10-bit claim is the one five documents
(`README.md`, `ROADMAP.md`, `crates/platform/README.md`, `docs/architecture.md`,
`docs/adr/0001-platform-ffi.md`) make and that no committed test used to check**; it is now a
`FIXTURES` row asserted *by name* in that test, so dropping the fixture fails the suite rather than
quietly weakening five sentences.

**The segmented fixture is the one that was missing, and it was missing in a way that mattered.** All
twenty-four pictures of the first VP9 run had `segmentation_enabled == 0`, so the eight-entry branch of
`va_dec_vp9.h:270-272` — *"otherwise, all eight entries should be valid"* — was never taken; only the
degenerate one was. A regression that carried the frame's own quantizer into `SegParam[1..7]` would have
passed every other test and produced a plausible wrong picture.
`vp9_p0_segmented.mp4` is `vp9_p0_parity.mp4` plus `-aq-mode 2`, and all 96 of its frames carry
`SEG_LVL_ALT_Q` on segments 0, 1, 2 and 4 with four distinct deltas
(`a_vp9_fixture_that_really_uses_segmentation`). The 316-byte `VASliceParameterBufferVP9` is then pinned
on both arms — every `segment_flags`, every `filter_level[4][2]` transcribed from VP9 8.8.1, and all
four quant scales read out of 8.6.1 — by
`vaapi::picture::tests::the_vp9_slice_record_carries_every_segment_it_should`.

**What is *not* claimed.** The segmentation deltas were checked against an independent implementation of
6.2.11 (GStreamer's `GstVp9Parser`) rather than against ffmpeg, because ffmpeg 9 no longer exposes VP9
segmentation as frame side data — `AV_FRAME_DATA_SEGMENTATION_PARAMS` is gone from `libavutil/frame.h`.
That is a change in the oracle, not in the claim: both decoders agree picture for picture on the same
files, which is the property that matters for this table.

---

## 5. AV1 decode — wired, and measured against hardware

**AV1 Main (profile 0) at 4:2:0 decodes on Linux, at 8 and 10 bits, and `plan_frame` plans it.**
All five fixtures are `Tier::EveryPicture` and every picture is bit-exact against FilmCraft's own
decoder through the real media stack: `av1_p0_8bit` 96/96, `av1_p0_10bit` 96/96, `av1_tiled` 48/48,
`av1_intra_only` 96/96, `av1_loop_restoration` 96/96.

**What the enabler actually was, correcting the framing this section used to carry.** It said the
blocker was that `crates/av1/src/header.rs:478` declares `pub(crate) fn parse` over private types —
a module-visibility wall, the same one VP9 had in `crates/vp9/src/lib.rs`. That was never quite
true, and `crates/av1` needed no such change:

| what the record builder needs | state before the record existed |
|---|---|
| `crates/av1/src/lib.rs:16` | **already `pub mod header;`**, and `:32` already re-exported `FrameHeader` |
| `HeaderState` (`header.rs:458`) | **already `pub struct`** |
| `header.rs` | 224 `pub` items against 3 `pub(crate)` — the module was already almost fully public |
| `BitReader` (`bits.rs:6`) | `pub(crate) struct` — **this**, plus a widened `parse` and a small `pub use` block, is the entire enabler |
| the spec constants the record names | `pub const` in the **private** `mod spec_tables` — re-exported |

So the wall was a `pub` on one struct and a `pub use` on one list, not a module boundary. The VP9
precedent does not transfer, and `crates/av1` should not get `pub mod bits;` for it.

**What the record needed that no header documents.** Two of its members are guesses the installed
headers do not settle, and both were wrong first:

- `lr_unit_shift` inverts to 256 → shift **2**, 64 → shift 0 — *the bigger the unit, the larger the
  shift*, from `size = 256 >> (2 - shift)` at `header.rs:1156`. Submitted backwards,
  `av1_loop_restoration.mp4` decoded **18 of 96**.
- `yframe_restoration_type` carries the **specification's** `FrameRestorationType` (NONE/WIENER/
  SGRPROJ/SWITCHABLE = 0/1/2/3), not the bitstream's two-bit field. `va_dec_av1.h:606-618` documents
  none of the five members; `va_enc_av1.h:855-863` documents two of them, and reading its silence on
  the other three is what points the wrong way. Submitted the wire numbering: **18 of 96** again.

**Both score the same 18, and the 18 are the frames that restore nothing.** The other four fixtures
have loop restoration off and would have passed under either reading, so without
`av1_loop_restoration.mp4` the suite could not have chosen. That is the whole lesson, and it is §6's
second entry rather than a new one.

**A fixture that cannot fail is worse than no fixture.** This section's Q1 note claimed a 2×4 tiled
fixture scored "68 of 96" under the rejected tile-number indexing. It scores **96 of 96** — a grid
whose columns are all one width makes both readings identical. The real measurement needs an 8×4
grid of differing columns and rows, where the readings score **48 of 48 against 0 of 48**, which is
what `av1_tiled.mp4` now encodes and what
`av1::tests::the_tiled_fixture_is_the_one_that_can_tell` enforces.

**Scope is Main profile (0), 4:2:0, 8 and 10 bits, and every decline outside it is in §2's table** with the
header line it comes from — 4:4:4 / 4:2:2 / monochrome, profiles 1 and 2 and 12-bit, applied film grain,
large-scale tile, and an in-band sequence header that differs from the one the surfaces were created for.

**Still open, and recorded as open rather than claimed:** what a *full* 64-entry
`width_in_sbs_minus_1` does (the record's array is 63 wide and the header says the last tile is computed,
so a 64-tile axis is the one shape it does not cover; no fixture this suite can generate produces one — a
640×500 frame has 10 × 8 superblocks, and libaom's `-tile-columns` will not exceed 32 at 8 bits); and
whether a rotated or affine `gm_params` survives the `wmmat` copy, since no fixture carries
non-identity global motion — `-enable-global-motion 1` changes nothing on `testsrc2`, measured.

The architectural asymmetry below still explains *why* any of this was needed.

---

## 6. Findings recorded because they were wrong here first

Nine conclusions on this branch were wrong and were caught by measurement rather than review. They are listed
because each was plausible, and each cost real time. The last three came out of **briefs written for this
branch**, and all three reached committed code or a design decision before the agents receiving them refuted
them — which is why they are here and not merely in a commit message:

- **A clean-room rule quoted from memory, and wrong.** A brief for the AV1 encode task stated that the AV1
  bitstream specification was unavailable and outside `AGENTS.md` §2, on the strength of a note this file
  carried. §2 permits it in as many words — *"the VP9 and AV1 bitstream specs"* (line 110) — and `crates/av1`
  is built from it. The task therefore spent its premise on an open question (`wmmat`'s scale) that was
  answerable: `WARPEDMODEL_PREC_BITS = 16`, identity `[0, 0, 65536, 0, 0, 65536]`, and the scale then turned
  out to be **never needed**, because `is_global = 0` reads no parameters at all. **A prohibition quoted
  second-hand is worse than no prohibition** — it manufactures a dead end and looks like diligence.
- **A clause that does not exist.** An H.264 "8.2.3.2.1" deriving separable per-position weights was proposed
  and acted on. Zero occurrences of that clause number — or of `lumaWeightL0`, `weightA`, `weightB`,
  `PocDiffB`, `w1`/`w2` — across six ITU editions. A fabricated clause reads exactly like a real one.
  **Specificity is not evidence.**
- **A fixture that could not fail.** §5's own note claimed a 2×4 tiled AV1 fixture scored "68 of 96"
  under the rejected tile-number indexing of `width_in_sbs_minus_1`. It scores 96 of 96 — a grid whose
  columns are all one width makes both readings produce identical arrays, so the number in the note
  could not have come from that fixture. **A passing fixture proves only what it can distinguish**, and
  the fix was to encode a grid that separates the hypotheses (48 of 48 against 0 of 48) plus a test
  that fails if a future ffmpeg makes the grid uniform again. A number no test can reproduce is a
  number nothing checked.
- **A grep that missed fields.** `RefPicList0`/`RefPicList1` are CamelCase; searching for `ref_pic_list`
  found nothing and produced a confident, wrong conclusion that the record had no reference lists at all.
- **A desync bug that was not there.** `luma_offset_lX` was suspected absent from HEVC's `pred_weight_table`.
  It is present — clause 7.3.6.3, in the same `for(i)` body as `delta_luma_weight`, and present since v1.
- **A struct miscounted while verifying.** `struct drm_prime_handle` was confirmed as 8 bytes from five
  lines of a header; line 933 of `/usr/include/drm/drm.h` holds a third field (`__s32 fd`). The
  verification read as rigorous and was wrong.
- **A libva constant that does not exist.** `VAProfileH265Enc` was named as the profile to ask for when
  HEVC encode was first planned. `grep -rn "VAProfileH265Enc" /usr/include/va/` returns **nothing**, and there
  is not even a separate encode profile: **libva reuses the codec profile with an encode entrypoint** —
  `VAProfileHEVCMain` (17) and `VAProfileHEVCMain10` (18) answer `[1 VLD, 6 EncSlice]` on this driver, which
  is why both decode and encode ride on the same constants. The same trap sits next to VP9, where
  `VAConfigAttribEncVP9` **does** exist (`va.h:1065`) and invites the matching mistake. A constant that
  reads like the surrounding ones is not thereby in the header.
- **An enum value read off a bitfield as 3.** A brief reported `support_cdef_channel_strength = 3` in the AV1
  encode feature word and called it `VA_FEATURE_REQUIRED`. **`va.h:398` defines `VA_FEATURE_REQUIRED` as 2,
  so 3 is not a valid `VA_FEATURE_*` value at all**, and the real value is **1** — CDEF is *supported*, and
  since **no** AV1 encode feature is required, CDEF in the sequence record is a choice rather than an
  obligation. Wrong twice over, and the second wrongness is the dangerous one: a decoder that cannot tell 2
  from 1 gets HEVC right by luck (its `amp` and `cu_qp_delta` genuinely read 2) and AV1 wrong on purpose.
- **A mechanism described as a clamp that actually refuses.** A brief described the key-frame-interval
  ceiling as clamping an interval down to what the encoder could carry. The gate **refuses**, naming both
  numbers, because "tell the caller" and "shorten it silently" are the only two honest answers and silently
  shortening would hand back a file a sixth larger than it was promised. An earlier draft of the gate's own
  comment claimed it clamped; that comment was wrong and now says so.

Also worth recording: a per-task review by fresh subagents **missed** the MMCO 3/6 spec deviation and the
HEVC reference-lists-built-once-per-picture bug, both of which a single CodeRabbit pass found. The
second silently produced wrong pictures on multi-segment HEVC and survived four review rounds. And a probe
with no working control is not evidence at all — see §3.8, where two separate exonerations had to be
downgraded from "the records are not the cause" to the weaker "no difference between these records
accounted for it".

---

## 7. `crates/gpu` — was red here, and the cause was not ill-conditioning

`cargo xtask ci` was red on `crates/gpu` for reasons that looked like pre-existing hardware trouble. It was
not, and both symptoms are fixed. The earlier diagnosis was wrong in kind: it read a sample-coordinate
residual as an unstable shader.

### `invert` on the alpha channel diverged, and the cause was a bilinear tap

`fx_tests::gpu_effects_match_cpu_exactly` — case `invert [("channel", Choice(4)), ("blend", Float(20.0))]`,
which allows **no** flipped value — failed on the alpha channel. **The compositor's `load4` bilinear tap let `f`
be a small nonzero number on a 1:1
draw**: `sp` comes from the vertex stage and the rasteriser interpolates it a few ULPs off each texel centre,
so `mix` blended in a fraction of the *neighbouring* texel. A fully transparent pixel then picked up a few
ULPs of alpha, and `Invert{channel: 4}` computes `rgb / a` on it — at 67×41 the leak is 3.8e-6 of a texel,
`1/a` is 2.6e5, and the product is **CPU 0.0, GPU 1.0**: a fully opaque pixel where the reference is fully
transparent, from a single transparent texel beside an opaque one. It is a sample-coordinate residual, not a
texture-format loss — the leak is the same fraction of neighbours 1000× apart.

Measured by having `load4` return its raw fraction, over **17 configurations** (1:1 from 48×32 to 4096×4096,
minifying and magnifying draws, rotations of 13/30/45 degrees): **exactly zero at every power-of-two
extent**, and at most **0.853 × f32::EPSILON × the source extent (1.25 ULP)** otherwise. The window is sized
on the **source extent**, not on this pixel's coordinate — the rasteriser interpolates across the whole
triangle, so the error is set by the attribute endpoints and does not shrink towards the origin.

The fix is **snapping the sample coordinate to the nearest texel centre** in `composite.wgsl`, with
`CENTRE_SNAP = 2^-22`. Comparing against the distance to the *nearest* centre rather than testing the
fraction makes it two-sided, which matters because the residual is signed: **5.7 % of the samples of a
1920×1080 1:1 draw (2384 of 41472) land on the far side**, where `f` wraps to ~1 and a one-sided test blends
in almost the whole previous texel.

Blast radius measured rather than assumed, because the tap is the compositor's under every transform, crop,
flip and resample: over 11 transforms at 192×108, **8 of the 10 genuine resamples are bit-identical** (non-
integer downscale, 1/3 downscale, 2.4× upscale, rotations of 7/30/45 degrees, a shear, half-, quarter- and
eighth-pixel offsets), and every transform that differs does so by **≤ 0.5 ULP of the Rgba16Float
accumulator** — 1/8 of one 8-bit code value. Only the 1:1 control changes, which is the leak being removed.
`one_to_one_draw_copies_the_source_exactly` pins it with black-and-white vertical stripes compared through
`to_bits()`, so the statement is "exactly 0" rather than "within tolerance", and it is red without the
change at every non-power-of-two size.

Also kept from the earlier diagnosis, which was right about these: a per-op tolerance for the effects whose
GPU path is a power function (`gamma_correction`, `levels` via `powf(g)`; `tint`, `extract` via
`linear_to_srgb`). A GPU evaluates `pow(x, g)` as `exp2(g·log2(x))` with its own transfer approximation — a
different algorithm, not a different rounding. **All 76 effect cases now match the CPU with no tolerance or
budget moved, and the worst of them improves from 3.84e-4 to 5.71e-6.**

### The suite segfaulted, and `--test-threads=1` was not the fix

**5/5 runs at 6 threads, 0/3 surviving at every thread count ≥ 2, serial never.** Under gdb the fault is in
`wgpu::Instance::default()` itself: `desired_extensions` called with a null `ash::Entry` while another thread
sits in `vkEnumerateInstanceExtensionProperties`, then a jump to address 0. Concurrent Vulkan instance
creation on this driver, **before any of our code runs** — not a limit, not flakiness, not ours.

Fixed by making the instance and device process-wide behind a `OnceLock`, with the `Instance` held separately
so it cannot be torn down under the device. **15/15 threaded runs now survive, where the same binary was
1/15.**

### Still open: the same defect in `crates/golden`

`crates/golden/tests/golden.rs:742` builds its **own `wgpu::Instance` per test** and requests a device on the
same physical GPU — the shape that faulted above, in a second binary, unfixed. It is not the same test run,
so it is not urgent, and it is recorded rather than claimed to be harmless: the fault is a race in driver
initialisation and this is the pattern that provokes it.

---

## 8. Open

- **ffmpeg's H.264 implicit weighting** — unexplained; §2.
- **VP9 segmentation beyond `SEG_LVL_ALT_Q`** — every frame of every fixture available here sets
  `SEG_LVL_ALT_L`, `SEG_LVL_REF_FRAME` or `SEG_LVL_SKIP`; no libvpx invocation reachable from this
  machine's libvpx produces them. `segment_flags` and the per-segment `filter_level` arms are pinned
  with those three features **clear**, and their active paths are covered by unit tests that build a
  header directly (`a_vp9_profile_this_backend_does_not_decode_is_an_error_not_a_record` and the
  `vp9_segment_param` tests in `vaapi::picture`). A real-world stream with any of the three would be
  the fixture that closes this; §4.
- **AV1, two scope limits that are recorded rather than resolved** — §5. A *full* 64-tile axis, where the
  record's `width_in_sbs_minus_1` is 63 wide and the header says the last tile is computed (no fixture this
  suite can generate has 64 tiles in one axis: a 640×500 frame has 10 × 8 superblocks, and libaom's
  `-tile-columns` will not exceed 32 at 8 bits), and whether a **rotated or affine `gm_params`** survives the
  `wmmat` copy — no fixture carries non-identity global motion, because `-enable-global-motion 1` changes
  nothing on `testsrc2`, which is a property of the source and not of the encoder settings. Large-scale tile
  is a third recorded limit, and it is a limit rather than a gate because there is no flag to read it from;
  §2.
- **HEVC encode, what is still refused** — §3.9. Each refusal carries its measurement: reference depth > 1
  (measured on HEVC's own numbers, and the same silent-divergence signature H.264 shows), multi-slice
  pictures, B pictures, `VA_RC_ICQ` and `VA_RC_QVBR` (**refuted**: both fail at `vaEndPicture` with and
  without a rate-control record, while five other modes succeeded through the same call in the same
  process), TCBRC (fails on the record, not the mode), a non-8-multiple picture size (no crop exists in
  HEVC's record to ask for one with), HDR, MXF, and non-square pixels. **`VA_RC_VCM` is no longer in
  this list**: it is wired, and §3.10 records the cap reaching the driver. **Constant-quality is no
  longer in it either**: `VA_RC_CQP` is measured working and is wired, with the export test asserting
  the quantiser reaches the pictures. **No HEVC preset ships**, so nothing here is reachable from one
  until one exists.
- **`VA_RC_MB` combined with a base mode** — `va.h:1122-1125` calls `VA_RC_MB` a modifier that "may be
  combined with other RC modes, except CQP", and the sweep submitted it alone. A bare `0x80` produces output
  **byte-identical to `VA_RC_CQP`'s**, which is what a modifier with nothing to modify looks like. The
  combination (`VA_RC_CBR | VA_RC_MB`) is **named untested rather than claimed**; measuring it is a separate
  sweep, and asserting anything about it now would repeat the exact mistake the sweep history records.
- **H.264 encode, what is still refused** — §3. Each refusal carries its measurement: reference depth > 1
  (measured on both codecs, §3.4), multi-slice pictures (a second coded segment is refused; reachable at 70
  pictures, refused unmeasured), B pictures (the driver reports it *could* — `VAConfigAttribEncMaxRefFrames`
  answers (3, 1) — and emits none over a 40-picture run spanning two mid-stream IDRs and two `frame_num`
  wraps), `Vbr2Pass`, HDR, MXF, a fractional frame rate, and five syscalls per encoded frame.
- **AV1 encode** — the submission path exists, is declared in `vaapi/mod.rs`, and is exercised from
  outside its module by `crates/platform/tests/av1enc.rs`. **The driver refuses the picture record**, and
  that is localised: `vaRenderPicture` accepts the sequence record alone and `vaEndPicture` then fails,
  while adding the picture record moves the failure earlier onto `vaRenderPicture` itself. §3.7.
  `Format::Av1` is deliberately **not** declared, and the reason is recorded where the next implementer
  reads it: a format variant with no encoder behind it fails an export as "encoder not available" while
  the format list claims it is available, and `job.rs` picks the container brand with `is_h26x()`, which
  AV1 is not. `SampleEntry::av1` does exist now, so the muxer is ready.
- **The FFI guard, and what CI does with it** — arity is bound mechanically, holes are pinned as gaps and
  each field's width and signedness is checked too, and `.github/workflows/abi-guard.yml` now runs
  `cargo test -p filmcraft-platform --lib` on every push to `main`/`release`, on pull requests that touch
  the platform crate, and weekly. 27 of that target's tests are the guard — the three `abi` modules that
  read or compile against the installed headers — and
  `the_abi_guard_ran_on_this_machine_or_announced_that_it_did_not` **fails** rather than skipping when the
  headers are not installed, so a green run means the guard ran rather than that nothing went wrong. Two
  facts about the runner, both measured here against the real headers, are why the workflow is not just
  `apt-get install libva-dev`: the guard needs **libva ≥ 2.24**
  (`VAConfigAttribEncVP9` first appears in 2.24's `va.h`; 2.17.0 and 2.22.0 both fail, and `ubuntu-latest`
  ships 2.23.0), so `libva-dev` 2.24.1 is pinned from Debian's package pool and only its `usr/include/va`
  is copied onto the runner; and it needs **`/usr/include/drm`**, which Debian's and Ubuntu's
  `libdrm-dev` do not provide (they install `/usr/include/libdrm/*.h`), so the workflow symlinks it.
  **What CI still does not run is `cargo test`** — the workspace, the integration targets and the
  fixtures are out of scope here and remain local gates.
- **iHD writes no packed parameter sets** — it accepts the mode, returns `0x3`, emits no NAL 7/8. Whether
  that is this driver or this submission is open. (This is the *decode-side* statement and is separate from
  the encode-side finding that the driver writes no slice header of its own either, §3.5.)
- **`crates/golden/tests/golden.rs:742` builds a `wgpu::Instance` per test** — the exact pattern that faulted
  `crates/gpu`'s suite at every thread count ≥ 2, in a second binary, unfixed; §7.
- **libva licence** — approved 2026-10-08 for the run-time `dlopen` (ADR 0001). The ruling rests on libva
  staying behind `dlopen`; linking it would void it. CI installs libva's **public headers** and nothing
  else — no library, no binary, no code, nothing vendored into this tree — so a C compiler has a header
  to measure against. That is a test oracle and not a dependency, and it changes nothing about the
  ruling.
---

## 9. Where this branch stands

Written as a hand-off, so a reader does not have to reconstruct the state from ~100 commits.

### Decode: done, and measured rather than asserted

**H.264, HEVC, VP9 and AV1**, all **bit-exact against ffmpeg picture for picture** — 14 fixtures at
`Tier::EveryPicture`, every number regenerated by a fresh clone from the committed `FIXTURES` table.
That reproducibility is the point: a claim a clone cannot regenerate is a claim nobody can check.

### Encode: two of four codecs, and the ceiling is stated everywhere

| codec | state |
|---|---|
| **H.264** | reachable — **8 of 11 presets** export; `Cbr`, `Vbr1Pass` and `Vbr2Pass`-free paths only |
| **HEVC** | reachable at **any frame rate including 30000/1001**, with a shipped preset |
| **AV1** | path complete and tested from outside its module; the driver refuses the picture record |
| **VP9** | this driver declines; muxer support and the capability check exist so a future driver needs no new plumbing |

**Encode verification is materially weaker than decode's, and that asymmetry is written at the top of
§3 rather than left for a reader to infer.** A hardware and a software encoder are different algorithms,
so bit-equality is unavailable at any level between them. Nothing on this branch says "matches ffmpeg"
about an encode.

### What changed most recently, and each was a measurement that overturned something

- **The encode-loss oracle was measuring a colour-range conversion.** `-pix_fmt gray` is *full-range*, so
  it expanded the limited-range luma by ~9/255 while converting. The figure being read as encoder loss
  was that expansion. Lossless now reads **0**; every tolerance moved *tighter*, and a flatness bound was
  replaced by a not-drifting assertion rather than loosened to fit.
- **An H.264 constant-quality export was encoded at a bitrate nobody chose** while the summary printed
  `, QP n`. Both gates now refuse it by name, quoting the quantiser.
- **The HEVC frame rate limit was a defect in two places.** The *VPS* carried a doubled rate — H.264's
  `time_scale / (2 × num_units_in_tick)` convention — and ffmpeg's HEVC reader prefers the VPS, so a 24 fps
  stream read as 48. HEVC now exports at any rate.
- **A rate-control sweep had no passing row at all.** It refused `VA_RC_CQP` in both columns, so the
  known-good mode appeared nowhere and seven modes stayed "untested rather than refuted". Re-swept with a
  control, `CBR`, `VBR`, `VCM` and `CQP` are now measured working and wired; `ICQ` and `QVBR` are refuted.

### Honest limits

- **Reference depth > 1 silently diverges** on this driver, on both H.264 and HEVC. It does not error — it
  renders, syncs, decodes cleanly, and is *wrong from the third picture on*. It is refused because a wrong
  picture that looks right is worse than an error.
- **The VCM per-picture cap is now user-selectable** (`a677e712`, `f75aac4b`, `29dd04b8`). `ExportSettings::max_frame_size`
  carries it in **bytes**; the driver's `max_frame_size` is in **bits** (`va.h:2757-2758`) and the single conversion is
  `checked_mul(BITS_PER_BYTE)` at `hevc_export.rs:563`. `None` still derives `per_picture_ceiling`, so a project written
  before the field keeps its meaning, and the summary names which of the two is in force. **What replaced it as the known
  gap: the derivation exists twice.** `crates/ui-egui/src/panels/export_mode.rs` carries its own copy of the arithmetic,
  because `filmcraft-platform` is only a *dev*-dependency of `filmcraft-ui-egui` and the layering gate forbids a real one.
  The two are pinned together by `export_ui.rs:270`, which has already caught one real disagreement.
- **`cargo fmt --check` was red on 3 pre-existing hunks** in `ffi.rs`, from the `config_attributes` commit and
  unrelated to any finding. Those hunks were then absorbed into `c01f46df` by the agent that fixed them rather
  than landing as a commit of their own. **Re-verified clean on this branch at `95cbdde0`** (queue item 5, now done),
  and it has stayed clean through the record-verification commits. Whether the hunks belonged in their own commit is a
  presentation question, not a correctness one.
- **One line can silently weaken the ABI guard.** `CK_CLASS` requires `__typeof__`; without it the union
  and pointer *shape* checks become passes rather than failures. Every other line in that file fails loudly
  when broken; this one does not.

### Position with respect to upstream

Upstream PR **#408** covers the same scope under a parallel architecture the maintainer has already asked to
be rebuilt onto main's `vaapi::va` + `Accel`; it is `CONFLICTING`, has never run CI, and its author has not
pushed since 2026-10-09. This branch does **not** hook the L0 codec crates, and its contribution is the
reproducible verification rather than a second FFI.

**Settled 2026-10-16: no further outreach.** Coordination was offered on #408 and went unanswered for four
days. The standing decision is that upstream merges the work or it does not; **this branch's goal is that it
works here**, so effort goes to usability rather than to persuading anyone. The one upstream artefact that is
already prepared and waiting is branch **`hevc-encode-records`** (`79de26a1`), branched from `upstream/main`:
the HEVC encode records with their ABI pins, declarations only, deliberately overlapping nothing that PR #1256
declares. It stays un-PR'd until #1256 lands, because #1256's H.264 encode port owns the session those records
would plug into.

### Standing queue — continue from here without asking

Per the global `AGENTS.md`: **autonomy is the default.** Proceed, dispatch and loop without checking in; the only
things that warrant stopping are destructive or irreversible, security-sensitive, side effects outside this
worktree, or genuinely unknowable. Nothing below is any of those.

**Serialization rule.** Two agents must never edit the same file, and — the failure mode that actually bites —
**only one agent may touch `crates/` at a time.** Two concurrent `cargo fmt` runs over one crate reformat each
other's half-written edits and lose work. Work that is genuinely disjoint (YAML-only, or a different worktree)
runs in parallel; Rust work queues.

| # | work | files | state |
|---|---|---|---|
| 1 | **VCM per-picture cap becomes user-settable** — `ExportSettings::max_frame_size: Option<u32>`, `None` still derives. Unit trap: `va.h:2757` says the driver's `max_frame_size` is **in bits** while every other rate on the struct is kbps. | `crates/export/src/lib.rs`, `crates/export/src/settings_tests.rs`, `crates/platform/src/vaapi/hevc_export.rs`, `crates/ui-egui` | **done**, `a677e712` + `f75aac4b` + `29dd04b8` |
| 2 | **Pin every third-party action to a SHA** — every `uses:` in `.github/`. Precedent already in `windows-arm64.yml`. | `.github/workflows/*.yml` | **done**, `a4e23195` |
| 3 | **Pin the ABI-guard runner image** — `runs-on: ubuntu-latest` is the last floating input in the job that compiles the C oracle. | `.github/workflows/abi-guard.yml` | **done**, `19392dbd` |
| 4 | **Fix the stale count** — `vp9_enc.rs` said H.264 answers "20 of 58" attribute ids. Re-measured over the range the header actually puts in range, `0..=58`, it is **21 of 59**: `vaMaxNumConfigAttributes` returns 59 as a *buffer size*, and `va.h:1610-1612` requires an array holding at least that many, so the old sweep dropped id 0 (`VAConfigAttribRTFormat`) that every control answers. HEVC 20, AV1 21; totals 885/708/177; VP9 still 0 of 708. | `crates/platform/src/vaapi/vp9_enc.rs`, `docs/vaapi-progress.md` | **done**, `423d751a` |
| 5 | **Re-verify `cargo fmt --check`** on a clean tree and record the result in *Honest limits* above. | — | **done**, `95cbdde0` |
| 6 | **A regression test for the `CK_CLASS` line.** Without `__typeof__` the union and pointer *shape* checks silently become passes rather than failures — every other line in `abi_guard.rs` fails loudly when broken, and this one does not. | `crates/platform/src/vaapi/abi_guard.rs`, `abi_tests.rs` | queued behind #1 |
| 7 | `stash@{0}` — debug scaffolding from a blocked readback, exported to `.tmpwork/stash0.patch` so nothing is lost. | — | **needs the go-ahead**: destructive, and the one item here that is |

**Do not do these.** Do not rebase this branch — the attempt stopped at commit 5 of 104 because upstream's 519-line
split `ffi.rs` and our 922-line monolith do not merge, and it re-imposes an architectural decision the maintainer
already made. Do not fold item 4 or item 6 into an unrelated commit. Do not break-test with `git checkout`,
`git stash` or `git reset --hard`; **use file copies under `TMPDIR=$PWD/.tmpwork`** — that has destroyed work on
this branch twice. `/tmp` is full; `TMPDIR` must be set.
