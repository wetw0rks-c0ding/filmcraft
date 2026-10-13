# VA-API HEVC encode on `main` — a minimal, reviewable PR

Status: **plan only**. Nothing here is implemented. Every file, line and number below was read at
`upstream/main` (`cee1feb1`) or measured on this machine today (2026-10-13); file:line citations are
to `upstream/main` unless stated otherwise.

---

## 1. The finding that decides everything: `Accel` cannot express encode — and must not be changed to

### 1.1 No, it cannot. Evidence.

`crates/platform/src/vaapi/mod.rs:32-59` gives `Accel` exactly five methods:

| method | why it cannot carry an encode |
|---|---|
| `surfaces() -> &[VASurfaceID]` (`mod.rs:34`) | codec-neutral and reusable |
| `decode_h264(..)` (`mod.rs:37-43`) | names an H.264 *decode* record |
| `decode_hevc(..)` (`mod.rs:46-52`) | names an HEVC *decode* record |
| `read(index) -> Result<VideoFrame, String>` (`mod.rs:54`) | returns **decoded NV12/P010 planes**, not coded bytes |
| `fill(index, from)` (`mod.rs:58`) | its own doc (`mod.rs:55-57`) says it is the stand-in for a reference picture a decode never produced |

There is no method that puts a picture **into** a surface (encode input), none that returns a coded
buffer's bytes (encode output), and none that waits on a buffer rather than a surface. Encode needs
all three, and all three are the *inverse* of what `read`/`fill` do.

The objects behind the trait are decode objects too, not merely decode-shaped views of
codec-neutral ones:

- `va.rs:301` and `va.rs:313` — `Session::on` hard-codes `ffi::VAEntrypointVLD` in **both** the
  `vaGetConfigAttributes` probe and the `vaCreateConfig`. The config and context are VLD objects.
- `ffi.rs:73` — the only entry-point constant in the whole module is `VAEntrypointVLD: VAEntrypoint = 1`.
  `VAEntrypointEncSlice = 6` (`/usr/include/va/va.h:558`) is not declared.
- `ffi.rs:82-85` — the only buffer types declared are `0` (picture parameter), `1` (IQ matrix),
  `4` (slice parameter), `5` (slice data). `VAEncPictureParameterBufferType = 23` and
  `VAEncSliceParameterBufferType = 24` (`va.h:2063-2064`) are absent.
- `ffi.rs:75` — the only config attribute declared is `VAConfigAttribRTFormat = 0`.
- `va.rs:24-53` — `Api` loads 26 entry points and none of them waits on a **buffer**;
  `vaSyncBuffer` and `vaQuerySurfaceStatus` are not loaded.

So an `encode_hevc` method on `Accel` would need, behind it, a config at a different entry point,
two buffer types that do not exist in `ffi.rs`, four config attributes that do not exist, and a
buffer sync. That is not a method, it is a session.

### 1.2 But the trait does not have to change, because `Accel` is not the layer the encoder sits at

`mod.rs:30-31` states what `Accel` is: *"The hardware a front end ([`h264::Front`], [`hevc::Front`])
drives"*. It is the **decode front end's** view of libva — a seam that exists so
`h264::Front<A: Accel>` (`h264.rs:50`) and `hevc::Front<A: Accel>` (`hevc.rs:57`) can be driven by a
recording stand-in in tests (`h264.rs:7`). The FFI boundary on main is `vaapi/ffi.rs` + `vaapi/va.rs`,
exactly as `vaapi/mod.rs:11` says: *"The FFI is in [`ffi`] (declarations) and `va` (calls); the rest of
this module is safe code."*

Main already contains the precedent for a second entry point at the *same* layer, and it has **no
trait at all**:

- `crates/platform/src/nvenc/` — `nvenc/ffi.rs` (declarations), `nvenc/session.rs` (calls),
  `nvenc/device.rs` (the device's lifetime), and then safe code. `nvenc/export.rs:30` is
  `struct NvencEncoder { enc: Nvenc, … }` — a **concrete** backend type. No `Accel`, no generics.
- `va.rs:121` `Display`'s fields are private and `va.rs:154` `shared()` is private, so an encode
  session **must live in `va.rs`** to share the one display — the same constraint commit `30ff11e6`
  ("VA-API decoders share one display instead of opening one each") already satisfies for decode.
- `nvenc/export.rs:213-216` leaves the gap open in so many words:
  ```rust
  if format == Format::Hevc {
      // H.265 through NVENC is Windows-only for now; on Linux this backend takes H.264 only
      return if cfg!(target_os = "windows") { hevc_factory(w, h, rate, s) } else { None };
  }
  ```
- `lib.rs:72` — Linux **already** registers an encoder factory
  (`filmcraft_export::register_encoder(nvenc::export::factory)`), so the encode front end exists on
  Linux. A VA-API HEVC factory registers beside it.

**Conclusion: `Accel` is left byte-for-byte unchanged.** The PR adds a *concrete* sibling
`va::EncSession` in `va.rs` and a safe `vaapi/hevc_encode.rs` that is **not** generic over `Accel`,
mirroring `nvenc/session.rs` + `nvenc/hevc.rs`. The only edit to `Accel`'s surroundings is one
sentence in `mod.rs:30-31` saying what the trait is for (the decode front ends) and that encode does
not use it. This is not "a second architecture"; it is the architecture `nvenc` already is.

The alternative — extending `Accel` with `encode_hevc` and friends — is rejected on the evidence:
it would put encode records and an encode session behind a trait whose only other implementor is a
test double for decoding, and it would force every future encode codec (VP9, AV1, H.264) through a
trait that has nothing to generalise over. If the maintainer prefers the trait route, the minimum
honest version is a *separate* `EncAccel` trait with `upload`, `encode_*`, `coded_bytes` — but that
buys nothing over a concrete `EncSession` and costs a third abstraction nobody else uses.

---

## 2. The seam, and what already exists on main to sit in it

| question | answer on `upstream/main` | citation |
|---|---|---|
| Is there an encode front end on Linux? | **Yes** — `register_encoder(nvenc::export::factory)` | `lib.rs:72` |
| Does the encode front end support HEVC on Linux? | **No, deliberately** — `return … else { None }` | `nvenc/export.rs:213-216` |
| How does the H.265 *format* become available? | `register_format_probe(Format, fn() -> bool)`, consulted by `available()` | `export/src/lib.rs:1016`, `1048-1053` |
| Can two factories coexist? | `Vec<EncoderFactory>`, newest first, `find_map` — first that answers wins | `export/src/lib.rs:779`, `job.rs:207-213` |
| Does HDR need anything? | No: with no `register_hdr_probe`, `hdr_output` tone-maps HEVC to 8-bit SDR | `job.rs:197-205` |
| Is an `hvcC` builder already on main? | **Yes** — `nvenc::hevc::hevc_config(vps, sps, pps, size, bit_depth)`, parses the SPS with `filmcraft_hevc::params::Sps` and **refuses** a profile / chroma / depth / size mismatch | `nvenc/hevc.rs:73`, `pub mod hevc;` at `nvenc/mod.rs:36` |
| Does the export crate need changing? | **No.** `BitrateMode` is `Cbr \| Vbr1Pass \| Vbr2Pass \| Crf` — no VCM, no constant quality | `export/src/settings.rs:97-108` |
| New dependencies? | **None.** `filmcraft-platform` already depends on `filmcraft-bitstream`, `-codecs`, `-export`, `-frame`, `-hevc`, `-isobmff`, `-time` | `crates/platform/Cargo.toml:23-33` |

`BitrateMode` is the scope lever. `Cbr → VA_RC_CBR` and `Vbr1Pass → VA_RC_VBR` are both measured
working (below); `Vbr2Pass`, `Crf` and everything else are refused by name. That keeps `crates/export`
— a crate with a UI, presets and serialised settings — out of a PR that is about a driver.

`nvenc::hevc::hevc_config` lives under `nvenc/`, which is an odd home for a codec record a VA-API
encoder calls. **This is a reviewer choice, stated rather than hidden:** (a) call it and say so in
the PR description, with "move it to `filmcraft_hevc`" as a named follow-up (my recommendation — it
is `pub`, tested, and moving it touches NVENC tests for no functional gain); or (b) move it to
`filmcraft_hevc::params` in this PR, which adds a fourth crate to the diff and a second thing to
review. The plan assumes (a).

---

## 3. Proposed PR

**Title:** `Linux: VA-API HEVC (H.265) hardware encoding — Main 8-bit 4:2:0 SDR, one reference`

**Body, first line:** *Closes the "HEVC encode on Linux" / "VA-API encode" gap named in
`docs/gaps.md` and `docs/roadmap.md`, by filling the slot `nvenc/export.rs:213` leaves open for
`Format::Hevc` on Linux. `Accel` is unchanged: it is the decode front ends' view of libva, and
`nvenc` has no equivalent trait at all.*

### File list

| # | file | kind | est. | must? | what |
|---|---|---|---|---|---|
| 1 | `crates/platform/src/vaapi/ffi.rs` | modify | **+300** | **MUST** | `VAEntrypointEncSlice = 6`; `VAEncPictureParameterBufferType = 23`, `VAEncSliceParameterBufferType = 24`; `VAConfigAttribRateControl = 5`, `EncPackedHeaders = 10`, `EncInterlaced = 11`, `EncMaxRefFrames = 13`, `EncSliceStructure = 15`; the `VA_RC_*` values; `VA_ENC_PACKED_HEADER_*`; `VAEncPackedHeaderSequence = 1` / `Slice = 3`; `HEVC_LAST_PICTURE_EOSEQ/EOSTREAM`; `VA_PADDING_HIGH`; `vaSyncBuffer` + `VA_TIMEOUT`; the four `#[repr(C)]` records and their `*_bits` modules. Module header (lines 1-2, 26) gains `va_enc_hevc.h`. **No behaviour.** |
| 2 | `crates/platform/src/vaapi/abi_tests.rs` | modify | **+130** | **MUST** | sizes/alignments, every `offsetof`, the five holes *as gaps*, and the bit position of every field the encoder writes — against a C compile of `/usr/include/va/va_enc_hevc.h`. The module note (lines 1-6) records the header generation. |
| 3 | `crates/platform/src/vaapi/va.rs` | modify | **+250** | **MUST** | `Api` gains `sync_buffer: Option<…>` (one optional symbol). `EncSession` — config at `VAEntrypointEncSlice` with attributes RTFormat / RateControl / PackedHeaders / Interlaced, `count` surfaces, context, coded buffer, upload image — plus `Drop`, `upload`, `encode`, `coded_bytes`, `vendor`. Codec-agnostic: no HEVC in this file. |
| 4 | `crates/platform/src/vaapi/hevc_encode.rs` | **new** | **+820** | **MUST** | Part 1 (step 4): `HevcParams` — VPS/SPS/PPS written with `filmcraft_bitstream::BitWriter`, each parsed back with `filmcraft_hevc::params::{Vps,Sps,Pps}::parse` and every field compared with the record it came from; the VPS/SPS rate-agreement test; the `slice_header` writer, parsed back with `filmcraft_hevc::slice::SliceHeader::parse` **before it is submitted**. Part 2 (step 5): the frame loop at reference depth 1. ~600 code, ~220 test. |
| 5 | `crates/platform/src/vaapi/export.rs` | **new** | **+450** | **MUST** | `config(format, w, h, rate, s) -> Result<Config, String>`; `factory`; `hevc_available()` + `warm_hevc_probe()`; `VaHevcEncoder: VideoEncoder`. ~330 code, ~120 test (the refusal table, modelled on `nvenc/export.rs:290-414`). |
| 6 | `crates/platform/src/vaapi/mod.rs` | modify | **+20** | **MUST** | `pub mod export;` / `pub mod hevc_encode;` (Linux-gated), "encoding" in the module header, and one sentence on `Accel` saying it is the decode front ends' view. **`Accel` itself: no change.** |
| 7 | `crates/platform/src/lib.rs` | modify | **+12** | **MUST** | `register_encoder(vaapi::export::factory)`, `register_format_probe(Format::Hevc, vaapi::export::hevc_available)`, `vaapi::export::warm_hevc_probe()` inside the existing `#[cfg(target_os = "linux")]` block (`lib.rs:105-134`); crate header (lines 1-12) gains a sentence. |
| 8 | `crates/platform/tests/vaapi_hevc_encode.rs` | **new** | **+300** | **MUST** | Six 320×240 pictures from a synthetic ramp → coded bytes → **FilmCraft's own HEVC decoder** → pictures within a stated deviation of the ramp; `ffmpeg -v error -xerror` silent. Skips without a driver or ffmpeg. |
| 9 | `crates/platform/tests/vaapi_hevc_export.rs` | **new** | **+300** | **MUST** | `hevc_exports_play_everywhere` — the `nvenc_hevc_export.rs` (318 lines) template. Serialised behind a `Mutex` (the encoder registry and counters are process-wide). |
| 10 | `crates/platform/README.md` | modify | **+50** | **MUST** | "Hardware H.265 (HEVC) encoding (Linux, VA-API)", modelled on the Windows/NVENC section at lines 191-270. |
| 11 | `docs/hardware-parity.md` | modify | **+2** | **MUST** | "HEVC Main 8-bit … **Ours Linux**: no" → "yes (VA-API)" (line 39), plus one measured line beside the existing NVENC PSNR note (line 45). |
| 12 | `docs/gaps.md` | modify | **±1** | **MUST** | line 113 drops "HEVC encode on Linux" and "VA-API encode". |
| 13 | `docs/roadmap.md` | modify | **±1** | **MUST** | HW row (line 76) "Have" gains VA-API HEVC encode. |

**3 new files, 10 modified, ≈ +2,340 lines** — 2,055 of them in `crates/platform`, 285 in docs.
Roughly one sixteenth of the 37,849-line branch, and in main's idiom rather than the branch's.

### What a reviewer could reasonably reject as scope

- **Step 5 (P frames / reference depth 1)** — the single biggest reduction available. Stopping after
  step 4 gives an IDR-only encoder: a real decodable bitstream, ~450 fewer lines, and no reference
  rotation, POC bookkeeping or reconstruction pool. The cost is honest and stated: every picture is
  an IDR, so the file is far larger than the bitrate asks for, and that is **not** something to
  merge. Offered so the choice is the reviewer's, not an accident of ordering.
- **Step 9 (the export test)** — "the unit test in step 8 already proves pictures come back". Correct,
  and it is the cut a reviewer is most likely to ask for. It is kept MUST because
  `nvenc_hevc_export.rs` is the file that established the bar on main, and a PR that does not reach
  that bar invites the comparison.
- **Steps 10-13 (docs)** — "the docs are somebody else's PR". Reasonable for a README-only version;
  not reasonable for `docs/gaps.md:113` and `docs/roadmap.md:76`, which would otherwise keep listing
  as a gap something this PR just built.
- **The `hvcC` via `nvenc::hevc::hevc_config` choice** (§2) — a reviewer may prefer the move.

---

## 4. The steps, in order

Each is small enough to review on its own, and each leaves the tree green.

1. **`ffi.rs`: declare the encode side of libva. No behaviour.** The four records as plain
   `#[repr(C)]` data with `*_bits` modules, matching the convention already in `ffi.rs:163-190` (`seq_bits`, `pic_bits`)
   (`seq_fields: u32` + `pub mod seq_bits { pub const … }`). No `unsafe`, no `#[allow(unsafe_code)]`
   — a file carrying the allowance without needing it is a file whose next edit could use it. Gate:
   `cargo xtask layers`, `cargo clippy -p filmcraft-platform`, `cargo test -p filmcraft-platform abi`
   (unchanged so far — nothing to fail).

2. **`abi_tests.rs`: pin every new declaration.** A C program compiled against
   `/usr/include/va/va_enc_hevc.h` prints `sizeof` / `alignof` / `offsetof` and, for each bit-field
   set, sets one member to all ones and reads `__builtin_ctz` / `31 - __builtin_clz` back. **Holes are
   asked for as gaps** (the difference of two `offsetof`s minus the member's own width), because
   `sizeof` is blind to a hole and a bit-field declared one position too high agrees with C on size,
   on every offset and on every hole, and produces *a picture* rather than an error. Record the
   header generation in the module note. Gate: `cargo test -p filmcraft-platform abi`.

3. **`va.rs`: an encode session, codec-agnostic.** `Api` gains `sync_buffer` as an **optional**
   symbol (`nm -D /usr/lib/libva.so.2` today: `vaSyncBuffer` present; `vaQueryBufferTypes` **absent**,
   so do not call it). `EncSession::new` takes the shared display from `shared()` — never a second
   one — configures at `VAEntrypointEncSlice`, creates `count` surfaces and a context, and one coded
   buffer created with `data = null` (a buffer of this kind with initialisation data is an error to
   iHD, not something it ignores). `upload` goes through a `VAImage` + `vaPutImage`; `encode` is
   begin / render / end + `vaSyncBuffer`; `coded_bytes` maps the coded buffer. Every buffer made in
   `encode` is destroyed after `vaEndPicture` returns, and a begun picture is always ended, so the
   context stays usable on a failure. Gate: the existing `cargo test -p filmcraft-platform vaapi`
   must still pass unchanged — nothing in `Accel` moved.

4. **`hevc_encode.rs` part 1: the parameter sets and the slice header.** `BitWriter` from
   `filmcraft-bitstream` (`write_bits`, `write_ue`, `rbsp_trailing`, `escape_rbsp`) writes VPS, SPS
   and PPS; the derived sets go into one `VAEncPackedHeaderSequence` data buffer **with the PPS**, as
   ffmpeg's own `hevc_vaapi` does on this driver. `slice_header` writes the packed slice header.
   **Every one of these is parsed back with FilmCraft's own parsers — `filmcraft_hevc::params`
   (`Vps`, `Sps`, `Pps`) and `filmcraft_hevc::slice::SliceHeader`, the same parsers the decode path
   trusts for real streams — and each field compared with the record it was written from.** The slice
   header is parsed *before* it is submitted, not after. This is the original blocker and the
   highest-value 600 lines in the PR.

5. **`hevc_encode.rs` part 2: the frame loop, at reference depth one.** Records → buffers, the packed
   header framing, POC, the reconstruction-surface rotation, readback, and
   `check_output_against_the_records` (the slice header in the coded picture's NAL type, `slice_type`,
   PPS id and reference indices, parsed back out of the driver's bytes). `references > 1` is
   **refused in the constructor**, with the measurement in its doc comment as the reason (§6, risk 3).
   Gate: unit tests in the file, all pure.

6. **`tests/vaapi_hevc_encode.rs`: the picture comes back.** Six 320×240 pictures from a synthetic
   ramp, one I then five P, through the real `EncSession`. The coded bytes are decoded by
   **FilmCraft's own HEVC decoder**; the mean absolute deviation from the ramp is printed and asserted
   against a stated bound; `ffmpeg -v error -xerror` must be silent. Skips cleanly without a driver
   or without ffmpeg (the pattern at `tests/vaapi.rs:10-11`). **This is the merge gate for step 5.**
   *A reviewer who wants the smallest possible PR stops here: the encoder works, nothing is wired into
   Export.*

7. **`export.rs`: the gate, the factory, the probe.** `config()` refuses anything that is not
   `Format::Hevc`, MP4/MOV, progressive, square pixels, one pass, `Cbr`/`Vbr1Pass`, and a size that
   is a whole multiple of 8 — **HEVC's sequence record carries no crop**, so a size that is not a
   multiple of `MIN_CB` cannot be expressed at all and is refused rather than rounded. `factory`
   calls `note_hw_encode_session` / `note_hw_encode_declined` and follows NVENC's split exactly: no
   HEVC encoder at all → `None` (the export's own "encoder not available", `nvenc/export.rs:281-287`);
   an encoder that cannot take *these* settings → `Some(Err(Unsupported(…)))` naming the reason,
   because HEVC has no software encoder to fall back to. `sample_entry()` is available from
   construction — the writer owns the parameter sets, so the `hvcC` is complete before the first
   picture — via `nvenc::hevc::hevc_config`. Tests: the refusal table and hostile sizes/rates
   (`nvenc/export.rs:290-414` is the template).

8. **`lib.rs` + `mod.rs`: register it.** In the existing Linux block. `register_encoder` inserts at
   the front of the factory list (`export/src/lib.rs:898`); `nvenc::export::factory` returns `None`
   for `Format::Hevc` on Linux, so there is no contention, and the VA-API factory returns `None` for
   every other format, so H.264 is untouched. `warm_hevc_probe()` on its own thread, under
   `catch_unwind`, as `nvenc/mod.rs:170-185` does.

9. **`tests/vaapi_hevc_export.rs`: the export plays everywhere.** The `nvenc_hevc_export.rs` bar, item
   for item: a software H.264 export of the same frames as the reference; our own decoder on the HEVC
   file; **worst luma PSNR > 48 dB**; counters exactly `(frames, 1 session, 0 declined)`;
   `ffprobe` says `hevc` / `Main` / `hvc1` / `yuv420p`, `bt709` `bt709` `bt709` `tv`, the size,
   `nb_frames`, `r_frame_rate`, `start_time` within 1e-3; keyframes at the requested distance;
   `ffmpeg -xerror` decodes it and says nothing.

10. **Documentation.** README section, the parity cell and one measured line, `docs/gaps.md:113`,
    `docs/roadmap.md:76`. Every number quoted in them is one this PR measured.

11. **The full gate.** `cargo xtask ci` on Linux; re-run the ABI generation and record the header
    generation in the PR description; put the **re-measured** driver answers and picture sizes in the
    description, not the branch's.

---

## 5. The verification story, honestly

**Decode and encode are not comparable and must not be blurred.** Decode on main is asserted
bit-exact against our software decoders over 14 fixtures, every number reproducible from a committed
table (`tests/vaapi.rs:1-11`). That is a genuinely stronger standard than anything available here.
**There is no bit-equality for an encode at any level** — the hardware encoder and a software encoder
are different algorithms — so **"matches ffmpeg" is not a claim this PR can make and must not try
to.**

Four claims, strongest first. The first three are exact; the fourth is lossy, and says so.

1. **ABI — exact.** Every new declaration agrees with a C compile of `/usr/include/va/va_enc_hevc.h`:
   size, alignment, every field offset, every hole as a gap, every bit position. This is the same
   standard decode holds, and it is verifiable on any machine with the headers and a C compiler.

2. **Well-formedness — exact.** Every parameter set and the packed slice header are parsed back with
   **FilmCraft's own HEVC parsers** — `filmcraft_hevc::params` and `filmcraft_hevc::slice`, the same
   parsers the decode path trusts for real streams — and each field compared against the record it was
   written from. The slice header is checked *before* submission. This catches the class of bug that
   parses and is wrong: the bitstream-restriction block written outside its flag, a PPS id three
   fields out of step, `slice_qp_delta` in the wrong chain. It is **not** evidence about the driver's
   internals, and it is not offered as such.

3. **Self-consistency — exact.** `hvcC` against the coded pictures, on every picture: the sample
   entry's SPS is re-read with `filmcraft_hevc::params::Sps` and the pictures' slice headers with
   `filmcraft_hevc::slice::SliceHeader::parse`, and a disagreement is an error. The record the writer
   produced is the record the driver was handed, so this is a check that the two parameter sets state
   one rate and one profile — the class of defect that produced a doubled frame rate which parsed
   cleanly and decoded to twice the rate.

4. **Pictures — lossy, and the only claim about the encoder.** The exported file is decoded by
   **FilmCraft's own HEVC decoder** and compared to a software H.264 export of the same frames:
   **worst luma PSNR, printed**, asserted above 48 dB. Plus `ffmpeg -v error -xerror` decoding the
   whole file with **empty stderr**, and `ffprobe`'s independent description (`hevc` / `Main` /
   `hvc1` / `yuv420p`, `bt709` in all three, `start_time` within 1e-3, the keyframe list).
   **Note the asymmetry and do not hide it: a picture that "looks close" is not a picture that "is
   right", and 48 dB is a tolerance, not a proof.** What makes it evidence rather than vibes is that
   it is the tolerance *main already asserts* for its own hardware HEVC encoder
   (`nvenc_hevc_export.rs:159`), and that a wrong stream fails it loudly rather than passing quietly.

Plus the **refusals measured, not assumed**: the PR prints the driver's own answers
(`vaQueryConfigEntrypoints(VAProfileHEVCMain)`, `VAConfigAttribRateControl`,
`VAConfigAttribEncPackedHeaders`, `VAConfigAttribEncMaxRefFrames`, `VAConfigAttribEncSliceStructure`)
so a reviewer can see the target existed and what it offered.

**What would convince a reviewer, concretely:** the CI log with the four printed driver numbers and
the PSNR; `hw_encode_stats()` reading `(72, 1, 0)`; `tests/vaapi_hevc_export.rs` passing; and the
observation that `tests/vaapi_hevc_encode.rs` proves FilmCraft's decoder reads what FilmCraft's
encoder wrote, which is a property no amount of ffmpeg comparison would give.

---

## 6. Risks, most likely failure first

1. **The branch's picture sizes and deviations must not be copied.** 184 / 51 / 45 / 43 / 55 / 48
   bytes and "9.1/255" were measured **against the branch's records**, in the branch's file layout.
   They are a property of *those records on this driver*, not of the driver. If the port's sequence or
   picture record differs in one bit the numbers change. They are re-measured in step 6 and the
   re-measured numbers are what the code comments and the PR description carry. This is the most
   likely way the PR ships a wrong committed table, because the wrong numbers are already written
   down and look authoritative. **The driver answers below *were* re-measured today and are safe to
   carry.**

2. **The four `VAConfigAttribType` numbers and the two `VABufferType` numbers are unverified against
   the header generation main pins.** Measured here on **VA-API 1.24.0**
   (`va.h:625, 694, 714, 738` and `va.h:2063-2064`); `abi_tests.rs:1-6` says main's table was
   generated against **VA-API 1.23 / libva 2.23**. If `VAConfigAttribRateControl` is not `5` on 1.23,
   `vaCreateConfig` is handed the wrong attribute: the encoder may silently encode nothing, or worse,
   succeed with a configuration nobody asked for. A ten-minute check in step 2 eliminates this
   entirely; skipping it is the single cheapest way to lose a week.

3. **Reference depth > 1 silently diverges, and this driver invites it.** With the depth-2 refusal
   lifted on the branch, a depth of 2 **renders, syncs, and decodes in ffmpeg with no error at all**,
   and every picture from the third on is wrong: mean absolute deviation from the source goes
   9.1, 9.1, **16.8, 26.6, 36.9, 37.1** (branch `hevc_submit.rs:979-986`). The third picture is the
   first to read a second reference, 137 bytes against the 43-55 of the ones that do not. The driver
   *asks* for it — re-measured today, `VAConfigAttribEncMaxRefFrames` is **`0x00030003`**, i.e.
   l0 = 3, l1 = 3. So the refusal must be a **constructor-level refusal whose doc comment carries
   this measurement**, not a comment and not a default. A later edit that derives depth from
   `keyframe_distance` re-opens the worst bug class in this whole area.

4. **"Not extending `Accel`" is a taste argument, and the maintainer may not accept it.** §1.2 is the
   strongest case available (nvenc has no trait; `EncSession` must live in `va.rs` to share the
   display; `nvenc/export.rs:213` left the slot open on purpose), but a reviewer who reads `Accel` as
   *the* seam this project uses will see a concrete `EncSession` as a second architecture — the exact
   objection in #408. Mitigation: put the `nvenc` comparison in the PR's first comment, keep
   `EncSession` inside `va.rs` beside `Session` on the same `Display` and `Api`, and make the one
   `Accel`-doc sentence an explicit statement of scope. If it is still rejected, the fallback is a
   `EncAccel` trait (§1.1) — same code, different wrapper, so nothing is lost but a day.

5. **The poisoned-attribute rule is not "gate on `VAStatus`, never the value".** Measured today: a
   `vaGetConfigAttributes` call for the poisoned type `0xDEADBEEF` returns **`VA_STATUS_SUCCESS` (0)**
   with the value overwritten to **`0x80000000`**. So the status is *not* a validity gate for an
   unknown attribute on this driver, and a probe written to the rule as stated will accept a garbage
   answer. The check that works is the other one: initialise `value` to the poison and treat
   "unchanged" as unknown. *(Correcting a premise of the brief, with the measurement.)*

6. **The header-generation skew may make the diff non-additive.** If step 2's regeneration shows main's
   *decode* assertions have drifted from 1.23 to 1.24, the PR stops being "add encode, change nothing"
   and becomes a second thing. Check in step 2, and if the decode lines move, land them as a separate
   one-line PR first.

7. **`vaSyncBuffer` is deprecated.** Present in libva 2.24 (verified by `nm -D`), absent from a future
   one. An **optional** symbol plus a decline that names the reason keeps that a "VA-API encode is
   unavailable here", not a crash and not a silent `None`.

8. **Scope creep through `VA_RC_CQP` / `VA_RC_VCM`.** Both are measured working, and both need a
   `BitrateMode` main does not have (`Crf` on main means H.264's CRF, not a constant QP). Every hour
   here is an hour in `crates/export` — a crate with a UI, presets and serialised settings — and this
   PR should not touch it. They belong to a second PR, and the branch has already earned them.

9. **`Format::Hevc` becomes available on Linux the moment the probe lands.** That is user-visible in
   `engine/src/export_tools.rs:468` and `project_tools.rs:1283` even when the encoder then declines a
   particular export. It is the intended behaviour and it is what `nvenc/export.rs:281-287` does, but
   the PR should say so in the description so nobody reads it as a regression.

10. **The two parameter sets must state one rate.** The VPS once carried `2 × fps` while the SPS
    stated it plainly; both parsed, and ffmpeg — which reads the VPS — reported double the rate. The
    test that catches it compares **the two sets with each other**, not either against a literal of
    the same value, because the defect was precisely a disagreement between them
    (`hevc_params.rs:1209-1214`). If step 4 keeps only a round-trip parse, this comes back silently.

---

## 7. What is out of scope, and why

| out | why |
|---|---|
| **Any change to `Accel`** | §1. It is the decode front ends' view; `nvenc` has no equivalent trait. |
| **Any change to `crates/export`** | `BitrateMode::Cbr` → `VA_RC_CBR` and `Vbr1Pass` → `VA_RC_VBR` cover the export; everything else is refused by name. |
| **H.264 VA-API encode** | The maintainer asked for it as its own PR. It also needs the H.264 encode records and a separate packed-header story. |
| **HEVC Main 10 / HDR** | Needs 10-bit records, P010 upload, `EncoderFrame::hdr`, and `register_hdr_probe`. Without a probe, `job.rs:197-205` already tone-maps HDR to 8-bit SDR — correct behaviour, free. |
| **`VA_RC_CQP` (constant quality), `VA_RC_VCM`** | Both measured working; both need a `BitrateMode` main does not have. Second PR. |
| **Reference depth > 1, B-frames, reordering** | Measured to silently diverge (risk 3). `media_start` stays `None`. |
| **Several slices per picture, tiles, WPP** | `VAConfigAttribEncSliceStructure` answers `0x0000000c` = `EQUAL_ROWS \| MAX_SLICE_SIZE` (measured today) — one slice per picture is inside it and is what the writer derives. |
| **Interlaced, non-square pixels, two-pass, MXF, 4:2:2/4:4:4** | Refused by the gate, each naming what it cannot express. |
| **`vaQueryBufferTypes`** | Not exported by libva 2.24 (`nm -D`: absent). Buffer types are gated by what actually succeeds. |
| **`VA_SURFACE_ATTRIB_GETTABLE` queries** | Surfaces are created with the plain `vaCreateSurfaces` the decode path already uses, at the RT format the config attribute names. |
| **`vaGetImage` for the coded readback** | `vaMapBuffer` on the coded buffer is what the branch measured; one path, one wait. |
| **FEI, encoder statistics, ROI, parallel rate control** | Unmeasured, unneeded. |
| **Moving `nvenc::hevc::hevc_config` into `filmcraft_hevc`** | §2. A real cleanup, a fourth crate in this diff. Named as a follow-up. |
| **`crates/platform/examples/vahevcprobe.rs`** | The branch's 2.9k-line probe harness. Main's convention for a backend question is a focused `tests/` binary, which steps 6 and 9 already are. |

---

## Appendix — what was verified today, on this machine

Machine: `/dev/dri/renderD128` opened `O_RDWR`; `libva.so.2` → `libva.so.2.2400.0` (libva **2.24**,
VA-API **1.24**); driver string from `/usr/lib/dri/iHD_drv_video.so`: *"Intel iHD driver for Intel(R)
Gen Graphics - 26.3.5 ()"*. **The same driver and libva the branch's numbers were taken on.** Probe:
`dlopen`, never linked; `vaGetConfigAttributes` only; **`vaQueryConfigAttributes` never called**.

```
vaInitialize                                  -> status 0, VA-API 1.24
vaQueryConfigEntrypoints(VAProfileHEVCMain)   -> status 0, 2 entry points: 1 (VLD), 6 (EncSlice)
VAConfigAttribRateControl      (type  5)      -> status 0, value 0x000014de
VAConfigAttribEncPackedHeaders (type 10)      -> status 0, value 0x0000001f
VAConfigAttribEncMaxRefFrames  (type 13)      -> status 0, value 0x00030003   (l0 = 3, l1 = 3)
VAConfigAttribEncSliceStructure(type 15)      -> status 0, value 0x0000000c   (EQUAL_ROWS|MAX_SLICE_SIZE)
poisoned type 0xDEADBEEF                      -> status 0 (!), value 0x80000000 (!)
```

`nm -D --defined-only /usr/lib/libva.so.2`: `vaSyncBuffer` **present**, `vaQuerySurfaceStatus`
present, `vaBufferSetNumElements` present, `vaQueryBufferTypes` **absent**.

C compile against `/usr/include/va/va_enc_hevc.h` (VA-API 1.24.0):

| type | `sizeof` | `alignof` | notable offset |
|---|---|---|---|
| `VAEncSequenceParameterBufferHEVC` | **116** | 4 | `seq_fields` @ 24 |
| `VAEncPictureParameterBufferHEVC` | **576** | 4 | `pic_fields` @ 508 |
| `VAEncSliceParameterBufferHEVC` | **1076** | 4 | `slice_fields` @ 1040 |
| `VAEncPackedHeaderParameterBuffer` | **28** | 4 | — |
| `VAPictureHEVC` | **28** | 4 | — |
| `VAConfigAttrib` | **8** | 4 | — |

The sequence record carries `bits_per_second` directly (`va_enc_hevc.h`), so **no `VAEncRateControl`
record and no misc buffer are needed** — a real reduction against the branch's 2,938-line
`hevc_submit.rs`.

Header constants (file:line in `/usr/include/va/va.h`): `VAEntrypointEncSlice = 6` (:558),
`VAConfigAttribRateControl = 5` (:625), `VAConfigAttribEncPackedHeaders = 10` (:694),
`VAConfigAttribEncMaxRefFrames = 13` (:714), `VAConfigAttribEncSliceStructure = 15` (:738),
`VAEncPictureParameterBufferType = 23` / `VAEncSliceParameterBufferType = 24` (:2063-2064).
`VA_RC_CBR 0x02`, `VA_RC_VBR 0x04`, `VA_RC_VCM 0x08`, `VA_RC_CQP 0x10`, `VA_RC_ICQ 0x40`,
`VA_RC_MB 0x80`, `VA_RC_QVBR 0x400`, `VA_RC_TCBRC 0x1000` (:1110-1164) — the eight modes of `0x14de`.
`VA_ENC_PACKED_HEADER_NONE 0x0`, `SEQUENCE 0x1`, `PICTURE 0x2`, `SLICE 0x4`, `MISC 0x8`,
`RAW_DATA 0x10` (:1199-1231).

**Measurements carried from the branch that are properties of the driver and survive unchanged:**
the driver writes no slice header of its own and refuses to encode without one (`vaMapBuffer` →
`VA_STATUS_ERROR_INTERNAL` while render, end and sync all succeed); `VAConfigAttribEncPackedHeaders`
is advertised but **not enforced** (0x0 / 0x2 / 0x3 / 0x5 all encode); the rate-control sweep with a
passing control — `CBR`, `VBR`, `VCM`, `CQP`, `MB` encode and decode, `ICQ` and `QVBR` are refuted
with and without a record, `TCBRC` fails on the record; the quantisation misc buffer is refused on
its own; HEVC's `slice_type` is I = 2, P = 1, B = 0, measured on a stream this driver encoded, because
no libva header states it; the sequence record carries no crop; reference depth > 1 diverges silently
(9.1, 9.1, 16.8, 26.6, 36.9, 37.1).

**Measurements that must be re-taken by the port:** the 88 record bit positions and five hole widths;
the per-picture byte sizes and deviations; which rate-control modes encode — each was measured against
*our* buffer set and record order.