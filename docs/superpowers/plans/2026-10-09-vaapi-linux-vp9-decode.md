# VP9 hardware decode on Linux through VA-API

Follow-on to the VA-API **decode** milestone. VP9 is the first codec added to a backend that already
exists, so this is a much smaller piece of work than H.264 or HEVC were — and it is worth saying why,
because the reason is a design property of the codec rather than an accident.

## Why VP9 is the cheap one

- **No slice headers.** VP9's frame header *is* the parameter set. H.264 and HEVC each needed a full
  slice-header parser before a picture could be described to the driver; VP9 needs one header and no
  slice layer.
- **No weighted prediction at all.** VP9 has loop filtering and segmentation, but no weighted
  prediction in either mode. The entire class of bug this milestone lost most of its budget to — the
  H.264 implicit biprediction weights — **cannot occur here**.
- **No reference list modification.** References are a fixed eight-slot array indexed by frame slot,
  not a per-slice list. That removes the class of bug where a list is built once and reused wrongly.

## The enabling change is one line, and that is the whole point

`crates/vp9/src/header.rs` already contains everything the driver needs, and it is already correct —
`crates/vp9` decodes VP9 bit-exact against ffmpeg today in software:

| what VA-API wants | what `crates/vp9` has |
|---|---|
| `VADecPictureParameterBufferVP9` (`/usr/include/va/va_dec_vp9.h:58`) | `FrameHeader` (`header.rs:60`), field-for-field |
| `reference_frames[8]` | `parse_uncompressed(..., refs: &[Option<RefInfo>; 8])` (`header.rs:222`) — **the shape already matches** |
| `ref_frame_idx`, `refresh_frame_flags`, `frame_context_idx`, `color` | all present and `pub` |

Every field of `FrameHeader`, `RefInfo`, `ColorConfig`, `Segmentation` and `LoopFilterParams` is
already `pub`. The only thing standing in the way is the module declaration:

```
crates/vp9/src/lib.rs:27:    mod header;
```

Make it reachable and the backend has its inputs. **This is the enabling change, and it is one line.**

## Scope

- **VP9 4:2:0, 8-bit (profile 0) and 10-bit (profile 2)** — the two profiles this project decodes.
- **Profiles 1 and 3 decline**, with the reason: they are 4:2:2 and 4:4:4, which are out of scope the
  same way they are for H.264 and HEVC. A decline with a stated reason, not a silent wrong picture.
- **VP9 profile 4 is reserved** by the spec and is rejected as unsupported, not parsed.

The driver already advertises `VLD: true` for all four VP9 profiles on this machine. Nothing is waiting
on hardware; this is bounded by our own work.

## The substantive part, and where the bugs will be

Not the field mapping — that is mechanical. The real work is **frame context and reference refresh**:
`frame_context_idx`, `reset_frame_context` and `refresh_frame_flags` are stateful across the stream, and
they decide which of the eight reference slots a picture updates. Getting them wrong produces plausible
pictures from the wrong references, which is exactly the failure mode that cost the H.264 milestone its
`[3, 3]` index-collision bug and its 60-of-72 wrong pictures. This deserves the same treatment: a test
that pins the state, not just pictures that happen to match.

`filmcraft_vp9` already models this (`HeaderState`), so the work is exposing it and driving the record
from it — but it should be treated as the risky part, not the mapping.

## Verification

The strong gate decode already has, and unlike encode it is fully available here: **bit-exact against
ffmpeg, picture for picture.** A `vp9_p0.mp4` fixture is already in the VA-API fixture set.

Plus, as with the other codecs:

- a test that pins the submitted record's bytes, so a `33`-instead-of-`32` class error cannot pass;
- a test proving the decline for profiles 1 and 3 actually declines rather than guessing;
- `cargo xtask ci` before merge — currently red on `crates/gpu` for pre-existing reasons.

## Task 1 — Make `crates/vp9`'s header module reachable

This is the enabling change, and it is deliberately small.

1. In `crates/vp9/src/lib.rs:27`, change `mod header;` so the module is reachable. Use the **narrowest
   surface that works** — `pub mod header;` plus explicit re-exports of the items the backend names
   (`FrameHeader`, `parse_uncompressed`, `RefInfo`, `HeaderState`, `ColorConfig`, `Segmentation`,
   `LoopFilterParams`), **not** a blanket `pub use header::*`. Re-check the field visibility of every type
   you re-export; the plan believes all fields are already `pub`, and that belief came from reading, so
   verify it rather than trusting it.
2. In `crates/platform/Cargo.toml`, add `filmcraft-vp9 = { workspace = true }`. It was removed in `2146b78`
   as unused — only doc comments mentioned it then. **Update the comment above the dependency list**, which
   currently says `filmcraft-vp9` and `filmcraft-av1` are "deliberately absent" because nothing parses those
   headers. That is now half false: VP9 is being parsed. Say why it is present.
3. Add a test, in the platform crate, that **names** `filmcraft_vp9::header::FrameHeader` and
   `filmcraft_vp9::header::parse_uncompressed`. This is the guard: if the dependency or the visibility
   regresses, this fails to compile rather than the failure surfacing later as a confusing error somewhere
   in the picture builder.

Do **not** change what `crates/vp9` computes. This task makes its parser reachable; it must not alter its
behaviour. `crates/vp9` is green today and the software path depends on it.

## Task 2 — `VADecPictureParameterBufferVP9` from `FrameHeader`

Build `VADecPictureParameterBufferVP9` (`/usr/include/va/va_dec_vp9.h:58`) field for field from the parsed
`FrameHeader`, with the decline table.

- **Profiles 0 and 2** (4:2:0, 8- and 10-bit) are decoded. **Profiles 1 and 3** decline with the chroma
  reason (4:2:2 and 4:4:4, out of scope). **Profile 4** is reserved by the spec and is rejected as
  unsupported, not parsed.
- Verify the struct against the header before writing it: `frame_width`, `frame_height`,
  `reference_frames[8]`, then the bitfield union (`subsampling_x`, `subsampling_y`, `frame_type`,
  `show_frame`, `error_resilient_mode`, `intra_only`, `allow_high_precision_mv`, `mcomp_filter_type`,
  `frame_parallel_decoding_mode`, `reset_frame_context`, `refresh_frame_context`, `frame_context_idx`,
  `segmentation_enabled`, `segmentation_temporal_update`, `loop_filter_*`, and the rest). Read the whole
  union out of the header rather than trusting the abbreviated list above.
- Use the header as ABI ground truth. `crates/platform/src/vaapi/ffi.rs` is hand-declared and a **recent
  audit found three structs declared smaller than the header's** — so a size that looks right must be
  measured, not assumed. Note the existing `offset_of!` tests in `ffi.rs` and follow their pattern.
- `reference_frames` is `VASurfaceID[8]` and is **the whole reference set**, not a list — this is the one
  place the shape differs from H.264/HEVC and the easiest thing to get wrong.

## Task 3 — Frame context and reference refresh

The substantive part, and where the bugs will be. Not the mapping — that is mechanical.

`frame_context_idx`, `reset_frame_context` and `refresh_frame_flags` are stateful across the stream and decide
which of the eight reference slots a picture updates. Getting them wrong yields **plausible pictures built
from the wrong references** — the failure mode that produced the H.264 milestone's `[3, 3]` index collision
and its 60-of-72 wrong pictures under a passing test.

`filmcraft_vp9` already models this in `HeaderState`, so the work is exposing it and driving the record from
it. Treat it as the risky part:

- **Pin the state with a test**, not just pictures that happen to match. A test asserting the reference
  mapping for a known sequence of frame headers is the deliverable here; the parity test in Task 5 is not a
  substitute for it.
- VP9's `show_existing_frame` / `frame_to_show_map_idx` path (a frame that re-shows an earlier one) is a real
  case with no H.264/HEVC analogue in this backend. Decide explicitly whether it is decoded or declined, and
  if decoded, test it; do not leave it falling out of the code by accident.
- If any part cannot be derived faithfully from the stream, it is an `Err`, never a default.

## Task 4 — Registration, decline wiring, and the docs

- Register the VP9 decoder factory on the `register()` Linux arm, alongside `vaapi_factory` and in front of
  the software decoders. The Windows backend already handles VP9, so this closes a platform asymmetry rather
  than adding a feature nobody asked for.
- **Split `a_frame_codec_is_declined_before_a_context_exists`** (`crates/platform/tests/vaapi.rs:676-690`).
  It currently asserts VP9 **and** AV1 are declined, and it will fail when VP9 is enabled. The VP9 half
  becomes a decline for *unsupported profiles* (1, 3, 4); AV1 keeps the "no parser reachable" decline.
- **Update the honest-limitations text** so it says VP9 is decoded and **still says AV1 is not**:
  `README.md`, `ROADMAP.md`, `crates/platform/README.md`, `docs/architecture.md`,
  `docs/adr/0001-platform-ffi.md`. `docs/vaapi-progress.md` also needs its VP9 rows corrected.

## Task 5 — Oracle tests

At the standard the other codecs hold: **bit-exact against ffmpeg, picture for picture.**

- Promote `vp9_p0.mp4` into the `FIXTURES` table in `crates/platform/tests/common/mod.rs` with
  `Tier::EveryPicture`. It is currently generated inline by the decline test, not in that table.
- **Add a profile-2 (10-bit) fixture** — `libvpx-vp9 -profile:v 2`. The scope names profile 2 and no fixture
  exists for it, so without this the 10-bit claim is unverified.
- Pin the submitted record's bytes with a test, so a `33`-instead-of-`32` class error cannot pass. Four
  separate times on this branch a wrong constant passed the whole integration suite because only a
  byte-pinning test would have caught it.
- Prove the declines for profiles 1, 3 and 4 actually decline rather than guessing.

## Constraints

Unchanged from the decode milestone: safe code outside `vaapi/ffi.rs`; `dlopen` only, never link (the
LGPL ruling rests on it); a value we cannot derive faithfully is an `Err`, never a default; every decline
documented with its reason.

**Do not touch `crates/vp9`'s behaviour.** This milestone makes its parser *reachable*; it must not change
what it computes. VP9 decode is green today and is used by the software path — a behaviour change there
would be a regression discovered by an unrelated test.