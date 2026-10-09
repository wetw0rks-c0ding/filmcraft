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

## Tasks

1. **Make `crates/vp9`'s header module reachable**, with the narrowest possible surface — `pub mod` plus
   re-exports of what the backend names, not a blanket `pub use header::*`. A test that the platform
   crate can name `FrameHeader` and `parse_uncompressed`, so the dependency cannot silently regress.
2. **`VADecPictureParameterBufferVP9` from `FrameHeader`**, field for field, with the decline table for
   profiles 1, 3 and 4.
3. **Frame context and reference refresh**, with state pinned by test.
4. **Registration, decline wiring, and the docs** — the honest-limitations text must then say VP9 is
   decoded, and must still say AV1 is not.
5. **Oracle tests** at the standard the other codecs hold.

## Constraints

Unchanged from the decode milestone: safe code outside `vaapi/ffi.rs`; `dlopen` only, never link (the
LGPL ruling rests on it); a value we cannot derive faithfully is an `Err`, never a default; every decline
documented with its reason.

**Do not touch `crates/vp9`'s behaviour.** This milestone makes its parser *reachable*; it must not change
what it computes. VP9 decode is green today and is used by the software path — a behaviour change there
would be a regression discovered by an unrelated test.