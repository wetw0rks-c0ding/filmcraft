# AV1 hardware decode on Linux through VA-API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive AV1 Main (4:2:0, 8- and 10-bit) through the existing VA-API decode path by building
`VADecPictureParameterBufferAV1` and `VASliceParameterBufferAV1` from `filmcraft-av1`'s own parsed
frame header, so a VA-API AV1 stream is bit-exact against ffmpeg picture for picture, and everything
outside that scope declines with a stated reason.

**Architecture:** Mirrors the VP9 milestone exactly: a per-decode `Av1State` threads one
`filmcraft_av1::header::HeaderState` across the stream and owns the eight reference slots; one
picture record and one slice record are built together per frame and submitted in one `vaRenderPicture`
beside the frame's bytes. The enabler is a **narrow public surface in `crates/av1`** — a `pub`
`BitReader`, a widened `FrameHeader::parse`, a `HeaderState` constructor and the handful of
spec constants the record names — none of which changes what the software decoder computes.

**Tech Stack:** Rust; libva 2.x resolved by `dlopen` as already set up in `crates/platform/src/vaapi/ffi.rs`;
the clean-room `filmcraft-av1` frame-header parser; ffmpeg (`libaom-av1`) as fixture generator and
decode oracle only.

**Spec:** This plan is self-contained. Its normative sources are the installed
`/usr/include/va/va_dec_av1.h`, the installed `/usr/include/va/va.h`, and the **AV1 Bitstream &
Decoding Process Specification v1.0.0 with Errata 1** (the edition `crates/av1/README.md` names).

## Global Constraints

- **`crates/av1` is bit-exact with libdav1d today and the software path depends on it. This plan
  must not put that at risk.** The enabler changes **visibility and additive surface only** — no
  existing function's arithmetic, control flow or output changes. Every enabler task ends with
  `cargo test -p filmcraft-av1` still green and still bit-exact (see Task 1 Step 5). Do **not**
  reimplement `FrameHeader::parse`'s reads or "clean up" its internals; widening `pub(crate)` to
  `pub` is the whole change. (AGENTS.md §2 clean-room: the AV1 parser is implemented from the public
  spec and must stay so; this plan adds no third-party decoder code.)
- **`unsafe` lives only in `crates/platform/src/vaapi/ffi.rs`**, which carries `#[allow(unsafe_code)]`.
  Every new `unsafe` block needs a `// SAFETY:` comment. No `unsafe` in `vaapi/picture.rs`,
  `vaapi/stream.rs` or `vaapi/decoder.rs`. (`clippy::undocumented_unsafe_blocks = "deny"`.)
- **Never-crash lints stay on** (`AGENTS.md` §0): no `unwrap`/`expect`/`panic!`/`unreachable!`/
  `todo!` outside `#[cfg(test)]`. Every public item returns `Result<T, String>`. Every number that
  comes from the bitstream is bounded with `try_from`/`checked_*`/`get()` before it reaches the
  record — a value we cannot derive faithfully is an `Err`, **never a guessed default**, because a
  guessed record is a valid-looking record the driver reads as a real picture.
- **Every absence claim in this plan is backed by the grep or the header line shown.** Three defects
  on this branch were *confident claims of absence* (a CamelCase grep miss; a fabricated spec clause
  "8.2.3.2.1"; a record "not declared" that the header declares). **Specificity is not evidence.**
  Before writing a step that says a field/record does not exist, re-run the grep or read the header
  line; if it exists, use it.
- **Fixtures live in gitignored `target/`.** A fixture named by **no committed `FIXTURES` table
  entry** in `crates/platform/tests/common/mod.rs` is unreproducible and proves nothing (this
  milestone's sibling failed a whole-branch review for exactly this: a 24/24 result that existed
  only in a gitignored report). Every fixture the plan relies on is added to that table, and every
  one intended to decode is wired to `Tier::EveryPicture` (the `Tier` enum in
  `crates/platform/tests/vaapi.rs`).
- **The record is measured against a C compile, not trusted.** Every AV1 record is pinned three ways
  in `abi_guard.rs` — `sizeof`, every `offsetof`, and every compiler **hole** named as a
  `padding_*` field (a hole changes no size, so `sizeof` alone is blind to it). This is the branch's
  standing doctrine after the `VASegmentParameterVP9` 2-byte hole decoded bit-exact while handing the
  driver two uninitialised bytes per record.
- **Gate before every commit:** `cargo xtask ci`. It is red on `crates/gpu` for pre-existing reasons
  unrelated to this branch (`docs/vaapi-progress.md` §7); every other gate must be green.

## Review Focus

The five input classes or failure modes a user of this software would hit first that no task's tests
otherwise exercise. Each is pinned by a test named in the owning task.

1. **A frame whose `frame_size_override_flag` (or the AV1 equivalent) changes the coded size mid-run.**
   The driver renders into a surface allocated at the stream's declared size; a record carrying a
   larger `frame_width` than the surface is a driver-side overflow or a silently cropped readback.
   Owned by Task 5 (`check_av1_shape`), tested there.
2. **A reference slot that is empty but submitted as a non-`VA_INVALID_SURFACE` value.** Surface ids
   come from `vaCreateSurfaces` and are not guaranteed to avoid any value, so a zero or a stale id
   names *some* surface and the driver reads it as a reference — a plausible picture from the wrong
   place. Owned by Task 4 (record), tested by `ref_frame_map` empty-slot assertions.
3. **A `show_existing_frame` temporal unit.** It carries no frame of its own; a submission built for
   it (or a skip that produces no picture where the stream requires one) yields a blank timeline.
   Owned by Task 5 (decode loop) + Task 4 (record refusal), both tested.
4. **A tile-group OBU split across more than one OBU** (a frame whose tiles do not all arrive in one
   `tile_group_obu`). The slice record is per tile, so a tile split from its OBU would be submitted
   with the wrong `slice_data_size`/`slice_data_offset`. Owned by Task 5, tested with a multi-tile,
   multi-group fixture.
5. **A driver that returns zero pictures after `vaEndPicture` reports success.** It looks like a
   blank timeline rather than an error, and is invisible to a byte-pinning test. Owned by the
   Task 6 oracle (`Tier::EveryPicture` + count floor), which fails on an empty decode.

---

## What the driver supports (measured on the dev machine)

Intel Arc B580, `/dev/dri/renderD128`, libva 1.24 (libva 2.x ABI), iHD 26.3.5. The driver advertises
profile `VAProfileAV1Profile0` (`= 32`, `va.h:537`, pinned at `ffi.rs:437`) with `VLD: true`. **This is
not a hardware limit** — the gap is ours. Nothing here waits on hardware.

`va_dec_av1.h:44` states the constraint plainly: *"This AV1 decoding API supports 8-bit/10bit 420
format only."* That matches this backend's scope (4:2:0 at 8/10) and is the basis for the chroma and
bit-depth declines below.

---

## The enabler — what it really is (and what `docs/vaapi-progress.md` §5 gets wrong)

**`docs/vaapi-progress.md` §5 says the blocker is that `crates/av1/src/header.rs:478` declares
`pub(crate) fn parse` over private types, and frames it as a module-visibility wall like VP9's
`mod header;`. That framing is stale, and this plan corrects it** (a sibling agent owns that file;
this is recorded here and Task 7 does not edit it):

- `crates/av1/src/lib.rs:16` is **already `pub mod header;`** (since `1813bba`), and `:32` already
  re-exports `pub use header::{ColorConfig, FrameHeader, SequenceHeader};`.
- `HeaderState` is **already `pub struct`** at `crates/av1/src/header.rs:458`.
- `header.rs` has **224 `pub` items and only 3 `pub(crate)`** (`relative_dist` :448, `parse` :478,
  `default_gm_params` :1316). The module is already almost fully public.

So it is **not** a module wall. Verified state of the actual gap:

| what a record builder needs | current state | what must change |
|---|---|---|
| `BitReader` (to parse a frame header) | `pub(crate) struct BitReader<'a>` at `crates/av1/src/bits.rs:6` | make `pub` (and re-export) |
| `FrameHeader::parse(r: &mut BitReader, …)` | `pub(crate) fn` at `header.rs:478` — `pub(crate)` **only because** its signature names the `pub(crate)` `BitReader` | widen to `pub` once `BitReader` is `pub` |
| a way to build a `HeaderState` | all fields `pub`, but **no public constructor** and `default_gm_params()` is `pub(crate)` | add `HeaderState::new` / make `default_gm_params` reachable |
| `REFS_PER_FRAME` / `NUM_REF_FRAMES` / `PRIMARY_REF_NONE` / `SEG_LVL_*` / frame-type consts | all `pub const` in the **private** `mod spec_tables` | re-export the ones the record names |

**So the enabler is a narrow public surface: a `pub` `BitReader` plus a widened `parse` plus a small
re-export block.** It is *not* a module-visibility wall, and the VP9 precedent (`mod header;` →
`pub mod header;`) does not apply — `crates/av1` did not need that change and should not get one.

**Keeping `crates/av1`'s existing API stable for its software consumers:** the change is strictly
additive — `pub(crate) → pub` on four items and a `pub use` block. No existing `pub` item changes
signature, visibility or value. The software `Decoder` keeps compiling against the same internals. The
regression gate is `cargo test -p filmcraft-av1` staying **bit-exact against libdav1d** (its
`tests/oracle_intra.rs` and `tests/conformance.rs`); see Task 1 Step 5. Because the plan only widens
what is already `pub(crate)`, the blast radius of any mistake is "the plan's new code does not
compile", never "the software decoder decodes differently".

---

## Scope

**In scope, with reasons:**

- **AV1 `seq_profile == 0` (Main) at 4:2:0, 8-bit and 10-bit** — the two depths `va_dec_av1.h:44`
  names and this backend already has surfaces for (`RtFormat::Nv12` / `RtFormat::Yuv420_10`).
  `plan_frame` already plans exactly these two (`stream.rs:176-177`); the milestone makes the plan
  true.
- **`refresh_frame_flags`, the primary reference bit and custom-order reference ops** — the stateful
  core (Task 5). These decide which of the eight slots a picture updates and are where a wrong answer
  yields *plausible pictures built from the wrong references*, the exact failure that cost H.264 its
  `[3,3]` index collision and its 60-of-72 wrong pictures.

**Declined, each with its reason** (a decline is always a clean fall back to the software decoder,
never a guessed record):

| declined | why |
|---|---|
| `seq_profile` 1 and 2 | 4:4:4 / 4:2:2 and 12-bit — `va_dec_av1.h:44` says the API is 8/10-bit 420 only, and every surface this backend creates is 4:2:0 at 8 or 10 bits. Same reasoning as H.264/HEVC/VP9 chroma declines. |
| 12-bit (`BitDepthIdx == 2`) | no 10-bit-or-less 12 surface; out of scope by decision, same as the profile decline. |
| monochrome | `plan_frame` already declines `info.mono` for every frame codec (`stream.rs:153`); an AV1 picture record has no way to put a luma-only picture on a 4:2:0 surface. |
| `film_grain_params_present == 1` with `apply_grain == 1` | the record has a full `VAFilmGrainStructAV1` (`va_dec_av1.h:153-201`) but this backend reads the decoded picture back via `vaDeriveImage`+`vaMapBuffer` and does **not** apply film grain. Submitting the grain parameters would have the driver render grain into `current_display_picture` that nothing reads back — a second surface, a second copy, and a picture that disagrees with the software path on grains. The driver capability probe should still *report* whether the feature is present, so `capabilities()` does not lie; the decode path declines the *stream*. |
| `large_scale_tile == 1` | `va_dec_av1.h` gates `tile_cols/tile_rows`, `width_in_sbs_minus_1`, `output_frame_*_in_tiles_minus_1`, `tile_count_minus_1`, `anchor_frames_*` and `tile_idx_in_tile_list` on large-scale tile ("Valid only when large_scale_tile == 1"). `filmcraft-av1` does not implement LST ("not planned", `crates/av1/README.md`) and this backend has no anchor-frame machinery. Declined as unsupported, not parsed. |
| a mid-run resolution change (`frame_size_override_flag` that contradicts the session surface) | refused in `check_av1_shape` (Task 5) rather than submitted: the surface is fixed for the run. |

**Stated open question (not a confident guess):** whether `filmcraft-av1`'s `TileInfo` alone can
produce `width_in_sbs_minus_1[63]`/`height_in_sbs_minus_1[63]` exactly, or whether the builder needs
the superblock grid. See `## Open questions`.

---

## The record, field by field

`VADecPictureParameterBufferAV1` is declared at `va_dec_av1.h:243-628`. Every offset and size below was
**measured by a C compile on this machine** (`cc -I /usr/include/va`), not read off the header text —
this is the discipline the branch learned the hard way (`struct drm_prime_handle` was "confirmed"
8 bytes from five lines; line 932 held a third field). The values in this section are what
`abi_guard.rs` will assert, so an implementer must reproduce them, not re-derive them.

**Record size and the six holes.** `sizeof(VADecPictureParameterBufferAV1) == 1160`, alignment 8.
It has **six compiler holes** and a trailing pad — none visible to `sizeof` alone. This is the
`VASegmentParameterVP9` shape at ten times the size, so the `padding_*` fields are mandatory:

| bytes | between | why |
|---|---|---|
| 17..24 | `anchor_frames_num` (16) and `anchor_frames_list` (24) | the pointer is 8-aligned |
| 81..84 | `order_hint` (80) and `seg_info` (84) | `VASegmentationStructAV1` is 4-aligned |
| 674..676 | `context_update_tile_id` (672) and `pic_info_fields` (676) | `pic_info_fields` is 4-aligned |
| 703..704 | `v_ac_delta_q` (702) and `qmatrix_fields` (704) | `qmatrix_fields` is 2-aligned |
| 706..708 | `qmatrix_fields` (704) and `mode_control_fields` (708) | `mode_control_fields` is 4-aligned |
| 1156..1160 | after `va_reserved` (1124, 32 bytes) to the 8-byte alignment | struct alignment |

Sub-record sizes (all measured): `VASegmentationStructAV1` = 156 (one hole, 148..152, before
`va_reserved`); `VAFilmGrainStructAV1` = 176 (one hole, 153..154, before `cb_offset`); and
`VAWarpedMotionParamsAV1` = 56 (one hole, 37..40, before `va_reserved`).

`VASliceParameterBufferAV1` (`va_dec_av1.h:647-686`) is **40 bytes** with one hole (21..22, before
`tile_idx_in_tile_list`). Its `tg_start`/`tg_end` are `va_deprecated`; they still occupy bytes and are
part of the size.

**The six bitfield unions.** Their bit positions were measured by setting one member and reading the
union's `value` (a C compile on this machine), not inferred from declaration order — a C bitfield's
layout is target-dependent and the VP9 milestone's own positions table was verified the same way.
Recorded so the implementer does not re-guess:

- `pic_info_fields` (32 bits at offset 676): `frame_type` 0:2, `show_frame` 2, `showable_frame` 3,
  `error_resilient_mode` 4, `disable_cdf_update` 5, `allow_screen_content_tools` 6, `force_integer_mv` 7,
  `allow_intrabc` 8, `use_superres` 9, `allow_high_precision_mv` 10, `is_motion_mode_switchable` 11,
  `use_ref_frame_mvs` 12, `disable_frame_end_update_cdf` 13, `uniform_tile_spacing_flag` 14,
  `allow_warped_motion` 15, `large_scale_tile` 16, `reserved` 17:32.
- `seq_info_fields` (32 bits at offset 4): `still_picture` 0 … `film_grain_params_present` 15,
  `chroma_sample_position` 14 is `va_deprecated` (zero it), `reserved` 16:32.
- `seg_info.segment_info_fields` (32 bits at offset 84): `enabled` 0, `update_map` 1,
  `temporal_update` 2, `update_data` 3, `reserved` 4:32.
- `mode_control_fields` (32 bits at offset 708): `delta_q_present_flag` 0, `log2_delta_q_res` 1:3,
  `delta_lf_present_flag` 3, `log2_delta_lf_res` 4:6, `delta_lf_multi` 6, `tx_mode` 7:9,
  `reference_select` 9, `reduced_tx_set_used` 10, `skip_mode_present` 11, `reserved` 12:32.
- `qmatrix_fields` (16 bits at offset 704): `using_qmatrix` 0, `qm_y` 1:5, `qm_u` 5:9, `qm_v` 9:13,
  `reserved` 13:16. `loop_restoration_fields` (16 bits at offset 730): `yframe_restoration_type` 0:2,
  `cbframe_restoration_type` 2:4, `crframe_restoration_type` 4:6, `lr_unit_shift` 6:8,
  `lr_uv_shift` 8, `reserved` 9:16. `loop_filter_info_fields` (8 bits at offset 686):
  `sharpness_level` 0:3, `mode_ref_delta_enabled` 3, `mode_ref_delta_update` 4, `reserved` 5:8.

**Field mapping from `FrameHeader` / `SequenceHeader`.** The builder reads a `FrameHeader` (already
parsed by `filmcraft-av1`, the same parse the software decoder runs). Where a bitstream value maps
1:1, this table says so; where the record wants a *derived* value the note says so.

- `profile` ← `seq.profile`; `order_hint_bits_minus_1` ← `seq.order_hint_bits - 1` (`u8::try_from`);
  `bit_depth_idx` ← derived from `seq.color.bit_depth`: 8→0, 10→1, 12→2 (the header's own table,
  `va_dec_av1.h:255-261`); `matrix_coefficients` ← `seq.color.matrix_coefficients`.
- `seq_info_fields` ← the corresponding `SequenceHeader` booleans plus `color_range`,
  `subsampling_x`, `subsampling_y`, `film_grain_params_present`. `chroma_sample_position` is
  `va_deprecated`: zero it.
- `current_frame` ← the surface chosen for this picture; `current_display_picture` ← the same
  (valid only when `apply_grain`, which we decline, so it is unused — set it equal to `current_frame`
  rather than leaving a stale value).
- `frame_width_minus1` / `frame_height_minus1` ← `FrameHeader::upscaled_width - 1` /
  `upscaled_height - 1`. **Upscaled, not coded**: the header says "Picture original resolution. If
  SuperRes is enabled, this is the upscaled resolution" (`va_dec_av1.h:332-336`). A superres stream is
  therefore *not* declined for size alone; only a `frame_size_override_flag` that contradicts the
  session surface is (Task 5).
- `ref_frame_map[8]` ← the whole decoded picture buffer **by slot**, `VA_INVALID_SURFACE` for an
  empty slot; `ref_frame_idx[7]` ← `h.ref_frame_idx[0..7]` (already in the record's LAST..ALTREF
  order, 0-based); `primary_ref_frame` ← `h.primary_ref_frame` (`PRIMARY_REF_NONE` = 7 for none);
  `order_hint` ← `h.order_hint`.
- `seg_info` ← `h.seg`: the four flags into `segment_info_fields`; `feature_data[8][8]` and
  `feature_mask[8]` from `h.seg.features` (`enabled[seg][feature]`, `data[seg][feature]`), each
  `data` narrowed to `i16` with `i16::try_from`.
- `tile_cols` / `tile_rows` ← `h.tile_info.cols` / `.rows` (a **count**, not a log2 — the record's
  comment says the driver regenerates `width_in_sbs_minus_1` from these when
  `uniform_tile_spacing_flag == 1`); `context_update_tile_id` ← `h.tile_info.context_update_tile_id`;
  `width_in_sbs_minus_1[63]` / `height_in_sbs_minus_1[63]` ← derived from `mi_col_starts`/`mi_row_starts`
  and the superblock size (open question below).
- `pic_info_fields` ← `h.frame_type`, `h.show_frame`, `h.showable_frame`, `h.error_resilient_mode`,
  `h.disable_cdf_update`, `h.allow_screen_content_tools`, `h.force_integer_mv`, `h.allow_intrabc`,
  `h.use_superres`, `h.allow_high_precision_mv`, `h.is_motion_mode_switchable`, `h.use_ref_frame_mvs`,
  `h.disable_frame_end_update_cdf`, `h.tile_info.cols_log2 > 0 || rows_log2 > 0` →
  `uniform_tile_spacing_flag`, `h.allow_warped_motion`. `large_scale_tile` = 0 (we decline LST).
- `superres_scale_denominator` ← `h.superres_denom` (8 when `use_superres == 0`; `u8::try_from`);
  `interp_filter` ← `h.interpolation_filter`. **Note:** `interpolation_filter` is read from the bitstream
  only for inter frames (`header.rs:639`); an intra frame's `FrameHeader` leaves it at its `Default`
  (0 = EIGHTTAP). That is the spec's own state (5.9.14 is not read intra) and matches VP9's
  `mcomp_filter_type` default, so submit 0 for intra and say so — see the constants discussion below.
- `filter_level[2]` ← `h.lf.level[0..2]`; `filter_level_u`/`_v` ← `h.lf.level[2]`/`[3]`;
  `loop_filter_info_fields` ← `h.lf.sharpness`, `h.lf.delta_enabled`, `h.lf.delta_update`;
  `ref_deltas[8]` / `mode_deltas[2]` ← `h.lf.deltas.ref_deltas` / `.mode_deltas` (`i8::try_from`).
- `base_qindex` ← `h.quant.base_q_idx`; `y_dc_delta_q` … `v_ac_delta_q` ←
  `h.quant.delta_q_y_dc` … `delta_q_v_ac` (`i8::try_from`); `qmatrix_fields` ← `h.quant.using_qmatrix`,
  `h.quant.qm_y`, `qm_u`, `qm_v`.
- `mode_control_fields` ← `h.delta_q_present`, `h.delta_q_res`, `h.delta_lf_present`, `h.delta_lf_res`,
  `h.delta_lf_multi`, `h.tx_mode`, `h.reference_select`, `h.reduced_tx_set`, `h.skip_mode_present`.
- `cdef_damping_minus_3` ← `h.cdef.damping - 3` (`u8`, the parser already stores `damping = r.f(2)+3`,
  `header.rs:1063`); `cdef_bits` ← `h.cdef.bits`; `cdef_y_strengths[8]` /
  `cdef_uv_strengths[8]` ← the header's packing `(pri << 2) | (sec & 0x03)` from `h.cdef.y_pri`,
  `y_sec`, `uv_pri`, `uv_sec` (`va_dec_av1.h:591-598`).
- `loop_restoration_fields` ← `h.lr.frame_restoration_type[0..3]` (each 2 bits, remapped to
  `NONE/SWITCHABLE/WIENER/SGRPROJ` — the parser stores the spec's own enum, and the record wants 2-bit
  restoration types; **verify the mapping against the header's comment and the spec**, they are the
  same enum), and `lr_unit_shift`/`lr_uv_shift` ← derived from `h.lr.loop_restoration_size` /
  `uses_lr`.
- `wm[7]` ← the seven `VAWarpedMotionParamsAV1`: `wmtype` ← `h.gm_type[1..8]` mapped to
  `VAAV1TransformationType` (IDENTITY/TRANSLATION/ROTZOOM/AFFINE), `wmmat[8]` ← `h.gm_params[rf][0..6]`
  with the last two entries zero, `invalid` = 0. The header says "`wmmat[]` corresponds to `gm_params[][]`
  in spec" (`va_dec_av1.h:222`).
- `va_reserved[8]` ← all zero.

### The slice record and the tile problem (the AV1 case VP9 did not have)

`VASliceParameterBufferAV1` is sent **once per tile** (`va_dec_av1.h:635`), with
`slice_data_size`/`slice_data_offset`/`tile_row`/`tile_column` per tile, and `slice_data_flag =
VA_SLICE_DATA_FLAG_ALL`. **This is the substantive difference from VP9, which has one slice record per
frame** (VP9's record is a "per-frame" buffer, `va_dec_vp9.h:265-268`).

To build it, the backend needs **each tile's byte extent within the frame's tile data** — which
`filmcraft-av1` computes in its private `Decoder::tile_group` (`crates/av1/src/decoder.rs:555-579`,
using `h.tile_info.tile_size_bytes` and the `tile_size_bytes`-wide length prefixes) but does **not
expose**. `FrameHeader::tile_info` exposes `cols`, `rows`, `mi_col_starts`, `mi_row_starts` and
`tile_size_bytes` (`header.rs:224-233`) — enough to compute the **grid**, but the per-tile *byte
offsets* come from walking the tile-group OBU, which is the decoder's private job.

**Decision (Task 4):** `FrameHeader::parse` already leaves `tile_info.tile_size_bytes` set. The record
builder is given the frame's own bytes and walks the tile group itself: for `num_tiles = cols*rows`,
each tile but the last is preceded by a `tile_size_bytes`-byte little-endian length prefix
(`+1` for the last), exactly as `Decoder::tile_group` does (`decoder.rs:564-576`). This mirrors what
VP9's builder does for `first_partition_size` — derive the length from the frame's own bytes rather
than ask the driver to guess. **This walk is the riskiest derived value in the milestone** and gets
its own test (Task 4, Step 4) that cross-checks the builder's tile extents against
`filmcraft_av1::Decoder`'s, on a real multi-tile fixture.

`tg_start`/`tg_end` in the slice record are `va_deprecated`; the header's comment and the spec's tile
group make one slice per tile, so they are left zero and the per-tile `tile_row`/`tile_column` carry
the position.

---

## The reference / state model — the risky part, decided explicitly

VP9's lesson: `reference_frames[8]` is **the whole decoded picture buffer addressed by slot**, not a
per-slice list. AV1's model differs again, and the differences are named here rather than left to
fall out of the code:

- **AV1 has `ref_frame_map[8]` (the buffer, by slot) AND `ref_frame_idx[7]` (per-frame indirection
  into it) AND `primary_ref_frame` (which entry of `ref_frame_idx` to load propagated info from).**
  Three levels where VP9 had one. The slot map is built by `Av1State` exactly as `Vp9State` builds
  its eight; `ref_frame_idx` and `primary_ref_frame` come straight from the parsed header.
- **`refresh_frame_flags` decides which slots the *rendered* picture becomes**, and is applied **only
  after the driver has rendered**, so a record describes the buffer *before* its own frame. This is
  `Av1State::commit`, mirroring `Vp9State::commit` (Task 5). A key frame sets `refresh_frame_flags`
  to all-ones (`header.rs:574`).
- **The `primary_ref_frame` bit and the propagated info it gates.** AV1 7.20 `load_previous` reads the
  loop filter deltas, segmentation features and global motion from the primary reference when
  `primary_ref_frame != PRIMARY_REF_NONE` (the parser already does this into `FrameHeader`: `h.lf.deltas`,
  `h.seg.features`, `h.prev_gm_params`, `header.rs:650-663`). The record's `ref_deltas`/`mode_deltas`
  and `wm[]` must therefore be built from the **primary reference's** inherited values for the fields
  the current header does not override — which the parsed `FrameHeader` already carries. **This is a
  case with no clean H.264/HEVC/VP9 analogue in this backend** (they have no
  `primary_ref_frame`/`setup_past_independence` split) and is decided explicitly: the record is built
  from the *resolved* `FrameHeader`, and the test pins that an inter frame with
  `primary_ref_frame != NONE` submits the primary reference's deltas (Task 4 Step 5).
- **`show_existing_frame`.** As with VP9 it re-shows an earlier picture and codes none: the parser
  returns early at `header.rs:489-509` (`if h.show_existing_frame { … return Ok(h) }`) with almost
  nothing set. **Decision: decoded** — the surface the named
  slot holds is output with no decode (AV1 7.21), which is what the software decoder does
  (`decoder.rs`). No picture record is built for it (Task 4 refuses it with a precise message); the
  decode loop re-shows the slot (Task 5). Declining it would put a hole where the stream requires a
  picture.
- **`show_frame == false`** — rendered, may still refresh references, produces no picture. Carried
  through, same as VP9 (`Vp9Frame::show`).
- **Every value the record cannot derive faithfully is an `Err`, never a default** — e.g. a
  `ref_frame_idx` naming an empty slot, a `film_grain apply_grain` we decline, an LST frame.

---

## Fixtures and the oracle

`crates/platform/tests/common/mod.rs` `FIXTURES` holds the ffmpeg argument lists and is the
committed, reproducible source of every fixture; `target/` is gitignored, so anything not in that table
cannot be regenerated by a reviewer and must not be relied on. **Add the AV1 entries to `FIXTURES`**
and wire the decodable ones to `Tier::EveryPicture`.

> **Line numbers below are as of this plan's date and will drift** — a concurrent agent added three
> VP9 fixtures to `FIXTURES` while this plan was written. **Search for the symbol**, not the line:
> `FIXTURES`, `Tier::EveryPicture`, `fn streams`, `bit_exact_with_the_software_decoder`,
> `an_av1_stream_is_declined_before_a_context_exists`.

`streams()` assigns `Tier::EveryPicture` to every `FIXTURES` entry that is not `h264_high.mp4`, so a
decodable AV1 entry is wired the moment it is in the table. Fixture plan (names are proposals; the
arguments are what matters):

1. **`av1_p0_8bit.mp4`** — `libaom-av1 -pix_fmt yuv420p`, one GOP per ~24 frames, B-frames/`lag-in-frames`
   to exercise reordering, ≥2 GOPs for the seek leg. `Tier::EveryPicture`.
2. **`av1_p0_10bit.mp4`** — same, `-pix_fmt yuv420p10le`. `Tier::EveryPicture`. **Scope names 10-bit,
   so a 10-bit fixture must exist** or the claim is unverified (this is VP9 Ruling 2 repeating).
3. **`av1_tiled.mp4`** — libaom tile options (`-tile-columns 1 -tile-rows 2 -tile-row-mt 0`) so more
   than one tile and one tile group exercise the per-tile slice record. `Tier::EveryPicture`.
4. **`av1_intra_only.mp4`** — all-intra (`-usage intra`), which drives `primary_ref_frame` to
   `PRIMARY_REF_NONE` and `setup_past_independence` — the inherited-deltas path. `Tier::EveryPicture`.

`libaom-av1`, `libsvtav1` and `librav1e` are all present in ffmpeg here (verified). The existing
decline test already generates `av1_p0.mp4` inline (`tests/vaapi.rs`, fn
`an_av1_stream_is_declined_before_a_context_exists`); that test is **split** in Task 6 (its AV1 half
currently asserts the decline and must become a *profile/feature* decline, per the VP9 Ruling 1
precedent), and the decodable fixtures move into `FIXTURES`.

**The oracle.** The gate is `Tier::EveryPicture`: every picture the hardware decoder returns is
bit-for-bit what the **software decoder** produces, over the whole stream, after `reset`+reseek to each
sync sample, and again after all the resets — plus the count floor in `bit_exact_with_the_software_decoder`,
which the AV1 entries raise and which must **name the AV1 fixtures explicitly** rather than only
raising a number. The VP9 milestone just set that precedent in the same test (it names its three
fixtures by name because a bare count would go stale the moment a fixture is dropped). A run that
silently dropped AV1 must be loud.

**Decode is comparable bit-for-bit; encode is not** — a hardware encoder and a software encoder are
different algorithms — so nothing here may be phrased as a hardware-vs-software *encode* comparison. A
VA-API decoder returning zero pictures after `vaEndPicture` succeeds (Review Focus 5) is caught by the
`a.len() >= 40` floor in `parity()`.

---

### Inert constants: decided from the spec, and how each is verified

VP9 shipped two values the record submits that are *not* read from the bitstream for some frames and
were assumed inert: `segment_pred_probs` inherits 0 where VP9 6.2.11 initialises 255, and
`mcomp_filter_type` is 0 (EIGHTTAP) for intra where 6.2.7 says `NORMAL`. AV1 has two structural
analogues, and this plan **decides each from the specification and pins it with a test**, rather than
assuming inertness:

- **`interp_filter` for an intra frame.** `FrameHeader::interpolation_filter` is read only inside the
  non-intra branch (`crates/av1/src/header.rs:636-648`); an intra frame leaves it at `Default` = 0
  (EIGHTTAP). The record field is `interp_filter` (`va_dec_av1.h:458`). **AV1 spec 5.9.5/5.9.14 is
  explicit that `interpolation_filter` is only present for inter frames** (and global motion is
  likewise identity for intra, `header.rs:1168-1170`, which is what the `wm[]` array needs). So 0 is
  the **spec's own state**, not a guess, and is a faithful value rather than an invented one.
  **Verification:** the record-bytes test (Task 4 Step 1) asserts an intra fixture's `interp_filter`
  is the EIGHTTAP the spec defines, and the oracle must stay bit-exact on the all-intra fixture —
  an intra frame that decoded wrong would show there. This is strictly better grounded than VP9's
  `mcomp_filter_type`, where the deviation was recorded rather than resolved.
- **Loop filter / CDEF defaults when a feature is absent.** `crates/av1` sets `cdef.damping = 3` when
  CDEF is off (`header.rs:1060`, matching the record's `cdef_damping_minus_3 = damping - 3 = 0`) and
  `lf.level = [0;4]` for a lossless/intrabc frame (`header.rs:1019-1023`). These are the parser's
  spec-derived values, not our defaults, and the record submits them unchanged. **Verification:** the
  record-bytes test asserts `cdef_damping_minus_3 == 0` on a CDEF-off fixture and the four
  `filter_level` entries on a lossless fixture, so the "inert" claim is a measurement.

Neither is *assumed* inert: both are the specification's own value for the case, and both are pinned
by assertions on real fixtures plus the bit-exact oracle. If a future AV1 fixture disagreed, the
oracle fails and the constant is revisited from the spec then — not carried on trust.

---

## File structure

**Created:**
- `crates/platform/src/vaapi/av1.rs` — `Av1State`, `Av1Frame`, the decline table and the tile walk.
  Kept separate from the 7000-line `picture.rs` so the AV1 state is one readable file.

**Modified:**
- `crates/av1/src/bits.rs` — `BitReader` `pub(crate)` → `pub`.
- `crates/av1/src/header.rs` — `FrameHeader::parse` and `default_gm_params` `pub(crate)` → `pub`;
  add `HeaderState::new`.
- `crates/av1/src/lib.rs` — `pub mod bits;` (or `pub use bits::BitReader;`) and a `pub use
  spec_tables::{…}` block for the constants the record names.
- `crates/platform/Cargo.toml` — add `filmcraft-av1 = { workspace = true }`, update the comment above
  the dependency list that currently says AV1 is "deliberately still absent".
- `crates/platform/src/vaapi/ffi.rs` — the five AV1 record structs + safe readers + layout tests.
- `crates/platform/src/vaapi/abi_guard.rs` — the AV1 records in `SUBMITTED_RECORDS` and the `sizeof`
  list.
- `crates/platform/src/vaapi/picture.rs` — `build_av1_frame` lives here (it is a picture-parameter
  builder) and calls into `av1.rs`; retire the `FRAME_HEADER_MISSING` AV1 arms.
- `crates/platform/src/vaapi/stream.rs` — `plan_frame` AV1 arm becomes a real plan; retire the AV1
  `FRAME_HEADER_MISSING` decline.
- `crates/platform/src/vaapi/decoder.rs` — `picture_frame` AV1 dispatch; `surface_pool` AV1 size; the
  surface-pool enumeration test.
- `crates/platform/tests/common/mod.rs` — the four `FIXTURES` entries.
- `crates/platform/tests/vaapi.rs` — split the AV1 decline test; wire `Tier::EveryPicture`.

**Not modified:** `docs/vaapi-progress.md` (owned by another agent this session — the §5 correction is
recorded above and left for that owner).

---

## Task breakdown

### Task 1 — Widen `crates/av1`'s surface (the enabler)

**Files:** `crates/av1/src/{bits,header,lib}.rs`

**Interfaces:**
- Consumes: nothing.
- Produces, reachable from outside `filmcraft_av1` afterwards: `filmcraft_av1::BitReader`
  (`pub fn new`, `bit`, `f`, `flag`, `uvlc`, `le`, `leb128`, `su`, `ns`, `byte_align`),
  `filmcraft_av1::header::{FrameHeader::parse, HeaderState, RefInfo, SequenceHeader, ColorConfig, …}`
  with `FrameHeader::parse(r: &mut BitReader, st: &mut HeaderState, temporal_id: u32, spatial_id: u32)
  -> Result<FrameHeader>` now `pub`; a `HeaderState::new(seq: &SequenceHeader, refs: &mut [RefInfo; 8])
  -> HeaderState` constructor; and `pub use spec_tables::{REFS_PER_FRAME, NUM_REF_FRAMES,
  PRIMARY_REF_NONE, MAX_SEGMENTS, SEG_LVL_*, LAST_FRAME, …}`.

- [ ] **Step 1: Write the failing guard test (in `crates/platform`)**

  A compile-time guard in `crates/platform/src/vaapi/mod.rs`'s test module (or a small
  `crates/platform/tests/` file) that *names* each newly reachable item, so a regression in
  `crates/av1`'s visibility fails to compile here rather than surfacing later as a confusing error in
  the picture builder:

  ```rust
  #[test]
  fn the_av1_frame_header_parser_is_reachable() {
      use filmcraft_av1::BitReader;
      let mut r = BitReader::new(&[0u8; 4]);
      let _ = r.f(3);
      let seq = filmcraft_av1::SequenceHeader::parse(&[]).expect_err("empty is not a sequence header");
      assert!(seq.to_string().len() > 0, "{seq}");
      let _ = filmcraft_av1::NUM_REF_FRAMES;
      let _ = filmcraft_av1::REFS_PER_FRAME;
      let _ = filmcraft_av1::PRIMARY_REF_NONE;
  }
  ```

- [ ] **Step 2: Run it to verify it fails to compile**

  Run: `cargo test -p filmcraft-platform --lib vaapi::tests::the_av1_frame_header_parser_is_reachable`
  Expected: FAIL to compile — `BitReader` / the spec constants are `pub(crate)`.

- [ ] **Step 3: Widen the visibility in `crates/av1`**

  - `crates/av1/src/lib.rs:10` — `mod bits;` → `pub mod bits;` (so `BitReader` is reachable as
    `filmcraft_av1::bits::BitReader`), **or** the narrower `pub use bits::BitReader;` — choose the
    narrower of the two that compiles and note which in the commit.
  - `crates/av1/src/bits.rs:6` — `pub(crate) struct BitReader<'a>` → `pub struct BitReader<'a>`.
  - `crates/av1/src/header.rs:478` — `pub(crate) fn parse` → `pub fn parse` (it already compiles;
    the only blocker was the `BitReader` visibility). `:448` `relative_dist` and `:1316`
    `default_gm_params` → `pub` as well, since `HeaderState::new` wants the latter.
  - `crates/av1/src/header.rs` — add `impl<'a> HeaderState<'a> { pub fn new(seq: &'a SequenceHeader,
    refs: &'a mut [RefInfo; NUM_REF_FRAMES]) -> Self { Self { seq, refs, ..a state seeded exactly as
    Decoder::new seeds it } } }`. **Seed it from the same initial values `Decoder::new` uses**
    (`decoder.rs:256-259`: `LoopFilterDeltas::defaults()`, `SegmentationFeatures::default()`,
    `default_gm_params()`, `current_frame_id: 0`) — a different seed is a different decode.
  - `crates/av1/src/lib.rs` — add `pub use spec_tables::{REFS_PER_FRAME, NUM_REF_FRAMES,
    PRIMARY_REF_NONE, MAX_SEGMENTS, SEG_LVL_ALT_Q, SEG_LVL_ALT_LF_Y_V, SEG_LVL_ALT_LF_Y_H,
    SEG_LVL_ALT_LF_U, SEG_LVL_ALT_LF_V, SEG_LVL_REF_FRAME, SEG_LVL_SKIP, SEG_LVL_GLOBALMV,
    SEG_LVL_MAX, LAST_FRAME, ALTREF_FRAME, KEY_FRAME, INTER_FRAME, INTRA_ONLY_FRAME, SWITCH_FRAME,
    RESTORE_NONE, RESTORE_WIENER, RESTORE_SGRPROJ, RESTORE_SWITCHABLE, WARPEDMODEL_PREC_BITS};`

- [ ] **Step 4: Run the guard test to verify it passes**

  Run: `cargo test -p filmcraft-platform --lib vaapi::tests::the_av1_frame_header_parser_is_reachable`
  Expected: PASS (compiles and runs).

- [ ] **Step 5: Prove the software path is untouched (the critical regression gate)**

  Run: `cargo test -p filmcraft-av1` (its `oracle_intra` and `conformance` suites compare bit-exact
  against libdav1d; `conformance_vectors_extended` is ignored and downloads vectors).
  Expected: **all green, still bit-exact** — the same count and hashes as before the change. This is
  the gate that keeps `crates/av1`'s software consumers safe; a visibility-only change must not move
  a single decoded sample. If it does, the plan's premise is wrong and the work stops here.

- [ ] **Step 6: Commit**

  ```bash
  git add crates/av1 crates/platform
  git commit -m "M-AV1: make filmcraft-av1's frame header parser reachable for the VA-API record"
  ```

---

### Task 2 — The AV1 records in `ffi.rs`, measured and pinned

**Files:** `crates/platform/src/vaapi/{ffi,abi_guard}.rs`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `VaPictureParameterBufferAv1`, `VaSegmentationStructAv1`, `VaFilmGrainStructAv1`,
  `VaWarpedMotionParamsAv1`, `VaSliceParameterBufferAv1` (`#[repr(C)]`, every byte a named field incl.
  `padding_*`), plus `picture_params_av1(&[u8]) -> Option<…>` and `slice_params_av1(&[u8]) ->
  Option<…>` safe readers (mirror the VP9 pair, `ffi.rs:1485-1498`).

- [ ] **Step 1: Write the failing layout test**

  In `ffi.rs`'s `#[cfg(test)] mod tests`, assert the measured sizes and the offsets that a plausible
  field-order reading gets wrong — the six holes, the two `uint16_t`-after-`uint8_t` alignments, and
  `wm` (7×56 = 392 at offset 732):

  ```rust
  assert_eq!(size_of::<VaPictureParameterBufferAv1>(), 1160, "VADecPictureParameterBufferAV1");
  assert_eq!(offset_of!(VaPictureParameterBufferAv1, anchor_frames_list), 24);
  assert_eq!(offset_of!(VaPictureParameterBufferAv1, padding_after_anchor_frames_num), 17);
  assert_eq!(offset_of!(VaPictureParameterBufferAv1, seg_info), 84);
  assert_eq!(offset_of!(VaPictureParameterBufferAv1, pic_info_fields), 676);
  assert_eq!(offset_of!(VaPictureParameterBufferAv1, qmatrix_fields), 704);
  assert_eq!(offset_of!(VaPictureParameterBufferAv1, wm), 732);
  assert_eq!(size_of::<VaSegmentationStructAv1>(), 156);
  assert_eq!(size_of::<VaFilmGrainStructAv1>(), 176);
  assert_eq!(size_of::<VaWarpedMotionParamsAv1>(), 56);
  assert_eq!(size_of::<VaSliceParameterBufferAv1>(), 40);
  ```

- [ ] **Step 2: Run it to verify it fails**

  Run: `cargo test -p filmcraft-platform --lib vaapi::ffi::tests` (or the specific test)
  Expected: FAIL — the structs do not exist.

- [ ] **Step 3: Declare the five structs in `ffi.rs`**

  Declare them `#[repr(C)]` **field for field in header declaration order**, with a `padding_*`
  field for each of the six picture-record holes plus the sub-record holes, and `va_reserved` sized to
  fill to the measured end (picture record: `va_reserved[8]` at 1124, then 4 bytes trailing
  `padding_after_va_reserved`). The exact widths and offsets are in *The record* above; the layout
  test in Step 1 is the oracle. The bitfield unions are declared as their raw `u32`/`u16`/`u8` words
  (`seq_info_fields: u32`, `pic_info_fields: u32`, `mode_control_fields: u32`, `qmatrix_fields: u16`,
  `loop_restoration_fields: u16`, `loop_filter_info_fields: u8`, and the two nested `segment_info_fields:
  u32` / `film_grain_info_fields: u32`), each documented with its member→bit table from *The record*.
  Add the two safe readers, mirroring `picture_params_vp9` (`ffi.rs:1485`), length-checking `bytes` and
  returning a copy.

- [ ] **Step 4: Run the layout test to verify it passes**

  Run: `cargo test -p filmcraft-platform --lib vaapi::ffi::tests`
  Expected: PASS.

- [ ] **Step 5: Add both pins to `abi_guard.rs`**

  (a) Add all five records to the `records:` array in `every_declaration_matches_what_a_c_compiler_measures`
  (`abi_guard.rs:796-819`) so their `sizeof` is `_Static_assert`ed against the real headers.
  (b) Add all five to `SUBMITTED_RECORDS` (`abi_guard.rs:294-650`) field-for-field — every member with
  its C name, every hole as `("-", padding_*, size_of::<[u8; N]>())` — so
  `every_record_the_backend_submits_tiles_byte_for_byte` (`:668`) proves they tile end to end **and**
  `no_submitted_record_has_a_byte_the_header_does_not_account_for` (`:697`) has the C compiler confirm
  every `offsetof`, every field `sizeof` and **every gap width** (including the six picture-record holes
  and the trailing 4-byte pad). Add `#include <va/va_dec_av1.h>` to the C source in both tests.
  This is the "size, every offsetof, and any compiler hole" pin the branch's doctrine requires.

- [ ] **Step 6: Run the ABI guard tests**

  Run: `cargo test -p filmcraft-platform --lib vaapi::abi_guard`
  Expected: PASS — and (worth stating) verify the pins *can* fail: temporarily widen one
  `padding_*` field by a byte and confirm the tiling test goes red with the record named. (The VP9
  record's own test was verified this way.)

- [ ] **Step 7: Commit**

  ```bash
  git add crates/platform/src/vaapi/ffi.rs crates/platform/src/vaapi/abi_guard.rs
  git commit -m "M-AV1: declare the AV1 decode records and pin them against a C compile"
  ```

---

### Task 3 — The decline table for AV1

**Files:** `crates/platform/src/vaapi/{stream,picture}.rs`

**Interfaces:**
- Consumes: Task 2's `ffi` types (none needed yet — the decline is profile/feature based).
- Produces: `av1_decline_reason(profile: u8, bit_depth: u32) -> Option<&'static str>` and
  `av1_decline(profile, bit_depth) -> Option<String>`, mirroring `vp9_decline` (`picture.rs:870-889`),
  called from **both** `plan_frame` (`stream.rs:152`) and `build_av1_frame` so they cannot disagree.

- [ ] **Step 1: Write the failing decline test**

  ```rust
  #[test]
  fn av1_declines_what_it_cannot_describe_and_says_why() {
      assert!(av1_decline(0, 8).is_none(), "profile 0 at 8 bits is decoded");
      assert!(av1_decline(0, 10).is_none(), "profile 0 at 10 bits is decoded");
      assert!(av1_decline(1, 8).is_some(), "profile 1 is declined");
      assert!(av1_decline(2, 10).is_some(), "profile 2 (12-bit / 4:2:2) is declined");
      assert!(av1_decline(0, 12).is_some(), "12-bit is declined");
      // The one sentence everywhere:
      let a = av1_decline(1, 8).unwrap(); let b = format!("{:?}", av1_decline_reason(1, 8)).to_lowercase();
      assert!(a.contains(&b[..b.len().min(20)]), "plan and record must say the same thing: {a}");
  }
  ```

- [ ] **Step 2: Run it to verify it fails**

  Run: `cargo test -p filmcraft-platform --lib vaapi::picture::tests::av1_declines`
  Expected: FAIL — `av1_decline` does not exist.

- [ ] **Step 3: Implement the decline table**

  `av1_decline_reason`: profile 1 → "4:4:4 / 4:2:2, which `va_dec_av1.h:44` says this API does not
  support and every surface this backend creates is 4:2:0"; profile 2 → the same plus 12-bit; bit
  depth 12 → "12-bit, out of scope by decision". Use the VP9 shape (`vp9_decline`, `picture.rs:870`)
  so the message format matches. In `plan_frame`, replace the AV1 `FRAME_HEADER_MISSING` `Err` arm
  (`stream.rs:185-190`) with: call `av1_decline`, and on `Ok` return the real
  `Plan { profile: Profile::Av1Main, entrypoint: Vld, format, size }`. The film-grain, LST and
  resolution declines are **feature-level** and live in the record builder / decode loop (Tasks 4-5),
  not here — `plan_frame` describes the stream, not the frame.

- [ ] **Step 4: Run the decline test to verify it passes**

  Run: `cargo test -p filmcraft-platform --lib vaapi::picture::tests::av1_declines`
  Expected: PASS.

- [ ] **Step 5: Run the existing stream tests**

  Run: `cargo test -p filmcraft-platform --lib vaapi::stream`
  Expected: PASS, **except** `frame_codecs_are_declined_until_their_frame_headers_are_public`
  (`stream.rs:356`) and the AV1 assertions at `stream.rs:363-376`, `443-460`, `594-629` — these
  currently assert AV1 is declined and must be updated in this task to assert the new plan (VP9 Ruling
  1: the decline test is split, not deleted). Update them here; they are this module's tests.

- [ ] **Step 6: Commit**

  ```bash
  git add crates/platform/src/vaapi/stream.rs crates/platform/src/vaapi/picture.rs
  git commit -m "M-AV1: decline what the record cannot describe, and plan the rest"
  ```

---

### Task 4 — `VADecPictureParameterBufferAV1` from the frame header

**Files:** `crates/platform/src/vaapi/picture.rs` (builder), `crates/platform/src/vaapi/av1.rs`
(helpers), `crates/platform/src/vaapi/ffi.rs` (bitfield packers if kept there).

**Interfaces:**
- Consumes: Task 1's public `crates/av1` surface; Task 2's record structs; Task 3's decline table.
- Produces: `pub fn av1_picture_record(h: &FrameHeader, st: &HeaderState, seq: &SequenceHeader,
  refs: &Av1ReferenceSet) -> Result<Vec<u8>, String>` and
  `pub fn av1_slice_records(h: &FrameHeader, frame_bytes: &[u8]) -> Result<Vec<Vec<u8>>, String>`
  (one per tile), plus `Av1ReferenceSet { slots: [Option<u32>; 8] }` and the six bitfield packers
  (`pic_info_fields(h)`, `seq_info_fields(seq)`, …).

- [ ] **Step 1: Write the failing record-bytes test**

  Build a `FrameHeader` **from a real fixture** (parse a key frame of `av1_p0_8bit.mp4` with the now
  public `FrameHeader::parse`), build the record, read it back with `picture_params_av1`, and assert
  the byte offsets and the derived state — the "pin the submitted record's bytes so a 33-instead-of-32
  class error cannot pass" test (four wrong constants passed the whole integration suite on this
  branch without it):

  ```rust
  let (h, st) = parse_key_frame(&fixture_bytes(), &fixture_seq());
  let rec = av1_picture_record(&h, &st, &seq, &refs).expect("a key frame is describable");
  assert_eq!(rec.len(), 1160);
  let r = picture_params_av1(&rec).expect("1160 bytes is an AV1 picture record");
  assert_eq!(r.frame_width_minus1, (h.upscaled_width - 1) as u16, "upscaled, not coded");
  assert_eq!(r.pic_info_fields & 0x3, u32::from(h.frame_type), "frame_type is bits 0:2");
  assert_eq!((r.pic_info_fields >> 9) & 1, u32::from(h.use_superres), "use_superres is bit 9");
  assert_eq!(r.base_qindex, h.quant.base_q_idx as u8);
  // An empty slot is VA_INVALID_SURFACE, never a zero:
  assert_eq!(r.ref_frame_map[3], INVALID_SURFACE, "an empty slot must not name some surface");
  ```

- [ ] **Step 2: Run it to verify it fails**

  Run: `cargo test -p filmcraft-platform --lib vaapi::picture::tests::an_av1_record_is_built_from_the_frame_header`
  Expected: FAIL — the builder does not exist.

- [ ] **Step 3: Implement the six bitfield packers**

  One packer per union, each a `for (at, width, name)` positions table like `vp9_pic_fields`
  (`picture.rs:983-1041`) — positions **from the measured table** in *The record*, each with the
  member's name so a failure names the field. Values come from `h`/`seq` per the field mapping.

- [ ] **Step 4: Implement `av1_picture_record`**

  Field for field per *The record*. Rules that are not obvious: (a) every fixed-width field is
  bounded with `u8::try_from`/`u16::try_from`/`i8::try_from`/`i16::try_from` **before** it is stored, so
  a bitstream value too wide for its record member is an `Err` (the VP9 `vp9_field_check` pattern);
  (b) `ref_frame_map` empty slots → `INVALID_SURFACE`; (c) `va_reserved` → zero; (d) the
  `padding_*` fields → zero (they are holes C leaves uninitialised — the whole reason they are named);
  (e) a `film_grain apply_grain == 1` or a `large_scale_tile` frame is declined with its reason, before
  any field is packed; (f) `frame_width_minus1`/`frame_height_minus1` are **upscaled**.

- [ ] **Step 5: Implement `av1_slice_records` (the tile walk) — and its test**

  Walk the frame's tile-group OBU bytes: for `num_tiles = tile_info.cols * tile_info.rows`, each tile
  but the last is preceded by a `tile_size_bytes`-byte little-endian length prefix (`+1` for the last),
  **cross-checked against `filmcraft_av1::Decoder`'s own `tile_group` split** on a real multi-tile
  fixture. The test:

  ```rust
  let h = parse_tile_header(&tiled_fixture_bytes());
  let ours = av1_slice_records(&h, &frame_bytes).expect("tiles are describable");
  let theirs = filmcraft_av1_decoder_tile_extents(&frame_bytes, &h.tile_info); // oracle via Decoder
  assert_eq!(ours.len(), h.tile_info.cols * h.tile_info.rows, "one slice record per tile");
  // each (size, offset, row, col) matches the decoder's own tile extents
  ```

  This is the riskest derived value in the milestone and gets its own assertion against the decoder
  that already computes it — not against a reimplementation of the same reasoning.

- [ ] **Step 6: Run the record and tile tests**

  Run: `cargo test -p filmcraft-platform --lib vaapi::picture::tests::an_av1`
  Expected: PASS.

- [ ] **Step 7: Prove the record test can fail (the mutation check this branch requires)**

  Flip one field in a copy of the test's input header (e.g. set `base_q_idx` to a value the record
  would truncate) and confirm `av1_picture_record` returns `Err` rather than a wrong record. Record
  the mutation and its result in the commit message.

- [ ] **Step 8: Commit**

  ```bash
  git add crates/platform/src/vaapi/picture.rs crates/platform/src/vaapi/av1.rs
  git commit -m "M-AV1: build the picture and slice records from the parsed frame header"
  ```

---

### Task 5 — `Av1State`: the reference slots, refresh, and the decode loop

**Files:** `crates/platform/src/vaapi/{av1,picture,decoder}.rs`

**Interfaces:**
- Consumes: Task 4's `av1_picture_record` / `av1_slice_records`.
- Produces: `pub struct Av1State { headers: HeaderState, seq: SequenceHeader, slots: [Option<Av1Ref>; 8] }`
  with `new`, `reset`, `references() -> Av1ReferenceSet`, `live_surfaces() -> Vec<u32>`,
  `frame(&mut self, sample: &[u8]) -> Result<Av1Frame, String>`, `commit(&mut self, &Av1Frame, surface: u32)`;
  `pub struct Av1Frame` with `header()`, `record() -> Option<&[u8]>`, `slices() -> &[Vec<u8>]`,
  `re_show() -> Option<(usize, u32)>`, `show()`, `refresh_flags() -> u8`. Plus
  `VaDecoder::picture_av1_frame` and the AV1 arm of `picture_frame`.

- [ ] **Step 1: Write the failing state test**

  The state model is the risky part, so the test pins the **mapping**, not just pictures that happen
  to match (VP9 Task 3's standing lesson). On a real key+inter sequence:

  ```rust
  let mut st = Av1State::new(&seq);
  let k = st.frame(&key_bytes).unwrap();            // key frame: refresh_frame_flags == 0xff
  assert_eq!(k.refresh_flags(), 0xff, "a key frame refreshes every slot");
  st.commit(&k, SURF_K);
  assert_eq!(st.live_surfaces(), vec![SURF_K; 8], "all eight slots hold the key");
  let p = st.frame(&inter_bytes).unwrap();           // inter: names 7 slots via ref_frame_idx
  assert_eq!(p.record().map(|r| picture_params_av1(r).unwrap().ref_frame_map), Some([SURF_K;8]));
  assert_ne!(p.refresh_flags(), 0xff, "an inter refreshes a subset");
  // An Err does not move the buffer:
  let before = st.live_surfaces();
  let _ = st.frame(&garbage);
  assert_eq!(st.live_surfaces(), before, "a refused frame must not refresh the slots");
  ```

- [ ] **Step 2: Run it to verify it fails**

  Run: `cargo test -p filmcraft-platform --lib vaapi::av1::tests`
  Expected: FAIL — `Av1State` does not exist.

- [ ] **Step 3: Implement `Av1State`**

  One `HeaderState` threaded across the run; eight slots; `reset` clears **both** halves (a stale slot
  is a reference predicted from the wrong place). `frame()` parses the header into `self.headers`,
  builds the picture record and the slice records together, and moves **nothing** — `commit(&frame,
  surface)` applies `refresh_frame_flags` only after the driver has rendered, so a record always
  describes the buffer *before* its own frame. Mirror `Vp9State::frame`/`commit`
  (`picture.rs:1906`, `1817`) including its documented "an `Err` may have advanced the parse state"
  note (AV1's parser writes `st.refs`/mark_ref_frames before the record is built — the same honest
  caveat applies and is the reason `refused` latches until `reset`).

- [ ] **Step 4: Implement the `show_existing_frame` re-show**

  `re_show()` returns the slot the header named and its surface; `record()`/`slices()` are `None`; the
  decode loop re-shows the surface with no decode (AV1 7.21), exactly as `VaDecoder` does for a VP9
  re-show (`decoder.rs`, `picture_vp9_frame`'s `re_show` branch). Test: a stream containing a
  `show_existing_frame` unit produces a picture at that pts and is bit-exact (Review Focus 3).

- [ ] **Step 5: Wire the decode loop (`decoder.rs`)**

  In `picture_frame`, dispatch AV1: parse the frame **first** (a re-show needs no surface, so the
  surface is taken after — the opposite of the NAL order, same as VP9), build the picture + slice
  records, submit them with the frame bytes in **one** `vaRenderPicture` beside the surface, sync,
  read back, `commit`, and on `show_frame == false` return no picture. Add the AV1 arm to
  `surface_pool` (`decoder.rs:137`): its `FrameCodec::Av1` arm already exists but currently refuses
  ("a frame-coded stream was planned without the picture-parameter records this backend cannot
  build", `decoder.rs:142-145`) — replace that with a real pool size of `NUM_REF_FRAMES`(8) + 1 = 9,
  same as VP9, and add AV1 to `the_surface_pool_is_always_larger_than_the_picture_buffer`
  (`decoder.rs:1058`), which today enumerates H.264, HEVC and VP9 only. Add `check_av1_shape`: refuse
  a frame whose `upscaled_width/height` disagree with the session surface (Review Focus 1). Route an
  `Av1State::frame` `Err` through `VaDecoder::decode`'s `refused` latch (VP9 Ruling 10.2) rather than
  converting it to a skip.

- [ ] **Step 6: Run the state and decode-loop tests**

  Run: `cargo test -p filmcraft-platform --lib vaapi::`
  Expected: PASS.

- [ ] **Step 7: Commit**

  ```bash
  git add crates/platform/src/vaapi/av1.rs crates/platform/src/vaapi/decoder.rs crates/platform/src/vaapi/picture.rs
  git commit -m "M-AV1: carry the reference state and the eight slots, and wire the decode loop"
  ```

---

### Task 6 — Fixtures and the `Tier::EveryPicture` wiring

**Files:** `crates/platform/tests/common/mod.rs`, `crates/platform/tests/vaapi.rs`

**Interfaces:**
- Consumes: Task 5's decode path.
- Produces: four `FIXTURES` entries; `Tier::EveryPicture` for the decodable ones.

- [ ] **Step 1: Add the four fixtures to `FIXTURES`**

  In `crates/platform/tests/common/mod.rs`, append `av1_p0_8bit.mp4`, `av1_p0_10bit.mp4`,
  `av1_tiled.mp4`, `av1_intra_only.mp4` with their full ffmpeg argument lists (libaom-av1, one GOP per
  ~24 frames, B-frames for reordering, ≥2 GOPs, the pixel formats and tile options per *Fixtures*),
  each with a comment saying **what the fixture proves** — the VP9 entries just added to this same
  table set that precedent, and the comment is what stops a fixture from silently becoming
  decorative. `streams()` already assigns `EveryPicture` to every non-`h264_high.mp4` entry.

- [ ] **Step 2: Split the AV1 decline test**

  `an_av1_stream_is_declined_before_a_context_exists` (`tests/vaapi.rs`) currently asserts AV1 is
  declined. Per VP9 Ruling 1: split it — the decodable half becomes a parity fixture
  (`EveryPicture`), and a new test asserts a **feature** decline (profile 1, or an LST / film-grain
  AV1) still declines with its reason. Do not simply delete the decline assertion.

- [ ] **Step 3: Raise the oracle count floor and name the AV1 fixtures**

  In `bit_exact_with_the_software_decoder`, raise the bare `compared >= 3` count floor and add a
  by-name assertion that each AV1 fixture is in the compared set — following the pattern the VP9
  milestone just established in the same test. A run that silently dropped AV1 must be loud (VP9
  Ruling 2: scope without a fixture is a claim, not coverage).

- [ ] **Step 4: Run the parity tests**

  Run: `cargo test -p filmcraft-platform --test vaapi bit_exact_with_the_software_decoder -- --nocapture`
  Expected: PASS, with the AV1 fixtures scored `EveryPicture`; a VA-API AV1 decode that returns the
  wrong count or wrong pixels fails here.

- [ ] **Step 5: Commit**

  ```bash
  git add crates/platform/tests/common/mod.rs crates/platform/tests/vaapi.rs
  git commit -m "M-AV1: AV1 parity fixtures in the FIXTURES table, decoded every picture"
  ```

---

### Task 7 — The docs

**Files:** `README.md`, `ROADMAP.md`, `crates/platform/README.md`, `docs/architecture.md`,
`docs/adr/0001-platform-ffi.md`. **Not** `docs/vaapi-progress.md` (owned by another agent this
session — this plan records the §5 enabler correction above and leaves the file alone).

- [ ] **Step 1: Update the honest-limitations text**

  The six locations the VP9 milestone updated (`README.md:187`/`:289`, `ROADMAP.md:39`/`:51`/`:157`,
  `crates/platform/README.md:73`/`:191`, `docs/architecture.md:75`/`:232-235`,
  `docs/adr/0001-platform-ffi.md:28`) must now say AV1 Main at 8/10-bit decodes on Linux, **and** list
  the declines (profiles 1-2, 12-bit, monochrome, film grain, LST). A stream left decoded in the docs
  that returns an error is the open issue the VP9 ledger flagged. **Confirm on real hardware that the
  decoded fixtures pass before claiming it in `README.md`** (VP9's docs-then-hardware ordering was
  deliberately gated for this reason and the gate has not been confirmed by the owner).

- [ ] **Step 2: Record the enabler correction**

  `docs/vaapi-progress.md` §5's "module-visibility wall" framing is corrected (in the file, by its
  owner) to the narrow-public-surface shape measured in this plan. This task **notes the correction
  in its commit message** but does not edit the file.

- [ ] **Step 3: Run the gates**

  Run: `cargo xtask ci` (expect red on `crates/gpu` only, pre-existing — §7 of
  `docs/vaapi-progress.md`; every other gate green). `cargo xtask assets` must pass.

- [ ] **Step 4: Commit**

  ```bash
  git add README.md ROADMAP.md crates/platform/README.md docs/architecture.md docs/adr/0001-platform-ffi.md
  git commit -m "docs: AV1 decodes on Linux at 4:2:0 8/10-bit, with its declines stated"
  ```

---

## Verification

- **Bit-exact against ffmpeg, picture for picture** — the `Tier::EveryPicture` gate (Task 6), the same
  standard H.264, HEVC and VP9 hold. Measured, not assumed: the tests fail on a wrong pixel.
- **Record pins: `sizeof`, every `offsetof`, every hole** — against a C compile, in `abi_guard.rs`
  (Task 2), and verified able to fail.
- **The record's bytes pinned** — a byte-offset + derived-state test per record (Task 4), plus a
  mutation check that a truncating field returns `Err`.
- **The tile walk cross-checked against the decoder** that already computes it (Task 4 Step 5).
- **The reference/refresh state pinned by a mapping test**, not just matching pictures (Task 5).
- **The enabler keeps the software path bit-exact** — `cargo test -p filmcraft-av1` green and
  bit-exact after every visibility change (Task 1 Step 5, and repeated in the final gate).
- **`cargo xtask ci`** before merge — red on `crates/gpu` for pre-existing, unrelated reasons.

**Decode is bit-for-bit comparable; encode is not.** A hardware decoder and a software decoder must
agree sample-for-sample (this is the milestone's contract — "interchangeable with FilmCraft's own
software decoder"), and a VA-API **decode** is asserted bit-exact. A hardware *encoder* and a software
encoder are different algorithms and **cannot** be compared bit-for-bit anywhere — that weaker gate
belongs to the encode milestone (`docs/superpowers/plans/2026-10-09-vaapi-linux-encode.md`), not here.
Nothing in this plan may be phrased as a hardware-vs-software encoder comparison.

---

## Open questions

Honest, unresolved items — each is an open question rather than a confident guess, per the house
style. None blocks Task 1-3.

1. **`width_in_sbs_minus_1[63]` / `height_in_sbs_minus_1[63]` derivation.** The record wants, per
   tile, the tile's width/height minus 1 in **superblocks**. `FrameHeader::tile_info` exposes
   `mi_col_starts`/`mi_row_starts` (mi units) and `cols`/`rows` but **not** the superblock grid, and
   `SequenceHeader::use_128x128_superblock` is what fixes the sb size. The builder needs
   `sb_cols = (mi_cols + (31 or 15)) >> (5 or 4)` from `mi_cols` + `use_128x128_superblock`, then the
   per-tile sb extents from the mi starts. **I could not confirm from `filmcraft-av1`'s public API
   that `mi_cols` and `use_128x128_superblock` are both reachable** — they are (`FrameHeader::mi_cols`
   :347, `SequenceHeader::use_128x128_superblock` :53), so this resolves in Task 4 by computing them;
   but **whether the last tile's sb extent must be omitted** (the header says the last tile's "are
   computed from ones of the other tiles and frame_width/height … not necessarily specified") is not
   settled here. Task 4's tile test should assert against the decoder and settle it empirically.
2. **`lr_unit_shift` / `lr_uv_shift` derivation.** The record wants 2-bit `lr_unit_shift` and 1-bit
   `lr_uv_shift`; `FrameHeader::lr` exposes `loop_restoration_size[3]` (pixels) and `uses_lr`, from
   which the shifts are *derivable* but not directly carried. Task 4 derives them and pins the record
   bytes; **whether the driver's interpretation matches ours is only settled by the hardware oracle**
   (Task 6). Until then a `loop_restoration_size` that derives to an out-of-range shift is an `Err`,
   not a clamp.
3. **`gm_params` scaling into `wmmat[8]`.** `VAWarpedMotionParamsAV1::wmmat[8]` corresponds to the
   spec's `gm_params[6]` (`va_dec_av1.h:222`); `FrameHeader::gm_params` is `[[i32;6];8]` already at
   `WARPEDMODEL_PREC_BITS` scale. The two extra `wmmat` entries are zero. **Whether the driver wants
   them at a different scale than the parser's is a hardware question**, answered by the oracle, not
   by the header.
4. **Whether `AVConfigAttribDecAV1Features` should gate the decode.** The header declares
   `lst_support` (`va_dec_av1.h:55-67`) and the driver advertises it; this plan declines LST at the
   *record* level. Whether `supports()` should also query the feature and pre-decline is a design
   choice left open — it is defensible either way and does not change the decode result.
5. **`error_resilient_mode` propagation.** The record carries the bit; whether the driver clears its
   own DPB on an error-resilient frame in a way our slot map must mirror is not settled from the
   header. Recorded, not assumed; the oracle would only reach it on a stream that uses error
   resilience.