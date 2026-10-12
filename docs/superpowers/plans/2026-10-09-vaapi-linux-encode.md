# Linux hardware video encode through VA-API

Follow-on to the VA-API **decode** milestone (`vaapi-linux-decode`). Decode is H.264 and HEVC at 4:2:0,
8/10-bit. Encode is the same two codecs, same platform crate, same `dlopen` boundary.

## Scope

- **H.264 and HEVC, VA-API slice encode**, opt-in per export through the existing
  `Export ▸ Hardware encoding` setting (`crates/export/src/settings.rs:64`), which already gates the macOS
  and Windows encoders. Off by default, because hardware encoders make different streams and exports are
  otherwise byte-identical across machines — that reason is already written down there and still holds.
- **No B-frames initially**, matching the constraint already documented for the macOS VideoToolbox encoder.
  Whether iHD emits B-frames is a measured question (Task 3), not an assumption.
- VP9 has no hardware encoder **on this driver**, which the capability report says per profile and
  `vaapi/vp9_enc.rs` carries the evidence for; see the task status below. AV1 Main reports `EncSlice` on
  this machine and is **out of scope** until something proves the encoder actually works.

## The decisive difference from decode

Decode is the app parsing a bitstream and handing the driver every parameter. **Encode inverts the data
flow: the app hands the driver a picture, and the driver writes the slice headers.** Two consequences shape
everything below:

1. **We must read the encoded bitstream back out** of the buffer the driver filled
   (`VAEncSliceParameterBuffer` → `slice_data_offset` / `slice_data_size` per slice), rather than
   submitting slice headers we parsed.
2. **The codec configuration is the driver's, not ours.** `VideoEncoder::sample_entry()` already allows
   for this — "may only be complete after the first frame" — because `avcC` / `hvcC` has to be built from
   the SPS/PPS/VPS the driver emitted. We parse them back out of our own output.

Neither has an analogue in the decode milestone. There is no existing code to reuse for either.

## Verification — and it is weaker than decode

Decode could assert 72 of 72 pictures bit-exact against ffmpeg. **Encode cannot**: a hardware encoder and
our software encoder are different algorithms, so bit-equality is not available at any level. The gate is
therefore:

- The exported file **decodes with ffmpeg to the right pictures**, compared against the *source* within a
  lossy tolerance appropriate to the bitrate — not against a software encode, which differs legitimately.
- The stream is **well-formed**: `avcC`/`hvcC` consistent with the bitstream it describes, no gaps,
  monotonic timestamps, and `sample_entry()` complete only once the driver has actually emitted parameter
  sets.
- Size and rate control land in the neighbourhood the settings asked for.

This is a materially weaker gate than decode's and should be described that way in the docs rather than
borrowing decode's language. A "matches ffmpeg" claim for an encode milestone would be meaningless.

## Tasks

1. **Spike: one synthetic frame, one slice, end to end.** Create a context for `H264Main`, feed one
   surface, read the driver's output, mux it, decode with ffmpeg. This is the task that de-risks the whole
   milestone and it is the one most likely to fail: every unknown above lives in it. Nothing else should
   start until it has produced a decodable NAL stream once.
2. **Encode FFI.** `VAH264Enc`, `VAH265Enc`, `VAConfigAttribRateControl`, the encode buffer types, slice
   header parsing structs. Hand-resolved like the decode FFI, `dlopen` only, never linked — the licence
   ruling in ADR 0001 rests on that and linking would void it.
3. **Parameter submission with honest declines.** Sequence/misc, picture, and per-slice records. Every
   setting we cannot express is a decline, never a default: a wrong QP or frame type is a silently wrong
   picture, the same failure class as the H.264 weight bug.
4. **`sample_entry` from the driver's parameter sets.** Parse the emitted SPS/PPS/VPS into `avcC`/`hvcC`.
5. **Frame loop, B-frames, flush.** `media_start()` is meaningful only if the driver emits B-frames;
   measure before implementing.
6. **Registration and settings wiring**, mirroring the decode side, plus the `perf.stats` counters.
7. **Oracle tests**, at the weaker standard above, stated as such.

## Task status

The seven tasks are done for H.264. What follows is what each one actually established, so a reader does
not have to take the list above on trust; the measurements themselves are in
[`docs/vaapi-progress.md`](../../vaapi-progress.md) §3.

| # | state | what it produced |
|---|---|---|
| 1 | done | One synthetic picture, one slice, end to end, through the driver's own readback (`vaAcquireBufferHandle` → `DRM_IOCTL_PRIME_HANDLE_TO_FD` → `mmap`) — the readback was the blocker, and every byte now comes from a mapping this backend issued. |
| 2 | done | Encode FFI, hand-resolved behind `dlopen` like the decode side. The audit of all 32 entry points found three records declared smaller than the header's, one of them live; every entry point's arity is now derived mechanically and checked against the installed headers. |
| 3 | done | Sequence / picture / per-slice records with honest declines, each carrying its reason. Rate control was the one the plan assumed would be hard and it was the one whose **measurement** was content-dependent: a luma ramp's ceiling on this driver is ~162 kbit/s, so an 8000 kbit/s target was unreachable by construction and made VBR look as if it ignored the target and CBR as if it faked one. On content that can carry the bits it tracks at 89–94 %. |
| 4 | done | The driver emits **no** NAL 7/8 under any `VAConfigAttribEncPackedHeaders` value, so the `avcC` is built from parameter sets FilmCraft derives from the records it submitted — and each picture's own NALs are parsed back against those sets and compared, so the derived path checks itself rather than merely agreeing with itself. `sample_entry()` refuses before the first picture. |
| 5 | done | Frame loop and flush, with B-frames **measured and refused**: the driver reports it could (`VAConfigAttribEncMaxRefFrames` answers (3, 1)) and emits none. `log2_max_frame_num` was widened from 4 to 8 bits so `MaxFrameNum` is 256, which is what makes a preset's key-frame interval expressible — **8 of the 11 H.264 presets FilmCraft ships now export**. |
| 6 | done | `vaapi_encoder_factory` registered behind a `Once`, every decline counted in `perf.stats`, plus `H264Profile::Auto` so a preset that names no profile gets what the machine produces and the summary prints what it resolved to. |
| 7 | done | `tests/vaenc.rs`, at the weaker gate: the muxed file decodes with ffmpeg to the right pictures **against the source**, the `avcC` is checked against the bitstream it describes, ffmpeg's stderr must be empty, and the size must land in a band derived from the measurement (0.80..=1.05, because the measurement is 0.94). |

**What the work did outside the seven tasks**, and where the scope above turned out to be wrong:

- **HEVC** was in scope ("Encode is the same two codecs") and is now proven: one I frame then P frames that
  predict, decoded clean by ffmpeg at 320×240. The blocker two agents failed on was **four packed-header
  buffers the path configured for and never sent** — this driver writes no slice header of its own. Its
  export reachability is in progress.
- **VP9** was scoped out on the assumption "there is no hardware VP9 encoder anywhere". That is **measured**,
  not assumed: all four VP9 profiles advertise decode only and every encode entrypoint answers
  `VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT` on this driver, while `va_enc_vp9.h` and `VAConfigAttribEncVP9`
  do exist. The answer is a file carrying the evidence rather than a stub.
- **AV1** was scoped out "until something proves the encoder actually works". The driver advertises
  `EncSlice` for profile 0, six records are now declared and pinned against a C compile, and **nothing
  submits them** — so the scope note stands and the records are groundwork.

## Constraints carried from the decode milestone

- Safe code outside `vaapi/ffi.rs` and `vaapi/drm.rs` (the two FFI modules — libva and plain DRM); never-crash
  `deny` lints outside `#[cfg(test)]`; every input-derived number hostile; a value we cannot derive faithfully
  is an `Err`, never a default.
- Every decline documented with the reason, in the same shape as the decode decline tables.
- `cargo xtask ci` before merge. It was red on `crates/gpu` for most of this plan's life, for reasons this
  work did not cause; both of those failures are fixed now (a texel-centre snap in the compositor's
  bilinear tap, and one process-wide `wgpu::Instance`), and one instance of the same instance-per-test
  pattern is still unfixed in `crates/golden/tests/golden.rs:742`.