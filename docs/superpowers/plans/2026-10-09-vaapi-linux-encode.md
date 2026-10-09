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
- VP9 has no hardware encoder anywhere; the capability report now says so per profile. AV1 Main reports
  `EncSlice` on this machine and is **out of scope** until something proves the encoder actually works.

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

## Constraints carried from the decode milestone

- Safe code outside `vaapi/ffi.rs`; never-crash `deny` lints outside `#[cfg(test)]`; every input-derived
  number hostile; a value we cannot derive faithfully is an `Err`, never a default.
- Every decline documented with the reason, in the same shape as the decode decline tables.
- `cargo xtask ci` before merge. It is currently red on `crates/gpu` for pre-existing reasons unrelated
  to this work — see the ledger.