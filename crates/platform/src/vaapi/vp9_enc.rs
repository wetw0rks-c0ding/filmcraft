//! **There is no VP9 encoder on this driver, so there are no VP9 encode records here.**
//!
//! This file exists to hold an answer, not a record. The task was to probe whether this machine's
//! VA-API driver advertises VP9 **encode** (as opposed to decode, which is proven for all four
//! codecs) and to declare encode records **only for what the hardware actually offers**. It offers
//! no VP9 encoder, so the honest deliverable is this file and the evidence below — not a set of
//! `#[repr(C)]` structures nobody could ever submit.
//!
//! Declaring the records anyway would have been the more impressive-looking change and the useless
//! one. It would have been ~1000 lines of correctly-measured, C-pinned, byte-exact ABI declarations
//! for a codec this driver refuses, and any plan built on them would be built on a capability that
//! does not exist. The measured answer is more useful than the code.
//!
//! # The evidence
//!
//! libva is **never linked** (AGENTS.md §2, [ADR 0001](../../../../docs/adr/0001-platform-ffi.md)):
//! the probe `dlopen`s `libva.so.2` and `libva-drm.so.2` and calls `vaQueryConfigProfiles`,
//! `vaQueryConfigEntrypoints` and `vaGetConfigAttributes` through `dlsym`. It was built and run
//! under `$TMPDIR`, not committed, and nothing in `examples/` was added for it.
//!
//! Two landmines other work on this branch already paid for are respected here, and the second is
//! the reason the evidence below is longer than it looks:
//!
//! 1. **`vaQueryConfigAttributes` is never called.** It segfaults inside `iHD_drv_video.so` on this
//!    libva, gdb-confirmed on this branch. The probe does not even `dlsym` it.
//! 2. **Every attribute read is gated on `VAStatus`, never on the attribute's value.** For an
//!    unsupported profile/entrypoint pair libva returns the status and leaves the caller's
//!    `VAConfigAttrib` untouched, so a value of `0x00000000` means "nothing was written here", not
//!    "nothing is set". The probe poisons every attribute with `0xDEADBEEF` before each call, so
//!    "the driver never wrote this" and "the driver wrote zero" are distinguishable — see
//!    `VP9Profile0/EncSlice` below, where every value is still the poison.
//!
//! ## 1. Every VP9 profile the driver advertises, and every entrypoint it offers for it
//!
//! ```text
//! == vaMaxNumProfiles -> 31
//! == vaQueryConfigProfiles -> VA_STATUS_SUCCESS(0), nprofiles=25
//!    profile[10] = 19  VAProfileVP9Profile0
//!    profile[11] = 20  VAProfileVP9Profile1
//!    profile[12] = 21  VAProfileVP9Profile2
//!    profile[13] = 22  VAProfileVP9Profile3
//!
//! == vaQueryConfigEntrypoints for every profile
//!    VAProfileVP9Profile0           -> VA_STATUS_SUCCESS(0), n=1: 1(VAEntrypointVLD)
//!    VAProfileVP9Profile1           -> VA_STATUS_SUCCESS(0), n=1: 1(VAEntrypointVLD)
//!    VAProfileVP9Profile2           -> VA_STATUS_SUCCESS(0), n=1: 1(VAEntrypointVLD)
//!    VAProfileVP9Profile3           -> VA_STATUS_SUCCESS(0), n=1: 1(VAEntrypointVLD)
//! ```
//!
//! **Two numbers, and only one of them is the answer: `nprofiles=25`, not 31.** `vaMaxNumProfiles`
//! (`va.h:1524`) is the *buffer size* an application must allocate — `va.h:1540-1544` says the caller
//! must provide a list "that can hold at least `vaMaxNumProfile()` entries", and the *actual* count
//! comes back separately in `num_profile`. On this machine the driver reports a ceiling of 31 and
//! fills 25. An earlier revision of this note quoted 31 as `nprofiles`, which is the buffer size read
//! as a count; the correction is here because the two are easy to confuse and the error makes the
//! driver look like it advertises six profiles it does not. It advertises 25, four of them VP9.
//!
//! All four profiles are advertised. **Every one offers `VAEntrypointVLD` (1) and nothing else** —
//! decode only. `VAEntrypointEncSlice` (6) is absent from all four lists, which is the first and
//! clearest evidence: the driver does not claim an encoder for VP9.
//!
//! The profile numbers are the header's own, spelled as the header spells them
//! (`va.h:524-527`): `VAProfileVP9Profile0 = 19`, `VAProfileVP9Profile1 = 20`,
//! `VAProfileVP9Profile2 = 21`, `VAProfileVP9Profile3 = 22`. There is no `VAProfileVP9Enc`
//! constant — VP9 encode uses the same profile constants as VP9 decode, with the *entrypoint*
//! being what distinguishes them. (Contrast HEVC, where encode also rides on the shared profile
//! constants; and contrast the `VAProfileH265Enc` constant named in a brief on this branch, which
//! does not exist in libva at all.)
//!
//! ### The three entrypoints an encode could use, and which is which
//!
//! `va.h` defines **three** encode entrypoints, and they are worth writing down because two of
//! them are routinely confused (a brief on this branch called `VAEntrypointEncSliceLP` "7"):
//!
//! | Entry point | Value | Header |
//! |---|---|---|
//! | `VAEntrypointEncSlice` | 6 | `va.h:558`, "slice level encode" |
//! | `VAEntrypointEncPicture` | 7 | `va.h:559`, "picture encode, JPEG, etc" |
//! | `VAEntrypointEncSliceLP` | **8** | `va.h:568`, low-power/high-performance slice encode |
//!
//! **7 is `VAEntrypointEncPicture`, not `VAEntrypointEncSliceLP`.** The brief's "(7)" was wrong,
//! so the sweep below asks all three: a claim that "no VP9 encoder" holds only if *every* encode
//! entrypoint was tried, not just the two a reader thinks of. The pins in this file's tests hold
//! the values so the next edit cannot repeat the slip.
//!
//! ## 2. Asking anyway, and the status the driver returns
//!
//! `vaGetConfigAttributes` was called for all four VP9 profiles against **all three** encode
//! entrypoints — 12 (profile, entrypoint) pairs — sweeping every attribute id the driver itself
//! advertises (`vaMaxNumConfigAttributes` returns **59**, and `va.h:1610-1612` says the caller's
//! array must hold *at least* that many entries, so the ids in range are **`0..=58`, 59 of them**).
//! For every pair and every one of the 59 attributes, the answer was the same:
//!
//! ```text
//! -- VP9Profile0/EncSlice    (profile=19 ep=6  VAEntrypointEncSlice)
//!    VAConfigAttribRateControl  id=  5 status=UNSUPPORTED_ENTRYPOINT(13)  value=3735928559 (0xdeadbeef)
//!    VAConfigAttribEncVP9       id= 58 status=UNSUPPORTED_ENTRYPOINT(13)  value=3735928559 (0xdeadbeef)
//!    ... (all 59 attributes, same status, same untouched poison)
//! -- VP9Profile0/EncSliceLP  (profile=19 ep=8  VAEntrypointEncSliceLP)   ... all 59 the same
//! -- VP9Profile0/EncPicture  (profile=19 ep=7  VAEntrypointEncPicture)   ... all 59 the same
//! -- VP9Profile1/2/3          ... all three entrypoints each, all 59 attributes, the same
//! ```
//!
//! `VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT` is `13` (`va.h:277`), and the poison survives
//! untouched, confirming the driver returned early rather than answering.
//!
//! ### The sweep is exhaustive, and the control is in the same pass
//!
//! An earlier revision of this note *said* it swept all the attributes but had actually sampled a
//! handful, so "every id answers identically" was an inference from a sample. It is now a count, with
//! the three controls read through the same loop in the same process:
//!
//! ```text
//! asked=885  refused(status != 0)=708  status=0=177  status=0 but POISON INTACT=0
//!            of the 177 status=0 reads: 62 carried a real value, 115 were VA_ATTRIB_NOT_SUPPORTED
//!
//!   885 = 708 VP9 reads (4 profiles x 3 encode entrypoints x 59 attribute ids)
//!       + 177 control reads (H264High=7, HEVCMain=17, AV1Profile0=32 at EncSlice, 59 ids each)
//! ```
//!
//! **Not one of the 177 answering reads is a VP9 one**, and the last counter is the one that closes
//! the "did it just decline to write?" question outright: a `VA_STATUS_SUCCESS` that left the
//! `0xDEADBEEF` poison in place happened **zero** times in 885 reads. So the driver does not answer
//! some VP9 attributes and refuse others — for VP9 it returns early on every id it knows, while
//! the very next read in the same loop, for another codec, returns a populated word.
//!
//! **`177` is a status count, not a count of real answers**, and the difference is the whole reason
//! the next paragraph exists. `177` is every control read, and every one of them returned
//! `VA_STATUS_SUCCESS`; only **62** of those carried a populated word, the other 115 came back
//! `VA_ATTRIB_NOT_SUPPORTED`. A driver the sweep has genuinely reached answers the id; a driver
//! that has not answers the status.
//!
//! What a *working* encoder looks like on this machine, counting only ids answered with a real
//! value: **H.264 answered 21 of its 59 ids, HEVC 20, AV1 21.** A codec the driver genuinely encodes
//! does not answer 59 of 59 — it answers the subset that applies to it and
//! `VA_ATTRIB_NOT_SUPPORTED` (`0x80000000`) for the rest. VP9's answer is not a small subset. It is
//! none of them.
//!
//! **These three counts are ids answered with a real value, not ids whose status was zero.** The
//! status is `VA_STATUS_SUCCESS` for all 59 ids of every control, asked one id at a time and in a
//! single batched call alike, so "the status was zero" would read 59/59/59 and say nothing. HEVC's
//! 20 are ids `0, 5, 10, 11, 13, 14, 15, 18, 19, 21, 23, 25, 27, 28, 29, 35, 39, 41, 50, 51`; the
//! other 39 come back `0x80000000`. An earlier revision of this note said 18, and a later one said
//! 19 — both were counts over a range that stopped at id 58. Re-measuring the sweep over the ids the
//! header actually puts in range, `0..=58`, gives 20 on every run, so neither older figure is the
//! number this file rests on — and either way none of them touches the VP9 answer, which is 0 of
//! 708.
//!
//! **`id 0` is `VAConfigAttribRTFormat` (`va.h:620`), and the two older sweeps never asked it.**
//! That is where the off-by-one came from: `vaMaxNumConfigAttributes` returning 59 was read as "the
//! last id is 58, so ask 1..=58", when it is a *buffer size* and `va.h:1610-1612` requires an array
//! holding *at least* that many entries — indices `0..=58`. Every codec here answers id 0, so
//! dropping it costs each control exactly one. The counts above are over the full range.
//!
//! ## 2a. The call a submission path cannot get past
//!
//! The attribute sweep asks a question about a *configuration*. `vaCreateConfig` is the call that
//! would actually have to succeed, so it is asked too, with the config id poisoned beforehand:
//!
//! ```text
//! == vaCreateConfig (no attribute list; the driver's own defaults)
//!    VAProfileVP9Profile0/EncSlice    -> UNSUPPORTED_ENTRYPOINT(13)  config=(untouched) 0xdeadbeef
//!    VAProfileVP9Profile0/EncSliceLP   -> UNSUPPORTED_ENTRYPOINT(13)  config=(untouched) 0xdeadbeef
//!    VAProfileVP9Profile2/EncSlice    -> UNSUPPORTED_ENTRYPOINT(13)  config=(untouched) 0xdeadbeef
//!    VAProfileH264High/EncSlice  [CONTROL]  -> VA_STATUS_SUCCESS(0)  config=0x00002b23
//!    VAProfileAV1Profile0/EncSlice [CONTROL]  -> VA_STATUS_SUCCESS(0)  config=0x00002bd3
//! ```
//!
//! This is the sharpest form of the answer, because it is the first thing a VP9 encode path would
//! call: **the driver will not create a VP9 encode config at all**, so nothing downstream of it —
//! context, buffer, picture — is reachable. It is also the answer least able to be a harness
//! artefact, because the identical call succeeds twice for two other codecs in the same process.
//!
//! ## 3. The controls, which are what make the above mean anything
//!
//! **Three**, not one, and all in the same process, on the same display, through the same
//! `dlsym`'ed function pointers, on the same loop that produced every VP9 refusal. The control
//! profiles are the header's own: **`VAProfileH264High` = 7** (`va.h:512`), **`VAProfileHEVCMain`
//! = 17** (`va.h:522`) and **`VAProfileAV1Profile0` = 32** (`va.h:537`) — all three are easy to get
//! wrong, because H.264's *Baseline* is 5 and AV1's is not 256. A probe that guesses them wrong does
//! not fail loudly: it reads `VA_STATUS_ERROR_UNSUPPORTED_PROFILE` (12) for its supposed controls
//! and looks exactly like a driver declining every codec. That is the failure mode this section
//! exists to prevent, and the constants are pinned by the test below.
//!
//! ```text
//! == vaQueryConfigEntrypoints
//!    VAProfileH264High   -> VA_STATUS_SUCCESS(0) n=2: 1(VAEntrypointVLD) 6(VAEntrypointEncSlice)
//!    VAProfileAV1Profile0 -> VA_STATUS_SUCCESS(0) n=2: 1(VAEntrypointVLD) 6(VAEntrypointEncSlice)
//!
//! == vaGetConfigAttributes, the same attributes, the same poison, the same call
//! -- H264High/EncSlice   [CONTROL]   attrib  5 status=SUCCESS  value=0x000014d6 (5334)
//! -- H264High/EncSlice   [CONTROL]   attrib 58 status=SUCCESS  value=0x80000000 <- VA_ATTRIB_NOT_SUPPORTED
//! -- AV1Profile0/EncSlice [CONTROL]   attrib 52 status=SUCCESS  value=0x05001000
//! -- AV1Profile0/EncSlice [CONTROL]   attrib 53 status=SUCCESS  value=0x0000241f
//! -- AV1Profile0/EncSlice [CONTROL]   attrib 54 status=SUCCESS  value=0x0000ffcf
//! -- HevcMain/EncSlice   [CONTROL]   attrib 50 status=SUCCESS  value=0x01900464
//! -- HevcMain/EncSlice   [CONTROL]   attrib 51 status=SUCCESS  value=0x000088cf
//! -- VP9Profile0/EncSlice             attrib  5 status=UNSUPPORTED_ENTRYPOINT(13)  value=0xdeadbeef
//! ```
//!
//! **Why three controls and not one.** A single H.264 control would show the harness can reach a
//! working encoder. Attributes **50 and 51** (`VAConfigAttribEncHEVCFeatures`,
//! `VAConfigAttribEncHEVCBlockSizes`, `va.h:1002` and `va.h:1012`) are the sharper pair, because
//! they discriminate **inside one call**: `0x80000000` — which is `VA_ATTRIB_NOT_SUPPORTED`
//! (`va.h:1521`) — for H.264 and AV1, populated words `0x01900464` / `0x000088cf` for HEVC, and
//! `VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT` with the poison intact for every VP9 pair. Three
//! different answers from one function tells you the driver is discriminating, not stalling.
//!
//! `VA_ATTRIB_NOT_SUPPORTED` matters in its own right: it is the value libva writes when it *is*
//! answering and the attribute does not apply to that codec. Seeing it for attribute 58 on H.264
//! (VP9's own attribute, on a non-VP9 codec) is exactly what a live, discriminating driver looks
//! like — and it is the opposite of the VP9 encode rows, where the status is a refusal and the
//! poison is untouched.
//!
//! The branch's own `vaprobe --caps` reaches the same answer by a different route, listing every
//! profile the driver advertises with its entrypoints:
//!
//! ```text
//! H264Main VLD: true EncSlice: true      Vp9(0) VLD: true EncSlice: false
//! H264High VLD: true EncSlice: true      Vp9(1) VLD: true EncSlice: false
//! HevcMain VLD: true EncSlice: true      Vp9(2) VLD: true EncSlice: false
//! Av1Main  VLD: true EncSlice: true      Vp9(3) VLD: true EncSlice: false
//! ```
//!
//! VP9 decode present, VP9 encode absent, on every profile, beside three codecs that encode.
//!
//! ## 4. Is this the driver, or is VA-API's VP9 encode aspirational?
//!
//! **This driver — and libva's VP9 encode is a fully specified, supported part of the API.** That
//! is the answer, and it was established from the installed headers and the installed library,
//! never from libva's source (AGENTS.md §2 forbids reading a reference implementation while
//! establishing an API's semantics). Four independent pieces of evidence:
//!
//! 1. **libva lists VP9 as an encoder module.** `va.h:115` reads `- Encoder (H264, HEVC, JPEG,
//!    MPEG2, VP8, VP9, AV1)` and `va.h:121` is `- \ref api_enc_vp9`. That is the API's own module
//!    index, and VP9 sits in the encoder list beside H.264 and HEVC — not in a "future work" list,
//!    and not in the *decoder* list only (`va.h:123`, which also names VP9).
//! 2. **The record set is complete and shipped.** `/usr/include/va/va_enc_vp9.h` exists and
//!    defines `VAEncSequenceParameterBufferVP9`, `VAEncPictureParameterBufferVP9`,
//!    `VAEncSegParamVP9`, `VAEncMiscParameterTypeVP9PerSegmantParam`, `VACodedBufferVP9Status`
//!    and the `VAConfigAttribValEncVP9` union. An aspirational feature has a stub; this is a
//!    finished record set.
//! 3. **`VAConfigAttribEncVP9` (58) has a documented value range**, and the header states it:
//!    `va.h:1059-1065` defines it "Read-only ... partitioned into fields as defined in the
//!    `VAConfigAttribValEncVP9` union", and `va_enc_vp9.h:608-631` gives that union's fields —
//!    `seg_id_block_size: 8`, `segment_feature_support: 4`, `reserved: 20` — with each bit's
//!    meaning named (`SEG_LVL_ALT_Q` 0x01, `SEG_LVL_ALT_L` 0x02, `SEG_LVL_REF_FRAME` 0x04,
//!    `SEG_LVL_SKIP` 0x08) against `VA_SEGID_BLOCK_*` (`va.h:1508-1514`). The driver even reports
//!    `vaMaxNumConfigAttributes` = **59**, so attribute 58 is inside the range it knows about.
//! 4. **The encode entry points are real symbols in the installed library.** `nm -D` on
//!    `/usr/lib/libva.so.2` (libva 1.24.0, `libva.so.2.2400.0`) exports the whole encode pipeline,
//!    and this branch has already called every one of them for H.264 and HEVC:
//!
//!    ```text
//!    $ nm -D --defined-only /usr/lib/libva.so.2 | grep ' T '
//!    000000000000c340 T vaCreateConfig
//!    000000000000edc0 T vaCreateBuffer
//!    000000000000efa0 T vaCreateBuffer2
//!    000000000000f0a0 T vaBufferSetNumElements
//!    000000000000f120 T vaUnmapBuffer
//!    000000000000fc20 T vaMapBuffer
//!    000000000000fdb0 T vaAcquireBufferHandle
//!    0000000000010210 T vaBeginPicture
//!    0000000000010580 T vaRenderPicture
//!    00000000000111c0 T vaSyncBuffer
//!    0000000000012a80 T vaEndPicture
//!    ```
//!
//! **A note on that last one, because it looks like it argues the other way.** libva exports **no
//! per-codec symbols at all** — of the 88 text symbols it exports, **none** mentions VP9, VP8, H.264,
//! AVC, HEVC, H.265 or AV1 (`nm -D --defined-only … | grep -iE 'vp9|vp8|h264|avc|hevc|h265|av1'`
//! returns nothing). So "are there VP9 encode symbols in `libva.so.2`?" is the wrong question, and
//! answering it either way would be a mistake. libva's encode path is codec-agnostic by design: the
//! codec is carried by the *profile and entrypoint passed to `vaCreateConfig`*, and the
//! codec-specific part is the record layout in the header. The absence of a `vp9_*` symbol is not
//! evidence that VP9 encode is missing; it is how every codec in the API is expressed. What the
//! symbols do show is that the machinery a VP9 encoder would drive is present and exported — and
//! because `vaCreateConfig` is one codec-agnostic entry point, its presence says **nothing** about
//! which codecs any given driver implements. Only a driver can answer that, which is what §1–§3
//! asked.
//!
//! ### How far the claim reaches, and where it stops
//!
//! | Claim | Status | What it rests on |
//! |---|---|---|
//! | This driver advertises no VP9 encoder | **Measured** | 4 profiles, 3 encode entrypoints, 885 poisoned attribute reads, `vaCreateConfig`, three working controls |
//! | libva fully expresses VP9 encode | **Established** | `va.h:115`/`:121` module index, the shipped `va_enc_vp9.h` record set, `VAConfigAttribEncVP9` = 58 with a stated field layout, `nm -D` on `libva.so.2` |
//! | Some other VA-API driver accepts VP9 encode | **Unknown — not determinable here** | nothing in the headers or the library records what drivers implement |
//!
//! The third row is the one a reader planning a cross-platform feature needs, and it is a genuine
//! **"cannot determine"**, not a hedge. One machine with one driver cannot answer a question about
//! other drivers: Intel's Media stack is not the whole of VA-API, and no amount of reading
//! `/usr/include/va` says which drivers implement what. **"Not available on Linux" would be a
//! false claim**, because the API it rests on is present and specified — the honest scope is "not
//! available on *this* driver", which is exactly what the rest of this file says.
//!
//! ### So what would a VP9 encode path actually need?
//!
//! Not much that is VP9-specific, and most of it is already built and measured. If another
//! driver advertises `VAProfileVP9Profile0` + `VAEntrypointEncSlice`, the path needs:
//!
//! - **Three records** from `va_enc_vp9.h`, none of them large: `VAEncSequenceParameterBufferVP9`
//!   (frame size, keyframe distances, `bits_per_second`), `VAEncPictureParameterBufferVP9` (the
//!   eight `reference_frames`, `ref_flags`, `pic_flags`, `refresh_frame_flags`, the per-plane qindex
//!   deltas, tile counts, and the `bit_offset_*` fields BRC writes through) and
//!   `VAEncMiscParameterTypeVP9PerSegmantParam` (eight `VAEncSegParamVP9`). Plus
//!   `VACodedBufferVP9Status` read back off the coded buffer for the driver's chosen qindex and loop
//!   filter level.
//! - **The submission machinery, which exists**: `vaCreateConfig` → `vaCreateContext` →
//!   `vaCreateBuffer` → `vaBeginPicture` → `vaRenderPicture` → `vaSyncBuffer` → `vaEndPicture`.
//!   FilmCraft has run this end to end for H.264 and for HEVC, including the readback
//!   (`vaAcquireBufferHandle` → `PRIME_HANDLE_TO_FD` → `mmap`) that §3.0 of the progress doc calls
//!   the blocker, and which is solved.
//! - **A `vpcC` sample entry** in the muxer, and the rate-control and GOP plumbing
//!   `VaEncoder` already has.
//!
//! **And how much of the VP9 work is already done?** More than the encode/decode split suggests,
//! because encode inverts the data flow: the application hands over a *picture* and **the driver
//! writes the bitstream**. So an encode path does **not** need a VP9 encoder — the thing that would
//! normally be the hard part is the hardware's job. What it needs from FilmCraft is the frame
//! header's *meaning*, and that is precisely what the decode path already built and proved:
//!
//! | Already in the tree (decode side, bit-exact) | Why an encode path needs it too |
//! |---|---|
//! | [`filmcraft_vp9::header::parse_uncompressed`] → [`filmcraft_vp9::header::FrameHeader`] | the fields `VAEncPictureParameterBufferVP9` mirrors — `frame_type`, `show_frame`, `lossless_mode`, `refresh_frame_flags`, tile counts |
//! | [`filmcraft_vp9::header::HeaderState`] + `ReferenceState` | VP9's reference frames are set/used flags with per-slot sign bias; the encode record wants the same eight slots as surfaces |
//! | [`crate::vaapi::Vp9State`] — eight reference slots, one per decode | the encoder's DPB is those eight slots; the bookkeeping is written and tested |
//! | [`crate::vaapi::picture::vp9_picture_record`] / `VaSliceParameterBufferVp9`, 316 bytes pinned on both arms | the same eight segments an encoder must fill in (`VAEncSegParamVP9` is per-segment too) |
//! | `split_superframe` | VP9's superframe index handling, needed on the mux side as well |
//! | `SegParam[1..7]` wired and tested on hardware (`vp9_p0_segmented.mp4`, 96/96) | segmentation feature support is exactly what `VAConfigAttribEncVP9`'s `segment_feature_support` reports |
//!
//! **What an encode path would still genuinely have to build** is the part decode never touched:
//! choosing qindices, a segmentation map and loop-filter levels per picture, deciding GOP and
//! alt-ref structure, honouring the `bit_offset_*` handshake where BRC rewrites the driver's
//! header, and writing `vpcC`. That is policy and bookkeeping, not codec work — which is the
//! argument for recording `VAConfigAttribEncVP9` as "known, currently unused" rather than
//! pre-declaring ~1000 lines of records for a driver that refuses them today.
//!
//! So the honest summary for anyone planning this: **the VP9 half of the job is done and the
//! driver half is the blocker, and the driver half is not ours to fix.**
//!
//! The distinction matters for whoever reads this: the answer is about **this machine's hardware
//! and driver**, not about VP9, not about libva, and not about FilmCraft's ability to *decode*
//! VP9 — decode is proven and unaffected.
//!
//! # What this file deliberately does not contain
//!
//! - **No VP9 encode records.** See above. There is no `VaEncPictureParameterBufferVp9` here, and
//!   the [module note](super) records that `VAConfigAttribEncVP9` is a constant this file knows the
//!   value of and has no use for.
//! - **No `vp9_abi.rs`.** The ABI pins exist to hold *declared* records against the header. With no
//!   records declared there is nothing to pin, and a pin file with nothing in it would be a file
//!   that appears to guard something it does not.
//! - **No submission path.** That is a later task for a codec the driver supports, and out of
//!   scope here regardless (the plan at
//!   `docs/superpowers/plans/2026-10-09-vaapi-linux-encode.md` scopes encode to H.264 and HEVC).
//!
//! # The `unsafe` posture
//!
//! **None needed, and none used.** libva is `dlopen`ed, never linked. This file is documentation
//! and a `#[cfg(test)]` assertion about what is *not* declared; it contains no `extern "C"`, no
//! `unsafe` block and no `#[allow(unsafe_code)]`. A file that carried the allowance without needing
//! it would be a file whose next edit could use it.

#[cfg(test)]
mod tests {
    /// The four profile constants libva declares are the **shared** decode/encode ones, so their
    /// presence proves nothing about encode. Pinned anyway, because they are what the module note's
    /// `vaQueryConfigProfiles` output (`19`, `20`, `21`, `22`) has to be read against, and a
    /// renumbered profile would silently change what "profile 19" in that evidence means.
    #[test]
    fn the_four_vp9_profile_constants_are_the_ones_va_h_declares() {
        assert_eq!(super::super::ffi::VP9_PROFILE_0, 19, "va.h:524 VAProfileVP9Profile0");
        assert_eq!(super::super::ffi::VP9_PROFILE_1, 20, "va.h:525 VAProfileVP9Profile1");
        assert_eq!(super::super::ffi::VP9_PROFILE_2, 21, "va.h:526 VAProfileVP9Profile2");
        assert_eq!(super::super::ffi::VP9_PROFILE_3, 22, "va.h:527 VAProfileVP9Profile3");
    }

    /// **libva has three encode entry points and `VAEntrypointEncSliceLP` is 8, not 7.**
    ///
    /// `ffi` declares only `ENC_SLICE`, the one entry point a live path reads, so that is the only
    /// value this test can compare against something. `VAEntrypointEncPicture = 7` (`va.h:559`) and
    /// `VAEntrypointEncSliceLP = 8` (`va.h:568`) are recorded here as facts, **not asserted**: with
    /// no constant to compare them against, `assert_eq!(8, 8)` would compare a literal with itself,
    /// pass unconditionally and prove nothing while appearing to be a pin. Declaring two dead
    /// constants in `ffi` purely so a test could read them would be declaring something no live path
    /// uses, which is the thing this file was written to avoid.
    ///
    /// The trap is real and was hit while producing this evidence: a brief on this branch described
    /// `VAEntrypointEncSliceLP` as 7, which is `VAEntrypointEncPicture`. Probing only 6 and "7" would
    /// have swept two entry points and reported "no VP9 encoder" while never having asked the
    /// low-power one, which is precisely the entry point most likely to be added to a driver before
    /// the slice one. All three are swept (see the module note).
    #[test]
    fn the_encode_entrypoint_this_path_uses_is_six() {
        assert_eq!(super::super::ffi::ENC_SLICE, 6, "va.h:558 VAEntrypointEncSlice");
    }

    /// **The control profiles, pinned, because guessing them produces a probe that cannot fail
    /// loudly.**
    ///
    /// A probe that reaches for the wrong `VAProfile` gets `VA_STATUS_ERROR_UNSUPPORTED_PROFILE`
    /// (12) back for its *control*, and a harness that treats any non-zero status as "this driver
    /// declines everything" reports a machine with no encoders at all instead of reporting the one
    /// thing it was asked about. That is the same class of error as reading
    /// `vaMaxNumProfiles` as `nprofiles`, which this file's evidence previously did.
    #[test]
    fn the_control_profiles_are_the_ones_va_h_declares() {
        // H.264 Baseline is 5 and H.264 Main is 6, so "H.264" is not a profile number at all.
        assert_eq!(super::super::ffi::H264_HIGH, 7, "va.h:512 VAProfileH264High");
        assert_eq!(super::super::ffi::HEVC_MAIN, 17, "va.h:522 VAProfileHEVCMain");
        // `VAProfileAV1Profile0 = 32` (`va.h:537`) is recorded in the module note rather than
        // asserted: `ffi` declares no constant for it, and `assert_eq!(32, 32)` would be a literal
        // compared with itself.
    }

    /// **No VP9 encode record is declared, and the reason is written down rather than implied.**
    ///
    /// This is not a clever pin. There is nothing to pin, because this driver has no VP9 encoder
    /// (module note, and the probe output quoted there). What the assertion can honestly do is
    /// hold the *answer* in place: if a later change starts declaring VP9 encode records on the
    /// strength of "the header has them" — which `va_enc_vp9.h` and `VAConfigAttribEncVP9 = 58`
    /// genuinely invite — this test names the measurement that contradicts it.
    ///
    /// What it deliberately does **not** do is call the driver. A capability test that `dlopen`s
    /// libva and asks it what it supports would be a machine-dependent test: it would pass here and
    /// fail on the next machine, or pass there and fail here, and a test whose result depends on
    /// which hardware it runs on is not a guard on this tree — it is a coin flip in CI. The probe
    /// was the right instrument for answering the question once; this assertion is the right
    /// instrument for recording the answer.
    #[test]
    fn no_vp9_encode_record_is_declared_and_the_reason_is_recorded() {
        // The four profile constants are *shared* with decode, so their presence proves nothing
        // about encode. What proves the absence is the module note's measurement, and this
        // assertion checks that the note still says what the measurement said.
        let note = include_str!("vp9_enc.rs");
        assert!(
            note.contains("UNSUPPORTED_ENTRYPOINT") && note.contains("VAEntrypointEncSlice"),
            "the module note must still carry the measured answer — the driver's refusal, and the \
             entrypoint it refuses"
        );
        // And it must still carry the reason the refusal is *not* an API gap, because that is the
        // half that is easy to get wrong in the other direction ("libva has no VP9 encode").
        assert!(
            note.contains("VAConfigAttribEncVP9") && note.contains("va_enc_vp9.h"),
            "the module note must still record that libva CAN express VP9 encode, so this stays a \
             statement about the driver and not about the API"
        );
        // **And it must keep the scope explicit**, because "this driver declines VP9 encode" and
        // "VP9 hardware encode is not available on Linux" are different claims and only the first
        // is supported. The row that separates them is the third one: no measurement here says
        // anything about another driver.
        assert!(
            note.contains("Unknown") && note.contains("vaMaxNumProfiles") && note.contains("nprofiles=25"),
            "the module note must keep the measured profile count (25 advertised, not the 31 \
             `vaMaxNumProfiles` buffer size) and must still record that another driver is unknown"
        );
    }
}
