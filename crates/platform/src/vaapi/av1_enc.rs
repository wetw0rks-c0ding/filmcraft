//! AV1 **encode** records: what to hand a VA-API driver so it encodes an AV1 picture.
//!
//! # Why this file exists, and what it does not contain
//!
//! libva declares six AV1 encode records in `va_enc_av1.h` and this is where they are declared.
//! **Nothing submits them yet.** Declaring a record is this change; building a submission from it
//! is a later one. The H.264 encode records in [`super::ffi`] are the reference for **shape and
//! conventions**, not for values.
//!
//! So what is deliberately absent is the point: there is no `encode()` here, no rate control, no
//! settings wiring and no `vaCreateSurface`/`vaCreateBuffer` call.
//!
//! # This driver advertises AV1 encode, and that is measured, not assumed
//!
//! A `dlopen` probe of `libva.so.2` + `libva-drm.so.2` against `/dev/dri/renderD128`, calling
//! `vaQueryConfigProfiles` and `vaQueryConfigEntrypoints` directly, reports:
//!
//! ```text
//! VAProfileAV1Profile0   -> n=2: 1(VAEntrypointVLD) 6(VAEntrypointEncSlice)
//! VAProfileAV1Profile1   -> absent from vaQueryConfigProfiles; EncSlice -> VA_STATUS_ERROR_UNSUPPORTED_PROFILE
//! VAProfileAV1Profile2   -> absent from vaQueryConfigProfiles; EncSlice -> VA_STATUS_ERROR_UNSUPPORTED_PROFILE
//! ```
//!
//! and `vaGetConfigAttributes(VAProfileAV1Profile0, VAEntrypointEncSlice, ...)` answers
//! `VAConfigAttribEncAV1` (52) `0x05001000`, `VAConfigAttribEncAV1Ext1` (53) `0x0000241f` and
//! `VAConfigAttribEncAV1Ext2` (54) `0x0000ffcf` — three populated words, against the H.264
//! control's `VA_ATTRIB_NOT_SUPPORTED` for the same three. **The driver has an AV1 encoder, and
//! it is AV1 Main (`seq_profile` 0) only.**
//!
//! # Why there is no `vp9_enc.rs` beside this one
//!
//! The same probe reports, for **all four** VP9 profiles
//! (`VAProfileVP9Profile0`..`Profile3`, 19-22):
//!
//! ```text
//! VAProfileVP9Profile0   -> n=1: 1(VAEntrypointVLD)
//! VAProfileVP9Profile1   -> n=1: 1(VAEntrypointVLD)
//! VAProfileVP9Profile2   -> n=1: 1(VAEntrypointVLD)
//! VAProfileVP9Profile3   -> n=1: 1(VAEntrypointVLD)
//! ```
//!
//! — **decode only**, no `VAEntrypointEncSlice` — and `vaGetConfigAttributes` on every
//! (VP9 profile, `EncSlice`/`EncSliceLP`) pair returns `VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT`.
//! `va_enc_vp9.h` and `VAConfigAttribEncVP9` (58) **do** exist in the installed headers, so this
//! is the driver declining a codec the API can express, not an API that lacks it. Declaring VP9
//! encode records here would be records for a codec this machine's hardware refuses: work thrown
//! away, and a plan built on an assumption the driver has already refused. **VP9 encode is not
//! available on this driver.** That is a statement about *this machine*, not about VP9 or about
//! libva, and it is why the answer lives in this note rather than in a record nobody can submit.
//!
//! # What is here, and what is measured
//!
//! Six records, every one measured with a C compiler against `/usr/include/va/va_enc_av1.h`
//! rather than transcribed from it:
//!
//! | record | measured `sizeof` | holes |
//! |---|---|---|
//! | [`VaEncSequenceParameterBufferAv1`] | 88 | 1 |
//! | [`VaEncSegParamAv1`] | 156 | 1 |
//! | [`VaEncWarpedMotionParamsAv1`] | 56 | 1 |
//! | [`VaRefFrameCtrlAv1`] | 4 | 0 |
//! | [`VaEncPictureParameterBufferAv1`] | 1032 | 1 |
//! | [`VaEncTileGroupBufferAv1`] | 20 | 1 |
//!
//! **Five holes, all named `padding_*` fields of the C-measured width**, one in every record that
//! has one:
//!
//! | record | bytes | between |
//! |---|---|---|
//! | `VAEncSequenceParameterBufferAV1` | 21-23 | `order_hint_bits_minus_1` and `va_reserved` |
//! | `VAEncSegParamAV1` | 138-139 | `feature_mask` and `va_reserved` |
//! | `VAEncWarpedMotionParamsAV1` | 37-39 | `invalid` and `va_reserved` |
//! | `VAEncPictureParameterBufferAV1` | 538-539 | `loop_restoration_flags` and `wm` |
//! | `VAEncTileGroupBufferAV1` | 2-3 | `tg_end` and `va_reserved` |
//!
//! `sizeof` is blind to a hole — padding changes no size and moves no named member — which is how
//! `VASegmentParameterVP9` agreed with a C compiler on every size and every offset for five commits
//! and handed the driver two uninitialised bytes per record. Each hole here is a declared field
//! and [`av1_abi`] asks the C compiler for its width **as a gap** (the difference between two
//! `offsetof`s minus the earlier member's own width), because C gives padding no name and no
//! `sizeof` can see one. **`VARefFrameCtrlAv1` has none**, and that is asserted as a C-measured gap
//! of zero rather than left unchecked.
//!
//! **Nine unions across these records and three attribute unions — all 100 bit positions measured
//! by running C**, setting one member to all ones and reading `__builtin_ctzll` /
//! `__builtin_clzll` back out of the union's `value`. The word width is carried per row, because
//! these records mix `u8`, `u16` and `u32` unions and reading an 8-bit union as a 32-bit one
//! reports a high bit of 31. `VARefFrameCtrlAV1` names its inner struct **`fields`**, where all
//! eight others name theirs **`bits`**; the table carries the inner name per row rather than
//! assuming it.
//!
//! # The `unsafe` posture
//!
//! **This file needs none and has none.** libva is LGPL and is never linked — its symbols are
//! resolved at run time out of the system's own library, which is what ADR 0001 rests on
//! ([docs/adr/0001-platform-ffi.md]). Declaring a record needs no symbol and no `extern "C"` call:
//! it is `#[repr(C)]` data. There is not a single `unsafe` block here, so there is no
//! `#[allow(unsafe_code)]` here either — a file that carried the allowance without needing it would
//! be a file whose next edit could use it.
//!
//! # Open questions this file records rather than answers
//!
//! Four things a submission will need and the records do not settle, checked and recorded instead
//! of guessed:
//!
//! 1. **The `wmmat` scale is not stated anywhere in the header.** `VAEncWarpedMotionParamsAV1`
//!    (`va_enc_av1.h:463-468`) says only "`wm.wmmat[]` corresponds to `gm_params[][]` in spec",
//!    and then points at *the AV1 specification section 5.9.24* and at libaom's source
//!    (`aomedia.googlesource.com`, `av1/decoder/decodeframe.c`). **Both pointers are outside what
//!    this tree may read**: §2 forbids reading a third-party decoder's source while implementing a
//!    format, so the C comment's second half is not a resource here. The AV1 *specification* is a
//!    permitted source (it is the published spec, like ITU-T for H.264), and section 5.9.24 is
//!    where the 6-bit-MV-fixed-point scale is defined — but that section is **not vendored into
//!    this repo**, so the scale is an open question for the submission task, not a fact to assert
//!    here. No `WMMAT_SCALE_*` constant is declared, because inventing one is the exact failure
//!    mode this file exists to avoid.
//! 2. **Per-tile slice records.** AV1's picture record has `tile_cols`, `tile_rows`,
//!    `num_tile_groups_minus1`, `width_in_sbs_minus_1[63]`, `height_in_sbs_minus_1[63]` and a
//!    `context_update_tile_id` (0..127), and a *separate* [`VaEncTileGroupBufferAv1`] record with
//!    a `tg_start`/`tg_end` range in 0..127. The probe answers
//!    `VAConfigAttribEncTileSupport` (35) with **1** for `VAProfileAV1Profile0` and
//!    `VAConfigAttribEncMaxTileRows`/`EncMaxTileCols` (56/57) with `VA_ATTRIB_NOT_SUPPORTED` — so
//!    tiles are on but their maximum count is not answered. Whether one `VAEncTileGroupBufferAV1`
//!    is submitted **per tile group** is not stated in the header; it is implied by the range in
//!    `tg_start`/`tg_end` and is a submission-time question.
//! 3. ~~**Which settings are `VA_FEATURE_REQUIRED`.**~~ **Answered: none, and the question this
//!    file used to carry open was itself wrong.** A `dlopen` probe of `vaGetConfigAttributes`
//!    reads `VAConfigAttribEncAV1` = `0x05001000` again (`vaGetConfigAttributes` returns
//!    `VA_STATUS_SUCCESS`), and decoding that word through the header's own union — measured by
//!    running C, see [`av1_abi::the_driver_s_av1_feature_word_has_no_required_field`] — gives
//!
//!    ```text
//!    support_palette_mode            = 1  VA_FEATURE_SUPPORTED
//!    support_allow_intrabc           = 1  VA_FEATURE_SUPPORTED
//!    support_cdef_channel_strength   = 1  VA_FEATURE_SUPPORTED
//!    (the other eleven feature fields) = 0  VA_FEATURE_NOT_SUPPORTED
//!    reserved                        = 0
//!    ```
//!
//!    **No 2-bit field in the word is `VA_FEATURE_REQUIRED`.** In particular
//!    `support_cdef_channel_strength` is **1**, not 3: CDEF is *supported*, so
//!    `enable_cdef` in the sequence record is a **choice, not an obligation**. An earlier
//!    revision of this note read that field as 3 and called it `VA_FEATURE_REQUIRED`; that was
//!    wrong twice over, because `va.h:398` defines `VA_FEATURE_REQUIRED` as **2** and **3 is no
//!    `VA_FEATURE_*` value at all**. The word is also *not* ambiguous: `0x05001000` is
//!    `0b0000_0101_0000_0000_0001_0000_0000_0000`, three single bits — bit 12 (`palette_mode`),
//!    bit 24 (`allow_intrabc`) and bit 26 (`cdef_channel_strength`) — each the low half of its
//!    2-bit field, so each reads exactly 1.
//!
//!    The contrast with HEVC is the evidence that this is a real distinction and not a parsing
//!!    accident: **the same call, same driver, same day** answers
//!    `VAConfigAttribEncHEVCFeatures` (50) = `0x01900464`, where `amp` (bits 4-5) and
//!    `cu_qp_delta` (bits 22-23) both read **2** — genuinely `VA_FEATURE_REQUIRED`, and
//!    genuinely binding on a submission. A decoder of these words that cannot tell 2 from 1 will
//!    get HEVC right by luck and AV1 wrong on purpose, which is why [`features::level`] and
//!    [`feature::REQUIRED`] exist here rather than a comparison written at each use site.
//! 4. **There is no `VAQMatrixBufferAV1`.** `qmatrix_flags` in the picture record is a *set of
//!    indexes into a quantization matrix buffer* (`qm_y`/`qm_u`/`qm_v`, 4 bits each, measured at
//!    bits 1-4, 5-8 and 9-12 of a 16-bit word) and `using_qmatrix` gates them — but libva 1.24.0
//!    declares **no AV1 QMatrix record**, exactly as it declares no `VAQMatrixBufferHEVC` sibling
//!    in a place AV1 could borrow. So `using_qmatrix` cannot honestly be set to 1 on this header,
//!    and the picture record's `qmatrix_flags` is declared as its `value` member so a later
//!    submission has somewhere to put it. `grep -rn "QMatrix" /usr/include/va/` returns H.264's
//!    and HEVC's records only.
//!
//! `VAEncSegMapBufferAV1` (`va_enc_av1.h:431`) is **not declared**: it holds a
//! `uint8_t *pSegmentMap`, which is an *application-side* description of a buffer the app maps, not
//! a record handed to the driver — and a raw pointer in a `#[repr(C)]` declaration would need
//! `unsafe` in a file whose whole point is that it has none.

// Nothing here has a caller until the AV1 submission path lands (the module note above), so the
// dead-code lint has nothing useful to say about it — except inside the tests below, which name
// every item. Kept here rather than on the `mod av1_enc;` line so this file is the whole change.
#![cfg_attr(not(test), allow(dead_code))]

use super::ffi::VA_PADDING_LOW;

/// `VABufferID` / `VASurfaceID` (`va.h:2038`, `:1645` over `typedef uint32_t VAGenericID`),
/// written as the `u32` they are.
///
/// [`super::ffi`] keeps its aliases to itself — they are private to it, and making them public
/// would mean editing the file the H.264 rate-control change is editing — so the width is spelled
/// here and **pinned against the header's own typedefs** in [`av1_abi`]
/// (`_Static_assert(sizeof(VABufferID) == 4)`, same for `VASurfaceID`), so it cannot drift into
/// another width by being written down twice.
type BufferId = u32;

/// `VASurfaceID` (`va.h:1645`) — the same width as [`BufferId`], pinned alongside it.
type SurfaceId = u32;

/// `VA_AV1_MAX_SEGMENTS` (`va_enc_av1.h:336`): the segments the segment record's two arrays are
/// sized for.
pub const MAX_SEGMENTS: usize = 8;

/// `VA_AV1_SEG_LVL_MAX` (`va_enc_av1.h:337`): the segment levels each segment's `feature_data`
/// row is sized for.
pub const SEG_LVL_MAX: usize = 8;

/// `VAEncSequenceParameterBufferAV1` (`va_enc_av1.h:238`): what to encode — the sequence header's
/// own syntax elements, the periods, the bitrate and the feature flags. **88 bytes**, alignof 4.
///
/// ```text
/// $ cc -w -o av1enc_measure av1enc_measure.c && ./av1enc_measure
/// == VAEncSequenceParameterBufferAV1 sizeof=88 alignof=4
///   GAP-BEFORE seq.va_reserved = 3
///   TRAILINGGAP seq.va_reserved = 0
/// ```
///
/// **One hole, C-measured:** bytes 21-23, between `order_hint_bits_minus_1` and `va_reserved`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaEncSequenceParameterBufferAv1 {
    /// The AV1 profile, 0..=2 (`va_enc_av1.h:239-242`). **Only 0 is available on this driver**:
    /// the probe shows `VAProfileAV1Profile1` and `Profile2` are not advertised at all.
    pub seq_profile: u8,
    /// The level of the current operating point, 0..=23 (`va_enc_av1.h:244-247`).
    pub seq_level_idx: u8,
    /// The tier, 0..=1 (`va_enc_av1.h:249-252`).
    pub seq_tier: u8,
    /// Dyadic hierarchical GOP structure, 0..=1 (`va_enc_av1.h:254-257`).
    pub hierarchical_flag: u8,
    /// Period between intra-only frames.
    pub intra_period: u32,
    /// Period between I/P frames — the anchor frame distance under a hierarchical structure.
    pub ip_period: u32,
    /// The initial bitrate in CBR, or the initial **maximum** bitrate in VBR
    /// (`va_enc_av1.h:270-280`). Rate control is a later task; nothing reads this yet.
    pub bits_per_second: u32,
    /// `seq_fields`: the sequence header's flags, packed into the union's `value` member.
    /// **Bit positions are measured, not inferred** — see the module note and [`av1_abi`]:
    /// `still_picture` is bit 0, `bit_depth_minus8` is bits 14-16 and `reserved_bits` is 20-31.
    pub seq_fields: u32,
    /// `order_hint_bits_minus_1`, 0..=7 (`va_enc_av1.h:327-330`).
    pub order_hint_bits_minus_1: u8,
    /// The C layout's hole at bytes 21-23, before `va_reserved`.
    pub padding_after_order_hint_bits_minus_1: [u8; 3],
    /// The header's own `va_reserved[VA_PADDING_HIGH]` — **sixteen** words, the full padding
    /// width with no subtraction (`va_enc_av1.h:333`). The record is 88 bytes and this ends at 88,
    /// so there is no trailing hole either.
    pub va_reserved: [u32; 16],
}

/// `VAEncSegParamAV1` (`va_enc_av1.h:342`): the segmentation parameters, embedded in the picture
/// record. **156 bytes**, alignof 4.
///
/// ```text
/// == VAEncSegParamAV1 sizeof=156 alignof=4
///   GAP-BEFORE VAEncSegParamAV1.va_reserved = 2
///   TRAILINGGAP VAEncSegParamAV1.va_reserved = 0
/// ```
///
/// **One hole, C-measured:** bytes 138-139, between `feature_mask` and `va_reserved`.
///
/// **A record that is embedded**, not submitted on its own: the picture record holds one of these
/// at offset 104.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaEncSegParamAv1 {
    /// `seg_flags`: the segmentation flags, packed into the union's `value` member — **a `u8`
    /// here, where the sequence record's union of the same shape is a `u32`**. Measured:
    /// `segmentation_enabled` is bit 0 and `reserved` is bits 3-7.
    pub seg_flags: u8,
    /// How many segments are conveyed, 0..=8 (`va_enc_av1.h:396`). **0 is the honest value for an
    /// encode that does no segmentation**, and it has a distinct meaning: 0 makes the driver
    /// choose both the segment count and the map, and forbids the app from sending a segment map
    /// buffer at all (`va_enc_av1.h:380-394`).
    pub segment_number: u8,
    /// The per-segment feature data — the specification's `FeatureData[][]`, which is the value
    /// *after* the spec's `clip3()` (`va_enc_av1.h:398-406`). Signed, because a feature delta is.
    pub feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
    /// One bit per feature id, per segment (`va_enc_av1.h:408-411`).
    pub feature_mask: [u8; MAX_SEGMENTS],
    /// The C layout's hole at bytes 138-139, before `va_reserved`.
    pub padding_after_feature_mask: [u8; 2],
    /// The header's own `va_reserved[VA_PADDING_LOW]` — four words (`va_enc_av1.h:414`).
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAEncWarpedMotionParamsAV1` (`va_enc_av1.h:458`): one reference frame's global motion.
/// **56 bytes**, alignof 4.
///
/// ```text
/// == VAEncWarpedMotionParamsAV1 sizeof=56 alignof=4
///   GAP-BEFORE VAEncWarpedMotionParamsAV1.va_reserved = 3
///   TRAILINGGAP VAEncWarpedMotionParamsAV1.va_reserved = 0
/// ```
///
/// **One hole, C-measured:** bytes 37-39, between `invalid` and `va_reserved`.
///
/// **The `wmmat` scale is not stated by the header** — see the module note's first open question.
/// This record is the one the brief's warning about "a `wmmat` scale is the kind of thing that bites
/// late" is about: the eight numbers are there, their scale is not, and no constant is declared
/// for it here.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaEncWarpedMotionParamsAv1 {
    /// Which transform: 0 identity, 1 translation, 2 rotzoom, 3 affine
    /// (`VAEncTransformationTypeAV1`, `va_enc_av1.h:445-456`). Written as the `u32` a C enum is.
    pub wmtype: u32,
    /// The warp parameters, the specification's `gm_params[][]` (`va_enc_av1.h:463-468`). **The
    /// header does not state the scale these are in** and the module note records why that is an
    /// open question rather than a constant.
    pub wmmat: [i32; 8],
    /// Whether this transform is valid, for the affine set (`va_enc_av1.h:470-471`).
    pub invalid: u8,
    /// The C layout's hole at bytes 37-39, before `va_reserved`.
    pub padding_after_invalid: [u8; 3],
    /// The header's own `va_reserved[VA_PADDING_LOW]` (`va_enc_av1.h:473-474`).
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VARefFrameCtrlAV1` (`va_enc_av1.h:488`): which reference frames to prefer, and in what order.
/// **4 bytes**, alignof 4.
///
/// ```text
/// == VARefFrameCtrlAV1 sizeof=4 alignof=4
///   TRAILINGGAP VARefFrameCtrlAV1.value = 0
/// ```
///
/// **No holes at all, and that is a measurement rather than an absence of one** — this is the only
/// AV1 encode record that is a bare union with no other member, so there is nothing *in* it that
/// could leave a hole, and [`av1_abi`] asserts a C-measured gap of zero rather than leaving it
/// unchecked.
///
/// **The union names its inner struct `fields`, not `bits`** (`va_enc_av1.h:489`, `:535`) — the one
/// exception among the nine unions in these records, and the reason the bitfield table carries the
/// inner member's name per row.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaRefFrameCtrlAv1 {
    /// `fields`: seven 3-bit search indexes and an 11-bit reserved, packed into the union's
    /// `value` member. **Measured, not inferred** — see [`av1_abi`]: `search_idx0` is bits 0-2,
    /// `search_idx6` is bits 18-20 and `Reserved` is bits 21-31. Each index is 0 (invalid) or
    /// 1..=7, naming an entry of the picture record's `ref_frame_idx` (`va_enc_av1.h:495-531`).
    pub value: u32,
}

/// `VAEncPictureParameterBufferAV1` (`va_enc_av1.h:545`): one picture's identity, its surfaces,
/// its reference frames, the quantiser, the loop filters, the tiles and the offsets into the
/// packed frame header. **1032 bytes**, alignof 4.
///
/// ```text
/// == VAEncPictureParameterBufferAV1 sizeof=1032 alignof=4
///   GAP-BEFORE pic.wm = 2
///   TRAILINGGAP pic.va_reserved = 0
/// ```
///
/// **One hole, C-measured:** bytes 538-539, between `loop_restoration_flags` and `wm`.
///
/// This is the largest encode record on this backend by a wide margin — HEVC's picture record is
/// 576 bytes and this is 1032 — and **there is no `VAPictureAV1`**: the AV1 picture record refers
/// to surfaces by bare `VASurfaceID` in `reference_frames`, where HEVC's embeds fifteen 28-byte
/// `VAPictureHEVC`s twice. The same concept, a materially different record.
///
/// **`Default` is hand-written, not derived**, for the one reason the AV1 *decode* picture record
/// already needs one: `width_in_sbs_minus_1` and `height_in_sbs_minus_1` are `[u16; 63]`, and
/// `[u16; 63]` has no `Default` — the blanket impl stops well below 63 entries. A derive therefore
/// does not compile here, and every other record in this file is small enough to keep its derive so
/// that a future field is added by the compiler noticing rather than by a human remembering.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VaEncPictureParameterBufferAv1 {
    /// The raw source frame width **in pixels, minus one** (`va_enc_av1.h:550-552`).
    pub frame_width_minus_1: u16,
    /// The height, same encoding.
    pub frame_height_minus_1: u16,
    /// The surface to store the *reconstructed* frame in. "Not used for enc only case"
    /// (`va_enc_av1.h:555-556`) — and `picture_flags.disable_frame_recon` says whether the driver
    /// produces one at all, which is how an encode that is never referred avoids the bandwidth.
    pub reconstructed_frame: SurfaceId,
    /// The buffer the compressed picture is written into.
    pub coded_buf: BufferId,
    /// The eight reference surfaces (`va_enc_av1.h:561-572`). **An entry that is not a live
    /// reference must be 0xFF**, and "other values are not allowed" — so a cleared picture must
    /// fill this with `0xFF`, *not* leave it zero, which would name surface 0. That is the same
    /// 0xFF-means-nothing rule HEVC's `collocated_ref_pic_index` has, on a different field.
    pub reference_frames: [SurfaceId; 8],
    /// The reference index list, 0..=7 (`va_enc_av1.h:574-587`): LAST, LAST2, LAST3, GOLDEN,
    /// BWDREF, ALTREF, ALTREF2.
    pub ref_frame_idx: [u8; 7],
    /// Hierarchical level **plus one** (`va_enc_av1.h:589-593`); 0 means no hierarchical level.
    pub hierarchical_level_plus1: u8,
    /// Index into `reference_frames` whose segment id map and context table are inherited.
    pub primary_ref_frame: u8,
    /// The frame's order hint (`va_enc_av1.h:603-604`).
    pub order_hint: u8,
    /// Which refresh flags apply to this frame (`va_enc_av1.h:606-607`).
    pub refresh_frame_flags: u8,
    /// The header's own reserved byte (`va_enc_av1.h:609-610`).
    pub reserved8bits1: u8,
    /// Suggested references for list 0, as a bare [`VaRefFrameCtrlAv1`].
    pub ref_frame_ctrl_l0: VaRefFrameCtrlAv1,
    /// The same for list 1.
    pub ref_frame_ctrl_l1: VaRefFrameCtrlAv1,
    /// `picture_flags`: the picture's own flags, packed into the union's `value` member.
    /// **Measured, not inferred** — see [`av1_abi`]. The one that decides what the picture is:
    /// `frame_type` is **bits 0-1**, where 0 = key frame, 1 = inter frame, 2 = intra-only frame
    /// and 3 = switch frame (`va_enc_av1.h:620-626`).
    pub picture_flags: u32,
    /// Block size for each segment id in the segment map (`va_enc_av1.h:671-674`).
    pub seg_id_block_size: u8,
    /// Tile group count **minus one**, 0..=255 (`va_enc_av1.h:676-679`). **This, not `tile_rows`,
    /// is what says the picture has more than one tile group**, and `enable_frame_obu` is
    /// "invalid if num_tile_groups_minus1 > 0" (`va_enc_av1.h:643-647`).
    pub num_tile_groups_minus1: u8,
    /// Temporal id of the frame (`va_enc_av1.h:681-682`).
    pub temporal_id: u8,
    /// Deblock filter level for Y, 0..=63 (`va_enc_av1.h:684-687`), two entries.
    pub filter_level: [u8; 2],
    /// The same for U.
    pub filter_level_u: u8,
    /// The same for V.
    pub filter_level_v: u8,
    /// `loop_filter_flags`: sharpness and the mode/ref delta enables, packed into the union's
    /// `value` member — **a `u8` here, where the picture record's other three unions are a `u32`,
    /// a `u16` and a `u32`.** Measured: `sharpness_level` is bits 0-2 (0..=7) and `reserved` is
    /// bits 5-7.
    pub loop_filter_flags: u8,
    /// Super-resolution scale denominator, 9..=16 (`va_enc_av1.h:705-708`).
    pub superres_scale_denominator: u8,
    /// The interpolation filter, 0..=3 (`va_enc_av1.h:709-710`). The probe read
    /// `VAConfigAttribEncAV1Ext1.interpolation_filter` (bits 0-4) as `0x1f`, i.e. all five
    /// declared filters supported.
    pub interpolation_filter: u8,
    /// Loop filter reference deltas, -63..=63 (`va_enc_av1.h:712-715`).
    pub ref_deltas: [i8; 8],
    /// Loop filter mode deltas, same range (`va_enc_av1.h:717-720`).
    pub mode_deltas: [i8; 2],
    /// The base quantiser index (`va_enc_av1.h:722-723`). `VA_RC_CQP` is the rate-control mode
    /// that uses it; the BRC modes use `min_base_qindex`/`max_base_qindex` instead.
    pub base_qindex: u8,
    /// DC delta for Y.
    pub y_dc_delta_q: i8,
    /// DC delta for U.
    pub u_dc_delta_q: i8,
    /// AC delta for U.
    pub u_ac_delta_q: i8,
    /// DC delta for V.
    pub v_dc_delta_q: i8,
    /// AC delta for V.
    pub v_ac_delta_q: i8,
    /// Minimum base q index for BRC, 1..=255 (`va_enc_av1.h:730-733`).
    pub min_base_qindex: u8,
    /// Maximum base q index for BRC, 1..=255 (`va_enc_av1.h:735-738`).
    pub max_base_qindex: u8,
    /// `qmatrix_flags`: whether a quantization matrix is in use and which ones, packed into the
    /// union's `value` member — **a `u16` here**. Measured: `using_qmatrix` is bit 0, `qm_y` bits
    /// 1-4, `qm_u` bits 5-8, `qm_v` bits 9-12 and `reserved` bits 13-15.
    ///
    /// **`using_qmatrix` cannot honestly be set to 1 on this header** — libva 1.24.0 declares no AV1
    /// QMatrix record for those indexes to point into. See the module note's fourth open question.
    pub qmatrix_flags: u16,
    /// The header's own reserved half-word (`va_enc_av1.h:757-758`).
    pub reserved16bits1: u16,
    /// `mode_control_flags`: tx mode, reference mode and the delta-present flags, packed into the
    /// union's `value` member. **Measured, not inferred** — see [`av1_abi`]: `tx_mode` is bits
    /// 7-8 (0 = ONLY_4X4, 1 = TX_MODE_LARGEST, 2 = TX_MODE_SELECT, 3 = invalid) and
    /// `reference_mode` is bits 9-10 (0 = single, 1 = compound, 2 = driver chooses).
    pub mode_control_flags: u32,
    /// The segmentation parameters, **embedded** rather than submitted on their own.
    pub segments: VaEncSegParamAv1,
    /// Number of tile columns (`va_enc_av1.h:809-810`).
    pub tile_cols: u8,
    /// Number of tile rows (`va_enc_av1.h:811-812`).
    pub tile_rows: u8,
    /// The header's own reserved half-word (`va_enc_av1.h:814-815`).
    pub reserved16bits2: u16,
    /// Widths in superblocks, minus one — **63 entries** (`va_enc_av1.h:817-819`). The last column
    /// or row's size is derived, which is what "minus 1" on the last entry means.
    pub width_in_sbs_minus_1: [u16; 63],
    /// Heights in superblocks, minus one, same 63 entries.
    pub height_in_sbs_minus_1: [u16; 63],
    /// Which tile supplies the CDF update, 0..=127 (`va_enc_av1.h:821-823`).
    pub context_update_tile_id: u16,
    /// CDEF damping, **minus three** (`va_enc_av1.h:825-828`).
    pub cdef_damping_minus_3: u8,
    /// The CDEF bit depth, 0..=3 (`va_enc_av1.h:829-832`).
    pub cdef_bits: u8,
    /// CDEF strengths for Y, 0..=63, eight entries (`va_enc_av1.h:833-835`).
    pub cdef_y_strengths: [u8; 8],
    /// CDEF strengths for UV, same range and count (`va_enc_av1.h:836-838`).
    pub cdef_uv_strengths: [u8; 8],
    /// `loop_restoration_flags`: the three restoration types and the LR shifts, packed into the
    /// union's `value` member — **a `u16` here**. Measured: `yframe_restoration_type` is bits 0-1
    /// and `reserved` is bits 9-15.
    pub loop_restoration_flags: u16,
    /// The C layout's hole at bytes 538-539, before `wm`.
    pub padding_after_loop_restoration_flags: [u8; 2],
    /// Per-reference global motion, **seven** entries of a 56-byte record at offset 540
    /// (`va_enc_av1.h:868-869`).
    pub wm: [VaEncWarpedMotionParamsAv1; 7],
    /// Bit offset of `base_q_idx` in the packed frame header (`va_enc_av1.h:871-878`). **In BRC
    /// the driver writes the quantiser back through this offset**; in CQP it is set to 0 and
    /// ignored.
    pub bit_offset_qindex: u32,
    /// Bit offset of `segmentation_enabled` in the packed frame header (`va_enc_av1.h:879-885`).
    pub bit_offset_segmentation: u32,
    /// Bit offset of `loop_filter_params()` in the packed frame header (`va_enc_av1.h:886-893`).
    pub bit_offset_loopfilter_params: u32,
    /// Bit offset of `cdef_params()` in the packed frame header (`va_enc_av1.h:893-902`).
    pub bit_offset_cdef_params: u32,
    /// The size in bits of that `cdef_params()` syntax (`va_enc_av1.h:903-908`).
    pub size_in_bits_cdef_params: u32,
    /// Byte offset of `obu_size` in the frame header OBU (`va_enc_av1.h:910-921`).
    pub byte_offset_frame_hdr_obu_size: u32,
    /// The frame header OBU's size in bits (`va_enc_av1.h:923-934`) — which depends on the
    /// encoded tile sizes, so it cannot be known until the picture is encoded.
    pub size_in_bits_frame_hdr_obu: u32,
    /// `tile_group_obu_hdr_info`: the tile group OBU header's flags, packed into the union's
    /// `value` member — **a `u8` here**, the third `u8` union in this one record. Measured:
    /// `temporal_id` is bits 2-4 and `spatial_id` is bits 5-6.
    pub tile_group_obu_hdr_info: u8,
    /// Frames skipped before this one (`va_enc_av1.h:961-967`), counting only ones not counted
    /// before.
    pub number_skip_frames: u8,
    /// The header's own reserved half-word (`va_enc_av1.h:968-969`).
    pub reserved16bits3: u16,
    /// How much to shrink the frame when frames are skipped (`va_enc_av1.h:971-978`); positive
    /// reduces the size, otherwise it increases it.
    pub skip_frames_reduced_size: i32,
    /// The header's own `va_reserved[VA_PADDING_HIGH]` (`va_enc_av1.h:979-980`).
    pub va_reserved: [u32; 16],
}

/// `Default` for [`VaEncPictureParameterBufferAv1`]: every byte zero, **including the hole at bytes
/// 538-539 and every entry of the two 63-entry tile arrays**.
///
/// Written out rather than derived because `[u16; 63]` has no `Default`. The cost of a hand-written
/// impl is that it *can* forget a field, which is the failure [`av1_abi`]'s `Default` pin exists for:
/// it names this record's hole and both tile arrays explicitly, so a field added here without being
/// added there is a red test rather than a byte handed to the driver from stack memory.
impl Default for VaEncPictureParameterBufferAv1 {
    fn default() -> Self {
        Self {
            frame_width_minus_1: 0,
            frame_height_minus_1: 0,
            reconstructed_frame: 0,
            coded_buf: 0,
            reference_frames: [0; 8],
            ref_frame_idx: [0; 7],
            hierarchical_level_plus1: 0,
            primary_ref_frame: 0,
            order_hint: 0,
            refresh_frame_flags: 0,
            reserved8bits1: 0,
            ref_frame_ctrl_l0: VaRefFrameCtrlAv1::default(),
            ref_frame_ctrl_l1: VaRefFrameCtrlAv1::default(),
            picture_flags: 0,
            seg_id_block_size: 0,
            num_tile_groups_minus1: 0,
            temporal_id: 0,
            filter_level: [0; 2],
            filter_level_u: 0,
            filter_level_v: 0,
            loop_filter_flags: 0,
            superres_scale_denominator: 0,
            interpolation_filter: 0,
            ref_deltas: [0; 8],
            mode_deltas: [0; 2],
            base_qindex: 0,
            y_dc_delta_q: 0,
            u_dc_delta_q: 0,
            u_ac_delta_q: 0,
            v_dc_delta_q: 0,
            v_ac_delta_q: 0,
            min_base_qindex: 0,
            max_base_qindex: 0,
            qmatrix_flags: 0,
            reserved16bits1: 0,
            mode_control_flags: 0,
            segments: VaEncSegParamAv1::default(),
            tile_cols: 0,
            tile_rows: 0,
            reserved16bits2: 0,
            width_in_sbs_minus_1: [0; 63],
            height_in_sbs_minus_1: [0; 63],
            context_update_tile_id: 0,
            cdef_damping_minus_3: 0,
            cdef_bits: 0,
            cdef_y_strengths: [0; 8],
            cdef_uv_strengths: [0; 8],
            loop_restoration_flags: 0,
            padding_after_loop_restoration_flags: [0; 2],
            wm: [VaEncWarpedMotionParamsAv1::default(); 7],
            bit_offset_qindex: 0,
            bit_offset_segmentation: 0,
            bit_offset_loopfilter_params: 0,
            bit_offset_cdef_params: 0,
            size_in_bits_cdef_params: 0,
            byte_offset_frame_hdr_obu_size: 0,
            size_in_bits_frame_hdr_obu: 0,
            tile_group_obu_hdr_info: 0,
            number_skip_frames: 0,
            reserved16bits3: 0,
            skip_frames_reduced_size: 0,
            va_reserved: [0; 16],
        }
    }
}

/// `VAEncTileGroupBufferAV1` (`va_enc_av1.h:986`): one tile group's position in the frame.
/// **20 bytes**, alignof 4.
///
/// ```text
/// == VAEncTileGroupBufferAV1 sizeof=20 alignof=4
///   GAP-BEFORE tile.va_reserved = 2
///   TRAILINGGAP tile.va_reserved = 0
/// ```
///
/// **One hole, C-measured:** bytes 2-3, between `tg_end` and `va_reserved`.
///
/// **A per-tile-group record, and the header does not say how many are submitted** — see the
/// module note's second open question. The driver here answers `VAConfigAttribEncTileSupport` with
/// 1, so tiles are on.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaEncTileGroupBufferAv1 {
    /// The first tile of this group, in raster scan order across the frame, 0..=127.
    pub tg_start: u8,
    /// The last tile of this group, same numbering.
    pub tg_end: u8,
    /// The C layout's hole at bytes 2-3, before `va_reserved`.
    pub padding_after_tg_end: [u8; 2],
    /// The header's own `va_reserved[VA_PADDING_LOW]` (`va_enc_av1.h:1000-1001`).
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAConfigAttribEncAV1` (`va.h:1020`): the AV1 encoder's feature set, read through
/// `vaGetConfigAttributes` and answered with an [`Av1Features`] word.
///
/// This driver answers it `0x05001000`, and the AV1-specific attributes are the clearest
/// discrimination between the AV1 and the H.264 profile: H.264 answers all three of 52/53/54 with
/// `VA_ATTRIB_NOT_SUPPORTED`.
pub const CONFIG_ATTRIB_AV1_FEATURES: u32 = 52;

/// `VAConfigAttribEncAV1Ext1` (`va.h:1028`): the interpolation filters, the minimum segment-id
/// block size accepted and which segment features are supported, in an [`Av1Ext1`] word.
pub const CONFIG_ATTRIB_AV1_EXT1: u32 = 53;

/// `VAConfigAttribEncAV1Ext2` (`va.h:1036`): the tile and OBU size field widths and the tx modes,
/// in an [`Av1Ext2`] word.
pub const CONFIG_ATTRIB_AV1_EXT2: u32 = 54;

/// The [`Av1Features`] word **this driver answers** `VAConfigAttribEncAV1` (52) with, for
/// `VAProfileAV1Profile0` + `VAEntrypointEncSlice`: `0x05001000`.
///
/// Recorded as a constant because it is the single fact that decides whether a submission is
/// constrained, and because it is the word a decoder is most likely to get wrong. Read through
/// the header's own union it is **three features at `VA_FEATURE_SUPPORTED` and nothing else**:
///
/// ```text
/// 0x05001000 = 0b0000_0101_0000_0000_0001_0000_0000_0000
/// bit 12 -> support_palette_mode          = 1  VA_FEATURE_SUPPORTED
/// bit 24 -> support_allow_intrabc         = 1  VA_FEATURE_SUPPORTED
/// bit 26 -> support_cdef_channel_strength = 1  VA_FEATURE_SUPPORTED
/// reserved (bits 28-31)                   = 0
/// ```
///
/// **No field is `VA_FEATURE_REQUIRED`, so no AV1 encode feature obliges the submission to set a
/// flag or supply extra configuration.** In particular `enable_cdef` is a choice: the driver can
/// CDEF and does not have to. `av1_abi` re-derives this decode by running C against the header
/// rather than trusting the paragraph above it.
///
/// The same driver, same call, answers HEVC's `VAConfigAttribEncHEVCFeatures` (50) with
/// `0x01900464`, whose `amp` and `cu_qp_delta` fields read `VA_FEATURE_REQUIRED` — so the ability
/// to tell a `2` from a `1` is load-bearing on this machine, for a neighbouring codec.
pub const DRIVER_AV1_FEATURES: Av1Features = 0x0500_1000;

/// `VAConfigAttribEncTileSupport` (`va.h:907`): whether the encoder does tiles at all.
///
/// **This driver answers 1** for `VAProfileAV1Profile0`, so tiles are on — which makes
/// [`VaEncTileGroupBufferAv1`] a record a submission would need rather than one it might not.
pub const CONFIG_ATTRIB_TILE_SUPPORT: u32 = 35;

/// `VAConfigAttribEncMaxTileRows` / `VAConfigAttribEncMaxTileCols` (`va.h:1050`, `:1057`): how
/// many tile rows and columns the encoder accepts.
///
/// Declared as one constant with its sibling rather than two, because **this driver answers both
/// with `VA_ATTRIB_NOT_SUPPORTED`**, so an encode cannot learn its tile limit from them and must
/// assume the picture record's own 63-entry arrays are the only bound.
pub const CONFIG_ATTRIB_MAX_TILE_ROWS: u32 = 56;

/// `VAConfigAttribEncPerBlockControl` (`va.h:1043`): per-block encoder control. **This driver
/// answers `VA_ATTRIB_NOT_SUPPORTED`**, so nothing in these records may assume it.
pub const CONFIG_ATTRIB_PER_BLOCK_CONTROL: u32 = 55;

/// `VAConfigAttribEncMaxSlices` (`va.h:721`): the most slices the driver will encode in one frame.
/// **`VA_ATTRIB_NOT_SUPPORTED` for AV1 on this driver**, where H.264 answers 256 — so an AV1
/// encode cannot assume a slice limit and also cannot read one.
pub const CONFIG_ATTRIB_MAX_SLICES: u32 = 14;

/// `VAEncPackedHeaderAV1_SPS` (`va_enc_av1.h:221`): the packed-header type carrying the
/// `sequence_header_obu()` syntax element.
///
/// **It is the same enumerator as [`PACKED_HEADER_PPS`] minus one — it equals
/// `VAEncPackedHeaderSequence`, which is 1.** Recorded because splitting it into a constant that
/// looks independent of the generic packed-header type is how a submission ends up sending a PPS
/// where a sequence header was meant.
pub const PACKED_HEADER_SPS: u32 = 1;

/// `VAEncPackedHeaderAV1_PPS` (`va_enc_av1.h:229`): the packed-header type carrying the
/// `frame_header_obu()` syntax element — which is `VAEncPackedHeaderPicture`, **2**. Note the
/// header's own naming: the "PPS" carries the **frame** header, not a picture parameter set.
pub const PACKED_HEADER_PPS: u32 = 2;

/// `VAAV1EncTransformationIdentity` / `Translation` / `Rotzoom` / `Affine`
/// (`va_enc_av1.h:445-455`): which global motion transform a [`VaEncWarpedMotionParamsAv1`]
/// names, and **how many parameters that transform reads out of `wmmat`** — 0, 2, 4 and 6 of the
/// eight entries respectively. The counts are what let a submission fill only what is used; the
/// remaining entries still must be zero, which is what the `Default` pin below checks.
pub mod transformation {
    /// Identity, 0-parameter.
    pub const IDENTITY: u32 = 0;
    /// Translational motion, 2-parameter.
    pub const TRANSLATION: u32 = 1;
    /// Simplified affine: rotation and zoom only, 4-parameter.
    pub const ROTZOOM: u32 = 2;
    /// Affine, 6-parameter.
    pub const AFFINE: u32 = 3;
    /// The enumeration's end sentinel, 4 — **not a transformation**.
    pub const COUNT: u32 = 4;

    /// How many of `wmmat[8]` the transform at `kind` reads, or `None` for a value that is not a
    /// transformation at all.
    ///
    /// This is a `match` over the four declared constants rather than a table, so a value outside
    /// 0..=3 — which `wmtype` is `VA_FEATURE`-adjacent hostile input — cannot index anything.
    pub const fn parameters(kind: u32) -> Option<u32> {
        match kind {
            IDENTITY => Some(0),
            TRANSLATION => Some(2),
            ROTZOOM => Some(4),
            AFFINE => Some(6),
            _ => None,
        }
    }
}

/// `VAConfigAttribValEncAV1` (`va_enc_av1.h:59`): the 32-bit word [`CONFIG_ATTRIB_AV1_FEATURES`]
/// is answered in. Fourteen 2-bit fields and a 4-bit `reserved`; see [`features`].
pub type Av1Features = u32;

/// `VAConfigAttribValEncAV1Ext1` (`va_enc_av1.h:139`): the 32-bit word
/// [`CONFIG_ATTRIB_AV1_EXT1`] is answered in. See [`ext1`].
pub type Av1Ext1 = u32;

/// `VAConfigAttribValEncAV1Ext2` (`va_enc_av1.h:176`): the 32-bit word
/// [`CONFIG_ATTRIB_AV1_EXT2`] is answered in. See [`ext2`].
pub type Av1Ext2 = u32;

/// The three `VA_FEATURE_*` support levels a field of [`Av1Features`] holds, from `va.h:381`,
/// `:391` and `:398`.
///
/// **These are three named values, not a boolean, and the third is 2 — not 3.** `va.h:392-397`
/// is explicit about what `REQUIRED` obliges: *"any corresponding feature flag must be set and
/// any additional configuration needed by the feature must be supplied."* That is a submission
/// constraint with teeth — a stream that leaves such a flag at 0 asks the driver for something it
/// has already said it cannot do, which surfaces as a **wrong file rather than an error**.
///
/// **`3` is not a `VA_FEATURE_*` value.** An earlier revision of this file's module note read
/// `support_cdef_channel_strength` as 3 and called it `REQUIRED`; both halves were wrong, and the
/// `3` is the more dangerous half because it is not merely mislabelled but undefined — a decoder
/// that treated any non-zero level as "supported" would have made the same mistake and passed.
///
/// [`av1_abi::the_feature_levels_are_the_ones_va_h_defines`] pins each of the three against the
/// header by compiling C, so a libva that ever renumbers them fails here rather than at a
/// submission.
pub mod feature {
    // The word these levels are read out of, and the bit offsets they are read at, are siblings of
    // this module rather than parents -- so they are named here instead of assumed to be in scope.
    use super::{Av1Features, features};

    /// `VA_FEATURE_NOT_SUPPORTED` (`va.h:381`): *"Any corresponding feature flag must not be set."*
    pub const NOT_SUPPORTED: u32 = 0;
    /// `VA_FEATURE_SUPPORTED` (`va.h:391`): *"The user may decide whether or not to use this
    /// feature."* Support is the hardware's ability, not an instruction to the application.
    pub const SUPPORTED: u32 = 1;
    /// `VA_FEATURE_REQUIRED` (`va.h:398`): the feature's flag **must** be set and any additional
    /// configuration the feature needs **must** be supplied.
    pub const REQUIRED: u32 = 2;

    /// Every level `va.h` defines, so a reader can say "this word holds something else" rather
    /// than fall through a `match` to a default that silently means one of the three.
    pub const LEVELS: [u32; 3] = [NOT_SUPPORTED, SUPPORTED, REQUIRED];

    /// The level `word` holds at the 2-bit field `bit` names — [`features::mask`] applied and
    /// shifted, so the caller cannot get the mask and the shift out of step.
    ///
    /// `bit` is not range-checked against this module's own constants; a caller that passes
    /// [`features::RESERVED`] (a 4-bit field at bit 28) gets those four bits, which is the honest
    /// answer for a width this function does not know.
    pub const fn level(word: Av1Features, bit: u32) -> u32 {
        (word & features::mask(bit)) >> bit
    }

    /// Whether the driver marked the feature at `bit` **required** — the only reading of this word
    /// that constrains a submission. `SUPPORTED` is not required, and neither is `NOT_SUPPORTED`.
    pub const fn is_required(word: Av1Features, bit: u32) -> bool {
        level(word, bit) == REQUIRED
    }
}

/// The bit a field of [`Av1Features`] occupies, **measured by running C** and pinned in
/// [`av1_abi`] rather than read off the declaration order.
///
/// **The values are `VA_FEATURE_*` support levels, not booleans** — the three are named in
/// [`feature`] and measured against `va.h`, so a field is 0 (not supported), 1 (supported) or
/// **2 (required)**, and reading one as a bool is how a `VA_FEATURE_REQUIRED` feature gets treated
/// as merely optional.
pub mod features {
    // The three `VA_FEATURE_*` levels the fields here hold, named once in [`super::feature`] and
    // read once here, so a level is never spelled as a bare number at a use site. `Av1Features` is
    // the word those levels are read out of, and it is a sibling too.
    use super::feature;
    use super::Av1Features;

    /// 2 bits, 0-1.
    pub const SUPPORT_128X128_SUPERBLOCK: u32 = 0;
    /// 2 bits, 2-3.
    pub const SUPPORT_FILTER_INTRA: u32 = 2;
    /// 2 bits, 4-5.
    pub const SUPPORT_INTRA_EDGE_FILTER: u32 = 4;
    /// 2 bits, 6-7.
    pub const SUPPORT_INTERINTRA_COMPOUND: u32 = 6;
    /// 2 bits, 8-9.
    pub const SUPPORT_MASKED_COMPOUND: u32 = 8;
    /// 2 bits, 10-11.
    pub const SUPPORT_WARPED_MOTION: u32 = 10;
    /// 2 bits, 12-13.
    pub const SUPPORT_PALETTE_MODE: u32 = 12;
    /// 2 bits, 14-15.
    pub const SUPPORT_DUAL_FILTER: u32 = 14;
    /// 2 bits, 16-17.
    pub const SUPPORT_JNT_COMP: u32 = 16;
    /// 2 bits, 18-19.
    pub const SUPPORT_REF_FRAME_MVS: u32 = 18;
    /// 2 bits, 20-21.
    pub const SUPPORT_SUPERRES: u32 = 20;
    /// 2 bits, 22-23.
    pub const SUPPORT_RESTORATION: u32 = 22;
    /// 2 bits, 24-25.
    pub const SUPPORT_ALLOW_INTRABC: u32 = 24;
    /// 2 bits, 26-27.
    pub const SUPPORT_CDEF_CHANNEL_STRENGTH: u32 = 26;
    /// 4 bits, 28-31: `reserved`, "for future use, must be zero" — and a non-zero value here is
    /// how a word from a driver that packs something else here would be misread as a feature.
    pub const RESERVED: u32 = 28;

    /// The 2-bit mask a feature field occupies, from its bit offset. Every field in this union
    /// except `RESERVED` is 2 bits wide.
    pub const fn mask(bit: u32) -> u32 {
        0b11 << bit
    }

    /// The 4-bit mask `RESERVED` occupies, from its bit offset. **The one field in this union that
    /// is not two bits**, and the reason it has its own mask rather than using [`mask`].
    pub const fn reserved_mask() -> u32 {
        0xF << RESERVED
    }

    /// The `VA_FEATURE_*` level a field of `word` holds at `bit` — the same reading as
    /// [`feature::level`], spelled from this module's own mask so the two cannot drift apart.
    pub const fn level(word: Av1Features, bit: u32) -> u32 {
        feature::level(word, bit)
    }
}

/// The bit a field of [`Av1Ext1`] occupies, **measured by running C** and pinned in [`av1_abi`].
///
/// This union is **not** a set of support levels: each field is a set of independent bits, and the
/// header spells out the meaning of each one.
pub mod ext1 {
    /// 5 bits, 0-4: which interpolation filters are supported, one bit each —
    /// `0x01` eight-tap, `0x02` eight-tap-smooth, `0x04` eight-sharp, `0x08` bilinear,
    /// `0x10` switchable (`va_enc_av1.h:142-148`). **This driver answered `0x1f` — all five.**
    pub const INTERPOLATION_FILTER: u32 = 0;
    /// 8 bits, 5-12: the smallest `seg_id_block_size` the encoder accepts, as a bit set of
    /// `VA_SEGID_BLOCKXXXX`; the application must send a PPS value at least this large.
    pub const MIN_SEGID_BLOCK_SIZE_ACCEPTED: u32 = 5;
    /// 8 bits, 13-20: which segment features are supported, one bit each, in the order
    /// `SEG_LVL_ALT_Q`, `SEG_LVL_ALT_LF_Y_V`, `SEG_LVL_ALT_LF_Y_H`, `SEG_LVL_ALT_LF_U`,
    /// `SEG_LVL_ALT_LF_V`, `SEG_LVL_REF_FRAME`, `SEG_LVL_SKIP`, `SEG_LVL_GLOBALMV`.
    pub const SEGMENT_FEATURE_SUPPORT: u32 = 13;
    /// 11 bits, 21-31: reserved, must be zero.
    pub const RESERVED: u32 = 21;

    /// The 5-bit mask `INTERPOLATION_FILTER` occupies.
    pub const fn interpolation_filter_mask() -> u32 {
        0x1F << INTERPOLATION_FILTER
    }

    /// The 8-bit mask a segment-id block-size or segment-feature field occupies.
    pub const fn byte_mask(bit: u32) -> u32 {
        0xFF << bit
    }

    /// The 11-bit mask `RESERVED` occupies.
    pub const fn reserved_mask() -> u32 {
        0x7FF << RESERVED
    }
}

/// The bit a field of [`Av1Ext2`] occupies, **measured by running C** and pinned in [`av1_abi`].
pub mod ext2 {
    /// 2 bits, 0-1: the number of bytes needed to code a tile size, minus one. **A 2-bit field for
    /// a byte count**, so the maximum is 4 bytes; the picture record's
    /// `VAConfigAttribEncAV1Ext2` reader is where the driver states its own.
    pub const TILE_SIZE_BYTES_MINUS1: u32 = 0;
    /// 2 bits, 2-3: the fixed number of bytes needed to code an `obu_size` syntax, minus one.
    pub const OBU_SIZE_BYTES_MINUS1: u32 = 2;
    /// 3 bits, 4-6: which tx modes are supported — `0x01` ONLY_4X4, `0x02` TX_MODE_LARGEST,
    /// `0x04` TX_MODE_SELECT (`va_enc_av1.h:190-195`).
    pub const TX_MODE_SUPPORT: u32 = 4;
    /// 13 bits, 7-19: the most tiles the driver supports, **minus one** — so the maximum tile
    /// count is 8192, and the picture record's 63-entry `width_in_sbs_minus_1` array is the
    /// tighter of the two bounds.
    pub const MAX_TILE_NUM_MINUS1: u32 = 7;
    /// 12 bits, 20-31: reserved, must be zero.
    pub const RESERVED: u32 = 20;

    /// The 2-bit mask a size field occupies.
    pub const fn size_mask(bit: u32) -> u32 {
        0b11 << bit
    }

    /// The 3-bit mask `TX_MODE_SUPPORT` occupies.
    pub const fn tx_mode_mask() -> u32 {
        0b111 << TX_MODE_SUPPORT
    }

    /// The 13-bit mask `MAX_TILE_NUM_MINUS1` occupies.
    pub const fn max_tile_mask() -> u32 {
        0x1FFF << MAX_TILE_NUM_MINUS1
    }

    /// The 12-bit mask `RESERVED` occupies.
    pub const fn reserved_mask() -> u32 {
        0xFFF << RESERVED
    }
}

#[cfg(test)]
#[path = "av1_abi.rs"]
mod av1_abi;