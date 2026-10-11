//! The VA-API FFI layer: the symbols of libva and of its DRM helper, resolved at run time, and the
//! `VA*` structures, constants and status codes the decoder passes across.
//!
//! FFI module (docs/adr/0001-platform-ffi.md): every `unsafe` block has a `// SAFETY:` comment, the
//! public items are safe and return `Result<_, String>`, and no libva type leaves this module —
//! `VADisplay`, the `VA*ID` handles and the function pointers are implementation details of this
//! file.
//!
//! libva is never linked (it is LGPL): the system's own library is opened with `dlopen` when the
//! backend first asks for it, exactly as the Windows backend opens `mfplat.dll` and NVENC's
//! driver, so a machine without VA-API still starts and decodes in software.
//!
//! Two libraries are involved, and the split matters: everything except `vaGetDisplayDRM` is in
//! `libva.so.2`, and `vaGetDisplayDRM` — which turns a DRM file descriptor into a display — is in
//! `libva-drm.so.2`. `dlclose` runs only when the last [`Api`] is gone, which is what keeps a
//! function pointer from outliving the code it points at; [`Display`] holds an `Arc<Api>` for the
//! same reason.
//!
//! `vaInitialize` is mandatory before anything else is asked of a display (the library segfaults
//! otherwise) and its status is checked: [`Display::new`] is where that happens.
//!
//! The structures and constants are hand-declared from `/usr/include/va/` (libva 2.x) rather than
//! generated, so the tests at the bottom of this file check them twice over: the constants and the
//! layout are pinned against the header, and the layout is compared with the library that is
//! actually installed ([`Display::probe_image_layout`]) — libva has no `sizeof` / `offsetof` of its
//! own to ask, so a behavioural check is the only way to hold the declaration to the driver.

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};

use super::drm;
use std::mem::size_of;
use std::sync::{Arc, MutexGuard, OnceLock};

// ---------------------------------------------------------------------------------------------
// libva's own types, none of which leaves this module.
// ---------------------------------------------------------------------------------------------

/// `VADisplay` (`va.h:260`): an opaque, window-system dependent display handle, `void *` to us.
type VADisplay = *mut c_void;

/// `VAGenericID` (`va.h:1580`): the `unsigned int` behind every libva handle. The five handles are
/// named apart below because the C API does — a `VABufferID` in a call out is not a `VAConfigID` —
/// and the three the decode loop has not called yet are `allow(dead_code)` rather than unnamed.
type VAGenericID = u32;
#[allow(dead_code)]
type VAConfigID = VAGenericID;
#[allow(dead_code)]
type VAContextID = VAGenericID;
#[allow(dead_code)]
type VASurfaceID = VAGenericID;
type VABufferID = VAGenericID;
type VAImageID = VAGenericID;

/// `VAStatus` (`va.h:262`): what every libva entry point returns.
type VAStatus = c_int;

/// `VABufferType` (`va.h:2044`), which C declares as an `enum` and so as an `int`.
type VaBufferType = c_int;
/// The two libva enumerations the decoder asks about, which C declares as plain `int`.
type VAProfile = c_int;
type VAEntrypoint = c_int;

/// `VAConfigAttrib` (`va.h:1072`): one configuration attribute — a type and its value. Eight bytes.
/// This backend passes none: `vaCreateConfig` takes a null list with a zero count for the defaults,
/// which is what a decoder wants (the profile, the entry point and the render format are the three
/// things it does choose, and those are arguments rather than attributes).
#[repr(C)]
#[derive(Default)]
pub struct VaConfigAttrib {
    type_: VAStatus,
    value: c_uint,
}

/// `VASurfaceAttrib` (`va.h:1742`): one surface attribute — a type, the flags saying whether it can
/// be read or written, and a value whose meaning the type decides. **24 bytes** — 4 for the type,
/// 4 for the flags, then a `VAGenericValue` that is 16.
///
/// This backend passes none, either: `vaCreateSurfaces` takes a null list with a zero count, and the
/// driver allocates surfaces in the default (`VA_SURFACE_ATTRIB_MEM_TYPE_VA`) memory type, which is
/// the only type a mapped readback path can rely on. It is declared because
/// `vaCreateSurfaces` takes the array's type, so a caller that passes nothing still has to hand the
/// library a pointer to *something*. **The size still has to be the header's**, because the array's
/// stride is what a driver walks when a caller does pass one: a record declared 8 bytes short is
/// the same class of bug as a function declared with the wrong arity, and it produces a plausible
/// wrong answer rather than a failure.
#[repr(C)]
#[derive(Default)]
pub struct VaSurfaceAttrib {
    type_: VAStatus,
    flags: c_uint,
    value: VaGenericValue,
}

/// `VAGenericValue` (`va.h:1662`): what a `VASurfaceAttrib` carries. **16 bytes, and a struct, not a
/// union** — a `VAGenericValueType` discriminator followed by the value it describes:
///
/// ```c
/// typedef struct _VAGenericValue {
///     VAGenericValueType  type;
///     union { int32_t i; float f; void *p; VAGenericFunc fn; } value;
/// } VAGenericValue;
/// ```
///
/// It was declared here as the bare eight-byte union the header's *inner* member is, which made
/// every `VASurfaceAttrib` 8 bytes instead of 24. Nothing noticed, because the only array this
/// backend ever passes is a null pointer — see the note on [`VaSurfaceAttrib`].
///
/// ## The 4 bytes at offsets 4-7, which the header also leaves unnamed
///
/// `value` is a union holding a `void *` (`va.h:1666-1675`), so C aligns it to 8 and puts it at
/// offset 8 — leaving **a 4-byte hole between the 4-byte `type` word and it**. This is the same
/// defect [`VaSegmentParameterVp9`] was fixed for, in a record nobody thought to look at because
/// it has been correct by accident for as long as it existed: `offset_of!` and `sizeof` both agree
/// with the header whether the padding is named or not, so nothing that reads a number out of the
/// struct can see it. It is named here for the reason the note above this file's structures gives
/// and no other: `#[derive(Default)]` initialises named fields only, so an unnamed hole is
/// indeterminate memory and [`record_bytes`] would hand it to a driver as whatever was on the
/// stack. `[u8; 4]` is align-1, so `value` stays 8-aligned at 8 and the record stays 16 bytes.
#[repr(C)]
pub struct VaGenericValue {
    /// `VAGenericValueType` (`va.h:1648-1656`): `VAGenericValueTypeInteger`, `…Float`,
    /// `…Pointer` or `…Func`.
    pub type_: VAStatus,
    /// The C layout's hole at offsets 4-7, before `value` — **the header has one too, and a
    /// declaration that did not name it left four indeterminate bytes in a record handed to a
    /// driver.** `value` is a union containing a `void *`, so C aligns it to 8 and starts it at 8.
    pub padding_before_value: [u8; 4],
    pub value: VaGenericValueValue,
}

/// The union `VAGenericValue::value` holds (`va.h:1666-1675`): a 32-bit integer, a `float`, a
/// pointer, or a function pointer. All four are 8 bytes wide on every platform this backend builds
/// for, which is what makes the enclosing struct 16 rather than 12.
#[repr(C)]
#[derive(Clone, Copy)]
pub union VaGenericValueValue {
    pub integer: i32,
    pub float: f32,
    pub pointer: *mut c_void,
    pub function: Option<unsafe extern "C" fn()>,
}

impl Default for VaGenericValue {
    fn default() -> Self {
        // Written out rather than derived, because `Default` is what the whole hole argument rests
        // on: it must zero `padding_before_value` too, or the four bytes it covers are
        // indeterminate and `record_bytes` hands them to a driver.
        Self { type_: 0, padding_before_value: [0; 4], value: VaGenericValueValue { integer: 0 } }
    }
}

// ---------------------------------------------------------------------------------------------
// Constants, all read out of `/usr/include/va/` on the target machine (libva 2.24).
// ---------------------------------------------------------------------------------------------

/// `VA_STATUS_SUCCESS` (`va.h:264`): the status every entry point returns on success.
pub const OK: i32 = 0;

/// `VA_PADDING_LOW` / `VA_PADDING_MEDIUM` (`va.h:360-361`): the reserved-field widths libva
/// pads its structures with, which is what makes them a stable size.
pub const VA_PADDING_LOW: usize = 4;
/// See [`VA_PADDING_LOW`].
pub const VA_PADDING_MEDIUM: usize = 8;

/// `VA_RT_FORMAT_YUV420` (`va.h:1083`): 8-bit 4:2:0.
pub const YUV420: u32 = 0x0000_0001;
/// `VA_RT_FORMAT_YUV420_10` (`va.h:1088`): 10-bit 4:2:0. `0x100`, *not* a packed bit count: the
/// low byte is the chroma subsampling and the byte above it the bit depth. Easy to get wrong by
/// orders of magnitude, so it is pinned by `the_format_constants_are_the_ones_libva_uses`.
pub const YUV420_10: u32 = 0x0000_0100;

/// `VA_FOURCC_NV12` (`va.h:4425`) and `VA_LSB_FIRST` (`va.h:4714`): the image format the ABI
/// probe asks the driver for.
const FOURCC_NV12: u32 = 0x3231_564e;
const BYTE_ORDER_LSB_FIRST: u32 = 1;

/// `VA_FOURCC_NV12` (`va.h:4425`): the four-character code of an 8-bit biplanar 4:2:0 image. The
/// readback path asks for this by name (and for [`FOURCC_P010`] at 10 bits) so a driver that
/// derives something else — NV21, say, which is the same picture with U and V swapped — is
/// refused rather than read as if the components were in the order FilmCraft expects.
pub const NV12_FOURCC: u32 = FOURCC_NV12;
/// `VA_FOURCC_P010` (`va.h:4478`): the 10-bit biplanar 4:2:0 image, each sample a 16-bit word with
/// the 10 significant bits in the high half.
pub const P010_FOURCC: u32 = 0x3031_3050;

/// `VAImageFormat.bits_per_pixel` as libva reports it for a 4:2:0 image: **12** for NV12 and
/// **24** for P010.
///
/// Not "8" and "10": the field counts the bits a *luma and chroma sample together* take across the
/// two planes, which for 4:2:0 is twice the sample depth (8 + 4 + 4, or 16 + 4 + 4). Every driver
/// agrees on this — an Intel iHD driver at libva 1.24 reports exactly these two for NV12 and P010
/// — so they are the two values [`crate::vaapi::validate_image`] takes, and anything else is a
/// picture this backend does not know how to read.
pub const NV12_BPP: u32 = 12;
/// See [`NV12_BPP`].
pub const P010_BPP: u32 = 24;

/// `VA_STATUS_ERROR_OPERATION_FAILED` (`va.h:266`): the status `vaDeriveImage` is documented to
/// return when a surface cannot be represented as a `VAImage`, which is the signal to read the
/// picture the indirect way instead (`vaCreateImage` + `vaPutImage`). See [`Session::read`].
pub const OPERATION_FAILED: i32 = 0x1;
/// `VA_STATUS_ERROR_HW_BUSY` (`va.h:291`): the video engine was resetting. `vaSyncSurface` reports
/// it while the reset runs and succeeds on a later call, so [`Session::sync`] asks again rather
/// than reporting a decode failure the stream may not have had.
pub const HW_BUSY: i32 = 0x22;
/// `VA_STATUS_ERROR_TIMEDOUT` (`va.h:284`): `vaSyncBuffer` was not finished. **`va.h:4381-4382` is
/// explicit that this means "synchronization is still in progress, client should call the function
/// again to complete synchronization"** — not that the encode failed — so [`Encoder::sync`] asks
/// again rather than reporting a failure the picture may not have had.
pub const TIMED_OUT: i32 = 0x26;

/// `VASurfaceRendering` / `VASurfaceDisplaying` / `VASurfaceReady` (`va.h:4296-4299`): what
/// `vaQuerySurfaceStatus` says about a surface. `READY` is the state a surface has to be in before
/// its pixels can be read.
pub const SURFACE_READY: u32 = 0x4;

/// `VA_PROGRESSIVE` (`va.h:1920`): the only frame-pictures flag a decoder passes to
/// `vaCreateContext`. Every stream this backend takes is progressive (`plan_nal` declines the rest),
/// so this is the whole flag word.
pub const PROGRESSIVE: c_int = 0x1;

/// `VABufferType` (`va.h:2042`): the buffer kinds a decode loop submits, in the order libva
/// numbers them.
pub const PICTURE_PARAMETER_BUFFER: VAStatus = 0;
/// `VAIQMatrixBufferType` (`va.h:2043`): the H.264 scaling lists (`VAIQMatrixBufferH264`, `va.h`).
/// **A decode that does not submit one of these does not decode.** The iHD driver takes a
/// submission without it, reports success at every call, and leaves the render target holding the
/// cleared surface — a uniform mid-grey picture, which is the plausible-looking wrong answer this
/// backend must never hand back. See [`crate::vaapi::picture`]'s `h264_iq_matrix`, which builds it
/// from the sequence parameter set's own lists.
pub const IQ_MATRIX_BUFFER: VAStatus = 1;
/// See [`PICTURE_PARAMETER_BUFFER`].
pub const SLICE_PARAMETER_BUFFER: VAStatus = 4;
/// See [`PICTURE_PARAMETER_BUFFER`].
pub const SLICE_DATA_BUFFER: VAStatus = 5;

// The encode buffer types. **`VAEncCodedBufferType` is 21, not 0 or 6**: libva numbers the decode
// kinds 0..=13 and the encode kinds from 21, so the two sets do not collide and neither is a suffix
// of the other. Submitting a decode buffer type to the encoder is not "an unknown type" — it is a
// different type, which the driver rejects.
/// `VAEncCodedBufferType` (`va.h:2061`): the buffer the encoder fills with the compressed bitstream.
/// Created with `data = null`, never with the record bytes: the driver writes into it, and its
/// mapped bytes are a header followed by one [`VaEncSliceParameterBufferH264`] per submitted slice
/// followed by the payload — see [`Encoder::read_coded`], which reads that structure out.
pub const ENC_CODED_BUFFER: VAStatus = 21;
/// `VAEncSequenceParameterBufferType` (`va.h:2062`): one [`VaEncSequenceParameterBufferH264`] for
/// the whole sequence, submitted with every picture.
pub const ENC_SEQUENCE_PARAMETER_BUFFER: VAStatus = 22;
/// `VAEncPictureParameterBufferType` (`va.h:2063`): one [`VaEncPictureParameterBufferH264`] per
/// access unit.
pub const ENC_PICTURE_PARAMETER_BUFFER: VAStatus = 23;
/// `VAEncSliceParameterBufferType` (`va.h:2064`): one [`VaEncSliceParameterBufferH264`] per slice.
/// This buffer is an **array** of them: `num_elements` is the slice count.
pub const ENC_SLICE_PARAMETER_BUFFER: VAStatus = 24;
#[cfg_attr(not(test), allow(dead_code))]
/// `VAEncPackedHeaderDataBufferType` (`va.h:2066`): the header bytes themselves, when the packed
/// header mode is used. A parameter spike that asks the driver to write the parameter sets does not
/// submit one of these — see [`PACKED_HEADER_SEQUENCE`].
pub const ENC_PACKED_HEADER_DATA_BUFFER: VAStatus = 26;
/// `VAEncMiscParameterBufferType` (`va.h:2067`): a [`VAEncMiscParameterBuffer`], which is a type
/// word followed by a flexible array — so its byte length is `size_of::<VAEncMiscParameterBuffer>()`
/// plus the record it carries, and every record below names what that is.
pub const ENC_MISC_PARAMETER_BUFFER: VAStatus = 27;

/// `VA_SLICE_DATA_FLAG_ALL` (`va.h:3080`): the whole slice is in the buffer. Every slice record
/// this backend builds carries its slice whole, so this is the flag they all get; the other
/// `VA_SLICE_DATA_FLAG_*` values (`BEGIN`, `MIDDLE`, `END`) describe a slice submitted in
/// pieces, which a single-frame-at-a-time decoder never does.
pub const SLICE_DATA_FLAG_ALL: u32 = 0x0;

/// `VAPictureH264::flags` bits (`va.h:3583-3592`).
pub const PICTURE_H264_SHORT_TERM_REFERENCE: u32 = 0x0000_0008;
/// See [`PICTURE_H264_SHORT_TERM_REFERENCE`].
pub const PICTURE_H264_LONG_TERM_REFERENCE: u32 = 0x0000_0010;
/// `VA_PICTURE_H264_INVALID` (`va.h:3583`): the entry is not a picture the driver may read. It is
/// what an unused `ReferenceFrames` slot or reference list entry carries, which is a defined value
/// rather than a surface id that might name a real but wrong surface.
pub const PICTURE_H264_INVALID: u32 = 0x0000_0001;

/// `VA_PICTURE_HEVC_INVALID` (`va.h:5310`): see [`PICTURE_H264_INVALID`].
pub const PICTURE_HEVC_INVALID: u32 = 0x0000_0001;
/// `VA_PICTURE_HEVC_LONG_TERM_REFERENCE` (`va.h:5316`): the `ReferenceFrames` entry is a long-term
/// reference of the stream (`va.h:5319`).
pub const PICTURE_HEVC_LONG_TERM_REFERENCE: u32 = 0x0000_0008;
/// `VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE` (`va.h:5325`): the entry is one of the
/// `NumPocStCurrBefore` pictures the current picture's short-term reference picture set refers to.
pub const PICTURE_HEVC_RPS_ST_CURR_BEFORE: u32 = 0x0000_0010;
/// `VA_PICTURE_HEVC_RPS_ST_CURR_AFTER` (`va.h:5331`): see
/// [`PICTURE_HEVC_RPS_ST_CURR_BEFORE`], for `NumPocStCurrAfter`.
pub const PICTURE_HEVC_RPS_ST_CURR_AFTER: u32 = 0x0000_0020;
/// `VA_PICTURE_HEVC_RPS_LT_CURR` (`va.h:5337`): see [`PICTURE_HEVC_RPS_ST_CURR_BEFORE`], for
/// `NumPocLtCurr`. The three are exclusive of one another and of
/// [`PICTURE_HEVC_LONG_TERM_REFERENCE`] (`va.h:5326`), so one entry carries one of the four.
pub const PICTURE_HEVC_RPS_LT_CURR: u32 = 0x0000_0040;

/// `VA_INVALID_SURFACE` (`va.h:1648`), which is `VA_INVALID_ID`: the picture id an invalid entry
/// carries.
pub const INVALID_SURFACE: u32 = 0xffff_ffff;

// `VAConfigAttribType` (`va.h:611-628`) and the encode attribute values. An encoder cannot be
// created with a null attribute list the way a decoder's is: these three are read/write, and a
// driver that is asked for one it does not support fails `vaCreateConfig` rather than falling back
// to a default. `the_encode_constants_are_the_ones_libva_uses` pins them against the header.
///
/// `VAConfigAttribRateControl` (`va.h:625`).
pub const CONFIG_ATTRIB_RATE_CONTROL: VaBufferType = 5;
/// `VAConfigAttribEncPackedHeaders` (`va.h:707`, `va.h:1199-1231`): asks the driver to write the
/// parameter sets itself. `VA_ENC_PACKED_HEADER_SEQUENCE | VA_ENC_PACKED_HEADER_PICTURE` is the
/// combination that makes `avcC` possible at all (Task 4 of the plan): the SPS and PPS then arrive
/// inside the coded buffer with every key frame, and FilmCraft never has to write them.
pub const CONFIG_ATTRIB_ENC_PACKED_HEADERS: VaBufferType = 10;
/// `VAConfigAttribRTFormat` (`va.h:612`): the render format the encoder writes. Zero in C, which
/// is why it is worth naming: it is the one attribute whose type number is also its value in the
/// common case, and a driver that is handed the wrong value here fails in a way that reads like a
/// codec problem.
pub const CONFIG_ATTRIB_RT_FORMAT: VaBufferType = 0;
/// `VAConfigAttribEncMaxRefFrames` (`va.h:734-745`): **the read-only attribute that matters to an
/// inter picture**, and the header's spelling is `MaxRefFrames` — there is no
/// `VAConfigAttribEncMaxReferenceFrames`, and searching for that name finds nothing.
///
/// The value is the maximum number of reference frames the driver supports for this configuration,
/// and for H.264 the low 16 bits are reference list 0 and the high 16 bits reference list 1
/// (`va.h:737-739`).
pub const CONFIG_ATTRIB_ENC_MAX_REF_FRAMES: VaBufferType = 13;
/// `VAConfigAttribEncSliceStructure` (`va.h:743-758`): the slice layouts the driver accepts,
/// against `VA_ENC_SLICE_STRUCTURE_*` (`va.h:1251-1263`).
pub const CONFIG_ATTRIB_ENC_SLICE_STRUCTURE: VaBufferType = 15;
// `VA_ENC_SLICE_STRUCTURE_*` (`va.h:1251-1263`) is **not** declared here. A whole-picture slice is
// `macroblock_address = 0, num_macroblocks = picture_width_in_mbs * picture_height_in_mbs`, which is
// an arbitrary run of macroblocks rather than a power of two in rows (640x368 is 23 rows), so
// `VA_ENC_SLICE_STRUCTURE_ARBITRARY_MACROBLOCKS` (`va.h:1253`) is the bit that covers this
// encoder's only slice structure. It is recorded here rather than named because **nothing checks
// it**: `vaCreateConfig` does not validate the slice structure against the records submitted, and a
// driver that rejected an unexpected structure reports it at `vaEndPicture` with
// `VA_STATUS_ERROR_ENCODING_ERROR` and no field name. Measured on iHD 26.3.5 with 640x368 (23
// macroblock rows, not a power of two) — accepted, so the driver's answer to this attribute was
// never needed.

/// `VA_RC_CQP` (`va.h:1116`): constant quantiser. The QP comes from the picture record's
/// `pic_init_qp` and each slice's `slice_qp_delta`, so nothing here is derived from a bitrate —
/// which is what makes it the honest first choice for a spike: a rate control mode would put the
/// picture's fidelity at the mercy of a number this backend does not have yet.
pub const RC_CQP: u32 = 0x0000_0010;
// The rate control modes and the encode buffer kinds that this spike does not submit are declared
// anyway and pinned by `the_encode_constants_are_the_ones_libva_uses`, for the same reason the
// decode side names the profiles it does not decode: a reader asking "is there a VA_RC_VCM?" needs
// the answer to be "not declared, therefore not used" rather than a guess. `cfg_attr` rather than a
// blanket `allow`, so a constant nothing references at all still warns.
#[cfg_attr(not(test), allow(dead_code))]
/// `VA_RC_CBR` (`va.h:1110`).
pub const RC_CBR: u32 = 0x0000_0002;
#[cfg_attr(not(test), allow(dead_code))]
/// `VA_RC_VBR` (`va.h:1112`).
pub const RC_VBR: u32 = 0x0000_0004;
#[cfg_attr(not(test), allow(dead_code))]
/// `VA_RC_ICQ` (`va.h:1121`): constant quality.
pub const RC_ICQ: u32 = 0x0000_0040;

/// `VA_ENC_PACKED_HEADER_SEQUENCE` (`va.h:1206`): the driver writes the SPS.
pub const PACKED_HEADER_SEQUENCE: u32 = 0x0000_0001;
/// `VA_ENC_PACKED_HEADER_PICTURE` (`va.h:1213`): the driver writes the PPS.
pub const PACKED_HEADER_PICTURE: u32 = 0x0000_0002;
#[cfg_attr(not(test), allow(dead_code))]
/// `VA_ENC_PACKED_HEADER_SLICE` (`va.h:1220`): the driver writes the slice headers. **Not** asked
/// for: a slice header the driver writes is one FilmCraft never sees as bytes, so an encoder that
/// wanted them would have to enable this and read them back out of the coded buffer. Recorded
/// because it is the difference between "the driver wrote the slice header" and "we told it what the
/// slice header says", and the second is the only one this backend can be honest about.
pub const PACKED_HEADER_SLICE: u32 = 0x0000_0004;

/// `VAEncMiscParameterType` (`va.h:1879-1915`). Only the three an iHD H.264 encoder is asked for
/// are named; the enum continues to `ExtensionData = 19` and none of the rest is used here.
///
/// `VAEncMiscParameterTypeFrameRate = 0`. An iHD H.264 encoder **refuses the render without it**:
/// `VA_STATUS_ERROR_INVALID_PARAMETER` out of `vaRenderPicture`. That is not in the header, and it
/// is the first thing this backend learned the hard way.
pub const MISC_FRAME_RATE: u32 = 0;
#[cfg_attr(not(test), allow(dead_code))]
/// `VAEncMiscParameterTypeRateControl = 1`, carrying a [`VaEncMiscParameterRateControl`].
pub const MISC_RATE_CONTROL: u32 = 1;
/// `VAEncMiscParameterTypeQuantization = 8`, carrying a [`VaEncMiscParameterQuantization`]. Under
/// [`RC_CQP`] this is what the iHD driver reads the picture's quantiser from, and it is submitted
/// with the sequence record rather than per picture (the header says "it's per-sequence parameter").
pub const MISC_QUANTIZATION: u32 = 8;

/// `VASliceType` as `VAEncSliceParameterBufferH264::slice_type` counts it: H.264's `slice_type`
/// minus five (`va_enc_h264.h:441`). So 0 = P, 1 = B, 2 = I, and 5..=7 are the "all slices in this
/// picture are of this type" forms — which is what a picture whose slices are all P wants.
///
/// **This offset-by-five is not documented in the header**, which says only "Range: 0..2, 5..7, i.e.
/// no switching slices" (`va_enc_h264.h:438-441`) and does not say what the numbers mean. Getting
/// it wrong does not fail a call: `slice_type = 2` on an IDR picture encodes correctly on iHD, and
/// `slice_type = 7` on a picture that should be P gives a picture a decoder may or may not accept.
/// It is asserted in `encoder.rs`'s tests against the driver's own output rather than left here as
/// a comment.
#[cfg_attr(not(test), allow(dead_code))]
pub const SLICE_TYPE_P: u8 = 0;
#[cfg_attr(not(test), allow(dead_code))]
/// See [`SLICE_TYPE_P`].
pub const SLICE_TYPE_B: u8 = 1;
#[cfg_attr(not(test), allow(dead_code))]
/// See [`SLICE_TYPE_P`].
pub const SLICE_TYPE_I: u8 = 2;
#[cfg_attr(not(test), allow(dead_code))]
/// See [`SLICE_TYPE_P`]: slice_type 5, every slice in the picture is a P slice.
pub const SLICE_TYPE_ALL_P: u8 = 5;
/// See [`SLICE_TYPE_P`]: slice_type 7, every slice in the picture is an I slice.
pub const SLICE_TYPE_ALL_I: u8 = 7;

/// `VAEntrypoint` (`va.h:553`): `VAEntrypointVLD` is the hardware video decoder, the entry point
/// every decode goes through.
pub const VLD: VAEntrypoint = 1;
/// `VAEntrypointEncSlice` (`va.h:558`): the slice-level encoder. Not used for decoding; listed so
/// a driver's entry points can be reported whole.
pub const ENC_SLICE: VAEntrypoint = 6;

/// `VAProfile` (`va.h:511-537`): the decoding profiles this backend can drive. A driver profile
/// outside this list still reaches [`Profile::Other`], so nothing is silently dropped.
pub const H264_MAIN: VAProfile = 6;
/// See [`H264_MAIN`].
pub const H264_HIGH: VAProfile = 7;
/// See [`H264_MAIN`].
pub const H264_CONSTRAINED_BASELINE: VAProfile = 13;
/// See [`H264_MAIN`].
pub const HEVC_MAIN: VAProfile = 17;
/// See [`H264_MAIN`].
pub const HEVC_MAIN_10: VAProfile = 18;
/// `VAProfileVP9Profile0` (`va.h:524`): the first of the four VP9 profiles.
pub const VP9_PROFILE_0: VAProfile = 19;
/// See [`VP9_PROFILE_0`].
pub const VP9_PROFILE_1: VAProfile = 20;
/// See [`VP9_PROFILE_0`].
pub const VP9_PROFILE_2: VAProfile = 21;
/// See [`VP9_PROFILE_0`].
pub const VP9_PROFILE_3: VAProfile = 22;
/// `VAProfileAV1Profile0` (`va.h:537`).
pub const AV1_MAIN: VAProfile = 32;

/// `RTLD_NOW | RTLD_LOCAL`: resolve every symbol at open time (a missing one is then an error
/// here rather than a jump through a null pointer later) and keep the library's symbols out of the
/// global namespace. Both are the Linux/glibc values, which is all `dlopen` needs; the workspace
/// has no `libc` to declare them from (`<dlfcn.h>`).
const RTLD_NOW: c_int = 0x2;
const RTLD_LOCAL: c_int = 0x0;

/// `O_RDWR` from `<fcntl.h>`: a DRM render node is opened read/write by the DRM client API.
const O_RDWR: c_int = 0x2;

/// The libraries to open, and where the symbols live (`libva.so.2` and `libva-drm.so.2`; the dash
/// is the distribution's, and the soname the only one to load).
const LIBVA: &[u8] = b"libva.so.2\0";
const LIBVA_DRM: &[u8] = b"libva-drm.so.2\0";

/// The DRM render node to open when the environment does not name one.
const DRM_NODE: &str = "/dev/dri/renderD128";
/// Names the DRM render node to open instead of [`DRM_NODE`]. A diagnostic escape hatch for a
/// machine whose render node is numbered differently (a second GPU), documented here because it is
/// the one thing in this file that changes behaviour from outside the program.
const DRM_NODE_ENV: &str = "FILMCRAFT_VAAPI_DRM_NODE";

/// How many entries a driver-reported list may hold: no driver has anything like this many
/// profiles or entry points, and the answer is what sizes the array we hand it.
const MAX_DRIVER_ENTRIES: usize = 4096;

/// The most surfaces one decode session allocates. A decoded picture buffer holds at most 16
/// pictures (15 for HEVC) and the decoder reads each picture back before decoding the next, so this
/// is far more than any conforming stream needs; it is here so that a stream description claiming an
/// absurd decoded picture buffer size costs a clamp rather than an allocation.
pub const MAX_SURFACES: usize = 32;

/// How many times `vaSyncSurface` is asked again while it reports the video engine busy. The engine
/// resets in a fraction of a frame; a driver still answering after this many is one that has stopped
/// answering, and the decode is reported as failed so the stream goes to the software decoder.
const MAX_SYNC_ATTEMPTS: u32 = 64;

/// The longest driver string copied out of libva (a vendor name is a few dozen bytes), so a
/// driver that hands back a huge or unterminated string cannot drive a large allocation.
const MAX_DRIVER_STRING: usize = 256;

/// The size the ABI probe image is created at: big enough for every driver to have a real pitch
/// and plane offsets, small enough to be free.
const PROBE_WIDTH: u16 = 64;
const PROBE_HEIGHT: u16 = 64;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
    fn dlclose(handle: *mut c_void) -> c_int;
    /// C's `open` is variadic; only the mode argument follows the flags, and only for `O_CREAT`,
    /// which is never used here. It is declared variadic so the call sets up the register the
    /// SysV ABI requires for a variadic call.
    fn open(path: *const c_char, flags: c_int, ...) -> c_int;
    fn close(fd: c_int) -> c_int;
}

// ---------------------------------------------------------------------------------------------
// The structures. Sizes and offsets follow libva 2.24's headers; see the tests at the bottom.
// ---------------------------------------------------------------------------------------------

/// `VAImageFormat` (`va.h:4717`): what a buffer of image data is — a four-character code, the bit
/// depth, and for packed formats the component masks. 48 bytes.
#[repr(C)]
#[derive(Default)]
pub struct VaImageFormat {
    pub fourcc: u32,
    /// `VA_LSB_FIRST` or `VA_MSB_FIRST` (`va.h:4714-4715`).
    pub byte_order: u32,
    pub bits_per_pixel: u32,
    /// Significant bits per pixel.
    pub depth: u32,
    pub red_mask: u32,
    pub green_mask: u32,
    pub blue_mask: u32,
    pub alpha_mask: u32,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAImage` (`va.h:4734`): the description libva hands back for a decoded picture — its size, how
/// many planes it has, and where each plane starts and how it is pitched. Every readback path
/// copies out of this, so a field declared at the wrong offset or width reads its neighbour's
/// value instead of its own; [`Display::probe_image_layout`] is the check against the library that
/// is installed, and `va_image_layout_matches_the_header` pins the offsets against the header.
/// 120 bytes.
///
/// The fields are `pub(crate)` rather than public: they are libva's memory layout rather than
/// FilmCraft's data, but [`crate::vaapi::validate_image`] is the single gate every readback goes
/// through and has to read the plane geometry the driver wrote — which is exactly what is unsafe to
/// read from anywhere else. Nothing outside this backend's one module tree reads them.
#[repr(C)]
#[derive(Default)]
pub struct VaImage {
    pub image_id: VAImageID,
    pub format: VaImageFormat,
    /// The buffer (`VAImageBufferType`) the image data lives in, to be mapped and unmapped.
    pub buf: VABufferID,
    pub width: u16,
    pub height: u16,
    /// Bytes the driver allocated for all planes together.
    pub data_size: u32,
    /// 1..=3 for the planar formats; whatever else a driver reports is not read.
    pub num_planes: u32,
    /// Bytes per row of each plane.
    pub pitches: [u32; 3],
    /// Byte offset of each plane within the mapped buffer.
    pub offsets: [u32; 3],
    /// Palette bookkeeping; zero for the YUV formats this backend takes.
    pub num_palette_entries: i32,
    pub entry_bytes: i32,
    pub component_order: [i8; 4],
    pub va_reserved: [u32; VA_PADDING_LOW],
}

// ---------------------------------------------------------------------------------------------
// The picture- and slice-parameter records: what a decode submits per access unit. Unlike the
// image above, these are built by [`crate::vaapi::picture`] and read back by its tests, so their
// fields are `pub`; the records themselves stay in this module's hands (`Picture` carries opaque
// owned bytes) and nothing here is handed to libva until the decode loop copies it into a buffer.
//
// The C declarations carry bitfields (`seq_fields`, `pic_fields`, `slice_parsing_fields`,
// `LongSliceFlags`), which Rust has no syntax for. Each is declared here as the union's `u32`
// `value` member, and `picture.rs` packs the syntax elements into it with the bit order
// `/usr/include/va/` writes them in — the same order a C compiler would, from bit 0 up. Reading a
// packed word back is the same operation the other way, which is what the record readers below
// and `picture.rs`'s tests do; each bit position is pinned there against the parsed value.
//
// Every field keeps the C name, including the deprecated ones the headers still declare
// (`num_slice_groups_minus1`, `tg_start`, …), because a renamed field is one the next reader has
// to look up and one a declaration fix has to match against the header.
//
// Where the C layout has a hole between two fields — alignment padding a C compiler inserts — the
// Rust declaration names it `padding_*` and gives it a width, which is the same thing libva does with
// its own `va_reserved` fields. Naming every byte matters here: `#[derive(Default)]` does not
// initialise implicit padding, so a record with unnamed padding would have indeterminate bytes and
// `record_bytes` could not hand them to a driver as a `[u8]`. With the holes named, every byte of a
// record is a field, and every field is initialised. The widths are pinned with the rest of the
// layout by `the_picture_parameter_records_match_the_header`.
//
// Sizes and offsets are libva 2.24's own (`va.h:3572` and `va_dec_hevc.h:57`), pinned by
// `the_picture_parameter_records_match_the_header`.
// ---------------------------------------------------------------------------------------------

/// `VAPictureH264` (`va.h:3572`): one picture's identity — which surface it decodes into, which
/// slot it keeps in the DPB and its picture order counts. 36 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaPictureH264 {
    /// The surface the picture is decoded into (the render target).
    pub picture_id: VABufferID,
    /// `frame_num` of the slice header, which is how the DPB slot is referred to.
    pub frame_idx: u32,
    /// `VA_PICTURE_H264_*` bits ([`PICTURE_H264_SHORT_TERM_REFERENCE`] and
    /// [`PICTURE_H264_LONG_TERM_REFERENCE`]).
    pub flags: u32,
    pub top_field_order_cnt: i32,
    pub bottom_field_order_cnt: i32,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAPictureParameterBufferH264` (`va.h:3599`): the picture-level record, one per access unit,
/// submitted before its slices. 672 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaPictureParameterBufferH264 {
    /// The picture this record describes.
    pub curr_pic: VaPictureH264,
    /// The DPB, as surfaces. This backend leaves them to the decode loop, which owns the surface
    /// ring; see `crate::vaapi::picture`.
    pub reference_frames: [VaPictureH264; 16],
    pub picture_width_in_mbs_minus1: u16,
    pub picture_height_in_mbs_minus1: u16,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    /// How many entries of `reference_frames` are valid.
    pub num_ref_frames: u8,
    /// The C layout's hole at offset 619, before `seq_fields`.
    pub padding_after_num_ref_frames: [u8; 1],
    /// `seq_fields`: the sequence parameter set's flags, packed into the union's `value` member.
    pub seq_fields: u32,
    /// FMO is not supported, so these stay at what the picture parameter set says and the
    /// decode is declined if it asks for more than one slice group.
    pub num_slice_groups_minus1: u8,
    pub slice_group_map_type: u8,
    pub slice_group_change_rate_minus1: u16,
    pub pic_init_qp_minus26: i8,
    pub pic_init_qs_minus26: i8,
    pub chroma_qp_index_offset: i8,
    pub second_chroma_qp_index_offset: i8,
    /// `pic_fields`: the picture parameter set's flags plus what this slice says, packed into the
    /// union's `value` member.
    pub pic_fields: u32,
    /// `frame_num` of the first slice of the picture.
    pub frame_num: u16,
    /// The C layout's hole at offsets 638-639, before `va_reserved`.
    pub padding_after_frame_num: [u8; 2],
    pub va_reserved: [u32; VA_PADDING_MEDIUM],
}

/// `VASliceParameterBufferH264` (`va.h:3664`): one slice's record, submitted per slice. 3128
/// bytes, most of it the two reference picture lists and the prediction weights.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaSliceParameterBufferH264 {
    /// Bytes of this slice in the slice data buffer.
    pub slice_data_size: u32,
    /// Byte offset of this slice's NAL unit in the slice data buffer.
    pub slice_data_offset: u32,
    /// `VA_SLICE_DATA_FLAG_*` ([`SLICE_DATA_FLAG_ALL`] for a whole slice).
    pub slice_data_flag: u32,
    /// Bits from the NAL header to the start of `slice_data()`, emulation prevention bytes
    /// removed — which is where FilmCraft's own parser leaves the bit position.
    pub slice_data_bit_offset: u16,
    pub first_mb_in_slice: u16,
    pub slice_type: u8,
    pub direct_spatial_mv_pred_flag: u8,
    pub num_ref_idx_l0_active_minus1: u8,
    pub num_ref_idx_l1_active_minus1: u8,
    pub cabac_init_idc: u8,
    pub slice_qp_delta: i8,
    pub disable_deblocking_filter_idc: u8,
    pub slice_alpha_c0_offset_div2: i8,
    pub slice_beta_offset_div2: i8,
    /// The C layout's hole at offsets 25-27, before `RefPicList0[]`.
    pub padding_before_ref_pic_list0: [u8; 3],
    pub ref_pic_list0: [VaPictureH264; 32],
    pub ref_pic_list1: [VaPictureH264; 32],
    pub luma_log2_weight_denom: u8,
    pub chroma_log2_weight_denom: u8,
    pub luma_weight_l0_flag: u8,
    /// The C layout's hole at offset 2335, before `luma_weight_l0[]`.
    pub padding_before_luma_weight_l0: [u8; 1],
    pub luma_weight_l0: [i16; 32],
    pub luma_offset_l0: [i16; 32],
    pub chroma_weight_l0_flag: u8,
    /// The C layout's hole at offset 2465, before `chroma_weight_l0[][]`.
    pub padding_before_chroma_weight_l0: [u8; 1],
    pub chroma_weight_l0: [[i16; 2]; 32],
    pub chroma_offset_l0: [[i16; 2]; 32],
    pub luma_weight_l1_flag: u8,
    /// The C layout's hole at offset 2723, before `luma_weight_l1[]`.
    pub padding_before_luma_weight_l1: [u8; 1],
    pub luma_weight_l1: [i16; 32],
    pub luma_offset_l1: [i16; 32],
    pub chroma_weight_l1_flag: u8,
    /// The C layout's hole at offset 2853, before `chroma_weight_l1[][]`.
    pub padding_before_chroma_weight_l1: [u8; 1],
    pub chroma_weight_l1: [[i16; 2]; 32],
    pub chroma_offset_l1: [[i16; 2]; 32],
    /// The C layout's hole at offsets 3110-3111, before `va_reserved`.
    pub padding_before_va_reserved: [u8; 2],
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAIQMatrixBufferH264` (`va.h`): the quantizer and inverse-quantizer scaling lists a picture is
/// decoded with. 240 bytes — six 4x4 lists, six 8x8 lists, then the reserved tail.
///
/// Both arrays are in **raster scan order**, which is the order FilmCraft's own parser keeps its
/// `ScalingMatrices` in (`params.rs`: `zz4_to_raster`), so filling this record from the sequence
/// parameter set is a copy rather than a re-scan. A stream whose sequence parameter set says
/// `seq_scaling_matrix_present_flag == 0` gets flat lists of 16 (`ScalingMatrices::flat`), which
/// is what 7.4.2.1.1 specifies as the default and what every encoder uses when it has no
/// quantizer matrices to send.
///
/// This is a buffer a decode cannot be without: see [`IQ_MATRIX_BUFFER`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VaIqMatrixBufferH264 {
    /// `[Intra Y, Intra Cb, Intra Cr, Inter Y, Inter Cb, Inter Cr]`, raster order.
    pub scaling_list4x4: [[u8; 16]; 6],
    /// `[Intra Y, Inter Y]`, raster order — the 8x8 arrays are `[2][64]`, so the record holds the
    /// luma pair and nothing for chroma: 8x8 chroma lists only exist at `chroma_format_idc == 3`
    /// (4:4:4), which [`super::plan_nal`] declines.
    pub scaling_list8x8: [[u8; 64]; 2],
    pub va_reserved: [u32; VA_PADDING_LOW],
}

impl Default for VaIqMatrixBufferH264 {
    /// Flat lists of 16 — the quantizer every stream without custom matrices decodes with
    /// (7.4.2.1.1's default), so the empty record is the right one to hand a driver rather than
    /// zeroes, which would name a quantizer nothing in H.264 allows.
    fn default() -> Self {
        Self { scaling_list4x4: [[16; 16]; 6], scaling_list8x8: [[16; 64]; 2], va_reserved: [0; VA_PADDING_LOW] }
    }
}

/// `VAPictureHEVC` (`va.h:5292`): the HEVC counterpart of [`VaPictureH264`]. 28 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaPictureHevc {
    /// The surface the picture is decoded into.
    pub picture_id: VABufferID,
    /// Picture order count, which is how the DPB slot is referred to.
    pub pic_order_cnt: i32,
    /// `VA_PICTURE_HEVC_*` bits.
    pub flags: u32,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAPictureParameterBufferHEVC` (`va_dec_hevc.h:57`): the HEVC picture-level record. 604 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaPictureParameterBufferHevc {
    pub curr_pic: VaPictureHevc,
    /// The DPB, as surfaces; the decode loop's to fill, like `reference_frames` above.
    pub reference_frames: [VaPictureHevc; 15],
    pub pic_width_in_luma_samples: u16,
    pub pic_height_in_luma_samples: u16,
    /// `pic_fields`: the sequence and picture parameter sets' flags, packed into the union's
    /// `value` member.
    pub pic_fields: u32,
    pub sps_max_dec_pic_buffering_minus1: u8,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub pcm_sample_bit_depth_luma_minus1: u8,
    pub pcm_sample_bit_depth_chroma_minus1: u8,
    pub log2_min_luma_coding_block_size_minus3: u8,
    pub log2_diff_max_min_luma_coding_block_size: u8,
    pub log2_min_transform_block_size_minus2: u8,
    pub log2_diff_max_min_transform_block_size: u8,
    pub log2_min_pcm_luma_coding_block_size_minus3: u8,
    pub log2_diff_max_min_pcm_luma_coding_block_size: u8,
    pub max_transform_hierarchy_depth_intra: u8,
    pub max_transform_hierarchy_depth_inter: u8,
    pub init_qp_minus26: i8,
    pub diff_cu_qp_delta_depth: u8,
    pub pps_cb_qp_offset: i8,
    pub pps_cr_qp_offset: i8,
    pub log2_parallel_merge_level_minus2: u8,
    pub num_tile_columns_minus1: u8,
    pub num_tile_rows_minus1: u8,
    pub column_width_minus1: [u16; 19],
    pub row_height_minus1: [u16; 21],
    /// `slice_parsing_fields`: what a driver needs to parse this slice header, packed into the
    /// union's `value` member.
    pub slice_parsing_fields: u32,
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    pub num_short_term_ref_pic_sets: u8,
    pub num_long_term_ref_pic_sps: u8,
    pub num_ref_idx_l0_default_active_minus1: u8,
    pub num_ref_idx_l1_default_active_minus1: u8,
    pub pps_beta_offset_div2: i8,
    pub pps_tc_offset_div2: i8,
    pub num_extra_slice_header_bits: u8,
    /// Bits `short_term_ref_pic_set()` takes in this slice header, so a driver can skip it.
    pub st_rps_bits: u32,
    pub va_reserved: [u32; VA_PADDING_MEDIUM],
}

/// `VASliceParameterBufferHEVC` (`va_dec_hevc.h:358`): one slice segment's record. 264 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaSliceParameterBufferHevc {
    pub slice_data_size: u32,
    pub slice_data_offset: u32,
    /// `VA_SLICE_DATA_FLAG_*` ([`SLICE_DATA_FLAG_ALL`] for a whole segment).
    pub slice_data_flag: u32,
    /// Bytes from the NAL header to the start of `slice_segment_data()`.
    pub slice_data_byte_offset: u32,
    pub slice_segment_address: u32,
    /// Indices into `reference_frames`; `0xFF` marks an entry the picture does not use.
    pub ref_pic_list: [[u8; 15]; 2],
    /// The C layout's hole at offsets 50-51, before `LongSliceFlags`.
    pub padding_before_long_slice_flags: [u8; 2],
    /// `LongSliceFlags`: the slice header's own syntax elements, packed into the union's `value`
    /// member.
    pub long_slice_flags: u32,
    /// Index of the collocated picture, `0xFF` when temporal MVP is off.
    pub collocated_ref_idx: u8,
    pub num_ref_idx_l0_active_minus1: u8,
    pub num_ref_idx_l1_active_minus1: u8,
    pub slice_qp_delta: i8,
    pub slice_cb_qp_offset: i8,
    pub slice_cr_qp_offset: i8,
    pub slice_beta_offset_div2: i8,
    pub slice_tc_offset_div2: i8,
    pub luma_log2_weight_denom: u8,
    pub delta_chroma_log2_weight_denom: i8,
    pub delta_luma_weight_l0: [i8; 15],
    pub luma_offset_l0: [i8; 15],
    pub delta_chroma_weight_l0: [[i8; 2]; 15],
    pub chroma_offset_l0: [[i8; 2]; 15],
    pub delta_luma_weight_l1: [i8; 15],
    pub luma_offset_l1: [i8; 15],
    pub delta_chroma_weight_l1: [[i8; 2]; 15],
    pub chroma_offset_l1: [[i8; 2]; 15],
    pub five_minus_max_num_merge_cand: u8,
    /// The C layout's hole at offset 247, before `num_entry_point_offsets`.
    pub padding_before_num_entry_point_offsets: [u8; 1],
    pub num_entry_point_offsets: u16,
    pub entry_offset_to_subset_array: u16,
    /// Emulation prevention bytes in this slice header, which is what the entry point offsets and
    /// `slice_data_byte_offset` are counted without.
    pub slice_data_num_emu_prevn_bytes: u16,
    /// The C layout's hole at offsets 254-255, before `va_reserved`.
    pub padding_before_va_reserved: [u8; 2],
    pub va_reserved: [u32; VA_PADDING_LOW - 2],
}

/// `VADecPictureParameterBufferVP9` (`va_dec_vp9.h:58`): the VP9 picture-level record. **92 bytes**.
///
/// **VP9's shape is not H.264's or HEVC's, and the one that matters is `reference_frames`.** It is
/// `VASurfaceID[8]` and it is the *whole* decoded picture buffer, not a list: there is no count
/// beside it, no per-slice list that indexes it, and nothing that says how many of the eight slots
/// are live. A slot that holds no picture says so with `VA_INVALID_SURFACE` (`va.h:1648`), because a
/// slot naming a surface the driver never created is a reference to whatever surface 0 happens to
/// hold. The `last_ref_frame` / `golden_ref_frame` / `alt_ref_frame` bits in `pic_fields` are indices
/// *into this array*, so it is filled in whole or not at all.
///
/// The other two fields with no counterpart in the per-frame header are the ones this backend's
/// parser supplies rather than a bitstream field:
/// - `filter_level` / `sharpness_level` are the loop filter's own state, which VP9 6.2.8 carries
///   across frames (its deltas are conditional updates to the previous frame's).
/// - `mb_segment_tree_probs` / `segment_pred_probs` are VP9 6.2.11's segmentation probabilities,
///   read only when `segmentation_update_map` is set and therefore persisting from frame to frame.
///   The header's own comment on them — "They should be parsed directly from bitstream by
///   application" (`va_dec_vp9.h:172-175`) — is what makes them this backend's job rather than the
///   driver's.
///
/// `frame_header_length_in_bytes` is the byte length of the **uncompressed** header
/// (`va_dec_vp9.h:157-164`) and `first_partition_size` that of the **compressed** one
/// (`va_dec_vp9.h:167-169`, "the byte count of compressed header the bitstream buffer"). They are
/// two different counts of the same frame and neither can be derived from the other alone, which is
/// why the record is built from the frame's bytes as well as from its parsed header.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaPictureParameterBufferVp9 {
    /// Coded picture width, "may not be multiple of 8" (`va_dec_vp9.h:59-62`).
    pub frame_width: u16,
    /// Coded picture height (`va_dec_vp9.h:63-66`).
    pub frame_height: u16,
    /// The whole decoded picture buffer, one surface per slot; see the note above.
    pub reference_frames: [VASurfaceID; 8],
    /// `pic_fields`: the header's flags and frame-context numbers, packed into the union's `value`
    /// member by [`crate::vaapi::picture::vp9_pic_fields`].
    pub pic_fields: u32,
    pub filter_level: u8,
    pub sharpness_level: u8,
    pub log2_tile_rows: u8,
    pub log2_tile_columns: u8,
    pub frame_header_length_in_bytes: u8,
    /// The C layout's hole at offset 45, before `first_partition_size`.
    pub padding_before_first_partition_size: [u8; 1],
    pub first_partition_size: u16,
    pub mb_segment_tree_probs: [u8; 7],
    pub segment_pred_probs: [u8; 3],
    /// VP9 profile, "value range [0..3]" (`va_dec_vp9.h:179-182`).
    pub profile: u8,
    /// Bit depth per sample, luma and chroma alike (`va_dec_vp9.h:184-187`).
    pub bit_depth: u8,
    pub va_reserved: [u32; VA_PADDING_MEDIUM],
}

/// `VASegmentParameterVP9` (`va_dec_vp9.h:204`): one segment's parameters. **36 bytes**, eight of
/// them in every [`VaSliceParameterBufferVp9`].
///
/// **36 is 34 bytes of fields plus a 2-byte hole**, and the hole is the header's as much as the C
/// compiler's: the four `int16_t` quant scales end at 18 and the `uint32_t[4]` reserved tail is
/// 4-aligned, so `va_reserved` starts at 20. The hole is named below
/// (`padding_before_va_reserved`) rather than left to the compiler to insert, because a record this
/// crate hands to a driver is turned into bytes by [`record_bytes`], and an unnamed hole is a pair of
/// bytes nothing wrote. See the note above this file's structures.
///
/// The record is thin for a reason worth stating, because "thin" is what makes it look optional:
/// **VP9 has no slice headers** — 6.2 returns from `frame_header()` after the tile and quantizer
/// syntax, and there is no slice layer at all — so none of H.264's or HEVC's per-slice parsing
/// fields has anything to go in it. What is left is the per-segment state, and that is **not**
/// derivable from the picture parameter record beside it: `VADecPictureParameterBufferVP9` carries
/// `base_q` nowhere (read the whole of `va_dec_vp9.h:58-192`: no quantizer field at all, only
/// `filter_level` and `sharpness_level`), so the **quantizer this frame is dequantized with
/// reaches the driver here and nowhere else**. A VP9 decode that omits this record does not merely
/// lose a refinement — the driver has no qindex at all.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaSegmentParameterVp9 {
    /// `segment_flags`, packed into the union's `value` member the way
    /// [`VaPictureParameterBufferVp9::pic_fields`] is: bit 0 is `segment_reference_enabled`, bits
    /// 1-2 `segment_reference`, bit 3 `segment_reference_skipped`
    /// (`va_dec_vp9.h:212-226`). `segment_reference` is VP9 6.2.11's `SEG_LVL_REF_FRAME` feature
    /// data in its own numbering: "0: reserved, 1: Last ref, 2: golden, 3: altref"
    /// (`va_dec_vp9.h:213-220`).
    pub segment_flags: u16,
    /// `filter_level[4][2]` (`va_dec_vp9.h:231-235`): "Specifies the filter level information per
    /// segment", indexed `[ref][mode]` — the header's own comment is that it "corresponds to
    /// variable `lfi->lvl[seg][ref][mode]` in VP9 code" (`va_dec_vp9.h:232-234`), so it is VP9
    /// 8.8.1's `FilterLevel[ref][mode]` table copied across with the segment index dropped.
    pub filter_level: [[u8; 2]; 4],
    /// "Specifies per segment Luma AC quantization scale" (`va_dec_vp9.h:236-239`). See
    /// [`crate::vaapi::picture::vp9_slice_record`] for which of the two specification lookup tables
    /// fills which of the four, and why the field name is not the only evidence for it.
    pub luma_ac_quant_scale: i16,
    /// "Specifies per segment Luma DC quantization scale" (`va_dec_vp9.h:240-244`).
    pub luma_dc_quant_scale: i16,
    /// "Specifies per segment Chroma AC quantization scale" (`va_dec_vp9.h:245-249`).
    pub chroma_ac_quant_scale: i16,
    /// "Specifies per segment Chroma DC quantization scale" (`va_dec_vp9.h:250-254`).
    pub chroma_dc_quant_scale: i16,
    /// The C layout's hole at offsets 18-19, before `va_reserved` — **the header has one too, and a
    /// declaration that did not name it handed uninitialised bytes to the driver.**
    ///
    /// The four quant scales end at 18 and `va_reserved` is a `uint32_t[4]`, which is 4-aligned and
    /// therefore starts at 20; C inserts two bytes of padding between them and `sizeof` is 36 rather
    /// than the 34 the fields add up to. Both halves of that are measured against
    /// `/usr/include/va/va_dec_vp9.h` by a C compiler (`the_vp9_records_are_the_sizes_the_header_declares`),
    /// not read off the field list.
    ///
    /// It is named, and given a width of exactly 2, for the reason the note above this file's
    /// structures gives and for no other: **`#[derive(Default)]` and a struct literal initialise
    /// named fields only, so an unnamed hole is indeterminate memory**, `record_bytes` would hand the
    /// driver whatever was on the stack in its place, and eight of these records are submitted for
    /// every VP9 picture. It decoded correctly anyway, because the driver ignores those bytes — which
    /// is a fact about this driver, not about the record. `[u8; 2]` fills 18-19 and aligns to 1, so
    /// `va_reserved` stays 4-aligned at 20 and every offset after it is still the header's.
    pub padding_before_va_reserved: [u8; 2],
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VASliceParameterBufferVP9` (`va_dec_vp9.h:279`): the record a VP9 decode submits **once per
/// frame**, with the frame's bytes. **316 bytes**.
///
/// ## A VP9 decode cannot do without this, and the header says so twice
///
/// It is not declared optional and it is not optional in practice: a submission without it is
/// accepted by `vaCreateBuffer`, `vaMapBuffer` and `vaRenderPicture` and refused at
/// `vaEndPicture` with `VA_STATUS_ERROR_INVALID_PARAMETER`, which is where this backend spent a
/// whole milestone's debugging time before the buffer was sent at all.
///
/// The two sentences in the header that make it unavoidable:
///
/// - *"Slice data buffer of `VASliceDataBufferType` is used to send the bitstream which should
///   include whole or part of partition 0 (at least compressed header) to the end of frame"*
///   (`va_dec_vp9.h:274-276`). The slice *is* the frame, from partition 0 to the end — which is
///   what [`slice_data_size`] counts and where [`slice_data_offset`] points.
/// - *"When segmentation is disabled, only `SegParam[0]` has valid values, all other entries
///   should be populated with 0. Otherwise, all eight entries should be valid"*
///   (`va_dec_vp9.h:270-272`). Even a frame with no segmentation at all has to carry the
///   quantizer in `SegParam[0]`, because the picture parameter record has nowhere else to put it.
///
/// ## What is *not* in it, and why that is not a reason to send something else
///
/// There is no slice-header parsing field, no reference list and no scaling list, because VP9 has
/// no slice headers and no scaling lists. That makes this record *small and fully derived from the
/// frame's own header plus the reference set* — it does not make it absent. The distinction that
/// caused a wrong turn once: **"VP9 has no slice headers" is a fact about the codec, and the
/// conclusion it does not license is "so the API has no slice record".** `va_dec_vp9.h:279`
/// declares one, in a file named `va_dec_vp9.h`, and its fields are the geometry and the
/// segmentation parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaSliceParameterBufferVp9 {
    /// "The byte count of current frame in the bitstream buffer, starting from first byte of the
    /// buffer. It uses the name `slice_data_size` to be consitent with other codec, but actually
    /// means `frame_data_size`" (`va_dec_vp9.h:280-284`). **The header names this out**: the count
    /// is the whole frame from byte 0, not the extent of a slice inside it.
    pub slice_data_size: u32,
    /// "offset to the first byte of partition data (control partition)" (`va_dec_vp9.h:286-289`).
    pub slice_data_offset: u32,
    /// "see `VA_SLICE_DATA_FLAG_XXX` definitions" (`va_dec_vp9.h:290-293`) —
    /// [`SLICE_DATA_FLAG_ALL`], since a frame is submitted whole.
    pub slice_data_flag: u32,
    /// "per segment information" (`va_dec_vp9.h:295-298`), eight of them: see
    /// [`VaSegmentParameterVp9`] and the rule the header states about when each one is valid.
    pub seg_param: [VaSegmentParameterVp9; 8],
    pub va_reserved: [u32; VA_PADDING_LOW],
}

// ---------------------------------------------------------------------------------------------
// The encode records. Encode is the mirror image of decode (the plan's "decisive difference"): the
// app hands the driver a picture and the driver writes the slice headers, so these are the records
// that *describe* what to encode rather than what was parsed out of a bitstream. They are built by
// [`crate::vaapi::encoder`] and handed to libva as bytes.
//
// The sizes and offsets below were read out of `/usr/include/va/va_enc_h264.h` and `va.h` on the
// target machine (libva 1.24.0) and are pinned by `the_encode_records_match_the_header`. Three of
// them are worth naming here, because each is a place where a plausible-looking declaration would
// be silently wrong and every example on the web copies the wrong one:
//
// - **`VAEncSliceParameterBufferH264` is 3140 bytes, not the 3128 of its decode counterpart.** The
//   decode record's two reference lists follow a five-byte prefix of small fields; the encode
//   record's follow `idr_pic_id` and `pic_order_cnt_lsb`, which are `uint16_t` and put the lists at
//   offsets 36 and 1188. Handing the driver 3128 bytes of a 3140-byte record leaves the tail of the
//   second list and every field after it uninitialised.
// - **`pic_order_cnt_lsb` is a `uint16_t` at offset 16, and `delta_pic_order_cnt_bottom` an `i32`
//   at 20** — with a 2-byte hole at 18-19 that the C layout has and `#[derive(Default)]` would not.
// - **`VAEncSequenceParameterBufferH264` is 1132 bytes** and 1028 of them are
//   `offset_for_ref_frame[256]` at offset 44, which every field after it is placed relative to.
//
// Where the C layout has a hole between two fields, the Rust declaration names it `padding_*` and
// gives it a width, for the reason the note above this section gives: `Default` does not
// initialise implicit padding, and a record with unnamed padding has bytes that
// [`record_bytes`] would hand to a driver as whatever was on the stack.
// ---------------------------------------------------------------------------------------------

/// `VACodedBufferSegment` (`va.h:3945`): one segment of a mapped [`ENC_CODED_BUFFER`] — how many
/// bytes, where they are, what the driver says about them, and the next segment. 48 bytes.
///
/// **This is where the encoded picture actually comes from**, and getting it wrong is the single
/// largest difference between this backend's encode path and its decode path. libva 1.24's
/// `vaMapBuffer` documents that for a coded buffer "`pbuf` points to link-list of
/// `VACodedBufferSegment`, and the list is terminated if `next` is NULL" (`va.h:3973-3975`). There is
/// no count, no index and no per-slice offset anywhere in the encode API of this libva:
///
/// - **`VAEncSliceParameterBufferH264` has no `slice_data_offset` and no `slice_data_size`.** Every
///   H.264 *decode* record has them (`va.h:3665-3667`), the encode record does not, and an encoder
///   written against the decode record's shape will not compile against this one. The plan's Task 1
///   says "read the driver's output using the `slice_data_offset` / `slice_data_size` it reports";
///   **on libva 1.24 there is nothing to read them from**, and that is the first thing the next
///   person has to know.
/// - The per-slice records the driver writes back are not described by any header either.
///
/// So the picture is read as this linked list, and [`Encoder::read_coded`] reads the **length** from
/// its head. The list is the driver's own, so a second segment is refused rather than followed (see
/// [`CodedReadback::picture`]); the bytes themselves come out of the buffer's own storage, mapped by
/// [`crate::vaapi::drm`] rather than through the `buf` pointer below.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaCodedBufferSegment {
    /// Bytes of the picture in this segment.
    pub size: u32,
    /// Bits from the start of the segment where the video data begins. **A driver may use a
    /// non-zero one**, which would mean the first bits of the segment are not the picture — this
    /// backend refuses anything but 0 rather than shifting a buffer whose unit it cannot see.
    pub bit_offset: u32,
    /// `VA_CODED_BUF_STATUS_*` (`va.h:3897-3933`). The low byte is the picture's average QP, which
    /// is the closest thing the driver says about the quality it actually delivered — worth
    /// reporting, and worth reporting *as a measurement* rather than as the QP that was asked for.
    pub status: u32,
    /// Reserved by the header.
    pub reserved: u32,
    /// Where this segment's bytes are. **The readback does not follow it.** libva documents it as
    /// "Pointer to the start of the data buffer" (`va.h:3949`) and vouches for no length, so reading
    /// it means reading at a driver-chosen pointer whose extent nothing reports — the thing
    /// [`crate::vaapi::drm`] exists to stop. The picture is found at the start of the buffer's own
    /// storage instead, at the length this record's `size` gives.
    pub buf: *mut c_void,
    /// The next segment, or null for the last.
    pub next: *mut c_void,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VABufferInfo` (`va.h:4032`): what a buffer's underlying storage *is*.
///
/// This is the only documented query libva offers of a buffer's memory, and it is the one that makes
/// the encode readback possible: `handle` is the DRM buffer object behind a
/// [`ENC_CODED_BUFFER`] and `mem_size` is "Size of the underlying buffer" (`va.h:4046`), which is
/// what [`crate::vaapi::drm`] maps. Measured on this stack: `mem_type` is
/// `VA_SURFACE_ATTRIB_MEM_TYPE_DRM` and `mem_size` is exactly the size the buffer was created with.
///
/// `va.h:4086` requires the struct to be zeroed on input — `VaBufferInfo::ZEROED` is that value —
/// because `mem_type` doubles as a hint on the way in.
///
/// **This one is 40 bytes, and every byte of it matters.** The four fields above are 24; the header's
/// `va_reserved[VA_PADDING_LOW]` is the other 16 (`va.h:4048-4049`). It was declared as 24 here, and
/// that is a live bug rather than a latent one: this is the one structure in the module that is
/// handed to libva **by pointer to a stack local**, and libva was compiled against the 40-byte one.
/// Measured on this stack, `vaAcquireBufferHandle` left every byte of 24..40 exactly as it found it,
/// so the truncation costs nothing *here* — but "the driver happens not to touch the tail" is not a
/// property the declaration may rely on, and a driver that clears its own reserved tail would be
/// writing 16 bytes past the end of our frame. The rule the whole file follows is "matches the
/// header", and a short struct is the size-only version of a wrong-arity call.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaBufferInfo {
    /// "Buffer handle" (`va.h:4034`). A DRM buffer-object handle on this stack, which is why it is
    /// `usize` here and narrowed to `u32` before it is used as one.
    pub handle: usize,
    /// "Buffer type (See VABufferType)." Out-parameter; unused, the buffer's type is known.
    pub buffer_type: u32,
    /// "Buffer memory type (See VASurfaceAttribMemoryType)." A hint on input, the type actually used
    /// on output. `0x20000000` is `VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME` (`va_drmcommon.h:84`; note
    /// there is no `VA_SURFACE_ATTRIB_MEM_TYPE_DRM` in any header — `va.h:1752` only reserves bits
    /// 31:28 for Linux DRM).
    pub mem_type: u32,
    /// "Size of the underlying buffer" (`va.h:4046`).
    pub mem_size: usize,
    /// "Reserved bytes for future use, must be zero" (`va.h:4048-4049`). Declared so this record is
    /// the size `vaAcquireBufferHandle` was compiled with; see the note above.
    pub va_reserved: [u32; VA_PADDING_LOW],
}

impl VaBufferInfo {
    /// The all-zero struct `va.h:4086` requires on input to `vaAcquireBufferHandle`.
    pub const ZEROED: Self = Self { handle: 0, buffer_type: 0, mem_type: 0, mem_size: 0, va_reserved: [0; VA_PADDING_LOW] };
}

/// `VAEncSequenceParameterBufferH264` (`va_enc_h264.h:148`): what to encode — the sequence
/// parameter set's own syntax elements, the periods, and the picture geometry in macroblocks.
/// 1132 bytes.
///
/// One of these is submitted with **every** picture, not once per sequence: iHD reads it per
/// `vaRenderPicture`, and a render that omits it is refused.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VaEncSequenceParameterBufferH264 {
    pub seq_parameter_set_id: u8,
    pub level_idc: u8,
    /// The C layout's hole at offsets 2-3, before `intra_period`.
    pub padding_after_level_idc: [u8; 2],
    /// Period between I frames; 0 for none.
    pub intra_period: u32,
    /// Period between IDR frames; 0 for none, which is what makes the first picture the only
    /// key frame.
    pub intra_idr_period: u32,
    /// Period between I/P frames; 0 for none.
    pub ip_period: u32,
    /// The bitrate, which [`RC_CQP`] ignores but which iHD still rejects as 0.
    pub bits_per_second: u32,
    /// `max_num_ref_frames` of the sequence parameter set.
    pub max_num_ref_frames: u32,
    /// **Macroblocks, not pixels** — 640x360 is 40x23 here, and 40x22 for the padded height.
    pub picture_width_in_mbs: u16,
    pub picture_height_in_mbs: u16,
    /// `seq_fields`: the sequence parameter set's flags, packed into the union's `value` member in
    /// the bit order `va_enc_h264.h:179-198` writes them.
    pub seq_fields: u32,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    /// Only meaningful when `seq_fields` says `pic_order_cnt_type == 1`.
    pub num_ref_frames_in_pic_order_cnt_cycle: u8,
    /// The C layout's hole at offset 35, before `offset_for_non_ref_pic`.
    pub padding_after_num_ref_frames_in_pic_order_cnt_cycle: [u8; 1],
    pub offset_for_non_ref_pic: i32,
    pub offset_for_top_to_bottom_field: i32,
    /// 256 entries at offset 44 — a kilobyte of the record, and every field after it sits past it.
    pub offset_for_ref_frame: [i32; 256],
    /// Whether the picture is cropped, which is what makes a size that is not a multiple of 16
    /// encodable: the surface is 640x368 and the crop says the bottom 4 rows are not picture.
    pub frame_cropping_flag: u8,
    /// The C layout's hole at offsets 1069-1071, before `frame_crop_left_offset`.
    pub padding_after_frame_cropping_flag: [u8; 3],
    pub frame_crop_left_offset: u32,
    pub frame_crop_right_offset: u32,
    pub frame_crop_top_offset: u32,
    pub frame_crop_bottom_offset: u32,
    pub vui_parameters_present_flag: u8,
    /// The C layout's hole at offsets 1089-1091, before `vui_fields`.
    pub padding_after_vui_parameters_present_flag: [u8; 3],
    /// `vui_fields`: the VUI's flags, packed like `seq_fields`.
    pub vui_fields: u32,
    pub aspect_ratio_idc: u8,
    /// The C layout's hole at offsets 1097-1099, before `sar_width`.
    pub padding_after_aspect_ratio_idc: [u8; 3],
    pub sar_width: u32,
    pub sar_height: u32,
    pub num_units_in_tick: u32,
    pub time_scale: u32,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

impl Default for VaEncSequenceParameterBufferH264 {
    /// Every field zero, which is [`Self::default`] by hand because `[i32; 256]` has no
    /// `Default`. The whole `offset_for_ref_frame` array is explicitly zeroed rather than left
    /// indeterminate: it is 1 KiB of the record, every byte of it reaches the driver, and an
    /// uninitialised word there would be a picture order count cycle this module never asked for.
    fn default() -> Self {
        Self {
            seq_parameter_set_id: 0,
            level_idc: 0,
            padding_after_level_idc: [0; 2],
            intra_period: 0,
            intra_idr_period: 0,
            ip_period: 0,
            bits_per_second: 0,
            max_num_ref_frames: 0,
            picture_width_in_mbs: 0,
            picture_height_in_mbs: 0,
            seq_fields: 0,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            padding_after_num_ref_frames_in_pic_order_cnt_cycle: [0; 1],
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            offset_for_ref_frame: [0; 256],
            frame_cropping_flag: 0,
            padding_after_frame_cropping_flag: [0; 3],
            frame_crop_left_offset: 0,
            frame_crop_right_offset: 0,
            frame_crop_top_offset: 0,
            frame_crop_bottom_offset: 0,
            vui_parameters_present_flag: 0,
            padding_after_vui_parameters_present_flag: [0; 3],
            vui_fields: 0,
            aspect_ratio_idc: 0,
            padding_after_aspect_ratio_idc: [0; 3],
            sar_width: 0,
            sar_height: 0,
            num_units_in_tick: 0,
            time_scale: 0,
            va_reserved: [0; VA_PADDING_LOW],
        }
    }
}

/// `VAEncPictureParameterBufferH264` (`va_enc_h264.h:296`): one picture's identity, which surface it
/// is reconstructed into, its reference frames, and which buffer the bitstream goes to. 648 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaEncPictureParameterBufferH264 {
    /// The picture to encode. **`CurrPic.picture_id` is the reconstructed (decoded) picture, not
    /// the input** (`va_enc_h264.h:302-303`): the driver is handed a *scratch* surface to
    /// reconstruct into, and `vaBeginPicture`'s surface is the input it reads. Putting the input
    /// surface here is the mistake this record's own comment exists to warn about.
    pub curr_pic: VaPictureH264,
    /// The decoded picture buffer the P and B slices predict from: reconstructed pictures, by
    /// surface, not by the indices a decoder would use.
    pub reference_frames: [VaPictureH264; 16],
    /// The [`ENC_CODED_BUFFER`] the compressed picture is written into.
    pub coded_buf: VABufferID,
    pub pic_parameter_set_id: u8,
    pub seq_parameter_set_id: u8,
    /// `H264_LAST_PICTURE_EOSEQ | H264_LAST_PICTURE_EOSTREAM`, or 0 for a picture in the middle.
    pub last_picture: u8,
    /// The C layout's hole at offset 619, before `frame_num`.
    pub padding_after_last_picture: [u8; 1],
    pub frame_num: u16,
    /// `pic_init_qp_minus26 + 26`, so the picture's quantiser *is* this number (26 is the offset,
    /// not a default).
    pub pic_init_qp: u8,
    pub num_ref_idx_l0_active_minus1: u8,
    pub num_ref_idx_l1_active_minus1: u8,
    pub chroma_qp_index_offset: i8,
    pub second_chroma_qp_index_offset: i8,
    /// The C layout's hole at offset 627, before `pic_fields`.
    pub padding_after_second_chroma_qp_index_offset: [u8; 1],
    /// `pic_fields`: the picture's own flags, packed into the union's `value` member in the bit
    /// order `va_enc_h264.h:358-384` writes them. `idr_pic_flag` is bit 0 and
    /// `reference_pic_flag` bits 1-2 — **bit 0 is the one that decides whether the picture is a
    /// key frame**, and leaving it at 0 for the first picture gives a stream no decoder can start
    /// from.
    pub pic_fields: u32,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAEncSliceParameterBufferH264` (`va_enc_h264.h:420`): one slice's record. **3140 bytes** — see
/// the note above this group for why that is not its decode counterpart's 3128.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaEncSliceParameterBufferH264 {
    /// First macroblock of the slice.
    pub macroblock_address: u32,
    /// How many macroblocks the slice covers. The whole picture is one slice in this backend's
    /// spike, so this is `picture_width_in_mbs * picture_height_in_mbs`.
    pub num_macroblocks: u32,
    /// Per-macroblock configuration, or `VA_INVALID_ID` to derive it from this slice's QP. The
    /// header is explicit that `VA_INVALID_ID` is the *right* value when no per-macroblock record
    /// is wanted (`va_enc_h264.h:426-428`), and iHD takes it.
    pub macroblock_info: VABufferID,
    /// [`SLICE_TYPE_I`] or [`SLICE_TYPE_P`]; see the constants for the offset-by-five encoding.
    pub slice_type: u8,
    pub pic_parameter_set_id: u8,
    /// Increments per IDR, and is what pairs two IDR pictures in the same sequence.
    pub idr_pic_id: u16,
    /// The picture order count, only meaningful when `seq_fields` says `pic_order_cnt_type == 0`.
    pub pic_order_cnt_lsb: u16,
    /// The C layout's hole at offsets 18-19, before `delta_pic_order_cnt_bottom`.
    pub padding_after_pic_order_cnt_lsb: [u8; 2],
    pub delta_pic_order_cnt_bottom: i32,
    /// Only for `pic_order_cnt_type == 1`; `[top, bottom]`.
    pub delta_pic_order_cnt: [i32; 2],
    pub direct_spatial_mv_pred_flag: u8,
    pub num_ref_idx_active_override_flag: u8,
    pub num_ref_idx_l0_active_minus1: u8,
    pub num_ref_idx_l1_active_minus1: u8,
    /// Reference picture list 0, by surface. At offset 36 — the encode record's own layout, not the
    /// decode record's 28.
    pub ref_pic_list0: [VaPictureH264; 32],
    /// Reference picture list 1, by surface, at offset 1188.
    pub ref_pic_list1: [VaPictureH264; 32],
    pub luma_log2_weight_denom: u8,
    pub chroma_log2_weight_denom: u8,
    pub luma_weight_l0_flag: u8,
    /// The C layout's hole at offset 2343, before `luma_weight_l0`.
    pub padding_after_luma_weight_l0_flag: [u8; 1],
    pub luma_weight_l0: [i16; 32],
    pub luma_offset_l0: [i16; 32],
    pub chroma_weight_l0_flag: u8,
    /// The C layout's hole at offset 2473, before `chroma_weight_l0`.
    pub padding_after_chroma_weight_l0_flag: [u8; 1],
    pub chroma_weight_l0: [[i16; 2]; 32],
    pub chroma_offset_l0: [[i16; 2]; 32],
    pub luma_weight_l1_flag: u8,
    /// The C layout's hole at offset 2731, before `luma_weight_l1`.
    pub padding_after_luma_weight_l1_flag: [u8; 1],
    pub luma_weight_l1: [i16; 32],
    pub luma_offset_l1: [i16; 32],
    pub chroma_weight_l1_flag: u8,
    /// The C layout's hole at offset 2861, before `chroma_weight_l1`.
    pub padding_after_chroma_weight_l1_flag: [u8; 1],
    pub chroma_weight_l1: [[i16; 2]; 32],
    pub chroma_offset_l1: [[i16; 2]; 32],
    pub cabac_init_idc: u8,
    /// Added to `pic_init_qp` to give this slice's quantiser. **0 means "the picture's QP"**, which
    /// is what a per-slice override record must say; a driver that added 1 here would encode the
    /// picture at a different quality than the one that was asked for.
    pub slice_qp_delta: i8,
    pub disable_deblocking_filter_idc: u8,
    pub slice_alpha_c0_offset_div2: i8,
    pub slice_beta_offset_div2: i8,
    /// The C layout's hole at offset 3123, before `va_reserved`.
    pub padding_after_slice_beta_offset_div2: [u8; 1],
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAEncMiscParameterBuffer` (`va.h:1973`): a type word followed by a **flexible array**, so it has
/// no fixed size of its own. 4 bytes of header; a record of type `t` is
/// `size_of::<VaEncMiscParameterBufferHeader>()` plus the bytes of `t`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaEncMiscParameterBufferHeader {
    /// One of `VAEncMiscParameterType*`: [`MISC_FRAME_RATE`], [`MISC_RATE_CONTROL`] or
    /// [`MISC_QUANTIZATION`].
    pub type_: u32,
}

/// `VAEncMiscParameterFrameRate` (`va.h:2617`): the framerate the rate control is told about. 24
/// bytes. See [`MISC_FRAME_RATE`] for why iHD insists on one.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaEncMiscParameterFrameRate {
    /// **A fraction packed high/low, not a number** (`va.h:2621-2631`): the denominator is the top
    /// two bytes and the numerator the bottom two, so 30 fps is `(1 << 16) | 30`. A frame rate of
    /// `30` here is 30/0, which the driver reads as 30/1 = 30 — the same picture, by luck — and a
    /// frame rate of `30000` is 30000/0, which is 30000 fps and a rate control that never
    /// throttles. This is the kind of field that produces a stream nobody can play and no error.
    pub framerate: u32,
    /// `framerate_flags`: `temporal_id` in the low byte.
    pub framerate_flags: u32,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAEncMiscParameterQuantization` (`va.h:2805`): the encoder's quantiser controls. 8 bytes.
///
/// Under [`RC_CQP`] this is the record iHD takes the picture's quantiser from, alongside the
/// picture record's `pic_init_qp`; every flag left at 0 means "let the driver decide", which is the
/// honest state for a picture whose quality comes from `pic_init_qp` alone.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaEncMiscParameterQuantization {
    /// `quantization_flags`: the trellis bits, packed into the union's `value` member.
    pub quantization_flags: u32,
    pub va_reserved: u32,
}

// `VAEncMiscParameterRateControl` (`va.h:2493`) — the record a `VA_RC_CBR` / `VA_RC_VBR` / `VA_RC_ICQ`
// config needs, 60 bytes — is deliberately **not** declared here. This backend runs under
// `VA_RC_CQP`, where the quantiser is the picture's `pic_init_qp` and a bitrate record would be a
// second, independent statement about the picture's fidelity. It is Task 3's record, and it will be
// declared from the header with the rest then, next to the misc buffer that carries it. Declaring it
// now, unused, is how a reader ends up thinking a bitrate is a thing this backend sends.

// ---------------------------------------------------------------------------------------------
// Reading the records back out of the opaque bytes `Picture` carries. The tests of
// `picture.rs` are what reads them today, so the `cfg_attr` keeps them out of the dead-code lint
// without letting them go unchecked when the decode loop does read them.
// ---------------------------------------------------------------------------------------------

/// The record of type `T` in `bytes`, or `None` when the buffer is too short for one.
///
/// `pub(crate)` rather than private because the encoder's tests read the records its builders
/// produced back this way: that is the only way a bit position inside a record can be checked
/// without a driver, and a bit position that is wrong produces a picture rather than an error.
///
/// # Safety
/// `T` must be a `#[repr(C)]` record this module declares, so that any `size_of::<T>()` bytes are a
/// valid `T`. Every caller passes one of the structs below and nothing else; a short buffer is
/// refused rather than read past.
pub(crate) unsafe fn record_of<T: Default + Copy>(bytes: &[u8]) -> Option<T> {
    if bytes.len() < size_of::<T>() {
        return None;
    }
    let mut record = T::default();
    // SAFETY: `bytes` is at least `size_of::<T>()` long (checked above) and `record` is a live,
    // initialised `T`, so this copies exactly one `T`'s worth of bytes into it.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), std::ptr::from_mut(&mut record).cast::<u8>(), size_of::<T>()) };
    Some(record)
}

/// The bytes of a record: exactly the bytes the driver reads, at the same addresses they land at
/// when a buffer is made from them.
///
/// Safe because a record is a plain `#[repr(C)]` value whose every byte is a named field: the holes
/// the C layout has between fields are declared as `padding_*` fields here (see the note above this
/// file's structures), so `Default` initialises all of them and a record of one has no indeterminate
/// byte anywhere. `record` is borrowed for the copy and nothing is read back through a pointer.
pub fn record_bytes<T: Copy>(record: &T) -> Vec<u8> {
    // SAFETY: `record` is a live `T` borrowed for the length of this function and is at least
    // `size_of::<T>()` bytes of readable memory, so the slice stays inside it and outlives nothing;
    // `T: Copy` means the value owns nothing the bytes could outlive; and every byte of a `T` is an
    // initialised field — the records' alignment holes are named `padding_*` fields, which
    // `#[derive(Default)]` zeroes — so the `&[u8]` never covers uninitialised memory.
    unsafe { std::slice::from_raw_parts(std::ptr::from_ref(record).cast::<u8>(), size_of::<T>()) }.to_vec()
}

/// The record of type `T` in `bytes`, or `None` when the buffer is too short for one — the safe
/// form of [`record_of`], which takes the `#[repr(C)]` obligation on the caller's behalf.
///
/// This exists for the encode module's tests, which read back the records its builders produced. A
/// bit position inside a record cannot be verified any other way: the builder produces bytes, and
/// the driver reading them wrong produces a *picture*, not an error. So the round trip
/// `build → bytes → record → check the bit` is what catches it.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn record_in<T: Default + Copy>(bytes: &[u8]) -> Option<T> {
    // SAFETY: every caller in this crate passes one of the `#[repr(C)]` records declared in this
    // file, which is `record_of`'s whole contract, and the length check is inside it.
    unsafe { record_of::<T>(bytes) }
}

/// The H.264 picture parameter record `bytes` carries, or `None` if it is not one long enough.
#[cfg_attr(not(test), allow(dead_code))]
pub fn picture_params_h264(bytes: &[u8]) -> Option<VaPictureParameterBufferH264> {
    // SAFETY: `VaPictureParameterBufferH264` is a `#[repr(C)]` struct declared above.
    unsafe { record_of::<VaPictureParameterBufferH264>(bytes) }
}

/// The H.264 slice parameter record `bytes` carries, or `None` if it is not one long enough.
#[cfg_attr(not(test), allow(dead_code))]
pub fn slice_params_h264(bytes: &[u8]) -> Option<VaSliceParameterBufferH264> {
    // SAFETY: `VaSliceParameterBufferH264` is a `#[repr(C)]` struct declared above.
    unsafe { record_of::<VaSliceParameterBufferH264>(bytes) }
}

/// The H.264 IQ matrix record `bytes` carries, or `None` if it is not one long enough.
#[cfg_attr(not(test), allow(dead_code))]
pub fn iq_matrix_h264(bytes: &[u8]) -> Option<VaIqMatrixBufferH264> {
    // SAFETY: `VaIqMatrixBufferH264` is a `#[repr(C)]` struct declared above.
    unsafe { record_of::<VaIqMatrixBufferH264>(bytes) }
}

/// The HEVC picture parameter record `bytes` carries, or `None` if it is not one long enough.
#[cfg_attr(not(test), allow(dead_code))]
pub fn picture_params_hevc(bytes: &[u8]) -> Option<VaPictureParameterBufferHevc> {
    // SAFETY: `VaPictureParameterBufferHevc` is a `#[repr(C)]` struct declared above.
    unsafe { record_of::<VaPictureParameterBufferHevc>(bytes) }
}

/// The HEVC slice parameter record `bytes` carries, or `None` if it is not one long enough.
#[cfg_attr(not(test), allow(dead_code))]
pub fn slice_params_hevc(bytes: &[u8]) -> Option<VaSliceParameterBufferHevc> {
    // SAFETY: `VaSliceParameterBufferHevc` is a `#[repr(C)]` struct declared above.
    unsafe { record_of::<VaSliceParameterBufferHevc>(bytes) }
}

/// The VP9 picture parameter record `bytes` carries, or `None` if it is not one long enough.
#[cfg_attr(not(test), allow(dead_code))]
pub fn picture_params_vp9(bytes: &[u8]) -> Option<VaPictureParameterBufferVp9> {
    // SAFETY: `VaPictureParameterBufferVp9` is a `#[repr(C)]` struct declared above.
    unsafe { record_of::<VaPictureParameterBufferVp9>(bytes) }
}

/// The VP9 slice parameter record `bytes` carries, or `None` if it is not one long enough.
///
/// The counterpart of [`picture_params_vp9`] on the other side of the same submission: a VP9 decode
/// sends both, and a test that reads only the picture record back would not notice a slice record
/// that was never built at all — which is the shape of the bug this backend spent a milestone on.
#[cfg_attr(not(test), allow(dead_code))]
pub fn slice_params_vp9(bytes: &[u8]) -> Option<VaSliceParameterBufferVp9> {
    // SAFETY: `VaSliceParameterBufferVp9` is a `#[repr(C)]` struct declared above.
    unsafe { record_of::<VaSliceParameterBufferVp9>(bytes) }
}

// ---------------------------------------------------------------------------------------------
// What the driver reports, as types the rest of the backend can compare.
// ---------------------------------------------------------------------------------------------

/// A codec profile the driver's decoder supports, as something FilmCraft can name and compare. A
/// profile this backend does not decode is kept as [`Profile::Other`] rather than dropped, so a
/// driver's capability list is reported in full.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    H264ConstrainedBaseline,
    H264Main,
    H264High,
    HevcMain,
    HevcMain10,
    /// VP9 profile 0..=3 (`VAProfileVP9Profile0`..`3`), which differ in bit depth and colour space
    /// handling, so the number is kept.
    Vp9(u8),
    Av1Main,
    /// A profile libva or the driver knows and this backend does not name.
    Other(i32),
}

impl Profile {
    /// The profile as the `VAProfile` libva takes, which is what `entrypoints` asks about and what
    /// a config is created with.
    pub fn bits(self) -> VAProfile {
        match self {
            Self::H264ConstrainedBaseline => H264_CONSTRAINED_BASELINE,
            Self::H264Main => H264_MAIN,
            Self::H264High => H264_HIGH,
            Self::HevcMain => HEVC_MAIN,
            Self::HevcMain10 => HEVC_MAIN_10,
            // Beyond 3 libva numbers no VP9 profile, so the number stays a number rather
            // than silently naming a different profile's bits.
            Self::Vp9(n) => VP9_PROFILE_0 + i32::from(n),
            Self::Av1Main => AV1_MAIN,
            Self::Other(bits) => bits,
        }
    }

    /// The profile libva reported.
    fn from_bits(bits: VAProfile) -> Self {
        match bits {
            H264_CONSTRAINED_BASELINE => Self::H264ConstrainedBaseline,
            H264_MAIN => Self::H264Main,
            H264_HIGH => Self::H264High,
            HEVC_MAIN => Self::HevcMain,
            HEVC_MAIN_10 => Self::HevcMain10,
            VP9_PROFILE_0 => Self::Vp9(0),
            VP9_PROFILE_1 => Self::Vp9(1),
            VP9_PROFILE_2 => Self::Vp9(2),
            VP9_PROFILE_3 => Self::Vp9(3),
            AV1_MAIN => Self::Av1Main,
            other => Self::Other(other),
        }
    }
}

/// An entry point the driver offers for a profile. Only [`Entrypoint::Vld`] decodes; the others are
/// reported so a driver's entry points are not filtered on the way in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entrypoint {
    /// `VAEntrypointVLD`: the hardware video decoder.
    Vld,
    /// `VAEntrypointEncSlice`: the slice-level encoder. Listed, not used for decoding.
    EncSlice,
    /// An entry point this backend does not name.
    Other(i32),
}

impl Entrypoint {
    /// The entry point as the `VAEntrypoint` libva takes.
    pub fn bits(self) -> VAEntrypoint {
        match self {
            Self::Vld => VLD,
            Self::EncSlice => ENC_SLICE,
            Self::Other(bits) => bits,
        }
    }

    /// The entry point libva reported.
    fn from_bits(bits: VAEntrypoint) -> Self {
        match bits {
            VLD => Self::Vld,
            ENC_SLICE => Self::EncSlice,
            other => Self::Other(other),
        }
    }
}

/// A render format (`VA_RT_FORMAT_*`) a surface can be created in. This backend takes 8- and
/// 10-bit 4:2:0; anything else is [`RtFormat::Other`], which the decoder declines.
#[allow(non_camel_case_types)] // `Yuv420_10` is libva's `VA_RT_FORMAT_YUV420_10`, not a typo.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RtFormat {
    Nv12,
    Yuv420_10,
    /// A format this backend does not take.
    Other(u32),
}

impl RtFormat {
    /// The format as the `VA_RT_FORMAT_*` libva takes, which is what a surface is created in.
    pub fn bits(self) -> u32 {
        match self {
            Self::Nv12 => YUV420,
            Self::Yuv420_10 => YUV420_10,
            Self::Other(bits) => bits,
        }
    }
}

impl From<u32> for RtFormat {
    fn from(bits: u32) -> Self {
        match bits {
            YUV420 => Self::Nv12,
            YUV420_10 => Self::Yuv420_10,
            other => Self::Other(other),
        }
    }
}

/// libva's own words for a status code, or the number when libva is not loaded or does not know the
/// code: error messages name the status, and this is where the text comes from.
///
/// This opens libva if it is not open yet. Code that already has an [`Api`] in hand — everything on
/// a [`Display`] error path — uses [`Api::status_text`] instead, so that decorating an error can
/// never be the thing that loads a library (and so can never re-enter the one-time load).
pub fn status(s: i32) -> String {
    match load() {
        Ok(api) => api.status_text(s),
        Err(_) => format!("VA-API status {s}"),
    }
}

// ---------------------------------------------------------------------------------------------
// The two open libraries and the entry points in them.
// ---------------------------------------------------------------------------------------------

/// The two `dlopen` handles, shared by every [`Api`] that was made from them.
struct Libraries {
    va: *mut c_void,
    va_drm: *mut c_void,
}

impl Drop for Libraries {
    fn drop(&mut self) {
        // SAFETY: both handles came from `dlopen` in `load_uncached` and this is their only owner,
        // so each is closed exactly once. Every function pointer resolved out of them belongs to an
        // `Api`, which holds a reference to this `Libraries` and is therefore dropped before it:
        // nothing can jump into a closed library.
        unsafe {
            dlclose(self.va_drm);
            dlclose(self.va);
        }
    }
}

// SAFETY: two `dlopen` handles and nothing else. They are opened before any `Api` exists and
// closed only in `Drop`, on whichever thread happens to drop the last reference, which the loader
// allows.
unsafe impl Send for Libraries {}
// SAFETY: see above.
unsafe impl Sync for Libraries {}

/// The libva entry points this backend needs, resolved once and kept for as long as the process
/// decodes with VA-API.
#[derive(Clone)]
pub struct Api {
    /// Read only to keep the libraries open: this is what outlives every function pointer below.
    _libraries: Arc<Libraries>,
    initialize: InitializeFn,
    terminate: TerminateFn,
    get_display_drm: GetDisplayDrmFn,
    query_vendor_string: QueryVendorStringFn,
    max_profiles: MaxProfilesFn,
    query_profiles: QueryProfilesFn,
    max_entrypoints: MaxEntrypointsFn,
    query_entrypoints: QueryEntrypointsFn,
    /// `vaGetConfigAttributes` (`va.h:686`): what a driver supports for a profile and entry point,
    /// asked before any config exists. Optional because a libva without it can still encode — the
    /// encoder then works at the reference depth it is given and finds out at `vaEndPicture`.
    get_config_attributes: Option<GetConfigAttributesFn>,
    /// Optional: `vaErrorStr` is libva 1.0-era and only decorates error messages.
    error_str: Option<ErrorStrFn>,
    /// Optional: the encoder waits for the coded buffer this way (`va.h:5110`). libva's
    /// `vaMapBuffer` is *not* a synchronisation point for an encoder — an unsynchronised read of a
    /// coded buffer is a truncated picture, not a stale one — and a libva without it still maps,
    /// so it stays optional.
    sync_buffer: Option<SyncBufferFn>,
    /// Optional: the image calls are the ABI layout probe, the indirect readback path (see
    /// [`Session::read`]) and the encoder's picture upload (see [`Encoder::upload`]).
    create_image: Option<CreateImageFn>,
    destroy_image: Option<DestroyImageFn>,
    put_image: Option<PutImageFn>,
    create_config: CreateConfigFn,
    destroy_config: DestroyConfigFn,
    create_surfaces: CreateSurfacesFn,
    destroy_surfaces: DestroySurfacesFn,
    create_context: CreateContextFn,
    destroy_context: DestroyContextFn,
    create_buffer: CreateBufferFn,
    destroy_buffer: DestroyBufferFn,
    acquire_buffer_handle: AcquireBufferHandleFn,
    release_buffer_handle: ReleaseBufferHandleFn,
    derive_image: DeriveImageFn,
    begin_picture: BeginPictureFn,
    render_picture: RenderPictureFn,
    end_picture: EndPictureFn,
    sync_surface: SyncSurfaceFn,
    query_surface_status: QuerySurfaceStatusFn,
    map_buffer: MapBufferFn,
    unmap_buffer: UnmapBufferFn,
}

// SAFETY: the function pointers are resolved once and are not mutated afterwards, and the
// libraries they point into stay open for as long as any `Api` exists (see `Libraries`).
unsafe impl Send for Api {}
// SAFETY: see above.
unsafe impl Sync for Api {}

/// The libva library and its DRM helper, opened and resolved once per process; a machine without
/// them gets the error once and never again.
static API: OnceLock<Result<Arc<Api>, String>> = OnceLock::new();

/// The libva entry points, opening `libva.so.2` and `libva-drm.so.2` on first use.
///
/// The `Arc` is kept for as long as the process uses VA-API, which is what keeps the libraries
/// open: [`Display`] takes a reference, so a function pointer can never outlive its library. A
/// failure is remembered rather than retried (probing a missing library on every clip would open
/// and close it thousands of times).
pub fn load() -> Result<Arc<Api>, String> {
    API.get_or_init(|| load_uncached().map(Arc::new)).clone()
}

fn load_uncached() -> Result<Api, String> {
    // SAFETY: opens the named library; a null handle is the documented failure and is reported.
    let va = unsafe { dlopen(LIBVA.as_ptr().cast(), RTLD_NOW | RTLD_LOCAL) };
    if va.is_null() {
        return Err(format!("libva.so.2 is not installed ({})", dl_error()));
    }
    // SAFETY: as above, for the DRM helper.
    let va_drm = unsafe { dlopen(LIBVA_DRM.as_ptr().cast(), RTLD_NOW | RTLD_LOCAL) };
    if va_drm.is_null() {
        // SAFETY: `va` was just opened here and is closed exactly once, on this error path.
        unsafe { dlclose(va) };
        return Err(format!("libva-drm.so.2 is not installed ({})", dl_error()));
    }
    let libraries = Arc::new(Libraries { va, va_drm });
    match resolve(va, va_drm, libraries) {
        Ok(api) => Ok(api),
        Err(why) => {
            log::info!("VA-API: {why}");
            Err(why)
        }
    }
}

/// Every symbol this backend calls, with the signature its header declares. The entry points the
/// decoder cannot work without are required; the rest are optional, and a missing optional symbol
/// is a capability this build simply does not have.
fn resolve(va: *mut c_void, va_drm: *mut c_void, libraries: Arc<Libraries>) -> Result<Api, String> {
    // SAFETY: every symbol below is looked up in a handle `load_uncached` just opened, with the
    // signature `/usr/include/va/va.h` (or `va_drm.h`) declares for it. All of them are part of
    // libva 2.x, so a missing one means the library is not what it claims to be.
    unsafe {
        Ok(Api {
            _libraries: libraries,
            initialize: symbol(va, c"vaInitialize")?,
            terminate: symbol(va, c"vaTerminate")?,
            get_display_drm: symbol(va_drm, c"vaGetDisplayDRM")?,
            query_vendor_string: symbol(va, c"vaQueryVendorString")?,
            max_profiles: symbol(va, c"vaMaxNumProfiles")?,
            query_profiles: symbol(va, c"vaQueryConfigProfiles")?,
            max_entrypoints: symbol(va, c"vaMaxNumEntrypoints")?,
            query_entrypoints: symbol(va, c"vaQueryConfigEntrypoints")?,
            get_config_attributes: optional_symbol(va, c"vaGetConfigAttributes"),
            error_str: optional_symbol(va, c"vaErrorStr"),
            // Not present on the libva this milestone was written against: `vaGetErrorName` does
            // not exist, so status text comes from `vaErrorStr` alone. Recorded because the name
            // is in a decade of blog posts and finding it absent costs an hour of dlsym archaeology.
            sync_buffer: optional_symbol(va, c"vaSyncBuffer"),
            create_image: optional_symbol(va, c"vaCreateImage"),
            destroy_image: optional_symbol(va, c"vaDestroyImage"),
            put_image: optional_symbol(va, c"vaPutImage"),
            create_config: symbol(va, c"vaCreateConfig")?,
            destroy_config: symbol(va, c"vaDestroyConfig")?,
            create_surfaces: symbol(va, c"vaCreateSurfaces")?,
            destroy_surfaces: symbol(va, c"vaDestroySurfaces")?,
            create_context: symbol(va, c"vaCreateContext")?,
            destroy_context: symbol(va, c"vaDestroyContext")?,
            create_buffer: symbol(va, c"vaCreateBuffer")?,
            destroy_buffer: symbol(va, c"vaDestroyBuffer")?,
            acquire_buffer_handle: symbol(va, c"vaAcquireBufferHandle")?,
            release_buffer_handle: symbol(va, c"vaReleaseBufferHandle")?,
            derive_image: symbol(va, c"vaDeriveImage")?,
            begin_picture: symbol(va, c"vaBeginPicture")?,
            render_picture: symbol(va, c"vaRenderPicture")?,
            end_picture: symbol(va, c"vaEndPicture")?,
            sync_surface: symbol(va, c"vaSyncSurface")?,
            query_surface_status: symbol(va, c"vaQuerySurfaceStatus")?,
            map_buffer: symbol(va, c"vaMapBuffer")?,
            unmap_buffer: symbol(va, c"vaUnmapBuffer")?,
        })
    }
}

/// What one libva entry point is declared as, and where that declaration came from.
///
/// The four fields are the guard against the bug this module had: a hand-written
/// `unsafe extern "C" fn` whose arity does not match libva's is undefined behaviour that compiles,
/// links, and — in an optimised build — returns plausible pictures. Nothing in the language can
/// catch it, because the *declaration* is the thing that is wrong, so the declaration has to be
/// written down somewhere an independent source can check: that is what `prototype` is.
#[cfg_attr(not(test), allow(dead_code))]
struct EntryPoint {
    /// The symbol `dlsym` is asked for.
    symbol: &'static str,
    /// `va.h:2470` and so on — the header and line the declaration was read from, so a failure names
    /// the place to go and look.
    header: &'static str,
    /// The declaration as the header spells it, normalised to `return(type, type, …)` with the
    /// parameter names dropped: `"VAStatus(VADisplay, VABufferID, uint64_t)"`.
    prototype: &'static str,
    /// How many parameters the declaration above has, **counted from the parameter list the function
    /// pointer type was built out of** — not written out, so it cannot disagree with the type.
    arity: usize,
}

/// Declares every libva entry point this backend resolves, once.
///
/// Each line produces the `unsafe extern "C" fn` pointer type the call sites use *and* an
/// [`EntryPoint`] whose `arity` is derived from the very parameter list that type was built from, so
/// the two cannot drift apart. `every_entry_point_matches_the_header` then re-reads the installed
/// headers and fails on any entry point whose `prototype` is not what the header now declares —
/// which is what would have caught `vaSyncBuffer` being declared with two parameters when libva's
/// third (`uint64_t timeout_ns`) was never written.
///
/// The C prototype is written out by hand next to the Rust type; the two are checked against the
/// *same* header, so neither can rot quietly, but the mapping from a Rust type to its C spelling is
/// a reviewer's job, not the compiler's.
macro_rules! va_entry_points {
    ($(
        $(#[$meta:meta])*
        $name:ident => $symbol:literal, $header:literal, $prototype:literal,
        fn($($arg:ty),* $(,)?) -> $ret:ty;
    )*) => {
        $(
            $(#[$meta])*
            type $name = unsafe extern "C" fn($($arg),*) -> $ret;
        )*
        #[cfg_attr(not(test), allow(dead_code))]
const ENTRY_POINTS: &[EntryPoint] = &[$(
            EntryPoint {
                symbol: $symbol,
                header: $header,
                prototype: $prototype,
                arity: [$(stringify!($arg)),*].len(),
            },
        )*];
    };
}

va_entry_points! {
    InitializeFn => "vaInitialize", "va.h:181", "VAStatus(VADisplay, int *, int *)",
        fn(VADisplay, *mut c_int, *mut c_int) -> VAStatus;
    TerminateFn => "vaTerminate", "va.h:186", "VAStatus(VADisplay)",
        fn(VADisplay) -> VAStatus;
    /// `va_drm.h:52`: turns a DRM file descriptor into a display. The one symbol that is **not** in
    /// `libva.so.2` — it is in `libva-drm.so.2`, which is why the loader opens two libraries.
    GetDisplayDrmFn => "vaGetDisplayDRM", "va_drm.h:52", "VADisplay(int)",
        fn(c_int) -> VADisplay;
    QueryVendorStringFn => "vaQueryVendorString", "va.h:191", "const char *(VADisplay)",
        fn(VADisplay) -> *const c_char;
    /// `va.h:1524`: "Get maximum number of profiles supported by the implementation". It returns an
    /// **`int`**, not a `VAStatus` — the two are the same width, so the ABI is the same either way,
    /// but the declaration here says what the header says, and a count read as a status is the kind
    /// of thing that reads as a bug in the wrong place.
    MaxProfilesFn => "vaMaxNumProfiles", "va.h:1524", "int(VADisplay)",
        fn(VADisplay) -> c_int;
    MaxEntrypointsFn => "vaMaxNumEntrypoints", "va.h:1529", "int(VADisplay)",
        fn(VADisplay) -> c_int;
    QueryProfilesFn => "vaQueryConfigProfiles", "va.h:1538", "VAStatus(VADisplay, VAProfile *, int *)",
        fn(VADisplay, *mut VAProfile, *mut c_int) -> VAStatus;
    QueryEntrypointsFn => "vaQueryConfigEntrypoints", "va.h:1546", "VAStatus(VADisplay, VAProfile, VAEntrypoint *, int *)",
        fn(VADisplay, VAProfile, *mut VAEntrypoint, *mut c_int) -> VAStatus;
    ErrorStrFn => "vaErrorStr", "va.h:403", "const char *(VAStatus)",
        fn(VAStatus) -> *const c_char;
    GetConfigAttributesFn => "vaGetConfigAttributes", "va.h:1571", "VAStatus(VADisplay, VAProfile, VAEntrypoint, VAConfigAttrib *, int)",
        fn(VADisplay, VAProfile, VAEntrypoint, *mut VaConfigAttrib, c_int) -> VAStatus;
    CreateImageFn => "vaCreateImage", "va.h:2670", "VAStatus(VADisplay, VAImageFormat *, int, int, VAImage *)",
        fn(VADisplay, *mut VaImageFormat, c_int, c_int, *mut VaImage) -> VAStatus;
    DestroyImageFn => "vaDestroyImage", "va.h:2676", "VAStatus(VADisplay, VAImageID)",
        fn(VADisplay, VAImageID) -> VAStatus;
    PutImageFn => "vaPutImage", "va.h:2709", "VAStatus(VADisplay, VASurfaceID, VAImageID, int, int, unsigned int, unsigned int, int, int, unsigned int, unsigned int)",
        fn(VADisplay, VASurfaceID, VAImageID, c_int, c_int, c_uint, c_uint, c_int, c_int, c_uint, c_uint) -> VAStatus;
    CreateConfigFn => "vaCreateConfig", "va.h:1589", "VAStatus(VADisplay, VAProfile, VAEntrypoint, VAConfigAttrib *, int, VAConfigID *)",
        fn(VADisplay, VAProfile, VAEntrypoint, *mut VaConfigAttrib, c_int, *mut VAConfigID) -> VAStatus;
    DestroyConfigFn => "vaDestroyConfig", "va.h:1601", "VAStatus(VADisplay, VAConfigID)",
        fn(VADisplay, VAConfigID) -> VAStatus;
    CreateSurfacesFn => "vaCreateSurfaces", "va.h:972", "VAStatus(VADisplay, unsigned int, unsigned int, unsigned int, VASurfaceID *, unsigned int, VASurfaceAttrib *, unsigned int)",
        fn(VADisplay, c_uint, c_uint, c_uint, *mut VASurfaceID, c_uint, *mut VaSurfaceAttrib, c_uint) -> VAStatus;
    DestroySurfacesFn => "vaDestroySurfaces", "va.h:1914", "VAStatus(VADisplay, VASurfaceID *, int)",
        fn(VADisplay, *mut VASurfaceID, c_int) -> VAStatus;
    CreateContextFn => "vaCreateContext", "va.h:1933", "VAStatus(VADisplay, VAConfigID, int, int, int, VASurfaceID *, int, VAContextID *)",
        fn(VADisplay, VAConfigID, c_int, c_int, c_int, *mut VASurfaceID, c_int, *mut VAContextID) -> VAStatus;
    DestroyContextFn => "vaDestroyContext", "va.h:1949", "VAStatus(VADisplay, VAContextID)",
        fn(VADisplay, VAContextID) -> VAStatus;
    CreateBufferFn => "vaCreateBuffer", "va.h:2240", "VAStatus(VADisplay, VAContextID, VABufferType, unsigned int, unsigned int, void *, VABufferID *)",
        fn(VADisplay, VAContextID, VaBufferType, c_uint, c_uint, *mut c_void, *mut VABufferID) -> VAStatus;
    DestroyBufferFn => "vaDestroyBuffer", "va.h:4023", "VAStatus(VADisplay, VABufferID)",
        fn(VADisplay, VABufferID) -> VAStatus;
    /// `va.h:2353`: fills in a [`VaBufferInfo`] naming the buffer's underlying storage.
    AcquireBufferHandleFn => "vaAcquireBufferHandle", "va.h:2353", "VAStatus(VADisplay, VABufferID, VABufferInfo *)",
        fn(VADisplay, VABufferID, *mut VaBufferInfo) -> VAStatus;
    /// `va.h:2357`: releases what [`AcquireBufferHandleFn`] locked.
    ReleaseBufferHandleFn => "vaReleaseBufferHandle", "va.h:2357", "VAStatus(VADisplay, VABufferID)",
        fn(VADisplay, VABufferID) -> VAStatus;
    DeriveImageFn => "vaDeriveImage", "va.h:4893", "VAStatus(VADisplay, VASurfaceID, VAImage *)",
        fn(VADisplay, VASurfaceID, *mut VaImage) -> VAStatus;
    BeginPictureFn => "vaBeginPicture", "va.h:4210", "VAStatus(VADisplay, VAContextID, VASurfaceID)",
        fn(VADisplay, VAContextID, VASurfaceID) -> VAStatus;
    RenderPictureFn => "vaRenderPicture", "va.h:4219", "VAStatus(VADisplay, VAContextID, VABufferID *, int)",
        fn(VADisplay, VAContextID, *mut VABufferID, c_int) -> VAStatus;
    EndPictureFn => "vaEndPicture", "va.h:4234", "VAStatus(VADisplay, VAContextID)",
        fn(VADisplay, VAContextID) -> VAStatus;
    SyncSurfaceFn => "vaSyncSurface", "va.h:4276", "VAStatus(VADisplay, VASurfaceID)",
        fn(VADisplay, VASurfaceID) -> VAStatus;
    QuerySurfaceStatusFn => "vaQuerySurfaceStatus", "va.h:4322", "VAStatus(VADisplay, VASurfaceID, VASurfaceStatus *)",
        fn(VADisplay, VASurfaceID, *mut c_uint) -> VAStatus;
    /// `vaSyncBuffer`'s third parameter (`va.h:4391`) is `uint64_t timeout_ns`. It was declared here
    /// as `c_ulong` and then **omitted from the call**, which is undefined behaviour: the driver
    /// read an argument slot the call never wrote, the status came back `VA_STATUS_ERROR_TIMEDOUT`
    /// in a debug build and success in a release one, and nothing else was wrong. `u64` rather than
    /// `c_ulong` because `uint64_t` is `unsigned long long` on a 32-bit target and `c_ulong` is not.
    SyncBufferFn => "vaSyncBuffer", "va.h:4389", "VAStatus(VADisplay, VABufferID, uint64_t)",
        fn(VADisplay, VABufferID, u64) -> VAStatus;
    MapBufferFn => "vaMapBuffer", "va.h:3976", "VAStatus(VADisplay, VABufferID, void **)",
        fn(VADisplay, VABufferID, *mut *mut c_void) -> VAStatus;
    UnmapBufferFn => "vaUnmapBuffer", "va.h:4009", "VAStatus(VADisplay, VABufferID)",
        fn(VADisplay, VABufferID) -> VAStatus;
}

/// The exported symbol `name` of the open library `handle` as a function pointer of type `F`.
///
/// # Safety
/// `F` must be the `extern "C" fn` type `name` really has in that library.
unsafe fn symbol<F: Copy>(handle: *mut c_void, name: &CStr) -> Result<F, String> {
    if size_of::<F>() != size_of::<usize>() {
        return Err("not a function pointer type".into());
    }
    // SAFETY: `handle` is an open `dlopen` handle and `name` a NUL-terminated symbol name.
    let p = unsafe { dlsym(handle, name.as_ptr()) };
    if p.is_null() {
        return Err(format!("libva does not export {}", name.to_string_lossy()));
    }
    // SAFETY: `F` is pointer-sized (checked above) and, by this function's contract, the type of
    // the function `p` points to.
    Ok(unsafe { std::mem::transmute_copy::<_, F>(&p) })
}

/// [`symbol`], for a symbol a driver may legitimately not have.
unsafe fn optional_symbol<F: Copy>(handle: *mut c_void, name: &CStr) -> Option<F> {
    // SAFETY: the caller of this function upholds the contract of `symbol` for the same `F`.
    unsafe { symbol(handle, name) }.ok()
}

/// What the last `dl*` call that failed said, for an error message.
fn dl_error() -> String {
    // SAFETY: `dlerror` returns either NULL or a NUL-terminated string owned by the loader that is
    // valid until the next `dlerror` call, and copies it out before making another.
    let p = unsafe { dlerror() };
    if p.is_null() { "unknown error".to_string() } else { string_from(p) }
}

/// A NUL-terminated C string as owned Rust text, at most [`MAX_DRIVER_STRING`] bytes: a driver
/// string that is not terminated would make the length unbounded, and one that is enormous has
/// nothing useful in it.
fn string_from(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: libva documents the return of `vaQueryVendorString` and `vaErrorStr` as a
    // NUL-terminated string (`va.h:403,486`), owned by the driver, and only reads from it here.
    let bytes = unsafe { CStr::from_ptr(p) }.to_bytes();
    // Lossy, so a truncated or non-UTF-8 driver string is still a valid `str` and never panics.
    String::from_utf8_lossy(bytes.get(..MAX_DRIVER_STRING).unwrap_or(bytes)).into_owned()
}

impl Api {
    /// libva's words for a status code, or the number when it has none for it.
    ///
    /// This is the [`status`] work with the library already in hand, which is what a caller that
    /// has an `Api` should use: it cannot load anything, so it cannot re-enter the one-time load.
    pub(crate) fn status_text(&self, s: i32) -> String {
        let Some(error_str) = self.error_str else { return format!("VA-API status {s}") };
        // SAFETY: `error_str` is `vaErrorStr` (`va.h:403`), which takes a status and returns a
        // NUL-terminated string (or null for a code it does not know) and does nothing else.
        let text = string_from(unsafe { error_str(s) });
        if text.is_empty() { format!("VA-API status {s}") } else { text }
    }
}

// ---------------------------------------------------------------------------------------------
// The display: an initialised VA-API connection to the GPU's video engine.
// ---------------------------------------------------------------------------------------------

/// An initialised VA-API display — what this backend decodes with, and the only place a libva
/// handle exists. Dropping it terminates the connection and closes the DRM node.
pub struct Display {
    api: Arc<Api>,
    /// The DRM render node, closed in `Drop` (`None` once it is).
    drm: Option<c_int>,
    dpy: VADisplay,
    version: (c_int, c_int),
    /// The driver's self-description, copied out of libva once when the display is opened.
    vendor: Option<&'static str>,
    /// Serialises our use of the display: libva does not document its entry points as safe to call
    /// concurrently on one display, and a decode drives them in sequence anyway.
    gate: std::sync::Mutex<()>,
}

// SAFETY: a `Display` is a libva handle, the DRM descriptor libva keeps using behind it, and the
// version and vendor string we read at construction. Sending one to the decoding thread is the
// point; concurrent calls into libva are serialised by `gate`, and `Drop` (which runs `vaTerminate`
// and closes the descriptor) can only run once the last `Arc` is gone, so no other reference is
// still in use.
unsafe impl Send for Display {}
// SAFETY: see above.
unsafe impl Sync for Display {}

impl Display {
    /// Open the DRM render node and initialise libva on it.
    ///
    /// `vaInitialize` is mandatory before anything else is asked of a display, so it happens here
    /// and its status is checked: a driver that refuses to initialise leaves FilmCraft with its own
    /// decoders rather than a half-open display. Every failure says which step it was.
    pub fn new(api: &Api) -> Result<Arc<Display>, String> {
        let fd = open_drm_node()?;
        // SAFETY: `fd` is an open DRM descriptor, and `get_display_drm` is `vaGetDisplayDRM`
        // (`va_drm.h:52`), which returns the display derived from the descriptor, or null on
        // failure. libva keeps using that descriptor for the life of the display, so ours stays
        // open until after `vaTerminate` in `Drop` — it is closed on every path that does not reach
        // a `Display`.
        let dpy = unsafe { (api.get_display_drm)(fd) };
        if dpy.is_null() {
            close_fd(fd);
            return Err("vaGetDisplayDRM returned no display".to_string());
        }
        let (mut major, mut minor) = (0 as c_int, 0 as c_int);
        // SAFETY: `dpy` came from `vaGetDisplayDRM` above and both out-pointers are live locals.
        // No other libva entry point has been called on `dpy`, which is what this call requires.
        let st = unsafe { (api.initialize)(dpy, &mut major, &mut minor) };
        if st != OK {
            // Nothing to terminate: the display was never initialised.
            close_fd(fd);
            let why = format!("vaInitialize failed: {}", api.status_text(st));
            log::info!("VA-API: {why}");
            return Err(why);
        }
        // libva returns the driver's own string, which lives as long as the library; it is copied
        // out here and leaked, so `vendor` can name it for as long as the display exists. A driver
        // with no vendor string returns null, which is `None` rather than an empty name. The leak
        // is a few dozen bytes once per display (there is one display per process).
        // SAFETY: `query_vendor_string` is `vaQueryVendorString` (`va.h:486`), which returns a
        // NUL-terminated string owned by the driver or null, and only reads from it here.
        let vendor = string_from(unsafe { (api.query_vendor_string)(dpy) });
        Ok(Arc::new(Display {
            // The display keeps its own handle on the libraries, so a function pointer below can
            // never outlive them.
            api: Arc::new(api.clone()),
            drm: Some(fd),
            dpy,
            version: (major, minor),
            vendor: (!vendor.is_empty()).then(|| vendor.leak() as &'static str),
            gate: std::sync::Mutex::new(()),
        }))
    }

    /// Our use of the display, serialised. A poisoned lock is not a failure here — the lock
    /// guards no state libva could have torn — so it is recovered rather than propagated.
    fn gate(&self) -> MutexGuard<'_, ()> {
        self.gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The libva version this driver implements, as `(major, minor)` from `vaInitialize`.
    pub fn va_version(&self) -> (i32, i32) {
        self.version
    }

    /// The driver's own description of itself ("Intel iHD"), or `None` when it has none.
    pub fn vendor(&self) -> Option<&'static str> {
        self.vendor
    }

    /// The DRM render node this display was opened on, or `None` once it has been closed.
    ///
    /// [`crate::vaapi::drm`] maps a buffer's storage on this descriptor, which is what lets the
    /// encode readback read the picture out of storage this backend chose rather than out of a
    /// pointer the driver handed over. It is the *same* descriptor libva is using — libva was given
    /// it by [`Display::new`] and keeps it for the life of the display — so a buffer object libva
    /// names is one this descriptor can be asked to map.
    pub fn drm_fd(&self) -> Option<c_int> {
        self.drm
    }

    /// Every profile the driver's decoder can be configured for. Unknown profiles are reported as
    /// [`Profile::Other`] rather than dropped.
    pub fn profiles(&self) -> Result<Vec<Profile>, String> {
        let _gate = self.gate();
        // SAFETY: the display is initialised; the out-pointer refers to the array just allocated,
        // which is `capacity()` entries long as libva's own `vaMaxNumProfiles` asked.
        let capacity = capacity(unsafe { (self.api.max_profiles)(self.dpy) });
        let mut profiles = vec![0 as VAProfile; capacity];
        let mut reported: c_int = 0;
        // SAFETY: as above, plus the count out-pointer, which is a live local.
        let st = unsafe { (self.api.query_profiles)(self.dpy, profiles.as_mut_ptr(), &mut reported) };
        if st != OK {
            return Err(format!("vaQueryConfigProfiles failed: {}", self.api.status_text(st)));
        }
        // The count is the driver's to write, so it is bounded by what was allocated before it is
        // used as a length.
        let count = usize::try_from(reported).unwrap_or(0).min(profiles.len());
        Ok(profiles.into_iter().take(count).map(Profile::from_bits).collect())
    }

    /// Every entry point the driver offers for `p` — [`Entrypoint::Vld`] is the one a decode uses.
    pub fn entrypoints(&self, p: Profile) -> Result<Vec<Entrypoint>, String> {
        let _gate = self.gate();
        // SAFETY: the display is initialised, and the array and count out-pointers below refer to
        // live memory `capacity()` long as libva's own `vaMaxNumEntrypoints` asked.
        let capacity = capacity(unsafe { (self.api.max_entrypoints)(self.dpy) });
        let mut entrypoints = vec![0 as VAEntrypoint; capacity];
        let mut reported: c_int = 0;
        // SAFETY: as above, plus the profile, which is a plain integer libva compares.
        let st = unsafe { (self.api.query_entrypoints)(self.dpy, p.bits(), entrypoints.as_mut_ptr(), &mut reported) };
        if st != OK {
            return Err(format!("vaQueryConfigEntrypoints failed: {}", self.api.status_text(st)));
        }
        let count = usize::try_from(reported).unwrap_or(0).min(entrypoints.len());
        Ok(entrypoints.into_iter().take(count).map(Entrypoint::from_bits).collect())
    }

    /// What the driver says about a configuration, for the attributes in `types` — or why it will
    /// not say.
    ///
    /// This is the read-side counterpart of `vaCreateConfig`: an attribute the driver does not
    /// support comes back with its bit **clear**, and libva does not tell a caller which bit meant
    /// "supported" and which meant "I have never heard of this". So a caller cannot read a
    /// read-only attribute off a config it did not create, and `vaGetConfigAttributes` — which asks
    /// about a profile and entry point *before* any config exists — is how a caller asks what an
    /// encoder may do at all.
    ///
    /// Each returned value is what the driver wrote back into the record, so the caller compares it
    /// against the bit it cares about rather than assuming the attribute was honoured.
    pub fn config_attributes(&self, profile: Profile, entrypoint: Entrypoint, types: &[VaBufferType]) -> Result<Vec<u32>, String> {
        if types.is_empty() {
            return Ok(Vec::new());
        }
        // A libva without `vaGetConfigAttributes` cannot answer, and that is reported rather than
        // papered over: a caller that needs this to know whether an inter picture is possible gets
        // an error, not zeroes that look like "the driver supports nothing".
        let Some(get) = self.api.get_config_attributes else {
            return Err("this libva has no vaGetConfigAttributes, so it cannot say what the driver supports for a configuration".into());
        };
        let _gate = self.gate();
        // SAFETY: the display is initialised; `attribs` is a live array of `types.len()` records
        // initialised to zero, which is what libva expects to be handed, and its length is the count
        // passed, so libva writes exactly into `attribs` and reads no further.
        let values = unsafe {
            let mut attribs: Vec<VaConfigAttrib> = types.iter().map(|t| VaConfigAttrib { type_: *t, value: 0 }).collect();
            let st = get(self.dpy, profile.bits(), entrypoint.bits(), attribs.as_mut_ptr(), attribs.len() as c_int);
            if st != OK {
                return Err(format!("vaGetConfigAttributes failed: {}", self.api.status_text(st)));
            }
            attribs.into_iter().map(|a| a.value).collect()
        };
        Ok(values)
    }

    /// Whether the installed driver fills a `VAImage` the way this module declares it: libva has
    /// no `sizeof` / `offsetof` to ask, so the check is behavioural. The driver is asked to create
    /// a real NV12 image, and the plane geometry it writes back is read out of *this* struct: if a
    /// field were declared at the wrong offset, it would read a neighbour's value and the plane
    /// arithmetic would stop adding up. `false` means the probe could not be run (a driver that
    /// will not create an NV12 image, or a libva without `vaCreateImage`).
    pub fn probe_image_layout(&self) -> Result<bool, String> {
        let (Some(create_image), Some(destroy_image)) = (self.api.create_image, self.api.destroy_image) else {
            return Ok(false);
        };
        let _gate = self.gate();
        let (width, height) = (i32::from(PROBE_WIDTH), i32::from(PROBE_HEIGHT));
        let mut format = VaImageFormat { fourcc: FOURCC_NV12, byte_order: BYTE_ORDER_LSB_FIRST, bits_per_pixel: 12, ..Default::default() };
        let mut image = VaImage::default();
        // SAFETY: the display is initialised, `format` and `image` are live locals, and the sizes
        // are the ones libva is asked to allocate for.
        let st = unsafe { create_image(self.dpy, &mut format, width, height, &mut image) };
        if st != OK {
            return Err(format!("vaCreateImage failed for the ABI probe: {}", self.api.status_text(st)));
        }
        // SAFETY: `image` is the image libva just created, and this is its matching destroy; the
        // id is the one the driver wrote above.
        unsafe { destroy_image(self.dpy, image.image_id) };
        Ok(plane_geometry_is_consistent(&image, usize::from(PROBE_WIDTH), usize::from(PROBE_HEIGHT)))
    }
}

impl Drop for Display {
    fn drop(&mut self) {
        // SAFETY: `dpy` came from `vaGetDisplayDRM` and `vaInitialize` succeeded on it, so this is
        // the matching `vaTerminate`, before the descriptor it was opened from goes away.
        unsafe { (self.api.terminate)(self.dpy) };
        if let Some(fd) = self.drm.take() {
            close_fd(fd);
        }
    }
}

/// Whether a driver-filled `VAImage` describes an NV12 image of `width` x `height` in the planes it
/// claims to have. Every number in it is the driver's, so nothing is indexed or added without a
/// bound: an image whose planes do not add up to its `data_size` is one whose fields were read from
/// somewhere other than where libva wrote them.
fn plane_geometry_is_consistent(image: &VaImage, width: usize, height: usize) -> bool {
    if usize::from(image.width) != width || usize::from(image.height) != height {
        return false;
    }
    let reported = image.num_planes;
    if !(1..=3).contains(&reported) {
        return false;
    }
    let planes = usize::try_from(reported).unwrap_or(0);
    (0..planes).all(|plane| {
        let (Some(&pitch), Some(&offset)) = (image.pitches.get(plane), image.offsets.get(plane)) else { return false };
        let Some(row_bytes) = (pitch as usize).checked_mul(height) else { return false };
        // A plane is at least a row per pixel row, and ends inside the buffer the driver sized.
        pitch as usize >= width && (offset as usize).checked_add(row_bytes).is_some_and(|end| end <= image.data_size as usize)
    })
}

/// How many entries to allocate for a driver query, from the `vaMaxNum*` answer: bounded, because
/// a driver's answer sizes the array we hand it, and never zero, so there is always somewhere for
/// the driver to write.
///
/// The parameter is a plain `int` because that is what `vaMaxNumProfiles` and `vaMaxNumEntrypoints`
/// return (`va.h:1524`, `va.h:1529`) — a count, not a status. A negative one is treated as "cannot
/// tell" rather than as an allocation of nothing.
fn capacity(reported: c_int) -> usize {
    usize::try_from(reported).unwrap_or(MAX_DRIVER_ENTRIES).clamp(1, MAX_DRIVER_ENTRIES)
}

/// The DRM render node to open: `FILMCRAFT_VAAPI_DRM_NODE` when it is set (see
/// [`DRM_NODE_ENV`]), else [`DRM_NODE`].
fn drm_node() -> String {
    match std::env::var(DRM_NODE_ENV) {
        Ok(path) if !path.is_empty() => path,
        _ => DRM_NODE.to_string(),
    }
}

/// Open the DRM render node. A node that is not there, or cannot be opened, is a normal, silent
/// "no hardware here" and not an error worth logging.
fn open_drm_node() -> Result<c_int, String> {
    let path = drm_node();
    let Ok(c_path) = CString::new(path.as_str()) else { return Err(format!("{DRM_NODE_ENV} is not a usable path: {path:?}")) };
    // SAFETY: `c_path` is a NUL-terminated path that outlives the call, and `open` copies it.
    let fd = unsafe { open(c_path.as_ptr(), O_RDWR) };
    if fd < 0 {
        log::debug!("VA-API: no DRM render node at {path}");
        return Err(format!("no DRM render node at {path}"));
    }
    Ok(fd)
}

/// Close a descriptor from [`open_drm_node`], ignoring the result: the caller is already reporting
/// a failure, and a failed `close` must not replace it with a different one.
fn close_fd(fd: c_int) {
    // SAFETY: `fd` is an open descriptor this module opened, and this closes it exactly once.
    unsafe { close(fd) };
}

// ---------------------------------------------------------------------------------------------
// The decode session: a config, a ring of surfaces, a decode context, and the picture readback
// that gets pixels back out of a surface. Everything that needs a libva handle lives behind these
// six types, so the decoder itself never names one and every pairing that has to happen on every
// path — buffer and image destroyed, buffer unmapped — is a `Drop`, not a line someone has to
// remember on the error return.
// ---------------------------------------------------------------------------------------------

/// A configured decode session: what [`crate::vaapi::VaDecoder`] decodes one stream with.
///
/// It owns the three libva objects a decoder needs, in the order libva requires them to be torn
/// down: the config, the surfaces (which cannot be destroyed while a context uses them) and the
/// context. Dropping it releases all three, so a decode that fails half-way cannot leave a context
/// or a surface behind.
///
/// `vaSyncSurface` is mandatory before a surface's pixels are read, and [`Session::sync`] is it:
/// rendering is asynchronous, and a readback that does not sync reads whatever the driver had
/// finished writing — which is a plausible-looking black picture rather than an error.
pub struct Session {
    display: Arc<Display>,
    config: VAConfigID,
    surfaces: Vec<u32>,
    /// `None` after [`Session::clear_context`], which is what a seek does: no reference picture
    /// survives it, so the next picture makes a fresh context.
    context: Option<VAContextID>,
    /// The four-character code and bit depth a decoded picture is read back as.
    fourcc: u32,
    bpp: u32,
    size: (u32, u32),
    /// Whether this session's surfaces are always read back the indirect way, remembered so that a
    /// driver that cannot derive reports it once rather than on every picture.
    indirect: bool,
}

impl Session {
    /// Configure `display` for `profile` at `entrypoint`, create `count` surfaces of `format` at
    /// `size`, and make a decode context for them.
    ///
    /// Every failure names the step and the driver's own status text, because "hardware decoding
    /// declined" is only actionable if it says which of the three steps the driver refused. `size`
    /// is the *coded* picture size (the surfaces hold the padding too) and `count` is the number of
    /// surfaces the stream's decoded picture buffer needs; both come from the plan and the stream
    /// description, which bound them.
    pub fn new(display: &Arc<Display>, profile: Profile, entrypoint: Entrypoint, format: RtFormat, size: (u32, u32), count: usize) -> Result<Self, String> {
        let count = count.clamp(1, MAX_SURFACES);
        let count_u = c_uint::try_from(count).map_err(|_| format!("{count} surfaces"))?;
        let width = c_uint::try_from(size.0).map_err(|_| format!("surface width {}", size.0))?;
        let height = c_uint::try_from(size.1).map_err(|_| format!("surface height {}", size.1))?;
        let context_width = c_int::try_from(size.0).map_err(|_| format!("picture width {}", size.0))?;
        let context_height = c_int::try_from(size.1).map_err(|_| format!("picture height {}", size.1))?;
        let (fourcc, bpp) = match format {
            RtFormat::Nv12 => (NV12_FOURCC, NV12_BPP),
            RtFormat::Yuv420_10 => (P010_FOURCC, P010_BPP),
            RtFormat::Other(bits) => return Err(format!("render format {bits:#x} is not 4:2:0 8- or 10-bit")),
        };
        let api = Arc::clone(&display.api);
        let mut session = Self { display: Arc::clone(display), config: 0, surfaces: Vec::new(), context: None, fourcc, bpp, size, indirect: false };

        let mut surfaces = vec![0 as VASurfaceID; count];
        {
            let _gate = display.gate();
            let mut config: VAConfigID = 0;
            // SAFETY: the display is initialised; no attributes are passed (null, zero), which
            // libva reads as "every attribute at its default"; and `config` is a live local the
            // call writes the config id into.
            let st = unsafe { (api.create_config)(display.dpy, profile.bits(), entrypoint.bits(), std::ptr::null_mut(), 0, &mut config) };
            if st != OK {
                return Err(format!("vaCreateConfig failed for {:?}/{:?}: {}", profile, entrypoint, api.status_text(st)));
            }
            session.config = config;
            // SAFETY: as above for the surfaces: `surfaces` is a live array `count_u` long, which is
            // what the driver is told to fill, and no attribute list is passed. A `Session` that is
            // dropped from here on destroys the config it holds, so this needs no manual unwind.
            let st = unsafe { (api.create_surfaces)(display.dpy, format.bits(), width, height, surfaces.as_mut_ptr(), count_u, std::ptr::null_mut(), 0) };
            if st != OK {
                return Err(format!("vaCreateSurfaces failed for {} surfaces at {}x{}: {}", count, size.0, size.1, api.status_text(st)));
            }
            session.surfaces = std::mem::take(&mut surfaces);
            let Ok(contexts) = c_int::try_from(session.surfaces.len()) else {
                return Err(format!("{count} surfaces does not fit a context's render target list"));
            };
            let mut context: VAContextID = 0;
            // SAFETY: `config` is the config created above, `surfaces` the array just filled and
            // `contexts` its length, and `context` is a live local for the id to be written into.
            let st = unsafe {
                (api.create_context)(display.dpy, config, context_width, context_height, PROGRESSIVE, session.surfaces.as_mut_ptr(), contexts, &mut context)
            };
            if st != OK {
                return Err(format!("vaCreateContext failed at {}x{}: {}", size.0, size.1, api.status_text(st)));
            }
            session.context = Some(context);
        }
        Ok(session)
    }

    /// The surfaces pictures are decoded into, in the order the driver gave them.
    pub fn surfaces(&self) -> &[u32] {
        &self.surfaces
    }

    /// The coded picture size the surfaces and the context were created at.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Whether this session's pictures have to be read back the indirect way.
    pub fn reads_indirectly(&self) -> bool {
        self.indirect
    }

    /// Drop the decode context, so that the next picture makes a new one. A seek does this: no
    /// reference picture survives it, and a context that still held the previous run's references
    /// would decode the new run against surfaces that hold something else.
    pub fn clear_context(&mut self) {
        if let Some(context) = self.context.take() {
            let api = &self.display.api;
            // SAFETY: `context` is this session's own context, taken out of the option so no other
            // reference to it exists, and this is its matching destroy.
            unsafe { (api.destroy_context)(self.display.dpy, context) };
        }
    }

    /// Make a decode context again, for the surfaces this session already owns.
    fn make_context(&mut self) -> Result<VAContextID, String> {
        let api = &self.display.api;
        let width = c_int::try_from(self.size.0).map_err(|_| format!("picture width {}", self.size.0))?;
        let height = c_int::try_from(self.size.1).map_err(|_| format!("picture height {}", self.size.1))?;
        let Ok(count) = c_int::try_from(self.surfaces.len()) else { return Err(format!("{} surfaces", self.surfaces.len())) };
        let mut context: VAContextID = 0;
        // SAFETY: `config` is this session's own config, `surfaces` the array this session owns and
        // `count` its length, and `context` is a live local for the id to be written into.
        let st = unsafe { (api.create_context)(self.display.dpy, self.config, width, height, PROGRESSIVE, self.surfaces.as_mut_ptr(), count, &mut context) };
        if st != OK {
            return Err(format!("vaCreateContext failed at {}x{}: {}", self.size.0, self.size.1, api.status_text(st)));
        }
        Ok(context)
    }

    /// A buffer holding `data` of kind `kind` (a [`PICTURE_PARAMETER_BUFFER`],
    /// [`SLICE_PARAMETER_BUFFER`] or [`SLICE_DATA_BUFFER`]), destroyed when it is dropped.
    ///
    /// `size` is the byte length and `num_elements` is 1. That is what libva's own decode examples
    /// pass for all three kinds (`va.h:3838`: "size" and "num_elements", which for a decoder's
    /// parameter buffers is a single record rather than an array of them), and it is what the
    /// drivers read back with `vaBufferGetNumElements` — a driver asked for one element per byte
    /// refuses the buffer with `VA_STATUS_ERROR_INVALID_PARAMETER`, which is what an Intel iHD driver
    /// does for a 672-byte picture parameter buffer.
    ///
    /// libva copies the data in (`va.h:3826`), so the buffer owns its own copy of the record and the
    /// caller's bytes do not have to outlive the call.
    pub fn buffer(&mut self, kind: VAStatus, data: &[u8]) -> Result<Buffer, String> {
        let Ok(size) = c_uint::try_from(data.len()) else { return Err(format!("{} byte buffer", data.len())) };
        let mut id: VABufferID = 0;
        // SAFETY: the context exists (a buffer is only made for a decode in flight) and `data` is
        // `size` bytes of readable memory, which is the count passed below, so libva copies exactly
        // that many bytes from it. `id` is a live local for the buffer id.
        let st = unsafe { (self.display.api.create_buffer)(self.display.dpy, self.context_id()?, kind, size, 1, data.as_ptr().cast_mut().cast(), &mut id) };
        if st != OK {
            return Err(format!("vaCreateBuffer failed for {size} bytes of type {kind}: {}", self.display.api.status_text(st)));
        }
        Ok(Buffer::owned(Arc::clone(&self.display), id))
    }

    /// The id of the decode context, making one if a seek dropped it.
    fn context_id(&mut self) -> Result<VAContextID, String> {
        if let Some(context) = self.context {
            return Ok(context);
        }
        let context = self.make_context()?;
        self.context = Some(context);
        Ok(context)
    }

    /// Decode into `surface`: begin the picture, submit `buffers`, end it.
    ///
    /// `vaEndPicture` is called on every path that got as far as `vaBeginPicture`, so a
    /// `vaRenderPicture` that fails cannot leave the context mid-picture — which libva would refuse
    /// every later call on.
    pub fn render(&mut self, surface: u32, buffers: &[u32]) -> Result<(), String> {
        let api = Arc::clone(&self.display.api);
        let context = self.context_id()?;
        if buffers.is_empty() {
            return Err("a picture with no buffers to render".into());
        }
        let Ok(count) = c_int::try_from(buffers.len()) else { return Err(format!("{} buffers", buffers.len())) };
        let _gate = self.display.gate();
        // SAFETY: `context` is this session's own context and `surface` one of its own surfaces;
        // both were created together and neither has been destroyed.
        let begun = unsafe { (api.begin_picture)(self.display.dpy, context, surface) };
        if begun != OK {
            return Err(format!("vaBeginPicture failed on surface {surface}: {}", api.status_text(begun)));
        }
        // SAFETY: as above, plus `buffers` is a live slice of `count` ids, all of them buffers this
        // session created and not yet destroyed.
        let st = unsafe { (api.render_picture)(self.display.dpy, context, buffers.as_ptr().cast_mut(), count) };
        // SAFETY: as above. `vaEndPicture` is the matching call for the `vaBeginPicture` above, and
        // is made whether or not `vaRenderPicture` succeeded: libva requires the pair.
        let ended = unsafe { (api.end_picture)(self.display.dpy, context) };
        if st != OK {
            return Err(format!("vaRenderPicture failed for {} buffers: {}", buffers.len(), api.status_text(st)));
        }
        if ended != OK {
            return Err(format!("vaEndPicture failed: {}", api.status_text(ended)));
        }
        Ok(())
    }

    /// Wait for the rendering on `surface` to finish. **Required before its pixels are read.**
    ///
    /// `VA_STATUS_ERROR_HW_BUSY` means the video engine was resetting; libva documents that the
    /// call succeeds on a later one, so this asks again rather than reporting a decode failure the
    /// stream may not have had. The retry count is bounded because a driver that keeps answering
    /// `HW_BUSY` is a driver that is not going to answer at all.
    pub fn sync(&self, surface: u32) -> Result<(), String> {
        let api = &self.display.api;
        let _gate = self.display.gate();
        for _ in 0..MAX_SYNC_ATTEMPTS {
            // SAFETY: `surface` is one of this session's own surfaces.
            let st = unsafe { (api.sync_surface)(self.display.dpy, surface) };
            match st {
                OK => return Ok(()),
                HW_BUSY => continue,
                // `VA_STATUS_ERROR_DECODING_ERROR` (`va.h:283`) is the important one: the driver ran
                // the decode and the picture in the surface is not decodable. It is reported like any
                // other failure, and `status_text` says which it was.
                other => return Err(format!("vaSyncSurface failed on surface {surface}: {}", api.status_text(other))),
            }
        }
        Err(format!("vaSyncSurface on surface {surface} still reports the engine busy"))
    }

    /// Whether the driver says `surface` is ready to be read — that is, still being rendered into,
    /// or being displayed. `Err` if the query itself failed.
    pub fn surface_ready(&self, surface: u32) -> Result<bool, String> {
        let api = &self.display.api;
        let _gate = self.display.gate();
        let mut status: c_uint = 0;
        // SAFETY: `surface` is one of this session's own surfaces and `status` is a live local for
        // the flag word libva writes.
        let st = unsafe { (api.query_surface_status)(self.display.dpy, surface, &mut status) };
        if st != OK {
            return Err(format!("vaQuerySurfaceStatus failed on surface {surface}: {}", api.status_text(st)));
        }
        Ok(status & SURFACE_READY != 0)
    }

    /// The pixels of `surface`, mapped and ready to read.
    ///
    /// Two paths, in the order libva documents. `vaDeriveImage` is the direct one: the surface's
    /// own memory is handed back as a `VAImage`, and reading it is one copy. When it returns
    /// `VA_STATUS_ERROR_OPERATION_FAILED` — the documented answer for a surface that cannot be
    /// represented as an image, which is most often a driver whose surfaces are tiled — the picture
    /// is read the indirect way instead: an image of its own is created and the surface copied into
    /// it. That is a real copy through the driver and it is slower, which is why whether a session
    /// needs it is remembered and reported rather than discovered once per picture.
    ///
    /// The caller checks [`VaImage`] against its own gate ([`crate::vaapi::validate_image`]) before
    /// reading a byte, and borrows the mapped bytes through [`Readback::with_bytes`]; the map is
    /// paired with its unmap and the image with its destroy in [`Readback`]'s `Drop`, so a failure
    /// between here and the read cannot leak either.
    pub fn read(&mut self, surface: u32) -> Result<Readback, String> {
        if self.indirect {
            return self.read_indirect(surface);
        }
        let mut image = VaImage::default();
        let status = {
            let _gate = self.display.gate();
            // SAFETY: `surface` is one of this session's own surfaces and `image` is a live local
            // libva fills in with the image's own description.
            unsafe { (self.display.api.derive_image)(self.display.dpy, surface, &mut image) }
        };
        if status == OPERATION_FAILED {
            self.indirect = true;
            log::info!("VA-API: this driver's surfaces cannot be derived into an image; reading pictures back through a copy");
            return self.read_indirect(surface);
        }
        if status != OK {
            return Err(format!("vaDeriveImage failed on surface {surface}: {}", self.display.api.status_text(status)));
        }
        let display = Arc::clone(&self.display);
        let read = Readback::map(display, image, true)?;
        if read.image().format.fourcc != self.fourcc {
            return Err(format!(
                "the driver derived surface {surface} as fourcc {:#x}, not the {:#x} this session decodes into",
                read.image().format.fourcc,
                self.fourcc
            ));
        }
        Ok(read)
    }

    /// The indirect readback: `vaCreateImage` + `vaPutImage` (`va.h:4892`), the fallback `vaDeriveImage`
    /// documents. An error on either step destroys the image, so a failure here leaves nothing
    /// behind.
    fn read_indirect(&self, surface: u32) -> Result<Readback, String> {
        let (Some(create_image), Some(destroy_image), Some(put_image)) =
            (self.display.api.create_image, self.display.api.destroy_image, self.display.api.put_image)
        else {
            return Err("this libva has no vaCreateImage / vaPutImage, so a surface that cannot be derived cannot be read".into());
        };
        let api = &self.display.api;
        let mut format = VaImageFormat { fourcc: self.fourcc, byte_order: BYTE_ORDER_LSB_FIRST, bits_per_pixel: self.bpp, ..Default::default() };
        let mut image = VaImage::default();
        let (width, height) = (
            c_int::try_from(self.size.0).map_err(|_| format!("picture width {}", self.size.0))?,
            c_int::try_from(self.size.1).map_err(|_| format!("picture height {}", self.size.1))?,
        );
        let st = {
            let _gate = self.display.gate();
            // SAFETY: the display is initialised, `format` and `image` are live locals, and the sizes
            // are the ones the image is allocated at.
            unsafe { create_image(self.display.dpy, &mut format, width, height, &mut image) }
        };
        if st != OK {
            return Err(format!("vaCreateImage failed for the indirect readback: {}", api.status_text(st)));
        }
        // From here the image exists, so every exit has to destroy it. `Readback::map` takes that
        // over as soon as the image is mapped; before that the two calls below do it themselves.
        let extent = |s: u32| c_uint::try_from(s).unwrap_or(0);
        let st = {
            let _gate = self.display.gate();
            // SAFETY: `image` is the image just created, and `surface` is one of this session's own
            // surfaces; the rectangle is the whole of both.
            unsafe {
                put_image(
                    self.display.dpy,
                    surface,
                    image.image_id,
                    0,
                    0,
                    extent(self.size.0),
                    extent(self.size.1),
                    0,
                    0,
                    extent(self.size.0),
                    extent(self.size.1),
                )
            }
        };
        if st != OK {
            // Under the gate like every other libva call in this file: this display is shared, and a
            // destroy that ran beside another thread's call would be a data race in the driver.
            let _gate = self.display.gate();
            // SAFETY: `image.image_id` is the image `vaCreateImage` filled in above, it was never
            // mapped (so never unmapped), and nothing else holds it — this is its one destroy.
            unsafe { destroy_image(self.display.dpy, image.image_id) };
            return Err(format!("vaPutImage failed on surface {surface}: {}", api.status_text(st)));
        }
        let display = Arc::clone(&self.display);
        // `vaCreateImage` made this image, so this function owns it until a `Readback` does — and
        // `Readback::map` only takes ownership on success. A map that fails therefore has to destroy
        // it here, which is what `derived == false` says: `Readback::map` destroys a *derived* image
        // itself (nothing else can tell libva the id is unwanted) but leaves a created one to the
        // caller, because a created image's id is the caller's to release.
        let image_id = image.image_id;
        match Readback::map(display, image, false) {
            Ok(read) => Ok(read),
            Err(why) => {
                let _gate = self.display.gate();
                // SAFETY: `image_id` is the image `vaCreateImage` filled in above, it was never mapped
                // (so never unmapped), and nothing else holds it — this is its one destroy.
                unsafe { destroy_image(self.display.dpy, image_id) };
                Err(why)
            }
        }
    }
}

/// An encoding session: a config, a surface the input picture is read from, a context and the
/// coded buffer the driver writes.
///
/// **Encode's data flow is decode's, inverted** (the plan's "decisive difference"): the app hands
/// the driver a picture and the driver writes the slice headers. So there is no bitstream to parse
/// and no parameter set to submit — there is a surface to fill and a buffer to read back, and every
/// value in the records between them is a statement the driver believes.
///
/// That is why this type is deliberately small and why [`crate::vaapi::encoder`] — the safe module
/// that builds the records — holds the settings. This type owns libva objects and nothing else: the
/// config, the surfaces and the context, in the order libva requires them destroyed, plus the one
/// upload path. Nothing here interprets a picture or chooses a parameter.
pub struct Encoder {
    display: Arc<Display>,
    config: VAConfigID,
    /// The surface `vaBeginPicture` is given, which is the picture the driver **reads**. Separate
    /// from the reconstruction pool below because the driver reconstructs the picture it is
    /// encoding and needs somewhere to put that; it may not overwrite the picture it is reading,
    /// and it may not be a reference, because the caller overwrites it with the next picture.
    input: VASurfaceID,
    /// Where reconstructions go, in rotation. **Length is what makes inter prediction possible**:
    /// an encoder with one surface has a reconstruction and nothing to predict from, so a P picture
    /// has no reference to name. `recon` is `recon_pool[0]` until the caller rotates it.
    recon_pool: Vec<VASurfaceID>,
    /// The surface `curr_pic.picture_id` names, where the reconstructed picture lands and where it
    /// is read back from as a reference frame.
    recon: VASurfaceID,
    context: VAContextID,
    /// The [`ENC_CODED_BUFFER`] the compressed picture goes into. Kept for the life of the session
    /// because it is mapped per picture rather than created per picture, and because its storage is
    /// what [`CodedReadback`] maps. **Its size is not recorded here**: the readback takes the length
    /// of the storage from libva's own [`VaBufferInfo`], which describes the bytes being read rather
    /// than the size this backend happened to ask for.
    coded: Buffer,

    size: (u32, u32),
    /// The render format's four-character code and bit depth, so an upload asks for the same layout
    /// the session was configured in rather than whatever the driver prefers.
    fourcc: u32,
    bpp: u32,
}

impl Encoder {
    /// Configure `display` as a constant-quantiser H.264 (or HEVC) slice encoder, create its two
    /// surfaces, make a context and a coded buffer.
    ///
    /// The configuration is the smallest one that is not a default: a render format (which one is
    /// what the encoder writes), a rate control mode, and the packed header mode that makes the
    /// driver write the parameter sets. Everything else is left to the driver, which is the right
    /// default for a decoder but **not** for an encoder — see the module note above.
    ///
    /// `rate_control` and `packed_headers` are arguments rather than chosen here so that the
    /// caller states what it is doing and the failure names the value; a caller that cannot derive
    /// them faithfully has no honest value to pass.
    pub fn new(
        display: &Arc<Display>,
        profile: Profile,
        format: RtFormat,
        size: (u32, u32),
        rate_control: u32,
        packed_headers: u32,
        coded_bytes: u32,
        refs: u32,
    ) -> Result<Self, String> {
        let (fourcc, bpp) = match format {
            RtFormat::Nv12 => (NV12_FOURCC, NV12_BPP),
            RtFormat::Yuv420_10 => (P010_FOURCC, P010_BPP),
            RtFormat::Other(bits) => return Err(format!("render format {bits:#x} is not 4:2:0 8- or 10-bit")),
        };
        let (Some(create_image), Some(destroy_image), Some(put_image)) = (display.api.create_image, display.api.destroy_image, display.api.put_image) else {
            return Err("this libva has no vaCreateImage / vaPutImage, so a picture cannot be uploaded to a surface".into());
        };
        let _ = (create_image, destroy_image, put_image);
        if size.0 == 0 || size.1 == 0 {
            return Err(format!("{}x{} picture", size.0, size.1));
        }
        // A coded buffer of zero bytes cannot hold a slice, and `vaCreateBuffer` would accept the
        // size and then write into nothing.
        if coded_bytes == 0 {
            return Err("a coded buffer of 0 bytes".into());
        }
        let api = Arc::clone(&display.api);
        let width = c_uint::try_from(size.0).map_err(|_| format!("surface width {}", size.0))?;
        let height = c_uint::try_from(size.1).map_err(|_| format!("surface height {}", size.1))?;
        let mut session = Self {
            display: Arc::clone(display),
            config: 0,
            input: 0,
            recon_pool: Vec::new(),
            recon: 0,
            context: 0,
            // A placeholder that is replaced below; `Drop` reads it only if this is reached, and
            // this returns an error rather than an `Encoder` in every such case.
            coded: Buffer::placeholder(Arc::clone(display)),
            size,
            fourcc,
            bpp,
        };
        {
            let _gate = display.gate();
            // The three attributes, in the order libva documents: format first because the others
            // are meaningless without it, then the rate control mode, then the packed header mode.
            // Mutable because `vaCreateConfig` takes a `VAConfigAttrib*` — libva reads these and
            // writes back the values it ended up with, which for an attribute it can only ignore
            // is the value that says so.
            let mut attribs = [
                VaConfigAttrib { type_: CONFIG_ATTRIB_RT_FORMAT, value: format.bits() },
                VaConfigAttrib { type_: CONFIG_ATTRIB_RATE_CONTROL, value: rate_control },
                VaConfigAttrib { type_: CONFIG_ATTRIB_ENC_PACKED_HEADERS, value: packed_headers },
            ];
            let mut config: VAConfigID = 0;
            // SAFETY: the display is initialised; `attribs` is a live array of three records the
            // call reads and does not keep; and `config` is a live local for the id it writes.
            let st = unsafe { (api.create_config)(display.dpy, profile.bits(), ENC_SLICE, attribs.as_mut_ptr(), attribs.len() as c_int, &mut config) };
            if st != OK {
                return Err(format!(
                    "vaCreateConfig failed for {:?}/EncSlice at rate control {rate_control:#x} and packed headers {packed_headers:#x}: {}",
                    profile,
                    api.status_text(st)
                ));
            }
            session.config = config;
            // **The surface count is what makes inter prediction possible at all**, and it is
            // `2 + max_ref_frames` for a reason worth stating:
            //
            // - one surface is the picture being read (`vaBeginPicture`'s argument), which must
            //   never be a reference: the caller overwrites it with the next picture, and a
            //   reference that has been overwritten is a reference to the wrong picture;
            // - one is the reconstruction target (`curr_pic.picture_id`), which must differ from
            //   the input or the driver reads its own output as the picture it is encoding;
            // - and `max_ref_frames` more hold the reconstructions a later picture predicts from.
            //
            // With the two surfaces Task 1 used, a P picture has nowhere to put its reference, so
            // the honest size is three at least. `max_ref_frames` is clamped because it comes from
            // the caller and each surface is real driver memory at the full picture size.
            let count = 2usize.saturating_add(refs as usize).clamp(2, MAX_SURFACES);
            let mut surfaces = vec![0 as VASurfaceID; count];
            // SAFETY: as above for the surfaces: `surfaces` is a live array of `count` ids, and no
            // attribute list is passed.
            let st =
                unsafe { (api.create_surfaces)(display.dpy, format.bits(), width, height, surfaces.as_mut_ptr(), count as c_uint, std::ptr::null_mut(), 0) };
            if st != OK {
                return Err(format!("vaCreateSurfaces failed for the encode session's {count} surfaces at {}x{}: {}", size.0, size.1, api.status_text(st)));
            }
            // `vaCreateSurfaces` filled these in; a surface id invented here would name one the
            // driver never made. **A zero id is legal and is not a failure**: surface ids are
            // allocated by the driver from its own counter, and the first surface a session creates
            // is id 0 on this driver (Task 1's trace shows `render_targets = 0x00000000`). Checking
            // for zero here would refuse every first session — which is the mistake this very probe
            // made before the comment above was written. What is checked instead is the count and
            // the absence of duplicates, neither of which has a legal zero.
            //
            // This matters beyond this call: `VA_INVALID_SURFACE` is `0xffff_ffff` precisely
            // because 0 is a real surface id, which is why every unused reference slot in this
            // module is `VA_INVALID_SURFACE` and never zero.
            if surfaces.len() < count {
                return Err(format!("vaCreateSurfaces filled in fewer than the {count} surface ids it was asked for"));
            }
            for (i, a) in surfaces.iter().enumerate() {
                if surfaces.get(i + 1..).is_some_and(|rest| rest.contains(a)) {
                    return Err("vaCreateSurfaces returned the same surface id twice".into());
                }
            }
            // Surface 0 is the input and is never a reference; the rest are the DPB.
            session.input = surfaces[0];
            session.recon_pool = surfaces[1..].to_vec();
            session.recon = session.recon_pool[0];
            let (context_width, context_height) = (
                c_int::try_from(size.0).map_err(|_| format!("picture width {}", size.0))?,
                c_int::try_from(size.1).map_err(|_| format!("picture height {}", size.1))?,
            );
            let mut context: VAContextID = 0;
            // SAFETY: `config` is the config created above, `surfaces` the array just filled, and
            // `context` a live local for the id to be written into. The whole pool is the context's
            // render target list: a reconstruction a later picture predicts from is rendered into
            // and read from as a surface, so it has to be one the context knows.
            let st = unsafe {
                (api.create_context)(display.dpy, config, context_width, context_height, PROGRESSIVE, surfaces.as_mut_ptr(), count as c_int, &mut context)
            };
            if st != OK {
                return Err(format!("vaCreateContext failed for the encoder at {}x{}: {}", size.0, size.1, api.status_text(st)));
            }
            session.context = context;
            // The coded buffer. **`data` is null and `num_elements` is 1**: this is the one buffer
            // the driver fills rather than reads, and iHD treats a buffer of this type with
            // initialisation data as an error rather than ignoring the data. The size is the
            // caller's, because only the caller knows what an uncompressed picture of this size
            // costs — this backend does not guess one.
            let mut coded: VABufferID = 0;
            // SAFETY: `context` is the context created above on this display, `coded_bytes` is the
            // size already validated above, `data` is null (which is how this buffer kind says "the
            // driver fills me"), and `coded` is a live local for the id libva writes.
            let st = unsafe { (api.create_buffer)(display.dpy, context, ENC_CODED_BUFFER, coded_bytes, 1, std::ptr::null_mut(), &mut coded) };
            if st != OK {
                return Err(format!("vaCreateBuffer failed for a {coded_bytes} byte coded buffer: {}", api.status_text(st)));
            }
            session.coded = Buffer::owned(Arc::clone(display), coded);
        }
        Ok(session)
    }

    /// The surface `vaBeginPicture` is given: the picture the driver reads.
    pub fn input_surface(&self) -> VASurfaceID {
        self.input
    }

    /// The surface `curr_pic.picture_id` names: where the reconstructed picture lands, and the
    /// surface a later P or B picture predicts from.
    pub fn recon_surface(&self) -> VASurfaceID {
        self.recon
    }

    /// **Move the reconstruction target to the next surface in the pool**, and say which one it is
    /// now.
    ///
    /// This is the whole of the surface management an inter picture needs, and its absence is why
    /// Task 1 could not encode a P picture: with one reconstruction surface there is nowhere for the
    /// *previous* picture's reconstruction to live while the current one is being written. Rotating
    /// through a pool of `2 + max_ref_frames` is enough for one reference, and the caller decides
    /// which surfaces are live references — this module only knows that the target has moved.
    ///
    /// It wraps rather than running off the end, and a caller that has not reserved enough surfaces
    /// for its reference depth will find a reference overwritten. The surface count is the caller's
    /// because only the caller knows its own `max_num_ref_frames`.
    pub fn rotate_reconstruction(&mut self) -> VASurfaceID {
        // `recon_pool` is non-empty whenever `recon` is set: `Encoder::new` fills it from
        // `surfaces[1..]`, and the count clamp guarantees at least one element there. An empty pool
        // means the encoder was never built, and rotating it would leave `recon` at 0 — a surface
        // that does not exist — so the length is checked rather than assumed.
        if self.recon_pool.len() > 1 {
            self.recon_pool.rotate_left(1);
            if let Some(next) = self.recon_pool.first().copied() {
                self.recon = next;
            }
        }
        self.recon
    }

    /// The coded buffer's id, which the picture record's `coded_buf` must name. The record cannot be
    /// built without it, so it is read here rather than kept by the caller.
    pub fn coded_buffer_id(&self) -> VABufferID {
        self.coded.id()
    }

    /// What the driver says it supports for [`CONFIG_ATTRIB_ENC_MAX_REF_FRAMES`] on this profile and
    /// entry point, as `(list 0, list 1)`.
    ///
    /// **This is a driver query, not this session's state**, so it is asked of the display before
    /// any config exists — see [`Display::config_attributes`]. A driver that does not answer is
    /// reported as `(0, 0)` rather than an error, because a libva without `vaGetConfigAttributes` can
    /// still encode; the caller learns the depth is unknown by the zero, and a zero depth is the
    /// honest answer for "I cannot tell you".
    pub fn driver_max_reference_frames(display: &Display, profile: Profile) -> (u32, u32) {
        let wanted = [CONFIG_ATTRIB_ENC_MAX_REF_FRAMES, CONFIG_ATTRIB_ENC_SLICE_STRUCTURE];
        match display.config_attributes(profile, Entrypoint::EncSlice, &wanted) {
            // The low 16 bits are list 0 and the high 16 list 1 (`va.h:737-739`). Masking each half
            // rather than shifting the whole word means a driver that fills the high half with
            // something else does not become a reference depth of 65535.
            Ok(values) => (values.first().copied().unwrap_or(0) & 0xffff, values.first().copied().unwrap_or(0) >> 16),
            Err(why) => {
                log::info!("VA-API: no vaGetConfigAttributes, so the driver's reference depth is unknown: {why}");
                (0, 0)
            }
        }
    }

    /// A buffer of `data` of kind `kind`, destroyed when it is dropped.
    ///
    /// `num_elements` is 1 for every record buffer this backend submits except the slice parameter
    /// buffer, which is an **array** of records. The distinction is not a detail: iHD refuses a
    /// slice parameter buffer created with one element per record the way it refuses a decode
    /// picture parameter buffer sized that way.
    pub fn buffer(&self, kind: VAStatus, data: &[u8], num_elements: c_uint) -> Result<Buffer, String> {
        let Some(size) = c_uint::try_from(data.len()).ok() else { return Err(format!("{} byte buffer", data.len())) };
        let mut id: VABufferID = 0;
        // SAFETY: `context` is this session's own context, `data` is `size` readable bytes (which
        // is what libva copies for a record buffer), and `id` is a live local for the buffer id.
        let st =
            unsafe { (self.display.api.create_buffer)(self.display.dpy, self.context, kind, size, num_elements, data.as_ptr().cast_mut().cast(), &mut id) };
        if st != OK {
            return Err(format!("vaCreateBuffer failed for {size} bytes of type {kind} in {num_elements} element(s): {}", self.display.api.status_text(st)));
        }
        Ok(Buffer::owned(Arc::clone(&self.display), id))
    }

    /// Encode: `vaBeginPicture` on the input surface, submit `buffers`, `vaEndPicture`.
    ///
    /// The picture is always rendered into the reconstruction surface and read from the input one.
    /// A driver that emits the picture it is asked to encode and not something else is a driver
    /// this backend has to catch, not work around.
    pub fn render(&self, buffers: &[u32]) -> Result<(), String> {
        if buffers.is_empty() {
            return Err("a picture with no buffers to render".into());
        }
        let Ok(count) = c_int::try_from(buffers.len()) else { return Err(format!("{} buffers", buffers.len())) };
        let _gate = self.display.gate();
        // SAFETY: `context` is this session's own context and `input` one of its own surfaces;
        // both were created together and neither has been destroyed.
        let begun = unsafe { (self.display.api.begin_picture)(self.display.dpy, self.context, self.input) };
        if begun != OK {
            return Err(format!("vaBeginPicture failed on surface {}: {}", self.input, self.display.api.status_text(begun)));
        }
        // SAFETY: as above, plus `buffers` is a live slice of `count` ids, all buffers this session
        // created and not yet destroyed.
        let st = unsafe { (self.display.api.render_picture)(self.display.dpy, self.context, buffers.as_ptr().cast_mut(), count) };
        // SAFETY: as above. `vaEndPicture` pairs with the `vaBeginPicture` above and is made
        // whether or not the render succeeded.
        let ended = unsafe { (self.display.api.end_picture)(self.display.dpy, self.context) };
        if st != OK {
            return Err(format!("vaRenderPicture failed for {} encode buffers: {}", buffers.len(), self.display.api.status_text(st)));
        }
        if ended != OK {
            return Err(format!("vaEndPicture failed: {}", self.display.api.status_text(ended)));
        }
        Ok(())
    }

    /// Wait for the encode to finish. Required before the coded buffer is read.
    ///
    /// **`vaSyncSurface` on the input surface is not a substitute.** The decode path synchronises
    /// the surface whose pixels it is about to read; the encoder's output is in a *buffer*, and
    /// libva's answer for that is `vaSyncBuffer`. Reading it without is a truncated picture, which
    /// on a driver that is fast enough is a picture that is usually right.
    pub fn sync(&self) -> Result<(), String> {
        let Some(sync_buffer) = self.display.api.sync_buffer else {
            // Not an error: the buffer can still be mapped, and the caller can see whether it holds
            // a whole picture (the slice records say how many bytes each slice is, and a picture
            // shorter than that is a truncated one).
            log::info!("VA-API: this libva has no vaSyncBuffer, so an encoded picture is read without synchronising first");
            return Ok(());
        };
        let _gate = self.display.gate();
        // **The timeout argument is passed explicitly.** `vaSyncBuffer`'s third parameter
        // (`va.h:4391`) is `uint64_t timeout_ns`, and omitting it leaves the driver reading
        // whatever happened to be in that argument slot: which is why this call reported
        // `VA_STATUS_ERROR_TIMEDOUT` in an unoptimised build and succeeded in an optimised one.
        // Ten milliseconds per attempt is far longer than a 640x360 picture takes on this driver.
        //
        // **The loop terminates.** `MAX_SYNC_ATTEMPTS` (`va.h:4381-4382` says a timed-out buffer is
        // still synchronising and the call is to be made again) is the bound: at most 64 attempts
        // of at most 10 ms each, so at worst this returns an error after about two thirds of a
        // second. A driver that answers `HW_BUSY` or `TIMEDOUT` forever is a driver that is not
        // going to answer at all, and an export must not hang on one.
        const SYNC_TIMEOUT_NS: u64 = 10_000_000;
        for _ in 0..MAX_SYNC_ATTEMPTS {
            // SAFETY: `coded` is this session's own coded buffer and has not been destroyed, and
            // `SYNC_TIMEOUT_NS` is the `uint64_t` the third parameter is declared as.
            let st = unsafe { sync_buffer(self.display.dpy, self.coded.id(), SYNC_TIMEOUT_NS) };
            match st {
                OK => return Ok(()),
                // Both of these are "not finished", not "failed": see `TIMED_OUT`'s doc comment.
                TIMED_OUT | HW_BUSY => continue,
                other => return Err(format!("vaSyncBuffer failed on the coded buffer: {other} ({})", self.display.api.status_text(other))),
            }
        }
        Err(format!("vaSyncBuffer on the coded buffer was still busy after {MAX_SYNC_ATTEMPTS} attempts"))
    }

    /// Put a picture on the input surface.
    ///
    /// `luma` is `width * height` bytes and `chroma` is `width * height / 2` bytes (interleaved
    /// Cb, Cr), both tightly packed. The driver is asked for a linear image of the session's own
    /// format and the picture is copied through `vaPutImage` into the surface.
    ///
    /// The alternative — writing into the surface's own memory through `vaDeriveImage` — is not
    /// available: iHD's surfaces are tiled, and the decode backend already learned that a surface
    /// it cannot derive has to be copied the indirect way. An encode upload has no choice but the
    /// copy, and it is the same cost the decode readback already pays.
    pub fn upload(&self, luma: &[u8], chroma: &[u8]) -> Result<(), String> {
        let (Some(create_image), Some(destroy_image), Some(put_image)) =
            (self.display.api.create_image, self.display.api.destroy_image, self.display.api.put_image)
        else {
            return Err("this libva has no vaCreateImage / vaPutImage, so a picture cannot be uploaded to a surface".into());
        };
        // The luma plane is `width * height` bytes and the chroma plane half as many, at the
        // session's own coded size. Every length is computed with `checked_*`: a size that would
        // overflow is an error rather than a wrapped one that then bounds a slice wrongly.
        let Some(luma_len) = self.size.0.checked_mul(self.size.1) else {
            return Err(format!("{}x{} luma plane overflows a length", self.size.0, self.size.1));
        };
        let Some(chroma_len) = self.size.1.div_ceil(2).checked_mul(self.size.0) else {
            return Err(format!("{}x{} chroma plane overflows a length", self.size.0, self.size.1));
        };
        let (luma_len, chroma_len) = (luma_len as usize, chroma_len as usize);
        // The lengths are the driver's geometry, so the caller's two planes are checked against
        // them rather than the other way round: a short slice is a bug in the caller, and writing
        // past the end of one would be the alternative.
        let (Some(luma), Some(chroma)) = (luma.get(..luma_len), chroma.get(..chroma_len)) else {
            return Err(format!(
                "a {}x{} 4:2:0 picture is {luma_len} luma bytes and {chroma_len} chroma bytes, not {} and {}",
                self.size.0,
                self.size.1,
                luma.len(),
                chroma.len()
            ));
        };
        let mut format = VaImageFormat { fourcc: self.fourcc, byte_order: BYTE_ORDER_LSB_FIRST, bits_per_pixel: self.bpp, ..Default::default() };
        let mut image = VaImage::default();
        let (width, height) = (
            c_int::try_from(self.size.0).map_err(|_| format!("picture width {}", self.size.0))?,
            c_int::try_from(self.size.1).map_err(|_| format!("picture height {}", self.size.1))?,
        );
        let st = {
            let _gate = self.display.gate();
            // SAFETY: the display is initialised, `format` and `image` are live locals, and the
            // sizes are the ones the image is allocated at.
            unsafe { create_image(self.display.dpy, &mut format, width, height, &mut image) }
        };
        if st != OK {
            return Err(format!("vaCreateImage failed for the encode upload: {}", self.display.api.status_text(st)));
        }
        // The image exists from here, so every exit destroys it. `Upload`'s `Drop` takes that over
        // as soon as the buffer is mapped.
        let mut mapped: *mut c_void = std::ptr::null_mut();
        let st = {
            let _gate = self.display.gate();
            // SAFETY: `image.buf` is the buffer of the image just created and `mapped` is a live
            // local for the address libva writes.
            unsafe { (self.display.api.map_buffer)(self.display.dpy, image.buf, &mut mapped) }
        };
        if st != OK {
            let _gate = self.display.gate();
            // SAFETY: the image was created above, never mapped (so never unmapped), and nothing
            // else holds its id — this is its one destroy.
            unsafe { destroy_image(self.display.dpy, image.image_id) };
            return Err(format!("vaMapBuffer failed for the encode upload: {}", self.display.api.status_text(st)));
        }
        if mapped.is_null() {
            let _gate = self.display.gate();
            // SAFETY: as above.
            unsafe { destroy_image(self.display.dpy, image.image_id) };
            return Err("vaMapBuffer succeeded for the encode upload but returned no address".into());
        }
        let image_id = image.image_id;
        let upload = Upload { display: Arc::clone(&self.display), image, data: mapped, mapped: true };
        // The two planes are copied through the driver's own geometry, and every number in it is
        // checked against `data_size` before a byte is written.
        upload.copy_in(luma, chroma)?;
        let st = {
            let _gate = self.display.gate();
            // SAFETY: `image` is the image just created and mapped, `input` is this session's own
            // surface, and the rectangle is the whole of both.
            let extent = |v: u32| c_uint::try_from(v).unwrap_or(0);
            // SAFETY: as the comment above says: the image is this value's own and still mapped, the
            // surface is this session's, and the rectangle is the whole of both. The `image_id`
            // local was taken before `upload` took the image, so it is the one `vaCreateImage` wrote.
            unsafe {
                put_image(
                    self.display.dpy,
                    self.input,
                    image_id,
                    0,
                    0,
                    extent(self.size.0),
                    extent(self.size.1),
                    0,
                    0,
                    extent(self.size.0),
                    extent(self.size.1),
                )
            }
        };
        // The image is unmapped and destroyed by `upload`'s `Drop` below, on every path.
        drop(upload);
        if st != OK {
            return Err(format!("vaPutImage failed onto the encode surface: {}", self.display.api.status_text(st)));
        }
        Ok(())
    }

    /// The coded picture the last render wrote, copied out of the buffer object's own storage.
    ///
    /// [`Encoder::sync`] has already been called, so the encode is finished.
    ///
    /// **Every byte comes from a mapping this backend made.** The picture is not read through
    /// `VaCodedBufferSegment::buf`, which is a pointer the driver chose that nothing bounds, and not
    /// through `vaMapBuffer`'s return, which on this stack is libva's own staging allocation rather
    /// than the picture. It is read from [`crate::vaapi::drm`]: libva names the buffer's DRM object
    /// and its size, and that object is mapped on FilmCraft's own descriptor, so the length of every
    /// read is one this backend chose.
    ///
    /// What is still the driver's is the **length**: `VaCodedBufferSegment::size` is the only place
    /// libva says how many bytes the picture is, so it is read, and it is checked to fit inside the
    /// storage libva described.
    pub fn read_coded(&self) -> Result<CodedPicture, String> {
        let mut mapped: *mut c_void = std::ptr::null_mut();
        let st = {
            let _gate = self.display.gate();
            // SAFETY: `coded` is this session's own coded buffer and `mapped` is a live local.
            unsafe { (self.display.api.map_buffer)(self.display.dpy, self.coded.id(), &mut mapped) }
        };
        if st != OK {
            return Err(format!("vaMapBuffer failed on the coded buffer: {}", self.display.api.status_text(st)));
        }
        if mapped.is_null() {
            return Err("vaMapBuffer succeeded on the coded buffer but returned no address".into());
        }
        let read = CodedReadback { display: Arc::clone(&self.display), id: self.coded.id(), data: mapped };
        // `read` owns the mapping and unmaps it in `Drop` on every path, so a malformed coded
        // buffer cannot leave the buffer mapped.
        read.picture()
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        let api = Arc::clone(&self.display.api);
        // The coded buffer is taken out first and released **after** the gate is dropped, because
        // `Buffer::drop` takes that same gate: dropping it inside the gated scope deadlocks on a
        // `std::sync::Mutex`, which is not reentrant. That is not a theoretical hazard — it is what
        // this method did the first time, and the symptom was a process that hung *after* libva had
        // already torn everything down and logged its last `vaDestroyConfig`.
        let coded = std::mem::replace(&mut self.coded, Buffer::placeholder(Arc::clone(&self.display)));
        {
            let _gate = self.display.gate();
            // libva's own order, as `Session::drop` does it: the context, then the surfaces it
            // rendered into, then the config. Each is released only if it exists, because an
            // `Encoder` that failed part-way through `Encoder::new` is dropped too and the ids it
            // never filled in are zero — which is not a surface or a context the driver made.
            if self.context != 0 {
                // SAFETY: `context` is this session's own context, created in `Encoder::new` and not
                // destroyed on any earlier path.
                unsafe { (api.destroy_context)(self.display.dpy, self.context) };
            }
            if self.input != 0 && !self.recon_pool.is_empty() {
                // The input plus the whole reconstruction pool, which is every surface the context
                // was created with. `self.input` is `surfaces[0]` and `recon_pool` is `surfaces[1..]`,
                // so this reassembles the exact array `Encoder::new` created, in the same order.
                let mut surfaces = vec![self.input];
                surfaces.extend_from_slice(&self.recon_pool);
                // SAFETY: `surfaces` is a live array of exactly the ids `vaCreateSurfaces` wrote in
                // `Encoder::new`, none destroyed on any earlier path, and the context that used them
                // was destroyed above — which is libva's own order.
                unsafe { (api.destroy_surfaces)(self.display.dpy, surfaces.as_mut_ptr(), surfaces.len() as c_int) };
            }
            if self.config != 0 {
                // SAFETY: `config` is this session's own config id, created in `Encoder::new` and
                // not destroyed on any earlier path.
                unsafe { (api.destroy_config)(self.display.dpy, self.config) };
            }
        }
        // The coded buffer last, because libva forbids destroying a buffer while a render into it is
        // live — so it has to outlive the context. The gate is free again, so its own `Drop` can
        // take it.
        drop(coded);
    }
}

/// A mapped image the encoder is copying a picture into: the driver's `VAImage`, its mapped bytes,
/// and the matching unmap and destroy in `Drop`.
struct Upload {
    display: Arc<Display>,
    image: VaImage,
    data: *mut c_void,
    mapped: bool,
}

impl Upload {
    /// Copy `luma` and `chroma` into the mapped image, through the driver's own plane geometry.
    ///
    /// Everything here is bounded against the same three numbers [`crate::vaapi::validate_image`]
    /// checks for a decode readback, and for the same reason: they were written by a driver, and a
    /// pitch narrower than a row makes a row write walk off the plane.
    fn copy_in(&self, luma: &[u8], chroma: &[u8]) -> Result<(), String> {
        // The reuse is deliberate and is the second of the two gates on a driver-reported geometry
        // (the first is `validate_image`, which is for pictures coming *out*; this is a picture
        // going in, and it reads the same numbers).
        crate::vaapi::validate_image(&self.image)?;
        // SAFETY: `data` is what `vaMapBuffer` returned for `image.buf` and is still mapped (only
        // `Drop` unmaps it, once, with this value borrowed throughout), so the slice below is
        // `data_size` writable bytes — and it is the only writable view of the mapping, created
        // here and never handed out.
        let bytes = unsafe { std::slice::from_raw_parts_mut(self.data.cast::<u8>(), self.image.data_size as usize) };
        let (Some(&luma_pitch), Some(&chroma_pitch), Some(&luma_at), Some(&chroma_at)) =
            (self.image.pitches.first(), self.image.pitches.get(1), self.image.offsets.first(), self.image.offsets.get(1))
        else {
            return Err("the upload image has no plane geometry".into());
        };
        if luma_pitch != chroma_pitch {
            return Err(format!("upload luma pitch {luma_pitch} and chroma pitch {chroma_pitch} differ"));
        }
        let (width, height) = (self.image.width as usize, self.image.height as usize);
        let stride = luma_pitch as usize;
        // Each plane is one row-copy per image row, at the plane's own pitch, and each row's
        // destination range is a `get_mut` on the mapped slice — so a pitch or an offset the driver
        // got wrong cannot make a write leave the buffer. Explicit loops rather than an iterator,
        // because two closures over one mutable borrow do not typecheck and the rows must be
        // copied in order for the error to name the row that did not fit.
        for row in 0..height {
            let Some(at) = (luma_at as usize).checked_add(row * stride) else {
                return Err(format!("luma row {row} overflows the plane's offset"));
            };
            let (Some(from), Some(to)) =
                (luma.get(row * width..).and_then(|rest| rest.get(..width)), bytes.get_mut(at..).and_then(|rest| rest.get_mut(..width)))
            else {
                return Err(format!("luma row {row} does not fit the {}x{} upload image", width, height));
            };
            to.copy_from_slice(from);
        }
        let chroma_w = width.div_ceil(2);
        let chroma_bytes = chroma_w * 2;
        for row in 0..height.div_ceil(2) {
            let Some(at) = (chroma_at as usize).checked_add(row * stride) else {
                return Err(format!("chroma row {row} overflows the plane's offset"));
            };
            let (Some(from), Some(to)) =
                (chroma.get(row * chroma_bytes..).and_then(|rest| rest.get(..chroma_bytes)), bytes.get_mut(at..).and_then(|rest| rest.get_mut(..chroma_bytes)))
            else {
                return Err(format!("chroma row {row} does not fit the {}x{} upload image", width, height));
            };
            to.copy_from_slice(from);
        }
        Ok(())
    }
}

impl Drop for Upload {
    fn drop(&mut self) {
        let _gate = self.display.gate();
        if self.mapped {
            // SAFETY: `image.buf` is the buffer mapped by `Encoder::upload` and has not been
            // unmapped since (only this `Drop` unmaps it, once, with the value borrowed
            // throughout).
            unsafe { (self.display.api.unmap_buffer)(self.display.dpy, self.image.buf) };
            self.mapped = false;
        }
        if let Some(destroy) = self.display.api.destroy_image {
            // SAFETY: `image.image_id` is the id `vaCreateImage` wrote, owned by this value, and
            // never destroyed elsewhere.
            unsafe { destroy(self.display.dpy, self.image.image_id) };
        }
    }
}

/// The coded buffer, mapped only long enough for the driver to say what it wrote about it.
///
/// `vaMapBuffer` on a coded buffer does not return the picture. On the stack this milestone was
/// measured on it returns libva's own staging allocation, and the pointer the picture's segment
/// record holds names a different allocation again — the measured evidence is in the milestone
/// report, and [`crate::vaapi::drm`] is what replaced reading through either.
///
/// What `vaMapBuffer` does return is documented, and is needed: `va.h:3946`, "if buffer type is
/// VAEncCodedBufferType, pbuf points to link-list of VACodedBufferSegment". That list is the only
/// place libva says **how many bytes** the picture is. So the mapping is held for exactly as long as
/// it takes to read that one record and copy the bytes out of the storage
/// [`crate::vaapi::drm`] mapped, and is unmapped in `Drop` on every path out of here.
struct CodedReadback {
    display: Arc<Display>,
    id: VABufferID,
    /// What `vaMapBuffer` returned: **the segment list is here, the picture is not.**
    data: *mut c_void,
}

impl CodedReadback {
    /// The head of the segment list, which the header states begins at the address `vaMapBuffer`
    /// returns.
    fn head_segment(&self) -> VaCodedBufferSegment {
        // SAFETY: `data` is what `vaMapBuffer` returned for this buffer and is still mapped (only
        // `Drop` unmaps it, once, with the value borrowed throughout), and `va.h:3946` states that
        // the list starts at that address, so one whole record is readable there. The read is
        // unaligned because nothing states the alignment of what `vaMapBuffer` returns, and
        // `VaCodedBufferSegment` is `Copy`, so nothing is moved and nothing is left initialised
        // behind.
        unsafe { std::ptr::read_unaligned(self.data.cast::<VaCodedBufferSegment>()) }
    }

    /// The coded picture, copied out of the buffer object's own storage.
    ///
    /// The only numbers the driver supplies are the length, the bit offset and the status word, and
    /// each is checked before a byte is used: a non-zero bit offset would mean the segment does not
    /// begin where this backend can name it, a length of 0 is not a picture, and a length past the
    /// storage libva described is refused rather than clamped to what happens to be there. The bytes
    /// themselves are then checked to be a whole run of Annex-B NAL units, which is what makes "the
    /// picture is at the start of the storage" something this backend verifies rather than assumes.
    fn picture(&self) -> Result<CodedPicture, String> {
        let segment = self.head_segment();
        if segment.bit_offset != 0 {
            // See the field's doc comment: a non-zero bit offset would mean the segment's first
            // bits are not picture data, and shifting by an amount whose unit this module cannot
            // verify would produce a plausible wrong picture. Measured, not assumed: iHD reports 0
            // on this machine.
            let bits = segment.bit_offset;
            return Err(format!("the coded picture starts {bits} bits into its segment, which this backend does not unshift"));
        }
        if !segment.next.is_null() {
            // **A second segment is refused, not read.** Following `next` means reading at a
            // pointer the driver chose, inside the staging allocation whose length no API reports —
            // exactly the unbounded read this readback exists to remove, and it would put back the
            // one this task took out. One slice per picture is what this backend asks the driver
            // for, so a driver that answers with more is reported rather than half-read.
            return Err("the driver reported the picture in more than one coded-buffer segment, whose links this backend does not follow".into());
        }
        let size = segment.size as usize;
        if size == 0 {
            return Err("the driver reported a coded picture of 0 bytes".into());
        }
        let storage = self.storage()?;
        let bytes = storage
            .bytes()
            .get(..size)
            .ok_or_else(|| format!("the driver reported a {size}-byte picture but libva described {} bytes of storage", storage.len()))?;
        check_annex_b(bytes)?;
        Ok(CodedPicture { bytes: bytes.to_vec(), status: vec![segment.status], segments: 1 })
    }

    /// The coded buffer's own storage, mapped by this process: libva names the DRM object and its
    /// size, and [`crate::vaapi::drm`] maps it on FilmCraft's own descriptor.
    fn storage(&self) -> Result<drm::Storage, String> {
        let mut info = VaBufferInfo::ZEROED;
        let st = {
            let _gate = self.display.gate();
            // SAFETY: `id` is this session's own coded buffer and `info` is a live local, zeroed on
            // input, which is what `va.h:4086` requires of `vaAcquireBufferHandle`.
            unsafe { (self.display.api.acquire_buffer_handle)(self.display.dpy, self.id, &mut info) }
        };
        // The handle is released on every path out of here, the errors below included, because
        // `vaReleaseBufferHandle` requires it (`va.h:4098-4104`).
        let _release = BufferHandle { display: Arc::clone(&self.display), id: self.id };
        if st != OK {
            return Err(format!("vaAcquireBufferHandle on the coded buffer: {}", self.display.api.status_text(st)));
        }
        // `VA_SURFACE_ATTRIB_MEM_TYPE_DRM`. `va.h:1752` reserves bits 31:28 of the memory type for
        // Linux DRM and this is the value in that range, measured on this stack; it is the only one
        // the export below means anything for, so anything else is reported rather than being sent
        // an ioctl it was not named for.
        const MEM_TYPE_DRM: u32 = 0x2000_0000;
        if info.mem_type != MEM_TYPE_DRM {
            return Err(format!("libva gave the coded buffer memory type {:#x}, not the DRM one this backend can map", info.mem_type));
        }
        // `VABufferInfo::handle` is a `uintptr_t` and a DRM handle is 32 bits: a wider value cannot
        // name a buffer object, so it is refused rather than truncated into one that might.
        let handle = u32::try_from(info.handle).map_err(|_| format!("libva reported buffer handle {}, which is not a DRM handle", info.handle))?;
        let drm_fd = self.display.drm_fd().ok_or_else(|| "the display holds no DRM descriptor to map the coded buffer with".to_string())?;
        drm::map(drm_fd, handle, info.mem_size)
    }
}

impl Drop for CodedReadback {
    fn drop(&mut self) {
        let _gate = self.display.gate();
        // SAFETY: `id` is the buffer mapped by `Encoder::read_coded` and has not been unmapped
        // since (only this `Drop` unmaps it, once).
        unsafe { (self.display.api.unmap_buffer)(self.display.dpy, self.id) };
    }
}

/// A buffer whose handle libva has been asked for, released when this is dropped.
struct BufferHandle {
    display: Arc<Display>,
    id: VABufferID,
}

impl Drop for BufferHandle {
    fn drop(&mut self) {
        let _gate = self.display.gate();
        // SAFETY: `id` is this session's own coded buffer and this is the one release of the one
        // `vaAcquireBufferHandle` in flight, which is the pairing `va.h:4098-4104` requires.
        unsafe { (self.display.api.release_buffer_handle)(self.display.dpy, self.id) };
    }
}

/// The offset and width of the Annex-B start code at or after `from`, if there is one.
///
/// A 4-byte code is preferred at the same offset, so a start code preceded by a zero bit of
/// `trailing_zero_8bits` is reported at the width that puts the NAL header after it where it is.
fn annex_b_start(bytes: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut at = from;
    while at + 3 <= bytes.len() {
        if bytes.get(at..at + 4) == Some(&[0, 0, 0, 1]) {
            return Some((at, 4));
        }
        if bytes.get(at..at + 3) == Some(&[0, 0, 1]) {
            return Some((at, 3));
        }
        at += 1;
    }
    None
}

/// `bytes` is a whole run of Annex-B start-code-prefixed NAL units: it begins with a start code, no
/// unit in it is empty, and the last one holds at least a NAL header.
///
/// This is what stops the readback from *assuming* where the picture is. `va.h:3949` calls
/// `VaCodedBufferSegment::buf` "Pointer to the start of the data buffer", and a buffer's data
/// buffer is its storage — but libva reports **no offset** for it, and the offset has to be 0 for
/// the read to be a read of storage this backend mapped. So it is not taken on faith: the window the
/// driver named is checked to be a whole run of NAL units, and a driver that put the picture
/// anywhere else produces an error here instead of bytes that decode to noise. Measured on this
/// stack: the picture is at offset 0 of all six pictures of a sequence, and the driver writes
/// nothing past the length it reports.
pub(crate) fn check_annex_b(bytes: &[u8]) -> Result<(), String> {
    let Some((mut at, mut width)) = annex_b_start(bytes, 0) else {
        return Err("the coded buffer's storage does not begin with an Annex-B start code".into());
    };
    if at != 0 {
        return Err(format!("the coded picture begins {at} bytes into the buffer's storage rather than at its start"));
    }
    let mut units = 1usize;
    // The next code is searched for **past the one just consumed**, `at + width` and not `at + 1`:
    // a 4-byte code `00 00 00 01` also *is* a 3-byte code one byte in, so a search from `at + 1`
    // finds the tail of the start code it is standing in and reports the same NAL unit twice.
    while let Some((next, next_width)) = annex_b_start(bytes, at + width) {
        let empty = bytes.get(at + width..next).is_none_or(|gap| gap.iter().all(|b| *b == 0));
        if empty {
            return Err(format!("NAL unit {units} of the coded picture is empty"));
        }
        at = next;
        width = next_width;
        units += 1;
    }
    if bytes.len().saturating_sub(at + width) < 1 {
        return Err(format!("the last of {units} NAL units of the coded picture holds no header"));
    }
    Ok(())
}

/// The compressed picture the driver wrote, as Annex-B NAL units with no header of our own.
///
/// Owned bytes, so it outlives the mapping it was copied out of — which is what makes it safe to
/// hand to a muxer.
pub struct CodedPicture {
    bytes: Vec<u8>,
    /// `VA_CODED_BUF_STATUS_*` for each segment (`va.h:3897-3933`). The low byte is the picture's
    /// average QP — the driver's own statement of what it encoded at, which is not the QP that was
    /// asked for and is worth having both of.
    status: Vec<u32>,
    /// How many segments the driver reported the picture in.
    segments: usize,
}

impl CodedPicture {
    /// The NAL units: start-code-prefixed, exactly as the driver wrote them.
    ///
    /// With [`PACKED_HEADER_SEQUENCE`] and [`PACKED_HEADER_PICTURE`] set, the driver puts the SPS
    /// and the PPS in this buffer ahead of the slice, so the bytes are a complete Annex-B access
    /// unit — which is what is read back out of here rather than a parameter set this backend
    /// assembled.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The per-segment status words the driver wrote, in order.
    pub fn status(&self) -> &[u32] {
        &self.status
    }

    /// How many bytes the driver wrote.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// How many segments the picture came back in. More than one is what a driver does when it
    /// packs the parameter sets as separate units rather than ahead of the slice.
    pub fn segments(&self) -> usize {
        self.segments
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let api = Arc::clone(&self.display.api);
        let _gate = self.display.gate();
        // libva's own order: surfaces may only be destroyed after the contexts that use them.
        if let Some(context) = self.context.take() {
            // SAFETY: `context` is this session's own context, taken out of the option so no other
            // reference to it exists, and this is its matching destroy.
            unsafe { (api.destroy_context)(self.display.dpy, context) };
        }
        if !self.surfaces.is_empty() {
            let count = c_int::try_from(self.surfaces.len()).unwrap_or(0);
            // SAFETY: `self.surfaces` is the array of surface ids this session created, `count` is its
            // length (zero on the conversion that cannot represent it, which destroys none), and this
            // is their one destroy.
            unsafe { (api.destroy_surfaces)(self.display.dpy, self.surfaces.as_mut_ptr(), count) };
        }
        if self.config != 0 {
            // SAFETY: `config` is this session's own config id, created in `Session::new` and not
            // destroyed on any earlier path, and this is its one destroy.
            unsafe { (api.destroy_config)(self.display.dpy, self.config) };
        }
    }
}

/// A buffer submitted to a decode, destroyed when it is dropped.
///
/// Destroying a buffer is how libva is told a decode is finished with it, and a buffer that outlives
/// its decode holds driver memory. `Drop` rather than a call the decode loop has to remember, so
/// that an error between `vaCreateBuffer` and `vaRenderPicture` cannot leak one.
pub struct Buffer {
    display: Arc<Display>,
    id: VABufferID,
    /// `id == 0` means this value owns no libva buffer — a placeholder that releases nothing.
    /// `VA_INVALID_ID` would not do: a driver is free to have made a buffer with that id, and
    /// destroying it would destroy something this value does not own.
    owns: bool,
}

impl Buffer {
    /// The buffer id `vaRenderPicture` takes.
    pub fn id(&self) -> u32 {
        self.id
    }

    /// A buffer for an id this display created, destroyed when it is dropped.
    fn owned(display: Arc<Display>, id: VABufferID) -> Self {
        Self { display, id, owns: true }
    }

    /// A value that owns nothing, so it can be a field before the real buffer exists. `Encoder`
    /// needs one because its fields are initialised before the libva calls that can fail, and a
    /// field that could not be filled has to be *some* valid `Buffer`.
    fn placeholder(display: Arc<Display>) -> Self {
        Self { display, id: 0, owns: false }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if !self.owns {
            return;
        }
        self.owns = false;
        let _gate = self.display.gate();
        // SAFETY: `id` is the buffer this value owns and was created by the same display, which it
        // holds a reference to; `owns` is cleared first so this can only run once; this is its one
        // destroy.
        unsafe { (self.display.api.destroy_buffer)(self.display.dpy, self.id) };
    }
}

/// A decoded picture's pixels, mapped and readable — what a readback returns.
///
/// The mapped region belongs to the driver and is only valid until this is dropped, which is why it
/// is lent rather than handed out: [`Readback::with_bytes`] borrows it for the duration of a
/// closure, so nothing can hold a slice past the unmap. Dropping unmaps the buffer and destroys the
/// image, on every path: a decode that fails between mapping and reading leaks neither.
pub struct Readback {
    /// Kept alive so the display — and through it the libva handle the mapping belongs to — outlives
    /// the unmap in `Drop`.
    display: Arc<Display>,
    /// The driver's own description of the image, which the readback gate reads.
    image: VaImage,
    /// The mapped buffer, or null when `vaMapBuffer` has not been called (it cannot have, or `map`
    /// would have failed).
    data: *mut c_void,
    mapped: bool,
}

impl Readback {
    /// Map `image`'s buffer. `derived` says whether the image came from `vaDeriveImage` or from
    /// `vaCreateImage`, which is what [`destroy_unread_image`] and `Drop` have to match.
    fn map(display: Arc<Display>, image: VaImage, derived: bool) -> Result<Self, String> {
        let mut data: *mut c_void = std::ptr::null_mut();
        let status = {
            let _gate = display.gate();
            // SAFETY: `image.buf` is the buffer of an image libva created on this display, and
            // `data` is a live local for the address libva writes into it.
            unsafe { (display.api.map_buffer)(display.dpy, image.buf, &mut data) }
        };
        if status != OK {
            let api = &display.api;
            destroy_unread_image(&display, &image, derived);
            return Err(format!("vaMapBuffer failed: {}", api.status_text(status)));
        }
        if data.is_null() {
            // A successful map with no address is a driver that did not do what it said. There is
            // nothing to unmap, but the image still has to be destroyed — see `destroy_unread_image`
            // for why that is only for a derived one.
            destroy_unread_image(&display, &image, derived);
            return Err("vaMapBuffer succeeded but returned no address".into());
        }
        Ok(Self { display, image, data, mapped: true })
    }

    /// The image as the driver described it: its size, planes, pitches and offsets. This is what the
    /// readback gate reads before a byte is read out of the mapped buffer.
    pub fn image(&self) -> &VaImage {
        &self.image
    }

    /// Borrow the mapped buffer's two planes for the duration of `f`, as
    /// `(luma, chroma, stride, rows)`: a luma plane of `rows` rows of `stride` bytes, and a chroma
    /// plane of `rows / 2` rows of the same stride, which is what a biplanar 4:2:0 image is.
    ///
    /// This is the one place a driver-reported offset becomes an address a pixel read happens at, so
    /// every number in it is bounded against `data_size` — the only length libva reports for a mapped
    /// buffer — and both planes are required to be inside it. The closure cannot keep either slice: it
    /// returns nothing, so the borrow ends when it does and the unmap in `Drop` cannot pull the
    /// ground out from under a caller.
    ///
    /// `data_size` is what the caller checked against the picture geometry (see
    /// [`crate::vaapi::validate_image`]); this is the second gate, and it does not rely on that check
    /// having been made, because this is the one that indexes.
    pub fn with_mapped(&self, f: impl FnOnce(&[u8], &[u8], usize, usize)) {
        // SAFETY: `data` is what `vaMapBuffer` returned for this image's buffer and is still mapped
        // (only `Drop` unmaps it, and only once, with this value borrowed throughout), so the slice
        // below is `data_size` readable bytes; both slices handed to `f` are subranges of it, which is
        // what the `get` calls check, and neither outlives the borrow of `self`.
        let bytes = unsafe { std::slice::from_raw_parts(self.data.cast::<u8>(), self.image.data_size as usize) };
        let (Some(&luma_pitch), Some(&chroma_pitch), Some(&luma_at), Some(&chroma_at)) =
            (self.image.pitches.first(), self.image.pitches.get(1), self.image.offsets.first(), self.image.offsets.get(1))
        else {
            return f(&[], &[], 0, 0);
        };
        let (stride, chroma_stride, rows) = (luma_pitch as usize, chroma_pitch as usize, self.image.height as usize);
        let chroma_rows = rows.div_ceil(2);
        // Each plane's own pitch, not the luma's: reading the chroma plane at the luma pitch would
        // walk off it. But `biplanar::Biplanar` has a *single* stride for both planes, so a driver
        // that pitches them differently cannot be read through it at all — `validate_image` refuses
        // that image before this is reached, and this second check is what makes the refusal hold even
        // if a caller skipped it (an empty pair makes `to_frame` fail, which is an error rather than
        // a picture read at the wrong stride).
        if chroma_stride != stride {
            return f(&[], &[], 0, 0);
        }
        let (Some(luma_len), Some(chroma_len)) = (stride.checked_mul(rows), chroma_stride.checked_mul(chroma_rows)) else {
            return f(&[], &[], 0, 0);
        };
        let (Some(luma), Some(chroma)) =
            (bytes.get(luma_at as usize..).and_then(|b| b.get(..luma_len)), bytes.get(chroma_at as usize..).and_then(|b| b.get(..chroma_len)))
        else {
            return f(&[], &[], 0, 0);
        };
        f(luma, chroma, stride, rows)
    }
}

impl Drop for Readback {
    fn drop(&mut self) {
        let _gate = self.display.gate();
        if self.mapped {
            // SAFETY: `image.buf` is the buffer mapped by `Readback::map` and has not been unmapped
            // since (only this `Drop` unmaps it, and only once, with the value borrowed throughout), so
            // this is its one unmap.
            unsafe { (self.display.api.unmap_buffer)(self.display.dpy, self.image.buf) };
            self.mapped = false;
        }
        if let Some(destroy) = self.display.api.destroy_image {
            // SAFETY: `image.image_id` is the id libva wrote for this image, which this value owns:
            // either `vaDeriveImage` filled it in or `vaCreateImage` created it, and nothing else holds
            // it. This is its one destroy.
            unsafe { destroy(self.display.dpy, self.image.image_id) };
        }
    }
}

/// Destroy an image that was created but never read: the paths out of [`Readback::map`] where the map
/// failed, so that a failed readback leaks neither the image nor its buffer.
///
/// Only for a *derived* image. An image from `vaCreateImage` whose map failed is the caller's to
/// destroy — [`Session::read_indirect`] does, on its own error path — because libva has no way to
/// tell a caller that its own `vaCreateImage` id is no longer wanted, and destroying it here would be
/// a double destroy.
fn destroy_unread_image(display: &Arc<Display>, image: &VaImage, derived: bool) {
    if !derived {
        return;
    }
    let _gate = display.gate();
    if let Some(destroy) = display.api.destroy_image {
        // SAFETY: `image.image_id` is the id `vaDeriveImage` wrote into `image`, the image was never
        // mapped and so never unmapped, and nothing else holds the id, so this is its one destroy.
        unsafe { destroy(display.dpy, image.image_id) };
    }
}

// ---------------------------------------------------------------------------------------------
// Tests: the hand-declared ABI against `/usr/include/va/`, and against the library that is
// installed. Mirrors `nvenc/abi_tests.rs`, except that the header is present on the machine that
// runs this, so the layout can be compared with libva itself.
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
#[path = "abi_guard.rs"]
mod abi_guard;

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// The loader and the mandatory `vaInitialize` work on a machine that has them: both libraries
    /// open, every required symbol resolves, and the display reports a version and a profile list.
    /// What a profile then supports is a separate question, answered by the decoder's own tests.
    #[test]
    fn the_libraries_load_and_the_display_initializes() {
        let Ok(api) = load() else {
            eprintln!("SKIPPED: no libva");
            return;
        };
        let Ok(d) = Display::new(&api) else {
            eprintln!("SKIPPED: no VA display");
            return;
        };
        // The loader opened both libraries, found every required symbol and initialised the
        // display, which is what this test is for. Beyond that, only what this backend promises:
        // a display reports its version and answers a profile query. Whether any profile offers a
        // video decoder is a capability question, not a load question — a driver that offers only
        // encode entry points is a machine with no hardware decoder, which is a legitimate answer
        // (see the module doc), not a failure.
        let (major, _) = d.va_version();
        assert!(major >= 1, "vaInitialize reported no libva version");
        assert!(!d.profiles().expect("vaQueryConfigProfiles").is_empty(), "the driver supports no profile at all");
    }

    /// `OK` is success, libva has words for it, and the load-free path a [`Display`] error uses
    /// says the same thing as the public `status` — otherwise decorating an error there would
    /// quietly produce a different message than everything else.
    #[test]
    fn va_status_zero_is_success() {
        assert_eq!(OK, 0);
        assert!(!status(OK).is_empty());
        let Ok(api) = load() else {
            eprintln!("SKIPPED: no libva");
            return;
        };
        assert_eq!(api.status_text(OK), status(OK), "status() and Api::status_text disagree");
        assert!(!api.status_text(OK).is_empty());
        // A code libva does not know still names itself rather than coming back empty.
        assert!(!api.status_text(-999).is_empty(), "an unknown status is not described");
    }

    /// Every libva 2.24 constant this module carries, pinned so a wrong one fails on every machine
    /// rather than only where a driver happens to be installed. The profile numbers are the ones
    /// that matter most: a wrong one does not fail a call, it configures the wrong codec, and the
    /// symptom is "this stream decodes in software" rather than a test failure.
    #[test]
    fn the_format_constants_are_the_ones_libva_uses() {
        assert_eq!(YUV420, 0x0000_0001);
        assert_eq!(YUV420_10, 0x0000_0100, "VA_RT_FORMAT_YUV420_10 is 0x100");
        assert_eq!(VA_PADDING_LOW, 4);
        assert_eq!(VA_PADDING_MEDIUM, 8);
        assert_eq!(PICTURE_PARAMETER_BUFFER, 0);
        assert_eq!(IQ_MATRIX_BUFFER, 1, "VAIQMatrixBufferType");
        assert_eq!(SLICE_PARAMETER_BUFFER, 4);
        assert_eq!(SLICE_DATA_BUFFER, 5);
        assert_eq!(VLD, 1);
        assert_eq!(ENC_SLICE, 6, "VAEntrypointEncSlice");
        assert_eq!(H264_MAIN, 6, "VAProfileH264Main");
        assert_eq!(H264_HIGH, 7, "VAProfileH264High");
        assert_eq!(H264_CONSTRAINED_BASELINE, 13, "VAProfileH264ConstrainedBaseline");
        assert_eq!(HEVC_MAIN, 17, "VAProfileHEVCMain");
        assert_eq!(HEVC_MAIN_10, 18, "VAProfileHEVCMain10");
        assert_eq!(VP9_PROFILE_0, 19, "VAProfileVP9Profile0");
        assert_eq!(VP9_PROFILE_1, 20, "VAProfileVP9Profile1");
        assert_eq!(VP9_PROFILE_2, 21, "VAProfileVP9Profile2");
        assert_eq!(VP9_PROFILE_3, 22, "VAProfileVP9Profile3");
        assert_eq!(AV1_MAIN, 32, "VAProfileAV1Profile0");
        assert_eq!(OPERATION_FAILED, 0x1, "VA_STATUS_ERROR_OPERATION_FAILED");
        assert_eq!(HW_BUSY, 0x22, "VA_STATUS_ERROR_HW_BUSY");
        assert_eq!(SURFACE_READY, 0x4, "VASurfaceReady");
        assert_eq!(PROGRESSIVE, 0x1, "VA_PROGRESSIVE");
        // The IQ matrix record's shape, which is what `h264_iq_matrix` fills: 96 bytes of 4x4
        // lists, 128 of 8x8, then the reserved tail — 240 in total, and the driver reads a
        // submission of any other length as something else entirely.
        assert_eq!(size_of::<VaIqMatrixBufferH264>(), 240, "VAIQMatrixBufferH264 is 240 bytes");
        assert_eq!(offset_of!(VaIqMatrixBufferH264, scaling_list4x4), 0);
        assert_eq!(offset_of!(VaIqMatrixBufferH264, scaling_list8x8), 96, "after the six 4x4 lists");
        assert_eq!(offset_of!(VaIqMatrixBufferH264, va_reserved), 224, "and the two 8x8 luma lists are 128 bytes");
        assert_eq!(NV12_BPP, 12, "libva reports 12 bits per pixel for NV12, not 8");
        assert_eq!(P010_BPP, 24, "and 24 for P010, not 10");
        assert_eq!(NV12_FOURCC, 0x3231_564e, "VA_FOURCC_NV12");
        assert_eq!(P010_FOURCC, 0x3031_3050, "VA_FOURCC_P010");
        assert_eq!(PICTURE_HEVC_LONG_TERM_REFERENCE, 0x8, "VA_PICTURE_HEVC_LONG_TERM_REFERENCE");
        assert_eq!(PICTURE_HEVC_RPS_ST_CURR_BEFORE, 0x10);
        assert_eq!(PICTURE_HEVC_RPS_ST_CURR_AFTER, 0x20);
        assert_eq!(PICTURE_HEVC_RPS_LT_CURR, 0x40);
    }

    /// The layouts of the two attribute structures a decoder passes a null array of: libva takes
    /// `(attrib_list, num_attribs)` and the array's type is in the signature, so a caller that passes
    /// nothing still needs the struct to exist and be the right size.
    ///
    /// **`VASurfaceAttrib` is 24 bytes, not 16.** `VAGenericValue` (`va.h:1662`) is a *struct* — a
    /// `VAGenericValueType` and then the union the header puts inside it — and it was declared here
    /// as that inner union alone, which made every surface attribute record eight bytes short. Nothing
    /// noticed because the only array this backend passes is a null pointer; an array's stride is
    /// what a driver walks, so a caller that did pass one would have had every attribute after the
    /// first read from the wrong place.
    #[test]
    fn the_attribute_records_match_the_header() {
        assert_eq!(size_of::<VaConfigAttrib>(), 8, "VAConfigAttrib (va.h:1072)");
        assert_eq!(offset_of!(VaConfigAttrib, value), 4);
        assert_eq!(size_of::<VaSurfaceAttrib>(), 24, "VASurfaceAttrib (va.h:1742)");
        assert_eq!(offset_of!(VaSurfaceAttrib, flags), 4);
        assert_eq!(offset_of!(VaSurfaceAttrib, value), 8, "the VAGenericValue tail");
        assert_eq!(size_of::<VaGenericValue>(), 16, "VAGenericValue (va.h:1662): a type word and a union");
        assert_eq!(offset_of!(VaGenericValue, value), 8, "the union inside it");
    }

    /// A driver that will not decode into a derived image says so with
    /// `VA_STATUS_ERROR_OPERATION_FAILED`, and that is the only status this backend reads a surface
    /// back the indirect way for. A driver that fails for any other reason is an error, not a reason
    /// to copy the picture a second way: the distinction is what keeps a broken driver from being
    /// quietly worked around.
    #[test]
    fn only_operation_failed_is_treated_as_a_derive_failure() {
        assert_eq!(OPERATION_FAILED, 1);
        assert_ne!(OPERATION_FAILED, OK);
        assert_ne!(OPERATION_FAILED, HW_BUSY);
    }

    /// The picture- and slice-parameter records, as libva 2.24's headers declare them, so a field
    /// declared with the wrong type or width — which the driver would read as a neighbouring field's
    /// value — fails here rather than as corruption on screen. The sizes come from the headers on
    /// this machine; the offsets are the ones the C declarations imply, and every struct's trailing
    /// reserved field is checked too, since that is what makes each record a stable size.
    #[test]
    fn the_picture_parameter_records_match_the_header() {
        // H.264 (va.h:3572-3721)
        assert_eq!(size_of::<VaPictureH264>(), 36, "VAPictureH264 (va.h:3572)");
        assert_eq!(offset_of!(VaPictureH264, frame_idx), 4);
        assert_eq!(offset_of!(VaPictureH264, flags), 8);
        assert_eq!(offset_of!(VaPictureH264, top_field_order_cnt), 12);
        assert_eq!(offset_of!(VaPictureH264, bottom_field_order_cnt), 16);
        assert_eq!(offset_of!(VaPictureH264, va_reserved), 20);
        assert_eq!(size_of::<VaPictureParameterBufferH264>(), 672, "VAPictureParameterBufferH264 (va.h:3599)");
        assert_eq!(offset_of!(VaPictureParameterBufferH264, curr_pic), 0);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, reference_frames), 36);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, picture_width_in_mbs_minus1), 612);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, picture_height_in_mbs_minus1), 614);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, bit_depth_luma_minus8), 616);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, num_ref_frames), 618);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, padding_after_num_ref_frames), 619, "the C layout's hole before seq_fields");
        assert_eq!(offset_of!(VaPictureParameterBufferH264, seq_fields), 620);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, num_slice_groups_minus1), 624);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, slice_group_change_rate_minus1), 626);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, pic_init_qp_minus26), 628);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, second_chroma_qp_index_offset), 631);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, pic_fields), 632);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, frame_num), 636);
        assert_eq!(offset_of!(VaPictureParameterBufferH264, padding_after_frame_num), 638, "the C layout's hole before va_reserved");
        assert_eq!(offset_of!(VaPictureParameterBufferH264, va_reserved), 640);
        assert_eq!(size_of::<VaSliceParameterBufferH264>(), 3128, "VASliceParameterBufferH264 (va.h:3664)");
        assert_eq!(offset_of!(VaSliceParameterBufferH264, slice_data_bit_offset), 12);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, first_mb_in_slice), 14);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, slice_type), 16);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, slice_beta_offset_div2), 24);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, padding_before_ref_pic_list0), 25, "the C layout's hole before RefPicList0[]");
        assert_eq!(offset_of!(VaSliceParameterBufferH264, ref_pic_list0), 28);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, ref_pic_list1), 1180);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, luma_log2_weight_denom), 2332);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, padding_before_luma_weight_l0), 2335, "the hole before luma_weight_l0[]");
        assert_eq!(offset_of!(VaSliceParameterBufferH264, luma_offset_l0), 2400);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, chroma_weight_l0_flag), 2464);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, padding_before_chroma_weight_l0), 2465, "the hole before chroma_weight_l0[][]");
        assert_eq!(offset_of!(VaSliceParameterBufferH264, chroma_offset_l0), 2594);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, luma_weight_l1_flag), 2722);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, padding_before_luma_weight_l1), 2723, "the hole before luma_weight_l1[]");
        assert_eq!(offset_of!(VaSliceParameterBufferH264, chroma_offset_l1), 2982);
        assert_eq!(offset_of!(VaSliceParameterBufferH264, padding_before_va_reserved), 3110, "the hole before va_reserved");
        assert_eq!(offset_of!(VaSliceParameterBufferH264, va_reserved), 3112);
        // HEVC (va.h:5292, va_dec_hevc.h:57-479)
        assert_eq!(size_of::<VaPictureHevc>(), 28, "VAPictureHEVC (va.h:5292)");
        assert_eq!(offset_of!(VaPictureHevc, pic_order_cnt), 4);
        assert_eq!(offset_of!(VaPictureHevc, flags), 8);
        assert_eq!(offset_of!(VaPictureHevc, va_reserved), 12);
        assert_eq!(size_of::<VaPictureParameterBufferHevc>(), 604, "VAPictureParameterBufferHEVC (va_dec_hevc.h:57)");
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, reference_frames), 28);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, pic_width_in_luma_samples), 448);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, pic_height_in_luma_samples), 450);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, pic_fields), 452);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, sps_max_dec_pic_buffering_minus1), 456);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, log2_min_luma_coding_block_size_minus3), 461);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, init_qp_minus26), 469);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, log2_parallel_merge_level_minus2), 473);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, column_width_minus1), 476);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, row_height_minus1), 514);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, slice_parsing_fields), 556);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, log2_max_pic_order_cnt_lsb_minus4), 560);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, num_extra_slice_header_bits), 567);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, st_rps_bits), 568);
        assert_eq!(offset_of!(VaPictureParameterBufferHevc, va_reserved), 572);
        assert_eq!(size_of::<VaSliceParameterBufferHevc>(), 264, "VASliceParameterBufferHEVC (va_dec_hevc.h:358)");
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, slice_data_byte_offset), 12);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, slice_segment_address), 16);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, ref_pic_list), 20);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, padding_before_long_slice_flags), 50, "the C layout's hole before LongSliceFlags");
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, long_slice_flags), 52);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, collocated_ref_idx), 56);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, slice_qp_delta), 59);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, slice_beta_offset_div2), 62);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, luma_log2_weight_denom), 64);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, chroma_offset_l1), 216);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, five_minus_max_num_merge_cand), 246);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, padding_before_num_entry_point_offsets), 247, "the hole before num_entry_point_offsets");
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, slice_data_num_emu_prevn_bytes), 252);
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, padding_before_va_reserved), 254, "the hole before va_reserved");
        assert_eq!(offset_of!(VaSliceParameterBufferHevc, va_reserved), 256);
        // VP9 (va_dec_vp9.h:58). **92 bytes, and two of the offsets are the ones a plausible reading of
        // the header gets wrong.** `first_partition_size` is a `uint16_t` at 46, not at 45: byte 45 is
        // the hole the 2-byte alignment makes in front of it, so `frame_header_length_in_bytes` ends at
        // 44 and the compressed-header count starts at 46. And `va_reserved[VA_PADDING_MEDIUM]` is 32
        // bytes at 60 — 60 is already a multiple of 4, so `bit_depth` at 59 is followed by no
        // padding at all, and the record is 92 bytes rather than the 96 a "round up to a multiple of
        // 8" reading would give. Both are measured against `/usr/include/va/va_dec_vp9.h` by a C
        // compiler in `every_declaration_matches_what_a_c_compiler_measures`, so the numbers below are
        // not the ones Rust and the header happen to agree on by accident.
        assert_eq!(size_of::<VaPictureParameterBufferVp9>(), 92, "VADecPictureParameterBufferVP9 (va_dec_vp9.h:58)");
        assert_eq!(offset_of!(VaPictureParameterBufferVp9, frame_width), 0);
        assert_eq!(offset_of!(VaPictureParameterBufferVp9, frame_height), 2);
        assert_eq!(offset_of!(VaPictureParameterBufferVp9, reference_frames), 4, "the eight surfaces are 32 bytes from offset 4");
        assert_eq!(offset_of!(VaPictureParameterBufferVp9, pic_fields), 36);
        assert_eq!(offset_of!(VaPictureParameterBufferVp9, filter_level), 40);
        assert_eq!(offset_of!(VaPictureParameterBufferVp9, sharpness_level), 41);
        assert_eq!(offset_of!(VaPictureParameterBufferVp9, log2_tile_rows), 42);
        assert_eq!(offset_of!(VaPictureParameterBufferVp9, log2_tile_columns), 43);
        assert_eq!(offset_of!(VaPictureParameterBufferVp9, frame_header_length_in_bytes), 44);
        assert_eq!(offset_of!(VaPictureParameterBufferVp9, padding_before_first_partition_size), 45, "the C layout's hole before first_partition_size");
        assert_eq!(offset_of!(VaPictureParameterBufferVp9, first_partition_size), 46, "the uint16_t is 2-aligned, not 1-aligned");
        assert_eq!(offset_of!(VaPictureParameterBufferVp9, mb_segment_tree_probs), 48);
        assert_eq!(offset_of!(VaPictureParameterBufferVp9, segment_pred_probs), 55);
        assert_eq!(offset_of!(VaPictureParameterBufferVp9, profile), 58);
        assert_eq!(offset_of!(VaPictureParameterBufferVp9, bit_depth), 59);
        assert_eq!(offset_of!(VaPictureParameterBufferVp9, va_reserved), 60, "bit_depth ends at 60, which is already 4-aligned");
        // The VP9 slice record (va_dec_vp9.h:204, :279). **316 bytes, made of eight 36-byte segments,
        // and every one of those numbers is measured, not read off the field list.** The header's
        // `VASegmentParameterVP9` is `uint16_t`, `uint8_t[4][2]`, four `int16_t` and a
        // `uint32_t[4]` — 2 + 8 + 8 + 16 = 34, so the struct is 36 because the four quant scales end
        // at 18, the `uint32_t` tail is 4-aligned and so begins at 20, and the **2 bytes the compiler
        // inserts at 18-19 are the header's alignment padding as much as ours** — which is why they
        // are named `padding_before_va_reserved` rather than left implicit. A declaration that packed
        // the quantizers into an array or dropped the reserved tail would shift all eight segments
        // and every field in them. The eight are 288 bytes at offset 12, and the reserved tail is 16
        // more.
        assert_eq!(size_of::<VaSegmentParameterVp9>(), 36, "VASegmentParameterVP9 (va_dec_vp9.h:204)");
        assert_eq!(offset_of!(VaSegmentParameterVp9, segment_flags), 0);
        assert_eq!(offset_of!(VaSegmentParameterVp9, filter_level), 2, "the uint16_t union leaves no hole");
        assert_eq!(offset_of!(VaSegmentParameterVp9, luma_ac_quant_scale), 10);
        assert_eq!(offset_of!(VaSegmentParameterVp9, luma_dc_quant_scale), 12);
        assert_eq!(offset_of!(VaSegmentParameterVp9, chroma_ac_quant_scale), 14);
        assert_eq!(offset_of!(VaSegmentParameterVp9, chroma_dc_quant_scale), 16);
        assert_eq!(offset_of!(VaSegmentParameterVp9, padding_before_va_reserved), 18, "chroma_dc_quant_scale ends at 18, and the uint32_t tail is 4-aligned");
        assert_eq!(
            size_of::<[u8; 2]>(),
            offset_of!(VaSegmentParameterVp9, va_reserved) - offset_of!(VaSegmentParameterVp9, padding_before_va_reserved),
            "the hole is exactly as wide as the padding that has to cover it — two bytes, not one"
        );
        assert_eq!(
            offset_of!(VaSegmentParameterVp9, va_reserved),
            20,
            "chroma_dc_quant_scale ends at 18, and 18 is 2-aligned, so the uint32_t tail starts at 20"
        );
        assert_eq!(size_of::<VaSliceParameterBufferVp9>(), 316, "VASliceParameterBufferVP9 (va_dec_vp9.h:279)");
        assert_eq!(offset_of!(VaSliceParameterBufferVp9, slice_data_size), 0);
        assert_eq!(offset_of!(VaSliceParameterBufferVp9, slice_data_offset), 4);
        assert_eq!(offset_of!(VaSliceParameterBufferVp9, slice_data_flag), 8);
        assert_eq!(offset_of!(VaSliceParameterBufferVp9, seg_param), 12, "eight 36-byte segments from offset 12");
        assert_eq!(offset_of!(VaSliceParameterBufferVp9, va_reserved), 300, "12 + 8 * 36");
    }

    /// **Every byte of every VP9 record belongs to a named field**, which is the property
    /// [`record_bytes`]'s safety argument rests on and the one no offset assertion above can see.
    ///
    /// The offsets above all agreed, byte for byte, with what a C compiler measures — and the record
    /// still had a hole. `chroma_dc_quant_scale` is an `i16` at 16 and `va_reserved` is a `[u32; 4]`
    /// at 20, so bytes 18-19 were alignment padding the *compiler* inserted and no field named: the
    /// sizes were right, the offsets were right, the C compiler agreed, and `#[derive(Default)]` and
    /// a struct literal both left those two bytes whatever was on the stack. Eight of those records
    /// are submitted for every VP9 picture, and the decode came out bit-exact on 24 of 24 —
    /// **because this driver ignores those bytes**, which is a fact about the driver and not about
    /// the record.
    ///
    /// So this test does not compare offsets against numbers. It lays the fields end to end and
    /// asserts they **tile the record**: no gap between two fields, no field starting inside the one
    /// before it, and no byte past the last field. A gap *is* the defect, named where it is, because a
    /// `padding_*` field is a named field like any other and is zeroed by `Default`.
    #[test]
    fn no_vp9_record_has_a_byte_no_field_names() {
        /// `fields` is `(name, offset, size)` in declaration order; `size` is `size_of::<T>()`.
        fn tiles_every_byte(record: &str, fields: &[(&str, usize, usize)], size: usize) {
            let mut at = 0usize;
            for (name, offset, width) in fields {
                assert_eq!(*offset, at, "{record}: {name} starts at {offset}, and the field before it ends at {at}");
                assert!(*width > 0, "{record}: {name} is {width} bytes wide");
                at += *width;
            }
            assert_eq!(at, size, "{record}: the named fields end at {at} and the record is {size} bytes");
        }
        tiles_every_byte(
            "VADecPictureParameterBufferVP9",
            &[
                ("frame_width", offset_of!(VaPictureParameterBufferVp9, frame_width), size_of::<u16>()),
                ("frame_height", offset_of!(VaPictureParameterBufferVp9, frame_height), size_of::<u16>()),
                ("reference_frames", offset_of!(VaPictureParameterBufferVp9, reference_frames), size_of::<[VASurfaceID; 8]>()),
                ("pic_fields", offset_of!(VaPictureParameterBufferVp9, pic_fields), size_of::<u32>()),
                ("filter_level", offset_of!(VaPictureParameterBufferVp9, filter_level), size_of::<u8>()),
                ("sharpness_level", offset_of!(VaPictureParameterBufferVp9, sharpness_level), size_of::<u8>()),
                ("log2_tile_rows", offset_of!(VaPictureParameterBufferVp9, log2_tile_rows), size_of::<u8>()),
                ("log2_tile_columns", offset_of!(VaPictureParameterBufferVp9, log2_tile_columns), size_of::<u8>()),
                ("frame_header_length_in_bytes", offset_of!(VaPictureParameterBufferVp9, frame_header_length_in_bytes), size_of::<u8>()),
                ("padding_before_first_partition_size", offset_of!(VaPictureParameterBufferVp9, padding_before_first_partition_size), size_of::<u8>()),
                ("first_partition_size", offset_of!(VaPictureParameterBufferVp9, first_partition_size), size_of::<u16>()),
                ("mb_segment_tree_probs", offset_of!(VaPictureParameterBufferVp9, mb_segment_tree_probs), 7),
                ("segment_pred_probs", offset_of!(VaPictureParameterBufferVp9, segment_pred_probs), 3),
                ("profile", offset_of!(VaPictureParameterBufferVp9, profile), size_of::<u8>()),
                ("bit_depth", offset_of!(VaPictureParameterBufferVp9, bit_depth), size_of::<u8>()),
                ("va_reserved", offset_of!(VaPictureParameterBufferVp9, va_reserved), VA_PADDING_MEDIUM * size_of::<u32>()),
            ],
            size_of::<VaPictureParameterBufferVp9>(),
        );
        tiles_every_byte(
            "VASegmentParameterVP9",
            &[
                ("segment_flags", offset_of!(VaSegmentParameterVp9, segment_flags), size_of::<u16>()),
                ("filter_level", offset_of!(VaSegmentParameterVp9, filter_level), 4 * 2),
                ("luma_ac_quant_scale", offset_of!(VaSegmentParameterVp9, luma_ac_quant_scale), size_of::<i16>()),
                ("luma_dc_quant_scale", offset_of!(VaSegmentParameterVp9, luma_dc_quant_scale), size_of::<i16>()),
                ("chroma_ac_quant_scale", offset_of!(VaSegmentParameterVp9, chroma_ac_quant_scale), size_of::<i16>()),
                ("chroma_dc_quant_scale", offset_of!(VaSegmentParameterVp9, chroma_dc_quant_scale), size_of::<i16>()),
                ("padding_before_va_reserved", offset_of!(VaSegmentParameterVp9, padding_before_va_reserved), size_of::<[u8; 2]>()),
                ("va_reserved", offset_of!(VaSegmentParameterVp9, va_reserved), VA_PADDING_LOW * size_of::<u32>()),
            ],
            size_of::<VaSegmentParameterVp9>(),
        );
        tiles_every_byte(
            "VASliceParameterBufferVP9",
            &[
                ("slice_data_size", offset_of!(VaSliceParameterBufferVp9, slice_data_size), size_of::<u32>()),
                ("slice_data_offset", offset_of!(VaSliceParameterBufferVp9, slice_data_offset), size_of::<u32>()),
                ("slice_data_flag", offset_of!(VaSliceParameterBufferVp9, slice_data_flag), size_of::<u32>()),
                ("seg_param", offset_of!(VaSliceParameterBufferVp9, seg_param), 8 * size_of::<VaSegmentParameterVp9>()),
                ("va_reserved", offset_of!(VaSliceParameterBufferVp9, va_reserved), VA_PADDING_LOW * size_of::<u32>()),
            ],
            size_of::<VaSliceParameterBufferVp9>(),
        );
    }

    /// The `pic_fields` word is a C bitfield, and its layout is not a choice this crate makes: the
    /// header declares its members least-significant-first on this target, and a word packed in any
    /// other order is a *different picture*, not a different build.
    ///
    /// So this reads the bit positions out of `/usr/include/va/va_dec_vp9.h` rather than out of a
    /// table someone typed, and compares them against the packing function that builds the word.
    /// A record whose field order drifts from the header's fails here, where the message names the
    /// member, instead of as a subtly wrong picture on screen.
    ///
    /// It is skipped, with a note, where the headers are not installed — which is every machine that
    /// cannot run this backend at all, so it is an assertion and not a courtesy. (`abi_guard`
    /// measures the *sizes* the same way; a bitfield's positions are not a `sizeof`, so nothing else
    /// here covers them.)
    #[test]
    fn the_vp9_pic_fields_word_is_the_one_the_header_declares() {
        let header = std::path::Path::new("/usr/include/va/va_dec_vp9.h");
        let Ok(text) = std::fs::read_to_string(header) else {
            eprintln!("SKIPPED: no {}, so there is no header to read the bit positions from", header.display());
            return;
        };
        // Every `uint32_t <name> : <width>;` inside the `pic_fields` union. Only that union is
        // scanned, so `VASegmentParameterVP9`'s own `uint16_t` bitfields below are not mistaken for
        // it — the two are different records and their members have the same names' shapes for
        // different reasons.
        let Some((union, _)) = text.split_once("union {").and_then(|(_, rest)| rest.split_once("pic_fields")) else {
            panic!("va_dec_vp9.h has no pic_fields union to read bit positions from");
        };
        let union = union.split_once("} bits;").map_or(union, |(bits, _)| bits);

        let mut declared = Vec::new();
        for line in union.lines() {
            let line = line.split_once("uint32_t").map_or("", |(_, rest)| rest);
            let Some((name, width)) = line.split_once(':') else { continue };
            let name = name.trim();
            let Some(width) = width.trim().strip_suffix(';').and_then(|w| w.trim().parse::<u32>().ok()) else { continue };
            declared.push((name, width));
        }
        assert!(!declared.is_empty(), "no `uint32_t name : width;` lines were read out of the pic_fields union, so nothing was checked");

        // The order the header declares them in is the order they occupy the word on this target, so
        // the expected positions are a prefix sum of the declared widths — derived, not transcribed.
        let mut at = 0u32;
        let mut expected: Vec<(&str, u32, u32)> = Vec::new();
        for (name, width) in &declared {
            expected.push((name, at, *width));
            at += *width;
        }
        assert_eq!(at, 32, "the header's members fill all 32 bits, so no bit of the word is unnamed");

        assert_eq!(crate::vaapi::picture::vp9_pic_fields_positions(), expected, "the bit positions `vp9_pic_fields` packs by do not match va_dec_vp9.h:83-140");
    }

    /// Every byte of a record is a named field, so no byte of a default record is indeterminate —
    /// which is what lets `record_bytes` hand them to a driver as a `&[u8]`. The C layout's
    /// alignment holes are declared as `padding_*` fields; this checks that every default record is
    /// all zeroes, so a field added without a value, or a padding field that stopped being one,
    /// shows up here.
    #[test]
    fn a_default_record_is_every_byte_zero() {
        for bytes in [
            record_bytes(&VaPictureH264::default()),
            record_bytes(&VaPictureParameterBufferH264::default()),
            record_bytes(&VaSliceParameterBufferH264::default()),
            record_bytes(&VaPictureHevc::default()),
            record_bytes(&VaPictureParameterBufferHevc::default()),
            record_bytes(&VaSliceParameterBufferHevc::default()),
        ] {
            assert!(!bytes.is_empty());
            assert!(
                bytes.iter().all(|&b| b == 0),
                "a default record has a non-zero byte: {} of {}",
                bytes.iter().position(|&b| b != 0).unwrap_or(0),
                bytes.len()
            );
        }
    }

    /// A record read back out of the bytes `picture.rs` built is the same record it built, and a
    /// buffer too short for one is refused rather than read past. The two sides only agree if
    /// `record_bytes` and `record_of` describe the same layout, which is what makes them a pair.
    #[test]
    fn a_record_round_trips_through_its_bytes() {
        let picture = VaPictureParameterBufferH264 {
            curr_pic: VaPictureH264 {
                picture_id: 7,
                frame_idx: 3,
                flags: PICTURE_H264_SHORT_TERM_REFERENCE,
                top_field_order_cnt: 11,
                bottom_field_order_cnt: 22,
                ..Default::default()
            },
            picture_width_in_mbs_minus1: 39,
            picture_height_in_mbs_minus1: 22,
            num_ref_frames: 4,
            seq_fields: 0x1234_5678,
            pic_fields: 0x9abc_def0,
            frame_num: 3,
            ..Default::default()
        };
        let bytes = record_bytes(&picture);
        assert_eq!(bytes.len(), size_of::<VaPictureParameterBufferH264>());
        assert_eq!(picture_params_h264(&bytes), Some(picture), "the record reads back as the one built");

        // A short buffer is `None`, and a longer one is still the record: a driver reads a prefix.
        assert_eq!(picture_params_h264(&bytes[..size_of::<VaPictureParameterBufferH264>() - 1]), None, "a byte short");
        assert_eq!(picture_params_h264(&[]), None);
        assert_eq!(picture_params_h264(&bytes[..10]), None, "an empty slice of it");
        assert_eq!(slice_params_h264(&bytes), None, "an H.264 picture record read as a slice record");

        // The four records are four different sizes. A buffer shorter than the record is refused; a longer
        // one still reads as the record at its front, which is why the caller has to know which
        // record it built rather than letting a reader guess.
        assert_eq!(record_bytes(&VaPictureParameterBufferH264::default()).len(), size_of::<VaPictureParameterBufferH264>());
        assert_eq!(record_bytes(&VaSliceParameterBufferH264::default()).len(), size_of::<VaSliceParameterBufferH264>());
        assert_eq!(record_bytes(&VaPictureParameterBufferHevc::default()).len(), size_of::<VaPictureParameterBufferHevc>());
        assert_eq!(record_bytes(&VaSliceParameterBufferHevc::default()).len(), size_of::<VaSliceParameterBufferHevc>());
        assert_eq!(slice_params_h264(&bytes), None, "an H.264 picture record is not long enough to be a slice record");
        let extra = [bytes.as_slice(), &[0u8; 8]].concat();
        assert_eq!(picture_params_h264(&extra).expect("a record at the front of a longer buffer"), picture, "trailing bytes are ignored");
    }

    /// The layout as `/usr/include/va/va.h` has it, so a field added or moved here is a deliberate,
    /// reviewed change rather than a silent ABI change.
    #[test]
    fn va_image_layout_matches_the_header() {
        assert_eq!(size_of::<VaImage>(), 120, "VAImage size (va.h:4734)");
        assert_eq!(size_of::<VaImageFormat>(), 48, "VAImageFormat size (va.h:4717)");
        assert_eq!(offset_of!(VaImage, format), 4, "VAImage.format");
        assert_eq!(offset_of!(VaImage, buf), 52, "VAImage.buf");
        assert_eq!(offset_of!(VaImage, width), 56, "VAImage.width");
        assert_eq!(offset_of!(VaImage, height), 58, "VAImage.height");
        assert_eq!(offset_of!(VaImage, data_size), 60, "VAImage.data_size");
        assert_eq!(offset_of!(VaImage, num_planes), 64, "VAImage.num_planes");
        assert_eq!(offset_of!(VaImage, pitches), 68, "VAImage.pitches");
        assert_eq!(offset_of!(VaImage, offsets), 80, "VAImage.offsets");
        assert_eq!(offset_of!(VaImage, num_palette_entries), 92, "VAImage.num_palette_entries");
        assert_eq!(offset_of!(VaImage, entry_bytes), 96, "VAImage.entry_bytes");
        assert_eq!(offset_of!(VaImage, component_order), 100, "VAImage.component_order");
    }

    /// The layout against the library itself: the driver fills an image and the plane geometry it
    /// writes has to read back as NV12 planes inside the size it reported.
    #[test]
    fn va_image_plane_geometry_comes_from_the_driver() {
        let Ok(api) = load() else {
            eprintln!("SKIPPED: no libva");
            return;
        };
        let Ok(d) = Display::new(&api) else {
            eprintln!("SKIPPED: no VA display");
            return;
        };
        let Ok(ran) = d.probe_image_layout() else {
            eprintln!("SKIPPED: the driver will not create a probe image");
            return;
        };
        assert!(ran, "the plane geometry libva wrote is not where this module reads it");
    }

    /// The profile and entry point mapping, including the values the mapping must not lose.
    #[test]
    fn profiles_and_entrypoints_round_trip() {
        for p in [
            Profile::H264ConstrainedBaseline,
            Profile::H264Main,
            Profile::H264High,
            Profile::HevcMain,
            Profile::HevcMain10,
            Profile::Vp9(0),
            Profile::Vp9(3),
            Profile::Av1Main,
            Profile::Other(1234),
        ] {
            assert_eq!(Profile::from_bits(p.bits()), p, "{p:?}");
        }
        assert_eq!(Profile::Vp9(0).bits(), 19, "VAProfileVP9Profile0");
        assert_eq!(Profile::Vp9(3).bits(), 22, "VAProfileVP9Profile3");
        assert_eq!(Profile::from_bits(45), Profile::Other(45), "a profile we do not name is kept");
        for e in [Entrypoint::Vld, Entrypoint::EncSlice, Entrypoint::Other(9)] {
            assert_eq!(Entrypoint::from_bits(e.bits()), e, "{e:?}");
        }
        for f in [RtFormat::Nv12, RtFormat::Yuv420_10, RtFormat::Other(0x1234)] {
            assert_eq!(RtFormat::from(f.bits()), f, "{f:?}");
        }
        assert_eq!(RtFormat::Nv12.bits(), YUV420);
        assert_eq!(RtFormat::Yuv420_10.bits(), YUV420_10);
    }

    /// A driver count is what sizes the array handed to it, so both ends are bounded.
    #[test]
    fn driver_counts_are_bounded() {
        assert_eq!(capacity(-1), MAX_DRIVER_ENTRIES, "a nonsense maximum still gets an array");
        assert_eq!(capacity(0), 1, "zero still leaves the driver somewhere to write");
        assert_eq!(capacity(12), 12);
        assert_eq!(capacity(c_int::MAX), MAX_DRIVER_ENTRIES, "an absurd maximum is capped");
    }

    /// Plane geometry that does not add up is refused: this is what a wrong field offset looks
    /// like, and what a hostile `data_size` would look like.
    #[test]
    fn plane_geometry_is_checked_not_trusted() {
        let mut image = VaImage { width: 64, height: 64, data_size: 64 * 64 * 3 / 2, num_planes: 1, pitches: [64, 0, 0], ..Default::default() };
        assert!(plane_geometry_is_consistent(&image, 64, 64));

        image.pitches = [64, 0, 0];
        image.offsets = [image.data_size - 64, 0, 0];
        assert!(!plane_geometry_is_consistent(&image, 64, 64), "a plane that ends outside data_size");

        image.offsets = [0, 0, 0];
        image.pitches = [u32::MAX, 0, 0];
        assert!(!plane_geometry_is_consistent(&image, 64, 64), "a pitch that overflows");

        image.pitches = [64, 0, 0];
        image.num_planes = 4;
        assert!(!plane_geometry_is_consistent(&image, 64, 64), "more planes than VAImage has");

        image.num_planes = 1;
        image.width = 32;
        assert!(!plane_geometry_is_consistent(&image, 64, 64), "a size we did not ask for");
    }

    /// The check that makes "the picture is at the start of the storage" something the readback
    /// **verifies** rather than assumes. Every rejection below is a case where the offset-0 window
    /// is not a whole run of Annex-B NAL units — which is what a driver that put the picture
    /// somewhere else, or that reported a length running past the picture, would look like.
    #[test]
    fn the_coded_picture_window_is_checked_as_a_whole_annex_b_run() {
        // What the driver writes, and what must be accepted: one slice, and three NAL units with
        // the parameter sets ahead of it.
        assert!(check_annex_b(&[0, 0, 1, 0x67, 1, 2]).is_ok());
        assert!(check_annex_b(&[0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 3, 0, 0, 1, 0x21, 9]).is_ok());
        // A zero byte ahead of a 3-byte start code is `trailing_zero_8bits`, not the start of it,
        // and must not be read as an empty NAL unit.
        assert!(check_annex_b(&[0, 0, 1, 0x65, 9, 0, 0, 0, 1, 0x41, 7]).is_ok());

        // **No start code at the start of the storage**: the picture is not where this backend
        // reads it, and returning these bytes would be returning something that decodes to noise.
        assert!(check_annex_b(&[0x42, 0, 0, 1, 0x65, 9]).is_err(), "a picture that does not begin at the start");
        assert!(check_annex_b(&[0x11; 64]).is_err(), "no start code anywhere");
        assert!(check_annex_b(&[]).is_err(), "no bytes at all");

        // A window too short to hold a start code and a NAL header.
        assert!(check_annex_b(&[0, 0, 1]).is_err(), "a start code with no NAL header");
        assert!(check_annex_b(&[0, 0, 1, 0x65, 9, 0, 0, 0, 1]).is_err(), "a trailing start code with no unit");

        // Two start codes with nothing but zero bits between them: the NAL unit between them is
        // empty, which a decoder would report and a muxer could not carry.
        assert!(check_annex_b(&[0, 0, 1, 0, 0, 0, 1, 0x41, 7]).is_err(), "an empty NAL unit");
    }

    /// The start-code scan itself: both widths, both offsets, and the fact that it finds a 4-byte
    /// code at its own offset rather than as the 3-byte code one byte in.
    #[test]
    fn annex_b_start_codes_are_found_at_both_widths_and_only_once() {
        let bytes = [0xaa, 0, 0, 0, 1, 0x67, 0, 0, 1, 0x68];
        assert_eq!(annex_b_start(&bytes, 0), Some((1, 4)), "the 4-byte code, at its own offset");
        // From one byte past it, the *3-byte* reading of the same four bytes is what a search from
        // the wrong place would find — which is why `check_annex_b` searches from `at + width`.
        assert_eq!(annex_b_start(&bytes, 2), Some((2, 3)));
        assert_eq!(annex_b_start(&bytes, 4), Some((6, 3)), "past the 4-byte code, the next code is found");
        assert_eq!(annex_b_start(&bytes, 6), Some((6, 3)));
        assert_eq!(annex_b_start(&bytes, 7), None, "nothing after the last start code");
        // A range too short to hold either code finds nothing rather than reading past the end.
        assert_eq!(annex_b_start(&[0, 0], 0), None);
        assert_eq!(annex_b_start(&[], 0), None);
    }
}
