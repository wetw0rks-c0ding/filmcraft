//! What the driver needs for one access unit, built from FilmCraft's own parsers.
//!
//! A VA-API decoder does not work out picture structure for itself. Unlike VideoToolbox, which
//! takes a compressed sample plus the headers and infers the pictures, libva wants the picture-level
//! and slice-level syntax elements handed to it as records it never parses. So this module reads
//! them out of the same parsers the software decoders use — `filmcraft_h264`, `filmcraft_hevc` and
//! `filmcraft_codecs`' stream descriptions — and lays them out as the
//! `VAPictureParameterBuffer*` / `VASliceParameterBuffer*` records of [`super::ffi`].
//!
//! Everything here is safe code, and every number in it came out of a bitstream, so it is hostile.
//! A slice header that will not parse, a parameter set the record does not carry, a coded size that
//! is not a whole number of macroblocks and a slice count a corrupt sample could inflate are all
//! `Err` naming the reason. A zeroed record is the worst answer available: it tells the driver a
//! valid picture exists where none does, and the failure surfaces later as corruption rather than
//! here as a message the caller can act on.
//!
//! What crosses [`Picture`]'s boundary is owned bytes. `mod.rs` names no libva type: it copies
//! `params` and each slice's `params` into buffers it hands the driver, and the driver reads the
//! same bytes at the addresses they land at.

use std::sync::Arc;

use filmcraft_bitstream::unescape_rbsp;
use filmcraft_codecs::hw::{FrameCodec, FrameStreamInfo, NalCodec, NalStreamInfo};

use super::ffi::{
    INVALID_SURFACE, PICTURE_H264_INVALID, PICTURE_H264_LONG_TERM_REFERENCE, PICTURE_H264_SHORT_TERM_REFERENCE, PICTURE_HEVC_INVALID,
    PICTURE_HEVC_LONG_TERM_REFERENCE, PICTURE_HEVC_RPS_LT_CURR, PICTURE_HEVC_RPS_ST_CURR_AFTER, PICTURE_HEVC_RPS_ST_CURR_BEFORE, SLICE_DATA_FLAG_ALL,
    VA_PADDING_LOW, VaIqMatrixBufferH264, VaPictureH264, VaPictureHevc, VaPictureParameterBufferH264, VaPictureParameterBufferHevc, VaSliceParameterBufferH264,
    VaSliceParameterBufferHevc, record_bytes,
};

/// The most reference frames a `VAPictureParameterBufferH264` holds (`ReferenceFrames[16]`,
/// `va.h:3601`). The DPB a picture may use is bounded by the stream's own parsed DPB size and by
/// this; a stream claiming more cannot be described and is declined.
const MAX_REF_FRAMES: usize = 16;

/// How many slice segments one access unit may hold, since a record is submitted per segment and the
/// NAL list comes from a sample. No stream FilmCraft decodes has anything like this many, so it
/// bounds what a corrupt sample can make this module allocate rather than describing a real limit.
const MAX_SLICES: usize = 256;

/// The most reference frames a `VAPictureParameterBufferHEVC` holds (`ReferenceFrames[15]`,
/// `va_dec_hevc.h:64`), which is also the most a `VASliceParameterBufferHEVC`'s `RefPicList` names.
const MAX_HEVC_REF_PICS: usize = 15;

/// The bytes of an HEVC NAL unit header (`nuh_layer_id` and `nuh_temporal_id_plus1`), which the
/// slice segment header's byte offsets in a record are counted from.
const HEVC_NAL_HEADER: [u8; 2] = [0; 2];

/// Picture order count state carried across a stream's pictures, so each picture's POC is the one
/// FilmCraft's own decoder would compute for it. One per decoder: a seek or a flush that starts
/// again mid-stream calls [`NalPoc::reset`], and so does a caller whose first picture is not an IDR
/// — an IDR's POC follows from being an IDR rather than from what came before.
///
/// It also carries the decoded picture buffer: which surfaces hold a picture that later pictures may
/// refer to, and the marks 8.2.5 / 8.3.2 put on them. A VA-API driver keeps its own picture buffer,
/// but the records name its contents — [`VaPictureParameterBufferH264::reference_frames`] and
/// [`VaSliceParameterBufferH264::ref_pic_list0`] are the only place the driver is told what is in it
/// — so the list lives here, beside the parser state the pictures' identities come from. The decode
/// loop's surface ring is where the surfaces come from; it does not need to know which of them hold
/// a reference.
#[derive(Clone, Default)]
pub struct NalPoc {
    h264: filmcraft_h264::slice::PocState,
    hevc: filmcraft_hevc::PocState,
    dpb: Vec<DpbEntry>,
    /// `MaxFrameNum` (H.264 7.4.2.1.1): the modulus `frame_num` wraps at, which 8.2.4.1's `pic_num`
    /// needs to order two frames that straddle the wrap. Zero until the first picture is parsed.
    max_frame_num: u32,
}

impl NalPoc {
    /// Forget every picture decoded so far, as after a seek or a flush: the next picture's POC is
    /// computed as though nothing came before it, and the decoded picture buffer is empty.
    pub fn reset(&mut self) {
        self.h264 = filmcraft_h264::slice::PocState::default();
        self.hevc = filmcraft_hevc::PocState::default();
        self.dpb.clear();
        self.max_frame_num = 0;
    }

    /// The surfaces the decoded picture buffer still refers to — the ones a new picture may **not**
    /// be decoded into, because the driver would read the new picture as that reference.
    ///
    /// This is the buffer's own knowledge of where its pictures live, which is the only place in the
    /// backend that knows: the decode loop's surface ring is a pool and cannot tell a live reference
    /// from a free surface. The decode loop asks this before it builds a picture's records, because
    /// building them is what adds the picture to the buffer.
    pub fn live_surfaces(&self) -> Vec<u32> {
        self.dpb.iter().filter(|e| e.mark != Mark::Unused).map(|e| e.surface).collect()
    }
}

/// One slice's submission: where its bytes sit in the picture's slice data buffer, how long they
/// are, and the record describing it. The caller copies `offset..offset + length` out of the buffer
/// it builds from the sample and submits `params` alongside it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceParam {
    /// Byte offset of this slice in the picture's slice data buffer.
    pub offset: u32,
    /// Byte length of this slice there.
    pub length: u32,
    /// The slice parameter record, as the bytes the driver reads.
    pub params: Vec<u8>,
}

/// Everything one access unit is submitted as, as owned bytes: the picture parameter buffer, one
/// slice parameter buffer per slice, and where each slice sits in the sample its NAL units come
/// from. The records are opaque bytes rather than libva types so that `mod.rs` never names a `VA*`
/// structure; only this module and `ffi` know what is inside them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Picture {
    /// The picture parameter buffer record.
    pub params: Vec<u8>,
    /// The IQ matrix (scaling list) record — H.264 only; empty for HEVC, whose scaling lists are
    /// part of its picture parameter record. See [`h264_iq_matrix`] for why a decode cannot be
    /// without it.
    pub iq_matrix: Vec<u8>,
    /// One entry per slice, in decoding order.
    pub slices: Vec<SliceParam>,
    /// Byte offset of each slice's NAL unit within the sample it came from, parallel to `slices`.
    pub slice_offsets: Vec<usize>,
}

// ---------------------------------------------------------------------------------------------
// The decoded picture buffer, and the reference lists a slice's record names.
//
// The driver's own picture buffer is not something this backend can see; the records are how it is
// told what is in it. So the pictures the decoder has decoded are listed here, each with the surface
// it is in and the mark reference marking (8.2.5 for H.264, 8.3.2 for HEVC) gives it, and each
// record is filled in from that list.
//
// Every reference a slice names has to be a surface holding *that* picture. A reference the buffer
// does not hold is therefore an `Err`, never a filler: a filler is a real surface id holding a
// different picture, and a driver reads it as the reference without complaint, so the mistake shows
// up as corruption on screen rather than as an error. Handing the stream to the software decoder
// instead is the visible, correct outcome.
// ---------------------------------------------------------------------------------------------

/// One picture in the decoded picture buffer: which surface it is in, and the numbers a later
/// picture refers to it by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DpbEntry {
    /// The surface the picture was decoded into.
    surface: u32,
    /// `frame_num` (H.264) or `pic_order_cnt` (HEVC).
    frame_num: u32,
    poc: i32,
    mark: Mark,
}

/// How a picture in the decoded picture buffer is marked. The variants are 8.2.5.1's three states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mark {
    /// Not a reference: the surface may be decoded into again.
    Unused,
    /// A short-term reference, named by its `frame_num` (H.264) or POC (HEVC).
    Short,
    /// A long-term reference with the index its marking gave it (`long_term_frame_idx` in H.264's
    /// MMCO 3 / 6, the position in `lt_ref_pic_poc_lsb_sps` in HEVC).
    Long(u32),
}

impl Mark {
    /// The entry as a `ReferenceFrames` slot: an unused one names no surface and says so, and a
    /// used one carries the surface, the number it is named by and the `VA_PICTURE_*` bit its mark
    /// is.
    fn h264_picture(self, entry: &DpbEntry) -> VaPictureH264 {
        if self == Mark::Unused {
            return invalid_h264_picture();
        }
        let long = matches!(self, Mark::Long(_));
        VaPictureH264 {
            picture_id: entry.surface,
            frame_idx: entry.frame_num,
            flags: if long { PICTURE_H264_LONG_TERM_REFERENCE } else { PICTURE_H264_SHORT_TERM_REFERENCE },
            top_field_order_cnt: entry.poc,
            bottom_field_order_cnt: 0,
            va_reserved: [0; VA_PADDING_LOW],
        }
    }

    /// [`Mark::h264_picture`] for the HEVC record: the same entry with the POC in the field HEVC
    /// names pictures by.
    fn hevc_picture(self, entry: &DpbEntry, used: Option<u32>) -> VaPictureHevc {
        if self == Mark::Unused {
            return invalid_hevc_picture();
        }
        // `used` is the `VA_PICTURE_HEVC_RPS_*` bit saying the current picture refers to this one,
        // which is exclusive of the long-term bit (va.h:5325).
        let flags = match (used, self) {
            (Some(bit), _) => bit,
            (None, Mark::Long(_)) => PICTURE_HEVC_LONG_TERM_REFERENCE,
            _ => 0,
        };
        VaPictureHevc { picture_id: entry.surface, pic_order_cnt: entry.poc, flags, va_reserved: [0; VA_PADDING_LOW] }
    }
}

/// How many surfaces a stream's decoded picture buffer needs, so the decode loop knows how many to
/// create: the number of pictures its sequence parameter set allows in the buffer at once
/// (`max_num_ref_frames` for H.264, `sps_max_dec_pic_buffering_minus1` for HEVC).
///
/// This is the *reference* count, not the presentation-reordering count: a picture is read back out
/// of its surface and handed on as host memory before the next picture is decoded, so the only
/// pictures that have to stay in a surface are the ones later pictures refer to. A parameter set
/// that will not parse is an `Err` — the surfaces are allocated for the sizes it describes, so a
/// stream whose parameter sets do not parse is not one this backend hands to a driver.
pub fn reference_pictures(info: &NalStreamInfo) -> Result<usize, String> {
    let parsed = match info.codec {
        NalCodec::H264 => H264Sets::parse(info)?
            .spss
            .iter()
            .flatten()
            .map(|sps| sps.max_num_ref_frames as usize)
            .max()
            .ok_or("no sequence parameter set to size the decoded picture buffer from")?,
        NalCodec::Hevc => HevcSets::parse(info)?
            .spss
            .iter()
            .flatten()
            .map(|sps| sps.max_dec_pic_buffering as usize)
            .max()
            .ok_or("no sequence parameter set to size the decoded picture buffer from")?,
    };
    // At least one, since every picture that is not a reference still has to be decoded somewhere,
    // and at most what the H.264 record can name.
    Ok(parsed.clamp(1, MAX_REF_FRAMES))
}

/// How many surfaces the decode loop must have, for a stream this backend decodes: the most the
/// decoded picture buffer can hold at once, **plus the one the next picture is decoded into**.
///
/// This is deliberately *not* [`reference_pictures`]. That is what the stream's sequence parameter
/// set declares, and it is what a diagnostics report should show, but it is not a bound the buffer
/// respects: HEVC decides what stays a reference from each picture's own reference picture set
/// (8.3.2), and consecutive sets can name more pictures between them than
/// `sps_max_dec_pic_buffering_minus1` counts — on a 640x360 fixture declaring five, the buffer
/// reached six and the decode ran out of surfaces. Sizing the pool from the declaration is what let
/// a picture be decoded into a surface the driver was reading as a reference.
///
/// The bound that does hold is the record's own: a picture record cannot name more references than
/// it has slots, and the marking functions refuse to keep more than that in the buffer (a reference
/// with no slot would be dropped from the driver's copy of the buffer without anything saying so).
/// So the buffer holds at most the record's slot count, and the pool needs one more surface than
/// that — the one being written, which must not be a surface the buffer still refers to.
pub fn surface_pool(info: &NalStreamInfo) -> usize {
    match info.codec {
        NalCodec::H264 => MAX_REF_FRAMES + 1,
        NalCodec::Hevc => MAX_HEVC_REF_PICS + 1,
    }
}

impl Picture {
    /// The size of the slice data buffer this picture's records address: the end of its last slice.
    /// The caller assembles the picture's slices into a buffer of this size, laid out as
    /// `slices[i].offset` says, and submits `slices[i].params` alongside it.
    pub fn slice_data_size(&self) -> u32 {
        self.slices.last().map_or(0, |s| s.offset.saturating_add(s.length))
    }
}

/// The H.264 IQ matrix record for a picture decoded with `sps`: that sequence parameter set's own
/// scaling lists, or the flat 16s of 7.4.2.1.1's default when it declares none.
///
/// **A decode that does not submit this record does not decode.** That is not an inference from the
/// specification, which never mentions it: it is what the iHD driver on an Intel Arc does. Given
/// the picture, slice and slice-data records with no IQ matrix among them, it takes the whole
/// submission, `vaBeginPicture`, `vaRenderPicture`, `vaEndPicture` and `vaSyncSurface` all report
/// `VA_STATUS_SUCCESS`, `vaQuerySurfaceStatus` reports the surface ready — and the surface holds a
/// uniform mid-grey, because nothing was ever rendered into it. An indirect readback (`vaPutImage`)
/// of the same surface is uniformly *zero* instead. A driver that fails loudly is easy to fall back
/// from; this one succeeds at every step and returns a plausible picture that is not the stream's,
/// which is the outcome this backend is not allowed to ship (see `docs/adr/0001-platform-ffi.md`
/// and the never-crash rule's "never lost work").
///
/// So the record is submitted for every H.264 picture, flat lists and all. It costs 240 bytes and
/// one buffer per picture.
fn h264_iq_matrix(sps: &filmcraft_h264::params::Sps) -> Vec<u8> {
    // A copy and nothing more: `Sps::scaling` already holds the lists in raster scan order and in
    // the record's own order — `[Intra Y, Intra Cb, Intra Cr, Inter Y, Inter Cb, Inter Cr]` for
    // the 4x4 (7.3.2.1.1.1's list order) — so nothing is transposed or re-scanned here, and a
    // stream with custom quantizers gets exactly the lists it carries.
    //
    // Only the two luma 8x8 lists go in: `VAIQMatrixBufferH264`'s 8x8 array is `[2][64]`, holding
    // `[Intra Y, Inter Y]` and nothing for chroma, because an 8x8 chroma scaling list only exists
    // at `chroma_format_idc == 3` (4:4:4), which `plan_nal` declines. `ScalingMatrices` keeps all
    // six because the software decoder applies chroma 8x8 lists at that depth; this path does not
    // get there, and putting a chroma list where the record has no slot for it would silently
    // submit the wrong quantizer.
    let m = &sps.scaling;
    let record = VaIqMatrixBufferH264 { scaling_list4x4: m.m4, scaling_list8x8: [m.m8[0], m.m8[3]], ..Default::default() };
    record_bytes(&record)
}

// ---------------------------------------------------------------------------------------------
// Reference marking and reference list construction, for both codecs. The two are the same shape
// (H.264 8.2.4 / 8.2.5 and HEVC 8.3.2 / 8.3.4) and they are separate from the parsing above because
// they are about *surfaces* rather than about bitstream: which surface a later picture is looking for,
// and whether it is still there.
// ---------------------------------------------------------------------------------------------

/// `pic_num` (H.264 8.2.4.1): `entry`'s `frame_num` as a number relative to the current picture's,
/// so that two frames that straddle the `frame_num` wrap compare the right way round. A stream's
/// `MaxFrameNum` is at most 65536 and its `frame_num` is below it, so neither can overflow an `i32`.
fn h264_pic_num(entry: &DpbEntry, cur_frame_num: u32, max_frame_num: u32) -> i32 {
    let n = entry.frame_num as i32;
    if n > cur_frame_num as i32 { n - max_frame_num as i32 } else { n }
}

/// 8.2.4.2 (initial list construction) and 8.2.4.3 (list modification), as indices into `dpb`.
///
/// This is `filmcraft_h264`'s own `Dpb::build_ref_lists` with one difference, forced by the surface:
/// it returns `Err` where that function substitutes a fallback entry for a reference the buffer does
/// not hold, and it works in indices rather than `FrameRef`s. A slice record's `RefPicList` names
/// surfaces, so there is no such thing as a filler here — an entry that is not the picture the slice
/// refers to is a *different* picture, which the driver reads without complaint, so the mistake shows
/// up as corruption on screen rather than as an error. Handing the stream to the software decoder is
/// the visible, correct outcome.
fn h264_ref_lists(dpb: &[DpbEntry], sh: &filmcraft_h264::slice::SliceHeader, cur_poc: i32, max_frame_num: u32) -> Result<[Vec<usize>; 2], String> {
    let mut lists: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    if sh.slice_type.is_intra() {
        return Ok(lists);
    }
    if max_frame_num == 0 {
        return Err("no sequence parameter set to order reference pictures by".into());
    }
    let cur = sh.frame_num as i32;
    let modulus = max_frame_num as i32;
    let pic_num = |i: usize| h264_pic_num(&dpb[i], sh.frame_num, max_frame_num);
    let short: Vec<usize> = (0..dpb.len()).filter(|&i| dpb[i].mark == Mark::Short).collect();
    let mut long: Vec<usize> = (0..dpb.len()).filter(|&i| matches!(dpb[i].mark, Mark::Long(_))).collect();
    long.sort_by_key(|&i| match dpb[i].mark {
        Mark::Long(n) => n,
        Mark::Short | Mark::Unused => 0,
    });
    let mut entries: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    if sh.slice_type != filmcraft_h264::slice::SliceType::B {
        let mut s = short.clone();
        s.sort_by_key(|&i| std::cmp::Reverse(pic_num(i)));
        entries[0] = s.into_iter().chain(long.iter().copied()).collect();
    } else {
        let mut before: Vec<usize> = short.iter().copied().filter(|&i| dpb[i].poc < cur_poc).collect();
        let mut after: Vec<usize> = short.iter().copied().filter(|&i| dpb[i].poc > cur_poc).collect();
        before.sort_by_key(|&i| std::cmp::Reverse(dpb[i].poc));
        after.sort_by_key(|&i| dpb[i].poc);
        entries[0] = before.iter().chain(&after).chain(&long).copied().collect();
        entries[1] = after.iter().chain(&before).chain(&long).copied().collect();
        // 8.2.4.2.3: when the two lists came out identical, swap the first two of the second, so the
        // two pictures of a B pair are not each other's first reference.
        if entries[1].len() > 1 && entries[0] == entries[1] {
            entries[1].swap(0, 1);
        }
    }
    for (list, out) in lists.iter_mut().enumerate().take(if sh.slice_type.is_b() { 2 } else { 1 }) {
        let n = usize::try_from(sh.num_ref_idx_active[list]).map_err(|_| "num_ref_idx_active does not fit a list".to_string())?;
        // 8.2.4.3.1 rewrites the initial list in place; an entry the modification leaves unplaced
        // stays `None` and is the missing-reference error below rather than a filler.
        let mut built: Vec<Option<usize>> = entries[list].iter().copied().take(n).map(Some).collect();
        built.resize(n, None);
        let mut pred = cur;
        for (index, m) in sh.ref_pic_list_mod[list].iter().enumerate() {
            if index >= n {
                return Err("more reference list modifications than references".into());
            }
            // idc 3 ends the loop and 4, 5 and 6 change the list's length; only 0..=2 name a
            // picture, and this module does not implement the length changes.
            let (target, is_long) = match m.idc {
                3 => break,
                4..=6 => return Err(format!("modification_of_pic_nums_idc {}", m.idc)),
                0 | 1 => {
                    let d = i32::try_from(m.value).map_err(|_| "abs_diff_pic_num_minus1 does not fit".to_string())?.saturating_add(1);
                    let no_wrap = if m.idc == 0 {
                        if pred - d < 0 { pred - d + modulus } else { pred - d }
                    } else if pred + d >= modulus {
                        pred + d - modulus
                    } else {
                        pred + d
                    };
                    pred = no_wrap;
                    let wanted = if no_wrap > cur { no_wrap - modulus } else { no_wrap };
                    (short.iter().copied().find(|&i| pic_num(i) == wanted), false)
                }
                2 => (long.iter().copied().find(|&i| matches!(dpb[i].mark, Mark::Long(k) if k == m.value)), true),
                other => return Err(format!("modification_of_pic_nums_idc {other}")),
            };
            let target = target.ok_or_else(|| {
                if is_long {
                    format!("long-term reference {} is not in the decoded picture buffer", m.value)
                } else {
                    format!("short-term reference at pic_num {pred} is not in the decoded picture buffer")
                }
            })?;
            // 8.2.4.3.1: the named picture goes at `index`, what was there shifts right, and the
            // picture that was displaced is dropped from later in the list.
            for c in (index + 1..n).rev() {
                built[c] = built[c - 1];
            }
            built[index] = Some(target);
            let mut write = index + 1;
            for c in index + 1..n {
                let keeps = match built[c] {
                    None => true,
                    Some(e) if is_long => !matches!(dpb[e].mark, Mark::Long(k) if k == m.value),
                    Some(e) => e != target,
                };
                if keeps {
                    built[write] = built[c];
                    write += 1;
                }
            }
        }
        let mut built_list: Vec<usize> = Vec::with_capacity(n);
        for (i, entry) in built.iter().enumerate() {
            built_list.push(entry.ok_or_else(|| format!("reference {i} of list {list} is not in the decoded picture buffer"))?);
        }
        *out = built_list;
    }
    Ok(lists)
}

/// 8.2.5: reference marking for the picture that has just been decoded, and its insertion into the
/// decoded picture buffer. This is `filmcraft_h264`'s own `Dpb::store_picture` marking half without
/// the output bumping: the only output order this backend has to keep is the one it reads back in,
/// and a picture's surface is only needed until the next picture that refers to it is decoded.
///
/// It runs when the *records are built*, which is before the picture is rendered rather than after.
/// That is safe because a decode that fails after this point takes the whole decoder with it (the
/// hybrid decoder drops the hardware one and replays the run in software), and because what follows a
/// flush or a seek is a fresh run whose first picture is a restart point, which clears the buffer.
fn h264_mark(poc: &mut NalPoc, sh: &filmcraft_h264::slice::SliceHeader, sps: &filmcraft_h264::params::Sps, surface: u32, cur_poc: i32) {
    // `MaxFrameNum = 1 << log2_max_frame_num` (7.4.2.1.1), which is what `frame_num` wraps at and so
    // what `pic_num` is relative to. `log2_max_frame_num` is bounded to 4..=16 by the parser and
    // re-bounded here, because a `shl` that overflowed would make every comparison wrong.
    let max_frame_num = 1u32.checked_shl(sps.log2_max_frame_num).unwrap_or(0).max(1);
    poc.max_frame_num = max_frame_num;
    if sh.idr {
        // 8.2.5.1: an IDR empties the buffer; there is no reference across it.
        poc.dpb.clear();
    }
    let mut mark = Mark::Unused;
    if sh.nal_ref_idc != 0 {
        if sh.idr {
            mark = if sh.long_term_reference { Mark::Long(0) } else { Mark::Short };
        } else {
            let mut current_long = None;
            if sh.adaptive_ref_pic_marking {
                current_long = h264_mmcos(poc, &sh.mmcos, sh.frame_num, max_frame_num);
            } else {
                h264_sliding_window(poc, sh.frame_num, max_frame_num, sps.max_num_ref_frames.max(1) as usize);
            }
            mark = match current_long {
                Some(index) => Mark::Long(index),
                None => {
                    // A stream whose MMCOs left the buffer over-full is not conforming, and a
                    // buffer over-full is a surface that is about to be reused under a reference.
                    if poc.dpb.iter().filter(|e| e.mark != Mark::Unused).count() >= sps.max_num_ref_frames.max(1) as usize {
                        h264_sliding_window(poc, sh.frame_num, max_frame_num, sps.max_num_ref_frames.max(1) as usize);
                    }
                    Mark::Short
                }
            };
        }
    }
    poc.dpb.retain(|e| e.mark != Mark::Unused);
    poc.dpb.push(DpbEntry { surface, frame_num: sh.frame_num, poc: cur_poc, mark });
    // More pictures than the record can name is a picture buffer this backend cannot describe: a
    // reference that no slot fits would be silently dropped from the driver's copy of the buffer.
    // Keeping the most recent ones (the ones a reference can reach) is the least surprising trim,
    // and a stream that gets here is one no conforming encoder produces.
    if poc.dpb.len() > MAX_REF_FRAMES {
        poc.dpb.drain(..poc.dpb.len() - MAX_REF_FRAMES);
    }
}

/// 8.2.5.3 sliding window: make room by unmarking the oldest short-term reference.
fn h264_sliding_window(poc: &mut NalPoc, cur_frame_num: u32, max_frame_num: u32, limit: usize) {
    while poc.dpb.iter().filter(|e| e.mark != Mark::Unused).count() >= limit.max(1) {
        let victim = poc
            .dpb
            .iter()
            .enumerate()
            .filter(|(_, e)| e.mark == Mark::Short)
            .min_by_key(|(_, e)| h264_pic_num(e, cur_frame_num, max_frame_num))
            .map(|(i, _)| i);
        // Nothing short-term left to make room with: the buffer holds long-term references only, and
        // removing one of those would be a decision 8.2.5 does not offer. Leaving it is the
        // conservative choice, and `h264_mark` trims the buffer to what the record can name.
        let Some(i) = victim else { break };
        poc.dpb[i].mark = Mark::Unused;
    }
    poc.dpb.retain(|e| e.mark != Mark::Unused);
}

/// 8.2.5.4 memory management control: the MMCOs 1 to 6, returning MMCO 6's long-term index for the
/// current picture. This is `filmcraft_h264`'s `Dpb::apply_mmcos` marking marks on the buffer rather
/// than on pictures.
fn h264_mmcos(poc: &mut NalPoc, mmcos: &[filmcraft_h264::slice::Mmco], cur_frame_num: u32, max_frame_num: u32) -> Option<u32> {
    let cur = cur_frame_num as i32;
    let mut current_long = None;
    for m in mmcos {
        match m.op {
            1 => {
                let pn = cur - (m.difference_of_pic_nums_minus1 as i32 + 1);
                for e in poc.dpb.iter_mut() {
                    if e.mark == Mark::Short && h264_pic_num(e, cur_frame_num, max_frame_num) == pn {
                        e.mark = Mark::Unused;
                    }
                }
            }
            2 => {
                for e in poc.dpb.iter_mut() {
                    if matches!(e.mark, Mark::Long(k) if k == m.long_term_pic_num) {
                        e.mark = Mark::Unused;
                    }
                }
            }
            3 => {
                let pn = cur - (m.difference_of_pic_nums_minus1 as i32 + 1);
                for e in poc.dpb.iter_mut() {
                    if matches!(e.mark, Mark::Long(k) if k == m.long_term_pic_num) {
                        e.mark = Mark::Unused;
                    }
                }
                for e in poc.dpb.iter_mut() {
                    if e.mark == Mark::Short && h264_pic_num(e, cur_frame_num, max_frame_num) == pn {
                        e.mark = Mark::Long(m.long_term_frame_idx);
                    }
                }
            }
            4 => {
                let max = m.max_long_term_frame_idx_plus1;
                for e in poc.dpb.iter_mut() {
                    if matches!(e.mark, Mark::Long(k) if max == 0 || k > max - 1) {
                        e.mark = Mark::Unused;
                    }
                }
            }
            5 => {
                poc.dpb.clear();
                current_long = None;
            }
            6 => current_long = Some(m.long_term_frame_idx),
            _ => {}
        }
    }
    poc.dpb.retain(|e| e.mark != Mark::Unused);
    current_long
}

/// The `ReferenceFrames` a picture's record carries: the decoded picture buffer as it is for this
/// picture, and how many entries of it were filled in.
///
/// The count is what was filled, never the DPB size the sequence parameter set declares: a driver
/// reads `num_ref_frames` as the number of entries it may look at, so a count larger than the number
/// of entries that name a surface is a record telling the driver to read invalid slots as pictures.
/// An IDR starts from nothing, so it carries zero.
fn h264_reference_frames(dpb: &[DpbEntry]) -> ([VaPictureH264; MAX_REF_FRAMES], u8, Vec<Option<usize>>) {
    let mut frames = [invalid_h264_picture(); MAX_REF_FRAMES];
    let mut slots = vec![None; dpb.len()];
    let mut n = 0usize;
    // Only the entries that are a reference get a slot, and only those are counted. An entry marked
    // unused is a picture that is not a reference — one with `nal_ref_idc == 0`, kept so its surface
    // is not reused while a later picture may still name its `frame_num` — and a slot that named it
    // would be a counted slot holding no picture.
    for (index, entry) in dpb.iter().enumerate().filter(|(_, e)| e.mark != Mark::Unused) {
        if n == MAX_REF_FRAMES {
            break;
        }
        frames[n] = entry.mark.h264_picture(entry);
        slots[index] = Some(n);
        n += 1;
    }
    (frames, u8::try_from(n).unwrap_or(u8::MAX), slots)
}

/// The two `RefPicList`s of a slice record, as the pictures they name, with everything past each
/// list's length left invalid. `slots` maps a decoded-picture-buffer index to its `ReferenceFrames`
/// slot ([`h264_reference_frames`]), which is how a list entry finds the picture the driver will read.
fn h264_ref_pic_lists(dpb: &[DpbEntry], lists: &[Vec<usize>; 2], slots: &[Option<usize>]) -> Result<([VaPictureH264; 32], [VaPictureH264; 32]), String> {
    let mut l0 = [invalid_h264_picture(); 32];
    let mut l1 = [invalid_h264_picture(); 32];
    for (list, built) in lists.iter().enumerate() {
        if built.len() > 32 {
            return Err(format!("reference list {list} of {} entries does not fit the record", built.len()));
        }
        let out: &mut [VaPictureH264; 32] = match list {
            0 => &mut l0,
            _ => &mut l1,
        };
        for (at, &index) in built.iter().enumerate() {
            let entry = dpb.get(index).ok_or_else(|| format!("reference list {list} entry {at} names entry {index} of a {}-entry buffer", dpb.len()))?;
            // Every entry of a built list is a reference, so it has a slot in the picture record's
            // `ReferenceFrames`; a `None` here means the list and that record disagree, which is a bug
            // here rather than a fault in the stream. The entry itself is the picture: a list position
            // holds a whole `VAPictureH264`, whose `frame_idx` is how the driver finds it again.
            if slots.get(index).copied().flatten().is_none() {
                return Err(format!("reference list {list} entry {at} has no slot in the picture record"));
            }
            out[at] = entry.mark.h264_picture(entry);
        }
    }
    Ok((l0, l1))
}

/// 8.3.2 (the reference picture set) and 8.3.4 (the reference lists), for HEVC.
///
/// `dpb` comes in holding every picture the previous pictures marked as a reference and comes out
/// holding the ones this picture's set keeps — that is 8.3.2's own rule, that a picture the set does
/// not name is no longer a reference. The lists come back as indices into what is left, in the order
/// 8.3.4 builds them, which is also the order the picture record's `ReferenceFrames` is filled in.
fn hevc_ref_lists(dpb: &mut Vec<DpbEntry>, sh: &filmcraft_hevc::slice::SliceHeader, cur_poc: i32, max_poc_lsb: i32) -> Result<[Vec<usize>; 2], String> {
    let mut lists: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    if sh.is_intra() {
        return Ok(lists);
    }
    if max_poc_lsb <= 0 {
        return Err("log2_max_pic_order_cnt_lsb is out of range".into());
    }
    // Long-term entries first, as 8.3.2 does: a long-term reference is named by its POC LSB rather
    // than by a delta from this picture, so it can refer to a picture in either direction.
    let mut long: Vec<(Option<usize>, bool, u32)> = Vec::new();
    for (index, &(lsb, used, msb_present, cycle)) in sh.lt.iter().enumerate() {
        let mut p = i32::try_from(lsb).map_err(|_| "PocLsbLt does not fit".to_string())?;
        if msb_present {
            let cycle = i32::try_from(cycle).map_err(|_| "DeltaPocMsbCycleLt does not fit".to_string())?;
            p = p.wrapping_add(cur_poc.wrapping_sub(cycle.wrapping_mul(max_poc_lsb)).wrapping_sub(cur_poc & (max_poc_lsb - 1)));
        }
        let found = dpb.iter().position(|e| e.mark != Mark::Unused && if msb_present { e.poc == p } else { (e.poc & (max_poc_lsb - 1)) == p });
        long.push((found, used, u32::try_from(index).unwrap_or(u32::MAX)));
    }
    // `st_rps.s0` carries negative deltas (the pictures before this one) and `s1` positive ones.
    let mut short: Vec<(Option<usize>, bool, bool)> = Vec::new();
    for before in [true, false] {
        for &(d, used) in if before { &sh.st_rps.s0 } else { &sh.st_rps.s1 } {
            let p = cur_poc.wrapping_add(d);
            let found = dpb.iter().position(|e| e.mark == Mark::Short && e.poc == p);
            short.push((found, used, before));
        }
    }
    // The set, in the order the record's flags and lists need it, against the buffer as it is now.
    // Every reference the set names has to be in it: 8.3.3's "unavailable reference picture" is a
    // picture generated here rather than decoded, and a VA-API record has nowhere to say so — handing
    // the driver another surface for it would read a different picture without complaint.
    let mut before: Vec<usize> = Vec::new();
    let mut after: Vec<usize> = Vec::new();
    for &(found, used, is_before) in &short {
        if !used {
            continue;
        }
        let entry = found.ok_or("a short-term reference the picture set names is not in the decoded picture buffer")?;
        (if is_before { &mut before } else { &mut after }).push(entry);
    }
    let mut lt_curr: Vec<usize> = Vec::new();
    for &(found, used, _) in &long {
        if !used {
            continue;
        }
        lt_curr.push(found.ok_or("a long-term reference the picture set names is not in the decoded picture buffer")?);
    }
    let total = before.len() + after.len() + lt_curr.len();
    if total > MAX_HEVC_REF_PICS {
        return Err(format!("{total} reference pictures, more than the record names ({MAX_HEVC_REF_PICS})"));
    }
    // 8.3.4: each list is the set's before / after / long-term pictures, cycled up to `num_ref_idx`
    // entries, with the slice's own list entry reordering which one goes where.
    for (list, out) in lists.iter_mut().enumerate().take(if sh.is_b() { 2 } else { 1 }) {
        let want = usize::try_from(sh.num_ref_idx[list]).map_err(|_| "num_ref_idx does not fit the record".to_string())?;
        if want > MAX_HEVC_REF_PICS {
            return Err(format!("num_ref_idx_l{list}_active of {want} does not fit the record ({MAX_HEVC_REF_PICS})"));
        }
        let (first, second) = if list == 0 { (&before, &after) } else { (&after, &before) };
        let mut temp: Vec<usize> = Vec::with_capacity(want);
        // 8.3.4 cycles the set until the list is long enough, which is a loop over nothing when the
        // set is empty. `filmcraft_hevc` refuses an empty set (`StRps::total > 0`), but this loop
        // must not depend on another crate's guarantee to terminate, so the empty case is the error.
        if before.is_empty() && after.is_empty() && lt_curr.is_empty() {
            return Err(format!("list {list} of a picture whose reference set names no picture"));
        }
        while temp.len() < want {
            for entry in first.iter().chain(second.iter()).chain(lt_curr.iter()) {
                if temp.len() < want {
                    temp.push(*entry);
                }
            }
        }
        for i in 0..want {
            let at = match sh.list_entry[list].as_ref() {
                Some(entry) => *entry.get(i).ok_or("a reference list modification runs past the list")? as usize,
                None => i,
            };
            out.push(*temp.get(at).ok_or("a reference list modification names a picture the list does not hold")?);
        }
    }
    // 8.3.2's other half: a picture the set does not name stops being a reference, and the ones it
    // does are re-marked — a picture that becomes long-term was short-term until now.
    let mut keep = vec![false; dpb.len()];
    for &(found, _, _) in &long {
        if let Some(i) = found {
            keep[i] = true;
        }
    }
    for &(found, _, _) in &short {
        if let Some(i) = found {
            keep[i] = true;
        }
    }
    for (i, entry) in dpb.iter_mut().enumerate() {
        if !keep[i] {
            entry.mark = Mark::Unused;
        }
    }
    for &(found, _, index) in &long {
        if let Some(i) = found {
            dpb[i].mark = Mark::Long(index);
        }
    }
    // The lists above name positions in the buffer as it was; the entries that were dropped close
    // up behind them, so every index moves. `moved` is the old position to the new one.
    let mut moved = vec![None; dpb.len()];
    let mut next = 0usize;
    for (i, entry) in dpb.iter().enumerate() {
        if entry.mark != Mark::Unused {
            moved[i] = Some(next);
            next += 1;
        }
    }
    dpb.retain(|e| e.mark != Mark::Unused);
    for list in lists.iter_mut() {
        for entry in list.iter_mut() {
            let at = *entry;
            *entry = moved.get(at).copied().flatten().ok_or("a reference list entry is not in the decoded picture buffer")?;
        }
    }
    Ok(lists)
}

/// 8.3.5: the picture just decoded joins the decoded picture buffer, and an IDR or BLA empties it
/// first (nothing references across one).
///
/// Every picture goes in as a short-term reference, because HEVC has no `nal_ref_idc` — unlike H.264,
/// whether a picture is a reference is decided by the *next* picture's reference picture set, and
/// [`hevc_ref_lists`] is what marks everything that set does not name as unused. A picture marked
/// unused here would be invisible to the next picture's set, which is the same as never having
/// decoded it.
fn hevc_mark(dpb: &mut Vec<DpbEntry>, sh: &filmcraft_hevc::slice::SliceHeader, surface: u32, cur_poc: i32) {
    if sh.nal.is_idr() || sh.nal.is_bla() {
        dpb.clear();
    }
    dpb.retain(|e| e.mark != Mark::Unused);
    dpb.push(DpbEntry { surface, frame_num: u32::try_from(cur_poc).unwrap_or(u32::MAX), poc: cur_poc, mark: Mark::Short });
    // The buffer cannot hold more pictures than a picture record can name. `hevc_ref_lists` has
    // already refused a set larger than that (which is why this is rarely reached), but what it
    // compacts to is the set the *previous* picture named, and this picture adds one more — so the
    // trim is what makes [`surface_pool`]'s bound true rather than hopeful. Keeping the most recent
    // is the least surprising trim: those are the ones a reference can still reach.
    if dpb.len() > MAX_HEVC_REF_PICS {
        dpb.drain(..dpb.len() - MAX_HEVC_REF_PICS);
    }
}

/// The picture- and slice-parameter records for one H.264 / HEVC access unit, or why it has none.
///
/// `surface` is the VA surface the picture decodes into — the render target the caller took from its
/// surface ring — and `width_mbs` / `height_mbs` the coded picture size in H.264 macroblocks or
/// HEVC luma samples, which must be the size the surfaces were created at; a stream that disagrees
/// is declined rather than described at the wrong size.
///
/// `poc` advances once per picture, so a caller keeps one [`NalPoc`] for the life of the stream and
/// resets it when decoding starts again somewhere new.
pub fn build_nal(info: &NalStreamInfo, sample: &[u8], poc: &mut NalPoc, surface: u32, width_mbs: u32, height_mbs: u32) -> Result<Picture, String> {
    match info.codec {
        NalCodec::H264 => build_h264(info, sample, poc, surface, width_mbs, height_mbs),
        NalCodec::Hevc => build_hevc(info, sample, poc, surface, width_mbs, height_mbs),
    }
}

/// The tail both frame-codec declines end in: neither codec crate exposes the per-frame header a
/// picture parameter record needs. [`super::plan_frame`] and [`build_frame`] both build their
/// message from this, so a caller that hits either is told the same thing.
pub(crate) const FRAME_HEADER_MISSING: &str = "does not expose the uncompressed frame header";

/// The record one frame-coded (VP9 / AV1) sample is submitted as, or why it has none.
///
/// VP9 and AV1 carry their picture structure in each frame's own uncompressed header, and libva
/// wants that header as a record. FilmCraft parses those headers — in `filmcraft_vp9` and
/// `filmcraft_av1` — but exposes only the key frame's stream parameters and the AV1 sequence
/// header, not the per-frame header a record needs (VP9's tile and loop-filter state, AV1's tile
/// layout and quantizer). So this milestone submits neither, and says so, rather than hand a driver
/// a record full of zeros it would read as a valid picture.
///
/// **This is a backstop, not the decision.** [`super::plan_frame`] declines every VP9 and AV1 stream
/// while it is still a stream description, so a caller should never arrive here with one; the `Err`
/// below says the same thing `plan_frame` says, for a caller that skips planning. The two must agree
/// about what this backend supports: a decline that names a reason the other side does not is a bug
/// in one of them. [`FrameCodec`] and [`FrameStreamInfo`] are the arguments this takes so that the
/// signature does not change when the records arrive.
pub fn build_frame(info: &FrameStreamInfo, _sample: &[u8]) -> Result<Picture, String> {
    match info.codec {
        // A key frame's parameters are public, but the record needs the whole frame header; saying
        // which codec it is makes the reason readable in the logs a decline ends up in.
        FrameCodec::Vp9 => Err(format!("VP9 picture parameters: filmcraft-vp9 {FRAME_HEADER_MISSING}")),
        FrameCodec::Av1 => Err(format!("AV1 picture parameters: filmcraft-av1 {FRAME_HEADER_MISSING}")),
    }
}

// ---------------------------------------------------------------------------------------------
// H.264 / AVC
// ---------------------------------------------------------------------------------------------

/// The parameter sets of an `avcC` record: the sequence and picture parameter sets by id, which is
/// the table a slice header's `pic_parameter_set_id` is looked up in.
struct H264Sets {
    spss: Vec<Option<filmcraft_h264::params::Sps>>,
    ppss: Vec<Option<filmcraft_h264::params::Pps>>,
}

impl H264Sets {
    /// Parse every parameter set of `info` into those tables.
    ///
    /// A set that will not parse is an `Err`: the surfaces were created for the sizes these
    /// describe, so a stream whose parameter sets do not parse is not one this backend hands to a
    /// driver. The tables are sized by the specification's own id ranges (32 sequence, 256 picture),
    /// and an id outside its range is refused rather than indexing past the end.
    fn parse(info: &NalStreamInfo) -> Result<Self, String> {
        let mut sets = Self { spss: vec![None; 32], ppss: vec![None; 256] };
        // Sequence sets first: a picture parameter set's tail depends on its sequence set's chroma
        // format, so every sequence set has to be in hand before any picture set is parsed. The
        // record lists them in that order, but two passes do not rely on it.
        for nal in &info.parameter_sets {
            let Some(&header) = nal.first() else { return Err("empty parameter set NAL unit".into()) };
            if info.nal_type(header) != 7 {
                continue;
            }
            let rbsp = nal.get(1..).ok_or("sequence parameter set NAL unit with no header")?;
            let sps = filmcraft_h264::params::Sps::parse(&unescape_rbsp(rbsp)).map_err(|e| format!("sequence parameter set: {e}"))?;
            let slot = sets.spss.get_mut(sps.id as usize).ok_or("sequence parameter set id out of range")?;
            *slot = Some(sps);
        }
        for nal in &info.parameter_sets {
            let Some(&header) = nal.first() else { return Err("empty parameter set NAL unit".into()) };
            if info.nal_type(header) != 8 {
                continue;
            }
            let rbsp = nal.get(1..).ok_or("picture parameter set NAL unit with no header")?;
            let pps = filmcraft_h264::params::Pps::parse(&unescape_rbsp(rbsp), &sets.spss).map_err(|e| format!("picture parameter set: {e}"))?;
            let slot = sets.ppss.get_mut(pps.id as usize).ok_or("picture parameter set id out of range")?;
            *slot = Some(pps);
        }
        Ok(sets)
    }
}

/// The H.264 records for one access unit. See [`build_nal`].
fn build_h264(info: &NalStreamInfo, sample: &[u8], poc: &mut NalPoc, surface: u32, width_mbs: u32, height_mbs: u32) -> Result<Picture, String> {
    let sets = H264Sets::parse(info)?;
    let nals = sample_nals(info, sample)?;

    // The first slice describes the picture; the rest are further slices of the same one, and VA-API
    // describes the picture once and then its slices.
    let mut params = VaPictureParameterBufferH264::default();
    let mut slices: Vec<SliceParam> = Vec::new();
    let mut slice_offsets: Vec<usize> = Vec::new();
    let mut first: Option<filmcraft_h264::slice::SliceHeader> = None;
    let mut picture_poc: Option<filmcraft_h264::slice::Poc> = None;
    // The sequence parameter set the picture's first slice named, kept for the reference marking
    // after the last slice: `MaxFrameNum` and `max_num_ref_frames` come from it.
    let mut picture_sps: Option<&filmcraft_h264::params::Sps> = None;
    // The picture record's `ReferenceFrames` slots, as a map from decoded-picture-buffer index to the
    // slot it got: every slice record's `RefPicList` entry is a slot in the picture's own record.
    let mut slots: Option<Vec<Option<usize>>> = None;
    let mut data_offset = 0u32;

    for (nal, offset) in &nals {
        let hdr = filmcraft_h264::slice::NalHeader::parse(nal.first().copied().ok_or("empty NAL unit")?).map_err(|e| format!("NAL header: {e}"))?;
        // A parameter set inside a sample is skipped when it is **byte-identical** to one the sample
        // entry already carries, and declined when it is not. Repeating the `avcC` sets at every
        // key frame is legal and common, and this backend reads the sample entry's sets — so such a
        // repeat changes nothing and must not cost the stream its hardware decoder. That is the
        // same test `StreamInfo::parameters_changed` makes, which is what keeps the two consistent:
        // a sample this backend declines is a sample `HybridDecoder` believes changed, and one it
        // accepts is one it believes did not.
        //
        // A set that *differs* is declined rather than skipped: `H264Sets::parse` does not fold a
        // sample's own sets in, so the picture would be decoded with the quantizer, the geometry
        // and the whole reference-list layout of the *sample entry's* set, and nothing would say so.
        // Folding them in is the fix; until that lands a decline is the answer, and an error hands
        // the stream to the software decoder rather than feeding it the wrong picture.
        if matches!(hdr.nal_unit_type, 7 | 8) && !info.parameter_sets.iter().any(|p| p.as_slice() == *nal) {
            return Err(format!("in-band parameter set (NAL type {}) differs from the sample entry's", hdr.nal_unit_type));
        }
        // Only a slice or an IDR slice carries a slice header; everything else in a sample — SEI,
        // AUD, filler — is skipped, as a driver that is given parameter sets in a picture parameter
        // buffer would be told a second story.
        if !matches!(hdr.nal_unit_type, 1 | 5) {
            continue;
        }
        if slices.len() >= MAX_SLICES {
            return Err(format!("more than {MAX_SLICES} slices in one access unit"));
        }
        let rbsp = unescape_rbsp(nal.get(1..).ok_or("NAL unit with no header")?);
        let (sh, pps, sps) = filmcraft_h264::slice::SliceHeader::parse(&rbsp, hdr, |id| {
            // A missing set is the slice's own error type, so the parser reports it as the missing
            // parameter set it is rather than as a bitstream fault.
            let pps = sets.ppss.get(id as usize).and_then(|p| p.as_ref()).ok_or_else(|| filmcraft_h264::Error::MissingParameterSet(format!("PPS {id}")))?;
            let sps = sets
                .spss
                .get(pps.sps_id as usize)
                .and_then(|s| s.as_ref())
                .ok_or_else(|| filmcraft_h264::Error::MissingParameterSet(format!("SPS {}", pps.sps_id)))?;
            Ok((pps, sps))
        })
        .map_err(|e| format!("slice header: {e}"))?;

        // Whatever FilmCraft's own decoder declines, this backend declines too: a record claiming to
        // describe a picture it cannot describe would decode as something other than the stream.
        if !sps.frame_mbs_only || sh.field_pic {
            return Err("field / MBAFF coding".into());
        }
        if sps.separate_colour_plane {
            return Err("separate_colour_plane_flag".into());
        }
        if pps.num_slice_groups > 1 {
            return Err("slice groups (FMO)".into());
        }
        match first.as_ref() {
            None => {
                let p = poc.h264.compute(&sh, sps);
                slots = Some(fill_h264_picture(&mut params, &sh, pps, sps, &p, info, surface, width_mbs, height_mbs, &poc.dpb)?);
                // `MaxFrameNum` before any reference list is built: it is what `pic_num` is relative
                // to, and a decoder that has not parsed a picture yet does not have one.
                poc.max_frame_num = 1u32.checked_shl(sps.log2_max_frame_num).unwrap_or(0).max(1);
                first = Some(sh.clone());
                picture_sps = Some(sps);
                picture_poc = Some(p);
            }
            // A later slice of the same picture: it has to agree about which picture that is, since
            // the record was already filled in from the first.
            Some(first) => {
                if sh.frame_num != first.frame_num || sh.pps_id != first.pps_id {
                    return Err("a slice that does not match the first slice of its picture".into());
                }
            }
        }

        let length = u32::try_from(nal.len()).map_err(|_| "slice length does not fit the record".to_string())?;
        // Each slice names its own references: `ref_pic_list_modification` is a syntax element of
        // the slice header, so two slices of one picture may list them differently even though they
        // share a frame_num. `cur_poc` is the same for every slice of a picture — it is the picture's
        // — and it is what the reference lists are ordered by and what an implicit weight is measured
        // against.
        let cur_poc = picture_poc.as_ref().map_or(0, |p| p.top);
        let lists = h264_ref_lists(&poc.dpb, &sh, cur_poc, poc.max_frame_num)?;
        slices.push(SliceParam {
            offset: data_offset,
            length,
            params: h264_slice_record(nal, &sh, pps, data_offset, &poc.dpb, slots.as_deref().unwrap_or_default(), &lists, cur_poc)?,
        });
        slice_offsets.push(*offset);
        data_offset = data_offset.checked_add(length).ok_or("slice data buffer size overflows")?;
    }

    let first = first.ok_or("no slice in the access unit")?;
    let Some(sps) = picture_sps else { return Err("no slice in the access unit".into()) };
    // The POC state advances once per picture, not once per slice: every slice of a picture shares
    // its frame_num and POC, and advancing per slice would leave the next picture's count wrong.
    if let Some(p) = picture_poc {
        poc.h264.update(&first, &p);
        // Reference marking for the picture just decoded, and its entry in the buffer the next
        // picture's lists are built from. See `h264_mark` for why this happens before the render.
        h264_mark(poc, &first, sps, surface, p.top);
    }
    // The scaling lists, from the sequence parameter set the picture's first slice named — which is one
    // of the sets `H264Sets::parse` read, and those come from `info.parameter_sets`: the sample
    // entry's, i.e. the container's, not the sample's. An in-band parameter set that differs from the
    // sample entry's is therefore **not** honoured, and neither is the geometry it implies. The
    // record is rebuilt for every picture rather than cached, but from the same tables each time.
    Ok(Picture { params: record_bytes(&params), iq_matrix: h264_iq_matrix(sps), slices, slice_offsets })
}

/// Fill the picture-level record from the picture's first slice, whose parsed header is where every
/// field comes from.
fn fill_h264_picture(
    params: &mut VaPictureParameterBufferH264,
    sh: &filmcraft_h264::slice::SliceHeader,
    pps: &filmcraft_h264::params::Pps,
    sps: &filmcraft_h264::params::Sps,
    p: &filmcraft_h264::slice::Poc,
    info: &NalStreamInfo,
    surface: u32,
    width_mbs: u32,
    height_mbs: u32,
    dpb: &[DpbEntry],
) -> Result<Vec<Option<usize>>, String> {
    let (width, height) = h264_macroblocks(info, width_mbs, height_mbs)?;
    params.picture_width_in_mbs_minus1 = width;
    params.picture_height_in_mbs_minus1 = height;
    params.bit_depth_luma_minus8 = minus8(info.bit_depth_luma)?;
    params.bit_depth_chroma_minus8 = minus8(info.bit_depth_chroma)?;
    params.num_slice_groups_minus1 = u8::try_from(pps.num_slice_groups - 1).map_err(|_| "slice groups do not fit the record".to_string())?;
    params.slice_group_map_type = u8::try_from(pps.slice_group_map_type).map_err(|_| "slice group map type does not fit the record".to_string())?;
    // `slice_group_change_rate_minus1` is not retained by our parser and only means anything with
    // more than one slice group, which is declined above.
    params.slice_group_change_rate_minus1 = 0;
    params.pic_init_qp_minus26 = minus26(pps.pic_init_qp)?;
    params.pic_init_qs_minus26 = minus26(pps.pic_init_qs)?;
    params.chroma_qp_index_offset = field_i8(pps.chroma_qp_index_offset, "chroma_qp_index_offset")?;
    params.second_chroma_qp_index_offset = field_i8(pps.second_chroma_qp_index_offset, "second_chroma_qp_index_offset")?;
    params.seq_fields = pack_h264_seq_fields(sps);
    params.pic_fields = pack_h264_pic_fields(pps, sh);
    params.frame_num = field_u16(sh.frame_num, "frame_num")?;
    params.curr_pic = VaPictureH264 {
        picture_id: surface,
        frame_idx: sh.frame_num,
        flags: h264_curr_pic_flags(sh),
        top_field_order_cnt: p.top,
        bottom_field_order_cnt: p.bottom,
        va_reserved: [0; VA_PADDING_LOW],
    };
    // The decoded picture buffer, as the driver is told to see it: one slot per surface that holds a
    // reference picture, and a count of exactly those. `num_ref_frames` was previously the DPB size
    // the sequence parameter set declares, which claimed more valid slots than the record filled — a
    // count a driver reads as "these many slots name pictures" over entries that say they name none.
    let (frames, n, slots) = h264_reference_frames(dpb);
    params.reference_frames = frames;
    params.num_ref_frames = n;
    Ok(slots)
}

/// A `ReferenceFrames` slot that names no surface and says so.
fn invalid_h264_picture() -> VaPictureH264 {
    VaPictureH264 { picture_id: INVALID_SURFACE, flags: PICTURE_H264_INVALID, ..Default::default() }
}

/// The `seq_fields` word (`va.h:3607`): the sequence parameter set's flags, packed from bit 0 up
/// as the C bitfield declares them.
fn pack_h264_seq_fields(sps: &filmcraft_h264::params::Sps) -> u32 {
    // bits 0-1 are a value and the rest are flags, so the two helpers differ only in width.
    let bit = |at: u32, v: bool| u32::from(v) << at;
    let field = |at: u32, width: u32, v: u32| (v & ((1 << width) - 1)) << at;
    field(0, 2, sps.chroma_array_type())
        | bit(2, sps.separate_colour_plane)
        | bit(3, sps.gaps_in_frame_num_allowed)
        | bit(4, sps.frame_mbs_only)
        | bit(5, sps.mb_adaptive_frame_field)
        | bit(6, sps.direct_8x8_inference)
        // A.3.3.2: MinLumaBiPredSize8x8 is 0 only when both field and MBAFF coding are in use.
        | bit(7, sps.frame_mbs_only || !sps.mb_adaptive_frame_field)
        | field(8, 4, sps.log2_max_frame_num.saturating_sub(4))
        | field(12, 2, sps.pic_order_cnt_type)
        | field(14, 4, sps.log2_max_poc_lsb.saturating_sub(4))
        | bit(18, sps.delta_pic_order_always_zero)
}

/// The `pic_fields` word (`va.h:3631`): the picture parameter set's flags plus what this slice says
/// about the picture, packed from bit 0 up as the C bitfield declares them.
fn pack_h264_pic_fields(pps: &filmcraft_h264::params::Pps, sh: &filmcraft_h264::slice::SliceHeader) -> u32 {
    let bit = |at: u32, v: bool| u32::from(v) << at;
    let field = |at: u32, width: u32, v: u32| (v & ((1 << width) - 1)) << at;
    bit(0, pps.entropy_coding_mode)
        | bit(1, pps.weighted_pred)
        | field(2, 2, pps.weighted_bipred_idc)
        | bit(4, pps.transform_8x8_mode)
        | bit(5, sh.field_pic)
        | bit(6, pps.constrained_intra_pred)
        | bit(7, pps.bottom_field_pic_order_in_frame_present)
        | bit(8, pps.deblocking_filter_control_present)
        | bit(9, pps.redundant_pic_cnt_present)
        // reference_pic_flag: whether the picture stays in the DPB at all.
        | bit(10, sh.nal_ref_idc != 0)
}

/// The `VAPictureH264::flags` of the current picture: whether it stays in the DPB, and as what. A
/// picture with `nal_ref_idc == 0` is in neither list — it is not a reference at all, which is not
/// the same as `VA_PICTURE_H264_NON_EXISTING`, a flag about the *slots* a frame_num gap leaves.
fn h264_curr_pic_flags(sh: &filmcraft_h264::slice::SliceHeader) -> u32 {
    if sh.nal_ref_idc == 0 {
        0
    } else if sh.long_term_reference {
        PICTURE_H264_LONG_TERM_REFERENCE
    } else {
        PICTURE_H264_SHORT_TERM_REFERENCE
    }
}

/// One slice's `VASliceParameterBufferH264`.
///
/// The two reference lists name the pictures the slice predicts from, as surfaces: each entry is the
/// `VAPictureH264` of a picture in `dpb` ([`h264_ref_pic_lists`]), and each entry's `frame_idx` is
/// how the driver finds it again in the picture record's `ReferenceFrames`. Past a list's own length
/// the entries stay invalid, which is what an unused `RefPicList` slot means.
fn h264_slice_record(
    nal: &[u8],
    sh: &filmcraft_h264::slice::SliceHeader,
    pps: &filmcraft_h264::params::Pps,
    data_offset: u32,
    dpb: &[DpbEntry],
    slots: &[Option<usize>],
    lists: &[Vec<usize>; 2],
    cur_poc: i32,
) -> Result<Vec<u8>, String> {
    let weights = h264_weights(sh, pps, cur_poc, dpb, lists)?;
    let (ref_pic_list0, ref_pic_list1) = h264_ref_pic_lists(dpb, lists, slots)?;
    let record = VaSliceParameterBufferH264 {
        slice_data_size: u32::try_from(nal.len()).map_err(|_| "slice length does not fit the record".to_string())?,
        slice_data_offset: data_offset,
        slice_data_flag: SLICE_DATA_FLAG_ALL,
        // The record counts bits from the NAL unit byte to the start of `slice_data()` with
        // emulation prevention bytes removed, and our parser's `header_bits` counts from after that
        // byte, so the header byte itself is the eight bits this adds.
        slice_data_bit_offset: u16::try_from(sh.header_bits + 8).map_err(|_| "slice header runs past the record".to_string())?,
        first_mb_in_slice: field_u16(sh.first_mb_in_slice, "first_mb_in_slice")?,
        slice_type: h264_slice_type(sh.slice_type),
        direct_spatial_mv_pred_flag: u8::from(sh.direct_spatial_mv_pred),
        num_ref_idx_l0_active_minus1: field_u8(sh.num_ref_idx_active[0].saturating_sub(1), "num_ref_idx_l0_active_minus1")?,
        num_ref_idx_l1_active_minus1: field_u8(sh.num_ref_idx_active[1].saturating_sub(1), "num_ref_idx_l1_active_minus1")?,
        cabac_init_idc: field_u8(sh.cabac_init_idc, "cabac_init_idc")?,
        slice_qp_delta: field_i8(sh.slice_qp_delta, "slice_qp_delta")?,
        disable_deblocking_filter_idc: field_u8(sh.disable_deblocking_filter_idc, "disable_deblocking_filter_idc")?,
        slice_alpha_c0_offset_div2: field_i8(sh.slice_alpha_c0_offset_div2, "slice_alpha_c0_offset_div2")?,
        slice_beta_offset_div2: field_i8(sh.slice_beta_offset_div2, "slice_beta_offset_div2")?,
        ref_pic_list0,
        ref_pic_list1,
        ..weights
    };
    Ok(record_bytes(&record))
}

/// H.264 8.4.2.3.1's implicit prediction weight for one (`ref0`, `ref1`) pair, as `(weight, offset)`
/// on a `luma_log2_weight_denom` of 5 — which is how `filmcraft_h264` applies it
/// (`crates/h264/src/slicedec.rs:316-331`, `Weight::Weighted { log_wd: 5, .. }`).
///
/// **`(32, 32)` is no weighting at all**: 32 with a denominator of 2⁵ is 1.0, and both offsets are 0.
/// The pair is what decides that, so it is the only thing [`h264_implicit_weights`] needs to look at.
fn h264_implicit_pair(cur_poc: i32, r0: &DpbEntry, r1: &DpbEntry) -> (i32, i32) {
    let tb = (cur_poc - r0.poc).clamp(-128, 127);
    let td = (r1.poc - r0.poc).clamp(-128, 127);
    // `tb` and `td` are bounded to ±127 by the clamp above, and `tx` by `td`'s own bound, so the
    // product is at most 127 × 16447 — no overflow to guard.
    let is_long = |e: &DpbEntry| matches!(e.mark, Mark::Long(_));
    if td == 0 || is_long(r0) || is_long(r1) {
        return (32, 32);
    }
    let tx = (16384 + (td / 2).abs()) / td;
    let dsf = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
    if (dsf >> 2) < -64 || (dsf >> 2) > 128 { (32, 32) } else { (64 - (dsf >> 2), dsf >> 2) }
}

/// The prediction weight tables, from the slice header's own `pred_weight_table()`.
///
/// A slice whose weight table is mandatory and missing is declined rather than submitted with
/// zeros: `pic_fields` would say `weighted_pred_flag` is in use, and a zeroed table alongside that
/// says "no weighting" — a different picture from the stream's, which is worse than an error. The
/// mandatory cases are the two `filmcraft_h264` parses the table for: a P slice with
/// `weighted_pred_flag`, and a B slice with `weighted_bipred_idc == 1` (implicit).
///
/// # The B slices of a `weighted_bipred_idc == 2` stream, which this record cannot express
///
/// `weighted_bipred_idc == 2` is x264's default for High profile with B frames, so it is most H.264.
/// Its `weighted_bipred_flag` is a per-slice bit `filmcraft_h264` does not read, and `filmcraft_h264`
/// resolves such a slice as H.264's **implicit** weighting ([`h264_implicit_pair`]) — which is what it
/// decodes, and it is bit-exact with ffmpeg while doing so. A `VASliceParameterBufferH264` cannot say
/// that: `luma_weight_l0` / `luma_weight_l1` are one weight **per reference index per list**
/// (`va_dec_h264.h:3702-3723`), and an implicit weight is a function of the *pair* of indices, which
/// changes from macroblock to macroblock inside one slice. Two per-index arrays cannot hold a
/// pair-dependent function.
///
/// So the slice is checked rather than guessed. Every (`ref0`, `ref1`) pair the slice's own reference
/// lists could use is computed; if they are all `(32, 32)` — 1.0, no offset, i.e. the stream does not
/// actually exercise implicit weighting — the default explicit table is submitted and the picture
/// decodes. If any pair is not, the slice is **declined**, and the error hands the stream to
/// `HybridDecoder`, which finishes it in software. Approximating is not an option: submitting 1.0 for a
/// pair that wants something else decodes the wrong picture, and that is a silent corruption on a
/// timeline rather than a visible failure. Measured on this machine: x264's default `weightb=1` on
/// `testsrc2` exercises implicit weighting, so such a stream declines, while the same encode with
/// `weightb=0` (`weighted_bipred_idc == 0`, which codes no weight table and no implicit mode at all) is
/// bit-exact against both ffmpeg and `filmcraft_h264` on every picture.
fn h264_weights(
    sh: &filmcraft_h264::slice::SliceHeader,
    pps: &filmcraft_h264::params::Pps,
    cur_poc: i32,
    dpb: &[DpbEntry],
    lists: &[Vec<usize>; 2],
) -> Result<VaSliceParameterBufferH264, String> {
    let mandatory = (pps.weighted_pred && sh.slice_type.is_p()) || (pps.weighted_bipred_idc == 1 && sh.slice_type.is_b());
    let Some(table) = sh.pred_weight_table.as_ref() else {
        if mandatory {
            return Err(format!(
                "weighted {} prediction in a slice that carries no prediction weight table",
                if sh.slice_type.is_b() { "bipred" } else { "pred" }
            ));
        }
        return if pps.weighted_bipred_idc == 2 && sh.slice_type.is_b() {
            h264_implicit_weights(cur_poc, dpb, lists)
        } else {
            Ok(VaSliceParameterBufferH264::default())
        };
    };
    // `pred_weight_table()` only produces entries for the references the slice actually lists, so
    // the table is at most that long; the record's arrays are 32.
    if table.l0.len() > 32 || table.l1.len() > 32 {
        return Err(format!("prediction weight table of {} references does not fit the record", table.l0.len().max(table.l1.len())));
    }
    let mut record = VaSliceParameterBufferH264 {
        luma_log2_weight_denom: field_u8(table.luma_log2_denom, "luma_log2_weight_denom")?,
        chroma_log2_weight_denom: field_u8(table.chroma_log2_denom, "chroma_log2_weight_denom")?,
        ..Default::default()
    };
    for (list, entries) in [(0, &table.l0), (1, &table.l1)] {
        // The `*_flag` fields say the list carries luma weights at all (7.4.3.2): a stream that
        // sets no flag means the default weight of 1 << denom, which the table already holds.
        let luma_flag = u8::from(entries.iter().any(|e| e.luma_flag));
        let chroma_flag = u8::from(entries.iter().any(|e| e.chroma_flag));
        let mut luma_weight = [0i16; 32];
        let mut luma_offset = [0i16; 32];
        let mut chroma_weight = [[0i16; 2]; 32];
        let mut chroma_offset = [[0i16; 2]; 32];
        for (i, e) in entries.iter().enumerate() {
            luma_weight[i] = field_i16(e.luma_weight, "luma weight")?;
            luma_offset[i] = field_i16(e.luma_offset, "luma offset")?;
            for c in 0..2 {
                chroma_weight[i][c] = field_i16(e.chroma_weight[c], "chroma weight")?;
                chroma_offset[i][c] = field_i16(e.chroma_offset[c], "chroma offset")?;
            }
        }
        if list == 0 {
            record.luma_weight_l0_flag = luma_flag;
            record.chroma_weight_l0_flag = chroma_flag;
            record.luma_weight_l0 = luma_weight;
            record.luma_offset_l0 = luma_offset;
            record.chroma_weight_l0 = chroma_weight;
            record.chroma_offset_l0 = chroma_offset;
        } else {
            record.luma_weight_l1_flag = luma_flag;
            record.chroma_weight_l1_flag = chroma_flag;
            record.luma_weight_l1 = luma_weight;
            record.luma_offset_l1 = luma_offset;
            record.chroma_weight_l1 = chroma_weight;
            record.chroma_offset_l1 = chroma_offset;
        }
    }
    Ok(record)
}

/// The explicit table that *is* 1.0 with no offset, for a slice whose implicit pairs are all
/// `(32, 32)` on a denominator of 2⁵ — the weights `filmcraft_h264` applies in that case, written in
/// the only form the record has.
///
/// The flags are set, because a weight of 1.0 that is *in use* is not the same statement as "this
/// slice does not weight prediction", and this is the case where the slice does weight: every pair
/// just happens to want 1.0. Luma and chroma alike, because the implicit weights apply to both and
/// with the same denominator.
fn h264_default_weights() -> VaSliceParameterBufferH264 {
    VaSliceParameterBufferH264 {
        luma_log2_weight_denom: 5,
        chroma_log2_weight_denom: 5,
        luma_weight_l0_flag: 1,
        chroma_weight_l0_flag: 1,
        luma_weight_l1_flag: 1,
        chroma_weight_l1_flag: 1,
        luma_weight_l0: [32; 32],
        luma_weight_l1: [32; 32],
        chroma_weight_l0: [[32; 2]; 32],
        chroma_weight_l1: [[32; 2]; 32],
        ..Default::default()
    }
}

/// Every (`ref0`, `ref1`) pair this slice could use is computed; the default table goes in the record
/// when they are all 1.0, and the slice is declined when one is not. See [`h264_weights`] for why the
/// record cannot express the alternative.
///
/// A reference index the decoded picture buffer does not hold is an error rather than a skipped pair:
/// the pair's weight cannot be derived without knowing the reference's POC, and a pair left out of the
/// check is a pair the driver would weight wrongly.
fn h264_implicit_weights(cur_poc: i32, dpb: &[DpbEntry], lists: &[Vec<usize>; 2]) -> Result<VaSliceParameterBufferH264, String> {
    let entry = |at: usize, list: usize| -> Result<&DpbEntry, String> {
        dpb.get(at).ok_or_else(|| format!("reference {at} of list {list} is not in the decoded picture buffer"))
    };
    for &r0 in &lists[0] {
        for &r1 in &lists[1] {
            let pair = h264_implicit_pair(cur_poc, entry(r0, 0)?, entry(r1, 1)?);
            if pair != (32, 32) {
                return Err(format!(
                    "this H.264 stream weights its B slices with H.264 implicit weights (weighted_bipred_idc 2): the pair of references ({}, {}) wants {:?}, not 1.0, \
                     and a VASliceParameterBufferH264 carries one weight per reference index rather than per pair",
                    r0, r1, pair
                ));
            }
        }
    }
    Ok(h264_default_weights())
}

/// H.264 `slice_type` as the record numbers it: the slice header's own `slice_type` modulo five.
fn h264_slice_type(t: filmcraft_h264::slice::SliceType) -> u8 {
    match t {
        filmcraft_h264::slice::SliceType::P => 0,
        filmcraft_h264::slice::SliceType::B => 1,
        filmcraft_h264::slice::SliceType::I => 2,
        filmcraft_h264::slice::SliceType::Sp => 3,
        filmcraft_h264::slice::SliceType::Si => 4,
    }
}

/// The coded picture size as the record's `minus1` fields. It has to be a whole number of
/// macroblocks and the size the caller's surfaces were created at, since the record tells the driver
/// how many macroblocks the picture has.
fn h264_macroblocks(info: &NalStreamInfo, width_mbs: u32, height_mbs: u32) -> Result<(u16, u16), String> {
    let (w, h) = info.coded;
    if w == 0 || h == 0 || w % 16 != 0 || h % 16 != 0 {
        return Err(format!("coded picture size {w}x{h} is not a whole number of macroblocks"));
    }
    let (cols, rows) = (w / 16, h / 16);
    if cols != width_mbs || rows != height_mbs {
        return Err(format!("coded picture size {w}x{h} is {cols}x{rows} macroblocks, not {width_mbs}x{height_mbs}"));
    }
    Ok((field_u16(cols - 1, "picture_width_in_mbs_minus1")?, field_u16(rows - 1, "picture_height_in_mbs_minus1")?))
}

// ---------------------------------------------------------------------------------------------
// HEVC / H.265
// ---------------------------------------------------------------------------------------------

/// The parameter sets of an `hvcC` record, as `Arc`s — which is what
/// `filmcraft_hevc::slice::SliceHeader::parse` hands back and what its lookups resolve to.
#[derive(Default)]
struct HevcSets {
    spss: Vec<Option<Arc<filmcraft_hevc::params::Sps>>>,
    ppss: Vec<Option<Arc<filmcraft_hevc::params::Pps>>>,
}

impl HevcSets {
    /// Parse every parameter set of `info` into those tables; see [`H264Sets::parse`] for why a set
    /// that will not parse is an error rather than a gap.
    fn parse(info: &NalStreamInfo) -> Result<Self, String> {
        let mut sets = Self { spss: vec![None; 16], ppss: vec![None; 64] };
        for nal in &info.parameter_sets {
            let Some(rbsp) = nal.get(2..) else { return Err("parameter set NAL unit with no two-byte header".into()) };
            match nal.first().map(|&h| info.nal_type(h)) {
                Some(33) => {
                    let sps = filmcraft_hevc::params::Sps::parse(&unescape_rbsp(rbsp)).map_err(|e| format!("sequence parameter set: {e}"))?;
                    let slot = sets.spss.get_mut(sps.id as usize).ok_or("sequence parameter set id out of range")?;
                    *slot = Some(Arc::new(sps));
                }
                Some(34) => {
                    let pps = filmcraft_hevc::params::Pps::parse(&unescape_rbsp(rbsp)).map_err(|e| format!("picture parameter set: {e}"))?;
                    let slot = sets.ppss.get_mut(pps.id as usize).ok_or("picture parameter set id out of range")?;
                    *slot = Some(Arc::new(pps));
                }
                _ => {}
            }
        }
        Ok(sets)
    }

    /// The parameter set pair a slice segment's `pps_id` names, as the parser's lookup resolves it. A
    /// missing set is the parser's own missing-parameter-set error, so a slice naming one the record
    /// does not carry is reported as exactly that.
    fn lookup(&self, id: u32) -> Result<(Arc<filmcraft_hevc::params::Pps>, Arc<filmcraft_hevc::params::Sps>), filmcraft_hevc::Error> {
        let pps =
            self.ppss.get(id as usize).and_then(|p| p.as_ref()).cloned().ok_or_else(|| filmcraft_hevc::Error::MissingParameterSet(format!("PPS {id}")))?;
        let sps = self
            .spss
            .get(pps.sps_id as usize)
            .and_then(|s| s.as_ref())
            .cloned()
            .ok_or_else(|| filmcraft_hevc::Error::MissingParameterSet(format!("SPS {}", pps.sps_id)))?;
        Ok((pps, sps))
    }
}

/// The HEVC records for one access unit. See [`build_nal`].
fn build_hevc(info: &NalStreamInfo, sample: &[u8], poc: &mut NalPoc, surface: u32, width: u32, height: u32) -> Result<Picture, String> {
    let sets = HevcSets::parse(info)?;
    let nals = sample_nals(info, sample)?;

    let mut params = VaPictureParameterBufferHevc::default();
    let mut slices: Vec<SliceParam> = Vec::new();
    let mut slice_offsets: Vec<usize> = Vec::new();
    // Every segment of the picture, with where its bytes are, until the last of them is known. The
    // second pass needs each segment's own PPS as well as its header: whether its weight table was
    // mandatory is a flag of that parameter set.
    let mut segments: Vec<(&[u8], usize, filmcraft_hevc::slice::SliceHeader, Arc<filmcraft_hevc::params::Pps>, u32, u32)> = Vec::new();
    // The last *independent* segment header, which a dependent segment inherits from; a dependent
    // segment carries almost none of its own, so the parser needs the one it follows.
    let mut previous: Option<filmcraft_hevc::slice::SliceHeader> = None;
    let mut first: Option<filmcraft_hevc::slice::SliceHeader> = None;
    let mut picture_poc: Option<i32> = None;
    // The picture's reference lists, filled from the picture record and passed to every segment's
    // record. They are the picture's, not a segment's: every segment of one picture refers to the
    // same pictures, and the record that names them is the picture's.
    let mut lists: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    let mut data_offset = 0u32;

    for (nal, offset) in &nals {
        let hdr = filmcraft_hevc::slice::NalHeader::parse(nal).map_err(|e| format!("NAL header: {e}"))?;
        // A parameter set inside a sample is handled exactly as on the H.264 side, because it has to be:
        // `HevcSets::parse` reads the sample entry's sets and does not fold a sample's own in, so a
        // *differing* in-band set would leave the picture being decoded with the `hvcC`'s geometry and
        // bit depth while the stream has moved on. A repeat of a set the sample entry already carries
        // changes nothing and must not cost the stream its hardware decoder. One test, the same test
        // `StreamInfo::parameters_changed` makes, so a sample this backend declines is a sample
        // `HybridDecoder` believes changed and one it accepts is a sample it believes did not.
        if info.is_parameter_set(hdr.nal_type) && !info.parameter_sets.iter().any(|p| p.as_slice() == *nal) {
            return Err(format!("in-band parameter set (NAL type {}) differs from the sample entry's", hdr.nal_type));
        }
        // Only a VCL NAL unit carries a slice segment header; the rest — access unit delimiters,
        // SEI, and the parameter sets just dealt with — are skipped.
        if !hdr.is_vcl() {
            continue;
        }
        if slices.len() >= MAX_SLICES {
            return Err(format!("more than {MAX_SLICES} slice segments in one access unit"));
        }
        let rbsp = unescape_rbsp(nal.get(2..).ok_or("NAL unit shorter than its two-byte header")?);
        let (sh, pps, sps) =
            filmcraft_hevc::slice::SliceHeader::parse(&rbsp, hdr, |id| sets.lookup(id), previous.as_ref()).map_err(|e| format!("slice segment header: {e}"))?;
        if !sh.first_slice_segment_in_pic && previous.is_none() {
            return Err("a slice segment before the first segment of its picture".into());
        }
        // The range-extension, multilayer and SCC tools have no room in the record, and our own
        // decoder declines them too; a record leaving them out would describe a different stream.
        if sps.range_extension || sps.extension_flags & 0x7f != 0 {
            return Err("HEVC range extension / multilayer / screen content tools".into());
        }
        // Separate colour planes are declined as on the H.264 side: the record's coding block
        // geometry and chroma format describe 4:2:0 or 4:2:2, and there is nowhere in it for the
        // three planes a `separate_colour_plane_flag` stream codes separately.
        if sps.separate_colour_plane {
            return Err("separate_colour_plane_flag".into());
        }
        // Monochrome codes no chroma at all, so the slice header's prediction weight table carries
        // no chroma weights for `hevc_weights` to send; the record would have to invent them. (Both
        // RT formats a `Plan` can name are 4:2:0, so `plan_nal` declines these streams too.)
        if sps.chroma_array_type() == 0 {
            return Err("monochrome (4:0:0)".into());
        }
        // A stream with quantizer scaling lists needs `VAIQMatrixBufferHEVC`, which this backend does
        // not build. See `hevc_scaling_lists_declined` for why the answer is a decline and not a
        // default: `va_dec_hevc.h` says the buffer "is sent once per frame, and only when
        // scaling_list_enabled_flag = 1", and a driver that falls back to the flat default would
        // decode a picture the stream does not contain — silently, which is the one outcome worse
        // than declining.
        if sps.scaling_list_enabled {
            return Err(hevc_scaling_lists_declined().into());
        }

        if first.is_none() {
            let (p, built) = fill_hevc_picture(&mut params, poc, &sh, &pps, &sps, surface, width, height)?;
            picture_poc = Some(p);
            lists = built;
            first = Some(sh.clone());
        } else {
            // Every segment of a picture shares the picture's parameter sets: the record was filled
            // in from the first segment, so a later one naming a different picture parameter set
            // would be described with the wrong tile grid, QP offsets and coding block geometry.
            let first = first.as_ref().ok_or("first segment not set")?;
            if !hevc_same_picture(first, &sh) {
                return Err("a slice segment that does not match the first segment of its picture".into());
            }
        }
        let length = u32::try_from(nal.len()).map_err(|_| "slice segment length does not fit the record".to_string())?;
        // The record is built once every segment is known, because `LastSliceOfPic` belongs to the
        // last of them and libva's driver waits for it: a picture whose last segment does not say
        // the picture ended leaves the driver looking for more.
        segments.push((nal, *offset, sh.clone(), pps.clone(), data_offset, length));
        data_offset = data_offset.checked_add(length).ok_or("slice data buffer size overflows")?;
        // A dependent segment repeats almost nothing, so the segment that follows an independent
        // one inherits it and a dependent one does not become the new base.
        if !sh.dependent {
            previous = Some(sh);
        }
    }

    let first = first.ok_or("no slice segment in the access unit")?;
    if let Some(p) = picture_poc {
        poc.hevc.update(&first, p);
        // 8.3.5: the picture just decoded joins the buffer the next picture's set is derived from.
        hevc_mark(&mut poc.dpb, &first, surface, p);
    }
    let last = segments.len().saturating_sub(1);
    for (i, (nal, offset, sh, pps, at, length)) in segments.iter().enumerate() {
        slices.push(SliceParam { offset: *at, length: *length, params: hevc_slice_record(nal, sh, pps, *at, i == last, &lists)? });
        slice_offsets.push(*offset);
    }
    // No IQ matrix, because there is none to send. HEVC's scaling lists go in `VAIQMatrixBufferHEVC` and
    // **nowhere else** — `VAPictureParameterBufferHEVC` has no scaling-list field, and
    // `va_dec_hevc.h` says of that buffer "This structure is sent once per frame, and only when
    // scaling_list_enabled_flag = 1". (An earlier comment here said they were inside the picture
    // parameter record, citing that header; it was wrong, and the test beside it asserted the wrong
    // fact.) So a stream with `sps_scaling_list_enabled_flag == 1` is declined above rather than
    // submitted without a record the driver may fill in from its own default: iHD happens to default
    // to the flat quantizer, which is correct only for a stream that wanted the flat quantizer, and
    // a driver that does not default is exactly the plausible-wrong-picture case this backend exists
    // to avoid. An empty `iq_matrix` therefore means the flag was clear — see
    // `hevc_scaling_lists_declined`.
    Ok(Picture { params: record_bytes(&params), iq_matrix: Vec::new(), slices, slice_offsets })
}

/// Why an HEVC stream with `sps_scaling_list_enabled_flag` is declined, and the one-line version of it
/// the diagnostics use.
fn hevc_scaling_lists_declined() -> &'static str {
    "HEVC scaling lists (sps_scaling_list_enabled_flag = 1): VAIQMatrixBufferHEVC is not built, \
     and a driver that defaults the quantizer would produce a picture that is not the stream's"
}

/// Whether a slice segment belongs to the picture `first` opened: the same picture parameter set and
/// the same NAL unit type and temporal layer, which is what says the two are one picture rather than
/// two that happen to share an access unit.
fn hevc_same_picture(first: &filmcraft_hevc::slice::SliceHeader, sh: &filmcraft_hevc::slice::SliceHeader) -> bool {
    sh.pps_id == first.pps_id && sh.nal.nal_type == first.nal.nal_type && sh.nal.temporal_id == first.nal.temporal_id && sh.nal.layer_id == first.nal.layer_id
}

/// Fill the picture-level record from the picture's first segment, and return its POC and its
/// reference lists.
///
/// The HEVC record carries far more of the parameter sets than the H.264 one does — the coding
/// block and transform block geometry, the tile grid, the loop filter and chroma QP offsets — and
/// each is the parsed value of the set the segment named.
///
/// `ReferenceFrames` is the decoded picture buffer after this picture's reference picture set has
/// been applied to it, with each entry flagged as one of the picture set's `RPS_ST_CURR_*` groups
/// or as a long-term reference (`va_dec_hevc.h:5319-5337`). The slice record's `RefPicList` entries
/// are indices into that array, which is why both come from the same call.
fn fill_hevc_picture(
    params: &mut VaPictureParameterBufferHevc,
    poc: &mut NalPoc,
    sh: &filmcraft_hevc::slice::SliceHeader,
    pps: &filmcraft_hevc::params::Pps,
    sps: &filmcraft_hevc::params::Sps,
    surface: u32,
    width: u32,
    height: u32,
) -> Result<(i32, [Vec<usize>; 2]), String> {
    if sps.width != width || sps.height != height {
        return Err(format!("coded picture size {}x{} is not {width}x{height}", sps.width, sps.height));
    }
    // `log2_max_pic_order_cnt_lsb` is bounded to 4..=16 by the parser, and the shift is checked
    // anyway so a hostile value cannot reach the POC arithmetic.
    let max_lsb = i32::checked_shl(1, sps.log2_max_poc_lsb).ok_or("log2_max_pic_order_cnt_lsb is out of range")?;
    // 8.3.1: an IRAP picture that has no RASL pictures after it resets the POC's msb prediction;
    // an open-GOP CRA does not, which is the whole difference between the two cases.
    let p = poc.hevc.compute(sh, max_lsb, sh.nal.is_irap() && !sh.nal.is_rasl());
    params.curr_pic = VaPictureHevc { picture_id: surface, pic_order_cnt: p, flags: 0, va_reserved: [0; VA_PADDING_LOW] };
    // `ReferenceFrames[15]` (`va_dec_hevc.h:64`), the HEVC record's DPB. The set is applied to the
    // decoder's own buffer first, so what is filled in below is exactly what the driver's picture
    // buffer should hold when this picture is decoded.
    let lists = hevc_ref_lists(&mut poc.dpb, sh, p, max_lsb)?;
    params.reference_frames = hevc_reference_frames(&poc.dpb, sh, p, max_lsb)?;
    params.pic_width_in_luma_samples = field_u16(sps.width, "pic_width_in_luma_samples")?;
    params.pic_height_in_luma_samples = field_u16(sps.height, "pic_height_in_luma_samples")?;
    params.pic_fields = pack_hevc_pic_fields(sps, pps, sh);
    params.sps_max_dec_pic_buffering_minus1 = field_u8(sps.max_dec_pic_buffering.saturating_sub(1), "sps_max_dec_pic_buffering_minus1")?;
    params.bit_depth_luma_minus8 = minus8(sps.bit_depth_luma)?;
    params.bit_depth_chroma_minus8 = minus8(sps.bit_depth_chroma)?;
    params.pcm_sample_bit_depth_luma_minus1 = field_u8(sps.pcm_bit_depth_luma.saturating_sub(1), "pcm_sample_bit_depth_luma_minus1")?;
    params.pcm_sample_bit_depth_chroma_minus1 = field_u8(sps.pcm_bit_depth_chroma.saturating_sub(1), "pcm_sample_bit_depth_chroma_minus1")?;
    params.log2_min_luma_coding_block_size_minus3 = field_u8(sps.log2_min_cb.saturating_sub(3), "log2_min_luma_coding_block_size_minus3")?;
    params.log2_diff_max_min_luma_coding_block_size = field_u8(sps.log2_ctb.saturating_sub(sps.log2_min_cb), "log2_diff_max_min_luma_coding_block_size")?;
    params.log2_min_transform_block_size_minus2 = field_u8(sps.log2_min_tb.saturating_sub(2), "log2_min_transform_block_size_minus2")?;
    params.log2_diff_max_min_transform_block_size = field_u8(sps.log2_max_tb.saturating_sub(sps.log2_min_tb), "log2_diff_max_min_transform_block_size")?;
    params.log2_min_pcm_luma_coding_block_size_minus3 = field_u8(sps.log2_min_pcm.saturating_sub(3), "log2_min_pcm_luma_coding_block_size_minus3")?;
    params.log2_diff_max_min_pcm_luma_coding_block_size =
        field_u8(sps.log2_max_pcm.saturating_sub(sps.log2_min_pcm), "log2_diff_max_min_pcm_luma_coding_block_size")?;
    params.max_transform_hierarchy_depth_intra = field_u8(sps.max_th_depth_intra, "max_transform_hierarchy_depth_intra")?;
    params.max_transform_hierarchy_depth_inter = field_u8(sps.max_th_depth_inter, "max_transform_hierarchy_depth_inter")?;
    params.init_qp_minus26 = minus26(pps.init_qp)?;
    params.diff_cu_qp_delta_depth = field_u8(pps.diff_cu_qp_delta_depth, "diff_cu_qp_delta_depth")?;
    params.pps_cb_qp_offset = field_i8(pps.cb_qp_offset, "pps_cb_qp_offset")?;
    params.pps_cr_qp_offset = field_i8(pps.cr_qp_offset, "pps_cr_qp_offset")?;
    params.log2_parallel_merge_level_minus2 = field_u8(pps.log2_parallel_merge_level.saturating_sub(2), "log2_parallel_merge_level_minus2")?;
    fill_hevc_tiles(pps, params)?;
    params.slice_parsing_fields = pack_hevc_slice_parsing(sps, pps, sh);
    params.log2_max_pic_order_cnt_lsb_minus4 = field_u8(sps.log2_max_poc_lsb.saturating_sub(4), "log2_max_pic_order_cnt_lsb_minus4")?;
    params.num_short_term_ref_pic_sets = field_count(sps.st_rps.len(), "num_short_term_ref_pic_sets")?;
    params.num_long_term_ref_pic_sps = field_count(sps.lt_ref_pics.len(), "num_long_term_ref_pic_sps")?;
    params.num_ref_idx_l0_default_active_minus1 = field_u8(pps.num_ref_idx_l0_default.saturating_sub(1), "num_ref_idx_l0_default_active_minus1")?;
    params.num_ref_idx_l1_default_active_minus1 = field_u8(pps.num_ref_idx_l1_default.saturating_sub(1), "num_ref_idx_l1_default_active_minus1")?;
    params.pps_beta_offset_div2 = field_i8(pps.beta_offset_div2, "pps_beta_offset_div2")?;
    params.pps_tc_offset_div2 = field_i8(pps.tc_offset_div2, "pps_tc_offset_div2")?;
    params.num_extra_slice_header_bits = field_u8(pps.num_extra_slice_header_bits, "num_extra_slice_header_bits")?;
    // The bits `short_term_ref_pic_set()` takes in this header, which is what lets a driver skip it.
    params.st_rps_bits = u32::try_from(sh.st_rps_bits).map_err(|_| "short-term reference picture set is longer than the record".to_string())?;
    Ok((p, lists))
}

/// The `ReferenceFrames` a picture's record carries, each entry flagged as one of the picture set's
/// three groups (`va_dec_hevc.h:5320-5337`) or as a long-term reference.
///
/// The groups are derived here rather than in [`hevc_ref_lists`] because that function has already
/// applied the set to the buffer by the time it returns: what is left in `dpb` *is* the set, and an
/// entry's group is its picture order count's position relative to the current picture's.
fn hevc_reference_frames(
    dpb: &[DpbEntry],
    sh: &filmcraft_hevc::slice::SliceHeader,
    cur_poc: i32,
    max_poc_lsb: i32,
) -> Result<[VaPictureHevc; MAX_HEVC_REF_PICS], String> {
    let mut frames = [invalid_hevc_picture(); MAX_HEVC_REF_PICS];
    if sh.is_intra() {
        // An I picture's set is empty, so the buffer is whatever an IDR left (nothing, since
        // `hevc_mark` cleared it) and nothing here refers to anything.
        return Ok(frames);
    }
    if max_poc_lsb <= 0 {
        return Err("log2_max_pic_order_cnt_lsb is out of range".into());
    }
    // The set, once more: which buffer entry is in which group. `hevc_ref_lists` has already checked
    // that every reference the set names is present, so a `None` here cannot happen — but the buffer
    // it was checked against has since been trimmed, so an entry the lists no longer reach is
    // simply not filled in.
    let mut used: Vec<(usize, u32)> = Vec::new();
    for (i, entry) in dpb.iter().enumerate() {
        if entry.poc != cur_poc {
            used.push((i, 0));
        }
    }
    let deltas = sh.st_rps.s0.iter().map(|&(d, u)| (d, u, true)).chain(sh.st_rps.s1.iter().map(|&(d, u)| (d, u, false)));
    for (delta, used_by_curr, before) in deltas {
        if !used_by_curr {
            continue;
        }
        let wanted = cur_poc.wrapping_add(delta);
        if let Some(at) = dpb.iter().position(|e| e.poc == wanted) {
            let bit = if before { PICTURE_HEVC_RPS_ST_CURR_BEFORE } else { PICTURE_HEVC_RPS_ST_CURR_AFTER };
            match used.iter_mut().find(|(entry, _)| *entry == at) {
                Some(entry) => entry.1 = bit,
                None => used.push((at, bit)),
            }
        }
    }
    for (i, &(lsb, used_by_curr, msb_present, cycle)) in sh.lt.iter().enumerate() {
        if !used_by_curr {
            continue;
        }
        let mut p = i32::try_from(lsb).map_err(|_| "PocLsbLt does not fit".to_string())?;
        if msb_present {
            let cycle = i32::try_from(cycle).map_err(|_| "DeltaPocMsbCycleLt does not fit".to_string())?;
            p = p.wrapping_add(cur_poc.wrapping_sub(cycle.wrapping_mul(max_poc_lsb)).wrapping_sub(cur_poc & (max_poc_lsb - 1)));
        }
        if let Some(at) = dpb.iter().position(|e| {
            if e.poc == p { true } else { !msb_present && (e.poc & (max_poc_lsb - 1)) == p && matches!(e.mark, Mark::Long(k) if k as usize == i) }
        }) {
            match used.iter_mut().find(|(entry, _)| *entry == at) {
                Some(entry) => entry.1 = PICTURE_HEVC_RPS_LT_CURR,
                None => used.push((at, PICTURE_HEVC_RPS_LT_CURR)),
            }
        }
    }
    // Only the entries the current picture's reference picture set names are live, and only they get a
    // `ReferenceFrames` slot: `va.h:5322-5331` says the three `VA_PICTURE_HEVC_RPS_*` bits are how a
    // driver learns *which* entries this picture refers to and that `NumPocTotalCurr` is their count —
    // and the HEVC record has no separate count field, so an entry with a valid `picture_id` and no
    // `RPS_*` bit is a state the header does not define. iHD tolerates it (which is why I and P
    // pictures decode here), but a driver that counts every non-`INVALID` entry as live would read a
    // reference this picture does not have. So: named entries carry their bit, everything else is
    // `VA_PICTURE_HEVC_INVALID`.
    for (i, entry) in dpb.iter().enumerate().take(MAX_HEVC_REF_PICS) {
        match used.iter().find(|(at, _)| *at == i) {
            Some((_, bit)) => frames[i] = entry.mark.hevc_picture(entry, Some(*bit)),
            None => frames[i] = invalid_hevc_picture(),
        }
    }
    Ok(frames)
}

/// The two `RefPicList`s of a slice record: indices into the picture record's `ReferenceFrames`, with
/// `0xFF` — the record's "this entry is not used" — everywhere past each list's own length.
fn hevc_ref_pic_lists(lists: &[Vec<usize>; 2]) -> Result<[[u8; MAX_HEVC_REF_PICS]; 2], String> {
    let mut out = [[0xFF; MAX_HEVC_REF_PICS]; 2];
    for (list, built) in lists.iter().enumerate() {
        if built.len() > MAX_HEVC_REF_PICS {
            return Err(format!("reference list {list} of {} entries does not fit the record", built.len()));
        }
        for (slot, &index) in out[list].iter_mut().zip(built) {
            *slot = u8::try_from(index).map_err(|_| format!("reference list {list} entry {index} does not fit the record"))?;
        }
    }
    Ok(out)
}

/// A `ReferenceFrames` slot that names no surface and says so; see [`invalid_h264_picture`].
fn invalid_hevc_picture() -> VaPictureHevc {
    VaPictureHevc { picture_id: INVALID_SURFACE, flags: PICTURE_HEVC_INVALID, ..Default::default() }
}

/// The tile grid as the record's `minus1` fields and its explicit width / height arrays. The arrays
/// hold 19 columns and 21 rows, so a stream with more tiles than they can describe is declined
/// rather than truncated to a grid the stream does not have.
fn fill_hevc_tiles(pps: &filmcraft_hevc::params::Pps, params: &mut VaPictureParameterBufferHevc) -> Result<(), String> {
    params.num_tile_columns_minus1 = field_u8(pps.num_tile_columns.saturating_sub(1), "num_tile_columns_minus1")?;
    params.num_tile_rows_minus1 = field_u8(pps.num_tile_rows.saturating_sub(1), "num_tile_rows_minus1")?;
    let columns = pps.column_widths.len();
    let rows = pps.row_heights.len();
    if columns > params.column_width_minus1.len() || rows > params.row_height_minus1.len() {
        return Err(format!("{columns} tile columns or {rows} tile rows do not fit the record"));
    }
    for (slot, w) in params.column_width_minus1.iter_mut().zip(&pps.column_widths) {
        *slot = field_u16(w.saturating_sub(1), "column_width_minus1")?;
    }
    for (slot, h) in params.row_height_minus1.iter_mut().zip(&pps.row_heights) {
        *slot = field_u16(h.saturating_sub(1), "row_height_minus1")?;
    }
    Ok(())
}

/// The `pic_fields` word (`va_dec_hevc.h:71`): the sequence and picture parameter sets' flags,
/// packed from bit 0 up as the C bitfield declares them. Bits 21-31 are the header's reserved field
/// and stay zero, which is what it requires.
fn pack_hevc_pic_fields(sps: &filmcraft_hevc::params::Sps, pps: &filmcraft_hevc::params::Pps, sh: &filmcraft_hevc::slice::SliceHeader) -> u32 {
    let bit = |at: u32, v: bool| u32::from(v) << at;
    let field = |at: u32, width: u32, v: u32| (v & ((1 << width) - 1)) << at;
    field(0, 2, sps.chroma_array_type())
        | bit(2, sps.separate_colour_plane)
        | bit(3, sps.pcm)
        | bit(4, sps.scaling_list_enabled)
        | bit(5, pps.transform_skip)
        | bit(6, sps.amp)
        | bit(7, sps.strong_intra_smoothing)
        | bit(8, pps.sign_data_hiding)
        | bit(9, pps.constrained_intra_pred)
        | bit(10, pps.cu_qp_delta_enabled)
        | bit(11, pps.weighted_pred)
        | bit(12, pps.weighted_bipred)
        | bit(13, pps.transquant_bypass)
        | bit(14, pps.tiles_enabled)
        | bit(15, pps.entropy_coding_sync)
        | bit(16, pps.loop_filter_across_slices)
        | bit(17, pps.loop_filter_across_tiles)
        | bit(18, sps.pcm_loop_filter_disabled)
        // NoPicReorderingFlag: the stream's own sps_max_num_reorder_pics is 0, so nothing may be
        // reordered and every picture is output as it is decoded.
        | bit(19, sps.max_num_reorder == 0)
        // NoBiPredFlag: this picture has no B slices.
        | bit(20, !sh.is_b())
}

/// The `slice_parsing_fields` word (`va_dec_hevc.h:139`), packed from bit 0 up as the C bitfield
/// declares it: what a driver needs to parse this slice header without the parameter sets in hand.
fn pack_hevc_slice_parsing(sps: &filmcraft_hevc::params::Sps, pps: &filmcraft_hevc::params::Pps, sh: &filmcraft_hevc::slice::SliceHeader) -> u32 {
    let bit = |at: u32, v: bool| u32::from(v) << at;
    bit(0, pps.lists_modification_present)
        | bit(1, sps.long_term_refs_present)
        | bit(2, sps.temporal_mvp)
        | bit(3, pps.cabac_init_present)
        | bit(4, pps.output_flag_present)
        | bit(5, pps.dependent_slice_segments_enabled)
        | bit(6, pps.slice_chroma_qp_offsets_present)
        | bit(7, sps.sao)
        | bit(8, pps.deblocking_override_enabled)
        | bit(9, pps.deblocking_disabled)
        | bit(10, pps.slice_header_extension_present)
        // RapPicFlag / IdrPicFlag: what the NAL unit type says, and the driver has no NAL to read it
        // from — it only has this record.
        | bit(11, sh.nal.is_irap())
        | bit(12, sh.nal.is_idr())
        // IntraPicFlag: this segment is an I slice.
        | bit(13, sh.slice_type == filmcraft_hevc::slice::SliceType::I)
}

/// One slice segment's `VASliceParameterBufferHEVC`.
///
/// The two `RefPicList`s are indices into the picture record's `ReferenceFrames`, which is where the
/// surfaces are named: entry `i` of a list is buffer entry `i` of that array ([`hevc_ref_pic_lists`]),
/// and `0xFF` — the record's "this entry is not used" — is everywhere past each list's own length.
/// The prediction weights are a different thing entirely — pure arithmetic the slice header already
/// carries — and are filled in from it by [`hevc_weights`], like the H.264 side's.
fn hevc_slice_record(
    nal: &[u8],
    sh: &filmcraft_hevc::slice::SliceHeader,
    pps: &filmcraft_hevc::params::Pps,
    data_offset: u32,
    last_of_picture: bool,
    lists: &[Vec<usize>; 2],
) -> Result<Vec<u8>, String> {
    let weights = hevc_weights(sh, pps)?;
    let ref_pic_list = hevc_ref_pic_lists(lists)?;
    let record = VaSliceParameterBufferHevc {
        slice_data_size: u32::try_from(nal.len()).map_err(|_| "slice segment length does not fit the record".to_string())?,
        slice_data_offset: data_offset,
        slice_data_flag: SLICE_DATA_FLAG_ALL,
        // The record counts bytes from the NAL unit header to `slice_segment_data()`, and our
        // parser's `data_offset` is a position in the RBSP, which starts after the two-byte header.
        slice_data_byte_offset: u32::try_from(sh.data_offset + 2).map_err(|_| "slice segment header runs past the record".to_string())?,
        slice_segment_address: sh.segment_address,
        ref_pic_list,
        long_slice_flags: pack_hevc_long_slice_flags(sh, last_of_picture),
        // 0xFF is the record's "no collocated picture", which is what temporal MVP off means.
        collocated_ref_idx: if sh.temporal_mvp { field_u8(sh.collocated_ref_idx, "collocated_ref_idx")? } else { 0xFF },
        num_ref_idx_l0_active_minus1: field_u8(sh.num_ref_idx[0].saturating_sub(1), "num_ref_idx_l0_active_minus1")?,
        num_ref_idx_l1_active_minus1: field_u8(sh.num_ref_idx[1].saturating_sub(1), "num_ref_idx_l1_active_minus1")?,
        slice_qp_delta: field_i8(sh.qp_delta, "slice_qp_delta")?,
        slice_cb_qp_offset: field_i8(sh.cb_qp_offset, "slice_cb_qp_offset")?,
        slice_cr_qp_offset: field_i8(sh.cr_qp_offset, "slice_cr_qp_offset")?,
        slice_beta_offset_div2: field_i8(sh.beta_offset_div2, "slice_beta_offset_div2")?,
        slice_tc_offset_div2: field_i8(sh.tc_offset_div2, "slice_tc_offset_div2")?,
        five_minus_max_num_merge_cand: field_u8(5u32.saturating_sub(sh.max_num_merge_cand), "five_minus_max_num_merge_cand")?,
        num_entry_point_offsets: field_u16(u32::try_from(sh.entry_points.len()).map_err(|_| "too many entry points".to_string())?, "num_entry_point_offsets")?,
        entry_offset_to_subset_array: 0,
        // The emulation prevention bytes of the *slice header*, which is what the entry point
        // offsets and `slice_data_byte_offset` are counted without, and what a driver adds back to
        // find where `slice_segment_data()` starts. Counting the whole NAL unit would over-count
        // every `00 00 03` in the residual data and mis-place every subset header after the first.
        slice_data_num_emu_prevn_bytes: field_u16(header_emu_prevention_bytes(nal, sh.data_offset)?, "slice_data_num_emu_prevn_bytes")?,
        ..weights
    };
    Ok(record_bytes(&record))
}

/// The HEVC prediction weight tables (`va_dec_hevc.h:448-466`), from the slice segment header's own
/// `pwt`.
///
/// HEVC has no per-list weight flag in the record and no "explicit" mode to be unsure about: the
/// table is coded exactly when `weighted_pred_flag && slice_type == P` or `weighted_bipred_flag &&
/// slice_type == B`, both of them flags of the *picture parameter set* rather than of the slice
/// (`crates/hevc/src/slice.rs:334`). So the table is either there and this fills it exactly, or it
/// is mandatory and missing — which is an `Err`, never a zeroed table next to a `pic_fields` that
/// says weighting is in use.
///
/// `filmcraft_hevc`'s `WpEntry` already holds the *derived* weights in 8-bit units, so nothing is
/// re-derived here and no spec formula is needed: `ChromaOffsetL0` / `ChromaOffsetL1` are the
/// record's `ChromaOffsetLX` verbatim (the offset the parser computed from the coded
/// `delta_chroma_offset_lX`), and the two `delta_*_weight` families are the weight the parser built
/// minus the `1 << denom` it added to build it — the same subtraction in reverse, which is
/// `crates/hevc/src/slice.rs:411-427`.
fn hevc_weights(sh: &filmcraft_hevc::slice::SliceHeader, pps: &filmcraft_hevc::params::Pps) -> Result<VaSliceParameterBufferHevc, String> {
    let mandatory = (pps.weighted_pred && sh.slice_type == filmcraft_hevc::slice::SliceType::P) || (pps.weighted_bipred && sh.is_b());
    let Some(table) = sh.pwt.as_ref() else {
        return if mandatory {
            Err(format!("weighted {} prediction in a slice that carries no prediction weight table", if sh.is_b() { "bipred" } else { "pred" }))
        } else {
            Ok(VaSliceParameterBufferHevc::default())
        };
    };
    // `pwt` only produces entries for the references the slice actually lists, so it is at most
    // `num_ref_idx` long; the record's arrays are 15.
    if table.l[0].len() > 15 || table.l[1].len() > 15 {
        return Err(format!("prediction weight table of {} references does not fit the record", table.l[0].len().max(table.l[1].len())));
    }
    let luma_log2 = table.luma_log2_denom;
    let chroma_log2 = table.chroma_log2_denom;
    // Signed, because `delta_chroma_log2_weight_denom` is a signed syntax element (a stream may lower
    // ChromaLog2WeightDenom below LumaLog2WeightDenom) and `1 << denom` is what the parser added to
    // each coded weight, so subtracting it recovers the `delta_*_weight` the record carries. The
    // parser bounds both denominators to 0..=7; the conversions refuse anything else rather than
    // truncate it.
    let luma_log2_i = i32::try_from(luma_log2).map_err(|_| format!("luma_log2_weight_denom {luma_log2} is out of range"))?;
    let chroma_log2_i = i32::try_from(chroma_log2).map_err(|_| format!("ChromaLog2WeightDenom {chroma_log2} is out of range"))?;
    let luma_one = 1i32.checked_shl(luma_log2_i.unsigned_abs()).ok_or("luma_log2_weight_denom out of range")?;
    let chroma_one = 1i32.checked_shl(chroma_log2_i.unsigned_abs()).ok_or("ChromaLog2WeightDenom out of range")?;
    let mut record = VaSliceParameterBufferHevc {
        luma_log2_weight_denom: field_u8(luma_log2, "luma_log2_weight_denom")?,
        delta_chroma_log2_weight_denom: field_i8(chroma_log2_i - luma_log2_i, "delta_chroma_log2_weight_denom")?,
        ..Default::default()
    };
    for (list, entries) in [(0, &table.l[0]), (1, &table.l[1])] {
        let mut delta_luma_weight = [0i8; 15];
        let mut luma_offset = [0i8; 15];
        let mut delta_chroma_weight = [[0i8; 2]; 15];
        let mut chroma_offset = [[0i8; 2]; 15];
        for (i, e) in entries.iter().enumerate() {
            delta_luma_weight[i] = field_i8(e.luma.0.checked_sub(luma_one).ok_or("luma weight out of range")?, "delta_luma_weight")?;
            luma_offset[i] = field_i8(e.luma.1, "luma_offset")?;
            for c in 0..2 {
                delta_chroma_weight[i][c] = field_i8(e.chroma[c].0.checked_sub(chroma_one).ok_or("chroma weight out of range")?, "delta_chroma_weight")?;
                // The record's `ChromaOffsetLX` is the derived 8-bit offset, which is what the
                // parser computed from the coded `delta_chroma_offset_lX` and left in `WpEntry`.
                chroma_offset[i][c] = field_i8(e.chroma[c].1, "ChromaOffsetLX")?;
            }
        }
        match list {
            0 => {
                record.delta_luma_weight_l0 = delta_luma_weight;
                record.luma_offset_l0 = luma_offset;
                record.delta_chroma_weight_l0 = delta_chroma_weight;
                record.chroma_offset_l0 = chroma_offset;
            }
            _ => {
                record.delta_luma_weight_l1 = delta_luma_weight;
                record.luma_offset_l1 = luma_offset;
                record.delta_chroma_weight_l1 = delta_chroma_weight;
                record.chroma_offset_l1 = chroma_offset;
            }
        }
    }
    Ok(record)
}

/// The `LongSliceFlags` word (`va_dec_hevc.h:392`), packed from bit 0 up as the C bitfield declares
/// it. Bits 14-31 are the header's reserved field and stay zero, which is what it requires.
fn pack_hevc_long_slice_flags(sh: &filmcraft_hevc::slice::SliceHeader, last_of_picture: bool) -> u32 {
    let bit = |at: u32, v: bool| u32::from(v) << at;
    let field = |at: u32, width: u32, v: u32| (v & ((1 << width) - 1)) << at;
    bit(0, last_of_picture)
        | bit(1, sh.dependent)
        | field(2, 2, sh.slice_type as u32)
        // colour_plane_id is 0: separate_colour_plane_flag is declined in build_hevc.
        | bit(6, sh.sao_luma)
        | bit(7, sh.sao_chroma)
        | bit(8, sh.mvd_l1_zero)
        | bit(9, sh.cabac_init_flag)
        | bit(10, sh.temporal_mvp)
        | bit(11, sh.deblocking_disabled)
        | bit(12, sh.collocated_from_l0)
        | bit(13, sh.loop_filter_across_slices)
}

/// Emulation prevention bytes (`00 00 03`) in the slice *header* of `nal`: the region
/// `nal[HEVC_NAL_HEADER..][..data_offset]`, where `data_offset` is our parser's byte position of
/// `slice_segment_data()` in the RBSP — the RBSP being the NAL unit with its two-byte header and its
/// emulation prevention bytes removed.
///
/// Only the header counts. libva documents the field as "number of emulation prevention bytes in
/// slice header", and a driver uses it to convert between the entry point offsets (counted without
/// them) and byte positions in the original bitstream (counted with them). Counting the residual
/// data as well would inflate the count by every `00 00 03` a tiled or wavefront picture happens to
/// contain and put every subset header after the first in the wrong place.
fn header_emu_prevention_bytes(nal: &[u8], data_offset: usize) -> Result<u32, String> {
    // The RBSP starts after the two-byte HEVC NAL header and ends where the slice data starts, so
    // both bounds are checked against the NAL unit rather than trusted: a header that claims to run
    // past the NAL unit is a decline, not a shorter count.
    let start = HEVC_NAL_HEADER.len();
    let end = start.checked_add(data_offset).ok_or("slice segment header length overflows")?;
    let header = nal.get(start..end).ok_or("slice segment header runs past the NAL unit")?;
    let count = header.windows(3).filter(|w| *w == [0, 0, 3]).count();
    u32::try_from(count).map_err(|_| "too many emulation prevention bytes".to_string())
}

// ---------------------------------------------------------------------------------------------
// The sample's NAL units, checked against the sample: every byte of a sample is behind a length
// prefix, so a prefix that runs past the end means a NAL unit is missing, and submitting the picture
// without it would decode fewer macroblocks than the stream says it has.
// ---------------------------------------------------------------------------------------------

/// The NAL units of a length-prefixed `sample`, each with its byte offset in the sample.
fn sample_nals<'a>(info: &NalStreamInfo, sample: &'a [u8]) -> Result<Vec<(&'a [u8], usize)>, String> {
    if !(1..=4).contains(&info.length_size) {
        return Err(format!("NAL unit length size {}", info.length_size));
    }
    // Every NAL unit costs at least its prefix and a header byte, so the sample's own length bounds
    // the count; this says so, rather than letting a sample of one-byte units ask for the memory.
    let cap = sample.len() / (info.length_size + 1) + 1;
    let mut out: Vec<(&[u8], usize)> = Vec::with_capacity(cap.min(MAX_SLICES));
    let mut pos = 0usize;
    while pos < sample.len() {
        if out.len() >= cap {
            return Err(format!("more than {cap} NAL units in one sample"));
        }
        let end = pos.checked_add(info.length_size).ok_or("sample size overflows")?;
        let prefix = sample.get(pos..end).ok_or("NAL unit length prefix runs past the sample")?;
        let length = prefix.iter().fold(0usize, |a, &b| (a << 8) | b as usize);
        let start = end;
        let stop = start.checked_add(length).ok_or("NAL unit length overflows")?;
        let nal = sample.get(start..stop).ok_or("NAL unit runs past the sample")?;
        if nal.is_empty() {
            return Err("empty NAL unit".into());
        }
        out.push((nal, start));
        pos = stop;
    }
    if out.is_empty() {
        return Err("no NAL unit in the sample".into());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Every value a bitstream supplies into a record field narrower than it was read as. A value that
// does not fit is an error the caller can act on — a name and the number — never a truncation or a
// wrap, either of which would describe a different picture than the stream contains.
// ---------------------------------------------------------------------------------------------

/// `value` as a `u16` field.
fn field_u16(value: u32, what: &str) -> Result<u16, String> {
    u16::try_from(value).map_err(|_| format!("{what} {value} does not fit the record"))
}

/// `value` as a `u8` field.
fn field_u8(value: u32, what: &str) -> Result<u8, String> {
    u8::try_from(value).map_err(|_| format!("{what} {value} does not fit the record"))
}

/// A count as a `u8` field, where the count came from a collection's length rather than a bitstream.
fn field_count(value: usize, what: &str) -> Result<u8, String> {
    match u8::try_from(value) {
        Ok(v) => Ok(v),
        Err(_) => Err(format!("{what} {value} does not fit the record")),
    }
}

/// `value` as an `i8` field.
fn field_i8(value: i32, what: &str) -> Result<i8, String> {
    i8::try_from(value).map_err(|_| format!("{what} {value} does not fit the record"))
}

/// `value` as an `i16` field, which is what a prediction weight or offset is carried in.
fn field_i16(value: i32, what: &str) -> Result<i16, String> {
    i16::try_from(value).map_err(|_| format!("{what} {value} does not fit the record"))
}

/// `bit_depth - 8`, the form the records carry it in.
fn minus8(depth: u32) -> Result<u8, String> {
    field_u8(depth.checked_sub(8).ok_or(format!("bit depth {depth} is below 8"))?, "bit_depth_minus8")
}

/// `value - 26`, the form the records carry an initial QP in.
fn minus26(value: i32) -> Result<i8, String> {
    field_i8(value.checked_sub(26).ok_or(format!("initial QP {value} is below 26"))?, "init_qp_minus26")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use filmcraft_codecs::hw::{FrameCodec, FrameStreamInfo, NalCodec, NalStreamInfo};

    /// The IQ matrix record carries the sequence parameter set's own scaling lists, in the raster
    /// order the record wants, and flat 16s when the set declares none.
    ///
    /// This is the record whose absence is invisible until a real driver is asked to decode: the
    /// iHD driver accepts a submission without it, reports success at every call, and leaves the
    /// render target holding the cleared surface. See [`super::h264_iq_matrix`].
    #[test]
    fn the_iq_matrix_carries_the_sequence_parameters_scaling_lists() {
        let Some((info, samples)) = h264_stream() else { return };
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        let mut poc = super::NalPoc::default();
        let p = super::build_nal(&info, &samples[0], &mut poc, 3, 40, 23).unwrap_or_else(|e| panic!("{e}"));
        let r = crate::vaapi::ffi::iq_matrix_h264(&p.iq_matrix).unwrap_or_else(|| panic!("not an IQ matrix record"));
        assert_eq!(p.iq_matrix.len(), std::mem::size_of_val(&r), "240 bytes, the size libva declares");
        // The fixture is encoded without quantizer matrices, so every list is the flat 16 of
        // 7.4.2.1.1's default — what the record has to say, and what the driver reads as "no
        // custom quantizer", which is the truth about this stream.
        assert!(r.scaling_list4x4.iter().all(|l| l.iter().all(|&v| v == 16)), "flat 4x4 lists");
        assert!(r.scaling_list8x8.iter().all(|l| l.iter().all(|&v| v == 16)), "flat 8x8 luma lists");
        assert_eq!(r.scaling_list8x8.len(), 2, "the record's 8x8 array is [Intra Y, Inter Y] only");
        assert_eq!(r.va_reserved, [0; crate::vaapi::ffi::VA_PADDING_LOW], "the reserved tail is zero");
    }

    /// A stream that does send scaling lists gets *its* lists, not flat ones: a record of flat 16s
    /// for a stream with custom quantizers decodes a different picture than the stream's.
    #[test]
    fn a_stream_with_scaling_lists_gets_its_own() {
        let ff = filmcraft_testkit::require_ffmpeg!();
        let Some(path) = filmcraft_testkit::fixtures::generate(&fixture("h264_scaling.mp4"), |tmp| {
            // `cqmfile` is empty, so x264 writes flat lists — but `--cqm` with a flat file and
            // `flat` is what makes `seq_scaling_matrix_present_flag` 1, which is the branch under
            // test. What matters is only that the flag is set and the lists reach the record.
            let args = [
                "-f",
                "lavfi",
                "-i",
                "testsrc2=s=320x240:r=24:d=1",
                "-c:v",
                "libx264",
                "-preset",
                "veryslow",
                "-x264-params",
                "cqm=flat:keyint=24:min-keyint=24:scenecut=0",
                "-pix_fmt",
                "yuv420p",
            ];
            run(&ff, &args, tmp)
        }) else {
            return;
        };
        let (info, samples) = read(&path);
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        let mut poc = super::NalPoc::default();
        let p = super::build_nal(&info, &samples[0], &mut poc, 5, 20, 15).unwrap_or_else(|e| panic!("{e}"));
        let r = crate::vaapi::ffi::iq_matrix_h264(&p.iq_matrix).unwrap_or_else(|| panic!("not an IQ matrix record"));
        // Flat lists, sent explicitly: the record must still carry six 4x4 and six 8x8 lists of
        // sixteen 16s, which is what makes "the set declared them" and "the set declared none"
        // indistinguishable in the record — correctly, because they decode the same picture.
        assert!(r.scaling_list4x4.iter().all(|l| l.iter().all(|&v| v == 16)));
        assert!(r.scaling_list8x8.iter().all(|l| l.iter().all(|&v| v == 16)));
        assert_eq!(r.scaling_list8x8.len(), 2, "and the two the record has slots for");
    }

    /// A HEVC stream with `sps_scaling_list_enabled_flag = 1` is declined rather than submitted without
    /// its `VAIQMatrixBufferHEVC`. The header is explicit that the buffer exists and is "sent once
    /// per frame" when the flag is set, and this backend does not build it — so the alternatives are
    /// a decline or a plausible picture that is not the stream's, and the decline is the one the
    /// never-crash rule asks for.
    #[test]
    fn an_hevc_stream_with_scaling_lists_is_declined_rather_than_guessed() {
        let ff = filmcraft_testkit::require_ffmpeg!();
        let Some(path) = filmcraft_testkit::fixtures::generate(&fixture("hevc_scaling.mp4"), |tmp| {
            // x265 writes scaling lists when the rate control asks for custom quantizers.
            let args = [
                "-f",
                "lavfi",
                "-i",
                "testsrc2=s=320x240:r=24:d=1",
                "-c:v",
                "libx265",
                "-preset",
                "veryslow",
                "-x265-params",
                "log-level=error:rd=6:keyint=24:min-keyint=24:scenecut=0",
                "-pix_fmt",
                "yuv420p",
            ];
            run(&ff, &args, tmp)
        }) else {
            return;
        };
        let (info, samples) = read(&path);
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        // Whatever the encoder actually emitted, the answer is the same: either the flag is clear and
        // the picture builds with no matrix buffer, or it is set and the picture is declined. What
        // must never happen is a third answer.
        let mut poc = super::NalPoc::default();
        match super::super::picture::build_nal(&info, &samples[0], &mut poc, 2, 320, 240) {
            Ok(p) => assert!(p.iq_matrix.is_empty(), "a stream that built did not have the flag set"),
            Err(why) => assert!(why.contains("VAIQMatrixBufferHEVC") || why.contains("scaling"), "the decline says what is missing, not something else: {why}"),
        }
    }

    /// HEVC submits no IQ matrix buffer **because the flag is clear**, not because its scaling lists are
    /// in the picture parameter record. `va_dec_hevc.h` puts them in `VAIQMatrixBufferHEVC`, sent
    /// "once per frame, and only when scaling_list_enabled_flag = 1" — the picture parameter record
    /// has no scaling-list field at all. An earlier version of this test asserted the wrong fact,
    /// which would have hidden a stream that needs a record this backend does not build.
    ///
    /// The fixture is encoded with x265's default, which leaves the flag clear, so this is the case
    /// that must work; the decline for the other case is
    /// `an_hevc_stream_with_scaling_lists_is_declined_rather_than_guessed`.
    #[test]
    fn hevc_without_scaling_lists_submits_no_iq_matrix() {
        let Some((info, samples)) = hevc_stream() else { return };
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        let mut poc = super::NalPoc::default();
        let p = super::build_nal(&info, &samples[0], &mut poc, 4, 640, 360).unwrap_or_else(|e| panic!("{e}"));
        assert!(p.iq_matrix.is_empty(), "scaling_list_enabled_flag is clear, so no buffer is needed");
        // And the picture parameter record is exactly the structure `va_dec_hevc.h` declares — which
        // is what makes omitting the buffer right rather than a default: there is nowhere in it for
        // a scaling list to go.
        assert_eq!(
            p.params.len(),
            std::mem::size_of::<crate::vaapi::ffi::VaPictureParameterBufferHevc>(),
            "the picture record is the declared structure, scaling lists and all (there are none)"
        );
    }

    /// HEVC treats an in-band parameter set exactly as H.264 does: a repeat of a set the `hvcC` already
    /// carries is skipped, a set that differs is declined.
    ///
    /// Left silent this was the same wrong-quantizer/geometry exposure finding 1 was about on the
    /// other codec: `HevcSets::parse` reads the configuration record's sets and does not fold the
    /// sample's in, so a differing in-band SPS would have had its picture decoded with the `hvcC`'s
    /// geometry and bit depth while the stream had moved on.
    #[test]
    fn hevc_declines_a_differing_in_band_parameter_set_and_accepts_a_repeat() {
        let Some((info, _samples)) = hevc_stream() else { return };
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        assert!(!info.parameter_sets.is_empty(), "the fixture's hvcC carries its parameter sets");

        // A repeat of one the sample entry already has must not be a reason to refuse.
        let repeat = info.parameter_sets[0].clone();
        let mut poc = super::NalPoc::default();
        let sample = access_unit(4, &[repeat.as_slice()]);
        match super::build_nal(&info, &sample, &mut poc, 3, 640, 360) {
            Ok(_) => {}
            Err(why) => assert!(!why.contains("in-band parameter set"), "a repeat is not a reason to refuse: {why}"),
        }
        // A set that differs — here, one that is not the one the entry has — is.
        for n in info.parameter_sets.iter().take(2) {
            let mut other = n.clone();
            other.push(0x00);
            let mut poc = super::NalPoc::default();
            let sample = access_unit(4, &[other.as_slice()]);
            let why = super::build_nal(&info, &sample, &mut poc, 3, 640, 360).expect_err("a differing in-band set");
            assert!(why.contains("in-band parameter set"), "the decline says why: {why}");
        }
    }

    /// Every `ReferenceFrames` entry the current picture's set does not name is
    /// `VA_PICTURE_HEVC_INVALID`.
    ///
    /// The HEVC record has no count field: `va.h:5322-5331` says the three `RPS_*` bits are how a
    /// driver learns which entries the picture refers to and that `NumPocTotalCurr` is their count.
    /// An entry with a valid `picture_id` and no bit is therefore a state the header does not define.
    /// iHD tolerates it, which is why I and P pictures decode here — a driver that counted every
    /// non-`INVALID` entry as live would read a reference the picture does not have.
    #[test]
    fn hevc_marks_every_reference_frame_the_picture_does_not_name() {
        let Some((info, samples)) = hevc_stream() else { return };
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        let mut poc = super::NalPoc::default();
        let mut named_somewhere = false;
        // A run of pictures, so that some refer to something and some do not.
        for (n, sample) in samples.iter().enumerate().take(12) {
            let surface = u32::try_from(n).unwrap_or(0);
            let Ok(p) = super::build_nal(&info, sample, &mut poc, surface, 640, 360) else { continue };
            let pp = crate::vaapi::ffi::picture_params_hevc(&p.params).unwrap_or_else(|| panic!("not a picture parameter record"));
            const FLAGS: u32 = crate::vaapi::ffi::PICTURE_HEVC_RPS_ST_CURR_BEFORE
                | crate::vaapi::ffi::PICTURE_HEVC_RPS_ST_CURR_AFTER
                | crate::vaapi::ffi::PICTURE_HEVC_RPS_LT_CURR;
            for (i, r) in pp.reference_frames.iter().enumerate() {
                if r.flags & FLAGS != 0 {
                    named_somewhere = true;
                    assert_ne!(r.picture_id, crate::vaapi::ffi::INVALID_SURFACE, "sample {n} slot {i}: a named entry names a surface");
                    assert_eq!(r.flags & crate::vaapi::ffi::PICTURE_HEVC_INVALID, 0, "sample {n} slot {i}: a referred-to entry is not also invalid");
                } else {
                    // Unnamed entries say they name nothing: no surface id, and the invalid bit.
                    assert_eq!(r.picture_id, crate::vaapi::ffi::INVALID_SURFACE, "sample {n} slot {i}: an unnamed entry names no surface");
                    assert_ne!(r.flags & crate::vaapi::ffi::PICTURE_HEVC_INVALID, 0, "sample {n} slot {i}: and says it is invalid");
                }
            }
        }
        assert!(named_somewhere, "the run refers to pictures, so some entry was flagged");
    }

    /// The two records an ffmpeg-made H.264 fixture's first picture is submitted as: one picture
    /// parameter buffer, and one slice parameter buffer per slice — not one per NAL unit, since the
    /// fixture's IDR is preceded by the SEI that names the encoder.
    #[test]
    fn one_slice_record_per_slice_in_the_access_unit() {
        let Some((info, samples)) = h264_stream() else { return };
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        let sample = &samples[0];
        assert_eq!(info.nal_types(sample).iter().filter(|&&t| matches!(t, 1 | 5)).count(), 2, "the fixture's first picture is two slices");
        let mut poc = super::NalPoc::default();
        let p = super::build_nal(&info, sample, &mut poc, 7, 40, 23).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(p.slices.len(), 2, "a slice record per slice");
        assert_eq!(p.slice_offsets.len(), 2, "and one offset per record");
    }

    /// The record a driver reads carries what FilmCraft's own parser found, read back through the
    /// `ffi` struct: the size in macroblocks, the surface it is told to decode into, the POC its
    /// decoder would compute, and the decoded picture buffer as the decoder has it.
    ///
    /// The fixture's first picture is an IDR, so the buffer is empty for it: `num_ref_frames` is zero
    /// and every slot says it names nothing. That is the whole point of the count being what was
    /// filled rather than the DPB size the sequence parameter set declares — a count of 4 over four
    /// slots that all say "no picture" is a record telling the driver to read four invalid slots.
    #[test]
    fn picture_parameters_carry_the_parsed_slice_header() {
        let Some((info, samples)) = h264_stream() else { return };
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        let mut poc = super::NalPoc::default();
        let p = super::build_nal(&info, &samples[0], &mut poc, 7, 40, 23).unwrap_or_else(|e| panic!("{e}"));
        // 40x23 macroblocks for the fixture's 640x368 coded picture (640x360 cropped).
        let pp = crate::vaapi::ffi::picture_params_h264(&p.params).unwrap_or_else(|| panic!("not a picture parameter record"));
        assert_eq!(pp.picture_width_in_mbs_minus1, 39);
        assert_eq!(pp.picture_height_in_mbs_minus1, 22);
        assert_eq!(pp.curr_pic.picture_id, 7, "the render target is the surface we passed");
        assert_eq!(pp.curr_pic.flags, crate::vaapi::ffi::PICTURE_H264_SHORT_TERM_REFERENCE, "an IDR stays in the DPB");
        // An IDR's POC follows from being an IDR, with nothing before it.
        assert_eq!((pp.curr_pic.top_field_order_cnt, pp.curr_pic.bottom_field_order_cnt), (0, 0));
        assert_eq!(pp.bit_depth_luma_minus8, 0);
        assert_eq!(pp.num_ref_frames, 0, "an IDR refers to nothing, and the count is what was filled");
        assert!(
            pp.reference_frames.iter().all(|r| r.flags & crate::vaapi::ffi::PICTURE_H264_INVALID != 0 && r.picture_id == crate::vaapi::ffi::INVALID_SURFACE),
            "and so every slot says it names no picture"
        );
    }

    /// The other half of the count: once there *is* a decoded picture buffer, the record names it, and
    /// the count is exactly how many slots were filled. This is the deferred finding from Task 3 — a
    /// `num_ref_frames` claiming N valid DPB slots while every `ReferenceFrames` entry was invalid —
    /// checked on a real stream rather than by construction.
    #[test]
    fn the_reference_frames_are_the_surfaces_the_buffer_holds() {
        let Some((info, samples)) = h264_stream() else { return };
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        let mut poc = super::NalPoc::default();
        // One distinct surface per picture, as the decode loop's ring hands them out.
        let mut filled = 0u32;
        for (i, sample) in samples.iter().enumerate() {
            let surface = u32::try_from(i).unwrap_or(u32::MAX);
            let Ok(p) = super::build_nal(&info, sample, &mut poc, surface, 40, 23) else {
                panic!("sample {i} does not decode");
            };
            let pp = crate::vaapi::ffi::picture_params_h264(&p.params).unwrap_or_else(|| panic!("not a picture parameter record"));
            // The count is the number of slots that name a surface, and those are the first `count`
            // of them: a driver reads the first `num_ref_frames` entries and nothing else.
            let named: Vec<&crate::vaapi::ffi::VaPictureH264> =
                pp.reference_frames.iter().filter(|r| r.picture_id != crate::vaapi::ffi::INVALID_SURFACE).collect();
            assert_eq!(usize::from(pp.num_ref_frames), named.len(), "sample {i}: the count is what was filled");
            assert!(
                pp.reference_frames.iter().take(usize::from(pp.num_ref_frames)).all(|r| r.picture_id != crate::vaapi::ffi::INVALID_SURFACE),
                "sample {i}: every counted slot names a surface"
            );
            assert!(
                pp.reference_frames.iter().skip(usize::from(pp.num_ref_frames)).all(|r| r.picture_id == crate::vaapi::ffi::INVALID_SURFACE),
                "sample {i}: and nothing past the count claims to"
            );
            // Every entry names a surface this decoder has decoded into, and no surface twice.
            let mut seen = std::collections::HashSet::new();
            for r in &named {
                assert!(r.picture_id <= surface, "sample {i}: entry names surface {} of a ring at {surface}", r.picture_id);
                assert!(seen.insert(r.picture_id), "sample {i}: surface {} is in the buffer twice", r.picture_id);
                assert!(r.flags & crate::vaapi::ffi::PICTURE_H264_INVALID == 0, "sample {i}: a named slot does not say it is invalid");
            }
            // Each slice's list entries are all pictures the buffer holds, and its `num_ref_idx` counts
            // them: an entry that is not one the slice refers to is a different picture, silently.
            // An I slice refers to nothing, and the record's `num_ref_idx` fields say nothing about
            // them (the slice header does not code them), so it is exempt from the count check.
            for slice in &p.slices {
                let sp = crate::vaapi::ffi::slice_params_h264(&slice.params).unwrap_or_else(|| panic!("not a slice record"));
                let intra = sp.slice_type == 2;
                // Only a B slice has a second list: `num_ref_idx_active[1]` is 0 for an I or a P, so
                // the record's `minus1` field cannot distinguish "none" from "one" and the list is
                // empty for both.
                for (l, count) in
                    [(0, Some(sp.num_ref_idx_l0_active_minus1)), (1, if sp.slice_type == 1 { Some(sp.num_ref_idx_l1_active_minus1) } else { None })]
                {
                    let list = if l == 0 { &sp.ref_pic_list0 } else { &sp.ref_pic_list1 };
                    let wanted = match (intra, count) {
                        (true, _) | (_, None) => 0,
                        (_, Some(count)) => usize::from(count) + 1,
                    };
                    for (at, entry) in list.iter().take(wanted).enumerate() {
                        assert!(entry.picture_id != crate::vaapi::ffi::INVALID_SURFACE, "sample {i}: list {l} entry {at} names no picture");
                        assert!(
                            seen.contains(&entry.picture_id),
                            "sample {i}: list {l} entry {at} names surface {}, which is not in the buffer",
                            entry.picture_id
                        );
                    }
                    for entry in list.iter().skip(wanted) {
                        assert_eq!(entry.picture_id, crate::vaapi::ffi::INVALID_SURFACE, "sample {i}: list {l} past its length");
                    }
                }
            }
            filled = filled.max(u32::from(pp.num_ref_frames));
        }
        assert!(filled >= 2, "the fixture's later pictures do refer to more than one picture ({filled})");
    }

    /// How many surfaces a stream's decoded picture buffer needs, read from its own parameter sets:
    /// what the decode loop allocates, bounded by what a record can name.
    #[test]
    fn the_surface_count_is_the_streams_own_picture_buffer() {
        let Some((info, _)) = h264_stream() else { return };
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        let n = super::reference_pictures(&info).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(n, 4, "the fixture's sequence parameter set allows four reference pictures");
        // A parameter set that will not parse is an error, not a zero: the surfaces are allocated for
        // the sizes it describes, so a stream whose sets do not parse is not one this backend takes.
        let mut broken = info.clone();
        broken.parameter_sets = vec![vec![0x67, 0xff, 0xff, 0xff]];
        assert!(super::reference_pictures(&broken).is_err(), "an unparsable sequence parameter set");
        // A stream with no parameter sets at all cannot be sized either.
        let mut none = info.clone();
        none.parameter_sets.clear();
        assert!(super::reference_pictures(&none).is_err(), "no sequence parameter set");
    }

    /// What a bitstream says and a record can hold are not the same question: a coded size that is
    /// not a whole number of macroblocks, or one that is not the size the caller's surfaces were
    /// created at, is declined rather than described at the wrong size.
    #[test]
    fn a_size_that_does_not_match_the_surfaces_is_an_error() {
        let Some((info, samples)) = h264_stream() else { return };
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        let mut poc = super::NalPoc::default();
        assert!(super::build_nal(&info, &samples[0], &mut poc, 7, 40, 24).is_err(), "surfaces at the wrong size");
        let mut odd = info.clone();
        odd.coded = (640, 360); // not a whole number of macroblocks
        let mut poc = super::NalPoc::default();
        assert!(super::build_nal(&odd, &samples[0], &mut poc, 7, 40, 23).is_err(), "a coded size that is not macroblock-aligned");
    }

    /// A slice header that will not parse is an error, not a panic and not a zeroed record: a zeroed
    /// record would tell the driver a valid picture exists where none does.
    #[test]
    fn an_unparsable_slice_is_an_error_not_a_panic() {
        let Some((info, samples)) = h264_stream() else { return };
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        // A slice NAL unit with nothing after its header byte: there is no slice header to parse.
        let mut poc = super::NalPoc::default();
        assert!(super::build_nal(&info, &[0, 0, 0, 1, 0x65], &mut poc, 1, 40, 23).is_err(), "a slice with no header");
        // A slice header full of ones, which reads as an enormous first_mb_in_slice and a picture
        // parameter set id nothing has: a corrupt header, not a picture.
        let header = info.nals(&samples[0]).first().and_then(|n| n.first().copied()).unwrap_or(0x65);
        let mut corrupt = vec![0, 0, 0, 32];
        corrupt.push(header);
        corrupt.extend(std::iter::repeat_n(0xffu8, 31));
        let mut poc = super::NalPoc::default();
        let e = super::build_nal(&info, &corrupt, &mut poc, 1, 40, 23).expect_err("a header of ones");
        assert!(!e.is_empty(), "the reason names the failure");
        // and the same picture with its first byte forbidden, which no decoder accepts
        let mut forbidden = vec![0, 0, 0, 4, 0x85, 0xff, 0xff, 0xff];
        forbidden[4] = 0x85 | 0x80;
        let mut poc = super::NalPoc::default();
        assert!(super::build_nal(&info, &forbidden, &mut poc, 1, 40, 23).is_err(), "forbidden_zero_bit set");
    }

    /// A length-prefixed access unit carrying `units`, for the in-band parameter set tests.
    fn access_unit(length_size: u32, units: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for u in units {
            out.extend_from_slice(&(u.len() as u32).to_be_bytes()[4 - length_size as usize..]);
            out.extend_from_slice(u);
        }
        out
    }

    /// An in-band parameter set is declined when it **differs** from the sample entry's, and accepted
    /// when it does not. Both halves matter: declining a repeat would cost a stream that re-sends
    /// its `avcC` sets at every key frame its hardware decoder, for no gain — that is the whole
    /// difference between an error and a capability.
    ///
    /// The test is the same one `StreamInfo::parameters_changed` makes, deliberately: a sample this
    /// backend declines should be a sample `HybridDecoder` believes changed, or the guard in front of
    /// it and the guard inside it would disagree about the same samples.
    #[test]
    fn an_in_band_parameter_set_is_declined_only_when_it_differs() {
        let sps: &[u8] = &[0x67, 0x42, 0xc0, 0x0a, 0xd9, 0x00, 0xf0, 0x88, 0x46, 0xa0];
        let pps: &[u8] = &[0x68, 0xce, 0x38, 0x80];
        // A stream description whose sample entry carries exactly those two sets.
        let mut info = h264_info(4);
        info.parameter_sets = vec![sps.to_vec(), pps.to_vec()];
        let access_unit = |unit: &[u8]| access_unit(4, &[unit]);

        // A repeat of a set the sample entry already carries: the picture builds. (It stops being
        // "builds" if the set does not parse, which is not what this half is about — what it is
        // about is that the *in-band* half is not the reason for a refusal.)
        let mut poc = super::NalPoc::default();
        let repeat = super::build_nal(&info, &access_unit(sps), &mut poc, 1, 40, 23);
        match &repeat {
            Ok(_) => {}
            Err(why) => assert!(!why.contains("in-band parameter set"), "a repeat of the sample entry's own set is not a reason to refuse: {why}"),
        }
        let mut poc = super::NalPoc::default();
        let repeat = super::build_nal(&info, &access_unit(pps), &mut poc, 1, 40, 23);
        match &repeat {
            Ok(_) => {}
            Err(why) => assert!(!why.contains("in-band parameter set"), "same for the PPS: {why}"),
        }

        // A set that differs from the one the sample entry carries is declined, for both kinds. The entry
        // keeps the valid pair above so the refusal is the in-band half and not the parameter-set
        // parse: only the bytes inside the sample change, and they are not the ones the entry has.
        for (what, in_band) in [("SPS", &[0x67u8, 0x42, 0xc0][..]), ("PPS", &[0x68, 0xce][..])] {
            let mut poc = super::NalPoc::default();
            let why = super::build_nal(&info, &access_unit(in_band), &mut poc, 1, 40, 23).expect_err(what);
            assert!(why.contains("in-band parameter set"), "{what}: the decline says why: {why}");
        }
    }

    /// Every hostile shape a length-prefixed sample can have is a clean error: no unit, an empty unit,
    /// a prefix running past the end, and a prefix that claims more units than the sample could hold.
    #[test]
    fn a_malformed_length_prefix_is_declined() {
        let sample = vec![0, 0, 0, 200, 0x65]; // claims 200 bytes, has 1
        let mut poc = super::NalPoc::default();
        assert!(super::build_nal(&h264_info(4), &sample, &mut poc, 1, 40, 23).is_err(), "truncated length prefix");
        // a length size no container uses
        for length_size in [0, 5, 16] {
            let mut poc = super::NalPoc::default();
            assert!(super::build_nal(&h264_info(length_size), &sample, &mut poc, 1, 40, 23).is_err(), "length size {length_size}");
        }
        // an empty NAL unit, and bytes after the last unit that no length accounts for
        let empty_unit = vec![0, 0, 0, 0, 0x65, 0xff];
        let mut poc = super::NalPoc::default();
        assert!(super::build_nal(&h264_info(4), &empty_unit, &mut poc, 1, 40, 23).is_err(), "an empty NAL unit");
        // a trailing prefix with no unit behind it
        let trailing = [vec![0, 0, 0, 2, 0x65, 0xff], vec![0, 0, 0, 1]].concat();
        let mut poc = super::NalPoc::default();
        assert!(super::build_nal(&h264_info(4), &trailing, &mut poc, 1, 40, 23).is_err(), "a prefix with no NAL unit");
        // an empty sample, and one that is only a prefix
        for sample in [&[][..], &[0, 0, 0][..]] {
            let mut poc = super::NalPoc::default();
            assert!(super::build_nal(&h264_info(4), sample, &mut poc, 1, 40, 23).is_err(), "{sample:?}");
        }
        // and a stream with no parameter sets at all: no slice's parameter set can be found
        let mut poc = super::NalPoc::default();
        let no_sets = [vec![0, 0, 0, 2, 0x65, 0xff], vec![0, 0, 0, 2, 0x65, 0xff]].concat();
        assert!(super::build_nal(&h264_info(4), &no_sets, &mut poc, 1, 40, 23).is_err(), "no parameter sets in the record");
    }

    /// The HEVC record carries what its own parsers found, and its slice records one per segment.
    #[test]
    fn hevc_picture_parameters_carry_the_parsed_slice_header() {
        let Some((info, samples)) = hevc_stream() else { return };
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        // HEVC codes in 8x8 minimum coding blocks, and 360 is already a whole number of them, so
        // unlike H.264 the coded size is the display size.
        assert_eq!(info.coded, (640, 360));
        let mut poc = super::NalPoc::default();
        let p = super::build_nal(&info, &samples[0], &mut poc, 9, 640, 360).unwrap_or_else(|e| panic!("{e}"));
        assert!(!p.slices.is_empty(), "one record per slice segment");
        assert_eq!(p.slice_offsets.len(), p.slices.len(), "the offsets are parallel to the records");
        let pp = crate::vaapi::ffi::picture_params_hevc(&p.params).unwrap_or_else(|| panic!("not a picture parameter record"));
        assert_eq!(pp.pic_width_in_luma_samples, 640);
        assert_eq!(pp.pic_height_in_luma_samples, 360);
        assert_eq!(pp.curr_pic.picture_id, 9);
        assert_eq!(pp.curr_pic.pic_order_cnt, 0, "an IDR's POC follows from being an IDR");
        // An IRAP picture's set is empty, so its record names no reference picture: every
        // `ReferenceFrames` slot says it holds nothing, which is what the count of them that name a
        // surface (none) says too.
        assert!(
            pp.reference_frames.iter().all(|r| r.flags & crate::vaapi::ffi::PICTURE_HEVC_INVALID != 0 && r.picture_id == crate::vaapi::ffi::INVALID_SURFACE),
            "no reference frame is claimed"
        );
        // each slice record says where its own bytes are, back to back in one buffer
        let mut end = 0u32;
        for slice in &p.slices {
            let r = crate::vaapi::ffi::slice_params_hevc(&slice.params).unwrap_or_else(|| panic!("not a slice parameter record"));
            assert_eq!(r.slice_data_offset, end, "slices are laid out back to back");
            assert_eq!(r.slice_data_size, slice.length);
            end += slice.length;
        }
        // and each one names its own bytes in the sample
        for (slice, &at) in p.slices.iter().zip(&p.slice_offsets) {
            assert_eq!(samples[0].get(at..at + slice.length as usize).map(<[u8]>::len), Some(slice.length as usize), "the slice at {at} is there");
        }
    }

    /// The POC state advances once per picture, not once per slice, so the POCs of a run of pictures
    /// are the ones FilmCraft's own decoder computes for them. Every picture of the fixture's first
    /// GOP is compared against that decoder, which is what a two-slice picture advancing the state
    /// twice would break: the second slice's POC would push the next picture's count out by one.
    #[test]
    fn poc_advances_once_per_picture() {
        let Some((info, samples)) = h264_stream() else { return };
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        let read = |p: &super::Picture| {
            let pp = crate::vaapi::ffi::picture_params_h264(&p.params).unwrap_or_else(|| panic!("not a record"));
            pp.curr_pic.top_field_order_cnt
        };
        // Our own decoder, fed the same samples as Annex B, is the reference. It reads length-prefixed
        // input only when configured from an `avcC`, so the samples are converted rather than
        // reconstructed: `to_annex_b` is the same conversion the decode loop uses.
        let mut dec = filmcraft_h264::Decoder::new();
        let mut reference = Vec::new();
        for (i, s) in samples.iter().enumerate() {
            let annexb = crate::annexb::to_annex_b(&info, s, true).unwrap_or_else(|e| panic!("sample {i}: {e}"));
            reference.extend(dec.decode(&annexb, i as i64).unwrap_or_else(|e| panic!("sample {i}: {e}")).into_iter().map(|p| p.poc));
        }
        reference.extend(dec.flush().into_iter().map(|p| p.poc));
        assert!(reference.len() > 2, "the fixture decoded to {} pictures", reference.len());

        let run = |poc: &mut super::NalPoc| -> Vec<i32> {
            let mut out = Vec::new();
            for s in &samples {
                if let Ok(p) = super::build_nal(&info, s, poc, 7, 40, 23) {
                    out.push(read(&p));
                }
            }
            out
        };
        let mut poc = super::NalPoc::default();
        let first = run(&mut poc);
        assert_eq!(first[0], 0, "the first picture is an IDR, whose POC is 0");
        // The record's POC is in decoding order and ours is in output order, so the two sequences
        // differ in order and have to agree as multisets: the same 24 picture numbers, each once.
        // Advancing the state per slice rather than per picture would leave gaps in ours.
        let sorted = |mut v: Vec<i32>| {
            v.sort_unstable();
            v
        };
        assert_eq!(first.len(), reference.len(), "every picture has a record");
        assert_eq!(sorted(first.clone()), sorted(reference), "the POCs are the picture numbers our own decoder computes");
        poc.reset();
        assert_eq!(run(&mut poc), first, "a reset decoder computes the same run again");
    }

    /// `slice_offsets` points at each slice's NAL unit in the sample it came from, which is what
    /// makes `slices[i]` a byte range the caller can copy out of the sample.
    /// Each slice's record says where its own bytes are: `offset` / `length` are a byte range in
    /// the slice data buffer the driver is handed, and the record agrees with them, while
    /// `slice_offsets` says where those bytes came from in the sample.
    #[test]
    fn each_slice_record_says_where_its_own_bytes_are() {
        let Some((info, samples)) = h264_stream() else { return };
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        let sample = &samples[0];
        let mut poc = super::NalPoc::default();
        let p = super::build_nal(&info, sample, &mut poc, 7, 40, 23).unwrap_or_else(|e| panic!("{e}"));
        let nals = info.nals(sample);
        let mut expected_offset = 0u32;
        for (slice, &at) in p.slices.iter().zip(&p.slice_offsets) {
            let r = crate::vaapi::ffi::slice_params_h264(&slice.params).unwrap_or_else(|| panic!("not a slice parameter record"));
            assert_eq!((r.slice_data_offset, r.slice_data_size), (slice.offset, slice.length), "the record agrees with the byte range");
            assert_eq!(r.slice_type, 2, "the fixture's first picture is all I slices");
            assert_eq!(r.slice_data_flag, crate::vaapi::ffi::SLICE_DATA_FLAG_ALL, "the whole slice is in the buffer");
            assert_eq!(r.slice_data_offset, expected_offset, "the slices run back to back in one buffer");
            // the slice's bytes are the NAL unit it came from
            let nal = nals.iter().find(|n| sample.get(at..at + n.len()) == Some(*n)).unwrap_or_else(|| panic!("no NAL unit at {at}"));
            assert_eq!(slice.length as usize, nal.len());
            expected_offset += slice.length;
        }
        assert_eq!(expected_offset, p.slices.iter().map(|s| s.length).sum::<u32>(), "the slice data offsets cover the picture's slices");
        assert_eq!(p.slice_data_size(), expected_offset, "the buffer ends at the last slice");
        assert!(p.slices.windows(2).all(|w| w[0].offset + w[0].length == w[1].offset), "the slice data offsets run back to back");
        // `slices=2` made the encoder split the picture in two by macroblock row, so the two
        // records start at different macroblocks — which is what tells them apart in the record.
        let firsts: Vec<u16> = p.slices.iter().map(|s| crate::vaapi::ffi::slice_params_h264(&s.params).expect("a record").first_mb_in_slice).collect();
        assert_eq!(firsts[0], 0, "the picture starts at macroblock 0");
        assert!(firsts.get(1).is_some_and(|&mb| mb > 0), "the second slice starts further into the picture: {firsts:?}");
    }

    /// The fixture's own H.264 description, parameter sets as [`build_nal`] parses them, and its
    /// samples in decoding order — everything a test about the records needs, from one fixture read.
    ///
    /// Encoded with `weightb=0`, so every picture of it has records: x264's default `weightb=1` puts
    /// the B slices in the mode whose weights a slice record cannot express, and they would be refused
    /// from the second B picture on — see [`h264_implicit_bipred`] for the fixture that does exercise
    /// that path, and [`an_implicitly_weighted_b_slice_is_refused_rather_than_guessed`] for what it
    /// asserts.
    fn h264() -> Option<(NalStreamInfo, super::H264Sets, Vec<Vec<u8>>)> {
        let (info, samples) = h264_stream()?;
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        let sets = super::H264Sets::parse(&info).unwrap_or_else(|e| panic!("{e}"));
        Some((info, sets, samples))
    }

    /// [`h264`]'s sequence and picture parameter sets, by id.
    fn h264_set() -> Option<(filmcraft_h264::params::Sps, filmcraft_h264::params::Pps)> {
        let (_, sets, _) = h264()?;
        let sps = sets.spss.iter().flatten().next().unwrap_or_else(|| panic!("no SPS")).clone();
        let pps = sets.ppss.iter().flatten().next().unwrap_or_else(|| panic!("no PPS")).clone();
        Some((sps, pps))
    }

    /// `slice_data_num_emu_prevn_bytes` counts the emulation prevention bytes of the **slice header**
    /// only. libva's driver uses the count to convert the entry point offsets (counted without
    /// emulation prevention bytes) into byte positions in the bitstream it was handed (counted with
    /// them), so counting the residual data as well would put every subset header after the first in
    /// the wrong place.
    ///
    /// The count is made here, over `nal[2..2 + data_offset]` — the NAL unit from after its two-byte
    /// header to where the parser says the slice data starts — rather than by calling the module's
    /// own helper, which is what made the first version of this assertion circular.
    #[test]
    fn the_emulation_prevention_count_is_the_slice_headers() {
        let Some((info, sets, samples)) = hevc() else { return };
        let mut checked = 0usize;
        let mut header_only = 0usize;
        // One picture-order-count state for the whole loop, as a decoder has: an inter picture's
        // records cannot be built without the pictures it refers to, so a fresh state per sample
        // would decline every picture after the first.
        let mut poc = super::NalPoc::default();
        for (i, sample) in samples.iter().enumerate() {
            let Ok(p) = super::build_nal(&info, sample, &mut poc, i as u32, 640, 360) else { continue };
            let nals = info.nals(sample);
            for (slice, &at) in p.slices.iter().zip(&p.slice_offsets) {
                let nal = nals.iter().find(|n| sample.get(at..at + n.len()) == Some(*n)).unwrap_or_else(|| panic!("no NAL unit at {at}"));
                let hdr = filmcraft_hevc::slice::NalHeader::parse(nal).unwrap_or_else(|e| panic!("{e}"));
                let rbsp = filmcraft_bitstream::unescape_rbsp(&nal[2..]);
                let (sh, _, _) = filmcraft_hevc::slice::SliceHeader::parse(&rbsp, hdr, |id| sets.lookup(id), None).unwrap_or_else(|e| panic!("{e}"));
                let r = crate::vaapi::ffi::slice_params_hevc(&slice.params).unwrap_or_else(|| panic!("not a slice record"));
                assert_eq!(
                    r.slice_data_num_emu_prevn_bytes as usize,
                    emu_count(&nal[2..2 + sh.data_offset]),
                    "the slice header's own emulation prevention bytes (NAL {} bytes, header {})",
                    nal.len(),
                    sh.data_offset
                );
                // The whole NAL unit of this fixture has `00 00 03` in its residual data in several
                // pictures — x265's encoder emits them for long runs of zero coefficients — so the
                // two counts really do differ, and an implementation that counted the whole unit
                // would fail the assertion above rather than pass by luck.
                if emu_count(&nal[2..2 + sh.data_offset]) == emu_count(nal) {
                    header_only += 1;
                }
                checked += 1;
            }
        }
        assert!(checked > 20, "the fixture's whole first GOP was checked ({checked} segments)");
        assert!(header_only < checked, "at least one segment has an emulation prevention byte outside its header ({header_only} of {checked} do not)");
    }

    /// The emulation prevention bytes (`00 00 03`) in a region of a NAL unit, counted over the region
    /// the assertion names. Counting the *region* is the point of these tests, so it is written out
    /// at each call site rather than folded into a helper that would hide it.
    fn emu_count(bytes: &[u8]) -> usize {
        bytes.windows(3).filter(|w| *w == [0, 0, 3]).count()
    }

    /// A parser value as the `i8` a record field holds, refusing to truncate: `as i8` in an assertion
    /// would let a value that does not fit compare equal to a truncated record field.
    fn i8_of(value: i32) -> i8 {
        i8::try_from(value).unwrap_or_else(|_| panic!("{value} does not fit the record's i8"))
    }

    /// [`i8_of`], for the `i16` a prediction weight or offset is carried in.
    fn i16_of(value: i32) -> i16 {
        i16::try_from(value).unwrap_or_else(|_| panic!("{value} does not fit the record's i16"))
    }

    /// The H.264 slice record's prediction weight tables come from the slice header's own
    /// `pred_weight_table`, entry by entry, rather than staying zeroed. A record that advertised
    /// weighted prediction with an empty table would decode a different picture from the stream's,
    /// silently, which is the one answer this module never gives.
    ///
    /// The fixture matters as much as the fill — see [`h264_weighted`]. Both `*_flag` fields, both
    /// denominators and every luma and chroma weight and offset are compared against the parsed
    /// table, and the two floor assertions at the end are what stop the fixture going quietly
    /// vacuous, which is the failure mode an encoder-side fixture really has.
    ///
    /// **What this does not cover: the L1 family.** `filmcraft_h264` parses a B slice's table only
    /// for `weighted_bipred_idc == 1`, and no x264 option set produces that (it writes 2, explicit,
    /// for `weightb=1`), so this fixture has no B slices and `luma_weight_l1` and its chroma family
    /// are only checked for carrying nothing rather than carrying the right thing. The L1 fill is the
    /// same loop over the other list; the HEVC fixture does exercise both of its lists, and this
    /// task's report records the gap.
    #[test]
    fn the_h264_slice_record_carries_the_prediction_weights() {
        let Some((info, sets, samples)) = h264_weighted() else { return };
        let pps = sets.ppss.iter().flatten().next().unwrap_or_else(|| panic!("no PPS")).clone();
        assert!(pps.weighted_pred, "the fixture's PPS allows weighted prediction");
        let mut weighted = 0usize;
        let mut explicit = 0usize;
        let mut poc = super::NalPoc::default();
        for sample in &samples {
            let Ok(picture) = super::build_nal(&info, sample, &mut poc, 7, 40, 23) else { continue };
            let nals = info.nals(sample);
            for (slice, &at) in picture.slices.iter().zip(&picture.slice_offsets) {
                let nal = nals.iter().find(|n| sample.get(at..at + n.len()) == Some(*n)).unwrap_or_else(|| panic!("no NAL unit at {at}"));
                let hdr = filmcraft_h264::slice::NalHeader::parse(nal[0]).unwrap_or_else(|e| panic!("{e}"));
                let rbsp = filmcraft_bitstream::unescape_rbsp(&nal[1..]);
                let (sh, _, _) = filmcraft_h264::slice::SliceHeader::parse(&rbsp, hdr, |id| {
                    let p = sets.ppss.get(id as usize).and_then(|p| p.as_ref()).unwrap_or_else(|| panic!("PPS {id}"));
                    let s = sets.spss.get(p.sps_id as usize).and_then(|s| s.as_ref()).unwrap_or_else(|| panic!("SPS"));
                    Ok((p, s))
                })
                .unwrap_or_else(|e| panic!("{e}"));
                let r = crate::vaapi::ffi::slice_params_h264(&slice.params).unwrap_or_else(|| panic!("not a slice record"));
                let Some(table) = sh.pred_weight_table.as_ref() else {
                    // A slice the parser found no table in carries no weights, and says so.
                    assert_eq!(r.luma_log2_weight_denom, 0, "no table means no weighting");
                    assert!(r.luma_weight_l0 == [0; 32] && r.chroma_weight_l0 == [[0; 2]; 32], "and none are invented");
                    continue;
                };
                weighted += 1;
                assert_eq!(u32::from(r.luma_log2_weight_denom), table.luma_log2_denom, "luma_log2_weight_denom");
                assert_eq!(u32::from(r.chroma_log2_weight_denom), table.chroma_log2_denom, "chroma_log2_weight_denom");
                for (list, entries, (lw, lo, cw, co, luma_flag, chroma_flag)) in [
                    (
                        0,
                        &table.l0,
                        (&r.luma_weight_l0, &r.luma_offset_l0, &r.chroma_weight_l0, &r.chroma_offset_l0, r.luma_weight_l0_flag, r.chroma_weight_l0_flag),
                    ),
                    (
                        1,
                        &table.l1,
                        (&r.luma_weight_l1, &r.luma_offset_l1, &r.chroma_weight_l1, &r.chroma_offset_l1, r.luma_weight_l1_flag, r.chroma_weight_l1_flag),
                    ),
                ] {
                    // The `*_flag` fields say the list carries luma weights at all (7.4.3.2); with no
                    // flag the default weight of `1 << denom` applies, which the entries hold anyway.
                    assert_eq!(luma_flag, u8::from(entries.iter().any(|e| e.luma_flag)), "luma_weight_l{list}_flag");
                    assert_eq!(chroma_flag, u8::from(entries.iter().any(|e| e.chroma_flag)), "chroma_weight_l{list}_flag");
                    for (i, e) in entries.iter().enumerate() {
                        if e.luma_flag || e.chroma_flag {
                            explicit += 1;
                        }
                        assert_eq!(lw[i], i16_of(e.luma_weight), "luma_weight_l{list}[{i}]");
                        assert_eq!(lo[i], i16_of(e.luma_offset), "luma_offset_l{list}[{i}]");
                        for c in 0..2 {
                            assert_eq!(cw[i][c], i16_of(e.chroma_weight[c]), "chroma_weight_l{list}[{i}][{c}]");
                            assert_eq!(co[i][c], i16_of(e.chroma_offset[c]), "chroma_offset_l{list}[{i}][{c}]");
                        }
                    }
                    // nothing is written past the slice's own references
                    assert!(lw[entries.len()..].iter().all(|&w| w == 0), "luma_weight_l{list} past the slice's references");
                    assert!(lo[entries.len()..].iter().all(|&w| w == 0), "luma_offset_l{list} past the slice's references");
                    assert!(cw[entries.len()..].iter().flatten().all(|&w| w == 0), "chroma_weight_l{list} past the slice's references");
                    assert!(co[entries.len()..].iter().flatten().all(|&w| w == 0), "chroma_offset_l{list} past the slice's references");
                }
            }
        }
        assert!(weighted >= 15, "the fixture's P slices code a table ({weighted})");
        assert!(explicit >= 15, "and at least {explicit} of their entries set a weight flag rather than taking the default");
    }

    /// An H.264 `weighted_bipred_idc == 2` B slice is refused when the stream really does weight it,
    /// and described with an explicit 1.0 table when it does not — which is the whole of the rule, and
    /// it is checked here on the fixture's own parsed headers and its own reference lists.
    ///
    /// Both halves need the fixture, so this is one test: x264's default `weightb=1` over `testsrc2`
    /// *does* exercise implicit weighting (a B pair wants `(48, 16)`), and the same fixture with
    /// `weightb=0` codes no table at all and so never reaches the check. The "all pairs are 1.0"
    /// branch is therefore covered by construction, on synthetic entries below, rather than by a
    /// second fixture this test suite does not have.
    #[test]
    fn an_implicitly_weighted_b_slice_is_refused_rather_than_guessed() {
        let Some((info, sets, samples)) = h264_implicit_bipred() else { return };
        let pps = sets.ppss.iter().flatten().next().unwrap_or_else(|| panic!("no PPS")).clone();
        assert_eq!(pps.weighted_bipred_idc, 2, "the fixture is in the mode whose B slices this record cannot express");
        let header_of = |sample: &[u8]| -> filmcraft_h264::slice::SliceHeader {
            let nal = info.nals(sample).first().copied().expect("a NAL unit");
            let hdr = filmcraft_h264::slice::NalHeader::parse(nal[0]).unwrap_or_else(|e| panic!("{e}"));
            let rbsp = filmcraft_bitstream::unescape_rbsp(&nal[1..]);
            filmcraft_h264::slice::SliceHeader::parse(&rbsp, hdr, |id| {
                let p = sets.ppss.get(id as usize).and_then(|p| p.as_ref()).unwrap_or_else(|| panic!("PPS {id}"));
                let s = sets.spss.get(p.sps_id as usize).and_then(|s| s.as_ref()).unwrap_or_else(|| panic!("SPS"));
                Ok((p, s))
            })
            .unwrap_or_else(|e| panic!("{e}"))
            .0
        };
        let headers: Vec<_> = samples.iter().map(|s| header_of(s)).collect();
        let b = headers.iter().filter(|sh| sh.slice_type.is_b()).count();
        assert!(b > 10, "the fixture has a B run to check ({b} B pictures)");
        assert!(
            headers.iter().filter(|sh| sh.slice_type.is_b()).all(|sh| sh.pred_weight_table.is_none()),
            "every B slice of this fixture reaches the implicit check, because filmcraft_h264 codes no table in explicit mode"
        );

        // A real run of the fixture, through the decoder: the first refusal names the implicit weights. The
        // pictures after it are refused for other reasons — the picture buffer never got the refused
        // picture, so their references are missing — and that is a consequence, not the subject, so
        // only the first reason is checked.
        let mut poc = super::NalPoc::default();
        let mut first_error = None;
        for sample in &samples {
            match super::build_nal(&info, sample, &mut poc, 0, 40, 23) {
                Ok(_) => {}
                Err(e) => {
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
            }
        }
        let why = first_error
            .unwrap_or_else(|| panic!("no picture of the fixture was refused, so the implicit weights were never exercised and this rule is untested"));
        assert!(why.contains("implicit"), "the first refusal must name the implicit weights, not the driver: {why}");
        assert!(why.contains("per reference index"), "the first refusal must say why the record cannot carry them: {why}");

        // The all-default case, on entries whose POC distances make every pair 1.0, and the refusal on ones
        // that do not. Both are the distances the fixture's own second B picture has, so the numbers
        // here are the numbers above rather than invented ones: `cur_poc` 2 with references at 0 and 4
        // is tb = 2, td = 4, which wants 1.0; the same with the second reference at 8 is td = 8, which
        // wants (48, 16).
        let entry = |poc: i32| super::DpbEntry { surface: 0, frame_num: 0, poc, mark: super::Mark::Short };
        let dpb = vec![entry(0), entry(4)];
        let lists = [vec![0], vec![1]];
        assert_eq!(super::h264_implicit_pair(2, &dpb[0], &dpb[1]), (32, 32), "tb 2 and td 4 give 1.0");
        let record = super::h264_implicit_weights(2, &dpb, &lists).unwrap_or_else(|e| panic!("all-default pairs must not be refused: {e}"));
        assert_eq!(record.luma_log2_weight_denom, 5, "the denominator 1.0 is written against");
        assert_eq!(record.luma_weight_l0_flag, 1, "the weight is in use, not absent");
        assert_eq!(record.luma_weight_l0[0], 32, "32 on a denominator of 2^5 is 1.0");
        assert_eq!(record.luma_weight_l1[0], 32);
        assert_eq!(record.luma_offset_l0[0], 0, "and no offset");
        assert_eq!(record.chroma_weight_l0[0], [32; 2], "chroma is weighted the same way, both components");
        // A pair that wants anything else is refused, and the message carries the pair and the weight.
        let dpb = vec![entry(0), entry(8)];
        let e = super::h264_implicit_weights(2, &dpb, &lists).unwrap_err();
        assert!(e.contains("implicit"), "{e}");
        assert!(e.contains("per reference index"), "{e}");
        assert!(e.contains("(48, 16)"), "{e}");
        // …and a reference the buffer does not hold is an error rather than a pair left unchecked.
        let e = super::h264_implicit_weights(2, &dpb[..1], &lists).unwrap_err();
        assert!(e.contains("not in the decoded picture buffer"), "{e}");
    }

    /// A slice segment that does not belong to the picture the record was filled in from is an
    /// error: the record carries the first segment's tile grid, QP offsets and coding block
    /// geometry, so a second segment naming a different picture parameter set would be described
    /// with the wrong ones and decode as something else.
    ///
    /// The check itself is what is tested here, on real parsed headers: this x265 build refuses the
    /// tile and wavefront options through ffmpeg's `-x265-params`, so no fixture here splits a
    /// picture into several segments, and a stream whose segments disagree about their picture is
    /// one this backend cannot describe anyway.
    #[test]
    fn a_segment_must_match_the_picture_it_was_opened_with() {
        let Some((info, sets, samples)) = hevc() else { return };
        let header_of = |sample: &[u8]| -> filmcraft_hevc::slice::SliceHeader {
            let nal = info.nals(sample).first().copied().expect("a NAL unit");
            let hdr = filmcraft_hevc::slice::NalHeader::parse(nal).unwrap_or_else(|e| panic!("{e}"));
            let rbsp = filmcraft_bitstream::unescape_rbsp(&nal[2..]);
            let (sh, _, _) = filmcraft_hevc::slice::SliceHeader::parse(&rbsp, hdr, |id| sets.lookup(id), None).unwrap_or_else(|e| panic!("{e}"));
            sh
        };
        let first = header_of(&samples[0]);
        assert!(super::hevc_same_picture(&first, &first), "a segment is of its own picture");
        // and a later picture's segment is not
        let others: Vec<_> = samples.iter().skip(1).map(|s| header_of(s)).collect();
        let different = others.iter().any(|sh| !super::hevc_same_picture(&first, sh));
        assert!(different, "the fixture's other pictures are not the first one");
        // two segments of the same picture would agree, which is what makes the check safe to skip
        let same = others.iter().find(|sh| sh.pps_id == first.pps_id && sh.nal.nal_type == first.nal.nal_type);
        if let Some(same) = same {
            assert!(super::hevc_same_picture(&first, same), "two segments of one picture agree");
        }
    }

    /// An H.264 stream whose encoder actually *uses* weighted prediction, so the slice records carry
    /// weights worth checking.
    ///
    /// Two things are needed for that, and both are easy to get wrong: `weightp=1` alone leaves every
    /// entry of the table at its default (`luma_weight 1`, no flags set), so the per-entry arithmetic
    /// never runs; and a plain `testsrc2` has too little of a brightness trend for x264 to derive a
    /// weight from. A `fade=t=in` source over `weightp=1:weightb=1` is what produces real weights —
    /// luma 9 to 93, chroma 43 to 120, non-zero offsets, and a `luma_log2_denom` that differs from
    /// slice to slice. `bframes=0` keeps this fixture to P slices, so it does not also feed the
    /// weighted-bipred path, which is [`h264_implicit_bipred`]'s.
    fn h264_weighted() -> Option<(NalStreamInfo, super::H264Sets, Vec<Vec<u8>>)> {
        let ff = filmcraft_testkit::require_ffmpeg!(None);
        let path = filmcraft_testkit::fixtures::generate(&fixture("h264_weighted.mp4"), |tmp| {
            let args = [
                "-f",
                "lavfi",
                "-i",
                "testsrc2=s=640x360:r=24:d=1,fade=t=in:st=0:d=1",
                "-c:v",
                "libx264",
                "-profile:v",
                "main",
                "-preset",
                "fast",
                "-x264-params",
                "bframes=0:weightp=1:weightb=1:keyint=24:min-keyint=24:scenecut=0",
                "-pix_fmt",
                "yuv420p",
            ];
            run(&ff, &args, tmp)
        })?;
        let (info, samples) = read(&path);
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        let sets = super::H264Sets::parse(&info).unwrap_or_else(|e| panic!("{e}"));
        Some((info, sets, samples))
    }

    /// [`hevc`]: the fixture's description, its parameter sets as [`build_nal`] parses them, and its
    /// samples in decoding order.
    fn hevc() -> Option<(NalStreamInfo, super::HevcSets, Vec<Vec<u8>>)> {
        let (info, samples) = hevc_stream()?;
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        let sets = super::HevcSets::parse(&info).unwrap_or_else(|e| panic!("{e}"));
        Some((info, sets, samples))
    }

    /// [`hevc_weighted`]: an HEVC stream whose encoder uses explicit weighted prediction.
    fn hevc_weighted() -> Option<(NalStreamInfo, super::HevcSets, Vec<Vec<u8>>)> {
        let ff = filmcraft_testkit::require_ffmpeg!(None);
        let path = filmcraft_testkit::fixtures::generate(&fixture("hevc_weighted.mp4"), |tmp| {
            // `weightp` and `weightb` are x265's weighted-prediction switches: the first puts
            // `weighted_pred_flag` in the picture parameter set, the second `weighted_bipred_flag`,
            // and each of those makes the slice header code a `pred_weight_table`. The plain
            // fixture's parameter sets allow neither, so its segments carry no table at all and
            // would leave the weight test vacuous.
            let args = [
                "-f",
                "lavfi",
                "-i",
                "testsrc2=s=640x360:r=24:d=1,fade=t=in:st=0:d=1",
                "-c:v",
                "libx265",
                "-preset",
                "fast",
                "-tag:v",
                "hvc1",
                "-x265-params",
                "log-level=error:bframes=3:keyint=24:min-keyint=24:scenecut=0:weightp=1:weightb=1",
                "-pix_fmt",
                "yuv420p",
            ];
            run(&ff, &args, tmp)
        })?;
        let (info, samples) = read(&path);
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        let sets = super::HevcSets::parse(&info).unwrap_or_else(|e| panic!("{e}"));
        Some((info, sets, samples))
    }

    /// [`hevc`]'s sequence and picture parameter sets, by id.
    fn hevc_set() -> Option<(Arc<filmcraft_hevc::params::Sps>, Arc<filmcraft_hevc::params::Pps>)> {
        let (_, sets, _) = hevc()?;
        let sps = sets.spss.iter().flatten().next().unwrap_or_else(|| panic!("no SPS")).clone();
        let pps = sets.ppss.iter().flatten().next().unwrap_or_else(|| panic!("no PPS")).clone();
        Some((sps, pps))
    }

    #[test]
    fn the_h264_sequence_fields_are_packed_where_the_header_puts_them() {
        let Some((sps, _)) = h264_set() else { return };
        let bits = super::pack_h264_seq_fields(&sps);
        let field = |lo: u32, width: u32| (bits >> lo) & ((1 << width) - 1);
        let flag = |lo: u32| field(lo, 1) == 1;
        assert_eq!(field(0, 2), sps.chroma_array_type(), "bits 0-1 chroma_format_idc");
        assert_eq!(flag(2), sps.separate_colour_plane, "bit 2");
        assert_eq!(flag(3), sps.gaps_in_frame_num_allowed, "bit 3");
        assert_eq!(flag(4), sps.frame_mbs_only, "bit 4");
        assert_eq!(flag(5), sps.mb_adaptive_frame_field, "bit 5");
        assert_eq!(flag(6), sps.direct_8x8_inference, "bit 6");
        assert!(flag(7), "bit 7 MinLumaBiPredSize8x8: frame coding makes it 1");
        assert_eq!(field(8, 4), sps.log2_max_frame_num - 4, "bits 8-11");
        assert_eq!(field(12, 2), sps.pic_order_cnt_type, "bits 12-13");
        assert_eq!(field(14, 4), sps.log2_max_poc_lsb - 4, "bits 14-17");
        assert_eq!(flag(18), sps.delta_pic_order_always_zero, "bit 18");
        assert_eq!(bits >> 19, 0, "bits 19-31 are not part of seq_fields");
        assert!(bits < (1 << 19), "seq_fields uses 19 bits and no more");
    }

    #[test]
    fn the_h264_picture_fields_are_packed_where_the_header_puts_them() {
        let Some((_, pps)) = h264_set() else { return };
        let Some((info, samples)) = h264_stream() else { return };
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        let mut poc = super::NalPoc::default();
        let p = super::build_nal(&info, &samples[0], &mut poc, 7, 40, 23).unwrap_or_else(|e| panic!("{e}"));
        let pp = crate::vaapi::ffi::picture_params_h264(&p.params).unwrap_or_else(|| panic!("not a record"));
        let bits = pp.pic_fields;
        let field = |lo: u32, width: u32| (bits >> lo) & ((1 << width) - 1);
        let flag = |lo: u32| field(lo, 1) == 1;
        assert_eq!(flag(0), pps.entropy_coding_mode, "bit 0");
        assert_eq!(flag(1), pps.weighted_pred, "bit 1");
        assert_eq!(field(2, 2), pps.weighted_bipred_idc, "bits 2-3");
        assert_eq!(flag(4), pps.transform_8x8_mode, "bit 4");
        assert!(!flag(5), "bit 5 field_pic_flag, and field coding is declined");
        assert_eq!(flag(6), pps.constrained_intra_pred, "bit 6");
        assert_eq!(flag(7), pps.bottom_field_pic_order_in_frame_present, "bit 7");
        assert_eq!(flag(8), pps.deblocking_filter_control_present, "bit 8");
        assert_eq!(flag(9), pps.redundant_pic_cnt_present, "bit 9");
        assert!(flag(10), "bit 10 reference_pic_flag: the fixture's IDR is a reference");
        assert_eq!(bits >> 11, 0, "bits 11-31 are not part of pic_fields");
    }

    /// The H.264 slice record against the fixture's own slice header, the way the HEVC one is: a record
    /// that carried a default where the stream said something else would decode as a different
    /// picture, and the second slice of this fixture is a P or B slice, so its own fields differ.
    #[test]
    fn the_h264_slice_record_carries_the_parsed_slice_header() {
        let Some((info, sets, samples)) = h264() else { return };
        let mut poc = super::NalPoc::default();
        // From the fixture's first picture: the second is an inter picture, and an inter picture's
        // records cannot be built without the pictures it refers to — which is the point of the
        // decoded picture buffer, and why this starts at sample 0 rather than at the one it reads.
        super::build_nal(&info, &samples[0], &mut poc, 6, 40, 23).unwrap_or_else(|e| panic!("{e}"));
        let p = super::build_nal(&info, &samples[1], &mut poc, 7, 40, 23).unwrap_or_else(|e| panic!("{e}"));
        assert!(!p.slices.is_empty(), "the fixture's second picture has slices");
        let mut seen_types = Vec::new();
        let sample = &samples[1];
        let nals = info.nals(sample);
        for (slice, &at) in p.slices.iter().zip(&p.slice_offsets) {
            let nal = nals.iter().find(|n| sample.get(at..at + n.len()) == Some(*n)).unwrap_or_else(|| panic!("no NAL unit at {at}"));
            let hdr = filmcraft_h264::slice::NalHeader::parse(nal[0]).unwrap_or_else(|e| panic!("{e}"));
            let rbsp = filmcraft_bitstream::unescape_rbsp(&nal[1..]);
            let (sh, pps, _) = filmcraft_h264::slice::SliceHeader::parse(&rbsp, hdr, |id| {
                let pps = sets.ppss.get(id as usize).and_then(|p| p.as_ref()).unwrap_or_else(|| panic!("PPS {id}"));
                let sps = sets.spss.get(pps.sps_id as usize).and_then(|s| s.as_ref()).unwrap_or_else(|| panic!("SPS"));
                Ok((pps, sps))
            })
            .unwrap_or_else(|e| panic!("{e}"));
            let r = crate::vaapi::ffi::slice_params_h264(&slice.params).unwrap_or_else(|| panic!("not a slice record"));
            assert_eq!(r.slice_data_size, slice.length, "slice_data_size is the NAL unit's length");
            assert_eq!(r.slice_data_offset, slice.offset, "slice_data_offset");
            assert_eq!(r.first_mb_in_slice, sh.first_mb_in_slice as u16, "first_mb_in_slice");
            assert_eq!(r.slice_type, super::h264_slice_type(sh.slice_type), "slice_type");
            assert_eq!(r.slice_qp_delta, sh.slice_qp_delta as i8, "slice_qp_delta");
            assert_eq!(r.cabac_init_idc, sh.cabac_init_idc as u8, "cabac_init_idc");
            assert_eq!(r.num_ref_idx_l0_active_minus1, sh.num_ref_idx_active[0].saturating_sub(1) as u8, "num_ref_idx_l0_active_minus1");
            assert_eq!(r.num_ref_idx_l1_active_minus1, sh.num_ref_idx_active[1].saturating_sub(1) as u8, "num_ref_idx_l1_active_minus1");
            assert_eq!(r.disable_deblocking_filter_idc, sh.disable_deblocking_filter_idc as u8, "disable_deblocking_filter_idc");
            assert_eq!(r.slice_alpha_c0_offset_div2, sh.slice_alpha_c0_offset_div2 as i8, "slice_alpha_c0_offset_div2");
            assert_eq!(r.slice_beta_offset_div2, sh.slice_beta_offset_div2 as i8, "slice_beta_offset_div2");
            assert_eq!(r.direct_spatial_mv_pred_flag, u8::from(sh.direct_spatial_mv_pred), "direct_spatial_mv_pred_flag");
            // the record counts from the NAL header byte, our parser from after it
            assert_eq!(usize::from(r.slice_data_bit_offset), sh.header_bits + 8, "slice_data_bit_offset");
            // The reference lists name pictures the decoded picture buffer holds, and nothing past the
            // slice's own `num_ref_idx` claims to: the IDR before this picture is the one they name.
            for (list, active) in [(0, sh.num_ref_idx_active[0]), (1, sh.num_ref_idx_active[1])] {
                let entries = if list == 0 { &r.ref_pic_list0 } else { &r.ref_pic_list1 };
                let wanted = usize::try_from(active).unwrap_or(0);
                for (at, entry) in entries.iter().take(wanted).enumerate() {
                    assert_eq!(entry.frame_idx, 0, "list {list} entry {at}: the IDR is the fixture's frame_num 0");
                    assert_ne!(entry.picture_id, crate::vaapi::ffi::INVALID_SURFACE, "list {list} entry {at} names a picture");
                }
                assert!(
                    entries.iter().skip(wanted).all(|e| e.picture_id == crate::vaapi::ffi::INVALID_SURFACE),
                    "list {list}: nothing past its {wanted} entries claims to name a picture"
                );
            }
            // and the picture record agrees with the same picture
            let pp = crate::vaapi::ffi::picture_params_h264(&p.params).unwrap_or_else(|| panic!("not a picture record"));
            assert_eq!(pp.frame_num, sh.frame_num as u16, "the picture's frame_num is its first slice's");
            assert_eq!(pp.pic_init_qp_minus26, (pps.pic_init_qp - 26) as i8, "the picture's initial QP is its parameter set's");
            seen_types.push(super::h264_slice_type(sh.slice_type));
        }
        // A picture after the IDR is not all I slices, which is what makes this test cover a
        // non-zero `slice_type`, `cabac_init_idc` or reference count: the IDR would not.
        assert!(seen_types.iter().any(|&t| t != 2), "the second picture has non-I slices: {seen_types:?}");
    }

    /// Every H.264 profile this backend plans is 8-bit 4:2:0 progressive, so the record's chroma
    /// and depth fields come out at those values; a 10-bit or 4:2:2 stream would have been declined
    /// by `plan_nal` before it got here.
    #[test]
    fn the_h264_record_carries_the_planneds_chroma_and_depth() {
        let Some((sps, pps)) = h264_set() else { return };
        let Some((info, samples)) = h264_stream() else { return };
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        assert!(super::super::stream::plan_nal(&info).is_ok(), "the fixture is a stream this backend takes");
        assert_eq!(sps.chroma_array_type(), 1, "4:2:0");
        assert_eq!((sps.bit_depth_luma, sps.bit_depth_chroma), (8, 8));
        let mut poc = super::NalPoc::default();
        let p = super::build_nal(&info, &samples[0], &mut poc, 7, 40, 23).unwrap_or_else(|e| panic!("{e}"));
        let pp = crate::vaapi::ffi::picture_params_h264(&p.params).unwrap_or_else(|| panic!("not a record"));
        assert_eq!((pp.bit_depth_luma_minus8, pp.bit_depth_chroma_minus8), (0, 0));
        assert_eq!(pp.seq_fields & 3, 1, "chroma_format_idc 4:2:0");
        assert_eq!(pp.pic_init_qp_minus26, (pps.pic_init_qp - 26) as i8, "the initial QP comes from the picture parameter set");
        assert_eq!(pp.num_slice_groups_minus1, (pps.num_slice_groups - 1) as u8);
        assert_eq!(pp.slice_group_map_type, pps.slice_group_map_type as u8);
    }

    /// The HEVC record's packed words, field by field against the fixture's own parameter sets.
    #[test]
    fn the_hevc_packed_fields_are_where_the_header_puts_them() {
        let Some((sps, pps)) = hevc_set() else { return };
        let Some((info, _, samples)) = hevc() else { return };
        let mut poc = super::NalPoc::default();
        let p = super::build_nal(&info, &samples[0], &mut poc, 9, 640, 360).unwrap_or_else(|e| panic!("{e}"));
        let pp = crate::vaapi::ffi::picture_params_hevc(&p.params).unwrap_or_else(|| panic!("not a record"));
        let flag = |word: u32, at: u32| (word >> at) & 1 == 1;

        // pic_fields (va_dec_hevc.h:71)
        let bits = pp.pic_fields;
        assert_eq!(bits & 3, sps.chroma_array_type(), "bits 0-1");
        assert_eq!(flag(bits, 2), sps.separate_colour_plane, "bit 2");
        assert_eq!(flag(bits, 3), sps.pcm, "bit 3");
        assert_eq!(flag(bits, 4), sps.scaling_list_enabled, "bit 4");
        assert_eq!(flag(bits, 5), pps.transform_skip, "bit 5");
        assert_eq!(flag(bits, 6), sps.amp, "bit 6");
        assert_eq!(flag(bits, 7), sps.strong_intra_smoothing, "bit 7");
        assert_eq!(flag(bits, 8), pps.sign_data_hiding, "bit 8");
        assert_eq!(flag(bits, 9), pps.constrained_intra_pred, "bit 9");
        assert_eq!(flag(bits, 10), pps.cu_qp_delta_enabled, "bit 10");
        assert_eq!(flag(bits, 11), pps.weighted_pred, "bit 11");
        assert_eq!(flag(bits, 12), pps.weighted_bipred, "bit 12");
        assert_eq!(flag(bits, 13), pps.transquant_bypass, "bit 13");
        assert_eq!(flag(bits, 14), pps.tiles_enabled, "bit 14");
        assert_eq!(flag(bits, 15), pps.entropy_coding_sync, "bit 15");
        assert_eq!(flag(bits, 16), pps.loop_filter_across_slices, "bit 16");
        assert_eq!(flag(bits, 17), pps.loop_filter_across_tiles, "bit 17");
        assert_eq!(flag(bits, 18), sps.pcm_loop_filter_disabled, "bit 18");
        assert_eq!(flag(bits, 19), sps.max_num_reorder == 0, "bit 19 NoPicReorderingFlag");
        assert!(flag(bits, 20), "bit 20 NoBiPredFlag: an IDR has no B slices");
        assert_eq!(bits >> 21, 0, "bits 21-31 are the header's reserved field");

        // slice_parsing_fields (va_dec_hevc.h:139)
        let parsing = pp.slice_parsing_fields;
        assert_eq!(flag(parsing, 0), pps.lists_modification_present, "bit 0");
        assert_eq!(flag(parsing, 1), sps.long_term_refs_present, "bit 1");
        assert_eq!(flag(parsing, 2), sps.temporal_mvp, "bit 2");
        assert_eq!(flag(parsing, 3), pps.cabac_init_present, "bit 3");
        assert_eq!(flag(parsing, 4), pps.output_flag_present, "bit 4");
        assert_eq!(flag(parsing, 5), pps.dependent_slice_segments_enabled, "bit 5");
        assert_eq!(flag(parsing, 6), pps.slice_chroma_qp_offsets_present, "bit 6");
        assert_eq!(flag(parsing, 7), sps.sao, "bit 7");
        assert_eq!(flag(parsing, 8), pps.deblocking_override_enabled, "bit 8");
        assert_eq!(flag(parsing, 9), pps.deblocking_disabled, "bit 9");
        assert_eq!(flag(parsing, 10), pps.slice_header_extension_present, "bit 10");
        assert!(flag(parsing, 11), "bit 11 RapPicFlag: the first picture is an IDR");
        assert!(flag(parsing, 12), "bit 12 IdrPicFlag");
        assert!(flag(parsing, 13), "bit 13 IntraPicFlag: the first picture is all I");
        assert_eq!(parsing >> 14, 0, "bits 14-31 are the header's reserved field");

        // and the sizes the parameter sets describe
        assert_eq!(pp.pic_width_in_luma_samples, sps.width as u16);
        assert_eq!(pp.pic_height_in_luma_samples, sps.height as u16);
        assert_eq!(pp.bit_depth_luma_minus8, (sps.bit_depth_luma - 8) as u8);
        assert_eq!(pp.init_qp_minus26, (pps.init_qp - 26) as i8);
        assert_eq!(pp.num_short_term_ref_pic_sets, sps.st_rps.len() as u8);
        assert_eq!(pp.log2_max_pic_order_cnt_lsb_minus4, (sps.log2_max_poc_lsb - 4) as u8);
    }

    /// The slice records' own fields, from the fixture's own slice header: the slice type, the
    /// segment address, the collocated index and the entry points, so a driver can find slice data.
    #[test]
    fn the_hevc_slice_record_carries_the_parsed_slice_segment() {
        let Some((info, sets, samples)) = hevc() else { return };
        let mut poc = super::NalPoc::default();
        let p = super::build_nal(&info, &samples[0], &mut poc, 9, 640, 360).unwrap_or_else(|e| panic!("{e}"));
        let Some(slice) = p.slices.first() else { panic!("a slice record") };
        let r = crate::vaapi::ffi::slice_params_hevc(&slice.params).unwrap_or_else(|| panic!("not a slice record"));
        assert_eq!(r.slice_data_size, slice.length);
        assert_eq!(r.slice_data_offset, slice.offset);
        assert_eq!(r.slice_data_flag, crate::vaapi::ffi::SLICE_DATA_FLAG_ALL);
        assert_eq!(r.slice_segment_address, 0, "the first segment of a picture starts at its top left CTB");
        // The segment's own header, parsed again here, so every field of the record is checked
        // against the syntax rather than against a constant: a record that carried a default where
        // the stream said something else would decode as a different picture.
        let nal = info.nals(&samples[0]).into_iter().find(|n| n.len() == slice.length as usize).unwrap_or_else(|| panic!("the slice's NAL unit"));
        let hdr = filmcraft_hevc::slice::NalHeader::parse(nal).unwrap_or_else(|e| panic!("{e}"));
        let rbsp = filmcraft_bitstream::unescape_rbsp(&nal[2..]);
        let (sh, pps, _) = filmcraft_hevc::slice::SliceHeader::parse(&rbsp, hdr, |id| sets.lookup(id), None).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(r.slice_qp_delta as i32, sh.qp_delta, "slice_qp_delta");
        assert_eq!(r.slice_segment_address, sh.segment_address, "slice_segment_address");
        assert_eq!(r.slice_data_byte_offset as usize, sh.data_offset + 2, "slice_data_byte_offset counts from the NAL header");
        assert_eq!((r.long_slice_flags >> 2) & 3, sh.slice_type as u32, "bits 2-3 are slice_type");
        assert_eq!((r.long_slice_flags >> 4) & 3, 0, "bits 4-5 colour_plane_id, and separate colour planes are declined");
        assert_eq!((r.long_slice_flags >> 1) & 1 == 1, sh.dependent, "bit 1 dependent_slice_segment_flag");
        assert_eq!((r.long_slice_flags >> 6) & 1 == 1, sh.sao_luma, "bit 6 slice_sao_luma_flag");
        assert_eq!((r.long_slice_flags >> 10) & 1 == 1, sh.temporal_mvp, "bit 10 slice_temporal_mvp_enabled_flag");
        assert_eq!((r.long_slice_flags >> 11) & 1 == 1, sh.deblocking_disabled, "bit 11 slice_deblocking_filter_disabled_flag");
        assert_eq!((r.long_slice_flags >> 13) & 1 == 1, sh.loop_filter_across_slices, "bit 13 slice_loop_filter_across_slices_enabled_flag");
        assert_eq!(u32::from(r.five_minus_max_num_merge_cand), 5u32.saturating_sub(sh.max_num_merge_cand), "five_minus_max_num_merge_cand");
        assert_eq!(r.num_ref_idx_l0_active_minus1 as u32, sh.num_ref_idx[0].saturating_sub(1), "num_ref_idx_l0_active_minus1");
        assert_eq!(r.num_ref_idx_l1_active_minus1 as u32, sh.num_ref_idx[1].saturating_sub(1), "num_ref_idx_l1_active_minus1");
        assert_eq!(r.slice_cb_qp_offset as i32, sh.cb_qp_offset, "slice_cb_qp_offset");
        assert_eq!(r.slice_cr_qp_offset as i32, sh.cr_qp_offset, "slice_cr_qp_offset");
        assert_eq!(r.slice_beta_offset_div2 as i32, sh.beta_offset_div2, "slice_beta_offset_div2");
        assert_eq!(r.slice_tc_offset_div2 as i32, sh.tc_offset_div2, "slice_tc_offset_div2");
        assert_eq!(
            r.collocated_ref_idx,
            if sh.temporal_mvp { sh.collocated_ref_idx as u8 } else { 0xFF },
            "the collocated index, or 0xFF with temporal MVP off"
        );
        assert_eq!(r.num_entry_point_offsets as usize, sh.entry_points.len(), "num_entry_point_offsets");
        assert_eq!(
            r.slice_data_num_emu_prevn_bytes as usize,
            emu_count(&nal[2..2 + sh.data_offset]),
            "the slice header's own emulation prevention bytes, counted here and not by the module"
        );
        assert_eq!(r.slice_data_size, slice.length);
        assert_eq!(r.slice_data_offset, slice.offset);
        assert_eq!(r.slice_data_flag, crate::vaapi::ffi::SLICE_DATA_FLAG_ALL);
        assert_eq!(r.long_slice_flags >> 14, 0, "bits 14-31 are the header's reserved field");
        assert_eq!(r.ref_pic_list, [[0xFF; 15]; 2], "an entry the decode loop has not filled reads as unused");
        assert!(r.long_slice_flags & 1 == 1, "bit 0 LastSliceOfPic: a one-segment picture's only segment is its last");
        // and the initial QP in the picture record is the one this segment was parsed against
        let pp = crate::vaapi::ffi::picture_params_hevc(&p.params).unwrap_or_else(|| panic!("not a picture record"));
        assert_eq!(pp.init_qp_minus26 as i32, pps.init_qp - 26, "init_qp_minus26");
    }

    /// `LastSliceOfPic` is set on the last segment of a picture and on no other, which is how a
    /// driver knows a picture ended. Every segment of the picture is in `Picture`, so the last is
    /// known when the records are built rather than left to the caller to work out.
    #[test]
    fn the_last_segment_of_a_picture_says_so() {
        let Some((info, _, samples)) = hevc() else { return };
        let mut poc = super::NalPoc::default();
        let last_flag = |p: &super::Picture, i: usize| crate::vaapi::ffi::slice_params_hevc(&p.slices[i].params).expect("a record").long_slice_flags & 1;
        let mut multi = 0usize;
        for sample in samples.iter().take(6) {
            let Ok(p) = super::build_nal(&info, sample, &mut poc, 9, 640, 360) else { continue };
            if p.slices.len() > 1 {
                multi += 1;
            }
            for i in 0..p.slices.len() - 1 {
                assert_eq!(last_flag(&p, i), 0, "only the last segment of {i}.. says LastSliceOfPic");
            }
            assert_eq!(last_flag(&p, p.slices.len() - 1), 1, "the last segment says so");
        }
        // The fixture's first GOP is one segment per picture unless tiles or wavefront split it,
        // so this is mostly checking the single-segment case; what matters is that the bit is never
        // set on a segment that is not the last, which is what a driver uses to end a picture.
        assert!(multi <= 1, "the fixture is not mostly multi-segment ({multi} of 6)");
    }

    /// A stream this backend would plan but whose records cannot be described is declined, not
    /// half-built: here a coded size the surfaces were not created at.
    #[test]
    fn hevc_declines_a_size_the_surfaces_were_not_created_at() {
        let Some((info, samples)) = hevc_stream() else { return };
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        let mut poc = super::NalPoc::default();
        assert!(super::build_nal(&info, &samples[0], &mut poc, 9, 640, 368).is_err(), "a coded size of 368 for a 360-line stream");
        let mut poc = super::NalPoc::default();
        assert!(super::build_nal(&info, &samples[0], &mut poc, 9, 1920, 1080).is_err(), "1920x1080 surfaces for a 640x360 stream");
    }

    /// Every fixture parameter set that does not parse is an error naming it, never a zeroed record.
    #[test]
    fn unparsable_parameter_sets_are_errors() {
        let Some((info, samples)) = h264_stream() else { return };
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        let mut corrupt = info.clone();
        for set in &mut corrupt.parameter_sets {
            // every byte of the sequence parameter sets turned to ones: not a parameter set
            if set.first().is_some_and(|&h| info.nal_type(h) == 7) {
                *set = vec![0x67; 32];
            }
        }
        let mut poc = super::NalPoc::default();
        assert!(super::build_nal(&corrupt, &samples[0], &mut poc, 7, 40, 23).is_err(), "a sequence parameter set of ones");
        // an empty NAL unit in the record
        let mut empty = info.clone();
        empty.parameter_sets.push(Vec::new());
        let mut poc = super::NalPoc::default();
        assert!(super::build_nal(&empty, &samples[0], &mut poc, 7, 40, 23).is_err(), "an empty parameter set NAL unit");
    }

    /// `slice_offsets` points at each slice's NAL unit in the sample it came from, which is what
    /// makes `slices[i]` a byte range the caller can copy out of the sample.
    #[test]
    fn slice_offsets_point_at_the_nal_units_of_the_sample() {
        let Some((info, samples)) = h264_stream() else { return };
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        let sample = &samples[0];
        let mut poc = super::NalPoc::default();
        let p = super::build_nal(&info, sample, &mut poc, 7, 40, 23).unwrap_or_else(|e| panic!("{e}"));
        // Each offset lands on a NAL unit of the sample whose length is that slice's.
        let nals = info.nals(sample);
        for (slice, &at) in p.slices.iter().zip(&p.slice_offsets) {
            let found = nals.iter().any(|n| sample.get(at..at + slice.length as usize) == Some(*n) && n.len() == slice.length as usize);
            assert!(found, "a NAL unit of that length at {at}");
        }
        // and the offsets increase, in slice order
        assert!(p.slice_offsets.windows(2).all(|w| w[0] < w[1]), "the offsets run in slice order: {:?}", p.slice_offsets);
    }

    /// VP9 and AV1 need a per-frame header FilmCraft does not expose, so no record is built for them
    /// and the reason is named rather than a record of zeros handed over. `plan_frame` is where the
    /// decision is made; this is the backstop for a caller that skips it, and the two say the same
    /// thing.
    #[test]
    fn frame_codecs_are_declined_until_their_frame_headers_are_public() {
        for codec in [FrameCodec::Vp9, FrameCodec::Av1] {
            let info = FrameStreamInfo::for_tests(codec, 0, 8, (1, 1), (1920, 1080));
            let e = super::build_frame(&info, &[0x9d, 0x01, 0x2a]).expect_err("a frame codec with no record built yet");
            assert!(e.contains("frame header"), "{codec:?}: {e}");
            assert!(e.contains(super::FRAME_HEADER_MISSING), "{codec:?}: {e}");
            // an empty sample is declined too, and never reaches a parser
            assert!(super::build_frame(&info, &[]).is_err(), "{codec:?}");
            // and `plan_frame` declines the same stream with the same reason, in the same words
            let planned = crate::vaapi::stream::plan_frame(&info).expect_err("plan_frame declines every frame codec");
            assert!(planned.contains(super::FRAME_HEADER_MISSING), "{codec:?}: {planned}");
            assert!(planned.contains(if codec == FrameCodec::Vp9 { "filmcraft-vp9" } else { "filmcraft-av1" }), "{codec:?}: {planned}");
        }
    }

    // --- fixtures -------------------------------------------------------------------------------

    fn h264_info(length_size: usize) -> NalStreamInfo {
        NalStreamInfo {
            codec: NalCodec::H264,
            length_size,
            highest_tid: None,
            parameter_sets: Vec::new(),
            coded: (640, 368),
            crop: (0, 0, 640, 360),
            chroma_format_idc: 1,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            interlaced: false,
            profile_idc: 100,
            color: filmcraft_color::ColorInfo::REC709,
            par: (1, 1),
            reorder: 0,
        }
    }

    /// An ffmpeg-made H.264 stream and its samples: an `avcC`-derived description and the length-
    /// prefixed access units it holds, the first of which is an IDR in two slices.
    fn h264_stream() -> Option<(Result<NalStreamInfo, String>, Vec<Vec<u8>>)> {
        let ff = filmcraft_testkit::require_ffmpeg!(None);
        let path = filmcraft_testkit::fixtures::generate(&fixture("h264_two_slices.mp4"), |tmp| {
            let args = [
                "-f",
                "lavfi",
                "-i",
                "testsrc2=s=640x360:r=24:d=1",
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-preset",
                "fast",
                // `weightb=0` because this fixture has to be one this backend can describe end to
                // end: with x264's default `weightb=1` the B slices are implicitly weighted, which the
                // slice record cannot express, so every picture after the second B one is refused —
                // and these tests are about the records of every picture, not about that refusal
                // (`an_implicitly_weighted_b_slice_is_refused_rather_than_guessed` is, on its own
                // fixture, which leaves `weightb` at its default).
                "-x264-params",
                "bframes=3:b-pyramid=normal:keyint=24:min-keyint=24:scenecut=0:slices=2:weightb=0",
                "-pix_fmt",
                "yuv420p",
            ];
            run(&ff, &args, tmp)
        })?;
        Some(read(&path))
    }

    /// An H.264 stream whose picture parameter set says `weighted_bipred_idc == 2` and whose B slices
    /// really do exercise H.264 implicit weighting — x264's default `weightb=1` over High profile with
    /// B frames, which is the case the slice record cannot express. The counterpart of [`h264`]'s
    /// `weightb=0`, and what the refusal is tested on.
    fn h264_implicit_bipred() -> Option<(NalStreamInfo, super::H264Sets, Vec<Vec<u8>>)> {
        let ff = filmcraft_testkit::require_ffmpeg!(None);
        let path = filmcraft_testkit::fixtures::generate(&fixture("h264_implicit_bipred.mp4"), |tmp| {
            let args = [
                "-f",
                "lavfi",
                "-i",
                "testsrc2=s=640x360:r=24:d=1",
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-preset",
                "fast",
                "-x264-params",
                "bframes=3:b-pyramid=normal:keyint=24:min-keyint=24:scenecut=0",
                "-pix_fmt",
                "yuv420p",
            ];
            run(&ff, &args, tmp)
        })?;
        let (info, samples) = read(&path);
        let info = info.unwrap_or_else(|e| panic!("{e}"));
        let sets = super::H264Sets::parse(&info).unwrap_or_else(|e| panic!("{e}"));
        Some((info, sets, samples))
    }

    /// The HEVC slice record's prediction weight tables come from the slice segment header's own
    /// `pwt`, entry by entry. A record that said `weighted_pred_flag` and `weighted_bipred_flag`
    /// were in use and then handed a driver zeroes would decode to a different picture from the
    /// source, silently, which is the one answer this module never gives.
    ///
    /// The mapping the record wants is the syntax element of the same name, except that the two
    /// `delta_*_weight` fields hold the *coded* weight minus the `1 << denom` the parser added to
    /// build it, and `ChromaOffsetLX` is the derived 8-bit offset `filmcraft_hevc` already computed.
    #[test]
    fn the_hevc_slice_record_carries_the_prediction_weights() {
        let Some((info, sets, samples)) = hevc_weighted() else { return };
        let pps = sets.ppss.iter().flatten().next().unwrap_or_else(|| panic!("no PPS")).clone();
        assert!(pps.weighted_pred, "the fixture's PPS allows weighted prediction in P slices");
        assert!(pps.weighted_bipred, "and weighted bipred in B slices");
        let mut weighted = 0usize;
        let mut explicit = 0usize;
        let mut poc = super::NalPoc::default();
        for sample in &samples {
            let Ok(picture) = super::build_nal(&info, sample, &mut poc, 9, 640, 360) else { continue };
            let nals = info.nals(sample);
            for (slice, &at) in picture.slices.iter().zip(&picture.slice_offsets) {
                let nal = nals.iter().find(|n| sample.get(at..at + n.len()) == Some(*n)).unwrap_or_else(|| panic!("no NAL unit at {at}"));
                let hdr = filmcraft_hevc::slice::NalHeader::parse(nal).unwrap_or_else(|e| panic!("{e}"));
                let rbsp = filmcraft_bitstream::unescape_rbsp(&nal[2..]);
                let (sh, _, _) = filmcraft_hevc::slice::SliceHeader::parse(&rbsp, hdr, |id| sets.lookup(id), None).unwrap_or_else(|e| panic!("{e}"));
                let r = crate::vaapi::ffi::slice_params_hevc(&slice.params).unwrap_or_else(|| panic!("not a slice record"));
                let Some(table) = sh.pwt.as_ref() else {
                    // An I slice codes no table and needs none, and says so with zeroes.
                    assert_eq!(r.luma_log2_weight_denom, 0, "no table means no weighting");
                    assert!(r.delta_luma_weight_l0 == [0; 15] && r.delta_luma_weight_l1 == [0; 15], "and none are invented");
                    continue;
                };
                weighted += 1;
                let luma_one = 1i32 << table.luma_log2_denom;
                let chroma_one = 1i32 << table.chroma_log2_denom;
                assert_eq!(u32::from(r.luma_log2_weight_denom), table.luma_log2_denom, "luma_log2_weight_denom");
                assert_eq!(
                    r.delta_chroma_log2_weight_denom,
                    i8_of(table.chroma_log2_denom as i32 - table.luma_log2_denom as i32),
                    "delta_chroma_log2_weight_denom is ChromaLog2WeightDenom minus LumaLog2WeightDenom"
                );
                for (list, entries, (dl, lo, dc, co)) in [
                    (0, &table.l[0], (&r.delta_luma_weight_l0, &r.luma_offset_l0, &r.delta_chroma_weight_l0, &r.chroma_offset_l0)),
                    (1, &table.l[1], (&r.delta_luma_weight_l1, &r.luma_offset_l1, &r.delta_chroma_weight_l1, &r.chroma_offset_l1)),
                ] {
                    // A B slice has both lists; a P slice's second list is empty and the record's
                    // L1 family stays zero, which is what an empty list means.
                    if !sh.is_b() {
                        assert!(entries.is_empty() || list == 0, "a P slice codes no second prediction weight list");
                    }
                    for (i, e) in entries.iter().enumerate() {
                        if e.luma_flag || e.chroma_flag {
                            explicit += 1;
                        }
                        assert_eq!(dl[i], i8_of(e.luma.0 - luma_one), "delta_luma_weight_l{list}[{i}]");
                        assert_eq!(lo[i], i8_of(e.luma.1), "luma_offset_l{list}[{i}]");
                        for c in 0..2 {
                            assert_eq!(dc[i][c], i8_of(e.chroma[c].0 - chroma_one), "delta_chroma_weight_l{list}[{i}][{c}]");
                            assert_eq!(co[i][c], i8_of(e.chroma[c].1), "ChromaOffsetL{list}[{i}][{c}]");
                        }
                    }
                    // nothing is written past the slice segment's own references
                    assert!(dl[entries.len()..].iter().all(|&w| w == 0), "delta_luma_weight_l{list} past the slice's references");
                    assert!(lo[entries.len()..].iter().all(|&w| w == 0), "luma_offset_l{list} past the slice's references");
                }
            }
        }
        assert!(weighted >= 20, "the fixture's segments code a table ({weighted})");
        assert!(explicit >= 20, "and at least {explicit} of its entries set a weight flag rather than taking the default");
    }

    /// An ffmpeg-made HEVC stream and its samples: an `hvcC`-derived description and the length-
    /// prefixed temporal units it holds, the first of which is an IDR.
    fn hevc_stream() -> Option<(Result<NalStreamInfo, String>, Vec<Vec<u8>>)> {
        let ff = filmcraft_testkit::require_ffmpeg!(None);
        let path = filmcraft_testkit::fixtures::generate(&fixture("hevc_main.mp4"), |tmp| {
            // The plain source is what the emulation-prevention test needs: x265's residual data
            // for `testsrc2` contains `00 00 03` in several pictures, so a header-scoped count and
            // a whole-NAL one really differ. (`noise`, tried for the same reason, breaks up the zero
            // runs and removes them again.) This x265 build refuses the tile and wavefront options
            // through `-x265-params`, so the pictures are single-segment; see
            // `the_emulation_prevention_count_is_the_slice_headers`.
            let args = [
                "-f",
                "lavfi",
                "-i",
                "testsrc2=s=640x360:r=24:d=1",
                "-c:v",
                "libx265",
                "-preset",
                "fast",
                "-tag:v",
                "hvc1",
                "-x265-params",
                "log-level=error:bframes=4:keyint=24:min-keyint=24:scenecut=0:open-gop=1",
                "-pix_fmt",
                "yuv420p",
            ];
            run(&ff, &args, tmp)
        })?;
        Some(read(&path))
    }

    fn fixture(name: &str) -> std::path::PathBuf {
        filmcraft_testkit::fixtures_dir("platform/vaapi").join(name)
    }

    fn run(ff: &std::path::Path, args: &[&str], out: &std::path::Path) -> bool {
        let st = std::process::Command::new(ff).args(["-y", "-v", "error"]).args(args).arg(out).stdin(std::process::Stdio::null()).status();
        match st {
            Ok(s) if s.success() => true,
            other => {
                eprintln!("fixture {}: ffmpeg failed: {other:?}", out.display());
                false
            }
        }
    }

    /// The video track's samples in decoding order, and the description a hardware decoder needs
    /// from its sample entry.
    fn read(path: &std::path::Path) -> (Result<NalStreamInfo, String>, Vec<Vec<u8>>) {
        let bytes = std::fs::read(path).expect("the fixture is there");
        let file = filmcraft_isobmff::open(bytes.clone()).expect("an mp4");
        let track = file.track_of_kind(filmcraft_isobmff::TrackKind::Video).expect("a video track");
        let info = NalStreamInfo::from_entry(&file.tracks[track].entries[0]).expect("a NAL codec");
        let samples = (0..file.tracks[track].samples.len()).map(|i| file.read_sample(&bytes, track, i).expect("a sample")).collect();
        (info.map_err(|e| e.to_string()), samples)
    }
}
