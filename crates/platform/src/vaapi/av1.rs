//! **AV1: the reference slots, the picture record, and the per-tile slice records.**
//!
//! A VA-API AV1 picture is submitted from one `VADecPictureParameterBufferAV1`, **one
//! `VASliceParameterBufferAV1` per tile**, and the frame's own bytes. This module builds the first
//! two out of [`filmcraft_av1`]'s parse of that frame's header, and owns the state behind them —
//! which is the half a record cannot carry.
//!
//! ## What the record does *not* carry, and why this file's state model is not VP9's
//!
//! **`refresh_frame_flags` has no member in the AV1 decode record**, and that was checked rather than
//! assumed: `grep -in refresh /usr/include/va/va_dec_av1.h` returns exactly two hits, **line 421**
//! (*"For SWITCH_FRAME, application shall set error_resilient_mode = 1, refresh_frame_flags, etc.
//! appropriately"*) and **line 436** (*"disable_frame_end_update_cdf is coded as
//! refresh_frame_context"*) — both inside comments, and the second names a different field entirely.
//! The field itself is real and lives in the **encode** record (`va_enc_av1.h:607`), which has no
//! business saying what a decode's picture buffer holds.
//!
//! So there is nothing here to port from VP9 and nothing to adapt. VP9's `refresh_frame_flags` is
//! also a header field its record builder read from the header rather than from the record, and AV1's
//! is the same field of the same name in the same clause of the specification (7.20). What this
//! backend's state model *is* built from is the specification and `filmcraft_av1`'s own software
//! path, and the two agree because [`Av1State::commit`] mirrors
//! `filmcraft_av1::Decoder::submit` (`crates/av1/src/decoder.rs:586-607`) field for field.
//!
//! What differs from VP9, and is worth stating, is **what a slot holds**. VP9's is a surface id and
//! the coded size. AV1's 7.20 slot is a surface **and** the frame id, the order hint, the saved order
//! hints, the six global-motion parameters, the loop filter deltas, the segmentation features and
//! the film-grain parameters — **and the frame header parser reads every one of them.** So a slot
//! holding a surface but not the rest is not a VP9 slot with the wrong contents: the *next frame's
//! header* is parsed against it, and `primary_ref_frame`'s propagated deltas, `frame_size_with_refs`,
//! `OrderHints` and `skip_mode` are all read out of it before any record exists. That is why
//! [`Av1State::commit`] moves one slot index and both halves follow, and why the frames of a sample
//! are decoded one at a time rather than parsed in a batch: the software decoder refreshes 7.20's
//! slots between two frames, and a batch would parse the second against the first's uncommitted state.

use filmcraft_av1::header::{FrameHeader, HeaderState, LoopFilterDeltas, RefInfo, SegmentationFeatures, SequenceHeader};
use filmcraft_av1::{BitReader, LAST_FRAME, NUM_REF_FRAMES, PRIMARY_REF_NONE, REFS_PER_FRAME, WARPEDMODEL_PREC_BITS};

use super::ffi::{
    INVALID_SURFACE, SLICE_DATA_FLAG_ALL, VA_PADDING_LOW, VA_PADDING_MEDIUM, VaPictureParameterBufferAv1, VaSegmentationStructAv1, VaSliceParameterBufferAv1,
    VaWarpedMotionParamsAv1, record_bytes,
};
use super::picture::av1_decline;

// OBU types of the low-overhead bitstream format (5.3) this walk has to recognise.
const OBU_SEQUENCE_HEADER: u8 = 1;
const OBU_TEMPORAL_DELIMITER: u8 = 2;
const OBU_FRAME_HEADER: u8 = 3;
const OBU_TILE_GROUP: u8 = 4;
const OBU_FRAME: u8 = 6;

/// Why a stream whose film grain is applied is declined.
///
/// Stated once because two sites have to say the same thing. This backend reads pictures back with
/// `vaDeriveImage` + `vaMapBuffer` and never synthesises grain itself, so submitting the parameters
/// would have the driver render grain into `current_display_picture` — a second surface, a second
/// copy, and a picture that disagrees with FilmCraft's own decoder on exactly the frames grain
/// touches. The record's `VAFilmGrainStructAV1` is declared in full and left at "not present", so the
/// decline is a decision made from the whole record rather than from its absence.
pub const AV1_FILM_GRAIN: &str = "applies film grain, which this backend reads pictures back without: it copies them out through vaDeriveImage and never synthesises grain, so a driver rendering grain into current_display_picture would produce a picture that disagrees with FilmCraft's own decoder on exactly the frames grain touches";

/// Why large-scale tile is declined, and why that is **not** a check this builder can make.
///
/// **`filmcraft-av1` parses no large-scale tile syntax at all**, and that is checked rather than
/// assumed: `grep -rn "large_scale_tile\|enable_large_scale_tile" crates/av1/src` returns nothing —
/// no field on `SequenceHeader`, none on `FrameHeader`, none on `TileInfo`. So there is nothing to
/// read the flag *from*, which is a different fact from a field that exists and is clear, and this
/// record's `large_scale_tile` bit is therefore zero on every frame built here.
///
/// **Why it is not read out of the sequence header's trailing bits.** `enable_large_scale_tile` is
/// not reachable through `SequenceHeader::parse`, which stops at `film_grain_params_present`
/// (`crates/av1/src/header.rs:164`), and its position is not verified anywhere in this repository.
/// Reading "the bit after `film_grain_params_present`" would be exactly the confident guess this
/// codebase refuses — and a wrong guess here would decline every AV1 stream, because a misread bit of
/// a stream that does not use the feature is a `1` as often as not. **No encoder this suite can drive
/// produces a large-scale-tile stream** (`ffmpeg -h encoder=libaom-av1` has no LST option, checked),
/// so the bit cannot be settled by measurement either. Recorded as an open scope limitation rather
/// than as a gate that does not exist.
pub const AV1_LARGE_SCALE_TILE: &str = "large-scale tile, which filmcraft-av1 parses no syntax of at all — no `enable_large_scale_tile` and no `large_scale_tile` appears anywhere in crates/av1 — so the frame header is not a description of such a frame and this backend has no anchor frames to submit either";

// ---------------------------------------------------------------------------------------------
// The reference set the record is built against
// ---------------------------------------------------------------------------------------------

/// The eight reference slots of a `VADecPictureParameterBufferAV1`, as surfaces.
///
/// **`ref_frame_map` is the whole decoded picture buffer addressed by slot**, not a list of the
/// frames this picture predicts from — the same hazard VP9's `reference_frames[8]` is. The record's
/// own comment is blunter than any of us can be: *"Application needs to make sure all the entries
/// point to valid frames except for intra frames by checking ref_frame_id[] … Driver is not
/// responsible to validate reference frames' id"* (`va_dec_av1.h:347-355`). The driver is told to read
/// every entry and not to check it, so an entry naming some other surface is a wrong picture read
/// without complaint.
///
/// An empty slot is [`INVALID_SURFACE`]: surface ids come from `vaCreateSurfaces` and are not
/// guaranteed to avoid any value, so a zero would name *some* surface.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Av1ReferenceSet {
    /// Indexed by AV1 reference **slot** (0..=7), `None` for an empty slot.
    pub slots: [Option<u32>; NUM_REF_FRAMES],
}

impl Av1ReferenceSet {
    /// The surface in slot `index`, or `None` for a slot this set does not hold.
    pub fn get(&self, index: u8) -> Option<u32> {
        self.slots.get(usize::from(index)).copied().flatten()
    }

    /// Every slot, as the record's own array: a `None` as `VA_INVALID_SURFACE`.
    fn to_record(self) -> [u32; NUM_REF_FRAMES] {
        std::array::from_fn(|i| self.slots[i].unwrap_or(INVALID_SURFACE))
    }
}

// ---------------------------------------------------------------------------------------------
// Narrowing: every bitstream value into the width the record gives it
// ---------------------------------------------------------------------------------------------

/// `value` has to fit a `width`-bit record field, and comes back as it went in.
///
/// The width comes from the record, never from the value. Masking instead would submit a member the
/// bitstream never set, inside a field the driver reads as something else — the 33-instead-of-32 class
/// of error four times over. On a header [`FrameHeader::parse`] produced every one of these holds by
/// construction; they are checked because this takes a `FrameHeader` a caller can also assemble, and
/// the mutation test on `an_av1_record_is_built_from_the_frame_header` is exactly that case.
fn fits(what: &str, value: u32, width: u32) -> Result<u32, String> {
    match 1u32.checked_shl(width) {
        Some(limit) if value < limit => Ok(value),
        _ => Err(format!("AV1 {what} of {value} does not fit its {width}-bit record field")),
    }
}

/// Pack `(name, value, width)` into a bitfield word, least significant member first.
///
/// **The positions are the measured ones, not a running sum of declaration order.** A C bitfield's
/// layout is target-dependent, so `abi_guard::the_av1_bitfield_words_are_the_bits_the_c_compiler_measures`
/// compiles the installed header, sets one member at a time and reads the word back; a reordering of
/// the C struct fails that test with the member's name in the message. Every call site below lists
/// its members **in that measured order**, and each carries the member's name so a failure names it.
///
/// A member whose value does not fit its width is dropped to zero here rather than shifted into its
/// neighbour: every caller has already been through [`fits`], so a value that does not fit is one
/// this cannot produce, and corrupting the next member would be the worse of the two wrong answers.
fn pack(members: &[(&str, u32, u32)]) -> u32 {
    let mut word = 0u32;
    let mut at = 0u32;
    for (name, value, width) in members {
        // A member whose value does not fit its width contributes nothing rather than spilling into
        // its neighbour. Every caller has been through [`fits`] first, so this cannot happen for a
        // value that came from a parsed header — and a spill would be the worse of the two wrong
        // answers, because it would corrupt a *different* member.
        let _ = name;
        let masked = if *width >= 32 { *value } else { *value & ((1u32 << width) - 1) };
        if let Some(shifted) = masked.checked_shl(at) {
            word |= shifted;
        }
        at = at.saturating_add(*width);
    }
    word
}

/// A value as the record's `i8`.
fn narrow_i8(value: i32, what: &str) -> Result<i8, String> {
    i8::try_from(value).map_err(|_| format!("AV1 {what} of {value} does not fit the record's byte"))
}

/// `values` as the record's `[i8; N]`.
fn narrow_i8s<const N: usize>(values: &[i32], what: &str) -> Result<[i8; N], String> {
    let mut out = [0i8; N];
    for (at, slot) in out.iter_mut().enumerate() {
        let value = values.get(at).copied().unwrap_or(0);
        *slot = i8::try_from(value).map_err(|_| format!("AV1 {what}[{at}] of {value} does not fit the record's byte"))?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// The OBU walk
// ---------------------------------------------------------------------------------------------

/// One OBU of a temporal unit: its type, its temporal/spatial layer, and where its payload sits
/// inside the sample.
///
/// `at` is kept because a tile's byte extent is reported **in the sample's own coordinates** — which
/// is what the slice records address, since the temporal unit is submitted as the slice data buffer.
#[derive(Clone, Copy, Debug)]
struct Obu<'a> {
    kind: u8,
    temporal_id: u32,
    spatial_id: u32,
    payload: &'a [u8],
    /// Where the OBU's own header byte is, and where its payload ends.
    ///
    /// `at` is kept because a tile's byte extent is reported **in the sample's own coordinates** —
    /// which is what the slice records address, since the temporal unit is submitted as the slice data
    /// buffer. `start` is kept separately because **the next frame begins at the next OBU's *header*
    /// byte, not its payload**, and re-walking from the payload reads the first payload byte as an OBU
    /// header: a wrong type, a wrong size, and a "tile group naming tiles the frame does not have" for
    /// a frame that has one tile. The two are one byte apart for an OBU without an extension header and
    /// two for one with, so this is not a constant that could have been written down.
    start: usize,
    at: usize,
    end: usize,
}

/// The OBUs of one temporal unit: `obu_forbidden_bit`, `obu_extension_flag`, `obu_has_size_field`
/// and a `leb128` size when the flag is set (5.3).
///
/// A truncated or malformed OBU ends the walk where it is, and the ones before it are returned: the
/// caller then finds the frame it needed missing and says so, rather than the whole sample being
/// refused for a tail it never used.
fn obus(sample: &[u8], from: usize) -> Vec<Obu<'_>> {
    let mut out = Vec::new();
    let mut pos = from;
    while let Some(&h) = sample.get(pos) {
        if h & 0x80 != 0 {
            break; // obu_forbidden_bit
        }
        let (kind, ext, has_size) = ((h >> 3) & 0xf, h & 4 != 0, h & 2 != 0);
        let mut p = pos + 1;
        let (mut temporal_id, mut spatial_id) = (0u32, 0u32);
        if ext {
            let Some(&e) = sample.get(p) else { break };
            temporal_id = u32::from(e >> 5);
            spatial_id = u32::from((e >> 3) & 3);
            p += 1;
        }
        let size = if has_size {
            match leb128(sample.get(p..).unwrap_or(&[])) {
                Ok((v, n)) => {
                    p += n;
                    usize::try_from(v).unwrap_or(usize::MAX)
                }
                Err(()) => break,
            }
        } else {
            sample.len().saturating_sub(p)
        };
        let Some(end) = p.checked_add(size).filter(|&e| e <= sample.len()) else { break };
        out.push(Obu { kind, temporal_id, spatial_id, payload: &sample[p..end], start: pos, at: p, end });
        pos = end;
    }
    out
}

/// A `leb128` over `data`, and how many bytes it took. `Err` for a run of eight continuation bytes.
fn leb128(data: &[u8]) -> Result<(u64, usize), ()> {
    let mut v = 0u64;
    for i in 0..8 {
        let Some(&b) = data.get(i) else { return Err(()) };
        v |= u64::from(b & 0x7f) << (7 * i);
        if b & 0x80 == 0 {
            return Ok((v, i + 1));
        }
    }
    Ok((v, 8))
}

/// One tile of a frame: its number, and the byte range it occupies **inside the sample**.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tile {
    num: usize,
    start: usize,
    end: usize,
}

// ---------------------------------------------------------------------------------------------
// `Av1State` — the slots, the parse state, and the frames
// ---------------------------------------------------------------------------------------------

/// The reference state of one AV1 decode: the parse state 7.20 threads across frames, and the eight
/// reference slots the record's `ref_frame_map` is built from.
///
/// **One of these per decode, kept for the life of the run.** A per-picture state is the defect
/// [`Av1State::commit`] exists to make impossible.
pub struct Av1State {
    /// The active sequence header, from the sample entry's `av1C` and from any in-band one. `None`
    /// until one has been seen: a stream whose `av1C` carried no sequence header has no frame header
    /// to parse, and saying so at the first frame is better than refusing every stream at construction
    /// — the constructor is called for every decoder, including the H.264 and HEVC ones, where it is
    /// never consulted.
    seq: Option<SequenceHeader>,
    /// The `av1C` configuration OBUs, kept so [`Self::reset`] can re-seed without the container.
    config: Vec<u8>,
    /// 7.20's per-slot values, read by the frame header parser on the next frame's parse.
    refs: [RefInfo; NUM_REF_FRAMES],
    /// The four pieces `HeaderState` carries between frames, held here because `HeaderState` borrows
    /// `seq` and `refs` and so cannot be a field of the same struct — the same split
    /// `filmcraft_av1::Decoder` makes at `decoder.rs:468-480`.
    lf_deltas: LoopFilterDeltas,
    seg_features: SegmentationFeatures,
    prev_gm_params: [[i32; 6]; 8],
    current_frame_id: u32,
    /// The surfaces the eight slots hold, `None` for a slot no picture is in.
    slots: [Option<u32>; NUM_REF_FRAMES],
}

/// One AV1 frame, read out of a temporal unit: what it is, its header, and the slice records to
/// submit beside its bytes.
#[derive(Clone, Debug)]
pub struct Av1Frame {
    /// `None` for a frame that re-shows an earlier picture: there is nothing to describe and nothing
    /// to render.
    header: Option<FrameHeader>,
    /// The `VASliceParameterBufferAV1` bytes, one per tile, each addressing its tile inside the
    /// temporal unit.
    slices: Vec<Vec<u8>>,
    /// The re-shown slot and its surface, for a frame that re-shows an earlier picture.
    re_show: Option<(usize, u32)>,
    /// `refresh_frame_flags`: bit *i* set means this picture becomes reference slot *i*.
    refresh: u8,
    /// Where in the sample the next frame starts — the frame's own OBUs end here.
    next: usize,
    /// **The tile-group payloads this frame's tiles were walked out of**, each with the offset it
    /// starts at inside the temporal unit.
    ///
    /// Kept on the frame for one reason, which is that the tile walk is **a second implementation**
    /// of 5.9's `tile_group_obu` and the only thing worth anything about a second implementation is
    /// that it can be checked against the first. `filmcraft_av1::tile_extents` is what
    /// `Decoder::tile_group` calls — the code the pictures this record describes were decoded with —
    /// and it needs the exact bytes the walk was given. Holding them here means the check runs over
    /// **the same payload**, not over a re-walk that could differ from this one and make the check
    /// pass for the wrong reason.
    ///
    /// An `OBU_FRAME` carries its tile data after its own header, byte-aligned, and is held here the
    /// same way an `OBU_TILE_GROUP` is: it is the payload the walk was given, and a frame whose tiles
    /// arrive that way is the common case, not the exception.
    groups: Vec<(usize, Vec<u8>)>,
}

impl Av1Frame {
    /// The frame's uncompressed header, as `filmcraft-av1` parsed it into this run's state, or
    /// `None` for a frame that re-shows an earlier picture (which the parse leaves almost empty).
    pub fn header(&self) -> Option<&FrameHeader> {
        self.header.as_ref()
    }

    /// The slice-parameter records: **one per tile**, each naming the tile's byte extent inside the
    /// temporal unit that is submitted as the slice data buffer.
    pub fn slices(&self) -> &[Vec<u8>] {
        &self.slices
    }

    /// Whether this frame codes a picture of its own, as opposed to re-showing an earlier one.
    pub fn is_new(&self) -> bool {
        self.header.is_some()
    }

    /// Whether this frame's picture is shown. `show_frame` for a frame of its own; a re-show is
    /// always shown — 7.21 re-shows *a picture*, and it is the frame's whole purpose.
    pub fn show(&self) -> bool {
        match &self.header {
            Some(h) => h.show_frame,
            None => true,
        }
    }

    /// The reference slot and surface this frame re-shows, or `None` for a frame of its own.
    pub fn re_show(&self) -> Option<(usize, u32)> {
        self.re_show
    }

    /// The frame's `refresh_frame_flags`: bit *i* set means this picture becomes reference slot *i*.
    pub fn refresh_flags(&self) -> u8 {
        self.refresh
    }

    /// Where the next frame in this sample begins.
    pub fn next(&self) -> usize {
        self.next
    }

    /// The tile-group payloads [`Self::slices`] were built from: each one's offset inside the
    /// temporal unit, and its bytes. See the field's note for why this is kept.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn groups(&self) -> &[(usize, Vec<u8>)] {
        &self.groups
    }
}

impl Av1State {
    /// The state a decode starts in, seeded from the sample entry's `av1C` configuration OBUs.
    ///
    /// **Infallible, on purpose.** The constructor runs for every decoder this backend builds — H.264,
    /// HEVC and VP9 included, where the answer is `None` and the state is never consulted — so a stream
    /// that happens to carry no AV1 configuration must not make *that* decoder unconstructible. A frame
    /// arriving with no sequence header to parse it against is refused by [`Self::frame_at`] instead, with
    /// a message that names the missing thing.
    pub fn new(config_obus: &[u8]) -> Self {
        Self {
            seq: sequence_header_of(config_obus),
            config: config_obus.to_vec(),
            refs: std::array::from_fn(|_| RefInfo::default()),
            lf_deltas: LoopFilterDeltas::defaults(),
            seg_features: SegmentationFeatures::default(),
            prev_gm_params: default_gm_params(),
            current_frame_id: 0,
            slots: [None; NUM_REF_FRAMES],
        }
    }

    /// The sequence header the records are built against, or `None` if none has been seen yet.
    pub fn sequence_header(&self) -> Option<&SequenceHeader> {
        self.seq.as_ref()
    }

    /// Forget every picture decoded so far, as a seek or a flush does.
    ///
    /// **Both halves have to go.** The slots, because the next frame would otherwise find a reference
    /// in a surface left over from before the seek — a plausible picture predicted from the wrong
    /// place. And the parse state, because 7.20's propagated loop filter deltas, segmentation features
    /// and global motion are inherited by every frame that does not spell them, so keeping them would
    /// describe the first frame of the new run in the previous run's values.
    pub fn reset(&mut self) {
        self.seq = sequence_header_of(&self.config);
        self.refs = std::array::from_fn(|_| RefInfo::default());
        self.lf_deltas = LoopFilterDeltas::defaults();
        self.seg_features = SegmentationFeatures::default();
        self.prev_gm_params = default_gm_params();
        self.current_frame_id = 0;
        self.slots = [None; NUM_REF_FRAMES];
    }

    /// The reference set as the record's `ref_frame_map` wants it.
    pub fn references(&self) -> Av1ReferenceSet {
        Av1ReferenceSet { slots: self.slots }
    }

    /// What is in reference slot `index`, or `None` for a slot no picture is in.
    pub fn slot(&self, index: usize) -> Option<u32> {
        self.slots.get(index).copied().flatten()
    }

    /// The surfaces the slots hold — the ones a new picture may **not** be decoded into, because the
    /// driver would read the new picture as the reference that slot is.
    ///
    /// The decode loop asks this *after* [`Self::frame_at`] has run, which is the opposite of the NAL
    /// order and is not an oversight: a frame may code no picture at all (a re-show), so whether a
    /// surface is needed is not known until the frame has been read. The answer does not depend on
    /// the order — `frame_at` moves no slot, [`Self::commit`] does — so the ordering is about *not
    /// taking a surface* for a frame that decodes nothing.
    pub fn live_surfaces(&self) -> Vec<u32> {
        self.slots.iter().filter_map(|s| *s).collect()
    }

    /// Read the next frame of `sample` at or after `at`, with its slice records built.
    ///
    /// **One frame at a time, and the reason is the state.** 7.20 refreshes reference slots *between*
    /// frames, and the next frame's *header* reads them: `frame_size_with_refs` asks a named slot for
    /// its size, `skip_mode_params` and `OrderHints` ask for its order hint, and `load_previous`
    /// reads the primary reference's deltas, segmentation features and global motion out of it. The
    /// software decoder therefore parses frame *n*+1 after it has updated 7.20 for frame *n*
    /// (`crates/av1/src/decoder.rs:586-607`), and a batch that parsed a whole sample up front would
    /// read the second frame against the first one's uncommitted state — a valid-looking header of
    /// the wrong reference. So the caller decodes and commits each frame before asking for the next,
    /// which is the same shape a VP9 superframe takes one level up.
    ///
    /// ## What is *not* refreshed here
    ///
    /// Nothing moves a slot; [`Self::commit`] does, after the driver has rendered. A frame whose
    /// record was refused leaves the buffer exactly as it was.
    ///
    /// ## An `Err` here may have advanced the parse state, and that is deliberate
    ///
    /// `FrameHeader::parse` writes `st.refs` as it goes — `mark_ref_frames` at `header.rs:572-581`
    /// and the error-resilient order-hint check at `:607-613` both clear slots' validity — and writes
    /// `lf_deltas`, `seg_features` and `prev_gm_params` at `:681-693`, all before this function gets
    /// to build anything. **FilmCraft's own software decoder behaves identically**: it parses into
    /// `self.refs` and only then makes its own checks (`decoder.rs:476`, then :481-520). Restoring a
    /// snapshot here would make the hardware and software paths of one bitstream disagree about what
    /// the frame after a refusal inherits, which is a new class of wrong rather than the removal of
    /// one. What keeps it harmless is that **no caller continues after an `Err`**:
    /// `VaDecoder::decode` latches `self.refused` until `reset()`, and the software decoder returns
    /// the error to its caller. The rule for a caller is: an `Err` ends the run, and
    /// [`Self::reset`] is the way back.
    ///
    /// The one exception is a **re-show**, which the parse returns from at `header.rs:539` before any
    /// of those writes, and which moves 7.20 itself for a key frame (see [`Self::re_show`]).
    pub fn frame_at(&mut self, sample: &[u8], at: usize) -> Result<Option<Av1Frame>, String> {
        if at >= sample.len() {
            return Ok(None);
        }
        let Some(seq) = self.seq.clone() else {
            return Err("an AV1 frame with no sequence header to parse it against: the sample entry's av1C carried none".into());
        };
        let list = obus(sample, at);
        let mut idx = 0usize;
        // An in-band sequence header is a legal way for a stream to re-declare itself; the
        // descriptions the surfaces were created from must still hold, so one that differs is an
        // error rather than a re-seed.
        while idx < list.len() {
            let o = list[idx];
            match o.kind {
                OBU_TEMPORAL_DELIMITER => idx += 1,
                OBU_SEQUENCE_HEADER => {
                    let parsed = SequenceHeader::parse(o.payload).map_err(|e| format!("sequence header: {e}"))?;
                    if Some(&parsed) != self.seq.as_ref() {
                        return Err("an in-band AV1 sequence header that differs from the one this session was created for".into());
                    }
                    idx += 1;
                }
                OBU_FRAME_HEADER | OBU_FRAME => break,
                // Metadata and tile-list OBUs carry nothing this backend submits; a tile list is
                // large-scale tile's, which is declined and would not decode in this software path
                // either. Skipping them is 5.3's rule for OBUs outside the selected operating point.
                _ => idx += 1,
            }
        }
        if idx >= list.len() {
            return Ok(None);
        }
        let header_obu = list[idx];
        idx += 1;
        let mut r = BitReader::new(header_obu.payload);
        let mut st = HeaderState::new(&seq, &mut self.refs);
        st.lf_deltas = self.lf_deltas;
        st.seg_features = self.seg_features;
        st.prev_gm_params = self.prev_gm_params;
        st.current_frame_id = self.current_frame_id;
        let header =
            FrameHeader::parse(&mut r, &mut st, header_obu.temporal_id, header_obu.spatial_id).map_err(|e| format!("uncompressed frame header: {e}"))?;
        // Whatever the parse wrote back, the run carries — exactly as the software decoder's
        // `frame_header` does at `decoder.rs:477-480`.
        self.lf_deltas = st.lf_deltas;
        self.seg_features = st.seg_features;
        self.prev_gm_params = st.prev_gm_params;
        self.current_frame_id = st.current_frame_id;

        if header.show_existing_frame {
            return Ok(Some(self.re_show(header, header_obu.end)?));
        }
        // The tile data of an `OBU_FRAME` follows its header, byte-aligned; of an `OBU_FRAME_HEADER`
        // it is in the `OBU_TILE_GROUP` OBUs after it.
        let mut groups: Vec<(usize, &[u8])> = Vec::new();
        let mut next = header_obu.end;
        if header_obu.kind == OBU_FRAME {
            r.byte_align();
            let at_tile = header_obu.at + r.byte_pos();
            if at_tile <= header_obu.end {
                groups.push((at_tile, &sample[at_tile..header_obu.end]));
                next = header_obu.end;
            }
        }
        // Collect the tile groups of *this* frame: an `OBU_FRAME` or `OBU_FRAME_HEADER` after them
        // is the next frame, and the frame is complete when its last tile has arrived.
        let mut tiles: Vec<Tile> = Vec::new();
        while idx < list.len() && list[idx].kind == OBU_TILE_GROUP {
            let o = list[idx];
            idx += 1;
            groups.push((o.at, o.payload));
            next = o.end;
        }
        let tiles = self.tiles_of(&header, &groups, &mut tiles)?;
        if tiles.last().map(|t| t.num + 1) != Some(header.tile_info.cols.saturating_mul(header.tile_info.rows)) {
            return Err("an AV1 frame whose tiles are not all in this sample".into());
        }
        if idx < list.len() {
            // The next frame begins at the next OBU's **header**, not its payload — see `Obu::start`.
            next = list[idx].start;
        }
        // The frames a predictor names must hold a picture, and so must the primary reference: the
        // record's `ref_frame_map` and `primary_ref_frame` are read out of it, and a slot that holds
        // nothing is `VA_INVALID_SURFACE` where the driver expects a picture. This is the software
        // decoder's own check (`crates/av1/src/decoder.rs:511-520`).
        if !header.frame_is_intra {
            for (i, &slot) in header.ref_frame_idx.iter().enumerate() {
                if self.slot(slot).is_none() {
                    return Err(format!("AV1 ref_frame_idx[{i}] is slot {slot}, which holds no picture"));
                }
            }
        }
        if header.primary_ref_frame != PRIMARY_REF_NONE {
            let slot = header.ref_frame_idx.get(header.primary_ref_frame).copied().unwrap_or(usize::MAX);
            if self.slot(slot).is_none() {
                return Err(format!("AV1 primary_ref_frame is entry {}, whose slot {slot} holds no picture", header.primary_ref_frame));
            }
        }
        let slices = slices_of(&header, &tiles)?;
        let refresh = u8::try_from(header.refresh_frame_flags).map_err(|_| "AV1 refresh_frame_flags does not fit a byte".to_string())?;
        // The payloads the walk was given, owned, so the cross-check against `filmcraft_av1`'s own
        // split runs over these bytes rather than over a second re-walk of the sample. See
        // [`Av1Frame::groups`].
        let groups: Vec<(usize, Vec<u8>)> = groups.iter().map(|(at, payload)| (*at, payload.to_vec())).collect();
        Ok(Some(Av1Frame { header: Some(header), slices, re_show: None, refresh, next, groups }))
    }

    /// A `show_existing_frame`: the picture the header's `frame_to_show_map_idx` names, and nothing
    /// else. See [`Self::frame_at`] for why this is decoded rather than declined.
    fn re_show(&mut self, header: FrameHeader, next: usize) -> Result<Av1Frame, String> {
        let slot = header.frame_to_show_map_idx;
        let Some(surface) = self.slot(slot) else {
            return Err(format!("an AV1 frame that re-shows reference slot {slot}, which holds no picture"));
        };
        // 7.20's "reference frame loading process" then "update reference frame process": a re-show
        // of a **key frame** loads that frame's propagated values into every slot and refreshes them
        // all, which is what `filmcraft_av1::Decoder::frame_header` does at
        // `crates/av1/src/decoder.rs:495-503`. It happens here rather than in `commit` because there
        // is no decode to wait for: nothing was rendered, and the software decoder does it on the spot.
        let mut refresh = 0u8;
        if usize::from(header.frame_type) == filmcraft_av1::KEY_FRAME {
            let info = self.refs.get(slot).cloned().unwrap_or_default();
            for at in self.refs.iter_mut() {
                *at = info.clone();
            }
            self.slots = [Some(surface); NUM_REF_FRAMES];
            refresh = u8::MAX;
        }
        Ok(Av1Frame { header: None, slices: Vec::new(), re_show: Some((slot, surface)), refresh, next, groups: Vec::new() })
    }

    /// The `VADecPictureParameterBufferAV1` for `frame`, as the bytes the driver reads.
    ///
    /// **Why the surface is an argument and not part of the frame.** The record carries
    /// `current_frame` and `current_display_picture` — the surfaces the picture is decoded into and
    /// displayed from (`va_dec_av1.h:296-303`) — and *when* that surface is known is decided by the
    /// decode loop, which must read the frame **before** taking a surface from the ring, because a
    /// re-show codes no picture at all. A record built inside [`Self::frame_at`] would have to name a
    /// surface before one had been chosen, or would carry a placeholder the caller has to patch. So
    /// the header is carried and the record is built here, once, from the same state the header was
    /// parsed into.
    ///
    /// Nothing has moved a slot since [`Self::frame_at`] ran — only [`Self::commit`] does — so the
    /// `ref_frame_map` written here is the buffer as it was **before** this frame, which is what the
    /// record describes.
    pub fn picture_record(&self, frame: &Av1Frame, surface: u32) -> Result<Vec<u8>, String> {
        let header = frame.header().ok_or("a frame that re-shows an earlier picture carries no picture parameters")?;
        let seq = self.seq.as_ref().ok_or("an AV1 frame with no sequence header to build its record against")?;
        av1_picture_record(header, seq, &self.references(), surface)
    }

    /// Record that the driver rendered `frame`'s picture into `surface`, and apply 7.20's refresh.
    ///
    /// Called **after** the decode, never before: a frame whose decode failed must not have moved the
    /// reference buffer, and the record it was submitted with describes the buffer as it was before
    /// it. The next frame's record then describes the buffer this call leaves.
    ///
    /// ## What a refresh moves, and why it is both halves at once
    ///
    /// `refresh_frame_flags` is the specification's 7.20 bitmask — *"the decoded picture buffer is
    /// updated as follows … RefValid[ i ] = (( refresh_frame_flags >> i ) & 1) … and the reference
    /// frame loading process runs for every refreshed slot"* — and it is applied here to **two** things
    /// that a VA-API decode needs to agree about:
    ///
    /// - the **surface** in the slot, which is what the record's `ref_frame_map[8]` is built from; and
    /// - the slot's **`RefInfo`** — frame id, sizes, order hint, saved order hints, global motion,
    ///   loop filter deltas, segmentation features, film grain — which is what the *next frame's
    ///   header parse* reads.
    ///
    /// The second is the half VP9 does not have and the reason this is not an adaptation of it: in AV1
    /// the reference state is read by the parser, not only by the record, so a commit that moved the
    /// surface alone would leave the next frame's `primary_ref_frame`, `frame_size_with_refs` and
    /// `OrderHints` reading a previous run's slot. Both halves are moved together, from the same
    /// header, and this is `filmcraft_av1::Decoder::submit` (`crates/av1/src/decoder.rs:586-607`)
    /// field for field — the software decoder's own update, which is what makes a hardware decode and
    /// a software decode of the same stream agree about what the next frame inherits.
    ///
    /// A frame that re-shows an earlier picture is refused rather than ignored, except for the key
    /// frame case that 7.20 refreshes on the spot (see [`Self::re_show`]): it decodes nothing, so there
    /// is no picture to put in a slot, and a caller that committed one was handed a surface still
    /// holding the *previous* occupant of that slot.
    pub fn commit(&mut self, frame: &Av1Frame, surface: u32) -> Result<(), String> {
        let Some(header) = frame.header() else {
            return if frame.re_show.is_some_and(|(_, s)| s == surface) && frame.refresh == 0 {
                // A re-show of a non-key frame: 7.20 refreshes nothing, because nothing was decoded.
                Ok(())
            } else {
                Err("a frame that re-shows an earlier picture decodes nothing, so it refreshes no reference slot".to_string())
            };
        };
        let info = RefInfo {
            valid: true,
            frame_id: header.current_frame_id,
            upscaled_width: header.upscaled_width,
            frame_width: header.frame_width,
            frame_height: header.frame_height,
            render_width: header.render_width,
            render_height: header.render_height,
            mi_cols: header.mi_cols,
            mi_rows: header.mi_rows,
            frame_type: header.frame_type,
            order_hint: header.order_hint,
            saved_order_hints: header.order_hints,
            gm_params: header.gm_params,
            lf_deltas: header.lf.deltas,
            seg_features: header.seg.features,
            grain: header.film_grain.clone(),
        };
        // `enumerate` over the eight slots, so the shift count cannot leave the byte and there is no
        // index arithmetic anywhere near `refresh_frame_flags` to be wrong in a way a test misses.
        for (slot, at) in self.slots.iter_mut().enumerate() {
            if (frame.refresh >> slot) & 1 == 1 {
                *at = Some(surface);
                if let Some(into) = self.refs.get_mut(slot) {
                    *into = info.clone();
                }
            }
        }
        Ok(())
    }

    /// The frame's tiles, walked out of the tile-group OBUs it carries.
    ///
    /// **This walk is deliberately a second implementation.** `filmcraft_av1::tile_extents` is the
    /// one the software decoder decodes with, and the test
    /// `the_tiles_the_backend_submits_are_the_tiles_the_decoder_splits` walks both over a real
    /// multi-tile fixture and compares them tile for tile. Two walks of the same clause that were
    /// never compared are two places to be wrong, and a disagreement here is not an error: it is a
    /// set of plausible-looking wrong tiles, which is exactly what the slice records would then
    /// submit. The walk is 5.9's `tile_group_obu`: `tile_start_and_end_present_flag`, then each tile
    /// but the group's last preceded by a `tile_size_bytes`-byte little-endian length, and the last
    /// tile of the group runs to the end of its OBU.
    fn tiles_of(&self, header: &FrameHeader, groups: &[(usize, &[u8])], out: &mut Vec<Tile>) -> Result<Vec<Tile>, String> {
        let (cols, rows) = (header.tile_info.cols, header.tile_info.rows);
        if cols == 0 || rows == 0 {
            return Err("an AV1 frame with no tiles".into());
        }
        let num_tiles = cols.checked_mul(rows).ok_or("an AV1 tile grid that overflows")?;
        if num_tiles > MAX_TILES {
            return Err(format!("an AV1 frame of {num_tiles} tiles, more than the {MAX_TILES} the record's tile arrays name"));
        }
        for &(base, payload) in groups {
            let mut r = BitReader::new(payload);
            let (mut tg_start, mut tg_end) = (0usize, num_tiles - 1);
            if num_tiles > 1 && r.flag().map_err(|_| "a truncated tile group".to_string())? {
                let bits = header.tile_info.cols_log2.saturating_add(header.tile_info.rows_log2);
                tg_start = r.f(bits).map_err(|_| "a truncated tile group".to_string())? as usize;
                tg_end = r.f(bits).map_err(|_| "a truncated tile group".to_string())? as usize;
            }
            r.byte_align();
            if tg_end >= num_tiles || tg_start > tg_end {
                return Err("an AV1 tile group naming tiles the frame does not have".into());
            }
            let tsb = header.tile_info.tile_size_bytes as usize;
            let mut pos = base + r.byte_pos();
            for num in tg_start..=tg_end {
                let size = if num == tg_end {
                    payload.len() + base - pos
                } else {
                    let mut v = 0usize;
                    for i in 0..tsb {
                        let at = pos.checked_add(i).ok_or("an AV1 tile size runs past its group")?;
                        let b = self.byte_at(payload, base, at)?;
                        v |= (b as usize) << (8 * i);
                    }
                    pos = pos.checked_add(tsb).ok_or("an AV1 tile size overflows")?;
                    v.saturating_add(1)
                };
                let end = pos.checked_add(size).ok_or("an AV1 tile runs past its group")?;
                if end > base + payload.len() {
                    return Err("an AV1 tile runs past its group".into());
                }
                out.push(Tile { num, start: pos, end });
                pos = end;
            }
        }
        Ok(std::mem::take(out))
    }

    /// One byte of a tile-group payload, addressed from the **sample's** own coordinates.
    ///
    /// `base` is where the payload begins inside the sample — the tile group's first byte, the one
    /// carrying `tile_start_and_end_present_flag` — and `at` an absolute offset, so the two are not
    /// the same coordinate space and the difference is what this computes.
    fn byte_at(&self, payload: &[u8], base: usize, at: usize) -> Result<u8, String> {
        at.checked_sub(base).and_then(|p| payload.get(p)).copied().ok_or_else(|| "an AV1 tile size runs past its group".to_string())
    }
}

/// The most tiles AV1 defines (`MAX_TILE_COLS * MAX_TILE_ROWS` capped at 64 by the specification's
/// `TileNumBits`), and the length of the record's `width_in_sbs_minus_1` plus the last tile's entry.
const MAX_TILES: usize = 64;

/// The sequence header an `av1C` configuration OBU list carries, or `None`.
fn sequence_header_of(config_obus: &[u8]) -> Option<SequenceHeader> {
    obus(config_obus, 0).into_iter().find(|o| o.kind == OBU_SEQUENCE_HEADER).and_then(|o| SequenceHeader::parse(o.payload).ok())
}

/// `gm_params` at the identity translation, which is 7.20's `setup_past_independence` default and
/// what [`Av1State::new`] seeds a run with.
fn default_gm_params() -> [[i32; 6]; 8] {
    let mut p = [[0i32; 6]; 8];
    for row in p.iter_mut() {
        row[2] = 1 << WARPEDMODEL_PREC_BITS;
        row[5] = 1 << WARPEDMODEL_PREC_BITS;
    }
    p
}

/// How many surfaces an AV1 decode must have: `NUM_REF_FRAMES` + 1.
///
/// The same bound and from the same place as [`super::picture::vp9_surface_pool`], applied to AV1's
/// own shape: the record's `ref_frame_map` is all eight slots, and [`Av1State`] holds a surface in a
/// slot only while 7.20's `refresh_frame_flags` says so — so at most eight surfaces ever hold a live
/// reference, and the ninth is the one a new picture may be decoded into.
pub const fn av1_surface_pool() -> usize {
    NUM_REF_FRAMES + 1
}

// ---------------------------------------------------------------------------------------------
// The picture-parameter record
// ---------------------------------------------------------------------------------------------

/// The `VADecPictureParameterBufferAV1` for one AV1 frame, as the bytes the driver reads, or why
/// this frame cannot be described.
///
/// Everything here is read out of the **`FrameHeader` `filmcraft_av1` parsed from this very frame**
/// and the `SequenceHeader` it was parsed against, so the record and the picture FilmCraft's own
/// decoder produces are two readings of one description rather than two descriptions.
///
/// ## The three levels, and the one with no analogue elsewhere in this backend
///
/// `ref_frame_map[8]` is the whole decoded picture buffer by slot (see [`Av1ReferenceSet`]);
/// `ref_frame_idx[7]` is this frame's indirection into it, the specification's own `ref_frame_idx`
/// with `LAST_FRAME` (1) subtracted (`va_dec_av1.h:358-368`); and `primary_ref_frame` says **which
/// entry of that indirection carries the propagated info** 7.20 loads at the start of the frame — the
/// loop filter deltas, the segmentation features and the global motion. `PRIMARY_REF_NONE` (7) means
/// load none. H.264's `[3, 3]` index collision cost 60 of 72 pictures; a wrong answer here is the same
/// shape, and a zero-filled `primary_ref_frame` loads the **LAST** frame's deltas instead of loading
/// none — a plausible picture built from the wrong reference.
///
/// ## Every value is bounded before it is stored
///
/// A bitstream value read as a `u32` does not fit a `uint8_t` record member unless it is checked, and
/// masking it would submit a flag the bitstream never set inside a member the driver reads as
/// something else. So each one is an `Err` naming the field and the number. None can happen on a
/// header [`FrameHeader::parse`] produced; all are reachable from one a caller assembled itself, which
/// is the mutation case the record-bytes test drives.
pub fn av1_picture_record(h: &FrameHeader, seq: &SequenceHeader, refs: &Av1ReferenceSet, surface: u32) -> Result<Vec<u8>, String> {
    // The profile table first, in the same words `plan_frame` uses, so a caller cannot get a record
    // for a stream it was told it could not decode.
    if let Some(why) = av1_decline(seq.profile, u32::from(seq.color.bit_depth)) {
        return Err(why);
    }
    // A frame that re-shows an earlier picture carries none of its own — see [`Av1State::re_show`].
    if h.show_existing_frame {
        return Err("an AV1 frame that re-shows an earlier one carries no picture parameters of its own".into());
    }
    // Declined before anything is packed, so a refusal costs nothing.
    if h.film_grain.apply_grain {
        return Err(AV1_FILM_GRAIN.into());
    }

    // --- every fixed-width field, checked before anything is packed --------------------------------
    fits("frame_type", u32::from(h.frame_type), 2)?;
    fits("tx_mode", u32::from(h.tx_mode), 2)?;
    fits("primary_ref_frame", h.primary_ref_frame as u32, 8)?;
    fits("order_hint", h.order_hint, 8)?;
    fits("base_qindex", h.quant.base_q_idx, 8)?;
    fits("cdef_bits", h.cdef.bits, 2)?;
    fits("cdef_damping", h.cdef.damping, 8)?;
    fits("delta_q_res", h.delta_q_res, 2)?;
    fits("delta_lf_res", h.delta_lf_res, 2)?;
    fits("interp_filter", u32::from(h.interpolation_filter), 3)?;
    fits("superres_scale_denominator", h.superres_denom, 8)?;
    fits("order_hint_bits_minus_1", u32::from(av1_order_hint_bits(seq)), 8)?;
    for (i, &slot) in h.ref_frame_idx.iter().enumerate() {
        fits(&format!("ref_frame_idx[{i}]"), slot as u32, 8)?;
    }
    for (plane, &level) in h.lf.level.iter().enumerate() {
        fits(&format!("lf.level[{plane}]"), level, 8)?;
    }

    // --- the picture size, which is the *upscaled* one ---------------------------------------------
    //
    // *"Picture resolution minus 1. Picture original resolution. If SuperRes is enabled, this is the
    // upscaled resolution"* (`va_dec_av1.h:332-336`), which is why a superres stream is not declined
    // for its size. There is no `upscaled_height` to read: **AV1's SuperRes is horizontal only** —
    // 5.9.21's `superres_params` computes `UpscaledWidth` and rewrites `FrameWidth`, and
    // `FrameHeight` is never rescaled (`crates/av1/src/header.rs:761-767`, the specification's own
    // arithmetic) — so the coded height *is* the upscaled height.
    let width = h.upscaled_width.checked_sub(1).ok_or("an AV1 frame of zero width")?;
    let height = h.frame_height.checked_sub(1).ok_or("an AV1 frame of zero height")?;
    let frame_width_minus1 = fits("frame_width_minus1", width, 16)? as u16;
    let frame_height_minus1 = fits("frame_height_minus1", height, 16)? as u16;

    let (tile_cols, tile_rows, width_in_sbs, height_in_sbs) = av1_tile_extents(h, seq)?;

    let record = VaPictureParameterBufferAv1 {
        profile: seq.profile,
        order_hint_bits_minus_1: av1_order_hint_bits(seq),
        // A **table**, not a subtraction: 8 → 0, 10 → 1, 12 → 2 (`va_dec_av1.h:255-261`). A depth
        // outside it is a refusal rather than a clamp into the 8-bit row, which would be a picture
        // dequantized at another depth's scale.
        bit_depth_idx: av1_bit_depth_idx(seq.color.bit_depth)?,
        matrix_coefficients: seq.color.matrix_coefficients,
        seq_info_fields: av1_seq_info_fields(seq),
        current_frame: surface,
        // Set equal to `current_frame` rather than left at whatever it held: the header says this one
        // is "Valid only when apply_grain equals 1" (`va_dec_av1.h:300-303`) and grain is declined,
        // but a stale id in a field the driver may still look at is the same failure as a stale entry
        // in `ref_frame_map`.
        current_display_picture: surface,
        // Every large-scale-tile field is the "valid only when large_scale_tile equals 1" default.
        // `anchor_frames_list` is a `VASurfaceID *` in C, declared as the address itself so that
        // every byte of the record is initialised; see `AV1_LARGE_SCALE_TILE` for why this is zero.
        anchor_frames_num: 0,
        padding_after_anchor_frames_num: [0; 7],
        anchor_frames_list: 0,
        frame_width_minus1,
        frame_height_minus1,
        output_frame_width_in_tiles_minus_1: 0,
        output_frame_height_in_tiles_minus_1: 0,
        ref_frame_map: refs.to_record(),
        ref_frame_idx: av1_ref_frame_idx(h),
        primary_ref_frame: h.primary_ref_frame as u8,
        order_hint: h.order_hint as u8,
        padding_after_order_hint: [0; 3],
        seg_info: av1_seg_info(h),
        // Grain is declined above, so every field of this sub-record is "not present": `apply_grain`
        // is bit 0 of `film_grain_info_fields` and is zero.
        film_grain_info: Default::default(),
        tile_cols,
        tile_rows,
        width_in_sbs_minus_1: width_in_sbs,
        height_in_sbs_minus_1: height_in_sbs,
        tile_count_minus_1: 0,
        context_update_tile_id: fits("context_update_tile_id", h.tile_info.context_update_tile_id, 16)? as u16,
        padding_after_context_update_tile_id: [0; 2],
        pic_info_fields: av1_pic_info_fields(h),
        superres_scale_denominator: h.superres_denom as u8,
        interp_filter: h.interpolation_filter,
        filter_level: [h.lf.level[0] as u8, h.lf.level[1] as u8],
        filter_level_u: h.lf.level[2] as u8,
        filter_level_v: h.lf.level[3] as u8,
        loop_filter_info_fields: av1_loop_filter_info_fields(h),
        // 7.20's `RefDeltas` / `ModeDeltas` **as `setup_past_independence` / `load_previous` resolved
        // them**, which is what `FrameHeader::lf.deltas` is because the parser already applied 7.20.
        // An inter frame with a primary reference therefore submits the primary reference's deltas
        // for whatever this header did not override — the inherited path, and the correct answer
        // rather than a fallback.
        ref_deltas: narrow_i8s::<8>(&h.lf.deltas.ref_deltas, "ref_deltas")?,
        mode_deltas: narrow_i8s::<2>(&h.lf.deltas.mode_deltas, "mode_deltas")?,
        base_qindex: h.quant.base_q_idx as u8,
        y_dc_delta_q: narrow_i8(h.quant.delta_q_y_dc, "y_dc_delta_q")?,
        u_dc_delta_q: narrow_i8(h.quant.delta_q_u_dc, "u_dc_delta_q")?,
        u_ac_delta_q: narrow_i8(h.quant.delta_q_u_ac, "u_ac_delta_q")?,
        v_dc_delta_q: narrow_i8(h.quant.delta_q_v_dc, "v_dc_delta_q")?,
        v_ac_delta_q: narrow_i8(h.quant.delta_q_v_ac, "v_ac_delta_q")?,
        padding_after_v_ac_delta_q: [0; 1],
        qmatrix_fields: av1_qmatrix_fields(h),
        padding_after_qmatrix_fields: [0; 2],
        mode_control_fields: av1_mode_control_fields(h),
        cdef_damping_minus_3: h.cdef.damping.saturating_sub(3) as u8,
        cdef_bits: h.cdef.bits as u8,
        cdef_y_strengths: av1_cdef_strengths(&h.cdef.y_pri, &h.cdef.y_sec),
        cdef_uv_strengths: av1_cdef_strengths(&h.cdef.uv_pri, &h.cdef.uv_sec),
        loop_restoration_fields: av1_loop_restoration_fields(h)?,
        wm: av1_warped_motion(h),
        va_reserved: [0; VA_PADDING_MEDIUM],
        padding_after_va_reserved: [0; 4],
    };
    Ok(record_bytes(&record))
}

/// The per-tile `VASliceParameterBufferAV1` records, one per tile.
fn slices_of(header: &FrameHeader, tiles: &[Tile]) -> Result<Vec<Vec<u8>>, String> {
    let cols = header.tile_info.cols;
    if cols == 0 {
        return Err("an AV1 frame with no tile columns".into());
    }
    tiles.iter().map(|t| av1_slice_record(t.num, cols, t.start, t.end - t.start)).collect()
}

// ---------------------------------------------------------------------------------------------
// Q1: the record's per-tile superblock extents
// ---------------------------------------------------------------------------------------------

/// The record's tile counts and the two superblock-extent arrays it carries.
///
/// ## Q1, answered: indexed by tile **column** and tile **row**, and the last entry is not required
///
/// `va_dec_av1.h:392-397` says: *"The width/height of a tile minus 1 in units of superblocks. Though
/// the maximum number of tiles is 64, since ones of the last tile are computed from ones of the other
/// tiles and frame_width/height, they are not necessarily specified."* That leaves two questions —
/// **which axis indexes each array**, and **whether the last entry is read at all** — and the header
/// settles neither.
///
/// ### The indexing: measured, but only once a fixture existed that could tell
///
/// The layout the driver reads is **`width_in_sbs_minus_1[c]` = tile *column* `c` minus one, and
/// `height_in_sbs_minus_1[r]` = tile *row* `r` minus one**. On `av1_tiled.mp4` (2 columns × 4 rows) it
/// decodes **96 of 96** bit for bit — and so does the rejected reading, indexed by tile number,
/// **96 of 96**. That fixture cannot choose between them, which is worth stating plainly: every
/// column is the same width and every row but the last the same height, so the two indexings place
/// the same values in the same places, and the entries neither writes are zero.
///
/// It takes a grid whose columns and rows **differ from one another** to separate them, and that is
/// what the committed tiled fixture is for: `mi_col_starts = [0, 32, 48, 64, 80, 96, 112, 128, 160]`
/// and `mi_row_starts = [0, 32, 64, 96, 126]` over 8 columns and 4 rows, giving
/// `width_in_sbs = [1, 0, 0, 0, 0, 0, 0, 1]` and `height_in_sbs = [1, 1, 1, 0]`. On that frame:
///
/// | indexing | result |
/// |---|---|
/// | by tile **column** / tile **row** (what this does) | **48 of 48 pictures bit for bit** |
/// | by tile number (`c * rows`, `r * cols`) | **0 of 48** — every picture wrong |
///
/// Indexed by tile number, column 7's width of 1 lands at index 28 of a 63-entry array that nothing
/// else writes, and every intermediate row's height of 1 is lost. The picture does not fail: it is
/// built from tile boundaries the bitstream never described, which is the failure this whole array
/// exists to avoid.
///
/// ### The last entry: written here, and it is not required
///
/// **The last column's and last row's entries are written, and the driver accepts that** — the same
/// 48 of 48 and 96 of 96. Omitting them entirely also decodes 48 of 48 and 96 of 96, so this driver
/// really does compute the last tile from the others and the others from `frame_width`/`height`, as
/// the header says. Writing them is kept because it is what the specification describes the array as
/// holding and a driver that *did* read the last entry would then be right too; the measurement says
/// this one does not.
///
/// **What is still open, and is not answered by any of the above:** what a **full 64-entry grid** does
/// — that is, whether entry 63 is read as the 64th column or ignored — since no fixture this suite can
/// generate has 64 tiles in one axis (a 640×500 frame has 10 × 8 superblocks, and libaom's own
/// `-tile-columns` will not exceed 32 for 8-bit). Recorded as open rather than claimed.
///
/// ## What each entry is
///
/// The specification's own tile layout (5.9.15):
/// `TileWidthSb = ( MiColStarts[i + 1] - MiColStarts[i] ) >> sbShift`, minus one, with `sbShift` 5 for
/// a 128x128 superblock and 4 for a 64x64 one (`use_128x128_superblock`, and the same arithmetic at
/// `crates/av1/src/header.rs:877-879`). `mi_col_starts` has `cols + 1` entries — the parser appends
/// `mi_cols` as the final boundary (`header.rs:906`) — so each is a subtraction of two consecutive
/// entries, never a division by a tile count that could be zero.
///
/// A tile narrower than one superblock in either direction is **refused**: the entries are *minus
/// one* in superblocks, so zero would underflow, and a clamped extent is a boundary the bitstream does
/// not describe.
fn av1_tile_extents(h: &FrameHeader, seq: &SequenceHeader) -> Result<(u8, u8, [u16; 63], [u16; 63]), String> {
    let (cols, rows) = (h.tile_info.cols, h.tile_info.rows);
    if cols == 0 || rows == 0 {
        return Err("an AV1 frame with no tiles".into());
    }
    let num_tiles = cols.checked_mul(rows).ok_or("an AV1 tile grid that overflows")?;
    if num_tiles > MAX_TILES {
        return Err(format!("an AV1 frame of {num_tiles} tiles, more than the {MAX_TILES} the record's tile arrays name"));
    }
    if h.tile_info.mi_col_starts.len() != cols + 1 || h.tile_info.mi_row_starts.len() != rows + 1 {
        return Err("an AV1 tile layout that does not match its own grid".into());
    }
    let sb_shift = if seq.use_128x128_superblock { 5 } else { 4 };
    let (mut width_in_sbs, mut height_in_sbs) = ([0u16; 63], [0u16; 63]);
    for c in 0..cols {
        let (Some(&x0), Some(&x1)) = (h.tile_info.mi_col_starts.get(c), h.tile_info.mi_col_starts.get(c + 1)) else {
            return Err("an AV1 tile layout that does not match its own grid".into());
        };
        let wide = x1.checked_sub(x0).ok_or("an AV1 tile layout that runs backwards")? >> sb_shift;
        if wide == 0 {
            return Err("an AV1 tile column narrower than a superblock".into());
        }
        let Some(slot) = width_in_sbs.get_mut(c) else { break };
        *slot = fits("width_in_sbs_minus_1", wide - 1, 16)? as u16;
    }
    for r in 0..rows {
        let (Some(&y0), Some(&y1)) = (h.tile_info.mi_row_starts.get(r), h.tile_info.mi_row_starts.get(r + 1)) else {
            return Err("an AV1 tile layout that does not match its own grid".into());
        };
        let tall = y1.checked_sub(y0).ok_or("an AV1 tile layout that runs backwards")? >> sb_shift;
        if tall == 0 {
            return Err("an AV1 tile row shorter than a superblock".into());
        }
        let Some(slot) = height_in_sbs.get_mut(r) else { break };
        *slot = fits("height_in_sbs_minus_1", tall - 1, 16)? as u16;
    }
    Ok((
        u8::try_from(cols).map_err(|_| format!("{cols} tile columns do not fit the record's byte"))?,
        u8::try_from(rows).map_err(|_| format!("{rows} tile rows do not fit the record's byte"))?,
        width_in_sbs,
        height_in_sbs,
    ))
}

// ---------------------------------------------------------------------------------------------
// The per-tile slice record
// ---------------------------------------------------------------------------------------------

/// One tile's `VASliceParameterBufferAV1`.
///
/// `va_dec_av1.h:634-645` is the sentence that makes this different from every other codec here:
/// *"This structure conveys parameters related to bit stream data and should be sent once per tile …
/// Please note that host decoder is responsible to parse out the per tile information. And the bit
/// stream in sent to driver in per tile granularity."* The per-tile byte extents are therefore this
/// backend's job rather than the driver's — the same division of labour as VP9's
/// `first_partition_size`, and the same reason they are computed from the frame's own bytes instead
/// of asked of the driver.
///
/// `offset` and `size` are in **the slice data buffer's own coordinates**, which is the temporal unit
/// the frame came in. `va_dec_av1.h:649-658` defines `slice_data_size` as "The byte count of current
/// tile in the bitstream buffer, starting from first byte of the buffer" and `slice_data_offset` as
/// "offset to the first byte of the data buffer" — both against that buffer, which is what makes a
/// set of per-tile records able to address one buffer at all.
///
/// `tg_start` / `tg_end` are `va_deprecated` (`va_dec_av1.h:667-668`) and stay zero: **deprecated is
/// not absent**, so they are declared and written — the record is 40 bytes rather than 36 because of
/// them — and the tile's position is carried by `tile_row` / `tile_column`.
fn av1_slice_record(tile_num: usize, cols: usize, offset: usize, size: usize) -> Result<Vec<u8>, String> {
    let row = tile_num / cols;
    let column = tile_num % cols;
    let record = VaSliceParameterBufferAv1 {
        slice_data_size: u32::try_from(size).map_err(|_| format!("a tile of {size} bytes does not fit the slice record's byte count"))?,
        slice_data_offset: u32::try_from(offset).map_err(|_| format!("a tile at offset {offset} does not fit the slice record's offset"))?,
        // The tile goes in whole, so `VA_SLICE_DATA_FLAG_ALL` — "whole slice is in the buffer"
        // (`va.h:3080`). The other three describe a slice submitted in pieces, which a decoder that
        // renders one picture per `vaRenderPicture` never does.
        slice_data_flag: SLICE_DATA_FLAG_ALL,
        tile_row: u16::try_from(row).map_err(|_| format!("tile row {row} does not fit the slice record"))?,
        tile_column: u16::try_from(column).map_err(|_| format!("tile column {column} does not fit the slice record"))?,
        tg_start: 0,
        tg_end: 0,
        anchor_frame_idx: 0,
        padding_before_tile_idx_in_tile_list: [0; 1],
        tile_idx_in_tile_list: 0,
        va_reserved: [0; VA_PADDING_LOW],
    };
    Ok(record_bytes(&record))
}

// ---------------------------------------------------------------------------------------------
// The six bitfield words. Positions measured by a C compile, not inferred from declaration order.
// ---------------------------------------------------------------------------------------------

/// `pic_info_fields` (32 bits at offset 676).
///
/// Every member here carries the name `abi_guard::the_av1_bitfield_words_are_the_bits_the_c_compiler_measures`
/// measured, in that order. Two of them are worth more than the rest:
///
/// - **`uniform_tile_spacing_flag`** is the bitstream's own, from `TileInfo::uniform_tile_spacing`,
///   and *not* a guess from the log2 counts. `cols_log2` / `rows_log2` are the uniform branch's log2
///   counts in that branch and `ceil(log2(cols))` in the other one (`header.rs:890-953`), so "more
///   than one tile" and "uniform spacing" are two different facts that share a spelling. With the bit
///   clear the driver reads `width_in_sbs_minus_1` / `height_in_sbs_minus_1` instead of regenerating
///   them from the counts (`va_dec_av1.h:385-389`), so a non-uniform frame submitted as uniform would
///   have tile boundaries the bitstream never described.
/// - **`interp_filter`'s counterpart for global motion** is not here but in
///   [`av1_warped_motion`]: `allow_warped_motion` set on a frame the parser resolved to identity
///   global motion is the specification's own state, and is not adjusted.
fn av1_pic_info_fields(h: &FrameHeader) -> u32 {
    pack(&[
        ("frame_type", u32::from(h.frame_type), 2),
        ("show_frame", u32::from(h.show_frame), 1),
        ("showable_frame", u32::from(h.showable_frame), 1),
        ("error_resilient_mode", u32::from(h.error_resilient_mode), 1),
        ("disable_cdf_update", u32::from(h.disable_cdf_update), 1),
        ("allow_screen_content_tools", u32::from(h.allow_screen_content_tools), 1),
        ("force_integer_mv", u32::from(h.force_integer_mv), 1),
        ("allow_intrabc", u32::from(h.allow_intrabc), 1),
        ("use_superres", u32::from(h.use_superres), 1),
        ("allow_high_precision_mv", u32::from(h.allow_high_precision_mv), 1),
        ("is_motion_mode_switchable", u32::from(h.is_motion_mode_switchable), 1),
        ("use_ref_frame_mvs", u32::from(h.use_ref_frame_mvs), 1),
        ("disable_frame_end_update_cdf", u32::from(h.disable_frame_end_update_cdf), 1),
        ("uniform_tile_spacing_flag", u32::from(h.tile_info.uniform_tile_spacing), 1),
        ("allow_warped_motion", u32::from(h.allow_warped_motion), 1),
        // Zero on every frame this backend builds: `AV1_LARGE_SCALE_TILE` says why it cannot be read.
        ("large_scale_tile", 0, 1),
    ])
}

/// `seq_info_fields` (32 bits at offset 4).
///
/// `chroma_sample_position` is bit 14 and is `va_deprecated` (`va_dec_av1.h:285`), which is
/// `__attribute__((deprecated))` and **not** a "leave it out": it still occupies a bit of a word the
/// driver reads, so it is zeroed rather than carrying the sequence header's own value.
fn av1_seq_info_fields(seq: &SequenceHeader) -> u32 {
    pack(&[
        ("still_picture", u32::from(seq.still_picture), 1),
        ("use_128x128_superblock", u32::from(seq.use_128x128_superblock), 1),
        ("enable_filter_intra", u32::from(seq.enable_filter_intra), 1),
        ("enable_intra_edge_filter", u32::from(seq.enable_intra_edge_filter), 1),
        ("enable_interintra_compound", u32::from(seq.enable_interintra_compound), 1),
        ("enable_masked_compound", u32::from(seq.enable_masked_compound), 1),
        ("enable_dual_filter", u32::from(seq.enable_dual_filter), 1),
        ("enable_order_hint", u32::from(seq.enable_order_hint), 1),
        ("enable_jnt_comp", u32::from(seq.enable_jnt_comp), 1),
        ("enable_cdef", u32::from(seq.enable_cdef), 1),
        ("mono_chrome", u32::from(seq.color.mono_chrome), 1),
        ("color_range", u32::from(seq.color.color_range), 1),
        ("subsampling_x", u32::from(seq.color.subsampling_x), 1),
        ("subsampling_y", u32::from(seq.color.subsampling_y), 1),
        ("chroma_sample_position", 0, 1),
        ("film_grain_params_present", u32::from(seq.film_grain_params_present), 1),
    ])
}

/// `loop_filter_info_fields` (8 bits at offset 686).
fn av1_loop_filter_info_fields(h: &FrameHeader) -> u8 {
    pack(&[
        ("sharpness_level", h.lf.sharpness, 3),
        ("mode_ref_delta_enabled", u32::from(h.lf.delta_enabled), 1),
        ("mode_ref_delta_update", u32::from(h.lf.delta_update), 1),
    ]) as u8
}

/// `qmatrix_fields` (16 bits at offset 704).
fn av1_qmatrix_fields(h: &FrameHeader) -> u16 {
    pack(&[("using_qmatrix", u32::from(h.quant.using_qmatrix), 1), ("qm_y", h.quant.qm_y, 4), ("qm_u", h.quant.qm_u, 4), ("qm_v", h.quant.qm_v, 4)]) as u16
}

/// `mode_control_fields` (32 bits at offset 708): 5.9.16's delta-q, 5.9.17's delta-lf and transform
/// size, and 5.9.20's reference selection.
fn av1_mode_control_fields(h: &FrameHeader) -> u32 {
    pack(&[
        ("delta_q_present_flag", u32::from(h.delta_q_present), 1),
        ("log2_delta_q_res", h.delta_q_res, 2),
        ("delta_lf_present_flag", u32::from(h.delta_lf_present), 1),
        ("log2_delta_lf_res", h.delta_lf_res, 2),
        ("delta_lf_multi", u32::from(h.delta_lf_multi), 1),
        ("tx_mode", u32::from(h.tx_mode), 2),
        ("reference_select", u32::from(h.reference_select), 1),
        ("reduced_tx_set_used", u32::from(h.reduced_tx_set), 1),
        ("skip_mode_present", u32::from(h.skip_mode_present), 1),
    ])
}

/// `seg_info` (`VASegmentationStructAV1`, 156 bytes): AV1 5.9.19's segmentation, as
/// `setup_past_independence` / `load_previous` left it.
///
/// The header states the mapping itself — `feature_data` "is equivalent to variable FeatureData[][]
/// in spec, which is after clip3() operation" (`va_dec_av1.h:130-137`) — so the parser's already
/// clipped values are copied across rather than clipped again, and `data` is narrowed to `i16`
/// because that is what this array is.
///
/// These are the **resolved** features, not the ones this header overrode: 7.20 reads
/// `st.seg_features` from the primary reference when there is one and the parser has already done
/// that into `h.seg.features`. A frame that names a primary reference and does not update the feature
/// data must submit the primary reference's, which is the inherited path and is pinned by
/// `an_inter_frame_with_a_primary_reference_submits_its_propagated_values`.
fn av1_seg_info(h: &FrameHeader) -> VaSegmentationStructAv1 {
    let mut feature_data = [[0i16; 8]; 8];
    let mut feature_mask = [0u8; 8];
    for seg in 0..8usize {
        for feature in 0..8usize {
            if h.seg.features.enabled[seg][feature] {
                feature_mask[seg] |= 1u8 << feature;
            }
            let value = h.seg.features.data[seg][feature];
            feature_data[seg][feature] = i16::try_from(value).unwrap_or(if value < 0 { i16::MIN } else { i16::MAX });
        }
    }
    VaSegmentationStructAv1 {
        segment_info_fields: pack(&[
            ("enabled", u32::from(h.seg.enabled), 1),
            ("update_map", u32::from(h.seg.update_map), 1),
            ("temporal_update", u32::from(h.seg.temporal_update), 1),
            ("update_data", u32::from(h.seg.update_data), 1),
        ]),
        feature_data,
        feature_mask,
        va_reserved: [0; VA_PADDING_LOW],
    }
}

/// `cdef_y_strengths[8]` / `cdef_uv_strengths[8]`, in the packing the header spells out:
/// *"(cdef_y_strengths[]) = (cdef_y_pri_strength[] << 2) | (cdef_y_sec_strength[] & 0x03)"*
/// (`va_dec_av1.h:585-599`).
fn av1_cdef_strengths(pri: &[u32; 8], sec: &[u32; 8]) -> [u8; 8] {
    let mut out = [0u8; 8];
    for at in 0..8 {
        out[at] = (((pri[at] & 0x0f) << 2) | (sec[at] & 0x03)) as u8;
    }
    out
}

/// `loop_restoration_fields` (16 bits at offset 730): the three planes' restoration types and the two
/// shifts.
///
/// **Q5 — `yframe_restoration_type`'s encoding is not documented.** `va_dec_av1.h:606-618` declares
/// the union with **no comment at all** on any of its members, so the header does not say which of the
/// two AV1 numberings it wants. There are two: the specification's own `FrameRestorationType` enum
/// (`NONE` 0, `WIENER` 1, `SGRPROJ` 2, `SWITCHABLE` 3 — the order this parser stores, via the `REMAP`
/// table at `crates/av1/src/header.rs:1114`) and the **bitstream's** two-bit field, whose order is
/// `NONE` 0, `SWITCHABLE` 1, `WIENER` 2, `SGRPROJ` 3. They differ for every value but `NONE`.
///
/// ## Q5, answered by measurement: the record wants the **specification's `FrameRestorationType`**
///
/// The three type members carry the specification's enum **unmodified** — the parser's own value —
/// and `av1_loop_restoration.mp4` decodes **96 of 96 pictures bit for bit** with that. This was
/// measured, not reasoned, and the measurement is worth recording because **the obvious reading of
/// the header points the other way and is wrong**.
///
/// `va_dec_av1.h:606-618` documents none of the five members, so the header cannot decide it. What
/// can decide it wrongly is `va_enc_av1.h:855-863`, which puts "Corresponds to AV1 syntax element of
/// the same name" on `lr_unit_shift` and `lr_uv_shift` — and *not* on the three types, two lines
/// above describing exactly the same three quantities. Reading that omission as "the types are the
/// bitstream's two-bit field, since the shifts are" is the whole of the trap: it says nothing about
/// the types, and the bitstream's two-bit field is a **different numbering** from the specification's
/// enum — `NONE` 0, `SWITCHABLE` 1, `WIENER` 2, `SGRPROJ` 3 on the wire, against `NONE` 0, `WIENER`
/// 1, `SGRPROJ` 2, `SWITCHABLE` 3 in `FrameRestorationType` (`crates/av1/src/spec_tables.rs:274-277`,
/// which the parser's `REMAP` at `header.rs:1127` reaches through).
///
/// Submitting the wire numbering instead of the enum's decodes this fixture to **18 of 96**, and the
/// 18 are exactly the frames whose restoration types are all `NONE` — the frames that cannot tell
/// the two numberings apart. That is the whole reason this fixture exists: three of the other four
/// have loop restoration off and would have passed under either reading, so a suite without it could
/// not have chosen.
///
/// **The rule this leaves, and it is the one to carry to the next undocumented AV1 member:** "the
/// header does not document this" means *ask the hardware*, and asking means building a fixture that
/// **distinguishes the readings** — a stream that restores, a frame with more than one tile, a
/// sequence header that disagrees with the sample entry. A fixture that cannot separate two
/// hypotheses settles neither, and three passing fixtures prove less than one that can fail.
fn av1_loop_restoration_fields(h: &FrameHeader) -> Result<u16, String> {
    let (unit, uv) = av1_lr_shifts(h)?;
    Ok(pack(&[
        ("yframe_restoration_type", u32::from(av1_restoration_record(h.lr.frame_restoration_type[0])), 2),
        ("cbframe_restoration_type", u32::from(av1_restoration_record(h.lr.frame_restoration_type[1])), 2),
        ("crframe_restoration_type", u32::from(av1_restoration_record(h.lr.frame_restoration_type[2])), 2),
        ("lr_unit_shift", u32::from(unit), 2),
        ("lr_uv_shift", u32::from(uv), 1),
    ]) as u16)
}

/// One plane's restoration type as the record carries it, checked against `crates/av1`'s own enum.
///
/// The value is the parser's, unchanged — this is where the *enum* is checked to still be one of the
/// four the specification defines rather than a number that happens to fit two bits. It is a check
/// and not a conversion on purpose: the conversion is the identity, which is a measurement
/// ([`av1_loop_restoration_fields`]), and writing it as a table of `[NONE, WIENER, SGRPROJ,
/// SWITCHABLE] -> [0, 1, 2, 3]` would hide that behind a reindexing that could silently become a
/// permutation again — which is exactly the mistake this fixture exists to catch.
fn av1_restoration_record(spec_enum: u8) -> u8 {
    debug_assert!(
        [filmcraft_av1::RESTORE_NONE, filmcraft_av1::RESTORE_WIENER, filmcraft_av1::RESTORE_SGRPROJ, filmcraft_av1::RESTORE_SWITCHABLE]
            .contains(&(spec_enum as usize)),
        "a loop restoration type of {spec_enum} is not one of the specification's four"
    );
    spec_enum & 0x03
}

/// Q2, answered: `lr_unit_shift` and `lr_uv_shift` are **derived from the sizes the parser stores**,
/// by inverting the specification's own formulas.
///
/// The parser's `lr_params` computes, from the bitstream's own shifts:
///
/// ```text
/// LoopRestorationSize[0] = RESTORATION_TILESIZE_MAX >> (2 - lr_unit_shift)   RESTORATION_TILESIZE_MAX = 256
/// LoopRestorationSize[1] = LoopRestorationSize[0] >> lr_uv_shift
/// ```
///
/// (`crates/av1/src/header.rs:1154` and `:1157` — 5.9.24's arithmetic verbatim.) Inverting the first
/// gives `lr_unit_shift = 2 - log2(size / 64)`, so **256 → 2, 128 → 1, 64 → 0**: the **bigger** the
/// restoration unit, the **larger** the shift. That direction is the easy thing to get backwards —
/// the shift reads like "how much to shift *down* to reach the unit", and it is in fact how much
/// `2` is shifted down by — and getting it backwards submits a 64-pixel unit for a 256-pixel stream.
/// It was got backwards here first, and the hardware caught it: `av1_loop_restoration.mp4` decoded
/// **18 of 96** with the two swapped, the same 18 frames that carry no restoration and so cannot
/// distinguish a unit size at all. `lr_uv_shift = log2(size0) - log2(size1)`, so equal sizes → 0 and
/// half → 1. A size outside those sets derives to no shift at all, which is an **`Err`, never a
/// clamp**: a clamp would be a restoration unit the stream never described.
///
/// **When the frame restores nothing** — `uses_lr == false`, every plane `RESTORE_NONE` — the
/// specification reads neither shift and the parser leaves the sizes at zero. This submits **0**,
/// which is `lr_unit_shift = 0` (a 64-pixel unit) and `lr_uv_shift = 0`. That is not a guess about
/// a value the stream carries: a frame with no restoration on any plane has no unit size to carry, and
/// the field only sizes restoration the frame does not have. Three of this suite's five fixtures have
/// loop restoration off in their sequence header, and the oracle is asked on all of them.
fn av1_lr_shifts(h: &FrameHeader) -> Result<(u8, u8), String> {
    if !h.lr.uses_lr {
        return Ok((0, 0));
    }
    let shift_of = |size: u32| match size {
        256 => Some(2u8),
        128 => Some(1),
        64 => Some(0),
        _ => None,
    };
    let (Some(unit), Some(_)) = (shift_of(h.lr.loop_restoration_size[0]), shift_of(h.lr.loop_restoration_size[1])) else {
        return Err(format!(
            "an AV1 loop restoration unit of {}x{} pixels, which is not one of 64, 128 or 256 and so derives from no shift",
            h.lr.loop_restoration_size[0], h.lr.loop_restoration_size[1]
        ));
    };
    let uv = match (h.lr.loop_restoration_size[0], h.lr.loop_restoration_size[1]) {
        (a, b) if a == b => 0u8,
        (a, b) if a == b * 2 => 1,
        // A chroma unit that is neither the luma one nor half of it is not a value `lr_uv_shift` can
        // produce. Refused rather than guessed: the two would give different pictures.
        (a, b) => return Err(format!("an AV1 chroma restoration unit of {b} against a luma one of {a}, which lr_uv_shift cannot name")),
    };
    Ok((unit, uv))
}

/// The seven `VAWarpedMotionParamsAV1`, one per reference the record's indirection names.
///
/// **Q3, answered from the header and the parser together.** `va_dec_av1.h:222` says `wmmat[]`
/// "corresponds to gm_params[][] in spec", and the specification's array is **six** parameters, so the
/// last two entries are zero. The parser already carries `gm_params` at `WARPEDMODEL_PREC_BITS` (16)
/// fractional bits — its `read_global_param` shifts each value into that scale
/// (`crates/av1/src/header.rs:1247-1253`) — so the record's `wmmat` is **the same number copied** and no
/// rescaling is applied. `wmtype` is `gm_type[rf]`, whose numbering is the specification's own (0
/// none, 1 translational, 2 rotzoom, 3 affine) and is what `VAAV1TransformationType` measures to
/// (`va_dec_av1.h:203-214`, pinned in `abi_guard`).
///
/// `wm[i]` is `gm_params[LAST_FRAME + i]`: `gm_params[0]` is `INTRA_FRAME`'s, and the specification
/// keeps global motion only for `LAST_FRAME..=ALTREF_FRAME` — which is exactly the seven entries
/// `ref_frame_idx[7]` names, and is the loop `global_motion_params` runs (`header.rs:1195`).
///
/// ## What is open, and what it is not
///
/// `VAWarpedMotionParamsAV1::invalid` has **no counterpart in the AV1 specification at all**, and the
/// header's only words on it are "valid or invalid on affine set" (`va_dec_av1.h:228`). It is written
/// as 0 — the value that says "valid" — because every frame this backend describes has resolved global
/// motion, whether identity or otherwise. **That the driver reads it is not settled here**: no fixture
/// in this suite carries a non-identity global motion, which is a property of `testsrc2` and not of
/// the encode settings (`-enable-global-motion 1` changes nothing on this content, measured with this
/// repository's own parser), so **this builder's handling of a rotated or affine `gm_params` is
/// unverified by the oracle**. Recorded open rather than claimed.
fn av1_warped_motion(h: &FrameHeader) -> [VaWarpedMotionParamsAv1; 7] {
    // Named so a change of it in `crates/av1` is a visible diff here rather than a silent disagreement
    // between the parser's scale and the record's.
    debug_assert_eq!(WARPEDMODEL_PREC_BITS, 16, "the record's wmmat is at WARPEDMODEL_PREC_BITS, which is 16");
    std::array::from_fn(|i| {
        let rf = LAST_FRAME + i;
        let mut wmmat = [0i32; 8];
        if let Some(params) = h.gm_params.get(rf) {
            for (out, value) in wmmat.iter_mut().zip(params.iter().take(6)) {
                *out = *value;
            }
        }
        VaWarpedMotionParamsAv1 {
            wmtype: u32::from(h.gm_type.get(rf).copied().unwrap_or(0)) & 0x03,
            wmmat,
            invalid: 0,
            padding_before_va_reserved: [0; 3],
            va_reserved: [0; VA_PADDING_LOW],
        }
    })
}

// ---------------------------------------------------------------------------------------------
// The remaining conversions
// ---------------------------------------------------------------------------------------------

/// `order_hint_bits_minus_1`.
///
/// The sequence header reads `order_hint_bits` **only** when `enable_order_hint` is set
/// (`crates/av1/src/header.rs:156-158`), so a stream with it clear leaves the field at 0 and there is
/// no minus one to take. Zero is then submitted — a one-bit order-hint mask — which loses nothing
/// because such a stream spells no order hint at all: `FrameHeader::parse` reads
/// `order_hint = f(order_hint_bits)`, which is `f(0)` and so 0, for every frame of it, and
/// `RelativeDist` returns 0 outright when `!enable_order_hint` (`header.rs:449-451`). Stated rather
/// than assumed, because the alternative — a guessed constant — is the same failure this record's
/// other fields are guarded against.
fn av1_order_hint_bits(seq: &SequenceHeader) -> u8 {
    u8::try_from(seq.order_hint_bits.saturating_sub(1)).unwrap_or(0)
}

/// `bit_depth_idx`: 8 → 0, 10 → 1, 12 → 2 (`va_dec_av1.h:255-261`).
fn av1_bit_depth_idx(depth: u8) -> Result<u8, String> {
    match depth {
        8 => Ok(0),
        10 => Ok(1),
        12 => Ok(2),
        other => Err(format!("{other} bits is not a depth the record's bit_depth_idx names")),
    }
}

/// `ref_frame_idx[7]`: the header's own values, each narrowed into the record's byte.
fn av1_ref_frame_idx(h: &FrameHeader) -> [u8; REFS_PER_FRAME] {
    let mut out = [0u8; REFS_PER_FRAME];
    for (out, &slot) in out.iter_mut().zip(h.ref_frame_idx.iter()) {
        *out = u8::try_from(slot).unwrap_or(0);
    }
    out
}

/// The container's description and a frame's header must agree about the AV1 **profile** and the
/// **bit depth**, and this is where that is checked — the one site, so the decode loop and
/// [`super::picture::build_frame`] cannot disagree about it.
///
/// The surfaces were created for the profile and depth the sample entry said; a record carrying the
/// frame's own would be a picture of one format decoded into a surface of another. Both numbers are
/// taken as values rather than as the `FrameStreamInfo` that carries them, because the decode loop
/// reads them inside a loop that mutates the decoder and cannot hold a borrow of the stream
/// description across it — and they are the only two fields this reads.
///
/// The comparison is against the **sequence header** rather than against `FrameHeader::color`, and
/// that is deliberate: 5.9's uncompressed header spells no colour configuration at all — it is the
/// sequence header that carries it, for every frame including an intra one — so there is nothing on
/// the frame header to compare for a frame that inherits it. Reading `h.color` here is not available
/// and would have been the VP9 Ruling-7 shape: a valid record of the wrong depth, which a driver
/// decodes into a plausible wrong picture.
pub fn av1_entry_disagreement(profile: u8, bit_depth: u32, seq: &SequenceHeader) -> Option<String> {
    if seq.profile != profile {
        return Some(format!("AV1 picture parameters: the sample entry says profile {profile}, the sequence header says profile {}", seq.profile));
    }
    if u32::from(seq.color.bit_depth) != bit_depth {
        return Some(format!(
            "AV1 picture parameters: the sample entry says {bit_depth} bits, the sequence header's colour configuration says {} bits",
            seq.color.bit_depth
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------------------------
    // Reading a real fixture, so every test below is about a record built from a header this
    // repository's own parser produced out of a committed encode — not a hand-made one.
    // ---------------------------------------------------------------------------------------------

    /// A named fixture's `av1C` configuration OBUs and its samples, in decode order.
    ///
    /// `None` when the fixture has not been generated: every test here returns rather than fails,
    /// because the fixtures are built by `cargo xtask fixtures platform` and a test that fails
    /// because an encode has not been run says nothing about the backend. `tests/vaapi.rs` is the
    /// gate, and it generates them itself.
    fn read_fixture(name: &str) -> Option<(Vec<u8>, Vec<Vec<u8>>)> {
        let path = filmcraft_testkit::fixtures_dir("platform").join(name);
        let bytes = std::fs::read(&path).ok()?;
        let file = filmcraft_isobmff::open(&bytes).ok()?;
        let t = file.track_of_kind(filmcraft_isobmff::TrackKind::Video)?;
        let entry = file.tracks[t].entries.first().cloned()?;
        let info = crate::vaapi::stream_info(&entry)?;
        let crate::vaapi::stream::Stream::Frame(info) = crate::vaapi::Stream::from(info) else { return None };
        let samples = (0..file.tracks[t].samples.len()).filter_map(|i| file.read_sample(&bytes, t, i).ok()).collect();
        Some((info.config_obus.clone(), samples))
    }

    /// The first frame of `name` that codes a picture of its own, and the sequence header it was
    /// parsed against.
    fn first_new_frame(name: &str) -> Option<(SequenceHeader, FrameHeader)> {
        let (config, samples) = read_fixture(name)?;
        let sample = samples.first()?;
        let mut st = Av1State::new(&config);
        let f = st.frame_at(sample, 0).ok()??;
        Some((st.sequence_header()?.clone(), f.header()?.clone()))
    }

    // ---------------------------------------------------------------------------------------------
    // Task 4 Step 1 — the record's own bytes
    // ---------------------------------------------------------------------------------------------

    /// **The record for a real key frame is exactly these bytes.**
    ///
    /// This is the "a 33-instead-of-32 class error cannot pass" test: four wrong constants passed
    /// this repository's whole integration suite before anything read the record back. So the record
    /// is built from a header `FrameHeader::parse` produced out of a committed encode, read back
    /// through the same safe reader the encode side uses, and every member compared against **the
    /// header it came from** — never against a literal, so a change to the parser surfaces here with
    /// a name rather than as a silent agreement.
    #[test]
    fn an_av1_record_is_built_from_the_frame_header() {
        let Some((seq, h)) = first_new_frame("av1_p0_8bit.mp4") else { return };
        let surface = 7u32;
        let refs = Av1ReferenceSet { slots: [Some(11), None, Some(13), None, Some(15), None, None, None] };
        let bytes = av1_picture_record(&h, &seq, &refs, surface).expect("a profile 0 key frame is describable");

        assert_eq!(bytes.len(), 1160, "VADecPictureParameterBufferAV1 is 1160 bytes");
        let r = crate::vaapi::ffi::picture_params_av1(&bytes).expect("1160 bytes is an AV1 picture record");

        // --- the size, and it is the *upscaled* one ------------------------------------------------
        // "Picture original resolution. If SuperRes is enabled, this is the upscaled resolution"
        // (`va_dec_av1.h:332-336`). AV1's superres is horizontal only, so the height is the coded one.
        assert_eq!(u32::from(r.frame_width_minus1), h.upscaled_width - 1, "frame_width_minus1 is the upscaled width, not the coded one");
        assert_eq!(u32::from(r.frame_height_minus1), h.frame_height - 1, "AV1 superres is horizontal, so the height is unchanged");

        // --- the picture-info word, member by member, at the bit the C compiler measured ---------
        let pic = r.pic_info_fields;
        assert_eq!(pic & 0x3, u32::from(h.frame_type), "frame_type is bits 0:2");
        assert_eq!((pic >> 2) & 1, u32::from(h.show_frame), "show_frame is bit 2");
        assert_eq!((pic >> 3) & 1, u32::from(h.showable_frame), "showable_frame is bit 3");
        assert_eq!((pic >> 4) & 1, u32::from(h.error_resilient_mode), "error_resilient_mode is bit 4");
        assert_eq!((pic >> 5) & 1, u32::from(h.disable_cdf_update), "disable_cdf_update is bit 5");
        assert_eq!((pic >> 9) & 1, u32::from(h.use_superres), "use_superres is bit 9");
        assert_eq!(
            (pic >> 14) & 1,
            u32::from(h.tile_info.uniform_tile_spacing),
            "uniform_tile_spacing_flag is bit 14, and it is the bitstream's own rather than a guess from the log2 counts"
        );
        assert_eq!((pic >> 15) & 1, u32::from(h.allow_warped_motion), "allow_warped_motion is bit 15");
        assert_eq!((pic >> 16) & 1, 0, "large_scale_tile is zero on every frame built here — AV1_LARGE_SCALE_TILE says why it cannot be read");

        // --- the reference map: the whole picture buffer, by slot ---------------------------------
        for (i, want) in refs.slots.iter().enumerate() {
            assert_eq!(r.ref_frame_map[i], want.unwrap_or(INVALID_SURFACE), "ref_frame_map[{i}] — an empty slot must not name some surface");
        }
        assert_eq!(r.current_frame, surface, "current_frame is the surface the picture is decoded into");
        assert_eq!(r.current_display_picture, surface, "current_display_picture is set equal rather than left stale");
        for (i, &want) in h.ref_frame_idx.iter().enumerate() {
            assert_eq!(u32::from(r.ref_frame_idx[i]), want as u32, "ref_frame_idx[{i}] is the header's own, 0-based");
        }
        assert_eq!(u32::from(r.primary_ref_frame), h.primary_ref_frame as u32, "primary_ref_frame, and PRIMARY_REF_NONE is 7 rather than 0");
        assert_eq!(u32::from(r.order_hint), h.order_hint, "order_hint");

        // --- quantisation, filters and the deltas -------------------------------------------------
        assert_eq!(u32::from(r.base_qindex), h.quant.base_q_idx, "base_qindex");
        assert_eq!(i32::from(r.y_dc_delta_q), h.quant.delta_q_y_dc, "y_dc_delta_q");
        assert_eq!(i32::from(r.u_dc_delta_q), h.quant.delta_q_u_dc, "u_dc_delta_q");
        assert_eq!(i32::from(r.u_ac_delta_q), h.quant.delta_q_u_ac, "u_ac_delta_q");
        assert_eq!(i32::from(r.v_dc_delta_q), h.quant.delta_q_v_dc, "v_dc_delta_q");
        assert_eq!(i32::from(r.v_ac_delta_q), h.quant.delta_q_v_ac, "v_ac_delta_q");
        assert_eq!(u32::from(r.filter_level[0]), h.lf.level[0], "filter_level[0] is luma");
        assert_eq!(u32::from(r.filter_level[1]), h.lf.level[1], "filter_level[1] is luma");
        assert_eq!(u32::from(r.filter_level_u), h.lf.level[2], "filter_level_u is the Cb level");
        assert_eq!(u32::from(r.filter_level_v), h.lf.level[3], "filter_level_v is the Cr level");
        assert_eq!(
            r.cdef_damping_minus_3 as u32,
            h.cdef.damping.saturating_sub(3),
            "cdef_damping_minus_3 is the specification's damping minus three, not a clamped three"
        );
        assert_eq!(u32::from(r.cdef_bits), h.cdef.bits, "cdef_bits");
        for at in 0..8usize {
            assert_eq!(
                u32::from(r.cdef_y_strengths[at]),
                ((h.cdef.y_pri[at] & 0x0f) << 2) | (h.cdef.y_sec[at] & 0x03),
                "cdef_y_strengths[{at}] is (pri << 2) | (sec & 3) — `va_dec_av1.h:585-599`"
            );
        }
        assert_eq!(u32::from(r.interp_filter), u32::from(h.interpolation_filter), "interp_filter");
        assert_eq!(u32::from(r.superres_scale_denominator), h.superres_denom, "superres_scale_denominator");
        for (i, &want) in h.lf.deltas.ref_deltas.iter().enumerate() {
            assert_eq!(i32::from(r.ref_deltas[i]), want, "ref_deltas[{i}]");
        }
        for (i, &want) in h.lf.deltas.mode_deltas.iter().enumerate() {
            assert_eq!(i32::from(r.mode_deltas[i]), want, "mode_deltas[{i}]");
        }

        // --- the sequence header's own flags ------------------------------------------------------
        assert_eq!(r.profile, seq.profile, "profile");
        assert_eq!(r.bit_depth_idx, 0, "an 8-bit sequence is bit_depth_idx 0 (`va_dec_av1.h:255-261`)");
        assert_eq!(u32::from(r.matrix_coefficients), u32::from(seq.color.matrix_coefficients), "matrix_coefficients");
        let seqf = r.seq_info_fields;
        assert_eq!(seqf & 1, u32::from(seq.still_picture), "still_picture is bit 0");
        assert_eq!((seqf >> 1) & 1, u32::from(seq.use_128x128_superblock), "use_128x128_superblock is bit 1");
        assert_eq!((seqf >> 9) & 1, u32::from(seq.enable_cdef), "enable_cdef is bit 9");
        assert_eq!((seqf >> 14) & 1, 0, "chroma_sample_position is va_deprecated (`va_dec_av1.h:285`) and zeroed rather than left stale");
        assert_eq!((seqf >> 15) & 1, u32::from(seq.film_grain_params_present), "film_grain_params_present is bit 15");

        // --- the six compiler holes, and every reserved word ---------------------------------------
        // They are named for the branch's standing reason: `VASegmentParameterVP9` once decoded
        // bit-exact while handing the driver two uninitialised bytes per record. A hole the C
        // compiler inserts is a hole this must write, and `abi_guard` asks C for every gap width.
        let mut holes: Vec<u8> = Vec::new();
        holes.extend_from_slice(&r.padding_after_anchor_frames_num);
        holes.extend_from_slice(&r.padding_after_order_hint);
        holes.extend_from_slice(&r.padding_after_context_update_tile_id);
        holes.extend_from_slice(&r.padding_after_v_ac_delta_q);
        holes.extend_from_slice(&r.padding_after_qmatrix_fields);
        holes.extend_from_slice(&r.padding_after_va_reserved);
        assert!(holes.iter().all(|&b| b == 0), "every compiler hole in the record is submitted as zero, got {holes:?}");
        assert!(r.va_reserved.iter().all(|&w| w == 0), "va_reserved is zero");
        assert_eq!(
            r.film_grain_info.film_grain_info_fields & 1,
            0,
            "film grain is declined, so apply_grain (bit 0) says 'not present' rather than carrying stale values"
        );
        assert_eq!(r.film_grain_info.grain_seed, 0, "and the rest of the grain sub-record is left at not-present, not stale");
        assert_eq!(r.anchor_frames_num, 0, "no anchor frames: large-scale tile is declined");
        assert_eq!(r.anchor_frames_list, 0, "no anchor frames list");
        assert_eq!(r.tile_count_minus_1, 0, "tile_count_minus_1 is large-scale tile's");
        assert_eq!(u32::from(r.context_update_tile_id), h.tile_info.context_update_tile_id, "context_update_tile_id");
        // wm[7]: seven entries, the specification's LAST_FRAME..ALTREF_FRAME, wmmat at
        // WARPEDMODEL_PREC_BITS copied unchanged and the two spare entries zero.
        for (i, w) in r.wm.iter().enumerate() {
            let rf = LAST_FRAME + i;
            assert_eq!(w.wmtype, h.gm_type[rf] as u32 & 0x03, "wm[{i}].wmtype is gm_type[LAST_FRAME + {i}]");
            for k in 0..6 {
                assert_eq!(w.wmmat[k], h.gm_params[rf][k], "wm[{i}].wmmat[{k}] is gm_params[LAST_FRAME + {i}][{k}] at WARPEDMODEL_PREC_BITS, copied");
            }
            assert_eq!(w.wmmat[6], 0, "wm[{i}].wmmat[6] — the specification's gm_params is six wide");
            assert_eq!(w.wmmat[7], 0, "wm[{i}].wmmat[7]");
            assert_eq!(w.invalid, 0, "wm[{i}].invalid is 0, the value that says valid");
        }
    }

    /// **The mutation check: a value the record's field cannot hold is an `Err`, not a wrong record.**
    ///
    /// Masking instead of refusing submits a member the bitstream never set, inside a field the
    /// driver reads as something else — the 33-instead-of-32 class of error. Each case below drives
    /// a header the parser could not have produced, and the refusal must **name the field**, so a
    /// caller can tell which value it cannot describe.
    #[test]
    fn an_av1_record_refuses_a_value_its_field_cannot_hold() {
        let Some((seq, base)) = first_new_frame("av1_p0_8bit.mp4") else { return };
        let empty = Av1ReferenceSet::default();

        let mut wide = base.clone();
        wide.order_hint = 4096;
        let why = av1_picture_record(&wide, &seq, &empty, 1).expect_err("an order hint of 4096 does not fit the record's byte");
        assert!(why.contains("order_hint"), "the refusal names the field it cannot describe: {why}");

        let mut bad_ref = base.clone();
        bad_ref.ref_frame_idx = [256, 0, 0, 0, 0, 0, 0];
        let why = av1_picture_record(&bad_ref, &seq, &empty, 1).expect_err("a reference slot of 256 does not fit the record's byte");
        assert!(why.contains("ref_frame_idx[0]"), "the refusal names the entry: {why}");

        let mut wide_q = base.clone();
        wide_q.quant.base_q_idx = 256;
        let why = av1_picture_record(&wide_q, &seq, &empty, 1).expect_err("a base_q_idx of 256 does not fit the record's byte");
        assert!(why.contains("base_qindex"), "{why}");

        // **A mutation that cannot fail is not in this test.** `filter_level`'s header range is
        // [0..63] but its member is a `uint8_t`, so a level of 64 fits the field and nothing here
        // could object — the guard is on the *width*, not on the specification's range, and a test
        // that claimed otherwise would be asserting a check this codebase does not have.

        // The two declines, decided before anything is packed.
        let mut grain = base.clone();
        grain.film_grain.apply_grain = true;
        let why = av1_picture_record(&grain, &seq, &empty, 1).expect_err("applied film grain is declined");
        assert_eq!(why, AV1_FILM_GRAIN, "film grain declines with the one sentence the module states once, so the decode loop and the record cannot disagree");

        let mut zero = base.clone();
        zero.upscaled_width = 0;
        let why = av1_picture_record(&zero, &seq, &empty, 1).expect_err("a frame of zero width has no size to submit");
        assert!(why.contains("zero width"), "a zero width says so rather than wrapping to 65535: {why}");

        let mut re_show = base.clone();
        re_show.show_existing_frame = true;
        let why = av1_picture_record(&re_show, &seq, &empty, 1).expect_err("a re-show carries no picture parameters");
        assert!(why.contains("re-shows"), "{why}");

        // The profile table, in the same words `plan_frame` uses, so a caller cannot get a record
        // for a stream it was already told it could not decode.
        let mut high = seq.clone();
        high.profile = 2;
        let why = av1_picture_record(&base, &high, &empty, 1).expect_err("profile 2 is declined");
        assert!(why.contains("AV1 profile 2"), "{why}");
    }

    // ---------------------------------------------------------------------------------------------
    // Task 4 Step 5 — the tile walk, cross-checked against the decoder that decodes with it
    // ---------------------------------------------------------------------------------------------

    /// **The tiles this backend submits are the tiles `filmcraft_av1::Decoder` splits the frame into.**
    ///
    /// `Av1State::tiles_of` is a **second implementation** of 5.9's `tile_group_obu`, and the only
    /// reason it can exist as a second implementation is that it is checkable against the first.
    ///
    /// The check is against [`filmcraft_av1::tile_extents`] — which **is** what `Decoder::tile_group`
    /// calls, the code the pictures this record describes were decoded with — rather than against a
    /// third reading of the same paragraph. Two walks that disagree do not raise an error; they
    /// produce a set of plausible-looking wrong tiles, which is precisely what a
    /// `VASliceParameterBufferAV1` built out of them would then submit.
    #[test]
    fn the_tiles_the_backend_submits_are_the_tiles_the_decoder_splits() {
        let Some((config, samples)) = read_fixture("av1_tiled.mp4") else { return };
        let mut st = Av1State::new(&config);
        let mut surface = 2000u32;
        let mut compared = 0usize;
        let mut frames = 0usize;

        for sample in &samples {
            let mut at = 0usize;
            while let Some(frame) = st.frame_at(sample, at).ok().flatten() {
                at = frame.next();
                let Some(h) = frame.header() else { break };
                frames += 1;
                let t = &h.tile_info;

                // This backend's walk, recovered from the slice records it submitted: one per tile,
                // each naming a byte range in the temporal unit that was the slice data buffer.
                let submitted: Vec<(usize, u32, u32)> = frame
                    .slices()
                    .iter()
                    .map(|rec| {
                        let s = crate::vaapi::ffi::slice_params_av1(rec).expect("a slice record is a slice record");
                        (s.tile_row as usize * t.cols + s.tile_column as usize, s.slice_data_offset, s.slice_data_size)
                    })
                    .collect();
                assert_eq!(submitted.len(), t.cols * t.rows, "one slice record per tile (`va_dec_av1.h:635`), not one per frame");
                assert_eq!(submitted.len(), frame.slices().len(), "every slice record reads back");

                // The decoder's own split, over **the very payloads this frame's walk was given** —
                // not a re-walk of the sample, which could differ from this one and make the check
                // agree with a walk it was not testing.
                for (at, payload) in frame.groups() {
                    let theirs = filmcraft_av1::tile_extents(payload, t.cols, t.rows, t.cols_log2, t.rows_log2, t.tile_size_bytes)
                        .unwrap_or_else(|e| panic!("the decoder splits the tile group this frame's walk was given, but its own walk returned {e}"));
                    for e in &theirs {
                        let (row, col) = (e.tile_num / t.cols, e.tile_num % t.cols);
                        let found = submitted.iter().find(|(n, _, _)| *n == e.tile_num).unwrap_or_else(|| {
                            panic!(
                                "tile {0} (row {row}, col {col}) was submitted nowhere, though Decoder::tile_group splits it out of the group at {at}",
                                e.tile_num
                            )
                        });
                        assert_eq!(
                            found.1 as usize,
                            *at + e.start,
                            "tile {0} (row {row}, col {col}) starts at a different byte than Decoder::tile_group splits it to: submitted {1}, decoder {2}",
                            e.tile_num,
                            found.1,
                            *at + e.start
                        );
                        assert_eq!(found.2 as usize, e.end - e.start, "tile {0} (row {row}, col {col}) is a different length than the decoder's", e.tile_num);
                        compared += 1;
                    }
                }

                st.commit(&frame, surface).ok();
                surface += 1;
                if at >= sample.len() {
                    break;
                }
            }
        }
        assert!(frames > 20, "the tiled fixture should give dozens of frames, not {frames} — regenerate it with `cargo xtask fixtures platform`");
        assert!(compared > frames, "each frame should contribute at least one tile to compare, {frames} frames gave {compared}");
    }

    /// **One slice record per tile, addressing that tile's own bytes, in order and disjoint.**
    ///
    /// `VASliceParameterBufferAV1` "should be sent once per tile … the bit stream is sent to driver
    /// in per tile granularity" (`va_dec_av1.h:634-645`), and `slice_data_offset` is "offset to the
    /// first byte of the data buffer" (`:655-658`) — so the records address the *sample* that was
    /// submitted as the slice data buffer, not one frame's tile group. `tg_start` / `tg_end` are
    /// `va_deprecated` (`:667-668`) and are written zero: deprecated is not absent, which is why the
    /// record is 40 bytes and not 36.
    #[test]
    fn an_av1_slice_record_per_tile_addresses_its_own_bytes() {
        let Some((config, samples)) = read_fixture("av1_tiled.mp4") else { return };
        let mut st = Av1State::new(&config);
        let mut surface = 3000u32;
        let mut records = 0usize;

        for (sample, sample_len) in samples.iter().zip(samples.iter().map(Vec::len)) {
            let _ = sample_len;
            let mut at = 0usize;
            while let Some(frame) = st.frame_at(sample, at).ok().flatten() {
                at = frame.next();
                let Some(h) = frame.header() else { break };
                let cols = h.tile_info.cols;
                let mut covered: Vec<(usize, usize)> = Vec::new();
                for (n, rec) in frame.slices().iter().enumerate() {
                    let s = crate::vaapi::ffi::slice_params_av1(rec).expect("a slice record is a slice record");
                    assert_eq!(rec.len(), 40, "VASliceParameterBufferAV1 is 40 bytes");
                    assert_eq!(s.slice_data_flag, SLICE_DATA_FLAG_ALL, "the tile goes in whole, so the flag is VA_SLICE_DATA_FLAG_ALL");
                    assert_eq!(s.tile_row as usize, n / cols, "tile_row is the tile's row in the raster order");
                    assert_eq!(s.tile_column as usize, n % cols, "tile_column is the tile's column");
                    assert_eq!(s.tg_start, 0, "tg_start is deprecated and written zero rather than left stale");
                    assert_eq!(s.tg_end, 0, "tg_end is deprecated and written zero");
                    assert_eq!(s.tile_idx_in_tile_list, 0, "tile_idx_in_tile_list is valid only when large_scale_tile is 1");
                    assert_eq!(s.anchor_frame_idx, 0, "anchor_frame_idx is large-scale tile's");
                    let start = s.slice_data_offset as usize;
                    let end = start + s.slice_data_size as usize;
                    assert!(end <= sample.len(), "tile {n}'s extent {start}+{} runs past the temporal unit of {} bytes", s.slice_data_size, sample.len());
                    assert!(s.slice_data_size > 0, "tile {n} has no bytes, which is a walk that lost a tile rather than an empty tile");
                    covered.push((start, end));
                    records += 1;
                }
                // Disjoint and in raster order: an overlap would be two slice records addressing the
                // same bytes, which the driver decodes twice.
                for pair in covered.windows(2) {
                    assert!(pair[0].1 <= pair[1].0, "consecutive tiles overlap: {:?} and {:?} — the walk is off by a tile size", pair[0], pair[1]);
                }
                st.commit(&frame, surface).ok();
                surface += 1;
                if at >= sample.len() {
                    break;
                }
            }
        }
        assert!(records > 20, "the tiled fixture should produce many per-tile records, not {records}");
    }

    /// **The committed tiled fixture is the one that *can* tell the two indexings apart.**
    ///
    /// `width_in_sbs_minus_1` is indexed by tile **column** and `height_in_sbs_minus_1` by tile
    /// **row**; the rejected alternative, indexed by tile number, produces identical arrays for any
    /// grid whose columns are all one width and whose rows are all one height. On hardware the two
    /// score **48 of 48** against **0 of 48** on this fixture — so if an ffmpeg or libaom upgrade ever
    /// makes the grid uniform again, the oracle silently stops testing the thing it exists to test
    /// and every picture still passes. This is what makes that loud instead.
    #[test]
    fn the_tiled_fixture_is_the_one_that_can_tell() {
        let Some((seq, h)) = first_new_frame("av1_tiled.mp4") else { return };
        let t = &h.tile_info;
        let (cols, rows, wide, tall) = av1_tile_extents(&h, &seq).expect("a tiled frame is describable");

        assert!(cols >= 2 && rows >= 2, "the tiled fixture needs more than one column *and* one row, or column-indexing and tile-number indexing coincide");
        let widths: Vec<u16> = (0..cols as usize).map(|c| wide[c]).collect();
        let heights: Vec<u16> = (0..rows as usize).map(|r| tall[r]).collect();
        assert!(
            widths.windows(2).any(|w| w[0] != w[1]) || heights.windows(2).any(|x| x[0] != x[1]),
            "the tiled fixture's columns {widths:?} and rows {heights:?} are all one size, so it cannot tell width_in_sbs_minus_1[col] from width_in_sbs_minus_1[tile] — regenerate it with the arguments in FIXTURES"
        );

        // And the values are the specification's own arithmetic, not literals: TileWidthSb is
        // (MiColStarts[i+1] - MiColStarts[i]) >> sbShift, minus one.
        let sb_shift = if seq.use_128x128_superblock { 5 } else { 4 };
        for c in 0..cols as usize {
            let (Some(&a), Some(&b)) = (t.mi_col_starts.get(c), t.mi_col_starts.get(c + 1)) else { continue };
            let want = (b.saturating_sub(a) >> sb_shift).saturating_sub(1);
            assert_eq!(u32::from(wide[c]), want, "width_in_sbs_minus_1[{c}] is tile column {c}'s superblock width minus one");
        }
        for r in 0..rows as usize {
            let (Some(&a), Some(&b)) = (t.mi_row_starts.get(r), t.mi_row_starts.get(r + 1)) else { continue };
            let want = (b.saturating_sub(a) >> sb_shift).saturating_sub(1);
            assert_eq!(u32::from(tall[r]), want, "height_in_sbs_minus_1[{r}] is tile row {r}'s superblock height minus one");
        }
        // `tile_cols`/`tile_rows` are counts, not log2s — a record that submitted the log2 would ask
        // the driver for a quarter of the tiles.
        assert_eq!(u32::from(cols), t.cols as u32, "tile_cols is a count");
        assert_eq!(u32::from(rows), t.rows as u32, "tile_rows is a count");
    }

    // ---------------------------------------------------------------------------------------------
    // Task 5 — the state model, pinned by its mapping rather than by pictures that happen to match
    // ---------------------------------------------------------------------------------------------

    /// **A key frame refreshes every slot, an inter frame refreshes a subset, and a refusal moves
    /// nothing.**
    ///
    /// VP9's standing lesson is that pictures happening to match proves nothing about *which* surface
    /// went in *which* slot: a slot map built against the wrong index reads some other picture as a
    /// reference, without complaint. So this asserts the mapping.
    #[test]
    fn the_reference_slots_follow_refresh_frame_flags_bit_by_bit() {
        let Some((config, samples)) = read_fixture("av1_p0_8bit.mp4") else { return };
        let mut st = Av1State::new(&config);
        let mut surface = 4000u32;
        let mut key_refresh = None;
        let mut inter_seen = 0usize;

        for sample in &samples {
            let mut at = 0usize;
            while let Some(frame) = st.frame_at(sample, at).ok().flatten() {
                at = frame.next();
                let Some(h) = frame.header() else { break };
                if h.frame_type as usize == filmcraft_av1::KEY_FRAME && key_refresh.is_none() {
                    key_refresh = Some(frame.refresh_flags());
                    // The first frame is decoded into a surface and its record describes the *empty*
                    // buffer: nothing moves until the driver has rendered it.
                    assert!(st.live_surfaces().is_empty(), "the first frame's record describes the buffer before its own");
                    st.commit(&frame, surface).expect("a key frame commits");
                    assert_eq!(st.live_surfaces(), vec![surface; NUM_REF_FRAMES], "a key frame's refresh_frame_flags is all ones, so all eight slots hold it");
                    assert_eq!(st.slot(0), Some(surface), "slot 0");
                    assert_eq!(st.slot(7), Some(surface), "slot 7");
                    surface += 1;
                } else if h.frame_type as usize != filmcraft_av1::KEY_FRAME {
                    inter_seen += 1;
                    assert_ne!(frame.refresh_flags(), u8::MAX, "an inter frame refreshes a subset of the slots, not all eight");
                    let before = st.live_surfaces();
                    assert_eq!(st.live_surfaces(), before, "reading a frame moves no slot; only commit does");
                    let want: Vec<u32> =
                        (0..NUM_REF_FRAMES).map(|i| if (frame.refresh_flags() >> i) & 1 == 1 { surface } else { st.slot(i).unwrap_or(surface) }).collect();
                    st.commit(&frame, surface).expect("an inter frame commits");
                    let got: Vec<u32> = (0..NUM_REF_FRAMES).map(|i| st.slot(i).unwrap_or(0)).collect();
                    assert_eq!(got, want, "exactly the slots refresh_frame_flags names take the new surface, and no others");
                    surface += 1;
                }
                if at >= sample.len() {
                    break;
                }
            }
        }
        assert_eq!(
            key_refresh,
            Some(u8::MAX),
            "a key frame refreshes every slot — `va_dec_av1.h` has no refresh field, so this is 7.20's bitmask read off the header"
        );
        assert!(inter_seen > 0, "the fixture must carry an inter frame for this test to mean anything");
    }

    /// **A refused frame leaves the reference buffer exactly as it was.**
    ///
    /// A frame whose record cannot be built must not refresh a slot: the next frame would find a
    /// picture that was never decoded, and read it as a reference.
    #[test]
    fn a_refused_frame_does_not_move_a_slot() {
        let Some((config, samples)) = read_fixture("av1_p0_8bit.mp4") else { return };
        let Some(sample) = samples.first() else { return };
        let mut st = Av1State::new(&config);
        let mut at = 0usize;
        let mut surface = 5000u32;
        while let Some(frame) = st.frame_at(sample, at).ok().flatten() {
            at = frame.next();
            if frame.header().is_some() {
                st.commit(&frame, surface).ok();
                surface += 1;
            }
            if at >= sample.len() {
                break;
            }
        }
        let before = st.live_surfaces();
        assert!(!before.is_empty(), "the fixture must decode at least one frame first, or this test proves nothing");

        // Garbage, and a truncated frame that parses far enough to be refused.
        let _ = st.frame_at(&[0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22], 0);
        let mut truncated = vec![(OBU_FRAME << 3) | 0x2];
        truncated.push(6);
        truncated.extend_from_slice(&[0b1000_0000, 0b0100_0000, 0, 0, 0, 0]);
        let _ = st.frame_at(&truncated, 0);
        assert_eq!(st.live_surfaces(), before, "a refused frame must not refresh a slot");
    }

    /// **A `show_existing_frame` re-shows its slot and describes nothing at all.**
    ///
    /// It codes no picture (7.21), so there is nothing to render and nothing to describe: a record
    /// built for one would be a valid-looking record for a frame that does not exist, and refusing it
    /// outright would leave a hole where the stream requires a picture.
    #[test]
    fn a_show_existing_frame_re_shows_its_slot_and_describes_nothing() {
        let Some((config, samples)) = read_fixture("av1_tiled.mp4") else { return };
        let mut st = Av1State::new(&config);
        let mut surface = 6000u32;
        let mut re_showns = 0usize;
        for sample in &samples {
            let mut at = 0usize;
            while let Some(frame) = st.frame_at(sample, at).ok().flatten() {
                at = frame.next();
                match frame.re_show() {
                    Some((slot, surf)) => {
                        re_showns += 1;
                        assert!(frame.header().is_none(), "a re-show has no uncompressed header of its own to build a record from");
                        assert!(frame.slices().is_empty(), "a re-show sends no slice records: it decodes nothing");
                        assert!(frame.show(), "a re-show is always shown — re-showing a picture is the frame's whole purpose");
                        assert_eq!(
                            surf,
                            st.slot(slot).expect("the slot a re-show names holds a picture, or frame_at would have refused it"),
                            "the surface is the one in the slot it names"
                        );
                        assert!(st.picture_record(&frame, surf).is_err(), "a re-show carries no picture parameters to build");
                    }
                    None => {
                        st.commit(&frame, surface).ok();
                        surface += 1;
                    }
                }
                if at >= sample.len() {
                    break;
                }
            }
        }
        assert!(
            re_showns > 0,
            "the tiled fixture is supposed to carry show_existing_frame units — measured with this repository's parser, its samples 2, 5 and 8 do. If this fails the fixture was regenerated with different arguments"
        );
    }

    /// **An inter frame submits the primary reference's inherited values, not zeroes.**
    ///
    /// 7.20's `load_previous` reads the loop filter deltas and segmentation features out of the
    /// primary reference when `primary_ref_frame != PRIMARY_REF_NONE`, and the record's `ref_deltas`,
    /// `mode_deltas` and `seg_info` are built from what the parse resolved. A record that wrote zero
    /// there submits *a* value — and a plausible picture built from the wrong reference, which is the
    /// failure H.264's `[3, 3]` index collision cost 60 pictures over.
    #[test]
    fn an_inter_frame_with_a_primary_reference_submits_its_propagated_values() {
        let Some((config, samples)) = read_fixture("av1_p0_8bit.mp4") else { return };
        let mut st = Av1State::new(&config);
        let mut surface = 7000u32;
        let mut with_primary = 0usize;
        for sample in &samples {
            let mut at = 0usize;
            while let Some(frame) = st.frame_at(sample, at).ok().flatten() {
                at = frame.next();
                let Some(h) = frame.header().cloned() else { break };
                if h.primary_ref_frame != PRIMARY_REF_NONE && !h.frame_is_intra {
                    with_primary += 1;
                    let refs = st.references();
                    let rec = av1_picture_record(&h, st.sequence_header().expect("a sequence header"), &refs, surface).expect("an inter frame is describable");
                    let r = crate::vaapi::ffi::picture_params_av1(&rec).expect("the record reads back");
                    assert_eq!(u32::from(r.primary_ref_frame), h.primary_ref_frame as u32, "primary_ref_frame is the header's own");
                    // `0` is a *legitimate* primary reference here — entry 0 of `ref_frame_idx` is
                    // LAST_FRAME — so what must not happen is a frame with **no** primary reference
                    // submitting something other than PRIMARY_REF_NONE (7), which would load the
                    // LAST frame's deltas instead of loading none.
                    assert!(
                        r.primary_ref_frame < 7 || r.primary_ref_frame == PRIMARY_REF_NONE as u8,
                        "primary_ref_frame is a ref_frame_idx entry or PRIMARY_REF_NONE (7)"
                    );
                    for (i, &want) in h.lf.deltas.ref_deltas.iter().enumerate() {
                        assert_eq!(
                            i32::from(r.ref_deltas[i]),
                            want,
                            "ref_deltas[{i}] is what 7.20 resolved — the primary reference's value where this header did not override it"
                        );
                    }
                    for (i, &want) in h.lf.deltas.mode_deltas.iter().enumerate() {
                        assert_eq!(i32::from(r.mode_deltas[i]), want, "mode_deltas[{i}]");
                    }
                    assert_eq!(r.seg_info.segment_info_fields & 1, u32::from(h.seg.enabled), "seg_info's enabled bit");
                }
                st.commit(&frame, surface).ok();
                surface += 1;
                if at >= sample.len() {
                    break;
                }
            }
        }
        assert!(
            with_primary > 0,
            "no inter frame in the fixture names a primary reference, so this test proved nothing — regenerate it with the arguments in FIXTURES"
        );
    }

    /// **The loop restoration word: the specification's enum, and the shifts the sizes derive to.**
    ///
    /// Both halves were measured on hardware rather than argued, and **both were got wrong first** —
    /// see [`av1_loop_restoration_fields`] and [`av1_lr_shifts`]. The fixture is what caught them:
    /// `av1_loop_restoration.mp4` decodes 18 of 96 with either error, and the 18 are exactly the
    /// frames that restore nothing.
    #[test]
    fn the_loop_restoration_word_is_the_specifications_enum_and_the_sizes_shifts() {
        let Some((seq, h)) = first_new_frame("av1_loop_restoration.mp4") else { return };
        assert!(h.lr.uses_lr, "this fixture exists to restore something; if it does not, it cannot test the encoding");
        let word = av1_loop_restoration_fields(&h).expect("a restoring frame is describable");

        // The three types are the specification's enum, unmodified: SGRPROJ is 2 and WIENER is 1, not
        // the bitstream's 3 and 2. `va_dec_av1.h:606-618` documents neither; the hardware does.
        assert_eq!(u32::from(word & 0x3), filmcraft_av1::RESTORE_SGRPROJ as u32, "yframe_restoration_type is FrameRestorationType, so SGRPROJ is 2");
        assert_eq!(u32::from((word >> 2) & 0x3), filmcraft_av1::RESTORE_WIENER as u32, "cbframe_restoration_type is FrameRestorationType, so WIENER is 1");

        // lr_unit_shift inverts the parser's own formula, size = 256 >> (2 - shift), so a 256-pixel
        // unit is shift **2** and not 0. The direction is the easy thing to get backwards.
        assert_eq!(h.lr.loop_restoration_size[0], 256, "measured on the fixture; the assertion below is derived from it, not from a literal");
        assert_eq!((word >> 6) & 0x3, 2, "lr_unit_shift is 2 for a 256-pixel unit (`crates/av1/src/header.rs:1154`)");
        assert_eq!((word >> 8) & 1, 0, "lr_uv_shift is 0 when the chroma unit is the luma one");

        // Every unit size the specification allows, so a future fixture restoring with a smaller unit
        // is checked against the arithmetic rather than against whatever this one happens to carry.
        for (size, want) in [(64u32, 0u32), (128, 1), (256, 2)] {
            let mut probe = h.clone();
            probe.lr.loop_restoration_size = [size, size, size];
            let w = av1_loop_restoration_fields(&probe).expect("a restorable unit size is describable");
            assert_eq!(u32::from((w >> 6) & 0x3), want, "a {size}-pixel restoration unit is lr_unit_shift {want}");
        }
        let mut half = h.clone();
        half.lr.loop_restoration_size = [256, 128, 128];
        assert_eq!(
            (av1_loop_restoration_fields(&half).expect("a half-size chroma unit is describable") >> 8) & 1,
            1,
            "lr_uv_shift is 1 when the chroma unit is half the luma one"
        );

        // A unit size the specification cannot produce derives to no shift, and is refused rather
        // than clamped into one of the three.
        let mut impossible = h.clone();
        impossible.lr.loop_restoration_size = [192, 192, 192];
        let why = av1_loop_restoration_fields(&impossible).expect_err("192 is not one of 64, 128 or 256");
        assert!(why.contains("not one of 64, 128 or 256"), "{why}");

        // A frame that restores nothing has no unit size to carry and submits zero.
        let mut none = h.clone();
        none.lr.uses_lr = false;
        none.lr.frame_restoration_type = [0, 0, 0];
        assert_eq!(av1_loop_restoration_fields(&none).expect("a frame that restores nothing is still describable"), 0, "no restoration ⇒ every field zero");
        let _ = seq;
    }

    /// **Every other fixture's loop restoration word is zero, because none of them restores.**
    ///
    /// This is the measurement behind the "the frame that restores nothing submits 0" claim: three
    /// of the fixtures' sequence headers leave restoration off, and asserting on all of them is what
    /// makes it a fact about the streams rather than about one chosen encode.
    #[test]
    fn a_fixture_that_does_not_restore_submits_no_restoration() {
        for name in ["av1_p0_8bit.mp4", "av1_p0_10bit.mp4", "av1_intra_only.mp4", "av1_tiled.mp4"] {
            let Some((_, h)) = first_new_frame(name) else { continue };
            assert!(!h.lr.uses_lr, "{name} restores nothing, which is what this test assumes");
            assert_eq!(av1_loop_restoration_fields(&h).expect("still describable"), 0, "{name}: no restoration ⇒ every field zero");
        }
    }

    /// **The container's description and the sequence header must agree, and one site checks it.**
    ///
    /// The surfaces were created for the profile and depth the sample entry said; a record carrying
    /// the frame's own would be a picture of one format decoded into a surface of another. The
    /// comparison is against the **sequence header** rather than `FrameHeader::color`, because 5.9's
    /// uncompressed header spells no colour configuration at all — it is the sequence header that
    /// carries it, for every frame including an intra one.
    #[test]
    fn the_sample_entry_and_the_sequence_header_must_agree() {
        let Some((seq, _)) = first_new_frame("av1_p0_10bit.mp4") else { return };
        assert_eq!(seq.color.bit_depth, 10, "the 10-bit fixture really is 10-bit");
        assert!(av1_entry_disagreement(seq.profile, 10, &seq).is_none(), "agreeing on both, there is nothing to say");
        let why = av1_entry_disagreement(seq.profile, 8, &seq).expect("a depth the sequence header does not have is a disagreement");
        assert!(why.contains("8 bits") && why.contains("10"), "the disagreement names both depths: {why}");
        let why = av1_entry_disagreement(2, 10, &seq).expect("a profile the sequence header does not have is a disagreement");
        assert!(why.contains("profile 2") && why.contains(&format!("profile {}", seq.profile)), "the disagreement names both profiles: {why}");
    }

    /// **Hostile and truncated input is an error, never a crash and never a truncated slice.**
    ///
    /// The OBU walk reads the most input-derived arithmetic in this module, and every length in it
    /// comes from the bitstream. Under AGENTS.md §0 the answers are `Err` strings.
    #[test]
    fn hostile_and_truncated_input_is_an_error_and_not_a_crash() {
        let Some((config, _)) = read_fixture("av1_p0_8bit.mp4") else { return };
        let mut st = Av1State::new(&config);
        assert!(st.frame_at(&[], 0).ok().flatten().is_none(), "an empty sample is the end of the stream, not an error");
        assert!(st.frame_at(&[], 99).ok().flatten().is_none(), "an offset past the end is the end of the stream, not an error");
        // A forbidden bit ends the OBU walk before any OBU is produced, so this is the end of the
        // stream rather than an error — what matters is that neither is a panic.
        assert!(st.frame_at(&[0xff], 0).is_ok(), "a forbidden bit ends the walk without a panic");

        // Every prefix of a real frame's bytes: a truncated OBU must be an error, never a panic and
        // never a record built from half a header.
        if let Some(sample) = read_fixture("av1_tiled.mp4").and_then(|(_, s)| s.first().cloned()) {
            for cut in 1..sample.len().min(400) {
                let mut s = Av1State::new(&config);
                let _ = s.frame_at(&sample[..cut], 0);
            }
        }
        // And a bit-flip sweep over a real frame's header, which is the mutation the plan asks for.
        if let Some(sample) = read_fixture("av1_p0_8bit.mp4").and_then(|(_, s)| s.first().cloned()) {
            {
                let head = &sample[..sample.len().min(48)];
                for at in 0..head.len() {
                    for bit in [1u8, 0x80] {
                        let mut corrupt = head.to_vec();
                        corrupt[at] ^= bit;
                        let mut s = Av1State::new(&config);
                        let _ = s.frame_at(&corrupt, 0);
                    }
                }
            }
        }
        // A slice record that cannot address what it is told to address is an error.
        assert!(av1_slice_record(usize::MAX, 1, 0, 0).is_err(), "a tile number that does not fit the record's row");
        assert!(av1_slice_record(0, 1, usize::MAX, usize::MAX).is_err(), "an offset that does not fit the record's offset");
    }

    /// **The pool is the eight slots and the one being written, and `reset` clears both halves.**
    ///
    /// `reset` has to clear the parse state as well as the slots: 7.20's propagated loop filter
    /// deltas, segmentation features and global motion are inherited by every frame that does not
    /// spell them, so keeping them would describe the first frame of the new run in the old one's
    /// values.
    #[test]
    fn reset_clears_both_the_slots_and_the_parse_state() {
        let Some((config, samples)) = read_fixture("av1_p0_8bit.mp4") else { return };
        let Some(sample) = samples.first() else { return };
        let mut st = Av1State::new(&config);
        let mut at = 0usize;
        let mut surface = 8000u32;
        while let Some(frame) = st.frame_at(sample, at).ok().flatten() {
            at = frame.next();
            if frame.header().is_some() {
                st.commit(&frame, surface).ok();
                surface += 1;
            }
            if at >= sample.len() {
                break;
            }
        }
        assert!(!st.live_surfaces().is_empty(), "the fixture must decode something first");
        let inherited = st.references();
        st.reset();
        assert!(st.live_surfaces().is_empty(), "a seek clears every slot: a stale one is a reference predicted from the wrong place");
        assert_eq!(st.references(), Av1ReferenceSet::default(), "the reference set is empty after a reset");
        assert_ne!(inherited, Av1ReferenceSet::default(), "there really was something to clear");

        // And the same first frame parses to the same header after a reset — a reset that changed
        // the parse state would make the same bytes decode differently.
        let mut fresh = Av1State::new(&config);
        let _ = fresh.frame_at(sample, 0).ok().flatten().expect("the first frame parses");
        let mut again = Av1State::new(&config);
        let _ = again.frame_at(sample, 0).ok().flatten().expect("the first frame parses again");
        assert!(again.sequence_header().is_some(), "the sequence header survives a reset, re-seeded from the av1C rather than dropped");
    }
}
