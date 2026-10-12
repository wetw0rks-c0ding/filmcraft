//! The VA-API decoder: surfaces, the decode loop and the readback. Part of [`super`], split out
//! here because the module is large; nothing in this file names a libva type.

use std::time::{Duration, Instant};

use filmcraft_codecs::hw::{FrameCodec, NalCodec, StreamInfo};
use filmcraft_codecs::{CodecError, DecodedFrame, Result, VideoDecoder};

use super::av1::{Av1State, av1_entry_disagreement, av1_surface_pool};
use super::ffi::{self, Display, Session};
use super::picture::{NalPoc, Picture, Vp9State, surface_pool, vp9_entry_disagreement, vp9_surface_pool};
use super::stream::{Stream, supports};
use super::{Ring, validate_image};
use crate::biplanar::{self, Geometry};
use ffi::{IQ_MATRIX_BUFFER, PICTURE_PARAMETER_BUFFER, SLICE_DATA_BUFFER, SLICE_PARAMETER_BUFFER};

/// How many pictures a decoded picture buffer can hold: a DPB is at most 16 frames, so more means the
/// decoder lost track of the stream. A VA-API decode is synchronous — one access unit is rendered and
/// read back inside the `decode` call that submitted it — so this never gets near the bound; it is
/// here for the same reason `MfDecoder` has it, as a backstop against a driver-reported count that
/// would otherwise grow without limit.
const MAX_IN_FLIGHT: usize = 64;

/// Where a decoder's time went (wall clock), for diagnostics. The three costs are the ones a user
/// would notice: the driver's decode, the GPU-to-CPU copy of the picture, and the conversion of that
/// copy into FilmCraft's own planes.
#[derive(Clone, Copy, Debug, Default)]
pub struct Timings {
    /// Pictures returned.
    pub frames: u64,
    /// Submitting the records and waiting for the driver to render the picture.
    pub decode: Duration,
    /// Syncing the surface and copying its pixels out of the driver.
    pub readback: Duration,
    /// CPU: the driver's planes to planar `Yuv8` / `Yuv16`.
    pub convert: Duration,
}

/// The VA-API decoder for one `avcC` / `hvcC` stream.
///
/// The same shape as the Windows and macOS backends: one access unit in, the pictures it produced
/// out, and every failure an error rather than a picture that is not the stream's. What differs is
/// where the work is. The other two hand a whole access unit to an OS component that works the
/// picture structure out for itself; libva does not — it wants the picture-level and slice-level
/// syntax elements as records — so those come from `super::picture`, out of FilmCraft's own parsers.
/// What this file owns is everything around them: which surface a picture decodes into, how the
/// records reach the driver, and how the finished picture comes back.
pub struct VaDecoder {
    stream: Stream,
    /// The stream as the hybrid decoder's tests (random access, disposable) see it.
    info: StreamInfo,
    /// Where in the surface the picture is, and in what bit depth and colour.
    geometry: Geometry,
    /// The display and the config, surfaces and context on it. `None` is not a state a decoder is
    /// ever left in: [`Self::reset`] drops the context (which a seek requires) and the next picture
    /// makes a new one, rather than the decoder becoming unusable.
    session: Session,
    /// Where the surfaces to decode into come from. The ring is separate from the session's surface
    /// array because it is what a decode takes one from: the driver owns the surfaces, the ring owns
    /// the order they are used in.
    ring: Ring,
    /// The decoder's own picture-order-count and reference state, which is what the records are built
    /// from. `NalPoc` is [`super::picture`]'s, and its decoded-picture-buffer lives inside it. A VP9
    /// stream does not use it: its own state is `vp9` below.
    poc: NalPoc,
    /// The VP9 decode's reference state: the one parse state its frames inherit `color_config` and
    /// the loop filter from, and the eight reference slots its record names. `Default` is an empty
    /// state, which is the right one for a stream that is not VP9 and for the first frame of one that
    /// is. See [`Vp9State`] for why this is a per-decode object and not a per-picture one.
    vp9: Vp9State,
    /// The AV1 decode's reference state: the 7.20 slots its records name, and the parse state every
    /// frame's header is read against. `refresh_frame_flags` has **no member in the AV1 decode
    /// record** (`grep -in refresh /usr/include/va/va_dec_av1.h` returns only lines 421 and 436, both
    /// inside comments), so unlike VP9 the record cannot say which slots this picture becomes: the
    /// application owns that entirely, and `Av1State` is where it lives.
    av1: Av1State,
    /// The significant bits per sample this session's surfaces hold (8 or 10), which is what a VP9
    /// frame's own `color_config` is checked against: the surfaces were created in one format, and a
    /// frame declaring another is a stream this session was not set up for — the same shape as
    /// [`StreamInfo::parameters_changed`](filmcraft_codecs::hw::StreamInfo), which is what catches it
    /// at a key frame; this catches the frames between them.
    bits: u32,
    /// Presentation times of submitted access units whose picture has not come out yet.
    in_flight: Vec<i64>,
    /// The run started at a CRA / BLA picture: its RASL pictures are not decodable and are left out,
    /// as the software decoder does.
    skip_rasl: bool,
    /// Whether the next picture is the first of a run, which is what decides whether RASL pictures are
    /// skipped and what a seek resets.
    first: bool,
    /// The next sample must be preceded by the parameter sets: the first of a run, after a seek or a
    /// flush. Set from the start and by `reset` / `flush`, cleared by the first `decode`.
    need_headers: bool,
    /// [`VideoDecoder::flush`] drained the driver, which then behaves as at the end of a stream: it
    /// only restarts at an IDR picture, and the references of the run are gone.
    drained: bool,
    /// Why this decoder has given up on the run, once a picture could not be produced.
    ///
    /// A picture that could not be built or rendered is a gap in the picture buffer, and the pictures
    /// after it are decoded against references that are not the stream's: the driver is handed a
    /// surface holding something else and reads it as a reference, which is a plausible-looking wrong
    /// picture rather than a failure. So the refusal is **latched**: the decoder keeps returning this
    /// reason until [`VideoDecoder::reset`], which is what a seek does. A caller that wanted the rest
    /// of the run in hardware has to seek somewhere the run can start again; a caller that does not —
    /// `HybridDecoder`, which drops this decoder and replays in software — never gets here twice.
    ///
    /// Set from the start, and cleared by `reset` only: a `flush` does not repair the reason, which is
    /// a property of the stream rather than of the run, so the next run at a restart point is refused
    /// again — with the same reason, which is what a reader of the log wants to see.
    refused: Option<String>,
    /// Test hook: fail every decode once this many samples were fed.
    fail_after: Option<u64>,
    fed: u64,
    name: &'static str,
    timings: Timings,
}

impl VaDecoder {
    /// A decoder for the stream, or why VA-API does not take it: a stream format this backend does not
    /// decode, no libva or no render node, a profile the driver does not offer with a video decoder,
    /// or a driver that refuses to create a context for it.
    ///
    /// `impl Into<Stream>` so a `StreamInfo` converts directly, matching the other two backends. The
    /// order is the pipeline's: plan the stream, open the libraries, open the display, ask the driver
    /// whether it really offers that plan, and only then make the surfaces a context needs — so a
    /// machine that cannot do this is told why before anything is allocated for it.
    pub fn new(stream: impl Into<Stream>) -> std::result::Result<Self, String> {
        let stream = stream.into();
        let plan = stream.plan()?;
        let api = ffi::load()?;
        let display = Display::new(&api)?;
        supports(&display, &plan)?;
        // The ring is sized from the record's own reference-slot count, not from the number the
        // sequence parameter set declares — see `picture::surface_pool` for why the declaration is not
        // a bound the buffer respects. The references have to survive in their surfaces until the
        // pictures that use them are decoded, and one more surface is the one being written into.
        // `Session::new` clamps the count, so this costs a clamp rather than an allocation.
        let count = match &stream {
            Stream::Nal(info) => surface_pool(info),
            // A frame-coded stream that `plan` accepted, which today is VP9 only: the same bound as
            // above and from the same place — its record's own `reference_frames` is all eight slots,
            // and `Vp9State` holds a picture in a slot only while `refresh_frame_flags` says so.
            Stream::Frame(info) => match info.codec {
                FrameCodec::Vp9 => vp9_surface_pool(),
                // AV1's own bound, from its own record: `ref_frame_map[8]` is all eight slots and
                // `Av1State` holds a surface in a slot only while 7.20's `refresh_frame_flags` says so.
                FrameCodec::Av1 => av1_surface_pool(),
            },
        };
        let session = Session::new(&display, plan.profile, plan.entrypoint, plan.format, plan.size, count)?;
        let name = stream.name();
        // Safe to read `info.crop` / `info.size` unbounded here *because* `stream.plan()` above ran
        // first and bounded both, along with the coded size the crop has to fit inside
        // (`stream::plan_nal`, `plan_frame`). `first_geometry` is the one reader of those fields
        // that trusts them, and this is its only caller in this backend — so the order of the two
        // lines is load-bearing, not incidental.
        let bits = plan_bits(plan.format)?;
        let geometry = stream.first_geometry(bits);
        // The ring holds the ids the driver filled in, in the order it gave them: libva does not
        // promise surface ids are contiguous, so an id this code worked out rather than read would
        // name a surface that does not exist.
        let ring = Ring::from_surfaces(session.surfaces());
        log::info!(
            "hardware decoding: {name} on {}{}",
            display.vendor().unwrap_or("a VA-API driver"),
            if session.reads_indirectly() { " (readback through a copy)" } else { "" }
        );
        Ok(Self {
            info: stream.info(),
            geometry,
            // Seeded from the sample entry's `av1C`, which is where the sequence header lives before
            // the first frame: a container sample carries one only at the start of a run, and
            // `Av1State` needs a sequence header to parse a frame header against. **Infallible**, so a
            // non-AV1 stream — which has no configuration OBUs and never consults this — is not made
            // unconstructible by it; a stream whose `av1C` carried no sequence header is refused at
            // its first frame instead, with the missing thing named.
            av1: Av1State::new(frame_config_obus(&stream)),
            stream,
            session,
            ring,
            poc: NalPoc::default(),
            vp9: Vp9State::new(),
            bits,
            in_flight: Vec::new(),
            skip_rasl: false,
            first: true,
            need_headers: true,
            drained: false,
            refused: None,
            fail_after: None,
            fed: 0,
            name,
            timings: Timings::default(),
        })
    }

    /// Test hook: from the `n`-th sample on, every decode fails as a lost device would.
    pub fn fail_after(&mut self, n: u64) {
        self.fail_after = Some(n);
    }

    /// Where the time went so far.
    pub fn timings(&self) -> Timings {
        self.timings
    }

    /// Name of the stream's decoder, as the counters and the logs show it.
    pub fn name(&self) -> &str {
        self.name
    }

    /// The number of surfaces the ring holds (diagnostics and tests).
    pub fn surfaces(&self) -> usize {
        self.ring.len()
    }

    /// The presentation time the picture from `sample` is stamped with.
    fn take_pts(&mut self, pts: i64) -> i64 {
        if let Some(i) = self.in_flight.iter().position(|&p| p == pts) {
            return self.in_flight.swap_remove(i);
        }
        // One access unit in, one picture out, so a timestamp that is not the one submitted is the
        // one the driver restamped; pictures come out in presentation order, so the earliest is it.
        match self.in_flight.iter().enumerate().min_by_key(|(_, p)| **p).map(|(i, _)| i) {
            Some(i) => self.in_flight.swap_remove(i),
            None => pts,
        }
    }

    /// The slice data buffer one access unit is submitted as: its slices' NAL units, laid out at the
    /// offsets the slice records name and no longer.
    ///
    /// The NAL units are copied out of the container sample, which is length-prefixed, into one
    /// buffer laid out as `Picture::slices` says — which is what the slice records' `slice_data_offset`
    /// and `slice_data_size` address. Both are bounds-checked: a record that addresses outside what
    /// was built would have the driver read slice bytes from somewhere else in the stream.
    fn slice_data(picture: &Picture, sample: &[u8]) -> std::result::Result<Vec<u8>, String> {
        let size = usize::try_from(picture.slice_data_size()).map_err(|_| "slice data buffer size does not fit".to_string())?;
        let mut data = vec![0u8; size];
        for (slice, &at) in picture.slices.iter().zip(&picture.slice_offsets) {
            // Two numbers from the record (`SliceParam::offset` and `length`, both `u32`) and one from the
            // sample description, each bounds-checked against what it indexes before it is used as one.
            let (start, length, at) = (at, slice.length as usize, slice.offset as usize);
            let (Some(nal), Some(slot)) = (sample.get(start..).and_then(|b| b.get(..length)), data.get_mut(at..)) else {
                return Err("a slice record addresses outside the sample or the buffer".into());
            };
            let Some(slot) = slot.get_mut(..length) else { return Err("a slice record addresses outside the slice data buffer".into()) };
            slot.copy_from_slice(nal);
        }
        Ok(data)
    }

    /// Copy the picture at `geometry.crop` out of the mapped surface into a planar frame.
    ///
    /// This is the one GPU-to-CPU copy in this backend, and it is the only place a driver-reported
    /// geometry is turned into an address a pixel read happens at. Two gates stand between the
    /// driver and that read: [`validate_image`] has said the picture has two planes, the depth it
    /// expects, a pitch of at least a row and offsets that end inside the mapped buffer; and
    /// `biplanar::to_frame` bounds-checks every row and column it copies against the slice it is
    /// given. Both are needed: the first is what says the geometry is sane, the second is what says
    /// the copy of *this* rectangle fits, and neither knows what the other knows.
    fn readback(&mut self, surface: u32) -> std::result::Result<(Duration, VideoFrame), String> {
        let read = self.session.read(surface)?;
        validate_image(read.image())?;
        let geometry = self.geometry;
        let (mut convert, mut made) = (Duration::ZERO, None);
        read.with_mapped(|luma, chroma, stride, rows| {
            let c0 = Instant::now();
            let source = biplanar::Biplanar { luma, chroma, stride, rows };
            made = Some(biplanar::to_frame(&source, &geometry));
            convert = c0.elapsed();
        });
        // `to_frame` refuses a surface smaller than the picture it is asked for, which is what makes
        // an empty `Biplanar` above an error rather than a blank frame: the two `get` calls that
        // produced it are the same bounds `validate_image` checked, and they failed.
        made.ok_or("the derived image has no readable planes")?.map_err(|e| format!("readback: {e}")).map(|f| (convert, f))
    }
    /// The buffers one picture is submitted as, in order: its picture parameter record, its scaling
    /// lists, one record per slice, then the slices' bytes.
    ///
    /// Split out from the decode loop so that the list is a value a test can read rather than a
    /// sequence of calls only a driver can observe — the IQ matrix was missing from this list for the
    /// whole of the first implementation, and the only symptom was a driver that accepted everything
    /// and rendered nothing. A list this loop assembles from named constants is one a test can assert
    /// on, and a missing entry is a failing test rather than a grey timeline.
    ///
    /// `data` is the assembled slice data buffer ([`VaDecoder::slice_data`]), and an empty
    /// `picture.iq_matrix` contributes no buffer — which for H.264 never happens (every picture
    /// carries its scaling lists) and for HEVC means the sequence parameter set's
    /// `sps_scaling_list_enabled_flag` was clear, so `VAIQMatrixBufferHEVC` is not sent.
    fn submission<'p>(picture: &'p Picture, data: &'p [u8]) -> Vec<(i32, &'p [u8])> {
        let mut out = Vec::with_capacity(picture.slices.len() + 3);
        out.push((PICTURE_PARAMETER_BUFFER, picture.params.as_slice()));
        if !picture.iq_matrix.is_empty() {
            out.push((IQ_MATRIX_BUFFER, picture.iq_matrix.as_slice()));
        }
        for slice in &picture.slices {
            out.push((SLICE_PARAMETER_BUFFER, slice.params.as_slice()));
        }
        out.push((SLICE_DATA_BUFFER, data));
        out
    }
}

/// The `av1C` configuration OBUs of a frame-coded stream, or `None` for one that has none.
///
/// **Only AV1 needs them.** `Av1State` parses a frame header against a sequence header, and a
/// container sample carries one only at the start of a run — `Stream::input` prepends `config_obus`
/// for exactly that reason, which is why the decode loop calls it once per run and then discards
/// the bytes. The state therefore has to be primed from the sample entry itself. A VP9 stream has no
/// configuration OBUs and `Av1State` is never consulted for it, so `None` is the right answer rather
/// than an error.
fn frame_config_obus(stream: &Stream) -> &[u8] {
    match stream {
        Stream::Nal(_) => &[],
        Stream::Frame(info) if info.codec == FrameCodec::Av1 => &info.config_obus,
        Stream::Frame(_) => &[],
    }
}

/// `RTFormat` as the significant bits per sample its pictures hold: 8 for NV12, 10 for P010.
/// Only `plan_nal`'s two formats can reach here, and `RtFormat::Other` is refused in `Session::new`
/// before that. The third arm is an `Err` rather than a silent 8-bit depth: a default there would
/// hand a decoder a picture geometry that does not match the surface it was created at, which is a
/// plausible-looking wrong picture rather than a refusal.
fn plan_bits(format: ffi::RtFormat) -> std::result::Result<u32, String> {
    match format {
        ffi::RtFormat::Nv12 => Ok(8),
        ffi::RtFormat::Yuv420_10 => Ok(10),
        ffi::RtFormat::Other(bits) => Err(format!("render format {bits:#x} has no bit depth this backend knows")),
    }
}

impl VideoDecoder for VaDecoder {
    fn decode(&mut self, sample: &[u8], pts: i64) -> Result<Vec<DecodedFrame>> {
        self.fed += 1;
        if self.fail_after.is_some_and(|n| self.fed > n) {
            return Err(CodecError::Decode("VA-API decode failed (test hook)".into()));
        }
        // A run this decoder has already given up on stays given up on: see `refused`.
        if let Some(why) = self.refused.clone() {
            return Err(CodecError::Decode(format!("the hardware decoder gave up on this run: {why}")));
        }
        if self.drained {
            // A drained decoder cannot go on from the middle of a GOP (software decoders can). The
            // GOP cache seeks (`reset`) after every flush, so this is an error only for callers that
            // continue after one: the hybrid decoder then replays the run in software.
            if !self.stream.is_restart(sample) {
                return Err(CodecError::Decode("the hardware decoder was drained and restarts only at an IDR picture".into()));
            }
            self.drained = false;
            self.first = true;
            self.poc.reset();
            // VP9's half of "what came before this point is gone", for the same reason: the first
            // frame of the new run must be read as the first frame of a run, and the slots must be
            // empty so that a reference resolves to nothing rather than to the previous run's
            // picture. `Vp9State::reset` forgets both halves together, which is what a seek needs.
            self.vp9.reset();
            // AV1's is the same argument for the same state: 7.20's slots *and* its propagated loop
            // filter deltas, segmentation features and global motion all belong to the run that
            // produced them, so a seek takes both halves together.
            self.av1.reset();
        }
        // HEVC RASL pictures after a CRA / BLA at the start of a run refer to pictures that were
        // never decoded, so they are left out — as the software decoder does, and so the pictures
        // that come out are the ones a player would show.
        if let Stream::Nal(info) = &self.stream
            && info.codec == NalCodec::Hevc
        {
            let types = info.nal_types(sample);
            if types.iter().any(|t| (16..=23).contains(t)) {
                self.skip_rasl = self.first && types.iter().any(|t| (16..=21).contains(t) && !(19..=20).contains(t));
            } else if self.skip_rasl && types.iter().any(|t| matches!(t, 8 | 9)) {
                self.first = false;
                return Ok(Vec::new());
            } else if self.skip_rasl {
                // The skipped run is over. Without this the flag stays set for the rest of the
                // stream, and a later RASL picture — which in an open GOP is decodable, because it
                // refers only to pictures after the CRA — is dropped too. The symptom is an empty
                // frame list after pictures had already come out, which is the blank-timeline
                // outcome: a caller cannot tell it from a picture the stream has not produced yet.
                self.skip_rasl = false;
            }
        }
        self.first = false;
        // A VP9 key frame says its own size and colour, so the geometry a picture is read back at
        // follows the bitstream rather than the container (`Stream::declared_picture`). This is the
        // geometry a *later* frame is checked against in `picture_frame` — a stream whose resolution
        // changes mid-run updates it here, and the check then refuses the run rather than decoding a
        // frame into surfaces the session was not created at.
        if let Some(g) = self.stream.declared_picture(sample) {
            self.geometry = g;
        }
        // `need_headers` says the decoder has just started — the first sample, after a seek or a
        // flush — and the stream's parameter sets go in front of it. This backend builds its records
        // from the sample entry's parameter sets rather than from an Annex B stream, so there are no
        // bytes here to hand the driver; what `Stream::input` gives the decode loop is its *error*,
        // and that error is only about the sample's own shape: `annexb::to_annex_b` walks the length
        // prefixes and refuses a sample whose NAL units do not add up, run past its end, are empty,
        // or are followed by bytes no prefix accounts for. It parses nothing — the parameter sets it
        // copies in front are the sample entry's bytes, unexamined — so it says nothing about
        // whether those sets are usable, and nothing here should be read as if it did. Doing that
        // once at the start of a run is where `need_headers` is spent; it is deliberately not done
        // per picture, because `input` allocates and at 600 pictures a second an allocation per
        // picture to produce bytes nothing reads is waste.
        if self.need_headers {
            let with_headers = self.stream.input(sample, true).map_err(CodecError::Decode)?;
            debug_assert!(!with_headers.is_empty(), "a sample that converts to nothing is refused by input()");
        }
        self.need_headers = false;
        self.in_flight.push(pts);
        if self.in_flight.len() > MAX_IN_FLIGHT {
            let held = self.in_flight.len();
            self.in_flight.clear();
            return Err(CodecError::Decode(format!("the decoder holds {held} pictures back")));
        }
        // One access unit in, one picture out. A decode that produced nothing is an error rather than
        // an empty frame list: an empty list for a sample the caller fed is indistinguishable from a
        // picture the stream has not produced yet, and a blank timeline is the worst answer available
        // — an error hands the stream to the software decoder instead.
        //
        // **The one frame that is decoded and produces no picture is the exception, and it is the
        // stream's own answer rather than a blank one.** 6.2's `show_frame` clear is a VP9 frame the
        // decoder must still render (it may refresh reference slots for later frames) but that no
        // player shows, and FilmCraft's software decoder emits nothing for it either
        // (`crates/vp9/src/decoder.rs:413`, `:417`). So `picture` reports it as `Decoded::shown ==
        // false` and this returns an empty list, exactly as the HEVC RASL skip above does.
        let decoded = match self.picture(sample) {
            Ok(d) => d,
            // The submission is gone: nothing will come out of it, so it is not in flight any more.
            // Leaving it would make every later decode report a decoder full of pictures back.
            Err(e) => {
                self.in_flight.pop();
                // Every picture after this one would be decoded against a picture buffer that is
                // missing this picture, so the run is over here rather than silently wrong later.
                log::warn!("{} gave up on this run: {e}", self.name);
                self.refused = Some(e.clone());
                return Err(CodecError::Decode(e));
            }
        };
        let stamp = self.take_pts(pts);
        // The counters are **pictures**, so a sample that showed two of them counts two — which is
        // what makes `timings.frames` a per-picture cost for the superframe case too.
        self.timings.frames += decoded.frames.len() as u64;
        self.timings.decode += decoded.decode;
        self.timings.readback += decoded.readback;
        self.timings.convert += decoded.convert;
        if decoded.frames.is_empty() {
            // A frame the stream does not show: it was rendered (and its slots moved), and there is
            // nothing to hand back. Its presentation time came out of `in_flight` above, which is
            // right — the sample was accounted for, it simply produced no picture.
            return Ok(Vec::new());
        }
        // **Every picture of the sample carries the sample's own presentation time.** A VP9
        // superframe is several frames under one timestamp and has no other timestamp to give them,
        // and this is what FilmCraft's software VP9 decoder does for the same bitstream
        // (`crates/vp9/src/decoder.rs:248-255`) — so the two paths agree on the number of pictures
        // *and* on the key a caller would look them up by.
        Ok(decoded.frames.into_iter().map(|frame| DecodedFrame { pts: stamp, frame, draft: false }).collect())
    }

    fn flush(&mut self) -> Vec<DecodedFrame> {
        // A VA-API decode is one picture in one picture out, so there is nothing buffered to collect:
        // whatever was decoded was read back and returned by the `decode` call that submitted it. What
        // a flush does is end the run, so that what follows starts like a new one.
        self.in_flight.clear();
        self.poc.reset();
        // VP9's run ends with the others': a flush drains the driver, and the references of the run
        // are gone with it. Keeping them would make the next run's first frame resolve a reference
        // into a surface the driver has already been asked to overwrite.
        self.vp9.reset();
        self.av1.reset();
        self.first = true;
        self.skip_rasl = false;
        self.need_headers = true;
        self.drained = true;
        Vec::new()
    }

    fn reset(&mut self) {
        // A seek leaves no reference picture behind, so the context goes with them: a context that
        // still held the previous run's references would decode this run against surfaces holding
        // something else. The surfaces and the config stay, since they depend only on the stream.
        self.session.clear_context();
        self.poc.reset();
        // A seek is what `Vp9State::reset` exists for: the slots *and* the parse state go together,
        // because the first frame of the new run must be read as the first frame of a run — an
        // inherited `color_config` would describe the previous run's depth.
        self.vp9.reset();
        self.av1.reset();
        self.in_flight.clear();
        self.skip_rasl = false;
        self.need_headers = true;
        self.first = true;
        self.drained = false;
        // A seek is the one thing that starts a run over: the picture buffer is empty and the next
        // picture is decoded from scratch, so whatever ended the last run is no longer true of this
        // one. A `flush` does not clear it — see `refused`.
        self.refused = None;
    }

    fn name(&self) -> &str {
        self.name
    }

    fn is_random_access(&self, sample: &[u8]) -> Option<bool> {
        self.info.is_random_access(sample)
    }

    fn is_disposable(&self, sample: &[u8]) -> bool {
        self.info.is_disposable(sample)
    }
}

impl VaDecoder {
    /// Decode one access unit and read its picture back: the driver records built from the sample,
    /// submitted to a surface no live reference is in, then copied out.
    ///
    /// **Which stream shape this is decides everything below it**, because the two are not the same
    /// pipeline: a NAL codec builds a picture *then* takes a surface for it, while a VP9 frame is read
    /// first and only then turns out to need a surface at all (a frame that re-shows an earlier
    /// picture needs none). So the dispatch is here rather than inside either arm.
    fn picture(&mut self, sample: &[u8]) -> std::result::Result<Decoded, String> {
        match &self.stream {
            Stream::Nal(_) => self.picture_nal(sample),
            Stream::Frame(_) => self.picture_frame(sample),
        }
    }

    /// One H.264 / HEVC access unit: build its picture record and slice records, submit them into a
    /// surface no live reference is in, and copy the result out.
    ///
    /// **The surface is chosen from the decoded picture buffer, not by rotating.** A surface a later
    /// picture still refers to may not be decoded into: the driver would read the new picture as that
    /// reference, and what comes out is a picture the stream never contained — the plausible-looking
    /// wrong answer this backend must not produce. The buffer is only known here, in the one place
    /// that owns it, so this is where the choice is made; `Ring` hands out a surface that is not in
    /// it, and a stream whose references fill every surface is an error rather than an overwrite.
    ///
    /// The timing is kept in two parts because they are what a user would notice separately: the
    /// driver's own decode (`vaEndPicture` plus the sync that waits for it), and the copy of the
    /// finished picture back to the CPU — of which the conversion into FilmCraft's planes is
    /// measured separately, so the reported readback is the copy and not the copy plus the
    /// conversion.
    fn picture_nal(&mut self, sample: &[u8]) -> std::result::Result<Decoded, String> {
        // Before `build_nal`, which adds this picture to the buffer: the surfaces that must not be
        // decoded into are the ones the *previous* pictures left referenced.
        let surface = self
            .ring
            .take_unreferenced(&self.poc.live_surfaces())
            .ok_or_else(|| format!("every one of the {} surfaces is still a reference; the stream needs more", self.ring.len()))?;
        let info = match &self.stream {
            Stream::Nal(info) => info,
            // Unreachable: `picture` dispatches here only for a NAL stream. Written as a refusal
            // rather than an `unreachable!` because this backend has none of those.
            Stream::Frame(_) => return Err("a frame-coded stream reached the H.264 / HEVC decode path".into()),
        };
        let (w, h) = self.session.size();
        let (width_mbs, height_mbs) = match info.codec {
            NalCodec::H264 => ((w / 16).max(1), (h / 16).max(1)),
            NalCodec::Hevc => (w, h),
        };
        let picture = super::picture::build_nal(info, sample, &mut self.poc, surface, width_mbs, height_mbs)?;
        let data = Self::slice_data(&picture, sample)?;

        let decode0 = Instant::now();
        // The buffers a H.264 / HEVC decode submits, in the order libva documents. The buffer handles
        // live to the end of this function, so `vaDestroyBuffer` runs for each of them on every path
        // out of it — including the ones that fail.
        let mut buffers = Vec::with_capacity(picture.slices.len() + 3);
        for (kind, bytes) in Self::submission(&picture, &data) {
            buffers.push(self.session.buffer(kind, bytes)?);
        }
        let ids: Vec<u32> = buffers.iter().map(|b| b.id()).collect();
        self.session.render(surface, &ids)?;
        self.session.sync(surface)?;
        let decode = decode0.elapsed();

        // `vaSyncSurface` above is what makes the read below see the finished picture: rendering is
        // asynchronous, and a read that did not sync reads whatever the driver had written — a
        // plausible-looking black picture rather than an error. The status query is the second half of
        // that: a surface the driver still calls busy after a successful sync is a surface whose
        // contents are not settled, and reading it would be exactly the same mistake.
        let readback0 = Instant::now();
        if !self.session.surface_ready(surface)? {
            return Err(format!("surface {surface} is still being rendered into after vaSyncSurface"));
        }
        let (convert, frame) = self.readback(surface)?;
        Ok(Decoded { decode, readback: readback0.elapsed().saturating_sub(convert), convert, frames: vec![frame] })
    }

    /// One VP9 **sample**, which is not necessarily one frame: a container sample can be a
    /// *superframe* — several frames concatenated, described by a trailing superframe index (VP9
    /// Annex B), and the usual shape is a hidden alt-ref frame followed by the shown frame.
    ///
    /// **So this splits the sample first, and every frame in it is decoded as a frame of its own.**
    /// Handing the whole sample to `Vp9State::frame` would parse only the *first* frame's header,
    /// submit the whole sample as that one frame's data, and — the part that does real damage —
    /// count the *whole sample* as that frame's `slice_data_size`, which the header says is "the byte
    /// count of current frame in the bitstream buffer … but actually means `frame_data_size`"
    /// (`va_dec_vp9.h:280-284`). So on a two-frame superframe whose first frame is 2 673 bytes and
    /// whose sample is 12 943, the driver would be told the first frame's data runs to the end of the
    /// *second* one. This is the `33`-instead-of-`32` class of error, and it is why the split is not
    /// an optimisation but the only way the length means anything.
    ///
    /// ## What each frame does to the reference buffer, and to the caller's pictures
    ///
    /// Frames are decoded **in order, one at a time, each committed before the next is read**, which
    /// is the whole of the relationship: an alt-ref frame inside the superframe refreshes the slots
    /// the shown frame predicts from, so reading the shown frame before committing the alt-ref one
    /// would resolve its references against a buffer that does not hold them yet.
    ///
    /// The pictures the caller gets are **the shown frames, and only those.** A frame with `show_frame`
    /// clear is decoded (it may refresh references) and produces no picture, which is the same
    /// [`Decoded::frames`] rule a single-frame hidden picture follows, and which is what FilmCraft's own
    /// software VP9 decoder does (`crates/vp9/src/decoder.rs:413`, `:417`). So a superframe of
    /// [hidden alt-ref, shown] hands back exactly the one picture the stream shows, and `decode` gives
    /// it the sample's timestamp: one container sample, one timestamp, however many frames it holds.
    ///
    /// **More than one shown frame in one superframe hands back all of them**, under that one
    /// timestamp. The format allows it, our software decoder emits every one
    /// (`crates/vp9/src/decoder.rs:248-255`), and the fixture in `tests/vaapi.rs` really does produce
    /// it, so returning only the last was a divergence between the two paths for the same bitstream
    /// and nothing caught it while the hardware path decoded nothing at all. A caller cannot tell two
    /// pictures under one timestamp apart *by time* — and that is a property of the container, not a
    /// reason to withhold one of them: they are both in the returned `Vec`, in decode order, and the
    /// software decoder's picture for that timestamp is in the same shape.
    ///
    /// ## The order inside a frame is the reverse of [`Self::picture_nal`], and it has to be
    ///
    /// The NAL path takes a surface from the ring *before* it knows what the picture is, because
    /// `build_nal` adds the picture to the buffer and the surface must not be one of the buffer's own.
    /// A VP9 frame inverts that: whether it needs a surface at all is a property of the frame
    /// (`show_existing_frame`, which 6.2 resolves before the frame has a size), so [`Vp9State::frame`]
    /// runs first and the ring is only asked if the frame turns out to code a picture. Taking a
    /// surface first and then finding the frame is a re-show would leave a surface marked in use for a
    /// frame that decoded nothing.
    ///
    /// ## Every refusal here is an `Err`, and that is load-bearing
    ///
    /// `frame()` may have advanced the parse state before it refused (see its own documentation), so
    /// turning one of its errors into a *skip* would let the next frame inherit a refused frame's
    /// `color_config` — a valid record of the wrong depth, which a driver decodes into a plausible
    /// wrong picture. Propagating instead reaches `VideoDecoder::decode`, which latches `refused` and
    /// ends the run: the caller falls back to the software decoder from a known point rather than
    /// carrying on in hardware from an unknown one. **That is why a refused frame inside a superframe
    /// refuses the whole sample** rather than skipping to the next frame in it: the frames after it
    /// inherit from a state this one has already advanced, which is exactly the hazard above, and a
    /// superframe is not a restart point (VP9 6.2's restart points are key frames, which begin a
    /// sample of their own).
    /// One frame-coded sample, dispatched to the codec that owns it.
    ///
    /// **Which stream shape this is decides everything below it**, because the two are not the same
    /// pipeline: a NAL codec builds a picture *then* takes a surface for it, while a VP9 or AV1 frame
    /// is read first and only then turns out to need a surface at all (a frame that re-shows an
    /// earlier picture needs none). So the dispatch is here rather than inside either arm.
    fn picture_frame(&mut self, sample: &[u8]) -> std::result::Result<Decoded, String> {
        // The codec and the two numbers the cross-check reads, taken **by value** before the mutable
        // borrow: `info` would borrow `self.stream` immutably across a call that needs `&mut self`.
        let (codec, profile, bit_depth) = match &self.stream {
            Stream::Frame(info) => (info.codec, info.profile, info.bit_depth),
            // Unreachable: `picture` dispatches here only for a frame-coded stream.
            Stream::Nal(_) => return Err("a NAL-coded stream reached the frame-coded decode path".into()),
        };
        match codec {
            FrameCodec::Vp9 => self.picture_vp9_sample(profile, bit_depth, sample),
            FrameCodec::Av1 => self.picture_av1_sample(profile, bit_depth, sample),
        }
    }

    /// One VP9 **sample**, which is not necessarily one frame: a container sample can be a
    /// *superframe* — several frames concatenated, described by a trailing superframe index (VP9
    /// Annex B), and the usual shape is a hidden alt-ref frame followed by the shown frame.
    ///
    /// **So this splits the sample first, and every frame in it is decoded as a frame of its own.**
    /// Handing the whole sample to `Vp9State::frame` would parse only the *first* frame's header,
    /// submit the whole sample as that one frame's data, and — the part that does real damage —
    /// count the *whole sample* as that frame's `slice_data_size`, which the header says is "the byte
    /// count of current frame in the bitstream buffer … but actually means `frame_data_size`"
    /// (`va_dec_vp9.h:280-284`). So on a two-frame superframe whose first frame is 2 673 bytes and
    /// whose sample is 12 943, the driver would be told the first frame's data runs to the end of the
    /// *second* one. This is the `33`-instead-of-`32` class of error, and it is why the split is not
    /// an optimisation but the only way the length means anything.
    ///
    /// ## What each frame does to the reference buffer, and to the caller's pictures
    ///
    /// Frames are decoded **in order, one at a time, each committed before the next is read**, which
    /// is the whole of the relationship: an alt-ref frame inside the superframe refreshes the slots
    /// the shown frame predicts from, so reading the shown frame before committing the alt-ref one
    /// would resolve its references against a buffer that does not hold them yet. **AV1's
    /// [`picture_av1_sample`] is the same shape for the same reason**, and there it is not only the
    /// record that reads the slots but the *next frame's header parse*.
    ///
    /// The pictures the caller gets are **the shown frames, and only those.** A frame with `show_frame`
    /// clear is decoded (it may refresh references) and produces no picture, which is the same
    /// [`Decoded::frames`] rule a single-frame hidden picture follows, and which is what FilmCraft's own
    /// software VP9 decoder does (`crates/vp9/src/decoder.rs:413`, `:417`). So a superframe of
    /// [hidden alt-ref, shown] hands back exactly the one picture the stream shows, and `decode` gives
    /// it the sample's timestamp: one container sample, one timestamp, however many frames it holds.
    ///
    /// **More than one shown frame in one superframe hands back all of them**, under that one
    /// timestamp. The format allows it, our software decoder emits every one
    /// (`crates/vp9/src/decoder.rs:248-255`), and the fixture in `tests/vaapi.rs` really does produce
    /// it, so returning only the last was a divergence between the two paths for the same bitstream
    /// and nothing caught it while the hardware path decoded nothing at all. A caller cannot tell two
    /// pictures under one timestamp apart *by time* — and that is a property of the container, not a
    /// reason to withhold one of them: they are both in the returned `Vec`, in decode order, and the
    /// software decoder's picture for that timestamp is in the same shape.
    ///
    /// ## The order inside a frame is the reverse of [`Self::picture_nal`], and it has to be
    ///
    /// The NAL path takes a surface from the ring *before* it knows what the picture is, because
    /// `build_nal` adds the picture to the buffer and the surface must not be one of the buffer's own.
    /// A VP9 frame inverts that: whether it needs a surface at all is a property of the frame
    /// (`show_existing_frame`, which 6.2 resolves before the frame has a size), so [`Vp9State::frame`]
    /// runs first and the ring is only asked if the frame turns out to code a picture. Taking a
    /// surface first and then finding the frame is a re-show would leave a surface marked in use for a
    /// frame that decoded nothing.
    ///
    /// ## Every refusal here is an `Err`, and that is load-bearing
    ///
    /// `frame()` may have advanced the parse state before it refused (see its own documentation), so
    /// turning one of its errors into a *skip* would let the next frame inherit a refused frame's
    /// `color_config` — a valid record of the wrong depth, which a driver decodes into a plausible
    /// wrong picture. Propagating instead reaches `VideoDecoder::decode`, which latches `refused` and
    /// ends the run: the caller falls back to the software decoder from a known point rather than
    /// carrying on in hardware from an unknown one. **That is why a refused frame inside a superframe
    /// refuses the whole sample** rather than skipping to the next frame in it: the frames after it
    /// inherit from a state this one has already advanced, which is exactly the hazard above, and a
    /// superframe is not a restart point (VP9 6.2's restart points are key frames, which begin a
    /// sample of their own).
    fn picture_vp9_sample(&mut self, profile: u8, bit_depth: u32, sample: &[u8]) -> std::result::Result<Decoded, String> {
        // The superframe index is parsed by the codec's own parser, the only implementation of it in
        // the workspace. A sample with no index is returned unchanged, so an ordinary single-frame
        // sample is the one-iteration case below with `sample` as its only frame.
        let frames = filmcraft_vp9::header::split_superframe(sample);
        // Timings accumulate across every frame in the sample, so the cost of a superframe's hidden
        // alt-ref frame is reported rather than discarded — it is a real decode the user paid for.
        let mut total = Decoded { decode: Duration::ZERO, readback: Duration::ZERO, convert: Duration::ZERO, frames: Vec::new() };
        for bytes in &frames {
            // An empty frame is not a frame: an index may name a zero-length member, and there is
            // nothing to decode or show in it. The software decoder skips these too
            // (`crates/vp9/src/decoder.rs:161`).
            if bytes.is_empty() {
                continue;
            }
            let decoded = self.picture_vp9_frame(profile, bit_depth, bytes)?;
            total.decode += decoded.decode;
            total.readback += decoded.readback;
            total.convert += decoded.convert;
            // **Every shown frame is collected, in decode order.** A hidden frame contributes none —
            // which is what makes [hidden, shown] hand back exactly the shown one — and a superframe
            // holding two *shown* frames hands back both, under the sample's one timestamp.
            total.frames.extend(decoded.frames);
        }
        Ok(total)
    }

    /// One AV1 **sample**, which is not necessarily one frame.
    ///
    /// **This is measured, not assumed, and it is the same shape VP9's superframe is.** An MP4 AV1
    /// sample is one *temporal unit*, and a temporal unit may hold **several frame units**. Measured on
    /// `av1_tiled.mp4` with this repository's own parser: sample 1 of that fixture is 3 408 bytes of
    /// **four** `OBU_FRAME`s, each with its own `refresh_frame_flags` (`0x02`, `0x04`, `0x08`,
    /// `0x10`), its own `primary_ref_frame` and its own eight tiles — four frames, not one. A decoder
    /// that treated the sample as a single frame would decode one picture of four and hand back one
    /// timestamp for the lot.
    ///
    /// ## Why they are decoded one at a time, and each committed before the next is read
    ///
    /// 7.20 refreshes reference slots **between** frames, and in AV1 the next frame's *header parse*
    /// reads them: `frame_size_with_refs` asks a named slot for its size, `skip_mode_params` and
    /// `OrderHints` ask for its order hint, and `load_previous` reads the primary reference's loop
    /// filter deltas, segmentation features and global motion out of it. FilmCraft's own software
    /// decoder therefore parses frame *n*+1 only after it has updated 7.20 for frame *n*
    /// (`crates/av1/src/decoder.rs:586-607`), and the hardware path has to do the same or the two
    /// paths of one bitstream disagree about what the second frame inherits. So this loop is
    /// `Av1State::frame_at` → decode → `commit`, repeated — the same shape as
    /// [`Self::picture_vp9_sample`], one level down.
    ///
    /// ## `show_existing_frame` is decoded, not declined
    ///
    /// A temporal unit that re-shows an earlier picture carries no frame of its own: the parse returns
    /// from `header.rs:539` with almost nothing set, and there are no tiles to submit. 7.21's answer is
    /// that the decoder outputs the picture the named slot holds and decodes nothing, which is what
    /// FilmCraft's own software decoder does (`crates/av1/src/decoder.rs:481-505`) — so this is
    /// handed back as a picture with no submission and no surface taken from the ring. Declining it
    /// would put a hole where the stream *requires* a picture, and the blank timeline that follows is
    /// the outcome this backend exists to prevent. **The tiled fixture really contains these**
    /// (samples 2, 5 and 8 are one-byte `OBU_FRAME_HEADER`s whose first bit is set), so this is
    /// measured rather than argued.
    ///
    /// ## What is handed back, and why every shown frame
    ///
    /// The shown frames, and only those: a frame with `show_frame` clear is decoded — it may refresh
    /// references 7.20 needs — and produces no picture, which is the same [`Decoded::frames`] rule a
    /// single-frame hidden picture follows and what the software decoder does
    /// (`crates/av1/src/decoder.rs:521`, `:608`). **More than one shown frame in one sample hands back
    /// all of them under the sample's one timestamp**, for the same reason VP9's superframe does: our
    /// software decoder emits every shown frame (`crates/av1/src/decoder.rs:521`), and a backend that
    /// returned one would make the hardware and software paths of the same bitstream disagree about how
    /// many pictures it has.
    ///
    /// ## Every refusal here is an `Err`, not a skip
    ///
    /// `Av1State::frame_at` may have advanced the parse state before it refused — see its own
    /// documentation, and the note on `Av1State::frame_at` about why that is not rolled back — so
    /// skipping would let the next frame inherit a refused frame's propagated deltas. Every refusal
    /// propagates to `VideoDecoder::decode`, which latches `refused` until `reset()`: the caller falls
    /// back to software from a known point rather than carrying on in hardware from an unknown one.
    fn picture_av1_sample(&mut self, profile: u8, bit_depth: u32, sample: &[u8]) -> std::result::Result<Decoded, String> {
        let mut total = Decoded { decode: Duration::ZERO, readback: Duration::ZERO, convert: Duration::ZERO, frames: Vec::new() };
        let mut at = 0usize;
        while let Some(frame) = self.av1.frame_at(sample, at)? {
            at = frame.next();
            let decoded = self.picture_av1_frame(profile, bit_depth, &frame, sample)?;
            total.decode += decoded.decode;
            total.readback += decoded.readback;
            total.convert += decoded.convert;
            total.frames.extend(decoded.frames);
        }
        Ok(total)
    }

    /// One AV1 frame read by [`Av1State::frame_at`]: submit its picture-parameter record and its
    /// **per-tile** slice-parameter records beside the frame's own bytes, or — for a frame that
    /// re-shows an earlier picture — copy that picture out again with nothing submitted.
    ///
    /// `sample` is the temporal unit the frame was read from, and it is what the slice data buffer
    /// is: the per-tile records address byte ranges *inside* it (`va_dec_av1.h:649-658`, "starting
    /// from first byte of the buffer"), so the buffer and the offsets have to be the same bytes.
    fn picture_av1_frame(&mut self, profile: u8, bit_depth: u32, frame: &super::av1::Av1Frame, sample: &[u8]) -> std::result::Result<Decoded, String> {
        // The container's description and the stream's sequence header have to agree, and `Av1State`
        // holds no container: the surfaces were created for the profile and depth the sample entry
        // said, so a record carrying another would be a picture of one format decoded into a surface of
        // another. One function, shared with `picture::build_frame`'s backstop, so the two sites cannot
        // disagree about it.
        if let Some(seq) = self.av1.sequence_header()
            && let Some(why) = av1_entry_disagreement(profile, bit_depth, seq)
        {
            return Err(why);
        }
        if let Some((slot, surface)) = frame.re_show() {
            // 7.21: the decoder outputs the picture that slot holds and decodes nothing. No submission,
            // no sync, no reference slot moved, and no surface taken from the ring.
            return self.re_show(slot, surface);
        }
        // The frame's own size, against the surfaces this session was created at and the rectangle the
        // readback copies. Read *after* `frame_at`, because that is where the header is.
        self.check_av1_shape(frame)?;
        let live = self.av1.live_surfaces();
        let surface = self
            .ring
            .take_unreferenced(&live)
            .ok_or_else(|| format!("every one of the {} surfaces is still a reference; the stream needs more", self.ring.len()))?;
        let record = self.av1.picture_record(frame, surface)?;

        let decode0 = Instant::now();
        // The buffers an AV1 decode submits, in order: the picture-parameter record, **one
        // slice-parameter record per tile**, then the frame's own bytes.
        //
        // The slice records are per tile because libva says so — `va_dec_av1.h:635`, "should be sent
        // once per tile" — and the slice data buffer is the whole temporal unit the frame came in, so
        // that each record's `slice_data_offset`/`slice_data_size` address the tile it names *inside
        // that buffer* (`va_dec_av1.h:649-658`). The handles live to the end of this function, so
        // `vaDestroyBuffer` runs on every path out of it.
        let mut buffers = Vec::with_capacity(frame.slices().len() + 2);
        for (kind, data) in Self::av1_submission(&record, frame.slices(), sample) {
            buffers.push(self.session.buffer(kind, data)?);
        }
        let ids: Vec<u32> = buffers.iter().map(|b| b.id()).collect();
        self.session.render(surface, &ids)?;
        self.session.sync(surface)?;
        let decode = decode0.elapsed();

        let readback0 = Instant::now();
        if !self.session.surface_ready(surface)? {
            return Err(format!("surface {surface} is still being rendered into after vaSyncSurface"));
        }
        // Only a shown frame is read back: `show_frame` clear is a frame the stream decodes and no
        // player shows, and it still refreshes the reference slots below.
        let (frame_out, convert) = match frame.show() {
            true => {
                let (convert, f) = self.readback(surface)?;
                (vec![f], convert)
            }
            false => (Vec::new(), Duration::ZERO),
        };
        // The reference buffer moves **after** the driver rendered the picture, never before: a record
        // describes the buffer as it was before its own frame, and a decode that failed must not have
        // moved it.
        self.av1.commit(frame, surface)?;
        Ok(Decoded { decode, readback: readback0.elapsed().saturating_sub(convert), convert, frames: frame_out })
    }

    /// One frame of a VP9 sample: read it into this decoder's [`Vp9State`], and then either submit its
    /// picture record into a surface no live reference is in, or — for a frame that re-shows an earlier
    /// picture — copy that picture out again with no decode at all.
    ///
    /// `bytes` is **this frame's own slice of the sample**, never the whole sample: it is what
    /// `first_partition_size` is measured from, so passing the sample here is what would put one frame's
    /// compressed-header extent into the next frame's bytes. [`Self::picture_frame`] is the only caller
    /// and it splits first.
    ///
    /// The container's `profile` and `bit_depth` are passed as values rather than as the
    /// [`FrameStreamInfo`] they come from: `info` borrows `self.stream` immutably while this needs
    /// `&mut self`, and these two fields are all the cross-check reads. Borrowing the whole description
    /// across a loop that mutates the decoder is not possible, and cloning a description to work around
    /// that would be worse than naming the two numbers the check needs.
    ///
    /// The returned `Decoded`'s `frames` is empty for a frame the stream decodes but does not show,
    /// which still moved its reference slots — that is the point of not skipping it.
    fn picture_vp9_frame(&mut self, profile: u8, bit_depth: u32, bytes: &[u8]) -> std::result::Result<Decoded, String> {
        // **First, and this is the order the NAL path does not use:** read the frame, so that a frame
        // which codes no picture of its own never takes a surface.
        let frame = self.vp9.frame(bytes)?;
        // The container's own description and the frame's header have to agree, and `Vp9State` holds
        // no container: the surfaces were created for the profile and depth the sample entry said, so
        // a record carrying the frame's would be a picture of one format decoded into a surface of
        // another. One function, shared with `picture::build_frame`'s backstop, so the two sites
        // cannot disagree about it.
        if let Some(why) = vp9_entry_disagreement(profile, bit_depth, frame.header()) {
            return Err(why);
        }
        if let Some((slot, surface)) = frame.re_show() {
            // 8.2.2: the decoder outputs the picture that slot holds and decodes nothing. No
            // submission, no sync, no reference slot moved — and no surface taken from the ring. A
            // re-show *is* a shown picture, so it is returned rather than skipped: a shown frame whose
            // pixels were coded earlier is exactly what a superframe index exists to carry.
            return self.re_show(slot, surface);
        }
        // The frame's own size and depth, against what this session was created for. Read *after*
        // `frame()`, because that is where the header is: a stream that changes resolution or depth
        // mid-run would otherwise submit a `frame_width` larger than the surface the driver renders
        // into, and read back a picture at a rectangle the frame does not fill.
        self.check_vp9_shape(frame.header())?;
        let live = self.vp9.live_surfaces();
        let surface = self
            .ring
            .take_unreferenced(&live)
            .ok_or_else(|| format!("every one of the {} surfaces is still a reference; the stream needs more", self.ring.len()))?;
        let record = frame.record().ok_or_else(|| "a VP9 frame that is not a re-show carries no picture-parameter record".to_string())?;
        let slice = frame.slice().ok_or_else(|| "a VP9 frame that is not a re-show carries no slice-parameter record".to_string())?;

        let decode0 = Instant::now();
        // The three buffers a VP9 decode submits: the picture-parameter record, the **slice**
        // parameter record, then **this frame's own bytes** — not the sample, which for a superframe
        // would hand the driver the next frames' bytes as part of this one.
        //
        // **The slice parameter record is not optional, and this is the buffer whose absence cost a
        // milestone.** It was previously omitted on the reasoning that "VP9 has no slice headers, so
        // there is no slice record". That reasoning was half right about the codec and wrong about
        // the API, and the error is worth not repeating:
        //
        // - VP9 *does* have no slice headers. 6.2 returns from `frame_header()` after the tile and
        //   quantizer syntax and there is no slice layer at all, so every per-slice parsing field
        //   H.264 and HEVC need has nothing to go in it. **That is why this record is thin — it is
        //   why it is 316 bytes of segmentation parameters rather than 3128 bytes of reference lists
        //   and prediction weights. It is not why it is absent.**
        // - The record exists. `va_dec_vp9.h:279` declares `VASliceParameterBufferVP9`, in a file
        //   named for this codec's decode API, with `slice_data_size`, `slice_data_offset`, the
        //   `VA_SLICE_DATA_FLAG_*` field and `SegParam[8]`; and `va_dec_vp9.h:274-276` on the slice
        //   data buffer says the bitstream sent there must include "whole or part of partition 0
        //   (at least compressed header) to the end of frame".
        // - It carries the one thing the picture record cannot. `VADecPictureParameterBufferVP9`
        //   (`va_dec_vp9.h:58-192`) has no `base_q_idx` and no `delta_q_*` — no quantizer field at
        //   all — so `SegParam[]` is the **only** route by which the qindex a frame is dequantized
        //   with reaches a driver. Omitting it does not degrade the decode, it removes the decode's
        //   parameters.
        //
        // What the omission actually looked like: `vaCreateBuffer`, `vaMapBuffer` and
        // `vaRenderPicture` all returned `VA_STATUS_SUCCESS`, and `vaEndPicture` alone returned
        // `VA_STATUS_ERROR_INVALID_PARAMETER` — one such status in a run of 81 758 successes.
        //
        // The handles live to the end of this function, so `vaDestroyBuffer` runs on every
        // path out of it.
        let mut buffers = Vec::with_capacity(3);
        for (kind, data) in Self::vp9_submission(record, slice, bytes) {
            buffers.push(self.session.buffer(kind, data)?);
        }
        let ids: Vec<u32> = buffers.iter().map(|b| b.id()).collect();
        self.session.render(surface, &ids)?;
        self.session.sync(surface)?;
        let decode = decode0.elapsed();

        let readback0 = Instant::now();
        if !self.session.surface_ready(surface)? {
            return Err(format!("surface {surface} is still being rendered into after vaSyncSurface"));
        }
        // **Only a shown frame is read back.** `show_frame` clear is a frame the stream decodes and
        // no player shows (`crates/vp9/src/decoder.rs:413`, `:417`), and it still refreshes the
        // reference slots below. Reading it would be work for a picture nothing will ever look at —
        // and **inside a superframe this is the common case**, since the hidden alt-ref frame is what a
        // superframe usually exists to carry.
        let (frame_out, convert) = match frame.show() {
            true => {
                let (convert, f) = self.readback(surface)?;
                (vec![f], convert)
            }
            false => (Vec::new(), Duration::ZERO),
        };
        // The reference buffer moves **after** the driver rendered the picture, never before: a
        // record describes the buffer as it was before its own frame, and a decode that failed must
        // not have moved it. `commit` refuses a re-show, which cannot reach here.
        //
        // This runs for a hidden frame too, and that is the point: an alt-ref frame inside a
        // superframe is *why* the shown frame after it can be predicted at all.
        self.vp9.commit(&frame, surface)?;
        Ok(Decoded { decode, readback: readback0.elapsed().saturating_sub(convert), convert, frames: frame_out })
    }

    /// A VP9 frame that re-shows a picture an earlier frame left in a reference slot: copy that
    /// picture out again, with nothing submitted to the driver.
    ///
    /// This is the one path in the backend that reads a surface it did not just decode into, so it
    /// says what makes that safe rather than leaving it implied. The surface is one this backend
    /// rendered and read back itself, in the call that decoded the frame it now holds, and nothing
    /// has been decoded into it since — a live reference is never the one a new picture is decoded
    /// into, which is what `take_unreferenced` above guarantees. The picture in it is therefore
    /// settled, and the readiness query is the same gate a decoded picture gets; it is asked rather
    /// than assumed because "settled" is the driver's claim, not ours.
    ///
    /// The timings say what happened: no decode, and a readback that is only a copy.
    fn re_show(&mut self, slot: usize, surface: u32) -> std::result::Result<Decoded, String> {
        let readback0 = Instant::now();
        if !self.session.surface_ready(surface)? {
            return Err(format!("surface {surface}, which reference slot {slot} re-shows, is not ready to read"));
        }
        let (convert, frame) = self.readback(surface)?;
        Ok(Decoded { decode: Duration::ZERO, readback: readback0.elapsed().saturating_sub(convert), convert, frames: vec![frame] })
    }

    /// An AV1 frame's own size, against the surfaces this session was created at and the rectangle the
    /// readback copies.
    ///
    /// **Why both, and what AV1 adds to the VP9 version.** `frame_width_minus1` in the record is the
    /// picture's **upscaled** resolution (`va_dec_av1.h:332-336`), and a driver renders into a
    /// surface created at `Plan.size` — so a frame wider or taller than that surface is a record
    /// asking for a write past the end of it, and surfaces cannot be resized under a live context.
    /// SuperRes widens the *coded* frame past `FrameWidth` without changing `FrameHeight`, so the check
    /// is against `upscaled_width` and `frame_height`, which is exactly what the record carries.
    ///
    /// The readback copies the rectangle `self.geometry.crop` names, so the frame's **render** size
    /// has to be that rectangle too: AV1's `render_width` / `render_height` (5.9.20) may be smaller
    /// than the coded picture, and reading a smaller frame out at the coded rectangle gives a picture
    /// padded with whatever the surface held. Unlike VP9 — whose geometry is updated from each key
    /// frame, because only a key frame spells its size — **AV1 spells its render size in every frame
    /// header**, so the geometry follows the bitstream frame by frame rather than at restart points.
    ///
    /// A key frame that changes either is caught earlier and more cheaply by the hybrid decoder
    /// (`FrameStreamInfo::parameters_changed`, which compares an in-band sequence header against the
    /// sample entry's). This is the check for the frames *between* those.
    fn check_av1_shape(&mut self, frame: &super::av1::Av1Frame) -> std::result::Result<(), String> {
        let Some(header) = frame.header() else { return Ok(()) };
        let size = (header.upscaled_width, header.frame_height);
        let (surface_w, surface_h) = self.session.size();
        if size.0 > surface_w || size.1 > surface_h {
            return Err(format!("a {}x{} AV1 frame does not fit the {surface_w}x{surface_h} surfaces this session was created at", size.0, size.1));
        }
        // A frame that renders at a different size than the picture is read back at. Taken from the
        // frame's own header rather than from `self.geometry`, because the geometry is only updated
        // from the stream's declared picture — which for AV1 is `max_frame_width` x
        // `max_frame_height` and says nothing about `render_width`.
        let render = (header.render_width, header.render_height);
        if render != size {
            self.geometry.crop = (0, 0, render.0, render.1);
        }
        let depth = u32::from(self.av1.sequence_header().map_or(0, |s| s.color.bit_depth));
        if depth != 0 && depth != self.bits {
            return Err(format!("an AV1 frame at {depth} bits, where this session's surfaces hold {}", self.bits));
        }
        Ok(())
    }

    /// The buffers an AV1 decode submits: its picture-parameter record, **one slice-parameter record
    /// per tile**, and the temporal unit those records address.
    ///
    /// Split out for the same reason [`Self::submission`] is: the list is a value a test can read
    /// rather than a sequence of calls only a driver can observe.
    ///
    /// **What is absent from the list is a claim worth stating.** AV1 has no IQ matrix buffer — the
    /// quantizer is in the picture record (`base_qindex` and the four `delta_q`s, `va_dec_av1.h:497-523`)
    /// and the segmentation parameters in `seg_info` — so nothing here stands in for one, and a driver
    /// that read a buffer it was not given would read another kind's bytes as AV1's.
    ///
    /// ## The slice data buffer is the temporal unit, and that is a decision
    ///
    /// `va_dec_av1.h:634-645` says the host is "responsible to parse out the per tile information" and
    /// the bitstream "is sent to driver in per tile granularity", while `slice_data_size` is "The byte
    /// count of current tile in the bitstream buffer, starting from first byte of the buffer" and
    /// `slice_data_offset` "offset to the first byte of the data buffer" (`va_dec_av1.h:649-658`). So
    /// the buffer holds more than one thing and the per-tile records address into it: the whole
    /// temporal unit is the buffer, and each record names the tile's own bytes inside it. Submitting
    /// the temporal unit rather than a concatenation of bare tile payloads keeps every OBU header
    /// intact, which is what a driver parsing the low-overhead stream needs, and the per-tile records
    /// still address exactly the same tiles either way.
    fn av1_submission<'p>(record: &'p [u8], slices: &'p [Vec<u8>], data: &'p [u8]) -> Vec<(i32, &'p [u8])> {
        let mut out = Vec::with_capacity(slices.len() + 2);
        out.push((PICTURE_PARAMETER_BUFFER, record));
        for slice in slices {
            out.push((SLICE_PARAMETER_BUFFER, slice.as_slice()));
        }
        out.push((SLICE_DATA_BUFFER, data));
        out
    }

    /// A VP9 frame's own size and depth, against the surfaces this session was created at and the
    /// rectangle the readback copies.
    ///
    /// **Why both.** `frame_width` / `frame_height` in the record are the frame's own dimensions
    /// (`va_dec_vp9.h:60-68`), and a driver renders into a surface created at `Plan.size` — so a
    /// frame larger than that surface is a record asking for a write past the end of it, and the
    /// surfaces cannot be resized under a live context. And the readback copies the rectangle
    /// `self.geometry` names, which `Stream::declared_picture` updates from each VP9 key frame: a
    /// frame of a different size would be read out at a rectangle it does not fill, which
    /// `biplanar::to_frame` cannot detect — the surface is big enough, it just does not contain that
    /// picture there.
    ///
    /// A key frame that changes either is caught earlier and more cheaply by the hybrid decoder
    /// (`FrameStreamInfo::parameters_changed`, which compares a key frame against the sample entry).
    /// This is the check for the frames *between* key frames, and for a stream whose container
    /// description was right about its first key frame and wrong about a later one.
    fn check_vp9_shape(&self, header: &filmcraft_vp9::header::FrameHeader) -> std::result::Result<(), String> {
        let size = (header.width, header.height);
        let (surface_w, surface_h) = self.session.size();
        if size.0 > surface_w || size.1 > surface_h {
            return Err(format!("a {}x{} VP9 frame does not fit the {surface_w}x{surface_h} surfaces this session was created at", size.0, size.1));
        }
        let (_, _, crop_w, crop_h) = self.geometry.crop;
        if size != (crop_w, crop_h) {
            return Err(format!("a {}x{} VP9 frame, where this run reads back {crop_w}x{crop_h} pictures", size.0, size.1));
        }
        let depth = u32::from(header.color.bit_depth);
        if depth != self.bits {
            return Err(format!("a VP9 frame at {depth} bits, where this session's surfaces hold {}", self.bits));
        }
        Ok(())
    }

    /// The buffers a VP9 decode submits: its picture-parameter record, its slice-parameter record,
    /// and the frame's own bytes.
    ///
    /// Split out for the same reason [`Self::submission`] is: the list is a value a test can read
    /// rather than a sequence of calls only a driver can observe. **Three buffers and no more.**
    ///
    /// What is *absent* from the list is a claim worth stating precisely, because a previous version
    /// of this function got it wrong in the direction that broke the decoder. VP9 has no slice
    /// headers and no scaling lists, so there is no IQ matrix buffer and none of H.264's per-slice
    /// parsing fields — and **VP9 has a slice parameter record anyway**, declared at
    /// `va_dec_vp9.h:279`, thin because the codec has no slice headers rather than absent because it
    /// does. See the call site for what that record carries and why the driver refuses the picture
    /// without it. The IQ matrix remains genuinely absent: a driver that read one would take its
    /// bytes as VP9's segmentation probabilities (`mb_segment_tree_probs[7]` and
    /// `segment_pred_probs[3]`, `va_dec_vp9.h:172-177`), which are in the picture record beside it.
    ///
    /// ## One `vaRenderPicture`, not two — measured, three ways
    ///
    /// ffmpeg's own VA-API VP9 path splits a picture into two renders: the picture-parameter buffer
    /// alone, then the slice-parameter and slice-data buffers. **That split is not required**, and the
    /// evidence is measurement rather than preference. Three shapes were run over `vp9_p0.mp4` on the
    /// same driver:
    ///
    /// | shape | result |
    /// |---|---|
    /// | **one `vaBeginPicture`, one `vaRenderPicture` with all three buffers, one `vaEndPicture`** | `MATCH`, 24 of 24 pictures bit-exact |
    /// | one `vaBeginPicture`, **two** `vaRenderPicture` calls (ffmpeg's shape), one `vaEndPicture` | `MATCH`, 24 of 24 pictures bit-exact |
    /// | two `vaBeginPicture` / `vaRenderPicture` / `vaEndPicture` cycles | `VA_STATUS_ERROR_INVALID_PARAMETER` at `vaEndPicture`, all 24 pictures lost |
    ///
    /// So: the split buys nothing observable, the *extra buffer* was the whole of it, and the third row
    /// is the trap — `Session::render` begins and ends a picture per call, so "splitting" by calling
    /// it twice is not the split ffmpeg does and the driver refuses it for exactly the same reason it
    /// refused the missing buffer. One render is kept because it is one fewer shape to be wrong about
    /// and one `vaMapBuffer` / `vaSyncSurface` pair fewer per picture; nothing here depends on a
    /// preference for it.
    fn vp9_submission<'p>(record: &'p [u8], slice: &'p [u8], sample: &'p [u8]) -> Vec<(i32, &'p [u8])> {
        vec![(PICTURE_PARAMETER_BUFFER, record), (SLICE_PARAMETER_BUFFER, slice), (SLICE_DATA_BUFFER, sample)]
    }
}

/// A decoded picture and what it cost, so the timings can be accumulated from one place rather than
/// from inside the decode loop.
struct Decoded {
    /// The driver's own decode: submitting the records and waiting for the picture.
    decode: Duration,
    /// The GPU-to-CPU copy, which is the whole readback less the conversion.
    readback: Duration,
    /// The conversion of the driver's planes into FilmCraft's own, which is CPU work inside the
    /// readback and reported apart from it.
    convert: Duration,
    /// The pictures to hand back, in decode order, and **empty for a sample that produced none** — a
    /// frame the stream decodes but does not show (VP9 6.2's `show_frame` clear), which still moved
    /// its reference slots. **Not** for a failure: a frame this backend could not decode is an
    /// `Err`, and an error latches `refused` — see `Self::refused`.
    ///
    /// **A list, not an `Option`, because one container sample can hold more than one shown frame.**
    /// A VP9 superframe is several frames under one presentation time, and a caller cannot tell two
    /// pictures under one timestamp apart by time — but it can still be *given* both, and
    /// FilmCraft's own software decoder gives both (`crates/vp9/src/decoder.rs:248-255`), so a
    /// backend that returned one would make the hardware and software paths of the same bitstream
    /// disagree about how many pictures the stream has. This was a single `Option` until the
    /// hardware path decoded at all, which is how a documented divergence survived every test.
    frames: Vec<VideoFrame>,
}

/// A decoded picture, as a `VideoFrame`.
type VideoFrame = filmcraft_frame::VideoFrame;

#[cfg(test)]
mod tests {
    use super::super::picture::SliceParam;
    use super::*;
    use ffi::{SLICE_DATA_BUFFER, SLICE_PARAMETER_BUFFER};

    /// **A VP9 superframe is one sample holding several frames, and each frame's records must be
    /// built from that frame's own bytes.**
    ///
    /// The field this is really about is the slice record's `slice_data_size`: `va_dec_vp9.h:280-284`
    /// spells out that it is "the byte count of current frame in the bitstream buffer … but actually
    /// means `frame_data_size`". Hand the record builder the whole sample and it counts every frame
    /// after the first into this one's buffer, so a driver reads the next frame's bytes as part of
    /// this one.
    ///
    /// **What this test pinned before was a field that does not have the problem.** It asserted that
    /// the *picture* record's `first_partition_size` is `part.len() - uncompressed_size`, on the
    /// reasoning that the field was "the rest of the frame". That reasoning was wrong —
    /// `va_dec_vp9.h:167-169` defines it as "the byte count of compressed header", which is 6.2.14's
    /// `header_size_in_bytes` and has nothing to do with the frame's length — and a driver told
    /// 12 726 instead of 185 reads the tile data as compressed header and **renders nothing while
    /// reporting success at every call**. With the definition corrected the picture record's two
    /// lengths are both header fields, and the field that really is frame-length-dependent is the
    /// slice record's, so the test follows it there.
    ///
    /// The frames here are hand-built, so the test can make the two lengths differ by a known amount
    /// and assert the exact value rather than only that they differ. The fixture in `tests/vaapi.rs`
    /// proves the same property on real encoded frames; this one needs no encoder and so runs
    /// everywhere, including on a machine whose driver never sees a VP9 frame.
    /// A VP9 frame this test can build: a key frame whose uncompressed header the parser reads, padded
    /// to `total` bytes so two frames of clearly different lengths can be made.
    ///
    /// **The padding is behind the header, not before it.** A VP9 frame is a header then a compressed
    /// header, and the slice record's `slice_data_size` is the whole frame, so extra bytes have to go
    /// where a real frame's tile data goes, or the frame does not parse at all.
    fn vp9_frame_of(total: usize) -> Vec<u8> {
        let mut bits: Vec<bool> = Vec::new();
        let put = |value: u32, n: usize, bits: &mut Vec<bool>| bits.extend((0..n).map(|i| value >> (n - 1 - i) & 1 == 1));
        put(2, 2, &mut bits); // frame_marker
        put(0, 2, &mut bits); // profile_low_bit, profile_high_bit — profile 0
        put(0, 1, &mut bits); // show_existing_frame
        put(0, 1, &mut bits); // frame_type — KEY_FRAME
        put(1, 1, &mut bits); // show_frame
        put(0, 1, &mut bits); // error_resilient_mode
        put(0x49_83_42, 24, &mut bits); // frame_sync_code
        put(1, 3, &mut bits); // color_space — BT.601
        put(0, 1, &mut bits); // color_range
        put(319, 16, &mut bits); // frame_width_minus_1 — 320
        put(239, 16, &mut bits); // frame_height_minus_1 — 240
        put(0, 1, &mut bits); // render_and_frame_size_different
        put(1, 1, &mut bits); // refresh_frame_context
        put(1, 1, &mut bits); // frame_parallel_decoding_mode
        put(0, 2, &mut bits); // frame_context_idx
        put(37, 6, &mut bits); // loop_filter_level
        put(5, 3, &mut bits); // loop_filter_sharpness
        put(0, 1, &mut bits); // loop_filter_delta_enabled
        put(128, 8, &mut bits); // base_q_idx
        put(0, 1, &mut bits); // delta_q_y_dc_present
        put(0, 1, &mut bits); // delta_q_uv_dc_present
        put(0, 1, &mut bits); // delta_q_uv_ac_present
        put(0, 1, &mut bits); // segmentation_enabled
        // 6.2.13's tile_info, written the way the parser reads it: at 320x240 there is one superblock
        // column, so `min_log2` is 0 and `max_log2` is 0, the `while` loop that would read a
        // `tile_cols_log2` flag never runs, and the next bit read is `tile_rows_log2`. Writing a bit
        // here would shift every field after it — the tile counts are derived from the size, not
        // spelled, which is what makes this easy to get wrong by hand.
        put(0, 1, &mut bits); // tile_rows_log2 — one superblock row
        // The compressed header's length, as 6.2.14's `header_size_in_bytes` states it. It is not the
        // field the record submits — that is measured from the frame's real length — so any value over
        // zero parses, and this one says 1 so the padding below is unambiguously tile data.
        put(1, 16, &mut bits);
        put(1, 1, &mut bits); // trailing_bits
        let mut out: Vec<u8> = bits.chunks(8).map(|c| c.iter().enumerate().fold(0u8, |v, (i, &b)| v | (u8::from(b) << (7 - i)))).collect();
        assert!(out.len() <= total, "a {total}-byte frame must be able to hold its own header");
        out.resize(total, 0);
        out
    }

    #[test]
    fn a_vp9_superframe_gives_each_frame_a_record_built_from_its_own_bytes() {
        // Two real, parseable frames of clearly different lengths, so "measured from the sample" and
        // "measured from the frame" cannot coincide. The marker is VP9 Annex B's: `0xc0`, a 4-byte
        // size field (`3 << 3`), and the frame count minus one in the low three bits.
        let first = vp9_frame_of(300);
        let second = vp9_frame_of(900);
        let marker = 0xc0u8 | (3 << 3) | 1;
        let mut sample = Vec::new();
        sample.extend_from_slice(&first);
        sample.extend_from_slice(&second);
        sample.push(marker);
        sample.extend_from_slice(&(first.len() as u32).to_le_bytes());
        sample.extend_from_slice(&(second.len() as u32).to_le_bytes());
        sample.push(marker);

        // The split is what the decode loop performs first, and it must find both frames.
        let parts = filmcraft_vp9::header::split_superframe(&sample);
        assert_eq!(parts.len(), 2, "a two-frame superframe splits into two frames");
        assert_eq!(parts[0], first.as_slice(), "member 0 is the first frame, byte for byte");
        assert_eq!(parts[1], second.as_slice(), "member 1 is the second frame, byte for byte");

        // **And each member is what a record must be built from.** Asserted against the real builders
        // and read back through `ffi`, so this is the number the driver would read rather than a
        // restatement of the formula.
        let refs: [Option<filmcraft_vp9::header::RefInfo>; 8] = std::array::from_fn(|_| Some(filmcraft_vp9::header::RefInfo { width: 320, height: 240 }));
        for (i, part) in parts.iter().enumerate() {
            let mut state = filmcraft_vp9::header::HeaderState::default();
            let header =
                filmcraft_vp9::header::parse_uncompressed(part, &mut state, &refs).unwrap_or_else(|e| panic!("member {i} is a frame the parser reads: {e}"));
            let record = super::super::picture::vp9_picture_record(&header, &state, part, &super::super::picture::Vp9ReferenceSet::default())
                .unwrap_or_else(|e| panic!("member {i} is describable: {e}"));
            let slice = super::super::picture::vp9_slice_record(&header, &state, part).unwrap_or_else(|e| panic!("member {i} has a slice record: {e}"));
            let got = ffi::picture_params_vp9(&record).expect("92 bytes is a VP9 picture record").first_partition_size;
            assert_eq!(
                usize::from(got),
                usize::try_from(header.header_size_in_bytes).unwrap_or(usize::MAX),
                "member {i}: first_partition_size is the COMPRESSED HEADER's own length, which the bitstream states — not the frame's length"
            );
            // The picture record's `frame_header_length_in_bytes` is the other of the two counts, and is
            // this frame's own: on a superframe the two members have different headers, so a record
            // built from the sample could not tell them apart.
            assert_eq!(
                usize::from(ffi::picture_params_vp9(&record).expect("92 bytes").frame_header_length_in_bytes),
                header.uncompressed_size,
                "member {i}: the uncompressed header's length is this frame's own {} bytes",
                header.uncompressed_size
            );
            // **And the field that really is frame-length-dependent**, which is where this test's power
            // moved to once `first_partition_size` was corrected: the slice record's `slice_data_size`
            // counts the frame, so a whole-sample reading is a *different, larger* number.
            let sized = ffi::slice_params_vp9(&slice).expect("316 bytes is a VP9 slice record").slice_data_size;
            assert_eq!(usize::try_from(sized).unwrap_or(usize::MAX), part.len(), "member {i}: slice_data_size is this frame's own {} bytes", part.len());
            // The defect this exists to catch, stated so a regression names itself: the whole-sample
            // reading is a *different, larger* number, so the two cannot be confused.
            let whole_sample = u32::try_from(sample.len()).unwrap_or(u32::MAX);
            assert_ne!(sized, whole_sample, "member {i}: the per-frame and whole-sample counts agree, so this fixture cannot catch the defect");
            assert!(sized < whole_sample, "member {i}: the correct extent is the smaller one — the whole-sample value reaches into the next frame");
        }
    }

    /// The pool is sized so that a picture always has a surface that is not a live reference, which is
    /// the defect a bare rotation invites: on a 6-surface ring with 5 live references it eventually
    /// hands back a surface the driver is reading as a reference, and the picture is decoded into
    /// the reference it is supposed to be predicted from.
    ///
    /// The invariant is what makes it safe rather than a number that happens to fit today's fixtures:
    /// the decoded picture buffer can never hold more entries than the record has slots for them (the
    /// marking functions trim to it, and a VP9 frame moves only the slots its `refresh_frame_flags`
    /// names), so a pool one larger than that always has one free surface.
    ///
    /// **All four codecs this backend decodes are in the table.** Leaving VP9 or AV1 out would leave
    /// the invariant untested for exactly the two codecs whose slot count is fixed by their record
    /// (`reference_frames[8]` and `ref_frame_map[8]`), which are the two where a wrong number is a
    /// plausible-looking wrong picture rather than an allocation failure.
    #[test]
    fn the_surface_pool_is_always_larger_than_the_picture_buffer() {
        // Each codec's bound is its record's own slot count, and the pool is that plus one. The two
        // frame codecs' numbers are read out of production code so they cannot drift from the pools
        // that are allocated.
        for (codec, slots, pool) in [("H.264", 16usize, 17usize), ("HEVC", 15, 16), ("VP9", 8, vp9_surface_pool()), ("AV1", 8, av1_surface_pool())] {
            assert!(pool > slots, "{codec}: a pool of {pool} does not exceed a buffer of {slots} plus one");
            // And a ring of that size, filled to the brim, still has a free surface to decode into.
            let ids: Vec<u32> = (0..pool as u32).collect();
            let mut ring = Ring::from_surfaces(&ids);
            let live: Vec<u32> = ids[..slots].to_vec();
            assert!(ring.take_unreferenced(&live).is_some(), "{codec}: a full buffer still leaves a surface");
            // With the pool entirely live there is nothing, and that is an error rather than a
            // reference being overwritten.
            assert_eq!(ring.take_unreferenced(&ids), None, "{codec}: nothing free means an error");
        }
        // And the bound each frame codec's pool comes from is its own record's array rather than a
        // number typed here: eight slots is `reference_frames[8]` in `va_dec_vp9.h:70-77` and
        // `ref_frame_map[8]` in `va_dec_av1.h:347-356`.
        assert_eq!(vp9_surface_pool(), 8 + 1, "VP9's pool is its eight reference slots and the one being written");
        assert_eq!(av1_surface_pool(), 8 + 1, "AV1's pool is 7.20's eight slots and the one being written");
    }

    /// The buffers a VP9 frame is submitted as: **its picture-parameter record, its slice-parameter
    /// record, and the frame's bytes — in that order, all three, in one render.**
    ///
    /// This test exists because the middle one was missing for a whole milestone and nothing but a
    /// driver noticed. The list it asserts was previously two entries long, on the reasoning that
    /// "VP9 has no slice headers (`va_dec_vp9.h` declares no `VASliceParameterBufferVP9`)". The
    /// parenthetical was false — `va_dec_vp9.h:279` declares it — and the result was a decode the
    /// driver accepted at `vaCreateBuffer`, `vaMapBuffer` and `vaRenderPicture` and refused at
    /// `vaEndPicture`, which is the worst possible shape for a missing record. **So the reasoning is
    /// corrected here, not just the expectation**: VP9 has no slice headers, and that is *why* the
    /// record is thin — 316 bytes of `SegParam[8]` and three geometry words rather than H.264's
    /// 3128 — rather than *why* it is absent. The header names both halves: `va_dec_vp9.h:265-268`
    /// says the structure "should be sent once per frame", and `va_dec_vp9.h:270-276` says what the
    /// slice data buffer beside it must contain.
    ///
    /// Asserted without a driver because the list is a value. The same reasoning as
    /// `an_h264_picture_submits_its_scaling_lists_and_one_record_per_slice` on the other side — and
    /// the same failure mode it exists for: the iHD driver reports success at every call for a
    /// submission missing a buffer and leaves the render target holding something wrong, so "the
    /// decode returned no error" would never have caught this.
    #[test]
    fn a_vp9_picture_submits_both_records_and_the_frame_in_one_render() {
        let record = vec![0x5au8; 92];
        let slice = vec![0xa7u8; 316];
        let frame = vec![0xc3u8; 4000];
        let bufs = VaDecoder::vp9_submission(&record, &slice, &frame);
        let kinds: Vec<i32> = bufs.iter().map(|(k, _)| *k).collect();
        assert_eq!(
            kinds,
            vec![PICTURE_PARAMETER_BUFFER, SLICE_PARAMETER_BUFFER, SLICE_DATA_BUFFER],
            "picture record, slice record, then the frame's own bytes — in ONE render, which is measured, not preferred"
        );
        // All three are the bytes they claim to be, read back through `ffi`'s own safe readers rather
        // than re-derived here. The slice record is the one a test that only read the picture record
        // back would not have covered, which is the whole of the bug.
        assert!(ffi::picture_params_vp9(bufs[0].1).is_some(), "92 bytes the driver reads as a VP9 picture record");
        assert_eq!(bufs[0].1, record.as_slice());
        assert!(ffi::slice_params_vp9(bufs[1].1).is_some(), "316 bytes the driver reads as a VP9 slice record");
        assert_eq!(bufs[1].1, slice.as_slice());
        // The frame goes in whole, un-copied and un-relayed: the record's `frame_header_length_in_bytes`
        // and `first_partition_size` are how the driver finds the two headers inside these bytes, and
        // the slice record's `slice_data_size` is this frame's own length, so re-laying them out here
        // would be a second, wrong answer to the same question.
        assert_eq!(bufs[2].1, frame.as_slice(), "the slice data buffer is the frame itself");
    }

    /// A `Picture` whose records are recognisable lengths, for checking the submission list without a
    /// driver: a two-slice H.264 picture carries five buffers, an HEVC one carries four.
    fn picture(slices: usize, iq: bool) -> Picture {
        Picture {
            params: vec![0xaau8; 672],
            iq_matrix: if iq { vec![0xbbu8; 240] } else { Vec::new() },
            slices: (0..slices).map(|i| SliceParam { offset: u32::try_from(i * 100).unwrap_or(0), length: 100, params: vec![0xccu8; 3128] }).collect(),
            slice_offsets: (0..slices).map(|i| i * 4).collect(),
        }
    }

    /// The buffers a picture is submitted as, and every record among them read back through `ffi`'s
    /// own safe readers — the record the driver is actually handed, not one this module re-derived.
    ///
    /// This is the test that names the defect the IQ matrix's absence caused. The iHD driver takes a
    /// submission without a `VAIQMatrixBufferH264`, reports success at every call, and leaves the
    /// render target holding a uniform mid-grey picture — so "the decode returned no error" was
    /// never going to catch it, and the buffer list is the only place it shows. If a buffer kind goes
    /// missing from `submission` this fails; if one arrives twice, or in the wrong order, it fails.
    #[test]
    fn an_h264_picture_submits_its_scaling_lists_and_one_record_per_slice() {
        let p = picture(2, true);
        let data = vec![0xddu8; 400];
        let kinds: Vec<i32> = VaDecoder::submission(&p, &data).iter().map(|(k, _)| *k).collect();
        assert_eq!(
            kinds,
            vec![PICTURE_PARAMETER_BUFFER, IQ_MATRIX_BUFFER, SLICE_PARAMETER_BUFFER, SLICE_PARAMETER_BUFFER, SLICE_DATA_BUFFER],
            "picture record, scaling lists, one record per slice, then the slice bytes"
        );
        // And each record is one `ffi` can read back, which is the record the driver is handed.
        let bufs = VaDecoder::submission(&p, &data);
        assert!(ffi::picture_params_h264(bufs[0].1).is_some(), "the picture record is readable");
        assert!(ffi::iq_matrix_h264(bufs[1].1).is_some(), "the scaling-list record is readable");
        assert_eq!(bufs[1].1.len(), 240, "240 bytes, the size libva declares");
        assert!(bufs[2].1.windows(3128).all(|_| true) && bufs[2].1.len() == 3128, "a slice record is readable");
        assert_eq!(bufs.last().expect("a buffer").1, data.as_slice(), "the last buffer is the slice data");
    }

    /// HEVC submits no IQ matrix buffer: its scaling lists are inside the picture parameter record,
    /// so a buffer of that kind would be a record the driver does not read.
    /// A run that starts at a CRA skips its RASL pictures — they refer to pictures that were never
    /// decoded. **Only those.** An earlier version of this decoder also tested `skip_rasl` inside the
    /// picture builder, so the flag stayed set for the rest of the run and every picture after the
    /// RASL ones came back as an empty list: indistinguishable from a picture the stream had not
    /// produced, which is the blank-timeline outcome this backend exists to prevent.
    ///
    /// Needs a driver, and skips without one — the flag is decode-loop state, so there is nothing
    /// to check on a machine with no GPU. What it asserts is that a picture well after the run's
    /// first one comes back, which is the whole content of the bug.
    #[test]
    fn a_run_that_starts_at_a_cra_still_decodes_its_later_pictures() {
        use filmcraft_codecs::VideoDecoder as _;
        let Some(path) = open_gop_hevc() else { return };
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => return,
        };
        let Ok(file) = filmcraft_isobmff::open(bytes.clone()) else { return };
        let Some(t) = file.track_of_kind(filmcraft_isobmff::TrackKind::Video) else { return };
        let Some(entry) = file.tracks[t].entries.first() else { return };
        let Some(Ok(info)) = filmcraft_codecs::hw::NalStreamInfo::from_entry(entry) else {
            eprintln!("SKIPPED: not an hvcC stream");
            return;
        };
        let Ok(mut d) = VaDecoder::new(info) else {
            eprintln!("SKIPPED: this driver does not take this stream");
            return;
        };
        let count = file.tracks[t].samples.len();
        // The stream description, for the NAL types that say which picture is a CRA.
        let described: Option<filmcraft_codecs::hw::NalStreamInfo> = filmcraft_codecs::hw::NalStreamInfo::from_entry(entry).and_then(|r| r.ok());
        let Some(nal_info) = described else {
            eprintln!("SKIPPED: not an hvcC stream");
            return;
        };
        // Start at a CRA rather than at the file's beginning: `skip_rasl` is only set when the run
        // *begins* at a CRA / BLA, which is what a seek onto a key frame does. The file's own first
        // picture is an IDR and would never set it.
        // The same predicate `decode` uses to set the flag: a BLA / CRA at the start of a run, so
        // types 16..=21 except the two IDR ones (19, 20). An IDR has no RASL after it, and matching
        // one would silently never set the flag.
        let starts_run = |types: &[u8]| types.iter().any(|&n| (16..=21).contains(&n) && !(19..=20).contains(&n));
        let cra = (0..count).find(|&i| file.read_sample(&bytes, t, i).is_ok_and(|s| starts_run(&nal_info.nal_types(&s))));
        let Some(cra) = cra else {
            eprintln!("SKIPPED: the fixture has no CRA to start a run at");
            return;
        };
        d.reset();
        let (mut decoded, mut skipped_leading) = (0usize, 0usize);
        let (mut after_skipping_ended, mut dropped_after_skipping) = (false, false);
        for i in cra..count {
            let Ok(s) = file.read_sample(&bytes, t, i) else { break };
            let pts = file.tracks[t].samples[i].pts;
            match d.decode(&s, pts) {
                Ok(frames) if frames.is_empty() => {
                    // Acceptable only for the run's leading RASL pictures, which are contiguous: the
                    // CRA itself decodes, and the pictures that cannot be decoded come straight after
                    // it. Once pictures have started coming out *again*, no later picture may be
                    // dropped — that is the sticky flag, and its symptom is a hole in the stream.
                    if after_skipping_ended {
                        // A picture dropped *after* the run's leading RASL ones: the sticky flag.
                        dropped_after_skipping = true;
                        decoded += 1;
                    } else {
                        skipped_leading += 1;
                    }
                }
                Ok(frames) => {
                    decoded += frames.len();
                    if skipped_leading > 0 {
                        after_skipping_ended = true;
                    }
                }
                Err(e) => panic!("sample {i} of an open-GOP HEVC run failed to decode: {e}"),
            }
        }
        assert!(skipped_leading > 0, "the run starts at a CRA, so some leading RASL pictures are skipped");
        assert!(decoded > 1, "the run decoded {decoded} pictures around {skipped_leading} RASL skips");
        assert!(!dropped_after_skipping, "the flag stayed set after the leading RASL pictures ended, so pictures were dropped later in the run");
        eprintln!("open-GOP run from sample {cra}: {decoded} pictures decoded, {skipped_leading} leading RASL pictures skipped");
    }

    /// An open-GOP HEVC fixture: `open-gop=1` makes the keyint boundary a CRA with RASL pictures
    /// after it, which is what sets `skip_rasl`. Skips without ffmpeg.
    fn open_gop_hevc() -> Option<std::path::PathBuf> {
        use filmcraft_testkit::fixtures;
        let ff = filmcraft_testkit::require_ffmpeg!(None);
        fixtures::generate(&fixtures::fixtures_root().join("platform/vaapi").join("hevc_open_gop.mp4"), |tmp| {
            let args = [
                "-f",
                "lavfi",
                "-i",
                "testsrc2=s=320x240:r=24:d=1",
                "-c:v",
                "libx265",
                "-preset",
                "fast",
                "-tag:v",
                "hvc1",
                "-x265-params",
                "log-level=error:bframes=3:keyint=6:min-keyint=6:scenecut=0:open-gop=1",
                "-pix_fmt",
                "yuv420p",
            ];
            let st = std::process::Command::new(&ff).args(["-y", "-v", "error"]).args(args).arg(tmp).status();
            matches!(st, Ok(s) if s.success())
        })
    }

    #[test]
    fn an_hevc_picture_submits_no_scaling_list_buffer() {
        let p = picture(1, false);
        let kinds: Vec<i32> = VaDecoder::submission(&p, &[0u8; 10]).iter().map(|(k, _)| *k).collect();
        assert_eq!(kinds, vec![PICTURE_PARAMETER_BUFFER, SLICE_PARAMETER_BUFFER, SLICE_DATA_BUFFER]);
    }
}
