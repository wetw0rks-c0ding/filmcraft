//! The VA-API decoder: surfaces, the decode loop and the readback. Part of [`super`], split out
//! here because the module is large; nothing in this file names a libva type.

use std::time::{Duration, Instant};

use filmcraft_codecs::hw::{NalCodec, StreamInfo};
use filmcraft_codecs::{CodecError, DecodedFrame, Result, VideoDecoder};

use super::ffi::{self, Display, Session};
use super::picture::{NalPoc, Picture, reference_pictures};
use super::stream::{Stream, plan_frame, supports};
use super::{Ring, validate_image};
use crate::biplanar::{self, Geometry};
use ffi::{IQ_MATRIX_BUFFER, PICTURE_PARAMETER_BUFFER, SLICE_DATA_BUFFER, SLICE_PARAMETER_BUFFER};

/// How many pictures a decoded picture buffer can hold: a DPB is at most 16 frames, so more means the
/// decoder lost track of the stream. A VA-API decode is synchronous — one access unit is rendered and
/// read back inside the `decode` call that submitted it — so this never gets near the bound; it is
/// here for the same reason `MfDecoder` has it, as a backstop against a driver-reported count that
/// would otherwise grow without limit.
const MAX_IN_FLIGHT: usize = 64;

/// Surfaces the ring holds beyond what the stream's decoded picture buffer needs. A picture is read
/// back out of its surface before the next one is decoded, so the ring has to hold the whole buffer
/// plus the picture being decoded into — a surface that is the current picture and a reference at the
/// same time would be read as a reference while it is being written.
const SURFACE_MARGIN: usize = 1;

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
    /// from. `NalPoc` is [`super::picture`]'s, and its decoded-picture-buffer lives inside it.
    poc: NalPoc,
    /// Presentation times of submitted access units whose picture has not come out yet.
    in_flight: Vec<i64>,
    /// The run started at a CRA / BLA picture: its RASL pictures are not decodable and are left out,
    /// as the software decoder does.
    skip_rasl: bool,
    /// Whether the next picture is the first of a run, which is what decides whether RASL pictures are
    /// skipped and what a seek resets.
    first: bool,
    /// [`VideoDecoder::flush`] drained the driver, which then behaves as at the end of a stream: it
    /// only restarts at an IDR picture, and the references of the run are gone.
    drained: bool,
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
        // The ring is sized from the stream's own decoded picture buffer, not from a constant: the
        // references have to survive in their surfaces until the pictures that use them are decoded,
        // so a stream that needs more surfaces gets them. `Session::new` clamps the count, so a
        // stream description claiming an absurd buffer costs a clamp and not an allocation.
        let count = match &stream {
            Stream::Nal(info) => reference_pictures(info).map_err(|e| format!("decoded picture buffer: {e}"))?,
            // Every frame-coded stream is declined by `plan` above, so this arm is unreachable today; it
            // is written rather than unreachable so that adding a frame codec is not a compile error. An
            // `Ok` here would mean a frame codec had been planned without the records to decode it, and
            // that is refused too rather than turned into an allocation for a stream this cannot submit.
            Stream::Frame(info) => {
                let Err(why) = plan_frame(info) else {
                    return Err("a frame-coded stream was planned without the picture-parameter records this backend cannot build".into());
                };
                return Err(why);
            }
        };
        let session = Session::new(&display, plan.profile, plan.entrypoint, plan.format, plan.size, count + SURFACE_MARGIN)?;
        let name = stream.name();
        // Safe to read `info.crop` / `info.size` unbounded here *because* `stream.plan()` above ran
        // first and bounded both, along with the coded size the crop has to fit inside
        // (`stream::plan_nal`, `plan_frame`). `first_geometry` is the one reader of those fields
        // that trusts them, and this is its only caller in this backend — so the order of the two
        // lines is load-bearing, not incidental.
        let geometry = stream.first_geometry(plan_bits(plan.format));
        let first = session.surfaces().first().copied().unwrap_or(0);
        log::info!(
            "hardware decoding: {name} on {}{}",
            display.vendor().unwrap_or("a VA-API driver"),
            if session.reads_indirectly() { " (readback through a copy)" } else { "" }
        );
        Ok(Self {
            info: stream.info(),
            geometry,
            stream,
            session,
            ring: Ring::new(count + SURFACE_MARGIN, first),
            poc: NalPoc::default(),
            in_flight: Vec::new(),
            skip_rasl: false,
            first: true,
            drained: false,
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
    /// `picture.iq_matrix` contributes no buffer: that is how an HEVC picture, whose scaling lists are
    /// inside its picture parameter record, says there is nothing to submit.
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

/// `RTFormat` as the significant bits per sample its pictures hold: 8 for NV12, 10 for P010.
fn plan_bits(format: ffi::RtFormat) -> u32 {
    match format {
        ffi::RtFormat::Yuv420_10 => 10,
        _ => 8,
    }
}

impl VideoDecoder for VaDecoder {
    fn decode(&mut self, sample: &[u8], pts: i64) -> Result<Vec<DecodedFrame>> {
        self.fed += 1;
        if self.fail_after.is_some_and(|n| self.fed > n) {
            return Err(CodecError::Decode("VA-API decode failed (test hook)".into()));
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
            }
        }
        self.first = false;
        // A frame-coded stream would say its picture size in the bitstream, which is not in the
        // container; every one is declined while it is still a description, so there is nothing here
        // to refresh — see `stream::plan_frame`.
        if let Some(g) = self.stream.declared_picture(sample) {
            self.geometry = g;
        }
        self.in_flight.push(pts);
        if self.in_flight.len() > MAX_IN_FLIGHT {
            return Err(CodecError::Decode(format!("the decoder holds {} pictures back", self.in_flight.len())));
        }
        let r = self.picture(sample);
        let mut out = Vec::new();
        match r {
            // A picture that decoded is returned as it is, with the presentation time it was
            // submitted under. A decode that produced nothing is an error rather than an empty
            // frame: an empty frame list for a sample the caller fed is indistinguishable from a
            // picture the stream has not produced yet, and a blank timeline is the worst answer
            // available — an error hands the stream to the software decoder instead.
            Ok(Some(decoded)) => {
                let stamp = self.take_pts(pts);
                self.timings.frames += 1;
                self.timings.decode += decoded.decode;
                self.timings.readback += decoded.readback;
                self.timings.convert += decoded.convert;
                out.push(DecodedFrame { pts: stamp, frame: decoded.frame, draft: false });
                Ok(out)
            }
            Ok(None) => Ok(out),
            Err(e) => Err(CodecError::Decode(e)),
        }
    }

    fn flush(&mut self) -> Vec<DecodedFrame> {
        // A VA-API decode is one picture in one picture out, so there is nothing buffered to collect:
        // whatever was decoded was read back and returned by the `decode` call that submitted it. What
        // a flush does is end the run, so that what follows starts like a new one.
        self.in_flight.clear();
        self.poc.reset();
        self.first = true;
        self.skip_rasl = false;
        self.drained = true;
        Vec::new()
    }

    fn reset(&mut self) {
        // A seek leaves no reference picture behind, so the context goes with them: a context that
        // still held the previous run's references would decode this run against surfaces holding
        // something else. The surfaces and the config stay, since they depend only on the stream.
        self.session.clear_context();
        self.poc.reset();
        self.in_flight.clear();
        self.skip_rasl = false;
        self.first = true;
        self.drained = false;
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
    /// submitted to the surface at the front of the ring, then copied out. Returns `None` when the
    /// access unit holds no picture (an HEVC RASL picture the run left out), which is not an error.
    ///
    /// The timing is kept in two parts because they are what a user would notice separately: the
    /// driver's own decode (`vaEndPicture` plus the sync that waits for it), and the copy of the
    /// finished picture back to the CPU — of which the conversion into FilmCraft's planes is
    /// measured separately, so the reported readback is the copy and not the copy plus the
    /// conversion.
    fn picture(&mut self, sample: &[u8]) -> std::result::Result<Option<Decoded>, String> {
        if self.skip_rasl {
            return Ok(None);
        }
        let surface = self.ring.take().ok_or("the surface ring is empty")?;
        let info = match &self.stream {
            Stream::Nal(info) => info,
            // Declined while it was still a description, so a caller that skipped planning is told
            // exactly what `plan_frame` says rather than a reason invented here. An `Ok` would mean a
            // frame codec got past `new`, which the ring sizing above already refused.
            Stream::Frame(info) => match plan_frame(info) {
                Err(why) => return Err(why),
                Ok(_) => return Err("a frame-coded stream reached the decode loop without picture-parameter records".into()),
            },
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
        Ok(Some(Decoded { decode, readback: readback0.elapsed().saturating_sub(convert), convert, frame }))
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
    frame: VideoFrame,
}

/// A decoded picture, as a `VideoFrame`.
type VideoFrame = filmcraft_frame::VideoFrame;

#[cfg(test)]
mod tests {
    use super::super::picture::SliceParam;
    use super::*;
    use ffi::{SLICE_DATA_BUFFER, SLICE_PARAMETER_BUFFER};

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
    #[test]
    fn an_hevc_picture_submits_no_scaling_list_buffer() {
        let p = picture(1, false);
        let kinds: Vec<i32> = VaDecoder::submission(&p, &[0u8; 10]).iter().map(|(k, _)| *k).collect();
        assert_eq!(kinds, vec![PICTURE_PARAMETER_BUFFER, SLICE_PARAMETER_BUFFER, SLICE_DATA_BUFFER]);
    }
}
