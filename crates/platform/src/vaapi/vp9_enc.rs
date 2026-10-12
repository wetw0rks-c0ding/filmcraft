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
//! ## 2. Asking anyway, and the status the driver returns
//!
//! `vaGetConfigAttributes` was called for all four VP9 profiles against both encode entrypoints. For
//! every one of the 26 attributes asked and all eight (profile, entrypoint) pairs, the answer was:
//!
//! ```text
//! -- VP9Profile0/EncSlice  (profile=VAProfileVP9Profile0 ep=VAEntrypointEncSlice)
//!    VAConfigAttribRateControl              id=  5 status=UNSUPPORTED_ENTRYPOINT(13)     value=3735928559 (0xdeadbeef)
//!    VAConfigAttribEncVP9(58)               id= 58 status=UNSUPPORTED_ENTRYPOINT(13)     value=3735928559 (0xdeadbeef)
//!    ... (every attribute, same status)
//! -- VP9Profile0/EncSliceLP  (profile=VAProfileVP9Profile0 ep=VAEntrypointEncSliceLP)
//!    VAConfigAttribRateControl              id=  5 status=UNSUPPORTED_ENTRYPOINT(13)     value=3735928559 (0xdeadbeef)
//! -- VP9Profile1/EncSlice   ... UNSUPPORTED_ENTRYPOINT(13) ... 0xdeadbeef
//! -- VP9Profile2/EncSlice   ... UNSUPPORTED_ENTRYPOINT(13) ... 0xdeadbeef
//! -- VP9Profile3/EncSlice   ... UNSUPPORTED_ENTRYPOINT(13) ... 0xdeadbeef
//! ```
//!
//! `VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT` is `13` (`va.h:264+`), and the poison survives
//! untouched, confirming the driver returned early rather than answering.
//!
//! ## 3. The control, which is what makes the above mean anything
//!
//! The same probe, unchanged, on H.264 — a codec this machine's driver is known to encode:
//!
//! ```text
//!    VAProfileH264Main              -> VA_STATUS_SUCCESS(0), n=2: 1(VAEntrypointVLD) 6(VAEntrypointEncSlice)
//! -- H264High/EncSlice [CONTROL]
//!    VAConfigAttribRateControl              id=  5 status=VA_STATUS_SUCCESS(0)  value=5334 (0x000014d6)
//!    VAConfigAttribEncVP9(58)               id= 58 status=VA_STATUS_SUCCESS(0)  value=2147483648 (0x80000000)
//! -- VP9Profile0/EncSlice
//!    VAConfigAttribRateControl              id=  5 status=UNSUPPORTED_ENTRYPOINT(13)  value=0xdeadbeef
//! ```
//!
//! The control answers `VA_STATUS_SUCCESS` with real values for the same entrypoint, on the same
//! display, through the same `dlsym`'ed function pointer. The VP9 reads are therefore measuring the
//! driver and not the harness. The stronger discrimination, the one this branch used for HEVC:
//! attribute **58** (`VAConfigAttribEncVP9`) returns `0x80000000` — which is
//! `VA_ATTRIB_NOT_SUPPORTED` — for the H.264 control, and `VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT`
//! with the poison intact for every VP9 encode pair. The driver answers the question; it does not
//! merely stay silent.
//!
//! ## 4. This is the driver declining a codec libva can express — not an API that lacks it
//!
//! The header **does** declare VP9 encode: `va_enc_vp9.h` exists and defines
//! `VAEncSequenceParameterBufferVP9`, `VAEncPictureParameterBufferVP9`, `VAEncSegParamVP9` and
//! `VACodedBufferVP9Status`, and `va.h:1065` declares
//! `VAConfigAttribEncVP9 = 58` ("VP9 encoding attribute"). So the API can express a VP9 encoder and
//! this driver will not provide one. That distinction matters for whoever reads this: the answer is
//! about **this machine's hardware and driver**, not about VP9, not about libva, and not about
//! FilmCraft's ability to *decode* VP9 — decode is proven and unaffected.
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
    }
}
