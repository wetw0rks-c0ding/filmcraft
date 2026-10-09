//! The guard against the bug this backend had: a hand-declared `extern "C"` type that does not
//! match the header it claims to come from.
//!
//! `vaSyncBuffer` was declared with two parameters where libva's third (`uint64_t timeout_ns`,
//! `va.h:4391`) was never written at the call site. That is undefined behaviour: the driver read an
//! argument slot nobody set, the call returned `VA_STATUS_ERROR_TIMEDOUT` in a debug build and
//! success in a release one, and every picture it produced was correct. Nothing in the language can
//! catch that, because the *declaration* is the thing that is wrong — the call site matched the
//! declaration perfectly, so `rustc`, the linker and every test that made the call were satisfied.
//!
//! So the declaration has to be written down somewhere an independent source can check, and there
//! are two oracles, not equally available:
//!
//! * [`every_entry_point_has_the_arity_the_header_declares`] needs nothing but the header text. It
//!   counts each prototype's parameters and compares them with an arity that `va_entry_points!`
//!   derives from the very parameter list the pointer type was built out of, so the two cannot
//!   drift apart. **This is the check that would have caught `vaSyncBuffer`**, and it needs no
//!   driver, no library and no compiler.
//! * [`every_declaration_matches_what_a_c_compiler_measures`] needs a C compiler, and it is the only
//!   thing that can judge a *struct*: it redeclares every entry point (a C redeclaration that
//!   disagrees with the header is a hard error, not a warning) and static-asserts every record's
//!   `sizeof`. A record declared eight bytes short is the size-only version of a wrong arity, and
//!   this module had two — `VASurfaceAttrib` and `VABufferInfo`.
//!
//! Both skip, with a note on stdout, where the thing they need is not installed, so neither can turn
//! a machine red for the wrong reason. Where the headers *are* installed — every machine that can
//! run the VA-API backend at all — they are an assertion, not a courtesy.

use super::*;
use std::mem::size_of;

/// Where the libva headers live on this machine.
const LIBVA_HEADERS: &str = "/usr/include/va";

/// The header text with its `/* … */` and `//` comments blanked out, so a prototype cannot be found
/// inside a doc comment and a parameter cannot be cut in half by one. Newlines are kept, so a byte
/// offset still names the line a reader would go and look at.
fn without_c_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let bytes = src.as_bytes();
    let (mut i, mut depth) = (0usize, 0usize);
    while i < bytes.len() {
        if depth == 0 && bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'*') {
            depth = 1;
            i += 2;
            continue;
        }
        if depth == 0 && bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'/') {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if depth > 0 {
            if bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/') {
                depth = 0;
                i += 2;
                continue;
            }
            out.push(if bytes[i] == b'\n' { '\n' } else { ' ' });
            i += 1;
            continue;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// The byte offset just past `name(` in `src`, if `name` appears there as a whole identifier.
/// `drm.drm_ioctl_prime_…`, and a longer name that merely ends in `name`, are not matches.
fn find_call(src: &str, name: &str) -> Option<usize> {
    let bytes = src.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = src[from..].find(name) {
        let at = from + rel;
        from = at + name.len();
        let standalone = at == 0 || !(bytes[at - 1].is_ascii_alphanumeric() || bytes[at - 1] == b'_' || bytes[at - 1] == b'.');
        let mut j = at + name.len();
        while bytes.get(j).is_some_and(u8::is_ascii_whitespace) {
            j += 1;
        }
        if standalone && bytes.get(j) == Some(&b'(') {
            return Some(j + 1);
        }
    }
    None
}

/// How many parameters the argument list starting at `from` has, counting the commas at its top
/// level. `()` and `(void)` are zero.
fn parameter_count(src: &str, from: usize) -> usize {
    // `from` is already *inside* the call's parentheses, so the depth starts at one and the first
    // `)` is the one that closes the list.
    let (mut depth, mut commas, mut any) = (1usize, 0usize, false);
    for &b in src.as_bytes().get(from..).unwrap_or_default() {
        match b {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    break;
                }
            }
            b',' if depth == 1 => commas += 1,
            b' ' | b'\t' | b'\n' | b'\r' if depth == 1 => {}
            _ if depth == 1 => any = true,
            _ => {}
        }
    }
    if !any {
        return 0;
    }
    commas + 1
}

/// Every entry point's arity, re-derived from the header that declares it.
///
/// Nothing in the toolchain can do this on its own — a call site type-checks against the
/// declaration, so a wrong arity in the *declaration* is invisible to the compiler, to the linker
/// and to every test that makes the call. The only oracle is the header, so this reads it.
#[test]
fn every_entry_point_has_the_arity_the_header_declares() {
    let dir = std::path::Path::new(LIBVA_HEADERS);
    if !dir.join("va.h").is_file() {
        eprintln!("SKIPPED: no {LIBVA_HEADERS}/va.h, so there is no header to check the declarations against");
        return;
    }
    for entry in ENTRY_POINTS {
        let file = entry.header.split(':').next().unwrap_or(entry.header);
        let path = dir.join(file);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{} names {}: {e}", entry.symbol, path.display()));
        let stripped = without_c_comments(&text);
        let at = find_call(&stripped, entry.symbol)
            .unwrap_or_else(|| panic!("{} does not declare {} at all, so the declaration of it is unchecked", entry.header, entry.symbol));
        let found = parameter_count(&stripped, at);
        assert_eq!(found, entry.arity, "{}: the header ({}) declares {found} parameters, our declaration has {}", entry.symbol, entry.header, entry.arity);
    }
    assert!(!ENTRY_POINTS.is_empty(), "the table of entry points is empty, so the loop above proved nothing");
}

/// Every declaration, judged by a C compiler reading the same headers.
///
/// The entry points are *redeclared*, not merely counted: C requires a redeclaration to be
/// compatible with the original, so a wrong parameter type, a missing or extra parameter, or a
/// different return type is a compile error that quotes the header's own declaration. The records
/// are `_Static_assert`ed on `sizeof`, and the number comes from `size_of` on the Rust side, so the
/// two are compared directly instead of both being compared with a number somebody typed twice.
#[test]
fn every_declaration_matches_what_a_c_compiler_measures() {
    if !std::path::Path::new(LIBVA_HEADERS).join("va.h").is_file() {
        eprintln!("SKIPPED: no {LIBVA_HEADERS}/va.h");
        return;
    }
    let Some(cc) = ["cc", "gcc", "clang"].into_iter().find_map(which) else {
        eprintln!("SKIPPED: no C compiler to measure the headers with");
        return;
    };
    let records: &[(&str, usize)] = &[
        ("VAImageFormat", size_of::<VaImageFormat>()),
        ("VAImage", size_of::<VaImage>()),
        ("VAConfigAttrib", size_of::<VaConfigAttrib>()),
        ("VASurfaceAttrib", size_of::<VaSurfaceAttrib>()),
        ("VAGenericValue", size_of::<VaGenericValue>()),
        ("VAPictureH264", size_of::<VaPictureH264>()),
        ("VAPictureParameterBufferH264", size_of::<VaPictureParameterBufferH264>()),
        ("VASliceParameterBufferH264", size_of::<VaSliceParameterBufferH264>()),
        ("VAIQMatrixBufferH264", size_of::<VaIqMatrixBufferH264>()),
        ("VAPictureHEVC", size_of::<VaPictureHevc>()),
        ("VAPictureParameterBufferHEVC", size_of::<VaPictureParameterBufferHevc>()),
        ("VASliceParameterBufferHEVC", size_of::<VaSliceParameterBufferHevc>()),
        ("VACodedBufferSegment", size_of::<VaCodedBufferSegment>()),
        ("VABufferInfo", size_of::<VaBufferInfo>()),
        ("VAEncSequenceParameterBufferH264", size_of::<VaEncSequenceParameterBufferH264>()),
        ("VAEncPictureParameterBufferH264", size_of::<VaEncPictureParameterBufferH264>()),
        ("VAEncSliceParameterBufferH264", size_of::<VaEncSliceParameterBufferH264>()),
        ("VAEncMiscParameterFrameRate", size_of::<VaEncMiscParameterFrameRate>()),
        ("VAEncMiscParameterQuantization", size_of::<VaEncMiscParameterQuantization>()),
    ];
    let mut source = String::from("#include <va/va.h>\n#include <va/va_drm.h>\n#include <va/va_dec_hevc.h>\n#include <va/va_enc_h264.h>\n");
    source.push_str(
        "/* One line per entry point, exactly as ffi.rs declares it. A redeclaration\n   that disagrees with the header is an error, not a warning. */\n",
    );
    for entry in ENTRY_POINTS {
        let (ret, args) = entry.prototype.split_once('(').unwrap_or((entry.prototype, ""));
        source.push_str(&format!("{} {}({});\n", ret.trim(), entry.symbol, args.trim_end_matches(')')));
    }
    source.push_str("\n/* Every record's size, as size_of sees it in Rust. */\n");
    for (name, size) in records {
        source.push_str(&format!("_Static_assert(sizeof({name}) == {size}, \"{name}\");\n"));
    }
    let dir = std::env::temp_dir().join(format!("filmcraft-va-abi-{}", std::process::id()));
    if std::fs::create_dir_all(&dir).is_err() {
        eprintln!("SKIPPED: no writable temporary directory");
        return;
    }
    let path = dir.join("abi.c");
    let written = std::fs::write(&path, &source).is_ok();
    let out = if written {
        std::process::Command::new(&cc).args(["-fsyntax-only", "-I", LIBVA_HEADERS, path.to_str().unwrap_or_default()]).output()
    } else {
        eprintln!("SKIPPED: cannot write {}", path.display());
        return;
    };
    let _ = std::fs::remove_dir_all(&dir);
    let out = out.unwrap_or_else(|e| panic!("running {cc} failed: {e}"));
    assert!(out.status.success(), "{cc} rejected our declarations against the headers:\n{}", String::from_utf8_lossy(&out.stderr));
}

/// `name` on the `PATH`, or `None`.
fn which(name: &str) -> Option<String> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).find_map(|dir| {
        let candidate = dir.join(name);
        candidate.is_file().then(|| candidate.to_string_lossy().into_owned())
    })
}
