//! The DRM descriptor's own view of a libva buffer object's storage.
//!
//! libva allocates every buffer it hands back as a DRM buffer object, and for the encode coded
//! buffer it never maps that object into the client's address space in a way the client can name:
//! `vaMapBuffer` returns a **separate** staging allocation (measured — see
//! [`Encoder::read_coded`](super::ffi::Encoder::read_coded)), and the picture is reachable only
//! through the driver's own `VACodedBufferSegment::buf` pointer, which nothing bounds.
//!
//! This module closes that gap from the other side. It is **libva that named the object**
//! (`vaAcquireBufferHandle` reports its handle and its size), and **DRM that maps it** — either the
//! descriptor libva itself handed back, or, on a stack that reports a bare handle, one this process
//! asks for by `ioctl` on the descriptor FilmCraft already holds the display open on. Every byte
//! this backend reads back comes out of a mapping this module created and sized itself, so no byte
//! of the read depends on a pointer the driver chose, on another process's view of the address
//! space, or on a length we had to discover.
//!
//! Nothing here links libva: it is plain DRM (one `ioctl`, `mmap`, `munmap`, `close`) plus the
//! handle libva already reported.
//!
//! **Ownership is the subtle part, and it is why the two routes below are kept apart.** A dma-buf
//! descriptor belongs to whoever opened it. `vaAcquireBufferHandle` opens one for a
//! `VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME` buffer and closes it again in `vaReleaseBufferHandle`
//! (measured on this machine: the number is open before any ioctl of ours, it points at a
//! `/dmabuf:` of the coded buffer's size, and it is closed by the release). A descriptor this
//! process asked the kernel for in [`map`] is ours, and only that one may be closed here. Closing
//! the other is a double close, and a double close takes whatever took the number in between — a
//! log file, a socket, another dma-buf.

use std::ffi::{c_int, c_void};

/// `mmap`'s failure return, `(void *) -1`.
const MAP_FAILED: *mut c_void = -1isize as *mut c_void;

/// `PROT_READ`.
const PROT_READ: c_int = 1;
/// `MAP_SHARED`: the mapping has to see the driver's writes, which is the whole point of it.
const MAP_SHARED: c_int = 1;

/// `_IOWR('d', 0x2d, struct drm_prime_handle)`, the request number of
/// `DRM_IOCTL_PRIME_HANDLE_TO_FD` (`drm.h:1168`).
///
/// The encoding is the kernel's `_IOC` one, spelled out rather than written as the constant every
/// DRM program copies around: `dir(2) | size(16) | type(8) | nr(8)`, with `dir = 3` for read/write.
/// `size` is `size_of` of the struct below, so the number cannot drift from the struct it describes —
/// and the struct is the header's, which is three words, not the two this module used to declare.
const fn iowr(kind: u8, nr: u8, size: usize) -> std::ffi::c_ulong {
    (3 << 30) | ((size as u64) << 16) | ((kind as u64) << 8) | nr as u64
}

/// `struct drm_prime_handle`: the handle in, the export flags, and **the descriptor out**.
///
/// `/usr/include/drm/drm.h:926-934` declares three `__u32`-sized fields — `__u32 handle`,
/// `__u32 flags`, `__s32 fd` — so the struct is **12 bytes** and the request number it gives is
/// `_IOWR('d', 0x2d, 12)` = `0xc00c642d`. It was declared here with two of them, which made the
/// struct eight bytes short and the request number `0xc008642d`.
///
/// That is the size-only version of the bug this audit is about, and it is not harmless. The kernel
/// accepts either number (it dispatches on the request number, not on the encoded size — measured on
/// this box, `_IOC_SIZE` of 4, 8, 12, 16 and 20 are all dispatched), but the descriptor it hands back
/// goes into the **third** field, which an eight-byte struct does not have: the export succeeds, the
/// kernel opens a descriptor, and that descriptor is never read and never closed. One per encoded
/// picture, for the life of the process.
///
/// **The record and its fields are `pub(crate)` for one reason: the ABI guard.** Until this was the
/// only record in the backend that nothing could check — it was not in
/// [`crate::vaapi::abi_guard`]'s record list, the C oracle included only `<va/*.h>` and never
/// `<drm/drm.h>`, and the two literals in the test below were read out of the same header by hand.
/// `docs/vaapi-progress.md` §6 records that exact style of verification as one that "read as
/// rigorous and was wrong" the first time it was used, and §3's claim that both guards were "verified
/// to fail on the exact historical bugs" was false for this record: the historical bug was the
/// twelve-byte struct, and nothing could fail on it. Nothing about the struct changes — the guard
/// now judges the same bytes from `/usr/include/drm/drm.h` and from `DRM_IOCTL_PRIME_HANDLE_TO_FD`
/// itself.
#[repr(C)]
pub(crate) struct DrmPrimeHandle {
    /// `__u32 handle`: the GEM handle to export.
    pub(crate) handle: u32,
    /// The `open` flags to export with. Zero is `O_RDONLY`, which is all a mapping of coded output
    /// needs: the driver has already written it, and this process never writes into the buffer.
    pub(crate) flags: u32,
    /// `__s32 fd`: "User-space sets `drm_prime_handle.handle` with the GEM handle to export and
    /// `drm_prime_handle.flags`, and gets back a DMA-BUF file descriptor in `drm_prime_handle->fd`"
    /// (`drm.h:1159-1161`). **Not** in `handle`.
    pub(crate) fd: c_int,
}

/// The request number the guard reads the header's own macro for, so a test can ask a C compiler
/// whether the two agree. The hand-typed literal in the test below is exactly what this replaces.
#[cfg(test)]
pub(crate) const PRIME_HANDLE_TO_FD_ABI_NAME: &str = "DRM_IOCTL_PRIME_HANDLE_TO_FD";

pub(crate) const PRIME_HANDLE_TO_FD: std::ffi::c_ulong = iowr(b'd', 0x2d, size_of::<DrmPrimeHandle>());

/// A read-only window on a buffer object's storage, mapped by this process and unmapped when it is
/// dropped.
///
/// `len` is the size libva reported for the object, not a length discovered afterwards: that is
/// what makes [`Storage::bytes`] the same length on every run.
pub struct Storage {
    base: *mut c_void,
    len: usize,
}

impl Storage {
    /// The whole object, as bytes. Every read in the readback comes from here.
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: `base` is what `mmap` returned for `len` bytes and has not been unmapped (only
        // `Drop` unmaps it, once, with the value borrowed throughout), and `len` is the size the
        // mapping was created with.
        unsafe { std::slice::from_raw_parts(self.base.cast::<u8>(), self.len) }
    }

    /// How many bytes the object holds, as libva reported it.
    pub fn len(&self) -> usize {
        self.len
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        // SAFETY: `base` and `len` are exactly what the `mmap` in `map` returned, this is the only
        // call that unmaps them, and `base` is not `MAP_FAILED` because `map` rejected that.
        unsafe { munmap(self.base, self.len) };
    }
}

/// A descriptor DRM exported for one buffer object. Closed when it is dropped.
struct Export {
    fd: c_int,
}

impl Drop for Export {
    fn drop(&mut self) {
        // SAFETY: `fd` is a descriptor this module obtained and has not closed.
        close_fd(self.fd);
    }
}

/// `F_GETFD`, for [`is_open`].
const F_GETFD: c_int = 1;

unsafe extern "C" {
    fn ioctl(fd: c_int, request: std::ffi::c_ulong, ...) -> c_int;
    fn mmap(addr: *mut c_void, length: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, length: usize) -> c_int;
    fn close(fd: c_int) -> c_int;
    fn fcntl(fd: c_int, command: c_int, ...) -> c_int;
}

/// Whether `fd` names an open descriptor of at least `mem_size` bytes.
///
/// Two questions, because "open" alone is not enough: a number can name *some* descriptor — the
/// render node, a log file, a socket — and mapping one of those hands back bytes that are not the
/// coded picture. The size is what tells a dma-buf export apart from everything else, and `mem_size`
/// is libva's own number for the object, so an open descriptor shorter than it is not this buffer.
fn is_the_buffer(fd: c_int, mem_size: usize) -> bool {
    if !is_open(fd) {
        return false;
    }
    // SAFETY: `BorrowedFd::borrow_raw` asserts only that `fd` is open, which the `fcntl` above has
    // just established.
    let borrowed = unsafe { std::os::fd::BorrowedFd::borrow_raw(fd) };
    // Duplicated rather than adopted, so asking the question does not take the descriptor away from
    // whoever owns it — which here is libva.
    borrowed.try_clone_to_owned().is_ok_and(|dup| std::fs::File::from(dup).metadata().is_ok_and(|m| m.len() >= mem_size as u64))
}

/// Whether `fd` names an open descriptor, asked of the kernel rather than inferred.
fn is_open(fd: c_int) -> bool {
    // SAFETY: `fcntl(F_GETFD)` reads the descriptor table and takes no argument, so the variadic
    // call sets up the registers the SysV ABI requires and reads nothing that does not exist.
    unsafe { fcntl(fd, F_GETFD) != -1 }
}

/// Map the coded buffer's storage, read-only, `mem_size` long.
///
/// `handle` and `mem_size` are the two numbers `vaAcquireBufferHandle` reported
/// (`va.h:4034`, `va.h:4046`); this function neither computes nor guesses either.
///
/// **Which of the two it is depends on the memory type, and the code has to know which.** For
/// `VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME` — the value libva reports for a coded buffer on a DRM
/// backend, measured — `VABufferInfo::handle` is **already an open dma-buf descriptor**: libva did
/// the prime export itself and handed the descriptor back, and closes it again in
/// `vaReleaseBufferHandle` (`va.h:4098-4104`). Measured on this machine: the number is open before
/// any ioctl of ours, it points at a `/dmabuf:` whose size is the coded buffer's, and mapping it
/// returns bytes identical to the ones `VACodedBufferSegment::buf` names.
///
/// So on that path there is nothing to export, and doing the ioctl anyway is actively harmful, which
/// is what this module used to do:
///
/// * it fed that descriptor number to `DRM_IOCTL_PRIME_HANDLE_TO_FD` as if it were a GEM handle,
///   which made the kernel export some *other* object (measured: 48 KiB, where the coded buffer's
///   storage is 768 KiB) and return a descriptor nobody could use;
/// * that descriptor was written into a field the struct did not have, so **it was leaked** — one
///   per encoded picture, measured as the open-descriptor count rising by one per frame;
/// * and the descriptor it mapped *was* libva's, which it then closed. libva closes it again, so
///   the number was closed twice, and the second close took whatever had taken it since.
///
/// The ioctl is therefore only used for a handle that is not already a descriptor, which is what a
/// stack that reports a GEM handle needs. That descriptor is ours, and only that one is closed here.
pub fn map(drm: c_int, handle: u32, mem_size: usize) -> Result<Storage, String> {
    if mem_size == 0 {
        return Err("libva reported a buffer object of 0 bytes".into());
    }
    if drm < 0 {
        return Err("no DRM descriptor is open".into());
    }
    // The common case: libva already exported the object and `handle` names its descriptor.
    let Ok(as_fd) = c_int::try_from(handle) else {
        return Err(format!("libva reported buffer handle {handle}, which is not a descriptor number"));
    };
    if is_the_buffer(as_fd, mem_size) {
        return map_it(as_fd, mem_size);
    }
    // Otherwise it is a handle to export, and the descriptor comes back in `fd` — the field the
    // header names and the one this struct now has.
    let mut request = DrmPrimeHandle { handle, flags: 0, fd: 0 };
    // SAFETY: `drm` is an open DRM render-node descriptor, `request` is a live
    // `struct drm_prime_handle` of the size the request number encodes, and the kernel writes at
    // most that many bytes back into it.
    let rc = unsafe { ioctl(drm, PRIME_HANDLE_TO_FD, &mut request as *mut DrmPrimeHandle) };
    if rc != 0 {
        return Err(format!("DRM_IOCTL_PRIME_HANDLE_TO_FD refused buffer handle {handle}"));
    }
    // A descriptor number that is negative, or that names nothing open, is not mappable — and
    // `mmap` on one would fail with a number nobody can act on.
    if request.fd < 0 {
        return Err(format!("DRM_IOCTL_PRIME_HANDLE_TO_FD returned descriptor {}, which is not one", request.fd));
    }
    // This one is ours: the kernel opened it for this call, so it is ours to close.
    let _export = Export { fd: request.fd };
    if !is_open(request.fd) {
        return Err(format!("DRM_IOCTL_PRIME_HANDLE_TO_FD returned descriptor {} for buffer handle {handle}, which is not open", request.fd));
    }
    map_it(request.fd, mem_size)
}

/// The mapping itself, shared by both routes above.
fn map_it(fd: c_int, mem_size: usize) -> Result<Storage, String> {
    // SAFETY: `fd` is an open descriptor, `mem_size` is what libva reported for the object behind
    // it, and the mapping is `PROT_READ` and `MAP_SHARED` with no offset.
    let base = unsafe { mmap(std::ptr::null_mut(), mem_size, PROT_READ, MAP_SHARED, fd, 0) };
    if base.is_null() || base == MAP_FAILED {
        return Err(format!("mmap of the {mem_size}-byte coded buffer failed"));
    }
    Ok(Storage { base, len: mem_size })
}

/// Close a descriptor. Shared with the rest of this backend's FFI module, which opens the render
/// node itself.
fn close_fd(fd: c_int) {
    // SAFETY: `fd` is a descriptor this process opened and is closing once.
    unsafe { close(fd) };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `struct drm_prime_handle` is **twelve** bytes and the request number that follows from it is
    /// `0xc00c642d`.
    ///
    /// `/usr/include/drm/drm.h:926-934` declares `__u32 handle; __u32 flags; __s32 fd;` and
    /// `drm.h:1159-1161` says the descriptor comes back in that third field. The two-field version
    /// this replaced gave `0xc008642d`, and the kernel — which dispatches on the request number and
    /// not on the size encoded in it — accepted that happily while writing its answer four bytes past
    /// the end of the struct, where nothing could read it.
    ///
    /// **These two assertions are the declaration restated, not checked.** They are here so a reader
    /// of the constant knows the number is *derived* from the struct rather than typed beside it,
    /// and on their own they prove nothing: both numbers were read out of the same header by hand,
    /// which is the style `docs/vaapi-progress.md` §6 records as having "read as rigorous and been
    /// wrong" the first time it was used here. The check that is worth something is
    /// [`crate::vaapi::abi_guard`], which asks a C compiler for `sizeof(struct drm_prime_handle)`,
    /// every `offsetof` in it and `DRM_IOCTL_PRIME_HANDLE_TO_FD` itself — all read from the installed
    /// `/usr/include/drm/drm.h`, none of them from this file. Verified to fail on the historical
    /// shape: a record declared wider than the header fails
    /// `static assertion failed: "struct drm_prime_handle"`, fields declared in the wrong order fail
    /// the `offsetof` and hole checks with the size still correct, and a request number no kernel
    /// defines fails `static assertion failed: "DRM_IOCTL_PRIME_HANDLE_TO_FD"`. Removing a field is
    /// caught at compile time instead, because the guard's record list names every one of them.
    #[test]
    fn prime_handle_to_fd_is_the_ioctl_its_struct_size_gives() {
        assert_eq!(size_of::<DrmPrimeHandle>(), 12, "struct drm_prime_handle: handle, flags, fd");
        assert_eq!(PRIME_HANDLE_TO_FD, 0xc00c_642d, "_IOWR('d', 0x2d, struct drm_prime_handle)");
    }

    /// A handle that is not an open descriptor is not the buffer, so `map` refuses it rather than
    /// mapping whatever else that number might name.
    #[test]
    fn a_handle_that_is_not_a_descriptor_is_not_mistaken_for_the_buffer() {
        // Descriptor 0 exists and is far too small to be a coded buffer.
        assert!(!is_the_buffer(0, 710656), "stdin's size is not the coded buffer's");
        assert!(!is_the_buffer(-1, 710656), "a negative descriptor is not one");
    }

    #[test]
    fn a_zero_sized_object_is_an_error_before_any_syscall() {
        // A descriptor of 0 bytes is refused rather than mapped, and refused before the ioctl, so
        // the test needs no DRM at all.
        assert!(map(-1, 1, 0).is_err());
    }

    #[test]
    fn a_missing_descriptor_is_an_error_rather_than_a_mapping() {
        // There is no DRM descriptor, so this cannot reach the ioctl; it is an `Err`, never a
        // mapping built from a descriptor number that means nothing.
        assert!(map(-1, 1, 4096).is_err());
    }
}
