//! The DRM descriptor's own view of a libva buffer object's storage.
//!
//! libva allocates every buffer it hands back as a DRM buffer object, and for the encode coded
//! buffer it never maps that object into the client's address space in a way the client can name:
//! `vaMapBuffer` returns a **separate** staging allocation (measured — see
//! [`Encoder::read_coded`](super::ffi::Encoder::read_coded)), and the picture is reachable only
//! through the driver's own `VACodedBufferSegment::buf` pointer, which nothing bounds.
//!
//! This module closes that gap from the other side. It is **libva that named the object**
//! (`vaAcquireBufferHandle` reports its DRM handle and its size), and **DRM that maps it** — the
//! descriptor FilmCraft already holds the display open on, asked to turn the handle into a
//! descriptor of its own, which is then mapped here. Every byte this backend reads back comes out
//! of a mapping this module created and sized itself, so no byte of the read depends on a pointer
//! the driver chose, on another process's view of the address space, or on a length we had to
//! discover.
//!
//! Nothing here links libva: it is plain DRM (one `ioctl`, `mmap`, `munmap`, `close`) plus the
//! handle libva already reported.

use std::ffi::{c_int, c_void};

/// `mmap`'s failure return, `(void *) -1`.
const MAP_FAILED: *mut c_void = -1isize as *mut c_void;

/// `PROT_READ`.
const PROT_READ: c_int = 1;
/// `MAP_SHARED`: the mapping has to see the driver's writes, which is the whole point of it.
const MAP_SHARED: c_int = 1;

/// `DRM_IOCTL_PRIME_HANDLE_TO_FD` = `_IOWR('d', 0x2d, struct drm_prime_handle)`.
///
/// The encoding is the kernel's `_IOC` one, spelled out rather than written as the constant every
/// DRM program copies around: `dir(2) | size(16) | type(8) | nr(8)`, with `dir = 3` for read/write.
/// `struct drm_prime_handle` is two `__u32` and nothing else, so `size` is `size_of` below and the
/// number cannot drift from the struct it describes.
const fn iowr(kind: u8, nr: u8, size: usize) -> std::ffi::c_ulong {
    (3 << 30) | ((size as u64) << 16) | ((kind as u64) << 8) | nr as u64
}

/// `struct drm_prime_handle`: the buffer handle in, the descriptor it exports to out.
#[repr(C)]
struct DrmPrimeHandle {
    handle: u32,
    /// The `open` flags to export with. Zero is `O_RDONLY`, which is all a mapping of coded output
    /// needs: the driver has already written it, and this process never writes into the buffer.
    flags: u32,
}

const PRIME_HANDLE_TO_FD: std::ffi::c_ulong = iowr(b'd', 0x2d, size_of::<DrmPrimeHandle>());

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

/// Whether `fd` names an open descriptor, asked of the kernel rather than inferred.
///
/// The ioctl reports failure by returning a negative `errno`, so a zero return already means the
/// export worked — but the descriptor it leaves in `handle` is only worth `mmap`ing if it really is
/// one. **It cannot be compared against the handle it replaced**: DRM hands the descriptor back in
/// the same field, and on this machine both come out `4`. So it is asked about instead.
fn is_open(fd: c_int) -> bool {
    // SAFETY: `fcntl(F_GETFD)` reads the descriptor table and takes no argument, so the variadic
    // call sets up the registers the SysV ABI requires and reads nothing that does not exist.
    unsafe { fcntl(fd, F_GETFD) != -1 }
}

/// Map the buffer object `handle` names, on the DRM descriptor `drm`, read-only, `mem_size` long.
///
/// `handle` and `mem_size` are the two numbers `vaAcquireBufferHandle` reported for it; this
/// function neither computes nor guesses either. A driver that exports a different number of bytes
/// than it reported is caught by the kernel, which fails the mapping.
pub fn map(drm: c_int, handle: u32, mem_size: usize) -> Result<Storage, String> {
    if mem_size == 0 {
        return Err("libva reported a buffer object of 0 bytes".into());
    }
    if drm < 0 {
        return Err("no DRM descriptor is open".into());
    }
    // DRM writes the exported descriptor back into `handle`, the field it came from.
    let mut request = DrmPrimeHandle { handle, flags: 0 };
    // SAFETY: `drm` is an open DRM render-node descriptor, `request` is a live
    // `struct drm_prime_handle`, and the request number is the one that struct's size gives.
    let rc = unsafe { ioctl(drm, PRIME_HANDLE_TO_FD, &mut request as *mut DrmPrimeHandle) };
    if rc != 0 {
        return Err(format!("DRM_IOCTL_PRIME_HANDLE_TO_FD refused buffer handle {handle}"));
    }
    // A descriptor number that is negative, or that names nothing open, is not mappable — and
    // `mmap` on one would fail with a number nobody can act on.
    let fd = c_int::try_from(request.handle).map_err(|_| format!("DRM_IOCTL_PRIME_HANDLE_TO_FD returned descriptor {}, which is not one", request.handle))?;
    if !is_open(fd) {
        return Err(format!("DRM_IOCTL_PRIME_HANDLE_TO_FD returned descriptor {fd} for buffer handle {handle}, which is not open"));
    }
    let export = Export { fd };
    // SAFETY: `export.fd` is the descriptor DRM just exported, `mem_size` is what libva reported
    // for the object behind it, and the mapping is `PROT_READ` and `MAP_SHARED` with no offset.
    let base = unsafe { mmap(std::ptr::null_mut(), mem_size, PROT_READ, MAP_SHARED, export.fd, 0) };
    if base.is_null() || base == MAP_FAILED {
        return Err(format!("mmap of the exported {mem_size}-byte coded buffer failed"));
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

    #[test]
    fn prime_handle_to_fd_is_the_ioctl_its_struct_size_gives() {
        // `_IOWR('d', 0x2d, struct { __u32 handle; __u32 flags; })` on the SysV Linux ABI: the
        // pinned value, written out, so a change to the struct cannot pass unnoticed.
        assert_eq!(size_of::<DrmPrimeHandle>(), 8);
        assert_eq!(PRIME_HANDLE_TO_FD, 0xc008_642d);
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
