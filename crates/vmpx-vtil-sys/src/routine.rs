use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr::{self, NonNull};

use crate::{Stats, Status, VtilError, ffi};

const ERROR_CAPACITY: usize = 512;

type ErrorBuffer = [c_char; ERROR_CAPACITY];

/// Handle for a register allocated in a [`Routine`] (1-based).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegisterId(pub u64);

/// Owned native VTIL routine; freed on drop.
pub struct Routine {
    raw: NonNull<ffi::RawRoutine>,
}

/// Pinned VTIL-Core revision the bridge was built against.
#[must_use]
pub fn revision() -> &'static str {
    // SAFETY: returns a static NUL-terminated string.
    unsafe { CStr::from_ptr(ffi::vmpx_vtil_revision()) }
        .to_str()
        .unwrap_or("unknown")
}

fn check(status: u32, buffer: &ErrorBuffer) -> Result<(), VtilError> {
    let status = Status::from_raw(status);
    if status == Status::Ok {
        return Ok(());
    }
    // SAFETY: the buffer is zero-initialised and the bridge NUL-terminates
    // within its capacity, so a terminator always exists.
    let message = unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    Err(VtilError { status, message })
}

impl Routine {
    /// # Errors
    /// Returns the bridge error if routine creation fails.
    pub fn new_amd64(entry_vip: u64) -> Result<Self, VtilError> {
        let mut out = ptr::null_mut();
        let mut err: ErrorBuffer = [0; ERROR_CAPACITY];
        // SAFETY: out-pointers are valid; error buffer capacity matches.
        let status = unsafe {
            ffi::vmpx_vtil_create_amd64(entry_vip, &raw mut out, err.as_mut_ptr(), ERROR_CAPACITY)
        };
        check(status, &err)?;
        let raw = NonNull::new(out).ok_or_else(|| VtilError {
            status: Status::InternalError,
            message: "bridge returned a null routine".into(),
        })?;
        Ok(Self { raw })
    }

    /// # Errors
    /// Returns the bridge error for an invalid width or state.
    pub fn alloc_register(&mut self, bit_width: u16) -> Result<RegisterId, VtilError> {
        let mut id = 0u64;
        let mut err: ErrorBuffer = [0; ERROR_CAPACITY];
        // SAFETY: `self.raw` is live; pointers are valid for the call.
        let status = unsafe {
            ffi::vmpx_vtil_alloc_register(
                self.raw.as_ptr(),
                bit_width,
                &raw mut id,
                err.as_mut_ptr(),
                ERROR_CAPACITY,
            )
        };
        check(status, &err)?;
        Ok(RegisterId(id))
    }

    /// # Errors
    /// Returns the bridge error for an unknown register or finished block.
    pub fn emit_mov_imm(&mut self, reg: RegisterId, value: u64) -> Result<(), VtilError> {
        let mut err: ErrorBuffer = [0; ERROR_CAPACITY];
        // SAFETY: `self.raw` is live; error buffer capacity matches.
        let status = unsafe {
            ffi::vmpx_vtil_emit_mov_imm(
                self.raw.as_ptr(),
                reg.0,
                value,
                err.as_mut_ptr(),
                ERROR_CAPACITY,
            )
        };
        check(status, &err)
    }

    /// # Errors
    /// Returns the bridge error for an unknown register or finished block.
    pub fn emit_add_imm(&mut self, reg: RegisterId, value: u64) -> Result<(), VtilError> {
        let mut err: ErrorBuffer = [0; ERROR_CAPACITY];
        // SAFETY: `self.raw` is live; error buffer capacity matches.
        let status = unsafe {
            ffi::vmpx_vtil_emit_add_imm(
                self.raw.as_ptr(),
                reg.0,
                value,
                err.as_mut_ptr(),
                ERROR_CAPACITY,
            )
        };
        check(status, &err)
    }

    /// # Errors
    /// Returns the bridge error for an unknown register or a finished block.
    pub fn finish(&mut self, output: RegisterId) -> Result<(), VtilError> {
        let mut err: ErrorBuffer = [0; ERROR_CAPACITY];
        // SAFETY: `self.raw` is live; error buffer capacity matches.
        let status = unsafe {
            ffi::vmpx_vtil_finish(self.raw.as_ptr(), output.0, err.as_mut_ptr(), ERROR_CAPACITY)
        };
        check(status, &err)
    }

    /// # Errors
    /// Returns the bridge error if the block is not finished or VTIL throws.
    pub fn optimize(&mut self) -> Result<(), VtilError> {
        let mut err: ErrorBuffer = [0; ERROR_CAPACITY];
        // SAFETY: `self.raw` is live; error buffer capacity matches.
        let status = unsafe {
            ffi::vmpx_vtil_optimize(self.raw.as_ptr(), err.as_mut_ptr(), ERROR_CAPACITY)
        };
        check(status, &err)
    }

    /// # Errors
    /// Returns the bridge error if VTIL throws.
    pub fn stats(&self) -> Result<Stats, VtilError> {
        let mut out = Stats::default();
        let mut err: ErrorBuffer = [0; ERROR_CAPACITY];
        // SAFETY: `self.raw` is live; `Stats` is `repr(C)` matching the header.
        let status = unsafe {
            ffi::vmpx_vtil_get_stats(
                self.raw.as_ptr(),
                &raw mut out,
                err.as_mut_ptr(),
                ERROR_CAPACITY,
            )
        };
        check(status, &err)?;
        Ok(out)
    }

    /// Serializes the routine in VTIL's binary format.
    ///
    /// # Errors
    /// Returns the bridge error if serialization fails.
    pub fn serialize(&self) -> Result<Vec<u8>, VtilError> {
        let mut required = 0usize;
        let mut err: ErrorBuffer = [0; ERROR_CAPACITY];
        // SAFETY: a null output with zero capacity only queries the size.
        let status = unsafe {
            ffi::vmpx_vtil_serialize(
                self.raw.as_ptr(),
                ptr::null_mut(),
                0,
                &raw mut required,
                err.as_mut_ptr(),
                ERROR_CAPACITY,
            )
        };
        check(status, &err)?;
        let mut bytes = vec![0u8; required];
        // SAFETY: `bytes` has `required` writable bytes.
        let status = unsafe {
            ffi::vmpx_vtil_serialize(
                self.raw.as_ptr(),
                bytes.as_mut_ptr(),
                bytes.len(),
                &raw mut required,
                err.as_mut_ptr(),
                ERROR_CAPACITY,
            )
        };
        check(status, &err)?;
        bytes.truncate(required);
        Ok(bytes)
    }
}

impl Drop for Routine {
    fn drop(&mut self) {
        // SAFETY: `raw` was created by the bridge and is destroyed exactly once.
        unsafe { ffi::vmpx_vtil_destroy(self.raw.as_ptr()) };
    }
}
