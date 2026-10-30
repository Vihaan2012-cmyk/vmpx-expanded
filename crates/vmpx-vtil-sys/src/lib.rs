//! Binding to `native/vtil-bridge` (`include/vmpx_vtil.h`).
//!
//! The types here are always available. The linked FFI and the safe
//! [`Routine`] wrapper exist only with the `vtil` feature, which is off by
//! default so the workspace builds without the native library.

#[cfg(feature = "vtil")]
mod ffi;
#[cfg(feature = "vtil")]
mod routine;

#[cfg(feature = "vtil")]
pub use routine::{RegisterId, Routine, revision};

use thiserror::Error;

/// True when this build links the native bridge.
pub const NATIVE_LINKED: bool = cfg!(feature = "vtil");

/// Mirror of `vmpx_vtil_status`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Status {
    Ok = 0,
    InvalidArgument = 1,
    InvalidState = 2,
    BufferTooSmall = 3,
    InternalError = 4,
}

impl Status {
    #[must_use]
    pub fn from_raw(raw: u32) -> Self {
        match raw {
            0 => Self::Ok,
            1 => Self::InvalidArgument,
            2 => Self::InvalidState,
            3 => Self::BufferTooSmall,
            _ => Self::InternalError,
        }
    }
}

/// Mirror of `vmpx_vtil_stats`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(C)]
pub struct Stats {
    pub blocks: u64,
    pub instructions: u64,
    pub branches: u64,
    pub local_optimizations: u64,
}

#[derive(Debug, Error)]
#[error("vtil bridge {status:?}: {message}")]
pub struct VtilError {
    pub status: Status,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_maps_header_values() {
        assert_eq!(Status::from_raw(0), Status::Ok);
        assert_eq!(Status::from_raw(3), Status::BufferTooSmall);
        assert_eq!(Status::from_raw(99), Status::InternalError);
    }

    #[test]
    fn stats_layout_matches_header() {
        assert_eq!(std::mem::size_of::<Stats>(), 32);
    }

    #[test]
    fn native_is_off_by_default() {
        assert_eq!(NATIVE_LINKED, cfg!(feature = "vtil"));
    }
}
