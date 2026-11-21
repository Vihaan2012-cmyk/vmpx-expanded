//! Inversion-of-control seam for the symbolic-decode fallback.
//!
//! The symbolic codec recovery lives in `vmpx-mutation`, which depends on this
//! crate — so `vmpx-vm` cannot call it directly without a dependency cycle.
//! Instead a provider closure is installed at runtime (by the top-level binary,
//! which depends on both). When bootstrap role-assignment finds no valid dispatch
//! and `VMPX_SYMBOLIC_DECODE` is set, `bootstrap.rs` consults the provider to
//! recover a candidate decode codec for the operator to read. The result is an
//! *unproven* candidate (it follows opaque-predicate arms), surfaced as a
//! diagnostic, never fed into a proof.

use std::sync::OnceLock;

use iced_x86::Register;
use vmpx_pe::{PeImage, Rva};

use crate::KeyedTransformCodec;

/// `(image, entry_rva, handler_register, key_candidates) -> candidate codec`.
pub type SymbolicCodecProvider = dyn for<'a, 'b> Fn(&'a PeImage<'b>, Rva, Register, &[Register]) -> Option<KeyedTransformCodec>
    + Send
    + Sync;

static PROVIDER: OnceLock<Box<SymbolicCodecProvider>> = OnceLock::new();

/// Install the symbolic-decode fallback. Idempotent — a later install is ignored,
/// so the first wins (the top-level binary installs once at startup).
pub fn set_symbolic_codec_provider(provider: Box<SymbolicCodecProvider>) {
    let _ = PROVIDER.set(provider);
}

/// Whether the opt-in symbolic-decode fallback is enabled for this run.
#[must_use]
pub fn symbolic_decode_enabled() -> bool {
    std::env::var_os("VMPX_SYMBOLIC_DECODE").is_some()
}

/// Consult the installed provider, if any, for a candidate decode codec.
#[must_use]
pub fn symbolic_codec(
    pe: &PeImage,
    entry: Rva,
    handler: Register,
    key_candidates: &[Register],
) -> Option<KeyedTransformCodec> {
    let provider = PROVIDER.get()?;
    provider(pe, entry, handler, key_candidates)
}
