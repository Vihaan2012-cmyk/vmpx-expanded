//! Generic (non-VMProtect) packer unpack routing.
//!
//! A detected easy packer is routed to the EXISTING contained dynamic
//! unpack-to-OEP path in `vmpx-emulate`: `detonate` with the `OepHook`, then
//! `dump_image` and `rebuild_module_pe`. Nothing here is packer-specific: the
//! named-section packers all self-unpack at runtime via a write-then-execute
//! handoff the OEP detector catches, so one generic contained run covers them.
//!
//! # Containment / bound
//!
//! The detonation is pure in-process software interpretation (no net/syscall/
//! fs/process/FFI path) and is HARD-BOUNDED by [`PACKER_UNPACK_MAX_STEPS`] so a
//! malformed/hostile stub can never run away or OOM. No global cap is raised.
//!
//! # Soundness / provenance
//!
//! A rebuilt PE is an OBSERVED artifact from ONE run's memory state, never
//! sound static output. The routing + bound logic is unit-tested against a
//! synthetic write-then-execute OEP fixture; **real UPX/ASPack/PECompact/MPRESS
//! end-to-end unpack is UNVALIDATED here (no packer samples available offline).**

use vmpx_detect::{PackerKind, detect_packer};
use vmpx_emulate::{DetonateOptions, Stop, detonate, rebuild_module_pe};
use vmpx_pe::PeImage;

/// Modest instruction budget for a packer-unpack detonation. Easy-packer stubs
/// are short (a few hundred K to low-millions of steps), so this bounds the run
/// well below any runaway/OOM risk. It is a local cap for this path only.
pub const PACKER_UNPACK_MAX_STEPS: u64 = 2_000_000;

/// The result of routing an input through generic unpack.
#[derive(Debug)]
pub enum UnpackOutcome {
    /// No easy-packer signature matched; nothing was routed.
    NoPacker,
    /// Detected, but not routed to the generic dynamic path (the low-confidence
    /// FSG heuristic): experimental/unsupported. No output is fabricated.
    Unsupported { kind: PackerKind, confidence: u8 },
    /// Detected + routed, reached OEP under the bound, and the memory dump was
    /// rebuilt into a (loader-valid) PE. OBSERVED artifact.
    UnpackedToOep { kind: PackerKind, oep: u64, steps: usize, pe_bytes: Vec<u8> },
    /// Detected + routed, but OEP was not reached within the step bound. Fail
    /// closed: no output.
    OepNotReached { kind: PackerKind, steps: usize, stop: Stop },
    /// Detected + routed, but the detonation could not even start.
    NotStarted { kind: PackerKind, reason: String },
    /// Detected + routed + reached OEP, but the dump could not be rebuilt.
    RebuildFailed { kind: PackerKind, reason: String },
}

/// Detect a packer in `pe` and, when it is a supported named-section packer,
/// attempt a bounded contained unpack-to-OEP. Never executes the sample
/// natively; the whole run is the software emulator capped at `max_steps`.
#[must_use]
pub fn route_generic_unpack(pe: &PeImage<'_>, max_steps: u64) -> UnpackOutcome {
    let detected = detect_packer(pe);
    let kind = detected.kind;
    match kind {
        PackerKind::None => return UnpackOutcome::NoPacker,
        k if !k.routes_to_generic_unpack() => {
            return UnpackOutcome::Unsupported { kind, confidence: detected.confidence };
        }
        _ => {}
    }

    let opts = DetonateOptions { max_steps, dump: true, ..DetonateOptions::default() };
    let result = match detonate(pe, &opts) {
        Ok(result) => result,
        Err(error) => {
            return UnpackOutcome::NotStarted { kind, reason: format!("{error:?}") };
        }
    };

    match (result.observed.oep, result.dump) {
        (Some(oep), Some(dump)) => {
            let size_of_image = u64::from(pe.optional.size_of_image);
            match rebuild_module_pe(&dump, size_of_image) {
                Ok(pe_bytes) => {
                    UnpackOutcome::UnpackedToOep { kind, oep, steps: result.steps, pe_bytes }
                }
                Err(error) => UnpackOutcome::RebuildFailed { kind, reason: error.to_string() },
            }
        }
        // No OEP observed within the bound (or no dump): fail closed.
        _ => UnpackOutcome::OepNotReached { kind, steps: result.steps, stop: result.stop },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMAGE_BASE: u64 = 0x1_4000_0000;
    const OEP_TARGET: u64 = IMAGE_BASE + 0x2100;

    fn put(d: &mut [u8], at: usize, bytes: &[u8]) {
        d[at..at + bytes.len()].copy_from_slice(bytes);
    }

    /// A minimal self-unpacking PE32+ mirroring the vmpx-emulate detonate
    /// fixture: a write-then-execute OEP handoff. `names` sets the two section
    /// names so the detector sees a chosen packer; `chars0` sets the first
    /// section's characteristics (for the FSG-heuristic negative case).
    fn synth_packed_pe(names: (&[u8], &[u8]), chars0: u32) -> Vec<u8> {
        let mut d = vec![0u8; 0x1000];
        put(&mut d, 0, &0x5A4Du16.to_le_bytes()); // MZ
        put(&mut d, 0x3C, &0x80u32.to_le_bytes()); // e_lfanew
        put(&mut d, 0x80, &0x0000_4550u32.to_le_bytes()); // PE\0\0
        put(&mut d, 0x84, &0x8664u16.to_le_bytes()); // x64
        put(&mut d, 0x86, &2u16.to_le_bytes()); // 2 sections
        put(&mut d, 0x94, &0xF0u16.to_le_bytes()); // SizeOfOptionalHeader
        put(&mut d, 0x96, &0x2022u16.to_le_bytes()); // characteristics (DLL)

        let o = 0x98usize;
        put(&mut d, o, &0x20Bu16.to_le_bytes()); // PE32+
        put(&mut d, o + 16, &0x1000u32.to_le_bytes()); // entry RVA
        put(&mut d, o + 24, &IMAGE_BASE.to_le_bytes()); // image base
        put(&mut d, o + 32, &0x1000u32.to_le_bytes()); // section alignment
        put(&mut d, o + 36, &0x200u32.to_le_bytes()); // file alignment
        put(&mut d, o + 56, &0x4000u32.to_le_bytes()); // size of image
        put(&mut d, o + 60, &0x200u32.to_le_bytes()); // size of headers
        put(&mut d, o + 68, &3u16.to_le_bytes()); // subsystem
        put(&mut d, o + 108, &16u32.to_le_bytes()); // # rva/sizes

        // Section 0 holds the entry stub; section 1 holds the written OEP page.
        for (i, (name, rva, raw, flags)) in [
            (names.0, 0x1000u32, 0x400u32, chars0),
            (names.1, 0x2000, 0x600, 0xC000_0040u32), // RW
        ]
        .into_iter()
        .enumerate()
        {
            let at = 0x188 + i * 40;
            put(&mut d, at, name);
            put(&mut d, at + 8, &0x200u32.to_le_bytes()); // VirtualSize
            put(&mut d, at + 12, &rva.to_le_bytes()); // RVA
            put(&mut d, at + 16, &0x200u32.to_le_bytes()); // SizeOfRawData
            put(&mut d, at + 20, &raw.to_le_bytes()); // PointerToRawData
            put(&mut d, at + 36, &flags.to_le_bytes()); // Characteristics
        }

        // Entry stub at file 0x400: mov rbx, OEP_TARGET ; mov [rbx], rbx ; jmp rbx.
        put(&mut d, 0x400, &[0x48, 0xBB]);
        put(&mut d, 0x402, &OEP_TARGET.to_le_bytes());
        put(&mut d, 0x40A, &[0x48, 0x89, 0x1B]); // mov [rbx], rbx
        put(&mut d, 0x40D, &[0xFF, 0xE3]); // jmp rbx
        d
    }

    const RX: u32 = 0x6000_0020;

    #[test]
    fn upx_packed_pe_unpacks_to_oep_within_bound() {
        let data = synth_packed_pe((b"UPX1", b"UPX0"), RX);
        let pe = PeImage::parse(&data).expect("parses");
        match route_generic_unpack(&pe, PACKER_UNPACK_MAX_STEPS) {
            UnpackOutcome::UnpackedToOep { kind, oep, pe_bytes, .. } => {
                assert_eq!(kind, PackerKind::Upx);
                assert_eq!(oep, OEP_TARGET);
                // Rebuilt bytes reparse as a PE (what the CLI verifies on disk).
                PeImage::parse(&pe_bytes).expect("rebuilt PE reparses");
            }
            other => panic!("expected UnpackedToOep, got {other:?}"),
        }
    }

    #[test]
    fn tiny_bound_fails_closed() {
        // A 1-step budget cannot reach OEP: the bound must trigger a fail-closed
        // path (no OEP, no fabricated output), not a panic.
        let data = synth_packed_pe((b"UPX1", b"UPX0"), RX);
        let pe = PeImage::parse(&data).expect("parses");
        match route_generic_unpack(&pe, 1) {
            UnpackOutcome::OepNotReached { kind, .. } => assert_eq!(kind, PackerKind::Upx),
            other => panic!("expected OepNotReached under tiny bound, got {other:?}"),
        }
    }

    #[test]
    fn clean_pe_is_not_routed() {
        let data = synth_packed_pe((b".text", b".data"), RX);
        let pe = PeImage::parse(&data).expect("parses");
        assert!(matches!(
            route_generic_unpack(&pe, PACKER_UNPACK_MAX_STEPS),
            UnpackOutcome::NoPacker
        ));
    }

    #[test]
    fn fsg_heuristic_is_unsupported_not_fabricated() {
        // Non-standard, writable+executable first section + 2 sections trips the
        // FSG heuristic, which is NOT routed to a dynamic run.
        let data = synth_packed_pe((b"\x00\x00", b".data"), 0xE000_0020 /* RWX */);
        let pe = PeImage::parse(&data).expect("parses");
        match route_generic_unpack(&pe, PACKER_UNPACK_MAX_STEPS) {
            UnpackOutcome::Unsupported { kind, .. } => assert_eq!(kind, PackerKind::Fsg),
            other => panic!("expected Unsupported for FSG heuristic, got {other:?}"),
        }
    }
}
