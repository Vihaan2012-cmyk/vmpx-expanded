use std::collections::{BTreeSet, VecDeque};

use iced_x86::{OpKind, Register};
use vmpx_pe::{PeImage, PeKind, Rva, Va};

use crate::{
    CfgLimits, CfgStatus, decode_one, is_executable_rva, recover_cfg, vmp_section_containing,
};

/// Result of scanning native code for direct IAT-slot references.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct IatReferenceScan {
    /// Parsed import slots that native code provably addresses.
    pub referenced: BTreeSet<Rva>,
    /// Every slot of the parsed import directory.
    pub total_slots: usize,
    /// Native roots whose CFG was recovered.
    pub roots_walked: usize,
    /// False when any root's CFG was incomplete or failed to recover, so
    /// `referenced` is a proven subset rather than a completeness claim.
    pub exhaustive: bool,
}

fn in_vmp(pe: &PeImage<'_>, rva: Rva) -> bool {
    vmp_section_containing(&pe.sections, rva).is_some()
}

fn roots(pe: &PeImage<'_>) -> Vec<Rva> {
    let report = pe.report();
    let mut roots = vec![pe.optional.address_of_entry_point];
    if let Some(exports) = &report.exports {
        roots.extend(
            exports
                .symbols
                .iter()
                .filter(|symbol| symbol.forwarder.is_none())
                .map(|symbol| symbol.rva),
        );
    }
    if let Some(tls) = &report.tls {
        roots.extend(
            tls.callbacks
                .iter()
                .filter_map(|callback| pe.optional.image_base.checked_rva_of(*callback)),
        );
    }
    roots.extend(report.exception_functions.iter().map(|f| f.begin));
    roots.retain(|rva| rva.0 != 0);
    roots
}

/// The memory operand's absolute address when it is IP-relative or (PE32 only)
/// a bare disp32 with no base or index register.
fn absolute_memory_va(kind: PeKind, instruction: &iced_x86::Instruction) -> Option<Va> {
    if instruction.is_ip_rel_memory_operand() {
        return Some(Va(instruction.ip_rel_memory_address()));
    }
    let has_memory = (0..instruction.op_count()).any(|i| instruction.op_kind(i) == OpKind::Memory);
    (kind == PeKind::Pe32
        && has_memory
        && instruction.memory_base() == Register::None
        && instruction.memory_index() == Register::None
        && instruction.memory_displ_size() > 0)
        .then(|| Va(instruction.memory_displacement64()))
}

/// Scans non-`.vmp` executable code reachable by decoded control flow from the
/// entry point, exports, TLS callbacks and x64 unwind function starts.
#[must_use]
pub fn scan_iat_references(pe: &PeImage<'_>, limits: CfgLimits) -> IatReferenceScan {
    let slots: BTreeSet<Rva> = pe
        .report()
        .imports
        .iter()
        .flat_map(|module| module.entries.iter().map(|entry| entry.iat_rva))
        .collect();
    let mut scan = IatReferenceScan {
        total_slots: slots.len(),
        exhaustive: true,
        ..IatReferenceScan::default()
    };
    if slots.is_empty() {
        return scan;
    }
    let mut queue = VecDeque::from(roots(pe));
    let mut seen = BTreeSet::new();
    let mut scanned_blocks = BTreeSet::new();
    while let Some(root) = queue.pop_front() {
        if !seen.insert(root) || in_vmp(pe, root) || !is_executable_rva(pe, root) {
            continue;
        }
        let Ok(graph) = recover_cfg(pe, root, limits) else {
            scan.exhaustive = false;
            continue;
        };
        scan.roots_walked += 1;
        if graph.status != CfgStatus::Complete {
            scan.exhaustive = false;
        }
        for block in &graph.blocks {
            if in_vmp(pe, block.start) || !scanned_blocks.insert(block.start) {
                continue;
            }
            queue.extend(block.direct_calls.iter().copied());
            for summary in &block.instructions {
                // Re-decode: the summary's memory list omits `lea` operands.
                let Ok(decoded) = decode_one(pe, summary.rva) else {
                    scan.exhaustive = false;
                    continue;
                };
                let Some(rva) = absolute_memory_va(pe.optional.kind, &decoded.instruction)
                    .and_then(|va| pe.optional.image_base.checked_rva_of(va))
                else {
                    continue;
                };
                if slots.contains(&rva) {
                    scan.referenced.insert(rva);
                }
            }
        }
    }
    scan
}

/// IAT slots with a provable direct reference from native (non-`.vmp`) code.
/// A subset of the real application imports: VM-emulated imports and calls
/// through computed pointers are not statically visible here.
#[must_use]
pub fn referenced_iat_slots(pe: &PeImage<'_>) -> BTreeSet<Rva> {
    scan_iat_references(
        pe,
        CfgLimits {
            max_blocks: 100_000,
            max_instructions: 2_000_000,
            max_instructions_per_block: 10_000,
        },
    )
    .referenced
}

#[cfg(test)]
mod tests {
    use iced_x86::{Decoder, DecoderOptions};

    use super::*;

    fn decode(bitness: u32, bytes: &[u8]) -> iced_x86::Instruction {
        Decoder::with_ip(bitness, bytes, 0x40_1000, DecoderOptions::NONE).decode()
    }

    #[test]
    fn pe32_absolute_disp32_is_a_reference() {
        let call = decode(32, &[0xFF, 0x15, 0x50, 0x20, 0x40, 0x00]);
        assert_eq!(absolute_memory_va(PeKind::Pe32, &call), Some(Va(0x40_2050)));
        let jmp = decode(32, &[0xFF, 0x25, 0x58, 0x20, 0x40, 0x00]);
        assert_eq!(absolute_memory_va(PeKind::Pe32, &jmp), Some(Va(0x40_2058)));
    }

    #[test]
    fn based_indexed_and_register_operands_are_not_references() {
        // call [eax+0x402050]; mov eax,[eax*4+0x402050]; mov eax,ecx
        for bytes in [
            &[0xFF, 0x90, 0x50, 0x20, 0x40, 0x00][..],
            &[0x8B, 0x04, 0x85, 0x50, 0x20, 0x40, 0x00],
            &[0x8B, 0xC1],
        ] {
            assert_eq!(absolute_memory_va(PeKind::Pe32, &decode(32, bytes)), None);
        }
    }

    #[test]
    fn pe32plus_ignores_bare_disp32() {
        // mov eax,[0x402050] via SIB no-base is not an image reference on x64.
        let mov = decode(64, &[0x8B, 0x04, 0x25, 0x50, 0x20, 0x40, 0x00]);
        assert_eq!(absolute_memory_va(PeKind::Pe32Plus, &mov), None);
        let rip = decode(64, &[0xFF, 0x15, 0x00, 0x10, 0x00, 0x00]);
        assert_eq!(
            absolute_memory_va(PeKind::Pe32Plus, &rip),
            Some(Va(0x40_1006 + 0x1000))
        );
    }
}
