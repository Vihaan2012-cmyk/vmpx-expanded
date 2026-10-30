use std::collections::{BTreeSet, VecDeque};

use serde::Serialize;
use vmpx_pe::{
    DirectoryAddressKind, FileOffset, IMAGE_DIRECTORY_ENTRY_TLS, PeImage, Rva, Section, Va,
};

use crate::{CfgLimits, CfgStatus, decode_one, is_executable_rva, recover_cfg};

const RELOC_HIGHLOW: u8 = 3;
const RELOC_DIR64: u8 = 10;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DanglingKind {
    EntryPoint,
    TlsCallback,
    TlsDirectory,
    Export,
    DataDirectory,
    Relocation,
    UnwindInfo,
    UnwindHandler,
    Branch,
    Call,
    RipRelative,
    /// A reference or code region that could not be resolved. Always blocks stripping.
    Unresolved,
}

/// One live reference into a protector section. `source` is `None` for header
/// fields; `target` is `None` only for [`DanglingKind::Unresolved`].
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct DanglingRef {
    pub source: Option<Rva>,
    pub target: Option<Rva>,
    pub kind: DanglingKind,
    pub detail: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TargetClass {
    Outside,
    Covered,
    Dangling,
}

#[must_use]
pub fn is_vmp_section_name(name: &str) -> bool {
    name.starts_with(".vmp")
}

#[must_use]
pub fn vmp_section_containing(sections: &[Section], rva: Rva) -> Option<&Section> {
    sections
        .iter()
        .find(|section| is_vmp_section_name(&section.name) && section.contains_mapped_rva(rva))
}

/// Covered targets are exact entry RVAs that now have a native replacement.
#[must_use]
pub fn classify_target(sections: &[Section], covered: &BTreeSet<Rva>, target: Rva) -> TargetClass {
    if vmp_section_containing(sections, target).is_none() {
        TargetClass::Outside
    } else if covered.contains(&target) {
        TargetClass::Covered
    } else {
        TargetClass::Dangling
    }
}

/// Lowest RVA of `[start, start + size)` that lies inside a protector section.
#[must_use]
pub fn first_vmp_overlap(sections: &[Section], start: u32, size: u32) -> Option<Rva> {
    let end = u64::from(start) + u64::from(size);
    sections
        .iter()
        .filter(|section| is_vmp_section_name(&section.name))
        .filter_map(|section| {
            let section_start = u64::from(section.virtual_address.0);
            let section_end = section_start + u64::from(section.mapped_span());
            let low = u64::from(start).max(section_start);
            (low < end.min(section_end)).then(|| u32::try_from(low).ok().map(Rva))?
        })
        .min()
}

struct Collector<'a, 'b> {
    pe: &'a PeImage<'b>,
    covered: &'a BTreeSet<Rva>,
    refs: BTreeSet<DanglingRef>,
}

impl Collector<'_, '_> {
    fn in_vmp(&self, rva: Rva) -> bool {
        vmp_section_containing(&self.pe.sections, rva).is_some()
    }

    fn target(&mut self, source: Option<Rva>, target: Rva, kind: DanglingKind, detail: &str) {
        if classify_target(&self.pe.sections, self.covered, target) == TargetClass::Dangling {
            self.refs.insert(DanglingRef {
                source,
                target: Some(target),
                kind,
                detail: detail.to_owned(),
            });
        }
    }

    fn unresolved(&mut self, source: Option<Rva>, detail: String) {
        self.refs.insert(DanglingRef {
            source,
            target: None,
            kind: DanglingKind::Unresolved,
            detail,
        });
    }

    fn va_target(&mut self, source: Option<Rva>, va: Va, kind: DanglingKind, detail: &str) {
        match self.pe.optional.image_base.checked_rva_of(va) {
            Some(rva) => self.target(source, rva, kind, detail),
            None => self.unresolved(source, format!("{detail}: VA {va} is outside the image")),
        }
    }

    fn data_directories(&mut self) {
        let pe = self.pe;
        for directory in pe.directories.iter().filter(|d| d.is_present()) {
            let start = match directory.address_kind {
                DirectoryAddressKind::Rva => Rva(directory.address),
                // A file-offset directory outside every section cannot be in .vmp.
                DirectoryAddressKind::FileOffset => {
                    match pe.file_offset_to_rva(FileOffset(u64::from(directory.address))) {
                        Ok(rva) => rva,
                        Err(_) => continue,
                    }
                }
            };
            if let Some(overlap) = first_vmp_overlap(&pe.sections, start.0, directory.size) {
                let detail = format!("{} directory", directory.name);
                self.target(None, overlap, DanglingKind::DataDirectory, &detail);
            }
        }
    }

    fn header_entries(&mut self, roots: &mut Vec<Rva>) {
        let pe = self.pe;
        let entry = pe.optional.address_of_entry_point;
        if entry.0 != 0 {
            self.target(
                None,
                entry,
                DanglingKind::EntryPoint,
                "address of entry point",
            );
            roots.push(entry);
        }
        let report = pe.report();
        if let Some(exports) = &report.exports {
            for symbol in exports.symbols.iter().filter(|s| s.forwarder.is_none()) {
                let detail = format!("export ordinal {}", symbol.ordinal);
                self.target(None, symbol.rva, DanglingKind::Export, &detail);
                roots.push(symbol.rva);
            }
        }
        match &report.tls {
            Some(tls) => {
                if tls.address_of_callbacks.0 != 0 {
                    self.va_target(
                        None,
                        tls.address_of_callbacks,
                        DanglingKind::TlsDirectory,
                        "TLS callback array",
                    );
                }
                for callback in &tls.callbacks {
                    self.va_target(None, *callback, DanglingKind::TlsCallback, "TLS callback");
                    if let Some(rva) = pe.optional.image_base.checked_rva_of(*callback) {
                        roots.push(rva);
                    }
                }
            }
            None if pe.directory(IMAGE_DIRECTORY_ENTRY_TLS).is_some() => {
                self.unresolved(None, "TLS directory present but not parsed".to_owned());
            }
            None => {}
        }
        for function in &report.exception_functions {
            if !self.in_vmp(function.begin) {
                roots.push(function.begin);
            }
        }
        self.unwind(&report.exception_functions);
        self.relocations(&report.relocations);
    }

    fn unwind(&mut self, functions: &[vmpx_pe::RuntimeFunction]) {
        for function in functions {
            if self.in_vmp(function.begin) {
                continue;
            }
            let source = Some(function.begin);
            self.target(
                source,
                function.unwind_info,
                DanglingKind::UnwindInfo,
                "unwind info",
            );
            match self.pe.parse_unwind_info(function.unwind_info) {
                Err(error) => self.unresolved(source, format!("unwind info unparsable: {error}")),
                Ok(info) => {
                    for rva in [info.exception_handler, info.handler_data]
                        .into_iter()
                        .flatten()
                    {
                        self.target(source, rva, DanglingKind::UnwindHandler, "unwind handler");
                    }
                    if let Some(chained) = info.chained_function {
                        for rva in [chained.begin, chained.unwind_info] {
                            self.target(source, rva, DanglingKind::UnwindInfo, "chained unwind");
                        }
                    }
                }
            }
        }
    }

    fn relocations(&mut self, blocks: &[vmpx_pe::BaseRelocationBlock]) {
        for entry in blocks.iter().flat_map(|block| &block.entries) {
            if entry.kind == 0 {
                continue;
            }
            let Some(slot) = entry.target_rva else {
                self.unresolved(None, format!("relocation kind {} has no slot", entry.kind));
                continue;
            };
            if self.in_vmp(slot) {
                continue;
            }
            let width = match entry.kind {
                RELOC_HIGHLOW => 4,
                RELOC_DIR64 => 8,
                kind => {
                    self.unresolved(Some(slot), format!("relocation kind {kind} unsupported"));
                    continue;
                }
            };
            let Some(value) = self.read_pointer(slot, width) else {
                self.unresolved(Some(slot), "relocation slot unreadable".to_owned());
                continue;
            };
            // A pointee outside the image is a constant, not a section reference.
            if let Some(rva) = self.pe.optional.image_base.checked_rva_of(Va(value)) {
                self.target(
                    Some(slot),
                    rva,
                    DanglingKind::Relocation,
                    "relocated pointer",
                );
            }
        }
    }

    fn read_pointer(&self, slot: Rva, width: usize) -> Option<u64> {
        let offset = usize::try_from(self.pe.rva_to_file_offset(slot).ok()?.0).ok()?;
        let bytes = self.pe.data().get(offset..offset.checked_add(width)?)?;
        let mut buffer = [0_u8; 8];
        buffer[..width].copy_from_slice(bytes);
        Some(u64::from_le_bytes(buffer))
    }

    fn walk(&mut self, roots: Vec<Rva>, limits: CfgLimits) {
        let pe = self.pe;
        let mut queue = VecDeque::from(roots);
        let mut seen = BTreeSet::new();
        let mut scanned_blocks = BTreeSet::new();
        while let Some(root) = queue.pop_front() {
            if !seen.insert(root) || self.in_vmp(root) || !is_executable_rva(pe, root) {
                continue;
            }
            let graph = match recover_cfg(pe, root, limits) {
                Ok(graph) => graph,
                Err(error) => {
                    self.unresolved(Some(root), format!("cfg recovery failed: {error}"));
                    continue;
                }
            };
            if graph.status != CfgStatus::Complete {
                self.unresolved(Some(root), format!("cfg incomplete: {:?}", graph.status));
            }
            for block in &graph.blocks {
                if self.in_vmp(block.start) || !scanned_blocks.insert(block.start) {
                    continue;
                }
                let branch_source = block.instructions.last().map_or(block.start, |i| i.rva);
                for successor in &block.successors {
                    self.target(
                        Some(branch_source),
                        *successor,
                        DanglingKind::Branch,
                        "branch target",
                    );
                }
                for call in &block.direct_calls {
                    if self.in_vmp(*call) {
                        self.target(Some(block.start), *call, DanglingKind::Call, "direct call");
                    } else {
                        queue.push_back(*call);
                    }
                }
                for instruction in &block.instructions {
                    // Re-decode: the summary's memory list omits `lea` operands.
                    let Ok(decoded) = decode_one(pe, instruction.rva) else {
                        self.unresolved(
                            Some(instruction.rva),
                            "instruction re-decode failed".into(),
                        );
                        continue;
                    };
                    if !decoded.instruction.is_ip_rel_memory_operand() {
                        continue;
                    }
                    let address = Va(decoded.instruction.ip_rel_memory_address());
                    if let Some(rva) = pe.optional.image_base.checked_rva_of(address) {
                        self.target(
                            Some(instruction.rva),
                            rva,
                            DanglingKind::RipRelative,
                            "rip-relative operand",
                        );
                    }
                }
            }
        }
    }
}

/// Finds every remaining reference into a `.vmp*` section that is not covered by
/// a native replacement. An empty result is the precondition for stripping.
#[must_use]
pub fn collect_dangling_refs(pe: &PeImage<'_>, covered: &BTreeSet<Rva>) -> Vec<DanglingRef> {
    collect_dangling_refs_with(
        pe,
        covered,
        &[],
        CfgLimits {
            max_blocks: 2_048,
            max_instructions: 60_000,
            max_instructions_per_block: 8_192,
        },
    )
}

/// As [`collect_dangling_refs`] with extra CFG roots and explicit per-root limits.
#[must_use]
pub fn collect_dangling_refs_with(
    pe: &PeImage<'_>,
    covered: &BTreeSet<Rva>,
    extra_roots: &[Rva],
    limits: CfgLimits,
) -> Vec<DanglingRef> {
    let mut collector = Collector {
        pe,
        covered,
        refs: BTreeSet::new(),
    };
    collector.data_directories();
    let mut roots = extra_roots.to_vec();
    collector.header_entries(&mut roots);
    collector.walk(roots, limits);
    collector.refs.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: u32 = 0x1000;
    const VMP: u32 = 0x2000;
    const VMP_VA: u64 = 0x1_4000_0000 + 0x2000;

    fn put16(data: &mut [u8], at: usize, value: u16) {
        data[at..at + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put32(data: &mut [u8], at: usize, value: u32) {
        data[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put64(data: &mut [u8], at: usize, value: u64) {
        data[at..at + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn section(data: &mut [u8], index: usize, name: &[u8], rva: u32, raw: u32) {
        let at = 0x84 + 20 + 0xF0 + index * 40;
        data[at..at + name.len()].copy_from_slice(name);
        put32(data, at + 8, 0x200);
        put32(data, at + 12, rva);
        put32(data, at + 16, 0x200);
        put32(data, at + 20, raw);
        put32(data, at + 36, 0x6000_0020);
    }

    /// `.text` at RVA 0x1000 (file 0x200), `.vmp0` at RVA 0x2000 (file 0x400).
    fn pe64(text: &[u8], vmp: &[u8]) -> Vec<u8> {
        let mut data = vec![0_u8; 0x600];
        put16(&mut data, 0, 0x5A4D);
        put32(&mut data, 0x3C, 0x80);
        put32(&mut data, 0x80, 0x0000_4550);
        let coff = 0x84;
        put16(&mut data, coff, 0x8664);
        put16(&mut data, coff + 2, 2);
        put16(&mut data, coff + 16, 0xF0);
        put16(&mut data, coff + 18, 0x2022);
        let optional = coff + 20;
        put16(&mut data, optional, 0x20B);
        put32(&mut data, optional + 16, TEXT);
        put32(&mut data, optional + 20, TEXT);
        put64(&mut data, optional + 24, 0x1_4000_0000);
        put32(&mut data, optional + 32, 0x1000);
        put32(&mut data, optional + 36, 0x200);
        put32(&mut data, optional + 56, 0x3000);
        put32(&mut data, optional + 60, 0x200);
        put16(&mut data, optional + 68, 3);
        put32(&mut data, optional + 108, 16);
        section(&mut data, 0, b".text", TEXT, 0x200);
        section(&mut data, 1, b".vmp0", VMP, 0x400);
        data[0x200..0x200 + text.len()].copy_from_slice(text);
        data[0x400..0x400 + vmp.len()].copy_from_slice(vmp);
        data
    }

    fn set_directory(data: &mut [u8], index: usize, rva: u32, size: u32) {
        let at = 0x84 + 20 + 112 + index * 8;
        put32(data, at, rva);
        put32(data, at + 4, size);
    }

    fn set_entry(data: &mut [u8], rva: u32) {
        put32(data, 0x84 + 20 + 16, rva);
    }

    fn refs(data: &[u8], covered: &[u32]) -> Vec<DanglingRef> {
        let pe = PeImage::parse(data).expect("valid synthetic PE");
        let covered = covered.iter().map(|rva| Rva(*rva)).collect();
        collect_dangling_refs(&pe, &covered)
    }

    fn kinds(refs: &[DanglingRef]) -> Vec<DanglingKind> {
        refs.iter().map(|r| r.kind).collect()
    }

    // call vmp; ret
    const CALL_VMP: [u8; 6] = [0xE8, 0xFB, 0x0F, 0x00, 0x00, 0xC3];

    #[test]
    fn section_classification_is_pure() {
        let data = pe64(&[0xC3], &[0x90]);
        let pe = PeImage::parse(&data).expect("valid synthetic PE");
        let covered = BTreeSet::from([Rva(VMP + 0x10)]);
        assert!(is_vmp_section_name(".vmp0") && is_vmp_section_name(".vmp12"));
        assert!(!is_vmp_section_name(".text") && !is_vmp_section_name("vmp"));
        let class = |rva| classify_target(&pe.sections, &covered, Rva(rva));
        assert_eq!(class(TEXT), TargetClass::Outside);
        assert_eq!(class(VMP + 0x10), TargetClass::Covered);
        assert_eq!(class(VMP + 0x11), TargetClass::Dangling);
        assert_eq!(class(VMP + 0x1FF), TargetClass::Dangling);
        assert_eq!(class(VMP + 0x200), TargetClass::Outside);
        assert_eq!(
            first_vmp_overlap(&pe.sections, VMP - 8, 0x10),
            Some(Rva(VMP))
        );
        assert_eq!(
            first_vmp_overlap(&pe.sections, VMP + 4, 1),
            Some(Rva(VMP + 4))
        );
        assert_eq!(first_vmp_overlap(&pe.sections, TEXT, 0x100), None);
        assert_eq!(first_vmp_overlap(&pe.sections, VMP, 0), None);
        assert_eq!(first_vmp_overlap(&pe.sections, u32::MAX, u32::MAX), None);
    }

    #[test]
    fn clean_image_has_no_dangling_refs() {
        let mut data = pe64(&[0xC3], &[0x90, 0xC3]);
        set_entry(&mut data, TEXT);
        assert!(refs(&data, &[]).is_empty());
    }

    #[test]
    fn direct_call_into_vmp_is_dangling_until_covered() {
        let mut data = pe64(&CALL_VMP, &[0xC3]);
        set_entry(&mut data, TEXT);
        let found = refs(&data, &[]);
        assert_eq!(kinds(&found), [DanglingKind::Call]);
        assert_eq!(found[0].source, Some(Rva(TEXT)));
        assert_eq!(found[0].target, Some(Rva(VMP)));
        assert!(refs(&data, &[VMP]).is_empty());
        // An unrelated covered RVA does not mask the reference.
        assert_eq!(refs(&data, &[VMP + 8]).len(), 1);
    }

    #[test]
    fn jump_and_rip_relative_operand_into_vmp_are_found() {
        // lea rax,[rip+0xFF9] (target 0x2000); jmp 0x2004
        let text = [
            0x48, 0x8D, 0x05, 0xF9, 0x0F, 0x00, 0x00, 0xE9, 0xF8, 0x0F, 0x00, 0x00,
        ];
        let mut data = pe64(&text, &[0x90; 8]);
        set_entry(&mut data, TEXT);
        let found = refs(&data, &[]);
        assert_eq!(
            kinds(&found),
            [DanglingKind::RipRelative, DanglingKind::Branch]
        );
        assert_eq!(found[0].source, Some(Rva(TEXT)));
        assert_eq!(found[0].target, Some(Rva(VMP)));
        assert_eq!(found[1].source, Some(Rva(TEXT + 7)));
        assert_eq!(found[1].target, Some(Rva(VMP + 4)));
    }

    #[test]
    fn entry_point_in_vmp_is_dangling_until_covered() {
        let mut data = pe64(&[0xC3], &[0xC3]);
        set_entry(&mut data, VMP);
        assert_eq!(kinds(&refs(&data, &[])), [DanglingKind::EntryPoint]);
        assert!(refs(&data, &[VMP]).is_empty());
    }

    #[test]
    fn data_directory_landing_in_vmp_is_dangling() {
        let mut data = pe64(&[0xC3], &[0; 0x20]);
        set_entry(&mut data, TEXT);
        set_directory(&mut data, 2, VMP + 0x10, 0x10);
        let found = refs(&data, &[]);
        assert_eq!(kinds(&found), [DanglingKind::DataDirectory]);
        assert_eq!(found[0].target, Some(Rva(VMP + 0x10)));
        set_directory(&mut data, 2, VMP - 0x8, 0x10);
        assert_eq!(refs(&data, &[]).len(), 1, "range overlap counts");
        set_directory(&mut data, 2, TEXT + 0x100, 0x10);
        assert!(refs(&data, &[]).is_empty());
    }

    #[test]
    fn tls_callback_into_vmp_is_dangling() {
        let mut text = vec![0_u8; 0x80];
        text[0] = 0xC3;
        // TLS directory at RVA 0x1040: AddressOfCallbacks -> array at 0x1070.
        put64(&mut text, 0x40 + 24, 0x1_4000_0000 + 0x1070);
        put64(&mut text, 0x70, VMP_VA);
        let mut data = pe64(&text, &[0xC3]);
        set_entry(&mut data, TEXT);
        set_directory(&mut data, IMAGE_DIRECTORY_ENTRY_TLS, TEXT + 0x40, 40);
        let found = refs(&data, &[]);
        assert_eq!(kinds(&found), [DanglingKind::TlsCallback]);
        assert_eq!(found[0].target, Some(Rva(VMP)));
        assert!(refs(&data, &[VMP]).is_empty());
    }

    #[test]
    fn relocated_pointer_into_vmp_is_dangling() {
        let mut text = vec![0_u8; 0x80];
        text[0] = 0xC3;
        put64(&mut text, 0x20, VMP_VA);
        // Reloc block for page 0x1000: one DIR64 entry at offset 0x20 plus padding.
        put32(&mut text, 0x40, TEXT);
        put32(&mut text, 0x44, 12);
        put16(&mut text, 0x48, u16::from(RELOC_DIR64) << 12 | 0x20);
        let mut data = pe64(&text, &[0xC3]);
        set_entry(&mut data, TEXT);
        set_directory(&mut data, 5, TEXT + 0x40, 12);
        let found = refs(&data, &[]);
        assert_eq!(kinds(&found), [DanglingKind::Relocation]);
        assert_eq!(found[0].source, Some(Rva(TEXT + 0x20)));
        assert!(refs(&data, &[VMP]).is_empty());
    }

    #[test]
    fn undecodable_code_is_unresolved_not_dropped() {
        // 0x06 is invalid in 64-bit mode.
        let mut data = pe64(&[0x06], &[0xC3]);
        set_entry(&mut data, TEXT);
        let found = refs(&data, &[VMP]);
        assert_eq!(kinds(&found), [DanglingKind::Unresolved]);
        assert_eq!(found[0].target, None);
    }

    #[test]
    fn call_reaches_nested_native_function() {
        // entry: call +5 -> TEXT+0x10 ; ret.  callee: call vmp; ret.
        let mut text = vec![0xCC_u8; 0x20];
        text[..6].copy_from_slice(&[0xE8, 0x0B, 0x00, 0x00, 0x00, 0xC3]);
        // call at 0x1010: next ip 0x1015, rel = 0x2000 - 0x1015
        text[0x10] = 0xE8;
        text[0x11..0x15].copy_from_slice(&(VMP - (TEXT + 0x15)).to_le_bytes());
        text[0x15] = 0xC3;
        let mut data = pe64(&text, &[0xC3]);
        set_entry(&mut data, TEXT);
        let found = refs(&data, &[]);
        assert_eq!(kinds(&found), [DanglingKind::Call]);
        assert_eq!(found[0].source, Some(Rva(TEXT + 0x10)));
    }
}
