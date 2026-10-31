//! Rebuild an unpacked, import-fixed PE from an OEP memory dump plus a set of
//! RESOLVED `slot -> module!name` import bindings.
//!
//! # Soundness rule
//!
//! Only bindings the caller marks resolved (exact-match against a known export
//! model) are written to the import directory. Unresolved slots are never
//! given a name: they are left byte-for-byte as dumped and listed in
//! [`UnpackedReport::unresolved_slots`]. A dump that does not cover the headers,
//! every section, the entry point, or every resolved slot is rejected rather
//! than turned into a broken PE.
//!
//! The IAT stays at its ORIGINAL slot RVAs, so unpacked code that calls
//! `[rip+slot]` needs no rewriting. Consecutive same-module slots share one
//! import descriptor; a module whose slots are scattered gets one descriptor
//! per contiguous run (a DLL may legally appear in several descriptors).

#![allow(clippy::cast_possible_truncation)]

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;
use vmpx_pe::{IMAGE_DIRECTORY_ENTRY_IAT, IMAGE_DIRECTORY_ENTRY_IMPORT, PeImage, PeKind, Rva};

use crate::{
    AppendSection, RebuildError, append_section, next_section_rva, relayout_conventional,
    validate_for_loader,
};

const SLOT: u64 = 8;
const IMAGE_SCN_CNT_INITIALIZED_DATA: u32 = 0x0000_0040;
const IMAGE_SCN_CNT_UNINITIALIZED_DATA: u32 = 0x0000_0080;
const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const IMAGE_SCN_MEM_READ: u32 = 0x4000_0000;
const IMAGE_SCN_MEM_WRITE: u32 = 0x8000_0000;
const DIR_SECURITY: usize = 4;
const DIR_BOUND_IMPORT: usize = 11;
const MAX_IMAGE: u64 = 1 << 30;

/// One contiguous run of dumped guest memory.
#[derive(Clone, Copy, Debug)]
pub struct DumpedRegion<'a> {
    pub va: u64,
    pub bytes: &'a [u8],
    pub write: bool,
    pub exec: bool,
}

/// A slot proven (by exact address match) to hold `module!name`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedSlot {
    pub slot_rva: u64,
    pub module: String,
    pub name: String,
}

pub struct UnpackedInput<'a> {
    pub image_base: u64,
    /// OEP as an RVA; must land in a dumped executable section.
    pub oep_rva: u32,
    pub regions: &'a [DumpedRegion<'a>],
    pub resolved: &'a [ResolvedSlot],
    /// Slot RVAs that were filled but did not resolve. Flagged, never named.
    pub unresolved: &'a [u64],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnpackedReport {
    pub size_of_image: u64,
    pub file_size: usize,
    pub oep_rva: u32,
    /// Distinct modules written to the import directory.
    pub modules: Vec<String>,
    pub descriptors: usize,
    pub imports_written: usize,
    /// Slots flagged unresolved and left untouched.
    pub unresolved_slots: Vec<u64>,
    pub import_table_rva: u32,
    pub import_table_size: u32,
    /// `[start, end)` RVA span written to the IAT directory.
    pub iat_span: (u32, u32),
    /// Security / bound-import directories cleared because they are invalid in
    /// a memory image.
    pub cleared_directories: Vec<usize>,
    /// `Some` when the loader-consumed structures failed validation (the PE
    /// still re-parses and its imports match; the packer's other structures
    /// are the cause).
    pub loader_issue: Option<String>,
}

#[derive(Debug, Error)]
pub enum UnpackedError {
    #[error("dump does not cover the PE headers at the image base")]
    MissingHeaders,
    #[error("dumped headers are not a PE32+ image: {0}")]
    BadHeaders(String),
    #[error("size_of_image {0:#X} is zero or exceeds the {MAX_IMAGE:#X} cap")]
    BadSizeOfImage(u64),
    #[error("dump is incomplete: {what} at RVA {rva:#X} (+{len:#X}) is not covered")]
    IncompleteDump {
        what: &'static str,
        rva: u64,
        len: u64,
    },
    #[error("OEP RVA {0:#X} is not in a dumped executable section")]
    OepNotExecutable(u32),
    #[error("slot RVA {0:#X} is not 8-aligned or lies outside the image")]
    BadSlot(u64),
    #[error("slot RVA {0:#X} is bound to two different imports")]
    ConflictingSlot(u64),
    #[error("slot RVA {0:#X} is both resolved and unresolved")]
    ResolvedAndUnresolved(u64),
    #[error("invalid {what} {value:?}: must be nonempty ASCII without NUL")]
    BadName { what: &'static str, value: String },
    #[error("no resolved imports: refusing to emit an import-less rebuild")]
    NothingResolved,
    #[error("pe: {0}")]
    Pe(#[from] RebuildError),
    #[error("layout: {0}")]
    Layout(String),
    #[error("rebuilt PE failed re-parse check: {0}")]
    Verify(String),
}

fn rd32(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}

fn wr32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}

/// `(optional_header_offset, data_directory_offset, number_of_directories)`.
fn header_offsets(b: &[u8]) -> Result<(usize, usize, u32), UnpackedError> {
    let bad = |m: &str| UnpackedError::BadHeaders(m.to_owned());
    if b.get(0..2) != Some(b"MZ") {
        return Err(bad("no MZ"));
    }
    let nt = rd32(b, 0x3C).ok_or_else(|| bad("no e_lfanew"))? as usize;
    if b.get(nt..nt + 4) != Some(b"PE\0\0") {
        return Err(bad("no PE signature"));
    }
    let opt = nt + 24;
    if b.get(opt..opt + 2) != Some(&0x20B_u16.to_le_bytes()) {
        return Err(bad("not PE32+"));
    }
    let n = rd32(b, opt + 108).ok_or_else(|| bad("truncated optional header"))?;
    Ok((opt, opt + 112, n))
}

fn is_clean_name(s: &str) -> bool {
    !s.is_empty() && s.is_ascii() && !s.contains('\0')
}

fn merge(mut spans: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    spans.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::new();
    for (lo, hi) in spans {
        match out.last_mut() {
            Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
            _ => out.push((lo, hi)),
        }
    }
    out
}

fn covered(spans: &[(u64, u64)], lo: u64, hi: u64) -> bool {
    lo >= hi || spans.iter().any(|&(a, b)| a <= lo && hi <= b)
}

struct Run {
    module: String,
    slots: Vec<(u64, String)>,
}

/// Rebuild an unpacked PE from an OEP dump with a resolved-only import table.
///
/// # Errors
///
/// [`UnpackedError`] for an incomplete dump, malformed headers or bindings,
/// no resolved imports, or a rebuilt image whose re-parsed import directory
/// does not match the bindings that went in.
#[allow(clippy::too_many_lines)]
#[allow(clippy::missing_panics_doc)]
pub fn rebuild_unpacked(
    input: &UnpackedInput<'_>,
) -> Result<(Vec<u8>, UnpackedReport), UnpackedError> {
    let base = input.image_base;

    // Validate and group bindings first; no allocation of the big image yet.
    let mut by_slot: BTreeMap<u64, (&str, &str)> = BTreeMap::new();
    for r in input.resolved {
        if r.slot_rva % SLOT != 0 {
            return Err(UnpackedError::BadSlot(r.slot_rva));
        }
        for (what, v) in [("module", &r.module), ("symbol", &r.name)] {
            if !is_clean_name(v) {
                return Err(UnpackedError::BadName {
                    what,
                    value: v.clone(),
                });
            }
        }
        if let Some(prev) = by_slot.insert(r.slot_rva, (&r.module, &r.name))
            && prev != (r.module.as_str(), r.name.as_str())
        {
            return Err(UnpackedError::ConflictingSlot(r.slot_rva));
        }
    }
    if by_slot.is_empty() {
        return Err(UnpackedError::NothingResolved);
    }
    let unresolved: BTreeSet<u64> = input.unresolved.iter().copied().collect();
    if let Some(&s) = unresolved.iter().find(|s| by_slot.contains_key(s)) {
        return Err(UnpackedError::ResolvedAndUnresolved(s));
    }

    // Header region -> SizeOfImage.
    let head = input
        .regions
        .iter()
        .find(|r| r.va <= base && base < r.va + r.bytes.len() as u64)
        .ok_or(UnpackedError::MissingHeaders)?;
    let head_bytes = &head.bytes[(base - head.va) as usize..];
    let (opt, _, _) = header_offsets(head_bytes)?;
    let size_of_image = u64::from(
        rd32(head_bytes, opt + 56).ok_or_else(|| UnpackedError::BadHeaders("truncated".into()))?,
    );
    if size_of_image == 0 || size_of_image > MAX_IMAGE {
        return Err(UnpackedError::BadSizeOfImage(size_of_image));
    }
    let len = size_of_image as usize;
    let image_end = base + size_of_image;

    // Flat RVA-layout image + coverage.
    let mut image = vec![0u8; len];
    let mut spans = Vec::new();
    for r in input.regions {
        if r.va < base || r.va >= image_end {
            continue;
        }
        let off = (r.va - base) as usize;
        let take = r.bytes.len().min(len - off);
        image[off..off + take].copy_from_slice(&r.bytes[..take]);
        spans.push((off as u64, (off + take) as u64));
    }
    let spans = merge(spans);

    let (opt, dirs, ndirs) = header_offsets(&image)?;
    if ndirs < 13 {
        return Err(UnpackedError::BadHeaders(format!(
            "only {ndirs} data directories"
        )));
    }

    // Section table edits (raw == VA), coverage + OEP checks.
    let mut edits: Vec<(usize, u32, u32, u32)> = Vec::new();
    let mut oep_ok = false;
    {
        let pe = PeImage::parse(&image).map_err(|e| UnpackedError::BadHeaders(e.to_string()))?;
        if pe.optional.kind != PeKind::Pe32Plus {
            return Err(UnpackedError::BadHeaders("not PE32+".into()));
        }
        let table =
            pe.nt_headers_offset.0 as usize + 24 + usize::from(pe.coff.size_of_optional_header);
        if !covered(&spans, 0, u64::from(pe.optional.size_of_headers)) {
            return Err(UnpackedError::IncompleteDump {
                what: "headers",
                rva: 0,
                len: u64::from(pe.optional.size_of_headers),
            });
        }
        for (i, s) in pe.sections.iter().enumerate() {
            let va = u64::from(s.virtual_address.0);
            let vs = u64::from(s.virtual_size).min(size_of_image.saturating_sub(va));
            if !covered(&spans, va, va + vs) {
                return Err(UnpackedError::IncompleteDump {
                    what: "section",
                    rva: va,
                    len: vs,
                });
            }
            let region = input
                .regions
                .iter()
                .find(|r| r.va <= base + va && base + va < r.va + r.bytes.len() as u64);
            let has_content = image[va as usize..(va + vs) as usize]
                .iter()
                .any(|&b| b != 0);
            let mut chars = s.characteristics
                & !(IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE | IMAGE_SCN_MEM_EXECUTE);
            if has_content {
                chars &= !IMAGE_SCN_CNT_UNINITIALIZED_DATA;
                chars |= IMAGE_SCN_CNT_INITIALIZED_DATA;
            }
            chars |= IMAGE_SCN_MEM_READ;
            if let Some(r) = region {
                if r.write {
                    chars |= IMAGE_SCN_MEM_WRITE;
                }
                if r.exec {
                    chars |= IMAGE_SCN_MEM_EXECUTE;
                }
            }
            if chars & IMAGE_SCN_MEM_EXECUTE != 0
                && (va..va + vs).contains(&u64::from(input.oep_rva))
            {
                oep_ok = true;
            }
            edits.push((table + i * 40, vs as u32, va as u32, chars));
        }
    }
    if !oep_ok {
        return Err(UnpackedError::OepNotExecutable(input.oep_rva));
    }

    // Every resolved slot must be a dumped, in-image 8-byte cell.
    for &s in by_slot.keys() {
        if s + SLOT > size_of_image || !covered(&spans, s, s + SLOT) {
            return Err(UnpackedError::BadSlot(s));
        }
    }
    let unresolved_slots: Vec<u64> = unresolved
        .into_iter()
        .filter(|s| s + SLOT <= size_of_image)
        .collect();

    for (h, raw_size, raw_ptr, chars) in edits {
        wr32(&mut image, h + 16, raw_size);
        wr32(&mut image, h + 20, raw_ptr);
        wr32(&mut image, h + 36, chars);
    }
    wr32(&mut image, opt + 16, input.oep_rva);
    let mut cleared = Vec::new();
    for d in [DIR_SECURITY, DIR_BOUND_IMPORT] {
        let o = dirs + d * 8;
        if rd32(&image, o) != Some(0) || rd32(&image, o + 4) != Some(0) {
            cleared.push(d);
            wr32(&mut image, o, 0);
            wr32(&mut image, o + 4, 0);
        }
    }

    // Group into descriptor runs: consecutive slots of one module.
    let mut runs: Vec<Run> = Vec::new();
    let mut prev: Option<u64> = None;
    for (&slot, &(module, name)) in &by_slot {
        let extend = prev == Some(slot - SLOT) && runs.last().is_some_and(|r| r.module == module);
        if extend {
            runs.last_mut()
                .expect("run")
                .slots
                .push((slot, name.to_owned()));
        } else {
            runs.push(Run {
                module: module.to_owned(),
                slots: vec![(slot, name.to_owned())],
            });
        }
        prev = Some(slot);
    }

    // Lay out the table: descriptors | INT arrays | hint/name | DLL names.
    let table_rva = u64::from(next_section_rva(&image)?.0);
    let desc_bytes = (runs.len() + 1) * 20;
    let int_bytes: usize = runs.iter().map(|r| (r.slots.len() + 1) * 8).sum();
    let mut hn_off: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    let mut hn: Vec<u8> = Vec::new();
    let hn_base = desc_bytes + int_bytes;
    for r in &runs {
        for (_, name) in &r.slots {
            hn_off
                .entry((r.module.as_str(), name.as_str()))
                .or_insert_with(|| {
                    let off = hn_base + hn.len();
                    hn.extend_from_slice(&[0, 0]);
                    hn.extend_from_slice(name.as_bytes());
                    hn.push(0);
                    if !hn.len().is_multiple_of(2) {
                        hn.push(0);
                    }
                    off
                });
        }
    }
    let dll_base = hn_base + hn.len();
    let mut dll_off: BTreeMap<&str, usize> = BTreeMap::new();
    let mut dlls: Vec<u8> = Vec::new();
    for r in &runs {
        dll_off.entry(r.module.as_str()).or_insert_with(|| {
            let off = dll_base + dlls.len();
            dlls.extend_from_slice(r.module.as_bytes());
            dlls.push(0);
            off
        });
    }
    let total = dll_base + dlls.len();
    let mut data = vec![0u8; total];
    let rva_of = |off: usize| (table_rva + off as u64) as u32;
    let mut int_cursor = desc_bytes;
    for (i, r) in runs.iter().enumerate() {
        let d = i * 20;
        wr32(&mut data, d, rva_of(int_cursor));
        wr32(&mut data, d + 12, rva_of(dll_off[r.module.as_str()]));
        wr32(&mut data, d + 16, r.slots[0].0 as u32);
        for (j, (slot, name)) in r.slots.iter().enumerate() {
            let thunk = u64::from(rva_of(hn_off[&(r.module.as_str(), name.as_str())]));
            data[int_cursor + j * 8..int_cursor + j * 8 + 8].copy_from_slice(&thunk.to_le_bytes());
            let s = *slot as usize;
            image[s..s + 8].copy_from_slice(&thunk.to_le_bytes());
        }
        int_cursor += (r.slots.len() + 1) * 8;
    }
    data[hn_base..hn_base + hn.len()].copy_from_slice(&hn);
    data[dll_base..].copy_from_slice(&dlls);

    // Conventional file layout, then the table section.
    let (laid_out, _) =
        relayout_conventional(&image).map_err(|e| UnpackedError::Layout(e.to_string()))?;
    drop(image);
    let appended = append_section(
        &laid_out,
        AppendSection {
            name: *b".vimp\0\0\0",
            data: &data,
            characteristics: IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ,
        },
    )?;
    drop(laid_out);
    if u64::from(appended.section.rva.0) != table_rva {
        return Err(UnpackedError::Verify(
            "appended section landed at an unexpected RVA".into(),
        ));
    }
    let mut out = appended.bytes;

    let first = *by_slot.keys().next().expect("nonempty");
    let last = *by_slot.keys().next_back().expect("nonempty");
    let iat_span = (first as u32, (last + SLOT) as u32);
    let (_, dirs, _) = header_offsets(&out)?;
    let imp = dirs + IMAGE_DIRECTORY_ENTRY_IMPORT * 8;
    wr32(&mut out, imp, rva_of(0));
    wr32(&mut out, imp + 4, desc_bytes as u32);
    let iat = dirs + IMAGE_DIRECTORY_ENTRY_IAT * 8;
    wr32(&mut out, iat, iat_span.0);
    wr32(&mut out, iat + 4, iat_span.1 - iat_span.0);

    // Re-parse and require the imports to be exactly what went in.
    let pe = PeImage::parse(&out).map_err(|e| UnpackedError::Verify(e.to_string()))?;
    let got = pe.report().imports;
    if got.len() != runs.len() {
        return Err(UnpackedError::Verify(format!(
            "{} descriptors re-parsed, {} written",
            got.len(),
            runs.len()
        )));
    }
    for (m, r) in got.iter().zip(&runs) {
        let same = m.name == r.module
            && m.first_thunk == Rva(r.slots[0].0 as u32)
            && m.entries.len() == r.slots.len()
            && m.entries.iter().zip(&r.slots).all(|(e, (slot, name))| {
                u64::from(e.iat_rva.0) == *slot && e.name.as_deref() == Some(name.as_str())
            });
        if !same {
            return Err(UnpackedError::Verify(format!(
                "module {} mismatch after re-parse",
                r.module
            )));
        }
    }
    drop(pe);
    let loader_issue = validate_for_loader(&out).err().map(|e| e.to_string());

    let modules: Vec<String> = runs
        .iter()
        .map(|r| r.module.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let report = UnpackedReport {
        size_of_image,
        file_size: out.len(),
        oep_rva: input.oep_rva,
        modules,
        descriptors: runs.len(),
        imports_written: by_slot.len(),
        unresolved_slots,
        import_table_rva: rva_of(0),
        import_table_size: total as u32,
        iat_span,
        cleared_directories: cleared,
        loader_issue,
    };
    Ok((out, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: u64 = 0x1_4000_0000;
    const SIZE: usize = 0x4000;

    fn put32(b: &mut [u8], o: usize, v: u32) {
        b[o..o + 4].copy_from_slice(&v.to_le_bytes());
    }

    /// RVA-layout PE32+: headers 0, .text 0x1000 (code), .data 0x2000 (IAT at 0x2000),
    /// 1 spare section slot, `SizeOfImage` 0x4000 (0x3000 unmapped tail is .bss-free).
    fn dump_bytes() -> Vec<u8> {
        let mut b = vec![0u8; SIZE];
        b[0..2].copy_from_slice(b"MZ");
        put32(&mut b, 0x3C, 0x80);
        let nt = 0x80;
        b[nt..nt + 4].copy_from_slice(b"PE\0\0");
        b[nt + 4..nt + 6].copy_from_slice(&0x8664_u16.to_le_bytes());
        b[nt + 6..nt + 8].copy_from_slice(&2_u16.to_le_bytes());
        b[nt + 20..nt + 22].copy_from_slice(&0xF0_u16.to_le_bytes());
        b[nt + 22..nt + 24].copy_from_slice(&0x2022_u16.to_le_bytes());
        let o = nt + 24;
        b[o..o + 2].copy_from_slice(&0x20B_u16.to_le_bytes());
        b[o + 24..o + 32].copy_from_slice(&BASE.to_le_bytes());
        put32(&mut b, o + 32, 0x1000);
        put32(&mut b, o + 36, 0x200);
        put32(&mut b, o + 56, SIZE as u32);
        put32(&mut b, o + 60, 0x400);
        b[o + 68..o + 70].copy_from_slice(&3_u16.to_le_bytes());
        put32(&mut b, o + 108, 16);
        let sec = o + 0xF0;
        for (i, (name, va, ch)) in [
            (b".text\0\0\0", 0x1000_u32, 0x6000_0020_u32),
            (b".data\0\0\0", 0x2000, 0xC000_0040),
        ]
        .into_iter()
        .enumerate()
        {
            let s = sec + i * 40;
            b[s..s + 8].copy_from_slice(name);
            put32(&mut b, s + 8, 0x1000);
            put32(&mut b, s + 12, va);
            put32(&mut b, s + 16, 0x1000);
            put32(&mut b, s + 20, va);
            put32(&mut b, s + 36, ch);
        }
        b[0x1000] = 0xC3;
        // Loader-style values already in the IAT (model addresses).
        for i in 0..4 {
            b[0x2000 + i * 8..0x2000 + i * 8 + 8]
                .copy_from_slice(&(0x20_0000_0000_u64 + i as u64).to_le_bytes());
        }
        b
    }

    fn slot(rva: u64, m: &str, n: &str) -> ResolvedSlot {
        ResolvedSlot {
            slot_rva: rva,
            module: m.into(),
            name: n.into(),
        }
    }

    fn input<'a>(
        regions: &'a [DumpedRegion<'a>],
        resolved: &'a [ResolvedSlot],
        unresolved: &'a [u64],
    ) -> UnpackedInput<'a> {
        UnpackedInput {
            image_base: BASE,
            oep_rva: 0x1000,
            regions,
            resolved,
            unresolved,
        }
    }

    #[test]
    fn rebuilds_exactly_the_resolved_imports() {
        let bytes = dump_bytes();
        let regions = [DumpedRegion {
            va: BASE,
            bytes: &bytes,
            write: false,
            exec: true,
        }];
        let resolved = [
            slot(0x2000, "kernel32.dll", "ExitProcess"),
            slot(0x2008, "kernel32.dll", "Sleep"),
            slot(0x2018, "user32.dll", "MessageBoxA"),
        ];
        // 0x2010 was filled but did not resolve: must be flagged, not named.
        let (out, rep) = rebuild_unpacked(&input(&regions, &resolved, &[0x2010])).unwrap();

        let pe = PeImage::parse(&out).unwrap();
        let mut listed: Vec<(String, String, u32)> = pe
            .report()
            .imports
            .iter()
            .flat_map(|m| {
                m.entries
                    .iter()
                    .map(|e| (m.name.clone(), e.name.clone().unwrap(), e.iat_rva.0))
            })
            .collect();
        listed.sort();
        assert_eq!(
            listed,
            vec![
                ("kernel32.dll".into(), "ExitProcess".into(), 0x2000),
                ("kernel32.dll".into(), "Sleep".into(), 0x2008),
                ("user32.dll".into(), "MessageBoxA".into(), 0x2018),
            ]
        );
        assert_eq!(rep.imports_written, 3);
        assert_eq!(rep.descriptors, 2);
        assert_eq!(rep.modules, ["kernel32.dll", "user32.dll"]);
        assert_eq!(rep.unresolved_slots, [0x2010]);
        assert_eq!(pe.optional.address_of_entry_point, Rva(0x1000));
        // The unresolved slot kept its dumped bytes; resolved slots hold thunks.
        let off = |rva: u32| pe.rva_to_file_offset(Rva(rva)).unwrap().0 as usize;
        assert_eq!(
            u64::from_le_bytes(out[off(0x2010)..off(0x2010) + 8].try_into().unwrap()),
            0x20_0000_0002
        );
        assert_ne!(
            u64::from_le_bytes(out[off(0x2000)..off(0x2000) + 8].try_into().unwrap()),
            0x20_0000_0000
        );
    }

    #[test]
    fn incomplete_dump_is_rejected() {
        let bytes = dump_bytes();
        // .data page missing from the dump.
        let regions = [DumpedRegion {
            va: BASE,
            bytes: &bytes[..0x2000],
            write: false,
            exec: true,
        }];
        let resolved = [slot(0x2000, "kernel32.dll", "Sleep")];
        assert!(matches!(
            rebuild_unpacked(&input(&regions, &resolved, &[])),
            Err(UnpackedError::IncompleteDump { .. })
        ));
        // No headers at all.
        let regions = [DumpedRegion {
            va: BASE + 0x1000,
            bytes: &bytes[0x1000..],
            write: false,
            exec: true,
        }];
        assert!(matches!(
            rebuild_unpacked(&input(&regions, &resolved, &[])),
            Err(UnpackedError::MissingHeaders)
        ));
    }

    #[test]
    fn refuses_to_invent_or_conflict() {
        let bytes = dump_bytes();
        let regions = [DumpedRegion {
            va: BASE,
            bytes: &bytes,
            write: false,
            exec: true,
        }];
        assert!(matches!(
            rebuild_unpacked(&input(&regions, &[], &[0x2000])),
            Err(UnpackedError::NothingResolved)
        ));
        let conflict = [slot(0x2000, "a.dll", "F"), slot(0x2000, "a.dll", "G")];
        assert!(matches!(
            rebuild_unpacked(&input(&regions, &conflict, &[])),
            Err(UnpackedError::ConflictingSlot(0x2000))
        ));
        let one = [slot(0x2000, "a.dll", "F")];
        assert!(matches!(
            rebuild_unpacked(&input(&regions, &one, &[0x2000])),
            Err(UnpackedError::ResolvedAndUnresolved(0x2000))
        ));
        let oob = [slot(0x9000, "a.dll", "F")];
        assert!(matches!(
            rebuild_unpacked(&input(&regions, &oob, &[])),
            Err(UnpackedError::BadSlot(0x9000))
        ));
    }
}
