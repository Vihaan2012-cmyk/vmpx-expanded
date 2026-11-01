//! Sound import-stub resolution.
//!
//! A direct `call`/`jmp` into another executable section is lifted (the stub is
//! followed through unconditional `jmp`s to its `ret`) and its dispatch target
//! is resolved symbolically. A site maps to a named import only when the
//! stub is proven to behave as `call [import]` / `jmp [import]`: the `ret`
//! target is exactly a load of an import's IAT slot, argument and callee-saved
//! registers are preserved, and the caller's stack is left as the real call
//! would. Anything else stays in `unresolved` with its reason.

use std::collections::{BTreeMap, BTreeSet};

use iced_x86::{Decoder, DecoderOptions, FlowControl, Instruction, Mnemonic, OpKind};
use serde::Serialize;
use vmpx_pe::{PeKind, Rva};

use crate::{
    BinOp, BlockSemantics, Gpr, LiftOptions, NativeAlgebra, NativeExpr, NativeNode, Width,
    decode_one, is_vmp_section_name, lift_block_with,
};

const MAX_STUB_INSTRUCTIONS: usize = 4096;
const MAX_BUMP: u64 = 15;
const STUB_LIFT: LiftOptions = LiftOptions {
    assume_stack_disjoint: false,
    absolute_disjoint_from_frame: true,
    model_xmm: false,
    defer_frame_alias: false,
};
const PRESERVED: [Gpr; 12] = [
    Gpr::Rcx,
    Gpr::Rdx,
    Gpr::Rbx,
    Gpr::Rbp,
    Gpr::Rsi,
    Gpr::Rdi,
    Gpr::R8,
    Gpr::R9,
    Gpr::R12,
    Gpr::R13,
    Gpr::R14,
    Gpr::R15,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StubShape {
    /// `call stub`
    Call,
    /// `push reg; call stub`
    PushCall,
    /// `jmp stub`
    Jmp,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ResolvedImportSite {
    pub site: Rva,
    pub stub: Rva,
    pub shape: StubShape,
    /// The stub dispatches as `jmp [import]` rather than `call [import]`.
    pub tail_call: bool,
    /// Bytes added to the return address (the filler after the call).
    pub return_bump: u8,
    pub iat_slot: Rva,
    pub module: String,
    pub symbol: Option<String>,
    pub ordinal: Option<u16>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct UnresolvedImportSite {
    pub site: Rva,
    pub stub: Rva,
    pub reason: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ImportStubReport {
    pub sites_examined: usize,
    pub stubs_examined: usize,
    pub resolved: Vec<ResolvedImportSite>,
    pub unresolved: Vec<UnresolvedImportSite>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StubSummary {
    /// Stack bytes consumed from stub entry `rsp` when the import is entered.
    d: i64,
    /// The slot at `d` holds the (bumped) entry return address.
    call_like: bool,
    bump: u8,
    slot: Rva,
}

/// Resolves every direct cross-section `call`/`jmp` that is a proven import stub.
#[must_use]
pub fn resolve_import_stubs(pe: &vmpx_pe::PeImage<'_>) -> ImportStubReport {
    let mut report = ImportStubReport::default();
    if pe.optional.kind != PeKind::Pe32Plus {
        return report;
    }
    let pe_report = pe.report();
    let slots: BTreeMap<Rva, (&str, &vmpx_pe::ImportSymbol)> = pe_report
        .imports
        .iter()
        .flat_map(|module| {
            module
                .entries
                .iter()
                .map(|entry| (entry.iat_rva, (module.name.as_str(), entry)))
        })
        .collect();
    let sites = collect_sites(pe, &pe_report.exception_functions);
    report.sites_examined = sites.len();
    let slot_set: BTreeSet<Rva> = slots.keys().copied().collect();
    let mut cache: BTreeMap<Rva, Result<StubSummary, String>> = BTreeMap::new();
    for (site, stub, is_call) in sites {
        let summary = cache
            .entry(stub)
            .or_insert_with(|| summarize_stub(pe, stub, &slot_set))
            .clone();
        let outcome = summary.and_then(|summary| bind_site(pe, site, is_call, summary));
        match outcome {
            Ok((shape, summary)) => {
                let (module, symbol) = slots[&summary.slot];
                report.resolved.push(ResolvedImportSite {
                    site,
                    stub,
                    shape,
                    tail_call: !summary.call_like,
                    return_bump: summary.bump,
                    iat_slot: summary.slot,
                    module: module.to_owned(),
                    symbol: symbol.name.clone(),
                    ordinal: symbol.ordinal,
                });
            }
            Err(reason) => report
                .unresolved
                .push(UnresolvedImportSite { site, stub, reason }),
        }
    }
    report.stubs_examined = cache.len();
    report
}

/// Direct near `call`/`jmp` sites (RVA, target, `is_call`) that leave their section.
fn collect_sites(
    pe: &vmpx_pe::PeImage<'_>,
    functions: &[vmpx_pe::RuntimeFunction],
) -> Vec<(Rva, Rva, bool)> {
    let begins: BTreeSet<u32> = functions.iter().map(|f| f.begin.0).collect();
    let mut sites = Vec::new();
    for section in &pe.sections {
        if section.characteristics & 0x2000_0000 == 0
            || is_vmp_section_name(&section.name)
            || section.pointer_to_raw_data.0 == 0
        {
            continue;
        }
        let Some(window) = crate::section_file_window(pe, section.virtual_address).ok() else {
            continue;
        };
        let start = section.virtual_address.0;
        let end = start.saturating_add(u32::try_from(window.len()).unwrap_or(u32::MAX));
        let mut cuts: Vec<u32> = begins.range(start..end).copied().collect();
        if cuts.first() != Some(&start) {
            cuts.insert(0, start);
        }
        cuts.push(end);
        for pair in cuts.windows(2) {
            let (lo, hi) = (pair[0], pair[1]);
            let bytes = &window[(lo - start) as usize..(hi - start) as usize];
            let Some(ip) = pe
                .optional
                .image_base
                .checked_add_rva(Rva(lo))
                .map(|va| va.0)
            else {
                continue;
            };
            sweep(pe, bytes, ip, section.index, &mut sites);
        }
    }
    sites
}

fn sweep(
    pe: &vmpx_pe::PeImage<'_>,
    bytes: &[u8],
    ip: u64,
    section_index: usize,
    sites: &mut Vec<(Rva, Rva, bool)>,
) {
    let mut decoder = Decoder::with_ip(64, bytes, ip, DecoderOptions::NONE);
    while decoder.can_decode() {
        let position = decoder.position();
        let instruction = decoder.decode();
        if instruction.is_invalid() {
            if decoder.set_position(position + 1).is_err() {
                break;
            }
            decoder.set_ip(ip + position as u64 + 1);
            continue;
        }
        let is_call = instruction.mnemonic() == Mnemonic::Call;
        if !(is_call || instruction.mnemonic() == Mnemonic::Jmp)
            || instruction.op0_kind() != OpKind::NearBranch64
        {
            continue;
        }
        let (Some(site), Some(target)) = (
            pe.optional
                .image_base
                .checked_rva_of(vmpx_pe::Va(instruction.ip())),
            pe.optional
                .image_base
                .checked_rva_of(vmpx_pe::Va(instruction.near_branch_target())),
        ) else {
            continue;
        };
        let Some(target_section) = pe
            .sections
            .iter()
            .find(|section| section.contains_mapped_rva(target))
        else {
            continue;
        };
        if target_section.index != section_index
            && target_section.characteristics & 0x2000_0000 != 0
        {
            sites.push((site, target, is_call));
        }
    }
}

fn bind_site(
    pe: &vmpx_pe::PeImage<'_>,
    site: Rva,
    is_call: bool,
    summary: StubSummary,
) -> Result<(StubShape, StubSummary), String> {
    if !is_call {
        return if summary.call_like && summary.d == 0 && summary.bump == 0 {
            Ok((StubShape::Jmp, summary))
        } else {
            Err("jmp site: stub does not enter the import with the caller's return address".into())
        };
    }
    let pushed = if summary.call_like {
        summary.d
    } else {
        summary.d - 8
    };
    match pushed {
        0 => Ok((StubShape::Call, summary)),
        8 if preceded_by_push(pe, site) => Ok((StubShape::PushCall, summary)),
        8 => Err("stub consumes a pushed slot but no `push reg` precedes the call".into()),
        other => Err(format!("stack discipline: {other} extra bytes consumed")),
    }
}

fn preceded_by_push(pe: &vmpx_pe::PeImage<'_>, site: Rva) -> bool {
    let byte = |back: u32| {
        let rva = Rva(site.0.checked_sub(back)?);
        let section = pe
            .sections
            .iter()
            .find(|section| section.contains_mapped_rva(rva))?;
        let offset = u64::from(rva.0 - section.virtual_address.0);
        if offset >= u64::from(section.size_of_raw_data) {
            return None;
        }
        pe.data()
            .get(usize::try_from(section.pointer_to_raw_data.0 + offset).ok()?)
            .copied()
    };
    byte(1).is_some_and(|b| (0x50..=0x57).contains(&b))
}

/// Follows `entry` through unconditional direct `jmp`s to a plain `ret`.
fn straight_line(pe: &vmpx_pe::PeImage<'_>, entry: Rva) -> Result<Vec<Instruction>, String> {
    let mut instructions = Vec::new();
    let mut seen = BTreeSet::new();
    let mut rva = entry;
    loop {
        if !seen.insert(rva) || seen.len() > MAX_STUB_INSTRUCTIONS {
            return Err("stub loops or exceeds the instruction budget".into());
        }
        let decoded = decode_one(pe, rva).map_err(|e| e.to_string())?;
        let instruction = decoded.instruction;
        match instruction.flow_control() {
            FlowControl::Next => {
                instructions.push(instruction);
                rva = Rva(u32::try_from(u64::from(rva.0) + instruction.len() as u64)
                    .map_err(|_| "address overflow")?);
            }
            FlowControl::UnconditionalBranch if instruction.op0_kind() == OpKind::NearBranch64 => {
                rva = pe
                    .optional
                    .image_base
                    .checked_rva_of(vmpx_pe::Va(instruction.near_branch_target()))
                    .ok_or("jmp leaves the image")?;
            }
            FlowControl::Return if instruction.op_count() == 0 => return Ok(instructions),
            other => return Err(format!("unmodeled control flow {other:?} in stub")),
        }
    }
}

/// `rsp_in + k`.
fn rsp_offset(expr: &NativeExpr) -> Option<i64> {
    match expr.node() {
        NativeNode::Input { name, .. } if *name == Gpr::Rsp.input_name() => Some(0),
        NativeNode::Binary {
            op: BinOp::Add,
            left,
            right,
        } => match (left.node(), right.as_const()) {
            (NativeNode::Input { name, .. }, Some(c)) if *name == Gpr::Rsp.input_name() => {
                Some(c.cast_signed())
            }
            _ => None,
        },
        _ => None,
    }
}

/// `base + c` with `c` a constant, or `base` alone.
fn split_add(expr: &NativeExpr) -> (NativeExpr, u64) {
    if let NativeNode::Binary {
        op: BinOp::Add,
        left,
        right,
    } = expr.node()
    {
        if let Some(c) = right.as_const() {
            let (base, inner) = split_add(left);
            return (base, inner.wrapping_add(c));
        }
        if let Some(c) = left.as_const() {
            let (base, inner) = split_add(right);
            return (base, inner.wrapping_add(c));
        }
    }
    (expr.clone(), 0)
}

fn entry_slot(offset: i64) -> NativeExpr {
    use crate::Algebra;
    let alg = NativeAlgebra;
    let base = alg.input(Width::W64, &Gpr::Rsp.input_name());
    let address = if offset == 0 {
        base
    } else {
        alg.binary(
            BinOp::Add,
            base,
            alg.constant(Width::W64, offset.cast_unsigned()),
        )
        .expect("equal widths")
    };
    alg.load(Width::W64, address)
}

fn summarize_stub(
    pe: &vmpx_pe::PeImage<'_>,
    entry: Rva,
    slots: &BTreeSet<Rva>,
) -> Result<StubSummary, String> {
    let instructions = straight_line(pe, entry)?;
    let semantics =
        lift_block_with(&NativeAlgebra, &instructions, STUB_LIFT).map_err(|e| e.to_string())?;
    analyze(pe, &semantics, slots)
}

fn analyze(
    pe: &vmpx_pe::PeImage<'_>,
    semantics: &BlockSemantics<NativeExpr>,
    slots: &BTreeSet<Rva>,
) -> Result<StubSummary, String> {
    let rsp = semantics.regs.get(&Gpr::Rsp).ok_or("rsp missing")?;
    let ret_offset = rsp_offset(rsp).ok_or("rsp is not entry rsp plus a constant")?;
    let d = ret_offset + 8;
    if !(0..=16).contains(&d) || d % 8 != 0 {
        return Err(format!("stub exits with rsp at entry+{d}"));
    }
    for gpr in PRESERVED {
        let preserved = semantics.regs.get(&gpr).is_some_and(|value| {
            matches!(value.node(), NativeNode::Input { name, .. } if *name == gpr.input_name())
        });
        if !preserved {
            return Err(format!("stub does not preserve {}", gpr.name()));
        }
    }
    let mut stores: BTreeMap<i64, &NativeExpr> = BTreeMap::new();
    for write in &semantics.writes {
        let offset = rsp_offset(&write.address).ok_or("stub writes outside its stack frame")?;
        if write.width != Width::W64 {
            return Err("stub makes a partial-width stack write".into());
        }
        if stores
            .keys()
            .any(|other| *other != offset && (other - offset).abs() < 8)
        {
            return Err("overlapping stack writes".into());
        }
        stores.insert(offset, &write.value);
    }
    let target = stores
        .get(&ret_offset)
        .map_or_else(|| entry_slot(ret_offset), |value| (*value).clone());
    let NativeNode::Load { width, address } = target.node() else {
        return Err("dispatch target is not a load of an import slot".into());
    };
    let va = address
        .as_const()
        .filter(|_| *width == Width::W64)
        .ok_or("dispatch target is not a load of a constant 64-bit address")?;
    let slot = pe
        .optional
        .image_base
        .checked_rva_of(vmpx_pe::Va(va))
        .filter(|rva| slots.contains(rva))
        .ok_or_else(|| format!("dispatch load {va:#x} is not an import slot"))?;
    for (&offset, value) in &stores {
        if offset > d && **value != entry_slot(offset) {
            return Err("stub modifies the caller's stack".into());
        }
    }
    let at_import = stores
        .get(&d)
        .map_or_else(|| entry_slot(d), |value| (*value).clone());
    let (base, bump) = split_add(&at_import);
    let call_like = base == entry_slot(0) && d <= 8;
    let mut return_bump = 0;
    if call_like {
        return_bump = u8::try_from(bump)
            .ok()
            .filter(|bump| u64::from(*bump) <= MAX_BUMP)
            .ok_or_else(|| format!("return address bumped by {}", bump.cast_signed()))?;
    } else if at_import != entry_slot(d) || d < 8 {
        return Err("return address slot is neither the caller's nor bumped".into());
    }
    Ok(StubSummary {
        d,
        call_like,
        bump: return_bump,
        slot,
    })
}

#[cfg(test)]
mod tests {
    use vmpx_pe::PeImage;

    use super::*;

    const TEXT: u32 = 0x1000;
    const STUBS: u32 = 0x2000;
    const IAT: u32 = 0x3060;

    fn put(data: &mut [u8], offset: usize, bytes: &[u8]) {
        data[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    fn section(data: &mut [u8], index: usize, name: &[u8], rva: u32, raw: u32, flags: u32) {
        let at = 0x188 + index * 40;
        put(data, at, name);
        put(data, at + 8, &0x200_u32.to_le_bytes());
        put(data, at + 12, &rva.to_le_bytes());
        put(data, at + 16, &0x200_u32.to_le_bytes());
        put(data, at + 20, &raw.to_le_bytes());
        put(data, at + 36, &flags.to_le_bytes());
    }

    /// `mov rax, [rip+disp]` at `at` loading `target` (both RVAs).
    fn load_rax(at: u32, target: u32) -> Vec<u8> {
        let disp = target.wrapping_sub(at + 7);
        [&[0x48, 0x8B, 0x05][..], &disp.to_le_bytes()].concat()
    }

    fn call(at: u32, target: u32) -> Vec<u8> {
        let disp = target.wrapping_sub(at + 5);
        [&[0xE8][..], &disp.to_le_bytes()].concat()
    }

    fn stubs() -> Vec<(u32, Vec<u8>)> {
        let store_target: &[u8] = &[0x48, 0x89, 0x04, 0x24];
        vec![
            // push rax; mov rax,[iat]; add [rsp+8],1; xchg [rsp],rax; ret
            (
                STUBS,
                [
                    &[0x50][..],
                    &load_rax(STUBS + 1, IAT),
                    &[
                        0x48, 0x83, 0x44, 0x24, 0x08, 0x01, 0x48, 0x87, 0x04, 0x24, 0xC3,
                    ],
                ]
                .concat(),
            ),
            // mov rax,[rsp]; add rax,1; mov [rsp+8],rax; mov rax,[iat]; mov [rsp],rax; ret
            (
                STUBS + 0x40,
                [
                    &[0x48, 0x8B, 0x04, 0x24, 0x48, 0x83, 0xC0, 0x01][..],
                    &[0x48, 0x89, 0x44, 0x24, 0x08],
                    &load_rax(STUBS + 0x40 + 13, IAT),
                    store_target,
                    &[0xC3],
                ]
                .concat(),
            ),
            // jmp over junk; mov rax,[iat]; mov [rsp],rax; ret
            (
                STUBS + 0x80,
                [
                    &[0xEB, 0x02, 0xCC, 0xCC][..],
                    &load_rax(STUBS + 0x80 + 4, IAT),
                    store_target,
                    &[0xC3],
                ]
                .concat(),
            ),
            // loads a slot that is not an import
            (
                STUBS + 0xC0,
                [&load_rax(STUBS + 0xC0, 0x3000)[..], store_target, &[0xC3]].concat(),
            ),
            // clobbers rcx
            (
                STUBS + 0x100,
                [
                    &[0x48, 0x31, 0xC9][..],
                    &load_rax(STUBS + 0x100 + 3, IAT),
                    store_target,
                    &[0xC3],
                ]
                .concat(),
            ),
            // rewrites the caller's stack above the return slot
            (
                STUBS + 0x140,
                [
                    &[0x48, 0x89, 0x44, 0x24, 0x10][..],
                    &load_rax(STUBS + 0x140 + 5, IAT),
                    store_target,
                    &[0xC3],
                ]
                .concat(),
            ),
        ]
    }

    fn synthetic() -> Vec<u8> {
        let mut data = vec![0_u8; 0x800];
        put(&mut data, 0, &0x5A4D_u16.to_le_bytes());
        put(&mut data, 0x3C, &0x80_u32.to_le_bytes());
        put(&mut data, 0x80, &0x0000_4550_u32.to_le_bytes());
        put(&mut data, 0x84, &0x8664_u16.to_le_bytes());
        put(&mut data, 0x86, &3_u16.to_le_bytes());
        put(&mut data, 0x94, &0xF0_u16.to_le_bytes());
        put(&mut data, 0x96, &0x2022_u16.to_le_bytes());
        put(&mut data, 0x98, &0x20B_u16.to_le_bytes());
        put(&mut data, 0x98 + 16, &TEXT.to_le_bytes());
        put(&mut data, 0x98 + 24, &0x1_4000_0000_u64.to_le_bytes());
        put(&mut data, 0x98 + 32, &0x1000_u32.to_le_bytes());
        put(&mut data, 0x98 + 36, &0x200_u32.to_le_bytes());
        put(&mut data, 0x98 + 56, &0x4000_u32.to_le_bytes());
        put(&mut data, 0x98 + 60, &0x200_u32.to_le_bytes());
        put(&mut data, 0x98 + 108, &16_u32.to_le_bytes());
        put(&mut data, 0x98 + 120, &0x3000_u32.to_le_bytes());
        put(&mut data, 0x98 + 124, &40_u32.to_le_bytes());
        section(&mut data, 0, b".text", TEXT, 0x200, 0x6000_0020);
        section(&mut data, 1, b".qzx", STUBS, 0x400, 0x6000_0020);
        section(&mut data, 2, b".idata", 0x3000, 0x600, 0xC000_0040);
        let at = |rva: u32| rva as usize - 0x3000 + 0x600;
        put(&mut data, at(0x3000), &0x3040_u32.to_le_bytes());
        put(&mut data, at(0x3000) + 12, &0x3080_u32.to_le_bytes());
        put(&mut data, at(0x3000) + 16, &0x3060_u32.to_le_bytes());
        put(&mut data, at(0x3040), &0x30A0_u64.to_le_bytes());
        put(&mut data, at(0x3060), &0x30A0_u64.to_le_bytes());
        put(&mut data, at(0x3080), b"kernel32.dll\0");
        put(&mut data, at(0x30A0) + 2, b"ExitProcess\0");
        let mut code = Vec::new();
        code.extend(call(TEXT, STUBS));
        code.push(0x50);
        code.extend(call(TEXT + 6, STUBS + 0x40));
        for (n, stub) in [0x80, 0xC0, 0x100, 0x140].into_iter().enumerate() {
            code.extend(call(TEXT + 11 + u32::try_from(n).expect("small") * 5, STUBS + stub));
        }
        code.push(0xC3);
        put(&mut data, 0x200, &code);
        for (rva, bytes) in stubs() {
            put(&mut data, 0x400 + (rva - STUBS) as usize, &bytes);
        }
        data
    }

    #[test]
    fn stub_shapes_resolve_and_unproven_stubs_fail_closed() {
        let data = synthetic();
        let image = PeImage::parse(&data).expect("synthetic pe parses");
        let report = resolve_import_stubs(&image);
        assert_eq!(report.sites_examined, 6);
        let by_site: BTreeMap<_, _> = report.resolved.iter().map(|r| (r.site.0, r)).collect();
        let a = by_site[&TEXT];
        assert_eq!(
            (a.shape, a.tail_call, a.return_bump, a.symbol.as_deref()),
            (StubShape::Call, false, 1, Some("ExitProcess"))
        );
        let b = by_site[&(TEXT + 6)];
        assert_eq!(
            (b.shape, b.tail_call, b.return_bump),
            (StubShape::PushCall, false, 1)
        );
        let c = by_site[&(TEXT + 11)];
        assert_eq!(
            (c.shape, c.tail_call, c.module.as_str()),
            (StubShape::Call, true, "kernel32.dll")
        );
        let reasons: BTreeMap<_, _> = report
            .unresolved
            .iter()
            .map(|u| (u.site.0, u.reason.as_str()))
            .collect();
        assert!(reasons[&(TEXT + 16)].contains("not an import slot"));
        assert!(reasons[&(TEXT + 21)].contains("rcx"));
        assert!(reasons[&(TEXT + 26)].contains("caller's stack"));
        assert_eq!(report.resolved.len() + report.unresolved.len(), 6);
    }

    #[test]
    fn push_call_stub_without_preceding_push_is_unresolved() {
        let mut data = synthetic();
        data[0x205] = 0x90;
        let image = PeImage::parse(&data).expect("synthetic pe parses");
        let report = resolve_import_stubs(&image);
        assert!(
            report
                .unresolved
                .iter()
                .any(|u| u.site.0 == TEXT + 6 && u.reason.contains("no `push reg`"))
        );
    }
}
