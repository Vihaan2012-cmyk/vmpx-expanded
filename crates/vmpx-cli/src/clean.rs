use std::collections::{BTreeMap, BTreeSet, VecDeque};

use anyhow::{Context, Result};
use serde::Serialize;
use vmpx_detect::{DetectionVerdict, ImageLayout, ProtectionReport, detect_vmprotect};
use vmpx_disasm::{CfgLimits, DirectJumpTraceLimits, collect_dangling_refs, recover_cfg};
use vmpx_pe::{PeImage, PeKind, Rva};
use vmpx_rebuild::{relayout_conventional, validate_for_loader};
use vmpx_unpack::{UnpackLimits, UnpackReport, unpack_to_mapped_image};
use vmpx_vm::{
    BitWidth, EntryPathRecoveryLimits, EntryProfileRecoveryLimits, Expr, ModuloImageTargetLimits,
    NativeContinuationLimits, NativeContinuationReport, RecoveryStatus, VmDiscoveryLimits,
    VmEntrySource, VmMemoryExceptionCoverageReport, VmProfileChainStop, VmProfileExplorerLimits,
    analyze_native_continuation, analyze_vm_entry_at,
    analyze_vm_memory_exception_coverage_with_state, continue_through_native_vm_entry, decide_oep,
    discover_vm_entries, recover_entry_paths, recover_modulo_image_targets,
};
use vmpx_vmlift::audit_entry_effect_liveness;

use crate::{
    clean_wiring::{self, CleanAntiDebugReport, CleanCoverageReport, InventoryLoss},
    coverage_gate::{CoverageInput, decide_coverage},
    sha256,
    vm_inventory::{VmInventoryEntry, merge_vm_inventory},
};

pub(crate) const IMAGE_FILE_DLL: u16 = 0x2000;
const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;

#[derive(Clone, Debug, Serialize)]
pub struct CleanInputReport {
    pub path: String,
    pub sha256: String,
    pub file_size: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CleanPeReport {
    pub kind: PeKind,
    pub machine: u16,
    pub image_base: vmpx_pe::ImageBase,
    pub entry_point: Rva,
    pub size_of_image: u32,
    pub sections: usize,
    pub import_modules: usize,
    pub import_symbols: usize,
    pub tls_callbacks: usize,
    pub exception_functions: usize,
    pub structural_diagnostics: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct CleanUnpackingReport {
    pub performed: bool,
    pub mapped_image_size: u64,
    pub report: Option<UnpackReport>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CleanImportReport {
    pub parsed_modules: usize,
    pub parsed_symbols: usize,
    pub reconstruction_complete: bool,
    pub proven_modules: usize,
    pub proven_symbols: usize,
    pub unresolved: Vec<String>,
    /// EXPERIMENTAL dynamic-assist (`--dynamic`): Observed-tier import
    /// reconstruction from a contained detonation. Absent on the default sound
    /// path; never influences `blockers`/`ready_to_write`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed: Option<CleanObservedImportReport>,
}

/// Observed-tier (never sound) import reconstruction surfaced on the
/// experimental `--dynamic` path. Every field reflects one concrete run.
#[derive(Clone, Debug, Serialize)]
pub struct CleanObservedImportReport {
    pub provenance: &'static str,
    pub detonation_steps: usize,
    pub detonation_stop: String,
    pub observed_oep: Option<u64>,
    pub observed_imports: usize,
    pub reconstructed_modules: usize,
    pub reconstructed_symbols: usize,
    pub reconstruction_complete: bool,
    pub import_directory_bytes: Option<usize>,
    pub note: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct CleanOepReport {
    pub protected_header_entry: Rva,
    pub recovered_entry: Option<Rva>,
    pub complete: bool,
}

/// Per-export dependency on protector code. An export is only eligible for
/// native replacement once every branch and call reachable from it stays
/// outside the protector sections.
#[derive(Clone, Debug, Serialize)]
pub struct CleanExportReport {
    pub name: String,
    pub rva: Rva,
    pub reachable_blocks: usize,
    /// Distinct virtualized routines this export can reach. The VM's own dispatch
    /// lands on handler interiors, which the CFG cannot tell from call targets, so
    /// only addresses that present an entry stub are counted.
    pub vm_references: usize,
    /// RVAs of those routines, so the remaining work is addressable rather than a
    /// bare count.
    pub vm_entries: Vec<Rva>,
    pub vm_free: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct CleanTlsEntryReport {
    pub entry: Rva,
    pub confidence: u8,
    pub recovered: bool,
    pub domain_complete: bool,
    pub all_paths_reach_vm_exit: bool,
    pub segments: usize,
    pub decoded_operations: usize,
    pub stop_reason: Option<VmProfileChainStop>,
    pub native_exit_recovered: bool,
    pub issues: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CleanHandlerMetrics {
    pub unique: usize,
    pub classified: usize,
    pub unknown: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct CleanVmPathReport {
    pub assignments: BTreeMap<String, u64>,
    pub segments: usize,
    pub decoded_operations: usize,
    pub stop_reason: VmProfileChainStop,
    pub native_exit_recovered: bool,
    pub continuation_targets: Option<vmpx_vm::ModuloImageTargetReport>,
    pub exception_coverage: Vec<VmMemoryExceptionCoverageReport>,
    pub issues: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CleanNestedContinuationReport {
    pub source_assignments: BTreeMap<String, u64>,
    pub target: Rva,
    pub vm_entry_confidence: u8,
    pub decoded_operations: usize,
    pub stop_reason: Option<VmProfileChainStop>,
    pub exception_coverage: Vec<VmMemoryExceptionCoverageReport>,
    pub native_cfg_blocks: Option<usize>,
    pub native_analysis: Option<NativeContinuationReport>,
    pub issues: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CleanStageReport {
    pub complete: bool,
    pub metrics: BTreeMap<String, u64>,
    pub diagnostics: Vec<String>,
}

/// Outcome of converting the RVA-layout analysis image into a conventional
/// file layout. This is a layout transform, not devirtualization: every VM
/// handler, protector section, and unresolved path is still present.
#[derive(Clone, Debug, Serialize)]
pub struct CleanLayoutReport {
    pub attempted: bool,
    pub file_size: Option<u64>,
    pub size_of_headers: Option<u32>,
    pub sections: Option<usize>,
    pub zero_fill_sections: Option<usize>,
    /// Result of walking every directory the Windows loader consumes.
    pub loader_valid: Option<bool>,
    pub loader_summary: Option<String>,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct CleanValidationReport {
    pub input_parsed: bool,
    pub mapped_image_parsed: bool,
    pub output_written: bool,
    pub output_reparsed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct CleanReport {
    pub schema_version: u32,
    pub input: CleanInputReport,
    pub pe: CleanPeReport,
    pub protection: ProtectionReport,
    pub unpacking: CleanUnpackingReport,
    pub oep: CleanOepReport,
    pub imports: CleanImportReport,
    pub vm_instances: Vec<vmpx_vm::VmInstance>,
    pub vm_entries: Vec<vmpx_vm::VmEntryCandidate>,
    pub vm_inventory: Vec<VmInventoryEntry>,
    /// Inventory entries whose single path reaches an input-closed native exit.
    /// Recovery only; nothing here is reintegrated or affects blockers.
    pub input_closed_functions: CleanStageReport,
    pub handlers: CleanHandlerMetrics,
    pub unknown_handlers: Vec<Rva>,
    pub virtualized_routines: Vec<CleanVmPathReport>,
    pub tls_entries: Vec<CleanTlsEntryReport>,
    pub exports: Vec<CleanExportReport>,
    pub native_continuations: Vec<CleanNestedContinuationReport>,
    pub vtil: CleanStageReport,
    pub recompilation: CleanStageReport,
    pub reintegration: CleanStageReport,
    /// Coverage-gate verdict; `None` when no gate was evaluated.
    pub coverage: Option<CleanCoverageReport>,
    pub anti_debug: CleanAntiDebugReport,
    pub reconstruction: CleanStageReport,
    pub layout: CleanLayoutReport,
    pub validation: CleanValidationReport,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub blockers: Vec<String>,
    pub ready_to_write: bool,
    pub experimental: bool,
}

fn pe_summary(pe: &PeImage<'_>) -> CleanPeReport {
    let report = pe.report();
    CleanPeReport {
        kind: report.optional.kind,
        machine: report.coff.machine,
        image_base: report.optional.image_base,
        entry_point: report.optional.address_of_entry_point,
        size_of_image: report.optional.size_of_image,
        sections: report.sections.len(),
        import_modules: report.imports.len(),
        import_symbols: report
            .imports
            .iter()
            .map(|module| module.entries.len())
            .sum(),
        tls_callbacks: report.tls.as_ref().map_or(0, |tls| tls.callbacks.len()),
        exception_functions: report.exception_functions.len(),
        structural_diagnostics: report.diagnostics.len(),
    }
}

/// Walks every branch and direct call reachable from each export and records
/// whether the reachable set stays outside the protector sections.
fn export_dependency_report(pe: &PeImage<'_>) -> Vec<CleanExportReport> {
    let report = pe.report();
    let Some(exports) = &report.exports else {
        return Vec::new();
    };
    // Protector regions are identified structurally, not by name: VMProtect
    // randomizes section names, so a `.vmp` prefix misses every randomized build.
    // A non-standard-named section owning the RVA is the protector signal.
    let is_protector = |rva: Rva| {
        pe.sections.iter().any(|section| {
            !vmpx_detect::is_standard_section_name(&section.name)
                && section.contains_mapped_rva(rva)
        })
    };
    let mut results = Vec::new();
    for symbol in &exports.symbols {
        if symbol.forwarder.is_some() {
            continue;
        }
        let mut seen = BTreeSet::new();
        let mut queue = VecDeque::from([symbol.rva]);
        // Distinct virtualized routines this export can reach. Counting edges
        // instead would report the same routine many times and overstate the work.
        let mut vm_entries: BTreeSet<u32> = BTreeSet::new();
        while let Some(start) = queue.pop_front() {
            if !seen.insert(start) {
                continue;
            }
            if is_protector(start) {
                continue;
            }
            let Ok(graph) = recover_cfg(
                pe,
                start,
                CfgLimits {
                    max_blocks: 2_048,
                    max_instructions: 60_000,
                    max_instructions_per_block: 8_192,
                },
            ) else {
                continue;
            };
            for block in &graph.blocks {
                for successor in &block.successors {
                    if is_protector(*successor) {
                        // Not every protector address is a routine: the VM's own
                        // dispatch lands on handler interiors, which the CFG sees
                        // as ordinary successors. Only an address that presents an
                        // entry stub is a virtualized routine worth replacing.
                        if vmpx_vm::analyze_vm_entry_at(
                            pe,
                            *successor,
                            vmpx_vm::VmDiscoveryLimits::default(),
                        )
                        .is_some_and(|candidate| candidate.confidence >= 60)
                        {
                            vm_entries.insert(successor.0);
                        }
                    } else if !seen.contains(successor) {
                        queue.push_back(*successor);
                    }
                }
                for call in &block.direct_calls {
                    if is_protector(*call) {
                        if vmpx_vm::analyze_vm_entry_at(
                            pe,
                            *call,
                            vmpx_vm::VmDiscoveryLimits::default(),
                        )
                        .is_some_and(|candidate| candidate.confidence >= 60)
                        {
                            vm_entries.insert(call.0);
                        }
                    } else if !seen.contains(call) {
                        queue.push_back(*call);
                    }
                }
            }
        }
        let vm_references = vm_entries.len();
        results.push(CleanExportReport {
            name: symbol
                .name
                .clone()
                .unwrap_or_else(|| format!("ordinal{}", symbol.ordinal)),
            rva: symbol.rva,
            reachable_blocks: seen.len(),
            vm_references,
            vm_entries: vm_entries.iter().map(|rva| Rva(*rva)).collect(),
            vm_free: vm_references == 0,
        });
    }
    results
}

/// Unifies executable-stub discovery with the VM entries reachable from exports
/// into one deduplicated list. Inventory only: nothing here is recovered.
fn build_vm_inventory(
    pe: &PeImage<'_>,
    exports: &[CleanExportReport],
    warnings: &mut Vec<String>,
) -> (Vec<VmInventoryEntry>, InventoryLoss) {
    let mut loss = InventoryLoss::default();
    let mut items: Vec<(Rva, u8, &'static str, bool)> = Vec::new();
    match discover_vm_entries(
        pe,
        VmDiscoveryLimits {
            scan_executable_stubs: true,
            ..VmDiscoveryLimits::default()
        },
    ) {
        Ok(report) => items.extend(report.entries.iter().map(|entry| {
            let label = match entry.source {
                vmpx_vm::VmEntrySource::PeEntryPoint => "header",
                vmpx_vm::VmEntrySource::TlsCallback => "tls",
                vmpx_vm::VmEntrySource::ExecutableScan => "scan",
            };
            (entry.rva, entry.confidence, label, false)
        })),
        Err(error) => {
            warnings.push(format!("VM entry inventory scan failed: {error}"));
            loss.scan_failure = Some(format!("VM entry inventory scan failed: {error}"));
        }
    }
    let export_rvas: BTreeSet<Rva> = exports
        .iter()
        .flat_map(|export| export.vm_entries.iter().copied())
        .collect();
    for rva in export_rvas {
        match analyze_vm_entry_at(pe, rva, VmDiscoveryLimits::default()) {
            Some(candidate) => items.push((rva, candidate.confidence, "export_xref", true)),
            None => loss.unanalyzed_exports.push(rva),
        }
    }
    (merge_vm_inventory(items), loss)
}

fn exception_coverage_for_chain(
    pe: &PeImage<'_>,
    chain: &vmpx_vm::VmProfileChainReport,
) -> Vec<VmMemoryExceptionCoverageReport> {
    let mut reports = Vec::new();
    for segment in &chain.segments {
        let Some(error) = &segment.symbolic_error else {
            continue;
        };
        let Some(instruction) = segment
            .decode
            .decode
            .instructions
            .get(error.instruction_index)
        else {
            continue;
        };
        let Some(handler) = segment
            .decode
            .handler_recoveries
            .get(&instruction.state_before.handler)
        else {
            continue;
        };
        if let Ok(report) =
            analyze_vm_memory_exception_coverage_with_state(pe, handler, segment.symbolic.as_ref())
        {
            reports.push(report);
        }
    }
    reports
}

fn handler_metrics(family: &vmpx_vm::RecoveredEntryPathFamily) -> (CleanHandlerMetrics, Vec<Rva>) {
    let mut status = BTreeMap::<Rva, RecoveryStatus>::new();
    for recovery in family
        .paths
        .iter()
        .flat_map(|path| &path.chain.segments)
        .flat_map(|segment| segment.decode.handler_recoveries.values())
        .chain(family.entry_profile.decode.handler_recoveries.values())
    {
        status
            .entry(recovery.entry)
            .and_modify(|current| {
                if *current != RecoveryStatus::Recovered {
                    *current = recovery.status;
                }
            })
            .or_insert(recovery.status);
    }
    let unknown_handlers = status
        .iter()
        .filter_map(|(rva, status)| (*status != RecoveryStatus::Recovered).then_some(*rva))
        .collect::<Vec<_>>();
    (
        CleanHandlerMetrics {
            unique: status.len(),
            classified: status
                .values()
                .filter(|status| **status == RecoveryStatus::Recovered)
                .count(),
            unknown: unknown_handlers.len(),
        },
        unknown_handlers,
    )
}

/// Symbolic input name to the architectural register that carries it on entry.
///
/// `vsp_initial` is the native RSP captured by the entry stub immediately before
/// the bootstrap reserves its own frame (`mov r11,rsp` at the PE entry stub's
/// call site). It is therefore a live architectural value at the stub, not a
/// fresh root: a replacement that reads the wrong register would compute every
/// stack-relative address from the wrong base.
pub(crate) fn replacement_inputs(
    pe: &PeImage<'_>,
    entry: &vmpx_vm::VmEntryCandidate,
    rewrites: &mut BTreeMap<String, vmpx_vm::Expr>,
) -> Result<Vec<(String, vmpx_recompile::PhysicalRegister)>, String> {
    use vmpx_recompile::PhysicalRegister;
    use vmpx_vm::X64NativeRegister;
    let frame = vmpx_vm::analyze_x64_native_frame(pe, entry.rva, DirectJumpTraceLimits::default())
        .map_err(|error| format!("native frame analysis failed: {error}"))?;
    let cursor = frame
        .cursor_architectural_register
        .ok_or_else(|| "entry stub has no proven stack-cursor register".to_owned())?;
    let mut inputs: Vec<(String, PhysicalRegister)> = X64NativeRegister::ALL
        .iter()
        .map(|register| (register.symbolic_name().to_owned(), physical_for(register)))
        .collect();
    // The VM stack root is whatever register the stub captured native RSP into.
    //
    // That register already carries `native_<reg>_initial` as an input: the
    // stub's own entry value. The two are different values that live in the same
    // register at different times - the stub's entry value is saved on the stack,
    // and the same register is then overwritten with the captured RSP. Binding
    // both names to one register would make them equal, which is wrong, and it is
    // not fixable by picking another register because every architectural
    // register is already an input.
    let cursor_name = cursor.symbolic_name();
    if inputs
        .iter()
        .any(|(name, register)| name == cursor_name && *register == physical_for(&cursor))
    {
        // The stub saved this register before overwriting it with the captured
        // stack pointer, and the frame analysis proves which slot holds the saved
        // value. Rewriting the name to a load from that slot keeps both values
        // correct while freeing the register for `vsp_initial`.
        let slot = frame
            .slots
            .iter()
            .find(|slot| {
                matches!(&slot.source,
                    vmpx_vm::NativeFrameSource::Register { register }
                        if register.symbolic_name() == cursor_name)
            })
            .ok_or_else(|| {
                format!(
                    "the stack cursor register {cursor_name} is also a symbolic input and has \
                     no proven save slot, so `vsp_initial` cannot be bound"
                )
            })?;
        let address = vmpx_vm::Expr::binary(
            vmpx_vm::BinaryOp::Add,
            vmpx_vm::Expr::register(vmpx_vm::BitWidth::W64, "vsp_initial"),
            vmpx_vm::Expr::constant(vmpx_vm::BitWidth::W64, u64::from(slot.frame_offset)),
        )
        .map_err(|error| format!("saved-slot address failed: {error}"))?;
        rewrites.insert(
            cursor_name.to_owned(),
            vmpx_vm::Expr::load(vmpx_vm::BitWidth::W64, address),
        );
    }
    // A rewritten symbol is no longer an input: its value comes from the frame,
    // so keeping it here would make the capture block store one register into two
    // slots and silently give both the same value.
    inputs.retain(|(name, _)| !rewrites.contains_key(name));
    inputs.push(("vsp_initial".to_owned(), physical_for(&cursor)));
    inputs.sort();
    inputs.dedup();
    Ok(inputs)
}

fn physical_for(register: &vmpx_vm::X64NativeRegister) -> vmpx_recompile::PhysicalRegister {
    use vmpx_recompile::PhysicalRegister as P;
    use vmpx_vm::X64NativeRegister as R;
    match register {
        R::Rax => P::Rax,
        R::Rbx => P::Rbx,
        R::Rcx => P::Rcx,
        R::Rdx => P::Rdx,
        R::Rbp => P::Rbp,
        R::Rsi => P::Rsi,
        R::Rdi => P::Rdi,
        R::R8 => P::R8,
        R::R9 => P::R9,
        R::R10 => P::R10,
        R::R11 => P::R11,
        R::R12 => P::R12,
        R::R13 => P::R13,
        R::R14 => P::R14,
        R::R15 => P::R15,
    }
}

/// Maps each fresh-allocation intrinsic in an exit state to the imported call it
/// stands for, so the lowering can emit a real call instead of refusing.
///
/// The intrinsic's first argument is the allocator's hint/name entry. The import
/// directory is searched for the symbol whose hint/name entry has that address;
/// its IAT slot becomes the call target. Any intrinsic that cannot be resolved
/// this way stays absent and is therefore still refused.
fn import_calls_for(
    pe: &PeImage<'_>,
    exit: &vmpx_vm::NativeExitState,
) -> BTreeMap<String, vmpx_vmlift::ImportCall> {
    let mut names: BTreeMap<u64, (String, u64)> = BTreeMap::new();
    for value in exit.registers.values() {
        let vmpx_vm::ExprKind::Intrinsic { name, arguments } = &value.kind else {
            continue;
        };
        if !name.starts_with("fresh_allocation_return_epoch_") || arguments.len() < 2 {
            continue;
        }
        let vmpx_vm::ExprKind::Const { value: target } = arguments[0].kind else {
            continue;
        };
        names.insert(target, (name.clone(), target));
    }
    let report = pe.report();
    let mut calls = BTreeMap::new();
    for (name, target) in names.values() {
        // The target is a hint/name entry; find the import whose name entry has
        // that address and take its IAT slot.
        // `lookup_value` is the original INT entry: the VA of the hint/name pair
        // for a named import, which is exactly what the intrinsic records.
        let Some(symbol) = report
            .imports
            .iter()
            .flat_map(|module| module.entries.iter())
            .find(|entry| entry.lookup_value == *target)
        else {
            continue;
        };
        // The backend encodes a direct call to this address, so it must be the
        // IAT slot's virtual address, not its file offset.
        if let Some(va) = pe.optional.image_base.checked_add_rva(symbol.iat_rva) {
            calls.insert(
                name.clone(),
                vmpx_vmlift::ImportCall {
                    iat_address: va.0,
                    argument_count: 1,
                },
            );
        }
    }
    calls
}

fn stage(diagnostic: impl Into<String>) -> CleanStageReport {
    CleanStageReport {
        complete: false,
        metrics: BTreeMap::new(),
        diagnostics: vec![diagnostic.into()],
    }
}

fn completed_stage(diagnostic: impl Into<String>) -> CleanStageReport {
    CleanStageReport {
        complete: true,
        metrics: BTreeMap::new(),
        diagnostics: vec![diagnostic.into()],
    }
}

#[allow(clippy::too_many_lines)]
#[allow(dead_code)]
pub fn analyze_clean(input_path: &str, input: &[u8]) -> Result<CleanReport> {
    analyze_clean_with_image(input_path, input, false, false).map(|(report, _, _)| report)
}

/// Runs the same analysis and additionally returns the conventional-layout
/// image when the mapped analysis image can be converted. The image is a
/// layout artifact: it is loadable and structurally valid, but still contains
/// every protector section and virtualized routine. Also returns the
/// reintegrated image when emitted replacement bodies were redirected into it.
///
/// # Errors
///
/// Propagates input, unpacking, VM-analysis, and exception-metadata failures.
#[allow(clippy::too_many_lines)]
pub fn analyze_clean_with_image(
    input_path: &str,
    input: &[u8],
    decoded_lift: bool,
    dynamic: bool,
) -> Result<(CleanReport, Option<Vec<u8>>, Option<Vec<u8>>)> {
    let input_pe = PeImage::parse(input).context("clean input is not a supported PE")?;
    let protection = detect_vmprotect(&input_pe);
    let (mapped, unpack_report) = match protection.image_layout {
        ImageLayout::PackedVirtualSections => {
            let result = unpack_to_mapped_image(input, UnpackLimits::default())?;
            (result.image, Some(result.report))
        }
        _ => (input.to_vec(), None),
    };
    let pe = PeImage::parse(&mapped).context("mapped analysis image failed reparse")?;
    let pe_report = pe_summary(&pe);
    let report = pe.report();
    let import_symbols = report
        .imports
        .iter()
        .map(|module| module.entries.len())
        .sum::<usize>();
    let discovery = discover_vm_entries(
        &pe,
        VmDiscoveryLimits {
            scan_executable_stubs: false,
            ..VmDiscoveryLimits::default()
        },
    )?;
    let mut blockers = Vec::new();
    let mut warnings = protection.warnings.clone();
    let mut errors = Vec::new();
    let mut paths = Vec::new();
    // EXPERIMENTAL dynamic-assist (`--dynamic`, requires `--experimental`): run
    // the sample in the contained software emulator and collect its observed,
    // dynamically-resolved imports. Observed-tier only — these NEVER touch the
    // sound `blockers` gate below; they are surfaced separately in the report.
    let mut dynamic_stop: Option<String> = None;
    let mut dynamic_steps = 0usize;
    let mut dynamic_oep: Option<u64> = None;
    // Observed runtime globals captured on the detonation path, converted into
    // the plain vmpx-vm `ObservedMemory` seed (never the `vmpx-emulate` type, to
    // keep vmpx-vm free of a dependency on vmpx-emulate). This seeds the
    // experimental decoded-lift recovery past provably-null image reads; it is
    // `None` on the default sound path, so that path is unchanged.
    let mut dynamic_globals: Vec<(u64, Vec<u8>)> = Vec::new();
    let observed_imports: Vec<vmpx_emulate::ObservedImport> = if dynamic {
        match vmpx_emulate::detonate(&input_pe, &vmpx_emulate::DetonateOptions::default()) {
            Ok(result) => {
                warnings.push(format!(
                    "[dynamic][experimental] detonation ran {} instr(s), stop={:?}, observed oep={:?}, GetProcAddress imports={}, observed globals={}",
                    result.steps,
                    result.stop,
                    result.observed.oep,
                    result.observed.imports.len(),
                    result.observed.globals.len(),
                ));
                dynamic_stop = Some(format!("{:?}", result.stop));
                dynamic_steps = result.steps;
                dynamic_oep = result.observed.oep;
                dynamic_globals = result
                    .observed
                    .globals
                    .iter()
                    .map(|global| (global.va, global.bytes.clone()))
                    .collect();
                result.observed.imports
            }
            Err(error) => {
                warnings
                    .push(format!("[dynamic][experimental] detonation could not start: {error:?}"));
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };
    // Observed-tier seed for the experimental decoded-lift path only. `None`
    // keeps the sound path's recovery seedless and unchanged.
    let observed_memory: Option<vmpx_vm::ObservedMemory> = if dynamic {
        Some(vmpx_vm::ObservedMemory::from_slots(dynamic_globals))
    } else {
        None
    };
    // Exact native exit states retained for the recompilation stage.
    // Each exit state carries the observable effects its replacement must
    // replay. A replacement that only restored registers would compute the right
    // values and skip the routine's real side effects.
    let mut exit_states: Vec<(
        BTreeMap<String, u64>,
        vmpx_vm::NativeExitState,
        vmpx_vmlift::CollectedEffects,
    )> = Vec::new();
    // Nested VM entries reached through a recovered continuation. Each one is
    // recovered recursively so a path that re-enters the VM still terminates in
    // a native exit state rather than stopping at a protector transition.
    let mut nested_exits: Vec<(
        BTreeMap<String, u64>,
        vmpx_vm::NativeExitState,
        vmpx_vmlift::CollectedEffects,
    )> = Vec::new();
    let mut continuations = Vec::new();
    let mut handlers = CleanHandlerMetrics {
        unique: 0,
        classified: 0,
        unknown: 0,
    };
    let mut unknown_handlers = Vec::new();

    let header_entry = discovery.entries.iter().find(|candidate| {
        candidate.source == VmEntrySource::PeEntryPoint
            && candidate.rva == pe.optional.address_of_entry_point
    });
    let mut header_entry_candidate: Option<vmpx_vm::VmEntryCandidate> = None;
    // Per header path: its single-resolved continuation RVA, or None.
    let mut oep_paths: Vec<Option<Rva>> = Vec::new();
    // Unknown counts as incomplete: a group only skips its dispatcher fallback
    // when its family was proven complete.
    let mut header_family_complete = false;
    let exe_header_rva = header_entry
        .filter(|_| pe.coff.characteristics & IMAGE_FILE_DLL == 0)
        .map(|entry| entry.rva);
    let mut exe_entry_function = None;
    if let Some(entry) = header_entry {
        header_entry_candidate = Some(entry.clone());
        if pe.coff.characteristics & IMAGE_FILE_DLL == 0 {
            // No DllMain-style reason selects a path: the executable entry is
            // recovered as a single complete input-closed path or stays a blocker.
            let decision =
                crate::input_closed_functions::decide_exe_entry(entry.confidence, || {
                    crate::input_closed_functions::recover_input_closed(&pe, entry.rva)
                });
            blockers.extend(decision.blocker);
            exe_entry_function = decision.function;
        } else {
            let (domains, limits) = dll_entry_recovery_inputs();
            match recover_entry_paths(&pe, entry, &domains, limits) {
                Ok(family) => {
                    header_family_complete =
                        family.domain_complete && family.all_paths_reach_vm_exit;
                    let audit = audit_entry_effect_liveness(&family);
                    (handlers, unknown_handlers) = handler_metrics(&family);
                    for (path, live) in family.paths.iter().zip(&audit.paths) {
                        let continuation_targets = path.native_exit.as_ref().and_then(|exit| {
                            recover_modulo_image_targets(
                                &pe,
                                &exit.return_address,
                                ModuloImageTargetLimits::default(),
                            )
                            .ok()
                        });
                        oep_paths.push(
                            continuation_targets
                                .as_ref()
                                .and_then(single_continuation_rva),
                        );
                        let mut issues = live.issues.clone();
                        if path.chain.stop_reason == VmProfileChainStop::SymbolicExecutionFailure {
                            issues.extend(path.chain.diagnostics.clone());
                        }
                        let exception_coverage = exception_coverage_for_chain(&pe, &path.chain);
                        if let Some(exit) = path.native_exit.as_ref() {
                            exit_states.push((
                                path.assignments.clone(),
                                exit.clone(),
                                vmpx_vmlift::collect_path_effects(&family, path),
                            ));
                            // Follow a continuation that re-enters the VM, so the
                            // emitted chain can end at a real native return.
                            if let Ok(targets) = vmpx_vm::recover_modulo_image_targets(
                                &pe,
                                &exit.return_address,
                                vmpx_vm::ModuloImageTargetLimits::default(),
                            ) && targets.complete
                                && targets.targets.len() == 1
                                && let Some(target_rva) = targets.targets[0].rva
                                && let Some(nested) = analyze_vm_entry_at(
                                    &pe,
                                    target_rva,
                                    VmDiscoveryLimits::default(),
                                )
                                && nested.confidence >= 60
                            {
                                // The nested entry is reached after the VM has run,
                                // so its inputs are the register values this exit
                                // produced, not fresh loader parameters. Analysing
                                // it with a DllMain reason domain would invent a
                                // value the VM never left there.
                                let nested_domains: BTreeMap<String, Vec<Expr>> = exit
                                    .registers
                                    .iter()
                                    .filter(|(_, value)| {
                                        matches!(value.kind, vmpx_vm::ExprKind::Const { .. })
                                    })
                                    .map(|(register, value)| {
                                        (register.symbolic_name().to_owned(), vec![value.clone()])
                                    })
                                    .collect();
                                if let Ok(nested_family) =
                                    recover_entry_paths(&pe, &nested, &nested_domains, limits)
                                {
                                    for nested_path in &nested_family.paths {
                                        if let Some(nested_exit) = nested_path.native_exit.as_ref()
                                        {
                                            // Label the nested body distinctly: it
                                            // inherits the outer assignments, so an
                                            // unmarked copy would be indistinguishable
                                            // from the outer path in diagnostics.
                                            let mut assignments = path.assignments.clone();
                                            assignments.extend(nested_path.assignments.clone());
                                            assignments.insert(
                                                "nested_entry".to_owned(),
                                                u64::from(target_rva.0),
                                            );
                                            nested_exits.push((
                                                assignments,
                                                nested_exit.clone(),
                                                vmpx_vmlift::collect_path_effects(
                                                    &nested_family,
                                                    nested_path,
                                                ),
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                        paths.push(CleanVmPathReport {
                            assignments: path.assignments.clone(),
                            segments: path.chain.segments.len(),
                            decoded_operations: path.chain.total_instructions,
                            stop_reason: path.chain.stop_reason,
                            native_exit_recovered: path.native_exit.is_some(),
                            continuation_targets: continuation_targets.clone(),
                            exception_coverage,
                            issues,
                        });

                        let Some(exit) = &path.native_exit else {
                            continue;
                        };
                        let Some(targets) = continuation_targets else {
                            continue;
                        };
                        if !targets.complete || targets.targets.len() != 1 {
                            continue;
                        }
                        let Some(target) = targets.targets[0].rva else {
                            continue;
                        };
                        let candidate =
                            analyze_vm_entry_at(&pe, target, VmDiscoveryLimits::default());
                        if let Some(candidate) = candidate
                            .as_ref()
                            .filter(|candidate| candidate.confidence >= 60)
                        {
                            let prior_symbolic = path
                                .chain
                                .segments
                                .last()
                                .and_then(|segment| segment.symbolic.as_ref());
                            if let Some(prior_symbolic) = prior_symbolic {
                                match continue_through_native_vm_entry(
                                    &pe,
                                    prior_symbolic,
                                    exit,
                                    candidate,
                                    EntryProfileRecoveryLimits::default(),
                                    limits.chain,
                                ) {
                                    Ok(nested) => {
                                        continuations.push(CleanNestedContinuationReport {
                                            source_assignments: path.assignments.clone(),
                                            target,
                                            vm_entry_confidence: candidate.confidence,
                                            decoded_operations: nested.chain.total_instructions,
                                            stop_reason: Some(nested.chain.stop_reason),
                                            exception_coverage: exception_coverage_for_chain(
                                                &pe,
                                                &nested.chain,
                                            ),
                                            native_cfg_blocks: None,
                                            native_analysis: None,
                                            issues: nested.chain.diagnostics.clone(),
                                        });
                                    }
                                    Err(error) => errors.push(format!(
                                        "native VM continuation at {target} failed: {error}"
                                    )),
                                }
                            }
                        } else {
                            let cfg = recover_cfg(
                                &pe,
                                target,
                                CfgLimits {
                                    max_blocks: 8_192,
                                    max_instructions: 131_072,
                                    max_instructions_per_block: 8_192,
                                },
                            );
                            let native_analysis = analyze_native_continuation(
                                &pe,
                                target,
                                exit,
                                NativeContinuationLimits::default(),
                            )
                            .ok();
                            let issues = native_analysis.as_ref().map_or_else(
                                || vec!["native continuation semantics are not lifted".to_owned()],
                                |analysis| {
                                    if analysis.status == RecoveryStatus::Recovered {
                                        vec![
                                            "native continuation returns through an unmodeled caller stack target"
                                                .to_owned(),
                                        ]
                                    } else {
                                        analysis.diagnostics.clone()
                                    }
                                },
                            );
                            continuations.push(CleanNestedContinuationReport {
                                source_assignments: path.assignments.clone(),
                                target,
                                vm_entry_confidence: candidate
                                    .as_ref()
                                    .map_or(0, |candidate| candidate.confidence),
                                decoded_operations: 0,
                                stop_reason: None,
                                exception_coverage: Vec::new(),
                                native_cfg_blocks: cfg.as_ref().ok().map(|cfg| cfg.blocks.len()),
                                native_analysis,
                                issues,
                            });
                        }
                    }
                    if !audit.ready_for_native_replacement {
                        blockers.push(
                            "entry path family is not proven safe for native replacement"
                                .to_owned(),
                        );
                    }
                }
                Err(error) => {
                    errors.push(format!("entry path recovery failed: {error}"));
                    blockers.push("header VM entry path recovery is incomplete".to_owned());
                }
            }
        }
    } else if protection.verdict != DetectionVerdict::NoStrongEvidence {
        blockers.push("protected header entry has no recovered VM profile".to_owned());
    }

    // A TLS callback is only a blocker if its VM behaviour is still unknown.
    // Analyzing each one turns "N protected TLS entries" into per-entry
    // evidence and clears the blocker when every path provably reaches a VM
    // exit with no remaining semantic gaps.
    let export_reports = export_dependency_report(&pe);
    let (vm_inventory, inventory_loss) = build_vm_inventory(&pe, &export_reports, &mut warnings);
    // Single-body functions (input-closed set plus the executable entry) that
    // reintegration below replaces in place.
    let mut single_body_functions: Vec<crate::input_closed_functions::InputClosedFunction> =
        Vec::new();
    let mut branching_functions: Vec<crate::input_closed_functions::BranchingFunction> = Vec::new();
    let input_closed_functions = if protection.verdict == DetectionVerdict::NoStrongEvidence {
        completed_stage("not required for an input with no strong protection evidence")
    } else {
        let mut recovered = crate::input_closed_functions::recover_input_closed_functions(
            &pe,
            &vm_inventory,
            exe_header_rva,
        );
        recovered.closed.extend(exe_entry_function);
        single_body_functions.clone_from(&recovered.closed);
        branching_functions.clone_from(&recovered.branching);
        let mut diagnostics = std::mem::take(&mut recovered.diagnostics);
        diagnostics.extend(recovered.closed.iter().map(|function| {
            format!(
                "{}: input-closed ({} inputs bound, live-in {:?})",
                function.rva,
                function.inputs.len(),
                function.live_in
            )
        }));
        diagnostics.extend(recovered.branching.iter().map(|function| {
            format!(
                "{}: branching, {} path(s), {} inputs bound, proven-exhaustive finite domain {:?} (live-in {:?})",
                function.rva,
                function.paths.len(),
                function.inputs.len(),
                function.certificates,
                function.live_in
            )
        }));
        CleanStageReport {
            complete: false,
            metrics: BTreeMap::from([
                (
                    "recovered".to_owned(),
                    u64::try_from(recovered.closed.len()).unwrap_or(u64::MAX),
                ),
                (
                    "recovered_branching".to_owned(),
                    u64::try_from(recovered.branching.len()).unwrap_or(u64::MAX),
                ),
            ]),
            diagnostics,
        }
    };
    let mut tls_entries = Vec::new();
    let mut tls_family_complete: BTreeMap<Rva, bool> = BTreeMap::new();
    let mut tls_exit_states: Vec<(
        Rva,
        BTreeMap<String, u64>,
        vmpx_vm::NativeExitState,
        vmpx_vmlift::CollectedEffects,
    )> = Vec::new();
    for entry in discovery
        .entries
        .iter()
        .filter(|entry| entry.source == VmEntrySource::TlsCallback)
    {
        let domains = BTreeMap::from([(
            "native_rdx_initial".to_owned(),
            NATIVE_RDX_DOMAIN
                .map(|reason| Expr::constant(BitWidth::W64, reason))
                .collect(),
        )]);
        let limits = EntryPathRecoveryLimits {
            chain: VmProfileExplorerLimits {
                max_segments: 32,
                max_instructions_per_segment: 2_048,
                max_total_instructions: 8_192,
                ..VmProfileExplorerLimits::default()
            },
            ..EntryPathRecoveryLimits::default()
        };
        let report = match recover_entry_paths(&pe, entry, &domains, limits) {
            Ok(family) => {
                let mut issues = Vec::new();
                let mut native_exit_recovered = true;
                let mut segments = 0_usize;
                let mut decoded_operations = 0_usize;
                let mut stop_reason = None;
                for path in &family.paths {
                    segments = segments.max(path.chain.segments.len());
                    decoded_operations = decoded_operations.max(path.chain.total_instructions);
                    stop_reason = stop_reason.or(Some(path.chain.stop_reason));
                    if path.native_exit.is_none() {
                        native_exit_recovered = false;
                        issues.push(format!(
                            "path {:?} has no exact native exit state: {:?}",
                            path.assignments, path.chain.stop_reason
                        ));
                    }
                    issues.extend(path.chain.diagnostics.iter().cloned());
                }
                tls_family_complete.insert(
                    entry.rva,
                    family.domain_complete && family.all_paths_reach_vm_exit,
                );
                if !family.domain_complete {
                    issues.push("the TLS entry domain is not complete".to_owned());
                }
                if !family.all_paths_reach_vm_exit {
                    issues.push("not every TLS entry path reaches a proven VM exit".to_owned());
                }
                // A loader-reachable TLS callback is a virtualized routine in its
                // own right. Its exit states belong in the emission set for the
                // same reason the header entry's do: they are the only code that
                // can replace it.
                for path in &family.paths {
                    if let Some(exit) = path.native_exit.as_ref() {
                        tls_exit_states.push((
                            entry.rva,
                            path.assignments.clone(),
                            exit.clone(),
                            vmpx_vmlift::collect_path_effects(&family, path),
                        ));
                    }
                }
                CleanTlsEntryReport {
                    entry: entry.rva,
                    confidence: entry.confidence,
                    recovered: family.domain_complete && family.all_paths_reach_vm_exit,
                    domain_complete: family.domain_complete,
                    all_paths_reach_vm_exit: family.all_paths_reach_vm_exit,
                    segments,
                    decoded_operations,
                    stop_reason,
                    native_exit_recovered,
                    issues,
                }
            }
            Err(error) => CleanTlsEntryReport {
                entry: entry.rva,
                confidence: entry.confidence,
                recovered: false,
                domain_complete: false,
                all_paths_reach_vm_exit: false,
                segments: 0,
                decoded_operations: 0,
                stop_reason: None,
                native_exit_recovered: false,
                issues: vec![format!("TLS entry recovery failed: {error}")],
            },
        };
        tls_entries.push(report);
    }
    let unresolved_tls = tls_entries.iter().filter(|entry| !entry.recovered).count();
    if unresolved_tls != 0 {
        blockers.push(format!(
            "{unresolved_tls} loader-reachable TLS VM entry/entries remain semantically unresolved"
        ));
    }
    // The conventional image is computed up front so compiled body addresses
    // match where `append_section` will place them below.
    let mut relayout = if protection.verdict == DetectionVerdict::NoStrongEvidence {
        None
    } else {
        Some(relayout_conventional(&mapped))
    };
    let planning_image = relayout
        .as_ref()
        .and_then(|result| result.as_ref().ok())
        .map(|(bytes, _)| bytes);
    // Native recompilation: emit a replacement function for every path whose
    // exit state is fully concrete and lowerable. Emission does not authorise
    // writing output, because the entry redirect and the remaining blockers are
    // still unproven.
    let mut recompilation = stage("no path reached a lowerable exit state");
    let mut reintegration = stage("native code reintegration was not attempted");
    let mut reintegrated: Option<Vec<u8>> = None;
    let mut header_entry_redirected = false;
    // Entry stubs whose redirect to an emitted replacement actually succeeded.
    let mut covered_rvas: BTreeSet<Rva> = BTreeSet::new();
    if protection.verdict != DetectionVerdict::NoStrongEvidence {
        let planned = planning_image.and_then(|bytes| {
            let rva = vmpx_rebuild::next_section_rva(bytes).ok()?;
            let va = pe.optional.image_base.checked_add_rva(rva)?.0;
            Some((bytes, rva, va))
        });
        let mut emitted = 0_u64;
        let mut total_bytes = 0_u64;
        let mut placed: Vec<(BTreeMap<String, u64>, Rva)> = Vec::new();
        let mut diagnostics = Vec::new();
        if let Some((image, planned_rva, va)) = planned {
            let mut symbol_rewrites: BTreeMap<String, vmpx_vm::Expr> = BTreeMap::new();
            let inputs = match header_entry_candidate.as_ref() {
                Some(entry) => match replacement_inputs(&pe, entry, &mut symbol_rewrites) {
                    Ok(inputs) => inputs,
                    Err(error) => {
                        diagnostics.push(error);
                        Vec::new()
                    }
                },
                None => Vec::new(),
            };

            // Groups are laid out in order: the header entry first, then each
            // loader-reachable TLS callback (sorted by RVA), so each group's own
            // dispatcher sits right after its own bodies and before the next
            // group's bodies start.
            type ExitRefs<'a> = (
                &'a BTreeMap<String, u64>,
                &'a vmpx_vm::NativeExitState,
                &'a vmpx_vmlift::CollectedEffects,
            );
            let mut tls_groups: BTreeMap<Rva, Vec<ExitRefs<'_>>> = BTreeMap::new();
            for (owner, assignments, exit, effects) in &tls_exit_states {
                tls_groups
                    .entry(*owner)
                    .or_default()
                    .push((assignments, exit, effects));
            }
            let mut groups: Vec<(GroupKind, bool, Vec<ExitRefs<'_>>)> = vec![(
                GroupKind::Header,
                header_family_complete,
                exit_states.iter().map(|(a, e, f)| (a, e, f)).collect(),
            )];
            groups.extend(tls_groups.into_iter().map(|(rva, items)| {
                let complete = tls_family_complete.get(&rva).copied().unwrap_or(false);
                (GroupKind::Tls(rva), complete, items)
            }));

            let mut layout_cursor: u32 = 0;
            let mut reintegration_payload: Vec<u8> = Vec::new();
            let mut header_redirect: Option<Rva> = None;
            let mut header_family_proven = false;
            let mut tls_redirects: Vec<(usize, Rva)> = Vec::new();
            let mut tls_redirect_stubs: Vec<(Rva, bool)> = Vec::new();
            let mut dispatcher_count: u64 = 0;
            let mut reintegration_diagnostics: Vec<String> = Vec::new();

            for (kind, family_complete, items) in &groups {
                let group_label = match kind {
                    GroupKind::Header => "header entry".to_owned(),
                    GroupKind::Tls(rva) => format!("TLS entry {rva}"),
                };
                // Eligibility is decided before compiling: a missing or
                // duplicate domain value must not consume address space that
                // the real payload below won't contain.
                let mut eligible: Vec<(
                    u64,
                    &BTreeMap<String, u64>,
                    &vmpx_vm::NativeExitState,
                    &vmpx_vmlift::CollectedEffects,
                )> = Vec::new();
                let mut seen_domain_values = BTreeSet::new();
                for (assignments, exit, effects) in items {
                    match assignments.get("native_rdx_initial") {
                        None => diagnostics.push(format!(
                            "path {assignments:?} in {group_label} has no native_rdx_initial \
                             domain value; excluded from reintegration"
                        )),
                        Some(&domain_value) => {
                            if seen_domain_values.insert(domain_value) {
                                eligible.push((domain_value, assignments, exit, effects));
                            } else {
                                diagnostics.push(format!(
                                    "duplicate native_rdx_initial={domain_value} in \
                                     {group_label}; keeping the first body and skipping this path"
                                ));
                            }
                        }
                    }
                }
                let group_start_cursor = layout_cursor;
                let placed_start_len = placed.len();
                let emitted_before = emitted;
                let total_bytes_before = total_bytes;
                let mut group_bodies: Vec<(u64, Vec<u8>)> = Vec::new();
                for (domain_value, assignments, exit, effects) in &eligible {
                    let body_va = va + u64::from(layout_cursor);
                    match compile_replacement_body(
                        &pe,
                        &inputs,
                        &symbol_rewrites,
                        exit,
                        effects,
                        body_va,
                    ) {
                        Ok(body) => {
                            // Record where this body landed so a redirect can
                            // target it: the entry stub has to jump somewhere
                            // specific.
                            let entry_rva = Rva(planned_rva
                                .0
                                .checked_add(layout_cursor)
                                .unwrap_or(planned_rva.0));
                            placed.push(((*assignments).clone(), entry_rva));
                            layout_cursor = layout_cursor
                                .saturating_add(u32::try_from(body.len()).unwrap_or(0));
                            total_bytes += u64::try_from(body.len()).unwrap_or(0);
                            emitted += 1;
                            group_bodies.push((*domain_value, body));
                        }
                        Err(error) => diagnostics.push(format!(
                            "path {assignments:?} has no native replacement: {error}"
                        )),
                    }
                }
                if group_bodies.is_empty() {
                    continue;
                }
                let fallback = match kind {
                    GroupKind::Header => pe.optional.address_of_entry_point,
                    GroupKind::Tls(rva) => *rva,
                };
                match commit_group_layout(
                    &mut layout_cursor,
                    &mut reintegration_payload,
                    group_start_cursor,
                    planned_rva,
                    pe.optional.image_base.0,
                    &group_bodies,
                    fallback,
                    *family_complete,
                ) {
                    Ok((redirect, dispatcher_used)) => {
                        if dispatcher_used {
                            dispatcher_count += 1;
                        }
                        match kind {
                            GroupKind::Header => {
                                header_redirect = Some(redirect);
                                header_family_proven = *family_complete;
                            }
                            GroupKind::Tls(rva) => match tls_callback_index(&pe, *rva) {
                                Some(index) => {
                                    tls_redirects.push((index, redirect));
                                    tls_redirect_stubs.push((*rva, *family_complete));
                                }
                                None => reintegration_diagnostics.push(format!(
                                    "TLS entry {rva} has no resolvable callback-array index; \
                                     its body is emitted but not redirected"
                                )),
                            },
                        }
                    }
                    Err(error) => {
                        // The cursor is already rolled back by
                        // `commit_group_layout`; undo this group's other
                        // contributions too, so a skipped group leaves no gap
                        // and no phantom bytes or counts behind it.
                        placed.truncate(placed_start_len);
                        emitted = emitted_before;
                        total_bytes = total_bytes_before;
                        reintegration_diagnostics
                            .push(format!("{group_label} dispatcher assembly failed: {error}"));
                    }
                }
            }

            // Single-body functions: each gets its own inputs and carried
            // effects, is laid out after the groups, and has its VM-entry stub
            // redirected once the section exists. A failure skips only that
            // function and leaves the cursor and payload untouched.
            let is_dll = pe.coff.characteristics & IMAGE_FILE_DLL != 0;
            let mut stub_redirects: Vec<(Rva, Rva)> = Vec::new();
            let mut branching_stubs: Vec<(Rva, Rva)> = Vec::new();
            let mut seen_stubs = BTreeSet::new();
            let mut single_body_bytes: u64 = 0;
            layout_cursor = u32::try_from(reintegration_payload.len()).unwrap_or(layout_cursor);
            for function in &single_body_functions {
                let handled_by_group = (is_dll
                    && function.rva == pe.optional.address_of_entry_point)
                    || tls_callback_index(&pe, function.rva).is_some();
                if handled_by_group || !seen_stubs.insert(function.rva) {
                    continue;
                }
                let Some(body_rva) = planned_rva.0.checked_add(layout_cursor).map(Rva) else {
                    reintegration_diagnostics.push(format!(
                        "{}: single-body layout overflow; skipped",
                        function.rva
                    ));
                    continue;
                };
                let body_va = va + u64::from(layout_cursor);
                match compile_single_body(&pe, function, body_va) {
                    Ok(body) => {
                        layout_cursor =
                            layout_cursor.saturating_add(u32::try_from(body.len()).unwrap_or(0));
                        let len = u64::try_from(body.len()).unwrap_or(0);
                        total_bytes += len;
                        single_body_bytes += len;
                        emitted += 1;
                        reintegration_payload.extend_from_slice(&body);
                        stub_redirects.push((function.rva, body_rva));
                    }
                    Err(error) => reintegration_diagnostics.push(format!(
                        "{}: single-body replacement not built ({} effect store(s)): {error}",
                        function.rva,
                        function.effects.stores.len()
                    )),
                }
            }

            // Branching functions: bodies, the branch-target callable, a
            // displaced copy of the stub and the value dispatcher form one
            // contiguous unit, committed only if the whole unit builds.
            let mut branching_bytes: u64 = 0;
            let mut branching_built: u64 = 0;
            for function in &branching_functions {
                let handled_by_group = (is_dll
                    && function.rva == pe.optional.address_of_entry_point)
                    || tls_callback_index(&pe, function.rva).is_some();
                if handled_by_group || !seen_stubs.insert(function.rva) {
                    continue;
                }
                let Some(unit_rva) = planned_rva.0.checked_add(layout_cursor).map(Rva) else {
                    reintegration_diagnostics.push(format!(
                        "{}: branching layout overflow; skipped",
                        function.rva
                    ));
                    continue;
                };
                match compile_branching_function(
                    &pe,
                    image,
                    function,
                    va + u64::from(layout_cursor),
                ) {
                    Ok((unit, dispatcher_offset)) => {
                        let Some(dispatcher_rva) =
                            unit_rva.0.checked_add(dispatcher_offset).map(Rva)
                        else {
                            reintegration_diagnostics.push(format!(
                                "{}: branching dispatcher RVA overflow; skipped",
                                function.rva
                            ));
                            continue;
                        };
                        let len = u64::try_from(unit.len()).unwrap_or(0);
                        layout_cursor =
                            layout_cursor.saturating_add(u32::try_from(unit.len()).unwrap_or(0));
                        total_bytes += len;
                        branching_bytes += len;
                        branching_built += 1;
                        emitted += u64::try_from(function.paths.len()).unwrap_or(0);
                        reintegration_payload.extend_from_slice(&unit);
                        branching_stubs.push((function.rva, dispatcher_rva));
                    }
                    Err(error) => reintegration_diagnostics.push(format!(
                        "{}: branching function deferred ({} path(s)): {error}",
                        function.rva,
                        function.paths.len()
                    )),
                }
            }

            // Decoded-stream lifts (opt-in, slow): every remaining inventory entry whose handler
            // stream lifts to a clean callee return. A failed lift skips only
            // that entry and leaves its stub; redirects are not counted as
            // coverage, so the sound gate is unaffected.
            let mut decoded_stubs: Vec<(Rva, Rva, BTreeSet<String>)> = Vec::new();
            let mut decoded_bytes: u64 = 0;
            let mut decoded_failed: u64 = 0;
            for entry in vm_inventory.iter().filter(|_| decoded_lift) {
                let handled_by_group = (is_dll && entry.rva == pe.optional.address_of_entry_point)
                    || tls_callback_index(&pe, entry.rva).is_some();
                if handled_by_group || !seen_stubs.insert(entry.rva) {
                    continue;
                }
                let Some(body_rva) = planned_rva.0.checked_add(layout_cursor).map(Rva) else {
                    continue;
                };
                match crate::decoded_splice::lift_entry(
                    &pe,
                    entry.rva,
                    va + u64::from(layout_cursor),
                    observed_memory.as_ref(),
                ) {
                    Ok(body) => {
                        let len = u64::try_from(body.bytes.len()).unwrap_or(0);
                        layout_cursor =
                            layout_cursor.saturating_add(u32::try_from(len).unwrap_or(0));
                        total_bytes += len;
                        decoded_bytes += len;
                        emitted += 1;
                        reintegration_payload.extend_from_slice(&body.bytes);
                        decoded_stubs.push((entry.rva, body_rva, body.imports));
                        reintegration_diagnostics.push(format!(
                            "{}: decoded-stream {} lift, {len} byte(s) at {body_rva}",
                            entry.rva, body.kind
                        ));
                    }
                    Err(error) => {
                        decoded_failed += 1;
                        reintegration_diagnostics.push(format!(
                            "{}: decoded-stream lift skipped: {error}",
                            entry.rva
                        ));
                    }
                }
            }

            // Nested-continuation bodies still compile and count here (nothing
            // is silently dropped), but reintegration is deferred: redirecting
            // to one means rewriting another body's continuation target, which
            // is out of scope here.
            for (assignments, exit, effects) in &nested_exits {
                let body_va = va + u64::from(layout_cursor);
                match compile_replacement_body(
                    &pe,
                    &inputs,
                    &symbol_rewrites,
                    exit,
                    effects,
                    body_va,
                ) {
                    Ok(body) => {
                        // This RVA is past the real payload (nested bodies are
                        // never appended, only counted), so it is not a live
                        // placement; diagnostics should read it that way.
                        let entry_rva = Rva(planned_rva
                            .0
                            .checked_add(layout_cursor)
                            .unwrap_or(planned_rva.0));
                        placed.push((assignments.clone(), entry_rva));
                        layout_cursor =
                            layout_cursor.saturating_add(u32::try_from(body.len()).unwrap_or(0));
                        total_bytes += u64::try_from(body.len()).unwrap_or(0);
                        emitted += 1;
                    }
                    Err(error) => diagnostics.push(format!(
                        "path {assignments:?} has no native replacement: {error}"
                    )),
                }
            }

            if reintegration_payload.is_empty() {
                if !reintegration_diagnostics.is_empty() {
                    reintegration = CleanStageReport {
                        complete: false,
                        metrics: BTreeMap::new(),
                        diagnostics: reintegration_diagnostics,
                    };
                }
            } else {
                reintegration = match vmpx_rebuild::reintegrate(
                    image,
                    vmpx_rebuild::ReintegrationRequest {
                        section_name: *b".vmpx0\0\0",
                        payload: &reintegration_payload,
                        entry_redirect: header_redirect,
                        tls_redirects: &tls_redirects,
                    },
                ) {
                    Ok((mut bytes, report)) => {
                        header_entry_redirected = report.entry_redirected;
                        // `reintegrate` fails as a whole if any TLS redirect fails,
                        // so on Ok every requested one landed.
                        if header_redirect.is_some() {
                            clean_wiring::group_covered(
                                &mut covered_rvas,
                                pe.optional.address_of_entry_point,
                                report.entry_redirected,
                                header_family_proven,
                            );
                        }
                        if report.tls_redirected == tls_redirect_stubs.len() {
                            for (stub, complete) in &tls_redirect_stubs {
                                clean_wiring::group_covered(
                                    &mut covered_rvas,
                                    *stub,
                                    true,
                                    *complete,
                                );
                            }
                        }
                        let mut functions_redirected: u64 = 0;
                        for (stub, body) in &stub_redirects {
                            let outcome = vmpx_rebuild::redirect_vm_stub(&bytes, *stub, *body);
                            clean_wiring::mark_covered(&mut covered_rvas, *stub, &outcome);
                            match outcome {
                                Ok(redirected) => {
                                    bytes = redirected;
                                    functions_redirected += 1;
                                    reintegration_diagnostics.push(format!(
                                        "{stub}: VM entry stub redirected to single-body                                          replacement at {body}"
                                    ));
                                }
                                Err(error) => reintegration_diagnostics.push(format!(
                                    "{stub}: stub redirect failed, original stub kept: {error}"
                                )),
                            }
                        }
                        let mut branching_redirected: u64 = 0;
                        for (stub, dispatcher) in &branching_stubs {
                            let outcome =
                                vmpx_rebuild::redirect_vm_stub(&bytes, *stub, *dispatcher);
                            clean_wiring::mark_covered(&mut covered_rvas, *stub, &outcome);
                            match outcome {
                                Ok(redirected) => {
                                    bytes = redirected;
                                    branching_redirected += 1;
                                    reintegration_diagnostics.push(format!(
                                        "{stub}: VM entry stub redirected to value dispatcher at {dispatcher}"
                                    ));
                                }
                                Err(error) => reintegration_diagnostics.push(format!(
                                    "{stub}: dispatcher redirect failed, original stub kept: {error}"
                                )),
                            }
                        }
                        let mut decoded_redirected: u64 = 0;
                        let mut resolved_imports: BTreeSet<String> = BTreeSet::new();
                        for (stub, body, imports) in &decoded_stubs {
                            match vmpx_rebuild::redirect_vm_stub(&bytes, *stub, *body) {
                                Ok(redirected) => {
                                    bytes = redirected;
                                    decoded_redirected += 1;
                                    resolved_imports.extend(imports.iter().cloned());
                                }
                                Err(error) => reintegration_diagnostics.push(format!(
                                    "{stub}: decoded-stream redirect failed, original stub kept: {error}"
                                )),
                            }
                        }
                        reintegrated = Some(bytes);
                        let entries_redirected = u64::from(report.entry_redirected)
                            + u64::try_from(report.tls_redirected).unwrap_or(0);
                        let mut diags = vec![format!(
                            "{entries_redirected} entry/entries redirected ({} header, {} TLS \
                             callback(s)); nested-continuation redirects remain deferred",
                            u64::from(report.entry_redirected),
                            report.tls_redirected
                        )];
                        if !resolved_imports.is_empty() {
                            diags.push(format!(
                                "spliced functions call proven imports: {}",
                                resolved_imports
                                    .iter()
                                    .map(String::as_str)
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ));
                        }
                        diags.extend(reintegration_diagnostics);
                        CleanStageReport {
                            complete: true,
                            metrics: BTreeMap::from([
                                ("section_rva".to_owned(), u64::from(report.section.rva.0)),
                                (
                                    "section_size".to_owned(),
                                    u64::from(report.section.virtual_size),
                                ),
                                ("entries_redirected".to_owned(), entries_redirected),
                                ("dispatchers".to_owned(), dispatcher_count),
                                (
                                    "single_body_candidates".to_owned(),
                                    u64::try_from(stub_redirects.len()).unwrap_or(0),
                                ),
                                ("single_body_redirected".to_owned(), functions_redirected),
                                ("single_body_bytes".to_owned(), single_body_bytes),
                                ("branching_built".to_owned(), branching_built),
                                ("branching_redirected".to_owned(), branching_redirected),
                                ("branching_bytes".to_owned(), branching_bytes),
                                (
                                    "decoded_lifted".to_owned(),
                                    u64::try_from(decoded_stubs.len()).unwrap_or(0),
                                ),
                                ("decoded_redirected".to_owned(), decoded_redirected),
                                (
                                    "resolved_imports".to_owned(),
                                    u64::try_from(resolved_imports.len()).unwrap_or(0),
                                ),
                                ("decoded_failed".to_owned(), decoded_failed),
                                ("decoded_bytes".to_owned(), decoded_bytes),
                                (
                                    "tls_redirected".to_owned(),
                                    u64::try_from(report.tls_redirected).unwrap_or(0),
                                ),
                            ]),
                            diagnostics: diags,
                        }
                    }
                    Err(error) => {
                        reintegration_diagnostics.push(format!("reintegrate failed: {error}"));
                        CleanStageReport {
                            complete: false,
                            metrics: BTreeMap::new(),
                            diagnostics: reintegration_diagnostics,
                        }
                    }
                };
            }
        }
        if emitted == 0 {
            recompilation = CleanStageReport {
                complete: false,
                metrics: BTreeMap::new(),
                diagnostics,
            };
        } else {
            recompilation = CleanStageReport {
                complete: false,
                metrics: BTreeMap::from([
                    ("emitted".to_owned(), emitted),
                    ("bytes".to_owned(), total_bytes),
                ]),
                diagnostics: {
                    // Where each body landed, so a redirect target is stated
                    // rather than implied by the layout.
                    for (assignments, rva) in &placed {
                        diagnostics.push(format!(
                            "replacement for {assignments:?} placed at RVA {:#x}",
                            rva.0
                        ));
                    }
                    diagnostics.push(format!(
                        "{emitted} native replacement function(s) emitted; the VM entry is not redirected because the remaining proofs are open"
                    ));
                    diagnostics
                },
            };
        }
    }
    let verified_noop = protection.verdict == DetectionVerdict::NoStrongEvidence
        && protection.image_layout == ImageLayout::ConventionalFile;
    let oep_decision = decide_oep(&oep_paths, header_family_complete, |rva| {
        is_native_continuation(&pe, rva)
    });
    if let (Some(bytes), Some(oep), false) = (
        reintegrated.as_mut(),
        oep_decision.recovered_entry,
        header_entry_redirected,
    ) {
        match vmpx_rebuild::set_entry_point(bytes, oep) {
            Ok(updated) => {
                *bytes = updated;
                reintegration.metrics.insert("oep_set".to_owned(), 1);
                reintegration
                    .diagnostics
                    .push(format!("entry point set to the proven OEP {oep}"));
            }
            Err(error) => reintegration
                .diagnostics
                .push(format!("proven OEP {oep} not applied: {error}")),
        }
    }
    let mut coverage: Option<CleanCoverageReport> = None;
    let mut import_proof: Option<clean_wiring::ImportProof> = None;
    let mut observed_import_report: Option<CleanObservedImportReport> = None;
    if verified_noop {
        warnings
            .push("input has no strong VMProtect evidence; clean is a verified no-op".to_owned());
    } else if protection.verdict == DetectionVerdict::NoStrongEvidence {
        blockers.push(
            "input has no strong VMProtect evidence but its layout is not a conventional PE file"
                .to_owned(),
        );
    } else {
        let (inventory, dropped_entries) = clean_wiring::build_inventory_items(
            &vm_inventory,
            discovery.entries.iter().map(|entry| entry.rva),
            &inventory_loss,
        );
        let dangling = clean_wiring::summarize_dangling(&collect_dangling_refs(&pe, &covered_rvas));
        let decision = decide_coverage(&CoverageInput {
            inventory: &inventory,
            recovered: &covered_rvas,
            oep_proven: oep_decision.recovered_entry.is_some(),
            dangling_vmp_refs: &dangling.targets,
            dropped_entries,
        });
        blockers.extend(clean_wiring::coverage_blockers(
            &decision,
            dangling.has_unresolved,
            reintegrated.is_some(),
        ));
        // SOUND path is unchanged: referenced slots stay empty, so the import
        // blocker is kept regardless of any dynamic observation.
        let proof = clean_wiring::prove_imports(
            &report.imports,
            &clean_wiring::derive_referenced_slots(None),
        );
        if !proof.complete {
            blockers.push(clean_wiring::IMPORT_BLOCKER.to_owned());
        }
        import_proof = Some(proof);

        // EXPERIMENTAL Observed-tier reconstruction (report only; the sound
        // blocker above is untouched). When the detonation resolved imports,
        // reconstruct the IAT from them and run the real installer.
        if !observed_imports.is_empty() {
            let recon = clean_wiring::reconstruct_observed_imports(
                pe.optional.kind,
                Rva(0),
                &observed_imports,
            );
            observed_import_report = Some(CleanObservedImportReport {
                provenance: "Observed",
                detonation_steps: dynamic_steps,
                detonation_stop: dynamic_stop.clone().unwrap_or_default(),
                observed_oep: dynamic_oep,
                observed_imports: observed_imports.len(),
                reconstructed_modules: recon.proof.proven_modules,
                reconstructed_symbols: recon.proof.proven_symbols,
                reconstruction_complete: recon.proof.complete,
                import_directory_bytes: recon.directory_bytes,
                note: "experimental dynamic-assist; Observed-tier, never sound output; the \
                       import blocker above is unaffected"
                    .to_owned(),
            });
        } else if dynamic {
            observed_import_report = Some(CleanObservedImportReport {
                provenance: "Observed",
                detonation_steps: dynamic_steps,
                detonation_stop: dynamic_stop.clone().unwrap_or_default(),
                observed_oep: dynamic_oep,
                observed_imports: 0,
                reconstructed_modules: 0,
                reconstructed_symbols: 0,
                reconstruction_complete: false,
                import_directory_bytes: None,
                note: "experimental dynamic-assist; detonation reached no dynamically-resolved \
                       imports before its next blocker (see warnings)"
                    .to_owned(),
            });
        }
        coverage = Some(CleanCoverageReport {
            can_clear_oep: decision.can_clear_oep_blocker,
            can_clear_vmp_referenced: decision.can_clear_vmp_referenced_blocker
                && !dangling.has_unresolved,
            covered_entries: covered_rvas.iter().copied().collect(),
            dangling_refs: dangling.targets.len(),
            has_unresolved_dangling: dangling.has_unresolved,
            dropped_entries,
            reasons: decision.reasons,
        });
    }
    let anti_debug = clean_wiring::recognize_anti_debug(&[], &BTreeMap::new());
    blockers.sort();
    blockers.dedup();

    let (layout, conventional) = if verified_noop {
        (
            CleanLayoutReport {
                attempted: false,
                file_size: None,
                size_of_headers: None,
                sections: None,
                zero_fill_sections: None,
                loader_valid: None,
                loader_summary: None,
                diagnostics: vec![
                    "input is already a conventional PE file; no relayout was required".to_owned(),
                ],
            },
            None,
        )
    } else {
        match relayout
            .take()
            .unwrap_or_else(|| relayout_conventional(&mapped))
        {
            Ok((bytes, report)) => {
                let validation = validate_for_loader(&bytes);
                let (loader_valid, loader_summary, mut layout_diagnostics) = match &validation {
                    Ok(validation) => (
                        Some(true),
                        Some(format!(
                            "{} imports in {} modules, {} exports, {} TLS callbacks, {} relocations",
                            validation.import_symbols,
                            validation.import_modules,
                            validation.exports,
                            validation.tls_callbacks,
                            validation.relocation_entries
                        )),
                        vec![
                            "conventional layout preserves every section RVA; it is not devirtualized"
                                .to_owned(),
                        ],
                    ),
                    Err(error) => (
                        Some(false),
                        Some(error.to_string()),
                        vec![format!("loader validation failed: {error}")],
                    ),
                };
                if validation.is_err() {
                    errors.push(loader_summary.clone().unwrap_or_default());
                }
                layout_diagnostics.push(
                    "the loaded memory image is byte-identical to the analysis image; only \
                     SizeOfRawData and PointerToRawData change"
                        .to_owned(),
                );
                (
                    CleanLayoutReport {
                        attempted: true,
                        file_size: Some(report.file_size),
                        size_of_headers: Some(report.size_of_headers),
                        sections: Some(report.sections.len()),
                        zero_fill_sections: Some(report.zero_fill_sections),
                        loader_valid,
                        loader_summary,
                        diagnostics: layout_diagnostics,
                    },
                    Some(bytes),
                )
            }
            Err(error) => {
                errors.push(format!("conventional layout failed: {error}"));
                (
                    CleanLayoutReport {
                        attempted: true,
                        file_size: None,
                        size_of_headers: None,
                        sections: None,
                        zero_fill_sections: None,
                        loader_valid: Some(false),
                        loader_summary: Some(error.to_string()),
                        diagnostics: vec![error.to_string()],
                    },
                    None,
                )
            }
        }
    };

    Ok((
        CleanReport {
            schema_version: 3,
            input: CleanInputReport {
                path: input_path.to_owned(),
                sha256: sha256(input),
                file_size: u64::try_from(input.len()).unwrap_or(u64::MAX),
            },
            pe: pe_report,
            protection,
            unpacking: CleanUnpackingReport {
                performed: unpack_report.is_some(),
                mapped_image_size: u64::try_from(mapped.len()).unwrap_or(u64::MAX),
                report: unpack_report,
            },
            oep: if verified_noop {
                CleanOepReport {
                    protected_header_entry: pe.optional.address_of_entry_point,
                    recovered_entry: Some(pe.optional.address_of_entry_point),
                    complete: true,
                }
            } else {
                CleanOepReport {
                    protected_header_entry: pe.optional.address_of_entry_point,
                    recovered_entry: oep_decision.recovered_entry,
                    complete: oep_decision.recovered_entry.is_some(),
                }
            },
            imports: CleanImportReport {
                parsed_modules: report.imports.len(),
                parsed_symbols: import_symbols,
                reconstruction_complete: verified_noop
                    || import_proof.as_ref().is_some_and(|proof| proof.complete),
                proven_modules: import_proof
                    .as_ref()
                    .map_or(0, |proof| proof.proven_modules),
                proven_symbols: import_proof
                    .as_ref()
                    .map_or(0, |proof| proof.proven_symbols),
                unresolved: if verified_noop
                    || import_proof.as_ref().is_some_and(|proof| proof.complete)
                {
                    Vec::new()
                } else {
                    vec![
                    "parsed protector-era imports are not yet reduced to proven application imports"
                        .to_owned(),
                ]
                },
                observed: observed_import_report,
            },
            vm_instances: discovery.instances,
            vm_entries: discovery.entries,
            vm_inventory,
            input_closed_functions,
            handlers,
            unknown_handlers,
            virtualized_routines: paths,
            tls_entries,
            exports: export_reports,
            native_continuations: continuations,
            vtil: if verified_noop {
                completed_stage("not required for an input with no strong protection evidence")
            } else {
                stage("semantic routine has not reached a complete exception-aware CFG")
            },
            recompilation,
            reintegration,
            coverage,
            anti_debug,
            reconstruction: if verified_noop {
                completed_stage("conventional input can be preserved byte-for-byte")
            } else {
                stage(
                    "a conventional layout image can be written, but no devirtualized routine is eligible for native replacement",
                )
            },
            layout,
            validation: CleanValidationReport {
                input_parsed: true,
                mapped_image_parsed: true,
                output_written: false,
                output_reparsed: false,
            },
            warnings,
            errors,
            ready_to_write: blockers.is_empty(),
            experimental: false,
            blockers,
        },
        conventional,
        reintegrated,
    ))
}

#[derive(Clone, Copy)]
enum GroupKind {
    Header,
    Tls(Rva),
}

/// Builds one replacement body, resolving its continuation target and import
/// calls from `pe`. Shared by every reintegration group and the
/// nested-continuation pass so they all use identical codegen.
fn compile_replacement_body(
    pe: &PeImage<'_>,
    inputs: &[(String, vmpx_recompile::PhysicalRegister)],
    symbol_rewrites: &BTreeMap<String, vmpx_vm::Expr>,
    exit: &vmpx_vm::NativeExitState,
    effects: &vmpx_vmlift::CollectedEffects,
    va: u64,
) -> Result<Vec<u8>, String> {
    // The VM exit transfers to a protector continuation rather than returning
    // to the caller. Resolve that continuation so the replacement can
    // tail-jump into it after restoring the exit state, which is what makes
    // it a drop-in for the VM entry.
    let continuation = vmpx_vm::recover_modulo_image_targets(
        pe,
        &exit.return_address,
        vmpx_vm::ModuloImageTargetLimits::default(),
    )
    .ok()
    .filter(|report| report.complete && report.targets.len() == 1)
    .map(|report| report.targets[0].va)
    .map(|va| va.0);
    let config = vmpx_vmlift::ReplacementConfig {
        inputs: inputs.to_vec(),
        max_expression_nodes: 200_000,
        max_spill_slots: 64,
        import_calls: import_calls_for(pe, exit),
        continuation,
        effects: effects
            .stores
            .iter()
            .map(|store| vmpx_vmlift::EffectStore {
                address: store.address.clone(),
                value: store.value.clone(),
                width: store.width,
            })
            .collect(),
        rflags: Some(exit.rflags.clone()),
        symbol_rewrites: symbol_rewrites.clone(),
        effect_calls: effects
            .calls
            .iter()
            .filter_map(|call| {
                let iat_va = pe
                    .optional
                    .image_base
                    .checked_add_rva(vmpx_pe::Rva(call.iat_rva))?;
                Some(vmpx_vmlift::EffectCall {
                    result_intrinsic: call.result_intrinsic.clone(),
                    call: vmpx_vmlift::ImportCall {
                        iat_address: iat_va.0,
                        argument_count: u8::try_from(call.arguments.len()).ok()?,
                    },
                    arguments: call.arguments.clone(),
                })
            })
            .collect(),
    };
    vmpx_vmlift::build_replacement(&exit.registers, &config, vmpx_recompile::CodeAddress(va))
        .map(|replacement| replacement.compiled.bytes)
        .map_err(|error| error.to_string())
}

/// Builds one recovered single-body function's replacement with its OWN entry
/// inputs (never the header entry's) and its carried effects.
fn compile_single_body(
    pe: &PeImage<'_>,
    function: &crate::input_closed_functions::InputClosedFunction,
    va: u64,
) -> Result<Vec<u8>, String> {
    let candidate = analyze_vm_entry_at(pe, function.rva, VmDiscoveryLimits::default())
        .ok_or_else(|| "no VM entry stub at this RVA".to_owned())?;
    let mut rewrites = BTreeMap::new();
    let inputs = replacement_inputs(pe, &candidate, &mut rewrites)?;
    compile_replacement_body(
        pe,
        &inputs,
        &rewrites,
        &function.exit,
        &function.effects,
        va,
    )
}

/// Registers a Win64 callee may clobber. The dispatcher saves all of them so the
/// chosen body, or the fallback stub, starts with the entry register state.
const DISPATCH_SAVED: [vmpx_recompile::PhysicalRegister; 7] = {
    use vmpx_recompile::PhysicalRegister as P;
    [P::Rax, P::Rcx, P::Rdx, P::R8, P::R9, P::R10, P::R11]
};

/// Compiles `branch_target` into a callable that returns it in `RAX`, preserving
/// nonvolatile registers, via the ordinary replacement lowering (the target is
/// the only register value). Refused unless it is a 64-bit expression over this
/// function's own entry inputs.
fn compile_branch_target(
    branch_target: &vmpx_vm::Expr,
    inputs: &[(String, vmpx_recompile::PhysicalRegister)],
    va: u64,
) -> Result<Vec<u8>, String> {
    if branch_target.width != BitWidth::W64 {
        return Err("branch target is not a 64-bit value".to_owned());
    }
    let mut selected = Vec::new();
    for name in branch_target.dependencies() {
        let input = inputs
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .ok_or_else(|| format!("branch target reads {name}, which is not an entry input"))?;
        selected.push(input.clone());
    }
    let config = vmpx_vmlift::ReplacementConfig {
        inputs: selected,
        max_expression_nodes: 200_000,
        max_spill_slots: 64,
        import_calls: BTreeMap::new(),
        continuation: None,
        effects: Vec::new(),
        effect_calls: Vec::new(),
        rflags: None,
        symbol_rewrites: BTreeMap::new(),
    };
    let registers = BTreeMap::from([(vmpx_vm::X64NativeRegister::Rax, branch_target.clone())]);
    vmpx_vmlift::build_replacement(&registers, &config, vmpx_recompile::CodeAddress(va))
        .map(|replacement| replacement.compiled.bytes)
        .map_err(|error| error.to_string())
}

/// Builds `[body_i]..[branch-target callable][displaced stub][dispatcher]` for a
/// recovered branching function, laid out from `va`. Returns the bytes and the
/// dispatcher's offset (the stub's redirect target). Nothing is partially
/// returned: any failure yields `Err`.
fn compile_branching_function(
    pe: &PeImage<'_>,
    image: &[u8],
    function: &crate::input_closed_functions::BranchingFunction,
    va: u64,
) -> Result<(Vec<u8>, u32), String> {
    if !function.domain_complete {
        return Err("path family domain is not proven complete".to_owned());
    }
    if function.paths.is_empty() {
        return Err("no recovered paths".to_owned());
    }
    let candidate = analyze_vm_entry_at(pe, function.rva, VmDiscoveryLimits::default())
        .ok_or_else(|| "no VM entry stub at this RVA".to_owned())?;
    let mut rewrites = BTreeMap::new();
    let inputs = replacement_inputs(pe, &candidate, &mut rewrites)?;
    let address = |bytes: &[u8]| va.saturating_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX));

    let mut out: Vec<u8> = Vec::new();
    let mut arms = Vec::with_capacity(function.paths.len());
    for path in &function.paths {
        let body_va = address(&out);
        let body =
            compile_replacement_body(pe, &inputs, &rewrites, &path.exit, &path.effects, body_va)
                .map_err(|error| format!("path for target {:#x}: {error}", path.target))?;
        arms.push(vmpx_recompile::DispatchArm {
            value: path.target,
            body: body_va,
        });
        out.extend(body);
    }
    let compute_va = address(&out);
    out.extend(compile_branch_target(
        &function.branch_target,
        &inputs,
        compute_va,
    )?);

    let stub_va = pe
        .optional
        .image_base
        .checked_add_rva(function.rva)
        .ok_or("stub VA overflow")?
        .0;
    let image_pe = PeImage::parse(image).map_err(|error| error.to_string())?;
    let stub_offset = image_pe
        .rva_to_file_offset(function.rva)
        .map_err(|error| error.to_string())?;
    let stub_start = usize::try_from(stub_offset.0).map_err(|_| "stub offset overflow")?;
    let stub_bytes = image.get(stub_start..).ok_or("stub is outside the image")?;
    let displaced_va = address(&out);
    out.extend(
        vmpx_recompile::emit_displaced_stub(
            &stub_bytes[..stub_bytes.len().min(32)],
            stub_va,
            5,
            displaced_va,
        )
        .map_err(|error| error.to_string())?,
    );

    let dispatcher_offset = u32::try_from(out.len()).map_err(|_| "layout too large")?;
    let dispatcher_va = address(&out);
    let dispatcher = vmpx_recompile::emit_value_dispatcher(
        &vmpx_recompile::ValueDispatch {
            value: vmpx_recompile::PhysicalRegister::Rax,
            scratch: vmpx_recompile::PhysicalRegister::Rcx,
            saved: &DISPATCH_SAVED,
            compute: compute_va,
            arms: &arms,
            fallback: displaced_va,
        },
        dispatcher_va,
    )
    .map_err(|error| error.to_string())?;
    out.extend(dispatcher);
    Ok((out, dispatcher_offset))
}

/// Resolves a TLS entry's RVA back to its position in the TLS directory's
/// callback array, which is the index `set_tls_callback`/`reintegrate` need.
fn tls_callback_index(pe: &PeImage<'_>, entry_rva: Rva) -> Option<usize> {
    let tls = pe.report().tls?;
    tls.callbacks
        .iter()
        .position(|va| pe.optional.image_base.checked_rva_of(*va) == Some(entry_rva))
}

// Returns (payload, entry_dispatcher_rva, per-entry dispatcher RVAs). Bodies are
// placed first at `base`, dispatchers after, so body VAs stay as compiled.
fn layout_reintegration_payload(
    base: Rva,
    image_base: u64,
    bodies: &[(u64, Vec<u8>)],
    domain_reg: vmpx_recompile::PhysicalRegister,
    original_entry_rva: Rva,
) -> Result<(Vec<u8>, Rva), String> {
    let mut payload = Vec::new();
    let mut arms = Vec::new();
    let mut offset = 0u32;
    for (value, bytes) in bodies {
        let body_rva = base.0.checked_add(offset).ok_or("body rva overflow")?;
        arms.push(vmpx_recompile::DispatchArm {
            value: *value,
            body: image_base + u64::from(body_rva),
        });
        payload.extend_from_slice(bytes);
        offset = offset
            .checked_add(u32::try_from(bytes.len()).map_err(|_| "body too large")?)
            .ok_or("offset overflow")?;
    }
    let dispatcher_rva = Rva(base
        .0
        .checked_add(offset)
        .ok_or("dispatcher rva overflow")?);
    let dispatcher = vmpx_recompile::emit_domain_dispatcher(
        domain_reg,
        &arms,
        image_base + u64::from(original_entry_rva.0),
        image_base + u64::from(dispatcher_rva.0),
    )
    .map_err(|e| e.to_string())?;
    payload.extend_from_slice(&dispatcher);
    Ok((payload, dispatcher_rva))
}

/// The `DllMain` reason values (`native_rdx_initial`) the entry paths are enumerated over.
const NATIVE_RDX_DOMAIN: std::ops::RangeInclusive<u64> = 0..=3;

/// Reason-domain assignments and limits used to recover a DLL entry's paths.
pub(crate) fn dll_entry_recovery_inputs() -> (BTreeMap<String, Vec<Expr>>, EntryPathRecoveryLimits)
{
    let domains = BTreeMap::from([(
        "native_rdx_initial".to_owned(),
        NATIVE_RDX_DOMAIN
            .map(|reason| Expr::constant(BitWidth::W64, reason))
            .collect(),
    )]);
    let limits = EntryPathRecoveryLimits {
        chain: VmProfileExplorerLimits {
            max_segments: 32,
            max_instructions_per_segment: 2_048,
            max_total_instructions: 8_192,
            ..VmProfileExplorerLimits::default()
        },
        ..EntryPathRecoveryLimits::default()
    };
    (domains, limits)
}

/// The continuation RVA when the report resolves exactly one concrete target.
pub(crate) fn single_continuation_rva(report: &vmpx_vm::ModuloImageTargetReport) -> Option<Rva> {
    if report.complete && report.targets.len() == 1 {
        report.targets[0].rva
    } else {
        None
    }
}

/// A continuation is native when it lies in an executable non-`.vmp` section and
/// is neither a VM entry stub nor the protected header entry itself.
pub(crate) fn is_native_continuation(pe: &PeImage<'_>, rva: Rva) -> bool {
    let in_native_section = pe.sections.iter().any(|section| {
        section.contains_mapped_rva(rva)
            && section.characteristics & IMAGE_SCN_MEM_EXECUTE != 0
            && !section.name.to_ascii_lowercase().starts_with(".vmp")
    });
    in_native_section
        && rva != pe.optional.address_of_entry_point
        && !analyze_vm_entry_at(pe, rva, VmDiscoveryLimits::default())
            .is_some_and(|candidate| candidate.confidence >= 60)
}

/// Recovers a DLL header entry's paths: whether the family is proven complete,
/// and each path's single-resolved continuation RVA (None when unresolved).
pub(crate) fn header_entry_continuations(
    pe: &PeImage<'_>,
    entry: &vmpx_vm::VmEntryCandidate,
) -> Result<(bool, Vec<Option<Rva>>), String> {
    let (domains, limits) = dll_entry_recovery_inputs();
    let family = recover_entry_paths(pe, entry, &domains, limits).map_err(|e| e.to_string())?;
    let paths = family
        .paths
        .iter()
        .map(|path| {
            let exit = path.native_exit.as_ref()?;
            let report = recover_modulo_image_targets(
                pe,
                &exit.return_address,
                ModuloImageTargetLimits::default(),
            )
            .ok()?;
            single_continuation_rva(&report)
        })
        .collect();
    Ok((
        family.domain_complete && family.all_paths_reach_vm_exit,
        paths,
    ))
}

/// A lone body may replace the entry outright only when the family is proven
/// complete and that body serves every domain value; otherwise the remaining
/// values must fall back to the original entry through a dispatcher.
fn needs_dispatcher(
    bodies: &[(u64, Vec<u8>)],
    family_complete: bool,
    mut domain: std::ops::RangeInclusive<u64>,
) -> bool {
    let covers = family_complete && domain.all(|value| bodies.iter().any(|(v, _)| *v == value));
    bodies.len() > 1 || !covers
}

/// Lays out one group's already-compiled bodies onto `payload`, starting at
/// `*cursor` (an offset from `planned_rva`). On success, advances `*cursor`
/// past the group's dispatcher (if any) and appends the group's bytes. On a
/// dispatcher-assembly failure, `*cursor` is rolled back to
/// `group_start_cursor` and `payload` is left untouched, so the failed group
/// contributes nothing and the next group's base is unaffected by it.
fn commit_group_layout(
    cursor: &mut u32,
    payload: &mut Vec<u8>,
    group_start_cursor: u32,
    planned_rva: Rva,
    image_base: u64,
    group_bodies: &[(u64, Vec<u8>)],
    fallback: Rva,
    family_complete: bool,
) -> Result<(Rva, bool), String> {
    let group_base = Rva(planned_rva
        .0
        .checked_add(group_start_cursor)
        .unwrap_or(planned_rva.0));
    let dispatcher_used = needs_dispatcher(group_bodies, family_complete, NATIVE_RDX_DOMAIN);
    let built = if dispatcher_used {
        layout_reintegration_payload(
            group_base,
            image_base,
            group_bodies,
            vmpx_recompile::PhysicalRegister::Rdx,
            fallback,
        )
    } else {
        Ok((group_bodies[0].1.clone(), group_base))
    };
    match built {
        Ok((group_payload, redirect)) => {
            let payload_len = u32::try_from(group_payload.len()).unwrap_or(0);
            // `group_payload` repeats the already-counted body bytes plus,
            // for a dispatcher group, the dispatcher itself; only the
            // dispatcher part still needs to advance the cursor.
            let already_counted = cursor.saturating_sub(group_start_cursor);
            *cursor = cursor.saturating_add(payload_len.saturating_sub(already_counted));
            payload.extend_from_slice(&group_payload);
            Ok((redirect, dispatcher_used))
        }
        Err(error) => {
            *cursor = group_start_cursor;
            Err(error)
        }
    }
}

#[cfg(test)]
mod reintegration_tests {
    use super::*;
    use vmpx_pe::Rva;
    use vmpx_recompile::PhysicalRegister;

    fn rax_input() -> Vec<(String, PhysicalRegister)> {
        vec![("native_rax_initial".to_owned(), PhysicalRegister::Rax)]
    }

    #[test]
    fn branch_target_over_an_entry_input_compiles_to_a_callable() {
        let target = vmpx_vm::Expr::binary(
            vmpx_vm::BinaryOp::Add,
            vmpx_vm::Expr::register(BitWidth::W64, "native_rax_initial"),
            vmpx_vm::Expr::constant(BitWidth::W64, 5),
        )
        .expect("expr");
        let bytes = compile_branch_target(&target, &rax_input(), 0x1_4000_2000).expect("compile");
        assert_eq!(bytes.last().copied(), Some(0xC3));
    }

    #[test]
    fn branch_target_over_a_non_input_symbol_is_deferred() {
        let target = vmpx_vm::Expr::register(BitWidth::W64, "native_rbx_initial");
        let error = compile_branch_target(&target, &rax_input(), 0x1_4000_2000)
            .expect_err("rbx is not bound");
        assert!(error.contains("not an entry input"));
    }

    #[test]
    fn narrow_branch_target_is_deferred() {
        let target = vmpx_vm::Expr::register(BitWidth::W32, "native_rax_initial");
        assert!(compile_branch_target(&target, &rax_input(), 0x1_4000_2000).is_err());
    }

    #[test]
    fn dispatcher_follows_bodies_and_targets_each_body() {
        let base = Rva(0x2000);
        let image_base = 0x1_4000_0000;
        let bodies = vec![(0u64, vec![0x90, 0xC3]), (1u64, vec![0x90, 0x90, 0xC3])];
        let (payload, dispatcher_rva) = layout_reintegration_payload(
            base,
            image_base,
            &bodies,
            PhysicalRegister::Rdx,
            Rva(0x1000),
        )
        .expect("layout");
        // bodies total 5 bytes, so the dispatcher starts at base + 5.
        assert_eq!(dispatcher_rva, Rva(0x2005));
        assert_eq!(&payload[..5], &[0x90, 0xC3, 0x90, 0x90, 0xC3]);
        assert!(payload.len() > 5); // dispatcher bytes appended
    }

    #[test]
    fn a_second_group_lays_out_after_the_first_groups_full_payload() {
        let image_base = 0x1_4000_0000;
        let header_base = Rva(0x2000);
        let header_bodies = vec![(0u64, vec![0x90, 0xC3]), (1u64, vec![0x90, 0x90, 0xC3])];
        let (header_payload, header_dispatcher_rva) = layout_reintegration_payload(
            header_base,
            image_base,
            &header_bodies,
            PhysicalRegister::Rdx,
            Rva(0x1000),
        )
        .expect("header layout");

        // The next group's base, as clean.rs computes it, is the running
        // cursor right after the header group's full payload (bodies plus its
        // dispatcher) — not just after the header bodies.
        let tls_base = Rva(header_base.0 + u32::try_from(header_payload.len()).expect("fits"));
        let tls_bodies = vec![(0u64, vec![0xC3]), (2u64, vec![0x90, 0xC3])];
        let (_tls_payload, tls_dispatcher_rva) = layout_reintegration_payload(
            tls_base,
            image_base,
            &tls_bodies,
            PhysicalRegister::Rdx,
            Rva(0x3000),
        )
        .expect("tls layout");

        assert!(
            tls_base.0 > header_dispatcher_rva.0,
            "tls group must start after the header dispatcher"
        );
        // The TLS group's own dispatcher sits after only its own bodies (1 + 2
        // bytes), not after the header group's bodies too.
        assert_eq!(tls_dispatcher_rva, Rva(tls_base.0 + 3));
    }

    #[test]
    fn single_body_without_a_complete_domain_gets_a_fallback_dispatcher() {
        let image_base = 0x1_4000_0000;
        let planned_rva = Rva(0x2000);
        let bodies = vec![(1u64, vec![0x90, 0xC3])];

        // Only reason 1 was recovered, so reasons 0/2/3 must reach the original entry.
        let mut cursor = 2u32;
        let mut payload = Vec::new();
        let (redirect, dispatcher_used) = commit_group_layout(
            &mut cursor,
            &mut payload,
            0,
            planned_rva,
            image_base,
            &bodies,
            Rva(0x1000),
            true,
        )
        .expect("layout");
        assert!(
            dispatcher_used,
            "a partial domain needs the fallback dispatcher"
        );
        assert_eq!(
            redirect,
            Rva(planned_rva.0 + 2),
            "redirect targets the dispatcher"
        );
        assert!(payload.len() > 2, "dispatcher bytes follow the body");

        // An unproven family counts as incomplete even when the values line up.
        let full: Vec<(u64, Vec<u8>)> = NATIVE_RDX_DOMAIN.map(|v| (v, vec![0xC3])).collect();
        let mut cursor = 4u32;
        let mut payload = Vec::new();
        let (_, used) = commit_group_layout(
            &mut cursor,
            &mut payload,
            0,
            planned_rva,
            image_base,
            &full,
            Rva(0x1000),
            false,
        )
        .expect("layout");
        assert!(used);
    }

    #[test]
    fn single_body_redirects_directly_only_when_the_domain_is_provably_complete() {
        let lone = vec![(1u64, vec![0xC3])];
        assert!(
            !needs_dispatcher(&lone, true, 1..=1),
            "complete one-value domain"
        );
        assert!(needs_dispatcher(&lone, false, 1..=1), "unproven family");
        assert!(
            needs_dispatcher(&lone, true, 0..=3),
            "body covers only part of the domain"
        );
        let pair = vec![(0u64, vec![0xC3]), (1u64, vec![0xC3])];
        assert!(
            needs_dispatcher(&pair, true, 0..=1),
            "multi-body always dispatches"
        );
    }

    #[test]
    fn a_failed_group_leaves_no_gap_for_the_next_group() {
        let image_base = 0x1_4000_0000;
        let planned_rva = Rva(0x2000);
        let mut cursor = 0u32;
        let mut payload = Vec::new();

        // Group 1 has a domain value too wide for the dispatcher encoder, so
        // its dispatcher assembly fails. Its bodies (5 bytes) are "compiled"
        // first, advancing the cursor exactly as the per-item compile loop in
        // clean.rs does before calling `commit_group_layout`.
        let group1_start = cursor;
        let group1_bodies = vec![
            (0u64, vec![0x90, 0xC3]),
            (u64::from(u32::MAX) + 1, vec![0x90, 0x90, 0xC3]),
        ];
        cursor += 5;
        let failed = commit_group_layout(
            &mut cursor,
            &mut payload,
            group1_start,
            planned_rva,
            image_base,
            &group1_bodies,
            Rva(0x1000),
            true,
        );
        assert!(
            failed.is_err(),
            "a too-wide domain value must fail dispatcher assembly"
        );
        assert_eq!(
            cursor, group1_start,
            "a failed group must roll the cursor back"
        );
        assert!(
            payload.is_empty(),
            "a failed group must not contribute any bytes"
        );

        // Group 2 "compiles" as if group 1 had never been attempted.
        let group2_start = cursor;
        let group2_bodies = vec![(0u64, vec![0xC3]), (1u64, vec![0x90, 0xC3])];
        cursor += 3;
        let (redirect, dispatcher_used) = commit_group_layout(
            &mut cursor,
            &mut payload,
            group2_start,
            planned_rva,
            image_base,
            &group2_bodies,
            Rva(0x3000),
            true,
        )
        .expect("group 2 layout");

        assert!(dispatcher_used);
        assert_eq!(
            group2_start, group1_start,
            "group 2 must start exactly where group 1 would have"
        );
        assert_eq!(redirect, Rva(planned_rva.0 + group2_start + 3));
        // The final payload holds only group 2's bytes: no gap and no
        // phantom bytes left behind by the failed group 1.
        assert_eq!(u32::try_from(payload.len()).unwrap(), cursor - group1_start);
    }
}
