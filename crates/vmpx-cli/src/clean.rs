use std::collections::{BTreeMap, BTreeSet, VecDeque};

use anyhow::{Context, Result};
use serde::Serialize;
use vmpx_detect::{DetectionVerdict, ImageLayout, ProtectionReport, detect_vmprotect};
use vmpx_disasm::{CfgLimits, DirectJumpTraceLimits, recover_cfg};
use vmpx_pe::{PeImage, PeKind, Rva};
use vmpx_rebuild::{relayout_conventional, validate_for_loader};
use vmpx_unpack::{UnpackLimits, UnpackReport, unpack_to_mapped_image};
use vmpx_vm::{
    BitWidth, EntryPathRecoveryLimits, EntryProfileRecoveryLimits, Expr, ModuloImageTargetLimits,
    NativeContinuationLimits, NativeContinuationReport, RecoveryStatus, VmDiscoveryLimits,
    VmEntrySource, VmMemoryExceptionCoverageReport, VmProfileChainStop, VmProfileExplorerLimits,
    analyze_native_continuation, analyze_vm_entry_at,
    analyze_vm_memory_exception_coverage_with_state, continue_through_native_vm_entry,
    discover_vm_entries, recover_entry_paths, recover_modulo_image_targets,
};
use vmpx_vmlift::audit_entry_effect_liveness;

use crate::sha256;

const IMAGE_FILE_DLL: u16 = 0x2000;

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
    pub unresolved: Vec<String>,
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
    pub handlers: CleanHandlerMetrics,
    pub unknown_handlers: Vec<Rva>,
    pub virtualized_routines: Vec<CleanVmPathReport>,
    pub tls_entries: Vec<CleanTlsEntryReport>,
    pub exports: Vec<CleanExportReport>,
    pub native_continuations: Vec<CleanNestedContinuationReport>,
    pub vtil: CleanStageReport,
    pub recompilation: CleanStageReport,
    pub reintegration: CleanStageReport,
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
    let is_protector = |rva: Rva| {
        pe.sections
            .iter()
            .any(|section| section.name.starts_with(".vmp") && section.contains_mapped_rva(rva))
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
fn replacement_inputs(
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
    analyze_clean_with_image(input_path, input).map(|(report, _, _)| report)
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
    // Unknown counts as incomplete: a group only skips its dispatcher fallback
    // when its family was proven complete.
    let mut header_family_complete = false;
    if let Some(entry) = header_entry {
        header_entry_candidate = Some(entry.clone());
        if pe.coff.characteristics & IMAGE_FILE_DLL == 0 {
            blockers.push(
                "generic finite input-domain recovery for an executable entry is not implemented"
                    .to_owned(),
            );
        } else {
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
            let mut tls_redirects: Vec<(usize, Rva)> = Vec::new();
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
                            let entry_rva = Rva(
                                planned_rva.0.checked_add(layout_cursor).unwrap_or(planned_rva.0),
                            );
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
                            GroupKind::Header => header_redirect = Some(redirect),
                            GroupKind::Tls(rva) => match tls_callback_index(&pe, *rva) {
                                Some(index) => tls_redirects.push((index, redirect)),
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
                        reintegration_diagnostics.push(format!(
                            "{group_label} dispatcher assembly failed: {error}"
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
                match compile_replacement_body(&pe, &inputs, &symbol_rewrites, exit, effects, body_va)
                {
                    Ok(body) => {
                        // This RVA is past the real payload (nested bodies are
                        // never appended, only counted), so it is not a live
                        // placement; diagnostics should read it that way.
                        let entry_rva =
                            Rva(planned_rva.0.checked_add(layout_cursor).unwrap_or(planned_rva.0));
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
                    Ok((bytes, report)) => {
                        reintegrated = Some(bytes);
                        let entries_redirected = u64::from(report.entry_redirected)
                            + u64::try_from(report.tls_redirected).unwrap_or(0);
                        let mut diags = vec![format!(
                            "{entries_redirected} entry/entries redirected ({} header, {} TLS \
                             callback(s)); nested-continuation redirects remain deferred",
                            u64::from(report.entry_redirected),
                            report.tls_redirected
                        )];
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
    if verified_noop {
        warnings
            .push("input has no strong VMProtect evidence; clean is a verified no-op".to_owned());
    } else if protection.verdict == DetectionVerdict::NoStrongEvidence {
        blockers.push(
            "input has no strong VMProtect evidence but its layout is not a conventional PE file"
                .to_owned(),
        );
    } else {
        blockers.extend([
            "original module entry point is not recovered".to_owned(),
            "application import set is not proven independently reconstructed".to_owned(),
            "native code reintegration has not produced a devirtualized replacement image"
                .to_owned(),
            "VMProtect sections remain referenced by unresolved paths".to_owned(),
        ]);
    }
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
        match relayout.take().unwrap_or_else(|| relayout_conventional(&mapped)) {
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
            schema_version: 1,
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
                    recovered_entry: None,
                    complete: false,
                }
            },
            imports: CleanImportReport {
                parsed_modules: report.imports.len(),
                parsed_symbols: import_symbols,
                reconstruction_complete: verified_noop,
                unresolved: if verified_noop {
                    Vec::new()
                } else {
                    vec![
                    "parsed protector-era imports are not yet reduced to proven application imports"
                        .to_owned(),
                ]
                },
            },
            vm_instances: discovery.instances,
            vm_entries: discovery.entries,
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
    let dispatcher_rva = Rva(base.0.checked_add(offset).ok_or("dispatcher rva overflow")?);
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
    let group_base = Rva(
        planned_rva
            .0
            .checked_add(group_start_cursor)
            .unwrap_or(planned_rva.0),
    );
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

    #[test]
    fn dispatcher_follows_bodies_and_targets_each_body() {
        let base = Rva(0x2000);
        let image_base = 0x1_4000_0000;
        let bodies = vec![(0u64, vec![0x90, 0xC3]), (1u64, vec![0x90, 0x90, 0xC3])];
        let (payload, dispatcher_rva) = layout_reintegration_payload(
            base, image_base, &bodies, PhysicalRegister::Rdx, Rva(0x1000),
        ).expect("layout");
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
            header_base, image_base, &header_bodies, PhysicalRegister::Rdx, Rva(0x1000),
        ).expect("header layout");

        // The next group's base, as clean.rs computes it, is the running
        // cursor right after the header group's full payload (bodies plus its
        // dispatcher) — not just after the header bodies.
        let tls_base = Rva(header_base.0 + u32::try_from(header_payload.len()).expect("fits"));
        let tls_bodies = vec![(0u64, vec![0xC3]), (2u64, vec![0x90, 0xC3])];
        let (_tls_payload, tls_dispatcher_rva) = layout_reintegration_payload(
            tls_base, image_base, &tls_bodies, PhysicalRegister::Rdx, Rva(0x3000),
        ).expect("tls layout");

        assert!(tls_base.0 > header_dispatcher_rva.0, "tls group must start after the header dispatcher");
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
            &mut cursor, &mut payload, 0, planned_rva, image_base, &bodies, Rva(0x1000), true,
        ).expect("layout");
        assert!(dispatcher_used, "a partial domain needs the fallback dispatcher");
        assert_eq!(redirect, Rva(planned_rva.0 + 2), "redirect targets the dispatcher");
        assert!(payload.len() > 2, "dispatcher bytes follow the body");

        // An unproven family counts as incomplete even when the values line up.
        let full: Vec<(u64, Vec<u8>)> = NATIVE_RDX_DOMAIN.map(|v| (v, vec![0xC3])).collect();
        let mut cursor = 4u32;
        let mut payload = Vec::new();
        let (_, used) = commit_group_layout(
            &mut cursor, &mut payload, 0, planned_rva, image_base, &full, Rva(0x1000), false,
        ).expect("layout");
        assert!(used);
    }

    #[test]
    fn single_body_redirects_directly_only_when_the_domain_is_provably_complete() {
        let lone = vec![(1u64, vec![0xC3])];
        assert!(!needs_dispatcher(&lone, true, 1..=1), "complete one-value domain");
        assert!(needs_dispatcher(&lone, false, 1..=1), "unproven family");
        assert!(needs_dispatcher(&lone, true, 0..=3), "body covers only part of the domain");
        let pair = vec![(0u64, vec![0xC3]), (1u64, vec![0xC3])];
        assert!(needs_dispatcher(&pair, true, 0..=1), "multi-body always dispatches");
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
            &mut cursor, &mut payload, group1_start, planned_rva, image_base,
            &group1_bodies, Rva(0x1000), true,
        );
        assert!(failed.is_err(), "a too-wide domain value must fail dispatcher assembly");
        assert_eq!(cursor, group1_start, "a failed group must roll the cursor back");
        assert!(payload.is_empty(), "a failed group must not contribute any bytes");

        // Group 2 "compiles" as if group 1 had never been attempted.
        let group2_start = cursor;
        let group2_bodies = vec![(0u64, vec![0xC3]), (1u64, vec![0x90, 0xC3])];
        cursor += 3;
        let (redirect, dispatcher_used) = commit_group_layout(
            &mut cursor, &mut payload, group2_start, planned_rva, image_base,
            &group2_bodies, Rva(0x3000), true,
        ).expect("group 2 layout");

        assert!(dispatcher_used);
        assert_eq!(group2_start, group1_start, "group 2 must start exactly where group 1 would have");
        assert_eq!(redirect, Rva(planned_rva.0 + group2_start + 3));
        // The final payload holds only group 2's bytes: no gap and no
        // phantom bytes left behind by the failed group 1.
        assert_eq!(u32::try_from(payload.len()).unwrap(), cursor - group1_start);
    }
}
