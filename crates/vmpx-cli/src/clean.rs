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
        let mut vm_references = 0_usize;
        while let Some(start) = queue.pop_front() {
            if !seen.insert(start) {
                continue;
            }
            if is_protector(start) {
                vm_references += 1;
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
        vm_references = vm_entries.len();
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
                (0_u64..=3)
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
    let mut tls_exit_states: Vec<(
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
            (0_u64..=3)
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
    let planning_image = if protection.verdict == DetectionVerdict::NoStrongEvidence {
        None
    } else {
        relayout_conventional(&mapped).ok().map(|(bytes, _)| bytes)
    };
    // Native recompilation: emit a replacement function for every path whose
    // exit state is fully concrete and lowerable. Emission does not authorise
    // writing output, because the entry redirect and the remaining blockers are
    // still unproven.
    let mut recompilation = stage("no path reached a lowerable exit state");
    let mut reintegration = stage("native code reintegration was not attempted");
    let mut reintegrated: Option<Vec<u8>> = None;
    if protection.verdict != DetectionVerdict::NoStrongEvidence {
        let planned = planning_image.as_deref().and_then(|bytes| {
            vmpx_rebuild::next_section_rva(bytes)
                .ok()
                .map(|rva| (bytes, rva))
        });
        let mut emitted = 0_u64;
        let mut total_bytes = 0_u64;
        let mut placed: Vec<(BTreeMap<String, u64>, Rva)> = Vec::new();
        let mut diagnostics = Vec::new();
        if let Some((image, planned_rva)) = planned {
            let va = pe.optional.image_base.0 + u64::from(planned_rva.0);
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
            // Header-entry bodies are kept apart so they can be laid out as one
            // dispatcher group below; nested and TLS bodies still compile and
            // count here but are not yet wired into a redirect.
            let mut header_bodies: Vec<(u64, Vec<u8>)> = Vec::new();
            let header_tagged = exit_states.iter().map(|(a, e, f)| (true, a, e, f));
            let nested_tagged = nested_exits.iter().map(|(a, e, f)| (false, a, e, f));
            let tls_tagged = tls_exit_states.iter().map(|(a, e, f)| (false, a, e, f));
            for (is_header, assignments, exit, effects) in
                header_tagged.chain(nested_tagged).chain(tls_tagged)
            {
                // The VM exit transfers to a protector continuation rather than
                // returning to the caller. Resolve that continuation so the
                // replacement can tail-jump into it after restoring the exit
                // state, which is what makes it a drop-in for the VM entry.
                let continuation = vmpx_vm::recover_modulo_image_targets(
                    &pe,
                    &exit.return_address,
                    vmpx_vm::ModuloImageTargetLimits::default(),
                )
                .ok()
                .filter(|report| report.complete && report.targets.len() == 1)
                .map(|report| report.targets[0].va)
                .map(|va| va.0);
                let config = vmpx_vmlift::ReplacementConfig {
                    inputs: inputs.clone(),
                    max_expression_nodes: 200_000,
                    max_spill_slots: 64,
                    import_calls: import_calls_for(&pe, exit),
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
                match vmpx_vmlift::build_replacement(
                    &exit.registers,
                    &config,
                    vmpx_recompile::CodeAddress(va + total_bytes),
                ) {
                    Ok(replacement) => {
                        // Record where this body landed so a redirect can target
                        // it: the entry stub has to jump somewhere specific.
                        let entry_rva = vmpx_pe::Rva(
                            u32::try_from(planned_rva.0)
                                .ok()
                                .and_then(|base| {
                                    u32::try_from(total_bytes)
                                        .ok()
                                        .and_then(|offset| base.checked_add(offset))
                                })
                                .unwrap_or(planned_rva.0),
                        );
                        placed.push((assignments.clone(), entry_rva));
                        let body = replacement.compiled.bytes;
                        total_bytes += u64::try_from(body.len()).unwrap_or(0);
                        if is_header
                            && let Some(&domain_value) = assignments.get("native_rdx_initial")
                        {
                            header_bodies.push((domain_value, body));
                        }
                        emitted += 1;
                    }
                    Err(error) => diagnostics.push(format!(
                        "path {assignments:?} has no native replacement: {error}"
                    )),
                }
            }
            if !header_bodies.is_empty() {
                let dispatcher_used = header_bodies.len() > 1;
                let built = if dispatcher_used {
                    layout_reintegration_payload(
                        planned_rva,
                        pe.optional.image_base.0,
                        &header_bodies,
                        vmpx_recompile::PhysicalRegister::Rdx,
                        pe.optional.address_of_entry_point,
                    )
                } else {
                    Ok((header_bodies[0].1.clone(), planned_rva))
                };
                reintegration = match built {
                    Ok((payload, redirect)) => match vmpx_rebuild::reintegrate(
                        image,
                        vmpx_rebuild::ReintegrationRequest {
                            section_name: *b".vmpx0\0\0",
                            payload: &payload,
                            entry_redirect: Some(redirect),
                            tls_redirects: &[],
                        },
                    ) {
                        Ok((bytes, report)) => {
                            reintegrated = Some(bytes);
                            CleanStageReport {
                                complete: true,
                                metrics: BTreeMap::from([
                                    ("section_rva".to_owned(), u64::from(report.section.rva.0)),
                                    (
                                        "section_size".to_owned(),
                                        u64::from(report.section.virtual_size),
                                    ),
                                    ("entries_redirected".to_owned(), 1),
                                    ("dispatchers".to_owned(), u64::from(dispatcher_used)),
                                ]),
                                diagnostics: vec![
                                    "header entry redirected; TLS-callback and \
                                     nested-continuation redirects are not yet wired"
                                        .to_owned(),
                                ],
                            }
                        }
                        Err(error) => stage(format!("reintegrate failed: {error}")),
                    },
                    Err(error) => stage(format!("header dispatcher assembly failed: {error}")),
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
        match relayout_conventional(&mapped) {
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
            blockers,
        },
        conventional,
        reintegrated,
    ))
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
}
