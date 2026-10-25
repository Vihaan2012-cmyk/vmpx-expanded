use std::collections::{BTreeMap, BTreeSet, VecDeque};

use anyhow::{Context, Result};
use serde::Serialize;
use vmpx_detect::{DetectionVerdict, ImageLayout, ProtectionReport, detect_vmprotect};
use vmpx_disasm::{CfgLimits, recover_cfg};
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
    pub vm_references: usize,
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
                        vm_references += 1;
                    } else if !seen.contains(successor) {
                        queue.push_back(*successor);
                    }
                }
                for call in &block.direct_calls {
                    if is_protector(*call) {
                        vm_references += 1;
                    } else if !seen.contains(call) {
                        queue.push_back(*call);
                    }
                }
            }
        }
        results.push(CleanExportReport {
            name: symbol
                .name
                .clone()
                .unwrap_or_else(|| format!("ordinal{}", symbol.ordinal)),
            rva: symbol.rva,
            reachable_blocks: seen.len(),
            vm_references,
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
fn replacement_inputs() -> Vec<(String, vmpx_recompile::PhysicalRegister)> {
    use vmpx_recompile::PhysicalRegister;
    use vmpx_vm::X64NativeRegister;
    let mut inputs: Vec<(String, PhysicalRegister)> = X64NativeRegister::ALL
        .iter()
        .map(|register| (register.symbolic_name().to_owned(), physical_for(register)))
        .collect();
    // The VM stack root and the original return address have no architectural
    // register of their own; the frame pointer carries both.
    inputs.push(("vsp_initial".to_owned(), PhysicalRegister::Rbp));
    inputs.push((
        "native_return_address_initial".to_owned(),
        PhysicalRegister::Rbp,
    ));
    inputs.sort();
    inputs.dedup();
    inputs
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
    analyze_clean_with_image(input_path, input).map(|(report, _)| report)
}

/// Runs the same analysis and additionally returns the conventional-layout
/// image when the mapped analysis image can be converted. The image is a
/// layout artifact: it is loadable and structurally valid, but still contains
/// every protector section and virtualized routine.
///
/// # Errors
///
/// Propagates input, unpacking, VM-analysis, and exception-metadata failures.
#[allow(clippy::too_many_lines)]
pub fn analyze_clean_with_image(
    input_path: &str,
    input: &[u8],
) -> Result<(CleanReport, Option<Vec<u8>>)> {
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
    let mut exit_states: Vec<(BTreeMap<String, u64>, vmpx_vm::NativeExitState)> = Vec::new();
    // Nested VM entries reached through a recovered continuation. Each one is
    // recovered recursively so a path that re-enters the VM still terminates in
    // a native exit state rather than stopping at a protector transition.
    let mut nested_exits: Vec<(BTreeMap<String, u64>, vmpx_vm::NativeExitState)> = Vec::new();
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
    if let Some(entry) = header_entry {
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
                            exit_states.push((path.assignments.clone(), exit.clone()));
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
                                let nested_domains = BTreeMap::from([(
                                    "native_rdx_initial".to_owned(),
                                    (0_u64..=3)
                                        .map(|reason| Expr::constant(BitWidth::W64, reason))
                                        .collect::<Vec<_>>(),
                                )]);
                                if let Ok(nested_family) =
                                    recover_entry_paths(&pe, &nested, &nested_domains, limits)
                                {
                                    for nested_path in &nested_family.paths {
                                        if let Some(nested_exit) = nested_path.native_exit.as_ref()
                                        {
                                            let mut assignments = path.assignments.clone();
                                            assignments.extend(nested_path.assignments.clone());
                                            nested_exits.push((assignments, nested_exit.clone()));
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
    // Native recompilation: emit a replacement function for every path whose
    // exit state is fully concrete and lowerable. Emission does not authorise
    // writing output, because the entry redirect and the remaining blockers are
    // still unproven.
    let mut recompilation = stage("no path reached a lowerable exit state");
    if protection.verdict != DetectionVerdict::NoStrongEvidence {
        let next_rva = vmpx_rebuild::next_section_rva(&mapped).ok();
        let mut emitted = 0_u64;
        let mut total_bytes = 0_u64;
        let mut emitted_section = None;
        let mut diagnostics = Vec::new();
        if let Some(next_rva) = next_rva {
            let va = pe.optional.image_base.0 + u64::from(next_rva.0);
            let inputs = replacement_inputs();
            let mut functions: Vec<(BTreeMap<String, u64>, Vec<u8>)> = Vec::new();
            for (assignments, exit) in exit_states.iter().chain(nested_exits.iter()) {
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
                };
                match vmpx_vmlift::build_replacement(
                    &exit.registers,
                    &config,
                    vmpx_recompile::CodeAddress(va + total_bytes),
                ) {
                    Ok(replacement) => {
                        total_bytes += u64::try_from(replacement.compiled.bytes.len()).unwrap_or(0);
                        functions.push((assignments.clone(), replacement.compiled.bytes.clone()));
                        emitted += 1;
                    }
                    Err(error) => diagnostics.push(format!(
                        "path {assignments:?} has no native replacement: {error}"
                    )),
                }
            }
            if !functions.is_empty() {
                emitted_section = Some(functions);
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
                    diagnostics.push(format!(
                        "{emitted} native replacement function(s) emitted; the VM entry is not redirected because the remaining proofs are open"
                    ));
                    diagnostics
                },
            };
        }
        let _ = emitted_section;
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
    ))
}
