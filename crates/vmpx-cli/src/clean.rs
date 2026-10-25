use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde::Serialize;
use vmpx_detect::{DetectionVerdict, ImageLayout, ProtectionReport, detect_vmprotect};
use vmpx_disasm::{CfgLimits, recover_cfg};
use vmpx_pe::{PeImage, PeKind, Rva};
use vmpx_rebuild::relayout_conventional;
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

    let protected_tls = discovery
        .entries
        .iter()
        .filter(|entry| entry.source == VmEntrySource::TlsCallback)
        .count();
    if protected_tls != 0 {
        blockers.push(format!(
            "{protected_tls} loader-reachable TLS VM entry/entries remain protected"
        ));
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
                diagnostics: vec![
                    "input is already a conventional PE file; no relayout was required".to_owned(),
                ],
            },
            None,
        )
    } else {
        match relayout_conventional(&mapped) {
            Ok((bytes, report)) => (
                CleanLayoutReport {
                    attempted: true,
                    file_size: Some(report.file_size),
                    size_of_headers: Some(report.size_of_headers),
                    sections: Some(report.sections.len()),
                    zero_fill_sections: Some(report.zero_fill_sections),
                    diagnostics: vec![
                        "conventional layout preserves every section RVA; it is not devirtualized"
                            .to_owned(),
                    ],
                },
                Some(bytes),
            ),
            Err(error) => {
                errors.push(format!("conventional layout failed: {error}"));
                (
                    CleanLayoutReport {
                        attempted: true,
                        file_size: None,
                        size_of_headers: None,
                        sections: None,
                        zero_fill_sections: None,
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
            native_continuations: continuations,
            vtil: if verified_noop {
                completed_stage("not required for an input with no strong protection evidence")
            } else {
                stage("semantic routine has not reached a complete exception-aware CFG")
            },
            recompilation: if verified_noop {
                completed_stage("not required for a verified no-op")
            } else {
                stage("no complete routine is eligible for native code generation")
            },
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
