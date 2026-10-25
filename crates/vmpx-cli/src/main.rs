use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use serde::Serialize;
use sha2::{Digest, Sha256};
use vmpx_detect::{ProtectionReport, detect_vmprotect};
use vmpx_disasm::{
    IpRelativeReference, find_ip_relative_references_resync,
    find_ip_relative_references_resync_in_ranges,
};
use vmpx_pe::{PeImage, PeReport};
use vmpx_unpack::{UnpackLimits, unpack_to_mapped_image};
use vmpx_vm::{VmDiscoveryLimits, VmDiscoveryReport, VmEntrySource, discover_vm_entries};

mod clean;

#[derive(Debug, Parser)]
#[command(
    name = "vmpx",
    version,
    about = "Evidence-driven VMProtect research framework"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the complete proof-gated recovery pipeline and emit a clean PE only when safe.
    Clean {
        input: PathBuf,
        output: PathBuf,
        /// Write the versioned recovery report even when the proof gate refuses output.
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Inspect a PE-family file by content, regardless of extension.
    Info {
        input: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Perform structural validation without executing the input.
    Verify {
        input: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Research-only structural comparison; never used as recovery input.
    Compare {
        left: PathBuf,
        right: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Recover VMProtect-packed sections into an explicit RVA-layout analysis image.
    Unpack {
        input: PathBuf,
        output: PathBuf,
        /// Write a versioned JSON unpacking report.
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Profile `VMProtect` evidence and packed/mapped layout.
    Detect {
        input: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Discover evidence-scored VM entry stubs and mutated bootstrap profiles.
    VmList {
        input: PathBuf,
        #[arg(long)]
        json: bool,
        /// Restrict discovery to PE entry and TLS roots.
        #[arg(long)]
        no_scan: bool,
    },
    /// Inventory imports and direct native RIP-relative IAT references.
    Imports {
        input: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Rank module-entry evidence without assuming the protected header entry is original.
    Oep {
        input: PathBuf,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Serialize)]
struct InfoOutput<'a> {
    schema_version: u32,
    path: String,
    sha256: String,
    pe: &'a PeReport,
}

#[derive(Debug, Serialize)]
struct Comparison {
    schema_version: u32,
    left: FileSummary,
    right: FileSummary,
    differences: Vec<String>,
}

#[derive(Debug, Serialize)]
struct FileSummary {
    path: String,
    sha256: String,
    file_size: u64,
    kind: vmpx_pe::PeKind,
    machine: u16,
    image_base: vmpx_pe::ImageBase,
    entry_point: vmpx_pe::Rva,
    size_of_image: u32,
    section_count: usize,
    import_modules: usize,
    import_symbols: usize,
    exports: usize,
    tls_callbacks: usize,
    exception_functions: usize,
    relocation_blocks: usize,
    relocation_entries: usize,
    diagnostics: usize,
}

#[derive(Debug, Serialize)]
struct ImportReference {
    reference: IpRelativeReference,
    discovery: &'static str,
    module: Option<String>,
    symbol: Option<String>,
    ordinal: Option<u16>,
}

#[derive(Debug, Serialize)]
struct ImportAnalysis {
    schema_version: u32,
    modules: usize,
    symbols: usize,
    named_symbols: usize,
    ordinal_symbols: usize,
    duplicate_symbol_groups: usize,
    iat_rva: Option<vmpx_pe::Rva>,
    iat_size: u32,
    analyzed_function_ranges: usize,
    analyzed_function_bytes: u64,
    executable_file_backed_bytes: u64,
    function_resync_candidate_decodes: usize,
    function_resync_resource_limited: bool,
    resync_candidate_decodes: usize,
    resync_resource_limited: bool,
    direct_native_references: Vec<ImportReference>,
    slots_with_direct_native_references: usize,
    slots_without_direct_native_references: usize,
    warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
struct OepAnalysis {
    schema_version: u32,
    header_entry: vmpx_pe::Rva,
    header_entry_vm_confidence: Option<u8>,
    header_entry_assessment: &'static str,
    recovered_entry: Option<vmpx_pe::Rva>,
    native_tls_candidates: Vec<vmpx_pe::Rva>,
    evidence: Vec<String>,
    warnings: Vec<String>,
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).with_context(|| format!("failed to read {}", path.display()))
}

fn sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:X}", hasher.finalize())
}

fn inspect(path: &Path) -> Result<(Vec<u8>, PeReport)> {
    let data = read_file(path)?;
    let report = PeImage::parse(&data)
        .with_context(|| {
            format!(
                "{} is not a supported, structurally parseable PE",
                path.display()
            )
        })?
        .report();
    Ok((data, report))
}

fn print_report(path: &Path, digest: &str, report: &PeReport) {
    println!(
        "[pe] {:?} {} image detected",
        report.optional.kind, report.coff.machine_name
    );
    println!("[file] {}", path.display());
    println!("[hash] SHA-256 {digest}");
    println!(
        "[headers] image_base={} entry_rva={} entry_va={} size_of_image={:#X}",
        report.optional.image_base,
        report.optional.address_of_entry_point,
        report
            .entry_point_va
            .map_or_else(|| "unrepresentable".to_owned(), |value| value.to_string()),
        report.optional.size_of_image
    );
    println!(
        "[headers] section_alignment={:#X} file_alignment={:#X} subsystem={}",
        report.optional.section_alignment,
        report.optional.file_alignment,
        report.optional.subsystem_name
    );
    println!("[sections] {}", report.sections.len());
    for section in &report.sections {
        println!(
            "  {:>2} {:<8} rva={} vsize={:#X} raw={} raw_size={:#X} chars={:#010X}",
            section.index,
            section.name,
            section.virtual_address,
            section.virtual_size,
            section.pointer_to_raw_data,
            section.size_of_raw_data,
            section.characteristics
        );
    }
    println!("[directories]");
    for directory in report
        .directories
        .iter()
        .filter(|directory| directory.is_present())
    {
        println!(
            "  {:<16} {:?}={:#X} size={:#X}",
            directory.name, directory.address_kind, directory.address, directory.size
        );
    }
    let import_symbols: usize = report
        .imports
        .iter()
        .map(|module| module.entries.len())
        .sum();
    println!(
        "[imports] {} modules, {} symbols",
        report.imports.len(),
        import_symbols
    );
    println!(
        "[exports] {} symbols",
        report
            .exports
            .as_ref()
            .map_or(0, |exports| exports.symbols.len())
    );
    println!(
        "[tls] {} callbacks",
        report.tls.as_ref().map_or(0, |tls| tls.callbacks.len())
    );
    println!(
        "[exceptions] {} runtime functions",
        report.exception_functions.len()
    );
    let relocation_entries: usize = report
        .relocations
        .iter()
        .map(|block| block.entries.len())
        .sum();
    println!(
        "[relocations] {} blocks, {} entries",
        report.relocations.len(),
        relocation_entries
    );
    if report.diagnostics.is_empty() {
        println!("[verify] no structural diagnostics");
    } else {
        println!(
            "[verify] {} structural diagnostics",
            report.diagnostics.len()
        );
        for diagnostic in &report.diagnostics {
            println!("  [{}] {}", diagnostic.stage, diagnostic.message);
        }
    }
}

fn summarize(path: &Path, data: &[u8], report: &PeReport) -> FileSummary {
    FileSummary {
        path: path.display().to_string(),
        sha256: sha256(data),
        file_size: report.file_size,
        kind: report.optional.kind,
        machine: report.coff.machine,
        image_base: report.optional.image_base,
        entry_point: report.optional.address_of_entry_point,
        size_of_image: report.optional.size_of_image,
        section_count: report.sections.len(),
        import_modules: report.imports.len(),
        import_symbols: report
            .imports
            .iter()
            .map(|module| module.entries.len())
            .sum(),
        exports: report
            .exports
            .as_ref()
            .map_or(0, |exports| exports.symbols.len()),
        tls_callbacks: report.tls.as_ref().map_or(0, |tls| tls.callbacks.len()),
        exception_functions: report.exception_functions.len(),
        relocation_blocks: report.relocations.len(),
        relocation_entries: report
            .relocations
            .iter()
            .map(|block| block.entries.len())
            .sum(),
        diagnostics: report.diagnostics.len(),
    }
}

fn compare_reports(left: &PeReport, right: &PeReport) -> Vec<String> {
    let mut differences = Vec::new();
    macro_rules! compare_field {
        ($label:literal, $left:expr, $right:expr) => {
            if $left != $right {
                differences.push(format!("{}: {:?} -> {:?}", $label, $left, $right));
            }
        };
    }
    compare_field!("file_size", left.file_size, right.file_size);
    compare_field!("PE kind", left.optional.kind, right.optional.kind);
    compare_field!("machine", left.coff.machine, right.coff.machine);
    compare_field!(
        "image_base",
        left.optional.image_base,
        right.optional.image_base
    );
    compare_field!(
        "entry_point",
        left.optional.address_of_entry_point,
        right.optional.address_of_entry_point
    );
    compare_field!(
        "size_of_image",
        left.optional.size_of_image,
        right.optional.size_of_image
    );
    compare_field!("section_count", left.sections.len(), right.sections.len());
    for index in 0..left.sections.len().max(right.sections.len()) {
        match (left.sections.get(index), right.sections.get(index)) {
            (Some(a), Some(b)) => {
                compare_field!("section name", (&index, &a.name), (&index, &b.name));
                compare_field!(
                    "section virtual geometry",
                    (&index, a.virtual_address, a.virtual_size),
                    (&index, b.virtual_address, b.virtual_size)
                );
                compare_field!(
                    "section raw geometry",
                    (&index, a.pointer_to_raw_data, a.size_of_raw_data),
                    (&index, b.pointer_to_raw_data, b.size_of_raw_data)
                );
                compare_field!(
                    "section characteristics",
                    (&index, a.characteristics),
                    (&index, b.characteristics)
                );
            }
            (Some(a), None) => differences.push(format!(
                "section {index} ({}) exists only in left image",
                a.name
            )),
            (None, Some(b)) => differences.push(format!(
                "section {index} ({}) exists only in right image",
                b.name
            )),
            (None, None) => {}
        }
    }
    compare_field!("data_directories", &left.directories, &right.directories);
    compare_field!("import_modules", left.imports.len(), right.imports.len());
    compare_field!(
        "import_symbols",
        left.imports
            .iter()
            .map(|module| module.entries.len())
            .sum::<usize>(),
        right
            .imports
            .iter()
            .map(|module| module.entries.len())
            .sum::<usize>()
    );
    compare_field!(
        "export_symbols",
        left.exports.as_ref().map_or(0, |value| value.symbols.len()),
        right
            .exports
            .as_ref()
            .map_or(0, |value| value.symbols.len())
    );
    compare_field!(
        "tls_callbacks",
        left.tls.as_ref().map_or(0, |value| value.callbacks.len()),
        right.tls.as_ref().map_or(0, |value| value.callbacks.len())
    );
    compare_field!(
        "exception_functions",
        left.exception_functions.len(),
        right.exception_functions.len()
    );
    differences
}

fn print_detection(report: &ProtectionReport) {
    println!(
        "[vmp] verdict={:?} confidence={} layout={:?}",
        report.verdict, report.confidence, report.image_layout
    );
    for item in &report.evidence {
        println!("  +{:>2} {}: {}", item.weight, item.id, item.statement);
    }
    for region in &report.executable_regions {
        println!(
            "[region] {:<8} rva={} size={:#X} entropy={} entry={}",
            region.section_name,
            region.start,
            region.size,
            region
                .entropy
                .map_or_else(|| "unavailable".to_owned(), |value| format!("{value:.4}")),
            region.owns_entry_point
        );
    }
    for warning in &report.warnings {
        println!("[vmp] warning: {warning}");
    }
}

fn print_vm_discovery(report: &VmDiscoveryReport) {
    println!(
        "[vm] {} entries grouped into {} bootstrap profiles ({} stub candidates checked)",
        report.entries.len(),
        report.instances.len(),
        report.scanned_stub_candidates
    );
    for instance in &report.instances {
        println!(
            "[vm] instance {} bootstrap={} entries={} seeds={}",
            instance.id,
            instance.bootstrap_rva,
            instance.entry_rvas.len(),
            instance.distinct_seeds.len()
        );
    }
    for entry in &report.entries {
        println!(
            "  entry={} source={:?} reachability={:?} confidence={} seed={} call={} bootstrap={}",
            entry.rva,
            entry.source,
            entry.reachability,
            entry.confidence,
            entry
                .seed
                .map_or_else(|| "unknown".to_owned(), |value| format!("{value:#X}")),
            entry
                .call_target
                .map_or_else(|| "unknown".to_owned(), |value| value.to_string()),
            entry
                .bootstrap_rva
                .map_or_else(|| "unknown".to_owned(), |value| value.to_string())
        );
    }
    for diagnostic in &report.diagnostics {
        println!("[vm] warning: {diagnostic}");
    }
}

#[allow(clippy::too_many_lines)]
fn analyze_imports(pe: &PeImage<'_>, report: &PeReport) -> Result<ImportAnalysis> {
    let symbols = report
        .imports
        .iter()
        .map(|module| module.entries.len())
        .sum();
    let named_symbols = report
        .imports
        .iter()
        .flat_map(|module| &module.entries)
        .filter(|symbol| symbol.name.is_some())
        .count();
    let ordinal_symbols = symbols - named_symbols;
    let mut duplicate_counts = BTreeMap::new();
    let mut slots = HashMap::new();
    for module in &report.imports {
        for symbol in &module.entries {
            let identity = (
                module.name.to_ascii_lowercase(),
                symbol.name.clone(),
                symbol.ordinal,
            );
            *duplicate_counts.entry(identity).or_insert(0_usize) += 1;
            slots.insert(
                symbol.iat_rva,
                (module.name.clone(), symbol.name.clone(), symbol.ordinal),
            );
        }
    }
    let duplicate_symbol_groups = duplicate_counts
        .values()
        .filter(|count| **count > 1)
        .count();
    let functions: Vec<_> = report
        .exception_functions
        .iter()
        .filter(|function| function.begin < function.end)
        .map(|function| (function.begin, function.end))
        .collect();
    let analyzed_function_bytes = functions.iter().fold(0_u64, |total, (begin, end)| {
        total.saturating_add(u64::from(end.0.saturating_sub(begin.0)))
    });
    let executable_file_backed_bytes = pe
        .sections
        .iter()
        .filter(|section| section.characteristics & 0x2000_0000 != 0)
        .fold(0_u64, |total, section| {
            total.saturating_add(u64::from(section.size_of_raw_data))
        });
    let iat = pe.directory(vmpx_pe::IMAGE_DIRECTORY_ENTRY_IAT);
    let iat_source_range = iat.and_then(|directory| {
        directory
            .address
            .checked_add(directory.size)
            .map(|end| (vmpx_pe::Rva(directory.address), vmpx_pe::Rva(end)))
    });
    let (iat_rva, iat_size) = iat.map_or((None, 0), |directory| {
        (Some(vmpx_pe::Rva(directory.address)), directory.size)
    });
    let exact_slots: BTreeSet<_> = slots.keys().copied().collect();
    let function_resync =
        find_ip_relative_references_resync_in_ranges(pe, &exact_slots, &functions, 30_000_000)?;
    let mut fallback_excluded_ranges = functions.clone();
    fallback_excluded_ranges.extend(iat_source_range);
    let resync = find_ip_relative_references_resync(
        pe,
        &exact_slots,
        &fallback_excluded_ranges,
        30_000_000,
    )?;
    let mut references = BTreeMap::new();
    for reference in resync.references {
        references.insert(reference, "executable_resync_candidate");
    }
    for reference in function_resync.references {
        references.insert(reference, "exception_range_resync_candidate");
    }
    let mut referenced_slots = BTreeSet::new();
    let direct_native_references = references
        .into_iter()
        .map(|(reference, discovery)| {
            referenced_slots.insert(reference.target_rva);
            let (module, symbol, ordinal) = slots
                .get(&reference.target_rva)
                .cloned()
                .map_or((None, None, None), |(module, symbol, ordinal)| {
                    (Some(module), symbol, ordinal)
                });
            ImportReference {
                reference,
                discovery,
                module,
                symbol,
                ordinal,
            }
        })
        .collect();
    let mut warnings = vec![
        "absence of a direct RIP-relative reference does not prove an import is unused; VM/native indirection remains unresolved".to_owned(),
        "duplicate descriptors and symbols are preserved as evidence rather than collapsed".to_owned(),
        "overlapping exception ranges are merged and scanned as bounded independent starts; exception metadata alone does not prove reachability".to_owned(),
        "independent-start findings are decoded exact-slot candidates; reachability requires CFG and local def-use evidence".to_owned(),
    ];
    if function_resync.resource_limited {
        warnings.push(
            "the exception-range resynchronizing scan reached its candidate-decode budget; coverage is truncated"
                .to_owned(),
        );
    }
    if resync.resource_limited {
        warnings.push(
            "the resynchronizing executable scan reached its candidate-decode budget; coverage is truncated"
                .to_owned(),
        );
    }
    Ok(ImportAnalysis {
        schema_version: 1,
        modules: report.imports.len(),
        symbols,
        named_symbols,
        ordinal_symbols,
        duplicate_symbol_groups,
        iat_rva,
        iat_size,
        analyzed_function_ranges: functions.len(),
        analyzed_function_bytes,
        executable_file_backed_bytes,
        function_resync_candidate_decodes: function_resync.candidate_decodes,
        function_resync_resource_limited: function_resync.resource_limited,
        resync_candidate_decodes: resync.candidate_decodes,
        resync_resource_limited: resync.resource_limited,
        direct_native_references,
        slots_with_direct_native_references: referenced_slots.len(),
        slots_without_direct_native_references: symbols.saturating_sub(referenced_slots.len()),
        warnings,
    })
}

fn analyze_oep(pe: &PeImage<'_>) -> Result<OepAnalysis> {
    let limits = VmDiscoveryLimits {
        scan_executable_stubs: false,
        ..VmDiscoveryLimits::default()
    };
    let vm = discover_vm_entries(pe, limits)?;
    let header_entry = pe.optional.address_of_entry_point;
    let header_vm = vm.entries.iter().find(|entry| entry.rva == header_entry);
    let native_tls_candidates = vm
        .rejected_roots
        .iter()
        .filter(|candidate| candidate.source == VmEntrySource::TlsCallback)
        .map(|candidate| candidate.rva)
        .collect();
    let mut evidence = Vec::new();
    let (assessment, confidence) = if let Some(entry) = header_vm {
        evidence.extend(
            entry
                .evidence
                .iter()
                .map(|item| format!("+{} {}", item.weight, item.statement)),
        );
        ("protector_vm_entry", Some(entry.confidence))
    } else {
        ("unresolved", None)
    };
    Ok(OepAnalysis {
        schema_version: 1,
        header_entry,
        header_entry_vm_confidence: confidence,
        header_entry_assessment: assessment,
        recovered_entry: None,
        native_tls_candidates,
        evidence,
        warnings: vec![
            "no original native module entry is emitted until a VM-exit/native-transition path is decoded".to_owned(),
            "the supplied workspace has no unprotected semantic oracle".to_owned(),
        ],
    })
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Clean {
            input,
            output,
            report,
        } => {
            if input == output {
                bail!("refusing to overwrite the input file");
            }
            if output.exists() && fs::canonicalize(&input).ok() == fs::canonicalize(&output).ok() {
                bail!("refusing to overwrite the input file through an alias");
            }
            if output.exists() {
                bail!("refusing to overwrite existing output {}", output.display());
            }
            if let Some(report_path) = &report
                && (report_path == &input
                    || report_path == &output
                    || (report_path.exists()
                        && fs::canonicalize(&input).ok() == fs::canonicalize(report_path).ok()))
            {
                bail!("report path aliases an input or output file");
            }
            let input_data = read_file(&input)?;
            let mut clean = clean::analyze_clean(&input.display().to_string(), &input_data)?;
            println!(
                "[clean] protection={:?} layout={:?} vm_entries={} routines={} blockers={}",
                clean.protection.verdict,
                clean.protection.image_layout,
                clean.vm_entries.len(),
                clean.virtualized_routines.len(),
                clean.blockers.len()
            );
            for path in &clean.virtualized_routines {
                println!(
                    "[decode] assignments={:?} operations={} stop={:?}",
                    path.assignments, path.decoded_operations, path.stop_reason
                );
            }
            for blocker in &clean.blockers {
                println!("[clean] blocker: {blocker}");
            }
            if !clean.ready_to_write {
                if let Some(report_path) = report {
                    fs::write(&report_path, serde_json::to_vec_pretty(&clean)?).with_context(
                        || format!("failed to write report {}", report_path.display()),
                    )?;
                    println!("[report] {}", report_path.display());
                }
                bail!(
                    "clean proof gate refused {}; output {} was not written",
                    input.display(),
                    output.display()
                );
            }
            if clean.protection.verdict != vmpx_detect::DetectionVerdict::NoStrongEvidence
                || clean.protection.image_layout != vmpx_detect::ImageLayout::ConventionalFile
            {
                bail!("clean report became write-ready without a completed reconstruction proof");
            }
            fs::write(&output, &input_data)
                .with_context(|| format!("failed to write {}", output.display()))?;
            let disk_data = read_file(&output)?;
            PeImage::parse(&disk_data).with_context(|| {
                format!("written output {} failed disk reparse", output.display())
            })?;
            clean.validation.output_written = true;
            clean.validation.output_reparsed = true;
            println!(
                "[clean] no strong protection evidence; preserved {} bytes in {}",
                input_data.len(),
                output.display()
            );
            println!("[verify] output reparsed successfully from disk");
            if let Some(report_path) = report {
                fs::write(&report_path, serde_json::to_vec_pretty(&clean)?)
                    .with_context(|| format!("failed to write report {}", report_path.display()))?;
                println!("[report] {}", report_path.display());
            }
        }
        Command::Info { input, json } => {
            let (data, report) = inspect(&input)?;
            let digest = sha256(&data);
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&InfoOutput {
                        schema_version: 1,
                        path: input.display().to_string(),
                        sha256: digest,
                        pe: &report,
                    })?
                );
            } else {
                print_report(&input, &digest, &report);
            }
        }
        Command::Verify { input, json } => {
            let (data, report) = inspect(&input)?;
            let digest = sha256(&data);
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&InfoOutput {
                        schema_version: 1,
                        path: input.display().to_string(),
                        sha256: digest,
                        pe: &report,
                    })?
                );
            } else {
                print_report(&input, &digest, &report);
            }
            if !report.diagnostics.is_empty() {
                bail!("structural verification reported diagnostics");
            }
        }
        Command::Compare { left, right, json } => {
            let (left_data, left_report) = inspect(&left)?;
            let (right_data, right_report) = inspect(&right)?;
            let comparison = Comparison {
                schema_version: 1,
                left: summarize(&left, &left_data, &left_report),
                right: summarize(&right, &right_data, &right_report),
                differences: compare_reports(&left_report, &right_report),
            };
            if json {
                println!("{}", serde_json::to_string_pretty(&comparison)?);
            } else {
                println!("[compare] {}", left.display());
                println!("[compare] {}", right.display());
                if comparison.differences.is_empty() {
                    println!("[compare] no modeled structural differences");
                } else {
                    println!(
                        "[compare] {} modeled structural differences",
                        comparison.differences.len()
                    );
                    for difference in &comparison.differences {
                        println!("  {difference}");
                    }
                }
            }
        }
        Command::Unpack {
            input,
            output,
            report,
        } => {
            if input == output {
                bail!("refusing to overwrite the input file");
            }
            if output.exists() && fs::canonicalize(&input).ok() == fs::canonicalize(&output).ok() {
                bail!("refusing to overwrite the input file through an alias");
            }
            let input_data = read_file(&input)?;
            let mut result = unpack_to_mapped_image(&input_data, UnpackLimits::default())?;
            println!(
                "[unpack] {:?} PACKER_INFO at {} with {} blocks",
                result.report.packer_table.kind,
                result.report.packer_table.file_offset,
                result.report.blocks.len()
            );
            for (index, block) in result.report.blocks.iter().enumerate() {
                println!(
                    "  block {index}: {:<8} src={} dst={} size={:#X}/{:#X} (vsize={:#X})",
                    block.section_name,
                    block.source_rva,
                    block.destination_rva,
                    block.decompressed_size,
                    block.destination_capacity,
                    block.section_virtual_size
                );
            }
            for warning in &result.report.warnings {
                println!("[unpack] warning: {warning}");
            }
            fs::write(&output, &result.image)
                .with_context(|| format!("failed to write {}", output.display()))?;
            let disk_data = read_file(&output)?;
            PeImage::parse(&disk_data).with_context(|| {
                format!("written output {} failed disk reparse", output.display())
            })?;
            result.report.disk_reparsed = true;
            println!(
                "[unpack] wrote {} bytes to {}",
                result.image.len(),
                output.display()
            );
            println!("[verify] output reparsed successfully from disk");
            if let Some(report_path) = report {
                let json = serde_json::to_vec_pretty(&result.report)?;
                fs::write(&report_path, json)
                    .with_context(|| format!("failed to write report {}", report_path.display()))?;
                println!("[report] {}", report_path.display());
            }
        }
        Command::Detect { input, json } => {
            let data = read_file(&input)?;
            let pe = PeImage::parse(&data).with_context(|| {
                format!(
                    "{} is not a supported, structurally parseable PE",
                    input.display()
                )
            })?;
            let report = detect_vmprotect(&pe);
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_detection(&report);
            }
        }
        Command::VmList {
            input,
            json,
            no_scan,
        } => {
            let data = read_file(&input)?;
            let pe = PeImage::parse(&data).with_context(|| {
                format!(
                    "{} is not a supported, structurally parseable PE",
                    input.display()
                )
            })?;
            let report = discover_vm_entries(
                &pe,
                VmDiscoveryLimits {
                    scan_executable_stubs: !no_scan,
                    ..VmDiscoveryLimits::default()
                },
            )?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_vm_discovery(&report);
            }
        }
        Command::Imports { input, json } => {
            let data = read_file(&input)?;
            let pe = PeImage::parse(&data).with_context(|| {
                format!(
                    "{} is not a supported, structurally parseable PE",
                    input.display()
                )
            })?;
            let pe_report = pe.report();
            let report = analyze_imports(&pe, &pe_report)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "[imports] {} modules, {} symbols ({} named, {} ordinal)",
                    report.modules, report.symbols, report.named_symbols, report.ordinal_symbols
                );
                println!(
                    "[imports] {} duplicate identity groups; {} IAT slots have direct native references",
                    report.duplicate_symbol_groups, report.slots_with_direct_native_references
                );
                println!(
                    "[imports] analyzed {} function ranges / {:#X} modeled bytes; file-backed executable sections total {:#X} bytes",
                    report.analyzed_function_ranges,
                    report.analyzed_function_bytes,
                    report.executable_file_backed_bytes
                );
                println!(
                    "[imports] exception-range recovery attempted {} independent candidate decodes (truncated={})",
                    report.function_resync_candidate_decodes,
                    report.function_resync_resource_limited
                );
                println!(
                    "[imports] remaining-executable fallback attempted {} independent candidate decodes (truncated={})",
                    report.resync_candidate_decodes, report.resync_resource_limited
                );
                for reference in &report.direct_native_references {
                    let identity = match (&reference.module, &reference.symbol, reference.ordinal) {
                        (Some(module), Some(symbol), _) => format!("{module}!{symbol}"),
                        (Some(module), None, Some(ordinal)) => {
                            format!("{module}!ordinal_{ordinal}")
                        }
                        _ => "unmapped_iat_slot".to_owned(),
                    };
                    println!(
                        "  {} -> {} ({identity}; {})",
                        reference.reference.instruction_rva,
                        reference.reference.target_rva,
                        reference.discovery
                    );
                }
                for warning in &report.warnings {
                    println!("[imports] warning: {warning}");
                }
            }
        }
        Command::Oep { input, json } => {
            let data = read_file(&input)?;
            let pe = PeImage::parse(&data).with_context(|| {
                format!(
                    "{} is not a supported, structurally parseable PE",
                    input.display()
                )
            })?;
            let report = analyze_oep(&pe)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "[oep] header entry {} assessment={} confidence={}",
                    report.header_entry,
                    report.header_entry_assessment,
                    report
                        .header_entry_vm_confidence
                        .map_or_else(|| "unknown".to_owned(), |value| value.to_string())
                );
                println!(
                    "[oep] recovered native module entry: {}",
                    report
                        .recovered_entry
                        .map_or_else(|| "unresolved".to_owned(), |value| value.to_string())
                );
                for item in &report.evidence {
                    println!("  {item}");
                }
                for warning in &report.warnings {
                    println!("[oep] warning: {warning}");
                }
            }
        }
    }
    Ok(())
}
