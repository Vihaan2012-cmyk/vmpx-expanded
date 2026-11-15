use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant, SystemTime},
};

use anyhow::{Context, Result};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Action {
    Detect,
    Info,
    Verify,
    Compare,
    Imports,
    VmList,
    Oep,
    Report,
    Decompile,
    VerifyEquiv,
    DiffOriginal,
    Unpack,
    GenericUnpack,
    Strip,
    Clean,
}

pub const GROUPS: [(&str, &[Action]); 4] = [
    (
        "INSPECT",
        &[
            Action::Detect,
            Action::Info,
            Action::Verify,
            Action::Imports,
            Action::Compare,
        ],
    ),
    ("DISCOVER", &[Action::VmList, Action::Oep]),
    (
        "ANALYSE",
        &[
            Action::Report,
            Action::Decompile,
            Action::VerifyEquiv,
            Action::DiffOriginal,
        ],
    ),
    (
        "RECOVER",
        &[
            Action::Unpack,
            Action::GenericUnpack,
            Action::Strip,
            Action::Clean,
        ],
    ),
];

impl Action {
    pub fn label(self) -> &'static str {
        match self {
            Self::Detect => "Detect",
            Self::Info => "Info",
            Self::Verify => "Verify",
            Self::Compare => "Compare",
            Self::Imports => "Imports",
            Self::VmList => "VM list",
            Self::Oep => "OEP",
            Self::Report => "Report",
            Self::Decompile => "Decompile",
            Self::VerifyEquiv => "Verify equiv",
            Self::DiffOriginal => "Diff original",
            Self::Unpack => "Unpack",
            Self::GenericUnpack => "Generic unpack",
            Self::Strip => "Strip",
            Self::Clean => "Clean",
        }
    }

    pub fn subcommand(self) -> &'static str {
        match self {
            Self::Detect => "detect",
            Self::Info => "info",
            Self::Verify => "verify",
            Self::Compare => "compare",
            Self::Imports => "imports",
            Self::VmList => "vm-list",
            Self::Oep => "oep",
            Self::Report => "report",
            Self::Decompile => "decompile",
            Self::VerifyEquiv => "verify-equiv",
            Self::DiffOriginal => "diff-original",
            Self::Unpack => "unpack",
            Self::GenericUnpack => "generic-unpack",
            Self::Strip => "strip",
            Self::Clean => "clean",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Detect => "Profile VMProtect evidence and packed/mapped layout.",
            Self::Info => "Inspect a PE-family file by content, regardless of extension.",
            Self::Verify => "Structural validation without executing the input.",
            Self::Compare => "Research-only structural comparison of two images.",
            Self::Imports => "Inventory imports and direct native RIP-relative IAT references.",
            Self::VmList => "Discover evidence-scored VM entry stubs and bootstrap profiles.",
            Self::Oep => "Rank module-entry evidence without trusting the protected header entry.",
            Self::Report => {
                "Tiered N1/N2/N3 report over the discovered VM entries. Writes nothing."
            }
            Self::Decompile => "Raise a recovered function to C-like pseudocode.",
            Self::VerifyEquiv => {
                "Equivalence gate: proves, refutes or leaves unproven each recovered function."
            }
            Self::DiffOriginal => {
                "Ground-truth differential against the unprotected original binary."
            }
            Self::Unpack => "Recover packed sections into an RVA-layout analysis image.",
            Self::GenericUnpack => {
                "Non-VMProtect packer (UPX/ASPack/PECompact/MPRESS/FSG): bounded unpack to OEP."
            }
            Self::Strip => "Remove named sections when the dangling-ref checker approves.",
            Self::Clean => "Proof-gated recovery pipeline; emits a clean PE only when safe.",
        }
    }

    /// The label for the second positional path, when the subcommand takes one.
    pub fn second_positional(self) -> Option<&'static str> {
        match self {
            Self::Compare => Some("Second image"),
            Self::DiffOriginal => Some("Original (unprotected)"),
            Self::GenericUnpack | Self::Strip => Some("Output"),
            _ => None,
        }
    }

    fn has_json_flag(self) -> bool {
        matches!(
            self,
            Self::Detect
                | Self::Info
                | Self::Verify
                | Self::Compare
                | Self::Imports
                | Self::VmList
                | Self::Oep
                | Self::Report
        )
    }
}

#[derive(Clone, Debug, Default)]
pub struct PathOpt {
    pub enabled: bool,
    pub path: String,
}

impl PathOpt {
    fn value(&self) -> Option<&str> {
        let path = self.path.trim();
        (self.enabled && !path.is_empty()).then_some(path)
    }

    pub fn incomplete(&self) -> bool {
        self.enabled && self.path.trim().is_empty()
    }
}

/// One environment variable row.
///
/// Much of what changes a run lives in the environment rather than on the
/// command line, and several of those knobs move the result materially, so the
/// UI surfaces them as first-class options instead of leaving them invisible.
#[derive(Clone, Debug)]
pub struct EnvRow {
    pub enabled: bool,
    pub name: String,
    pub value: String,
    pub help: &'static str,
}

impl EnvRow {
    fn new(name: &str, value: &str, help: &'static str) -> Self {
        Self {
            enabled: false,
            name: name.to_owned(),
            value: value.to_owned(),
            help,
        }
    }
}

/// The knobs worth one click, pre-filled with the value usually wanted. The long
/// tail stays reachable: the final row is blank and editable.
fn default_env() -> Vec<EnvRow> {
    vec![
        EnvRow::new(
            "VMPX_FOLLOW_CROSS_VMP_SECTION",
            "1",
            "Follow a branch target in another executable section (labels the CFG)",
        ),
        EnvRow::new(
            "VMPX_INSTRUCTION_LIMIT",
            "4",
            "Multiplier on the per-block VM instruction budget; report's base is 512",
        ),
        EnvRow::new("VMPX_WALL_SECS", "400", "Per-entry wall-clock budget in seconds"),
        EnvRow::new(
            "VMPX_DECOMPILE_SERVE_RO",
            "1",
            "Serve-RO for decompile; report has its own --serve-ro flag instead",
        ),
        EnvRow::new("VMPX_ONLY_ENTRIES", "", "Comma-separated hex entry RVAs to restrict the sweep"),
        EnvRow::new("VMPX_PROGRESS", "", "File given running done/n1/n2/incomplete counts"),
        EnvRow::new("VMPX_STALL_DUMP", "", "JSONL, one line per incomplete entry"),
        EnvRow::new("VMPX_ENTRY_TRACE", "", "Per-entry start/done with elapsed and outcome"),
        EnvRow::new("VMPX_TRACE_PHASES", "1", "Phase markers; the last unpaired one names a hang"),
        EnvRow::new("VMPX_MEM_TRACE", "1", "Enable the [slow] timing probes"),
        EnvRow::new("VMPX_SLOW_MS", "200", "Timing-probe floor in ms; the default is 5000"),
        EnvRow::new("VMPX_BUDGET_PASSES", "", "tb,visits,total,instr multipliers; passes split by ;"),
        EnvRow::new("VMPX_CFG_AUDIT", "", "Record any complete CFG with a dangling successor"),
        EnvRow::new("VMPX_DISCHARGE_DUMP", "", "Why a complete entry could not reach N2"),
        EnvRow::new("", "", "Any other variable"),
    ]
}

#[derive(Clone, Debug)]
pub struct Options {
    pub json: bool,
    pub no_scan: bool,
    pub compare_right: String,
    pub clean_output: String,
    pub clean_report: PathOpt,
    pub reconstructed: PathOpt,
    pub experimental: PathOpt,
    pub decoded_lift: bool,
    pub clean_dynamic: bool,
    pub dump_pe: PathOpt,
    pub clean_strip: bool,
    pub unpack_output: String,
    pub unpack_report: PathOpt,
    pub conventional: PathOpt,
    pub all_entries: bool,
    pub sample: String,
    pub threads: String,
    pub serve_ro: bool,
    pub serve_observed_ro: bool,
    pub dynamic: bool,
    pub volatility_snapshot: PathOpt,
    pub rva: String,
    pub function: String,
    pub instruction_limit: String,
    pub exhaustive_bits: String,
    pub timeout_ms: String,
    pub native: bool,
    pub probe: bool,
    pub original: PathOpt,
    pub max_functions: String,
    pub no_optimize: bool,
    pub solver: bool,
    pub strict_memory: bool,
    pub verbose: bool,
    pub strict_gates: bool,
    pub max_blocks: String,
    pub max_paths: String,
    pub control: bool,
    pub remove: String,
    pub drop_directory: String,
    pub second_output: String,
    pub env: Vec<EnvRow>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            json: true,
            no_scan: false,
            compare_right: String::new(),
            clean_output: String::new(),
            clean_report: PathOpt::default(),
            reconstructed: PathOpt::default(),
            experimental: PathOpt::default(),
            decoded_lift: false,
            clean_dynamic: false,
            dump_pe: PathOpt::default(),
            clean_strip: false,
            unpack_output: String::new(),
            unpack_report: PathOpt::default(),
            conventional: PathOpt::default(),
            all_entries: true,
            sample: String::new(),
            threads: String::new(),
            serve_ro: true,
            serve_observed_ro: false,
            dynamic: false,
            volatility_snapshot: PathOpt::default(),
            rva: String::new(),
            function: String::new(),
            instruction_limit: String::new(),
            exhaustive_bits: String::new(),
            timeout_ms: String::new(),
            native: false,
            probe: false,
            original: PathOpt::default(),
            max_functions: String::new(),
            no_optimize: false,
            solver: false,
            strict_memory: false,
            verbose: false,
            strict_gates: false,
            max_blocks: String::new(),
            max_paths: String::new(),
            control: false,
            remove: String::new(),
            drop_directory: String::new(),
            second_output: String::new(),
            env: default_env(),
        }
    }
}

impl Options {
    /// Why the action cannot run yet, if it cannot.
    pub fn missing(&self, action: Action) -> Option<&'static str> {
        match action {
            Action::Compare if self.compare_right.trim().is_empty() => {
                Some("choose the second image to compare against")
            }
            Action::Clean
                if self.clean_report.incomplete()
                    || self.reconstructed.incomplete()
                    || self.experimental.incomplete() =>
            {
                Some("an enabled output path is empty")
            }
            Action::Unpack if self.unpack_report.incomplete() || self.conventional.incomplete() => {
                Some("an enabled output path is empty")
            }
            Action::DiffOriginal if self.second_output.trim().is_empty() => {
                Some("choose the unprotected original image")
            }
            Action::GenericUnpack | Action::Strip if self.second_output.trim().is_empty() => {
                Some("choose the output path")
            }
            Action::Strip if self.remove.trim().is_empty() => {
                Some("--remove needs at least one section name")
            }
            Action::Report if self.dynamic && !self.volatility_snapshot.enabled => {
                Some("--dynamic observes nothing without --volatility-snapshot")
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Launcher {
    Binary(PathBuf),
    Cargo,
}

#[derive(Clone, Debug)]
pub struct Job {
    pub action: Action,
    pub launcher: Launcher,
    pub input: PathBuf,
    pub opts: Options,
}

#[derive(Debug)]
pub struct JobResult {
    pub action: Action,
    pub command_line: String,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    /// Parsed stdout (or the report file) when it was valid JSON.
    pub json: Option<Value>,
    pub elapsed: Duration,
    pub error: Option<String>,
}

impl JobResult {
    pub fn failed(&self) -> bool {
        self.error.is_some() || self.exit_code != Some(0)
    }
}

/// Resolved argument list plus the files the run produces or must clean up.
struct Plan {
    program: OsString,
    args: Vec<OsString>,
    /// Environment the child runs with, from the enabled `EnvRow`s.
    env: Vec<(String, String)>,
    /// Report file to parse after the run, when the command writes one.
    report: Option<PathBuf>,
    scratch: Vec<PathBuf>,
}

fn scratch_path(action: Action, suffix: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "vmpx-ui-{}-{}.{suffix}",
        std::process::id(),
        action.subcommand()
    ))
}

fn plan(job: &Job) -> Plan {
    let opts = &job.opts;
    let (program, mut args): (OsString, Vec<OsString>) = match &job.launcher {
        Launcher::Binary(path) => (path.clone().into(), Vec::new()),
        Launcher::Cargo => (
            "cargo".into(),
            ["run", "--release", "-q", "-p", "vmpx-cli", "--"]
                .map(OsString::from)
                .into(),
        ),
    };
    args.push(job.action.subcommand().into());
    args.push(job.input.clone().into());
    let mut report = None;
    let mut scratch = Vec::new();
    let flag = |args: &mut Vec<OsString>, name: &str, value: &Path| {
        args.push(name.into());
        args.push(value.into());
    };
    // Repeatable value flags take a comma- or space-separated list in the UI.
    let many = |args: &mut Vec<OsString>, name: &str, text: &str| {
        for part in text.split([',', ' ', '\t']).filter(|p| !p.trim().is_empty()) {
            args.push(name.into());
            args.push(part.trim().into());
        }
    };
    let value = |args: &mut Vec<OsString>, name: &str, text: &str| {
        let text = text.trim();
        if !text.is_empty() {
            args.push(name.into());
            args.push(text.into());
        }
    };
    match job.action {
        Action::Compare => args.push(opts.compare_right.trim().into()),
        Action::DiffOriginal => {
            args.push(opts.second_output.trim().into());
            many(&mut args, "--function", &opts.function);
            value(&mut args, "--exhaustive-bits", &opts.exhaustive_bits);
            value(&mut args, "--timeout-ms", &opts.timeout_ms);
            value(&mut args, "--instruction-limit", &opts.instruction_limit);
            value(&mut args, "--max-blocks", &opts.max_blocks);
            value(&mut args, "--max-paths", &opts.max_paths);
            if opts.strict_gates {
                args.push("--strict-gates".into());
            }
            if opts.control {
                args.push("--control".into());
            }
        }
        Action::GenericUnpack => args.push(opts.second_output.trim().into()),
        Action::Strip => {
            args.push(opts.second_output.trim().into());
            many(&mut args, "--remove", &opts.remove);
            many(&mut args, "--drop-directory", &opts.drop_directory);
        }
        Action::Report => {
            if opts.all_entries {
                args.push("--all".into());
            } else {
                value(&mut args, "--sample", &opts.sample);
            }
            value(&mut args, "--threads", &opts.threads);
            if opts.serve_ro {
                args.push("--serve-ro".into());
            }
            if let Some(path) = opts.volatility_snapshot.value() {
                flag(&mut args, "--volatility-snapshot", Path::new(path));
            }
            if opts.serve_observed_ro {
                args.push("--serve-observed-ro".into());
            }
            if opts.dynamic {
                args.push("--dynamic".into());
            }
        }
        Action::Decompile => {
            many(&mut args, "--rva", &opts.rva);
            many(&mut args, "--function", &opts.function);
            if opts.native {
                args.push("--native".into());
            }
            if let Some(path) = opts.original.value() {
                flag(&mut args, "--original", Path::new(path));
            }
            if opts.probe {
                args.push("--probe".into());
            }
            value(&mut args, "--instruction-limit", &opts.instruction_limit);
        }
        Action::VerifyEquiv => {
            value(&mut args, "--max-functions", &opts.max_functions);
            many(&mut args, "--rva", &opts.rva);
            value(&mut args, "--exhaustive-bits", &opts.exhaustive_bits);
            value(&mut args, "--timeout-ms", &opts.timeout_ms);
            value(&mut args, "--instruction-limit", &opts.instruction_limit);
            if opts.no_optimize {
                args.push("--no-optimize".into());
            }
            if opts.solver {
                args.push("--solver".into());
            }
            if opts.strict_memory {
                args.push("--strict-memory".into());
            }
            if opts.verbose {
                args.push("--verbose".into());
            }
        }
        Action::Clean | Action::Unpack => {
            let (output, report_opt, extras) = if job.action == Action::Clean {
                (
                    &opts.clean_output,
                    &opts.clean_report,
                    [
                        ("--reconstructed", &opts.reconstructed),
                        ("--experimental", &opts.experimental),
                    ]
                    .to_vec(),
                )
            } else {
                (
                    &opts.unpack_output,
                    &opts.unpack_report,
                    [("--conventional", &opts.conventional)].to_vec(),
                )
            };
            let output = match output.trim() {
                "" => {
                    let path = scratch_path(job.action, "out.bin");
                    scratch.push(path.clone());
                    path
                }
                path => path.into(),
            };
            args.push(output.into());
            // The report is always requested so the summary can be rendered.
            let report_path = if let Some(path) = report_opt.value() {
                PathBuf::from(path)
            } else {
                let path = scratch_path(job.action, "report.json");
                scratch.push(path.clone());
                path
            };
            flag(&mut args, "--report", &report_path);
            report = Some(report_path);
            for (name, opt) in extras {
                if let Some(path) = opt.value() {
                    flag(&mut args, name, Path::new(path));
                }
            }
            if job.action == Action::Clean {
                if opts.decoded_lift {
                    args.push("--decoded-lift".into());
                }
                if opts.clean_dynamic {
                    args.push("--dynamic".into());
                }
                if let Some(path) = opts.dump_pe.value() {
                    flag(&mut args, "--dump-pe", Path::new(path));
                }
                if opts.clean_strip {
                    args.push("--strip".into());
                }
            }
        }
        _ => {}
    }
    if job.action.has_json_flag() && opts.json {
        args.push("--json".into());
    }
    if job.action == Action::VmList && opts.no_scan {
        args.push("--no-scan".into());
    }
    let env = opts
        .env
        .iter()
        .filter(|row| row.enabled && !row.name.trim().is_empty())
        .map(|row| (row.name.trim().to_owned(), row.value.trim().to_owned()))
        .collect();
    Plan {
        program,
        args,
        env,
        report,
        scratch,
    }
}

fn quote(arg: &std::ffi::OsStr) -> String {
    let text = arg.to_string_lossy();
    if text.is_empty() || text.contains(char::is_whitespace) {
        format!("\"{text}\"")
    } else {
        text.into_owned()
    }
}

/// The exact command line `run` will execute.
pub fn command_line(job: &Job) -> String {
    let plan = plan(job);
    // The environment is part of the command: a line pasted into a shell without
    // it does not reproduce the run.
    let mut parts: Vec<String> = plan
        .env
        .iter()
        .map(|(name, value)| format!("{name}={value}"))
        .collect();
    parts.extend(
        std::iter::once(&plan.program)
            .chain(&plan.args)
            .map(|arg| quote(arg)),
    );
    parts.join(" ")
}

fn run_inner(job: &Job, result: &mut JobResult) -> Result<()> {
    let plan = plan(job);
    result.command_line = command_line(job);
    // Clean refuses to overwrite an existing output, so scratch files start absent.
    for path in &plan.scratch {
        let _ = std::fs::remove_file(path);
    }
    let started = SystemTime::now();
    let output = Command::new(&plan.program)
        .args(&plan.args)
        .envs(plan.env.iter().map(|(k, v)| (k.as_str(), v.as_str())))
        .output()
        .with_context(|| format!("failed to launch {}", result.command_line))?;
    result.exit_code = output.status.code();
    result.stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    result.stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    if let Some(report) = &plan.report {
        // A report left by an earlier run must not be mistaken for this run's.
        let fresh = std::fs::metadata(report)
            .and_then(|meta| meta.modified())
            .is_ok_and(|modified| modified >= started);
        if fresh && let Ok(text) = std::fs::read_to_string(report) {
            result.json = serde_json::from_str(&text).ok();
            if result.json.is_none() {
                result.error = Some("report file was not valid JSON".into());
            }
        }
    } else {
        result.json = serde_json::from_str(&result.stdout).ok();
    }
    for path in &plan.scratch {
        let _ = std::fs::remove_file(path);
    }
    Ok(())
}

pub fn run(job: &Job) -> JobResult {
    let started = Instant::now();
    let mut result = JobResult {
        action: job.action,
        command_line: String::new(),
        exit_code: None,
        stdout: String::new(),
        stderr: String::new(),
        json: None,
        elapsed: Duration::ZERO,
        error: None,
    };
    if let Err(error) = run_inner(job, &mut result) {
        result.error = Some(format!("{error:#}"));
    }
    result.elapsed = started.elapsed();
    result
}

/// Locates a built `vmpx` binary under a `target/release` above the cwd or this exe.
pub fn find_release_binary() -> Option<PathBuf> {
    const NAMES: [&str; 4] = ["vmpx.exe", "vmpx-cli.exe", "vmpx", "vmpx-cli"];
    let roots = [
        std::env::current_dir().ok(),
        std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(Path::to_path_buf)),
    ];
    for root in roots.into_iter().flatten() {
        for dir in root.ancestors() {
            for name in NAMES {
                let candidate = dir.join("target").join("release").join(name);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
            // The exe's own directory may already be target/release.
            for name in NAMES {
                let candidate = dir.join(name);
                if dir.ends_with("release") && candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(action: Action, opts: Options) -> Job {
        Job {
            action,
            launcher: Launcher::Binary("vmpx".into()),
            input: "a.exe".into(),
            opts,
        }
    }

    #[test]
    fn compare_passes_both_inputs_and_json() {
        let opts = Options {
            compare_right: "b.exe".into(),
            ..Options::default()
        };
        assert_eq!(
            command_line(&job(Action::Compare, opts)),
            "vmpx compare a.exe b.exe --json"
        );
    }

    #[test]
    fn clean_emits_only_enabled_paths() {
        let mut opts = Options {
            clean_output: "out.exe".into(),
            ..Options::default()
        };
        opts.experimental = PathOpt {
            enabled: true,
            path: "exp.exe".into(),
        };
        opts.reconstructed.path = "ignored.exe".into();
        let line = command_line(&job(Action::Clean, opts));
        assert!(line.starts_with("vmpx clean a.exe out.exe --report "));
        assert!(line.ends_with("--experimental exp.exe"));
        assert!(!line.contains("--reconstructed"));
    }

    #[test]
    fn vm_list_no_scan() {
        let opts = Options {
            no_scan: true,
            ..Options::default()
        };
        assert_eq!(
            command_line(&job(Action::VmList, opts)),
            "vmpx vm-list a.exe --json --no-scan"
        );
    }
}
