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
    Unpack,
    Clean,
}

pub const GROUPS: [(&str, &[Action]); 3] = [
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
    ("RECOVER", &[Action::Unpack, Action::Clean]),
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
            Self::Unpack => "Unpack",
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
            Self::Unpack => "unpack",
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
            Self::Unpack => "Recover packed sections into an RVA-layout analysis image.",
            Self::Clean => "Proof-gated recovery pipeline; emits a clean PE only when safe.",
        }
    }

    fn has_json_flag(self) -> bool {
        !matches!(self, Self::Unpack | Self::Clean)
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

#[derive(Clone, Debug)]
pub struct Options {
    pub json: bool,
    pub no_scan: bool,
    pub compare_right: String,
    pub clean_output: String,
    pub clean_report: PathOpt,
    pub reconstructed: PathOpt,
    pub experimental: PathOpt,
    pub unpack_output: String,
    pub unpack_report: PathOpt,
    pub conventional: PathOpt,
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
            unpack_output: String::new(),
            unpack_report: PathOpt::default(),
            conventional: PathOpt::default(),
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
    match job.action {
        Action::Compare => args.push(opts.compare_right.trim().into()),
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
        }
        _ => {}
    }
    if job.action.has_json_flag() && opts.json {
        args.push("--json".into());
    }
    if job.action == Action::VmList && opts.no_scan {
        args.push("--no-scan".into());
    }
    Plan {
        program,
        args,
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
    std::iter::once(&plan.program)
        .chain(&plan.args)
        .map(|arg| quote(arg))
        .collect::<Vec<_>>()
        .join(" ")
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
