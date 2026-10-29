use std::{
    path::PathBuf,
    process::Command,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Action {
    Detect,
    Info,
    Verify,
    Imports,
    VmList,
    Oep,
    Clean,
}

impl Action {
    pub fn label(self) -> &'static str {
        match self {
            Self::Detect => "Detect",
            Self::Info => "Info",
            Self::Verify => "Verify",
            Self::Imports => "Imports",
            Self::VmList => "VM list",
            Self::Oep => "OEP",
            Self::Clean => "Clean",
        }
    }

    fn subcommand(self) -> &'static str {
        match self {
            Self::Detect => "detect",
            Self::Info => "info",
            Self::Verify => "verify",
            Self::Imports => "imports",
            Self::VmList => "vm-list",
            Self::Oep => "oep",
            Self::Clean => "clean",
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
    pub no_scan: bool,
    /// `Some(path)` passes `--experimental <path>` to `clean`.
    pub experimental_out: Option<String>,
}

#[derive(Debug)]
pub struct JobResult {
    pub action: Action,
    pub command_line: String,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    /// Parsed stdout (or the clean report file) when it was valid JSON.
    pub json: Option<Value>,
    pub elapsed: Duration,
    pub error: Option<String>,
}

struct CleanPaths {
    output: PathBuf,
    report: PathBuf,
}

fn command_for(job: &Job, clean_paths: Option<&CleanPaths>) -> Command {
    let mut cmd = match &job.launcher {
        Launcher::Binary(path) => Command::new(path),
        Launcher::Cargo => {
            let mut cmd = Command::new("cargo");
            cmd.args(["run", "--release", "-q", "-p", "vmpx-cli", "--"]);
            cmd
        }
    };
    cmd.arg(job.action.subcommand());
    cmd.arg(&job.input);
    match job.action {
        Action::Clean => {
            if let Some(paths) = clean_paths {
                cmd.arg(&paths.output);
                cmd.arg("--report").arg(&paths.report);
                if let Some(path) = &job.experimental_out {
                    cmd.arg("--experimental").arg(path);
                }
            }
        }
        Action::VmList => {
            cmd.arg("--json");
            if job.no_scan {
                cmd.arg("--no-scan");
            }
        }
        _ => {
            cmd.arg("--json");
        }
    }
    cmd
}

fn describe(cmd: &Command) -> String {
    let mut parts = vec![cmd.get_program().to_string_lossy().into_owned()];
    parts.extend(cmd.get_args().map(|arg| arg.to_string_lossy().into_owned()));
    parts.join(" ")
}

fn run_inner(job: &Job, result: &mut JobResult) -> Result<()> {
    let clean_paths = (job.action == Action::Clean).then(|| {
        let dir = std::env::temp_dir();
        let stamp = std::process::id();
        CleanPaths {
            output: dir.join(format!("vmpx-ui-{stamp}-clean.out.exe")),
            report: dir.join(format!("vmpx-ui-{stamp}-clean.report.json")),
        }
    });
    if let Some(paths) = &clean_paths {
        // A stale report from an earlier run must not be mistaken for this run's.
        let _ = std::fs::remove_file(&paths.report);
    }
    let mut cmd = command_for(job, clean_paths.as_ref());
    result.command_line = describe(&cmd);
    let output = cmd
        .output()
        .with_context(|| format!("failed to launch {}", result.command_line))?;
    result.exit_code = output.status.code();
    result.stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    result.stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    if let Some(paths) = &clean_paths {
        if let Ok(text) = std::fs::read_to_string(&paths.report) {
            result.json = serde_json::from_str(&text).ok();
            if result.json.is_none() {
                result.error = Some("clean report file was not valid JSON".into());
            }
        }
        let _ = std::fs::remove_file(&paths.report);
        let _ = std::fs::remove_file(&paths.output);
    } else {
        result.json = serde_json::from_str(&result.stdout).ok();
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
