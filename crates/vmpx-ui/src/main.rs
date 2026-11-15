//! Desktop front-end for the `vmpx` CLI. All analysis is done by the CLI binary;
//! this app only builds argument lists, runs the process and renders its output.

mod runner;
mod theme;
mod view;

use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::time::{Duration, Instant};

use eframe::egui::{self, Align, Layout, RichText};
use runner::{Action, GROUPS, Job, JobResult, Launcher, Options};
use theme::{ACCENT, BAD, BG_RAISED, BG_SIDE, OK, WARN};
use view::ResultTab;

struct Running {
    action: Action,
    started: Instant,
}

struct App {
    target: String,
    vmpx_bin: String,
    use_cargo: bool,
    bin_found: bool,
    action: Action,
    opts: Options,
    rx: Option<Receiver<JobResult>>,
    running: Option<Running>,
    last: Option<JobResult>,
    tab: ResultTab,
    note: String,
}

impl App {
    fn new() -> Self {
        let found = runner::find_release_binary();
        Self {
            target: String::new(),
            bin_found: found.is_some(),
            use_cargo: found.is_none(),
            vmpx_bin: found.map_or_else(
                || "target/release/vmpx.exe".into(),
                |path| path.display().to_string(),
            ),
            action: Action::Detect,
            opts: Options::default(),
            rx: None,
            running: None,
            last: None,
            tab: ResultTab::Summary,
            note: String::new(),
        }
    }

    fn launcher(&self) -> Launcher {
        if self.use_cargo {
            Launcher::Cargo
        } else {
            Launcher::Binary(self.vmpx_bin.trim().into())
        }
    }

    fn job(&self) -> Job {
        Job {
            action: self.action,
            launcher: self.launcher(),
            input: self.target.trim().into(),
            opts: self.opts.clone(),
        }
    }

    /// Why Run is unavailable, if it is.
    fn blocked(&self) -> Option<&'static str> {
        if self.running.is_some() {
            Some("a run is already in progress")
        } else if self.target.trim().is_empty() {
            Some("choose a target PE")
        } else if !self.use_cargo && self.vmpx_bin.trim().is_empty() {
            Some("set the vmpx binary or enable cargo")
        } else {
            self.opts.missing(self.action)
        }
    }

    fn start(&mut self, ctx: &egui::Context) {
        let job = self.job();
        let (tx, rx) = channel();
        let ctx = ctx.clone();
        self.running = Some(Running {
            action: job.action,
            started: Instant::now(),
        });
        self.rx = Some(rx);
        self.note.clear();
        std::thread::spawn(move || {
            let result = runner::run(&job);
            let _ = tx.send(result);
            ctx.request_repaint();
        });
    }

    fn finish(&mut self, result: JobResult) {
        self.tab = view::default_tab(&result);
        self.last = Some(result);
        self.rx = None;
        self.running = None;
    }

    fn poll(&mut self) {
        let Some(rx) = &self.rx else { return };
        match rx.try_recv() {
            Ok(result) => self.finish(result),
            Err(TryRecvError::Empty) => {}
            // The worker dropped its sender without a result: it panicked.
            Err(TryRecvError::Disconnected) => {
                if let Some(running) = self.running.take() {
                    self.finish(JobResult {
                        action: running.action,
                        command_line: String::new(),
                        exit_code: None,
                        stdout: String::new(),
                        stderr: String::new(),
                        json: None,
                        elapsed: Duration::ZERO,
                        error: Some("the worker thread stopped without a result".to_owned()),
                    });
                }
                self.rx = None;
            }
        }
    }

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("vmpx").size(21.0).strong());
            ui.label(RichText::new("VMProtect research console").weak());
        });
        ui.add_space(6.0);
        egui::Grid::new("globals")
            .num_columns(4)
            .spacing([10.0, 6.0])
            .show(ui, |ui| {
                let width = (ui.available_width() - 330.0).max(160.0);
                ui.label(RichText::new("Launcher").strong());
                ui.add_enabled(
                    !self.use_cargo,
                    egui::TextEdit::singleline(&mut self.vmpx_bin).desired_width(width),
                );
                ui.add_enabled_ui(!self.use_cargo, |ui| {
                    if ui.button("Browse").clicked()
                        && let Some(path) = rfd::FileDialog::new().pick_file()
                    {
                        self.vmpx_bin = path.display().to_string();
                        self.bin_found = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.use_cargo, "cargo run");
                    if self.use_cargo {
                        ui.colored_label(WARN, "recompiles per run");
                    } else if std::path::Path::new(self.vmpx_bin.trim()).is_file() {
                        ui.colored_label(OK, "binary found");
                    } else {
                        ui.colored_label(BAD, "binary missing");
                    }
                });
                ui.end_row();
            });
    }

    fn sidebar(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("vmpx").heading().strong().color(ACCENT));
        ui.label(RichText::new("VMProtect research").small().weak());
        ui.add_space(6.0);
        for (group, actions) in GROUPS {
            ui.add_space(8.0);
            ui.label(RichText::new(group).small().weak());
            for &action in actions {
                let selected = self.action == action;
                let text = RichText::new(action.label()).size(15.0);
                let text = if selected { text.strong() } else { text };
                let response = ui
                    .with_layout(Layout::top_down_justified(Align::LEFT), |ui| {
                        ui.selectable_label(selected, text)
                    })
                    .inner
                    .on_hover_text(action.description());
                if response.clicked() {
                    self.action = action;
                }
            }
        }
    }

    fn options_panel(&mut self, ui: &mut egui::Ui) {
        let action = self.action;

        // 1. The binary, first, because nothing else can be decided without it.
        theme::section(ui, "1. Target binary");
        ui.horizontal(|ui| {
            let width = (ui.available_width() - 130.0).max(200.0);
            ui.add(
                egui::TextEdit::singleline(&mut self.target)
                    .desired_width(width)
                    .hint_text("path to the protected image, or drop a file anywhere on this window"),
            );
            if ui.button("Browse...").clicked()
                && let Some(path) = rfd::FileDialog::new().pick_file()
            {
                self.target = path.display().to_string();
            }
        });

        theme::section(ui, "2. Command");
        ui.horizontal(|ui| {
            ui.label(RichText::new(action.label()).size(16.0).strong());
            ui.label(
                RichText::new(format!("vmpx {}", action.subcommand()))
                    .monospace()
                    .weak(),
            );
        });
        ui.label(RichText::new(action.description()).weak());

        theme::section(ui, "3. Options");
        let o = &mut self.opts;
        egui::Grid::new("options")
            .num_columns(3)
            .spacing([10.0, 8.0])
            .show(ui, |ui| match action {
                Action::Compare => {
                    ui.label("Second image");
                    let width = (ui.available_width() - 90.0).max(120.0);
                    ui.add(
                        egui::TextEdit::singleline(&mut o.compare_right)
                            .desired_width(width)
                            .hint_text("right-hand image (the target is the left)"),
                    );
                    if ui.button("Browse").clicked()
                        && let Some(path) = rfd::FileDialog::new().pick_file()
                    {
                        o.compare_right = path.display().to_string();
                    }
                    ui.end_row();
                }
                Action::DiffOriginal | Action::GenericUnpack | Action::Strip => {
                    let label = action.second_positional().unwrap_or("Output");
                    let save = action != Action::DiffOriginal;
                    theme::path_row(
                        ui,
                        label,
                        &mut o.second_output,
                        save,
                        if save {
                            "written by this command"
                        } else {
                            "the unprotected original of the same binary"
                        },
                    );
                    if action == Action::Strip {
                        theme::value_row(
                            ui,
                            "--remove",
                            &mut o.remove,
                            ".vmp0, .vmp1",
                            "section names to remove; at least one is required",
                        );
                        theme::value_row(
                            ui,
                            "--drop-directory",
                            &mut o.drop_directory,
                            "e.g. 9, 11",
                            "only a directory wholly inside a removed section may be dropped",
                        );
                    }
                    if action == Action::DiffOriginal {
                        theme::value_row(
                            ui,
                            "--function",
                            &mut o.function,
                            "export names",
                            "restrict to these exports; blank = all",
                        );
                        theme::value_row(
                            ui,
                            "--instruction-limit",
                            &mut o.instruction_limit,
                            "100000",
                            "entry-stream recovery budget",
                        );
                        theme::value_row(
                            ui,
                            "--exhaustive-bits",
                            &mut o.exhaustive_bits,
                            "16",
                            "free input bits the built-in backend enumerates",
                        );
                        theme::value_row(
                            ui,
                            "--timeout-ms",
                            &mut o.timeout_ms,
                            "60000",
                            "per-cell wall-clock limit per backend",
                        );
                        theme::value_row(
                            ui,
                            "--max-blocks",
                            &mut o.max_blocks,
                            "200000",
                            "block-execution budget per side",
                        );
                        theme::value_row(
                            ui,
                            "--max-paths",
                            &mut o.max_paths,
                            "64",
                            "path budget for the recovered side",
                        );
                        theme::flag_row(
                            ui,
                            "--strict-gates",
                            &mut o.strict_gates,
                            "treat excluded environment-gate arms as unproven",
                        );
                        theme::flag_row(
                            ui,
                            "--control",
                            &mut o.control,
                            "negative control: every verdict must come back REFUTED",
                        );
                    }
                }
                Action::Report => {
                    theme::flag_row(
                        ui,
                        "--all",
                        &mut o.all_entries,
                        "sweep every discovered entry; off uses --sample",
                    );
                    if !o.all_entries {
                        theme::value_row(
                            ui,
                            "--sample",
                            &mut o.sample,
                            "50",
                            "entries to sweep",
                        );
                    }
                    theme::value_row(
                        ui,
                        "--threads",
                        &mut o.threads,
                        "16",
                        "worker threads; default is logical CPUs capped at 16",
                    );
                    theme::flag_row(
                        ui,
                        "--serve-ro",
                        &mut o.serve_ro,
                        "serve read-only image bytes; raises N1, never counts toward N2",
                    );
                    theme::opt_row(
                        ui,
                        "--volatility-snapshot",
                        &mut o.volatility_snapshot,
                        "post-init VMPXSNP1 snapshot; build it with xpl_init_steps",
                    );
                    theme::flag_row(
                        ui,
                        "--serve-observed-ro",
                        &mut o.serve_observed_ro,
                        "needs the snapshot; carries ObservedGlobal",
                    );
                    theme::flag_row(
                        ui,
                        "--dynamic",
                        &mut o.dynamic,
                        "contained detonation + Observed-tier re-sweep; needs the snapshot",
                    );
                }
                Action::Decompile => {
                    theme::value_row(
                        ui,
                        "--rva",
                        &mut o.rva,
                        "0xc43d80e",
                        "VM entry RVAs, comma separated",
                    );
                    theme::value_row(
                        ui,
                        "--function",
                        &mut o.function,
                        "export names",
                        "restrict to these exports",
                    );
                    theme::value_row(
                        ui,
                        "--instruction-limit",
                        &mut o.instruction_limit,
                        "100000",
                        "entry-stream recovery budget",
                    );
                    theme::flag_row(
                        ui,
                        "--native",
                        &mut o.native,
                        "input is an unprotected PE; decompile its native CFG",
                    );
                    theme::flag_row(
                        ui,
                        "--probe",
                        &mut o.probe,
                        "print the recovered path/loop structure instead of code",
                    );
                    theme::opt_row(
                        ui,
                        "--original",
                        &mut o.original,
                        "unprotected original, to cross-check against",
                    );
                }
                Action::VerifyEquiv => {
                    theme::value_row(
                        ui,
                        "--rva",
                        &mut o.rva,
                        "0xc43d80e",
                        "VM entry RVAs, comma separated",
                    );
                    theme::value_row(
                        ui,
                        "--max-functions",
                        &mut o.max_functions,
                        "blank = all",
                        "stop after this many recovered functions",
                    );
                    theme::value_row(
                        ui,
                        "--exhaustive-bits",
                        &mut o.exhaustive_bits,
                        "16",
                        "free input bits the built-in backend enumerates",
                    );
                    theme::value_row(
                        ui,
                        "--timeout-ms",
                        &mut o.timeout_ms,
                        "10000",
                        "per-cell wall-clock limit per backend",
                    );
                    theme::value_row(
                        ui,
                        "--instruction-limit",
                        &mut o.instruction_limit,
                        "512",
                        "entry-stream recovery budget",
                    );
                    theme::flag_row(
                        ui,
                        "--solver",
                        &mut o.solver,
                        "also consult z3/bitwuzla/cvc5 from PATH",
                    );
                    theme::flag_row(
                        ui,
                        "--no-optimize",
                        &mut o.no_optimize,
                        "check the unoptimized lowering",
                    );
                    theme::flag_row(
                        ui,
                        "--strict-memory",
                        &mut o.strict_memory,
                        "a load after a possibly-aliasing store counts as unproven",
                    );
                    theme::flag_row(ui, "--verbose", &mut o.verbose, "one line per function");
                }
                Action::Clean => {
                    theme::path_row(
                        ui,
                        "Output",
                        &mut o.clean_output,
                        true,
                        "blank = scratch file, discarded after the run",
                    );
                    theme::opt_row(ui, "--report", &mut o.clean_report, "recovery report JSON");
                    theme::opt_row(
                        ui,
                        "--reconstructed",
                        &mut o.reconstructed,
                        "conventional-layout image, not devirtualized",
                    );
                    theme::opt_row(
                        ui,
                        "--experimental",
                        &mut o.experimental,
                        "unverified devirtualized image",
                    );
                    theme::flag_row(
                        ui,
                        "--decoded-lift",
                        &mut o.decoded_lift,
                        "with --experimental; lift from decoded handler streams (slow)",
                    );
                    theme::flag_row(
                        ui,
                        "--dynamic",
                        &mut o.clean_dynamic,
                        "with --experimental; feed Observed-tier resolved imports",
                    );
                    theme::opt_row(
                        ui,
                        "--dump-pe",
                        &mut o.dump_pe,
                        "with --dynamic; standalone PE from the detonation dump",
                    );
                    theme::flag_row(
                        ui,
                        "--strip",
                        &mut o.clean_strip,
                        "with --experimental; strip protector sections if the checker approves",
                    );
                }
                Action::Unpack => {
                    theme::path_row(
                        ui,
                        "Output",
                        &mut o.unpack_output,
                        true,
                        "blank = scratch file, discarded after the run",
                    );
                    theme::opt_row(
                        ui,
                        "--report",
                        &mut o.unpack_report,
                        "unpacking report JSON",
                    );
                    theme::opt_row(
                        ui,
                        "--conventional",
                        &mut o.conventional,
                        "loader-compatible file layout",
                    );
                }
                Action::VmList => {
                    ui.checkbox(&mut o.no_scan, "--no-scan")
                        .on_hover_text("Restrict discovery to PE entry and TLS roots.");
                    ui.label(RichText::new("skip the executable stub scan").weak());
                    ui.end_row();
                }
                _ => {}
            });
        if matches!(
            action,
            Action::Detect
                | Action::Info
                | Action::Verify
                | Action::Compare
                | Action::Imports
                | Action::VmList
                | Action::Oep
                | Action::Report
        ) {
            theme::section(ui, "4. Output style");
            ui.horizontal(|ui| {
                ui.radio_value(&mut o.json, true, "Structured (--json)");
                ui.radio_value(&mut o.json, false, "Plain text report");
            });
            ui.label(
                RichText::new(
                    "Structured output is parsed into the summary below; plain text is shown as the CLI printed it. Either can be saved from the results pane.",
                )
                .size(12.5)
                .weak(),
            );
        } else {
            theme::section(ui, "4. Output style");
            ui.label(
                RichText::new(
                    "This command writes files rather than a report. Its destinations are set above; the run log is shown below and can be saved from the results pane.",
                )
                .size(12.5)
                .weak(),
            );
        }
        // The environment is where most of the behaviour lives, so it is shown
        // beside the flags rather than left to the shell.
        theme::group(ui, "5. Environment", |ui| {
            ui.label(
                RichText::new(
                    "Tick a variable to pass it to the run. It is included in the command line below.",
                )
                .size(12.5)
                .weak(),
            );
            ui.add_space(6.0);
            // Fixed sizes, not desired widths: in a grid the help column's long
            // text competes for space and squeezes the name field down to
            // "VMPX_". `add_sized` pins the two inputs and the help label
            // truncates into whatever is left.
            egui::Grid::new("env")
                .num_columns(4)
                .spacing([10.0, 6.0])
                .show(ui, |ui| {
                    for row in &mut o.env {
                        ui.checkbox(&mut row.enabled, "");
                        ui.add_sized(
                            [290.0, 26.0],
                            egui::TextEdit::singleline(&mut row.name).hint_text("VMPX_..."),
                        );
                        ui.add_sized(
                            [120.0, 26.0],
                            egui::TextEdit::singleline(&mut row.value).hint_text("value"),
                        );
                        ui.add(
                            egui::Label::new(RichText::new(row.help).size(12.5).weak())
                                .truncate(),
                        )
                        .on_hover_text(row.help);
                        ui.end_row();
                    }
                });
        });
        if matches!(action, Action::Unpack | Action::Clean) {
            ui.label(
                RichText::new(
                    "The summary is read from the report file; --report only keeps a copy.",
                )
                .small()
                .weak(),
            );
        }
        ui.add_space(6.0);
        self.run_row(ui);
    }

    fn run_row(&mut self, ui: &mut egui::Ui) {
        let blocked = self.blocked();
        theme::section(ui, "6. Run");
        ui.horizontal(|ui| {
            let run = egui::Button::new(
                RichText::new("Run")
                    .size(15.0)
                    .strong()
                    .color(egui::Color32::WHITE),
            )
            .fill(ACCENT)
            .min_size(egui::vec2(120.0, 34.0));
            let response = ui.add_enabled(blocked.is_none(), run);
            let hotkey = ui.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Enter));
            if blocked.is_none() && (response.clicked() || hotkey) {
                self.start(ui.ctx());
            }
            if let Some(running) = &self.running {
                ui.spinner();
                ui.label(format!(
                    "running {} ... {:.1}s",
                    running.action.label(),
                    running.started.elapsed().as_secs_f64()
                ));
                ui.ctx().request_repaint_after(Duration::from_millis(100));
            } else if let Some(reason) = blocked {
                ui.label(RichText::new(reason).color(WARN));
            } else if let Some(last) = &self.last {
                ui.label(format!(
                    "last: {} in {:.2}s",
                    last.action.label(),
                    last.elapsed.as_secs_f64()
                ));
            }
            if !self.note.is_empty() {
                ui.label(RichText::new(&self.note).weak());
            }
        });
        let line = runner::command_line(&self.job());
        ui.horizontal(|ui| {
            ui.label(RichText::new("Command").small().weak());
            if ui.small_button("Copy").clicked() {
                ui.ctx().copy_text(line.clone());
                self.note = "command copied".into();
            }
        });
        egui::Frame::none()
            .fill(egui::Color32::from_rgb(249, 249, 249))
            .stroke(egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(214, 214, 214)))
            .rounding(4.0)
            .inner_margin(10.0)
            .show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut line.as_str())
                        .font(egui::TextStyle::Monospace)
                        .desired_width(f32::INFINITY)
                        .desired_rows(1)
                        .frame(false)
                        .interactive(true),
                );
            });
        ui.add_space(4.0);
    }

    fn results(&mut self, ui: &mut egui::Ui) {
        if let Some(running) = &self.running {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(format!("running {}...", running.action.label()));
            });
            return;
        }
        if let Some(result) = &self.last {
            if let Some(note) = view::show(ui, result, &mut self.tab) {
                self.note = note;
            }
        } else {
            ui.add_space(24.0);
            ui.vertical_centered(|ui| {
                ui.label(RichText::new("No results yet").heading().weak());
                ui.label(
                    RichText::new("Pick a target PE and a command, then press Run (Ctrl+Enter).")
                        .weak(),
                );
            });
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll();
        if let Some(path) = ctx.input(|i| i.raw.dropped_files.first().and_then(|f| f.path.clone()))
        {
            self.target = path.display().to_string();
        }

        egui::TopBottomPanel::top("globals")
            .frame(theme::panel(BG_RAISED, 14.0, 10.0))
            .show(ctx, |ui| self.top_bar(ui));
        egui::SidePanel::left("commands")
            .resizable(false)
            .exact_width(210.0)
            .frame(theme::panel(BG_SIDE, 12.0, 12.0))
            .show(ctx, |ui| self.sidebar(ui));
        // The control panel is a single top-to-bottom sequence — binary, command,
        // options, output, environment, run — so it scrolls as one page rather
        // than being clipped by a fixed-height panel.
        egui::TopBottomPanel::top("options")
            .resizable(true)
            .default_height(ctx.screen_rect().height() * 0.62)
            .frame(theme::panel(theme::BG_MAIN, 20.0, 12.0))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| self.options_panel(ui));
            });
        egui::TopBottomPanel::bottom("status")
            .frame(theme::panel(theme::BG_MAIN, 6.0, 3.0))
            .show(ctx, |ui| {
                let state = if self.running.is_some() {
                    "Running...".to_owned()
                } else if let Some(why) = self.blocked() {
                    why.to_owned()
                } else {
                    "Ready".to_owned()
                };
                let set = self
                    .opts
                    .env
                    .iter()
                    .filter(|row| row.enabled && !row.name.trim().is_empty())
                    .count();
                let env = format!("{set} env set");
                let cmd = format!("vmpx {}", self.action.subcommand());
                theme::status_bar(ui, &[(&state, 0.0), (&cmd, 150.0), (&env, 86.0)]);
            });
        egui::CentralPanel::default()
            .frame(theme::panel(theme::BG_MAIN, 16.0, 6.0))
            .show(ctx, |ui| self.results(ui));
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1180.0, 780.0])
            .with_min_inner_size([820.0, 520.0])
            .with_title("vmpx"),
        ..Default::default()
    };
    eframe::run_native(
        "vmpx",
        options,
        Box::new(|cc| {
            theme::apply(&cc.egui_ctx);
            Ok(Box::new(App::new()))
        }),
    )
}
