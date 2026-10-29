//! Desktop front-end for the `vmpx` CLI. All analysis is done by the CLI binary;
//! this app only builds argument lists, runs the process and renders its output.

mod runner;
mod view;

use std::sync::mpsc::{Receiver, channel};

use eframe::egui;
use runner::{Action, Job, JobResult, Launcher};

struct App {
    target: String,
    vmpx_bin: String,
    use_cargo: bool,
    experimental: bool,
    experimental_out: String,
    no_scan: bool,
    rx: Option<Receiver<JobResult>>,
    running: Option<String>,
    last: Option<JobResult>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            target: String::new(),
            vmpx_bin: "target/release/vmpx.exe".into(),
            use_cargo: false,
            experimental: false,
            experimental_out: String::new(),
            no_scan: false,
            rx: None,
            running: None,
            last: None,
        }
    }
}

impl App {
    fn launcher(&self) -> Launcher {
        if self.use_cargo {
            Launcher::Cargo
        } else {
            Launcher::Binary(self.vmpx_bin.trim().into())
        }
    }

    fn start(&mut self, ctx: &egui::Context, action: Action) {
        let job = Job {
            action,
            launcher: self.launcher(),
            input: self.target.trim().into(),
            no_scan: self.no_scan,
            experimental_out: (self.experimental && action == Action::Clean)
                .then(|| self.experimental_out.trim().to_owned()),
        };
        let (tx, rx) = channel();
        let ctx = ctx.clone();
        self.running = Some(action.label().to_owned());
        self.rx = Some(rx);
        std::thread::spawn(move || {
            let result = runner::run(&job);
            let _ = tx.send(result);
            ctx.request_repaint();
        });
    }

    fn poll(&mut self) {
        let Some(rx) = &self.rx else { return };
        if let Ok(result) = rx.try_recv() {
            self.last = Some(result);
            self.rx = None;
            self.running = None;
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("vmpx");
        ui.separator();
        ui.label("Target PE");
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut self.target).desired_width(190.0));
            if ui.button("...").clicked()
                && let Some(path) = rfd::FileDialog::new().pick_file()
            {
                self.target = path.display().to_string();
            }
        });
        ui.add_space(6.0);
        ui.checkbox(&mut self.use_cargo, "run via cargo run -p vmpx-cli --");
        ui.add_enabled_ui(!self.use_cargo, |ui| {
            ui.label("vmpx binary");
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.vmpx_bin).desired_width(190.0));
                if ui.button("...").clicked()
                    && let Some(path) = rfd::FileDialog::new().pick_file()
                {
                    self.vmpx_bin = path.display().to_string();
                }
            });
        });
        ui.separator();

        let idle = self.running.is_none();
        let ready = idle && !self.target.trim().is_empty();
        ui.label("Analyze");
        ui.checkbox(&mut self.no_scan, "vm-list: --no-scan");
        for action in [
            Action::Detect,
            Action::Info,
            Action::Verify,
            Action::Imports,
            Action::VmList,
            Action::Oep,
        ] {
            if ui
                .add_enabled(ready, egui::Button::new(action.label()))
                .clicked()
            {
                self.start(ui.ctx(), action);
            }
        }
        ui.separator();
        ui.label("Clean");
        ui.checkbox(&mut self.experimental, "experimental");
        let mut clean_ready = ready;
        if self.experimental {
            ui.label("Experimental output path");
            ui.horizontal(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut self.experimental_out).desired_width(190.0),
                );
                if ui.button("...").clicked()
                    && let Some(path) = rfd::FileDialog::new().save_file()
                {
                    self.experimental_out = path.display().to_string();
                }
            });
            clean_ready &= !self.experimental_out.trim().is_empty();
        }
        if ui
            .add_enabled(clean_ready, egui::Button::new("Clean"))
            .clicked()
        {
            self.start(ui.ctx(), Action::Clean);
        }
        ui.add_space(8.0);
        if let Some(label) = &self.running {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(format!("running {label}..."));
            });
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll();
        egui::SidePanel::left("controls")
            .min_width(260.0)
            .show(ctx, |ui| self.controls(ui));

        if let Some(result) = &self.last
            && !result.stderr.trim().is_empty()
        {
            egui::TopBottomPanel::bottom("stderr")
                .resizable(true)
                .min_height(60.0)
                .show(ctx, |ui| {
                    ui.colored_label(egui::Color32::LIGHT_RED, "stderr");
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut result.stderr.as_str())
                                .font(egui::TextStyle::Monospace)
                                .desired_width(f32::INFINITY),
                        );
                    });
                });
        }

        egui::CentralPanel::default().show(ctx, |ui| match (&self.last, &self.running) {
            (_, Some(label)) => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(format!("running {label}..."));
                });
            }
            (Some(result), None) => view::show(ui, result),
            (None, None) => {
                ui.label("Choose a target PE and run an action.");
            }
        });
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 720.0]),
        ..Default::default()
    };
    eframe::run_native("vmpx", options, Box::new(|_cc| Ok(Box::<App>::default())))
}
