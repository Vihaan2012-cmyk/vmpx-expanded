use eframe::egui::{self, Color32, RichText, Ui};
use serde_json::Value;

use crate::runner::{Action, JobResult};
use crate::theme::{ACCENT, BAD, OK};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResultTab {
    Summary,
    RawJson,
    Stdout,
    Stderr,
}

pub fn default_tab(result: &JobResult) -> ResultTab {
    if result.json.is_some() {
        ResultTab::Summary
    } else if result.stdout.trim().is_empty() && !result.stderr.trim().is_empty() {
        ResultTab::Stderr
    } else {
        ResultTab::Stdout
    }
}

/// The text the Copy and Save buttons act on for a tab.
fn tab_text(result: &JobResult, tab: ResultTab) -> String {
    match (tab, &result.json) {
        (ResultTab::Summary | ResultTab::RawJson, Some(json)) => pretty(json),
        (ResultTab::Stderr, _) => result.stderr.clone(),
        _ => result.stdout.clone(),
    }
}

fn banner(ui: &mut Ui, result: &JobResult) {
    egui::Frame::none()
        .fill(Color32::from_rgb(58, 28, 30))
        .stroke(egui::Stroke::new(1.0_f32, BAD))
        .rounding(5.0)
        .inner_margin(10.0)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            let title = match (&result.error, result.exit_code) {
                (Some(error), _) => error.clone(),
                (None, Some(code)) => format!("Process exited with code {code}"),
                (None, None) => "Process ended without an exit code".to_owned(),
            };
            ui.label(RichText::new(title).strong().color(BAD));
            let stderr = result.stderr.trim();
            if !stderr.is_empty() {
                let head: Vec<&str> = stderr.lines().take(8).collect();
                ui.label(
                    RichText::new(head.join(
                        "
",
                    ))
                    .monospace(),
                );
                if stderr.lines().count() > head.len() {
                    ui.weak("... see the Stderr tab for the rest");
                }
            }
        });
    ui.add_space(4.0);
}

/// Returns a status note after a copy or save.
pub fn show(ui: &mut Ui, result: &JobResult, tab: &mut ResultTab) -> Option<String> {
    let mut note = None;
    ui.horizontal(|ui| {
        match result.exit_code {
            Some(0) => ui.colored_label(OK, "exit 0"),
            Some(code) => ui.colored_label(BAD, format!("exit {code}")),
            None => ui.colored_label(BAD, "no exit code"),
        };
        ui.label(format!("{:.2}s", result.elapsed.as_secs_f64()));
        ui.separator();
        ui.selectable_value(tab, ResultTab::Summary, "Summary");
        if result.json.is_some() {
            ui.selectable_value(tab, ResultTab::RawJson, "Raw JSON");
        }
        ui.selectable_value(tab, ResultTab::Stdout, "Stdout");
        let stderr = if result.stderr.trim().is_empty() {
            RichText::new("Stderr")
        } else {
            RichText::new("Stderr").color(BAD)
        };
        ui.selectable_value(tab, ResultTab::Stderr, stderr);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("Save").clicked() {
                note = save(result, *tab);
            }
            if ui.button("Copy").clicked() {
                let text = tab_text(result, *tab);
                note = Some(format!("copied {} bytes", text.len()));
                ui.ctx().copy_text(text);
            }
        });
    });
    if *tab == ResultTab::RawJson && result.json.is_none() {
        *tab = ResultTab::Stdout;
    }
    ui.separator();
    if result.failed() {
        banner(ui, result);
    }

    egui::ScrollArea::both()
        .auto_shrink([false; 2])
        .show(ui, |ui| match *tab {
            ResultTab::Summary => match &result.json {
                Some(json) => summary(ui, result.action, json),
                None => text_or_empty(ui, &result.stdout, "No structured output."),
            },
            ResultTab::RawJson => match &result.json {
                Some(json) => raw(ui, &pretty(json)),
                None => text_or_empty(ui, &result.stdout, "No JSON output."),
            },
            ResultTab::Stdout => text_or_empty(ui, &result.stdout, "No stdout."),
            ResultTab::Stderr => text_or_empty(ui, &result.stderr, "No stderr."),
        });
    note
}

fn save(result: &JobResult, tab: ResultTab) -> Option<String> {
    let json_tab = matches!(tab, ResultTab::Summary | ResultTab::RawJson) && result.json.is_some();
    let (name, filter) = if json_tab {
        (format!("vmpx-{}.json", result.action.subcommand()), "json")
    } else {
        (format!("vmpx-{}.txt", result.action.subcommand()), "txt")
    };
    let path = rfd::FileDialog::new()
        .set_file_name(name)
        .add_filter(filter, &[filter])
        .save_file()?;
    Some(match std::fs::write(&path, tab_text(result, tab)) {
        Ok(()) => format!("saved {}", path.display()),
        Err(error) => format!("save failed: {error}"),
    })
}

fn text_or_empty(ui: &mut Ui, text: &str, empty: &str) {
    if text.trim().is_empty() {
        ui.weak(empty);
    } else {
        raw(ui, text);
    }
}

fn pretty(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_default()
}

fn raw(ui: &mut Ui, text: &str) {
    let mut text = text;
    ui.add(
        egui::TextEdit::multiline(&mut text)
            .font(egui::TextStyle::Monospace)
            .desired_width(f32::INFINITY),
    );
}

fn scalar(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn field(ui: &mut Ui, label: &str, value: Option<&Value>) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("{label}:")).strong());
        ui.label(value.map_or_else(|| "-".into(), scalar));
    });
}

fn items(value: Option<&Value>) -> &[Value] {
    value.and_then(Value::as_array).map_or(&[], Vec::as_slice)
}

fn list(ui: &mut Ui, title: &str, value: Option<&Value>) {
    let entries = items(value);
    ui.collapsing(format!("{title} ({})", entries.len()), |ui| {
        if entries.is_empty() {
            ui.weak("none");
        }
        for item in entries {
            ui.label(format!("- {}", scalar(item)));
        }
    });
}

fn summary(ui: &mut Ui, action: Action, json: &Value) {
    match action {
        Action::Detect => protection(ui, json),
        Action::Clean => clean(ui, json),
        Action::Compare => compare(ui, json),
        Action::VmList => vm_list(ui, json),
        _ => generic(ui, json),
    }
}

fn protection(ui: &mut Ui, json: &Value) {
    field(ui, "Verdict", json.get("verdict"));
    field(ui, "Confidence", json.get("confidence"));
    field(ui, "Image layout", json.get("image_layout"));
    field(ui, "Entry point", json.get("entry_point"));
    let evidence = items(json.get("evidence"));
    ui.collapsing(format!("Evidence ({})", evidence.len()), |ui| {
        for item in evidence {
            ui.label(format!(
                "[{}] w={}  {}",
                item.get("id").map_or_else(String::new, scalar),
                item.get("weight").map_or_else(String::new, scalar),
                item.get("statement").map_or_else(String::new, scalar),
            ));
        }
    });
    list(ui, "Warnings", json.get("warnings"));
    let regions = items(json.get("executable_regions"));
    ui.collapsing(format!("Executable regions ({})", regions.len()), |ui| {
        for region in regions {
            ui.monospace(region.to_string());
        }
    });
}

fn count(json: &Value, key: &str) -> String {
    json.get(key)
        .and_then(Value::as_array)
        .map_or_else(|| "-".into(), |a| a.len().to_string())
}

fn stage(ui: &mut Ui, name: &str, json: &Value) {
    let Some(stage) = json.get(name) else { return };
    let (color, text) = match stage.get("complete").and_then(Value::as_bool) {
        Some(true) => (OK, "complete"),
        Some(false) => (BAD, "incomplete"),
        None => (Color32::GRAY, "unknown"),
    };
    ui.collapsing(
        RichText::new(format!("{name}: {text}")).color(color),
        |ui| {
            if let Some(metrics) = stage.get("metrics").and_then(Value::as_object) {
                for (key, value) in metrics {
                    field(ui, key, Some(value));
                }
            }
            list(ui, "Diagnostics", stage.get("diagnostics"));
        },
    );
}

fn clean(ui: &mut Ui, json: &Value) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("Ready to write:").strong());
        if json.get("ready_to_write").and_then(Value::as_bool) == Some(true) {
            ui.colored_label(OK, "yes");
        } else {
            ui.colored_label(BAD, "no");
        }
    });
    field(ui, "Experimental", json.get("experimental"));
    if let Some(report) = json.get("protection") {
        egui::CollapsingHeader::new("Protection verdict")
            .default_open(true)
            .show(ui, |ui| protection(ui, report));
    }
    egui::CollapsingHeader::new("Counts")
        .default_open(true)
        .show(ui, |ui| {
            ui.label(format!("vm_entries: {}", count(json, "vm_entries")));
            ui.label(format!("vm_instances: {}", count(json, "vm_instances")));
            ui.label(format!(
                "virtualized_routines: {}",
                count(json, "virtualized_routines")
            ));
            if let Some(handlers) = json.get("handlers") {
                for key in ["unique", "classified", "unknown"] {
                    field(ui, &format!("handlers.{key}"), handlers.get(key));
                }
            }
        });
    let blockers = items(json.get("blockers"));
    let color = if blockers.is_empty() { OK } else { BAD };
    egui::CollapsingHeader::new(
        RichText::new(format!("Blockers ({})", blockers.len())).color(color),
    )
    .default_open(!blockers.is_empty())
    .show(ui, |ui| {
        if blockers.is_empty() {
            ui.weak("none");
        }
        for blocker in blockers {
            ui.label(format!("- {}", scalar(blocker)));
        }
    });
    sp2_sections(ui, json);
    egui::CollapsingHeader::new("Stages")
        .default_open(true)
        .show(ui, |ui| {
            for name in ["vtil", "recompilation", "reintegration", "reconstruction"] {
                stage(ui, name, json);
            }
        });
    if let Some(validation) = json.get("validation").and_then(Value::as_object) {
        ui.collapsing("Validation", |ui| {
            for (key, value) in validation {
                field(ui, key, Some(value));
            }
        });
    }
    list(ui, "Warnings", json.get("warnings"));
    list(ui, "Errors", json.get("errors"));
}

fn rva_text(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => "-".into(),
        Some(Value::Number(n)) => n
            .as_u64()
            .map_or_else(|| n.to_string(), |v| format!("0x{v:08X}")),
        Some(other) => scalar(other),
    }
}

fn metric(stage: Option<&Value>, key: &str) -> Option<u64> {
    stage?.get("metrics")?.get(key)?.as_u64()
}

/// Sections for fields added in report schema 2; each degrades to nothing or
/// "-" when the field is absent.
fn sp2_sections(ui: &mut Ui, json: &Value) {
    if let Some(oep) = json.get("oep") {
        egui::CollapsingHeader::new("OEP")
            .default_open(true)
            .show(ui, |ui| {
                let recovered = oep.get("recovered_entry").filter(|v| !v.is_null());
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Recovered entry:").strong());
                    match recovered {
                        Some(v) => ui.colored_label(OK, rva_text(Some(v))),
                        None => ui.colored_label(BAD, "not recovered"),
                    };
                });
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Complete:").strong());
                    match oep.get("complete").and_then(Value::as_bool) {
                        Some(true) => ui.colored_label(OK, "yes"),
                        Some(false) => ui.colored_label(BAD, "no"),
                        None => ui.weak("-"),
                    };
                });
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Protected header entry:").strong());
                    ui.label(rva_text(oep.get("protected_header_entry")));
                });
            });
    }
    if json.get("vm_inventory").is_some() {
        vm_inventory(ui, items(json.get("vm_inventory")));
    }
    if let Some(stage) = json.get("input_closed_functions") {
        let recovered = metric(Some(stage), "recovered");
        let title = format!(
            "Input-closed functions (recovered: {})",
            recovered.map_or_else(|| "-".into(), |n| n.to_string())
        );
        egui::CollapsingHeader::new(title)
            .default_open(recovered.unwrap_or(0) > 0)
            .show(ui, |ui| {
                if let Some(metrics) = stage.get("metrics").and_then(Value::as_object) {
                    for (key, value) in metrics {
                        field(ui, key, Some(value));
                    }
                }
                list(ui, "Per-entry diagnostics", stage.get("diagnostics"));
            });
    }
    if let Some(stage) = json.get("reintegration") {
        egui::CollapsingHeader::new("Reintegration metrics")
            .default_open(true)
            .show(ui, |ui| {
                let metrics = stage.get("metrics").and_then(Value::as_object);
                if metrics.is_none_or(serde_json::Map::is_empty) {
                    ui.weak("no metrics");
                }
                for (key, value) in metrics.into_iter().flatten() {
                    if key == "section_rva" {
                        field(ui, key, Some(&Value::String(rva_text(Some(value)))));
                    } else {
                        field(ui, key, Some(value));
                    }
                }
            });
    }
}

fn vm_inventory(ui: &mut Ui, entries: &[Value]) {
    let exported = entries
        .iter()
        .filter(|e| e.get("reachable_from_export").and_then(Value::as_bool) == Some(true))
        .count();
    egui::CollapsingHeader::new(format!(
        "VM-entry inventory ({}, {exported} export-reachable)",
        entries.len()
    ))
    .default_open(true)
    .show(ui, |ui| {
        if entries.is_empty() {
            ui.weak("none");
            return;
        }
        egui::Grid::new("vm_inventory_grid")
            .striped(true)
            .num_columns(4)
            .show(ui, |ui| {
                for head in ["rva", "confidence", "sources", "reachable_from_export"] {
                    ui.strong(head);
                }
                ui.end_row();
                for entry in entries {
                    ui.monospace(rva_text(entry.get("rva")));
                    ui.label(entry.get("confidence").map_or_else(|| "-".into(), scalar));
                    let sources = items(entry.get("sources"))
                        .iter()
                        .map(scalar)
                        .collect::<Vec<_>>()
                        .join(", ");
                    ui.label(if sources.is_empty() {
                        "-".into()
                    } else {
                        sources
                    });
                    match entry.get("reachable_from_export").and_then(Value::as_bool) {
                        Some(true) => ui.colored_label(OK, "yes"),
                        Some(false) => ui.label("no"),
                        None => ui.weak("-"),
                    };
                    ui.end_row();
                }
            });
    });
}

fn vm_list(ui: &mut Ui, json: &Value) {
    field(
        ui,
        "Scanned stub candidates",
        json.get("scanned_stub_candidates"),
    );
    field(ui, "Resource limited", json.get("resource_limited"));
    for (key, title) in [
        ("entries", "Entries"),
        ("instances", "Instances"),
        ("rejected_roots", "Rejected roots"),
    ] {
        let entries = items(json.get(key));
        egui::CollapsingHeader::new(format!("{title} ({})", entries.len()))
            .default_open(key == "entries")
            .show(ui, |ui| {
                for (index, item) in entries.iter().enumerate() {
                    ui.collapsing(format!("#{index}"), |ui| raw(ui, &pretty(item)));
                }
            });
    }
    list(ui, "Diagnostics", json.get("diagnostics"));
}

fn compare(ui: &mut Ui, json: &Value) {
    let differences = items(json.get("differences"));
    let color = if differences.is_empty() { OK } else { ACCENT };
    egui::CollapsingHeader::new(
        RichText::new(format!("Differences ({})", differences.len())).color(color),
    )
    .default_open(true)
    .show(ui, |ui| {
        if differences.is_empty() {
            ui.weak("no modeled structural differences");
        }
        for difference in differences {
            ui.monospace(scalar(difference));
        }
    });
    let (left, right) = (json.get("left"), json.get("right"));
    let keys: Vec<&String> = left
        .and_then(Value::as_object)
        .map(|o| o.keys().collect())
        .unwrap_or_default();
    egui::CollapsingHeader::new("Side by side")
        .default_open(true)
        .show(ui, |ui| {
            egui::Grid::new("compare_grid")
                .striped(true)
                .show(ui, |ui| {
                    ui.strong("field");
                    ui.strong("left");
                    ui.strong("right");
                    ui.end_row();
                    for key in keys {
                        let (l, r) = (
                            left.and_then(|v| v.get(key)),
                            right.and_then(|v| v.get(key)),
                        );
                        ui.label(key);
                        ui.monospace(l.map_or_else(|| "-".into(), scalar));
                        let same = l == r;
                        let text = RichText::new(r.map_or_else(|| "-".into(), scalar)).monospace();
                        ui.label(if same { text } else { text.color(ACCENT) });
                        ui.end_row();
                    }
                });
        });
}

/// Top-level scalars inline; arrays and objects as collapsible sections.
fn generic(ui: &mut Ui, json: &Value) {
    let Some(object) = json.as_object() else {
        raw(ui, &pretty(json));
        return;
    };
    for (key, value) in object {
        match value {
            Value::Array(entries) => {
                ui.collapsing(format!("{key} ({})", entries.len()), |ui| {
                    raw(ui, &pretty(value));
                });
            }
            Value::Object(_) => {
                ui.collapsing(key, |ui| raw(ui, &pretty(value)));
            }
            other => field(ui, key, Some(other)),
        }
    }
}
