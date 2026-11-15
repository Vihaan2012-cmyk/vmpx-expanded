//! Modern Windows desktop look: white surfaces, hairline borders, a single blue
//! accent, and type large enough to read at arm's length.
//!
//! The target is a current WinUI-style utility window — a titled page, bold
//! section headings, bordered list and log boxes, wide full-label controls — not
//! a dense tool palette.

use eframe::egui::{
    self, Color32, Context, FontFamily, FontId, Frame, Margin, RichText, Rounding, Stroke,
    TextStyle, Ui, Vec2,
};

use crate::runner::PathOpt;

const SURFACE: Color32 = Color32::from_rgb(255, 255, 255);
const SURFACE_ALT: Color32 = Color32::from_rgb(249, 249, 249);
const SURFACE_SUNK: Color32 = Color32::from_rgb(243, 243, 243);
const BORDER: Color32 = Color32::from_rgb(214, 214, 214);
const BORDER_STRONG: Color32 = Color32::from_rgb(186, 186, 186);
const INK: Color32 = Color32::from_rgb(27, 27, 27);
const INK_SOFT: Color32 = Color32::from_rgb(98, 98, 98);
const BLUE: Color32 = Color32::from_rgb(0, 103, 192);
const BLUE_SOFT: Color32 = Color32::from_rgb(222, 236, 249);

pub const ACCENT: Color32 = BLUE;
pub const OK: Color32 = Color32::from_rgb(16, 124, 16);
pub const BAD: Color32 = Color32::from_rgb(196, 43, 28);
pub const WARN: Color32 = Color32::from_rgb(157, 93, 0);
pub const BG_SIDE: Color32 = SURFACE_ALT;
pub const BG_MAIN: Color32 = SURFACE;
pub const BG_RAISED: Color32 = SURFACE;

const RADIUS: f32 = 4.0;

/// Uses Segoe UI when Windows has it, so the window matches the rest of the
/// desktop rather than egui's bundled face.
fn install_ui_font(ctx: &Context) {
    const CANDIDATES: [&str; 3] = [
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\SegoeUI.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
    ];
    for path in CANDIDATES {
        if let Ok(bytes) = std::fs::read(path) {
            let mut fonts = egui::FontDefinitions::default();
            fonts
                .font_data
                .insert("ui".to_owned(), egui::FontData::from_owned(bytes));
            fonts
                .families
                .entry(FontFamily::Proportional)
                .or_default()
                .insert(0, "ui".to_owned());
            ctx.set_fonts(fonts);
            return;
        }
    }
}

pub fn apply(ctx: &Context) {
    install_ui_font(ctx);

    let mut v = egui::Visuals::light();
    v.panel_fill = SURFACE;
    v.window_fill = SURFACE;
    v.extreme_bg_color = SURFACE;
    v.faint_bg_color = SURFACE_ALT;
    v.override_text_color = Some(INK);
    v.selection.bg_fill = BLUE_SOFT;
    v.selection.stroke = Stroke::new(1.0_f32, BLUE);
    v.hyperlink_color = BLUE;
    v.window_rounding = Rounding::same(8.0);
    v.menu_rounding = Rounding::same(6.0);
    v.window_stroke = Stroke::new(1.0_f32, BORDER);

    let radius = Rounding::same(RADIUS);
    for widget in [
        &mut v.widgets.noninteractive,
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        widget.rounding = radius;
        widget.fg_stroke = Stroke::new(1.0_f32, INK);
        widget.expansion = 0.0_f32;
    }
    // Buttons and fields: white, hairline border, a touch darker on press.
    v.widgets.noninteractive.bg_fill = SURFACE;
    v.widgets.noninteractive.weak_bg_fill = SURFACE;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, BORDER);
    v.widgets.inactive.bg_fill = SURFACE;
    v.widgets.inactive.weak_bg_fill = SURFACE;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, BORDER);
    v.widgets.hovered.bg_fill = SURFACE_ALT;
    v.widgets.hovered.weak_bg_fill = SURFACE_ALT;
    v.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, BORDER_STRONG);
    v.widgets.active.bg_fill = SURFACE_SUNK;
    v.widgets.active.weak_bg_fill = SURFACE_SUNK;
    v.widgets.active.bg_stroke = Stroke::new(1.0_f32, BORDER_STRONG);
    v.widgets.open.bg_fill = SURFACE_ALT;
    v.widgets.open.weak_bg_fill = SURFACE_ALT;
    v.widgets.open.bg_stroke = Stroke::new(1.0_f32, BORDER);
    ctx.set_visuals(v);

    ctx.style_mut(|style| {
        style.spacing.item_spacing = Vec2::new(10.0, 9.0);
        style.spacing.button_padding = Vec2::new(14.0, 7.0);
        style.spacing.interact_size.y = 30.0;
        style.spacing.indent = 20.0;
        style.spacing.scroll.bar_width = 10.0;
        style.spacing.window_margin = Margin::same(12.0);
        style.spacing.menu_margin = Margin::same(8.0);
        let sizes = [
            (TextStyle::Heading, 21.0, FontFamily::Proportional),
            (TextStyle::Body, 14.5, FontFamily::Proportional),
            (TextStyle::Button, 14.5, FontFamily::Proportional),
            (TextStyle::Small, 12.5, FontFamily::Proportional),
            (TextStyle::Monospace, 13.0, FontFamily::Monospace),
        ];
        for (text_style, size, family) in sizes {
            style
                .text_styles
                .insert(text_style, FontId::new(size, family));
        }
    });
}

/// A bold section heading, as the reference app titles each block.
pub fn section(ui: &mut Ui, title: &str) {
    ui.add_space(12.0);
    ui.label(RichText::new(title).size(16.0).strong().color(INK));
    ui.add_space(5.0);
}

/// A bordered content box: white fill, hairline border, rounded corners. Used
/// for the lists and logs that the reference app frames the same way.
pub fn boxed<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    Frame::none()
        .fill(SURFACE)
        .stroke(Stroke::new(1.0_f32, BORDER))
        .rounding(Rounding::same(RADIUS))
        .inner_margin(Margin::symmetric(12.0, 10.0))
        .show(ui, add)
        .inner
}

/// A headed, bordered group: the heading sits above the box, not inside it.
pub fn group<R>(ui: &mut Ui, title: &str, add: impl FnOnce(&mut Ui) -> R) -> R {
    section(ui, title);
    boxed(ui, add)
}

/// The foot of the window: one hairline rule, then plain status text.
pub fn status_bar(ui: &mut Ui, panels: &[(&str, f32)]) {
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 1.0), egui::Sense::hover());
    ui.painter().line_segment(
        [rect.left_top(), rect.right_top()],
        Stroke::new(1.0_f32, BORDER),
    );
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        for (index, (text, _)) in panels.iter().enumerate() {
            if index > 0 {
                ui.label(RichText::new("·").color(INK_SOFT));
            }
            let color = if index == 0 { INK } else { INK_SOFT };
            ui.label(RichText::new(*text).size(12.5).color(color));
        }
    });
}

pub fn panel(fill: Color32, x: f32, y: f32) -> Frame {
    Frame::none()
        .fill(fill)
        .inner_margin(Margin::symmetric(x, y))
}

fn browse(ui: &mut Ui, text: &mut String, save: bool) {
    if ui.button("Browse...").clicked() {
        let dialog = rfd::FileDialog::new();
        let picked = if save {
            dialog.save_file()
        } else {
            dialog.pick_file()
        };
        if let Some(path) = picked {
            *text = path.display().to_string();
        }
    }
}

/// One grid row: label, path box, browse button. Call inside a 3-column grid.
pub fn path_row(ui: &mut Ui, label: &str, text: &mut String, save: bool, hint: &str) {
    ui.label(label);
    let width = (ui.available_width() - 120.0).max(180.0);
    ui.add(
        egui::TextEdit::singleline(text)
            .desired_width(width)
            .hint_text(hint),
    );
    browse(ui, text, save);
    ui.end_row();
}

/// A grid row whose checkbox gates the path box.
pub fn opt_row(ui: &mut Ui, label: &str, opt: &mut PathOpt, hint: &str) {
    ui.checkbox(&mut opt.enabled, label);
    let width = (ui.available_width() - 120.0).max(180.0);
    ui.add_enabled(
        opt.enabled,
        egui::TextEdit::singleline(&mut opt.path)
            .desired_width(width)
            .hint_text(hint),
    );
    ui.add_enabled_ui(opt.enabled, |ui| browse(ui, &mut opt.path, true));
    ui.end_row();
}

/// A grid row holding one short value, with its explanation beside it.
pub fn value_row(ui: &mut Ui, label: &str, text: &mut String, hint: &str, help: &str) {
    ui.label(label);
    ui.add(
        egui::TextEdit::singleline(text)
            .desired_width(150.0)
            .hint_text(hint),
    );
    ui.label(RichText::new(help).size(12.5).color(INK_SOFT));
    ui.end_row();
}

/// A grid row holding one flag, with its explanation beside it.
pub fn flag_row(ui: &mut Ui, label: &str, value: &mut bool, help: &str) {
    ui.checkbox(value, label);
    ui.label(RichText::new(help).size(12.5).color(INK_SOFT));
    ui.label("");
    ui.end_row();
}
