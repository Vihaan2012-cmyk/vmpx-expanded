use eframe::egui::{
    self, Color32, Context, FontFamily, FontId, Frame, Margin, RichText, Rounding, Stroke,
    TextStyle, Ui, Vec2,
};

use crate::runner::PathOpt;

// Stock egui-dark palette (no custom recolor): neutral grays + the default
// egui selection blue. Only OK/BAD/WARN stay as semantic status colors.
pub const ACCENT: Color32 = Color32::from_rgb(90, 170, 255);
pub const ACCENT_DIM: Color32 = Color32::from_rgb(0, 92, 128);
pub const OK: Color32 = Color32::from_rgb(90, 200, 120);
pub const BAD: Color32 = Color32::from_rgb(232, 104, 104);
pub const WARN: Color32 = Color32::from_rgb(224, 176, 80);
pub const BG_SIDE: Color32 = Color32::from_gray(24);
pub const BG_MAIN: Color32 = Color32::from_gray(27);
pub const BG_RAISED: Color32 = Color32::from_gray(32);
const LINE: Color32 = Color32::from_gray(55);

pub fn apply(ctx: &Context) {
    let mut v = egui::Visuals::dark();
    v.panel_fill = BG_MAIN;
    v.window_fill = BG_RAISED;
    v.extreme_bg_color = Color32::from_gray(10);
    v.faint_bg_color = Color32::from_gray(34);
    v.selection.bg_fill = ACCENT_DIM;
    v.selection.stroke = Stroke::new(1.0_f32, ACCENT);
    v.hyperlink_color = ACCENT;
    v.window_rounding = Rounding::same(8.0);
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, LINE);
    let radius = Rounding::same(5.0);
    for (widget, fill) in [
        (&mut v.widgets.inactive, Color32::from_gray(48)),
        (&mut v.widgets.hovered, Color32::from_gray(64)),
        (&mut v.widgets.active, ACCENT_DIM),
    ] {
        widget.rounding = radius;
        widget.bg_fill = fill;
        widget.weak_bg_fill = fill;
    }
    v.widgets.noninteractive.rounding = radius;
    v.widgets.open.rounding = radius;
    v.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, ACCENT);
    v.widgets.active.bg_stroke = Stroke::new(1.0_f32, ACCENT);
    ctx.set_visuals(v);

    ctx.style_mut(|style| {
        style.spacing.item_spacing = Vec2::new(8.0, 7.0);
        style.spacing.button_padding = Vec2::new(10.0, 5.0);
        style.spacing.interact_size.y = 26.0;
        style.spacing.indent = 18.0;
        style.spacing.scroll.bar_width = 8.0;
        let sizes = [
            (TextStyle::Heading, 20.0, FontFamily::Proportional),
            (TextStyle::Body, 14.5, FontFamily::Proportional),
            (TextStyle::Button, 14.5, FontFamily::Proportional),
            (TextStyle::Small, 11.5, FontFamily::Proportional),
            (TextStyle::Monospace, 13.5, FontFamily::Monospace),
        ];
        for (text_style, size, family) in sizes {
            style
                .text_styles
                .insert(text_style, FontId::new(size, family));
        }
    });
}

pub fn panel(fill: Color32, x: f32, y: f32) -> Frame {
    Frame::none()
        .fill(fill)
        .inner_margin(Margin::symmetric(x, y))
}

pub fn section(ui: &mut Ui, title: &str) {
    ui.add_space(4.0);
    ui.label(RichText::new(title).small().strong().color(ACCENT));
    ui.separator();
}

fn browse(ui: &mut Ui, text: &mut String, save: bool) {
    if ui.button("Browse").clicked() {
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
    let width = (ui.available_width() - 90.0).max(120.0);
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
    let width = (ui.available_width() - 90.0).max(120.0);
    ui.add_enabled(
        opt.enabled,
        egui::TextEdit::singleline(&mut opt.path)
            .desired_width(width)
            .hint_text(hint),
    );
    ui.add_enabled_ui(opt.enabled, |ui| browse(ui, &mut opt.path, true));
    ui.end_row();
}
