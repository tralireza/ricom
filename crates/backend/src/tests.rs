//! Proof the render seam is usable without any GL: a no-op `Backend` held behind
//! `Box<dyn Backend>`. This is the concrete win the abstraction buys — the seam is
//! now exercisable anywhere (and by any future test double), not only on a live X server.

use super::*;

/// A backend that renders nothing. Only exists to prove `Backend` is object-safe
/// and swappable; a future capability/gating test would build on this.
struct FakeBackend;

impl Backend for FakeBackend {
    fn present_windows(
        &self,
        _items: &[WindowDraw],
        _screen_w: i32,
        _screen_h: i32,
        _hud: Option<&Hud>,
        _osd: Option<&Osd>,
        _clear: &[Rect],
    ) -> Result<()> {
        Ok(())
    }
    fn set_render_params(&mut self, _render: RenderParams) {}
    fn set_font(&mut self, _path: &str, _size: f32) {}
    fn has_text(&self) -> bool {
        false
    }
    fn render_ms(&self) -> f32 {
        0.0
    }
    fn buffer_age(&self) -> i32 {
        0
    }
    fn caps(&self) -> BackendCaps {
        // Model a reduced-capability backend (like XRender): no shaders/mesh/scale/blur.
        BackendCaps {
            shaders: false,
            mesh: false,
            scale: false,
            blur: false,
            shadow: false,
            rounded_corners: false,
        }
    }
}

#[test]
fn backend_is_object_safe_and_swappable() {
    // The whole point: hold a backend behind `dyn` and drive every seam method
    // through the vtable — no EGL/GL/X in sight.
    let mut b: Box<dyn Backend> = Box::new(FakeBackend);
    b.set_render_params(RenderParams::default());
    b.set_font("", 1.0);
    assert!(!b.has_text());
    assert_eq!(b.buffer_age(), 0);
    assert_eq!(b.render_ms(), 0.0);
    b.present_windows(&[], 1920, 1080, None, None, &[]).unwrap();
    // Capabilities flow through the vtable; the reduced fake advertises no shaders,
    // while the full-featured default (GL) is all-true.
    assert!(!b.caps().shaders);
    assert_eq!(
        BackendCaps::all(),
        BackendCaps {
            shaders: true,
            mesh: true,
            scale: true,
            blur: true,
            shadow: true,
            rounded_corners: true,
        }
    );
}

fn display(output: &str, hz: Option<f32>) -> HudDisplay {
    HudDisplay { width: 3840, height: 2160, refresh_hz: hz, output: output.to_string() }
}

#[test]
fn hud_display_mode_label() {
    // xrandr's two decimals, trailing zeros trimmed; no `@…Hz` without a timing.
    assert_eq!(display("", Some(60.0)).mode_label(), "3840x2160@60Hz");
    assert_eq!(display("", Some(59.9997)).mode_label(), "3840x2160@60Hz");
    assert_eq!(display("", Some(59.94)).mode_label(), "3840x2160@59.94Hz");
    assert_eq!(display("", Some(143.856)).mode_label(), "3840x2160@143.86Hz");
    assert_eq!(display("", Some(74.9)).mode_label(), "3840x2160@74.9Hz");
    assert_eq!(display("", Some(120.0)).mode_label(), "3840x2160@120Hz");
    assert_eq!(display("", None).mode_label(), "3840x2160");
}

#[test]
fn hud_display_text_never_widens_for_the_name() {
    let w = |t: &str| t.len() as f32 * 10.0; // fake font: 10 px per char
    let d = display("DP-2", Some(60.0));
    // Full line = 20 chars = 200 px: fits a 260 px numbers line, and exactly 200.
    assert_eq!(hud_display_text(&d, 260.0, w), ("3840x2160@60Hz  DP-2".to_string(), 200.0));
    assert_eq!(hud_display_text(&d, 200.0, w).0, "3840x2160@60Hz  DP-2");
    // Any narrower and the name would widen the panel → the mode alone (14 chars).
    assert_eq!(hud_display_text(&d, 199.0, w), ("3840x2160@60Hz".to_string(), 140.0));
    // The mode shows even when it alone is wider; no output name → the mode alone.
    assert_eq!(hud_display_text(&d, 50.0, w).0, "3840x2160@60Hz");
    assert_eq!(hud_display_text(&display("", Some(60.0)), 1e6, w).0, "3840x2160@60Hz");
}
