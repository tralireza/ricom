//! Proof the render seam is usable without any GL: a no-op `Backend` held behind
//! `Box<dyn Backend>`. This is the concrete win the abstraction buys — the seam is
//! now exercisable on the Mac (and by any future test double), not only on i7.

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
}
