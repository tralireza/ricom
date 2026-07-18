//! The render-backend factory — the single place that names concrete backends.
//!
//! `session` holds a `Box<dyn Backend>` and drives it purely through the trait, so it
//! must never name a concrete backend. This crate owns [`make_backend`] (which builds
//! the backend selected by `config.backend`) and its [`render_params`] helper, keeping
//! `backend-gl` / `backend-xrender` out of `session`'s dependency graph — the seam is
//! now enforced at the Cargo manifest level, not just by convention. A future backend
//! (e.g. glx) slots in as one more match arm here, touching neither `session` nor the
//! [`Backend`] trait. Constructors take `window: u32` (an X id), so this crate needs no
//! x11rb dependency.

use anyhow::Result;
use backend::{Backend, RenderParams};
use backend_gl::GlBackend;
use backend_xrender::XrenderBackend;
use config::{BackendKind, Config};

/// Build a backend's render parameters from the config.
pub fn render_params(cfg: &Config) -> RenderParams {
    RenderParams {
        shadow_radius: cfg.shadow.radius,
        shadow_strength: cfg.shadow.strength,
        background: cfg.background,
        corner_radius: cfg.corner_radius,
        blur_enabled: cfg.blur.enabled,
        blur_passes: cfg.blur.passes,
        blur_radius: cfg.blur.radius,
        burn_seg_scale: cfg.burn.seg_scale,
        burn_ember: cfg.burn.ember_width,
        burn_ember_cool: cfg.burn.ember_cool,
        burn_ember_hot: cfg.burn.ember_hot,
        text_outline: cfg.font.outline_width,
        text_outline_color: cfg.font.outline_color,
        text_shadow: cfg.font.shadow_offset,
        text_shadow_color: cfg.font.shadow_color,
        text_outline_drop: cfg.font.outline_style.eq_ignore_ascii_case("drop"),
    }
}

/// Build the render backend named by the config (`backend = …`), returning a
/// `Box<dyn Backend>` so callers never name a concrete backend. `window` is the
/// composite-overlay X id; `visual` its visual id. A new backend slots in as another
/// match arm here.
pub fn make_backend(config: &Config, window: u32, visual: u32) -> Result<Box<dyn Backend>> {
    match config.backend {
        BackendKind::Gl => Ok(Box::new(GlBackend::new(window, visual, render_params(config))?)),
        BackendKind::Xrender => {
            Ok(Box::new(XrenderBackend::new(window, visual, render_params(config))?))
        }
    }
}
