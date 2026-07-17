//! Backend-neutral descriptor + trait crate.
//!
//! The *descriptive* types the compositor hands a render backend
//! (`WindowDraw` / `Quad` / `Hud` / `Osd` / `RenderParams` / effect params) plus
//! the `Backend` trait itself — so `session` and every backend agree on the seam
//! without any backend depending on another. Pure data; no EGL / GL / X.

use anyhow::Result;

/// Re-exported so callers can build clip/clear rects without a direct `region` dep.
pub use region::Rect;

/// Runtime render parameters (from the config file): set when the backend is
/// created and swapped in on config reload via [`Backend::set_render_params`].
/// Defaults reproduce the previously compiled-in constants.
#[derive(Debug, Clone, Copy)]
pub struct RenderParams {
    /// Drop-shadow falloff distance to the left/bottom (px).
    pub shadow_radius: f32,
    /// Peak shadow alpha.
    pub shadow_strength: f32,
    /// Composite background colour (RGB), shown where no window covers.
    pub background: [f32; 3],
    /// Window corner radius (px); `0.0` = square.
    pub corner_radius: f32,
    /// Background blur on/off (frost the backdrop behind translucent windows).
    pub blur_enabled: bool,
    /// Dual-Kawase iterations (down+up steps); clamped to `MAX_BLUR_LEVELS - 1`.
    pub blur_passes: i32,
    /// Blur sample offset per pass (px).
    pub blur_radius: f32,
    /// Burn/dissolve segment granularity (`u_segscale`): higher = finer patches.
    pub burn_seg_scale: f32,
    /// Burn/dissolve ember-band half-width (`u_ember`): smaller = thinner glow.
    pub burn_ember: f32,
    /// Cooler trailing ember colour (`u_ember_cool`, RGB).
    pub burn_ember_cool: [f32; 3],
    /// Hottest leading-edge ember colour (`u_ember_hot`, RGB).
    pub burn_ember_hot: [f32; 3],
    /// On-screen-text outline width (px at 1080p, scaled with the surface); `0` = none.
    /// Shared by HUD/OSD; each surface toggles whether to apply it (`Hud.outline`/`Osd.outline`).
    pub text_outline: f32,
    /// Text outline colour (RGB).
    pub text_outline_color: [f32; 3],
    /// Text drop-shadow offset (px at 1080p, down-right, scaled); `0` = none.
    pub text_shadow: f32,
    /// Text drop-shadow colour (RGB).
    pub text_shadow_color: [f32; 3],
    /// Outline direction: `false` = all-around (rings the glyph); `true` = drop, masking
    /// the `text_outline` band to the bottom-right only so it reads as a tight drop-shadow.
    pub text_outline_drop: bool,
}

impl Default for RenderParams {
    fn default() -> Self {
        RenderParams {
            shadow_radius: 12.0,
            shadow_strength: 0.45,
            background: [0.05, 0.05, 0.07],
            corner_radius: 0.0,
            blur_enabled: false,
            blur_passes: 3,
            blur_radius: 4.0,
            burn_seg_scale: 36.0,
            burn_ember: 0.07,
            burn_ember_cool: [0.28, 0.02, 0.0],
            burn_ember_hot: [0.75, 0.22, 0.04],
            text_outline: 1.5,
            text_outline_color: [0.0, 0.0, 0.0],
            text_shadow: 0.0,
            text_shadow_color: [0.0, 0.0, 0.0],
            text_outline_drop: false,
        }
    }
}

/// One window to composite: its named pixmap, on-screen rect (top-left origin,
/// pixels, border included), whole-window opacity (`0.0..=1.0`), and whether to
/// draw a drop shadow behind it.
#[derive(Debug, Clone, Copy)]
pub struct Quad {
    pub pixmap: u32,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub opacity: f32,
    pub shadow: bool,
    /// Frost the backdrop under this window (set for translucent windows when
    /// blur is enabled; ignored for opaque windows whose backdrop is hidden).
    pub blur: bool,
    /// Corner radius (px) for this window; `0.0` = square.
    pub corner_radius: f32,
}

/// Per-window burn/dissolve state for the close animation. When a [`WindowDraw`]
/// carries `Some`, the backend draws it through the burn/dissolve program instead of the plain
/// blit (no shadow / frost / corner rounding, like the wobble-mesh path).
#[derive(Debug, Clone, Copy)]
pub struct Burn {
    /// `0.0` = intact … `1.0` = fully burnt away.
    pub progress: f32,
    /// Per-window random offset so no two windows burn with the same pattern.
    pub seed: f32,
}

/// Radial-ripple (water refraction) parameters for the ripple program. When a
/// [`WindowDraw`] carries `Some`, the backend draws it through the ripple program
/// (per-pixel UV warp; no shadow / frost / corner — like the mesh / spin paths).
#[derive(Debug, Clone, Copy)]
pub struct RippleParams {
    /// Ripple centre in UV (`[0.5, 0.5]` = window centre).
    pub center: [f32; 2],
    /// Peak radial UV displacement (aspect-corrected units).
    pub amp: f32,
    /// Ring spacing as a fraction of the aspect-corrected radius.
    pub wavelength: f32,
    /// Phase (cycles); advances so the rings expand outward.
    pub phase: f32,
    /// Spread constant — amplitude falls with radius (large centre, faint rim).
    pub r0: f32,
}

/// Traveling-wave (content refraction) parameters for the wave program. When a
/// [`WindowDraw`] carries `Some`, the backend draws it through the wave program
/// (per-pixel UV warp; no shadow / frost / corner — like the ripple path). Replaces
/// the old mesh-based wave, so it's smooth at any amplitude.
#[derive(Debug, Clone, Copy)]
pub struct WaveParams {
    /// Peak perpendicular UV displacement.
    pub amp: f32,
    /// Wavelength as a fraction of the travel axis (`1.0` = one cycle across).
    pub wavelength: f32,
    /// Phase (cycles); advances so the crest travels.
    pub phase: f32,
    /// Travel axis: `0` = along X (displaces V), `1` = along Y (displaces U).
    pub axis: u32,
}

/// Drain / whirlpool close parameters for the drain program. When a [`WindowDraw`] carries
/// `Some`, the backend draws it through the drain program (per-pixel; no shadow / frost
/// / corner). A close driver like burn: `progress` 0→1 spirals + shrinks the content
/// into a vanishing point at `center` (no self-fade — the shrink + out-of-bounds mask
/// carry it away); the window is reaped at `1`.
#[derive(Debug, Clone, Copy)]
pub struct DrainParams {
    /// Drain centre in UV (`[0.5, 0.5]` = window centre).
    pub center: [f32; 2],
    /// Progress `0.0` (intact) → `1.0` (fully drained / gone).
    pub progress: f32,
    /// Swirl rotations at full progress.
    pub turns: f32,
    /// Turbulence amount (`u_turb`): how strongly the seeded noise makes the vortex
    /// arms rotate unevenly. `0.0` = a smooth, uniform, deterministic spiral.
    pub turbulence: f32,
    /// Per-window seed so each drain's rate-turbulence differs.
    pub seed: f32,
}

/// A window to composite plus the screen-space rectangles it's actually visible
/// in (region-level occlusion): [`Backend::present_windows`] scissors each of
/// the quad's draws to `clip`, so pixels covered by an opaque window on top are
/// never shaded. An empty `clip` is a fully-occluded window (callers omit those).
pub struct WindowDraw {
    pub quad: Quad,
    pub clip: Vec<region::Rect>,
    /// Wobble mesh: `MESH_N × MESH_N` deformed vertices as `[x_px, y_px, u, v]`,
    /// row-major (from `wm::anim::Wobble::vertices`). `Some` → draw the textured
    /// mesh (no shadow / frost / corner rounding — square while wobbling); `None`
    /// → the normal quad path. `quad.x/y/w/h` still give the un-deformed rect (for
    /// texture binding and, when settled, the quad path).
    pub mesh: Option<Vec<[f32; 4]>>,
    /// Burn/dissolve close effect. `Some` → draw via the burn/dissolve program at this progress
    /// (mutually exclusive with `mesh`; a closing window doesn't wobble).
    pub burn: Option<Burn>,
    /// Rotation about the window centre (radians) for the `spin` primitive. `Some`
    /// → draw via the spin program (no shadow/frost; corners suppressed), mutually
    /// exclusive with `mesh`/`burn`. `None` → the normal quad path.
    pub spin: Option<f32>,
    /// Radial water-refraction ripple. `Some` → draw via the ripple program (per-pixel UV
    /// warp; no shadow / frost / corner), mutually exclusive with `mesh`/`burn`/`spin`.
    pub ripple: Option<RippleParams>,
    /// Traveling wave (content refraction). `Some` → draw via the wave program (per-pixel UV
    /// warp; no shadow / frost / corner), mutually exclusive with `mesh`/`burn`/`spin`/`ripple`.
    pub wave: Option<WaveParams>,
    /// Drain / whirlpool close. `Some` → draw via the drain program (per-pixel; no shadow /
    /// frost / corner), mutually exclusive with `mesh`/`burn`/`spin`/`ripple`/`wave`.
    pub drain: Option<DrainParams>,
}

impl WindowDraw {
    /// Draw `quad` in full — a single clip rect equal to its own bounds (no
    /// occlusion), no wobble mesh. Used by the diagnostic `--blit-test` /
    /// `--opacity-test` paths.
    pub fn whole(quad: Quad) -> Self {
        WindowDraw {
            quad,
            clip: vec![region::Rect::from_xywh(quad.x, quad.y, quad.w, quad.h)],
            mesh: None,
            burn: None,
            spin: None,
            ripple: None,
            wave: None,
            drain: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GlInfo {
    pub vendor: String,
    pub renderer: String,
    pub version: String,
}

/// Which screen corner the FPS HUD anchors to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HudCorner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// 1m/5m/15m compositor load averages for the HUD's load block: present rate
/// (fps) and mean GPU render time (ms; `None` for a window that had no frames —
/// idle or bypassed). Toggled independently of the numbers/graph.
pub struct HudLoad {
    pub fps: [f32; 3],
    pub render_ms: [Option<f32>; 3],
}

/// One frame's HUD data, drawn by [`Backend::present_windows`] when `Some`. The
/// graph itself is fed by the backend's own GPU render-time samples.
pub struct Hud {
    /// Present rate (frames composited in the last second).
    pub fps: u32,
    /// Draw the render-time graph beneath the numbers.
    pub graph: bool,
    /// Which screen corner to anchor to.
    pub corner: HudCorner,
    /// Extra size multiplier on top of the automatic screen-height scaling.
    pub scale: f32,
    /// Current display refresh rate (Hz) — one refresh interval is the render budget.
    pub refresh_hz: f32,
    /// Optional 1m/5m/15m load block, shown under the graph (`Super+Shift+L`).
    pub load: Option<HudLoad>,
    /// Outline the HUD text (per `RenderParams` text style) so it reads without the
    /// panel. `false` = plain text (the panel provides contrast).
    pub outline: bool,
    /// Whole-HUD opacity (panel + text + graph), `1.0` = opaque. The caller fades
    /// this to 0 and back during an auto-hop (hide at the old corner, show at the
    /// new one — an in-place hop, no slide); `1.0` otherwise.
    pub opacity: f32,
}

/// How the OSD toast appears/disappears — the caller picks the open or close
/// effect per phase (see `session`). Mirrors `config::OsdEffect`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OsdEffect {
    Fade,
    Slide,
    Pop,
    Unroll,
    Stretch,
}

/// An on-screen notification banner ("toast"), drawn top-center over everything.
/// `presence` (0..1) drives the current `effect`: 0 = fully hidden, 1 = fully shown.
pub struct Osd {
    pub text: String,
    pub presence: f32,
    /// Size multiplier on top of the automatic screen-height scaling.
    pub scale: f32,
    /// The effect for this phase (open while appearing, close while disappearing).
    pub effect: OsdEffect,
    /// Text colour (RGB); alpha comes from the fade.
    pub color: [f32; 3],
    /// Banner background colour (RGBA); alpha `0.0` = text-only (no box drawn).
    pub background: [f32; 4],
    /// Draw a dark outline behind the glyphs (legibility without a box).
    pub outline: bool,
}

/// The render-backend seam. `session` builds a `Vec<WindowDraw>` (all occlusion /
/// opacity / effect policy) and calls `present_windows`; the rest is config +
/// observation. A backend is built by a factory (see `session`), not this trait —
/// `dyn` traits can't carry a constructor.
pub trait Backend {
    /// Composite the window stack for one frame and present it. `clear` is the
    /// region cleared first (partial-repaint aware); `hud`/`osd` draw on top.
    ///
    /// Takes `&self` (this is the frame-time method, unlike the config-time
    /// `&mut self` setters) so the caller can hold the backend by shared borrow
    /// across a whole composite pass while it reads other state. The cost of that
    /// choice is a contract: an implementor that mutates per-frame GPU caches
    /// (blur pyramid, GPU-timer ring, render-time samples) must do so through
    /// interior mutability (`Cell`/`RefCell`) — as the GL backend does.
    fn present_windows(
        &self,
        items: &[WindowDraw],
        screen_w: i32,
        screen_h: i32,
        hud: Option<&Hud>,
        osd: Option<&Osd>,
        clear: &[Rect],
    ) -> Result<()>;

    /// Swap in new render parameters (config reload).
    fn set_render_params(&mut self, render: RenderParams);

    /// Load / replace the on-screen-text font (path + size multiplier).
    fn set_font(&mut self, path: &str, size: f32);

    /// Whether a usable on-screen-text font is loaded.
    fn has_text(&self) -> bool;

    /// Most recent GPU render time (ms); `0.0` if unmeasured / unsupported.
    fn render_ms(&self) -> f32;

    /// Back-buffer age for partial repaint; `0` = undefined / unsupported (repaint full).
    fn buffer_age(&self) -> i32;
}

#[cfg(test)]
mod tests;
