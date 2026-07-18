//! The compositor session: owns the X connection, the tracked window stack, and
//! the GL backend, and runs the calloop event loop. Mirrors picom's `session_t`
//! + main loop (`src/picom.c`) and event handlers (`src/event.c`).
//!
//! Live compositor: become CM, redirect the screen, render into the composite
//! overlay via `backend-gl`, and recomposite the window stack whenever damage
//! (or a structural change) arrives. On exit the X server auto-releases our
//! resources (redirect, overlay, pixmaps, damage), restoring normal drawing.

use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;
use std::os::fd::AsFd;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use calloop::generic::Generic;
#[cfg(target_os = "linux")]
use calloop::signals::{Signal, Signals};
use calloop::timer::{TimeoutAction, Timer};
use calloop::{EventLoop, Interest, LoopHandle, LoopSignal, Mode, PostAction, RegistrationToken};
use x11rb::connection::Connection;
use x11rb::protocol::Event;
use x11rb::protocol::xproto::{NotifyDetail, NotifyMode, Place, Window};

use backend::{
    Backend, BackendCaps, Burn, DrainParams, Hud, HudCorner, HudLoad, Osd, Quad, RenderParams,
    RippleParams, WaveParams, WindowDraw,
};
use backend_gl::GlBackend;
use backend_xrender::XrenderBackend;
use region::{Rect, Region};

/// Max frames of damage history kept for EGL buffer-age partial repaint.
const MAX_BUFFER_AGE: usize = 4;
/// Extra px around a wobble's mesh bbox when damaging/clipping it — headroom for
/// spring overshoot and the AA fringe, so no jiggling pixel is ever left stale.
const WOBBLE_PAD: f32 = 8.0;
/// Default `spin` rotation in degrees when a `Spin` block sets none (a full turn).
const SPIN_DEFAULT_DEG: f64 = 360.0;
use config::{Axis, BackendKind, Category, Config, Edge, FocusSource, OsdEffect, Primitive, RuleResult, WindowMatch};
use wm::anim::Fade;
use wm::{Win, WindowId, WindowStack};
use xconn::XConn;

mod hotkey;

/// Build the GL backend's render parameters from the config.
fn render_params(cfg: &Config) -> RenderParams {
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

/// Build the render backend named by the config (`backend = …`). Only the GL backend
/// exists today; a future xrender/glx backend slots in as another match arm. Returns
/// a `Box<dyn Backend>` so `session` never names a concrete backend past this point.
pub fn make_backend(config: &Config, window: Window, visual: u32) -> Result<Box<dyn Backend>> {
    match config.backend {
        BackendKind::Gl => Ok(Box::new(GlBackend::new(window, visual, render_params(config))?)),
        BackendKind::Xrender => {
            Ok(Box::new(XrenderBackend::new(window, visual, render_params(config))?))
        }
    }
}

/// Log any non-fatal config problems (unknown animation presets, blocks used in a
/// category they don't fit). Parsing already rejects typos/unknown keys; these are
/// softer, user-education warnings surfaced at load and reload.
fn log_config_warnings(cfg: &Config) {
    for w in cfg.validate() {
        tracing::warn!("config: {w}");
    }
}

/// Strictly parse a corner string to [`HudCorner`]; `None` for anything
/// unrecognised. Used for the `[fps] auto_move_avoid` list, where a typo must be
/// dropped (and warned about by `config::validate`) rather than silently coerced.
fn parse_corner(s: &str) -> Option<HudCorner> {
    match s {
        "top-left" => Some(HudCorner::TopLeft),
        "top-right" => Some(HudCorner::TopRight),
        "bottom-left" => Some(HudCorner::BottomLeft),
        "bottom-right" => Some(HudCorner::BottomRight),
        _ => None,
    }
}

/// Map the config's corner string to the backend's [`HudCorner`] (defaults to
/// top-right for anything unrecognised).
fn hud_corner(s: &str) -> HudCorner {
    parse_corner(s).unwrap_or(HudCorner::TopRight)
}

/// Does an outer rectangle (position, size, uniform border) cover the whole root?
/// ricom's "fullscreen" test, shared by unredir-if-possible and rule matching.
fn covers_root(x: i32, y: i32, w: i32, h: i32, bw: i32, rw: i32, rh: i32) -> bool {
    x <= 0 && y <= 0 && x + w + 2 * bw >= rw && y + h + 2 * bw >= rh
}

/// A stable-but-varied burn seed per window id, so each window's dissolve differs.
fn burn_seed(id: WindowId) -> f32 {
    (id.wrapping_mul(2_654_435_761) % 100_000) as f32 / 100_000.0
}

/// Map a config easing curve to the `wm` animation easing (keeps `wm` config-free).
fn map_easing(e: config::Easing) -> wm::anim::Easing {
    match e {
        config::Easing::EaseOut => wm::anim::Easing::EaseOut,
        config::Easing::EaseIn => wm::anim::Easing::EaseIn,
        config::Easing::Linear => wm::anim::Easing::Linear,
    }
}

/// Map a config scale axis to the `wm` animation axis.
fn map_axis(a: Axis) -> wm::anim::Axis {
    match a {
        Axis::Both => wm::anim::Axis::Both,
        Axis::X => wm::anim::Axis::X,
        Axis::Y => wm::anim::Axis::Y,
    }
}

/// Whether the active backend can render an animation primitive. Opacity/Scale/
/// Translate are pure alpha/geometry — always available; the shader effects
/// (Spin/Wave/Ripple/Burn/Drain) need `caps.shaders` and Wobble needs `caps.mesh`.
/// When this is false, `start_anim` skips the primitive and falls back to a fade —
/// see the close arm, where burn/drain must reroute to `begin_fade_out` so the
/// window still reaps (they are close *drivers*, not decorations).
fn caps_allow(caps: BackendCaps, block: &Primitive) -> bool {
    match block {
        Primitive::Opacity { .. } | Primitive::Scale { .. } | Primitive::Translate { .. } => true,
        Primitive::Wobble { .. } => caps.mesh,
        Primitive::Spin { .. }
        | Primitive::Wave { .. }
        | Primitive::Ripple { .. }
        | Primitive::Burn
        | Primitive::Drain { .. } => caps.shaders,
    }
}

// ── `ricomctl animate` param overrides ────────────────────────────────────────
// Effects take optional `key=value` overrides on the CLI (empty ⇒ configured
// defaults). These free fns type + validate them; [`App::apply_effect`] applies.

/// A `key=value` override parsed as `f32`. `Ok(None)` if the key is absent; `Err`
/// if present but not a number.
fn param_f32(params: &[(String, String)], key: &str) -> Result<Option<f32>, String> {
    match params.iter().find(|(k, _)| k == key) {
        None => Ok(None),
        Some((_, v)) => v
            .parse::<f32>()
            .map(Some)
            .map_err(|_| format!("param '{key}' wants a number, got '{v}'")),
    }
}

/// An `axis=` override (`x`/`y`/`both`). `Ok(None)` if absent; `Err` if unrecognised.
fn param_axis(params: &[(String, String)]) -> Result<Option<wm::anim::Axis>, String> {
    match params.iter().find(|(k, _)| k == "axis") {
        None => Ok(None),
        Some((_, v)) => match v.as_str() {
            "both" => Ok(Some(wm::anim::Axis::Both)),
            "x" => Ok(Some(wm::anim::Axis::X)),
            "y" => Ok(Some(wm::anim::Axis::Y)),
            _ => Err(format!("param 'axis' wants x|y|both, got '{v}'")),
        },
    }
}

/// An `easing=` override (`ease-out`/`ease-in`/`linear`). `Ok(None)` if absent.
fn param_easing(params: &[(String, String)]) -> Result<Option<wm::anim::Easing>, String> {
    match params.iter().find(|(k, _)| k == "easing") {
        None => Ok(None),
        Some((_, v)) => match v.as_str() {
            "ease-out" => Ok(Some(wm::anim::Easing::EaseOut)),
            "ease-in" => Ok(Some(wm::anim::Easing::EaseIn)),
            "linear" => Ok(Some(wm::anim::Easing::Linear)),
            _ => Err(format!("param 'easing' wants ease-out|ease-in|linear, got '{v}'")),
        },
    }
}

/// Reject any provided key not valid for `effect` (strict), so a typo like `amplitud=`
/// is flagged instead of silently ignored. Valid keys come from the shared
/// [`proto::effect_params`] schema — the one source used by `animate`, `set`, and help.
fn check_keys(effect: &str, params: &[(String, String)]) -> Result<(), String> {
    let valid = proto::effect_params(effect).unwrap_or(&[]);
    for (k, _) in params {
        if !valid.iter().any(|(vk, _)| vk == k) {
            let list = if valid.is_empty() {
                "none".to_string()
            } else {
                valid.iter().map(|(vk, _)| *vk).collect::<Vec<_>>().join(", ")
            };
            return Err(format!("effect '{effect}' has no param '{k}' (valid: {list})"));
        }
    }
    Ok(())
}

/// The pixel offset for a `translate` block: explicit `dx`/`dy`, or — if an
/// `edge` is given — the offset that moves the window's outer `rect`
/// (`[x, y, w, h]`) fully off that screen edge (`screen` = root w×h). This is the
/// away-from-rest offset: the window slides *in from* it on open, *out to* it on close.
fn resolve_offset(dx: f32, dy: f32, edge: Option<Edge>, rect: [f32; 4], screen: (i32, i32)) -> [f32; 2] {
    let Some(edge) = edge else {
        return [dx, dy];
    };
    let [x, y, w, h] = rect;
    let (sw, sh) = (screen.0 as f32, screen.1 as f32);
    match edge {
        Edge::Left => [-(x + w), 0.0],
        Edge::Right => [sw - x, 0.0],
        Edge::Top => [0.0, -(y + h)],
        Edge::Bottom => [0.0, sh - y],
    }
}

/// Shrink an outer rect `[x, y, w, h]` about its centre by `factor` — the
/// compressed start rect for a spawn "boing" wobble on open.
fn squash_rect(rect: [f32; 4], factor: f32) -> [f32; 4] {
    let [x, y, w, h] = rect;
    let (cx, cy) = (x + w / 2.0, y + h / 2.0);
    let (nw, nh) = (w * factor, h * factor);
    [cx - nw / 2.0, cy - nh / 2.0, nw, nh]
}

/// One compositable window for [`App::composite`]: quad, optional wobble mesh,
/// optional burn, always-on-top flag (from the `above` rule), and optional spin
/// angle (radians).
type CompositeItem =
    (Quad, Option<Vec<[f32; 4]>>, Option<Burn>, bool, Option<f32>, Option<RippleParams>, Option<WaveParams>, Option<DrainParams>);

/// Axis-aligned bounding box of a rect rotated `angle` radians about its centre —
/// the footprint/damage a spinning window covers. `rect` is `[x, y, w, h]`.
fn rotated_aabb(rect: [f32; 4], angle: f32) -> Rect {
    let [x, y, w, h] = rect;
    let (cx, cy) = (x + w / 2.0, y + h / 2.0);
    let (c, s) = (angle.cos().abs(), angle.sin().abs());
    let (hw, hh) = (w / 2.0 * c + h / 2.0 * s, w / 2.0 * s + h / 2.0 * c);
    Rect::new(
        (cx - hw).floor() as i32,
        (cy - hh).floor() as i32,
        (cx + hw).ceil() as i32,
        (cy + hh).ceil() as i32,
    )
}

/// Padded, pixel-aligned bounding box of a wobble mesh's vertices (`[x, y, u, v]`
/// per control point). Used as the deforming window's damage/clip footprint.
fn mesh_bbox(verts: &[[f32; 4]], pad: f32) -> Rect {
    if verts.is_empty() {
        return Rect::new(0, 0, 0, 0);
    }
    let (mut mnx, mut mny, mut mxx, mut mxy) =
        (f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for v in verts {
        mnx = mnx.min(v[0]);
        mny = mny.min(v[1]);
        mxx = mxx.max(v[0]);
        mxy = mxy.max(v[1]);
    }
    Rect::new(
        (mnx - pad).floor() as i32,
        (mny - pad).floor() as i32,
        (mxx + pad).ceil() as i32,
        (mxy + pad).ceil() as i32,
    )
}

/// Direction for the HUD move hotkeys (a modifier + an arrow key).
#[derive(Clone, Copy)]
enum Move {
    Left,
    Right,
    Up,
    Down,
}

/// Move the HUD `corner` one axis toward `dir`, keeping the other — so e.g. `Left`
/// from top-right lands on top-left, `Down` on bottom-right.
fn move_corner(corner: HudCorner, dir: Move) -> HudCorner {
    use HudCorner::*;
    let (top, left) = match corner {
        TopLeft => (true, true),
        TopRight => (true, false),
        BottomLeft => (false, true),
        BottomRight => (false, false),
    };
    let (top, left) = match dir {
        Move::Left => (top, true),
        Move::Right => (top, false),
        Move::Up => (true, left),
        Move::Down => (false, left),
    };
    match (top, left) {
        (true, true) => TopLeft,
        (true, false) => TopRight,
        (false, true) => BottomLeft,
        (false, false) => BottomRight,
    }
}

/// HUD auto-hop animation: the corner the HUD hops FROM → TO, with linear
/// progress `t` (0→1) over `dur` seconds. The hop is an in-place fade — the HUD
/// fades out at `from` over the first half, then fades in at `to` over the second
/// (see [`hop_view`], applied when building the `Hud` in `composite`).
struct HudMove {
    from: HudCorner,
    to: HudCorner,
    t: f64,
    dur: f64,
}

impl HudMove {
    /// Advance progress by `dt`; returns `true` while still moving (t < 1).
    fn advance(&mut self, dt: f64) -> bool {
        self.t = (self.t + dt / self.dur.max(0.001)).min(1.0);
        self.t < 1.0
    }
}

/// Smoothstep ease-in-out (0→1).
fn ease_in_out(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// One frame of an in-place hop: fade OUT at `from` over the first half of `t`,
/// then fade IN at `to` over the second half. Returns the corner to anchor the
/// HUD to and its (eased) whole-HUD opacity. The panel never moves — it hides
/// where it is and shows at the destination.
fn hop_view(from: HudCorner, to: HudCorner, t: f64) -> (HudCorner, f32) {
    if t < 0.5 {
        (from, ease_in_out(1.0 - t / 0.5) as f32) // 1 → 0 at the old corner
    } else {
        (to, ease_in_out((t - 0.5) / 0.5) as f32) // 0 → 1 at the new corner
    }
}

/// One xorshift64 step — a tiny inline PRNG (no `rand` dependency).
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// A random corner for an auto-hop: uniform over the corners that are neither
/// `current` nor in `avoid`. `None` when nowhere is eligible — every corner is
/// avoided, or the only non-avoided corner is where the HUD already sits — so the
/// caller leaves the HUD put.
fn random_corner(current: HudCorner, avoid: &[HudCorner], rng: &mut u64) -> Option<HudCorner> {
    use HudCorner::*;
    const ALL: [HudCorner; 4] = [TopLeft, TopRight, BottomLeft, BottomRight];
    let cands: Vec<HudCorner> =
        ALL.iter().copied().filter(|c| *c != current && !avoid.contains(c)).collect();
    if cands.is_empty() {
        return None;
    }
    Some(cands[(xorshift64(rng) % cands.len() as u64) as usize])
}

/// A non-zero PRNG seed from the OS-randomised `RandomState` (no time / `rand` dep).
fn seed_rng() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    std::collections::hash_map::RandomState::new().build_hasher().finish() | 1
}

/// Per-window X resources used for compositing.
#[derive(Default)]
struct WinGfx {
    /// Named off-screen pixmap of the window's current contents (incl. border).
    pixmap: Option<u32>,
    /// Damage object signalling when the window needs recompositing.
    damage: Option<u32>,
}

/// Rolling frame-rate meter, fed one sample per composited frame (post-present).
/// FPS is frames in the last second, so on a damage-driven idle screen it decays
/// to zero (there simply are no frames); `samples` holds recent frame-times (ms)
/// for the HUD graph.
struct FpsMeter {
    frames: VecDeque<Instant>,
    samples: Vec<f32>,
    last_ms: f32,
}

impl FpsMeter {
    const WINDOW: Duration = Duration::from_secs(1);
    const GRAPH: usize = 120;

    fn new() -> Self {
        FpsMeter { frames: VecDeque::new(), samples: Vec::new(), last_ms: 0.0 }
    }

    /// Record a present at `now`: update the last frame-time, push a graph sample
    /// (capped to `GRAPH`, oldest dropped), and drop frames older than one second.
    fn tick(&mut self, now: Instant) {
        if let Some(&prev) = self.frames.back() {
            let ms = now.duration_since(prev).as_secs_f32() * 1000.0;
            self.last_ms = ms;
            self.samples.push(ms);
            if self.samples.len() > Self::GRAPH {
                self.samples.remove(0);
            }
        }
        self.frames.push_back(now);
        while self.frames.front().is_some_and(|&t| now.duration_since(t) > Self::WINDOW) {
            self.frames.pop_front();
        }
    }

    /// Frames composited in the last second.
    fn fps(&self) -> u32 {
        self.frames.len() as u32
    }

    /// Most recent frame-to-frame time in milliseconds.
    fn last_ms(&self) -> f32 {
        self.last_ms
    }

    /// Recent frame times (ms), oldest first — for the HUD graph.
    fn samples(&self) -> &[f32] {
        &self.samples
    }
}

/// One window's load figure: present rate (fps) and mean GPU render time (ms;
/// `None` when the window had no composited frames — idle or bypassed).
struct LoadAvg {
    fps: f32,
    render_ms: Option<f32>,
}

/// Per-second ring for the 1m/5m/15m compositor load averages. Advanced lazily
/// (on each present and each query), so an idle screen costs nothing — no
/// periodic timer. Each bucket holds the frames composited that second and their
/// summed GPU render time; over a window, fps = Σframes ÷ seconds and render =
/// Σrender_ms ÷ Σframes (frame-weighted).
struct LoadTracker {
    start: Instant,
    frames: [u32; Self::WINDOW],
    render_sum: [f32; Self::WINDOW],
    /// Absolute second index (since `start`) of the newest bucket, at `head`.
    cur_sec: u64,
    head: usize,
}

impl LoadTracker {
    const WINDOW: usize = 900; // 15 minutes of one-second buckets
    const SPANS: [usize; 3] = [60, 300, 900]; // 1m / 5m / 15m

    fn new(now: Instant) -> Self {
        LoadTracker {
            start: now,
            frames: [0; Self::WINDOW],
            render_sum: [0.0; Self::WINDOW],
            cur_sec: 0,
            head: 0,
        }
    }

    /// Roll the ring forward to `now`, zeroing buckets for any elapsed seconds so
    /// idle time correctly dilutes the averages.
    fn advance_to(&mut self, now: Instant) {
        let target = now.duration_since(self.start).as_secs();
        let gap = target.saturating_sub(self.cur_sec);
        if gap == 0 {
            return;
        }
        if gap as usize >= Self::WINDOW {
            self.frames = [0; Self::WINDOW];
            self.render_sum = [0.0; Self::WINDOW];
            self.head = 0;
        } else {
            for _ in 0..gap {
                self.head = (self.head + 1) % Self::WINDOW;
                self.frames[self.head] = 0;
                self.render_sum[self.head] = 0.0;
            }
        }
        self.cur_sec = target;
    }

    /// Record one composited frame (GPU `render_ms`) at `now`.
    fn record(&mut self, now: Instant, render_ms: f32) {
        self.advance_to(now);
        self.frames[self.head] += 1;
        self.render_sum[self.head] += render_ms.max(0.0);
    }

    /// The 1m/5m/15m averages as of `now`. Each divides by the seconds actually
    /// available, so early readings aren't diluted by the not-yet-elapsed window.
    fn averages(&mut self, now: Instant) -> [LoadAvg; 3] {
        self.advance_to(now);
        let avail = (self.cur_sec as usize + 1).min(Self::WINDOW);
        Self::SPANS.map(|span| {
            let n = span.min(avail);
            let mut frames = 0u64;
            let mut render = 0.0f32;
            for i in 0..n {
                let idx = (self.head + Self::WINDOW - i) % Self::WINDOW;
                frames += self.frames[idx] as u64;
                render += self.render_sum[idx];
            }
            LoadAvg {
                fps: frames as f32 / n as f32,
                render_ms: (frames > 0).then(|| render / frames as f32),
            }
        })
    }
}

/// Top-level compositor state.
/// Cached window identity (WM_CLASS / type / title) for rule matching — read on
/// map and refreshed on the relevant `PropertyNotify`. Kept alongside `gfx` so
/// the `wm` crate stays X-agnostic.
#[derive(Default, Clone)]
struct WinIdentity {
    instance: String,
    class: String,
    window_type: String,
    title: String,
}

/// OSD text colours by kind: content (light), ack (green), error (red), cool (blue).
const OSD_FG: [f32; 3] = [0.92, 0.98, 1.0];
const OSD_ACK: [f32; 3] = [0.62, 1.0, 0.72];
const OSD_ERR: [f32; 3] = [1.0, 0.55, 0.5];
const OSD_COOL: [f32; 3] = [0.50, 0.78, 1.0];

/// Lifecycle phase of the on-screen toast.
enum OsdPhase {
    In,
    Hold,
    Out,
}

/// What the OSD did on a frame tick: nothing, a moving frame (repaint), or a
/// static hold (keep the clock alive to count down, but don't repaint).
#[derive(PartialEq)]
enum OsdTick {
    Idle,
    Moving,
    Holding,
}

/// On-screen notification state: a `presence` fade (0→1 in, 1→0 out) drives both
/// the slide and the alpha; `hold_remaining` is the time left at full presence.
struct OsdState {
    text: String,
    presence: Fade,
    phase: OsdPhase,
    hold_remaining: f64,
    open: OsdEffect,
    close: OsdEffect,
    color: [f32; 3],
}

/// Map a config OSD effect to the backend's rendering effect. `OsdEffect` (the
/// param) is `config::OsdEffect`; the return type is the neutral `backend::OsdEffect`
/// — named through `backend::`, not the GL crate's re-export, so the seam holds.
fn osd_effect(e: OsdEffect) -> backend::OsdEffect {
    match e {
        OsdEffect::Fade => backend::OsdEffect::Fade,
        OsdEffect::Slide => backend::OsdEffect::Slide,
        OsdEffect::Pop => backend::OsdEffect::Pop,
        OsdEffect::Unroll => backend::OsdEffect::Unroll,
        OsdEffect::Stretch => backend::OsdEffect::Stretch,
    }
}

pub struct App {
    pub x: XConn,
    windows: WindowStack,
    overlay: Window,
    backend: Option<Box<dyn Backend>>,
    /// What the active backend can render (cached from `backend.caps()` at build).
    /// `session` gates shader-only effects to fade when a cap is missing.
    caps: BackendCaps,
    gfx: HashMap<WindowId, WinGfx>,
    /// Cached per-window identity (WM_CLASS / type / title) for rule matching.
    identities: HashMap<WindowId, WinIdentity>,
    /// The EWMH active (focused) window (`_NET_ACTIVE_WINDOW`), for inactive-dim.
    /// `None` if no EWMH WM sets it → dimming stays inert.
    active_window: Option<WindowId>,
    dirty: bool,
    /// Damage accumulated for the next composite (screen coords) — the paint region
    /// unless a structural change forces a full repaint.
    frame_damage: Region,
    /// A structural change this frame → repaint the whole screen (safe default; only
    /// same-frame DamageNotify batches stay partial).
    force_full: bool,
    /// Last frame's damage extent per animating window (scale pop / wobble), so a
    /// tick can repaint `prev ∪ curr` — the retreating side of a moving animation
    /// — under use-damage instead of forcing a full repaint. Empty when idle.
    anim_damage: HashMap<WindowId, Rect>,
    /// Own-damage of recent presented frames (newest first) for EGL buffer-age
    /// partial repaint; bounded to `MAX_BUFFER_AGE`.
    damage_history: VecDeque<Region>,
    /// Whether the screen is redirected (i.e. we are compositing). When false we
    /// have unredirected + unmapped the overlay so a fullscreen window bypasses
    /// the compositor (unredir-if-possible).
    redirected: bool,
    /// calloop handle, kept so event handlers can (re)arm the fade frame clock.
    loop_handle: Option<LoopHandle<'static, App>>,
    /// calloop stop handle: a shutdown request (SIGTERM/SIGINT, or `ricomctl quit`)
    /// calls `stop()` to break `event_loop.run`, so it returns into the single
    /// `teardown()` path. Read by the signal source (Linux) and the control channel;
    /// allow it to be unused where neither reader is compiled.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    loop_signal: Option<LoopSignal>,
    /// The frame-clock timer while any window is fading; `None` when settled, so
    /// there are zero timer wakeups on an idle screen.
    frame_timer: Option<RegistrationToken>,
    /// Wall-clock of the previous frame-clock tick, for computing `dt`.
    last_frame: Option<Instant>,
    /// Effect settings (fade, shadow, unredir, background) from the config file.
    config: Config,
    /// Where the config was loaded from, so `SIGHUP` re-reads the same source.
    /// Only consulted by the Linux-only reload path.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    config_path: Option<PathBuf>,
    /// Whether the on-demand FPS HUD is currently visible (toggled by the hotkey).
    show_fps: bool,
    /// The resolved FPS-toggle hotkey as `(keycode, modifier_mask)`, or `None`
    /// when unbound (no/invalid hotkey, or the key isn't on the keyboard).
    fps_key: Option<(u8, u16)>,
    /// Runtime HUD corner (starts at `config.fps.corner`, moved by the arrow keys).
    hud_corner: HudCorner,
    /// Directional HUD-move hotkeys: `(keycode, modifier_mask, direction)`.
    move_keys: Vec<(u8, u16, Move)>,
    /// In-flight HUD auto-hop (sliding to a new corner), else `None`.
    hud_move: Option<HudMove>,
    /// Periodic auto-hop timer source; present iff `[fps] auto_move` is on.
    automove_timer: Option<RegistrationToken>,
    /// xorshift PRNG state for picking a random corner.
    rng: u64,
    /// Cached display refresh rate (Hz) for the HUD graph's budget; refreshed on RandR.
    refresh_hz: f64,
    /// Rolling frame-rate meter, sampled each composite while redirected.
    fps_meter: FpsMeter,
    /// 1m/5m/15m compositor-load ring, fed one sample per composited frame; shown
    /// in the HUD and logged on `SIGUSR1`.
    load: LoadTracker,
    /// Bound control-socket path, kept so we can unlink it on graceful exit.
    /// `None` when the socket failed to bind (control channel simply unavailable;
    /// signals remain). Startup reclaim handles the Ctrl-C-left-stale case.
    #[cfg(unix)]
    socket_path: Option<PathBuf>,
    /// Active on-screen notification toast, if any (driven by the frame clock).
    osd: Option<OsdState>,
}

/// Build a `proto::WinInfo` snapshot of one tracked window for the control channel.
#[cfg(unix)]
fn win_info(w: &Win, id: Option<&WinIdentity>) -> proto::WinInfo {
    proto::WinInfo {
        id: w.id,
        class: id.map(|i| i.class.clone()).unwrap_or_default(),
        instance: id.map(|i| i.instance.clone()).unwrap_or_default(),
        window_type: id.map(|i| i.window_type.clone()).unwrap_or_default(),
        title: id.map(|i| i.title.clone()).unwrap_or_default(),
        mapped: w.is_mapped(),
        x: w.x as i32,
        y: w.y as i32,
        width: w.width as u32,
        height: w.height as u32,
        opacity: w.fade.current() * w.dim.current(),
        closing: w.closing,
        anim: None, // filled in by `Inspect` only (see `win_anim`); `List` stays lean
    }
}

/// Read one `\n`-terminated line into `buf` (newline excluded), erroring past
/// `cap` bytes so a client can't make us buffer unboundedly.
#[cfg(unix)]
fn read_line_capped<R: std::io::BufRead>(r: &mut R, buf: &mut Vec<u8>, cap: usize) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};
    let mut byte = [0u8; 1];
    loop {
        if r.read(&mut byte)? == 0 {
            break; // EOF before newline — take what we have
        }
        if byte[0] == b'\n' {
            break;
        }
        buf.push(byte[0]);
        if buf.len() > cap {
            return Err(Error::new(ErrorKind::InvalidData, "control request too large"));
        }
    }
    Ok(())
}

impl Drop for App {
    fn drop(&mut self) {
        // Backstop for panic / early-error paths that never reach `teardown()`: only
        // the socket — no X calls during unwind (ordering-fragile). On the normal
        // path `teardown()` already took `socket_path`, so this is a no-op then.
        #[cfg(unix)]
        if let Some(p) = self.socket_path.take() {
            let _ = std::fs::remove_file(&p);
        }
    }
}

impl App {
    /// Connect to X and negotiate the extensions we depend on. `config` holds the
    /// effect settings; `config_path` is remembered for `SIGHUP` reloads.
    pub fn new(config: Config, config_path: Option<PathBuf>) -> Result<Self> {
        let x = XConn::connect()?;
        x.setup_extensions()?;
        let refresh_hz = x.refresh_hz().unwrap_or(60.0);
        Ok(App {
            x,
            windows: WindowStack::new(),
            overlay: 0,
            backend: None,
            caps: BackendCaps::all(), // replaced with the real caps when the backend is built
            gfx: HashMap::new(),
            identities: HashMap::new(),
            active_window: None,
            dirty: true,
            frame_damage: Region::new(),
            force_full: true,
            anim_damage: HashMap::new(),
            damage_history: VecDeque::new(),
            redirected: false,
            loop_handle: None,
            loop_signal: None,
            frame_timer: None,
            last_frame: None,
            show_fps: config.fps.enabled,
            hud_corner: hud_corner(&config.fps.corner),
            config,
            config_path,
            fps_key: None,
            move_keys: Vec::new(),
            hud_move: None,
            automove_timer: None,
            rng: seed_rng(),
            refresh_hz,
            fps_meter: FpsMeter::new(),
            load: LoadTracker::new(Instant::now()),
            #[cfg(unix)]
            socket_path: None,
            osd: None,
        })
    }

    /// Default animation duration (seconds) for live opacity re-targets not tied
    /// to an open/close/move spec — a config reload, or a resize that changes the
    /// effective opacity. Per-transition durations come from the resolved spec.
    fn anim_duration(&self) -> f64 {
        self.config.anim.duration
    }

    /// (Re)bind the FPS-HUD toggle hotkey: drop any previous grab, parse the
    /// configured spec, resolve it to a keycode, and passively grab it on the root
    /// (including the CapsLock/NumLock lock variants). Logged, never fatal.
    fn grab_fps_hotkey(&mut self) {
        const LOCK: u16 = 0x02; // CapsLock
        const MOD2: u16 = 0x10; // NumLock
        let variants = |mods: u16| [mods, mods | LOCK, mods | MOD2, mods | LOCK | MOD2];
        // Drop the previous grab first (a reload may change the combo).
        if let Some((kc, mods)) = self.fps_key.take() {
            for m in variants(mods) {
                let _ = self.x.ungrab_key(kc, m);
            }
        }
        for (kc, mods, _) in std::mem::take(&mut self.move_keys) {
            for m in variants(mods) {
                let _ = self.x.ungrab_key(kc, m);
            }
        }
        let spec = self.config.fps.hotkey.clone();
        let Some((mods, keysym)) = hotkey::parse_hotkey(&spec) else {
            tracing::warn!(hotkey = %spec, "fps: unparseable hotkey — HUD keys disabled");
            return;
        };
        // Toggle key.
        match self.x.keysym_to_keycode(keysym) {
            Ok(Some(kc)) => {
                for m in variants(mods) {
                    if let Err(e) = self.x.grab_key(kc, m) {
                        tracing::warn!(hotkey = %spec, "fps: grab_key failed: {e}");
                    }
                }
                self.fps_key = Some((kc, mods));
                tracing::info!(hotkey = %spec, keycode = kc, mods, "fps: HUD toggle hotkey bound");
            }
            Ok(None) => tracing::warn!(hotkey = %spec, "fps: key not on the keyboard — HUD toggle disabled"),
            Err(e) => tracing::warn!(hotkey = %spec, "fps: keymap lookup failed: {e}"),
        }
        // Directional move keys: the toggle's modifiers + the arrow keys.
        for (ks, dir) in [
            (0xFF51u32, Move::Left),
            (0xFF53u32, Move::Right),
            (0xFF52u32, Move::Up),
            (0xFF54u32, Move::Down),
        ] {
            if let Ok(Some(kc)) = self.x.keysym_to_keycode(ks) {
                for m in variants(mods) {
                    let _ = self.x.grab_key(kc, m);
                }
                self.move_keys.push((kc, mods, dir));
            }
        }
        if !self.move_keys.is_empty() {
            tracing::info!(mods, count = self.move_keys.len(), "fps: HUD move keys bound (modifier + arrows)");
        }
    }

    /// Re-read the config (from the same source) and apply it live. Called on
    /// `SIGHUP` (Linux only — the signal source needs `signalfd`). A parse error
    /// is logged and the running config is kept unchanged.
    #[cfg(target_os = "linux")]
    fn reload_config(&mut self) {
        match Config::load(self.config_path.as_deref()) {
            Ok(cfg) => {
                let source = self
                    .config_path
                    .clone()
                    .or_else(config::default_path)
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "defaults".to_string());
                let changes = cfg.diff(&self.config);
                if changes.is_empty() {
                    tracing::info!(%source, "config reloaded — no changes");
                } else {
                    tracing::info!(%source, changes = %changes.join(", "), "config reloaded");
                }
                let hotkey_changed = cfg.fps.hotkey != self.config.fps.hotkey;
                let corner_changed = cfg.fps.corner != self.config.fps.corner;
                let opacity_changed =
                    cfg.default_opacity != self.config.default_opacity || cfg.rules != self.config.rules;
                let dim_changed = cfg.dim != self.config.dim || cfg.rules != self.config.rules;
                let font_changed = cfg.font != self.config.font;
                self.config = cfg;
                log_config_warnings(&self.config);
                if let Some(b) = self.backend.as_mut() {
                    b.set_render_params(render_params(&self.config));
                    // Re-rasterise glyphs from the new font (or disable text) live.
                    if font_changed {
                        b.set_font(&self.config.font.path, self.config.font.size);
                    }
                }
                // A changed FPS hotkey: drop the old grabs and bind the new combo.
                if hotkey_changed {
                    self.grab_fps_hotkey();
                }
                // A changed config corner repositions the runtime HUD corner.
                if corner_changed {
                    self.hud_corner = hud_corner(&self.config.fps.corner);
                }
                // A reload may toggle auto-move on/off (interval/duration take effect
                // at the next hop). Only (dis)arm on an actual state change so an
                // unrelated reload doesn't reset the countdown.
                if self.automove_timer.is_some() != self.config.fps.auto_move {
                    self.set_automove(self.config.fps.auto_move);
                }
                // Changed default_opacity/rules: re-target every window's opacity
                // (blur/shadow/corner re-resolve on the repaint below).
                if opacity_changed {
                    let ids: Vec<WindowId> = self
                        .windows
                        .iter_bottom_to_top()
                        .filter(|w| w.is_mapped() && !w.closing)
                        .map(|w| w.id)
                        .collect();
                    let d = self.anim_duration();
                    for id in &ids {
                        let o = self.read_opacity(*id);
                        self.windows.retarget_opacity(*id, o, d);
                    }
                    if !ids.is_empty() {
                        self.ensure_frame_timer();
                    }
                }
                // Changed [dim]/rules: re-seed the active window (the focus source
                // may have changed) and re-apply inactive-dim to every window.
                if dim_changed {
                    self.active_window = self.read_active_window();
                    self.apply_dim(true);
                }
                // unredir may have toggled; re-evaluate, then repaint.
                self.update_redirection();
                self.damage_full();
            }
            Err(e) => tracing::error!("config reload failed, keeping current: {e}"),
        }
    }

    /// Log the current 1m/5m/15m compositor load — present rate and mean GPU
    /// render time (+ % of the refresh budget). Called on `SIGUSR1` (Linux only).
    #[cfg(target_os = "linux")]
    fn log_load(&mut self) {
        let a = self.load.averages(Instant::now());
        let budget = 1000.0 / self.refresh_hz.max(1.0);
        let fmt = |la: &LoadAvg| match la.render_ms {
            Some(ms) => format!("{:.1}ms {:.0}%", ms, ms as f64 / budget * 100.0),
            None => "idle".to_string(),
        };
        tracing::info!(
            "load (1m/5m/15m):  fps {:.1} / {:.1} / {:.1}   render {} / {} / {}   (@{:.0}Hz, budget {:.1}ms)",
            a[0].fps, a[1].fps, a[2].fps,
            fmt(&a[0]), fmt(&a[1]), fmt(&a[2]),
            self.refresh_hz, budget
        );
    }

    /// Become the CM, redirect + acquire the overlay, build the GL backend, then
    /// run the compositing event loop until the process is killed.
    pub fn run(&mut self) -> Result<()> {
        log_config_warnings(&self.config);
        self.x.become_cm()?;
        self.x.select_root_events()?;
        self.x.select_screen_change()?;
        self.grab_fps_hotkey();

        self.overlay = self.x.get_overlay()?;
        self.x.overlay_input_passthrough(self.overlay)?;
        let visual = self.x.window_visual(self.overlay)?;
        self.x.redirect_subwindows()?;
        self.redirected = true;
        let backend = make_backend(&self.config, self.overlay, visual)?;
        self.caps = backend.caps(); // cache once; constant for a live backend
        self.backend = Some(backend);
        // Load the on-screen-text font (HUD/OSD/notify). A missing/invalid font just
        // disables text; the compositor runs regardless.
        if let Some(b) = self.backend.as_mut() {
            b.set_font(&self.config.font.path, self.config.font.size);
        }

        // Seed the stack + per-window resources from the current tree.
        for w in self.x.list_tree()? {
            if w.window == self.overlay {
                continue;
            }
            self.windows.add_top(Win::new(
                w.window, w.x, w.y, w.width, w.height, w.border_width, false, w.mapped,
            ));
            let _ = self.x.select_window_events(w.window);
            self.refresh_identity(w.window);
            // Already on screen at startup — show at its opacity with no fade-in.
            let o = self.read_opacity(w.window);
            self.windows.set_opacity_settled(w.window, o);
            if w.mapped {
                self.acquire_gfx(w.window);
            }
        }
        // A window may already be fullscreen at startup — pick the redirect state now.
        self.update_redirection();
        // Seed the active window (from the configured focus source) + apply the
        // initial inactive-dim instantly (no-op unless [dim] is enabled and a focus
        // signal is available).
        self.active_window = self.read_active_window();
        self.apply_dim(false);
        self.x.flush()?;
        self.composite();
        tracing::info!(
            mapped = self.windows.mapped_count(),
            redirected = self.redirected,
            "ricom compositing (Ctrl-C to quit)"
        );

        let mut event_loop: EventLoop<'static, App> = EventLoop::try_new().context("create event loop")?;
        let handle = event_loop.handle();
        self.loop_handle = Some(handle.clone());
        // Kept so a shutdown request (a caught signal, or `ricomctl quit`) can break
        // `run` below and fall through to the single `teardown()` path.
        self.loop_signal = Some(event_loop.get_signal());
        let fd = self
            .x
            .conn
            .stream()
            .as_fd()
            .try_clone_to_owned()
            .context("clone X connection fd")?;
        handle
            .insert_source(Generic::new(fd, Interest::READ, Mode::Level), |_r, _fd, app: &mut App| {
                // Just drain (which marks dirty on damage); the single composite
                // for this loop iteration happens in the run callback below, so
                // damage + fade-tick coalesce into one vsync-paced repaint.
                app.drain_x_events();
                Ok(PostAction::Continue)
            })
            .map_err(|e| anyhow::anyhow!("insert X source: {e}"))?;

        // SIGHUP -> reload the config file live; SIGUSR1 -> log the 1m/5m/15m load.
        // Linux-only: calloop's signal source is built on signalfd.
        #[cfg(target_os = "linux")]
        {
            let signals = Signals::new(&[
                Signal::SIGHUP,
                Signal::SIGUSR1,
                Signal::SIGTERM,
                Signal::SIGINT,
            ])
            .context("create signal source")?;
            handle
                .insert_source(signals, |event, _meta, app: &mut App| match event.signal() {
                    Signal::SIGHUP => app.reload_config(),
                    Signal::SIGUSR1 => app.log_load(),
                    // Graceful stop: break the loop so `run` returns into teardown().
                    // Catching SIGINT here replaces the default Ctrl-C terminate, so
                    // Ctrl-C now tears down cleanly too.
                    Signal::SIGTERM | Signal::SIGINT => {
                        tracing::info!(signal = ?event.signal(), "shutting down");
                        if let Some(s) = app.loop_signal.as_ref() {
                            s.stop();
                        }
                    }
                    _ => {}
                })
                .map_err(|e| anyhow::anyhow!("insert signal source: {e}"))?;
        }

        // Control channel: a Unix-domain socket `ricomctl` connects to — one more
        // calloop source beside the X fd + signals. Best-effort: a bind failure
        // just disables the control channel (signals still work).
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            use std::os::unix::net::{UnixListener, UnixStream};
            let path = proto::socket_path();
            // Reclaim a stale socket from a previous run: if nothing answers a
            // connect, it's dead — remove it so we can rebind. (Ctrl-C exit skips
            // the on-exit unlink, so this is the load-bearing cleanup.)
            if path.exists() && UnixStream::connect(&path).is_err() {
                let _ = std::fs::remove_file(&path);
            }
            match UnixListener::bind(&path) {
                Ok(listener) => {
                    let _ = listener.set_nonblocking(true);
                    // Lock down the /tmp fallback (XDG_RUNTIME_DIR is already 0700).
                    let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
                    let src = Generic::new(listener, Interest::READ, Mode::Level);
                    match handle.insert_source(src, |_r, listener, app: &mut App| {
                        // Level-triggered: accept everything ready, then WouldBlock.
                        loop {
                            match listener.accept() {
                                Ok((stream, _addr)) => app.serve_control_conn(stream),
                                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                                Err(e) => {
                                    tracing::debug!("control accept: {e}");
                                    break;
                                }
                            }
                        }
                        Ok(PostAction::Continue)
                    }) {
                        Ok(_) => {
                            self.socket_path = Some(path.clone());
                            tracing::info!(socket = %path.display(), "control channel listening");
                        }
                        Err(e) => {
                            tracing::warn!(socket = %path.display(), "control insert failed: {e}");
                            let _ = std::fs::remove_file(&path);
                        }
                    }
                }
                Err(e) => tracing::warn!(
                    socket = %path.display(),
                    "control socket bind failed: {e} — control channel disabled"
                ),
            }
        }

        // Auto-hop: arm the periodic HUD corner-mover if enabled in the config.
        self.set_automove(self.config.fps.auto_move);

        // Prime the pump: setup already flushed, which may have buffered events
        // inside x11rb that will never re-trigger the fd watch. Drain them (and
        // repaint) before blocking so we don't start out stalled.
        self.drain_x_events();
        if self.dirty {
            self.composite();
            self.dirty = false;
        }

        // One composite per loop iteration, after all sources (X events, fade
        // ticks) have run — coalesces every trigger into a single vsync-paced repaint.
        event_loop
            .run(None, self, |app| {
                if app.dirty {
                    app.composite();
                    app.dirty = false;
                }
            })
            .context("event loop")?;

        // The loop returns only once a shutdown was requested (SIGTERM/SIGINT, or
        // `ricomctl quit` → LoopSignal::stop()). Run the single teardown path.
        self.teardown();
        Ok(())
    }

    /// Tear the compositor down in reverse of `run()` setup — best-effort and
    /// idempotent. Reached when the event loop stops (a caught signal, or a
    /// `ricomctl quit`). A hard `SIGKILL` can't run this; the startup stale-socket
    /// reclaim covers that case.
    fn teardown(&mut self) {
        tracing::info!("tearing down: restoring the display");
        // 1. Control socket: unlink so a fresh ricom / ricomctl won't hit a dead one.
        #[cfg(unix)]
        if let Some(p) = self.socket_path.take() {
            let _ = std::fs::remove_file(&p);
        }
        // 2. Free every per-window pixmap + damage handle before the connection goes.
        for (_, g) in std::mem::take(&mut self.gfx) {
            self.free_gfx(g);
        }
        // 3. Tear down EGL/GL *before* releasing the overlay (its surface lives there).
        self.backend = None;
        // 4. Restore normal server drawing if we're still compositing.
        if self.redirected {
            let _ = self.x.unredirect_subwindows();
            self.redirected = false;
        }
        // 5. Give the composite overlay back to the server.
        if self.overlay != 0 {
            let _ = self.x.release_overlay();
            self.overlay = 0;
        }
        // 6. Push it all to the server before XConn drops and closes the connection.
        let _ = self.x.flush();
    }

    /// Serve one control-channel connection: read a single command line, dispatch
    /// it, write the reply, close. A short read/write timeout + a byte cap keep a
    /// slow or malformed *local* client (the only kind — the socket is user-private)
    /// from stalling the compositor for more than the timeout.
    #[cfg(unix)]
    fn serve_control_conn(&mut self, stream: std::os::unix::net::UnixStream) {
        use std::io::{BufReader, Write};
        let timeout = Duration::from_millis(250);
        let _ = stream.set_nonblocking(false); // accepted sockets are blocking; be explicit
        let _ = stream.set_read_timeout(Some(timeout));
        let _ = stream.set_write_timeout(Some(timeout));
        let mut reader = BufReader::new(stream);
        let mut line = Vec::with_capacity(256);
        let reply = match read_line_capped(&mut reader, &mut line, 64 * 1024) {
            Ok(()) => match proto::decode::<proto::Command>(&line) {
                Ok(cmd) => self.handle_command(cmd),
                Err(e) => proto::Reply::Error(format!("bad request: {e}")),
            },
            Err(e) => proto::Reply::Error(format!("read: {e}")),
        };
        let mut stream = reader.into_inner();
        let _ = stream.write_all(&proto::encode(&reply));
        let _ = stream.flush();
    }

    /// Dispatch one control command against live compositor state → the reply.
    #[cfg(unix)]
    fn handle_command(&mut self, cmd: proto::Command) -> proto::Reply {
        use proto::{Command as C, Reply};
        match cmd {
            C::Ping => {
                if self.config.osd.enabled && self.config.osd.ack {
                    self.show_osd(">> pong!".into(), self.config.osd.duration.min(1.2), OSD_ACK);
                }
                Reply::Text(format!(
                    "ricom {} (control v{})",
                    env!("CARGO_PKG_VERSION"),
                    proto::PROTOCOL_VERSION
                ))
            }
            C::Reload => {
                #[cfg(target_os = "linux")]
                {
                    self.reload_config();
                    if self.config.osd.enabled && self.config.osd.ack {
                        self.show_osd(">> config reloaded.".into(), self.config.osd.duration.min(1.2), OSD_ACK);
                    }
                    Reply::Ok
                }
                #[cfg(not(target_os = "linux"))]
                {
                    Reply::Error("reload is only supported on Linux".into())
                }
            }
            C::FpsToggle => {
                self.show_fps = !self.show_fps;
                self.damage_full();
                if self.config.osd.enabled && self.config.osd.ack {
                    let s = if self.show_fps { "on" } else { "off" };
                    self.show_osd(format!(">> FPS HUD {s}."), self.config.osd.duration.min(1.2), OSD_ACK);
                }
                Reply::Ok
            }
            C::FpsAutoMove { enable } => {
                let on = enable.unwrap_or(!self.config.fps.auto_move);
                self.set_automove(on);
                if self.config.osd.enabled && self.config.osd.ack {
                    let s = if on { "on" } else { "off" };
                    self.show_osd(format!(">> fps auto {s}."), self.config.osd.duration.min(1.2), OSD_ACK);
                }
                Reply::Text(format!("fps auto-move {}", if on { "on" } else { "off" }))
            }
            C::List => {
                // Heal any startup-adopted blanks before reporting (their first read
                // may have raced); collect ids first — can't borrow `windows` + mutate
                // `identities` at once.
                let ids: Vec<WindowId> = self.windows.iter_bottom_to_top().map(|w| w.id).collect();
                for id in ids {
                    self.ensure_identity(id);
                }
                let list: Vec<proto::WinInfo> = self
                    .windows
                    .iter_bottom_to_top()
                    .map(|w| win_info(w, self.identities.get(&w.id)))
                    .collect();
                // Also show the mapped windows on-screen (one per line).
                if self.config.osd.enabled {
                    let text = list
                        .iter()
                        .filter(|w| w.mapped)
                        .map(|w| {
                            let cls: String = w.class.chars().take(12).collect();
                            format!("0x{:07x}  {cls:<12}  {}", w.id, w.title)
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    if !text.is_empty() {
                        self.show_osd(text, self.config.osd.duration, OSD_FG);
                    }
                }
                Reply::Windows(list)
            }
            C::Inspect { win } => {
                if self.windows.get(win).is_none() {
                    if self.config.osd.enabled {
                        self.show_osd(">> no such window.".into(), self.config.osd.duration.min(1.2), OSD_ERR);
                    }
                    return Reply::Error(format!("no such window {win:#x}"));
                }
                self.ensure_identity(win); // heal a startup-adopted blank before reporting
                let w = self.windows.get(win).expect("existence checked above");
                let mut info = win_info(w, self.identities.get(&win));
                info.anim = Some(Box::new(self.win_anim(win))); // per-window rules (inspect only)
                if self.config.osd.enabled {
                    let cls: String = info.class.chars().take(16).collect();
                    self.show_osd(
                        format!("0x{:07x} {cls} {}x{}", info.id, info.width, info.height),
                        self.config.osd.duration,
                        OSD_FG,
                    );
                }
                Reply::Window(info)
            }
            C::Notify { text, timeout_ms } => {
                if !self.config.osd.enabled {
                    return Reply::Error("osd disabled in config".into());
                }
                let hold = timeout_ms
                    .map(|ms| ms as f64 / 1000.0)
                    .unwrap_or(self.config.osd.duration);
                self.show_osd(text, hold, OSD_FG);
                Reply::Ok
            }
            C::Version => {
                let banner = format!(
                    "ricom {} (control v{})",
                    env!("CARGO_PKG_VERSION"),
                    proto::PROTOCOL_VERSION
                );
                if self.config.osd.enabled {
                    self.show_osd(format!("_.* {banner} *._"), self.config.osd.duration, OSD_COOL);
                }
                Reply::Text(banner)
            }
            C::Animate { win, effect, params } => self.animate_window(win, &effect, &params),
            C::SetAnim { category, effect, params } => self.set_anim(&category, &effect, &params),
            C::GetAnim => self.get_anim(),
            C::Unredir { enable } => self.set_unredir(enable),
            C::Font { path, size } => self.set_font_cmd(path, size),
            C::Quit => {
                tracing::info!("quit requested via control channel");
                // Break the loop → `run` returns into the single teardown path. The
                // reply below is written synchronously before calloop observes the
                // stop flag, so the client still receives it.
                if let Some(s) = self.loop_signal.as_ref() {
                    s.stop();
                }
                proto::Reply::Text("shutting down ...".into())
            }
        }
    }

    /// Toggle unredir-if-possible live (session-only; a `Reload`/SIGHUP reverts to the
    /// config file). `enable = Some(v)` sets it, `None` flips the current state. We
    /// re-run the redirect decision straight away, so a lone fullscreen window starts
    /// compositing (`off`) or is allowed to page-flip past us (`on`) immediately.
    fn set_unredir(&mut self, enable: Option<bool>) -> proto::Reply {
        let on = enable.unwrap_or(!self.config.unredir);
        self.config.unredir = on;
        self.update_redirection();
        if self.config.osd.enabled && self.config.osd.ack {
            let s = if on { "on" } else { "off" };
            self.show_osd(format!(">> unredir {s}."), self.config.osd.duration.min(1.2), OSD_ACK);
        }
        proto::Reply::Text(format!(
            "unredir {} — {}",
            if on { "on (fullscreen may bypass)" } else { "off (always compositing)" },
            if self.redirected { "compositing now" } else { "fullscreen bypass now" },
        ))
    }

    /// Live-swap the on-screen-text font (session-only; a `Reload`/SIGHUP reverts to the
    /// config `[font]`). Mirrors `set_unredir`: it keeps `self.config.font` in sync so the
    /// reload diff re-asserts the file's font, then rebuilds the backend's glyph cache. An
    /// empty/unreadable/unparsable path disables on-screen text (the compositor keeps
    /// running) and returns an error so `ricomctl` reports it.
    fn set_font_cmd(&mut self, path: String, size: Option<f32>) -> proto::Reply {
        let sz = size.unwrap_or(self.config.font.size);
        // Keep the in-memory config current so a later Reload/SIGHUP reverts to the file.
        self.config.font.path = path.clone();
        self.config.font.size = sz;
        // Scope the backend borrow so `show_osd` (needs &mut self) can run afterwards.
        let loaded = {
            let Some(b) = self.backend.as_mut() else {
                return proto::Reply::Error("backend not ready".into());
            };
            b.set_font(&path, sz);
            b.has_text()
        };
        if loaded {
            if self.config.osd.enabled && self.config.osd.ack {
                let name = std::path::Path::new(&path)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.clone());
                // The ack renders in the just-loaded font — instant visual confirmation.
                self.show_osd(format!(">> font: {name}"), self.config.osd.duration.min(1.5), OSD_ACK);
            }
            proto::Reply::Text(format!("font set: {path}"))
        } else {
            proto::Reply::Error(format!("font '{path}' unusable — on-screen text disabled"))
        }
    }

    /// Start (or replace) the OSD toast with `text` in `color`, held for `hold` seconds.
    #[cfg(unix)]
    fn show_osd(&mut self, text: String, hold: f64, color: [f32; 3]) {
        self.osd = Some(OsdState {
            text,
            presence: Fade::animating(0.0, 1.0, self.config.osd.in_dur),
            phase: OsdPhase::In,
            hold_remaining: hold,
            open: self.config.osd.open,
            close: self.config.osd.close,
            color,
        });
        self.ensure_frame_timer();
        self.damage_full();
    }

    /// Play a one-shot, self-restoring in-place effect on `win` — the transform
    /// vocabulary shared by `ricomctl animate` and the focus trigger. Each self-
    /// settles back to rest (spin→0, scale→1, translate→0); `reset` snaps back at
    /// once. Returns `false` for an unknown effect name (`"none"` included); a
    /// no-op if `win` is untracked.
    /// Apply an in-place effect to `win` with optional `key=value` param overrides
    /// (empty ⇒ the configured `[anim]` defaults). Strictly validates the effect name
    /// and every param key/value; `Ok(())` = applied, `Err(msg)` = a user-facing reason.
    fn apply_effect(&mut self, win: WindowId, effect: &str, params: &[(String, String)]) -> Result<(), String> {
        use std::f64::consts::{PI, TAU};
        use wm::anim::{Axis, Easing};
        const DUR: f64 = 0.6;
        // Reject effects the active backend can't render (e.g. shader effects on XRender),
        // so `ricomctl animate` reports it rather than silently arming a transform the
        // compositor would drop. pop/stretch/unroll/slide/reset are pure geometry — always ok.
        let (renderable, needs) = match effect {
            "spin" | "wave" | "ripple" | "drain" => (self.caps.shaders, "a shader-capable"),
            "wobble" => (self.caps.mesh, "a mesh-capable"),
            _ => (true, ""),
        };
        if !renderable {
            return Err(format!("effect '{effect}' needs {needs} backend (the active backend can't render it)"));
        }
        match effect {
            "spin" => {
                check_keys(effect, params)?;
                let rad = param_f32(params, "degrees")?.map(|d| f64::from(d) * PI / 180.0).unwrap_or(TAU);
                let dur = param_f32(params, "duration")?.map(f64::from).unwrap_or(DUR);
                let ease = param_easing(params)?.unwrap_or(Easing::EaseOut);
                self.windows.spin_in(win, rad, dur, ease);
            }
            "pop" | "stretch" | "unroll" => {
                check_keys(effect, params)?;
                let (default_from, axis) = match effect {
                    "pop" => (0.4, Axis::Both),
                    "stretch" => (0.0, Axis::X),
                    _ => (0.0, Axis::Y), // unroll
                };
                let from = f64::from(param_f32(params, "from")?.unwrap_or(default_from));
                let dur = param_f32(params, "duration")?.map(f64::from).unwrap_or(DUR);
                let ease = param_easing(params)?.unwrap_or(Easing::EaseOut);
                self.windows.scale_in(win, from, dur, axis, ease);
            }
            "slide" => {
                check_keys(effect, params)?;
                let dx = param_f32(params, "dx")?.unwrap_or(-160.0);
                let dy = param_f32(params, "dy")?.unwrap_or(0.0);
                let dur = param_f32(params, "duration")?.map(f64::from).unwrap_or(DUR);
                let ease = param_easing(params)?.unwrap_or(Easing::EaseOut);
                self.windows.translate_in(win, [dx, dy], dur, ease);
            }
            "wobble" => {
                check_keys(effect, params)?;
                let spring = param_f32(params, "spring")?.unwrap_or(self.config.anim.wobble_spring);
                let friction = param_f32(params, "friction")?.unwrap_or(self.config.anim.wobble_friction);
                if let Some(rest) = self
                    .windows
                    .get(win)
                    .map(|w| [w.x as f32, w.y as f32, w.width as f32, w.height as f32])
                {
                    let p = 36.0; // perturb the outer rect outward, then spring back to rest
                    let old = [rest[0] - p, rest[1] - p, rest[2] + 2.0 * p, rest[3] + 2.0 * p];
                    self.windows.wobble_to(win, old, rest, spring, friction);
                }
            }
            "wave" => {
                check_keys(effect, params)?;
                let amp = param_f32(params, "amplitude")?.unwrap_or(self.config.anim.wave_amplitude);
                let wl = param_f32(params, "wavelength")?.unwrap_or(self.config.anim.wave_wavelength);
                let speed = param_f32(params, "speed")?.unwrap_or(self.config.anim.wave_speed);
                let axis = param_axis(params)?.unwrap_or(Axis::X);
                let duration = param_f32(params, "duration")?.unwrap_or(self.config.anim.wave_duration);
                self.windows.wave_to(win, amp, wl, speed, axis, duration);
            }
            "ripple" => {
                check_keys(effect, params)?;
                let amp = param_f32(params, "amplitude")?.unwrap_or(self.config.anim.ripple_amplitude);
                let wl = param_f32(params, "wavelength")?.unwrap_or(self.config.anim.ripple_wavelength);
                let speed = param_f32(params, "speed")?.unwrap_or(self.config.anim.ripple_speed);
                let r0 = param_f32(params, "r0")?.unwrap_or(self.config.anim.ripple_r0);
                let duration = param_f32(params, "duration")?.unwrap_or(self.config.anim.ripple_duration);
                self.windows.ripple_to(win, [0.5, 0.5], amp, wl, speed, r0, duration);
            }
            "drain" => {
                check_keys(effect, params)?;
                let turns = param_f32(params, "turns")?.unwrap_or(self.config.anim.drain_turns);
                let turb = param_f32(params, "turbulence")?.unwrap_or(self.config.anim.drain_turbulence);
                let depth = param_f32(params, "depth")?.unwrap_or(self.config.anim.drain_depth);
                let dur = param_f32(params, "duration")?
                    .map(f64::from)
                    .unwrap_or(f64::from(self.config.anim.drain_duration));
                // Non-destructive: drain to `depth` and HOLD there (shrink to a tiny
                // point and stay) — unlike the close driver, which reaps the window.
                // Restore with `ricomctl animate <win> reset`.
                self.windows.drain_to(win, turns, turb, depth, dur, burn_seed(win));
            }
            "reset" => {
                check_keys(effect, params)?;
                self.windows.reset_transforms(win);
            }
            _ => {
                return Err(format!(
                    "unknown effect '{effect}' (spin|pop|stretch|unroll|slide|wobble|wave|ripple|drain|reset)"
                ));
            }
        }
        self.ensure_frame_timer();
        self.damage_full();
        Ok(())
    }

    /// `ricomctl animate <win> <effect> [k=v …]` — apply an in-place effect with
    /// optional param overrides, mapping the outcome to a reply. Unknown window →
    /// `Reply::Error`; an unknown effect or bad param → `apply_effect`'s message.
    #[cfg(unix)]
    fn animate_window(&mut self, win: WindowId, effect: &str, params: &[(String, String)]) -> proto::Reply {
        if self.windows.get(win).is_none() {
            return proto::Reply::Error(format!("no such window {win:#x}"));
        }
        match self.apply_effect(win, effect, params) {
            Ok(()) => proto::Reply::Ok,
            Err(e) => proto::Reply::Error(e),
        }
    }

    /// `ricomctl set <category> <effect> [k=v…]` — live-select a transition's effect
    /// (+ optional params), session-only: mutates `self.config.anim.<category>`, so a
    /// `reload`/SIGHUP reverts. Takes effect on the next open/close/move (resolved
    /// per-window). `focus` is a bare effect name (no params in the current model).
    #[cfg(unix)]
    fn set_anim(&mut self, category: &str, effect: &str, params: &[(String, String)]) -> proto::Reply {
        use config::{AnimSel, Category};
        if category == "focus" {
            if !params.is_empty() {
                return proto::Reply::Error("focus takes no params".into());
            }
            if !config::FOCUS_EFFECTS.contains(&effect) {
                return proto::Reply::Error(format!(
                    "unknown focus effect '{effect}' (valid: {})",
                    config::FOCUS_EFFECTS.join(", ")
                ));
            }
            // Reject a focus effect the active backend can't render (wobble→mesh; the
            // rest are shader effects). On a full-caps backend this never triggers.
            let renderable = if effect == "wobble" { self.caps.mesh } else { self.caps.shaders };
            if !renderable {
                return proto::Reply::Error(format!(
                    "focus effect '{effect}' needs a shader/mesh-capable backend (the active backend can't render it)"
                ));
            }
            self.config.anim.focus = effect.to_string();
        } else {
            let cat = match category {
                "open" => Category::Open,
                "close" => Category::Close,
                "move" => Category::Move,
                _ => {
                    return proto::Reply::Error(format!(
                        "unknown category '{category}' (open|close|move|focus)"
                    ));
                }
            };
            let sel = if params.is_empty() {
                if !config::PRESETS.contains(&effect) {
                    return proto::Reply::Error(format!("unknown effect '{effect}' (see `ricomctl effects`)"));
                }
                AnimSel::Preset(effect.to_string())
            } else {
                // With params: validate keys via the shared schema (serde won't reject
                // unknown keys), then build the one-block spec.
                if proto::effect_params(effect).is_none() {
                    return proto::Reply::Error(format!("effect '{effect}' takes no params"));
                }
                if let Err(e) = check_keys(effect, params) {
                    return proto::Reply::Error(e);
                }
                match config::anim_spec_from(effect, params) {
                    Ok(spec) => AnimSel::Spec(spec),
                    Err(e) => return proto::Reply::Error(e),
                }
            };
            match cat {
                Category::Open => self.config.anim.open = sel,
                Category::Close => self.config.anim.close = sel,
                Category::Move => self.config.anim.r#move = sel,
            }
        }
        tracing::info!(category, effect, params = params.len(), "set anim (session-only)");
        if self.config.osd.enabled && self.config.osd.ack {
            self.show_osd(format!(">> {category} = {effect}"), self.config.osd.duration.min(1.2), OSD_ACK);
        }
        proto::Reply::Ok
    }

    /// `ricomctl get` — report the current + compiled-default effect & resolved
    /// params for each transition category. Reflects live `set` overrides (they
    /// mutate `self.config.anim`); a `reload`/SIGHUP reverts to the config file.
    #[cfg(unix)]
    fn get_anim(&self) -> proto::Reply {
        let cur = &self.config.anim;
        let def = config::Anim::default();
        let mut anims = Vec::with_capacity(4);
        for (event, c, d) in [
            ("open", &cur.open, &def.open),
            ("close", &cur.close, &def.close),
            ("move", &cur.r#move, &def.r#move),
        ] {
            anims.push(proto::AnimInfo {
                event: event.to_string(),
                effect: c.label(),
                params: config::effective_params(c, cur),
                default_effect: d.label(),
                default_params: config::effective_params(d, &def),
            });
        }
        anims.push(proto::AnimInfo {
            event: "focus".to_string(),
            effect: cur.focus.clone(),
            params: config::focus_params(&cur.focus, cur),
            default_effect: def.focus.clone(),
            default_params: config::focus_params(&def.focus, &def),
        });
        proto::Reply::Anims(anims)
    }

    /// Per-window effective animation for `ricomctl inspect` — each transition's
    /// effect label, with a matching `[[rule]]` override taking precedence over the
    /// global `[anim]`. `overridden` lists the categories a rule set (rule specs are
    /// already expanded, so a preset override shows its block names).
    #[cfg(unix)]
    fn win_anim(&self, id: WindowId) -> proto::WinAnim {
        let rr = self.resolve_rules(id);
        let g = &self.config.anim;
        let mut overridden = Vec::new();
        let open = match &rr.open {
            Some(s) => {
                overridden.push("open".to_string());
                s.label()
            }
            None => g.open.label(),
        };
        let close = match &rr.close {
            Some(s) => {
                overridden.push("close".to_string());
                s.label()
            }
            None => g.close.label(),
        };
        let r#move = match &rr.r#move {
            Some(s) => {
                overridden.push("move".to_string());
                s.label()
            }
            None => g.r#move.label(),
        };
        let focus = match &rr.focus {
            Some(f) => {
                overridden.push("focus".to_string());
                f.clone()
            }
            None => g.focus.clone(),
        };
        proto::WinAnim { open, close, r#move, focus, overridden }
    }

    /// The focus-triggered effect for `id`: a matching rule's `focus`, else the
    /// global `[anim] focus`. `"none"` = no focus effect.
    fn focus_effect(&self, id: WindowId) -> String {
        self.resolve_rules(id).focus.unwrap_or_else(|| self.config.anim.focus.clone())
    }

    /// Update the active (focused) window: re-apply inactive-dim and fire the
    /// per-window focus effect on the newly-focused window. No-op if unchanged.
    fn set_active_window(&mut self, new: Option<WindowId>) {
        if new == self.active_window {
            return;
        }
        self.active_window = new;
        self.apply_dim(true);
        if let Some(id) = new
            && self.windows.get(id).is_some()
        {
            let fx = self.focus_effect(id);
            if fx != "none" {
                let _ = self.apply_effect(id, &fx, &[]); // focus effect: configured defaults, no params
            }
        }
        self.damage_full();
    }

    /// Effective opacity target for a window: an explicit `_NET_WM_WINDOW_OPACITY`
    /// wins; else a matching rule's `opacity`; else `config.default_opacity`.
    fn read_opacity(&self, win: WindowId) -> f64 {
        match self.x.get_window_opacity(win) {
            Ok(Some(o)) => o, // explicit client opacity wins
            Ok(None) => self.resolve_rules(win).opacity.unwrap_or(self.config.default_opacity),
            Err(e) => {
                tracing::debug!(window = win, "opacity read failed: {e}");
                self.config.default_opacity
            }
        }
    }

    /// Read + cache a window's identity (WM_CLASS / type / title) for rule matching.
    fn refresh_identity(&mut self, win: WindowId) {
        let (instance, class) = self.x.get_wm_class(win).ok().flatten().unwrap_or_default();
        let window_type = self.x.get_window_type(win).ok().flatten().unwrap_or_default();
        let title = self.x.get_window_title(win).ok().flatten().unwrap_or_default();
        self.identities.insert(win, WinIdentity { instance, class, window_type, title });
    }

    /// Re-read a window's identity only when we have nothing cached yet — either no
    /// entry, or a fully-blank one. That blank case is the startup-adoption gap: a
    /// window already open when ricom starts is read once at `run()` time, and if that
    /// first `WM_CLASS`/title read comes back empty it never fires a `PropertyNotify`
    /// (the props don't change again), so the blank sticks. Healing it lazily — on the
    /// paths that actually consume identity (`list`/`inspect`, rule resolution at
    /// open/close/move) — costs 3 GetProperty round-trips once, then nothing.
    fn ensure_identity(&mut self, win: WindowId) {
        let blank = self.identities.get(&win).is_none_or(|i| {
            i.class.is_empty() && i.instance.is_empty() && i.title.is_empty() && i.window_type.is_empty()
        });
        if blank {
            self.refresh_identity(win);
        }
    }

    /// Build a window's [`WindowMatch`] (cached identity + live fullscreen state)
    /// for rule matching / animation resolution.
    fn window_match(&self, win: WindowId) -> WindowMatch {
        let id = self.identities.get(&win);
        WindowMatch {
            class: id.map(|i| i.class.clone()).unwrap_or_default(),
            instance: id.map(|i| i.instance.clone()).unwrap_or_default(),
            window_type: id.map(|i| i.window_type.clone()).unwrap_or_default(),
            title: id.map(|i| i.title.clone()).unwrap_or_default(),
            fullscreen: self.windows.get(win).is_some_and(|w| self.covers_screen(w)),
        }
    }

    /// Fold the config rules for a window into the net per-window overrides.
    fn resolve_rules(&self, win: WindowId) -> RuleResult {
        self.config.resolve(&self.window_match(win))
    }

    /// The inactive-dim brightness target for a window: `1.0` (full) if it's the
    /// active window, dim is disabled, or a rule exempts it (`dim = false`); else
    /// `1 - [dim] strength`.
    fn dim_target(&self, id: WindowId) -> f64 {
        // Inert when dim is off, or when no active window is known (no EWMH WM) —
        // don't dim *everything* just because we can't tell which is focused.
        if !self.config.dim.enabled || self.active_window.is_none() || Some(id) == self.active_window {
            return 1.0;
        }
        if self.resolve_rules(id).dim == Some(false) {
            return 1.0;
        }
        (1.0 - self.config.dim.strength).clamp(0.0, 1.0)
    }

    /// (Re)apply the inactive-dim target to every mapped, non-closing window — on
    /// focus change, reload, or startup. `animate` eases over the anim duration
    /// (focus change) vs. snapping instantly (startup).
    fn apply_dim(&mut self, animate: bool) {
        let d = if animate { self.anim_duration() } else { 0.0 };
        let ids: Vec<WindowId> = self
            .windows
            .iter_bottom_to_top()
            .filter(|w| w.is_mapped() && !w.closing)
            .map(|w| w.id)
            .collect();
        for id in ids {
            let t = self.dim_target(id);
            self.windows.set_dim(id, t, d);
        }
        if animate {
            self.ensure_frame_timer();
        }
    }

    /// The X-input-focused window mapped to a tracked top-level (else `None`) —
    /// the `x11` dim focus source.
    fn focused_top_level(&self) -> Option<WindowId> {
        let f = self.x.get_input_focus().ok().flatten()?;
        self.windows.get(f).map(|_| f)
    }

    /// The current active window from the configured focus source — the root
    /// `_NET_ACTIVE_WINDOW` (`ewmh`) or X input focus (`x11`). Used to (re)seed
    /// `active_window` at startup and on reload.
    fn read_active_window(&self) -> Option<WindowId> {
        match self.config.dim.focus {
            FocusSource::Ewmh => self.x.get_active_window().ok().flatten(),
            FocusSource::X11 => self.focused_top_level(),
        }
    }

    /// Root screen size in px.
    fn screen(&self) -> (i32, i32) {
        (self.x.root_width as i32, self.x.root_height as i32)
    }

    /// A window's outer rect `[x, y, w, h]` by id (border included), or zeros if
    /// untracked.
    fn outer_rect_of(&self, id: WindowId) -> [f32; 4] {
        self.windows.get(id).map(|w| self.outer_rect(w)).unwrap_or([0.0; 4])
    }

    /// Resolve the animation spec for `id`'s `cat` and start each primitive block
    /// (opacity / scale / translate / wobble / burn). `destroyed` marks a close
    /// for removal (vs merely unmapped) on completion. Returns whether something
    /// visible was started — close sites use it to keep the "animate vs drop now"
    /// gate. Move is driven separately by `ConfigureNotify` (it needs old+new
    /// rects), so this handles `Open`/`Close`.
    fn start_anim(&mut self, id: WindowId, cat: Category, destroyed: bool) -> bool {
        self.ensure_identity(id); // heal a startup-adopted blank so class/title rules match
        let spec = self.config.spec_for(&self.window_match(id), cat);
        let dur = spec.duration.unwrap_or(self.config.anim.duration);
        match cat {
            Category::Open => {
                // Clean slate so blocks absent from the spec leave their property
                // at rest (a window re-mapped after fading/sliding out).
                self.windows.reset_transforms(id);
                let o = self.read_opacity(id);
                let has_opacity = spec.blocks.iter().any(|b| matches!(b, Primitive::Opacity { .. }));
                // Open-time motion primitives (all but the fade + close-only drivers), and
                // whether ANY survive the backend's caps. If the spec wanted motion but the
                // backend gated it all away, fall back to a fade so the window animates in
                // rather than popping straight to full opacity.
                let is_open_motion = |b: &Primitive| {
                    matches!(b, Primitive::Scale { .. } | Primitive::Translate { .. }
                        | Primitive::Wobble { .. } | Primitive::Spin { .. }
                        | Primitive::Wave { .. } | Primitive::Ripple { .. })
                };
                let wants_motion = spec.blocks.iter().any(&is_open_motion);
                let has_renderable_motion =
                    spec.blocks.iter().any(|b| is_open_motion(b) && caps_allow(self.caps, b));
                if has_opacity || (wants_motion && !has_renderable_motion) {
                    self.windows.fade_in(id, o, dur); // 0 -> target; clears closing
                } else {
                    // No fade block -> appear at full opacity, not mid-close (motion carries it).
                    self.windows.set_opacity_settled(id, o);
                    self.windows.clear_closing(id);
                }
                for block in &spec.blocks {
                    if !caps_allow(self.caps, block) {
                        continue; // backend can't render this primitive; the fade above covers it
                    }
                    match block {
                        Primitive::Scale { from, axis, easing } => {
                            let from = from.unwrap_or(self.config.anim.scale_from);
                            self.windows.scale_in(id, from, dur, map_axis(*axis), map_easing(*easing));
                        }
                        Primitive::Translate { dx, dy, edge, easing } => {
                            let off = resolve_offset(*dx, *dy, *edge, self.outer_rect_of(id), self.screen());
                            self.windows.translate_in(id, off, dur, map_easing(*easing));
                        }
                        Primitive::Wobble { spring, friction } => {
                            // "boing": spawn compressed, spring out to the full rect.
                            let rect = self.outer_rect_of(id);
                            self.windows.wobble_to(
                                id,
                                squash_rect(rect, 0.6),
                                rect,
                                spring.unwrap_or(self.config.anim.wobble_spring),
                                friction.unwrap_or(self.config.anim.wobble_friction),
                            );
                        }
                        Primitive::Spin { degrees, easing } => {
                            let rad = degrees.unwrap_or(SPIN_DEFAULT_DEG as f32) as f64 * PI / 180.0;
                            self.windows.spin_in(id, rad, dur, map_easing(*easing));
                        }
                        Primitive::Wave { amplitude, wavelength, speed, axis, duration } => {
                            self.windows.wave_to(
                                id,
                                amplitude.unwrap_or(self.config.anim.wave_amplitude),
                                wavelength.unwrap_or(self.config.anim.wave_wavelength),
                                speed.unwrap_or(self.config.anim.wave_speed),
                                map_axis(*axis),
                                duration.unwrap_or(self.config.anim.wave_duration),
                            );
                        }
                        Primitive::Ripple { amplitude, wavelength, speed, r0, duration } => {
                            self.windows.ripple_to(
                                id,
                                [0.5, 0.5],
                                amplitude.unwrap_or(self.config.anim.ripple_amplitude),
                                wavelength.unwrap_or(self.config.anim.ripple_wavelength),
                                speed.unwrap_or(self.config.anim.ripple_speed),
                                r0.unwrap_or(self.config.anim.ripple_r0),
                                duration.unwrap_or(self.config.anim.ripple_duration),
                            );
                        }
                        Primitive::Opacity { .. } | Primitive::Burn | Primitive::Drain { .. } => {}
                    }
                }
                true
            }
            Category::Close => {
                if spec.blocks.is_empty() {
                    return false; // "none" -> instant close, nothing to animate
                }
                // Gate close drivers on caps: a shaderless backend can't run burn/drain, so
                // treat them as absent → the fade/collapse driver below runs and still reaps
                // the window. (Merely dropping the WindowDraw field would hold it opaque for
                // the whole close, then pop — burn/drain OWN the reap, they aren't decoration.)
                let has_burn = self.caps.shaders && spec.blocks.iter().any(|b| matches!(b, Primitive::Burn));
                let has_drain =
                    self.caps.shaders && spec.blocks.iter().any(|b| matches!(b, Primitive::Drain { .. }));
                // A scale block targeting ~0 collapses the window to a line — that
                // drives it invisible on its own, so no opacity fade is needed.
                let collapses = spec.blocks.iter().any(|b| {
                    matches!(b, Primitive::Scale { from, .. }
                        if from.unwrap_or(self.config.anim.scale_from) <= 1e-3)
                });
                let mut started = false;
                for block in &spec.blocks {
                    if !caps_allow(self.caps, block) {
                        continue; // backend can't render this; fade/collapse (below) carries the close
                    }
                    match block {
                        Primitive::Scale { from, axis, easing } => {
                            let to = from.unwrap_or(self.config.anim.scale_from);
                            self.windows.retarget_scale(id, to, dur, map_axis(*axis), map_easing(*easing));
                        }
                        Primitive::Translate { dx, dy, edge, easing } => {
                            let off = resolve_offset(*dx, *dy, *edge, self.outer_rect_of(id), self.screen());
                            self.windows.translate_out(id, off, dur, map_easing(*easing));
                        }
                        Primitive::Burn => {
                            started |= self.windows.begin_burn(id, dur, burn_seed(id), destroyed);
                        }
                        Primitive::Drain { turns, duration, turbulence } => {
                            started |= self.windows.begin_drain(
                                id,
                                duration.map(f64::from).unwrap_or(f64::from(self.config.anim.drain_duration)),
                                turns.unwrap_or(self.config.anim.drain_turns),
                                turbulence.unwrap_or(self.config.anim.drain_turbulence),
                                burn_seed(id),
                                destroyed,
                            );
                        }
                        Primitive::Spin { degrees, easing } => {
                            let rad = degrees.unwrap_or(SPIN_DEFAULT_DEG as f32) as f64 * PI / 180.0;
                            self.windows.spin_out(id, rad, dur, map_easing(*easing));
                        }
                        Primitive::Wave { amplitude, wavelength, speed, axis, duration } => {
                            // Wave while the opacity fade (below) carries the window out —
                            // the per-pixel warp renders textured × the fading alpha, reaped at 0.
                            self.windows.wave_to(
                                id,
                                amplitude.unwrap_or(self.config.anim.wave_amplitude),
                                wavelength.unwrap_or(self.config.anim.wave_wavelength),
                                speed.unwrap_or(self.config.anim.wave_speed),
                                map_axis(*axis),
                                duration.unwrap_or(self.config.anim.wave_duration),
                            );
                        }
                        Primitive::Ripple { amplitude, wavelength, speed, r0, duration } => {
                            // Ripple while the opacity fade (below) carries the window out.
                            self.windows.ripple_to(
                                id,
                                [0.5, 0.5],
                                amplitude.unwrap_or(self.config.anim.ripple_amplitude),
                                wavelength.unwrap_or(self.config.anim.ripple_wavelength),
                                speed.unwrap_or(self.config.anim.ripple_speed),
                                r0.unwrap_or(self.config.anim.ripple_r0),
                                duration.unwrap_or(self.config.anim.ripple_duration),
                            );
                        }
                        // Wobble on close is ignored — a closing window uses the fade path.
                        Primitive::Opacity { .. } | Primitive::Wobble { .. } => {}
                    }
                }
                // Exactly one completion driver + the closing flag: burn dissolves and
                // drain spirals+shrinks away (both begun above, both self-complete at
                // progress 1 — no fade); a scale-to-0 collapses while staying opaque;
                // otherwise fade to transparent (carrying scale/translate ride-alongs).
                if !has_burn && !has_drain {
                    if collapses {
                        started |= self.windows.begin_collapse(id, destroyed);
                    } else {
                        started |= self.windows.begin_fade_out(id, dur, destroyed);
                    }
                }
                started
            }
            Category::Move => false,
        }
    }

    /// A window's outer rect `[x, y, w, h]` (border included) as floats — the
    /// anchor rect the wobble mesh (and its damage) is built over.
    fn outer_rect(&self, w: &Win) -> [f32; 4] {
        let bw = w.border_width as f32;
        [w.x as f32, w.y as f32, w.width as f32 + 2.0 * bw, w.height as f32 + 2.0 * bw]
    }

    /// Whether `w`'s outer rectangle (border included) covers the whole root.
    fn covers_screen(&self, w: &Win) -> bool {
        covers_root(
            w.x as i32, w.y as i32, w.width as i32, w.height as i32,
            w.border_width as i32, self.x.root_width as i32, self.x.root_height as i32,
        )
    }

    /// Whether `atom` is one of the window-identity properties we cache.
    fn is_identity_atom(&self, atom: u32) -> bool {
        ["WM_CLASS", "WM_NAME", "_NET_WM_WINDOW_TYPE", "_NET_WM_NAME"]
            .iter()
            .any(|n| self.x.atom(n).is_ok_and(|a| a == atom))
    }

    /// Arm the fade frame clock if it isn't already running: a `calloop` timer
    /// that advances every window's fade and recomposites each tick, dropping
    /// itself once all fades settle (so an idle screen has no timer wakeups).
    fn ensure_frame_timer(&mut self) {
        if self.frame_timer.is_some() {
            return;
        }
        let Some(handle) = self.loop_handle.clone() else {
            return;
        };
        self.last_frame = None;
        let token = handle.insert_source(Timer::immediate(), |_deadline, _meta, app: &mut App| {
            let now = Instant::now();
            let dt = app.last_frame.map_or(0.0, |t| now.duration_since(t).as_secs_f64());
            app.last_frame = Some(now);
            let animating = app.windows.advance_anims(dt);
            let osd = app.advance_osd(dt);
            let hud_moving = app.advance_hud_move(dt);
            if app.reap_finished_fadeouts() {
                app.update_redirection(); // a reaped window can change the top window
            }
            tracing::trace!(dt, animating, "anim tick");
            // Damage only the animating windows' moving paths (this frame's extent
            // ∪ last frame's), so animations ride use-damage instead of forcing a
            // full-screen repaint every tick. `prev ∪ curr` covers the retreating
            // side of a move and clears windows that just settled or were reaped.
            let sr = app.config.shadow.radius as i32;
            let cur: Vec<(WindowId, Rect)> = app
                .windows
                .iter_bottom_to_top()
                .filter(|w| {
                    w.fade.is_animating()
                        || w.dim.is_animating()
                        || w.scale.is_animating()
                        || w.translate.is_animating()
                        || w.spin.is_animating()
                        || w.wobble.is_some()
                        || w.wave.is_some()
                        || w.ripple.is_some()
                        || w.burn.as_ref().is_some_and(|b| b.progress.is_animating())
                        || w.drain.as_ref().is_some_and(|d| d.progress.is_animating())
                })
                .map(|w| {
                    let bw = w.border_width as i32;
                    let (x, y) = (w.x as i32, w.y as i32);
                    let (ow, oh) = (w.width as i32 + 2 * bw, w.height as i32 + 2 * bw);
                    // Un-scaled outer rect grown by the shadow reach — the scale pop
                    // only shrinks inward, so this always covers it — then unioned
                    // with the translate offset and the padded wobble bbox.
                    let mut r = Rect::new(x - sr, y - sr, x + ow + sr, y + oh + sr);
                    // Slide/drop: union the rest rect with the shifted rect so the
                    // vacated strip repaints (with prev∪curr this covers the trail).
                    let [tx, ty] = w.translate.current();
                    if tx != 0.0 || ty != 0.0 {
                        let (tx, ty) = (tx.round() as i32, ty.round() as i32);
                        r = Rect::new(
                            r.x1.min(r.x1 + tx),
                            r.y1.min(r.y1 + ty),
                            r.x2.max(r.x2 + tx),
                            r.y2.max(r.y2 + ty),
                        );
                    }
                    if let Some(wob) = &w.wobble {
                        let b = wob.bounds(WOBBLE_PAD);
                        let wr = Rect::new(
                            b[0].floor() as i32, b[1].floor() as i32,
                            b[2].ceil() as i32, b[3].ceil() as i32,
                        );
                        r = Rect::new(r.x1.min(wr.x1), r.y1.min(wr.y1), r.x2.max(wr.x2), r.y2.max(wr.y2));
                    }
                    // Wave/ripple are per-pixel (content warps within the fixed rect),
                    // so the outer rect above already covers them — no extra bbox.
                    // Spin: the rotated bounding box exceeds the rect at ~45°.
                    if w.spin.current() != 0.0 {
                        let sa = rotated_aabb(
                            [x as f32, y as f32, ow as f32, oh as f32],
                            w.spin.current() as f32,
                        );
                        r = Rect::new(r.x1.min(sa.x1), r.y1.min(sa.y1), r.x2.max(sa.x2), r.y2.max(sa.y2));
                    }
                    (w.id, r)
                })
                .collect();
            let had_prev = !app.anim_damage.is_empty();
            for (_, r) in &cur {
                app.frame_damage.add_rect(*r);
            }
            for r in app.anim_damage.values() {
                app.frame_damage.add_rect(*r);
            }
            let has_cur = !cur.is_empty();
            app.anim_damage = cur.into_iter().collect();
            if has_cur || had_prev || osd == OsdTick::Moving || hud_moving {
                app.dirty = true; // repaint happens once, in the run callback
            }
            let moving = animating || osd == OsdTick::Moving || hud_moving;
            if moving {
                // While compositing, eglSwapBuffers(vsync) paces us to the refresh
                // rate, so an immediate re-arm self-throttles; when unredirected
                // there is no swap to block on, so step to avoid busy-looping.
                let step = if app.redirected { Duration::ZERO } else { Duration::from_millis(16) };
                TimeoutAction::ToDuration(step)
            } else if osd == OsdTick::Holding {
                // OSD sitting at full presence: keep the clock alive to count the
                // hold down, but don't repaint (the banner is static).
                TimeoutAction::ToDuration(Duration::from_millis(50))
            } else {
                app.frame_timer = None;
                TimeoutAction::Drop
            }
        });
        match token {
            Ok(tok) => self.frame_timer = Some(tok),
            Err(e) => tracing::error!("insert frame timer: {e}"),
        }
    }

    /// Advance an in-flight HUD auto-hop; returns `true` if one was active this tick
    /// (so the frame clock repaints). On arrival, commit the destination corner.
    fn advance_hud_move(&mut self, dt: f64) -> bool {
        let Some(m) = self.hud_move.as_mut() else { return false };
        let still = m.advance(dt);
        let to = m.to;
        if !still {
            self.hud_corner = to;
            self.hud_move = None;
        }
        true
    }

    /// Kick off an auto-hop to a random *different* corner. No-op while the HUD is
    /// hidden (nothing to move).
    fn trigger_automove(&mut self) {
        if !self.show_fps {
            return;
        }
        let avoid: Vec<HudCorner> =
            self.config.fps.auto_move_avoid.iter().filter_map(|s| parse_corner(s)).collect();
        let Some(to) = random_corner(self.hud_corner, &avoid, &mut self.rng) else {
            return; // every corner is off-limits — leave the HUD where it is
        };
        let dur = self.config.fps.auto_move_duration.max(0.05);
        self.hud_move = Some(HudMove { from: self.hud_corner, to, t: 0.0, dur });
        self.ensure_frame_timer();
        self.dirty = true;
        tracing::debug!(?to, "fps HUD auto-hop");
    }

    /// Arm or disarm the periodic auto-hop timer to match `on` (session-only; a
    /// reload re-reads `[fps] auto_move`). Idempotent: removes any existing timer first.
    fn set_automove(&mut self, on: bool) {
        self.config.fps.auto_move = on;
        if let Some(tok) = self.automove_timer.take()
            && let Some(h) = self.loop_handle.clone()
        {
            h.remove(tok);
        }
        if !on {
            return;
        }
        let Some(handle) = self.loop_handle.clone() else { return };
        let interval = Duration::from_secs_f64(self.config.fps.auto_move_interval.max(1.0));
        let tok = handle.insert_source(
            Timer::from_duration(interval),
            |_deadline, _meta, app: &mut App| {
                app.trigger_automove();
                TimeoutAction::ToDuration(Duration::from_secs_f64(
                    app.config.fps.auto_move_interval.max(1.0),
                ))
            },
        );
        match tok {
            Ok(t) => self.automove_timer = Some(t),
            Err(e) => tracing::error!("insert automove timer: {e}"),
        }
    }

    /// Advance the OSD toast one frame; returns what it did this tick. `In`/`Out`
    /// are moving (need a repaint); `Hold` just counts down (no repaint).
    fn advance_osd(&mut self, dt: f64) -> OsdTick {
        let out_dur = self.config.osd.out_dur;
        let Some(osd) = self.osd.as_mut() else {
            return OsdTick::Idle;
        };
        match osd.phase {
            OsdPhase::In => {
                if !osd.presence.advance(dt) {
                    osd.phase = OsdPhase::Hold;
                }
                OsdTick::Moving
            }
            OsdPhase::Hold => {
                osd.hold_remaining -= dt;
                if osd.hold_remaining <= 0.0 {
                    osd.presence.retarget(0.0, out_dur);
                    osd.phase = OsdPhase::Out;
                    OsdTick::Moving
                } else {
                    OsdTick::Holding
                }
            }
            OsdPhase::Out => {
                if osd.presence.advance(dt) {
                    OsdTick::Moving
                } else {
                    self.osd = None;
                    OsdTick::Idle
                }
            }
        }
    }

    /// Name a pixmap + create a damage object for a (now-mapped) window.
    fn acquire_gfx(&mut self, win: WindowId) {
        let pixmap = self.x.name_window_pixmap(win).map_err(|e| tracing::warn!("name pixmap {win}: {e}")).ok();
        let damage = self.x.create_damage(win).map_err(|e| tracing::warn!("create damage {win}: {e}")).ok();
        tracing::debug!(window = win, ?pixmap, ?damage, "acquire gfx");
        if let Some(old) = self.gfx.insert(win, WinGfx { pixmap, damage }) {
            self.free_gfx(old);
        }
    }

    fn release_gfx(&mut self, win: WindowId) {
        self.identities.remove(&win);
        if let Some(g) = self.gfx.remove(&win) {
            tracing::debug!(window = win, "release gfx");
            self.free_gfx(g);
        }
    }

    /// Reap windows whose fade-out has completed: free their resources, and drop
    /// destroyed ones from the stack (keep merely-unmapped ones, cleared). Returns
    /// whether any were reaped, so the caller can re-check the redirect decision.
    fn reap_finished_fadeouts(&mut self) -> bool {
        let finished = self.windows.finished_fadeouts();
        if finished.is_empty() {
            return false;
        }
        for (id, destroyed) in finished {
            tracing::debug!(window = id, destroyed, "fade-out complete");
            self.release_gfx(id);
            if destroyed {
                self.windows.remove(id);
            } else {
                self.windows.clear_closing(id);
            }
        }
        true
    }

    fn free_gfx(&self, g: WinGfx) {
        if let Some(p) = g.pixmap {
            let _ = self.x.free_pixmap(p);
        }
        if let Some(d) = g.damage {
            let _ = self.x.destroy_damage(d);
        }
    }

    /// Re-name the pixmap after a resize (the old one is stale), keeping damage.
    fn rebind_pixmap(&mut self, win: WindowId) {
        tracing::debug!(window = win, "rebind pixmap (resize)");
        let fresh = self.x.name_window_pixmap(win).ok();
        let old = match self.gfx.get_mut(&win) {
            Some(g) => std::mem::replace(&mut g.pixmap, fresh),
            None => fresh, // not tracked: free the freshly-named pixmap below
        };
        if let Some(p) = old {
            let _ = self.x.free_pixmap(p);
        }
    }

    /// Mark the whole screen for repaint next composite (any structural change).
    fn damage_full(&mut self) {
        self.force_full = true;
        self.dirty = true;
    }

    /// Mark a window's on-screen rect for repaint; falls back to full if unknown.
    fn damage_window(&mut self, win: WindowId) {
        match self.windows.get(win) {
            Some(w) => {
                let bw = w.border_width as i32;
                self.frame_damage.add_rect(Rect::from_xywh(
                    w.x as i32,
                    w.y as i32,
                    w.width as i32 + 2 * bw,
                    w.height as i32 + 2 * bw,
                ));
                self.dirty = true;
            }
            None => self.damage_full(),
        }
    }

    /// Composite the visible window stack (bottom-to-top) onto the overlay —
    /// mapped windows plus any fading out.
    fn composite(&mut self) {
        // Nothing to paint while unredirected: the overlay is unmapped and the
        // fullscreen window draws straight to the screen.
        if !self.redirected {
            return;
        }
        // Compute the HUD load block up front (it mutates the load ring) so it
        // doesn't clash with the backend borrow below.
        let hud_load = if self.show_fps {
            let a = self.load.averages(Instant::now());
            Some(HudLoad {
                fps: [a[0].fps, a[1].fps, a[2].fps],
                render_ms: [a[0].render_ms, a[1].render_ms, a[2].render_ms],
            })
        } else {
            None
        };
        let Some(backend) = self.backend.as_ref() else {
            return;
        };
        // Each entry is a quad, an optional wobble mesh (`Some` while the window is
        // wobbling → the backend draws the deformed grid instead), an optional burn,
        // and an always-on-top flag (from the `above` rule) used to reorder below.
        let mut items: Vec<CompositeItem> = Vec::new();
        for w in self.windows.visible_bottom_to_top() {
            if w.id == self.overlay {
                continue;
            }
            if let Some(pm) = self.gfx.get(&w.id).and_then(|g| g.pixmap) {
                let bw = w.border_width as i32;
                let (ow, oh) = (w.width as i32 + 2 * bw, w.height as i32 + 2 * bw);
                // Per-window rule overrides, falling back to the global config.
                let rr = self.resolve_rules(w.id);
                // "wobbling" = the deformed-mesh path (spring wobble). Skips scale /
                // shadow / blur / corners and draws the textured grid.
                let wobbling = w.wobble.is_some();
                // Per-pixel refraction (RIPPLE_FS / WAVE_FS): draws the quad but, like
                // spin, skips shadow / frost / corner rounding. Not a mesh.
                let per_pixel = w.ripple.is_some() || w.wave.is_some() || w.drain.is_some();
                let burning = w.burn.is_some();
                let burn = w.burn.as_ref().map(|b| Burn { progress: b.progress.current() as f32, seed: b.seed });
                // Scale-about-centre for the open/close pop. Skipped while wobbling —
                // the mesh path positions the window from the spring sim instead.
                let s = w.scale.current();
                // Per-axis scale factors: `Both` = uniform pop; `X`/`Y` = directional
                // stretch (one dimension animates, the other stays full).
                let (fx, fy) = match w.scale_axis {
                    wm::anim::Axis::Both => (s, s),
                    wm::anim::Axis::X => (s, 1.0),
                    wm::anim::Axis::Y => (1.0, s),
                };
                let scaling = !wobbling && !burning && (s - 1.0).abs() > f64::EPSILON;
                // A directional stretch skips corner rounding + shadow while active (a
                // rounded/shadowed 1-px sliver looks wrong); they return once settled.
                let directional = scaling && !matches!(w.scale_axis, wm::anim::Axis::Both);
                // Spin (rotate-about-centre): active whenever the angle isn't upright.
                // Drawn via the spin GL program; like a directional stretch it takes
                // no shadow/frost/corner rounding while turning.
                let spin = (w.spin.current() != 0.0).then_some(w.spin.current() as f32);
                // Animated translate (slide/drop) offset, added to the on-screen
                // position — CPU-side, so the blit needs no shader change.
                let off = w.translate.current();
                let (tx, ty) = (off[0].round() as i32, off[1].round() as i32);
                let (qx, qy, qw, qh) = if scaling {
                    let (cx, cy) = (w.x as f64 + ow as f64 / 2.0, w.y as f64 + oh as f64 / 2.0);
                    let (sw, sh) = (ow as f64 * fx, oh as f64 * fy);
                    (
                        (cx - sw / 2.0).round() as i32 + tx,
                        (cy - sh / 2.0).round() as i32 + ty,
                        (sw.round() as i32).max(1), // keep a ≥1px seed line visible
                        (sh.round() as i32).max(1),
                    )
                } else {
                    (w.x as i32 + tx, w.y as i32 + ty, ow, oh)
                };
                // A wobbling window draws as a bare textured mesh: no shadow, frost,
                // or corner rounding (square while it jiggles; they return on settle).
                // A translate offset shifts the mesh vertices too (both are screen px).
                let mesh = w.wobble.as_ref().map(|wob| wob.vertices()).map(|mut v| {
                    if off != [0.0, 0.0] {
                        for p in &mut v {
                            p[0] += off[0];
                            p[1] += off[1];
                        }
                    }
                    v
                });
                // Ripple params for the shader: centre + current amp/wavelength/phase/r0.
                let ripple = w.ripple.as_ref().map(|r| {
                    let (center, amp, wavelength, phase, r0) = r.params();
                    RippleParams { center, amp, wavelength, phase, r0 }
                });
                // Wave params for the shader: amp/wavelength/phase + travel axis (0=X, 1=Y).
                let wave = w.wave.as_ref().map(|wv| {
                    let (amp, wavelength, phase, axis) = wv.params();
                    WaveParams { amp, wavelength, phase, axis: matches!(axis, wm::anim::Axis::Y) as u32 }
                });
                // Drain params for the shader: centre + progress (0→1) + swirl turns.
                let drain = w.drain.as_ref().map(|d| DrainParams {
                    center: [0.5, 0.5],
                    progress: d.progress.current() as f32, // close ramp, or an animate hold
                    turns: d.turns,
                    turbulence: d.turbulence,
                    seed: d.seed,
                });
                // Caps backstop (defense-in-depth): never hand an effect to a backend that
                // can't render it. Arming is already caps-gated in start_anim/apply_effect;
                // this guarantees safety if any arming path is missed or a new primitive is
                // added without its gate. Inert on a full-caps backend (GL).
                let mesh = if self.caps.mesh { mesh } else { None };
                let (burn, spin, ripple, wave, drain) = if self.caps.shaders {
                    (burn, spin, ripple, wave, drain)
                } else {
                    (None, None, None, None, None)
                };
                items.push((
                    Quad {
                        pixmap: pm,
                        x: qx,
                        y: qy,
                        w: qw,
                        h: qh,
                        // Opacity animates via the fade (target folds explicit /
                        // rule / default opacity — see read_opacity), times the
                        // inactive-dim factor (1.0 unless unfocused with [dim] on).
                        opacity: (w.fade.current() * w.dim.current()) as f32,
                        // Drop the shadow the instant a window starts closing (so it
                        // disappears on close rather than lingering through the fade)
                        // or while it wobbles. Size test uses the un-scaled rect.
                        shadow: self.caps.shadow
                            && !wobbling
                            && !directional
                            && spin.is_none()
                            && !per_pixel
                            && rr.shadow.unwrap_or(self.config.shadow.enabled)
                            && ow >= self.config.shadow.min_size
                            && oh >= self.config.shadow.min_size
                            && !w.closing,
                        // Frost the backdrop only for translucent windows (opaque
                        // ones hide their backdrop); never while wobbling.
                        blur: self.caps.blur
                            && !wobbling
                            && !per_pixel
                            && rr.blur.unwrap_or(self.config.blur.enabled)
                            && w.fade.current() < 1.0,
                        corner_radius: if !self.caps.rounded_corners
                            || wobbling
                            || directional
                            || spin.is_some()
                            || per_pixel
                        {
                            0.0
                        } else {
                            rr.corner_radius.unwrap_or(self.config.corner_radius)
                        },
                    },
                    mesh,
                    burn,
                    rr.above.unwrap_or(false),
                    spin,
                    ripple,
                    wave,
                    drain,
                ));
            }
        }
        // Always-on-top: stable-sort so `above` windows move to the end of the
        // bottom-to-top list (i.e. topmost), keeping relative order within each
        // group. The occlusion walk below then treats them as the top of the stack
        // — they occlude what's beneath and are never occluded by normal windows,
        // regardless of the X stacking order.
        items.sort_by_key(|it| it.3);
        // Region-level occlusion: walk top-to-bottom, accumulating the region
        // covered by opaque windows above. Each window is drawn only in its
        // visible part (footprint ∩ screen − covered); an empty visible region
        // means fully occluded, so it's dropped from the draw list entirely.
        let (sw, sh) = (self.x.root_width as i32, self.x.root_height as i32);
        let screen = Rect::from_xywh(0, 0, sw, sh);
        // Paint region: only the damaged area (buffer-age partial repaint), unless a
        // structural change / HUD / disabled damage / unusable buffer age forces full.
        let age = backend.buffer_age();
        let own_full =
            self.force_full || !self.config.use_damage || self.show_fps || self.osd.is_some();
        let paint = if own_full || age <= 0 || age as usize > self.damage_history.len() + 1 {
            Region::from_rect(screen)
        } else {
            let mut p = self.frame_damage.clone();
            for h in self.damage_history.iter().take(age as usize - 1) {
                p.union(h);
            }
            p.intersect_rect(&screen);
            p
        };
        let sr = self.config.shadow.radius as i32;
        let mut covered = Region::new();
        let mut draws: Vec<WindowDraw> = Vec::with_capacity(items.len());
        for (q, mesh, burn, _above, spin, ripple, wave, drain) in items.iter().rev() {
            let rect = Rect::from_xywh(q.x, q.y, q.w, q.h);
            // Footprint = the area the window might touch this frame. A spinner
            // sweeps its rotated bounding box; a wobbler can deform outside its rect
            // (padded mesh bbox); otherwise the rect, grown by the shadow reach when
            // it casts one. Clamped to screen below.
            let footprint = match (spin, mesh) {
                (Some(a), _) => rotated_aabb([q.x as f32, q.y as f32, q.w as f32, q.h as f32], *a),
                (None, Some(v)) => mesh_bbox(v, WOBBLE_PAD),
                (None, None) if q.shadow => {
                    Rect::new(q.x - sr, q.y - sr, q.x + q.w + sr, q.y + q.h + sr)
                }
                (None, None) => rect,
            };
            let mut visible = Region::from_rect(footprint);
            visible.intersect_rect(&screen);
            visible.subtract(&covered);
            visible.intersect(&paint); // repaint only the damaged part
            if !visible.is_empty() {
                draws.push(WindowDraw {
                    quad: *q,
                    clip: visible.rects().to_vec(),
                    mesh: mesh.clone(),
                    burn: *burn,
                    spin: *spin,
                    ripple: *ripple,
                    wave: *wave,
                    drain: *drain,
                });
            }
            // Opaque windows occlude what's below: a square one covers its whole
            // rect; a rounded one covers all but its (transparent) corner squares.
            // A wobbling window is *deforming*, so it never occludes (its rect is
            // unreliable) — draw everything beneath it.
            if mesh.is_none() && burn.is_none() && spin.is_none() && ripple.is_none() && wave.is_none() && drain.is_none() && q.opacity >= 1.0 {
                let cr = q.corner_radius as i32;
                if cr <= 0 {
                    covered.add_rect(rect);
                } else {
                    let (x0, y0, x1, y1) = (q.x, q.y, q.x + q.w, q.y + q.h);
                    let mut occ = Region::from_rect(rect);
                    occ.subtract_rect(&Rect::new(x0, y0, x0 + cr, y0 + cr));
                    occ.subtract_rect(&Rect::new(x1 - cr, y0, x1, y0 + cr));
                    occ.subtract_rect(&Rect::new(x0, y1 - cr, x0 + cr, y1));
                    occ.subtract_rect(&Rect::new(x1 - cr, y1 - cr, x1, y1));
                    covered.union(&occ);
                }
            }
        }
        draws.reverse(); // restore bottom-to-top for correct layering
        tracing::trace!(windows = items.len(), drawn = draws.len(), "composite");
        let hud = if self.show_fps {
            // While auto-hopping, fade out at the current corner then fade in at the
            // destination — an in-place hide/show, no slide; static otherwise.
            let (corner, opacity) = match &self.hud_move {
                Some(m) => hop_view(m.from, m.to, m.t),
                None => (self.hud_corner, 1.0),
            };
            Some(Hud {
                fps: self.fps_meter.fps(),
                graph: self.config.fps.graph,
                corner,
                scale: self.config.fps.scale,
                refresh_hz: self.refresh_hz as f32,
                load: hud_load,
                outline: self.config.fps.outline,
                opacity,
            })
        } else {
            None
        };
        let osd = self.osd.as_ref().map(|o| Osd {
            text: o.text.clone(),
            presence: o.presence.current() as f32,
            scale: self.config.osd.scale,
            effect: osd_effect(if matches!(o.phase, OsdPhase::Out) { o.close } else { o.open }),
            color: o.color,
            background: self.config.osd.background,
            outline: self.config.osd.outline,
        });
        tracing::debug!(paint_rects = paint.rects().len(), paint_px = paint.area(), age, "damage");
        if let Err(e) = backend.present_windows(
            &draws,
            sw,
            sh,
            hud.as_ref(),
            osd.as_ref(),
            paint.rects(),
        ) {
            tracing::error!("composite failed: {e}");
        }
        // Sample the present cadence (post-swap = vsync-paced). Damage-driven, so
        // this only advances while the screen is actually repainting.
        let now = Instant::now();
        let render_ms = self.backend.as_ref().map_or(0.0, |b| b.render_ms());
        self.fps_meter.tick(now);
        self.load.record(now, render_ms);
        tracing::debug!(
            fps = self.fps_meter.fps(),
            ms = self.fps_meter.last_ms(),
            samples = self.fps_meter.samples().len(),
            "fps: frame"
        );
        // Record this frame's own damage for future frames' buffer-age; a forced
        // full repaint counts as a whole-screen change.
        let own = if own_full {
            Region::from_rect(screen)
        } else {
            self.frame_damage.clone()
        };
        self.damage_history.push_front(own);
        self.damage_history.truncate(MAX_BUFFER_AGE);
        self.frame_damage.clear();
        self.force_full = false;
    }

    /// unredir-if-possible: should the screen be unredirected? True when the
    /// topmost mapped window covers the whole screen (a fullscreen app such as
    /// mpv) — it can then page-flip directly and bypass the compositor. If the
    /// topmost window is small (e.g. a corner overlay) this is false, so we keep
    /// compositing — which is exactly the case that would otherwise tear.
    fn should_unredirect(&self) -> bool {
        // Config can disable unredir entirely: always composite, even a lone
        // fullscreen window (never step aside to let it page-flip).
        if !self.config.unredir {
            return false;
        }
        let Some(top) = self
            .windows
            .mapped_bottom_to_top()
            .filter(|w| w.id != self.overlay)
            .last()
        else {
            return false;
        };
        // A per-window rule can force this window to stay composited (no bypass).
        if self.resolve_rules(top.id).unredir == Some(false) {
            return false;
        }
        self.covers_screen(top)
    }

    /// Re-evaluate the redirect decision and transition if it changed.
    fn update_redirection(&mut self) {
        let want = self.should_unredirect();
        tracing::debug!(want_unredirect = want, redirected = self.redirected, "redir check");
        match (want, self.redirected) {
            (true, true) => self.redir_stop(),
            (false, false) => self.redir_start(),
            _ => {}
        }
    }

    /// Enter compositing: map the overlay, redirect the screen, (re)bind pixmaps.
    fn redir_start(&mut self) {
        if self.redirected {
            return;
        }
        let _ = self.x.map_window(self.overlay);
        if let Err(e) = self.x.redirect_subwindows() {
            tracing::error!("redirect_subwindows: {e}");
            let _ = self.x.unmap_window(self.overlay);
            return;
        }
        self.redirected = true;
        let mapped: Vec<WindowId> = self
            .windows
            .mapped_bottom_to_top()
            .filter(|w| w.id != self.overlay)
            .map(|w| w.id)
            .collect();
        for id in mapped {
            self.acquire_gfx(id);
        }
        let _ = self.x.flush();
        tracing::info!("redirected — compositing");
        // Paint immediately: otherwise the overlay we just mapped sits unpainted
        // over the (previously bypassing) fullscreen window for a frame — a flash.
        self.force_full = true;
        self.composite();
    }

    /// Leave compositing: free pixmaps, unredirect the screen, and unmap the
    /// overlay so the fullscreen window draws straight to the display.
    fn redir_stop(&mut self) {
        if !self.redirected {
            return;
        }
        let ids: Vec<WindowId> = self.gfx.keys().copied().collect();
        for id in ids {
            self.release_gfx(id);
        }
        if let Err(e) = self.x.unredirect_subwindows() {
            tracing::error!("unredirect_subwindows: {e}");
        }
        let _ = self.x.unmap_window(self.overlay);
        let _ = self.x.flush();
        self.redirected = false;
        tracing::info!("unredirected — fullscreen bypass");
    }

    /// Drain all pending X events and flush our replies/requests.
    ///
    /// x11rb's `flush()` itself does a non-blocking socket read and enqueues any
    /// newly-arrived events into its *internal* queue. So a naive
    /// "drain-until-empty, then flush once" strands the events that flush just
    /// read: the OS socket is now empty, calloop's fd watch never signals again,
    /// and those events (e.g. a DamageNotify) sit unprocessed until unrelated
    /// socket traffic happens to wake us. That is the video-freeze bug — a
    /// DRI3/GL client produces little fresh socket traffic after redirect, so
    /// the loop stalls. Loop flush+drain until a whole pass yields nothing, so
    /// no event is ever left buffered and every queued request is sent.
    fn drain_x_events(&mut self) {
        loop {
            let _ = self.x.flush();
            let mut progressed = false;
            loop {
                match self.x.conn.poll_for_event() {
                    Ok(Some(ev)) => {
                        self.handle_event(ev);
                        progressed = true;
                    }
                    Ok(None) => break,
                    Err(e) => {
                        tracing::error!("X connection error: {e}");
                        return;
                    }
                }
            }
            if !progressed {
                break;
            }
        }
    }

    fn handle_event(&mut self, ev: Event) {
        match ev {
            Event::CreateNotify(e) if e.window != self.overlay => {
                tracing::debug!(window = e.window, x = e.x, y = e.y, w = e.width, h = e.height, "create");
                self.windows.add_top(Win::new(
                    e.window, e.x, e.y, e.width, e.height, e.border_width, e.override_redirect, false,
                ));
                let _ = self.x.select_window_events(e.window);
                // Not visible yet — record its opacity; the fade-in starts on map.
                let o = self.read_opacity(e.window);
                self.windows.set_opacity_settled(e.window, o);
            }
            Event::DestroyNotify(e) => {
                tracing::debug!(window = e.window, "destroy");
                self.windows.set_mapped(e.window, false);
                // A CompositeNameWindowPixmap pixmap outlives the window, so we can
                // keep compositing the last frame and animate it out; the window is
                // reaped from the stack once the close completes. Nothing to animate
                // (no gfx, already invisible, or close="none") -> drop now.
                let has_gfx = self.gfx.contains_key(&e.window);
                if has_gfx && self.start_anim(e.window, Category::Close, true) {
                    self.ensure_frame_timer();
                } else {
                    self.windows.remove(e.window);
                    self.release_gfx(e.window);
                }
                self.update_redirection();
                self.damage_full();
            }
            Event::MapNotify(e) if e.window != self.overlay => {
                tracing::debug!(window = e.window, "map");
                self.windows.set_mapped(e.window, true);
                self.refresh_identity(e.window); // identity first — the open spec may be rule-gated
                // Start the open animation *before* (re)painting, so if this map
                // triggers an unredir->redirect transition (redir_start paints
                // immediately), that first frame already shows the resolved start
                // state (e.g. opacity 0 / scaled-down) — no full-size/opacity flash.
                self.start_anim(e.window, Category::Open, false);
                self.update_redirection();
                // Always re-acquire on (re)map: a window that unmapped/closed kept its
                // old named pixmap for the fade/burn, but that pixmap is now stale — a
                // remap (or an X id reused before the closing window is reaped) must get
                // a fresh one, else create_image fails every frame and it never draws.
                if self.redirected {
                    self.acquire_gfx(e.window);
                }
                self.ensure_frame_timer();
                self.damage_full();
            }
            Event::UnmapNotify(e) => {
                tracing::debug!(window = e.window, "unmap");
                self.windows.set_mapped(e.window, false);
                // Animate the last frame out if we have it (keep the pixmap); else
                // drop now. Unmapped (not destroyed): keep the window in the stack.
                let has_gfx = self.gfx.contains_key(&e.window);
                if has_gfx && self.start_anim(e.window, Category::Close, false) {
                    self.ensure_frame_timer();
                } else {
                    self.release_gfx(e.window);
                }
                self.update_redirection();
                self.damage_full();
            }
            Event::ConfigureNotify(e) => {
                let above = (e.above_sibling != 0).then_some(e.above_sibling);
                let resized = self
                    .windows
                    .get(e.window)
                    .is_some_and(|w| w.width != e.width || w.height != e.height);
                tracing::debug!(
                    window = e.window, x = e.x, y = e.y, w = e.width, h = e.height,
                    above = e.above_sibling, resized, "configure"
                );
                // Capture the OLD outer rect before `configure` overwrites it — the
                // wobble mesh lags from here toward the new geometry. Only a mapped,
                // non-closing window wobbles (skips create-time placement + fade-outs).
                let old_wobble = self.windows.get(e.window).map(|w| (self.outer_rect(w), w.is_mapped() && !w.closing));
                self.windows
                    .configure(e.window, e.x, e.y, e.width, e.height, e.border_width, above);
                // Move/resize wobble: if the resolved `move` spec has a wobble block,
                // aim the spring mesh at the new outer rect. `mv` is Some only for a
                // mapped, non-closing window that actually moved; the whole thing is
                // gated on compositing being on.
                let wobble = self
                    .config
                    .spec_for(&self.window_match(e.window), Category::Move)
                    .blocks
                    .iter()
                    .find_map(|b| match b {
                        Primitive::Wobble { spring, friction } => Some((
                            spring.unwrap_or(self.config.anim.wobble_spring),
                            friction.unwrap_or(self.config.anim.wobble_friction),
                        )),
                        _ => None,
                    });
                let mv = old_wobble.filter(|&(_, animatable)| animatable).and_then(|(old_rect, _)| {
                    let new_rect = self.windows.get(e.window).map(|w| self.outer_rect(w))?;
                    (new_rect != old_rect).then_some((old_rect, new_rect))
                });
                if self.redirected
                    && self.caps.mesh // a mesh-less backend (XRender) can't wobble; skip (window just moves)
                    && let Some((spring, friction)) = wobble
                    && let Some((old_rect, new_rect)) = mv
                {
                    self.windows.wobble_to(e.window, old_rect, new_rect, spring, friction);
                    self.ensure_frame_timer();
                }
                // A resize can cross the fullscreen threshold (or a size-sensitive
                // match), changing the effective opacity — re-target if it moved.
                if resized {
                    let cur = self
                        .windows
                        .get(e.window)
                        .filter(|w| w.is_mapped() && !w.closing)
                        .map(|w| w.fade.target());
                    if let Some(cur_t) = cur {
                        let o = self.read_opacity(e.window);
                        if (o - cur_t).abs() > f64::EPSILON {
                            let d = self.anim_duration();
                            self.windows.retarget_opacity(e.window, o, d);
                            self.ensure_frame_timer();
                        }
                    }
                }
                // Restack or resize can change which window is topmost/fullscreen.
                self.update_redirection();
                if self.redirected && resized && self.gfx.contains_key(&e.window) {
                    self.rebind_pixmap(e.window);
                }
                self.damage_full();
            }
            Event::ReparentNotify(e) => {
                if e.parent != self.x.root {
                    tracing::debug!(window = e.window, parent = e.parent, "reparent (off-root)");
                    self.windows.remove(e.window);
                    self.release_gfx(e.window);
                    self.update_redirection();
                    self.damage_full();
                }
            }
            Event::CirculateNotify(e) => {
                let on_top = e.place == Place::ON_TOP;
                tracing::debug!(window = e.window, on_top, "circulate");
                if on_top {
                    self.windows.raise(e.window);
                } else {
                    self.windows.lower(e.window);
                }
                self.update_redirection();
                self.damage_full();
            }
            Event::PropertyNotify(e) if e.window == self.x.root => {
                // Root _NET_ACTIVE_WINDOW change = focus moved → re-apply inactive-dim
                // (only for the `ewmh` focus source; `x11` uses FocusChange instead).
                if self.config.dim.focus == FocusSource::Ewmh
                    && self.x.atom("_NET_ACTIVE_WINDOW").is_ok_and(|a| a == e.atom)
                {
                    let new = self.x.get_active_window().ok().flatten();
                    self.set_active_window(new);
                }
            }
            Event::PropertyNotify(e) => {
                // Opacity, or an identity property (WM_CLASS/type/title) whose change
                // could alter which rules match — re-read identity, re-target opacity,
                // and repaint (blur/shadow/corner re-resolve on the next composite).
                let opacity_atom = self.x.atom("_NET_WM_WINDOW_OPACITY").is_ok_and(|a| a == e.atom);
                let identity_atom = self.is_identity_atom(e.atom);
                if identity_atom {
                    self.refresh_identity(e.window);
                }
                if opacity_atom || identity_atom {
                    let o = self.read_opacity(e.window);
                    let d = self.anim_duration();
                    self.windows.retarget_opacity(e.window, o, d);
                    self.ensure_frame_timer();
                    self.damage_full();
                }
            }
            Event::FocusIn(e) | Event::FocusOut(e) if self.config.dim.focus == FocusSource::X11 => {
                // `x11` dim focus source: X input focus moved → re-apply dim. Skip
                // pointer-crossing + grab churn; query the authoritative focus rather
                // than trust the event's window (avoids FocusIn detail ambiguity).
                let real = (e.mode == NotifyMode::NORMAL || e.mode == NotifyMode::WHILE_GRABBED)
                    && e.detail != NotifyDetail::POINTER
                    && e.detail != NotifyDetail::POINTER_ROOT;
                if real {
                    let new = self.focused_top_level();
                    self.set_active_window(new);
                }
            }
            Event::DamageNotify(e) => {
                tracing::trace!(damage = e.damage, "damage");
                let _ = self.x.subtract_damage(e.damage);
                self.damage_window(e.drawable);
            }
            Event::RandrScreenChangeNotify(e) => {
                // Screen resized (e.g. xrandr). The composite overlay + EGL surface
                // track the root automatically; we just refresh the cached size so the
                // next composite sets the GL viewport + u_screen to the new dimensions.
                if e.width != self.x.root_width || e.height != self.x.root_height {
                    tracing::info!(
                        old_w = self.x.root_width, old_h = self.x.root_height,
                        new_w = e.width, new_h = e.height,
                        "root screen size changed"
                    );
                    self.x.root_width = e.width;
                    self.x.root_height = e.height;
                }
                // The mode (and thus refresh rate) may have changed — re-read it for
                // the HUD graph budget, even when the resolution is unchanged.
                self.refresh_hz = self.x.refresh_hz().unwrap_or(self.refresh_hz);
                // A new resolution changes the fullscreen threshold — re-decide.
                self.update_redirection();
                self.damage_full();
            }
            Event::KeyPress(e) => {
                // Ignore CapsLock (Lock) / NumLock (Mod2) so binds match regardless
                // of lock state — we grab every lock variant.
                let state = u16::from(e.state) & !(0x02 | 0x10);
                if self.fps_key == Some((e.detail, state)) {
                    self.show_fps = !self.show_fps;
                    tracing::info!(show_fps = self.show_fps, "fps: HUD toggled");
                    self.damage_full();
                } else if let Some(&(_, _, dir)) =
                    self.move_keys.iter().find(|&&(kc, m, _)| kc == e.detail && m == state)
                {
                    self.hud_corner = move_corner(self.hud_corner, dir);
                    tracing::info!(corner = ?self.hud_corner, "fps: HUD moved");
                    self.damage_full();
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests;
