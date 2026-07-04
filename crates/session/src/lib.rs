//! The compositor session: owns the X connection, the tracked window stack, and
//! the GL backend, and runs the calloop event loop. Mirrors picom's `session_t`
//! + main loop (`src/picom.c`) and event handlers (`src/event.c`).
//!
//! Live compositor: become CM, redirect the screen, render into the composite
//! overlay via `backend-gl`, and recomposite the window stack whenever damage
//! (or a structural change) arrives. On exit the X server auto-releases our
//! resources (redirect, overlay, pixmaps, damage), restoring normal drawing.

use std::collections::{HashMap, VecDeque};
use std::os::fd::AsFd;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use calloop::generic::Generic;
#[cfg(target_os = "linux")]
use calloop::signals::{Signal, Signals};
use calloop::timer::{TimeoutAction, Timer};
use calloop::{EventLoop, Interest, LoopHandle, Mode, PostAction, RegistrationToken};
use x11rb::connection::Connection;
use x11rb::protocol::Event;
use x11rb::protocol::xproto::{Place, Window};

use backend_gl::{Burn, GlBackend, Hud, HudCorner, HudLoad, Quad, RenderParams, WindowDraw};
use region::{Rect, Region};

/// Max frames of damage history kept for EGL buffer-age partial repaint.
const MAX_BUFFER_AGE: usize = 4;
/// Extra px around a wobble's mesh bbox when damaging/clipping it — headroom for
/// spring overshoot and the AA fringe, so no jiggling pixel is ever left stale.
const WOBBLE_PAD: f32 = 8.0;
use config::{Category, Config, Edge, Primitive, RuleResult, WindowMatch};
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

/// Map the config's corner string to the backend's [`HudCorner`] (defaults to
/// top-right for anything unrecognised).
fn hud_corner(s: &str) -> HudCorner {
    match s {
        "top-left" => HudCorner::TopLeft,
        "bottom-left" => HudCorner::BottomLeft,
        "bottom-right" => HudCorner::BottomRight,
        _ => HudCorner::TopRight,
    }
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

/// One compositable window for [`App::composite`]: quad + optional wobble mesh +
/// optional burn + always-on-top flag (from the `above` rule).
type CompositeItem = (Quad, Option<Vec<[f32; 4]>>, Option<Burn>, bool);

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

pub struct App {
    pub x: XConn,
    windows: WindowStack,
    overlay: Window,
    backend: Option<GlBackend>,
    gfx: HashMap<WindowId, WinGfx>,
    /// Cached per-window identity (WM_CLASS / type / title) for rule matching.
    identities: HashMap<WindowId, WinIdentity>,
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
    /// Cached display refresh rate (Hz) for the HUD graph's budget; refreshed on RandR.
    refresh_hz: f64,
    /// Rolling frame-rate meter, sampled each composite while redirected.
    fps_meter: FpsMeter,
    /// 1m/5m/15m compositor-load ring, fed one sample per composited frame; shown
    /// in the HUD and logged on `SIGUSR1`.
    load: LoadTracker,
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
            gfx: HashMap::new(),
            identities: HashMap::new(),
            dirty: true,
            frame_damage: Region::new(),
            force_full: true,
            anim_damage: HashMap::new(),
            damage_history: VecDeque::new(),
            redirected: false,
            loop_handle: None,
            frame_timer: None,
            last_frame: None,
            show_fps: config.fps.enabled,
            hud_corner: hud_corner(&config.fps.corner),
            config,
            config_path,
            fps_key: None,
            move_keys: Vec::new(),
            refresh_hz,
            fps_meter: FpsMeter::new(),
            load: LoadTracker::new(Instant::now()),
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
                self.config = cfg;
                log_config_warnings(&self.config);
                if let Some(b) = self.backend.as_mut() {
                    b.set_render_params(render_params(&self.config));
                }
                // A changed FPS hotkey: drop the old grabs and bind the new combo.
                if hotkey_changed {
                    self.grab_fps_hotkey();
                }
                // A changed config corner repositions the runtime HUD corner.
                if corner_changed {
                    self.hud_corner = hud_corner(&self.config.fps.corner);
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
        self.x.select_root_substructure()?;
        self.x.select_screen_change()?;
        self.grab_fps_hotkey();

        self.overlay = self.x.get_overlay()?;
        self.x.overlay_input_passthrough(self.overlay)?;
        let visual = self.x.window_visual(self.overlay)?;
        self.x.redirect_subwindows()?;
        self.redirected = true;
        self.backend = Some(GlBackend::new(self.overlay, visual, render_params(&self.config))?);

        // Seed the stack + per-window resources from the current tree.
        for w in self.x.list_tree()? {
            if w.window == self.overlay {
                continue;
            }
            self.windows.add_top(Win::new(
                w.window, w.x, w.y, w.width, w.height, w.border_width, false, w.mapped,
            ));
            let _ = self.x.select_property_changes(w.window);
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
            let signals =
                Signals::new(&[Signal::SIGHUP, Signal::SIGUSR1]).context("create signal source")?;
            handle
                .insert_source(signals, |event, _meta, app: &mut App| match event.signal() {
                    Signal::SIGHUP => app.reload_config(),
                    Signal::SIGUSR1 => app.log_load(),
                    _ => {}
                })
                .map_err(|e| anyhow::anyhow!("insert signal source: {e}"))?;
        }

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
        Ok(())
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
        let spec = self.config.spec_for(&self.window_match(id), cat);
        let dur = spec.duration.unwrap_or(self.config.anim.duration);
        match cat {
            Category::Open => {
                // Clean slate so blocks absent from the spec leave their property
                // at rest (a window re-mapped after fading/sliding out).
                self.windows.reset_transforms(id);
                let o = self.read_opacity(id);
                if spec.blocks.iter().any(|b| matches!(b, Primitive::Opacity { .. })) {
                    self.windows.fade_in(id, o, dur); // 0 -> target; clears closing
                } else {
                    // No fade block -> appear at full opacity, not mid-close.
                    self.windows.set_opacity_settled(id, o);
                    self.windows.clear_closing(id);
                }
                for block in &spec.blocks {
                    match block {
                        Primitive::Scale { from, .. } => {
                            self.windows.scale_in(id, from.unwrap_or(self.config.anim.scale_from), dur);
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
                        Primitive::Opacity { .. } | Primitive::Burn => {}
                    }
                }
                true
            }
            Category::Close => {
                if spec.blocks.is_empty() {
                    return false; // "none" -> instant close, nothing to animate
                }
                let has_burn = spec.blocks.iter().any(|b| matches!(b, Primitive::Burn));
                let mut started = false;
                for block in &spec.blocks {
                    match block {
                        Primitive::Scale { from, .. } => {
                            self.windows.retarget_scale(id, from.unwrap_or(self.config.anim.scale_from), dur);
                        }
                        Primitive::Translate { dx, dy, edge, easing } => {
                            let off = resolve_offset(*dx, *dy, *edge, self.outer_rect_of(id), self.screen());
                            self.windows.translate_out(id, off, dur, map_easing(*easing));
                        }
                        Primitive::Burn => {
                            started |= self.windows.begin_burn(id, dur, burn_seed(id), destroyed);
                        }
                        Primitive::Opacity { .. } | Primitive::Wobble { .. } => {}
                    }
                }
                // A non-burn close always fades to transparent — that's what drives
                // completion + reaping (see `finished_fadeouts`), so scale/translate
                // ride along with it. Burn owns alpha, so it fades nothing.
                if !has_burn {
                    started |= self.windows.begin_fade_out(id, dur, destroyed);
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
                        || w.scale.is_animating()
                        || w.translate.is_animating()
                        || w.wobble.is_some()
                        || w.burn.as_ref().is_some_and(|b| b.progress.is_animating())
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
            if has_cur || had_prev {
                app.dirty = true; // repaint happens once, in the run callback
            }
            if animating {
                // While compositing, eglSwapBuffers(vsync) paces us to the refresh
                // rate, so an immediate re-arm self-throttles; when unredirected
                // there is no swap to block on, so step to avoid busy-looping.
                let step = if app.redirected { Duration::ZERO } else { Duration::from_millis(16) };
                TimeoutAction::ToDuration(step)
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
                let wobbling = w.wobble.is_some();
                let burning = w.burn.is_some();
                let burn = w.burn.as_ref().map(|b| Burn { progress: b.progress.current() as f32, seed: b.seed });
                // Scale-about-centre for the open/close pop. Skipped while wobbling —
                // the mesh path positions the window from the spring sim instead.
                let s = w.scale.current();
                // Animated translate (slide/drop) offset, added to the on-screen
                // position — CPU-side, so the blit needs no shader change.
                let off = w.translate.current();
                let (tx, ty) = (off[0].round() as i32, off[1].round() as i32);
                let (qx, qy, qw, qh) = if !wobbling && !burning && (s - 1.0).abs() > f64::EPSILON {
                    let (cx, cy) = (w.x as f64 + ow as f64 / 2.0, w.y as f64 + oh as f64 / 2.0);
                    let (sw, sh) = (ow as f64 * s, oh as f64 * s);
                    (
                        (cx - sw / 2.0).round() as i32 + tx,
                        (cy - sh / 2.0).round() as i32 + ty,
                        sw.round() as i32,
                        sh.round() as i32,
                    )
                } else {
                    (w.x as i32 + tx, w.y as i32 + ty, ow, oh)
                };
                // A wobbling window draws as a bare textured mesh: no shadow, frost,
                // or corner rounding (square while it jiggles; they return on settle).
                // A translate offset shifts the mesh vertices too (both are screen px).
                let mesh = w.wobble.as_ref().map(|wob| {
                    let mut v = wob.vertices();
                    if off != [0.0, 0.0] {
                        for p in &mut v {
                            p[0] += off[0];
                            p[1] += off[1];
                        }
                    }
                    v
                });
                items.push((
                    Quad {
                        pixmap: pm,
                        x: qx,
                        y: qy,
                        w: qw,
                        h: qh,
                        // Opacity animates via the fade, whose target already folds
                        // explicit / rule / default opacity (see read_opacity).
                        opacity: w.fade.current() as f32,
                        // Drop the shadow the instant a window starts closing (so it
                        // disappears on close rather than lingering through the fade)
                        // or while it wobbles. Size test uses the un-scaled rect.
                        shadow: !wobbling
                            && rr.shadow.unwrap_or(self.config.shadow.enabled)
                            && ow >= self.config.shadow.min_size
                            && oh >= self.config.shadow.min_size
                            && !w.closing,
                        // Frost the backdrop only for translucent windows (opaque
                        // ones hide their backdrop); never while wobbling.
                        blur: !wobbling
                            && rr.blur.unwrap_or(self.config.blur.enabled)
                            && w.fade.current() < 1.0,
                        corner_radius: if wobbling {
                            0.0
                        } else {
                            rr.corner_radius.unwrap_or(self.config.corner_radius)
                        },
                    },
                    mesh,
                    burn,
                    rr.above.unwrap_or(false),
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
        let own_full = self.force_full || !self.config.use_damage || self.show_fps;
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
        for (q, mesh, burn, _above) in items.iter().rev() {
            let rect = Rect::from_xywh(q.x, q.y, q.w, q.h);
            // Footprint = the area the window might touch this frame. A wobbler can
            // deform outside its rect, so use its padded mesh bbox; otherwise the
            // rect, grown by the shadow reach when it casts one. Clamped to screen.
            let footprint = match mesh {
                Some(v) => mesh_bbox(v, WOBBLE_PAD),
                None if q.shadow => Rect::new(q.x - sr, q.y - sr, q.x + q.w + sr, q.y + q.h + sr),
                None => rect,
            };
            let mut visible = Region::from_rect(footprint);
            visible.intersect_rect(&screen);
            visible.subtract(&covered);
            visible.intersect(&paint); // repaint only the damaged part
            if !visible.is_empty() {
                draws.push(WindowDraw { quad: *q, clip: visible.rects().to_vec(), mesh: mesh.clone(), burn: *burn });
            }
            // Opaque windows occlude what's below: a square one covers its whole
            // rect; a rounded one covers all but its (transparent) corner squares.
            // A wobbling window is *deforming*, so it never occludes (its rect is
            // unreliable) — draw everything beneath it.
            if mesh.is_none() && burn.is_none() && q.opacity >= 1.0 {
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
            Some(Hud {
                fps: self.fps_meter.fps(),
                graph: self.config.fps.graph,
                corner: self.hud_corner,
                scale: self.config.fps.scale,
                refresh_hz: self.refresh_hz as f32,
                load: hud_load,
            })
        } else {
            None
        };
        tracing::debug!(paint_rects = paint.rects().len(), paint_px = paint.area(), age, "damage");
        if let Err(e) = backend.present_windows(
            &draws,
            sw,
            sh,
            hud.as_ref(),
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
                let _ = self.x.select_property_changes(e.window);
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
