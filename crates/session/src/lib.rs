//! The compositor session: owns the X connection, the tracked window stack, and
//! the GL backend, and runs the calloop event loop. Mirrors picom's `session_t`
//! + main loop (`src/picom.c`) and event handlers (`src/event.c`).
//!
//! Live compositor: become CM, redirect the screen, render into the composite
//! overlay via `backend-gl`, and recomposite the window stack whenever damage
//! (or a structural change) arrives. On exit the X server auto-releases our
//! resources (redirect, overlay, pixmaps, damage), restoring normal drawing.

use std::collections::HashMap;
use std::os::fd::AsFd;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use calloop::generic::Generic;
use calloop::timer::{TimeoutAction, Timer};
use calloop::{EventLoop, Interest, LoopHandle, Mode, PostAction, RegistrationToken};
use x11rb::connection::Connection;
use x11rb::protocol::Event;
use x11rb::protocol::xproto::{Place, Window};

use backend_gl::{GlBackend, Quad};
use wm::{Win, WindowId, WindowStack};
use xconn::XConn;

/// Fade duration for map-in and opacity changes (seconds). Config-driven later.
const FADE_DURATION: f64 = 0.2;

/// Skip drop shadows for windows smaller than this (px) — avoids specks under
/// tiny override-redirect helper windows (e.g. mpv's 1x1 input windows).
const SHADOW_MIN_SIZE: i32 = 24;

/// Per-window X resources used for compositing.
#[derive(Default)]
struct WinGfx {
    /// Named off-screen pixmap of the window's current contents (incl. border).
    pixmap: Option<u32>,
    /// Damage object signalling when the window needs recompositing.
    damage: Option<u32>,
}

/// Top-level compositor state.
pub struct App {
    pub x: XConn,
    windows: WindowStack,
    overlay: Window,
    backend: Option<GlBackend>,
    gfx: HashMap<WindowId, WinGfx>,
    dirty: bool,
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
}

impl App {
    /// Connect to X and negotiate the extensions we depend on.
    pub fn new() -> Result<Self> {
        let x = XConn::connect()?;
        x.setup_extensions()?;
        Ok(App {
            x,
            windows: WindowStack::new(),
            overlay: 0,
            backend: None,
            gfx: HashMap::new(),
            dirty: true,
            redirected: false,
            loop_handle: None,
            frame_timer: None,
            last_frame: None,
        })
    }

    /// Become the CM, redirect + acquire the overlay, build the GL backend, then
    /// run the compositing event loop until the process is killed.
    pub fn run(&mut self) -> Result<()> {
        self.x.become_cm()?;
        self.x.select_root_substructure()?;
        self.x.select_screen_change()?;

        self.overlay = self.x.get_overlay()?;
        self.x.overlay_input_passthrough(self.overlay)?;
        let visual = self.x.window_visual(self.overlay)?;
        self.x.redirect_subwindows()?;
        self.redirected = true;
        self.backend = Some(GlBackend::new(self.overlay, visual)?);

        // Seed the stack + per-window resources from the current tree.
        for w in self.x.list_tree()? {
            if w.window == self.overlay {
                continue;
            }
            self.windows.add_top(Win::new(
                w.window, w.x, w.y, w.width, w.height, w.border_width, false, w.mapped,
            ));
            let _ = self.x.select_property_changes(w.window);
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

    /// Read a window's `_NET_WM_WINDOW_OPACITY` as a `0.0..=1.0` fraction,
    /// defaulting to fully opaque when the property is absent or unreadable.
    fn read_opacity(&self, win: WindowId) -> f64 {
        match self.x.get_window_opacity(win) {
            Ok(o) => o.unwrap_or(1.0),
            Err(e) => {
                tracing::debug!(window = win, "opacity read failed: {e}");
                1.0
            }
        }
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
            let animating = app.windows.advance_fades(dt);
            if app.reap_finished_fadeouts() {
                app.update_redirection(); // a reaped window can change the top window
            }
            tracing::trace!(dt, animating, "fade tick");
            app.dirty = true; // repaint happens once, in the run callback
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

    /// Composite the visible window stack (bottom-to-top) onto the overlay —
    /// mapped windows plus any fading out.
    fn composite(&self) {
        // Nothing to paint while unredirected: the overlay is unmapped and the
        // fullscreen window draws straight to the screen.
        if !self.redirected {
            return;
        }
        let Some(backend) = self.backend.as_ref() else {
            return;
        };
        let mut items: Vec<Quad> = Vec::new();
        for w in self.windows.visible_bottom_to_top() {
            if w.id == self.overlay {
                continue;
            }
            if let Some(pm) = self.gfx.get(&w.id).and_then(|g| g.pixmap) {
                let bw = w.border_width as i32;
                let (qw, qh) = (w.width as i32 + 2 * bw, w.height as i32 + 2 * bw);
                items.push(Quad {
                    pixmap: pm,
                    x: w.x as i32,
                    y: w.y as i32,
                    w: qw,
                    h: qh,
                    opacity: w.fade.current() as f32,
                    // Drop the shadow the instant a window starts closing, so it
                    // disappears on close/hide rather than lingering through the fade-out.
                    shadow: qw >= SHADOW_MIN_SIZE && qh >= SHADOW_MIN_SIZE && !w.closing,
                });
            }
        }
        tracing::trace!(items = items.len(), "composite");
        if let Err(e) = backend.present_windows(&items, self.x.root_width as i32, self.x.root_height as i32) {
            tracing::error!("composite failed: {e}");
        }
    }

    /// unredir-if-possible: should the screen be unredirected? True when the
    /// topmost mapped window covers the whole screen (a fullscreen app such as
    /// mpv) — it can then page-flip directly and bypass the compositor. If the
    /// topmost window is small (e.g. a corner overlay) this is false, so we keep
    /// compositing — which is exactly the case that would otherwise tear.
    fn should_unredirect(&self) -> bool {
        let (rw, rh) = (self.x.root_width as i32, self.x.root_height as i32);
        let Some(top) = self
            .windows
            .mapped_bottom_to_top()
            .filter(|w| w.id != self.overlay)
            .last()
        else {
            return false;
        };
        let bw = top.border_width as i32;
        let (x, y) = (top.x as i32, top.y as i32);
        x <= 0
            && y <= 0
            && x + top.width as i32 + 2 * bw >= rw
            && y + top.height as i32 + 2 * bw >= rh
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
                // keep compositing the last frame and fade it out; the window is
                // reaped from the stack once transparent. Nothing visible -> drop now.
                if self.gfx.contains_key(&e.window)
                    && self.windows.begin_fade_out(e.window, FADE_DURATION, true)
                {
                    self.ensure_frame_timer();
                } else {
                    self.windows.remove(e.window);
                    self.release_gfx(e.window);
                }
                self.update_redirection();
                self.dirty = true;
            }
            Event::MapNotify(e) if e.window != self.overlay => {
                tracing::debug!(window = e.window, "map");
                self.windows.set_mapped(e.window, true);
                // Start the fade-in *before* (re)painting, so if this map triggers
                // an unredir->redirect transition (redir_start paints immediately),
                // that first frame already shows the window at 0 — no full-opacity flash.
                let o = self.read_opacity(e.window);
                self.windows.fade_in(e.window, o, FADE_DURATION);
                self.update_redirection();
                if self.redirected && !self.gfx.contains_key(&e.window) {
                    self.acquire_gfx(e.window);
                }
                self.ensure_frame_timer();
                self.dirty = true;
            }
            Event::UnmapNotify(e) => {
                tracing::debug!(window = e.window, "unmap");
                self.windows.set_mapped(e.window, false);
                // Fade the last frame out if we have it (keep the pixmap); else drop now.
                if self.gfx.contains_key(&e.window)
                    && self.windows.begin_fade_out(e.window, FADE_DURATION, false)
                {
                    self.ensure_frame_timer();
                } else {
                    self.release_gfx(e.window);
                }
                self.update_redirection();
                self.dirty = true;
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
                self.windows
                    .configure(e.window, e.x, e.y, e.width, e.height, e.border_width, above);
                // Restack or resize can change which window is topmost/fullscreen.
                self.update_redirection();
                if self.redirected && resized && self.gfx.contains_key(&e.window) {
                    self.rebind_pixmap(e.window);
                }
                self.dirty = true;
            }
            Event::ReparentNotify(e) => {
                if e.parent != self.x.root {
                    tracing::debug!(window = e.window, parent = e.parent, "reparent (off-root)");
                    self.windows.remove(e.window);
                    self.release_gfx(e.window);
                    self.update_redirection();
                    self.dirty = true;
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
                self.dirty = true;
            }
            Event::PropertyNotify(e) => {
                // Only _NET_WM_WINDOW_OPACITY concerns us; a Delete (property
                // removed) reads back as absent → refresh restores full opacity.
                if self.x.atom("_NET_WM_WINDOW_OPACITY").is_ok_and(|a| a == e.atom) {
                    let o = self.read_opacity(e.window);
                    tracing::debug!(window = e.window, opacity = o, "opacity property changed");
                    self.windows.retarget_opacity(e.window, o, FADE_DURATION);
                    self.ensure_frame_timer();
                    self.dirty = true;
                }
            }
            Event::DamageNotify(e) => {
                tracing::trace!(damage = e.damage, "damage");
                let _ = self.x.subtract_damage(e.damage);
                self.dirty = true;
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
                // A new resolution changes the fullscreen threshold — re-decide.
                self.update_redirection();
                self.dirty = true;
            }
            _ => {}
        }
    }
}
