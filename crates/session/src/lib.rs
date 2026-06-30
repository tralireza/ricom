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

use anyhow::{Context, Result};
use calloop::generic::Generic;
use calloop::{EventLoop, Interest, Mode, PostAction};
use x11rb::connection::Connection;
use x11rb::protocol::Event;
use x11rb::protocol::xproto::{Place, Window};

use backend_gl::GlBackend;
use wm::{Win, WindowId, WindowStack};
use xconn::XConn;

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
        })
    }

    /// Become the CM, redirect + acquire the overlay, build the GL backend, then
    /// run the compositing event loop until the process is killed.
    pub fn run(&mut self) -> Result<()> {
        self.x.become_cm()?;
        self.x.select_root_substructure()?;

        self.overlay = self.x.get_overlay()?;
        self.x.overlay_input_passthrough(self.overlay)?;
        let visual = self.x.window_visual(self.overlay)?;
        self.x.redirect_subwindows()?;
        self.backend = Some(GlBackend::new(self.overlay, visual)?);

        // Seed the stack + per-window resources from the current tree.
        for w in self.x.list_tree()? {
            if w.window == self.overlay {
                continue;
            }
            self.windows.add_top(Win::new(
                w.window, w.x, w.y, w.width, w.height, w.border_width, false, w.mapped,
            ));
            if w.mapped {
                self.acquire_gfx(w.window);
            }
        }
        self.x.flush()?;
        self.composite();
        tracing::info!(
            mapped = self.windows.mapped_count(),
            "ricom compositing (Ctrl-C to quit)"
        );

        let mut event_loop: EventLoop<App> = EventLoop::try_new().context("create event loop")?;
        let handle = event_loop.handle();
        let fd = self
            .x
            .conn
            .stream()
            .as_fd()
            .try_clone_to_owned()
            .context("clone X connection fd")?;
        handle
            .insert_source(Generic::new(fd, Interest::READ, Mode::Level), |_r, _fd, app: &mut App| {
                app.drain_x_events();
                if app.dirty {
                    app.composite();
                    app.dirty = false;
                }
                Ok(PostAction::Continue)
            })
            .map_err(|e| anyhow::anyhow!("insert X source: {e}"))?;

        event_loop.run(None, self, |_app| {}).context("event loop")?;
        Ok(())
    }

    /// Name a pixmap + create a damage object for a (now-mapped) window.
    fn acquire_gfx(&mut self, win: WindowId) {
        let pixmap = self.x.name_window_pixmap(win).map_err(|e| tracing::warn!("name pixmap {win}: {e}")).ok();
        let damage = self.x.create_damage(win).map_err(|e| tracing::warn!("create damage {win}: {e}")).ok();
        if let Some(old) = self.gfx.insert(win, WinGfx { pixmap, damage }) {
            self.free_gfx(old);
        }
    }

    fn release_gfx(&mut self, win: WindowId) {
        if let Some(g) = self.gfx.remove(&win) {
            self.free_gfx(g);
        }
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
        let fresh = self.x.name_window_pixmap(win).ok();
        let old = match self.gfx.get_mut(&win) {
            Some(g) => std::mem::replace(&mut g.pixmap, fresh),
            None => fresh, // not tracked: free the freshly-named pixmap below
        };
        if let Some(p) = old {
            let _ = self.x.free_pixmap(p);
        }
    }

    /// Composite the mapped window stack (bottom-to-top) onto the overlay.
    fn composite(&self) {
        let Some(backend) = self.backend.as_ref() else {
            return;
        };
        let mut items: Vec<(u32, i32, i32, i32, i32)> = Vec::new();
        for w in self.windows.mapped_bottom_to_top() {
            if w.id == self.overlay {
                continue;
            }
            if let Some(pm) = self.gfx.get(&w.id).and_then(|g| g.pixmap) {
                let bw = w.border_width as i32;
                items.push((
                    pm,
                    w.x as i32,
                    w.y as i32,
                    w.width as i32 + 2 * bw,
                    w.height as i32 + 2 * bw,
                ));
            }
        }
        if let Err(e) = backend.present_windows(&items, self.x.root_width as i32, self.x.root_height as i32) {
            tracing::error!("composite failed: {e}");
        }
    }

    fn drain_x_events(&mut self) {
        loop {
            match self.x.conn.poll_for_event() {
                Ok(Some(ev)) => self.handle_event(ev),
                Ok(None) => break,
                Err(e) => {
                    tracing::error!("X connection error: {e}");
                    break;
                }
            }
        }
        let _ = self.x.flush();
    }

    fn handle_event(&mut self, ev: Event) {
        match ev {
            Event::CreateNotify(e) if e.window != self.overlay => {
                self.windows.add_top(Win::new(
                    e.window, e.x, e.y, e.width, e.height, e.border_width, e.override_redirect, false,
                ));
            }
            Event::DestroyNotify(e) => {
                self.windows.remove(e.window);
                self.release_gfx(e.window);
                self.dirty = true;
            }
            Event::MapNotify(e) if e.window != self.overlay => {
                self.windows.set_mapped(e.window, true);
                self.acquire_gfx(e.window);
                self.dirty = true;
            }
            Event::UnmapNotify(e) => {
                self.windows.set_mapped(e.window, false);
                self.release_gfx(e.window);
                self.dirty = true;
            }
            Event::ConfigureNotify(e) => {
                let above = (e.above_sibling != 0).then_some(e.above_sibling);
                let resized = self
                    .windows
                    .get(e.window)
                    .is_some_and(|w| w.width != e.width || w.height != e.height);
                self.windows
                    .configure(e.window, e.x, e.y, e.width, e.height, e.border_width, above);
                if resized && self.gfx.contains_key(&e.window) {
                    self.rebind_pixmap(e.window);
                }
                self.dirty = true;
            }
            Event::ReparentNotify(e) => {
                if e.parent != self.x.root {
                    self.windows.remove(e.window);
                    self.release_gfx(e.window);
                    self.dirty = true;
                }
            }
            Event::CirculateNotify(e) => {
                if e.place == Place::ON_TOP {
                    self.windows.raise(e.window);
                } else {
                    self.windows.lower(e.window);
                }
                self.dirty = true;
            }
            Event::DamageNotify(e) => {
                let _ = self.x.subtract_damage(e.damage);
                self.dirty = true;
            }
            _ => {}
        }
    }
}
