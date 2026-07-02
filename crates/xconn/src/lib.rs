//! Thin x11rb wrapper: connection, extension negotiation, atom cache, and a few
//! helpers. Mirrors the role of picom's `src/x.c` + `src/atom.c`.

use std::cell::RefCell;
use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use x11rb::connection::{Connection, RequestConnection};
use x11rb::protocol::xproto::{
    Atom, AtomEnum, ChangeWindowAttributesAux, ConnectionExt as _, CreateWindowAux, EventMask,
    GrabMode, MapState, ModMask, Window, WindowClass,
};
use x11rb::rust_connection::RustConnection;

/// Owns the X connection and screen/root state.
pub struct XConn {
    pub conn: RustConnection,
    pub screen_num: usize,
    pub root: Window,
    pub root_width: u16,
    pub root_height: u16,
    atoms: RefCell<HashMap<String, Atom>>,
}

/// A snapshot of a top-level window, for logging / layout.
#[derive(Debug, Clone, Copy)]
pub struct WinInfo {
    pub window: Window,
    pub x: i16,
    pub y: i16,
    pub width: u16,
    pub height: u16,
    pub border_width: u16,
    pub mapped: bool,
}

impl XConn {
    /// Connect to the X server named by `$DISPLAY` (pure-Rust `RustConnection`).
    pub fn connect() -> Result<Self> {
        let (conn, screen_num) = x11rb::connect(None).context("connecting to the X server")?;
        let screen = &conn.setup().roots[screen_num];
        let root = screen.root;
        let root_width = screen.width_in_pixels;
        let root_height = screen.height_in_pixels;
        tracing::info!(root, screen_num, root_width, root_height, "connected to X");
        Ok(XConn {
            conn,
            screen_num,
            root,
            root_width,
            root_height,
            atoms: RefCell::new(HashMap::new()),
        })
    }

    pub fn flush(&self) -> Result<()> {
        self.conn.flush().context("flushing X requests")?;
        Ok(())
    }

    /// Intern an atom, caching the result.
    pub fn atom(&self, name: &str) -> Result<Atom> {
        if let Some(a) = self.atoms.borrow().get(name) {
            return Ok(*a);
        }
        let atom = self
            .conn
            .intern_atom(false, name.as_bytes())
            .with_context(|| format!("intern_atom({name})"))?
            .reply()
            .with_context(|| format!("intern_atom({name}) reply"))?
            .atom;
        self.atoms.borrow_mut().insert(name.to_owned(), atom);
        Ok(atom)
    }

    /// Verify every X extension we depend on is present, and negotiate versions
    /// (which also activates them server-side). Bails loudly if any is missing.
    pub fn setup_extensions(&self) -> Result<()> {
        use x11rb::protocol::{composite, damage, present, randr, render, shape, sync, xfixes};

        for (label, name) in [
            ("Composite", composite::X11_EXTENSION_NAME),
            ("DAMAGE", damage::X11_EXTENSION_NAME),
            ("RENDER", render::X11_EXTENSION_NAME),
            ("Present", present::X11_EXTENSION_NAME),
            ("RANDR", randr::X11_EXTENSION_NAME),
            ("SHAPE", shape::X11_EXTENSION_NAME),
            ("SYNC", sync::X11_EXTENSION_NAME),
            ("XFIXES", xfixes::X11_EXTENSION_NAME),
        ] {
            if self
                .conn
                .extension_information(name)
                .with_context(|| format!("querying extension {label}"))?
                .is_none()
            {
                bail!("required X extension not available: {label}");
            }
        }

        {
            use composite::ConnectionExt as _;
            let v = self.conn.composite_query_version(0, 4)?.reply()?;
            tracing::info!("Composite {}.{}", v.major_version, v.minor_version);
        }
        {
            use damage::ConnectionExt as _;
            let v = self.conn.damage_query_version(1, 1)?.reply()?;
            tracing::info!("DAMAGE {}.{}", v.major_version, v.minor_version);
        }
        {
            use xfixes::ConnectionExt as _;
            let v = self.conn.xfixes_query_version(5, 0)?.reply()?;
            tracing::info!("XFIXES {}.{}", v.major_version, v.minor_version);
        }
        {
            use render::ConnectionExt as _;
            let v = self.conn.render_query_version(0, 11)?.reply()?;
            tracing::info!("RENDER {}.{}", v.major_version, v.minor_version);
        }
        {
            use present::ConnectionExt as _;
            let v = self.conn.present_query_version(1, 2)?.reply()?;
            tracing::info!("Present {}.{}", v.major_version, v.minor_version);
        }
        {
            use randr::ConnectionExt as _;
            let v = self.conn.randr_query_version(1, 5)?.reply()?;
            tracing::info!("RANDR {}.{}", v.major_version, v.minor_version);
        }
        {
            use shape::ConnectionExt as _;
            let v = self.conn.shape_query_version()?.reply()?;
            tracing::info!("SHAPE {}.{}", v.major_version, v.minor_version);
        }
        {
            use sync::ConnectionExt as _;
            let v = self.conn.sync_initialize(3, 1)?.reply()?;
            tracing::info!("SYNC {}.{}", v.major_version, v.minor_version);
        }
        Ok(())
    }

    /// Snapshot the direct children of the root window (P0 logging / layout seed).
    pub fn list_tree(&self) -> Result<Vec<WinInfo>> {
        let children = self.conn.query_tree(self.root)?.reply()?.children;
        let mut out = Vec::with_capacity(children.len());
        for w in children {
            let attr = match self.conn.get_window_attributes(w)?.reply() {
                Ok(a) => a,
                Err(_) => continue, // window may have vanished between calls
            };
            let geo = match self.conn.get_geometry(w)?.reply() {
                Ok(g) => g,
                Err(_) => continue,
            };
            out.push(WinInfo {
                window: w,
                x: geo.x,
                y: geo.y,
                width: geo.width,
                height: geo.height,
                border_width: geo.border_width,
                mapped: attr.map_state == MapState::VIEWABLE,
            });
        }
        Ok(out)
    }

    /// Become the compositing manager by owning `_NET_WM_CM_S<screen>`.
    pub fn become_cm(&self) -> Result<Window> {
        let owner = self.conn.generate_id().context("generate_id for CM owner")?;
        self.conn
            .create_window(
                x11rb::COPY_DEPTH_FROM_PARENT,
                owner,
                self.root,
                0,
                0,
                1,
                1,
                0,
                WindowClass::INPUT_OUTPUT,
                x11rb::COPY_FROM_PARENT,
                &CreateWindowAux::new(),
            )
            .context("create CM owner window")?;
        let sel = self.atom(&format!("_NET_WM_CM_S{}", self.screen_num))?;
        self.conn
            .set_selection_owner(owner, sel, x11rb::CURRENT_TIME)
            .context("set_selection_owner")?;
        let cur = self.conn.get_selection_owner(sel)?.reply()?.owner;
        if cur != owner {
            bail!(
                "could not acquire _NET_WM_CM_S{} (another compositor running?)",
                self.screen_num
            );
        }
        tracing::info!(owner, "became compositing manager (_NET_WM_CM_S{})", self.screen_num);
        Ok(owner)
    }

    /// Subscribe to substructure changes on the root window (map/unmap/configure/...).
    pub fn select_root_substructure(&self) -> Result<()> {
        self.conn
            .change_window_attributes(
                self.root,
                &ChangeWindowAttributesAux::new().event_mask(EventMask::SUBSTRUCTURE_NOTIFY),
            )
            .context("select SubstructureNotify on root")?;
        Ok(())
    }

    /// Resolve a keysym to the first keycode that yields it, via the server's
    /// keyboard mapping. `None` if no key on the keyboard produces that keysym.
    pub fn keysym_to_keycode(&self, keysym: u32) -> Result<Option<u8>> {
        let setup = self.conn.setup();
        let (min, max) = (setup.min_keycode, setup.max_keycode);
        let reply = self
            .conn
            .get_keyboard_mapping(min, max - min + 1)
            .context("get_keyboard_mapping")?
            .reply()
            .context("get_keyboard_mapping reply")?;
        let per = (reply.keysyms_per_keycode as usize).max(1);
        for (i, chunk) in reply.keysyms.chunks(per).enumerate() {
            if chunk.contains(&keysym) {
                return Ok(Some(min + i as u8));
            }
        }
        Ok(None)
    }

    /// Passively grab `keycode`+`mods` on the root so the combo reaches us
    /// regardless of which window has focus (a global hotkey). Async grab modes so
    /// we never freeze the keyboard/pointer.
    pub fn grab_key(&self, keycode: u8, mods: u16) -> Result<()> {
        // `.check()` forces a round-trip so a BadAccess (combo already held by
        // another client) surfaces here instead of as a stray async X error.
        self.conn
            .grab_key(false, self.root, ModMask::from(mods), keycode, GrabMode::ASYNC, GrabMode::ASYNC)
            .context("grab_key")?
            .check()
            .context("grab_key rejected (combo may be held by another client)")?;
        Ok(())
    }

    /// Release a passive key grab previously set with [`grab_key`](Self::grab_key).
    pub fn ungrab_key(&self, keycode: u8, mods: u16) -> Result<()> {
        self.conn
            .ungrab_key(keycode, self.root, ModMask::from(mods))
            .context("ungrab_key")?;
        Ok(())
    }

    /// Ask for RandR screen-change notifications on the root so we learn when the
    /// resolution changes (e.g. via `xrandr`). Delivered as `RandrScreenChangeNotify`,
    /// whose `width`/`height` carry the new screen size.
    pub fn select_screen_change(&self) -> Result<()> {
        use x11rb::protocol::randr::{ConnectionExt as _, NotifyMask};
        self.conn
            .randr_select_input(self.root, NotifyMask::SCREEN_CHANGE)
            .context("randr_select_input(SCREEN_CHANGE)")?;
        Ok(())
    }

    /// Current display refresh rate (Hz) from the first enabled CRTC's mode
    /// (`dot_clock / (htotal * vtotal)`). `None` if RandR reports nothing usable.
    pub fn refresh_hz(&self) -> Option<f64> {
        use x11rb::protocol::randr::ConnectionExt as _;
        let res = self.conn.randr_get_screen_resources_current(self.root).ok()?.reply().ok()?;
        for &crtc in &res.crtcs {
            let Ok(cookie) = self.conn.randr_get_crtc_info(crtc, res.config_timestamp) else {
                continue;
            };
            let Ok(info) = cookie.reply() else { continue };
            if info.mode == 0 {
                continue; // disabled CRTC
            }
            if let Some(m) = res.modes.iter().find(|m| m.id == info.mode) {
                let total = m.htotal as f64 * m.vtotal as f64;
                if total > 0.0 {
                    return Some(m.dot_clock as f64 / total);
                }
            }
        }
        None
    }

    /// MANUAL-redirect all top-levels. NOTE: after this the server stops drawing
    /// the redirected windows — the compositor must paint or the screen freezes.
    pub fn redirect_subwindows(&self) -> Result<()> {
        use x11rb::protocol::composite::{ConnectionExt as _, Redirect};
        self.conn
            .composite_redirect_subwindows(self.root, Redirect::MANUAL)
            .context("composite_redirect_subwindows")?;
        Ok(())
    }

    pub fn unredirect_subwindows(&self) -> Result<()> {
        use x11rb::protocol::composite::{ConnectionExt as _, Redirect};
        self.conn
            .composite_unredirect_subwindows(self.root, Redirect::MANUAL)
            .context("composite_unredirect_subwindows")?;
        Ok(())
    }

    /// Get (and map) the composite overlay window.
    pub fn get_overlay(&self) -> Result<Window> {
        use x11rb::protocol::composite::ConnectionExt as _;
        let w = self
            .conn
            .composite_get_overlay_window(self.root)?
            .reply()?
            .overlay_win;
        Ok(w)
    }

    pub fn release_overlay(&self) -> Result<()> {
        use x11rb::protocol::composite::ConnectionExt as _;
        self.conn
            .composite_release_overlay_window(self.root)
            .context("composite_release_overlay_window")?;
        Ok(())
    }

    /// Map / unmap a window — used to show/hide the composite overlay when
    /// toggling redirection for fullscreen bypass (picom's redir_start/stop).
    pub fn map_window(&self, win: Window) -> Result<()> {
        self.conn.map_window(win).context("map_window")?;
        Ok(())
    }

    pub fn unmap_window(&self, win: Window) -> Result<()> {
        self.conn.unmap_window(win).context("unmap_window")?;
        Ok(())
    }

    /// The X visual id a window was created with.
    pub fn window_visual(&self, win: Window) -> Result<u32> {
        let attr = self.conn.get_window_attributes(win)?.reply()?;
        Ok(attr.visual)
    }

    /// Make a window (the overlay) input-transparent — empty input shape — so
    /// pointer events pass through to the windows below.
    pub fn overlay_input_passthrough(&self, overlay: Window) -> Result<()> {
        use x11rb::protocol::shape;
        use x11rb::protocol::xfixes::ConnectionExt as _;
        let region = self.conn.generate_id().context("generate_id for input region")?;
        self.conn.xfixes_create_region(region, &[])?; // empty region
        self.conn
            .xfixes_set_window_shape_region(overlay, shape::SK::INPUT, 0, 0, region)?;
        self.conn.xfixes_destroy_region(region)?;
        Ok(())
    }

    /// Name a redirected window's current off-screen pixmap (for binding as a
    /// texture). The window must be redirected; the pixmap goes stale on
    /// unmap/resize and must be re-named.
    pub fn name_window_pixmap(&self, window: Window) -> Result<u32> {
        use x11rb::protocol::composite::ConnectionExt as _;
        let pixmap = self.conn.generate_id().context("generate_id for window pixmap")?;
        self.conn
            .composite_name_window_pixmap(window, pixmap)
            .context("composite_name_window_pixmap")?;
        Ok(pixmap)
    }

    pub fn free_pixmap(&self, pixmap: u32) -> Result<()> {
        self.conn.free_pixmap(pixmap).context("free_pixmap")?;
        Ok(())
    }

    /// Create a Damage object on a window (NON_EMPTY: one event per dirtying
    /// until subtracted) so we know when to recomposite it.
    pub fn create_damage(&self, window: Window) -> Result<u32> {
        use x11rb::protocol::damage::{ConnectionExt as _, ReportLevel};
        let id = self.conn.generate_id().context("generate_id for damage")?;
        self.conn
            .damage_create(id, window, ReportLevel::NON_EMPTY)
            .context("damage_create")?;
        Ok(id)
    }

    pub fn destroy_damage(&self, damage: u32) -> Result<()> {
        use x11rb::protocol::damage::ConnectionExt as _;
        self.conn.damage_destroy(damage).context("damage_destroy")?;
        Ok(())
    }

    /// Acknowledge a damage region (repair=None, parts=None) to re-arm the event.
    pub fn subtract_damage(&self, damage: u32) -> Result<()> {
        use x11rb::protocol::damage::ConnectionExt as _;
        self.conn
            .damage_subtract(damage, 0u32, 0u32)
            .context("damage_subtract")?;
        Ok(())
    }

    /// Read `_NET_WM_WINDOW_OPACITY` — a CARDINAL in `0..=0xFFFF_FFFF` where
    /// `0xFFFF_FFFF` is fully opaque — as a `0.0..=1.0` fraction. Returns `None`
    /// when the property is absent (the caller then treats the window as opaque).
    pub fn get_window_opacity(&self, win: Window) -> Result<Option<f64>> {
        let prop = self.atom("_NET_WM_WINDOW_OPACITY")?;
        let reply = self
            .conn
            .get_property(false, win, prop, AtomEnum::CARDINAL, 0, 1)
            .context("get_property(_NET_WM_WINDOW_OPACITY)")?
            .reply()
            .context("get_property(_NET_WM_WINDOW_OPACITY) reply")?;
        Ok(reply
            .value32()
            .and_then(|mut it| it.next())
            .map(|v| v as f64 / u32::MAX as f64))
    }

    /// Ask the server to deliver `PropertyNotify` for this window, so we learn
    /// when e.g. `_NET_WM_WINDOW_OPACITY` changes at runtime. Event masks are
    /// per-client, so this does not disturb the owning client's own selection.
    /// May fail (async `BadWindow`) if the window vanishes — callers tolerate it.
    pub fn select_property_changes(&self, win: Window) -> Result<()> {
        self.conn
            .change_window_attributes(
                win,
                &ChangeWindowAttributesAux::new().event_mask(EventMask::PROPERTY_CHANGE),
            )
            .context("select PropertyChange")?;
        Ok(())
    }
}
