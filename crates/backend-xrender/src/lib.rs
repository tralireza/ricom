//! XRender rendering backend — pure x11rb (no EGL / GL / Xlib).
//!
//! A shaderless compositing backend: it composites the window stack with server-side
//! `RENDER` `Composite(OVER)` — each window's named pixmap (in `Quad.pixmap`) is
//! wrapped in a source `Picture` and blended onto a target `Picture` created over the
//! composite-overlay window. Per-window opacity is a cached constant-alpha solid-fill
//! mask; occlusion + damage clipping is a rectangle clip on the target (the RENDER
//! equivalent of GL's `SCISSOR_TEST`).
//!
//! It advertises [`BackendCaps`] with everything false, so `session` caps-gates every
//! shader/mesh effect (spin/ripple/wave/burn/drain/wobble) to a fade — this backend
//! therefore only ever receives plain quads (enforced by `debug_assert!`s below), and
//! drop-shadow / rounded-corners / blur are forced off upstream too.
//!
//! Minimal-correct first cut: textured blit + per-window opacity + occlusion/damage
//! clip + inactive dim (free — dim is folded into `Quad.opacity` upstream). Deferred to
//! Phase 2: drop shadow, rounded corners, backdrop blur, HUD/OSD text, affine
//! spin/drain via `SetPictureTransform`, and a back-buffer for tear-free present (this
//! cut draws straight to the overlay — a single persistent buffer, so it may tear).

use std::cell::RefCell;
use std::collections::HashMap;

use anyhow::{anyhow, Context, Result};
use x11rb::connection::Connection as _;
use x11rb::protocol::render::{
    Color, ConnectionExt as _, CreatePictureAux, PictOp, PictType, Pictforminfo, Pictformat, Picture,
};
use x11rb::protocol::xproto::{ConnectionExt as _, Rectangle};
use x11rb::rust_connection::RustConnection;

pub use backend::*;

/// Convert a compositor `region::Rect` (inclusive-exclusive `x1,y1,x2,y2`) to an X
/// `Rectangle` (origin + size). Widths clamp at 0 so a degenerate rect is harmless.
fn to_rect(r: &region::Rect) -> Rectangle {
    Rectangle {
        x: r.x1 as i16,
        y: r.y1 as i16,
        width: (r.x2 - r.x1).max(0) as u16,
        height: (r.y2 - r.y1).max(0) as u16,
    }
}

/// Find a standard 8-8-8-8 DIRECT pict-format. `want_alpha` selects `a8r8g8b8`
/// (depth 32, alpha in the high byte) vs `x8r8g8b8` (depth 24, no alpha channel).
fn find_format(formats: &[Pictforminfo], depth: u8, want_alpha: bool) -> Option<Pictformat> {
    formats
        .iter()
        .find(|f| {
            f.type_ == PictType::DIRECT
                && f.depth == depth
                && f.direct.red_shift == 16
                && f.direct.red_mask == 0xff
                && f.direct.green_shift == 8
                && f.direct.green_mask == 0xff
                && f.direct.blue_shift == 0
                && f.direct.blue_mask == 0xff
                && if want_alpha {
                    f.direct.alpha_shift == 24 && f.direct.alpha_mask == 0xff
                } else {
                    f.direct.alpha_mask == 0
                }
        })
        .map(|f| f.id)
}

/// Offscreen back-buffer: the whole frame is composited here, then copied to the
/// overlay in one `Composite` — so the visible overlay only ever shows a *complete*
/// frame (no clear-then-blit tearing/flicker). Recreated on a screen-size change.
struct BackBuffer {
    pixmap: u32,
    pic: Picture,
    w: i32,
    h: i32,
}

/// The GL-less render backend. Owns its **own** `RustConnection`: X resource ids are
/// server-global, so the overlay window and the per-window named pixmaps created on
/// `session`'s connection are valid here, and the `make_backend(config, window, visual)`
/// factory signature stays unchanged (no threading a non-`Clone` connection through it).
pub struct XrenderBackend {
    conn: RustConnection,
    /// The composite-overlay window id (drawable for the back-buffer pixmap + target).
    overlay: u32,
    /// Picture over the overlay window — where the finished back-buffer is copied.
    target: Picture,
    /// Overlay depth, for creating the back-buffer pixmap.
    depth: u8,
    /// `x8r8g8b8` (depth-24, opaque) and `a8r8g8b8` (depth-32, per-pixel alpha).
    fmt_rgb24: Pictformat,
    fmt_argb32: Pictformat,
    render: RenderParams,
    /// Per-pixmap depth cache (pixmap id → depth) so the source-Picture format is
    /// picked without a `GetGeometry` round-trip every frame. Pixmap ids are transient
    /// (they change on resize), so this is bounded by clearing when it grows large.
    depth_cache: RefCell<HashMap<u32, u8>>,
    /// Constant-alpha solid-fill Pictures for per-window opacity, keyed by quantized
    /// alpha (0..=254; 255 uses no mask). picom's `alpha_pict[]` technique.
    alpha_cache: RefCell<HashMap<u8, Picture>>,
    /// Offscreen back-buffer, (re)built lazily to the screen size (kills tearing).
    back: RefCell<Option<BackBuffer>>,
}

impl XrenderBackend {
    /// Bring up the XRender backend on the overlay `window`. `visual_id` is unused in
    /// the first cut (formats are picked by depth); it is kept for signature parity with
    /// the GL backend and for future visual-exact format matching.
    pub fn new(window: u32, _visual_id: u32, render: RenderParams) -> Result<Self> {
        let (conn, _screen) = RustConnection::connect(None).context("xrender: connect to X")?;
        conn.render_query_version(0, 11)
            .context("xrender: RENDER query_version")?
            .reply()
            .context("xrender: RENDER extension unavailable")?;

        let formats = conn
            .render_query_pict_formats()
            .context("xrender: QueryPictFormats")?
            .reply()
            .context("xrender: QueryPictFormats reply")?;
        let fmt_rgb24 = find_format(&formats.formats, 24, false)
            .ok_or_else(|| anyhow!("xrender: no x8r8g8b8 (depth-24) pict format"))?;
        let fmt_argb32 = find_format(&formats.formats, 32, true)
            .ok_or_else(|| anyhow!("xrender: no a8r8g8b8 (depth-32) pict format"))?;

        // Target Picture over the overlay window, format by the overlay's depth.
        let odepth = conn
            .get_geometry(window)
            .context("xrender: overlay GetGeometry")?
            .reply()
            .context("xrender: overlay GetGeometry reply")?
            .depth;
        let tfmt = if odepth == 32 { fmt_argb32 } else { fmt_rgb24 };
        let target = conn.generate_id().context("xrender: generate target id")?;
        conn.render_create_picture(target, window, tfmt, &CreatePictureAux::new())
            .context("xrender: create target Picture")?;
        conn.flush().ok();

        tracing::info!(overlay = window, depth = odepth, "xrender backend up (pure x11rb, no EGL)");
        Ok(XrenderBackend {
            conn,
            overlay: window,
            target,
            depth: odepth,
            fmt_rgb24,
            fmt_argb32,
            render,
            depth_cache: RefCell::new(HashMap::new()),
            alpha_cache: RefCell::new(HashMap::new()),
            back: RefCell::new(None),
        })
    }

    /// Ensure the offscreen back-buffer pixmap exists and matches `w × h`, rebuilding it
    /// (freeing the old) on a resize. Returns the back-buffer `Picture`, or an error if
    /// allocation fails. Depth/format follow the overlay.
    fn ensure_back(&self, w: i32, h: i32) -> Result<Picture> {
        let mut back = self.back.borrow_mut();
        if back.as_ref().is_none_or(|b| b.w != w || b.h != h) {
            if let Some(old) = back.take() {
                let _ = self.conn.render_free_picture(old.pic);
                let _ = self.conn.free_pixmap(old.pixmap);
            }
            let bfmt = if self.depth == 32 { self.fmt_argb32 } else { self.fmt_rgb24 };
            let pixmap = self.conn.generate_id().context("xrender: gen back pixmap id")?;
            self.conn
                .create_pixmap(self.depth, pixmap, self.overlay, w.max(1) as u16, h.max(1) as u16)
                .context("xrender: create back pixmap")?;
            let pic = self.conn.generate_id().context("xrender: gen back pic id")?;
            self.conn
                .render_create_picture(pic, pixmap, bfmt, &CreatePictureAux::new())
                .context("xrender: create back Picture")?;
            *back = Some(BackBuffer { pixmap, pic, w, h });
        }
        Ok(back.as_ref().unwrap().pic)
    }

    /// Depth of a window's pixmap, cached by pixmap id (a `GetGeometry` round-trip on
    /// first sight / after a resize gives a new id). Falls back to 24 if the query fails
    /// (e.g. the pixmap already went stale — the frame's blit is skipped anyway).
    fn pixmap_depth(&self, pixmap: u32) -> u8 {
        if let Some(&d) = self.depth_cache.borrow().get(&pixmap) {
            return d;
        }
        let d = self
            .conn
            .get_geometry(pixmap)
            .ok()
            .and_then(|c| c.reply().ok())
            .map(|g| g.depth)
            .unwrap_or(24);
        let mut cache = self.depth_cache.borrow_mut();
        if cache.len() > 1024 {
            cache.clear(); // bound memory as transient pixmap ids churn
        }
        cache.insert(pixmap, d);
        d
    }

    /// A cached constant-alpha solid-fill Picture for per-window opacity, or
    /// `x11rb::NONE` (0 = "no mask") when fully opaque or on allocation failure (the
    /// window is then blit at full opacity). Alpha `a/255` scaled to the 16-bit channel.
    fn alpha_mask(&self, a: u8) -> Picture {
        if a == 255 {
            return x11rb::NONE;
        }
        if let Some(&p) = self.alpha_cache.borrow().get(&a) {
            return p;
        }
        let Ok(pid) = self.conn.generate_id() else { return x11rb::NONE };
        let color = Color { red: 0, green: 0, blue: 0, alpha: (a as u16) * 257 };
        if self.conn.render_create_solid_fill(pid, color).is_err() {
            return x11rb::NONE;
        }
        self.alpha_cache.borrow_mut().insert(a, pid);
        pid
    }
}

impl backend::Backend for XrenderBackend {
    fn present_windows(
        &self,
        items: &[WindowDraw],
        screen_w: i32,
        screen_h: i32,
        _hud: Option<&Hud>,
        _osd: Option<&Osd>,
        _clear: &[Rect], // whole frame is recomposited each present (buffer_age()==0)
    ) -> Result<()> {
        let bg = Color {
            red: (self.render.background[0].clamp(0.0, 1.0) * 65535.0) as u16,
            green: (self.render.background[1].clamp(0.0, 1.0) * 65535.0) as u16,
            blue: (self.render.background[2].clamp(0.0, 1.0) * 65535.0) as u16,
            alpha: 65535,
        };
        let full = Rectangle {
            x: 0,
            y: 0,
            width: screen_w.max(0) as u16,
            height: screen_h.max(0) as u16,
        };

        // Compose the ENTIRE frame into the offscreen back-buffer, then copy it to the
        // overlay in one op — so the visible overlay never shows a half-painted (cleared
        // but not-yet-blitted) frame. Drawing straight to the overlay tears badly.
        let back = self.ensure_back(screen_w, screen_h)?;
        // Fresh frame: clear the whole back-buffer to the background colour.
        self.conn.render_set_picture_clip_rectangles(back, 0, 0, &[full])?;
        self.conn.render_fill_rectangles(PictOp::SRC, back, bg, &[full])?;

        // Composite each window bottom-to-top (session already ordered `items`) onto the back-buffer.
        for it in items {
            let q = &it.quad;
            debug_assert!(
                it.mesh.is_none()
                    && it.burn.is_none()
                    && it.spin.is_none()
                    && it.ripple.is_none()
                    && it.wave.is_none()
                    && it.drain.is_none(),
                "xrender: caps-gating should have zeroed all shader/mesh effect fields"
            );
            if q.w <= 0 || q.h <= 0 || it.clip.is_empty() {
                continue;
            }
            let depth = self.pixmap_depth(q.pixmap);
            let fmt = if depth == 32 { self.fmt_argb32 } else { self.fmt_rgb24 };
            let Ok(src) = self.conn.generate_id() else { continue };
            if self.conn.render_create_picture(src, q.pixmap, fmt, &CreatePictureAux::new()).is_err() {
                continue;
            }
            // Clip this window's draws to its visible + damaged rects.
            let clip: Vec<Rectangle> = it.clip.iter().map(to_rect).collect();
            let _ = self.conn.render_set_picture_clip_rectangles(back, 0, 0, &clip);
            // Per-window opacity (fade × inactive-dim, already folded upstream) via a
            // cached constant-alpha mask; `NONE` = fully opaque.
            let a = (q.opacity.clamp(0.0, 1.0) * 255.0).round() as u8;
            let mask = self.alpha_mask(a);
            let _ = self.conn.render_composite(
                PictOp::OVER,
                src,
                mask,
                back,
                0,
                0,
                0,
                0,
                q.x as i16,
                q.y as i16,
                q.w as u16,
                q.h as u16,
            );
            let _ = self.conn.render_free_picture(src);
        }

        // Present: copy the finished back-buffer to the overlay in a single Composite
        // (SRC = overwrite the whole screen). One op ⇒ no clear-then-blit banding.
        self.conn.render_set_picture_clip_rectangles(self.target, 0, 0, &[full])?;
        self.conn.render_composite(
            PictOp::SRC,
            back,
            x11rb::NONE,
            self.target,
            0,
            0,
            0,
            0,
            0,
            0,
            screen_w.max(0) as u16,
            screen_h.max(0) as u16,
        )?;
        self.conn.flush().ok();
        Ok(())
    }

    fn set_render_params(&mut self, render: RenderParams) {
        self.render = render;
    }

    fn set_font(&mut self, _path: &str, _size: f32) {
        // On-screen text (HUD/OSD/notify) via RENDER glyphs is Phase 2; no font yet.
    }

    fn has_text(&self) -> bool {
        false
    }

    fn render_ms(&self) -> f32 {
        0.0 // no GPU timer over the wire; sentinel = unmeasured (HUD shows idle)
    }

    fn buffer_age(&self) -> i32 {
        0 // no buffer-age; session repaints full each frame
    }

    fn caps(&self) -> BackendCaps {
        // Shaderless, meshless, no blur/shadow/rounded-corners in the first cut → session
        // fade-falls-back every effect it can't render (see the module docs).
        BackendCaps {
            shaders: false,
            mesh: false,
            blur: false,
            shadow: false,
            rounded_corners: false,
        }
    }
}

impl Drop for XrenderBackend {
    fn drop(&mut self) {
        let _ = self.conn.render_free_picture(self.target);
        for &p in self.alpha_cache.borrow().values() {
            let _ = self.conn.render_free_picture(p);
        }
        if let Some(b) = self.back.borrow_mut().take() {
            let _ = self.conn.render_free_picture(b.pic);
            let _ = self.conn.free_pixmap(b.pixmap);
        }
        let _ = self.conn.flush();
    }
}
