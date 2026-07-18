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

/// Map a slice of compositor rects to X `Rectangle`s (see [`to_rect`]) — used for both
/// the background-clear rects and each window's clip.
fn rects(clip: &[region::Rect]) -> Vec<Rectangle> {
    clip.iter().map(to_rect).collect()
}

/// A window contributes nothing this frame — no area, or no visible/damaged clip rects
/// (a fully-occluded window has an empty clip). Skip it: no source Picture, no damage.
fn should_skip(it: &WindowDraw) -> bool {
    it.quad.w <= 0 || it.quad.h <= 0 || it.clip.is_empty()
}

/// Quantize a `0.0..=1.0` opacity (fade × inactive-dim, folded upstream) to 8-bit alpha
/// for the cached constant-alpha mask.
fn quantize_opacity(opacity: f32) -> u8 {
    (opacity.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Scale a `0.0..=1.0` colour channel to RENDER's 16-bit range.
fn scale_channel(c: f32) -> u16 {
    (c.clamp(0.0, 1.0) * 65535.0) as u16
}

/// Expand an 8-bit alpha to RENDER's 16-bit range (`a/255 · 65535`, i.e. `a · 257`).
fn alpha16(a: u8) -> u16 {
    a as u16 * 257
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
            // Clear the fresh buffer fully so undamaged areas aren't garbage before a
            // force-full frame paints them (with buffer_age()==1 we only repaint damage).
            let fr = Rectangle { x: 0, y: 0, width: w.max(1) as u16, height: h.max(1) as u16 };
            let _ = self.conn.render_set_picture_clip_rectangles(pic, 0, 0, &[fr]);
            let _ = self.conn.render_fill_rectangles(PictOp::SRC, pic, self.bg_color(), &[fr]);
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
        let color = Color { red: 0, green: 0, blue: 0, alpha: alpha16(a) };
        if self.conn.render_create_solid_fill(pid, color).is_err() {
            return x11rb::NONE;
        }
        self.alpha_cache.borrow_mut().insert(a, pid);
        pid
    }

    /// The configured background colour as an opaque RENDER `Color`.
    fn bg_color(&self) -> Color {
        Color {
            red: scale_channel(self.render.background[0]),
            green: scale_channel(self.render.background[1]),
            blue: scale_channel(self.render.background[2]),
            alpha: 65535,
        }
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
        clear: &[Rect], // buffer_age()==1: only the damaged region is repainted each present
    ) -> Result<()> {
        let bg = self.bg_color();
        let full = Rectangle {
            x: 0,
            y: 0,
            width: screen_w.max(0) as u16,
            height: screen_h.max(0) as u16,
        };

        // Partial repaint: the back-buffer PERSISTS (holds last frame; buffer_age()==1), so
        // update only the damaged region — clear the damaged background rects, composite the
        // (already damage-clipped) windows, then copy just the damage to the overlay. Reset
        // the back clip to full first so a leftover per-window clip doesn't restrict the fill.
        let back = self.ensure_back(screen_w, screen_h)?;
        self.conn.render_set_picture_clip_rectangles(back, 0, 0, &[full])?;
        // The total region that changed this frame (bg clears ∪ window updates); copied to
        // the overlay at the end (the rest of the overlay retains the previous frame).
        let mut damage: Vec<Rectangle> = rects(clear);
        if !damage.is_empty() {
            self.conn.render_fill_rectangles(PictOp::SRC, back, bg, &damage)?;
        }

        // Composite each window bottom-to-top (session already ordered + damage-clipped `items`).
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
            if should_skip(it) {
                continue;
            }
            let depth = self.pixmap_depth(q.pixmap);
            let fmt = if depth == 32 { self.fmt_argb32 } else { self.fmt_rgb24 };
            let Ok(src) = self.conn.generate_id() else { continue };
            if self.conn.render_create_picture(src, q.pixmap, fmt, &CreatePictureAux::new()).is_err() {
                continue;
            }
            // Clip this window's draws to its visible + damaged rects; those rects are also
            // part of the region copied to the overlay below.
            let clip = rects(&it.clip);
            damage.extend_from_slice(&clip);
            let _ = self.conn.render_set_picture_clip_rectangles(back, 0, 0, &clip);
            // Per-window opacity (fade × inactive-dim, already folded upstream) via a
            // cached constant-alpha mask; `NONE` = fully opaque.
            let a = quantize_opacity(q.opacity);
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

        // Present: copy ONLY the damaged region from the persistent back-buffer to the
        // overlay (the rest retains the previous frame). Nothing changed -> nothing to do.
        if damage.is_empty() {
            self.conn.flush().ok();
            return Ok(());
        }
        self.conn.render_set_picture_clip_rectangles(back, 0, 0, &[full])?; // read back fully as source
        self.conn.render_set_picture_clip_rectangles(self.target, 0, 0, &damage)?;
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
        1 // single persistent back-buffer holds last frame -> session damage-clips (partial repaint)
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

#[cfg(test)]
mod tests;
