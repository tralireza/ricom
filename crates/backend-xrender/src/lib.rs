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
//! Textured blit + per-window opacity + occlusion/damage clip + inactive dim (free — dim is
//! folded into `Quad.opacity` upstream), plus HUD/OSD text via A8 glyph masks.
//!
//! **Tear-free present: a `POOL_N`-buffer flip swapchain.** Each frame composites into the
//! oldest idle pool buffer (its content is `POOL_N` frames old; `session`'s buffer-age
//! damage-union repaints exactly what changed since, reconstructing the current full frame),
//! then hands the whole buffer to the **Present** extension as a page-**flip** scheduled for
//! the next vblank (`update = NONE`, confirmed `mode=FLIP` — a partial `update` region would
//! force a tear-prone copy). Presentation is non-blocking (idle buffers are reclaimed from
//! `IdleNotify`), so it paces to the refresh without capping throughput. If Present is
//! unavailable (or `RICOM_XRENDER_NO_PRESENT` is set) it falls back to a single-buffer direct
//! RENDER copy (`buffer_age()==1`, can tear). Still deferred: drop shadow, rounded corners,
//! backdrop blur, and affine spin/drain via `SetPictureTransform`.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use x11rb::connection::Connection as _;
use x11rb::protocol::Event;
use x11rb::protocol::present::{self, ConnectionExt as _};
use x11rb::protocol::render::{
    Color, ConnectionExt as _, CreatePictureAux, PictOp, PictType, Pictforminfo, Pictformat, Picture,
};
use x11rb::protocol::xproto::{ConnectionExt as _, CreateGCAux, ImageFormat, Rectangle};
use x11rb::rust_connection::RustConnection;

pub use backend::*;

mod text;

/// Number of render-time samples the HUD graph shows (matches the GL backend's ring).
const HUD_GRAPH_SAMPLES: usize = 120;

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

/// Find the standard A8 (8-bit alpha-only) pict-format — the coverage-mask format for text
/// glyphs. DIRECT, depth 8, alpha in the low byte, no RGB channels.
fn find_a8_format(formats: &[Pictforminfo]) -> Option<Pictformat> {
    formats
        .iter()
        .find(|f| {
            f.type_ == PictType::DIRECT
                && f.depth == 8
                && f.direct.alpha_mask == 0xff
                && f.direct.red_mask == 0
                && f.direct.green_mask == 0
                && f.direct.blue_mask == 0
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

/// The flip swapchain's `buffer_age()` for a present count: a buffer is reused every `POOL_N`
/// presents, so its content is `POOL_N` frames old once every buffer has been drawn once
/// (`count >= POOL_N`); before that, `0` = full repaint. (Copy fallback returns 1 separately.)
fn pool_buffer_age(count: u64) -> i32 {
    if count >= POOL_N as u64 {
        POOL_N as i32
    } else {
        0
    }
}

/// Bytes per row for an A8 `PutImage`: X pads ZPixmap scanlines to the 32-bit boundary, so a
/// `w`-wide row of 8-bit coverage occupies `(w + 3) & !3` bytes.
fn a8_stride(w: usize) -> usize {
    (w + 3) & !3
}

/// Premultiply straight `rgba` (0..1) into a RENDER 16-bit `Color`, so `PictOp::OVER` blends
/// text + translucent panels correctly.
fn premul_color(rgba: [f32; 4]) -> Color {
    let a = rgba[3].clamp(0.0, 1.0);
    Color {
        red: scale_channel(rgba[0] * a),
        green: scale_channel(rgba[1] * a),
        blue: scale_channel(rgba[2] * a),
        alpha: scale_channel(a),
    }
}

/// Cache key for a straight `rgba` colour fill: packed RGBA8 (`0xAARRGGBB`).
fn color_key(rgba: [f32; 4]) -> u32 {
    let q = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u32;
    (q(rgba[3]) << 24) | (q(rgba[0]) << 16) | (q(rgba[1]) << 8) | q(rgba[2])
}

/// Frame-time graph bar colour (straight RGBA, before the whole-HUD opacity multiply): green
/// with ≥½ budget headroom, amber to 85%, red at/over the vblank budget (a missed frame).
fn graph_bar_color(ms: f32, budget: f32) -> [f32; 4] {
    if ms <= budget * 0.5 {
        [0.40, 0.90, 0.50, 0.90]
    } else if ms <= budget * 0.85 {
        [0.95, 0.80, 0.30, 0.90]
    } else {
        [0.95, 0.40, 0.35, 0.90]
    }
}

/// Outline tap offsets (px) for the multi-draw text outline: a full 8-tap ring for an
/// all-around outline, or 3 taps toward the bottom-right for a drop-shadow-style outline.
fn outline_ring(r: f32, drop: bool) -> Vec<(f32, f32)> {
    let h = r * std::f32::consts::FRAC_1_SQRT_2; // diagonal ring taps
    if drop {
        vec![(r, 0.0), (0.0, r), (h, h)]
    } else {
        vec![(r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r), (h, h), (-h, -h), (h, -h), (-h, h)]
    }
}

/// HUD panel top-left for a corner anchor, inset by `margin` from the screen edge.
fn hud_anchor(corner: HudCorner, sw: f32, sh: f32, panel_w: f32, panel_h: f32, margin: f32) -> (f32, f32) {
    match corner {
        HudCorner::TopLeft => (margin, margin),
        HudCorner::TopRight => (sw - margin - panel_w, margin),
        HudCorner::BottomLeft => (margin, sh - margin - panel_h),
        HudCorner::BottomRight => (sw - margin - panel_w, sh - margin - panel_h),
    }
}

/// Number of buffers in the flip swapchain. 2 = classic double-buffer (front on screen +
/// back being drawn); each buffer is reused every `POOL_N` frames → `buffer_age() == POOL_N`.
const POOL_N: usize = 2;

/// One buffer in the flip pool: a full-screen pixmap + Picture, plus the present serial it
/// is in-flight under (`0` = idle, safe to redraw). Presented whole-pixmap via Present flip.
struct FlipBuf {
    pixmap: u32,
    pic: Picture,
    busy: u32,
}

/// The flip swapchain: `POOL_N` full-screen buffers presented via Present **page-flip** (true
/// vblank sync — confirmed `mode=FLIP` on this driver). `count` is the monotonic present
/// index; frame N draws into `count % POOL_N`, so a buffer's content is `POOL_N` frames old
/// when reused → `buffer_age() == POOL_N`, and `session`'s damage-union reconstructs the
/// current frame onto it. Rebuilt on a screen-size change (resets `count`; session force-fulls).
struct FlipPool {
    bufs: Vec<FlipBuf>,
    w: i32,
    h: i32,
    count: u64,
}

/// Present-extension state. `None` when Present is unavailable or `RICOM_XRENDER_NO_PRESENT`
/// is set → the direct RENDER-copy fallback runs instead (single buffer, may tear). The
/// event-context id is allocated in `setup_present` and freed on connection close.
struct PresentState {
    /// Monotonic present serial, matched against `IdleNotify` to free the pool buffer.
    serial: Cell<u32>,
    /// Last completed present's MSC (vblank counter) from `CompleteNotify`; the next flip
    /// targets `last_msc + 1` so it lands at the next vblank.
    last_msc: Cell<u64>,
}

/// Text decoration (outline + drop-shadow) from `RenderParams`, scaled per draw. The
/// shaderless A8 path renders it by multi-offset draws (vs GL's single-pass shader dilation).
#[derive(Clone, Copy)]
struct TextStyle {
    outline_px: f32,
    outline_color: [f32; 4],
    outline_drop: bool,
    shadow_px: f32,
    shadow_color: [f32; 4],
}

const TEXT_STYLE_NONE: TextStyle = TextStyle {
    outline_px: 0.0,
    outline_color: [0.0; 4],
    outline_drop: false,
    shadow_px: 0.0,
    shadow_color: [0.0; 4],
};

/// A cached rasterised glyph: an A8 coverage-mask Picture (+ its px placement). Painted by
/// `Composite(OVER, colour_fill, this_mask, back)`. Freed in `Drop`. `Copy` (all ids/scalars).
#[derive(Clone, Copy)]
struct GlyphPic {
    pixmap: u32,
    pic: Picture,
    off_x: f32,
    off_y: f32,
    w: u16,
    h: u16,
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
    /// Flip swapchain, (re)built lazily to the screen size. Presented via Present page-flip.
    pool: RefCell<Option<FlipPool>>,
    /// Present-extension state for vblank-synced presentation; `None` → direct-copy fallback.
    present: Option<PresentState>,
    /// A8 (alpha-only) pict-format for text glyph coverage masks.
    fmt_a8: Pictformat,
    /// Parsed HUD/OSD font (`None` = text disabled). Set by `set_font`.
    text_font: RefCell<Option<text::TextFont>>,
    /// Config font-size multiplier (from `set_font`); scales HUD/OSD like the GL backend.
    font_scale: Cell<f32>,
    /// Per-`(char, size_px)` glyph mask cache; `None` = a blank glyph (space).
    glyph_cache: RefCell<HashMap<(char, u32), Option<GlyphPic>>>,
    /// A depth-8 GC (+ the 1×1 pixmap it was created on) for `PutImage`ing glyph coverage.
    text_gc: u32,
    text_gc_pixmap: u32,
    /// Premultiplied solid-fill colour Pictures for text/panels, keyed by packed RGBA8.
    color_cache: RefCell<HashMap<u32, Picture>>,
    /// Last frame's composite wall-clock (ms) — the HUD's render-time metric (there's no GPU
    /// timer over the wire, so we measure the CPU composite cost instead of returning 0).
    render_ms: Cell<f32>,
    /// Ring of recent composite times (ms) feeding the HUD render-time graph.
    render_samples: RefCell<VecDeque<f32>>,
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
        let fmt_a8 = find_a8_format(&formats.formats)
            .ok_or_else(|| anyhow!("xrender: no a8 (depth-8) pict format for text"))?;

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

        // Bring up Present + XFixes for tear-free vblank-synced presentation (best-effort;
        // falls back to a direct RENDER copy if unavailable).
        let present = Self::setup_present(&conn, window);

        // A GC is bound to a drawable depth, so make a depth-8 GC (on a throwaway 1×1
        // depth-8 pixmap, kept for the GC's lifetime) for `PutImage`ing glyph coverage.
        let text_gc_pixmap = conn.generate_id().context("xrender: gen text-gc pixmap id")?;
        conn.create_pixmap(8, text_gc_pixmap, window, 1, 1).context("xrender: text-gc pixmap")?;
        let text_gc = conn.generate_id().context("xrender: gen text gc id")?;
        conn.create_gc(text_gc, text_gc_pixmap, &CreateGCAux::new()).context("xrender: create text gc")?;

        tracing::info!(
            overlay = window,
            depth = odepth,
            present = present.is_some(),
            "xrender backend up (pure x11rb, no EGL)"
        );
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
            pool: RefCell::new(None),
            present,
            fmt_a8,
            text_font: RefCell::new(None),
            font_scale: Cell::new(1.0),
            glyph_cache: RefCell::new(HashMap::new()),
            text_gc,
            text_gc_pixmap,
            color_cache: RefCell::new(HashMap::new()),
            render_ms: Cell::new(0.0),
            render_samples: RefCell::new(VecDeque::new()),
        })
    }

    /// Ensure the flip pool exists at `w × h`, rebuilding it (freeing the old) on a resize.
    /// Allocates `POOL_N` full-screen buffers, each cleared to the background, and resets
    /// `count` (so `buffer_age()` restarts at 0 — session force-fulls the first frames anyway).
    fn ensure_pool(&self, w: i32, h: i32) -> Result<()> {
        let mut pool = self.pool.borrow_mut();
        if pool.as_ref().is_none_or(|p| p.w != w || p.h != h) {
            if let Some(old) = pool.take() {
                for b in old.bufs {
                    let _ = self.conn.render_free_picture(b.pic);
                    let _ = self.conn.free_pixmap(b.pixmap);
                }
            }
            let bfmt = if self.depth == 32 { self.fmt_argb32 } else { self.fmt_rgb24 };
            let fr = Rectangle { x: 0, y: 0, width: w.max(1) as u16, height: h.max(1) as u16 };
            let mut bufs = Vec::with_capacity(POOL_N);
            for _ in 0..POOL_N {
                let pixmap = self.conn.generate_id().context("xrender: gen pool pixmap id")?;
                self.conn
                    .create_pixmap(self.depth, pixmap, self.overlay, w.max(1) as u16, h.max(1) as u16)
                    .context("xrender: create pool pixmap")?;
                let pic = self.conn.generate_id().context("xrender: gen pool pic id")?;
                self.conn
                    .render_create_picture(pic, pixmap, bfmt, &CreatePictureAux::new())
                    .context("xrender: create pool Picture")?;
                // Clear fresh buffers to bg so a flip never shows garbage before paint.
                let _ = self.conn.render_set_picture_clip_rectangles(pic, 0, 0, &[fr]);
                let _ = self.conn.render_fill_rectangles(PictOp::SRC, pic, self.bg_color(), &[fr]);
                bufs.push(FlipBuf { pixmap, pic, busy: 0 });
            }
            *pool = Some(FlipPool { bufs, w, h, count: 0 });
        }
        Ok(())
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

    /// Try to bring up Present + XFixes for tear-free vblank-synced presentation on
    /// `overlay`. Returns `None` (→ direct-copy fallback) if either extension is missing
    /// or `RICOM_XRENDER_NO_PRESENT` is set. Best-effort: any failure degrades gracefully.
    fn setup_present(conn: &RustConnection, overlay: u32) -> Option<PresentState> {
        if std::env::var_os("RICOM_XRENDER_NO_PRESENT").is_some() {
            tracing::info!("xrender: Present disabled via RICOM_XRENDER_NO_PRESENT → direct-copy present");
            return None;
        }
        conn.present_query_version(1, 0).ok()?.reply().ok()?;
        // Select Complete/Idle notifies on the overlay. The event-context id is freed on
        // connection close (backend lifetime == session), so we don't retain it.
        let eid = conn.generate_id().ok()?;
        conn.present_select_input(
            eid,
            overlay,
            present::EventMask::COMPLETE_NOTIFY | present::EventMask::IDLE_NOTIFY,
        )
        .ok()?;
        conn.flush().ok();
        tracing::info!("xrender: Present up (page-flip vsync, {POOL_N}-buffer swapchain)");
        Some(PresentState { serial: Cell::new(0), last_msc: Cell::new(0) })
    }

    /// Drain (non-blocking) any pending Present events: advance the MSC clock from
    /// `CompleteNotify`, and free pool buffers whose in-flight serial went idle.
    fn drain_present_events(&self) {
        let Some(p) = &self.present else { return };
        let mut pool = self.pool.borrow_mut();
        while let Ok(Some(ev)) = self.conn.poll_for_event() {
            match ev {
                Event::PresentCompleteNotify(e) => p.last_msc.set(e.msc),
                Event::PresentIdleNotify(e) => {
                    if let Some(pool) = pool.as_mut() {
                        for b in pool.bufs.iter_mut() {
                            if b.busy == e.serial {
                                b.busy = 0;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Wait (bounded) until pool buffer `idx` is idle (its flip released it), so it's safe to
    /// redraw. Normally already free (released a frame ago); the poll+sleep is a backstop with
    /// a hard deadline so a missing notify degrades to a skipped pace, never a freeze.
    fn wait_buf_idle(&self, idx: usize) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(100);
        loop {
            {
                let pool = self.pool.borrow();
                if pool.as_ref().is_none_or(|p| p.bufs[idx].busy == 0) {
                    return;
                }
            }
            match self.conn.poll_for_event() {
                Ok(Some(Event::PresentCompleteNotify(e))) => {
                    if let Some(p) = &self.present {
                        p.last_msc.set(e.msc);
                    }
                }
                Ok(Some(Event::PresentIdleNotify(e))) => {
                    if let Some(pool) = self.pool.borrow_mut().as_mut() {
                        for b in pool.bufs.iter_mut() {
                            if b.busy == e.serial {
                                b.busy = 0;
                            }
                        }
                    }
                }
                Ok(Some(_)) => {}
                Ok(None) => {
                    if std::time::Instant::now() >= deadline {
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Err(_) => return,
            }
        }
    }

    /// A **premultiplied** solid-fill Picture for straight `rgba` (0..1), cached by packed
    /// RGBA8. Premultiplied so `PictOp::OVER` blends text/translucent panels correctly.
    fn solid(&self, rgba: [f32; 4]) -> Picture {
        let key = color_key(rgba);
        if let Some(&p) = self.color_cache.borrow().get(&key) {
            return p;
        }
        let color = premul_color(rgba);
        let Ok(pid) = self.conn.generate_id() else { return x11rb::NONE };
        if self.conn.render_create_solid_fill(pid, color).is_err() {
            return x11rb::NONE;
        }
        self.color_cache.borrow_mut().insert(key, pid);
        pid
    }

    /// Upload one rasterised glyph's A8 coverage into a depth-8 pixmap + Picture. Scanlines
    /// are padded to the X 32-bit boundary (ZPixmap requirement). `None` on any X failure.
    fn upload_glyph(&self, gr: &text::GlyphRaster) -> Option<GlyphPic> {
        let (w, h) = (gr.w.max(1) as u16, gr.h.max(1) as u16);
        let stride = a8_stride(gr.w); // pad each row to the 32-bit scanline boundary
        let mut buf = vec![0u8; stride * gr.h];
        for row in 0..gr.h {
            let (src, dst) = (row * gr.w, row * stride);
            buf[dst..dst + gr.w].copy_from_slice(&gr.cov[src..src + gr.w]);
        }
        let pixmap = self.conn.generate_id().ok()?;
        if self.conn.create_pixmap(8, pixmap, self.overlay, w, h).is_err() {
            return None;
        }
        if self
            .conn
            .put_image(ImageFormat::Z_PIXMAP, pixmap, self.text_gc, w, h, 0, 0, 0, 8, &buf)
            .is_err()
        {
            let _ = self.conn.free_pixmap(pixmap);
            return None;
        }
        let Ok(pic) = self.conn.generate_id() else {
            let _ = self.conn.free_pixmap(pixmap);
            return None;
        };
        if self
            .conn
            .render_create_picture(pic, pixmap, self.fmt_a8, &CreatePictureAux::new())
            .is_err()
        {
            let _ = self.conn.free_pixmap(pixmap);
            return None;
        }
        Some(GlyphPic { pixmap, pic, off_x: gr.off_x, off_y: gr.off_y, w, h })
    }

    /// Glyph mask for `(ch, size)`, rasterising + uploading on first use (cached). `None` for
    /// a blank glyph (space) or on failure / when no font is set.
    fn glyph_pic(&self, ch: char, size: u32) -> Option<GlyphPic> {
        if let Some(entry) = self.glyph_cache.borrow().get(&(ch, size)) {
            return *entry;
        }
        let raster = self.text_font.borrow().as_ref().and_then(|f| f.raster(ch, size as f32));
        let entry = raster.and_then(|gr| self.upload_glyph(&gr));
        self.glyph_cache.borrow_mut().insert((ch, size), entry);
        entry
    }

    /// Measure `s` at glyph height `px` → `(width, line_height)` in px (`(0,0)` if no font).
    fn measure(&self, px: f32, s: &str) -> (f32, f32) {
        self.text_font.borrow().as_ref().map(|f| f.measure(px, s)).unwrap_or((0.0, 0.0))
    }

    /// Line pitch at glyph height `px` (0 if no font).
    fn text_line_height(&self, px: f32) -> f32 {
        self.text_font.borrow().as_ref().map(|f| f.line_height(px)).unwrap_or(0.0)
    }

    /// Fill a rectangle on `back` with straight `rgba` via `OVER` (blends translucent panels
    /// + graph bars). Corner radius is not modelled (square — first cut).
    #[allow(clippy::too_many_arguments)]
    fn fill_rect(&self, back: Picture, sw: i32, sh: i32, x: f32, y: f32, w: f32, h: f32, rgba: [f32; 4]) {
        if w <= 0.0 || h <= 0.0 || rgba[3] <= 0.0 {
            return;
        }
        let full = Rectangle { x: 0, y: 0, width: sw.max(0) as u16, height: sh.max(0) as u16 };
        let _ = self.conn.render_set_picture_clip_rectangles(back, 0, 0, &[full]);
        let c = premul_color(rgba);
        let rect = Rectangle {
            x: x.round() as i16,
            y: y.round() as i16,
            width: w.round().max(0.0) as u16,
            height: h.round().max(0.0) as u16,
        };
        let _ = self.conn.render_fill_rectangles(PictOp::OVER, back, c, &[rect]);
    }

    /// Draw `s` at (`x`, `y`) — `y` is the line top — glyph height `px`, straight `color`, onto
    /// `back`. Each glyph = `Composite(OVER, colour_fill, glyph_A8_mask, back)` at the pen.
    #[allow(clippy::too_many_arguments)]
    fn draw_run(&self, back: Picture, sw: i32, sh: i32, x: f32, y: f32, px: f32, color: [f32; 4], s: &str) {
        let size = text::px_key(px);
        let src = self.solid(color);
        if src == x11rb::NONE {
            return;
        }
        let full = Rectangle { x: 0, y: 0, width: sw.max(0) as u16, height: sh.max(0) as u16 };
        let _ = self.conn.render_set_picture_clip_rectangles(back, 0, 0, &[full]);
        let mut pen = x;
        for ch in s.chars() {
            let adv = self.text_font.borrow().as_ref().map(|f| f.advance_px(ch, px)).unwrap_or(0.0);
            if let Some(g) = self.glyph_pic(ch, size) {
                let gx = (pen + g.off_x).round() as i16;
                let gy = (y + g.off_y).round() as i16;
                let _ = self.conn.render_composite(
                    PictOp::OVER, src, g.pic, back, 0, 0, 0, 0, gx, gy, g.w, g.h,
                );
            }
            pen += adv;
        }
    }

    /// Build the text decoration (outline + drop-shadow) from `RenderParams`, scaled by `s`,
    /// with whole-run alpha `a`. Mirrors the GL backend's `text_style`.
    fn text_style(&self, s: f32, a: f32) -> TextStyle {
        let r = &self.render;
        TextStyle {
            outline_px: r.text_outline * s,
            outline_color: [r.text_outline_color[0], r.text_outline_color[1], r.text_outline_color[2], a],
            outline_drop: r.text_outline_drop,
            shadow_px: r.text_shadow * s,
            shadow_color: [r.text_shadow_color[0], r.text_shadow_color[1], r.text_shadow_color[2], a],
        }
    }

    /// Draw `s` with an optional drop-shadow + outline (see [`TextStyle`]). The shaderless A8
    /// path fakes GL's single-pass dilation with a ring of offset draws in the outline colour,
    /// then the fill on top (drop = a bottom-right band only). `NONE` ⇒ a plain `draw_run`.
    #[allow(clippy::too_many_arguments)]
    fn draw_styled(&self, back: Picture, sw: i32, sh: i32, x: f32, y: f32, px: f32, fill: [f32; 4], style: &TextStyle, s: &str) {
        if style.shadow_px > 0.0 && style.shadow_color[3] > 0.0 {
            self.draw_run(back, sw, sh, x + style.shadow_px, y + style.shadow_px, px, style.shadow_color, s);
        }
        if style.outline_px > 0.0 && style.outline_color[3] > 0.0 {
            for (dx, dy) in outline_ring(style.outline_px, style.outline_drop) {
                self.draw_run(back, sw, sh, x + dx, y + dy, px, style.outline_color, s);
            }
        }
        self.draw_run(back, sw, sh, x, y, px, fill, s);
    }

    /// Draw the FPS HUD onto `back` — panel + fps/ms numbers + optional render-time graph +
    /// optional 1m/5m/15m load block — anchored to `hud.corner`. Ported from the GL layout;
    /// plain text (no outline), square panel (no rounding) in this first cut.
    fn draw_hud(&self, back: Picture, hud: &Hud, sw: i32, sh: i32) {
        if self.text_font.borrow().is_none() {
            return;
        }
        let s = (sh as f32 / 1080.0).max(0.5) * hud.scale * self.font_scale.get();
        let pad = 8.0 * s;
        let margin = 28.0 * s;
        let text_px = 20.0 * s;
        let budget = 1000.0 / hud.refresh_hz.max(1.0);
        let render_ms = self.render_ms.get();
        let th = self.text_line_height(text_px);
        let fps_s = format!("{}", hud.fps);
        let ms_s = format!("{render_ms:.1}");
        let numw = self.measure(text_px, "000").0;
        let msw = self.measure(text_px, "000.0").0;
        let sep1 = " fps   ";
        let sep2 = " ms";
        let sep1w = self.measure(text_px, sep1).0;
        let sep2w = self.measure(text_px, sep2).0;
        let tw = numw + sep1w + msw + sep2w;
        let graph_h = if hud.graph { 34.0 * s } else { 0.0 };
        let graph_gap = if hud.graph { 6.0 * s } else { 0.0 };
        let load_px = 15.0 * s;
        let has_load = hud.load.is_some();
        let load_lbl_w = if has_load { self.measure(load_px, "fps  ").0 } else { 0.0 };
        let load_col_w = if has_load { self.measure(load_px, "  000.0").0 } else { 0.0 };
        let load_w = if has_load { load_lbl_w + 3.0 * load_col_w } else { 0.0 };
        let load_pitch = load_px * 1.2;
        let load_cell = if has_load { self.text_line_height(load_px) } else { 0.0 };
        let load_gap = if has_load { 8.0 * s } else { 0.0 };
        let load_block_h = if has_load { load_gap + load_pitch + load_cell } else { 0.0 };
        let content_w = tw.max(load_w);
        let bar_w = if hud.graph { (content_w / HUD_GRAPH_SAMPLES as f32).max(1.0) } else { 0.0 };
        let panel_w = content_w + pad * 2.0;
        let panel_h = th + graph_gap + graph_h + load_block_h + pad * 2.0;
        let a = hud.opacity;
        let (px, py) = hud_anchor(hud.corner, sw as f32, sh as f32, panel_w, panel_h, margin);
        self.fill_rect(back, sw, sh, px, py, panel_w, panel_h, [0.05, 0.05, 0.07, 0.72 * a]);
        if hud.graph {
            let samples = self.render_samples.borrow();
            if !samples.is_empty() {
                let gx = px + pad;
                let gy = py + pad + th + graph_gap;
                for (i, &ms) in samples.iter().enumerate() {
                    let bx = gx + i as f32 * bar_w;
                    if bx >= gx + content_w {
                        break;
                    }
                    let norm = (ms / budget).clamp(0.0, 1.0);
                    let bh = (norm * graph_h).max(1.0);
                    let mut col = graph_bar_color(ms, budget);
                    col[3] *= a;
                    self.fill_rect(back, sw, sh, bx, gy + (graph_h - bh), (bar_w - 0.5 * s).max(1.0), bh, col);
                }
                self.fill_rect(back, sw, sh, gx, gy, content_w, s.max(1.0), [1.0, 1.0, 1.0, 0.22 * a]);
            }
        }
        let x0 = px + pad;
        let ny = py + pad;
        let col = [0.90, 1.0, 0.95, a];
        // Outlined when `hud.outline` (reads without the panel), else plain — matches GL.
        let hstyle = if hud.outline { self.text_style(s, a) } else { TEXT_STYLE_NONE };
        let fw = self.measure(text_px, &fps_s).0;
        self.draw_styled(back, sw, sh, x0 + numw - fw, ny, text_px, col, &hstyle, &fps_s);
        self.draw_styled(back, sw, sh, x0 + numw, ny, text_px, col, &hstyle, sep1);
        let mx0 = x0 + numw + sep1w;
        let mw = self.measure(text_px, &ms_s).0;
        self.draw_styled(back, sw, sh, mx0 + msw - mw, ny, text_px, col, &hstyle, &ms_s);
        self.draw_styled(back, sw, sh, mx0 + msw, ny, text_px, col, &hstyle, sep2);
        if let Some(l) = &hud.load {
            let lcol = [0.80, 0.88, 1.0, a];
            let rows: [(&str, [Option<f32>; 3]); 2] = [
                ("fps", [Some(l.fps[0]), Some(l.fps[1]), Some(l.fps[2])]),
                ("ms", [l.render_ms[0], l.render_ms[1], l.render_ms[2]]),
            ];
            let mut ly = py + pad + th + graph_gap + graph_h + load_gap;
            for (label, vals) in rows {
                self.draw_styled(back, sw, sh, x0, ly, load_px, lcol, &hstyle, label);
                for (k, v) in vals.iter().enumerate() {
                    let vs = match v {
                        Some(x) => format!("{x:.1}"),
                        None => "--".to_string(),
                    };
                    let cx = x0 + load_lbl_w + k as f32 * load_col_w;
                    let vw = self.measure(load_px, &vs).0;
                    self.draw_styled(back, sw, sh, cx + load_col_w - vw, ly, load_px, lcol, &hstyle, &vs);
                }
                ly += load_pitch;
            }
        }
    }

    /// Draw the OSD toast onto `back` — a top-centred banner. Ported from the GL layout with
    /// per-glyph ellipsis truncation + Pop/Slide positioning; Unroll/Stretch fade in this cut
    /// (no scissor-reveal); plain text, square box.
    fn draw_osd(&self, back: Picture, osd: &Osd, sw: i32, sh: i32) {
        let p = osd.presence.clamp(0.0, 1.0);
        if p <= 0.0 || self.text_font.borrow().is_none() {
            return;
        }
        let sb = (sh as f32 / 1080.0).max(0.5) * osd.scale * self.font_scale.get();
        let zoom = if osd.effect == OsdEffect::Pop { 0.6 + 0.4 * p } else { 1.0 };
        let s = sb * zoom;
        let pad = 20.0 * s;
        let text_px = 34.0 * s;
        let line_h = self.text_line_height(text_px);
        let hmargin = 40.0 * sb;
        let min_w = self.measure(text_px, "x").0;
        let max_text_w = (sw as f32 - 2.0 * hmargin - 2.0 * pad).max(min_w);
        let ell_w = self.measure(text_px, "...").0;
        let lines: Vec<String> = osd
            .text
            .split('\n')
            .map(|l| {
                if self.measure(text_px, l).0 <= max_text_w {
                    return l.to_string();
                }
                let budget = (max_text_w - ell_w).max(0.0);
                let mut kept = String::new();
                let mut w = 0.0;
                let mut buf = [0u8; 4];
                for ch in l.chars() {
                    let cw = self.measure(text_px, ch.encode_utf8(&mut buf)).0;
                    if w + cw > budget {
                        break;
                    }
                    w += cw;
                    kept.push(ch);
                }
                kept.push_str("...");
                kept
            })
            .collect();
        let tw = lines.iter().map(|l| self.measure(text_px, l).0).fold(0.0_f32, f32::max);
        let panel_w = tw + 2.0 * pad;
        let panel_h = lines.len() as f32 * line_h + 2.0 * pad;
        let px = (sw as f32 - panel_w) * 0.5;
        let rest_y = 28.0 * sb;
        let py = if osd.effect == OsdEffect::Slide {
            -panel_h + (rest_y + panel_h) * p
        } else {
            rest_y
        };
        let alpha = if matches!(osd.effect, OsdEffect::Unroll | OsdEffect::Stretch) { 1.0 } else { p };
        let bg = osd.background;
        if bg[3] > 0.001 {
            self.fill_rect(back, sw, sh, px, py, panel_w, panel_h, [bg[0], bg[1], bg[2], bg[3] * alpha]);
        }
        let tx = px + pad;
        let c = osd.color;
        let style = if osd.outline { self.text_style(s, alpha) } else { TEXT_STYLE_NONE };
        for (i, line) in lines.iter().enumerate() {
            let ly = py + pad + i as f32 * line_h;
            self.draw_styled(back, sw, sh, tx, ly, text_px, [c[0], c[1], c[2], alpha], &style, line);
        }
    }
}

impl backend::Backend for XrenderBackend {
    fn present_windows(
        &self,
        items: &[WindowDraw],
        screen_w: i32,
        screen_h: i32,
        hud: Option<&Hud>,
        osd: Option<&Osd>,
        clear: &[Rect], // buffer_age()==1: only the damaged region is repainted each present
    ) -> Result<()> {
        let t0 = Instant::now();
        let bg = self.bg_color();
        let full = Rectangle {
            x: 0,
            y: 0,
            width: screen_w.max(0) as u16,
            height: screen_h.max(0) as u16,
        };

        // Pick this frame's pool buffer. Flip mode alternates `count % POOL_N` (so each buffer
        // is POOL_N frames old on reuse → buffer_age()==POOL_N); the copy fallback always uses
        // buffer 0 (age 1). `session`'s damage-union (driven by buffer_age) repaints exactly
        // what changed since this buffer was last drawn, reconstructing the current full frame.
        self.ensure_pool(screen_w, screen_h)?;
        let flip = self.present.is_some();
        self.drain_present_events(); // free buffers whose flip has completed
        let idx = if flip {
            let pool = self.pool.borrow();
            (pool.as_ref().map(|p| p.count).unwrap_or(0) % POOL_N as u64) as usize
        } else {
            0
        };
        if flip {
            self.wait_buf_idle(idx); // ensure the target buffer's previous flip released it
        }
        let back = self.pool.borrow().as_ref().unwrap().bufs[idx].pic;
        self.conn.render_set_picture_clip_rectangles(back, 0, 0, &[full])?;
        // Repaint the damaged region onto this buffer (clear damaged bg, then composite the
        // damage-clipped windows below); the whole buffer is then flipped at vblank.
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

        // HUD / OSD overlays. `session` force-fulls whenever either is up (own_full includes
        // show_fps || osd), so `back` was bg-cleared + fully recomposited this frame — drawing
        // them fresh with OVER doesn't accumulate, and they sit inside the full-screen damage.
        if hud.is_some() || osd.is_some() {
            let _ = self.conn.render_set_picture_clip_rectangles(back, 0, 0, &[full]);
            if let Some(o) = osd {
                self.draw_osd(back, o, screen_w, screen_h);
            }
            if let Some(h) = hud {
                self.draw_hud(back, h, screen_w, screen_h);
            }
        }
        // Record this frame's composite cost (CPU wall-clock — no GPU timer over the wire) for
        // the HUD number + graph. Measured before the vblank wait, so it's the render cost.
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        self.render_ms.set(ms);
        {
            let mut samples = self.render_samples.borrow_mut();
            samples.push_back(ms);
            while samples.len() > HUD_GRAPH_SAMPLES {
                samples.pop_front();
            }
        }

        // Nothing changed -> nothing to present.
        if damage.is_empty() {
            self.conn.flush().ok();
            return Ok(());
        }
        let back_pixmap = self.pool.borrow().as_ref().unwrap().bufs[idx].pixmap;
        if let Some(p) = &self.present {
            // Tear-free: page-FLIP the whole buffer at the next vblank (confirmed mode=FLIP).
            // `update = NONE` presents the whole pixmap so the server can flip (an update region
            // would force a copy); `options = 0` allows the flip. Non-blocking — the next frame's
            // `wait_buf_idle` handles reuse — so presentation pipelines smoothly (no rate cap).
            let serial = p.serial.get().wrapping_add(1);
            p.serial.set(serial);
            let target_msc = p.last_msc.get().wrapping_add(1); // next vblank
            self.conn.present_pixmap(
                self.overlay,
                back_pixmap,
                serial,
                x11rb::NONE, // valid
                x11rb::NONE, // update = NONE → whole pixmap → lets the server page-flip
                0,
                0,           // x_off, y_off
                x11rb::NONE, // target_crtc: any
                x11rb::NONE, // wait_fence
                x11rb::NONE, // idle_fence
                0,           // options = 0 → allow flip
                target_msc,
                0,           // divisor
                0,           // remainder
                &[],         // notifies
            )?;
            self.conn.flush().ok();
            if let Some(pool) = self.pool.borrow_mut().as_mut() {
                pool.bufs[idx].busy = serial;
                pool.count = pool.count.wrapping_add(1);
            }
            self.drain_present_events(); // pick up an immediate completion / idle
        } else {
            // Fallback (Present unavailable / RICOM_XRENDER_NO_PRESENT): copy the whole buffer to
            // the overlay — no vblank sync, so it can tear under motion (buffer_age()==1).
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
        }
        Ok(())
    }

    fn set_render_params(&mut self, render: RenderParams) {
        self.render = render;
    }

    fn set_font(&mut self, path: &str, size: f32) {
        self.font_scale.set(if size > 0.0 { size } else { 1.0 });
        // Free the old glyph masks (size/font-specific) + reset the cache on any font change.
        for g in self.glyph_cache.borrow().values().flatten() {
            let _ = self.conn.render_free_picture(g.pic);
            let _ = self.conn.free_pixmap(g.pixmap);
        }
        self.glyph_cache.borrow_mut().clear();
        let loaded = if path.is_empty() {
            None
        } else {
            match std::fs::read(path) {
                Ok(bytes) => match text::TextFont::from_bytes(&bytes) {
                    Ok(f) => Some(f),
                    Err(e) => {
                        tracing::warn!("xrender: font parse failed ({path}): {e:#}");
                        None
                    }
                },
                Err(e) => {
                    tracing::warn!("xrender: font read failed ({path}): {e}");
                    None
                }
            }
        };
        let enabled = loaded.is_some();
        *self.text_font.borrow_mut() = loaded;
        self.conn.flush().ok();
        tracing::info!(text = enabled, "xrender: font set");
    }

    fn has_text(&self) -> bool {
        self.text_font.borrow().is_some()
    }

    fn render_ms(&self) -> f32 {
        self.render_ms.get() // CPU composite cost (no GPU timer over the wire)
    }

    fn buffer_age(&self) -> i32 {
        // Copy fallback: one retained buffer holds last frame → age 1.
        if self.present.is_none() {
            return 1;
        }
        // Flip mode: the buffer we'll draw next (`count % POOL_N`) was last drawn POOL_N frames
        // ago → age POOL_N, once every buffer has been drawn once (else 0 = full repaint). No
        // pool yet → 0. (Resize resets `count`; session force-fulls that frame regardless.)
        self.drain_present_events();
        match self.pool.borrow().as_ref() {
            Some(p) => pool_buffer_age(p.count),
            None => 0,
        }
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
        if let Some(pool) = self.pool.borrow_mut().take() {
            for b in pool.bufs {
                let _ = self.conn.render_free_picture(b.pic);
                let _ = self.conn.free_pixmap(b.pixmap);
            }
        }
        for g in self.glyph_cache.borrow().values().flatten() {
            let _ = self.conn.render_free_picture(g.pic);
            let _ = self.conn.free_pixmap(g.pixmap);
        }
        for &p in self.color_cache.borrow().values() {
            let _ = self.conn.render_free_picture(p);
        }
        let _ = self.conn.free_gc(self.text_gc);
        let _ = self.conn.free_pixmap(self.text_gc_pixmap);
        let _ = self.conn.flush();
    }
}

#[cfg(test)]
mod tests;
