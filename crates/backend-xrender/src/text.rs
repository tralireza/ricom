//! Pure fontdue text metrics + rasterisation for the XRender backend — **no X here**.
//!
//! xrender is shaderless, so on-screen text can't use GL's atlas+shader path. Instead the
//! backend turns each rasterised glyph's grayscale coverage into an X **A8 mask Picture**
//! and composites `Composite(OVER, colour_fill, glyph_A8_mask, back)` at the pen — the
//! RENDER analogue of a coverage blit. This module owns only the *pure* half: the font,
//! the metric caches, and the per-glyph coverage raster (mirrors the fontdue half of
//! `backend-gl/src/text.rs`, minus the GL atlas). Being X-free, it unit-tests on the Mac.
//!
//! Layout is one glyph per codepoint, left to right by advance width; sizes round to the
//! nearest integer px (the cache key). No shaping/hinting (fontdue is unhinted) — the
//! same "lean" contract as the GL text path.

use std::cell::RefCell;
use std::collections::HashMap;

use anyhow::{anyhow, Result};
use fontdue::{Font, FontSettings};

/// Clamp on the rasterised glyph pixel size (sanity bound on the cache key).
const MAX_PX: f32 = 512.0;

/// Round a requested (possibly fractional) size to the integer cache key, clamped.
pub fn px_key(px: f32) -> u32 {
    px.round().clamp(1.0, MAX_PX) as u32
}

/// One rasterised glyph: its A8 coverage (row-major `w×h`) plus placement in px at that
/// size. `off_x`/`off_y` position the coverage box relative to the pen and the line's top
/// (ascent-based), matching the GL backend's placement so layout is identical.
pub struct GlyphRaster {
    pub cov: Vec<u8>,
    pub w: usize,
    pub h: usize,
    pub off_x: f32,
    pub off_y: f32,
}

/// A parsed font with cached advances + line metrics. Interior-mutable caches so
/// `measure`/`line_height` take `&self` (as the backend's `present_windows` requires).
pub struct TextFont {
    font: Font,
    /// Advance widths (px) keyed by (char, size_px).
    advances: RefCell<HashMap<(char, u32), f32>>,
    /// (ascent, line_height) in px keyed by size_px.
    lines: RefCell<HashMap<u32, (f32, f32)>>,
}

impl TextFont {
    /// Parse a `.ttf`/`.otf` byte buffer. Errors if fontdue can't parse it (the backend
    /// then keeps `text: None` → no HUD/OSD text, compositor still runs).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let font = Font::from_bytes(bytes, FontSettings::default())
            .map_err(|e| anyhow!("parse font: {e}"))?;
        Ok(TextFont {
            font,
            advances: RefCell::new(HashMap::new()),
            lines: RefCell::new(HashMap::new()),
        })
    }

    /// (ascent, line_height) in px at integer size `size` (cached).
    fn line_metrics(&self, size: u32) -> (f32, f32) {
        if let Some(v) = self.lines.borrow().get(&size) {
            return *v;
        }
        let v = self
            .font
            .horizontal_line_metrics(size as f32)
            .map(|m| (m.ascent, m.new_line_size))
            .unwrap_or((size as f32 * 0.8, size as f32 * 1.2));
        self.lines.borrow_mut().insert(size, v);
        v
    }

    /// Advance width of `ch` at integer size `size` in px (cached; no rasterisation).
    fn advance(&self, ch: char, size: u32) -> f32 {
        if let Some(a) = self.advances.borrow().get(&(ch, size)) {
            return *a;
        }
        let a = self.font.metrics(ch, size as f32).advance_width;
        self.advances.borrow_mut().insert((ch, size), a);
        a
    }

    /// On-screen size of `s` at glyph height `px`: `(width, line_height)` in pixels.
    pub fn measure(&self, px: f32, s: &str) -> (f32, f32) {
        let sz = px_key(px);
        let w: f32 = s.chars().map(|c| self.advance(c, sz)).sum();
        (w, self.line_metrics(sz).1)
    }

    /// The line height (row pitch) at glyph size `px`, in pixels.
    pub fn line_height(&self, px: f32) -> f32 {
        self.line_metrics(px_key(px)).1
    }

    /// Advance of one char at size `px` (for pen stepping in the draw loop).
    pub fn advance_px(&self, ch: char, px: f32) -> f32 {
        self.advance(ch, px_key(px))
    }

    /// Rasterise `ch` at size `px` to A8 coverage + placement. `None` for a blank glyph
    /// (space / zero-area) — advance only, nothing to draw.
    pub fn raster(&self, ch: char, px: f32) -> Option<GlyphRaster> {
        let size = px_key(px);
        let (m, cov) = self.font.rasterize(ch, size as f32);
        if m.width == 0 || m.height == 0 {
            return None;
        }
        let ascent = self.line_metrics(size).0;
        Some(GlyphRaster {
            cov,
            w: m.width,
            h: m.height,
            off_x: m.xmin as f32,
            off_y: ascent - (m.ymin + m.height as i32) as f32,
        })
    }
}
