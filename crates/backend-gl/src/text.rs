//! Runtime TrueType text renderer: a **per-size native glyph cache**. Each glyph is
//! rasterised at its exact on-screen pixel size (via the pure-Rust `fontdue` crate)
//! into a dynamic grayscale (R8) atlas, cached by `(char, size_px)`. There is no SDF:
//! text uses native-ppem antialiasing at the size actually drawn, so it's crisp at
//! small sizes (no distance-field minification softness), and it's cheap — a glyph is
//! rasterised once per `(glyph, size)` and thereafter just blitted.
//!
//! Handles any UTF-8 codepoint the font contains, proportionally (per-glyph advances);
//! window titles and `ricomctl notify` text render in whatever scripts the font covers.
//! Missing glyphs draw the font's `.notdef`. There is no fallback face: constructing a
//! [`TextRenderer`] needs a usable font, else the backend keeps `text: None` and draws
//! no HUD/OSD/notify text.
//!
//! Reuses the backend's unit-quad VAO and `BLIT_VS` (position via `u_rect`/`u_screen`),
//! so [`TextRenderer::draw`] assumes the caller has that VAO bound and premultiplied
//! blending enabled — exactly the state inside `GlBackend::present_windows`.
//!
//! Not implemented (by design — the "lean" path): complex shaping (ligatures / BiDi /
//! cluster shaping), colour-emoji (COLR/bitmap) glyphs, and hinting (`fontdue` is
//! unhinted). Layout is one glyph per codepoint, left to right, by advance width.
//! Requested sizes are rounded to the nearest integer px (the cache key), so a
//! continuously-animating size steps through 1px increments — imperceptible at the
//! sizes ricom uses.

use std::cell::Cell;
use std::cell::RefCell;
use std::collections::HashMap;

use anyhow::{anyhow, Result};
use fontdue::{Font, FontSettings};
use glow::HasContext as _;

use crate::{make_program, BLIT_VS};

/// Fragment shader: sample the glyph's coverage from the atlas sub-rect and emit it as
/// premultiplied alpha in `u_color`. Native antialiasing lives in the rasterised
/// coverage itself, so — unlike the old SDF path — there's no threshold/`fwidth` step.
const GLYPH_FS: &str = r#"#version 330 core
in vec2 v_tex;
uniform sampler2D u_atlas;
uniform vec4 u_uv;      // atlas sub-rect: xy = origin, zw = size (0..1)
uniform vec4 u_color;   // straight RGBA
out vec4 frag;
void main() {
    float cov = texture(u_atlas, u_uv.xy + v_tex * u_uv.zw).r; // 0..1 coverage
    float a = cov * u_color.a;
    frag = vec4(u_color.rgb * a, a); // premultiplied, matches ONE/1-SRC_ALPHA blend
}
"#;

/// 1px transparent border around each glyph so LINEAR sampling at a quad edge fades to
/// zero rather than catching a neighbour.
const PAD: usize = 1;
/// 1px gutter between packed glyphs (belt-and-braces with `PAD` against bleed).
const GUTTER: i32 = 1;
/// Initial atlas edge (px); grows (doubling, cache re-packed) up to `ATLAS_MAX`.
const ATLAS_START: i32 = 1024;
const ATLAS_MAX: i32 = 4096;
/// Clamp on the rasterised glyph pixel size (sanity bound on the cache key + atlas use).
const MAX_PX: f32 = 512.0;

/// Eight unit offsets (axis + diagonal) used to lay down an all-around text outline.
const OUTLINE_OFFSETS: [(f32, f32); 8] = [
    (-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0),
    (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0),
];

/// Text decoration for [`TextRenderer::draw_styled`]: an all-around outline and/or an
/// offset drop-shadow, both in screen px (`0` disables that layer). Colours are straight
/// RGBA (alpha lets the caller fade the whole run).
///
/// A1 renders these as multi-pass offset blits; A2 will render the outline in-shader —
/// the field set + call sites stay identical, so that swap is internal to this module.
#[derive(Clone, Copy)]
pub struct TextStyle {
    pub outline_px: f32,
    pub outline_color: [f32; 4],
    /// Drop-shadow offset (down-right) in px; `0` = no shadow.
    pub shadow_px: f32,
    pub shadow_color: [f32; 4],
}

impl TextStyle {
    /// No decoration — `draw_styled` then behaves exactly like `draw`.
    pub const NONE: TextStyle =
        TextStyle { outline_px: 0.0, outline_color: [0.0; 4], shadow_px: 0.0, shadow_color: [0.0; 4] };
}

/// One cached glyph's placement in the atlas + the geometry to position its quad, all in
/// **actual pixels at that glyph's size** (no scale factor — the raster is native).
#[derive(Clone, Copy)]
struct AtlasGlyph {
    /// Atlas sub-rect (0..1): xy = origin, zw = size, of the padded coverage region.
    uv: [f32; 4],
    /// Padded-region left edge relative to the pen (px).
    off_x: f32,
    /// Padded-region top edge relative to the line's top-left y (px).
    off_y: f32,
    /// Padded region size (px).
    pw: f32,
    ph: f32,
}

/// Simple shelf packer cursor over the atlas texture.
#[derive(Default, Clone, Copy)]
struct Shelf {
    x: i32,
    y: i32,
    row_h: i32,
}

/// Draws strings from a per-size native glyph cache. Owns its GLSL program, the font,
/// the atlas texture, and the (interior-mutable) caches + packer, so `draw`/`measure`
/// can take `&self` (as `present_windows` requires).
pub struct TextRenderer {
    program: glow::NativeProgram,
    atlas: glow::NativeTexture,
    atlas_w: Cell<i32>,
    atlas_h: Cell<i32>,
    u_screen: Option<glow::NativeUniformLocation>,
    u_rect: Option<glow::NativeUniformLocation>,
    u_uv: Option<glow::NativeUniformLocation>,
    u_color: Option<glow::NativeUniformLocation>,
    u_atlas: Option<glow::NativeUniformLocation>,
    font: Font,
    /// Drawable glyphs keyed by (char, size_px) → placement; `None` = a blank glyph.
    glyphs: RefCell<HashMap<(char, u32), Option<AtlasGlyph>>>,
    /// Advance widths (px) keyed by (char, size_px); used by `measure` without GL.
    advances: RefCell<HashMap<(char, u32), f32>>,
    /// (ascent, line_height) in px keyed by size_px.
    lines: RefCell<HashMap<u32, (f32, f32)>>,
    shelf: RefCell<Shelf>,
}

/// Round a requested (possibly fractional) size to the integer cache key, clamped.
fn px_key(px: f32) -> u32 {
    px.round().clamp(1.0, MAX_PX) as u32
}

impl TextRenderer {
    /// Compile the coverage program, parse `font_bytes` (a `.ttf`), and allocate an empty
    /// atlas texture. Requires a current GL context (the caller is inside the backend's
    /// context). Errors if the font can't be parsed or GL objects can't be created — the
    /// backend then leaves `text` as `None` (text disabled).
    pub fn new(gl: &glow::Context, font_bytes: &[u8]) -> Result<Self> {
        let font = Font::from_bytes(font_bytes, FontSettings::default())
            .map_err(|e| anyhow!("parse font: {e}"))?;

        let program = make_program(gl, BLIT_VS, GLYPH_FS)?;
        let (atlas, u_screen, u_rect, u_uv, u_color, u_atlas) = unsafe {
            let atlas = gl.create_texture().map_err(|e| anyhow!("text atlas texture: {e}"))?;
            gl.bind_texture(glow::TEXTURE_2D, Some(atlas));
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1); // single-channel rows
            // Zero-initialise so unwritten atlas regions sample as "no coverage".
            let zeros = vec![0u8; (ATLAS_START * ATLAS_START) as usize];
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::R8 as i32, ATLAS_START, ATLAS_START, 0,
                glow::RED, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(Some(&zeros)),
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            gl.bind_texture(glow::TEXTURE_2D, None);
            (
                atlas,
                gl.get_uniform_location(program, "u_screen"),
                gl.get_uniform_location(program, "u_rect"),
                gl.get_uniform_location(program, "u_uv"),
                gl.get_uniform_location(program, "u_color"),
                gl.get_uniform_location(program, "u_atlas"),
            )
        };
        Ok(TextRenderer {
            program,
            atlas,
            atlas_w: Cell::new(ATLAS_START),
            atlas_h: Cell::new(ATLAS_START),
            u_screen, u_rect, u_uv, u_color, u_atlas,
            font,
            glyphs: RefCell::new(HashMap::new()),
            advances: RefCell::new(HashMap::new()),
            lines: RefCell::new(HashMap::new()),
            shelf: RefCell::new(Shelf::default()),
        })
    }

    /// Delete this renderer's GL objects (program + atlas texture). Called when the
    /// font is swapped on reload so the old objects don't leak.
    pub fn destroy(self, gl: &glow::Context) {
        unsafe {
            gl.delete_program(self.program);
            gl.delete_texture(self.atlas);
        }
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

    /// Advance width of `ch` at integer size `size` in px (cached). No GL / rasterisation.
    fn advance(&self, ch: char, size: u32) -> f32 {
        if let Some(a) = self.advances.borrow().get(&(ch, size)) {
            return *a;
        }
        let a = self.font.metrics(ch, size as f32).advance_width;
        self.advances.borrow_mut().insert((ch, size), a);
        a
    }

    /// On-screen size of `s` at glyph height `px`: `(width, line_height)` in pixels.
    /// Proportional; measured at the same rounded size `draw` uses, so layout matches.
    pub fn measure(&self, px: f32, s: &str) -> (f32, f32) {
        let sz = px_key(px);
        let w: f32 = s.chars().map(|c| self.advance(c, sz)).sum();
        (w, self.line_metrics(sz).1)
    }

    /// The line height at glyph size `px` (pixels) — one text row's vertical pitch.
    pub fn line_height(&self, px: f32) -> f32 {
        self.line_metrics(px_key(px)).1
    }

    /// Fetch a glyph's atlas placement at integer size `size`, rasterising + packing it
    /// on first use. `None` for a blank glyph (e.g. space) or if it can't be atlased.
    fn glyph(&self, gl: &glow::Context, ch: char, size: u32) -> Option<AtlasGlyph> {
        if let Some(entry) = self.glyphs.borrow().get(&(ch, size)) {
            return *entry;
        }
        let (m, cov) = self.font.rasterize(ch, size as f32);
        let entry = if m.width == 0 || m.height == 0 {
            None // blank (space / zero-area glyph): advance only, nothing to draw
        } else {
            self.insert_glyph(gl, size, m.xmin, m.ymin, m.width, m.height, &cov)
        };
        self.glyphs.borrow_mut().insert((ch, size), entry);
        entry
    }

    /// Pad one rasterised glyph's coverage, pack it into the atlas, upload it, and return
    /// its placement (all in px at `size`). `None` if the atlas is full and can't grow.
    #[allow(clippy::too_many_arguments)]
    fn insert_glyph(
        &self,
        gl: &glow::Context,
        size: u32,
        xmin: i32,
        ymin: i32,
        gw: usize,
        gh: usize,
        cov: &[u8],
    ) -> Option<AtlasGlyph> {
        let (pw, ph) = (gw + 2 * PAD, gh + 2 * PAD);
        // Native coverage, centred in a padded buffer (transparent border).
        let mut buf = vec![0u8; pw * ph];
        for row in 0..gh {
            let dst = (row + PAD) * pw + PAD;
            let src = row * gw;
            buf[dst..dst + gw].copy_from_slice(&cov[src..src + gw]);
        }

        let (ax, ay) = self.pack(gl, pw as i32, ph as i32)?;
        let (aw, ah) = (self.atlas_w.get() as f32, self.atlas_h.get() as f32);
        unsafe {
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.atlas));
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D, 0, ax, ay, pw as i32, ph as i32,
                glow::RED, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(Some(&buf)),
            );
        }
        let ascent = self.line_metrics(size).0;
        Some(AtlasGlyph {
            uv: [ax as f32 / aw, ay as f32 / ah, pw as f32 / aw, ph as f32 / ah],
            off_x: xmin as f32 - PAD as f32,
            off_y: ascent - (ymin + gh as i32) as f32 - PAD as f32,
            pw: pw as f32,
            ph: ph as f32,
        })
    }

    /// Reserve a `w×h` slot in the atlas (shelf packing), growing the texture if needed.
    /// Returns the top-left atlas coords, or `None` if it can't fit at max size.
    fn pack(&self, gl: &glow::Context, w: i32, h: i32) -> Option<(i32, i32)> {
        loop {
            let (aw, ah) = (self.atlas_w.get(), self.atlas_h.get());
            {
                let mut sh = self.shelf.borrow_mut();
                if sh.x + w + GUTTER > aw {
                    sh.x = 0;
                    sh.y += sh.row_h + GUTTER;
                    sh.row_h = 0;
                }
                if sh.y + h + GUTTER <= ah {
                    let pos = (sh.x, sh.y);
                    sh.x += w + GUTTER;
                    sh.row_h = sh.row_h.max(h);
                    return Some(pos);
                }
            }
            // Out of vertical room — grow (resets the packer + clears the glyph cache,
            // since old placements are invalidated) and retry.
            if !self.grow(gl) {
                tracing::warn!("glyph atlas full at {ATLAS_MAX}px — dropping glyph");
                return None;
            }
        }
    }

    /// Double the atlas (capped at `ATLAS_MAX`), re-specifying the texture and clearing
    /// the glyph cache + packer (advances/line-metrics stay valid — atlas-independent).
    /// Returns `false` if already at the cap.
    fn grow(&self, gl: &glow::Context) -> bool {
        let (aw, ah) = (self.atlas_w.get(), self.atlas_h.get());
        if aw >= ATLAS_MAX && ah >= ATLAS_MAX {
            return false;
        }
        let (nw, nh) = ((aw * 2).min(ATLAS_MAX), (ah * 2).min(ATLAS_MAX));
        unsafe {
            let zeros = vec![0u8; (nw * nh) as usize];
            gl.bind_texture(glow::TEXTURE_2D, Some(self.atlas));
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::R8 as i32, nw, nh, 0,
                glow::RED, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(Some(&zeros)),
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
        }
        self.atlas_w.set(nw);
        self.atlas_h.set(nh);
        *self.shelf.borrow_mut() = Shelf::default();
        self.glyphs.borrow_mut().clear();
        tracing::debug!(nw, nh, "glyph atlas grown");
        true
    }

    /// Draw `s` with its top-left at (`x`, `y`), text `px` tall, in `color` (straight
    /// RGBA). Assumes the unit-quad VAO is bound and premultiplied-alpha blending is
    /// enabled (as in `present_windows`). The requested `px` is rounded to the nearest
    /// integer for rasterisation + layout, so glyphs draw at their native pixel size.
    /// Glyphs are rasterised + atlased on first use; missing/blank glyphs advance the pen.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        gl: &glow::Context,
        screen_w: i32,
        screen_h: i32,
        x: f32,
        y: f32,
        px: f32,
        color: [f32; 4],
        s: &str,
    ) {
        let sz = px_key(px);
        unsafe {
            gl.use_program(Some(self.program));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.atlas));
            gl.uniform_1_i32(self.u_atlas.as_ref(), 0);
            gl.uniform_2_f32(self.u_screen.as_ref(), screen_w as f32, screen_h as f32);
            gl.uniform_4_f32(self.u_color.as_ref(), color[0], color[1], color[2], color[3]);
            let mut pen = x;
            for ch in s.chars() {
                let adv = self.advance(ch, sz);
                if let Some(g) = self.glyph(gl, ch, sz) {
                    // A grow() during glyph() may have re-specced the atlas texture;
                    // re-bind so this draw samples the current one.
                    gl.bind_texture(glow::TEXTURE_2D, Some(self.atlas));
                    gl.uniform_4_f32(self.u_uv.as_ref(), g.uv[0], g.uv[1], g.uv[2], g.uv[3]);
                    gl.uniform_4_f32(self.u_rect.as_ref(), pen + g.off_x, y + g.off_y, g.pw, g.ph);
                    gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
                }
                pen += adv;
            }
        }
    }

    /// Draw `s` at (`x`, `y`) with an optional outline + drop-shadow (see [`TextStyle`]);
    /// `fill` is the main glyph colour. Layered back-to-front: shadow, outline, fill.
    /// `TextStyle::NONE` ⇒ a plain `draw`. Same VAO / premultiplied-blend assumptions.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_styled(
        &self,
        gl: &glow::Context,
        screen_w: i32,
        screen_h: i32,
        x: f32,
        y: f32,
        px: f32,
        fill: [f32; 4],
        style: &TextStyle,
        s: &str,
    ) {
        // Drop-shadow behind everything (one offset pass).
        if style.shadow_px > 0.0 && style.shadow_color[3] > 0.0 {
            self.draw(gl, screen_w, screen_h, x + style.shadow_px, y + style.shadow_px, px, style.shadow_color, s);
        }
        // All-around outline: eight offset passes at `outline_px`.
        if style.outline_px > 0.0 && style.outline_color[3] > 0.0 {
            for (dx, dy) in OUTLINE_OFFSETS {
                self.draw(gl, screen_w, screen_h, x + dx * style.outline_px, y + dy * style.outline_px, px, style.outline_color, s);
            }
        }
        // Fill on top.
        self.draw(gl, screen_w, screen_h, x, y, px, fill, s);
    }
}
