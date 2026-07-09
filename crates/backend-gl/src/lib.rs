//! GL (EGL) rendering backend. Mirrors picom's `src/backend/gl/egl.c`.
//!
//! - [`first_light`]: headless pbuffer smoke test (validates EGL + glow + GL).
//! - [`GlBackend`]: an EGL **window** surface on the composite overlay, with a
//!   textured-quad blit program and [`GlBackend::present_window_pixmap`] —
//!   bind an X window's pixmap as a GL texture (EGLImage) and draw it. This is
//!   the heart of compositing; the renderer drives it over the window stack.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::ffi::c_void;

use anyhow::{anyhow, bail, Result};
use glow::HasContext as _;
use khronos_egl as egl;

mod text;

/// `glEGLImageTargetTexture2DOES(target, image)` — loaded via eglGetProcAddress.
type ImageTargetTexture2DOes = unsafe extern "system" fn(target: u32, image: *const c_void);

/// `EGL_NATIVE_PIXMAP_KHR` (from EGL_KHR_image_pixmap; not exported by khronos-egl).
const EGL_NATIVE_PIXMAP_KHR: egl::Enum = 0x30B0;
/// `EGL_BUFFER_AGE_EXT` (from EGL_EXT_buffer_age) — queried per frame for partial repaint.
const EGL_BUFFER_AGE_EXT: egl::Int = 0x313D;

/// Re-exported so callers can build clip/clear rects without a direct `region` dep.
pub use region::Rect;

/// How many recent per-composite render times the HUD graph keeps.
const HUD_GRAPH_SAMPLES: usize = 120;

const BLIT_VS: &str = r#"#version 330 core
layout(location = 0) in vec2 a_pos;   // unit quad, 0..1
uniform vec4 u_rect;                   // x, y, w, h  (pixels, top-left origin)
uniform vec2 u_screen;                 // screen w, h
out vec2 v_tex;
void main() {
    v_tex = a_pos;                                  // (0,0) = window top-left
    vec2 px = u_rect.xy + a_pos * u_rect.zw;        // pixel position
    vec2 ndc = vec2(px.x / u_screen.x * 2.0 - 1.0,
                    1.0 - px.y / u_screen.y * 2.0); // flip Y into GL NDC
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"#;

const BLIT_FS: &str = r#"#version 330 core
in vec2 v_tex;
uniform sampler2D u_tex;
uniform float u_opacity;              // whole-window opacity, 0..1
uniform vec4 u_rect;                  // window rect x,y,w,h (px) — shared with the vertex shader
uniform float u_corner;              // corner radius (px); 0 = square
out vec4 frag;
// Premultiplied-alpha output: paired with glBlendFunc(ONE, ONE_MINUS_SRC_ALPHA)
// over an opaque clear this yields  dst = rgb*a + dst*(1-a)  — straight "over".
void main() {
    float a = u_opacity;
    if (u_corner > 0.0) {
        // Rounded-box mask: fade alpha to 0 outside the rounded rectangle so the
        // corners reveal what's beneath. `d` is the signed distance outside it.
        vec2 hs = u_rect.zw * 0.5;
        float r = min(u_corner, min(hs.x, hs.y));
        vec2 p = abs(v_tex * u_rect.zw - hs);
        float d = length(max(p - (hs - r), vec2(0.0))) - r;
        float aa = fwidth(d);
        a *= 1.0 - smoothstep(-aa, aa, d);     // derivative-based AA (~2px), like the SDF text
    }
    frag = vec4(texture(u_tex, v_tex).rgb * a, a);
}
"#;

/// Spin vertex shader: like [`BLIT_VS`] but rotates the quad about the window
/// centre by `u_angle` radians **in pixel space** (before the NDC transform, so a
/// non-square window doesn't shear). Paired with [`BLIT_FS`].
const SPIN_VS: &str = r#"#version 330 core
layout(location = 0) in vec2 a_pos;   // unit quad, 0..1
uniform vec4 u_rect;                   // x, y, w, h  (pixels, top-left origin)
uniform vec2 u_screen;                 // screen w, h
uniform float u_angle;                 // rotation about the centre (radians)
out vec2 v_tex;
void main() {
    v_tex = a_pos;
    vec2 px = u_rect.xy + a_pos * u_rect.zw;          // pixel position
    vec2 c = u_rect.xy + u_rect.zw * 0.5;             // window centre (px)
    float s = sin(u_angle), co = cos(u_angle);
    vec2 rel = px - c;
    px = c + vec2(rel.x * co - rel.y * s, rel.x * s + rel.y * co); // rotate in px space
    vec2 ndc = vec2(px.x / u_screen.x * 2.0 - 1.0,
                    1.0 - px.y / u_screen.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"#;

/// Radial water-refraction ripple (reuses [`BLIT_VS`]: `u_rect`/`u_screen`/`v_tex`).
/// Per-pixel it offsets the sampled UV radially from `u_center` by `sin` of the
/// distance `r`; amplitude spreads down with radius (`u_r0`) so the rings are large
/// at the centre and die toward the rim, and `u_phase` advances to expand them out.
/// Aspect-corrected via `u_rect` so rings stay circular. Premultiplied output.
const RIPPLE_FS: &str = r#"#version 330 core
in vec2 v_tex;
uniform sampler2D u_tex;
uniform float u_opacity;
uniform vec4  u_rect;        // x,y,w,h (px) — aspect only
uniform vec2  u_center;      // ripple centre in UV (0..1)
uniform float u_amp;         // peak radial UV displacement (aspect-corrected)
uniform float u_wavelength;  // ring spacing (aspect-corrected radius)
uniform float u_phase;       // advances -> rings expand outward
uniform float u_r0;          // spread constant (big centre, faint rim)
out vec4 frag;
void main() {
    float aspect = u_rect.z / max(u_rect.w, 1.0);
    vec2  d = v_tex - u_center; d.x *= aspect;          // circular rings
    float r = length(d);
    vec2  dir = r > 1e-4 ? d / r : vec2(0.0);
    float env  = u_amp * (u_r0 / (u_r0 + r));           // large centre -> dies at rim
    float disp = env * sin(6.2831853 * (r / max(u_wavelength, 1e-3) - u_phase));
    vec2  off = dir * disp; off.x /= aspect;            // back into UV space
    vec2  uv  = clamp(v_tex + off, 0.0, 1.0);
    frag = vec4(texture(u_tex, uv).rgb * u_opacity, u_opacity);
}
"#;

/// Traveling-wave content refraction (reuses [`BLIT_VS`]: `u_rect`/`u_screen`/`v_tex`).
/// The per-pixel replacement for the old mesh `wave`: a sine crest travels along
/// `u_axis` (0 = X, displaces V; 1 = Y, displaces U), offsetting the sampled UV
/// perpendicular to travel by `u_amp` (UV) — smooth at any amplitude, where the
/// 16×16 grid faceted. `u_phase` advances so the crest sweeps. Premultiplied output.
const WAVE_FS: &str = r#"#version 330 core
in vec2 v_tex;
uniform sampler2D u_tex;
uniform float u_opacity;
uniform float u_amp;         // peak perpendicular UV displacement
uniform float u_wavelength;  // fraction of the travel axis (1.0 = one cycle across)
uniform float u_phase;       // advances -> crest travels
uniform int   u_axis;        // 0 = travel along X (displaces V); 1 = along Y (displaces U)
out vec4 frag;
void main() {
    float along = (u_axis == 1) ? v_tex.y : v_tex.x;
    float off = u_amp * sin(6.2831853 * (along / max(u_wavelength, 1e-3) - u_phase));
    vec2 uv = v_tex;
    if (u_axis == 1) { uv.x += off; } else { uv.y += off; }
    uv = clamp(uv, 0.0, 1.0);
    frag = vec4(texture(u_tex, uv).rgb * u_opacity, u_opacity);
}
"#;

/// Drain / whirlpool close (reuses [`BLIT_VS`]: `u_rect`/`u_screen`/`v_tex`). The
/// window's content spirals into a vanishing point at `u_center` and fades: per pixel,
/// rotate the sampled UV about the centre by an angle that is **strong near the sink and
/// tapers to the rim** (a `1/(1+k·r)` falloff — a real vortex shears, it doesn't spin as
/// a rigid disc), scaled by `u_turns · u_progress`, then sample from a disk that shrinks
/// as `1 − u_progress`, so content compresses to a point; pixels whose source lands
/// outside the window read as transparent (drained away). `u_progress` runs 0 → 1.
/// Premultiplied output.
const DRAIN_FS: &str = r#"#version 330 core
in vec2 v_tex;
uniform sampler2D u_tex;
uniform float u_opacity;
uniform vec4  u_rect;       // x,y,w,h (px) — aspect only
uniform vec2  u_center;     // drain centre in UV (0..1)
uniform float u_progress;   // 0 intact .. 1 fully drained
uniform float u_turns;      // swirl rotations at full progress
uniform float u_seed;       // per-window seed → each drain's turbulence differs
out vec4 frag;
// Smooth value noise (hash + bilinear) for turbulent, per-region rotation rates.
float dhash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float dnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(dhash(i), dhash(i + vec2(1.0, 0.0)), u.x),
               mix(dhash(i + vec2(0.0, 1.0)), dhash(i + vec2(1.0, 1.0)), u.x), u.y);
}
void main() {
    float aspect = u_rect.z / max(u_rect.w, 1.0);
    vec2  rel = v_tex - u_center; rel.x *= aspect;      // aspect-correct for a round swirl
    float r = length(rel);                              // distance from the sink (centre)
    // Vortex shear: angular speed is high near the sink and tapers outward, so the
    // content spirals rather than spinning as a rigid disc. 1/(1 + k·r) falloff.
    float swirl = 1.0 / (1.0 + 6.0 * r);
    // Turbulence: a smooth seeded noise field varies the rotation RATE across the
    // vortex (uneven arms), while swirl(r) keeps the slower-outer/faster-inner curve.
    float n = dnoise(rel * 6.0 + vec2(u_seed * 3.1, u_seed * 1.7)) * 2.0 - 1.0; // ~[-1,1]
    float theta = 6.2831853 * u_turns * u_progress * swirl * (1.0 + 0.45 * n);
    float c = cos(theta), s = sin(theta);
    vec2  rot = vec2(rel.x * c - rel.y * s, rel.x * s + rel.y * c);
    float scale = max(1.0 - u_progress, 1e-3);          // visible disk shrinks to a point
    vec2  src = u_center + vec2(rot.x / aspect, rot.y) / scale;
    float inb = step(0.0, src.x) * step(src.x, 1.0) * step(0.0, src.y) * step(src.y, 1.0);
    float a = u_opacity * (1.0 - u_progress) * inb;     // fade out as it shrinks away
    frag = vec4(texture(u_tex, clamp(src, 0.0, 1.0)).rgb * a, a);
}
"#;

/// Burn / dissolve close animation (reuses [`BLIT_VS`], so it has `u_rect`/
/// `u_screen`/`v_tex`). The window erodes away on animated value-noise with a
/// glowing ember band at the dissolving front — "uniform segments": chunky
/// patches ignite all over (no directional axis). `u_progress` marches the
/// threshold 0→1; `u_seed` de-correlates each window so no two burns match.
/// `u_segscale`/`u_ember` are the live-tunable size knobs (from `[burn]` config,
/// via [`RenderParams`]): patch granularity and ember-band half-width.
/// Output is premultiplied for the same ONE/ONE_MINUS_SRC_ALPHA blend as the blit.
const BURN_FS: &str = r#"#version 330 core
in vec2 v_tex;
uniform sampler2D u_tex;
uniform float u_opacity;   // whole-window opacity, 0..1
uniform float u_progress;  // burn progress, 0 (intact) .. 1 (fully gone)
uniform float u_seed;      // per-window offset so each burn differs
uniform float u_segscale;  // segment/hole granularity (higher = finer patches)
uniform float u_ember;     // ember / transition band half-width (smaller = thinner glow)
uniform vec3 u_ember_cool; // cooler trailing ember colour (em→0)
uniform vec3 u_ember_hot;  // hottest leading-edge ember colour (em→1)
out vec4 frag;
// Cheap hash value-noise + 4-octave fbm (no textures/assets).
float hash(vec2 p) { p = fract(p * vec2(127.31, 311.7)); p += dot(p, p + 34.23); return fract(p.x * p.y); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash(i), b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0)), d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbm(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { s += a * vnoise(p); p *= 2.0; a *= 0.5; } return s; }
void main() {
    vec2 uv = v_tex + vec2(u_seed, u_seed * 1.7);
    float nHi  = fbm(uv * u_segscale);          // ragged hole detail
    float nLo  = fbm(uv * u_segscale * 0.25);   // chunky segments + wavy front
    float bias = 0.6 * nLo;                     // segments: no positional axis
    float thr  = mix(-u_ember, 1.0 + u_ember, u_progress) - bias;
    float dis  = smoothstep(thr - u_ember, thr, nHi);              // 1 intact .. 0 gone
    float em   = clamp(1.0 - abs(nHi - thr) / u_ember, 0.0, 1.0);  // hot band at the front
    float a    = dis * u_opacity;                               // premultiplied coverage
    vec3 win   = texture(u_tex, v_tex).rgb;
    vec3 ember = mix(u_ember_cool, u_ember_hot, em); // cool -> hot ember ramp (config)
    frag = vec4(win * a + ember * em, a);      // window + glowing ember (premultiplied)
}
"#;

/// Wobble mesh grid dimension (`N×N` control points). **Must match**
/// `wm::anim::WOBBLE_N`: `session` builds the deformed vertex grid there and the
/// backend builds the matching triangle index buffer here. `(N-1)²·2` triangles.
const MESH_N: usize = 16;

/// Wobbly-windows mesh vertex shader: positions come straight from the CPU spring
/// sim (already in screen pixels), so — unlike [`BLIT_VS`] — there's no `u_rect`;
/// each vertex carries its own position + UV. Same top-left→NDC Y-flip as the blit.
const MESH_VS: &str = r#"#version 330 core
layout(location = 0) in vec2 a_px;    // control-point position (px, top-left origin)
layout(location = 1) in vec2 a_uv;    // texture coord, (0,0) = window top-left
uniform vec2 u_screen;                // screen w, h
out vec2 v_tex;
void main() {
    v_tex = a_uv;
    vec2 ndc = vec2(a_px.x / u_screen.x * 2.0 - 1.0,
                    1.0 - a_px.y / u_screen.y * 2.0); // flip Y into GL NDC
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"#;

/// Wobbly-windows mesh fragment shader: sample the window texture × opacity,
/// premultiplied for the ONE/ONE_MINUS_SRC_ALPHA blend. No corner rounding — a
/// window is square while it wobbles (rounding returns via the quad path once it
/// settles).
const MESH_FS: &str = r#"#version 330 core
in vec2 v_tex;
uniform sampler2D u_tex;
uniform float u_opacity;   // whole-window opacity, 0..1
out vec4 frag;
void main() {
    float a = u_opacity;
    frag = vec4(texture(u_tex, v_tex).rgb * a, a);
}
"#;

/// Soft drop-shadow fragment shader (reuses [`BLIT_VS`], so it also has `u_rect`
/// = the shadow quad and `u_screen`). Casts only to the **left** and **bottom**
/// of the window (light from the top-right), fading off over the radius with a
/// rounded bottom-left corner. Output is premultiplied black for the same
/// ONE/ONE_MINUS_SRC_ALPHA blend as the blit.
const SHADOW_FS: &str = r#"#version 330 core
in vec2 v_tex;
uniform vec4 u_rect;     // shadow quad: x, y, w, h  (px, top-left origin)
uniform vec4 u_inner;    // caster window rect: x, y, w, h  (px)
uniform vec2 u_shadow;   // x = radius (falloff px), y = strength (max alpha)
uniform float u_scorner; // window corner radius (px) — match the blit's rounding
out vec4 frag;
void main() {
    vec2 p  = u_rect.xy + v_tex * u_rect.zw;   // fragment pixel position
    vec2 lo = u_inner.xy;
    vec2 hi = u_inner.xy + u_inner.zw;
    float r  = u_shadow.x;
    float cr = min(u_scorner, min(u_inner.z, u_inner.w) * 0.5); // clamped corner radius
    float t  = max(cr, r); // where a band ends at the top-left / bottom-right:
                           // the corner bend when rounded, else a soft taper by r
    float dist = 1e9;
    // Left edge segment: ends at the top-left corner bend and at the bottom-left
    // arc — cast only to the left.
    if (p.x <= lo.x) {
        float cy = clamp(p.y, lo.y + t, hi.y - cr);
        dist = min(dist, length(vec2(lo.x - p.x, p.y - cy)));
    }
    // Bottom edge segment: starts after the bottom-left arc, ends at the
    // bottom-right corner bend — cast only below.
    if (p.y >= hi.y) {
        float cx = clamp(p.x, lo.x + cr, hi.x - t);
        dist = min(dist, length(vec2(p.x - cx, p.y - hi.y)));
    }
    // Bottom-left corner: hug the window's rounded corner (arc of radius cr),
    // so the shadow follows it instead of the square corner.
    vec2 cc = vec2(lo.x + cr, hi.y - cr);
    if (p.x <= cc.x && p.y >= cc.y) {
        dist = min(dist, max(length(p - cc) - cr, 0.0));
    }
    float a = u_shadow.y * (1.0 - smoothstep(0.0, r, dist));
    frag = vec4(0.0, 0.0, 0.0, a);
}
"#;

// --- Background blur (dual-Kawase): downsample + upsample pyramid ---------------
//
// The backdrop under a translucent window is copied into a full-res FBO texture,
// then blurred by repeatedly downsampling (5-tap) and upsampling (8-tap) through a
// half-res pyramid — the standard efficient compositor blur. Both passes render a
// screen-filling quad, so this vertex shader just maps the 0..1 unit quad straight
// to NDC and hands the fragment shader a 0..1 UV.
const BLUR_VS: &str = r#"#version 330 core
layout(location = 0) in vec2 a_pos;   // unit quad, 0..1
out vec2 v_uv;
void main() {
    v_uv = a_pos;
    gl_Position = vec4(a_pos * 2.0 - 1.0, 0.0, 1.0);
}
"#;

/// Dual-Kawase downsample (5-tap): sample the larger source into the half-size
/// target. `u_halfpixel` is 0.5/source-size; `u_offset` scales the blur reach.
const DOWN_FS: &str = r#"#version 330 core
in vec2 v_uv;
uniform sampler2D u_src;
uniform vec2 u_halfpixel;
uniform float u_offset;
out vec4 frag;
void main() {
    vec2 o = u_halfpixel * u_offset;
    vec4 s = texture(u_src, v_uv) * 4.0;
    s += texture(u_src, v_uv - o);
    s += texture(u_src, v_uv + o);
    s += texture(u_src, v_uv + vec2(o.x, -o.y));
    s += texture(u_src, v_uv - vec2(o.x, -o.y));
    frag = s / 8.0;
}
"#;

/// Dual-Kawase upsample (8-tap): sample the smaller source back up into the
/// larger target. `u_halfpixel` is 0.5/source-size.
const UP_FS: &str = r#"#version 330 core
in vec2 v_uv;
uniform sampler2D u_src;
uniform vec2 u_halfpixel;
uniform float u_offset;
out vec4 frag;
void main() {
    vec2 o = u_halfpixel * u_offset;
    vec4 s = texture(u_src, v_uv + vec2(-o.x * 2.0, 0.0));
    s += texture(u_src, v_uv + vec2(-o.x, o.y)) * 2.0;
    s += texture(u_src, v_uv + vec2(0.0, o.y * 2.0));
    s += texture(u_src, v_uv + vec2(o.x, o.y)) * 2.0;
    s += texture(u_src, v_uv + vec2(o.x * 2.0, 0.0));
    s += texture(u_src, v_uv + vec2(o.x, -o.y)) * 2.0;
    s += texture(u_src, v_uv + vec2(0.0, -o.y * 2.0));
    s += texture(u_src, v_uv + vec2(-o.x, -o.y)) * 2.0;
    frag = s / 12.0;
}
"#;

/// Draw the blurred backdrop into a window's rect. Reuses [`BLIT_VS`] to position
/// the quad (so it has `u_rect`/`u_screen`/`v_tex`), but samples the *full-screen*
/// blurred texture by `gl_FragCoord` — which shares the framebuffer's bottom-left
/// origin, so no manual Y-flip is needed. Masked to the same rounded rect as the
/// window and emitted opaque (premultiplied) so the window blends on top.
const FROST_FS: &str = r#"#version 330 core
in vec2 v_tex;
uniform sampler2D u_tex;   // full-screen blurred backdrop
uniform vec2 u_screen;
uniform vec4 u_rect;       // window rect x,y,w,h (px) — shared with BLIT_VS
uniform float u_corner;    // corner radius (px); 0 = square
out vec4 frag;
void main() {
    vec2 uv = gl_FragCoord.xy / u_screen;
    float a = 1.0;
    if (u_corner > 0.0) {
        vec2 hs = u_rect.zw * 0.5;
        float r = min(u_corner, min(hs.x, hs.y));
        vec2 p = abs(v_tex * u_rect.zw - hs);
        float d = length(max(p - (hs - r), vec2(0.0))) - r;
        float aa = fwidth(d);
        a *= 1.0 - smoothstep(-aa, aa, d);
    }
    frag = vec4(texture(u_tex, uv).rgb * a, a);
}
"#;

/// Number of pyramid levels allocated (level 0 = full res, level i = size >> i).
/// Bounds the usable `blur.passes` (down/up steps) at `MAX_BLUR_LEVELS - 1`.
const MAX_BLUR_LEVELS: i32 = 7;

/// Runtime render parameters (from the config file): set when the backend is
/// created and swapped in on config reload via [`GlBackend::set_render_params`].
/// Defaults reproduce the previously compiled-in constants.
#[derive(Debug, Clone, Copy)]
pub struct RenderParams {
    /// Drop-shadow falloff distance to the left/bottom (px).
    pub shadow_radius: f32,
    /// Peak shadow alpha.
    pub shadow_strength: f32,
    /// Composite background colour (RGB), shown where no window covers.
    pub background: [f32; 3],
    /// Window corner radius (px); `0.0` = square.
    pub corner_radius: f32,
    /// Background blur on/off (frost the backdrop behind translucent windows).
    pub blur_enabled: bool,
    /// Dual-Kawase iterations (down+up steps); clamped to `MAX_BLUR_LEVELS - 1`.
    pub blur_passes: i32,
    /// Blur sample offset per pass (px).
    pub blur_radius: f32,
    /// Burn/dissolve segment granularity (`u_segscale`): higher = finer patches.
    pub burn_seg_scale: f32,
    /// Burn/dissolve ember-band half-width (`u_ember`): smaller = thinner glow.
    pub burn_ember: f32,
    /// Cooler trailing ember colour (`u_ember_cool`, RGB).
    pub burn_ember_cool: [f32; 3],
    /// Hottest leading-edge ember colour (`u_ember_hot`, RGB).
    pub burn_ember_hot: [f32; 3],
}

impl Default for RenderParams {
    fn default() -> Self {
        RenderParams {
            shadow_radius: 12.0,
            shadow_strength: 0.45,
            background: [0.05, 0.05, 0.07],
            corner_radius: 0.0,
            blur_enabled: false,
            blur_passes: 3,
            blur_radius: 4.0,
            burn_seg_scale: 36.0,
            burn_ember: 0.07,
            burn_ember_cool: [0.28, 0.02, 0.0],
            burn_ember_hot: [0.75, 0.22, 0.04],
        }
    }
}

/// One window to composite: its named pixmap, on-screen rect (top-left origin,
/// pixels, border included), whole-window opacity (`0.0..=1.0`), and whether to
/// draw a drop shadow behind it.
#[derive(Debug, Clone, Copy)]
pub struct Quad {
    pub pixmap: u32,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub opacity: f32,
    pub shadow: bool,
    /// Frost the backdrop under this window (set for translucent windows when
    /// blur is enabled; ignored for opaque windows whose backdrop is hidden).
    pub blur: bool,
    /// Corner radius (px) for this window; `0.0` = square.
    pub corner_radius: f32,
}

/// Per-window burn/dissolve state for the close animation. When a [`WindowDraw`]
/// carries `Some`, the backend draws it through [`BURN_FS`] instead of the plain
/// blit (no shadow / frost / corner rounding, like the wobble-mesh path).
#[derive(Debug, Clone, Copy)]
pub struct Burn {
    /// `0.0` = intact … `1.0` = fully burnt away.
    pub progress: f32,
    /// Per-window random offset so no two windows burn with the same pattern.
    pub seed: f32,
}

/// Radial-ripple (water refraction) parameters for [`RIPPLE_FS`]. When a
/// [`WindowDraw`] carries `Some`, the backend draws it through the ripple program
/// (per-pixel UV warp; no shadow / frost / corner — like the mesh / spin paths).
#[derive(Debug, Clone, Copy)]
pub struct RippleParams {
    /// Ripple centre in UV (`[0.5, 0.5]` = window centre).
    pub center: [f32; 2],
    /// Peak radial UV displacement (aspect-corrected units).
    pub amp: f32,
    /// Ring spacing as a fraction of the aspect-corrected radius.
    pub wavelength: f32,
    /// Phase (cycles); advances so the rings expand outward.
    pub phase: f32,
    /// Spread constant — amplitude falls with radius (large centre, faint rim).
    pub r0: f32,
}

/// Traveling-wave (content refraction) parameters for [`WAVE_FS`]. When a
/// [`WindowDraw`] carries `Some`, the backend draws it through the wave program
/// (per-pixel UV warp; no shadow / frost / corner — like the ripple path). Replaces
/// the old mesh-based wave, so it's smooth at any amplitude.
#[derive(Debug, Clone, Copy)]
pub struct WaveParams {
    /// Peak perpendicular UV displacement.
    pub amp: f32,
    /// Wavelength as a fraction of the travel axis (`1.0` = one cycle across).
    pub wavelength: f32,
    /// Phase (cycles); advances so the crest travels.
    pub phase: f32,
    /// Travel axis: `0` = along X (displaces V), `1` = along Y (displaces U).
    pub axis: u32,
}

/// Drain / whirlpool close parameters for [`DRAIN_FS`]. When a [`WindowDraw`] carries
/// `Some`, the backend draws it through the drain program (per-pixel; no shadow / frost
/// / corner). A close driver like burn: `progress` 0→1 spirals the content into a
/// vanishing point at `center` and fades it out; the window is reaped at `1`.
#[derive(Debug, Clone, Copy)]
pub struct DrainParams {
    /// Drain centre in UV (`[0.5, 0.5]` = window centre).
    pub center: [f32; 2],
    /// Progress `0.0` (intact) → `1.0` (fully drained / gone).
    pub progress: f32,
    /// Swirl rotations at full progress.
    pub turns: f32,
    /// Per-window seed so each drain's rate-turbulence differs.
    pub seed: f32,
}

/// A window to composite plus the screen-space rectangles it's actually visible
/// in (region-level occlusion): [`GlBackend::present_windows`] scissors each of
/// the quad's draws to `clip`, so pixels covered by an opaque window on top are
/// never shaded. An empty `clip` is a fully-occluded window (callers omit those).
pub struct WindowDraw {
    pub quad: Quad,
    pub clip: Vec<region::Rect>,
    /// Wobble mesh: `MESH_N × MESH_N` deformed vertices as `[x_px, y_px, u, v]`,
    /// row-major (from `wm::anim::Wobble::vertices`). `Some` → draw the textured
    /// mesh (no shadow / frost / corner rounding — square while wobbling); `None`
    /// → the normal quad path. `quad.x/y/w/h` still give the un-deformed rect (for
    /// texture binding and, when settled, the quad path).
    pub mesh: Option<Vec<[f32; 4]>>,
    /// Burn/dissolve close effect. `Some` → draw via [`BURN_FS`] at this progress
    /// (mutually exclusive with `mesh`; a closing window doesn't wobble).
    pub burn: Option<Burn>,
    /// Rotation about the window centre (radians) for the `spin` primitive. `Some`
    /// → draw via [`SPIN_VS`] (no shadow/frost; corners suppressed), mutually
    /// exclusive with `mesh`/`burn`. `None` → the normal quad path.
    pub spin: Option<f32>,
    /// Radial water-refraction ripple. `Some` → draw via [`RIPPLE_FS`] (per-pixel UV
    /// warp; no shadow / frost / corner), mutually exclusive with `mesh`/`burn`/`spin`.
    pub ripple: Option<RippleParams>,
    /// Traveling wave (content refraction). `Some` → draw via [`WAVE_FS`] (per-pixel UV
    /// warp; no shadow / frost / corner), mutually exclusive with `mesh`/`burn`/`spin`/`ripple`.
    pub wave: Option<WaveParams>,
    /// Drain / whirlpool close. `Some` → draw via [`DRAIN_FS`] (per-pixel; no shadow /
    /// frost / corner), mutually exclusive with `mesh`/`burn`/`spin`/`ripple`/`wave`.
    pub drain: Option<DrainParams>,
}

impl WindowDraw {
    /// Draw `quad` in full — a single clip rect equal to its own bounds (no
    /// occlusion), no wobble mesh. Used by the diagnostic `--blit-test` /
    /// `--opacity-test` paths.
    pub fn whole(quad: Quad) -> Self {
        WindowDraw {
            quad,
            clip: vec![region::Rect::from_xywh(quad.x, quad.y, quad.w, quad.h)],
            mesh: None,
            burn: None,
            spin: None,
            ripple: None,
            wave: None,
            drain: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GlInfo {
    pub vendor: String,
    pub renderer: String,
    pub version: String,
}

/// Which screen corner the FPS HUD anchors to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HudCorner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// 1m/5m/15m compositor load averages for the HUD's load block: present rate
/// (fps) and mean GPU render time (ms; `None` for a window that had no frames —
/// idle or bypassed). Toggled independently of the numbers/graph.
pub struct HudLoad {
    pub fps: [f32; 3],
    pub render_ms: [Option<f32>; 3],
}

/// One frame's HUD data, drawn by [`GlBackend::present_windows`] when `Some`. The
/// graph itself is fed by the backend's own GPU render-time samples.
pub struct Hud {
    /// Present rate (frames composited in the last second).
    pub fps: u32,
    /// Draw the render-time graph beneath the numbers.
    pub graph: bool,
    /// Which screen corner to anchor to.
    pub corner: HudCorner,
    /// Extra size multiplier on top of the automatic screen-height scaling.
    pub scale: f32,
    /// Current display refresh rate (Hz) — one refresh interval is the render budget.
    pub refresh_hz: f32,
    /// Optional 1m/5m/15m load block, shown under the graph (`Super+Shift+L`).
    pub load: Option<HudLoad>,
}

/// How the OSD toast appears/disappears — the caller picks the open or close
/// effect per phase (see `session`). Mirrors `config::OsdEffect`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OsdEffect {
    Fade,
    Slide,
    Pop,
    Unroll,
    Stretch,
}

/// An on-screen notification banner ("toast"), drawn top-center over everything.
/// `presence` (0..1) drives the current `effect`: 0 = fully hidden, 1 = fully shown.
pub struct Osd {
    pub text: String,
    pub presence: f32,
    /// Size multiplier on top of the automatic screen-height scaling.
    pub scale: f32,
    /// The effect for this phase (open while appearing, close while disappearing).
    pub effect: OsdEffect,
    /// Text colour (RGB); alpha comes from the fade.
    pub color: [f32; 3],
    /// Banner background colour (RGBA); alpha `0.0` = text-only (no box drawn).
    pub background: [f32; 4],
    /// Draw a dark outline behind the glyphs (legibility without a box).
    pub outline: bool,
}

/// Solid-colour fill with optional rounded corners (HUD panel; graph bars/budget
/// line pass radius 0). Reuses `BLIT_VS` (position via `u_rect`/`u_screen`, and its
/// `v_tex` for the corner mask); premultiplied output to match the compositor's blend.
const SOLID_FS: &str = r#"#version 330 core
in vec2 v_tex;
uniform vec4 u_color;   // straight RGBA
uniform vec4 u_rect;    // x, y, w, h (px) — shared with BLIT_VS
uniform float u_radius; // corner radius (px); 0 = square
out vec4 frag;
void main() {
    float a = u_color.a;
    if (u_radius > 0.0) {
        // Same rounded-box SDF as the window blit: fade alpha to 0 outside the
        // rounded rect so the panel corners round off. `d` = distance outside it.
        vec2 hs = u_rect.zw * 0.5;
        float r = min(u_radius, min(hs.x, hs.y));
        vec2 p = abs(v_tex * u_rect.zw - hs);
        float d = length(max(p - (hs - r), vec2(0.0))) - r;
        float aa = fwidth(d);
        a *= 1.0 - smoothstep(-aa, aa, d);     // derivative-based AA (~2px), like the SDF text
    }
    frag = vec4(u_color.rgb * a, a);
}
"#;

fn load_glow(egl: &egl::DynamicInstance<egl::EGL1_5>) -> glow::Context {
    unsafe {
        glow::Context::from_loader_function(|name| match egl.get_proc_address(name) {
            Some(p) => p as *const c_void,
            None => std::ptr::null(),
        })
    }
}

/// Compile + link a vertex/fragment program. Requires a current GL context
/// (caller invokes this from within the backend's context).
fn make_program(gl: &glow::Context, vs: &str, fs: &str) -> Result<glow::NativeProgram> {
    unsafe {
        let program = gl.create_program().map_err(|e| anyhow!("create_program: {e}"))?;
        let mut shaders = Vec::new();
        for (ty, src) in [(glow::VERTEX_SHADER, vs), (glow::FRAGMENT_SHADER, fs)] {
            let sh = gl.create_shader(ty).map_err(|e| anyhow!("create_shader: {e}"))?;
            gl.shader_source(sh, src);
            gl.compile_shader(sh);
            if !gl.get_shader_compile_status(sh) {
                bail!("shader compile failed: {}", gl.get_shader_info_log(sh));
            }
            gl.attach_shader(program, sh);
            shaders.push(sh);
        }
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            bail!("program link failed: {}", gl.get_program_info_log(program));
        }
        for sh in shaders {
            gl.detach_shader(program, sh);
            gl.delete_shader(sh);
        }
        Ok(program)
    }
}

/// Headless EGL/GL smoke test (pbuffer + clear + readback of GL info).
pub fn first_light() -> Result<GlInfo> {
    let xlib = x11_dl::xlib::Xlib::open().map_err(|e| anyhow!("dlopen libX11: {e}"))?;
    // Must precede the first Xlib call: lets libX11 install its locks so Mesa's
    // driver threads don't trip the "Xlib is not thread-safe" stderr warning.
    if unsafe { (xlib.XInitThreads)() } == 0 {
        bail!("XInitThreads failed");
    }
    let xdisplay = unsafe { (xlib.XOpenDisplay)(std::ptr::null()) };
    if xdisplay.is_null() {
        bail!("XOpenDisplay(NULL) failed (is DISPLAY set?)");
    }
    let egl = unsafe { egl::DynamicInstance::<egl::EGL1_5>::load_required() }
        .map_err(|e| anyhow!("load libEGL.so.1 (>=1.5): {e}"))?;
    let display = unsafe { egl.get_display(xdisplay as egl::NativeDisplayType) }
        .ok_or_else(|| anyhow!("eglGetDisplay returned no display"))?;
    let (major, minor) = egl.initialize(display).map_err(|e| anyhow!("eglInitialize: {e:?}"))?;
    tracing::info!("EGL {major}.{minor} initialised");
    egl.bind_api(egl::OPENGL_API).map_err(|e| anyhow!("eglBindAPI: {e:?}"))?;

    let config_attribs = [
        egl::SURFACE_TYPE, egl::PBUFFER_BIT,
        egl::RENDERABLE_TYPE, egl::OPENGL_BIT,
        egl::RED_SIZE, 8, egl::GREEN_SIZE, 8, egl::BLUE_SIZE, 8, egl::ALPHA_SIZE, 8,
        egl::NONE,
    ];
    let config = egl
        .choose_first_config(display, &config_attribs)
        .map_err(|e| anyhow!("eglChooseConfig: {e:?}"))?
        .ok_or_else(|| anyhow!("no matching EGLConfig"))?;
    let surface = egl
        .create_pbuffer_surface(display, config, &[egl::WIDTH, 64, egl::HEIGHT, 64, egl::NONE])
        .map_err(|e| anyhow!("eglCreatePbufferSurface: {e:?}"))?;
    let context = egl
        .create_context(display, config, None, &[egl::NONE])
        .map_err(|e| anyhow!("eglCreateContext: {e:?}"))?;
    egl.make_current(display, Some(surface), Some(surface), Some(context))
        .map_err(|e| anyhow!("eglMakeCurrent: {e:?}"))?;

    let gl = load_glow(&egl);
    let info = unsafe {
        let info = GlInfo {
            vendor: gl.get_parameter_string(glow::VENDOR),
            renderer: gl.get_parameter_string(glow::RENDERER),
            version: gl.get_parameter_string(glow::VERSION),
        };
        gl.clear_color(0.10, 0.40, 0.80, 1.0);
        gl.clear(glow::COLOR_BUFFER_BIT);
        gl.finish();
        let err = gl.get_error();
        if err != glow::NO_ERROR {
            bail!("GL error after clear: 0x{err:04x}");
        }
        info
    };
    tracing::info!(vendor=%info.vendor, renderer=%info.renderer, version=%info.version, "GL first light OK");

    let _ = egl.make_current(display, None, None, None);
    let _ = egl.destroy_context(display, context);
    let _ = egl.destroy_surface(display, surface);
    let _ = egl.terminate(display);
    unsafe { (xlib.XCloseDisplay)(xdisplay) };
    Ok(info)
}

/// One level of the blur pyramid: an FBO with a colour-texture attachment.
/// Level 0 is full screen resolution; each subsequent level is half the previous.
struct BlurLevel {
    fbo: glow::NativeFramebuffer,
    tex: glow::NativeTexture,
    w: i32,
    h: i32,
}

/// The lazily-built dual-Kawase pyramid, sized to the current screen and rebuilt
/// (all levels freed + recreated) when the screen resolution changes.
struct BlurChain {
    w: i32,
    h: i32,
    levels: Vec<BlurLevel>,
}

/// An EGL/GL context rendering into a target X window (the composite overlay).
pub struct GlBackend {
    xlib: x11_dl::xlib::Xlib,
    xdisplay: *mut x11_dl::xlib::Display,
    egl: egl::DynamicInstance<egl::EGL1_5>,
    display: egl::Display,
    surface: egl::Surface,
    /// Whether `EGL_EXT_buffer_age` is available (enables damage-based partial repaint).
    buffer_age_supported: bool,
    context: egl::Context,
    gl: glow::Context,
    program: glow::NativeProgram,
    vao: glow::NativeVertexArray,
    u_rect: Option<glow::NativeUniformLocation>,
    u_screen: Option<glow::NativeUniformLocation>,
    u_tex: Option<glow::NativeUniformLocation>,
    u_opacity: Option<glow::NativeUniformLocation>,
    u_corner: Option<glow::NativeUniformLocation>,
    // Wobble mesh program: its own VAO with a dynamic vertex VBO (re-uploaded each
    // frame from the spring sim) + a static triangle index EBO (built once).
    mesh_program: glow::NativeProgram,
    mesh_vao: glow::NativeVertexArray,
    mesh_vbo: glow::NativeBuffer,
    mesh_index_count: i32,
    m_screen: Option<glow::NativeUniformLocation>,
    m_tex: Option<glow::NativeUniformLocation>,
    m_opacity: Option<glow::NativeUniformLocation>,
    // Drop-shadow program (shares the unit-quad VAO and BLIT_VS).
    shadow_program: glow::NativeProgram,
    s_rect: Option<glow::NativeUniformLocation>,
    s_screen: Option<glow::NativeUniformLocation>,
    s_inner: Option<glow::NativeUniformLocation>,
    s_shadow: Option<glow::NativeUniformLocation>,
    s_corner: Option<glow::NativeUniformLocation>,
    // Dual-Kawase blur: down/up programs (share BLUR_VS) + a frost pass (reuses
    // BLIT_VS), and a lazily-built FBO pyramid (rebuilt on resize).
    down_program: glow::NativeProgram,
    d_src: Option<glow::NativeUniformLocation>,
    d_halfpixel: Option<glow::NativeUniformLocation>,
    d_offset: Option<glow::NativeUniformLocation>,
    up_program: glow::NativeProgram,
    up_src: Option<glow::NativeUniformLocation>,
    up_halfpixel: Option<glow::NativeUniformLocation>,
    up_offset: Option<glow::NativeUniformLocation>,
    frost_program: glow::NativeProgram,
    f_tex: Option<glow::NativeUniformLocation>,
    f_screen: Option<glow::NativeUniformLocation>,
    f_rect: Option<glow::NativeUniformLocation>,
    f_corner: Option<glow::NativeUniformLocation>,
    // Burn/dissolve close program (reuses BLIT_VS for placement).
    burn_program: glow::NativeProgram,
    bu_rect: Option<glow::NativeUniformLocation>,
    bu_screen: Option<glow::NativeUniformLocation>,
    bu_tex: Option<glow::NativeUniformLocation>,
    bu_opacity: Option<glow::NativeUniformLocation>,
    bu_progress: Option<glow::NativeUniformLocation>,
    bu_seed: Option<glow::NativeUniformLocation>,
    bu_segscale: Option<glow::NativeUniformLocation>,
    bu_ember: Option<glow::NativeUniformLocation>,
    bu_ember_cool: Option<glow::NativeUniformLocation>,
    bu_ember_hot: Option<glow::NativeUniformLocation>,
    // Spin (rotate-about-centre) program: SPIN_VS + the shared BLIT_FS, own VAO.
    spin_program: glow::NativeProgram,
    spin_vao: glow::NativeVertexArray,
    sp_rect: Option<glow::NativeUniformLocation>,
    sp_screen: Option<glow::NativeUniformLocation>,
    sp_tex: Option<glow::NativeUniformLocation>,
    sp_opacity: Option<glow::NativeUniformLocation>,
    sp_corner: Option<glow::NativeUniformLocation>,
    sp_angle: Option<glow::NativeUniformLocation>,
    // Ripple (radial UV refraction) program: BLIT_VS + RIPPLE_FS, reuses the blit VAO.
    ripple_program: glow::NativeProgram,
    rp_rect: Option<glow::NativeUniformLocation>,
    rp_screen: Option<glow::NativeUniformLocation>,
    rp_tex: Option<glow::NativeUniformLocation>,
    rp_opacity: Option<glow::NativeUniformLocation>,
    rp_center: Option<glow::NativeUniformLocation>,
    rp_amp: Option<glow::NativeUniformLocation>,
    rp_wavelength: Option<glow::NativeUniformLocation>,
    rp_phase: Option<glow::NativeUniformLocation>,
    rp_r0: Option<glow::NativeUniformLocation>,
    // Wave (traveling-crest UV refraction) program: BLIT_VS + WAVE_FS, reuses the blit VAO.
    wave_program: glow::NativeProgram,
    wv_rect: Option<glow::NativeUniformLocation>,
    wv_screen: Option<glow::NativeUniformLocation>,
    wv_tex: Option<glow::NativeUniformLocation>,
    wv_opacity: Option<glow::NativeUniformLocation>,
    wv_amp: Option<glow::NativeUniformLocation>,
    wv_wavelength: Option<glow::NativeUniformLocation>,
    wv_phase: Option<glow::NativeUniformLocation>,
    wv_axis: Option<glow::NativeUniformLocation>,
    // Drain (whirlpool close) program: BLIT_VS + DRAIN_FS, reuses the blit VAO.
    drain_program: glow::NativeProgram,
    dr_rect: Option<glow::NativeUniformLocation>,
    dr_screen: Option<glow::NativeUniformLocation>,
    dr_tex: Option<glow::NativeUniformLocation>,
    dr_opacity: Option<glow::NativeUniformLocation>,
    dr_center: Option<glow::NativeUniformLocation>,
    dr_progress: Option<glow::NativeUniformLocation>,
    dr_turns: Option<glow::NativeUniformLocation>,
    dr_seed: Option<glow::NativeUniformLocation>,
    blur: RefCell<Option<BlurChain>>,
    image_target: ImageTargetTexture2DOes,
    render: RenderParams,
    /// Runtime TrueType text renderer (FPS HUD / OSD / notify). `None` when no
    /// usable font is configured — on-screen text is then disabled.
    text: Option<text::TextRenderer>,
    /// Global on-screen-text size multiplier (`[font] size`), applied to HUD/OSD px.
    font_size: f32,
    /// Solid-fill program (HUD panel + graph bars), shares `BLIT_VS`.
    solid_program: glow::NativeProgram,
    sol_rect: Option<glow::NativeUniformLocation>,
    sol_screen: Option<glow::NativeUniformLocation>,
    sol_color: Option<glow::NativeUniformLocation>,
    sol_radius: Option<glow::NativeUniformLocation>,
    /// Double-buffered `GL_TIME_ELAPSED` queries for per-composite GPU render time
    /// (`None` if unsupported). Read two frames later, so reading never stalls.
    gpu_timers: [Option<glow::NativeQuery>; 2],
    timer_slot: Cell<usize>,
    timer_count: Cell<u8>,
    /// Last pixmap id whose `create_image` failed, so the per-frame warning is
    /// logged once per distinct failing pixmap instead of every repaint (a stale
    /// pixmap would otherwise flood the log). `0` = none.
    warned_pixmap: Cell<u32>,
    /// Last measured composite render time (ms) + a ring of recent values (graph).
    render_ms: Cell<f32>,
    render_samples: RefCell<VecDeque<f32>>,
}

impl GlBackend {
    /// Create an EGL window surface on X window `window` (X visual `visual_id`),
    /// a current GL context with vsync, and the blit program + quad.
    pub fn new(window: u32, visual_id: u32, render: RenderParams) -> Result<Self> {
        let xlib = x11_dl::xlib::Xlib::open().map_err(|e| anyhow!("dlopen libX11: {e}"))?;
        // Must precede the first Xlib call: lets libX11 install its locks so Mesa's
        // driver threads don't trip the "Xlib is not thread-safe" stderr warning.
        if unsafe { (xlib.XInitThreads)() } == 0 {
            bail!("XInitThreads failed");
        }
        let xdisplay = unsafe { (xlib.XOpenDisplay)(std::ptr::null()) };
        if xdisplay.is_null() {
            bail!("XOpenDisplay(NULL) failed");
        }
        let egl = unsafe { egl::DynamicInstance::<egl::EGL1_5>::load_required() }
            .map_err(|e| anyhow!("load libEGL.so.1 (>=1.5): {e}"))?;
        let display = unsafe { egl.get_display(xdisplay as egl::NativeDisplayType) }
            .ok_or_else(|| anyhow!("eglGetDisplay returned no display"))?;
        let (major, minor) = egl.initialize(display).map_err(|e| anyhow!("eglInitialize: {e:?}"))?;
        tracing::info!("EGL {major}.{minor} initialised (window backend)");
        egl.bind_api(egl::OPENGL_API).map_err(|e| anyhow!("eglBindAPI: {e:?}"))?;

        let attribs = [
            egl::SURFACE_TYPE, egl::WINDOW_BIT,
            egl::RENDERABLE_TYPE, egl::OPENGL_BIT,
            egl::RED_SIZE, 8, egl::GREEN_SIZE, 8, egl::BLUE_SIZE, 8,
            egl::NONE,
        ];
        let mut configs: Vec<egl::Config> = Vec::with_capacity(64);
        egl.choose_config(display, &attribs, &mut configs)
            .map_err(|e| anyhow!("eglChooseConfig: {e:?}"))?;
        let config = configs
            .into_iter()
            .find(|c| {
                egl.get_config_attrib(display, *c, egl::NATIVE_VISUAL_ID).ok()
                    == Some(visual_id as egl::Int)
            })
            .ok_or_else(|| anyhow!("no EGLConfig matching overlay visual 0x{visual_id:x}"))?;

        let surface = unsafe {
            egl.create_window_surface(display, config, (window as usize) as egl::NativeWindowType, None)
        }
        .map_err(|e| anyhow!("eglCreateWindowSurface: {e:?}"))?;
        let context = egl
            .create_context(display, config, None, &[egl::NONE])
            .map_err(|e| anyhow!("eglCreateContext: {e:?}"))?;
        egl.make_current(display, Some(surface), Some(surface), Some(context))
            .map_err(|e| anyhow!("eglMakeCurrent: {e:?}"))?;
        let _ = egl.swap_interval(display, 1); // vsync
        // EGL_EXT_buffer_age lets us repaint only the damaged region each frame.
        let buffer_age_supported = egl
            .query_string(Some(display), egl::EXTENSIONS)
            .map(|s| s.to_string_lossy().contains("EGL_EXT_buffer_age"))
            .unwrap_or(false);
        tracing::info!(buffer_age = buffer_age_supported, "EGL surface ready");

        let gl = load_glow(&egl);
        let renderer = unsafe { gl.get_parameter_string(glow::RENDERER) };

        // glEGLImageTargetTexture2DOES (texture-from-EGLImage).
        let image_target: ImageTargetTexture2DOes = {
            let f = egl
                .get_proc_address("glEGLImageTargetTexture2DOES")
                .ok_or_else(|| anyhow!("glEGLImageTargetTexture2DOES unavailable"))?;
            unsafe { std::mem::transmute::<_, ImageTargetTexture2DOes>(f) }
        };

        // Blit program + unit-quad VAO.
        let (program, vao, u_rect, u_screen, u_tex, u_opacity, u_corner) = unsafe {
            let program = make_program(&gl, BLIT_VS, BLIT_FS)?;
            let vao = gl.create_vertex_array().map_err(|e| anyhow!("vao: {e}"))?;
            gl.bind_vertex_array(Some(vao));
            let vbo = gl.create_buffer().map_err(|e| anyhow!("vbo: {e}"))?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            let verts: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
            let bytes = std::slice::from_raw_parts(verts.as_ptr() as *const u8, 32);
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes, glow::STATIC_DRAW);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 0, 0);
            gl.bind_vertex_array(None);
            (
                program,
                vao,
                gl.get_uniform_location(program, "u_rect"),
                gl.get_uniform_location(program, "u_screen"),
                gl.get_uniform_location(program, "u_tex"),
                gl.get_uniform_location(program, "u_opacity"),
                gl.get_uniform_location(program, "u_corner"),
            )
        };

        // Spin program: SPIN_VS + the shared BLIT_FS, its own unit-quad VAO.
        // Selected per-window when a spin angle is active (rotate about centre).
        let (spin_program, spin_vao, sp_rect, sp_screen, sp_tex, sp_opacity, sp_corner, sp_angle) = unsafe {
            let program = make_program(&gl, SPIN_VS, BLIT_FS)?;
            let vao = gl.create_vertex_array().map_err(|e| anyhow!("spin vao: {e}"))?;
            gl.bind_vertex_array(Some(vao));
            let vbo = gl.create_buffer().map_err(|e| anyhow!("spin vbo: {e}"))?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            let verts: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
            let bytes = std::slice::from_raw_parts(verts.as_ptr() as *const u8, 32);
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes, glow::STATIC_DRAW);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 0, 0);
            gl.bind_vertex_array(None);
            (
                program,
                vao,
                gl.get_uniform_location(program, "u_rect"),
                gl.get_uniform_location(program, "u_screen"),
                gl.get_uniform_location(program, "u_tex"),
                gl.get_uniform_location(program, "u_opacity"),
                gl.get_uniform_location(program, "u_corner"),
                gl.get_uniform_location(program, "u_angle"),
            )
        };

        // Ripple program: BLIT_VS + RIPPLE_FS, reuses the blit unit-quad VAO (same
        // `a_pos` layout). Selected per-window when a radial ripple is active.
        let (ripple_program, rp_rect, rp_screen, rp_tex, rp_opacity, rp_center, rp_amp, rp_wavelength, rp_phase, rp_r0) = unsafe {
            let program = make_program(&gl, BLIT_VS, RIPPLE_FS)?;
            (
                program,
                gl.get_uniform_location(program, "u_rect"),
                gl.get_uniform_location(program, "u_screen"),
                gl.get_uniform_location(program, "u_tex"),
                gl.get_uniform_location(program, "u_opacity"),
                gl.get_uniform_location(program, "u_center"),
                gl.get_uniform_location(program, "u_amp"),
                gl.get_uniform_location(program, "u_wavelength"),
                gl.get_uniform_location(program, "u_phase"),
                gl.get_uniform_location(program, "u_r0"),
            )
        };

        // Wave program: BLIT_VS + WAVE_FS, reuses the blit unit-quad VAO. Selected
        // per-window when a traveling wave is active (per-pixel; replaces mesh wave).
        let (wave_program, wv_rect, wv_screen, wv_tex, wv_opacity, wv_amp, wv_wavelength, wv_phase, wv_axis) = unsafe {
            let program = make_program(&gl, BLIT_VS, WAVE_FS)?;
            (
                program,
                gl.get_uniform_location(program, "u_rect"),
                gl.get_uniform_location(program, "u_screen"),
                gl.get_uniform_location(program, "u_tex"),
                gl.get_uniform_location(program, "u_opacity"),
                gl.get_uniform_location(program, "u_amp"),
                gl.get_uniform_location(program, "u_wavelength"),
                gl.get_uniform_location(program, "u_phase"),
                gl.get_uniform_location(program, "u_axis"),
            )
        };

        // Drain program: BLIT_VS + DRAIN_FS, reuses the blit unit-quad VAO. Selected
        // per-window while a whirlpool close is in progress (per-pixel; close driver).
        let (drain_program, dr_rect, dr_screen, dr_tex, dr_opacity, dr_center, dr_progress, dr_turns, dr_seed) = unsafe {
            let program = make_program(&gl, BLIT_VS, DRAIN_FS)?;
            (
                program,
                gl.get_uniform_location(program, "u_rect"),
                gl.get_uniform_location(program, "u_screen"),
                gl.get_uniform_location(program, "u_tex"),
                gl.get_uniform_location(program, "u_opacity"),
                gl.get_uniform_location(program, "u_center"),
                gl.get_uniform_location(program, "u_progress"),
                gl.get_uniform_location(program, "u_turns"),
                gl.get_uniform_location(program, "u_seed"),
            )
        };

        // Wobble mesh program + its VAO: a dynamic vertex VBO ([x,y,u,v] per
        // control point, re-uploaded per frame) and a static triangle-index EBO
        // (two triangles per grid cell). The EBO binding is captured in the VAO.
        let (mesh_program, mesh_vao, mesh_vbo, mesh_index_count, m_screen, m_tex, m_opacity) = unsafe {
            let program = make_program(&gl, MESH_VS, MESH_FS)?;
            let vao = gl.create_vertex_array().map_err(|e| anyhow!("mesh vao: {e}"))?;
            gl.bind_vertex_array(Some(vao));
            let vbo = gl.create_buffer().map_err(|e| anyhow!("mesh vbo: {e}"))?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            // N*N vertices × (vec2 pos + vec2 uv) × 4 bytes; filled each frame.
            let vbo_bytes = (MESH_N * MESH_N * 4 * std::mem::size_of::<f32>()) as i32;
            gl.buffer_data_size(glow::ARRAY_BUFFER, vbo_bytes, glow::DYNAMIC_DRAW);
            let stride = (4 * std::mem::size_of::<f32>()) as i32; // [x, y, u, v]
            gl.enable_vertex_attrib_array(0); // a_px
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, stride, 0);
            gl.enable_vertex_attrib_array(1); // a_uv
            gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, stride, 2 * std::mem::size_of::<f32>() as i32);
            let ebo = gl.create_buffer().map_err(|e| anyhow!("mesh ebo: {e}"))?;
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ebo));
            let mut idx: Vec<u32> = Vec::with_capacity((MESH_N - 1) * (MESH_N - 1) * 6);
            for j in 0..MESH_N - 1 {
                for i in 0..MESH_N - 1 {
                    let v00 = (j * MESH_N + i) as u32;
                    let v10 = (j * MESH_N + i + 1) as u32;
                    let v01 = ((j + 1) * MESH_N + i) as u32;
                    let v11 = ((j + 1) * MESH_N + i + 1) as u32;
                    idx.extend_from_slice(&[v00, v10, v11, v00, v11, v01]);
                }
            }
            let idx_bytes =
                std::slice::from_raw_parts(idx.as_ptr() as *const u8, idx.len() * std::mem::size_of::<u32>());
            gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, idx_bytes, glow::STATIC_DRAW);
            let count = idx.len() as i32;
            gl.bind_vertex_array(None); // captures the EBO binding in the VAO
            (
                program,
                vao,
                vbo,
                count,
                gl.get_uniform_location(program, "u_screen"),
                gl.get_uniform_location(program, "u_tex"),
                gl.get_uniform_location(program, "u_opacity"),
            )
        };

        // Shadow program: same vertex shader (unit quad -> u_rect), shadow FS.
        let (shadow_program, s_rect, s_screen, s_inner, s_shadow, s_corner) = unsafe {
            let sp = make_program(&gl, BLIT_VS, SHADOW_FS)?;
            (
                sp,
                gl.get_uniform_location(sp, "u_rect"),
                gl.get_uniform_location(sp, "u_screen"),
                gl.get_uniform_location(sp, "u_inner"),
                gl.get_uniform_location(sp, "u_shadow"),
                gl.get_uniform_location(sp, "u_scorner"),
            )
        };

        // Blur programs: dual-Kawase down/up (share BLUR_VS) + a frost pass that
        // reuses BLIT_VS to place the blurred backdrop under a translucent window.
        let (down_program, d_src, d_halfpixel, d_offset) = unsafe {
            let p = make_program(&gl, BLUR_VS, DOWN_FS)?;
            (
                p,
                gl.get_uniform_location(p, "u_src"),
                gl.get_uniform_location(p, "u_halfpixel"),
                gl.get_uniform_location(p, "u_offset"),
            )
        };
        let (up_program, up_src, up_halfpixel, up_offset) = unsafe {
            let p = make_program(&gl, BLUR_VS, UP_FS)?;
            (
                p,
                gl.get_uniform_location(p, "u_src"),
                gl.get_uniform_location(p, "u_halfpixel"),
                gl.get_uniform_location(p, "u_offset"),
            )
        };
        let (frost_program, f_tex, f_screen, f_rect, f_corner) = unsafe {
            let p = make_program(&gl, BLIT_VS, FROST_FS)?;
            (
                p,
                gl.get_uniform_location(p, "u_tex"),
                gl.get_uniform_location(p, "u_screen"),
                gl.get_uniform_location(p, "u_rect"),
                gl.get_uniform_location(p, "u_corner"),
            )
        };
        // Burn/dissolve program (reuses BLIT_VS → u_rect/u_screen).
        let (burn_program, bu_rect, bu_screen, bu_tex, bu_opacity, bu_progress, bu_seed, bu_segscale, bu_ember, bu_ember_cool, bu_ember_hot) = unsafe {
            let p = make_program(&gl, BLIT_VS, BURN_FS)?;
            (
                p,
                gl.get_uniform_location(p, "u_rect"),
                gl.get_uniform_location(p, "u_screen"),
                gl.get_uniform_location(p, "u_tex"),
                gl.get_uniform_location(p, "u_opacity"),
                gl.get_uniform_location(p, "u_progress"),
                gl.get_uniform_location(p, "u_seed"),
                gl.get_uniform_location(p, "u_segscale"),
                gl.get_uniform_location(p, "u_ember"),
                gl.get_uniform_location(p, "u_ember_cool"),
                gl.get_uniform_location(p, "u_ember_hot"),
            )
        };
        let (solid_program, sol_rect, sol_screen, sol_color, sol_radius) = unsafe {
            let p = make_program(&gl, BLIT_VS, SOLID_FS)?;
            (
                p,
                gl.get_uniform_location(p, "u_rect"),
                gl.get_uniform_location(p, "u_screen"),
                gl.get_uniform_location(p, "u_color"),
                gl.get_uniform_location(p, "u_radius"),
            )
        };
        // Double-buffered GPU timer queries for HUD render time (all-or-nothing).
        let gpu_timers = unsafe {
            match (gl.create_query(), gl.create_query()) {
                (Ok(a), Ok(b)) => [Some(a), Some(b)],
                _ => {
                    tracing::warn!("GPU timer queries unavailable — HUD render time disabled");
                    [None, None]
                }
            }
        };
        tracing::info!(%renderer, window, "GL window backend ready (blit + shadow + blur + text + solid programs loaded)");

        Ok(GlBackend {
            xlib, xdisplay, egl, display, surface, buffer_age_supported, context, gl,
            program, vao, u_rect, u_screen, u_tex, u_opacity, u_corner,
            mesh_program, mesh_vao, mesh_vbo, mesh_index_count, m_screen, m_tex, m_opacity,
            shadow_program, s_rect, s_screen, s_inner, s_shadow, s_corner,
            down_program, d_src, d_halfpixel, d_offset,
            up_program, up_src, up_halfpixel, up_offset,
            frost_program, f_tex, f_screen, f_rect, f_corner,
            burn_program, bu_rect, bu_screen, bu_tex, bu_opacity, bu_progress, bu_seed, bu_segscale, bu_ember, bu_ember_cool, bu_ember_hot,
            spin_program, spin_vao, sp_rect, sp_screen, sp_tex, sp_opacity, sp_corner, sp_angle,
            ripple_program, rp_rect, rp_screen, rp_tex, rp_opacity, rp_center, rp_amp, rp_wavelength, rp_phase, rp_r0,
            wave_program, wv_rect, wv_screen, wv_tex, wv_opacity, wv_amp, wv_wavelength, wv_phase, wv_axis,
            drain_program, dr_rect, dr_screen, dr_tex, dr_opacity, dr_center, dr_progress, dr_turns, dr_seed,
            blur: RefCell::new(None),
            image_target, render, text: None, font_size: 1.0,
            solid_program, sol_rect, sol_screen, sol_color, sol_radius,
            gpu_timers,
            timer_slot: Cell::new(0),
            timer_count: Cell::new(0),
            warned_pixmap: Cell::new(0),
            render_ms: Cell::new(0.0),
            render_samples: RefCell::new(VecDeque::new()),
        })
    }

    /// Swap in new render parameters (shadow size/strength, background) — used on
    /// config reload. Takes effect on the next `present_windows`.
    pub fn set_render_params(&mut self, render: RenderParams) {
        self.render = render;
    }

    /// (Re)load the on-screen-text font from `path` (a `.ttf`) at global size
    /// multiplier `size`. An empty, unreadable, or unparsable path **disables**
    /// on-screen text (`self.text = None`) rather than erroring — the compositor
    /// keeps running, just draws no HUD/OSD/notify text. Called at startup and on
    /// config reload; requires the backend's GL context to be current (it is on the
    /// single compositor thread).
    pub fn set_font(&mut self, path: &str, size: f32) {
        self.font_size = if size > 0.0 { size } else { 1.0 };
        let next = if path.is_empty() {
            tracing::info!("font path empty — on-screen text disabled");
            None
        } else {
            match std::fs::read(path) {
                Ok(bytes) => match text::TextRenderer::new(&self.gl, &bytes) {
                    Ok(tr) => {
                        tracing::info!(path, "on-screen text font loaded");
                        Some(tr)
                    }
                    Err(e) => {
                        tracing::warn!(path, "font parse failed: {e} — on-screen text disabled");
                        None
                    }
                },
                Err(e) => {
                    tracing::warn!(path, "font read failed: {e} — on-screen text disabled");
                    None
                }
            }
        };
        // Free the previous font's GL objects before replacing (reload-safe).
        if let Some(old) = self.text.take() {
            old.destroy(&self.gl);
        }
        self.text = next;
    }

    /// Whether on-screen text is currently available (a usable font is loaded).
    pub fn has_text(&self) -> bool {
        self.text.is_some()
    }

    /// Clear the surface to a colour and present.
    pub fn clear_present(&self, r: f32, g: f32, b: f32, a: f32) -> Result<()> {
        unsafe {
            self.gl.clear_color(r, g, b, a);
            self.gl.clear(glow::COLOR_BUFFER_BIT);
        }
        self.egl
            .swap_buffers(self.display, self.surface)
            .map_err(|e| anyhow!("eglSwapBuffers: {e:?}"))?;
        Ok(())
    }

    /// Bind an X window's pixmap as a GL texture and blit it at `(x,y,w,h)` over
    /// a cleared overlay, then present. (Single-window path; the renderer will
    /// loop this over the stack without clearing between windows.)
    #[allow(clippy::too_many_arguments)]
    pub fn present_window_pixmap(
        &self,
        pixmap: u32,
        x: i32, y: i32, w: i32, h: i32,
        screen_w: i32, screen_h: i32,
    ) -> Result<()> {
        let buffer = unsafe { egl::ClientBuffer::from_ptr((pixmap as usize) as egl::EGLClientBuffer) };
        let no_ctx = unsafe { egl::Context::from_ptr(egl::NO_CONTEXT) };
        let attribs = [egl::IMAGE_PRESERVED as egl::Attrib, 1, egl::ATTRIB_NONE];
        let image = self
            .egl
            .create_image(self.display, no_ctx, EGL_NATIVE_PIXMAP_KHR, buffer, &attribs)
            .map_err(|e| anyhow!("eglCreateImage(pixmap 0x{pixmap:x}): {e:?}"))?;

        let (e_bind, e_draw) = unsafe {
            let tex = self.gl.create_texture().map_err(|e| anyhow!("create_texture: {e}"))?;
            self.gl.active_texture(glow::TEXTURE0);
            self.gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            (self.image_target)(glow::TEXTURE_2D, image.as_ptr() as *const c_void);
            let e_bind = self.gl.get_error();

            self.gl.viewport(0, 0, screen_w, screen_h);
            let bg = self.render.background;
            self.gl.clear_color(bg[0], bg[1], bg[2], 1.0);
            self.gl.clear(glow::COLOR_BUFFER_BIT);

            self.gl.use_program(Some(self.program));
            self.gl.uniform_4_f32(self.u_rect.as_ref(), x as f32, y as f32, w as f32, h as f32);
            self.gl.uniform_2_f32(self.u_screen.as_ref(), screen_w as f32, screen_h as f32);
            self.gl.uniform_1_i32(self.u_tex.as_ref(), 0);
            self.gl.uniform_1_f32(self.u_opacity.as_ref(), 1.0);
            self.gl.uniform_1_f32(self.u_corner.as_ref(), 0.0);
            self.gl.bind_vertex_array(Some(self.vao));
            self.gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            self.gl.bind_vertex_array(None);
            let e_draw = self.gl.get_error();

            self.gl.delete_texture(tex);
            (e_bind, e_draw)
        };
        if e_bind != glow::NO_ERROR || e_draw != glow::NO_ERROR {
            tracing::warn!("GL errors: after image-bind=0x{e_bind:04x}, after draw=0x{e_draw:04x}");
        } else {
            tracing::info!("blit GL ok (no errors)");
        }
        self.egl
            .swap_buffers(self.display, self.surface)
            .map_err(|e| anyhow!("eglSwapBuffers: {e:?}"))?;
        let _ = self.egl.destroy_image(self.display, image);
        Ok(())
    }

    /// Ensure the blur pyramid exists and matches the screen size, rebuilding it
    /// (freeing old GL objects) on a resolution change. Each level is a
    /// colour-texture FBO: level 0 is full res, each subsequent level half.
    fn ensure_blur_chain(&self, sw: i32, sh: i32) {
        // Already sized to this screen? (Avoid a let-chain — the deploy target's
        // stable rustc rejects them.)
        let up_to_date =
            self.blur.borrow().as_ref().is_some_and(|c| c.w == sw && c.h == sh);
        if up_to_date {
            return;
        }
        let gl = &self.gl;
        let mut slot = self.blur.borrow_mut();
        let mut levels = Vec::with_capacity(MAX_BLUR_LEVELS as usize);
        unsafe {
            if let Some(old) = slot.take() {
                for lvl in old.levels {
                    gl.delete_framebuffer(lvl.fbo);
                    gl.delete_texture(lvl.tex);
                }
            }
            for i in 0..MAX_BLUR_LEVELS {
                let (lw, lh) = ((sw >> i).max(1), (sh >> i).max(1));
                let tex = match gl.create_texture() {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::warn!("blur texture: {e}");
                        break;
                    }
                };
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                gl.tex_image_2d(
                    glow::TEXTURE_2D, 0, glow::RGBA8 as i32, lw, lh, 0,
                    glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None),
                );
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
                let fbo = match gl.create_framebuffer() {
                    Ok(f) => f,
                    Err(e) => {
                        gl.delete_texture(tex);
                        tracing::warn!("blur fbo: {e}");
                        break;
                    }
                };
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                gl.framebuffer_texture_2d(
                    glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(tex), 0,
                );
                let st = gl.check_framebuffer_status(glow::FRAMEBUFFER);
                if st != glow::FRAMEBUFFER_COMPLETE {
                    tracing::warn!("blur FBO level {i} incomplete: 0x{st:04x}");
                }
                levels.push(BlurLevel { fbo, tex, w: lw, h: lh });
            }
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.bind_texture(glow::TEXTURE_2D, None);
        }
        tracing::debug!(sw, sh, levels = levels.len(), "blur pyramid (re)built");
        *slot = Some(BlurChain { w: sw, h: sh, levels });
    }

    /// Frost the current overlay framebuffer into blur level 0: copy the whole
    /// composited backdrop, then dual-Kawase down/up `passes` times. Returns the
    /// blurred level-0 texture (for the frost draw), leaving the default
    /// framebuffer + full viewport bound and blending re-enabled.
    fn blur_backdrop(&self, sw: i32, sh: i32, passes: i32, offset: f32) -> Option<glow::NativeTexture> {
        self.ensure_blur_chain(sw, sh);
        let slot = self.blur.borrow();
        let chain = slot.as_ref()?;
        if chain.levels.len() < 2 {
            return None;
        }
        let passes = passes.clamp(1, chain.levels.len() as i32 - 1) as usize;
        let gl = &self.gl;
        unsafe {
            gl.disable(glow::BLEND);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_vertex_array(Some(self.vao));

            // Copy the composited overlay (the backdrop below this window) into level 0.
            gl.bind_texture(glow::TEXTURE_2D, Some(chain.levels[0].tex));
            gl.copy_tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, 0, 0, sw, sh);

            // Downsample 0 -> 1 -> ... -> passes.
            gl.use_program(Some(self.down_program));
            gl.uniform_1_i32(self.d_src.as_ref(), 0);
            for i in 0..passes {
                let (src, dst) = (&chain.levels[i], &chain.levels[i + 1]);
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(dst.fbo));
                gl.viewport(0, 0, dst.w, dst.h);
                gl.bind_texture(glow::TEXTURE_2D, Some(src.tex));
                gl.uniform_2_f32(self.d_halfpixel.as_ref(), 0.5 / src.w as f32, 0.5 / src.h as f32);
                gl.uniform_1_f32(self.d_offset.as_ref(), offset);
                gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            }
            // Upsample passes -> ... -> 1 -> 0 (blurred result ends up in level 0).
            gl.use_program(Some(self.up_program));
            gl.uniform_1_i32(self.up_src.as_ref(), 0);
            for i in (0..passes).rev() {
                let (src, dst) = (&chain.levels[i + 1], &chain.levels[i]);
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(dst.fbo));
                gl.viewport(0, 0, dst.w, dst.h);
                gl.bind_texture(glow::TEXTURE_2D, Some(src.tex));
                gl.uniform_2_f32(self.up_halfpixel.as_ref(), 0.5 / src.w as f32, 0.5 / src.h as f32);
                gl.uniform_1_f32(self.up_offset.as_ref(), offset);
                gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            }
            // Restore the overlay's default framebuffer + full viewport for the caller.
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.viewport(0, 0, sw, sh);
            gl.enable(glow::BLEND);
        }
        Some(chain.levels[0].tex)
    }

    /// Fill a screen-space rect (optionally rounded — `radius` px, 0 = square) with
    /// a solid (premultiplied) colour. Assumes the unit-quad VAO is bound and
    /// blending is enabled (as in `present_windows`).
    #[allow(clippy::too_many_arguments)]
    fn fill_rect(&self, x: f32, y: f32, w: f32, h: f32, radius: f32, color: [f32; 4], sw: i32, sh: i32) {
        unsafe {
            self.gl.use_program(Some(self.solid_program));
            self.gl.uniform_2_f32(self.sol_screen.as_ref(), sw as f32, sh as f32);
            self.gl.uniform_4_f32(self.sol_rect.as_ref(), x, y, w, h);
            self.gl.uniform_1_f32(self.sol_radius.as_ref(), radius);
            self.gl.uniform_4_f32(self.sol_color.as_ref(), color[0], color[1], color[2], color[3]);
            self.gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
        }
    }

    /// Draw the OSD toast — a top-center rounded pill + text that slides down and
    /// fades in as `presence` goes 0→1. Drawn over everything (after the HUD).
    fn draw_osd(&self, osd: &Osd, sw: i32, sh: i32) {
        let p = osd.presence.clamp(0.0, 1.0);
        if p <= 0.0 {
            return;
        }
        // No usable font → no on-screen text (the compositor still runs).
        let Some(text) = self.text.as_ref() else { return };
        let sb = (sh as f32 / 1080.0).max(0.5) * osd.scale * self.font_size;
        // Pop scales the whole banner uniformly; others draw at full size.
        let zoom = if osd.effect == OsdEffect::Pop { 0.6 + 0.4 * p } else { 1.0 };
        let s = sb * zoom;
        let pad = 20.0 * s;
        let text_px = 34.0 * s;
        let radius = 14.0 * s;
        let line_h = text.line_height(text_px);
        // Grow the box up to the screen width (minus a margin); any line wider than
        // that is trimmed to fit with a trailing "..." — measured per glyph, so it's
        // correct for a proportional font (no fixed-advance assumption).
        let hmargin = 40.0 * sb;
        let min_w = text.measure(text_px, "x").0;
        let max_text_w = (sw as f32 - 2.0 * hmargin - 2.0 * pad).max(min_w);
        let ell_w = text.measure(text_px, "...").0;
        let lines: Vec<String> = osd
            .text
            .split('\n')
            .map(|l| {
                if text.measure(text_px, l).0 <= max_text_w {
                    return l.to_string();
                }
                let budget = (max_text_w - ell_w).max(0.0);
                let mut kept = String::new();
                let mut w = 0.0;
                let mut buf = [0u8; 4];
                for ch in l.chars() {
                    let cw = text.measure(text_px, ch.encode_utf8(&mut buf)).0;
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
        let tw = lines.iter().map(|l| text.measure(text_px, l).0).fold(0.0_f32, f32::max);
        let panel_w = tw + 2.0 * pad;
        let panel_h = lines.len() as f32 * line_h + 2.0 * pad;
        let px = (sw as f32 - panel_w) * 0.5; // horizontally centred
        let rest_y = 28.0 * sb; // inset from the top edge (independent of pop zoom)
        // Slide drops in from above the top edge; every other effect rests in place.
        let py = if osd.effect == OsdEffect::Slide {
            -panel_h + (rest_y + panel_h) * p
        } else {
            rest_y
        };
        // Unroll/Stretch reveal the banner from a centre line via a scissor window
        // (text isn't distorted); those stay crisp, the rest fade with `p`.
        let reveal = matches!(osd.effect, OsdEffect::Unroll | OsdEffect::Stretch);
        let alpha = if reveal { 1.0 } else { p };
        if reveal {
            let (rx, ry, rw, rh) = if osd.effect == OsdEffect::Unroll {
                let h = panel_h * p;
                (px, py + (panel_h - h) * 0.5, panel_w, h)
            } else {
                let w = panel_w * p;
                (px + (panel_w - w) * 0.5, py, w, panel_h)
            };
            unsafe {
                self.gl.enable(glow::SCISSOR_TEST);
                // GL scissor origin is bottom-left, so flip Y from our top-left rect.
                self.gl.scissor(rx as i32, sh - (ry + rh) as i32, rw as i32, rh as i32);
            }
        }
        // Background box — skip entirely when transparent (text-only toast).
        let bg = osd.background;
        if bg[3] > 0.001 {
            self.fill_rect(px, py, panel_w, panel_h, radius, [bg[0], bg[1], bg[2], bg[3] * alpha], sw, sh);
        }
        // Left-aligned: each glyph carries its own bearing, and `draw` places text so
        // `y` is the line's top (baseline = y + ascent), so rows stack by `line_h`.
        let tx = px + pad;
        let c = osd.color;
        // An 8-way dark halo behind each line keeps text legible over any backdrop
        // — the contrast the box used to give, so a transparent box still reads.
        let o = 1.6 * s;
        const HALO: [(f32, f32); 8] = [
            (-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0),
            (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0),
        ];
        for (i, line) in lines.iter().enumerate() {
            let ly = py + pad + i as f32 * line_h;
            if osd.outline {
                for (dx, dy) in HALO {
                    text.draw(&self.gl, sw, sh, tx + dx * o, ly + dy * o, text_px, [0.0, 0.0, 0.0, alpha], line);
                }
            }
            text.draw(&self.gl, sw, sh, tx, ly, text_px, [c[0], c[1], c[2], alpha], line);
        }
        if reveal {
            unsafe {
                self.gl.disable(glow::SCISSOR_TEST);
            }
        }
    }

    /// Draw the FPS HUD — a translucent panel, an optional frame-time graph, and
    /// the numbers — anchored to `hud.corner`.
    fn draw_hud(&self, hud: &Hud, sw: i32, sh: i32) {
        // No usable font → no HUD text (the compositor still runs).
        let Some(text) = self.text.as_ref() else { return };
        // Scale the whole HUD with the screen height (1080p = 1×, 4K/2160p = 2×)
        // times the config + global font multipliers, so it stays legible at any DPI.
        let s = (sh as f32 / 1080.0).max(0.5) * hud.scale * self.font_size;
        let pad = 8.0 * s;
        let margin = 28.0 * s; // inset from the screen edge (the HUD floats in a bit)
        let radius = 10.0 * s; // HUD panel corner radius (special-cased, not the window setting)
        let text_px = 20.0 * s;
        // One refresh interval is the render budget: a composite must finish within
        // it to hit vsync. Colour by headroom, not by the (content-driven) frame rate.
        let budget = 1000.0 / hud.refresh_hz.max(1.0);
        let render_ms = self.render_ms.get();
        let th = text.line_height(text_px);
        // Measured-column layout (font-agnostic): the fps/ms numbers are right-aligned
        // within fixed-width fields (sized to a worst-case digit count) and the labels
        // sit at fixed offsets — so a proportional font neither reflows the panel nor
        // jitters the digits as the values change (the old space-padding needed a
        // monospace face). Works tabularly with any TTF.
        let fps_s = format!("{}", hud.fps);
        let ms_s = format!("{render_ms:.1}");
        let numw = text.measure(text_px, "000").0; // fps field: up to 3 digits
        let msw = text.measure(text_px, "000.0").0; // ms field: up to 5 chars
        let sep1 = " fps   ";
        let sep2 = " ms";
        let sep1w = text.measure(text_px, sep1).0;
        let sep2w = text.measure(text_px, sep2).0;
        let tw = numw + sep1w + msw + sep2w;
        let samples = self.render_samples.borrow();
        let bar_w = 2.0 * s;
        let graph_h = if hud.graph { 34.0 * s } else { 0.0 };
        let graph_gap = if hud.graph { 6.0 * s } else { 0.0 };
        let graph_w = if hud.graph { (samples.len() as f32 * bar_w).max(tw) } else { 0.0 };
        // Optional 1m/5m/15m load block (Super+Shift+L): a label column + three
        // right-aligned value columns, all measured — proportional-font-safe.
        let load_px = 15.0 * s;
        let has_load = hud.load.is_some();
        let load_lbl_w = if has_load { text.measure(load_px, "fps  ").0 } else { 0.0 };
        let load_col_w = if has_load { text.measure(load_px, "  000.0").0 } else { 0.0 };
        let load_w = if has_load { load_lbl_w + 3.0 * load_col_w } else { 0.0 };
        let load_pitch = load_px * 1.2;
        let load_cell = if has_load { text.line_height(load_px) } else { 0.0 };
        let load_gap = if has_load { 8.0 * s } else { 0.0 };
        // Two rows: gap + one inter-row pitch + one cell height.
        let load_block_h = if has_load { load_gap + load_pitch + load_cell } else { 0.0 };
        let content_w = tw.max(graph_w).max(load_w);
        let panel_w = content_w + pad * 2.0;
        let panel_h = th + graph_gap + graph_h + load_block_h + pad * 2.0;
        let (px, py) = match hud.corner {
            HudCorner::TopLeft => (margin, margin),
            HudCorner::TopRight => (sw as f32 - margin - panel_w, margin),
            HudCorner::BottomLeft => (margin, sh as f32 - margin - panel_h),
            HudCorner::BottomRight => (sw as f32 - margin - panel_w, sh as f32 - margin - panel_h),
        };
        // Panel background.
        self.fill_rect(px, py, panel_w, panel_h, radius, [0.05, 0.05, 0.07, 0.72], sw, sh);
        // Render-time graph: one bar per composite, full height = one refresh budget.
        // Green = plenty of headroom, amber = tight, red = at/over budget (missed vsync).
        if hud.graph && !samples.is_empty() {
            let gx = px + pad;
            let gy = py + pad + th + graph_gap;
            for (i, &ms) in samples.iter().enumerate() {
                let bx = gx + i as f32 * bar_w;
                if bx + bar_w > gx + content_w {
                    break;
                }
                let norm = (ms / budget).clamp(0.0, 1.0);
                let bh = (norm * graph_h).max(1.0);
                let col = if ms <= budget * 0.5 {
                    [0.40, 0.90, 0.50, 0.90]
                } else if ms <= budget * 0.85 {
                    [0.95, 0.80, 0.30, 0.90]
                } else {
                    [0.95, 0.40, 0.35, 0.90]
                };
                self.fill_rect(bx, gy + (graph_h - bh), (bar_w - 0.5 * s).max(1.0), bh, 0.0, col, sw, sh);
            }
            // Budget ceiling line at the top of the graph (= one refresh interval).
            self.fill_rect(gx, gy, content_w, s.max(1.0), 0.0, [1.0, 1.0, 1.0, 0.22], sw, sh);
        }
        drop(samples);
        // Numbers on top — fps + ms, each number right-aligned in its field.
        let x0 = px + pad;
        let ny = py + pad;
        let col = [0.90, 1.0, 0.95, 1.0];
        let fw = text.measure(text_px, &fps_s).0;
        text.draw(&self.gl, sw, sh, x0 + numw - fw, ny, text_px, col, &fps_s);
        text.draw(&self.gl, sw, sh, x0 + numw, ny, text_px, col, sep1);
        let mx0 = x0 + numw + sep1w;
        let mw = text.measure(text_px, &ms_s).0;
        text.draw(&self.gl, sw, sh, mx0 + msw - mw, ny, text_px, col, &ms_s);
        text.draw(&self.gl, sw, sh, mx0 + msw, ny, text_px, col, sep2);
        // Load block under the graph: label + three right-aligned value columns.
        if let Some(l) = &hud.load {
            let lcol = [0.80, 0.88, 1.0, 1.0];
            let rows: [(&str, [Option<f32>; 3]); 2] = [
                ("fps", [Some(l.fps[0]), Some(l.fps[1]), Some(l.fps[2])]),
                ("ms", [l.render_ms[0], l.render_ms[1], l.render_ms[2]]),
            ];
            let mut ly = py + pad + th + graph_gap + graph_h + load_gap;
            for (label, vals) in rows {
                text.draw(&self.gl, sw, sh, x0, ly, load_px, lcol, label);
                for (k, v) in vals.iter().enumerate() {
                    let vs = match v {
                        Some(x) => format!("{x:.1}"),
                        None => "--".to_string(),
                    };
                    let right = x0 + load_lbl_w + (k as f32 + 1.0) * load_col_w;
                    let vw = text.measure(load_px, &vs).0;
                    text.draw(&self.gl, sw, sh, right - vw, ly, load_px, lcol, &vs);
                }
                ly += load_pitch;
            }
        }
    }

    /// The most recent GPU render time (ms) — the value the HUD shows, measured
    /// ~2 composites ago. `0.0` before the first measurement (or if the GPU timer
    /// is unavailable).
    pub fn render_ms(&self) -> f32 {
        self.render_ms.get()
    }

    /// Age of the back buffer about to be drawn: frames since it was last the front
    /// buffer (1 = last frame, N = N-frames stale). `0` = undefined / unsupported
    /// (`EGL_EXT_buffer_age` absent) → the caller must repaint fully.
    pub fn buffer_age(&self) -> i32 {
        if !self.buffer_age_supported {
            return 0;
        }
        self.egl
            .query_surface(self.display, self.surface, EGL_BUFFER_AGE_EXT)
            .unwrap_or(0)
    }

    /// Read the render-time query that finished ~2 frames ago (so the read never
    /// stalls the pipeline), and push it to the HUD graph ring. No-op until both
    /// double-buffered queries have been recorded at least once.
    fn collect_render_time(&self) {
        let slot = self.timer_slot.get();
        let Some(q) = self.gpu_timers[slot] else { return };
        if self.timer_count.get() < 2 {
            return; // this slot hasn't been recorded yet
        }
        unsafe {
            if self.gl.get_query_parameter_u32(q, glow::QUERY_RESULT_AVAILABLE) == 0 {
                return; // not ready (shouldn't happen 2 frames on); keep last value
            }
            let ns = self.gl.get_query_parameter_u32(q, glow::QUERY_RESULT);
            let ms = ns as f32 / 1_000_000.0;
            self.render_ms.set(ms);
            let mut ring = self.render_samples.borrow_mut();
            ring.push_back(ms);
            while ring.len() > HUD_GRAPH_SAMPLES {
                ring.pop_front();
            }
        }
    }

    /// Composite a stack of windows: clear once, then draw each bottom-to-top —
    /// but only inside its `clip` rectangles (region-level occlusion), so pixels
    /// hidden behind an opaque window on top are never shaded. Draws the optional
    /// `hud` on top, presents once. Items that fail to bind are skipped.
    pub fn present_windows(
        &self,
        items: &[WindowDraw],
        screen_w: i32,
        screen_h: i32,
        hud: Option<&Hud>,
        osd: Option<&Osd>,
        clear: &[region::Rect],
    ) -> Result<()> {
        tracing::trace!(items = items.len(), screen_w, screen_h, "present");
        // Collect the render time measured ~2 frames ago, then time this composite.
        self.collect_render_time();
        let timer = self.gpu_timers[self.timer_slot.get()];
        let RenderParams {
            shadow_radius, shadow_strength, background, blur_passes, blur_radius, ..
        } = self.render;
        unsafe {
            if let Some(q) = timer {
                self.gl.begin_query(glow::TIME_ELAPSED, q);
            }
            self.gl.viewport(0, 0, screen_w, screen_h);
            // Clear only the region being repainted this frame (damage-scissored).
            // GL's scissor origin is bottom-left, so flip Y from our top-left rects.
            self.gl.clear_color(background[0], background[1], background[2], 1.0);
            self.gl.enable(glow::SCISSOR_TEST);
            for r in clear {
                self.gl.scissor(r.x1, screen_h - r.y2, r.width(), r.height());
                self.gl.clear(glow::COLOR_BUFFER_BIT);
            }
            self.gl.disable(glow::SCISSOR_TEST);
            // Premultiplied-alpha "over" so per-window opacity (and the black
            // shadows) blend onto the clear and the windows already drawn beneath.
            self.gl.enable(glow::BLEND);
            self.gl.blend_func(glow::ONE, glow::ONE_MINUS_SRC_ALPHA);
            self.gl.active_texture(glow::TEXTURE0);
            self.gl.bind_vertex_array(Some(self.vao));
            // Shadow program's per-frame constants.
            self.gl.use_program(Some(self.shadow_program));
            self.gl.uniform_2_f32(self.s_screen.as_ref(), screen_w as f32, screen_h as f32);
            self.gl.uniform_2_f32(self.s_shadow.as_ref(), shadow_radius, shadow_strength);
            // Blit program's per-frame constants (corner radius is per-window, set in the loop).
            self.gl.use_program(Some(self.program));
            self.gl.uniform_2_f32(self.u_screen.as_ref(), screen_w as f32, screen_h as f32);
            self.gl.uniform_1_i32(self.u_tex.as_ref(), 0);
        }
        for WindowDraw { quad, clip, mesh, burn, spin, ripple, wave, drain } in items {
            if clip.is_empty() {
                continue; // fully occluded — nothing visible to draw
            }
            let &Quad { pixmap, x, y, w, h, opacity, shadow, blur, corner_radius } = quad;
            let (fx, fy, fw, fh) = (x as f32, y as f32, w as f32, h as f32);
            let win_rect = region::Rect::from_xywh(x, y, w, h);
            // Frost the backdrop first (renders the offscreen blur pyramid — must
            // run BEFORE the scissor test is enabled, or those passes get clipped).
            // Skipped while wobbling — the mesh path draws no frost.
            let frost = if blur && mesh.is_none() {
                self.blur_backdrop(screen_w, screen_h, blur_passes, blur_radius)
            } else {
                None
            };
            // Bind this window's pixmap as a texture once, reused across clip rects.
            let buffer =
                unsafe { egl::ClientBuffer::from_ptr((pixmap as usize) as egl::EGLClientBuffer) };
            let no_ctx = unsafe { egl::Context::from_ptr(egl::NO_CONTEXT) };
            let attribs = [egl::IMAGE_PRESERVED as egl::Attrib, 1, egl::ATTRIB_NONE];
            let image = match self
                .egl
                .create_image(self.display, no_ctx, EGL_NATIVE_PIXMAP_KHR, buffer, &attribs)
            {
                Ok(i) => i,
                Err(e) => {
                    // Throttle: a stale pixmap fails every repaint, so log only when
                    // the failing pixmap changes (once per distinct pixmap) — not each frame.
                    if self.warned_pixmap.get() != pixmap {
                        tracing::warn!("create_image(pixmap 0x{pixmap:x}) failed: {e:?}");
                        self.warned_pixmap.set(pixmap);
                    }
                    continue;
                }
            };
            let tex = match unsafe { self.gl.create_texture() } {
                Ok(t) => t,
                Err(_) => {
                    let _ = self.egl.destroy_image(self.display, image);
                    continue;
                }
            };
            unsafe {
                self.gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
                self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
                self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
                self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
                (self.image_target)(glow::TEXTURE_2D, image.as_ptr() as *const c_void);
                // Burning window: dissolve via the burn program (no shadow / frost /
                // corner rounding — like the mesh path), scissored to each clip rect.
                if let Some(b) = burn {
                    self.gl.use_program(Some(self.burn_program));
                    self.gl.uniform_2_f32(self.bu_screen.as_ref(), screen_w as f32, screen_h as f32);
                    self.gl.uniform_1_i32(self.bu_tex.as_ref(), 0);
                    self.gl.uniform_4_f32(self.bu_rect.as_ref(), fx, fy, fw, fh);
                    self.gl.uniform_1_f32(self.bu_opacity.as_ref(), opacity);
                    self.gl.uniform_1_f32(self.bu_progress.as_ref(), b.progress);
                    self.gl.uniform_1_f32(self.bu_seed.as_ref(), b.seed);
                    self.gl.uniform_1_f32(self.bu_segscale.as_ref(), self.render.burn_seg_scale);
                    self.gl.uniform_1_f32(self.bu_ember.as_ref(), self.render.burn_ember);
                    let ec = self.render.burn_ember_cool;
                    let eh = self.render.burn_ember_hot;
                    self.gl.uniform_3_f32(self.bu_ember_cool.as_ref(), ec[0], ec[1], ec[2]);
                    self.gl.uniform_3_f32(self.bu_ember_hot.as_ref(), eh[0], eh[1], eh[2]);
                    self.gl.enable(glow::SCISSOR_TEST);
                    for r in clip {
                        self.gl.scissor(r.x1, screen_h - r.y2, r.width(), r.height());
                        self.gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
                    }
                    self.gl.disable(glow::SCISSOR_TEST);
                    self.gl.use_program(Some(self.program)); // restore blit for the next item
                    self.gl.delete_texture(tex);
                    let _ = self.egl.destroy_image(self.display, image);
                    continue;
                }
                // Spinning window: rotate about the centre via the spin program (no
                // shadow / frost; corners suppressed — square while it spins),
                // scissored to each clip rect. GL's scissor origin is bottom-left.
                if let Some(angle) = spin {
                    self.gl.use_program(Some(self.spin_program));
                    self.gl.bind_vertex_array(Some(self.spin_vao));
                    self.gl.uniform_2_f32(self.sp_screen.as_ref(), screen_w as f32, screen_h as f32);
                    self.gl.uniform_1_i32(self.sp_tex.as_ref(), 0);
                    self.gl.uniform_4_f32(self.sp_rect.as_ref(), fx, fy, fw, fh);
                    self.gl.uniform_1_f32(self.sp_opacity.as_ref(), opacity);
                    self.gl.uniform_1_f32(self.sp_corner.as_ref(), 0.0);
                    self.gl.uniform_1_f32(self.sp_angle.as_ref(), *angle);
                    self.gl.enable(glow::SCISSOR_TEST);
                    for r in clip {
                        self.gl.scissor(r.x1, screen_h - r.y2, r.width(), r.height());
                        self.gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
                    }
                    self.gl.disable(glow::SCISSOR_TEST);
                    self.gl.bind_vertex_array(Some(self.vao)); // restore unit-quad VAO
                    self.gl.use_program(Some(self.program)); // restore blit for the next item
                    self.gl.delete_texture(tex);
                    let _ = self.egl.destroy_image(self.display, image);
                    continue;
                }
                // Wobbling window: draw the deformed textured mesh (no shadow /
                // frost / corner rounding — square while it jiggles), scissored to
                // each visible clip rect. GL's scissor origin is bottom-left.
                if let Some(v) = mesh {
                    if v.len() == MESH_N * MESH_N {
                        self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.mesh_vbo));
                        let bytes = std::slice::from_raw_parts(
                            v.as_ptr() as *const u8,
                            v.len() * std::mem::size_of::<[f32; 4]>(),
                        );
                        self.gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0, bytes);
                        self.gl.use_program(Some(self.mesh_program));
                        self.gl.uniform_2_f32(self.m_screen.as_ref(), screen_w as f32, screen_h as f32);
                        self.gl.uniform_1_i32(self.m_tex.as_ref(), 0);
                        self.gl.uniform_1_f32(self.m_opacity.as_ref(), opacity);
                        self.gl.bind_vertex_array(Some(self.mesh_vao));
                        self.gl.enable(glow::SCISSOR_TEST);
                        for r in clip {
                            self.gl.scissor(r.x1, screen_h - r.y2, r.width(), r.height());
                            self.gl.draw_elements(
                                glow::TRIANGLES, self.mesh_index_count, glow::UNSIGNED_INT, 0,
                            );
                        }
                        self.gl.disable(glow::SCISSOR_TEST);
                        // Restore the unit-quad VAO + blit program for the next item.
                        self.gl.bind_vertex_array(Some(self.vao));
                        self.gl.use_program(Some(self.program));
                    } else {
                        tracing::warn!(
                            len = v.len(), expected = MESH_N * MESH_N,
                            "wobble mesh vertex count mismatch — skipping window"
                        );
                    }
                    self.gl.delete_texture(tex);
                    let _ = self.egl.destroy_image(self.display, image);
                    continue;
                }
                // Rippling window: per-pixel radial UV refraction via the ripple
                // program (no shadow / frost; corners suppressed), scissored to each
                // clip rect — LINEAR filtering so the offset sampling stays smooth.
                if let Some(rp) = ripple {
                    self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
                    self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
                    self.gl.use_program(Some(self.ripple_program));
                    self.gl.bind_vertex_array(Some(self.vao));
                    self.gl.uniform_2_f32(self.rp_screen.as_ref(), screen_w as f32, screen_h as f32);
                    self.gl.uniform_1_i32(self.rp_tex.as_ref(), 0);
                    self.gl.uniform_4_f32(self.rp_rect.as_ref(), fx, fy, fw, fh);
                    self.gl.uniform_1_f32(self.rp_opacity.as_ref(), opacity);
                    self.gl.uniform_2_f32(self.rp_center.as_ref(), rp.center[0], rp.center[1]);
                    self.gl.uniform_1_f32(self.rp_amp.as_ref(), rp.amp);
                    self.gl.uniform_1_f32(self.rp_wavelength.as_ref(), rp.wavelength);
                    self.gl.uniform_1_f32(self.rp_phase.as_ref(), rp.phase);
                    self.gl.uniform_1_f32(self.rp_r0.as_ref(), rp.r0);
                    self.gl.enable(glow::SCISSOR_TEST);
                    for r in clip {
                        self.gl.scissor(r.x1, screen_h - r.y2, r.width(), r.height());
                        self.gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
                    }
                    self.gl.disable(glow::SCISSOR_TEST);
                    self.gl.use_program(Some(self.program)); // restore blit for the next item
                    self.gl.delete_texture(tex);
                    let _ = self.egl.destroy_image(self.display, image);
                    continue;
                }
                // Waving window: per-pixel traveling-crest UV refraction via the wave
                // program (no shadow / frost; corners suppressed), scissored to each
                // clip rect — LINEAR filtering so the offset sampling stays smooth.
                if let Some(wv) = wave {
                    self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
                    self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
                    self.gl.use_program(Some(self.wave_program));
                    self.gl.bind_vertex_array(Some(self.vao));
                    self.gl.uniform_2_f32(self.wv_screen.as_ref(), screen_w as f32, screen_h as f32);
                    self.gl.uniform_1_i32(self.wv_tex.as_ref(), 0);
                    self.gl.uniform_4_f32(self.wv_rect.as_ref(), fx, fy, fw, fh);
                    self.gl.uniform_1_f32(self.wv_opacity.as_ref(), opacity);
                    self.gl.uniform_1_f32(self.wv_amp.as_ref(), wv.amp);
                    self.gl.uniform_1_f32(self.wv_wavelength.as_ref(), wv.wavelength);
                    self.gl.uniform_1_f32(self.wv_phase.as_ref(), wv.phase);
                    self.gl.uniform_1_i32(self.wv_axis.as_ref(), wv.axis as i32);
                    self.gl.enable(glow::SCISSOR_TEST);
                    for r in clip {
                        self.gl.scissor(r.x1, screen_h - r.y2, r.width(), r.height());
                        self.gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
                    }
                    self.gl.disable(glow::SCISSOR_TEST);
                    self.gl.use_program(Some(self.program)); // restore blit for the next item
                    self.gl.delete_texture(tex);
                    let _ = self.egl.destroy_image(self.display, image);
                    continue;
                }
                // Draining window: per-pixel whirlpool close via the drain program (no
                // shadow / frost; corners suppressed), scissored to each clip rect —
                // LINEAR filtering so the shrink sampling stays smooth.
                if let Some(dr) = drain {
                    self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
                    self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
                    self.gl.use_program(Some(self.drain_program));
                    self.gl.bind_vertex_array(Some(self.vao));
                    self.gl.uniform_2_f32(self.dr_screen.as_ref(), screen_w as f32, screen_h as f32);
                    self.gl.uniform_1_i32(self.dr_tex.as_ref(), 0);
                    self.gl.uniform_4_f32(self.dr_rect.as_ref(), fx, fy, fw, fh);
                    self.gl.uniform_1_f32(self.dr_opacity.as_ref(), opacity);
                    self.gl.uniform_2_f32(self.dr_center.as_ref(), dr.center[0], dr.center[1]);
                    self.gl.uniform_1_f32(self.dr_progress.as_ref(), dr.progress);
                    self.gl.uniform_1_f32(self.dr_turns.as_ref(), dr.turns);
                    self.gl.uniform_1_f32(self.dr_seed.as_ref(), dr.seed);
                    self.gl.enable(glow::SCISSOR_TEST);
                    for r in clip {
                        self.gl.scissor(r.x1, screen_h - r.y2, r.width(), r.height());
                        self.gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
                    }
                    self.gl.disable(glow::SCISSOR_TEST);
                    self.gl.use_program(Some(self.program)); // restore blit for the next item
                    self.gl.delete_texture(tex);
                    let _ = self.egl.destroy_image(self.display, image);
                    continue;
                }
                // Draw only where this window is visible: scissor to each clip rect.
                // GL's scissor origin is bottom-left, so flip Y from our top-left rects.
                self.gl.enable(glow::SCISSOR_TEST);
                for r in clip {
                    // Shadow spans the whole clip rect (its L reaches into the fringe).
                    if shadow {
                        self.gl.scissor(r.x1, screen_h - r.y2, r.width(), r.height());
                        self.gl.use_program(Some(self.shadow_program));
                        self.gl.uniform_1_f32(self.s_corner.as_ref(), corner_radius);
                        // Quad = bounding box of the left+bottom L: extend left and down by the radius.
                        self.gl.uniform_4_f32(
                            self.s_rect.as_ref(),
                            fx - shadow_radius, fy, fw + shadow_radius, fh + shadow_radius,
                        );
                        self.gl.uniform_4_f32(self.s_inner.as_ref(), fx, fy, fw, fh);
                        self.gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
                    }
                    // Frost + blit only cover the window body: clip to rect ∩ window
                    // (a pure shadow-fringe rect has no body — skip it).
                    let Some(br) = r.intersect(&win_rect) else { continue };
                    self.gl.scissor(br.x1, screen_h - br.y2, br.width(), br.height());
                    if let Some(btex) = frost {
                        self.gl.use_program(Some(self.frost_program));
                        self.gl.uniform_2_f32(self.f_screen.as_ref(), screen_w as f32, screen_h as f32);
                        self.gl.uniform_1_f32(self.f_corner.as_ref(), corner_radius);
                        self.gl.uniform_1_i32(self.f_tex.as_ref(), 0);
                        self.gl.bind_texture(glow::TEXTURE_2D, Some(btex));
                        self.gl.uniform_4_f32(self.f_rect.as_ref(), fx, fy, fw, fh);
                        self.gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
                    }
                    self.gl.use_program(Some(self.program)); // back to blit
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                    self.gl.uniform_4_f32(self.u_rect.as_ref(), fx, fy, fw, fh);
                    self.gl.uniform_1_f32(self.u_opacity.as_ref(), opacity);
                    self.gl.uniform_1_f32(self.u_corner.as_ref(), corner_radius);
                    self.gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
                }
                self.gl.disable(glow::SCISSOR_TEST);
                self.gl.delete_texture(tex);
            }
            let _ = self.egl.destroy_image(self.display, image);
        }
        if let Some(hud) = hud {
            self.draw_hud(hud, screen_w, screen_h);
        }
        if let Some(osd) = osd {
            self.draw_osd(osd, screen_w, screen_h);
        }
        unsafe {
            if timer.is_some() {
                self.gl.end_query(glow::TIME_ELAPSED);
            }
            self.gl.bind_vertex_array(None);
        }
        if timer.is_some() {
            self.timer_slot.set(1 - self.timer_slot.get());
            let c = self.timer_count.get();
            if c < 2 {
                self.timer_count.set(c + 1);
            }
        }
        self.egl
            .swap_buffers(self.display, self.surface)
            .map_err(|e| anyhow!("eglSwapBuffers: {e:?}"))?;
        Ok(())
    }
}

impl Drop for GlBackend {
    fn drop(&mut self) {
        let _ = self.egl.make_current(self.display, None, None, None);
        let _ = self.egl.destroy_surface(self.display, self.surface);
        let _ = self.egl.destroy_context(self.display, self.context);
        let _ = self.egl.terminate(self.display);
        unsafe { (self.xlib.XCloseDisplay)(self.xdisplay) };
    }
}
