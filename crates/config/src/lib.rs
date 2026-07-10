//! ricom configuration: a TOML file mapped to typed settings, with defaults that
//! reproduce the compiled-in behaviour. Pure (serde + a file read), unit-tested
//! like `region`/`wm`. Loaded once at startup and re-read on `SIGHUP` (see the
//! `session` crate). Every field defaults, so a partial — or absent — file still
//! yields a complete [`Config`].
//!
//! ```
//! use config::Config;
//!
//! // A partial file overrides one shadow field; every other field defaults.
//! let cfg: Config = toml::from_str("[shadow]\nradius = 20.0\n").unwrap();
//! assert_eq!(cfg.shadow.radius, 20.0);   // from the file
//! assert_eq!(cfg.shadow.strength, 0.45); // default fills in
//! assert!(cfg.unredir);                  // default
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Top-level configuration.
///
/// Every field defaults (a partial or absent file still yields a complete
/// `Config`), and parsing is strict: an unknown key or a wrong-typed value is an
/// error, so typos surface loudly rather than being silently ignored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// `unredir-if-possible`: when `true` (default) a lone fullscreen window
    /// bypasses the compositor and page-flips straight to the display. When
    /// `false`, ricom always composites — even a single fullscreen window.
    pub unredir: bool,
    /// Repaint only the damaged region each frame (buffer-age partial repaint)
    /// instead of the whole screen. `true` (default); `false` forces full repaints.
    pub use_damage: bool,
    /// Render backend: `"gl"` (EGL + OpenGL) — the only one today, chosen at startup.
    /// `xrender` / `glx` are the roadmap alternatives.
    pub backend: BackendKind,
    /// Composite background colour (RGB, `0.0..=1.0`), seen where no window covers.
    pub background: [f32; 3],
    /// Window corner radius in px. `0.0` (default) = square corners.
    pub corner_radius: f32,
    /// Opacity for windows that set no `_NET_WM_WINDOW_OPACITY` and match no rule
    /// (`0.0..=1.0`; `1.0` = opaque). The bottom layer of the opacity stack.
    pub default_opacity: f64,
    pub anim: Anim,
    pub shadow: Shadow,
    pub blur: Blur,
    pub dim: Dim,
    pub fps: Fps,
    pub osd: Osd,
    pub burn: Burn,
    pub font: Font,
    /// Per-window override rules, applied in order (last match wins per field).
    /// Written as `[[rule]]` tables in TOML.
    #[serde(rename = "rule")]
    pub rules: Vec<Rule>,
}

/// Conditions for a [`Rule`]. Every specified (`Some`) field must match (AND);
/// `class`/`instance`/`window_type` match exactly, `title` matches as a substring.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Match {
    pub class: Option<String>,
    pub instance: Option<String>,
    pub window_type: Option<String>,
    pub title: Option<String>,
    pub fullscreen: Option<bool>,
}

/// A per-window rule: a [`Match`] plus the settings it overrides. Omitted
/// (`None`) overrides leave the global/config value in place.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Rule {
    #[serde(rename = "match")]
    pub matcher: Match,
    pub opacity: Option<f64>,
    pub blur: Option<bool>,
    pub shadow: Option<bool>,
    pub corner_radius: Option<f32>,
    pub unredir: Option<bool>,
    /// Keep matching windows composited on top of all others (always-on-top),
    /// regardless of the X stacking order. `None`/`false` = normal stacking.
    pub above: Option<bool>,
    /// Override inactive-dimming for matching windows: `false` = never dim (stay
    /// bright even when unfocused), `true` = always dim. `None` = follow `[dim]`.
    pub dim: Option<bool>,
    /// Override the open animation for matching windows (preset name or an
    /// explicit block spec). `None` = use the global `[anim] open`.
    pub open: Option<AnimSel>,
    /// Override the close animation. `None` = use the global `[anim] close`.
    pub close: Option<AnimSel>,
    /// Override the move/resize animation. `None` = use the global `[anim] move`.
    #[serde(rename = "move")]
    pub r#move: Option<AnimSel>,
    /// Override the focus effect for matching windows — an in-place effect name
    /// (`wave`/`wobble`/`spin`/`pop`/…/`none`) played when the window gains focus.
    /// `None` = use the global `[anim] focus`.
    pub focus: Option<String>,
}

/// A window's identity + state, matched against [`Rule`]s. Built by `session`
/// (empty strings where a property is absent). Not serialised — a matcher input.
#[derive(Debug, Clone, Default)]
pub struct WindowMatch {
    pub class: String,
    pub instance: String,
    pub window_type: String,
    pub title: String,
    pub fullscreen: bool,
}

/// Net per-window overrides after folding all matching rules. `None` = untouched
/// (the caller uses the global/config default).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RuleResult {
    pub opacity: Option<f64>,
    pub blur: Option<bool>,
    pub shadow: Option<bool>,
    pub corner_radius: Option<f32>,
    pub unredir: Option<bool>,
    pub above: Option<bool>,
    pub dim: Option<bool>,
    /// Per-window animation overrides, already expanded from preset/spec.
    pub open: Option<AnimSpec>,
    pub close: Option<AnimSpec>,
    pub r#move: Option<AnimSpec>,
    /// Resolved focus effect name (`None` = fall back to `[anim] focus`).
    pub focus: Option<String>,
}

impl Match {
    /// True if every specified condition holds for `w`.
    fn matches(&self, w: &WindowMatch) -> bool {
        self.class.as_ref().is_none_or(|c| *c == w.class)
            && self.instance.as_ref().is_none_or(|i| *i == w.instance)
            && self.window_type.as_ref().is_none_or(|t| *t == w.window_type)
            && self.title.as_ref().is_none_or(|t| w.title.contains(t.as_str()))
            && self.fullscreen.is_none_or(|fs| fs == w.fullscreen)
    }
}

/// Background blur behind translucent windows (frosted glass).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Blur {
    /// Off by default — it's extra GPU work; opt in.
    pub enabled: bool,
    /// Dual-Kawase iterations (more = wider/softer blur).
    pub passes: i32,
    /// Sample offset per pass (px); scales the blur reach.
    pub radius: f32,
}

/// Where inactive-dimming learns which window is focused. `Ewmh` reads the root
/// `_NET_ACTIVE_WINDOW` property (needs an EWMH window manager). `X11` tracks X
/// `FocusChange` events — works with a non-EWMH WM, focus-follows-mouse, or any
/// client calling `XSetInputFocus`, but still needs *something* to move focus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FocusSource {
    #[default]
    Ewmh,
    X11,
}

/// Which render backend the compositor uses. `Gl` (EGL + OpenGL) is the only one
/// today; `xrender` / `glx` are roadmap alternatives. An unknown value is a load error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    #[default]
    Gl,
}

/// Inactive-window dimming: unfocused windows fade toward transparent so the
/// focused one stands out. Needs a focus signal (see [`FocusSource`]); with none,
/// it's inert. A per-`[[rule]]` `dim = false` keeps an app bright (e.g. a player).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Dim {
    /// Off by default — opt in.
    pub enabled: bool,
    /// How much to dim an unfocused window: `0.0` = none, `1.0` = fully
    /// transparent. An unfocused window renders at `1 - strength` of its opacity.
    pub strength: f64,
    /// Where to get the focused window from (`ewmh` root property vs `x11`
    /// FocusChange events). Default `ewmh`.
    pub focus: FocusSource,
}

/// OSD toast appear/disappear effect (`[osd] open` / `[osd] close`, independent).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OsdEffect {
    /// Fade in/out in place.
    Fade,
    /// Slide down from the top edge (+ fade).
    Slide,
    /// Scale up from small / back down (+ fade).
    Pop,
    /// A centre line grows to full height / squeezes back (vertical reveal).
    Unroll,
    /// A centre line grows to full width / squeezes back (horizontal reveal).
    #[default]
    Stretch,
}

/// On-screen notification ("toast") shown by `ricomctl notify` — a top-center
/// banner drawn by the compositor via the SDF text engine. `open`/`close` pick
/// how it appears and disappears.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Osd {
    /// Whether `ricomctl notify` shows a toast at all (`true` = on).
    pub enabled: bool,
    /// Default on-screen hold time in seconds when the command omits one.
    pub duration: f64,
    /// Size multiplier on top of the automatic screen-height scaling.
    pub scale: f32,
    /// Effect when the toast appears.
    pub open: OsdEffect,
    /// Effect when the toast disappears.
    pub close: OsdEffect,
    /// Appear-animation duration in seconds (lower = snappier).
    pub in_dur: f64,
    /// Disappear-animation duration in seconds.
    pub out_dur: f64,
    /// Toast a short confirmation for content-less commands (ping/reload/fps toggle).
    pub ack: bool,
    /// Banner background colour (RGBA, `0.0..=1.0`). Set alpha to `0.0` for a
    /// text-only toast (no box); keep `outline` on so the text stays legible.
    pub background: [f32; 4],
    /// Draw a dark outline/halo behind the glyphs so text reads over any backdrop
    /// — essential when `background` is transparent.
    pub outline: bool,
}

/// On-demand FPS / frame-time HUD, toggled by a global hotkey. Drawn by the
/// compositor via the SDF text engine; damage-driven (updates only while the
/// screen is repainting).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Fps {
    /// Whether the HUD is visible at startup (the hotkey toggles it live).
    pub enabled: bool,
    /// Global shortcut that toggles the HUD, e.g. `"Super+Shift+F"`.
    pub hotkey: String,
    /// Screen corner: `"top-left"`, `"top-right"`, `"bottom-left"`, `"bottom-right"`.
    pub corner: String,
    /// Draw the rolling frame-time graph beneath the numbers.
    pub graph: bool,
    /// Extra size multiplier for the HUD, on top of the automatic screen-height
    /// scaling (`1.0` = auto only; e.g. `1.5` = 1.5× larger).
    pub scale: f32,
    /// Outline the HUD text (using the `[font]` outline/shadow style) so it reads
    /// without the panel. `false` (default) = plain text; the panel gives contrast.
    pub outline: bool,
}

/// On-screen text font. ricom rasterises glyphs at runtime from this TrueType
/// font for all on-screen text (the FPS HUD, OSD toasts, `ricomctl notify`). There
/// is no built-in fallback face: if `path` is empty or not a usable `.ttf`, on-screen
/// text is simply **disabled** — the compositor keeps running, it just draws no text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Font {
    /// Path to a TrueType (`.ttf`) font file. Empty disables on-screen text.
    pub path: String,
    /// Global size multiplier for all on-screen text, applied on top of the
    /// automatic screen-height scaling and the per-surface (`[osd]`/`[fps]`) `scale`.
    pub size: f32,
    /// All-around text outline width in px (at 1080p; scales with the surface). `0.0`
    /// = no outline. Lets text read over any backdrop without a background box; each
    /// surface opts in (`[osd] outline`, `[fps] outline`).
    pub outline_width: f32,
    /// Text outline colour (RGB, `0.0..=1.0`).
    pub outline_color: [f32; 3],
    /// Text drop-shadow offset in px (at 1080p, down-right; scales). `0.0` = no shadow.
    pub shadow_offset: f32,
    /// Text drop-shadow colour (RGB, `0.0..=1.0`).
    pub shadow_color: [f32; 3],
    /// Outline direction: `"around"` (default) rings the glyph on all sides; `"drop"`
    /// masks it to the bottom-right only, so the `outline_width` band reads as a tight
    /// drop-shadow that hugs the glyph. Unknown values fall back to `"around"`.
    pub outline_style: String,
}

/// Burn / dissolve close animation: the window disintegrates on animated noise
/// with a glowing ember front. These two size knobs are live-tunable (reload with
/// `SIGHUP`, then close a window to see the change). The on/off switch, duration,
/// and propagation mode are still compiled-in for now (Phase 4 remainder).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Burn {
    /// Segment/hole granularity (shader `u_segscale`): higher = finer, smaller
    /// patches; lower = chunkier. Default `9.0`.
    pub seg_scale: f32,
    /// Ember hot-band half-width (shader `u_ember`): smaller = a thinner, tighter
    /// glowing edge (also crisps the dissolve front). Default `0.13`.
    pub ember_width: f32,
    /// Cooler trailing ember colour (RGB `0.0..=1.0`), at the edge of the glow.
    /// Default `[0.6, 0.05, 0.0]` (dark red). Lower for a moodier smoulder.
    pub ember_cool: [f32; 3],
    /// Hottest leading-edge ember colour (RGB `0.0..=1.0`). Default
    /// `[1.0, 0.8, 0.25]` (bright yellow). Lower for a darker fire.
    pub ember_hot: [f32; 3],
}

/// Which window-lifecycle transition an [`AnimSpec`] applies to. Runtime-only
/// (not serialised); `session` asks for a spec per category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    Open,
    Close,
    Move,
}

/// Easing curve for eased primitives (opacity/scale/translate). Mirrors
/// `wm::anim::Easing`; `session` maps between them (this crate stays `wm`-free).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Easing {
    #[default]
    EaseOut,
    EaseIn,
    Linear,
}

/// Screen edge a `translate` starts from (open) / slides toward (close), when
/// `edge` is given instead of explicit `dx`/`dy`. The offset is sized at runtime
/// to move the window fully off that edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Edge {
    Left,
    Right,
    Top,
    Bottom,
}

/// Which axis/axes a `scale` block affects (always about the window centre).
/// `both` = uniform pop; `x`/`y` = a directional stretch (a centre line growing to
/// full width/height, with content shown squashed into the growing rect).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Axis {
    #[default]
    Both,
    X,
    Y,
}

/// One animation primitive (building block) with its params. Blocks layer in
/// order; `burn` owns alpha and suppresses a co-listed `opacity`. Serialised as a
/// TOML table tagged by `block`, e.g.
/// `{ block = "translate", dy = -60.0, easing = "ease-in" }`.
///
/// This is the single type a new primitive is added to (plus its running twin in
/// `wm` and, if GPU-visible, `backend-gl` wiring).
//
// NB: `deny_unknown_fields` is intentionally absent — serde does not support it
// alongside an internal `tag`. Unknown per-block keys are ignored (the top-level
// `deny_unknown_fields` still catches most typos).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "block", rename_all = "kebab-case")]
pub enum Primitive {
    /// Opacity fade. `from` overrides the start alpha (else 0 on open / current
    /// on close).
    Opacity {
        #[serde(default)]
        from: Option<f64>,
        #[serde(default)]
        easing: Easing,
    },
    /// Scale-about-centre. `from` is the start factor on open / end on close
    /// (default: `[anim] scale_from`). `axis` restricts it to one dimension:
    /// `x`/`y` give a directional stretch (line ↔ full width/height); `both`
    /// (default) is the uniform pop.
    Scale {
        #[serde(default)]
        from: Option<f64>,
        #[serde(default)]
        axis: Axis,
        #[serde(default)]
        easing: Easing,
    },
    /// Pixel translate: either explicit `dx`/`dy` (offset away from rest), or an
    /// `edge` the window slides from (open) / to (close).
    Translate {
        #[serde(default)]
        dx: f32,
        #[serde(default)]
        dy: f32,
        #[serde(default)]
        edge: Option<Edge>,
        #[serde(default)]
        easing: Easing,
    },
    /// Spring-mesh wobble (move category). Falls back to `[anim]` spring/friction.
    Wobble {
        #[serde(default)]
        spring: Option<f32>,
        #[serde(default)]
        friction: Option<f32>,
    },
    /// Traveling sinusoidal wave (per-pixel refraction; open/close/animate/focus).
    /// Params fall back to the `[anim] wave_*` defaults; `axis` is the crest's travel
    /// direction (`x` displaces V, `y` displaces U). Settles over `duration` seconds.
    Wave {
        #[serde(default)]
        amplitude: Option<f32>,
        #[serde(default)]
        wavelength: Option<f32>,
        #[serde(default)]
        speed: Option<f32>,
        #[serde(default)]
        axis: Axis,
        #[serde(default)]
        duration: Option<f32>,
    },
    /// Radial water-refraction ripple (per-pixel; open/close/animate/focus). Rings
    /// expand from the window centre, spreading + ringing down. Params fall back to
    /// the `[anim] ripple_*` defaults.
    Ripple {
        #[serde(default)]
        amplitude: Option<f32>,
        #[serde(default)]
        wavelength: Option<f32>,
        #[serde(default)]
        speed: Option<f32>,
        #[serde(default)]
        r0: Option<f32>,
        #[serde(default)]
        duration: Option<f32>,
    },
    /// Rotate about the window centre by `degrees` (default 360, a full turn):
    /// spin in on open (rotated → 0) / spin out on close (0 → rotated). GPU primitive.
    Spin {
        #[serde(default)]
        degrees: Option<f32>,
        #[serde(default)]
        easing: Easing,
    },
    /// Noise dissolve with ember front (close). Shader params from `[burn]`.
    Burn,
    /// Whirlpool drain (close): content spirals into a vanishing point at the centre
    /// and fades. Params fall back to the `[anim] drain_*` defaults.
    Drain {
        #[serde(default)]
        turns: Option<f32>,
        #[serde(default)]
        duration: Option<f32>,
    },
}

impl Primitive {
    /// Short name for diagnostics.
    pub fn name(&self) -> &'static str {
        match self {
            Primitive::Opacity { .. } => "opacity",
            Primitive::Scale { .. } => "scale",
            Primitive::Translate { .. } => "translate",
            Primitive::Wobble { .. } => "wobble",
            Primitive::Wave { .. } => "wave",
            Primitive::Ripple { .. } => "ripple",
            Primitive::Spin { .. } => "spin",
            Primitive::Burn => "burn",
            Primitive::Drain { .. } => "drain",
        }
    }

    /// Whether this block is meaningful for `cat` (used by [`Config::validate`]).
    /// open/close accept any transform; move only geometry blocks.
    fn valid_for(&self, cat: Category) -> bool {
        match cat {
            Category::Open | Category::Close => true,
            Category::Move => matches!(self, Primitive::Wobble { .. } | Primitive::Translate { .. }),
        }
    }
}

/// An ordered set of primitive blocks for one category. `duration` (seconds)
/// applies to the eased blocks; `None` falls back to `[anim] duration`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnimSpec {
    pub duration: Option<f64>,
    pub blocks: Vec<Primitive>,
}

/// Per-category selection: either a preset name (`open = "pop"`) or an explicit
/// [`AnimSpec`] table. Normalised to an `AnimSpec` at resolve time via
/// [`expand_sel`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnimSel {
    Preset(String),
    Spec(AnimSpec),
}

/// Known preset names, for diagnostics + docs.
pub const PRESETS: &[&str] = &[
    "none", "fade", "pop", "slide", "drop", "boing", "burn", "drain", "wobble", "stretch", "unroll", "minimize",
    "spin", "wave", "ripple",
];

/// In-place effect names valid for `[anim] focus` / `[[rule]] focus` (and the
/// `ricomctl animate` vocabulary), plus `"none"`.
pub const FOCUS_EFFECTS: &[&str] =
    &["none", "spin", "pop", "stretch", "unroll", "slide", "wobble", "wave", "ripple", "reset"];

/// Expand a preset name into its block list. `None` if the name is unknown.
fn expand_preset(name: &str) -> Option<Vec<Primitive>> {
    use Primitive::*;
    let opacity = Opacity { from: None, easing: Easing::EaseOut };
    Some(match name {
        "none" => vec![],
        "fade" => vec![opacity],
        "pop" => vec![opacity, Scale { from: None, axis: Axis::Both, easing: Easing::EaseOut }],
        "slide" => {
            vec![opacity, Translate { dx: 0.0, dy: 0.0, edge: Some(Edge::Left), easing: Easing::EaseOut }]
        }
        "drop" => {
            // Fall downward (+y). ease-out front-loads the travel so the fall reads
            // as motion before the fade finishes (ease-in read mostly as a fade).
            vec![opacity, Translate { dx: 0.0, dy: 140.0, edge: None, easing: Easing::EaseOut }]
        }
        "boing" => vec![Wobble { spring: None, friction: None }],
        "burn" => vec![Burn],
        "drain" => vec![Drain { turns: None, duration: None }],
        "wobble" => vec![Wobble { spring: None, friction: None }],
        // Directional stretch: a centre line grows to full width (x) / height (y),
        // content shown squashed throughout. Opaque (no opacity block) by design.
        "stretch" => vec![Scale { from: Some(0.0), axis: Axis::X, easing: Easing::EaseOut }],
        "unroll" => vec![Scale { from: Some(0.0), axis: Axis::Y, easing: Easing::EaseOut }],
        // Minimize: shrink to a point (both axes → 0) while sliding off the bottom
        // edge — a simplified "genie" (no curved warp; that would need a mesh
        // primitive). Stays opaque; the scale-to-0 collapse drives completion.
        "minimize" => vec![
            Scale { from: Some(0.0), axis: Axis::Both, easing: Easing::EaseIn },
            Translate { dx: 0.0, dy: 0.0, edge: Some(Edge::Bottom), easing: Easing::EaseIn },
        ],
        // Rotate in/out about the centre (with a fade); half-turn by default.
        "spin" => vec![opacity, Spin { degrees: None, easing: Easing::EaseOut }],
        // Traveling ripple: one-shot that rings down (see `[anim] wave_*`).
        "wave" => vec![Wave { amplitude: None, wavelength: None, speed: None, axis: Axis::X, duration: None }],
        // Radial water-refraction ripple (per-pixel; see `[anim] ripple_*`).
        "ripple" => vec![Ripple { amplitude: None, wavelength: None, speed: None, r0: None, duration: None }],
        _ => return None,
    })
}

/// Normalise a selection to a concrete [`AnimSpec`]. An unknown preset yields an
/// empty spec (no animation); [`Config::validate`] surfaces the warning.
pub fn expand_sel(sel: &AnimSel) -> AnimSpec {
    match sel {
        AnimSel::Preset(name) => AnimSpec { duration: None, blocks: expand_preset(name).unwrap_or_default() },
        AnimSel::Spec(spec) => spec.clone(),
    }
}

/// Build a one-block [`AnimSpec`] from an effect name + `(key, value)` params — the
/// live-`set`-with-params path. Maps the high-level names to their `Primitive` block
/// (`pop`/`stretch`/`unroll` → `scale` with an implicit `axis`; `slide` → `translate`;
/// others = the block of the same name), routes a `duration` param onto the block for the
/// per-pixel effects that own one (wave/ripple/drain) or onto the spec otherwise, and
/// deserialises the rest through the `#[serde(tag = "block")]` mapping (so value types are
/// validated). Unknown *keys* are the caller's job to reject first (via
/// `proto::effect_params`) — serde on `Primitive` does not `deny_unknown_fields`.
pub fn anim_spec_from(effect: &str, params: &[(String, String)]) -> Result<AnimSpec, String> {
    // High-level name → (block tag, implicit axis for the scale-based ones).
    let (block, axis) = match effect {
        "pop" => ("scale", Some("both")),
        "stretch" => ("scale", Some("x")),
        "unroll" => ("scale", Some("y")),
        "slide" => ("translate", None),
        other => (other, None),
    };
    // `duration` is a block field only for the per-pixel effects; elsewhere it's the
    // category-level (AnimSpec) duration.
    let block_has_duration = matches!(effect, "wave" | "ripple" | "drain");
    let mut table = toml::value::Table::new();
    table.insert("block".into(), toml::Value::String(block.to_string()));
    if let Some(a) = axis {
        table.insert("axis".into(), toml::Value::String(a.to_string()));
    }
    let mut duration = None;
    for (k, v) in params {
        if k == "duration" && !block_has_duration {
            duration = Some(
                v.parse::<f64>().map_err(|_| format!("param 'duration' wants a number, got '{v}'"))?,
            );
            continue;
        }
        // Numeric-looking values become TOML floats (the f32 fields); the rest stay
        // strings (the `axis` / `easing` enums).
        let val = v.parse::<f64>().map(toml::Value::Float).unwrap_or_else(|_| toml::Value::String(v.clone()));
        table.insert(k.clone(), val);
    }
    let block: Primitive = toml::Value::Table(table)
        .try_into()
        .map_err(|e| format!("bad params for '{effect}': {e}"))?;
    Ok(AnimSpec { duration, blocks: vec![block] })
}

/// Open / close / move animation selection + shared primitive param defaults.
/// Replaces the old `[fade]` and `[animation]` blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Anim {
    /// Shared animation duration in seconds (opacity/scale/translate).
    pub duration: f64,
    /// Default scale factor for the `scale` primitive (open start / close end).
    pub scale_from: f64,
    /// Default wobble spring stiffness (pull toward target geometry).
    pub wobble_spring: f32,
    /// Default wobble velocity damping.
    pub wobble_friction: f32,
    /// Default `wave` amplitude (UV — fraction of the perpendicular dimension).
    pub wave_amplitude: f32,
    /// Default `wave` wavelength as a fraction of the travel axis (`1.0` = one full
    /// cycle across; `0.5` = two). Cycles across ≈ `1.0 / wave_wavelength`.
    pub wave_wavelength: f32,
    /// Default `wave` travel speed (cycles per second).
    pub wave_speed: f32,
    /// Default `wave` settle time in seconds (how long it lasts; `<= 0` loops).
    pub wave_duration: f32,
    /// Default `ripple` peak radial UV displacement (aspect-corrected units).
    pub ripple_amplitude: f32,
    /// Default `ripple` ring spacing (fraction of the aspect-corrected radius).
    pub ripple_wavelength: f32,
    /// Default `ripple` phase speed (cycles/second; rings expand outward).
    pub ripple_speed: f32,
    /// Default `ripple` spread constant (large centre, faint rim).
    pub ripple_r0: f32,
    /// Default `ripple` settle time in seconds (how long it lasts; `<= 0` loops).
    pub ripple_duration: f32,
    /// Default `drain` swirl rotations at full progress (whirlpool close).
    pub drain_turns: f32,
    /// Default `drain` close duration in seconds (progress 0→1, then reaped).
    pub drain_duration: f32,
    /// Open animation (window mapped). Default preset `"pop"`.
    pub open: AnimSel,
    /// Close animation (window unmapped/destroyed). Default preset `"fade"`.
    pub close: AnimSel,
    /// Move/resize animation. Default preset `"wobble"`.
    #[serde(rename = "move")]
    pub r#move: AnimSel,
    /// Focus effect: an in-place effect name (`wave`/`wobble`/`spin`/`pop`/…) played
    /// when a window gains focus; `"none"` (default) = off. Per-window override via a
    /// `[[rule]] focus = …`.
    pub focus: String,
}

impl Anim {
    /// The selection for a category.
    fn sel(&self, cat: Category) -> &AnimSel {
        match cat {
            Category::Open => &self.open,
            Category::Close => &self.close,
            Category::Move => &self.r#move,
        }
    }
}

/// Left+bottom drop shadows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Shadow {
    pub enabled: bool,
    /// Falloff distance to the left/bottom (px).
    pub radius: f32,
    /// Peak shadow alpha (`0.0..=1.0`).
    pub strength: f32,
    /// Skip shadows for windows smaller than this (px) — avoids tiny specks.
    pub min_size: i32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            unredir: true,
            use_damage: true,
            backend: BackendKind::default(),
            background: [0.05, 0.05, 0.07],
            corner_radius: 0.0,
            default_opacity: 1.0,
            anim: Anim::default(),
            shadow: Shadow::default(),
            blur: Blur::default(),
            dim: Dim::default(),
            fps: Fps::default(),
            osd: Osd::default(),
            burn: Burn::default(),
            font: Font::default(),
            rules: Vec::new(),
        }
    }
}

impl Default for Font {
    fn default() -> Self {
        // A commonly-present monospace TTF; on i7 this is the same face the old
        // baked atlas used. Point it at any `.ttf` you like, or clear it to disable
        // on-screen text. A missing/invalid path degrades to "text disabled".
        Font {
            path: "/usr/share/fonts/liberation-mono/LiberationMono-Regular.ttf".to_string(),
            size: 1.0,
            outline_width: 1.5,
            outline_color: [0.0, 0.0, 0.0],
            shadow_offset: 0.0,
            shadow_color: [0.0, 0.0, 0.0],
            outline_style: "around".to_string(),
        }
    }
}

impl Default for Burn {
    fn default() -> Self {
        // "Deep smoulder": fine segments, a thin-but-present ember band, and a
        // dark-maroon → burnt-orange ramp (no bright yellow). Dialed in by eye.
        Burn {
            seg_scale: 36.0,
            ember_width: 0.07,
            ember_cool: [0.28, 0.02, 0.0],
            ember_hot: [0.75, 0.22, 0.04],
        }
    }
}

impl Default for Anim {
    fn default() -> Self {
        Anim {
            duration: 0.2,
            scale_from: 0.85,
            wobble_spring: 350.0,
            wobble_friction: 14.0,
            wave_amplitude: 0.04,
            wave_wavelength: 0.5,
            wave_speed: 1.5,
            wave_duration: 1.5,
            ripple_amplitude: 0.08,
            ripple_wavelength: 0.18,
            ripple_speed: 1.2,
            ripple_r0: 0.12,
            ripple_duration: 2.5,
            drain_turns: 1.5,
            drain_duration: 0.6,
            open: AnimSel::Preset("pop".into()),
            close: AnimSel::Preset("fade".into()),
            r#move: AnimSel::Preset("wobble".into()),
            focus: "none".into(),
        }
    }
}

impl Default for Blur {
    fn default() -> Self {
        Blur { enabled: false, passes: 3, radius: 4.0 }
    }
}

impl Default for Dim {
    fn default() -> Self {
        Dim { enabled: false, strength: 0.3, focus: FocusSource::Ewmh }
    }
}

impl Default for Osd {
    fn default() -> Self {
        Osd {
            enabled: true,
            duration: 2.5,
            scale: 1.0,
            open: OsdEffect::Stretch,
            close: OsdEffect::Stretch,
            in_dur: 0.06,
            out_dur: 0.08,
            ack: true,
            background: [0.05, 0.05, 0.07, 0.88],
            outline: true,
        }
    }
}

impl Default for Fps {
    fn default() -> Self {
        Fps {
            enabled: false,
            hotkey: "Super+Shift+F".to_string(),
            corner: "bottom-left".to_string(),
            graph: true,
            scale: 1.0,
            outline: false,
        }
    }
}

impl Default for Shadow {
    fn default() -> Self {
        Shadow { enabled: true, radius: 12.0, strength: 0.45, min_size: 24 }
    }
}

impl Config {
    /// Load from `explicit` if given, else the default location
    /// ([`default_path`]). An explicit path that can't be read or parsed is an
    /// error. A *missing* default-location file yields [`Config::default`], but a
    /// present-but-invalid one is an error (so typos surface loudly).
    pub fn load(explicit: Option<&Path>) -> Result<Config> {
        match explicit {
            Some(p) => Self::from_file(p),
            None => match default_path() {
                Some(p) if p.exists() => Self::from_file(&p),
                _ => Ok(Config::default()),
            },
        }
    }

    fn from_file(path: &Path) -> Result<Config> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing config {}", path.display()))
    }

    /// Serialise to TOML (for `--print-config`). Round-trips: parsing the output
    /// back yields an equal [`Config`].
    ///
    /// ```
    /// use config::Config;
    ///
    /// let cfg = Config::default();
    /// let back: Config = toml::from_str(&cfg.to_toml()).unwrap();
    /// assert_eq!(cfg, back);
    /// ```
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).unwrap_or_else(|e| format!("# serialise error: {e}\n"))
    }

    /// Field-by-field changes from `prev` to `self`, as `"field old→new"` strings
    /// (empty if identical). Used to report what a reload actually changed.
    ///
    /// ```
    /// use config::Config;
    ///
    /// let old = Config::default();
    /// let mut new = old.clone();
    /// assert!(new.diff(&old).is_empty()); // identical → nothing changed
    ///
    /// new.corner_radius = 8.0;
    /// let changes = new.diff(&old);
    /// assert_eq!(changes.len(), 1);
    /// assert!(changes[0].contains("corner_radius"));
    /// ```
    pub fn diff(&self, prev: &Config) -> Vec<String> {
        let mut out = Vec::new();
        macro_rules! chg {
            ($label:expr, $old:expr, $new:expr) => {
                if $old != $new {
                    out.push(format!("{} {:?}→{:?}", $label, $old, $new));
                }
            };
        }
        chg!("unredir", prev.unredir, self.unredir);
        chg!("use_damage", prev.use_damage, self.use_damage);
        chg!("background", prev.background, self.background);
        chg!("corner_radius", prev.corner_radius, self.corner_radius);
        chg!("default_opacity", prev.default_opacity, self.default_opacity);
        chg!("anim.duration", prev.anim.duration, self.anim.duration);
        chg!("anim.scale_from", prev.anim.scale_from, self.anim.scale_from);
        chg!("anim.wobble_spring", prev.anim.wobble_spring, self.anim.wobble_spring);
        chg!("anim.wobble_friction", prev.anim.wobble_friction, self.anim.wobble_friction);
        chg!("anim.wave_amplitude", prev.anim.wave_amplitude, self.anim.wave_amplitude);
        chg!("anim.wave_wavelength", prev.anim.wave_wavelength, self.anim.wave_wavelength);
        chg!("anim.wave_speed", prev.anim.wave_speed, self.anim.wave_speed);
        chg!("anim.wave_duration", prev.anim.wave_duration, self.anim.wave_duration);
        chg!("anim.ripple_amplitude", prev.anim.ripple_amplitude, self.anim.ripple_amplitude);
        chg!("anim.ripple_wavelength", prev.anim.ripple_wavelength, self.anim.ripple_wavelength);
        chg!("anim.ripple_speed", prev.anim.ripple_speed, self.anim.ripple_speed);
        chg!("anim.ripple_r0", prev.anim.ripple_r0, self.anim.ripple_r0);
        chg!("anim.ripple_duration", prev.anim.ripple_duration, self.anim.ripple_duration);
        chg!("anim.drain_turns", prev.anim.drain_turns, self.anim.drain_turns);
        chg!("anim.drain_duration", prev.anim.drain_duration, self.anim.drain_duration);
        chg!("anim.open", prev.anim.open, self.anim.open);
        chg!("anim.close", prev.anim.close, self.anim.close);
        chg!("anim.move", prev.anim.r#move, self.anim.r#move);
        chg!("anim.focus", prev.anim.focus, self.anim.focus);
        chg!("shadow.enabled", prev.shadow.enabled, self.shadow.enabled);
        chg!("shadow.radius", prev.shadow.radius, self.shadow.radius);
        chg!("shadow.strength", prev.shadow.strength, self.shadow.strength);
        chg!("shadow.min_size", prev.shadow.min_size, self.shadow.min_size);
        chg!("blur.enabled", prev.blur.enabled, self.blur.enabled);
        chg!("blur.passes", prev.blur.passes, self.blur.passes);
        chg!("blur.radius", prev.blur.radius, self.blur.radius);
        chg!("dim.enabled", prev.dim.enabled, self.dim.enabled);
        chg!("dim.strength", prev.dim.strength, self.dim.strength);
        chg!("dim.focus", prev.dim.focus, self.dim.focus);
        chg!("fps.enabled", prev.fps.enabled, self.fps.enabled);
        chg!("fps.hotkey", prev.fps.hotkey, self.fps.hotkey);
        chg!("fps.corner", prev.fps.corner, self.fps.corner);
        chg!("fps.graph", prev.fps.graph, self.fps.graph);
        chg!("fps.scale", prev.fps.scale, self.fps.scale);
        chg!("fps.outline", prev.fps.outline, self.fps.outline);
        chg!("osd.enabled", prev.osd.enabled, self.osd.enabled);
        chg!("osd.duration", prev.osd.duration, self.osd.duration);
        chg!("osd.scale", prev.osd.scale, self.osd.scale);
        chg!("osd.open", prev.osd.open, self.osd.open);
        chg!("osd.close", prev.osd.close, self.osd.close);
        chg!("osd.in_dur", prev.osd.in_dur, self.osd.in_dur);
        chg!("osd.out_dur", prev.osd.out_dur, self.osd.out_dur);
        chg!("osd.ack", prev.osd.ack, self.osd.ack);
        chg!("osd.background", prev.osd.background, self.osd.background);
        chg!("osd.outline", prev.osd.outline, self.osd.outline);
        chg!("burn.seg_scale", prev.burn.seg_scale, self.burn.seg_scale);
        chg!("burn.ember_width", prev.burn.ember_width, self.burn.ember_width);
        chg!("burn.ember_cool", prev.burn.ember_cool, self.burn.ember_cool);
        chg!("burn.ember_hot", prev.burn.ember_hot, self.burn.ember_hot);
        chg!("font.path", prev.font.path, self.font.path);
        chg!("font.size", prev.font.size, self.font.size);
        chg!("font.outline_width", prev.font.outline_width, self.font.outline_width);
        chg!("font.outline_color", prev.font.outline_color, self.font.outline_color);
        chg!("font.shadow_offset", prev.font.shadow_offset, self.font.shadow_offset);
        chg!("font.shadow_color", prev.font.shadow_color, self.font.shadow_color);
        chg!("font.outline_style", prev.font.outline_style, self.font.outline_style);
        if prev.rules != self.rules {
            out.push(format!("rules {}→{}", prev.rules.len(), self.rules.len()));
        }
        out
    }

    /// Fold the built-in fullscreen default rule and the user [`rules`](Config::rules)
    /// (in order, last match wins per field) into the net overrides for a window.
    ///
    /// ```
    /// use config::{Config, WindowMatch};
    ///
    /// let cfg: Config = toml::from_str(r#"
    /// [[rule]]
    /// match = { class = "mpv" }
    /// opacity = 0.9
    /// "#).unwrap();
    ///
    /// // The mpv rule applies → opacity overridden.
    /// let mpv = WindowMatch { class: "mpv".into(), ..Default::default() };
    /// assert_eq!(cfg.resolve(&mpv).opacity, Some(0.9));
    ///
    /// // A fullscreen window matching no user rule keeps the built-in opaque default.
    /// let fs = WindowMatch { fullscreen: true, ..Default::default() };
    /// assert_eq!(cfg.resolve(&fs).opacity, Some(1.0));
    /// ```
    pub fn resolve(&self, w: &WindowMatch) -> RuleResult {
        let mut r = RuleResult::default();
        // Built-in default: a fullscreen window stays opaque and unblurred so
        // video/games aren't dimmed. User rules below may override it.
        if w.fullscreen {
            r.opacity = Some(1.0);
            r.blur = Some(false);
            r.dim = Some(false); // don't dim a fullscreen window (video/games)
        }
        for rule in &self.rules {
            if rule.matcher.matches(w) {
                r.opacity = rule.opacity.or(r.opacity);
                r.blur = rule.blur.or(r.blur);
                r.shadow = rule.shadow.or(r.shadow);
                r.corner_radius = rule.corner_radius.or(r.corner_radius);
                r.unredir = rule.unredir.or(r.unredir);
                r.above = rule.above.or(r.above);
                r.dim = rule.dim.or(r.dim);
                // Anim overrides: last matching rule that specifies one wins.
                if let Some(s) = &rule.open {
                    r.open = Some(expand_sel(s));
                }
                if let Some(s) = &rule.close {
                    r.close = Some(expand_sel(s));
                }
                if let Some(s) = &rule.r#move {
                    r.r#move = Some(expand_sel(s));
                }
                r.focus = rule.focus.clone().or(r.focus);
            }
        }
        r
    }

    /// The effective animation spec for a window in `cat`: a matching rule's
    /// override, else the global `[anim]` default. Always concrete (preset names
    /// already expanded), so `session` can drive `wm` directly.
    pub fn spec_for(&self, w: &WindowMatch, cat: Category) -> AnimSpec {
        let over = match cat {
            Category::Open => self.resolve(w).open,
            Category::Close => self.resolve(w).close,
            Category::Move => self.resolve(w).r#move,
        };
        over.unwrap_or_else(|| expand_sel(self.anim.sel(cat)))
    }

    /// Non-fatal config problems to log at load (never rejects — parsing already
    /// rejects typos/unknown keys). Currently: unknown preset names, and blocks
    /// used in a category they don't fit (e.g. `burn` on `move`).
    pub fn validate(&self) -> Vec<String> {
        let mut warns = Vec::new();
        for (label, sel, cat) in [
            ("anim.open", &self.anim.open, Category::Open),
            ("anim.close", &self.anim.close, Category::Close),
            ("anim.move", &self.anim.r#move, Category::Move),
        ] {
            validate_sel(label, sel, cat, &mut warns);
        }
        if !FOCUS_EFFECTS.contains(&self.anim.focus.as_str()) {
            warns.push(format!("anim.focus: unknown effect {:?} (no focus animation)", self.anim.focus));
        }
        // Font: flag the one static case here — an empty path means on-screen text is
        // off. The authoritative "path not found / not a usable TTF" warning is emitted
        // by the backend at load time (it parses the font via fontdue); this crate stays
        // pure (no filesystem probing in `validate`).
        if self.font.path.is_empty() {
            warns.push("font.path is empty — on-screen text (HUD/OSD/notify) is disabled".to_string());
        }
        if !matches!(self.font.outline_style.as_str(), "around" | "drop") {
            warns.push(format!(
                "font.outline_style: unknown value {:?} (using \"around\")",
                self.font.outline_style
            ));
        }
        for (i, rule) in self.rules.iter().enumerate() {
            if let Some(s) = &rule.open {
                validate_sel(&format!("rule[{i}].open"), s, Category::Open, &mut warns);
            }
            if let Some(s) = &rule.close {
                validate_sel(&format!("rule[{i}].close"), s, Category::Close, &mut warns);
            }
            if let Some(s) = &rule.r#move {
                validate_sel(&format!("rule[{i}].move"), s, Category::Move, &mut warns);
            }
            if let Some(f) = &rule.focus
                && !FOCUS_EFFECTS.contains(&f.as_str())
            {
                warns.push(format!("rule[{i}].focus: unknown effect {f:?} (no focus animation)"));
            }
        }
        warns
    }
}

/// Warn on an unknown preset name or category-invalid blocks in one selection.
fn validate_sel(label: &str, sel: &AnimSel, cat: Category, warns: &mut Vec<String>) {
    if let AnimSel::Preset(name) = sel
        && expand_preset(name).is_none()
    {
        warns.push(format!("{label}: unknown preset {name:?} (no animation)"));
        return;
    }
    for b in expand_sel(sel).blocks {
        if !b.valid_for(cat) {
            warns.push(format!("{label}: block \"{}\" is not valid for {cat:?}", b.name()));
        }
    }
}

/// Default config path: `$XDG_CONFIG_HOME/ricom/ricom.toml`, else
/// `$HOME/.config/ricom/ricom.toml`. `None` if neither variable is set.
pub fn default_path() -> Option<PathBuf> {
    if let Some(x) = std::env::var("XDG_CONFIG_HOME").ok().filter(|s| !s.is_empty()) {
        return Some(PathBuf::from(x).join("ricom").join("ricom.toml"));
    }
    std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".config").join("ricom").join("ricom.toml"))
}

#[cfg(test)]
mod tests;
