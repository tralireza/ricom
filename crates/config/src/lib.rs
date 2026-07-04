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
    pub fps: Fps,
    pub burn: Burn,
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
    /// Override the open animation for matching windows (preset name or an
    /// explicit block spec). `None` = use the global `[anim] open`.
    pub open: Option<AnimSel>,
    /// Override the close animation. `None` = use the global `[anim] close`.
    pub close: Option<AnimSel>,
    /// Override the move/resize animation. `None` = use the global `[anim] move`.
    #[serde(rename = "move")]
    pub r#move: Option<AnimSel>,
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
    /// Per-window animation overrides, already expanded from preset/spec.
    pub open: Option<AnimSpec>,
    pub close: Option<AnimSpec>,
    pub r#move: Option<AnimSpec>,
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
    /// Uniform scale-about-centre. `from` is the start factor on open / end on
    /// close (default: `[anim] scale_from`).
    Scale {
        #[serde(default)]
        from: Option<f64>,
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
    /// Noise dissolve with ember front (close). Shader params from `[burn]`.
    Burn,
}

impl Primitive {
    /// Short name for diagnostics.
    pub fn name(&self) -> &'static str {
        match self {
            Primitive::Opacity { .. } => "opacity",
            Primitive::Scale { .. } => "scale",
            Primitive::Translate { .. } => "translate",
            Primitive::Wobble { .. } => "wobble",
            Primitive::Burn => "burn",
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
pub const PRESETS: &[&str] = &["none", "fade", "pop", "slide", "drop", "boing", "burn", "wobble"];

/// Expand a preset name into its block list. `None` if the name is unknown.
fn expand_preset(name: &str) -> Option<Vec<Primitive>> {
    use Primitive::*;
    let opacity = Opacity { from: None, easing: Easing::EaseOut };
    Some(match name {
        "none" => vec![],
        "fade" => vec![opacity],
        "pop" => vec![opacity, Scale { from: None, easing: Easing::EaseOut }],
        "slide" => {
            vec![opacity, Translate { dx: 0.0, dy: 0.0, edge: Some(Edge::Left), easing: Easing::EaseOut }]
        }
        "drop" => {
            // Fall downward (+y) with an accelerating ease-in as it fades.
            vec![opacity, Translate { dx: 0.0, dy: 120.0, edge: None, easing: Easing::EaseIn }]
        }
        "boing" => vec![Wobble { spring: None, friction: None }],
        "burn" => vec![Burn],
        "wobble" => vec![Wobble { spring: None, friction: None }],
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
    /// Open animation (window mapped). Default preset `"pop"`.
    pub open: AnimSel,
    /// Close animation (window unmapped/destroyed). Default preset `"fade"`.
    pub close: AnimSel,
    /// Move/resize animation. Default preset `"wobble"`.
    #[serde(rename = "move")]
    pub r#move: AnimSel,
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
            background: [0.05, 0.05, 0.07],
            corner_radius: 0.0,
            default_opacity: 1.0,
            anim: Anim::default(),
            shadow: Shadow::default(),
            blur: Blur::default(),
            fps: Fps::default(),
            burn: Burn::default(),
            rules: Vec::new(),
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
            open: AnimSel::Preset("pop".into()),
            close: AnimSel::Preset("fade".into()),
            r#move: AnimSel::Preset("wobble".into()),
        }
    }
}

impl Default for Blur {
    fn default() -> Self {
        Blur { enabled: false, passes: 3, radius: 4.0 }
    }
}

impl Default for Fps {
    fn default() -> Self {
        Fps {
            enabled: false,
            hotkey: "Super+Shift+F".to_string(),
            corner: "top-right".to_string(),
            graph: true,
            scale: 1.0,
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
        chg!("anim.open", prev.anim.open, self.anim.open);
        chg!("anim.close", prev.anim.close, self.anim.close);
        chg!("anim.move", prev.anim.r#move, self.anim.r#move);
        chg!("shadow.enabled", prev.shadow.enabled, self.shadow.enabled);
        chg!("shadow.radius", prev.shadow.radius, self.shadow.radius);
        chg!("shadow.strength", prev.shadow.strength, self.shadow.strength);
        chg!("shadow.min_size", prev.shadow.min_size, self.shadow.min_size);
        chg!("blur.enabled", prev.blur.enabled, self.blur.enabled);
        chg!("blur.passes", prev.blur.passes, self.blur.passes);
        chg!("blur.radius", prev.blur.radius, self.blur.radius);
        chg!("fps.enabled", prev.fps.enabled, self.fps.enabled);
        chg!("fps.hotkey", prev.fps.hotkey, self.fps.hotkey);
        chg!("fps.corner", prev.fps.corner, self.fps.corner);
        chg!("fps.graph", prev.fps.graph, self.fps.graph);
        chg!("fps.scale", prev.fps.scale, self.fps.scale);
        chg!("burn.seg_scale", prev.burn.seg_scale, self.burn.seg_scale);
        chg!("burn.ember_width", prev.burn.ember_width, self.burn.ember_width);
        chg!("burn.ember_cool", prev.burn.ember_cool, self.burn.ember_cool);
        chg!("burn.ember_hot", prev.burn.ember_hot, self.burn.ember_hot);
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
        }
        for rule in &self.rules {
            if rule.matcher.matches(w) {
                r.opacity = rule.opacity.or(r.opacity);
                r.blur = rule.blur.or(r.blur);
                r.shadow = rule.shadow.or(r.shadow);
                r.corner_radius = rule.corner_radius.or(r.corner_radius);
                r.unredir = rule.unredir.or(r.unredir);
                r.above = rule.above.or(r.above);
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
mod tests {
    use super::*;

    #[test]
    fn defaults_match_compiled_behaviour() {
        let c = Config::default();
        assert!(c.unredir);
        assert!(c.use_damage);
        assert_eq!(c.background, [0.05, 0.05, 0.07]);
        assert_eq!(c.corner_radius, 0.0);
        assert_eq!(c.anim.duration, 0.2);
        assert_eq!(
            (c.shadow.enabled, c.shadow.radius, c.shadow.strength, c.shadow.min_size),
            (true, 12.0, 0.45, 24)
        );
        assert_eq!((c.blur.enabled, c.blur.passes, c.blur.radius), (false, 3, 4.0));
        assert!(!c.fps.enabled);
        assert_eq!(c.fps.hotkey, "Super+Shift+F");
        assert_eq!(c.fps.corner, "top-right");
        assert!(c.fps.graph);
        assert_eq!(c.fps.scale, 1.0);
        assert_eq!(c.default_opacity, 1.0);
        assert_eq!(c.anim.scale_from, 0.85);
        assert_eq!((c.anim.wobble_spring, c.anim.wobble_friction), (350.0, 14.0));
        assert_eq!(c.anim.open, AnimSel::Preset("pop".into()));
        assert_eq!(c.anim.close, AnimSel::Preset("fade".into()));
        assert_eq!(c.anim.r#move, AnimSel::Preset("wobble".into()));
        assert_eq!((c.burn.seg_scale, c.burn.ember_width), (36.0, 0.07));
        assert_eq!(c.burn.ember_cool, [0.28, 0.02, 0.0]);
        assert_eq!(c.burn.ember_hot, [0.75, 0.22, 0.04]);
        assert!(c.rules.is_empty());
    }

    #[test]
    fn full_toml_parses() {
        let t = r#"
unredir = false
background = [0.1, 0.2, 0.3]
corner_radius = 8.0
[shadow]
enabled = true
radius = 30.0
strength = 0.7
min_size = 40
[blur]
enabled = true
passes = 5
radius = 6.0
[fps]
enabled = true
hotkey = "Control+Alt+P"
corner = "bottom-left"
graph = false
scale = 2.0
[anim]
duration = 0.4
scale_from = 0.7
wobble_spring = 500.0
wobble_friction = 20.0
open = "slide"
close = "burn"
move = "none"
[burn]
seg_scale = 18.0
ember_width = 0.08
ember_cool = [0.3, 0.02, 0.0]
ember_hot = [0.75, 0.25, 0.05]
"#;
        let c: Config = toml::from_str(t).unwrap();
        assert!(!c.unredir);
        assert_eq!(c.background, [0.1, 0.2, 0.3]);
        assert_eq!(c.corner_radius, 8.0);
        assert_eq!(c.anim.duration, 0.4);
        assert_eq!((c.shadow.radius, c.shadow.min_size), (30.0, 40));
        assert_eq!((c.blur.enabled, c.blur.passes, c.blur.radius), (true, 5, 6.0));
        assert!(c.fps.enabled);
        assert_eq!(c.fps.hotkey, "Control+Alt+P");
        assert_eq!(c.fps.corner, "bottom-left");
        assert!(!c.fps.graph);
        assert_eq!(c.fps.scale, 2.0);
        assert_eq!(c.anim.scale_from, 0.7);
        assert_eq!((c.anim.wobble_spring, c.anim.wobble_friction), (500.0, 20.0));
        assert_eq!(c.anim.open, AnimSel::Preset("slide".into()));
        assert_eq!(c.anim.close, AnimSel::Preset("burn".into()));
        assert_eq!(c.anim.r#move, AnimSel::Preset("none".into()));
        assert_eq!((c.burn.seg_scale, c.burn.ember_width), (18.0, 0.08));
        assert_eq!(c.burn.ember_cool, [0.3, 0.02, 0.0]);
        assert_eq!(c.burn.ember_hot, [0.75, 0.25, 0.05]);
    }

    #[test]
    fn partial_toml_fills_defaults() {
        // Override just one shadow field; everything else stays at its default.
        let c: Config = toml::from_str("[shadow]\nradius = 20.0\n").unwrap();
        assert_eq!(c.shadow.radius, 20.0);
        assert_eq!(c.shadow.strength, 0.45); // default
        assert!(c.shadow.enabled); // default
        assert!(c.unredir); // default
        assert_eq!(c.anim.duration, 0.2); // default
    }

    #[test]
    fn unknown_key_errors() {
        assert!(toml::from_str::<Config>("wobble = true\n").is_err());
    }

    #[test]
    fn wrong_type_errors() {
        assert!(toml::from_str::<Config>("unredir = \"yes\"\n").is_err());
    }

    #[test]
    fn diff_reports_changed_fields_only() {
        let a = Config::default();
        assert!(a.diff(&a).is_empty()); // identical -> no changes
        let mut b = Config::default();
        b.shadow.radius = 30.0;
        b.unredir = false;
        let d = b.diff(&a);
        assert_eq!(d.len(), 2);
        assert!(d.iter().any(|s| s.contains("shadow.radius") && s.contains("30.0")));
        assert!(d.iter().any(|s| s.contains("unredir") && s.contains("false")));
    }

    #[test]
    fn toml_roundtrips() {
        let c = Config::default();
        let back: Config = toml::from_str(&c.to_toml()).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn builtin_fullscreen_rule_keeps_opaque_unblurred() {
        let c = Config::default();
        let r = c.resolve(&WindowMatch { fullscreen: true, ..Default::default() });
        assert_eq!(r.opacity, Some(1.0));
        assert_eq!(r.blur, Some(false));
        // non-fullscreen, no user rules -> nothing overridden
        assert_eq!(c.resolve(&WindowMatch::default()), RuleResult::default());
    }

    #[test]
    fn rules_match_and_override() {
        let t = r#"
[[rule]]
match = { class = "mpv" }
opacity = 1.0
blur = false
shadow = false

[[rule]]
match = { title = "Picture-in-Picture" }
corner_radius = 12.0
"#;
        let c: Config = toml::from_str(t).unwrap();
        assert_eq!(c.rules.len(), 2);
        let r = c.resolve(&WindowMatch { class: "mpv".into(), ..Default::default() });
        assert_eq!((r.opacity, r.blur, r.shadow), (Some(1.0), Some(false), Some(false)));
        // substring title match
        let pip = WindowMatch { title: "YouTube - Picture-in-Picture".into(), ..Default::default() };
        assert_eq!(c.resolve(&pip).corner_radius, Some(12.0));
        // no match -> empty
        assert_eq!(
            c.resolve(&WindowMatch { class: "firefox".into(), ..Default::default() }),
            RuleResult::default()
        );
    }

    #[test]
    fn user_rule_overrides_builtin_and_last_wins() {
        let t = r#"
[[rule]]
match = { fullscreen = true }
opacity = 0.5

[[rule]]
match = { class = "mpv" }
opacity = 0.9
"#;
        let c: Config = toml::from_str(t).unwrap();
        // fullscreen non-mpv: the user fullscreen rule overrides the built-in 1.0
        assert_eq!(c.resolve(&WindowMatch { fullscreen: true, ..Default::default() }).opacity, Some(0.5));
        // fullscreen mpv: the later mpv rule wins
        let fs_mpv = WindowMatch { class: "mpv".into(), fullscreen: true, ..Default::default() };
        assert_eq!(c.resolve(&fs_mpv).opacity, Some(0.9));
    }

    #[test]
    fn rule_unknown_field_errors() {
        assert!(toml::from_str::<Config>("[[rule]]\nwobble = true\n").is_err());
    }

    #[test]
    fn above_rule_matches_by_title_substring() {
        let c: Config = toml::from_str(
            "[[rule]]\nmatch = { title = \"intel-gpu-top\" }\nabove = true\n",
        )
        .unwrap();
        // Substring match against the live WM_NAME (e.g. "intel-gpu-top: Intel …").
        let w = WindowMatch { title: "intel-gpu-top: Intel Kabylake".into(), ..Default::default() };
        assert_eq!(c.resolve(&w).above, Some(true));
        // A non-matching window is untouched.
        assert_eq!(c.resolve(&WindowMatch { title: "xterm".into(), ..Default::default() }).above, None);
    }

    // --- Composable animation blocks -----------------------------------------

    const FADE_BLOCKS: [Primitive; 1] = [Primitive::Opacity { from: None, easing: Easing::EaseOut }];

    #[test]
    fn preset_expands_to_blocks() {
        assert_eq!(expand_sel(&AnimSel::Preset("fade".into())).blocks, FADE_BLOCKS);
        let pop = expand_sel(&AnimSel::Preset("pop".into())).blocks;
        assert_eq!(pop.len(), 2);
        assert!(matches!(pop[0], Primitive::Opacity { .. }));
        assert!(matches!(pop[1], Primitive::Scale { .. }));
        assert_eq!(expand_sel(&AnimSel::Preset("burn".into())).blocks, [Primitive::Burn]);
        assert!(expand_sel(&AnimSel::Preset("none".into())).blocks.is_empty());
        // Unknown preset -> empty spec (validate() reports it separately).
        assert!(expand_sel(&AnimSel::Preset("bogus".into())).blocks.is_empty());
    }

    #[test]
    fn explicit_blocks_parse() {
        let t = r#"
[anim.open]
duration = 0.3
blocks = [
  { block = "opacity", easing = "ease-out" },
  { block = "translate", dy = -60.0, easing = "ease-in" },
]
"#;
        let c: Config = toml::from_str(t).unwrap();
        let AnimSel::Spec(s) = &c.anim.open else { panic!("expected explicit spec") };
        assert_eq!(s.duration, Some(0.3));
        assert_eq!(s.blocks.len(), 2);
        assert!(matches!(s.blocks[0], Primitive::Opacity { easing: Easing::EaseOut, .. }));
        let Primitive::Translate { dy, easing, .. } = s.blocks[1] else { panic!("expected translate") };
        assert_eq!(dy, -60.0);
        assert_eq!(easing, Easing::EaseIn);
    }

    #[test]
    fn catchall_rule_sets_close_fade_for_all() {
        let t = r#"
[anim]
close = "burn"

[[rule]]
match = {}
close = "fade"
"#;
        let c: Config = toml::from_str(t).unwrap();
        // Global default is burn, but the empty-match rule overrides every window.
        let any = WindowMatch { class: "anything".into(), ..Default::default() };
        assert_eq!(c.spec_for(&any, Category::Close).blocks, FADE_BLOCKS);
    }

    #[test]
    fn mpv_rule_keeps_burn_while_default_is_fade() {
        let t = r#"
[[rule]]
match = { class = "mpv" }
close = "burn"
"#;
        let c: Config = toml::from_str(t).unwrap();
        // Default close is fade (opacity only)…
        let ff = WindowMatch { class: "firefox".into(), ..Default::default() };
        assert_eq!(c.spec_for(&ff, Category::Close).blocks, FADE_BLOCKS);
        // …but mpv still burns.
        let mpv = WindowMatch { class: "mpv".into(), ..Default::default() };
        assert_eq!(c.spec_for(&mpv, Category::Close).blocks, [Primitive::Burn]);
    }

    #[test]
    fn spec_for_falls_back_to_global_default() {
        let c = Config::default();
        let w = WindowMatch::default();
        assert_eq!(c.spec_for(&w, Category::Open).blocks.len(), 2); // pop = opacity + scale
        assert_eq!(c.spec_for(&w, Category::Close).blocks, FADE_BLOCKS);
        assert_eq!(
            c.spec_for(&w, Category::Move).blocks,
            [Primitive::Wobble { spring: None, friction: None }]
        );
    }

    #[test]
    fn validate_warns_unknown_preset_and_bad_combo() {
        let c: Config = toml::from_str("[anim]\nopen = \"sparkle\"\n").unwrap();
        assert!(c.validate().iter().any(|s| s.contains("anim.open") && s.contains("sparkle")));

        // burn is not a valid `move` block.
        let c: Config = toml::from_str("[anim]\nmove = \"burn\"\n").unwrap();
        assert!(c.validate().iter().any(|s| s.contains("anim.move") && s.contains("burn")));

        // The default config is clean.
        assert!(Config::default().validate().is_empty());
    }
}
