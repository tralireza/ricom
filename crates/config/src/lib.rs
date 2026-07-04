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
    pub fade: Fade,
    pub shadow: Shadow,
    pub blur: Blur,
    pub fps: Fps,
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
}

impl Match {
    /// True if every specified condition holds for `w`.
    fn matches(&self, w: &WindowMatch) -> bool {
        self.class.as_ref().map_or(true, |c| *c == w.class)
            && self.instance.as_ref().map_or(true, |i| *i == w.instance)
            && self.window_type.as_ref().map_or(true, |t| *t == w.window_type)
            && self.title.as_ref().map_or(true, |t| w.title.contains(t.as_str()))
            && self.fullscreen.map_or(true, |fs| fs == w.fullscreen)
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

/// Window fade-in (on map) / fade-out (on unmap/destroy).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Fade {
    pub enabled: bool,
    /// Fade duration in seconds.
    pub duration: f64,
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
            fade: Fade::default(),
            shadow: Shadow::default(),
            blur: Blur::default(),
            fps: Fps::default(),
            rules: Vec::new(),
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

impl Default for Fade {
    fn default() -> Self {
        Fade { enabled: true, duration: 0.2 }
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
        chg!("fade.enabled", prev.fade.enabled, self.fade.enabled);
        chg!("fade.duration", prev.fade.duration, self.fade.duration);
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
            }
        }
        r
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
        assert_eq!((c.fade.enabled, c.fade.duration), (true, 0.2));
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
        assert!(c.rules.is_empty());
    }

    #[test]
    fn full_toml_parses() {
        let t = r#"
unredir = false
background = [0.1, 0.2, 0.3]
corner_radius = 8.0
[fade]
enabled = false
duration = 0.4
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
"#;
        let c: Config = toml::from_str(t).unwrap();
        assert!(!c.unredir);
        assert_eq!(c.background, [0.1, 0.2, 0.3]);
        assert_eq!(c.corner_radius, 8.0);
        assert_eq!((c.fade.enabled, c.fade.duration), (false, 0.4));
        assert_eq!((c.shadow.radius, c.shadow.min_size), (30.0, 40));
        assert_eq!((c.blur.enabled, c.blur.passes, c.blur.radius), (true, 5, 6.0));
        assert!(c.fps.enabled);
        assert_eq!(c.fps.hotkey, "Control+Alt+P");
        assert_eq!(c.fps.corner, "bottom-left");
        assert!(!c.fps.graph);
        assert_eq!(c.fps.scale, 2.0);
    }

    #[test]
    fn partial_toml_fills_defaults() {
        // Override just one shadow field; everything else stays at its default.
        let c: Config = toml::from_str("[shadow]\nradius = 20.0\n").unwrap();
        assert_eq!(c.shadow.radius, 20.0);
        assert_eq!(c.shadow.strength, 0.45); // default
        assert!(c.shadow.enabled); // default
        assert!(c.unredir); // default
        assert_eq!(c.fade.duration, 0.2); // default
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
}
