//! ricom configuration: a TOML file mapped to typed settings, with defaults that
//! reproduce the compiled-in behaviour. Pure (serde + a file read), unit-tested
//! like `region`/`wm`. Loaded once at startup and re-read on `SIGHUP` (see the
//! `session` crate). Every field defaults, so a partial — or absent — file still
//! yields a complete [`Config`].

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Top-level configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// `unredir-if-possible`: when `true` (default) a lone fullscreen window
    /// bypasses the compositor and page-flips straight to the display. When
    /// `false`, ricom always composites — even a single fullscreen window.
    pub unredir: bool,
    /// Composite background colour (RGB, `0.0..=1.0`), seen where no window covers.
    pub background: [f32; 3],
    /// Window corner radius in px. `0.0` (default) = square corners.
    pub corner_radius: f32,
    pub fade: Fade,
    pub shadow: Shadow,
    pub blur: Blur,
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
            background: [0.05, 0.05, 0.07],
            corner_radius: 0.0,
            fade: Fade::default(),
            shadow: Shadow::default(),
            blur: Blur::default(),
        }
    }
}

impl Default for Blur {
    fn default() -> Self {
        Blur { enabled: false, passes: 3, radius: 4.0 }
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

    /// Serialise to TOML (for `--print-config`).
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).unwrap_or_else(|e| format!("# serialise error: {e}\n"))
    }

    /// Field-by-field changes from `prev` to `self`, as `"field old→new"` strings
    /// (empty if identical). Used to report what a reload actually changed.
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
        chg!("background", prev.background, self.background);
        chg!("corner_radius", prev.corner_radius, self.corner_radius);
        chg!("fade.enabled", prev.fade.enabled, self.fade.enabled);
        chg!("fade.duration", prev.fade.duration, self.fade.duration);
        chg!("shadow.enabled", prev.shadow.enabled, self.shadow.enabled);
        chg!("shadow.radius", prev.shadow.radius, self.shadow.radius);
        chg!("shadow.strength", prev.shadow.strength, self.shadow.strength);
        chg!("shadow.min_size", prev.shadow.min_size, self.shadow.min_size);
        chg!("blur.enabled", prev.blur.enabled, self.blur.enabled);
        chg!("blur.passes", prev.blur.passes, self.blur.passes);
        chg!("blur.radius", prev.blur.radius, self.blur.radius);
        out
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
        assert_eq!(c.background, [0.05, 0.05, 0.07]);
        assert_eq!(c.corner_radius, 0.0);
        assert_eq!((c.fade.enabled, c.fade.duration), (true, 0.2));
        assert_eq!(
            (c.shadow.enabled, c.shadow.radius, c.shadow.strength, c.shadow.min_size),
            (true, 12.0, 0.45, 24)
        );
        assert_eq!((c.blur.enabled, c.blur.passes, c.blur.radius), (false, 3, 4.0));
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
"#;
        let c: Config = toml::from_str(t).unwrap();
        assert!(!c.unredir);
        assert_eq!(c.background, [0.1, 0.2, 0.3]);
        assert_eq!(c.corner_radius, 8.0);
        assert_eq!((c.fade.enabled, c.fade.duration), (false, 0.4));
        assert_eq!((c.shadow.radius, c.shadow.min_size), (30.0, 40));
        assert_eq!((c.blur.enabled, c.blur.passes, c.blur.radius), (true, 5, 6.0));
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
}
