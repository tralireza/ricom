//! ricom control-channel wire protocol.
//!
//! Shared by the compositor (server, the `session` crate) and `ricomctl` (the
//! client). One JSON request line → one JSON reply line, one connection per
//! command (newline-delimited JSON, "NDJSON"). Keeping the wire types in a tiny
//! shared crate means client and server agree on the format at compile time — a
//! client built against a different `proto` fails to decode rather than
//! silently misbehaving.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Wire-protocol version. Bump on any incompatible `Command`/`Reply` change.
pub const PROTOCOL_VERSION: u32 = 3;

/// Raw X window id (mirrors `wm::WindowId`).
pub type WinId = u32;

/// A control command sent from `ricomctl` to a running `ricom`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Command {
    /// Liveness check → `Reply::Text(banner)`.
    Ping,
    /// Re-read + apply the config file (same as `SIGHUP`; Linux-only server-side).
    Reload,
    /// Toggle the FPS HUD.
    FpsToggle,
    /// List tracked windows.
    List,
    /// Detailed info for one window.
    Inspect { win: WinId },
    /// Show an on-screen notification (OSD toast). `timeout_ms` overrides the
    /// configured default hold time.
    Notify { text: String, timeout_ms: Option<u32> },
    /// Show ricom's version as an OSD toast; the reply also carries the banner text.
    Version,
    /// Play a one-shot, self-restoring animation on one window — the transform
    /// effects with no external X trigger (`spin`/`pop`/`stretch`/`unroll`/`slide`/
    /// `wobble`/`wave`/`ripple`, or `reset` to snap back). The server validates
    /// `effect` and each param.
    Animate {
        win: WinId,
        effect: String,
        /// Per-effect parameter overrides as `(key, value)` pairs (e.g.
        /// `("amplitude", "0.12")`); the server types + validates them per effect.
        /// Empty ⇒ use the configured defaults (`#[serde(default)]`, so older
        /// clients that omit the field still decode).
        #[serde(default)]
        params: Vec<(String, String)>,
    },
    /// Live-select the effect (preset name + optional params) for a transition category
    /// (`open`/`close`/`move`/`focus`) — session-only (a `Reload`/SIGHUP reverts). The
    /// server validates the category, effect, and each param.
    SetAnim {
        /// `"open"` | `"close"` | `"move"` | `"focus"`.
        category: String,
        /// Preset / effect name (e.g. `pop`, `drain`, `wave`).
        effect: String,
        /// Optional `(key, value)` param overrides (empty ⇒ a bare preset).
        #[serde(default)]
        params: Vec<(String, String)>,
    },
}

/// Every effect/preset name `animate` and `set` accept — for help + validation.
pub const EFFECTS: &[&str] =
    &["spin", "pop", "stretch", "unroll", "slide", "wobble", "wave", "ripple", "drain", "reset"];

/// Per-effect parameter keys + one-line descriptions, shared by `ricomctl` (help) and
/// the compositor (param validation) so the two can't drift. Keys mirror the config
/// `[anim]` / block fields. `Some(&[])` = the effect takes no params; `None` = unknown.
pub fn effect_params(effect: &str) -> Option<&'static [(&'static str, &'static str)]> {
    Some(match effect {
        "spin" => &[
            ("degrees", "rotation in degrees (default 360)"),
            ("duration", "seconds"),
            ("easing", "ease-out | ease-in | linear"),
        ],
        "pop" | "stretch" | "unroll" => &[
            ("from", "start scale 0..1 (0 = a line)"),
            ("duration", "seconds"),
            ("easing", "ease-out | ease-in | linear"),
        ],
        "slide" => &[
            ("dx", "x offset in px"),
            ("dy", "y offset in px"),
            ("duration", "seconds"),
            ("easing", "ease-out | ease-in | linear"),
        ],
        "wobble" => &[("spring", "spring stiffness"), ("friction", "velocity damping")],
        "wave" => &[
            ("amplitude", "peak UV displacement"),
            ("wavelength", "fraction of the travel axis"),
            ("speed", "crest travel, cycles/s"),
            ("axis", "x | y (travel direction)"),
            ("duration", "settle seconds (<=0 loops)"),
        ],
        "ripple" => &[
            ("amplitude", "peak UV refraction"),
            ("wavelength", "ring spacing"),
            ("speed", "ring expansion, cycles/s"),
            ("r0", "spread (big centre, faint rim)"),
            ("duration", "settle seconds (<=0 loops)"),
        ],
        "drain" => &[
            ("turns", "swirl rotations at full progress"),
            ("duration", "close seconds"),
        ],
        "reset" => &[],
        _ => return None,
    })
}

/// The compositor's reply to a [`Command`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reply {
    /// Command succeeded, nothing to return.
    Ok,
    /// A line of human-readable text (e.g. the `Ping` banner).
    Text(String),
    /// A list of windows (for `List`).
    Windows(Vec<WinInfo>),
    /// One window (for `Inspect`).
    Window(WinInfo),
    /// Command failed; the string explains why.
    Error(String),
}

/// A snapshot of one tracked window, as reported by `List`/`Inspect`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WinInfo {
    pub id: WinId,
    pub class: String,
    pub instance: String,
    pub window_type: String,
    pub title: String,
    pub mapped: bool,
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    /// Effective composited opacity now (`fade * dim`).
    pub opacity: f64,
    pub closing: bool,
}

/// Encode a value as one NDJSON line (trailing `\n`).
pub fn encode<T: Serialize>(v: &T) -> Vec<u8> {
    let mut buf = serde_json::to_vec(v).expect("proto value is always serializable");
    buf.push(b'\n');
    buf
}

/// Decode one JSON line (a trailing `\n`, if present, is ignored by serde_json).
pub fn decode<T: DeserializeOwned>(line: &[u8]) -> serde_json::Result<T> {
    serde_json::from_slice(line)
}

/// Derive the control-socket path from explicit inputs. Pure (no env / fs
/// access) so it is unit-testable without env races; [`socket_path`] wraps it.
///
/// - With `$XDG_RUNTIME_DIR` set (the normal case): `<xdg>/ricom-<display>.sock`
///   (that directory is user-private `0700`).
/// - Otherwise: `/tmp/ricom-<uid>-<display>.sock` (caller must `chmod 0600`).
///
/// `display` is sanitised to a single legal filename component
/// (`:0` → `0`, `:0.0` → `0_0`).
pub fn path_for(xdg: Option<&str>, uid: u32, display: &str) -> PathBuf {
    let tag = sanitize_display(display);
    match xdg {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir).join(format!("ricom-{tag}.sock")),
        _ => PathBuf::from("/tmp").join(format!("ricom-{uid}-{tag}.sock")),
    }
}

/// Map a `$DISPLAY` value to a safe single filename component: drop a leading
/// `:`, and replace every non-alphanumeric character with `_`.
fn sanitize_display(display: &str) -> String {
    let d = display.strip_prefix(':').unwrap_or(display);
    d.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// The control-socket path both server and client must agree on, derived from
/// the live environment (`$XDG_RUNTIME_DIR`, `$DISPLAY`, and the owner of `$HOME`).
#[cfg(unix)]
pub fn socket_path() -> PathBuf {
    let xdg = std::env::var("XDG_RUNTIME_DIR").ok();
    let display = std::env::var("DISPLAY").unwrap_or_default();
    path_for(xdg.as_deref(), current_uid(), &display)
}

/// Best-effort uid without a libc dependency: the numeric owner of `$HOME`.
#[cfg(unix)]
fn current_uid() -> u32 {
    use std::os::unix::fs::MetadataExt;
    std::env::var_os("HOME")
        .and_then(|h| std::fs::metadata(h).ok())
        .map(|m| m.uid())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests;
