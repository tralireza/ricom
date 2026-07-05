//! Parse a human keybind string like `"Super+Shift+F"` into an X11 modifier
//! mask + keysym. Pure (no X calls) so it unit-tests without a server, like
//! `region`/`anim`; `session` resolves the keysym to a keycode via `xconn` and
//! grabs it. Modifier bits match X11's `KeyButMask`/`ModMask`
//! (Shift=0x01, Control=0x04, Mod1/Alt=0x08, Mod4/Super=0x40).

/// X11 modifier-mask bits we recognise.
pub const SHIFT: u16 = 0x01;
pub const CONTROL: u16 = 0x04;
pub const ALT: u16 = 0x08; // Mod1
pub const SUPER: u16 = 0x40; // Mod4

/// Parse `spec` (e.g. `"Super+Shift+F"`, case-insensitive, `+`-separated) into
/// `(modifier_mask, keysym)`. Exactly one non-modifier token (the key) is
/// required. Returns `None` on empty input, an empty/unknown token, a missing
/// key, or more than one key.
pub fn parse_hotkey(spec: &str) -> Option<(u16, u32)> {
    let mut mods: u16 = 0;
    let mut key: Option<u32> = None;
    for raw in spec.split('+') {
        let tok = raw.trim();
        if tok.is_empty() {
            return None;
        }
        if let Some(m) = modifier(tok) {
            mods |= m;
            continue;
        }
        let ks = keysym(tok)?;
        if key.replace(ks).is_some() {
            return None; // more than one non-modifier key
        }
    }
    key.map(|ks| (mods, ks))
}

/// A recognised modifier name → its mask bit (case-insensitive).
fn modifier(tok: &str) -> Option<u16> {
    match tok.to_ascii_lowercase().as_str() {
        "shift" => Some(SHIFT),
        "control" | "ctrl" | "ctl" => Some(CONTROL),
        "alt" | "mod1" | "meta" => Some(ALT),
        "super" | "win" | "cmd" | "logo" | "mod4" => Some(SUPER),
        _ => None,
    }
}

/// Map a single key token to its base (unshifted) keysym. Covers function keys
/// `F1..=F24`, single ASCII letters/digits, and a handful of named keys.
fn keysym(tok: &str) -> Option<u32> {
    // Function keys: F1..=F24 -> XK_F1 (0xFFBE) upward.
    if let Some(n) = tok.strip_prefix(['F', 'f']).and_then(|d| d.parse::<u32>().ok())
        && (1..=24).contains(&n)
    {
        return Some(0xFFBE + (n - 1));
    }
    // Single ASCII letter/digit -> its (lowercase) Latin-1 keysym, which equals
    // the ASCII code. Shift, if present, is a separate modifier bit.
    if tok.len() == 1 {
        let c = tok.chars().next().unwrap();
        if c.is_ascii_alphabetic() {
            return Some(c.to_ascii_lowercase() as u32);
        }
        if c.is_ascii_digit() {
            return Some(c as u32);
        }
    }
    // A few named keys (X11 keysymdef values).
    Some(match tok.to_ascii_lowercase().as_str() {
        "space" => 0x0020,
        "tab" => 0xFF09,
        "return" | "enter" => 0xFF0D,
        "escape" | "esc" => 0xFF1B,
        "backspace" => 0xFF08,
        "delete" | "del" => 0xFFFF,
        "home" => 0xFF50,
        "end" => 0xFF57,
        "prior" | "pageup" => 0xFF55,
        "next" | "pagedown" => 0xFF56,
        "left" => 0xFF51,
        "up" => 0xFF52,
        "right" => 0xFF53,
        "down" => 0xFF54,
        _ => return None,
    })
}

#[cfg(test)]
mod tests;
