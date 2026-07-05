//! hotkey: spec parser tests (moved out of the parent module; see `#[cfg(test)] mod tests;`).

use super::*;

#[test]
fn parses_super_shift_f() {
    // 'f' keysym = 0x66; Shift is a separate modifier bit.
    assert_eq!(parse_hotkey("Super+Shift+F"), Some((SUPER | SHIFT, 0x66)));
}

#[test]
fn order_and_case_insensitive() {
    assert_eq!(parse_hotkey("shift+super+f"), Some((SUPER | SHIFT, 0x66)));
    assert_eq!(parse_hotkey("CTRL+ALT+p"), Some((CONTROL | ALT, 0x70)));
}

#[test]
fn function_and_digit_keys() {
    assert_eq!(parse_hotkey("F5"), Some((0, 0xFFBE + 4)));
    assert_eq!(parse_hotkey("Super+1"), Some((SUPER, 0x31)));
}

#[test]
fn named_keys() {
    assert_eq!(parse_hotkey("Super+space"), Some((SUPER, 0x20)));
    assert_eq!(parse_hotkey("Control+Escape"), Some((CONTROL, 0xFF1B)));
}

#[test]
fn rejects_bad_specs() {
    assert_eq!(parse_hotkey(""), None); // empty
    assert_eq!(parse_hotkey("Super+"), None); // trailing '+'
    assert_eq!(parse_hotkey("Shift+Control"), None); // no key
    assert_eq!(parse_hotkey("A+B"), None); // two keys
    assert_eq!(parse_hotkey("Hyper+F"), None); // unknown token
}
