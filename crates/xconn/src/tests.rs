//! xconn: pure-helper tests (no X server needed).

use super::*;

/// A mode with just the timing fields `mode_refresh` reads.
fn mode(dot_clock: u32, htotal: u16, vtotal: u16, mode_flags: ModeFlag) -> ModeInfo {
    ModeInfo { dot_clock, htotal, vtotal, mode_flags, ..Default::default() }
}

#[test]
fn mode_refresh_matches_xrandr() {
    let hz = |m: ModeInfo| mode_refresh(&m).unwrap();
    // CEA-861 1080p60: 148.5 MHz over 2200×1125.
    assert!((hz(mode(148_500_000, 2200, 1125, ModeFlag::default())) - 60.0).abs() < 1e-9);
    // 1080i60: half the clock over the same totals — interlace reports the field rate.
    assert!((hz(mode(74_250_000, 2200, 1125, ModeFlag::INTERLACE)) - 60.0).abs() < 1e-9);
    // Doublescan sends every line twice: 12.6 MHz over 400×(2·225) = 70 Hz, not 140.
    assert!((hz(mode(12_600_000, 400, 225, ModeFlag::DOUBLE_SCAN)) - 70.0).abs() < 1e-9);
    // No timings (virtual outputs) → unknown, not a divide-by-zero.
    assert_eq!(mode_refresh(&mode(0, 2200, 1125, ModeFlag::default())), None);
    assert_eq!(mode_refresh(&mode(148_500_000, 0, 0, ModeFlag::default())), None);
}
