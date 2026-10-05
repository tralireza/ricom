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

#[test]
fn composited_skips_input_only() {
    // Only a viewable InputOutput window has pixels to composite; an InputOnly one is
    // never "mapped" for ricom, even when the server reports it viewable.
    assert!(composited(MapState::VIEWABLE, WindowClass::INPUT_OUTPUT));
    assert!(!composited(MapState::VIEWABLE, WindowClass::INPUT_ONLY));
    assert!(!composited(MapState::UNVIEWABLE, WindowClass::INPUT_OUTPUT));
    assert!(!composited(MapState::UNMAPPED, WindowClass::INPUT_OUTPUT));
}

/// `find_client` over a fake tree — `(window, children)` pairs; the windows in `managed`
/// carry `WM_STATE` — returning the client and the round trips (probe calls) it took.
fn search(top: Window, tree: &[(Window, Vec<Window>)], managed: &[Window]) -> (Option<Window>, usize) {
    let mut trips = 0;
    let found = find_client(top, |level, descend| {
        trips += 1;
        level
            .iter()
            .map(|w| {
                let children = tree.iter().find(|(id, _)| id == w).map(|(_, c)| c.clone());
                (managed.contains(w), children.filter(|_| descend).unwrap_or_default())
            })
            .collect()
    });
    (found, trips)
}

#[test]
fn find_client_follows_icccm() {
    // A non-reparenting WM marks the top-level itself: one round trip.
    assert_eq!(search(1, &[(1, vec![2])], &[1]), (Some(1), 1));
    // A reparenting WM's frame holds the client one level down: one trip per level.
    assert_eq!(search(1, &[(1, vec![2])], &[2]), (Some(2), 2));
    // Breadth-first: a client one level down beats a deeper match under an earlier
    // sibling (decoration window 2).
    assert_eq!(search(1, &[(1, vec![2, 3]), (2, vec![4])], &[3, 4]), (Some(3), 2));
    // Nothing managed (an override-redirect popup, or no WM): none, and the search ends
    // at the leaves rather than probing empty levels.
    assert_eq!(search(1, &[(1, vec![2])], &[]), (None, 2));
}

#[test]
fn find_client_stops_at_the_depth_cap() {
    // A chain 1 → 2 → 3 → 4 → 5: WM_STATE CLIENT_SEARCH_DEPTH (3) levels down is found…
    let chain = [(1, vec![2]), (2, vec![3]), (3, vec![4]), (4, vec![5])];
    assert_eq!(CLIENT_SEARCH_DEPTH, 3);
    assert_eq!(search(1, &chain, &[4]), (Some(4), 4));
    // …one level deeper isn't.
    assert_eq!(search(1, &chain, &[5]), (None, 4));
}
