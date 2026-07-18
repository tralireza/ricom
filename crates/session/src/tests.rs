//! Pure-helper tests for `session` — the `ricomctl animate` param parsing +
//! validation. No X / GL / socket, so these run on the Mac (unlike the rest of the
//! crate, which only *runs* on Linux).

use super::*;

/// Build a `Vec<(String, String)>` param list from `&str` pairs.
fn p(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs.iter().map(|(k, v)| ((*k).to_string(), (*v).to_string())).collect()
}

#[test]
fn param_f32_absent_present_bad() {
    let ps = p(&[("amplitude", "0.12"), ("duration", "abc")]);
    assert_eq!(param_f32(&ps, "amplitude").unwrap(), Some(0.12));
    assert_eq!(param_f32(&ps, "missing").unwrap(), None); // absent → None, not an error
    assert!(param_f32(&ps, "duration").is_err()); // "abc" isn't a number
}

#[test]
fn param_axis_and_easing() {
    assert!(matches!(param_axis(&p(&[("axis", "y")])).unwrap(), Some(wm::anim::Axis::Y)));
    assert!(param_axis(&p(&[])).unwrap().is_none());
    assert!(param_axis(&p(&[("axis", "diagonal")])).is_err());
    assert!(matches!(param_easing(&p(&[("easing", "linear")])).unwrap(), Some(wm::anim::Easing::Linear)));
    assert!(param_easing(&p(&[("easing", "bouncy")])).is_err());
}

#[test]
fn check_keys_strict() {
    // valid keys come from the shared proto::effect_params schema (single source).
    assert!(check_keys("ripple", &p(&[("amplitude", "0.1"), ("duration", "3")])).is_ok());
    // an unknown key is rejected, and the message names it + lists the valid set
    let err = check_keys("ripple", &p(&[("amplitud", "0.1")])).unwrap_err();
    assert!(err.contains("amplitud") && err.contains("amplitude"));
    // reset takes no params
    assert!(check_keys("reset", &p(&[("x", "1")])).is_err());
    assert!(check_keys("reset", &p(&[])).is_ok());
}

#[test]
fn random_corner_differs_and_covers_others() {
    use HudCorner::*;
    let mut rng = 0x9e37_79b9_7f4a_7c15u64; // fixed seed → deterministic test
    let start = TopLeft;
    let others = [TopRight, BottomLeft, BottomRight];
    let mut hits = [false; 3];
    for _ in 0..300 {
        let c = random_corner(start, &[], &mut rng).expect("a corner is always free with no avoid");
        assert_ne!(c, start, "auto-hop never picks the current corner");
        let i = others.iter().position(|&o| o == c).expect("one of the other three");
        hits[i] = true;
    }
    assert!(hits.iter().all(|&h| h), "all three other corners get picked");
}

#[test]
fn random_corner_respects_avoid() {
    use HudCorner::*;
    let mut rng = 0x1234_5678_9abc_def0u64;
    // From top-left, forbid bottom-right → only top-right / bottom-left remain.
    for _ in 0..300 {
        let c = random_corner(TopLeft, &[BottomRight], &mut rng).expect("two corners free");
        assert!(matches!(c, TopRight | BottomLeft), "never the current or an avoided corner");
    }
}

#[test]
fn random_corner_moves_off_a_forbidden_current() {
    use HudCorner::*;
    // Parked on bottom-left while bottom-left is avoided → still hops away.
    let mut rng = 0xdead_beef_cafe_0001u64;
    for _ in 0..100 {
        let c = random_corner(BottomLeft, &[BottomLeft], &mut rng).expect("three corners free");
        assert_ne!(c, BottomLeft);
    }
}

#[test]
fn random_corner_none_when_no_corner_is_free() {
    use HudCorner::*;
    let mut rng = 0x0f0f_0f0f_0f0f_0f0fu64;
    // All four avoided → nowhere to go.
    assert!(random_corner(TopLeft, &[TopLeft, TopRight, BottomLeft, BottomRight], &mut rng).is_none());
    // Current is the only non-avoided corner → stays put.
    assert!(random_corner(TopRight, &[TopLeft, BottomLeft, BottomRight], &mut rng).is_none());
}

#[test]
fn hop_view_hides_at_from_then_shows_at_to() {
    use HudCorner::*;
    // Start: fully visible at the old corner.
    let (c, o) = hop_view(TopRight, BottomLeft, 0.0);
    assert_eq!(c, TopRight);
    assert!((o - 1.0).abs() < 1e-6, "opaque at t=0, got {o}");
    // Just before the midpoint: still the old corner, faded to ~0.
    let (c, o) = hop_view(TopRight, BottomLeft, 0.499);
    assert_eq!(c, TopRight, "still hiding at the old corner before the swap");
    assert!(o < 0.05, "nearly hidden before the swap, got {o}");
    // Midpoint: swapped to the new corner, still ~0 (invisible during the swap).
    let (c, o) = hop_view(TopRight, BottomLeft, 0.5);
    assert_eq!(c, BottomLeft, "swaps to the new corner at the midpoint");
    assert!(o < 1e-6, "invisible at the swap, got {o}");
    // End: fully visible at the new corner.
    let (c, o) = hop_view(TopRight, BottomLeft, 1.0);
    assert_eq!(c, BottomLeft);
    assert!((o - 1.0).abs() < 1e-6, "opaque at t=1, got {o}");
    // Opacity stays within [0, 1] across the whole hop.
    for i in 0..=100 {
        let t = i as f64 / 100.0;
        let (_, o) = hop_view(TopRight, BottomLeft, t);
        assert!((0.0..=1.0).contains(&o), "opacity {o} out of range at t={t}");
    }
}

// ── paint_region: the buffer-age partial-repaint decision ──────────────────────
// The pure region math extracted from `App::composite` (the rest of composite is
// X/GL-bound and only runs on i7; this decision does not, so it's Mac-testable).

/// A single-rect `Region` for terse fixtures.
fn reg(x: i32, y: i32, w: i32, h: i32) -> Region {
    Region::from_rect(Rect::from_xywh(x, y, w, h))
}

/// The screen rect used across these fixtures.
fn screen() -> Rect {
    Rect::from_xywh(0, 0, 100, 100)
}

#[test]
fn paint_region_own_full_overrides_age() {
    let hist: VecDeque<Region> = VecDeque::new();
    let dmg = reg(10, 10, 5, 5);
    // own_full forces a whole-screen repaint even with an otherwise-usable age.
    let p = paint_region(true, 1, &dmg, &hist, screen());
    assert_eq!(p.rects(), Region::from_rect(screen()).rects());
}

#[test]
fn paint_region_unusable_age_is_full() {
    let hist: VecDeque<Region> = VecDeque::new(); // no history retained
    let dmg = reg(10, 10, 5, 5);
    let full = Region::from_rect(screen());
    // age <= 0 is the sentinel "backend can't report an age" → full.
    assert_eq!(paint_region(false, 0, &dmg, &hist, screen()).rects(), full.rects());
    assert_eq!(paint_region(false, -1, &dmg, &hist, screen()).rects(), full.rects());
    // age older than history+1 (here > 1, with 0 history) can't be reconstructed → full.
    assert_eq!(paint_region(false, 2, &dmg, &hist, screen()).rects(), full.rects());
}

#[test]
fn paint_region_age1_is_this_frames_damage() {
    let hist: VecDeque<Region> = VecDeque::new();
    let dmg = reg(10, 10, 5, 5);
    // age 1: the buffer holds last frame → repaint only this frame's damage (no history).
    let p = paint_region(false, 1, &dmg, &hist, screen());
    assert_eq!(p.rects(), dmg.rects());
}

#[test]
fn paint_region_age2_unions_previous_frame() {
    let mut hist: VecDeque<Region> = VecDeque::new();
    hist.push_front(reg(50, 50, 5, 5)); // previous frame's damage (most-recent at front)
    let dmg = reg(10, 10, 5, 5);
    // age 2: this frame's damage ∪ the previous frame's (take(age-1) = take(1)).
    let p = paint_region(false, 2, &dmg, &hist, screen());
    let mut want = reg(10, 10, 5, 5);
    want.union(&reg(50, 50, 5, 5));
    assert_eq!(p.rects(), want.rects());
    // and it does NOT reach back further than age-1 allows:
    let mut hist3 = hist.clone();
    hist3.push_back(reg(80, 80, 5, 5)); // an older frame that age 2 must NOT include
    let p2 = paint_region(false, 2, &dmg, &hist3, screen());
    assert_eq!(p2.rects(), want.rects());
}

#[test]
fn paint_region_clips_to_screen() {
    let hist: VecDeque<Region> = VecDeque::new();
    let dmg = reg(90, 90, 50, 50); // pokes past the 100×100 screen edge
    let p = paint_region(false, 1, &dmg, &hist, screen());
    let mut want = reg(90, 90, 50, 50);
    want.intersect_rect(&screen());
    assert_eq!(p.rects(), want.rects());
    for r in p.rects() {
        assert!(r.x1 >= 0 && r.y1 >= 0 && r.x2 <= 100 && r.y2 <= 100, "rect {r:?} escaped screen");
    }
}
