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
