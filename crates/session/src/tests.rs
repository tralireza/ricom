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
        let c = random_corner(start, &mut rng);
        assert_ne!(c, start, "auto-hop never picks the current corner");
        let i = others.iter().position(|&o| o == c).expect("one of the other three");
        hits[i] = true;
    }
    assert!(hits.iter().all(|&h| h), "all three other corners get picked");
}
