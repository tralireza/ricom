//! wm::anim: fade / easing / offset tests (moved out of the parent module; see `#[cfg(test)] mod tests;`).

use super::*;

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[test]
fn settled_never_animates() {
    let mut f = Fade::settled(0.5);
    assert!(!f.is_animating());
    assert_eq!(f.current(), 0.5);
    assert!(!f.advance(1.0));
    assert_eq!(f.current(), 0.5);
}

#[test]
fn zero_duration_is_settled() {
    let f = Fade::animating(0.0, 1.0, 0.0);
    assert!(!f.is_animating());
    assert_eq!(f.current(), 1.0);
}

#[test]
fn ease_out_leads_the_linear_midpoint() {
    // At t=0.5, ease-out = 1-(0.5)^2 = 0.75, i.e. ahead of a linear 0.5.
    let mut f = Fade::animating(0.0, 1.0, 0.2);
    assert!(f.is_animating());
    assert_eq!(f.current(), 0.0);
    assert!(f.advance(0.1)); // halfway in time
    assert!(approx(f.current(), 0.75));
}

#[test]
fn advance_past_end_settles_at_target() {
    let mut f = Fade::animating(0.0, 1.0, 0.2);
    assert!(f.advance(0.1));
    assert!(!f.advance(0.2)); // overshoots the end
    assert_eq!(f.current(), 1.0);
    assert!(!f.is_animating());
}

#[test]
fn retarget_eases_from_current_value() {
    let mut f = Fade::animating(0.0, 1.0, 0.2);
    f.advance(0.1); // current ~0.75
    let mid = f.current();
    f.retarget(0.0, 0.2); // now fade back out from 0.75
    assert_eq!(f.target(), 0.0);
    assert_eq!(f.current(), mid); // starts from where it was
    assert!(f.is_animating());
    f.advance(0.2);
    assert_eq!(f.current(), 0.0);
}

#[test]
fn retarget_to_same_target_is_noop() {
    let mut f = Fade::animating(0.0, 1.0, 0.2);
    f.advance(0.05);
    let before = f.current();
    f.retarget(1.0, 0.2); // same target -> don't restart the curve
    assert_eq!(f.current(), before);
}

// --- Offset (translate primitive) ----------------------------------------

#[test]
fn offset_settled_never_animates() {
    let mut o = Offset::settled();
    assert!(!o.is_animating());
    assert_eq!(o.current(), [0.0, 0.0]);
    assert!(!o.advance(1.0));
    assert_eq!(o.current(), [0.0, 0.0]);
}

#[test]
fn offset_eases_both_axes_to_rest() {
    // "translate_in" shape: slide from [40, -60] to [0, 0].
    let mut o = Offset::animating([40.0, -60.0], [0.0, 0.0], 0.2, Easing::EaseOut);
    assert!(o.is_animating());
    assert_eq!(o.current(), [40.0, -60.0]);
    assert!(o.advance(0.1)); // halfway in time
    // ease-out at t=0.5 = 0.75 of the way there.
    let [x, y] = o.current();
    assert!((x - 10.0).abs() < 1e-3, "x={x}"); // 40 -> 40*(1-0.75)=10
    assert!((y + 15.0).abs() < 1e-3, "y={y}"); // -60 -> -60*0.25=-15
    assert!(!o.advance(0.2)); // overshoot end -> settled at rest
    assert_eq!(o.current(), [0.0, 0.0]);
    assert!(!o.is_animating());
}

#[test]
fn offset_reaches_target_on_close() {
    // "translate_out" shape: slide from rest out to [200, 0].
    let mut o = Offset::animating([0.0, 0.0], [200.0, 0.0], 0.2, Easing::Linear);
    o.advance(0.1); // linear, halfway
    let [x, _] = o.current();
    assert!((x - 100.0).abs() < 1e-3, "x={x}");
    o.advance(0.2);
    assert_eq!(o.current(), [200.0, 0.0]);
}

#[test]
fn offset_ease_in_lags_the_linear_midpoint() {
    // ease-in at t=0.5 = 0.25, i.e. behind a linear 0.5.
    let mut o = Offset::animating([0.0, 0.0], [100.0, 0.0], 0.2, Easing::EaseIn);
    o.advance(0.1);
    let [x, _] = o.current();
    assert!((x - 25.0).abs() < 1e-3, "x={x}");
}

// --- Wobble spring sim ---------------------------------------------------

const K: f32 = 350.0; // representative default spring
const C: f32 = 14.0; // representative default friction (underdamped -> visible jiggle)

/// Drive a wobble to rest at 60 Hz, asserting it never blows up. Returns the
/// number of steps it took to settle (panics if it never does).
fn run_to_rest(wob: &mut Wobble) -> usize {
    let mut steps = 0;
    while wob.advance(1.0 / 60.0) {
        steps += 1;
        assert!(steps < 100_000, "wobble never settled");
        let b = wob.bounds(0.0);
        assert!(b.iter().all(|v| v.is_finite()), "non-finite bounds: {b:?}");
        assert!(
            b[2] - b[0] < 100_000.0 && b[3] - b[1] < 100_000.0,
            "wobble blew up: {b:?}"
        );
    }
    steps
}

#[test]
fn wobble_at_rest_is_settled() {
    // Created on a rect with no retarget: already at its anchors, no motion.
    let mut wob = Wobble::new([10.0, 20.0, 100.0, 80.0], K, C);
    assert!(!wob.advance(1.0 / 60.0)); // settled immediately
    let v = wob.vertices();
    assert_eq!(v.len(), WOBBLE_N * WOBBLE_N);
    assert!((v[0][0] - 10.0).abs() < 1e-4 && (v[0][1] - 20.0).abs() < 1e-4); // corner (0,0)
}

#[test]
fn wobble_translate_settles_to_target_grid() {
    let mut wob = Wobble::new([0.0, 0.0, 100.0, 100.0], K, C);
    wob.retarget([40.0, 25.0, 100.0, 100.0]); // pure move
    assert!(wob.advance(1.0 / 60.0)); // now lagging -> animating
    let steps = run_to_rest(&mut wob);
    assert!(steps > 0 && steps < 1200, "unreasonable settle time: {steps} steps");
    // Snapped exactly to the translated anchor grid.
    let v = wob.vertices();
    let last = v.len() - 1;
    assert!((v[0][0] - 40.0).abs() < 1e-3 && (v[0][1] - 25.0).abs() < 1e-3);
    assert!((v[last][0] - 140.0).abs() < 1e-3 && (v[last][1] - 125.0).abs() < 1e-3);
}

#[test]
fn wobble_resize_settles_to_new_spacing() {
    let mut wob = Wobble::new([0.0, 0.0, 100.0, 100.0], K, C);
    wob.retarget([0.0, 0.0, 240.0, 160.0]); // grow
    run_to_rest(&mut wob);
    let v = wob.vertices();
    let last = v.len() - 1; // bottom-right corner -> new size
    assert!((v[last][0] - 240.0).abs() < 1e-3 && (v[last][1] - 160.0).abs() < 1e-3);
}

#[test]
fn wobble_survives_a_huge_move() {
    // A pathological jump must not blow up — it must still settle.
    let mut wob = Wobble::new([0.0, 0.0, 200.0, 200.0], K, C);
    wob.retarget([9000.0, -9000.0, 200.0, 200.0]);
    run_to_rest(&mut wob); // asserts finite + bounded throughout
    let v = wob.vertices();
    assert!((v[0][0] - 9000.0).abs() < 1e-2 && (v[0][1] + 9000.0).abs() < 1e-2);
}

#[test]
fn wobble_actually_wobbles_before_settling() {
    // Underdamped: at least one point should overshoot past its anchor.
    let mut wob = Wobble::new([0.0, 0.0, 100.0, 100.0], K, C);
    wob.retarget([0.0, 60.0, 100.0, 100.0]); // move down by 60
    let mut overshot = false;
    for _ in 0..600 {
        if !wob.advance(1.0 / 60.0) {
            break;
        }
        // Top-left corner anchor is y=60; overshoot means a point passes y>60.
        if wob.vertices().iter().any(|v| v[1] > 60.5) {
            overshot = true;
        }
    }
    assert!(overshot, "expected an underdamped overshoot (a real wobble)");
}

#[test]
fn wobble_zero_spring_settles_instantly() {
    let mut wob = Wobble::new([0.0, 0.0, 100.0, 100.0], 0.0, C);
    wob.retarget([50.0, 50.0, 100.0, 100.0]);
    assert!(!wob.advance(1.0 / 60.0)); // no force -> snap to rest, don't hang
    let v = wob.vertices();
    assert!((v[0][0] - 50.0).abs() < 1e-4 && (v[0][1] - 50.0).abs() < 1e-4);
}

// --- Wave (traveling-crest params) ---------------------------------------

#[test]
fn wave_settles_at_requested_duration() {
    // `duration` is the settle time: the derived decay drives amp to the threshold at ~D.
    for d in [1.0_f32, 2.0, 3.5] {
        let mut w = Wave::new(0.06, 0.5, 1.5, Axis::X, d);
        let (a0, wl, ph0, _axis) = w.params();
        assert!(a0 > 0.0 && wl > 0.0 && ph0 == 0.0);
        let mut steps = 0;
        while w.advance(1.0 / 60.0) {
            steps += 1;
            assert!(steps < 100_000, "wave never settled");
        }
        let secs = steps as f32 / 60.0;
        assert!((secs - d).abs() < d * 0.15, "duration {d}: settled in {secs}s");
        assert!(w.params().2 > 0.0, "phase should advance (crest travels)");
    }
}

#[test]
fn wave_loops_when_duration_zero() {
    let mut w = Wave::new(0.06, 0.5, 1.5, Axis::X, 0.0);
    for _ in 0..600 {
        assert!(w.advance(1.0 / 60.0), "duration 0 should loop, not settle");
    }
}

#[test]
fn wave_axis_is_preserved() {
    let w = Wave::new(0.06, 0.5, 1.5, Axis::Y, 2.0);
    assert!(matches!(w.params().3, Axis::Y));
}

// --- Ripple (radial refraction params) -----------------------------------

#[test]
fn ripple_settles_at_requested_duration() {
    // `duration` is the settle time: the derived decay drives amp to the threshold at ~D.
    for d in [1.5_f32, 2.5, 4.0] {
        let mut rp = Ripple::new([0.5, 0.5], 0.06, 0.18, 1.2, 0.12, d);
        let (c, a0, wl, ph0, r0) = rp.params();
        assert_eq!(c, [0.5, 0.5]);
        assert!(a0 > 0.0 && wl > 0.0 && r0 > 0.0 && ph0 == 0.0);
        let mut steps = 0;
        while rp.advance(1.0 / 60.0) {
            steps += 1;
            assert!(steps < 100_000, "ripple never settled");
        }
        let secs = steps as f32 / 60.0;
        assert!((secs - d).abs() < d * 0.15, "duration {d}: settled in {secs}s");
        assert!(rp.params().3 > 0.0, "phase should advance (rings expand)");
    }
}

#[test]
fn ripple_loops_when_duration_zero() {
    let mut rp = Ripple::new([0.5, 0.5], 0.06, 0.18, 1.2, 0.12, 0.0);
    for _ in 0..600 {
        assert!(rp.advance(1.0 / 60.0), "duration 0 should loop, not settle");
    }
}
