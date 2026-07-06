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

// --- Wave ripple ---------------------------------------------------------

#[test]
fn wave_is_flat_grid_at_zero_amp() {
    let w = Wave::new([10.0, 20.0, 100.0, 80.0], 0.0, 0.5, 1.5, Axis::X, 0.12);
    let v = w.vertices();
    assert_eq!(v.len(), WOBBLE_N * WOBBLE_N);
    assert!((v[0][0] - 10.0).abs() < 1e-4 && (v[0][1] - 20.0).abs() < 1e-4); // corner (0,0)
    let last = v.len() - 1;
    assert!((v[last][0] - 110.0).abs() < 1e-4 && (v[last][1] - 100.0).abs() < 1e-4); // bottom-right
}

#[test]
fn wave_x_axis_displaces_y_only() {
    // Axis::X travels along X and displaces Y; X coords stay on the anchor grid.
    let d = (WOBBLE_N - 1) as f32;
    let w = Wave::new([0.0, 0.0, 100.0, 100.0], 15.0, 0.5, 1.0, Axis::X, 0.5);
    for (k, p) in w.vertices().iter().enumerate() {
        let anchor_x = (k % WOBBLE_N) as f32 / d * 100.0;
        assert!((p[0] - anchor_x).abs() < 1e-4, "x should not move on an X-axis wave");
    }
}

#[test]
fn wave_displaces_then_rings_down_to_flat() {
    let mut w = Wave::new([0.0, 0.0, 200.0, 200.0], 20.0, 0.5, 1.5, Axis::X, 0.05);
    let d = (WOBBLE_N - 1) as f32;
    // At t=0 some vertex is displaced off its anchor row.
    let displaced = w.vertices().iter().enumerate().any(|(k, p)| {
        let anchor_y = (k / WOBBLE_N) as f32 / d * 200.0;
        (p[1] - anchor_y).abs() > 1.0
    });
    assert!(displaced, "wave should deform the grid at t=0");
    // A decaying wave must settle in finite time.
    let mut steps = 0;
    while w.advance(1.0 / 60.0) {
        steps += 1;
        assert!(steps < 100_000, "wave never settled");
    }
    assert!(steps > 0, "a decaying wave should take some frames to ring down");
}

#[test]
fn wave_loops_forever_when_decay_is_one() {
    let mut w = Wave::new([0.0, 0.0, 100.0, 100.0], 12.0, 0.5, 1.5, Axis::X, 1.0);
    for _ in 0..600 {
        assert!(w.advance(1.0 / 60.0), "decay = 1.0 should loop, not settle");
    }
}
