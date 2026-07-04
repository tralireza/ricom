//! Pure opacity-fade animation: a value easing toward a target over a fixed
//! duration. X- and clock-agnostic — the caller advances it by an elapsed `dt`
//! — so it unit-tests without a compositor or a wall clock, like `region` and
//! the window model. `session` owns the clock (a `calloop` timer) and feeds `dt`.

/// Quadratic ease-out: fast start, gentle finish. `t` is clamped to `0..=1`.
fn ease_out(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    1.0 - (1.0 - t) * (1.0 - t)
}

/// An opacity value easing from `start` toward `target` over `duration` seconds,
/// sampling with [`ease_out`]. `current` is what should be displayed now.
#[derive(Debug, Clone, Copy)]
pub struct Fade {
    start: f64,
    target: f64,
    current: f64,
    elapsed: f64,
    duration: f64,
}

impl Fade {
    /// A fade already resting at `value` (no animation) — for windows that are
    /// already on screen (e.g. present at compositor startup).
    pub fn settled(value: f64) -> Self {
        Fade { start: value, target: value, current: value, elapsed: 0.0, duration: 0.0 }
    }

    /// A fade animating from `from` to `to` over `duration` seconds. A
    /// non-positive duration collapses to a settled fade at `to`.
    pub fn animating(from: f64, to: f64, duration: f64) -> Self {
        if duration <= 0.0 {
            return Fade::settled(to);
        }
        Fade { start: from, target: to, current: from, elapsed: 0.0, duration }
    }

    pub fn current(&self) -> f64 {
        self.current
    }

    pub fn target(&self) -> f64 {
        self.target
    }

    /// Whether the fade is still moving toward its target.
    pub fn is_animating(&self) -> bool {
        self.elapsed < self.duration && self.current != self.target
    }

    /// Aim at a new target, easing from the *current displayed* value over
    /// `duration`. No-op if already heading there (avoids restarting the curve).
    pub fn retarget(&mut self, to: f64, duration: f64) {
        if to == self.target {
            return;
        }
        if duration <= 0.0 {
            *self = Fade::settled(to);
            return;
        }
        self.start = self.current;
        self.target = to;
        self.elapsed = 0.0;
        self.duration = duration;
    }

    /// Advance by `dt` seconds (clamped to non-negative). Returns whether the
    /// fade is still animating after the step.
    pub fn advance(&mut self, dt: f64) -> bool {
        if !self.is_animating() {
            self.current = self.target;
            return false;
        }
        self.elapsed += dt.max(0.0);
        if self.elapsed >= self.duration {
            self.current = self.target;
            return false;
        }
        self.current = self.start + (self.target - self.start) * ease_out(self.elapsed / self.duration);
        true
    }
}

/// Number of control points per side of the wobble mesh grid (`N×N` total).
/// The backend builds its mesh index buffer for this same `N` (`backend-gl`'s
/// `MESH_N`) — keep the two in sync.
pub const WOBBLE_N: usize = 8;

/// Coupling of the neighbour (structural) springs relative to the anchor spring.
/// Keeps the grid coherent — a dragged corner tugs its neighbours — without
/// over-stiffening. Internal; not exposed in config.
const STRUCT_RATIO: f32 = 0.5;

/// Sub-steps per [`Wobble::advance`]. The integrator is sub-stepped so a stiff
/// spring stays stable at a 60 Hz `dt`.
const WOBBLE_SUBSTEPS: usize = 4;

/// Settle thresholds: once every point is within this of its anchor (px) **and**
/// slower than this (px/s), the wobble is done and snaps exactly to rest.
const WOBBLE_EPS_POS: f32 = 0.35;
const WOBBLE_EPS_VEL: f32 = 1.5;

/// Speed clamp (px/s) — a backstop so a mis-tuned spring can never blow up to
/// infinity/NaN; well above any real wobble velocity.
const WOBBLE_VMAX: f32 = 20_000.0;

/// One control point of the [`Wobble`] mesh: a position and velocity in screen
/// pixels. Mass is 1, so force == acceleration.
#[derive(Debug, Clone, Copy)]
struct CtrlPt {
    pos: [f32; 2],
    vel: [f32; 2],
}

/// A spring-mass mesh that lags and jiggles a window toward a target rect — the
/// Compiz "wobbly windows" jelly. An `N×N` grid of control points is pulled
/// toward its ideal grid over the window's outer rect (the *anchor* springs) and
/// toward its neighbours (the *structural* springs that give the jelly its
/// coupling), under implicit velocity damping. Advance by `dt` until it settles.
///
/// Pure and clock-agnostic like [`Fade`]: no GL, no X — the caller feeds `dt`
/// and reads [`vertices`](Wobble::vertices) / [`bounds`](Wobble::bounds), which
/// keeps this crate dependency-free (the renderer/session own pixels & regions).
#[derive(Debug, Clone)]
pub struct Wobble {
    /// Row-major control points, index `j * WOBBLE_N + i` (`i` = column/x, `j` = row/y).
    pts: [CtrlPt; WOBBLE_N * WOBBLE_N],
    /// Target outer rect `[x, y, w, h]` (px). Anchors derive from this; a
    /// [`retarget`](Wobble::retarget) moves it, leaving the points behind to wobble.
    target: [f32; 4],
    /// Anchor-spring stiffness `k` (pull toward the ideal grid).
    spring: f32,
    /// Velocity damping (higher = settles faster, less jiggle).
    friction: f32,
}

impl Wobble {
    /// Ideal grid position of control point `(i, j)` over outer rect `target`.
    fn anchor(target: [f32; 4], i: usize, j: usize) -> [f32; 2] {
        let [x, y, w, h] = target;
        let d = (WOBBLE_N - 1) as f32;
        [x + (i as f32 / d) * w, y + (j as f32 / d) * h]
    }

    /// A mesh resting on outer rect `rect` (`[x, y, w, h]` px): all points at
    /// their anchors, no motion. `spring`/`friction` tune the sim (see [`Wobble`]).
    pub fn new(rect: [f32; 4], spring: f32, friction: f32) -> Self {
        let mut pts = [CtrlPt { pos: [0.0; 2], vel: [0.0; 2] }; WOBBLE_N * WOBBLE_N];
        for j in 0..WOBBLE_N {
            for i in 0..WOBBLE_N {
                pts[j * WOBBLE_N + i].pos = Self::anchor(rect, i, j);
            }
        }
        Wobble { pts, target: rect, spring: spring.max(0.0), friction: friction.max(0.0) }
    }

    /// Aim the anchors at a new outer rect. The control points keep their current
    /// positions, so they now lag the anchors — the spring pulls them in and they
    /// overshoot/jiggle into place (the wobble). Cheap to call on every
    /// `ConfigureNotify` (a WM drag) or once per programmatic move.
    pub fn retarget(&mut self, rect: [f32; 4]) {
        self.target = rect;
    }

    /// Current mesh vertices as `[x_px, y_px, u, v]`, row-major (`j * N + i`), for
    /// the renderer. UVs are the static grid `(i/(N-1), j/(N-1))`, matching the
    /// blit's top-left texture origin; positions are the live wobble.
    pub fn vertices(&self) -> Vec<[f32; 4]> {
        let d = (WOBBLE_N - 1) as f32;
        let mut v = Vec::with_capacity(WOBBLE_N * WOBBLE_N);
        for j in 0..WOBBLE_N {
            for i in 0..WOBBLE_N {
                let p = self.pts[j * WOBBLE_N + i].pos;
                v.push([p[0], p[1], i as f32 / d, j as f32 / d]);
            }
        }
        v
    }

    /// Axis-aligned bounds of the deformed mesh as `[min_x, min_y, max_x, max_y]`,
    /// expanded by `pad` px (headroom for overshoot). The caller turns this into a
    /// damage/clip rect — `wm` stays region-agnostic.
    pub fn bounds(&self, pad: f32) -> [f32; 4] {
        let mut mnx = f32::INFINITY;
        let mut mny = f32::INFINITY;
        let mut mxx = f32::NEG_INFINITY;
        let mut mxy = f32::NEG_INFINITY;
        for p in &self.pts {
            mnx = mnx.min(p.pos[0]);
            mny = mny.min(p.pos[1]);
            mxx = mxx.max(p.pos[0]);
            mxy = mxy.max(p.pos[1]);
        }
        [mnx - pad, mny - pad, mxx + pad, mxy + pad]
    }

    /// Whether every control point has reached its anchor and (near-)stopped.
    fn is_settled(&self) -> bool {
        for j in 0..WOBBLE_N {
            for i in 0..WOBBLE_N {
                let pt = &self.pts[j * WOBBLE_N + i];
                let a = Self::anchor(self.target, i, j);
                let (dx, dy) = (a[0] - pt.pos[0], a[1] - pt.pos[1]);
                if dx * dx + dy * dy > WOBBLE_EPS_POS * WOBBLE_EPS_POS {
                    return false;
                }
                if pt.vel[0] * pt.vel[0] + pt.vel[1] * pt.vel[1] > WOBBLE_EPS_VEL * WOBBLE_EPS_VEL {
                    return false;
                }
            }
        }
        true
    }

    /// Snap every point exactly to its anchor at rest (kills residual drift).
    fn snap_to_rest(&mut self) {
        for j in 0..WOBBLE_N {
            for i in 0..WOBBLE_N {
                self.pts[j * WOBBLE_N + i] =
                    CtrlPt { pos: Self::anchor(self.target, i, j), vel: [0.0; 2] };
            }
        }
    }

    /// One integration sub-step of `h` seconds. Jacobi-style semi-implicit Euler:
    /// all accelerations are read from the pre-step snapshot, then applied at once
    /// (order-independent). Damping is implicit (`v /= 1 + friction·h`), which is
    /// unconditionally stable.
    fn step(&mut self, h: f32) {
        let n = WOBBLE_N;
        let [_, _, w, ht] = self.target;
        let d = (n - 1) as f32;
        let (rest_x, rest_y) = (w / d, ht / d);
        let struct_k = self.spring * STRUCT_RATIO;

        let mut acc = [[0.0f32; 2]; WOBBLE_N * WOBBLE_N];
        for j in 0..n {
            for i in 0..n {
                let idx = j * n + i;
                let p = self.pts[idx].pos;
                // Anchor spring toward the ideal grid position.
                let a = Self::anchor(self.target, i, j);
                let mut fx = self.spring * (a[0] - p[0]);
                let mut fy = self.spring * (a[1] - p[1]);
                // Structural springs to the 4 axis neighbours: force =
                // k·((neighbour − p) − rest_offset). Zero at the anchor grid, so
                // the rest configuration is an exact equilibrium.
                if i + 1 < n {
                    let q = self.pts[idx + 1].pos;
                    fx += struct_k * ((q[0] - p[0]) - rest_x);
                    fy += struct_k * (q[1] - p[1]);
                }
                if i > 0 {
                    let q = self.pts[idx - 1].pos;
                    fx += struct_k * ((q[0] - p[0]) + rest_x);
                    fy += struct_k * (q[1] - p[1]);
                }
                if j + 1 < n {
                    let q = self.pts[idx + n].pos;
                    fx += struct_k * (q[0] - p[0]);
                    fy += struct_k * ((q[1] - p[1]) - rest_y);
                }
                if j > 0 {
                    let q = self.pts[idx - n].pos;
                    fx += struct_k * (q[0] - p[0]);
                    fy += struct_k * ((q[1] - p[1]) + rest_y);
                }
                acc[idx] = [fx, fy];
            }
        }

        let damp = 1.0 / (1.0 + self.friction * h);
        for (pt, a) in self.pts.iter_mut().zip(acc.iter()) {
            let mut v = [(pt.vel[0] + a[0] * h) * damp, (pt.vel[1] + a[1] * h) * damp];
            v[0] = v[0].clamp(-WOBBLE_VMAX, WOBBLE_VMAX);
            v[1] = v[1].clamp(-WOBBLE_VMAX, WOBBLE_VMAX);
            pt.vel = v;
            pt.pos[0] += v[0] * h;
            pt.pos[1] += v[1] * h;
        }
    }

    /// Advance the sim by `dt` seconds. Returns whether it is *still* wobbling (so
    /// the frame clock keeps running); on settling it snaps every point exactly to
    /// its anchor and returns `false`. A non-positive `spring` settles instantly
    /// (defensive — config keeps it positive).
    pub fn advance(&mut self, dt: f32) -> bool {
        if self.spring <= 0.0 {
            self.snap_to_rest();
            return false;
        }
        if dt > 0.0 {
            let h = dt / WOBBLE_SUBSTEPS as f32;
            for _ in 0..WOBBLE_SUBSTEPS {
                self.step(h);
            }
        }
        if self.is_settled() {
            self.snap_to_rest();
            return false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
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
}
