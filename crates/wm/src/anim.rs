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
}
