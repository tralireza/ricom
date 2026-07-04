//! Window tracking: a [`WindowStack`] of top-level [`Win`]s kept in
//! bottom-to-top stacking order, updated incrementally from X structure events.
//! Mirrors the role of picom's `src/wm/{win,wm,tree}.c` (MVP subset).
//!
//! This crate is intentionally X-agnostic (plain `u32` ids, `i16/u16` geometry)
//! so it can be unit-tested with no X server. The `session` crate translates
//! x11rb events into the calls below; the renderer reads back the mapped windows
//! in stacking order to know what to composite.

use std::collections::HashMap;

pub mod anim;
use anim::{Fade, Wobble};

/// An X window id.
pub type WindowId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapState {
    Unmapped,
    Mapped,
}

/// Burn/dissolve close state: `progress` animates `0.0 → 1.0` over the burn
/// duration (a `Fade` reused as a generic eased scalar); `seed` de-correlates each
/// window's noise so no two burns look the same.
#[derive(Debug, Clone)]
pub struct BurnState {
    pub progress: Fade,
    pub seed: f32,
}

/// A tracked top-level window.
#[derive(Debug, Clone)]
pub struct Win {
    pub id: WindowId,
    pub x: i16,
    pub y: i16,
    pub width: u16,
    pub height: u16,
    pub border_width: u16,
    pub override_redirect: bool,
    pub map_state: MapState,
    /// Animated whole-window opacity. `fade.current()` is what to display now;
    /// `fade.target()` is the goal (from `_NET_WM_WINDOW_OPACITY`, default 1.0).
    pub fade: Fade,
    /// Animated scale-about-centre for the open/close "pop". `scale.current()` is
    /// the factor to render at now (1.0 = full size); settled at 1.0 when idle.
    pub scale: Fade,
    /// Active move/resize wobble (spring-mesh), or `None` when the window is not
    /// wobbling. Dropped by [`WindowStack::advance_anims`] once it settles.
    pub wobble: Option<Wobble>,
    /// Active burn/dissolve close, or `None`. When `Some`, the window is dissolving
    /// (progress 0→1) instead of fading; reaped when progress reaches 1.
    pub burn: Option<BurnState>,
    /// Fading out (unmapped/destroyed) — kept in the composite set until the fade
    /// reaches 0, then reaped. See [`WindowStack::begin_fade_out`].
    pub closing: bool,
    /// The X window is gone (DestroyNotify), so on fade-out completion it is
    /// removed from the stack rather than just released (kept if merely unmapped).
    pub destroyed: bool,
}

impl Win {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: WindowId,
        x: i16,
        y: i16,
        width: u16,
        height: u16,
        border_width: u16,
        override_redirect: bool,
        mapped: bool,
    ) -> Self {
        Win {
            id,
            x,
            y,
            width,
            height,
            border_width,
            override_redirect,
            map_state: if mapped { MapState::Mapped } else { MapState::Unmapped },
            fade: Fade::settled(1.0),
            scale: Fade::settled(1.0),
            wobble: None,
            burn: None,
            closing: false,
            destroyed: false,
        }
    }

    pub fn is_mapped(&self) -> bool {
        self.map_state == MapState::Mapped
    }
}

/// Top-level windows in bottom-to-top stacking order.
///
/// `order` holds ids bottom→top; `wins` maps id→[`Win`]. The two stay in sync:
/// every id in `order` has an entry in `wins` and vice-versa.
#[derive(Debug, Default)]
pub struct WindowStack {
    order: Vec<WindowId>,
    wins: HashMap<WindowId, Win>,
}

impl WindowStack {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    pub fn get(&self, id: WindowId) -> Option<&Win> {
        self.wins.get(&id)
    }

    /// All windows, bottom-to-top.
    pub fn iter_bottom_to_top(&self) -> impl Iterator<Item = &Win> {
        self.order.iter().filter_map(move |id| self.wins.get(id))
    }

    /// Mapped windows only, bottom-to-top (the compositing order).
    pub fn mapped_bottom_to_top(&self) -> impl Iterator<Item = &Win> {
        self.iter_bottom_to_top().filter(|w| w.is_mapped())
    }

    /// Windows to composite, bottom-to-top: those mapped, plus those fading out
    /// (unmapped/destroyed but not yet fully transparent).
    pub fn visible_bottom_to_top(&self) -> impl Iterator<Item = &Win> {
        self.iter_bottom_to_top().filter(|w| w.is_mapped() || w.closing)
    }

    pub fn mapped_count(&self) -> usize {
        self.mapped_bottom_to_top().count()
    }

    /// Add a window at the top of the stack (X creates windows topmost).
    pub fn add_top(&mut self, win: Win) {
        let id = win.id;
        self.order.retain(|&w| w != id);
        self.order.push(id);
        self.wins.insert(id, win);
    }

    pub fn remove(&mut self, id: WindowId) {
        self.wins.remove(&id);
        self.order.retain(|&w| w != id);
    }

    pub fn set_mapped(&mut self, id: WindowId, mapped: bool) {
        if let Some(w) = self.wins.get_mut(&id) {
            w.map_state = if mapped { MapState::Mapped } else { MapState::Unmapped };
        }
    }

    /// Set a window's opacity immediately, no fade (already-visible windows at
    /// startup, or an unmapped window's initial value). No-op if untracked.
    pub fn set_opacity_settled(&mut self, id: WindowId, opacity: f64) {
        if let Some(w) = self.wins.get_mut(&id) {
            w.fade = Fade::settled(opacity);
        }
    }

    /// Begin fading a window in from fully transparent to `target` over
    /// `duration` seconds (a window just mapped). Cancels any pending fade-out.
    /// No-op if untracked.
    pub fn fade_in(&mut self, id: WindowId, target: f64, duration: f64) {
        if let Some(w) = self.wins.get_mut(&id) {
            w.fade = Fade::animating(0.0, target, duration);
            w.closing = false;
            w.destroyed = false;
            w.burn = None;
        }
    }

    /// Begin fading a window out to fully transparent (it was unmapped, or
    /// destroyed if `destroyed`). Returns `true` if there is something still
    /// visible to fade (so the caller keeps its pixmap and runs the frame clock);
    /// `false` if it's already invisible/untracked and can be dropped immediately.
    pub fn begin_fade_out(&mut self, id: WindowId, duration: f64, destroyed: bool) -> bool {
        match self.wins.get_mut(&id) {
            Some(w) if w.fade.current() > 0.0 => {
                w.closing = true;
                w.destroyed |= destroyed;
                w.fade.retarget(0.0, duration);
                true
            }
            _ => false,
        }
    }

    /// Begin a burn/dissolve close: run `progress` 0→1 over `duration` (keeping the
    /// window composited via its pixmap) with noise `seed`. `destroyed` marks it for
    /// removal (vs merely unmapped) on completion. Returns `true` if there is
    /// something still visible to burn, `false` if it can be dropped immediately.
    pub fn begin_burn(&mut self, id: WindowId, duration: f64, seed: f32, destroyed: bool) -> bool {
        match self.wins.get_mut(&id) {
            Some(w) if w.fade.current() > 0.0 => {
                w.closing = true;
                w.destroyed |= destroyed;
                w.burn = Some(BurnState { progress: Fade::animating(0.0, 1.0, duration), seed });
                true
            }
            _ => false,
        }
    }

    /// Clear a window's closing flag (its fade-out/burn completed but the window
    /// still exists — merely unmapped, not destroyed).
    pub fn clear_closing(&mut self, id: WindowId) {
        if let Some(w) = self.wins.get_mut(&id) {
            w.closing = false;
            w.burn = None;
        }
    }

    /// Closing windows whose fade-out has finished (fully transparent), as
    /// `(id, destroyed)`. The caller releases their resources and either removes
    /// them (destroyed) or clears their closing flag (still-mapped-able).
    pub fn finished_fadeouts(&self) -> Vec<(WindowId, bool)> {
        self.wins
            .values()
            .filter(|w| {
                w.closing
                    && match &w.burn {
                        Some(b) => !b.progress.is_animating() && b.progress.current() >= 1.0,
                        None => !w.fade.is_animating() && w.fade.current() <= 0.0,
                    }
            })
            .map(|w| (w.id, w.destroyed))
            .collect()
    }

    /// Ease a window toward a new opacity from its current displayed value
    /// (a live `_NET_WM_WINDOW_OPACITY` change). No-op if untracked.
    pub fn retarget_opacity(&mut self, id: WindowId, target: f64, duration: f64) {
        if let Some(w) = self.wins.get_mut(&id) {
            w.fade.retarget(target, duration);
        }
    }

    /// Start the open "pop": scale-about-centre from `from` (e.g. 0.85) up to 1.0
    /// over `duration`. Pairs with [`fade_in`](Self::fade_in). No-op if untracked.
    pub fn scale_in(&mut self, id: WindowId, from: f64, duration: f64) {
        if let Some(w) = self.wins.get_mut(&id) {
            w.scale = Fade::animating(from, 1.0, duration);
        }
    }

    /// Ease a window's scale toward `to` from its current value (e.g. down to 0.85
    /// on close, paired with [`begin_fade_out`](Self::begin_fade_out)). No-op if
    /// untracked.
    pub fn retarget_scale(&mut self, id: WindowId, to: f64, duration: f64) {
        if let Some(w) = self.wins.get_mut(&id) {
            w.scale.retarget(to, duration);
        }
    }

    /// Start (or continue) a move/resize wobble: aim the spring mesh at outer rect
    /// `new`, creating it from `old` first if the window isn't already wobbling.
    /// Rects are `[x, y, w, h]` in screen px. No-op if untracked.
    pub fn wobble_to(
        &mut self,
        id: WindowId,
        old: [f32; 4],
        new: [f32; 4],
        spring: f32,
        friction: f32,
    ) {
        if let Some(w) = self.wins.get_mut(&id) {
            match &mut w.wobble {
                Some(wob) => wob.retarget(new),
                None => {
                    let mut wob = Wobble::new(old, spring, friction);
                    wob.retarget(new);
                    w.wobble = Some(wob);
                }
            }
        }
    }

    /// Advance every window's animations (opacity fade, scale pop, and wobble) by
    /// `dt` seconds; settled wobbles are dropped. Returns whether any window is
    /// still animating (i.e. the frame clock should keep running).
    pub fn advance_anims(&mut self, dt: f64) -> bool {
        let mut animating = false;
        for w in self.wins.values_mut() {
            if w.fade.advance(dt) {
                animating = true;
            }
            if w.scale.advance(dt) {
                animating = true;
            }
            if let Some(wob) = &mut w.wobble {
                if wob.advance(dt as f32) {
                    animating = true;
                } else {
                    w.wobble = None;
                }
            }
            if let Some(b) = &mut w.burn
                && b.progress.advance(dt)
            {
                animating = true;
            }
        }
        animating
    }

    /// Update geometry and restack relative to `above` (the sibling this window
    /// is now directly on top of; `None` means bottom of the stack).
    #[allow(clippy::too_many_arguments)]
    pub fn configure(
        &mut self,
        id: WindowId,
        x: i16,
        y: i16,
        width: u16,
        height: u16,
        border_width: u16,
        above: Option<WindowId>,
    ) {
        match self.wins.get_mut(&id) {
            Some(w) => {
                w.x = x;
                w.y = y;
                w.width = width;
                w.height = height;
                w.border_width = border_width;
            }
            None => return,
        }
        self.restack(id, above);
    }

    /// Place `id` directly above sibling `above` (or at the bottom if `None`).
    pub fn restack(&mut self, id: WindowId, above: Option<WindowId>) {
        if !self.wins.contains_key(&id) {
            return;
        }
        self.order.retain(|&w| w != id);
        match above {
            None => self.order.insert(0, id),
            Some(s) => match self.order.iter().position(|&w| w == s) {
                Some(pos) => self.order.insert(pos + 1, id),
                None => self.order.push(id), // unknown sibling -> top
            },
        }
    }

    /// Raise a window to the top (CirculateNotify on-top).
    pub fn raise(&mut self, id: WindowId) {
        if self.wins.contains_key(&id) {
            self.order.retain(|&w| w != id);
            self.order.push(id);
        }
    }

    /// Lower a window to the bottom (CirculateNotify on-bottom).
    pub fn lower(&mut self, id: WindowId) {
        if self.wins.contains_key(&id) {
            self.order.retain(|&w| w != id);
            self.order.insert(0, id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn win(id: WindowId, mapped: bool) -> Win {
        Win::new(id, 0, 0, 100, 100, 0, false, mapped)
    }

    fn order(s: &WindowStack) -> Vec<WindowId> {
        s.iter_bottom_to_top().map(|w| w.id).collect()
    }

    #[test]
    fn add_remove_and_top_ordering() {
        let mut s = WindowStack::new();
        s.add_top(win(1, true));
        s.add_top(win(2, true));
        s.add_top(win(3, true));
        assert_eq!(order(&s), vec![1, 2, 3]); // bottom -> top
        s.remove(2);
        assert_eq!(order(&s), vec![1, 3]);
        assert!(s.get(2).is_none());
        // re-adding an existing id moves it to top without duplicating
        s.add_top(win(1, true));
        assert_eq!(order(&s), vec![3, 1]);
    }

    #[test]
    fn map_state_and_mapped_iter() {
        let mut s = WindowStack::new();
        s.add_top(win(1, true));
        s.add_top(win(2, false));
        s.add_top(win(3, true));
        assert_eq!(s.mapped_count(), 2);
        let mapped: Vec<_> = s.mapped_bottom_to_top().map(|w| w.id).collect();
        assert_eq!(mapped, vec![1, 3]);
        s.set_mapped(2, true);
        assert_eq!(s.mapped_count(), 3);
        s.set_mapped(1, false);
        assert_eq!(
            s.mapped_bottom_to_top().map(|w| w.id).collect::<Vec<_>>(),
            vec![2, 3]
        );
    }

    #[test]
    fn configure_updates_geometry_and_restacks() {
        let mut s = WindowStack::new();
        s.add_top(win(1, true));
        s.add_top(win(2, true));
        s.add_top(win(3, true)); // order: 1,2,3
        // move 1 to just above 3 -> 2,3,1, and change its geometry
        s.configure(1, 10, 20, 200, 150, 1, Some(3));
        assert_eq!(order(&s), vec![2, 3, 1]);
        let w = s.get(1).unwrap();
        assert_eq!((w.x, w.y, w.width, w.height, w.border_width), (10, 20, 200, 150, 1));
        // restack 3 to bottom
        s.configure(3, 0, 0, 100, 100, 0, None);
        assert_eq!(order(&s), vec![3, 2, 1]);
    }

    #[test]
    fn opacity_defaults_opaque_and_settles() {
        let mut s = WindowStack::new();
        s.add_top(win(1, true));
        assert_eq!(s.get(1).unwrap().fade.current(), 1.0); // default: fully opaque
        s.set_opacity_settled(1, 0.5);
        assert_eq!(s.get(1).unwrap().fade.current(), 0.5);
        assert!(!s.get(1).unwrap().fade.is_animating());
        s.set_opacity_settled(99, 0.25); // untracked id -> no-op, no panic
        assert!(s.get(99).is_none());
    }

    #[test]
    fn fade_in_animates_then_settles() {
        let mut s = WindowStack::new();
        s.add_top(win(1, true));
        s.fade_in(1, 1.0, 0.2);
        assert_eq!(s.get(1).unwrap().fade.current(), 0.0); // starts transparent
        assert!(s.get(1).unwrap().fade.is_animating());
        assert!(s.advance_anims(0.1)); // still going
        let mid = s.get(1).unwrap().fade.current();
        assert!(mid > 0.0 && mid < 1.0);
        assert!(!s.advance_anims(0.2)); // past the end -> settled
        assert_eq!(s.get(1).unwrap().fade.current(), 1.0);
    }

    #[test]
    fn fade_out_unmapped_stays_visible_then_reaps() {
        let mut s = WindowStack::new();
        s.add_top(win(1, true));
        s.set_mapped(1, false); // unmapped...
        assert!(s.begin_fade_out(1, 0.2, false)); // ...but fading out
        assert!(s.get(1).unwrap().closing && !s.get(1).unwrap().destroyed);
        // unmapped yet still composited while fading
        assert_eq!(s.visible_bottom_to_top().count(), 1);
        assert!(s.advance_anims(0.1));
        assert!(s.finished_fadeouts().is_empty());
        assert!(!s.advance_anims(0.2)); // finishes
        assert_eq!(s.finished_fadeouts(), vec![(1, false)]);
        s.clear_closing(1); // not destroyed -> keep in stack, just cleared
        assert!(s.get(1).is_some());
        assert_eq!(s.visible_bottom_to_top().count(), 0); // unmapped, not closing
    }

    #[test]
    fn destroy_fade_out_marks_for_removal() {
        let mut s = WindowStack::new();
        s.add_top(win(1, true));
        assert!(s.begin_fade_out(1, 0.2, true)); // destroyed
        s.advance_anims(0.3); // finish
        assert_eq!(s.finished_fadeouts(), vec![(1, true)]);
    }

    #[test]
    fn fade_in_cancels_pending_fade_out() {
        let mut s = WindowStack::new();
        s.add_top(win(1, true));
        s.begin_fade_out(1, 0.2, true);
        s.fade_in(1, 1.0, 0.2); // window re-mapped mid-fade-out
        assert!(!s.get(1).unwrap().closing);
        assert!(!s.get(1).unwrap().destroyed);
    }

    #[test]
    fn begin_fade_out_noop_when_already_invisible() {
        let mut s = WindowStack::new();
        s.add_top(win(1, true));
        s.set_opacity_settled(1, 0.0); // already fully transparent
        assert!(!s.begin_fade_out(1, 0.2, false));
        assert!(!s.get(1).unwrap().closing);
    }

    #[test]
    fn raise_and_lower() {
        let mut s = WindowStack::new();
        for id in [1, 2, 3] {
            s.add_top(win(id, true));
        }
        s.lower(3);
        assert_eq!(order(&s), vec![3, 1, 2]);
        s.raise(3);
        assert_eq!(order(&s), vec![1, 2, 3]);
    }
}
