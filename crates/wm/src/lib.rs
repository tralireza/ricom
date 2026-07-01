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
use anim::Fade;

/// An X window id.
pub type WindowId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapState {
    Unmapped,
    Mapped,
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
    /// `duration` seconds (a window just mapped). No-op if untracked.
    pub fn fade_in(&mut self, id: WindowId, target: f64, duration: f64) {
        if let Some(w) = self.wins.get_mut(&id) {
            w.fade = Fade::animating(0.0, target, duration);
        }
    }

    /// Ease a window toward a new opacity from its current displayed value
    /// (a live `_NET_WM_WINDOW_OPACITY` change). No-op if untracked.
    pub fn retarget_opacity(&mut self, id: WindowId, target: f64, duration: f64) {
        if let Some(w) = self.wins.get_mut(&id) {
            w.fade.retarget(target, duration);
        }
    }

    /// Advance every window's fade by `dt` seconds; returns whether any window
    /// is still animating (i.e. the frame clock should keep running).
    pub fn advance_fades(&mut self, dt: f64) -> bool {
        let mut animating = false;
        for w in self.wins.values_mut() {
            if w.fade.advance(dt) {
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
        assert!(s.advance_fades(0.1)); // still going
        let mid = s.get(1).unwrap().fade.current();
        assert!(mid > 0.0 && mid < 1.0);
        assert!(!s.advance_fades(0.2)); // past the end -> settled
        assert_eq!(s.get(1).unwrap().fade.current(), 1.0);
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
