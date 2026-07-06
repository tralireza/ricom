//! wm: window-stack + animation-state tests (moved out of the parent module; see `#[cfg(test)] mod tests;`).

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
fn scale_collapse_close_reaps_without_fade() {
    let mut s = WindowStack::new();
    s.add_top(win(1, true));
    s.set_mapped(1, false); // unmapped (closing out)
    // Stretch-close: mark closing without a fade, collapse scale on X toward 0.
    assert!(s.begin_collapse(1, true));
    s.retarget_scale(1, 0.0, 0.2, anim::Axis::X, anim::Easing::EaseOut);
    assert!(s.get(1).unwrap().closing);
    assert_eq!(s.get(1).unwrap().fade.current(), 1.0); // opacity untouched
    assert!(s.advance_anims(0.1)); // scale still collapsing
    assert!(s.finished_fadeouts().is_empty());
    assert!(!s.advance_anims(0.2)); // scale settles at 0 (a line)
    // Completes via the collapse (not a fade) and is marked for removal.
    assert_eq!(s.finished_fadeouts(), vec![(1, true)]);
}

#[test]
fn drain_close_reaps_at_full_progress() {
    let mut s = WindowStack::new();
    s.add_top(win(1, true));
    s.set_mapped(1, false); // unmapped (closing out)
    // Whirlpool close: progress 0->1 over the duration; opacity untouched (the shader fades).
    assert!(s.begin_drain(1, 0.2, 1.5, 0.5, true));
    assert!(s.get(1).unwrap().closing && s.get(1).unwrap().drain.is_some());
    assert_eq!(s.get(1).unwrap().fade.current(), 1.0); // opacity untouched — drain drives the vanish
    assert!(s.advance_anims(0.1)); // still draining
    assert!(s.finished_fadeouts().is_empty());
    assert!(!s.advance_anims(0.2)); // progress settles at 1
    // Completes via the drain progress (not a fade) and is marked for removal.
    assert_eq!(s.finished_fadeouts(), vec![(1, true)]);
}

#[test]
fn spin_in_eases_angle_to_zero() {
    let mut s = WindowStack::new();
    s.add_top(win(1, true));
    s.spin_in(1, 2.0, 0.2, anim::Easing::Linear);
    assert_eq!(s.get(1).unwrap().spin.current(), 2.0);
    assert!(s.advance_anims(0.1)); // still rotating
    assert!((s.get(1).unwrap().spin.current() - 1.0).abs() < 1e-2); // ~halfway (linear)
    assert!(!s.advance_anims(0.2)); // settles upright at 0
    assert_eq!(s.get(1).unwrap().spin.current(), 0.0);
}

#[test]
fn set_dim_eases_to_target() {
    let mut s = WindowStack::new();
    s.add_top(win(1, true));
    assert_eq!(s.get(1).unwrap().dim.current(), 1.0); // full bright at rest
    s.set_dim(1, 0.7, 0.2); // unfocused → dim toward 0.7
    assert!(s.advance_anims(0.1)); // dimming
    assert!(!s.advance_anims(0.2)); // settled
    assert!((s.get(1).unwrap().dim.current() - 0.7).abs() < 1e-6);
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
