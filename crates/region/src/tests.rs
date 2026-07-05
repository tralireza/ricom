//! region: rectangle-region maths tests (moved out of the parent module; see `#[cfg(test)] mod tests;`).

use super::*;
use std::collections::HashSet;

/// Rasterise the region into the pixel set within `[0, n) x [0, n)`.
fn raster(g: &Region, n: i32) -> HashSet<(i32, i32)> {
    let mut s = HashSet::new();
    for r in g.rects() {
        for y in r.y1.max(0)..r.y2.min(n) {
            for x in r.x1.max(0)..r.x2.min(n) {
                s.insert((x, y));
            }
        }
    }
    s
}

fn assert_disjoint(g: &Region) {
    let rs = g.rects();
    for i in 0..rs.len() {
        assert!(!rs[i].is_empty(), "empty rect stored: {:?}", rs[i]);
        for j in (i + 1)..rs.len() {
            assert!(
                rs[i].intersect(&rs[j]).is_none(),
                "overlap between {:?} and {:?}",
                rs[i],
                rs[j]
            );
        }
    }
}

#[test]
fn rect_intersect_and_area() {
    let a = Rect::new(0, 0, 4, 4);
    let b = Rect::new(2, 2, 6, 6);
    assert_eq!(a.intersect(&b), Some(Rect::new(2, 2, 4, 4)));
    assert_eq!(a.area(), 16);
    assert!(Rect::new(5, 5, 5, 9).is_empty());
    assert!(a.intersect(&Rect::new(4, 4, 8, 8)).is_none()); // edge-touch only
}

#[test]
fn rect_subtract_covers_exactly() {
    let a = Rect::new(0, 0, 10, 10);
    let b = Rect::new(3, 3, 7, 7); // hole in the middle -> 4 strips
    let mut out = Vec::new();
    rect_subtract(&a, &b, &mut out);
    let total: i64 = out.iter().map(|r| r.area()).sum();
    assert_eq!(total, a.area() - b.area());
    assert_eq!(out.len(), 4);
}

#[test]
fn union_overlap_no_double_count() {
    let mut g = Region::from_rect(Rect::new(0, 0, 4, 4));
    g.add_rect(Rect::new(2, 2, 6, 6));
    assert_disjoint(&g);
    assert_eq!(g.area(), 16 + 16 - 4); // overlap of 2x2 counted once
}

#[test]
fn subtract_to_empty() {
    let mut g = Region::from_rect(Rect::new(0, 0, 5, 5));
    g.subtract_rect(&Rect::new(-1, -1, 6, 6));
    assert!(g.is_empty());
    assert_eq!(g.area(), 0);
}

#[test]
fn translate_and_extents() {
    let mut g = Region::new();
    g.add_rect(Rect::new(0, 0, 2, 2));
    g.add_rect(Rect::new(5, 5, 7, 8));
    assert_eq!(g.extents(), Some(Rect::new(0, 0, 7, 8)));
    g.translate(10, 20);
    assert_eq!(g.extents(), Some(Rect::new(10, 20, 17, 28)));
    assert!(Region::new().extents().is_none());
}

/// Validate union / intersection / subtraction against true set semantics
/// on a small grid, and confirm the disjoint invariant + exact area.
#[test]
fn brute_force_set_semantics() {
    let n = 12;
    let a = {
        let mut g = Region::new();
        g.add_rect(Rect::new(0, 0, 6, 6));
        g.add_rect(Rect::new(4, 4, 10, 10));
        g
    };
    let b = {
        let mut g = Region::new();
        g.add_rect(Rect::new(2, 2, 8, 5));
        g.add_rect(Rect::new(7, 0, 12, 12));
        g
    };
    assert_disjoint(&a);
    assert_disjoint(&b);
    let sa = raster(&a, n);
    let sb = raster(&b, n);

    let mut u = a.clone();
    u.union(&b);
    assert_disjoint(&u);
    assert_eq!(raster(&u, n), sa.union(&sb).cloned().collect());
    assert_eq!(u.area() as usize, sa.union(&sb).count()); // all rects within grid

    let it = a.intersection(&b);
    assert_disjoint(&it);
    assert_eq!(raster(&it, n), sa.intersection(&sb).cloned().collect());
    assert_eq!(it.area() as usize, sa.intersection(&sb).count());

    let mut d = a.clone();
    d.subtract(&b);
    assert_disjoint(&d);
    assert_eq!(raster(&d, n), sa.difference(&sb).cloned().collect());
    assert_eq!(d.area() as usize, sa.difference(&sb).count());
}
