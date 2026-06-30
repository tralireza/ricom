//! Pure-Rust rectangle regions (pixman-style) — the damage-tracking foundation.
//!
//! A [`Region`] is a set of pixels stored as a list of **pairwise-disjoint,
//! non-empty** rectangles. Coordinates are `i32`; rectangles are **half-open**:
//! a [`Rect`] covers `x1 <= x < x2` and `y1 <= y < y2`.
//!
//! Every operation preserves the disjoint-rectangles invariant, so
//! [`Region::area`] is exact (no double counting) and membership is unambiguous.
//! Rectangles are not band-merged yet, so a region may use more rectangles than
//! the minimal pixman form — correctness first; minimisation can come later.

/// A half-open rectangle: covers `x1 <= x < x2`, `y1 <= y < y2`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Rect {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

impl Rect {
    pub const fn new(x1: i32, y1: i32, x2: i32, y2: i32) -> Self {
        Rect { x1, y1, x2, y2 }
    }

    /// Construct from origin + size.
    pub const fn from_xywh(x: i32, y: i32, w: i32, h: i32) -> Self {
        Rect { x1: x, y1: y, x2: x + w, y2: y + h }
    }

    pub const fn width(&self) -> i32 {
        self.x2 - self.x1
    }

    pub const fn height(&self) -> i32 {
        self.y2 - self.y1
    }

    /// True if the rectangle has no area.
    pub const fn is_empty(&self) -> bool {
        self.x1 >= self.x2 || self.y1 >= self.y2
    }

    pub const fn area(&self) -> i64 {
        if self.is_empty() {
            0
        } else {
            (self.width() as i64) * (self.height() as i64)
        }
    }

    pub const fn contains_point(&self, x: i32, y: i32) -> bool {
        x >= self.x1 && x < self.x2 && y >= self.y1 && y < self.y2
    }

    /// Geometric intersection, or `None` if the rectangles don't overlap.
    pub fn intersect(&self, o: &Rect) -> Option<Rect> {
        let r = Rect {
            x1: self.x1.max(o.x1),
            y1: self.y1.max(o.y1),
            x2: self.x2.min(o.x2),
            y2: self.y2.min(o.y2),
        };
        if r.is_empty() {
            None
        } else {
            Some(r)
        }
    }

    pub const fn translated(&self, dx: i32, dy: i32) -> Rect {
        Rect { x1: self.x1 + dx, y1: self.y1 + dy, x2: self.x2 + dx, y2: self.y2 + dy }
    }
}

/// Append `a \ b` to `out`, decomposed into up to 4 disjoint rectangles
/// (top / bottom strips spanning `a`'s width, then left / right within the
/// overlap's vertical band). The pieces are disjoint and exactly cover `a \ b`.
fn rect_subtract(a: &Rect, b: &Rect, out: &mut Vec<Rect>) {
    if a.is_empty() {
        return;
    }
    let i = match a.intersect(b) {
        None => {
            out.push(*a);
            return;
        }
        Some(i) => i,
    };
    if i.y1 > a.y1 {
        out.push(Rect::new(a.x1, a.y1, a.x2, i.y1)); // top
    }
    if i.y2 < a.y2 {
        out.push(Rect::new(a.x1, i.y2, a.x2, a.y2)); // bottom
    }
    if i.x1 > a.x1 {
        out.push(Rect::new(a.x1, i.y1, i.x1, i.y2)); // left
    }
    if i.x2 < a.x2 {
        out.push(Rect::new(i.x2, i.y1, a.x2, i.y2)); // right
    }
}

/// A set of pixels as pairwise-disjoint, non-empty rectangles.
#[derive(Clone, Default, Debug)]
pub struct Region {
    rects: Vec<Rect>,
}

impl Region {
    pub fn new() -> Self {
        Region { rects: Vec::new() }
    }

    pub fn from_rect(r: Rect) -> Self {
        let mut g = Region::new();
        if !r.is_empty() {
            g.rects.push(r);
        }
        g
    }

    pub fn from_xywh(x: i32, y: i32, w: i32, h: i32) -> Self {
        Region::from_rect(Rect::from_xywh(x, y, w, h))
    }

    /// The disjoint rectangles making up this region.
    pub fn rects(&self) -> &[Rect] {
        &self.rects
    }

    pub fn is_empty(&self) -> bool {
        self.rects.is_empty()
    }

    pub fn len(&self) -> usize {
        self.rects.len()
    }

    pub fn clear(&mut self) {
        self.rects.clear();
    }

    /// Exact total area (rectangles are disjoint).
    pub fn area(&self) -> i64 {
        self.rects.iter().map(|r| r.area()).sum()
    }

    /// Smallest rectangle containing the whole region, or `None` if empty.
    pub fn extents(&self) -> Option<Rect> {
        let mut it = self.rects.iter();
        let mut e = *it.next()?;
        for r in it {
            e.x1 = e.x1.min(r.x1);
            e.y1 = e.y1.min(r.y1);
            e.x2 = e.x2.max(r.x2);
            e.y2 = e.y2.max(r.y2);
        }
        Some(e)
    }

    pub fn contains_point(&self, x: i32, y: i32) -> bool {
        self.rects.iter().any(|r| r.contains_point(x, y))
    }

    pub fn translate(&mut self, dx: i32, dy: i32) {
        for r in &mut self.rects {
            *r = r.translated(dx, dy);
        }
    }

    /// `self := self \ rect`
    pub fn subtract_rect(&mut self, b: &Rect) {
        if b.is_empty() || self.rects.is_empty() {
            return;
        }
        let mut out = Vec::with_capacity(self.rects.len());
        for a in &self.rects {
            rect_subtract(a, b, &mut out);
        }
        self.rects = out;
    }

    /// `self := self \ other`
    pub fn subtract(&mut self, other: &Region) {
        for b in &other.rects {
            if self.rects.is_empty() {
                break;
            }
            self.subtract_rect(b);
        }
    }

    /// `self := self ∩ rect`
    pub fn intersect_rect(&mut self, b: &Rect) {
        if b.is_empty() {
            self.rects.clear();
            return;
        }
        self.rects.retain_mut(|a| match a.intersect(b) {
            Some(i) => {
                *a = i;
                true
            }
            None => false,
        });
    }

    /// Returns `self ∩ other`. Disjoint inputs → disjoint output.
    pub fn intersection(&self, other: &Region) -> Region {
        let mut out = Vec::new();
        for a in &self.rects {
            for b in &other.rects {
                if let Some(i) = a.intersect(b) {
                    out.push(i);
                }
            }
        }
        Region { rects: out }
    }

    /// `self := self ∩ other`
    pub fn intersect(&mut self, other: &Region) {
        *self = self.intersection(other);
    }

    /// `self := self ∪ rect`. Only the parts of `rect` not already covered are
    /// added, preserving disjointness.
    pub fn add_rect(&mut self, r: Rect) {
        if r.is_empty() {
            return;
        }
        let mut parts = vec![r];
        for a in &self.rects {
            if parts.is_empty() {
                break;
            }
            let mut next = Vec::new();
            for p in &parts {
                rect_subtract(p, a, &mut next);
            }
            parts = next;
        }
        self.rects.extend(parts);
    }

    /// `self := self ∪ other`
    pub fn union(&mut self, other: &Region) {
        for r in &other.rects {
            self.add_rect(*r);
        }
    }
}

#[cfg(test)]
mod tests {
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
}
