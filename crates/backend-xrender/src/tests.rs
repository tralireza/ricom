//! Pure-helper tests for `backend-xrender` — the geometry / colour / opacity / text / flip
//! math behind each frame: damage rects, per-window blend, A8 glyph padding, premultiplied
//! colour fills, the flip swapchain's buffer-age, and HUD layout (anchor / graph / outline).
//! No X connection, so these run on the Mac; the RENDER side effects in `present_windows`
//! (`ensure_pool`, per-window composites, the Present page-flip) need a live server → i7 only.

use super::*;
use x11rb::protocol::render::Directformat;

/// A minimal opaque `WindowDraw` with the given size + clip rects (all effect slots
/// `None`, as this backend's caps-gating guarantees upstream).
fn wd(w: i32, h: i32, clip: Vec<region::Rect>) -> WindowDraw {
    WindowDraw {
        quad: Quad {
            pixmap: 0,
            x: 0,
            y: 0,
            w,
            h,
            opacity: 1.0,
            shadow: false,
            blur: false,
            corner_radius: 0.0,
        },
        clip,
        mesh: None,
        burn: None,
        spin: None,
        ripple: None,
        wave: None,
        drain: None,
    }
}

#[test]
fn to_rect_converts_and_clamps() {
    // inclusive-exclusive (x1,y1,x2,y2) → origin + size
    let r = to_rect(&region::Rect::from_xywh(3, 4, 10, 20));
    assert_eq!((r.x, r.y, r.width, r.height), (3, 4, 10, 20));
    // a degenerate (inverted) rect clamps width/height at 0 — never underflows u16
    let d = to_rect(&region::Rect::from_xywh(5, 5, -3, -4));
    assert_eq!((d.width, d.height), (0, 0));
}

#[test]
fn rects_maps_every_clip_rect() {
    let out = rects(&[region::Rect::from_xywh(0, 0, 4, 4), region::Rect::from_xywh(10, 10, 2, 3)]);
    assert_eq!(out.len(), 2);
    assert_eq!((out[1].x, out[1].y, out[1].width, out[1].height), (10, 10, 2, 3));
    assert!(rects(&[]).is_empty());
}

#[test]
fn should_skip_zero_area_or_empty_clip() {
    let full = vec![region::Rect::from_xywh(0, 0, 10, 10)];
    assert!(!should_skip(&wd(10, 10, full.clone()))); // a real, visible window → keep
    assert!(should_skip(&wd(0, 10, full.clone()))); // zero width
    assert!(should_skip(&wd(10, 0, full.clone()))); // zero height
    assert!(should_skip(&wd(-1, 10, full.clone()))); // negative width
    assert!(should_skip(&wd(10, -1, full))); // negative height
    assert!(should_skip(&wd(10, 10, vec![]))); // fully occluded (empty clip)
}

#[test]
fn quantize_opacity_rounds_and_clamps() {
    assert_eq!(quantize_opacity(0.0), 0);
    assert_eq!(quantize_opacity(1.0), 255);
    assert_eq!(quantize_opacity(0.5), 128); // 127.5 rounds up
    assert_eq!(quantize_opacity(2.0), 255); // clamps high
    assert_eq!(quantize_opacity(-1.0), 0); // clamps low
}

#[test]
fn scale_channel_clamps_to_16bit() {
    assert_eq!(scale_channel(0.0), 0);
    assert_eq!(scale_channel(1.0), 65535);
    assert_eq!(scale_channel(2.0), 65535); // clamps high
    assert_eq!(scale_channel(-0.5), 0); // clamps low
}

#[test]
fn alpha16_expands_full_range() {
    assert_eq!(alpha16(0), 0);
    assert_eq!(alpha16(255), 65535); // 255 · 257 = 65535 (exact endpoint)
    assert_eq!(alpha16(128), 32896);
}

#[test]
fn find_format_matches_depth_and_alpha() {
    let mk = |id: u32, depth: u8, ashift: u16, amask: u16| Pictforminfo {
        id,
        type_: PictType::DIRECT,
        depth,
        direct: Directformat {
            red_shift: 16,
            red_mask: 0xff,
            green_shift: 8,
            green_mask: 0xff,
            blue_shift: 0,
            blue_mask: 0xff,
            alpha_shift: ashift,
            alpha_mask: amask,
        },
        colormap: 0,
    };
    let formats = vec![mk(24, 24, 0, 0), mk(32, 32, 24, 0xff)];
    // depth-24 no-alpha → x8r8g8b8; depth-32 alpha → a8r8g8b8
    assert_eq!(find_format(&formats, 24, false), Some(24));
    assert_eq!(find_format(&formats, 32, true), Some(32));
    // mismatches: depth-32 without alpha, depth-24 with alpha, and an absent depth
    assert_eq!(find_format(&formats, 32, false), None);
    assert_eq!(find_format(&formats, 24, true), None);
    assert_eq!(find_format(&formats, 8, false), None);
}

#[test]
fn pool_buffer_age_zero_until_primed_then_pool_n() {
    // 0 (= full repaint) until every buffer has been drawn once, then the pool size.
    assert_eq!(pool_buffer_age(0), 0);
    assert_eq!(pool_buffer_age(POOL_N as u64 - 1), 0);
    assert_eq!(pool_buffer_age(POOL_N as u64), POOL_N as i32);
    assert_eq!(pool_buffer_age(POOL_N as u64 + 5), POOL_N as i32);
}

#[test]
fn a8_stride_pads_to_32_bits() {
    // ZPixmap depth-8 rows pad to a 4-byte (32-bit) boundary.
    assert_eq!(a8_stride(0), 0);
    assert_eq!(a8_stride(1), 4);
    assert_eq!(a8_stride(4), 4);
    assert_eq!(a8_stride(5), 8);
    assert_eq!(a8_stride(7), 8);
    assert_eq!(a8_stride(8), 8);
}

#[test]
fn premul_color_premultiplies_and_clamps() {
    // opaque white → all channels full
    let w = premul_color([1.0, 1.0, 1.0, 1.0]);
    assert_eq!((w.red, w.green, w.blue, w.alpha), (65535, 65535, 65535, 65535));
    // half-alpha red → red & alpha premultiplied to ~half; g/b zero
    let r = premul_color([1.0, 0.0, 0.0, 0.5]);
    assert_eq!((r.green, r.blue), (0, 0));
    assert_eq!(r.alpha, scale_channel(0.5));
    assert_eq!(r.red, scale_channel(0.5)); // 1.0 · 0.5 premultiplied
    // out-of-range clamps (alpha clamps to 1 → red = 1·1)
    let c = premul_color([2.0, -1.0, 0.5, 2.0]);
    assert_eq!((c.red, c.green, c.alpha), (65535, 0, 65535));
}

#[test]
fn color_key_packs_rgba8() {
    assert_eq!(color_key([0.0, 0.0, 0.0, 0.0]), 0);
    assert_eq!(color_key([1.0, 0.0, 0.0, 1.0]), 0xFF_FF_00_00); // a=FF, r=FF
    assert_eq!(color_key([0.0, 1.0, 0.0, 1.0]), 0xFF_00_FF_00); // a=FF, g=FF
    assert_ne!(color_key([0.1, 0.2, 0.3, 1.0]), color_key([0.3, 0.2, 0.1, 1.0]));
}

#[test]
fn graph_bar_color_buckets_by_budget() {
    let budget = 16.6;
    let green = [0.40, 0.90, 0.50, 0.90];
    let amber = [0.95, 0.80, 0.30, 0.90];
    let red = [0.95, 0.40, 0.35, 0.90];
    assert_eq!(graph_bar_color(0.0, budget), green); // idle → green
    assert_eq!(graph_bar_color(budget * 0.5, budget), green); // exactly ½ → still green
    assert_eq!(graph_bar_color(budget * 0.7, budget), amber); // tight → amber
    assert_eq!(graph_bar_color(budget, budget), red); // at budget → red (missed vblank)
    assert_eq!(graph_bar_color(budget * 2.0, budget), red); // over → red
}

#[test]
fn outline_ring_taps() {
    // all-around = 8 taps; drop = 3 taps, all toward the bottom-right (dx,dy ≥ 0).
    assert_eq!(outline_ring(2.0, false).len(), 8);
    let drop = outline_ring(2.0, true);
    assert_eq!(drop.len(), 3);
    assert!(outline_ring(2.0, false).iter().any(|&(dx, dy)| (dx - 2.0).abs() < 1e-6 && dy == 0.0));
    assert!(drop.iter().all(|&(dx, dy)| dx >= 0.0 && dy >= 0.0));
}

#[test]
fn hud_anchor_places_each_corner() {
    let (sw, sh, pw, ph, m) = (1000.0, 800.0, 100.0, 50.0, 10.0);
    assert_eq!(hud_anchor(HudCorner::TopLeft, sw, sh, pw, ph, m), (10.0, 10.0));
    assert_eq!(hud_anchor(HudCorner::TopRight, sw, sh, pw, ph, m), (890.0, 10.0));
    assert_eq!(hud_anchor(HudCorner::BottomLeft, sw, sh, pw, ph, m), (10.0, 740.0));
    assert_eq!(hud_anchor(HudCorner::BottomRight, sw, sh, pw, ph, m), (890.0, 740.0));
}
