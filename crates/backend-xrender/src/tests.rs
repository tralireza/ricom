//! Pure-helper tests for `backend-xrender` — the damage-geometry + colour/opacity math
//! that builds each frame's repaint region and per-window blend. No X connection, so
//! these run on the Mac (the RENDER side effects in `present_windows` — `ensure_back`,
//! the per-window composites, the damage→overlay copy — need a live server → i7 only).

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
