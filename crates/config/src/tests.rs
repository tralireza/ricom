//! config: parsing / resolve / diff tests (moved out of the parent module; see `#[cfg(test)] mod tests;`).

use super::*;

#[test]
fn osd_defaults_and_parse() {
    let c = Config::default();
    assert!(c.osd.enabled);
    assert_eq!(c.osd.duration, 2.5);
    assert_eq!(c.osd.scale, 1.0);
    assert_eq!(c.osd.open, OsdEffect::Stretch);
    assert_eq!(c.osd.in_dur, 0.06);
    let c: Config = toml::from_str("[osd]\nenabled = false\nduration = 4.0\nopen = \"pop\"\nclose = \"fade\"\n").unwrap();
    assert!(!c.osd.enabled);
    assert_eq!(c.osd.duration, 4.0);
    assert_eq!(c.osd.open, OsdEffect::Pop);
    assert_eq!(c.osd.close, OsdEffect::Fade);
}

#[test]
fn defaults_match_compiled_behaviour() {
    let c = Config::default();
    assert!(c.unredir);
    assert!(c.use_damage);
    assert_eq!(c.background, [0.05, 0.05, 0.07]);
    assert_eq!(c.corner_radius, 0.0);
    assert_eq!(c.anim.duration, 0.2);
    assert_eq!(
        (c.shadow.enabled, c.shadow.radius, c.shadow.strength, c.shadow.min_size),
        (true, 12.0, 0.45, 24)
    );
    assert_eq!((c.blur.enabled, c.blur.passes, c.blur.radius), (false, 3, 4.0));
    assert_eq!((c.dim.enabled, c.dim.strength), (false, 0.3));
    assert_eq!(c.dim.focus, FocusSource::Ewmh);
    assert!(!c.fps.enabled);
    assert_eq!(c.fps.hotkey, "Super+Shift+F");
    assert_eq!(c.fps.corner, "top-right");
    assert!(c.fps.graph);
    assert_eq!(c.fps.scale, 1.0);
    assert_eq!(c.default_opacity, 1.0);
    assert_eq!(c.anim.scale_from, 0.85);
    assert_eq!((c.anim.wobble_spring, c.anim.wobble_friction), (350.0, 14.0));
    assert_eq!(c.anim.open, AnimSel::Preset("pop".into()));
    assert_eq!(c.anim.close, AnimSel::Preset("fade".into()));
    assert_eq!(c.anim.r#move, AnimSel::Preset("wobble".into()));
    assert_eq!((c.burn.seg_scale, c.burn.ember_width), (36.0, 0.07));
    assert_eq!(c.burn.ember_cool, [0.28, 0.02, 0.0]);
    assert_eq!(c.burn.ember_hot, [0.75, 0.22, 0.04]);
    assert!(c.rules.is_empty());
}

#[test]
fn full_toml_parses() {
    let t = r#"
unredir = false
background = [0.1, 0.2, 0.3]
corner_radius = 8.0
[shadow]
enabled = true
radius = 30.0
strength = 0.7
min_size = 40
[blur]
enabled = true
passes = 5
radius = 6.0
[fps]
enabled = true
hotkey = "Control+Alt+P"
corner = "bottom-left"
graph = false
scale = 2.0
[anim]
duration = 0.4
scale_from = 0.7
wobble_spring = 500.0
wobble_friction = 20.0
open = "slide"
close = "burn"
move = "none"
[burn]
seg_scale = 18.0
ember_width = 0.08
ember_cool = [0.3, 0.02, 0.0]
ember_hot = [0.75, 0.25, 0.05]
"#;
    let c: Config = toml::from_str(t).unwrap();
    assert!(!c.unredir);
    assert_eq!(c.background, [0.1, 0.2, 0.3]);
    assert_eq!(c.corner_radius, 8.0);
    assert_eq!(c.anim.duration, 0.4);
    assert_eq!((c.shadow.radius, c.shadow.min_size), (30.0, 40));
    assert_eq!((c.blur.enabled, c.blur.passes, c.blur.radius), (true, 5, 6.0));
    assert!(c.fps.enabled);
    assert_eq!(c.fps.hotkey, "Control+Alt+P");
    assert_eq!(c.fps.corner, "bottom-left");
    assert!(!c.fps.graph);
    assert_eq!(c.fps.scale, 2.0);
    assert_eq!(c.anim.scale_from, 0.7);
    assert_eq!((c.anim.wobble_spring, c.anim.wobble_friction), (500.0, 20.0));
    assert_eq!(c.anim.open, AnimSel::Preset("slide".into()));
    assert_eq!(c.anim.close, AnimSel::Preset("burn".into()));
    assert_eq!(c.anim.r#move, AnimSel::Preset("none".into()));
    assert_eq!((c.burn.seg_scale, c.burn.ember_width), (18.0, 0.08));
    assert_eq!(c.burn.ember_cool, [0.3, 0.02, 0.0]);
    assert_eq!(c.burn.ember_hot, [0.75, 0.25, 0.05]);
}

#[test]
fn partial_toml_fills_defaults() {
    // Override just one shadow field; everything else stays at its default.
    let c: Config = toml::from_str("[shadow]\nradius = 20.0\n").unwrap();
    assert_eq!(c.shadow.radius, 20.0);
    assert_eq!(c.shadow.strength, 0.45); // default
    assert!(c.shadow.enabled); // default
    assert!(c.unredir); // default
    assert_eq!(c.anim.duration, 0.2); // default
}

#[test]
fn unknown_key_errors() {
    assert!(toml::from_str::<Config>("wobble = true\n").is_err());
}

#[test]
fn wrong_type_errors() {
    assert!(toml::from_str::<Config>("unredir = \"yes\"\n").is_err());
}

#[test]
fn diff_reports_changed_fields_only() {
    let a = Config::default();
    assert!(a.diff(&a).is_empty()); // identical -> no changes
    let mut b = Config::default();
    b.shadow.radius = 30.0;
    b.unredir = false;
    let d = b.diff(&a);
    assert_eq!(d.len(), 2);
    assert!(d.iter().any(|s| s.contains("shadow.radius") && s.contains("30.0")));
    assert!(d.iter().any(|s| s.contains("unredir") && s.contains("false")));
}

#[test]
fn toml_roundtrips() {
    let c = Config::default();
    let back: Config = toml::from_str(&c.to_toml()).unwrap();
    assert_eq!(c, back);
}

#[test]
fn builtin_fullscreen_rule_keeps_opaque_unblurred() {
    let c = Config::default();
    let r = c.resolve(&WindowMatch { fullscreen: true, ..Default::default() });
    assert_eq!(r.opacity, Some(1.0));
    assert_eq!(r.blur, Some(false));
    // non-fullscreen, no user rules -> nothing overridden
    assert_eq!(c.resolve(&WindowMatch::default()), RuleResult::default());
}

#[test]
fn rules_match_and_override() {
    let t = r#"
[[rule]]
match = { class = "mpv" }
opacity = 1.0
blur = false
shadow = false

[[rule]]
match = { title = "Picture-in-Picture" }
corner_radius = 12.0
"#;
    let c: Config = toml::from_str(t).unwrap();
    assert_eq!(c.rules.len(), 2);
    let r = c.resolve(&WindowMatch { class: "mpv".into(), ..Default::default() });
    assert_eq!((r.opacity, r.blur, r.shadow), (Some(1.0), Some(false), Some(false)));
    // substring title match
    let pip = WindowMatch { title: "YouTube - Picture-in-Picture".into(), ..Default::default() };
    assert_eq!(c.resolve(&pip).corner_radius, Some(12.0));
    // no match -> empty
    assert_eq!(
        c.resolve(&WindowMatch { class: "firefox".into(), ..Default::default() }),
        RuleResult::default()
    );
}

#[test]
fn user_rule_overrides_builtin_and_last_wins() {
    let t = r#"
[[rule]]
match = { fullscreen = true }
opacity = 0.5

[[rule]]
match = { class = "mpv" }
opacity = 0.9
"#;
    let c: Config = toml::from_str(t).unwrap();
    // fullscreen non-mpv: the user fullscreen rule overrides the built-in 1.0
    assert_eq!(c.resolve(&WindowMatch { fullscreen: true, ..Default::default() }).opacity, Some(0.5));
    // fullscreen mpv: the later mpv rule wins
    let fs_mpv = WindowMatch { class: "mpv".into(), fullscreen: true, ..Default::default() };
    assert_eq!(c.resolve(&fs_mpv).opacity, Some(0.9));
}

#[test]
fn dim_rule_override_and_fullscreen() {
    let t = r#"
[dim]
enabled = true
strength = 0.5
focus = "x11"

[[rule]]
match = { class = "mpv" }
dim = false
"#;
    let c: Config = toml::from_str(t).unwrap();
    assert_eq!((c.dim.enabled, c.dim.strength), (true, 0.5));
    assert_eq!(c.dim.focus, FocusSource::X11);
    // mpv rule: never dim
    assert_eq!(c.resolve(&WindowMatch { class: "mpv".into(), ..Default::default() }).dim, Some(false));
    // other window: no per-window override → follows global [dim]
    assert_eq!(c.resolve(&WindowMatch { class: "x".into(), ..Default::default() }).dim, None);
    // fullscreen: built-in never-dim
    assert_eq!(c.resolve(&WindowMatch { fullscreen: true, ..Default::default() }).dim, Some(false));
}

#[test]
fn rule_unknown_field_errors() {
    assert!(toml::from_str::<Config>("[[rule]]\nwobble = true\n").is_err());
}

#[test]
fn above_rule_matches_by_title_substring() {
    let c: Config = toml::from_str(
        "[[rule]]\nmatch = { title = \"intel-gpu-top\" }\nabove = true\n",
    )
    .unwrap();
    // Substring match against the live WM_NAME (e.g. "intel-gpu-top: Intel …").
    let w = WindowMatch { title: "intel-gpu-top: Intel Kabylake".into(), ..Default::default() };
    assert_eq!(c.resolve(&w).above, Some(true));
    // A non-matching window is untouched.
    assert_eq!(c.resolve(&WindowMatch { title: "xterm".into(), ..Default::default() }).above, None);
}

// --- Composable animation blocks -----------------------------------------

const FADE_BLOCKS: [Primitive; 1] = [Primitive::Opacity { from: None, easing: Easing::EaseOut }];

#[test]
fn preset_expands_to_blocks() {
    assert_eq!(expand_sel(&AnimSel::Preset("fade".into())).blocks, FADE_BLOCKS);
    let pop = expand_sel(&AnimSel::Preset("pop".into())).blocks;
    assert_eq!(pop.len(), 2);
    assert!(matches!(pop[0], Primitive::Opacity { .. }));
    assert!(matches!(pop[1], Primitive::Scale { .. }));
    assert_eq!(expand_sel(&AnimSel::Preset("burn".into())).blocks, [Primitive::Burn]);
    assert!(expand_sel(&AnimSel::Preset("none".into())).blocks.is_empty());
    // Unknown preset -> empty spec (validate() reports it separately).
    assert!(expand_sel(&AnimSel::Preset("bogus".into())).blocks.is_empty());
}

#[test]
fn stretch_and_unroll_are_directional_scales() {
    assert_eq!(
        expand_sel(&AnimSel::Preset("stretch".into())).blocks,
        [Primitive::Scale { from: Some(0.0), axis: Axis::X, easing: Easing::EaseOut }]
    );
    assert_eq!(
        expand_sel(&AnimSel::Preset("unroll".into())).blocks,
        [Primitive::Scale { from: Some(0.0), axis: Axis::Y, easing: Easing::EaseOut }]
    );
}

#[test]
fn minimize_shrinks_and_slides_off_bottom() {
    let b = expand_sel(&AnimSel::Preset("minimize".into())).blocks;
    assert_eq!(
        b,
        [
            Primitive::Scale { from: Some(0.0), axis: Axis::Both, easing: Easing::EaseIn },
            Primitive::Translate { dx: 0.0, dy: 0.0, edge: Some(Edge::Bottom), easing: Easing::EaseIn },
        ]
    );
}

#[test]
fn spin_preset_fades_and_rotates() {
    let b = expand_sel(&AnimSel::Preset("spin".into())).blocks;
    assert_eq!(b.len(), 2);
    assert!(matches!(b[0], Primitive::Opacity { .. }));
    assert_eq!(b[1], Primitive::Spin { degrees: None, easing: Easing::EaseOut });
}

#[test]
fn spin_degrees_parses() {
    let c: Config =
        toml::from_str("[anim.open]\nblocks = [ { block = \"spin\", degrees = 360.0 } ]\n").unwrap();
    let AnimSel::Spec(s) = &c.anim.open else { panic!("expected spec") };
    assert_eq!(s.blocks, [Primitive::Spin { degrees: Some(360.0), easing: Easing::EaseOut }]);
}

#[test]
fn scale_axis_parses() {
    let c: Config =
        toml::from_str("[anim.open]\nblocks = [ { block = \"scale\", from = 0.0, axis = \"x\" } ]\n")
            .unwrap();
    let AnimSel::Spec(s) = &c.anim.open else { panic!("expected explicit spec") };
    assert_eq!(s.blocks, [Primitive::Scale { from: Some(0.0), axis: Axis::X, easing: Easing::EaseOut }]);
    // axis defaults to Both when omitted.
    let c: Config = toml::from_str("[anim.open]\nblocks = [ { block = \"scale\" } ]\n").unwrap();
    let AnimSel::Spec(s) = &c.anim.open else { panic!("expected explicit spec") };
    assert_eq!(s.blocks, [Primitive::Scale { from: None, axis: Axis::Both, easing: Easing::EaseOut }]);
}

#[test]
fn explicit_blocks_parse() {
    let t = r#"
[anim.open]
duration = 0.3
blocks = [
  { block = "opacity", easing = "ease-out" },
  { block = "translate", dy = -60.0, easing = "ease-in" },
]
"#;
    let c: Config = toml::from_str(t).unwrap();
    let AnimSel::Spec(s) = &c.anim.open else { panic!("expected explicit spec") };
    assert_eq!(s.duration, Some(0.3));
    assert_eq!(s.blocks.len(), 2);
    assert!(matches!(s.blocks[0], Primitive::Opacity { easing: Easing::EaseOut, .. }));
    let Primitive::Translate { dy, easing, .. } = s.blocks[1] else { panic!("expected translate") };
    assert_eq!(dy, -60.0);
    assert_eq!(easing, Easing::EaseIn);
}

#[test]
fn catchall_rule_sets_close_fade_for_all() {
    let t = r#"
[anim]
close = "burn"

[[rule]]
match = {}
close = "fade"
"#;
    let c: Config = toml::from_str(t).unwrap();
    // Global default is burn, but the empty-match rule overrides every window.
    let any = WindowMatch { class: "anything".into(), ..Default::default() };
    assert_eq!(c.spec_for(&any, Category::Close).blocks, FADE_BLOCKS);
}

#[test]
fn mpv_rule_keeps_burn_while_default_is_fade() {
    let t = r#"
[[rule]]
match = { class = "mpv" }
close = "burn"
"#;
    let c: Config = toml::from_str(t).unwrap();
    // Default close is fade (opacity only)…
    let ff = WindowMatch { class: "firefox".into(), ..Default::default() };
    assert_eq!(c.spec_for(&ff, Category::Close).blocks, FADE_BLOCKS);
    // …but mpv still burns.
    let mpv = WindowMatch { class: "mpv".into(), ..Default::default() };
    assert_eq!(c.spec_for(&mpv, Category::Close).blocks, [Primitive::Burn]);
}

#[test]
fn spec_for_falls_back_to_global_default() {
    let c = Config::default();
    let w = WindowMatch::default();
    assert_eq!(c.spec_for(&w, Category::Open).blocks.len(), 2); // pop = opacity + scale
    assert_eq!(c.spec_for(&w, Category::Close).blocks, FADE_BLOCKS);
    assert_eq!(
        c.spec_for(&w, Category::Move).blocks,
        [Primitive::Wobble { spring: None, friction: None }]
    );
}

#[test]
fn validate_warns_unknown_preset_and_bad_combo() {
    let c: Config = toml::from_str("[anim]\nopen = \"sparkle\"\n").unwrap();
    assert!(c.validate().iter().any(|s| s.contains("anim.open") && s.contains("sparkle")));

    // burn is not a valid `move` block.
    let c: Config = toml::from_str("[anim]\nmove = \"burn\"\n").unwrap();
    assert!(c.validate().iter().any(|s| s.contains("anim.move") && s.contains("burn")));

    // The default config is clean.
    assert!(Config::default().validate().is_empty());
}
