//! Wire-format round-trips + socket-path derivation for `proto` — pure, no env / fs.

use super::*;

fn roundtrip_cmd(c: Command) {
    let bytes = encode(&c);
    assert_eq!(*bytes.last().unwrap(), b'\n');
    let back: Command = decode(&bytes).unwrap();
    assert_eq!(c, back);
}

#[test]
fn command_roundtrip() {
    roundtrip_cmd(Command::Ping);
    roundtrip_cmd(Command::Reload);
    roundtrip_cmd(Command::FpsToggle);
    roundtrip_cmd(Command::List);
    roundtrip_cmd(Command::Inspect { win: 0x1a00007 });
    roundtrip_cmd(Command::Notify { text: "hello".into(), timeout_ms: Some(3000) });
    roundtrip_cmd(Command::Notify { text: "no timeout".into(), timeout_ms: None });
    roundtrip_cmd(Command::Version);
    roundtrip_cmd(Command::Animate { win: 0x1a00007, effect: "spin".into(), params: vec![] });
    roundtrip_cmd(Command::Animate {
        win: 0x1a00007,
        effect: "ripple".into(),
        params: vec![("amplitude".into(), "0.12".into()), ("duration".into(), "4".into())],
    });
    roundtrip_cmd(Command::SetAnim { category: "close".into(), effect: "drain".into(), params: vec![] });
    roundtrip_cmd(Command::SetAnim {
        category: "close".into(),
        effect: "drain".into(),
        params: vec![("turns".into(), "3".into())],
    });
    roundtrip_cmd(Command::Unredir { enable: Some(true) });
    roundtrip_cmd(Command::Unredir { enable: Some(false) });
    roundtrip_cmd(Command::Unredir { enable: None });
    roundtrip_cmd(Command::Font { path: "/usr/share/fonts/DejaVuSans.ttf".into(), size: Some(1.25) });
    roundtrip_cmd(Command::Font { path: String::new(), size: None });
    roundtrip_cmd(Command::Quit);
    roundtrip_cmd(Command::FpsAutoMove { enable: Some(true) });
    roundtrip_cmd(Command::FpsAutoMove { enable: Some(false) });
    roundtrip_cmd(Command::FpsAutoMove { enable: None });
}

#[test]
fn effect_params_schema() {
    assert!(effect_params("ripple").unwrap().iter().any(|(k, _)| *k == "amplitude"));
    assert!(effect_params("reset").unwrap().is_empty());
    assert!(effect_params("bogus").is_none());
    assert!(EFFECTS.contains(&"drain"));
}

#[test]
fn effect_schematic_covers_effects() {
    // Every animate/set effect has a 3-row schematic + gloss, except `reset`
    // (a snap-to-rest with nothing to draw). Guards future EFFECTS additions.
    for &fx in EFFECTS {
        match effect_schematic(fx) {
            Some((gloss, art)) => {
                assert_ne!(fx, "reset", "reset has no motion to draw");
                assert!(!gloss.is_empty(), "{fx} gloss");
                assert_eq!(art.lines().count(), 3, "{fx} art should be 3 rows");
            }
            None => assert_eq!(fx, "reset", "{fx} is missing a schematic"),
        }
    }
    assert!(effect_schematic("bogus").is_none());
}

#[test]
fn reply_roundtrip() {
    let info = WinInfo {
        id: 42,
        class: "mpv".into(),
        instance: "mpv".into(),
        window_type: "normal".into(),
        title: "video".into(),
        mapped: true,
        x: 10,
        y: 20,
        width: 640,
        height: 480,
        opacity: 0.8,
        closing: false,
    };
    for r in [
        Reply::Ok,
        Reply::Text("ricom 0.1.0".into()),
        Reply::Error("no such window".into()),
        Reply::Window(info.clone()),
        Reply::Windows(vec![info]),
    ] {
        let back: Reply = decode(&encode(&r)).unwrap();
        assert_eq!(r, back);
    }
}

#[test]
fn path_xdg_preferred() {
    assert_eq!(
        path_for(Some("/run/user/1000"), 1000, ":0"),
        PathBuf::from("/run/user/1000/ricom-0.sock")
    );
}

#[test]
fn path_tmp_fallback() {
    assert_eq!(path_for(None, 1000, ":0"), PathBuf::from("/tmp/ricom-1000-0.sock"));
    assert_eq!(path_for(Some(""), 1000, ":1"), PathBuf::from("/tmp/ricom-1000-1.sock"));
}

#[test]
fn display_sanitised() {
    assert_eq!(sanitize_display(":0"), "0");
    assert_eq!(sanitize_display(":0.0"), "0_0");
    assert_eq!(sanitize_display(""), "");
    assert_eq!(sanitize_display(":10.2"), "10_2");
}
