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
    roundtrip_cmd(Command::Animate { win: 0x1a00007, effect: "spin".into() });
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
