//! Arg-parsing tests for `ricomctl` — pure (no socket / env / X), Mac-runnable.

use super::*;

fn parse(args: &[&str]) -> Result<Cli, Exit> {
    Cli::parse_from(args.iter().map(|s| (*s).to_string()))
}
fn cmd(args: &[&str]) -> Command {
    parse(args).unwrap().command
}

#[test]
fn commands_map() {
    assert_eq!(cmd(&["ping"]), Command::Ping);
    assert_eq!(cmd(&["reload"]), Command::Reload);
    assert_eq!(cmd(&["list"]), Command::List);
    assert_eq!(cmd(&["fps", "toggle"]), Command::FpsToggle);
    assert_eq!(cmd(&["inspect", "0x1a00007"]), Command::Inspect { win: 0x1a00007 });
    assert_eq!(cmd(&["inspect", "42"]), Command::Inspect { win: 42 });
    assert_eq!(cmd(&["notify", "hi"]), Command::Notify { text: "hi".into(), timeout_ms: None });
    assert_eq!(cmd(&["notify", "hi", "3"]), Command::Notify { text: "hi".into(), timeout_ms: Some(3000) });
    assert_eq!(cmd(&["version"]), Command::Version);
    assert_eq!(
        cmd(&["animate", "0x1a00007", "spin"]),
        Command::Animate { win: 0x1a00007, effect: "spin".into(), params: vec![] }
    );
    assert_eq!(
        cmd(&["animate", "42", "wobble"]),
        Command::Animate { win: 42, effect: "wobble".into(), params: vec![] }
    );
    // Trailing key=value tokens become ordered params (server types + validates them).
    assert_eq!(
        cmd(&["animate", "0x1", "ripple", "amplitude=0.1", "duration=3"]),
        Command::Animate {
            win: 1,
            effect: "ripple".into(),
            params: vec![("amplitude".into(), "0.1".into()), ("duration".into(), "3".into())],
        }
    );
}

#[test]
fn globals_before_command() {
    let cli = parse(&["--json", "--socket", "/tmp/x.sock", "list"]).unwrap();
    assert!(cli.json);
    assert_eq!(cli.socket, Some(PathBuf::from("/tmp/x.sock")));
    assert_eq!(cli.command, Command::List);
}

#[test]
fn help_and_version_go_to_stdout() {
    assert!(matches!(parse(&["-h"]), Err(Exit::Stdout(_))));
    assert!(matches!(parse(&["--version"]), Err(Exit::Stdout(_))));
}

#[test]
fn errors_are_usage() {
    assert!(matches!(parse(&[]), Err(Exit::Usage(_))));
    assert!(matches!(parse(&["bogus"]), Err(Exit::Usage(_))));
    assert!(matches!(parse(&["fps"]), Err(Exit::Usage(_))));
    assert!(matches!(parse(&["fps", "nope"]), Err(Exit::Usage(_))));
    assert!(matches!(parse(&["inspect"]), Err(Exit::Usage(_))));
    assert!(matches!(parse(&["inspect", "zz"]), Err(Exit::Usage(_))));
    assert!(matches!(parse(&["list", "extra"]), Err(Exit::Usage(_))));
    assert!(matches!(parse(&["--socket"]), Err(Exit::Usage(_))));
    assert!(matches!(parse(&["notify"]), Err(Exit::Usage(_))));
    assert!(matches!(parse(&["animate"]), Err(Exit::Usage(_))));
    assert!(matches!(parse(&["animate", "0x1"]), Err(Exit::Usage(_))));
    assert!(matches!(parse(&["animate", "zz", "spin"]), Err(Exit::Usage(_))));
    assert!(matches!(parse(&["animate", "0x1", "spin", "extra"]), Err(Exit::Usage(_))));
}
