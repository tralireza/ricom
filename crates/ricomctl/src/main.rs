//! ricomctl — a thin control client for a running ricom compositor.
//!
//! Connects to ricom's Unix control socket, sends one command, prints the reply.
//! Arg parsing is hand-rolled but *clap-shaped* (no `clap` dependency) so a later
//! `#[derive(Parser)]` swap would be mechanical. It links only `proto` + std — no
//! EGL/GL, nothing from the compositor's render graph.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use proto::{Command, Reply};

const HELP: &str = "\
ricomctl — control a running ricom compositor

USAGE:
    ricomctl [OPTIONS] <COMMAND>

OPTIONS:
    --json            Print the raw JSON reply instead of formatted text
    --socket <PATH>   Control socket path (default: ricom's per-DISPLAY socket)
    -h, --help        Print this help
    -V, --version     Print version

COMMANDS:
    ping              Check the compositor is alive
    reload            Re-read + apply the config file (same as SIGHUP)
    fps toggle        Toggle the FPS HUD
    list              List tracked windows
    inspect <win>     Show one window (id: decimal or 0x hex)
    notify <text> [s] Show an on-screen message for [s] seconds (default: config)
    version           Show ricom's version (on-screen toast + stdout)
    animate <win> <fx> Play a transform on one window
                      (fx: spin|pop|stretch|unroll|slide|wobble|reset)

EXAMPLES:
    ricomctl list
    ricomctl inspect 0x1a00007
    ricomctl notify \"hello ricom\" 3
    ricomctl animate 0x1a00007 spin
    ricomctl reload
";

/// A parsed invocation: global options + the command to send.
struct Cli {
    json: bool,
    socket: Option<PathBuf>,
    command: Command,
}

/// How to terminate before connecting: print help/version (stdout, exit 0) or a
/// usage error (stderr, exit 2). Mirrors `clap::Error`'s two outcomes.
#[derive(Debug)]
enum Exit {
    Stdout(String),
    Usage(String),
}

impl Cli {
    fn parse() -> Result<Cli, Exit> {
        Cli::parse_from(std::env::args().skip(1))
    }

    /// Core parser (testable, no argv/env). Global options come *before* the
    /// command (`ricomctl --json list`); the first non-option token starts the
    /// command and everything after it belongs to the command.
    fn parse_from<I: IntoIterator<Item = String>>(args: I) -> Result<Cli, Exit> {
        let mut json = false;
        let mut socket = None;
        let mut rest: Vec<String> = Vec::new();
        let mut it = args.into_iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "-h" | "--help" => return Err(Exit::Stdout(HELP.to_string())),
                "-V" | "--version" => {
                    return Err(Exit::Stdout(format!("ricomctl {}\n", env!("CARGO_PKG_VERSION"))));
                }
                "--json" => json = true,
                "--socket" => {
                    let v = it.next().ok_or_else(|| Exit::Usage("--socket needs a PATH\n".into()))?;
                    socket = Some(PathBuf::from(v));
                }
                _ => {
                    rest.push(a);
                    rest.extend(it);
                    break;
                }
            }
        }
        let command = parse_command(&rest)?;
        Ok(Cli { json, socket, command })
    }
}

/// Parse the subcommand tokens into a `proto::Command`.
fn parse_command(args: &[String]) -> Result<Command, Exit> {
    let mut a = args.iter().map(String::as_str);
    let cmd = a
        .next()
        .ok_or_else(|| Exit::Usage(format!("missing command\n\n{HELP}")))?;
    let out = match cmd {
        "ping" => Command::Ping,
        "reload" => Command::Reload,
        "list" => Command::List,
        "version" => Command::Version,
        "fps" => match a.next() {
            Some("toggle") => Command::FpsToggle,
            Some(other) => {
                return Err(Exit::Usage(format!("unknown fps subcommand '{other}' (want: toggle)\n")));
            }
            None => return Err(Exit::Usage("fps needs a subcommand (toggle)\n".into())),
        },
        "inspect" => {
            let w = a.next().ok_or_else(|| Exit::Usage("inspect needs a <win> id\n".into()))?;
            Command::Inspect { win: parse_win(w)? }
        }
        "notify" => {
            let text = a.next().ok_or_else(|| Exit::Usage("notify needs <text>\n".into()))?;
            let timeout_ms = match a.next() {
                Some(s) => Some(
                    s.parse::<f64>()
                        .map(|secs| (secs * 1000.0) as u32)
                        .map_err(|_| Exit::Usage(format!("invalid timeout '{s}' (seconds)\n")))?,
                ),
                None => None,
            };
            Command::Notify { text: text.to_string(), timeout_ms }
        }
        "animate" => {
            let w = a.next().ok_or_else(|| Exit::Usage("animate needs a <win> id and an <effect>\n".into()))?;
            let fx = a
                .next()
                .ok_or_else(|| Exit::Usage("animate needs an <effect> (spin|pop|stretch|unroll|slide|wobble|reset)\n".into()))?;
            Command::Animate { win: parse_win(w)?, effect: fx.to_string() }
        }
        other => return Err(Exit::Usage(format!("unknown command '{other}'\n\n{HELP}"))),
    };
    if let Some(extra) = a.next() {
        return Err(Exit::Usage(format!("unexpected argument '{extra}'\n")));
    }
    Ok(out)
}

/// Parse an X window id as decimal or `0x…` hex.
fn parse_win(s: &str) -> Result<proto::WinId, Exit> {
    let parsed = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => u32::from_str_radix(hex, 16),
        None => s.parse::<u32>(),
    };
    parsed.map_err(|_| Exit::Usage(format!("invalid window id '{s}' (want decimal or 0x hex)\n")))
}

fn main() -> ExitCode {
    let cli = match Cli::parse() {
        Ok(c) => c,
        Err(Exit::Stdout(s)) => {
            print!("{s}");
            return ExitCode::SUCCESS;
        }
        Err(Exit::Usage(s)) => {
            eprint!("{s}");
            return ExitCode::from(2);
        }
    };
    let path = cli.socket.clone().unwrap_or_else(proto::socket_path);
    let stream = match std::os::unix::net::UnixStream::connect(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ricomctl: cannot reach ricom at {} ({e})", path.display());
            eprintln!(
                "  is the compositor running for DISPLAY={}?",
                std::env::var("DISPLAY").unwrap_or_default()
            );
            return ExitCode::from(1);
        }
    };
    match exchange(stream, &cli.command) {
        Ok(reply) => print_reply(&reply, cli.json),
        Err(e) => {
            eprintln!("ricomctl: {e}");
            ExitCode::from(1)
        }
    }
}

/// Send one command and read the single reply line.
fn exchange(mut stream: std::os::unix::net::UnixStream, cmd: &Command) -> std::io::Result<Reply> {
    stream.set_read_timeout(Some(std::time::Duration::from_secs(2)))?;
    stream.write_all(&proto::encode(cmd))?;
    stream.flush()?;
    let mut reader = BufReader::new(stream);
    let mut line = Vec::new();
    reader.read_until(b'\n', &mut line)?;
    proto::decode::<Reply>(&line).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Render a reply to the terminal, returning the process exit code.
fn print_reply(reply: &Reply, json: bool) -> ExitCode {
    if json {
        std::io::stdout().write_all(&proto::encode(reply)).ok();
        return match reply {
            Reply::Error(_) => ExitCode::from(1),
            _ => ExitCode::SUCCESS,
        };
    }
    match reply {
        Reply::Ok => ExitCode::SUCCESS,
        Reply::Text(s) => {
            println!("{s}");
            ExitCode::SUCCESS
        }
        Reply::Window(w) => {
            print_windows(std::slice::from_ref(w));
            ExitCode::SUCCESS
        }
        Reply::Windows(ws) => {
            print_windows(ws);
            ExitCode::SUCCESS
        }
        Reply::Error(e) => {
            eprintln!("ricomctl: {e}");
            ExitCode::from(1)
        }
    }
}

/// Print windows as an aligned table (bottom-to-top order, as received).
fn print_windows(ws: &[proto::WinInfo]) {
    if ws.is_empty() {
        println!("(no windows)");
        return;
    }
    println!("{:<12} {:<20} {:^4} {:>5}  {:<14} TITLE", "ID", "CLASS", "MAP", "OPAC", "GEOMETRY");
    for w in ws {
        let geom = format!("{}x{}+{}+{}", w.width, w.height, w.x, w.y);
        println!(
            "0x{:<10x} {:<20} {:^4} {:>5.2}  {:<14} {}",
            w.id,
            trunc(&w.class, 20),
            if w.mapped { "yes" } else { "no" },
            w.opacity,
            geom,
            w.title, // full title — it's the last column, so no alignment concern
        );
    }
}

/// Truncate to `n` chars with an ellipsis if longer.
fn trunc(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n.saturating_sub(1)).chain(['…']).collect()
    }
}

// ── Migration to clap (when a dep is acceptable) ──────────────────────────────
// cargo add clap --features derive; then:
//   #[derive(Parser)] struct Cli { #[arg(long)] json: bool,
//     #[arg(long)] socket: Option<PathBuf>, #[command(subcommand)] command: Cmd }
//   #[derive(Subcommand)] enum Cmd { Ping, Reload, Fps{..}, List, Inspect{win:String} }
// and delete parse_from / parse_command / parse_win / HELP / Exit.

#[cfg(test)]
mod tests;
