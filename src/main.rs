//! ricom: an X11 compositor (Rust rewrite of picom). P0: bring-up + event log.

use anyhow::Result;
use tracing_subscriber::EnvFilter;

const HELP: &str = "\
ricom — a minimalistic X11 compositor (EGL/GL, tear-free vsync)

USAGE:
    ricom [OPTION]

With no option, ricom runs as the compositor: it becomes the compositing
manager (_NET_WM_CM_S0), redirects the screen, and composites every window
onto the X composite overlay until killed (Ctrl-C).

OPTIONS:
    --gl-check      Headless EGL/GL smoke test: print GPU vendor/renderer/version, exit.
    --paint-test    Clear the composite overlay to a solid colour for 4s, then exit.
    --blit-test     Composite all mapped windows onto the overlay for 5s, then exit.
    --opacity-test [FRAC]
                    Like --blit-test but draw every window at opacity FRAC
                    (0.0..1.0, default 0.5) — exercises the alpha-blend path.
    --config <PATH> Use this config file instead of the default location.
    --print-config  Print the effective config as TOML and exit (no X needed).
    --fps           Start with the FPS HUD visible (toggle live with its hotkey).
    -h, --help      Print this help and exit.
    -V, --version   Print version and exit.

ENVIRONMENT:
    DISPLAY         X display to connect to (e.g. :0). Required.
    RUST_LOG        Log level: error|warn|info|debug|trace (default: info).

CONFIG:
    TOML read from $XDG_CONFIG_HOME/ricom/ricom.toml (or ~/.config/ricom/ricom.toml)
    if present, else built-in defaults. Send SIGHUP (`kill -HUP <pid>`) to reload live.

EXAMPLES:
    DISPLAY=:0 ricom                   # run as the compositor
    DISPLAY=:0 ricom --gl-check        # verify EGL/GL works on this GPU
    RUST_LOG=debug DISPLAY=:0 ricom    # run with debug logging
    DISPLAY=:0 ricom --blit-test       # one-shot: composite current windows for 5s
    DISPLAY=:0 ricom --opacity-test 0.5   # composite current windows at 50% opacity
    ricom --print-config               # show effective settings and exit

Stop any other compositor first (e.g. `pkill -x picom`), since ricom must own
_NET_WM_CM_S0.
";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Handle --help / --version first — before logging or any X connection.
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{HELP}");
        return Ok(());
    }
    if args.iter().any(|a| a == "-V" || a == "--version") {
        println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    // Reject unknown flags so a typo doesn't silently launch the compositor.
    const FLAGS: &[&str] = &[
        "--gl-check", "--paint-test", "--blit-test", "--opacity-test", "--config", "--print-config",
        "--fps",
    ];
    if let Some(bad) = args[1..]
        .iter()
        .find(|a| a.starts_with('-') && !FLAGS.contains(&a.as_str()))
    {
        eprintln!("ricom: unknown option '{bad}'\nTry `ricom --help`.");
        std::process::exit(2);
    }

    // `RUST_LOG=debug ricom` to raise verbosity; defaults to info.
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    // `ricom --gl-check` runs a headless EGL/GL smoke test and exits (no compositor).
    if args.iter().any(|a| a == "--gl-check") {
        let info = backend_gl::first_light()?;
        println!("vendor:   {}", info.vendor);
        println!("renderer: {}", info.renderer);
        println!("version:  {}", info.version);
        return Ok(());
    }

    // `ricom --paint-test` grabs the composite overlay and clears it to colour via GL.
    if args.iter().any(|a| a == "--paint-test") {
        return paint_test();
    }

    // `ricom --blit-test` redirects + composites all mapped windows onto the overlay.
    if args.iter().any(|a| a == "--blit-test") {
        return composite_windows_test(1.0);
    }

    // `ricom --opacity-test [FRAC]` composites all windows at a fixed opacity
    // (0.0..1.0, default 0.5) to exercise the alpha-blend path.
    if let Some(pos) = args.iter().position(|a| a == "--opacity-test") {
        let frac = args
            .get(pos + 1)
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.5)
            .clamp(0.0, 1.0);
        return composite_windows_test(frac);
    }

    // Config: `--config <path>` overrides the default XDG location.
    let config_path = args
        .iter()
        .position(|a| a == "--config")
        .and_then(|i| args.get(i + 1))
        .map(std::path::PathBuf::from);
    let mut cfg = match config::Config::load(config_path.as_deref()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ricom: {e:#}");
            std::process::exit(2);
        }
    };
    // `--fps` starts with the HUD visible (still toggleable via its hotkey).
    if args.iter().any(|a| a == "--fps") {
        cfg.fps.enabled = true;
    }

    // `ricom --print-config` dumps the effective settings as TOML and exits.
    if args.iter().any(|a| a == "--print-config") {
        print!("{}", cfg.to_toml());
        return Ok(());
    }

    tracing::info!("ricom starting");
    let mut app = session::App::new(cfg, config_path)?;
    app.run()?;
    Ok(())
}

/// Increment 2a: acquire the composite overlay, make it input-transparent,
/// create a GL window surface on it, and clear it to teal for a few seconds.
fn paint_test() -> Result<()> {
    use xconn::XConn;
    let x = XConn::connect()?;
    x.setup_extensions()?;
    let overlay = x.get_overlay()?;
    x.overlay_input_passthrough(overlay)?;
    let visual = x.window_visual(overlay)?;
    tracing::info!(overlay, visual, "composite overlay acquired (input-passthrough)");
    x.flush()?;
    {
        let backend = backend_gl::GlBackend::new(overlay, visual, backend_gl::RenderParams::default())?;
        for _ in 0..3 {
            backend.clear_present(0.06, 0.45, 0.55, 1.0)?;
        }
        tracing::info!("overlay painted teal; holding 4s");
        std::thread::sleep(std::time::Duration::from_secs(4));
    } // backend dropped here -> EGL torn down
    x.release_overlay()?;
    x.flush()?;
    tracing::info!("overlay released");
    Ok(())
}

/// Increment 2b/3: redirect, then composite ALL mapped windows (bottom-to-top)
/// onto the overlay via texture-from-pixmap, looped for a few seconds. Every
/// window is drawn at `opacity` (1.0 = the plain --blit-test; <1.0 exercises the
/// alpha-blend path for --opacity-test).
fn composite_windows_test(opacity: f32) -> Result<()> {
    use backend_gl::Quad;
    use xconn::XConn;
    let x = XConn::connect()?;
    x.setup_extensions()?;
    let overlay = x.get_overlay()?;
    x.overlay_input_passthrough(overlay)?;
    let visual = x.window_visual(overlay)?;

    x.redirect_subwindows()?; // server stops drawing; ricom owns the screen now

    // Name a pixmap for every mapped window (bottom-to-top), excluding the overlay.
    let mut items: Vec<Quad> = Vec::new();
    let mut pixmaps: Vec<u32> = Vec::new();
    for w in x.list_tree()? {
        if !w.mapped || w.window == overlay {
            continue;
        }
        match x.name_window_pixmap(w.window) {
            Ok(pm) => {
                // Named pixmap includes the border (size w+2bw); geometry x,y is
                // already the outer corner, so blit AT (x,y) with the bordered size.
                let bw = w.border_width as i32;
                let (qw, qh) = (w.width as i32 + 2 * bw, w.height as i32 + 2 * bw);
                items.push(Quad {
                    pixmap: pm,
                    x: w.x as i32,
                    y: w.y as i32,
                    w: qw,
                    h: qh,
                    opacity,
                    shadow: qw >= 24 && qh >= 24,
                    blur: false, // diagnostic path: blur is a compositor-runtime effect
                });
                pixmaps.push(pm);
            }
            Err(e) => tracing::warn!("name pixmap for {}: {e}", w.window),
        }
    }
    tracing::info!(count = items.len(), opacity, "compositing mapped windows");
    x.flush()?;
    {
        let backend = backend_gl::GlBackend::new(overlay, visual, backend_gl::RenderParams::default())?;
        let start = std::time::Instant::now();
        while start.elapsed().as_secs() < 5 {
            backend.present_windows(&items, x.root_width as i32, x.root_height as i32, None)?;
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        tracing::info!("composited {} windows for 5s", items.len());
    }
    for pm in pixmaps {
        let _ = x.free_pixmap(pm);
    }
    x.unredirect_subwindows()?;
    x.release_overlay()?;
    x.flush()?;
    tracing::info!("done; redirect released");
    Ok(())
}
