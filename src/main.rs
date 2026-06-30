//! ricom: an X11 compositor (Rust rewrite of picom). P0: bring-up + event log.

use anyhow::Result;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    // `RUST_LOG=debug ricom` to raise verbosity; defaults to info.
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    // `ricom --gl-check` runs a headless EGL/GL smoke test and exits (no compositor).
    if std::env::args().any(|a| a == "--gl-check") {
        let info = backend_gl::first_light()?;
        println!("vendor:   {}", info.vendor);
        println!("renderer: {}", info.renderer);
        println!("version:  {}", info.version);
        return Ok(());
    }

    // `ricom --paint-test` grabs the composite overlay and clears it to colour via GL.
    if std::env::args().any(|a| a == "--paint-test") {
        return paint_test();
    }

    // `ricom --blit-test` redirects + blits the largest window's pixmap onto the overlay.
    if std::env::args().any(|a| a == "--blit-test") {
        return blit_test();
    }

    tracing::info!("ricom starting");
    let mut app = session::App::new()?;
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
        let backend = backend_gl::GlBackend::new(overlay, visual)?;
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
/// onto the overlay via texture-from-pixmap, looped for a few seconds.
fn blit_test() -> Result<()> {
    use xconn::XConn;
    let x = XConn::connect()?;
    x.setup_extensions()?;
    let overlay = x.get_overlay()?;
    x.overlay_input_passthrough(overlay)?;
    let visual = x.window_visual(overlay)?;

    x.redirect_subwindows()?; // server stops drawing; ricom owns the screen now

    // Name a pixmap for every mapped window (bottom-to-top), excluding the overlay.
    let mut items: Vec<(u32, i32, i32, i32, i32)> = Vec::new();
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
                items.push((
                    pm,
                    w.x as i32,
                    w.y as i32,
                    w.width as i32 + 2 * bw,
                    w.height as i32 + 2 * bw,
                ));
                pixmaps.push(pm);
            }
            Err(e) => tracing::warn!("name pixmap for {}: {e}", w.window),
        }
    }
    tracing::info!(count = items.len(), "compositing mapped windows");
    x.flush()?;
    {
        let backend = backend_gl::GlBackend::new(overlay, visual)?;
        let start = std::time::Instant::now();
        while start.elapsed().as_secs() < 5 {
            backend.present_windows(&items, x.root_width as i32, x.root_height as i32)?;
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
