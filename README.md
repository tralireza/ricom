# ricom

A small **X11 compositor written in Rust** — a from-scratch reimplementation of the core of
[picom](https://github.com/yshui/picom).

`ricom` redirects the screen and composites all windows onto the X composite overlay using
**OpenGL via EGL** (texture-from-pixmap), presenting tear-free with vsync.

## Screenshots

![ricom compositing — per-window opacity, fades, and drop shadows](screenshots/ricom.png)

*Per-window opacity, fade in/out, and left+bottom drop shadows — composited tear-free.*

## Status — MVP (compositing core)

Working today:

- **X11 bring-up** — connect, negotiate Composite / Damage / Render / Present / RandR / Shape /
  Sync / XFixes, and become the compositing manager (`_NET_WM_CM_S0`).
- **Window tracking** — an incremental bottom-to-top stack maintained from X structure events
  (create / map / unmap / configure / restack / destroy).
- **GL backend** — an EGL context on the composite overlay, **texture-from-pixmap**
  (`EGLImage` → `glEGLImageTargetTexture2DOES`), a GLSL blit, and `eglSwapBuffers` with
  `swap_interval(1)` for vsync.
- **Renderer** — composite the mapped window stack back-to-front, **damage-driven** redraw loop.
- **Resolution changes** — follows RandR screen-size changes (`xrandr`) and re-composites at the new size.
- **unredir-if-possible** — when one window covers the whole screen (e.g. fullscreen video), ricom
  unredirects and steps aside so it page-flips straight to the display (compositor cost → ~0); it drops
  back to compositing the instant a smaller window sits on top (e.g. a corner overlay), so that case
  stays tear-free.

Validated on an Intel HD Graphics 630 (Mesa): tear-free fullscreen + windowed video at 1920×1080@60
(on par with picom), and 3840×2160@30 with fullscreen bypass.

**Not yet implemented** (beyond the MVP): effects (shadow, fade, blur, rounded corners, dimming),
the `use-damage` partial-repaint optimisation, animations, a config file, the xrender/glx
backends, and D-Bus IPC. See [Roadmap](#roadmap).

## How it works

ricom redirects every top-level window into an off-screen pixmap, binds each pixmap as a
GL texture, and draws them back-to-front onto the X composite overlay — which the X server
then scans out as a single tear-free frame:

```
        +-----------------------------------------------------+
        |  X CLIENTS:  mpv, browser, xterm, ...               |
        +-----------------------------------------------------+
             |  each app draws into its own top-level window
             v
        +-----------------------------------------------------+
        |  X SERVER  (Composite + Damage extensions)          |
        +-----------------------------------------------------+
             |  Composite redirect_subwindows(Manual):
             |  every window is rendered to an OFF-SCREEN pixmap
             |     +--------+ +--------+ +--------+  one per window
             |     |pixmap A| |pixmap B| |pixmap C|
             |     +--------+ +--------+ +--------+
             |  Damage -> DamageNotify when a window's pixels change
             v
        +-----------------------------------------------------+
        |  ricom  (xconn, wm, region, backend-gl)             |
        +-----------------------------------------------------+
             |  - bind each pixmap as a GL texture, zero-copy:
             |      eglCreateImage(EGL_NATIVE_PIXMAP_KHR)
             |        -> glEGLImageTargetTexture2DOES
             |  - draw mapped windows bottom-to-top as textured
             |    quads at their on-screen geometry
             v
        +-----------------------------------------------------+
        |  COMPOSITE OVERLAY WINDOW  (owned by ricom)         |
        +-----------------------------------------------------+
             |  eglSwapBuffers + swap_interval(1)  =>  vsync
             v
        +-----------------------------------------------------+
        |  MONITOR - one tear-free, fully-composited frame    |
        +-----------------------------------------------------+
```

The loop is **damage-driven**: ricom waits on the X connection with `calloop`, and X events
drive a single dirty flag —

```
DamageNotify, MapNotify, UnmapNotify, ConfigureNotify, ...  ->  mark dirty
   dirty  ->  recomposite the mapped stack  ->  eglSwapBuffers (vsync)
```

The stages map onto the crates: **xconn** speaks the X protocol (extension setup, become-CM,
overlay + redirect, `NameWindowPixmap`, damage); **wm** keeps the bottom-to-top window stack
in sync with structure events; **backend-gl** owns the EGL context and does texture-from-pixmap,
the blit, and the vsync present; **region** is the pixman-style damage maths; and **session**
ties them together in the event loop. (Today every frame is a full-screen repaint — `region`
is there for the `use-damage` partial-repaint optimisation on the roadmap.)

When one window covers the whole screen with nothing on top (e.g. fullscreen video), ricom
**unredirects** — it unmaps the overlay and steps out of the way so that window page-flips directly to
the display, dropping the compositor's GPU and memory-bandwidth cost to ~0 (`unredir-if-possible`). The
moment a smaller window appears on top (a corner overlay), it re-redirects and resumes compositing, so
the overlay-over-video case stays tear-free.

## Architecture

A Cargo workspace whose root package is the `ricom` binary; the library crates live under
`crates/`:

```
ricom             workspace root + binary (event-loop wiring, CLI)
└─ crates/
   ├─ region      pure-Rust pixman-style rectangle regions (damage maths)
   ├─ xconn       x11rb wrapper: connection, extensions, atoms, overlay/redirect, pixmap/damage
   ├─ wm          window model + bottom-to-top stacking, updated from X events
   ├─ backend-gl  EGL context on the overlay, texture-from-pixmap, GLSL blit, present
   └─ session     the compositor: owns X + wm + backend, runs the calloop event loop
```

Dependencies are pure-Rust: [`x11rb`](https://github.com/psychon/x11rb) (XCB protocol),
[`calloop`](https://github.com/Smithay/calloop) (event loop),
[`khronos-egl`](https://crates.io/crates/khronos-egl) + [`glow`](https://github.com/grovesNL/glow)
(EGL / GL), [`x11-dl`](https://crates.io/crates/x11-dl) (Xlib handle for EGL only), `tracing`,
`anyhow`.

> EGL needs a native display handle that the pure-Rust `x11rb` connection doesn't expose, so —
> exactly as picom does — `ricom` opens an Xlib `Display` purely as EGL's display / window-surface
> handle while doing all protocol and events over `x11rb`. X window ids are server-global, so the
> overlay id is shared between the two.

## Build & run

Requires a Rust toolchain and a Linux system with X11 + EGL (Mesa).

```sh
cargo build --release
DISPLAY=:0 ./target/release/ricom            # run as the compositor (Ctrl-C to quit)
```

Diagnostics:

```sh
DISPLAY=:0 ./target/release/ricom --gl-check    # headless EGL/GL smoke test (no screen impact)
DISPLAY=:0 ./target/release/ricom --paint-test  # clear the overlay to a colour
DISPLAY=:0 ./target/release/ricom --blit-test   # composite all windows for 5s
./target/release/ricom --help                   # usage + examples (no X needed)
./target/release/ricom --version                # print version
```

`RUST_LOG=debug` raises log verbosity.

> Running `ricom` acquires `_NET_WM_CM_S0`; stop any other compositor (`pkill -x picom`) first.
> On exit the X server auto-releases the redirect, so the screen returns to normal drawing.

## Roadmap

1. `use-damage` partial repaint — repaint only damaged regions (biggest win on mostly-static screens).
2. Effects: shadow, fade, per-window opacity.
3. Config file.
4. Blur, rounded corners, window rules.
5. Animations (picom-style transition scripts).

## License

MPL-2.0. `ricom` is a port of picom (MPL-2.0); data structures and GLSL are derived from it.
