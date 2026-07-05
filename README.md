# ricom

A **minimalistic X11 compositor written in Rust** — a from-scratch reimplementation of the core of
[picom](https://github.com/yshui/picom).

`ricom` redirects the screen and composites all windows onto the X composite overlay using
**OpenGL via EGL** (texture-from-pixmap), presenting tear-free with vsync.

## Screenshots

![ricom compositing — dual-Kawase background blur behind a translucent window](screenshots/ricom-blur.png)

*Dual-Kawase background blur — the backdrop behind a translucent terminal is frosted, over fullscreen
video with a picture-in-picture corner overlay. Composited tear-free.*

![ricom compositing — per-window opacity, fades, and drop shadows](screenshots/ricom.png)

*Per-window opacity, fade in/out, and left+bottom drop shadows — composited tear-free.*

![ricom's on-demand FPS HUD with the 1m/5m/15m load block, over running video](screenshots/ricom-fps.png)

*The on-demand FPS HUD (toggled with `Super+Shift+F`) — FPS, frame-time, a rolling frame-time
graph, and a `loadavg`-style 1m/5m/15m block (fps + GPU render time), drawn by the built-in SDF
text engine over running video. The same figures are logged on `SIGUSR1`.*

## Status

Working today:

- **X11 bring-up** — connect, negotiate Composite / Damage / Render / Present / RandR / Shape /
  Sync / XFixes, and become the compositing manager (`_NET_WM_CM_S0`).
- **Window tracking** — an incremental bottom-to-top stack maintained from X structure events
  (create / map / unmap / configure / restack / destroy).
- **GL backend** — an EGL context on the composite overlay, **texture-from-pixmap**
  (`EGLImage` → `glEGLImageTargetTexture2DOES`), a GLSL blit, and `eglSwapBuffers` with
  `swap_interval(1)` for vsync.
- **Renderer** — composite the visible window stack (mapped + fading-out) back-to-front with
  per-window opacity and drop shadows; **damage-driven**, plus a frame clock while anything animates.
  **Region-level occlusion culling** paints each window only where it isn't hidden behind an opaque
  one, and **`use-damage` partial repaint** (EGL buffer-age) redraws only the region that changed —
  so a static screen with one updating window repaints just that window, not the whole surface.
- **Resolution changes** — follows RandR screen-size changes (`xrandr`) and re-composites at the new size.
- **unredir-if-possible** — when one window covers the whole screen (e.g. fullscreen video), ricom
  unredirects and steps aside so it page-flips straight to the display (compositor cost → ~0); it drops
  back to compositing the instant a smaller window sits on top (e.g. a corner overlay), so that case
  stays tear-free.
- **Effects** — **per-window opacity** (`_NET_WM_WINDOW_OPACITY`), **fades** in on map and out on
  unmap/destroy (200 ms ease-out on a `calloop` frame clock; a closing window's last frame is kept and
  faded), soft **left+bottom drop shadows**, **rounded corners** (shadow follows the corner), and
  **background blur** — dual-Kawase frost behind translucent windows.
- **Transition animations** — a composable **animation-block** system: each transition (open / close
  / move) plays a set of layered primitives — **opacity, scale, translate, wobble, burn** — chosen by a
  named preset (`fade`, `pop`, `slide`, `drop`, `boing`, `burn`, `wobble`, `stretch`, `unroll`, `minimize`, `spin`) or an
  explicit block spec, set globally (`[anim]`) or per-window (`[[rule]]`). Includes the scale-about-centre
  **open/close "pop"**, **wobbly-windows** (a spring-mesh move/resize jelly on a dedicated GL mesh path),
  **slide/drop** (an eased translate), **directional stretch/unroll** (a centre line growing to full
  width/height), and **spin** (a GPU rotate-about-centre). All ride `use-damage`, so an animating window
  repaints only its moving path, not the whole screen.
- **On-demand FPS HUD** — a global hotkey (`Super+Shift+F` by default) toggles an overlay showing
  FPS, frame-time, and a rolling frame-time graph, drawn with a general **SDF text engine**
  (arbitrary strings, crisp at any size, no runtime font dependency) — ricom's first on-screen text.
  The hotkey's modifiers + arrow keys move it between corners live, and it auto-scales with resolution
  (2× at 4K).
- **Load average** — a `loadavg`-style 1m/5m/15m rolling average of compositor FPS and GPU
  render time (from a per-second ring), shown as a block in the FPS HUD and logged on demand
  with `kill -USR1 $(pidof ricom)`. Damage-driven, so it reads ~idle during fullscreen bypass
  (ricom stepped aside) rather than showing false load.
- **Window rules** — per-window overrides matched on `WM_CLASS` (class/instance),
  `_NET_WM_WINDOW_TYPE`, title (substring), and fullscreen state, each setting `opacity` /
  `blur` / `shadow` / `corner_radius` / `unredir` / `above`, plus the per-transition animations
  `open` / `close` / `move` (a preset or explicit block spec; an empty `match = {}` is a global
  default). Precedence: an explicit `_NET_WM_WINDOW_OPACITY` beats a rule, which beats a built-in
  "fullscreen → opaque + unblurred" rule, which beats the global `default_opacity`. Live-reloads
  with the rest of the config.

Runs tear-free as the compositor on an Intel HD Graphics 630 (Mesa): fullscreen + windowed video at
1920×1080@60 (on par with picom), and 3840×2160@30 with fullscreen bypass.

**Not yet implemented:** window dimming, and the xrender/glx backends + D-Bus IPC.
See [Roadmap](#roadmap).

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
ties them together in the event loop. (`region` drives both occlusion culling — each window is
painted only where it isn't covered by an opaque window above — and `use-damage` partial repaint:
only the region that changed since the back buffer was last drawn, tracked via EGL buffer age.)

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
   ├─ backend-gl  EGL context on the overlay, texture-from-pixmap, blit/shadow/blur/mesh/SDF-text shaders, present
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
DISPLAY=:0 ./target/release/ricom --fps      # …with the FPS HUD visible (toggle: Super+Shift+F)
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

## Configuration

ricom reads an optional TOML file from `$XDG_CONFIG_HOME/ricom/ricom.toml` (falling back to
`~/.config/ricom/ricom.toml`); with no file it uses built-in defaults. Pass `--config <path>` to
use a different file, `--print-config` to dump the effective settings, and **`kill -HUP $(pidof
ricom)`** to reload live — no restart. Every key is optional and falls back to its default:

```toml
unredir = true                  # false = always composite, even a lone fullscreen window
background = [0.05, 0.05, 0.07]  # composite background colour (RGB, seen where no window covers)
corner_radius = 0.0             # window corner radius in px (0 = square)
default_opacity = 1.0           # opacity for windows with no _NET_WM_WINDOW_OPACITY and no rule

[shadow]
enabled = true
radius = 12.0                   # left/bottom falloff distance (px)
strength = 0.45                 # peak shadow alpha
min_size = 24                   # skip shadows for windows smaller than this (px)

[blur]
enabled = false                 # frost the backdrop behind translucent windows
passes = 3                      # dual-Kawase iterations (wider/softer)
radius = 4.0                    # sample offset per pass (px)

[anim]                          # per-transition animations built from composable blocks
open  = "pop"                   # presets: none|fade|pop|slide|drop|boing|burn|wobble|stretch|unroll|minimize|spin
close = "fade"                  # …or compose blocks explicitly (see ricom.toml.example)
move  = "wobble"
duration = 0.2                  # default seconds (opacity / scale / translate)
scale_from = 0.85               # default `scale` start factor (open) / end factor (close)
wobble_spring = 350.0           # wobble spring stiffness k (higher = snappier)
wobble_friction = 14.0          # wobble velocity damping (higher = less jiggle)

[fps]
enabled = false                 # start with the FPS HUD visible (also toggled by the hotkey)
hotkey = "Super+Shift+F"        # toggle shortcut (XGrabKey); its modifiers + arrows move corners live
corner = "top-right"            # initial corner: top-left | top-right | bottom-left | bottom-right
graph = true                    # rolling frame-time graph under the numbers
scale = 1.0                     # size multiplier on top of auto screen-height scaling (4K = 2×)

# Per-window rules (none by default). Each [[rule]] has a `match` (all conditions must hold —
# class/instance/window_type exact, title substring, fullscreen state) plus the fields it
# overrides; applied in order, last match wins. A built-in rule keeps fullscreen windows opaque.
[[rule]]
match = { class = "mpv" }       # video: never dim or blur
opacity = 1.0
blur = false
shadow = false

[[rule]]
match = { class = "com.mitchellh.ghostty" }  # frosted terminals (Ghostty's default X11
opacity = 0.85                               # class; confirm with `xprop WM_CLASS`)
blur = true

[[rule]]
match = { window_type = "dock" }  # no shadow on panels/bars
shadow = false

[[rule]]
match = { instance = "Alacritty" }  # terminal: pop open, slide away on close
open = "pop"
close = "slide"
```

See [`ricom.toml.example`](ricom.toml.example) for the full schema, every preset, and
explicit block composition.

## Roadmap

Done: per-window opacity, fade in/out, left+bottom drop shadows, rounded corners, background blur
(dual-Kawase), a TOML config file with live (SIGHUP) reload, an on-demand FPS HUD (global hotkey)
built on a general SDF text engine, per-window rules (match on class/type/title/fullscreen), a
loadavg-style 1m/5m/15m FPS + render-time meter (SIGUSR1 / HUD block), region-level occlusion
culling (skip windows/pixels hidden behind an opaque one), `use-damage` partial repaint
(EGL buffer-age; repaint only the changed region), and a composable transition-animation system —
layered primitives (opacity / scale / translate / wobble / burn) selected per transition (open /
close / move) by a named preset or explicit block spec, globally or per-rule: pop, slide/drop,
wobbly-windows, burn dissolve, directional stretch/unroll, and a GPU spin (rotate-about-centre).

Next:

1. Window dimming (per-window / rule — slots into the rules engine).

## License

MPL-2.0. `ricom` is a port of picom (MPL-2.0); data structures and GLSL are derived from it.

The bundled SDF glyph atlas (`crates/backend-gl/src/glyphs.bin`) is generated from
[Liberation Mono](https://github.com/liberationfonts) (SIL Open Font License 1.1) — see [`NOTICE`](NOTICE).
