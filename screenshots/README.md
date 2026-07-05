# Screenshots

Captures of `ricom` running as the compositor.

- `ricom-blur.png` — dual-Kawase background blur frosting the backdrop behind a translucent terminal, over fullscreen video with a picture-in-picture corner overlay.
- `ricom.png` — per-window opacity, fade in/out, and left+bottom drop shadows on live windows.
- `ricom-demo.gif` — a `ricomctl`-driven effects showreel: boing-open, dissolve, and live spin/stretch/wobble on video windows with OSD captions (15 s loop, no audio, 480 px). Made with the `ffmpeg` x11grab + `palettegen`/`paletteuse` recipe.
- The full-quality 1080p mp4 is embedded in the README as a GitHub user-attachment (uploaded via the web editor, not committed here — keeps the repo light). Local master copy: `target/ricom_demo_15s.mp4`.

Grabbed on an Intel HD Graphics 630 / Mesa box (X11, no window manager) with
[`maim`](https://github.com/naelstrof/maim) (`-u` hides the cursor; the root window — full screen — is
the default):

```sh
DISPLAY=:0 maim -u ricom-blur.png
```
