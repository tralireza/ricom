# Screenshots

Captures of `ricom` running as the compositor.

- `ricom-blur.png` — dual-Kawase background blur frosting the backdrop behind a translucent terminal, over fullscreen video with a picture-in-picture corner overlay.
- `ricom.png` — per-window opacity, fade in/out, and left+bottom drop shadows on live windows.

Grabbed on an Intel HD Graphics 630 / Mesa box (X11, no window manager) with
[`maim`](https://github.com/naelstrof/maim) (`-u` hides the cursor; the root window — full screen — is
the default):

```sh
DISPLAY=:0 maim -u ricom-blur.png
```
