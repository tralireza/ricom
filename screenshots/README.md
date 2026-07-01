# Screenshots

Captures of `ricom` running as the compositor.

- `ricom.png` — per-window opacity, fade in/out, and left+bottom drop shadows on live windows.

Grabbed on an Intel HD Graphics 630 / Mesa box (X11, no window manager) with:

```sh
DISPLAY=:0 ffmpeg -y -f x11grab -video_size 1920x1080 -i :0.0 -frames:v 1 ricom.png
```
