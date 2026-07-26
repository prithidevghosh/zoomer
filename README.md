# zoomer

Zoom and scroll any PDF with your thumb and index finger.

zoomer watches your hand through the camera you already have and drives whatever
document is in front of you. It does not open PDFs itself — it sends the same
keystrokes and wheel events a keyboard and mouse would, so it works in Preview,
Adobe Acrobat, Chrome, Firefox, Edge, Okular, Evince, or a PDF embedded in a web
page, on macOS, Windows, and Linux.

```
  ✋  →  📷  →  hand landmarks  →  gesture  →  keystrokes & wheel  →  📄
```

| Gesture | What the document does |
| --- | --- |
| Thumb and index finger **widening** | Zooms in |
| Thumb and index finger **closing** | Zooms out |
| Index finger **rising** | Scrolls up |
| Index finger **lowering** | Scrolls down |

Only the thumb and index finger are watched. Your other fingers can do whatever
they like.

## Install

Requires Python 3.11 or newer.

```bash
pip install zoomer
```

Or from a clone:

```bash
git clone https://github.com/prithidevghosh/zoomer
cd zoomer
pip install -e ".[dev]"
```

The hand-tracking model (about 7 MB) downloads automatically on first run and is
cached in `~/.cache/zoomer/`. To fetch it ahead of time:

```bash
zoomer download-model
```

## Use it

```bash
zoomer
```

A small preview window opens showing your hand. **Click your PDF viewer to give
it focus**, then gesture at the camera. Press <kbd>q</kbd> or <kbd>Esc</kbd> in
the preview window, or <kbd>Ctrl</kbd>+<kbd>C</kbd> in the terminal, to stop.

Try it safely first — this recognises gestures and shows the preview, but sends
nothing to any application:

```bash
zoomer --backend none
```

### Permissions

Using the camera is privileged, and so is synthesising input.

- **macOS** — grant your terminal both *Camera* and *Accessibility* permission
  under System Settings → Privacy & Security. Accessibility is what allows
  zoomer to control other applications; without it, gestures are recognised but
  nothing moves.
- **Windows** — no special setup.
- **Linux** — works under X11. Wayland compositors restrict synthetic input, so
  an X11 session may be required.

## Tuning

If gestures feel too sensitive or too sluggish, measure your own range of motion:

```bash
zoomer calibrate
```

Follow the prompt for eight seconds. It prints a snippet to paste into
`~/.config/zoomer/config.toml`.

<details>
<summary>Full configuration reference</summary>

Every value below is a default; omit anything you do not want to change.

```toml
show_hud = true             # open the camera preview window

[camera]
index = 0                   # try 1, 2, ... for an external camera
width = 640
height = 480
mirror = true               # preview only; does not affect recognition

[tracking]
min_detection_confidence = 0.5   # raise if the background is mistaken for a hand
min_tracking_confidence = 0.5    # lower if your hand is dropped mid-gesture
# model_path = "/path/to/hand_landmarker.task"

[input]
backend = "desktop"         # "none" recognises gestures without acting
zoom_mode = "keyboard"      # or "modifier_scroll" — see below
scroll_lines_per_click = 3  # raise to page through long documents faster

[gestures]
zoom_gain = 6.0             # zoom steps per hand-width of finger separation
scroll_gain = 8.0           # wheel clicks per hand-width of vertical travel
zoom_deadzone = 0.05        # pinch speed treated as noise
scroll_deadzone = 0.08      # pointer speed treated as noise
max_steps_per_frame = 3     # ceiling, so one bad frame cannot fire a burst
max_missing_frames = 5      # dropped frames tolerated before a gesture is dropped
min_cutoff = 0.8            # lower = steadier hand, more lag
beta = 0.01                 # higher = less lag on fast gestures, more jitter
derivative_cutoff = 1.0

[gestures.mode_lock]
zoom_enter = 0.55           # pinch speed at which zooming takes over
zoom_exit = 0.20            # ...and below which it lets go
scroll_enter = 0.70
scroll_exit = 0.25
settle_seconds = 0.25       # how long the hand must rest before switching gesture
```

Speeds are in hand-widths per second, so they mean the same thing regardless of
how far you sit from the camera.

**`zoom_mode`** — `keyboard` sends the platform zoom shortcut
(<kbd>Cmd</kbd>/<kbd>Ctrl</kbd> with <kbd>=</kbd> or <kbd>-</kbd>). It is the
only option native viewers such as Preview and Acrobat understand.
`modifier_scroll` holds the modifier and turns the wheel, which browsers zoom
more smoothly in but most native viewers ignore.

Command-line flags override the file, which is handy for trying a setting before
committing to it:

```bash
zoomer --camera 1 --zoom-mode modifier_scroll --no-hud
```

An unknown setting is a startup error rather than a silent no-op, so a typo
tells you about itself instead of leaving you wondering why nothing changed.

</details>

## Troubleshooting

**Nothing happens at all.** Check the percentage printed when the run ends. If
your hand was detected in fewer than half the frames, the problem is tracking,
not tuning: try more light, a plainer background, or keeping your hand away from
the edges of the frame. If detection is good but the document does not move,
zoomer likely lacks permission to control other applications — see
[Permissions](#permissions).

**It zooms when I meant to scroll.** Only one gesture is active at a time,
decided by whichever you started more decisively. Pause briefly before switching;
`settle_seconds` controls how long that pause needs to be.

**The document drifts on its own.** Raise `zoom_deadzone` and `scroll_deadzone`,
or lower `min_cutoff` for heavier smoothing.

**It reacts too slowly.** Raise `beta` to cut lag during fast gestures, and raise
the gains so a smaller movement goes further.

## How it works

Five stages, each independently testable:

1. **Capture** — OpenCV pulls frames from the camera.
2. **Track** — MediaPipe's hand landmarker locates the hand. Four of its
   twenty-one points are read: the thumb and index tips, plus the wrist and index
   knuckle.
3. **Measure** — the fingertip gap and the index height are divided by the
   wrist-to-knuckle span. That span is rigid, so it does not change when you
   pinch, which makes it a reliable yardstick — and dividing by it is what lets
   one set of thresholds work whether you sit near the camera or far from it.
4. **Decide** — a [One-Euro filter][one-euro] removes jitter without adding lag,
   a deadzone clamps what survives to zero, and a state machine grants control to
   exactly one gesture at a time, with hysteresis so it cannot flicker between
   them.
5. **Apply** — continuous motion is integrated into whole steps and sent as
   keystrokes and wheel events.

Stages 3 and 4 import nothing but the standard library, which is why the gesture
behaviour can be tested exhaustively without a camera.

[one-euro]: https://gery.casiez.net/1euro/

## Development

```bash
pip install -e ".[dev]"

pytest                    # the full suite; no camera or display needed
pytest -m hardware        # opt-in: real camera, model download, real input
ruff check . && ruff format --check . && mypy
```

The suite substitutes only the camera, MediaPipe, and the preview window.
Everything else under test is the code that ships, running with the same default
settings users get — so a change that makes the defaults unusable fails the
tests rather than reaching a release.

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Licence

[Apache License 2.0](LICENSE).
