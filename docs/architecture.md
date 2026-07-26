# Architecture

This document explains how zoomer is put together and, more usefully, why. The
[README](../README.md) covers what it does; this covers the decisions behind it.

## The pipeline

```
Camera ──frames──▶ HandTracker ──observations──▶ GestureEngine ──events──▶ InputBackend
   │                    │                             │                          │
 OpenCV             MediaPipe                  standard library only         pynput
```

Each arrow is a protocol, not a concrete class. That is what allows the whole
chain to be assembled from scripted stand-ins in tests without changing a line
of the code under test.

## Module map

| Module | Responsibility | Heavy dependencies |
| --- | --- | --- |
| `types.py` | Points, observations, events | none |
| `gestures/features.py` | Landmarks → pinch and pointer signals | none |
| `gestures/filters.py` | One-Euro filtering, deadzone | none |
| `gestures/state_machine.py` | Which gesture owns the hand | none |
| `gestures/engine.py` | Signals → discrete zoom and scroll steps | none |
| `calibration.py` | Range of motion → tuned gains | none |
| `config.py` | TOML loading and validation | none |
| `tracking/camera.py` | Frame capture | OpenCV |
| `tracking/hand_tracker.py` | Hand detection | MediaPipe |
| `tracking/landmarks.py` | MediaPipe results → domain types | none |
| `backends/desktop.py` | Synthetic keyboard and mouse input | pynput |
| `hud.py` | Diagnostic preview window | OpenCV |
| `app.py` | The run loop | none directly |
| `cli.py` | Argument parsing, error presentation | none directly |

The "none" column is the point. Two thirds of the codebase — including all of
the behaviour users actually experience — imports nothing but the standard
library, so it can be tested exhaustively and deterministically.

Modules that do need heavy dependencies import them lazily inside functions, so
`import zoomer` succeeds on a machine with no camera and no display server.

## Design decisions

### Why not render the PDF ourselves?

A built-in viewer would be far easier to control precisely and far easier to
test. It would also be the wrong product: people already have a PDF viewer they
like, with their bookmarks, their annotations, and their tabs in it. Emitting
synthetic input instead means zoomer works with every viewer that exists,
including ones that had not been written when it shipped.

The cost is real and worth stating: we cannot read the document's current zoom
level, we depend on the platform's accessibility permissions, and we are at the
mercy of each viewer's own keyboard shortcuts.

### Why normalise by hand size?

MediaPipe reports positions relative to the frame, so a hand held close to the
camera produces larger numbers than the same hand held further away. Any fixed
threshold on those raw numbers is correct at exactly one distance.

Dividing everything by the wrist-to-knuckle span removes that dependence. The
span is the right yardstick specifically because it is rigid: unlike a fingertip
distance, it does not change when the user pinches, so it measures apparent size
without being contaminated by the gesture being measured.

Every threshold in the configuration is therefore in hand-widths (or hand-widths
per second) and means the same thing for every user at every distance.

### Why arbitrate between gestures?

Opening the fingers to zoom drags the index tip upward. Raising the hand to
scroll changes the fingertip gap slightly. An unarbitrated pipeline responds to
every gesture by moving the document diagonally.

`ModeLock` grants ownership to one gesture at a time, decided by which signal
exceeded *its own* threshold by the larger proportion — comparing raw magnitudes
would be meaningless, since pinch speed and pointer speed are different physical
quantities.

Release uses a lower threshold than acquisition. With a single threshold, a
gesture performed near that speed would flicker in and out of activation several
times a second, which is unusable.

### Why a One-Euro filter rather than a moving average?

The requirement is contradictory on its face: a resting hand must be smoothed
hard enough not to drift, while a fast deliberate gesture must not lag. A filter
with a fixed cutoff can satisfy one or the other.

The [One-Euro filter][one-euro] raises its cutoff in proportion to the observed
speed of the signal, satisfying both. Its speed estimate is itself low-passed,
so that a single noisy sample cannot masquerade as a fast gesture and switch the
smoothing off exactly when it is needed most.

[one-euro]: https://gery.casiez.net/1euro/

### Why discrete events?

Hand motion is continuous; PDF viewers are not. Zoom is a keystroke and
scrolling is a wheel click.

The engine integrates the conditioned signal into an accumulator and emits a step
each time it crosses a whole unit, carrying the remainder forward. Because
nothing is rounded away, a gesture performed slowly moves the document exactly as
far as the same gesture performed quickly — it simply arrives more gradually.

### Guarding against imperfect input

Three failure modes are handled explicitly rather than left to chance:

- **Teleporting landmarks.** Trackers occasionally emit a wildly wrong point.
  `max_steps_per_frame` caps what one frame can do, and the clamped surplus is
  *dropped* rather than banked — carrying it forward would leave the document
  drifting for seconds after a single bad frame.
- **Dropped frames.** Losing the hand for a frame or two is routine.
  `max_missing_frames` holds the gesture across brief gaps; only a sustained
  absence abandons it. Without this, a slightly flaky feed never completes a
  single step.
- **Non-monotonic clocks.** Frame timestamps come from hardware. Repeated and
  rewound timestamps are skipped rather than divided by.

## Testing strategy

The dividing line is hardware. The camera, MediaPipe, and the preview window are
substituted; everything else under test is the code that ships.

- **Unit tests** cover each module exhaustively, including edge cases that would
  be impractical to produce with a real hand.
- **End-to-end tests** run whole sessions with the **default production
  settings**, so the tuning constants users actually receive are themselves under
  test. Every gesture is replayed a second time with reproducible landmark
  jitter, because a pipeline that only works on perfect input would not survive a
  webcam.
- **Hardware tests**, deselected by default, verify the assumptions the doubles
  encode: that the real model loads, that the real camera yields frames, that our
  timestamp handling satisfies MediaPipe's video mode.

Coverage sits above 90% with no hardware involved. The uncovered remainder is
almost entirely the thin wrappers around OpenCV, MediaPipe, and pynput, which the
hardware suite exercises.
