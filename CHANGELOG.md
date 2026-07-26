# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Thumb-and-index pinch gestures that zoom the focused PDF viewer in and out.
- Vertical index-finger movement that scrolls the focused document up and down.
- Mode-lock state machine with hysteresis so pinching and swiping never fight each other.
- One-Euro filtering, deadzones, and hand-size normalisation for stable, distance-invariant input.
- Cross-platform input backends for macOS, Windows, and Linux that drive any focused
  application: Preview, Acrobat, Chrome, Firefox, Edge, Okular, Evince.
- Optional heads-up display showing the camera feed, tracked landmarks, and live signals.
- TOML configuration with strict validation, plus an interactive calibration
  command that measures the user's range of motion and prints tuned gains.
- Tolerance for brief tracking dropouts, so a gesture survives the frames a hand
  tracker routinely loses.
- A `none` input backend for trying gestures out without touching any
  application.

[Unreleased]: https://github.com/prithidevghosh/zoomer/commits/main
