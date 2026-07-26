"""Command-line entry point.

Three commands cover the whole product: ``run`` to control a document, ``calibrate``
to learn the user's range of motion, and ``download-model`` to fetch the tracking
bundle ahead of time for offline use.

Errors from every layer surface here as a single readable line rather than a
traceback. The failures users actually hit — a camera in use, a missing
permission, a typo in the config — are ordinary conditions, not bugs, and a
stack trace would obscure the one sentence that tells them what to do.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from zoomer import __version__
from zoomer.app import Session, SessionStats, run_calibration
from zoomer.backends import create_backend
from zoomer.calibration import CalibrationError, CalibrationResult, Calibrator
from zoomer.config import Config, ConfigError, default_config_path, load_config
from zoomer.gestures.engine import GestureEngine
from zoomer.hud import Hud, HudView, NullHud
from zoomer.tracking.camera import Camera, CameraError
from zoomer.tracking.hand_tracker import (
    MediaPipeHandTracker,
    TrackerError,
    default_model_path,
    ensure_model,
)

__all__ = ["build_parser", "main"]

_EXIT_OK = 0
_EXIT_ERROR = 1
_EXIT_INTERRUPTED = 130  # The conventional status for a run ended by Ctrl-C.


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        A parser covering every command and option.
    """
    parser = argparse.ArgumentParser(
        prog="zoomer",
        description=(
            "Zoom and scroll any PDF viewer with thumb-and-index hand gestures. "
            "Point your camera at your hand, put a document in front of you, and "
            "pinch to zoom or raise your index finger to scroll."
        ),
    )
    parser.add_argument("--version", action="version", version=f"zoomer {__version__}")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        metavar="FILE",
        help=f"configuration file to load (default: {default_config_path()})",
    )

    commands = parser.add_subparsers(dest="command")

    run = commands.add_parser("run", help="control the focused document (default)")
    _add_run_options(run)

    calibrate = commands.add_parser(
        "calibrate",
        help="measure your range of motion and print tuned gains",
    )
    calibrate.add_argument(
        "--seconds",
        type=float,
        default=8.0,
        metavar="N",
        help="how long to record for (default: 8)",
    )
    _add_camera_options(calibrate)

    commands.add_parser(
        "download-model",
        help="fetch the hand tracking model now, for later offline use",
    )

    # Options also accepted with no subcommand, so plain `zoomer` runs.
    _add_run_options(parser)
    return parser


def _add_camera_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--camera",
        type=int,
        metavar="N",
        help="camera device index (default: 0)",
    )
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help="hide the camera preview window",
    )


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    _add_camera_options(parser)
    parser.add_argument(
        "--backend",
        choices=["desktop", "none"],
        help="where to send gestures; 'none' recognises without acting (default: desktop)",
    )
    parser.add_argument(
        "--zoom-mode",
        choices=["keyboard", "modifier_scroll"],
        help=(
            "how to express zoom: 'keyboard' works in every viewer, "
            "'modifier_scroll' is smoother but browser-only (default: keyboard)"
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line.

    Args:
        argv: Arguments to parse. Defaults to :data:`sys.argv`.

    Returns:
        A process exit status.
    """
    args = build_parser().parse_args(argv)

    try:
        config = _apply_overrides(load_config(args.config), args)

        match args.command:
            case "calibrate":
                return _calibrate(config, args.seconds)
            case "download-model":
                return _download_model(config)
            case _:
                return _run(config)

    except KeyboardInterrupt:
        print("\nstopped.", file=sys.stderr)
        return _EXIT_INTERRUPTED
    except (ConfigError, CameraError, TrackerError, CalibrationError, ValueError) as error:
        print(f"zoomer: {error}", file=sys.stderr)
        return _EXIT_ERROR
    except RuntimeError as error:
        print(f"zoomer: {error}", file=sys.stderr)
        return _EXIT_ERROR


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Let command-line flags win over the configuration file.

    Args:
        config: Configuration as loaded from disk.
        args: Parsed command-line arguments.

    Returns:
        The configuration with any explicitly supplied flags applied.
    """
    camera = config.camera
    if getattr(args, "camera", None) is not None:
        camera = replace(camera, index=args.camera)

    inputs = config.input
    if getattr(args, "backend", None) is not None:
        inputs = replace(inputs, backend=args.backend)
    if getattr(args, "zoom_mode", None) is not None:
        inputs = replace(inputs, zoom_mode=args.zoom_mode)

    show_hud = config.show_hud and not getattr(args, "no_hud", False)

    return replace(config, camera=camera, input=inputs, show_hud=show_hud)


def _make_hud(config: Config) -> HudView:
    return Hud(mirror=config.camera.mirror) if config.show_hud else NullHud()


def _open_camera(config: Config) -> Camera:
    return Camera(
        index=config.camera.index,
        width=config.camera.width,
        height=config.camera.height,
    )


def _open_tracker(config: Config) -> MediaPipeHandTracker:
    model = Path(config.tracking.model_path) if config.tracking.model_path else None
    return MediaPipeHandTracker(
        model,
        min_detection_confidence=config.tracking.min_detection_confidence,
        min_tracking_confidence=config.tracking.min_tracking_confidence,
    )


def _run(config: Config) -> int:
    """Control the focused document until the user quits."""
    session = Session(
        source=_open_camera(config),
        tracker=_open_tracker(config),
        engine=GestureEngine(config.gestures),
        backend=create_backend(
            config.input.backend,
            zoom_mode=config.input.zoom_mode,
            scroll_lines_per_click=config.input.scroll_lines_per_click,
        ),
        hud=_make_hud(config),
    )

    if config.input.backend == "none":
        print("recognising gestures only; nothing will be sent to any application.")
    print("press q or escape in the preview window, or ctrl-c here, to stop.")

    with session:
        stats = session.run()

    _report(stats)
    return _EXIT_OK


def _report(stats: SessionStats) -> None:
    """Summarise a finished run, flagging poor tracking if that is why it did little."""
    print(
        f"processed {stats.frames} frames, "
        f"saw a hand in {stats.detection_rate:.0%}, "
        f"applied {stats.zoom_steps:+d} zoom steps and {stats.scroll_clicks:+d} scroll clicks."
    )
    if stats.frames and stats.detection_rate < 0.5:
        print(
            "tip: your hand was detected in fewer than half the frames. Try more "
            "light, a plainer background, or holding your hand further from the edges.",
            file=sys.stderr,
        )


def _calibrate(config: Config, seconds: float) -> int:
    """Measure the user's range of motion and print the settings it implies."""
    if seconds <= 0:
        raise ValueError(f"--seconds must be positive, got {seconds}")

    print(
        f"calibrating for {seconds:.0f} seconds. Keep your hand in view and repeat "
        "both gestures fully: open and close your thumb and index finger a few "
        "times, then raise and lower your hand a few times."
    )

    camera = _open_camera(config)
    tracker = _open_tracker(config)
    hud = _make_hud(config)
    calibrator = Calibrator()

    try:
        result = run_calibration(_TimeLimited(camera, seconds), tracker, calibrator, hud)
    finally:
        for component in (hud, tracker, camera):
            component.close()

    _print_calibration(result, config)
    return _EXIT_OK


def _print_calibration(result: CalibrationResult, config: Config) -> None:
    """Print the measured ranges and the TOML the user should save."""
    tuned = result.tune(config.gestures)

    print(f"\nmeasured over {result.samples} frames with a hand in view:")
    print(f"  pinch range:   {result.pinch_range:.2f} hand-widths")
    print(f"  pointer range: {result.pointer_range:.2f} hand-widths")
    print(f"\nadd this to {default_config_path()}:\n")
    print("[gestures]")
    print(f"zoom_gain = {tuned.zoom_gain:.2f}")
    print(f"scroll_gain = {tuned.scroll_gain:.2f}")


def _download_model(config: Config) -> int:
    """Fetch the tracking model so a later run can work offline."""
    target = Path(config.tracking.model_path) if config.tracking.model_path else None
    path = ensure_model(target)
    size_mb = path.stat().st_size / 1_048_576
    print(f"hand tracking model ready at {path} ({size_mb:.1f} MB)")
    return _EXIT_OK


class _TimeLimited:
    """Wrap a frame source so it stops after a fixed duration.

    Calibration needs a bounded recording, but a camera streams indefinitely.
    Deriving the cut-off from frame timestamps rather than a wall clock keeps
    the behaviour identical for a scripted source in tests.

    Args:
        source: The source to limit.
        seconds: How long to yield frames for, measured from the first frame.
    """

    def __init__(self, source: Camera, seconds: float) -> None:
        self._source = source
        self._seconds = seconds

    def frames(self):  # type: ignore[no-untyped-def]
        """Yield frames until the time limit is reached."""
        started: float | None = None
        for frame in self._source.frames():
            if started is None:
                started = frame.timestamp
            if frame.timestamp - started > self._seconds:
                return
            yield frame

    def close(self) -> None:
        """Close the wrapped source."""
        self._source.close()


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    raise SystemExit(main())
