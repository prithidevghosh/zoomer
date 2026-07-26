"""Wire the pipeline together and run it.

The session is the only place the four stages meet: frames come from a source,
a tracker turns them into observations, the engine turns those into events, and
a backend applies them. Every stage is injected, so the same loop runs against a
real webcam or a scripted sequence of frames — which is what makes end-to-end
testing possible without hardware.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

from zoomer.backends.base import InputBackend, dispatch
from zoomer.calibration import CalibrationResult, Calibrator
from zoomer.gestures.engine import GestureEngine
from zoomer.gestures.features import extract_features
from zoomer.hud import HudView, NullHud
from zoomer.tracking.camera import FrameSource
from zoomer.tracking.hand_tracker import HandTracker
from zoomer.types import GestureMode, ScrollEvent, ZoomEvent

__all__ = ["Session", "SessionStats", "run_calibration"]


@dataclass(frozen=True, slots=True)
class SessionStats:
    """What happened during a run, for reporting when it ends.

    Args:
        frames: Frames pulled from the source.
        frames_with_hand: Frames in which a hand was located.
        zoom_steps: Net zoom steps applied; positive means the document ended
            larger than it started.
        scroll_clicks: Net scroll clicks applied; positive means the document
            moved up.
    """

    frames: int = 0
    frames_with_hand: int = 0
    zoom_steps: int = 0
    scroll_clicks: int = 0

    @property
    def detection_rate(self) -> float:
        """Fraction of frames containing a hand, in ``[0, 1]``.

        A low rate on a run the user believed was working points at lighting,
        framing, or camera choice rather than at gesture tuning.
        """
        return self.frames_with_hand / self.frames if self.frames else 0.0


class Session:
    """One run of the gesture pipeline.

    The session takes ownership of everything passed to it and closes each
    component when the run ends, so a single ``with`` block cleans up the
    camera, the model, the window, and any held modifier key.

    Args:
        source: Where frames come from.
        tracker: What finds a hand in each frame.
        engine: What turns hands into events.
        backend: Where events are applied.
        hud: Optional preview window. Defaults to showing nothing.
    """

    def __init__(
        self,
        source: FrameSource,
        tracker: HandTracker,
        engine: GestureEngine,
        backend: InputBackend,
        hud: HudView | None = None,
    ) -> None:
        self._source = source
        self._tracker = tracker
        self._engine = engine
        self._backend = backend
        self._hud: HudView = hud or NullHud()

    def run(self) -> SessionStats:
        """Process frames until the source is exhausted or the user quits.

        Returns:
            A summary of the run.
        """
        frames = hands = zoom_steps = scroll_clicks = 0

        for frame in self._source.frames():
            frames += 1

            observation = self._tracker.detect(frame)
            if observation is not None:
                hands += 1

            events = self._engine.update(observation)
            for event in events:
                if isinstance(event, ZoomEvent):
                    zoom_steps += event.steps
                elif isinstance(event, ScrollEvent):
                    scroll_clicks += event.clicks
            dispatch(self._backend, events)

            if not self._hud.render(frame, observation, self._engine.mode):
                break

        return SessionStats(
            frames=frames,
            frames_with_hand=hands,
            zoom_steps=zoom_steps,
            scroll_clicks=scroll_clicks,
        )

    def close(self) -> None:
        """Shut down every component, even if some of them fail.

        Each close is attempted independently so that one misbehaving component
        cannot leave the camera held open or, worse, a modifier key stuck down.
        """
        for component in (self._hud, self._backend, self._tracker, self._source):
            # Shutdown must never mask the reason the run ended.
            with contextlib.suppress(Exception):
                component.close()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def run_calibration(
    source: FrameSource,
    tracker: HandTracker,
    calibrator: Calibrator,
    hud: HudView | None = None,
) -> CalibrationResult:
    """Collect range-of-motion samples from a live feed.

    Frames without a hand are skipped rather than counted, so a user who pauses
    or moves out of shot simply contributes nothing for those frames instead of
    biasing the measured range.

    Args:
        source: Where frames come from.
        tracker: What finds a hand in each frame.
        calibrator: Collects the samples and summarises them.
        hud: Optional preview window, so the user can see they are in frame.

    Returns:
        The measured range of motion.

    Raises:
        CalibrationError: If too few samples were gathered to be meaningful.
    """
    view: HudView = hud or NullHud()

    for frame in source.frames():
        observation = tracker.detect(frame)
        if observation is not None:
            calibrator.observe(extract_features(observation))

        if not view.render(frame, observation, GestureMode.IDLE):
            break

    return calibrator.result()
