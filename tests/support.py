"""Test doubles and gesture scripting shared by the unit and end-to-end suites.

The point of these is that they plug into the *real* pipeline. A scripted hand
built here flows through the same feature extraction, filtering, arbitration,
and accumulation that a webcam would drive, so an end-to-end test verifies the
production code path rather than a parallel one.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

from zoomer.tracking.camera import Frame
from zoomer.types import GestureMode, HandObservation, Point

__all__ = [
    "FRAME_RATE",
    "FakeHud",
    "ScriptedFrameSource",
    "ScriptedTracker",
    "hold",
    "make_hand",
    "pinch_close",
    "pinch_open",
    "swipe_down",
    "swipe_up",
    "with_jitter",
]

FRAME_RATE = 30.0
"""Frames per second the scripted sequences are generated at."""


class _Image:
    """The smallest stand-in for a camera image that Frame can measure."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.shape = (height, width, 3)


@dataclass
class ScriptedFrameSource:
    """Replays a fixed number of frames, one per scripted observation.

    Args:
        count: How many frames to yield.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """

    count: int
    width: int = 640
    height: int = 480
    closed: bool = False
    delivered: int = 0

    def frames(self) -> Iterator[Frame]:
        """Yield frames at a steady rate until the script is exhausted."""
        for i in range(self.count):
            if self.closed:
                return
            self.delivered += 1
            yield Frame(image=_Image(self.width, self.height), timestamp=i / FRAME_RATE)

    def close(self) -> None:
        """Mark the source closed."""
        self.closed = True


@dataclass
class ScriptedTracker:
    """Returns a pre-built observation per frame, ignoring the pixels.

    This substitutes for MediaPipe. Using it keeps tests deterministic and fast
    while leaving every stage downstream of detection genuinely under test.

    Args:
        script: Observations to return in order. ``None`` entries stand for
            frames in which no hand was found.
    """

    script: list[HandObservation | None] = field(default_factory=list)
    closed: bool = False
    calls: int = 0

    def detect(self, frame: Frame) -> HandObservation | None:
        """Return the next scripted observation, retimed to this frame."""
        if self.calls >= len(self.script):
            self.calls += 1
            return None

        observation = self.script[self.calls]
        self.calls += 1
        if observation is None:
            return None

        # Retime onto the frame's clock so the script and the source cannot
        # disagree about when things happened.
        return HandObservation(
            timestamp=frame.timestamp,
            thumb_tip=observation.thumb_tip,
            index_tip=observation.index_tip,
            index_mcp=observation.index_mcp,
            wrist=observation.wrist,
            aspect_ratio=frame.aspect_ratio,
        )

    def close(self) -> None:
        """Mark the tracker closed."""
        self.closed = True


@dataclass
class FakeHud:
    """Records what it was asked to draw, and can request a quit.

    Args:
        quit_after: Number of frames to display before asking the session to
            stop. ``None`` never asks.
    """

    quit_after: int | None = None
    rendered: list[tuple[HandObservation | None, GestureMode]] = field(default_factory=list)
    closed: bool = False

    def render(
        self, frame: Frame, observation: HandObservation | None, mode: GestureMode
    ) -> bool:
        """Record the frame and report whether the session should continue."""
        self.rendered.append((observation, mode))
        if self.quit_after is None:
            return True
        return len(self.rendered) < self.quit_after

    def close(self) -> None:
        """Mark the display closed."""
        self.closed = True


def make_hand(
    *,
    pinch_gap: float,
    index_y: float,
    timestamp: float = 0.0,
    centre_x: float = 0.50,
    scale: float = 0.20,
) -> HandObservation:
    """Build one hand pose.

    The thumb and index tips straddle ``centre_x`` at ``index_y``, and the wrist
    sits far enough below the knuckle that ``hand_scale`` equals ``scale``.

    Args:
        pinch_gap: Horizontal distance between the fingertips, in normalised
            units.
        index_y: Height of the fingertips, with 0 at the top of the frame.
        timestamp: Capture time in seconds.
        centre_x: Horizontal centre of the pinch.
        scale: Apparent hand size, standing in for distance from the camera.

    Returns:
        The posed observation.
    """
    return HandObservation(
        timestamp=timestamp,
        thumb_tip=Point(centre_x - pinch_gap / 2, index_y),
        index_tip=Point(centre_x + pinch_gap / 2, index_y),
        index_mcp=Point(centre_x, index_y + 0.10),
        wrist=Point(centre_x, index_y + 0.10 + scale),
    )


def _sequence(
    frames: int,
    *,
    gap_from: float,
    gap_to: float,
    y_from: float,
    y_to: float,
    start_time: float,
    scale: float,
) -> list[HandObservation]:
    """Interpolate a hand smoothly between two poses."""
    if frames < 1:
        raise ValueError(f"frames must be at least 1, got {frames}")
    span = max(frames - 1, 1)
    return [
        make_hand(
            pinch_gap=gap_from + (gap_to - gap_from) * i / span,
            index_y=y_from + (y_to - y_from) * i / span,
            timestamp=start_time + i / FRAME_RATE,
            scale=scale,
        )
        for i in range(frames)
    ]


def pinch_open(
    frames: int = 20,
    *,
    start_time: float = 0.0,
    index_y: float = 0.50,
    scale: float = 0.20,
) -> list[HandObservation]:
    """Thumb and index finger widening: the zoom-in gesture."""
    return _sequence(
        frames,
        gap_from=0.04,
        gap_to=0.34,
        y_from=index_y,
        y_to=index_y,
        start_time=start_time,
        scale=scale,
    )


def pinch_close(
    frames: int = 20,
    *,
    start_time: float = 0.0,
    index_y: float = 0.50,
    scale: float = 0.20,
) -> list[HandObservation]:
    """Thumb and index finger closing: the zoom-out gesture."""
    return _sequence(
        frames,
        gap_from=0.34,
        gap_to=0.04,
        y_from=index_y,
        y_to=index_y,
        start_time=start_time,
        scale=scale,
    )


def swipe_up(
    frames: int = 20,
    *,
    start_time: float = 0.0,
    pinch_gap: float = 0.16,
    scale: float = 0.20,
) -> list[HandObservation]:
    """Index finger rising: the scroll-up gesture."""
    return _sequence(
        frames,
        gap_from=pinch_gap,
        gap_to=pinch_gap,
        y_from=0.70,
        y_to=0.30,
        start_time=start_time,
        scale=scale,
    )


def swipe_down(
    frames: int = 20,
    *,
    start_time: float = 0.0,
    pinch_gap: float = 0.16,
    scale: float = 0.20,
) -> list[HandObservation]:
    """Index finger lowering: the scroll-down gesture."""
    return _sequence(
        frames,
        gap_from=pinch_gap,
        gap_to=pinch_gap,
        y_from=0.30,
        y_to=0.70,
        start_time=start_time,
        scale=scale,
    )


def hold(
    pose: HandObservation, frames: int = 20, *, start_time: float | None = None
) -> list[HandObservation]:
    """Repeat one pose, standing in for a hand resting still.

    Args:
        pose: The pose to hold.
        frames: How many frames to hold it for.
        start_time: When the hold begins. Defaults to continuing from ``pose``.

    Returns:
        The held sequence.
    """
    begin = pose.timestamp if start_time is None else start_time
    return [
        HandObservation(
            timestamp=begin + i / FRAME_RATE,
            thumb_tip=pose.thumb_tip,
            index_tip=pose.index_tip,
            index_mcp=pose.index_mcp,
            wrist=pose.wrist,
            aspect_ratio=pose.aspect_ratio,
        )
        for i in range(frames)
    ]


def with_jitter(
    observations: list[HandObservation], amplitude: float = 0.004, seed: int = 20260726
) -> list[HandObservation]:
    """Add reproducible tracking noise to every landmark.

    Real MediaPipe output is never perfectly steady. Running the scripted
    gestures through this makes the end-to-end tests answer a stronger question:
    not merely "does clean input work" but "does the filtering hold up against
    the noise a real camera produces".

    Args:
        observations: The clean sequence.
        amplitude: Peak displacement applied to each coordinate.
        seed: Fixed so failures are reproducible.

    Returns:
        A new, noisy sequence.
    """
    # A deterministic hash-free wobble: cheap, reproducible, and independent per
    # landmark and axis, without depending on any particular RNG implementation.
    def wobble(index: int, channel: int) -> float:
        phase = (index + 1) * (channel + 1) * (seed % 97 + 1)
        return amplitude * math.sin(phase * 2.399963229728653)

    def nudge(point: Point, index: int, base: int) -> Point:
        return Point(x=point.x + wobble(index, base), y=point.y + wobble(index, base + 1))

    return [
        HandObservation(
            timestamp=observation.timestamp,
            thumb_tip=nudge(observation.thumb_tip, i, 0),
            index_tip=nudge(observation.index_tip, i, 2),
            index_mcp=nudge(observation.index_mcp, i, 4),
            wrist=nudge(observation.wrist, i, 6),
            aspect_ratio=observation.aspect_ratio,
        )
        for i, observation in enumerate(observations)
    ]
