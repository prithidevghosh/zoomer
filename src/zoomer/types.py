"""Domain types shared across the gesture pipeline.

This module is deliberately dependency-free: it imports nothing beyond the
standard library. Everything downstream of the camera speaks in these types, so
the gesture logic can be exercised in tests without a webcam, without MediaPipe,
and without a desktop session.

Coordinate conventions
----------------------
Landmark coordinates follow MediaPipe's image-space convention: ``x`` and ``y``
are normalised to ``[0, 1]`` against the frame width and height respectively,
with the origin at the **top-left** corner. Consequently ``y`` *decreases* as the
hand moves *upward*, which is the opposite of the scroll direction it produces.
:mod:`zoomer.gestures.features` is the single place where that sign is flipped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "GestureEvent",
    "GestureMode",
    "HandObservation",
    "Point",
    "ScrollEvent",
    "ZoomEvent",
]


@dataclass(frozen=True, slots=True)
class Point:
    """A landmark position in normalised image space.

    Args:
        x: Horizontal position in ``[0, 1]``, measured from the left edge.
        y: Vertical position in ``[0, 1]``, measured from the top edge.
    """

    x: float
    y: float

    def distance_to(self, other: Point, aspect_ratio: float = 1.0) -> float:
        """Return the Euclidean distance to ``other``.

        Normalised coordinates are scaled independently by the frame width and
        height, so on a non-square frame one unit of ``x`` covers more physical
        distance than one unit of ``y``. ``aspect_ratio`` (width / height)
        rescales ``x`` back into ``y`` units so the result is geometrically
        faithful rather than stretched.

        Args:
            other: The point to measure to.
            aspect_ratio: Frame width divided by frame height.

        Returns:
            The distance in height-normalised units.
        """
        dx = (self.x - other.x) * aspect_ratio
        dy = self.y - other.y
        return math.hypot(dx, dy)


@dataclass(frozen=True, slots=True)
class HandObservation:
    """A single tracked hand at one instant in time.

    Only the four landmarks the gesture logic actually needs are carried. The
    thumb and index tips drive the gestures themselves; the wrist and the index
    knuckle exist purely to measure how large the hand appears, which is what
    makes the gestures behave identically near to and far from the camera.

    Args:
        timestamp: Monotonic capture time in seconds.
        thumb_tip: MediaPipe landmark 4.
        index_tip: MediaPipe landmark 8.
        index_mcp: MediaPipe landmark 5, the index knuckle.
        wrist: MediaPipe landmark 0.
        aspect_ratio: Frame width divided by frame height.
    """

    timestamp: float
    thumb_tip: Point
    index_tip: Point
    index_mcp: Point
    wrist: Point
    aspect_ratio: float = 1.0

    @property
    def hand_scale(self) -> float:
        """Return the apparent size of the hand.

        The wrist-to-index-knuckle span is used as the yardstick because it is
        rigid: unlike a fingertip span it does not change when the user pinches,
        so dividing by it isolates the gesture from the hand's distance to the
        camera.

        Returns:
            The span in height-normalised units. Always strictly positive.
        """
        return max(self.wrist.distance_to(self.index_mcp, self.aspect_ratio), 1e-6)


class GestureMode(Enum):
    """Which gesture, if any, currently owns the user's hand.

    Exactly one mode is active at a time. See
    :class:`zoomer.gestures.state_machine.ModeLock` for the transition rules.
    """

    IDLE = "idle"
    ZOOMING = "zooming"
    SCROLLING = "scrolling"


@dataclass(frozen=True, slots=True)
class ZoomEvent:
    """A request to change the document's zoom level by discrete steps.

    Args:
        steps: Number of steps to apply. Positive zooms in (fingers widening),
            negative zooms out (fingers closing). Never zero.
    """

    steps: int

    def __post_init__(self) -> None:
        if self.steps == 0:
            raise ValueError("ZoomEvent.steps must be non-zero")


@dataclass(frozen=True, slots=True)
class ScrollEvent:
    """A request to scroll the document by discrete wheel clicks.

    Args:
        clicks: Number of clicks to apply. Positive scrolls up (index finger
            moving upward), negative scrolls down. Never zero.
    """

    clicks: int

    def __post_init__(self) -> None:
        if self.clicks == 0:
            raise ValueError("ScrollEvent.clicks must be non-zero")


GestureEvent = ZoomEvent | ScrollEvent
"""Anything the gesture engine can ask an input backend to perform."""
