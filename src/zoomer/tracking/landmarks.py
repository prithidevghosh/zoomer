"""Convert MediaPipe landmarks into domain observations.

This module is the seam between the vision library and the rest of the program.
It is deliberately kept free of any MediaPipe import: it accepts anything that
looks like an indexable sequence of objects with ``x`` and ``y`` attributes.
That makes the conversion — including its handling of malformed results —
testable without a camera or a model file.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from zoomer.types import HandObservation, Point

__all__ = [
    "INDEX_MCP",
    "INDEX_TIP",
    "REQUIRED_LANDMARK_COUNT",
    "THUMB_TIP",
    "WRIST",
    "Landmark",
    "to_observation",
]

# Indices into MediaPipe's 21-point hand model. Only these four are read: the
# brief limits gestures to the thumb and index finger, and the remaining two
# provide the rigid span used to normalise for distance from the camera.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8

REQUIRED_LANDMARK_COUNT = INDEX_TIP + 1
"""The shortest landmark list this module can read."""


class Landmark(Protocol):
    """The shape of a single MediaPipe normalised landmark."""

    x: float
    y: float


def to_observation(
    landmarks: Sequence[Landmark],
    *,
    timestamp: float,
    aspect_ratio: float,
) -> HandObservation | None:
    """Extract the four landmarks the gesture pipeline needs.

    Args:
        landmarks: Normalised landmarks for one detected hand.
        timestamp: Monotonic capture time of the frame, in seconds.
        aspect_ratio: Frame width divided by frame height.

    Returns:
        The observation, or ``None`` if the result is too short to contain the
        landmarks of interest. A partial detection is treated as no detection
        rather than raising, because a dropped frame is an ordinary event in a
        live video stream and should not interrupt the run loop.
    """
    if len(landmarks) < REQUIRED_LANDMARK_COUNT:
        return None

    return HandObservation(
        timestamp=timestamp,
        thumb_tip=_point(landmarks[THUMB_TIP]),
        index_tip=_point(landmarks[INDEX_TIP]),
        index_mcp=_point(landmarks[INDEX_MCP]),
        wrist=_point(landmarks[WRIST]),
        aspect_ratio=aspect_ratio,
    )


def _point(landmark: Landmark) -> Point:
    return Point(x=float(landmark.x), y=float(landmark.y))
