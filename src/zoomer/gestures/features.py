"""Reduce a tracked hand to the two signals the gestures are built from.

The product brief is deliberately narrow: only the thumb and index finger are
watched. That yields exactly two degrees of freedom, and this module names them.

``pinch``
    How far apart the thumb and index tips are. Widening zooms in, closing
    zooms out.

``pointer``
    How high the index tip sits. Raising it scrolls up, lowering it scrolls
    down.

Both are divided by :attr:`~zoomer.types.HandObservation.hand_scale`, which is
what makes the gesture feel the same whether the user's hand is 30 cm or 80 cm
from the lens. Without that division a hand held close to the camera would
produce a pinch several times larger than the identical gesture performed
further away, and any fixed threshold would be wrong for one of them.

``pointer`` is additionally sign-flipped, because image ``y`` grows downward
while scrolling up is the positive direction. That flip happens here and
nowhere else, so the rest of the pipeline can reason in ordinary
"up is positive" terms.
"""

from __future__ import annotations

from dataclasses import dataclass

from zoomer.types import HandObservation

__all__ = ["HandFeatures", "extract_features"]


@dataclass(frozen=True, slots=True)
class HandFeatures:
    """The scale-invariant signals derived from one :class:`HandObservation`.

    Args:
        timestamp: Monotonic capture time in seconds, carried through from the
            observation so downstream velocity maths has a clock.
        pinch: Thumb-to-index distance in hand-widths. Roughly ``0.1`` when the
            fingers touch and ``1.5``-plus when they are spread wide.
        pointer: Height of the index tip in hand-widths, increasing upward.
            Only differences between frames are meaningful; the absolute value
            depends on where the hand happens to be in frame.
    """

    timestamp: float
    pinch: float
    pointer: float


def extract_features(observation: HandObservation) -> HandFeatures:
    """Convert a tracked hand into scale-invariant gesture signals.

    Args:
        observation: A single tracked hand.

    Returns:
        The derived :class:`HandFeatures`.
    """
    scale = observation.hand_scale
    aspect = observation.aspect_ratio

    pinch = observation.thumb_tip.distance_to(observation.index_tip, aspect) / scale

    # Negated so that "up" is positive: MediaPipe's y axis points downward.
    pointer = -observation.index_tip.y / scale

    return HandFeatures(timestamp=observation.timestamp, pinch=pinch, pointer=pointer)
