"""An optional preview window showing what the tracker sees.

Gesture control fails silently by nature: when nothing happens, the user cannot
tell whether the camera is blocked, the hand is out of frame, the lighting is
too dim, or a threshold is simply set too high. The heads-up display answers
that question directly by drawing the tracked landmarks and naming the mode
currently in effect.

It is strictly a diagnostic. Gesture recognition reads landmarks, never pixels,
so closing this window changes nothing about how the tool behaves.
"""

from __future__ import annotations

from typing import Any, Protocol

from zoomer.tracking.camera import Frame
from zoomer.types import GestureMode, HandObservation, Point

__all__ = ["Hud", "HudView", "NullHud"]

_QUIT_KEYS = frozenset({ord("q"), ord("Q"), 27})  # 27 is Escape

# BGR, since that is the order OpenCV draws in.
_COLOUR_IDLE = (170, 170, 170)
_COLOUR_ZOOM = (80, 200, 255)
_COLOUR_SCROLL = (120, 230, 120)
_COLOUR_TEXT = (255, 255, 255)
_COLOUR_SHADOW = (0, 0, 0)

_MODE_COLOURS = {
    GestureMode.IDLE: _COLOUR_IDLE,
    GestureMode.ZOOMING: _COLOUR_ZOOM,
    GestureMode.SCROLLING: _COLOUR_SCROLL,
}


class HudView(Protocol):
    """Anything that can display the tracker's view."""

    def render(
        self, frame: Frame, observation: HandObservation | None, mode: GestureMode
    ) -> bool:
        """Draw one frame.

        Returns:
            ``False`` if the user asked to quit, ``True`` to keep running.
        """
        ...

    def close(self) -> None:
        """Tear down the window."""
        ...


class NullHud:
    """A display that shows nothing, for headless runs."""

    def render(
        self, frame: Frame, observation: HandObservation | None, mode: GestureMode
    ) -> bool:
        """Ignore the frame and never ask to quit."""
        return True

    def close(self) -> None:
        """Do nothing; no window was ever opened."""


class Hud:
    """An OpenCV window drawing the camera feed and the tracked hand.

    Args:
        title: Window title.
        mirror: Whether to flip the image horizontally. A mirrored preview is
            what users expect from a webcam — raising the right hand should
            raise the hand on the right of the screen — and landmark positions
            are flipped to match so the overlay stays aligned.
    """

    def __init__(self, title: str = "zoomer", mirror: bool = True) -> None:
        import cv2

        self._cv2 = cv2
        self._title = title
        self._mirror = mirror
        self._closed = False

    def render(
        self, frame: Frame, observation: HandObservation | None, mode: GestureMode
    ) -> bool:
        """Draw one frame and check for a quit request.

        Args:
            frame: The frame just captured.
            observation: The tracked hand, if one was found.
            mode: The gesture currently in effect.

        Returns:
            ``False`` if the user pressed Q or Escape, ``True`` otherwise.
        """
        if self._closed:
            return False

        cv2 = self._cv2
        canvas = cv2.flip(frame.image, 1) if self._mirror else frame.image.copy()

        if observation is not None:
            self._draw_hand(canvas, observation, mode)
        self._draw_status(canvas, observation, mode)

        cv2.imshow(self._title, canvas)
        return cv2.waitKey(1) & 0xFF not in _QUIT_KEYS

    def close(self) -> None:
        """Destroy the window. Safe to call more than once."""
        if not self._closed:
            self._closed = True
            self._cv2.destroyWindow(self._title)

    def _to_pixels(self, point: Point, width: int, height: int) -> tuple[int, int]:
        """Map a normalised landmark onto the canvas, honouring the mirror."""
        x = 1.0 - point.x if self._mirror else point.x
        return int(x * width), int(point.y * height)

    def _draw_hand(self, canvas: Any, observation: HandObservation, mode: GestureMode) -> None:
        """Mark the two fingertips that drive gestures and the span between them."""
        cv2 = self._cv2
        height, width = canvas.shape[:2]
        colour = _MODE_COLOURS[mode]

        thumb = self._to_pixels(observation.thumb_tip, width, height)
        index = self._to_pixels(observation.index_tip, width, height)

        cv2.line(canvas, thumb, index, colour, 2, cv2.LINE_AA)
        for tip in (thumb, index):
            cv2.circle(canvas, tip, 9, colour, -1, cv2.LINE_AA)
            cv2.circle(canvas, tip, 9, _COLOUR_SHADOW, 1, cv2.LINE_AA)

        # The wrist-to-knuckle span is what the pinch is measured against, so
        # showing it makes the scale normalisation visible when tuning.
        wrist = self._to_pixels(observation.wrist, width, height)
        knuckle = self._to_pixels(observation.index_mcp, width, height)
        cv2.line(canvas, wrist, knuckle, _COLOUR_IDLE, 1, cv2.LINE_AA)

    def _draw_status(
        self, canvas: Any, observation: HandObservation | None, mode: GestureMode
    ) -> None:
        """Write the current mode and pinch reading in the corner."""
        if observation is None:
            lines = ["no hand detected"]
        else:
            pinch = (
                observation.thumb_tip.distance_to(observation.index_tip, observation.aspect_ratio)
                / observation.hand_scale
            )
            lines = [f"mode: {mode.value}", f"pinch: {pinch:.2f}"]
        lines.append("q or esc to quit")

        for i, text in enumerate(lines):
            origin = (12, 28 + i * 24)
            self._draw_text(canvas, text, origin)

    def _draw_text(self, canvas: Any, text: str, origin: tuple[int, int]) -> None:
        """Draw text with a dark outline so it stays legible over any footage."""
        cv2 = self._cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, text, origin, font, 0.6, _COLOUR_SHADOW, 3, cv2.LINE_AA)
        cv2.putText(canvas, text, origin, font, 0.6, _COLOUR_TEXT, 1, cv2.LINE_AA)
