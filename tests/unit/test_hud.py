"""Unit tests for the heads-up display.

Pixel-perfect drawing is not worth asserting, but two things are: that the quit
keys are honoured, and that mirroring flips the overlay along with the image. A
mirror bug would leave the landmarks drawn on the wrong side of the screen,
which is exactly the diagnostic the display exists to provide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from tests.support import make_hand
from zoomer.hud import Hud, NullHud
from zoomer.tracking.camera import Frame
from zoomer.types import GestureMode


class FakeCanvas:
    """A stand-in image that reports a shape and can be copied or flipped."""

    def __init__(self, width: int = 640, height: int = 480, flipped: bool = False) -> None:
        self.shape = (height, width, 3)
        self.flipped = flipped

    def copy(self) -> FakeCanvas:
        return FakeCanvas(self.shape[1], self.shape[0], self.flipped)


@dataclass
class FakeCv2:
    """Records the drawing calls the display makes."""

    key: int = 255
    circles: list[tuple[int, int]] = field(default_factory=list)
    lines: list[tuple[tuple[int, int], tuple[int, int]]] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    shown: list[str] = field(default_factory=list)
    destroyed: list[str] = field(default_factory=list)
    flips: int = 0

    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def flip(self, image: FakeCanvas, _axis: int) -> FakeCanvas:
        self.flips += 1
        return FakeCanvas(image.shape[1], image.shape[0], flipped=True)

    def line(self, _canvas: Any, start: tuple[int, int], end: tuple[int, int], *_: Any) -> None:
        self.lines.append((start, end))

    def circle(self, _canvas: Any, centre: tuple[int, int], *_: Any) -> None:
        self.circles.append(centre)

    def putText(self, _canvas: Any, text: str, *_: Any) -> None:  # noqa: N802 - mirrors OpenCV
        self.texts.append(text)

    def imshow(self, title: str, _canvas: Any) -> None:
        self.shown.append(title)

    def waitKey(self, _delay: int) -> int:  # noqa: N802 - mirrors OpenCV
        return self.key

    def destroyWindow(self, title: str) -> None:  # noqa: N802 - mirrors OpenCV
        self.destroyed.append(title)


def make_hud(*, mirror: bool = True, key: int = 255) -> tuple[Hud, FakeCv2]:
    hud = Hud.__new__(Hud)
    cv2 = FakeCv2(key=key)
    hud._cv2 = cv2  # type: ignore[attr-defined]
    hud._title = "zoomer"  # type: ignore[attr-defined]
    hud._mirror = mirror  # type: ignore[attr-defined]
    hud._closed = False  # type: ignore[attr-defined]
    return hud, cv2


def frame() -> Frame:
    return Frame(image=FakeCanvas(width=640, height=480), timestamp=0.0)


class TestNullHud:
    def test_never_asks_to_quit(self) -> None:
        assert NullHud().render(frame(), None, GestureMode.IDLE) is True

    def test_closing_is_harmless(self) -> None:
        NullHud().close()


class TestQuitKeys:
    @pytest.mark.parametrize("key", [ord("q"), ord("Q"), 27])
    def test_q_and_escape_end_the_run(self, key: int) -> None:
        hud, _ = make_hud(key=key)
        assert hud.render(frame(), None, GestureMode.IDLE) is False

    @pytest.mark.parametrize("key", [255, ord("a"), ord(" ")])
    def test_any_other_key_lets_the_run_continue(self, key: int) -> None:
        hud, _ = make_hud(key=key)
        assert hud.render(frame(), None, GestureMode.IDLE) is True

    def test_a_closed_display_reports_that_the_run_should_stop(self) -> None:
        hud, _ = make_hud()
        hud.close()
        assert hud.render(frame(), None, GestureMode.IDLE) is False


class TestMirroring:
    def test_a_mirrored_display_flips_the_image(self) -> None:
        hud, cv2 = make_hud(mirror=True)
        hud.render(frame(), None, GestureMode.IDLE)
        assert cv2.flips == 1

    def test_an_unmirrored_display_leaves_the_image_alone(self) -> None:
        hud, cv2 = make_hud(mirror=False)
        hud.render(frame(), None, GestureMode.IDLE)
        assert cv2.flips == 0

    def test_landmarks_are_flipped_to_match_the_mirrored_image(self) -> None:
        # A hand on the left of the frame must be drawn on the right, or the
        # overlay would sit nowhere near the hand the user sees.
        hand = make_hand(pinch_gap=0.10, index_y=0.50, centre_x=0.20)
        hud, cv2 = make_hud(mirror=True)
        hud.render(frame(), hand, GestureMode.ZOOMING)
        assert all(x > 320 for x, _ in cv2.circles)

    def test_landmarks_keep_their_side_when_not_mirrored(self) -> None:
        hand = make_hand(pinch_gap=0.10, index_y=0.50, centre_x=0.20)
        hud, cv2 = make_hud(mirror=False)
        hud.render(frame(), hand, GestureMode.ZOOMING)
        assert all(x < 320 for x, _ in cv2.circles)

    def test_vertical_positions_are_never_flipped(self) -> None:
        hand = make_hand(pinch_gap=0.10, index_y=0.25)
        hud, cv2 = make_hud(mirror=True)
        hud.render(frame(), hand, GestureMode.SCROLLING)
        assert all(y == 120 for _, y in cv2.circles)


class TestDrawing:
    def test_marks_both_fingertips_that_drive_gestures(self) -> None:
        hud, cv2 = make_hud()
        hud.render(frame(), make_hand(pinch_gap=0.20, index_y=0.50), GestureMode.ZOOMING)
        # Each tip is drawn as a filled circle plus an outline.
        assert len(cv2.circles) == 4

    def test_draws_the_span_between_the_fingertips(self) -> None:
        hud, cv2 = make_hud()
        hud.render(frame(), make_hand(pinch_gap=0.20, index_y=0.50), GestureMode.ZOOMING)
        assert cv2.lines

    def test_names_the_active_mode(self) -> None:
        hud, cv2 = make_hud()
        hud.render(frame(), make_hand(pinch_gap=0.20, index_y=0.50), GestureMode.SCROLLING)
        assert any("scrolling" in text for text in cv2.texts)

    def test_shows_the_live_pinch_reading_for_tuning(self) -> None:
        hud, cv2 = make_hud()
        hud.render(frame(), make_hand(pinch_gap=0.20, index_y=0.50), GestureMode.ZOOMING)
        assert any(text.startswith("pinch:") for text in cv2.texts)

    def test_says_so_when_no_hand_is_found(self) -> None:
        # The single most useful diagnostic: it distinguishes bad tuning from
        # a blocked camera or poor lighting.
        hud, cv2 = make_hud()
        hud.render(frame(), None, GestureMode.IDLE)
        assert any("no hand" in text for text in cv2.texts)

    def test_always_explains_how_to_quit(self) -> None:
        hud, cv2 = make_hud()
        hud.render(frame(), None, GestureMode.IDLE)
        assert any("quit" in text for text in cv2.texts)

    def test_displays_the_frame_in_the_named_window(self) -> None:
        hud, cv2 = make_hud()
        hud.render(frame(), None, GestureMode.IDLE)
        assert cv2.shown == ["zoomer"]


class TestShutdown:
    def test_closing_destroys_the_window(self) -> None:
        hud, cv2 = make_hud()
        hud.close()
        assert cv2.destroyed == ["zoomer"]

    def test_closing_twice_destroys_the_window_only_once(self) -> None:
        hud, cv2 = make_hud()
        hud.close()
        hud.close()
        assert cv2.destroyed == ["zoomer"]
