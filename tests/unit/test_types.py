"""Unit tests for the dependency-free domain types."""

from __future__ import annotations

import math

import pytest

from zoomer.types import HandObservation, Point, ScrollEvent, ZoomEvent


class TestPointDistance:
    def test_measures_plain_euclidean_distance_on_a_square_frame(self) -> None:
        assert Point(0.0, 0.0).distance_to(Point(3.0, 4.0)) == pytest.approx(5.0)

    def test_is_symmetric(self) -> None:
        a, b = Point(0.1, 0.9), Point(0.7, 0.2)
        assert a.distance_to(b) == pytest.approx(b.distance_to(a))

    def test_stretches_horizontal_distance_by_the_aspect_ratio(self) -> None:
        # On a 16:9 frame one unit of x spans 16/9 as much as one unit of y.
        horizontal = Point(0.0, 0.5).distance_to(Point(0.5, 0.5), aspect_ratio=16 / 9)
        assert horizontal == pytest.approx(0.5 * 16 / 9)

    def test_leaves_vertical_distance_untouched_by_the_aspect_ratio(self) -> None:
        vertical = Point(0.5, 0.0).distance_to(Point(0.5, 0.5), aspect_ratio=16 / 9)
        assert vertical == pytest.approx(0.5)

    def test_aspect_correction_makes_a_physical_square_measure_square(self) -> None:
        # A gesture 0.2 wide in x and 0.2*(9/16) tall in y is physically square
        # on a 16:9 frame; both legs must come out equal after correction.
        aspect = 16 / 9
        leg_x = Point(0.0, 0.0).distance_to(Point(0.2, 0.0), aspect_ratio=aspect)
        leg_y = Point(0.0, 0.0).distance_to(Point(0.0, 0.2 * aspect), aspect_ratio=aspect)
        assert leg_x == pytest.approx(leg_y)


def _observation(**overrides: object) -> HandObservation:
    defaults: dict[str, object] = {
        "timestamp": 0.0,
        "thumb_tip": Point(0.40, 0.50),
        "index_tip": Point(0.60, 0.50),
        "index_mcp": Point(0.50, 0.60),
        "wrist": Point(0.50, 0.80),
        "aspect_ratio": 1.0,
    }
    return HandObservation(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestHandScale:
    def test_measures_the_wrist_to_knuckle_span(self) -> None:
        assert _observation().hand_scale == pytest.approx(0.20)

    def test_is_unaffected_by_the_pinch_itself(self) -> None:
        # The wrist and knuckle are rigid relative to each other, so squeezing
        # the fingertips together must not change the yardstick.
        open_hand = _observation(thumb_tip=Point(0.30, 0.50), index_tip=Point(0.70, 0.50))
        closed_hand = _observation(thumb_tip=Point(0.49, 0.50), index_tip=Point(0.51, 0.50))
        assert open_hand.hand_scale == pytest.approx(closed_hand.hand_scale)

    def test_grows_in_proportion_when_the_hand_moves_closer(self) -> None:
        near = _observation(index_mcp=Point(0.50, 0.40), wrist=Point(0.50, 0.80))
        far = _observation(index_mcp=Point(0.50, 0.60), wrist=Point(0.50, 0.80))
        assert near.hand_scale == pytest.approx(2 * far.hand_scale)

    def test_stays_positive_when_landmarks_collapse_onto_each_other(self) -> None:
        degenerate = _observation(index_mcp=Point(0.5, 0.5), wrist=Point(0.5, 0.5))
        assert degenerate.hand_scale > 0.0
        assert math.isfinite(1.0 / degenerate.hand_scale)


class TestEvents:
    @pytest.mark.parametrize("steps", [-3, -1, 1, 4])
    def test_zoom_event_accepts_any_non_zero_step_count(self, steps: int) -> None:
        assert ZoomEvent(steps).steps == steps

    @pytest.mark.parametrize("clicks", [-5, -1, 1, 2])
    def test_scroll_event_accepts_any_non_zero_click_count(self, clicks: int) -> None:
        assert ScrollEvent(clicks).clicks == clicks

    def test_a_zero_step_zoom_is_rejected_as_meaningless(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            ZoomEvent(0)

    def test_a_zero_click_scroll_is_rejected_as_meaningless(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            ScrollEvent(0)

    def test_events_are_comparable_by_value(self) -> None:
        assert ZoomEvent(2) == ZoomEvent(2)
        assert ZoomEvent(2) != ScrollEvent(2)

    def test_events_are_immutable(self) -> None:
        with pytest.raises((AttributeError, TypeError)):
            ZoomEvent(1).steps = 2  # type: ignore[misc]
