"""Unit tests for landmark-to-signal feature extraction.

The invariants here are what let a single set of thresholds work for every user
and every seating distance.
"""

from __future__ import annotations

import pytest

from zoomer.gestures.features import extract_features
from zoomer.types import HandObservation, Point


def make_hand(
    *,
    pinch_gap: float = 0.20,
    centre_x: float = 0.50,
    index_y: float = 0.50,
    scale: float = 0.20,
    aspect_ratio: float = 1.0,
    timestamp: float = 0.0,
) -> HandObservation:
    """Build an observation with a chosen fingertip gap and apparent hand size.

    The thumb and index tips straddle ``centre_x`` at ``index_y``; the wrist sits
    ``scale`` below the index knuckle so ``hand_scale`` is exactly ``scale``.
    """
    return HandObservation(
        timestamp=timestamp,
        thumb_tip=Point(centre_x - pinch_gap / 2, index_y),
        index_tip=Point(centre_x + pinch_gap / 2, index_y),
        index_mcp=Point(centre_x, index_y + 0.10),
        wrist=Point(centre_x, index_y + 0.10 + scale),
        aspect_ratio=aspect_ratio,
    )


class TestPinch:
    def test_is_the_fingertip_gap_measured_in_hand_widths(self) -> None:
        features = extract_features(make_hand(pinch_gap=0.20, scale=0.20))
        assert features.pinch == pytest.approx(1.0)

    def test_grows_as_the_fingers_open(self) -> None:
        closed = extract_features(make_hand(pinch_gap=0.02)).pinch
        neutral = extract_features(make_hand(pinch_gap=0.20)).pinch
        wide = extract_features(make_hand(pinch_gap=0.40)).pinch
        assert closed < neutral < wide

    def test_is_never_negative_even_when_the_tips_coincide(self) -> None:
        assert extract_features(make_hand(pinch_gap=0.0)).pinch == pytest.approx(0.0)

    def test_is_unchanged_when_the_hand_moves_around_the_frame(self) -> None:
        # Panning the hand must not be mistaken for a zoom.
        left = extract_features(make_hand(centre_x=0.20, index_y=0.30)).pinch
        right = extract_features(make_hand(centre_x=0.80, index_y=0.70)).pinch
        assert left == pytest.approx(right)

    def test_is_unchanged_when_the_hand_moves_closer_to_the_camera(self) -> None:
        # The headline invariant: an identical physical gesture at half the
        # distance doubles every pixel measurement, and must still read the same.
        far = extract_features(make_hand(pinch_gap=0.20, scale=0.20)).pinch
        near = extract_features(make_hand(pinch_gap=0.40, scale=0.40)).pinch
        assert far == pytest.approx(near)

    def test_the_same_physical_hand_reads_the_same_on_a_widescreen_frame(self) -> None:
        # A wider sensor packs the same physical width into a smaller share of
        # the normalised x range, so the identical hand arrives with its
        # horizontal gap divided by the aspect ratio. Correcting x back into y
        # units must cancel that out exactly.
        aspect = 16 / 9
        square = extract_features(make_hand(pinch_gap=0.20, aspect_ratio=1.0)).pinch
        widescreen = extract_features(make_hand(pinch_gap=0.20 / aspect, aspect_ratio=aspect)).pinch
        assert square == pytest.approx(widescreen)

    def test_a_pinch_reads_the_same_whether_held_flat_or_upright(self) -> None:
        # Without aspect correction, rotating the hand 90 degrees on a 16:9
        # camera would change the measured gap by ~78% and silently zoom.
        aspect = 16 / 9
        gap = 0.20

        horizontal = HandObservation(
            timestamp=0.0,
            thumb_tip=Point(0.50 - gap / 2 / aspect, 0.50),
            index_tip=Point(0.50 + gap / 2 / aspect, 0.50),
            index_mcp=Point(0.50, 0.60),
            wrist=Point(0.50, 0.80),
            aspect_ratio=aspect,
        )
        upright = HandObservation(
            timestamp=0.0,
            thumb_tip=Point(0.50, 0.50 - gap / 2),
            index_tip=Point(0.50, 0.50 + gap / 2),
            index_mcp=Point(0.50, 0.60),
            wrist=Point(0.50, 0.80),
            aspect_ratio=aspect,
        )
        assert extract_features(horizontal).pinch == pytest.approx(extract_features(upright).pinch)


class TestPointer:
    def test_increases_when_the_index_finger_rises(self) -> None:
        # MediaPipe y grows downward, so a smaller y is a higher finger.
        low = extract_features(make_hand(index_y=0.80)).pointer
        high = extract_features(make_hand(index_y=0.20)).pointer
        assert high > low

    def test_reports_upward_travel_in_hand_widths(self) -> None:
        start = extract_features(make_hand(index_y=0.60, scale=0.20)).pointer
        end = extract_features(make_hand(index_y=0.40, scale=0.20)).pointer
        assert end - start == pytest.approx(0.20 / 0.20)

    def test_the_same_physical_travel_reads_the_same_at_any_distance(self) -> None:
        far = extract_features(make_hand(index_y=0.60, scale=0.20)).pointer
        far_moved = extract_features(make_hand(index_y=0.50, scale=0.20)).pointer

        # Twice as close: the finger covers twice the normalised distance.
        near = extract_features(make_hand(index_y=0.60, scale=0.40)).pointer
        near_moved = extract_features(make_hand(index_y=0.40, scale=0.40)).pointer

        assert (far_moved - far) == pytest.approx(near_moved - near)

    def test_is_unaffected_by_opening_and_closing_the_pinch(self) -> None:
        # Zooming must not be mistaken for scrolling: the index tip stays at the
        # same height while only the horizontal gap changes.
        closed = extract_features(make_hand(pinch_gap=0.02, index_y=0.5)).pointer
        wide = extract_features(make_hand(pinch_gap=0.40, index_y=0.5)).pointer
        assert closed == pytest.approx(wide)


class TestTimestamp:
    def test_is_carried_through_unchanged(self) -> None:
        assert extract_features(make_hand(timestamp=12.5)).timestamp == pytest.approx(12.5)
