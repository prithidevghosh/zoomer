"""End-to-end tests: scripted hands in, document commands out.

Each test drives a complete session — the real feature extraction, the real
One-Euro filters, the real arbitration state machine, the real accumulator, and
the real event dispatch — using default production settings. Only the three
pieces of hardware are substituted: the camera, MediaPipe, and the window.

That boundary is deliberate. The tuning constants users actually run with are
under test here, so a change that makes the defaults unusable fails the suite
rather than shipping.

Every sequence is additionally run with reproducible landmark jitter, because a
pipeline that only works on mathematically perfect input would not survive
contact with a webcam.
"""

from __future__ import annotations

import pytest
from tests.support import (
    OPEN_PINCH,
    FakeHud,
    ScriptedFrameSource,
    ScriptedTracker,
    hold,
    make_hand,
    pinch_close,
    pinch_open,
    swipe_down,
    swipe_up,
    with_jitter,
)

from zoomer.app import Session, SessionStats
from zoomer.backends.recording import RecordingBackend
from zoomer.config import Config
from zoomer.gestures.engine import GestureEngine
from zoomer.types import HandObservation

Script = list[HandObservation | None]


def perform(script: Script, *, hud: FakeHud | None = None) -> tuple[RecordingBackend, SessionStats]:
    """Run a full session over a scripted gesture using default settings.

    Args:
        script: The hand poses to replay, one per frame.
        hud: Optional display, to test early quitting.

    Returns:
        The backend that recorded what the document was told to do, and the run
        statistics.
    """
    config = Config()
    backend = RecordingBackend()
    session = Session(
        source=ScriptedFrameSource(count=len(script)),
        tracker=ScriptedTracker(script=list(script)),
        engine=GestureEngine(config.gestures),
        backend=backend,
        hud=hud,
    )
    with session:
        stats = session.run()
    return backend, stats


def settle(script: Script, frames: int = 15) -> Script:
    """Hold the script's final pose still, long enough to release the mode lock.

    Continuing from where the previous gesture ended is essential: teleporting
    the hand back to some canonical resting pose would itself be a large,
    abrupt movement, and the pipeline would rightly interpret it as a gesture.
    """
    last = script[-1]
    assert last is not None, "cannot settle from a frame with no hand"
    return list(hold(last, frames=frames))


def then(*parts: Script) -> Script:
    """Concatenate gesture fragments into one continuous script."""
    combined: Script = []
    for part in parts:
        combined.extend(part)
    return combined


class TestZoomIn:
    """Brief item 4: opening the thumb and index finger zooms in."""

    def test_a_widening_pinch_zooms_the_document_in(self) -> None:
        backend, _ = perform(list(pinch_open(30)))
        assert backend.net_zoom > 0

    def test_it_zooms_by_a_useful_amount_rather_than_a_token_step(self) -> None:
        # One deliberate gesture should visibly change the document; a single
        # step would mean users must repeat the gesture endlessly.
        backend, _ = perform(list(pinch_open(30)))
        assert backend.net_zoom >= 3

    def test_it_does_not_scroll_while_zooming_in(self) -> None:
        # Opening the fingers moves the index tip; arbitration must suppress
        # the scroll that would otherwise imply.
        backend, _ = perform(list(pinch_open(30)))
        assert backend.scroll_calls == []

    def test_it_still_works_with_realistic_tracking_noise(self) -> None:
        backend, _ = perform(list(with_jitter(pinch_open(30))))
        assert backend.net_zoom > 0
        assert backend.scroll_calls == []


class TestZoomOut:
    """Brief item 3: closing the thumb and index finger zooms out."""

    def test_a_closing_pinch_zooms_the_document_out(self) -> None:
        backend, _ = perform(list(pinch_close(30)))
        assert backend.net_zoom < 0

    def test_it_zooms_out_by_a_useful_amount(self) -> None:
        backend, _ = perform(list(pinch_close(30)))
        assert backend.net_zoom <= -3

    def test_it_does_not_scroll_while_zooming_out(self) -> None:
        backend, _ = perform(list(pinch_close(30)))
        assert backend.scroll_calls == []

    def test_it_still_works_with_realistic_tracking_noise(self) -> None:
        backend, _ = perform(list(with_jitter(pinch_close(30))))
        assert backend.net_zoom < 0


class TestScrollUp:
    """Brief item 5: raising the index finger scrolls up."""

    def test_a_rising_hand_scrolls_the_document_up(self) -> None:
        backend, _ = perform(list(swipe_up(30)))
        assert backend.net_scroll > 0

    def test_it_scrolls_by_a_useful_amount(self) -> None:
        backend, _ = perform(list(swipe_up(30)))
        assert backend.net_scroll >= 3

    def test_it_does_not_zoom_while_scrolling_up(self) -> None:
        backend, _ = perform(list(swipe_up(30)))
        assert backend.zoom_calls == []

    def test_it_still_works_with_realistic_tracking_noise(self) -> None:
        backend, _ = perform(list(with_jitter(swipe_up(30))))
        assert backend.net_scroll > 0
        assert backend.zoom_calls == []


class TestScrollDown:
    """Brief item 5: lowering the index finger scrolls down."""

    def test_a_falling_hand_scrolls_the_document_down(self) -> None:
        backend, _ = perform(list(swipe_down(30)))
        assert backend.net_scroll < 0

    def test_it_scrolls_down_by_a_useful_amount(self) -> None:
        backend, _ = perform(list(swipe_down(30)))
        assert backend.net_scroll <= -3

    def test_it_does_not_zoom_while_scrolling_down(self) -> None:
        backend, _ = perform(list(swipe_down(30)))
        assert backend.zoom_calls == []

    def test_it_still_works_with_realistic_tracking_noise(self) -> None:
        backend, _ = perform(list(with_jitter(swipe_down(30))))
        assert backend.net_scroll < 0


class TestDoingNothing:
    """The property users notice first: it must sit still when they do."""

    def test_a_hand_resting_in_frame_moves_the_document_not_at_all(self) -> None:
        backend, stats = perform(list(hold(make_hand(pinch_gap=0.16, index_y=0.50), frames=150)))
        assert backend.zoom_calls == []
        assert backend.scroll_calls == []
        assert stats.frames_with_hand == 150

    def test_a_resting_hand_with_tracking_noise_still_moves_nothing(self) -> None:
        # This is the real-world version of the test above and the single most
        # important guarantee in the product: unfiltered jitter here would make
        # the document creep continuously.
        noisy = with_jitter(hold(make_hand(pinch_gap=0.16, index_y=0.50), frames=150))
        backend, _ = perform(list(noisy))
        assert backend.zoom_calls == []
        assert backend.scroll_calls == []

    def test_an_empty_frame_moves_the_document_not_at_all(self) -> None:
        backend, stats = perform([None] * 60)
        assert backend.zoom_calls == []
        assert backend.scroll_calls == []
        assert stats.frames_with_hand == 0

    def test_a_hand_drifting_far_slower_than_a_gesture_is_ignored(self) -> None:
        # An arm settling on a desk over several seconds must not be read as
        # a deliberate scroll.
        drift = [
            make_hand(pinch_gap=0.16, index_y=0.50 + 0.00015 * i, timestamp=i / 30)
            for i in range(150)
        ]
        backend, _ = perform(list(drift))
        assert backend.scroll_calls == []


class TestGestureSequences:
    def test_zooming_in_then_out_leaves_the_document_where_it_started(self) -> None:
        opening = pinch_open(30)
        script = then(opening, settle(opening), pinch_close(30))
        backend, _ = perform(script)
        assert backend.net_zoom == pytest.approx(0, abs=2)

    def test_scrolling_down_then_up_returns_to_the_starting_position(self) -> None:
        down = swipe_down(30)
        script = then(down, settle(down), swipe_up(30, start_y=down[-1].index_tip.y))
        backend, _ = perform(script)
        assert backend.net_scroll == pytest.approx(0, abs=2)

    def test_a_user_can_zoom_then_settle_then_scroll(self) -> None:
        # The switch the mode lock exists to permit.
        opening = pinch_open(30)
        script = then(
            opening,
            settle(opening),
            swipe_up(30, start_y=opening[-1].index_tip.y, pinch_gap=OPEN_PINCH),
        )
        backend, _ = perform(script)
        assert backend.net_zoom > 0
        assert backend.net_scroll > 0

    def test_a_user_can_scroll_then_settle_then_zoom(self) -> None:
        rising = swipe_up(30)
        script = then(rising, settle(rising), pinch_open(30, index_y=rising[-1].index_tip.y))
        backend, _ = perform(script)
        assert backend.net_scroll > 0
        assert backend.net_zoom > 0

    def test_repeating_a_gesture_keeps_moving_the_document_the_same_way(self) -> None:
        # Each stroke resumes from where the last one stopped, as a hand
        # actually moves. Restarting from a fixed height would teleport the
        # hand downward between strokes, which is itself a scroll-down gesture.
        script: Script = []
        height = 0.85
        for _ in range(3):
            stroke = swipe_up(20, start_y=height, travel=1.0)
            height = stroke[-1].index_tip.y
            script = then(script, stroke, settle(stroke))
        backend, _ = perform(script)
        assert backend.net_scroll > 0
        assert all(clicks > 0 for clicks in backend.scroll_calls)


class TestInterruptionsAndRecovery:
    def test_a_hand_leaving_mid_gesture_stops_the_document_immediately(self) -> None:
        script: Script = [*pinch_open(15), *([None] * 45)]
        backend, _ = perform(script)
        during = len(backend.zoom_calls)

        uninterrupted, _ = perform(list(pinch_open(15)))
        assert during == len(uninterrupted.zoom_calls)

    def test_a_hand_reappearing_elsewhere_does_not_jump_the_document(self) -> None:
        # Without a reset, the jump from the old position to the new one would
        # read as an enormous gesture and fling the document.
        script: Script = [
            *hold(make_hand(pinch_gap=0.04, index_y=0.80), frames=15),
            *([None] * 15),
            *hold(make_hand(pinch_gap=0.34, index_y=0.20), frames=15),
        ]
        backend, _ = perform(script)
        assert backend.zoom_calls == []
        assert backend.scroll_calls == []

    def test_intermittent_dropped_frames_do_not_break_a_gesture(self) -> None:
        # Real trackers lose the hand for a frame here and there.
        gesture = pinch_open(45)
        script: Script = [None if i % 7 == 6 else pose for i, pose in enumerate(gesture)]
        backend, _ = perform(script)
        assert backend.net_zoom > 0

    def test_the_session_stops_when_the_user_closes_the_window(self) -> None:
        hud = FakeHud(quit_after=10)
        _, stats = perform(list(pinch_open(200)), hud=hud)
        assert stats.frames == 10


class TestDistanceInvariance:
    """The same gesture must behave the same wherever the user sits."""

    def test_the_same_gesture_zooms_alike_near_and_far_from_the_camera(self) -> None:
        near, _ = perform(list(pinch_open(30, scale=0.32)))
        far, _ = perform(list(pinch_open(30, scale=0.16)))
        assert near.net_zoom == pytest.approx(far.net_zoom, abs=1)

    def test_the_same_gesture_scrolls_alike_near_and_far_from_the_camera(self) -> None:
        near, _ = perform(list(swipe_up(30, scale=0.32)))
        far, _ = perform(list(swipe_up(30, scale=0.16)))
        assert near.net_scroll == pytest.approx(far.net_scroll, abs=1)


class TestReporting:
    def test_the_run_summary_matches_what_the_backend_received(self) -> None:
        opening = pinch_open(30)
        script = then(opening, settle(opening), swipe_up(30, pinch_gap=OPEN_PINCH))
        backend, stats = perform(script)
        assert stats.zoom_steps == backend.net_zoom
        assert stats.scroll_clicks == backend.net_scroll

    def test_the_detection_rate_reflects_the_frames_that_had_a_hand(self) -> None:
        script: Script = [*pinch_open(30), *([None] * 10)]
        _, stats = perform(script)
        assert stats.detection_rate == pytest.approx(0.75)

    def test_every_command_sent_is_a_real_movement(self) -> None:
        opening = pinch_open(30)
        script = then(opening, settle(opening), swipe_up(30, pinch_gap=OPEN_PINCH))
        backend, _ = perform(script)
        assert all(steps != 0 for steps in backend.zoom_calls)
        assert all(clicks != 0 for clicks in backend.scroll_calls)
