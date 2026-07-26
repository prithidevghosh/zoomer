"""Unit tests for the session run loop."""

from __future__ import annotations

import pytest

from tests.support import (
    FakeHud,
    ScriptedFrameSource,
    ScriptedTracker,
    hold,
    make_hand,
    pinch_open,
    swipe_up,
)
from zoomer.app import Session, SessionStats, run_calibration
from zoomer.backends.recording import RecordingBackend
from zoomer.calibration import CalibrationError, Calibrator
from zoomer.gestures.engine import GestureEngine
from zoomer.types import GestureMode, HandObservation


def build(
    script: list[HandObservation | None],
    *,
    hud: FakeHud | None = None,
) -> tuple[Session, RecordingBackend, ScriptedFrameSource, ScriptedTracker, FakeHud]:
    source = ScriptedFrameSource(count=len(script))
    tracker = ScriptedTracker(script=script)
    backend = RecordingBackend()
    view = hud or FakeHud()
    session = Session(
        source=source,
        tracker=tracker,
        engine=GestureEngine(),
        backend=backend,
        hud=view,
    )
    return session, backend, source, tracker, view


class TestStats:
    def test_counts_every_frame_it_processed(self) -> None:
        session, _, _, _, _ = build(list(pinch_open(25)))
        assert session.run().frames == 25

    def test_counts_only_the_frames_containing_a_hand(self) -> None:
        script: list[HandObservation | None] = [*pinch_open(10), None, None, None]
        session, _, _, _, _ = build(script)
        stats = session.run()
        assert stats.frames == 13
        assert stats.frames_with_hand == 10

    def test_reports_the_detection_rate(self) -> None:
        script: list[HandObservation | None] = [*pinch_open(15), *([None] * 5)]
        session, _, _, _, _ = build(script)
        assert session.run().detection_rate == pytest.approx(0.75)

    def test_a_run_with_no_frames_reports_no_detection_rate(self) -> None:
        # Guards the division in detection_rate.
        assert SessionStats().detection_rate == 0.0

    def test_reports_the_net_movement_it_applied(self) -> None:
        session, backend, _, _, _ = build(list(pinch_open(30)))
        stats = session.run()
        assert stats.zoom_steps == backend.net_zoom
        assert stats.scroll_clicks == backend.net_scroll


class TestEventDelivery:
    def test_a_widening_pinch_reaches_the_backend_as_zoom(self) -> None:
        session, backend, _, _, _ = build(list(pinch_open(30)))
        session.run()
        assert backend.net_zoom > 0
        assert backend.scroll_calls == []

    def test_a_rising_hand_reaches_the_backend_as_scroll(self) -> None:
        session, backend, _, _, _ = build(list(swipe_up(30)))
        session.run()
        assert backend.net_scroll > 0
        assert backend.zoom_calls == []

    def test_a_still_hand_sends_nothing_at_all(self) -> None:
        script = hold(make_hand(pinch_gap=0.18, index_y=0.50), frames=90)
        session, backend, _, _, _ = build(list(script))
        session.run()
        assert backend.zoom_calls == backend.scroll_calls == []

    def test_a_feed_with_no_hand_sends_nothing(self) -> None:
        session, backend, _, _, _ = build([None] * 60)
        session.run()
        assert backend.zoom_calls == backend.scroll_calls == []


class TestHudIntegration:
    def test_every_frame_is_offered_to_the_display(self) -> None:
        hud = FakeHud()
        session, _, _, _, _ = build(list(pinch_open(12)), hud=hud)
        session.run()
        assert len(hud.rendered) == 12

    def test_the_display_is_told_the_current_mode(self) -> None:
        hud = FakeHud()
        session, _, _, _, _ = build(list(pinch_open(30)), hud=hud)
        session.run()
        assert GestureMode.ZOOMING in {mode for _, mode in hud.rendered}

    def test_the_display_is_told_when_no_hand_was_found(self) -> None:
        hud = FakeHud()
        session, _, _, _, _ = build([None] * 5, hud=hud)
        session.run()
        assert all(observation is None for observation, _ in hud.rendered)

    def test_a_quit_request_stops_the_run_early(self) -> None:
        hud = FakeHud(quit_after=5)
        session, _, source, _, _ = build(list(pinch_open(100)), hud=hud)
        assert session.run().frames == 5
        assert source.delivered == 5

    def test_a_session_runs_without_any_display(self) -> None:
        source = ScriptedFrameSource(count=20)
        session = Session(
            source=source,
            tracker=ScriptedTracker(script=list(pinch_open(20))),
            engine=GestureEngine(),
            backend=RecordingBackend(),
        )
        assert session.run().frames == 20


class TestShutdown:
    def test_closing_releases_every_component(self) -> None:
        session, backend, source, tracker, hud = build(list(pinch_open(5)))
        session.run()
        session.close()
        assert (source.closed, tracker.closed, backend.closed, hud.closed) == (
            True,
            True,
            True,
            True,
        )

    def test_the_context_manager_closes_on_the_way_out(self) -> None:
        session, backend, source, tracker, hud = build(list(pinch_open(5)))
        with session:
            session.run()
        assert backend.closed and source.closed and tracker.closed and hud.closed

    def test_components_are_closed_even_if_the_body_raised(self) -> None:
        session, backend, _, _, _ = build(list(pinch_open(5)))
        with pytest.raises(RuntimeError, match="boom"):
            with session:
                raise RuntimeError("boom")
        assert backend.closed

    def test_one_failing_component_does_not_strand_the_others(self) -> None:
        # A stuck modifier key or an open camera is a far worse outcome than a
        # noisy shutdown, so every close is attempted independently.
        session, backend, source, tracker, hud = build(list(pinch_open(5)))

        def explode() -> None:
            raise OSError("window server went away")

        hud.close = explode  # type: ignore[method-assign]
        session.close()
        assert backend.closed and source.closed and tracker.closed


class TestCalibrationRun:
    def test_collects_one_sample_per_frame_containing_a_hand(self) -> None:
        script: list[HandObservation | None] = [*pinch_open(20), None, None]
        calibrator = Calibrator(min_samples=5)
        run_calibration(
            ScriptedFrameSource(count=len(script)),
            ScriptedTracker(script=script),
            calibrator,
        )
        assert calibrator.sample_count == 20

    def test_measures_the_range_the_gestures_covered(self) -> None:
        script = [*pinch_open(30), *swipe_up(30, start_time=1.0)]
        result = run_calibration(
            ScriptedFrameSource(count=len(script)),
            ScriptedTracker(script=list(script)),
            Calibrator(min_samples=5),
        )
        assert result.pinch_range > 0
        assert result.pointer_range > 0
        assert result.samples == 60

    def test_a_quit_request_ends_calibration_early(self) -> None:
        hud = FakeHud(quit_after=20)
        calibrator = Calibrator(min_samples=5)
        run_calibration(
            ScriptedFrameSource(count=200),
            ScriptedTracker(script=list(pinch_open(200))),
            calibrator,
            hud,
        )
        assert calibrator.sample_count == 20

    def test_a_feed_with_no_hand_cannot_be_calibrated(self) -> None:
        with pytest.raises(CalibrationError, match="need at least"):
            run_calibration(
                ScriptedFrameSource(count=30),
                ScriptedTracker(script=[None] * 30),
                Calibrator(min_samples=15),
            )
