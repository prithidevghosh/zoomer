"""End-to-end tests through the command-line entry point.

These start where a user starts — at ``zoomer`` with some arguments — and check
that the whole chain holds: arguments parsed, configuration resolved, components
built, frames processed, events delivered, and a summary printed. Only the
camera, MediaPipe, and the window are substituted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.support import (
    OPEN_PINCH,
    ScriptedFrameSource,
    ScriptedTracker,
    hold,
    make_hand,
    pinch_close,
    pinch_open,
    swipe_up,
)

from zoomer.backends.recording import RecordingBackend
from zoomer.cli import main
from zoomer.hud import NullHud
from zoomer.types import HandObservation

Script = list[HandObservation | None]


@pytest.fixture
def stage(monkeypatch: pytest.MonkeyPatch) -> Stage:
    """Replace the hardware in the CLI with scripted stand-ins."""
    return Stage(monkeypatch)


class Stage:
    """Holds the doubles the CLI will build, so tests can inspect them after."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch
        self.backend = RecordingBackend()
        self.source: ScriptedFrameSource | None = None
        self.tracker: ScriptedTracker | None = None

    def perform(self, script: Script, argv: list[str] | None = None) -> int:
        """Run the CLI over a scripted gesture and return its exit status."""
        self.source = ScriptedFrameSource(count=len(script))
        self.tracker = ScriptedTracker(script=list(script))

        self._monkeypatch.setattr("zoomer.cli._open_camera", lambda _: self.source)
        self._monkeypatch.setattr("zoomer.cli._open_tracker", lambda _: self.tracker)
        self._monkeypatch.setattr("zoomer.cli._make_hud", lambda _: NullHud())
        self._monkeypatch.setattr("zoomer.cli.create_backend", lambda *_, **__: self.backend)

        return main(argv if argv is not None else ["--no-hud"])


class TestRunningFromTheCommandLine:
    def test_a_zoom_gesture_reaches_the_document_and_exits_cleanly(self, stage: Stage) -> None:
        assert stage.perform(list(pinch_open(30))) == 0
        assert stage.backend.net_zoom > 0

    def test_a_zoom_out_gesture_reaches_the_document(self, stage: Stage) -> None:
        assert stage.perform(list(pinch_close(30))) == 0
        assert stage.backend.net_zoom < 0

    def test_a_scroll_gesture_reaches_the_document(self, stage: Stage) -> None:
        assert stage.perform(list(swipe_up(30))) == 0
        assert stage.backend.net_scroll > 0

    def test_a_mixed_session_delivers_both_gestures(self, stage: Stage) -> None:
        opening = pinch_open(30)
        script: Script = [
            *opening,
            *hold(opening[-1], frames=15),
            *swipe_up(30, pinch_gap=OPEN_PINCH),
        ]
        assert stage.perform(script) == 0
        assert stage.backend.net_zoom > 0
        assert stage.backend.net_scroll > 0

    def test_the_run_command_behaves_the_same_as_no_command(self, stage: Stage) -> None:
        assert stage.perform(list(pinch_open(30)), ["run", "--no-hud"]) == 0
        assert stage.backend.net_zoom > 0


class TestCleanup:
    def test_every_component_is_released_when_the_run_ends(self, stage: Stage) -> None:
        stage.perform(list(pinch_open(20)))
        assert stage.source is not None and stage.tracker is not None
        assert stage.source.closed
        assert stage.tracker.closed
        assert stage.backend.closed


class TestReporting:
    def test_the_summary_states_what_was_applied(
        self, stage: Stage, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stage.perform(list(pinch_open(30)))
        output = capsys.readouterr().out
        assert "processed 30 frames" in output
        assert "zoom steps" in output

    def test_poor_tracking_earns_an_explanatory_hint(
        self, stage: Stage, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The most common real complaint is "nothing happens", and the usual
        # cause is detection rather than tuning.
        script: Script = [*pinch_open(10), *([None] * 40)]
        stage.perform(script)
        assert "fewer than half the frames" in capsys.readouterr().err

    def test_good_tracking_earns_no_hint(
        self, stage: Stage, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stage.perform(list(pinch_open(30)))
        assert "fewer than half" not in capsys.readouterr().err


class TestFocusWarning:
    """The preview window takes focus, which silently breaks zoom but not scroll."""

    def test_running_with_the_preview_warns_about_focus(
        self, stage: Stage, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("zoomer.cli._make_hud", lambda _: NullHud())
        stage.perform(list(pinch_open(20)), ["run"])
        assert "click your PDF viewer" in capsys.readouterr().out

    def test_running_without_the_preview_does_not(
        self, stage: Stage, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stage.perform(list(pinch_open(20)), ["--no-hud"])
        assert "click your PDF viewer" not in capsys.readouterr().out


class TestBackendSelection:
    def test_the_none_backend_announces_that_nothing_will_be_sent(
        self, stage: Stage, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stage.perform(list(pinch_open(20)), ["--backend", "none", "--no-hud"])
        assert "nothing will be sent" in capsys.readouterr().out


class TestConfigurationTakesEffect:
    def test_a_configured_gain_changes_how_far_a_gesture_moves_the_document(
        self, stage: Stage, tmp_path: Path
    ) -> None:
        config = tmp_path / "config.toml"
        config.write_text("[gestures]\nzoom_gain = 60.0\n")
        stage.perform(list(pinch_open(30)), ["-c", str(config), "--no-hud"])
        strong = stage.backend.net_zoom

        stage.backend = RecordingBackend()
        config.write_text("[gestures]\nzoom_gain = 6.0\n")
        stage.perform(list(pinch_open(30)), ["-c", str(config), "--no-hud"])

        assert strong > stage.backend.net_zoom

    def test_a_configured_deadzone_can_suppress_a_gesture_entirely(
        self, stage: Stage, tmp_path: Path
    ) -> None:
        config = tmp_path / "config.toml"
        config.write_text("[gestures]\nzoom_deadzone = 50.0\n")
        stage.perform(list(pinch_open(30)), ["-c", str(config), "--no-hud"])
        assert stage.backend.zoom_calls == []


class TestCalibrationFromTheCommandLine:
    def test_calibration_prints_settings_the_user_can_paste_into_a_config(
        self, stage: Stage, capsys: pytest.CaptureFixture[str]
    ) -> None:
        opening = pinch_open(40)
        script: Script = [*opening, *swipe_up(40, pinch_gap=OPEN_PINCH)]

        assert stage.perform(script, ["calibrate", "--seconds", "60", "--no-hud"]) == 0

        output = capsys.readouterr().out
        assert "[gestures]" in output
        assert "zoom_gain =" in output
        assert "scroll_gain =" in output

    def test_calibration_reports_the_measured_ranges(
        self, stage: Stage, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stage.perform(list(pinch_open(40)), ["calibrate", "--seconds", "60", "--no-hud"])
        output = capsys.readouterr().out
        assert "pinch range" in output
        assert "pointer range" in output

    def test_calibration_stops_at_the_requested_duration(
        self, stage: Stage, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # 300 frames at 30 fps is ten seconds of footage; only two are wanted.
        stage.perform(list(pinch_open(300)), ["calibrate", "--seconds", "2", "--no-hud"])
        assert "over 61 frames" in capsys.readouterr().out

    def test_calibrating_with_no_hand_in_view_fails_helpfully(
        self, stage: Stage, capsys: pytest.CaptureFixture[str]
    ) -> None:
        status = stage.perform([None] * 60, ["calibrate", "--seconds", "60", "--no-hud"])
        assert status == 1
        assert "Keep your hand in view" in capsys.readouterr().err

    def test_calibration_releases_the_camera_even_when_it_fails(self, stage: Stage) -> None:
        stage.perform([None] * 60, ["calibrate", "--seconds", "60", "--no-hud"])
        assert stage.source is not None and stage.tracker is not None
        assert stage.source.closed
        assert stage.tracker.closed


class TestQuietFeeds:
    def test_a_still_hand_produces_a_run_that_changed_nothing(self, stage: Stage) -> None:
        script = list(hold(make_hand(pinch_gap=0.16, index_y=0.50), frames=120))
        assert stage.perform(script) == 0
        assert stage.backend.zoom_calls == []
        assert stage.backend.scroll_calls == []

    def test_a_camera_that_delivers_nothing_exits_cleanly(
        self, stage: Stage, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert stage.perform([]) == 0
        assert "processed 0 frames" in capsys.readouterr().out
