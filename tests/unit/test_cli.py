"""Unit tests for the command line.

Argument parsing, the precedence of flags over the config file, and — most
importantly — the promise that every foreseeable failure prints one readable
line instead of a traceback.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from zoomer import __version__
from zoomer.cli import build_parser, main
from zoomer.config import Config
from zoomer.tracking.camera import CameraError
from zoomer.tracking.hand_tracker import TrackerError


def parse(argv: list[str]) -> argparse.Namespace:
    return build_parser().parse_args(argv)


class TestParsing:
    def test_runs_by_default_with_no_arguments(self) -> None:
        assert parse([]).command is None

    def test_accepts_an_explicit_run_command(self) -> None:
        assert parse(["run"]).command == "run"

    def test_accepts_the_calibrate_command(self) -> None:
        args = parse(["calibrate", "--seconds", "12"])
        assert args.command == "calibrate"
        assert args.seconds == 12.0

    def test_calibration_has_a_sensible_default_duration(self) -> None:
        assert parse(["calibrate"]).seconds > 0

    def test_accepts_the_model_download_command(self) -> None:
        assert parse(["download-model"]).command == "download-model"

    def test_reads_the_camera_index(self) -> None:
        assert parse(["--camera", "2"]).camera == 2

    def test_reads_the_backend_choice(self) -> None:
        assert parse(["--backend", "none"]).backend == "none"

    def test_reads_the_zoom_mode(self) -> None:
        assert parse(["--zoom-mode", "modifier_scroll"]).zoom_mode == "modifier_scroll"

    def test_reads_the_config_path_as_a_path(self) -> None:
        assert parse(["-c", "/tmp/z.toml"]).config == Path("/tmp/z.toml")

    def test_camera_options_work_on_the_calibrate_command_too(self) -> None:
        args = parse(["calibrate", "--camera", "3", "--no-hud"])
        assert args.camera == 3
        assert args.no_hud is True

    def test_rejects_an_unknown_backend(self) -> None:
        with pytest.raises(SystemExit):
            parse(["--backend", "carrier-pigeon"])

    def test_rejects_an_unknown_command(self) -> None:
        with pytest.raises(SystemExit):
            parse(["levitate"])

    def test_reports_the_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exit_info:
            parse(["--version"])
        assert exit_info.value.code == 0
        assert __version__ in capsys.readouterr().out


class TestFlagPrecedence:
    """Flags must win over the config file, or they would be useless."""

    def test_the_camera_flag_overrides_the_file(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("[camera]\nindex = 5\n")
        captured = _capture_config(["-c", str(config), "--camera", "9"])
        assert captured.camera.index == 9

    def test_the_file_is_used_when_no_flag_is_given(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("[camera]\nindex = 5\n")
        assert _capture_config(["-c", str(config)]).camera.index == 5

    def test_the_backend_flag_overrides_the_file(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text('[input]\nbackend = "desktop"\n')
        captured = _capture_config(["-c", str(config), "--backend", "none"])
        assert captured.input.backend == "none"

    def test_no_hud_overrides_a_file_that_enables_it(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("show_hud = true\n")
        assert _capture_config(["-c", str(config), "--no-hud"]).show_hud is False

    def test_a_file_that_disables_the_hud_is_respected_without_the_flag(
        self, tmp_path: Path
    ) -> None:
        config = tmp_path / "config.toml"
        config.write_text("show_hud = false\n")
        assert _capture_config(["-c", str(config)]).show_hud is False

    def test_unspecified_settings_keep_their_defaults(self) -> None:
        captured = _capture_config(["--camera", "1"])
        assert captured.gestures == Config().gestures


_captured: list[Config] = []


def _capture_config(argv: list[str]) -> Config:
    """Run main() far enough to see the resolved config, then stop."""
    import zoomer.cli as cli

    _captured.clear()

    def spy(config: Config) -> int:
        _captured.append(config)
        return 0

    original = cli._run
    cli._run = spy  # type: ignore[assignment]
    try:
        main(argv)
    finally:
        cli._run = original  # type: ignore[assignment]

    assert _captured, "main() did not reach the run stage"
    return _captured[0]


class TestErrorReporting:
    """Users meet these failures routinely; none of them is a bug."""

    def test_a_missing_config_file_prints_one_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        status = main(["-c", str(tmp_path / "absent.toml")])
        error = capsys.readouterr().err
        assert status == 1
        assert error.startswith("zoomer: ")
        assert "Traceback" not in error

    def test_a_malformed_config_file_prints_one_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = tmp_path / "config.toml"
        config.write_text("[camera\n")
        assert main(["-c", str(config)]) == 1
        assert "not valid TOML" in capsys.readouterr().err

    def test_a_typo_in_the_config_names_the_offending_key(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = tmp_path / "config.toml"
        config.write_text("[camera]\nindx = 1\n")
        assert main(["-c", str(config)]) == 1
        assert "indx" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "error",
        [
            CameraError("could not open camera 0"),
            TrackerError("could not download the hand tracking model"),
            RuntimeError("could not open an input device"),
        ],
    )
    def test_hardware_failures_are_reported_without_a_traceback(
        self, error: Exception, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(_: Config) -> int:
            raise error

        monkeypatch.setattr("zoomer.cli._run", explode)
        assert main([]) == 1

        reported = capsys.readouterr().err
        assert reported.startswith("zoomer: ")
        assert str(error) in reported

    def test_ctrl_c_exits_with_the_conventional_status(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def interrupt(_: Config) -> int:
            raise KeyboardInterrupt

        monkeypatch.setattr("zoomer.cli._run", interrupt)
        assert main([]) == 130
        assert "stopped" in capsys.readouterr().err

    def test_a_non_positive_calibration_duration_is_rejected(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["calibrate", "--seconds", "0"]) == 1
        assert "--seconds must be positive" in capsys.readouterr().err


class TestModelDownloadCommand:
    def test_reports_where_the_model_was_cached(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cached = tmp_path / "hand_landmarker.task"
        cached.write_bytes(b"x" * 2_097_152)
        monkeypatch.setattr("zoomer.cli.ensure_model", lambda _: cached)

        assert main(["download-model"]) == 0
        output = capsys.readouterr().out
        assert str(cached) in output
        assert "2.0 MB" in output

    def test_a_failed_download_is_reported_cleanly(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(_: Any) -> None:
            raise TrackerError("could not download the hand tracking model: offline")

        monkeypatch.setattr("zoomer.cli.ensure_model", explode)
        assert main(["download-model"]) == 1
        assert "could not download" in capsys.readouterr().err
