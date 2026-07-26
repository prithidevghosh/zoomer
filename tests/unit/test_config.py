"""Unit tests for TOML configuration loading and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from zoomer.config import (
    CameraConfig,
    Config,
    ConfigError,
    InputConfig,
    TrackingConfig,
    build_config,
    default_config_path,
    load_config,
)


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


class TestDefaults:
    def test_an_empty_mapping_yields_a_fully_populated_config(self) -> None:
        config = build_config({})
        assert config.camera.index == 0
        assert config.gestures.zoom_gain > 0
        assert config.input.backend == "desktop"

    def test_defaults_are_not_shared_between_instances(self) -> None:
        # Mutable defaults leaking between configs would let one run's
        # calibration silently affect another.
        assert Config().camera is not Config().camera

    def test_the_default_path_is_under_the_users_config_directory(self) -> None:
        path = default_config_path()
        assert path.name == "config.toml"
        assert path.parent.name == "zoomer"


class TestLoading:
    def test_a_missing_default_file_falls_back_to_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A first run with no setup at all must work.
        monkeypatch.setattr("zoomer.config.default_config_path", lambda: tmp_path / "absent.toml")
        assert load_config() == Config()

    def test_a_missing_explicit_file_is_an_error(self, tmp_path: Path) -> None:
        # The user named a file; silently ignoring it would hide their intent.
        with pytest.raises(ConfigError, match="configuration file not found"):
            load_config(tmp_path / "absent.toml")

    def test_reads_values_from_disk(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            """
            show_hud = false

            [camera]
            index = 2
            width = 1280
            height = 720

            [input]
            backend = "none"
            zoom_mode = "modifier_scroll"
            """,
        )
        config = load_config(path)
        assert config.show_hud is False
        assert config.camera.index == 2
        assert config.camera.width == 1280
        assert config.input.backend == "none"
        assert config.input.zoom_mode == "modifier_scroll"

    def test_omitted_sections_keep_their_defaults(self, tmp_path: Path) -> None:
        path = write(tmp_path, "[camera]\nindex = 3\n")
        config = load_config(path)
        assert config.camera.index == 3
        assert config.tracking == TrackingConfig()
        assert config.input == InputConfig()

    def test_reports_malformed_toml_with_the_file_name(self, tmp_path: Path) -> None:
        path = write(tmp_path, "[camera\nindex = 1\n")
        with pytest.raises(ConfigError, match="is not valid TOML"):
            load_config(path)


class TestNestedGestureSettings:
    def test_reads_engine_tuning(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            """
            [gestures]
            zoom_gain = 12.5
            max_steps_per_frame = 5
            """,
        )
        config = load_config(path)
        assert config.gestures.zoom_gain == 12.5
        assert config.gestures.max_steps_per_frame == 5

    def test_reads_arbitration_thresholds_from_the_nested_table(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            """
            [gestures]
            scroll_gain = 9.0

            [gestures.mode_lock]
            zoom_enter = 0.9
            zoom_exit = 0.3
            settle_seconds = 0.5
            """,
        )
        config = load_config(path)
        assert config.gestures.scroll_gain == 9.0
        assert config.gestures.mode_lock.zoom_enter == 0.9
        assert config.gestures.mode_lock.settle_seconds == 0.5

    def test_untouched_thresholds_keep_their_defaults(self, tmp_path: Path) -> None:
        path = write(tmp_path, "[gestures.mode_lock]\nzoom_enter = 0.9\n")
        config = load_config(path)
        assert config.gestures.mode_lock.scroll_enter > 0


class TestValidation:
    """Every setting's own validation must surface as a ConfigError."""

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            ("[camera]\nindex = -1\n", "camera.index must be non-negative"),
            ("[camera]\nwidth = 0\n", "camera.width must be positive"),
            ("[camera]\nheight = -720\n", "camera.height must be positive"),
            (
                "[tracking]\nmin_detection_confidence = 0.0\n",
                "min_detection_confidence must be in",
            ),
            ("[tracking]\nmin_tracking_confidence = 1.5\n", "min_tracking_confidence must be in"),
            ('[input]\nbackend = "carrier pigeon"\n', "input.backend must be one of"),
            ('[input]\nzoom_mode = "telepathy"\n', "input.zoom_mode must be one of"),
            ("[input]\nscroll_lines_per_click = 0\n", "scroll_lines_per_click must be at least 1"),
            ("[gestures]\nzoom_gain = 0.0\n", "zoom_gain must be positive"),
            ("[gestures]\nscroll_deadzone = -1.0\n", "scroll_deadzone must be non-negative"),
            ("[gestures]\nmax_steps_per_frame = 0\n", "max_steps_per_frame must be at least 1"),
            (
                "[gestures.mode_lock]\nzoom_enter = 0.1\nzoom_exit = 0.9\n",
                "must not exceed zoom_enter",
            ),
            (
                "[gestures.mode_lock]\nsettle_seconds = -1.0\n",
                "settle_seconds must be non-negative",
            ),
        ],
    )
    def test_rejects_an_out_of_range_value(self, tmp_path: Path, body: str, message: str) -> None:
        with pytest.raises(ConfigError, match=message):
            load_config(write(tmp_path, body))

    def test_the_error_names_the_file_it_came_from(self, tmp_path: Path) -> None:
        path = write(tmp_path, "[camera]\nindex = -1\n")
        with pytest.raises(ConfigError, match=str(path)):
            load_config(path)

    def test_the_error_names_the_section_it_came_from(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=r"\[gestures\.mode_lock\]"):
            load_config(write(tmp_path, "[gestures.mode_lock]\nsettle_seconds = -1.0\n"))

    def test_rejects_a_value_of_the_wrong_shape(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=r"\[camera\] must be a table"):
            load_config(write(tmp_path, 'camera = "front"\n'))

    def test_rejects_a_mode_lock_that_is_not_a_table(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=r"\[gestures\.mode_lock\] must be a table"):
            load_config(write(tmp_path, '[gestures]\nmode_lock = "strict"\n'))


class TestUnknownKeys:
    """A silently ignored typo is worse than a startup error."""

    def test_rejects_an_unknown_top_level_key(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="unknown setting 'show_hood'"):
            load_config(write(tmp_path, "show_hood = true\n"))

    def test_rejects_an_unknown_key_inside_a_section(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="unknown setting 'framerate'"):
            load_config(write(tmp_path, "[camera]\nframerate = 60\n"))

    def test_rejects_an_unknown_key_inside_a_nested_table(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="unknown setting 'zoom_entry'"):
            load_config(write(tmp_path, "[gestures.mode_lock]\nzoom_entry = 0.5\n"))

    def test_the_error_lists_the_settings_that_would_have_been_valid(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=r"Valid settings are:.*index"):
            load_config(write(tmp_path, "[camera]\nindx = 1\n"))

    def test_reports_several_typos_together(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="unknown settings 'heigth', 'widht'"):
            load_config(write(tmp_path, "[camera]\nwidht = 1\nheigth = 2\n"))

    def test_names_the_section_containing_the_typo(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=r"in \[input\]"):
            load_config(write(tmp_path, "[input]\nbackends = 'none'\n"))


class TestRoundTrip:
    def test_the_documented_example_config_loads(self, tmp_path: Path) -> None:
        # This is the exact sample documented in the README. Asserting it
        # equals the built-in defaults keeps the documentation honest: if a
        # default changes and the README does not, this fails.
        path = write(
            tmp_path,
            """
            show_hud = true

            [camera]
            index = 0
            width = 640
            height = 480
            mirror = true

            [tracking]
            min_detection_confidence = 0.5
            min_tracking_confidence = 0.5

            [input]
            backend = "desktop"
            zoom_mode = "keyboard"
            scroll_lines_per_click = 3

            [gestures]
            zoom_gain = 6.0
            scroll_gain = 8.0
            zoom_deadzone = 0.05
            scroll_deadzone = 0.08
            max_steps_per_frame = 3
            max_missing_frames = 5
            min_cutoff = 0.8
            beta = 0.01
            derivative_cutoff = 1.0

            [gestures.mode_lock]
            zoom_enter = 0.55
            zoom_exit = 0.20
            scroll_enter = 0.70
            scroll_exit = 0.25
            settle_seconds = 0.25
            """,
        )
        assert load_config(path) == Config()

    def test_every_camera_field_is_settable_from_toml(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            "[camera]\nindex = 1\nwidth = 800\nheight = 600\nmirror = false\n",
        )
        assert load_config(path).camera == CameraConfig(
            index=1, width=800, height=600, mirror=False
        )
