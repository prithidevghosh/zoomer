"""User-editable configuration, loaded from TOML.

Gesture tuning is genuinely personal: hand size, camera placement, seating
distance, and how briskly someone moves all shift the ideal thresholds. Rather
than burying constants in the code, everything adjustable lives in one TOML file
that :command:`zoomer calibrate` can also write.

Unknown keys are rejected rather than ignored. A silently misspelled setting
would leave the user adjusting a value that has no effect and concluding the
tool is broken, which is a far worse experience than a startup error naming the
typo.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from zoomer.backends import BACKEND_NAMES, ZOOM_MODES, BackendName, ZoomMode
from zoomer.gestures.engine import EngineConfig
from zoomer.gestures.state_machine import ModeLockConfig

__all__ = [
    "CameraConfig",
    "Config",
    "ConfigError",
    "InputConfig",
    "TrackingConfig",
    "build_config",
    "default_config_path",
    "load_config",
]

T = TypeVar("T")


class ConfigError(Exception):
    """Raised when a configuration file is malformed or contains bad values."""


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Which camera to open and at what resolution.

    Args:
        index: Device index as understood by OpenCV. ``0`` is the default
            camera; try ``1`` and upward if an external camera is preferred.
        width: Requested capture width in pixels.
        height: Requested capture height in pixels. Cameras may substitute the
            nearest supported mode.
        mirror: Whether to flip the preview horizontally. A mirrored preview
            matches what users expect from a webcam; it has no effect on gesture
            recognition, which reads landmarks rather than pixels.

    Raises:
        ValueError: If the index is negative or a dimension is not positive.
    """

    index: int = 0
    width: int = 640
    height: int = 480
    mirror: bool = True

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"camera.index must be non-negative, got {self.index}")
        for name in ("width", "height"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"camera.{name} must be positive, got {value}")


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """How aggressively to look for a hand.

    Args:
        min_detection_confidence: Score a hand must reach before tracking
            begins. Raise it if objects in the background are mistaken for
            hands; lower it if a real hand is not picked up.
        min_tracking_confidence: Score required to keep following a hand that
            has already been found.
        model_path: Path to a MediaPipe hand-landmarker bundle. Leave unset to
            use the copy cached automatically on first run.

    Raises:
        ValueError: If either confidence lies outside ``(0, 1]``.
    """

    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_path: str | None = None

    def __post_init__(self) -> None:
        for name in ("min_detection_confidence", "min_tracking_confidence"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"tracking.{name} must be in (0, 1], got {value}")


@dataclass(frozen=True, slots=True)
class InputConfig:
    """Where recognised gestures are sent.

    Args:
        backend: ``"desktop"`` to control the focused application, or
            ``"none"`` to recognise gestures without acting on them.
        zoom_mode: ``"keyboard"`` for the platform zoom shortcut, or
            ``"modifier_scroll"`` for modifier-plus-wheel.
        scroll_lines_per_click: Wheel lines emitted per scroll click.

    Raises:
        ValueError: If a name is unrecognised or the scroll rate is not
            positive.
    """

    backend: BackendName = "desktop"
    zoom_mode: ZoomMode = "keyboard"
    scroll_lines_per_click: int = 3

    def __post_init__(self) -> None:
        if self.backend not in BACKEND_NAMES:
            raise ValueError(f"input.backend must be one of {BACKEND_NAMES}, got {self.backend!r}")
        if self.zoom_mode not in ZOOM_MODES:
            raise ValueError(f"input.zoom_mode must be one of {ZOOM_MODES}, got {self.zoom_mode!r}")
        if self.scroll_lines_per_click < 1:
            raise ValueError(
                "input.scroll_lines_per_click must be at least 1, "
                f"got {self.scroll_lines_per_click}"
            )


@dataclass(frozen=True, slots=True)
class Config:
    """Everything zoomer can be told to do differently.

    Args:
        camera: Capture device settings.
        tracking: Hand-detection settings.
        gestures: Gesture engine tuning, including arbitration thresholds.
        input: Where recognised gestures are delivered.
        show_hud: Whether to open the heads-up preview window.
    """

    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    gestures: EngineConfig = field(default_factory=EngineConfig)
    input: InputConfig = field(default_factory=InputConfig)
    show_hud: bool = True


def default_config_path() -> Path:
    """Return the conventional location of the user's configuration file.

    Returns:
        ``~/.config/zoomer/config.toml``. The file need not exist; when it is
        absent, built-in defaults are used.
    """
    return Path.home() / ".config" / "zoomer" / "config.toml"


def load_config(path: Path | None = None) -> Config:
    """Read configuration from disk, falling back to defaults.

    Args:
        path: File to read. When ``None``, :func:`default_config_path` is used
            and a missing file is not an error, so a first run works with no
            setup. An explicitly requested file that is missing *is* an error,
            since the user clearly expected it to be applied.

    Returns:
        The loaded configuration, with defaults filled in for anything absent.

    Raises:
        ConfigError: If the file cannot be parsed, contains an unknown key, or
            holds a value the corresponding setting rejects.
    """
    explicit = path is not None
    target = path or default_config_path()

    if not target.exists():
        if explicit:
            raise ConfigError(f"configuration file not found: {target}")
        return Config()

    try:
        raw = tomllib.loads(target.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{target} is not valid TOML: {error}") from error
    except OSError as error:
        raise ConfigError(f"could not read {target}: {error}") from error

    return build_config(raw, source=str(target))


def build_config(raw: dict[str, Any], *, source: str = "configuration") -> Config:
    """Turn a parsed TOML mapping into a validated :class:`Config`.

    Args:
        raw: The mapping as produced by :mod:`tomllib`.
        source: A human-readable name for the mapping's origin, used in errors.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If a key is unknown or a value is rejected.
    """
    _reject_unknown_keys(raw, {f.name for f in fields(Config)}, source, "")

    return _construct(
        Config,
        {
            "camera": _section(CameraConfig, raw, "camera", source),
            "tracking": _section(TrackingConfig, raw, "tracking", source),
            "gestures": _gestures_section(raw, source),
            "input": _section(InputConfig, raw, "input", source),
            **({"show_hud": raw["show_hud"]} if "show_hud" in raw else {}),
        },
        source,
        "",
    )


def _gestures_section(raw: dict[str, Any], source: str) -> EngineConfig:
    """Build the engine config, including its nested mode-lock table."""
    table = dict(raw.get("gestures", {}))
    if not isinstance(table, dict):
        raise ConfigError(f"{source}: [gestures] must be a table")

    nested = table.pop("mode_lock", {})
    if not isinstance(nested, dict):
        raise ConfigError(f"{source}: [gestures.mode_lock] must be a table")

    mode_lock = _construct(ModeLockConfig, nested, source, "gestures.mode_lock")
    return _construct(EngineConfig, {**table, "mode_lock": mode_lock}, source, "gestures")


def _section(cls: type[T], raw: dict[str, Any], name: str, source: str) -> T:
    """Build one top-level section from its TOML table."""
    table = raw.get(name, {})
    if not isinstance(table, dict):
        raise ConfigError(f"{source}: [{name}] must be a table")
    return _construct(cls, table, source, name)


def _construct(cls: type[T], values: dict[str, Any], source: str, prefix: str) -> T:
    """Instantiate a config dataclass, surfacing its validation as ConfigError.

    Args:
        cls: The dataclass to build.
        values: Candidate field values.
        source: Origin of the values, for error messages.
        prefix: Dotted path to this section, for error messages.

    Returns:
        The constructed instance.

    Raises:
        ConfigError: If a key is unknown or a value is rejected by the
            dataclass's own validation.
    """
    assert is_dataclass(cls)
    known = {f.name for f in fields(cls)}
    _reject_unknown_keys(values, known, source, prefix)

    try:
        return cls(**values)
    except (TypeError, ValueError) as error:
        location = f"[{prefix}] " if prefix else ""
        raise ConfigError(f"{source}: {location}{error}") from error


def _reject_unknown_keys(values: dict[str, Any], known: set[str], source: str, prefix: str) -> None:
    """Fail on any key the schema does not define, suggesting the closest match."""
    unknown = sorted(set(values) - known)
    if not unknown:
        return

    location = f"[{prefix}]" if prefix else "the top level"
    detail = ", ".join(repr(key) for key in unknown)
    valid = ", ".join(sorted(known))
    raise ConfigError(
        f"{source}: unknown setting{'s' if len(unknown) > 1 else ''} {detail} in {location}. "
        f"Valid settings are: {valid}"
    )
