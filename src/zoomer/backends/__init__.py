"""Where gesture events go: the real desktop, nowhere, or a test recorder."""

from __future__ import annotations

from typing import Literal

from zoomer.backends.base import InputBackend, NoopBackend, dispatch
from zoomer.backends.desktop import ZOOM_MODES, DesktopBackend, ZoomMode
from zoomer.backends.recording import RecordingBackend

__all__ = [
    "BACKEND_NAMES",
    "ZOOM_MODES",
    "BackendName",
    "DesktopBackend",
    "InputBackend",
    "NoopBackend",
    "RecordingBackend",
    "ZoomMode",
    "create_backend",
    "dispatch",
]

BackendName = Literal["desktop", "none"]
BACKEND_NAMES: tuple[BackendName, ...] = ("desktop", "none")


def create_backend(
    name: BackendName = "desktop",
    *,
    zoom_mode: ZoomMode = "keyboard",
    scroll_lines_per_click: int = 3,
) -> InputBackend:
    """Build the backend the user asked for.

    Args:
        name: ``"desktop"`` to drive the focused application with synthetic
            input, or ``"none"`` to recognise gestures without acting on them.
        zoom_mode: Passed to :class:`~zoomer.backends.desktop.DesktopBackend`.
        scroll_lines_per_click: Passed to
            :class:`~zoomer.backends.desktop.DesktopBackend`.

    Returns:
        A ready-to-use backend.

    Raises:
        ValueError: If ``name`` is not a recognised backend.
        RuntimeError: If the desktop backend cannot reach an input device.
    """
    match name:
        case "desktop":
            return DesktopBackend(
                zoom_mode=zoom_mode,
                scroll_lines_per_click=scroll_lines_per_click,
            )
        case "none":
            return NoopBackend()
        case _:
            raise ValueError(f"unknown backend {name!r}; expected one of {BACKEND_NAMES}")
