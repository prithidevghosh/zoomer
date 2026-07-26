"""Drive the focused application with synthetic keyboard and mouse input.

This backend is what fulfils the requirement that the tool work with *any* PDF
viewer. Rather than rendering documents itself, it emits the same events a
keyboard and mouse would, so whatever happens to be focused responds: Preview,
Adobe Acrobat, Chrome, Firefox, Edge, Okular, Evince, or a PDF embedded in a web
page.

Choosing the right synthetic events
-----------------------------------
Scrolling is easy and universal: every viewer on every platform responds to a
mouse wheel.

Zoom is not. There is no single event all viewers agree on, so two strategies
are offered:

``keyboard`` (the default)
    Send the platform's zoom-in/zoom-out shortcut, ``Cmd`` and ``-``/``=`` on
    macOS and ``Ctrl`` and ``-``/``=`` elsewhere. This is the most widely
    supported option and is the only one that works in native viewers such as
    Preview and Acrobat. Zoom arrives in the viewer's own preset increments.

``modifier_scroll``
    Hold the platform modifier and turn the wheel. Browsers zoom in finer
    increments this way, which feels smoother, but most native viewers ignore
    it entirely.

Operating-system permissions
----------------------------
Synthesising input is a privileged operation. On macOS the terminal or app
running zoomer must be granted Accessibility permission under System Settings >
Privacy & Security. On Linux under Wayland, synthetic input is restricted by
the compositor and an X11 session may be required.
"""

from __future__ import annotations

import contextlib
import sys
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from pynput import keyboard, mouse

__all__ = ["ZOOM_MODES", "DesktopBackend", "ZoomMode"]

ZoomMode = Literal["keyboard", "modifier_scroll"]
ZOOM_MODES: tuple[ZoomMode, ...] = ("keyboard", "modifier_scroll")

_ZOOM_IN_KEY = "="
"""Zoom in is bound to '=' rather than '+' because '+' requires Shift on most
layouts, and viewers accept the unshifted key."""

_ZOOM_OUT_KEY = "-"


class DesktopBackend:
    """Apply gestures to the focused application via synthetic input.

    Args:
        zoom_mode: How to express a zoom step. See the module docstring for the
            trade-off between ``"keyboard"`` and ``"modifier_scroll"``.
        scroll_lines_per_click: Wheel lines emitted per scroll click. Raise it
            for faster paging through long documents.

    Raises:
        ValueError: If ``zoom_mode`` is not recognised or
            ``scroll_lines_per_click`` is not positive.
        RuntimeError: If synthetic input is unavailable, for example because
            ``pynput`` cannot reach a display server or the process has not been
            granted permission to control the computer.
    """

    def __init__(
        self,
        zoom_mode: ZoomMode = "keyboard",
        scroll_lines_per_click: int = 3,
    ) -> None:
        if zoom_mode not in ZOOM_MODES:
            raise ValueError(f"zoom_mode must be one of {ZOOM_MODES}, got {zoom_mode!r}")
        if scroll_lines_per_click < 1:
            raise ValueError(
                f"scroll_lines_per_click must be at least 1, got {scroll_lines_per_click}"
            )

        self._zoom_mode = zoom_mode
        self._scroll_lines_per_click = scroll_lines_per_click

        # Imported lazily so that the pure gesture pipeline, and its tests, can
        # be used on a machine with no display server at all.
        try:
            from pynput import keyboard, mouse
        except ImportError as error:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "synthetic input requires pynput; install it with 'pip install zoomer'"
            ) from error

        try:
            self._keyboard: keyboard.Controller = keyboard.Controller()
            self._mouse: mouse.Controller = mouse.Controller()
        except Exception as error:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "could not open an input device. On macOS, grant Accessibility "
                "permission to your terminal under System Settings > Privacy & "
                "Security > Accessibility. On Linux, an X11 session may be required."
            ) from error

        self._modifier: Any = keyboard.Key.cmd if sys.platform == "darwin" else keyboard.Key.ctrl

    @property
    def zoom_mode(self) -> ZoomMode:
        """How this backend expresses a zoom step."""
        return self._zoom_mode

    def zoom(self, steps: int) -> None:
        """Zoom the focused application.

        Args:
            steps: Positive to zoom in, negative to zoom out.
        """
        if steps == 0:
            return
        if self._zoom_mode == "keyboard":
            key = _ZOOM_IN_KEY if steps > 0 else _ZOOM_OUT_KEY
            self._tap_with_modifier(key, abs(steps))
        else:
            self._scroll_with_modifier(steps)

    def scroll(self, clicks: int) -> None:
        """Scroll the focused application vertically.

        The wheel turns wherever the pointer happens to rest, which is how every
        desktop already behaves, so the user aims by leaving the cursor over the
        document.

        Args:
            clicks: Positive to scroll up, negative to scroll down.
        """
        if clicks == 0:
            return
        self._mouse.scroll(0, clicks * self._scroll_lines_per_click)

    def close(self) -> None:
        """Release the modifier key in case a gesture was interrupted mid-press.

        Without this, quitting during a zoom could strand the platform modifier
        in a held state and leave the desktop in a confusing mode.
        """
        # Best-effort: a display server that has already gone away must not
        # turn shutting down into a traceback.
        with contextlib.suppress(Exception):  # pragma: no cover - environment dependent
            self._keyboard.release(self._modifier)

    def _tap_with_modifier(self, key: str, count: int) -> None:
        """Hold the platform modifier and tap ``key`` ``count`` times."""
        with self._keyboard.pressed(self._modifier):
            for _ in range(count):
                self._keyboard.tap(key)

    def _scroll_with_modifier(self, steps: int) -> None:
        """Hold the platform modifier and turn the wheel, for browser zoom."""
        with self._keyboard.pressed(self._modifier):
            self._mouse.scroll(0, steps)
