"""The contract every input backend implements.

Keeping this interface to two verbs is what allows the gesture engine to remain
completely unaware of how — or whether — its output reaches a real application.
Tests substitute a recording backend; the desktop uses a synthetic-input one.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from zoomer.types import GestureEvent, ScrollEvent, ZoomEvent

__all__ = ["InputBackend", "NoopBackend", "dispatch"]


@runtime_checkable
class InputBackend(Protocol):
    """Applies gesture events to whatever the user is looking at."""

    def zoom(self, steps: int) -> None:
        """Change the zoom level of the focused application.

        Args:
            steps: Positive to zoom in, negative to zoom out. Never zero.
        """
        ...

    def scroll(self, clicks: int) -> None:
        """Scroll the focused application vertically.

        Args:
            clicks: Positive to scroll up, negative to scroll down. Never zero.
        """
        ...

    def close(self) -> None:
        """Release any operating-system resources held by the backend."""
        ...


def dispatch(backend: InputBackend, events: list[GestureEvent]) -> None:
    """Apply a batch of events to a backend in order.

    Centralising the event-to-method mapping means every backend, real or fake,
    interprets the event types identically.

    Args:
        backend: The backend to drive.
        events: Events to apply, in the order the engine produced them.

    Raises:
        TypeError: If an event of an unrecognised type is supplied.
    """
    for event in events:
        match event:
            case ZoomEvent(steps=steps):
                backend.zoom(steps)
            case ScrollEvent(clicks=clicks):
                backend.scroll(clicks)
            case _:  # pragma: no cover - guards against an unhandled event type
                raise TypeError(f"unsupported gesture event: {event!r}")


class NoopBackend:
    """A backend that discards everything.

    Useful for trying out gesture recognition and the heads-up display without
    letting the tool touch any real application — the recommended way to tune
    thresholds before granting the tool input permissions.
    """

    def zoom(self, steps: int) -> None:
        """Discard a zoom request."""

    def scroll(self, clicks: int) -> None:
        """Discard a scroll request."""

    def close(self) -> None:
        """Do nothing; this backend holds no resources."""
