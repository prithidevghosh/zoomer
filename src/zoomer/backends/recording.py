"""An in-memory backend that records instead of acting.

This is what makes end-to-end testing possible without a desktop session: a test
can drive the whole pipeline and then assert on exactly what the document was
told to do, with no operating-system permissions and nothing to clean up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["RecordingBackend"]


@dataclass
class RecordingBackend:
    """Captures every command it is given.

    Attributes:
        zoom_calls: Each zoom request, in the order received.
        scroll_calls: Each scroll request, in the order received.
        closed: Whether :meth:`close` has been called.
    """

    zoom_calls: list[int] = field(default_factory=list)
    scroll_calls: list[int] = field(default_factory=list)
    closed: bool = False

    def zoom(self, steps: int) -> None:
        """Record a zoom request."""
        self.zoom_calls.append(steps)

    def scroll(self, clicks: int) -> None:
        """Record a scroll request."""
        self.scroll_calls.append(clicks)

    def close(self) -> None:
        """Record that the backend was shut down."""
        self.closed = True

    @property
    def net_zoom(self) -> int:
        """Total zoom steps applied; positive means the document ended larger."""
        return sum(self.zoom_calls)

    @property
    def net_scroll(self) -> int:
        """Total scroll clicks applied; positive means the document moved up."""
        return sum(self.scroll_calls)

    def clear(self) -> None:
        """Forget everything recorded so far, keeping the backend usable."""
        self.zoom_calls.clear()
        self.scroll_calls.clear()
