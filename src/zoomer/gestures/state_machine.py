"""Decide which single gesture owns the hand at any moment.

One hand producing two gestures is inherently ambiguous. Opening the fingers to
zoom drags the index tip upward, so a naive pipeline scrolls at the same time;
raising the hand to scroll changes the fingertip gap slightly, so it also zooms.
The result is a document that lurches diagonally in response to every gesture.

:class:`ModeLock` resolves this by allowing exactly one gesture at a time.
Whichever signal crosses its activation threshold first takes ownership, and the
other is suppressed until the hand settles. Three properties make it feel
natural rather than restrictive:

**Relative arbitration.** When both signals cross in the same frame, the winner
is the one that exceeded its own threshold by the larger *proportion*, not the
larger raw value. Pinch speed and pointer speed are different quantities and
their magnitudes are not comparable directly.

**Hysteresis.** Releasing a mode uses a lower threshold than entering it. A
single shared threshold would make a gesture hovering near it flicker between
active and idle several times a second.

**A settle delay.** A mode is not released the instant its signal dips; the
signal must stay quiet for a short dwell. This spans the natural pauses in the
middle of a gesture, and it means the deliberate act of stopping is what hands
control over to the other gesture.
"""

from __future__ import annotations

from dataclasses import dataclass

from zoomer.types import GestureMode

__all__ = ["ModeLock", "ModeLockConfig"]


@dataclass(frozen=True, slots=True)
class ModeLockConfig:
    """Thresholds governing gesture arbitration.

    Speeds are in hand-widths per second, the units produced by
    :mod:`zoomer.gestures.features`.

    Args:
        zoom_enter: Pinch speed at which zooming takes ownership of the hand.
        zoom_exit: Pinch speed below which zooming is considered to have
            stopped. Must not exceed ``zoom_enter``.
        scroll_enter: Pointer speed at which scrolling takes ownership.
        scroll_exit: Pointer speed below which scrolling is considered to have
            stopped. Must not exceed ``scroll_enter``.
        settle_seconds: How long a signal must stay below its exit threshold
            before the mode is released.

    Raises:
        ValueError: If any threshold is non-positive, an exit threshold exceeds
            its matching enter threshold, or ``settle_seconds`` is negative.
    """

    zoom_enter: float = 0.55
    zoom_exit: float = 0.20
    scroll_enter: float = 0.70
    scroll_exit: float = 0.25
    settle_seconds: float = 0.25

    def __post_init__(self) -> None:
        for name in ("zoom_enter", "zoom_exit", "scroll_enter", "scroll_exit"):
            value = getattr(self, name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.zoom_exit > self.zoom_enter:
            raise ValueError(
                f"zoom_exit ({self.zoom_exit}) must not exceed zoom_enter ({self.zoom_enter}); "
                "hysteresis requires releasing a mode to be easier than entering it"
            )
        if self.scroll_exit > self.scroll_enter:
            raise ValueError(
                f"scroll_exit ({self.scroll_exit}) must not exceed "
                f"scroll_enter ({self.scroll_enter}); hysteresis requires releasing a mode "
                "to be easier than entering it"
            )
        if self.settle_seconds < 0.0:
            raise ValueError(f"settle_seconds must be non-negative, got {self.settle_seconds}")


class ModeLock:
    """Track which gesture currently owns the hand.

    Args:
        config: Thresholds to arbitrate with. Defaults are tuned for a hand
            roughly an arm's length from a laptop webcam at 30 fps.
    """

    def __init__(self, config: ModeLockConfig | None = None) -> None:
        self._config = config or ModeLockConfig()
        self._mode = GestureMode.IDLE
        self._last_active: float = 0.0

    @property
    def mode(self) -> GestureMode:
        """The gesture that currently owns the hand."""
        return self._mode

    @property
    def config(self) -> ModeLockConfig:
        """The thresholds this lock arbitrates with."""
        return self._config

    def update(self, *, zoom_speed: float, scroll_speed: float, timestamp: float) -> GestureMode:
        """Advance the state machine by one frame.

        Args:
            zoom_speed: Rate of change of the pinch signal, in hand-widths per
                second. Only the magnitude matters here; direction is the
                caller's concern.
            scroll_speed: Rate of change of the pointer signal, in hand-widths
                per second.
            timestamp: Monotonic time of this frame, in seconds.

        Returns:
            The mode now in effect, which the caller should use to decide
            whether to emit zoom events, scroll events, or neither.
        """
        zoom_magnitude = abs(zoom_speed)
        scroll_magnitude = abs(scroll_speed)

        if self._mode is GestureMode.ZOOMING:
            self._sustain_or_release(zoom_magnitude, self._config.zoom_exit, timestamp)
        elif self._mode is GestureMode.SCROLLING:
            self._sustain_or_release(scroll_magnitude, self._config.scroll_exit, timestamp)

        # Falling through rather than returning early means a mode released on
        # this frame can be replaced on the same frame, so switching gestures
        # after a settle costs no extra latency.
        if self._mode is GestureMode.IDLE:
            self._acquire(zoom_magnitude, scroll_magnitude, timestamp)

        return self._mode

    def release(self) -> None:
        """Drop ownership immediately, as when the hand leaves the frame.

        No settle delay applies: a hand that is gone cannot be mid-gesture, and
        holding a stale mode would misattribute the first frames after it
        returns.
        """
        self._mode = GestureMode.IDLE
        self._last_active = 0.0

    def _sustain_or_release(self, magnitude: float, exit_threshold: float, timestamp: float) -> None:
        """Keep the active mode alive, or release it once the hand has settled."""
        if magnitude >= exit_threshold:
            self._last_active = timestamp
        elif timestamp - self._last_active >= self._config.settle_seconds:
            self._mode = GestureMode.IDLE

    def _acquire(self, zoom_magnitude: float, scroll_magnitude: float, timestamp: float) -> None:
        """Hand ownership to whichever signal most decisively cleared its bar."""
        zoom_excess = zoom_magnitude / self._config.zoom_enter
        scroll_excess = scroll_magnitude / self._config.scroll_enter

        if zoom_excess < 1.0 and scroll_excess < 1.0:
            return

        # An exact tie is vanishingly unlikely with real data, but resolving it
        # deterministically keeps the state machine reproducible in tests.
        self._mode = GestureMode.ZOOMING if zoom_excess >= scroll_excess else GestureMode.SCROLLING
        self._last_active = timestamp
