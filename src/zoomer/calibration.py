"""Learn a user's comfortable range of motion and tune the gains to match.

Default gains assume an average hand at an average distance moving at an average
speed. Real users vary enough that a fixed default leaves some people flicking
their wrist across the whole frame for one zoom step and others sending the
document flying.

Calibration measures the range the user actually covers when asked to perform
each gesture fully, then picks gains so that one comfortable sweep produces a
useful amount of movement. It is a pure computation over feature samples, so it
is fully testable without a camera.

Robustness to glitches
----------------------
The measured range is taken from percentiles rather than the outright minimum
and maximum. Hand trackers occasionally emit a wildly wrong landmark, and a
single such sample would otherwise inflate the range and leave the gains far too
low for the rest of the session.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from zoomer.gestures.engine import EngineConfig
from zoomer.gestures.features import HandFeatures

__all__ = [
    "CalibrationError",
    "CalibrationResult",
    "Calibrator",
    "percentile",
]

MIN_SAMPLES = 15
"""Fewest samples that can produce a meaningful range — about half a second of
video. Below this the percentiles are dominated by whichever frames happened to
arrive."""

TRIM = 0.05
"""Fraction discarded from each end of the sample range as suspected glitches."""


class CalibrationError(Exception):
    """Raised when calibration cannot produce a usable result."""


def percentile(values: list[float], fraction: float) -> float:
    """Return the value at ``fraction`` through a sorted copy of ``values``.

    Uses linear interpolation between neighbouring samples, which keeps the
    result stable as samples are added rather than stepping between them.

    Args:
        values: Samples to summarise. Must not be empty.
        fraction: Position in ``[0, 1]``; ``0`` is the smallest sample and ``1``
            the largest.

    Returns:
        The interpolated sample at that position.

    Raises:
        ValueError: If ``values`` is empty or ``fraction`` lies outside
            ``[0, 1]``.
    """
    if not values:
        raise ValueError("cannot take a percentile of no samples")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """The range of motion a user covered during calibration.

    Args:
        pinch_range: Distance between the user's closed and widest pinch, in
            hand-widths.
        pointer_range: Vertical distance the index finger travelled, in
            hand-widths.
        samples: How many frames contributed.
    """

    pinch_range: float
    pointer_range: float
    samples: int

    def tune(
        self,
        config: EngineConfig,
        *,
        zoom_steps_per_sweep: float = 10.0,
        scroll_clicks_per_sweep: float = 12.0,
    ) -> EngineConfig:
        """Return ``config`` with its gains rescaled to this user's range.

        Args:
            config: The configuration to adjust. It is not modified in place.
            zoom_steps_per_sweep: Zoom steps one full open-and-close should
                produce.
            scroll_clicks_per_sweep: Wheel clicks one full vertical sweep
                should produce.

        Returns:
            A new configuration with ``zoom_gain`` and ``scroll_gain`` set so
            that a comfortable sweep delivers the requested movement. Gains for
            a range too small to measure are left untouched, since dividing by
            a near-zero range would produce an unusably twitchy setting.

        Raises:
            ValueError: If either target movement is not positive.
        """
        if zoom_steps_per_sweep <= 0 or scroll_clicks_per_sweep <= 0:
            raise ValueError("target movement per sweep must be positive")

        changes: dict[str, float] = {}
        if self.pinch_range > _MEASURABLE:
            changes["zoom_gain"] = zoom_steps_per_sweep / self.pinch_range
        if self.pointer_range > _MEASURABLE:
            changes["scroll_gain"] = scroll_clicks_per_sweep / self.pointer_range

        return replace(config, **changes)  # type: ignore[arg-type]


_MEASURABLE = 0.05
"""Ranges below this are indistinguishable from a hand that never moved."""


class Calibrator:
    """Accumulate feature samples and summarise the user's range of motion.

    Args:
        min_samples: Fewest samples required before a result can be produced.
        trim: Fraction discarded from each end of the range as suspected
            tracking glitches.

    Raises:
        ValueError: If ``min_samples`` is below one or ``trim`` is not in
            ``[0, 0.5)``.
    """

    def __init__(self, min_samples: int = MIN_SAMPLES, trim: float = TRIM) -> None:
        if min_samples < 1:
            raise ValueError(f"min_samples must be at least 1, got {min_samples}")
        if not 0.0 <= trim < 0.5:
            raise ValueError(f"trim must be in [0, 0.5), got {trim}")

        self._min_samples = min_samples
        self._trim = trim
        self._pinches: list[float] = []
        self._pointers: list[float] = []

    @property
    def sample_count(self) -> int:
        """How many samples have been recorded so far."""
        return len(self._pinches)

    @property
    def ready(self) -> bool:
        """Whether enough samples have been recorded to produce a result."""
        return self.sample_count >= self._min_samples

    def observe(self, features: HandFeatures) -> None:
        """Record one frame's signals.

        Args:
            features: Signals extracted from a tracked hand.
        """
        self._pinches.append(features.pinch)
        self._pointers.append(features.pointer)

    def result(self) -> CalibrationResult:
        """Summarise the recorded samples.

        Returns:
            The measured ranges.

        Raises:
            CalibrationError: If too few samples were recorded to be meaningful.
        """
        if not self.ready:
            raise CalibrationError(
                f"need at least {self._min_samples} samples to calibrate, "
                f"got {self.sample_count}. Keep your hand in view of the camera "
                "and repeat the gesture."
            )

        return CalibrationResult(
            pinch_range=self._trimmed_range(self._pinches),
            pointer_range=self._trimmed_range(self._pointers),
            samples=self.sample_count,
        )

    def reset(self) -> None:
        """Discard all recorded samples."""
        self._pinches.clear()
        self._pointers.clear()

    def _trimmed_range(self, values: list[float]) -> float:
        """Return the spread of ``values``, ignoring outliers at both ends."""
        low = percentile(values, self._trim)
        high = percentile(values, 1.0 - self._trim)
        return high - low
