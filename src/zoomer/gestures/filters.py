"""Signal conditioning for noisy landmark streams.

Raw MediaPipe landmarks jitter by a pixel or two every frame even when the hand
is perfectly still. Fed straight into a gesture engine that jitter reads as
continuous motion, so the document drifts on its own. Two complementary tools
fix that:

* :class:`OneEuroFilter` removes jitter while the hand is slow but gets out of
  the way the moment the hand moves quickly, so deliberate gestures stay crisp.
* :func:`apply_deadzone` clamps whatever residual noise survives to exactly
  zero, guaranteeing a still hand produces no events at all.

The One-Euro filter is Casiez, Roussel, and Vogel's design from CHI 2012,
"1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive
Systems". It is a low-pass filter whose cutoff frequency rises with the observed
speed of the signal, which is exactly the jitter-versus-lag trade-off a hand
tracker needs.
"""

from __future__ import annotations

import math

__all__ = ["OneEuroFilter", "apply_deadzone", "smoothing_alpha"]


def smoothing_alpha(cutoff_hz: float, dt: float) -> float:
    """Return the exponential smoothing factor for a given cutoff and timestep.

    Args:
        cutoff_hz: Cutoff frequency of the low-pass filter, in hertz. Must be
            strictly positive.
        dt: Time elapsed since the previous sample, in seconds. Must be
            strictly positive.

    Returns:
        A weight in ``(0, 1]`` to apply to the newest sample. Values near 1
        follow the input closely; values near 0 smooth heavily.

    Raises:
        ValueError: If ``cutoff_hz`` or ``dt`` is not strictly positive.
    """
    if cutoff_hz <= 0.0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class _ExponentialFilter:
    """A one-pole low-pass filter with a caller-supplied weight per sample."""

    def __init__(self) -> None:
        self._value: float | None = None

    @property
    def value(self) -> float | None:
        """The most recent filtered value, or ``None`` before the first sample."""
        return self._value

    def update(self, sample: float, alpha: float) -> float:
        """Blend ``sample`` into the running value and return the result."""
        if self._value is None:
            self._value = sample
        else:
            self._value = alpha * sample + (1.0 - alpha) * self._value
        return self._value

    def reset(self) -> None:
        """Forget all history so the next sample is taken verbatim."""
        self._value = None


class OneEuroFilter:
    """An adaptive low-pass filter that trades lag for jitter based on speed.

    The filter estimates how fast the signal is changing and widens its cutoff
    frequency in proportion. A hand held still is smoothed aggressively (no
    drift); a hand snapped open is barely smoothed at all (no lag).

    Args:
        min_cutoff: Cutoff frequency in hertz applied when the signal is
            stationary. Lower values smooth a resting hand more firmly but add
            lag to the start of a gesture.
        beta: How sharply the cutoff rises with speed. Higher values cut lag
            during fast motion at the cost of letting more jitter through.
        derivative_cutoff: Cutoff frequency in hertz for the internal speed
            estimate. Smoothing the derivative stops noise spikes from briefly
            disabling the filter.

    Raises:
        ValueError: If ``min_cutoff`` or ``derivative_cutoff`` is not strictly
            positive, or if ``beta`` is negative.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        derivative_cutoff: float = 1.0,
    ) -> None:
        if min_cutoff <= 0.0:
            raise ValueError(f"min_cutoff must be positive, got {min_cutoff}")
        if derivative_cutoff <= 0.0:
            raise ValueError(f"derivative_cutoff must be positive, got {derivative_cutoff}")
        if beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta}")

        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff

        self._value = _ExponentialFilter()
        self._derivative = _ExponentialFilter()
        self._last_timestamp: float | None = None

    def filter(self, sample: float, timestamp: float) -> float:
        """Filter one sample.

        The first sample, and any sample that does not advance the clock, is
        returned unchanged: with no elapsed time there is no meaningful cutoff
        to compute. Out-of-order timestamps are treated the same way rather than
        producing a negative timestep.

        Args:
            sample: The raw value to filter.
            timestamp: Monotonic time of the sample, in seconds.

        Returns:
            The filtered value.
        """
        previous = self._last_timestamp
        self._last_timestamp = timestamp

        if previous is None or timestamp <= previous:
            self._derivative.update(0.0, 1.0)
            return self._value.update(sample, 1.0)

        dt = timestamp - previous

        # Estimate speed from the *filtered* history so a single noisy sample
        # cannot masquerade as a fast gesture and unlock the filter.
        last_value = self._value.value
        raw_derivative = 0.0 if last_value is None else (sample - last_value) / dt
        speed = self._derivative.update(
            raw_derivative, smoothing_alpha(self.derivative_cutoff, dt)
        )

        cutoff = self.min_cutoff + self.beta * abs(speed)
        return self._value.update(sample, smoothing_alpha(cutoff, dt))

    def reset(self) -> None:
        """Discard all history, as when the hand leaves and re-enters the frame."""
        self._value.reset()
        self._derivative.reset()
        self._last_timestamp = None


def apply_deadzone(value: float, threshold: float) -> float:
    """Suppress small magnitudes and rescale the remainder continuously.

    A plain threshold (``0`` below, ``value`` above) makes the output jump
    discontinuously the instant the threshold is crossed, which feels like the
    document lurching. Subtracting the threshold instead means the output leaves
    zero smoothly, so a gesture ramps up from nothing.

    Args:
        value: The signal to gate.
        threshold: Magnitude below which the signal is treated as noise. Must be
            non-negative; zero passes the signal through untouched.

    Returns:
        Zero if ``abs(value) <= threshold``, otherwise ``value`` shrunk toward
        zero by ``threshold`` with its sign preserved.

    Raises:
        ValueError: If ``threshold`` is negative.
    """
    if threshold < 0.0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    magnitude = abs(value)
    if magnitude <= threshold:
        return 0.0
    return math.copysign(magnitude - threshold, value)
