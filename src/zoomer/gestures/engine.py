"""Turn a stream of tracked hands into discrete zoom and scroll commands.

This is the top of the camera-free half of the system. It owns the whole
translation from "where the fingers are" to "what the document should do", and
it deliberately depends on nothing but the standard library and its sibling
modules, so the entire behaviour of the product can be tested without hardware.

Why discrete events
-------------------
Hand motion is continuous, but the things a PDF viewer understands are not:
zoom is a keystroke, scrolling is a wheel click. The engine therefore integrates
the continuous signal into an accumulator and emits one step each time the
accumulator passes a whole unit, carrying the remainder forward. Nothing is
rounded away, so a slow gesture spread over many frames produces exactly as much
movement as the same gesture performed quickly -- it just arrives more gradually.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from zoomer.gestures.features import HandFeatures, extract_features
from zoomer.gestures.filters import OneEuroFilter, apply_deadzone
from zoomer.gestures.state_machine import ModeLock, ModeLockConfig
from zoomer.types import GestureEvent, GestureMode, HandObservation, ScrollEvent, ZoomEvent

__all__ = ["EngineConfig", "GestureEngine"]


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Tuning for the gesture engine.

    Args:
        zoom_gain: Zoom steps produced per hand-width of fingertip separation.
            Raise it to make a small pinch travel further.
        scroll_gain: Wheel clicks produced per hand-width of vertical travel.
        zoom_deadzone: Pinch speed, in hand-widths per second, treated as noise.
        scroll_deadzone: Pointer speed, in hand-widths per second, treated as
            noise.
        max_steps_per_frame: Ceiling on the events emitted from a single frame.
            A tracking glitch can teleport a landmark across the frame; without
            a ceiling that single bad sample would fire a burst of commands at
            the document.
        min_cutoff: One-euro ``min_cutoff`` for both signals, in hertz.
        beta: One-euro ``beta`` for both signals.
        derivative_cutoff: One-euro ``derivative_cutoff`` for both signals.
        mode_lock: Thresholds used to arbitrate between the two gestures.

    Raises:
        ValueError: If any gain, deadzone, or step ceiling is out of range.
    """

    zoom_gain: float = 6.0
    scroll_gain: float = 8.0
    zoom_deadzone: float = 0.05
    scroll_deadzone: float = 0.08
    max_steps_per_frame: int = 3
    min_cutoff: float = 0.8
    beta: float = 0.01
    derivative_cutoff: float = 1.0
    mode_lock: ModeLockConfig = field(default_factory=ModeLockConfig)

    def __post_init__(self) -> None:
        for name in ("zoom_gain", "scroll_gain"):
            value = getattr(self, name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        for name in ("zoom_deadzone", "scroll_deadzone"):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if self.max_steps_per_frame < 1:
            raise ValueError(
                f"max_steps_per_frame must be at least 1, got {self.max_steps_per_frame}"
            )


class GestureEngine:
    """Translate tracked hands into zoom and scroll events.

    The engine is a pure state machine over time: feed it observations in
    chronological order and it returns the commands to apply. It performs no
    input/output of its own.

    Args:
        config: Tuning parameters. Defaults suit a hand roughly an arm's length
            from a laptop webcam.
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self._config = config or EngineConfig()
        self._mode_lock = ModeLock(self._config.mode_lock)

        self._pinch_filter = self._make_filter()
        self._pointer_filter = self._make_filter()

        self._previous: HandFeatures | None = None
        self._zoom_accumulator = 0.0
        self._scroll_accumulator = 0.0

    def _make_filter(self) -> OneEuroFilter:
        return OneEuroFilter(
            min_cutoff=self._config.min_cutoff,
            beta=self._config.beta,
            derivative_cutoff=self._config.derivative_cutoff,
        )

    @property
    def mode(self) -> GestureMode:
        """The gesture currently in effect, for display in the heads-up view."""
        return self._mode_lock.mode

    @property
    def config(self) -> EngineConfig:
        """The tuning this engine was built with."""
        return self._config

    def update(self, observation: HandObservation | None) -> list[GestureEvent]:
        """Advance the engine by one camera frame.

        Args:
            observation: The tracked hand, or ``None`` if no hand was found in
                this frame.

        Returns:
            The events to apply, in the order they should be applied. Usually
            empty: a typical gesture spans many frames and only some of them
            complete a whole step.
        """
        if observation is None:
            self.reset()
            return []

        current = self._condition(extract_features(observation))
        previous, self._previous = self._previous, current

        # The first frame after the hand appears has nothing to difference
        # against, so it establishes a baseline and produces no movement.
        if previous is None:
            return []

        dt = current.timestamp - previous.timestamp
        if dt <= 0.0:
            # A stalled or rewound clock yields no meaningful velocity. Skip the
            # frame rather than dividing by it.
            return []

        pinch_delta = current.pinch - previous.pinch
        pointer_delta = current.pointer - previous.pointer

        zoom_speed = apply_deadzone(pinch_delta / dt, self._config.zoom_deadzone)
        scroll_speed = apply_deadzone(pointer_delta / dt, self._config.scroll_deadzone)

        mode = self._mode_lock.update(
            zoom_speed=zoom_speed,
            scroll_speed=scroll_speed,
            timestamp=current.timestamp,
        )

        if mode is GestureMode.ZOOMING:
            # Integrating the deadzoned speed rather than the raw delta keeps
            # the accumulator consistent with the thresholds that gated it.
            return self._emit_zoom(zoom_speed * dt)
        if mode is GestureMode.SCROLLING:
            return self._emit_scroll(scroll_speed * dt)
        return []

    def reset(self) -> None:
        """Forget all history, as when the hand leaves the frame.

        Partial accumulators are discarded on purpose. Resuming a gesture
        minutes later with a half-step of credit left over from the previous
        one would move the document without the user having asked.
        """
        self._pinch_filter.reset()
        self._pointer_filter.reset()
        self._mode_lock.release()
        self._previous = None
        self._zoom_accumulator = 0.0
        self._scroll_accumulator = 0.0

    def _condition(self, features: HandFeatures) -> HandFeatures:
        """Smooth both raw signals, leaving the timestamp untouched."""
        return HandFeatures(
            timestamp=features.timestamp,
            pinch=self._pinch_filter.filter(features.pinch, features.timestamp),
            pointer=self._pointer_filter.filter(features.pointer, features.timestamp),
        )

    def _emit_zoom(self, travel: float) -> list[GestureEvent]:
        self._zoom_accumulator += travel * self._config.zoom_gain
        steps, self._zoom_accumulator = self._take_whole_steps(self._zoom_accumulator)
        return [ZoomEvent(steps)] if steps else []

    def _emit_scroll(self, travel: float) -> list[GestureEvent]:
        self._scroll_accumulator += travel * self._config.scroll_gain
        clicks, self._scroll_accumulator = self._take_whole_steps(self._scroll_accumulator)
        return [ScrollEvent(clicks)] if clicks else []

    def _take_whole_steps(self, accumulator: float) -> tuple[int, float]:
        """Split an accumulator into whole steps to emit and a remainder to keep.

        Truncating toward zero is what makes the engine lossless: the fraction
        that did not qualify as a step stays in the accumulator and contributes
        to the next one, so a slow gesture and a fast gesture covering the same
        distance produce the same total movement.

        Args:
            accumulator: Fractional steps banked so far.

        Returns:
            The whole steps to emit, and the remainder to carry forward. When
            the ceiling clamps the emission, the surplus is *dropped* rather
            than carried, because it came from a tracking glitch rather than
            from the user.
        """
        steps = math.trunc(accumulator)
        if steps == 0:
            return 0, accumulator

        ceiling = self._config.max_steps_per_frame
        if abs(steps) > ceiling:
            return int(math.copysign(ceiling, steps)), 0.0

        return steps, accumulator - steps
