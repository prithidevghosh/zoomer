"""Unit tests for the gesture engine.

These assert the five behaviours the product brief names, plus the robustness
properties that keep the engine from misbehaving on imperfect tracking data.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from zoomer.gestures.engine import EngineConfig, GestureEngine
from zoomer.gestures.state_machine import ModeLockConfig
from zoomer.types import GestureEvent, GestureMode, HandObservation, Point, ScrollEvent, ZoomEvent

# Filtering is disabled (a very high cutoff follows the input exactly) so these
# tests measure the engine's own arithmetic rather than the filter's response
# curve, which has its own dedicated tests.
UNFILTERED = EngineConfig(
    zoom_gain=10.0,
    scroll_gain=10.0,
    zoom_deadzone=0.0,
    scroll_deadzone=0.0,
    max_steps_per_frame=100,
    min_cutoff=1000.0,
    beta=0.0,
    mode_lock=ModeLockConfig(
        zoom_enter=0.10,
        zoom_exit=0.05,
        scroll_enter=0.10,
        scroll_exit=0.05,
        settle_seconds=0.2,
    ),
)


def hand(*, pinch_gap: float, index_y: float, timestamp: float) -> HandObservation:
    """Build an observation whose hand_scale is exactly 0.20."""
    return HandObservation(
        timestamp=timestamp,
        thumb_tip=Point(0.50 - pinch_gap / 2, index_y),
        index_tip=Point(0.50 + pinch_gap / 2, index_y),
        index_mcp=Point(0.50, index_y + 0.10),
        wrist=Point(0.50, index_y + 0.30),
    )


def total_zoom(events: list[GestureEvent]) -> int:
    return sum(e.steps for e in events if isinstance(e, ZoomEvent))


def total_scroll(events: list[GestureEvent]) -> int:
    return sum(e.clicks for e in events if isinstance(e, ScrollEvent))


def run(
    engine: GestureEngine, observations: list[HandObservation | None]
) -> list[GestureEvent]:
    events: list[GestureEvent] = []
    for observation in observations:
        events.extend(engine.update(observation))
    return events


def widening(frames: int = 20, *, start: float = 0.05, step: float = 0.02) -> list[HandObservation]:
    """Fingers steadily opening at a fixed height."""
    return [
        hand(pinch_gap=start + i * step, index_y=0.50, timestamp=i / 30) for i in range(frames)
    ]


def closing(frames: int = 20, *, start: float = 0.45, step: float = 0.02) -> list[HandObservation]:
    """Fingers steadily coming together at a fixed height."""
    return [
        hand(pinch_gap=start - i * step, index_y=0.50, timestamp=i / 30) for i in range(frames)
    ]


def rising(frames: int = 20, *, start: float = 0.80, step: float = 0.02) -> list[HandObservation]:
    """A steady hand travelling upward (image y decreasing)."""
    return [hand(pinch_gap=0.20, index_y=start - i * step, timestamp=i / 30) for i in range(frames)]


def falling(frames: int = 20, *, start: float = 0.20, step: float = 0.02) -> list[HandObservation]:
    """A steady hand travelling downward (image y increasing)."""
    return [hand(pinch_gap=0.20, index_y=start + i * step, timestamp=i / 30) for i in range(frames)]


class TestConfigValidation:
    @pytest.mark.parametrize("field", ["zoom_gain", "scroll_gain"])
    def test_rejects_a_non_positive_gain(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} must be positive"):
            EngineConfig(**{field: 0.0})  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", ["zoom_deadzone", "scroll_deadzone"])
    def test_rejects_a_negative_deadzone(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} must be non-negative"):
            EngineConfig(**{field: -0.1})  # type: ignore[arg-type]

    def test_rejects_a_step_ceiling_below_one(self) -> None:
        with pytest.raises(ValueError, match="max_steps_per_frame must be at least 1"):
            EngineConfig(max_steps_per_frame=0)

    def test_each_engine_gets_its_own_mode_lock_config(self) -> None:
        assert EngineConfig().mode_lock is not EngineConfig().mode_lock


class TestZoomDirection:
    """Brief items 3 and 4: closing zooms out, widening zooms in."""

    def test_widening_the_fingers_zooms_in(self) -> None:
        events = run(GestureEngine(UNFILTERED), list(widening()))
        assert total_zoom(events) > 0

    def test_closing_the_fingers_zooms_out(self) -> None:
        events = run(GestureEngine(UNFILTERED), list(closing()))
        assert total_zoom(events) < 0

    def test_zooming_in_then_back_out_returns_to_the_starting_level(self) -> None:
        engine = GestureEngine(UNFILTERED)
        out = total_zoom(run(engine, list(widening())))
        engine.reset()
        back = total_zoom(run(engine, list(closing(start=0.43))))
        assert out + back == pytest.approx(0, abs=1)

    def test_a_zoom_gesture_emits_no_scroll(self) -> None:
        events = run(GestureEngine(UNFILTERED), list(widening()))
        assert total_scroll(events) == 0


class TestScrollDirection:
    """Brief item 5: index up scrolls up, index down scrolls down."""

    def test_raising_the_index_finger_scrolls_up(self) -> None:
        events = run(GestureEngine(UNFILTERED), list(rising()))
        assert total_scroll(events) > 0

    def test_lowering_the_index_finger_scrolls_down(self) -> None:
        events = run(GestureEngine(UNFILTERED), list(falling()))
        assert total_scroll(events) < 0

    def test_a_scroll_gesture_emits_no_zoom(self) -> None:
        events = run(GestureEngine(UNFILTERED), list(rising()))
        assert total_zoom(events) == 0

    def test_scrolling_up_then_down_returns_to_the_starting_position(self) -> None:
        engine = GestureEngine(UNFILTERED)
        up = total_scroll(run(engine, list(rising())))
        engine.reset()
        down = total_scroll(run(engine, list(falling(start=0.42))))
        assert up + down == pytest.approx(0, abs=1)


class TestStillness:
    def test_a_motionless_hand_produces_nothing(self) -> None:
        frames = [hand(pinch_gap=0.20, index_y=0.50, timestamp=i / 30) for i in range(120)]
        assert run(GestureEngine(), frames) == []

    def test_a_gently_trembling_hand_produces_nothing(self) -> None:
        # Sub-deadzone wobble is what a real resting hand looks like; it must
        # never accumulate into movement.
        frames = [
            hand(
                pinch_gap=0.20 + 0.0015 * (-1) ** i,
                index_y=0.50 + 0.0015 * (-1) ** (i // 2),
                timestamp=i / 30,
            )
            for i in range(300)
        ]
        assert run(GestureEngine(), frames) == []

    def test_no_hand_in_frame_produces_nothing(self) -> None:
        assert run(GestureEngine(), [None] * 30) == []

    def test_the_first_frame_only_establishes_a_baseline(self) -> None:
        engine = GestureEngine(UNFILTERED)
        assert engine.update(hand(pinch_gap=0.05, index_y=0.5, timestamp=0.0)) == []


class TestAccumulation:
    def test_events_never_carry_a_zero_magnitude(self) -> None:
        events = run(GestureEngine(UNFILTERED), list(widening()) + list(rising()))
        assert events
        for event in events:
            magnitude = event.steps if isinstance(event, ZoomEvent) else event.clicks
            assert magnitude != 0

    def test_a_slow_gesture_moves_as_far_as_a_fast_one(self) -> None:
        # The same fingertip travel, at a third of the speed. Truncation must
        # bank the remainder rather than discarding it, or slow gestures would
        # quietly lose ground.
        fast = total_zoom(run(GestureEngine(UNFILTERED), list(widening(21, step=0.03))))
        slow = total_zoom(run(GestureEngine(UNFILTERED), list(widening(61, step=0.01))))
        assert slow == pytest.approx(fast, abs=1)

    def test_travel_translates_into_steps_at_the_configured_gain(self) -> None:
        # 0.40 of fingertip travel over a hand 0.20 wide is 2.0 hand-widths;
        # at a gain of 10 that is 20 steps.
        events = run(GestureEngine(UNFILTERED), list(widening(21, start=0.05, step=0.02)))
        assert total_zoom(events) == pytest.approx(20, abs=1)

    def test_gain_scales_the_response_proportionally(self) -> None:
        low = replace(UNFILTERED, zoom_gain=5.0)
        high = replace(UNFILTERED, zoom_gain=10.0)
        assert total_zoom(run(GestureEngine(high), list(widening()))) == pytest.approx(
            2 * total_zoom(run(GestureEngine(low), list(widening()))), abs=1
        )


class TestGlitchResistance:
    def test_a_teleporting_landmark_cannot_fire_a_burst_of_commands(self) -> None:
        config = replace(UNFILTERED, max_steps_per_frame=2)
        engine = GestureEngine(config)
        engine.update(hand(pinch_gap=0.05, index_y=0.50, timestamp=0.0))
        events = engine.update(hand(pinch_gap=0.95, index_y=0.50, timestamp=1 / 30))
        assert all(abs(e.steps) <= 2 for e in events if isinstance(e, ZoomEvent))

    def test_the_surplus_from_a_glitch_is_dropped_not_replayed_later(self) -> None:
        # Carrying a glitch's overflow forward would make the document keep
        # drifting for seconds after a single bad frame.
        config = replace(UNFILTERED, max_steps_per_frame=1)
        engine = GestureEngine(config)
        engine.update(hand(pinch_gap=0.05, index_y=0.50, timestamp=0.0))
        engine.update(hand(pinch_gap=0.95, index_y=0.50, timestamp=1 / 30))

        settled = [hand(pinch_gap=0.95, index_y=0.50, timestamp=(2 + i) / 30) for i in range(60)]
        assert run(engine, list(settled)) == []

    def test_a_repeated_timestamp_is_skipped_rather_than_dividing_by_zero(self) -> None:
        engine = GestureEngine(UNFILTERED)
        engine.update(hand(pinch_gap=0.05, index_y=0.50, timestamp=1.0))
        assert engine.update(hand(pinch_gap=0.40, index_y=0.50, timestamp=1.0)) == []

    def test_a_rewound_clock_is_skipped(self) -> None:
        engine = GestureEngine(UNFILTERED)
        engine.update(hand(pinch_gap=0.05, index_y=0.50, timestamp=5.0))
        assert engine.update(hand(pinch_gap=0.40, index_y=0.50, timestamp=4.0)) == []


class TestModeExposure:
    def test_starts_idle(self) -> None:
        assert GestureEngine().mode is GestureMode.IDLE

    def test_reports_zooming_while_the_fingers_open(self) -> None:
        engine = GestureEngine(UNFILTERED)
        run(engine, list(widening(6)))
        assert engine.mode is GestureMode.ZOOMING

    def test_reports_scrolling_while_the_hand_rises(self) -> None:
        engine = GestureEngine(UNFILTERED)
        run(engine, list(rising(6)))
        assert engine.mode is GestureMode.SCROLLING

    def test_returns_to_idle_when_the_hand_leaves(self) -> None:
        engine = GestureEngine(UNFILTERED)
        run(engine, list(widening(6)))
        engine.update(None)
        assert engine.mode is GestureMode.IDLE


class TestReset:
    def test_a_lost_hand_resets_the_engine_implicitly(self) -> None:
        engine = GestureEngine(UNFILTERED)
        run(engine, list(widening(6)))
        engine.update(None)
        # The frame after the hand returns is a baseline, not a jump.
        assert engine.update(hand(pinch_gap=0.90, index_y=0.10, timestamp=10.0)) == []

    def test_banked_fractions_are_discarded_rather_than_resumed(self) -> None:
        # Half a step of credit left over from a gesture minutes ago must not
        # move the document when the user comes back.
        engine = GestureEngine(UNFILTERED)
        run(engine, list(widening(3)))
        engine.reset()
        assert engine.update(hand(pinch_gap=0.20, index_y=0.50, timestamp=99.0)) == []
