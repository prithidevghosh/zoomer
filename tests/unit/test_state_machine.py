"""Unit tests for gesture arbitration.

Every transition, the hysteresis band, and the settle delay are exercised
directly, since this state machine is what stands between the user and a
document that zooms and scrolls at the same time.
"""

from __future__ import annotations

import pytest

from zoomer.gestures.state_machine import ModeLock, ModeLockConfig
from zoomer.types import GestureMode

CONFIG = ModeLockConfig(
    zoom_enter=0.50,
    zoom_exit=0.20,
    scroll_enter=1.00,
    scroll_exit=0.40,
    settle_seconds=0.30,
)

QUIET = {"zoom_speed": 0.0, "scroll_speed": 0.0}


class TestConfigValidation:
    @pytest.mark.parametrize("field", ["zoom_enter", "zoom_exit", "scroll_enter", "scroll_exit"])
    def test_rejects_a_non_positive_threshold(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} must be positive"):
            ModeLockConfig(**{field: 0.0})  # type: ignore[arg-type]

    def test_rejects_a_zoom_exit_above_its_enter_threshold(self) -> None:
        with pytest.raises(ValueError, match=r"zoom_exit .* must not exceed zoom_enter"):
            ModeLockConfig(zoom_enter=0.3, zoom_exit=0.9)

    def test_rejects_a_scroll_exit_above_its_enter_threshold(self) -> None:
        with pytest.raises(ValueError, match=r"scroll_exit .* must not exceed scroll_enter"):
            ModeLockConfig(scroll_enter=0.3, scroll_exit=0.9)

    def test_allows_equal_enter_and_exit_meaning_no_hysteresis(self) -> None:
        assert ModeLockConfig(zoom_enter=0.4, zoom_exit=0.4).zoom_exit == 0.4

    def test_rejects_a_negative_settle_delay(self) -> None:
        with pytest.raises(ValueError, match="settle_seconds must be non-negative"):
            ModeLockConfig(settle_seconds=-0.1)


class TestAcquiringAMode:
    def test_starts_idle(self) -> None:
        assert ModeLock(CONFIG).mode is GestureMode.IDLE

    def test_stays_idle_for_a_motionless_hand(self) -> None:
        lock = ModeLock(CONFIG)
        assert lock.update(**QUIET, timestamp=0.0) is GestureMode.IDLE

    def test_stays_idle_for_signals_just_below_their_thresholds(self) -> None:
        lock = ModeLock(CONFIG)
        mode = lock.update(zoom_speed=0.49, scroll_speed=0.99, timestamp=0.0)
        assert mode is GestureMode.IDLE

    def test_a_pinch_crossing_its_threshold_claims_zooming(self) -> None:
        lock = ModeLock(CONFIG)
        assert lock.update(zoom_speed=0.60, scroll_speed=0.0, timestamp=0.0) is GestureMode.ZOOMING

    def test_a_swipe_crossing_its_threshold_claims_scrolling(self) -> None:
        lock = ModeLock(CONFIG)
        mode = lock.update(zoom_speed=0.0, scroll_speed=1.20, timestamp=0.0)
        assert mode is GestureMode.SCROLLING

    @pytest.mark.parametrize("sign", [1, -1])
    def test_direction_of_travel_does_not_affect_which_mode_is_claimed(self, sign: int) -> None:
        # Zooming in and zooming out are the same gesture as far as ownership
        # is concerned; only the magnitude decides.
        lock = ModeLock(CONFIG)
        mode = lock.update(zoom_speed=sign * 0.60, scroll_speed=0.0, timestamp=0.0)
        assert mode is GestureMode.ZOOMING

    def test_exactly_reaching_a_threshold_is_enough(self) -> None:
        lock = ModeLock(CONFIG)
        assert lock.update(zoom_speed=0.50, scroll_speed=0.0, timestamp=0.0) is GestureMode.ZOOMING


class TestArbitrationBetweenSimultaneousSignals:
    def test_the_signal_that_clears_its_bar_by_more_wins(self) -> None:
        # Scroll is numerically larger but only 1.1x its threshold, while the
        # pinch is 1.8x its own. Comparing raw magnitudes would pick wrongly.
        lock = ModeLock(CONFIG)
        mode = lock.update(zoom_speed=0.90, scroll_speed=1.10, timestamp=0.0)
        assert mode is GestureMode.ZOOMING

    def test_the_comparison_works_in_the_other_direction_too(self) -> None:
        lock = ModeLock(CONFIG)
        mode = lock.update(zoom_speed=0.55, scroll_speed=2.50, timestamp=0.0)
        assert mode is GestureMode.SCROLLING

    def test_only_one_signal_needs_to_cross_for_the_other_to_be_ignored(self) -> None:
        lock = ModeLock(CONFIG)
        mode = lock.update(zoom_speed=0.10, scroll_speed=1.50, timestamp=0.0)
        assert mode is GestureMode.SCROLLING

    def test_an_exact_tie_resolves_deterministically(self) -> None:
        first = ModeLock(CONFIG).update(zoom_speed=0.50, scroll_speed=1.00, timestamp=0.0)
        second = ModeLock(CONFIG).update(zoom_speed=0.50, scroll_speed=1.00, timestamp=0.0)
        assert first is second is GestureMode.ZOOMING


class TestHoldingAMode:
    def test_a_locked_mode_ignores_the_rival_signal_entirely(self) -> None:
        # The core promise: while zooming, even a very fast swipe cannot scroll.
        lock = ModeLock(CONFIG)
        lock.update(zoom_speed=0.80, scroll_speed=0.0, timestamp=0.0)
        for frame in range(1, 10):
            mode = lock.update(zoom_speed=0.80, scroll_speed=9.99, timestamp=frame / 30)
            assert mode is GestureMode.ZOOMING

    def test_scrolling_likewise_suppresses_incidental_pinching(self) -> None:
        lock = ModeLock(CONFIG)
        lock.update(zoom_speed=0.0, scroll_speed=1.50, timestamp=0.0)
        for frame in range(1, 10):
            mode = lock.update(zoom_speed=9.99, scroll_speed=1.50, timestamp=frame / 30)
            assert mode is GestureMode.SCROLLING

    def test_a_mode_survives_dipping_into_the_hysteresis_band(self) -> None:
        # Between exit (0.20) and enter (0.50) the gesture is neither strong
        # enough to start nor weak enough to stop; it must simply continue.
        lock = ModeLock(CONFIG)
        lock.update(zoom_speed=0.80, scroll_speed=0.0, timestamp=0.0)
        for frame in range(1, 60):
            mode = lock.update(zoom_speed=0.30, scroll_speed=0.0, timestamp=frame / 30)
            assert mode is GestureMode.ZOOMING

    def test_a_signal_hovering_at_the_threshold_does_not_flicker(self) -> None:
        # Alternating either side of the enter threshold is exactly the case a
        # single-threshold design would chatter on.
        lock = ModeLock(CONFIG)
        lock.update(zoom_speed=0.80, scroll_speed=0.0, timestamp=0.0)
        modes = {
            lock.update(
                zoom_speed=0.45 if frame % 2 else 0.55,
                scroll_speed=0.0,
                timestamp=frame / 30,
            )
            for frame in range(1, 60)
        }
        assert modes == {GestureMode.ZOOMING}


class TestReleasingAMode:
    def test_a_brief_pause_does_not_release_the_mode(self) -> None:
        lock = ModeLock(CONFIG)
        lock.update(zoom_speed=0.80, scroll_speed=0.0, timestamp=0.0)
        assert lock.update(**QUIET, timestamp=0.20) is GestureMode.ZOOMING

    def test_the_mode_releases_once_the_hand_has_settled(self) -> None:
        lock = ModeLock(CONFIG)
        lock.update(zoom_speed=0.80, scroll_speed=0.0, timestamp=0.0)
        assert lock.update(**QUIET, timestamp=0.31) is GestureMode.IDLE

    def test_the_settle_timer_restarts_whenever_the_gesture_resumes(self) -> None:
        lock = ModeLock(CONFIG)
        lock.update(zoom_speed=0.80, scroll_speed=0.0, timestamp=0.0)
        lock.update(**QUIET, timestamp=0.20)
        lock.update(zoom_speed=0.80, scroll_speed=0.0, timestamp=0.25)  # resumed
        assert lock.update(**QUIET, timestamp=0.50) is GestureMode.ZOOMING

    def test_scrolling_releases_on_the_same_rules(self) -> None:
        lock = ModeLock(CONFIG)
        lock.update(zoom_speed=0.0, scroll_speed=1.50, timestamp=0.0)
        assert lock.update(**QUIET, timestamp=0.20) is GestureMode.SCROLLING
        assert lock.update(**QUIET, timestamp=0.31) is GestureMode.IDLE

    def test_a_zero_settle_delay_releases_immediately(self) -> None:
        lock = ModeLock(ModeLockConfig(settle_seconds=0.0))
        lock.update(zoom_speed=5.0, scroll_speed=0.0, timestamp=0.0)
        assert lock.update(**QUIET, timestamp=0.0) is GestureMode.IDLE


class TestSwitchingBetweenModes:
    def test_the_user_can_switch_gestures_after_settling(self) -> None:
        lock = ModeLock(CONFIG)
        lock.update(zoom_speed=0.80, scroll_speed=0.0, timestamp=0.0)
        lock.update(**QUIET, timestamp=0.40)  # settle
        mode = lock.update(zoom_speed=0.0, scroll_speed=1.50, timestamp=0.50)
        assert mode is GestureMode.SCROLLING

    def test_switching_costs_no_extra_frame_once_settled(self) -> None:
        # The frame that releases zooming may claim scrolling straight away,
        # so a settled hand starting a new gesture responds immediately.
        lock = ModeLock(CONFIG)
        lock.update(zoom_speed=0.80, scroll_speed=0.0, timestamp=0.0)
        lock.update(**QUIET, timestamp=0.20)
        mode = lock.update(zoom_speed=0.0, scroll_speed=1.50, timestamp=0.35)
        assert mode is GestureMode.SCROLLING


class TestLosingTheHand:
    def test_release_drops_ownership_at_once(self) -> None:
        lock = ModeLock(CONFIG)
        lock.update(zoom_speed=0.80, scroll_speed=0.0, timestamp=0.0)
        lock.release()
        assert lock.mode is GestureMode.IDLE

    def test_release_bypasses_the_settle_delay(self) -> None:
        # A hand that has left the frame cannot be mid-gesture.
        lock = ModeLock(CONFIG)
        lock.update(zoom_speed=0.0, scroll_speed=1.50, timestamp=0.0)
        lock.release()
        assert lock.update(**QUIET, timestamp=0.01) is GestureMode.IDLE

    def test_a_returning_hand_can_claim_a_different_mode_at_once(self) -> None:
        lock = ModeLock(CONFIG)
        lock.update(zoom_speed=0.80, scroll_speed=0.0, timestamp=0.0)
        lock.release()
        mode = lock.update(zoom_speed=0.0, scroll_speed=1.50, timestamp=0.02)
        assert mode is GestureMode.SCROLLING

    def test_release_is_safe_to_call_when_already_idle(self) -> None:
        lock = ModeLock(CONFIG)
        lock.release()
        lock.release()
        assert lock.mode is GestureMode.IDLE
