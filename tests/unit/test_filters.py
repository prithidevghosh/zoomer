"""Unit tests for signal conditioning.

The properties asserted here are the ones the user actually feels: a still hand
must not drift, a fast gesture must not lag, and noise must not accumulate.
"""

from __future__ import annotations

import random
from itertools import pairwise

import pytest

from zoomer.gestures.filters import OneEuroFilter, apply_deadzone, smoothing_alpha


class TestSmoothingAlpha:
    def test_stays_within_the_open_unit_interval(self) -> None:
        for cutoff in (0.01, 1.0, 5.0, 120.0):
            alpha = smoothing_alpha(cutoff, dt=1 / 30)
            assert 0.0 < alpha <= 1.0

    def test_a_higher_cutoff_follows_the_input_more_closely(self) -> None:
        slow = smoothing_alpha(0.5, dt=1 / 30)
        fast = smoothing_alpha(20.0, dt=1 / 30)
        assert fast > slow

    def test_a_longer_gap_between_samples_follows_the_input_more_closely(self) -> None:
        # After a long stall the stored history is stale, so the new sample
        # should carry more weight.
        assert smoothing_alpha(1.0, dt=1.0) > smoothing_alpha(1.0, dt=1 / 60)

    @pytest.mark.parametrize(("cutoff", "dt"), [(0.0, 0.1), (-1.0, 0.1), (1.0, 0.0), (1.0, -0.1)])
    def test_rejects_non_positive_arguments(self, cutoff: float, dt: float) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            smoothing_alpha(cutoff, dt)


class TestOneEuroFilterConstruction:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"min_cutoff": 0.0}, "min_cutoff must be positive"),
            ({"min_cutoff": -1.0}, "min_cutoff must be positive"),
            ({"derivative_cutoff": 0.0}, "derivative_cutoff must be positive"),
            ({"beta": -0.1}, "beta must be non-negative"),
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs: dict[str, float], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            OneEuroFilter(**kwargs)

    def test_accepts_a_beta_of_zero_meaning_a_fixed_cutoff(self) -> None:
        assert OneEuroFilter(beta=0.0).beta == 0.0


class TestOneEuroFilterBehaviour:
    def test_passes_the_very_first_sample_through_untouched(self) -> None:
        assert OneEuroFilter().filter(0.42, timestamp=0.0) == pytest.approx(0.42)

    def test_a_perfectly_constant_signal_is_left_alone(self) -> None:
        f = OneEuroFilter(min_cutoff=1.0, beta=0.01)
        for i in range(60):
            assert f.filter(0.30, timestamp=i / 30) == pytest.approx(0.30)

    def test_a_still_hand_does_not_drift_under_noise(self) -> None:
        # The headline requirement: jitter must not integrate into motion.
        rng = random.Random(20260726)
        f = OneEuroFilter(min_cutoff=0.6, beta=0.005)
        raw, filtered = [], []
        for i in range(240):
            sample = 0.30 + rng.gauss(0, 0.004)
            raw.append(sample)
            filtered.append(f.filter(sample, timestamp=i / 30))

        settled_raw, settled = raw[60:], filtered[60:]
        # The excursion the user would see is a fraction of the raw jitter...
        assert (max(settled) - min(settled)) < (max(settled_raw) - min(settled_raw)) / 3
        # ...and it is centred, so the noise cannot integrate into slow drift.
        assert sum(settled) / len(settled) == pytest.approx(0.30, abs=0.001)

    def test_attenuates_noise_relative_to_the_raw_signal(self) -> None:
        rng = random.Random(7)
        f = OneEuroFilter(min_cutoff=0.6, beta=0.0)
        raw, filtered = [], []
        for i in range(200):
            sample = 0.5 + rng.gauss(0, 0.01)
            raw.append(sample)
            filtered.append(f.filter(sample, timestamp=i / 30))

        def spread(values: list[float]) -> float:
            mean = sum(values) / len(values)
            return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

        assert spread(filtered[50:]) < spread(raw[50:]) / 3

    def test_converges_onto_a_new_resting_value(self) -> None:
        f = OneEuroFilter(min_cutoff=1.0, beta=0.01)
        f.filter(0.0, timestamp=0.0)
        last = 0.0
        for i in range(1, 120):
            last = f.filter(1.0, timestamp=i / 30)
        assert last == pytest.approx(1.0, abs=1e-3)

    def test_never_overshoots_a_step_input(self) -> None:
        f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
        f.filter(0.0, timestamp=0.0)
        for i in range(1, 90):
            assert 0.0 <= f.filter(1.0, timestamp=i / 30) <= 1.0

    def test_tracks_fast_motion_more_faithfully_than_a_fixed_cutoff(self) -> None:
        # This is the whole point of the "speed-based" part of the design.
        def lag_over_a_ramp(beta: float) -> float:
            f = OneEuroFilter(min_cutoff=0.5, beta=beta)
            error = 0.0
            for i in range(40):
                t = i / 30
                truth = 2.0 * t  # fast, deliberate opening of the fingers
                error += abs(f.filter(truth, timestamp=t) - truth)
            return error

        assert lag_over_a_ramp(beta=0.5) < lag_over_a_ramp(beta=0.0)

    def test_a_repeated_timestamp_does_not_divide_by_zero(self) -> None:
        f = OneEuroFilter()
        f.filter(0.1, timestamp=1.0)
        assert f.filter(0.9, timestamp=1.0) == pytest.approx(0.9)

    def test_a_backwards_timestamp_is_handled_without_a_negative_timestep(self) -> None:
        f = OneEuroFilter()
        f.filter(0.1, timestamp=5.0)
        assert f.filter(0.7, timestamp=2.0) == pytest.approx(0.7)

    def test_reset_clears_history_so_the_next_sample_is_taken_verbatim(self) -> None:
        f = OneEuroFilter(min_cutoff=0.5)
        for i in range(30):
            f.filter(0.2, timestamp=i / 30)
        f.reset()
        assert f.filter(0.95, timestamp=10.0) == pytest.approx(0.95)


class TestDeadzone:
    @pytest.mark.parametrize("value", [0.0, 0.01, -0.01, 0.05, -0.05])
    def test_suppresses_magnitudes_at_or_below_the_threshold(self, value: float) -> None:
        assert apply_deadzone(value, threshold=0.05) == 0.0

    def test_preserves_the_sign_of_a_surviving_signal(self) -> None:
        assert apply_deadzone(0.20, threshold=0.05) > 0
        assert apply_deadzone(-0.20, threshold=0.05) < 0

    def test_shrinks_a_surviving_signal_by_exactly_the_threshold(self) -> None:
        assert apply_deadzone(0.20, threshold=0.05) == pytest.approx(0.15)
        assert apply_deadzone(-0.20, threshold=0.05) == pytest.approx(-0.15)

    def test_leaves_zero_smoothly_rather_than_jumping(self) -> None:
        # Just past the threshold the output must be near zero, not near the
        # threshold, or the document would lurch the moment a gesture starts.
        assert apply_deadzone(0.0501, threshold=0.05) == pytest.approx(0.0001, abs=1e-9)

    def test_a_zero_threshold_is_a_pass_through(self) -> None:
        assert apply_deadzone(-0.33, threshold=0.0) == pytest.approx(-0.33)

    def test_is_monotonic_so_a_stronger_gesture_always_wins(self) -> None:
        outputs = [apply_deadzone(v / 100, threshold=0.05) for v in range(100)]
        assert all(b >= a for a, b in pairwise(outputs))

    def test_rejects_a_negative_threshold(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            apply_deadzone(1.0, threshold=-0.1)
