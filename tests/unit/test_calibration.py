"""Unit tests for range-of-motion calibration."""

from __future__ import annotations

import pytest

from zoomer.calibration import (
    CalibrationError,
    CalibrationResult,
    Calibrator,
    percentile,
)
from zoomer.gestures.engine import EngineConfig
from zoomer.gestures.features import HandFeatures


def features(pinch: float, pointer: float = 0.0, timestamp: float = 0.0) -> HandFeatures:
    return HandFeatures(timestamp=timestamp, pinch=pinch, pointer=pointer)


def sweep(low: float, high: float, count: int) -> list[float]:
    """Evenly spaced samples from ``low`` to ``high`` inclusive."""
    if count == 1:
        return [low]
    step = (high - low) / (count - 1)
    return [low + i * step for i in range(count)]


class TestPercentile:
    def test_the_zeroth_percentile_is_the_smallest_sample(self) -> None:
        assert percentile([3.0, 1.0, 2.0], 0.0) == 1.0

    def test_the_hundredth_percentile_is_the_largest_sample(self) -> None:
        assert percentile([3.0, 1.0, 2.0], 1.0) == 3.0

    def test_the_median_of_an_odd_sample_is_the_middle_value(self) -> None:
        assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0

    def test_interpolates_between_neighbouring_samples(self) -> None:
        # Stepping between samples instead would make the result jump around as
        # frames arrive.
        assert percentile([0.0, 10.0], 0.25) == pytest.approx(2.5)

    def test_a_single_sample_is_its_own_every_percentile(self) -> None:
        assert percentile([7.0], 0.0) == percentile([7.0], 1.0) == 7.0

    def test_does_not_care_what_order_samples_arrived_in(self) -> None:
        values = [5.0, 1.0, 4.0, 2.0, 3.0]
        assert percentile(values, 0.5) == percentile(sorted(values), 0.5)

    def test_rejects_an_empty_sample(self) -> None:
        with pytest.raises(ValueError, match="no samples"):
            percentile([], 0.5)

    @pytest.mark.parametrize("fraction", [-0.1, 1.1])
    def test_rejects_a_fraction_outside_the_unit_interval(self, fraction: float) -> None:
        with pytest.raises(ValueError, match=r"fraction must be in \[0, 1\]"):
            percentile([1.0], fraction)


class TestCalibratorValidation:
    def test_rejects_a_sample_requirement_below_one(self) -> None:
        with pytest.raises(ValueError, match="min_samples must be at least 1"):
            Calibrator(min_samples=0)

    @pytest.mark.parametrize("trim", [-0.01, 0.5, 0.9])
    def test_rejects_a_trim_that_would_discard_the_whole_range(self, trim: float) -> None:
        with pytest.raises(ValueError, match=r"trim must be in \[0, 0.5\)"):
            Calibrator(trim=trim)


class TestCollectingSamples:
    def test_starts_empty_and_not_ready(self) -> None:
        calibrator = Calibrator()
        assert calibrator.sample_count == 0
        assert calibrator.ready is False

    def test_counts_the_samples_it_receives(self) -> None:
        calibrator = Calibrator(min_samples=5)
        for value in sweep(0.2, 1.2, 3):
            calibrator.observe(features(value))
        assert calibrator.sample_count == 3

    def test_becomes_ready_once_the_threshold_is_reached(self) -> None:
        calibrator = Calibrator(min_samples=3)
        for value in sweep(0.2, 1.2, 3):
            calibrator.observe(features(value))
        assert calibrator.ready is True

    def test_refuses_to_summarise_too_few_samples(self) -> None:
        calibrator = Calibrator(min_samples=10)
        calibrator.observe(features(0.5))
        with pytest.raises(CalibrationError, match="need at least 10 samples"):
            calibrator.result()

    def test_the_shortfall_error_tells_the_user_what_to_do(self) -> None:
        with pytest.raises(CalibrationError, match="Keep your hand in view"):
            Calibrator().result()

    def test_reset_discards_everything_collected(self) -> None:
        calibrator = Calibrator(min_samples=2)
        for value in sweep(0.2, 1.2, 5):
            calibrator.observe(features(value))
        calibrator.reset()
        assert calibrator.sample_count == 0
        assert calibrator.ready is False


class TestMeasuringRange:
    def test_measures_the_span_the_pinch_covered(self) -> None:
        calibrator = Calibrator(min_samples=5, trim=0.0)
        for value in sweep(0.2, 1.2, 50):
            calibrator.observe(features(value))
        assert calibrator.result().pinch_range == pytest.approx(1.0)

    def test_measures_the_span_the_index_finger_travelled(self) -> None:
        calibrator = Calibrator(min_samples=5, trim=0.0)
        for value in sweep(-2.0, 1.0, 50):
            calibrator.observe(features(pinch=0.5, pointer=value))
        assert calibrator.result().pointer_range == pytest.approx(3.0)

    def test_reports_how_many_samples_contributed(self) -> None:
        calibrator = Calibrator(min_samples=5)
        for value in sweep(0.2, 1.2, 40):
            calibrator.observe(features(value))
        assert calibrator.result().samples == 40

    def test_a_motionless_hand_measures_no_range(self) -> None:
        calibrator = Calibrator(min_samples=5)
        for _ in range(30):
            calibrator.observe(features(0.5, pointer=1.0))
        result = calibrator.result()
        assert result.pinch_range == pytest.approx(0.0)
        assert result.pointer_range == pytest.approx(0.0)

    def test_one_wild_landmark_does_not_inflate_the_measured_range(self) -> None:
        # A tracker glitch reaching the range unchecked would leave the gains
        # far too low for the whole session.
        honest = Calibrator(min_samples=5)
        glitched = Calibrator(min_samples=5)
        for value in sweep(0.2, 1.2, 100):
            honest.observe(features(value))
            glitched.observe(features(value))
        glitched.observe(features(87.0))

        assert glitched.result().pinch_range == pytest.approx(
            honest.result().pinch_range, rel=0.15
        )

    def test_trimming_can_be_switched_off(self) -> None:
        calibrator = Calibrator(min_samples=5, trim=0.0)
        for value in [*sweep(0.2, 1.2, 20), 50.0]:
            calibrator.observe(features(value))
        assert calibrator.result().pinch_range == pytest.approx(49.8)


class TestTuning:
    def test_a_narrow_range_earns_a_higher_gain(self) -> None:
        # Someone who moves less must get more movement per unit of motion.
        base = EngineConfig()
        narrow = CalibrationResult(pinch_range=0.5, pointer_range=1.0, samples=50)
        wide = CalibrationResult(pinch_range=2.0, pointer_range=1.0, samples=50)
        assert narrow.tune(base).zoom_gain > wide.tune(base).zoom_gain

    def test_a_full_sweep_delivers_the_requested_zoom(self) -> None:
        result = CalibrationResult(pinch_range=2.0, pointer_range=1.0, samples=50)
        tuned = result.tune(EngineConfig(), zoom_steps_per_sweep=10.0)
        assert tuned.zoom_gain * result.pinch_range == pytest.approx(10.0)

    def test_a_full_sweep_delivers_the_requested_scroll(self) -> None:
        result = CalibrationResult(pinch_range=1.0, pointer_range=3.0, samples=50)
        tuned = result.tune(EngineConfig(), scroll_clicks_per_sweep=12.0)
        assert tuned.scroll_gain * result.pointer_range == pytest.approx(12.0)

    def test_settings_unrelated_to_gain_are_left_alone(self) -> None:
        base = EngineConfig(max_steps_per_frame=7, beta=0.5)
        tuned = CalibrationResult(pinch_range=1.0, pointer_range=1.0, samples=50).tune(base)
        assert tuned.max_steps_per_frame == 7
        assert tuned.beta == 0.5
        assert tuned.mode_lock == base.mode_lock

    def test_the_original_configuration_is_not_modified(self) -> None:
        base = EngineConfig()
        original_gain = base.zoom_gain
        CalibrationResult(pinch_range=0.3, pointer_range=0.3, samples=50).tune(base)
        assert base.zoom_gain == original_gain

    def test_an_unmeasurably_small_range_leaves_the_gain_untouched(self) -> None:
        # Dividing by a near-zero range would produce an unusably twitchy gain.
        base = EngineConfig()
        barely_moved = CalibrationResult(pinch_range=0.001, pointer_range=0.001, samples=50)
        tuned = barely_moved.tune(base)
        assert tuned.zoom_gain == base.zoom_gain
        assert tuned.scroll_gain == base.scroll_gain

    def test_one_measurable_axis_is_tuned_even_if_the_other_is_not(self) -> None:
        base = EngineConfig()
        result = CalibrationResult(pinch_range=2.0, pointer_range=0.0, samples=50)
        tuned = result.tune(base)
        assert tuned.zoom_gain != base.zoom_gain
        assert tuned.scroll_gain == base.scroll_gain

    def test_the_tuned_configuration_is_valid(self) -> None:
        # tune() must never produce a config the engine would reject.
        result = CalibrationResult(pinch_range=1.5, pointer_range=2.5, samples=50)
        tuned = result.tune(EngineConfig())
        assert tuned.zoom_gain > 0
        assert tuned.scroll_gain > 0

    @pytest.mark.parametrize(
        "kwargs",
        [{"zoom_steps_per_sweep": 0.0}, {"scroll_clicks_per_sweep": -1.0}],
    )
    def test_rejects_a_non_positive_target(self, kwargs: dict[str, float]) -> None:
        result = CalibrationResult(pinch_range=1.0, pointer_range=1.0, samples=50)
        with pytest.raises(ValueError, match="must be positive"):
            result.tune(EngineConfig(), **kwargs)  # type: ignore[arg-type]
