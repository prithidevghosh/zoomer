"""Unit tests for frame capture and landmark conversion.

The MediaPipe model itself is not tested here — that is Google's code — but
everything around it is: the conversion from landmarks to domain types, the
handling of malformed detections, and the model-download cache.
"""

from __future__ import annotations

import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from zoomer.tracking.camera import Frame
from zoomer.tracking.hand_tracker import TrackerError, default_model_path, ensure_model
from zoomer.tracking.landmarks import (
    INDEX_MCP,
    INDEX_TIP,
    REQUIRED_LANDMARK_COUNT,
    THUMB_TIP,
    WRIST,
    to_observation,
)


@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float = 0.0


def hand_landmarks(count: int = 21) -> list[FakeLandmark]:
    """A plausible 21-point detection with recognisable coordinates."""
    landmarks = [FakeLandmark(x=0.5, y=0.5) for _ in range(count)]
    if count > WRIST:
        landmarks[WRIST] = FakeLandmark(x=0.50, y=0.90)
    if count > THUMB_TIP:
        landmarks[THUMB_TIP] = FakeLandmark(x=0.40, y=0.55)
    if count > INDEX_MCP:
        landmarks[INDEX_MCP] = FakeLandmark(x=0.52, y=0.70)
    if count > INDEX_TIP:
        landmarks[INDEX_TIP] = FakeLandmark(x=0.60, y=0.50)
    return landmarks


class FakeImage:
    """Stands in for a numpy array, exposing only the shape Frame reads."""

    def __init__(self, height: int, width: int) -> None:
        self.shape = (height, width, 3)


class TestLandmarkConversion:
    def test_reads_the_four_landmarks_the_gestures_need(self) -> None:
        observation = to_observation(hand_landmarks(), timestamp=1.5, aspect_ratio=4 / 3)
        assert observation is not None
        assert (observation.thumb_tip.x, observation.thumb_tip.y) == (0.40, 0.55)
        assert (observation.index_tip.x, observation.index_tip.y) == (0.60, 0.50)
        assert (observation.index_mcp.x, observation.index_mcp.y) == (0.52, 0.70)
        assert (observation.wrist.x, observation.wrist.y) == (0.50, 0.90)

    def test_carries_the_timestamp_and_aspect_ratio_through(self) -> None:
        observation = to_observation(hand_landmarks(), timestamp=1.5, aspect_ratio=16 / 9)
        assert observation is not None
        assert observation.timestamp == 1.5
        assert observation.aspect_ratio == pytest.approx(16 / 9)

    def test_ignores_the_landmarks_the_brief_does_not_use(self) -> None:
        # Only the thumb and index finger drive gestures, so moving the other
        # fingers must change nothing at all.
        baseline = to_observation(hand_landmarks(), timestamp=0.0, aspect_ratio=1.0)
        moved = hand_landmarks()
        for i in (12, 16, 20):  # middle, ring, and little fingertips
            moved[i] = FakeLandmark(x=0.01, y=0.99)
        assert to_observation(moved, timestamp=0.0, aspect_ratio=1.0) == baseline

    def test_a_truncated_detection_is_treated_as_no_hand(self) -> None:
        # Dropped and partial results are ordinary in a live stream; they must
        # not raise and interrupt the run loop.
        short = hand_landmarks(REQUIRED_LANDMARK_COUNT - 1)
        assert to_observation(short, timestamp=0.0, aspect_ratio=1.0) is None

    def test_an_empty_detection_is_treated_as_no_hand(self) -> None:
        assert to_observation([], timestamp=0.0, aspect_ratio=1.0) is None

    def test_the_shortest_usable_detection_is_accepted(self) -> None:
        exact = hand_landmarks(REQUIRED_LANDMARK_COUNT)
        assert to_observation(exact, timestamp=0.0, aspect_ratio=1.0) is not None

    def test_integer_coordinates_are_coerced_to_floats(self) -> None:
        landmarks = hand_landmarks()
        landmarks[INDEX_TIP] = FakeLandmark(x=1, y=0)  # type: ignore[arg-type]
        observation = to_observation(landmarks, timestamp=0.0, aspect_ratio=1.0)
        assert observation is not None
        assert isinstance(observation.index_tip.x, float)


class TestFrame:
    def test_reports_its_dimensions_from_the_image(self) -> None:
        frame = Frame(image=FakeImage(height=480, width=640), timestamp=0.0)
        assert (frame.width, frame.height) == (640, 480)

    def test_computes_the_aspect_ratio_used_to_correct_coordinates(self) -> None:
        frame = Frame(image=FakeImage(height=720, width=1280), timestamp=0.0)
        assert frame.aspect_ratio == pytest.approx(16 / 9)

    def test_a_degenerate_frame_does_not_divide_by_zero(self) -> None:
        frame = Frame(image=FakeImage(height=0, width=640), timestamp=0.0)
        assert frame.aspect_ratio == 1.0


class TestModelCache:
    def test_the_default_location_is_under_the_users_cache_directory(self) -> None:
        path = default_model_path()
        assert path.name == "hand_landmarker.task"
        assert path.parent.name == "zoomer"

    def test_an_already_cached_bundle_is_not_downloaded_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cached = tmp_path / "hand_landmarker.task"
        cached.write_bytes(b"model")

        def fail(*_: Any, **__: Any) -> None:
            raise AssertionError("a cached bundle must not trigger a download")

        monkeypatch.setattr("urllib.request.urlopen", fail)
        assert ensure_model(cached) == cached

    def test_a_missing_bundle_is_downloaded_and_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "nested" / "hand_landmarker.task"

        class FakeResponse:
            def read(self) -> bytes:
                return b"downloaded model"

            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *_: object) -> None:
                pass

        monkeypatch.setattr("urllib.request.urlopen", lambda *_, **__: FakeResponse())
        assert ensure_model(target).read_bytes() == b"downloaded model"

    def test_an_interrupted_download_leaves_no_truncated_bundle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A half-written model would load and then fail confusingly on every
        # later run, so the cache must be all-or-nothing.
        target = tmp_path / "hand_landmarker.task"

        def explode(*_: Any, **__: Any) -> None:
            raise urllib.error.URLError("connection reset")

        monkeypatch.setattr("urllib.request.urlopen", explode)
        with pytest.raises(TrackerError, match="could not download"):
            ensure_model(target)

        assert not target.exists()
        assert list(target.parent.glob("*.partial")) == []

    def test_the_download_error_explains_how_to_recover(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "hand_landmarker.task"
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_, **__: (_ for _ in ()).throw(urllib.error.URLError("offline")),
        )
        with pytest.raises(TrackerError, match=r"tracking\.model_path"):
            ensure_model(target)
