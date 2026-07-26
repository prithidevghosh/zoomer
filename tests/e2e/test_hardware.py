"""Opt-in tests that touch real hardware.

Everything else in the suite substitutes the camera, MediaPipe, and the window,
which is what keeps it fast and deterministic. These tests exist to catch the
one class of bug that approach cannot: a mismatch between our assumptions and
the real libraries — an API that changed, a model bundle that will not load, a
platform that refuses synthetic input.

They are deselected by default. Run them deliberately:

    pytest -m hardware

The MediaPipe test downloads a model bundle on first run and therefore needs
network access once. The camera test needs a working camera and, on macOS,
camera permission for the terminal.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from zoomer.backends.desktop import DesktopBackend
from zoomer.tracking.camera import Camera, CameraError
from zoomer.tracking.hand_tracker import MediaPipeHandTracker, TrackerError, ensure_model

pytestmark = pytest.mark.hardware


class TestRealModel:
    def test_the_published_model_bundle_downloads_and_is_not_empty(self) -> None:
        path = ensure_model()
        assert path.exists()
        assert path.stat().st_size > 1_000_000

    def test_the_tracker_loads_the_real_model(self) -> None:
        with MediaPipeHandTracker() as tracker:
            assert tracker is not None

    def test_the_tracker_reports_no_hand_in_a_blank_frame(self) -> None:
        # Also proves our timestamp handling satisfies MediaPipe's video mode,
        # which rejects timestamps that fail to advance.
        numpy = pytest.importorskip("numpy")
        from zoomer.tracking.camera import Frame

        blank = numpy.zeros((480, 640, 3), dtype=numpy.uint8)
        with MediaPipeHandTracker() as tracker:
            for i in range(5):
                assert tracker.detect(Frame(image=blank, timestamp=i / 30)) is None

    def test_repeated_frame_timestamps_are_accepted(self) -> None:
        numpy = pytest.importorskip("numpy")
        from zoomer.tracking.camera import Frame

        blank = numpy.zeros((480, 640, 3), dtype=numpy.uint8)
        with MediaPipeHandTracker() as tracker:
            for _ in range(3):
                tracker.detect(Frame(image=blank, timestamp=1.0))

    def test_a_closed_tracker_refuses_further_frames(self) -> None:
        numpy = pytest.importorskip("numpy")
        from zoomer.tracking.camera import Frame

        tracker = MediaPipeHandTracker()
        tracker.close()
        with pytest.raises(TrackerError, match="closed tracker"):
            tracker.detect(Frame(image=numpy.zeros((480, 640, 3), numpy.uint8), timestamp=0.0))


class TestRealCamera:
    def test_the_default_camera_delivers_frames(self) -> None:
        try:
            camera = Camera(index=0)
        except CameraError as error:
            pytest.skip(f"no usable camera: {error}")

        with camera:
            frames = []
            for frame in camera.frames():
                frames.append(frame)
                if len(frames) == 5:
                    break

        assert len(frames) == 5
        assert all(frame.width > 0 and frame.height > 0 for frame in frames)
        assert all(later.timestamp >= earlier.timestamp for earlier, later in pairwise(frames))

    def test_an_absurd_camera_index_fails_with_a_helpful_message(self) -> None:
        with pytest.raises(CameraError, match="could not open camera"):
            Camera(index=99)


class TestRealDesktopBackend:
    def test_the_backend_opens_on_this_platform(self) -> None:
        # Deliberately does *not* send any input: a test that typed into
        # whatever window happened to be focused would be hostile.
        try:
            backend = DesktopBackend()
        except RuntimeError as error:
            pytest.skip(f"synthetic input unavailable: {error}")
        backend.close()
