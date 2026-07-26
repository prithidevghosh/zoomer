"""Find a hand in a frame using MediaPipe's hand landmarker.

The tracker is expressed as a protocol so the run loop never depends on
MediaPipe directly; tests substitute a scripted tracker and exercise the same
loop the camera drives.

The model bundle is not vendored into the repository — it is several megabytes
of binary that would bloat every clone — so it is fetched once on first run and
cached under the user's cache directory.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from zoomer.tracking.camera import Frame
from zoomer.tracking.landmarks import to_observation
from zoomer.types import HandObservation

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    pass

__all__ = [
    "MODEL_URL",
    "HandTracker",
    "MediaPipeHandTracker",
    "TrackerError",
    "default_model_path",
    "ensure_model",
]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
"""Google's published hand-landmarker bundle. The float16 build is chosen for
its size and speed; the accuracy difference is immaterial for gestures this
coarse."""


class TrackerError(RuntimeError):
    """Raised when the hand tracker cannot be prepared or run."""


class HandTracker(Protocol):
    """Anything that can locate a hand in a frame."""

    def detect(self, frame: Frame) -> HandObservation | None:
        """Return the tracked hand, or ``None`` if no hand was found."""
        ...

    def close(self) -> None:
        """Release any resources held by the tracker."""
        ...


def default_model_path() -> Path:
    """Return where the downloaded model bundle is cached.

    Returns:
        ``~/.cache/zoomer/hand_landmarker.task``.
    """
    return Path.home() / ".cache" / "zoomer" / "hand_landmarker.task"


def ensure_model(path: Path | None = None, *, url: str = MODEL_URL) -> Path:
    """Return a local model bundle, downloading it if it is not yet cached.

    The download is written to a temporary file and moved into place only once
    complete, so an interrupted download cannot leave a truncated bundle that
    would fail confusingly on every later run.

    Args:
        path: Where to cache the bundle. Defaults to :func:`default_model_path`.
        url: Where to fetch it from if it is absent.

    Returns:
        The path to a model bundle that exists on disk.

    Raises:
        TrackerError: If the bundle is absent and cannot be downloaded.
    """
    target = path or default_model_path()
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")

    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https URL
            partial.write_bytes(response.read())
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        partial.unlink(missing_ok=True)
        raise TrackerError(
            f"could not download the hand tracking model from {url}: {error}. "
            f"Download it manually and save it to {target}, or point "
            "tracking.model_path at an existing copy."
        ) from error

    partial.replace(target)
    return target


class MediaPipeHandTracker:
    """Locate a single hand per frame with MediaPipe's landmarker.

    Args:
        model_path: A model bundle to load. When ``None``, the cached bundle is
            used and downloaded if necessary.
        min_detection_confidence: Score a hand must reach before tracking
            begins.
        min_tracking_confidence: Score required to keep following a hand.

    Raises:
        TrackerError: If MediaPipe is unavailable or the model cannot be loaded.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        bundle = ensure_model(model_path)

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as error:  # pragma: no cover - depends on environment
            raise TrackerError(
                "hand tracking requires mediapipe; install it with 'pip install zoomer'"
            ) from error

        self._mp = mp

        try:
            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(bundle)),
                # VIDEO mode carries tracking state between frames, which is
                # both faster and steadier than re-detecting from scratch.
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._landmarker: Any = vision.HandLandmarker.create_from_options(options)
        except Exception as error:  # pragma: no cover - depends on environment
            raise TrackerError(f"could not load the hand tracking model from {bundle}: {error}") from error

        self._closed = False

        # MediaPipe's VIDEO mode requires strictly increasing integer
        # millisecond timestamps. Frame clocks are floats and can repeat at high
        # frame rates, so a monotonic counter is kept rather than trusting them.
        self._last_timestamp_ms = -1

    def detect(self, frame: Frame) -> HandObservation | None:
        """Locate a hand in one frame.

        Args:
            frame: The captured frame, in BGR order.

        Returns:
            The tracked hand, or ``None`` if none was found.

        Raises:
            TrackerError: If the tracker has already been closed.
        """
        if self._closed:
            raise TrackerError("cannot detect with a closed tracker")

        import cv2

        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = max(int(frame.timestamp * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return None

        return to_observation(
            result.hand_landmarks[0],
            timestamp=frame.timestamp,
            aspect_ratio=frame.aspect_ratio,
        )

    def close(self) -> None:
        """Release the landmarker. Safe to call more than once."""
        if not self._closed:
            self._closed = True
            self._landmarker.close()

    def __enter__(self) -> MediaPipeHandTracker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
