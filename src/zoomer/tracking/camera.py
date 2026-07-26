"""Live frames from the device camera.

The rest of the program depends on the :class:`FrameSource` protocol rather than
on OpenCV, so tests can feed a scripted sequence of frames through the identical
code path a real webcam would drive.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    import numpy as np

__all__ = ["Camera", "CameraError", "Frame", "FrameSource"]


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or read."""


@dataclass(frozen=True, slots=True)
class Frame:
    """One captured image and the moment it was captured.

    Args:
        image: Pixel data in BGR order, as OpenCV produces.
        timestamp: Monotonic capture time in seconds.
    """

    image: Any  # np.ndarray; typed loosely so numpy stays an optional import
    timestamp: float

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self.image.shape[0])

    @property
    def aspect_ratio(self) -> float:
        """Width divided by height, for correcting normalised coordinates."""
        return self.width / self.height if self.height else 1.0


class FrameSource(Protocol):
    """Anything that can supply a stream of frames."""

    def frames(self) -> Iterator[Frame]:
        """Yield frames until the source is exhausted or closed."""
        ...

    def close(self) -> None:
        """Release the underlying device."""
        ...


class Camera:
    """A live capture device, read through OpenCV.

    Args:
        index: Device index. ``0`` is the default camera.
        width: Requested capture width in pixels.
        height: Requested capture height in pixels.

    Raises:
        CameraError: If the device cannot be opened, which most often means it
            is in use by another application or camera permission has not been
            granted.
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        # Imported lazily so importing zoomer does not pull in OpenCV on a
        # machine that only runs the pure gesture logic.
        import cv2

        self._cv2 = cv2
        self._capture = cv2.VideoCapture(index)
        if not self._capture.isOpened():
            self._capture.release()
            raise CameraError(
                f"could not open camera {index}. Check that no other application is "
                "using it, and that camera permission has been granted to your terminal."
            )

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._closed = False

    def frames(self) -> Iterator[Frame]:
        """Yield frames until the device stops delivering them.

        A failed read ends the stream rather than raising: cameras are routinely
        unplugged or claimed by another application mid-session, and the caller
        should shut down cleanly instead of seeing a traceback.

        Yields:
            Each captured :class:`Frame`, newest last.
        """
        while not self._closed:
            ok, image = self._capture.read()
            if not ok or image is None:
                return
            yield Frame(image=image, timestamp=time.monotonic())

    def close(self) -> None:
        """Release the capture device. Safe to call more than once."""
        if not self._closed:
            self._closed = True
            self._capture.release()

    def __enter__(self) -> Camera:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
