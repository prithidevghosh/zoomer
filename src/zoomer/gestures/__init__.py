"""Camera-free gesture logic: filtering, feature extraction, and arbitration."""

from zoomer.gestures.filters import OneEuroFilter, apply_deadzone

__all__ = ["OneEuroFilter", "apply_deadzone"]
