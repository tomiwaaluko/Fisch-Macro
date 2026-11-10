from __future__ import annotations

import cv2
import numpy as np

from fisch_macro.types import Observation

# Brightness below this is a weak false positive and must not latch CATCH_RESULT.
_SUCCESS_BRIGHT_RATIO = 0.25
_FAIL_RED_RATIO = 0.12


def detect_result(image_bgr: np.ndarray) -> Observation:
    h, w = image_bgr.shape[:2]
    roi = image_bgr[int(h * 0.25) : int(h * 0.55), int(w * 0.25) : int(w * 0.75)]
    if roi.size == 0:
        return Observation(False, extras={"bright_ratio": 0.0, "red_ratio": 0.0, "outcome": "unknown"})
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bright_ratio = float((gray > 200).mean())
    red = roi[:, :, 2].astype(np.int16)
    green = roi[:, :, 1].astype(np.int16)
    blue = roi[:, :, 0].astype(np.int16)
    red_ratio = float(((red > 160) & (green < 100) & (blue < 100)).mean())
    extras = {"bright_ratio": bright_ratio, "red_ratio": red_ratio}
    if red_ratio >= _FAIL_RED_RATIO and red_ratio > bright_ratio:
        extras["outcome"] = "fail"
        return Observation(found=True, confidence=red_ratio, extras=extras)
    if bright_ratio >= _SUCCESS_BRIGHT_RATIO:
        extras["outcome"] = "success"
        return Observation(found=True, confidence=bright_ratio, extras=extras)
    extras["outcome"] = "unknown"
    return Observation(
        found=False,
        confidence=max(bright_ratio, red_ratio),
        extras=extras,
    )
