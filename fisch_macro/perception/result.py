from __future__ import annotations

import cv2
import numpy as np

from fisch_macro.types import Observation


def detect_result(image_bgr: np.ndarray) -> Observation:
    h, w = image_bgr.shape[:2]
    roi = image_bgr[int(h * 0.25) : int(h * 0.55), int(w * 0.25) : int(w * 0.75)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bright_ratio = float((gray > 200).mean())
    found = bright_ratio > 0.08
    return Observation(found=found, confidence=bright_ratio, extras={"bright_ratio": bright_ratio})
