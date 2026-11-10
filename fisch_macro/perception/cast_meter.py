# fisch_macro/perception/cast_meter.py
from __future__ import annotations

import cv2
import numpy as np

from fisch_macro.types import Observation, Rect


def detect_cast_meter(image_bgr: np.ndarray) -> Observation:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    if int(white.sum()) < 50:
        return Observation(False)
    ys, xs = np.where(white > 0)
    if len(ys) == 0:
        return Observation(False)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    h = image_bgr.shape[0]
    fill_ratio = 1.0 - (y0 / max(h, 1))
    return Observation(
        found=True,
        confidence=0.7,
        geometry=Rect(x0, y0, max(1, x1 - x0), max(1, y1 - y0)),
        extras={"fill_ratio": float(np.clip(fill_ratio, 0.0, 1.0)), "green_band": True},
    )
