from __future__ import annotations

import cv2
import numpy as np

from fisch_macro.types import Observation, Rect


def detect_reel(image_bgr: np.ndarray) -> Observation:
    h, w = image_bgr.shape[:2]
    roi = image_bgr[int(h * 0.70) : int(h * 0.95), int(w * 0.20) : int(w * 0.80)]
    if roi.size == 0:
        return Observation(False)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    ys, xs = np.where(bright > 0)
    if len(xs) < 30:
        return Observation(False)
    bar_left = float(xs.min()) / max(roi.shape[1] - 1, 1)
    bar_right = float(xs.max()) / max(roi.shape[1] - 1, 1)
    col_means = gray.mean(axis=0)
    fish_idx = int(np.argmin(col_means))
    fish_x = fish_idx / max(roi.shape[1] - 1, 1)
    bottom = bright[int(bright.shape[0] * 0.7) :]
    progress = float(np.clip(bottom.mean() / 255.0, 0.0, 1.0))
    return Observation(
        found=True,
        confidence=0.65,
        geometry=Rect(int(w * 0.20), int(h * 0.70), int(w * 0.60), int(h * 0.25)),
        extras={
            "stable": True,
            "fish_x": fish_x,
            "bar_left": bar_left,
            "bar_right": bar_right,
            "progress": progress,
        },
    )
