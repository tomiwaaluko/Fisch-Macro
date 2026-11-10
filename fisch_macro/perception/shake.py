from __future__ import annotations

import cv2
import numpy as np

from fisch_macro.types import Observation, Rect


def radius_limits_from_prior(mean_radius: float) -> tuple[int, int]:
    """Hough bounds from the learned radius. Smoothing stays in the profiler."""
    mean = max(1.0, float(mean_radius))
    min_radius = max(12, int(round(mean * 0.45)))
    max_radius = max(min_radius + 20, int(round(mean * 2.0)))
    return min_radius, max_radius


def detect_shake(image_bgr: np.ndarray, min_radius: int = 20, max_radius: int = 160) -> Observation:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(40, min_radius),
        param1=100,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return Observation(False)
    circles = np.round(circles[0]).astype(int)
    x, y, r = max(circles, key=lambda c: c[2])
    return Observation(
        found=True,
        confidence=0.8,
        geometry=Rect(int(x - r), int(y - r), int(2 * r), int(2 * r)),
        extras={"radius": int(r), "center": (int(x), int(y))},
    )
