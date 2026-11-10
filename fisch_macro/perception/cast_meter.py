# fisch_macro/perception/cast_meter.py
from __future__ import annotations

import cv2
import numpy as np

from fisch_macro.types import Observation, Rect


def detect_cast_meter(image_bgr: np.ndarray) -> Observation:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (33, 40, 40), (91, 255, 255))
    bands, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for band in bands:
        x, y, width, height = cv2.boundingRect(band)
        if width < 3 or width > hsv.shape[0] // 8 or not 2 <= height <= 1.5 * width:
            continue
        if cv2.countNonZero(green[y : y + height, x : x + width]) < 0.6 * width * height:
            continue
        meter = _track_below_band(hsv, Rect(x, y, width, height))
        if meter is not None:
            candidates.append(meter)

    if len(candidates) != 1:
        return Observation(False)
    track, fill_height = candidates[0]
    return Observation(
        found=True,
        confidence=0.7,
        geometry=track,
        extras={"fill_ratio": fill_height / track.h, "green_band": True},
    )


def _track_below_band(hsv: np.ndarray, band: Rect) -> tuple[Rect, int] | None:
    # The green cap marks the track's top. White fill rises from its bottom,
    # so the bottom of an aligned fill identifies the other end of the track.
    padding = max(2, band.w // 3)
    left = max(0, band.x - padding)
    right = min(hsv.shape[1], band.x + band.w + padding)
    top = band.y + band.h
    bottom = min(hsv.shape[0], band.y + 40 * band.w)
    if top >= bottom:
        return None
    strip = hsv[top:bottom, left:right]
    white = cv2.inRange(strip, (0, 0, 180), (180, 40, 255))
    fills, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for x, y, width, height in sorted(map(cv2.boundingRect, fills), key=lambda box: box[1]):
        track_height = band.h + y + height
        clipped_horizontally = x == 0 or x + width == strip.shape[1]
        clipped_below = y + height == strip.shape[0] and bottom < hsv.shape[0]
        if track_height < 4 * band.w or clipped_horizontally or clipped_below:
            continue
        if not 0.5 * band.w <= width <= 1.4 * band.w or height < max(3, band.w // 2):
            continue
        if abs(left + x + width / 2 - (band.x + band.w / 2)) > band.w / 4:
            continue
        if cv2.countNonZero(white[y : y + height, x : x + width]) < 0.8 * width * height:
            continue
        gap = hsv[top : top + y, band.x : band.x + band.w, 2]
        if np.count_nonzero(gap > 128) > max(band.w, 0.025 * gap.size):
            continue
        return Rect(band.x, band.y, band.w, track_height), height
    return None
