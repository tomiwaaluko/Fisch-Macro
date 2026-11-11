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
    height, width = gray.shape
    short_side = min(height, width)
    # Bound saved radius priors when the capture size changes or the prior is stale.
    min_radius = min(min_radius, max(12, round(short_side * 0.07)))
    max_radius = min(max_radius, short_side // 4)
    if max_radius <= min_radius:
        return Observation(False)

    scale = min(1.0, 540 / short_side)
    search = gray
    if scale < 1.0:
        search = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(search, (5, 5), 1),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min_radius * scale),
        param1=70,
        param2=30,
        minRadius=round(min_radius * scale),
        maxRadius=round(max_radius * scale),
    )
    if circles is None:
        return Observation(False)

    for x, y, radius in np.rint(circles[0] / scale).astype(int).tolist():
        if not (radius <= x < width - radius and radius <= y < height - radius):
            continue
        if _has_shake_label(gray, x, y, radius) and _has_button_border(gray, x, y, radius):
            return Observation(
                found=True,
                confidence=0.8,
                geometry=Rect(x - radius, y - radius, 2 * radius, 2 * radius),
                extras={"radius": radius, "center": (x, y)},
            )
    return Observation(False)


def _has_shake_label(gray: np.ndarray, x: int, y: int, radius: int) -> bool:
    """Look for the bright, centered row of five glyphs inside the button."""
    half_width = round(0.83 * radius)
    half_height = round(0.29 * radius)
    label = gray[y - half_height : y + half_height + 1, x - half_width : x + half_width + 1]
    background = cv2.medianBlur(label, max(3, round(0.11 * radius) | 1))
    strokes = ((label > 135) & (cv2.subtract(label, background) > 24)).astype(np.uint8)
    _, _, components, _ = cv2.connectedComponentsWithStats(strokes)
    glyph_slots = set()
    for left, top, width, height, area in components[1:].tolist():
        if not (0.10 * radius <= width <= 0.34 * radius and 0.25 * radius <= height <= 0.45 * radius):
            continue
        if not 0.12 * width * height <= area <= 0.72 * width * height:
            continue
        if abs(top + height / 2 - half_height) > 0.13 * radius:
            continue
        offset = (left + width / 2 - half_width) / radius
        slot = round(offset / 0.31)
        if -2 <= slot <= 2 and abs(offset - 0.31 * slot) < 0.115:
            glyph_slots.add(slot)
    # A pointer or an overlay can obscure one letter.
    return len(glyph_slots) >= 4


def _has_button_border(gray: np.ndarray, x: int, y: int, radius: int) -> bool:
    """Require a pale outline around most of the circle, beyond any text."""
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    margin = max(6, round(0.12 * radius))
    radii = radius + np.arange(-margin, margin + 1)[:, None]
    ys = np.clip(np.rint(y + radii * np.sin(angles)).astype(int), 0, gray.shape[0] - 1)
    xs = np.clip(np.rint(x + radii * np.cos(angles)).astype(int), 0, gray.shape[1] - 1)
    samples = gray[ys, xs].astype(np.int16)
    contrast = samples.max(axis=0) - np.maximum(samples[0], samples[-1])
    return float((contrast >= 10).mean()) >= 0.70
