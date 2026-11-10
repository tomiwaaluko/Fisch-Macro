from __future__ import annotations

import cv2
import numpy as np

from fisch_macro.types import Observation, Rect


def _track_bounds(gray: np.ndarray, bar: Rect, frame_width: int) -> tuple[int, int] | None:
    # The white bar protrudes slightly above and below the dark track.
    slop = max(2, round(bar.h * 0.22))
    edges = np.abs(np.diff(gray.astype(np.int16), axis=0))
    top = edges[max(0, bar.y - slop) : bar.y + slop].max(axis=0) > 10
    bottom = edges[bar.y + bar.h - slop : bar.y + bar.h + slop].max(axis=0) > 10
    interior = gray[bar.y + bar.h // 4 : bar.y + 3 * bar.h // 4]
    dark = np.median(interior, axis=0) < 62
    dark[bar.x : bar.x + bar.w] = True
    aligned = (top & bottom & dark).astype(np.uint8)[None, :]
    # Bridge the short gap where the fish marker crosses both edges.
    kernel = np.ones((1, max(3, round(frame_width * 0.025) | 1)), dtype=np.uint8)
    joined = cv2.morphologyEx(aligned, cv2.MORPH_CLOSE, kernel)[0]
    runs = np.flatnonzero(np.diff(np.r_[0, joined, 0])).reshape(-1, 2)
    for left, right in runs:
        if left <= bar.x and right >= bar.x + bar.w and right - left >= 0.33 * frame_width:
            return int(left), int(right)
    return None


def _marker_x(track: np.ndarray, fringe: int) -> float | None:
    # A fish marker is a narrow stroke with similar color across the track and
    # into both fringes. The arrows are confined to the white bar.
    bar_height = track.shape[0] - 2 * fringe
    stroke = max(5, round(bar_height * 0.62) | 1)
    kernel = np.ones((1, stroke), dtype=np.uint8)
    narrow = np.maximum(
        cv2.morphologyEx(track, cv2.MORPH_BLACKHAT, kernel),
        cv2.morphologyEx(track, cv2.MORPH_TOPHAT, kernel),
    )
    fringe_contrast = np.minimum(narrow[:fringe].mean(axis=0), narrow[-fringe:].mean(axis=0))
    inner_contrast = narrow[fringe:-fringe].mean(axis=0)
    shade_range = np.ptp(
        [np.median(part, axis=0) for part in (track[:fringe], track[fringe:-fringe], track[-fringe:])],
        axis=0,
    )
    valid = (inner_contrast >= 14) & (fringe_contrast >= 9) & (shade_range <= 30)
    scores = np.where(valid, fringe_contrast, 0)
    # Favor a stroke supported by neighboring columns over a single noisy pixel.
    support = max(1, round(bar_height * 0.10) | 1)
    scores = np.where(valid, np.convolve(scores, np.ones(support), mode="same"), 0)
    peak = float(scores.max())
    if peak == 0:
        return None
    candidates = np.flatnonzero(scores >= 0.8 * peak)
    if candidates[-1] - candidates[0] >= stroke:
        return None
    return float(np.average(candidates, weights=scores[candidates]))


def detect_reel(image_bgr: np.ndarray) -> Observation:
    h, w = image_bgr.shape[:2]
    left, top = int(w * 0.15), int(h * 0.72)
    roi = image_bgr[top : int(h * 0.95), left : int(w * 0.85)]
    if roi.size == 0:
        return Observation(False)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bright = cv2.inRange(gray, 181, 255)
    kernel = np.ones((1, max(5, round(h * 0.014) | 1)), dtype=np.uint8)
    white_bar = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    _, _, stats, _ = cv2.connectedComponentsWithStats(white_bar)
    for x, y, width, height, area in sorted(stats[1:], key=lambda rect: rect[-1], reverse=True):
        if not (max(4, 0.018 * h) <= height <= 0.08 * h and width >= max(2 * height, 0.035 * w)):
            continue
        if area < 0.8 * width * height:
            continue
        bar = Rect(int(x), int(y), int(width), int(height))
        fringe = max(2, bar.h // 5)
        if bar.y < fringe or bar.y + bar.h + fringe > gray.shape[0]:
            continue
        bounds = _track_bounds(gray, bar, w)
        if bounds is None:
            continue
        track_left, track_right = bounds
        track = gray[bar.y - fringe : bar.y + bar.h + fringe, track_left:track_right]
        fish_x = _marker_x(track, fringe)
        if fish_x is None:
            continue
        span = track_right - track_left - 1
        # Keep the existing lower-ROI progress estimate.
        progress_top, progress_bottom = int(h * 0.70), int(h * 0.95)
        progress_start = progress_top + int((progress_bottom - progress_top) * 0.7)
        progress_mask = bright[progress_start - top :, int(w * 0.20) - left : int(w * 0.80) - left]
        progress = float(progress_mask.mean() / 255.0) if progress_mask.size else 0.0
        return Observation(
            found=True,
            confidence=0.65,
            geometry=Rect(left + track_left, top + bar.y, track_right - track_left, bar.h),
            extras={
                "stable": True,
                "fish_x": fish_x / span,
                "bar_left": (bar.x - track_left) / span,
                "bar_right": (bar.x + bar.w - 1 - track_left) / span,
                "progress": progress,
            },
        )
    return Observation(False)
