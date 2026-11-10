from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from fisch_macro.types import FramePacket


@dataclass
class WindowInfo:
    left: int
    top: int
    width: int
    height: int
    title: str


def find_roblox_window(title_substr: str = "Roblox") -> WindowInfo | None:
    import pygetwindow as gw

    matches = [w for w in gw.getAllWindows() if title_substr.lower() in (w.title or "").lower()]
    if not matches:
        return None
    w = matches[0]
    if w.width <= 0 or w.height <= 0:
        return None
    return WindowInfo(w.left, w.top, w.width, w.height, w.title)


def grab_frame(window: WindowInfo) -> FramePacket:
    import mss
    from PIL import Image

    monitor = {
        "left": int(window.left),
        "top": int(window.top),
        "width": int(window.width),
        "height": int(window.height),
    }
    with mss.mss() as sct:
        raw = sct.grab(monitor)
    img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    bgr = np.array(img)[:, :, ::-1].copy()
    return FramePacket(
        image=bgr,
        timestamp_ms=int(time.time() * 1000),
        client_origin=(monitor["left"], monitor["top"]),
    )
