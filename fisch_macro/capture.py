from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from fisch_macro.types import FramePacket


@dataclass(frozen=True)
class WindowInfo:
    left: int
    top: int
    width: int
    height: int
    title: str
    handle: int


def find_roblox_window() -> WindowInfo | None:
    import pygetwindow as gw

    try:
        window = gw.getActiveWindow()
        if window is None or window.isMinimized or not window.visible:
            return None
        title = window.title
        if title.strip().casefold() != "roblox":
            return None
        left, top, width, height = window.box
        if width > 0 and height > 0:
            return WindowInfo(left, top, width, height, title, window._hWnd)
    except gw.PyGetWindowException:
        # The foreground window can close while its bounds are being read.
        return None
    return None


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
