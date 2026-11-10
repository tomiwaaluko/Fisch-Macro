from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


class BotState(Enum):
    IDLE = auto()
    PRE_CAST = auto()
    CASTING = auto()
    WAITING_SHAKE = auto()
    SHAKING = auto()
    BITE_DETECTED = auto()
    REEL_INITIALIZING = auto()
    REELING = auto()
    CATCH_RESULT = auto()
    RESET_TO_CAST = auto()
    RECOVERY = auto()


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


@dataclass
class Observation:
    found: bool
    confidence: float = 0.0
    geometry: Rect | None = None
    age_ms: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class FramePacket:
    """BGR image in client-relative coordinates."""

    image: np.ndarray
    timestamp_ms: int
    client_origin: tuple[int, int]  # top-left of Roblox client on desktop


@dataclass
class ProfileData:
    name: str
    saved_at: str
    cast_hold_ms_prior: float = 580.0
    cast_latency_ms: float = 90.0
    shake_radius_px_mean: float = 110.0
    shake_click_hold_ms: float = 70.0
    bar_width_ratio: float = 0.40
    fish_volatility: float = 0.5
    reel_safe_margin_ratio: float = 0.25
    reel_tighten_margin_ratio: float = 0.12
