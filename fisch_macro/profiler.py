from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from fisch_macro.types import ProfileData


class Profiler:
    def __init__(self, data: ProfileData) -> None:
        self.data = data

    @classmethod
    def load(cls, path: str | Path) -> "Profiler":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(ProfileData(**raw))

    def save(self, path: str | Path) -> None:
        self.data.saved_at = date.today().isoformat()
        Path(path).write_text(json.dumps(asdict(self.data), indent=2), encoding="utf-8")

    def update_shake_radius(self, radius: float) -> None:
        self.data.shake_radius_px_mean = 0.8 * self.data.shake_radius_px_mean + 0.2 * radius

    def update_bar_width_ratio(self, ratio: float) -> None:
        self.data.bar_width_ratio = 0.8 * self.data.bar_width_ratio + 0.2 * ratio

    def update_fish_volatility(self, volatility: float) -> None:
        self.data.fish_volatility = 0.8 * self.data.fish_volatility + 0.2 * volatility
