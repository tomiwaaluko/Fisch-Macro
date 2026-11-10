# fisch_macro/controllers/cast.py
from __future__ import annotations

from fisch_macro.input_controller import InputController
from fisch_macro.types import ProfileData


class CastController:
    def __init__(
        self,
        input_ctl: InputController,
        profile: ProfileData,
        safe_xy: tuple[int, int],
        release_fill: float = 0.90,
    ) -> None:
        self.input = input_ctl
        self.profile = profile
        self.safe_xy = safe_xy
        self.release_fill = release_fill
        self._holding = False

    def begin_cast(self, client_origin: tuple[int, int]) -> bool:
        ox, oy = client_origin
        x, y = self.safe_xy
        if not self.input.move_to(ox + x, oy + y):
            return False
        ok = self.input.mouse_down()
        self._holding = ok
        return ok

    def update_fill(self, fill_ratio: float) -> bool:
        if not self._holding:
            return False
        lead = min(0.08, self.profile.cast_latency_ms / 1000.0)
        target = max(0.5, self.release_fill - lead)
        if fill_ratio >= target:
            self.input.mouse_up()
            self._holding = False
            return True
        return False

    def force_release(self) -> None:
        self.input.mouse_up()
        self._holding = False
