from __future__ import annotations

from fisch_macro.input_controller import InputController
from fisch_macro.types import Observation, ProfileData


class ShakeController:
    def __init__(self, input_ctl: InputController, profile: ProfileData) -> None:
        self.input = input_ctl
        self.profile = profile
        self._last_center: tuple[int, int] | None = None

    def tick(self, shake: Observation, bite_or_reel: bool, client_origin: tuple[int, int]) -> bool:
        if bite_or_reel:
            self.input.cancel()
            self.input.release_all()
            self.input.clear_cancel()
            return False
        if not shake.found or shake.geometry is None:
            return False
        cx, cy = shake.extras.get("center") or shake.geometry.center
        ox, oy = client_origin
        screen_x, screen_y = ox + int(cx), oy + int(cy)
        if self._last_center and abs(self._last_center[0] - cx) < 8 and abs(self._last_center[1] - cy) < 8:
            return False
        ok = self.input.click(screen_x, screen_y, hold_ms=self.profile.shake_click_hold_ms)
        if ok:
            self._last_center = (int(cx), int(cy))
        return ok

    def reset_target_memory(self) -> None:
        self._last_center = None
