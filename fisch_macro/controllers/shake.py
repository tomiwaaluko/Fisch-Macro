from __future__ import annotations

import time

from fisch_macro.input_controller import InputController
from fisch_macro.types import Observation, ProfileData

_RETRY_DELAY_SECONDS = 0.25
_MAX_ATTEMPTS = 2


class ShakeController:
    def __init__(self, input_ctl: InputController, profile: ProfileData) -> None:
        self.input = input_ctl
        self.profile = profile
        self._last_center: tuple[int, int] | None = None
        self._attempts = 0
        self._last_click_at = 0.0

    def tick(
        self,
        shake: Observation,
        bite_or_reel: bool,
        client_origin: tuple[int, int],
    ) -> bool:
        if bite_or_reel:
            self.input.release_all()
            self.reset_target_memory()
            return False
        if not shake.found or shake.geometry is None:
            self.reset_target_memory()
            return False
        cx, cy = shake.extras.get("center") or shake.geometry.center
        if self._last_center is None or max(
            abs(self._last_center[0] - cx), abs(self._last_center[1] - cy)
        ) >= 8:
            self.reset_target_memory()
        elif (
            self._attempts >= _MAX_ATTEMPTS
            or time.monotonic() - self._last_click_at < _RETRY_DELAY_SECONDS
        ):
            return False
        ox, oy = client_origin
        screen_x, screen_y = ox + int(cx), oy + int(cy)
        ok = self.input.click(screen_x, screen_y, hold_ms=self.profile.shake_click_hold_ms)
        if ok:
            if self._last_center is None:
                self._last_center = (int(cx), int(cy))
            self._attempts += 1
            self._last_click_at = time.monotonic()
        return ok

    def reset_target_memory(self) -> None:
        self._last_center = None
        self._attempts = 0
