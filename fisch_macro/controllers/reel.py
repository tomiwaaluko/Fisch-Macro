from __future__ import annotations

from fisch_macro.input_controller import InputController
from fisch_macro.types import ProfileData


class ReelController:
    def __init__(self, profile: ProfileData, input_ctl: InputController | None = None) -> None:
        self.profile = profile
        self.input = input_ctl
        self._holding = False

    def select_mode(self, progress: float, progress_slope: float, fish_volatility: float) -> str:
        if progress_slope < 0 or fish_volatility > 0.85:
            return "emergency"
        if progress >= 0.35 and progress_slope > 0.02 and fish_volatility < 0.55:
            return "tighten"
        return "safe"

    def decide_hold(
        self,
        fish_x: float,
        bar_left: float,
        bar_right: float,
        progress: float,
        progress_slope: float,
        fish_volatility: float,
    ) -> bool:
        mode = self.select_mode(progress, progress_slope, fish_volatility)
        width = max(1e-6, bar_right - bar_left)
        eps = 1e-9
        center = (bar_left + bar_right) / 2.0
        if mode == "tighten":
            margin = self.profile.reel_tighten_margin_ratio * width
        elif mode == "emergency":
            margin = self.profile.reel_safe_margin_ratio * width * 1.3
        else:
            margin = self.profile.reel_safe_margin_ratio * width

        if fish_x + eps >= center + margin:
            return True
        if fish_x <= center - margin + eps:
            return False
        return self._holding

    def tick(
        self,
        fish_x: float,
        bar_left: float,
        bar_right: float,
        progress: float,
        progress_slope: float,
        fish_volatility: float,
        safe_screen_xy: tuple[int, int] | None = None,
    ) -> str:
        want_hold = self.decide_hold(
            fish_x, bar_left, bar_right, progress, progress_slope, fish_volatility
        )
        mode = self.select_mode(progress, progress_slope, fish_volatility)
        if self.input is None:
            self._holding = want_hold
            return mode
        if want_hold and not self._holding:
            if safe_screen_xy is not None:
                if not self.input.move_to(int(safe_screen_xy[0]), int(safe_screen_xy[1])):
                    return mode
            if not self.input.mouse_down():
                return mode
            self._holding = True
        elif not want_hold and self._holding:
            self.input.mouse_up()
            self._holding = False
        return mode

    def force_release(self) -> None:
        if self.input is not None:
            self.input.mouse_up()
        self._holding = False

