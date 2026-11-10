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
        self._hold_started_ms: int | None = None

    def begin_cast(self, client_origin: tuple[int, int], now_ms: int = 0) -> bool:
        ox, oy = client_origin
        x, y = self.safe_xy
        if not self.input.move_to(ox + x, oy + y):
            return False
        ok = self.input.mouse_down()
        self._holding = ok
        self._hold_started_ms = int(now_ms) if ok else None
        return ok

    def release_threshold(self) -> float:
        """Release early by the fraction of the hold that input latency consumes."""
        prior = max(float(self.profile.cast_hold_ms_prior), 1.0)
        lead = min(0.35, max(0.0, float(self.profile.cast_latency_ms) / prior))
        return max(0.5, self.release_fill - lead)

    def update_fill(
        self,
        fill_ratio: float,
        now_ms: int | None = None,
        meter_found: bool = True,
    ) -> bool:
        """Keep the button down across a missing meter.

        A visible meter releases on the latency-adjusted fill. A missing meter
        releases only after cast_hold_ms_prior. A meter that never reaches the
        target releases after twice that prior.
        """
        if not self._holding:
            return False
        elapsed = self._elapsed_ms(now_ms)
        if meter_found and fill_ratio >= self.release_threshold():
            self.force_release()
            return True
        if elapsed is None:
            return False
        if not meter_found and elapsed >= float(self.profile.cast_hold_ms_prior):
            self.force_release()
            return True
        if elapsed >= self._max_hold_ms():
            self.force_release()
            return True
        return False

    def force_release(self) -> None:
        self.input.mouse_up()
        self._holding = False
        self._hold_started_ms = None

    def _elapsed_ms(self, now_ms: int | None) -> float | None:
        if now_ms is None or self._hold_started_ms is None:
            return None
        return float(int(now_ms) - self._hold_started_ms)

    def _max_hold_ms(self) -> float:
        prior = float(self.profile.cast_hold_ms_prior)
        return max(prior * 2.0, prior + 1.0)
