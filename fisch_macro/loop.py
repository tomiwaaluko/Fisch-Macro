"""Main fishing loop: perception, transitions, and stage actions."""

from __future__ import annotations

import threading
from pathlib import Path

from fisch_macro.controllers.cast import CastController
from fisch_macro.controllers.recovery import RecoveryController
from fisch_macro.controllers.reel import ReelController
from fisch_macro.controllers.shake import ShakeController
from fisch_macro.input_controller import InputController
from fisch_macro.perception.cast_meter import detect_cast_meter
from fisch_macro.perception.frame_bundle import FrameBundle
from fisch_macro.perception.reel import detect_reel
from fisch_macro.perception.result import detect_result
from fisch_macro.perception.shake import detect_shake, radius_limits_from_prior
from fisch_macro.profiler import Profiler
from fisch_macro.state_machine import StateMachine
from fisch_macro.types import BotState, FramePacket, Observation, ProfileData

# Per-frame |delta fish_x| that maps to volatility 1 before the profiler EMA.
_FULL_FISH_VOLATILITY_DELTA = 0.25
_CAST_OWNED = {BotState.CASTING, BotState.WAITING_SHAKE, BotState.SHAKING, BotState.PRE_CAST}


def _blank_result(result: Observation) -> Observation:
    return Observation(
        found=False,
        confidence=result.confidence,
        geometry=result.geometry,
        age_ms=result.age_ms,
        extras=dict(result.extras),
    )


class _NullMouse:
    """Inert pointer used when no InputController is injected (unit tests)."""

    position = (0, 0)

    def move_to(self, x: int, y: int) -> None:
        self.position = (int(x), int(y))

    def down(self) -> None:
        return None

    def up(self) -> None:
        return None


class BotRuntime:
    def __init__(
        self,
        state_machine: StateMachine,
        profile: ProfileData,
        input_ctl: InputController | None = None,
        dry_run: bool = False,
    ) -> None:
        self.state_machine = state_machine
        self.profile = profile
        self.dry_run = dry_run
        self.input = input_ctl if input_ctl is not None else InputController(
            mouse=_NullMouse(),
            sleep_fn=lambda _seconds: None,
        )
        self.profiler = Profiler(profile)
        self.cast = CastController(self.input, profile, safe_xy=(400, 300))
        self.shake = ShakeController(self.input, profile)
        self.reel = ReelController(profile, self.input)
        self.recovery = RecoveryController(self.input)
        self.state = BotState.IDLE
        self.fail_streak = 0
        self.max_fail_streak = 5
        # Missing the Roblox window is not a catch failure. ~30s at 33 Hz.
        self.missing_window_ticks = 0
        self.max_missing_window_ticks = 990
        self.result_min_confidence = 0.25
        self.result_confirm_frames = 3
        self.catch_result_timeout_frames = 8 * 33
        self._result_hit_streak = 0
        self._suppress_result = False
        self._catch_result_frames = 0
        self._last_reel_progress: float | None = None
        self._last_fish_x: float | None = None
        self._running = False
        self._accepting = True
        self._lock = threading.Lock()

    def bind_profile(self, profile: ProfileData) -> None:
        """Point the profiler and every controller at one ProfileData object."""
        self.profile = profile
        self.profiler.data = profile
        self.cast.profile = profile
        self.shake.profile = profile
        self.reel.profile = profile

    def load_profile(self, path: str | Path) -> ProfileData:
        data = Profiler.load(path).data
        self.bind_profile(data)
        return data

    def start(self) -> None:
        with self._lock:
            self.input.clear_cancel()
            self._running = True
            self._accepting = True
            self.fail_streak = 0
            self.missing_window_ticks = 0
            self._result_hit_streak = 0
            self._suppress_result = False
            self._catch_result_frames = 0
            self._last_reel_progress = None
            self._last_fish_x = None
            self.state = BotState.PRE_CAST

    def stop(self) -> None:
        """Flag cancel, then release LMB. cancel() does not release by itself."""
        self._running = False
        self._accepting = False
        self.input.cancel()
        with self._lock:
            self.recovery.run()
            self.cast.force_release()
            self.reel.force_release()
            self.shake.reset_target_memory()
            self.state = BotState.IDLE

    def perceive(self, frame: FramePacket) -> FrameBundle:
        image = frame.image
        cast = detect_cast_meter(image)
        min_radius, max_radius = radius_limits_from_prior(self.profile.shake_radius_px_mean)
        shake = detect_shake(image, min_radius=min_radius, max_radius=max_radius)
        reel = detect_reel(image)
        result = self._gate_result(detect_result(image))
        bite_found = reel.found and not shake.found
        ready_found = not (cast.found or shake.found or reel.found or result.found)
        return FrameBundle(
            cast=cast,
            shake=shake,
            bite=Observation(found=bite_found, confidence=reel.confidence if bite_found else 0.0),
            reel=reel,
            result=result,
            ready=Observation(found=ready_found, confidence=1.0 if ready_found else 0.0),
        )

    def _gate_result(self, result: Observation) -> Observation:
        """Require a confident multi-frame result, and drop a stuck popup."""
        if self._suppress_result:
            if not result.found:
                self._suppress_result = False
                self._result_hit_streak = 0
            else:
                return _blank_result(result)
        strong = result.found and (
            float(result.confidence) >= self.result_min_confidence
            or result.extras.get("outcome") == "fail"
        )
        if strong:
            self._result_hit_streak += 1
        else:
            self._result_hit_streak = 0
        if result.found and self._result_hit_streak < self.result_confirm_frames:
            return _blank_result(result)
        return result

    def apply_bundle(self, bundle: FrameBundle, operator_stop: bool) -> BotState:
        tripped = False
        with self._lock:
            if not self._accepting:
                self.state = BotState.IDLE
                return self.state
            if self._suppress_result and bundle.result.found:
                bundle = FrameBundle(
                    cast=bundle.cast,
                    shake=bundle.shake,
                    bite=bundle.bite,
                    reel=bundle.reel,
                    result=_blank_result(bundle.result),
                    ready=bundle.ready,
                )
            nxt = self.state_machine.transition(
                self.state,
                bundle,
                operator_stop,
                cast_holding=self.cast._holding,
            )
            if nxt == BotState.CATCH_RESULT and self.state != BotState.CATCH_RESULT:
                outcome = bundle.result.extras.get("outcome")
                if outcome == "fail":
                    self.fail_streak += 1
                    tripped = self.fail_streak >= self.max_fail_streak
                elif outcome == "success":
                    self.fail_streak = 0
            if nxt == BotState.CATCH_RESULT:
                if self.state != BotState.CATCH_RESULT:
                    self._catch_result_frames = 1
                else:
                    self._catch_result_frames += 1
                if self._catch_result_frames >= self.catch_result_timeout_frames:
                    nxt = BotState.PRE_CAST
                    self._suppress_result = True
                    self._result_hit_streak = 0
                    self._catch_result_frames = 0
            else:
                self._catch_result_frames = 0
            self.state = nxt
        if tripped:
            self.stop()
        return self.state

    def enter_recovery(self) -> bool:
        """Release inputs and sit in RECOVERY when the window is missing.

        Does not consume fail_streak. Consecutive misses use a separate long
        timeout. note_window_present() clears that counter when the client returns.
        """
        with self._lock:
            if not self._accepting:
                return False
            self._release_buttons()
            self.state = BotState.RECOVERY
            self.missing_window_ticks += 1
            tripped = self.missing_window_ticks >= self.max_missing_window_ticks
        if tripped:
            self.stop()
            return True
        return False

    def note_window_present(self) -> None:
        """Reset the missing-window streak. Catch-fail budget is unchanged."""
        with self._lock:
            self.missing_window_ticks = 0

    def release_after_error(self) -> None:
        """Drop held buttons after a worker exception so the loop can continue."""
        try:
            self.recovery.run()
            self.cast.force_release()
            self.reel.force_release()
        except Exception:
            try:
                self.input.release_all()
            except Exception:
                return

    def act(self, bundle: FrameBundle, frame: FramePacket) -> None:
        with self._lock:
            if self.state == BotState.CATCH_RESULT:
                self.reel.force_release()
                self.cast.force_release()
            elif self.state == BotState.RECOVERY:
                self._release_buttons()
            if self.state == BotState.REEL_INITIALIZING:
                self._observe_reel_progress(bundle)
            elif self.state != BotState.REELING:
                self._last_reel_progress = None
                self._last_fish_x = None
            if self.dry_run or not self._running:
                return
            self._act_live(bundle, frame)

    def _act_live(self, bundle: FrameBundle, frame: FramePacket) -> None:
        if self.state == BotState.PRE_CAST and not self.cast._holding:
            self._begin_cast(frame)
            return
        if self.cast._holding and self.state in _CAST_OWNED:
            self._continue_cast(bundle, frame)
            if self.cast._holding:
                self.state = BotState.CASTING
            return
        if self.state == BotState.CASTING:
            self._continue_cast(bundle, frame)
            return
        if self.state in {BotState.WAITING_SHAKE, BotState.SHAKING}:
            self.shake.tick(
                bundle.shake,
                bundle.bite.found or bundle.reel.found,
                frame.client_origin,
            )
            self._note_shake_radius(bundle)
        elif self.state == BotState.BITE_DETECTED:
            if self.cast._holding:
                self.cast.force_release()
            self.shake.tick(bundle.shake, True, frame.client_origin)
        elif self.state == BotState.REEL_INITIALIZING:
            if self.cast._holding:
                self.cast.force_release()
        elif self.state == BotState.REELING:
            if self.cast._holding:
                self.cast.force_release()
            self._tick_reel(bundle, frame)

    def _begin_cast(self, frame: FramePacket) -> None:
        height, width = frame.image.shape[:2]
        self.cast.safe_xy = (max(1, width // 2), max(1, int(height * 0.45)))
        if self.cast.begin_cast(frame.client_origin, now_ms=frame.timestamp_ms):
            self.state = BotState.CASTING

    def _continue_cast(self, bundle: FrameBundle, frame: FramePacket) -> None:
        fill = float(bundle.cast.extras.get("fill_ratio", 0.0)) if bundle.cast.found else 0.0
        self.cast.update_fill(
            fill,
            now_ms=frame.timestamp_ms,
            meter_found=bundle.cast.found,
        )

    def _observe_reel_progress(self, bundle: FrameBundle) -> None:
        """Remember reel progress during init. Do not hold or release."""
        self._last_reel_progress = float(bundle.reel.extras.get("progress", 0.0))

    def _tick_reel(self, bundle: FrameBundle, frame: FramePacket) -> None:
        extras = bundle.reel.extras
        progress = float(extras.get("progress", 0.0))
        if self._last_reel_progress is None:
            slope = 0.0
        else:
            slope = progress - self._last_reel_progress
        self._last_reel_progress = progress
        fish_x = float(extras.get("fish_x", 0.5))
        volatility = self._fish_volatility(fish_x)
        ox, oy = frame.client_origin
        sx, sy = self.cast.safe_xy
        self.reel.tick(
            fish_x=fish_x,
            bar_left=float(extras.get("bar_left", 0.2)),
            bar_right=float(extras.get("bar_right", 0.6)),
            progress=progress,
            progress_slope=slope,
            fish_volatility=volatility,
            safe_screen_xy=(int(ox) + int(sx), int(oy) + int(sy)),
        )
        if "bar_left" in extras and "bar_right" in extras:
            width = float(extras["bar_right"]) - float(extras["bar_left"])
            if width > 0:
                self.profiler.update_bar_width_ratio(width)

    def _fish_volatility(self, fish_x: float) -> float:
        previous = self._last_fish_x
        self._last_fish_x = fish_x
        if previous is None:
            return float(self.profile.fish_volatility)
        instant = min(1.0, abs(fish_x - previous) / _FULL_FISH_VOLATILITY_DELTA)
        self.profiler.update_fish_volatility(instant)
        return float(self.profile.fish_volatility)

    def _note_shake_radius(self, bundle: FrameBundle) -> None:
        if not bundle.shake.found:
            return
        radius = bundle.shake.extras.get("radius")
        if radius is None:
            return
        self.profiler.update_shake_radius(float(radius))

    def _release_buttons(self) -> None:
        self.input.release_all()
        self.cast.force_release()
        self.reel.force_release()
