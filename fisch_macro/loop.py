"""Main fishing loop: perception, transitions, and stage actions."""

from __future__ import annotations

import threading

from fisch_macro.controllers.cast import CastController
from fisch_macro.controllers.recovery import RecoveryController
from fisch_macro.controllers.reel import ReelController
from fisch_macro.controllers.shake import ShakeController
from fisch_macro.input_controller import InputController
from fisch_macro.perception.cast_meter import detect_cast_meter
from fisch_macro.perception.frame_bundle import FrameBundle
from fisch_macro.perception.reel import detect_reel
from fisch_macro.perception.result import detect_result
from fisch_macro.perception.shake import detect_shake
from fisch_macro.profiler import Profiler
from fisch_macro.state_machine import StateMachine
from fisch_macro.types import BotState, FramePacket, Observation, ProfileData


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
        self._last_reel_progress: float | None = None
        self._running = False
        self._accepting = True
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            self.input.clear_cancel()
            self._running = True
            self._accepting = True
            self.fail_streak = 0
            self._last_reel_progress = None
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
        shake = detect_shake(image)
        reel = detect_reel(image)
        result = detect_result(image)
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

    def apply_bundle(self, bundle: FrameBundle, operator_stop: bool) -> BotState:
        with self._lock:
            if not self._accepting:
                self.state = BotState.IDLE
                return self.state
            nxt = self.state_machine.transition(self.state, bundle, operator_stop)
            if nxt == BotState.CATCH_RESULT and self.state != BotState.CATCH_RESULT:
                self.fail_streak = 0
            self.state = nxt
            return self.state

    def enter_recovery(self) -> bool:
        """Release inputs and sit in RECOVERY. Stop once fail_streak hits the cap."""
        with self._lock:
            if not self._accepting:
                return False
            fresh = self.state != BotState.RECOVERY
            self._release_buttons()
            self.state = BotState.RECOVERY
            if fresh:
                self.fail_streak += 1
            tripped = self.fail_streak >= self.max_fail_streak
        if tripped:
            self.stop()
            return True
        return False

    def act(self, bundle: FrameBundle, frame: FramePacket) -> None:
        with self._lock:
            if self.state == BotState.CATCH_RESULT:
                self.reel.force_release()
                self.cast.force_release()
            elif self.state == BotState.RECOVERY:
                self._release_buttons()
            if self.state not in {BotState.REEL_INITIALIZING, BotState.REELING}:
                self._last_reel_progress = None
            if self.dry_run or not self._running:
                return
            self._act_live(bundle, frame)

    def _act_live(self, bundle: FrameBundle, frame: FramePacket) -> None:
        origin = frame.client_origin
        if self.state == BotState.PRE_CAST:
            height, width = frame.image.shape[:2]
            self.cast.safe_xy = (max(1, width // 2), max(1, int(height * 0.45)))
            self.cast.begin_cast(origin)
            self.state = BotState.CASTING
        elif self.state == BotState.CASTING:
            fill = float(bundle.cast.extras.get("fill_ratio", 0.0))
            self.cast.update_fill(fill)
        elif self.state in {BotState.WAITING_SHAKE, BotState.SHAKING}:
            if self.cast._holding:
                self.cast.force_release()
            self.shake.tick(
                bundle.shake,
                bundle.bite.found or bundle.reel.found,
                origin,
            )
            self._note_shake_radius(bundle)
        elif self.state == BotState.BITE_DETECTED:
            self.shake.tick(bundle.shake, True, origin)
            if self.cast._holding:
                self.cast.force_release()
        elif self.state in {BotState.REEL_INITIALIZING, BotState.REELING}:
            self._tick_reel(bundle)

    def _tick_reel(self, bundle: FrameBundle) -> None:
        extras = bundle.reel.extras
        progress = float(extras.get("progress", 0.0))
        if self._last_reel_progress is None:
            slope = 0.0
        else:
            slope = progress - self._last_reel_progress
        self._last_reel_progress = progress
        self.reel.tick(
            fish_x=float(extras.get("fish_x", 0.5)),
            bar_left=float(extras.get("bar_left", 0.2)),
            bar_right=float(extras.get("bar_right", 0.6)),
            progress=progress,
            progress_slope=slope,
            fish_volatility=float(self.profile.fish_volatility),
        )
        if "bar_left" in extras and "bar_right" in extras:
            width = float(extras["bar_right"]) - float(extras["bar_left"])
            if width > 0:
                self.profiler.update_bar_width_ratio(width)

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
