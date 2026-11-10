# tests/test_loop_transitions.py
import threading

import cv2
import numpy as np
import pytest

from fisch_macro import loop
from fisch_macro.input_controller import InputController
from fisch_macro.loop import BotRuntime
from fisch_macro.perception.frame_bundle import FrameBundle
from fisch_macro.perception.result import detect_result
from fisch_macro.perception.shake import radius_limits_from_prior
from fisch_macro.state_machine import StateMachine
from fisch_macro.types import BotState, FramePacket, Observation, ProfileData, Rect


class FakeMouse:
    def __init__(self):
        self.events = []
        self.position = (0, 0)

    def move_to(self, x, y):
        self.position = (x, y)
        self.events.append(("move", x, y))

    def down(self):
        self.events.append(("down",))

    def up(self):
        self.events.append(("up",))


def _profile() -> ProfileData:
    return ProfileData(name="t", saved_at="2026-11-09")


def _runtime(**kwargs) -> BotRuntime:
    mouse = kwargs.pop("mouse", None)
    ctl = kwargs.pop("input_ctl", None)
    if ctl is None and mouse is not None:
        ctl = InputController(mouse=mouse, sleep_fn=lambda _s: None)
    return BotRuntime(
        state_machine=StateMachine(),
        profile=_profile(),
        input_ctl=ctl,
        dry_run=kwargs.pop("dry_run", True),
        **kwargs,
    )


def _bundle(**overrides) -> FrameBundle:
    base = dict(
        cast=Observation(False),
        shake=Observation(False),
        bite=Observation(False),
        reel=Observation(False),
        result=Observation(False),
        ready=Observation(True),
    )
    base.update(overrides)
    return FrameBundle(**base)


def _frame(image=None, timestamp_ms: int = 0) -> FramePacket:
    if image is None:
        image = np.zeros((120, 160, 3), dtype=np.uint8)
    return FramePacket(image=image, timestamp_ms=timestamp_ms, client_origin=(10, 20))


def _step(rt: BotRuntime, bundle: FrameBundle, frame: FramePacket | None = None) -> None:
    packet = _frame() if frame is None else frame
    rt.apply_bundle(bundle, operator_stop=False)
    rt.act(bundle, packet)


def test_runtime_applies_state_machine_priority():
    rt = BotRuntime(
        state_machine=StateMachine(),
        profile=ProfileData(name="t", saved_at="2026-11-09"),
        dry_run=True,
    )
    rt.state = BotState.SHAKING
    bundle = FrameBundle(
        cast=Observation(False),
        shake=Observation(True, 0.9),
        bite=Observation(False),
        reel=Observation(True, 0.9, extras={"stable": True}),
        result=Observation(False),
        ready=Observation(False),
    )
    rt.apply_bundle(bundle, operator_stop=False)
    assert rt.state in {BotState.BITE_DETECTED, BotState.REEL_INITIALIZING, BotState.REELING}


def test_start_and_stop_release_lmb():
    mouse = FakeMouse()
    rt = _runtime(mouse=mouse, dry_run=False)
    rt.start()
    assert rt.state == BotState.PRE_CAST
    assert rt.input.is_cancelled() is False
    rt.input.mouse_down()
    rt.cast._holding = True
    rt.reel._holding = True
    rt.stop()
    assert rt.state == BotState.IDLE
    assert ("up",) in mouse.events
    assert rt.cast._holding is False
    assert rt.reel._holding is False


def test_start_while_reeling_keeps_existing_hold_and_state():
    mouse = FakeMouse()
    rt = _runtime(mouse=mouse, dry_run=False)
    rt.start()
    rt.state = BotState.REELING
    rt.fail_streak = 2
    rt.input.mouse_down()
    rt.reel._holding = True
    rt.start()
    assert rt.state == BotState.REELING
    assert rt.fail_streak == 2
    assert rt.reel._holding
    assert mouse.events == [("down",)]


def test_stop_interrupts_a_long_shake_click():
    pressed = threading.Event()

    class SignalingMouse(FakeMouse):
        def down(self):
            super().down()
            pressed.set()

    mouse = SignalingMouse()
    rt = _runtime(input_ctl=InputController(mouse=mouse), dry_run=False)
    rt.profile.shake_click_hold_ms = 60_000
    rt.start()
    rt.state = BotState.SHAKING
    shake = Observation(True, 0.9, geometry=Rect(50, 50, 20, 20))
    worker = threading.Thread(target=rt.act, args=(_bundle(shake=shake), _frame()), daemon=True)
    worker.start()
    assert pressed.wait(2)
    restart = threading.Thread(target=rt.start, daemon=True)
    restart.start()
    restart.join(timeout=2)
    assert not restart.is_alive()
    stop = threading.Thread(target=rt.stop, daemon=True)
    stop.start()
    stop.join(timeout=2)
    assert not stop.is_alive()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert mouse.events[-1] == ("up",)
    assert rt.state == BotState.IDLE
    assert rt.input.is_cancelled()


def test_perceive_bite_is_reel_without_shake_and_ready_is_clear():
    rt = _runtime()
    ready_bundle = rt.perceive(_frame())
    assert ready_bundle.ready.found is True
    assert ready_bundle.bite.found is False

    image = np.full((200, 300, 3), 100, dtype=np.uint8)
    image[165:177, 65:235] = 25
    image[164:178, 100:170] = 240
    image[160:182, 135:138] = 65
    bite_bundle = rt.perceive(_frame(image))
    assert bite_bundle.reel.found is True
    assert bite_bundle.shake.found is False
    assert bite_bundle.bite.found is True
    assert bite_bundle.ready.found is False


def test_dry_run_act_releases_on_catch_and_does_not_cast():
    mouse = FakeMouse()
    rt = _runtime(mouse=mouse, dry_run=True)
    rt.start()
    rt.input.mouse_down()
    rt.cast._holding = True
    rt.reel._holding = True
    rt.state = BotState.CATCH_RESULT
    rt.act(_bundle(result=Observation(True, 0.9)), _frame())
    assert ("up",) in mouse.events
    assert rt.state == BotState.CATCH_RESULT

    mouse.events.clear()
    rt.state = BotState.PRE_CAST
    rt.act(_bundle(), _frame())
    assert mouse.events == []
    assert rt.state == BotState.PRE_CAST


def test_act_pre_cast_begins_cast_and_sets_casting():
    mouse = FakeMouse()
    rt = _runtime(mouse=mouse, dry_run=False)
    rt.start()
    rt.act(_bundle(), _frame())
    assert ("down",) in mouse.events
    assert rt.state == BotState.CASTING


def test_act_shaking_updates_profiler_radius():
    mouse = FakeMouse()
    rt = _runtime(mouse=mouse, dry_run=False)
    rt.start()
    rt.state = BotState.SHAKING
    before = rt.profiler.data.shake_radius_px_mean
    radius = before + 40
    shake = Observation(True, 0.9, extras={"radius": radius, "center": (12, 12)})
    rt.act(_bundle(shake=shake), _frame())
    assert rt.profiler.data.shake_radius_px_mean == pytest.approx(0.8 * before + 0.2 * radius)


def _reel_hold_extras(**overrides) -> Observation:
    extras = {"stable": False, "fish_x": 0.8, "bar_left": 0.2, "bar_right": 0.6, "progress": 0.4}
    extras.update(overrides)
    return Observation(True, 0.9, extras=extras)


def test_lost_reel_releases_immediately_and_recovers_after_grace(monkeypatch):
    monkeypatch.setattr(loop, "_REEL_LOSS_GRACE_FRAMES", 2)
    mouse = FakeMouse()
    rt = _runtime(mouse=mouse, dry_run=False)
    rt.start()
    rt.state = BotState.REELING
    reel = _bundle(reel=_reel_hold_extras(stable=True))
    missing = _bundle(shake=Observation(True, 0.9, geometry=Rect(20, 20, 40, 40)))

    _step(rt, reel)
    assert rt.reel._holding
    _step(rt, missing)
    assert rt.state == BotState.REELING
    assert not rt.reel._holding
    assert mouse.events[-1] == ("up",)
    _step(rt, reel)
    assert rt.reel._holding
    _step(rt, missing)
    assert rt.state == BotState.REELING
    _step(rt, missing)
    assert rt.state == BotState.RECOVERY
    assert rt.fail_streak == 0


@pytest.mark.parametrize("phase", [BotState.WAITING_SHAKE, BotState.SHAKING])
def test_stalled_shake_phases_recover(monkeypatch, phase):
    monkeypatch.setitem(loop._SHAKE_TIMEOUT_FRAMES, phase, 2)
    rt = _runtime()
    rt.start()
    rt.state = phase
    shake = Observation(phase == BotState.SHAKING)
    stalled = _bundle(shake=shake)
    assert rt.apply_bundle(stalled, operator_stop=False) == phase
    assert rt.apply_bundle(stalled, operator_stop=False) == BotState.RECOVERY


def test_act_reel_initializing_does_not_mouse_down():
    reel = _reel_hold_extras()
    for dry_run in (True, False):
        mouse = FakeMouse()
        rt = _runtime(mouse=mouse, dry_run=dry_run)
        rt.start()
        rt.state = BotState.REEL_INITIALIZING
        rt.act(_bundle(reel=reel, ready=Observation(False)), _frame())
        assert ("down",) not in mouse.events
        assert rt.reel._holding is False
        assert rt._last_reel_progress == 0.4
        assert rt.state == BotState.REEL_INITIALIZING


def test_missing_window_does_not_burn_fail_streak():
    mouse = FakeMouse()
    rt = _runtime(mouse=mouse, dry_run=False)
    rt.start()
    rt.input.mouse_down()
    rt.fail_streak = 2
    misses = rt.max_fail_streak + 3
    for _ in range(misses):
        assert rt.enter_recovery() is False
        assert rt.state == BotState.RECOVERY
        assert rt._running is True
    assert rt.fail_streak == 2
    assert rt.missing_window_ticks == misses
    assert ("up",) in mouse.events
    rt.note_window_present()
    assert rt.missing_window_ticks == 0
    assert rt.fail_streak == 2


def test_missing_window_uses_separate_long_timeout():
    rt = _runtime(dry_run=False)
    rt.start()
    rt.max_missing_window_ticks = 3
    assert rt.enter_recovery() is False
    assert rt.enter_recovery() is False
    assert rt.enter_recovery() is True
    assert rt.fail_streak == 0
    assert rt.state == BotState.IDLE
    assert rt._running is False


def test_release_after_error_releases_without_stopping():
    mouse = FakeMouse()
    rt = _runtime(mouse=mouse, dry_run=False)
    rt.start()
    rt.input.mouse_down()
    rt.reel._holding = True
    rt.cast._holding = True
    rt.release_after_error()
    assert ("up",) in mouse.events
    assert rt.reel._holding is False
    assert rt.cast._holding is False
    assert rt._running is True
    assert rt.state == BotState.PRE_CAST


def test_act_reeling_invokes_reel_with_progress_slope():
    mouse = FakeMouse()
    rt = _runtime(mouse=mouse, dry_run=False)
    rt.start()
    rt.state = BotState.REELING
    reel = Observation(
        True,
        0.9,
        extras={"stable": True, "fish_x": 0.8, "bar_left": 0.2, "bar_right": 0.6, "progress": 0.5},
    )
    rt._last_reel_progress = 0.2
    rt.act(_bundle(reel=reel, ready=Observation(False)), _frame())
    assert rt._last_reel_progress == 0.5
    assert rt.reel._holding is True


def test_cast_hold_continues_until_fill_threshold():
    mouse = FakeMouse()
    rt = _runtime(mouse=mouse, dry_run=False)
    rt.start()
    _step(rt, _bundle(), _frame(timestamp_ms=0))
    assert ("move", 90, 74) in mouse.events
    assert ("down",) in mouse.events
    assert rt.state == BotState.CASTING
    assert rt.cast._holding is True
    mouse.events.clear()

    _step(rt, _bundle(), _frame(timestamp_ms=40))
    assert rt.state == BotState.CASTING
    assert rt.cast._holding is True
    assert ("up",) not in mouse.events

    for timestamp_ms, fill in ((80, 0.30), (120, 0.60)):
        cast = Observation(True, 0.85, extras={"fill_ratio": fill})
        _step(
            rt,
            _bundle(cast=cast, ready=Observation(False)),
            _frame(timestamp_ms=timestamp_ms),
        )
        assert rt.state == BotState.CASTING
        assert rt.cast._holding is True
        assert ("up",) not in mouse.events

    cast = Observation(True, 0.95, extras={"fill_ratio": 0.95})
    _step(rt, _bundle(cast=cast, ready=Observation(False)), _frame(timestamp_ms=160))
    assert ("up",) in mouse.events
    assert rt.cast._holding is False


def test_perceived_cast_holds_until_the_vertical_meter_fills():
    mouse = FakeMouse()
    rt = _runtime(mouse=mouse, dry_run=False)
    rt.start()
    image = np.full((300, 400, 3), 20, dtype=np.uint8)
    # Scenery and fading rings must not turn the initial cast into short clicks.
    cv2.circle(image, (80, 220), 60, (220, 220, 220), 3)

    frame = _frame(image, timestamp_ms=0)
    _step(rt, rt.perceive(frame), frame)
    assert rt.state == BotState.CASTING
    assert rt.input.is_held()
    assert mouse.events == [("move", 210, 155), ("down",)]

    image[40:52, 290:310] = (0, 220, 0)
    for timestamp, fill in ((100, 0.2), (200, 0.4), (300, 0.6), (350, 0.74)):
        charging = image.copy()
        charging[240 - round(200 * fill) : 240, 290:310] = 240
        frame = _frame(charging, timestamp_ms=timestamp)
        bundle = rt.perceive(frame)
        assert bundle.cast.extras["fill_ratio"] == pytest.approx(fill)
        _step(rt, bundle, frame)
        assert rt.input.is_held()
        assert mouse.events == [("move", 210, 155), ("down",)]

    charged = image.copy()
    charged[90:240, 290:310] = 240
    frame = _frame(charged, timestamp_ms=400)
    bundle = rt.perceive(frame)
    assert bundle.cast.extras["fill_ratio"] == pytest.approx(0.75)
    _step(rt, bundle, frame)
    assert not rt.input.is_held()
    assert mouse.events == [("move", 210, 155), ("down",), ("up",)]


def test_visible_meter_on_pre_cast_still_presses():
    mouse = FakeMouse()
    rt = _runtime(mouse=mouse, dry_run=False)
    rt.start()
    cast = Observation(True, 0.8, extras={"fill_ratio": 0.1})
    _step(rt, _bundle(cast=cast, ready=Observation(False)), _frame(timestamp_ms=0))
    assert ("down",) in mouse.events
    assert rt.cast._holding is True
    assert rt.state == BotState.CASTING


def test_shake_does_not_release_an_active_cast():
    mouse = FakeMouse()
    rt = _runtime(mouse=mouse, dry_run=False)
    rt.start()
    _step(rt, _bundle(), _frame(timestamp_ms=0))
    mouse.events.clear()
    shake = Observation(True, 0.9, extras={"radius": 40, "center": (12, 12)})
    _step(rt, _bundle(shake=shake, ready=Observation(False)), _frame(timestamp_ms=25))
    assert rt.state == BotState.CASTING
    assert rt.cast._holding is True
    assert ("up",) not in mouse.events


def test_bite_releases_cast_hold():
    mouse = FakeMouse()
    rt = _runtime(mouse=mouse, dry_run=False)
    rt.start()
    _step(rt, _bundle(), _frame(timestamp_ms=0))
    reel = Observation(
        True,
        0.9,
        extras={"stable": False, "fish_x": 0.5, "bar_left": 0.2, "bar_right": 0.6, "progress": 0.1},
    )
    _step(rt, _bundle(reel=reel, ready=Observation(False)), _frame(timestamp_ms=40))
    assert rt.cast._holding is False
    assert rt.state == BotState.BITE_DETECTED
    assert ("up",) in mouse.events


def test_missing_meter_releases_on_cast_hold_prior():
    mouse = FakeMouse()
    rt = _runtime(mouse=mouse, dry_run=False)
    rt.start()
    prior = int(rt.profile.cast_hold_ms_prior)
    _step(rt, _bundle(), _frame(timestamp_ms=0))
    _step(rt, _bundle(), _frame(timestamp_ms=prior - 1))
    assert rt.cast._holding is True
    assert ("up",) not in mouse.events
    _step(rt, _bundle(), _frame(timestamp_ms=prior))
    assert rt.cast._holding is False
    assert ("up",) in mouse.events


def test_fail_streak_counts_explicit_fails_only():
    rt = _runtime(dry_run=True)
    rt.start()
    fail = _bundle(
        result=Observation(True, 0.8, extras={"outcome": "fail", "red_ratio": 0.8}),
        ready=Observation(False),
    )
    for i in range(rt.max_fail_streak - 1):
        rt.state = BotState.PRE_CAST
        rt.apply_bundle(fail, operator_stop=False)
        assert rt.fail_streak == i + 1
        assert rt.state == BotState.CATCH_RESULT
        assert rt._running is True
    rt.apply_bundle(fail, operator_stop=False)
    assert rt.fail_streak == rt.max_fail_streak - 1
    rt.state = BotState.PRE_CAST
    rt.apply_bundle(fail, operator_stop=False)
    assert rt.fail_streak == rt.max_fail_streak
    assert rt._running is False

    rt = _runtime(dry_run=True)
    rt.start()
    rt.fail_streak = 2
    unknown = _bundle(
        result=Observation(True, 0.9, extras={"outcome": "unknown"}),
        ready=Observation(False),
    )
    rt.state = BotState.REELING
    rt.apply_bundle(unknown, operator_stop=False)
    assert rt.fail_streak == 2
    assert rt._running is True
    success = _bundle(
        result=Observation(True, 0.9, extras={"outcome": "success"}),
        ready=Observation(False),
    )
    rt.state = BotState.REELING
    rt.apply_bundle(success, operator_stop=False)
    assert rt.fail_streak == 0
    assert rt.state == BotState.CATCH_RESULT


def test_catch_result_timeout_unsticks_without_ready():
    rt = _runtime(dry_run=True)
    rt.start()
    rt.catch_result_timeout_frames = 3
    result = Observation(True, 0.9, extras={"outcome": "success", "bright_ratio": 0.9})
    bundle = _bundle(result=result, ready=Observation(False))
    assert rt.apply_bundle(bundle, operator_stop=False) == BotState.CATCH_RESULT
    assert rt.apply_bundle(bundle, operator_stop=False) == BotState.CATCH_RESULT
    assert rt.apply_bundle(bundle, operator_stop=False) == BotState.PRE_CAST
    assert rt.apply_bundle(bundle, operator_stop=False) == BotState.PRE_CAST


def test_result_requires_consecutive_confident_frames():
    rt = _runtime(dry_run=True)
    rt.start()
    strong = Observation(True, 0.9, extras={"outcome": "success", "bright_ratio": 0.9})
    assert rt._gate_result(strong).found is False
    assert rt._gate_result(strong).found is False
    assert rt._gate_result(strong).found is True
    weak = Observation(True, 0.1, extras={"outcome": "unknown", "bright_ratio": 0.1})
    rt._result_hit_streak = 0
    assert rt._gate_result(weak).found is False
    assert rt._result_hit_streak == 0


def test_detect_result_rejects_weak_brightness_and_marks_red_fail():
    weak = np.zeros((100, 100, 3), dtype=np.uint8)
    weak[25:28, 25:75] = 255
    obs = detect_result(weak)
    assert obs.found is False
    assert obs.extras["outcome"] == "unknown"

    fail = np.zeros((100, 100, 3), dtype=np.uint8)
    fail[25:55, 25:75] = (0, 0, 220)
    obs = detect_result(fail)
    assert obs.found is True
    assert obs.extras["outcome"] == "fail"

    success = np.zeros((100, 100, 3), dtype=np.uint8)
    success[25:55, 25:75] = (240, 240, 240)
    obs = detect_result(success)
    assert obs.found is True
    assert obs.extras["outcome"] == "success"


def test_reel_moves_to_cast_safe_before_first_mouse_down():
    mouse = FakeMouse()
    rt = _runtime(mouse=mouse, dry_run=False)
    rt.start()
    rt.cast.safe_xy = (50, 60)
    rt.state = BotState.REELING
    reel = Observation(
        True,
        0.9,
        extras={"stable": True, "fish_x": 0.8, "bar_left": 0.2, "bar_right": 0.6, "progress": 0.4},
    )
    rt.act(_bundle(reel=reel, ready=Observation(False)), _frame())
    assert mouse.events[0] == ("move", 60, 80)
    assert mouse.events[1] == ("down",)
    mouse.events.clear()
    rt.act(_bundle(reel=reel, ready=Observation(False)), _frame())
    assert mouse.events == []


def test_reel_gets_volatility_estimated_from_fish_deltas():
    mouse = FakeMouse()
    rt = _runtime(mouse=mouse, dry_run=False)
    rt.start()
    rt.profile.fish_volatility = 0.1
    rt.state = BotState.REELING
    seen: list[float] = []
    real = rt.reel.tick

    def wrapped(*args, **kwargs):
        seen.append(kwargs["fish_volatility"])
        return real(*args, **kwargs)

    rt.reel.tick = wrapped

    def reel(fish_x: float, progress: float) -> Observation:
        return Observation(
            True,
            0.9,
            extras={
                "stable": True,
                "fish_x": fish_x,
                "bar_left": 0.2,
                "bar_right": 0.6,
                "progress": progress,
            },
        )

    rt.act(_bundle(reel=reel(0.20, 0.4), ready=Observation(False)), _frame())
    rt.act(_bundle(reel=reel(0.45, 0.5), ready=Observation(False)), _frame())
    assert seen == pytest.approx([0.1, 0.28])
    assert rt.profile.fish_volatility == pytest.approx(0.28)


def test_perceive_uses_shake_radius_prior(monkeypatch):
    captured: dict[str, int] = {}

    def fake(image, min_radius=20, max_radius=160):
        captured["min"] = min_radius
        captured["max"] = max_radius
        return Observation(False)

    monkeypatch.setattr("fisch_macro.loop.detect_shake", fake)
    rt = _runtime()
    rt.profile.shake_radius_px_mean = 100
    rt.perceive(_frame())
    assert (captured["min"], captured["max"]) == radius_limits_from_prior(100)


def test_load_profile_rebinds_shared_profile(tmp_path):
    rt = _runtime()
    rt.profile.name = "night"
    rt.profile.fish_volatility = 0.15
    path = tmp_path / "night.json"
    rt.profiler.save(path)
    rt.profile.fish_volatility = 0.9
    loaded = rt.load_profile(path)
    assert loaded is rt.profile
    assert rt.profile is rt.profiler.data is rt.cast.profile is rt.shake.profile is rt.reel.profile
    assert rt.profile.fish_volatility == pytest.approx(0.15)
