# tests/test_loop_transitions.py
import numpy as np

from fisch_macro.input_controller import InputController
from fisch_macro.loop import BotRuntime
from fisch_macro.perception.frame_bundle import FrameBundle
from fisch_macro.state_machine import StateMachine
from fisch_macro.types import BotState, FramePacket, Observation, ProfileData


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


def _frame(image=None) -> FramePacket:
    if image is None:
        image = np.zeros((120, 160, 3), dtype=np.uint8)
    return FramePacket(image=image, timestamp_ms=0, client_origin=(10, 20))


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


def test_perceive_bite_is_reel_without_shake_and_ready_is_clear():
    rt = _runtime()
    ready_bundle = rt.perceive(_frame())
    assert ready_bundle.ready.found is True
    assert ready_bundle.bite.found is False

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[80:90, 30:70] = (255, 255, 255)
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
    shake = Observation(True, 0.9, extras={"radius": before + 40, "center": (12, 12)})
    rt.act(_bundle(shake=shake), _frame())
    assert rt.profiler.data.shake_radius_px_mean != before


def _reel_hold_extras(**overrides) -> Observation:
    extras = {"stable": False, "fish_x": 0.8, "bar_left": 0.2, "bar_right": 0.6, "progress": 0.4}
    extras.update(overrides)
    return Observation(True, 0.9, extras=extras)


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


def test_enter_recovery_counts_each_missing_window_iteration():
    mouse = FakeMouse()
    rt = _runtime(mouse=mouse, dry_run=False)
    rt.start()
    rt.input.mouse_down()
    for i in range(rt.max_fail_streak - 1):
        assert rt.enter_recovery() is False
        assert rt.fail_streak == i + 1
        assert rt.state == BotState.RECOVERY
        assert rt._running is True
    assert ("up",) in mouse.events
    assert rt.enter_recovery() is True
    assert rt.fail_streak == rt.max_fail_streak
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
