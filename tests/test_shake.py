# tests/test_shake.py
import numpy as np

from fisch_macro.controllers import shake as shake_controller
from fisch_macro.controllers.shake import ShakeController
from fisch_macro.input_controller import InputController
from fisch_macro.perception.shake import detect_shake
from fisch_macro.types import Observation, ProfileData, Rect


def _synthetic_shake(h=200, w=300, radius=40, center=(150, 100)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2
    img[mask] = (40, 40, 40)
    ring = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= (radius + 3) ** 2) & ~mask
    img[ring] = (220, 220, 220)
    return img


def test_detect_shake_finds_circle():
    img = _synthetic_shake()
    obs = detect_shake(img)
    assert obs.found
    assert obs.geometry is not None
    assert obs.extras["radius"] >= 30


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


def test_shake_controller_clicks_once_and_stops_on_bite():
    mouse = FakeMouse()
    inp = InputController(mouse=mouse, sleep_fn=lambda _s: None)
    ctl = ShakeController(inp, ProfileData(name="t", saved_at="2026-11-09"))
    shake = Observation(True, 0.9, Rect(130, 80, 40, 40), extras={"radius": 40})
    acted = ctl.tick(shake=shake, bite_or_reel=False, client_origin=(0, 0))
    assert acted is True
    assert ("down",) in mouse.events
    # Prior shake click left the button held; bite must release only.
    inp.mouse_down()
    mouse.events.clear()
    acted2 = ctl.tick(shake=shake, bite_or_reel=True, client_origin=(0, 0))
    assert acted2 is False
    assert mouse.events == [("up",)]
    assert not any(e[0] == "down" for e in mouse.events)
    assert not any(e[0] == "move" for e in mouse.events)


def test_shake_retries_once_and_rearms_when_target_changes(monkeypatch):
    now = 0.0
    monkeypatch.setattr(shake_controller.time, "monotonic", lambda: now)
    inp = InputController(mouse=FakeMouse(), sleep_fn=lambda _s: None)
    ctl = ShakeController(inp, ProfileData(name="t", saved_at="2026-11-09"))
    target = Observation(True, geometry=Rect(10, 20, 40, 40))
    moved = Observation(True, geometry=Rect(30, 20, 40, 40))

    assert ctl.tick(target, False, (0, 0))
    assert not ctl.tick(target, False, (0, 0))
    now += 1.0
    assert ctl.tick(target, False, (0, 0))
    now += 1.0
    assert not ctl.tick(target, False, (0, 0))
    assert ctl.tick(moved, False, (0, 0))
    assert not ctl.tick(Observation(False), False, (0, 0))
    assert ctl.tick(moved, False, (0, 0))
    ctl.reset_target_memory()
    assert ctl.tick(moved, False, (0, 0))
