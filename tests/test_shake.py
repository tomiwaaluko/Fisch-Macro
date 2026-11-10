# tests/test_shake.py
import numpy as np

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
    mouse.events.clear()
    acted2 = ctl.tick(shake=shake, bite_or_reel=True, client_origin=(0, 0))
    assert acted2 is False
