import numpy as np

from fisch_macro.controllers.cast import CastController
from fisch_macro.input_controller import InputController
from fisch_macro.perception.cast_meter import detect_cast_meter
from fisch_macro.types import ProfileData


def _meter_image(fill_ratio: float) -> np.ndarray:
    h, w = 200, 40
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (20, 20, 20)
    fill_h = int(h * fill_ratio)
    img[h - fill_h : h, 10:30] = (240, 240, 240)
    img[0:12, 10:30] = (0, 220, 0)
    return img


def test_detect_cast_meter_reports_fill():
    obs = detect_cast_meter(_meter_image(0.85))
    assert obs.found
    assert obs.extras["fill_ratio"] > 0.7


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


def test_cast_controller_releases_near_target_fill():
    mouse = FakeMouse()
    inp = InputController(mouse=mouse, sleep_fn=lambda _s: None)
    ctl = CastController(inp, ProfileData(name="t", saved_at="2026-11-09"), safe_xy=(100, 100))
    assert ctl.begin_cast(client_origin=(0, 0)) is True
    assert ("down",) in mouse.events
    released = ctl.update_fill(fill_ratio=0.92)
    assert released is True
    assert ("up",) in mouse.events
