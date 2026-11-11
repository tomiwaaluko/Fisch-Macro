# tests/test_shake.py
import cv2
import numpy as np

from fisch_macro.controllers import shake as shake_controller
from fisch_macro.controllers.shake import ShakeController
from fisch_macro.input_controller import InputController
from fisch_macro.perception.shake import detect_shake
from fisch_macro.types import Observation, ProfileData, Rect


def _synthetic_shake(h=200, w=300, radius=40, center=(150, 100), label=True) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2
    img[mask] = (40, 40, 40)
    ring = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= (radius + 3) ** 2) & ~mask
    img[ring] = (220, 220, 220)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = radius / 73
        (width, height), _ = cv2.getTextSize("SHAKE", font, scale, 1)
        origin = (center[0] - width // 2, center[1] + height // 2)
        cv2.putText(img, "SHAKE", origin, font, scale, (240, 240, 240), 1, cv2.LINE_AA)
    return img


def test_detect_shake_finds_labeled_circle():
    img = _synthetic_shake()
    obs = detect_shake(img)
    assert obs.found
    assert obs.geometry is not None
    assert obs.extras["radius"] >= 30


def test_detect_shake_returns_full_size_coordinates_after_downsampling():
    center = (853, 302)
    radius = 64
    image = _synthetic_shake(h=720, w=1280, radius=radius, center=center)

    obs = detect_shake(image)

    assert obs.found
    x, y = obs.extras["center"]
    assert abs(x - center[0]) <= 5
    assert abs(y - center[1]) <= 5
    assert abs(obs.extras["radius"] - radius) <= 5


def test_detect_shake_rejects_unlabeled_ring():
    assert not detect_shake(_synthetic_shake(label=False)).found


def test_detect_shake_bounds_an_inflated_saved_radius():
    assert detect_shake(_synthetic_shake(), min_radius=1000, max_radius=3000).found


def test_detect_shake_prefers_active_button_over_larger_unlabeled_ring():
    button = _synthetic_shake(w=400, center=(100, 100))
    old_ring = _synthetic_shake(w=400, radius=52, center=(285, 115), label=False)
    obs = detect_shake(np.maximum(button, old_ring))
    assert obs.found
    x, y = obs.extras["center"]
    assert abs(x - 100) <= 5 and abs(y - 100) <= 5


def test_detect_shake_rejects_text_without_enclosing_button(monkeypatch):
    image = _synthetic_shake()
    yy, xx = np.ogrid[:200, :300]
    image[(xx - 150) ** 2 + (yy - 100) ** 2 > 25**2] = 0
    monkeypatch.setattr(cv2, "HoughCircles", lambda *a, **kw: np.array([[[150, 100, 40]]]))
    assert not detect_shake(image).found


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
