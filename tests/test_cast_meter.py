import numpy as np
import pytest

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


def test_latency_lead_releases_before_the_raw_fill_target():
    mouse = FakeMouse()
    inp = InputController(mouse=mouse, sleep_fn=lambda _s: None)
    profile = ProfileData(name="t", saved_at="2026-11-09", cast_latency_ms=90, cast_hold_ms_prior=580)
    ctl = CastController(inp, profile, safe_xy=(100, 100))
    assert ctl.release_threshold() == pytest.approx(0.90 - 90 / 580)
    assert ctl.begin_cast((0, 0), now_ms=0) is True
    assert ctl.update_fill(0.70, now_ms=50, meter_found=True) is False
    assert ctl._holding is True
    assert ctl.update_fill(0.80, now_ms=80, meter_found=True) is True
    assert ("up",) in mouse.events


def test_missing_meter_holds_until_cast_prior_then_releases():
    mouse = FakeMouse()
    inp = InputController(mouse=mouse, sleep_fn=lambda _s: None)
    profile = ProfileData(name="t", saved_at="2026-11-09", cast_hold_ms_prior=580)
    ctl = CastController(inp, profile, safe_xy=(100, 100))
    assert ctl.begin_cast((0, 0), now_ms=1000) is True
    assert ctl.update_fill(0.0, now_ms=1100, meter_found=False) is False
    assert ctl._holding is True
    assert ctl.update_fill(0.99, now_ms=1200, meter_found=True) is True


def test_visible_low_meter_does_not_use_the_short_prior():
    mouse = FakeMouse()
    inp = InputController(mouse=mouse, sleep_fn=lambda _s: None)
    profile = ProfileData(name="t", saved_at="2026-11-09", cast_hold_ms_prior=580)
    ctl = CastController(inp, profile, safe_xy=(100, 100))
    assert ctl.begin_cast((0, 0), now_ms=0) is True
    assert ctl.update_fill(0.2, now_ms=580, meter_found=True) is False
    assert ctl.update_fill(0.2, now_ms=1160, meter_found=True) is True
