from fisch_macro.controllers.reel import ReelController
from fisch_macro.types import ProfileData


def test_safe_mode_holds_when_fish_right_of_center():
    ctl = ReelController(ProfileData(name="t", saved_at="2026-11-09"))
    hold = ctl.decide_hold(
        fish_x=0.70,
        bar_left=0.40,
        bar_right=0.80,
        progress=0.3,
        progress_slope=0.01,
        fish_volatility=0.4,
    )
    assert hold is True


def test_emergency_releases_when_fish_left_of_center():
    ctl = ReelController(ProfileData(name="t", saved_at="2026-11-09"))
    hold = ctl.decide_hold(
        fish_x=0.20,
        bar_left=0.30,
        bar_right=0.70,
        progress=0.2,
        progress_slope=-0.05,
        fish_volatility=0.9,
    )
    assert hold is False


def test_tighten_allowed_when_progress_healthy():
    ctl = ReelController(ProfileData(name="t", saved_at="2026-11-09"))
    mode = ctl.select_mode(progress=0.6, progress_slope=0.05, fish_volatility=0.3)
    assert mode == "tighten"
