import numpy as np
import pytest

from fisch_macro.perception.reel import detect_reel
from fisch_macro.types import Rect


def _reel_frame(marker_x: int | None) -> np.ndarray:
    image = np.full((200, 300, 3), 100, dtype=np.uint8)
    image[144:190, 48:60] = 0  # Darker than the marker and track.
    image[165:177, 65:235] = 25
    image[164:178, 100:170] = 240
    image[167:175, 108:116] = 105  # Arrow contained within the white bar.
    if marker_x is not None:
        image[160:182, marker_x - 1 : marker_x + 2] = 65
    return image


@pytest.mark.parametrize("marker_x", [80, 136, 220])
def test_detect_reel_tracks_fish_inside_and_outside_white_bar(marker_x):
    observation = detect_reel(_reel_frame(marker_x))
    assert observation.found
    assert observation.geometry == Rect(65, 164, 170, 14)
    assert observation.extras["fish_x"] == pytest.approx((marker_x - 65) / 169)
    assert observation.extras["bar_left"] == pytest.approx((100 - 65) / 169)
    assert observation.extras["bar_right"] == pytest.approx((169 - 65) / 169)


def test_detect_reel_rejects_track_without_marker():
    assert not detect_reel(_reel_frame(marker_x=None)).found


def test_detect_reel_rejects_bright_bar_without_track():
    frame = np.full((200, 300, 3), 100, dtype=np.uint8)
    frame[164:178, 100:170] = 240
    frame[160:182, 135:138] = 65
    assert not detect_reel(frame).found
