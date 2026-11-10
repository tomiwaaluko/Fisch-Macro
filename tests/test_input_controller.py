# tests/test_input_controller.py
from fisch_macro.input_controller import InputController


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


def test_click_emits_down_up_and_respects_cancel():
    mouse = FakeMouse()
    ctl = InputController(mouse=mouse, sleep_fn=lambda _s: None)
    ctl.cancel()
    assert ctl.click(10, 20, hold_ms=50) is False
    assert mouse.events == []


def test_release_all_ups_if_held():
    mouse = FakeMouse()
    ctl = InputController(mouse=mouse, sleep_fn=lambda _s: None)
    ctl.mouse_down()
    ctl.release_all()
    assert ("down",) in mouse.events
    assert ("up",) in mouse.events
