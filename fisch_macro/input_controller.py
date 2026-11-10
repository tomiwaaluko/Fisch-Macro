# fisch_macro/input_controller.py
from __future__ import annotations

import time
from typing import Callable, Protocol


class MouseBackend(Protocol):
    position: tuple[int, int]

    def move_to(self, x: int, y: int) -> None: ...
    def down(self) -> None: ...
    def up(self) -> None: ...


class PynputMouse:
    def __init__(self) -> None:
        from pynput.mouse import Button, Controller

        self._ctl = Controller()
        self._button = Button.left

    @property
    def position(self) -> tuple[int, int]:
        pos = self._ctl.position
        return (int(pos[0]), int(pos[1]))

    def move_to(self, x: int, y: int) -> None:
        self._ctl.position = (x, y)

    def down(self) -> None:
        self._ctl.press(self._button)

    def up(self) -> None:
        self._ctl.release(self._button)


class InputController:
    def __init__(
        self,
        mouse: MouseBackend | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._mouse = mouse or PynputMouse()
        self._sleep = sleep_fn
        self._cancelled = False
        self._held = False

    def cancel(self) -> None:
        self._cancelled = True

    def clear_cancel(self) -> None:
        self._cancelled = False

    def is_cancelled(self) -> bool:
        return self._cancelled

    def move_to(self, x: int, y: int) -> bool:
        if self._cancelled:
            return False
        self._mouse.move_to(x, y)
        return True

    def mouse_down(self) -> bool:
        if self._cancelled:
            return False
        self._mouse.down()
        self._held = True
        return True

    def mouse_up(self) -> None:
        if self._held:
            self._mouse.up()
            self._held = False

    def release_all(self) -> None:
        self.mouse_up()

    def click(self, x: int, y: int, hold_ms: float = 70.0) -> bool:
        if self._cancelled:
            return False
        if not self.move_to(x, y):
            return False
        if not self.mouse_down():
            return False
        self._sleep(hold_ms / 1000.0)
        if self._cancelled:
            self.release_all()
            return False
        self.mouse_up()
        return True
