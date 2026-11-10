from __future__ import annotations

from fisch_macro.input_controller import InputController


class RecoveryController:
    def __init__(self, input_ctl: InputController) -> None:
        self.input = input_ctl

    def run(self) -> None:
        self.input.cancel()
        self.input.release_all()
        self.input.clear_cancel()
