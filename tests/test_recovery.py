from unittest.mock import MagicMock, call

from fisch_macro.controllers.recovery import RecoveryController


def test_recovery_run_cancel_release_clear():
    inp = MagicMock()
    RecoveryController(inp).run()
    assert inp.method_calls == [call.cancel(), call.release_all(), call.clear_cancel()]
