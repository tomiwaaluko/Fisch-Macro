import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fisch_macro import capture, ui
from fisch_macro.types import BotState


@pytest.mark.parametrize(
    "title,minimized,visible,expected",
    [
        ("Roblox", False, True, True),
        ("Roblox - Browser", False, True, False),
        ("Roblox", True, True, False),
        ("Roblox", False, False, False),
    ],
)
def test_only_usable_foreground_roblox_is_accepted(monkeypatch, title, minimized, visible, expected):
    active = SimpleNamespace(
        title=title, isMinimized=minimized, visible=visible, box=(20, 10, 640, 480), _hWnd=7
    )
    gw = SimpleNamespace(getActiveWindow=lambda: active, PyGetWindowException=RuntimeError)
    monkeypatch.setitem(sys.modules, "pygetwindow", gw)
    assert (capture.find_roblox_window() is not None) is expected


@pytest.mark.parametrize(
    "replacement",
    [None, capture.WindowInfo(40, 10, 640, 480, "Roblox", 7)],
)
@pytest.mark.parametrize("before_transition", [True, False])
def test_worker_discards_frame_if_window_loses_focus_or_moves(
    monkeypatch, replacement, before_transition
):
    original = capture.WindowInfo(20, 10, 640, 480, "Roblox", 7)
    windows = iter([original, replacement] if before_transition else [original, original, replacement])
    monkeypatch.setattr(ui, "find_roblox_window", lambda: next(windows))
    frame = object()
    grab = Mock(return_value=frame)
    monkeypatch.setattr(ui, "grab_frame", grab)
    app = ui.App.__new__(ui.App)
    app._stop_event = threading.Event()
    app.runtime = Mock(state=BotState.REELING, _running=True)
    app.runtime.enter_recovery.return_value = False
    app._set_status = Mock()

    app._tick()

    grab.assert_called_once_with(original)
    app.runtime.enter_recovery.assert_called_once_with()
    assert app.runtime.apply_bundle.called is not before_transition
    app.runtime.act.assert_not_called()
