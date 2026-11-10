# Fisch Autonomous Macro Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the prototype recorder with a modular classical-CV fishing loop that casts, shakes, reels, and recasts autonomously at a fixed spot while adapting to shake size/speed and reel difficulty.

**Architecture:** Capture client ROI → perception detectors emit structured observations → priority state machine selects stage → interruptible controllers issue mouse actions → profiler adapts priors and can save/load named profiles.

**Tech Stack:** Python 3.11+, OpenCV (`opencv-python`), NumPy, MSS or `pygetwindow`+Pillow for capture, `pynput` for input, `keyboard` for global hotkeys, Tkinter for operator UI, `pytest` for tests.

## Global Constraints

- Fixed fishing spot only (no travel/sell/bait automation in v1)
- No recorded click-playback as the autonomy engine
- Result and reel evidence always outrank stale SHAKE detections
- Client-relative coordinates only (never hardcode full-desktop paths like `C:\Users\...`)
- Reel policy: safe-first, tighten only when progress/tracking healthy
- Auto-adapt by default; optional named JSON profiles
- Kill hotkey must release LMB immediately
- Spec reference: `docs/superpowers/specs/2026-11-09-fisch-autonomous-macro-design.md`

---

## File Structure

```text
fisch_macro/
  __init__.py
  types.py                 # Observation, FramePacket, BotState, ProfileData
  capture.py               # Roblox window ROI capture
  input_controller.py      # Interruptible mouse move/down/up
  state_machine.py         # Priority transitions
  loop.py                  # Main tick: capture→perceive→transition→act
  profiler.py              # Online stats + profile save/load
  logging_util.py          # Structured JSONL logs
  perception/
    __init__.py
    frame_bundle.py        # Aggregated detections for one frame
    cast_meter.py
    shake.py
    reel.py
    result.py
  controllers/
    __init__.py
    cast.py
    shake.py
    reel.py
    recovery.py
  ui.py                    # Tkinter operator panel
  __main__.py              # python -m fisch_macro
tests/
  test_types.py
  test_state_machine.py
  test_input_controller.py
  test_cast_meter.py
  test_shake.py
  test_reel_controller.py
  test_profiler.py
  test_loop_transitions.py
  fixtures/
    README.md              # How to add synthetic PNGs from video frames
requirements.txt
README.md                  # Update run instructions
```

Keep legacy `Fisch Macro.py` until Task 10, then rename to `legacy_fisch_macro.py`.

---

### Task 1: Project skeleton, types, and dependencies

**Files:**
- Create: `requirements.txt`
- Create: `fisch_macro/__init__.py`
- Create: `fisch_macro/types.py`
- Create: `tests/test_types.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: none
- Produces: `BotState` enum; `Observation` dataclass; `Rect`, `FramePacket`, `ProfileData`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_types.py
from fisch_macro.types import BotState, Observation, Rect


def test_observation_defaults():
    obs = Observation(found=False)
    assert obs.found is False
    assert obs.confidence == 0.0
    assert obs.age_ms == 0
    assert obs.geometry is None


def test_bot_state_has_required_values():
    names = {s.name for s in BotState}
    assert {
        "IDLE",
        "PRE_CAST",
        "CASTING",
        "WAITING_SHAKE",
        "SHAKING",
        "BITE_DETECTED",
        "REEL_INITIALIZING",
        "REELING",
        "CATCH_RESULT",
        "RESET_TO_CAST",
        "RECOVERY",
    } <= names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_types.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'fisch_macro'`

- [ ] **Step 3: Write minimal implementation**

```text
# requirements.txt
opencv-python>=4.8
numpy>=1.26
Pillow>=10.0
mss>=9.0
pygetwindow>=0.0.9
pynput>=1.7
keyboard>=0.13
pytest>=8.0
```

```python
# fisch_macro/__init__.py
"""Autonomous Fisch fishing loop (classical CV + feedback control)."""

__version__ = "0.1.0"
```

```python
# fisch_macro/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


class BotState(Enum):
    IDLE = auto()
    PRE_CAST = auto()
    CASTING = auto()
    WAITING_SHAKE = auto()
    SHAKING = auto()
    BITE_DETECTED = auto()
    REEL_INITIALIZING = auto()
    REELING = auto()
    CATCH_RESULT = auto()
    RESET_TO_CAST = auto()
    RECOVERY = auto()


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


@dataclass
class Observation:
    found: bool
    confidence: float = 0.0
    geometry: Rect | None = None
    age_ms: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class FramePacket:
    """BGR image in client-relative coordinates."""

    image: np.ndarray
    timestamp_ms: int
    client_origin: tuple[int, int]  # top-left of Roblox client on desktop


@dataclass
class ProfileData:
    name: str
    saved_at: str
    cast_hold_ms_prior: float = 580.0
    cast_latency_ms: float = 90.0
    shake_radius_px_mean: float = 110.0
    shake_click_hold_ms: float = 70.0
    bar_width_ratio: float = 0.40
    fish_volatility: float = 0.5
    reel_safe_margin_ratio: float = 0.25
    reel_tighten_margin_ratio: float = 0.12
```

Update `README.md` Installation/Usage to mention `pip install -r requirements.txt` and `python -m fisch_macro` (entrypoint lands in Task 9).

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install -r requirements.txt` then `pytest tests/test_types.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt fisch_macro/__init__.py fisch_macro/types.py tests/test_types.py README.md
git commit -m "feat: add autonomous macro package skeleton and core types"
```

---

### Task 2: Interruptible input controller

**Files:**
- Create: `fisch_macro/input_controller.py`
- Create: `tests/test_input_controller.py`

**Interfaces:**
- Consumes: none
- Produces: `InputController` with `move_to`, `click`, `mouse_down`, `mouse_up`, `release_all`, `cancel`, `is_cancelled`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_input_controller.py -v`  
Expected: FAIL with import error for `input_controller`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_input_controller.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fisch_macro/input_controller.py tests/test_input_controller.py
git commit -m "feat: add interruptible mouse input controller"
```

---

### Task 3: Priority state machine

**Files:**
- Create: `fisch_macro/state_machine.py`
- Create: `fisch_macro/perception/frame_bundle.py`
- Create: `fisch_macro/perception/__init__.py`
- Create: `tests/test_state_machine.py`

**Interfaces:**
- Consumes: `BotState`, `Observation`
- Produces: `FrameBundle`; `StateMachine.transition(state, bundle, operator_stop) -> BotState`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_state_machine.py
from fisch_macro.perception.frame_bundle import FrameBundle
from fisch_macro.state_machine import StateMachine
from fisch_macro.types import BotState, Observation


def bundle(**kwargs) -> FrameBundle:
    base = dict(
        cast=Observation(False),
        shake=Observation(False),
        bite=Observation(False),
        reel=Observation(False),
        result=Observation(False),
        ready=Observation(False),
    )
    base.update(kwargs)
    return FrameBundle(**base)


def test_result_outranks_shake_while_reeling():
    sm = StateMachine()
    nxt = sm.transition(
        BotState.REELING,
        bundle(shake=Observation(True, 0.9), result=Observation(True, 0.95)),
        operator_stop=False,
    )
    assert nxt == BotState.CATCH_RESULT


def test_reel_outranks_shake():
    sm = StateMachine()
    nxt = sm.transition(
        BotState.SHAKING,
        bundle(shake=Observation(True, 0.9), reel=Observation(True, 0.9)),
        operator_stop=False,
    )
    assert nxt in {BotState.BITE_DETECTED, BotState.REEL_INITIALIZING, BotState.REELING}


def test_operator_stop_forces_idle():
    sm = StateMachine()
    nxt = sm.transition(BotState.SHAKING, bundle(), operator_stop=True)
    assert nxt == BotState.IDLE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_state_machine.py -v`  
Expected: FAIL with import errors

- [ ] **Step 3: Write minimal implementation**

```python
# fisch_macro/perception/__init__.py
"""Vision detectors for Fisch UI stages."""
```

```python
# fisch_macro/perception/frame_bundle.py
from __future__ import annotations

from dataclasses import dataclass

from fisch_macro.types import Observation


@dataclass
class FrameBundle:
    cast: Observation
    shake: Observation
    bite: Observation
    reel: Observation
    result: Observation
    ready: Observation
```

```python
# fisch_macro/state_machine.py
from __future__ import annotations

from fisch_macro.perception.frame_bundle import FrameBundle
from fisch_macro.types import BotState


class StateMachine:
    def transition(
        self,
        state: BotState,
        bundle: FrameBundle,
        operator_stop: bool,
    ) -> BotState:
        if operator_stop:
            return BotState.IDLE

        if bundle.result.found:
            return BotState.CATCH_RESULT

        if bundle.reel.found:
            if state == BotState.REELING:
                return BotState.REELING
            if state == BotState.REEL_INITIALIZING:
                if bundle.reel.extras.get("stable", False):
                    return BotState.REELING
                return BotState.REEL_INITIALIZING
            if state == BotState.BITE_DETECTED:
                return BotState.REEL_INITIALIZING
            return BotState.BITE_DETECTED

        if bundle.bite.found:
            return BotState.BITE_DETECTED

        if bundle.shake.found:
            return BotState.SHAKING

        if bundle.cast.found:
            return BotState.CASTING

        if state == BotState.CATCH_RESULT:
            if bundle.ready.found:
                return BotState.RESET_TO_CAST
            return BotState.CATCH_RESULT

        if state == BotState.RESET_TO_CAST and bundle.ready.found:
            return BotState.PRE_CAST

        if state in {BotState.CASTING, BotState.WAITING_SHAKE} and not bundle.shake.found:
            return BotState.WAITING_SHAKE

        if state == BotState.PRE_CAST:
            return BotState.PRE_CAST

        if state == BotState.RECOVERY:
            if bundle.ready.found:
                return BotState.PRE_CAST
            if bundle.reel.found:
                return BotState.REEL_INITIALIZING
            if bundle.shake.found:
                return BotState.SHAKING
            return BotState.RECOVERY

        return state
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_state_machine.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fisch_macro/perception/__init__.py fisch_macro/perception/frame_bundle.py fisch_macro/state_machine.py tests/test_state_machine.py
git commit -m "feat: add priority fishing state machine"
```

---

### Task 4: Capture + structured logging

**Files:**
- Create: `fisch_macro/capture.py`
- Create: `fisch_macro/logging_util.py`
- Create: `tests/test_logging_util.py`
- Create: `tests/fixtures/README.md`

**Interfaces:**
- Consumes: `FramePacket`
- Produces: `find_roblox_window()`, `grab_frame(window) -> FramePacket`; `JsonlLogger.log(event, **fields)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_logging_util.py
import json
from pathlib import Path

from fisch_macro.logging_util import JsonlLogger


def test_jsonl_logger_writes_event(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    log = JsonlLogger(path)
    log.log("state_change", from_state="IDLE", to_state="PRE_CAST")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["event"] == "state_change"
    assert row["to_state"] == "PRE_CAST"
    assert "ts" in row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_logging_util.py -v`  
Expected: FAIL with import error

- [ ] **Step 3: Write minimal implementation**

```python
# fisch_macro/logging_util.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> None:
        row = {"ts": time.time(), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
```

```python
# fisch_macro/capture.py
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from fisch_macro.types import FramePacket


@dataclass
class WindowInfo:
    left: int
    top: int
    width: int
    height: int
    title: str


def find_roblox_window(title_substr: str = "Roblox") -> WindowInfo | None:
    import pygetwindow as gw

    matches = [w for w in gw.getAllWindows() if title_substr.lower() in (w.title or "").lower()]
    if not matches:
        return None
    w = matches[0]
    if w.width <= 0 or w.height <= 0:
        return None
    return WindowInfo(w.left, w.top, w.width, w.height, w.title)


def grab_frame(window: WindowInfo) -> FramePacket:
    import mss
    from PIL import Image

    monitor = {
        "left": int(window.left),
        "top": int(window.top),
        "width": int(window.width),
        "height": int(window.height),
    }
    with mss.mss() as sct:
        raw = sct.grab(monitor)
    img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    bgr = np.array(img)[:, :, ::-1].copy()
    return FramePacket(
        image=bgr,
        timestamp_ms=int(time.time() * 1000),
        client_origin=(monitor["left"], monitor["top"]),
    )
```

```markdown
# tests/fixtures/README.md

Add synthetic PNGs cropped from the evidence video for detector tests:
- `cast_meter.png`
- `shake_button.png`
- `reel_bar.png`
- `perfect_catch.png`

Prefer client-only crops (no taskbar). Keep filenames stable; tests load by name.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_logging_util.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fisch_macro/capture.py fisch_macro/logging_util.py tests/test_logging_util.py tests/fixtures/README.md
git commit -m "feat: add frame capture helpers and JSONL logger"
```

---

### Task 5: SHAKE detector + transactional shake controller

**Files:**
- Create: `fisch_macro/perception/shake.py`
- Create: `fisch_macro/controllers/__init__.py`
- Create: `fisch_macro/controllers/shake.py`
- Create: `tests/test_shake.py`

**Interfaces:**
- Consumes: `FramePacket`, `InputController`, `ProfileData` priors
- Produces: `detect_shake(image) -> Observation` with `extras["radius"]`; `ShakeController.tick(...)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_shake.py
import numpy as np

from fisch_macro.controllers.shake import ShakeController
from fisch_macro.input_controller import InputController
from fisch_macro.perception.shake import detect_shake
from fisch_macro.types import Observation, ProfileData, Rect


def _synthetic_shake(h=200, w=300, radius=40, center=(150, 100)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2
    img[mask] = (40, 40, 40)
    ring = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= (radius + 3) ** 2) & ~mask
    img[ring] = (220, 220, 220)
    return img


def test_detect_shake_finds_circle():
    img = _synthetic_shake()
    obs = detect_shake(img)
    assert obs.found
    assert obs.geometry is not None
    assert obs.extras["radius"] >= 30


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
    mouse.events.clear()
    acted2 = ctl.tick(shake=shake, bite_or_reel=True, client_origin=(0, 0))
    assert acted2 is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_shake.py -v`  
Expected: FAIL with import errors

- [ ] **Step 3: Write minimal implementation**

```python
# fisch_macro/controllers/__init__.py
"""Stage controllers for cast/shake/reel/recovery."""
```

```python
# fisch_macro/perception/shake.py
from __future__ import annotations

import cv2
import numpy as np

from fisch_macro.types import Observation, Rect


def detect_shake(image_bgr: np.ndarray, min_radius: int = 20, max_radius: int = 160) -> Observation:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(40, min_radius),
        param1=100,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return Observation(False)
    circles = np.round(circles[0]).astype(int)
    x, y, r = max(circles, key=lambda c: c[2])
    return Observation(
        found=True,
        confidence=0.8,
        geometry=Rect(int(x - r), int(y - r), int(2 * r), int(2 * r)),
        extras={"radius": int(r), "center": (int(x), int(y))},
    )
```

```python
# fisch_macro/controllers/shake.py
from __future__ import annotations

from fisch_macro.input_controller import InputController
from fisch_macro.types import Observation, ProfileData


class ShakeController:
    def __init__(self, input_ctl: InputController, profile: ProfileData) -> None:
        self.input = input_ctl
        self.profile = profile
        self._last_center: tuple[int, int] | None = None

    def tick(self, shake: Observation, bite_or_reel: bool, client_origin: tuple[int, int]) -> bool:
        if bite_or_reel:
            self.input.cancel()
            self.input.release_all()
            self.input.clear_cancel()
            return False
        if not shake.found or shake.geometry is None:
            return False
        cx, cy = shake.extras.get("center") or shake.geometry.center
        ox, oy = client_origin
        screen_x, screen_y = ox + int(cx), oy + int(cy)
        if self._last_center and abs(self._last_center[0] - cx) < 8 and abs(self._last_center[1] - cy) < 8:
            return False
        ok = self.input.click(screen_x, screen_y, hold_ms=self.profile.shake_click_hold_ms)
        if ok:
            self._last_center = (int(cx), int(cy))
            radius = float(shake.extras.get("radius", self.profile.shake_radius_px_mean))
            self.profile.shake_radius_px_mean = 0.8 * self.profile.shake_radius_px_mean + 0.2 * radius
        return ok

    def reset_target_memory(self) -> None:
        self._last_center = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_shake.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fisch_macro/perception/shake.py fisch_macro/controllers/__init__.py fisch_macro/controllers/shake.py tests/test_shake.py
git commit -m "feat: add shake detection and transactional click controller"
```

---

### Task 6: Cast meter detector + cast controller

**Files:**
- Create: `fisch_macro/perception/cast_meter.py`
- Create: `fisch_macro/controllers/cast.py`
- Create: `tests/test_cast_meter.py`

**Interfaces:**
- Consumes: `InputController`, `ProfileData`
- Produces: `detect_cast_meter(image) -> Observation` with `extras["fill_ratio"]`; `CastController`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cast_meter.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cast_meter.py -v`  
Expected: FAIL with import errors

- [ ] **Step 3: Write minimal implementation**

```python
# fisch_macro/perception/cast_meter.py
from __future__ import annotations

import cv2
import numpy as np

from fisch_macro.types import Observation, Rect


def detect_cast_meter(image_bgr: np.ndarray) -> Observation:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    if int(white.sum()) < 50:
        return Observation(False)
    ys, xs = np.where(white > 0)
    if len(ys) == 0:
        return Observation(False)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    h = image_bgr.shape[0]
    fill_ratio = 1.0 - (y0 / max(h, 1))
    return Observation(
        found=True,
        confidence=0.7,
        geometry=Rect(x0, y0, max(1, x1 - x0), max(1, y1 - y0)),
        extras={"fill_ratio": float(np.clip(fill_ratio, 0.0, 1.0)), "green_band": True},
    )
```

```python
# fisch_macro/controllers/cast.py
from __future__ import annotations

from fisch_macro.input_controller import InputController
from fisch_macro.types import ProfileData


class CastController:
    def __init__(
        self,
        input_ctl: InputController,
        profile: ProfileData,
        safe_xy: tuple[int, int],
        release_fill: float = 0.90,
    ) -> None:
        self.input = input_ctl
        self.profile = profile
        self.safe_xy = safe_xy
        self.release_fill = release_fill
        self._holding = False

    def begin_cast(self, client_origin: tuple[int, int]) -> bool:
        ox, oy = client_origin
        x, y = self.safe_xy
        if not self.input.move_to(ox + x, oy + y):
            return False
        ok = self.input.mouse_down()
        self._holding = ok
        return ok

    def update_fill(self, fill_ratio: float) -> bool:
        if not self._holding:
            return False
        lead = min(0.08, self.profile.cast_latency_ms / 1000.0)
        target = max(0.5, self.release_fill - lead)
        if fill_ratio >= target:
            self.input.mouse_up()
            self._holding = False
            return True
        return False

    def force_release(self) -> None:
        self.input.mouse_up()
        self._holding = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cast_meter.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fisch_macro/perception/cast_meter.py fisch_macro/controllers/cast.py tests/test_cast_meter.py
git commit -m "feat: add cast meter detection and release controller"
```

---

### Task 7: Reel perception + safe/tighten/emergency controller

**Files:**
- Create: `fisch_macro/perception/reel.py`
- Create: `fisch_macro/controllers/reel.py`
- Create: `tests/test_reel_controller.py`

**Interfaces:**
- Consumes: `InputController`, `ProfileData`
- Produces: `detect_reel(image) -> Observation`; `ReelController.decide_hold(...)`, `select_mode(...)`, `tick(...)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reel_controller.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reel_controller.py -v`  
Expected: FAIL with import error

- [ ] **Step 3: Write minimal implementation**

```python
# fisch_macro/perception/reel.py
from __future__ import annotations

import cv2
import numpy as np

from fisch_macro.types import Observation, Rect


def detect_reel(image_bgr: np.ndarray) -> Observation:
    h, w = image_bgr.shape[:2]
    roi = image_bgr[int(h * 0.70) : int(h * 0.95), int(w * 0.20) : int(w * 0.80)]
    if roi.size == 0:
        return Observation(False)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    ys, xs = np.where(bright > 0)
    if len(xs) < 30:
        return Observation(False)
    bar_left = float(xs.min()) / max(roi.shape[1] - 1, 1)
    bar_right = float(xs.max()) / max(roi.shape[1] - 1, 1)
    col_means = gray.mean(axis=0)
    fish_idx = int(np.argmin(col_means))
    fish_x = fish_idx / max(roi.shape[1] - 1, 1)
    bottom = bright[int(bright.shape[0] * 0.7) :]
    progress = float(np.clip(bottom.mean() / 255.0, 0.0, 1.0))
    return Observation(
        found=True,
        confidence=0.65,
        geometry=Rect(int(w * 0.20), int(h * 0.70), int(w * 0.60), int(h * 0.25)),
        extras={
            "stable": True,
            "fish_x": fish_x,
            "bar_left": bar_left,
            "bar_right": bar_right,
            "progress": progress,
        },
    )
```

```python
# fisch_macro/controllers/reel.py
from __future__ import annotations

from fisch_macro.input_controller import InputController
from fisch_macro.types import ProfileData


class ReelController:
    def __init__(self, profile: ProfileData, input_ctl: InputController | None = None) -> None:
        self.profile = profile
        self.input = input_ctl
        self._holding = False

    def select_mode(self, progress: float, progress_slope: float, fish_volatility: float) -> str:
        if progress_slope < 0 or fish_volatility > 0.85:
            return "emergency"
        if progress >= 0.35 and progress_slope > 0.02 and fish_volatility < 0.55:
            return "tighten"
        return "safe"

    def decide_hold(
        self,
        fish_x: float,
        bar_left: float,
        bar_right: float,
        progress: float,
        progress_slope: float,
        fish_volatility: float,
    ) -> bool:
        mode = self.select_mode(progress, progress_slope, fish_volatility)
        width = max(1e-6, bar_right - bar_left)
        center = (bar_left + bar_right) / 2.0
        if mode == "tighten":
            margin = self.profile.reel_tighten_margin_ratio * width
        elif mode == "emergency":
            margin = self.profile.reel_safe_margin_ratio * width * 1.3
        else:
            margin = self.profile.reel_safe_margin_ratio * width

        if fish_x > center + margin:
            return True
        if fish_x < center - margin:
            return False
        return self._holding

    def tick(
        self,
        fish_x: float,
        bar_left: float,
        bar_right: float,
        progress: float,
        progress_slope: float,
        fish_volatility: float,
    ) -> str:
        want_hold = self.decide_hold(
            fish_x, bar_left, bar_right, progress, progress_slope, fish_volatility
        )
        mode = self.select_mode(progress, progress_slope, fish_volatility)
        if self.input is None:
            self._holding = want_hold
            return mode
        if want_hold and not self._holding:
            self.input.mouse_down()
            self._holding = True
        elif not want_hold and self._holding:
            self.input.mouse_up()
            self._holding = False
        return mode

    def force_release(self) -> None:
        if self.input is not None:
            self.input.mouse_up()
        self._holding = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reel_controller.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fisch_macro/perception/reel.py fisch_macro/controllers/reel.py tests/test_reel_controller.py
git commit -m "feat: add reel perception and adaptive hold/release controller"
```

---

### Task 8: Result detection, recovery, profiler profiles

**Files:**
- Create: `fisch_macro/perception/result.py`
- Create: `fisch_macro/controllers/recovery.py`
- Create: `fisch_macro/profiler.py`
- Create: `tests/test_profiler.py`
- Create: `profiles/.gitkeep`

**Interfaces:**
- Consumes: `ProfileData`
- Produces: `detect_result(image) -> Observation`; `Profiler.load/save/update_*`; `RecoveryController.run()`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_profiler.py
from pathlib import Path

from fisch_macro.profiler import Profiler
from fisch_macro.types import ProfileData


def test_profiler_save_and_load_roundtrip(tmp_path: Path):
    path = tmp_path / "steady.json"
    p = Profiler(ProfileData(name="steady", saved_at="2026-11-09", shake_radius_px_mean=95.0))
    p.update_shake_radius(120.0)
    p.save(path)
    loaded = Profiler.load(path)
    assert loaded.data.name == "steady"
    assert loaded.data.shake_radius_px_mean == p.data.shake_radius_px_mean
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_profiler.py -v`  
Expected: FAIL with import error

- [ ] **Step 3: Write minimal implementation**

```python
# fisch_macro/perception/result.py
from __future__ import annotations

import cv2
import numpy as np

from fisch_macro.types import Observation


def detect_result(image_bgr: np.ndarray) -> Observation:
    h, w = image_bgr.shape[:2]
    roi = image_bgr[int(h * 0.25) : int(h * 0.55), int(w * 0.25) : int(w * 0.75)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bright_ratio = float((gray > 200).mean())
    found = bright_ratio > 0.08
    return Observation(found=found, confidence=bright_ratio, extras={"bright_ratio": bright_ratio})
```

```python
# fisch_macro/controllers/recovery.py
from __future__ import annotations

from fisch_macro.input_controller import InputController


class RecoveryController:
    def __init__(self, input_ctl: InputController) -> None:
        self.input = input_ctl

    def run(self) -> None:
        self.input.cancel()
        self.input.release_all()
        self.input.clear_cancel()
```

```python
# fisch_macro/profiler.py
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from fisch_macro.types import ProfileData


class Profiler:
    def __init__(self, data: ProfileData) -> None:
        self.data = data

    @classmethod
    def load(cls, path: str | Path) -> "Profiler":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(ProfileData(**raw))

    def save(self, path: str | Path) -> None:
        self.data.saved_at = date.today().isoformat()
        Path(path).write_text(json.dumps(asdict(self.data), indent=2), encoding="utf-8")

    def update_shake_radius(self, radius: float) -> None:
        self.data.shake_radius_px_mean = 0.8 * self.data.shake_radius_px_mean + 0.2 * radius

    def update_bar_width_ratio(self, ratio: float) -> None:
        self.data.bar_width_ratio = 0.8 * self.data.bar_width_ratio + 0.2 * ratio

    def update_fish_volatility(self, volatility: float) -> None:
        self.data.fish_volatility = 0.8 * self.data.fish_volatility + 0.2 * volatility
```

Create empty `profiles/.gitkeep`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_profiler.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fisch_macro/perception/result.py fisch_macro/controllers/recovery.py fisch_macro/profiler.py tests/test_profiler.py profiles/.gitkeep
git commit -m "feat: add result cue, recovery controller, and profile save/load"
```

---

### Task 9: Main loop + operator UI + entrypoint

**Files:**
- Create: `fisch_macro/loop.py`
- Create: `fisch_macro/ui.py`
- Create: `fisch_macro/__main__.py`
- Create: `tests/test_loop_transitions.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: all prior modules
- Produces: `BotRuntime.tick` helpers (`perceive`, `apply_bundle`, `act`); `python -m fisch_macro`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loop_transitions.py
from fisch_macro.loop import BotRuntime
from fisch_macro.perception.frame_bundle import FrameBundle
from fisch_macro.state_machine import StateMachine
from fisch_macro.types import BotState, Observation, ProfileData


def test_runtime_applies_state_machine_priority():
    rt = BotRuntime(
        state_machine=StateMachine(),
        profile=ProfileData(name="t", saved_at="2026-11-09"),
        dry_run=True,
    )
    rt.state = BotState.SHAKING
    bundle = FrameBundle(
        cast=Observation(False),
        shake=Observation(True, 0.9),
        bite=Observation(False),
        reel=Observation(True, 0.9, extras={"stable": True}),
        result=Observation(False),
        ready=Observation(False),
    )
    rt.apply_bundle(bundle, operator_stop=False)
    assert rt.state in {BotState.BITE_DETECTED, BotState.REEL_INITIALIZING, BotState.REELING}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_loop_transitions.py -v`  
Expected: FAIL with import error

- [ ] **Step 3: Write minimal implementation**

Implement `BotRuntime` in `fisch_macro/loop.py` with `start`, `stop`, `perceive`, `apply_bundle`, and `act` wiring cast/shake/reel/recovery as in the design.

Implement Tkinter `App` in `fisch_macro/ui.py` with Start, Stop/Kill, Save Profile, status label, F10 kill, and a ~33 Hz worker thread that:
1. finds Roblox window
2. grabs frame
3. perceives → transitions → acts
4. logs state changes

Create `fisch_macro/__main__.py` calling `ui.main()`.

Update README Usage:

```bash
pip install -r requirements.txt
python -m fisch_macro
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ -v`  
Expected: PASS for full suite

- [ ] **Step 5: Commit**

```bash
git add fisch_macro/loop.py fisch_macro/ui.py fisch_macro/__main__.py tests/test_loop_transitions.py README.md
git commit -m "feat: wire main fishing loop and operator UI"
```

---

### Task 10: Legacy cleanup and live validation checklist

**Files:**
- Rename: `Fisch Macro.py` → `legacy_fisch_macro.py`
- Modify: `README.md`
- Create: `docs/superpowers/plans/live-validation-checklist.md`

**Interfaces:**
- Consumes: running `python -m fisch_macro`
- Produces: documented live validation steps; legacy script demoted

- [ ] **Step 1: Write checklist file**

```markdown
# Live Validation Checklist

1. Launch Roblox Fisch windowed at a fixed spot with rod equipped
2. `python -m fisch_macro` → Start
3. Confirm cast meter release occurs without stuck hold
4. Confirm SHAKE clicks follow relocating targets and stop on bite
5. Confirm reel hold/release keeps fish roughly inside bar
6. Confirm catch result releases mouse and recasts
7. Press F10 mid-shake and mid-reel: mouse must release immediately
8. Save profile after a good session; restart and load priors
9. Confirm stop after configured consecutive failures
```

- [ ] **Step 2: Rename legacy entrypoint**

```bash
git mv "Fisch Macro.py" legacy_fisch_macro.py
```

Update README so primary usage is `python -m fisch_macro`; mention legacy file as obsolete prototype.

- [ ] **Step 3: Run full unit suite**

Run: `pytest tests/ -v`  
Expected: PASS

- [ ] **Step 4: Manual dry check**

Run UI once without Roblox to confirm Start shows recovery/no-window handling and F10 stops cleanly.

- [ ] **Step 5: Commit**

```bash
git add legacy_fisch_macro.py README.md docs/superpowers/plans/live-validation-checklist.md
git commit -m "docs: demote legacy macro and add live validation checklist"
```

---

## Spec Coverage Self-Review

| Spec requirement | Task |
|---|---|
| Modular capture/perception/state/controllers/profiler/UI | 1–9 |
| Priority state machine; result/reel outrank shake | 3, 9 |
| Structured observations | 1, 5–8 |
| Transactional adaptive SHAKE | 5 |
| Cast meter + latency-aware release | 6 |
| Safe→tighten→emergency reel control | 7 |
| Result override + ready reset | 3, 8, 9 |
| Recovery releases inputs | 8, 9 |
| Hybrid profiles save/load | 8, 9 |
| Kill hotkey / operator stop | 9, 10 |
| Logging for tuning | 4, 9 |
| No sell/travel/ML scope creep | respected throughout |
| Replace prototype runtime | 9–10 |

## Notes for implementers

- Detector heuristics in Tasks 5–8 are classical and unit-tested with synthetic images; refine thresholds after exporting real crops into `tests/fixtures/`.
- `CastController.safe_xy` default should be treated as a tunable client-relative seed.
- `detect_result` starts as a brightness heuristic; keep the `Observation` API stable when upgrading to templates/OCR.
