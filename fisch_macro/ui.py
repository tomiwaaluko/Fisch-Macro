"""Operator window for the autonomous fishing loop."""

from __future__ import annotations

import json
import queue
import threading
import time
from datetime import date
from pathlib import Path

from fisch_macro.capture import find_roblox_window, grab_frame
from fisch_macro.input_controller import InputController
from fisch_macro.logging_util import JsonlLogger
from fisch_macro.loop import BotRuntime
from fisch_macro.state_machine import StateMachine
from fisch_macro.types import BotState, ProfileData

_TICK_HZ = 33.0
_INVALID_NAME_CHARS = '<>:"/\\|?*'


def _default_runtime() -> BotRuntime:
    profile = ProfileData(name="default", saved_at=date.today().isoformat())
    return BotRuntime(
        state_machine=StateMachine(),
        profile=profile,
        input_ctl=InputController(),
        dry_run=False,
    )


class App:
    def __init__(
        self,
        runtime: BotRuntime | None = None,
        logger: JsonlLogger | None = None,
    ) -> None:
        import tkinter as tk

        self.runtime = runtime if runtime is not None else _default_runtime()
        self.logger = logger if logger is not None else JsonlLogger(Path("logs") / "session.jsonl")
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status_queue: queue.Queue[str] = queue.Queue()
        self._status_lock = threading.Lock()
        self.root = tk.Tk()
        self.root.title("Fisch Macro")
        self.status = tk.StringVar(value="Idle")
        self.profile_name = tk.StringVar(value=self.runtime.profile.name)
        tk.Button(self.root, text="Start", command=self.start).pack(fill="x", padx=8, pady=4)
        tk.Button(self.root, text="Stop/Kill", command=self.stop).pack(fill="x", padx=8, pady=4)
        tk.Label(self.root, text="Profile").pack(anchor="w", padx=8)
        tk.Entry(self.root, textvariable=self.profile_name).pack(fill="x", padx=8)
        tk.Button(self.root, text="Save Profile", command=self.save_profile).pack(fill="x", padx=8, pady=4)
        tk.Button(self.root, text="Load Profile", command=self.load_profile).pack(fill="x", padx=8, pady=4)
        tk.Label(self.root, textvariable=self.status).pack(padx=8, pady=8)
        self.root.bind("<F10>", self.stop)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._pump_status()

    def start(self) -> None:
        self._stop_event.clear()
        self.runtime.start()
        self._set_status(self.runtime.state.name)
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, name="fisch-loop", daemon=True)
            self._thread.start()

    def stop(self, _event=None) -> str:
        self._stop_event.set()
        self.runtime.stop()
        self._set_status("Stopped")
        return "break"

    def _clean_profile_name(self, raw: str | None) -> str:
        text = (raw or self.runtime.profile.name).strip() or "default"
        name = "".join("-" if char in _INVALID_NAME_CHARS else char for char in text).strip(" .")
        if name.lower().endswith(".json"):
            name = name[:-5]
        return name or "default"

    def _profile_path(self, name: str) -> Path:
        return Path("profiles") / f"{name}.json"

    def save_profile(self) -> None:
        name = self._clean_profile_name(self.profile_name.get())
        self.runtime.profile.name = name
        self.profile_name.set(name)
        path = self._profile_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.runtime.profiler.save(path)
        self._set_status(f"Saved {path.as_posix()}")

    def load_profile(self) -> None:
        name = self._clean_profile_name(self.profile_name.get())
        path = self._profile_path(name)
        if not path.is_file():
            self._set_status(f"Missing {path.as_posix()}")
            return
        try:
            self.runtime.load_profile(path)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self._set_status(f"Load failed: {exc}")
            return
        self.profile_name.set(self.runtime.profile.name)
        self._set_status(f"Loaded {path.as_posix()}")

    def _worker(self) -> None:
        period = 1.0 / _TICK_HZ
        while not self._stop_event.is_set():
            started = time.perf_counter()
            try:
                self._tick()
            except Exception as exc:
                try:
                    self.logger.log("error", message=str(exc))
                    self._set_status(f"Error: {exc}")
                finally:
                    self.runtime.release_after_error()
            remaining = period - (time.perf_counter() - started)
            if remaining > 0 and not self._stop_event.is_set():
                time.sleep(remaining)

    def _tick(self) -> None:
        if self._stop_event.is_set():
            return
        window = find_roblox_window()
        if window is None:
            self._recover_window()
            return
        frame = grab_frame(window)
        if self._stop_event.is_set() or not self.runtime._running:
            return
        bundle = self.runtime.perceive(frame)
        if find_roblox_window() != window:
            self._recover_window()
            return
        self.runtime.note_window_present()
        previous = self.runtime.state
        self.runtime.apply_bundle(bundle, operator_stop=False)
        self._log_state(previous)
        if not self._stop_event.is_set() and not self.runtime._running:
            self._stop_event.set()
            self._set_status("Stopped after consecutive catch failures")
            return
        if self._stop_event.is_set() or not self.runtime._running:
            return
        if find_roblox_window() != window:
            self._recover_window()
            return
        previous = self.runtime.state
        self.runtime.act(bundle, frame)
        self._log_state(previous)
        self._set_status(self.runtime.state.name)

    def _recover_window(self) -> None:
        previous = self.runtime.state
        tripped = self.runtime.enter_recovery()
        self._log_state(previous)
        if tripped:
            self._stop_event.set()
            self._set_status("Stopped: Roblox window unavailable too long")
        else:
            self._set_status("Waiting for the active Roblox window")

    def _log_state(self, previous: BotState) -> None:
        current = self.runtime.state
        if current != previous:
            self.logger.log("state", from_state=previous.name, to_state=current.name)

    def _set_status(self, text: str) -> None:
        with self._status_lock:
            if self._stop_event.is_set() and not text.startswith("Stopped"):
                return
            if text.startswith("Stopped"):
                try:
                    while True:
                        self._status_queue.get_nowait()
                except queue.Empty:
                    pass
            self._status_queue.put(text)

    def _pump_status(self) -> None:
        latest = None
        try:
            while True:
                latest = self._status_queue.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            try:
                self.status.set(latest)
            except Exception:
                return
        try:
            self.root.after(30, self._pump_status)
        except Exception:
            return

    def _hook_f10(self) -> None:
        try:
            import keyboard

            keyboard.add_hotkey("f10", self.stop)
        except Exception as exc:
            self.logger.log("hotkey", message=f"F10 global hook unavailable: {exc}")

    def _unhook_f10(self) -> None:
        try:
            import keyboard

            keyboard.remove_hotkey("f10")
        except Exception:
            return

    def _on_close(self) -> None:
        self.stop()
        self._unhook_f10()
        self.root.destroy()

    def run(self) -> None:
        self._hook_f10()
        self.root.mainloop()


def main() -> None:
    App().run()
