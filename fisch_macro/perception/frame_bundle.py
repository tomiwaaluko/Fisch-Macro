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
