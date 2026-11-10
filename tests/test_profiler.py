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
