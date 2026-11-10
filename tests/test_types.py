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
