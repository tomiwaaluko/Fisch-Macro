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
