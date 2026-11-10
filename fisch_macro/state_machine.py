from __future__ import annotations

from fisch_macro.perception.frame_bundle import FrameBundle
from fisch_macro.types import BotState


class StateMachine:
    def transition(
        self,
        state: BotState,
        bundle: FrameBundle,
        operator_stop: bool,
        cast_holding: bool = False,
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

        # Ownership stays with the cast hold. A missing meter or a shake
        # circle must not drop CASTING while the button is still down.
        # Result, reel, and bite already returned above.
        if cast_holding and state == BotState.CASTING:
            return BotState.CASTING

        if bundle.shake.found:
            return BotState.SHAKING

        # Meter presence must not skip the states that actually press LMB.
        if state == BotState.RESET_TO_CAST:
            if bundle.ready.found:
                return BotState.PRE_CAST
            return BotState.RESET_TO_CAST

        if state == BotState.PRE_CAST:
            return BotState.PRE_CAST

        if state == BotState.RECOVERY:
            if bundle.ready.found:
                return BotState.PRE_CAST
            return BotState.RECOVERY

        if bundle.cast.found:
            return BotState.CASTING

        if state == BotState.CATCH_RESULT:
            if bundle.ready.found:
                return BotState.RESET_TO_CAST
            return BotState.CATCH_RESULT

        if state in {BotState.CASTING, BotState.WAITING_SHAKE} and not bundle.shake.found:
            return BotState.WAITING_SHAKE

        return state
