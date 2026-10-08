"""The bundled model chooses exactly what it would choose inside HuangEnvironment."""

import random

import numpy as np

from export_view import compact_market, engine_to_view, game_action_to_engine, public_history_scores, resolve_chance
from huang.action_codec import NUM_DISTINCT_ACTIONS, legal_action_map
from huang.engine import HuangState
from huangbot.environment import ActionInfo, DecisionContext
from huangbot.model import RecurrentModelAgent
from huangbot.remote.adapter import ZhanguoAdapter


def native_context(state: HuangState, seat: int) -> DecisionContext:
    """What ``HuangEnvironment.decision_context()`` builds (without OpenSpiel)."""
    legal = legal_action_map(state)
    mask = np.zeros(NUM_DISTINCT_ACTIONS, dtype=np.bool_)
    mask[list(legal)] = True
    observation = np.concatenate([state.observation_tensor(seat), public_history_scores(state)]).astype(np.float32)
    return DecisionContext(
        player=seat,
        observation=observation,
        legal_action_mask=mask,
        legal_actions=tuple(legal),
        action_info=tuple(ActionInfo(i, a.kind, a.parameters, str(a)) for i, a in legal.items()),
        public_snapshot={},
        player_decisions=0,
    )


def test_adapter_and_native_environment_drive_the_model_identically(real_model):
    rng = random.Random(21)
    state = HuangState(2, short_game=True, starting_player=-1)
    resolve_chance(state, rng)
    native = RecurrentModelAgent(real_model, players=2, deterministic=True)
    remote = RecurrentModelAgent(real_model, players=2, deterministic=True)
    adapters = [ZhanguoAdapter(remote, seat=seat, use_score_history=True) for seat in (0, 1)]
    compared = 0
    while not state.game_over and compared < 120:
        seat = state.current_actor()
        # the game compacts its market, so the native side is shown the compacted market too
        reference = compact_market(state)
        expected = native.select_action(native_context(reference, seat))
        view = engine_to_view(state, seat)
        decision = adapters[seat].decide(view)
        assert decision.action_id == expected
        for player in (0, 1):
            assert np.array_equal(native.memory.get(player).numpy(), remote.memory.get(player).numpy())
        for action in game_action_to_engine(state, decision.game_action, view):
            state.apply_action(action)
        resolve_chance(state, rng)
        compared += 1
    assert compared >= 60
