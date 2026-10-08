"""Random games: engine -> game view -> shadow state -> decision -> game action -> engine."""

import os
import random
from collections import Counter

import numpy as np
import pytest

from agents import RandomAgent
from export_view import (
    compact_market,
    engine_to_view,
    game_action_to_engine,
    public_history_scores,
    resolve_chance,
)
from huang.action_codec import NUM_DISTINCT_ACTIONS, legal_action_map
from huang.engine import HuangState, Phase
from huangbot.remote.adapter import ZhanguoAdapter
from huangbot.remote.view_state import build_shadow_state

SLOW = os.environ.get("HUANGBOT_REMOTE_SLOW") == "1"
GAMES = 300 if SLOW else 25
CONFIGS = [(2, True), (2, False), (3, False), (4, False)]

SEEN = Counter()


def play_game(players: int, short_game: bool, seed: int) -> int:
    rng = random.Random(seed)
    state = HuangState(players, short_game=short_game, starting_player=-1)
    resolve_chance(state, rng)
    agent = RandomAgent(seed)
    decisions = 0
    while not state.game_over:
        actor = state.current_actor()
        view = engine_to_view(state, actor)
        reference = compact_market(state)
        reference_legal = legal_action_map(reference)

        shadow = build_shadow_state(view, seat=actor, strict=True)
        assert shadow.current_actor() == actor
        assert np.array_equal(shadow.observation_tensor(actor), reference.observation_tensor(actor))

        adapter = ZhanguoAdapter(agent, seat=actor, use_score_history=True, strict=True)
        first_call = len(agent.contexts)
        decision = adapter.decide(view)

        context = agent.contexts[first_call]
        expected_observation = np.concatenate(
            [reference.observation_tensor(actor), public_history_scores(state)]
        )
        assert np.array_equal(context.observation, expected_observation)
        assert context.observation.dtype == np.float32
        assert context.legal_action_mask.shape == (NUM_DISTINCT_ACTIONS,)
        assert set(np.flatnonzero(context.legal_action_mask)) == set(reference_legal)
        assert context.legal_actions == tuple(reference_legal)
        assert context.player == actor

        true_actions = game_action_to_engine(state, decision.game_action, view)
        if decision.game_action["type"] == "warRemove":
            cells = decision.game_action["spaces"]
            assert agent.calls - first_call == len(cells) == state.war.removals_remaining
            assert len(set(cells)) == len(cells)
            assert set(cells) <= set(view["pending"]["candidates"])
        elif decision.game_action["type"] == "market":
            # the game's market is compacted, so the slot of the true state differs
            assert game_action_to_engine(reference, decision.game_action, view) == [decision.action]
            assert true_actions[0].kind == decision.action.kind
        else:
            assert true_actions == [decision.action]
        SEEN[state.phase.value] += 1
        SEEN["kind:" + decision.game_action["type"]] += 1
        for action in true_actions:
            state.apply_action(action)
        resolve_chance(state, rng)
        decisions += 1
    return decisions


@pytest.mark.parametrize("game", range(GAMES))
def test_random_game_roundtrip(game):
    players, short_game = CONFIGS[game % len(CONFIGS)]
    assert play_game(players, short_game, seed=1000 + game) > 20


def test_every_action_kind_survives_translation_somewhere():
    """Guards the oracle: random play must reach the rare phases in slow mode."""
    if not SLOW:
        pytest.skip("coverage is only asserted in slow mode")
    for phase in (Phase.REVOLT_ATTACK, Phase.WAR_CONTRIBUTE, Phase.PAGODA_OFFER, Phase.BLUE_CONTINUE):
        assert SEEN[phase.value] > 0, phase
