import numpy as np
import pytest

from agents import FirstAgent, RandomAgent
from export_view import engine_to_view, game_action_to_engine
from handstates import war_state
from huang.action_codec import WAR_REMOVE_BASE, legal_action_map
from huang.engine import Phase
from huangbot.remote.adapter import ZhanguoAdapter
from huangbot.remote.protocol import NoLegalActionError, ShadowStateError


def decide(state, agent, seat=0):
    view = engine_to_view(state, seat)
    adapter = ZhanguoAdapter(agent, seat=seat, use_score_history=True, strict=True)
    return view, adapter.decide(view)


def test_two_removals_are_combined_into_one_action_with_two_agent_calls():
    state, red_cells = war_state(Phase.WAR_REMOVE, red_in_winner=3, removals=2)
    agent = RandomAgent(seed=3)
    view, decision = decide(state, agent)

    cells = decision.game_action["spaces"]
    assert decision.game_action["type"] == "warRemove"
    assert len(cells) == 2 and len(set(cells)) == 2
    assert set(cells) <= set(red_cells)
    assert agent.calls == 2
    # the second decision saw the board after the first removal
    first, second = agent.contexts
    assert not np.array_equal(first.observation, second.observation)
    assert not second.legal_action_mask[WAR_REMOVE_BASE + cells[0]]
    assert len(second.legal_actions) == len(first.legal_actions) - 1
    # the game's one-shot action is replayable on the real engine, one tile at a time
    for action in game_action_to_engine(state, decision.game_action, view):
        state.apply_action(action)
    assert state.war is None


def test_first_agent_removes_the_two_lowest_cells():
    state, red_cells = war_state(Phase.WAR_REMOVE, red_in_winner=3, removals=2)
    agent = FirstAgent()
    _, decision = decide(state, agent)
    assert decision.game_action["spaces"] == red_cells[:2]


def test_single_removal_needs_only_one_call():
    state, _ = war_state(Phase.WAR_REMOVE, red_in_winner=2, removals=1)
    agent = RandomAgent()
    _, decision = decide(state, agent)
    assert len(decision.game_action["spaces"]) == 1
    assert agent.calls == 1


def test_more_removals_than_candidates_is_an_error_not_a_short_action():
    state, _ = war_state(Phase.WAR_REMOVE, red_in_winner=2, removals=3)
    view = engine_to_view(state, 0)
    assert view["pending"]["count"] == 2  # the oracle's game clamps to the candidates
    view["pending"]["count"] = 3  # a game that does not would ask for more than it offers
    adapter = ZhanguoAdapter(RandomAgent(), seat=0, use_score_history=True)
    with pytest.raises(NoLegalActionError):
        adapter.decide(view)


def test_exclusion_applies_to_every_removal_step():
    state, red_cells = war_state(Phase.WAR_REMOVE, red_in_winner=3, removals=2)
    view = engine_to_view(state, 0)
    ids = {a.parameters[0]: i for i, a in legal_action_map(state).items()}
    adapter = ZhanguoAdapter(FirstAgent(), seat=0, use_score_history=True)
    decision = adapter.decide(view, exclude=frozenset({ids[red_cells[0]]}))
    assert decision.game_action["spaces"] == red_cells[1:]
    with pytest.raises(NoLegalActionError):
        adapter.decide(view, exclude=frozenset(ids[c] for c in red_cells[:2]))


def test_tie_break_is_a_single_war_winner_decision():
    state, _ = war_state(Phase.WAR_TIE, strengths=[2, 2])
    agent = FirstAgent()
    view, decision = decide(state, agent)
    assert decision.game_action == {"type": "warTie", "side": 0}
    assert game_action_to_engine(state, decision.game_action, view) == [decision.action]
    assert agent.calls == 1


def test_war_pending_without_war_data_is_a_shadow_error():
    state, _ = war_state(Phase.WAR_TIE, strengths=[2, 2])
    view = engine_to_view(state, 0)
    view["war"] = None
    adapter = ZhanguoAdapter(RandomAgent(), seat=0, use_score_history=True)
    with pytest.raises(ShadowStateError):
        adapter.decide(view)
