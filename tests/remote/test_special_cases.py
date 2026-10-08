import pytest

from action_schema import is_valid_game_action
from agents import FirstAgent
from export_view import engine_to_view, game_action_to_engine
from handstates import blank_state, state_with_two_yellow_pagodas
from huang.engine import Action, Color, Phase
from huangbot.remote.adapter import ZhanguoAdapter, translate_action
from huangbot.remote.fallback import safe_action
from huangbot.remote.protocol import TranslationError


class ChoiceAgent:
    """Chooses the legal action whose ``(kind, parameters)`` matches ``wanted``."""

    def __init__(self, wanted_kind, wanted_parameters):
        self.wanted = (wanted_kind, wanted_parameters)
        self.calls = 0

    def select_action(self, context):
        self.calls += 1
        for info in context.action_info:
            if (info.kind, info.parameters) == self.wanted:
                return info.action_id
        raise AssertionError(f"{self.wanted} is not legal")

    def reset_episode(self):
        pass


def decide(state, agent, seat=0):
    view = engine_to_view(state, seat)
    adapter = ZhanguoAdapter(agent, seat=seat, use_score_history=True, strict=True)
    return view, adapter.decide(view)


def pagoda_ids(view):
    return {tuple(e["tri"]): e["id"] for e in view["pagodas"] if e["tri"]}


@pytest.mark.parametrize("payment", [1, 2])
def test_paid_pagoda_moves_a_pagoda_when_the_supply_is_empty(payment):
    state, first, second, third = state_with_two_yellow_pagodas()
    view, decision = decide(state, ChoiceAgent("paid_pagoda", (payment, third, second)))
    assert decision.game_action == {
        "type": "pagoda",
        "tri": list(third),
        "useLeader": payment == 1,
        "source": pagoda_ids(view)[second],
    }
    assert game_action_to_engine(state, decision.game_action, view) == [decision.action]


def test_pagoda_offer_with_source():
    state, first, second, third = state_with_two_yellow_pagodas()
    state.phase = Phase.PAGODA_OFFER
    state.actions_remaining = 1
    state.post_placement = {"cell": third[0], "color": int(Color.YELLOW), "candidate_triangles": (third,)}
    view, decision = decide(state, ChoiceAgent("pagoda_place", (third, first)))
    assert view["pending"]["sources"] == [pagoda_ids(view)[first], pagoda_ids(view)[second]]
    assert decision.game_action == {"type": "pagodaBuild", "tri": list(third), "source": pagoda_ids(view)[first]}
    assert game_action_to_engine(state, decision.game_action, view) == [decision.action]


def test_pagoda_offer_with_free_supply_has_no_source():
    state, first, second, third = state_with_two_yellow_pagodas()
    del state.pagodas[second]
    state.phase = Phase.PAGODA_OFFER
    state.actions_remaining = 1
    state.post_placement = {"cell": third[0], "color": int(Color.YELLOW), "candidate_triangles": (third,)}
    view, decision = decide(state, ChoiceAgent("pagoda_place", (third, None)))
    assert view["pending"]["sources"] == []
    assert decision.game_action == {"type": "pagodaBuild", "tri": list(third)}


def test_turn_actions_are_valid_game_actions_and_invert():
    state, first, second, third = state_with_two_yellow_pagodas()
    view = engine_to_view(state, 0)
    for action in state.legal_actions():
        game_action = translate_action(action, view)
        assert is_valid_game_action(game_action), game_action
        assert game_action_to_engine(state, game_action, view) == [action]


@pytest.mark.parametrize(
    "kind, parameters, expected",
    [
        ("leader", (1, None), {"type": "leader", "color": "red", "to": None}),
        ("leader", (4, 17), {"type": "leader", "color": "white", "to": 17}),
        ("tile", (3, 40), {"type": "tile", "color": "green", "space": 40}),
        ("riot", (1, 9), {"type": "riot", "space": 9, "useLeader": True}),
        ("riot", (2, 9), {"type": "riot", "space": 9, "useLeader": False}),
        ("replace", (2, 0, 1, 0, 3), {"type": "replace", "tiles": ["yellow", "yellow", "blue", "white", "white", "white"]}),
        ("market_decline", (), {"type": "market", "index": None}),
        ("market_take", (3,), {"type": "market", "index": 3}),
        ("pagoda_decline", (), {"type": "pagodaBuild", "tri": None}),
        ("blue_stop", (), {"type": "chain", "space": None}),
        ("blue_tile", (12,), {"type": "chain", "space": 12}),
        ("revolt_commit", (2, 1), {"type": "revoltCommit", "count": 2, "useLeader": True}),
        ("revolt_commit", (0, 0), {"type": "revoltCommit", "count": 0, "useLeader": False}),
        ("war_pass", (), {"type": "warCommit", "side": None, "count": 0}),
        ("war_commit", (1, 3, 1), {"type": "warCommit", "side": 1, "count": 3, "useLeader": True}),
        ("war_commit", (0, 0, 1), {"type": "warCommit", "side": 0, "count": 0, "useLeader": True}),
        ("war_winner", (1,), {"type": "warTie", "side": 1}),
    ],
)
def test_translation_table(kind, parameters, expected):
    view = engine_to_view(blank_state(), 0)
    assert translate_action(Action(kind, parameters), view) == expected
    assert is_valid_game_action(expected)


def test_moving_a_pagoda_that_is_not_on_the_board_is_a_translation_error():
    state, first, second, third = state_with_two_yellow_pagodas()
    view = engine_to_view(state, 0)
    action = Action("pagoda_place", (third, (1, 2, 3)))
    with pytest.raises(TranslationError):
        translate_action(action, view)
    with pytest.raises(TranslationError):
        translate_action(Action("war_remove", (3,)), view)
    with pytest.raises(TranslationError):
        translate_action(Action("nonsense", ()), view)


def test_the_first_action_of_a_fresh_turn_is_legal_for_every_kind_of_hand():
    state = blank_state()
    view, decision = decide(state, FirstAgent())
    assert is_valid_game_action(decision.game_action)
    assert decision.phase == "turn"


@pytest.mark.parametrize(
    "pending, expected",
    [
        (None, {"type": "replace", "tiles": []}),
        ({"kind": "pagodaBuild"}, {"type": "pagodaBuild", "tri": None}),
        ({"kind": "market"}, {"type": "market", "index": None}),
        ({"kind": "chain"}, {"type": "chain", "space": None}),
        ({"kind": "revolt"}, {"type": "revoltCommit", "count": 0}),
        ({"kind": "war"}, {"type": "warCommit", "side": None, "count": 0}),
        ({"kind": "warTie", "tied": [2, 1]}, {"type": "warTie", "side": 2}),
        ({"kind": "warRemove", "count": 2, "candidates": [5, 6, 7]}, {"type": "warRemove", "spaces": [5, 6]}),
    ],
)
def test_fallback_actions(pending, expected):
    action = safe_action({"pending": pending})
    assert action == expected
    assert is_valid_game_action(action)


def test_fallback_actions_are_legal_in_the_engine_where_the_engine_has_the_move():
    state, first, second, third = state_with_two_yellow_pagodas()
    state.phase = Phase.PAGODA_OFFER
    state.post_placement = {"cell": third[0], "color": int(Color.YELLOW), "candidate_triangles": (third,)}
    view = engine_to_view(state, 0)
    assert game_action_to_engine(state, safe_action(view), view)[0].kind == "pagoda_decline"
