import random
from collections import Counter

import numpy as np

from agents import RandomAgent
from export_view import engine_to_view, resolve_chance
from huang.action_codec import legal_action_map
from huang.engine import Color, HuangState
from huangbot.remote.adapter import ZhanguoAdapter
from huangbot.remote.view_state import build_shadow_state


def mid_game_state(players: int, decisions: int, seed: int) -> HuangState:
    rng = random.Random(seed)
    state = HuangState(players, starting_player=-1)
    resolve_chance(state, rng)
    for _ in range(decisions):
        state.apply_action(rng.choice(state.legal_actions()))
        resolve_chance(state, rng)
    return state


def context_of(state: HuangState):
    seat = state.current_actor()
    agent = RandomAgent()
    ZhanguoAdapter(agent, seat=seat, use_score_history=True).decide(engine_to_view(state, seat))
    return agent.contexts[0]


def scramble_other_seats(state: HuangState, seat: int, rng: random.Random) -> HuangState:
    """Same hand sizes, other colours, other secret points."""
    clone = state.clone()
    for other in range(clone.players):
        if other == seat:
            continue
        size = sum(clone.hands[other].values())
        clone.hands[other] = Counter(rng.choices(list(Color), k=size))
        clone.points[other] = [rng.randrange(0, 12) for _ in Color]
    return clone


def test_other_seats_secrets_do_not_change_the_context():
    for players, seed in ((2, 1), (3, 2), (4, 3)):
        state = mid_game_state(players, 30, seed)
        seat = state.current_actor()
        rng = random.Random(seed)
        base = context_of(state)
        for _ in range(3):
            other = scramble_other_seats(state, seat, rng)
            assert engine_to_view(other, seat) == engine_to_view(state, seat)
            context = context_of(other)
            assert context.observation.tobytes() == base.observation.tobytes()
            assert context.legal_action_mask.tobytes() == base.legal_action_mask.tobytes()
            assert context.legal_actions == base.legal_actions


def test_placeholder_composition_is_never_observed():
    state = mid_game_state(3, 30, seed=5)
    seat = state.current_actor()
    shadow = build_shadow_state(engine_to_view(state, seat), seat=seat)
    observation = shadow.observation_tensor(seat).copy()
    legal = list(legal_action_map(shadow))
    for color in Color:
        for other in range(shadow.players):
            if other != seat:
                size = sum(shadow.hands[other].values())
                shadow.hands[other] = Counter({color: size}) if size else Counter()
        assert np.array_equal(shadow.observation_tensor(seat), observation)
        assert list(legal_action_map(shadow)) == legal


def test_hand_sizes_of_other_seats_are_observed():
    state = mid_game_state(3, 30, seed=5)
    seat = state.current_actor()
    shadow = build_shadow_state(engine_to_view(state, seat), seat=seat)
    other = (seat + 1) % shadow.players
    before = shadow.observation_tensor(seat)
    shadow.hands[other] = Counter({Color.WHITE: sum(shadow.hands[other].values()) + 1})
    assert not np.array_equal(shadow.observation_tensor(seat), before)


def test_own_hand_and_points_are_taken_from_the_view():
    state = mid_game_state(2, 40, seed=7)
    seat = state.current_actor()
    shadow = build_shadow_state(engine_to_view(state, seat), seat=seat)
    assert shadow.hands[seat] == +state.hands[seat]
    assert shadow.points[seat] == state.points[seat]
    other = 1 - seat
    assert shadow.points[other] == [0] * 5
    assert sum(shadow.hands[other].values()) == sum(state.hands[other].values())
