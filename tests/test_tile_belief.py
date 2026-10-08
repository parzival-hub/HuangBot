from collections import Counter
import random

import numpy as np
import pytest

from huangbot.environment import HuangEnvironment
from huangbot.tile_belief import TileTracker
from huangbot.baselines import UniformRandomAgent
from huang.engine import Color, TILE_COUNTS


def test_full_game_setup_and_exact_unseen_inventory():
    environment = HuangEnvironment(starting_player=0,seed=31)
    step = environment.reset()
    engine = environment.open_spiel_state.engine
    knowledge = environment.tile_knowledge(0)
    assert not environment.short_game
    assert step.chance_events == 18
    assert knowledge.bag_total == 113
    assert knowledge.unknown_discarded == 0
    actual = Counter(engine.bag)
    actual.update(engine.hands[1])
    assert knowledge.unknown_pool == tuple(actual[color] for color in Color)
    assert sum(knowledge.draw_probabilities) == pytest.approx(1)
    assert sum(knowledge.expected_bag_counts) == pytest.approx(113)


@pytest.mark.parametrize('short_game',[False,True])
def test_hidden_world_changes_cannot_change_belief_or_samples(short_game):
    environment = HuangEnvironment(starting_player=0,seed=31,short_game=short_game)
    environment.reset()
    first = environment.open_spiel_state.engine
    if not short_game:
        # Opponent exchanged two concealed tiles. Two latent discard colours
        # produce the same public history, own hand and public bag size.
        first.public_history.append('P1 replaced 2 hidden tile(s)')
        first.bag[Color.RED] -= 2
    second = first.clone()
    second.bag[Color.RED] += 1
    second.bag[Color.BLUE] -= 1
    second.private_history[1].append('replaced 0,0,2,0,0')
    second.points[1] = [30]*5
    assert np.array_equal(first.observation_tensor(0),second.observation_tensor(0))
    a,b = TileTracker(0).observe(first),TileTracker(0).observe(second)
    assert a == b
    assert a.feature_vector() == b.feature_vector()
    for seed in range(20):
        assert a.sample(random.Random(seed)) == b.sample(random.Random(seed))


def test_permanent_ledger_survives_board_removal_and_reset():
    environment = HuangEnvironment(starting_player=0,seed=31)
    environment.reset()
    context = environment.decision_context()
    action = next(x for x in context.action_info if x.kind == 'tile')
    environment.step(action.action_id)
    engine = environment.open_spiel_state.engine
    knowledge = environment.tile_knowledge(0)
    color,cell = action.parameters
    engine._remove_tile(cell)
    assert environment.tile_knowledge(0) == knowledge
    assert knowledge.placed[color] >= 1
    environment.reset(seed=31)
    assert environment.tile_knowledge(0).placed == (7,0,0,0,0)


@pytest.mark.parametrize('players',[2,3,4])
def test_beliefs_conserve_tiles_through_complete_games_with_exchanges(players):
    environment = HuangEnvironment(players=players,starting_player=0,seed=17,max_player_decisions=None)
    step = environment.reset()
    random_agent = UniformRandomAgent(seed=12)
    replacements = 0
    while not step.done:
        context = environment.decision_context()
        for player in range(players):
            knowledge = environment.tile_knowledge(player)
            bag,hands = knowledge.sample(random.Random(23))
            assert sum(bag.values()) == environment.open_spiel_state.engine.bag_total
            assert tuple(sum(hand.values()) for hand in hands) == knowledge.hand_sizes
            assert hands[player] == environment.open_spiel_state.engine.hands[player]
        exchanges = [x for x in context.action_info if x.kind == 'replace']
        if exchanges and context.player_decisions%3 == 0:
            action = max(exchanges,key=lambda x:sum(x.parameters)).action_id
            replacements += 1
        else:
            action = random_agent.select_action(context)
        step = environment.step(action)
        assert step.player_decisions < 2000
    assert step.terminated and replacements > 0
    for player in range(players):
        assert sum(environment.tile_knowledge(player).expected_bag_counts) == pytest.approx(0)


def test_known_market_acquisition_and_hypergeometric_probabilities():
    environment = HuangEnvironment(starting_player=0,seed=31)
    environment.reset()
    engine = environment.open_spiel_state.engine
    color = engine.market[0]
    engine.market[0] = None
    engine.hands[1][color] += 1
    engine.public_history.append(f'P1 took {color.name.lower()} from market slot 0')
    knowledge = environment.tile_knowledge(0)
    assert knowledge.known_hands[1][color] == 1
    samples = [knowledge.sample(random.Random(seed)) for seed in range(1000)]
    assert all(hand[1][color] >= 1 for bag,hand in samples)
    means = np.mean([[bag[c] for c in Color] for bag,_ in samples],axis=0)
    assert np.allclose(means,knowledge.expected_bag_counts,atol=.5)
    variances = np.var([[bag[c] for c in Color] for bag,_ in samples],axis=0)
    assert np.allclose(variances,knowledge.bag_count_variances,atol=.3)


def test_neural_tile_features_are_appended_and_reset_with_the_game():
    old = HuangEnvironment(seed=31,starting_player=0,include_public_score_history=True)
    new = HuangEnvironment(seed=31,starting_player=0,include_tile_belief=True)
    old.reset()
    new.reset()
    assert new.observation_size == old.observation_size+37
    for player in range(2):
        observation = new.observation_for_player(player)
        assert np.array_equal(observation[:old.observation_size],old.observation_for_player(player))
        assert np.all(np.isfinite(observation))
        assert np.allclose(observation[-37:],new.tile_knowledge(player).feature_vector())
