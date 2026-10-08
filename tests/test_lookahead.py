from collections import Counter
import random
from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")

from huangbot.environment import HuangEnvironment
from huangbot.lookahead import (
    InformationSetLookaheadAgent,
    LookaheadConfig,
    determinize_hidden_tiles,
    _model_observation,
)
from huangbot.model import HuangActorCritic, ModelConfig


def _small_model(environment):
    return HuangActorCritic(
        ModelConfig(
            observation_size=environment.observation_size,
            num_actions=environment.num_actions,
            encoder_width=32,
            encoder_layers=1,
            recurrent_hidden_size=16,
            recurrent_layers=1,
        )
    )


def test_determinization_preserves_public_state_observer_hand_and_unseen_pool():
    environment = HuangEnvironment(starting_player=0, seed=31)
    environment.reset()
    engine = environment.open_spiel_state.engine
    original_unseen = Counter(engine.bag)
    original_unseen.update(engine.hands[1])
    sampled = determinize_hidden_tiles(engine, 0, random.Random(7))
    sampled_unseen = Counter(sampled.bag)
    sampled_unseen.update(sampled.hands[1])

    assert sampled is not engine
    assert sampled.tiles == engine.tiles
    assert sampled.market == engine.market
    assert sampled.hands[0] == engine.hands[0]
    assert sum(sampled.hands[1].values()) == sum(engine.hands[1].values())
    assert sampled_unseen == original_unseen


def test_simulated_tile_features_respect_concealed_discards_for_every_seat():
    environment = HuangEnvironment(starting_player=0,seed=31,include_tile_belief=True)
    environment.reset()
    model = _small_model(environment)
    engine = environment.open_spiel_state.engine
    from huang.engine import Color
    engine.public_history.append('P1 replaced 2 hidden tile(s)')
    engine.bag[Color.RED] -= 2
    engine.private_history[1].append('replaced 0,2,0,0,0')
    for seed in range(5):
        sampled = determinize_hidden_tiles(engine,0,random.Random(seed))
        for player in range(2):
            observation = _model_observation(model,sampled,player)
            assert observation.shape == (model.config.observation_size,)
            assert torch.isfinite(torch.tensor(observation)).all()
    context = environment.decision_context()
    agent = InformationSetLookaheadAgent(model,players=2,
        config=LookaheadConfig(depth=2,top_k=2,simulations=2,adaptive=False),seed=8)
    assert agent.select_action_from_environment(context,environment) in context.legal_actions


def test_information_set_lookahead_returns_a_legal_action_with_batched_leaves():
    environment = HuangEnvironment(starting_player=0, seed=32)
    environment.reset()
    model = _small_model(environment)
    agent = InformationSetLookaheadAgent(
        model,
        players=2,
        config=LookaheadConfig(depth=2, top_k=2, simulations=2),
        seed=9,
    )
    context = environment.decision_context()
    action = agent.select_action_from_environment(context, environment)

    assert action in context.legal_actions
    assert agent.last_diagnostics is not None
    assert len(agent.last_diagnostics.candidates) == 2
    assert agent.last_diagnostics.selected_action == action


def test_adaptive_budget_increases_for_endgame_uncertainty_and_conflicts():
    environment = HuangEnvironment(starting_player=0,seed=32)
    environment.reset()
    knowledge = environment.tile_knowledge(0)
    config = LookaheadConfig(depth=2,simulations=2)
    early = config.budget(knowledge)
    middle = config.budget(replace(knowledge,bag_total=30))
    late = config.budget(replace(knowledge,bag_total=10))
    assert early[0] < middle[0] < late[0]
    assert early[1] < middle[1] < late[1]
    assert config.budget(knowledge,conflict=True)[1] >= 16
    certain = replace(knowledge,unknown_pool=(10,0,0,0,0),bag_total=10)
    assert config.budget(certain)[1] < late[1]


def test_determinization_does_not_copy_opponent_private_knowledge():
    environment = HuangEnvironment(starting_player=0,seed=31)
    environment.reset()
    engine = environment.open_spiel_state.engine
    engine.points[1] = [999]*5
    sampled = determinize_hidden_tiles(engine,0,random.Random(7))
    assert sampled.private_history[1] == []
    assert sampled.points[1] == [0]*5
    assert sampled.private_history[0] == engine.private_history[0]


def test_rollout_finishes_revolt_before_value_evaluation(monkeypatch):
    from huang import board
    from huang.engine import Leader,Color,Phase
    environment = HuangEnvironment(starting_player=0,seed=31)
    environment.reset()
    engine = environment.open_spiel_state.engine
    capital = board.CAPITAL_CELLS[0]
    neighbors = [cell for cell in board.NEIGHBORS[capital]
                 if cell not in board.RIVER_CELLS and not engine.occupied(cell)]
    assert len(neighbors) >= 2
    engine.leaders[neighbors[0]] = Leader(1,Color.BLUE)
    action = next(x for x in engine.legal_actions()
                  if x.kind == 'leader' and x.parameters == (int(Color.BLUE),neighbors[1]))
    engine.apply_action(action)
    assert engine.phase == Phase.REVOLT_ATTACK
    model = _small_model(environment)
    agent = InformationSetLookaheadAgent(model,players=2,config=LookaheadConfig(depth=1,simulations=1,adaptive=False))
    sampled = engine.clone()
    original = model.predict_values
    def stable_values(*args,**kwargs):
        assert sampled.phase == Phase.TURN
        assert sampled.revolt is None
        return original(*args,**kwargs)
    monkeypatch.setattr(model,'predict_values',stable_values)
    values,leaves,extended = agent._rollout([sampled],[[model.initial_hidden(1),model.initial_hidden(1)]],0,1)
    assert leaves == 1 and extended >= 2
    assert sampled.revolt is None


def test_time_budget_completes_equal_samples_for_every_candidate():
    environment = HuangEnvironment(starting_player=0,seed=32)
    environment.reset()
    model = _small_model(environment)
    agent = InformationSetLookaheadAgent(model,players=2,
        config=LookaheadConfig(top_k=2,simulations=20,max_time_seconds=1e-9))
    context = environment.decision_context()
    assert agent.select_action_from_environment(context,environment) in context.legal_actions
    assert agent.last_diagnostics.simulations_per_candidate == 1
    assert agent.last_diagnostics.time_budget_reached


def test_search_decision_is_identical_for_indistinguishable_hidden_worlds():
    from huang.engine import Color
    environments = [HuangEnvironment(starting_player=0,seed=31) for _ in range(2)]
    for environment in environments:
        environment.reset()
        engine = environment.open_spiel_state.engine
        engine.public_history.append('P1 replaced 2 hidden tile(s)')
        engine.bag[Color.RED] -= 2
    second = environments[1].open_spiel_state.engine
    second.bag[Color.RED] += 1
    second.bag[Color.BLUE] -= 1
    second.private_history[1].append('replaced 0,0,2,0,0')
    second.points[1] = [999]*5
    model = _small_model(environments[0])
    diagnostics = []
    for environment in environments:
        agent = InformationSetLookaheadAgent(model,players=2,seed=19,
            config=LookaheadConfig(depth=2,top_k=2,simulations=2,adaptive=False))
        context = environment.decision_context()
        agent.select_action_from_environment(context,environment)
        diagnostics.append(agent.last_diagnostics)
    assert diagnostics[0].selected_action == diagnostics[1].selected_action
    assert diagnostics[0].mean_values == diagnostics[1].mean_values


def test_rollout_finishes_war_contributions_tie_and_removal(monkeypatch):
    from huang import board
    from huang.engine import Leader,Color,Phase
    environment = HuangEnvironment(starting_player=0,seed=31)
    environment.reset()
    engine = environment.open_spiel_state.engine
    cell = board.COORD_TO_INDEX
    engine.tiles = {cell[(6,2)]:Color.YELLOW,cell[(6,1)]:Color.RED,
                    cell[(6,4)]:Color.YELLOW,cell[(6,5)]:Color.RED}
    engine.leaders = {cell[(7,1)]:Leader(0,Color.BLUE),cell[(7,4)]:Leader(1,Color.BLUE)}
    engine.hands = [Counter({Color.WHITE:1}),Counter()]
    action = next(x for x in engine.legal_actions()
                  if x.kind == 'tile' and x.parameters == (int(Color.WHITE),cell[(6,3)]))
    engine.apply_action(action)
    assert engine.phase == Phase.WAR_CONTRIBUTE
    model = _small_model(environment)
    agent = InformationSetLookaheadAgent(model,players=2,config=LookaheadConfig(depth=1,adaptive=False))
    from huang.action_codec import legal_action_map
    original_actions = model.select_actions
    seen_phases = []
    def pass_then_resolve(*args,**kwargs):
        seen_phases.append(engine.phase)
        kind = {Phase.WAR_CONTRIBUTE:'war_pass',Phase.WAR_TIE:'war_winner',
                Phase.WAR_REMOVE:'war_remove'}[engine.phase]
        action_id = next(key for key,action in legal_action_map(engine).items() if action.kind == kind)
        selection = original_actions(*args,**kwargs)
        return replace(selection,actions=(action_id,))
    monkeypatch.setattr(model,'select_actions',pass_then_resolve)
    original = model.predict_values
    def stable_values(*args,**kwargs):
        assert engine.phase == Phase.TURN and engine.war is None
        return original(*args,**kwargs)
    monkeypatch.setattr(model,'predict_values',stable_values)
    values,leaves,extended = agent._rollout([engine],[[model.initial_hidden(1),model.initial_hidden(1)]],0,1)
    assert leaves == 1 and extended >= 4
    assert engine.war is None
    assert Phase.WAR_TIE in seen_phases and Phase.WAR_REMOVE in seen_phases
