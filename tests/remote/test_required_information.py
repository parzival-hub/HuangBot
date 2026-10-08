import copy
import random

import numpy as np
import pytest

from agents import FirstAgent
from export_view import compact_market, engine_to_view, resolve_chance
from huang.action_codec import legal_action_map
from huang.engine import HuangState
from huangbot.model import RecurrentModelAgent, HuangActorCritic, ModelConfig
from huangbot.remote.adapter import ZhanguoAdapter
from huangbot.remote.protocol import GameInformationError
from huangbot.remote.runner import RemoteRunner, RunnerConfig
from huangbot.tile_belief import TileTracker
from mock_server import MockZhanguo
from test_runner_logic import FakeClient


def fresh_state():
    state = HuangState(2, starting_player=0)
    resolve_chance(state, random.Random(47))
    return state


def tile_adapter(seat=0):
    return ZhanguoAdapter(FirstAgent(), seat=seat, use_score_history=True, use_tile_belief=True)


@pytest.mark.parametrize('players', [2, 3, 4])
def test_complete_game_tile_features_match_training_even_after_exchanges(players):
    state = HuangState(players, short_game=False, starting_player=0)
    rng = random.Random(19)
    resolve_chance(state, rng)
    replacements = 0
    adapters = [tile_adapter(seat) for seat in range(players)]
    decisions = 0
    while not state.game_over:
        seat = state.current_actor()
        view = engine_to_view(state, seat)
        shadow = compact_market(state)
        knowledge = TileTracker(seat).observe(state)
        context = adapters[seat].make_context(shadow, view)
        assert context.observation.shape == (5467,)
        np.testing.assert_array_equal(context.observation[-37:], np.asarray(knowledge.feature_vector(), dtype=np.float32))
        assert context.tile_knowledge == knowledge
        actions = state.legal_actions()
        replace = [action for action in actions if action.kind == 'replace']
        if replace and decisions % 3 == 0:
            action = max(replace, key=lambda a: sum(a.parameters))
            replacements += 1
        else:
            action = rng.choice(actions)
        state.apply_action(action)
        resolve_chance(state, rng)
        decisions += 1
        assert decisions < 2000
    assert replacements > 0


@pytest.mark.parametrize('field', ['tiles', 'leaders', 'pagodas', 'market', 'players', 'bagCount',
    'current', 'turn', 'actionsLeft', 'inAction', 'pending', 'log', 'logComplete', 'options', 'tileHistory'])
def test_missing_information_is_a_fatal_typed_error(field):
    view = engine_to_view(fresh_state(), 0)
    del view[field]
    adapter = tile_adapter()
    with pytest.raises(GameInformationError, match=field):
        adapter.decide(view)
    assert adapter.agent.calls == 0


@pytest.mark.parametrize('field', ['hand', 'vp', 'handCount'])
def test_missing_own_player_information_is_rejected(field):
    view = engine_to_view(fresh_state(), 0)
    del view['players'][0][field]
    with pytest.raises(GameInformationError, match=field):
        tile_adapter().decide(view)


@pytest.mark.parametrize('history', ['logComplete', 'tileHistory'])
def test_incomplete_history_is_rejected(history):
    view = engine_to_view(fresh_state(), 0)
    if history == 'logComplete':
        view[history] = False
    else:
        view[history]['complete'] = False
    with pytest.raises(GameInformationError, match='complete|Complete'):
        tile_adapter().decide(view)


def state_after_replacement(player):
    state = fresh_state()
    state.active_player = player
    action = max((a for a in state.legal_actions() if a.kind == 'replace'), key=lambda a: sum(a.parameters))
    state.apply_action(action)
    resolve_chance(state, random.Random(17))
    return state


def test_missing_own_replacement_colours_cannot_be_guessed():
    state = state_after_replacement(0)
    view = engine_to_view(state, 0)
    event = next(e for e in view['tileHistory']['events'] if e['kind'] == 'replace')
    del event['colors']
    with pytest.raises(GameInformationError, match='colors'):
        tile_adapter().decide(view)


def test_opponent_secrets_are_never_required_or_used():
    state = state_after_replacement(1)
    view = engine_to_view(state, 0)
    event = next(e for e in view['tileHistory']['events'] if e['kind'] == 'replace')
    assert 'colors' not in event
    adapter = tile_adapter()
    before = adapter.make_context(compact_market(state), view).observation.copy()
    # Even an incorrectly unfiltered server response must not influence beliefs.
    event['colors'] = ['white'] * 6
    view['players'][1]['hand'] = ['yellow'] * 6
    view['players'][1]['vp'] = {'red': 999}
    after = adapter.make_context(compact_market(state), view).observation
    np.testing.assert_array_equal(before, after)


def test_missing_past_tile_event_is_detected_by_inventory_conservation():
    state = fresh_state()
    state.apply_action(next(a for a in state.legal_actions() if a.kind == 'tile'))
    view = engine_to_view(state, 0)
    view['tileHistory']['events'] = []
    with pytest.raises(GameInformationError, match='ledger'):
        tile_adapter().make_context(compact_market(state), view)


def test_received_history_cannot_be_rewritten_and_reset_clears_it():
    state = fresh_state()
    state.apply_action(next(a for a in state.legal_actions() if a.kind == 'tile'))
    view = engine_to_view(state, 0)
    adapter = tile_adapter()
    adapter.make_context(compact_market(state), view)
    changed = copy.deepcopy(view)
    changed['tileHistory']['events'][0]['space'] = (changed['tileHistory']['events'][0]['space'] + 1) % 203
    with pytest.raises(GameInformationError, match='changed'):
        adapter.make_context(compact_market(state), changed)
    adapter.reset_episode()
    adapter.make_context(compact_market(state), view)


def test_missing_score_awards_are_not_silently_zero_filled():
    view = engine_to_view(fresh_state(), 0)
    view['players'][0]['vp']['red'] = 1
    with pytest.raises(GameInformationError, match='own points'):
        tile_adapter().decide(view)


def test_model_8000_is_supported_and_missing_ledger_exits_without_any_action(real_model, caplog):
    mock = MockZhanguo(short_game=False, seed=1)
    mock.start()
    state = mock.ext_state()
    del state['view']['tileHistory']
    client = FakeClient([state])
    runner = RemoteRunner(client, real_model, RunnerConfig(interval=0, min_think=0))
    assert runner.use_tile_belief and runner.use_score_history
    assert runner.run() == 2
    assert client.actions == []
    assert runner.fallbacks == 0 and runner.accepted == 0
    assert 'tileHistory' in caplog.text


def test_missing_private_replacement_data_is_never_sent_as_a_fallback(real_model, caplog):
    mock = MockZhanguo(short_game=False, seed=1)
    mock.start()
    state = mock.ext_state()
    replace = max((a for a in mock.state.legal_actions() if a.kind == 'replace'), key=lambda a: sum(a.parameters))
    mock.state.apply_action(replace)
    resolve_chance(mock.state, mock.rng)
    state = mock.ext_state()
    event = next(e for e in state['view']['tileHistory']['events'] if e['kind'] == 'replace')
    del event['colors']
    client = FakeClient([state])
    runner = RemoteRunner(client, real_model, RunnerConfig(interval=0, min_think=0))
    assert runner.run() == 2
    assert client.actions == [] and runner.fallbacks == 0
    assert 'colors' in caplog.text


def test_adapter_can_join_a_game_late_with_full_history(real_model):
    state = fresh_state()
    rng = random.Random(11)
    for _ in range(20):
        if state.game_over:
            break
        state.apply_action(rng.choice(state.legal_actions()))
        resolve_chance(state, rng)
    seat = state.current_actor()
    agent = RecurrentModelAgent(real_model, players=2, deterministic=True)
    adapter = ZhanguoAdapter(agent, seat=seat, use_score_history=True, use_tile_belief=True)
    decision = adapter.decide(engine_to_view(state, seat))
    assert decision.action_id in legal_action_map(compact_market(state))


@pytest.mark.parametrize('value', [None, -1, True, '113'])
def test_invalid_bag_information_is_rejected(value):
    view = engine_to_view(fresh_state(), 0)
    view['bagCount'] = value
    with pytest.raises(GameInformationError, match='bagCount'):
        tile_adapter().decide(view)


@pytest.mark.parametrize('field', ['hand', 'vp'])
def test_hidden_own_information_is_rejected(field):
    view = engine_to_view(fresh_state(), 0)
    view['players'][0][field] = None
    with pytest.raises(GameInformationError, match=field):
        tile_adapter().decide(view)


def test_partial_own_replacement_colours_are_rejected():
    view = engine_to_view(state_after_replacement(0), 0)
    event = next(e for e in view['tileHistory']['events'] if e['kind'] == 'replace')
    event['colors'].pop()
    with pytest.raises(GameInformationError, match='colours'):
        tile_adapter().decide(view)


@pytest.mark.parametrize('field', ['gameId', 'version', 'yourTurn', 'view', 'lobby'])
def test_missing_protocol_information_stops_without_fallback(field, real_model):
    mock = MockZhanguo(short_game=False, seed=1)
    mock.start()
    state = mock.ext_state()
    del state[field]
    client = FakeClient([state])
    runner = RemoteRunner(client, real_model, RunnerConfig(interval=0, min_think=0))
    assert runner.run() == 2
    assert client.actions == [] and runner.fallbacks == 0


def test_old_score_checkpoint_needs_no_tile_ledger():
    model = HuangActorCritic(ModelConfig(observation_size=5430, encoder_width=8,
                                       encoder_layers=1, recurrent_hidden_size=4))
    mock = MockZhanguo(seed=1)
    mock.start()
    state = mock.ext_state()
    del state['view']['tileHistory']
    client = FakeClient([state])
    runner = RemoteRunner(client, model, RunnerConfig(interval=0, min_think=0))
    assert runner.use_score_history and not runner.use_tile_belief
    runner._tick()
    assert len(client.actions) == 1


def test_missing_history_over_http_stops_the_bundled_bot(real_model, make_runner):
    with MockZhanguo(short_game=False, seed=1) as mock:
        original = mock.bot_view

        def missing_history():
            view = original()
            del view['tileHistory']
            return view

        mock.bot_view = missing_history
        runner = make_runner(mock, real_model)
        assert runner.run() == 2
    assert not mock.action_posts and runner.fallbacks == 0
