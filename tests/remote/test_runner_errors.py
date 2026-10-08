import logging

import pytest
import torch

from conftest import RecordingAgent
from export_view import GOLDEN_ROWS
from huangbot.remote.client import ZhanguoClient
from huangbot.remote.runner import ConfigError, RemoteRunner
from mock_server import MockZhanguo


@pytest.fixture
def agents():
    created = []

    def factory_for(model):
        def factory(players):
            agent = RecordingAgent(model, players)
            created.append(agent)
            return agent

        return factory

    factory_for.created = created
    return factory_for


def start_game(runner, mock):
    """First tick: the lobby (status report, the mock starts the game)."""
    runner._tick()
    assert mock.status == "playing"


def test_rejected_actions_restore_the_gru_memory_and_the_third_try_succeeds(tiny_model, make_runner, agents):
    with MockZhanguo(seed=2) as mock:
        runner = make_runner(mock, tiny_model, agent_factory=agents(tiny_model))
        start_game(runner, mock)
        mock.action_script = ["reject", "reject"]
        runner._tick()
        agent = agents.created[0]
        zero = torch.zeros_like(agent.hidden_before[0])

    assert len(agent.hidden_before) == 3
    assert all(torch.equal(h, zero) for h in agent.hidden_before)  # restored before every retry
    assert not torch.equal(agent.memory.get(0), zero)  # advanced exactly once, by the accepted decision
    bodies = [body["action"] for body, _ in mock.action_posts]
    assert [code for _, code in mock.action_posts[:3]] == [400, 400, 200]
    assert len({str(b) for b in bodies[:3]}) == 3  # a rejected action is excluded from the retry
    assert runner.rejections == 2 and runner.fallbacks == 0 and runner.accepted == 1


def test_gru_advances_once_per_accepted_decision(tiny_model, make_runner, agents):
    with MockZhanguo(seed=2) as mock:
        runner = make_runner(mock, tiny_model, agent_factory=agents(tiny_model))
        start_game(runner, mock)
        for _ in range(3):
            runner._tick()
        agent = agents.created[0]
    assert len(agent.hidden_before) == runner.accepted == 3
    assert not torch.equal(agent.hidden_before[0], agent.hidden_before[1])


def test_stale_409_refetches_without_counting_an_error(tiny_model, make_runner, agents, caplog):
    sleeps = []
    with MockZhanguo(seed=3) as mock:
        runner = make_runner(mock, tiny_model, agent_factory=agents(tiny_model), sleep=sleeps.append)
        start_game(runner, mock)
        mock.action_script = ["stale"]
        runner._tick()
        agent = agents.created[0]
    assert [code for _, code in mock.action_posts[:2]] == [409, 200]
    assert runner.rejections == 0 and runner.fallbacks == 0 and runner.accepted == 1
    assert len(agent.hidden_before) == 2
    assert torch.equal(agent.hidden_before[0], agent.hidden_before[1])  # the stale decision was undone
    assert "stale" in caplog.text
    assert sleeps == []  # no backoff


def test_three_rejections_send_the_fallback(tiny_model, make_runner, agents, caplog):
    with MockZhanguo(seed=4) as mock:
        runner = make_runner(mock, tiny_model, agent_factory=agents(tiny_model))
        start_game(runner, mock)
        mock.action_script = ["reject", "reject", "reject"]
        runner._tick()
        agent = agents.created[0]
    assert [code for _, code in mock.action_posts[:4]] == [400, 400, 400, 200]
    assert mock.action_posts[3][0]["action"] == {"type": "replace", "tiles": []}
    assert runner.fallbacks == 1 and runner.rejections == 3
    assert torch.equal(agent.memory.get(0), torch.zeros_like(agent.memory.get(0)))
    assert "sending fallback" in caplog.text


def test_a_rejected_fallback_is_logged_and_the_bot_keeps_polling(tiny_model, make_runner, caplog):
    with MockZhanguo(seed=4) as mock:
        runner = make_runner(mock, tiny_model)
        start_game(runner, mock)
        mock.reject_all = True
        runner._tick()
        posted = len(mock.action_posts)
        assert posted == 4  # 3 model actions + the fallback
        for _ in range(3):
            runner._tick()  # same version: nothing more is sent
        assert len(mock.action_posts) == posted
        mock.reject_all = False
    assert any(r.levelno == logging.ERROR and "fallback" in r.getMessage() for r in caplog.records)


def test_debug_dumps_for_rejections_and_fallbacks(tiny_model, make_runner, tmp_path):
    import json

    with MockZhanguo(seed=4) as mock:
        runner = make_runner(mock, tiny_model, debug_dir=tmp_path)
        start_game(runner, mock)
        mock.reject_all = True
        runner._tick()
    reasons = sorted(p.name.split("-", 3)[3].removesuffix(".json") for p in tmp_path.iterdir())
    assert sum(r.split("-")[0] == "rejected" for r in reasons) == 3
    assert "fallback" in reasons and "fallback-rejected" in reasons
    dump = json.loads(next(p for p in tmp_path.iterdir() if "-rejected" in p.name).read_text())
    assert dump["view"]["you"] == 0 and "error" in dump and "traceback" in dump
    assert mock.token not in "".join(p.read_text() for p in tmp_path.iterdir())


def test_model_path_exception_falls_back_with_a_dump(tiny_model, make_runner, tmp_path):
    class Broken:
        memory = None

        def select_action(self, context):
            raise RuntimeError("boom")

        def reset_episode(self):
            pass

    class Memory:
        def get(self, seat):
            return torch.zeros(1)

        def update(self, seat, hidden):
            pass

    Broken.memory = Memory()
    with MockZhanguo(seed=4) as mock:
        runner = make_runner(mock, tiny_model, agent_factory=lambda players: Broken(), debug_dir=tmp_path)
        start_game(runner, mock)
        runner._tick()
    assert mock.action_posts[0][0]["action"] == {"type": "replace", "tiles": []}
    assert runner.fallbacks == 1
    assert any("decide-error" in p.name for p in tmp_path.iterdir())


def test_network_errors_back_off_and_the_game_still_finishes(tiny_model, make_runner):
    delays = []
    with MockZhanguo(seed=6) as mock:
        mock.get_script = ["drop", "drop", "drop", "500"]
        runner = make_runner(mock, tiny_model, sleep=delays.append)
        assert runner.run() == 0
    assert [d for d in delays if d >= 1] == [1.0, 2.0, 4.0, 8.0]
    assert mock.status == "finished"
    assert mock.codes[("action", 400)] == 0


def test_a_lost_action_answer_is_retried_with_a_fresh_state(tiny_model, make_runner, agents):
    delays = []
    with MockZhanguo(seed=6) as mock:
        runner = make_runner(mock, tiny_model, agent_factory=agents(tiny_model), sleep=delays.append)
        start_game(runner, mock)
        mock.action_script = ["drop", "500"]
        assert runner.run() == 0
        agent = agents.created[0]
    assert [d for d in delays if d >= 1][:2] == [1.0, 2.0]
    assert torch.equal(agent.hidden_before[0], agent.hidden_before[1])
    assert torch.equal(agent.hidden_before[1], agent.hidden_before[2])
    assert mock.status == "finished" and runner.rejections == 0


def test_max_errors_gives_up_with_exit_code_1(tiny_model, make_runner):
    with MockZhanguo(seed=6) as mock:
        mock.get_script = ["drop"] * 10
        runner = make_runner(mock, tiny_model, sleep=lambda s: None, max_errors=3)
        assert runner.run() == 1


def test_wrong_token_exits_with_code_2(tiny_model, make_runner, caplog):
    with MockZhanguo() as mock:
        runner = make_runner(mock, tiny_model, token="not-the-token")
        assert runner.run() == 2
    assert "token invalid or seat removed" in caplog.text
    assert "not-the-token" not in caplog.text


def test_unknown_game_exits_with_code_2(tiny_model, make_runner, caplog):
    with MockZhanguo() as mock:
        runner = make_runner(mock, tiny_model, game_id="nope")
        assert runner.run() == 2
    assert "game not found" in caplog.text


def test_unsupported_protocol_exits_with_code_2(tiny_model, make_runner):
    with MockZhanguo() as mock:
        original = mock.ext_state
        mock.ext_state = lambda: {**original(), "protocol": 2}
        assert make_runner(mock, tiny_model).run() == 2


def test_model_with_the_wrong_input_size_is_refused(make_runner):
    from huangbot.model import HuangActorCritic, ModelConfig

    odd = HuangActorCritic(ModelConfig(observation_size=1000, encoder_width=8, encoder_layers=1, recurrent_hidden_size=4))
    client = ZhanguoClient("http://127.0.0.1:1", "g", "t")
    with pytest.raises(ConfigError, match="1000"):
        RemoteRunner(client, odd)
    wrong_actions = HuangActorCritic(ModelConfig(num_actions=100, encoder_width=8, encoder_layers=1, recurrent_hidden_size=4))
    with pytest.raises(ConfigError, match="actions"):
        RemoteRunner(client, wrong_actions)


def test_model_without_score_history_is_supported(make_runner):
    from huangbot.model import HuangActorCritic, ModelConfig

    plain = HuangActorCritic(ModelConfig(encoder_width=8, encoder_layers=1, recurrent_hidden_size=4))
    with MockZhanguo(seed=1) as mock:
        runner = make_runner(mock, plain)
        assert runner.use_score_history is False
        assert runner.run() == 0
    assert mock.status == "finished" and mock.codes[("action", 400)] == 0


def test_min_think_delays_the_answer(tiny_model, make_runner):
    waits = []
    with MockZhanguo(seed=3) as mock:
        runner = make_runner(mock, tiny_model, min_think=0.9, sleep=waits.append)
        start_game(runner, mock)
        runner._tick()
    assert waits and 0 < waits[0] <= 0.9
    assert mock.action_posts


def test_the_board_changing_to_a_wrong_one_stops_the_bot(tiny_model, make_runner):
    with MockZhanguo(seed=1, start_when_ready=False) as mock:
        runner = make_runner(mock, tiny_model)
        mock.start()
        mock.rows = ["." * 15] * 15
        runner._tick()
    assert mock.action_posts == []
    assert mock.status_posts[0]["ready"] is False
    assert mock.rows != GOLDEN_ROWS
