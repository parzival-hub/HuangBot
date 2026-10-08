"""Runner bookkeeping against a scripted client (no HTTP)."""

import copy

from agents import RandomAgent
from huangbot.remote.client import StateResponse
from huangbot.remote.runner import RemoteRunner, RunnerConfig
from mock_server import MockZhanguo


class FakeClient:
    base_url = "fake://game"

    def __init__(self, states):
        self.states = list(states)
        self.actions = []
        self.statuses = []

    def redact(self, text):
        return text

    def get_state(self, if_none_match=None):
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return StateResponse(copy.deepcopy(state), None)

    def post_action(self, action, version):
        self.actions.append((action, version))
        return {}

    def post_status(self, agent, ready, note=None):
        self.statuses.append((agent, ready, note))


class CountingAgent(RandomAgent):
    resets = 0

    def reset_episode(self):
        type(self).resets += 1


class Memory:
    def get(self, seat):
        import torch

        return torch.zeros(1)

    def update(self, seat, hidden):
        pass


def playing_state():
    mock = MockZhanguo(seed=1)
    mock.start()
    state = mock.ext_state()
    assert state["yourTurn"]
    return state


def make(states, tiny_model, agent):
    agent.memory = Memory()
    client = FakeClient(states)
    runner = RemoteRunner(
        client, tiny_model, RunnerConfig(interval=0, min_think=0), agent_factory=lambda players: agent, sleep=lambda s: None
    )
    return runner, client


def test_version_going_backwards_resets_the_gru_memory(tiny_model):
    first = playing_state()
    first["version"] = 10
    second = copy.deepcopy(first)
    second["version"] = 3
    agent = CountingAgent()
    CountingAgent.resets = 0
    runner, client = make([first, second], tiny_model, agent)
    runner._tick()
    assert CountingAgent.resets == 0
    runner._tick()
    assert CountingAgent.resets == 1
    assert [version for _, version in client.actions] == [10, 3]


def test_the_same_version_does_not_reset(tiny_model):
    state = playing_state()
    agent = CountingAgent()
    CountingAgent.resets = 0
    runner, client = make([state, state], tiny_model, agent)
    runner._tick()
    runner._tick()
    assert CountingAgent.resets == 0


def test_status_is_reported_once_per_change(tiny_model):
    state = playing_state()
    runner, client = make([state], tiny_model, CountingAgent())
    for _ in range(3):
        runner._tick()
    assert len(client.statuses) == 1
    assert client.statuses[0][1] is True


def test_a_finished_game_ends_the_run_with_exit_code_0(tiny_model, caplog):
    import logging

    caplog.set_level(logging.INFO)
    state = playing_state()
    state["status"] = "finished"
    state["yourTurn"] = False
    runner, client = make([state], tiny_model, CountingAgent())
    assert runner.run() == 0
    assert client.actions == []
    assert "fallbacks used: 0" in caplog.text


def test_not_my_turn_does_nothing(tiny_model):
    state = playing_state()
    state["yourTurn"] = False
    runner, client = make([state], tiny_model, CountingAgent())
    runner._tick()
    assert client.actions == []
