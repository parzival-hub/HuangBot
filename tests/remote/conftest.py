import logging

import pytest
import torch

from huangbot.checkpoints import load_checkpoint
from huangbot.model import HuangActorCritic, ModelConfig, RecurrentModelAgent
from huangbot.remote.client import ZhanguoClient
from huangbot.remote.runner import RemoteRunner, RunnerConfig


@pytest.fixture(scope="session")
def real_model():
    torch.set_num_threads(1)
    return load_checkpoint(device="cpu")


@pytest.fixture(scope="session")
def tiny_model():
    """Random weights, standard 5410 inputs: fast and deterministic."""
    torch.manual_seed(0)
    model = HuangActorCritic(ModelConfig(encoder_width=8, encoder_layers=1, recurrent_hidden_size=4))
    model.eval()
    return model


class RecordingAgent:
    """A real ``RecurrentModelAgent`` that remembers the hidden state it started from."""

    def __init__(self, model, players):
        self.inner = RecurrentModelAgent(model, players=players, deterministic=True)
        self.memory = self.inner.memory
        self.hidden_before = []
        self.excluded_seen = []

    def select_action(self, context):
        self.hidden_before.append(self.memory.get(context.player).clone())
        return self.inner.select_action(context)

    def reset_episode(self):
        self.inner.reset_episode()


@pytest.fixture
def make_runner():
    def factory(mock, model, *, token=None, game_id=None, agent_factory=None, sleep=None, **config):
        client = ZhanguoClient(
            mock.origin, game_id or mock.game_id, mock.token if token is None else token, timeout=5
        )
        settings = dict(interval=0.0, min_think=0.0)
        settings.update(config)
        return RemoteRunner(
            client, model, RunnerConfig(**settings), agent_factory=agent_factory, sleep=sleep
        )

    return factory


@pytest.fixture(autouse=True)
def quiet_logs(caplog):
    caplog.set_level(logging.INFO, logger="huangbot.remote")
