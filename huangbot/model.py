"""Masked recurrent PyTorch actor-critic model for HUANG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn

from .environment import DecisionContext


@dataclass(frozen=True)
class ModelConfig:
    observation_size: int = 5410
    num_actions: int = 6599
    encoder_width: int = 512
    encoder_layers: int = 2
    recurrent_hidden_size: int = 256
    recurrent_layers: int = 1

    def validate(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    values: torch.Tensor
    hidden_state: torch.Tensor




@dataclass(frozen=True)
class ActionSelection:
    action: int
    log_probability: float
    value: float
    entropy: float
    hidden_state: torch.Tensor


@dataclass(frozen=True)
class BatchActionSelection:
    """Selections produced by one model call for independent games."""

    actions: tuple[int, ...]
    log_probabilities: tuple[float, ...]
    values: tuple[float, ...]
    entropies: tuple[float, ...]
    hidden_state: torch.Tensor


class HuangActorCritic(nn.Module):
    """MLP encoder, GRU memory, masked policy head and scalar value head."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.config.validate()

        encoder_layers = []
        input_size = self.config.observation_size
        for _ in range(self.config.encoder_layers):
            encoder_layers.extend(
                (
                    nn.Linear(input_size, self.config.encoder_width),
                    nn.LayerNorm(self.config.encoder_width),
                    nn.ReLU(),
                )
            )
            input_size = self.config.encoder_width
        self.encoder = nn.Sequential(*encoder_layers)
        self.recurrent = nn.GRU(
            input_size=self.config.encoder_width,
            hidden_size=self.config.recurrent_hidden_size,
            num_layers=self.config.recurrent_layers,
            batch_first=True,
        )
        self.policy_head = nn.Linear(
            self.config.recurrent_hidden_size,
            self.config.num_actions,
        )
        self.value_head = nn.Linear(self.config.recurrent_hidden_size, 1)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device | str] = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return torch.zeros(
            self.config.recurrent_layers,
            batch_size,
            self.config.recurrent_hidden_size,
            dtype=torch.float32,
            device=device or self.device,
        )

    def forward(
        self,
        observations: torch.Tensor,
        legal_action_masks: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """Evaluate a batch of single steps or a batch of sequences.

        Accepted observation shapes are ``[batch, observation]`` and
        ``[batch, time, observation]``. Masks follow the same leading shape.
        """
        if observations.ndim not in (2, 3):
            raise ValueError("observations must have rank 2 or 3")
        if observations.shape[-1] != self.config.observation_size:
            raise ValueError(
                f"expected observation size {self.config.observation_size}, "
                f"got {observations.shape[-1]}"
            )
        single_step = observations.ndim == 2
        if single_step:
            observations = observations.unsqueeze(1)
            legal_action_masks = legal_action_masks.unsqueeze(1)
        expected_mask_shape = observations.shape[:-1] + (self.config.num_actions,)
        if tuple(legal_action_masks.shape) != tuple(expected_mask_shape):
            raise ValueError(
                f"expected legal mask shape {expected_mask_shape}, "
                f"got {tuple(legal_action_masks.shape)}"
            )
        legal_action_masks = legal_action_masks.to(
            device=observations.device,
            dtype=torch.bool,
        )
        if not torch.all(legal_action_masks.any(dim=-1)):
            raise ValueError("every model step must contain at least one legal action")

        batch_size, time_steps, _ = observations.shape
        encoded = self.encoder(observations.reshape(-1, self.config.observation_size))
        encoded = encoded.reshape(batch_size, time_steps, self.config.encoder_width)
        if hidden_state is None:
            hidden_state = self.initial_hidden(batch_size, device=observations.device)
        recurrent_output, next_hidden = self.recurrent(encoded, hidden_state)
        logits = self.policy_head(recurrent_output)
        minimum = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~legal_action_masks, minimum)
        values = self.value_head(recurrent_output).squeeze(-1)

        if single_step:
            logits = logits.squeeze(1)
            values = values.squeeze(1)
        return ModelOutput(logits=logits, values=values, hidden_state=next_hidden)


    @torch.no_grad()
    def select_action(
        self,
        observation,
        legal_action_mask,
        hidden_state: Optional[torch.Tensor] = None,
        *,
        deterministic: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> ActionSelection:
        observation_tensor = torch.tensor(
            np.asarray(observation),
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)
        mask_tensor = torch.tensor(
            np.asarray(legal_action_mask),
            dtype=torch.bool,
            device=self.device,
        ).reshape(1, -1)
        if hidden_state is not None:
            hidden_state = hidden_state.to(self.device)
        output = self(observation_tensor, mask_tensor, hidden_state)
        log_probabilities = torch.log_softmax(output.logits, dim=-1)
        probabilities = torch.softmax(output.logits, dim=-1)
        if deterministic:
            action_tensor = torch.argmax(output.logits, dim=-1, keepdim=True)
        else:
            action_tensor = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            )
        selected_log_probability = log_probabilities.gather(1, action_tensor).squeeze()
        entropy = -(probabilities * log_probabilities).sum(dim=-1).squeeze()
        action = int(action_tensor.item())
        if not bool(mask_tensor[0, action]):
            raise RuntimeError("masked policy selected an illegal action")
        return ActionSelection(
            action=action,
            log_probability=float(selected_log_probability.item()),
            value=float(output.values.item()),
            entropy=float(entropy.item()),
            hidden_state=output.hidden_state.detach(),
        )

    @torch.no_grad()
    def select_actions(
        self,
        observations,
        legal_action_masks,
        hidden_state: Optional[torch.Tensor] = None,
        *,
        deterministic: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> BatchActionSelection:
        """Select actions for independent games in one batched forward pass."""
        observation_tensor = torch.as_tensor(
            np.asarray(observations),
            dtype=torch.float32,
            device=self.device,
        )
        mask_tensor = torch.as_tensor(
            np.asarray(legal_action_masks),
            dtype=torch.bool,
            device=self.device,
        )
        if observation_tensor.ndim != 2:
            raise ValueError("batched observations must have rank 2")
        if hidden_state is not None:
            hidden_state = hidden_state.to(self.device)
        output = self(observation_tensor, mask_tensor, hidden_state)
        log_probabilities = torch.log_softmax(output.logits, dim=-1)
        probabilities = torch.softmax(output.logits, dim=-1)
        if deterministic:
            action_tensor = torch.argmax(output.logits, dim=-1, keepdim=True)
        else:
            action_tensor = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            )
        selected_log_probabilities = log_probabilities.gather(
            1, action_tensor
        ).squeeze(1)
        entropies = -(probabilities * log_probabilities).sum(dim=-1)
        actions = action_tensor.squeeze(1)
        if not bool(mask_tensor.gather(1, action_tensor).all()):
            raise RuntimeError("masked policy selected an illegal action")
        return BatchActionSelection(
            actions=tuple(int(value) for value in actions.cpu().tolist()),
            log_probabilities=tuple(
                float(value)
                for value in selected_log_probabilities.cpu().tolist()
            ),
            values=tuple(float(value) for value in output.values.cpu().tolist()),
            entropies=tuple(float(value) for value in entropies.cpu().tolist()),
            hidden_state=output.hidden_state.detach(),
        )

    @torch.no_grad()
    def predict_values(
        self,
        observations,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> tuple[float, ...]:
        """Estimate values for independent states with one batched call."""
        observation_tensor = torch.as_tensor(
            np.asarray(observations),
            dtype=torch.float32,
            device=self.device,
        )
        if observation_tensor.ndim != 2:
            raise ValueError("batched observations must have rank 2")
        legal_action_mask = torch.ones(
            (observation_tensor.shape[0], self.config.num_actions),
            dtype=torch.bool,
            device=self.device,
        )
        if hidden_state is not None:
            hidden_state = hidden_state.to(self.device)
        output = self(observation_tensor, legal_action_mask, hidden_state)
        return tuple(float(value) for value in output.values.cpu().tolist())

    @torch.no_grad()
    def predict_value(
        self,
        observation,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> float:
        """Estimate a state's value without requiring a meaningful action mask."""
        observation_tensor = torch.tensor(
            np.asarray(observation),
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)
        legal_action_mask = torch.ones(
            (1, self.config.num_actions),
            dtype=torch.bool,
            device=self.device,
        )
        if hidden_state is not None:
            hidden_state = hidden_state.to(self.device)
        output = self(observation_tensor, legal_action_mask, hidden_state)
        return float(output.values.item())






class PlayerHiddenStates:
    """Keep recurrent memory isolated between player seats."""

    def __init__(self, model: HuangActorCritic, players: int):
        if not 2 <= players <= 4:
            raise ValueError("players must be between 2 and 4")
        self.model = model
        self.players = players
        self._states = []
        self.reset()

    def reset(self, player: Optional[int] = None) -> None:
        if player is None:
            self._states = [
                self.model.initial_hidden(1) for _ in range(self.players)
            ]
            return
        self._validate_player(player)
        self._states[player] = self.model.initial_hidden(1)

    def get(self, player: int) -> torch.Tensor:
        self._validate_player(player)
        return self._states[player]

    def update(self, player: int, hidden_state: torch.Tensor) -> None:
        self._validate_player(player)
        expected = (
            self.model.config.recurrent_layers,
            1,
            self.model.config.recurrent_hidden_size,
        )
        if tuple(hidden_state.shape) != expected:
            raise ValueError(
                f"expected hidden state shape {expected}, got {tuple(hidden_state.shape)}"
            )
        self._states[player] = hidden_state.detach().to(self.model.device)


    def _validate_player(self, player: int) -> None:
        if not 0 <= player < self.players:
            raise ValueError(f"player {player} is outside the player range")


class RecurrentModelAgent:
    """Use one actor-critic model with separate memory for every seat."""

    name = "recurrent-model"

    def __init__(
        self,
        model: HuangActorCritic,
        *,
        players: int,
        deterministic: bool = False,
        seed: Optional[int] = None,
    ):
        self.model = model
        self.deterministic = deterministic
        self.memory = PlayerHiddenStates(model, players)
        generator_device = "cuda" if model.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=generator_device)
        if seed is not None:
            self.generator.manual_seed(seed)
        self.last_selection: Optional[ActionSelection] = None

    def reset_episode(self) -> None:
        self.memory.reset()
        self.last_selection = None

    def select_action(self, context: DecisionContext) -> int:
        self.model.eval()
        selection = self.model.select_action(
            context.observation,
            context.legal_action_mask,
            self.memory.get(context.player),
            deterministic=self.deterministic,
            generator=self.generator,
        )
        self.memory.update(context.player, selection.hidden_state)
        self.last_selection = selection
        return selection.action
