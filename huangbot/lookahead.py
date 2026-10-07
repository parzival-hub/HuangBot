"""Information-set lookahead with batched neural policy/value inference."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
from typing import Optional

import numpy as np
import torch

from huang.action_codec import legal_action_map
from huang.engine import CHANCE, TERMINAL

from .environment import (
    PUBLIC_SCORE_HISTORY_SIZE,
    DecisionContext,
    HuangEnvironment,
    public_score_history_from_engine,
)
from .model import HuangActorCritic, PlayerHiddenStates


@dataclass(frozen=True)
class LookaheadConfig:
    """Small search budget intended for selective tactical evaluation."""

    depth: int = 2
    top_k: int = 8
    simulations: int = 16
    prior_weight: float = 0.05

    def validate(self) -> None:
        if self.depth <= 0:
            raise ValueError("lookahead depth must be positive")
        if self.top_k <= 0:
            raise ValueError("lookahead top_k must be positive")
        if self.simulations <= 0:
            raise ValueError("lookahead simulations must be positive")
        if not math.isfinite(self.prior_weight) or self.prior_weight < 0:
            raise ValueError("lookahead prior_weight must be finite and non-negative")


@dataclass(frozen=True)
class LookaheadDiagnostics:
    candidates: tuple[int, ...]
    mean_values: tuple[float, ...]
    selected_action: int
    leaf_batch_size: int


def determinize_hidden_tiles(engine, observer: int, random_source: random.Random):
    """Clone a state and resample hidden hands from the unseen tile pool.

    The observer's hand, public board, market and all public history remain
    untouched.  Only the allocation between opponent hands and the bag changes.
    """
    if not 0 <= observer < engine.players:
        raise ValueError("observer is outside the game")
    sampled = engine.clone()
    hidden_players = [player for player in range(engine.players) if player != observer]
    hand_sizes = {
        player: sum(sampled.hands[player].values()) for player in hidden_players
    }
    unseen = Counter(sampled.bag)
    for player in hidden_players:
        unseen.update(sampled.hands[player])
        sampled.hands[player] = Counter()

    for player in hidden_players:
        for _ in range(hand_sizes[player]):
            color = _sample_counter(unseen, random_source)
            sampled.hands[player][color] += 1
            unseen[color] -= 1
            if unseen[color] == 0:
                del unseen[color]
    sampled.bag = Counter(unseen)
    return sampled


def _sample_counter(counts: Counter, random_source: random.Random):
    total = sum(counts.values())
    if total <= 0:
        raise RuntimeError("cannot sample from an empty hidden tile pool")
    target = random_source.randrange(total)
    cumulative = 0
    for item, count in sorted(counts.items(), key=lambda pair: int(pair[0])):
        cumulative += count
        if target < cumulative:
            return item
    raise AssertionError("hidden tile sample fell outside the pool")


def _resolve_chance(engine, random_source: random.Random) -> None:
    while engine.current_actor() == CHANCE:
        outcomes = engine.chance_outcomes()
        if not outcomes:
            raise RuntimeError("chance node has no outcomes")
        threshold = random_source.random()
        cumulative = 0.0
        selected = outcomes[-1][0]
        for outcome, probability in outcomes:
            cumulative += probability
            if threshold <= cumulative:
                selected = outcome
                break
        engine.apply_chance(selected)


def _model_observation(model: HuangActorCritic, engine, player: int) -> np.ndarray:
    base = np.asarray(engine.observation_tensor(player), dtype=np.float32)
    if model.config.observation_size == base.size:
        return base
    if model.config.observation_size == base.size + PUBLIC_SCORE_HISTORY_SIZE:
        return np.concatenate(
            (base, public_score_history_from_engine(engine))
        ).astype(np.float32, copy=False)
    raise ValueError("model observation size is incompatible with lookahead state")


def _legal_context(model: HuangActorCritic, engine):
    player = engine.current_actor()
    if player in (CHANCE, TERMINAL):
        raise RuntimeError("lookahead requested a policy at a non-player node")
    action_map = legal_action_map(engine)
    mask = np.zeros(model.config.num_actions, dtype=np.bool_)
    mask[list(action_map)] = True
    return player, action_map, _model_observation(model, engine, player), mask


class InformationSetLookaheadAgent:
    """Policy-guided depth search that never uses the true hidden allocation."""

    name = "information-set-lookahead"

    def __init__(
        self,
        model: HuangActorCritic,
        *,
        players: int,
        config: Optional[LookaheadConfig] = None,
        seed: int = 0,
    ):
        if players != 2:
            raise ValueError("lookahead currently supports two-player HUANG")
        self.model = model
        self.players = players
        self.config = config or LookaheadConfig()
        self.config.validate()
        self.memory = PlayerHiddenStates(model, players)
        self.random = random.Random(seed)
        generator_device = "cuda" if model.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=generator_device)
        self.generator.manual_seed(seed)
        self.last_diagnostics: Optional[LookaheadDiagnostics] = None

    def reset_episode(self) -> None:
        self.memory.reset()
        self.last_diagnostics = None

    def select_action(self, context: DecisionContext) -> int:
        """Safe fallback for runners that do not expose a clonable state."""
        selection = self.model.select_action(
            context.observation,
            context.legal_action_mask,
            self.memory.get(context.player),
            deterministic=True,
        )
        self.memory.update(context.player, selection.hidden_state)
        return selection.action

    @torch.no_grad()
    def select_action_from_environment(
        self,
        context: DecisionContext,
        environment: HuangEnvironment,
    ) -> int:
        self.model.eval()
        player = context.player
        observation = torch.tensor(
            np.asarray(context.observation),
            dtype=torch.float32,
            device=self.model.device,
        ).unsqueeze(0)
        mask = torch.tensor(
            np.asarray(context.legal_action_mask),
            dtype=torch.bool,
            device=self.model.device,
        ).unsqueeze(0)
        root_output = self.model(observation, mask, self.memory.get(player))
        self.memory.update(player, root_output.hidden_state)
        legal = torch.as_tensor(
            context.legal_actions, dtype=torch.long, device=self.model.device
        )
        legal_logits = root_output.logits[0, legal]
        candidate_count = min(self.config.top_k, len(context.legal_actions))
        top = torch.topk(legal_logits, k=candidate_count)
        candidates = tuple(int(value) for value in legal[top.indices].cpu().tolist())
        priors = torch.softmax(top.values, dim=0).cpu().tolist()

        engines = []
        owners = []
        hidden_states = []
        root_hidden = root_output.hidden_state.detach()
        for candidate_index, action_id in enumerate(candidates):
            for _ in range(self.config.simulations):
                engine = determinize_hidden_tiles(
                    environment.open_spiel_state.engine,
                    player,
                    self.random,
                )
                action_map = legal_action_map(engine)
                engine.apply_action(action_map[action_id])
                _resolve_chance(engine, self.random)
                engines.append(engine)
                owners.append(candidate_index)
                hidden_states.append(
                    [
                        root_hidden.clone() if seat == player else self.model.initial_hidden(1)
                        for seat in range(self.players)
                    ]
                )

        for _ply in range(1, self.config.depth):
            active = [
                index
                for index, engine in enumerate(engines)
                if engine.current_actor() != TERMINAL
            ]
            if not active:
                break
            contexts = [_legal_context(self.model, engines[index]) for index in active]
            batch_hidden = torch.cat(
                [
                    hidden_states[index][actor]
                    for index, (actor, _actions, _observation, _mask) in zip(
                        active, contexts
                    )
                ],
                dim=1,
            )
            selections = self.model.select_actions(
                np.stack([entry[2] for entry in contexts]),
                np.stack([entry[3] for entry in contexts]),
                batch_hidden,
                deterministic=False,
                generator=self.generator,
            )
            for batch_index, (state_index, context_entry) in enumerate(
                zip(active, contexts)
            ):
                actor, action_map, _observation, _mask = context_entry
                action_id = selections.actions[batch_index]
                hidden_states[state_index][actor] = selections.hidden_state[
                    :, batch_index : batch_index + 1, :
                ].detach()
                engines[state_index].apply_action(action_map[action_id])
                _resolve_chance(engines[state_index], self.random)

        values = [0.0] * len(engines)
        nonterminal = [
            index
            for index, engine in enumerate(engines)
            if engine.current_actor() != TERMINAL
        ]
        for index, engine in enumerate(engines):
            if engine.current_actor() == TERMINAL:
                values[index] = float(engine.returns()[player])
        if nonterminal:
            leaf_contexts = [
                _legal_context(self.model, engines[index]) for index in nonterminal
            ]
            leaf_hidden = torch.cat(
                [
                    hidden_states[index][actor]
                    for index, (actor, _actions, _observation, _mask) in zip(
                        nonterminal, leaf_contexts
                    )
                ],
                dim=1,
            )
            estimates = self.model.predict_values(
                np.stack([entry[2] for entry in leaf_contexts]),
                leaf_hidden,
            )
            for index, context_entry, estimate in zip(
                nonterminal, leaf_contexts, estimates
            ):
                actor = context_entry[0]
                values[index] = float(estimate if actor == player else -estimate)

        totals = [0.0] * len(candidates)
        counts = [0] * len(candidates)
        for owner, value in zip(owners, values):
            totals[owner] += value
            counts[owner] += 1
        means = tuple(total / count for total, count in zip(totals, counts))
        ranked = [
            mean + self.config.prior_weight * prior
            for mean, prior in zip(means, priors)
        ]
        selected_index = max(range(len(candidates)), key=ranked.__getitem__)
        action = candidates[selected_index]
        self.last_diagnostics = LookaheadDiagnostics(
            candidates=candidates,
            mean_values=means,
            selected_action=action,
            leaf_batch_size=len(nonterminal),
        )
        return action
