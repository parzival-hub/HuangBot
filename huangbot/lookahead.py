"""Information-set lookahead with batched neural policy/value inference."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Optional

import numpy as np
import torch

from huang.action_codec import legal_action_map
from huang.engine import CHANCE, TERMINAL, Phase

from .environment import (
    PUBLIC_SCORE_HISTORY_SIZE,
    DecisionContext,
    HuangEnvironment,
    public_score_history_from_engine,
    observation_options,
)
from .model import HuangActorCritic, PlayerHiddenStates
from .tile_belief import TileKnowledge, TileTracker


@dataclass(frozen=True)
class LookaheadConfig:
    """Small search budget intended for selective tactical evaluation."""

    depth: int = 2
    top_k: int = 4
    simulations: int = 4
    prior_weight: float = 0.05
    adaptive: bool = True
    max_time_seconds: float = 3.0
    max_resolution_steps: int = 256

    def validate(self) -> None:
        if self.depth <= 0:
            raise ValueError("lookahead depth must be positive")
        if self.top_k <= 0:
            raise ValueError("lookahead top_k must be positive")
        if self.simulations <= 0:
            raise ValueError("lookahead simulations must be positive")
        if not math.isfinite(self.prior_weight) or self.prior_weight < 0:
            raise ValueError("lookahead prior_weight must be finite and non-negative")
        if not math.isfinite(self.max_time_seconds) or self.max_time_seconds <= 0:
            raise ValueError('max_time_seconds must be finite and positive')
        if self.max_resolution_steps <= 0:
            raise ValueError('max_resolution_steps must be positive')

    def budget(self, knowledge: TileKnowledge, *, conflict=False):
        depth, simulations = self.depth, self.simulations
        if self.adaptive:
            if knowledge.bag_total <= 3:
                depth, simulations = max(depth,12), max(simulations,64)
            elif knowledge.bag_total <= 10:
                depth, simulations = max(depth,8), max(simulations,32)
            elif knowledge.bag_total <= 30:
                depth, simulations = max(depth,4), max(simulations,16)
            else:
                depth, simulations = max(depth,2), max(simulations,4)
            if knowledge.uncertainty > 0.8:
                simulations = max(simulations,min(64,simulations*2))
            if conflict:
                depth, simulations = max(depth,4), max(simulations,16)
        return depth, simulations


@dataclass(frozen=True)
class LookaheadDiagnostics:
    candidates: tuple[int, ...]
    mean_values: tuple[float, ...]
    selected_action: int
    leaf_batch_size: int
    effective_depth: int = 0
    simulations_per_candidate: int = 0
    requested_simulations: int = 0
    resolution_steps: int = 0
    bag_total: int = 0
    draw_probabilities: tuple[float, ...] = ()
    uncertainty: float = 0.0
    elapsed_seconds: float = 0.0
    time_budget_reached: bool = False


def determinize_hidden_tiles(engine, observer: int, random_source: random.Random,
                             knowledge: TileKnowledge | None = None):
    """Clone a state and resample hidden hands from the unseen tile pool.

    The observer's hand, board, market and public history remain untouched.
    Hidden hand, bag and discard colours come only from observed tile counts.
    """
    if not 0 <= observer < engine.players:
        raise ValueError("observer is outside the game")
    knowledge = knowledge or TileTracker(observer).observe(engine)
    if knowledge.observer != observer:
        raise ValueError('tile knowledge belongs to a different player')
    sampled = engine.clone()
    sampled.bag, sampled.hands, discarded = knowledge.sample_world(random_source)
    public_scores = public_score_history_from_engine(engine).reshape(4,5)*30
    for player in range(engine.players):
        if player != observer:
            sampled.private_history[player] = []
            if discarded[player]:
                sampled.private_history[player].append('replaced '+','.join(
                    str(discarded[player][color]) for color in range(5)))
            sampled.points[player] = [int(round(value)) for value in public_scores[player]]
    return sampled


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
    options = observation_options(model.config.observation_size,base.size)
    pieces = [base]
    if options['include_public_score_history']:
        pieces.append(public_score_history_from_engine(engine))
    if options['include_tile_belief']:
        pieces.append(TileTracker(player).observe(engine).feature_vector())
    return np.concatenate(pieces).astype(np.float32,copy=False)


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
        started = time.monotonic()
        self.model.eval()
        player = context.player
        source = environment.open_spiel_state.engine
        knowledge = context.tile_knowledge or environment.tile_knowledge(player)
        conflict = source.phase in (Phase.REVOLT_ATTACK,Phase.REVOLT_DEFEND,
                                    Phase.WAR_CONTRIBUTE,Phase.WAR_TIE,Phase.WAR_REMOVE)
        depth, simulations = self.config.budget(knowledge, conflict=conflict)
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

        root_hidden = root_output.hidden_state.detach()
        totals = [0.0]*len(candidates)
        leaf_count = resolution_steps = completed = 0
        deadline = started+self.config.max_time_seconds
        for _simulation in range(simulations):
            # All candidates face the same sampled hidden world in each round.
            world = determinize_hidden_tiles(source,player,self.random,knowledge)
            engines, hidden_states = [], []
            for action_id in candidates:
                engine = world.clone()
                action_map = legal_action_map(engine)
                engine.apply_action(action_map[action_id])
                _resolve_chance(engine, self.random)
                engines.append(engine)
                hidden_states.append(
                    [
                        root_hidden.clone() if seat == player else self.model.initial_hidden(1)
                        for seat in range(self.players)
                    ]
                )

            values, leaves, extended = self._rollout(engines,hidden_states,player,depth)
            for index,value in enumerate(values):
                totals[index] += value
            leaf_count = max(leaf_count,leaves)
            resolution_steps = max(resolution_steps,extended)
            completed += 1
            if time.monotonic() >= deadline:
                break
        means = tuple(total/completed for total in totals)
        ranked = [mean+self.config.prior_weight*prior for mean,prior in zip(means,priors)]
        action = candidates[max(range(len(candidates)),key=ranked.__getitem__)]
        self.last_diagnostics = LookaheadDiagnostics(
            candidates,means,action,leaf_count,depth,completed,simulations,
            resolution_steps,knowledge.bag_total,knowledge.draw_probabilities,
            knowledge.uncertainty,time.monotonic()-started,
            completed<simulations and time.monotonic()>=deadline)
        return action

    def _rollout(self, engines, hidden_states, player, depth):
        ply = 1
        while True:
            active = [
                index
                for index, engine in enumerate(engines)
                if engine.current_actor() != TERMINAL
                and (ply < depth or engine.phase != Phase.TURN)
            ]
            if not active:
                break
            if ply >= depth+self.config.max_resolution_steps:
                raise RuntimeError('lookahead could not resolve a tactical phase safely')
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
            ply += 1

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

        return values,len(nonterminal),max(0,ply-depth)
