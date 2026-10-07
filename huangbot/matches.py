"""Utilities for running complete matches between compatible agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

from .environment import DecisionContext, HuangEnvironment


class Agent(Protocol):
    """Minimal interface required by the match runner."""

    name: str

    def select_action(self, context: DecisionContext) -> int:
        ...


@dataclass(frozen=True)
class MatchResult:
    returns: tuple[float, ...]
    winner: Optional[int]
    player_decisions: int
    chance_events: int
    terminated: bool
    truncated: bool
    score_keys: tuple[tuple[tuple[int, ...], int], ...] | None


def play_match(
    environment: HuangEnvironment,
    agents: Sequence[Agent],
    *,
    seed: Optional[int] = None,
) -> MatchResult:
    if len(agents) != environment.players:
        raise ValueError("one agent is required for every player seat")
    for agent in {id(agent): agent for agent in agents}.values():
        reset_episode = getattr(agent, "reset_episode", None)
        if reset_episode is not None:
            reset_episode()
    step = environment.reset(seed=seed)
    while not step.done:
        context = environment.decision_context()
        agent = agents[context.player]
        environment_selector = getattr(
            agent, "select_action_from_environment", None
        )
        action = (
            environment_selector(context, environment)
            if environment_selector is not None
            else agent.select_action(context)
        )
        if action not in context.legal_actions:
            raise RuntimeError(
                f"{agents[context.player].name} selected illegal action {action}"
            )
        step = environment.step(action)

    returns = tuple(float(value) for value in step.rewards)
    positive = [player for player, value in enumerate(returns) if value > 0]
    winner = positive[0] if len(positive) == 1 else None
    score_keys = None
    if step.terminated:
        score_keys = tuple(
            (tuple(int(value) for value in balanced), int(white_tiebreak))
            for balanced, white_tiebreak in (
                environment.open_spiel_state.engine.score_keys()
            )
        )
    return MatchResult(
        returns=returns,
        winner=winner,
        player_decisions=step.player_decisions,
        chance_events=step.chance_events,
        terminated=step.terminated,
        truncated=step.truncated,
        score_keys=score_keys,
    )
