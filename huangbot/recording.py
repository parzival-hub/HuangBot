"""Record complete agent matches as public HUANG viewer playbacks."""

from __future__ import annotations

from typing import Optional, Sequence

from huang.playback import create_frame, create_playback

from .environment import HuangEnvironment
from .matches import Agent, MatchResult


def record_match(
    environment: HuangEnvironment,
    agents: Sequence[Agent],
    *,
    seed: Optional[int] = None,
    title: str = "HUANG – aufgezeichnete Partie",
) -> tuple[dict, MatchResult]:
    """Play one match and retain a public snapshot after every player action."""
    if len(agents) != environment.players:
        raise ValueError("one agent is required for every player seat")
    for agent in {id(agent): agent for agent in agents}.values():
        reset_episode = getattr(agent, "reset_episode", None)
        if reset_episode is not None:
            reset_episode()

    step = environment.reset(seed=seed)
    frames = [
        create_frame(
            environment.snapshot(title=f"{title} – Spielbeginn"),
            index=0,
            kind="initial",
            description="Spielbereiter Ausgangszustand",
            chance_events=step.chance_events,
        )
    ]
    previous_chance_events = step.chance_events

    while not step.done:
        context = environment.decision_context()
        agent = agents[context.player]
        selector = getattr(agent, 'select_action_from_environment', None)
        action_id = (selector(context, environment) if selector
                     else agent.select_action(context))
        action_by_id = {action.action_id: action for action in context.action_info}
        if action_id not in action_by_id:
            raise RuntimeError(
                f"{agents[context.player].name} selected illegal action {action_id}"
            )
        action = action_by_id[action_id]
        step = environment.step(action_id)
        rewards = tuple(float(value) for value in step.rewards) if step.done else None
        frames.append(
            create_frame(
                environment.snapshot(
                    title=f"{title} – Entscheidung {step.player_decisions}"
                ),
                index=len(frames),
                kind="player_action",
                description=action.description,
                player=context.player + 1,
                action_id=action_id,
                action_kind=action.kind,
                action_parameters=action.parameters,
                chance_events=step.chance_events - previous_chance_events,
                rewards=rewards,
            )
        )
        previous_chance_events = step.chance_events

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
    result = MatchResult(
        returns=returns,
        winner=winner,
        player_decisions=step.player_decisions,
        chance_events=step.chance_events,
        terminated=step.terminated,
        truncated=step.truncated,
        score_keys=score_keys,
    )
    playback = create_playback(
        frames,
        title=title,
        players=environment.players,
        metadata={
            "seed": seed,
            "agents": [agent.name for agent in agents],
            "player_decisions": result.player_decisions,
            "chance_events": result.chance_events,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "returns": list(result.returns),
            "winner": None if winner is None else winner + 1,
        },
    )
    return playback, result
