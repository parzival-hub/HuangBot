"""Bridge between a Zhanguo view and the unchanged ``RecurrentModelAgent``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from huang.action_codec import NUM_DISTINCT_ACTIONS, legal_action_map
from huang.engine import Action, HuangState, Phase
from huang.snapshot import snapshot_from_engine
from huangbot.environment import ActionInfo, DecisionContext

from .protocol import (
    COLOR_NAMES,
    JsonObject,
    NoLegalActionError,
    TranslationError,
)
from .score_history import public_score_history
from .view_state import build_shadow_state, pagoda_ids, sorted_triangle


@dataclass(frozen=True)
class Decision:
    game_action: JsonObject
    action_id: int
    note: str
    action: Optional[Action] = None
    phase: str = ""


class ZhanguoAdapter:
    """Decide in ``view`` for ``seat`` with an agent that only knows HuangEngine.

    ``use_score_history`` appends the 20 public-score features that checkpoints
    with ``observation_size == 5410 + 20`` expect.
    """

    def __init__(
        self,
        agent,
        *,
        seat: int,
        use_score_history: bool,
        strict: bool = False,
    ):
        self.agent = agent
        self.seat = seat
        self.use_score_history = use_score_history
        self.strict = strict
        self.player_decisions = 0

    # -- bookkeeping ---------------------------------------------------

    def record_accepted(self) -> None:
        self.player_decisions += 1

    def reset_episode(self) -> None:
        self.agent.reset_episode()
        self.player_decisions = 0

    def snapshot_memory(self):
        """Copy of this seat's recurrent state, to undo a discarded decision."""
        return self.agent.memory.get(self.seat).clone()

    def restore_memory(self, snapshot) -> None:
        self.agent.memory.update(self.seat, snapshot.clone())

    # -- deciding ------------------------------------------------------

    def decide(self, view: Mapping[str, Any], *, exclude: frozenset[int] = frozenset()) -> Decision:
        state = build_shadow_state(view, seat=self.seat, strict=self.strict)
        pending = view.get("pending")
        if pending and pending["kind"] == "warRemove":
            return self._decide_war_remove(state, view, pending, exclude)
        action_id, action = self._choose(state, view, exclude)
        return Decision(
            game_action=self.to_game_action(state, action, view),
            action_id=action_id,
            note=str(action),
            action=action,
            phase=state.phase.value,
        )

    def make_context(
        self,
        state: HuangState,
        view: Mapping[str, Any],
        legal: Optional[Mapping[int, Action]] = None,
    ) -> DecisionContext:
        """The context ``HuangEnvironment.decision_context()`` would have built."""
        if legal is None:
            legal = legal_action_map(state)
        mask = np.zeros(NUM_DISTINCT_ACTIONS, dtype=np.bool_)
        mask[list(legal)] = True
        observation = state.observation_tensor(self.seat)
        if self.use_score_history:
            observation = np.concatenate(
                [observation, public_score_history(view, state.players)]
            )
        observation = observation.astype(np.float32, copy=False)
        observation.setflags(write=False)
        mask.setflags(write=False)
        return DecisionContext(
            player=self.seat,
            observation=observation,
            legal_action_mask=mask,
            legal_actions=tuple(legal),
            action_info=tuple(
                ActionInfo(action_id, action.kind, action.parameters, str(action))
                for action_id, action in legal.items()
            ),
            public_snapshot=snapshot_from_engine(state),
            player_decisions=self.player_decisions,
        )

    def _choose(
        self, state: HuangState, view: Mapping[str, Any], exclude: frozenset[int]
    ) -> tuple[int, Action]:
        legal = {
            action_id: action
            for action_id, action in legal_action_map(state).items()
            if action_id not in exclude
        }
        if not legal:
            raise NoLegalActionError("no legal action left after exclusions")
        action_id = int(self.agent.select_action(self.make_context(state, view, legal)))
        if action_id not in legal:
            raise NoLegalActionError(f"agent chose action {action_id} outside the legal set")
        return action_id, legal[action_id]

    def _decide_war_remove(
        self,
        state: HuangState,
        view: Mapping[str, Any],
        pending: Mapping[str, Any],
        exclude: frozenset[int],
    ) -> Decision:
        """The game wants all ``count`` removals at once; the engine removes one at a time."""
        count = int(pending["count"])
        cells: list[int] = []
        first_id: Optional[int] = None
        notes: list[str] = []
        for index in range(count):
            if state.phase != Phase.WAR_REMOVE:
                break
            action_id, action = self._choose(state, view, exclude)
            first_id = action_id if first_id is None else first_id
            cells.append(int(action.parameters[0]))
            notes.append(str(action))
            if index < count - 1:
                state.apply_action(action)
        if len(cells) != count or first_id is None:
            raise NoLegalActionError(
                f"war removal needs {count} tiles but only {len(cells)} could be chosen"
            )
        return Decision(
            game_action={"type": "warRemove", "spaces": cells},
            action_id=first_id,
            note="; ".join(notes),
            phase=Phase.WAR_REMOVE.value,
        )

    # -- translation ---------------------------------------------------

    def to_game_action(
        self, state: HuangState, action: Action, view: Mapping[str, Any]
    ) -> JsonObject:
        return translate_action(action, view)


def translate_action(action: Action, view: Mapping[str, Any]) -> JsonObject:
    """Convert a single engine ``Action`` into the game's JSON ``Action``."""
    kind, p = action.kind, action.parameters
    if kind == "leader":
        return {"type": "leader", "color": COLOR_NAMES[p[0]], "to": p[1]}
    if kind == "tile":
        return {"type": "tile", "color": COLOR_NAMES[p[0]], "space": p[1]}
    if kind == "riot":
        return {"type": "riot", "space": p[1], "useLeader": p[0] == 1}
    if kind == "paid_pagoda":
        payment, target, source = p
        result: JsonObject = {
            "type": "pagoda",
            "tri": list(target),
            "useLeader": payment == 1,
        }
        _add_source(result, source, view)
        return result
    if kind == "replace":
        return {
            "type": "replace",
            "tiles": [
                COLOR_NAMES[color] for color, count in enumerate(p) for _ in range(count)
            ],
        }
    if kind == "market_decline":
        return {"type": "market", "index": None}
    if kind == "market_take":
        return {"type": "market", "index": p[0]}
    if kind == "pagoda_decline":
        return {"type": "pagodaBuild", "tri": None}
    if kind == "pagoda_place":
        result = {"type": "pagodaBuild", "tri": list(p[0])}
        _add_source(result, p[1], view)
        return result
    if kind == "blue_stop":
        return {"type": "chain", "space": None}
    if kind == "blue_tile":
        return {"type": "chain", "space": p[0]}
    if kind == "revolt_commit":
        return {"type": "revoltCommit", "count": p[0], "useLeader": bool(p[1])}
    if kind == "war_pass":
        return {"type": "warCommit", "side": None, "count": 0}
    if kind == "war_commit":
        return {"type": "warCommit", "side": p[0], "count": p[1], "useLeader": bool(p[2])}
    if kind == "war_winner":
        return {"type": "warTie", "side": p[0]}
    if kind == "war_remove":
        raise TranslationError("war_remove is multi-step; it is combined by decide()")
    raise TranslationError(f"unknown engine action kind {kind!r}")


def _add_source(result: JsonObject, source, view: Mapping[str, Any]) -> None:
    if source is None:
        return
    pagoda_id = pagoda_ids(view).get(sorted_triangle(source))
    if pagoda_id is None:
        raise TranslationError(f"no pagoda on triangle {tuple(source)} to move")
    result["source"] = pagoda_id
