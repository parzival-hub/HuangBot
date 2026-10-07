"""Player-facing wrapper around the OpenSpiel ``python_huang`` game."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from huang.action_codec import encode_action, legal_action_map
from huang.snapshot import snapshot_from_open_spiel_state


MAX_OBSERVATION_PLAYERS = 4
PUBLIC_SCORE_COLORS = ("yellow", "red", "blue", "green", "white")
PUBLIC_SCORE_HISTORY_SIZE = MAX_OBSERVATION_PLAYERS * len(PUBLIC_SCORE_COLORS)
_PUBLIC_AWARD_PATTERN = re.compile(
    r"^P(?P<player>\d+) gained (?P<amount>\d+) "
    r"(?P<color>yellow|red|blue|green|white) VP$"
)
_PUBLIC_SCORE_COLOR_INDEX = {
    color: index for index, color in enumerate(PUBLIC_SCORE_COLORS)
}


def public_score_history_from_engine(engine) -> np.ndarray:
    """Encode publicly announced VP awards without reading hidden scores."""
    scores = np.zeros(
        (MAX_OBSERVATION_PLAYERS, len(PUBLIC_SCORE_COLORS)),
        dtype=np.float32,
    )
    for event in engine.public_history:
        match = _PUBLIC_AWARD_PATTERN.match(event)
        if match is None:
            continue
        player = int(match.group("player"))
        if not 0 <= player < MAX_OBSERVATION_PLAYERS:
            continue
        color = _PUBLIC_SCORE_COLOR_INDEX[match.group("color")]
        scores[player, color] += int(match.group("amount"))
    return (scores / 30.0).reshape(-1)


@dataclass(frozen=True)
class ActionInfo:
    """Public description of one action legal for the acting player."""

    action_id: int
    kind: str
    parameters: tuple
    description: str


@dataclass(frozen=True)
class EnvironmentStep:
    """One player-facing point in an episode; chance nodes are never exposed."""

    observation: np.ndarray
    legal_action_mask: np.ndarray
    legal_actions: tuple[int, ...]
    current_player: Optional[int]
    rewards: np.ndarray
    terminated: bool
    truncated: bool
    player_decisions: int
    chance_events: int

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(frozen=True)
class DecisionContext:
    """Information-safe input supplied to non-neural baseline agents."""

    player: int
    observation: np.ndarray
    legal_action_mask: np.ndarray
    legal_actions: tuple[int, ...]
    action_info: tuple[ActionInfo, ...]
    public_snapshot: dict[str, Any]
    player_decisions: int


@dataclass(frozen=True)
class PolicyContext:
    """Minimal decision data for the neural policy."""

    player: int
    observation: np.ndarray
    legal_action_mask: np.ndarray
    legal_actions: tuple[int, ...]
    player_decisions: int


class HuangEnvironment:
    """Resolve chance internally and expose only actual player decisions.

    ``max_player_decisions`` is a optional match time limit. Reaching it marks
    the episode as truncated without changing the official HUANG engine state.
    """

    def __init__(
        self,
        *,
        players: int = 2,
        short_game: bool = True,
        starting_player: int = -1,
        seed: Optional[int] = None,
        max_player_decisions: Optional[int] = None,
        include_public_score_history: bool = False,
    ):
        if not 2 <= players <= 4:
            raise ValueError("players must be between 2 and 4")
        if short_game and players != 2:
            raise ValueError("short_game is available only with two players")
        if starting_player < -1 or starting_player >= players:
            raise ValueError("starting_player is outside the player range")
        if max_player_decisions is not None and max_player_decisions <= 0:
            raise ValueError("max_player_decisions must be positive or None")

        try:
            import pyspiel
        except ImportError as error:
            raise RuntimeError(
                "HuangEnvironment requires OpenSpiel"
            ) from error
        from huang import open_spiel_game  # noqa: F401

        self.players = players
        self.short_game = bool(short_game)
        self.starting_player = starting_player
        self.max_player_decisions = max_player_decisions
        self.include_public_score_history = bool(include_public_score_history)
        self._pyspiel = pyspiel
        self._random = random.Random(seed)
        self._seed = seed
        self._game = pyspiel.load_game(
            "python_huang("
            f"players={players},"
            f"short_game={self.short_game},"
            f"starting_player={starting_player}"
            ")"
        )
        self._state = None
        self._current_action_map = {}
        self._terminated = False
        self._truncated = False
        self.player_decisions = 0
        self.chance_events = 0
        self.episode_index = -1
        self._public_score_history = np.zeros(
            (MAX_OBSERVATION_PLAYERS, len(PUBLIC_SCORE_COLORS)),
            dtype=np.float32,
        )
        self._public_history_cursor = 0

    @property
    def num_actions(self) -> int:
        return self._game.num_distinct_actions()

    @property
    def observation_size(self) -> int:
        base_size = self._game.observation_tensor_shape()[0]
        if self.include_public_score_history:
            return base_size + PUBLIC_SCORE_HISTORY_SIZE
        return base_size

    @property
    def open_spiel_game(self):
        """Underlying game object for diagnostics and advanced integrations."""
        return self._game

    @property
    def open_spiel_state(self):
        """Full state for diagnostics; it contains hidden information."""
        if self._state is None:
            raise RuntimeError("reset() must be called before accessing state")
        return self._state

    def reset(self, *, seed: Optional[int] = None) -> EnvironmentStep:
        if seed is not None:
            self._seed = seed
            self._random.seed(seed)
        self._state = self._game.new_initial_state()
        self._terminated = False
        self._truncated = False
        self.player_decisions = 0
        self.chance_events = 0
        self.episode_index += 1
        self._public_score_history.fill(0.0)
        self._public_history_cursor = 0
        self._resolve_chance_nodes()
        self._update_public_score_history()
        return self._make_step(np.zeros(self.players, dtype=np.float32))

    def step(self, action_id: int) -> EnvironmentStep:
        action = self.resolve_action(action_id)
        return self.step_resolved_action(action_id, action)

    def resolve_action(self, action_id: int):
        """Resolve an action ID from the map cached with the current step."""
        self._require_active_episode()
        action_id = int(action_id)
        if action_id not in self._current_action_map:
            legal_actions = tuple(self._current_action_map)
            raise ValueError(
                f"action {action_id} is illegal; legal actions: {legal_actions[:20]}"
                + ("..." if len(legal_actions) > 20 else "")
            )
        return self._current_action_map[action_id]

    def step_resolved_action(self, action_id: int, action) -> EnvironmentStep:
        """Apply a previously resolved engine action without decoding it again."""
        self._require_active_episode()
        action_id = int(action_id)
        cached = self._current_action_map.get(action_id)
        if cached is None or cached != action:
            raise ValueError("resolved action does not match the current legal map")
        if encode_action(self._state.engine, action) != action_id:
            raise ValueError("resolved action ID changed before application")
        self._state.engine.apply_action(action)
        self.player_decisions += 1
        self._resolve_chance_nodes()
        self._update_public_score_history()
        self._terminated = self._state.is_terminal()
        if (
            not self._terminated
            and self.max_player_decisions is not None
            and self.player_decisions >= self.max_player_decisions
        ):
            self._truncated = True

        rewards = (
            np.asarray(self._state.returns(), dtype=np.float32)
            if self._terminated
            else np.zeros(self.players, dtype=np.float32)
        )
        return self._make_step(rewards)

    def policy_context(self) -> PolicyContext:
        """Return model inputs without snapshots or human-readable actions."""
        self._require_active_episode()
        current_player = int(self._state.current_player())
        if current_player < 0:
            raise RuntimeError("chance node escaped automatic resolution")
        observation = self.observation_for_player(current_player)
        legal_actions = tuple(self._current_action_map)
        legal_mask = np.zeros(self.num_actions, dtype=np.bool_)
        legal_mask[list(legal_actions)] = True
        observation.setflags(write=False)
        legal_mask.setflags(write=False)
        return PolicyContext(
            player=current_player,
            observation=observation,
            legal_action_mask=legal_mask,
            legal_actions=legal_actions,
            player_decisions=self.player_decisions,
        )

    def decision_context(self) -> DecisionContext:
        """Return only the acting player's view plus public board information."""
        self._require_active_episode()
        context = self.policy_context()
        action_info = tuple(
            ActionInfo(
                action_id=action_id,
                kind=action.kind,
                parameters=action.parameters,
                description=str(action),
            )
            for action_id, action in self._current_action_map.items()
        )
        return DecisionContext(
            player=context.player,
            observation=context.observation,
            legal_action_mask=context.legal_action_mask,
            legal_actions=context.legal_actions,
            action_info=action_info,
            public_snapshot=snapshot_from_open_spiel_state(self._state),
            player_decisions=self.player_decisions,
        )

    def snapshot(self, *, title: Optional[str] = None) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("reset() must be called before creating a snapshot")
        return snapshot_from_open_spiel_state(
            self._state,
            title=title,
            metadata={
                "match_terminated": self._terminated,
                "match_truncated": self._truncated,
                "match_player_decisions": self.player_decisions,
                "match_chance_events": self.chance_events,
            },
        )

    def observation_for_player(self, player: int) -> np.ndarray:
        """Return the configured information-safe observation for one seat."""
        if self._state is None:
            raise RuntimeError("reset() must be called before observing state")
        if not 0 <= player < self.players:
            raise ValueError("player is outside the environment")
        observation = np.asarray(
            self._state.observation_tensor(player),
            dtype=np.float32,
        )
        if self.include_public_score_history:
            public_scores = (self._public_score_history / 30.0).reshape(-1)
            observation = np.concatenate((observation, public_scores)).astype(
                np.float32,
                copy=False,
            )
        observation.setflags(write=False)
        return observation

    def _resolve_chance_nodes(self) -> None:
        while not self._state.is_terminal() and self._state.is_chance_node():
            outcomes = self._state.chance_outcomes()
            if not outcomes:
                raise RuntimeError("OpenSpiel chance node has no outcomes")
            threshold = self._random.random()
            cumulative = 0.0
            selected = outcomes[-1][0]
            for outcome, probability in outcomes:
                cumulative += probability
                if threshold <= cumulative:
                    selected = outcome
                    break
            self._state.apply_action(selected)
            self.chance_events += 1

    def _update_public_score_history(self) -> None:
        history = self._state.engine.public_history
        for event in history[self._public_history_cursor:]:
            match = _PUBLIC_AWARD_PATTERN.match(event)
            if match is None:
                continue
            player = int(match.group("player"))
            if not 0 <= player < MAX_OBSERVATION_PLAYERS:
                continue
            color = _PUBLIC_SCORE_COLOR_INDEX[match.group("color")]
            self._public_score_history[player, color] += int(
                match.group("amount")
            )
        self._public_history_cursor = len(history)

    def _make_step(self, rewards: np.ndarray) -> EnvironmentStep:
        if self._terminated or self._truncated:
            self._current_action_map = {}
            observation = np.zeros(self.observation_size, dtype=np.float32)
            legal_mask = np.zeros(self.num_actions, dtype=np.bool_)
            legal_actions = ()
            current_player = None
        else:
            current_player = int(self._state.current_player())
            if current_player < 0:
                raise RuntimeError("chance node escaped automatic resolution")
            observation = self.observation_for_player(current_player)
            self._current_action_map = legal_action_map(self._state.engine)
            legal_actions = tuple(self._current_action_map)
            legal_mask = np.zeros(self.num_actions, dtype=np.bool_)
            legal_mask[list(legal_actions)] = True

        observation.setflags(write=False)
        legal_mask.setflags(write=False)
        rewards = np.asarray(rewards, dtype=np.float32)
        rewards.setflags(write=False)
        return EnvironmentStep(
            observation=observation,
            legal_action_mask=legal_mask,
            legal_actions=legal_actions,
            current_player=current_player,
            rewards=rewards,
            terminated=self._terminated,
            truncated=self._truncated,
            player_decisions=self.player_decisions,
            chance_events=self.chance_events,
        )

    def _require_active_episode(self) -> None:
        if self._state is None:
            raise RuntimeError("reset() must be called before step()")
        if self._terminated:
            raise RuntimeError("episode has terminated; call reset()")
        if self._truncated:
            raise RuntimeError("episode has been truncated; call reset()")
