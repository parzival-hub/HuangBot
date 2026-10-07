"""Information-safe, non-neural reference agents for HUANG."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional

from huang import board
from huang.engine import COLOR_NAMES, Color

from .environment import ActionInfo, DecisionContext


ACTION_FAMILIES = {
    "market_decline": "market",
    "market_take": "market",
    "pagoda_decline": "pagoda_offer",
    "pagoda_place": "pagoda_offer",
    "blue_stop": "blue_chain",
    "blue_tile": "blue_chain",
    "revolt_commit": "revolt",
    "war_pass": "war_support",
    "war_commit": "war_support",
    "war_winner": "war_winner",
    "war_remove": "war_remove",
}


class BaselineAgent:
    name = "baseline"

    def __init__(self, *, seed: Optional[int] = None):
        self.random = random.Random(seed)

    def select_action(self, context: DecisionContext) -> int:
        raise NotImplementedError

    def reset_episode(self) -> None:
        """Reset episode-local state; stateless baselines need no action."""


class UniformRandomAgent(BaselineAgent):
    """Choose uniformly among all currently legal action identifiers."""

    name = "uniform-random"

    def select_action(self, context: DecisionContext) -> int:
        if not context.legal_actions:
            raise ValueError("decision context has no legal actions")
        return self.random.choice(context.legal_actions)


class TypeBalancedRandomAgent(BaselineAgent):
    """Choose an action family first, then a concrete action uniformly."""

    name = "type-random"

    def select_action(self, context: DecisionContext) -> int:
        grouped = defaultdict(list)
        for action in context.action_info:
            family = ACTION_FAMILIES.get(action.kind, action.kind)
            grouped[family].append(action.action_id)
        if not grouped:
            raise ValueError("decision context has no legal actions")
        family = self.random.choice(sorted(grouped))
        return self.random.choice(grouped[family])


class HeuristicAgent(BaselineAgent):
    """Simple public-information heuristic used as a stronger sanity check."""

    name = "heuristic"

    def select_action(self, context: DecisionContext) -> int:
        if not context.action_info:
            raise ValueError("decision context has no legal actions")
        scored = [
            (self._score(action, context), action.action_id)
            for action in context.action_info
        ]
        best_score = max(score for score, _ in scored)
        tied = [action_id for score, action_id in scored if score == best_score]
        return self.random.choice(tied)

    def _score(self, action: ActionInfo, context: DecisionContext) -> float:
        snapshot = context.public_snapshot
        tile_by_cell = {tile["cell"]: tile["color"] for tile in snapshot["tiles"]}
        leader_by_cell = {leader["cell"]: leader for leader in snapshot["leaders"]}
        pagoda_cells = {
            cell
            for pagoda in snapshot["pagodas"]
            for cell in pagoda["triangle"]
        }
        kind = action.kind

        if kind == "tile":
            color_value, destination = action.parameters
            color = COLOR_NAMES[Color(color_value)]
            occupied_neighbors = sum(
                neighbor in tile_by_cell or neighbor in leader_by_cell
                for neighbor in board.NEIGHBORS[destination]
            )
            score = 4.0 + occupied_neighbors * 0.35
            for neighbor in board.NEIGHBORS[destination]:
                leader = leader_by_cell.get(neighbor)
                if not leader or leader["player"] != context.player + 1:
                    continue
                if leader["color"] == color:
                    score += 3.0
                elif leader["color"] == "yellow":
                    score += 1.3
            score += self._triangle_bonus(destination, color, tile_by_cell, pagoda_cells)
            if color == "green" and any(snapshot["market"]):
                score += 1.5
            if color == "blue":
                score += 0.6
            return score

        if kind == "leader":
            _, destination = action.parameters
            if destination is None:
                return -8.0
            adjacent_yellow = sum(
                tile_by_cell.get(neighbor) == "yellow"
                for neighbor in board.NEIGHBORS[destination]
            )
            adjacent_occupied = sum(
                neighbor in tile_by_cell or neighbor in leader_by_cell
                for neighbor in board.NEIGHBORS[destination]
            )
            return 2.0 + adjacent_yellow * 1.4 + adjacent_occupied * 0.25

        if kind == "riot":
            payment, target = action.parameters
            score = -0.7 * payment
            if target in pagoda_cells:
                score += 5.0
            if tile_by_cell.get(target) == "yellow":
                for neighbor in board.NEIGHBORS[target]:
                    leader = leader_by_cell.get(neighbor)
                    if not leader:
                        continue
                    score += 4.0 if leader["player"] != context.player + 1 else -5.0
            return score

        if kind == "paid_pagoda":
            payment = action.parameters[0]
            return 9.0 - payment
        if kind == "replace":
            return 0.15 * sum(action.parameters)
        if kind == "market_take":
            color = snapshot["market"][action.parameters[0]]
            return {
                "white": 3.0,
                "green": 2.4,
                "yellow": 2.0,
                "red": 1.8,
                "blue": 1.8,
            }[color]
        if kind == "market_decline":
            return 0.0
        if kind == "pagoda_place":
            return 10.0
        if kind == "pagoda_decline":
            return 0.0
        if kind == "blue_tile":
            return 4.0
        if kind == "blue_stop":
            return 0.0
        if kind == "revolt_commit":
            tiles, leader_bonus = action.parameters
            return leader_bonus * 1.8 + tiles * 0.15
        if kind == "war_commit":
            side, tiles, leader_bonus = action.parameters
            war = snapshot.get("war") or {}
            sides = war.get("sides", [])
            base_red = 0
            if side < len(sides):
                base_red = sum(tile_by_cell.get(cell) == "red" for cell in sides[side])
            return base_red * 0.25 + leader_bonus * 1.2 - tiles * 0.12
        if kind == "war_pass":
            return 0.1
        if kind == "war_winner":
            side = action.parameters[0]
            strengths = (snapshot.get("war") or {}).get("strengths", [])
            return strengths[side] if side < len(strengths) else 0.0
        if kind == "war_remove":
            cell = action.parameters[0]
            score = -sum(
                neighbor in tile_by_cell or neighbor in leader_by_cell
                for neighbor in board.NEIGHBORS[cell]
            )
            if cell in pagoda_cells:
                score -= 8.0
            return score
        return 0.0

    @staticmethod
    def _triangle_bonus(destination, color, tile_by_cell, pagoda_cells) -> float:
        bonus = 0.0
        for triangle in board.TRIANGLES_BY_CELL[destination]:
            other_cells = [cell for cell in triangle if cell != destination]
            if all(tile_by_cell.get(cell) == color for cell in other_cells):
                bonus = max(
                    bonus,
                    7.0 if not set(triangle) & pagoda_cells else 1.0,
                )
        return bonus


def create_baseline(name: str, *, seed: Optional[int] = None) -> BaselineAgent:
    """Create a baseline by its command-line-friendly name."""
    factories = {
        "uniform-random": UniformRandomAgent,
        "type-random": TypeBalancedRandomAgent,
        "heuristic": HeuristicAgent,
    }
    try:
        return factories[name](seed=seed)
    except KeyError as error:
        raise ValueError(
            f"unknown baseline {name!r}; choose from {sorted(factories)}"
        ) from error
