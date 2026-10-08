"""Hand-built engine positions for the rare decision types."""

from __future__ import annotations

from collections import Counter

from huang import board
from huang.engine import Color, HuangState, Phase, WarContext


def blank_state(players: int = 2, active: int = 0) -> HuangState:
    state = HuangState(players, starting_player=-1)
    state.pending_draws.clear()
    state.draw_continuation = ""
    state.phase = Phase.TURN
    state.active_player = active
    state.actions_remaining = 2
    state.turn_number = 5
    state.bag = Counter({Color.YELLOW: 60})
    state.market = [Color.RED, Color.GREEN, None, Color.WHITE, None, Color.YELLOW]
    return state


def free_land_triangles(count: int):
    """``count`` pairwise disjoint triangles of land spaces that are not capitals."""
    banned = set(board.RIVER_CELLS) | set(board.CAPITAL_CELLS)
    chosen, used = [], set()
    for triangle in board.TRIANGLES:
        if set(triangle) & (banned | used):
            continue
        chosen.append(triangle)
        used |= set(triangle)
        if len(chosen) == count:
            return chosen
    raise AssertionError("not enough triangles")


def state_with_two_yellow_pagodas():
    """Yellow pagoda supply is empty; a third yellow triangle is free."""
    from huang.engine import Pagoda

    state = blank_state()
    first, second, third = free_land_triangles(3)
    for triangle in (first, second, third):
        for cell in triangle:
            state.tiles[cell] = Color.YELLOW
    for triangle in (first, second):
        state.pagodas[triangle] = Pagoda(Color.YELLOW, triangle)
    state.hands[0] = Counter({Color.GREEN: 2, Color.RED: 1, Color.BLUE: 2, Color.YELLOW: 1})
    state.hands[1] = Counter({Color.WHITE: 6})
    return state, first, second, third


def war_state(phase: Phase, *, red_in_winner: int = 3, removals: int = 2, strengths=None):
    """A war between two states around two capitals, resolved up to ``phase``."""
    state = blank_state()
    first_capital, second_capital = board.CAPITAL_CELLS[0], board.CAPITAL_CELLS[1]
    side_a = {first_capital}
    side_b = {second_capital}
    red_cells = [c for c in board.NEIGHBORS[first_capital] if c not in board.RIVER_CELLS][:red_in_winner]
    for cell in red_cells:
        state.tiles[cell] = Color.RED
        side_a.add(cell)
    other = [c for c in board.NEIGHBORS[second_capital] if c not in board.RIVER_CELLS][:1]
    for cell in other:
        state.tiles[cell] = Color.RED
        side_b.add(cell)
    unifying = next(c for c in range(board.NUM_CELLS) if c not in state.tiles and c not in board.RIVER_CELLS)
    state.tiles[unifying] = Color.RED
    state.war = WarContext(
        active_player=0,
        unifying_cell=unifying,
        sides=[frozenset(side_a), frozenset(side_b)],
        contributor_order=[1, 0],
        contributor_index=2,
        committed_tiles=[1, 0],
        committed_leaders=[0, 0],
        strengths=list(strengths or [red_in_winner + 1, 1 + 0]),
        tied_sides=[0, 1] if phase == Phase.WAR_TIE else [],
        winner=0 if phase == Phase.WAR_REMOVE else None,
        removals_remaining=removals if phase == Phase.WAR_REMOVE else 0,
    )
    state.phase = phase
    state.hands[0] = Counter({Color.RED: 2, Color.YELLOW: 2})
    state.hands[1] = Counter({Color.RED: 1, Color.GREEN: 5})
    return state, sorted(red_cells)
