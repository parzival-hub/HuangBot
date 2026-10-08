"""Test oracle: HuangState -> Zhanguo GameView, and Zhanguo action -> engine action.

``engine_to_view`` is the inverse of ``build_shadow_state`` (PART D) and
``game_action_to_engine`` is a second, independent implementation of the action
table of PART E. Neither is shipped.
"""

from __future__ import annotations

import re

from huang import board
from huang.engine import Color, HuangState, Phase

COLORS = ["yellow", "red", "blue", "green", "white"]
PAGODA_COUNTS = [2, 2, 2, 2, 1]
GOLDEN_ROWS = [
    ".........~~~~##",
    "..C....~~..~.##",
    "......~~......#",
    "~...~~........#",
    "~....~C.......C",
    "~....~........#",
    "~.....~...C....",
    "~.....~~~.....#",
    "~C.......~~~~~~",
    "~....C.......~#",
    "#~...........~#",
    "#~.........C..#",
    "#~~............",
    "#~~...........#",
    "##~~~~~~...####",
]
AWARD = re.compile(r"^P(\d+) gained (\d+) (yellow|red|blue|green|white) VP$")
COORDINATE_CELLS = {board.coordinate_name(cell): cell for cell in range(board.NUM_CELLS)}


def _tile_history(state, viewer):
    """Seat-filtered wire ledger, derived independently from engine events."""
    replacements = iter(line for line in state.private_history[viewer] if line.startswith("replaced "))
    events = []
    for line in state.public_history:
        placed = re.match(r"^P(\d+) placed (\w+) tile at (.+)$", line)
        market = re.match(r"^P(\d+) took (\w+) from market slot (\d+)$", line)
        replace = re.match(r"^P(\d+) replaced (\d+) hidden tile\(s\)$", line)
        if placed:
            player, name, coordinate = placed.groups()
            events.append({"kind": "place", "player": int(player), "color": name,
                           "space": COORDINATE_CELLS[coordinate]})
        elif market:
            player, name, slot = market.groups()
            events.append({"kind": "market", "player": int(player), "color": name, "slot": int(slot)})
        elif replace:
            player, count = map(int, replace.groups())
            event = {"kind": "replace", "player": player, "count": count}
            if player == viewer:
                private = next(replacements)
                counts = list(map(int, private[len("replaced "):].split(",")))
                event["colors"] = [name for name, amount in zip(COLORS, counts) for _ in range(amount)]
            events.append(event)
        else:
            for kind, pattern in (
                ("riot", r"^P(\d+) caused a riot with (\d+) blue tile"),
                ("pagoda", r"^P(\d+) paid (\d+) green tile"),
                ("revolt", r"^P(\d+) committed (\d+) yellow tile"),
                ("war", r"^P(\d+) supported war side \d+ with (\d+) red tile"),
            ):
                match = re.match(pattern, line)
                if match:
                    player, count = map(int, match.groups())
                    events.append({"kind": kind, "player": player, "count": count})
                    break
    return {"version": 1, "complete": True, "events": events}


def winning_red_cells(state: HuangState) -> list[int]:
    war = state.war
    return sorted(c for c in war.sides[war.winner] if state.tiles.get(c) == Color.RED)


def compact_market(state: HuangState) -> HuangState:
    """Copy of ``state`` as the game presents it.

    The market is compacted like ``view.market``, and a pending tile removal is
    clamped to the tiles that can be removed (the engine itself may owe more
    removals than there are red tiles and then simply removes them all; the
    oracle assumes the game never asks for more spaces than it offers).
    """
    clone = state.clone()
    tiles = [color for color in clone.market if color is not None]
    clone.market = tiles + [None] * (6 - len(tiles))
    if clone.phase == Phase.WAR_REMOVE:
        clone.war.removals_remaining = min(clone.war.removals_remaining, len(winning_red_cells(clone)))
    return clone


def _tri(triangle) -> list[int]:
    return sorted(triangle)


def _pagoda_entries(state: HuangState) -> list[dict]:
    entries, next_id = [], 0
    for color, total in zip(Color, PAGODA_COUNTS):
        placed = sorted(t for t, p in state.pagodas.items() if p.color == color)
        for index in range(total):
            entries.append(
                {
                    "id": next_id,
                    "color": COLORS[color],
                    "tri": _tri(placed[index]) if index < len(placed) else None,
                }
            )
            next_id += 1
    return entries


def _war_data(state: HuangState):
    war = state.war
    if war is None:
        return None
    sides = []
    for index, cells in enumerate(war.sides):
        red = sum(state.tiles.get(cell) == Color.RED for cell in cells)
        leaders = war.committed_leaders[index]
        if war.strengths:
            red = war.strengths[index] - war.committed_tiles[index] - leaders
        sides.append(
            {
                "spaces": sorted(cells),
                "leaders": [
                    {"space": c, "player": state.leaders[c].player, "color": COLORS[state.leaders[c].color]}
                    for c in sorted(cells)
                    if c in state.leaders
                ],
                "board": red,
                "committed": war.committed_tiles[index],
                "leaderSupport": war.contributor_order[:leaders],
            }
        )
    data = {
        "unifier": war.active_player,
        "tile": war.unifying_cell,
        "sides": sides,
        "order": list(war.contributor_order),
        "idx": war.contributor_index,
    }
    if war.winner is not None:
        data["winner"] = war.winner
    return data


def _pending(state: HuangState):
    actor = state.current_actor()
    phase = state.phase
    if phase in (Phase.TURN, Phase.TERMINAL) or state.game_over:
        return None
    if phase == Phase.MARKET_CHOICE:
        return {"kind": "market", "player": actor}
    if phase == Phase.PAGODA_OFFER:
        placement = state.post_placement
        color = Color(placement["color"])
        existing = sorted(t for t, p in state.pagodas.items() if p.color == color)
        ids = {t: entry["id"] for entry in _pagoda_entries(state) if entry["tri"] for t in [tuple(entry["tri"])]}
        supply = PAGODA_COUNTS[color] - len(existing)
        return {
            "kind": "pagodaBuild",
            "player": actor,
            "color": COLORS[color],
            "triangles": [_tri(t) for t in placement["candidate_triangles"]],
            "sources": [] if supply > 0 else [ids[t] for t in existing],
        }
    if phase == Phase.BLUE_CONTINUE:
        targets = [a.parameters[0] for a in state.legal_actions() if a.kind == "blue_tile"]
        return {
            "kind": "chain",
            "player": actor,
            "from": state.post_placement["cell"],
            "targets": targets,
        }
    if phase in (Phase.REVOLT_ATTACK, Phase.REVOLT_DEFEND):
        revolt = state.revolt
        yellow = lambda cell: sum(state.tiles.get(n) == Color.YELLOW for n in board.NEIGHBORS[cell])
        return {
            "kind": "revolt",
            "player": actor,
            "stage": "attacker" if phase == Phase.REVOLT_ATTACK else "defender",
            "attacker": revolt.attacker,
            "defender": revolt.defender,
            "color": COLORS[revolt.color],
            "aSpace": revolt.attacker_cell,
            "dSpace": revolt.defender_cell,
            "aBase": yellow(revolt.attacker_cell),
            "dBase": yellow(revolt.defender_cell),
            "aCommit": revolt.attacker_tiles,
            "dCommit": 0,
            "aLeader": bool(revolt.attacker_leader),
            "dLeader": False,
        }
    if phase == Phase.WAR_CONTRIBUTE:
        return {"kind": "war", "player": actor}
    if phase == Phase.WAR_TIE:
        return {"kind": "warTie", "player": actor, "tied": list(state.war.tied_sides)}
    if phase == Phase.WAR_REMOVE:
        return {
            "kind": "warRemove",
            "player": actor,
            "count": min(state.war.removals_remaining, len(winning_red_cells(state))),
            "candidates": winning_red_cells(state),
        }
    raise AssertionError(f"no pending for phase {phase}")


def _log(state: HuangState) -> list[dict]:
    log = []
    for line in state.public_history:
        match = AWARD.match(line)
        if match:
            player, amount, color = int(match.group(1)), int(match.group(2)), match.group(3)
            log.append(
                {
                    "turn": state.turn_number,
                    "p": player,
                    "t": "tile",
                    "vp": [{"p": player, "color": color, "n": amount}],
                }
            )
        else:
            log.append({"turn": state.turn_number, "p": state.active_player, "t": "note", "text": line})
    return log


def engine_to_view(state: HuangState, viewer: int) -> dict:
    """The game view that ``viewer`` would receive in ``state`` (market compacted)."""
    in_action = state.phase != Phase.TURN
    market = [COLORS[c] for c in state.market if c is not None]
    players = []
    for index in range(state.players):
        own = index == viewer
        players.append(
            {
                "name": f"P{index}",
                "dynasty": "Han",
                "handCount": sum(state.hands[index].values()),
                "hand": [COLORS[c] for c in Color for _ in range(state.hands[index][c])] if own else None,
                "vp": dict(zip(COLORS, state.points[index])) if own else None,
            }
        )
    return {
        "options": {"shortGame": state.short_game},
        "map": {"name": "Bot-Brett", "rows": list(GOLDEN_ROWS)},
        "market": market,
        "tiles": [COLORS[state.tiles[c]] if c in state.tiles else None for c in range(board.NUM_CELLS)],
        "leaders": [
            {"player": state.leaders[c].player, "color": COLORS[state.leaders[c].color]}
            if c in state.leaders
            else None
            for c in range(board.NUM_CELLS)
        ],
        "pagodas": _pagoda_entries(state),
        "current": state.active_player,
        "turn": state.turn_number,
        "actionsLeft": state.actions_remaining + 1 if in_action else state.actions_remaining,
        "inAction": in_action,
        "pending": _pending(state),
        "queue": [],
        "war": _war_data(state),
        "unification": None,
        "log": _log(state),
        "logComplete": True,
        "tileHistory": _tile_history(state, viewer),
        "gameOver": None,
        "lastPlaced": [],
        "bagCount": state.bag_total,
        "players": players,
        "you": viewer,
    }


def game_action_to_engine(state: HuangState, game_action: dict, view: dict) -> list:
    """Independent inverse of the PART E table; returns the legal engine action(s)."""
    kind = game_action["type"]
    ids = {entry["id"]: tuple(entry["tri"]) for entry in view["pagodas"] if entry["tri"]}

    def source():
        return ids[game_action["source"]] if "source" in game_action else None

    def payment():
        return 1 if game_action.get("useLeader") else 2

    if kind == "warRemove":
        scratch, result = state.clone(), []
        for cell in game_action["spaces"]:
            action = _lookup(scratch, "war_remove", (cell,))
            result.append(action)
            scratch.apply_action(action)
        return result
    colour = {name: index for index, name in enumerate(COLORS)}
    table = {
        "leader": lambda: ("leader", (colour[game_action["color"]], game_action["to"])),
        "tile": lambda: ("tile", (colour[game_action["color"]], game_action["space"])),
        "riot": lambda: ("riot", (payment(), game_action["space"])),
        "pagoda": lambda: ("paid_pagoda", (payment(), tuple(game_action["tri"]), source())),
        "replace": lambda: ("replace", tuple(game_action["tiles"].count(name) for name in COLORS)),
        "pagodaBuild": lambda: ("pagoda_decline", ())
        if game_action["tri"] is None
        else ("pagoda_place", (tuple(game_action["tri"]), source())),
        "market": lambda: ("market_decline", ())
        if game_action["index"] is None
        else (
            "market_take",
            ([slot for slot, c in enumerate(state.market) if c is not None][game_action["index"]],),
        ),
        "chain": lambda: ("blue_stop", ()) if game_action["space"] is None else ("blue_tile", (game_action["space"],)),
        "revoltCommit": lambda: ("revolt_commit", (game_action["count"], int(game_action.get("useLeader", False)))),
        "warCommit": lambda: ("war_pass", ())
        if game_action["side"] is None
        else (
            "war_commit",
            (game_action["side"], game_action["count"], int(game_action.get("useLeader", False))),
        ),
        "warTie": lambda: ("war_winner", (game_action["side"],)),
    }
    kind_name, parameters = table[kind]()
    return [_lookup(state, kind_name, parameters)]


def _lookup(state: HuangState, kind: str, parameters: tuple):
    for action in state.legal_actions():
        if action.kind == kind and tuple(action.parameters) == tuple(parameters):
            return action
    raise AssertionError(f"{kind}{parameters} is not legal in phase {state.phase}")


def resolve_chance(state: HuangState, rng) -> None:
    """Resolve draws and the starting-player choice randomly."""
    while state.current_actor() == -1:
        outcomes = state.chance_outcomes()
        if not outcomes:
            raise AssertionError("chance node without outcomes")
        values, weights = zip(*outcomes)
        state.apply_chance(rng.choices(values, weights)[0])


def public_history_scores(state: HuangState):
    """Copy of ``HuangEnvironment._update_public_score_history`` over a whole history."""
    import numpy as np

    scores = np.zeros((4, 5), dtype=np.float32)
    for event in state.public_history:
        match = AWARD.match(event)
        if match is None:
            continue
        player = int(match.group(1))
        if not 0 <= player < 4:
            continue
        scores[player, COLORS.index(match.group(3))] += int(match.group(2))
    return (scores / 30.0).reshape(-1)
