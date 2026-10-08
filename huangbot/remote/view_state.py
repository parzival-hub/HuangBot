"""Turn a Zhanguo ``GameView`` into a faithful ``HuangState`` ("shadow state")."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Mapping, Optional

from huang import board
from huang.engine import (
    Color,
    HuangState,
    Leader,
    Pagoda,
    Phase,
    RevoltContext,
    WarContext,
)

from .protocol import ShadowStateError
from .information import validate_view, fail

LOGGER = logging.getLogger(__name__)

MARKET_SLOTS = 6


def to_color(name: Any) -> Color:
    try:
        return Color[str(name).upper()]
    except KeyError:
        raise ShadowStateError(f"unknown colour {name!r}") from None


def sorted_triangle(triangle) -> tuple[int, int, int]:
    return tuple(sorted(int(space) for space in triangle))  # type: ignore[return-value]


def pagoda_ids(view: Mapping[str, Any]) -> dict[tuple[int, int, int], int]:
    """Map the sorted triangle of every pagoda on the board to its game id."""
    return {
        sorted_triangle(pagoda["tri"]): int(pagoda["id"])
        for pagoda in view["pagodas"]
        if pagoda.get("tri")
    }


def build_shadow_state(
    view: Mapping[str, Any], *, seat: int, strict: bool = False
) -> HuangState:
    validate_view(view, seat)
    try:
        return _build_shadow_state(view, seat=seat, strict=strict)
    except ShadowStateError as error:
        fail("view", str(error))
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as error:
        fail("view", f"invalid or missing decision data ({type(error).__name__})")


def _build_shadow_state(
    view: Mapping[str, Any], *, seat: int, strict: bool = False
) -> HuangState:
    """Build the engine state in which ``seat`` faces the decision of ``view``.

    Hidden information that the bot cannot see is replaced by placeholders that
    the observation never reads (other seats' hand composition and points).
    ``strict`` turns mismatches between the game's pending decision and the
    engine's own legal actions from warnings into errors.
    """
    players = view["players"]
    try:
        state = HuangState(len(players), short_game=view.get("options", {}).get("shortGame", False), starting_player=-1)
    except ValueError as error:
        raise ShadowStateError(str(error)) from None
    if not 0 <= seat < len(players):
        raise ShadowStateError(f"seat {seat} is outside {len(players)} players")
    state.pending_draws.clear()
    state.draw_continuation = ""
    state.game_over = False

    _fill_board(state, view)
    _fill_players(state, view, seat)
    state.market = _market_slots(view["market"])
    state.bag = Counter({Color.YELLOW: int(view["bagCount"])})
    state.turn_number = int(view["turn"])
    state.active_player = int(view["current"])
    left = int(view["actionsLeft"])
    state.actions_remaining = left - 1 if view["inAction"] else left

    _apply_pending(state, view)

    actor = state.current_actor()
    if actor != seat:
        raise ShadowStateError(
            f"shadow state expects player {actor} to act, not seat {seat} "
            f"(phase {state.phase.value}, pending {_pending_kind(view)})"
        )
    try:
        state.validate_invariants()
    except AssertionError as error:
        raise ShadowStateError(f"engine invariants violated by the view: {error!r}") from None
    for problem in check_pending_consistency(view, state):
        if strict:
            raise ShadowStateError(problem)
        LOGGER.warning("view/engine mismatch: %s", problem)
    return state


def _pending_kind(view: Mapping[str, Any]) -> str:
    pending = view.get("pending")
    return pending["kind"] if pending else "none"


def _fill_board(state: HuangState, view: Mapping[str, Any]) -> None:
    tiles, leaders = view["tiles"], view["leaders"]
    if len(tiles) != board.NUM_CELLS or len(leaders) != board.NUM_CELLS:
        raise ShadowStateError(
            f"view has {len(tiles)} spaces, the bot board has {board.NUM_CELLS}"
        )
    state.tiles = {
        space: to_color(name) for space, name in enumerate(tiles) if name
    }
    state.leaders = {
        space: Leader(int(leader["player"]), to_color(leader["color"]))
        for space, leader in enumerate(leaders)
        if leader
    }
    state.pagodas = {}
    for pagoda in view["pagodas"]:
        if pagoda.get("tri"):
            triangle = sorted_triangle(pagoda["tri"])
            state.pagodas[triangle] = Pagoda(to_color(pagoda["color"]), triangle)


def _fill_players(state: HuangState, view: Mapping[str, Any], seat: int) -> None:
    for index, player in enumerate(view["players"]):
        if index == seat:
            if player.get("hand") is None or player.get("vp") is None:
                raise ShadowStateError("the view hides the bot's own hand or points")
            state.hands[index] = Counter(to_color(name) for name in player["hand"])
            state.points[index] = [int(player["vp"][name]) for name in _COLOR_KEYS]
        else:
            count = int(player["handCount"])
            state.hands[index] = Counter({Color.WHITE: count}) if count > 0 else Counter()
            state.points[index] = [0] * len(Color)


_COLOR_KEYS = tuple(color.name.lower() for color in Color)


def _market_slots(market) -> list[Optional[Color]]:
    colors = [to_color(name) for name in market]
    if len(colors) > MARKET_SLOTS:
        raise ShadowStateError(f"market has {len(colors)} tiles, the engine has {MARKET_SLOTS} slots")
    return colors + [None] * (MARKET_SLOTS - len(colors))


def _apply_pending(state: HuangState, view: Mapping[str, Any]) -> None:
    pending = view.get("pending")
    if pending is None:
        state.phase = Phase.TURN
        return
    kind = pending["kind"]
    if kind == "pagodaBuild":
        triangles = tuple(sorted_triangle(t) for t in pending["triangles"])
        state.phase = Phase.PAGODA_OFFER
        state.post_placement = {
            "cell": triangles[0][0] if triangles else 0,
            "color": int(to_color(pending["color"])),
            "candidate_triangles": triangles,
        }
    elif kind == "market":
        state.phase = Phase.MARKET_CHOICE
        state.post_placement = {
            "cell": 0,
            "color": int(Color.GREEN),
            "candidate_triangles": (),
        }
    elif kind == "chain":
        state.phase = Phase.BLUE_CONTINUE
        state.post_placement = {
            "cell": int(pending["from"]),
            "color": int(Color.BLUE),
            "candidate_triangles": (),
        }
    elif kind == "revolt":
        attacking = pending["stage"] == "attacker"
        state.phase = Phase.REVOLT_ATTACK if attacking else Phase.REVOLT_DEFEND
        state.revolt = RevoltContext(
            attacker=int(pending["attacker"]),
            defender=int(pending["defender"]),
            attacker_cell=int(pending["aSpace"]),
            defender_cell=int(pending["dSpace"]),
            color=to_color(pending["color"]),
            attacker_tiles=int(pending["aCommit"]),
            attacker_leader=int(bool(pending["aLeader"])),
        )
    elif kind in ("war", "warTie", "warRemove"):
        _apply_war(state, view, pending)
    else:
        raise ShadowStateError(f"unknown pending kind {kind!r}")


def _apply_war(state: HuangState, view: Mapping[str, Any], pending: Mapping[str, Any]) -> None:
    war = view.get("war")
    if not war:
        raise ShadowStateError(f"pending {pending['kind']} but the view has no war")
    sides = war["sides"]
    order = [int(player) for player in war["order"]]
    context = WarContext(
        active_player=int(war["unifier"]),
        unifying_cell=int(war["tile"]),
        sides=[frozenset(int(space) for space in side["spaces"]) for side in sides],
        contributor_order=order,
        contributor_index=int(war["idx"]),
        committed_tiles=[int(side["committed"]) for side in sides],
        committed_leaders=[len(side["leaderSupport"]) for side in sides],
    )
    kind = pending["kind"]
    if kind == "war":
        state.phase = Phase.WAR_CONTRIBUTE
    else:
        context.contributor_index = len(order)
        context.strengths = [
            int(side["board"]) + int(side["committed"]) + len(side["leaderSupport"])
            for side in sides
        ]
        if kind == "warTie":
            state.phase = Phase.WAR_TIE
            context.tied_sides = [int(side) for side in pending["tied"]]
        else:
            state.phase = Phase.WAR_REMOVE
            winner = war.get("winner")
            if winner is None:
                raise ShadowStateError("pending warRemove but the war has no winner")
            context.winner = int(winner)
            context.removals_remaining = int(pending["count"])
    state.war = context


def check_pending_consistency(view: Mapping[str, Any], state: HuangState) -> list[str]:
    """Compare the game's own option lists with the engine's legal actions."""
    pending = view.get("pending")
    if not pending:
        return []
    kind = pending["kind"]
    legal = state.legal_actions()
    problems = []
    if kind == "pagodaBuild":
        ids = pagoda_ids(view)
        engine = set()
        for action in legal:
            if action.kind == "pagoda_place":
                target, source = action.parameters
                engine.add((target, None if source is None else ids.get(tuple(source))))
        sources = [int(source) for source in pending.get("sources") or ()]
        game = {
            (sorted_triangle(triangle), source)
            for triangle in pending["triangles"]
            for source in (sources or [None])
        }
        if engine != game:
            problems.append(f"pagodaBuild options differ: engine {sorted(engine, key=str)} vs game {sorted(game, key=str)}")
    elif kind == "chain":
        engine_targets = {a.parameters[0] for a in legal if a.kind == "blue_tile"}
        game_targets = {int(space) for space in pending["targets"]}
        if engine_targets != game_targets:
            problems.append(f"chain targets differ: engine {sorted(engine_targets)} vs game {sorted(game_targets)}")
    elif kind == "warRemove":
        engine_cells = {a.parameters[0] for a in legal if a.kind == "war_remove"}
        game_cells = {int(space) for space in pending["candidates"]}
        if engine_cells != game_cells:
            problems.append(f"warRemove candidates differ: engine {sorted(engine_cells)} vs game {sorted(game_cells)}")
    return problems
