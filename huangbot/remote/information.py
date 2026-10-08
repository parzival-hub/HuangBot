"""Validate the player-visible information required by each observation schema."""
from __future__ import annotations

from collections.abc import Mapping

from huang import board

from .protocol import COLOR_NAMES, GameInformationError


def fail(path, reason):
    raise GameInformationError(f"Required game information {path}: {reason}")


def required(value, key, path):
    if not isinstance(value, Mapping) or key not in value:
        fail(f"{path}.{key}", "missing field")
    return value[key]


def mapping(value, path):
    if not isinstance(value, Mapping):
        fail(path, "expected an object")
    return value


def array(value, path):
    if not isinstance(value, list):
        fail(path, "expected an array")
    return value


def integer(value, path, *, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        fail(path, "expected an integer in the allowed range")
    return value


def boolean(value, path):
    if type(value) is not bool:
        fail(path, "expected a boolean")
    return value


def color(value, path):
    if value not in COLOR_NAMES:
        fail(path, "expected a known tile colour")
    return value


def cell(value, path):
    return integer(value, path, maximum=board.NUM_CELLS - 1)


def validate_view(view, seat, *, score_history=False, tile_belief=False):
    mapping(view, "view")
    map_def = mapping(required(view, "map", "view"), "view.map")
    rows = array(required(map_def, "rows", "view.map"), "view.map.rows")
    if not all(isinstance(row, str) for row in rows):
        fail("view.map.rows", "expected board rows as strings")
    players = array(required(view, "players", "view"), "view.players")
    if not 2 <= len(players) <= 4:
        fail("view.players", "expected two to four players")
    integer(seat, "seat", maximum=len(players) - 1)
    for index, player in enumerate(players):
        path = f"view.players[{index}]"
        mapping(player, path)
        count = integer(required(player, "handCount", path), path + ".handCount")
        if index == seat:
            hand = array(required(player, "hand", path), path + ".hand")
            for name in hand:
                color(name, path + ".hand")
            if len(hand) != count:
                fail(path + ".handCount", "does not match the bot's own hand")
            vp = mapping(required(player, "vp", path), path + ".vp")
            for name in COLOR_NAMES:
                integer(required(vp, name, path + ".vp"), path + ".vp." + name)
        # Other players' hand colours and private points are never read.
    for field in ("tiles", "leaders"):
        values = array(required(view, field, "view"), "view." + field)
        if len(values) != board.NUM_CELLS:
            fail("view." + field, f"expected {board.NUM_CELLS} board spaces")
        for index, value in enumerate(values):
            if value is None:
                continue
            path = f"view.{field}[{index}]"
            if field == "tiles":
                color(value, path)
            else:
                mapping(value, path)
                color(required(value, "color", path), path + ".color")
                integer(required(value, "player", path), path + ".player", maximum=len(players) - 1)
    market = array(required(view, "market", "view"), "view.market")
    if len(market) > 6:
        fail("view.market", "expected at most six tiles")
    for name in market:
        color(name, "view.market")
    for index, pagoda in enumerate(array(required(view, "pagodas", "view"), "view.pagodas")):
        path = f"view.pagodas[{index}]"
        mapping(pagoda, path)
        integer(required(pagoda, "id", path), path + ".id")
        color(required(pagoda, "color", path), path + ".color")
        triangle = required(pagoda, "tri", path)
        if triangle is not None:
            array(triangle, path + ".tri")
            if len(triangle) != 3:
                fail(path + ".tri", "expected three spaces")
            for space in triangle:
                cell(space, path + ".tri")
            if len(set(triangle)) != 3:
                fail(path + ".tri", "expected distinct spaces")
    integer(required(view, "bagCount", "view"), "view.bagCount")
    integer(required(view, "turn", "view"), "view.turn")
    integer(required(view, "current", "view"), "view.current", maximum=len(players) - 1)
    integer(required(view, "actionsLeft", "view"), "view.actionsLeft", maximum=3)
    boolean(required(view, "inAction", "view"), "view.inAction")
    pending = required(view, "pending", "view")
    if pending is not None:
        mapping(pending, "view.pending")
        kind = required(pending, "kind", "view.pending")
        fields = {
            "market": (), "pagodaBuild": ("color", "triangles", "sources"),
            "chain": ("from", "targets"),
            "revolt": ("stage", "attacker", "defender", "aSpace", "dSpace", "color", "aCommit", "aLeader"),
            "war": (), "warTie": ("tied",), "warRemove": ("count", "candidates"),
        }
        if not isinstance(kind, str) or kind not in fields:
            fail("view.pending.kind", "unsupported decision kind")
        integer(required(pending, "player", "view.pending"), "view.pending.player", maximum=len(players) - 1)
        for field in fields[kind]:
            required(pending, field, "view.pending")
        if kind == "pagodaBuild":
            color(pending["color"], "view.pending.color")
            for triangle in array(pending["triangles"], "view.pending.triangles"):
                array(triangle, "view.pending.triangles[]")
                if len(triangle) != 3:
                    fail("view.pending.triangles", "expected three spaces per triangle")
                for space in triangle:
                    cell(space, "view.pending.triangles")
            for source in array(pending["sources"], "view.pending.sources"):
                integer(source, "view.pending.sources")
        elif kind == "chain":
            cell(pending["from"], "view.pending.from")
            for target in array(pending["targets"], "view.pending.targets"):
                cell(target, "view.pending.targets")
        elif kind == "revolt":
            if pending["stage"] not in ("attacker", "defender"):
                fail("view.pending.stage", "expected attacker or defender")
            for field in ("attacker", "defender"):
                integer(pending[field], "view.pending." + field, maximum=len(players) - 1)
            for field in ("aSpace", "dSpace"):
                cell(pending[field], "view.pending." + field)
            color(pending["color"], "view.pending.color")
            integer(pending["aCommit"], "view.pending.aCommit", maximum=6)
            boolean(pending["aLeader"], "view.pending.aLeader")
        elif kind == "warTie":
            array(pending["tied"], "view.pending.tied")
        elif kind == "warRemove":
            integer(pending["count"], "view.pending.count", minimum=1)
            for space in array(pending["candidates"], "view.pending.candidates"):
                cell(space, "view.pending.candidates")
        if kind.startswith("war"):
            war = mapping(required(view, "war", "view"), "view.war")
            for field in ("sides", "order", "unifier", "tile", "idx"):
                required(war, field, "view.war")
            sides = array(war["sides"], "view.war.sides")
            if len(sides) < 2:
                fail("view.war.sides", "expected at least two warring sides")
            order = array(war["order"], "view.war.order")
            for player in order:
                integer(player, "view.war.order", maximum=len(players) - 1)
            integer(war["unifier"], "view.war.unifier", maximum=len(players) - 1)
            cell(war["tile"], "view.war.tile")
            integer(war["idx"], "view.war.idx", maximum=len(order))
            for index, side in enumerate(sides):
                path = f"view.war.sides[{index}]"
                mapping(side, path)
                for field in ("spaces", "committed", "leaderSupport"):
                    required(side, field, path)
                for space in array(side["spaces"], path + ".spaces"):
                    cell(space, path + ".spaces")
                integer(side["committed"], path + ".committed")
                for player in array(side["leaderSupport"], path + ".leaderSupport"):
                    integer(player, path + ".leaderSupport", maximum=len(players) - 1)
                if kind != "war":
                    integer(required(side, "board", path), path + ".board")
            if kind == "warRemove":
                integer(required(war, "winner", "view.war"), "view.war.winner", maximum=len(sides) - 1)
            if kind == "warTie":
                for side in pending["tied"]:
                    integer(side, "view.pending.tied", maximum=len(sides) - 1)
    if score_history or tile_belief:
        if required(view, "logComplete", "view") is not True:
            fail("view.logComplete", "must confirm the full public score log from game setup")
        log = array(required(view, "log", "view"), "view.log")
        for index, entry in enumerate(log):
            path = f"view.log[{index}]"
            mapping(entry, path)
            for gain in array(entry.get("vp", []), path + ".vp"):
                mapping(gain, path + ".vp")
                integer(required(gain, "p", path + ".vp"), path + ".vp.p", maximum=len(players) - 1)
                color(required(gain, "color", path + ".vp"), path + ".vp.color")
                integer(required(gain, "n", path + ".vp"), path + ".vp.n")
    if tile_belief:
        options = mapping(required(view, "options", "view"), "view.options")
        boolean(required(options, "shortGame", "view.options"), "view.options.shortGame")
        history = mapping(required(view, "tileHistory", "view"), "view.tileHistory")
        integer(required(history, "version", "view.tileHistory"), "view.tileHistory.version", minimum=1, maximum=1)
        if required(history, "complete", "view.tileHistory") is not True:
            fail("view.tileHistory.complete", "must confirm the full tile ledger from game setup")
        array(required(history, "events", "view.tileHistory"), "view.tileHistory.events")


def validate_ext_state(state):
    mapping(state, "state")
    for field in ("protocol", "gameId", "status", "lobby", "seat", "version", "yourTurn", "view"):
        required(state, field, "state")
    integer(state["seat"], "state.seat")
    integer(state["version"], "state.version")
    boolean(state["yourTurn"], "state.yourTurn")
    if state["status"] not in ("lobby", "playing", "finished"):
        fail("state.status", "unsupported game status")
    if state["status"] == "playing" and state["view"] is None:
        fail("state.view", "missing game view during play")
    lobby = mapping(state["lobby"], "state.lobby")
    map_def = mapping(required(lobby, "map", "state.lobby"), "state.lobby.map")
    rows = array(required(map_def, "rows", "state.lobby.map"), "state.lobby.map.rows")
    if not all(isinstance(row, str) for row in rows):
        fail("state.lobby.map.rows", "expected board rows as strings")
    array(required(lobby, "seats", "state.lobby"), "state.lobby.seats")
