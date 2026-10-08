"""A small structural validator for the game's ``Action`` type (PART B4)."""

from __future__ import annotations

COLORS = {"yellow", "red", "blue", "green", "white"}


def _int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _optional_int(value):
    return value is None or _int(value)


def _bool_or_missing(action, key):
    return key not in action or isinstance(action[key], bool)


def _triangle(value):
    return isinstance(value, list) and len(value) == 3 and all(_int(v) for v in value)


_SCHEMA = {
    "leader": ({"color", "to"}, set(), lambda a: a["color"] in COLORS and _optional_int(a["to"])),
    "tile": ({"color", "space"}, set(), lambda a: a["color"] in COLORS and _int(a["space"])),
    "riot": ({"space"}, {"useLeader"}, lambda a: _int(a["space"]) and _bool_or_missing(a, "useLeader")),
    "pagoda": (
        {"tri"},
        {"useLeader", "source"},
        lambda a: _triangle(a["tri"]) and _bool_or_missing(a, "useLeader") and _optional_int(a.get("source")),
    ),
    "replace": (
        {"tiles"},
        set(),
        lambda a: isinstance(a["tiles"], list) and len(a["tiles"]) <= 6 and all(t in COLORS for t in a["tiles"]),
    ),
    "pagodaBuild": (
        {"tri"},
        {"source"},
        lambda a: (a["tri"] is None or _triangle(a["tri"])) and _optional_int(a.get("source")),
    ),
    "market": ({"index"}, set(), lambda a: _optional_int(a["index"])),
    "chain": ({"space"}, set(), lambda a: _optional_int(a["space"])),
    "revoltCommit": ({"count"}, {"useLeader"}, lambda a: _int(a["count"]) and _bool_or_missing(a, "useLeader")),
    "warCommit": (
        {"side", "count"},
        {"useLeader"},
        lambda a: _optional_int(a["side"]) and _int(a["count"]) and _bool_or_missing(a, "useLeader"),
    ),
    "warTie": ({"side"}, set(), lambda a: _int(a["side"])),
    "warRemove": (
        {"spaces"},
        set(),
        lambda a: isinstance(a["spaces"], list) and all(_int(s) for s in a["spaces"]),
    ),
}


def is_valid_game_action(action) -> bool:
    if not isinstance(action, dict) or action.get("type") not in _SCHEMA:
        return False
    required, optional, check = _SCHEMA[action["type"]]
    keys = set(action) - {"type"}
    return required <= keys <= required | optional and bool(check(action))
