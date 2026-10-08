"""Last-resort actions derived from the view alone, so a game never hangs."""

from __future__ import annotations

from typing import Any, Mapping

from .protocol import JsonObject


def safe_action(view: Mapping[str, Any]) -> JsonObject:
    """Return the most passive action the game accepts for ``view.pending``."""
    pending = view.get("pending")
    if not pending:
        return {"type": "replace", "tiles": []}
    kind = pending["kind"]
    if kind == "pagodaBuild":
        return {"type": "pagodaBuild", "tri": None}
    if kind == "market":
        return {"type": "market", "index": None}
    if kind == "chain":
        return {"type": "chain", "space": None}
    if kind == "revolt":
        return {"type": "revoltCommit", "count": 0}
    if kind == "war":
        return {"type": "warCommit", "side": None, "count": 0}
    if kind == "warTie":
        return {"type": "warTie", "side": pending["tied"][0]}
    if kind == "warRemove":
        return {"type": "warRemove", "spaces": list(pending["candidates"][: pending["count"]])}
    raise ValueError(f"unknown pending kind {kind!r}")
