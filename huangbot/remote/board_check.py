"""The one supported board ("Bot-Brett") and lobby compatibility checks."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from huang import board

MIN_PLAYERS = 2
MAX_PLAYERS = 4


def expected_rows() -> list[str]:
    """Zhanguo ``MapDef.rows`` of the bot engine's board ('.' land, '~' river, 'C' capital, '#' void)."""
    width = max(column for _, column in board.COORDS) + 1
    height = max(row for row, _ in board.COORDS) + 1
    rows = []
    for row in range(height):
        cells = []
        for column in range(width):
            coordinate = (row, column)
            if coordinate not in board.COORD_TO_INDEX:
                cells.append("#")
            elif coordinate in board.RIVER_COORDS:
                cells.append("~")
            elif coordinate in board.CAPITAL_COORDS:
                cells.append("C")
            else:
                cells.append(".")
        rows.append("".join(cells))
    return rows


def is_supported_map(map_def: Optional[Mapping[str, Any]]) -> tuple[bool, str]:
    """Compare ``rows`` with the engine's board; name and ``names`` are ignored."""
    rows = map_def.get("rows") if isinstance(map_def, Mapping) else None
    if not isinstance(rows, list) or not all(isinstance(row, str) for row in rows):
        return False, "map has no rows"
    expected = expected_rows()
    if rows == expected:
        return True, ""
    if len(rows) != len(expected):
        return False, f"unsupported map: {len(rows)} rows, the bot plays only the 15-row Bot-Brett"
    for index, (got, want) in enumerate(zip(rows, expected)):
        if got != want:
            return False, f"unsupported map: row {index} differs, the bot plays only Bot-Brett"
    return False, "unsupported map"


def check_lobby(lobby: Mapping[str, Any], seat: int) -> tuple[bool, str]:
    """Return ``(ready, note)`` for the board, player count and seat of a lobby."""
    ok, reason = is_supported_map(lobby.get("map"))
    if not ok:
        return False, reason
    seats = lobby.get("seats")
    count = len(seats) if isinstance(seats, list) else 0
    if not MIN_PLAYERS <= count <= MAX_PLAYERS:
        return False, f"unsupported player count: {count} (the bot plays {MIN_PLAYERS} to {MAX_PLAYERS})"
    if not 0 <= seat < count:
        return False, f"seat {seat} is outside the {count} seats of this game"
    return True, ""
