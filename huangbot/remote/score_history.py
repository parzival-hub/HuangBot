"""The 20 public-score features appended to the observation of newer checkpoints."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from huangbot.environment import MAX_OBSERVATION_PLAYERS, PUBLIC_SCORE_COLORS

from .protocol import COLOR_INDEX


def public_score_history(view: Mapping[str, Any], players: Optional[int] = None) -> np.ndarray:
    """Cumulative publicly announced VP per seat and colour, divided by 30 once.

    This is the arithmetic of ``HuangEnvironment.observation_for_player``: sum
    every ``{p, color, n}`` found in ``view.log[*].vp``, divide by 30 and
    flatten player-major in colour order yellow, red, blue, green, white.
    """
    scores = np.zeros(
        (MAX_OBSERVATION_PLAYERS, len(PUBLIC_SCORE_COLORS)), dtype=np.float32
    )
    limit = MAX_OBSERVATION_PLAYERS if players is None else min(players, MAX_OBSERVATION_PLAYERS)
    for entry in view.get("log") or ():
        for gain in entry.get("vp") or ():
            player = int(gain["p"])
            if 0 <= player < limit:
                scores[player, COLOR_INDEX[gain["color"]]] += int(gain["n"])
    return (scores / 30.0).reshape(-1)
