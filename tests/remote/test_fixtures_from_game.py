"""Replay ExtState fixtures saved from the real Zhanguo server (yourTurn: true)."""

import json
from pathlib import Path

import pytest

from action_schema import is_valid_game_action
from agents import RandomAgent
from huang.action_codec import legal_action_map
from huangbot.remote.adapter import ZhanguoAdapter
from huangbot.remote.view_state import build_shadow_state

FIXTURES = Path(__file__).parent / "fixtures" / "zhanguo"
FILES = sorted(FIXTURES.glob("*.json")) if FIXTURES.is_dir() else []


def test_fixture_directory_is_optional():
    if not FILES:
        pytest.skip("no fixtures in tests/remote/fixtures/zhanguo/ (copy them from the game repo)")


@pytest.mark.parametrize("path", FILES, ids=[p.name for p in FILES])
def test_fixture(path):
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["yourTurn"] is True
    seat, view = state["seat"], state["view"]
    shadow = build_shadow_state(view, seat=seat, strict=True)
    assert shadow.current_actor() == seat
    assert legal_action_map(shadow)
    agent = RandomAgent(seed=1)
    adapter = ZhanguoAdapter(agent, seat=seat, use_score_history=True, strict=True)
    decision = adapter.decide(view)
    assert is_valid_game_action(decision.game_action), decision.game_action
