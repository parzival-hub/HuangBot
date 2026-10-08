"""The real bundled model plays a whole game against an in-process Zhanguo stand-in."""

import threading
import time

from export_view import GOLDEN_ROWS
from mock_server import MockZhanguo

DEFAULT_ROWS = ["." * 15] * 15


def wait_for(condition, timeout=20.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_full_game_with_the_bundled_model(real_model, make_runner, caplog):
    with MockZhanguo(players=2, short_game=True, bot_seat=0, seed=11) as mock:
        runner = make_runner(mock, real_model)
        assert runner.use_score_history is True
        started = time.monotonic()
        assert runner.run() == 0
        assert time.monotonic() - started < 120

    assert mock.status == "finished"
    assert mock.codes[("action", 400)] == 0
    assert mock.codes[("action", 409)] == 0
    assert runner.rejections == 0 and runner.fallbacks == 0
    assert runner.accepted > 30
    assert len(mock.action_posts) == runner.accepted
    first = mock.status_posts[0]
    assert first["ready"] is True
    assert first["agent"].startswith("huangbot-remote ")
    assert mock.seen_auth == {"Bearer secret-token"}
    assert "secret-token" not in caplog.text
    assert "fallbacks used: 0" in caplog.text


def test_three_player_game_against_random_opponents(tiny_model, make_runner):
    with MockZhanguo(players=3, short_game=False, bot_seat=1, seed=5) as mock:
        runner = make_runner(mock, tiny_model)
        assert runner.run() == 0
    assert mock.status == "finished"
    assert mock.codes[("action", 400)] == 0
    assert runner.fallbacks == 0


def test_wrong_board_reports_not_ready_and_never_acts(real_model, make_runner):
    with MockZhanguo(rows=DEFAULT_ROWS, start_when_ready=False) as mock:
        runner = make_runner(mock, real_model, interval=0.01)
        thread = threading.Thread(target=runner.run)
        thread.start()
        wait_for(lambda: mock.status_posts)
        mock.start()  # the host starts the game anyway
        time.sleep(0.3)
        runner.stop()
        thread.join(5)
        assert not thread.is_alive()
    assert mock.status_posts[0]["ready"] is False
    assert "unsupported map" in mock.status_posts[0]["note"]
    assert len(mock.status_posts) == 1  # reported once, not on every poll
    assert mock.action_posts == []
    assert mock.status == "playing"
    assert mock.codes[("state", 304)] >= 1  # unchanged state: the ETag saves the transfer


def test_player_count_outside_two_to_four_is_not_ready(tiny_model, make_runner):
    with MockZhanguo(players=2, start_when_ready=False) as mock:
        mock.players = 5
        runner = make_runner(mock, tiny_model, interval=0.01)
        thread = threading.Thread(target=runner.run)
        thread.start()
        wait_for(lambda: mock.status_posts)
        runner.stop()
        thread.join(5)
    assert mock.status_posts[0]["ready"] is False
    assert "player count" in mock.status_posts[0]["note"]


def test_status_is_sent_again_when_the_lobby_changes(tiny_model, make_runner):
    with MockZhanguo(rows=DEFAULT_ROWS, start_when_ready=False) as mock:
        runner = make_runner(mock, tiny_model, interval=0.01)
        thread = threading.Thread(target=runner.run)
        thread.start()
        wait_for(lambda: len(mock.status_posts) == 1)
        with mock.lock:
            mock.rows = list(GOLDEN_ROWS)
            mock.lobby_revision += 1
        wait_for(lambda: len(mock.status_posts) == 2)
        runner.stop()
        thread.join(5)
    assert [p["ready"] for p in mock.status_posts] == [False, True]
