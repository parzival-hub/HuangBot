import numpy as np

from export_view import engine_to_view, public_history_scores
from huangbot.environment import public_score_history_from_engine
from huangbot.remote.score_history import public_score_history
from test_hidden_info import mid_game_state


def test_sums_vp_entries_player_major_and_divides_by_30_once():
    view = {
        "log": [
            {"turn": 1, "p": 0, "t": "note"},
            {"turn": 1, "p": 0, "t": "tile", "vp": [{"p": 0, "color": "red", "n": 3}]},
            {"turn": 2, "p": 1, "t": "pagodaVp", "vp": [{"p": 1, "color": "white", "n": 6}, {"p": 0, "color": "red", "n": 3}]},
            {"turn": 3, "p": 1, "t": "warEnd", "vp": [{"p": 3, "color": "yellow", "n": 30}]},
        ]
    }
    history = public_score_history(view, 4)
    assert history.shape == (20,) and history.dtype == np.float32
    assert history[1] == np.float32(6 / 30)  # player 0, red
    assert history[5 + 4] == np.float32(6 / 30)  # player 1, white
    assert history[15] == np.float32(1.0)  # player 3, yellow
    assert np.count_nonzero(history) == 3


def test_players_beyond_the_game_are_ignored():
    view = {"log": [{"vp": [{"p": 3, "color": "red", "n": 4}]}]}
    assert not public_score_history(view, 2).any()
    assert public_score_history(view).any()


def test_matches_the_environment_arithmetic_on_real_games():
    for players, seed in ((2, 1), (3, 2), (4, 3)):
        state = mid_game_state(players, 150, seed)
        view = engine_to_view(state, state.current_actor())
        mine = public_score_history(view, players)
        assert np.array_equal(mine, public_history_scores(state))
        # the helper next to HuangEnvironment divides by 30 exactly once as well
        assert np.array_equal(mine, public_score_history_from_engine(state))
    assert mine.any()
