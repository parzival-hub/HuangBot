from huang import board
from huang.action_codec import NUM_DISTINCT_ACTIONS

from export_view import GOLDEN_ROWS
from huangbot.remote.board_check import check_lobby, expected_rows, is_supported_map

DEFAULT_ZHANGUO_ROWS = ["." * 15] * 15


def test_expected_rows_equal_the_golden_rows():
    assert expected_rows() == GOLDEN_ROWS


def test_golden_numbers():
    assert board.NUM_CELLS == 203
    assert len(board.RIVER_CELLS) == 44
    assert list(board.CAPITAL_CELLS) == [15, 60, 68, 93, 113, 132, 164]
    assert len(board.TRIANGLES) == 353
    rivers = set(board.RIVER_CELLS)
    assert sum(set(t) <= rivers for t in board.TRIANGLES) == 8
    assert sum(not (set(t) & rivers) for t in board.TRIANGLES) == 234
    assert NUM_DISTINCT_ACTIONS == 6599


def test_cell_index_is_row_major_over_playable_spaces():
    index = 0
    for row_number, row in enumerate(GOLDEN_ROWS):
        for column, symbol in enumerate(row):
            if symbol != "#":
                assert board.COORD_TO_INDEX[(row_number, column)] == index
                index += 1
    assert index == 203


def test_supported_map_ignores_names():
    assert is_supported_map({"name": "Bot-Brett", "rows": GOLDEN_ROWS, "names": {"2,1": "x"}}) == (True, "")
    assert is_supported_map({"rows": list(GOLDEN_ROWS)})[0]


def test_other_maps_are_rejected():
    ok, reason = is_supported_map({"name": "Standard", "rows": DEFAULT_ZHANGUO_ROWS})
    assert not ok and "unsupported map" in reason
    changed = list(GOLDEN_ROWS)
    changed[3] = changed[3].replace("~", ".", 1)
    assert not is_supported_map({"rows": changed})[0]
    assert not is_supported_map({"rows": GOLDEN_ROWS[:-1]})[0]
    assert not is_supported_map({})[0]
    assert not is_supported_map(None)[0]


def test_lobby_checks_player_count_and_seat():
    seats = lambda n: [{"name": str(i)} for i in range(n)]
    lobby = lambda n, rows=GOLDEN_ROWS: {"map": {"rows": rows}, "seats": seats(n)}
    assert check_lobby(lobby(2), 1) == (True, "")
    assert check_lobby(lobby(4), 3) == (True, "")
    assert not check_lobby(lobby(5), 0)[0]
    assert not check_lobby(lobby(1), 0)[0]
    assert not check_lobby(lobby(3), 3)[0]
    assert not check_lobby(lobby(2, DEFAULT_ZHANGUO_ROWS), 0)[0]
