import subprocess
import sys

from huangbot.remote.cli import checkpoint_label, main
from mock_server import MockZhanguo


def connect_string(mock):
    return f"{mock.origin}/api/ext/{mock.game_id}#{mock.token}"


def test_cli_plays_a_whole_game_and_reports_the_documented_agent_string(tmp_path, capsys):
    with MockZhanguo(seed=8) as mock:
        code = main([connect_string(mock), "--interval", "0.01", "--min-think", "0", "--debug-dir", str(tmp_path)])
    assert code == 0
    assert mock.status == "finished"
    assert mock.status_posts[0]["agent"] == "huangbot-remote 0.1 (high_elo_continuous_20260928)"
    assert mock.codes[("action", 400)] == 0
    assert list(tmp_path.iterdir()) == []  # nothing went wrong, nothing dumped
    assert mock.token not in capsys.readouterr().err


def test_cli_exits_with_2_for_a_missing_checkpoint(tmp_path, capsys):
    code = main(["http://127.0.0.1:1/api/ext/g#tok", "--checkpoint", str(tmp_path / "nope.pt")])
    assert code == 2
    assert "checkpoint" in capsys.readouterr().err


def test_cli_exits_with_2_for_a_wrong_token():
    with MockZhanguo() as mock:
        assert main([f"{mock.origin}/api/ext/{mock.game_id}#wrong", "--interval", "0.01"]) == 2


def test_cli_rejects_nonsense_options(capsys):
    for argv in (["x#y", "--threads", "0"], ["x#y", "--interval", "0"], ["x#y", "--log-level", "loud"]):
        try:
            main(argv)
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError(argv)


def test_checkpoint_label_prefers_the_provenance_of_the_bundled_model(tmp_path):
    assert checkpoint_label(None) == "high_elo_continuous_20260928"
    assert checkpoint_label(tmp_path / "custom.pt") == "custom"


def test_python_dash_m_entry_point_works():
    result = subprocess.run([sys.executable, "-m", "huangbot.remote", "--version"], capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout.startswith("huangbot-remote ")
