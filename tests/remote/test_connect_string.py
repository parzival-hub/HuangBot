import logging

import pytest

from huangbot.remote.cli import _RedactFilter, main
from huangbot.remote.client import ZhanguoClient
from huangbot.remote.protocol import ConnectStringError, parse_connect_string, redact


def test_parses_origin_game_id_and_token():
    parsed = parse_connect_string("https://h.example/api/ext/ab12cd#tok")
    assert parsed.origin == "https://h.example"
    assert parsed.game_id == "ab12cd"
    assert parsed.token == "tok"
    assert parsed.base_url == "https://h.example/api/ext/ab12cd"


def test_keeps_a_path_prefix_and_ignores_surrounding_whitespace():
    parsed = parse_connect_string("  http://localhost:8080/zh/api/ext/g1/#a-b_c\n")
    assert parsed.origin == "http://localhost:8080/zh"
    assert parsed.game_id == "g1"
    assert parsed.token == "a-b_c"


@pytest.mark.parametrize(
    "text",
    [
        "https://h.example/api/ext/ab12cd",
        "https://h.example/api/ext/ab12cd#",
        "https://h.example/games/ab12cd#tok",
        "https://h.example/api/ext/#tok",
        "https://h.example/api/ext/a/b#tok",
        "ftp://h.example/api/ext/ab#tok",
        "h.example/api/ext/ab#tok",
        "https://h.example/api/ext/ab#to k",
        "",
    ],
)
def test_rejects_malformed_strings(text):
    with pytest.raises(ConnectStringError):
        parse_connect_string(text)


def test_the_token_is_redacted_everywhere_it_could_be_printed(caplog):
    secret = "SuperSecretSeatToken"
    parsed = parse_connect_string(f"https://h.example/api/ext/ab#{secret}")
    client = ZhanguoClient(parsed.origin, parsed.game_id, parsed.token)
    for text in (repr(parsed), str(parsed), f"{parsed}", repr(client), str(client)):
        assert secret not in text
    assert redact(f"GET https://x/?t={secret}", secret) == "GET https://x/?t=<redacted>"

    logger = logging.getLogger("redaction-test")
    handler = logging.Handler()
    handler.emit = lambda record: seen.append(record.getMessage())
    handler.addFilter(_RedactFilter(secret))
    seen = []
    logger.addHandler(handler)
    logger.warning("failed with %s", secret)
    assert seen == ["failed with <redacted>"]


def test_cli_exits_with_code_2_for_a_bad_connect_string(capsys):
    assert main(["https://h.example/api/ext/ab12cd"]) == 2
    assert "#<token>" in capsys.readouterr().err
