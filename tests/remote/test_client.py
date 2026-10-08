import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from huangbot.remote import __version__
from huangbot.remote.client import (
    Backoff,
    ZhanguoApiError,
    ZhanguoClient,
    ZhanguoFatalError,
    ZhanguoNetworkError,
)
from mock_server import MockZhanguo


def test_etag_round_trip_and_headers():
    with MockZhanguo(start_when_ready=False) as mock:
        client = ZhanguoClient(mock.origin, mock.game_id, mock.token)
        first = client.get_state()
        assert first.state["status"] == "lobby" and first.etag and not first.not_modified
        second = client.get_state(first.etag)
        assert second.not_modified and second.state is None and second.etag == first.etag
        client.post_status("agent x", True, "note")
        assert client.get_state(first.etag).state is not None  # lobby change changes the ETag
        assert mock.status_posts[-1] == {"agent": "agent x", "ready": True, "note": "note"}
    assert mock.seen_auth == {f"Bearer {mock.token}"}


def test_error_bodies_become_api_errors():
    with MockZhanguo(seed=1) as mock:
        client = ZhanguoClient(mock.origin, mock.game_id, mock.token)
        client.post_status("a", True)
        with pytest.raises(ZhanguoApiError) as stale:
            client.post_action({"type": "replace", "tiles": []}, version=99)
        assert (stale.value.status, stale.value.message, stale.value.version) == (409, "stale", mock.version)
        with pytest.raises(ZhanguoApiError) as rejected:
            client.post_action({"type": "leader", "color": "red", "to": 0}, version=mock.version)
        assert rejected.value.status == 400 and "Ungültiger Zug" in rejected.value.message


def test_401_and_404_are_fatal_and_never_leak_the_token():
    with MockZhanguo() as mock:
        for client, status in (
            (ZhanguoClient(mock.origin, mock.game_id, "wrong-secret"), 401),
            (ZhanguoClient(mock.origin, "other", mock.token), 404),
        ):
            with pytest.raises(ZhanguoFatalError) as error:
                client.get_state()
            assert error.value.status == status
            assert "wrong-secret" not in str(error.value)


def test_connection_refused_is_a_network_error():
    client = ZhanguoClient("http://127.0.0.1:1", "g", "tok", timeout=1)
    with pytest.raises(ZhanguoNetworkError):
        client.get_state()


def test_redirects_are_not_followed_because_they_would_forward_the_token():
    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(301)
            self.send_header("Location", "https://elsewhere.example/api/ext/g/state")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Redirect)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = ZhanguoClient(f"http://127.0.0.1:{server.server_address[1]}", "g", "tok")
        with pytest.raises(ZhanguoFatalError, match="elsewhere.example"):
            client.get_state()
    finally:
        server.shutdown()
        server.server_close()


def test_invalid_json_is_a_network_error():
    class Garbage(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "8")
            self.end_headers()
            self.wfile.write(b"not json")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Garbage)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = ZhanguoClient(f"http://127.0.0.1:{server.server_address[1]}", "g", "tok")
        with pytest.raises(ZhanguoNetworkError):
            client.get_state()
    finally:
        server.shutdown()
        server.server_close()


def test_user_agent_is_sent():
    seen = []

    class Echo(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers["User-Agent"])
            body = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Echo)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        ZhanguoClient(f"http://127.0.0.1:{server.server_address[1]}", "g", "tok").get_state()
    finally:
        server.shutdown()
        server.server_close()
    assert seen == [f"huangbot-remote/{__version__}"]


def test_backoff_doubles_up_to_the_maximum_and_resets():
    backoff = Backoff()
    assert [backoff.next_delay() for _ in range(8)] == [1, 2, 4, 8, 16, 30, 30, 30]
    backoff.reset()
    assert backoff.next_delay() == 1
