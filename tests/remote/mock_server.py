"""In-process Zhanguo stand-in implementing External Bot Protocol v1 (PART B).

The game behind it is a real ``HuangState``; seat ``bot_seat`` is the external
bot and every other seat plays random legal moves on the server side.
"""

from __future__ import annotations

import json
import random
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from export_view import GOLDEN_ROWS, engine_to_view, game_action_to_engine, resolve_chance
from huang.engine import HuangState


class MockZhanguo:
    def __init__(
        self,
        *,
        players=2,
        short_game=True,
        bot_seat=0,
        rows=None,
        seed=0,
        start_when_ready=True,
        token="secret-token",
        game_id="g1",
    ):
        self.players, self.short_game, self.bot_seat = players, short_game, bot_seat
        self.rows = list(GOLDEN_ROWS if rows is None else rows)
        self.rng = random.Random(seed)
        self.start_when_ready = start_when_ready
        self.token, self.game_id = token, game_id
        self.lock = threading.RLock()
        self.status = "lobby"
        self.state = None
        self.version = 0
        self.lobby_revision = 0
        self.reported = None
        self.status_posts = []
        self.action_posts = []  # (body, http status)
        self.codes = Counter()  # (endpoint, status code) -> count
        self.get_script = []  # "drop" | "500"
        self.action_script = []  # "reject" | "stale" | "drop" | "500"
        self.reject_all = False
        self.requests = 0
        self.seen_auth = set()
        self._server = None

    # -- lifecycle -----------------------------------------------------

    def __enter__(self):
        handler = type("Handler", (_Handler,), {"mock": self})
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()

    @property
    def origin(self):
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    # -- game ----------------------------------------------------------

    def start(self):
        with self.lock:
            self.state = HuangState(self.players, short_game=self.short_game, starting_player=-1)
            resolve_chance(self.state, self.rng)
            self.status = "playing"
            self.lobby_revision += 1
            self._advance_others()

    def _advance_others(self):
        state = self.state
        while not state.game_over and state.current_actor() != self.bot_seat:
            state.apply_action(self.rng.choice(state.legal_actions()))
            resolve_chance(state, self.rng)
            self.version += 1
        if state.game_over:
            self.status = "finished"

    def bot_view(self):
        return engine_to_view(self.state, self.bot_seat)

    def ext_state(self):
        playing = self.status == "playing"
        view = None if self.state is None else self.bot_view()
        if view is not None and self.status == "finished":
            view["gameOver"] = [{"player": p, "points": self.state.points[p]} for p in range(self.players)]
        return {
            "protocol": 1,
            "gameId": self.game_id,
            "seat": self.bot_seat,
            "status": self.status,
            "version": self.version,
            "yourTurn": playing and self.state.current_actor() == self.bot_seat,
            "lobby": {
                "id": self.game_id,
                "name": "mock",
                "status": self.status,
                "seats": [
                    {
                        "name": f"P{i}",
                        "dynasty": "Han",
                        "online": True,
                        "bot": False,
                        "external": i == self.bot_seat,
                        **({"ext": self.reported} if i == self.bot_seat and self.reported else {}),
                    }
                    for i in range(self.players)
                ],
                "options": {"shortGame": self.short_game},
                "map": {"name": "Bot-Brett", "rows": self.rows},
                "hasWebhook": False,
                "createdAt": 0,
                "actionCount": self.version,
            },
            "view": view,
        }

    def etag(self):
        return f'"{self.version}-{self.lobby_revision}-{self.status}"'

    def handle_action(self, body):
        """Returns ``(http status, json payload)``."""
        with self.lock:
            if self.status != "playing":
                return 400, {"error": "Das Spiel läuft nicht."}
            if body.get("version") != self.version:
                return 409, {"error": "stale", "version": self.version}
            if self.state.current_actor() != self.bot_seat:
                return 400, {"error": "Du bist nicht am Zug."}
            action = body["action"]
            try:
                if action == {"type": "replace", "tiles": []}:  # the game's pass
                    self.state.actions_remaining -= 1
                    self.state._finish_action()
                else:
                    for engine_action in game_action_to_engine(self.state, action, self.bot_view()):
                        self.state.apply_action(engine_action)
            except (AssertionError, KeyError, ValueError, IndexError, TypeError) as error:
                return 400, {"error": f"Ungültiger Zug ({type(error).__name__})"}
            resolve_chance(self.state, self.rng)
            self.version += 1
            self._advance_others()
            return 200, {"ok": True, "version": self.version}

    def handle_status(self, body):
        with self.lock:
            self.reported = {k: body[k] for k in ("agent", "ready", "note") if k in body}
            self.status_posts.append(dict(body))
            self.lobby_revision += 1
            if self.start_when_ready and body.get("ready") and self.status == "lobby":
                self.start()
            return 200, {"ok": True}

    @property
    def rejected_actions(self):
        return sum(1 for _, code in self.action_posts if code == 400)


class _Handler(BaseHTTPRequestHandler):
    mock: MockZhanguo
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def _send(self, code, payload=None, headers=None):
        body = b"" if payload is None else json.dumps(payload).encode("utf-8")
        self.send_response(code)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        if body:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _route(self):
        mock = self.mock
        mock.requests += 1
        prefix = f"/api/ext/{mock.game_id}/"
        auth = self.headers.get("Authorization", "")
        mock.seen_auth.add(auth)
        if not self.path.startswith("/api/ext/"):
            self._send(404, {"error": "Nicht gefunden."})
            return None
        if auth != f"Bearer {mock.token}":
            self._send(401, {"error": "Nicht autorisiert."})
            return None
        if not self.path.startswith(prefix):
            self._send(404, {"error": "Spiel nicht gefunden."})
            return None
        return self.path[len(prefix):]

    def _scripted(self, script, endpoint):
        mock = self.mock
        with mock.lock:
            step = script.pop(0) if script else None
        if step == "drop":
            self.close_connection = True
            mock.codes[(endpoint, "drop")] += 1
            return True
        if step == "500":
            mock.codes[(endpoint, 500)] += 1
            self._send(500, {"error": "kaputt"})
            return True
        return False

    def do_GET(self):
        endpoint = self._route()
        if endpoint != "state":
            if endpoint is not None:
                self._send(404, {"error": "?"})
            return
        mock = self.mock
        if self._scripted(mock.get_script, "state"):
            return
        with mock.lock:
            etag = mock.etag()
            if self.headers.get("If-None-Match") == etag:
                mock.codes[("state", 304)] += 1
                self.send_response(304)
                self.send_header("ETag", etag)
                self.end_headers()
                return
            payload = mock.ext_state()
        mock.codes[("state", 200)] += 1
        self._send(200, payload, {"ETag": etag})

    def do_POST(self):
        endpoint = self._route()
        if endpoint is None:
            return
        mock = self.mock
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if endpoint == "status":
            code, payload = mock.handle_status(body)
            mock.codes[("status", code)] += 1
            self._send(code, payload)
        elif endpoint == "action":
            if self._scripted_action(body):
                return
            code, payload = mock.handle_action(body)
            mock.action_posts.append((body, code))
            mock.codes[("action", code)] += 1
            self._send(code, payload)
        else:
            self._send(404, {"error": "?"})

    def _scripted_action(self, body):
        mock = self.mock
        if mock.reject_all:
            mock.action_posts.append((body, 400))
            mock.codes[("action", 400)] += 1
            self._send(400, {"error": "Ungültiger Zug (abgelehnt)"})
            return True
        with mock.lock:
            step = mock.action_script.pop(0) if mock.action_script else None
        if step == "reject":
            mock.action_posts.append((body, 400))
            mock.codes[("action", 400)] += 1
            self._send(400, {"error": "Ungültiger Zug (abgelehnt)"})
        elif step == "stale":
            mock.action_posts.append((body, 409))
            mock.codes[("action", 409)] += 1
            self._send(409, {"error": "stale", "version": mock.version})
        elif step == "drop":
            self.close_connection = True
            mock.codes[("action", "drop")] += 1
        elif step == "500":
            mock.codes[("action", 500)] += 1
            self._send(500, {"error": "kaputt"})
        else:
            return False
        return True
