"""Standard-library HTTP client for the External Bot Protocol v1."""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from . import __version__
from .protocol import API_MARKER, JsonObject, redact

DEFAULT_TIMEOUT = 10.0


class ZhanguoApiError(Exception):
    """The server answered with an error status."""

    def __init__(self, status: int, message: str, version: Optional[int] = None):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message
        self.version = version

    @property
    def transient(self) -> bool:
        return self.status >= 500


class ZhanguoFatalError(ZhanguoApiError):
    """Wrong token, unknown game or a redirect: retrying cannot help."""


class ZhanguoNetworkError(Exception):
    """No usable answer (connection, timeout, invalid response body)."""


@dataclass(frozen=True)
class StateResponse:
    state: Optional[JsonObject]
    etag: Optional[str]
    not_modified: bool = False


class Backoff:
    """1, 2, 4, ... seconds up to ``maximum``; ``reset`` after a success."""

    def __init__(self, first: float = 1.0, maximum: float = 30.0):
        self.first = first
        self.maximum = maximum
        self._next = first

    def next_delay(self) -> float:
        delay = self._next
        self._next = min(self._next * 2, self.maximum)
        return delay

    def reset(self) -> None:
        self._next = self.first


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow redirects: urllib would forward the bearer token."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ZhanguoClient:
    def __init__(
        self,
        origin: str,
        game_id: str,
        token: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        user_agent: Optional[str] = None,
    ):
        self.base_url = f"{origin.rstrip('/')}{API_MARKER}{game_id}"
        self.game_id = game_id
        self.timeout = timeout
        self._token = token
        self._user_agent = user_agent or f"huangbot-remote/{__version__}"
        self._opener = urllib.request.build_opener(_NoRedirect)

    def __repr__(self) -> str:
        return f"ZhanguoClient({self.base_url!r}, token=<redacted>)"

    def get_state(self, if_none_match: Optional[str] = None) -> StateResponse:
        headers = {"If-None-Match": if_none_match} if if_none_match else {}
        status, body, response_headers = self._request("GET", "/state", headers=headers)
        if status == 304:
            return StateResponse(None, if_none_match, not_modified=True)
        if not isinstance(body, dict):
            raise ZhanguoNetworkError("the state answer is not a JSON object")
        return StateResponse(body, response_headers.get("ETag"))

    def post_action(self, action: JsonObject, version: int) -> JsonObject:
        _, body, _ = self._request("POST", "/action", {"action": action, "version": version})
        return body if isinstance(body, dict) else {}

    def post_status(self, agent: str, ready: bool, note: Optional[str] = None) -> None:
        payload: JsonObject = {"agent": agent, "ready": ready}
        if note:
            payload["note"] = note
        self._request("POST", "/status", payload)

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[JsonObject] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> tuple[int, Any, Any]:
        all_headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
            "User-Agent": self._user_agent,
            **(headers or {}),
        }
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            all_headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path, data=data, headers=all_headers, method=method
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read()
                return response.status, self._decode(raw), response.headers
        except urllib.error.HTTPError as error:
            if error.code == 304:
                return 304, None, error.headers
            raise self._api_error(error) from None
        except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
            raise ZhanguoNetworkError(self.redact(str(getattr(error, "reason", error)))) from None

    def _decode(self, raw: bytes) -> Any:
        if not raw.strip():
            return None
        try:
            return json.loads(raw)
        except ValueError:
            raise ZhanguoNetworkError("the server answer is not valid JSON") from None

    def _api_error(self, error: urllib.error.HTTPError) -> ZhanguoApiError:
        message, version = error.reason or "error", None
        try:
            body = json.loads(error.read())
            if isinstance(body, dict):
                message = str(body.get("error") or message)
                version = body.get("version")
        except (ValueError, OSError):
            pass
        message = self.redact(str(message))
        if error.code == 401:
            return ZhanguoFatalError(401, "token invalid or seat removed")
        if error.code == 404:
            return ZhanguoFatalError(404, "game not found")
        if 300 <= error.code < 400:
            target = self.redact(error.headers.get("Location", "?"))
            return ZhanguoFatalError(
                error.code, f"unexpected redirect to {target}; use the final https URL"
            )
        return ZhanguoApiError(error.code, message, version if isinstance(version, int) else None)

    def redact(self, text: str) -> str:
        return redact(text, self._token)
