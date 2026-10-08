"""Constants and small helpers for the External Bot Protocol v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlsplit

PROTOCOL_VERSION = 1

# Index order equals the bot engine's ``Color`` enum.
COLOR_NAMES = ("yellow", "red", "blue", "green", "white")
COLOR_INDEX = {name: index for index, name in enumerate(COLOR_NAMES)}

API_MARKER = "/api/ext/"
REDACTED = "<redacted>"

# A game view / state / action is plain decoded JSON.
JsonObject = dict[str, Any]


class ConnectStringError(ValueError):
    """The connect string is not ``<origin>/api/ext/<gameId>#<token>``."""


class ShadowStateError(ValueError):
    """The game view cannot be turned into a consistent engine state."""


class TranslationError(ValueError):
    """An engine action has no counterpart in the game's action format."""


class NoLegalActionError(RuntimeError):
    """Every legal action was excluded (or none exists) for this position."""


@dataclass(frozen=True)
class ConnectString:
    """Parsed connect string. The token never appears in ``repr``/``str``."""

    origin: str
    game_id: str
    token: str

    @property
    def base_url(self) -> str:
        return f"{self.origin}{API_MARKER}{self.game_id}"

    def __repr__(self) -> str:
        return f"ConnectString(base_url={self.base_url!r}, token={REDACTED})"

    __str__ = __repr__


def parse_connect_string(text: str) -> ConnectString:
    """Split ``<origin>/api/ext/<gameId>#<token>`` into its parts."""
    text = text.strip()
    base, separator, token = text.partition("#")
    if not separator or not token:
        raise ConnectStringError(
            "connect string must end in '#<token>' (quote it in the shell: "
            "'#' starts a comment in bash)"
        )
    if any(character.isspace() for character in token):
        raise ConnectStringError("connect string token must not contain whitespace")
    prefix, marker, game_id = base.rpartition(API_MARKER)
    game_id = game_id.strip("/")
    if not marker or not game_id or "/" in game_id:
        raise ConnectStringError(
            "connect string must look like <origin>/api/ext/<gameId>#<token>"
        )
    parts = urlsplit(prefix + "/")
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ConnectStringError("connect string must start with http:// or https://")
    if parts.query or parts.fragment:
        raise ConnectStringError("connect string origin must not contain a query")
    return ConnectString(origin=prefix.rstrip("/"), game_id=game_id, token=token)


def redact(text: str, token: Optional[str]) -> str:
    """Remove the secret seat token from any text that might be printed."""
    if token and token in text:
        return text.replace(token, REDACTED)
    return text
