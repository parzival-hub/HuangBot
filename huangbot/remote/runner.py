"""The poll / decide / act loop of the remote player."""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from huang.action_codec import NUM_DISTINCT_ACTIONS
from huang.engine import HuangState
from huangbot.environment import PUBLIC_SCORE_HISTORY_SIZE

from . import __version__
from .adapter import ZhanguoAdapter
from .board_check import check_lobby, is_supported_map
from .client import (
    Backoff,
    ZhanguoApiError,
    ZhanguoClient,
    ZhanguoFatalError,
    ZhanguoNetworkError,
)
from .fallback import safe_action
from .protocol import PROTOCOL_VERSION

LOGGER = logging.getLogger("huangbot.remote")

EXIT_FINISHED = 0
EXIT_ERRORS = 1
EXIT_CONFIG = 2
EXIT_INTERRUPTED = 130

_SETTLED_LOG_SECONDS = 30.0


class ConfigError(Exception):
    """Model, protocol or game setup that this client cannot work with."""


@dataclass
class RunnerConfig:
    interval: float = 1.0
    min_think: float = 0.9
    max_attempts: int = 3
    max_stale: int = 20
    max_errors: int = 0  # consecutive transient errors before giving up; 0 = never
    debug_dir: Optional[Path] = None
    deterministic: bool = True
    strict: bool = False
    agent_name: str = f"huangbot-remote {__version__}"


class DebugDumper:
    """Write ``{timestamp}-{version}-{reason}.json`` files for the two repo owners."""

    def __init__(self, directory: Optional[Path], redact: Callable[[str], str]):
        self.directory = directory
        self._redact = redact

    def dump(
        self,
        reason: str,
        view: Optional[Mapping[str, Any]],
        version: Any,
        error: Optional[BaseException] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self.directory is None:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            payload = {
                "view": view,
                "error": None if error is None else f"{type(error).__name__}: {error}",
                "traceback": None
                if error is None
                else "".join(traceback.format_exception(error)),
                **(extra or {}),
            }
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = self.directory / f"{stamp}-{version}-{reason}.json"
            counter = 1
            while path.exists():
                path = self.directory / f"{stamp}-{version}-{reason}-{counter}.json"
                counter += 1
            path.write_text(self._redact(json.dumps(payload, indent=1)), encoding="utf-8")
        except OSError as problem:
            LOGGER.warning("could not write debug dump: %s", problem)


class RemoteRunner:
    def __init__(
        self,
        client: ZhanguoClient,
        model,
        config: Optional[RunnerConfig] = None,
        *,
        agent_factory: Optional[Callable[[int], Any]] = None,
        stop_event: Optional[threading.Event] = None,
        sleep: Optional[Callable[[float], None]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.client = client
        self.config = config or RunnerConfig()
        self.use_score_history = self._check_model(model)
        self._agent_factory = agent_factory or self._default_agent_factory(model)
        self._stop = stop_event or threading.Event()
        self._sleep = sleep or (lambda seconds: self._stop.wait(seconds))
        self._clock = clock
        self._dumper = DebugDumper(self.config.debug_dir, client.redact)

        self.fallbacks = 0
        self.rejections = 0
        self.accepted = 0
        self._etag: Optional[str] = None
        self._reported: Optional[tuple[bool, str]] = None
        self._agent: Any = None
        self._adapter: Optional[ZhanguoAdapter] = None
        self._players = 0
        self._game_id: Optional[str] = None
        self._last_version = -1
        self._turn_key: Optional[tuple[str, int]] = None
        self._turn_seen = 0.0
        self._exhausted_key: Optional[tuple[str, int]] = None
        self._exhausted_logged = 0.0
        self._incompatible_logged: Optional[str] = None

    # -- setup ---------------------------------------------------------

    @staticmethod
    def _check_model(model) -> bool:
        config = model.config
        if config.num_actions != NUM_DISTINCT_ACTIONS:
            raise ConfigError(
                f"checkpoint has {config.num_actions} actions, the engine has {NUM_DISTINCT_ACTIONS}"
            )
        base = HuangState.observation_tensor_size()
        if config.observation_size == base:
            return False
        if config.observation_size == base + PUBLIC_SCORE_HISTORY_SIZE:
            return True
        raise ConfigError(
            f"checkpoint expects {config.observation_size} observation features; "
            f"supported are {base} and {base + PUBLIC_SCORE_HISTORY_SIZE}"
        )

    def _default_agent_factory(self, model) -> Callable[[int], Any]:
        from huangbot.model import RecurrentModelAgent

        return lambda players: RecurrentModelAgent(
            model, players=players, deterministic=self.config.deterministic
        )

    def stop(self) -> None:
        self._stop.set()

    # -- main loop -----------------------------------------------------

    def run(self) -> int:
        LOGGER.info("connecting to %s", self.client.base_url)
        backoff = Backoff()
        failures = 0
        try:
            while not self._stop.is_set():
                started = self._clock()
                try:
                    finished = self._tick()
                except (ConfigError, ZhanguoFatalError) as error:
                    LOGGER.error("fatal: %s", error)
                    return EXIT_CONFIG
                except (ZhanguoNetworkError, ZhanguoApiError) as error:
                    failures += 1
                    self._etag = None
                    LOGGER.warning("server problem (%s), retrying", error)
                    if self._too_many(failures):
                        return EXIT_ERRORS
                    self._sleep(backoff.next_delay())
                    continue
                except Exception as error:  # a bug must not abandon a running game
                    failures += 1
                    self._etag = None
                    LOGGER.exception("unexpected error, retrying")
                    self._dumper.dump("unexpected", None, self._last_version, error)
                    if self._too_many(failures):
                        return EXIT_ERRORS
                    self._sleep(backoff.next_delay())
                    continue
                backoff.reset()
                failures = 0
                if finished:
                    return EXIT_FINISHED
                self._sleep(max(0.0, self.config.interval - (self._clock() - started)))
        except KeyboardInterrupt:
            LOGGER.info("interrupted")
            return EXIT_INTERRUPTED
        return EXIT_FINISHED

    def _too_many(self, failures: int) -> bool:
        limit = self.config.max_errors
        if limit and failures >= limit:
            LOGGER.error("giving up after %d consecutive errors", failures)
            return True
        return False

    def _tick(self) -> bool:
        response = self.client.get_state(self._etag)
        if response.not_modified:
            return False
        state = response.state
        assert state is not None
        self._etag = None
        settled, finished = self._handle_state(state)
        if settled:
            self._etag = response.etag
        return finished

    # -- one state -----------------------------------------------------

    def _handle_state(self, state: Mapping[str, Any]) -> tuple[bool, bool]:
        """Returns ``(settled, finished)``; ``settled`` means the ETag may be reused."""
        if state.get("protocol") != PROTOCOL_VERSION:
            raise ConfigError(f"unsupported protocol {state.get('protocol')!r}")
        status = state["status"]
        ready, note = self._compatibility(state)
        self._report_status(ready, note)
        if status == "lobby":
            return True, False
        if status == "finished":
            self._log_finished(state)
            return True, True
        if not ready:
            if self._incompatible_logged != note:
                LOGGER.error("this game cannot be played by the bot: %s", note)
                self._incompatible_logged = note
            return True, False
        self._ensure_agent(state)
        if not state["yourTurn"]:
            self._turn_key = None
            return True, False
        key = (state["gameId"], state["version"])
        if key == self._exhausted_key:
            now = self._clock()
            if now - self._exhausted_logged >= _SETTLED_LOG_SECONDS:
                self._exhausted_logged = now
                LOGGER.error(
                    "still waiting at version %d: every action was rejected, a human has to act",
                    state["version"],
                )
            return True, False
        if key != self._turn_key:
            self._turn_key = key
            self._turn_seen = self._clock()
        self._act(state)
        return False, False

    def _compatibility(self, state: Mapping[str, Any]) -> tuple[bool, str]:
        ready, note = check_lobby(state["lobby"], int(state["seat"]))
        view = state.get("view")
        if ready and view is not None:
            ready, note = is_supported_map(view.get("map"))
        return ready, note

    def _report_status(self, ready: bool, note: str) -> None:
        if self._reported == (ready, note):
            return
        try:
            self.client.post_status(self.config.agent_name, ready, note or None)
        except ZhanguoFatalError:
            raise
        except (ZhanguoNetworkError, ZhanguoApiError) as error:
            LOGGER.warning("could not report status: %s", error)
            return
        self._reported = (ready, note)
        LOGGER.info("reported status ready=%s%s", ready, f" ({note})" if note else "")

    def _ensure_agent(self, state: Mapping[str, Any]) -> None:
        view = state["view"]
        players = len(view["players"])
        seat = int(state["seat"])
        version = int(state["version"])
        if self._adapter is None or players != self._players or state["gameId"] != self._game_id:
            self._agent = self._agent_factory(players)
            self._adapter = ZhanguoAdapter(
                self._agent,
                seat=seat,
                use_score_history=self.use_score_history,
                strict=self.config.strict,
            )
            self._players = players
            LOGGER.info("playing seat %d of %d players", seat, players)
        elif version < self._last_version:
            LOGGER.info("version went backwards (%d -> %d): new game, resetting memory", self._last_version, version)
            self._adapter.reset_episode()
            self._exhausted_key = None
        self._game_id = state["gameId"]
        self._last_version = version

    def _log_finished(self, state: Mapping[str, Any]) -> None:
        view = state.get("view") or {}
        LOGGER.info(
            "game finished after %d actions: %s",
            state.get("version", 0),
            json.dumps(view.get("gameOver"), separators=(",", ":")),
        )
        LOGGER.info(
            "bot decisions accepted: %d, fallbacks used: %d, rejected actions: %d",
            self.accepted,
            self.fallbacks,
            self.rejections,
        )

    # -- acting --------------------------------------------------------

    def _act(self, state: Mapping[str, Any]) -> None:
        adapter = self._adapter
        assert adapter is not None
        attempts = 0
        stale = 0
        excluded: set[int] = set()
        while not self._stop.is_set():
            view, version = state["view"], int(state["version"])
            snapshot = adapter.snapshot_memory()
            try:
                decision = adapter.decide(view, exclude=frozenset(excluded))
            except Exception as error:
                adapter.restore_memory(snapshot)
                LOGGER.warning("model path failed (%s: %s)", type(error).__name__, error)
                self._dumper.dump("decide-error", view, version, error)
                self._send_fallback(state, f"{type(error).__name__}: {error}")
                return
            self._wait_min_think()
            try:
                self.client.post_action(decision.game_action, version)
            except ZhanguoFatalError:
                raise
            except ZhanguoApiError as error:
                adapter.restore_memory(snapshot)
                if error.status == 409:
                    stale += 1
                    LOGGER.info("state was stale (version %d), refetching", version)
                    if stale > self.config.max_stale:
                        return
                    state = self._refetch(state)
                    if state is None:
                        return
                    continue
                if error.status != 400:
                    raise
                self.rejections += 1
                attempts += 1
                LOGGER.warning(
                    "action rejected: %s -> %s (attempt %d of %d)",
                    decision.note, error.message, attempts, self.config.max_attempts,
                )
                self._dumper.dump(
                    "rejected", view, version, error,
                    {"action": decision.game_action, "engine_action": decision.note},
                )
                excluded.add(decision.action_id)
                if attempts >= self.config.max_attempts:
                    self._send_fallback(state, f"{attempts} actions rejected")
                    return
                continue
            except ZhanguoNetworkError:
                adapter.restore_memory(snapshot)
                raise
            adapter.record_accepted()
            self.accepted += 1
            LOGGER.info(
                "turn %s %s: %s (attempt %d)",
                view["turn"],
                (view.get("pending") or {}).get("kind", "action"),
                decision.note,
                attempts + 1,
            )
            return

    def _refetch(self, state: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        fresh = self.client.get_state().state
        assert fresh is not None
        if fresh["status"] != "playing" or not fresh["yourTurn"]:
            return None
        self._turn_key = (fresh["gameId"], fresh["version"])
        return fresh

    def _wait_min_think(self) -> None:
        remaining = self.config.min_think - (self._clock() - self._turn_seen)
        if remaining > 0:
            self._sleep(remaining)

    def _send_fallback(self, state: Mapping[str, Any], reason: str) -> None:
        view, version = state["view"], int(state["version"])
        try:
            action = safe_action(view)
        except (ValueError, KeyError, IndexError) as error:
            LOGGER.error("no fallback for this position (%s)", error)
            self._exhausted_key = (state["gameId"], version)
            return
        LOGGER.warning("sending fallback %s because: %s", json.dumps(action), reason)
        self._dumper.dump("fallback", view, version, extra={"reason": reason, "action": action})
        try:
            self.client.post_action(action, version)
        except ZhanguoFatalError:
            raise
        except ZhanguoApiError as error:
            if error.status == 409:
                return
            if error.status != 400:
                raise
            LOGGER.error("fallback action rejected too (%s); waiting for a human", error.message)
            self._dumper.dump("fallback-rejected", view, version, error, {"action": action})
            self._exhausted_key = (state["gameId"], version)
            return
        self.fallbacks += 1
        if self._adapter is not None:
            self._adapter.record_accepted()
        self.accepted += 1
