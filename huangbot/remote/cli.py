"""``huangbot-remote``: let the pretrained bot play a game in Zhanguo."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .protocol import ConnectString, ConnectStringError, parse_connect_string, redact

LOGGER = logging.getLogger("huangbot.remote")


class _RedactFilter(logging.Filter):
    """Belt and braces: the seat token must never reach a log line."""

    def __init__(self, token: str):
        super().__init__()
        self._token = token

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if self._token in message:
            record.msg, record.args = redact(message, self._token), ()
        return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="huangbot-remote",
        description="Play the pretrained HUANG bot in a Zhanguo game (External Bot Protocol v1).",
    )
    parser.add_argument(
        "connect_string",
        help="<origin>/api/ext/<gameId>#<token> from the Zhanguo lobby (put it in quotes)",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between polls")
    parser.add_argument(
        "--min-think", type=float, default=0.9,
        help="minimum seconds between seeing our turn and answering",
    )
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of arg-max")
    parser.add_argument("--debug-dir", type=Path, help="write JSON dumps of problems here")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--max-errors", type=int, default=0,
        help="exit after this many consecutive server errors (0 = keep trying)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="treat disagreements between the game's option lists and the engine as errors",
    )
    parser.add_argument("--version", action="version", version=f"huangbot-remote {__version__}")
    return parser


def checkpoint_label(path: Optional[Path]) -> str:
    """Short model description for the lobby (``high_elo_continuous_20260928``)."""
    from huangbot.checkpoints import DEFAULT_CHECKPOINT

    checkpoint = Path(path) if path is not None else DEFAULT_CHECKPOINT
    try:
        metadata = json.loads((checkpoint.parent / "model.json").read_text(encoding="utf-8"))
        if path is None and isinstance(metadata.get("source_run"), str):
            return metadata["source_run"]
    except (OSError, ValueError):
        pass
    return checkpoint.stem


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.interval <= 0 or args.min_think < 0 or args.max_errors < 0:
        parser.error("--interval must be positive; --min-think and --max-errors must not be negative")
    level = getattr(logging, str(args.log_level).upper(), None)
    if not isinstance(level, int):
        parser.error(f"unknown log level {args.log_level!r}")
    try:
        connection = parse_connect_string(args.connect_string)
    except ConnectStringError as error:
        print(f"huangbot-remote: {error}", file=sys.stderr)
        return 2

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    handler.addFilter(_RedactFilter(connection.token))
    package_logger = logging.getLogger("huangbot.remote")
    package_logger.addHandler(handler)
    package_logger.setLevel(level)
    package_logger.propagate = False
    try:
        return _run(args, connection)
    finally:
        package_logger.removeHandler(handler)
        package_logger.setLevel(logging.NOTSET)
        package_logger.propagate = True


def _run(args: argparse.Namespace, connection: ConnectString) -> int:
    import torch

    from huangbot.checkpoints import load_checkpoint

    from .client import ZhanguoClient
    from .runner import ConfigError, RemoteRunner, RunnerConfig

    torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = load_checkpoint(args.checkpoint, device=device)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"huangbot-remote: cannot load the checkpoint: {error}", file=sys.stderr)
        return 2
    config = RunnerConfig(
        interval=args.interval,
        min_think=args.min_think,
        max_errors=args.max_errors,
        debug_dir=args.debug_dir,
        deterministic=not args.stochastic,
        strict=args.strict,
        agent_name=f"huangbot-remote {__version__} ({checkpoint_label(args.checkpoint)})",
    )
    client = ZhanguoClient(connection.origin, connection.game_id, connection.token)
    try:
        runner = RemoteRunner(client, model, config)
    except ConfigError as error:
        print(f"huangbot-remote: {error}", file=sys.stderr)
        return 2
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
