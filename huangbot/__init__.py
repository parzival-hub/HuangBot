"""Run pretrained HUANG bots against the independent HuangEngine."""

from pathlib import Path
import sys


# Support a source checkout as well as wheels containing the engine package.
_engine_source = Path(__file__).resolve().parents[1] / "huang-game-engine"
if _engine_source.is_dir():
    sys.path.insert(0, str(_engine_source))

from .environment import HuangEnvironment

__all__ = ["HuangEnvironment"]
