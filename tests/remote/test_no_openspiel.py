import importlib
import pkgutil
import subprocess
import sys

import huangbot.remote


def test_every_remote_module_imports_without_openspiel():
    code = (
        "import sys; sys.modules['pyspiel'] = None\n"
        "import importlib, pkgutil, huangbot.remote as pkg\n"
        "names = [m.name for m in pkgutil.iter_modules(pkg.__path__) if m.name != '__main__']\n"
        "for name in names: importlib.import_module('huangbot.remote.' + name)\n"
        "assert 'huangbot.remote.cli' in sys.modules and 'huang.open_spiel_game' not in sys.modules\n"
        "print(len(names))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) >= 9


def test_module_list_matches_the_documented_layout():
    names = {m.name for m in pkgutil.iter_modules(huangbot.remote.__path__)}
    assert names == {
        "__main__", "adapter", "board_check", "cli", "client", "fallback",
        "protocol", "runner", "score_history", "view_state",
        "information", "tile_history",
    }


def test_in_process_import_with_pyspiel_blocked(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyspiel", None)
    for name in list(sys.modules):
        if name.startswith("huangbot.remote."):
            monkeypatch.delitem(sys.modules, name)
    for module in pkgutil.iter_modules(huangbot.remote.__path__):
        if module.name != "__main__":
            importlib.import_module(f"huangbot.remote.{module.name}")
