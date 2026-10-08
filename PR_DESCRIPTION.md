# Add a "remote Zhanguo" adapter (`huangbot.remote`)

## Motivation

Max has a web board game, **Zhanguo**, that re-implements the game HuangEngine
implements. This PR lets the pretrained bot play in it as an external player:
Zhanguo exposes a small HTTP API ("External Bot Protocol v1"), the bot runs on a
PC at home, polls it once per second and answers when it is its turn. Only
outbound HTTPS calls are made.

The design rule: the bot code does not know which game it plays. The adapter
builds exactly the `DecisionContext` that `HuangEnvironment.decision_context()`
would have produced for the same position and translates the chosen engine
action back into Zhanguo's JSON. `model.py`, `checkpoints.py`,
`RecurrentModelAgent` and the weights are untouched.

## What changed

New, optional subpackage `huangbot/remote/` (standard library for HTTP, no new
dependencies):

| File | Purpose |
| --- | --- |
| `protocol.py` | constants, connect-string parsing, error types, token redaction |
| `board_check.py` | the 203-cell board as Zhanguo `rows`; map / player-count checks |
| `view_state.py` | `build_shadow_state`: game view -> `HuangState` |
| `score_history.py` | the 20 public-score observation features |
| `adapter.py` | `ZhanguoAdapter`: view -> `DecisionContext`, engine action -> Zhanguo action, multi-step war removal, GRU rollback |
| `fallback.py` | last-resort safe action |
| `client.py` | `urllib` client with ETag, error mapping, backoff |
| `runner.py` | poll / decide / act loop |
| `cli.py`, `__main__.py` | `huangbot-remote` / `python -m huangbot.remote` |

Other files:

* `pyproject.toml`: `packages = ["huang", "huangbot", "huangbot.remote"]` and the console script `huangbot-remote`.
* `README.md`: a short pointer section.
* `docs/REMOTE.md`: setup, usage, protocol, rule differences, limitations, findings.
* `tests/remote/`: the new tests (about 140 test cases; 25 random games by default, 300 in slow mode) and their helpers.
* `PR_DESCRIPTION.md`: this file (delete it before merging if you prefer).

## What did not change

Nothing in `huang-game-engine/` (the submodule pointer is untouched),
`model.py`, `environment.py`, `checkpoints.py`, `baselines.py`, `lookahead.py`,
the weights or the existing tests.

## How to test

```bash
git submodule update --init --recursive
python -m venv .venv && source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install numpy "torch>=2.9,<3" pytest
pip install --no-deps -e .
pytest -q tests/remote                                  # no network, no OpenSpiel
HUANGBOT_REMOTE_SLOW=1 pytest -q tests/remote/test_roundtrip.py   # 300 random games
```

The tests include random games against an engine-to-view oracle (shadow state
gives the same legal moves and observation as the engine, translation
round-trips), a whole game of the bundled model against an in-process mock of
the HTTP protocol (0 rejected actions), error paths (rejections, stale state,
network failures, 401/404), hidden-information checks and a check that every
module imports with `pyspiel` blocked.

`tests/remote/fixtures/zhanguo/*.json` (saved `ExtState` answers from the real
game) are replayed when present.

## OpenSpiel install note and suggestion

`pyproject.toml` lists `open_spiel>=2.0,<3` as a hard dependency; it has no
Windows wheel, so a plain `pip install -e .` fails there. `huangbot.remote` does
not need it (it never imports `pyspiel`, `huang.open_spiel_game` or the
`HuangEnvironment` class), so the install recipe is
`pip install numpy "torch>=2.9,<3"` followed by `pip install --no-deps -e .`.

Suggestion (not done here): move `open_spiel` to an optional extra, e.g.
`[project.optional-dependencies] openspiel = ["open_spiel>=2.0,<3"]`, so that the
remote mode and the engine-only tools install everywhere. The four tests in
`tests/test_runtime.py` construct `HuangEnvironment` and would then need
`pytest.importorskip("pyspiel")`.

## Known rule differences between the two engines

| Topic | Zhanguo | HuangEngine | Handling |
| --- | --- | --- | --- |
| After a green tile | pagoda offer, then market | market, then pagoda offer | the game's current decision selects the engine phase |
| Passing | `replace` with 0 tiles | needs 1 to 6 tiles | the model never passes; only the fallback does |
| Wars of 3+ states | only states with a leader colour that occurs elsewhere fight | every adjacent state with a leader | decide over the sides given; simulation may differ in rare cases |
| War tile removal | one action with all tiles | one tile per decision | `count` model calls, one combined action |
| Market | compacted list | six slots with holes | padded with `None` at the end |

The market difference matters a little more than the others: after any market
pick the tiles sit in different slots than the policy saw in training (the
engine refills holes in place, the game appends). Legality is unaffected.

## Limitations

Only the 203-cell "Bot-Brett" board and 2 to 4 players; the GRU memory is lost
if the process restarts mid-game; no lookahead in remote mode; the oracle in
the tests encodes the contract's assumptions, so the first run against the real
server may reveal mapping mismatches (use `--debug-dir`).

Not verified: a run against the real Zhanguo server (it was not available), and
the existing `tests/test_runtime.py` on a machine with OpenSpiel.
