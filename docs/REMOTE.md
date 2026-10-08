# Playing in Zhanguo (remote mode)

`huangbot.remote` lets the pretrained HuangBot play as an external player in
**Zhanguo**, a private web implementation of the same board game. The game
server exposes a small HTTP API ("External Bot Protocol v1"); the bot runs on
your own PC, polls the API once per second and answers when it is its turn. The
bot only makes outbound HTTPS calls: no port forwarding, no inbound connection.

The rule that shaped the design: **the bot does not know which game it plays.**
`model.py`, `checkpoints.py`, `RecurrentModelAgent` and the weights are
untouched. The adapter hands the agent exactly the `DecisionContext` that
`HuangEnvironment.decision_context()` would have produced for the same
position (same observation tensor, same legal-action mask, same action ids) and
translates the chosen engine action back into Zhanguo's JSON.

```
 Zhanguo server (Node, behind Caddy)                    your PC
 ┌──────────────────────────┐   HTTPS, JSON    ┌────────────────────────────────────────────┐
 │ rules engine of Zhanguo  │ <─────────────── │ ZhanguoClient   GET /state (ETag, 1 Hz)    │
 │ GET  /api/ext/<id>/state │ ───────────────> │      │ view (JSON)                         │
 │ POST /api/ext/<id>/action│                  │      v                                     │
 │ POST /api/ext/<id>/status│                  │ build_shadow_state  ->  HuangState         │
 └──────────────────────────┘                  │      │ (HuangEngine, hidden info removed)  │
                                               │      v                                     │
                                               │ DecisionContext  ->  RecurrentModelAgent   │
                                               │      │ action id        (unchanged)        │
                                               │      v                                     │
                                               │ engine Action -> Zhanguo Action -> POST    │
                                               └────────────────────────────────────────────┘
```

## Install

The package does **not** need OpenSpiel. `pyproject.toml` lists `open_spiel` as
a hard dependency, which has no wheel on Windows, so a plain `pip install -e .`
fails there. This recipe works everywhere (Python 3.11 or newer; PyPI has
`torch` 2.14 wheels for 3.11, 3.12, 3.13 and 3.14 on 64-bit Windows):

```bash
git clone --recurse-submodules <repository-url> HuangBot
cd HuangBot                       # for an existing clone: git submodule update --init --recursive
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1      Linux/macOS: source .venv/bin/activate
pip install numpy "torch>=2.9,<3"
pip install --no-deps -e .
pip install pytest                # only needed for the tests
```

Check it: `pytest -q tests/remote` (no network, no OpenSpiel needed).
`tests/test_runtime.py` (the original tests) exercises `HuangEnvironment` and
therefore does need OpenSpiel; it fails with "HuangEnvironment requires
OpenSpiel" on a machine without it.

## Get a connect string

1. In Zhanguo, create a game on the map **Bot-Brett** with 2 to 4 seats.
2. The host clicks **"Externen Bot hinzufügen"** in the lobby. The lobby shows
   one copyable string:

```
https://zhanguo.example.de/api/ext/ab12cd34ef#Zk3Jq...seatToken
└────────── base URL ───────────┘ └ gameId ┘ └ secret seat token
```

The token is sent only in the header `Authorization: Bearer <token>`, never in a
URL (a fragment is never sent to a server anyway). **Quote the string**: `#`
starts a comment in bash. In PowerShell use quotes too.

## Run

```bash
huangbot-remote "https://zhanguo.example.de/api/ext/ab12cd34ef#Zk3Jq..."
# or: python -m huangbot.remote "<connect string>"
```

The seat then shows as connected in the lobby (with agent name, "ready" and a
note). The host starts the game; the bot plays in the real UI. The process ends
by itself when the game is finished.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--checkpoint PATH` | bundled `best.pt` | other compatible weights (5410 or 5430 inputs) |
| `--device cpu\|cuda\|auto` | `cpu` | where the model runs |
| `--threads N` | 1 | torch threads |
| `--interval S` | 1.0 | seconds between polls |
| `--min-think S` | 0.9 | minimum time between seeing its turn and answering, so humans can follow |
| `--stochastic` | off | sample from the policy instead of arg-max |
| `--debug-dir DIR` | off | write JSON dumps of every problem (see below) |
| `--log-level L` | `INFO` | Python log level |
| `--max-errors N` | 0 | exit with code 1 after N consecutive server/network errors (0: keep trying) |
| `--strict` | off | treat disagreements between the game's option lists and the engine as errors |

Exit codes: `0` game finished, `1` `--max-errors` reached, `2` usage or
configuration error (bad connect string, unusable checkpoint, wrong token, game
not found, unsupported protocol version), `130` Ctrl+C.

### Log lines

```
12:00:01 INFO reported status ready=True
12:00:02 INFO playing seat 1 of 3 players
12:00:05 INFO turn 7 action: place green tile at (10,4) (attempt 1)
12:00:07 INFO turn 7 market: take white from market slot 0 (attempt 1)
12:00:09 WARNING action rejected: place red tile at ... -> <server message> (attempt 1 of 3)
12:00:12 WARNING sending fallback {"type": "market", "index": null} because: 3 actions rejected
12:30:40 INFO game finished after 213 actions: [...]
12:30:40 INFO bot decisions accepted: 96, fallbacks used: 0, rejected actions: 0
```

One INFO line per decision: game turn, the kind of decision (`action` for a
normal action, otherwise the game's pending kind), the engine action chosen,
and the attempt number. The token never appears in logs, dumps or error
messages.

## Errors, retries and the fallback

* **Network errors / HTTP 5xx**: logged, retried with backoff 1, 2, 4, ... up to
  30 s. Never fatal (unless `--max-errors`).
* **401** ("token invalid or seat removed"), **404** ("game not found"), a
  redirect, or a protocol version other than 1: fatal, exit code 2. Redirects
  are never followed because the HTTP library would forward the token.
* **409 stale**: the state changed under us. The decision is discarded (the GRU
  memory is rolled back), the state is fetched again and the bot decides again.
  Not counted as an error.
* **400 rejected** by the game's rules engine: that action id is excluded, the
  GRU memory is rolled back and the model decides again (up to 3 attempts).
* **Fallback**: after 3 rejections, or on any exception inside the model path
  (`ShadowStateError`, `TranslationError`, ...), the most passive legal answer
  derived from the view alone is sent (pass / decline / commit nothing /
  first candidates), so a game never hangs. It is logged at WARNING and counted.
  If even that is rejected, the runner logs ERROR and keeps polling; a human can
  act for the seat in the UI.
* **Wrong board or player count**: the bot reports `ready: false` with a note
  and does nothing else, even if the host starts the game anyway.

### Debug dumps

With `--debug-dir DIR` every `ShadowStateError`, `TranslationError`, rejected
action and fallback writes `{timestamp}-{version}-{reason}.json` containing
`{view, error, traceback}` (plus the rejected action). Send those to both repo
owners: they are how rule and mapping mismatches are found. A dump of a
position that the shadow state or the translation could not handle can be
copied to `tests/remote/fixtures/zhanguo/` to become a regression test (see
[Tests](#tests)).

## How it works

| Module | Role |
| --- | --- |
| `protocol.py` | constants, `parse_connect_string`, error types, token redaction |
| `board_check.py` | the one supported board; map/player-count compatibility |
| `view_state.py` | `build_shadow_state(view, seat=...)` -> `HuangState` |
| `score_history.py` | the 20 public-score observation features |
| `adapter.py` | `ZhanguoAdapter.decide(view)`: view -> `DecisionContext` -> agent -> Zhanguo action |
| `fallback.py` | `safe_action(view)` |
| `client.py` | standard-library HTTP client, ETag, backoff |
| `runner.py` | the poll / decide / act loop |
| `cli.py`, `__main__.py` | `huangbot-remote` |

**Shadow state.** The view is turned into a `HuangState` whose fields are
overwritten from the view: tiles, leaders, pagodas, the bot's own hand and
points, the market (padded with `None` to six slots), `bagCount`, turn, active
player and `actions_remaining` (`actionsLeft - 1` while `inAction`, because the
game decrements when an action finishes and the engine when it starts). The
game's `pending` decision selects the engine phase and its revolt/war context.
Other seats' hands are unknown: a placeholder of the right size is used, and the
observation only reads hand sizes (tests prove the composition does not matter).
Other seats' points are zero; they reach the model only through the public
score history, summed from `view.log[*].vp` and divided by 30 once.

**Observation size.** A checkpoint with 5430 inputs gets the 20 score-history
features appended (as `HuangEnvironment(include_public_score_history=True)`
does); one with 5410 does not; any other size refuses to start.

**War tile removal.** The game wants all `count` tiles in one action, the engine
removes one tile per decision. The adapter runs the model `count` times on the
shadow state (applying each removal to the shadow) and sends one combined
`warRemove`. The GRU therefore advances once per sub-decision, as in training.

**GRU memory.** One hidden state for the bot's seat. A discarded decision
(rejection, stale state, lost answer) is rolled back so the memory advances
exactly once per accepted action.

## Protocol summary (External Bot Protocol v1)

Two programs talk over HTTPS + JSON; neither knows the other's internals.

### Connect string

`<origin>/api/ext/<gameId>#<token>`. The bot splits at `#`. The token goes only
in `Authorization: Bearer <token>`.

### Endpoints (all under `<origin>/api/ext/<gameId>`, all need the Bearer header)

| Method + path | Purpose |
| --- | --- |
| `GET /state` | Current state for the bot's seat. Supports `If-None-Match` with the ETag of the last answer, then answers `304` with an empty body when nothing changed. |
| `POST /action` | Body `{ "action": Action, "version": <number from last state> }`. |
| `POST /status` | Body `{ "agent": "huangbot-remote 0.1 (high_elo_continuous_20260928)", "ready": true, "note": "optional text" }`. Tells the lobby who the bot is and whether it can play this game. |

Status codes: `200` ok, `304` unchanged, `400 {error}` rejected by the rules
engine (German message), `401` missing/invalid token, `404` unknown game,
`409 {error:"stale", version}` when `version` is not the current one. Error
bodies are always `{ "error": string, "version"?: number }`.

### `GET /state` (ExtState)

```ts
interface ExtState {
  protocol: 1;
  gameId: string;
  seat: number;                      // index of the bot's seat (0-based)
  status: 'lobby' | 'playing' | 'finished';
  version: number;                   // number of accepted actions so far (+1 per accepted action)
  yourTurn: boolean;                 // status==='playing' && the engine's currentActor === seat
                                     //   (also true when the bot must decide in someone else's revolt/war)
  lobby: LobbyInfo;                  // always present
  view: GameView | null;             // null while status==='lobby'; the seat's filtered view
}
```

`yourTurn === false` means: do nothing, poll again.

### Data types

```ts
type Color = 'yellow' | 'red' | 'blue' | 'green' | 'white';   // yellow=governor, red=soldier, blue=farmer, green=trader, white=artisan
type VP = Record<Color, number>;
interface LeaderRef { player: number; color: Color }
interface Pagoda { id: number; color: Color; tri: [number, number, number] | null }   // null: in supply
interface VpGain { p: number; color: Color; n: number }

interface WarSide {
  spaces: number[];                 // occupied spaces of this state BEFORE the unifying tile
  leaders: { space: number; player: number; color: Color }[];
  board: number;                    // red tiles in this state
  committed: number;                // red tiles committed from hands so far
  leaderSupport: number[];          // players that added their off-board red leader to this side
}
interface WarData {
  unifier: number; tile: number;    // tile = space of the unifying tile
  sides: WarSide[];
  order: number[];                  // contribution order: (unifier+1)%n, ..., unifier last
  idx: number;                      // index into `order` of the next contributor
  winner?: number;
  result?: { totals: number[]; removedLeaders: LeaderRef[]; removedTiles: number; vp: VpGain[] };
}

type Pending =
  | { kind: 'pagodaBuild'; player: number; color: Color; triangles: [number, number, number][]; sources: number[] }
  | { kind: 'market'; player: number }
  | { kind: 'chain'; player: number; from: number; targets: number[] }
  | { kind: 'revolt'; player: number; stage: 'attacker' | 'defender'; attacker: number; defender: number; color: Color;
      aSpace: number; dSpace: number; aBase: number; dBase: number; aCommit: number; dCommit: number; aLeader: boolean; dLeader: boolean }
  | { kind: 'war'; player: number }
  | { kind: 'warTie'; player: number; tied: number[] }
  | { kind: 'warRemove'; player: number; count: number; candidates: number[] };

type Action =                                                         // what the bot sends
  | { type: 'leader'; color: Color; to: number | null }               // to=null: withdraw the leader
  | { type: 'tile'; color: Color; space: number }
  | { type: 'riot'; space: number; useLeader?: boolean }
  | { type: 'pagoda'; tri: [number, number, number]; useLeader?: boolean; source?: number }
  | { type: 'replace'; tiles: Color[] }                               // 0..6 tiles; [] = pass
  | { type: 'pagodaBuild'; tri: [number, number, number] | null; source?: number }  // null declines
  | { type: 'market'; index: number | null }                          // index into view.market, null declines
  | { type: 'chain'; space: number | null }                           // null stops the blue chain
  | { type: 'revoltCommit'; count: number; useLeader?: boolean }
  | { type: 'warCommit'; side: number | null; count: number; useLeader?: boolean }
  | { type: 'warTie'; side: number }
  | { type: 'warRemove'; spaces: number[] };                          // exactly pending.count spaces, ALL IN ONE ACTION

interface PlayerView { name: string; dynasty: string; handCount: number; hand: Color[] | null; vp: VP | null }

interface GameView {                // = GameState minus bag, seed and players, plus:
  options: { shortGame?: boolean };
  map: MapDef;                      // { name, rows: string[], names? }  '.' land, '~' river, 'C' capital, '#' not playable
  market: Color[];                  // COMPACTED, length <= 6
  tiles: (Color | null)[];          // indexed by space id
  leaders: (LeaderRef | null)[];    // indexed by space id
  pagodas: Pagoda[];                // 9 entries: 2 yellow, 2 red, 2 blue, 2 green, 1 white; id = index
  current: number;                  // player whose turn it is
  turn: number;
  actionsLeft: number;              // 2 at turn start. NOT yet decremented for an action still being resolved (inAction)
  inAction: boolean;
  pending: Pending | null;          // the decision the engine waits for; pending.player must answer
  queue: unknown[];                 // ignored
  war: WarData | null;
  unification: number | null;
  log: LogEntry[];                  // FULL history; entries that carry points have `vp: VpGain[]`
  gameOver: unknown[] | null;
  lastPlaced: number[];
  bagCount: number;                 // tiles left in the hidden bag
  players: PlayerView[];
  you: number | null;               // = seat
}

interface MapDef { name: string; rows: string[]; names?: Record<string, string> }
interface SeatInfo { name: string; dynasty: string; online: boolean; bot: boolean; external?: boolean;
                     ext?: { agent?: string; ready?: boolean; note?: string } }
interface LobbyInfo { id: string; name: string; status: 'lobby'|'playing'|'finished'; seats: SeatInfo[];
                      options: { shortGame?: boolean }; map: MapDef; hasWebhook: boolean; createdAt: number; actionCount: number }
```

### Semantics the bot relies on

* Player indices are 0-based seat indices; turn order is increasing index (wrapping); `view.you === seat`.
* Space ids are numbered row by row over the non-`#` cells; for the supported board the space id is the engine's cell index (neighbours and the 353 triangles are identical).
* Colours are the strings above; the engine's `Color` enum has the same order (`YELLOW=0 ... WHITE=4`).
* Hidden information: the bot sees only its own hand and points; everyone else has `hand: null, vp: null` and a `handCount`. Their publicly announced points are in `log[*].vp`.
* `version` changes only when an action is accepted. Lobby changes change the ETag, not the version.
* One action per call. Many actions open follow-up decisions (`pending`), each a separate `POST /action` expected from `pending.player`.
* The server never trusts the bot: an illegal action returns 400 and changes nothing.
* Game over: `status === 'finished'`, `yourTurn === false` forever; the bot exits.

### The one supported board: "Bot-Brett" (`bot203`)

15 rows, odd-r offset, pointy-top hexes, 203 playable spaces (159 land incl.
capitals, 44 river), 7 capitals at space ids `[15, 60, 68, 93, 113, 132, 164]`,
353 triangles (8 river-only, 234 land-only). `MapDef.rows`:

```
".........~~~~##"  "..C....~~..~.##"  "......~~......#"  "~...~~........#"
"~....~C.......C"  "~....~........#"  "~.....~...C...."  "~.....~~~.....#"
"~C.......~~~~~~"  "~....C.......~#"  "#~...........~#"  "#~.........C..#"
"#~~............"  "#~~...........#"  "##~~~~~~...####"
```

The bot compares `lobby.map.rows` with the rows generated from its own
`huang.board` constants and refuses to play on any other board. It plays 2 to 4
players (the game allows 2 to 5; a 5-player game is reported `ready: false`).

## Rule differences between the two engines

The repositories were written independently. Handled as follows (nobody "fixes"
the other side's rules):

| # | Topic | Zhanguo | HuangEngine | Handling |
| --- | --- | --- | --- | --- |
| 1 | Decisions after a green tile | pagoda offer first, then market | market first, then pagoda offer | the game's current `pending` selects the matching engine phase; nothing is reordered |
| 2 | Passing | `replace` with 0 tiles is legal | `replace` needs 1 to 6 tiles; no pass | the model never passes; `replace []` is only the last-resort fallback |
| 3 | Wars of 3+ states | only states with a leader whose colour appears in another united state fight | every adjacent state with a leader is a side | the bot decides over the sides the view carries; its war simulation may differ in rare 3-state wars (accepted for v1) |
| 4 | War tile removal | one action with all `count` spaces | one `war_remove` per tile | `count` model calls on the shadow state, one combined action |
| 5 | Market slots | `view.market` is compacted | six fixed slots with holes | padded with `None` at the end; slot index = list index |

Additionally observed while building this (see also *Findings*):

* **Market slot positions drift from the training distribution.** The engine
  refills holes in place; the game compacts the list and appends refills. After
  any market pick the tiles therefore sit in different slots than the policy
  saw in training, and `market_take(slot)` addresses the compacted list. Legality
  is unaffected (the game's own indices are used) but the observation differs
  from HuangEnvironment's. The adapter cannot reconstruct the engine layout from
  a compacted list. Tests compare against an engine state with a compacted
  market for exactly this reason.
* **`board` of a war side at `warRemove` time.** The shadow state computes
  strengths as `board + committed + len(leaderSupport)`. If the game recomputes
  `board` after the losing sides' red tiles were removed, the strength features
  seen in the `warRemove` observation differ from the engine's (they only feed
  the tile-selection decision). Please check with a fixture.

## Limitations

* Only the 203-cell board "Bot-Brett" and 2 to 4 players.
* The GRU memory lives in the process. If it is restarted mid-game the bot
  continues with a fresh memory (and says nothing about it).
* Lookahead (`InformationSetLookaheadAgent`) is not supported in remote mode.
* Differences listed above (3-state wars, market layout, decision order).
* The bot assumes `view.log[*].vp` contains every publicly announced point
  award exactly once (tile scoring, revolt, war, pagoda income). It does not
  look at `war.result.vp`.
* One game per process, one seat.

## Security

The connect string contains the secret seat token: whoever has it can play that
seat. Share it only with the person who runs the bot, do not paste it into chats
or screenshots, and treat a leaked string like a leaked password (kick the seat
and add a new external bot in the lobby). The bot never writes the token to
logs, error messages or debug dumps, does not follow HTTP redirects (so it can
not be forwarded to another host), and sends it only in the `Authorization`
header.

## Tests

```bash
pytest -q tests/remote                     # fast, deterministic, no network, no OpenSpiel
HUANGBOT_REMOTE_SLOW=1 pytest -q tests/remote/test_roundtrip.py   # 300 random games instead of 25
```

* `export_view.py` (test helper) builds a game-format view from a `HuangState`;
  it is the oracle for `test_roundtrip.py`, which plays random 2-, 3- and
  4-player games and checks at every decision that the shadow state yields the
  same legal moves and the same observation as the engine, that the score
  history matches `HuangEnvironment`'s arithmetic, and that every chosen action
  survives translation into the game format and back (an independent second
  implementation of the action table).
* `test_runner_mock_server.py` runs the real bundled model through a whole game
  against an in-process HTTP server that implements the protocol on top of a real
  engine game.
* `test_fixtures_from_game.py` replays saved `ExtState` files from the real game
  if you put them in `tests/remote/fixtures/zhanguo/` (each with `yourTurn:
  true`); it is skipped otherwise.

## Findings

Things found while building and testing this; none was changed in this PR.

1. `pyproject.toml` requires `open_spiel` unconditionally, so `pip install -e .`
   fails on Windows; the four tests in `tests/test_runtime.py` need it too
   (`HuangEnvironment` imports `pyspiel` in its constructor). Suggestion: make it
   an optional extra.
2. `public_score_history_from_engine` in `environment.py` divides by 30 exactly
   once, like `HuangEnvironment.observation_for_player`; the two are identical
   (`tests/remote/test_score_history.py` asserts it). `huangbot.remote` still
   re-implements the arithmetic so it does not depend on an engine object.
3. Market layout difference (see above); not fixable without changing either
   side.
4. The oracle used by the tests encodes the assumptions of the contract
   (shape of `pending`, `war`, `log[*].vp`). Whether the real server agrees can
   only be shown by fixtures from the game (`test_fixtures_from_game.py`).

## Proposed core changes

None. Everything is contained in `huangbot/remote/`; only `pyproject.toml`
(package list, one console script) and `README.md` (a pointer) changed outside
of it.
