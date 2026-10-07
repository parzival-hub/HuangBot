# HuangBot

Run a pretrained bot for Reiner Knizia's HUANG. This repository includes the
neural policy, recurrent player memory, optional information-set lookahead,
reference opponents, and a match recorder. The rules engine is pinned as a
submodule from [HuangEngine](https://github.com/parzival-hub/HuangEngine).

## Install

Use Python 3.11 or newer:

```bash
git clone --recurse-submodules https://github.com/parzival-hub/HuangBot.git
cd HuangBot
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on Linux, then install:

```bash
python -m pip install .
```

For an existing clone, run `git submodule update --init --recursive` first.
The built wheel contains both the engine package and pretrained weights.

## Run the bot

Play the bundled bot against the heuristic opponent and save a replay:

```bash
python -m huangbot playbacks/bot-game.json
huang-viewer playbacks/bot-game.json --autoplay
```

The default device is CUDA when available, otherwise CPU. To force CPU:

```bash
python -m huangbot playbacks/bot-game.json --device cpu
```

Use `--player-1 heuristic --player-2 model` to change seats, `--stochastic-model`
to sample policy actions, or `--checkpoint path/to/model.pt` to load other
compatible weights. Matches normally run to the official game end; an optional
`--max-player-decisions 2000` imposes a decision limit.

Optional lookahead samples hidden tile allocations using only the acting
player's information:

```bash
python -m huangbot playbacks/search-game.json --lookahead-depth 2 --lookahead-top-k 8 --lookahead-simulations 16
```

Run `python -m huangbot --help` for all options.

## Use in Python

```python
from huangbot import HuangEnvironment
from huangbot.checkpoints import load_checkpoint
from huangbot.model import RecurrentModelAgent

model = load_checkpoint(device="cpu")
environment = HuangEnvironment(seed=42, include_public_score_history=True)
agent = RecurrentModelAgent(model, players=2, deterministic=True)
step = environment.reset()
while not step.done:
    context = environment.decision_context()
    step = environment.step(agent.select_action(context))
agent.reset_episode()  # reset memory before the next game
```

The bundled model uses the standard 5,410 observation features plus 20 features
derived from publicly announced point awards. Other checkpoints may use only
the standard features; the command-line runner detects both formats.

## Bundled model

`huangbot/models/best.pt` contains the weights of the best checkpoint selected
by the `high_elo_continuous_20260928` run (update 2240). Its recorded selection
evaluation scored 85.5% match points over 100 games against five checkpoint
opponents. This is the run's selection result, not a new benchmark.
`huangbot/models/model.json` records its provenance. Optimizer state, training
logs, and earlier checkpoints are omitted.

HUANG is a game by Reiner Knizia. This is an unofficial project; no third-party
rule PDFs or game artwork are included.
