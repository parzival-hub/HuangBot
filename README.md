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

The bot now plays full games without the 24 initial short-game removals.
Adaptive lookahead is enabled by default. A permanent tile ledger counts
placements and public payments even after tiles leave the board. Own concealed
exchanges are counted by colour; opponent exchanges retain unknown colours.
Known opponent market acquisitions constrain possible hands. Remaining bag
colours are estimated from an exchangeable, without-replacement distribution
over unknown hand and discard slots; hidden engine colours are never consulted.

Search starts with depth 2 and 4 simulations per candidate. At 30 or fewer bag
tiles the targets grow to depth 4 and 16 simulations; at 10 or fewer, depth 8
and 32; at 3 or fewer, depth 12 and 64. High next-draw entropy increases the
simulation target, up to 64 for the default settings. Conflicts request at
least 16 simulations. Wars, revolts, blue chains and placement choices are
resolved before a leaf is evaluated, and all chance draws are resolved.

Each move has a soft three-second search budget. Every candidate receives the
same number of sampled worlds; a started round finishes, including its tactical
phases, before the time limit is checked. Adjust the search with:

```bash
python -m huangbot playbacks/search-game.json --lookahead-depth 2 --lookahead-top-k 4 --lookahead-simulations 4 --lookahead-time-budget 3
```

Run `python -m huangbot --help` for all options.
Use `--lookahead-depth 0` for policy-only play, or `--no-adaptive-lookahead`
to retain a fixed depth and simulation target. Tactical phases still finish.
From Python, `environment.tile_knowledge(player).to_dict()` exposes the acting
player's colour probabilities, expected bag counts and variances. These are
search inputs. Checkpoints trained with the additional 37 neural tile features
receive them automatically; existing checkpoints retain their input format.

## Use in Python

```python
from huangbot import HuangEnvironment
from huangbot.checkpoints import load_checkpoint
from huangbot.model import RecurrentModelAgent
from huangbot.environment import observation_options

model = load_checkpoint(device="cpu")
environment = HuangEnvironment(seed=42, **observation_options(model.config.observation_size))
agent = RecurrentModelAgent(model, players=2, deterministic=True)
step = environment.reset()
while not step.done:
    context = environment.decision_context()
    step = environment.step(agent.select_action(context))
agent.reset_episode()  # reset memory before the next game
```

The current models use the standard 5,410 observation features, 20 features
derived from publicly announced point awards, and 37 features for permanent
tile counts, draw probabilities and uncertainty. The command-line runner also
supports checkpoints using the original 5,410 or 5,430 input formats.

## Bundled model

`huangbot/models/best.pt` contains checkpoint **8000** from
`full_game_tile_belief_unlimited_20261008`. It was selected from nine checkpoints using
144 screening games, followed by 96 independent confirmation games between
the top three candidates. Its confirmation match-point rate was
56.2% over 64 games against the other two finalists.

All benchmark games used the full rules, balanced seats and the same adaptive
lookahead settings, with a soft 0.25-second search budget per decision.
The scores describe this benchmark's candidate pool.
`huangbot/models/model.json` records the checkpoint provenance, pair results,
search settings and weight checksum. The package contains inference weights.

HUANG is a game by Reiner Knizia. This is an unofficial project; no third-party
rule PDFs or game artwork are included.
