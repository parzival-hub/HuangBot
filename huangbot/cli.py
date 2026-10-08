"""Play pretrained HUANG bots and save viewer-compatible matches."""

import argparse
from pathlib import Path

import torch

from huang.playback import save_playback

from .baselines import create_baseline
from .checkpoints import load_checkpoint
from .environment import HuangEnvironment, observation_options
from .lookahead import InformationSetLookaheadAgent, LookaheadConfig
from .model import RecurrentModelAgent
from .recording import record_match


def main(argv=None):
    parser = argparse.ArgumentParser(description="Play the pretrained HUANG bot.")
    parser.add_argument("output_file", type=Path, nargs="?", default=Path("playbacks/bot-game.json"))
    parser.add_argument("--player-1", choices=("model", "heuristic", "type-random", "uniform-random"), default="model")
    parser.add_argument("--player-2", choices=("model", "heuristic", "type-random", "uniform-random"), default="heuristic")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--stochastic-model", action="store_true")
    parser.add_argument("--max-player-decisions", type=int)
    parser.add_argument("--lookahead-depth", type=int, default=2)
    parser.add_argument("--lookahead-top-k", type=int, default=4)
    parser.add_argument("--lookahead-simulations", type=int, default=4)
    parser.add_argument("--lookahead-time-budget", type=float, default=3.0)
    parser.add_argument("--no-adaptive-lookahead", action="store_true")
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.max_player_decisions is not None and args.max_player_decisions <= 0:
        parser.error("--max-player-decisions must be positive")
    if args.lookahead_depth < 0:
        parser.error("--lookahead-depth must be non-negative")
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    names = (args.player_1, args.player_2)
    model = load_checkpoint(args.checkpoint, device=device) if "model" in names else None
    environment = HuangEnvironment(seed=args.seed, max_player_decisions=args.max_player_decisions)
    if model is not None:
        environment = HuangEnvironment(
            seed=args.seed,
            max_player_decisions=args.max_player_decisions,
            **observation_options(model.config.observation_size),
        )
        if model.config.observation_size != environment.observation_size or model.config.num_actions != environment.num_actions:
            parser.error("checkpoint does not match the engine's observations and actions")
    model_agent = None
    if model is not None:
        if args.lookahead_depth:
            config = LookaheadConfig(depth=args.lookahead_depth, top_k=args.lookahead_top_k,
                simulations=args.lookahead_simulations,
                max_time_seconds=args.lookahead_time_budget, adaptive=not args.no_adaptive_lookahead)
            try:
                config.validate()
            except ValueError as error:
                parser.error(str(error))
            model_agent = InformationSetLookaheadAgent(model, players=2, config=config, seed=args.seed)
        else:
            model_agent = RecurrentModelAgent(model, players=2, deterministic=not args.stochastic_model, seed=args.seed)
    agents = [model_agent if name == "model" else create_baseline(name, seed=args.seed + seat) for seat, name in enumerate(names)]
    playback, result = record_match(environment, agents, seed=args.seed)
    destination = save_playback(playback, args.output_file)
    ending = "finished" if result.terminated else "decision limit reached"
    print(f"Saved {destination}: {result.player_decisions} decisions, {ending}.")
    return 0
