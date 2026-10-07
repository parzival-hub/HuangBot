import torch

from huang.playback import load_playback
from huangbot import HuangEnvironment
from huangbot.checkpoints import DEFAULT_CHECKPOINT, load_checkpoint
from huangbot.cli import main
from huangbot.model import RecurrentModelAgent


def test_bundled_weights_are_inference_only_and_select_legal_actions():
    torch.set_num_threads(1)
    payload = torch.load(DEFAULT_CHECKPOINT, weights_only=True)
    assert set(payload) == {"format_version", "model_config", "model_state"}
    model = load_checkpoint()
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    environment = HuangEnvironment(seed=42, include_public_score_history=True)
    step = environment.reset()
    assert step.observation.shape == (model.config.observation_size,)
    agent = RecurrentModelAgent(model, players=2, deterministic=True)
    for _ in range(8):
        context = environment.decision_context()
        other_seat = 1 - context.player
        other_memory = agent.memory.get(other_seat).clone()
        action = agent.select_action(context)
        assert action in context.legal_actions
        assert torch.equal(other_memory, agent.memory.get(other_seat))
        step = environment.step(action)
        if step.done:
            break
    agent.reset_episode()
    assert all(torch.count_nonzero(agent.memory.get(seat)) == 0 for seat in (0, 1))


def test_cli_runs_bundled_model_and_records_playback(tmp_path):
    destination = tmp_path / "game.json"
    assert main([str(destination), "--device", "cpu", "--max-player-decisions", "4"]) == 0
    playback = load_playback(destination)
    assert playback["metadata"]["agents"] == ["recurrent-model", "heuristic"]
    assert playback["metadata"]["player_decisions"] == 4
    assert len(playback["frames"]) == 5


def test_lookahead_uses_bundled_observation_format(tmp_path):
    destination = tmp_path / "search.json"
    assert main([
        str(destination), "--device", "cpu", "--max-player-decisions", "2",
        "--lookahead-depth", "2", "--lookahead-top-k", "2", "--lookahead-simulations", "2",
    ]) == 0
    playback = load_playback(destination)
    assert playback["metadata"]["agents"][0] == "information-set-lookahead"


def test_loading_old_observation_format_is_supported(tmp_path):
    from huangbot.model import HuangActorCritic, ModelConfig

    model = HuangActorCritic(ModelConfig(encoder_width=8, encoder_layers=1, recurrent_hidden_size=4))
    checkpoint = tmp_path / "old.pt"
    torch.save({"format_version": 1, "model_config": vars(model.config), "model_state": model.state_dict()}, checkpoint)
    destination = tmp_path / "old-game.json"
    assert main([str(destination), "--checkpoint", str(checkpoint), "--device", "cpu", "--max-player-decisions", "2"]) == 0
    assert load_playback(destination)["metadata"]["player_decisions"] == 2
