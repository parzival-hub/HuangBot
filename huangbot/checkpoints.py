"""Load pretrained weights for inference."""

from pathlib import Path

import torch

from .model import HuangActorCritic, ModelConfig


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "models" / "best.pt"


def load_checkpoint(path=None, *, device="cpu") -> HuangActorCritic:
    payload = torch.load(
        DEFAULT_CHECKPOINT if path is None else path,
        map_location=device,
        weights_only=True,
    )
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    model = HuangActorCritic(ModelConfig(**payload["model_config"]))
    model.to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model
