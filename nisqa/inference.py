"""Modern inference helpers for pretrained NISQA checkpoints."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch

from . import NISQA_lib as NL


_MODEL_CLASSES = {
    "NISQA": NL.NISQA,
    "NISQA_DIM": NL.NISQA_DIM,
}


def _model_kwargs(model_class: type[torch.nn.Module], args: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(model_class).parameters
    return {name: args[name] for name in parameters if name in args}


def load_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Load a shipped NISQA checkpoint for inference.

    Checkpoints distributed with this repository contain tensors and primitive
    configuration values, so the restricted ``weights_only`` loader avoids
    executing arbitrary pickle globals. The returned model is in evaluation
    mode and ready for ``predict_batch``.
    """
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )
    try:
        args = checkpoint["args"]
        model_state_dict = checkpoint["model_state_dict"]
        model_class = _MODEL_CLASSES[args["model"]]
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid NISQA checkpoint structure") from exc

    model = model_class(**_model_kwargs(model_class, args))
    model.load_state_dict(model_state_dict, strict=True)
    return model.to(device).eval()


def predict_batch(
    model: torch.nn.Module,
    features: torch.Tensor,
    n_wins: torch.Tensor,
) -> torch.Tensor:
    """Run one inference batch without constructing autograd state."""
    with torch.inference_mode():
        return model(features, n_wins)
