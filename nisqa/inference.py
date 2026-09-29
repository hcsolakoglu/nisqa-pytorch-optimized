"""Modern inference helpers for pretrained NISQA checkpoints."""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data

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


# Column order of NISQA_lib.predict_dim for NISQA_DIM outputs.
DIM_COLUMNS = ("mos_pred", "noi_pred", "dis_pred", "col_pred", "loud_pred")


class BatchMaxPadCollate:
    """Collate unpadded ``SpeechQualityDataset`` items, padding to the batch maximum.

    Build the dataset with ``max_length=None`` so ``segment_specs`` skips the
    fixed ``ms_max_segments`` padding. The models pack by ``n_wins`` and unpack
    to ``n_wins.max()``, so padding beyond the batch maximum never reaches them;
    this avoids allocating and moving ``ms_max_segments`` windows per file.
    ``max_segments`` keeps the stock "Increase max window length" guard.
    A class rather than a closure so DataLoader workers can pickle it.
    """

    def __init__(self, max_segments: int | None = None) -> None:
        self.max_segments = max_segments

    def __call__(self, items: list[Any]) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        n_wins = [int(item[2][1]) for item in items]
        longest = max(n_wins)
        if self.max_segments is not None and longest > self.max_segments:
            raise ValueError(f"n_wins {longest} > max_length {self.max_segments}")
        first = torch.as_tensor(items[0][0])
        x = torch.zeros((len(items), longest, *first.shape[1:]), dtype=first.dtype)
        for row, (spec, _y, _meta) in enumerate(items):
            spec = torch.as_tensor(spec)
            x[row, : spec.shape[0]] = spec
        y = torch.stack([torch.as_tensor(item[1]) for item in items])
        index = torch.tensor([int(item[2][0]) for item in items])
        return x, y, (index, torch.tensor(n_wins))


class LimitWorkerThreads:
    """DataLoader ``worker_init_fn`` capping BLAS/OpenMP pools per worker.

    Forked workers inherit host-sized thread pools, so N workers running
    librosa FFT/mel matmuls oversubscribe the CPU; on a 48-vCPU host this made
    extra workers useless. Requires ``threadpoolctl`` (a scikit-learn, hence
    librosa, dependency).
    """

    def __init__(self, threads: int = 1) -> None:
        self.threads = threads

    def __call__(self, _worker_id: int) -> None:
        from threadpoolctl import threadpool_limits

        global _WORKER_THREAD_LIMIT  # keep the limiter alive for the worker's lifetime
        _WORKER_THREAD_LIMIT = threadpool_limits(self.threads)


_WORKER_THREAD_LIMIT: Any = None


def predict_dataset(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    device: str | torch.device = "cpu",
    num_workers: int = 0,
    max_segments: int | None = None,
    worker_threads: int = 1,
) -> np.ndarray:
    """Score an unpadded dataset with batch-max padding; rows follow dataset order.

    Equivalent to ``NISQA_lib.predict_mos``/``predict_dim`` on the same
    features (output columns in model order, see ``DIM_COLUMNS``), within
    float reduction-order differences. ``worker_threads`` caps worker pools,
    or just BLAS in the calling process when ``num_workers == 0``; the latter
    leaves PyTorch's CPU parallelism intact. Limits are restored on return.
    Use 0 to keep library thread defaults.
    """
    pin_memory = torch.device(device).type == "cuda"
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn=BatchMaxPadCollate(max_segments),
        worker_init_fn=LimitWorkerThreads(worker_threads) if worker_threads > 0 and num_workers > 0 else None,
    )
    model.to(device).eval()
    limits = nullcontext()
    if num_workers == 0 and worker_threads > 0:
        from threadpoolctl import threadpool_limits

        limits = threadpool_limits(worker_threads, user_api="blas")
    with limits:
        outputs = [
            predict_batch(model, x.to(device, non_blocking=pin_memory), n_wins).cpu().numpy()
            for x, _y, (_idx, n_wins) in loader
        ]
    return np.concatenate(outputs, axis=0)
