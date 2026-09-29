from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from nisqa import BatchMaxPadCollate, load_model, predict_batch, predict_dataset

ROOT = Path(__file__).parents[1]
MAX_SEGMENTS = 24


class _Specs(torch.utils.data.Dataset):
    """Unpadded (n_wins, 1, 48, 15) items shaped like SpeechQualityDataset output."""

    def __init__(self, lengths: list[int], seed: int = 7) -> None:
        g = torch.Generator().manual_seed(seed)
        self.specs = [torch.randn((n, 1, 48, 15), generator=g) for n in lengths]

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, i: int):
        return self.specs[i], np.full(5, np.nan, dtype=np.float32), (i, np.array(self.specs[i].shape[0]))


def _static_reference(model, ds: _Specs, batch_size: int) -> np.ndarray:
    """Stock behaviour: every item padded to MAX_SEGMENTS, same batching."""
    out = []
    for start in range(0, len(ds), batch_size):
        chunk = ds.specs[start : start + batch_size]
        x = torch.zeros((len(chunk), MAX_SEGMENTS, 1, 48, 15))
        for row, spec in enumerate(chunk):
            x[row, : spec.shape[0]] = spec
        n_wins = torch.tensor([s.shape[0] for s in chunk])
        out.append(predict_batch(model, x, n_wins).numpy())
    return np.concatenate(out, axis=0)


@pytest.mark.parametrize(
    ("lengths", "batch_size"),
    [
        ([9, 17, 5, 12, 20], 2),  # mixed lengths, tail batch of 1 (dense path in dynamic mode)
        ([11, 11, 11, 11], 4),  # equal lengths: dynamic batch hits the dense path
        ([6, 23, 14], 1),  # batch size one: always dense in dynamic mode
        ([MAX_SEGMENTS, 3, MAX_SEGMENTS], 3),  # longest item equals the stock pad
    ],
)
def test_batch_max_padding_matches_fixed_padding(lengths, batch_size) -> None:
    torch.manual_seed(0)
    model = load_model(ROOT / "weights" / "nisqa.tar")
    ds = _Specs(lengths)
    expected = _static_reference(model, ds, batch_size)
    got = predict_dataset(model, ds, batch_size=batch_size, max_segments=MAX_SEGMENTS)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-5)


def test_thread_capped_workers_match_single_process() -> None:
    pytest.importorskip("threadpoolctl")
    model = load_model(ROOT / "weights" / "nisqa.tar")
    ds = _Specs([9, 17, 5, 12, 20, 8])
    single = predict_dataset(model, ds, batch_size=2)
    workers = predict_dataset(model, ds, batch_size=2, num_workers=2, worker_threads=1)
    np.testing.assert_allclose(workers, single, rtol=0.0, atol=1e-6)


def test_collate_pads_and_keeps_stock_guard() -> None:
    ds = _Specs([3, 7])
    x, y, (index, n_wins) = BatchMaxPadCollate(10)([ds[0], ds[1]])
    assert x.shape == (2, 7, 1, 48, 15)
    assert torch.equal(x[0, :3], ds.specs[0]) and torch.count_nonzero(x[0, 3:]) == 0
    assert n_wins.tolist() == [3, 7] and index.tolist() == [0, 1] and y.shape == (2, 5)
    with pytest.raises(ValueError, match="max_length"):
        BatchMaxPadCollate(5)([ds[1]])
