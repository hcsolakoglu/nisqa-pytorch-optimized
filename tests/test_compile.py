from pathlib import Path

import pytest
import torch

from nisqa import load_model


ROOT = Path(__file__).parents[1]


@pytest.mark.compile
def test_torch_compile_fixed_shape_preserves_inference_output() -> None:
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")

    model = load_model(ROOT / "weights" / "nisqa_mos_only.tar")
    features = torch.randn(
        (2, 8, 1, 48, 15),
        generator=torch.Generator().manual_seed(216),
    )
    n_wins = torch.tensor([8, 8])
    compiled = torch.compile(model, backend="eager")

    with torch.inference_mode():
        expected = model(features, n_wins)
        actual = compiled(features, n_wins)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=5e-5)
