from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

import nisqa.NISQA_lib as NISQA_lib
from nisqa import load_model, predict_batch
from nisqa.NISQA_lib import PositionalEncoding, _mel_filter_bank, get_librosa_melspec


EXPECTED = {
    "nisqa_mos_only": [[0.8233922123908997], [0.849410355091095]],
    "nisqa": [
        [0.7895912528038025, 1.7630531787872314, 2.7837040424346924, 1.4422436952590942, 1.5878502130508423],
        [0.7976071834564209, 1.7207328081130981, 2.77801513671875, 1.4494729042053223, 1.5845518112182617],
    ],
    "nisqa_tts": [[1.6888978481292725], [1.6480071544647217]],
}


ROOT = Path(__file__).parents[1]


def _features(name: str, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    steps = 12 if name == "nisqa_tts" else 8
    n_wins = torch.tensor([steps, steps - 3], dtype=torch.int64)
    features = torch.randn((2, steps, 1, 48, 15), generator=generator)
    return features, n_wins


def test_shipped_checkpoints_load_and_preserve_outputs() -> None:
    torch.manual_seed(216)
    generator = torch.Generator(device="cpu").manual_seed(216)
    for name, expected in EXPECTED.items():
        model = load_model(ROOT / "weights" / f"{name}.tar")
        features, n_wins = _features(name, generator)
        output = predict_batch(model, features, n_wins)
        np.testing.assert_allclose(
            output.numpy(),
            np.asarray(expected, dtype=np.float32),
            rtol=0.0,
            atol=5e-5,
        )


def test_packed_sequence_path_accepts_unsorted_lengths() -> None:
    model = load_model(ROOT / "weights" / "nisqa_tts.tar")
    features = torch.randn((2, 8, 1, 48, 15), generator=torch.Generator().manual_seed(216))
    output = predict_batch(model, features, torch.tensor([5, 8]))
    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()


def test_framewise_uses_dense_path_for_full_eval_batches(monkeypatch) -> None:
    model = load_model(ROOT / "weights" / "nisqa_tts.tar")
    features = torch.randn((2, 12, 1, 48, 15), generator=torch.Generator().manual_seed(216))

    def fail_if_packed(*args, **kwargs):
        raise AssertionError("full eval batches should use the dense CNN path")

    monkeypatch.setattr(NISQA_lib, "pack_padded_sequence", fail_if_packed)
    with torch.inference_mode():
        output = model.cnn(features, torch.tensor([12, 12]))
    assert output.shape[:2] == (2, 12)
    assert torch.isfinite(output).all()


def test_tts_uses_dense_lstm_for_all_valid_lengths() -> None:
    model = load_model(ROOT / "weights" / "nisqa_tts.tar")
    features = torch.randn((2, 12, 1, 48, 15), generator=torch.Generator().manual_seed(216))
    seen_inputs: list[object] = []
    handle = model.time_dependency.model.lstm.register_forward_pre_hook(
        lambda _module, args: seen_inputs.append(args[0])
    )
    try:
        output = predict_batch(model, features, torch.tensor([12, 12]))
    finally:
        handle.remove()
    assert output.shape == (2, 1)
    assert len(seen_inputs) == 1
    assert isinstance(seen_inputs[0], torch.Tensor)
    assert not isinstance(seen_inputs[0], PackedSequence)


def test_tts_keeps_packed_lstm_for_mixed_lengths() -> None:
    model = load_model(ROOT / "weights" / "nisqa_tts.tar")
    features = torch.randn((2, 12, 1, 48, 15), generator=torch.Generator().manual_seed(216))
    seen_inputs: list[object] = []
    handle = model.time_dependency.model.lstm.register_forward_pre_hook(
        lambda _module, args: seen_inputs.append(args[0])
    )
    try:
        output = predict_batch(model, features, torch.tensor([12, 8]))
    finally:
        handle.remove()
    assert output.shape == (2, 1)
    assert len(seen_inputs) == 1
    assert isinstance(seen_inputs[0], PackedSequence)


def test_tts_dense_lstm_matches_packed_reference() -> None:
    model = load_model(ROOT / "weights" / "nisqa_tts.tar")
    features = torch.randn((2, 12, 1, 48, 15), generator=torch.Generator().manual_seed(216))
    n_wins = torch.tensor([12, 12])
    with torch.inference_mode():
        actual = predict_batch(model, features, n_wins)
        framewise = model.cnn(features, n_wins)
        packed = pack_padded_sequence(framewise, n_wins, batch_first=True, enforce_sorted=True)
        packed_output = model.time_dependency.model.lstm(packed)[0]
        packed_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=12,
        )
        reference = model.pool(packed_output, n_wins)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)


def test_positional_encoding_is_batch_first() -> None:
    encoding = PositionalEncoding(64).eval()
    output = encoding(torch.zeros(2, 7, 64))
    assert output.shape == (2, 7, 64)


def test_mel_filter_bank_is_cached(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone.wav"
    samples = (0.1 * np.sin(2.0 * np.pi * 440.0 * np.arange(4800) / 48000.0) * 32767).astype("<i2")
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(samples.tobytes())

    _mel_filter_bank.cache_clear()
    first = get_librosa_melspec(
        audio_path,
        sr=48000,
        n_fft=1024,
        hop_length=0.01,
        win_length=0.02,
        n_mels=48,
        fmax=16000,
    )
    second = get_librosa_melspec(
        audio_path,
        sr=48000,
        n_fft=1024,
        hop_length=0.01,
        win_length=0.02,
        n_mels=48,
        fmax=16000,
    )
    np.testing.assert_array_equal(first, second)
    assert _mel_filter_bank.cache_info().hits == 1
