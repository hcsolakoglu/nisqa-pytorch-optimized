from __future__ import annotations

import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nisqa import DIM_COLUMNS
from nisqa.NISQA_model import nisqaModel
from run_predict import main

ROOT = Path(__file__).parents[1]


@pytest.fixture
def audio_files(tmp_path):
    rng = np.random.default_rng(216)
    names = []
    for index, seconds in enumerate((0.65, 0.9, 1.2)):
        name = f"sample_{index}.wav"
        t = np.arange(int(48000 * seconds)) / 48000
        samples = 0.2 * np.sin(2 * np.pi * (180 + 30 * index) * t) + 0.01 * rng.standard_normal(len(t))
        with wave.open(str(tmp_path / name), "wb") as wav:
            wav.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
            wav.writeframes((samples * 32767).astype("<i2").tobytes())
        names.append(name)
    pd.DataFrame({"input_audio": [names[2], names[0], names[1]], "row_id": [7, 2, 9]}).to_csv(
        tmp_path / "files.csv", index=False
    )
    return tmp_path, names


def _arguments(audio_files, checkpoint, mode, batch_size):
    directory, names = audio_files
    args = ["--mode", mode, "--pretrained_model", str(ROOT / "weights" / f"{checkpoint}.tar"),
            "--output_dir", str(directory), "--bs", str(batch_size)]
    if mode == "predict_file":
        args += ["--deg", str(directory / names[0])]
    else:
        args += ["--data_dir", str(directory)]
        if mode == "predict_csv":
            args += ["--csv_file", "files.csv", "--csv_deg", "input_audio"]
    return args


@pytest.mark.parametrize("checkpoint", ("nisqa", "nisqa_mos_only", "nisqa_tts"))
@pytest.mark.parametrize("mode", ("predict_file", "predict_dir", "predict_csv"))
@pytest.mark.parametrize("batch_size", (1, 2))
def test_cli_prediction_matches_legacy(audio_files, checkpoint, mode, batch_size, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    args = _arguments(audio_files, checkpoint, mode, batch_size)
    expected = main(args + ["--num_workers", "0", "--legacy_padding"])
    columns = list(DIM_COLUMNS) if checkpoint == "nisqa" else ["mos_pred"]
    metadata = [column for column in expected if column not in columns]
    import nisqa.NISQA_model as module

    datasets = []
    real_predict = module.predict_dataset

    def record_dataset(model, dataset, **kwargs):
        datasets.append(dataset)
        assert dataset.max_length is None
        return real_predict(model, dataset, **kwargs)

    monkeypatch.setattr(module, "predict_dataset", record_dataset)
    single = main(args + ["--num_workers", "0"])
    workers = main(args + ["--num_workers", "2", "--worker_threads", "1"])
    assert len(datasets) == 2
    for actual in (single, workers):
        assert actual.columns.tolist() == expected.columns.tolist()
        pd.testing.assert_frame_equal(actual[metadata], expected[metadata])
        np.testing.assert_allclose(actual[columns], expected[columns], rtol=0, atol=1e-4)
    np.testing.assert_allclose(workers[columns], single[columns], rtol=0, atol=1e-5)
    saved = pd.read_csv(audio_files[0] / "NISQA_results.csv")
    pd.testing.assert_frame_equal(saved, workers, check_dtype=False, atol=1e-7, rtol=0)
    if mode == "predict_csv":
        assert workers.row_id.tolist() == [7, 2, 9]
    assert workers.columns[-len(columns) - 1:].tolist() == columns + ["model"]


def test_cli_subprocess_with_workers(audio_files):
    args = _arguments(audio_files, "nisqa", "predict_csv", 2)
    expected = main(args + ["--num_workers", "0", "--legacy_padding"])
    result = subprocess.run(
        [sys.executable, str(ROOT / "run_predict.py"), *args, "--num_workers", "2"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    actual = pd.read_csv(audio_files[0] / "NISQA_results.csv")
    np.testing.assert_allclose(actual[list(DIM_COLUMNS)], expected[list(DIM_COLUMNS)], rtol=0, atol=1e-4)


@pytest.mark.parametrize("cpu_count,expected", ((None, 1), (2, 2), (48, 8)))
def test_prediction_worker_defaults(monkeypatch, cpu_count, expected):
    monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)
    model = nisqaModel.__new__(nisqaModel)
    model.args = {"mode": "predict_csv", "double_ended": False, "ms_seg_length": 15, "ms_max_segments": 1300}
    model._configurePrediction(None)
    assert model.args["tr_num_workers"] == expected
    assert model._prediction_max_length is None
    model.args["mode"] = "predict_file"
    model._configurePrediction(None)
    assert model.args["tr_num_workers"] == 0
    model._configurePrediction(3)
    assert model.args["tr_num_workers"] == 3


@pytest.mark.parametrize("fallback", ({"legacy_padding": True}, {"double_ended": True}, {"ms_seg_length": None}))
def test_prediction_falls_back_to_legacy(fallback, monkeypatch):
    model = nisqaModel.__new__(nisqaModel)
    model.args = {"mode": "predict_csv", "double_ended": False, "ms_seg_length": 15, "ms_max_segments": 1300,
                  "dim": False, "tr_parallel": False, "tr_bs_val": 2, "output_dir": None, **fallback}
    model._configurePrediction(None)
    assert not model._fast_prediction
    assert model._prediction_max_length == 1300
    assert model.args["tr_num_workers"] == 0
    model.model, model.dev = object(), "cpu"
    model.ds_val = type("Dataset", (), {"df": pd.DataFrame({"deg": ["sample.wav"]})})()
    calls = []

    def legacy_predict(*args, **kwargs):
        calls.append(kwargs)
        model.ds_val.df["mos_pred"] = 3.0
        return np.array([[3.0]]), np.array([[np.nan]])

    monkeypatch.setattr("nisqa.NISQA_model.NL.predict_mos", legacy_predict)
    assert model.predict().mos_pred.tolist() == [3.0]
    assert calls == [{"num_workers": 0}]


def test_worker_threads_zero_keeps_fast_padding(audio_files, monkeypatch):
    calls = []
    import nisqa.NISQA_model as module

    real_predict = module.predict_dataset

    def record_threads(model, dataset, **kwargs):
        calls.append(kwargs["worker_threads"])
        assert dataset.max_length is None
        return real_predict(model, dataset, **kwargs)

    monkeypatch.setattr(module, "predict_dataset", record_threads)
    main(_arguments(audio_files, "nisqa", "predict_file", 1) + ["--worker_threads", "0"])
    assert calls == [0]


@pytest.mark.parametrize("workers,threads", ((-1, 1), (0, -1)))
def test_prediction_rejects_negative_worker_settings(workers, threads):
    model = nisqaModel.__new__(nisqaModel)
    model.args = {"mode": "predict_csv", "double_ended": False, "ms_seg_length": 15, "ms_max_segments": 1300,
                  "worker_threads": threads}
    with pytest.raises(ValueError, match="non-negative"):
        model._configurePrediction(workers)
