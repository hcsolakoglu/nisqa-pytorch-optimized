"""CPU wall-time benchmark of CLI prediction, including model loading.

Run with CUDA_VISIBLE_DEVICES='' python -m benchmarks.benchmark_cli_prediction
--work_dir /path/on/a/spacious/disk --output /path/to/results.json.
Generated audio and caches should live on a disk with sufficient free space.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import statistics
import time
import wave
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_info

from nisqa import DIM_COLUMNS
from run_predict import main

ROOT = Path(__file__).parents[1]


def benchmark(work_dir: Path, count: int, repeats: int) -> dict:
    if torch.version.cuda is not None and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("set CUDA_VISIBLE_DEVICES='' for this CPU benchmark")
    data_dir = work_dir / "audio"
    data_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(216)
    durations = np.linspace(3, 15, count)
    for index, seconds in enumerate(durations):
        t = np.arange(int(48000 * seconds)) / 48000
        samples = 0.15 * np.sin(2 * np.pi * (130 + index) * t) + 0.02 * rng.standard_normal(len(t))
        with wave.open(str(data_dir / f"sample_{index:03}.wav"), "wb") as wav:
            wav.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
            wav.writeframes((samples * 32767).astype("<i2").tobytes())

    base = ["--mode", "predict_dir", "--pretrained_model", str(ROOT / "weights" / "nisqa.tar"),
            "--data_dir", str(data_dir), "--bs", "16"]
    configurations = [("legacy", 0), ("legacy", 2), ("fast", 0), ("fast_uncapped", 0),
                      ("fast", 2), ("fast", 4), ("fast", 8)]
    timings = {config: [] for config in configurations}
    results = {}
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        # Warm imports/JIT and model kernels before timing; no features are cached.
        main(["--mode", "predict_file", "--pretrained_model", str(ROOT / "weights" / "nisqa.tar"),
              "--deg", str(data_dir / "sample_000.wav"), "--num_workers", "0"])
        for repeat in range(repeats):
            # Rotate order to reduce drift from always running legacy first.
            order = configurations[repeat:] + configurations[:repeat]
            for mode, workers in order:
                args = base + ["--num_workers", str(workers)]
                if mode == "legacy":
                    args += ["--legacy_padding"]
                elif mode == "fast_uncapped":
                    args += ["--worker_threads", "0"]
                start = time.perf_counter()
                scores = main(args)
                timings[(mode, workers)].append(time.perf_counter() - start)
                results[(mode, workers)] = scores

    reference = results[("legacy", 0)]
    rows = []
    for (mode, workers), samples in timings.items():
        actual = results[(mode, workers)]
        assert actual.deg.tolist() == reference.deg.tolist()
        values, expected = actual[list(DIM_COLUMNS)].to_numpy(), reference[list(DIM_COLUMNS)].to_numpy()
        np.testing.assert_allclose(values, expected, atol=1e-4, rtol=0)
        rows.append({"mode": mode, "workers": workers, "seconds": samples,
                     "median_seconds": statistics.median(samples),
                     "max_abs_score_difference": float(np.max(np.abs(values - expected)))})
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    return {"cpu": cpu, "cpu_count": os.cpu_count(), "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(), "thread_pools": [
                {k: pool[k] for k in ("user_api", "internal_api", "num_threads")}
                for pool in threadpool_info()
            ], "files": count, "audio_seconds": float(durations.sum()), "sample_rate": 48000,
            "checkpoint": "weights/nisqa.tar", "duration_range_seconds": [3, 15], "seed": 216,
            "batch_size": 16, "repeats": repeats, "includes_model_loading": True, "results": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.count < 2 or args.repeats < 1:
        parser.error("count must be at least 2 and repeats at least 1")
    report = benchmark(args.work_dir, args.count, args.repeats)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["results"]:
        print(f'{row["mode"]:6} workers={row["workers"]}: {row["median_seconds"]:.3f} s (median)')
