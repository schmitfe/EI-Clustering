#!/usr/bin/env python3
"""Generate `spiketools` documentation with pdoc from the base Conda env."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PDOC_PYTHON = Path("/home/fschmitt/anaconda3/bin/python")
SPIKETOOLS_ASSETS = ROOT / "docs" / "spiketools_assets"
SPIKETOOLS_RASTER = SPIKETOOLS_ASSETS / "shared_example_raster.png"


def generate_shared_spike_raster(output_path: Path) -> None:
    sys.path.insert(0, str(ROOT))

    from spiketools.surrogates import gamma_spikes

    rates = np.array([5.6, 6.3, 5.9, 6.5, 5.8, 6.1, 5.7, 6.4, 6.0, 5.5], dtype=float)
    orders = np.array([1, 2, 2, 3, 1, 2, 3, 2, 1, 3], dtype=int)
    spiketimes = gamma_spikes(rates=rates, order=orders, tlim=[0.0, 5000.0], dt=1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    valid = np.isfinite(spiketimes[0]) & np.isfinite(spiketimes[1])
    ax.scatter(
        spiketimes[0, valid],
        spiketimes[1, valid],
        marker="|",
        s=90.0,
        linewidths=0.8,
        color="black",
    )
    ax.set_xlim(0.0, 5000.0)
    ax.set_ylim(-0.5, 9.5)
    ax.set_yticks(np.arange(10))
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Trial")
    ax.set_title("Shared example: 10 gamma-process trials")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    generate_shared_spike_raster(SPIKETOOLS_RASTER)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    cmd = [
        str(PDOC_PYTHON),
        "-m",
        "pdoc",
        "-d",
        "numpy",
        "-o",
        str(ROOT / "docs"),
        "spiketools",
    ]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    print(f"Wrote {ROOT / 'docs'}")


if __name__ == "__main__":
    main()
