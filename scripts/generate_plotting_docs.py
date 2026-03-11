#!/usr/bin/env python3
"""Generate `plotting` documentation with pdoc from the base Conda env."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PDOC_PYTHON = Path("/home/fschmitt/anaconda3/bin/python")


def main() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_plotting_examples.py")],
        cwd=ROOT,
        env=env,
        check=True,
    )
    cmd = [
        str(PDOC_PYTHON),
        "-m",
        "pdoc",
        "-d",
        "numpy",
        "-o",
        str(ROOT / "docs"),
        "plotting",
    ]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    print(f"Wrote {ROOT / 'docs'}")


if __name__ == "__main__":
    main()
