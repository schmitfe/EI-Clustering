#!/usr/bin/env python3
"""Generate one combined pdoc site for the main repository packages."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDOC_PYTHON = Path("/home/fschmitt/anaconda3/bin/python")
PDOC_PYTHON = Path(
    os.environ.get(
        "PDOC_PYTHON",
        str(DEFAULT_PDOC_PYTHON if DEFAULT_PDOC_PYTHON.exists() else Path(sys.executable)),
    )
)
OUTPUT_DIR = ROOT / "docs"
DOC_IMAGE_STYLE_MARKER = "/* codex-doc-image-sizing */"
DOC_IMAGE_STYLE = f"""<style>{DOC_IMAGE_STYLE_MARKER}
main img {{
    display: block;
    max-width: min(100%, 560px);
    max-height: 340px;
    width: auto;
    height: auto;
    margin: 0.75rem auto;
}}
</style>"""
HIDDEN_OUTPUTS = [
    OUTPUT_DIR / "plotting" / "time_axis.html",
    OUTPUT_DIR / "MeanField" / "solver_utils.html",
]
MODULES = [
    "spiketools",
    "plotting",
    "BinaryNetwork",
    "MeanField",
    "sim_config",
    "!BinaryNetwork.Figure_Simulations",
]


def inject_doc_css(output_dir: Path) -> None:
    for html_path in output_dir.rglob("*.html"):
        text = html_path.read_text(encoding="utf-8")
        if DOC_IMAGE_STYLE_MARKER in text:
            continue
        if "</head>" not in text:
            continue
        html_path.write_text(text.replace("</head>", f"{DOC_IMAGE_STYLE}\n</head>", 1), encoding="utf-8")


def main() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_spiketools_docs.py"), "--assets-only"],
        cwd=ROOT,
        env=env,
        check=True,
    )
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
        "--math",
        "-o",
        str(OUTPUT_DIR),
        *MODULES,
    ]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    inject_doc_css(OUTPUT_DIR)
    for path in HIDDEN_OUTPUTS:
        if path.exists():
            path.unlink()
    print(f"Wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
