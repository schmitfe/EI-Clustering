from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from sim_config import add_override_arguments, load_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repository-integrated population-state analysis on binary or spiking simulation outputs."
    )
    add_override_arguments(parser)
    parser.add_argument("--folder", type=str, help="Existing simulation output folder to analyze.")
    parser.add_argument("--input-file", type=str, help="Explicit .npz simulation output file to analyze.")
    parser.add_argument(
        "--source-type",
        type=str,
        default="auto",
        choices=["auto", "binary", "snn"],
        help="Explicitly select the input adapter.",
    )
    parser.add_argument("--output-dir", type=str, help="Optional explicit output directory for analysis results.")
    return parser.parse_args()


def _resolve_source(args: argparse.Namespace) -> Path:
    if args.input_file:
        return Path(args.input_file)
    if args.folder:
        return Path(args.folder)
    raise ValueError("Provide either --folder or --input-file.")


def main() -> None:
    args = parse_args()
    from analysis.pipeline import run_analysis_from_source

    parameter: Dict[str, Any] = load_from_args(args)
    source = _resolve_source(args)
    analysis_cfg = dict(parameter.get("analysis") or {})
    analysis_cfg["enabled"] = True
    result = run_analysis_from_source(
        source,
        parameter=parameter,
        analysis_cfg=analysis_cfg,
        source_type=args.source_type,
        output_dir=args.output_dir,
    )
    methods = sorted((result.get("results") or {}).keys())
    print(f"Stored analysis outputs at {result['output_dir']}")
    print(f"Completed methods: {', '.join(methods) if methods else 'none'}")


if __name__ == "__main__":
    main()
