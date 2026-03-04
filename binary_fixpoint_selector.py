from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence

from binary_pipeline import (
    available_rep_values,
    focus_rep_grid,
    load_fixpoint_summary,
    simulate_fixpoint_reps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive helper to select all_fixpoints bundles and run binary simulations."
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root folder to search for all_fixpoints_*.pkl files (default: %(default)s).",
    )
    parser.add_argument(
        "--pattern",
        default="all_fixpoints_*.pkl",
        help="Glob pattern (relative to the root) used to identify fixpoint bundles.",
    )
    parser.add_argument("--warmup-steps", type=int, help="Override binary.warmup_steps from the metadata.")
    parser.add_argument("--simulation-steps", type=int, help="Override binary.simulation_steps from the metadata.")
    parser.add_argument("--sample-interval", type=int, help="Override binary.sample_interval from the metadata.")
    parser.add_argument("--batch-size", type=int, help="Override binary.batch_size from the metadata.")
    parser.add_argument("--seed", type=int, help="Random seed for numpy.")
    parser.add_argument(
        "--output-name",
        type=str,
        help="Base output name for traces (defaults to the metadata or 'activity_trace').",
    )
    parser.add_argument(
        "--plot-activity",
        action="store_true",
        help="Also render population activity plots for each simulation.",
    )
    return parser.parse_args()


def discover_fixpoint_files(root: str, pattern: str) -> List[str]:
    search_root = Path(root)
    if not search_root.exists():
        return []
    files = sorted(str(path) for path in search_root.rglob(pattern) if path.is_file())
    return files


def prompt_file_selection(files: Sequence[str]) -> List[int]:
    while True:
        raw = input("Select fixpoint files by index ('all' for every file, 'q' to quit): ").strip().lower()
        if not raw:
            continue
        if raw in {"q", "quit"}:
            return []
        if raw in {"all", "a"}:
            return list(range(len(files)))
        tokens = [token.strip() for token in raw.replace(";", ",").split(",") if token.strip()]
        try:
            selection = sorted({int(token) for token in tokens})
        except ValueError:
            print("Invalid selection. Please enter integers separated by commas.")
            continue
        out_of_range = [idx for idx in selection if idx < 0 or idx >= len(files)]
        if out_of_range:
            print(f"Indices {out_of_range} are out of range. Try again.")
            continue
        if selection:
            return selection
        print("No indices selected. Try again.")


def prompt_rep_selection(available: Sequence[float]) -> List[float]:
    if not available:
        return []
    prompt = (
        "Enter rep values (comma separated), 'all' for every rep, 'skip' to skip this file: "
    )
    while True:
        raw = input(prompt).strip().lower()
        if not raw or raw in {"all", "a"}:
            return list(available)
        if raw in {"skip", "s"}:
            return []
        tokens = [token.strip() for token in raw.replace(";", ",").split(",") if token.strip()]
        try:
            reps = [float(token) for token in tokens]
        except ValueError:
            print("Invalid rep selection. Please enter numeric values.")
            continue
        return reps


def print_rep_grid(grid: dict[int, List[float]]) -> None:
    if not grid:
        print("No focus_count grid available.")
        return
    print("Available rep grid (focus_count → R_Eplus values):")
    for focus_count in sorted(grid):
        values = grid[focus_count]
        joined = ", ".join(f"{value:g}" for value in values) if values else "—"
        print(f"  focus {focus_count}: {joined}")


def main() -> None:
    args = parse_args()
    files = discover_fixpoint_files(args.data_root, args.pattern)
    if not files:
        print(f"No fixpoint files matching '{args.pattern}' were found in {args.data_root}.")
        return
    overrides = SimpleNamespace(
        warmup_steps=args.warmup_steps,
        simulation_steps=args.simulation_steps,
        sample_interval=args.sample_interval,
        batch_size=args.batch_size,
        seed=args.seed,
        output_name=args.output_name,
        plot_activity=args.plot_activity,
    )
    while True:
        print("\nAvailable fixpoint bundles:")
        for idx, path in enumerate(files):
            print(f"  [{idx}] {path}")
        selection = prompt_file_selection(files)
        if not selection:
            print("No files selected. Exiting.")
            return
        for idx in selection:
            path = files[idx]
            print(f"\nLoading fixpoint bundle {path}")
            try:
                summary = load_fixpoint_summary(path)
            except ValueError as exc:
                print(f"  Skipping: {exc}")
                continue
            grid = focus_rep_grid(summary.get("fixpoints", {}))
            print_rep_grid(grid)
            available_reps = available_rep_values(summary.get("fixpoints", {}))
            if not available_reps:
                print("  No reps available in this file. Skipping.")
                continue
            reps = prompt_rep_selection(available_reps)
            if not reps:
                print("  Skipping this file.")
                continue
            try:
                simulate_fixpoint_reps(path, reps, overrides)
            except ValueError as exc:
                print(f"  Simulation aborted: {exc}")
        continue_prompt = input("\nProcess more fixpoint files? [y/N]: ").strip().lower()
        if continue_prompt not in {"y", "yes"}:
            break


if __name__ == "__main__":
    main()
