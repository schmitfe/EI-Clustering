#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figure_cli import resolve_float_values


DEFAULT_KAPPAS = ("0", "0.25", "0.5", "0.75", "1.0")
DEFAULT_OUTPUT_PREFIX = "Figures/Figure4"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kappas: List[float]
    mean_connectivity: List[float] | None
    output_prefix: str


def _format_float(value: float) -> str:
    return format(float(value), ".15g")


def _chunked(values: Sequence[float], chunk_size: int) -> List[List[float]]:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")
    return [list(values[idx: idx + chunk_size]) for idx in range(0, len(values), chunk_size)]


def _slug(prefix: str, values: Sequence[float]) -> str:
    encoded = "-".join(_format_float(value).replace(".", "p") for value in values)
    return f"{prefix}{encoded}"


def _build_task_specs(
    *,
    kappas: Sequence[float],
    mean_connectivity: Sequence[float] | None,
    kappas_per_job: int,
    connectivities_per_job: int,
    partial_dir: Path,
) -> List[TaskSpec]:
    specs: List[TaskSpec] = []
    kappa_chunks = _chunked(kappas, kappas_per_job)
    connectivity_chunks = _chunked(mean_connectivity, connectivities_per_job) if mean_connectivity else [None]
    for conn_chunk in connectivity_chunks:
        for kappa_chunk in kappa_chunks:
            name = _slug("kappa", kappa_chunk)
            if conn_chunk:
                name = f"{_slug('conn', conn_chunk)}__{name}"
            specs.append(
                TaskSpec(
                    name=name,
                    kappas=list(kappa_chunk),
                    mean_connectivity=list(conn_chunk) if conn_chunk else None,
                    output_prefix=str((partial_dir / name).as_posix()),
                )
            )
    return specs


def _shell_array(values: Sequence[str]) -> str:
    return "(" + " ".join(shlex.quote(value) for value in values) + ")"


def _extract_figure_args(figure_args: Sequence[str]) -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--kappas", nargs="+", default=list(DEFAULT_KAPPAS))
    parser.add_argument("--mean-connectivity", nargs="+")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--jobs")
    parser.add_argument("--analysis-only", action="store_true")
    known, remaining = parser.parse_known_args(list(figure_args))
    return known, list(remaining)


def _build_figure_command(
    figure_args: Sequence[str],
    *,
    jobs: int,
    analysis_only: bool,
    output_prefix: str | None,
    kappas: Sequence[float] | None = None,
    mean_connectivity: Sequence[float] | None = None,
) -> List[str]:
    cmd: List[str] = ["python", "Figure4.py", *figure_args, "--jobs", str(int(jobs))]
    if kappas:
        cmd.extend(["--kappas", *[_format_float(value) for value in kappas]])
    if mean_connectivity:
        cmd.extend(["--mean-connectivity", *[_format_float(value) for value in mean_connectivity]])
    if analysis_only:
        cmd.append("--analysis-only")
    if output_prefix:
        cmd.extend(["--output-prefix", str(output_prefix)])
    return cmd


def _write_tasks_file(path: Path, tasks: Sequence[TaskSpec]) -> None:
    lines = []
    for task in tasks:
        lines.append(
            "\t".join(
                (
                    task.name,
                    ",".join(_format_float(value) for value in task.kappas),
                    "" if not task.mean_connectivity else ",".join(_format_float(value) for value in task.mean_connectivity),
                    task.output_prefix,
                )
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_worker_script(path: Path, *, common_args: Sequence[str]) -> None:
    script = f"""#!/bin/bash -l
set -euo pipefail

TASK_FILE="$1"
REPO_ROOT="$2"
TASK_ID="${{SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}}"
TASK_LINE="$(sed -n "$((TASK_ID + 1))p" "$TASK_FILE")"
if [[ -z "$TASK_LINE" ]]; then
  echo "No task line found for array index $TASK_ID" >&2
  exit 1
fi

IFS=$'\\t' read -r TASK_NAME KAPPA_CSV CONN_CSV TASK_OUTPUT <<<"$TASK_LINE"
IFS=',' read -r -a KAPPA_VALUES <<<"$KAPPA_CSV"
if [[ -n "$CONN_CSV" ]]; then
  IFS=',' read -r -a CONN_VALUES <<<"$CONN_CSV"
else
  CONN_VALUES=()
fi

module load python3/miniforge3-24.3-py3.12
eval "$(conda shell.bash hook)"
conda activate ei-cluster

cd "$REPO_ROOT"

COMMON_ARGS={_shell_array(list(common_args))}
CMD=("${{COMMON_ARGS[@]}}" --kappas "${{KAPPA_VALUES[@]}}")
if (( ${{#CONN_VALUES[@]}} > 0 )); then
  CMD+=(--mean-connectivity "${{CONN_VALUES[@]}}")
fi
CMD+=(--output-prefix "$TASK_OUTPUT")

echo "[task=$TASK_NAME] starting on $(hostname)"
printf '[task=%s] command:' "$TASK_NAME"
printf ' %q' "${{CMD[@]}}"
printf '\\n'
"${{CMD[@]}}"
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def _write_final_script(path: Path, *, repo_root: Path, final_args: Sequence[str]) -> None:
    script = f"""#!/bin/bash -l
set -euo pipefail

module load python3/miniforge3-24.3-py3.12
eval "$(conda shell.bash hook)"
conda activate ei-cluster

cd "{repo_root.as_posix()}"

CMD={_shell_array(list(final_args))}
echo "[final] starting on $(hostname)"
printf '[final] command:'
printf ' %q' "${{CMD[@]}}"
printf '\\n'
"${{CMD[@]}}"
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def _sbatch_command(
    *,
    script_path: Path,
    partition: str,
    time_limit: str,
    cpus_per_task: int,
    nodes: int,
    log_dir: Path,
    job_name: str,
    array: str | None = None,
    dependency: str | None = None,
    script_args: Sequence[str] = (),
    mem: str | None = None,
    nodelist: str | None = None,
) -> List[str]:
    cmd = [
        "sbatch",
        "--parsable",
        "--partition",
        partition,
        "--time",
        time_limit,
        "--nodes",
        str(int(nodes)),
        "--cpus-per-task",
        str(int(cpus_per_task)),
        "--job-name",
        job_name,
        "--output",
        str((log_dir / f"{job_name}_%A_%a.out").as_posix()) if array else str((log_dir / f"{job_name}_%j.out").as_posix()),
        "--error",
        str((log_dir / f"{job_name}_%A_%a.err").as_posix()) if array else str((log_dir / f"{job_name}_%j.err").as_posix()),
    ]
    if array:
        cmd.extend(["--array", array])
    if dependency:
        cmd.extend(["--dependency", dependency])
    if mem:
        cmd.extend(["--mem", mem])
    if nodelist:
        cmd.extend(["--nodelist", nodelist])
    cmd.append(str(script_path.as_posix()))
    cmd.extend(str(arg) for arg in script_args)
    return cmd


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Submit Figure 4 as a Slurm job array over kappa/connectivity chunks. "
            "Pass the normal Figure4.py arguments directly and add wrapper flags via --slurm-*."
        )
    )
    parser.add_argument("--slurm-partition", default="smp,gpu")
    parser.add_argument("--slurm-time", default="00:20:00")
    parser.add_argument("--slurm-nodes", type=int, default=1)
    parser.add_argument("--slurm-nodelist")
    parser.add_argument("--slurm-cpus-per-task", type=int, default=20)
    parser.add_argument("--slurm-mem")
    parser.add_argument("--slurm-array-parallelism", type=int, default=0)
    parser.add_argument("--slurm-final-partition")
    parser.add_argument("--slurm-final-time")
    parser.add_argument("--slurm-final-nodes", type=int)
    parser.add_argument("--slurm-final-nodelist")
    parser.add_argument("--slurm-final-cpus-per-task", type=int)
    parser.add_argument("--slurm-final-mem")
    parser.add_argument("--slurm-final-jobs", type=int)
    parser.add_argument("--slurm-job-name-prefix", default="figure4")
    parser.add_argument("--slurm-staging-root", default="slurm/figure4")
    parser.add_argument("--slurm-partial-output-root", default="Figures/Figure4_slurm_parts")
    parser.add_argument("--slurm-kappas-per-job", type=int, default=1)
    parser.add_argument("--slurm-connectivities-per-job", type=int, default=1)
    parser.add_argument("--slurm-no-final-job", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, figure_args = parser.parse_known_args(argv)
    return args, figure_args


def main() -> int:
    args, figure_args = parse_args()
    repo_root = REPO_ROOT
    figure_known, figure_common_args = _extract_figure_args(figure_args)
    kappas = list(resolve_float_values(figure_known.kappas, option_name="--kappas", default=tuple(float(value) for value in DEFAULT_KAPPAS)) or [])
    mean_connectivity = None
    if figure_known.mean_connectivity:
        mean_connectivity = list(resolve_float_values(figure_known.mean_connectivity, option_name="--mean-connectivity") or [])

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_dir = (repo_root / args.slurm_staging_root / run_stamp).resolve()
    logs_dir = staging_dir / "logs"
    partial_dir = (repo_root / args.slurm_partial_output_root / run_stamp).resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)

    tasks = _build_task_specs(
        kappas=kappas,
        mean_connectivity=mean_connectivity,
        kappas_per_job=int(args.slurm_kappas_per_job),
        connectivities_per_job=int(args.slurm_connectivities_per_job),
        partial_dir=partial_dir,
    )
    if not tasks:
        raise SystemExit("No task chunks were generated.")

    worker_jobs = max(1, int(args.slurm_cpus_per_task))
    common_args = _build_figure_command(
        figure_common_args,
        jobs=worker_jobs,
        analysis_only=False,
        output_prefix=None,
    )
    tasks_path = staging_dir / "tasks.tsv"
    worker_script = staging_dir / "worker.sbatch"
    final_script = staging_dir / "final.sbatch"
    _write_tasks_file(tasks_path, tasks)
    _write_worker_script(worker_script, common_args=common_args)

    array_spec = f"0-{len(tasks) - 1}"
    array_parallelism = int(args.slurm_array_parallelism)
    if array_parallelism > 0:
        array_spec = f"{array_spec}%{array_parallelism}"

    worker_job_name = f"{args.slurm_job_name_prefix}_work"
    worker_cmd = _sbatch_command(
        script_path=worker_script,
        partition=str(args.slurm_partition),
        time_limit=str(args.slurm_time),
        nodes=int(args.slurm_nodes),
        cpus_per_task=int(args.slurm_cpus_per_task),
        log_dir=logs_dir,
        job_name=worker_job_name,
        array=array_spec,
        mem=args.slurm_mem,
        nodelist=args.slurm_nodelist,
        script_args=(str(tasks_path), str(repo_root)),
    )

    final_partition = args.slurm_final_partition or args.slurm_partition
    final_time = args.slurm_final_time or args.slurm_time
    final_nodes = int(args.slurm_final_nodes or args.slurm_nodes)
    final_cpus = int(args.slurm_final_cpus_per_task or args.slurm_cpus_per_task)
    final_jobs = int(args.slurm_final_jobs or final_cpus)
    final_nodelist = args.slurm_final_nodelist or args.slurm_nodelist
    final_args = _build_figure_command(
        figure_common_args,
        jobs=final_jobs,
        analysis_only=True,
        output_prefix=str(figure_known.output_prefix or DEFAULT_OUTPUT_PREFIX),
        kappas=kappas,
        mean_connectivity=mean_connectivity,
    )
    _write_final_script(final_script, repo_root=repo_root, final_args=final_args)

    if args.dry_run:
        print(f"Staging directory: {staging_dir}")
        print(f"Task count: {len(tasks)}")
        print("Figure4 passthrough arguments:")
        print("  " + shlex.join(["python", "Figure4.py", *figure_args]) if figure_args else "  python Figure4.py")
        for index, task in enumerate(tasks):
            conn_text = "base-config" if not task.mean_connectivity else ",".join(_format_float(v) for v in task.mean_connectivity)
            print(f"[{index:02d}] {task.name}: kappas={','.join(_format_float(v) for v in task.kappas)} mean_connectivity={conn_text}")
        print("Worker submit command:")
        print("  " + shlex.join(worker_cmd))
        if not args.slurm_no_final_job:
            final_cmd = _sbatch_command(
                script_path=final_script,
                partition=str(final_partition),
                time_limit=str(final_time),
                nodes=final_nodes,
                cpus_per_task=final_cpus,
                log_dir=logs_dir,
                job_name=f"{args.slurm_job_name_prefix}_final",
                mem=args.slurm_final_mem,
                nodelist=final_nodelist,
            )
            print("Final submit command:")
            print("  " + shlex.join(final_cmd))
        return 0

    worker_proc = subprocess.run(worker_cmd, check=True, capture_output=True, text=True)
    worker_job_id = worker_proc.stdout.strip()
    print(f"Submitted worker array job {worker_job_id}")

    if args.slurm_no_final_job:
        return 0

    final_cmd = _sbatch_command(
        script_path=final_script,
        partition=str(final_partition),
        time_limit=str(final_time),
        nodes=final_nodes,
        cpus_per_task=final_cpus,
        log_dir=logs_dir,
        job_name=f"{args.slurm_job_name_prefix}_final",
        dependency=f"afterok:{worker_job_id}",
        mem=args.slurm_final_mem,
        nodelist=final_nodelist,
    )
    final_proc = subprocess.run(final_cmd, check=True, capture_output=True, text=True)
    final_job_id = final_proc.stdout.strip()
    print(f"Submitted final aggregation job {final_job_id} (depends on {worker_job_id})")
    print(f"Staging directory: {staging_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
