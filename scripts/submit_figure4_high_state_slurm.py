#!/usr/bin/env python3
"""Submit the staged Figure4_HighState workflow to Slurm."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figure_cli import resolve_float_values


def _extract(args: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--kappas", nargs="+", default=["0", "0.25", "0.5", "0.75", "1"])
    parser.add_argument("--mean-connectivity", nargs="+")
    parser.add_argument("--n-networks", type=int, default=15)
    parser.add_argument("--n-inits", type=int, default=3)
    parser.add_argument("--output-dir", default="plots/Figure4_HighState")
    parser.add_argument("--output-prefix", default="Figures/Figure4_HighState")
    parser.add_argument("--jobs")
    known, remaining = parser.parse_known_args(list(args))
    return known, remaining


def _shell_array(values: Sequence[str]) -> str:
    return "(" + " ".join(shlex.quote(value) for value in values) + ")"


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _worker_script(
    path: Path,
    common: Sequence[str],
    *,
    prepare: bool,
    data_root: str | None,
) -> None:
    mode = "--prepare-only" if prepare else "--use-existing-bundle --no-aggregate"
    task_fields = "CONN KAPPA" if prepare else "CONN KAPPA NETWORK INIT"
    extra = "" if prepare else ' --network-indices "$NETWORK" --init-indices "$INIT"'
    text = f'''#!/bin/bash -l
set -euo pipefail
TASK_FILE="$1"
REPO_ROOT="$2"
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")"
read -r {task_fields} <<<"$LINE"
module load python3/miniforge3-24.3-py3.12
eval "$(conda shell.bash hook)"
conda activate ei-cluster
cd "$REPO_ROOT"
{f'export EI_CLUSTER_DATA_ROOT={shlex.quote(data_root)}' if data_root else ''}
CMD={_shell_array(common)}
CMD+=(--kappas "$KAPPA" {mode}{extra})
if [[ "$CONN" != "BASE" ]]; then CMD+=(--mean-connectivity "$CONN"); fi
printf 'command:'; printf ' %q' "${{CMD[@]}}"; printf '\n'
"${{CMD[@]}}"
'''
    _write(path, text)


def _final_script(path: Path, command: Sequence[str], repo: Path, *, data_root: str | None) -> None:
    _write(
        path,
        f'''#!/bin/bash -l
set -euo pipefail
module load python3/miniforge3-24.3-py3.12
eval "$(conda shell.bash hook)"
conda activate ei-cluster
cd {shlex.quote(str(repo))}
{f'export EI_CLUSTER_DATA_ROOT={shlex.quote(data_root)}' if data_root else ''}
CMD={_shell_array(command)}
"${{CMD[@]}}"
''',
    )


def _sbatch(
    script: Path,
    *,
    partition: str,
    time: str,
    cpus: int,
    log_dir: Path,
    name: str,
    array: str | None = None,
    dependency: str | None = None,
    mem: str | None = None,
    nodelist: str | None = None,
    args: Sequence[str] = (),
) -> list[str]:
    command = [
        "sbatch", "--parsable", "--partition", partition, "--time", time,
        "--nodes", "1", "--cpus-per-task", str(cpus), "--job-name", name,
        "--output", str(log_dir / (f"{name}_%A_%a.out" if array else f"{name}_%j.out")),
        "--error", str(log_dir / (f"{name}_%A_%a.err" if array else f"{name}_%j.err")),
    ]
    if array:
        command += ["--array", array]
    if dependency:
        command += ["--dependency", dependency]
    if mem:
        command += ["--mem", mem]
    if nodelist:
        command += ["--nodelist", nodelist]
    return command + [str(script), *map(str, args)]


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slurm-partition", default="smp,gpu")
    parser.add_argument("--slurm-nodelist")
    parser.add_argument("--slurm-prep-time", default="00:30:00")
    parser.add_argument("--slurm-worker-time", default="24:00:00")
    parser.add_argument("--slurm-final-time", default="00:30:00")
    parser.add_argument("--slurm-cpus-per-task", type=int, default=1)
    parser.add_argument("--slurm-worker-mem", default="24G")
    parser.add_argument("--slurm-prep-mem", default="8G")
    parser.add_argument("--slurm-final-mem", default="8G")
    parser.add_argument("--slurm-array-parallelism", type=int, default=0)
    parser.add_argument("--slurm-job-name-prefix", default="figure4_high")
    parser.add_argument("--slurm-staging-root", default="slurm/figure4_high_state")
    parser.add_argument(
        "--slurm-data-root",
        help="Shared cache root exported as EI_CLUSTER_DATA_ROOT in every Slurm stage.",
    )
    parser.add_argument("--slurm-no-final-job", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    slurm, figure_args = parse_args(argv)
    known, common_args = _extract(figure_args)
    kappas = resolve_float_values(known.kappas, option_name="--kappas") or []
    connectivities = resolve_float_values(known.mean_connectivity, option_name="--mean-connectivity")
    conns = ["BASE"] if not connectivities else [format(value, ".15g") for value in connectivities]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging = (REPO_ROOT / slurm.slurm_staging_root / stamp).resolve()
    logs = staging / "logs"
    staging.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    prep_tasks = [(conn, format(kappa, ".15g")) for conn in conns for kappa in kappas]
    work_tasks = [
        (conn, kappa, network, init)
        for conn, kappa in prep_tasks
        for network in range(int(known.n_networks))
        for init in range(int(known.n_inits))
    ]
    prep_file, work_file = staging / "prepare.tsv", staging / "work.tsv"
    prep_file.write_text("\n".join("\t".join(row) for row in prep_tasks) + "\n", encoding="utf-8")
    work_file.write_text(
        "\n".join(f"{conn}\t{kappa}\t{network}\t{init}" for conn, kappa, network, init in work_tasks) + "\n",
        encoding="utf-8",
    )
    common = [
        "python", "Figure4_HighState.py", *common_args,
        "--n-networks", str(known.n_networks), "--n-inits", str(known.n_inits),
        "--output-dir", known.output_dir, "--output-prefix", known.output_prefix, "--jobs", "1",
    ]
    prep_script, worker_script, final_script = staging / "prepare.sbatch", staging / "worker.sbatch", staging / "final.sbatch"
    _worker_script(prep_script, common, prepare=True, data_root=slurm.slurm_data_root)
    _worker_script(worker_script, common, prepare=False, data_root=slurm.slurm_data_root)
    _final_script(
        final_script,
        [*common, "--aggregate-only"],
        REPO_ROOT,
        data_root=slurm.slurm_data_root,
    )
    cap = f"%{slurm.slurm_array_parallelism}" if slurm.slurm_array_parallelism > 0 else ""
    prep_cmd = _sbatch(
        prep_script, partition=slurm.slurm_partition, time=slurm.slurm_prep_time, cpus=1,
        mem=slurm.slurm_prep_mem, log_dir=logs, name=f"{slurm.slurm_job_name_prefix}_prep",
        array=f"0-{len(prep_tasks)-1}{cap}", nodelist=slurm.slurm_nodelist, args=(prep_file, REPO_ROOT),
    )
    if slurm.dry_run:
        worker_template = _sbatch(
            worker_script, partition=slurm.slurm_partition, time=slurm.slurm_worker_time,
            cpus=int(slurm.slurm_cpus_per_task), mem=slurm.slurm_worker_mem, log_dir=logs,
            name=f"{slurm.slurm_job_name_prefix}_work", array=f"0-{len(work_tasks)-1}{cap}",
            dependency="afterok:PREP_JOB_ID", nodelist=slurm.slurm_nodelist, args=(work_file, REPO_ROOT),
        )
        print(f"Staging directory: {staging}")
        print(f"Preparation tasks: {len(prep_tasks)}")
        print(f"Simulation/analysis tasks: {len(work_tasks)}")
        print("Prepare submit command:\n  " + shlex.join(prep_cmd))
        print("Worker submit command:\n  " + shlex.join(worker_template))
        if not slurm.slurm_no_final_job:
            final_template = _sbatch(
                final_script, partition=slurm.slurm_partition, time=slurm.slurm_final_time, cpus=1,
                mem=slurm.slurm_final_mem, log_dir=logs, name=f"{slurm.slurm_job_name_prefix}_final",
                dependency="afterok:WORK_JOB_ID", nodelist=slurm.slurm_nodelist,
            )
            print("Final submit command:\n  " + shlex.join(final_template))
        else:
            print("Final job disabled")
        return 0
    prep_id = subprocess.run(prep_cmd, check=True, capture_output=True, text=True).stdout.strip()
    worker_cmd = _sbatch(
        worker_script, partition=slurm.slurm_partition, time=slurm.slurm_worker_time,
        cpus=int(slurm.slurm_cpus_per_task), mem=slurm.slurm_worker_mem, log_dir=logs,
        name=f"{slurm.slurm_job_name_prefix}_work", array=f"0-{len(work_tasks)-1}{cap}",
        dependency=f"afterok:{prep_id}", nodelist=slurm.slurm_nodelist, args=(work_file, REPO_ROOT),
    )
    worker_id = subprocess.run(worker_cmd, check=True, capture_output=True, text=True).stdout.strip()
    print(f"Submitted preparation {prep_id} and worker array {worker_id}")
    if not slurm.slurm_no_final_job:
        final_cmd = _sbatch(
            final_script, partition=slurm.slurm_partition, time=slurm.slurm_final_time, cpus=1,
            mem=slurm.slurm_final_mem, log_dir=logs, name=f"{slurm.slurm_job_name_prefix}_final",
            dependency=f"afterok:{worker_id}", nodelist=slurm.slurm_nodelist,
        )
        final_id = subprocess.run(final_cmd, check=True, capture_output=True, text=True).stdout.strip()
        print(f"Submitted final aggregation {final_id}")
    print(f"Staging directory: {staging}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
