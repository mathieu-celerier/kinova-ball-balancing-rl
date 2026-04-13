from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mjlab_kinova.train_set import (
    _base_parameters,
    _build_run_parameters,
    _load_training_set,
    _merge_override_mappings,
    _normalize_run,
    _normalize_stop_policy,
    _timestamped_project_name,
    _train_command,
    _wandb_run_id,
    _write_temp_params,
)


@dataclass(frozen=True)
class DurationEstimate:
    run_name: str
    num_envs: int
    num_steps_per_env: int
    policy_dt_s: float
    min_iterations: int
    max_iterations: int
    min_env_steps_per_env: int
    max_env_steps_per_env: int
    min_total_env_steps: int
    max_total_env_steps: int
    min_sim_seconds_per_env: float
    max_sim_seconds_per_env: float
    min_wall_seconds: float | None = None
    max_wall_seconds: float | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    run_name: str
    env_steps_per_second: float
    iteration_time_seconds: float | None
    total_steps: int | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate min/max duration bounds for each run in a Kinova training set."
    )
    parser.add_argument("training_set", help="Path to a training-set YAML file.")
    parser.add_argument(
        "--run",
        action="append",
        dest="run_names",
        default=[],
        help="Only include the named run. Can be passed multiple times.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override the number of environments for every run in the estimate.",
    )
    parser.add_argument(
        "--env-steps-per-second",
        type=float,
        default=None,
        help="If provided, estimate wall-clock time using total environment steps / env_steps_per_second.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a real 1-iteration benchmark and use the measured throughput for wall-clock estimates.",
    )
    parser.add_argument(
        "--benchmark-run",
        default=None,
        help="Run name to benchmark. Defaults to the first selected run, or the first run in the set.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the estimate as JSON instead of a text table.",
    )
    return parser.parse_args()


def _format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    hours, rem = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}h{minutes:02d}m{secs:02d}s"


def _resolve_benchmark_run_name(
    *,
    configured_runs: list[str],
    selected_runs: set[str],
    benchmark_run: str | None,
) -> str:
    if benchmark_run is not None:
        if benchmark_run not in configured_runs:
            valid = ", ".join(configured_runs)
            raise ValueError(f"Unknown benchmark run `{benchmark_run}`. Valid runs: {valid}")
        if selected_runs and benchmark_run not in selected_runs:
            raise ValueError(
                f"Benchmark run `{benchmark_run}` is not part of the selected run filter"
            )
        return benchmark_run

    for run_name in configured_runs:
        if not selected_runs or run_name in selected_runs:
            return run_name

    requested = ", ".join(sorted(selected_runs))
    raise ValueError(f"No runs matched the requested names: {requested}")


def _parse_benchmark_metric(output: str, *, pattern: str, label: str) -> float | None:
    match = re.search(pattern, output, re.MULTILINE)
    if match is None:
        return None
    return float(match.group(1))


def _benchmark_env_steps_per_second(
    *,
    training_set_cfg: dict[str, Any],
    training_set_name: str,
    training_set_path: Path,
    base_params,
    run_name: str,
    preset_refs: list[str],
    overrides: dict[str, Any],
    num_envs_override: int | None,
) -> BenchmarkResult:
    benchmark_overrides = overrides
    if num_envs_override is not None:
        benchmark_overrides = {
            **benchmark_overrides,
            "simulation": {
                **benchmark_overrides.get("simulation", {}),
                "num_envs": int(num_envs_override),
            },
        }

    params = _build_run_parameters(
        run_name=run_name,
        preset_refs=preset_refs,
        overrides=_merge_override_mappings(
            {
                "training": {
                    "wandb_project": training_set_name,
                    "run_name": run_name,
                }
            },
            benchmark_overrides,
        ),
        training_set_path=training_set_path,
        base_params=base_params,
    )
    temp_config_path = _write_temp_params(params)
    cmd = _train_command(
        variant=training_set_cfg["variant"],
        extra_args=["--agent.logger", "tensorboard", "--agent.max_iterations", "1"],
        require_executable=True,
    )
    env = os.environ.copy()
    env["MJLAB_KINOVA_TASK_PARAMS"] = temp_config_path
    env["WANDB_PROJECT"] = training_set_name
    env["WANDB_NAME"] = run_name
    env["WANDB_RUN_ID"] = _wandb_run_id(run_name)

    try:
        completed = subprocess.run(
            cmd,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        Path(temp_config_path).unlink(missing_ok=True)

    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0:
        raise RuntimeError(
            "Benchmark training run failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {completed.returncode}\n"
            f"Output:\n{output}"
        )

    env_steps_per_second = _parse_benchmark_metric(
        output,
        pattern=r"Steps per second:\s*([0-9]+(?:\.[0-9]+)?)",
        label="Steps per second",
    )
    if env_steps_per_second is None:
        total_steps = _parse_benchmark_metric(
            output,
            pattern=r"Total steps:\s*([0-9]+(?:\.[0-9]+)?)",
            label="Total steps",
        )
        iteration_time_seconds = _parse_benchmark_metric(
            output,
            pattern=r"Iteration time:\s*([0-9]+(?:\.[0-9]+)?)s",
            label="Iteration time",
        )
        if total_steps is None or iteration_time_seconds is None or iteration_time_seconds <= 0.0:
            raise RuntimeError(
                "Could not parse benchmark throughput from training output.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Output:\n{output}"
            )
        env_steps_per_second = total_steps / iteration_time_seconds
    else:
        total_steps = _parse_benchmark_metric(
            output,
            pattern=r"Total steps:\s*([0-9]+(?:\.[0-9]+)?)",
            label="Total steps",
        )
        iteration_time_seconds = _parse_benchmark_metric(
            output,
            pattern=r"Iteration time:\s*([0-9]+(?:\.[0-9]+)?)s",
            label="Iteration time",
        )

    return BenchmarkResult(
        run_name=run_name,
        env_steps_per_second=float(env_steps_per_second),
        iteration_time_seconds=iteration_time_seconds,
        total_steps=None if total_steps is None else int(round(total_steps)),
    )


def _iteration_bounds(params, stop_policy: dict[str, Any] | None) -> tuple[int, int]:
    ppo_max_iterations = int(params.ppo.max_iterations)
    if stop_policy is None:
        return ppo_max_iterations, ppo_max_iterations

    min_iterations = int(stop_policy["min_iterations"])
    stop_max = stop_policy["max_iterations"]
    max_iterations = ppo_max_iterations if stop_max is None else min(ppo_max_iterations, int(stop_max))
    min_iterations = min(min_iterations, max_iterations)
    return min_iterations, max_iterations


def _estimate_run(
    *,
    run_name: str,
    params,
    stop_policy: dict[str, Any] | None,
    num_envs_override: int | None,
    env_steps_per_second: float | None,
) -> DurationEstimate:
    num_envs = int(num_envs_override if num_envs_override is not None else params.simulation.num_envs)
    num_steps_per_env = int(params.ppo.num_steps_per_env)
    policy_dt_s = float(params.simulation.timestep * params.simulation.decimation)
    min_iterations, max_iterations = _iteration_bounds(params, stop_policy)

    min_env_steps_per_env = min_iterations * num_steps_per_env
    max_env_steps_per_env = max_iterations * num_steps_per_env
    min_total_env_steps = min_env_steps_per_env * num_envs
    max_total_env_steps = max_env_steps_per_env * num_envs
    min_sim_seconds_per_env = min_env_steps_per_env * policy_dt_s
    max_sim_seconds_per_env = max_env_steps_per_env * policy_dt_s

    min_wall_seconds = None
    max_wall_seconds = None
    if env_steps_per_second is not None:
        min_wall_seconds = min_total_env_steps / env_steps_per_second
        max_wall_seconds = max_total_env_steps / env_steps_per_second

    return DurationEstimate(
        run_name=run_name,
        num_envs=num_envs,
        num_steps_per_env=num_steps_per_env,
        policy_dt_s=policy_dt_s,
        min_iterations=min_iterations,
        max_iterations=max_iterations,
        min_env_steps_per_env=min_env_steps_per_env,
        max_env_steps_per_env=max_env_steps_per_env,
        min_total_env_steps=min_total_env_steps,
        max_total_env_steps=max_total_env_steps,
        min_sim_seconds_per_env=min_sim_seconds_per_env,
        max_sim_seconds_per_env=max_sim_seconds_per_env,
        min_wall_seconds=min_wall_seconds,
        max_wall_seconds=max_wall_seconds,
    )


def main() -> int:
    args = _parse_args()
    training_set_path = Path(args.training_set).resolve()
    training_set_cfg = _load_training_set(training_set_path)
    training_set_name = _timestamped_project_name(training_set_path.stem)
    base_params = _base_parameters(training_set_cfg, training_set_path=training_set_path)
    selected_runs = set(args.run_names)
    default_stop_policy = training_set_cfg.get(
        "default_stop_policy",
        training_set_cfg.get("default_stop_condition"),
    )
    configured_runs: list[tuple[str, list[str], dict[str, Any], dict[str, Any]]] = []
    configured_run_names: list[str] = []
    for index, raw_run_cfg in enumerate(training_set_cfg["runs"]):
        run_name, preset_refs, overrides = _normalize_run(raw_run_cfg, index=index)
        configured_runs.append((run_name, preset_refs, overrides, raw_run_cfg))
        configured_run_names.append(run_name)

    benchmark: BenchmarkResult | None = None
    env_steps_per_second = args.env_steps_per_second
    if args.benchmark:
        benchmark_run_name = _resolve_benchmark_run_name(
            configured_runs=configured_run_names,
            selected_runs=selected_runs,
            benchmark_run=args.benchmark_run,
        )
        for run_name, preset_refs, overrides, _raw_run_cfg in configured_runs:
            if run_name != benchmark_run_name:
                continue
            benchmark = _benchmark_env_steps_per_second(
                training_set_cfg=training_set_cfg,
                training_set_name=training_set_name,
                training_set_path=training_set_path,
                base_params=base_params,
                run_name=run_name,
                preset_refs=preset_refs,
                overrides=overrides,
                num_envs_override=args.num_envs,
            )
            env_steps_per_second = benchmark.env_steps_per_second
            break

    estimates: list[DurationEstimate] = []
    for run_name, preset_refs, overrides, raw_run_cfg in configured_runs:
        if selected_runs and run_name not in selected_runs:
            continue

        params = _build_run_parameters(
            run_name=run_name,
            preset_refs=preset_refs,
            overrides=overrides,
            training_set_path=training_set_path,
            base_params=base_params,
        )
        stop_policy = _normalize_stop_policy(
            raw_run_cfg.get(
                "stop_policy",
                raw_run_cfg.get("stop_condition", default_stop_policy),
            ),
            run_name=run_name,
        )
        estimates.append(
            _estimate_run(
                run_name=run_name,
                params=params,
                stop_policy=stop_policy,
                num_envs_override=args.num_envs,
                env_steps_per_second=env_steps_per_second,
            )
        )

    if selected_runs and not estimates:
        requested = ", ".join(sorted(selected_runs))
        raise ValueError(f"No runs matched the requested names: {requested}")

    totals = {
        "min_iterations": sum(item.min_iterations for item in estimates),
        "max_iterations": sum(item.max_iterations for item in estimates),
        "min_total_env_steps": sum(item.min_total_env_steps for item in estimates),
        "max_total_env_steps": sum(item.max_total_env_steps for item in estimates),
        "min_sim_seconds_per_env": sum(item.min_sim_seconds_per_env for item in estimates),
        "max_sim_seconds_per_env": sum(item.max_sim_seconds_per_env for item in estimates),
        "min_wall_seconds": (
            sum(item.min_wall_seconds for item in estimates if item.min_wall_seconds is not None)
            if env_steps_per_second is not None
            else None
        ),
        "max_wall_seconds": (
            sum(item.max_wall_seconds for item in estimates if item.max_wall_seconds is not None)
            if env_steps_per_second is not None
            else None
        ),
    }

    if args.json:
        print(
            json.dumps(
                {
                    "training_set": str(training_set_path),
                    "benchmark": None if benchmark is None else asdict(benchmark),
                    "runs": [asdict(item) for item in estimates],
                    "set_totals": totals,
                },
                indent=2,
            )
        )
        return 0

    print(f"Training set: {training_set_path.name}")
    print()
    if benchmark is not None:
        benchmark_line = (
            f"Benchmark: run={benchmark.run_name}, "
            f"throughput={benchmark.env_steps_per_second:.0f} env_steps/s"
        )
        if benchmark.iteration_time_seconds is not None:
            benchmark_line += f", iteration={benchmark.iteration_time_seconds:.2f}s"
        if benchmark.total_steps is not None:
            benchmark_line += f", total_steps={benchmark.total_steps}"
        print(benchmark_line)
        print()
    for item in estimates:
        print(
            f"{item.run_name}: "
            f"iterations {item.min_iterations}-{item.max_iterations}, "
            f"env_steps {item.min_total_env_steps}-{item.max_total_env_steps}, "
            f"sim/env {_format_seconds(item.min_sim_seconds_per_env)}-{_format_seconds(item.max_sim_seconds_per_env)}"
        )
        if item.min_wall_seconds is not None and item.max_wall_seconds is not None:
            print(
                f"  wall-clock {_format_seconds(item.min_wall_seconds)}-{_format_seconds(item.max_wall_seconds)} "
                f"at {env_steps_per_second:.0f} env_steps/s"
            )
    print()
    print(
        "Set total: "
        f"iterations {totals['min_iterations']}-{totals['max_iterations']}, "
        f"env_steps {totals['min_total_env_steps']}-{totals['max_total_env_steps']}, "
        f"sim/env {_format_seconds(totals['min_sim_seconds_per_env'])}-{_format_seconds(totals['max_sim_seconds_per_env'])}"
    )
    if totals["min_wall_seconds"] is not None and totals["max_wall_seconds"] is not None:
        print(
            "Set wall-clock: "
            f"{_format_seconds(totals['min_wall_seconds'])}-{_format_seconds(totals['max_wall_seconds'])} "
            f"at {env_steps_per_second:.0f} env_steps/s"
        )
    print()
    print("Note: min duration is an optimistic lower bound based on `min_iterations`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
