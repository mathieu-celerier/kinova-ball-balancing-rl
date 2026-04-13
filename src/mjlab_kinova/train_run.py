from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from mjlab_kinova.train_set import (
    _base_parameters,
    _build_run_parameters,
    _load_training_set,
    _normalize_run,
    _normalize_stop_policy,
    _resolve_run_cfg,
    _timestamped_project_name,
    _train_command,
    _wandb_run_id,
    _write_temp_params,
)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run one Kinova training job from a single-run training-set YAML file."
    )
    parser.add_argument(
        "run_config",
        help="Path to a single-run training-set YAML file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without launching training.",
    )
    args, train_args = parser.parse_known_args()
    return args, train_args


def main() -> int:
    args, extra_train_args = _parse_args()
    run_config_path = Path(args.run_config).resolve()
    training_set_cfg = _load_training_set(run_config_path)
    base_params = _base_parameters(training_set_cfg, training_set_path=run_config_path)

    if extra_train_args and extra_train_args[0] == "--":
        extra_train_args = extra_train_args[1:]

    raw_runs = training_set_cfg["runs"]
    if len(raw_runs) != 1:
        raise ValueError(
            f"{run_config_path} must contain exactly one run for `kinova-train-run`, got {len(raw_runs)}"
        )

    raw_run_cfg = _resolve_run_cfg(
        raw_runs[0],
        index=0,
        relative_to=run_config_path.parent,
    )
    run_name, preset_refs, overrides = _normalize_run(raw_run_cfg, index=0)
    default_stop_policy = training_set_cfg.get(
        "default_stop_policy",
        training_set_cfg.get("default_stop_condition"),
    )
    stop_policy = _normalize_stop_policy(
        raw_run_cfg.get(
            "stop_policy",
            raw_run_cfg.get("stop_condition", default_stop_policy),
        ),
        run_name=run_name,
    )

    project_name = "kinova_ping_pong"
    enriched_overrides = {
        **overrides,
        "training": {
            **overrides.get("training", {}),
            "wandb_project": project_name,
            "run_name": run_name,
        },
    }
    params = _build_run_parameters(
        run_name=run_name,
        preset_refs=preset_refs,
        overrides=enriched_overrides,
        training_set_path=run_config_path,
        base_params=base_params,
    )

    temp_config_path = _write_temp_params(params)
    cmd = _train_command(
        variant=training_set_cfg["variant"],
        extra_args=extra_train_args,
        require_executable=not args.dry_run,
    )
    env = os.environ.copy()
    env["MJLAB_KINOVA_TASK_PARAMS"] = temp_config_path
    env["WANDB_PROJECT"] = project_name
    env["WANDB_NAME"] = run_name
    env["WANDB_RUN_ID"] = _wandb_run_id(run_name)
    if stop_policy is not None:
        env["MJLAB_KINOVA_STOP_CONDITION"] = json.dumps(stop_policy)
    else:
        env.pop("MJLAB_KINOVA_STOP_CONDITION", None)

    try:
        print(f"[kinova-train-run] run={run_name} variant={training_set_cfg['variant']}")
        print(f"[kinova-train-run] params={temp_config_path}")
        print(f"[kinova-train-run] wandb_project={project_name}")
        print(f"[kinova-train-run] wandb_run_id={env['WANDB_RUN_ID']}")
        if stop_policy is not None:
            print(f"[kinova-train-run] stop_policy={json.dumps(stop_policy, sort_keys=True)}")
        print(f"[kinova-train-run] cmd={' '.join(cmd)}")
        if args.dry_run:
            return 0

        completed = subprocess.run(cmd, env=env, check=False)
        return completed.returncode
    finally:
        Path(temp_config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
