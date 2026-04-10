from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from mjlab_kinova.tasks.task_parameters import (
    merge_task_parameter_overrides,
    load_default_task_parameters,
    load_task_parameter_overrides,
    load_task_parameters,
    task_parameters_to_dict,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESET_DIR = REPO_ROOT / "config" / "presets"
TASK_ID_BY_VARIANT = {
    "joint": "Mjlab-BallBalancing-Kinova",
    "cartesian": "Mjlab-BallBalancing-Kinova-Cartesian",
}


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run a set of Kinova training jobs for one control-space variant."
    )
    parser.add_argument("training_set", help="Path to a training-set YAML file.")
    parser.add_argument(
        "--run",
        action="append",
        dest="run_names",
        default=[],
        help="Only execute the named run. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved commands without launching training.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue launching later runs if one training command fails.",
    )
    args, train_args = parser.parse_known_args()
    return args, train_args


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def _merge_override_mappings(*mappings: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = _merge_override_mappings(merged[key], value)
            else:
                merged[key] = value
    return merged


def _resolve_path(path_str: str, *, relative_to: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (relative_to / path).resolve()
    return path


def _resolve_preset_path(preset_ref: str, *, relative_to: Path) -> Path:
    candidate = Path(preset_ref)
    if candidate.suffix in {".yaml", ".yml"} or "/" in preset_ref:
        return _resolve_path(preset_ref, relative_to=relative_to)
    return PRESET_DIR / f"{preset_ref}.yaml"


def _load_training_set(path: Path) -> dict[str, Any]:
    config = _load_yaml_mapping(path)
    runs = config.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"`runs` must be a non-empty list in {path}")
    variant = config.get("variant")
    if variant not in TASK_ID_BY_VARIANT:
        valid = ", ".join(sorted(TASK_ID_BY_VARIANT))
        raise ValueError(f"`variant` must be one of {valid} in {path}")
    return config


def _base_parameters(training_set_cfg: dict[str, Any], *, training_set_path: Path):
    base_config = training_set_cfg.get("base_config")
    if base_config is None:
        base_params = load_default_task_parameters()
    else:
        base_path = _resolve_path(str(base_config), relative_to=training_set_path.parent)
        base_params = load_task_parameters(base_path)

    global_overrides = training_set_cfg.get("global_overrides", {})
    if not isinstance(global_overrides, dict):
        raise TypeError("`global_overrides` must be a mapping when provided")
    return merge_task_parameter_overrides(global_overrides, base=base_params)


def _normalize_run(run_cfg: dict[str, Any], *, index: int) -> tuple[str, list[str], dict[str, Any]]:
    if not isinstance(run_cfg, dict):
        raise TypeError(f"Run #{index + 1} must be a mapping, got {type(run_cfg).__name__}")
    name = run_cfg.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Run #{index + 1} is missing a non-empty `name`")

    presets = run_cfg.get("presets", [])
    if not isinstance(presets, list) or any(not isinstance(preset, str) for preset in presets):
        raise TypeError(f"`presets` for run `{name}` must be a list of strings")

    overrides = run_cfg.get("overrides", {})
    if not isinstance(overrides, dict):
        raise TypeError(f"`overrides` for run `{name}` must be a mapping")

    return name, presets, overrides


def _normalize_stop_criterion(
    criterion: Any, *, run_name: str
) -> dict[str, Any] | None:
    if criterion is None:
        return None
    if not isinstance(criterion, dict):
        raise TypeError(f"Stop criterion for run `{run_name}` must be a mapping")

    metric = criterion.get("metric")
    threshold = criterion.get("threshold")
    if not isinstance(metric, str) or not metric:
        raise ValueError(f"Stop criterion `metric` for run `{run_name}` must be a non-empty string")
    if not isinstance(threshold, (int, float)):
        raise ValueError(f"Stop criterion `threshold` for run `{run_name}` must be numeric")

    normalized = {
        "metric": metric,
        "threshold": float(threshold),
        "mode": criterion.get("mode", "max"),
        "patience": int(criterion.get("patience", 1)),
        "window": int(criterion.get("window", 1)),
    }
    if normalized["mode"] not in {"max", "min"}:
        raise ValueError(f"Stop criterion `mode` for run `{run_name}` must be `max` or `min`")
    if normalized["patience"] < 1 or normalized["window"] < 1:
        raise ValueError(
            f"Stop criterion for run `{run_name}` requires patience>=1 and window>=1"
        )
    return normalized


def _normalize_stop_policy(
    stop_policy: Any,
    *,
    run_name: str,
) -> dict[str, Any] | None:
    if stop_policy is None:
        return None
    if not isinstance(stop_policy, dict):
        raise TypeError(f"Stop policy for run `{run_name}` must be a mapping")

    if "criteria" in stop_policy:
        raw_criteria = stop_policy.get("criteria")
        if not isinstance(raw_criteria, list) or not raw_criteria:
            raise ValueError(f"`criteria` for run `{run_name}` must be a non-empty list")
        criteria = [
            _normalize_stop_criterion(criterion, run_name=run_name)
            for criterion in raw_criteria
        ]
        combine = stop_policy.get("combine", "all")
        min_iterations = int(stop_policy.get("min_iterations", 0))
        max_iterations_raw = stop_policy.get("max_iterations")
        max_iterations = None if max_iterations_raw is None else int(max_iterations_raw)
    else:
        criteria = [_normalize_stop_criterion(stop_policy, run_name=run_name)]
        combine = "all"
        min_iterations = int(stop_policy.get("min_iterations", 0))
        max_iterations_raw = stop_policy.get("max_iterations")
        max_iterations = None if max_iterations_raw is None else int(max_iterations_raw)

    assert all(criterion is not None for criterion in criteria)
    if combine not in {"all", "any"}:
        raise ValueError(f"Stop policy `combine` for run `{run_name}` must be `all` or `any`")
    if min_iterations < 0:
        raise ValueError(f"Stop policy `min_iterations` for run `{run_name}` must be >= 0")
    if max_iterations is not None and max_iterations < 1:
        raise ValueError(f"Stop policy `max_iterations` for run `{run_name}` must be >= 1")
    if max_iterations is not None and max_iterations < min_iterations:
        raise ValueError(
            f"Stop policy for run `{run_name}` requires max_iterations >= min_iterations"
        )

    return {
        "criteria": criteria,
        "combine": combine,
        "min_iterations": min_iterations,
        "max_iterations": max_iterations,
    }


def _build_run_parameters(
    *,
    run_name: str,
    preset_refs: list[str],
    overrides: dict[str, Any],
    training_set_path: Path,
    base_params,
):
    preset_overrides = [
        load_task_parameter_overrides(
            _resolve_preset_path(preset_ref, relative_to=training_set_path.parent)
        )
        for preset_ref in preset_refs
    ]
    params = merge_task_parameter_overrides(*preset_overrides, overrides, base=base_params)
    if params.training.experiment_name_suffix is None:
        params = merge_task_parameter_overrides(
            {"training": {"experiment_name_suffix": run_name}},
            base=params,
        )
    return params


def _wandb_run_id(run_name: str) -> str:
    run_id = re.sub(r"[^A-Za-z0-9_-]+", "-", run_name).strip("-_")
    if not run_id:
        raise ValueError(
            f"Run name `{run_name}` does not contain any characters valid for a W&B run id"
        )
    return run_id[:64]


def _write_temp_params(params) -> str:
    payload = task_parameters_to_dict(params)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".yaml",
        prefix="kinova-train-set-",
        delete=False,
    )
    with handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return handle.name


def _train_command(*, variant: str, extra_args: list[str], require_executable: bool) -> list[str]:
    train_exe = shutil.which("train")
    if train_exe is None:
        if require_executable:
            raise RuntimeError("Could not find `train` on PATH. Run this command through `uv run`.")
        train_exe = "train"
    return [train_exe, TASK_ID_BY_VARIANT[variant], *extra_args]


def main() -> int:
    args, extra_train_args = _parse_args()
    training_set_path = Path(args.training_set).resolve()
    training_set_cfg = _load_training_set(training_set_path)
    training_set_name = training_set_path.stem
    base_params = _base_parameters(training_set_cfg, training_set_path=training_set_path)

    if extra_train_args and extra_train_args[0] == "--":
        extra_train_args = extra_train_args[1:]

    selected_runs = set(args.run_names)
    exit_code = 0
    executed_runs = 0
    default_stop_policy = training_set_cfg.get(
        "default_stop_policy",
        training_set_cfg.get("default_stop_condition"),
    )

    for index, raw_run_cfg in enumerate(training_set_cfg["runs"]):
        run_name, preset_refs, overrides = _normalize_run(raw_run_cfg, index=index)
        if selected_runs and run_name not in selected_runs:
            continue
        executed_runs += 1
        wandb_run_id = _wandb_run_id(run_name)
        stop_policy = _normalize_stop_policy(
            raw_run_cfg.get(
                "stop_policy",
                raw_run_cfg.get("stop_condition", default_stop_policy),
            ),
            run_name=run_name,
        )

        enriched_overrides = _merge_override_mappings(
            {
                "training": {
                    "wandb_project": training_set_name,
                    "run_name": run_name,
                }
            },
            overrides,
        )

        params = _build_run_parameters(
            run_name=run_name,
            preset_refs=preset_refs,
            overrides=enriched_overrides,
            training_set_path=training_set_path,
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
        env["WANDB_PROJECT"] = training_set_name
        env["WANDB_NAME"] = run_name
        env["WANDB_RUN_ID"] = wandb_run_id
        if stop_policy is not None:
            env["MJLAB_KINOVA_STOP_CONDITION"] = json.dumps(stop_policy)
        else:
            env.pop("MJLAB_KINOVA_STOP_CONDITION", None)

        try:
            print(f"[kinova-train-set] run={run_name} variant={training_set_cfg['variant']}")
            print(f"[kinova-train-set] params={temp_config_path}")
            print(f"[kinova-train-set] wandb_project={training_set_name}")
            print(f"[kinova-train-set] wandb_run_id={wandb_run_id}")
            if stop_policy is not None:
                print(f"[kinova-train-set] stop_policy={json.dumps(stop_policy, sort_keys=True)}")
            print(f"[kinova-train-set] cmd={' '.join(cmd)}")
            if args.dry_run:
                continue

            completed = subprocess.run(cmd, env=env, check=False)
            if completed.returncode != 0:
                exit_code = completed.returncode
                if not args.continue_on_error:
                    return exit_code
        finally:
            Path(temp_config_path).unlink(missing_ok=True)

    if selected_runs and executed_runs == 0:
        requested = ", ".join(sorted(selected_runs))
        raise ValueError(f"No runs matched the requested names: {requested}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
