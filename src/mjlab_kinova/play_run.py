from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from mjlab_kinova.play_set import _play_command
from mjlab_kinova.train_set import (
    _base_parameters,
    _build_run_parameters,
    _load_training_set,
    _normalize_run,
    _resolve_run_cfg,
    _write_temp_params,
)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch play from a single-run Kinova training-set YAML file."
    )
    parser.add_argument(
        "run_config",
        help="Path to a single-run training-set YAML file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved play command without launching it.",
    )
    parser.add_argument(
        "--ball-kick",
        action="store_true",
        help="Enable ball-kick interval disturbances while replaying.",
    )
    args, play_args = parser.parse_known_args()
    return args, play_args


def main() -> int:
    args, extra_play_args = _parse_args()
    run_config_path = Path(args.run_config).resolve()
    training_set_cfg = _load_training_set(run_config_path)
    base_params = _base_parameters(training_set_cfg, training_set_path=run_config_path)

    if extra_play_args and extra_play_args[0] == "--":
        extra_play_args = extra_play_args[1:]

    raw_runs = training_set_cfg["runs"]
    if len(raw_runs) != 1:
        raise ValueError(
            f"{run_config_path} must contain exactly one run for `kinova-play-run`, got {len(raw_runs)}"
        )

    raw_run_cfg = _resolve_run_cfg(
        raw_runs[0],
        index=0,
        relative_to=run_config_path.parent,
    )
    run_name, preset_refs, overrides = _normalize_run(raw_run_cfg, index=0)
    params = _build_run_parameters(
        run_name=run_name,
        preset_refs=preset_refs,
        overrides=overrides,
        training_set_path=run_config_path,
        base_params=base_params,
    )
    if args.ball_kick:
        params = replace(
            params,
            training=replace(params.training, enable_ball_kick_in_play=True),
        )

    temp_config_path = _write_temp_params(params)
    cmd = _play_command(
        variant=training_set_cfg["variant"],
        extra_args=extra_play_args,
        require_executable=not args.dry_run,
    )
    env = os.environ.copy()
    env["MJLAB_KINOVA_TASK_PARAMS"] = temp_config_path

    try:
        print(f"[kinova-play-run] run={run_name} variant={training_set_cfg['variant']}")
        print(f"[kinova-play-run] params={temp_config_path}")
        print(f"[kinova-play-run] cmd={' '.join(cmd)}")
        if args.dry_run:
            return 0

        completed = subprocess.run(cmd, env=env, check=False)
        return completed.returncode
    finally:
        Path(temp_config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
