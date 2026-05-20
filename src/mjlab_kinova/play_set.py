from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

from mjlab_kinova.train_set import (
    _base_parameters,
    _build_run_parameters,
    _load_training_set,
    _normalize_run,
    _resolve_run_variant,
    _resolve_run_cfg,
    _write_temp_params,
)

PLAY_TASK_ID_BY_VARIANT = {
    "joint": "Mjlab-BallBalancing-Kinova-Play",
    "cartesian": "Mjlab-BallBalancing-Kinova-Cartesian",
}


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch play with parameters resolved from a Kinova training-set run."
    )
    parser.add_argument("training_set", help="Path to a training-set YAML file.")
    parser.add_argument(
        "--run",
        required=True,
        dest="run_name",
        help="Name of the run entry to resolve from the training set.",
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


def _play_command(*, variant: str, extra_args: list[str], require_executable: bool) -> list[str]:
    play_exe = shutil.which("play")
    if play_exe is None:
        if require_executable:
            raise RuntimeError("Could not find `play` on PATH. Run this command through `uv run`.")
        play_exe = "play"
    return [play_exe, PLAY_TASK_ID_BY_VARIANT[variant], *extra_args]


def main() -> int:
    args, extra_play_args = _parse_args()
    training_set_path = Path(args.training_set).resolve()
    training_set_cfg = _load_training_set(training_set_path)
    base_params = _base_parameters(training_set_cfg, training_set_path=training_set_path)

    if extra_play_args and extra_play_args[0] == "--":
        extra_play_args = extra_play_args[1:]

    matched_run: tuple[str, str, list[str], dict] | None = None
    for index, raw_run_cfg in enumerate(training_set_cfg["runs"]):
        raw_run_cfg = _resolve_run_cfg(
            raw_run_cfg,
            index=index,
            relative_to=training_set_path.parent,
        )
        run_name, preset_refs, overrides = _normalize_run(raw_run_cfg, index=index)
        run_variant = _resolve_run_variant(training_set_cfg, raw_run_cfg, run_name=run_name)
        if run_name == args.run_name:
            matched_run = (run_name, run_variant, preset_refs, overrides)
            break

    if matched_run is None:
        raise ValueError(f"Run `{args.run_name}` was not found in {training_set_path}")

    run_name, run_variant, preset_refs, overrides = matched_run
    params = _build_run_parameters(
        run_name=run_name,
        preset_refs=preset_refs,
        overrides=overrides,
        training_set_path=training_set_path,
        base_params=base_params,
    )
    if args.ball_kick:
        params = replace(
            params,
            training=replace(params.training, enable_ball_kick_in_play=True),
        )
    temp_config_path = _write_temp_params(params)
    cmd = _play_command(
        variant=run_variant,
        extra_args=extra_play_args,
        require_executable=not args.dry_run,
    )
    env = os.environ.copy()
    env["MJLAB_KINOVA_TASK_PARAMS"] = temp_config_path

    try:
        print(f"[kinova-play-set] run={run_name} variant={run_variant}")
        print(f"[kinova-play-set] params={temp_config_path}")
        print(f"[kinova-play-set] cmd={' '.join(cmd)}")
        if args.dry_run:
            return 0
        completed = subprocess.run(cmd, env=env, check=False)
        return completed.returncode
    finally:
        Path(temp_config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
