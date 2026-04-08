"""Project-level compatibility patches for mjlab integration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal


def _safe_select_gpus(
    gpu_ids: list[int] | Literal["all"] | None,
) -> tuple[list[int] | None, int]:
    """Mirror mjlab GPU selection but fall back cleanly when no GPU is present."""
    if gpu_ids is None:
        return None, 0

    existing_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing_visible_devices is not None:
        available_gpus = [int(x.strip()) for x in existing_visible_devices.split(",") if x.strip()]
        if not available_gpus:
            return None, 0
    else:
        import torch.cuda

        available_gpus = list(range(torch.cuda.device_count()))
        if not available_gpus:
            return None, 0

    if gpu_ids == "all":
        selected_gpus = available_gpus
    else:
        if any(gpu_id < 0 or gpu_id >= len(available_gpus) for gpu_id in gpu_ids):
            raise ValueError(
                f"Requested GPU indices {gpu_ids}, but only {len(available_gpus)} visible GPU(s) are "
                f"available: {available_gpus}"
            )
        selected_gpus = [available_gpus[i] for i in gpu_ids]

    return selected_gpus, len(selected_gpus)


def _patch_mjlab_gpu_selection() -> None:
    import mjlab.utils.gpu as mjlab_gpu

    mjlab_gpu.select_gpus = _safe_select_gpus

    try:
        import mjlab.scripts.train as mjlab_train
    except ImportError:
        return

    mjlab_train.select_gpus = _safe_select_gpus


def _wandb_is_configured() -> bool:
    """Return whether this shell has enough local W&B setup to use the wandb logger."""
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False

    if os.environ.get("WANDB_API_KEY"):
        return True

    netrc_path = os.environ.get("NETRC")
    path = Path(netrc_path).expanduser() if netrc_path else Path.home() / ".netrc"
    if not path.is_file():
        return False

    try:
        contents = path.read_text(encoding="utf-8")
    except OSError:
        return False

    return "machine api.wandb.ai" in contents or "machine wandb.ai" in contents


def _patch_mjlab_wandb_fallback() -> None:
    try:
        import mjlab.scripts.train as mjlab_train
    except ImportError:
        return

    if getattr(mjlab_train, "_kinova_wandb_fallback_patched", False):
        return

    original_run_train = mjlab_train.run_train

    def _run_train_with_wandb_fallback(task_id, cfg, log_dir):
        if getattr(cfg.agent, "logger", None) == "wandb" and not _wandb_is_configured():
            print(
                "[INFO] W&B is not configured in this shell; "
                "falling back to TensorBoard logging."
            )
            cfg.agent.logger = "tensorboard"
        return original_run_train(task_id, cfg, log_dir)

    mjlab_train.run_train = _run_train_with_wandb_fallback
    mjlab_train._kinova_wandb_fallback_patched = True


_patch_mjlab_gpu_selection()
_patch_mjlab_wandb_fallback()
