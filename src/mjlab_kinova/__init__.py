"""Project-level compatibility patches for mjlab integration."""

from __future__ import annotations

import os
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


_patch_mjlab_gpu_selection()
