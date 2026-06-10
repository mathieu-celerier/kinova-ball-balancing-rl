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


def _patch_mjlab_video_upload_setting() -> None:
    try:
        import mjlab.scripts.train as mjlab_train
    except ImportError:
        return

    if getattr(mjlab_train, "_kinova_video_upload_setting_patched", False):
        return

    original_run_train = mjlab_train.run_train

    def _run_train_with_video_upload_setting(task_id, cfg, log_dir):
        previous_upload_setting = os.environ.get("MJLAB_KINOVA_UPLOAD_VIDEOS_TO_WANDB")
        if previous_upload_setting is None:
            upload_videos = getattr(cfg.env, "_kinova_upload_videos_to_wandb", True)
            os.environ["MJLAB_KINOVA_UPLOAD_VIDEOS_TO_WANDB"] = (
                "1" if upload_videos else "0"
            )
        else:
            upload_videos = previous_upload_setting.lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
        if not upload_videos:
            print("[INFO] Training videos will be kept locally and not uploaded to W&B.")
        try:
            return original_run_train(task_id, cfg, log_dir)
        finally:
            if previous_upload_setting is None:
                os.environ.pop("MJLAB_KINOVA_UPLOAD_VIDEOS_TO_WANDB", None)
            else:
                os.environ["MJLAB_KINOVA_UPLOAD_VIDEOS_TO_WANDB"] = previous_upload_setting

    mjlab_train.run_train = _run_train_with_video_upload_setting
    mjlab_train._kinova_video_upload_setting_patched = True


def _patch_wandb_video_upload() -> None:
    try:
        from rsl_rl.utils.wandb_utils import WandbSummaryWriter
    except ImportError:
        return

    if getattr(WandbSummaryWriter, "_kinova_video_upload_patched", False):
        return

    original_save_video = WandbSummaryWriter.save_video

    def _save_video_if_enabled(self, video, it):
        enabled = os.environ.get("MJLAB_KINOVA_UPLOAD_VIDEOS_TO_WANDB", "1").lower()
        if enabled not in {"0", "false", "no", "off"}:
            return original_save_video(self, video, it)
        return None

    WandbSummaryWriter.save_video = _save_video_if_enabled
    WandbSummaryWriter._kinova_video_upload_patched = True


_patch_mjlab_gpu_selection()
_patch_mjlab_video_upload_setting()
_patch_wandb_video_upload()
