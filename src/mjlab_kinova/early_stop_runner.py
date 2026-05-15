from __future__ import annotations

import json
import os
import statistics
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb
import torch
from rsl_rl.utils import check_nan

from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner

STOP_CONDITION_ENV_VAR = "MJLAB_KINOVA_STOP_CONDITION"


@dataclass(frozen=True)
class StopCriterion:
    metric: str
    threshold: float
    mode: str = "max"
    patience: int = 1
    window: int = 1


@dataclass(frozen=True)
class StopPolicy:
    criteria: tuple[StopCriterion, ...]
    combine: str = "all"
    min_iterations: int = 0
    max_iterations: int | None = None


@dataclass
class _CriterionState:
    criterion: StopCriterion
    metric_window: deque[float]
    successful_iterations: int = 0


class KinovaOnPolicyRunner(MjlabOnPolicyRunner):
    """Kinova runner with optional metric-based early stopping."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stop_policy = self._load_stop_policy()
        self._criterion_states: list[_CriterionState] = []
        if self._stop_policy is not None:
            self._criterion_states = [
                _CriterionState(
                    criterion=criterion,
                    metric_window=deque(maxlen=criterion.window),
                )
                for criterion in self._stop_policy.criteria
            ]

    def _parse_criterion(self, payload: dict[str, Any]) -> StopCriterion:
        metric = payload.get("metric")
        threshold = payload.get("threshold")
        if not isinstance(metric, str) or not metric:
            raise ValueError("Stop criterion requires a non-empty string `metric`")
        if not isinstance(threshold, (int, float)):
            raise ValueError("Stop criterion requires numeric `threshold`")

        mode = payload.get("mode", "max")
        if mode not in {"max", "min"}:
            raise ValueError("Stop criterion `mode` must be `max` or `min`")

        patience = int(payload.get("patience", 1))
        window = int(payload.get("window", 1))
        if patience < 1 or window < 1:
            raise ValueError("Stop criterion requires patience>=1 and window>=1")

        return StopCriterion(
            metric=metric,
            threshold=float(threshold),
            mode=mode,
            patience=patience,
            window=window,
        )

    def _load_stop_policy(self) -> StopPolicy | None:
        raw = os.environ.get(STOP_CONDITION_ENV_VAR, "").strip()
        if not raw:
            return None

        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise TypeError(
                f"{STOP_CONDITION_ENV_VAR} must decode to a mapping, got {type(payload).__name__}"
            )

        if "criteria" in payload:
            raw_criteria = payload.get("criteria")
            if not isinstance(raw_criteria, list) or not raw_criteria:
                raise ValueError("Stop policy `criteria` must be a non-empty list")
            criteria = tuple(self._parse_criterion(item) for item in raw_criteria)
            combine = payload.get("combine", "all")
            min_iterations = int(payload.get("min_iterations", 0))
            max_iterations_raw = payload.get("max_iterations")
            max_iterations = None if max_iterations_raw is None else int(max_iterations_raw)
        else:
            criteria = (self._parse_criterion(payload),)
            combine = "all"
            min_iterations = int(payload.get("min_iterations", 0))
            max_iterations_raw = payload.get("max_iterations")
            max_iterations = None if max_iterations_raw is None else int(max_iterations_raw)

        if combine not in {"all", "any"}:
            raise ValueError("Stop policy `combine` must be `all` or `any`")
        if min_iterations < 0:
            raise ValueError("Stop policy requires min_iterations>=0")
        if max_iterations is not None and max_iterations < 1:
            raise ValueError("Stop policy requires max_iterations>=1 when provided")
        if max_iterations is not None and max_iterations < min_iterations:
            raise ValueError("Stop policy requires max_iterations>=min_iterations")

        return StopPolicy(
            criteria=criteria,
            combine=combine,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
        )

    def _collect_metrics(
        self,
        loss_dict: dict[str, Any],
        learning_rate: float,
        action_std: torch.Tensor,
        rnd_weight: float | None,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}

        if self.logger.ep_extras:
            extra_keys = {
                key
                for ep_info in self.logger.ep_extras
                for key in ep_info
            }
            for key in extra_keys:
                values: list[float] = []
                for ep_info in self.logger.ep_extras:
                    value = ep_info.get(key, 0.0)
                    if isinstance(value, torch.Tensor):
                        tensor = value.detach().reshape(-1).float()
                        if tensor.numel() > 0:
                            values.extend(tensor.cpu().tolist())
                        else:
                            values.append(0.0)
                    else:
                        values.append(float(value))
                if values:
                    metric_key = key if "/" in key else f"Episode/{key}"
                    metrics[metric_key] = float(sum(values) / len(values))

        for key, value in loss_dict.items():
            metrics[f"Loss/{key}"] = float(value)
        metrics["Loss/learning_rate"] = float(learning_rate)
        metrics["Policy/mean_std"] = float(action_std.mean().item())

        if len(self.logger.rewbuffer) > 0:
            metrics["Train/mean_reward"] = float(statistics.mean(self.logger.rewbuffer))
            metrics["Train/mean_episode_length"] = float(statistics.mean(self.logger.lenbuffer))

        if self.cfg["algorithm"]["rnd_cfg"]:
            metrics["Rnd/weight"] = float(rnd_weight if rnd_weight is not None else 0.0)
            if len(self.logger.erewbuffer) > 0:
                metrics["Rnd/mean_extrinsic_reward"] = float(statistics.mean(self.logger.erewbuffer))
                metrics["Rnd/mean_intrinsic_reward"] = float(statistics.mean(self.logger.irewbuffer))

        return metrics

    def _should_stop(self, it: int, metrics: dict[str, float]) -> tuple[bool, str | None]:
        policy = self._stop_policy
        if policy is None:
            return False, None
        if it + 1 < policy.min_iterations:
            return False, None

        results: list[tuple[bool, str | None]] = []
        for state in self._criterion_states:
            criterion = state.criterion
            metric_value = metrics.get(criterion.metric)
            if metric_value is None:
                state.successful_iterations = 0
                results.append((False, None))
                continue

            state.metric_window.append(metric_value)
            if len(state.metric_window) < criterion.window:
                state.successful_iterations = 0
                results.append((False, None))
                continue

            smoothed_value = float(sum(state.metric_window) / len(state.metric_window))
            passed = (
                smoothed_value >= criterion.threshold
                if criterion.mode == "max"
                else smoothed_value <= criterion.threshold
            )
            if passed:
                state.successful_iterations += 1
            else:
                state.successful_iterations = 0

            criterion_satisfied = state.successful_iterations >= criterion.patience
            reason = (
                f"{criterion.metric}: smoothed={smoothed_value:.4f}, "
                f"threshold={criterion.threshold:.4f}, mode={criterion.mode}, "
                f"window={criterion.window}, patience={criterion.patience}"
            )
            results.append((criterion_satisfied, reason))

        satisfied = [flag for flag, _ in results]
        if not satisfied:
            return False, None

        stop = all(satisfied) if policy.combine == "all" else any(satisfied)
        if not stop:
            return False, None

        reasons = [reason for flag, reason in results if flag and reason is not None]
        prefix = "early stop" if policy.combine == "all" else "early stop (any criterion)"
        return True, f"{prefix}: " + "; ".join(reasons)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        if self._stop_policy is not None and self._stop_policy.max_iterations is not None:
            total_it = min(total_it, start_it + self._stop_policy.max_iterations)
        stopped_early = False
        stop_reason: str | None = None

        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            rnd_weight = self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=rnd_weight,
            )

            metrics = self._collect_metrics(
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=rnd_weight,
            )
            should_stop, stop_reason = self._should_stop(it, metrics)
            if should_stop:
                print(f"[INFO]: {stop_reason}", flush=True)
                stopped_early = True
                break

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

        if self.logger.writer is not None:
            infos = {"early_stop_reason": stop_reason} if stopped_early and stop_reason else None
            self.save(
                os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"),  # type: ignore[arg-type]
                infos=infos,
            )
            self.logger.stop_logging_writer()

    def save(self, path: str, infos=None) -> None:
        super().save(path, infos)
        checkpoint_path = Path(path)
        policy_dir = checkpoint_path.parent
        filename = checkpoint_path.with_suffix(".onnx").name
        onnx_path = policy_dir / filename
        try:
            self.export_policy_to_onnx(str(policy_dir), filename)
            try:
                run_name = (
                    wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
                )
                metadata = get_base_metadata(self.env.unwrapped, run_name)
                attach_metadata_to_onnx(str(onnx_path), metadata)
            except Exception as e:
                print(f"[WARN] ONNX metadata attachment failed (training continues): {e}")
            if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
                wandb.save(str(onnx_path), base_path=str(policy_dir))
        except Exception as e:
            print(f"[WARN] ONNX export failed (training continues): {e}")
