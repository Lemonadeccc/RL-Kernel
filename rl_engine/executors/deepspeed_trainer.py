# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch

from rl_engine.executors.bridge import (
    WeightBridgeUnavailableError,
    WeightPublisher,
    WeightUpdateManifest,
    make_weight_bridge,
)
from rl_engine.executors.training_contract import (
    RolloutBatchMixin,
    RolloutStageResult,
    TorchRLTrainingConfig,
    TrainingStageResult,
)
from rl_engine.testing import (
    compute_policy_ratio,
    compute_reference_kl,
    masked_mean,
    selected_logprobs_reference,
)


class DeepSpeedUnavailableError(RuntimeError):
    """Raised when the optional DeepSpeed runtime cannot be imported."""


@dataclass(frozen=True)
class DeepSpeedTrainingConfig(TorchRLTrainingConfig):
    """Configuration for the optional DeepSpeed training worker."""

    zero_stage: int = 0
    deepspeed_config: Mapping[str, Any] = field(default_factory=dict)
    initialize_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.zero_stage < 0:
            raise ValueError("zero_stage must be >= 0")


class DeepSpeedTrainingWorker(RolloutBatchMixin):
    """
    Training worker implementation backed by a real DeepSpeed engine contract.

    DeepSpeed is optional for RL-Kernel, so importing this module never imports
    DeepSpeed. The runtime is loaded only when a worker is constructed.
    """

    config: DeepSpeedTrainingConfig

    def __init__(
        self,
        config: Optional[DeepSpeedTrainingConfig] = None,
        *,
        weight_bridge: Optional[WeightPublisher] = None,
        weight_transport: str = "local-clone",
    ):
        self.config = config or DeepSpeedTrainingConfig()
        self.device = torch.device(self.config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training requested but torch.cuda.is_available() is false")
        self.weight_bridge = weight_bridge or make_weight_bridge(
            weight_transport,
            source_worker="deepspeed-training",
            source_rank=0,
        )
        self._latest_published_weight_version = -1

        deepspeed = _load_deepspeed()
        torch.manual_seed(self.config.seed)
        self.model = torch.nn.Sequential(
            torch.nn.Embedding(self.config.vocab_size, self.config.hidden_dim),
            torch.nn.Linear(self.config.hidden_dim, self.config.vocab_size),
        ).to(device=self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)

        init_result = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            optimizer=self.optimizer,
            config=self._resolved_deepspeed_config(),
            **dict(self.config.initialize_kwargs),
        )
        self.engine = _first_initialize_result(init_result)
        engine_device = getattr(self.engine, "device", None)
        if engine_device is not None:
            self.device = torch.device(engine_device)

    def train(self, rollout: RolloutStageResult) -> TrainingStageResult:
        started_at = time.perf_counter()
        batch, payload_metrics = self._batch_from_rollout_or_synthetic(rollout)

        logits = _extract_logits(self.engine(batch.token_ids.long()))
        current_logps = selected_logprobs_reference(
            logits,
            batch.token_ids,
            mask=batch.completion_mask,
            output_dtype=torch.float32,
        )
        old_logps = current_logps.detach() - 0.01
        ref_logps = current_logps.detach() - 0.02
        ratio = compute_policy_ratio(current_logps, old_logps, batch.completion_mask)
        unclipped = ratio * batch.advantages.float()
        clipped = torch.clamp(ratio, 0.8, 1.2) * batch.advantages.float()
        policy_loss = -torch.minimum(unclipped, clipped)
        kl = compute_reference_kl(current_logps, ref_logps, batch.completion_mask)
        loss = masked_mean(policy_loss + 0.01 * kl, batch.completion_mask)

        if hasattr(self.engine, "zero_grad"):
            try:
                self.engine.zero_grad(set_to_none=True)
            except TypeError:
                self.engine.zero_grad()
        elif hasattr(self.optimizer, "zero_grad"):
            self.optimizer.zero_grad(set_to_none=True)
        self.engine.backward(loss)
        self.engine.step()

        finished_at = time.perf_counter()
        published = self._next_published_weight_version(rollout.weight_version)
        return TrainingStageResult(
            iteration=rollout.iteration,
            consumed_weight_version=rollout.weight_version,
            published_weight_version=published,
            metrics={
                "loss": float(loss.detach().cpu().item()),
                "active_tokens": int(batch.completion_mask.sum().item()),
                "payload_type": type(rollout.payload).__name__,
                "training_backend": "deepspeed",
                "training_device": str(self.device),
                "deepspeed_engine": type(self.engine).__name__,
                "deepspeed_zero_stage": self.config.zero_stage,
                **payload_metrics,
            },
            started_at=started_at,
            finished_at=finished_at,
        )

    def publish_weights(
        self,
        *,
        weight_version: int,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> WeightUpdateManifest:
        """
        Publish the current DeepSpeed model state as a complete weight manifest.

        ZeRO-3 partitions parameters across ranks, so a production publication
        must all-gather a full state first. Until that path exists, the worker
        fails explicitly instead of publishing an incomplete shard.
        """
        if self.config.zero_stage >= 3:
            raise WeightBridgeUnavailableError(
                "DeepSpeed ZeRO-3 publish requires a real all-gather/full-state "
                "export before rollout workers can consume the weight manifest."
            )

        manifest_metadata = dict(metadata or {})
        layout = {
            "kind": "full-state",
            "zero_stage": self.config.zero_stage,
            "world_size": 1,
            "rank": 0,
        }
        layout.update(dict(manifest_metadata.get("layout", {})))
        manifest_metadata["layout"] = layout
        return self.weight_bridge.publish(
            self.model,
            weight_version=weight_version,
            metadata=manifest_metadata,
        )

    def release_weights(self, update_id: str) -> None:
        self.weight_bridge.release(update_id)

    def _next_published_weight_version(self, consumed_weight_version: int) -> int:
        published = max(
            int(consumed_weight_version) + 1,
            self._latest_published_weight_version + 1,
        )
        self._latest_published_weight_version = published
        return published

    def _resolved_deepspeed_config(self) -> dict[str, Any]:
        batch_size = max(1, self.config.num_prompts * self.config.samples_per_prompt)
        base = {
            "train_micro_batch_size_per_gpu": batch_size,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": self.config.zero_stage},
            "fp16": {"enabled": self.config.dtype == torch.float16},
            "bf16": {"enabled": self.config.dtype == torch.bfloat16},
        }
        return _deep_merge(base, dict(self.config.deepspeed_config))


def _load_deepspeed() -> Any:
    try:
        return importlib.import_module("deepspeed")
    except ImportError as exc:
        raise DeepSpeedUnavailableError(
            "DeepSpeed is not installed or cannot be imported. Install a DeepSpeed "
            "runtime supported by the active Python/PyTorch/CUDA environment before "
            "running DeepSpeedTrainingWorker."
        ) from exc


def _first_initialize_result(init_result: Any) -> Any:
    if isinstance(init_result, tuple):
        if not init_result:
            raise RuntimeError("deepspeed.initialize returned an empty tuple")
        return init_result[0]
    return init_result


def _extract_logits(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, Mapping) and "logits" in model_output:
        return model_output["logits"]
    logits = getattr(model_output, "logits", None)
    if logits is not None:
        return logits
    if isinstance(model_output, (tuple, list)) and model_output:
        return _extract_logits(model_output[0])
    raise TypeError(f"DeepSpeed model output does not expose logits: {type(model_output)!r}")


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged
