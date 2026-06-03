# Executable Plan - Issue #47 Stateless Reference/Reward Forward Engine

## Goal

Add a memory-efficient, stateless forward executor for Reference and Reward
models so RLHF scoring does not allocate generation-oriented paged KV cache.

## Selected Candidate

- **Candidate ID:** `issue-47-candidate-1`
- **Scope:** PyTorch stateless scoring executor, strict tensor validation,
  reference selected-logprob scoring, reward adapter contract, focused tests,
  smoke benchmark scaffold, and issue evidence.
- **Explicit non-goal:** do not replace the policy generation engine, add
  distributed scoring, or implement custom attention kernels in this candidate.

## Why This Matters

Reference and Reward models score completed sequences. They do not need to keep
KV states around for future token decoding. A generation engine can reserve
large paged KV buffers even when the scoring workload only needs one forward
pass. Avoiding that reservation can leave more VRAM for the Policy model, which
is usually the scarce resource in RLHF training.

## Steps

- [x] Step 1: Add the public executor module.
  - Create `rl_engine/executors/stateless_executor.py`.
  - Add `StatelessForwardConfig`.
  - Add `StatelessForwardInputs`.
  - Add `StatelessForwardResult`.
  - Add `StatelessForwardExecutor`.
- [x] Step 2: Define strict validation.
  - Validate `input_ids` and `attention_mask` share `[B, S]`.
  - Validate `completion_mask` shape and dtype.
  - Validate labels/scoring ids match model logits.
  - Validate reward adapter output is `[B]`.
  - Reject unsupported modes with clear messages.
- [x] Step 3: Implement stateless forward execution.
  - Call model forward with `use_cache=False` when accepted.
  - Fall back cleanly for models that do not accept `use_cache`.
  - Avoid importing vLLM, DeepSpeed, Ray, or FlashAttention at module import.
  - Detach outputs by default.
- [x] Step 4: Implement Reference scoring.
  - Compute causal next-token selected logprobs.
  - Align logprobs to completion tokens.
  - Mask prompt and padded tokens.
  - Compare toy outputs against a direct PyTorch reference.
- [x] Step 5: Implement Reward scoring.
  - Add a reward adapter callback.
  - Provide a simple default adapter for `[B]` or `[B, 1]` logits.
  - Report one scalar reward per sequence.
- [x] Step 6: Add metrics.
  - Record mode, batch size, sequence length, active completion tokens, dtype,
    device, elapsed time, and whether outputs were detached.
  - Record CUDA peak allocated/reserved memory when CUDA is available.
- [x] Step 7: Add focused tests.
  - Reference logprob math.
  - Reward adapter behavior.
  - Both-mode behavior.
  - Shape mismatch errors.
  - `use_cache=False` behavior.
  - Import-light behavior.
- [x] Step 8: Add benchmark scaffold.
  - Add `benchmarks/benchmark_stateless_executor.py --smoke --mode reference`.
  - Add `--mode reward`.
  - Add optional local paged-KV reservation comparison rows.
- [x] Step 9: Integrate with pipeline only after tests pass.
  - Add a narrow scoring stage interface for `overlap_pipeline.py`.
  - Keep realistic scoring optional in CPU-only tests.
  - Preserve synthetic fallback metrics when real models are unavailable.
- [x] Step 10: Run validation.
  ```powershell
  py -3.13 -m pytest tests/test_stateless_executor.py -q
  py -3.13 -m pytest tests/test_stateless_executor.py tests/test_reference_ops.py tests/test_grpo_objective.py -q
  py -3.13 -m compileall rl_engine tests benchmarks
  ```
- [x] Step 11: Record evidence.
  - Append candidate status to `candidates.jsonl`.
  - Append benchmark or blocker rows to `benchmark.csv`.
  - Save optional JSON metrics under `outputs/`.
- [x] Step 12: Add local paged-KV scoring baseline.
  - Add a generation-engine-style paged KV reservation baseline.
  - Allocate K/V page tensors, block tables, and sequence length metadata.
  - Run the wrapped model with `use_cache=True` when supported.
  - Record `paged_kv_baseline_mb` in the benchmark CSV.

## Acceptance Criteria

- Reference and Reward scoring can run without a generation loop.
- The executor uses `use_cache=False` or an equivalent no-cache path.
- Reference selected logprobs match exact PyTorch reference math.
- Reward scoring returns one scalar per sequence.
- Prompt and padded completion positions do not affect scoring outputs or
  metrics.
- Outputs passed into training are detached by default.
- Importing the executor does not require vLLM, DeepSpeed, Ray, CUDA, or
  FlashAttention.
- Memory benchmark evidence records whether stateless scoring avoids paged KV
  allocation in the local environment.

## Validation Result

- `py -3.13 -m pytest tests/test_stateless_executor.py -q`: passed, 9 tests.
- `py -3.13 -m pytest tests/test_stateless_executor.py tests/test_reference_ops.py tests/test_grpo_objective.py -q`: passed, 23 tests.
- `py -3.13 -m pytest tests/test_overlap_pipeline.py -q`: passed, 15 tests.
- `py -3.13 -m pytest tests/test_stateless_executor.py tests/test_reference_ops.py tests/test_grpo_objective.py tests/test_deepspeed_training_worker.py tests/test_overlap_pipeline.py -q`: passed, 42 tests.
- `py -3.13 -m pytest tests/test_stateless_executor.py tests/test_reference_ops.py tests/test_grpo_objective.py tests/test_deepspeed_training_worker.py tests/test_overlap_pipeline.py -q`: passed, 44 tests after adding rollout collator and payload reference-logp ingestion.
- `py -3.13 -m pytest tests/test_paged_kv_baseline.py -q`: passed, 4 tests.
- `py -3.13 -m pytest tests/test_paged_kv_baseline.py tests/test_stateless_executor.py tests/test_reference_ops.py tests/test_grpo_objective.py tests/test_deepspeed_training_worker.py tests/test_overlap_pipeline.py -q`: passed, 48 tests.
- `py -3.13 -m compileall rl_engine tests benchmarks`: passed.

## Benchmark Result

- `py -3.13 benchmarks/benchmark_stateless_executor.py --smoke --mode reference --json-output task-workspace/issues/issue_47/outputs/stateless-reference-smoke.json`: passed.
  - `batch_size=2`
  - `seq_len=16`
  - `active_tokens=16`
  - `device=cpu`
  - latest `elapsed_ms=2.5144`
  - `use_cache_passed=True`
  - `detached_outputs=True`
- `py -3.13 benchmarks/benchmark_stateless_executor.py --smoke --mode reward --json-output task-workspace/issues/issue_47/outputs/stateless-reward-smoke.json`: passed.
  - `batch_size=2`
  - `seq_len=16`
  - `active_tokens=16`
  - `device=cpu`
  - latest `elapsed_ms=0.5898`
  - `use_cache_passed=True`
  - `detached_outputs=True`
- `py -3.13 benchmarks/benchmark_stateless_executor.py --smoke --mode both --json-output task-workspace/issues/issue_47/outputs/stateless-both-smoke.json`: passed.
  - `batch_size=2`
  - `seq_len=16`
  - `active_tokens=16`
  - `device=cpu`
  - latest `elapsed_ms=4.5607`
  - `use_cache_passed=True`
  - `detached_outputs=True`

The benchmark environment reports `torch=2.7.1+cu126`, CUDA 12.6, and an NVIDIA
GeForce RTX 3050, but these smoke rows were run on CPU.

- `py -3.13 benchmarks/benchmark_stateless_executor.py --smoke --mode paged-kv-compare --json-output task-workspace/issues/issue_47/outputs/paged-kv-baseline-smoke.json`: passed.
  - `batch_size=2`
  - `seq_len=16`
  - `active_tokens=16`
  - `device=cpu`
  - latest `elapsed_ms=2.8782`
  - `paged_kv_baseline_mb=0.0039`
  - `paged_kv_blocks=2`
  - `paged_kv_required_blocks=2`
  - `use_cache_passed=True`
- `py -3.13 benchmarks/benchmark_stateless_executor.py --smoke --mode reference --compare-paged-kv --json-output task-workspace/issues/issue_47/outputs/stateless-reference-with-paged-kv-baseline-smoke.json`: passed.
  - `batch_size=2`
  - `seq_len=16`
  - `active_tokens=16`
  - `device=cpu`
  - latest stateless `elapsed_ms=2.5050`
  - `paged_kv_baseline_mb=0.0039`
  - `paged_kv_reference_allclose=True`

## Implementation Notes

- Added `rl_engine/executors/stateless_executor.py` with:
  - `StatelessForwardConfig`
  - `StatelessForwardInputs`
  - `StatelessForwardOutputs`
  - `StatelessForwardResult`
  - `StatelessForwardExecutor`
  - `score_reference_logprobs`
  - `score_rewards`
  - `default_reward_adapter`
- Reference scoring uses causal next-token alignment and returns full-sequence
  `[B, S]` logprobs masked to active completion positions.
- Reward scoring accepts a pluggable adapter and validates one scalar reward per
  sequence.
- The executor calls model forward with `use_cache=False` when supported and
  falls back cleanly for models that do not accept the keyword.
- Added `StatelessScoringWorker` and `attach_stateless_scores_to_payload(...)`
  in `rl_engine/executors/overlap_pipeline.py` so rollout payloads can be scored
  between rollout and training without changing the scheduler loop.
- Added `build_stateless_inputs_from_rollout_payload(...)` for a default dense
  collator from grouped rollout candidates into full-sequence scoring inputs.
- Stateless scoring payloads now attach both scalar `reward` values and
  per-candidate `reference_logps` back onto rollout candidates when available.
- `TorchRLTrainingWorker` now prefers payload `reference_logps` for the GRPO
  reference KL path and records `reference_logp_source=payload_reference_logps`;
  if payload logprobs are absent or incomplete, it records
  `reference_logp_source=synthetic_current_offset` and uses the existing smoke
  fallback.
- Added `tests/test_stateless_executor.py` and a pipeline bridge test proving
  stateless rewards are attached to rollout candidates and consumed as
  `payload_rewards` by the existing training worker.
- Added end-to-end overlap tests proving the default rollout collator preserves
  candidate order/masks and that stateless `reference_logps` reach the trainer.
- Added `benchmarks/benchmark_stateless_executor.py` with `reference`, `reward`,
  and `both` smoke modes.
- Added `rl_engine/executors/paged_kv_baseline.py` with:
  - `PagedKVScoringConfig`
  - `PagedKVCacheReservation`
  - `PagedKVScoringBaseline`
  - `reserve_paged_kv_cache`
- The paged-KV baseline allocates generation-style K/V block tensors plus a
  block table and records their reserved bytes. This is a local reservation
  baseline, not a vLLM performance measurement.
- `benchmark_stateless_executor.py` now supports `--mode paged-kv-compare` and
  `--compare-paged-kv` so rows can populate `paged_kv_baseline_mb`.
- Added `tests/test_paged_kv_baseline.py` for block-table layout, reservation
  accounting, `use_cache=True` behavior, and reference-logprob parity.

## Follow-up Roadmap

The code path is now complete for local stateless scoring smoke tests:

```text
rollout payload
  -> build_stateless_inputs_from_rollout_payload(...)
  -> StatelessForwardExecutor.score(...)
  -> attach rewards/reference_logps to candidates
  -> TorchRLTrainingWorker consumes payload rewards/reference_logps
  -> GRPO objective
```

Remaining production work:

- Add a HuggingFace tokenizer/model loader around the existing model-object API.
- Run CUDA benchmarks with real or tiny HF Reference/Reward models and record
  peak allocated/reserved memory.
- Replace or complement the local paged-KV reservation baseline with a real
  vLLM/generation-engine measurement when that runtime is available locally.
- Optionally extend `OverlapPipeline` from two stages to a native
  rollout -> scoring -> training scheduler. The current `StatelessScoringWorker`
  is intentionally a narrow wrapper that can be composed outside the scheduler.

## Rollback

If Candidate 1 is rejected, revert only issue #47 scoped changes:

- `rl_engine/executors/stateless_executor.py`
- `rl_engine/executors/paged_kv_baseline.py`
- `tests/test_stateless_executor.py`
- `tests/test_paged_kv_baseline.py`
- `benchmarks/benchmark_stateless_executor.py`
- `StatelessScoringWorker` and stateless payload attachment helpers in
  `rl_engine/executors/overlap_pipeline.py`
- issue #47 workspace evidence if explicitly requested.
