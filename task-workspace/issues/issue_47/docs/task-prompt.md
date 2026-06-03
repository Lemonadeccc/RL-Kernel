# Kernel Design Agents - Kernel-Align Task Prompt

Use `task-workspace/issues/issue_47/` for all plans, evidence, benchmark rows,
profiler notes, and iteration records.

This prompt is for GitHub issue **#47**:

```text
[FEAT][executors]: implement stateless (Zero-KV-Cache) forward engine for Reference and Reward models
```

## Task Contract

- **Task name:** Issue #47 - Stateless forward executor for reference and reward
  scoring.
- **Objective:** Add a lightweight executor path for Reference and Reward
  models that runs a single full-sequence forward pass without allocating a
  paged KV cache, so RLHF scoring leaves more VRAM available for the Policy
  model batch.
- **Problem statement:** Reference and Reward models in RLHF pipelines do not
  perform autoregressive decoding for their scoring step. They consume complete
  prompt/completion sequences and return logprobs, scalar rewards, or token-level
  scores. Using a generation-oriented engine such as full vLLM for this path can
  pre-allocate large paged KV cache buffers that are unused by the scoring
  workload, reducing the batch size available to the trainable policy model.
- **Current baseline to verify before planning:**
  - `rl_engine/executors/vllm_sampler.py` owns generation-oriented rollout
    sampling and may depend on a runtime that is optimized around KV cache.
  - `rl_engine/executors/overlap_pipeline.py` coordinates rollout/training
    payload flow and should be able to call a scoring executor without importing
    vLLM at module import time.
  - `rl_engine/executors/deepspeed_trainer.py` consumes old/reference logprobs
    and rewards for GRPO training once they are present in rollout payloads.
  - `rl_engine/testing/reference_ops.py` provides CPU/PyTorch reference
    selected-logprob helpers that can validate the stateless executor output.
  - Issue #16 supplies the GRPO objective that consumes reward scores and
    reference selected logprobs.
  - Issue #63 and #64 extend realistic GRPO integration and validation; issue
    #47 should provide the memory-efficient scoring path those issues can use.
- **Correctness requirements:**
  - The scoring engine must run full-sequence forward passes with no generation
    loop and no persistent KV cache allocation.
  - It must expose a clear mode for Reference scoring, Reward scoring, or both
    when a model supports both surfaces.
  - Reference-mode output must include selected token logprobs for completion
    tokens and preserve the same mask semantics used by GRPO.
  - Reward-mode output must include one scalar reward per sequence, or an
    explicit adapter contract for converting model outputs into scalar rewards.
  - Prompt tokens and padded completion tokens must not be reported as active
    completion scoring positions.
  - Old-policy/reference logprobs and reward values must be detached before they
    enter the training objective.
  - The executor must fail explicitly when `input_ids`, `attention_mask`,
    `completion_mask`, labels, or model outputs disagree in shape.
  - CPU tests must validate exact PyTorch reference math. CUDA tests should
    validate the same public API when a GPU model/runtime is available.
- **Performance and memory requirements:**
  - Bypass standard Paged Attention and paged KV-cache allocation for scoring.
  - Prefer standard dense attention or FlashAttention-2 prefill-style kernels
    without caching when available.
  - Record peak allocated and reserved memory for stateless scoring and, when
    available, compare against a generation-engine baseline.
  - Keep import-time dependencies light. vLLM, DeepSpeed, Ray, and optional
    FlashAttention packages must not be required just to import the executor
    module.
- **Production-alignment requirements:**
  - Add a small public executor module, preferably
    `rl_engine/executors/stateless_executor.py`.
  - Keep tokenizer/model loading pluggable so the executor can wrap a HuggingFace
    model, a local model object, or a future engine adapter.
  - Make batch collation explicit: sequence ids, completion masks, device,
    dtype, max length, truncation behavior, and padding side.
  - Provide metrics for batch size, sequence length, active completion tokens,
    mode, dtype, device, elapsed time, and peak memory.
  - Keep scoring evidence separate from training evidence in issue files.
- **Relationship to issue #16:** Issue #16 consumes reference logprobs and reward
  scores in the GRPO objective. Issue #47 supplies a memory-efficient way to
  produce those inputs.
- **Relationship to issue #11:** Issue #11 generates grouped rollout candidates.
  Issue #47 scores those completed candidates; it does not sample new tokens.
- **Relationship to issue #18:** Issue #18 schedules rollout and training. Issue
  #47 should fit into the same pipeline as a scoring stage between rollout and
  objective computation.
- **Relationship to issue #63/#64:** Issue #63 and #64 need realistic reward and
  reference inputs for production-shaped GRPO validation. Issue #47 should make
  those inputs cheaper to obtain.
- **Out of scope for Candidate 1:**
  - Training a reward model.
  - Distributed reference/reward serving.
  - Multi-node scheduling.
  - Custom CUDA attention kernels.
  - Replacing the policy generation engine.
  - Implementing a full vLLM fork.

## Expected Public Surface

Prefer a compact module such as `rl_engine/executors/stateless_executor.py` with:

- `StatelessForwardConfig`
- `StatelessForwardInputs`
- `StatelessForwardResult`
- `StatelessForwardExecutor`
- `score_reference_logprobs(...)`
- `score_rewards(...)`

The first implementation can accept an already-loaded PyTorch model. A follow-up
can add factory helpers for HuggingFace model names, quantized loading, tensor
parallel wrappers, or remote scoring services.

## Validation Commands

```powershell
py -3.13 -m pytest tests/test_stateless_executor.py -q
py -3.13 -m pytest tests/test_stateless_executor.py tests/test_grpo_objective.py tests/test_deepspeed_training_worker.py tests/test_overlap_pipeline.py -q
py -3.13 -m compileall rl_engine tests benchmarks
```

If CUDA and optional FlashAttention-backed model execution are available, also
run a GPU smoke test and record memory deltas:

```powershell
py -3.13 benchmarks/benchmark_stateless_executor.py --smoke --mode reference
py -3.13 benchmarks/benchmark_stateless_executor.py --smoke --mode reward
```

If a benchmark mode cannot run in the current environment, append an exact
blocker row to `benchmark.csv` and keep CPU reference evidence separate from
CUDA evidence.

## Evaluation Commands

```powershell
py -3.13 benchmarks/benchmark_stateless_executor.py --mode reference --batch-size 8 --seq-len 1024
py -3.13 benchmarks/benchmark_stateless_executor.py --mode reward --batch-size 8 --seq-len 1024
```

When a generation-engine baseline is available, compare peak memory against a
paged-KV path and record whether the stateless path avoids KV cache reservation.

## Workspace Paths

- Plan draft: `task-workspace/issues/issue_47/docs/draft.md`
- Executable plan: `task-workspace/issues/issue_47/docs/plan.md`
- Benchmark log: `task-workspace/issues/issue_47/benchmark.csv`
- Candidates: `task-workspace/issues/issue_47/candidates.jsonl`
- Profiler output: `task-workspace/issues/issue_47/profile/`
- Run artifacts: `task-workspace/issues/issue_47/runs/`
- Output artifacts: `task-workspace/issues/issue_47/outputs/`

## Workflow

1. Read the current rollout, training, and reference-op baseline:
   - `rl_engine/executors/vllm_sampler.py`
   - `rl_engine/executors/overlap_pipeline.py`
   - `rl_engine/executors/deepspeed_trainer.py`
   - `rl_engine/testing/reference_ops.py`
   - `rl_engine/alignment/grpo.py`
   - `tests/test_reference_ops.py`
   - `tests/test_grpo_objective.py`
2. Define the stateless scoring contract: model inputs, completion masks,
   reference selected logprobs, scalar rewards, detach semantics, and metrics.
3. Draft candidate options in `docs/draft.md`.
4. Promote one candidate into `docs/plan.md`.
5. Implement CPU/PyTorch reference scoring first.
6. Add optional CUDA/FlashAttention-backed execution only after reference tests
   pass.
7. Wire the executor into the overlap/training path only through an import-light
   interface.
8. Record validation, benchmarks, memory evidence, and blockers under issue #47.

## Plan Draft Requirements

Include:

- Why Reference/Reward scoring is different from policy generation.
- Exact tensor shape and mask contract.
- How the executor avoids paged KV-cache allocation.
- Public API proposal and dependency boundaries.
- Reference logprob scoring semantics.
- Reward scoring adapter semantics.
- Worker/pipeline integration path.
- Validation and benchmark commands.
- Evidence needed to promote, revise, or reject the candidate.
