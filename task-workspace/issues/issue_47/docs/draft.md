# Plan Draft - Issue #47 Stateless Reference/Reward Forward Engine

## Goal

Implement a stateless scoring executor for Reference and Reward models:

```text
completed prompt/completion batch
  -> full-sequence model forward
  -> selected completion logprobs and/or scalar rewards
  -> detached scoring payload for GRPO
```

The important difference from rollout generation is that this path does not
decode token by token. It should not reserve a paged KV cache for future decode
steps that will never happen.

## Background

Kernel-Align already has generation, objective, and training-side pieces:

- vLLM-style rollout sampling can create multiple completions per prompt.
- reference operators can compute selected token logprobs from logits.
- GRPO can consume reward scores and reference logprobs.
- overlap/training workers can pass rollout payloads into the loss path.

The missing piece is a scoring executor that is shaped like RLHF scoring rather
than serving/generation:

```text
Policy model:   generate new tokens, benefits from KV cache.
Reference model: score existing tokens, no generation loop.
Reward model:   score existing sequences, no generation loop.
```

If Reference and Reward models are run through a generation engine that
pre-allocates paged KV cache, VRAM is spent on a cache that does not help the
single forward pass. That memory could instead increase policy batch size,
sequence length, or number of parallel candidates.

## Candidate Directions

1. **Candidate 1: PyTorch stateless executor wrapper.**
   - Accept an already-loaded model.
   - Run `model(input_ids, attention_mask=..., use_cache=False)`.
   - Compute selected completion logprobs with existing reference ops.
   - Run a reward adapter for scalar rewards.
   - Lowest risk and easiest to test.
2. **Candidate 2: HuggingFace loader and collator.**
   - Add model-name loading, tokenizer padding, dtype/device placement, and
     optional `attn_implementation` selection.
   - Useful, but more integration risk than Candidate 1.
3. **Candidate 3: Optional FlashAttention-2 path.**
   - Prefer dense prefill attention without cache where supported.
   - Should remain optional and skip cleanly when unavailable.
4. **Candidate 4: Integration with overlap pipeline scoring stage.**
   - Insert stateless scoring between rollout and training.
   - Should depend on Candidate 1 public outputs, not private model details.

Recommended starting point: Candidate 1, with enough API shape to avoid painting
the later HuggingFace/FlashAttention integration into a corner.

## Proposed Public API

Add `rl_engine/executors/stateless_executor.py`.

Suggested dataclasses:

```python
@dataclass(frozen=True)
class StatelessForwardConfig:
    mode: Literal["reference", "reward", "both"] = "both"
    use_cache: bool = False
    detach_outputs: bool = True
    return_token_scores: bool = False
    max_batch_size: int | None = None

@dataclass(frozen=True)
class StatelessForwardInputs:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    labels: torch.Tensor | None = None

@dataclass(frozen=True)
class StatelessForwardResult:
    reference_logps: torch.Tensor | None
    rewards: torch.Tensor | None
    token_scores: torch.Tensor | None
    metrics: Mapping[str, float | str]
```

Suggested executor:

```python
class StatelessForwardExecutor:
    def __init__(self, model: torch.nn.Module, config: StatelessForwardConfig, reward_adapter=None): ...

    @torch.no_grad()
    def score(self, inputs: StatelessForwardInputs) -> StatelessForwardResult: ...
```

The executor should keep the first version small. It should not own distributed
serving, Ray actors, or DeepSpeed model initialization.

## Tensor Contract

Preferred dense shape:

```text
batch_size = B
sequence_len = S
completion_len = T, where T <= S

input_ids:        [B, S]
attention_mask:   [B, S]
completion_mask:  [B, S] or [B, T], but the API must choose one shape
labels:           [B, S] optional, token ids to score
logits:           [B, S, vocab_size]
reference_logps:  [B, S] or [B, T], aligned with completion_mask
rewards:          [B]
```

Recommended Candidate 1 simplification:

- Keep `completion_mask` aligned to `[B, S]`.
- Compute selected logprobs for next-token prediction positions.
- Return a `[B, S]` tensor where prompt/pad positions are zeroed or masked by
  `completion_mask`.
- Let downstream GRPO convert to compact completion-only tensors if needed.

## Reference Scoring Semantics

For a causal LM:

```text
input_ids:  [BOS, prompt tokens..., completion tokens...]
logits[t]:  predicts token at t + 1
```

Selected logprob for a completion token should use the logits from the previous
position. For example:

```text
tokens:        [A, B, C, D]
logits index:   0  1  2
label scored:   B  C  D
```

If `C` and `D` are completion tokens, their selected logprobs come from logits at
positions `1` and `2`. Tests should include an explicit toy logits tensor so the
off-by-one behavior is nailed down.

## Reward Scoring Semantics

Reward models vary:

- sequence-classification models may return `logits: [B, 1]`;
- token-classification models may return `scores: [B, S]`;
- causal LMs with a reward head may return hidden-state-derived values.

The executor should not guess all of these in Candidate 1. Instead, accept a
small `reward_adapter(model_outputs, inputs) -> torch.Tensor` callback and
validate that it returns `[B]`.

Default adapter candidates:

```text
outputs.logits shape [B, 1] -> squeeze to [B]
outputs.logits shape [B]    -> use directly
otherwise                   -> raise clear error
```

## How to Avoid Paged KV Cache Allocation

Candidate 1 should enforce the simple local contract:

```python
model(..., use_cache=False)
```

and should not instantiate a generation engine for scoring. Later optional
factory helpers may set:

```text
attn_implementation="flash_attention_2"
```

when supported by the local model stack, but the key requirement is still:

```text
full prefill-style forward, no decode loop, no persistent KV cache.
```

For validation, record:

- `torch.cuda.max_memory_allocated()`
- `torch.cuda.max_memory_reserved()`
- batch size
- sequence length
- dtype
- whether `use_cache` was false
- whether a generation-engine baseline was available

## Integration Path

Candidate 1:

1. Add stateless executor module.
2. Add focused tests with tiny fake models.
3. Validate selected-logprob math against `rl_engine/testing/reference_ops.py`.
4. Validate reward adapter shape checks.
5. Add a smoke benchmark that uses a small local fake or tiny model.

Candidate 2+:

1. Add optional HuggingFace loader.
2. Add CUDA memory benchmark.
3. Add overlap-pipeline scoring stage.
4. Feed `reference_logps` and `rewards` into GRPO training payloads.

## Test Cases

- Reference mode returns selected logprobs only for active completion positions.
- Reward mode returns exactly one scalar per sequence.
- Both mode returns both surfaces in one model forward when possible.
- `use_cache=False` is passed to the model.
- Outputs are detached by default.
- Shape mismatch errors are explicit.
- Padded tokens do not affect reported active-token metrics.
- Importing `rl_engine.executors.stateless_executor` does not require vLLM,
  Ray, DeepSpeed, CUDA extensions, or FlashAttention.

## Validation Commands

```powershell
py -3.13 -m pytest tests/test_stateless_executor.py -q
py -3.13 -m pytest tests/test_stateless_executor.py tests/test_reference_ops.py tests/test_grpo_objective.py -q
py -3.13 -m compileall rl_engine tests benchmarks
```

## Evaluation Commands

```powershell
py -3.13 benchmarks/benchmark_stateless_executor.py --smoke --mode reference
py -3.13 benchmarks/benchmark_stateless_executor.py --smoke --mode reward
```

If CUDA is unavailable or a real model is not installed, record a blocker row
instead of mixing synthetic CPU evidence with GPU memory claims.

## Evidence Required

- Candidate decision in `candidates.jsonl`.
- Test output in `docs/plan.md`.
- Benchmark rows in `benchmark.csv`.
- Optional memory JSON under `outputs/`.
- Any profiler output under `profile/`.
