# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch
import torch.nn.functional as F

from rl_engine.testing import (
    active_token_count,
    compute_policy_ratio,
    compute_reference_kl,
    masked_mean,
    masked_sum,
    rms_norm_reference,
    selected_logprobs_reference,
    summarize_kernel_drift,
)


def test_selected_logprobs_reference_matches_pytorch():
    logits = torch.tensor([[[1.0, 2.0, 3.0], [0.5, -0.5, 1.5]]])
    token_ids = torch.tensor([[2, 0]])

    actual = selected_logprobs_reference(logits, token_ids)
    expected = (
        torch.log_softmax(logits.float(), dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    )

    assert torch.allclose(actual, expected)
    assert actual.dtype == torch.float32


def test_selected_logprobs_reference_mask_and_dtype():
    logits = torch.randn(2, 3, 5)
    token_ids = torch.tensor([[0, 1, 2], [3, 4, 0]])
    mask = torch.tensor([[True, False, True], [False, True, True]])

    actual = selected_logprobs_reference(logits, token_ids, mask=mask, output_dtype=torch.float16)

    assert actual.dtype == torch.float16
    assert torch.equal(actual[~mask], torch.zeros_like(actual[~mask]))


def test_selected_logprobs_reference_temperature():
    logits = torch.tensor([[1.0, 3.0, -1.0]])
    token_ids = torch.tensor([1])

    actual = selected_logprobs_reference(logits, token_ids, temperature=0.5)
    expected = (
        torch.log_softmax(logits.float() / 0.5, dim=-1)
        .gather(-1, token_ids.unsqueeze(-1))
        .squeeze(-1)
    )

    assert torch.allclose(actual, expected)


def test_selected_logprobs_reference_rejects_bad_temperature():
    with pytest.raises(ValueError, match="temperature"):
        selected_logprobs_reference(torch.randn(1, 3), torch.tensor([0]), temperature=0.0)


def test_masked_reductions_ignore_inactive_tokens():
    values = torch.tensor([[1.0, 100.0], [3.0, 4.0]])
    mask = torch.tensor([[True, False], [True, True]])

    assert torch.equal(active_token_count(mask), torch.tensor(3.0))
    assert torch.equal(masked_sum(values, mask), torch.tensor(8.0))
    assert torch.allclose(masked_mean(values, mask), torch.tensor(8.0 / 3.0))


def test_ratio_kl_and_drift_helpers():
    current = torch.tensor([[0.0, -1.0], [-2.0, -3.0]])
    old = torch.tensor([[-0.5, -1.5], [-2.5, -3.5]])
    ref = torch.tensor([[-0.25, -1.25], [-2.25, -3.25]])
    mask = torch.tensor([[True, False], [True, True]])

    ratio = compute_policy_ratio(current, old, mask)
    kl = compute_reference_kl(current, ref, mask)

    assert torch.equal(ratio[~mask], torch.zeros_like(ratio[~mask]))
    assert torch.equal(kl[~mask], torch.zeros_like(kl[~mask]))

    summary = summarize_kernel_drift(current + 0.1, current, mask)

    assert summary["active_count"] == 3
    assert summary["max_abs_error"] == pytest.approx(0.1, rel=1e-6)
    assert summary["mean_abs_error"] == pytest.approx(0.1, rel=1e-6)


def test_rms_norm_reference_match_pytorch():
    torch.manual_seed(0)
    d = 16
    x = torch.randn(4, 7, d)
    weight = torch.randn(d)
    eps = 1e-5

    actual = rms_norm_reference(x, weight, eps)
    expected = F.rms_norm(x, (d,), weight, eps)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert actual.dtype == torch.float32


def test_rms_norm_reference_fp32_accumulation():
    torch.manual_seed(0)
    d = 16
    x = torch.randn(3, d)
    weight = torch.randn(d)
    eps = 1e-5

    for dtype in (torch.float16, torch.bfloat16):
        x_low, weight_low = x.to(dtype), weight.to(dtype)
        actual = rms_norm_reference(x_low, weight_low, eps)
        # Computing in fp32 then casting back must match the low-precision path.
        expected = rms_norm_reference(x_low.float(), weight_low.float(), eps).to(dtype)

        assert actual.dtype == dtype
        assert torch.equal(actual, expected)


def test_rms_norm_reference_output_dtype():
    x = torch.randn(2, 8)
    weight = torch.randn(8)

    actual = rms_norm_reference(x, weight, eps=1e-5, output_dtype=torch.float16)

    assert actual.dtype == torch.float16


def test_rms_norm_reference_gradcheck():
    torch.manual_seed(0)
    d = 8
    x = torch.randn(3, d, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(d, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(lambda a, b: rms_norm_reference(a, b, 1e-5), (x, weight))


def test_rms_norm_reference_rejects_bad_input():
    x = torch.randn(2, 8)
    weight = torch.randn(8)

    with pytest.raises(ValueError, match="eps"):
        rms_norm_reference(x, weight, eps=0.0)

    with pytest.raises(ValueError, match="weight"):
        rms_norm_reference(x, torch.randn(9), eps=1e-5)
