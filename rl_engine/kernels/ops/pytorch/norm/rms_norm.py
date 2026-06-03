# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch


class NativeRMSNormOp:
    """Pure PyTorch native fallback for fused RMSNorm.

    Computes ``x / sqrt(mean(x**2, dim=-1) + eps) * weight`` using only
    primitive autograd-aware ops, so the backward pass is provided for free by
    PyTorch autograd. This is the portable fallback the kernel registry routes
    to when the CUDA / Triton backends are unavailable, and it doubles as the
    numerical baseline the fused kernels are validated against.
    """

    def __init__(self) -> None:
        # Native fallback has no external dependencies, so nothing to set up.
        pass

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        return self.apply(x, weight, eps)

    def _rms_norm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Core RMSNorm math shared by every public entry point.

        The mean-square is accumulated in at least float32 for numerical
        stability (float64 inputs are preserved so this stays gradcheck-safe),
        then the result is cast to ``output_dtype``.
        """
        if eps <= 0.0:
            raise ValueError("eps must be greater than zero")
        if weight.shape != x.shape[-1:]:
            raise ValueError(
                f"weight shape {tuple(weight.shape)} must match the last dim of x "
                f"{tuple(x.shape[-1:])}"
            )

        # Promote half precision to fp32 for the reduction; keep fp64 as fp64.
        compute_dtype = torch.promote_types(x.dtype, torch.float32)

        x_c = x.to(compute_dtype)
        variance = x_c.pow(2).mean(dim=-1, keepdim=True)  # E[x^2] over last dim
        normed = x_c * torch.rsqrt(variance + eps)  # eps lives inside the sqrt
        out = normed * weight.to(compute_dtype)
        return out.to(output_dtype)

    def _validate_output_shape(self, output: torch.Tensor, x: torch.Tensor) -> None:
        # RMSNorm is elementwise, so the output keeps the full input shape
        # (unlike logp, which reduces the last dim away).
        if output.shape != x.shape:
            raise ValueError(
                f"output shape {tuple(output.shape)} must match x shape "
                f"{tuple(x.shape)}"
            )

    def apply(
        self, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """Forward pass returning a freshly allocated tensor in x's dtype."""
        return self._rms_norm(x, weight, eps, output_dtype=x.dtype)

    def out(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Zero-allocation variant writing the result into ``output``."""
        self._validate_output_shape(output, x)
        result = self._rms_norm(x, weight, eps, output_dtype=output.dtype)
        output.copy_(result)
        return output
