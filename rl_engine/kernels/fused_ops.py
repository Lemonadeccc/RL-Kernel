# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kernel-Align Contributors

import torch
import kernel_align_ops


class FusedLogp:
    @staticmethod
    def apply(logits: torch.Tensor, token_ids: torch.Tensor):
        """
        输入:
            logits: [G*L, V]
            token_ids: [G*L]
        返回:
            log_probs: [G*L]
        """
        orig_shape = logits.shape[:-1]
        logits_2d = logits.view(-1, logits.size(-1))
        token_ids_1d = token_ids.view(-1)
        results = kernel_align_ops.fused_logp(logits_2d, token_ids_1d)
        return results.view(orig_shape)
