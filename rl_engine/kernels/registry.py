# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kernel-Align Contributors

import importlib
from enum import Enum, EnumMeta
from typing import Optional, Dict, Any, Type, Set
from rl_engine.platforms.device import device_ctx
from rl_engine.utils.logger import logger


class _KernelEnumMeta(EnumMeta):
    """Metaclass to provide enhanced error messaging for backend lookups."""

    def __getitem__(cls, name: str):
        try:
            return super().__getitem__(name)
        except KeyError as e:  # 修复 B904: 捕获异常对象 e
            valid_ops = ", ".join(cls.__members__.keys())
            raise ValueError(f"Operator '{name}' not found. Supported backends: {valid_ops}") from e


class OpBackend(Enum, metaclass=_KernelEnumMeta):
    # NVIDIA optimized stack
    FLASH_ATTN = "rl_engine.kernels.cuda.flash_attn.FlashAttentionOp"
    FLASHINFER = "rl_engine.kernels.cuda.flashinfer.FlashInferOp"

    # AMD ROCm optimized stack
    ROCM_AITER = "rl_engine.kernels.rocm.aiter.AiterOp"
    ROCM_CK = "rl_engine.kernels.rocm.composable_kernel.CKOp"

    # Generic fallback
    TRITON_GENERIC = "rl_engine.kernels.triton.generic.TritonOp"
    PYTORCH_NATIVE = "rl_engine.kernels.native.pytorch_op.NativeOp"


class KernelRegistry:
    """
    Central dispatcher for high-performance kernels.
    Handles dynamic routing between ROCm and CUDA backends at runtime.
    """

    def __init__(self):
        # 优化：单例实例缓存，消除高频训练循环中重复创建算子对象的 GC 开销
        self._instance_cache: Dict[str, Any] = {}
        # 优化：负面缓存（黑名单），因环境缺失 import 失败的后端直接拉黑，避免高频触发 try...except
        self._failed_backends: Set[str] = set()

        self._priority_map = {
            "cuda": {
                "logp": [OpBackend.FLASHINFER, OpBackend.TRITON_GENERIC, OpBackend.PYTORCH_NATIVE],
                "attn": [OpBackend.FLASH_ATTN, OpBackend.TRITON_GENERIC],
            },
            "rocm": {
                "logp": [OpBackend.ROCM_AITER, OpBackend.TRITON_GENERIC, OpBackend.PYTORCH_NATIVE],
                "attn": [OpBackend.TRITON_GENERIC],
            },
        }
        logger.info(f"KernelRegistry initialized for {device_ctx.device_type}")

    def get_op(self, op_type: str) -> Any:
        """Core distribution logic: Automatically select the best operator
        based on hardware and priority.
        """  # 修复 E501: 将超过 100 字符的单行 Docstring 折行
        platform = "rocm" if device_ctx.is_rocm else "cuda"
        candidates = self._priority_map.get(platform, {}).get(op_type, [OpBackend.PYTORCH_NATIVE])

        for backend in candidates:
            # 1. 命中单例缓存直接返回
            if backend.name in self._instance_cache:
                return self._instance_cache[backend.name]

            # 2. 命黑名单则直接跳过，守护 CPU 周期
            if backend.name in self._failed_backends:
                continue

            op_class = self._load_backend(backend)
            if op_class:
                try:
                    op_instance = op_class()
                    self._instance_cache[backend.name] = op_instance
                    return op_instance
                except Exception as e:
                    logger.error(f"Failed to instantiate {backend.name}: {e}")
                    self._failed_backends.add(backend.name)
            else:
                self._failed_backends.add(backend.name)

        raise RuntimeError(f"No functional backend found for {op_type} on {platform}")

    def _load_backend(self, backend: OpBackend) -> Optional[Type]:
        """Dynamic loading technique: Import modules only when needed
        and check environment dependencies.
        """  # 修复 E501: 将超过 100 字符的单行 Docstring 折行
        module_path, class_name = backend.value.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError, ModuleNotFoundError) as e:
            # 优化：工程级 Bug 隔离
            # 如果报错的缺失模块属于项目自身路径，说明是我们自己写的代码写错了，直接 raise 暴露
            missing_module = str(e.name) if hasattr(e, "name") else ""
            if missing_module and (missing_module in module_path or "rl_engine" in missing_module):
                logger.critical(f"Internal wrapper implementation bug in '{module_path}': {e}")
                raise e

            # 如果只是纯粹缺失第三方芯片硬加速库，则打印 warning 正常触发 fallback
            logger.warning(f"Backend {backend.name} unavailable: {e}. Falling back...")
            return None


kernel_registry = KernelRegistry()
