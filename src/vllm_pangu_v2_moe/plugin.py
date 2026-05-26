from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Callable

from vllm.logger import init_logger

from .config import (
    PANGU_V2_ARCH,
    PANGU_V2_MODEL_TYPE,
    is_pangu_v2_moe_config,
    normalize_pangu_v2_config,
)

logger = init_logger(__name__)


_PATCHED = False


def _add_omni_src_from_env() -> None:
    omni_src = os.getenv("VLLM_PANGU_V2_OMNI_SRC")
    if omni_src and omni_src not in sys.path:
        sys.path.insert(0, omni_src)


def _choose_model_impl() -> str:
    impl = os.getenv("VLLM_PANGU_V2_MODEL_IMPL", "auto").strip().lower()
    if impl not in {"auto", "omni", "fallback"}:
        raise ValueError(
            "VLLM_PANGU_V2_MODEL_IMPL must be one of auto, omni, fallback; "
            f"got {impl!r}"
        )

    _add_omni_src_from_env()
    omni_available = importlib.util.find_spec("omni_npu") is not None
    if impl == "omni" and not omni_available:
        raise RuntimeError(
            "VLLM_PANGU_V2_MODEL_IMPL=omni was requested but omni_npu is not "
            "importable. Install this plugin package or set "
            "VLLM_PANGU_V2_OMNI_SRC=/path/to/omni-npu/src."
        )

    if impl in {"auto", "omni"} and omni_available:
        logger.info(
            "Registering %s with bundled omni_npu PanguUltraMoE implementation.",
            PANGU_V2_ARCH,
        )
        return "omni_npu.v1.models.pangu.pangu_ultra_moe:PanguUltraMoEForCausalLM"

    logger.warning(
        "omni_npu is not importable; registering %s with fallback OpenPangu "
        "implementation. Full DSA/SWA/MoME/MHC execution requires the bundled "
        "omni_npu package to be importable.",
        PANGU_V2_ARCH,
    )
    return "vllm_pangu_v2_moe.model:PanguV2MoEForCausalLM"


def _patch_arch_convertor() -> None:
    from vllm.transformers_utils import model_arch_config_convertor as convertor

    base = convertor.ModelArchConfigConvertorBase
    old_is_deepseek_mla: Callable[[Any], bool] = base.is_deepseek_mla

    def is_deepseek_mla(self):
        hf = self.hf_text_config
        if getattr(hf, "model_type", None) == PANGU_V2_MODEL_TYPE:
            return getattr(hf, "kv_lora_rank", None) is not None
        return old_is_deepseek_mla(self)

    base.is_deepseek_mla = is_deepseek_mla


def _patch_model_config_init() -> None:
    from vllm.config import model as model_config_module

    model_config_cls = model_config_module.ModelConfig
    old_post_init = model_config_cls.__post_init__

    def __post_init__(self, *args, **kwargs):
        result = old_post_init(self, *args, **kwargs)
        normalize_pangu_v2_config(self.hf_config)
        hf_text = getattr(self, "hf_text_config", None)
        if hf_text is not None:
            normalize_pangu_v2_config(hf_text)
        return result

    model_config_cls.__post_init__ = __post_init__


def _patch_modelslim_mapping() -> None:
    try:
        from vllm_ascend.quantization import modelslim_config
    except Exception as exc:  # pragma: no cover - depends on NPU env
        logger.info("Skipping vllm-ascend modelslim patch: %s", exc)
        return

    mapping = modelslim_config.packed_modules_model_mapping
    if PANGU_V2_MODEL_TYPE not in mapping:
        mapping[PANGU_V2_MODEL_TYPE] = {
            "gate_up_proj": ["gate_proj", "up_proj"],
            "experts": [
                "experts.0.gate_proj",
                "experts.0.up_proj",
                "experts.0.down_proj",
            ],
            "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        }


def _patch_ascend_worker_model_extra_config() -> None:
    try:
        from vllm_ascend.worker.worker import NPUWorker
    except Exception as exc:  # pragma: no cover - depends on NPU env
        logger.info("Skipping vllm-ascend worker patch: %s", exc)
        return

    old_init_device = NPUWorker.init_device

    def init_device(self, *args, **kwargs):
        device = old_init_device(self, *args, **kwargs)
        hf_config = getattr(self.model_config, "hf_config", None)
        if not is_pangu_v2_moe_config(hf_config):
            return device
        if getattr(self, "_pangu_v2_extra_config_loaded", False):
            return device

        try:
            from omni_npu.model_config.config_loader.loader import (
                load_model_extra_config,
            )

            load_model_extra_config(
                self.model_config,
                self.vllm_config,
                self.scheduler_config,
            )
            self._pangu_v2_extra_config_loaded = True
            logger.info("Loaded bundled omni_npu model_extra_config for %s.", PANGU_V2_ARCH)
        except Exception as exc:
            logger.warning(
                "Failed to load bundled omni_npu model_extra_config for %s: %s",
                PANGU_V2_ARCH,
                exc,
            )
        return device

    NPUWorker.init_device = init_device


def _patch_kv_cache_diagnostics() -> None:
    if os.getenv("VLLM_PANGU_V2_KV_DEBUG", "0") != "1":
        return

    def describe_config(kv_cache_config):
        groups = getattr(kv_cache_config, "kv_cache_groups", []) or []
        for idx, group in enumerate(groups):
            spec = getattr(group, "kv_cache_spec", None)
            layer_names = getattr(group, "layer_names", []) or []
            page_size = getattr(spec, "page_size_bytes", None)
            logger.warning(
                "PanguV2 KV group[%d]: spec=%s layers=%d block_size=%s "
                "page_size_bytes=%s",
                idx,
                type(spec).__name__ if spec is not None else None,
                len(layer_names),
                getattr(spec, "block_size", None),
                page_size,
            )

    for module_name, cls_name in (
        ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner"),
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
    ):
        try:
            module = __import__(module_name, fromlist=[cls_name])
            cls = getattr(module, cls_name)
            old = cls.initialize_kv_cache
        except Exception:
            continue

        def initialize_kv_cache(self, kv_cache_config, _old=old):
            describe_config(kv_cache_config)
            return _old(self, kv_cache_config)

        cls.initialize_kv_cache = initialize_kv_cache


def _register_model() -> None:
    from vllm.model_executor.models import ModelRegistry

    model_impl = _choose_model_impl()
    ModelRegistry.register_model(PANGU_V2_ARCH, model_impl)
    # Keep a compatibility alias for the typo observed in the local config.
    ModelRegistry.register_model("PanguV2aMoEForCausalLM", model_impl)


def register() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    _patch_arch_convertor()
    _patch_model_config_init()
    _patch_modelslim_mapping()
    _patch_ascend_worker_model_extra_config()
    _patch_kv_cache_diagnostics()
    _register_model()
    logger.info("Pangu V2 MoE vLLM plugin registered.")
