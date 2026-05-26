from __future__ import annotations

from vllm.logger import init_logger
from vllm.model_executor.models.openpangu import PanguUltraMoEForCausalLM

from .config import normalize_pangu_v2_config

logger = init_logger(__name__)


class PanguV2MoEForCausalLM(PanguUltraMoEForCausalLM):
    """Minimal upstream fallback for import-only checks.

    The real NPU implementation is bundled under `omni_npu` and registered by
    default from `plugin.py`. This class remains available only as an explicit
    `VLLM_PANGU_V2_MODEL_IMPL=fallback` escape hatch when debugging plugin
    loading outside an Ascend runtime.
    """

    def __init__(self, *args, **kwargs):
        vllm_config = kwargs.get("vllm_config")
        if vllm_config is not None:
            hf = vllm_config.model_config.hf_config
            normalize_pangu_v2_config(hf)
            if getattr(hf, "use_mome", False) or getattr(hf, "use_mhc", False):
                logger.warning_once(
                    "Using fallback PanguV2MoEForCausalLM without omni-npu "
                    "DSA/SWA/MoME/MHC kernels. This is useful for import and "
                    "weight-loading smoke tests only; install this plugin "
                    "with its bundled omni_npu package for the full NPU model "
                    "path."
                )
        super().__init__(*args, **kwargs)
