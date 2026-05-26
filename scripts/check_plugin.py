#!/usr/bin/env python3
from __future__ import annotations

import os

os.environ.setdefault("VLLM_PANGU_V2_KV_DEBUG", "1")

from vllm.model_executor.models import ModelRegistry

import vllm_pangu_v2_moe

vllm_pangu_v2_moe.register()
print("PanguV2MoEForCausalLM" in ModelRegistry.get_supported_archs())
print("PanguV2aMoEForCausalLM" in ModelRegistry.get_supported_archs())

