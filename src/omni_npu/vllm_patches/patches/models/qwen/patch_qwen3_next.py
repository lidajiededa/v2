# This patch is used to hijack the Qwen3Next architecture and apply vLLM core patches.
# OMNI_NPU_VLLM_PATCHES="KVCacheUtilsPatch,SchedulerPatch" before vllm serve

import torch
import numpy as np
from dataclasses import replace
import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
    AttentionSpec,
    MambaSpec,
)
from vllm.v1.request import Request
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.utils.torch_utils import get_dtype_size
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.worker.npu_model_runner import NPUModelRunner


# ==========================================
# THE GOD-MODE REGISTRY HIJACK
# ==========================================
from vllm.model_executor.models.registry import ModelRegistry, _RegisteredModel, _ModelInfo

try:
    # 1. Import your local workspace class
    from omni_npu.v1.models.qwen.qwen3_next import Qwen3NextForCausalLM as LocalQwen3Next

    # 2. Manually construct ModelInfo. 
    # This completely bypasses vLLM's broken `isinstance` checks and missing embed_input_ids errors.
    hardcoded_model_info = _ModelInfo(
        architecture="Qwen3NextForCausalLM",
        is_text_generation_model=True,  # <--- FORCES VLLM TO ACCEPT IT
        is_pooling_model=False,
        attn_type="decoder",
        default_seq_pooling_type="LAST",
        default_tok_pooling_type="ALL",
        supports_cross_encoding=False,
        supports_multimodal=False,
        supports_multimodal_raw_input_only=False,
        requires_raw_input_tokens=False,
        supports_multimodal_encoder_tp_data=False,
        supports_pp=True,
        has_inner_state=True,
        is_attention_free=False,
        is_hybrid=True,
        has_noops=False,
        supports_mamba_prefix_caching=False,
        supports_transcription=False,
        supports_transcription_only=False
    )

    # 3. Create the registered model object natively
    registered_model = _RegisteredModel(
        interfaces=hardcoded_model_info, 
        model_cls=LocalQwen3Next
    )

    # 4. Inject it directly into the exact dictionary vLLM uses
    ModelRegistry.models["Qwen3NextForCausalLM"] = registered_model
    print("[Omni-NPU] HARD HIJACK SUCCESSFUL: Qwen3NextForCausalLM is now using the local workspace.")

except Exception as e:
    print(f"[Omni-NPU] CRITICAL FAIL during Hard Hijack: {e}")
# ==========================================


@register_patch("KVCacheUtilsPatch", kv_cache_utils)
class KVCacheUtilsPatch(VLLMPatch):
    _attr_names_to_apply = ['get_kv_cache_config_from_groups', 'unify_hybrid_kv_cache_specs']

    def get_kv_cache_config_from_groups(
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
        available_memory: int,
    ) -> KVCacheConfig:
        if len(kv_cache_groups) == 0:
            return KVCacheConfig(
                num_blocks=1,
                kv_cache_tensors=[],
                kv_cache_groups=kv_cache_groups,
            )

        if len(kv_cache_groups) == 1 and isinstance(
            kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
        ):
            num_blocks = (
                available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
            )
            num_blocks = kv_cache_utils.may_override_num_blocks(vllm_config, num_blocks)
            per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
            kv_cache_tensors = [
                KVCacheTensor(
                    size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                    shared_by=[layer_name],
                )
                for layer_name in kv_cache_groups[0].layer_names
            ]
        else:
            group_size = max(len(group.layer_names) for group in kv_cache_groups)
            page_size = kv_cache_utils.get_uniform_page_size(
                [group.kv_cache_spec for group in kv_cache_groups]
            )
            assert group_size > 0, "group_size must be greater than 0"
            num_blocks = kv_cache_utils.get_num_blocks(
                vllm_config, group_size, available_memory, page_size
            )
            
            # patch begin
            num_blocks //= len(kv_cache_groups)
            kv_cache_tensors = []
            for i in range(group_size):
                for j in range(len(kv_cache_groups)):
                    kv_cache_spec = kv_cache_groups[j].kv_cache_spec
                    if isinstance(kv_cache_spec, MambaSpec):
                        kv_cache_spec = replace(kv_cache_spec, page_size_padded=None)
                    page_size = kv_cache_spec.page_size_bytes
                    kv_cache_tensors.append(
                        KVCacheTensor(size=page_size * num_blocks, shared_by=[kv_cache_groups[j].layer_names[i]])
                    )
            # patch end

        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
        )

    def unify_hybrid_kv_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]):
        pass

@register_patch("SchedulerPatch", Scheduler)
class SchedulerPatch(VLLMPatch):
    _attr_names_to_apply = ['_update_waiting_for_remote_kv']

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        if request.request_id in self.failed_recving_kv_req_ids:
            if request.num_computed_tokens:
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                self.kv_cache_manager.free(request)
            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # patch begin
            block_ids = self.kv_cache_manager.get_block_ids(request.request_id)
            if len(block_ids) == 1:
                block_ids = block_ids[0]
            else:
                for idx, group in enumerate(self.kv_cache_manager.kv_cache_config.kv_cache_groups):
                    if isinstance(group.kv_cache_spec, AttentionSpec):
                        block_ids = block_ids[idx]
                        break
            # patch end
            
            num_computed_tokens = len(block_ids) * self.block_size
            num_computed_tokens = min(num_computed_tokens, request.num_tokens)
            if num_computed_tokens == request.num_tokens:
                num_computed_tokens -= 1
            self.kv_cache_manager.cache_blocks(request, num_computed_tokens)
            request.num_computed_tokens = num_computed_tokens

        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True

@register_patch("Qwen3NextReshapeKVCachePatch", NPUModelRunner)
class NPUModelRunnerPatch(VLLMPatch):
    _attr_names_to_apply = ['_reshape_kv_cache_tensors']

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        kv_caches: dict[str, torch.Tensor] = {}
        has_tensor, has_tuple = False, False
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                
                # 1. Intercept and remove padding BEFORE the assertion
                if isinstance(kv_cache_spec, MambaSpec):
                    kv_cache_spec = replace(kv_cache_spec, page_size_padded=None)

                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0, \
                    f"{kv_cache_spec=}, {raw_tensor.numel()=}, {kv_cache_spec.page_size_bytes=}"
                
                num_blocks = (raw_tensor.numel() // kv_cache_spec.page_size_bytes)

                # 2. Bypass the missing backend method for Mamba/GDN
                if isinstance(kv_cache_spec, AttentionSpec):
                    kwargs = {}
                    kv_cache_tensors = attn_backend.reshape_kv_cache(
                        raw_tensor, num_blocks, kv_cache_spec, **kwargs
                    )
                    kv_caches[layer_name] = kv_cache_tensors
                    if isinstance(kv_cache_tensors, torch.Tensor) and kv_cache_tensors.is_contiguous():
                        has_tensor = True

                elif isinstance(kv_cache_spec, MambaSpec):
                    state_tensors = []
                    offset = 0
                    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                        dtype_size = get_dtype_size(dtype)
                        target_shape = (num_blocks, *shape)
                        size_bytes = int(np.prod(target_shape)) * dtype_size
                        final_tensor = raw_tensor[offset:offset + size_bytes].view(dtype).view(target_shape)
                        offset += size_bytes
                        assert final_tensor.is_contiguous()
                        state_tensors.append(final_tensor)

                    kv_caches[layer_name] = tuple(state_tensors)
                    has_tuple = True
                else:
                    raise NotImplementedError(f"Unsupported kv_cache_spec type: {type(kv_cache_spec)}")

        if has_tensor and has_tuple:
            self._update_hybrid_attention_mamba_layout(kv_caches)

        return kv_caches