# This patch is used to hijack the Qwen3Next architecture for Qwen3.5.
# OMNI_NPU_VLLM_PATCHES="SchedulerPatch,KVCacheUtilsPatch,Qwen3NextReshapeKVCachePatch,Qwen3NextAttentionMetadataBuilderPatch,Qwen3NextAttentionBackendPatch" before vllm serve

import torch
import numpy as np
from dataclasses import replace
import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.config import VllmConfig
from vllm.utils.torch_utils import get_dtype_size
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
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.worker.npu_model_runner import NPUModelRunner
from omni_npu.attention.backends.attention import (
    NPUAttentionMetadataBuilder, 
    NPUMetadata, 
    NPUAttentionBackendImpl
)

# ==========================================
# THE GOD-MODE REGISTRY HIJACK
# ==========================================
from vllm.model_executor.models.registry import ModelRegistry, _RegisteredModel, _ModelInfo

try:
    from omni_npu.v1.models.qwen.qwen3_next import Qwen3NextForCausalLM as LocalQwen3Next

    hardcoded_model_info = _ModelInfo(
        architecture="Qwen3NextForCausalLM",
        is_text_generation_model=True,
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

    registered_model = _RegisteredModel(
        interfaces=hardcoded_model_info, 
        model_cls=LocalQwen3Next
    )

    ModelRegistry.models["Qwen3NextForCausalLM"] = registered_model
    print("[Omni-NPU] HARD HIJACK SUCCESSFUL: Qwen3NextForCausalLM (Qwen3.5) is active.")

except Exception as e:
    print(f"[Omni-NPU] CRITICAL FAIL during Hard Hijack: {e}")


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


@register_patch("KVCacheUtilsPatch", kv_cache_utils)
class KVCacheUtilsPatch(VLLMPatch):
    _attr_names_to_apply = ['get_kv_cache_config_from_groups', 'unify_hybrid_kv_cache_specs']

    @staticmethod
    def get_kv_cache_config_from_groups(
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
        available_memory: int,
    ) -> KVCacheConfig:
        
        if len(kv_cache_groups) == 0:
            return KVCacheConfig(num_blocks=1, kv_cache_tensors=[], kv_cache_groups=[])

        if len(kv_cache_groups) == 1 and isinstance(kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs):
            num_blocks = available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
            num_blocks = kv_cache_utils.may_override_num_blocks(vllm_config, num_blocks)
            per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
            kv_cache_tensors = [
                KVCacheTensor(size=per_layer_specs[layer_name].page_size_bytes * num_blocks, shared_by=[layer_name])
                for layer_name in kv_cache_groups[0].layer_names
            ]
            return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=kv_cache_tensors, kv_cache_groups=kv_cache_groups)

        group_size = max(len(group.layer_names) for group in kv_cache_groups)
        assert group_size > 0, "group_size must be greater than 0"
        
        gdn_groups = [g for g in kv_cache_groups if isinstance(g.kv_cache_spec, MambaSpec)]
        attn_groups = [g for g in kv_cache_groups if isinstance(g.kv_cache_spec, AttentionSpec)]
        
        gdn_page_size = 0
        if gdn_groups:
            spec = gdn_groups[0].kv_cache_spec
            if isinstance(spec, MambaSpec):
                spec = replace(spec, page_size_padded=None)
            gdn_page_size = spec.page_size_bytes
                
        attn_page_size = sum(g.kv_cache_spec.page_size_bytes for g in attn_groups)
        
        # patch begin
        # Dummy State Filter for Monolithic Tensor Coupling
        for g in kv_cache_groups:
            if isinstance(g.kv_cache_spec, MambaSpec):
                manba_spec = g.kv_cache_spec
        cnt_eq = 0
        for shape, dtype in zip(manba_spec.shapes, manba_spec.dtypes):
            dtype_size = get_dtype_size(dtype)
            size_bytes = int(np.prod(shape)) * dtype_size
            if size_bytes == attn_page_size // 2:
                cnt_eq += 1
            else:
                gdn_page_size = size_bytes
        
        assert cnt_eq == 1
        
        total_page_size = (gdn_page_size * group_size) + (attn_page_size * group_size)
        
        if total_page_size > 0:
            num_blocks = available_memory // total_page_size
        else:
            num_blocks = 1024
            
        num_blocks = kv_cache_utils.may_override_num_blocks(vllm_config, num_blocks)
        kv_cache_tensors = []
        
        # Monolithic Tensor Coupling
        for i in range(group_size):
            gdn_layer_names = [g.layer_names[i] for g in gdn_groups]
            attn_layer_names = [g.layer_names[i] for g in attn_groups]
            
            shared_by = gdn_layer_names + [attn_groups[0].layer_names[i]]
            tensor = KVCacheTensor(
                        size=(gdn_page_size + attn_page_size) * num_blocks,
                        shared_by=shared_by 
                    )
            kv_cache_tensors.append(tensor)
        # patch end

        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
        )

    def unify_hybrid_kv_cache_specs(self, kv_cache_spec: dict[str, KVCacheSpec]):
        pass


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

        # patch begin
        attn_page_size = 0
        for group in self._kv_cache_spec_attn_group_iterator():
            if isinstance(group.kv_cache_spec, AttentionSpec):
                attn_page_size = group.kv_cache_spec.page_size_bytes
                break
                
        for group in self._kv_cache_spec_attn_group_iterator():
            if isinstance(group.kv_cache_spec, MambaSpec):
                manba_spec = group.kv_cache_spec
                cnt_eq = 0
                for shape, dtype in zip(manba_spec.shapes, manba_spec.dtypes):
                    dtype_size = get_dtype_size(dtype)
                    size_bytes = int(np.prod(shape)) * dtype_size
                    if size_bytes == attn_page_size // 2:
                        cnt_eq += 1
                    else:
                        gdn_page_size = size_bytes
                assert cnt_eq == 1
                break
        # patch end

        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            
            for layer_name in group.layer_names:
                if layer_name in getattr(self, 'runner_only_attn_layers', []):
                    continue

                raw_tensor = kv_cache_raw_tensors[layer_name]
                num_blocks = kv_cache_config.num_blocks

                if isinstance(kv_cache_spec, AttentionSpec):
                    # patch begin
                    slice_end = kv_cache_spec.page_size_bytes * num_blocks
                    kv_cache_tensors = attn_backend.reshape_kv_cache(
                        raw_tensor[0:slice_end], num_blocks, kv_cache_spec
                    )
                    # patch end
                    
                    kv_caches[layer_name] = kv_cache_tensors
                    if isinstance(kv_cache_tensors, torch.Tensor) and kv_cache_tensors.is_contiguous():
                        has_tensor = True

                elif isinstance(kv_cache_spec, MambaSpec):
                    # patch begin
                    state_tensors = []
                    offset = 0 
                    for shape, dtype in zip(reversed(kv_cache_spec.shapes), reversed(kv_cache_spec.dtypes)):
                        dtype_size = get_dtype_size(dtype)
                        target_shape = (num_blocks, *shape)
                        size_bytes = int(np.prod(target_shape)) * dtype_size
                        
                        final_tensor = raw_tensor[offset:offset + size_bytes].view(dtype).view(target_shape)
                        offset += size_bytes * 2
                        assert final_tensor.is_contiguous()
                        state_tensors.append(final_tensor)
                    kv_caches[layer_name] = tuple(reversed(state_tensors))
                    # patch end
                    has_tuple = True
                else:
                    raise NotImplementedError(f"Unsupported type: {type(kv_cache_spec)}")

        if has_tensor and has_tuple:
            if hasattr(self, '_update_hybrid_attention_mamba_layout'):
                self._update_hybrid_attention_mamba_layout(kv_caches)

        return kv_caches


# ==========================================
# HARDWARE LIMIT SPOOFING PATCHES
# ==========================================
_vanilla_init = NPUAttentionMetadataBuilder.__init__
_vanilla_build = NPUAttentionMetadataBuilder.build

@register_patch("Qwen3NextAttentionMetadataBuilderPatch", NPUAttentionMetadataBuilder)
class Qwen3NextAttentionMetadataBuilderPatch(VLLMPatch):
    _attr_names_to_apply = ['__init__', 'build']

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        _vanilla_init(self, kv_cache_spec, layer_names, vllm_config, device)
        if getattr(self, 'block_size', 0) > 512:
            self.kernel_block_tables = None

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        max_query_len = common_attn_metadata.max_query_len
        
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
            )
        )
        
        attn_metadata = NPUMetadata(
            num_actual_tokens=num_actual_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc.tolist(),
            seq_lens=seq_lens.tolist(),
            max_query_len=max_query_len,
            slot_mapping=slot_mapping,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            decode_threshold=self.reorder_batch_threshold
        )
        
        # patch begin
        block_size = getattr(self, 'block_size', getattr(getattr(self, 'kv_cache_spec', None), 'block_size', 0))
        if block_size > 512:
            assert block_size % 384 == 0
            ratio = block_size // 384
            
            if block_table is not None and block_table.numel() > 0:
                target_shape = (*block_table.shape[:-1], -1)
                bt_scaled = (block_table * ratio).unsqueeze(-1)
                offsets = torch.arange(ratio, device=block_table.device)
                new_block_table = (bt_scaled + offsets).reshape(target_shape).to(torch.int32)
            else:
                device = block_table.device if block_table is not None else getattr(self, 'device', 'cpu')
                new_block_table = torch.empty(0, dtype=torch.int32, device=device)
            
            if getattr(self, 'kernel_block_tables', None) is None:
                attn_metadata.kernel_block_tables = new_block_table
                self.kernel_block_tables = new_block_table
            else:
                limit = new_block_table.shape[0]
                self.kernel_block_tables[:limit].copy_(new_block_table)
                attn_metadata.kernel_block_tables = self.kernel_block_tables[:limit]
        # patch end

        return attn_metadata


_vanilla_forward = NPUAttentionBackendImpl.forward

@register_patch("Qwen3NextAttentionBackendPatch", NPUAttentionBackendImpl)
class Qwen3NextAttentionBackendPatch(VLLMPatch):
    _attr_names_to_apply = ['forward']

    def forward(self, *args, **kwargs):
        kv_cache = kwargs.get('kv_cache') if 'kv_cache' in kwargs else (args[4] if len(args) > 4 else None)
        attn_metadata = kwargs.get('attn_metadata') if 'attn_metadata' in kwargs else (args[5] if len(args) > 5 else None)
        
        if kv_cache is not None and attn_metadata is not None:
            orig_bs = getattr(self, 'block_size', 0)
            k_shape = list(kv_cache[0].shape)
            if orig_bs == 0 and 768 in k_shape:
                orig_bs = 768
            
            # patch begin
            if orig_bs > 512:
                assert orig_bs % 384 == 0
                ratio = orig_bs // 384
                
                if orig_bs in k_shape:
                    bs_idx = k_shape.index(orig_bs)
                    if bs_idx == 1:
                        new_k = kv_cache[0].view(k_shape[0] * ratio, 384, *k_shape[2:])
                        new_v = kv_cache[1].view(k_shape[0] * ratio, 384, *k_shape[2:])
                    elif bs_idx == 2:
                        new_k = kv_cache[0].view(k_shape[0], k_shape[1], ratio, 384, *k_shape[3:]).permute(0, 2, 1, 3, 4).reshape(k_shape[0] * ratio, k_shape[1], 384, *k_shape[3:])
                        new_v = kv_cache[1].view(k_shape[0], k_shape[1], ratio, 384, *k_shape[3:]).permute(0, 2, 1, 3, 4).reshape(k_shape[0] * ratio, k_shape[1], 384, *k_shape[3:])
                    else:
                        new_k, new_v = kv_cache[0], kv_cache[1]
                else:
                    new_k, new_v = kv_cache[0], kv_cache[1]
                    
                new_kv_cache = (new_k, new_v)
                original_block_tables = attn_metadata.block_tables
                
                if getattr(attn_metadata, 'kernel_block_tables', None) is not None:
                    attn_metadata.block_tables = attn_metadata.kernel_block_tables
                else:
                    bt = attn_metadata.block_tables
                    if bt is not None and bt.numel() > 0:
                        target_shape = (*bt.shape[:-1], -1)
                        bt_scaled = (bt * ratio).unsqueeze(-1)
                        offsets = torch.arange(ratio, device=bt.device)
                        attn_metadata.block_tables = (bt_scaled + offsets).reshape(target_shape).to(torch.int32)
                    else:
                        device = bt.device if bt is not None else kv_cache[0].device
                        attn_metadata.block_tables = torch.empty(0, dtype=torch.int32, device=device)
                
                new_args = list(args)
                if len(args) > 4:
                    new_args[4] = new_kv_cache
                else:
                    kwargs['kv_cache'] = new_kv_cache
                    
                import torch_npu
                orig_infer = torch_npu.npu_fused_infer_attention_score
                orig_prompt = getattr(torch_npu, 'npu_prompt_flash_attention', None)
                
                def spoofed_infer(*a, **k):
                    new_a = list(a)
                    for i, val in enumerate(new_a):
                        if isinstance(val, int) and val == orig_bs:
                            new_a[i] = 384
                    for key in list(k.keys()):
                        if isinstance(k[key], int) and k[key] == orig_bs:
                            k[key] = 384
                    return orig_infer(*new_a, **k)
                    
                def spoofed_prompt(*a, **k):
                    new_a = list(a)
                    for i, val in enumerate(new_a):
                        if isinstance(val, int) and val == orig_bs:
                            new_a[i] = 384
                    for key in list(k.keys()):
                        if isinstance(k[key], int) and k[key] == orig_bs:
                            k[key] = 384
                    return orig_prompt(*new_a, **k)
                    
                torch_npu.npu_fused_infer_attention_score = spoofed_infer
                if orig_prompt: 
                    torch_npu.npu_prompt_flash_attention = spoofed_prompt
                
                try:
                    return _vanilla_forward(self, *new_args, **kwargs)
                finally:
                    torch_npu.npu_fused_infer_attention_score = orig_infer
                    if orig_prompt: 
                        torch_npu.npu_prompt_flash_attention = orig_prompt
                    attn_metadata.block_tables = original_block_tables
            # patch end
                    
        return _vanilla_forward(self, *args, **kwargs)