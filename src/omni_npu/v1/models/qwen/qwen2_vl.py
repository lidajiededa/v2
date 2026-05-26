# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections.abc import Iterable
from functools import partial
from typing import Callable, Optional, Set, Tuple, Type

import torch
import torch.nn as nn
import torch_npu
from einops import rearrange
from transformers.models.qwen2_vl.configuration_qwen2_vl import \
    Qwen2VLVisionConfig
from vllm.config import VllmConfig
from vllm.distributed import utils as dist_utils
from vllm.model_executor.layers.activation import QuickGELU
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VisionAttention, Qwen2VisionBlock, Qwen2VisionPatchEmbed,
    Qwen2VisionTransformer, Qwen2VLDummyInputsBuilder,
    Qwen2VLForConditionalGeneration, Qwen2VLMultiModalProcessor,
    Qwen2VLProcessingInfo)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm.logger import init_logger

MIN_PAD_SIZE = 64  # min_size to pad weight
MAX_PAD_SIZE = 128  # max_size to pad weight

logger = init_logger(__name__)

class NPUQwen2VisionAttention(Qwen2VisionAttention):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            projection_size=projection_size,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.cu_seqlens = None
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads)
        self.origin_hidden_size_per_attention_head = self.hidden_size_per_attention_head
        if MIN_PAD_SIZE < self.hidden_size_per_attention_head < MAX_PAD_SIZE:
            self.hidden_size_per_attention_head = MAX_PAD_SIZE

    def forward(                                                                                                                                                                                                                                                                                                         
        self,                                                                                                                                                                                                                                                                                                            
        x: torch.Tensor,                                                                                                                                                                                                                                                                                                 
        cu_seqlens: torch.Tensor,                                                                                                                                                                                                                                                                                        
        cos: torch.Tensor,                                    
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # QKV projection: [s, b, c] -> 3 * [s, b, head, head_dim]
        x, _ = self.qkv(x)
        q, k, v = self.split_qkv(x)
        batch_size = q.shape[1]

        # Rearrange and apply rotary embedding
        q, k, v = [rearrange(t, "s b ... -> b s ...").contiguous() for t in (q, k, v)]
        q = torch_npu.npu_rotary_mul(q, cos, sin)
        k = torch_npu.npu_rotary_mul(k, cos, sin)
        q, k, v = [rearrange(t, "b s h d -> (b s) h d").contiguous() for t in (q, k, v)]

        # Flash attention (requires pta version >= 2.5.1)
        context_layer = torch.empty_like(q)
        torch_npu._npu_flash_attention_unpad(
            query=q,
            key=k,
            value=v,
            seq_len=cu_seqlens,
            scale_value=self.origin_hidden_size_per_attention_head**-0.5,
            num_heads=self.num_attention_heads_per_partition,
            num_kv_heads=self.num_attention_heads_per_partition,
            out=context_layer,
        )

        # Output projection: [s, b, head * head_dim]
        context_layer = rearrange(context_layer, "(b s) h d -> s b (h d)", b=batch_size)
        output, _ = self.proj(context_layer)
        return output


class NPUQwen2VisionBlock(Qwen2VisionBlock):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        act_layer: Type[nn.Module] = QuickGELU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=prefix,
        )
        self.attn = NPUQwen2VisionAttention(embed_dim=dim,
                                               num_heads=num_heads,
                                               projection_size=dim,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.attn")

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            cos=cos,
            sin=sin,
        )

        x = x + self.mlp(self.norm2(x))
        return x


class NPUQwen2VisionPatchEmbed(Qwen2VisionPatchEmbed):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.matmul(
            self.proj.weight.data.view(self.embed_dim, -1).transpose(0, 1))
        return x


class NPUQwen2VisionTransformer(Qwen2VisionTransformer):

    def __init__(
        self,
        vision_config: Qwen2VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        interleaved=False,
    ) -> None:
        super().__init__(vision_config=vision_config, norm_eps=norm_eps, quant_config=quant_config, prefix=prefix)

        self.interleaved = interleaved
        self.enable_pad = False
        self.depth = vision_config.depth
        self.hidden_size = vision_config.embed_dim
        self.num_heads = vision_config.num_heads
        self.patch_embed = NPUQwen2VisionPatchEmbed(
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            in_channels=vision_config.in_channels,
            embed_dim=vision_config.embed_dim,
        )

        self.blocks = nn.ModuleList([
            NPUQwen2VisionBlock(dim=self.embed_dim,
                                   num_heads=self.num_heads,
                                   mlp_ratio=vision_config.mlp_ratio,
                                   norm_layer=partial(nn.LayerNorm,
                                                      eps=norm_eps),
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.blocks.{layer_idx}")
            for layer_idx in range(vision_config.depth)
        ])

        self.hidden_size_per_attention_head = dist_utils.divide(
            self.hidden_size, self.num_heads)

        if MIN_PAD_SIZE< self.hidden_size_per_attention_head < MAX_PAD_SIZE:
            self.enable_pad = True
            self.origin_hidden_size_per_attention_head = self.hidden_size_per_attention_head
            self.half_origin_hidden_size_per_attention_head = self.hidden_size_per_attention_head // 2
            self.half_pad_hidden_size_per_attention_head = (
                MAX_PAD_SIZE - self.hidden_size_per_attention_head) // 2
            self.hidden_size_per_attention_head = MAX_PAD_SIZE

    def cal_cos_sin(self, rotary_pos_emb):
        cos, sin = rotary_pos_emb  # [seqlen, rotary_dim / 2]
        
        # Optional padding
        if self.enable_pad:
            pad = (0, self.half_pad_hidden_size_per_attention_head)
            cos = torch.nn.functional.pad(cos, pad)
            sin = torch.nn.functional.pad(sin, pad)
        
        # Create doubled tensors: either concatenated or interleaved
        if self.interleaved:
            # Interleave: [cos, cos] -> [cos0, cos1, cos0, cos1, ...]
            cos_new = torch.stack((cos, cos), dim=-1).flatten(-2)
            sin_new = torch.stack((sin, sin), dim=-1).flatten(-2)
        else:
            # Concatenate: [cos, cos] -> [cos, cos, cos, cos]
            cos_new = torch.cat((cos, cos), dim=-1)
            sin_new = torch.cat((sin, sin), dim=-1)
        
        # Reshape to [1, seqlen, 1, hidden_size_per_attention_head]
        shape = (1, -1, 1, self.hidden_size_per_attention_head)
        return cos_new.view(*shape), sin_new.view(*shape)

    def pad_qkv_bias(self, bias):
        # Reshape once: [N, 3, hidden_size]
        x = bias.view(-1, 3, self.origin_hidden_size_per_attention_head)
        
        half = self.half_origin_hidden_size_per_attention_head
        pad = self.half_pad_hidden_size_per_attention_head
        
        # Split, pad each half, concatenate, then flatten
        return torch.cat([
            torch.nn.functional.pad(x[:, :, :half], (0, pad)),
            torch.nn.functional.pad(x[:, :, half:], (0, pad))
        ], dim=2).flatten()

    def pad_qkv_weight(self, data):
        # Reshape once: [N, 3, origin_hidden_size, hidden_size]
        x = data.view(-1, 3, self.origin_hidden_size_per_attention_head, self.hidden_size)
        
        half = self.half_origin_hidden_size_per_attention_head
        pad = self.half_pad_hidden_size_per_attention_head
        
        # Split, pad each half (along dim 2), concatenate
        return torch.cat([
            torch.nn.functional.pad(x[:, :, :half, :], (0, 0, 0, pad)),
            torch.nn.functional.pad(x[:, :, half:, :], (0, 0, 0, pad))
        ], dim=2).view(-1, self.hidden_size)

    def pad_proj_weight(self, data):
        half = self.half_origin_hidden_size_per_attention_head
        pad = self.half_pad_hidden_size_per_attention_head
        
        # Reshape to 3D, pad last dimension on the right, flatten back
        x = data.view(self.hidden_size, -1, half)
        return torch.nn.functional.pad(x, (0, pad)).view(self.hidden_size, -1)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        # Mapping for stacking q/k/v into qkv_proj: (target, source, shard_id)
        stacked_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        
        for name, weight in weights:
            # Try to match stacked params (q/k/v -> qkv_proj)
            for param_name, weight_name, shard_id in stacked_mapping:
                if weight_name not in name:
                    continue
                
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, weight, shard_id)
                loaded_params.add(name)
                break
            else:
                # Non-stacked param: load directly
                param = params_dict[name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, weight)
                
                # Apply padding for attention weights/biases
                if self.enable_pad:
                    if "attn.proj.weight" in name:
                        param.data = self.pad_proj_weight(param.data)
                    elif "attn.qkv.weight" in name:
                        param.data = self.pad_qkv_weight(param.data)
                    elif "attn.qkv.bias" in name:
                        param.data = self.pad_qkv_bias(param.data)
                
                loaded_params.add(name)
        
        return loaded_params

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        # Compute cu_seqlens for packed attention (avoid cumsum for unpadFA operator)
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cpu().to(torch.int32)
        
        # Patchify and ensure correct device/dtype
        x = self.patch_embed(x.to(device=self.device, dtype=self.dtype))
        
        # Rotary position embeddings
        cos, sin = self.cal_cos_sin(self.rot_pos_emb(grid_thw))
        
        # Transformer blocks: x is [B, 1, L, C] after unsqueeze
        x = x.unsqueeze(1)
        for blk in self.blocks:
            x = blk(x, cu_seqlens=cu_seqlens, cos=cos, sin=sin)
        
        return self.merger(x)

@MULTIMODAL_REGISTRY.register_processor(
    Qwen2VLMultiModalProcessor,
    info=Qwen2VLProcessingInfo,
    dummy_inputs=Qwen2VLDummyInputsBuilder
)
class NpuQwen2VLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        
        norm_eps = getattr(self.config, "rms_norm_eps", 1e-6)
        self.visual = NPUQwen2VisionTransformer(
            self.config.vision_config,
            norm_eps=norm_eps,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )