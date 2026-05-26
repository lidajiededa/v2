# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3Next model."""

from collections.abc import Iterable
from typing import Optional, List, Tuple, Callable
from math import log

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.library import Library
from transformers.activations import ACT2FN
from transformers import PretrainedConfig

from vllm.v1.attention.backend import AttentionMetadata
from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ModelConfig,
    SpeculativeConfig,
    VllmConfig,
    get_current_vllm_config
)
from vllm.distributed import (
    divide,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    QKVParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    vllm_lib
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsLoRA,
    SupportsPP
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory, make_layers,
    maybe_prefix
)

# from omni_npu.v1.layers.fused_moe.layer import NPUFusedMoEV1
from vllm.model_executor.layers.fused_moe import FusedMoE
from omni_npu.layers.fused_moe.layer import NPUSharedFusedMoE
from omni_npu.v1.layers.linear import (
    RowParallelFlashCommLinear, 
    QKVParallelFlashCommLinear
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from omni_npu.layers.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from omni_npu.layers.ops.fla_ops import (
    Qwen3NextRMSNormGated as RMSNormGated,
    GemmaRMSNorm as Qwen3NextRMSNorm, 
    fused_recurrent_gated_delta_rule, 
    chunk_gated_delta_rule
)

logger = init_logger(__name__)

KVCache = tuple[torch.Tensor, torch.Tensor]


class Qwen3NextSparseMoeBlock(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.prefix = prefix

        self.experts = NPUSharedFusedMoE(shared_experts=None,
                                num_experts=config.num_experts,
                                top_k=config.num_experts_per_tok,
                                hidden_size=config.hidden_size,
                                intermediate_size=config.moe_intermediate_size,
                                renormalize=config.norm_topk_prob,
                                quant_config=quant_config,
                                prefix=f"{prefix}.experts")

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.num_experts,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")

        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
            )
        else:
            self.shared_expert = None
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size,
                                                  1,
                                                  bias=False)

    def _maybe_ignore_quant_config(self, quant_config: QuantizationConfig):
        # GPTQ configs do not have a list of ignored modules, however AutoGPTQ
        # seems to avoid gate quantization.
        # See: https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4
        if isinstance(quant_config, (GPTQConfig, GPTQMarlinConfig)):
            return None
        return quant_config

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states) 
            if self.shared_expert_gate is not None:
                shared_output = F.sigmoid(
                    self.shared_expert_gate(hidden_states)) * shared_output

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)

        final_hidden_states = self.experts(hidden_states=hidden_states,
                                           router_logits=router_logits)[1]

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states.view(orig_shape)


class Qwen3NextGatedDeltaNet(nn.Module, MambaBase):

    @property
    def mamba_type(self) -> str:
        return "gdn_attention"

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype)

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def __init__(
        self,
        config: PretrainedConfig,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        speculative_config: Optional[SpeculativeConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = extract_layer_index(prefix)
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps
        self.prefix = prefix

        self.config = config
        self.model_config = model_config
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.speculative_config = speculative_config
        self.num_spec = (self.speculative_config.num_speculative_tokens
                         if self.speculative_config else 0)

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # projection of the input hidden states
        self.projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        self.projection_size_ba = self.num_v_heads * 2
        self.in_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.projection_size_qkvz, self.projection_size_ba],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight, {
                "weight_loader":
                mamba_v2_sharded_weight_loader([
                    query_key_settings,
                    query_key_settings,
                    value_settings,
                ], self.tp_size, self.tp_rank)
            })

        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.tp_size), )
        self.A_log = nn.Parameter(
            torch.empty(
                divide(self.num_v_heads, self.tp_size),
                dtype=torch.float32,
            ))

        set_weight_attrs(self.A_log,
                         {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias,
                         {"weight_loader": sharded_weight_loader(0)})

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            device=current_platform.device_type,
            dtype=config.torch_dtype,
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz,
        mixed_ba,
    ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            (self.head_k_dim + self.head_k_dim +
             (self.head_v_dim + self.head_v_dim) * self.num_v_heads //
             self.num_k_heads),
        )
        new_tensor_shape_ba = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads
        ]

        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn],
        #  [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        (query, key, value, z) = torch.split(mixed_qkvz,
                                             split_arg_list_qkvz,
                                             dim=2)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

        return query, key, value, z, b, a
    
    def rearrange_mixed_qkv(self, mixed_qkv, apply_l2_norm=True):
        if mixed_qkv is None:
            return None, None, None
            
        # 1. Split QK block from Value
        qk_dim = 2 * (self.key_dim // self.tp_size)
        qk, value = torch.split(
            mixed_qkv,
            [qk_dim, self.value_dim // self.tp_size],
            dim=-1,
        )
        qk_view = qk.view(qk.shape[0], -1, self.head_k_dim)
        
        if apply_l2_norm:
            qk_sq = torch.sum(qk_view * qk_view, dim=-1, keepdim=True)
            qk_view = qk_view * torch.rsqrt(qk_sq + 1e-12)
            
        query, key = torch.chunk(qk_view, 2, dim=1) 
        query = query.unsqueeze(0)  # [1, seq_len, num_h, head_dim]
        key = key.unsqueeze(0)      # [1, seq_len, num_h, head_dim]

        value = rearrange(value, 'l (h d) -> 1 l h d', d=self.head_v_dim)
        
        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        return gdn_attention(
            hidden_states,
            output,
            self.prefix,
        )

    def _forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        if attn_metadata.num_prefills > 0 and not hasattr(attn_metadata, 'non_spec_seqlens_list'):
            ns_q_loc = attn_metadata.non_spec_query_start_loc
            ns_state_idx = attn_metadata.non_spec_state_indices_tensor
            has_init = attn_metadata.has_initial_state

            attn_metadata.no_initial_state_indices = torch.where(~has_init)[0]
            attn_metadata.has_initial_state_list = has_init.tolist()
            attn_metadata.non_spec_query_start_loc_cpu = ns_q_loc.tolist()
            attn_metadata.non_spec_seqlens_list = (ns_q_loc[1:] - ns_q_loc[:-1]).tolist()
            attn_metadata.non_spec_state_indices_tensor_list = (
                ns_state_idx.tolist() if ns_state_idx is not None else None
            )
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        conv_state_native = self_kv_cache[0] # NATIVE LAYOUT
        conv_state = conv_state_native.transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_tokens = hidden_states.size(0)
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        # 1. Set up dimensions for reshapes later
        projected_states, _ = self.in_proj(hidden_states)

        projected_states_qkvz, projected_states_ba = torch.split(
            projected_states,
            [
                self.projection_size_qkvz // self.tp_size,
                self.projection_size_ba // self.tp_size
            ],
            dim=-1,
        )
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba)
        query, key, value = map(lambda x: rearrange(x, 'l p d -> l (p d)'),
                                (query, key, value))
        mixed_qkv = torch.cat((query, key, value), dim=-1)

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 2.1: process the mutli-query part
        if spec_sequence_masks is not None:
            mixed_qkv_spec = mixed_qkv_spec.view(
                attn_metadata.num_spec_decodes, -1, mixed_qkv_spec.size(-1))
            mixed_qkv_spec = rearrange(mixed_qkv_spec, 'b l d -> b d l')
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state_native,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][
                    : attn_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )
            mixed_qkv_spec = rearrange(mixed_qkv_spec, 'b d l -> (b l) d')

        # 2.2: process the remaining part
        num_prefills = attn_metadata.num_prefills
        num_decodes = attn_metadata.num_decodes
        is_pure_prefill = (num_prefills > 0 and num_decodes == 0)
        is_pure_decode  = (num_decodes > 0 and num_prefills == 0)

        if is_pure_prefill:
            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec, # Assuming the wrapper handles transposes
                conv_weights,
                self.conv1d.bias,
                conv_states=conv_state_native, 
                seqlens_list=attn_metadata.non_spec_seqlens_list,
                cache_indices_list=attn_metadata.non_spec_state_indices_tensor_list,
                has_initial_state_list=attn_metadata.has_initial_state_list,
                activation=self.activation,
            )

        elif is_pure_decode:
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state_native,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[:num_decodes],
                validate_data=True,
            )

        elif num_prefills > 0 and num_decodes > 0: # True Hybrid Batch
            # --- DECODES FIRST ---
            mixed_qkv_decode = mixed_qkv_non_spec[:num_decodes].contiguous()
            mixed_qkv_decode_out = causal_conv1d_update(
                mixed_qkv_decode,
                conv_state_native,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[:num_decodes].contiguous(),
                validate_data=True,
            )
            
            # --- PREFILLS SECOND ---
            mixed_qkv_prefill = mixed_qkv_non_spec[num_decodes:].contiguous()
            mixed_qkv_prefill_out = causal_conv1d_fn(
                mixed_qkv_prefill,
                conv_weights,
                self.conv1d.bias,
                conv_states=conv_state_native,
                seqlens_list=attn_metadata.non_spec_seqlens_list[num_decodes:],
                cache_indices_list=attn_metadata.non_spec_state_indices_tensor_list[num_decodes:],
                has_initial_state_list=attn_metadata.has_initial_state_list[num_decodes:],
                activation=self.activation,
            )
            
            # Reconstruct the batch in the exact order vLLM expects
            mixed_qkv_non_spec = torch.cat([mixed_qkv_decode_out, mixed_qkv_prefill_out], dim=0)
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(
            mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        g, beta = map(lambda x: rearrange(x, 'l d -> 1 l d'), (g, beta))

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                g_spec = g
                beta_spec = beta
                g_non_spec = None
                beta_non_spec = None
            else:
                g_spec = g.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                g_non_spec = g.index_select(1, non_spec_token_indx)
                beta_non_spec = beta.index_select(1, non_spec_token_indx)
        else:
            g_spec = None
            beta_spec = None
            g_non_spec = g
            beta_non_spec = beta

        # 3. Recurrent attention

        # 3.1: process the mutlti-query part
        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = fused_recurrent_gated_delta_rule(
                q=query_spec,
                k=key_spec,
                v=value_spec,
                g=g_spec,
                beta=beta_spec,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                actual_seqlens=spec_query_start_loc[1 : attn_metadata.num_spec_decodes + 1] - spec_query_start_loc[: attn_metadata.num_spec_decodes],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=False,
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 3.2: process the remaining part
        if is_pure_prefill:
            initial_state = ssm_state[non_spec_state_indices_tensor].clone()
            # initial_state[~has_initial_state, ...] = 0
            initial_state[attn_metadata.no_initial_state_indices] = 0
            core_attn_out_non_spec, last_recurrent_state = chunk_gated_delta_rule(
                q=query_non_spec, k=key_non_spec, v=value_non_spec, g=g_non_spec, beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=non_spec_query_start_loc,
                cu_seqlens_cpu=attn_metadata.non_spec_query_start_loc_cpu,
                head_first=False,
                use_qk_l2norm_in_kernel=False,
            )
            ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(ssm_state.dtype)

        elif is_pure_decode:
            core_attn_out_non_spec, last_recurrent_state = fused_recurrent_gated_delta_rule(
                q=query_non_spec, k=key_non_spec, v=value_non_spec, g=g_non_spec, beta=beta_non_spec,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=non_spec_query_start_loc[:num_decodes + 1],
                actual_seqlens=non_spec_query_start_loc[1:num_decodes + 1] - non_spec_query_start_loc[:num_decodes],
                ssm_state_indices=non_spec_state_indices_tensor[:num_decodes],
                use_qk_l2norm_in_kernel=False,
            )

        elif num_prefills > 0 and num_decodes > 0: # True Hybrid Batch
            if not hasattr(attn_metadata, '_npu_retain_gc'):
                attn_metadata._npu_retain_gc = []
                
            # --- DECODES FIRST ---
            q_decode = query_non_spec[:, :num_decodes, ...].contiguous()
            k_decode = key_non_spec[:, :num_decodes, ...].contiguous()
            v_decode = value_non_spec[:, :num_decodes, ...].contiguous()
            g_decode = g_non_spec[:, :num_decodes, ...].contiguous()
            beta_decode = beta_non_spec[:, :num_decodes, ...].contiguous()
            
            decode_indices = non_spec_state_indices_tensor[:num_decodes].contiguous()
            decode_cu_seqlens = torch.arange(num_decodes + 1, dtype=torch.int32, device=q_decode.device)
            decode_actual_seqlens = torch.ones(num_decodes, dtype=torch.int32, device=q_decode.device)
            attn_metadata._npu_retain_gc.extend([decode_cu_seqlens, decode_actual_seqlens])
            
            core_attn_out_decode, last_recurrent_state = fused_recurrent_gated_delta_rule(
                q=q_decode, k=k_decode, v=v_decode, g=g_decode, beta=beta_decode,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=decode_cu_seqlens,
                actual_seqlens=decode_actual_seqlens,
                ssm_state_indices=decode_indices,
                use_qk_l2norm_in_kernel=False,
            )
            
            # --- PREFILLS SECOND ---
            q_prefill = query_non_spec[:, num_decodes:, ...].contiguous()
            k_prefill = key_non_spec[:, num_decodes:, ...].contiguous()
            v_prefill = value_non_spec[:, num_decodes:, ...].contiguous()
            g_prefill = g_non_spec[:, num_decodes:, ...].contiguous()
            beta_prefill = beta_non_spec[:, num_decodes:, ...].contiguous()
            
            prefill_indices = non_spec_state_indices_tensor[num_decodes:].contiguous()
            prefill_has_init = has_initial_state[num_decodes:].contiguous()
            
            initial_state = ssm_state[prefill_indices].clone()
            no_init_indices = attn_metadata.no_initial_state_indices - num_decodes
            initial_state[no_init_indices] = 0
            # initial_state[~prefill_has_init, ...] = 0
            attn_metadata._npu_retain_gc.append(initial_state)
            
            prefill_start_loc = non_spec_query_start_loc[num_decodes:].contiguous()
            prefill_start_loc = prefill_start_loc - prefill_start_loc[0]

            prefill_start_loc_cpu = attn_metadata.non_spec_query_start_loc_cpu[num_decodes:]
            offset = prefill_start_loc_cpu[0]
            prefill_start_loc_cpu = [x - offset for x in prefill_start_loc_cpu]
            
            core_attn_out_prefill, last_recurrent_state = chunk_gated_delta_rule(
                q=q_prefill, k=k_prefill, v=v_prefill, g=g_prefill, beta=beta_prefill,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=prefill_start_loc,
                cu_seqlens_cpu=prefill_start_loc_cpu,
                head_first=False,
                use_qk_l2norm_in_kernel=False,
            )
            ssm_state[prefill_indices] = last_recurrent_state.to(ssm_state.dtype)
            out_d = core_attn_out_decode.reshape(-1, core_attn_out_decode.shape[-2], core_attn_out_decode.shape[-1])
            out_p = core_attn_out_prefill.reshape(-1, core_attn_out_prefill.shape[-2], core_attn_out_prefill.shape[-1])
            # Reconstruct batch
            core_attn_out_non_spec = torch.cat([out_d, out_p], dim=0).unsqueeze(0)
        else:
            core_attn_out_non_spec = None
        # 3.2: process the remaining part
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        # Merge core attention output
        if (spec_sequence_masks is not None
                and core_attn_out_non_spec is not None):
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, '... h d -> ... (h d)')
        output[:num_tokens], _ = self.out_proj(core_attn_out)


class Qwen3NextAttention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % self.tp_size == 0
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None)
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        self.qkv_proj = QKVParallelFlashCommLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj"
        )

        rope_params = getattr(config, "rope_parameters", None)

        if rope_params is None:
            # Fallback for qwen3_coder_next (top-level attributes, missing rope_type)
            rope_params = {
                "rope_theta": getattr(config, "rope_theta", 10000000),
                "partial_rotary_factor": getattr(config, "partial_rotary_factor", 0.25),
                "rope_type": "default" # Manually inject the required default
            }
        else:
            # Handle qwen3.5 (dict exists, ensure it is mutable and has safe fallbacks)
            rope_params = dict(rope_params)
            rope_params["rope_type"] = "default" # hard-coding to avoid unknown scaling type mrope error
            rope_params.setdefault("rope_theta", getattr(config, "rope_theta", 10000000))
            rope_params.setdefault("partial_rotary_factor", getattr(config, "partial_rotary_factor", 0.25))

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_params,
            dual_chunk_attention_config=self.dual_chunk_attention_config,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config":
                self.dual_chunk_attention_config,
            } if self.dual_chunk_attention_config else {},
        )

        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        forward_context: ForwardContext = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[next(iter(attn_metadata))]
            
        is_prefill = attn_metadata is None or attn_metadata.num_prefills != 0

        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_normed = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(-1, self.num_heads * self.head_dim)
        k_normed = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.num_kv_heads * self.head_dim)

        # OPTIMIZATION: Cross-Layer Cos/Sin Caching
        if not is_prefill:
            # Decode: Route to NPU kernel
            q, k = self.rotary_emb(positions, q_normed, k_normed, is_prefill=False)
        else:
            # Prefill: Handle PyTorch logic here to avoid repetitive ViewCopies
            num_tokens = positions.shape[-1] if positions.ndim > 1 else positions.shape[0]
            
            cos, sin = None, None
            
            # Safely check if we have a valid metadata object and a cached tuple
            if attn_metadata is not None and hasattr(attn_metadata, '_npu_cached_cos_sin'):
                cos, sin = attn_metadata._npu_cached_cos_sin
            else:
                # Compute it (Layer 0, or Profile Run)
                self.rotary_emb._match_cos_sin_cache_dtype(q_normed)
                cos_sin = self.rotary_emb.cos_sin_cache[positions]
                cos, sin = cos_sin.chunk(2, dim=-1)
                
                if positions.ndim == 2 and self.rotary_emb.mrope_section:
                    from vllm.model_executor.layers.rotary_embedding.mrope import apply_interleaved_rope
                    if self.rotary_emb.mrope_interleaved:
                        cos = apply_interleaved_rope(cos, self.rotary_emb.mrope_section)
                        sin = apply_interleaved_rope(sin, self.rotary_emb.mrope_section)
                    else:
                        cos = torch.cat([m[i] for i, m in enumerate(cos.split(self.rotary_emb.mrope_section, dim=-1))], dim=-1)
                        sin = torch.cat([m[i] for i, m in enumerate(sin.split(self.rotary_emb.mrope_section, dim=-1))], dim=-1)
                
                cos = cos.contiguous()
                sin = sin.contiguous()
                
                # Cache the contiguous blocks for Layers 1-63
                if attn_metadata is not None:
                    attn_metadata._npu_cached_cos_sin = (cos, sin)
            
            # Apply Rotary mathematically
            def apply_rotary(x):
                x_shape = x.shape
                x = x.view(num_tokens, -1, self.head_dim)
                x_rot = x[..., : self.rotary_emb.rotary_dim]
                x_pass = x[..., self.rotary_emb.rotary_dim :]
                x_rot = self.rotary_emb.apply_rotary_emb(x_rot, cos, sin)
                return torch.cat((x_rot, x_pass), dim=-1).reshape(x_shape)

            q = apply_rotary(q_normed)
            k = apply_rotary(k_normed)

        attn_output = self.attn(q, k, v)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output[:], _ = self.o_proj(attn_output)

        if attn_metadata is not None:
            num_tokens = attn_metadata.num_actual_tokens
            output[num_tokens:] = 0


class Qwen3NextDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_type: str,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        speculative_config: Optional[SpeculativeConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGatedDeltaNet(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                speculative_config=speculative_config,
                prefix=f'{prefix}.linear_attn')
            self.layer_name = f'{prefix}.linear_attn'
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f'{prefix}.self_attn',
            )
            self.layer_name = f'{prefix}.self_attn.attn'
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        if (self.layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and
            (self.layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3NextSparseMoeBlock(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
            )

        self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    self.config.hidden_size,
                    dtype=config.torch_dtype,
                ), )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    self.config.hidden_size,
                    dtype=config.torch_dtype,
                ), )
            
    def _apply_mlp(self, hidden_states, is_prefill):
        """Helper function to apply MLP with appropriate arguments."""
        if isinstance(self.mlp, Qwen3NextSparseMoeBlock):
            return self.mlp(hidden_states)
        else:
            return self.mlp(hidden_states, is_prefill=is_prefill)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        self_attention_output = torch.empty_like(hidden_states)
        if self.layer_type == "linear_attention":
            self.linear_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
            )
        elif self.layer_type == "full_attention":
            self.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1)
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[next(iter(attn_metadata))]
        is_prefill = attn_metadata is None or attn_metadata.num_prefills != 0
        if is_prefill:
            local_length = hidden_states.shape[0]
            hidden_states = self._apply_mlp(hidden_states, is_prefill)[:local_length]
        else:
            hidden_states = self._apply_mlp(hidden_states, is_prefill)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1)
            else:
                assert len(hidden_states.shape) == len(
                    self.ffn_layer_scale.shape
                ), f'shape must be the same {len(hidden_states.shape)}, {len(self.ffn_layer_scale.shape)}'  # noqa: E501
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1)

        return hidden_states, residual


class Qwen3NextModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: PretrainedConfig = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        lora_config = vllm_config.lora_config
        speculative_config = vllm_config.speculative_config

        self.config = config
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        def get_layer(prefix: str):
            return Qwen3NextDecoderLayer(
                config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                speculative_config=speculative_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers")
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        self.norm = Qwen3NextRMSNorm(config.hidden_size,
                                     eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("in_proj", "in_proj_qkvz", 0),
            ("in_proj", "in_proj_ba", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            ### The weight_scale often has shape (n,1)
            # if 'weight_scale' in name:
            #     loaded_weight = loaded_weight.view(-1)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # name = apply_attn_prefix(name, params_dict)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    if name not in params_dict:
                        logger.warning(f"UNUSED WEIGHT, IGNORING {name}")
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        logger.warning(f"UNUSED WEIGHT, IGNORING {name}")
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


@support_torch_compile
class Qwen3NextForCausalLM(nn.Module, HasInnerState, SupportsLoRA, SupportsPP, IsHybrid):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["up_proj", "down_proj"],
        "in_proj": ["in_proj_qkvz", "in_proj_ba"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        scheduler_config = vllm_config.scheduler_config
        assert not cache_config.enable_prefix_caching, \
            "Qwen3Next currently does not support prefix caching"
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3NextModel(vllm_config=vllm_config,
                                    prefix=maybe_prefix(prefix, "model"))
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
        )
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        # Set MoE hyperparameters
        self.expert_weights = []

        self.moe_layers = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, Qwen3NextDecoderLayer)
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            raise RuntimeError("No Qwen3Next layer found in the model.layers.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ):
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)

        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype)

    @classmethod
    def get_mamba_state_shape_from_config(
            cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (vllm_config.speculative_config.num_speculative_tokens
                    if vllm_config.speculative_config else 0)
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def should_use_eager_mode(self, *args, **kwargs):
        attn_metadata = get_forward_context().attn_metadata
        if not attn_metadata:
            return True

        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.model.layers[self.model.start_layer].layer_name]

        return attn_metadata.num_prefills != 0

    def mark_static_for_graph(self, input_ids, positions, attn_metadata, kv_caches, **kwargs):
        pass

def gdn_attention(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward(hidden_states=hidden_states, output=output)


def gdn_attention_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return

def ascend_direct_register_custom_op(
        op_name: str,
        op_func: Callable,
        mutates_args: list[str],
        fake_impl: Optional[Callable] = None,
        target_lib: Optional[Library] = None,
        dispatch_key: str = "CUDA",
        tags: Tuple[torch.Tag, ...] = (),
):
    # In pytorch 2.5.1, torch.library.infer_schema require the input function to
    # have annotations supported by typing library. But in pytorch 2.7.0 which
    # vllm using, torch.library.infer_schema require the python builtin type. In
    # this case, we should revert built type to typing type for 2.5.1 backward
    # compatibility.
    for k, v in op_func.__annotations__.items():
        if v == list[int]:
            op_func.__annotations__[k] = List[int]
        if v == Optional[list[int]]:
            op_func.__annotations__[k] = Optional[List[int]]
    import torch.library
    schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    my_lib = target_lib or vllm_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)

direct_register_custom_op = ascend_direct_register_custom_op

direct_register_custom_op(
    op_name="gdn_attention",
    op_func=gdn_attention,
    mutates_args=["output"],
    fake_impl=gdn_attention_fake,
    dispatch_key=current_platform.dispatch_key,
)