# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Inference-only MiniMax M2 model compatible with NPU."""

from collections.abc import Iterable
from functools import partial
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
)
from omni_npu.v1.layers.linear import (
    RowParallelFlashCommLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.linear_attn import MiniMaxText01RMSNormTP
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
)
from omni_npu.v1.layers.vocab_parallel_embedding import NPUVocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import SupportsEagle3, SupportsPP
from vllm.model_executor.models.minimax_m2 import (
    get_spec_layer_idx_from_weight_name,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from omni_npu.layers.npu_rms_norm import NPUMiniMaxText01RMSNormTP
from omni_npu.attention.backends.utils import SPManager
from omni_npu.model_config.config_loader.loader import model_extra_config


class NPUMiniMaxRMSNormTP(MiniMaxText01RMSNormTP):
    """RMSNormTP with custom weight sharding for kv_heads < tp_size."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        weight_shard_world_size: int | None = None,
        weight_shard_rank: int | None = None,
    ) -> None:
        CustomOp.__init__(self)
        self.tp_world = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.weight_shard_world = weight_shard_world_size or self.tp_world
        self.weight_shard_rank = (
            self.tp_rank if weight_shard_rank is None else weight_shard_rank
        )

        if hidden_size % self.weight_shard_world != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by "
                f"weight_shard_world_size={self.weight_shard_world}"
            )

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.tp_world = get_tensor_model_parallel_world_size()

    @staticmethod
    def weight_loader(
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        param.data.copy_(loaded_weight)


class MiniMaxM2MoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > config.num_local_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_local_experts}."
            )

        self.use_routing_bias = getattr(config, "use_routing_bias", False)
        if self.use_routing_bias:
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(config.num_local_experts, dtype=torch.float32)
            )
            self.e_score_correction_bias.weight_loader = (
                self._ebias_weight_loader
            )
        else:
            self.e_score_correction_bias = None

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.experts = FusedMoE(
            gate=self.gate,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            scoring_func=config.scoring_func,
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            e_score_correction_bias=self.e_score_correction_bias,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            reduce_results=False,
            renormalize=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    @staticmethod
    def _ebias_weight_loader(
        param: nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Gate routing + expert computation handled internally by NPUSharedFusedMoE
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=None
        )
        if isinstance(final_hidden_states, tuple):
            _, final_hidden_states = final_hidden_states

        if self.tp_size > 1:
            final_hidden_states = (
                self.experts.maybe_all_reduce_tensor_model_parallel(
                    final_hidden_states
                )
            )
        return final_hidden_states.view(num_tokens, hidden_dim)


class MiniMaxM2Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rotary_dim: int,
        rope_parameters: dict[str, Any] | None = None,
        attn_window_size: int | None = None,
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.num_kv_head_replicas = max(1, tp_size // self.total_num_kv_heads)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size, self.head_dim, self.total_num_heads,
            self.total_num_kv_heads, bias=qkv_bias,
            quant_config=quant_config, prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelFlashCommLinear(
            self.total_num_heads * self.head_dim, hidden_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.o_proj",
        )

        if (
            rope_parameters is not None
            and "partial_rotary_factor" not in rope_parameters
        ):
            rope_parameters["partial_rotary_factor"] = rotary_dim / self.head_dim
        self.rotary_emb = get_rope(
            self.head_dim, max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
        )
        self.attn = Attention(
            self.num_heads, self.head_dim, self.scaling,
            num_kv_heads=self.num_kv_heads,
            per_layer_sliding_window=attn_window_size,
            cache_config=cache_config, quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        # NPU RMSNormTP with custom shard support + fast path
        self.q_norm = NPUMiniMaxRMSNormTP(
            self.head_dim * self.total_num_heads, eps=rms_norm_eps
        )
        self.q_norm.head_dim = self.head_dim
        if self.total_num_kv_heads < tp_size:
            self.k_norm = NPUMiniMaxRMSNormTP(
                self.head_dim * self.total_num_kv_heads,
                eps=rms_norm_eps,
                weight_shard_world_size=self.total_num_kv_heads,
                weight_shard_rank=(
                    get_tensor_model_parallel_rank()
                    // self.num_kv_head_replicas
                ),
            )
        else:
            self.k_norm = NPUMiniMaxRMSNormTP(
                self.head_dim * self.total_num_kv_heads, eps=rms_norm_eps
            )
        self.k_norm.head_dim = self.head_dim

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if model_extra_config.parall_config.ena_seq_parallel:
            sp_manager = SPManager.init_sp(
                sp_group=get_tp_group(),
                tok=positions.size(0),
            )
            hidden_states = sp_manager.ag_tokens(hidden_states)
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = NPUMiniMaxText01RMSNormTP.forward_qk(
            self.q_norm, self.k_norm, q.contiguous(), k.contiguous()
        )
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        if model_extra_config.parall_config.ena_seq_parallel:
            attn_output = sp_manager.align_tokens(attn_output)
        output, _ = self.o_proj(attn_output)
        return output


class MiniMaxM2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        if hasattr(config, "max_model_len") and isinstance(config.max_model_len, int):
            max_position_embeddings = max(
                config.max_position_embeddings, config.max_model_len
            )
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        self.self_attn = MiniMaxM2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rotary_dim=config.rotary_dim,
            rope_parameters=config.rope_parameters,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        self.block_sparse_moe = MiniMaxM2MoE(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)

        return hidden_states, residual


@support_torch_compile
class MiniMaxM2Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = NPUVocabParallelEmbedding(
                config.vocab_size, config.hidden_size,
                quant_config=None, prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MiniMaxM2DecoderLayer(
                config, prefix,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        ena_sp = model_extra_config.parall_config.ena_seq_parallel
        return self.embed_tokens(input_ids, enable_scatter=ena_sp)

    def _maybe_add_aux_hidden_state(
        self,
        aux_hidden_states: list[torch.Tensor],
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> None:
        if layer_idx in getattr(self, "aux_hidden_state_layers", ()):
            aux_hidden_states.append(
                hidden_states + residual if residual is not None else hidden_states
            )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        ena_sp = model_extra_config.parall_config.ena_seq_parallel
        sp_manager = None
        if ena_sp:
            sp_manager = SPManager.init_sp(
                sp_group=get_tp_group(),
                tok=positions.size(0),
            )

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
                if sp_manager is not None:
                    hidden_states = sp_manager.slice_tokens(hidden_states)
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states: list[torch.Tensor] = []
        self._maybe_add_aux_hidden_state(
            aux_hidden_states, 0, hidden_states, residual
        )
        for idx, layer in enumerate(self.layers[self.start_layer:self.end_layer]):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_aux_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if sp_manager is not None:
            hidden_states = sp_manager.ag_tokens(hidden_states)
            if residual is not None:
                residual = sp_manager.ag_tokens(residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return FusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        expert_params_mapping = self.get_expert_mapping()
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
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
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param, loaded_weight, name,
                        shard_id=shard_id, expert_id=expert_id,
                    )
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class MiniMaxM2ForCausalLM(nn.Module, SupportsPP, SupportsEagle3):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        if hasattr(vllm_config.model_config, "max_model_len"):
            self.config.max_model_len = vllm_config.model_config.max_model_len
        self.model = MiniMaxM2Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size, quant_config=None
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return self.get_eagle3_aux_hidden_state_layers()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self, hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
