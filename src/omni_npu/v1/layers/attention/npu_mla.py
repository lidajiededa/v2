# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional, Union

import torch
import torch_npu
from torch import nn
from transformers import PretrainedConfig

from vllm.platforms import current_platform
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tp_group,
    split_tensor_along_last_dim,
)
from vllm.config import VllmConfig, CacheConfig, get_current_vllm_config
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.attention.layer import MLAAttention
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.logger import init_logger
logger = init_logger(__name__)

try:
    from vllm.model_executor.layers.attention.static_sink_attention import StaticSinkMLAAttention
except ImportError:
    logger.warning("StaticSinkMLAAttention has not being defined, skipping...")

try:
    from vllm.model_executor.layers.npumome import MomeAttention
except ImportError:
    logger.warning("MomeAttention has not being defined, skipping...")

try:
    from vllm.model_executor.layers.mome import AggregateConv
except ImportError:
    logger.warning("AggregateConv has not being defined, skipping...")

from omni_npu.attention.backends.mla import NPUMLAImpl, NPUMLAMetadata
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.attention.backends.utils import (
    cache_fit_shape,
    get_batch_desc,
    sp_disabled,
    lazy_zero_like,
    SPManager,
)
from omni_npu.plugin_decorators import attn_decorator
from omni_npu.compilation.utils import (
    capture_graph_task,
    OP_FIA_V1,
    OP_FIA_SINK,
)
from omni_npu.v1.layers.utils import (
    yarn_get_mscale,
    named_stream,
)
from omni_npu.v1.layers.linear import (
    ColumnParallelFlashCommLinear,
    RowParallelFlashCommLinear,
)

try:
    import omni_training_custom_ops
except:
    logger.warning_once("Failed to import omni_training_custom_ops")
try:
    import omni_custom_ops
except:
    logger.warning_once("Failed to import omni_custom_ops")
from omni_npu.v1.utils import on_ascend950


class NPUDeepseekMLAAttention(torch.nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.default_cfg = {"device": current_platform.device_type, "dtype": config.dtype}

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.quant_symbol = quant_config is not None
        self.prefix = prefix
        self._init_wuk_t_uv = False
        self.is_pd_disagg = getattr(vllm_config, "kv_transfer_config", None) is not None
        scheduler_cfg = getattr(vllm_config, "scheduler_config", None)
        cache_cfg = getattr(vllm_config, "cache_config", None)
        task_cfg = getattr(model_extra_config, "task_config", None)
        self.enable_chunked_prefill = bool(getattr(
            scheduler_cfg, "enable_chunked_prefill",
            getattr(task_cfg, "enable_chunked_prefill", False)))
        self.enable_prefix_caching = bool(getattr(
            cache_cfg, "enable_prefix_caching",
            getattr(task_cfg, "enable_prefix_caching", False)))

        self.kv_nz = model_extra_config.operator_opt_config.kv_nz
        self.ena_sp = model_extra_config.parall_config.ena_seq_parallel
        self.on_ascend950 = on_ascend950()
        self.param_sink_number = getattr(config, "param_sink_number", 0)
        self.mla_absorb = (
            model_extra_config.operator_opt_config.enable_prefill_mla_absorb_pa or
            self.enable_chunked_prefill or
            self.enable_prefix_caching
        ) and not (self.param_sink_number > 0 and self.on_ascend950)
        self.use_mome = getattr(config, "use_mome", False)
        self.noncontiguous_kv = model_extra_config.operator_opt_config.use_noncontiguous_kv
        self.num_spec_tokens = vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config is not None else 0

        self.q_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_a_proj",
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelFlashCommLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelFlashCommLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )

        self.o_proj = RowParallelFlashCommLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
        self.rope_interleaved = getattr(config,"rope_interleaved", True)
        rope_scaling = getattr(config, "rope_scaling", None)
        is_mrope = rope_scaling is not None and rope_scaling.get("mrope_section") is not None
        if is_mrope:
            cache_layer = config.num_hidden_layers
            is_mtp_layer = getattr(config, "is_mtp_layer", False)
            if is_mtp_layer:
                cache_layer = config.num_nextn_predict_layers
            from vllm.model_executor.layers.rotary_embedding import get_rope_wrapper
            self.rotary_emb = get_rope_wrapper(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rotary_dim=qk_rope_head_dim,
                base=config.rope_parameters["rope_theta"],
                rope_scaling=rope_scaling,
                num_hidden_layers_cache=cache_layer
            )
        else:
            self.rotary_emb = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=False if self.rope_interleaved else True,
            )
        if config.rope_parameters["rope_type"] == "deepseek_yarn":
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        # SWA
        self.sliding_window = None
        if hasattr(config, "sliding_window") or hasattr(config, "sliding_window_list"): # same sliding window size or different
            layer_idx = extract_layer_index(prefix)
            if not hasattr(config, "swa_layers") or layer_idx in config.swa_layers:     # all swa layer or partly
                if hasattr(config, "sliding_window_list") and hasattr(config, "swa_layers") and layer_idx in config.swa_layers:
                    self.sliding_window = config.sliding_window_list[config.swa_layers.index(layer_idx)]
                elif hasattr(config, "sliding_window"):
                    self.sliding_window = config.sliding_window
            elif layer_idx >= config.num_hidden_layers:
                # MTP layer
                self.sliding_window = config.sliding_window_list[-1]

        if self.use_mome:
            if self.noncontiguous_kv:
                num_extra_token = 1 if self.is_pd_disagg else 0
                fake_num_spec_tokens = max(self.num_spec_tokens, num_extra_token)
                self.mome_state_shapes = (
                    (self.q_lora_rank,),
                    (self.kv_lora_rank,),
                    (self.num_heads * self.v_head_dim,),
                )
                self.mome_state_dtypes = (
                    torch.bfloat16,
                    torch.bfloat16,
                    torch.bfloat16,
                )
                self.kernel_size = getattr(config, 'router_sliding_window', 0)
                self.cache_dtype_str = None

                mome_kwargs = {
                    "kernel_size": self.kernel_size,
                    "num_spec_tokens": fake_num_spec_tokens,
                    "state_dtypes": self.mome_state_dtypes,
                    "state_shapes": self.mome_state_shapes,
                    "quant_config": None,
                    "vllm_config": vllm_config,
                    "prefix": f"{prefix}.conv",
                }
                self.conv = MomeAttention(**mome_kwargs)
            else:
                self.merge_q_kv_conv = model_extra_config.operator_opt_config.merge_q_kv_conv
                self.qa_conv = AggregateConv(self.q_lora_rank, config, vllm_config, output_parallel=False, attn_prefix=f"{prefix}.attn")
                self.compresskv_conv = AggregateConv(self.kv_lora_rank, config, vllm_config, output_parallel=False, attn_prefix=f"{prefix}.attn")
                if self.merge_q_kv_conv:
                    self.merge_conv = AggregateConv(self.q_lora_rank + self.kv_lora_rank, config, vllm_config, output_parallel=False, attn_prefix=f"{prefix}.attn")
                else:
                    self.merge_conv = None
                self.o_conv = AggregateConv(self.num_local_heads * self.v_head_dim, config, vllm_config, output_parallel=True, attn_prefix=f"{prefix}.attn")

        if self.param_sink_number == 0:
            self.attn = MLAAttention(
                num_heads=self.num_local_heads,
                scale=self.scaling,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
                kv_b_proj=self.kv_b_proj,
                use_sparse=False,
                indexer=None,
            )
        else:
            self.attn = StaticSinkMLAAttention(
                num_heads=self.num_local_heads,
                scale=self.scaling,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
                kv_b_proj=self.kv_b_proj,
                use_sparse=False,
                indexer=None,
                sink_len=self.param_sink_number,
                sliding_window=self.sliding_window,
            )
            self._register_sink_params(config)

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self.post_weight_load()

    def _register_sink_params(self, config):
        weight_attrs = {
            "output_dim": 1,
            "weight_loader": self.sink_kv_weight_loader,
        }

        self.param_sink_k_pe = torch.empty(
            self.param_sink_number, self.qk_rope_head_dim, **self.default_cfg)
        self.param_sink_k_pe = torch.nn.Parameter(self.param_sink_k_pe, requires_grad=False)
        set_weight_attrs(self.param_sink_k_pe, weight_attrs)

        self.param_sink_compressed_kv = torch.zeros(
            self.param_sink_number, self.kv_lora_rank, **self.default_cfg)
        if getattr(config, "param_sink_with_value", False):
            self.param_sink_compressed_kv = torch.nn.Parameter(self.param_sink_compressed_kv, requires_grad=False)
            set_weight_attrs(self.param_sink_compressed_kv, weight_attrs)

    def sink_kv_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        output_dim = getattr(param, "output_dim", None)
        is_sharded_weight = getattr(param, "is_sharded_weight", False)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow
        is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit
        # Special case for GGUF
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()
        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, nn.UninitializedParameter):
            final_shape = list(loaded_weight.shape)
            if output_dim is not None:
                tp_size = getattr(self, "tp_size", 1)
                assert final_shape[output_dim] % tp_size == 0
                final_shape[output_dim] = final_shape[output_dim] // tp_size
            param.materialize(final_shape, dtype=loaded_weight.dtype)
        param_data = param.data
        if output_dim is not None and not is_sharded_weight:
            shard_size = param_data.shape[output_dim]
            tp_rank = getattr(self, "tp_rank", 0)
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def post_weight_load(self):
        if self._init_wuk_t_uv and getattr(self.attn.impl, "W_UK_T", None) is not None:
            is_weight_nz = getattr(self.kv_b_proj.weight, "is_weight_nz", False)
            if is_weight_nz:
                self.kv_b_proj.weight.data = torch_npu.npu_format_cast(self.kv_b_proj.weight.data, torch_npu.Format.ND)
            self.attn.impl.process_weights_after_loading(self.kv_b_proj.weight.dtype)
            if is_weight_nz:
                self.kv_b_proj.weight.data = torch_npu.npu_format_cast(self.kv_b_proj.weight.data, torch_npu.Format.FRACTAL_NZ)
        else:
            self._init_wuk_t_uv = True
        if self.param_sink_number > 0:
            param_sink_compressed_kv = self.kv_a_layernorm(self.param_sink_compressed_kv)
            self.attn.update_sink_kv(self.param_sink_k_pe, param_sink_compressed_kv)
        if self.use_mome and not self.noncontiguous_kv:
            if self.merge_q_kv_conv and self.merge_conv is not None:
                self.merge_conv.merge_conv.weight.data = torch.cat([self.qa_conv.merge_conv.weight.data, self.compresskv_conv.merge_conv.weight.data], dim=0).contiguous()
                self.merge_conv.conv_weight = self.merge_conv.merge_conv.weight.data.squeeze(1).transpose(0, 1).contiguous()

    # ========================= linear =========================

    def _maybe_quant(self, x: torch.Tensor):
        if not self.quant_symbol:
            return x
        x_int8, scale = torch_npu.npu_dynamic_quant(x)
        return {"x_int8": x_int8, "pertoken_scale": scale}

    def _apply_rope(
        self,
        x: torch.Tensor,   # BNSD or TND or TD
        cos: torch.Tensor, # BNSD
        sin: torch.Tensor, # BNSD
    ) -> torch.Tensor:
        assert x.dim() in [2, 3, 4]
        assert cos.dim() == 4 and sin.dim() == 4
        T, D, shape = x.size(0), x.size(-1), x.shape
        x = x.view(T, -1, 1, D) # BNSD
        if self.rope_interleaved:
            x = torch_npu.npu_interleave_rope(x, cos, sin)
        else:
            x = torch_npu.npu_rotary_mul(x, cos, sin)
        return x.view(shape)

    def _q_absorb(
        self,
        q_lora: torch.Tensor, # TD
        cos: torch.Tensor,    # BNSD
        sin: torch.Tensor,    # BNSD
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Q, R = self.qk_nope_head_dim, self.qk_rope_head_dim
        tok = q_lora.size(0)
        q = self.q_b_proj(q_lora)[0].view(tok, -1, Q + R) # TND
        q_nope, q_pe = torch.split(q, [Q, R], dim=-1)     # TND
        q_pe = self._apply_rope(q_pe, cos, sin)           # TND
        q_nope = torch_npu.npu_transpose_batchmatmul(
            q_nope.transpose(0, 1),       # TND -> NTD
            weight=self.attn.impl.W_UK_T, # [Q, L]
            perm_y=(1, 0, 2),             # NTD -> TND
        )
        return q_nope, q_pe # TND, TND

    def _kv_norm_rope_cache(
        self,
        latent_kv: torch.Tensor, # TD
        cos: torch.Tensor,       # BNSD
        sin: torch.Tensor,       # BNSD
        slots: torch.Tensor | NPUMLAMetadata | None, # None for dummy_run
        kv_cache: tuple[torch.Tensor] | None,        # None for dummy_run
        fused_op: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]: # k_nope: T1D, k_pe: T1D
        R, L = self.qk_rope_head_dim, self.kv_lora_rank
        slots_2d = getattr(slots, "slot_mapping_2d", None)
        slots = getattr(slots, "slot_mapping", slots)
        assert latent_kv.dim() == 2 and latent_kv.size(1) == R + L
        # for pd-mixed with sequence parallel
        if slots is not None and slots.size(0) != latent_kv.size(0):
            n = slots.size(0)
            latent_kv, cos, sin = latent_kv[:n], cos[:n], sin[:n]
        valid_cache = None not in [kv_cache, slots]
        nope_cache, rope_cache = kv_cache or (None, None)
        fused_op = fused_op and not self.on_ascend950

        if valid_cache and fused_op and self.noncontiguous_kv:
            k_pe, k_nope = torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2(
                latent_kv.view(-1, 1, 1, L + R), # BNSD
                self.kv_a_layernorm.weight,
                cos, sin, # BNSD
                slots,
                cache_fit_shape(rope_cache, "4D"),
                cache_fit_shape(nope_cache, "4D"),
                k_rope_scale=None,
                k_rope_offset=None,
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode="PA_NZ" if self.kv_nz else "PA",
                rotary_mode="half" if not self.rope_interleaved else "interleave",
                quant_mode="none",
                is_output_kv=True,
            )
        elif valid_cache and fused_op and self.rope_interleaved:
            _, _, k_pe, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
                latent_kv.view(-1, 1, 1, L + R), # BNSD
                self.kv_a_layernorm.weight,
                cos, sin, # BNSD
                slots,
                cache_fit_shape(rope_cache, "4D"),
                cache_fit_shape(nope_cache, "4D"),
                k_rope_scale=None,
                k_rope_offset=None,
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode="PA_NZ" if self.kv_nz else "PA",
                is_output_kv=True,
            ) # -> [*, pg, 1, L], [*, pg, 1, R], BNSD, BNSD
        else:
            k_nope, k_pe = torch.split(latent_kv, [L, R], dim=-1) # TD
            k_nope = self.kv_a_layernorm(k_nope)                  # TD
            k_pe = self._apply_rope(k_pe, cos, sin)               # TD
            if valid_cache:
                def update(x: torch.Tensor, cache: torch.Tensor):
                    if self.on_ascend950:
                        assert slots_2d is not None
                        torch_npu.npu_scatter_nd_update_(cache, slots_2d, x)
                    else:
                        assert not self.noncontiguous_kv
                        torch_npu.npu_scatter_nd_update_(
                            cache.view(-1, 1, cache.size(-1)), # T1D
                            slots.view(-1, 1),                 # T1
                            x.view(-1, 1, x.size(-1)),         # T1D
                        )
                update(k_nope, nope_cache)
                update(k_pe, rope_cache)
        return k_nope.view(-1, 1, L), k_pe.view(-1, 1, R)

    def _post_attn_absorb(self, out: torch.Tensor) -> torch.Tensor:
        N = self.num_local_heads
        L = self.kv_lora_rank
        V = self.v_head_dim
        return torch_npu.npu_transpose_batchmatmul(
            out.view(N, -1, L), # NTD
            self.attn.impl.W_UV,
            perm_y=(1, 0, 2),
        ).reshape(-1, N * V) # T,ND

    # ========================= mome =========================

    def _apply_mome(self, x: torch.Tensor, state_indice, get_mome_args):
        if self.noncontiguous_kv:
            return self.conv(x, state_indice, **get_mome_args())
        conv = [self.qa_conv, self.compresskv_conv, self.o_conv]
        return x + conv[state_indice](x, **get_mome_args())

    def _maybe_mome_q(self, q: torch.Tensor, get_mome_args):
        return self._apply_mome(q, 0, get_mome_args) if self.use_mome else q

    def _maybe_mome_kv(self, kv: torch.Tensor, get_mome_args):
        if self.use_mome:
            L, R = self.kv_lora_rank, self.qk_rope_head_dim
            kv_c, k_pe = kv.split([L, R], dim=-1)
            kv_c = self._apply_mome(kv_c, 1, get_mome_args)
            kv = torch.cat([kv_c, k_pe], dim=-1)
        return kv

    def _maybe_mome_out(self, out: torch.Tensor, get_mome_args):
        if self.use_mome:
            if self.kv_b_proj.tp_size > 1:
                assert self.kv_b_proj.tp_size == get_tp_group().world_size
                out = get_tp_group().all_gather(out, dim=1)
            out = self._apply_mome(out, 2, get_mome_args)
            if self.o_proj.tp_size > 1:
                out = split_tensor_along_last_dim(out, num_partitions=self.o_proj.tp_size)
                out = out[self.o_proj.tp_rank].contiguous()
        return out

    # ========================= attention =========================

    @staticmethod
    def _insert_tensor_by_start_loc(
        raw_tensor: torch.Tensor,
        insert_segment: torch.Tensor,
        start_loc: list[int],
    ) -> torch.Tensor:
        segment_len = insert_segment.shape[0]
        num_inserts = len(start_loc) - 1
        total_len = segment_len * num_inserts + raw_tensor.shape[0]
        offset = 0
        # allocate result tensor
        result = torch.empty(total_len, *raw_tensor.shape[1:], device=raw_tensor.device, dtype=raw_tensor.dtype)

        for i in range(num_inserts):
            # write insert segment to result
            result[offset:offset+segment_len] = insert_segment
            offset += segment_len
            # write raw tensor to result
            seg_len = start_loc[i + 1] - start_loc[i]
            result[offset:offset+seg_len] = raw_tensor[start_loc[i]:start_loc[i+1]]
            offset += seg_len
        return result

    @attn_decorator(type="mla")
    def _apply_attention(
        self,
        q_nope: torch.Tensor,             # [T, N, kv_lora_rank]
        q_pe: torch.Tensor,               # [T, N, qk_rope_head_dim]
        keys: tuple[torch.Tensor] = None, # [2], TND or kv_cache, None for dummy_run
        q_cumlens: torch.Tensor = None,   # int32 [B], None for dummy_run
        kv_lens: torch.Tensor = None,     # int32 [B], None for dummy_run
        values: torch.Tensor = None,      # None for absorb, [T, N, v_head_dim]
        block_table: torch.Tensor = None, # int32 [B, *], PA only support absorb
        num_tokens: int = None,           # output shape, could be None in prefill
        layer_name: str = "",             # for capture_graph_task
        sink_k_nope: torch.Tensor = None, # Sink key for prefill sink attention
        sink_k_pe: torch.Tensor = None,   # Sink rope for prefill sink attention
        sink_v: torch.Tensor = None,      # Sink value for prefill sink attention
        attn_metadata: NPUMLAMetadata = None, # attn_metadata for sink attention
    ) -> torch.Tensor:                    # -> TND(no_absorb) or NTD(absorb)
        assert q_nope.dim() == q_pe.dim() and q_pe.dim() == 3
        assert keys is None or len(keys) == 2
        assert q_nope.size(1) == self.num_local_heads
        assert num_tokens is None or num_tokens >= q_nope.size(0)
        assert block_table is None or values is None # PA only support absorb

        if None in [q_cumlens, kv_lens, keys]:                        # dummy run
            return (torch.zeros_like(q_nope) if values is None else   # absorb
                q_nope.new_zeros(*q_nope.shape[:2], values.size(-1))) # no_absorb

        if self.param_sink_number > 0:
            return self._apply_sink_attention(
                q_nope, q_pe, keys, values,
                q_cumlens, kv_lens, block_table,
                num_tokens, layer_name,
                sink_k_nope, sink_k_pe, sink_v,
            )
        else:
            return self._apply_standard_attention(
                q_nope, q_pe, keys, values,
                q_cumlens, kv_lens, block_table,
                num_tokens, layer_name,
            )

    def _apply_sink_attention(
        self,
        q_nope: torch.Tensor,        # TND
        q_pe: torch.Tensor,          # TND
        keys: tuple[torch.Tensor],   # [2], TND or kv_cache
        values: torch.Tensor | None, # None for absorb
        q_cumlens: torch.Tensor,
        kv_lens: torch.Tensor,
        block_table: torch.Tensor,
        num_tokens: int,
        layer_name: str,
        sink_k_nope: torch.Tensor = None, # [sink_number, N, qk_nope_head_dim]
        sink_k_pe: torch.Tensor = None,   # [sink_number, N, qk_rope_head_dim]
        sink_v: torch.Tensor = None,      # [sink_number, N, v_head_dim]
    ) -> torch.Tensor:                    # TND(no_absorb) or NTD(absorb)
        valid_tok = q_cumlens[-1]
        q_heads = self.num_local_heads
        NPUMLAImpl.ensure_decode_attn_mask()
        if self.sliding_window is not None:
            window_size = self.sliding_window - 1
        else:
            window_size = NPUMLAImpl.MAX_WINDOW_SIZE

        if block_table is None: # not PA
            if self.on_ascend950:
                assert values is not None
                assert None not in [sink_k_nope, sink_k_pe, sink_v]
                q_cumlens_i64 = torch.as_tensor(
                    q_cumlens, dtype=torch.int64, device=q_nope.device)
                actual_seq_kvlen_i64 = q_cumlens_i64

                assert num_tokens is None or q_nope.size(0) == num_tokens
                k_nope, k_pe = keys
                num_tokens = num_tokens or q_nope.size(0)
                out = q_nope.new_zeros(num_tokens, q_heads, values.size(-1)) # TND
                query = torch.cat([q_nope[:valid_tok], q_pe[:valid_tok]], dim=-1)
                key = torch.cat([k_nope, k_pe], dim=-1)
                key_sink = torch.cat([sink_k_nope, sink_k_pe], dim=-1)
                meta_data = torch.ops.custom.npu_ai_infra_attention_pioneer_metadata(
                    q_heads, q_heads,
                    self.qk_head_dim,
                    self.v_head_dim,
                    actual_seq_lengths=q_cumlens_i64.npu(),
                    actual_seq_lengths_kv=actual_seq_kvlen_i64.npu(),
                    batch_size=q_cumlens_i64.numel(),
                    sparse_mode=4,
                    pre_tokens=window_size,
                    next_tokens=0,
                    input_layout="TND",
                    sink_number=self.param_sink_number,
                    rope_head_dim=self.qk_rope_head_dim,
                    soc_version="ascend950",
                )
                out[:valid_tok] = torch.ops.custom.npu_ai_infra_attention_pioneer(
                    query,
                    key,
                    values.contiguous(),
                    meta_data,
                    atten_mask=NPUMLAImpl.SHARE_MASK_TRIL_SPARSE,
                    actual_seq_lengths=q_cumlens_i64,
                    actual_seq_lengths_kv=actual_seq_kvlen_i64,
                    key_sink=key_sink,
                    value_sink=sink_v.contiguous(),
                    num_heads=q_heads,
                    softmax_scale=self.scaling,
                    pre_tokens=window_size,
                    next_tokens=0,
                    input_layout="TND",
                    num_key_value_heads=q_heads,
                    sparse_mode=4,
                    softmax_lse_flag=False,
                )[0]
                return out

            if self.noncontiguous_kv:
                assert None not in [sink_k_nope, sink_k_pe, sink_v]
                extra_args = {
                    "actual_seq_kvlen": q_cumlens,
                    "key_sink": sink_k_nope.contiguous(),
                    "value_sink": sink_v.contiguous(),
                    "key_rope_sink": sink_k_pe,
                }
            else:
                extra_args = {
                    "actual_seq_kvlen": kv_lens,
                    "sink_number": self.param_sink_number,
                }

            assert num_tokens is None or q_nope.size(0) == num_tokens
            k_nope, k_pe = keys
            values = k_nope if values is None else values # absorb or not
            out = q_nope.new_zeros(*q_nope.shape[:2], values.size(-1)) # TND
            out[:valid_tok] = torch.ops.custom.npu_fused_infer_attention_sink(
                query=q_nope[:valid_tok].contiguous(),
                query_rope=q_pe[:valid_tok],
                key=k_nope.contiguous(),
                value=values.contiguous(),
                key_rope=k_pe,
                num_query_heads=q_heads, # prefill does not need padding
                num_key_value_heads=q_heads,
                input_layout="TND",
                softmax_scale=self.scaling,
                sparse_mode=4,
                atten_mask=NPUMLAImpl.SHARE_MASK_TRIL_SPARSE,
                actual_seq_qlen=q_cumlens,
                pre_tokens=window_size,
                next_tokens=0,
                **extra_args,
            )[0] # -> TND
        else: # PA
            nope_cache, rope_cache = keys
            block_size = nope_cache.size(-2)
            nope_cache = cache_fit_shape(nope_cache, "3D")
            rope_cache = cache_fit_shape(rope_cache, "3D")
            kwargs = {
                "query": q_nope if self.on_ascend950 else q_nope[:valid_tok],
                "query_rope": q_pe if self.on_ascend950 else q_pe[:valid_tok],
                "key": nope_cache,
                "value": nope_cache,
                "key_rope": rope_cache,
                "num_query_heads": q_heads,
                "num_key_value_heads": 1,
                "input_layout": "TND",
                "softmax_scale": self.scaling,
                "block_table": block_table,
                "block_size": block_size,
                "actual_seq_qlen": q_cumlens,
                "actual_seq_kvlen": kv_lens,
                "atten_mask": NPUMLAImpl.SHARE_MASK_TRIL_SPARSE,
                "sparse_mode": 4,
                "pre_tokens": window_size,
                "next_tokens": 0,
            }
            if self.noncontiguous_kv:
                kwargs.update({
                    "key_sink": self.attn.impl.sink_compressed_kv,
                    "value_sink": self.attn.impl.sink_compressed_kv,
                    "key_rope_sink": self.attn.impl.sink_k_pe,
                })
                if model_extra_config.operator_opt_config.use_batch_invariant_op:
                    kwargs["batch_invariant"] = True
            else:
                kwargs.update({
                    "sink_number": self.param_sink_number,
                })
            num_tokens = num_tokens or q_nope.size(0)
            if self.on_ascend950:
                q_cumlens_i64 = torch.as_tensor(
                    q_cumlens, dtype=torch.int64, device=q_nope.device)
                kv_lens_i64 = torch.as_tensor(
                    kv_lens, dtype=torch.int64, device=q_nope.device)
                kwargs["key"] = kwargs["key"].contiguous()
                kwargs["value"] = kwargs["value"].contiguous()
                kwargs["key_rope"] = kwargs["key_rope"].contiguous()
                meta_data = torch.ops.custom.npu_ai_infra_attention_pioneer_metadata(
                    q_heads, 1,
                    self.kv_lora_rank,
                    self.kv_lora_rank,
                    actual_seq_lengths=q_cumlens_i64.npu(),
                    actual_seq_lengths_kv=kv_lens_i64.npu(),
                    batch_size=block_table.shape[0],
                    sparse_mode=4,
                    pre_tokens=window_size,
                    next_tokens=0,
                    input_layout="TND_NTD",
                    sink_number=self.param_sink_number,
                    rope_head_dim=self.qk_rope_head_dim,
                    block_size=block_size,
                    soc_version="ascend950",
                )
                return torch.ops.custom.npu_ai_infra_attention_pioneer(
                    kwargs["query"],
                    kwargs["key"],
                    kwargs["value"],
                    meta_data,
                    atten_mask=kwargs["atten_mask"],
                    actual_seq_lengths=q_cumlens_i64,
                    actual_seq_lengths_kv=kv_lens_i64,
                    block_table=kwargs["block_table"],
                    query_rope=kwargs["query_rope"],
                    key_rope=kwargs["key_rope"],
                    key_sink=kwargs.get("key_sink"),
                    value_sink=kwargs.get("value_sink"),
                    key_rope_sink=kwargs.get("key_rope_sink"),
                    num_heads=kwargs["num_query_heads"],
                    num_key_value_heads=kwargs["num_key_value_heads"],
                    softmax_scale=kwargs["softmax_scale"],
                    pre_tokens=kwargs["pre_tokens"],
                    next_tokens=kwargs["next_tokens"],
                    input_layout="TND_NTD",
                    sparse_mode=kwargs["sparse_mode"],
                    block_size=kwargs["block_size"],
                    softmax_lse_flag=False,
                )[0]

            out = q_nope.new_zeros(num_tokens, *q_nope.shape[1:]) # TND
            lse = q_nope.new_empty(num_tokens) # [T]
            if get_forward_context().capturing:
                capture_graph_task(
                    op_desc=OP_FIA_SINK,
                    op_kwargs=kwargs,
                    out_tensors=[out, lse],
                    num_tokens=num_tokens,
                    layer_name=layer_name,
                )
            else:
                out[:valid_tok] = torch.ops.custom.npu_fused_infer_attention_sink(**kwargs)[0]
            out = out.transpose(0, 1) # TND -> NTD
        return out

    def _apply_standard_attention(
        self,
        q_nope: torch.Tensor,        # TND
        q_pe: torch.Tensor,          # TND
        keys: tuple[torch.Tensor],   # [2], TND or kv_cache
        values: torch.Tensor | None, # None for absorb
        q_cumlens: torch.Tensor,
        kv_lens: torch.Tensor,
        block_table: torch.Tensor,
        num_tokens: int,
        layer_name: str,
    ) -> torch.Tensor: # TND(no_absorb) or NTD(absorb)
        valid_tok = q_cumlens[-1]
        q_head = q_nope.size(1)

        if block_table is None: # not PA
            k_nope, k_pe = keys
            metadata = get_forward_context().attn_metadata
            if isinstance(metadata, dict):
                metadata = metadata.get(f"{self.prefix}.attn")
            max_query_len = 1 if metadata is None else metadata.max_query_len
            sparse_mode = 3 if max_query_len > 1 else 0
            attn_mask = self.attn.impl.SHARE_MASK_TRIL_SPARSE if sparse_mode == 3 else None
            v_head = 1 if values is None else q_head
            value = k_nope if values is None else values
            assert value.dim() in [3, 4]

            assert num_tokens is None or num_tokens == q_nope.size(0)
            out = q_nope.new_zeros(*q_nope.shape[:2], value.size(-1))
            out[:valid_tok] = torch.ops.npu.npu_fused_infer_attention_score(
                q_nope[:valid_tok],
                k_nope,
                value,
                query_rope=q_pe[:valid_tok],
                key_rope=k_pe,
                num_heads=q_head,
                num_key_value_heads=v_head,
                input_layout="TND",
                atten_mask=attn_mask,
                sparse_mode=sparse_mode,
                actual_seq_lengths=q_cumlens,
                actual_seq_lengths_kv=kv_lens,
                scale=self.scaling,
                next_tokens=0,
            )[0] # -> TND
        else: # PA
            nope_cache, rope_cache = keys
            block_size = nope_cache.size(1)
            nope_cache = cache_fit_shape(nope_cache, "3D")
            rope_cache = cache_fit_shape(rope_cache, "3D")
            NPUMLAImpl.ensure_decode_attn_mask()
            kwargs = {
                "query": q_nope[:valid_tok],
                "key": nope_cache,
                "value": nope_cache,
                "query_rope": q_pe[:valid_tok],
                "key_rope": rope_cache,
                "num_heads": q_head,
                "num_key_value_heads": 1,
                "input_layout": "TND_NTD",
                "atten_mask": NPUMLAImpl.SHARE_MASK_TRIL_SPARSE,
                "sparse_mode": 3,
                "scale": self.scaling,
                "antiquant_mode": 0,
                "antiquant_scale": None,
                "block_table": block_table,
                "block_size": block_size,
                "actual_seq_lengths": q_cumlens,
                "actual_seq_lengths_kv": kv_lens,
            }

            num_tokens = num_tokens or q_nope.size(0)
            out = q_nope.new_zeros(q_head, num_tokens, self.kv_lora_rank) # NTD
            lse = q_nope.new_empty(num_tokens) # [T]
            if get_forward_context().capturing:
                capture_graph_task(
                    op_desc=OP_FIA_V1,
                    op_kwargs=kwargs,
                    out_tensors=[out, lse],
                    num_tokens=num_tokens,
                    layer_name=layer_name,
                )
            else:
                out[:, :valid_tok, :] = torch.ops.npu.npu_fused_infer_attention_score(**kwargs)[0]
        return out

    # ========================= forward =========================

    def forward(self, x, cos, sin): # adapter
        return torch.ops.vllm.npu_mla_forward(x, cos, sin, self.prefix)

    def _forward_decode(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: "NPUMLADecodeMetadata" = None,
        pd_mixed_flag: int = 0,
    ) -> torch.Tensor:
        assert attn_metadata is not None
        kv_cache = self.attn.kv_cache[get_forward_context().virtual_engine]
        def get_mome_args():
            if self.noncontiguous_kv:
                return {}
            else:
                return {
                    "force_decode": True if pd_mixed_flag == 1 else False,
                    "short_prefill": True if pd_mixed_flag == 2 else False,
                }
        x = self._maybe_quant(x)

        q_lora = self.q_a_proj(x)[0]                       # TD
        q_lora = self._maybe_mome_q(q_lora, get_mome_args) # TD
        q_lora = self.q_a_layernorm(q_lora)                # TD
        q_nope, q_pe = self._q_absorb(q_lora, cos, sin)    # TND

        kv = self.kv_a_proj_with_mqa(x)[0]          # TD
        kv = self._maybe_mome_kv(kv, get_mome_args) # TD
        self._kv_norm_rope_cache(kv, cos, sin, attn_metadata, kv_cache)

        out = self._apply_attention(
            q_nope, q_pe, # TND
            kv_cache,     # PA, absorb
            q_cumlens=attn_metadata.query_cumlens,
            kv_lens=attn_metadata.seq_lens,
            block_table=attn_metadata.block_table,
            num_tokens=q_nope.size(0),
            layer_name=f"{self.prefix}.attn", # for capture
        ) # -> NTD
        out = self._post_attn_absorb(out)              # NTD -> T,ND
        out = self._maybe_mome_out(out, get_mome_args) # [T, ND]
        return self.o_proj(out)[0]                     # TD

    def _forward_prefill(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: "NPUMLAPrefillMetadata" = None,
        pd_mixed_flag: bool = False,
    ) -> torch.Tensor:
        def get_mome_args():
            if self.noncontiguous_kv:
                return {"is_prefill": True}
            else:
                return {"only_prefill": bool(pd_mixed_flag)}

        if not self.ena_sp:
            x = self._maybe_quant(x)
        use_prefill_absorb = self.mla_absorb
        if use_prefill_absorb:
            return self._forward_prefill_absorb_pa(x, cos, sin, get_mome_args, attn_metadata)
        return self._forward_prefill_standard(x, cos, sin, get_mome_args, attn_metadata)

    def _forward_prefill_absorb_pa(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        get_mome_args,
        attn_metadata: "NPUMLAPrefillMetadata",
    ) -> torch.Tensor:
        sp_manager = SPManager.init_sp(tok=cos.size(0)) if self.ena_sp else None

        if attn_metadata:
            kv_cache = self.attn.kv_cache[get_forward_context().virtual_engine]
            q_cumlens = attn_metadata.query_cumlens
            kv_lens = attn_metadata.seq_lens
            block_table = attn_metadata.block_table
        else: # dummy_run
            q_cumlens, kv_lens, block_table = None, None, None
            kv_cache = None

        q = self.q_a_proj(x)[0]                    # TD
        if self.ena_sp and self.use_mome:
            q = sp_manager.ag_tokens(q)            # TD
        q = self._maybe_mome_q(q, get_mome_args)   # TD
        q = self.q_a_layernorm(q)                  # TD
        if self.ena_sp and not self.use_mome:
            q = sp_manager.ag_tokens(q)            # TD
        q_nope, q_pe = self._q_absorb(q, cos, sin) # TND

        kv = self.kv_a_proj_with_mqa(x)[0]          # TD
        if self.ena_sp:
            kv = sp_manager.ag_tokens(kv)           # TD
        kv = self._maybe_mome_kv(kv, get_mome_args) # TD
        self._kv_norm_rope_cache(kv, cos, sin, attn_metadata, kv_cache)

        out = self._apply_attention(
            q_nope, q_pe,            # TND
            kv_cache,                # PA, absorb
            q_cumlens, kv_lens,      # None for dummy_run
            block_table=block_table, # [B, *]
            attn_metadata=attn_metadata,
        ) # -> NTD
        out = self._post_attn_absorb(out) # NTD -> T,ND
        out = self._maybe_mome_out(out, get_mome_args)

        if sp_manager and self.o_proj.tp_size > 1:
            out = sp_manager.align_tokens(out)
        out = self.o_proj(out)[0] # T,ND -> TD
        if sp_manager and self.o_proj.tp_size == 1:
            out = sp_manager.slice_tokens(out)
        return out

    def _forward_prefill_standard(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        get_mome_args,
        attn_metadata: "NPUMLAPrefillMetadata",
    ) -> torch.Tensor:
        N = self.num_local_heads
        R = self.qk_rope_head_dim
        V = self.v_head_dim
        QK = self.qk_nope_head_dim

        sp_manager = SPManager.init_sp(tok=cos.size(0)) if self.ena_sp else None

        if attn_metadata:
            q_cumlens = kv_cumlens = attn_metadata.query_cumlens
            kv_cache = self.attn.kv_cache[get_forward_context().virtual_engine]
        else: # dummy_run
            q_cumlens, kv_cumlens, kv_cache = None, None, None
        sink_k_nope, sink_k_pe, sink_v = None, None, None

        q = self.q_a_proj(x)[0]                  # TD
        if self.ena_sp and self.use_mome:
            q = sp_manager.ag_tokens(q)          # TD
        q = self._maybe_mome_q(q, get_mome_args) # TD
        q = self.q_a_layernorm(q)                # TD
        if self.ena_sp and not self.use_mome:
            q = sp_manager.ag_tokens(q)          # TD

        q = self.q_b_proj(q)[0].view(-1, N, QK + R)     # TND
        q_nope, q_pe = torch.split(q, [QK, R],  dim=-1) # TND
        q_pe = self._apply_rope(q_pe, cos, sin)         # TND

        kv = self.kv_a_proj_with_mqa(x)[0] # TD
        if self.ena_sp:
            kv = sp_manager.ag_tokens(kv)             # TD
        kv = self._maybe_mome_kv(kv, get_mome_args)   # TD
        kv_a, k_pe = self._kv_norm_rope_cache(
            kv, cos, sin, attn_metadata, kv_cache)    # T1D
        kv_a, k_pe = kv_a.squeeze(1), k_pe.squeeze(1) # TD

        if self.param_sink_number > 0:
            if self.noncontiguous_kv:
                sink_kv = self.kv_b_proj.forward(self.attn.sink_compressed_kv)[0] # [T, ND]
                sink_kv = sink_kv.view(-1, N, QK + V)                             # TND
                sink_k_nope, sink_v = torch.split(sink_kv, [QK, V], dim=-1)       # TND
                sink_k_pe = self.attn.sink_k_pe.view(-1, 1, R).repeat(1, N, 1)    # TND
            elif q_cumlens is not None: # avoid dummy_run
                # prepend sink tokens
                k_pe = self._insert_tensor_by_start_loc(
                    k_pe, self.attn.sink_k_pe, attn_metadata.query_start_loc) # TD
                kv_a = self._insert_tensor_by_start_loc(
                    kv_a, self.attn.sink_compressed_kv, attn_metadata.query_start_loc) # TD
                # apply kv_cumlens offset
                sink_len_offset = [self.param_sink_number * (i + 1) for i in range(len(q_cumlens))]
                kv_cumlens = [a + b for a, b in zip(q_cumlens, sink_len_offset)]

        k_pe = k_pe.view(-1, 1, R).repeat(1, N, 1)   # TND
        kv = self.kv_b_proj.forward(kv_a)[0]         # TD
        kv = kv.view(-1, N, QK + V)                  # TND
        k_nope, v = torch.split(kv, [QK, V], dim=-1) # TND

        out = self._apply_attention(
            q_nope, q_pe,            # TND
            (k_nope, k_pe),          # TND
            q_cumlens, kv_cumlens,   # None for dummy_run
            values=v,                # no_absorb
            sink_k_nope=sink_k_nope, # TND
            sink_k_pe=sink_k_pe,     # TND
            sink_v=sink_v,           # TND
            attn_metadata=attn_metadata,
        ).view(-1, N * V)            # [T, ND]

        out = self._maybe_mome_out(out, get_mome_args) # fit o_proj.tp_size
        if sp_manager and self.o_proj.tp_size > 1:
            out = sp_manager.align_tokens(out)
        out = self.o_proj(out)[0] # T,ND -> TD
        if sp_manager and self.o_proj.tp_size == 1:
            out = sp_manager.slice_tokens(out)
        return out


def npu_mla_forward(
    hs: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    forward_context = get_forward_context()
    self: NPUDeepseekMLAAttention = forward_context.no_compile_layers[layer_name]
    attn_metadata: NPUMLAMetadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[f"{self.prefix}.attn"]
    p_slice, d_slice, has_prefill, has_decode = get_batch_desc(attn_metadata)
    prefill = getattr(attn_metadata, "prefill", None) # None for dummy_run
    decode = getattr(attn_metadata, "decode", None)   # None for dummy_run

    if self.param_sink_number > 0 and not self.noncontiguous_kv:
        assert self.attn.sink_k_pe is not None and self.attn.sink_compressed_kv is not None, (
            "sink_k_pe and sink_compressed_kv have not been prepared"
        )
        if not self.attn.sink_populated:
            self_kv_cache = self.attn.kv_cache[forward_context.virtual_engine]
            if self_kv_cache is not None and len(self_kv_cache) > 0:
                self.attn.populate_sink_kv(self_kv_cache[0], self_kv_cache[1])

    if has_decode and has_prefill:
        with sp_disabled(self, hs) as (x, y, out):
            y[p_slice] = self._forward_prefill(
                x[p_slice], cos[p_slice], sin[p_slice], prefill, pd_mixed_flag=True)
            # short prefill in decode or pure decode
            pd_mixed_flag = 2 if attn_metadata.num_decode_tokens > attn_metadata.num_decodes else 1
            y[d_slice] = self._forward_decode(
                x[d_slice], cos[d_slice], sin[d_slice], decode, pd_mixed_flag)
    elif has_prefill:
        # SP: hidden is this rank's slice, cos/sin are full-length, matching _forward_prefill's ag_tokens;
        # Non-SP: slice the effective prefill segment by global batch index.
        out = lazy_zero_like(hs)
        if self.ena_sp:
            out[:] = self._forward_prefill(hs, cos, sin, prefill)
        else:
            out[p_slice] = self._forward_prefill(
                hs[p_slice], cos[p_slice], sin[p_slice], prefill)
    else: # has_decode
        with sp_disabled(self, hs) as (x, y, out):
            y[d_slice] = self._forward_decode(
                x[d_slice], cos[d_slice], sin[d_slice], decode)
    return out.tensor()

def npu_mla_forward_fake(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="npu_mla_forward",
    op_func=npu_mla_forward,
    mutates_args=[],
    fake_impl=npu_mla_forward_fake,
    dispatch_key="PrivateUse1",
)
