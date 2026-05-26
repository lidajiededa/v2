# Pangu V2 MoE adaptation notes

This workspace targets OpenPangu-505B on vLLM/vllm-ascend 0.18.0 with four
Ascend 910B nodes, `DP=4`, `TP=8`, and expert parallelism.

## Model implementation

The local `config.json` uses `model_type=pangu_v2_moe` and requires:

- MLA: `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`
- DSA: `dsa_layers`, `index_topk`, `index_head_dim`
- SWA: `swa_layers`, `sliding_window_list`
- static sink MLA: `param_sink_number`, `param_sink_with_value`
- MoME: `use_mome`, `router_sliding_window`
- MHC: `use_mhc`, `block_post_layernorm_idx`

vLLM 0.18.0 has an upstream OpenPangu MoE implementation, but it does not cover
the full Pangu V2 DSA/SWA/SinkMLA/MoME/MHC path used by this checkpoint. The
plugin therefore bundles the omni-npu `release_1.2.1` `omni_npu` package under
`src/omni_npu` and registers:

```text
PanguV2MoEForCausalLM -> omni_npu.v1.models.pangu.pangu_ultra_moe:PanguUltraMoEForCausalLM
```

The vllm-ascend image does not need a preinstalled omni-npu package after this
adapter is installed. `VLLM_PANGU_V2_MODEL_IMPL=fallback` is kept only for
non-NPU import smoke tests.

## Required model config normalization

Run once on the model directory:

```bash
cd /mnt/d/workspace/pangu-v2-moe-vllm-ascend018
bash scripts/prepare_model_config.sh /path/to/openpangu-505B
```

This fixes:

- `architectures` -> `["PanguV2MoEForCausalLM"]`
- `rope_parameters.rope_theta` -> top-level `rope_theta`
- `pad_token_id` -> `eos_token_id` if missing
- `layer_types` sentinel for hybrid detection

## Install on every node

Inside the vllm-ascend 0.18.0 NPU Python environment:

```bash
cd /mnt/d/workspace/pangu-v2-moe-vllm-ascend018
bash scripts/install_plugin_only.sh
```

Use `scripts/install_editable.sh` only if you also want editable vLLM/vLLM-Ascend
source installs from `third_party/`.

## 4-node DP4TP8 mixed bring-up

vllm-ascend 0.18.0's current multi-node serving pattern for this shape is to run
one `vllm serve` process per DP rank. Node0 starts the API server; the other
nodes run the same command with `--headless`.

On node0:

```bash
bash scripts/run_4node_mixed_node.sh \
  /path/to/openpangu-505B \
  10.0.0.1 \
  0 \
  10.0.0.1 \
  8000
```

On node1..node3:

```bash
bash scripts/run_4node_mixed_node.sh /path/to/openpangu-505B 10.0.0.1 1 10.0.0.2 8000
bash scripts/run_4node_mixed_node.sh /path/to/openpangu-505B 10.0.0.1 2 10.0.0.3 8000
bash scripts/run_4node_mixed_node.sh /path/to/openpangu-505B 10.0.0.1 3 10.0.0.4 8000
```

The defaults are:

- `DP_SIZE=4`, `DP_LOCAL_SIZE=1`, `TP_SIZE=8`
- `--distributed-executor-backend mp`
- `--enable-expert-parallel`
- `--no-disable-hybrid-kv-cache-manager`
- `--no-enable-prefix-caching`
- `--block-size 128`
- `--enforce-eager`
- `VLLM_ASCEND_ENABLE_MLAPO=0`
- `VLLM_ASCEND_ENABLE_FUSED_MC2=0`
- `VLLM_PANGU_V2_KV_DEBUG=1`
- `CUSTOM_MODEL_CONFIG_PATH=low_latency/openpangu_v2/pangu_v2_moe_bf16_a2_dp4tp8_hybrid.json`

The bundled A2 config disables weight prefetch sizes and A3-oriented fused
paths by default. Override `CUSTOM_MODEL_CONFIG_PATH` only after the service can
finish KV cache initialization.

Increase `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, and `GPU_MEMORY_UTILIZATION` only
after KV cache initialization succeeds.

## KV OOM checks

If D node still OOMs immediately after model load, check the log lines emitted
by `VLLM_PANGU_V2_KV_DEBUG=1`. The important fields are:

- number of KV groups
- layer count in each group
- `block_size`
- `page_size_bytes`

For this model, a wrong architecture fallback usually shows up as ordinary
OpenPangu/MLA grouping instead of the bundled Pangu V2 hybrid path. A wrong
block size is also risky for hybrid attention plus MoME state allocation, so the
launch script pins `--block-size 128`.
