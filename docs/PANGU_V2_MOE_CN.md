# OpenPangu-505B PanguV2MoE vllm-ascend 0.18.0 适配说明

本文档说明在 4 台 Ascend 910B 机器上用 `DP=4, TP=8` 混布启动
OpenPangu-505B `PanguV2MoEForCausalLM` 的方式，并解释 vLLM 启动该模型时
从命令行到模型加载、显存 profiling、KV cache 分配的主要逻辑。

## 目录结构

适配工程目录：

```text
/mnt/d/workspace/pangu-v2-moe-vllm-ascend018
```

关键文件：

- `src/vllm_pangu_v2_moe/plugin.py`：vLLM 插件入口，注册模型架构和补丁。
- `src/vllm_pangu_v2_moe/config.py`：修正 `config.json` 中的架构名、RoPE、pad token 和 hybrid 标记。
- `src/omni_npu/`：随适配包一起安装的 PanguV2MoE NPU 实现。
- `scripts/run_4node_mixed_node.sh`：4 机 `DP=4, TP=8` 启动脚本。
- `scripts/prepare_model_config.sh`：修正模型目录下的 `config.json`。
- `src/omni_npu/model_config/configs/low_latency/openpangu_v2/pangu_v2_moe_bf16_a2_dp4tp8_hybrid.json`：
  910B/A2 默认运行配置。

## 为什么不能只用 vLLM 原生 OpenPangu

你的模型 `config.json` 里模型类型是：

```json
{
  "model_type": "pangu_v2_moe",
  "architectures": ["PanguV2MoEForCausalLM"]
}
```

该模型不仅是普通 MoE，还用到了：

- MLA：`q_lora_rank`、`kv_lora_rank`、`qk_nope_head_dim`、`qk_rope_head_dim`
- DSA：`dsa_layers`、`index_topk`、`index_head_dim`
- SWA：`swa_layers`、`sliding_window_list`
- Sink MLA：`param_sink_number`、`param_sink_with_value`
- MoME：`use_mome`、`router_sliding_window`
- MHC：`use_mhc`、`block_post_layernorm_idx`

vLLM 0.18.0 自带的 OpenPangu 路径不能完整覆盖这些 PanguV2MoE 逻辑。因此本适配包把
omni-npu `release_1.2.1` 里的 `omni_npu` 实现带进来，并通过插件把架构注册为：

```text
PanguV2MoEForCausalLM
  -> omni_npu.v1.models.pangu.pangu_ultra_moe:PanguUltraMoEForCausalLM
```

这样 vllm-ascend 镜像里即使没有预装 omni-npu，也能使用包内实现。

## 安装

每台机器都执行：

```bash
cd /mnt/d/workspace/pangu-v2-moe-vllm-ascend018
bash scripts/install_plugin_only.sh
```

如果还需要把本目录下下载的 vLLM/vllm-ascend 源码也 editable 安装，才使用：

```bash
bash scripts/install_editable.sh
```

一般在正式 vllm-ascend 0.18.0 镜像里，只需要 `install_plugin_only.sh`。

## 修正模型 config.json

首次启动前，对模型目录执行一次：

```bash
cd /mnt/d/workspace/pangu-v2-moe-vllm-ascend018
bash scripts/prepare_model_config.sh /path/to/openpangu-505B
```

它会修正：

- `architectures` 改为 `["PanguV2MoEForCausalLM"]`
- 增加 `rope_parameters.rope_theta`
- 缺少 `pad_token_id` 时使用 `eos_token_id`
- 对 `use_mome=true` 的模型增加 `layer_types`，让 vLLM 走 hybrid 检测

## 4 机 DP4TP8 启动

当前脚本采用 vllm-ascend 0.18.0 推荐的多机 MP 方式：每台机器都启动一个
`vllm serve` 进程，每台机器是 1 个 DP rank，每个 DP rank 内部使用 8 张卡做 TP。

node0 是 API server：

```bash
bash scripts/run_4node_mixed_node.sh /path/to/openpangu-505B 10.0.0.1 0 10.0.0.1 8000
```

node1 到 node3 是 headless worker：

```bash
bash scripts/run_4node_mixed_node.sh /path/to/openpangu-505B 10.0.0.1 1 10.0.0.2 8000
bash scripts/run_4node_mixed_node.sh /path/to/openpangu-505B 10.0.0.1 2 10.0.0.3 8000
bash scripts/run_4node_mixed_node.sh /path/to/openpangu-505B 10.0.0.1 3 10.0.0.4 8000
```

脚本默认参数：

```bash
DP_SIZE=4
DP_LOCAL_SIZE=1
TP_SIZE=8
MAX_MODEL_LEN=4096
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=128
GPU_MEMORY_UTILIZATION=0.75
```

默认会设置：

```bash
export VLLM_USE_V1=1
export VLLM_PLUGINS=ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling,pangu_v2_moe
export VLLM_PANGU_V2_MODEL_IMPL=auto
export VLLM_PANGU_V2_KV_DEBUG=1
export VLLM_ASCEND_ENABLE_MLAPO=0
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export CUSTOM_MODEL_CONFIG_PATH=low_latency/openpangu_v2/pangu_v2_moe_bf16_a2_dp4tp8_hybrid.json
```

`vllm serve` 关键参数：

```bash
--data-parallel-size 4
--data-parallel-size-local 1
--data-parallel-start-rank ${NODE_RANK}
--data-parallel-address ${MASTER_IP}
--data-parallel-rpc-port ${RPC_PORT}
--tensor-parallel-size 8
--distributed-executor-backend mp
--enable-expert-parallel
--no-disable-hybrid-kv-cache-manager
--no-enable-prefix-caching
--block-size 128
--enforce-eager
```

node0 会额外加：

```bash
--api-server-count 4
```

node1 到 node3 会额外加：

```bash
--headless
```

## vLLM 启动该模型的主流程

下面按实际启动顺序解释。

### 1. 命令行解析

执行 `vllm serve /path/to/openpangu-505B ...` 后，vLLM 先解析命令行参数，生成
`VllmConfig`。其中比较关键的是：

- `ModelConfig`：模型路径、`hf_config`、架构名、dtype、trust remote code 等。
- `ParallelConfig`：DP、TP、EP、distributed executor backend。
- `SchedulerConfig`：`max_num_seqs`、`max_num_batched_tokens`、hybrid KV manager 开关。
- `CacheConfig`：`block_size`、`gpu_memory_utilization`、KV cache dtype。

### 2. 插件加载

vLLM 会加载 `vllm.general_plugins` entry point。本适配包在 `pyproject.toml` 中注册：

```toml
[project.entry-points."vllm.general_plugins"]
pangu_v2_moe = "vllm_pangu_v2_moe:register"
```

由于脚本设置了：

```bash
VLLM_PLUGINS=ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling,pangu_v2_moe
```

所以 `pangu_v2_moe` 插件会被加载并执行 `register()`。

插件做几件事：

- 注册 `PanguV2MoEForCausalLM` 到 vLLM `ModelRegistry`。
- 兼容原始 config 里的错误架构名 `PanguV2aMoEForCausalLM`。
- 修正 `ModelConfig.__post_init__`，保证读取到的 `hf_config` 被标准化。
- 给 vllm-ascend 的 modelslim mapping 增加 `pangu_v2_moe`。
- 给 vllm-ascend worker 补加载 omni `model_extra_config` 的逻辑。
- 打开 KV cache 诊断日志。

### 3. 架构注册和模型类选择

插件默认 `VLLM_PANGU_V2_MODEL_IMPL=auto`。只要本适配包安装成功，Python 能 import 到
`omni_npu`，就会注册为：

```text
omni_npu.v1.models.pangu.pangu_ultra_moe:PanguUltraMoEForCausalLM
```

这一步很重要。如果注册失败，vLLM 会找不到 `PanguV2MoEForCausalLM`，或者退到不完整的
OpenPangu fallback，后续 hybrid KV、MoME、MHC 等行为就会错。

### 4. 多进程和分布式初始化

本方案使用：

```bash
--distributed-executor-backend mp
--data-parallel-size 4
--data-parallel-size-local 1
--tensor-parallel-size 8
```

含义是：

- 全局有 4 个 DP rank。
- 每台机器只放 1 个本地 DP rank。
- 每个 DP rank 内用本机 8 张 NPU 做 TP。
- MoE 使用 `--enable-expert-parallel`。

node0 是 API server，node1 到 node3 是 headless worker。它们通过
`--data-parallel-address` 和 `--data-parallel-rpc-port` 建立 DP 侧通信。

### 5. vllm-ascend NPUWorker 初始化

进入 vllm-ascend 后，会创建 `NPUWorker`。其初始化大致包括：

- 设置 `npu:{local_rank}`。
- 清理 NPU cache。
- 根据 `gpu_memory_utilization` 计算允许使用的显存目标。
- 初始化 HCCL/TP/DP 分布式环境。
- 初始化 workspace manager。
- 创建 `NPUModelRunner`。

本适配包在 `NPUWorker.init_device()` 后补了一步：

```text
load_model_extra_config(model_config, vllm_config, scheduler_config)
```

这会读取：

```text
CUSTOM_MODEL_CONFIG_PATH=low_latency/openpangu_v2/pangu_v2_moe_bf16_a2_dp4tp8_hybrid.json
```

该 A2 配置主要关闭 A3 倾向的激进预取/融合路径，避免 910B 上模型刚加载完、KV cache
还没分配时就被额外 workspace 顶爆。

### 6. 模型构建和权重加载

`NPUModelRunner.load_model()` 会调用 vLLM 的 `get_model(vllm_config=...)`。

此时 vLLM 根据 `hf_config.architectures` 找到：

```text
PanguV2MoEForCausalLM
```

再通过 `ModelRegistry` 找到插件注册的类：

```text
omni_npu.v1.models.pangu.pangu_ultra_moe:PanguUltraMoEForCausalLM
```

随后构建模型模块并加载权重。你之前说“权重加载没问题，加载完后 OOM”，说明大概率已经过了
这个阶段，问题在后面的 profiling 或 KV cache 初始化阶段。

### 7. profiling 阶段计算可用于 KV cache 的显存

模型加载后，vLLM/vllm-ascend 会执行一次 profile run，用 dummy input 跑一遍模型，统计：

```text
non_kv_cache_memory = 权重 + 临时激活 + workspace + 运行期额外内存
available_kv_cache_memory = requested_memory - non_kv_cache_memory
```

其中：

```text
requested_memory = total_device_memory * gpu_memory_utilization
```

如果这里 profile run 使用了错误模型路径、错误 hybrid 设置、过大的 batch/token 配置，或者额外预取内存过大，
就会导致后续 KV cache 可用显存非常小，甚至直接 OOM。

### 8. KV cache config 生成和分配

profiling 后，vLLM 根据可用显存、`block_size`、attention 类型、hybrid KV 分组等生成
`KVCacheConfig`，然后调用 vllm-ascend：

```text
NPUModelRunner.initialize_kv_cache(kv_cache_config)
```

这里会做：

- 按 KV cache group 初始化 attention backend。
- 判断是否有多个 attention group，即 hybrid blocks。
- 分配原始 KV cache tensor。
- reshape 成每层 attention 需要的 KV cache 形状。
- 绑定到每一层 attention。

本模型同时有 MLA、DSA、SWA、MoME/Mamba-like state，因此必须让 hybrid KV manager 生效。
脚本中使用：

```bash
--no-disable-hybrid-kv-cache-manager
--block-size 128
```

目的就是避免所有层被错误地按同一种 KV cache 规格粗暴分配。

### 9. 为什么 OOM 会发生在“加载后”

你观察到“模型加载完不 OOM，加载后 OOM”，通常对应这两个阶段之一：

- profile run：dummy forward 触发额外 workspace、临时 tensor 或预取内存。
- KV cache allocation：根据错误的 KV 规格、错误 block size、错误 hybrid 分组分配过大 tensor。

如果无论 `max_model_len`、`max_num_seqs`、`gpu_memory_utilization` 怎么调都会 OOM，要优先怀疑：

- 没有真正加载到 `PanguV2MoEForCausalLM -> bundled omni_npu`。
- `config.json` 里架构名还是错误的 `PanguV2aMoEForCausalLM`。
- hybrid KV manager 没生效。
- `block_size` 被 vLLM/vllm-ascend 改成不适合 hybrid state 的值。
- A3 的 prefetch 或 fused path 在 910B 上仍然打开。
- `model_extra_config` 没有加载，导致默认开关不符合 PanguV2MoE/A2。

## 排查日志重点

启动时先看这些日志：

```text
Pangu V2 MoE vLLM plugin registered.
Registering PanguV2MoEForCausalLM with bundled omni_npu PanguUltraMoE implementation.
Loaded bundled omni_npu model_extra_config for PanguV2MoEForCausalLM.
Loading model weights took ...
Available KV cache memory: ...
PanguV2 KV group[...]
```

如果没有看到 `bundled omni_npu PanguUltraMoE implementation`，说明没有走到包内 PanguV2MoE 实现。

如果没有看到 `PanguV2 KV group[...]`，说明 `VLLM_PANGU_V2_KV_DEBUG=1` 没生效，或者 KV 初始化还没进入。

如果看到 KV group 只有普通 OpenPangu/MLA 的分组，而不是 hybrid 多组，要重点查：

- `config.json` 是否已被 `prepare_model_config.sh` 修正。
- 是否带了 `--no-disable-hybrid-kv-cache-manager`。
- 插件是否在所有 worker 进程都被加载。

## 建议的第一次启动参数

第一次只为了跑通，不追性能：

```bash
export MAX_MODEL_LEN=4096
export MAX_NUM_SEQS=1
export MAX_NUM_BATCHED_TOKENS=128
export GPU_MEMORY_UTILIZATION=0.75
export VLLM_PANGU_V2_KV_DEBUG=1
```

跑通 KV cache 初始化后，再逐步提升：

1. 先提升 `GPU_MEMORY_UTILIZATION`，例如 `0.80`、`0.85`。
2. 再提升 `MAX_MODEL_LEN`。
3. 最后提升 `MAX_NUM_SEQS` 和 `MAX_NUM_BATCHED_TOKENS`。
4. 性能稳定后再考虑打开 fused/multistream/prefetch 类优化。

不要在 KV cache 尚未稳定前同时打开多个优化开关。
