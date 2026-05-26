# https://gitee.com/omniai/omni-npu/blob/release_1.2.1 我使用omni-npu的这个分支，在8机910B上1p1d部署openpangu-505B,模型架构是pangu_v2_moe，走的分支是配置是https://gitee.com/omniai/omni-npu/blob/release_1.2.1/src/omni_npu/model_config/configs/low_latency/openpangu_v2/pangu_v2_moe_bf16_a3_xp1d_p.json，原本这个只是支持a3（910C）机器，我们修改了他让他支持了a2（910B）机器。你能分析下要怎么启动服务吗，以及启动服务后为什么d节点一直会oom

> https://gitee.com/omniai/omni-npu/blob/release_1.2.1 我使用omni-npu的这个分支，在8机910B上1p1d部署openpangu-505B,模型架构是pangu_v2_moe，走的分支是配置是https://gitee.com/omniai/omni-npu/blob/release_1.2.1/src/omni_npu/model_config/configs/low_latency/openpangu_v2/pangu_v2_moe_bf16_a3_xp1d_p.json，原本这个只是支持a3（910C）机器，我们修改了他让他支持了a2（910B）机器。你能分析下要怎么启动服务吗，以及启动服务后为什么d节点一直会oom

<details><summary>7 previous messages</summary>

> 我先把本地仓库和你给的 `release_1.2.1` 配置对上看一遍，重点找启动入口、1p1d/low_latency 的部署脚本，以及 `pangu_v2_moe` 在 A2/A3 上的内存相关参数。
>
> <details><summary>Searched web 1 time</summary>
>
> <details><summary>Searched web 1 time</summary>
>
> - Searched web for `site:gitee.com/omniai/omni-npu release_1.2.1 pangu_v2_moe_bf16_a3_xp1d_p.json ...`
>
> </details>
>
> </details>
>
> 本地目录看起来只有 `.git`，像是还没 checkout 出工作树。我会直接从 Gitee raw/仓库侧取配置和文档，同时看本地 git remote，避免基于空目录乱猜。
>
> 远端也没有配置，当前目录只是一个空的 git 工作树。接下来我把 `omni-npu` 的 `release_1.2.1` 浅克隆到临时目录，只读分析启动脚本和配置，不碰你当前目录。
>
> 已经拿到关键线索：仓库里除了你提到的 low_latency `xp1d_p`，还有配套 `xp1d_d`，以及 high_throughout 下的 `openpangu_v2_505b_bf16_a3_1p1d_p/d`。我会把这些配置差异和启动脚本里的 PD 参数串起来看。
>
> 这里有个很可能踩坑点：配置加载器不是看 `--kv-transfer-config` 判断 P/D，而是看环境变量 `ROLE=prefill|decode`，并且用 `PREFILL_POD_NUM`/`DECODE_POD_NUM` 选择 `1P1D`、`2P1D` 等配置。也就是说启动命令里缺这些变量时，可能会加载成 hybrid 或默认配置。
>
> <details><summary>Searched web 4 times</summary>
>
> <details><summary>Searched web 4 times</summary>
>
> - Searched web for `vLLM additional_config CLI --additional-config ...`
> - Searched web for `gitee.com/omniai/omni-npu/blob/release_1.2.1/src/omni_npu/model_config/config_loader/loader.py ...`
> - Searched web for `https://docs.vllm.ai/en/v0.14.0/cli/serve/`
> - Searched web for `'--distributed-executor-backend' in https://docs.vllm.ai/en/v0.14.0/cli/serve/`
>
> </details>
>
> </details>
</details>

我按 `release_1.2.1` 代码看下来，启动这套 1P1D 时最关键的不是只有 `--kv-transfer-config`，还要让 omni-npu 的配置加载器正确识别“这是 PD、这是 P 还是 D、要走 low_latency”。否则它很容易加载默认配置或 hybrid 配置，D 侧 OOM 就很常见。

**启动要点**

P、D 两边都要有这些环境变量：

```bash
export VLLM_PLUGINS="omni-npu,omni_npu_patches,omni_custom_models"
export OMNI_NPU_PATCHES_DIR="pangu_v2_moe"   # 如果自动识别不稳，建议显式指定
export PREFILL_POD_NUM=1
export DECODE_POD_NUM=1

# 910B 每台机器实际可见多少个 die/卡就填多少；代码默认是 16
export NUM_DIE_PER_MACH=8

# 改成你们实际 RoCE/NIC
export GLOO_SOCKET_IFNAME=xxx
export HCCL_SOCKET_IFNAME=xxx
export HCCL_INTRA_ROCE_ENABLE=1
export ASCEND_GLOBAL_LOG_LEVEL=3
```

P 侧：

```bash
export ROLE=prefill

vllm serve "$MODEL_PATH" \
  --served-model-name openpangu-505b \
  --host 0.0.0.0 \
  --port 8081 \
  --dtype bfloat16 \
  --trust-remote-code \
  --tensor-parallel-size "$P_TP_SIZE" \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --distributed-executor-backend ray \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 16 \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching \
  --additional-config '{"enable_low_latency":true}' \
  --kv-transfer-config '{"kv_connector":"LLMDataDistConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'
```

D 侧先用保守参数启动：

```bash
export ROLE=decode

vllm serve "$MODEL_PATH" \
  --served-model-name openpangu-505b \
  --host 0.0.0.0 \
  --port 8082 \
  --dtype bfloat16 \
  --trust-remote-code \
  --tensor-parallel-size "$D_TP_SIZE" \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --distributed-executor-backend ray \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.78 \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching \
  --additional-config '{"enable_low_latency":true}' \
  --kv-transfer-config '{"kv_connector":"LLMDataDistConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}' \
  --enforce-eager
```

等 D 侧不 OOM 后，再去掉 `--enforce-eager`，打开 graph，并逐步把 `max-num-seqs`、`gpu-memory-utilization` 调上去。

**D 节点一直 OOM的最可能原因**

第一优先检查：D 侧有没有真的加载到你改过的 A2 low_latency D 配置。`loader.py` 是靠 `ROLE=decode`、`PREFILL_POD_NUM=1`、`DECODE_POD_NUM=1` 和 `additional_config.enable_low_latency=true` 来选 `1P1D` 的 D 配置；不是靠 `kv_role=kv_consumer` 选。decode 日志里必须看到类似：

```text
load configuration file ... low_latency/openpangu_v2/...xp1d_d.json
ModelExtraConfig ... hardware_platform: A2 ... is_prefill_node: false
```

如果日志里出现 “configuration ... was not found” 或加载 default，就是配置没命中。

第二个常见原因是 D 侧 KV cache 把显存吃满了。仓库示例默认 `--gpu-memory-utilization 0.88`、`max_len=32768`、`bsz=16`，对 505B bf16 + 910B + D graph 很激进。D 侧还要留 ACL graph/capture、workspace、prefetch、LLMDataDist cache 注册的余量，所以建议从：

```bash
--gpu-memory-utilization 0.75~0.80
--max-num-seqs 4 或 8
--enforce-eager
```

开始确认模型和 KV 链路能起来，再逐步加。

第三个点是你们把 A3 配置改成 A2 时，不能只改 `best_practice_configs.json` 的 hardware。A3 的 low_latency D 配置默认仍会启用一些图/预取/通信策略，910B 上可能内存余量不够。若 OOM 发生在 graph/capture 或 profile 后，给 A2 D 配置单独加：

```json
"enable_prefetch": false,
"expert_gate_up_prefetch": 0,
"expert_down_prefetch": 0,
"attn_prefetch": 0,
"dense_mlp_prefetch": 0,
"lm_head_prefetch": 0
```

先把内存稳住，再按性能需求逐项打开。

我会重点看这三类日志：`ModelExtraConfig` 是否命中 A2/low_latency/D，`Available KV cache memory` 估算是多少，以及 OOM 是发生在 `profile_run`、`initialize_kv_cache` 还是 `capture_model`。这能很快区分是“配置没命中”、KV 过大，还是 graph/prefetch 额外内存导致。

参考源码：你给的 [low_latency P 配置](https://gitee.com/omniai/omni-npu/blob/release_1.2.1/src/omni_npu/model_config/configs/low_latency/openpangu_v2/pangu_v2_moe_bf16_a3_xp1d_p.json)、配套 D 配置、[配置加载器](https://gitee.com/omniai/omni-npu/blob/release_1.2.1/src/omni_npu/model_config/config_loader/loader.py)、[PD 启动示例](https://gitee.com/omniai/omni-npu/blob/release_1.2.1/examples/serve-pd-disaggregate.sh)。

> "enable_prefetch": false,
> "expert_gate_up_prefetch": 0,
> "expert_down_prefetch": 0,
> "attn_prefetch": 0,
> "dense_mlp_prefetch": 0,
> "lm_head_prefetch": 0 这些东西你是在哪个文件里看到的，要加在哪里

<details><summary>1 previous message</summary>

> 我去源码里把这几个字段的定义和生效位置精确定位一下，顺便确认它们应该落在哪个 JSON 层级里。
</details>

这些字段在配置加载器里定义的：

[loader.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/model_config/config_loader/loader.py:142>) 里 `ModelOperatorOptConfig` 定义了：

```python
enable_prefetch: bool = True
expert_gate_up_prefetch: int = 50
expert_down_prefetch: int = 28
dense_mlp_prefetch: int = 56
lm_head_prefetch: int = 135
attn_prefetch: int = 96
```

同一个文件 [loader.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/model_config/config_loader/loader.py:170>) 里也写了：只要 `enable_prefetch=false`，这些 prefetch size 会被强制置 0。

另外 [features.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/model_config/config_loader/features.py:20>) 里 `eager_mode` 也会把它们关掉，所以我才建议先用这组参数排查 D 侧 OOM。

要加的位置是你实际加载的 **D 节点 JSON** 里的 `operator_optimization_config` 下。比如你现在基于：

`src/omni_npu/model_config/configs/low_latency/openpangu_v2/pangu_v2_moe_bf16_a3_xp1d_d.json`

那就改成类似：

```json
{
  "model_parallel_config": {
    "enable_aicpu_dp_sync": false,
    "layer_parallel_config": {}
  },
  "operator_optimization_config": {
    "moe_comm_strategy": "dispatch_combine",
    "use_aicpu_fa_tiling": true,
    "use_noncontiguous_kv": true,
    "use_rope_fusion_op": true,
    "use_mhc_fusion_op": false,
    "use_mome_inplace_update": true,
    "split_q_up_in_multistream": false,
    "disable_npu_top_k_top_p_sample": true,

    "enable_prefetch": false,
    "expert_gate_up_prefetch": 0,
    "expert_down_prefetch": 0,
    "attn_prefetch": 0,
    "dense_mlp_prefetch": 0,
    "lm_head_prefetch": 0
  }
}
```

更推荐你不要直接复用 `a3` 文件名，而是新建一个 A2 D 配置，比如：

`src/omni_npu/model_config/configs/low_latency/openpangu_v2/pangu_v2_moe_bf16_a2_xp1d_d.json`

然后在 `low_latency/best_practice_configs.json` 里让 `hardware: "A2"` 的 `decode_config_file` 指到这个文件。这样日志里一眼能看出 D 节点到底加载的是 A2 还是 A3 配置。

> d测oom发生在加载完模型之后，我设置的是4台机器dp32，加载权重没有问题，加载后oom

<details><summary>1 previous message</summary>

> 这个信息很关键：如果“权重加载完之后才 OOM”，那问题大概率已经不是模型权重本身，而是加载后 vLLM/omni-npu 继续做的三件事之一：profile run、KV cache 分配、decode graph capture。我再对源码里的启动阶段顺序确认一下，给你一个更精确的判断路径。
</details>

明白了，那判断要改一下：**D 侧权重能加载完，OOM 发生在加载后**，优先怀疑不是 `enable_prefetch` 这组配置，而是 **加载后阶段的 KV cache / profile / graph capture**。

源码顺序在 [npu_worker.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/worker/npu_worker.py:151>)：

1. `load_model()`：加载权重，已成功。
2. `determine_available_memory()`：会跑一次 `model_runner.profile_run()`。
3. 按 `gpu_memory_utilization` 算可用 KV cache 显存。
4. `initialize_from_config()`：真正分配 KV cache，并注册给 `LLMDataDistConnector`。
5. `compile_or_warm_up_model()`：如果没开 `--enforce-eager`，会 `capture_model()` 做 graph capture。

所以你这个“加载后 OOM”通常分三种：

```text
权重加载完成
  -> profile_run OOM
  -> KV cache allocation OOM
  -> graph capture OOM
```

你先看 D 侧日志里 OOM 前最后一行：

- 如果还没打印 `Available KV cache memory`，就是 `profile_run` OOM。
- 如果打印了 `Available KV cache memory` 后 OOM，多半是 KV cache 分配太激进。
- 如果看到 `capture_model`、`cudagraph`、`ACL Graph` 相关日志后 OOM，就是 graph capture。

对于你说的 **4 台机器 dp32**，最常见的是第二种：每个 D rank 都会按自己的 `--gpu-memory-utilization` 尽量吃满显存给 KV cache。`dp32` 提高总吞吐，但不代表单 rank 的 KV 预算会自动变小；如果每个 rank 都按 0.88 去分配，910B 上很容易权重刚好能放下，后面 KV 或 graph 一来就炸。

建议先这样定位：

```bash
# D 侧先加
--enforce-eager
--gpu-memory-utilization 0.70
--max-num-seqs 4
```

如果这样能起来，说明不是权重问题。然后按顺序调：

1. 先把 `--gpu-memory-utilization` 从 `0.70 -> 0.75 -> 0.80` 慢慢加。
2. 再把 `--max-num-seqs` 从 `4 -> 8 -> 16` 加。
3. 最后再去掉 `--enforce-eager` 开 graph。

如果一去掉 `--enforce-eager` 就 OOM，那就是 D 侧 graph capture 的额外显存不够，这时要改 D 侧的 `--compilation-config`，不要像示例脚本那样按 `BSZ` 捕获一长串 gear。比如先只捕获小档位：

```bash
--compilation-config '{"level":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8],"backend":"eager","compile_sizes":[1,2,4]}'
```

`enable_prefetch=false` 那组可以加在 D 配置里，但它不是第一优先。它主要减少 forward/profile/graph 里的额外 buffer；真正控制 KV cache 吃多少显存的是 `--gpu-memory-utilization` 或显式 KV cache memory 参数。

所以我现在会先让你确认一件事：D 日志里 OOM 前，最后出现的是 `Available KV cache memory`、`initialize_kv_cache`，还是 `capture_model`？这三种对应的修法不一样。

> 不管设置多短的序列，max-num-seqs设置多小，gpu-memory-utilization如何调整，都会oom，是出自KV cache分配时，请帮我分析下哪里会有问题

<details><summary>3 previous messages</summary>

> 这个现象很像“KV cache 规格/分组算错”或“D 侧实际不是按你以为的配置在分配”，因为正常情况下 `max-num-seqs`、`max-model-len`、`gpu-memory-utilization` 至少应该显著改变 KV blocks 数。 我继续沿着 pangu_v2 的 KV cache spec、PD 注册和配置命中三条线往下挖。
>
> 我看到一个很可疑的方向：`pangu_v2_moe` 和 `openpangu_v2` 的 patch 路径不同，KV cache page size/混合注意力相关 patch 也不同。你说模型架构是 `pangu_v2_moe`，但配置文件来自 `low_latency/openpangu_v2`，如果自动 patch 没加载到正确目录，KV spec 很可能会按错误模型逻辑走。继续核对 `pangu_v2_moe` 的模型结构转换 patch。
>
> 我找到了一个更像根因的点：`pangu_v2_moe` 里有 MoME/Mamba-like cache，这类 cache 不是普通 attention KV，它的内存估算路径里有一段是按 `max_model_len` / block size 算“最大状态内存”，并不完全跟 `gpu-memory-utilization`、`max-num-seqs` 同步缩放。这个能解释“怎么调 util/seqs 都还在 KV 分配处炸”。我再把对应代码行扒出来。
</details>

这个现象基本可以排除普通 attention KV 太大。普通 KV 如果太大，`--gpu-memory-utilization`、`--max-model-len`、`--max-num-seqs` 至少会让 OOM 位置/申请量变化。你这里“不管怎么调都在 KV cache 分配 OOM”，我会优先怀疑两处。

**1. pangu_v2_moe 没有自动加载完整 KV 相关 patch**

`src/omni_npu/vllm_patches/__init__.py` 的模型 patch 映射里有：

```python
"openpangu_v2": "pangu_v2_base,pangu_sink_swa_mla",
"pangu_v2_hybrid": "pangu_v2_base,pangu_v2_hybrid",
```

但没有：

```python
"pangu_v2_moe": ...
```

所以如果你模型 `config.json` 里是：

```json
"model_type": "pangu_v2_moe"
```

自动加载时很可能只进 `patches/models/pangu_v2_moe`，不会加载 `pangu_v2_hybrid` 里的 KV cache spec / MOME manager patch。

而关键 KV patch 在这些文件：

```text
pangu_v2_hybrid/patch_kv_cache_interface.py
pangu_v2_hybrid/patch_single_type_kv_cache_manager.py
pangu_v2_hybrid/patch_models_config.py
pangu_v2_hybrid/patch_mome_hybrid.py
```

建议你 D/P 两侧都显式加：

```bash
export OMNI_NPU_PATCHES_DIR="pangu_v2_hybrid,pangu_v2_moe"
```

或者改源码映射：

```python
"pangu_v2_moe": "pangu_v2_hybrid,pangu_v2_moe",
```

启动日志里要确认看到这些 patch registered/applied：

```text
PanguNewKVCacheSpecsPatch
SingleTypeKVCacheManagerPatch
PanguV2HybridModelsConfigMapPatch
MambaStateShapeCalculatorMomePatch
```

如果没看到，KV 分配路径就不可信。

**2. MoME/Mamba-like cache 不完全受 gpu-memory-utilization 控制**

在 `pangu_v2_hybrid/patch_kv_cache_interface.py` 里，`MomeSpec.max_memory_usage_bytes()` 是这样算的：

```python
max_model_len = vllm_config.model_config.max_model_len
return cdiv(max_model_len, self.block_size) * self.page_size_bytes
```

也就是说这类 cache 主要看 **effective max_model_len** 和 `MomeSpec.page_size_bytes`，不是简单按 `gpu-memory-utilization` 缩放。`max-num-seqs` 调小也不一定救得了。

所以要确认你调小的是启动参数：

```bash
--max-model-len 2048
```

而不是只调请求里的输入长度。日志里最好打印实际生效的：

```text
model_config.max_model_len
cache_config.block_size
cache_config.mamba_block_size
cache_config.mamba_page_size_padded
```

**我建议的排查顺序**

先用最小启动验证：

```bash
export OMNI_NPU_PATCHES_DIR="pangu_v2_hybrid,pangu_v2_moe"
export NUM_DIE_PER_MACH=8   # 4机 dp32，910B 通常每机 8 卡，别用默认 16

vllm serve ... \
  --max-model-len 2048 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.5 \
  --enforce-eager
```

如果这样还在 KV cache 分配 OOM，那基本就是 KV spec/page size 算错或 patch 没生效，不是容量调参问题。

最有价值的日志是在 `NPUModelRunner.initialize_kv_cache()` 里，`super().initialize_kv_cache_tensors(...)` 前，把每个 group 打出来：

```python
for i, group in enumerate(kv_cache_config.kv_cache_groups):
    spec = group.kv_cache_spec
    logger.warning(
        "KV group %s type=%s layers=%s block_size=%s page=%s max_mem=%s",
        i, type(spec).__name__, len(group.layer_names),
        getattr(spec, "block_size", None),
        getattr(spec, "page_size_bytes", None),
        spec.max_memory_usage_bytes(self.vllm_config)
        if hasattr(spec, "max_memory_usage_bytes") else None,
    )
```

看哪个 group 的 `page_size_bytes/max_mem` 异常大。大概率会落在 `MomeSpec` 或 `SinkMLAAttentionSpec/DSAAttentionSpec` 其中一个。

> 我甚至把d节点机器数量增加为8台，仍然报oom

<details><summary>1 previous message</summary>

> 8 台 D 还 OOM 这个信息反而更坐实了一点：增加 D 节点/DP 副本通常不会降低单卡 KV cache 的分配压力，它只是多了更多同构 decode replica。单卡上那份模型、MoME 状态/KV 规格、每个 rank 预留的 cache 仍然基本一样。我再把这个和你们的 1P1D 形态对应起来说清楚。
</details>

8 台 D 仍然 OOM，基本说明：**加 D 机器没有降低单卡 KV cache 分配量**。

在 1P1D 里，D 的 `data parallel` 更像“增加 decode 副本数”。每个 D rank 仍然会在自己卡上加载一份它负责的模型分片，并分配一份本地 KV/MoME cache。你从 4 台 `dp32` 增到 8 台，如果变成 `dp64`，只是多了 32 个 decode rank，**单个 rank 的 KV cache 规格没变**，所以同一个分配点仍然会 OOM。

我现在更怀疑两类问题：

**1. 不是普通 KV，而是 MoME/Mamba cache 固定按 max_model_len 分配**

`pangu_v2_moe` 有 MoME/Mamba-like state。源码里 `MomeSpec.max_memory_usage_bytes()` 是：

```python
max_model_len = vllm_config.model_config.max_model_len
return cdiv(max_model_len, self.block_size) * self.page_size_bytes
```

这个路径主要看 `--max-model-len` 和 `MomeSpec.page_size_bytes`，不是看 D 机器总数。你说“序列多短都 OOM”，要确认你调的是启动参数：

```bash
--max-model-len 2048
```

而不是请求输入长度。请求短不影响启动时的 KV/MoME 预分配。

**2. pangu_v2_moe 的 KV patch 没完整加载，导致 cache spec/page size 算错**

`release_1.2.1` 的 patch 自动映射里没有 `pangu_v2_moe` 指到 `pangu_v2_hybrid`，但关键 KV/MoME patch 在 `pangu_v2_hybrid` 目录下。建议 P/D 两边显式设置：

```bash
export OMNI_NPU_PATCHES_DIR="pangu_v2_hybrid,pangu_v2_moe"
```

日志里确认有这些 patch：

```text
PanguNewKVCacheSpecsPatch
SingleTypeKVCacheManagerPatch
PanguV2HybridModelsConfigMapPatch
MambaStateShapeCalculatorMomePatch
```

没有这些，KV cache 分配很可能走错规格。

下一步我建议直接打日志定位是哪一组 KV 爆了。在 `src/omni_npu/worker/npu_model_runner.py` 的 `initialize_kv_cache()`，`initialize_kv_cache_tensors()` 前打印：

```python
for i, group in enumerate(kv_cache_config.kv_cache_groups):
    spec = group.kv_cache_spec
    logger.warning(
        "KV group=%s type=%s layers=%s block_size=%s page_size=%s max_mem=%s",
        i,
        type(spec).__name__,
        len(group.layer_names),
        getattr(spec, "block_size", None),
        getattr(spec, "page_size_bytes", None),
        spec.max_memory_usage_bytes(self.vllm_config)
        if hasattr(spec, "max_memory_usage_bytes") else None,
    )
```

如果看到某个 `MomeSpec` 的 `page_size` 或 `max_mem` 巨大，那就找到根了。
如果 `--max-model-len 2048 --max-num-seqs 1 --gpu-memory-utilization 0.5 --enforce-eager` 仍然 OOM，几乎可以确定是 KV spec/page size 错，不是集群容量问题。

> patch 自动映射文件在哪里

在这个文件：

[src/omni_npu/vllm_patches/__init__.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/__init__.py:110>)

关键函数是 `_get_patch_dir_names(model_type)`，里面有 `patch_dir_map`：

```python
patch_dir_map = {
    "deepseek_v3": "deepseek",
    "deepseek_v32": "deepseek",
    "qwen3": "qwen",
    "pangu_v2_hybrid": "pangu_v2_base,pangu_v2_hybrid",
    "openpangu_v2": "pangu_v2_base,pangu_sink_swa_mla",
    ...
}
```

你这个模型如果 `config.json` 里是：

```json
"model_type": "pangu_v2_moe"
```

这里默认没有对应映射。可以加一行：

```python
"pangu_v2_moe": "pangu_v2_hybrid,pangu_v2_moe",
```

或者不改代码，启动前显式指定：

```bash
export OMNI_NPU_PATCHES_DIR="pangu_v2_hybrid,pangu_v2_moe"
```

显式指定更适合先排查。

> pangu_v2_hybrid,pangu_v2_moe这两个patch路径在哪里，为什么要打这些patch

这两个 patch 目录在：

```text
src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid
src/omni_npu/vllm_patches/patches/models/pangu_v2_moe
```

对应你本地临时分析目录是：

[pangu_v2_hybrid](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid:1>)

[pangu_v2_moe](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/patches/models/pangu_v2_moe:1>)

为什么要打：

`pangu_v2_moe` 目录主要补的是这个模型类型本身，例如：

```text
patch_model_arch_config_convertor.py
patch_mamba_utils.py
patch_kv_cache_dtype.py
patch_process_weights_after_loading.py
```

它让 vLLM 正确认出 `pangu_v2_moe` 是 MLA/MoME 这类架构，并补 Mamba/MoME state shape、dtype、权重后处理、KV dtype 等。

`pangu_v2_hybrid` 目录更关键，它补的是 **KV cache/MoME/hybrid attention 的运行时机制**，例如：

```text
patch_kv_cache_interface.py
patch_single_type_kv_cache_manager.py
patch_kv_cache_utils.py
patch_models_config.py
patch_static_sink_attention.py
patch_mome_hybrid.py
```

这些 patch 会新增/修正：

```text
MomeSpec
DSAAttentionSpec
ShareKVSlidingWindowSpec
SinkMLAAttentionSpec
MomeManager
hybrid KV cache grouping
mamba_page_size_padded
StaticSinkAttention 的 get_kv_cache_spec
```

你的 OOM 发生在 D 侧 **KV cache 分配时**，而这些正是决定“每个 KV group 分配多大、按什么 page size、哪个 manager 管”的代码。如果不打 `pangu_v2_hybrid`，`pangu_v2_moe` 的 MoME/MLA/SWA/DSA KV cache 很可能按 vLLM 默认逻辑或不完整逻辑分配，导致 page size/group 数异常，机器加到 8 台也没用，因为错的是单 rank 的 KV 分配规格。

所以排查时建议先显式指定：

```bash
export OMNI_NPU_PATCHES_DIR="pangu_v2_hybrid,pangu_v2_moe"
export VLLM_PLUGINS="omni-npu,omni_npu_patches,omni_custom_models"
```

然后看日志里有没有这些 patch registered/applied。

> openpangu_v2base补丁需要吗

一般来说：**`pangu_v2_moe` 不优先需要 `pangu_v2_base`**，先用：

```bash
export OMNI_NPU_PATCHES_DIR="pangu_v2_hybrid,pangu_v2_moe"
```

`pangu_v2_base` 目录主要是 Pangu V2 家族的通用 MLA wrapper patch，仓库说明里写的是面向：

```text
openpangu_v2 -> pangu_v2_base,pangu_sink_swa_mla
openpangu_ultra_omni -> pangu_v2_base,pangu_sink_swa_mla,openpangu_ultra_omni
```

它的核心文件是：

```text
src/omni_npu/vllm_patches/patches/models/pangu_v2_base/patch_mla.py
```

主要 patch `vllm.model_executor.layers.mla.StaticSinkMultiHeadLatentAttentionWrapper`。

但你说的模型 `model_type` 是：

```text
pangu_v2_moe
```

它更关键的是：

```text
pangu_v2_moe
pangu_v2_hybrid
```

尤其是 `pangu_v2_hybrid` 里的 KV/MoME cache patch。你的 OOM 在 KV cache 分配，所以优先验证这些 patch 是否生效。

如果你想保守一点，也可以试：

```bash
export OMNI_NPU_PATCHES_DIR="pangu_v2_base,pangu_v2_hybrid,pangu_v2_moe"
```

但我建议先不要混太多，避免引入 openpangu_v2 路径的 MLA wrapper 行为干扰判断。先跑 `pangu_v2_hybrid,pangu_v2_moe`，看 KV group 日志和 OOM 是否变化；如果模型初始化阶段报缺 `StaticSinkMultiHeadLatentAttentionWrapper` 或 MLA wrapper 相关错误，再加 `pangu_v2_base`。

> OMNI_NPU_PATCHES_DIR 这个环境变量是哪里设置的

`OMNI_NPU_PATCHES_DIR` 不是仓库里某个固定文件设置的，它是**启动服务前由你在 shell / 启动脚本 / 容器环境里设置的环境变量**。

读取它的位置在：

[src/omni_npu/vllm_patches/__init__.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/__init__.py:166>)

逻辑大概是：

```python
env_var_name = "OMNI_NPU_PATCHES_DIR"
current_env_value = os.getenv(env_var_name)
```

如果你设置了：

```bash
export OMNI_NPU_PATCHES_DIR="pangu_v2_hybrid,pangu_v2_moe"
```

它会按这个值去 `src/omni_npu/vllm_patches/patches/models/` 下面精确找目录并加载。

如果你没设置，它会读模型目录的 `config.json` 里的 `model_type`，再通过 `_get_patch_dir_names()` 自动映射。

设置方式示例：

```bash
export VLLM_PLUGINS="omni-npu,omni_npu_patches,omni_custom_models"
export OMNI_NPU_PATCHES_DIR="pangu_v2_hybrid,pangu_v2_moe"

vllm serve ...
```

如果你是用 `examples/serve-pd-disaggregate.sh`，就把这两行加在脚本启动 `vllm serve` 之前，或者在调用脚本前先 export。

> 推理代码中是哪里开始进行打patch的

<details><summary>1 previous message</summary>

> 我把入口链路按代码调用顺序给你捋一下：vLLM 启动时不是业务代码里手动 import 某个 patch 文件，而是通过 `VLLM_PLUGINS` 触发插件 entry point，再由 omni-npu 的 `apply_patches()` 扫描并应用 patch。
</details>

是在 **vLLM 启动加载插件阶段**开始打 patch 的，不是在模型 forward 里。

入口链路是：

1. `pyproject.toml` 注册 vLLM 插件入口：

[pyproject.toml](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/pyproject.toml:36>)

```toml
[project.entry-points."vllm.general_plugins"]
omni_npu_patches = "omni_npu.vllm_patches:apply_patches"
omni_custom_models = "omni_npu.v1.models:register_models"
```

2. 启动时如果设置了：

```bash
export VLLM_PLUGINS="omni-npu,omni_npu_patches,omni_custom_models"
```

vLLM 会加载 `omni_npu_patches`，也就是调用：

[__init__.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/__init__.py:219>)

```python
def apply_patches():
    auto_import_patches()
    manager.apply_patches()
```

3. `auto_import_patches()` 会先加载 common patch，再根据 `OMNI_NPU_PATCHES_DIR` 或模型 `config.json` 的 `model_type` 加载模型 patch 目录：

[__init__.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/__init__.py:168>)

```python
def auto_import_patches():
    ...
    env_var_name = "OMNI_NPU_PATCHES_DIR"
    ...
```

4. patch 文件被 import 后，里面的装饰器会注册 patch：

[core.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/core.py:64>)

```python
def register_patch(name: str, target):
    ...
```

5. 最后 `PatchManager.apply_patches()` 真正把目标类/模块的方法替换掉：

[patch_manager.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/patch_manager.py:66>)

```python
def apply_patches(self):
    ...
```

所以简化调用链是：

```text
vllm serve
  -> vLLM plugin system reads VLLM_PLUGINS
  -> loads omni_npu_patches entry point
  -> omni_npu.vllm_patches.apply_patches()
  -> auto_import_patches()
  -> import patch_*.py
  -> @register_patch(...)
  -> PatchManager.apply_patches()
```

如果没设置 `VLLM_PLUGINS` 包含 `omni_npu_patches`，这些 patch 不会开始执行。启动日志里应该能看到类似：

```text
patch class xxx registered as xxx
patch applied: xxx => target.method
```

> 有pangu_v2_moe这个模型的注册文件吗，读到config是pangu_v2_moe推理会用到哪些文件

<details><summary>1 previous message</summary>

> 我去仓库里按 `pangu_v2_moe`、`PanguV2MoEForCausalLM`、模型注册入口三条线查一下，确认它是由 omni-npu 注册，还是走 vLLM/模型自带 remote code。
</details>

有，但要分清楚两种“注册”：

**1. 模型类注册：没有看到 pangu_v2_moe 的显式 ModelRegistry 注册**

在 omni-npu 里模型类注册入口是：

[src/omni_npu/v1/models/__init__.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/v1/models/__init__.py:4>)

里面有很多：

```python
ModelRegistry.register_model(...)
```

但我没看到：

```python
PanguV2MoEForCausalLM
```

所以 `pangu_v2_moe` 本身不是在这个文件里注册成 omni 自定义模型类的。真正模型类通常是通过模型 `config.json` 的 `architectures` 字段让 vLLM 解析，比如：

```json
"model_type": "pangu_v2_moe",
"architectures": ["PanguV2MoEForCausalLM"]
```

如果用了 `--trust-remote-code`，也可能来自模型目录自己的 remote code。

**2. patch/配置注册：pangu_v2_moe 有**

当 `config.json` 读到：

```json
"model_type": "pangu_v2_moe"
```

会用到这些地方：

第一，patch 自动加载入口：

[src/omni_npu/vllm_patches/__init__.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/__init__.py:51>)

它会读模型目录的 `config.json`：

```python
model_type = config.get("model_type")
```

如果没有设置 `OMNI_NPU_PATCHES_DIR`，它会自动 fuzzy 匹配目录。`pangu_v2_moe` 会匹配到：

```text
src/omni_npu/vllm_patches/patches/models/pangu_v2_moe
```

也就是这些文件：

```text
patch_kv_cache_dtype.py
patch_mamba_utils.py
patch_model_arch_config_convertor.py
patch_process_weights_after_loading.py
```

但注意：**默认不会自动加载 `pangu_v2_hybrid`**。

第二，模型最佳配置匹配：

[src/omni_npu/model_config/configs/match_hf_configs.json](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/model_config/configs/match_hf_configs.json:257>)

这里有：

```json
"pangu_v2_moe_505B": {
  "model_type": "pangu_v2_moe",
  ...
}
```

然后配置加载器：

[src/omni_npu/model_config/config_loader/loader.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/model_config/config_loader/loader.py:237>)

会把 HF config 匹配成：

```text
model_name = pangu_v2_moe_505B
quant_type = bf16
hardware = A2/A3
ROLE = prefill/decode
```

再去 `best_practice_configs.json` 里找配置，比如 low latency 下：

[src/omni_npu/model_config/configs/low_latency/best_practice_configs.json](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/model_config/configs/low_latency/best_practice_configs.json:65>)

会选到：

```text
src/omni_npu/model_config/configs/low_latency/openpangu_v2/pangu_v2_moe_bf16_a3_xp1d_p.json
src/omni_npu/model_config/configs/low_latency/openpangu_v2/pangu_v2_moe_bf16_a3_xp1d_d.json
```

**所以读到 `model_type=pangu_v2_moe` 后主要会用到：**

```text
src/omni_npu/vllm_patches/__init__.py
src/omni_npu/vllm_patches/patches/common/*
src/omni_npu/vllm_patches/patches/models/pangu_v2_moe/*
src/omni_npu/model_config/config_loader/loader.py
src/omni_npu/model_config/configs/match_hf_configs.json
src/omni_npu/model_config/configs/low_latency/best_practice_configs.json
src/omni_npu/model_config/configs/low_latency/openpangu_v2/pangu_v2_moe_bf16_a3_xp1d_d.json
```

如果你显式加：

```bash
export OMNI_NPU_PATCHES_DIR="pangu_v2_hybrid,pangu_v2_moe"
```

才会额外用到：

```text
src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/*
```

我建议你们排查 D 侧 OOM 时加这个，因为关键 KV/MoME cache 相关 patch 大多在 `pangu_v2_hybrid`。

> src/omni_npu/vllm_patches/patches/models/pangu_v2_moe/ 这个本身就会加载吗，不用添加也会？

对，**如果你没有手动设置 `OMNI_NPU_PATCHES_DIR`，并且模型 `config.json` 里的 `model_type` 是 `pangu_v2_moe`，那 `src/omni_npu/vllm_patches/patches/models/pangu_v2_moe/` 本身会自动加载**。

原因是自动匹配逻辑在：

[src/omni_npu/vllm_patches/__init__.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/__init__.py:149>)

没有映射命中时，它会 fuzzy 匹配：

```python
if (model_type_lower.startswith(subdir_name_lower)
        or subdir_name_lower in model_type_lower):
    patch_dirs.append(subdir)
```

`model_type_lower = "pangu_v2_moe"`，目录名也叫 `pangu_v2_moe`，所以会匹配并加载。

但是有个重要细节：**一旦你设置了 `OMNI_NPU_PATCHES_DIR`，就会走精确匹配模式，不再自动根据 `model_type` 加载。**

所以：

```bash
# 不设置 OMNI_NPU_PATCHES_DIR
# 会自动加载 pangu_v2_moe
```

但如果你设置：

```bash
export OMNI_NPU_PATCHES_DIR="pangu_v2_hybrid"
```

那只会加载 `pangu_v2_hybrid`，不会自动再加载 `pangu_v2_moe`。

因此建议写全：

```bash
export OMNI_NPU_PATCHES_DIR="pangu_v2_hybrid,pangu_v2_moe"
```

这样两个目录都会加载。

> 你可以分析一下pangu_v2_hybrid这里面的补丁做了些什么吗？

<details><summary>1 previous message</summary>

> 我把 `pangu_v2_hybrid` 目录下每个 patch 文件逐个看一下，按“改了 vLLM 哪个对象、解决什么问题、和 KV OOM 是否相关”整理给你。
</details>

可以。`pangu_v2_hybrid` 这组 patch 不是单纯“某个模型类注册”，它主要是在补 vLLM 对 **Pangu V2 混合注意力结构** 的支持：MLA / DSA / SWA / MoME / static sink attention / hybrid KV cache。

目录是：

[pangu_v2_hybrid](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid:1>)

**最关键的 KV 相关 patch**

[patch_kv_cache_interface.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/patch_kv_cache_interface.py:1>)

它新增几种 vLLM 原生没有的 KV cache spec：

```text
DSAAttentionSpec
ShareKVSlidingWindowSpec
MomeSpec
SinkMLAAttentionSpec
```

作用是告诉 vLLM：这些层的 KV cache 每个 block/page 多大、怎么 merge、是不是 full/sliding/mamba-like cache。

和 OOM 最相关的是 `MomeSpec`：

```python
def max_memory_usage_bytes(self, vllm_config):
    max_model_len = vllm_config.model_config.max_model_len
    return cdiv(max_model_len, self.block_size) * self.page_size_bytes
```

所以 MoME cache 这块不是普通 attention KV，分配逻辑很特殊。

[patch_single_type_kv_cache_manager.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/patch_single_type_kv_cache_manager.py:1>)

它把上面的新 KV spec 接进 vLLM 的 KV manager：

```text
DSAAttentionSpec -> FullAttentionManager
ShareKVSlidingWindowSpec -> SlidingWindowManager
MomeSpec -> MomeManager
SinkMLAAttentionSpec -> SinkFullAttentionManager
```

如果没有这个，vLLM 即使知道有 `MomeSpec`，也不知道该用哪个 manager 去管理它。

[patch_kv_cache_utils.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/patch_kv_cache_utils.py:1>)

它 patch vLLM 的 hybrid KV group 划分逻辑，并增加：

```bash
HYBRID_ATTN_GROUP_SIZE
```

这个环境变量可以强制控制 hybrid attention 的 group size。这个也可能影响 KV 分组数量和 padding 浪费。

**MoME / Mamba-like 相关**

[patch_mome_hybrid.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/patch_mome_hybrid.py:1>)

它新增 `MomeAttention`，核心是让 MoME 层提供：

```text
get_kv_cache_spec() -> MomeSpec
get_state_shape()
get_state_dtype()
```

也就是 MoME 的 q_cache、kv_cache、o_cache 三类状态怎么被 vLLM 分配。

[patch_mome.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/patch_mome.py:1>)

它补 `AggregateConv`，用于 MoME 里的卷积状态更新/权重加载。

**模型配置 / page size 对齐**

[patch_models_config.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/patch_models_config.py:1>)

这个很重要。它把一些 Pangu 架构加入 vLLM 的 `MODELS_CONFIG_MAP`，并计算：

```python
cache_config.mamba_page_size_padded
```

也就是 hybrid 模型里 MLA / DSA / MoME 这些不同 cache 类型对齐后的 page size。page size 算错，就非常容易在 KV cache 分配时 OOM。

里面包含：

```python
"PanguV2MoEForCausalLM": PanguV2MoEForCausalLMConfig
```

所以如果你的 `architectures` 是 `PanguV2MoEForCausalLM`，这个 patch 是相关的。

**Static sink / MLA / DSA KV spec**

[patch_static_sink_attention.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/patch_static_sink_attention.py:1>)

它 patch `StaticSinkMLAAttention`，最关键是 `get_kv_cache_spec()` 会根据层类型返回：

```text
ShareKVSlidingWindowSpec
DSAAttentionSpec
SinkMLAAttentionSpec
```

也就是说这里决定 static sink / sparse DSA / sliding MLA 层到底分配哪种 KV cache。

**运行时 metadata / 绑定修正**

[patch_num_prompt_tokens.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/patch_num_prompt_tokens.py:1>)

patch `GPUModelRunner._build_attention_metadata()`，给 MoME metadata builder 额外传：

```python
num_prompt_tokens
```

这个更偏运行时正确性，不是主要 OOM 点。

[patch_worker_utils.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/patch_worker_utils.py:1>)

patch `bind_kv_cache()`，允许一个 decoder block 里有多个 attention-like 层。Pangu V2 hybrid 这种模型可能一层里有 attention + mome/sink 等多个 cache 绑定对象，原生检查可能不适配。

**其它**

[patch_modelconfig.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/patch_modelconfig.py:1>)

主要让 `openpangu_v2` / `openpangu_mtp` 被识别成 MLA 模型。这个对 `pangu_v2_moe` 本身帮助有限，因为 `pangu_v2_moe` 的 MLA 识别在 `pangu_v2_moe/patch_model_arch_config_convertor.py` 里。

[patch_speculative.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/patch_speculative.py:1>)

主要是 MTP/spec decode 映射。如果你没开 speculative/MTP，它不是当前 OOM 的重点。

**一句话总结**

`pangu_v2_hybrid` 里真正和你 D 侧 KV cache OOM 相关的是：

```text
patch_kv_cache_interface.py
patch_single_type_kv_cache_manager.py
patch_kv_cache_utils.py
patch_models_config.py
patch_static_sink_attention.py
patch_mome_hybrid.py
```

它们共同决定：

```text
有哪些 KV cache 类型
每类 page_size 多大
怎么 hybrid grouping
用哪个 KV manager
MoME cache 怎么按 max_model_len 分配
static sink / DSA / SWA 返回什么 KV spec
```

所以你这个问题如果卡在 KV cache 分配，必须确认这些 patch 是否加载和生效。

> 我能否使用vllm-ascend0.18.0正式版本进行该模型的适配，我的本地没有vllm-ascend以及vllm，你先把仓下载下来，然后分析下如何适配

<details><summary>19 previous messages</summary>

> Stream error
>
> Reconnecting... 2/5
> timeout waiting for child process to exit
>
> Stream error
>
> Reconnecting... 3/5
> timeout waiting for child process to exit
>
> Stream error
>
> Reconnecting... 4/5
> timeout waiting for child process to exit
>
> Stream error
>
> Reconnecting... 5/5
> timeout waiting for child process to exit
>
> 我先把 `vllm-ascend` 的正式仓和对应 tag 拉到本地临时目录，再看它和 vLLM 的版本绑定、模型注册、Ascend attention/KV cache 路径。重点会对照你现在的 `pangu_v2_moe`、PD 分离和 KV 分配问题。
>
> <details><summary>Ran 1 command</summary>
>
> - Ran `$base = Join-Path $env:TEMP 'vllm-ascend-analysis'; New-Item -ItemType Directory -Force -Path $base | Out-Null; Set-Location $base; if (!(Test-Path 'vllm-ascend\.git')) { git clone https://github.com/vllm-project/vllm-ascend.git vllm-ascend } else { Set-Location vllm-ascend; git fetch --tags --prune; Set-Location .. }; Set-Location vllm-ascend; git tag --list | Select-String -Pattern '0\.18|v0\.18|0\.18\.0|v0\.18\.0'; git branch -a | Select-String -Pattern '0\.18|v0\.18|release'`
>
> </details>
>
> 第一次 clone 超时了，可能仓比较大或者网络慢。我检查一下目录是否已经部分拉下来了，能继续 fetch 就继续，不能就改用浅克隆 tag。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Get-ChildItem -Force $env:TEMP\vllm-ascend-analysis; if (Test-Path $env:TEMP\vllm-ascend-analysis\vllm-ascend\.git) { git -C $env:TEMP\vllm-ascend-analysis\vllm-ascend status --short; git -C $env:TEMP\vllm-ascend-analysis\vllm-ascend tag --list | Select-String -Pattern '0\.18|v0\.18|0\.18\.0|v0\.18\.0' }`
> - Ran `git ls-remote --tags https://github.com/vllm-project/vllm-ascend.git | Select-String -Pattern '0\.18|v0\.18|0\.18\.0|v0\.18\.0'`
> - Ran `$base = Join-Path $env:TEMP 'vllm-ascend-analysis'; $dst = Join-Path $base 'vllm-ascend-v0.18.0'; if (!(Test-Path (Join-Path $dst '.git'))) { git clone --depth 1 --branch v0.18.0 https://github.com/vllm-project/vllm-ascend.git $dst } else { git -C $dst fetch --tags --prune; git -C $dst checkout v0.18.0 }; git -C $dst rev-parse --short HEAD; git -C $dst status --short`
>
> </details>
>
> 浅克隆也超时了，但 tag 能查到，说明网络至少能连。为了不被 git 慢下载卡住，我改用 GitHub 的 tag 压缩包下载；这通常比 clone 快很多。
>
> <details><summary>Ran 1 command</summary>
>
> - Ran `$base = Join-Path $env:TEMP 'vllm-ascend-analysis'; New-Item -ItemType Directory -Force -Path $base | Out-Null; $zip = Join-Path $base 'vllm-ascend-v0.18.0.zip'; $out = Join-Path $base 'vllm-ascend-0.18.0-src'; if (!(Test-Path $out)) { if (!(Test-Path $zip)) { Invoke-WebRequest -Uri 'https://github.com/vllm-project/vllm-ascend/archive/refs/tags/v0.18.0.zip' -OutFile $zip -TimeoutSec 120 }; Expand-Archive -Path $zip -DestinationPath $base -Force; $expanded = Join-Path $base 'vllm-ascend-0.18.0'; if (Test-Path $expanded) { Rename-Item -Path $expanded -NewName 'vllm-ascend-0.18.0-src' -Force } }; Get-ChildItem -Force $base; Get-ChildItem -Force $out | Select-Object -First 20`
>
> </details>
>
> `vllm-ascend v0.18.0` 源码已经下载到临时目录。我现在看它声明依赖的 vLLM 版本、插件入口、模型支持和 KV cache/PD 代码路径。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `rg --files | rg "(model|registry|pangu|openpangu|kv|cache|attention|worker|platform|connector|pd|disagg|expert|moe|README|install|quickstart|config)"`
> - Ran `rg -n "vllm==|vllm |vllm-ascend|entry-points|platform_plugins|general_plugins|ModelRegistry|register_model|Pangu|pangu|openpangu|kv_transfer|KV|cache|PD|prefill|decode|disagg|data_parallel|Ascend" -S pyproject.toml README.md docs examples vllm_ascend tests`
> - Ran `Get-Content -Path pyproject.toml; Get-Content -Path requirements.txt -ErrorAction SilentlyContinue; Get-Content -Path requirements-dev.txt -ErrorAction SilentlyContinue`
>
> </details>
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> 我已经把 `vllm-ascend` 的 0.18.0 源码包拉到临时目录了；接下来我会把同版本 `vllm` 也取下来，因为 0.18.0 的适配边界主要卡在 vLLM 核心模型/KV cache API 上，而不是 Ascend 插件单边能决定。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `$base='C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis'; Get-ChildItem -Path $base -Force | Select-Object FullName,Length,LastWriteTime`
> - Ran `rg -n "Pangu|pangu|Mome|MoME|StaticSink|SinkMLA|DSAAttention|ShareKV|PanguV2|model_type" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src"`
> - Ran `rg -n "vllm.general_plugins|entry-points|register_model|MODELS_CONFIG_MAP|ModelRegistry|kv_cache|Mamba|MLA|mooncake|KVTransfer|kv_connector" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src"`
>
> </details>
>
> 从第一轮扫出来的结果看，0.18.0 正式版里确实出现了 `PanguUltraMoE` 的发布记录和量化映射，但还没看到 `pangu_v2_moe` 或 `PanguV2MoEForCausalLM` 的直接模型入口。这一点很关键：它说明可以借官方 0.18.0 底座，但大概率不是“换包即跑”。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `$ErrorActionPreference='Stop'
> $base='C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis'
> $zip=Join-Path $base 'vllm-v0.18.0.zip'
> $out=Join-Path $base 'vllm-0.18.0-src'
> if (!(Test-Path $out)) {
>   if (!(Test-Path $zip)) {
>     Invoke-WebRequest -Uri 'https://github.com/vllm-project/vllm/archive/refs/tags/v0.18.0.zip' -OutFile $zip
>   }
>   $extract=Join-Path $base 'vllm-v0.18.0'
>   if (Test-Path $extract) { Remove-Item -LiteralPath $extract -Recurse -Force }
>   Expand-Archive -LiteralPath $zip -DestinationPath $extract -Force
>   $inner=Get-ChildItem -LiteralPath $extract | Select-Object -First 1
>   Move-Item -LiteralPath $inner.FullName -Destination $out
> }
> Get-ChildItem -Path $base -Force | Select-Object FullName,Length,LastWriteTime`
> - Ran `rg -n "Pangu|pangu|PanguV2|PanguUltra|pangu_ultra|pangu_v2_moe|Mome|MoME|StaticSink|SinkMLA|DSAAttention|ShareKV|glm_moe_dsa" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\tests"`
> - Ran `rg -n "Pangu|pangu|PanguV2|PanguUltra|pangu_ultra|pangu_v2_moe|Mome|MoME|StaticSink|SinkMLA|DSAAttention|ShareKV|glm_moe_dsa" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\vllm_ascend" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\docs"`
> - Ran `rg -n "vllm==|torch==|torch-npu|CANN|Python|Requires-Python|transformers|triton-ascend" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\pyproject.toml" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\requirements.txt" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\README.md" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\setup.py"`
>
> </details>
>
> 同版本 `vllm` 也已经下载好了。这里有个好消息：vLLM 0.18.0 本体已经包含 `openpangu.py`、`PanguUltraMoEForCausalLM`、`PanguProMoEV2ForCausalLM`、`StaticSinkAttention`，所以适配不是从零写模型；坏消息是它没有 `PanguV2MoEForCausalLM/pangu_v2_moe` 这个名字，仍要处理架构名、KV 规格和 Ascend 后端路径。
>
> <details><summary>Ran 26 commands</summary>
>
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\model_executor\models\openpangu.py" | Select-Object -Skip 250 -First 230`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\model_executor\models\openpangu.py" | Select-Object -Skip 1300 -First 100`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\model_executor\models\openpangu.py" -TotalCount 220`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\model_executor\models\openpangu.py" | Select-Object -Skip 540 -First 170`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\model_executor\models\openpangu.py" | Select-Object -Skip 800 -First 170`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\vllm_ascend\quantization\modelslim_config.py" | Select-Object -Skip 70 -First 65`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\model_executor\models\registry.py" | Select-Object -Skip 160 -First 35`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\transformers_utils\model_arch_config_convertor.py" | Select-Object -Skip 190 -First 80`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\model_executor\layers\attention\static_sink_attention.py" | Select-Object -Skip 100 -First 110`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\vllm_ascend\patch\platform\patch_kv_cache_interface.py" -TotalCount 260`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\v1\kv_cache_interface.py" -TotalCount 260`
> - Ran `rg -n "class .*Spec|page_size_bytes|max_memory_usage_bytes|StaticSink|DSA|mamba_page_size|KVCacheSpec" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\v1" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\vllm_ascend\patch"`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\model_executor\layers\attention\static_sink_attention.py" | Select-Object -Skip 185 -First 80`
> - Ran `rg -n "class Pangu|PanguV2|pangu_v2_moe|Mome|StaticSink|DSAAttention|ShareKVSliding|SinkMLA|ModelRegistry|architectures" "C:\Users\李杰\AppData\Local\Temp\omni-npu-release_1.2.1\src\omni_npu\vllm_patches\patches\models\pangu_v2_moe" "C:\Users\李杰\AppData\Local\Temp\omni-npu-release_1.2.1\src\omni_npu\vllm_patches\patches\models\pangu_v2_hybrid"`
> - Ran `Get-ChildItem -Path "C:\Users\李杰\AppData\Local\Temp\omni-npu-release_1.2.1\src\omni_npu\vllm_patches\patches\models\pangu_v2_hybrid" -Force | Select-Object Name,Length`
> - Ran `Get-ChildItem -Path "C:\Users\李杰\AppData\Local\Temp\omni-npu-release_1.2.1\src\omni_npu\vllm_patches\patches\models\pangu_v2_moe" -Force | Select-Object Name,Length`
> - Ran `rg -n "pangu_v2_moe|pangu_v2_hybrid|PanguV2MoE|PanguUltra|openpangu" "C:\Users\李杰\AppData\Local\Temp\omni-npu-release_1.2.1\src\omni_npu\vllm_patches\patches\models" "C:\Users\李杰\AppData\Local\Temp\omni-npu-release_1.2.1\src\omni_npu\v1\models" "C:\Users\李杰\AppData\Local\Temp\omni-npu-release_1.2.1\src\omni_npu\model_config"`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\vllm_ascend\patch\__init__.py" -TotalCount 260`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\vllm_ascend\__init__.py" -TotalCount 120`
> - Ran `rg -n "get_attn_backend|MLA|StaticSink|Sink|FlashAttentionDiffKV|attn_backend|model_type.*pangu|pangu_ultra" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\vllm_ascend\attention" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\vllm_ascend\platform.py" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\vllm_ascend\ops"`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\setup.py" | Select-Object -Skip 528 -First 20`
> - Ran `rg -n "def get_kv_cache_spec|MLAAttentionSpec|MambaSpec|get_attn_backend|MultiHeadLatentAttentionWrapper|class MLAAttention" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\model_executor\layers\mla.py" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\model_executor\layers\attention\mla_attention.py"`
> - Ran `rg -n "MambaStateDtypeCalculator|MambaStateShapeCalculator|mamba_cache|MambaSpec|mamba_page_size" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\vllm_ascend"`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\model_executor\layers\mla.py" | Select-Object -Skip 640 -First 170`
> - Ran `rg -n "Pangu-Pro-MoE|Pangu Pro|openPangu|PanguUltra|PanguUltraMoE|vllm serve|--model|tensor-parallel|expert-parallel|data-parallel" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\docs\source\tutorials" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\examples"`
> - Ran `rg -n "disaggregated|Mooncake|kv_connector|kv_role|producer|consumer|prefill|decode|--kv-transfer-config" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\docs\source\tutorials" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\examples\disaggregated_prefill_v1"`
>
> </details>
</details>

可以用 `vllm-ascend==0.18.0` 正式版做底座，但不能指望直接把 `omni-npu release_1.2.1` 那套配置平移过去就能跑。核心判断是：vLLM 0.18.0 本体已经有 OpenPangu/Pangu Ultra MoE 支持，但没有你这个 `pangu_v2_moe` / `PanguV2MoEForCausalLM` 的完整入口。

我已经下载到本地：

- `vllm-ascend 0.18.0`: [vllm-ascend-0.18.0-src](</C:/Users/李杰/AppData/Local/Temp/vllm-ascend-analysis/vllm-ascend-0.18.0-src>)
- `vllm 0.18.0`: [vllm-0.18.0-src](</C:/Users/李杰/AppData/Local/Temp/vllm-ascend-analysis/vllm-0.18.0-src>)

**结论**
推荐路线是“以 vLLM/vllm-ascend 0.18.0 为底座，迁移少量 PanguV2MoE 专用 patch”，而不是继续在 omni-npu 的旧 vLLM patch 体系里硬撑。原因是 0.18.0 已经原生包含：

- OpenPangu 模型主体：[openpangu.py](</C:/Users/李杰/AppData/Local/Temp/vllm-ascend-analysis/vllm-0.18.0-src/vllm/model_executor/models/openpangu.py>)
- `PanguUltraMoEForCausalLM` / `PanguProMoEV2ForCausalLM` 注册：[registry.py](</C:/Users/李杰/AppData/Local/Temp/vllm-ascend-analysis/vllm-0.18.0-src/vllm/model_executor/models/registry.py:181>)
- `StaticSinkAttention` 和 `SinkFullAttentionSpec`
- vllm-ascend 的 Ascend MLA 后端：[mla_v1.py](</C:/Users/李杰/AppData/Local/Temp/vllm-ascend-analysis/vllm-ascend-0.18.0-src/vllm_ascend/attention/mla_v1.py>)
- Mooncake PD 分离，不再走 omni 的 `LLMDataDistConnector`

但缺的东西也很明确：

1. `PanguV2MoEForCausalLM` 没注册
   vLLM 0.18.0 只注册了 `PanguEmbeddedForCausalLM`、`PanguProMoEV2ForCausalLM`、`PanguUltraMoEForCausalLM`。所以如果你的模型 `config.json` 里是：
   ```json
   "architectures": ["PanguV2MoEForCausalLM"],
   "model_type": "pangu_v2_moe"
   ```
   需要加 alias 或注册一个薄类，把它映射到现有 OpenPangu MoE 实现。

2. `pangu_v2_moe` 不在 MLA 识别白名单里
   这里要补：[model_arch_config_convertor.py](</C:/Users/李杰/AppData/Local/Temp/vllm-ascend-analysis/vllm-0.18.0-src/vllm/transformers_utils/model_arch_config_convertor.py:216>)。现在有 `pangu_ultra_moe`，没有 `pangu_v2_moe`。否则模型可能不会按 MLA 路径走。

3. 如果 505B 里真的用了 MoME/Mamba-like 层，必须迁移 omni 的 MoME patch
   这是最关键的 OOM 风险点。omni 里的 `pangu_v2_hybrid` patch 自己定义了 `MomeSpec`、`MomeAttention`、`MomeManager`，而 vLLM 0.18.0 只有通用 `MambaSpec`，没有 omni 那个三段式 `q_cache/kv_cache/o_cache` 的 `MomeSpec`。需要重点迁移这些文件里的逻辑：
   - [patch_kv_cache_interface.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/patch_kv_cache_interface.py>)
   - [patch_mome_hybrid.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/patch_mome_hybrid.py>)
   - [patch_single_type_kv_cache_manager.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/patch_single_type_kv_cache_manager.py>)
   - [patch_num_prompt_tokens.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/patch_num_prompt_tokens.py>)

4. 量化/权重后处理也要补
   如果你的权重是 bf16，量化映射可以先放后面；如果有 w8a8/hif8/mxfp8，则至少要把 `pangu_v2_moe` 加到 vllm-ascend 的 `modelslim_config.py`，参考已有 `pangu_ultra_moe`：
   [modelslim_config.py](</C:/Users/李杰/AppData/Local/Temp/vllm-ascend-analysis/vllm-ascend-0.18.0-src/vllm_ascend/quantization/modelslim_config.py:88>)

**我建议的适配顺序**

1. 先做最小模型入口适配：
   加 `PanguV2MoEForCausalLM -> openpangu.OpenPanguMoEModel` 的 alias，或者在 `openpangu.py` 里加：
   ```python
   class PanguV2MoEForCausalLM(OpenPanguMoEModel):
       pass
   ```
   并在 registry 加：
   ```python
   "PanguV2MoEForCausalLM": ("openpangu", "PanguV2MoEForCausalLM")
   ```

2. 把 `pangu_v2_moe` 加进 MLA 判断。
   否则 505B 这种 MLA/MoE 架构很容易走错 KV cache spec。

3. 单机/单实例先跑到“权重加载 + KV cache 初始化成功”。
   先不要 PD，不要图模式，不要 MLAPO：
   ```bash
   export VLLM_ASCEND_ENABLE_MLAPO=0
   export VLLM_ASCEND_ENABLE_FUSED_MC2=0
   vllm serve /path/to/openpangu-505B \
     --trust-remote-code \
     --tensor-parallel-size 8 \
     --enable-expert-parallel \
     --max-model-len 2048 \
     --max-num-seqs 1 \
     --gpu-memory-utilization 0.75
   ```

4. 如果这一步仍然在 KV cache 分配 OOM，优先查 `MomeSpec/MambaSpec` 的 page size，而不是调 `max-num-seqs`。
   你前面说“序列多短、max-num-seqs 多小、gpu-memory-utilization 怎么调都 OOM”，这非常像固定 cache spec 算大了，尤其是 MoME/Mamba 类 cache，不是普通 attention KV。

5. PD 分离最后再接。
   vllm-ascend 0.18.0 正式版应走 Mooncake：
   ```json
   {
     "kv_connector": "MooncakeConnectorV1",
     "kv_role": "kv_producer 或 kv_consumer",
     "kv_connector_module_path": "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector"
   }
   ```
   参考官方示例：[mooncake_connector_deployment_guide.md](</C:/Users/李杰/AppData/Local/Temp/vllm-ascend-analysis/vllm-ascend-0.18.0-src/examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md>)

**对你 D 节点 OOM 的判断**
换到 vllm-ascend 0.18.0 不一定自动解决 D OOM。它会带来更现代的 KV/MLA/PD 框架，但如果 `pangu_v2_moe` 的 MoME cache spec 没迁移对，D 节点仍会在“模型加载后、KV cache 初始化时”炸，而且加 D 机器数也没用，因为每个 D rank 还是要分配自己的本地 cache。

所以第一优先级不是启动脚本，而是确认 0.18.0 下 `pangu_v2_moe` 每层实际生成的 `KVCacheSpec` 是什么：普通 `MLAAttentionSpec`、`SinkFullAttentionSpec`，还是缺失的 `MomeSpec` 被错误替代了。这个地方对了，服务启动才有意义。

> 我先跑通4机混布即可，我在该目录下放了一个config.json，你可以看下，如何适配

<details><summary>8 previous messages</summary>

> 我先看你放的 `config.json`，重点确认 `model_type`、`architectures`、MLA/MoE/MoME 相关字段，以及它和 vLLM 0.18.0 现有 OpenPangu 实现能不能直接对上。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Get-ChildItem -Path "C:\Users\李杰\Documents\omni-infer" -Force | Select-Object Name,Length,LastWriteTime`
> - Ran `rg -n "class OpenPanguModel|def load_weights|packed_modules_mapping|first_k_dense_replace|qk_nope|moe|mome|mamba|num_hidden_layers|architectures|model_type" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\model_executor\models\openpangu.py"`
> - Ran `Get-Content -Path "C:\Users\李杰\Documents\omni-infer\config.json" -Raw`
>
> </details>
>
> 这个 `config.json` 比我预想的更“混合”：它不是普通 OpenPangu MoE。它同时有 MLA、DSA 层、SWA 层、static sink、`use_mome: true`、MHC 字段，而且架构名是 `PanguV2aMoEForCausalLM`。这意味着 0.18.0 现有 `openpangu.py` 不能只靠 alias 跑通，会把很多层类型忽略掉。
>
> <details><summary>Ran 13 commands</summary>
>
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\omni-npu-release_1.2.1\src\omni_npu\v1\models\pangu\pangu_ultra_moe.py" | Select-Object -Skip 160 -First 180`
> - Ran `rg -n "use_mome|use_mhc|dsa_layers|swa_layers|sliding_window_list|PanguV2a|PanguV2MoE|MomeAttention|StaticSinkMLA|block_post_layernorm|router_sliding_window|param_sink_number" "C:\Users\李杰\AppData\Local\Temp\omni-npu-release_1.2.1\src\omni_npu"`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\model_executor\models\openpangu.py" | Select-Object -Skip 1020 -First 170`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\model_executor\models\openpangu.py" | Select-Object -Skip 1240 -First 90`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\omni-npu-release_1.2.1\src\omni_npu\v1\models\pangu\pangu_ultra_moe.py" | Select-Object -Skip 340 -First 135`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\omni-npu-release_1.2.1\src\omni_npu\v1\models\pangu\pangu_ultra_moe.py" | Select-Object -Skip 800 -First 110`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\omni-npu-release_1.2.1\src\omni_npu\v1\layers\attention\npu_dsa.py" | Select-Object -Skip 388 -First 95`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\omni-npu-release_1.2.1\src\omni_npu\v1\layers\attention\npu_mla.py" | Select-Object -Skip 120 -First 190`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\omni-npu-release_1.2.1\src\omni_npu\v1\models\pangu\pangu_ultra_moe.py" -TotalCount 140`
> - Ran `rg -n "pad_token_id|rope_parameters|_normalize_rope_parameters|class OpenPanguModelBase|packed_modules_mapping|PanguUltraMoEForCausalLM|PanguV2a" "C:\Users\李杰\AppData\Local\Temp\omni-npu-release_1.2.1\src\omni_npu\v1\models\pangu\pangu_ultra_moe.py"`
> - Ran `rg -n "PanguUltraMoE|PanguV2a|PanguV2MoE|architectures|model_type|openpangu" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\docs\source\tutorials" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\tests"`
> - Ran `rg -n "register_model\(|ModelRegistry|_VLLM_MODELS|_TEXT_GENERATION_MODELS|models.registry" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\model_executor\models\registry.py" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm"`
> - Ran `rg -n "--hf-overrides|hf_overrides|architectures" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\engine\arg_utils.py" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\entrypoints"`
>
> </details>
>
> 架构名写错了，是PanguV2MoEForCausalLM
>
> <details><summary>Ran 1 command</summary>
>
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\model_executor\models\registry.py" | Select-Object -Skip 880 -First 45`
>
> </details>
>
> 收到，架构名按 `PanguV2MoEForCausalLM` 来看。这个修正很关键：它正好对应 omni 里 `pangu_v2_hybrid` patch 提到的名字，但 vLLM 0.18.0 官方本体仍没有注册这个 arch，所以适配点没有消失，只是从“架构名不确定”变成了“确认要补这个 arch”。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `rg -n "def set_default_rope_theta|set_default_rope_theta" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\transformers_utils\config.py" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm"`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\model_executor\models\openpangu.py" | Select-Object -Skip 1140 -First 90`
> - Ran `rg -n "def get_attn_backend_cls|is_deepseek_mla|get_attn_backend_cls" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\vllm_ascend\platform.py" "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\v1\attention\selector.py"`
> - Ran `Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-0.18.0-src\vllm\transformers_utils\config.py" | Select-Object -Skip 350 -First 30
> Get-Content -Path "C:\Users\李杰\AppData\Local\Temp\vllm-ascend-analysis\vllm-ascend-0.18.0-src\vllm_ascend\platform.py" | Select-Object -Skip 575 -First 42`
>
> </details>
</details>

对，按 `PanguV2MoEForCausalLM` 看，这个 config 说明：**不能只给 vLLM 0.18.0 加一个架构 alias 就算适配完成**。

你的模型实际特征是：

- `model_type: pangu_v2_moe`
- MLA：`q_lora_rank / kv_lora_rank / qk_nope_head_dim / qk_rope_head_dim`
- DSA：`dsa_layers + index_topk/index_head_dim`
- SWA：`swa_layers + sliding_window_list`
- static sink：`param_sink_number: 128`
- MoME：`use_mome: true`
- MHC：`use_mhc: true`
- MTP 权重：`num_nextn_predict_layers: 3`，并且 `swa_layers` 里有 `50,51,52`

所以官方 vLLM 0.18.0 自带的 `PanguUltraMoEForCausalLM` 只能覆盖一部分：普通 OpenPangu MoE + MLA。它不会正确处理你这个 config 里的 DSA/SWA/MoME/MHC。

**最小适配建议**
先跑通 4 机混布，我建议不要上 PD，不要 Mooncake，先走单个 vLLM 服务，4 机 Ray，`TP=8, PP=4, DP=1`：

```bash
export VLLM_ASCEND_ENABLE_MLAPO=0
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export VLLM_USE_V1=1

vllm serve /path/to/openpangu-505B \
  --trust-remote-code \
  --distributed-executor-backend ray \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 4 \
  --enable-expert-parallel \
  --block-size 128 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.75 \
  --enforce-eager
```

这个是启动形态。真正适配代码要补这些点：

1. 注册 `PanguV2MoEForCausalLM`
   vLLM 0.18.0 的 [registry.py](</C:/Users/李杰/AppData/Local/Temp/vllm-ascend-analysis/vllm-0.18.0-src/vllm/model_executor/models/registry.py:181>) 没有这个 arch。要加：
   ```python
   "PanguV2MoEForCausalLM": ("openpangu", "PanguV2MoEForCausalLM")
   ```

2. 在 [openpangu.py](</C:/Users/李杰/AppData/Local/Temp/vllm-ascend-analysis/vllm-0.18.0-src/vllm/model_executor/models/openpangu.py>) 里加类：
   ```python
   class PanguV2MoEForCausalLM(OpenPanguMoEModel):
       pass
   ```

3. 把 `pangu_v2_moe` 加进 MLA 识别
   文件是 [model_arch_config_convertor.py](</C:/Users/李杰/AppData/Local/Temp/vllm-ascend-analysis/vllm-0.18.0-src/vllm/transformers_utils/model_arch_config_convertor.py:216>)，现在只有 `pangu_ultra_moe`，没有 `pangu_v2_moe`。

4. 必须迁移 omni 的 PanguV2 专用逻辑
   尤其是这些：
   - `DSA/SWA/SinkMLA`：omni 的 `StaticSinkMLAAttention`
   - `MoME`：`MomeSpec / MomeAttention / MomeManager`
   - `MHC`：`NPUmHCRL` 和 `block_post_layernorm`

   关键来源文件：
   - [pangu_ultra_moe.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/v1/models/pangu/pangu_ultra_moe.py>)
   - [npu_mla.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/v1/layers/attention/npu_mla.py>)
   - [npu_dsa.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/v1/layers/attention/npu_dsa.py>)
   - [patch_kv_cache_interface.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/patch_kv_cache_interface.py>)
   - [patch_mome_hybrid.py](</C:/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/patch_mome_hybrid.py>)

另外你的 `config.json` 缺一个对 vLLM 很重要的字段：`rope_parameters`。否则 vLLM 0.18 的 OpenPangu MLA 会把 `rope_theta` 默认成 `10000`，不是你配置里的 `6400000`。建议补：

```json
"rope_parameters": {
  "rope_type": "default",
  "rope_theta": 6400000
}
```

我的判断：**4 机混布要正确跑通，最稳路线是把 omni 的 `PanguUltraMoEForCausalLM` 这套实现移植到 vLLM 0.18/vllm-ascend 0.18，而不是只用官方 OpenPangu 类。** 只 alias 到 `PanguUltraMoEForCausalLM` 可以做启动烟测，但 DSA/MoME/MHC 都会被忽略，结果不可信，KV cache 形态也不对。

> 你来帮我实现一下，可以在wsl下D:\workspace\里创一个目录来完成，先请你规划一下你完成的步骤

<details><summary>2 previous messages</summary>

> 可以，我先只规划，不动文件。这个适配我会按“先能启动并正确建 KV cache，再谈性能优化”的顺序做，避免一上来把 omni-npu 全量搬过去，后面很难定位问题。
>
> <details><summary>Plan</summary>
>
> - [ ] 确认 WSL 与 D 盘工作目录路径，创建独立适配目录
> - [ ] 下载/准备 vLLM 0.18.0 与 vllm-ascend 0.18.0 源码，并记录基线差异
> - [ ] 搭建 pangu_v2_moe 适配插件或补丁包，优先减少对上游源码的侵入
> - [ ] 补齐 PanguV2MoEForCausalLM 注册、pangu_v2_moe MLA 识别和 config 规范化
> - [ ] 迁移 PanguV2 专用 attention/KV 逻辑：DSA、SWA、SinkMLA、MoME
> - [ ] 处理 MHC、block_post_layernorm、权重加载与 MTP 层跳过逻辑
> - [ ] 准备 4 机混布启动脚本和最小验证命令
> - [ ] 加入 KV cache spec 打印/诊断，验证 OOM 是否来自普通 KV、Sink、DSA 还是 MoME
>
> </details>
</details>

可以。我会按下面这条路线做，目标先定为：**4 机混布，单服务跑通模型加载、KV cache 初始化、首 token 推理**，先不做 1P1D 分离和性能极限优化。

**实施规划**

1. 在 WSL 下创建独立目录
   目标路径类似：
   `/mnt/d/workspace/pangu-v2-moe-vllm-ascend018`

2. 拉取源码
   准备两份固定版本源码：
   - `vllm` tag `v0.18.0`
   - `vllm-ascend` tag `v0.18.0`

3. 采用“小插件/补丁包”方式适配
   我会优先做一个独立包，比如 `vllm_pangu_v2_moe`，通过 vLLM plugin 机制加载。这样不直接大改 vLLM/vllm-ascend 主仓，后面升级、回滚、diff 都清楚。

4. 第一阶段补最小入口
   - 注册 `PanguV2MoEForCausalLM`
   - 把 `model_type: pangu_v2_moe` 加入 MLA 判断
   - 规范化你的 `config.json`，补 `rope_parameters.rope_theta=6400000`
   - 兼容 `architectures: ["PanguV2MoEForCausalLM"]`

5. 第二阶段迁移核心模型逻辑
   从 omni-npu 里迁移必要逻辑，不全搬：
   - DSA 层：`dsa_layers/index_topk/index_head_dim`
   - SWA 层：`swa_layers/sliding_window_list`
   - Sink MLA：`param_sink_number`
   - MoME：`use_mome/router_sliding_window/MomeSpec/MomeAttention/MomeManager`
   - MHC：`use_mhc/block_post_layernorm_idx`

6. 第三阶段处理权重加载
   确认这些名字能正确映射：
   - `q_a_proj + kv_a_proj_with_mqa -> fused_qkv_a_proj`
   - `gate_proj + up_proj -> gate_up_proj`
   - MoE expert 权重
   - MTP 层 `50/51/52` 先跳过或单独处理，保证主模型先跑通

7. 准备 4 机混布启动脚本
   初始建议：
   - `TP=8`
   - `PP=4`
   - `DP=1`
   - `--enable-expert-parallel`
   - `--enforce-eager`
   - 先关 `MLAPO/FUSED_MC2`
   - `max-model-len` 先小，例如 `4096`

8. 加 KV cache 诊断日志
   在初始化 KV cache 前打印：
   - 每个 group 的 spec 类型
   - layer 数量
   - block size
   - page size
   - max memory usage
   - 是否 MoME/Sink/DSA/SWA

这样后面如果仍 OOM，我们能直接知道是普通 MLA KV、Sink KV、DSA sparse cache，还是 MoME cache 算大了。

我下一步会先在 WSL 的 `/mnt/d/workspace` 下创建适配目录、拉源码并搭项目骨架。

> 现在可以开始动手完成了，要注意运行环境是npu，写代码的过程请认真一点

<details><summary>17 previous messages</summary>

> 开始动手。我会先确认 WSL 可用性和 `/mnt/d/workspace` 映射，然后在独立目录里落地源码和适配包；这一步会尽量保持上游源码干净，把 PanguV2MoE 逻辑放在单独插件里。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `wsl -e bash -lc "set -e; uname -a; pwd; ls -ld /mnt/d /mnt/d/workspace 2>/dev/null || true"`
> - Ran `Get-ChildItem -Path D:\ -Force | Where-Object { $_.Name -eq 'workspace' } | Select-Object FullName,LastWriteTime`
> - Ran `Get-Content -Path "C:\Users\李杰\Documents\omni-infer\config.json" -Raw`
>
> </details>
>
> WSL 和 `/mnt/d/workspace` 都可用；本地 `config.json` 里架构名仍是旧的 `PanguV2aMoEForCausalLM`，我会在适配目录生成一份修正后的模板，不直接改你原始文件。接下来创建工程目录并把 vLLM/vllm-ascend 0.18.0 源码放进去。
>
> <details><summary>Ran 5 commands</summary>
>
> - Ran `wsl -e bash -lc "set -euo pipefail
> ROOT=/mnt/d/workspace/pangu-v2-moe-vllm-ascend018
> mkdir -p \"$ROOT\"/{src,third_party,configs,scripts,docs}
> printf '%s\n' \"$ROOT\""`
> - Ran `wsl -e bash -lc 'set -euo pipefail; ROOT=/mnt/d/workspace/pangu-v2-moe-vllm-ascend018; mkdir -p "$ROOT"/{src,third_party,configs,scripts,docs}; printf "%s\n" "$ROOT"'`
> - Ran `wsl -e bash -lc 'set -euo pipefail; ROOT=/mnt/d/workspace/pangu-v2-moe-vllm-ascend018; if [ ! -d "$ROOT/third_party/vllm-ascend-0.18.0" ]; then cp -a /mnt/c/Users/李杰/AppData/Local/Temp/vllm-ascend-analysis/vllm-ascend-0.18.0-src "$ROOT/third_party/vllm-ascend-0.18.0"; fi; find "$ROOT/third_party/vllm-ascend-0.18.0" -maxdepth 1 -type f -name "pyproject.toml" -o -name "setup.py"'`
> - Ran `wsl -e bash -lc 'set -euo pipefail; ROOT=/mnt/d/workspace/pangu-v2-moe-vllm-ascend018; if [ ! -d "$ROOT/third_party/vllm-0.18.0" ]; then cp -a /mnt/c/Users/李杰/AppData/Local/Temp/vllm-ascend-analysis/vllm-0.18.0-src "$ROOT/third_party/vllm-0.18.0"; fi; find "$ROOT/third_party/vllm-0.18.0" -maxdepth 1 -type f -name "pyproject.toml" -o -name "setup.py"'`
> - Ran `wsl -e bash -lc 'set -euo pipefail; ROOT=/mnt/d/workspace/pangu-v2-moe-vllm-ascend018; cp /mnt/c/Users/李杰/Documents/omni-infer/config.json "$ROOT/configs/config.original.json"; ls -la "$ROOT" "$ROOT/configs"'`
>
> </details>
>
> `vllm-ascend` 已复制完成，`vllm` 体积更大，第一次复制超时了；我先检查是否已经完整落下，缺的话继续补齐。之后我会开始写插件骨架和可安装脚本。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `wsl -e bash -lc 'ROOT=/mnt/d/workspace/pangu-v2-moe-vllm-ascend018; ls -la "$ROOT/third_party"; test -f "$ROOT/third_party/vllm-0.18.0/pyproject.toml" && echo VLLM_OK || echo VLLM_MISSING; du -sh "$ROOT/third_party"/* 2>/dev/null || true'`
> - Ran `wsl -e bash -lc 'du -sh /mnt/c/Users/李杰/AppData/Local/Temp/vllm-ascend-analysis/vllm-0.18.0-src /mnt/c/Users/李杰/AppData/Local/Temp/vllm-ascend-analysis/vllm-ascend-0.18.0-src'`
>
> </details>
>
> 源码完整落下了。现在进入实现阶段：我会先做一个可安装的 `vllm_pangu_v2_moe` 插件，插件启动时负责注册 arch、修 config、patch KV 诊断；模型主体我会先以 vLLM 0.18 的 OpenPangu MoE 为基线扩展，不直接改上游文件。
>
> <details><summary>Ran 7 commands</summary>
>
> - Ran `wsl -e bash -lc 'rg -n "class MLAAttention|def __init__|use_sparse|indexer|sink_len|get_kv_cache_spec|sliding_window" /mnt/d/workspace/pangu-v2-moe-vllm-ascend018/third_party/vllm-0.18.0/vllm/model_executor/layers/attention/mla_attention.py | head -80'`
> - Ran `wsl -e bash -lc 'sed -n "34,140p" /mnt/d/workspace/pangu-v2-moe-vllm-ascend018/third_party/vllm-0.18.0/vllm/model_executor/layers/mla.py'`
> - Ran `wsl -e bash -lc 'sed -n "277,380p" /mnt/d/workspace/pangu-v2-moe-vllm-ascend018/third_party/vllm-0.18.0/vllm/model_executor/layers/attention/mla_attention.py; sed -n "830,865p" /mnt/d/workspace/pangu-v2-moe-vllm-ascend018/third_party/vllm-0.18.0/vllm/model_executor/layers/attention/mla_attention.py'`
> - Ran `wsl -e bash -lc 'rg -n "class Indexer|index_topk|index_head_dim|use_sparse|AscendSFA|SFA" /mnt/d/workspace/pangu-v2-moe-vllm-ascend018/third_party/vllm-0.18.0/vllm /mnt/d/workspace/pangu-v2-moe-vllm-ascend018/third_party/vllm-ascend-0.18.0/vllm_ascend | head -120'`
> - Ran `wsl -e bash -lc 'command -v rg || true; /usr/bin/grep -RInE "class Indexer|index_topk|index_head_dim|use_sparse|AscendSFA|SFA" /mnt/d/workspace/pangu-v2-moe-vllm-ascend018/third_party/vllm-0.18.0/vllm /mnt/d/workspace/pangu-v2-moe-vllm-ascend018/third_party/vllm-ascend-0.18.0/vllm_ascend | head -120'`
> - Ran `Get-ChildItem -Path "D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-0.18.0\vllm","D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-ascend-0.18.0\vllm_ascend" -Recurse -File | Select-String -Pattern "class Indexer|index_topk|index_head_dim|use_sparse|AscendSFA|SFA" | Select-Object -First 120 Path,LineNumber,Line`
> - Ran `Get-ChildItem -Path "C:\Users\李杰\AppData\Local\Temp\omni-npu-release_1.2.1\src\omni_npu" -Recurse -File | Select-String -Pattern "class Indexer|index_topk|index_head_dim" | Select-Object -First 60 Path,LineNumber,Line`
>
> </details>
>
> 我这里看到一个重要取舍：真正完整支持你这个 config 的代码其实已经在 omni 的 `PanguUltraMoEForCausalLM` 里，而 vLLM 0.18 官方 OpenPangu 类缺 DSA/MoME/MHC。为了先跑通 4 机混布，我会做“双路径”插件：优先桥接 omni 的完整模型实现；如果运行环境没有 omni-npu，再退回到 vLLM 官方 OpenPangu 基线并明确告警。这样不会把风险藏起来。
>
> <details><summary>Created 6 files, edited 1 file, ran 10 commands</summary>
>
> - Ran `wsl -e bash -lc 'find /mnt/d/workspace/pangu-v2-moe-vllm-ascend018 -maxdepth 3 -type d | sort'`
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/README.md` (+3 -0)
> - Ran `wsl -e bash -lc 'sed -n "1,120p" /mnt/d/workspace/pangu-v2-moe-vllm-ascend018/third_party/vllm-0.18.0/vllm/transformers_utils/model_arch_config_convertor.py; sed -n "400,455p" /mnt/d/workspace/pangu-v2-moe-vllm-ascend018/third_party/vllm-0.18.0/vllm/transformers_utils/model_arch_config_convertor.py'`
> - Ran `wsl -e bash -lc 'grep -n "def is_deepseek_mla" -A35 /mnt/d/workspace/pangu-v2-moe-vllm-ascend018/third_party/vllm-0.18.0/vllm/transformers_utils/model_arch_config_convertor.py'`
> - Ran `wsl -e bash -lc 'grep -n "is_deepseek_mla" /mnt/d/workspace/pangu-v2-moe-vllm-ascend018/third_party/vllm-0.18.0/vllm/transformers_utils/model_arch_config_convertor.py /mnt/d/workspace/pangu-v2-moe-vllm-ascend018/third_party/vllm-0.18.0/vllm/config/model.py | head -40'`
> - Ran `wsl -e bash -lc 'sed -n "208,238p" /mnt/d/workspace/pangu-v2-moe-vllm-ascend018/third_party/vllm-0.18.0/vllm/transformers_utils/model_arch_config_convertor.py'`
> - Ran `wsl -e bash -lc 'grep -RIn "def initialize_kv_cache" /mnt/d/workspace/pangu-v2-moe-vllm-ascend018/third_party/vllm-0.18.0/vllm/v1/worker /mnt/d/workspace/pangu-v2-moe-vllm-ascend018/third_party/vllm-ascend-0.18.0/vllm_ascend/worker | head -40'`
> - Ran `Get-ChildItem -Path "D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-0.18.0\vllm\v1\worker","D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-ascend-0.18.0\vllm_ascend\worker" -Recurse -File | Select-String -Pattern "def initialize_kv_cache|initialize_kv_cache" | Select-Object -First 50 Path,LineNumber,Line`
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/pyproject.toml` (+17 -0)
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/src/vllm_pangu_v2_moe/__init__.py` (+4 -0)
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/src/vllm_pangu_v2_moe/config.py` (+68 -0)
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/src/vllm_pangu_v2_moe/model.py` (+34 -0)
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/src/vllm_pangu_v2_moe/plugin.py` (+165 -0)
> - Ran `Select-String -Path "D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-0.18.0\vllm\config\model.py" -Pattern "def __post_init__" | Select-Object -First 10`
> - Ran `Get-Content -Path "D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-0.18.0\vllm\config\model.py" | Select-Object -Skip 410 -First 18`
> - Ran `Get-Content -Path "D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-0.18.0\vllm\config\model.py" | Select-Object -Skip 430 -First 45`
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/src/vllm_pangu_v2_moe/plugin.py` (+3 -3)
>
> </details>
>
> 插件骨架已经落下，包含 arch 注册、MLA 识别 patch、量化映射 patch、KV cache 诊断钩子，以及“优先 omni 完整模型、退回 vLLM 基线模型”的选择逻辑。接下来我会补配置规范化脚本和 4 机混布启动脚本。
>
> <details><summary>Created 4 files, ran 2 commands</summary>
>
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/scripts/check_plugin.py` (+15 -0)
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/scripts/install_editable.sh` (+11 -0)
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/scripts/normalize_config.py` (+63 -0)
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/scripts/run_4node_mixed_node.sh` (+77 -0)
> - Ran `wsl -e bash -lc 'set -euo pipefail; ROOT=/mnt/d/workspace/pangu-v2-moe-vllm-ascend018; chmod +x "$ROOT"/scripts/*.sh "$ROOT"/scripts/*.py; python3 "$ROOT/scripts/normalize_config.py" "$ROOT/configs/config.original.json" "$ROOT/configs/config.pangu_v2_moe.normalized.json"; python3 -m json.tool "$ROOT/configs/config.pangu_v2_moe.normalized.json" >/dev/null; head -30 "$ROOT/configs/config.pangu_v2_moe.normalized.json"'`
> - Ran `wsl -e bash -lc 'set -euo pipefail; ROOT=/mnt/d/workspace/pangu-v2-moe-vllm-ascend018; find "$ROOT/src/vllm_pangu_v2_moe" -type f -maxdepth 1 -print -exec python3 -m py_compile {} \;'`
>
> </details>
>
> 规范化配置已生成，`py_compile` 通过。现在我会用本地源码的 `PYTHONPATH` 做一次轻量导入检查，不依赖 NPU 运行，只确认插件注册逻辑不会在 Python 层直接炸掉。
>
> <details><summary>Edited 1 file, ran 4 commands</summary>
>
> - Ran `wsl -e bash -lc 'set -euo pipefail; ROOT=/mnt/d/workspace/pangu-v2-moe-vllm-ascend018; PYTHONPATH="$ROOT/src:$ROOT/third_party/vllm-0.18.0:$ROOT/third_party/vllm-ascend-0.18.0" python3 "$ROOT/scripts/check_plugin.py"'`
> - Ran `Select-String -Path "D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-0.18.0\vllm\plugins\*.py","D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-0.18.0\vllm\*.py" -Pattern "VLLM_PLUGINS|general_plugins|platform_plugins" | Select-Object -First 80 Path,LineNumber,Line`
> - Ran `Get-Content -Path "D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-0.18.0\vllm\plugins\__init__.py" -TotalCount 100`
> - Ran `Get-Content -Path "D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-0.18.0\vllm\envs.py" | Select-Object -Skip 856 -First 12`
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/scripts/run_4node_mixed_node.sh` (+1 -2)
>
> </details>
>
> 我发现启动脚本里 `VLLM_PLUGINS` 需要同时放入 vllm-ascend 的 general plugins，否则限制插件列表后会把 `ascend_kv_connector/ascend_model_loader` 漏掉；已经修正。现在我会把 omni-npu release_1.2.1 也作为 bridge 源码放进 `third_party`，这样完整 PanguV2MoE 模型路径可以在同一工作目录内启用。
>
> <details><summary>Created 3 files, edited 2 files, ran 7 commands</summary>
>
> - Ran `wsl -e bash -lc 'set -euo pipefail; ROOT=/mnt/d/workspace/pangu-v2-moe-vllm-ascend018; if [ ! -d "$ROOT/third_party/omni-npu-release_1.2.1" ]; then cp -a /mnt/c/Users/李杰/AppData/Local/Temp/omni-npu-release_1.2.1 "$ROOT/third_party/omni-npu-release_1.2.1"; fi; test -f "$ROOT/third_party/omni-npu-release_1.2.1/src/omni_npu/v1/models/pangu/pangu_ultra_moe.py"; du -sh "$ROOT/third_party/omni-npu-release_1.2.1"'`
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/scripts/run_4node_mixed_node.sh` (+4 -0)
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/docs/ADAPTATION.md` (+86 -0)
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/scripts/prepare_model_config.sh` (+24 -0)
> - Ran `wsl -e bash -lc 'set -euo pipefail; ROOT=/mnt/d/workspace/pangu-v2-moe-vllm-ascend018; chmod +x "$ROOT"/scripts/*.sh "$ROOT"/scripts/*.py; bash -n "$ROOT/scripts/install_editable.sh"; bash -n "$ROOT/scripts/run_4node_mixed_node.sh"; bash -n "$ROOT/scripts/prepare_model_config.sh"; python3 -m py_compile "$ROOT/scripts/normalize_config.py" "$ROOT/scripts/check_plugin.py"; find "$ROOT" -maxdepth 2 -type f | sort'`
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/scripts/install_plugin_only.sh` (+9 -0)
> - Ran `wsl -e bash -lc 'set -euo pipefail; ROOT=/mnt/d/workspace/pangu-v2-moe-vllm-ascend018; chmod +x "$ROOT"/scripts/*.sh "$ROOT"/scripts/*.py; bash -n "$ROOT/scripts/install_plugin_only.sh"; grep -n "PanguV2MoEForCausalLM\|rope_parameters\|pad_token_id\|layer_types" "$ROOT/configs/config.pangu_v2_moe.normalized.json" | head -20'`
> - Ran `wsl -e bash -lc 'ROOT=/mnt/d/workspace/pangu-v2-moe-vllm-ascend018; ls -l "$ROOT/configs/config.pangu_v2_moe.normalized.json"; grep -n "PanguV2MoEForCausalLM" "$ROOT/configs/config.pangu_v2_moe.normalized.json"; grep -n "rope_parameters" "$ROOT/configs/config.pangu_v2_moe.normalized.json"; grep -n "pad_token_id" "$ROOT/configs/config.pangu_v2_moe.normalized.json"; grep -n "layer_types" "$ROOT/configs/config.pangu_v2_moe.normalized.json" | head -1'`
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/docs/ADAPTATION.md` (+1 -2)
> - Ran `Get-ChildItem -Path "D:\workspace\pangu-v2-moe-vllm-ascend018" -Recurse -File | Where-Object { $_.FullName -notmatch '\\third_party\\' } | Select-Object FullName,Length | Format-Table -AutoSize`
> - Ran `Remove-Item -LiteralPath "D:\workspace\pangu-v2-moe-vllm-ascend018\scripts\__pycache__" -Recurse -Force -ErrorAction SilentlyContinue; Remove-Item -LiteralPath "D:\workspace\pangu-v2-moe-vllm-ascend018\src\vllm_pangu_v2_moe\__pycache__" -Recurse -Force -ErrorAction SilentlyContinue; Get-ChildItem -Path "D:\workspace\pangu-v2-moe-vllm-ascend018" -Recurse -Directory -Filter __pycache__`
> - Ran `Get-ChildItem -Path "D:\workspace\pangu-v2-moe-vllm-ascend018" -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force; Get-ChildItem -Path "D:\workspace\pangu-v2-moe-vllm-ascend018" -Recurse -Directory -Filter __pycache__`
>
> </details>
>
> <details><summary>Plan</summary>
>
> 为了先跑通 4 机混布，我把完整 DSA/SWA/SinkMLA/MoME/MHC 路径做成 omni-npu bridge，而不是把 omni-npu 的 NPU kernel 栈全量复制到插件里；fallback 路径只用于导入/基线烟测。
>
> - [x] 确认 WSL 与 D 盘工作目录路径，创建独立适配目录
> - [x] 下载/准备 vLLM 0.18.0 与 vllm-ascend 0.18.0 源码，并记录基线差异
> - [x] 搭建 pangu_v2_moe 适配插件或补丁包，优先减少对上游源码的侵入
> - [x] 补齐 PanguV2MoEForCausalLM 注册、pangu_v2_moe MLA 识别和 config 规范化
> - [x] 迁移 PanguV2 专用 attention/KV 逻辑：DSA、SWA、SinkMLA、MoME
> - [x] 处理 MHC、block_post_layernorm、权重加载与 MTP 层跳过逻辑
> - [x] 准备 4 机混布启动脚本和最小验证命令
> - [x] 加入 KV cache spec 打印/诊断，验证 OOM 是否来自普通 KV、Sink、DSA 还是 MoME
>
> </details>
</details>

已完成第一版适配工程，位置在：

[D:\workspace\pangu-v2-moe-vllm-ascend018](</D:/workspace/pangu-v2-moe-vllm-ascend018>)

我做了这些内容：

- 放入固定源码：
  - `third_party/vllm-0.18.0`
  - `third_party/vllm-ascend-0.18.0`
  - `third_party/omni-npu-release_1.2.1`
- 新增插件包：[vllm_pangu_v2_moe](</D:/workspace/pangu-v2-moe-vllm-ascend018/src/vllm_pangu_v2_moe/plugin.py>)
  - 注册 `PanguV2MoEForCausalLM`
  - 兼容你原 config 里误写的 `PanguV2aMoEForCausalLM`
  - 将 `pangu_v2_moe` 识别为 MLA 模型
  - 补 vllm-ascend 的量化 mapping
  - 加 KV cache group 诊断日志
  - 优先桥接 omni-npu 的完整 Pangu 实现，保证 DSA/SWA/SinkMLA/MoME/MHC 逻辑不被丢掉
- 生成规范化 config：
  - [config.pangu_v2_moe.normalized.json](</D:/workspace/pangu-v2-moe-vllm-ascend018/configs/config.pangu_v2_moe.normalized.json>)
  - 已修正 `architectures`
  - 已补 `rope_parameters.rope_theta=6400000`
  - 已补 `pad_token_id`
- 写了启动和安装脚本：
  - [install_plugin_only.sh](</D:/workspace/pangu-v2-moe-vllm-ascend018/scripts/install_plugin_only.sh>)
  - [prepare_model_config.sh](</D:/workspace/pangu-v2-moe-vllm-ascend018/scripts/prepare_model_config.sh>)
  - [run_4node_mixed_node.sh](</D:/workspace/pangu-v2-moe-vllm-ascend018/scripts/run_4node_mixed_node.sh>)
- 写了说明文档：
  - [ADAPTATION.md](</D:/workspace/pangu-v2-moe-vllm-ascend018/docs/ADAPTATION.md>)

我做过的本地验证：

- `normalize_config.py` 运行成功
- 生成后的 JSON 格式校验通过
- 插件和脚本 `py_compile` / `bash -n` 通过
- WSL 本地没有 `torch`，所以不能在这台机器上做 vLLM 导入和 NPU 实跑验证

在 NPU 环境上，建议先执行：

```bash
cd /mnt/d/workspace/pangu-v2-moe-vllm-ascend018
bash scripts/install_plugin_only.sh
bash scripts/prepare_model_config.sh /path/to/openpangu-505B
```

然后 4 台机器分别跑 `run_4node_mixed_node.sh`，`NODE_RANK` 用 `0,1,2,3`。node0 会启动 `vllm serve`，其他节点只加入 Ray。

> 我使用dp4tp8来部署，并且，你去查看下vllm-ascend的最新官方文档以及一些vllm-ascend最新pr，现在的主流多机混布命令好像不太一样。然后模型架构我的vllm-ascend镜像里是没有omni-npu的，你要把架构写出来，可以参考omni-npu的推理实现

<details><summary>19 previous messages</summary>

> 明白，两个方向我会一起修正：部署形态改成 `DP=4, TP=8` 的 4 机混布；模型实现不能依赖镜像里的 `omni-npu`，我要把 `PanguV2MoEForCausalLM` 代码放进适配包里。先查最新 vllm-ascend 官方多机/混布文档和相关 PR，再回头改脚本和模型实现。
>
> <details><summary>Searched web 5 times</summary>
>
> <details><summary>Searched web 5 times</summary>
>
> - Searched web for `site:vllm-ascend.readthedocs.io vllm-ascend multi node data parallel deployment v0.18.0 DP TP ...`
> - Searched web for `https://github.com/vllm-project/vllm-ascend/releases`
> - Searched web for `'data-parallel' in https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/ray.html`
> - Searched web for `https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-235B-A22B.html`
> - Searched web for `https://github.com/vllm-project/vllm-ascend/pull/7528`
>
> </details>
>
> </details>
>
> 查到的最新官方形态和我们上一版脚本确实不一样：现在 vllm-ascend 文档里的多机 DP+TP 在线服务主流写法是每台机器都跑 `vllm serve`，node0 是入口，其他节点加 `--headless`，通过 `--data-parallel-size-local` 和 `--data-parallel-start-rank` 组织 DP rank，而不是先起 Ray 再只在 node0 跑一个 serve。我会把脚本改成这个形态，并把 `DP=4 TP=8` 作为默认。
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> 我先把两件事并行推进：一边核对 vllm-ascend 最新官方多机 DP/TP 启动方式，一边检查现有适配目录，确认哪些文件需要从“桥接 omni-npu”改成“包内自带 PanguV2MoE 实现”。
>
> <details><summary>Ran 2 commands, searched web 7 times</summary>
>
> - Ran `Get-ChildItem -Force D:\workspace\pangu-v2-moe-vllm-ascend018; Get-ChildItem -Force D:\workspace\pangu-v2-moe-vllm-ascend018\src; Get-ChildItem -Force D:\workspace\pangu-v2-moe-vllm-ascend018\scripts; Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\src\vllm_pangu_v2_moe\plugin.py; Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\scripts\run_4node_mixed_node.sh`
>
> - Ran `Get-ChildItem -Recurse -Depth 2 D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\omni-npu-release_1.2.1\src\omni_npu\v1\models\pangu | Select-Object FullName; Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\pyproject.toml; Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\docs\ADAPTATION.md`
>
> <details><summary>Searched web 7 times</summary>
>
> - Searched web for `site:docs.vllm.ai/projects/ascend vllm-ascend multi node data parallel headless --data-parallel-start-rank ...`
> - Searched web for `https://docs.vllm.ai/projects/ascend/en/v0.9.1/tutorials/multi_node.html`
> - Searched web for `github vllm-project vllm-ascend PR headless data parallel start rank ...`
> - Searched web for `https://github.com/vllm-project/vllm-ascend`
> - Searched web for `site:docs.vllm.ai/projects/ascend/en/v0.18.0 multi node dp vllm-ascend`
> - Searched web for `https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen3.5-397B-A17B.html`
> - Searched web for `https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen3.5-397B-A17B.html`
>
> </details>
>
> </details>
>
> 官方文档这边已经确认：0.18.0 的大模型多机场景主推 `mp + DP/TP` 的多进程方式，每台机器都跑 `vllm serve`，非 0 节点用 `--headless`，不是老的 “先起 Ray 集群、只在 head 跑 serve”。我现在开始改脚本和适配包结构，把 PanguV2MoE 的 omni 实现随插件一起带进去。
>
> <details><summary>Ran 3 commands, searched web 7 times</summary>
>
> - Ran `Get-ChildItem -Recurse D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\omni-npu-release_1.2.1\src\omni_npu | Measure-Object | Select-Object Count; Get-ChildItem D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\omni-npu-release_1.2.1\src\omni_npu -Force`
>
> - Ran `Select-String -Path D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\omni-npu-release_1.2.1\src\omni_npu\v1\models\pangu\pangu_ultra_moe.py -Pattern "class|import|from omni_npu|DSA|SWA|MOME|MHC|dsa|swa|mome|mhc|sink" | Select-Object -First 120`
>
> - Ran `Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\src\vllm_pangu_v2_moe\model.py; Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\src\vllm_pangu_v2_moe\__init__.py; Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\src\vllm_pangu_v2_moe\config.py`
>
> <details><summary>Searched web 7 times</summary>
>
> - Searched web for `vllm-ascend v0.18.0 Qwen3.5-397B data-parallel-size-local headless`
> - Searched web for `https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen3.5-397B-A17B.html`
> - Searched web for `https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen3.5-397B-A17B.html`
> - Searched web for `site:github.com/vllm-project/vllm-ascend/pull data_parallel_start_rank headless mp deployment ...`
> - Searched web for `https://github.com/vllm-project/vllm-ascend/releases`
> - Searched web for `'## v0.18.0' in https://github.com/vllm-project/vllm-ascend/releases`
> - Searched web for `https://github.com/vllm-project/vllm-ascend/pull/7528`
>
> </details>
>
> </details>
>
> 这里有一个很关键的点：vllm-ascend 官方 0.18.0 release notes 还专门提到 hybrid attention + mamba 在升级后可能因为 block size 被改错导致启动问题，PR #7528 修了“保留 hybrid block size”。你的 PanguV2MoE 也是 hybrid/MoME 路径，所以脚本里我会显式给 `--block-size 128`，并在插件里保留 KV group 诊断，方便确认是不是块大小/混合 KV 规格走错。
>
> <details><summary>Ran 1 command</summary>
>
> - Ran `if (Test-Path D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu) { Remove-Item -Recurse -Force D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu }
> Copy-Item -Recurse -Force D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\omni-npu-release_1.2.1\src\omni_npu D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu
> Get-ChildItem -Recurse D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu | Measure-Object | Select-Object Count`
>
> </details>
>
> 我现在做的实现取舍是：不是把 omni-npu 当成运行时外部依赖，而是把它的 PanguV2MoE 相关包随这个适配包一起安装。这样镜像里即使没有 `omni_npu`，`PanguV2MoEForCausalLM` 注册时也能落到包内的完整 DSA/SWA/SinkMLA/MoME/MHC 实现。
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/pyproject.toml` (+2 -0)
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/src/vllm_pangu_v2_moe/plugin.py` (+5 -3)
>
> </details>
>
> Stream error
>
> Reconnecting... 1/5
> stream disconnected before completion: error sending request for url (https://chatgpt.com/backend-api/codex/responses)
>
> Stream error
>
> Reconnecting... 2/5
> stream disconnected before completion: error sending request for url (https://chatgpt.com/backend-api/codex/responses)
>
> Stream error
>
> Reconnecting... 3/5
> stream disconnected before completion: error sending request for url (https://chatgpt.com/backend-api/codex/responses)
>
> Stream error
>
> Reconnecting... 4/5
> stream disconnected before completion: error sending request for url (https://chatgpt.com/backend-api/codex/responses)
>
> Stream error
>
> Reconnecting... 5/5
> stream disconnected before completion: error sending request for url (https://chatgpt.com/backend-api/codex/responses)
>
> System error
>
> stream disconnected before completion: error sending request for url (https://chatgpt.com/backend-api/codex/responses)
>
> <details><summary>Plan</summary>
>
> - [ ] 按最新官方文档调整 4 机 DP4TP8 混布启动方式
> - [ ] 把 PanguV2MoE 架构实现内置到适配包，不依赖镜像已有 omni-npu
> - [ ] 修正插件注册，默认使用内置架构实现
> - [ ] 重新生成配置与文档，说明 DP4TP8 启动和调试参数
> - [ ] 做静态检查和脚本语法检查
>
> </details>
</details>

> 刚刚网络断了，请继续

<details><summary>16 previous messages</summary>

> 网络断线没影响本地改动，我继续接着做。刚才已经把 `omni_npu` 源码拷进适配包，并把插件默认注册改成使用包内的完整 PanguUltraMoE 实现；接下来改 4 机 DP4TP8 的新式启动脚本和文档。
>
> <details><summary>Created 1 file, edited 1 file</summary>
>
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/scripts/run_4node_mixed_node.sh` (+71 -0)
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/src/vllm_pangu_v2_moe/model.py` (+8 -9)
>
> </details>
>
> 启动脚本已经从 Ray 版改成了每节点 `vllm serve` 版：4 个 DP rank，每个 rank 本地 TP8，node0 开 API，node1-3 `--headless`。我继续把文档改成这个口径，并把“为什么不用外部 omni-npu”写清楚。
>
> <details><summary>Created 2 files, ran 8 commands</summary>
>
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/docs/ADAPTATION.md` (+109 -0)
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/README.md` (+10 -0)
> - Ran `Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\scripts\install_plugin_only.sh; Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\scripts\install_editable.sh; Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\scripts\check_plugin.py`
> - Ran `Select-String -Path D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu\**\*.py -Pattern "OMNI_NPU_PATCHES_DIR|VLLM_PANGU|openpangu|pangu_v2_moe" -List | Select-Object Path,LineNumber,Line`
> - Ran `Select-String -Path D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu\v1\models\pangu\pangu_ultra_moe.py -Pattern "dsa_layers|swa_layers|sliding_window|param_sink|use_mome|use_mhc|block_post_layernorm_idx|make_empty_intermediate_tensors|get_seqlen_agnostic_capture_inputs" -Context 2,3`
> - Ran `Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu\__init__.py; Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu\layers\__init__.py; Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu\vllm_patches\__init__.py`
> - Ran `Get-ChildItem D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu\model_config\config_loader -Force; Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu\model_config\config_loader\loader.py`
> - Ran `Select-String -Path D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu\v1\models\pangu\pangu_ultra_moe.py -Pattern "load_model_extra_config|model_extra_config|CUSTOM_MODEL_CONFIG_PATH|enable_prefetch|parall_config|operator_opt_config" -Context 2,2`
> - Ran `Select-String -Path D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu\**\*.py -Pattern "load_model_extra_config" -List | Select-Object Path,LineNumber,Line`
> - Ran `Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu\worker\npu_worker.py | Select-Object -First 130`
>
> </details>
>
> 我发现一个不能忽略的差异：omni 的模型实现读取 `model_extra_config`，原版 omni worker 会在 `init_device()` 里加载这些最佳实践配置；但 vllm-ascend 镜像走的是自己的 worker，不会自动跑 omni 的 worker 初始化。我要在插件里补一个轻量 patch：在 vllm-ascend worker 初始化 NPU 设备时加载 omni 的 `model_extra_config`，否则很多 PanguV2 优化开关会保持默认值。
>
> <details><summary>Edited 1 file, ran 8 commands</summary>
>
> - Ran `Get-ChildItem -Recurse D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-ascend-0.18.0\vllm_ascend -Filter *.py | Select-String -Pattern "class .*Worker|def init_device|load_model_extra_config|ModelRunner" | Select-Object Path,LineNumber,Line | Select-Object -First 120`
> - Ran `Get-ChildItem -Recurse D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-ascend-0.18.0\vllm_ascend -Filter *.py | Select-String -Pattern "initialize_kv_cache|KVCacheConfig|block_size|kv_cache_groups" | Select-Object Path,LineNumber,Line | Select-Object -First 160`
> - Ran `Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-ascend-0.18.0\vllm_ascend\worker\worker.py | Select-Object -First 220`
> - Ran `Select-String -Path D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-ascend-0.18.0\vllm_ascend\worker\worker.py -Pattern "def init_device|init_device" -Context 0,100`
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/src/vllm_pangu_v2_moe/plugin.py` (+41 -0)
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/src/vllm_pangu_v2_moe/plugin.py` (+6 -1)
> - Ran `Get-ChildItem -Recurse D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu\model_config\configs\low_latency\openpangu_v2 | Select-Object FullName; Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu\model_config\configs\low_latency\openpangu_v2\pangu_v2_moe_bf16_a3_xp1d_p.json`
> - Ran `Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu\model_config\configs\low_latency\best_practice_configs.json | Select-Object -First 220`
> - Ran `Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu\model_config\configs\match_hf_configs.json`
> - Ran `Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu\model_config\configs\low_latency\openpangu_v2\pangu_v2_moe_bf16_a3_hybrid.json; Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu\model_config\configs\low_latency\openpangu_v2\pangu_v2_moe_bf16_a3_xp1d_d.json`
>
> </details>
>
> 我还补了一个 A2/910B 专用的 `CUSTOM_MODEL_CONFIG_PATH` 默认配置。这里沿用 omni 的 hybrid PanguV2 配置，但把预取类开关关掉，避免 910B 上在权重加载后、KV 初始化前后被额外 workspace/预取内存顶爆。
>
> <details><summary>Created 1 file, edited 1 file</summary>
>
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/src/omni_npu/model_config/configs/low_latency/openpangu_v2/pangu_v2_moe_bf16_a2_dp4tp8_hybrid.json` (+29 -0)
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/scripts/run_4node_mixed_node.sh` (+1 -0)
>
> </details>
>
> Stream error
>
> Reconnecting... 1/5
> stream disconnected before completion: error sending request for url (https://chatgpt.com/backend-api/codex/responses)
>
> <details><summary>Edited 1 file</summary>
>
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/docs/ADAPTATION.md` (+5 -0)
>
> </details>
>
> Stream error
>
> Reconnecting... 1/5
> stream disconnected before completion: Transport error: network error: error decoding response body
>
> Stream error
>
> Reconnecting... 2/5
> stream disconnected before completion: error sending request for url (https://chatgpt.com/backend-api/codex/responses)
>
> Stream error
>
> Reconnecting... 3/5
> stream disconnected before completion: error sending request for url (https://chatgpt.com/backend-api/codex/responses)
>
> Stream error
>
> Reconnecting... 4/5
> stream disconnected before completion: error sending request for url (https://chatgpt.com/backend-api/codex/responses)
>
> Stream error
>
> Reconnecting... 5/5
> stream disconnected before completion: error sending request for url (https://chatgpt.com/backend-api/codex/responses)
>
> System error
>
> stream disconnected before completion: error sending request for url (https://chatgpt.com/backend-api/codex/responses)
</details>

> 刚刚忘了又断了，有未完成的任务请继续

<details><summary>10 previous messages</summary>

> 继续中。现在剩下的是收尾检查：确认脚本语法、Python 语法、打包清单，以及把官方文档/PR 的依据补进最终说明。
>
> <details><summary>Edited 2 files, ran 12 commands, searched web 3 times</summary>
>
> - Ran `Get-ChildItem -Recurse D:\workspace\pangu-v2-moe-vllm-ascend018 -Force | Where-Object { $_.Name -eq '__pycache__' } | Select-Object FullName`
>
> - Ran `python -m compileall -q D:\workspace\pangu-v2-moe-vllm-ascend018\src\vllm_pangu_v2_moe D:\workspace\pangu-v2-moe-vllm-ascend018\scripts`
>
> - Ran `bash -n /mnt/d/workspace/pangu-v2-moe-vllm-ascend018/scripts/*.sh`
>
> <details><summary>Searched web 3 times</summary>
>
> - Searched web for `vllm ascend v0.18.0 Qwen3.5 397B DP TP headless data-parallel-size-local ...`
> - Searched web for `https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen3.5-397B-A17B.html`
> - Searched web for `https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen3.5-397B-A17B.html`
>
> </details>
>
> - Ran `python -m compileall -q D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu`
>
> - Ran `Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\src\vllm_pangu_v2_moe\plugin.py; Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\scripts\run_4node_mixed_node.sh; Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\pyproject.toml`
>
> - Ran `python D:\workspace\pangu-v2-moe-vllm-ascend018\scripts\check_plugin.py`
>
> - Ran `Get-ChildItem -Recurse D:\workspace\pangu-v2-moe-vllm-ascend018 -Force | Where-Object { $_.Name -eq '__pycache__' } | Remove-Item -Recurse -Force
> Get-ChildItem -Recurse D:\workspace\pangu-v2-moe-vllm-ascend018 -Force | Where-Object { $_.Name -eq '__pycache__' } | Select-Object FullName`
>
> - Ran `Get-ChildItem -Recurse D:\workspace\pangu-v2-moe-vllm-ascend018 -Force | Where-Object { -not $_.PSIsContainer } | Select-Object FullName,Length | Sort-Object FullName | Select-Object -First 80`
>
> - Ran `Get-ChildItem -Recurse D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu -File | Where-Object { $_.Extension -notin '.py','.json','.yaml','.yml','.txt' } | Select-Object FullName,Extension | Sort-Object FullName`
>
> - Ran `Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\configs\config.pangu_v2_moe.normalized.json | Select-Object -First 60`
>
> - Ran `$url='https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen3.5-397B-A17B.html'; try { (Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 20).Content | Select-String -Pattern 'data-parallel-size|headless|api-server-count|Qwen3.5' -Context 2,2 | Select-Object -First 80 } catch { $_.Exception.Message }`
>
> - Ran `$url='https://github.com/vllm-project/vllm-ascend/releases/tag/v0.18.0'; try { $c=(Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 20).Content; $c | Select-String -Pattern 'hybrid|block|7528|release|0.18.0' -Context 1,1 | Select-Object -First 40 } catch { $_.Exception.Message }`
>
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/scripts/run_4node_mixed_node.sh` (+1 -0)
>
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/docs/ADAPTATION.md` (+1 -0)
>
> </details>
>
> 我把 `--distributed-executor-backend mp` 也显式加回去了，这和官方 0.18.0 的多机 MP 示例一致。现在再跑一遍语法检查，然后我给你一版可以直接照着试的说明。
>
> <details><summary>Edited 2 files, ran 5 commands</summary>
>
> - Ran `bash -n /mnt/d/workspace/pangu-v2-moe-vllm-ascend018/scripts/*.sh`
> - Ran `python -m compileall -q D:\workspace\pangu-v2-moe-vllm-ascend018\src\vllm_pangu_v2_moe D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu D:\workspace\pangu-v2-moe-vllm-ascend018\scripts`
> - Ran `Get-ChildItem -Recurse D:\workspace\pangu-v2-moe-vllm-ascend018 -Force | Where-Object { $_.Name -eq '__pycache__' } | Remove-Item -Recurse -Force`
> - Ran `Select-String -Path D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-0.18.0\vllm\**\*.py -Pattern "disable_hybrid_kv_cache_manager|hybrid-kv-cache" | Select-Object Path,LineNumber,Line | Select-Object -First 40`
> - Ran `Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-0.18.0\vllm\engine\arg_utils.py | Select-Object -Index 1208..1228; Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-0.18.0\vllm\config\scheduler.py | Select-Object -First 145`
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/scripts/run_4node_mixed_node.sh` (+4 -0)
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/docs/ADAPTATION.md` (+4 -2)
>
> </details>
>
> Stream error
>
> Reconnecting... 1/5
> stream disconnected before completion: Transport error: network error: error decoding response body
>
> <details><summary>Edited 1 file, ran 3 commands</summary>
>
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/docs/ADAPTATION.md` (+2 -0)
> - Ran `bash -n /mnt/d/workspace/pangu-v2-moe-vllm-ascend018/scripts/*.sh`
> - Ran `python -m compileall -q D:\workspace\pangu-v2-moe-vllm-ascend018\src\vllm_pangu_v2_moe D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu D:\workspace\pangu-v2-moe-vllm-ascend018\scripts`
> - Ran `Get-ChildItem -Recurse D:\workspace\pangu-v2-moe-vllm-ascend018 -Force | Where-Object { $_.Name -eq '__pycache__' } | Remove-Item -Recurse -Force`
>
> </details>
>
> 已继续完成，适配目录在：
>
> [D:\workspace\pangu-v2-moe-vllm-ascend018](</D:/workspace/pangu-v2-moe-vllm-ascend018>)
>
> 我做了这些关键改动：
>
> - 把 `omni_npu` 的 PanguV2MoE 推理实现直接打进适配包：`src/
>
> Stream error
>
> Reconnecting... 1/5
> stream disconnected before completion: Transport error: network error: error decoding response body
>
> Stream error
>
> Reconnecting... 2/5
> stream disconnected before completion: error sending request for url (https://chatgpt.com/backend-api/codex/responses)
>
> Stream error
>
> Reconnecting... 3/5
> stream disconnected before completion: error sending request for url (https://chatgpt.com/backend-api/codex/responses)
</details>

已继续完成，核心代码在 `D:\workspace\pangu-v2-moe-vllm-ascend018`。

这次我把适配方向从“依赖镜像里已有 omni-npu”改成了“适配包内自带 PanguV2MoE 实现”：

- 插件入口：[plugin.py](</D:/workspace/pangu-v2-moe-vllm-ascend018/src/vllm_pangu_v2_moe/plugin.py>)
- 4 机 DP4TP8 启动脚本：[run_4node_mixed_node.sh](</D:/workspace/pangu-v2-moe-vllm-ascend018/scripts/run_4node_mixed_node.sh>)
- A2/910B 配置：[pangu_v2_moe_bf16_a2_dp4tp8_hybrid.json](</D:/workspace/pangu-v2-moe-vllm-ascend018/src/omni_npu/model_config/configs/low_latency/openpangu_v2/pangu_v2_moe_bf16_a2_dp4tp8_hybrid.json>)
- 说明文档：[ADAPTATION.md](</D:/workspace/pangu-v2-moe-vllm-ascend018/docs/ADAPTATION.md>)

我做了这些关键改动：

1. 把 omni-npu `release_1.2.1` 的 `omni_npu` 源码打进本适配包，注册 `PanguV2MoEForCausalLM -> omni_npu.v1.models.pangu.pangu_ultra_moe:PanguUltraMoEForCausalLM`，镜像里不需要预装 omni-npu。

2. 启动脚本改成 vllm-ascend 0.18.0 官方当前主流的多机 MP/DP 方式：每台机器都跑 `vllm serve`，node0 开 API，node1-3 用 `--headless`，并显式使用：
   - `--data-parallel-size 4`
   - `--data-parallel-size-local 1`
   - `--data-parallel-start-rank NODE_RANK`
   - `--tensor-parallel-size 8`
   - `--distributed-executor-backend mp`

3. 针对你说的“加载完模型后 KV cache 分配 OOM”，我默认加了几项保护：
   - `--no-disable-hybrid-kv-cache-manager`
   - `--block-size 128`
   - `--no-enable-prefix-caching`
   - `MAX_NUM_BATCHED_TOKENS=128`
   - `CUSTOM_MODEL_CONFIG_PATH=low_latency/openpangu_v2/pangu_v2_moe_bf16_a2_dp4tp8_hybrid.json`
   - 这个 A2 配置里已关闭 `enable_prefetch` 和所有 prefetch MB。

4. 插件里补了 vllm-ascend worker patch，会在 NPU worker 初始化后加载 omni 的 `model_extra_config`，否则 PanguV2MoE 的一些关键优化配置不会生效。

启动方式示例：

```bash
cd /mnt/d/workspace/pangu-v2-moe-vllm-ascend018
bash scripts/install_plugin_only.sh
bash scripts/prepare_model_config.sh /path/to/openpangu-505B
```

node0：

```bash
bash scripts/run_4node_mixed_node.sh /path/to/openpangu-505B 10.0.0.1 0 10.0.0.1 8000
```

node1-3：

```bash
bash scripts/run_4node_mixed_node.sh /path/to/openpangu-505B 10.0.0.1 1 10.0.0.2 8000
bash scripts/run_4node_mixed_node.sh /path/to/openpangu-505B 10.0.0.1 2 10.0.0.3 8000
bash scripts/run_4node_mixed_node.sh /path/to/openpangu-505B 10.0.0.1 3 10.0.0.4 8000
```

我查了 vllm-ascend 官方 0.18.0 文档，Qwen3.5-397B 的多机 MP 示例也是这种 `--data-parallel-size-local`、`--data-parallel-start-rank`、`--headless`、`--distributed-executor-backend mp` 形态：[官方文档](https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen3.5-397B-A17B.html)。另外 hybrid KV/block size 这块我也参考了 vllm-ascend 的相关修复方向：[PR #7528](https://github.com/vllm-project/vllm-ascend/pull/7528)。

已做验证：

```text
bash -n scripts/*.sh                      通过
python -m compileall src scripts           通过
```

本地 Windows 环境没有安装 `vllm`，所以 `scripts/check_plugin.py` 不能在这里跑通导入，报的是 `No module named 'vllm'`；这个需要在你的 vllm-ascend NPU 镜像里执行。

> 再给我写一个中文文档，并且讲一下运行该模型是vllm的启动逻辑

<details><summary>6 previous messages</summary>

> 我来补一份中文文档，重点写成“怎么部署”和“vLLM 从命令到模型/KV 初始化的启动链路”两部分，方便你后面排 OOM 时按阶段对照日志。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\docs\ADAPTATION.md`
> - Ran `Select-String -Path D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-0.18.0\vllm\**\*.py,D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-ascend-0.18.0\vllm_ascend\**\*.py -Pattern "def serve|class ModelConfig|register_general_plugins|general_plugins|ModelRegistry|initialize_kv_cache|determine_available_memory|init_device|load_model" | Select-Object Path,LineNumber,Line | Select-Object -First 160`
>
> </details>
>
> 我已经定位了 vLLM 插件加载、ModelConfig、vllm-ascend worker、ModelRunner/KV cache 几个关键源码点。现在直接新增一份中文文档，不改现有英文文档。
>
> <details><summary>Created 1 file, ran 3 commands</summary>
>
> - Ran `Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-ascend-0.18.0\vllm_ascend\worker\model_runner_v1.py | Select-Object -Skip 2540 -First 130`
> - Ran `Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-ascend-0.18.0\vllm_ascend\worker\worker.py | Select-Object -Skip 240 -First 130`
> - Ran `Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-0.18.0\vllm\plugins\__init__.py`
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/docs/PANGU_V2_MOE_CN.md` (+381 -0)
>
> </details>
>
> 中文文档已写好。我再快速检查一下文档和脚本，避免 Markdown 里命令或文件路径有明显笔误。
>
> <details><summary>Edited 1 file, ran 3 commands</summary>
>
> - Ran `Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\docs\PANGU_V2_MOE_CN.md | Select-Object -First 80`
> - Ran `Select-String -Path D:\workspace\pangu-v2-moe-vllm-ascend018\docs\PANGU_V2_MOE_CN.md -Pattern "PanguV2a|PanguV2MoE|hybrid|KV cache|vllm serve|CUSTOM_MODEL_CONFIG_PATH" | Select-Object -First 60`
> - Ran `bash -n /mnt/d/workspace/pangu-v2-moe-vllm-ascend018/scripts/*.sh`
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/README.md` (+1 -0)
>
> </details>
</details>

已补中文文档：[PANGU_V2_MOE_CN.md](</D:/workspace/pangu-v2-moe-vllm-ascend018/docs/PANGU_V2_MOE_CN.md>)。

里面主要写了：

- 4 机 `DP=4, TP=8` 的安装、config 修正、启动命令。
- 为什么该模型不能只走 vLLM 原生 OpenPangu，而要注册到包内 `omni_npu` PanguV2MoE 实现。
- vLLM 启动该模型的完整链路：命令行解析、插件加载、架构注册、多进程初始化、NPUWorker 初始化、模型加载、profile run、KV cache config 生成和 KV tensor 分配。
- “模型加载完后 OOM”最可能发生在哪两个阶段，以及对应看哪些日志。
- 第一次跑通建议参数和后续逐步放量顺序。

我也在 [README.md](</D:/workspace/pangu-v2-moe-vllm-ascend018/README.md>) 里加了中文文档入口。脚本语法检查已通过。

> vllm-ascend:v0.9.1-dev适配过openPangu-Ultra-MoE-718B，当然他和我们的模型完全不一样，看一下之前是怎么适配的，参考下相关pr

<details><summary>10 previous messages</summary>

> 我去查 vllm-ascend 早期 `v0.9.1-dev` 里 OpenPangu-Ultra-MoE-718B 的适配脉络，重点看：模型注册、Ascend patch、KV/hybrid cache、权重加载和启动参数。然后我把和我们 PanguV2MoE 505B 可复用、不可复用的部分拆出来。
>
> <details><summary>Searched web 2 times</summary>
>
> <details><summary>Searched web 2 times</summary>
>
> - Searched web for `vllm-ascend OpenPangu-Ultra-MoE-718B PR v0.9.1-dev ...`
> - Searched web for `"PanguUltraMoEForCausalLM" github ...`
>
> </details>
>
> </details>
>
> 我找到了 718B 官方部署文档和相关代码线索：那个版本已经走了 `vllm serve + data-parallel-size-local + headless` 这种多机 DP/TP 启动方式，而且模型类型是 `pangu_ultra_moe`。我继续从 GitHub commit/PR 里追它当时进 vLLM/vllm-ascend 的改动点。
>
> <details><summary>Ran 3 commands, searched web 13 times</summary>
>
> - Ran `git ls-remote --tags https://github.com/vllm-project/vllm-ascend.git | Select-String -Pattern "v0.9.1|0.9"`
>
> - Ran `git ls-remote https://github.com/vllm-project/vllm-ascend.git refs/heads/main refs/heads/* | Select-String -Pattern "pangu|0.9|dev"`
>
> - Ran `git ls-remote --tags https://github.com/vllm-project/vllm.git | Select-String -Pattern "v0.9|v0.10|v0.12|v0.14" | Select-Object -First 80`
>
> <details><summary>Searched web 13 times</summary>
>
> - Searched web for `docs.vllm.ai projects ascend v0.9.1-dev openPangu-Ultra-MoE-718B ...`
> - Searched web for `https://huggingface.co/openpangu/openPangu-Ultra-MoE-718B-model/blob/main/doc/vllm_ascend_for_openpangu_ultra_moe_718b.md`
> - Searched web for `https://huggingface.co/openpangu/openPangu-Ultra-MoE-718B-model/tree/main`
> - Searched web for `https://huggingface.co/openpangu/openPangu-Ultra-MoE-718B-model/tree/main/inference`
> - Searched web for `https://huggingface.co/openpangu/openPangu-Ultra-MoE-718B-model/tree/main/inference/vllm_ascend`
> - Searched web for `https://huggingface.co/openpangu/openPangu-Ultra-MoE-718B-model/tree/main/inference/vllm_ascend/models`
> - Searched web for `https://huggingface.co/openpangu/openPangu-Ultra-MoE-718B-model/blob/main/inference/vllm_ascend/models/open_pangu.py`
> - Searched web for `'class OpenPanguUltraMoEForCausalLM' in https://huggingface.co/openpangu/openPangu-Ultra-MoE-718B-model/blob/main/inference/vllm_ascend/models/open_pangu.py`
> - Searched web for `"PanguUltraMoEForCausalLM" "github.com/vllm-project/vllm" ...`
> - Searched web for `site:github.com/vllm-project/vllm/pull openpangu ...`
> - Searched web for `https://github.com/vllm-project/vllm/pulls?q=is%3Apr+openpangu`
> - Searched web for `https://github.com/vllm-project/vllm-ascend/pull/4615`
> - Searched web for `https://github.com/vllm-project/vllm-ascend/pull/4615/files`
>
> </details>
>
> </details>
>
> PR 线索已经比较清楚：vLLM 主仓先通过 #27521 加原生 `openpangu.py` / `openpangu_mtp.py`，后面 #28775 继续补 openPangu MoE；vllm-ascend #4615 主要不是重写模型，而是补 Ascend 量化映射和 MTP proposer。现在我再拉一下 patch diff，确认具体文件级改动。
>
> <details><summary>Ran 7 commands</summary>
>
> - Ran `$urls=@('https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/27521.patch','https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/28775.patch','https://patch-diff.githubusercontent.com/raw/vllm-project/vllm-ascend/pull/4615.patch'); foreach($u in $urls){ Write-Output "URL $u"; try { (Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 30).Content | Select-String -Pattern '^diff --git|^Subject:|^\+class|^\+.*Pangu|^\+.*pangu|^\+.*ModelRegistry|^\+.*packed|^\+.*MTP|^\+.*fused_qkv' | Select-Object -First 120 } catch { Write-Output $_.Exception.Message } }`
> - Ran `$raw='https://huggingface.co/openpangu/openPangu-Ultra-MoE-718B-model/raw/main/inference/vllm_ascend/models/open_pangu.py'; try { $c=(Invoke-WebRequest -Uri $raw -UseBasicParsing -TimeoutSec 30).Content; $c | Select-String -Pattern 'class OpenPanguMoE|class OpenPanguMLAAttention|class OpenPanguDecoderLayer|class OpenPanguForCausalLM|class PanguUltraMoEForCausalLM|def load_weights|def load_expert_weight|def load_attn_mlp_weight|enable_force_load_balance|AscendFusedMoE' -Context 1,2 | Select-Object -First 120 } catch { Write-Output $_.Exception.Message }`
> - Ran `$dir='D:\workspace\pangu-v2-moe-vllm-ascend018\tmp_prs'; New-Item -ItemType Directory -Force $dir | Out-Null; $prs=@{ 'vllm_27521.patch'='https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/27521.patch'; 'vllm_28775.patch'='https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/28775.patch'; 'ascend_4615.patch'='https://patch-diff.githubusercontent.com/raw/vllm-project/vllm-ascend/pull/4615.patch'}; foreach($k in $prs.Keys){ try { Invoke-WebRequest -Uri $prs[$k] -UseBasicParsing -TimeoutSec 60 -OutFile (Join-Path $dir $k); Write-Output "saved $k" } catch { Write-Output "fail $k $($_.Exception.Message)" } }; Get-ChildItem $dir | Select-Object Name,Length`
> - Ran `$raw='https://huggingface.co/openpangu/openPangu-Ultra-MoE-718B-model/raw/main/doc/vllm_ascend_for_openpangu_ultra_moe_718b.md'; try { (Invoke-WebRequest -Uri $raw -UseBasicParsing -TimeoutSec 60).Content | Select-String -Pattern 'vllm-ascend|data-parallel|headless|tensor-parallel|OpenPangu|718B|block-size|hybrid|VLLM|quantization' -Context 1,1 | Select-Object -First 120 } catch { Write-Output $_.Exception.Message }`
> - Ran `Select-String -Path D:\workspace\pangu-v2-moe-vllm-ascend018\tmp_prs\vllm_27521.patch -Pattern '^diff --git|^Subject:|openpangu|Pangu|ModelRegistry|architectures|mtp|MTP|packed_modules|pangu' | Select-Object -First 200`
> - Ran `Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\tmp_prs\ascend_4615.patch`
> - Ran `Select-String -Path D:\workspace\pangu-v2-moe-vllm-ascend018\tmp_prs\vllm_28775.patch -Pattern '^diff --git|^Subject:|openpangu|Pangu|ModelRegistry|architectures|mtp|MTP|packed_modules|pangu' | Select-Object -First 220`
>
> </details>
>
> 结论方向已经出来：718B 的适配是“两段式”的，先把通用 OpenPangu 模型/MTP 放进 vLLM 主仓，再在 vllm-ascend 侧补 Ascend 量化 mapping 和 MTP proposer；模型仓里还提供了一份覆盖 vllm_ascend 的 NPU 专用实现。这个思路和我们现在“插件注册 + 包内模型实现 + Ascend worker/config patch”的路线是同一类，只是 505B PanguV2MoE 的 DSA/SWA/MoME/MHC 更复杂。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `Select-String -Path D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-0.18.0\vllm\model_executor\models\openpangu.py -Pattern "PanguProMoEV2ForCausalLM|OpenPanguSinkAttention|Static|Diffkv|param_sink|packed_modules_mapping|fused_qkv_a_proj|get_seqlen_agnostic_capture_inputs|PanguUltraMoEForCausalLM|class OpenPanguMoE" -Context 1,2`
> - Ran `Select-String -Path D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu\v1\models\pangu\pangu_ultra_moe.py -Pattern "PanguProMoEV2ForCausalLM|OpenPanguSinkAttention|NPUDeepseekSparseAttention|NPUDeepseekMLAAttention|MomeSpec|IsHybrid|param_sink|dsa_layers|use_mome|get_seqlen_agnostic_capture_inputs|class PanguUltraMoEForCausalLM|packed_modules_mapping|fused_qkv_a_proj" -Context 1,2`
>
> </details>
>
> 我会把这次追 PR 的结果落一份参考文档，后面你对照我们当前适配时不用再翻 PR。文档会明确：718B 那套哪些可以借鉴，哪些不能直接套到 505B PanguV2MoE。
>
> <details><summary>Created 1 file, edited 1 file, ran 2 commands</summary>
>
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/docs/OPENPANGU_ULTRA_718B_REFERENCE_CN.md` (+228 -0)
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/README.md` (+1 -0)
> - Ran `Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\docs\OPENPANGU_ULTRA_718B_REFERENCE_CN.md | Select-Object -First 80; Get-ChildItem D:\workspace\pangu-v2-moe-vllm-ascend018\tmp_prs | Select-Object Name,Length`
> - Ran `Remove-Item -Recurse -Force D:\workspace\pangu-v2-moe-vllm-ascend018\tmp_prs`
>
> </details>
</details>

我看完并整理了，已经新增一份参考文档：

[OPENPANGU_ULTRA_718B_REFERENCE_CN.md](</D:/workspace/pangu-v2-moe-vllm-ascend018/docs/OPENPANGU_ULTRA_718B_REFERENCE_CN.md>)

核心结论是：718B 当时不是简单靠 vllm-ascend 一个 PR 适配的，而是三段式：

1. vLLM 主仓先加 OpenPangu 架构
   [vLLM PR #27521](https://github.com/vllm-project/vllm/pull/27521) 新增了 `openpangu.py`、`openpangu_mtp.py`、`PanguUltraMoEForCausalLM`、`OpenPanguMTPModel` 等。

2. vLLM 主仓继续补 Pangu 系 attention/KV 能力
   [vLLM PR #28775](https://github.com/vllm-project/vllm/pull/28775) 涉及 static sink attention、diff-KV attention、KV cache manager、block table、model runner 等。这说明 Pangu 系模型的适配不只是模型类，还经常牵涉 KV/cache 后端。

3. vllm-ascend 侧只补 Ascend 后端映射
   [vllm-ascend PR #4615](https://github.com/vllm-project/vllm-ascend/pull/4615) 主要加了 `pangu_ultra_moe` / `pangu_ultra_moe_mtp` 的 quant packed mapping，以及 MTP proposer 映射。

对我们现在 505B 的启发是：启动方式、packed mapping、`--enable-expert-parallel`、`--headless`、`--no-enable-prefix-caching` 这些可以借鉴；但模型实现不能直接套 718B 的 `PanguUltraMoEForCausalLM`，因为我们的 `pangu_v2_moe` 多了 DSA/SWA/MoME/MHC/hybrid state，所以仍然要走当前适配包里 bundled `omni_npu` 的 PanguV2MoE 路径。

我也把 README 加了入口：[README.md](</D:/workspace/pangu-v2-moe-vllm-ascend018/README.md>)。

718B 官方部署文档参考：[openPangu-Ultra-MoE-718B vllm-ascend 部署指导](https://huggingface.co/openpangu/openPangu-Ultra-MoE-718B-model/blob/main/doc/vllm_ascend_for_openpangu_ultra_moe_718b.md)。

> 帮我写一个中文文档，告诉我如果我直接采用修改vllm和vllm-ascend源码的方式，或者通过patch在源码上打补丁，要怎么做才能适配505B,按步骤来。

<details><summary>4 previous messages</summary>

> 我来补一份“源码改造/打补丁方案”的中文步骤文档，按 vLLM 主仓、vllm-ascend、模型 config、启动验证四条线写清楚。这个文档会和前面插件方案并列，方便你决定是走侵入式改源码，还是后续维护 patch。
>
> <details><summary>Created 1 file</summary>
>
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/docs/SOURCE_PATCH_ADAPTATION_CN.md` (+704 -0)
>
> </details>
>
> 源码/patch 方案文档已经写好，我把 README 也加上入口，再做一次简单检查。
>
> <details><summary>Edited 1 file, ran 2 commands</summary>
>
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/README.md` (+1 -0)
> - Ran `Select-String -Path D:\workspace\pangu-v2-moe-vllm-ascend018\docs\SOURCE_PATCH_ADAPTATION_CN.md -Pattern "步骤|PanguV2MoEForCausalLM|pangu_v2_moe|vllm-ascend|KV|CUSTOM_MODEL_CONFIG_PATH|git apply" | Select-Object -First 80`
> - Ran `Get-Content D:\workspace\pangu-v2-moe-vllm-ascend018\README.md`
>
> </details>
</details>

已写好中文步骤文档：

[SOURCE_PATCH_ADAPTATION_CN.md](</D:/workspace/pangu-v2-moe-vllm-ascend018/docs/SOURCE_PATCH_ADAPTATION_CN.md>)

文档按源码/patch 方式拆成了完整步骤，包括：

- 准备 vLLM / vllm-ascend 源码环境
- 在 vLLM 中加入 `PanguV2MoEForCausalLM`
- 迁移/引用 omni-npu 的 PanguV2MoE 实现
- 修改 `ModelRegistry`
- 修正 MLA / hybrid 检测
- 修改 vllm-ascend packed module / quant mapping
- 让 vllm-ascend worker 加载 `model_extra_config`
- 增加 910B/A2 保守配置
- 修正模型 `config.json`
- 按 `DP=4, TP=8` 启动
- 增加 KV cache debug 日志
- 如何生成和应用 `git apply` patch
- 建议把 patch 拆成哪些文件维护
- 常见失败点和排查顺序

我也在 [README.md](</D:/workspace/pangu-v2-moe-vllm-ascend018/README.md>) 里加了这个文档入口。

> 这里的步骤和你之前的适配工程一致吗，请仔细检查一下，另外源码/patch 方式文档里的修改点请把具体要在源码哪几行修改都写清楚，并把修改的代码写完整。

<details><summary>11 previous messages</summary>

> 我会按当前适配工程逐项对照源码方式文档：先确认插件方案实际做了哪些 runtime patch，再在 vLLM/vllm-ascend 0.18.0 源码里定位对应文件和行号，然后把文档改成“在哪一行附近改、替换/新增什么完整代码”。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Select-String -Path D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-0.18.0\vllm\transformers_utils\model_arch_config_convertor.py -Pattern "is_deepseek_mla|deepseek|kv_lora_rank|model_type" -Context 3,8`
> - Ran `Select-String -Path D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-0.18.0\vllm\model_executor\models\registry.py -Pattern "_VLLM_MODELS|PanguUltra|ModelRegistry|register_model|PanguEmbedded" -Context 3,6`
> - Ran `Select-String -Path D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-0.18.0\vllm\config\model.py -Pattern "class ModelConfig|def __post_init__|hf_config|architectures|rope_parameters|pad_token_id" -Context 2,6`
>
> </details>
>
> 我已经发现一个需要修正的点：源码文档和插件工程在“模型实现放哪里”这件事上表述不够一致。插件工程是把 `omni_npu` 随包带入并用 registry 注册到它；源码方式如果要完全一致，也应该优先把 `omni_npu` 放进环境，再用一个很薄的 vLLM 模型入口文件转发到 omni 实现，而不是建议大段手工迁移模型代码。接下来我会按这个方向重写具体修改点。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Select-String -Path D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-ascend-0.18.0\vllm_ascend\quantization\quant_config.py -Pattern "pangu|packed|fused_qkv_a_proj|gate_up_proj|pangu_ultra" -Context 2,6`
> - Ran `Select-String -Path D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-ascend-0.18.0\vllm_ascend\quantization\modelslim_config.py -Pattern "packed_modules_model_mapping|deepseek|pangu|fused_qkv_a_proj|gate_up_proj" -Context 2,6`
> - Ran `Select-String -Path D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-ascend-0.18.0\vllm_ascend\worker\worker.py -Pattern "def _init_device|def init_device|self.model_runner|init_workspace_manager|set_random_seed|init_device_properties_triton" -Context 3,8`
>
> </details>
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> 我继续把这份“源码/patch 方式”文档对齐到我们已经做出来的适配工程。刚才检查到一个关键点：之前文档里把“迁移模型实现到 vLLM”说得太宽了，和当前工程的实际做法不完全一致，应该明确成“vLLM 只加薄入口，真实实现复用/引入 omni_npu”。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `Select-String -Path 'D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-ascend-0.18.0\vllm_ascend\worker\model_runner_v1.py' -Pattern 'def initialize_kv_cache|kv_cache_groups|self.kv_cache_config|deepcopy\(kv_cache_config\)|logger' -Context 2,8`
> - Ran `Get-Content -Path 'D:\workspace\pangu-v2-moe-vllm-ascend018\docs\SOURCE_PATCH_ADAPTATION_CN.md' -Raw`
> - Ran `Select-String -Path 'D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-0.18.0\vllm\model_executor\models\registry.py' -Pattern 'PanguEmbeddedForCausalLM|PanguProMoEV2ForCausalLM|PanguUltraMoEForCausalLM|_VLLM_MODELS|ModelRegistry\(' -Context 2,6`
> - Ran `Select-String -Path 'D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-0.18.0\vllm\config\model.py','D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-0.18.0\vllm\transformers_utils\model_arch_config_convertor.py','D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-ascend-0.18.0\vllm_ascend\quantization\modelslim_config.py','D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-ascend-0.18.0\vllm_ascend\worker\worker.py' -Pattern 'def __post_init__|hf_config = get_config|self.hf_config = hf_config|self.hf_text_config = get_hf_text_config|def is_deepseek_mla|pangu_ultra_moe|packed_modules_model_mapping|def init_device|self.device = self._init_device|init_workspace_manager' -Context 2,6`
>
> </details>
>
> 我会直接重写这份文档，而不是在旧文档上小修小补：这样可以把“插件工程做法”和“源码/patch 做法”一一对齐，并把每个修改点固定到当前下载的 vLLM/vllm-ascend 0.18.0 源码位置。
>
> <details><summary>Created 1 file</summary>
>
> - Wrote `D:/workspace/pangu-v2-moe-vllm-ascend018/docs/SOURCE_PATCH_ADAPTATION_CN.md` (+799 -0)
>
> </details>
>
> 文档已经重写完成，现在我做一轮核对：重点看它是否包含当前工程的真实做法、每个源码文件的具体位置、完整代码块，以及启动命令是否仍是 DP4TP8 的 4 机混布方案。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Select-String -Path 'D:\workspace\pangu-v2-moe-vllm-ascend018\docs\SOURCE_PATCH_ADAPTATION_CN.md' -Pattern '与当前适配工程是否一致|PanguV2MoEForCausalLM|vllm/model_executor/models/pangu_v2_moe.py|registry.py|model.py|model_arch_config_convertor.py|modelslim_config.py|worker.py|model_runner_v1.py|DP4TP8|--headless|load_model_extra_config'`
> - Ran `git -C 'D:\workspace\pangu-v2-moe-vllm-ascend018' diff -- docs/SOURCE_PATCH_ADAPTATION_CN.md`
> - Ran `Get-Content -Path 'D:\workspace\pangu-v2-moe-vllm-ascend018\docs\SOURCE_PATCH_ADAPTATION_CN.md' -TotalCount 60`
>
> </details>
</details>

已仔细检查并重写了源码/patch 方式文档，文件在：

[docs/SOURCE_PATCH_ADAPTATION_CN.md](D:/workspace/pangu-v2-moe-vllm-ascend018/docs/SOURCE_PATCH_ADAPTATION_CN.md)

我主要修正了两点：

1. **和之前适配工程对齐**
   文档现在明确采用我们当前工程一致的方案：vLLM 只加 `PanguV2MoEForCausalLM` 薄入口和注册，真实 PanguV2MoE 实现复用 `omni_npu.v1.models.pangu.pangu_ultra_moe.PanguUltraMoEForCausalLM`，不是把整套模型实现硬迁移进 vLLM 原生 `openpangu.py`。

2. **把源码修改点写到具体文件和行附近**
   已补全每个修改点的源码位置和完整代码，包括：
   - `vllm/model_executor/models/pangu_v2_moe.py` 新增完整文件
   - `vllm/model_executor/models/registry.py` 第 181 行附近注册架构
   - `vllm/config/model.py` 第 101、490 行附近增加 config 归一化
   - `vllm/transformers_utils/model_arch_config_convertor.py` 第 212、227 行附近补 MLA 检测
   - `vllm_ascend/quantization/modelslim_config.py` 第 88 行附近补 packed mapping
   - `vllm_ascend/worker/worker.py` 第 309、313 行附近加载 `model_extra_config`
   - `vllm_ascend/worker/model_runner_v1.py` 第 2608 行附近加 KV cache 诊断日志
   - omni-npu A2/910B DP4TP8 extra config 完整 JSON
   - 4 机 `DP4TP8` 的启动命令

另外我也把“为什么权重加载后 KV cache 阶段 OOM”的排查重点写进去了：重点看是否真正走到了 omni-npu 实现、`load_model_extra_config()` 是否执行、hybrid KV group 是否正确，以及是否仍误用了 A3 优化路径。

> 有没有什么执行过程中可能遇到的算子问题？

<details><summary>3 previous messages</summary>

> 我先按当前适配路径把可能触发的 NPU 算子点梳理一下，重点会看 omni-npu 的 PanguV2/Ultra MoE 实现里用了哪些自定义/融合算子，以及哪些开关在 910B 上最容易踩坑。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `rg -n "torch_npu|npu_|Ascend|Fused|fused|moe|topk|top_k|dispatch|combine|MLA|MambaSpec|KVCache|Moe|Expert|custom" "D:\workspace\pangu-v2-moe-vllm-ascend018\third_party\vllm-ascend-0.18.0\vllm_ascend"`
> - Ran `rg -n "torch_npu|npu_|npu\.|custom|moe|MHC|mome|prefetch|fusion|fused|topk|top_k|dispatch|combine|MLA|Deepseek|Sparse|MambaSpec|MomeSpec|load_model_extra_config|model_extra_config" "D:\workspace\pangu-v2-moe-vllm-ascend018\src\omni_npu"`
>
> </details>
>
> 我看到风险主要集中在四类：attention/MLA 融合算子、MoE dispatch/combine 通信算子、MoME/MHC/DSA 相关自定义算子，以及采样/rope/KV 更新这类小融合算子。下面我按“最可能先遇到”的顺序讲。
</details>

有，而且 910B 上第一版跑通时我建议把这些算子风险当成重点防线来看。

**最可能遇到的问题**

1. **Attention / MLA 融合算子不支持某些 shape**
   代码里会走这些 NPU 算子：

   - `torch_npu.npu_fused_infer_attention_score`
   - `torch_npu.npu_fused_infer_attention_score_v2`
   - `torch_npu._npu_fused_infer_attention_score_v2_infer_output`
   - `torch.ops.npu.npu_fused_infer_attention_score`
   - `torch.ops.custom.npu_fused_infer_attention_sink`

   omni-npu 代码里已经有注释说 `npu_fused_infer_attention_score_v2` 对部分 `TND + dim=256` 场景不支持。这类问题通常表现为：
   - `ACLNN` / `aclnn...` 算子报 shape 不支持
   - tiling 失败
   - `EZ9999` / `AICore Error`
   - profile run 阶段直接挂

   所以我们现在先设置：
   ```bash
   export VLLM_ASCEND_ENABLE_MLAPO=0
   ```
   并且 extra config 里保守打开：
   ```json
   "use_aicpu_fa_tiling": true,
   "use_noncontiguous_kv": true
   ```

2. **MoE dispatch/combine 通信算子问题**
   你是 `DP4TP8`，又开了：
   ```bash
   --enable-expert-parallel
   ```
   vllm-ascend 会根据 A2/A3、EP size、量化方式选择 MoE 通信路径。910B/A2 上最容易出问题的是：
   - fused MC2 路径误开
   - dispatch/combine shape 不符合
   - HCCL/ROCE 初始化正常，但 MoE 通信算子运行失败
   - expert routing 后 token 分布导致 workspace 异常

   所以现在建议继续保留：
   ```bash
   export VLLM_ASCEND_ENABLE_FUSED_MC2=0
   ```
   omni extra config 里先用：
   ```json
   "moe_comm_strategy": "dispatch_combine",
   "enable_multi_stream": false,
   "split_q_up_in_multistream": false
   ```

   如果这里出错，日志一般会出现 `dispatch`, `combine`, `moe`, `alltoall`, `mc2`, `gmm` 相关字样。

3. **MoME / hybrid state 相关算子或 spec 不匹配**
   这个模型有 `use_mome`，如果 `layer_types`、hybrid KV manager、`MomeSpec` 没接上，可能不是普通算子报错，而是：
   - KV cache 分配异常大
   - Mamba/MoME state 维度不对
   - decode 时 state update 报 shape mismatch
   - 很短序列也 OOM

   所以你现在要特别看我们加的 KV 日志：
   ```text
   PanguV2 KV group[...]: spec=...
   ```
   如果所有层都变成普通 `AttentionSpec`，那后面即使不报算子错，也容易在 KV cache 分配阶段炸。

4. **自定义算子库没加载**
   omni-npu 的 MLA backend 会尝试 import：
   - `omni_training_custom_ops`
   - `omni_custom_ops`

   并且可能调用：
   ```python
   torch.ops.custom.npu_fused_infer_attention_sink
   ```

   如果镜像里没有对应 so 或注册没成功，常见错误是：
   ```text
   '_OpNamespace' object has no attribute ...
   No such operator custom::...
   Failed to import omni_custom_ops
   ```

   第一版如果遇到这个，不要先怀疑模型权重，先确认镜像里 custom ops 是否存在、`LD_LIBRARY_PATH` 是否正确、`import omni_custom_ops` 是否成功。

**我建议第一版先禁用或保守处理的点**

保持这些设置：

```bash
export VLLM_ASCEND_ENABLE_MLAPO=0
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

extra config 里先这样：

```json
"enable_prefetch": false,
"expert_gate_up_prefetch": 0,
"expert_down_prefetch": 0,
"attn_prefetch": 0,
"dense_mlp_prefetch": 0,
"lm_head_prefetch": 0,
"enable_multi_stream": false,
"use_mhc_fusion_op": false,
"disable_npu_top_k_top_p_sample": true
```

**如果启动后报算子错，优先按这个顺序定位**

1. 先看报错算子名，是 `attention`、`moe dispatch/combine`、`custom`、`topk/sample` 还是 `scatter/update`。
2. 确认 `load_model_extra_config()` 已执行，不然你可能根本没用到 A2 保守配置。
3. 确认 KV group 日志正常，避免把 hybrid/MoME 错看成普通 attention。
4. 如果是 custom op 缺失，先在容器里跑：
   ```bash
   python - <<'PY'
   import torch
   import torch_npu
   try:
       import omni_custom_ops
       print("omni_custom_ops ok")
   except Exception as e:
       print("omni_custom_ops failed:", e)
   print(hasattr(torch.ops, "custom"))
   print(hasattr(torch.ops, "npu"))
   PY
   ```

整体判断：这个模型第一阶段最容易卡的不是普通 matmul，而是 **MLA attention shape、MoE 通信、MoME/hybrid KV spec、自定义算子加载** 这四块。先把这些路径保守化，跑过 KV cache 初始化和第一条 decode，再逐项开性能优化。
