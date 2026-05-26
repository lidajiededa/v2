# openPangu-Ultra-MoE-718B 适配参考

本文档记录 vllm-ascend `v0.9.1-dev` 附近对
`openPangu-Ultra-MoE-718B` 的适配方式，并说明它对当前
OpenPangu-505B `PanguV2MoEForCausalLM` 适配的参考价值。

## 参考来源

- 部署文档：
  `https://huggingface.co/openpangu/openPangu-Ultra-MoE-718B-model/blob/main/doc/vllm_ascend_for_openpangu_ultra_moe_718b.md`
- vLLM PR #27521：
  `https://github.com/vllm-project/vllm/pull/27521`
- vLLM PR #28775：
  `https://github.com/vllm-project/vllm/pull/28775`
- vllm-ascend PR #4615：
  `https://github.com/vllm-project/vllm-ascend/pull/4615`
- 模型仓内 vllm-ascend 覆盖代码：
  `https://huggingface.co/openpangu/openPangu-Ultra-MoE-718B-model/tree/main/inference/vllm_ascend`

## 718B 当时的适配方式

718B 的适配不是单点改动，而是三层组合。

第一层是 vLLM 主仓模型支持。PR #27521 新增了：

- `vllm/model_executor/models/openpangu.py`
- `vllm/model_executor/models/openpangu_mtp.py`
- `PanguUltraMoEForCausalLM` 注册
- `OpenPanguMTPModel` 注册
- `pangu_ultra_moe` / `pangu_ultra_moe_mtp` 的 speculative config 支持

这一层解决的是 vLLM 能识别架构、构图、加载权重、执行普通 OpenPangu MoE/MTP。

第二层是 vLLM 主仓继续补 Pangu 系模型能力。PR #28775 主要扩展了：

- `PanguProMoEV2ForCausalLM`
- static sink attention / diff-KV attention 相关路径
- `StaticSinkAttention`
- `FlashAttentionDiffKVBackend`
- KV cache manager、block table、input batch、model runner 对 sink/diff-KV 的支持

这一层说明 Pangu 系模型不只是模型文件问题，还可能需要 vLLM core attention/KV cache
能力配合。

第三层是 vllm-ascend 侧适配。PR #4615 改动很小，但很关键：

- 在 Ascend quant config 中加入 `pangu_ultra_moe` 的 packed module mapping
- 在 Ascend quant config 中加入 `pangu_ultra_moe_mtp` 的 packed module mapping
- 在 Ascend MTP proposer 中加入：

```text
PanguUltraMoEForCausalLM -> vllm.model_executor.models.openpangu_mtp:OpenPanguMTP
```

也就是说，718B 在 vllm-ascend 侧没有重新发明整套模型架构，而是复用 vLLM 主仓
OpenPangu 实现，再补 Ascend 后端需要知道的量化/MTP 映射。

此外，模型仓自己的 `inference/vllm_ascend` 目录提供了一套可覆盖镜像内
`vllm_ascend/` 的代码。它包含 NPU 专用 `open_pangu.py`，里面有：

- AscendFusedMoE
- MLA attention
- fused qkv_a projection
- expert/gate_up/down 权重重映射
- profile run 时 MoE force load balance
- NPU prefetch/dispose_tensor 等显存优化

这说明 718B 的实际交付方式仍然带了模型仓侧补丁，而不是只依赖 pip 包。

## 718B 启动方式

718B 文档里 BF16 启动使用 `vllm serve`，每个节点都跑一个 serve 进程，非 0 节点加
`--headless`。

核心参数形态：

```bash
--data-parallel-size $((NUM_NODES * DATA_PARALLEL_SIZE_LOCAL))
--data-parallel-size-local $DATA_PARALLEL_SIZE_LOCAL
--data-parallel-start-rank $((DATA_PARALLEL_SIZE_LOCAL * NODE_RANK))
--data-parallel-address $MASTER_NODE_IP
--data-parallel-rpc-port 13389
--tensor-parallel-size $((NUM_NPUS_LOCAL / DATA_PARALLEL_SIZE_LOCAL))
--enable-expert-parallel
--no-enable-prefix-caching
```

这个模式和我们当前 `DP=4, TP=8` 的脚本是一致的：每台机器是一个或多个本地 DP rank，
每个 DP rank 内做 TP，node0 负责 API server，其余节点 headless。

## 和当前 505B PanguV2MoE 的相同点

可以借鉴的点：

- 用 `vllm.general_plugins` 或 registry 机制注册缺失架构。
- 必须补 packed module mapping，例如：

```text
gate_up_proj -> gate_proj + up_proj
experts -> experts.0.gate_proj + experts.0.up_proj + experts.0.down_proj
fused_qkv_a_proj -> q_a_proj + kv_a_proj_with_mqa
```

- 多机启动使用 `vllm serve + data_parallel_size_local + start_rank + headless`。
- `--enable-expert-parallel` 是 MoE 大模型必需项。
- `--no-enable-prefix-caching` 在 Pangu 大模型初始 bring-up 时更稳。
- profile run 阶段要特别注意 MoE token 分布，否则单 rank 可能出现显存峰值异常。

我们当前适配已经吸收了这些点：

- `plugin.py` 注册 `PanguV2MoEForCausalLM`
- `plugin.py` patch vllm-ascend modelslim/packed mapping
- `run_4node_mixed_node.sh` 使用 DP/TP/headless 启动
- 默认关闭 prefix caching
- 默认打开 KV cache debug

## 和当前 505B PanguV2MoE 的关键差异

718B 是 `pangu_ultra_moe`，而当前模型是 `pangu_v2_moe`。两者不能直接互相套模型类。

当前 505B 多了这些特性：

- DSA：`dsa_layers`、`index_topk`、`index_head_dim`
- SWA：`swa_layers`、`sliding_window_list`
- MoME：`use_mome`、`router_sliding_window`
- MHC：`use_mhc`、`block_post_layernorm_idx`
- PanguV2 hybrid state：attention KV cache 之外还有 MoME/Mamba-like state

因此，vLLM 0.18.0 自带的 `PanguUltraMoEForCausalLM` 虽然已经包含 static sink
attention 和 OpenPangu MoE，但仍不足以完整运行当前 PanguV2MoE 505B。

这也是为什么当前适配没有简单注册到：

```text
vllm.model_executor.models.openpangu:PanguUltraMoEForCausalLM
```

而是注册到随包携带的 omni-npu 实现：

```text
omni_npu.v1.models.pangu.pangu_ultra_moe:PanguUltraMoEForCausalLM
```

该实现里明确包含：

- `NPUDeepseekSparseAttention`，用于 DSA 层
- `NPUDeepseekMLAAttention`，用于普通 MLA 层
- `IsHybrid`
- `MomeSpec` / MoME state 形状估计
- MHC 模块
- PanguV2 的权重加载映射

## 对当前适配的检查清单

参考 718B 适配后，当前 505B 适配至少要确认下面几项。

1. 架构注册是否成功：

```text
Registering PanguV2MoEForCausalLM with bundled omni_npu PanguUltraMoE implementation.
```

2. packed module mapping 是否包含：

```text
gate_up_proj
experts
fused_qkv_a_proj
```

3. 多机启动是否使用新的 DP/TP 方式：

```bash
--distributed-executor-backend mp
--data-parallel-size 4
--data-parallel-size-local 1
--data-parallel-start-rank ${NODE_RANK}
--tensor-parallel-size 8
--headless
```

4. hybrid KV manager 是否开启：

```bash
--no-disable-hybrid-kv-cache-manager
```

5. prefix caching 是否先关闭：

```bash
--no-enable-prefix-caching
```

6. A2/910B 上是否先关闭激进预取：

```json
{
  "enable_prefetch": false,
  "expert_gate_up_prefetch": 0,
  "expert_down_prefetch": 0,
  "dense_mlp_prefetch": 0,
  "lm_head_prefetch": 0,
  "attn_prefetch": 0
}
```

7. KV cache 日志是否能看到分组：

```text
PanguV2 KV group[...]
```

如果启动后仍然在加载权重后 OOM，优先看 profile run 和 KV cache allocation，不要只调
`max_model_len`。718B 适配也说明：Pangu 系 MoE 的显存问题经常来自后端映射、profile
run、KV/hybrid state 分配，而不是单纯序列长度。

## 结论

718B 的经验可以总结为：

- 模型架构要先让 vLLM registry 正确识别。
- Ascend 后端必须知道 packed module 和 MTP/quant mapping。
- 多机 MoE 应按 DP/TP/headless 方式启动。
- Pangu 系 attention/KV cache 不是普通 causal attention，必须保留对应 backend/hybrid
  cache 逻辑。

对当前 505B 来说，718B 的启动方式和 packed mapping 很有参考价值；但模型实现不能直接复用
718B 的 `PanguUltraMoEForCausalLM`，必须继续使用 PanguV2MoE/omni-npu 路径。
