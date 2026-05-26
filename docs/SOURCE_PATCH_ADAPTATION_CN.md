# 通过修改 vLLM / vllm-ascend 源码适配 OpenPangu-505B PanguV2MoE

本文档说明：如果不使用本工程的独立 vLLM plugin，而是直接修改 vLLM 和
vllm-ascend 源码，或者把这些修改做成 patch 打进镜像，需要改哪些文件、
改在什么位置、代码应该怎么写。

下面的行号基于本工程下载的源码：

- `third_party/vllm-0.18.0`
- `third_party/vllm-ascend-0.18.0`
- `third_party/omni-npu-release_1.2.1`

如果你本地源码有额外改动，行号可能会前后偏移，但附近代码结构应一致。

## 与当前适配工程是否一致

结论：本文档已经按当前适配工程的实际方式对齐。

当前工程不是把整个 PanguV2MoE 模型实现硬迁移进 vLLM 原生
`openpangu.py`，而是采用更保守的方式：

1. 让 `omni_npu` 在运行环境中可 import。
2. 在 vLLM 里新增一个很薄的 `pangu_v2_moe.py` 模型入口。
3. 这个入口把 `PanguV2MoEForCausalLM` 映射到 omni-npu 已有的
   `PanguUltraMoEForCausalLM` 实现。
4. vLLM 负责识别架构、修正 config、走 hybrid/MLA 检测。
5. vllm-ascend 负责补齐 packed module mapping、加载 omni-npu 的
   `model_extra_config`，以及临时增加 KV cache 诊断。

也就是说，源码/patch 方式和插件工程的核心逻辑应保持一致：vLLM 只加入口
和注册，PanguV2MoE 的 DSA/SWA/MoME/MHC/hybrid state 逻辑仍复用 omni-npu。

## 目录约定

容器内示例路径如下：

```bash
/workspace/vllm
/workspace/vllm-ascend
/workspace/omni-npu
/data/openpangu-505B
```

本工程对应路径如下：

```bash
/mnt/d/workspace/pangu-v2-moe-vllm-ascend018/third_party/vllm-0.18.0
/mnt/d/workspace/pangu-v2-moe-vllm-ascend018/third_party/vllm-ascend-0.18.0
/mnt/d/workspace/pangu-v2-moe-vllm-ascend018/third_party/omni-npu-release_1.2.1
```

后文用 `/workspace` 举例。

## 步骤 1：安装源码环境

```bash
cd /workspace/vllm
python -m pip install -e .

cd /workspace/vllm-ascend
python -m pip install -e .
```

确认 import 到的是你修改的源码：

```bash
python - <<'PY'
import vllm, vllm_ascend
print("vllm:", vllm.__file__)
print("vllm_ascend:", vllm_ascend.__file__)
PY
```

## 步骤 2：让 omni_npu 可 import

源码/patch 方式也建议复用 omni-npu 的模型实现。不要第一版就把
`pangu_ultra_moe.py`、attention、MoME、MHC、loader 等大量文件拆散迁移到
vLLM 内部。

推荐做法：

```bash
export PYTHONPATH=/workspace/omni-npu/src:$PYTHONPATH
```

或者在镜像构建时安装：

```bash
cd /workspace/omni-npu
python -m pip install -e .
```

运行前确认：

```bash
python - <<'PY'
import omni_npu
from omni_npu.v1.models.pangu.pangu_ultra_moe import PanguUltraMoEForCausalLM
print("omni_npu:", omni_npu.__file__)
print(PanguUltraMoEForCausalLM)
PY
```

## 步骤 3：新增 vLLM 模型入口

新增文件：

```text
vllm/model_executor/models/pangu_v2_moe.py
```

完整代码：

```python
# SPDX-License-Identifier: Apache-2.0

from omni_npu.v1.models.pangu.pangu_ultra_moe import (
    PanguUltraMoEForCausalLM as _OmniPanguUltraMoEForCausalLM,
)


class PanguV2MoEForCausalLM(_OmniPanguUltraMoEForCausalLM):
    """OpenPangu-505B PanguV2MoE entrypoint.

    The implementation is provided by omni-npu. vLLM only needs a stable
    architecture name so ModelRegistry can resolve the model.
    """

    pass


PanguV2aMoEForCausalLM = PanguV2MoEForCausalLM

__all__ = ["PanguV2MoEForCausalLM", "PanguV2aMoEForCausalLM"]
```

这一步和插件工程一致：插件里也是把 `PanguV2MoEForCausalLM` 注册到
omni-npu 的 `PanguUltraMoEForCausalLM` 实现。

## 步骤 4：注册模型架构

修改文件：

```text
vllm/model_executor/models/registry.py
```

在本工程的 vLLM 0.18.0 源码中，Pangu 相关注册位于第 181 到 183 行：

```python
"PanguEmbeddedForCausalLM": ("openpangu", "PanguEmbeddedForCausalLM"),
"PanguProMoEV2ForCausalLM": ("openpangu", "PanguProMoEV2ForCausalLM"),
"PanguUltraMoEForCausalLM": ("openpangu", "PanguUltraMoEForCausalLM"),
```

改成下面这样，新增两行即可：

```python
"PanguEmbeddedForCausalLM": ("openpangu", "PanguEmbeddedForCausalLM"),
"PanguProMoEV2ForCausalLM": ("openpangu", "PanguProMoEV2ForCausalLM"),
"PanguUltraMoEForCausalLM": ("openpangu", "PanguUltraMoEForCausalLM"),
"PanguV2MoEForCausalLM": ("pangu_v2_moe", "PanguV2MoEForCausalLM"),
"PanguV2aMoEForCausalLM": ("pangu_v2_moe", "PanguV2MoEForCausalLM"),
```

说明：

- `PanguV2MoEForCausalLM` 是正确架构名。
- `PanguV2aMoEForCausalLM` 只用于兼容之前写错的 config，长期建议修正模型目录里的 `config.json`。
- vLLM 0.18.0 在第 1262 到 1268 行把这里的 `mod_relname` 拼成
  `vllm.model_executor.models.<mod_relname>`，所以前一步必须新增
  `pangu_v2_moe.py`。

## 步骤 5：修正 vLLM 对 PanguV2MoE config 的识别

修改文件：

```text
vllm/config/model.py
```

### 5.1 增加 config 归一化函数

在 `class ModelConfig` 之前增加。当前源码中 `class ModelConfig` 在第 101 行附近，
建议把下面代码放在 class 定义之前。

完整代码：

```python
_PANGU_V2_MOE_ARCH = "PanguV2MoEForCausalLM"
_PANGU_V2_MOE_ARCH_ALIASES = {"PanguV2MoEForCausalLM", "PanguV2aMoEForCausalLM"}
_PANGU_V2_MOE_MODEL_TYPE = "pangu_v2_moe"


def _is_pangu_v2_moe_config(config: PretrainedConfig | None) -> bool:
    if config is None:
        return False
    model_type = getattr(config, "model_type", None)
    architectures = set(getattr(config, "architectures", []) or [])
    return model_type == _PANGU_V2_MOE_MODEL_TYPE or bool(
        architectures & _PANGU_V2_MOE_ARCH_ALIASES
    )


def _normalize_pangu_v2_moe_config(config: PretrainedConfig | None) -> None:
    if not _is_pangu_v2_moe_config(config):
        return

    config.architectures = [_PANGU_V2_MOE_ARCH]

    rope_theta = getattr(config, "rope_theta", 10000)
    rope_parameters = getattr(config, "rope_parameters", None)
    if not isinstance(rope_parameters, dict):
        rope_parameters = {"rope_type": "default"}
    rope_parameters.setdefault("rope_type", "default")
    rope_parameters["rope_theta"] = rope_theta
    config.rope_parameters = rope_parameters

    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = getattr(config, "eos_token_id", None)

    if getattr(config, "use_mome", False) and getattr(config, "layer_types", None) is None:
        num_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
        config.layer_types = ["attention"] * num_layers + ["mome"]
```

### 5.2 在 `ModelConfig.__post_init__()` 中调用

当前源码中：

- `ModelConfig.__post_init__` 在第 416 行开始。
- `self.hf_config = hf_config` 在第 490 行。
- `self.hf_text_config = get_hf_text_config(self.hf_config)` 在第 493 行。

原代码：

```python
self.hf_config = hf_config
if dict_overrides:
    self._apply_dict_overrides(hf_config, dict_overrides)
self.hf_text_config = get_hf_text_config(self.hf_config)
```

替换为完整代码：

```python
self.hf_config = hf_config
if dict_overrides:
    self._apply_dict_overrides(hf_config, dict_overrides)
_normalize_pangu_v2_moe_config(self.hf_config)
self.hf_text_config = get_hf_text_config(self.hf_config)
_normalize_pangu_v2_moe_config(self.hf_text_config)
```

这一步对应插件工程中的 `ModelConfig.__post_init__` monkey patch，目的是保证即使
模型原始 `config.json` 不完整，vLLM 内部也能看到：

- 正确的 `architectures`
- `rope_parameters`
- `pad_token_id`
- 用于 hybrid 检测的 `layer_types`

## 步骤 6：让 vLLM 把 pangu_v2_moe 识别为 MLA 模型

修改文件：

```text
vllm/transformers_utils/model_arch_config_convertor.py
```

当前源码中 `ModelArchConfigConvertorBase.is_deepseek_mla()` 在第 212 行开始，
Pangu Ultra 相关项在第 227 到 228 行：

```python
"pangu_ultra_moe",
"pangu_ultra_moe_mtp",
```

改成：

```python
"pangu_ultra_moe",
"pangu_ultra_moe_mtp",
"pangu_v2_moe",
```

完整上下文应类似：

```python
elif self.hf_text_config.model_type in (
    "AXK1",
    "deepseek_v2",
    "deepseek_v3",
    "deepseek_mtp",
    "deepseek_v31",
    "ernie",
    "ernie_mtp",
    "glm4_moe",
    "glm_ocr_moe",
    "glm_ocr_mtp",
    "kimi_k2",
    "kimi_linear",
    "longcat_flash",
    "pangu_ultra_moe",
    "pangu_ultra_moe_mtp",
    "pangu_v2_moe",
    "bailing_hybrid",
):
    return self.hf_text_config.kv_lora_rank is not None
```

这一步对应插件工程里的 `is_deepseek_mla` patch。你的 505B config 中存在
`kv_lora_rank`、`q_lora_rank`、`qk_nope_head_dim`、`qk_rope_head_dim` 等 MLA
字段，如果这里没识别，后续 attention/KV spec 容易走错。

## 步骤 7：补 vllm-ascend packed module mapping

修改文件：

```text
vllm_ascend/quantization/modelslim_config.py
```

当前 vllm-ascend 0.18.0 中：

- `packed_modules_model_mapping` 在第 52 行开始。
- `pangu_ultra_moe` mapping 在第 88 到 92 行。

在 `pangu_ultra_moe` 后面新增 `pangu_v2_moe`。完整片段如下：

```python
"pangu_ultra_moe": {
    "gate_up_proj": ["gate_proj", "up_proj"],
    "experts": ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
    "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
},
"pangu_v2_moe": {
    "gate_up_proj": ["gate_proj", "up_proj"],
    "experts": ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
    "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
},
```

注意：vllm-ascend 0.18.0 里重点是 `modelslim_config.py`。有些早期讨论里会提到
`quant_config.py`，但当前 0.18.0 源码没有这个同名修改点，不要照搬旧文件名。

这一步参考了 718B 适配在 vllm-ascend 中给 `pangu_ultra_moe` 增加 mapping 的方式，
只是这里模型类型换成 `pangu_v2_moe`。

## 步骤 8：让 vllm-ascend worker 加载 omni model_extra_config

修改文件：

```text
vllm_ascend/worker/worker.py
```

当前源码中：

- `_init_device()` 在第 249 行附近。
- `init_device()` 在第 309 行开始。
- `self.device = self._init_device()` 在第 313 行。
- `init_workspace_manager(self.device, num_ubatches)` 在第 316 行。

在 `_init_device()` 后、`init_device()` 前增加一个方法，然后在 `init_device()` 里调用。

完整新增方法：

```python
def _load_pangu_v2_model_extra_config(self) -> None:
    hf_config = getattr(self.model_config, "hf_config", None)
    if getattr(hf_config, "model_type", None) != "pangu_v2_moe":
        return

    try:
        from omni_npu.model_config.config_loader.loader import load_model_extra_config

        load_model_extra_config(
            self.model_config,
            self.vllm_config,
            self.scheduler_config,
        )
        logger.info("Loaded omni_npu model_extra_config for PanguV2MoE.")
    except Exception:
        logger.exception("Failed to load omni_npu model_extra_config for PanguV2MoE.")
        raise
```

然后把 `init_device()` 开头从：

```python
def init_device(self):
    # NOTE: KEEP device the member of `NPUWorker`, as it will be checked
    # in ray scenario. see https://github.com/vllm-project/vllm/pull/26845
    # for more details
    self.device = self._init_device()
    # Initialize workspace manager
    num_ubatches = 1
    init_workspace_manager(self.device, num_ubatches)
```

改成：

```python
def init_device(self):
    # NOTE: KEEP device the member of `NPUWorker`, as it will be checked
    # in ray scenario. see https://github.com/vllm-project/vllm/pull/26845
    # for more details
    self.device = self._init_device()
    self._load_pangu_v2_model_extra_config()
    # Initialize workspace manager
    num_ubatches = 1
    init_workspace_manager(self.device, num_ubatches)
```

这一步非常关键。omni-npu 的 PanguV2MoE 路径依赖
`CUSTOM_MODEL_CONFIG_PATH` 指向的 extra config。如果只改 vLLM registry，但没有执行
`load_model_extra_config()`，就很容易出现“权重加载成功，加载后 KV cache/profile 阶段
OOM”的现象。

## 步骤 9：增加 KV cache 诊断日志

这是临时诊断 patch，建议先加上，跑通后可以移除。

修改文件：

```text
vllm_ascend/worker/model_runner_v1.py
```

当前源码中：

- `NPUModelRunner.initialize_kv_cache()` 在第 2608 行附近。
- `kv_cache_config = deepcopy(kv_cache_config)` 在第 2611 行附近。
- `self.kv_cache_config = kv_cache_config` 在第 2613 行附近。

在 `initialize_kv_cache()` 一开始增加日志。完整函数开头应改成：

```python
def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
    groups = getattr(kv_cache_config, "kv_cache_groups", []) or []
    for idx, group in enumerate(groups):
        spec = getattr(group, "kv_cache_spec", None)
        layer_names = getattr(group, "layer_names", []) or []
        logger.warning(
            "PanguV2 KV group[%d]: spec=%s layers=%d block_size=%s page_size_bytes=%s",
            idx,
            type(spec).__name__ if spec is not None else None,
            len(layer_names),
            getattr(spec, "block_size", None),
            getattr(spec, "page_size_bytes", None),
        )

    kv_cache_config = deepcopy(kv_cache_config)
    self.kv_cache_config = kv_cache_config
    self._mamba_copy_bufs = None
```

启动后至少要看到这些 KV group 日志。重点检查：

- group 数量是否符合 hybrid 预期。
- 是否所有层都被错误合并成一个普通 attention group。
- 是否存在 `MambaSpec` 或 omni-npu 的 MoME/hybrid spec。
- `block_size` 是否为你启动命令里的 `128`。
- `page_size_bytes` 是否异常巨大。

如果你设置极短序列、极小 `max-num-seqs`、降低 `gpu-memory-utilization` 仍然 OOM，
而日志显示 KV group/spec 不对，那么问题通常不是请求长度，而是模型 spec/hybrid state
被识别错了。

## 步骤 10：增加 910B/A2 的 omni-npu extra config

新增文件：

```text
omni_npu/model_config/configs/low_latency/openpangu_v2/pangu_v2_moe_bf16_a2_dp4tp8_hybrid.json
```

完整内容：

```json
{
  "model_parallel_config": {
    "ena_seq_parallel": false,
    "ena_context_parallel": false,
    "enable_flashcomm2": false,
    "enable_aicpu_dp_sync": false,
    "layer_parallel_config": {}
  },
  "operator_optimization_config": {
    "moe_comm_strategy": "dispatch_combine",
    "use_aicpu_fa_tiling": true,
    "use_noncontiguous_kv": true,
    "use_mome_inplace_update": true,
    "use_rope_fusion_op": true,
    "use_mhc_fusion_op": false,
    "enable_multi_stream": false,
    "split_q_up_in_multistream": false,
    "disable_npu_top_k_top_p_sample": true,
    "enable_prefetch": false,
    "expert_gate_up_prefetch": 0,
    "expert_down_prefetch": 0,
    "dense_mlp_prefetch": 0,
    "lm_head_prefetch": 0,
    "attn_prefetch": 0,
    "shared_expert_gate_up_prefetch": 0,
    "shared_expert_down_prefetch": 0
  }
}
```

启动时设置：

```bash
export CUSTOM_MODEL_CONFIG_PATH=low_latency/openpangu_v2/pangu_v2_moe_bf16_a2_dp4tp8_hybrid.json
```

这一步对应当前工程里的：

```text
src/omni_npu/model_config/configs/low_latency/openpangu_v2/pangu_v2_moe_bf16_a2_dp4tp8_hybrid.json
```

这些 prefetch/multistream/fused 选项先关闭，是为了先在 910B/A2 上把模型加载、
profile、KV cache 初始化跑通。跑通后再逐项恢复性能优化。

## 步骤 11：修正模型目录 config.json

即使已经在 vLLM 源码中做了归一化，也建议把模型目录里的 `config.json` 改正确，
避免多机多进程中某些辅助逻辑读到原始错误值。

```bash
python - <<'PY'
import json
from pathlib import Path

model_dir = Path("/data/openpangu-505B")
path = model_dir / "config.json"
cfg = json.loads(path.read_text())

cfg["architectures"] = ["PanguV2MoEForCausalLM"]

rope_theta = cfg.get("rope_theta", 10000)
rope_parameters = cfg.get("rope_parameters") or {"rope_type": "default"}
rope_parameters.setdefault("rope_type", "default")
rope_parameters["rope_theta"] = rope_theta
cfg["rope_parameters"] = rope_parameters

if cfg.get("pad_token_id") is None:
    cfg["pad_token_id"] = cfg.get("eos_token_id")

if cfg.get("use_mome") and cfg.get("layer_types") is None:
    n = int(cfg.get("num_hidden_layers", 0))
    cfg["layer_types"] = ["attention"] * n + ["mome"]

path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
print("updated", path)
PY
```

确认：

```bash
grep -n '"architectures"' -A3 /data/openpangu-505B/config.json
grep -n '"model_type"' /data/openpangu-505B/config.json
grep -n '"layer_types"' /data/openpangu-505B/config.json
```

## 步骤 12：4 机 DP4TP8 启动命令

每台机器运行一个 `vllm serve`，每台 8 张卡，`DP=4, TP=8`。

公共环境变量：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_IF_IP=${NODE_IP}
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_MLAPO=0
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export CUSTOM_MODEL_CONFIG_PATH=low_latency/openpangu_v2/pangu_v2_moe_bf16_a2_dp4tp8_hybrid.json
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/omni-npu/src:$PYTHONPATH
```

node0：

```bash
vllm serve /data/openpangu-505B \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --data-parallel-size 4 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 0 \
  --data-parallel-address ${MASTER_IP} \
  --data-parallel-rpc-port 13389 \
  --tensor-parallel-size 8 \
  --distributed-executor-backend mp \
  --enable-expert-parallel \
  --no-disable-hybrid-kv-cache-manager \
  --no-enable-prefix-caching \
  --block-size 128 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 128 \
  --gpu-memory-utilization 0.75 \
  --enforce-eager \
  --api-server-count 4
```

node1 到 node3：

```bash
vllm serve /data/openpangu-505B \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --data-parallel-size 4 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank ${NODE_RANK} \
  --data-parallel-address ${MASTER_IP} \
  --data-parallel-rpc-port 13389 \
  --tensor-parallel-size 8 \
  --distributed-executor-backend mp \
  --enable-expert-parallel \
  --no-disable-hybrid-kv-cache-manager \
  --no-enable-prefix-caching \
  --block-size 128 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 128 \
  --gpu-memory-utilization 0.75 \
  --enforce-eager \
  --headless
```

`NODE_RANK` 分别是 `1`、`2`、`3`。

这个命令形态和当前工程的 `scripts/run_4node_mixed_node.sh` 一致，并且与
vllm-ascend 0.18.0 官方文档中多机 MP/DP 的主流方式一致：每节点一个
`vllm serve`，非主节点加 `--headless`，用
`--data-parallel-start-rank` 区分节点。

## 步骤 13：验证注册和 patch 是否生效

```bash
python - <<'PY'
from vllm.model_executor.models import ModelRegistry
print("PanguV2MoEForCausalLM" in ModelRegistry.get_supported_archs())
PY
```

期望输出：

```text
True
```

再验证 config：

```bash
python - <<'PY'
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("/data/openpangu-505B", trust_remote_code=True)
print(cfg.model_type)
print(cfg.architectures)
print(getattr(cfg, "rope_parameters", None))
print(getattr(cfg, "layer_types", None) is not None)
PY
```

启动日志里必须看到：

- `Loaded omni_npu model_extra_config for PanguV2MoE.`
- `PanguV2 KV group[...]`

如果没有第一条，说明 vllm-ascend worker 没执行 extra config 加载。
如果没有第二条，说明还没进入 KV 初始化或诊断 patch 没生效。

## 步骤 14：生成 patch

建议拆成多个 patch，便于升级 vLLM/vllm-ascend 时定位冲突：

```text
0001-vllm-register-pangu-v2-moe.patch
0002-vllm-add-pangu-v2-moe-entrypoint.patch
0003-vllm-normalize-pangu-v2-moe-config.patch
0004-vllm-detect-pangu-v2-moe-mla.patch
0005-vllm-ascend-add-pangu-v2-packed-mapping.patch
0006-vllm-ascend-load-pangu-v2-extra-config.patch
0007-vllm-ascend-add-kv-debug.patch
0008-omni-npu-add-a2-dp4tp8-extra-config.patch
```

生成方式：

```bash
cd /workspace/vllm
git diff > /workspace/patches/vllm-pangu-v2-moe.patch

cd /workspace/vllm-ascend
git diff > /workspace/patches/vllm-ascend-pangu-v2-moe.patch

cd /workspace/omni-npu
git diff > /workspace/patches/omni-npu-pangu-v2-moe-a2-config.patch
```

应用方式：

```bash
cd /workspace/vllm
git apply --check /workspace/patches/vllm-pangu-v2-moe.patch
git apply /workspace/patches/vllm-pangu-v2-moe.patch

cd /workspace/vllm-ascend
git apply --check /workspace/patches/vllm-ascend-pangu-v2-moe.patch
git apply /workspace/patches/vllm-ascend-pangu-v2-moe.patch

cd /workspace/omni-npu
git apply --check /workspace/patches/omni-npu-pangu-v2-moe-a2-config.patch
git apply /workspace/patches/omni-npu-pangu-v2-moe-a2-config.patch
```

应用后重新安装：

```bash
cd /workspace/vllm
python -m pip install -e .

cd /workspace/vllm-ascend
python -m pip install -e .
```

## 常见失败点

### 1. vLLM 找不到架构

错误类似：

```text
Model architectures ['PanguV2MoEForCausalLM'] are not supported
```

检查：

- `vllm/model_executor/models/registry.py` 是否加了注册。
- `vllm/model_executor/models/pangu_v2_moe.py` 是否存在。
- 当前 Python import 的 vLLM 是否是你修改的源码。

### 2. 权重加载成功，加载后 OOM

优先检查：

- 是否真正 import 到 omni-npu 的模型实现。
- `load_model_extra_config()` 是否执行。
- `CUSTOM_MODEL_CONFIG_PATH` 是否指向 A2/910B 配置。
- prefetch/multistream/fused 配置是否先关闭。
- `--no-disable-hybrid-kv-cache-manager` 是否存在。
- KV group 日志是否符合 hybrid 预期。

如果序列长度、`max-num-seqs`、`gpu-memory-utilization` 怎么调都 OOM，尤其是 D 节点
在加载权重后、KV cache 分配时 OOM，通常不是普通 KV 容量问题，而是：

- hybrid/MoME spec 没识别。
- PanguV2MoE 走成了普通 OpenPangu fallback。
- omni-npu extra config 没加载，仍在走 A3 或默认优化路径。
- vllm-ascend packed module/MLA mapping 不完整，导致 profile 或 cache spec 异常。

### 3. 多机行为不一致

四台机器必须源码、patch、环境变量完全一致。每台都检查：

```bash
python - <<'PY'
import vllm, vllm_ascend, omni_npu
print(vllm.__file__)
print(vllm_ascend.__file__)
print(omni_npu.__file__)
PY
```

不要只在 node0 改源码。headless 节点也会加载模型、初始化 worker、分配 KV cache。

## 最低必需改动总结

源码/patch 方式至少需要这些改动：

1. vLLM 新增 `vllm/model_executor/models/pangu_v2_moe.py`。
2. vLLM `registry.py` 注册 `PanguV2MoEForCausalLM`。
3. vLLM `model.py` 归一化 PanguV2MoE config。
4. vLLM `model_arch_config_convertor.py` 把 `pangu_v2_moe` 识别为 MLA。
5. vllm-ascend `modelslim_config.py` 增加 `pangu_v2_moe` packed mapping。
6. vllm-ascend `worker.py` 调用 omni-npu 的 `load_model_extra_config()`。
7. omni-npu 增加 A2/910B 的保守 extra config。
8. 启动使用 DP4TP8 MP/headless 方式，并保留 hybrid KV manager。

先以“能完成 KV cache 初始化”为第一目标。跑通后再逐项恢复 prefetch、graph、
fused、multistream 等性能优化。
