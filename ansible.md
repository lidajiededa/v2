# 克隆 openPangu-2.0-Infer 仓库

> 把这个代码仓clone下来 https://gitcode.com/ascend-tribe/openPangu-2.0-Infer/

<details><summary>5 previous messages</summary>

> 我先看一下当前目录里有没有同名文件夹，然后把仓库克隆到工作区下面。
>
> 当前工作区看起来是空的，`git` 也可用。我现在直接从 GitCode 拉取这个仓库。
>
> 第一次连接 GitCode 的 443 端口时 SSL 握手断了，看起来像瞬时网络问题。我再用原始 URL 重试一次，并顺手检查一下站点是否能连通。
>
> 网页本身能访问；`git clone` 已经开始重定向到 `.git` 地址，现在还在传输中。我等它跑完再确认目录状态。
>
> 克隆命令这次完成了，我检查一下仓库内容和当前分支，确认不是空目录。
</details>

克隆好了，仓库在：

[openPangu-2.0-Infer](./openPangu-2.0-Infer)

当前分支是 `main`，并且已经跟踪 `origin/main`。第一次 SSL 连接断了一下，重试后成功完成。

> 模型名称注册的代码在哪里

<details><summary>4 previous messages</summary>

> 我来在仓库里搜一下“模型名称/注册”相关的入口，先从 README 里提到的 model name、registry、register 这些关键词顺着找。
>
> 搜出来有两类“名称”：一类是服务对外暴露的 `served-model-name`，另一类是 vLLM 里按 `config.json` 的 `architectures` 注册模型类。你问“注册代码”更像第二类，我把相关文件打开确认一下。
>
> 另外还找到了 HuggingFace `model_type` 的 `AutoConfig.register(...)`，以及补丁系统把 `openpangu_v2` 映射到对应 patch 目录的地方；这两个也算“注册链路”的一部分。
>
> 我已经把注册链路定位出来了。最后再取一下精确行号，方便你直接跳过去。
</details>

在这个仓里，“模型名称注册”主要有几处，看你指的是哪种名称：

1. **vLLM 模型类注册**
   [components/omni-npu/src/omni_npu/v1/models/__init__.py](./openPangu-2.0-Infer/components/omni-npu/src/omni_npu/v1/models/__init__.py:7)

   这里的 `register_models()` 里注册了：
   - `AutoConfig.register("openpangu_v2", OpenPanguV2Config)`
   - `ModelRegistry.register_model("OpenPanguV2ForCausalLM", ...)`
   - `ModelRegistry.register_model("OpenPanguV2MTPModel", ...)`

2. **按 `architectures` 做配置映射**
   [components/omni-npu/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/patch_models_config.py](./openPangu-2.0-Infer/components/omni-npu/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/patch_models_config.py:121)

   这里往 `MODELS_CONFIG_MAP` 里加了：
   - `PanguUltraMoEForCausalLM`
   - `OpenPanguV2VLForConditionalGeneration`
   - `OpenPanguUltraOmniForConditionalGeneration`
   - `OpenPanguV2ForCausalLM`

3. **按 `model_type` 选择 patch 目录**
   [components/omni-npu/src/omni_npu/vllm_patches/__init__.py](./openPangu-2.0-Infer/components/omni-npu/src/omni_npu/vllm_patches/__init__.py:111)

   例如 `openpangu_v2` 会映射到：
   `pangu_v2_base,pangu_sink_swa_mla`

4. **对外服务名 `served-model-name`**
   [tools/scripts/pd_run.sh](./openPangu-2.0-Infer/tools/scripts/pd_run.sh:38)

   默认是：
   `SERVED_MODEL_NAME="openPangu-2.0-Flash"`

如果你说的是模型 `config.json` 里 `architectures` 对应的注册，重点看第 1 和第 2 个文件。

> 切换分支为a2分支

<details><summary>6 previous messages</summary>

> 我先看一下仓库当前分支和工作区状态，确认没有未提交改动会被切分支影响，然后切到 `a2`。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `git branch --all --list '*a2*'`
> - Ran `git status --short --branch`
>
> </details>
>
> 本地目前在 `main`，工作区干净；本地还没有看到叫 `a2` 的分支。我拉一下远端分支列表再找。
>
> <details><summary>Ran a command</summary>
>
> - Ran `git fetch origin --prune && git branch --all --list '*a2*'`
>
> </details>
>
> 远端只有一个匹配 `a2` 的分支：`origin/92B_A2`。我按这个切出本地跟踪分支。
>
> <details><summary>Ran a command</summary>
>
> - Ran `git switch --track origin/92B_A2 && git status --short --branch`
>
> </details>
</details>

已切换到 `a2` 对应的远端分支：

`92B_A2`

当前状态是：`92B_A2...origin/92B_A2`，工作区干净。

> --kv-cache-dtype li_int8_ds_mla --enable-lopt --lopt-pool-size 16 --lopt-chunk-size 4096 这些配置是什么作用

<details><summary>8 previous messages</summary>

> 我查一下这几个参数在当前分支里的定义和使用位置，这样解释能贴着代码说，而不是只按参数名猜。
>
> <details><summary>Explored a search, ran a command</summary>
>
> - Ran `git status --short --branch`
>
> <details><summary>Explored a search</summary>
>
> - Searched for `kv-cache-dtype|li_int8_ds_mla|enable-lopt|lopt-pool-size|lopt-chunk-size|lopt` in `.`
>
> </details>
>
> </details>
>
> 关键词搜索还在跑，仓库比较大。我先等完整结果，然后会把参数对应到实际代码路径。
>
> 结果显示参数主要分两块：`kv-cache-dtype` 走 NPU attention/KV cache 量化路径，`lopt` 是 tokenizer 侧的并行优化。我打开关键实现看细节。
>
> <details><summary>Explored 4 files</summary>
>
> - Read `./components/omni-npu/src/omni_npu/vllm_patches/patches/common/patch_lopt.py`
> - Read `./components/omni-npu/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/patch_kv_cache_interface.py`
> - Read `./components/omni-npu/src/omni_npu/attention/backends/dsa.py`
> - Read `./components/omni-npu/src/omni_npu/lopt/lopt_wrapper.py`
>
> </details>
>
> KV cache 这块还涉及实际 attention kernel 调用，我再看 `npu_pangu.py` 里对 `li_int8_ds_mla` 的分支，确认它到底量化了哪部分。
>
> <details><summary>Explored 2 files, a search</summary>
>
> - Searched for `li_int8_ds_mla|int8_ds_mla|cache_dtype` in `npu_pangu.py`
> - Read `./components/omni-npu/src/omni_npu/v1/layers/attention/npu_pangu.py`
> - Read `./components/omni-npu/src/omni_npu/vllm_patches/patches/models/pangu_v2_moe/patch_kv_cache_dtype.py`
>
> </details>
>
> 代码注释里把 `li_int8_ds_mla` 写成 “Li-Quant-Only”，并且实现上只把 DSA indexer/cache 的一部分走 int8。下面我按“做什么、收益、代价/注意事项”给你串起来。
</details>

这几个参数分两类：

`--kv-cache-dtype li_int8_ds_mla`

这是 **KV cache 的自定义量化格式**，用于 Pangu V2 的 DSA/MLA attention 场景。代码里把 `li_int8_ds_mla` 扩展进了 vLLM 支持的 `cache_dtype` 列表：

[patch_kv_cache_dtype.py](./openPangu-2.0-Infer/components/omni-npu/src/omni_npu/vllm_patches/patches/models/pangu_v2_moe/patch_kv_cache_dtype.py:19)

从实现看，`li_int8_ds_mla` 的 KV cache 布局是：

- 主 KV 部分：`576` 维，`bfloat16`
- indexer key 部分：`128` 维，`int8`
- scale：`1` 个，`float16`

对应代码在：

[patch_kv_cache_interface.py](./openPangu-2.0-Infer/components/omni-npu/src/omni_npu/vllm_patches/patches/models/pangu_v2_hybrid/patch_kv_cache_interface.py:57)

[dsa.py](./openPangu-2.0-Infer/components/omni-npu/src/omni_npu/attention/backends/dsa.py:88)

实际 attention 里会对 query/key 做动态量化，然后走 `npu_ai_infra_quant_lightning_indexer` 这类 NPU 自定义算子：

[npu_pangu.py](./openPangu-2.0-Infer/components/omni-npu/src/omni_npu/v1/layers/attention/npu_pangu.py:281)

简单说：它是 **DSA/MLA KV cache 的轻量 int8 量化模式**，主要目的是减少 KV cache 显存/内存占用，并让 indexer 相关计算走量化算子。代价是会引入量化/反量化路径，效果依赖对应 NPU kernel 支持。

`--enable-lopt`

开启 **LoPT: Lossless Parallel Tokenizer**，也就是长文本并行分词优化。它不是模型计算优化，而是 OpenAI server 收到长 prompt 后，在 tokenize 阶段用并行 tokenizer 加速。

代码里参数说明写得很直接：

[patch_lopt.py](./openPangu-2.0-Infer/components/omni-npu/src/omni_npu/vllm_patches/patches/common/patch_lopt.py:89)

开启后，`OpenAIServing` 会用 `lopt_tokenizer` 替代普通 tokenizer 来处理 prompt：

[patch_lopt.py](./openPangu-2.0-Infer/components/omni-npu/src/omni_npu/vllm_patches/patches/common/patch_lopt.py:201)

`--lopt-pool-size 16`

LoPT 使用的并行进程数。这里是开 `16` 个 tokenizer worker 进程：

[lopt_wrapper.py](./openPangu-2.0-Infer/components/omni-npu/src/omni_npu/lopt/lopt_wrapper.py:52)

进程数越大，长文本分词吞吐可能越高，但 CPU 占用和内存也更高。

`--lopt-chunk-size 4096`

LoPT 切分文本的 chunk 大小，单位是 **字符数**。默认这个配置下，每块 4096 字符，代码里 overlap 是 `chunk_size * 0.125`，所以重叠区大约是 512 字符：

[lopt_wrapper.py](./openPangu-2.0-Infer/components/omni-npu/src/omni_npu/lopt/lopt_wrapper.py:35)

注意：文本长度小于 `chunk_size * 2` 时，不走并行，直接用普通 tokenizer。所以 `4096` 下，少于大约 `8192` 字符的 prompt 基本不会触发 LoPT 并行路径。

整体看，这组参数是给长上下文推理用的：

- `li_int8_ds_mla`：省 KV cache，优化 DSA/MLA cache 格式。
- `enable-lopt`：加速长 prompt 的分词。
- `pool-size 16`：用 16 个进程并行分词。
- `chunk-size 4096`：每 4096 字符切一块，适合较长输入。

> 我启动时92B w8a8要两条命令 ansible-playbook -i omni_infer_inventory_used_for_1P1D.yml omni_infer_server_template_performance1P1D_92B_w8a8_A2_open.yml --tags run_docker 以及 ansible-playbook -i omni_infer_inventory_used_for_1P1D.yml omni_infer_server_template_performance1P1D_92B_w8a8_A2_open.yml --tags run_server,run_proxy，请帮我脱离ansible，拆解出启动步骤

<details><summary>6 previous messages</summary>

> 我来把这两个 playbook 的 `run_docker`、`run_server`、`run_proxy` 三段任务拆开读一遍，再还原成手工命令。这里变量和模板比较多，我会先定位 inventory 里的主机/环境变量，再展开容器启动、服务启动、proxy 启动三块。
>
> <details><summary>Explored 2 files, a search</summary>
>
> - Read `./tools/ansible/template/omni_infer_server_template_performance1P1D_92B_w8a8_A2_open.yml`
> - Searched for `run_docker|run_server|run_proxy|docker run|EXTRA_ARGS|CUSTOM_MODEL_CONFIG_PATH|omni_proxy|api_server|start_api_servers` in `omni_infer_server_template_performance1P1D_92B_w8a8_A2_open.yml`
> - Read `./tools/ansible/template/omni_infer_inventory_used_for_1P1D.yml`
>
> </details>
>
> 前半段已经看到：Ansible 其实是把几段 bash 写到 `$SCRIPTS_PATH`，再 `docker exec` 进去执行。现在我继续看任务部分，确认它为 P/D/C 三类容器分别传了哪些环境变量。
>
> <details><summary>Explored 2 files</summary>
>
> - Read `./tools/ansible/template/omni_infer_server_template_performance1P1D_92B_w8a8_A2_open.yml`
> - Read `./tools/ansible/template/omni_infer_server_template_performance1P1D_92B_w8a8_A2_open.yml`
>
> </details>
>
> 关键执行命令已经完整了；还差 Ansible 生成的几个列表变量，比如 prefill/decode endpoint、pod 数、server offset。我把这一段变量计算也展开，这样手工版不会漏。
>
> <details><summary>Explored a file</summary>
>
> - Read `./tools/ansible/template/omni_infer_server_template_performance1P1D_92B_w8a8_A2_open.yml`
>
> </details>
</details>

可以。这个 playbook 本质上拆成三件事：起 3 个容器、在 P/D 容器里起 vLLM、在 C 容器里起 proxy。

下面按默认 `1P1D` inventory 展开，变量里这些你要替换成真实值：

```bash
export LOG_PATH=/path/to/server/log
export MODEL_PATH=/path/to/model/weights
export DOCKER_IMAGE_ID=image_name:image_tag
export SCRIPTS_PATH=/tmp/scripts_path
export KV_CONNECTOR=LLMDataDistConnector

export DOCKER_NAME_P=docker_p_p0
export DOCKER_NAME_D=docker_d_d0
export DOCKER_NAME_C=docker_c_c0
```

**1. 起容器，对应 `--tags run_docker`**
```bash
mkdir -p "$LOG_PATH"/{p0,d0,c0} "$SCRIPTS_PATH"

docker rm -f "$DOCKER_NAME_P" "$DOCKER_NAME_D" "$DOCKER_NAME_C" 2>/dev/null || true

COMMON_DOCKER_ARGS="--shm-size=500g \
  -e LOG_PATH=$LOG_PATH \
  -e PYTHONHASHSEED=1234 \
  --net=host \
  --privileged=true \
  -u root \
  -w /data \
  --device=/dev/davinci_manager \
  --device=/dev/hisi_hdc \
  --device=/dev/devmm_svm \
  --entrypoint=bash \
  -v /data:/data \
  -v /mnt:/mnt \
  -v /tmp:/tmp \
  -v /home:/home \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
  -v $LOG_PATH:$LOG_PATH \
  -v $MODEL_PATH:$MODEL_PATH \
  -v $SCRIPTS_PATH:$SCRIPTS_PATH \
  -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime"

docker run -it -d --name "$DOCKER_NAME_P" $COMMON_DOCKER_ARGS "$DOCKER_IMAGE_ID"
docker run -it -d --name "$DOCKER_NAME_D" $COMMON_DOCKER_ARGS "$DOCKER_IMAGE_ID"
docker run -it -d --name "$DOCKER_NAME_C" $COMMON_DOCKER_ARGS -e PROXY_NODE_PORT=7000 "$DOCKER_IMAGE_ID"
```

**2. 起 Prefill，对应 P 节点 `run_server`**
```bash
docker exec -d \
  -e MODEL_PATH="$MODEL_PATH" \
  -e MODEL_LEN_MAX_PREFILL=524288 \
  -e PREFILL_SERVER_LIST="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" \
  -e PREFILL_TENSOR_PARALLEL_SIZE=16 \
  -e DECODE_TENSOR_PARALLEL_SIZE=1 \
  -e HOST_IP=127.0.0.1 \
  -e MASTER_PORT=8000 \
  -e API_PORT=9000 \
  -e SERVER_IP_LIST=127.0.0.2 \
  -e PREFILL_POD_NUM=1 \
  -e SOCKET_IFNAME="$(ip -4 route list 0/0 | awk '{print $5}' | head -1)" \
  -e KV_RANK=0 \
  -e NNODES=1 \
  -e NODE_RANK=0 \
  -e NODE_IP_LIST=127.0.0.1 \
  -e KV_CONNECTOR="$KV_CONNECTOR" \
  "$DOCKER_NAME_P" bash -lc '
. ~/.bashrc
export HCCL_BUFFSIZE=100
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export HCCL_OP_RETRY_ENABLE="L0:0,L1:0,L2:0"
export HCCL_OP_EXPANSION_MODE="AIV"
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export OMNI_REUSE_PREFILLED_TOKENS=1
export OMNI_SKIP_DECODE_TOKENIZE=1
export VLLM_MQ_MAX_CHUNK_BYTES_MB=100
export VLLM_PLUGINS="omni-npu,omni_npu_patches,omni_custom_models"
export HYBRID_ATTN_GROUP_SIZE=17
export OMNI_NPU_PATCHES_DIR="pangu_v2_hybrid, pangu_v2_moe"
export OMNI_NPU_VLLM_PATCHES="ALL"
export CUSTOM_MODEL_CONFIG_PATH="low_latency/openpangu_v2/pangu_v2_moe_bf16_a3_92B_xp1d_p_open.json"
export PANGU_TOOL_CALL_ENDS_THINKING=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

EXTRA_ARGS="--no-disable-hybrid-kv-cache-manager --max-num-batched-tokens 32768 --enable-expert-parallel --max-num-seqs 12 --enable-prefix-caching --enable-chunked-prefill --dtype bfloat16 --enforce-eager --reasoning-parser pangu --enable-auto-tool-choice --tool-call-parser pangu --kv-cache-dtype li_int8_ds_mla --enable-lopt --lopt-pool-size 16 --lopt-chunk-size 4096 --reasoning-config {\"reasoning_start_str\":\"<think>\",\"reasoning_end_str\":\"</think>\"} --distributed-executor-backend mp"

cd /workspace/omniinfer/tools/scripts
bash pd_run.sh \
  --additional-config "{\"enable_low_latency\": true}" \
  --local-decode-server-ip-list "$SERVER_IP_LIST" \
  --global-decode-server-ip-list "$SERVER_IP_LIST" \
  --prefill-pod-num "$PREFILL_POD_NUM" \
  --gloo-socket-ifname "$SOCKET_IFNAME" \
  --tp-socket-ifname "$SOCKET_IFNAME" \
  --model-path "$MODEL_PATH" \
  --master-ip "$HOST_IP" \
  --role prefill \
  --kv-role kv_producer \
  --max-model-len "$MODEL_LEN_MAX_PREFILL" \
  --master-port "$MASTER_PORT" \
  --base-api-port "$API_PORT" \
  --tp "$PREFILL_TENSOR_PARALLEL_SIZE" \
  --ascend-rt-visible-devices "$PREFILL_SERVER_LIST" \
  --kv-rank "$KV_RANK" \
  --kv-engine-id "$KV_RANK" \
  --kv-parallel-size 2 \
  --kv-connector "$KV_CONNECTOR" \
  --gpu-util 0.85 \
  --vllm-enable-mc2 1 \
  --extra-args "$EXTRA_ARGS" \
  --hccl-buffsize "$HCCL_BUFFSIZE" \
  --hccl-op-expansion-mode "$HCCL_OP_EXPANSION_MODE" \
  --num-speculative-tokens 3 \
  --log-dir "$LOG_PATH/p0" > "$LOG_PATH/p0/run_prefill.log" 2>&1
'
```

**3. 起 Decode，对应 D 节点 `run_server`**
```bash
docker exec -d \
  -e MODEL_PATH="$MODEL_PATH" \
  -e MODEL_LEN_MAX_DECODE=524288 \
  -e DECODE_SERVER_LIST="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" \
  -e DECODE_TENSOR_PARALLEL_SIZE=1 \
  -e DECODE_DATA_PARALLEL_SIZE="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" \
  -e HOST_IP=127.0.0.2 \
  -e MASTER_PORT=8100 \
  -e API_PORT=9100 \
  -e DECODE_SERVER_IP_LIST_BY_GROUP=127.0.0.2 \
  -e DECODE_SERVER_IP_LIST_ALL=127.0.0.2 \
  -e PREFILL_POD_NUM=1 \
  -e DECODE_POD_NUM=1 \
  -e SOCKET_IFNAME="$(ip -4 route list 0/0 | awk '{print $5}' | head -1)" \
  -e NUM_SERVERS=16 \
  -e HOST=d0 \
  -e KV_CONNECTOR="$KV_CONNECTOR" \
  "$DOCKER_NAME_D" bash -lc '
. ~/.bashrc
export HCCL_BUFFSIZE=1200
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export HCCL_OP_RETRY_ENABLE="L0:0,L1:0,L2:0"
export HCCL_OP_EXPANSION_MODE="AI_CPU"
export OMNI_NPU_USE_DEVICE_COMM_A2A=1
export OMNI_REUSE_PREFILLED_TOKENS=1
export OMNI_SKIP_DECODE_TOKENIZE=1
export VLLM_MQ_MAX_CHUNK_BYTES_MB=100
export VLLM_PLUGINS="omni-npu,omni_npu_patches,omni_custom_models"
export HYBRID_ATTN_GROUP_SIZE=17
export OMNI_NPU_PATCHES_DIR="pangu_v2_hybrid, pangu_v2_moe"
export OMNI_NPU_VLLM_PATCHES="ALL"
export CUSTOM_MODEL_CONFIG_PATH="low_latency/openpangu_v2/pangu_v2_moe_bf16_a3_92B_xp1d_d_open.json"
export PANGU_TOOL_CALL_ENDS_THINKING=1
export TORCH_COMPILE_DISABLE=1
export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LOCAL_WORLD_SIZE=8

EXTRA_ARGS="--no-disable-hybrid-kv-cache-manager --enable-expert-parallel --disable-log-requests --max-num-seqs 4 --enable-prefix-caching --enable-chunked-prefill --distributed-executor-backend mp --dtype bfloat16 --compilation-config {\"level\":3,\"cudagraph_mode\":\"FULL\",\"cudagraph_capture_sizes\":[16],\"backend\":\"\",\"compile_sizes\":[16]} --reasoning-parser pangu --enable-auto-tool-choice --tool-call-parser pangu --kv-cache-dtype li_int8_ds_mla --enable-lopt --lopt-pool-size 16 --lopt-chunk-size 4096 --reasoning-config {\"reasoning_start_str\":\"<think>\",\"reasoning_end_str\":\"</think>\"}"

cd /workspace/omniinfer/tools/scripts
bash pd_run.sh \
  --additional-config "{\"enable_low_latency\": true, \"npugraph_ex_config\":{\"enable\": true, \"super_kernel_optimize\": false, \"static_kernel_compile\": false}}" \
  --local-decode-server-ip-list "$DECODE_SERVER_IP_LIST_ALL" \
  --global-decode-server-ip-list "$DECODE_SERVER_IP_LIST_ALL" \
  --prefill-pod-num "$PREFILL_POD_NUM" \
  --gloo-socket-ifname "$SOCKET_IFNAME" \
  --tp-socket-ifname "$SOCKET_IFNAME" \
  --num-servers "$NUM_SERVERS" \
  --num-dp 16 \
  --server-offset 0 \
  --model-path "$MODEL_PATH" \
  --master-ip "$HOST_IP" \
  --role decode \
  --kv-role kv_consumer \
  --max-model-len "$MODEL_LEN_MAX_DECODE" \
  --master-port "$MASTER_PORT" \
  --base-api-port "$API_PORT" \
  --tp "$DECODE_TENSOR_PARALLEL_SIZE" \
  --kv-rank "$PREFILL_POD_NUM" \
  --kv-engine-id "$PREFILL_POD_NUM" \
  --kv-parallel-size 2 \
  --kv-connector "$KV_CONNECTOR" \
  --gpu-util 0.85 \
  --vllm-enable-mc2 1 \
  --extra-args "$EXTRA_ARGS" \
  --hccl-buffsize "$HCCL_BUFFSIZE" \
  --hccl-op-expansion-mode "$HCCL_OP_EXPANSION_MODE" \
  --num-speculative-tokens 3 \
  --log-dir "$LOG_PATH/d0" > "$LOG_PATH/d0/run_decode.log" 2>&1
'
```

**4. 起 Proxy，对应 `--tags run_proxy`**
```bash
docker exec -d \
  -e PREFILL_POD_NUM=1 \
  -e DECODE_POD_NUM=1 \
  -e MODEL_PATH="$MODEL_PATH" \
  -e PROXY_NODE_PORT=7000 \
  "$DOCKER_NAME_C" bash -lc '
. ~/.bashrc
export PYTHONHASHSEED=1234
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

cd /workspace/omniinfer/components/omni-proxy/omni_proxy
bash omni_proxy.sh \
  --listen-port 7000 \
  --prefill-endpoints "127.0.0.1:9000" \
  --decode-endpoints "127.0.0.2:9100,127.0.0.2:9101,127.0.0.2:9102,127.0.0.2:9103,127.0.0.2:9104,127.0.0.2:9105,127.0.0.2:9106,127.0.0.2:9107,127.0.0.2:9108,127.0.0.2:9109,127.0.0.2:9110,127.0.0.2:9111,127.0.0.2:9112,127.0.0.2:9113,127.0.0.2:9114,127.0.0.2:9115" \
  --log-file "$LOG_PATH/c0/nginx_error.log" \
  --log-level info \
  --access-log-file "$LOG_PATH/c0/nginx_access.log" \
  --core-num 4 \
  --start-core-index 16 \
  --subrequest-output-buffer-size 10M \
  --prefill-pod-size 1 \
  --decode-pod-size 1 \
  --omni-proxy-pd-policy sequential \
  --omni-proxy-max-batch-num-token 100000 \
  --omni-proxy-prefill-max-num-seqs 12 \
  --omni-proxy-decode-max-num-seqs 4
'
```

核心映射是：

- `run_docker`：清理旧容器，然后启动 `docker_p_p0`、`docker_d_d0`、`docker_c_c0`
- `run_server`：先起 decode，再起 prefill，实际都调用 `/workspace/omniinfer/tools/scripts/pd_run.sh`
- `run_proxy`：调用 `/workspace/omniinfer/components/omni-proxy/omni_proxy/omni_proxy.sh`
- 默认 proxy 入口端口是 `7000`

注意：上面按 inventory 里的 `127.0.0.1 / 127.0.0.2` 展开了。如果你实际是多机部署，需要把 P、D、C 分别放到对应机器上执行，并把 `HOST_IP`、`SERVER_IP_LIST`、`prefill/decode endpoints` 换成真实通信 IP。
