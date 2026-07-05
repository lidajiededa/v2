# https://gitcode.com/ascend-tribe/openPangu-2.0-Infer/tree/9…

> https://gitcode.com/ascend-tribe/openPangu-2.0-Infer/tree/92B_A2/tools 把这个仓下载下来

<details><summary>4 previous messages</summary>

> 我来把这个 GitCode 仓库拉到当前工作区里。链接指向 `92B_A2` 分支下的 `tools` 目录，所以我会优先克隆该分支的完整仓库，方便目录结构保持原样。
>
> 当前目录本身已经是一个 Git 工作区（有 `.git`），所以我会把远端仓库克隆到子目录 `openPangu-2.0-Infer`，避免覆盖你当前目录的 Git 状态。
>
> 克隆已经完成了，我再快速确认一下分支和 `tools` 目录是否在预期位置。
>
> `tools` 目录已经在本地了，里面有 `ansible`、`docker`、`quant`、`scripts`。最后一个状态检查还没回显，我等它结束确认一下仓库干净。
</details>

下载好了，仓库在：

[C:\Users\李杰\Documents\openpangu\openPangu-2.0-Infer](</C:/Users/李杰/Documents/openpangu/openPangu-2.0-Infer>)

已切到 `92B_A2` 分支，`tools` 目录也确认存在，路径是：

[C:\Users\李杰\Documents\openpangu\openPangu-2.0-Infer\tools](</C:/Users/李杰/Documents/openpangu/openPangu-2.0-Infer/tools>)

> 比较一下tools/ansible/template/omni_infer_server_template_performance1P1D_92B_bf16_A2_open.yml 和tools/ansible/template/omni_infer_server_template_performance1P1D_92B_bf16_open.yml 这两个有什么区别？

<details><summary>3 previous messages</summary>

> 我来直接对这两个模板做结构化 diff，再读一下差异附近的上下文，给你归纳成可执行层面的区别。
>
> 差异范围很小，基本集中在 prefill/decode 启动脚本里的环境变量和资源参数。我再取一下行号，方便你定位。
>
> 定位到了：`A2_open` 多了几组日志/profiler/A2A 环境变量；普通 `open` 则把部分并发/显存参数调得不一样。下面我按运行影响归纳。
</details>

这两个文件绝大部分内容一样，差异只在 `run_vllm_server_prefill_cmd` 和 `run_vllm_server_decode_cmd` 两段里。

**主要区别**

| 位置 | `*_A2_open.yml` | `*_open.yml` | 影响 |
|---|---:|---:|---|
| Prefill `GPU_UTIL` | `0.85` | `0.8` | A2 版 prefill 显存利用率更高 |
| Decode `GPU_UTIL` | `0.85` | `0.82` | A2 版 decode 显存利用率更高 |
| Decode `LOCAL_WORLD_SIZE` | `8` | `16` | 普通 open 版按 16 个本地 world size 配置，A2 版是 8 |
| Decode A2A 通信 | 有 `export OMNI_NPU_USE_DEVICE_COMM_A2A=1` | 无 | A2 版启用了 NPU device-to-device A2A 通信相关开关 |
| Prefill 日志配置 | 有 `ASCEND_SLOG_PRINT_TO_STDOUT`、`ASCEND_GLOBAL_EVENT_ENABLE`、`ASCEND_PROCESS_LOG_PATH`、清理 `${LOG_PATH}/plog/*` | 无 | A2 版额外配置 Ascend plog/slog |
| Decode 日志配置 | 同上 | 无 | A2 版额外配置 Ascend plog/slog |
| Prefill profiler | 有 `VLLM_TORCH_PROFILER_DIR=${LOG_PATH}/profiling` 和 `WITH_STACK=0` | 无 | A2 版会设置 profiler 输出目录 |
| Decode profiler | 有 `VLLM_TORCH_PROFILER_DIR=${LOG_PATH}/profiling` 和重复设置 `WITH_STACK=0` | 无 | A2 版会设置 profiler 输出目录 |

对应行号大致是：

- A2 文件：[omni_infer_server_template_performance1P1D_92B_bf16_A2_open.yml](</C:/Users/李杰/Documents/openpangu/openPangu-2.0-Infer/tools/ansible/template/omni_infer_server_template_performance1P1D_92B_bf16_A2_open.yml:87>)
- 普通 open 文件：[omni_infer_server_template_performance1P1D_92B_bf16_open.yml](</C:/Users/李杰/Documents/openpangu/openPangu-2.0-Infer/tools/ansible/template/omni_infer_server_template_performance1P1D_92B_bf16_open.yml:93>)

一句话总结：`A2_open` 更像是面向 A2 硬件/通信路径的模板，启用了 `OMNI_NPU_USE_DEVICE_COMM_A2A`，显存利用率更激进，并保留了更多 Ascend 日志和 profiler 配置；普通 `open` 版去掉了这些日志/profiler/A2A 开关，同时把 decode 的 `LOCAL_WORLD_SIZE` 调成了 `16`。

> 看一下 a2版启动服务，要做哪些事情

<details><summary>3 previous messages</summary>

> 我来顺着 A2 版模板和 ansible 目录看启动链路：需要准备哪些配置、跑哪些 tag、以及模板实际会在容器里生成/执行什么脚本。
>
> A2 的 inventory 模板也在同一个目录，启动不是只跑一个 yml：需要先把 inventory 里的机器/IP/卡号/端口这些填好，再按 tag 分阶段执行 docker、server、proxy。
>
> README 里给的是普通 bf16 的命令；套到 A2 版时，文件名要换成 `*_A2_open.yml`，inventory 也换成 `omni_infer_inventory_used_for_1P1D_A2.yml`。我再看一下 README 的具体准备项。
</details>

A2 版启动服务大概分成三件事：改配置、起容器、起服务。

**1. 先改两个文件**

在目录 [tools/ansible/template](</C:/Users/李杰/Documents/openpangu/openPangu-2.0-Infer/tools/ansible/template>) 下用这两个 A2 文件：

- inventory：[omni_infer_inventory_used_for_1P1D_A2.yml](</C:/Users/李杰/Documents/openpangu/openPangu-2.0-Infer/tools/ansible/template/omni_infer_inventory_used_for_1P1D_A2.yml>)
- playbook：[omni_infer_server_template_performance1P1D_92B_bf16_A2_open.yml](</C:/Users/李杰/Documents/openpangu/openPangu-2.0-Infer/tools/ansible/template/omni_infer_server_template_performance1P1D_92B_bf16_A2_open.yml>)

inventory 里要填：

- `P` 节点 IP：prefill 节点
- `D` 节点 IP：decode 节点
- `C` 节点 IP：proxy 节点，通常放在 P 节点
- `ascend_rt_visible_devices`：A2 版默认是 `"0,1,2,3,4,5,6,7"`
- 端口：`global_port_base`、`base_api_port`、`proxy_port`

playbook 头部 `environment` 要改：

- `LOG_PATH`：日志目录
- `MODEL_PATH`：模型权重目录，P/D 节点路径要一致
- `LOG_PATH_IN_EXECUTOR`：拉日志用
- `DOCKER_IMAGE_ID`：实际 A2 镜像名
- `DOCKER_NAME_P` / `DOCKER_NAME_D` / `DOCKER_NAME_C`
- `SCRIPTS_PATH`：生成启动脚本的位置
- 必要时改 `MODEL_LEN_MAX_PREFILL`、`MODEL_LEN_MAX_DECODE`、`DECODE_TENSOR_PARALLEL_SIZE`

**2. 准备机器环境**

执行机需要装 `ansible`，并且能免密 SSH 到 P/D/C 节点，默认用户是 `root`。

各节点需要提前有：

- Docker
- Ascend 驱动/CANN 环境
- `/dev/davinci_manager`、`/dev/hisi_hdc`、`/dev/devmm_svm`
- `/usr/local/Ascend/driver`
- `/usr/local/dcmi`
- `/etc/hccn.conf`
- `/usr/bin/hccn_tool`
- 模型目录和日志目录

playbook 里的 docker 会挂载 `/data`、`/mnt`、`/tmp`、`/home`、Ascend 驱动目录、模型目录、日志目录、脚本目录。

**3. 启动命令**

进入模板目录：

```bash
cd tools/ansible/template
```

先创建 P/D/C 三类容器：

```bash
ansible-playbook -i omni_infer_inventory_used_for_1P1D_A2.yml \
  omni_infer_server_template_performance1P1D_92B_bf16_A2_open.yml \
  --tags run_docker
```

然后启动推理服务和 proxy：

```bash
ansible-playbook -i omni_infer_inventory_used_for_1P1D_A2.yml \
  omni_infer_server_template_performance1P1D_92B_bf16_A2_open.yml \
  --tags run_server,run_proxy
```

注意：`run_docker` 会检查并删除同名容器，再重新创建，所以同名容器里有东西的话要小心。

**它实际会做什么**

`run_server,run_proxy` 阶段会：

1. 计算 P/D/C 的服务列表、DP 数、offset 等变量。
2. 给 P 容器生成 `$SCRIPTS_PATH/vllm_run_for_p.sh`。
3. 给 D 容器生成 `$SCRIPTS_PATH/vllm_run_for_d.sh`。
4. 自动取默认网卡作为通信网卡。
5. 在 D 节点先启动 decode 服务。
6. 在 P 节点启动 prefill 服务。
7. 给 C 容器生成并启动 `run_proxy_server.sh`，里面会跑 `omni_proxy.sh`。
8. proxy 默认监听 inventory 里的 `proxy_port`，A2 inventory 默认是 `2000`。

A2 版特有的关键点是 decode 里启了：

```bash
export OMNI_NPU_USE_DEVICE_COMM_A2A=1
export LOCAL_WORLD_SIZE=8
```

同时 P/D 都会跑 `/workspace/omniinfer/tools/scripts/pd_run.sh`，并带上低时延配置、Pangu reasoning/tool-call parser、prefix caching、chunked prefill、speculative tokens 等参数。

> 仓内没有提供混部方式，但理论上可以pd部署就可以混部，你能否分析下源码看下如何混部？

<details><summary>5 previous messages</summary>

> 我来从 ansible 模板、`pd_run.sh` 和 proxy/脚本参数一路追一下。重点看它是否允许 P/D/C 指向同一台机器、如何用不同卡号隔离，以及哪些地方默认假设 P 和 D 是不同主机。
>
> `pd_run.sh` 这里已经看到一个关键条件：P/D 的隔离主要靠 `--ascend-rt-visible-devices` 传入不同卡号，以及 `kv_rank/kv_parallel_size` 区分 KV 通信角色；它本身没有要求 P/D 必须是不同物理机。接下来我看 `start_api_servers.py` 怎么分配端口/设备。
>
> 有一个很重要的发现：inventory 里的 `ascend_rt_visible_devices` 只被模板用于算数量和传给 prefill，但最终 `start_api_servers.py` 会把每个 vLLM 子进程的 `ASCEND_RT_VISIBLE_DEVICES` 重写成从 0 开始的连续编号。这意味着“同机 P 用 0-3、D 用 4-7”光改 inventory 大概率不生效，得改脚本或用容器级隔离。
>
> 仓里其实还有一个 `components/omni-npu/examples/serve-pd-disaggregate.sh`，它不是 ansible，但写法更接近“同机 PD 混部”：直接给 prefill/decode 分别设置 `ASCEND_RT_VISIBLE_DEVICES`。我看它的启动参数，能反推 ansible 需要补哪些东西。
>
> 另一个混部坑：默认端口在分机没问题，但同机时可能撞。P 的 `api_port` 默认 7000，模板里还算了一个 `ENDPOINT_PORT=api_port+100`，而 D 的 `api_port` 默认也是 7100；即使这个 `KV_EVENTS_CONFIG` 当前没继续传进 `EXTRA_ARGS`，混部时最好把 D 端口段整体后移，避免后续打开事件配置时踩雷。
</details>

可以，源码看下来结论是：**PD 混部理论上能走通，但当前 ansible 模板不是完整的混部实现，直接只改 inventory 会有设备隔离问题。**

**源码里的启动模型**
它的 PD 部署本质是三类进程：

- P：prefill vLLM，`kv_role=kv_producer`
- D：decode vLLM，`kv_role=kv_consumer`
- C：proxy，把请求先发 P，再带着 `kv_transfer_params` 发 D

关键链路在：

- [A2 ansible 模板](</C:/Users/李杰/Documents/openpangu/openPangu-2.0-Infer/tools/ansible/template/omni_infer_server_template_performance1P1D_92B_bf16_A2_open.yml:794>)
- [pd_run.sh](</C:/Users/李杰/Documents/openpangu/openPangu-2.0-Infer/tools/scripts/pd_run.sh:274>)
- [start_api_servers.py](</C:/Users/李杰/Documents/openpangu/openPangu-2.0-Infer/tools/scripts/start_api_servers.py:161>)

**混部应该怎么表达**
inventory 里可以用不同 host alias 指向同一台机器：

```yaml
P:
  hosts:
    p0:
      ansible_host: "10.0.0.1"
      host_ip: "10.0.0.1"
      ascend_rt_visible_devices: "0,1,2,3"
      kv_rank: 0
      node_rank: 0

D:
  hosts:
    d0:
      ansible_host: "10.0.0.1"
      host_ip: "10.0.0.1"
      ascend_rt_visible_devices: "4,5,6,7"
      node_rank: 0

C:
  hosts:
    c0:
      ansible_host: "10.0.0.1"
```

这样 ansible 会在同一台物理机上创建 P/D/C 三个容器，容器名因为带 `inventory_hostname` 后缀，所以不会冲突。

**但现在直接这样改还不够**
最大的问题在 [start_api_servers.py:161](</C:/Users/李杰/Documents/openpangu/openPangu-2.0-Infer/tools/scripts/start_api_servers.py:161>)：

```python
env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, range(rank*tp*pp, (rank+1)*tp*pp)))
```

它会把传进来的设备列表重写成从 `0` 开始的连续编号。结果是：

- 你配置 P 用 `0,1,2,3`
- D 配 `4,5,6,7`
- 但 D 最终启动时仍可能用 `0,1,2,3`

所以混部必须改这里，让它尊重父进程传入的 `ASCEND_RT_VISIBLE_DEVICES`，按列表切片，而不是硬编码从 0 开始。

同时 A2 模板的 decode 调 `pd_run.sh` 时目前没有传：

```bash
--ascend-rt-visible-devices "${DECODE_SERVER_LIST}"
```

这个也要补上。

**建议改法**
`start_api_servers.py` 里把设备分配改成这种逻辑：

```python
visible = [
    x.strip()
    for x in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
    if x.strip()
]
start = rank * tp * pp
end = (rank + 1) * tp * pp

if visible:
    env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(visible[start:end])
else:
    env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, range(start, end)))
```

然后在 A2 模板 decode 的 `pd_run.sh` 参数里补：

```bash
--ascend-rt-visible-devices "${DECODE_SERVER_LIST}" \
```

另外，如果 D 只混部用 4 张卡，A2 模板里的：

```bash
export LOCAL_WORLD_SIZE=8
```

也建议改成：

```bash
export LOCAL_WORLD_SIZE=${NUM_SERVERS}
```

**端口也要避开**
默认 P 的 API 是 `7000`，D 是 `7100`。模板里 P 还算了 `ENDPOINT_PORT=api_port+100`，也就是 `7100`。当前这个 `KV_EVENTS_CONFIG` 没实际传给 vLLM，但混部时建议直接把 D 端口段后移，比如：

```yaml
port_offset:
  P: 0
  D: 200
```

这样 P API 是 `7000`，D API 从 `7200` 开始，更稳。

**一句话方案**
混部方式就是：**P/D/C 用同一个 `ansible_host`，P/D 用不同 `ascend_rt_visible_devices` 和不同端口段；再修 `start_api_servers.py` 和 A2 playbook，让 decode 真正继承设备列表。**

仓里 `components/omni-npu/examples/serve-pd-disaggregate.sh` 已经证明了这个思路：它就是同机启动 P 和 D，并分别设置 `ASCEND_RT_VISIBLE_DEVICES`。ansible 版差的就是把这套设备隔离逻辑接完整。

> 有不通过pd分离的部署方法吗，直接单机8卡不pd部署

<details><summary>4 previous messages</summary>

> 我来查一下仓里有没有单实例/非 PD 的启动脚本和说明。你问的是“单机 8 卡一个 vLLM 服务”，所以我会重点看 `examples`、`tools/docker` 和 README 里是否有不带 `kv-transfer-config` 的 serve 命令。
>
> 找到了：仓里有 `components/omni-npu/examples/serve-single-instance.sh`，这就是非 PD 的单实例启动脚本。现在我读一下它的参数，确认能不能直接按 8 卡 TP 来跑。
>
> 这个单实例脚本默认更像通用示例：模型名默认是 `deepseek`，网卡和可见卡也写死了。对于 openPangu 92B，要把它改成/替换成 Pangu 的插件、模型名和 8 卡 TP 参数。
>
> Pangu 92B 的模型配置里有 `config_file` 这种单实例配置项，不只是 PD 的 `prefill_config_file/decode_config_file`。这说明非 PD 运行应该使用 hybrid/single 的 config，而不是 P/D 专用 config。
</details>

有。仓里提供了非 PD 的单实例脚本：

[serve-single-instance.sh](</C:/Users/李杰/Documents/openpangu/openPangu-2.0-Infer/components/omni-npu/examples/serve-single-instance.sh:1>)

它本质就是直接跑一个 `vllm serve`，不启动 P/D 两套服务，不启动 proxy，也不传 `--kv-transfer-config`。

直接单机 8 卡可以按这个思路跑：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_PLUGINS="omni-npu,omni_npu_patches,omni_custom_models"
export OMNI_NPU_PATCHES_DIR="pangu_v2_hybrid,pangu_v2_moe"
export OMNI_NPU_VLLM_PATCHES="ALL"
export CUSTOM_MODEL_CONFIG_PATH="low_latency/openpangu_v2/pangu_v2_moe_bf16_a3_hybrid.json"
export ASCEND_GLOBAL_LOG_LEVEL=3
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=fork
export TORCH_COMPILE_DISABLE=1

vllm serve /path/to/model/weights \
  --served-model-name openPangu-2.0-Flash \
  --host 0.0.0.0 \
  --port 7000 \
  --dtype bfloat16 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.85 \
  --max-model-len 524288 \
  --enable-expert-parallel \
  --reasoning-parser pangu \
  --enable-auto-tool-choice \
  --tool-call-parser pangu
```

关键点：

- 不要跑 `pd_run.sh`
- 不要起 `omni_proxy.sh`
- 不要加 `--kv-transfer-config`
- `--tensor-parallel-size 8` 对应单机 8 卡
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` 控制用哪 8 张卡

仓里的 [serve-single-instance.sh](</C:/Users/李杰/Documents/openpangu/openPangu-2.0-Infer/components/omni-npu/examples/serve-single-instance.sh:122>) 默认写的是通用模型，最后还硬编码了 `0..15`，所以用于 A2 单机 8 卡时建议直接用上面的命令，或者改脚本最后一行的 `ASCEND_RT_VISIBLE_DEVICES` 和默认模型名。

注意一点：`92B bf16` 单机 8 卡是否能稳跑，取决于每卡显存和 `max-model-len`。如果 OOM，先把 `--max-model-len` 降下来，比如 `32768` 或 `65536`，再逐步调大。

> 把一些图模式和mtp之类的参数，参照pd的，也加上啊

<details><summary>1 previous message</summary>

> 对，单实例命令如果只是裸 `vllm serve` 会少掉 PD 模板里那些性能开关。我按 A2 bf16 PD 模板里的 decode 侧为主、prefill 侧补充 `max-num-batched-tokens` 和低时延参数，整理一版更接近它的单机 8 卡启动命令。
</details>

可以，单机非 PD 版可以加上这些参数。更贴近 A2 PD 模板的命令如下：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_PLUGINS="omni-npu,omni_npu_patches,omni_custom_models"
export HYBRID_ATTN_GROUP_SIZE=17
export OMNI_NPU_PATCHES_DIR="pangu_v2_hybrid, pangu_v2_moe"
export OMNI_NPU_VLLM_PATCHES="ALL"
export CUSTOM_MODEL_CONFIG_PATH="low_latency/openpangu_v2/pangu_v2_moe_bf16_a3_hybrid.json"

export ASCEND_GLOBAL_LOG_LEVEL=3
export HCCL_OP_EXPANSION_MODE="AI_CPU"
export HCCL_BUFFSIZE=1200
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export HCCL_OP_RETRY_ENABLE="L0:0,L1:0,L2:0"

export OMNI_REUSE_PREFILLED_TOKENS=1
export OMNI_SKIP_DECODE_TOKENIZE=1
export VLLM_MQ_MAX_CHUNK_BYTES_MB=100
export PANGU_TOOL_CALL_ENDS_THINKING=1
export TORCH_COMPILE_DISABLE=1
export VLLM_CACHE_ROOT="./graph_cache"
export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=False
export VLLM_TORCH_PROFILER_WITH_STACK=False
export VLLM_TORCH_PROFILER_WITH_FLOPS=False
export TASK_QUEUE_ENABLE=1
export VLLM_ENABLE_MC2=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LOCAL_WORLD_SIZE=8

vllm serve /path/to/model/weights \
  --served-model-name openPangu-2.0-Flash \
  --host 0.0.0.0 \
  --port 7000 \
  --dtype bfloat16 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.85 \
  --max-model-len 524288 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 3 \
  --no-disable-hybrid-kv-cache-manager \
  --enable-expert-parallel \
  --disable-log-requests \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --reasoning-parser pangu \
  --enable-auto-tool-choice \
  --tool-call-parser pangu \
  --enable-lopt \
  --lopt-pool-size 16 \
  --lopt-chunk-size 4096 \
  --compilation-config '{"level":3,"cudagraph_mode":"FULL","cudagraph_capture_sizes":[12],"backend":"","compile_sizes":[12]}' \
  --additional-config '{"enable_low_latency":true,"npugraph_ex_config":{"enable":true,"super_kernel_optimize":false,"static_kernel_compile":false}}' \
  --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
  --speculative_config '{"method":"mtp","num_speculative_tokens":3}'
```

这里对应 PD 里的几个关键增强：

- 图模式：`--compilation-config ... cudagraph_mode FULL`
- NPU graph：`--additional-config ... npugraph_ex_config`
- MTP：`--speculative_config '{"method":"mtp","num_speculative_tokens":3}'`
- 低时延：`enable_low_latency`
- prefix cache / chunked prefill / lopt 都按 PD 模板补上了
- 单机 8 卡：`ASCEND_RT_VISIBLE_DEVICES=0..7` + `--tensor-parallel-size 8`

如果启动报显存不够，优先降这几个：`--max-model-len`、`--max-num-batched-tokens`、`--gpu-memory-utilization`、`--max-num-seqs`。

> 如果我要两天a2，以非pd的方式，tp16来部署，要怎么做

<details><summary>1 previous message</summary>

> 两台 A2 做非 PD 的 TP16，思路就是“一个 vLLM 实例跨两台机器做 TP”，不要 P/D/proxy。仓里的 PD prefill 多机分支也是用 Ray 把多节点拼起来，所以非 PD 也建议按 Ray executor 来跑。
</details>

我按“**两台 A2**”理解。非 PD、TP16 的做法是：两台机器各 8 卡，先拉一个 Ray 集群，然后只在 head 节点启动一个 `vllm serve`，参数用 `--tensor-parallel-size 16 --distributed-executor-backend ray`。

**两台机器都先设环境**
把网卡名换成你的实际通信网卡：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0

export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export VLLM_PLUGINS="omni-npu,omni_npu_patches,omni_custom_models"
export HYBRID_ATTN_GROUP_SIZE=17
export OMNI_NPU_PATCHES_DIR="pangu_v2_hybrid, pangu_v2_moe"
export OMNI_NPU_VLLM_PATCHES="ALL"
export CUSTOM_MODEL_CONFIG_PATH="low_latency/openpangu_v2/pangu_v2_moe_bf16_a3_hybrid.json"

export ASCEND_GLOBAL_LOG_LEVEL=3
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1200
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export HCCL_OP_RETRY_ENABLE="L0:0,L1:0,L2:0"

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=fork
export VLLM_ENABLE_MC2=1
export TASK_QUEUE_ENABLE=1
export TORCH_COMPILE_DISABLE=1
export PANGU_TOOL_CALL_ENDS_THINKING=1
```

**head 节点启动 Ray**
假设 head IP 是 `10.0.0.1`：

```bash
ray stop --force
ray start --head --node-ip-address=10.0.0.1 --port=6377 --num-gpus=8
```

**worker 节点加入 Ray**
假设 worker IP 是 `10.0.0.2`：

```bash
ray stop --force
ray start --address=10.0.0.1:6377 --node-ip-address=10.0.0.2 --num-gpus=8
```

**只在 head 节点启动 vLLM**
模型路径要求两台机器都能访问，最好路径一致：

```bash
vllm serve /path/to/model/weights \
  --served-model-name openPangu-2.0-Flash \
  --host 0.0.0.0 \
  --port 7000 \
  --dtype bfloat16 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --data-parallel-size 1 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.82 \
  --max-model-len 524288 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 3 \
  --no-disable-hybrid-kv-cache-manager \
  --enable-expert-parallel \
  --disable-log-requests \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --reasoning-parser pangu \
  --enable-auto-tool-choice \
  --tool-call-parser pangu \
  --enable-lopt \
  --lopt-pool-size 16 \
  --lopt-chunk-size 4096 \
  --compilation-config '{"level":3,"cudagraph_mode":"FULL","cudagraph_capture_sizes":[12],"backend":"","compile_sizes":[12]}' \
  --additional-config '{"enable_low_latency":true,"npugraph_ex_config":{"enable":true,"super_kernel_optimize":false,"static_kernel_compile":false}}' \
  --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
  --speculative_config '{"method":"mtp","num_speculative_tokens":3}'
```

核心区别就是：

- 单机 8 卡：`mp + tp8`
- 两台 A2 非 PD：`ray + tp16`
- 不启动 proxy
- 不传 `--kv-transfer-config`
- 不区分 prefill/decode

如果启动时 OOM，优先把 `--max-model-len 524288` 降到 `65536` 或 `32768` 验证链路，确认 TP16 跨机跑通后再往上调。

> 好的，然后请你深入代码，看看按照之前的默认1p1d部署方式，a2部署有哪些性能上可以优化的点

<details><summary>8 previous messages</summary>

> 我会从 A2 的默认 1P1D 模板出发，把“实际启动参数 -> vLLM 子进程 -> 模型配置/patch”串起来看。重点找三类东西：已经打开但可能不匹配 A2 的参数、默认没打开但代码支持的性能开关、以及模板里看起来写了但没有真正生效的配置。
>
> 第一轮看到两个“值得重点盯”的点：A2 模板仍在用 `pangu_v2_moe_bf16_a3_92B_xp1d_*_open.json` 这类配置文件；另外 `pd_run.sh` 会把 `HCCL_EXEC_TIMEOUT` 从模板里的 1800 覆盖成 120。下面我看这些配置文件和 patch 里还有哪些可调项。
>
> 配置层面有个很明显的优化线索：仓里存在 `pangu_v2_moe_bf16_a2_92B_xp1d_d_perf.json`，里面打开了 `use_mhc_fusion_op`、`use_topk_topp_stream`、`num_extra_reserved_blocks=8`，而默认 A2 模板使用的是 `*_open.json`，这些没打开。继续追这些字段具体影响。
>
> 这里又看到一个行为差异：`_validate_config()` 会在 eager 模式下关掉一批优化，比如 prefetch/super kernel；A2 模板的 prefill 明确 `--enforce-eager`，所以 prefill 侧天然放弃图模式和预取类优化。decode 侧走 `compilation-config`，所以优化空间主要集中在 decode。
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> 我继续把源码里已经抓到的开关串起来看，重点会放在“默认 1P1D A2 配置里已经开了什么、哪些开关被配置绕过了、哪些值得优先压测”。这类性能点我会尽量按收益和风险分层，不把所有参数都混成一锅粥。
>
> 这里面有一个挺关键的发现：A2 模板虽然叫 A2，但 decode 默认绑的是 `a3_92B_xp1d_d_open.json`，而仓里同时存在一个 `a2_92B_xp1d_d_perf.json`，里面打开了好几个 decode 热路径优化。这个不是“玄学调参”，是源码配置层面直接能看到的差异。
>
> 另一个有价值的点是 graph/static kernel：decode 这边已经走 `cudagraph_mode FULL`，但模板又把 `static_kernel_compile` 和 `super_kernel_optimize` 关掉，并且每次启动都删静态 kernel/graph cache。对于固定 decode shape 的 1P1D，这个明显是“先求稳”的配置，不是性能极限配置。
</details>

我看了一圈源码后，结论是：默认 A2 1P1D 配置是一个“稳态可跑”的配置，不是性能拉满配置。最值得先优化的是 decode 侧模型配置、decode 并发形状、graph/static kernel，以及 prefill 吞吐参数。

**优先级最高**

1. **decode 侧没有用 A2 perf 配置**

   A2 模板里 decode 明确写的是 `a3_92B_xp1d_d_open.json`：
   [A2 模板](./openPangu-2.0-Infer/tools/ansible/template/omni_infer_server_template_performance1P1D_92B_bf16_A2_open.yml:180)

   但仓里有更激进的 A2 decode perf 配置：
   [a2_92B_xp1d_d_perf.json](./openPangu-2.0-Infer/components/omni-npu/src/omni_npu/model_config/configs/low_latency/openpangu_v2/pangu_v2_moe_bf16_a2_92B_xp1d_d_perf.json:13)

   它相比当前 open 配置多开/改了这些关键项：

   - `use_mhc_fusion_op: true`
   - `use_topk_topp_stream: true`
   - `disable_npu_top_k_top_p_sample: false`
   - `num_extra_reserved_blocks: 8`
   - `optimize_first_chunk: false`

   其中 `use_mhc_fusion_op` 在模型 forward 里只对小 token batch 生效，源码条件是 `hidden_states.shape[0] <= 256`，这正好是 decode 热路径：
   [pangu_v2_moe.py](./openPangu-2.0-Infer/components/omni-npu/src/omni_npu/v1/models/pangu/pangu_v2_moe.py:1537)

   建议先做一个 A/B：把 decode 的 `CUSTOM_MODEL_CONFIG_PATH` 改成 `pangu_v2_moe_bf16_a2_92B_xp1d_d_perf.json`，只动 decode，不动 prefill，看 TPOT、吞吐、显存和采样正确性。

2. **decode 并发偏保守：bf16 是 3，w8a8 模板是 4**

   当前 bf16 A2 decode 是：

   - `--max-num-seqs 3`
   - MTP tokens = 3
   - `cudagraph_capture_sizes: [12]`

   这套是自洽的，因为 `3 seqs * (1 + 3 speculative tokens) = 12`：
   [A2 模板](./openPangu-2.0-Infer/tools/ansible/template/omni_infer_server_template_performance1P1D_92B_bf16_A2_open.yml:218)

   但 w8a8 1P1D 模板 decode 用到了 `max-num-seqs 4` 和 capture size `16`。所以 A2 bf16 可以压测一组：

   - `--max-num-seqs 4`
   - `cudagraph_capture_sizes: [16]`
   - `compile_sizes: [16]`
   - proxy 同步改 `--omni-proxy-decode-max-num-seqs 4`

   这是吞吐方向最直接的调优点，风险是显存和尾延迟。

3. **graph/static kernel 现在没有拉满**

   decode 已经用了 `cudagraph_mode: FULL`，但 `additional-config` 里关了：

   - `super_kernel_optimize: false`
   - `static_kernel_compile: false`

   [A2 模板](./openPangu-2.0-Infer/tools/ansible/template/omni_infer_server_template_performance1P1D_92B_bf16_A2_open.yml:232)

   源码里这两个开关是真有用的，开启后会走 static kernel / super kernel 路径：
   [npugraph_ex.py](./openPangu-2.0-Infer/components/omni-npu/src/omni_npu/compilation/npugraph_ex.py:65)

   但模板每次启动还会删 static kernel 和 graph cache：
   [A2 模板](./openPangu-2.0-Infer/tools/ansible/template/omni_infer_server_template_performance1P1D_92B_bf16_A2_open.yml:214)

   建议生产压测时分两档：

   - 稳妥档：`static_kernel_compile: true`
   - 激进档：`static_kernel_compile: true, super_kernel_optimize: true`

   同时不要每次启动都删 cache，除非 CANN、模型、shape 或编译参数变了。

**第二优先级**

4. **prefill 侧 batch token 偏小**

   bf16 默认 prefill 是 `--max-num-batched-tokens 16384 --max-num-seqs 4`：
   [A2 模板](./openPangu-2.0-Infer/tools/ansible/template/omni_infer_server_template_performance1P1D_92B_bf16_A2_open.yml:100)

   w8a8 模板是 `32768 / 12`。bf16 显存压力更大，但可以试：

   - `24576 / 4`
   - `32768 / 4`
   - 如果请求多且 prompt 不太长，再试 `max-num-seqs 6 或 8`

   proxy 的 `--omni-proxy-prefill-max-num-seqs` 也要同步，否则前端调度会卡住上限。

5. **prefill 强制 eager 会关掉一些优化**

   prefill 带了 `--enforce-eager`。源码里 eager mode 会禁用 prefetch/super kernel 等优化：
   [features.py](./openPangu-2.0-Infer/components/omni-npu/src/omni_npu/model_config/config_loader/features.py:20)

   这大概率是为了兼容动态长 prompt。除非你的 prompt shape 很稳定，否则不建议第一轮就去掉；但如果是固定 benchmark，可以单独试“不 enforce eager”的 prefill 图模式收益。

6. **GPU 利用率可以小步上调**

   当前 P/D 都是 `GPU_UTIL=0.85`。如果实际显存有余量，可以试 `0.88`、`0.90`。这个主要提升 KV capacity 和并发空间，但要和 `num_extra_reserved_blocks=8`、decode max seqs 一起看，别一次改太多。

**稳定性/工程项**

7. `pd_run.sh` 会把模板的 `HCCL_EXEC_TIMEOUT=1800` 覆盖成 `120`：
   [pd_run.sh](./openPangu-2.0-Infer/tools/scripts/pd_run.sh:331)

   这不是性能收益点，但长上下文/高压时可能导致误超时，建议改成“如果外部已设置就保留”。

8. `start_api_servers.py` 会重写 `ASCEND_RT_VISIBLE_DEVICES`：
   [start_api_servers.py](./openPangu-2.0-Infer/tools/scripts/start_api_servers.py:161)

   默认 1P1D 没问题，但如果你后面做混部或非连续卡分配，这是性能和正确性风险点。

我的建议压测顺序是：先只换 decode A2 perf config；再把 decode `3/12` 改成 `4/16`；然后试 static kernel；最后再调 prefill batch tokens 和 GPU util。这样每轮变量少，收益归因会清楚。

> 92B A2如果要使用omni_cache,仓里有a3 4p1d的yaml,我a2 1p1d,需要做哪些改动，注意a2一台机器的cpu内存是1.5T，而A3是2T

<details><summary>1 previous message</summary>

> 我先把仓里的 4P1D omni_cache 模板和 A2 1P1D 模板对齐看一下，尤其看它到底通过哪些 env、config json、proxy 参数和 playbook 任务启用 cache。CPU 内存 1.5T 这个约束也得落到 cache 大小和 P/D 配比上，不能只照抄 A3。
</details>

仓里这个 omni_cache 模板不是简单多加一个 `ENABLE_OMNI_CACHE=1`：它还会准备 hugetlbfs mmap 文件、注册 connector、覆盖 `NUM_GPU_BLOCKS_OVERRIDE/KV_CACHE_MEMORY_BYTES`，并且 P/D 两边的默认 cache 大小不一样。接下来我重点看这些大小怎么被消费，避免给你一个会把 1.5T 内存打满的配置。

> 上次终止了，请继续

<details><summary>4 previous messages</summary>

> 继续。刚才已经确认了核心差异：4P1D omni_cache 模板按 16 die/A3 和更大内存写了不少硬编码值，A2 1P1D 不能原样搬，尤其是 `LOCAL_WORLD_SIZE/NUM_DIE_PER_MACH/OMNI_CACHE_LOCAL_DP_SIZE` 和 hugepage `MAP_SIZE_BYTES`。我再补几处源码消费路径，然后给你一版可落地的改动清单。
>
> 还有一个容易踩的点：hugetlbfs 脚本默认会 zero-fill 整个 mmap 文件，也就是你配 1.2T 它就会实打实预留/触碰 1.2T，启动会慢，而且内存余量不足会直接失败。所以 A2 1.5T 机器上我会建议把 P/D 的 `MAP_SIZE_BYTES` 做成显式变量，不用模板里的 A3 默认值。
>
> 我又确认了一处模板问题：4P1D omni_cache 模板给 decode 容器传了 `OMNI_CACHE_DECODE_NUM_GPU_BLOCKS_OVERRIDE`，但 decode 启动块里没有像 prefill 那样把它转成 `NUM_GPU_BLOCKS_OVERRIDE` 并追加到 `EXTRA_ARGS`。所以 A2 1P1D 迁移时，这一处建议顺手补齐，否则你以为调了 decode blocks，实际可能没生效。
>
> 拓扑这块我按“1P1D = 一台 A2 做 Prefill、另一台 A2 做 Decode”来分析；如果你其实想 P/D 混在同一台 A2 上，内存预算要再砍一刀，因为两个 hugepage 池会在同一台 1.5T 机器上相加。
</details>

可以继续。按源码看，A2 1P1D 要从 A3 4P1D omni_cache 模板迁过来，核心不是改 P/D 数量，而是把 A3/16 die/2T 内存假设全部收掉。

**结论**
建议以 [A2 1P1D 模板](./openPangu-2.0-Infer/tools/ansible/template/omni_infer_server_template_performance1P1D_92B_bf16_A2_open.yml:1) 为底座，把 [4P1D omni_cache 模板](./openPangu-2.0-Infer/tools/ansible/template/omni_infer_server_template_performance4P1D_92B_w8a8_open_omni_cache.yml:17) 里的 omni_cache 启动块移植过去。不要反过来拿 4P1D 模板改成 A2，因为里面有不少 A3/16 die 假设。

**必须改的点**

1. 全局环境增加 omni_cache 变量，但 A2 要改成 8 die：

```yaml
ENABLE_OMNI_CACHE: "{{ enable_omni_cache | default('1') }}"
ENABLE_HOST_MAPPING: "{{ enable_host_mapping | default('0') }}"
OMNI_CACHE_PREFILL_MMAP_FILE: "{{ omni_cache_prefill_mmap_file | default('omni_cache_p') }}"
OMNI_CACHE_DECODE_MMAP_FILE: "{{ omni_cache_decode_mmap_file | default('omni_cache_d') }}"
DISABLE_GATHER_SELECTION: "{{ disable_gather_selection | default('1') }}"
ENABLE_OMNI_CACHE_DSA_SPLIT: "{{ enable_omni_cache_dsa_split | default('0') }}"
OMNI_CACHE_DSA_MMAP_FILE: "{{ omni_cache_dsa_mmap_file | default('omni_cache_decode_dsa') }}"
NUM_DIE_PER_MACH: "{{ num_die_per_mach | default('8') }}"
```

4P1D 模板默认是 `NUM_DIE_PER_MACH=16`，A2 必须是 `8`。

2. Prefill 启动块加入 omni_cache 初始化逻辑：

关键是这些：

```bash
export ENABLE_OMNI_CACHE=1
export ENABLE_HOST_MAPPING=0
export DISABLE_GATHER_SELECTION=1
export OMNI_CACHE_MMAP_FILE="${OMNI_CACHE_PREFILL_MMAP_FILE:-omni_cache_p}"
export OMNI_CACHE_MMAP_PATH="/dev/hugepages/${OMNI_CACHE_MMAP_FILE}"
export OMNI_CACHE_PACKED_HBM=1
export NUM_DIE_PER_MACH=8
export OMNI_CACHE_LOCAL_DP_SIZE=8
export P_NODE_LIST="${P_NODE_LIST:-${HOST_IP}}"
export ROLE=prefill
python -c "from omni_cache.connector import register_connectors; register_connectors()"
KV_CONNECTOR="OmniCacheConnector"
KV_PARALLEL_SIZE=1
```

然后追加：

```bash
EXTRA_ARGS="${EXTRA_ARGS} --kv-cache-memory-bytes ${KV_CACHE_MEMORY_BYTES}"
EXTRA_ARGS="${EXTRA_ARGS} --num-gpu-blocks-override ${NUM_GPU_BLOCKS_OVERRIDE}"
```

3. Decode 启动块也加 omni_cache 初始化逻辑：

```bash
export ENABLE_OMNI_CACHE=1
export ENABLE_HOST_MAPPING="${ENABLE_HOST_MAPPING:-0}"
export OMNI_CACHE_MMAP_FILE="${OMNI_CACHE_DECODE_MMAP_FILE:-omni_cache_d}"
export OMNI_CACHE_MMAP_PATH="/dev/hugepages/${OMNI_CACHE_MMAP_FILE}"
export NUM_DIE_PER_MACH=8
export OMNI_CACHE_LOCAL_DP_SIZE=8
export P_NODE_LIST="${P_NODE_LIST:-${HOST_IP}}"
export ROLE=decode
export DISABLE_GATHER_SELECTION=1
python -c "from omni_cache.connector import register_connectors; register_connectors()"
KV_CONNECTOR="OmniCacheConnector"
KV_PARALLEL_SIZE=$((dp + 1))
```

4P1D 模板 decode 里写了：

```bash
export LOCAL_WORLD_SIZE=16
export OMNI_CACHE_LOCAL_DP_SIZE=16
```

A2 必须改成：

```bash
export LOCAL_WORLD_SIZE=8
export OMNI_CACHE_LOCAL_DP_SIZE=8
```

源码里 `OMNI_CACHE_LOCAL_DP_SIZE` 会参与 decode cache shape 计算，写错会直接影响每个 DP rank 的 host cache 分配：
[constants.py](./openPangu-2.0-Infer/components/omni-cache/omni_cache/cache/core/constants.py:20)
[shapes.py](./openPangu-2.0-Infer/components/omni-cache/omni_cache/cache/transfer_engine/shapes.py:53)

**内存要改**

A3 4P1D 模板里 Prefill 默认：

```bash
OMNI_CACHE_LAYER_BYTES=88046829568
MAP_SIZE_BYTES=1610612736000  # 1500GiB
```

这对 A2 1.5T 机器太满了。hugetlbfs 脚本会按 `MAP_SIZE_BYTES` 预留 hugepages，并且默认 zero-fill，是真的会碰内存：
[setup_hugetlbfs_2MB.sh](./openPangu-2.0-Infer/components/omni-cache/tools/setup/setup_hugetlbfs_2MB.sh:145)

我建议 A2 1P1D 先用保守档：

```bash
# Prefill, 一台 A2 独占
OMNI_CACHE_LAYER_BYTES=64424509440      # 60GiB
MAP_SIZE_BYTES=1099511627776            # 1024GiB
NUM_GPU_BLOCKS_OVERRIDE=320000

# Decode, 一台 A2 独占
OMNI_CACHE_LAYER_BYTES=32212254720      # 30GiB
MAP_SIZE_BYTES=824633720832             # 768GiB
NUM_GPU_BLOCKS_OVERRIDE=10000
```

如果跑稳、有内存余量，再往上试：

```bash
# Prefill 激进一点
MAP_SIZE_BYTES=1288490188800            # 1200GiB
OMNI_CACHE_LAYER_BYTES=75161927680      # 70GiB

# Decode 激进一点
MAP_SIZE_BYTES=1099511627776            # 1024GiB
OMNI_CACHE_LAYER_BYTES=42949672960      # 40GiB
```

不要在 A2 1.5T 上照搬 P 1500GiB。那会几乎不给 OS、容器、页表、运行时、proxy、日志和临时内存留余量。

**还要注意两个坑**

4P1D omni_cache 模板给 decode 传了 `OMNI_CACHE_DECODE_NUM_GPU_BLOCKS_OVERRIDE`，但 decode 脚本里没有真正转成 `--num-gpu-blocks-override`。如果你希望这个变量生效，需要补：

```bash
export NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-${OMNI_CACHE_DECODE_NUM_GPU_BLOCKS_OVERRIDE:-10000}}"
EXTRA_ARGS="${EXTRA_ARGS} --num-gpu-blocks-override ${NUM_GPU_BLOCKS_OVERRIDE}"
```

另外，A2 原模板里的 A2 专用项要保留，比如 decode 的：

```bash
export OMNI_NPU_USE_DEVICE_COMM_A2A=1
export HCCL_OP_EXPANSION_MODE="AI_CPU"
export LOCAL_WORLD_SIZE=8
```

这些不要被 A3 4P1D 模板覆盖掉。

**如果 P/D 混在同一台 A2**

那就不能用上面的 1024GiB + 768GiB，因为两个 mmap 池会相加。建议先压到：

```bash
Prefill MAP_SIZE_BYTES=687194767360     # 640GiB
Decode  MAP_SIZE_BYTES=412316860416     # 384GiB
```

总共约 1TiB，给 1.5T 机器留足运行余量。

最稳的迁移路线是：先只打开 `ENABLE_OMNI_CACHE=1`，用小 MAP 跑通；再调大 Prefill cache；最后调 Decode cache 和 blocks。不要一上来照搬 A3 的 1500GiB/1000GiB。

> OmniCache有什么用，能提高性能吗

OmniCache 的作用可以简单理解为：把 PD 分离里的 KV Cache 从“只放 NPU HBM”扩展到“主机内存 hugepage 池 + HBM 工作集”。

它主要解决两个问题：

1. **省 HBM**

   长上下文、多轮对话、prefix caching 时，KV Cache 很容易吃满显存。OmniCache 把大量 KV 放到 CPU 内存池里，需要 decode 时再搬回/映射到 HBM，可以让同样的卡承载更长上下文或更多并发。

2. **提升复用**

   多轮对话、重复前缀、长 prompt 复用时，Prefill 产出的 KV 可以在主机内存里持久化，Decode 侧复用这些 KV，减少重复 prefill 或减少 HBM 压力。

能不能提高性能，要看场景：

- **会提高**：长上下文、多轮对话、prompt 重复率高、HBM 被 KV 卡住、并发受显存限制的场景。
- **不一定提高**：短 prompt、无复用、低并发、HBM 本来够用的场景。
- **可能变慢**：单请求极低延迟场景，因为它引入了 host memory、hugepage、KV 搬运/映射、connector 调度等额外路径。

所以它更像是“容量和高并发/长上下文优化”，不是无条件降低单 token 延迟的魔法开关。对于你这个 92B A2 1P1D，如果目标是长上下文或多轮并发，值得试；如果只是短请求 benchmark，收益可能不明显，甚至会有额外开销。

> OMNI_NPU_USE_DEVICE_COMM_A2A这个环境变量是控制什么的？

<details><summary>1 previous message</summary>

> 我查一下仓里这个变量的实际消费点，避免只按名字猜。A2 相关的通信开关通常埋在 patch 或 runtime 侧，得看它被谁读、默认值是什么。
</details>

查到了，它不是全局 A2A 通信总开关，只在 logits processor 里控制一个 DP 组 `all_to_all` 的实现路径。也就是说它主要影响 decode 后处理/采样前后的 logits 重排，而不是 MoE expert dispatch 那类通信。

> 继续

<details><summary>1 previous message</summary>

> 继续。刚才已经定位到了它唯一的源码消费点：`NPULogitsProcessor._get_logits()`，也就是 lm_head/logits 这段通信路径；下面我把它讲清楚。
</details>

`OMNI_NPU_USE_DEVICE_COMM_A2A` 控制的是 **logits 处理阶段的 all-to-all 通信实现路径**，不是一个全局通信总开关。

源码位置在：
[logits_processor.py](./openPangu-2.0-Infer/components/omni-npu/src/omni_npu/v1/layers/logits_processor.py:49)

逻辑是：

```python
if os.environ.get("OMNI_NPU_USE_DEVICE_COMM_A2A", "0") == "1":
    logits = comm_group.device_communicator.all_to_all(
        logits, scatter_dim=0, gather_dim=-1,
    )[:local_n]
else:
    torch.distributed.all_to_all_single(...)
```

也就是说：

- `0`：走 `torch.distributed.all_to_all_single`
- `1`：走 vLLM 的 `device_communicator.all_to_all`

这个路径只在 `lm_head.local_lmhead_parallel` 或 `lm_head.dp_parallel` 开启时生效，用来把各 DP/local rank 上算出来的 logits 做重排/聚合。A2 模板里 decode 默认开了它：
[A2 bf16 模板](./openPangu-2.0-Infer/tools/ansible/template/omni_infer_server_template_performance1P1D_92B_bf16_A2_open.yml:167)

它的目的大概率是：在 A2 decode 的 DP 并行 logits 路径上，用 NPU 平台自己的 device communicator，减少 Python/torch.distributed 通用路径的开销，或者规避 `all_to_all_single` 在这个 shape/后端上的性能问题。

需要注意两点：

1. 它**不控制 MoE expert 的 all-to-all**，也不控制 PD KV 传输。
2. 它只有在 `use_local_comm` 或 `use_dp_comm` 为真时才有影响；普通 TP logits all gather 不走这个分支。

所以建议 A2 decode 保持：

```bash
export OMNI_NPU_USE_DEVICE_COMM_A2A=1
```

如果遇到 logits 结果异常、通信 hang、或 all-to-all 相关报错，可以临时设成 `0` 回退到 `torch.distributed.all_to_all_single` 做对比验证。
