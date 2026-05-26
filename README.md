# vllm-pangu-v2-moe

Adapter package for running OpenPangu-505B `PanguV2MoEForCausalLM` on
vLLM/vllm-ascend 0.18.0.

It registers the missing model architecture, normalizes the model config, and
bundles the omni-npu Pangu V2 MoE implementation so the runtime image does not
need a separate omni-npu install.

See `docs/ADAPTATION.md` for the 4-node `DP=4, TP=8` launch flow.
中文部署和 vLLM 启动逻辑说明见 `docs/PANGU_V2_MOE_CN.md`。
718B 适配 PR 参考见 `docs/OPENPANGU_ULTRA_718B_REFERENCE_CN.md`。
直接修改 vLLM/vllm-ascend 源码或维护 patch 的步骤见 `docs/SOURCE_PATCH_ADAPTATION_CN.md`。
