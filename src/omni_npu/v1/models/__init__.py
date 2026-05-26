from vllm import ModelRegistry


def register_models():
    ModelRegistry.register_model(
        "DeepseekV32ForCausalLM",
        "omni_npu.v1.models.deepseek.deepseek_v3:DeepseekV3ForCausalLM")
    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM",
        "omni_npu.v1.models.deepseek.deepseek_v3:DeepseekV3ForCausalLM")
    ModelRegistry.register_model(
        "DeepSeekMTPModel",
        "omni_npu.v1.models.deepseek.deepseek_mtp:DeepSeekMTP")
    ModelRegistry.register_model(
        "Qwen2VLForConditionalGeneration",
        "omni_npu.v1.models.qwen.qwen2_vl:NpuQwen2VLForConditionalGeneration")
    ModelRegistry.register_model(
        "Qwen3ForCausalLM",
        "omni_npu.v1.models.qwen.qwen3:Qwen3ForCausalLM")
    ModelRegistry.register_model(
        "Qwen3MoeForCausalLM",
        "omni_npu.v1.models.qwen.qwen3_moe:Qwen3MoeForCausalLM")
    ModelRegistry.register_model(
        "Qwen3VLForConditionalGeneration",
        "omni_npu.v1.models.qwen.qwen3_vl:Qwen3VLForConditionalGeneration")
    ModelRegistry.register_model(
        "Qwen3VLMoeForConditionalGeneration",
        "omni_npu.v1.models.qwen.qwen3_vl_moe:Qwen3VLMoeForConditionalGeneration")
    ModelRegistry.register_model(
        "Qwen3NextForCausalLM",
        "omni_npu.v1.models.qwen.qwen3_next:Qwen3NextForCausalLM")
    ModelRegistry.register_model(
        "Qwen3_5ForCausalLM",
        "omni_npu.v1.models.qwen.qwen3_5:Qwen3_5ForCausalLM")
    ModelRegistry.register_model(
        "Qwen3_5MoeForCausalLM",
        "omni_npu.v1.models.qwen.qwen3_5:Qwen3_5MoeForCausalLM")
    ModelRegistry.register_model(
        "Qwen3_5ForConditionalGeneration",
        "omni_npu.v1.models.qwen.qwen3_5:Qwen3_5ForConditionalGeneration")
    ModelRegistry.register_model(
        "Qwen3_5MoeForConditionalGeneration",
        "omni_npu.v1.models.qwen.qwen3_5:Qwen3_5MoeForConditionalGeneration")
    ModelRegistry.register_model(
        "PanguUltraMoEForCausalLM",
        "omni_npu.v1.models.pangu.pangu_ultra_moe:PanguUltraMoEForCausalLM")
    ModelRegistry.register_model(
        "Glm4MoeForCausalLM",
        "omni_npu.v1.models.glm.glm4_moe:Glm4MoeForCausalLM")
    ModelRegistry.register_model(
        "Glm4MoeMTPModel",
        "omni_npu.v1.models.glm.glm4_moe_mtp:Glm4MoeMTP")
    ModelRegistry.register_model(
        "GptOssForCausalLM",
        "omni_npu.v1.models.gpt_oss.gpt_oss:GptOssForCausalLM")
    ModelRegistry.register_model(
        "OpenPanguMTPModel",
        "omni_npu.v1.models.pangu.pangu_ultra_moe_mtp:OpenPanguMTP")
    ModelRegistry.register_model(
        "PanguProMoEV2ForCausalLM",
        "omni_npu.v1.models.pangu.pangu_pro_moe:PanguProMoEV2ForCausalLM")
    ModelRegistry.register_model(
        "KimiK25ForConditionalGeneration",
        "omni_npu.vllm_patches.patches.models.kimi.kimi_k25:KimiK25ForConditionalGeneration")
    ModelRegistry.register_model(
        "GlmMoeDsaForCausalLM",
        "omni_npu.v1.models.deepseek.deepseek_v3:DeepseekV3ForCausalLM")
    ModelRegistry.register_model(
        "MiniMaxM2ForCausalLM",
        "omni_npu.v1.models.minimax.minimax_m2:MiniMaxM2ForCausalLM")
    ModelRegistry.register_model(
        "Eagle3MiniMaxM2ForCausalLM",
        "vllm.model_executor.models.llama_eagle3:Eagle3LlamaForCausalLM")
    import os

    if (
        int(os.getenv("RANDOM_MODE", default='0'))
        or int(os.getenv("CAPTURE_MODE", default='0'))
        or int(os.getenv("REPLAY_MODE", default='0'))
    ):
        from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
        from omni_npu.v1.models.mock.mock import mock_model_class_factory

        ModelRegistry.register_model(
            "Qwen2ForCausalLM",
            mock_model_class_factory(Qwen2ForCausalLM))
