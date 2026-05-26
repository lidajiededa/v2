# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""A layer that samples the next tokens from the model's outputs."""
import torch_npu

from vllm.config.model import LogprobsMode
from vllm.v1.sample.sampler import Sampler as SamplerV1

from omni_npu.sample.ops.topk_topp_sampler import NPUTopKTopPSampler


class NPUSamplerV1(SamplerV1):
    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs"):
        super().__init__(logprobs_mode)
        self.dsa_stream = torch_npu.npu.Stream()
        self.topk_topp_sampler = NPUTopKTopPSampler(logprobs_mode, self.dsa_stream)
        
