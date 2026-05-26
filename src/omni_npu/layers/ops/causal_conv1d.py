# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
import torch_npu
import torch.nn.functional as F
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

def custom_depthwise_conv1d(x_t: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None):
    """
    A PyTorch-native 1D depthwise convolution that bypasses the AI_CORE.
    x_t: [batch, seqlen, dim] (Optimized contiguous layout)
    weight: [dim, width] or [dim, 1, width]
    bias: [dim] or None

    Returns: [batch, seqlen - width + 1, dim]
    """
    batch, seqlen, dim = x_t.shape
    width = weight.shape[-1]
    out_len = seqlen - width + 1

    # 1. Reshape weight to [width, dim] for native broadcasting
    w = weight.view(dim, width).transpose(0, 1).contiguous()
    x_t = x_t.contiguous()

    # 2. Unroll the sliding window (Vector Core MACs)
    out_t = x_t[:, 0:out_len, :] * w[0]
    for i in range(1, width):
        out_t += x_t[:, i : i + out_len, :] * w[i]

    # 3. Add bias if present
    if bias is not None:
        out_t += bias

    # Return in [batch, out_len, dim] layout to avoid redundant transposes downstream
    return out_t

def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = "silu",
):
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    dtype_in = x.dtype
    x = x.to(weight.dtype)

    # 1. Auto-align x to [seqlen, dim] natively
    if x.shape[0] == weight.shape[-2]: 
        x = x.transpose(0, 1)
        
    x_for_cat = x.unsqueeze(0) # [1, seqlen, dim]
    batch, seqlen, dim = x_for_cat.shape
    width = weight.shape[-1]

    # 2. Auto-align initial states to [1, state_len, dim]
    if initial_states is None:
        pad_tensor = torch.zeros((1, width - 1, dim), dtype=x.dtype, device=x.device)
        x_padded_native = torch.cat([pad_tensor, x_for_cat], dim=1)
    else:
        # Check if state arrived as [1, dim, state_len] instead of [1, state_len, dim]
        if initial_states.shape[1] == dim:
            initial_t = initial_states.transpose(1, 2)
        else:
            initial_t = initial_states
        x_padded_native = torch.cat([initial_t, x_for_cat], dim=1)

    # 3. Forward to our optimized layout-native kernel
    out_t = custom_depthwise_conv1d(x_padded_native, weight, bias) # [1, seqlen, dim]

    # 4. Auto-align cache write-back
    if return_final_states:
        final_states = x_padded_native[:, -(width - 1):, :].to(dtype_in)
        if final_states_out is not None:
            # If destination buffer expects [1, dim, state_len]
            if final_states_out.shape[1] == dim:
                final_states_out.copy_(final_states.transpose(1, 2))
            else:
                final_states_out.copy_(final_states)

    out = out_t.squeeze(0) # [seqlen, dim]
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    **kwargs # swallows other legacy args like num_accepted_tokens, pad_slot_id, etc.
):
    """
    Combined Update + Ref function. 
    Guarantees shape alignment regardless of incoming legacy wrapper logic.
    """
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    # 1. Force x into [batch, 1, dim] layout 
    if x.dim() == 2:
        x_t = x.unsqueeze(1) 
    elif x.dim() == 3:
        if x.shape[1] == 1:
            x_t = x # Already [batch, 1, dim]
        else:
            x_t = x.transpose(1, 2).contiguous() # Was [batch, dim, 1], fix it
    
    dim = x_t.shape[2]
    width = weight.shape[-1]
    state_len = width - 1

    # 2. Extract and Force conv_state into [batch, state_len, dim]
    batch_indices = torch.clamp(conv_state_indices, 0)
    state_slice = conv_state[batch_indices] 
    
    if state_slice.shape[1] == dim:
        # Cache natively stored as [batch, dim, state_len]. Fix it.
        state_t = state_slice.transpose(1, 2)
    else:
        # Already [batch, state_len, dim]
        state_t = state_slice

    # 3. Concatenate securely (Both are strictly [batch, L, dim] here)
    x_new_t = torch.cat([state_t, x_t], dim=1).to(weight.dtype)

    # 4. Extract new state and write back to cache safely
    new_state_t = x_new_t[:, -state_len:, :]
    if state_slice.shape[1] == dim:
        # Revert back to [batch, dim, state_len] for storage
        new_state = new_state_t.transpose(1, 2)
    else:
        new_state = new_state_t
        
    torch_npu.npu_scatter_nd_update_(conv_state, conv_state_indices.unsqueeze(1), new_state)

    # 5. Run optimized native convolution
    out_t = custom_depthwise_conv1d(x_new_t, weight, bias) # [batch, 1, dim]
    
    # 6. Format output natively
    out = out_t.squeeze(1) # [batch, dim]
    if activation is not None:
        out = F.silu(out)
        
    return out.to(x.dtype)

def causal_conv1d_fn_bubble(
    x: torch.Tensor, # Natively accepts [total_seqlen, dim]
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor, 
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = -1,
    metadata=None,
    validate_data=False,
    seq_lens: torch.Tensor | None = None, 
):
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    
    bias = bias.contiguous() if bias is not None else None

    if seq_lens is not None:
        seqlens_list = seq_lens.tolist()
    else:
        seqlens_tensor = query_start_loc[1:] - query_start_loc[:-1]
        seqlens_list = seqlens_tensor.tolist()

    # Split natively along seqlen dim! (dim=0)
    splits = torch.split(x, seqlens_list, dim=0)
    out_ref_b = []
    
    for i in range(len(seqlens_list)):
        x_s = splits[i] # [seqlen, dim]
        if cache_indices[i] == pad_slot_id:
            continue
        
        # conv_states[cache_indices[i]] is [dim, state_len]
        # unsqueeze(0) makes it [1, dim, state_len]
        state_slice = conv_states[cache_indices[i]].unsqueeze(0)
        
        out_b, _ = causal_conv1d_ref(
            x_s,
            weight,
            bias,
            activation=activation,
            return_final_states=True,
            final_states_out=state_slice,
            initial_states=state_slice if has_initial_state[i] else None
        )
        out_ref_b.append(out_b)
        
    out_ref_tensor = torch.cat(out_ref_b, dim=0)
    return out_ref_tensor

def causal_conv1d_fn(
    x: torch.Tensor, 
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    seqlens_list: list[int],                   # ADDED: Native list
    cache_indices_list: list[int],             # ADDED: Native list
    has_initial_state_list: list[bool],        # ADDED: Native list
    activation: str | None = "silu",
    pad_slot_id: int = -1,
    metadata=None,
    validate_data=False,
):
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    
    bias = bias.contiguous() if bias is not None else None

    # We now split natively using the pre-computed Python list! 
    # Zero NPU sync required here.
    splits = torch.split(x, seqlens_list, dim=0)
    out_ref_b = []
    
    for i in range(len(seqlens_list)):
        x_s = splits[i] # [seqlen, dim]
        
        # Pure Python integer comparison
        if cache_indices_list[i] == pad_slot_id or seqlens_list[i] == 0:
            continue
        
        # Native device indexing, but without conditional branching blocks
        state_slice = conv_states[cache_indices_list[i]].unsqueeze(0)
        
        # Pure Python boolean check
        initial_s = state_slice if has_initial_state_list[i] else None
        
        out_b, _ = causal_conv1d_ref(
            x_s,
            weight,
            bias,
            activation=activation,
            return_final_states=True,
            final_states_out=state_slice,
            initial_states=initial_s
        )
        out_ref_b.append(out_b)
        
    out_ref_tensor = torch.cat(out_ref_b, dim=0)
    return out_ref_tensor
