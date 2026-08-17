#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from collections.abc import Iterable
# mypy: ignore-errors


import torch
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import chunk_gated_delta_rule_310
from vllm_ascend._310p.ops.fla.fused_gdn_gating import fused_gdn_gating_pytorch
from vllm_ascend._310p.ops.fla.l2norm import l2norm_310p
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.gdn_execution import GDNExecutionPlan, execute_gdn_core
from vllm_ascend.utils import enable_sp


def _zero_padded_tokens(
    tensor: torch.Tensor,
    valid_tokens: torch.Tensor,
    token_dim: int,
) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor

    token_count = tensor.shape[token_dim]
    if token_count == 0:
        return tensor

    positions = torch.arange(
        token_count,
        device=tensor.device,
        dtype=valid_tokens.dtype,
    )
    valid_mask = positions < valid_tokens.to(device=tensor.device)
    mask_shape = [1] * tensor.ndim
    mask_shape[token_dim] = token_count
    return tensor * valid_mask.reshape(mask_shape).to(dtype=tensor.dtype)


def _flatten_state_indices(
    ssm_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_tokens: int,
) -> torch.Tensor:
    if ssm_state_indices.ndim == 1:
        return ssm_state_indices[:total_tokens].to(torch.int32).contiguous()

    num_seqs = (cu_seqlens[1:] - cu_seqlens[:-1]).shape[0]
    seq_lens = cu_seqlens[1 : num_seqs + 1] - cu_seqlens[:num_seqs]
    ssm_state_indices = ssm_state_indices[:num_seqs]

    # Uniform spec-decode ACL graph uses fixed q_len per request; reshape avoids
    # NPU masked_select which breaks stream capture (aclnnMaskedSelect / 107027).
    if _EXTRA_CTX.capturing or (seq_lens.numel() > 0 and torch.all(seq_lens == seq_lens[0])):
        q_per_seq = ssm_state_indices.shape[1]
        flat = ssm_state_indices[:, :q_per_seq].reshape(-1)
        return flat[:total_tokens].to(torch.int32).contiguous()

    # Eager mixed batches with variable seq_lens: compact on CPU, copy back async.
    ssm_cpu = ssm_state_indices.cpu()
    seq_lens_cpu = seq_lens.cpu()
    q_per_seq = ssm_cpu.shape[1]
    positions = torch.arange(q_per_seq)
    valid = positions.unsqueeze(0) < seq_lens_cpu.unsqueeze(1)
    flat_cpu = ssm_cpu.masked_select(valid).to(torch.int32).contiguous()[:total_tokens]
    if not flat_cpu.is_pinned:
        flat_cpu = flat_cpu.pin_memory()
    flat_dev = torch.empty(flat_cpu.numel(), dtype=torch.int32, device=ssm_state_indices.device)
    flat_dev.copy_(flat_cpu, non_blocking=True)
    return flat_dev.contiguous()


def _mask_padded_recurrent_accepted_tokens(
    num_accepted_tokens: torch.Tensor,
    actual_seq_lengths: torch.Tensor,
) -> torch.Tensor:
    accepted_tokens = num_accepted_tokens[: actual_seq_lengths.shape[0]].to(torch.int32).contiguous()
    return torch.where(
        actual_seq_lengths > 0,
        accepted_tokens,
        torch.zeros_like(accepted_tokens),
    ).contiguous()


def npu_recurrent_gated_delta_rule_310(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    if use_qk_l2norm_in_kernel:
        q = l2norm_310p(q)
        k = l2norm_310p(k)

    total_tokens = v.shape[1]
    flat_state_indices = _flatten_state_indices(ssm_state_indices, cu_seqlens, total_tokens)
    actual_seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32).contiguous()
    flat_state_indices = torch.clamp_min(
        flat_state_indices,
        0,
    ).contiguous()
    accepted_tokens = None
    if num_accepted_tokens is not None:
        accepted_tokens = _mask_padded_recurrent_accepted_tokens(
            num_accepted_tokens,
            actual_seq_lengths,
        )

    out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310(
        query=q.squeeze(0).to(torch.float16).contiguous(),
        key=k.squeeze(0).to(torch.float16).contiguous(),
        value=v.squeeze(0).to(torch.float16).contiguous(),
        g=None if g is None else g.squeeze(0).to(torch.float32).contiguous(),
        gk=None,
        beta=beta.squeeze(0).to(torch.float16).contiguous(),
        state=state,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=flat_state_indices,
        num_accepted_tokens=accepted_tokens,
        scale_value=k.shape[-1] ** -0.5,
    ).unsqueeze(0)
    return out


def _310p_get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
    conv_state_dtype, _ = _original_get_state_dtype(self)
    return conv_state_dtype, torch.float16


_original_get_state_dtype = GatedDeltaNetAttention.get_state_dtype


def _merge_spec_and_non_spec_outputs_310(
    core_attn_out: torch.Tensor,
    num_actual_tokens: int,
    spec_token_indx: torch.Tensor,
    non_spec_token_indx: torch.Tensor,
    core_attn_out_spec: torch.Tensor,
    core_attn_out_non_spec: torch.Tensor,
) -> None:
    """Merge spec/non-spec GDN outputs back into the batch layout.

    Avoid NPU ``index_copy_`` (IndexPutV2) which fails on some layouts; use
    direct indexing instead. Validate lengths so mixed prefill+spec batches
    do not pass mismatched tensors from spec ops.
    """
    spec_out = core_attn_out_spec.squeeze(0)
    non_spec_out = core_attn_out_non_spec.squeeze(0)
    n_spec = spec_token_indx.numel()
    n_non_spec = non_spec_token_indx.numel()
    if spec_out.shape[0] != n_spec:
        raise RuntimeError(f"GDN spec output length {spec_out.shape[0]} != spec_token_indx {n_spec}")
    if non_spec_out.shape[0] != n_non_spec:
        raise RuntimeError(f"GDN non-spec output length {non_spec_out.shape[0]} != non_spec_token_indx {n_non_spec}")
    out = core_attn_out[:num_actual_tokens]
    out[spec_token_indx] = spec_out
    out[non_spec_token_indx] = non_spec_out


class AscendGatedDeltaNetAttention310(GatedDeltaNetAttention):
    get_state_dtype = _310p_get_state_dtype

    def get_attn_backend(self):
        from vllm_ascend._310p.ops.gdn_attn_builder_310 import (
            AscendGDNAttentionBackend310,
        )

        return AscendGDNAttentionBackend310

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        execute_gdn_core(
            self,
            _GDN_310_EXECUTION_ADAPTER,
            mixed_qkv,
            b,
            a,
            core_attn_out,
        )


class _GDN310ExecutionAdapter:
    """310P kernels and state layouts behind the shared GDN execution seam."""

    split_mixed_non_spec_decode = False

    def initialize_capabilities(self, layer: AscendGatedDeltaNetAttention310) -> None:
        del layer

    def prepare_inputs(
        self,
        plan: GDNExecutionPlan,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if enable_sp():
            return mixed_qkv, b, a
        return (
            mixed_qkv[: plan.num_actual_tokens],
            b[: plan.num_actual_tokens],
            a[: plan.num_actual_tokens],
        )

    @staticmethod
    def _conv_weights(layer: AscendGatedDeltaNetAttention310) -> torch.Tensor:
        return layer.conv1d.weight.view(
            layer.conv1d.weight.size(0),
            layer.conv1d.weight.size(2),
        ).transpose(0, 1)

    def causal_conv_spec(
        self,
        layer: AscendGatedDeltaNetAttention310,
        metadata: GDNAttentionMetadata,
        plan: GDNExecutionPlan,
        mixed_qkv: torch.Tensor,
    ) -> torch.Tensor:
        spec_metadata = metadata.spec_decode_metadata.spec_causal_conv1d
        if plan.is_uniform_spec_decode:
            mixed_qkv = _zero_padded_tokens(
                mixed_qkv,
                spec_metadata.query_start_loc[-1],
                token_dim=0,
            )
        return torch.ops._C_ascend.npu_causal_conv1d_310(
            mixed_qkv,
            self._conv_weights(layer),
            bias=layer.conv1d.bias,
            conv_states=layer.kv_cache[0],
            query_start_loc=spec_metadata.query_start_loc,
            cache_indices=spec_metadata.cache_indices,
            initial_state_mode=None,
            num_accepted_tokens=spec_metadata.num_accepted_tokens,
            activation_mode=1 if layer.activation else 0,
            pad_slot_id=PAD_SLOT_ID,
            run_mode=1,
        )

    def causal_conv_prefill(
        self,
        layer: AscendGatedDeltaNetAttention310,
        metadata: GDNAttentionMetadata,
        mixed_qkv: torch.Tensor,
    ) -> torch.Tensor:
        state_indices = metadata.non_spec_state_indices_tensor
        assert state_indices is not None
        return torch.ops._C_ascend.npu_causal_conv1d_310(
            mixed_qkv,
            self._conv_weights(layer),
            bias=layer.conv1d.bias,
            conv_states=layer.kv_cache[0],
            query_start_loc=metadata.non_spec_query_start_loc,
            cache_indices=state_indices,
            initial_state_mode=metadata.has_initial_state,
            num_accepted_tokens=None,
            activation_mode=1 if layer.activation else 0,
            pad_slot_id=PAD_SLOT_ID,
            run_mode=0,
        )

    def causal_conv_decode(
        self,
        layer: AscendGatedDeltaNetAttention310,
        metadata: GDNAttentionMetadata,
        mixed_qkv: torch.Tensor,
    ) -> torch.Tensor:
        state_indices = metadata.non_spec_state_indices_tensor
        assert state_indices is not None
        return torch.ops._C_ascend.npu_causal_conv1d_310(
            mixed_qkv,
            self._conv_weights(layer),
            bias=layer.conv1d.bias,
            conv_states=layer.kv_cache[0],
            query_start_loc=None,
            cache_indices=state_indices[: metadata.num_actual_tokens],
            initial_state_mode=None,
            num_accepted_tokens=None,
            activation_mode=1 if layer.activation else 0,
            pad_slot_id=PAD_SLOT_ID,
            run_mode=1,
        )

    def gating(
        self,
        layer: AscendGatedDeltaNetAttention310,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return fused_gdn_gating_pytorch(layer.A_log, a, b, layer.dt_bias)

    def recurrent_spec(
        self,
        layer: AscendGatedDeltaNetAttention310,
        metadata: GDNAttentionMetadata,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        spec_metadata = metadata.spec_decode_metadata.spec_causal_conv1d
        return npu_recurrent_gated_delta_rule_310(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            state=layer.kv_cache[1],
            cu_seqlens=metadata.spec_query_start_loc[: metadata.num_spec_decodes + 1],
            ssm_state_indices=metadata.spec_state_indices_tensor,
            num_accepted_tokens=spec_metadata.num_accepted_tokens,
            use_qk_l2norm_in_kernel=True,
        )

    def recurrent_decode(
        self,
        layer: AscendGatedDeltaNetAttention310,
        metadata: GDNAttentionMetadata,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        mixed_batch: bool,
    ) -> torch.Tensor:
        del mixed_batch
        return npu_recurrent_gated_delta_rule_310(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            state=layer.kv_cache[1],
            cu_seqlens=metadata.non_spec_query_start_loc,
            ssm_state_indices=metadata.non_spec_state_indices_tensor,
            use_qk_l2norm_in_kernel=True,
        )

    def recurrent_prefill(
        self,
        layer: AscendGatedDeltaNetAttention310,
        metadata: GDNAttentionMetadata,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        excludes_decode: bool,
    ) -> torch.Tensor:
        del excludes_decode
        state_indices = metadata.non_spec_state_indices_tensor
        has_initial_state = metadata.has_initial_state
        assert state_indices is not None
        assert has_initial_state is not None
        initial_state = layer.kv_cache[1][state_indices].contiguous()
        initial_state[~has_initial_state, ...] = 0
        output, last_state = chunk_gated_delta_rule_310(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=metadata.non_spec_query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        layer.kv_cache[1][state_indices] = last_state.to(layer.kv_cache[1].dtype)
        return output

    def write_mixed_output(
        self,
        output: torch.Tensor,
        plan: GDNExecutionPlan,
        spec_output: torch.Tensor,
        non_spec_output: torch.Tensor,
    ) -> None:
        assert plan.spec_token_indices is not None
        assert plan.non_spec_token_indices is not None
        _merge_spec_and_non_spec_outputs_310(
            output,
            plan.num_actual_tokens,
            plan.spec_token_indices,
            plan.non_spec_token_indices,
            spec_output,
            non_spec_output,
        )

    def write_single_output(
        self,
        output: torch.Tensor,
        plan: GDNExecutionPlan,
        value: torch.Tensor,
    ) -> None:
        value = value.squeeze(0)
        if enable_sp():
            value = value[: plan.num_actual_tokens]
        output[: plan.num_actual_tokens] = value

    def finalize_output(
        self,
        output: torch.Tensor,
        metadata: GDNAttentionMetadata,
        plan: GDNExecutionPlan,
    ) -> None:
        if plan.is_uniform_spec_decode:
            output.copy_(
                _zero_padded_tokens(
                    output,
                    metadata.spec_decode_metadata.spec_causal_conv1d.query_start_loc[-1],
                    token_dim=0,
                )
            )


_GDN_310_EXECUTION_ADAPTER = _GDN310ExecutionAdapter()
