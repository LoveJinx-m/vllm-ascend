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
#

import torch
import torch_npu
from einops import rearrange
from vllm.distributed import get_pcp_group
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
from vllm.triton_utils import triton
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionBackend
from vllm_ascend.ops.gdn_execution import GDNExecutionPlan, execute_gdn_core
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
from vllm_ascend.ops.triton.fla.fused_qkvzba_split_reshape import fused_qkvzba_split_reshape_cat
from vllm_ascend.ops.triton.fla.utils import clear_ssm_states
from vllm_ascend.ops.triton.mamba.causal_conv1d import extract_last_width


class AscendGatedDeltaNetAttention(GatedDeltaNetAttention):
    # Cached fused-op availability probe result, shared across all layers so the
    # smoke call runs at most once per process.
    _fused_chunk_available: bool | None = None

    @classmethod
    def _probe_fused_chunk(cls) -> bool:
        """Whether ``torch_npu.npu_chunk_gated_delta_rule`` can actually be used.

        The interface must exist AND a minimal smoke call must succeed (the op is
        unavailable on some CANN builds / devices). Any failure disables the
        fused path so we fall back to the Triton pipeline. The result is cached
        on the class, so only the first profiled layer runs the smoke call.
        """
        if cls._fused_chunk_available is not None:
            return cls._fused_chunk_available

        if not hasattr(torch_npu, "npu_chunk_gated_delta_rule"):
            cls._fused_chunk_available = False
            return False

        # TODO(2026/8/6): The A5‑specific implementation is not available in the official release.
        # Invoking npu_chunk_gated_delta_rule will result in errors.
        # Remove this conditional block after the new A5 CANN package is released.
        try:
            # Minimal smoke call matching the op constraints (Dk == Dv == 128,
            # Nv % Nk == 0). B=1, one short sequence.
            device = torch.npu.current_device()
            dk = dv = 128
            nk, nv, seqlen = 1, 1, 64
            q = torch.zeros((seqlen, nk, dk), dtype=torch.bfloat16, device=device)
            k = torch.zeros((seqlen, nk, dk), dtype=torch.bfloat16, device=device)
            v = torch.zeros((seqlen, nv, dv), dtype=torch.bfloat16, device=device)
            beta = torch.full((seqlen, nv), 0.5, dtype=torch.bfloat16, device=device)
            g = torch.full((seqlen, nv), -0.1, dtype=torch.float32, device=device)
            initial_state = torch.zeros((1, nv, dv, dk), dtype=torch.bfloat16, device=device)
            actual_seq_lengths = torch.tensor([seqlen], dtype=torch.int32, device=device)
            torch_npu.npu_chunk_gated_delta_rule(
                q,
                k,
                v,
                beta=beta,
                initial_state=initial_state,
                actual_seq_lengths=actual_seq_lengths,
                scale=dk**-0.5,
                g=g,
            )
            torch.npu.synchronize()
            cls._fused_chunk_available = True
        except Exception:
            cls._fused_chunk_available = False
        return cls._fused_chunk_available

    @staticmethod
    def _chunk_gated_delta_rule_fused(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused prefill path using ``torch_npu.npu_chunk_gated_delta_rule``.

        Drop-in replacement for the Triton ``chunk_gated_delta_rule`` pipeline
        (chunk_scaled_dot_kkt_fwd + solve_tril + recompute_w_u_fwd + ...).
        The fused CANN operator expects TND layout and does NOT apply q/k L2 norm
        or the chunk-local cumsum of ``g`` internally, so q/k are normalized here
        and the raw ``g`` is passed through.

        Args:
            q, k: ``[1, T, Nk, Dk]``   v: ``[1, T, Nv, Dv]``
            g, beta: ``[1, T, Nv]``    g is fp32 (<=0), beta is (0, 1).
            initial_state: ``[N, Nv, Dv, Dk]`` — same layout as ``ssm_state``,
                no transpose required.
            cu_seqlens: cumulative prefill query start locations ``[N+1]``.
            scale: query scaling factor (``Dk ** -0.5``).

        Returns:
            o: ``[1, T, Nv, Dv]`` and final_state: ``[N, Nv, Dv, Dk]``.
        """
        # TND layout: drop the leading batch dim (batch size is always 1 here).
        q = l2norm_fwd(q).squeeze(0).contiguous()  # [T, Nk, Dk]
        k = l2norm_fwd(k).squeeze(0).contiguous()  # [T, Nk, Dk]
        v = v.squeeze(0).contiguous()  # [T, Nv, Dv]
        g = g.squeeze(0).to(torch.float32).contiguous()  # [T, Nv]
        beta = beta.squeeze(0).to(v.dtype).contiguous()  # [T, Nv]

        # The fused op only supports a bfloat16 initial_state, while ssm_state may
        # be float32 (the recurrent path keeps fp32 state). Cast to bf16 here.
        initial_state = initial_state.to(torch.bfloat16).contiguous()

        # actual_seq_lengths is per-batch sequence length [N] (per the interface
        # doc), derived from the cumulative query_start_loc.
        actual_seq_lengths = torch.diff(cu_seqlens).to(torch.int32)

        o, final_state = torch_npu.npu_chunk_gated_delta_rule(
            q,
            k,
            v,
            beta=beta,
            initial_state=initial_state,
            actual_seq_lengths=actual_seq_lengths,
            scale=scale,
            g=g,
        )
        return o.unsqueeze(0), final_state

    def _split_ba_for_tp(self, ba: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self, "split_ba"):
            return self.split_ba(ba)
        return ba.chunk(2, dim=-1)

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def _warmup_prefill_kernels(self, qkv_or_qkvz: torch.Tensor, v_dim: int) -> None:
        return

    def _warmup_prefill_kernels_v0202(self, mixed_qkv: torch.Tensor) -> None:
        return

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AscendGDNAttentionBackend

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor = None,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)
        if hasattr(self, "in_proj_qkv"):
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = self._split_ba_for_tp(ba)
            b = b.contiguous()
            a = a.contiguous()
        else:
            if not self.gqa_interleaved_layout:
                mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
                num_tokens = mixed_qkvz.size(0)
                qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
                z_size = self.value_dim // self.tp_size
                mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
                z = z.reshape(z.size(0), -1, self.head_v_dim)
                ba, _ = self.in_proj_ba(hidden_states)
                b, a = self._split_ba_for_tp(ba)

                b = b.contiguous()
                a = a.contiguous()
            else:
                projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
                projected_states_ba, _ = self.in_proj_ba(hidden_states)
                num_tokens = projected_states_qkvz.size(0)

                mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(
                    projected_states_qkvz,
                    projected_states_ba,
                    triton.cdiv(self.num_k_heads, self.tp_size),
                    triton.cdiv(self.num_v_heads, self.tp_size),
                    self.head_k_dim,
                    self.head_v_dim,
                )

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.qwen_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
            False,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        out, _ = self.out_proj(core_attn_out)
        if output is not None:
            output[:num_tokens] = out
        return out

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        execute_gdn_core(
            self,
            _COMMON_GDN_EXECUTION_ADAPTER,
            mixed_qkv,
            b,
            a,
            core_attn_out,
        )


class _CommonGDNExecutionAdapter:
    """A2/A3/A5 kernels behind the shared GDN execution seam."""

    split_mixed_non_spec_decode = True

    def initialize_capabilities(self, layer: AscendGatedDeltaNetAttention) -> None:
        del layer
        AscendGatedDeltaNetAttention._probe_fused_chunk()

    def prepare_inputs(
        self,
        plan: GDNExecutionPlan,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            mixed_qkv[: plan.num_actual_tokens],
            b[: plan.num_actual_tokens],
            a[: plan.num_actual_tokens],
        )

    def causal_conv_spec(
        self,
        layer: AscendGatedDeltaNetAttention,
        metadata: GDNAttentionMetadata,
        plan: GDNExecutionPlan,
        mixed_qkv: torch.Tensor,
    ) -> torch.Tensor:
        del plan
        spec_metadata = metadata.spec_decode_metadata.spec_causal_conv1d
        conv_weights = layer.conv1d.weight.view(
            layer.conv1d.weight.size(0),
            layer.conv1d.weight.size(2),
        ).transpose(0, 1)
        output = torch.empty_like(mixed_qkv)
        torch.ops._C_ascend.npu_causal_conv1d_custom(
            output,
            mixed_qkv,
            conv_weights,
            conv_state=layer.kv_cache[0],
            bias_opt=layer.conv1d.bias,
            query_start_loc_opt=spec_metadata.query_start_loc,
            cache_indices_opt=spec_metadata.cache_indices,
            initial_state_mode_opt=None,
            num_accepted_tokens_opt=spec_metadata.num_accepted_tokens,
            activation_mode=1 if layer.activation else 0,
            pad_slot_id=PAD_SLOT_ID,
            run_mode=1,
        )
        return output

    def causal_conv_prefill(
        self,
        layer: AscendGatedDeltaNetAttention,
        metadata: GDNAttentionMetadata,
        mixed_qkv: torch.Tensor,
    ) -> torch.Tensor:
        prefill_metadata = metadata.non_spec_prefill_metadata.causal_conv1d
        conv_weights = layer.conv1d.weight.view(
            layer.conv1d.weight.size(0),
            layer.conv1d.weight.size(2),
        )
        conv_weights_t = conv_weights.transpose(0, 1)
        output = torch.empty_like(mixed_qkv)

        pcp_group = get_pcp_group()
        prefill_cache_indices = None
        all_last_width_prefill_x = None
        if pcp_group.world_size > 1:
            query_start_loc = metadata.non_spec_query_start_loc
            state_indices = metadata.non_spec_state_indices_tensor
            assert query_start_loc is not None
            assert state_indices is not None
            state_len = conv_weights.shape[1] - 1
            num_sequences = query_start_loc.shape[0] - 1
            prefill_offset = max(0, num_sequences - metadata.num_prefills)
            prefill_cache_indices = state_indices[prefill_offset:]
            last_width_prefill_x = extract_last_width(
                mixed_qkv.transpose(0, 1),
                query_start_loc[prefill_offset:],
                state_len,
            )
            all_last_width_prefill_x = pcp_group.all_gather(
                last_width_prefill_x.unsqueeze(0).contiguous(),
                0,
            )
            if pcp_group.rank_in_group > 0 and prefill_cache_indices.shape[0] > 0:
                layer.kv_cache[0][prefill_cache_indices, :state_len, :] = all_last_width_prefill_x[
                    pcp_group.rank_in_group - 1, ...
                ].transpose(-1, -2)

        torch.ops._C_ascend.npu_causal_conv1d_custom(
            output,
            mixed_qkv,
            conv_weights_t,
            conv_state=layer.kv_cache[0],
            bias_opt=layer.conv1d.bias,
            query_start_loc_opt=prefill_metadata.query_start_loc,
            cache_indices_opt=prefill_metadata.cache_indices,
            initial_state_mode_opt=prefill_metadata.initial_state_mode,
            num_accepted_tokens_opt=None,
            activation_mode=1 if layer.activation else 0,
            pad_slot_id=PAD_SLOT_ID,
            run_mode=0,
        )

        if prefill_cache_indices is not None and prefill_cache_indices.shape[0] > 0:
            assert all_last_width_prefill_x is not None
            state_len = conv_weights.shape[1] - 1
            layer.kv_cache[0][prefill_cache_indices, :state_len, :] = all_last_width_prefill_x[-1, ...].transpose(
                -1, -2
            )
        return output

    def causal_conv_decode(
        self,
        layer: AscendGatedDeltaNetAttention,
        metadata: GDNAttentionMetadata,
        mixed_qkv: torch.Tensor,
    ) -> torch.Tensor:
        decode_metadata = metadata.non_spec_decode_metadata.causal_conv1d
        conv_weights = layer.conv1d.weight.view(
            layer.conv1d.weight.size(0),
            layer.conv1d.weight.size(2),
        ).transpose(0, 1)
        output = torch.empty_like(mixed_qkv)
        torch.ops._C_ascend.npu_causal_conv1d_custom(
            output,
            mixed_qkv,
            conv_weights,
            conv_state=layer.kv_cache[0],
            bias_opt=layer.conv1d.bias,
            query_start_loc_opt=decode_metadata.query_start_loc,
            cache_indices_opt=decode_metadata.cache_indices,
            initial_state_mode_opt=None,
            num_accepted_tokens_opt=None,
            activation_mode=1 if layer.activation else 0,
            pad_slot_id=PAD_SLOT_ID,
            run_mode=1,
        )
        return output

    def gating(
        self,
        layer: AscendGatedDeltaNetAttention,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return DeviceOperator.fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)

    def recurrent_spec(
        self,
        layer: AscendGatedDeltaNetAttention,
        metadata: GDNAttentionMetadata,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        spec_metadata = metadata.spec_decode_metadata
        query = l2norm_fwd(query)
        key = l2norm_fwd(key)
        return torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
            query=query.squeeze(0),
            key=key.squeeze(0),
            value=value.squeeze(0),
            g=g.squeeze(0),
            beta=beta.squeeze(0),
            state=layer.kv_cache[1],
            scale=key.shape[-1] ** -0.5,
            actual_seq_lengths=spec_metadata.actual_seq_lengths,
            ssm_state_indices=metadata.spec_state_indices_tensor.flatten(),
            num_accepted_tokens=spec_metadata.spec_causal_conv1d.num_accepted_tokens.to(torch.int32),
        ).unsqueeze(0)

    def recurrent_decode(
        self,
        layer: AscendGatedDeltaNetAttention,
        metadata: GDNAttentionMetadata,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        mixed_batch: bool,
    ) -> torch.Tensor:
        query = l2norm_fwd(query)
        key = l2norm_fwd(key)
        state_indices = metadata.non_spec_state_indices_tensor
        assert state_indices is not None
        if mixed_batch:
            state_indices = state_indices[: metadata.num_decodes]
        return torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
            query=query.squeeze(0),
            key=key.squeeze(0),
            value=value.squeeze(0),
            g=g.squeeze(0),
            beta=beta.squeeze(0),
            state=layer.kv_cache[1],
            scale=key.shape[-1] ** -0.5,
            actual_seq_lengths=metadata.non_spec_decode_metadata.actual_seq_lengths,
            ssm_state_indices=state_indices,
        ).unsqueeze(0)

    def recurrent_prefill(
        self,
        layer: AscendGatedDeltaNetAttention,
        metadata: GDNAttentionMetadata,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        excludes_decode: bool,
    ) -> torch.Tensor:
        del excludes_decode
        state_indices = metadata.prefill_state_indices
        has_initial_state = metadata.prefill_has_initial_state
        query_start_loc = metadata.prefill_query_start_loc
        assert state_indices is not None
        assert has_initial_state is not None
        assert query_start_loc is not None
        ssm_state = layer.kv_cache[1]

        use_fused_chunk = (
            AscendGatedDeltaNetAttention._fused_chunk_available is True and get_pcp_group().world_size == 1
        )
        if use_fused_chunk:
            initial_state = ssm_state[state_indices]
            clear_ssm_states(initial_state, has_initial_state)
            output, last_state = AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                cu_seqlens=query_start_loc,
                scale=key.shape[-1] ** -0.5,
            )
            ssm_state[state_indices] = last_state.to(ssm_state.dtype)
            return output

        initial_state = ssm_state[state_indices].transpose(-1, -2).contiguous()
        clear_ssm_states(initial_state, has_initial_state)
        output, last_state = chunk_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=query_start_loc,
            prebuilt_meta=metadata.non_spec_prefill_metadata.chunk,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        ssm_state[state_indices] = last_state.transpose(-1, -2).contiguous().to(ssm_state.dtype)
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
        merged = torch.empty(
            (1, plan.num_actual_tokens, *spec_output.shape[2:]),
            dtype=non_spec_output.dtype,
            device=non_spec_output.device,
        )
        merged.index_copy_(1, plan.spec_token_indices, spec_output)
        merged.index_copy_(1, plan.non_spec_token_indices, non_spec_output)
        output[: plan.num_actual_tokens] = merged.squeeze(0)

    def write_single_output(
        self,
        output: torch.Tensor,
        plan: GDNExecutionPlan,
        value: torch.Tensor,
    ) -> None:
        output[: plan.num_actual_tokens] = value.squeeze(0)

    def finalize_output(
        self,
        output: torch.Tensor,
        metadata: GDNAttentionMetadata,
        plan: GDNExecutionPlan,
    ) -> None:
        del output, metadata, plan


_COMMON_GDN_EXECUTION_ADAPTER = _CommonGDNExecutionAdapter()
