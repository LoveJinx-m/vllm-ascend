# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.attention.utils import maybe_save_kv_layer_to_connector


@dataclass(frozen=True)
class GDNExecutionPlan:
    """Builder-owned request classification consumed by GDN attention.

    Counts remain the source of truth for real work. Graph padding may extend
    request-level tensors, but it must not change the phases executed here.
    """

    num_actual_tokens: int
    num_prefills: int
    num_decode_tokens: int
    num_decodes: int
    num_spec_decodes: int
    spec_token_indices: torch.Tensor | None
    non_spec_token_indices: torch.Tensor | None

    @property
    def has_spec_decode(self) -> bool:
        return self.num_spec_decodes > 0

    @property
    def has_prefill(self) -> bool:
        return self.num_prefills > 0

    @property
    def has_decode(self) -> bool:
        return self.num_decodes > 0

    @property
    def has_non_spec(self) -> bool:
        return self.has_prefill or self.has_decode

    @property
    def is_uniform_spec_decode(self) -> bool:
        return self.has_spec_decode and not self.has_non_spec

    @property
    def has_mixed_non_spec(self) -> bool:
        return not self.has_spec_decode and self.has_prefill and self.has_decode

    @classmethod
    def from_metadata(cls, metadata: GDNAttentionMetadata) -> GDNExecutionPlan:
        counts = (
            metadata.num_actual_tokens,
            metadata.num_prefills,
            metadata.num_decode_tokens,
            metadata.num_decodes,
            metadata.num_spec_decodes,
        )
        if any(count < 0 for count in counts):
            raise ValueError(f"GDN execution counts must be non-negative, got {counts}")

        has_spec_decode = metadata.num_spec_decodes > 0
        if has_spec_decode:
            if metadata.spec_token_indx is None or metadata.non_spec_token_indx is None:
                raise ValueError("Speculative GDN execution requires spec and non-spec token indices")
        elif metadata.spec_token_indx is not None or metadata.non_spec_token_indx is not None:
            raise ValueError("Non-speculative GDN execution must not carry token partition indices")

        return cls(
            num_actual_tokens=metadata.num_actual_tokens,
            num_prefills=metadata.num_prefills,
            num_decode_tokens=metadata.num_decode_tokens,
            num_decodes=metadata.num_decodes,
            num_spec_decodes=metadata.num_spec_decodes,
            spec_token_indices=metadata.spec_token_indx,
            non_spec_token_indices=metadata.non_spec_token_indx,
        )


class GDNExecutionAdapter(Protocol):
    """Narrow hardware seam used by the shared GDN execution flow."""

    split_mixed_non_spec_decode: bool

    def initialize_capabilities(self, layer: Any) -> None: ...

    def prepare_inputs(
        self,
        plan: GDNExecutionPlan,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def causal_conv_spec(
        self,
        layer: Any,
        metadata: GDNAttentionMetadata,
        plan: GDNExecutionPlan,
        mixed_qkv: torch.Tensor,
    ) -> torch.Tensor: ...

    def causal_conv_prefill(
        self,
        layer: Any,
        metadata: GDNAttentionMetadata,
        mixed_qkv: torch.Tensor,
    ) -> torch.Tensor: ...

    def causal_conv_decode(
        self,
        layer: Any,
        metadata: GDNAttentionMetadata,
        mixed_qkv: torch.Tensor,
    ) -> torch.Tensor: ...

    def gating(
        self,
        layer: Any,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def recurrent_spec(
        self,
        layer: Any,
        metadata: GDNAttentionMetadata,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor: ...

    def recurrent_decode(
        self,
        layer: Any,
        metadata: GDNAttentionMetadata,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        mixed_batch: bool,
    ) -> torch.Tensor: ...

    def recurrent_prefill(
        self,
        layer: Any,
        metadata: GDNAttentionMetadata,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        excludes_decode: bool,
    ) -> torch.Tensor: ...

    def write_mixed_output(
        self,
        output: torch.Tensor,
        plan: GDNExecutionPlan,
        spec_output: torch.Tensor,
        non_spec_output: torch.Tensor,
    ) -> None: ...

    def write_single_output(
        self,
        output: torch.Tensor,
        plan: GDNExecutionPlan,
        value: torch.Tensor,
    ) -> None: ...

    def finalize_output(
        self,
        output: torch.Tensor,
        metadata: GDNAttentionMetadata,
        plan: GDNExecutionPlan,
    ) -> None: ...


def attach_gdn_execution_plan(metadata: GDNAttentionMetadata) -> GDNAttentionMetadata:
    metadata.execution_plan = GDNExecutionPlan.from_metadata(metadata)
    return metadata


def _get_execution_plan(metadata: GDNAttentionMetadata) -> GDNExecutionPlan:
    # Compatibility for metadata assembled directly by tests or external model
    # patches. Production metadata always receives the plan in the builder.
    plan = getattr(metadata, "execution_plan", None)
    if plan is None:
        return GDNExecutionPlan.from_metadata(metadata)
    if not isinstance(plan, GDNExecutionPlan):
        raise TypeError(f"Expected GDNExecutionPlan, got {type(plan).__name__}")
    return plan


def _partition_tensor(
    tensor: torch.Tensor,
    plan: GDNExecutionPlan,
    token_dim: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not plan.has_spec_decode:
        return None, tensor
    if plan.is_uniform_spec_decode:
        return tensor, None

    assert plan.spec_token_indices is not None
    assert plan.non_spec_token_indices is not None
    return (
        tensor.index_select(token_dim, plan.spec_token_indices),
        tensor.index_select(token_dim, plan.non_spec_token_indices),
    )


def _get_gdn_metadata(prefix: str) -> GDNAttentionMetadata | None:
    forward_context = get_forward_context()
    attn_metadata: AttentionMetadata = forward_context.attn_metadata
    if attn_metadata is None:
        return None
    if not isinstance(attn_metadata, dict):
        raise TypeError(f"Expected attention metadata mapping, got {type(attn_metadata).__name__}")
    metadata = attn_metadata[prefix]
    if not isinstance(metadata, GDNAttentionMetadata):
        raise TypeError(f"Expected GDNAttentionMetadata, got {type(metadata).__name__}")
    return metadata


def execute_gdn_core(
    layer: Any,
    adapter: GDNExecutionAdapter,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
) -> None:
    """Run the common GDN phase orchestration through a hardware adapter."""
    metadata = _get_gdn_metadata(layer.prefix)
    if metadata is None:
        # V1 profile run happens before KV cache allocation. Resolve optional
        # operator support here instead of adding a probe to the first request.
        adapter.initialize_capabilities(layer)
        return

    plan = _get_execution_plan(metadata)
    mixed_qkv, b, a = adapter.prepare_inputs(plan, mixed_qkv, b, a)

    mixed_qkv_spec, mixed_qkv_non_spec = _partition_tensor(mixed_qkv, plan, token_dim=0)
    if plan.has_spec_decode:
        assert mixed_qkv_spec is not None
        mixed_qkv_spec = adapter.causal_conv_spec(
            layer,
            metadata,
            plan,
            mixed_qkv_spec,
        )
    if plan.has_prefill:
        assert mixed_qkv_non_spec is not None
        mixed_qkv_non_spec = adapter.causal_conv_prefill(layer, metadata, mixed_qkv_non_spec)
    elif plan.has_decode:
        assert mixed_qkv_non_spec is not None
        mixed_qkv_non_spec = adapter.causal_conv_decode(layer, metadata, mixed_qkv_non_spec)
    else:
        mixed_qkv_non_spec = None

    query_spec, key_spec, value_spec = layer.rearrange_mixed_qkv(mixed_qkv_spec)
    query_non_spec, key_non_spec, value_non_spec = layer.rearrange_mixed_qkv(mixed_qkv_non_spec)

    g, beta = adapter.gating(layer, a, b)
    g_spec, g_non_spec = _partition_tensor(g, plan, token_dim=1)
    beta_spec, beta_non_spec = _partition_tensor(beta, plan, token_dim=1)

    core_attn_out_spec = None
    if plan.has_spec_decode:
        assert query_spec is not None and key_spec is not None and value_spec is not None
        assert g_spec is not None and beta_spec is not None
        core_attn_out_spec = adapter.recurrent_spec(
            layer,
            metadata,
            query_spec,
            key_spec,
            value_spec,
            g_spec,
            beta_spec,
        )

    core_attn_out_decode = None
    split_non_spec = plan.has_mixed_non_spec and adapter.split_mixed_non_spec_decode
    if split_non_spec:
        assert query_non_spec is not None and key_non_spec is not None and value_non_spec is not None
        assert g_non_spec is not None and beta_non_spec is not None
        decode_tokens = plan.num_decode_tokens
        core_attn_out_decode = adapter.recurrent_decode(
            layer,
            metadata,
            query_non_spec[:, :decode_tokens],
            key_non_spec[:, :decode_tokens],
            value_non_spec[:, :decode_tokens],
            g_non_spec[:, :decode_tokens],
            beta_non_spec[:, :decode_tokens],
            mixed_batch=True,
        )
        query_non_spec = query_non_spec[:, decode_tokens:]
        key_non_spec = key_non_spec[:, decode_tokens:]
        value_non_spec = value_non_spec[:, decode_tokens:]
        g_non_spec = g_non_spec[:, decode_tokens:]
        beta_non_spec = beta_non_spec[:, decode_tokens:]

    core_attn_out_non_spec = None
    if plan.has_prefill:
        assert query_non_spec is not None and key_non_spec is not None and value_non_spec is not None
        assert g_non_spec is not None and beta_non_spec is not None
        core_attn_out_non_spec = adapter.recurrent_prefill(
            layer,
            metadata,
            query_non_spec,
            key_non_spec,
            value_non_spec,
            g_non_spec,
            beta_non_spec,
            excludes_decode=split_non_spec,
        )
        if core_attn_out_decode is not None:
            core_attn_out_non_spec = torch.cat(
                (core_attn_out_decode, core_attn_out_non_spec),
                dim=1,
            )
    elif plan.has_decode:
        assert query_non_spec is not None and key_non_spec is not None and value_non_spec is not None
        assert g_non_spec is not None and beta_non_spec is not None
        core_attn_out_non_spec = adapter.recurrent_decode(
            layer,
            metadata,
            query_non_spec,
            key_non_spec,
            value_non_spec,
            g_non_spec,
            beta_non_spec,
            mixed_batch=False,
        )

    if core_attn_out_spec is not None and core_attn_out_non_spec is not None:
        adapter.write_mixed_output(
            core_attn_out,
            plan,
            core_attn_out_spec,
            core_attn_out_non_spec,
        )
    elif core_attn_out_spec is not None:
        adapter.write_single_output(core_attn_out, plan, core_attn_out_spec)
    elif core_attn_out_non_spec is not None:
        adapter.write_single_output(core_attn_out, plan, core_attn_out_non_spec)

    adapter.finalize_output(core_attn_out, metadata, plan)
    maybe_save_kv_layer_to_connector("", [])
