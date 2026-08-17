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

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.ops.gdn_execution import GDNExecutionPlan, execute_gdn_core


class _Layer:
    prefix = "gdn"

    @staticmethod
    def rearrange_mixed_qkv(tensor: torch.Tensor | None):
        if tensor is None:
            return None, None, None
        value = tensor.unsqueeze(0)
        return value, value, value


class _RecordingAdapter:
    split_mixed_non_spec_decode = True

    def __init__(self):
        self.calls = []

    def initialize_capabilities(self, layer):
        self.calls.append("initialize")

    def prepare_inputs(self, plan, mixed_qkv, b, a):
        self.calls.append("prepare")
        return mixed_qkv, b, a

    def causal_conv_spec(self, layer, metadata, plan, mixed_qkv):
        self.calls.append("conv_spec")
        return mixed_qkv

    def causal_conv_prefill(self, layer, metadata, mixed_qkv):
        self.calls.append("conv_prefill")
        return mixed_qkv

    def causal_conv_decode(self, layer, metadata, mixed_qkv):
        self.calls.append("conv_decode")
        return mixed_qkv

    def gating(self, layer, a, b):
        self.calls.append("gating")
        return a.unsqueeze(0), b.unsqueeze(0)

    def recurrent_spec(self, layer, metadata, query, key, value, g, beta):
        self.calls.append(("recurrent_spec", query.shape[1]))
        return value

    def recurrent_decode(
        self,
        layer,
        metadata,
        query,
        key,
        value,
        g,
        beta,
        mixed_batch,
    ):
        self.calls.append(("recurrent_decode", query.shape[1], mixed_batch))
        return value

    def recurrent_prefill(
        self,
        layer,
        metadata,
        query,
        key,
        value,
        g,
        beta,
        excludes_decode,
    ):
        self.calls.append(("recurrent_prefill", query.shape[1], excludes_decode))
        return value

    def write_mixed_output(self, output, plan, spec_output, non_spec_output):
        self.calls.append("write_mixed")

    def write_single_output(self, output, plan, value):
        self.calls.append(("write_single", value.shape[1]))

    def finalize_output(self, output, metadata, plan):
        self.calls.append("finalize")


def _run(plan: GDNExecutionPlan, adapter: _RecordingAdapter):
    metadata = SimpleNamespace(execution_plan=plan)
    mixed_qkv = torch.zeros(plan.num_actual_tokens, 1, 1)
    b = torch.zeros(plan.num_actual_tokens, 1)
    a = torch.zeros(plan.num_actual_tokens, 1)
    output = torch.zeros_like(mixed_qkv)
    with (
        patch("vllm_ascend.ops.gdn_execution._get_gdn_metadata", return_value=metadata),
        patch("vllm_ascend.ops.gdn_execution.maybe_save_kv_layer_to_connector") as save_state,
    ):
        execute_gdn_core(_Layer(), adapter, mixed_qkv, b, a, output)
    save_state.assert_called_once_with("", [])


def test_shared_execution_routes_mixed_spec_and_prefill_once():
    adapter = _RecordingAdapter()
    plan = GDNExecutionPlan(
        num_actual_tokens=4,
        num_prefills=1,
        num_decode_tokens=0,
        num_decodes=0,
        num_spec_decodes=1,
        spec_token_indices=torch.tensor([1, 3]),
        non_spec_token_indices=torch.tensor([0, 2]),
    )

    _run(plan, adapter)

    assert adapter.calls == [
        "prepare",
        "conv_spec",
        "conv_prefill",
        "gating",
        ("recurrent_spec", 2),
        ("recurrent_prefill", 2, False),
        "write_mixed",
        "finalize",
    ]


def test_shared_execution_splits_common_mixed_decode_and_prefill():
    adapter = _RecordingAdapter()
    plan = GDNExecutionPlan(
        num_actual_tokens=3,
        num_prefills=1,
        num_decode_tokens=1,
        num_decodes=1,
        num_spec_decodes=0,
        spec_token_indices=None,
        non_spec_token_indices=None,
    )

    _run(plan, adapter)

    assert adapter.calls == [
        "prepare",
        "conv_prefill",
        "gating",
        ("recurrent_decode", 1, True),
        ("recurrent_prefill", 2, True),
        ("write_single", 3),
        "finalize",
    ]
