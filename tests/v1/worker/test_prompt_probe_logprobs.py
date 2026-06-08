# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class FakeModel:
    def __init__(self):
        self.calls = 0
        self.shapes = []

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.shapes.append(tuple(hidden_states.shape))
        return hidden_states


class FakeRunner(SimpleNamespace):
    _compute_prompt_probe_logprob_records = (
        GPUModelRunner._compute_prompt_probe_logprob_records
    )


def test_prompt_probe_logprobs_batches_all_probe_rows_in_one_lm_head_call() -> None:
    model = FakeModel()
    runner = FakeRunner(
        model=model,
        device=torch.device("cpu"),
        prompt_probe_logprobs={
            "req-a": ([2], 2),
            "req-b": ([1, 3], 1),
        },
        in_progress_prompt_probe_logprobs={"req-a": [], "req-b": []},
        requests={
            "req-a": SimpleNamespace(prompt_token_ids=[10, 11, 12], num_computed_tokens=0),
            "req-b": SimpleNamespace(prompt_token_ids=[20, 21], num_computed_tokens=0),
        },
        input_batch=SimpleNamespace(req_id_to_index={"req-a": 0, "req-b": 1}),
        query_start_loc=SimpleNamespace(np=np.array([0, 3], dtype=np.int32)),
    )
    hidden_states = torch.tensor(
        [
            [0.0, 0.1, 0.2, 0.3],
            [1.0, 1.1, 1.2, 1.3],
            [2.0, 2.1, 2.2, 2.3],
            [3.0, 3.1, 3.2, 3.3],
            [4.0, 4.1, 4.2, 4.3],
        ],
        dtype=torch.float32,
    )

    result = GPUModelRunner._get_prompt_probe_logprobs_dict(
        runner,
        hidden_states,
        {"req-a": 3, "req-b": 2},
    )

    assert model.calls == 1
    assert model.shapes == [(3, 4)]
    assert set(result) == {"req-a", "req-b"}
    assert runner.prompt_probe_logprobs == {}
    assert runner.in_progress_prompt_probe_logprobs == {}

    assert [(r["position"], r["token_id"]) for r in result["req-a"]] == [
        (1, 2),
        (2, 2),
    ]
    assert [(r["position"], r["token_id"]) for r in result["req-b"]] == [
        (1, 1),
        (1, 3),
    ]

    expected_rows = {
        ("req-a", 1, 2): hidden_states[1],
        ("req-a", 2, 2): hidden_states[2],
        ("req-b", 1, 1): hidden_states[4],
        ("req-b", 1, 3): hidden_states[4],
    }
    for req_id, records in result.items():
        for record in records:
            logits = expected_rows[(req_id, record["position"], record["token_id"])]
            token_id = record["token_id"]
            expected_logprob = logits[token_id] - torch.logsumexp(logits, dim=-1)
            assert record["logit"] == pytest.approx(float(logits[token_id]))
            assert record["logprob"] == pytest.approx(float(expected_logprob))


def test_prompt_probe_logprobs_accumulates_across_prefill_chunks() -> None:
    model = FakeModel()
    runner = FakeRunner(
        model=model,
        device=torch.device("cpu"),
        prompt_probe_logprobs={"req": ([2], 3)},
        in_progress_prompt_probe_logprobs={"req": []},
        requests={
            "req": SimpleNamespace(prompt_token_ids=[10, 11, 12, 13], num_computed_tokens=0),
        },
        input_batch=SimpleNamespace(req_id_to_index={"req": 0}),
        query_start_loc=SimpleNamespace(np=np.array([0], dtype=np.int32)),
    )

    first = GPUModelRunner._get_prompt_probe_logprobs_dict(
        runner,
        torch.tensor(
            [[0.0, 0.1, 0.2, 0.3], [1.0, 1.1, 1.2, 1.3]],
            dtype=torch.float32,
        ),
        {"req": 2},
    )

    assert first == {}
    assert model.calls == 1
    assert [r["position"] for r in runner.in_progress_prompt_probe_logprobs["req"]] == [
        1
    ]

    runner.requests["req"].num_computed_tokens = 2
    second = GPUModelRunner._get_prompt_probe_logprobs_dict(
        runner,
        torch.tensor(
            [[2.0, 2.1, 2.2, 2.3], [3.0, 3.1, 3.2, 3.3]],
            dtype=torch.float32,
        ),
        {"req": 2},
    )

    assert model.calls == 2
    assert [r["position"] for r in second["req"]] == [1, 2, 3]
