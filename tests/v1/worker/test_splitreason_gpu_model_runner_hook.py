# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-anchored guard for the SplitReason sampled-token fork hook.

The SplitReason Architecture D tap is invoked from a single block inside
``GPUModelRunner._bookkeeping_sync`` (gpu_model_runner.py): right after the
sampler runs, while the sampled token ids are still a GPU tensor, the runner
checks for an attached ``splitreason_tap`` and forwards the batch to it. That
block is the only vLLM-side modification on the worker hot path, and it is the
kind of edit a vLLM submodule rebase silently drops or reshapes.

These tests do NOT import the splitreason package (vLLM stays splitreason-free)
and need no GPU. They parse the runner's own source and assert the hook block is
present with its exact call contract, so a rebase that removes or renames the
hook, drops a kwarg, or swaps the host-side discard view (``.np``) for the GPU
tensor (``.gpu``) fails loudly here. The matching consumer-side contract — that
SplitReasonTokenTap.on_sampled_token_ids accepts exactly these four kwargs — is
pinned from the splitreason side in tests/splitreason/inference.
"""

import ast
import inspect

import pytest

import vllm.v1.worker.gpu_model_runner as gpu_model_runner

TAP_ATTR = "splitreason_tap"
HOOK_METHOD = "on_sampled_token_ids"
EXPECTED_KWARGS = {
    "sampled_token_ids",
    "discard_request_mask",
    "req_ids",
    "req_id_to_index",
}


@pytest.fixture(scope="module")
def runner_source() -> str:
    return inspect.getsource(gpu_model_runner)


@pytest.fixture(scope="module")
def hook_call(runner_source: str) -> ast.Call:
    """The single ``<tap>.on_sampled_token_ids(...)`` call node in the runner."""
    tree = ast.parse(runner_source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == HOOK_METHOD
    ]
    assert len(calls) == 1, (
        f"expected exactly one {HOOK_METHOD} call in gpu_model_runner, "
        f"found {len(calls)} — the SplitReason hook moved or was duplicated"
    )
    return calls[0]


def test_tap_is_read_with_getattr_default_none(runner_source: str) -> None:
    # The hook must degrade to a no-op on a stock runner: it reads the tap with
    # a defaulted getattr so an engine that never attached one is unaffected.
    assert f'getattr(self, "{TAP_ATTR}", None)' in runner_source


def test_hook_call_is_positionally_clean(hook_call: ast.Call) -> None:
    # All arguments are passed by keyword; a stray positional would make the
    # contract order-sensitive and harder to verify against the tap signature.
    assert hook_call.args == []
    assert all(kw.arg is not None for kw in hook_call.keywords), (
        "the hook must not use **kwargs splat — every argument is explicit"
    )


def test_hook_call_passes_exact_kwargs(hook_call: ast.Call) -> None:
    passed = {kw.arg for kw in hook_call.keywords}
    assert passed == EXPECTED_KWARGS, (
        f"hook kwargs drifted: {sorted(passed)} != {sorted(EXPECTED_KWARGS)}"
    )


def test_discard_mask_uses_host_numpy_view(hook_call: ast.Call) -> None:
    # The mask must be the HOST numpy slice (self.discard_request_mask.np[
    # :num_reqs]), not the GPU tensor (.gpu). The tap reads it on the host to
    # avoid a per-row device sync, and the design doc's .gpu snippet is stale —
    # this is the exact line that bit us, so it gets its own assertion.
    kw = _keyword(hook_call, "discard_request_mask")
    rendered = ast.unparse(kw.value)
    assert rendered == "self.discard_request_mask.np[:num_reqs]", rendered
    assert ".gpu" not in rendered


def test_sampled_token_ids_passed_through(hook_call: ast.Call) -> None:
    kw = _keyword(hook_call, "sampled_token_ids")
    assert ast.unparse(kw.value) == "sampled_token_ids"


def test_req_ids_come_from_input_batch(hook_call: ast.Call) -> None:
    assert ast.unparse(_keyword(hook_call, "req_ids").value) == (
        "self.input_batch.req_ids"
    )
    assert ast.unparse(_keyword(hook_call, "req_id_to_index").value) == (
        "self.input_batch.req_id_to_index"
    )


def _keyword(call: ast.Call, name: str) -> ast.keyword:
    for kw in call.keywords:
        if kw.arg == name:
            return kw
    raise AssertionError(f"hook call has no {name!r} keyword")


# --- Gate hook: force-discard the passive cooperative mirror --------------
#
# A second SplitReason edit lives a few lines up, inside _prepare_inputs: right
# after the natural discard mask is computed (seq_lens < num_tokens) and BEFORE
# copy_to_gpu, the runner forces the passive-mirror rows back to discarded via
# the attached tap. It must read the HOST numpy view (writes are seen by the
# same-step discard clear) and pass (mask, req_ids) in that order. A rebase that
# drops it, moves it after copy_to_gpu, or swaps .np for .gpu fails here.

GATE_METHOD = "apply_gate_to_discard_mask"


@pytest.fixture(scope="module")
def gate_call(runner_source: str) -> ast.Call:
    tree = ast.parse(runner_source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == GATE_METHOD
    ]
    assert len(calls) == 1, (
        f"expected exactly one {GATE_METHOD} call in gpu_model_runner, "
        f"found {len(calls)} — the SplitReason gate hook moved or was duplicated"
    )
    return calls[0]


def test_gate_hook_passes_host_numpy_mask_then_req_ids(gate_call: ast.Call) -> None:
    # The gate writes the mask in place, so it must take the HOST numpy slice
    # (.np[:num_reqs]) — the .gpu tensor would not be seen by the host-side
    # discard clear. Order is (mask, req_ids), both positional.
    assert gate_call.keywords == []
    assert len(gate_call.args) == 2
    mask_arg = ast.unparse(gate_call.args[0])
    assert mask_arg == "self.discard_request_mask.np[:num_reqs]", mask_arg
    assert ".gpu" not in mask_arg
    assert ast.unparse(gate_call.args[1]) == "self.input_batch.req_ids"


def test_gate_hook_is_guarded_by_getattr_default_none(runner_source: str) -> None:
    # Same no-op-on-stock-runner discipline as the sampled-token hook: the gate
    # only fires when a tap was attached, so a vanilla engine is unaffected.
    assert runner_source.count(f'getattr(self, "{TAP_ATTR}", None)') >= 2


def test_gate_runs_before_discard_mask_copy_to_gpu(runner_source: str) -> None:
    # The forced mask must reach the GPU too: the gate call has to precede the
    # copy_to_gpu that uploads the discard mask, or the host and device masks
    # would disagree for the rest of the step.
    gate_idx = runner_source.index(GATE_METHOD)
    copy_idx = runner_source.index("discard_request_mask.copy_to_gpu", gate_idx)
    assert gate_idx < copy_idx
