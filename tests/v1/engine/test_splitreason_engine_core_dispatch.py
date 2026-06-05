# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch tests for the SplitReason EngineCore forwarders.

The SplitReason coordinator runs in the host process; the scheduler it needs to
poke (apply a warm-mirror streaming delta, read a request's view lengths) lives
in the engine-core process. Two thin EngineCore methods bridge that gap — they
are reached from the host via the client's ``call_utility`` path, which resolves
the method with ``getattr(self, name)`` and calls it with positional args. Each
forwarder must do nothing but pass its positional args straight to the matching
scheduler method and return the scheduler's result verbatim.

These tests call the EngineCore methods UNBOUND on a hand-built stub whose
``.scheduler`` records the forwarded call — no GPU, no real engine, no model
load. They pin the forwarding contract (method name, positional arg order,
verbatim return) so a refactor that reorders args or post-processes the result
fails here rather than silently corrupting a live handoff.
"""

from vllm.v1.engine.core import EngineCore


class RecordingScheduler:
    """Records forwarded scheduler calls and returns a unique sentinel each."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def splitreason_apply_streaming_delta(self, request_id, token_ids):
        self.calls.append(("apply_streaming_delta", request_id, token_ids))
        return {"sentinel": "delta", "request_id": request_id}

    def splitreason_request_view_lengths(
        self, request_id, prompt_len, open_id, close_id
    ):
        self.calls.append(
            ("request_view_lengths", request_id, prompt_len, open_id, close_id)
        )
        return {"sentinel": "lengths", "request_id": request_id}

    def splitreason_arm_decode(self, request_id):
        self.calls.append(("arm_decode", request_id))
        return {"sentinel": "arm", "request_id": request_id}


class EngineCoreStub:
    """Minimal stand-in carrying only the ``scheduler`` the forwarders touch."""

    def __init__(self) -> None:
        self.scheduler = RecordingScheduler()


def test_apply_streaming_delta_forwards_to_scheduler() -> None:
    stub = EngineCoreStub()

    result = EngineCore.splitreason_apply_streaming_delta(stub, "req-7", [11, 22])

    assert stub.scheduler.calls == [("apply_streaming_delta", "req-7", [11, 22])]
    # Returned verbatim — the forwarder adds no post-processing of its own.
    assert result == {"sentinel": "delta", "request_id": "req-7"}


def test_request_view_lengths_forwards_to_scheduler() -> None:
    stub = EngineCoreStub()

    result = EngineCore.splitreason_request_view_lengths(stub, "req-9", 8, 99, 100)

    assert stub.scheduler.calls == [
        ("request_view_lengths", "req-9", 8, 99, 100)
    ]
    assert result == {"sentinel": "lengths", "request_id": "req-9"}


def test_forwarders_preserve_positional_arg_order() -> None:
    # call_utility passes positional args, so a silent arg-order swap in either
    # forwarder would not be caught by name. Use all-distinct values so any
    # transposition (e.g. open_id/close_id, prompt_len/open_id) is observable.
    stub = EngineCoreStub()

    EngineCore.splitreason_request_view_lengths(stub, "r", 3, 4, 5)

    assert stub.scheduler.calls[0] == ("request_view_lengths", "r", 3, 4, 5)


def test_arm_decode_forwards_to_scheduler() -> None:
    stub = EngineCoreStub()

    result = EngineCore.splitreason_arm_decode(stub, "req-3")

    assert stub.scheduler.calls == [("arm_decode", "req-3")]
    assert result == {"sentinel": "arm", "request_id": "req-3"}


def test_engine_core_exposes_all_forwarders() -> None:
    # Guard against a rebase dropping any forwarder from EngineCore entirely.
    assert callable(getattr(EngineCore, "splitreason_apply_streaming_delta"))
    assert callable(getattr(EngineCore, "splitreason_request_view_lengths"))
    assert callable(getattr(EngineCore, "splitreason_arm_decode"))
