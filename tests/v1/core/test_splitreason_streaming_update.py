# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Scheduler.splitreason_apply_streaming_delta.

This is the scheduler-side primitive of the SplitReason Architecture D warm
mirror: while one model decodes, the idle model's mirror request is extended
in place with freshly produced context and re-queued so only the appended
delta is prefilled, keeping its KV prefix warm for an instant handoff. These
exercise scheduler logic only -- no model weights are loaded and no GPU compute
runs -- but they build on the shared create_scheduler / create_requests helpers,
which construct a VllmConfig with device="auto"; that still needs vLLM to
resolve a platform, so these run on the project's GPU node and are not portable
to a platform-less CPU-only sandbox.
"""

import pytest

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

OUTPUT_TOKEN = 777


def _drive_to_warm_mirror(scheduler, request):
    """Schedule a fresh request and simulate one decode step.

    Returns it RUNNING with a fully-prefilled (computed) prompt prefix and one
    just-sampled, still-uncomputed output token — exactly the warm-mirror
    mid-state splitreason_apply_streaming_delta is designed to extend.
    """
    scheduler.add_request(request)
    sched_out = scheduler.schedule()
    mro = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[OUTPUT_TOKEN]],
    )
    scheduler.update_from_output(sched_out, mro)


def test_streaming_delta_folds_output_and_preserves_warm_prefix():
    scheduler = create_scheduler(max_num_seqs=4, block_size=16)
    request = create_requests(num_requests=1, num_tokens=8, req_ids=["req-0"])[0]
    prompt = list(request.prompt_token_ids)

    _drive_to_warm_mirror(scheduler, request)
    assert request.status == RequestStatus.RUNNING
    assert request.num_computed_tokens == 8  # whole prompt prefilled
    assert request.num_tokens == 9  # + 1 uncomputed output token

    result = scheduler.splitreason_apply_streaming_delta("req-0", [555, 556])

    expected = prompt + [OUTPUT_TOKEN, 555, 556]
    # The committed output is folded into the prompt and the delta appended,
    # so prompt_token_ids carries the whole sequence and _all_token_ids equals
    # it (the worker's _update_streaming_request rebuilds context from
    # prompt_token_ids and clears the output view).
    assert list(request.prompt_token_ids) == expected
    assert list(request._all_token_ids) == expected
    assert list(request._output_token_ids) == []
    assert request.num_prompt_tokens == len(expected)
    # The warm prefix is preserved (NOT reset, unlike a preemption): the
    # uncomputed tail token is kept and only the delta is new.
    assert request.num_computed_tokens == 8
    assert request.status == RequestStatus.WAITING
    assert request not in scheduler.running

    assert result["num_computed_tokens"] == 8
    assert result["num_tokens"] == len(expected)
    assert result["num_new_tokens"] == len(expected) - 8  # tail(1) + delta(2)
    assert result["delta_len"] == 2
    assert result["was_running"] is True


def test_streaming_delta_reschedules_only_the_uncomputed_tail():
    scheduler = create_scheduler(max_num_seqs=4, block_size=16)
    request = create_requests(num_requests=1, num_tokens=8, req_ids=["req-0"])[0]

    _drive_to_warm_mirror(scheduler, request)
    scheduler.splitreason_apply_streaming_delta("req-0", [555, 556])

    # On the next schedule the warm mirror is resumed: num_computed_tokens > 0
    # skips the prefix-cache lookup, so only num_tokens - num_computed_tokens
    # (the uncomputed tail plus the delta) is scheduled, on top of the blocks
    # the request already holds (they were never freed).
    sched_out = scheduler.schedule()
    assert sched_out.num_scheduled_tokens["req-0"] == 3
    scheduled_new_ids = [r.req_id for r in sched_out.scheduled_new_reqs]
    assert "req-0" in scheduled_new_ids  # WAITING -> scheduled_new_reqs
    assert request.status == RequestStatus.RUNNING


def test_streaming_delta_on_waiting_request_appends_without_output_fold():
    # A mirror that has only been added (never decoded) is WAITING with no
    # committed output; the delta extends the prompt directly.
    scheduler = create_scheduler(max_num_seqs=4, block_size=16)
    request = create_requests(num_requests=1, num_tokens=8, req_ids=["req-0"])[0]
    prompt = list(request.prompt_token_ids)
    scheduler.add_request(request)
    assert request.status == RequestStatus.WAITING

    result = scheduler.splitreason_apply_streaming_delta("req-0", [555, 556])

    assert list(request.prompt_token_ids) == prompt + [555, 556]
    assert list(request._all_token_ids) == prompt + [555, 556]
    assert request.num_computed_tokens == 0
    assert request.status == RequestStatus.WAITING
    assert result["was_running"] is False


def test_streaming_delta_on_waiting_mirror_does_not_duplicate_in_queue():
    # Regression: an already-WAITING mirror (the normal idle warm-mirror state)
    # is already in self.waiting, so the method must NOT re-add it. A duplicate
    # reference would be popped twice by schedule() — the second pop, after the
    # first copy is set RUNNING, crashes on the "Invalid request status" guard.
    scheduler = create_scheduler(max_num_seqs=4, block_size=16)
    request = create_requests(num_requests=1, num_tokens=8, req_ids=["req-0"])[0]
    scheduler.add_request(request)
    assert len(scheduler.waiting) == 1

    scheduler.splitreason_apply_streaming_delta("req-0", [555, 556])
    # A second delta on the same still-waiting mirror must also not duplicate.
    scheduler.splitreason_apply_streaming_delta("req-0", [557])
    assert len(scheduler.waiting) == 1

    # The next schedule must not raise and must schedule the request exactly
    # once with the full extended prompt.
    sched_out = scheduler.schedule()
    assert sched_out.num_scheduled_tokens["req-0"] == 8 + 3  # prompt + 3 delta
    assert request.status == RequestStatus.RUNNING
    assert scheduler.running.count(request) == 1


def test_streaming_delta_on_waiting_for_streaming_req_mirror():
    # The mirror can also be parked in WAITING_FOR_STREAMING_REQ: _handle_stopped
    # _request puts a resumable streaming request there (scheduler.py:1653-1656)
    # — status set, num_waiting_for_streaming_input incremented, already added to
    # self.waiting. Applying a delta must decrement that counter exactly once,
    # NOT re-add to the queue, and schedule cleanly.
    scheduler = create_scheduler(max_num_seqs=4, block_size=16)
    request = create_requests(num_requests=1, num_tokens=8, req_ids=["req-0"])[0]
    _drive_to_warm_mirror(scheduler, request)
    # Reproduce the post-stop state exactly as the stop path leaves it.
    scheduler.running.remove(request)
    request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    scheduler.num_waiting_for_streaming_input += 1
    scheduler.waiting.add_request(request)
    assert len(scheduler.waiting) == 1

    result = scheduler.splitreason_apply_streaming_delta("req-0", [555, 556])

    assert request.status == RequestStatus.WAITING
    assert scheduler.num_waiting_for_streaming_input == 0  # decremented once
    assert len(scheduler.waiting) == 1  # not duplicated
    assert result["was_running"] is False

    sched_out = scheduler.schedule()
    assert sched_out.num_scheduled_tokens["req-0"] == 3  # tail(1) + delta(2)
    assert request.status == RequestStatus.RUNNING
    assert scheduler.running.count(request) == 1


def test_streaming_delta_empty_tail_is_rejected_at_call_site():
    # An empty delta on a mirror whose whole sequence is already computed would
    # leave num_tokens == num_computed_tokens, which the next schedule() crashes
    # on (assert num_new_tokens > 0). The method must reject it up front instead
    # of deferring that crash.
    scheduler = create_scheduler(max_num_seqs=4, block_size=16)
    request = create_requests(num_requests=1, num_tokens=8, req_ids=["req-0"])[0]
    scheduler.add_request(request)
    request.num_computed_tokens = request.num_tokens  # fully computed, no tail

    with pytest.raises(ValueError, match="no uncomputed tokens"):
        scheduler.splitreason_apply_streaming_delta("req-0", [])


def test_streaming_delta_unknown_request_raises():
    scheduler = create_scheduler()
    with pytest.raises(KeyError, match="unknown request_id"):
        scheduler.splitreason_apply_streaming_delta("nope", [1, 2])


def test_streaming_delta_rejects_finished_request():
    scheduler = create_scheduler(max_num_seqs=4, block_size=16)
    request = create_requests(num_requests=1, num_tokens=8, req_ids=["req-0"])[0]
    scheduler.add_request(request)
    # Force a status outside the {RUNNING, WAITING, WAITING_FOR_STREAMING_REQ}
    # whitelist; applying a delta to a finished request is a coordinator bug.
    scheduler.requests["req-0"].status = RequestStatus.FINISHED_STOPPED

    with pytest.raises(ValueError, match="cannot apply a streaming delta"):
        scheduler.splitreason_apply_streaming_delta("req-0", [1, 2])


# ---------------------------------------------------------------------------
# splitreason_request_view_lengths
#
# The companion read-side primitive: between steps the coordinator re-binds the
# tap's small/large view lengths to ground truth, because the tap goes blind
# during an offload span (boundary samples are discarded prefill; large-authored
# tokens are folded into the mirror prompt without passing the tap). This method
# measures both views directly from the scheduler's authoritative token state.
# The large view strips the two control ids (OPEN_ID/CLOSE_ID below); the small
# view counts the whole generated suffix.
# ---------------------------------------------------------------------------

OPEN_ID = 90001
CLOSE_ID = 90002


def test_view_lengths_strip_control_ids_from_large_view():
    scheduler = create_scheduler(max_num_seqs=4, block_size=16)
    request = create_requests(num_requests=1, num_tokens=8, req_ids=["req-0"])[0]
    scheduler.add_request(request)
    # Append a generated suffix mixing small-authored control tokens with
    # ordinary tokens — exactly what a post-handoff small mirror carries.
    suffix = [500, OPEN_ID, 501, 502, CLOSE_ID, 503]
    for tok in suffix:
        request.append_output_token_ids(tok)

    result = scheduler.splitreason_request_view_lengths(
        "req-0", prompt_len=8, open_id=OPEN_ID, close_id=CLOSE_ID
    )

    assert result["small_view_len"] == 6  # whole suffix
    assert result["large_view_len"] == 4  # suffix minus the two control ids
    assert result["num_tokens"] == 8 + 6
    assert result["prompt_len"] == 8
    assert result["request_id"] == "req-0"


def test_view_lengths_equal_for_control_free_suffix():
    scheduler = create_scheduler(max_num_seqs=4, block_size=16)
    request = create_requests(num_requests=1, num_tokens=8, req_ids=["req-0"])[0]
    scheduler.add_request(request)
    for tok in [500, 501, 502]:
        request.append_output_token_ids(tok)

    result = scheduler.splitreason_request_view_lengths(
        "req-0", prompt_len=8, open_id=OPEN_ID, close_id=CLOSE_ID
    )

    # No control tokens in the suffix -> small and large views agree.
    assert result["small_view_len"] == 3
    assert result["large_view_len"] == 3


def test_view_lengths_empty_suffix_when_prompt_len_is_full_length():
    scheduler = create_scheduler(max_num_seqs=4, block_size=16)
    request = create_requests(num_requests=1, num_tokens=8, req_ids=["req-0"])[0]
    scheduler.add_request(request)

    result = scheduler.splitreason_request_view_lengths(
        "req-0", prompt_len=8, open_id=OPEN_ID, close_id=CLOSE_ID
    )

    assert result["small_view_len"] == 0
    assert result["large_view_len"] == 0
    assert result["num_tokens"] == 8


def test_view_lengths_unknown_request_raises():
    scheduler = create_scheduler()
    with pytest.raises(KeyError, match="unknown request_id"):
        scheduler.splitreason_request_view_lengths(
            "nope", prompt_len=0, open_id=OPEN_ID, close_id=CLOSE_ID
        )


def test_view_lengths_rejects_out_of_range_prompt_len():
    scheduler = create_scheduler(max_num_seqs=4, block_size=16)
    request = create_requests(num_requests=1, num_tokens=8, req_ids=["req-0"])[0]
    scheduler.add_request(request)

    with pytest.raises(ValueError, match="out of range"):
        scheduler.splitreason_request_view_lengths(
            "req-0", prompt_len=99, open_id=OPEN_ID, close_id=CLOSE_ID
        )
    with pytest.raises(ValueError, match="out of range"):
        scheduler.splitreason_request_view_lengths(
            "req-0", prompt_len=-1, open_id=OPEN_ID, close_id=CLOSE_ID
        )


# ---------------------------------------------------------------------------
# splitreason_arm_decode
#
# The un-gate half of an offload handoff. While the other model decodes, this
# engine keep-prefills a force-discarded warm mirror, so after its last gated
# step the mirror is exactly caught up (num_computed == num_tokens) with nothing
# queued. arm_decode rewinds num_computed by one so the next schedule() re-runs
# that single already-cached position; on the worker the recomputed seq_lens
# reaches num_tokens, the (seq_lens < num_tokens) discard bit is False, and the
# sampled continuation is KEPT and appended — dropping the mirror into decode.
# These exercise the scheduler rewind + re-schedule only (the kept/discard
# decision is the worker's, validated on the real pair in the handoff probe).
# ---------------------------------------------------------------------------


def _caught_up_mirror(scheduler, num_tokens=8):
    """Return a RUNNING mirror at num_computed == num_tokens (nothing queued).

    This is the post-gated-step state: the prompt is fully prefilled and the
    boundary sample was force-discarded, so no output token was appended. The
    real path reaches it via the worker's discard mask; here it is set directly
    (the same shortcut the empty-tail streaming-delta test uses) so the rewind
    is tested in isolation from worker compute.
    """
    request = create_requests(num_requests=1, num_tokens=num_tokens, req_ids=["req-0"])[0]
    scheduler.add_request(request)
    sched_out = scheduler.schedule()
    # Drive the prefill to completion with a discarded (empty) sample row, the
    # exact ModelRunnerOutput a gated warm-mirror step produces: num_computed
    # advances to cover the prefill, nothing is appended.
    scheduler.update_from_output(
        sched_out,
        ModelRunnerOutput(
            req_ids=["req-0"],
            req_id_to_index={"req-0": 0},
            sampled_token_ids=[[]],
        ),
    )
    assert request.status == RequestStatus.RUNNING
    assert request.num_computed_tokens == request.num_tokens == num_tokens
    return request


def test_arm_decode_rewinds_caught_up_mirror_by_one():
    scheduler = create_scheduler(max_num_seqs=4, block_size=16)
    request = _caught_up_mirror(scheduler, num_tokens=8)

    result = scheduler.splitreason_arm_decode("req-0")

    # Rewound by exactly one: one uncomputed position, the last real token.
    assert request.num_computed_tokens == 7
    assert request.num_tokens == 8
    assert request.status == RequestStatus.RUNNING
    assert result["num_computed_tokens"] == 7
    assert result["num_tokens"] == 8
    assert result["num_new_tokens"] == 1


def test_arm_decode_schedules_one_already_cached_position():
    # After arming, the next schedule() runs exactly one token and needs NO new
    # block: position num_tokens-1 was already computed and its block is still
    # held (the mirror was never preempted), so the request schedules without
    # evicting anyone and stays RUNNING.
    scheduler = create_scheduler(max_num_seqs=4, block_size=16)
    request = _caught_up_mirror(scheduler, num_tokens=8)
    scheduler.splitreason_arm_decode("req-0")

    sched_out = scheduler.schedule()

    assert sched_out.num_scheduled_tokens["req-0"] == 1
    assert request.status == RequestStatus.RUNNING
    assert scheduler.running.count(request) == 1
    # A re-run of an already-cached position requires no fresh blocks.
    new_blocks = sched_out.scheduled_cached_reqs.new_block_ids
    assert all(not any(group) for group in (new_blocks or []) if group is not None)


def test_arm_decode_then_kept_sample_resumes_normal_decode():
    # Arm, schedule the one position, then deliver a KEPT sample (a real decode
    # token, not the empty discarded row): it appends and the mirror is back to
    # an ordinary decode cadence — num_computed one behind num_tokens.
    scheduler = create_scheduler(max_num_seqs=4, block_size=16)
    request = _caught_up_mirror(scheduler, num_tokens=8)
    scheduler.splitreason_arm_decode("req-0")
    sched_out = scheduler.schedule()

    scheduler.update_from_output(
        sched_out,
        ModelRunnerOutput(
            req_ids=["req-0"],
            req_id_to_index={"req-0": 0},
            sampled_token_ids=[[OUTPUT_TOKEN]],
        ),
    )

    assert list(request._all_token_ids)[-1] == OUTPUT_TOKEN
    assert request.num_tokens == 9  # the kept span token was appended
    assert request.num_computed_tokens == 8  # one behind: normal decode state
    assert request.status == RequestStatus.RUNNING


def test_arm_decode_rejects_mid_prefill_mirror():
    # A mirror still carrying an uncomputed tail is not ours to arm: arming would
    # skip its queued tokens. Reject it rather than silently dropping context.
    scheduler = create_scheduler(max_num_seqs=4, block_size=16)
    request = create_requests(num_requests=1, num_tokens=8, req_ids=["req-0"])[0]
    scheduler.add_request(request)  # WAITING, num_computed=0 < num_tokens=8

    with pytest.raises(ValueError, match="fully caught-up"):
        scheduler.splitreason_arm_decode("req-0")


def test_arm_decode_unknown_request_raises():
    scheduler = create_scheduler()
    with pytest.raises(KeyError, match="unknown request_id"):
        scheduler.splitreason_arm_decode("nope")


def test_arm_decode_rejects_finished_request():
    scheduler = create_scheduler(max_num_seqs=4, block_size=16)
    request = _caught_up_mirror(scheduler, num_tokens=8)
    request.status = RequestStatus.FINISHED_STOPPED

    with pytest.raises(ValueError, match="must be a scheduled .* or queued"):
        scheduler.splitreason_arm_decode("req-0")


def test_arm_decode_rejects_speculative_tokens():
    scheduler = create_scheduler(max_num_seqs=4, block_size=16)
    request = _caught_up_mirror(scheduler, num_tokens=8)
    request.spec_token_ids = [123]

    with pytest.raises(ValueError, match="speculative decode is not supported"):
        scheduler.splitreason_arm_decode("req-0")
