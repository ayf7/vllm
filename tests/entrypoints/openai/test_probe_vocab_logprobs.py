# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponseChoice,
)
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.sampling_params import (
    PROBE_VOCAB_LOGPROBS_KEY,
    PROBE_VOCAB_LOGPROBS_LAST_N_KEY,
    SamplingParams,
)


class FakeModelConfig:
    max_logprobs = 20

    @staticmethod
    def get_vocab_size() -> int:
        return 100


def test_completion_probe_vocab_logprobs_accepts_token_ids() -> None:
    request = CompletionRequest(
        model="model",
        prompt="prompt",
        probe_vocab_logprobs=[7, 9],
        probe_vocab_logprobs_last_n_tokens=4,
    )

    assert request.probe_vocab_logprobs == [7, 9]
    assert request.probe_vocab_logprobs_last_n_tokens == 4


def test_chat_probe_vocab_logprobs_accepts_token_ids() -> None:
    request = ChatCompletionRequest(
        model="model",
        messages=[{"role": "user", "content": "prompt"}],
        probe_vocab_logprobs=[7],
        probe_vocab_logprobs_last_n_tokens=4,
    )

    assert request.probe_vocab_logprobs == [7]
    assert request.probe_vocab_logprobs_last_n_tokens == 4


@pytest.mark.parametrize("request_cls", [CompletionRequest, ChatCompletionRequest])
def test_probe_vocab_logprobs_rejects_string_items(request_cls) -> None:
    kwargs = _base_request_kwargs(request_cls)
    kwargs.update(
        probe_vocab_logprobs=["</bigmodel>"],
        probe_vocab_logprobs_last_n_tokens=4,
    )

    with pytest.raises(Exception, match="token ids"):
        request_cls(**kwargs)


@pytest.mark.parametrize("request_cls", [CompletionRequest, ChatCompletionRequest])
def test_probe_vocab_logprobs_rejects_streaming(request_cls) -> None:
    kwargs = _base_request_kwargs(request_cls)
    kwargs.update(
        stream=True,
        probe_vocab_logprobs=[7],
        probe_vocab_logprobs_last_n_tokens=4,
    )

    with pytest.raises(Exception, match="stream=True"):
        request_cls(**kwargs)


@pytest.mark.parametrize("request_cls", [CompletionRequest, ChatCompletionRequest])
def test_probe_vocab_logprobs_rejects_negative_token_ids(request_cls) -> None:
    kwargs = _base_request_kwargs(request_cls)
    kwargs.update(
        probe_vocab_logprobs=[-1],
        probe_vocab_logprobs_last_n_tokens=4,
    )

    with pytest.raises(Exception, match="non-negative"):
        request_cls(**kwargs)


@pytest.mark.parametrize("request_cls", [CompletionRequest, ChatCompletionRequest])
def test_probe_vocab_logprobs_requires_last_n(request_cls) -> None:
    kwargs = _base_request_kwargs(request_cls)
    kwargs.update(probe_vocab_logprobs=[7])

    with pytest.raises(Exception, match="last_n"):
        request_cls(**kwargs)


def test_attach_probe_vocab_logprobs_uses_extra_args_and_keeps_prefix_cache() -> None:
    serving = OpenAIServing.__new__(OpenAIServing)
    serving.model_config = FakeModelConfig()
    sampling_params = SamplingParams(max_tokens=1)
    request = SimpleNamespace(
        probe_vocab_logprobs=[7, 9],
        probe_vocab_logprobs_last_n_tokens=4,
    )

    serving._attach_probe_vocab_logprobs(request, sampling_params)

    assert sampling_params.extra_args == {
        PROBE_VOCAB_LOGPROBS_KEY: [7, 9],
        PROBE_VOCAB_LOGPROBS_LAST_N_KEY: 4,
    }
    assert sampling_params.skip_reading_prefix_cache is False


def test_attach_probe_vocab_logprobs_rejects_out_of_vocab_ids() -> None:
    serving = OpenAIServing.__new__(OpenAIServing)
    serving.model_config = FakeModelConfig()
    sampling_params = SamplingParams(max_tokens=1)
    request = SimpleNamespace(
        probe_vocab_logprobs=[101],
        probe_vocab_logprobs_last_n_tokens=4,
    )

    with pytest.raises(Exception, match="out-of-vocab"):
        serving._attach_probe_vocab_logprobs(request, sampling_params)


def test_sampling_params_probe_vocab_logprobs_keeps_prefix_cache() -> None:
    sampling_params = SamplingParams(
        max_tokens=1,
        extra_args={
            PROBE_VOCAB_LOGPROBS_KEY: [7],
            PROBE_VOCAB_LOGPROBS_LAST_N_KEY: 4,
        },
    )

    assert sampling_params.skip_reading_prefix_cache is False


def test_response_models_include_probe_vocab_logprobs() -> None:
    records = [{"position": 3, "token_id": 7, "logit": 1.25, "logprob": -0.5}]

    completion_choice = CompletionResponseChoice(
        index=0,
        text="x",
        finish_reason="length",
        probe_vocab_logprobs=records,
    )
    assert completion_choice.probe_vocab_logprobs == records

    chat_response = ChatCompletionResponse(
        model="model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="x"),
            )
        ],
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        probe_vocab_logprobs=records,
    )
    assert chat_response.probe_vocab_logprobs == records


def _base_request_kwargs(request_cls) -> dict:
    if request_cls is CompletionRequest:
        return {"model": "model", "prompt": "prompt"}
    return {
        "model": "model",
        "messages": [{"role": "user", "content": "prompt"}],
    }
