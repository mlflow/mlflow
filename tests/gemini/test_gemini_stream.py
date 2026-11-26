from google.genai import types

from mlflow.gemini.stream import construct_streaming_response


def _chunk(*parts, index=0, finish_reason=None, total_tokens=None):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=list(parts)),
                index=index,
                finish_reason=finish_reason,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(total_token_count=total_tokens),
    )


def _parts(response, index=0):
    return [part.model_dump(exclude_none=True) for part in response.candidates[index].content.parts]


def test_construct_streaming_response_empty():
    assert construct_streaming_response([]) is None


def test_construct_streaming_response_joins_text_and_keeps_signature():
    # Shape of a live gemini-3 text stream: text pieces, then an empty part carrying the signature
    response = construct_streaming_response([
        _chunk(types.Part(text="The sea "), total_tokens=10),
        _chunk(types.Part(text="is vast."), total_tokens=20),
        _chunk(
            types.Part(text="", thought_signature=b"sig"),
            finish_reason=types.FinishReason.STOP,
            total_tokens=25,
        ),
    ])

    assert _parts(response) == [{"text": "The sea is vast.", "thought_signature": b"sig"}]
    assert response.candidates[0].finish_reason == types.FinishReason.STOP
    assert response.usage_metadata.total_token_count == 25


def test_construct_streaming_response_keeps_thoughts_separate_from_answer():
    response = construct_streaming_response([
        _chunk(types.Part(text="Multiplying ", thought=True)),
        _chunk(types.Part(text="17 by 23.", thought=True)),
        _chunk(types.Part(text="17 * 23 = 391")),
    ])

    assert _parts(response) == [
        {"text": "Multiplying 17 by 23.", "thought": True},
        {"text": "17 * 23 = 391"},
    ]


def test_construct_streaming_response_does_not_merge_function_calls():
    function_call = types.Part(
        function_call=types.FunctionCall(name="get_weather", args={"city": "Paris"}),
        thought_signature=b"sig",
    )
    response = construct_streaming_response([
        _chunk(types.Part(text="Let me check.")),
        _chunk(function_call),
        _chunk(types.Part(text=""), finish_reason=types.FinishReason.STOP),
    ])

    assert _parts(response) == [
        {"text": "Let me check."},
        function_call.model_dump(exclude_none=True),
        {"text": ""},
    ]


def test_construct_streaming_response_does_not_merge_two_signatures():
    response = construct_streaming_response([
        _chunk(types.Part(text="a", thought_signature=b"sig1")),
        _chunk(types.Part(text="b", thought_signature=b"sig2")),
    ])

    assert _parts(response) == [
        {"text": "a", "thought_signature": b"sig1"},
        {"text": "b", "thought_signature": b"sig2"},
    ]


def test_construct_streaming_response_merges_per_candidate():
    response = construct_streaming_response([
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(role="model", parts=[types.Part(text="A1 ")]),
                    index=0,
                ),
                types.Candidate(
                    content=types.Content(role="model", parts=[types.Part(text="B1 ")]),
                    index=1,
                ),
            ]
        ),
        _chunk(types.Part(text="B2"), index=1),
        _chunk(types.Part(text="A2"), index=0),
    ])

    assert [c.index for c in response.candidates] == [0, 1]
    assert _parts(response, 0) == [{"text": "A1 A2"}]
    assert _parts(response, 1) == [{"text": "B1 B2"}]
