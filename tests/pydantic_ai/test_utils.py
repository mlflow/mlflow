from mlflow.pydantic_ai.utils import parse_usage
from mlflow.tracing.constant import TokenUsageKey


class _FakeRunUsageWithCache:
    input_tokens = 1500
    output_tokens = 50
    total_tokens = 1550
    cache_read_tokens = 1200
    cache_write_tokens = 300


class _FakeRunUsageNoCache:
    input_tokens = 1500
    output_tokens = 50
    total_tokens = 1550
    cache_read_tokens = 0
    cache_write_tokens = 0


class _FakeRunUsageWithoutCacheAttrs:
    input_tokens = 1500
    output_tokens = 50
    total_tokens = 1550


class _FakeStreamedResult:
    def __init__(self, usage):
        self._usage = usage

    def usage(self):
        return self._usage


class _FakeRunResult:
    def __init__(self, usage):
        self.usage = usage


def test_parse_usage_emits_anthropic_cache_keys_when_populated():
    assert parse_usage(_FakeRunResult(_FakeRunUsageWithCache())) == {
        TokenUsageKey.INPUT_TOKENS: 1500,
        TokenUsageKey.OUTPUT_TOKENS: 50,
        TokenUsageKey.TOTAL_TOKENS: 1550,
        TokenUsageKey.CACHE_READ_INPUT_TOKENS: 1200,
        TokenUsageKey.CACHE_CREATION_INPUT_TOKENS: 300,
    }


def test_parse_usage_omits_cache_keys_when_zero():
    assert parse_usage(_FakeRunResult(_FakeRunUsageNoCache())) == {
        TokenUsageKey.INPUT_TOKENS: 1500,
        TokenUsageKey.OUTPUT_TOKENS: 50,
        TokenUsageKey.TOTAL_TOKENS: 1550,
    }


def test_parse_usage_streamed_result_callable_usage_path():
    usage = parse_usage(_FakeStreamedResult(_FakeRunUsageWithCache()))
    assert usage[TokenUsageKey.CACHE_READ_INPUT_TOKENS] == 1200
    assert usage[TokenUsageKey.CACHE_CREATION_INPUT_TOKENS] == 300


def test_parse_usage_skips_cache_keys_when_attrs_missing():
    assert parse_usage(_FakeRunResult(_FakeRunUsageWithoutCacheAttrs())) == {
        TokenUsageKey.INPUT_TOKENS: 1500,
        TokenUsageKey.OUTPUT_TOKENS: 50,
        TokenUsageKey.TOTAL_TOKENS: 1550,
    }
