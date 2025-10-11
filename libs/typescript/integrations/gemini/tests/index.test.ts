/**
 * Unit tests for the Gemini tracing integration.
 */

type RecordedSpan = {
  span: any;
  options: any;
};

jest.mock('mlflow-tracing', () => {
  const spans: RecordedSpan[] = [];

  const withSpan = jest.fn(async (fn: (span: any) => any, options: any) => {
    const span = {
      setInputs: jest.fn(),
      setOutputs: jest.fn(),
      setAttribute: jest.fn()
    };
    spans.push({ span, options });
    return await fn(span);
  });

  return {
    withSpan,
    SpanType: { LLM: 'LLM', EMBEDDING: 'EMBEDDING' },
    SpanAttributeKey: {
      TOKEN_USAGE: 'mlflow.chat.tokenUsage'
    },
    __spans: spans,
    __reset: () => {
      spans.length = 0;
      withSpan.mockClear();
    }
  };
});

import { tracedGemini } from '../src';

const tracing = jest.requireMock('mlflow-tracing') as any;

describe('tracedGemini', () => {
  beforeEach(() => {
    tracing.__reset();
  });

  it('wraps models.generateContent with a span and records token usage', async () => {
    class Models {
      async generateContent(request: any) {
        return {
          text: () => 'Hello!',
          usageMetadata: {
            promptTokenCount: 12,
            candidatesTokenCount: 4,
            totalTokenCount: 16
          },
          request
        };
      }
    }

    class MockClient {
      models = new Models();
    }

    const client = tracedGemini(new MockClient());
    const payload = { contents: 'Hello Gemini' };

    const response = await client.models.generateContent(payload);

    expect(response.request).toBe(payload);

    expect(tracing.withSpan).toHaveBeenCalledTimes(1);
    expect(tracing.withSpan.mock.calls[0][1]).toEqual({
      name: 'Gemini.generateContent',
      spanType: tracing.SpanType.LLM
    });

    const lastSpan = tracing.__spans.at(-1)?.span;
    expect(lastSpan?.setInputs).toHaveBeenCalledWith(payload);
    expect(lastSpan?.setOutputs).toHaveBeenCalledWith(response);
    expect(lastSpan?.setAttribute).toHaveBeenCalledWith(tracing.SpanAttributeKey.TOKEN_USAGE, {
      input_tokens: 12,
      output_tokens: 4,
      total_tokens: 16
    });
  });

  it('records error status and exception on span if Gemini call throws', async () => {
    class Models {
      async generateContent(_request?: any) {
        throw new Error('rate limited');
      }
    }

    class MockClient {
      models = new Models();
    }

    const client = tracedGemini(new MockClient());

    await expect(client.models.generateContent({ prompt: 'boom' })).rejects.toThrow('rate limited');

    expect(tracing.withSpan).toHaveBeenCalledTimes(1);
    const spanEntry = tracing.__spans.at(-1)?.span;
    expect(spanEntry?.setInputs).toHaveBeenCalledWith({ prompt: 'boom' });
    expect(spanEntry?.setOutputs).not.toHaveBeenCalled();
    expect(spanEntry?.setAttribute).not.toHaveBeenCalled();
  });

  it('wraps embedContent with EMBEDDING span type and records token usage', async () => {
    class Models {
      async embedContent(document: any) {
        return {
          embedding: { values: [0.1, 0.2, 0.3] },
          usageMetadata: {
            inputTokenCount: 3,
            outputTokenCount: 0,
            totalTokenCount: 3
          },
          document
        };
      }
    }

    class MockClient {
      models = new Models();
    }

    const client = tracedGemini(new MockClient());

    const doc = { text: 'Embed me' };
    const result = await client.models.embedContent(doc);

    expect(result.document).toBe(doc);
    expect(tracing.withSpan).toHaveBeenCalledTimes(1);
    expect(tracing.withSpan.mock.calls[0][1]).toEqual({
      name: 'Gemini.embedContent',
      spanType: tracing.SpanType.EMBEDDING
    });

    const spanEntry = tracing.__spans.at(-1)?.span;
    expect(spanEntry?.setInputs).toHaveBeenCalledWith(doc);
    expect(spanEntry?.setOutputs).toHaveBeenCalledWith(result);
    expect(spanEntry?.setAttribute).toHaveBeenCalledWith(tracing.SpanAttributeKey.TOKEN_USAGE, {
      input_tokens: 3,
      output_tokens: 0,
      total_tokens: 3
    });
  });

  it('derives token usage for countTokens responses without usageMetadata', async () => {
    class Models {
      async countTokens(request: any) {
        return {
          totalTokens: 9,
          request
        };
      }
    }

    class MockClient {
      models = new Models();
    }

    const client = tracedGemini(new MockClient());

    const payload = { contents: [{ role: 'user', parts: [{ text: 'token check' }] }] };
    const response = await client.models.countTokens(payload);

    expect(response.request).toBe(payload);
    expect(tracing.withSpan.mock.calls[0][1]).toEqual({
      name: 'Gemini.countTokens',
      spanType: tracing.SpanType.LLM
    });

    const spanEntry = tracing.__spans.at(-1)?.span;
    expect(spanEntry?.setInputs).toHaveBeenCalledWith(payload);
    expect(spanEntry?.setOutputs).toHaveBeenCalledWith(response);
    expect(spanEntry?.setAttribute).toHaveBeenCalledWith(tracing.SpanAttributeKey.TOKEN_USAGE, {
      input_tokens: 9,
      output_tokens: 0,
      total_tokens: 9
    });
  });

  it('does not trace methods outside SUPPORTED_METHODS', async () => {
    class Models {
      async unrelatedMethod(_x: any) {
        return { foo: 42 };
      }
    }
    class MockClient {
      models = new Models();
    }
    const client = tracedGemini(new MockClient());
    const result = await client.models.unrelatedMethod('bar');
    expect(result).toEqual({ foo: 42 });
    expect(tracing.withSpan).not.toHaveBeenCalled();
  });

  it('does not trace nested submodules that are not named models', async () => {
    class MockSubModule {
      async generateContent(request: any) {
        return {
          usageMetadata: {
            promptTokenCount: 1,
            candidatesTokenCount: 2,
            totalTokenCount: 3
          },
          request
        };
      }
    }
    class Models {
      submodule = new MockSubModule();
    }
    class MockClient {
      models = new Models();
    }
    const client = tracedGemini(new MockClient());
    const payload = { contents: 'nested' };
    const response = await client.models.submodule.generateContent(payload);
    expect(response.request).toBe(payload);
    expect(tracing.withSpan).not.toHaveBeenCalled();
  });
});
