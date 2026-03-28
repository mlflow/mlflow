import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import { translateSpansForMlflow } from '../src/translate';

function makeSpan(attributes: Record<string, unknown> = {}): ReadableSpan {
  return {
    attributes,
    name: 'test-span',
    spanContext: () => ({
      traceId: 'abc123',
      spanId: Math.random().toString(16).slice(2, 18),
      traceFlags: 1,
    }),
  } as unknown as ReadableSpan;
}

function getAttr(span: ReadableSpan, key: string): unknown {
  return (span.attributes as Record<string, unknown>)[key];
}

function parseAttr(span: ReadableSpan, key: string): unknown {
  return JSON.parse(getAttr(span, key) as string);
}

describe('translateSpansForMlflow', () => {
  // ── Span type mapping ──────────────────────────────────────────────

  describe('span type mapping', () => {
    it.each([
      ['ai.generateText', 'LLM'],
      ['ai.generateText.doGenerate', 'LLM'],
      ['ai.streamText', 'LLM'],
      ['ai.streamText.doStream', 'LLM'],
      ['ai.generateObject', 'LLM'],
      ['ai.generateObject.doGenerate', 'LLM'],
      ['ai.streamObject', 'LLM'],
      ['ai.streamObject.doStream', 'LLM'],
      ['ai.toolCall', 'TOOL'],
      ['ai.embed', 'EMBEDDING'],
      ['ai.embed.doEmbed', 'EMBEDDING'],
      ['ai.embedMany', 'EMBEDDING'],
      ['ai.embedMany.doEmbed', 'EMBEDDING'],
    ])('maps %s → %s', (operationId, expectedType) => {
      const span = makeSpan({ 'ai.operationId': operationId });
      translateSpansForMlflow([span]);
      expect(getAttr(span, 'mlflow.spanType')).toBe(expectedType);
    });
  });

  // ── Existing attribute preservation ────────────────────────────────

  describe('does not overwrite existing attributes', () => {
    it.each([
      ['mlflow.spanType', 'CUSTOM'],
      ['mlflow.spanInputs', '{"existing":true}'],
      ['mlflow.spanOutputs', '{"existing":true}'],
      ['mlflow.llm.model', 'my-model'],
      ['mlflow.llm.provider', 'my-provider'],
      ['mlflow.message.format', 'custom_format'],
      ['mlflow.chat.tokenUsage', '{"input_tokens":99}'],
    ])('preserves existing %s', (attrKey, existingValue) => {
      const span = makeSpan({
        'ai.operationId': 'ai.generateText.doGenerate',
        'ai.prompt.messages': '[{"role":"user","content":"hello"}]',
        'ai.response.text': '"hi"',
        'ai.model.id': 'gpt-4',
        'ai.model.provider': 'openai',
        'ai.usage.promptTokens': 10,
        'ai.usage.completionTokens': 5,
        [attrKey]: existingValue,
      });
      translateSpansForMlflow([span]);
      expect(getAttr(span, attrKey)).toBe(existingValue);
    });
  });

  // ── Input extraction ───────────────────────────────────────────────

  describe('input extraction', () => {
    it.each([
      'ai.generateText.doGenerate',
      'ai.streamText.doStream',
      'ai.generateObject.doGenerate',
      'ai.streamObject.doStream',
    ])('extracts ai.prompt.* prefix attributes for %s', (operationId) => {
      const span = makeSpan({
        'ai.operationId': operationId,
        'ai.prompt.messages': '[{"role":"user","content":"hello"}]',
        'ai.prompt.temperature': '0.7',
        'ai.prompt.tools': '[]',
      });
      translateSpansForMlflow([span]);
      expect(parseAttr(span, 'mlflow.spanInputs')).toEqual({
        messages: [{ role: 'user', content: 'hello' }],
        temperature: 0.7,
        tools: [],
      });
    });

    it('extracts ai.prompt for top-level generate spans', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.generateText',
        'ai.prompt': '{"messages":[{"role":"user","content":"hi"}]}',
      });
      translateSpansForMlflow([span]);
      expect(parseAttr(span, 'mlflow.spanInputs')).toEqual({
        prompt: { messages: [{ role: 'user', content: 'hi' }] },
      });
    });

    it('extracts ai.toolCall.args for tool spans', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.toolCall',
        'ai.toolCall.args': '{"query":"weather"}',
      });
      translateSpansForMlflow([span]);
      expect(parseAttr(span, 'mlflow.spanInputs')).toEqual({
        args: { query: 'weather' },
      });
    });

    it('extracts ai.value for embed spans', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.embed',
        'ai.value': '"some text to embed"',
      });
      translateSpansForMlflow([span]);
      expect(parseAttr(span, 'mlflow.spanInputs')).toEqual({
        value: 'some text to embed',
      });
    });

    it('extracts ai.values for embedMany spans', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.embedMany',
        'ai.values': '["text1","text2"]',
      });
      translateSpansForMlflow([span]);
      expect(parseAttr(span, 'mlflow.spanInputs')).toEqual({
        value: ['text1', 'text2'],
      });
    });

    it('does not set inputs when no matching keys exist', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.generateText',
      });
      translateSpansForMlflow([span]);
      expect(getAttr(span, 'mlflow.spanInputs')).toBeUndefined();
    });
  });

  // ── Output extraction ──────────────────────────────────────────────

  describe('output extraction', () => {
    it.each([
      'ai.generateText.doGenerate',
      'ai.streamText.doStream',
      'ai.generateObject.doGenerate',
      'ai.streamObject.doStream',
    ])('extracts ai.response.* prefix attributes for %s', (operationId) => {
      const span = makeSpan({
        'ai.operationId': operationId,
        'ai.response.text': '"Hello there!"',
        'ai.response.finishReason': '"stop"',
      });
      translateSpansForMlflow([span]);
      expect(parseAttr(span, 'mlflow.spanOutputs')).toEqual({
        text: 'Hello there!',
        finishReason: 'stop',
      });
    });

    it('extracts ai.response.text for top-level spans', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.streamText',
        'ai.response.text': 'plain text response',
      });
      translateSpansForMlflow([span]);
      expect(parseAttr(span, 'mlflow.spanOutputs')).toEqual({
        text: 'plain text response',
      });
    });

    it('extracts ai.toolCall.result for tool spans', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.toolCall',
        'ai.toolCall.args': '{"q":"test"}',
        'ai.toolCall.result': '{"temp":72}',
      });
      translateSpansForMlflow([span]);
      expect(parseAttr(span, 'mlflow.spanOutputs')).toEqual({
        result: { temp: 72 },
      });
    });

    it('extracts ai.response.object for object generation', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.generateObject',
        'ai.response.object': '{"name":"test"}',
      });
      translateSpansForMlflow([span]);
      expect(parseAttr(span, 'mlflow.spanOutputs')).toEqual({
        object: { name: 'test' },
      });
    });

    it('extracts ai.embedding for embed spans', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.embed.doEmbed',
        'ai.value': '"text"',
        'ai.embedding': '[0.1,0.2,0.3]',
      });
      translateSpansForMlflow([span]);
      expect(parseAttr(span, 'mlflow.spanOutputs')).toEqual({
        embedding: [0.1, 0.2, 0.3],
      });
    });

    it('extracts ai.embeddings for embedMany spans', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.embedMany.doEmbed',
        'ai.values': '["a","b"]',
        'ai.embeddings': '[[0.1,0.2],[0.3,0.4]]',
      });
      translateSpansForMlflow([span]);
      expect(parseAttr(span, 'mlflow.spanOutputs')).toEqual({
        embeddings: [
          [0.1, 0.2],
          [0.3, 0.4],
        ],
      });
    });

    it('does not set outputs when no matching keys exist', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.generateText',
      });
      translateSpansForMlflow([span]);
      expect(getAttr(span, 'mlflow.spanOutputs')).toBeUndefined();
    });
  });

  // ── Model and provider ─────────────────────────────────────────────

  describe('model and provider', () => {
    it('extracts model from ai.model.id', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.generateText',
        'ai.model.id': 'claude-sonnet-4-20250514',
      });
      translateSpansForMlflow([span]);
      expect(getAttr(span, 'mlflow.llm.model')).toBe('claude-sonnet-4-20250514');
    });

    it('falls back to gen_ai.request.model', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.generateText',
        'gen_ai.request.model': 'gpt-4',
      });
      translateSpansForMlflow([span]);
      expect(getAttr(span, 'mlflow.llm.model')).toBe('gpt-4');
    });

    it('falls back to gen_ai.response.model', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.generateText',
        'gen_ai.response.model': 'gpt-4-turbo',
      });
      translateSpansForMlflow([span]);
      expect(getAttr(span, 'mlflow.llm.model')).toBe('gpt-4-turbo');
    });

    it('prefers ai.model.id over gen_ai fallbacks', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.generateText',
        'ai.model.id': 'primary-model',
        'gen_ai.request.model': 'fallback-model',
      });
      translateSpansForMlflow([span]);
      expect(getAttr(span, 'mlflow.llm.model')).toBe('primary-model');
    });

    it('extracts provider from ai.model.provider', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.generateText',
        'ai.model.provider': 'anthropic',
      });
      translateSpansForMlflow([span]);
      expect(getAttr(span, 'mlflow.llm.provider')).toBe('anthropic');
    });

    it('falls back to gen_ai.system for provider', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.generateText',
        'gen_ai.system': 'openai',
      });
      translateSpansForMlflow([span]);
      expect(getAttr(span, 'mlflow.llm.provider')).toBe('openai');
    });

    it('does not set model when no source exists', () => {
      const span = makeSpan({ 'ai.operationId': 'ai.generateText' });
      translateSpansForMlflow([span]);
      expect(getAttr(span, 'mlflow.llm.model')).toBeUndefined();
    });

    it('does not set provider when no source exists', () => {
      const span = makeSpan({ 'ai.operationId': 'ai.generateText' });
      translateSpansForMlflow([span]);
      expect(getAttr(span, 'mlflow.llm.provider')).toBeUndefined();
    });
  });

  // ── Message format ─────────────────────────────────────────────────

  describe('message format', () => {
    it.each([
      'ai.generateText.doGenerate',
      'ai.streamText.doStream',
      'ai.generateObject.doGenerate',
      'ai.streamObject.doStream',
    ])('sets vercel_ai for %s', (operationId) => {
      const span = makeSpan({ 'ai.operationId': operationId });
      translateSpansForMlflow([span]);
      expect(getAttr(span, 'mlflow.message.format')).toBe('vercel_ai');
    });

    it.each([
      'ai.generateText',
      'ai.streamText',
      'ai.generateObject',
      'ai.streamObject',
      'ai.toolCall',
      'ai.embed',
      'ai.embed.doEmbed',
      'ai.embedMany',
      'ai.embedMany.doEmbed',
    ])('does not set message format for %s', (operationId) => {
      const span = makeSpan({ 'ai.operationId': operationId });
      translateSpansForMlflow([span]);
      expect(getAttr(span, 'mlflow.message.format')).toBeUndefined();
    });
  });

  // ── Token usage ────────────────────────────────────────────────────

  describe('token usage', () => {
    it('maps both prompt and completion tokens', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.generateText',
        'ai.usage.promptTokens': 100,
        'ai.usage.completionTokens': 50,
      });
      translateSpansForMlflow([span]);
      expect(parseAttr(span, 'mlflow.chat.tokenUsage')).toEqual({
        input_tokens: 100,
        output_tokens: 50,
        total_tokens: 150,
      });
    });

    it('defaults output_tokens to 0 when only promptTokens present', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.generateText',
        'ai.usage.promptTokens': 100,
      });
      translateSpansForMlflow([span]);
      expect(parseAttr(span, 'mlflow.chat.tokenUsage')).toEqual({
        input_tokens: 100,
        output_tokens: 0,
        total_tokens: 100,
      });
    });

    it('defaults input_tokens to 0 when only completionTokens present', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.generateText',
        'ai.usage.completionTokens': 50,
      });
      translateSpansForMlflow([span]);
      expect(parseAttr(span, 'mlflow.chat.tokenUsage')).toEqual({
        input_tokens: 0,
        output_tokens: 50,
        total_tokens: 50,
      });
    });

    it('does not set tokenUsage when neither token count exists', () => {
      const span = makeSpan({ 'ai.operationId': 'ai.generateText' });
      translateSpansForMlflow([span]);
      expect(getAttr(span, 'mlflow.chat.tokenUsage')).toBeUndefined();
    });

    it('parses string-encoded token counts', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.generateText',
        'ai.usage.promptTokens': '150',
        'ai.usage.completionTokens': '75',
      });
      translateSpansForMlflow([span]);
      expect(parseAttr(span, 'mlflow.chat.tokenUsage')).toEqual({
        input_tokens: 150,
        output_tokens: 75,
        total_tokens: 225,
      });
    });
  });

  // ── Non-AI spans ───────────────────────────────────────────────────

  describe('non-AI spans', () => {
    it('leaves spans without ai.operationId fully untouched', () => {
      const span = makeSpan({
        'http.method': 'GET',
        'http.url': 'https://example.com',
      });
      const originalAttrs = { ...span.attributes };
      translateSpansForMlflow([span]);
      expect(span.attributes).toEqual(originalAttrs);
    });
  });

  // ── Double-encoded JSON ────────────────────────────────────────────

  describe('double-encoded JSON handling', () => {
    it('parses double-encoded JSON strings', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.generateText.doGenerate',
        'ai.prompt.messages': '"[{\\"role\\":\\"user\\",\\"content\\":\\"hi\\"}]"',
      });
      translateSpansForMlflow([span]);
      expect(parseAttr(span, 'mlflow.spanInputs')).toEqual({
        messages: [{ role: 'user', content: 'hi' }],
      });
    });

    it('handles plain string values gracefully', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.generateText.doGenerate',
        'ai.response.text': 'just a plain string',
      });
      translateSpansForMlflow([span]);
      expect(parseAttr(span, 'mlflow.spanOutputs')).toEqual({
        text: 'just a plain string',
      });
    });

    it('handles invalid JSON strings gracefully', () => {
      const span = makeSpan({
        'ai.operationId': 'ai.generateText.doGenerate',
        'ai.response.text': '{not valid json',
      });
      translateSpansForMlflow([span]);
      expect(parseAttr(span, 'mlflow.spanOutputs')).toEqual({
        text: '{not valid json',
      });
    });
  });

  // ── Error resilience ───────────────────────────────────────────────

  describe('error resilience', () => {
    it('continues translating after a span throws', () => {
      // Create a span with a getter that throws
      const badSpan = makeSpan({});
      Object.defineProperty(badSpan, 'attributes', {
        get() {
          throw new Error('boom');
        },
      });

      const goodSpan = makeSpan({
        'ai.operationId': 'ai.generateText',
        'ai.model.id': 'gpt-4',
      });

      // Both spans should be in the array after translation
      const spans = [badSpan, goodSpan];
      translateSpansForMlflow(spans);

      // Good span was translated
      expect(getAttr(goodSpan, 'mlflow.spanType')).toBe('LLM');
      expect(getAttr(goodSpan, 'mlflow.llm.model')).toBe('gpt-4');

      // Both spans remain in the array (nothing dropped)
      expect(spans).toHaveLength(2);
    });
  });

  // ── Multiple spans in a batch ──────────────────────────────────────

  describe('batch processing', () => {
    it('translates all spans in a batch', () => {
      const llmSpan = makeSpan({
        'ai.operationId': 'ai.generateText',
        'ai.model.id': 'gpt-4',
      });
      const toolSpan = makeSpan({
        'ai.operationId': 'ai.toolCall',
        'ai.toolCall.args': '{"q":"test"}',
      });
      const embedSpan = makeSpan({
        'ai.operationId': 'ai.embed',
        'ai.value': '"hello"',
      });
      const httpSpan = makeSpan({
        'http.method': 'GET',
      });

      translateSpansForMlflow([llmSpan, toolSpan, embedSpan, httpSpan]);

      expect(getAttr(llmSpan, 'mlflow.spanType')).toBe('LLM');
      expect(getAttr(toolSpan, 'mlflow.spanType')).toBe('TOOL');
      expect(getAttr(embedSpan, 'mlflow.spanType')).toBe('EMBEDDING');
      expect(getAttr(httpSpan, 'mlflow.spanType')).toBeUndefined();
    });
  });
});
