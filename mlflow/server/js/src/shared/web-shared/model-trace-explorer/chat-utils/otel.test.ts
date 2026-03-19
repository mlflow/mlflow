import { describe, expect, it } from '@jest/globals';

import { normalizeConversation } from '../ModelTraceExplorer.utils';

describe('normalizeConversation (OTEL GenAI)', () => {
  it('normalizes simple text messages', () => {
    const input = [
      {
        role: 'user',
        parts: [{ type: 'text', content: 'Hello there' }],
      },
      {
        role: 'assistant',
        parts: [{ type: 'text', content: 'Hi! How can I help?' }],
      },
    ];

    const result = normalizeConversation(input);

    expect(result).toEqual([
      expect.objectContaining({ role: 'user', content: 'Hello there' }),
      expect.objectContaining({ role: 'assistant', content: 'Hi! How can I help?' }),
    ]);
  });

  it('normalizes messages with non-standard role casing', () => {
    const input = [
      {
        role: 'System',
        parts: [{ type: 'text', content: 'You are a helpful assistant.' }],
      },
      {
        role: 'User',
        parts: [{ type: 'text', content: 'What is MLflow?' }],
      },
      {
        role: 'Assistant',
        parts: [{ type: 'text', content: 'MLflow is an open-source platform.' }],
      },
    ];

    const result = normalizeConversation(input);

    expect(result).toEqual([
      expect.objectContaining({ role: 'system', content: 'You are a helpful assistant.' }),
      expect.objectContaining({ role: 'user', content: 'What is MLflow?' }),
      expect.objectContaining({ role: 'assistant', content: 'MLflow is an open-source platform.' }),
    ]);
  });

  it('normalizes tool call request and response', () => {
    const input = [
      {
        role: 'assistant',
        parts: [
          { type: 'text', content: 'Let me check the weather.' },
          { type: 'tool_call', id: 'call_weather_1', name: 'get_weather', arguments: { city: 'NYC' } },
        ],
      },
      {
        role: 'assistant',
        parts: [{ type: 'tool_call_response', id: 'call_weather_1', response: { tempC: 22 } }],
      },
    ];

    const result = normalizeConversation(input);
    expect(result).toHaveLength(2);

    // First message: assistant with a tool_call
    expect(result?.[0]).toEqual(
      expect.objectContaining({
        role: 'assistant',
        content: expect.stringContaining('Let me check the weather.'),
        tool_calls: [
          expect.objectContaining({
            id: 'call_weather_1',
            function: expect.objectContaining({ name: 'get_weather', arguments: expect.any(String) }),
          }),
        ],
      }),
    );

    // Second message: tool response mapped to role 'tool'
    expect(result?.[1]).toEqual(
      expect.objectContaining({
        role: 'tool',
        tool_call_id: 'call_weather_1',
        content: expect.stringContaining('tempC'),
      }),
    );
  });
});

describe('normalizeConversation with JSON-string encoded input (issue #21812)', () => {
  it('parses JSON-string encoded OTel GenAI messages (e.g., from Java OTel SDK)', () => {
    // Java OTel SDK stores gen_ai.input.messages as a JSON string because
    // OTel Java only supports primitive attribute values.
    const messagesArray = [
      {
        role: 'user',
        parts: [{ type: 'text', content: 'What is MLflow?' }],
      },
      {
        role: 'assistant',
        parts: [{ type: 'text', content: 'MLflow is an open-source ML platform.' }],
      },
    ];
    // Simulate the value arriving as a JSON-encoded string
    const jsonStringInput = JSON.stringify(messagesArray);

    const result = normalizeConversation(jsonStringInput);

    expect(result).not.toBeNull();
    expect(result).toHaveLength(2);
    expect(result?.[0]).toEqual(expect.objectContaining({ role: 'user', content: 'What is MLflow?' }));
    expect(result?.[1]).toEqual(expect.objectContaining({ role: 'assistant', content: 'MLflow is an open-source ML platform.' }));
  });

  it('parses JSON-string encoded simple chat messages', () => {
    const messages = [
      { role: 'user', content: 'Hello!' },
      { role: 'assistant', content: 'Hi there!' },
    ];
    const jsonStringInput = JSON.stringify(messages);

    const result = normalizeConversation(jsonStringInput);

    expect(result).not.toBeNull();
    expect(result).toHaveLength(2);
    expect(result?.[0]).toEqual(expect.objectContaining({ role: 'user', content: 'Hello!' }));
    expect(result?.[1]).toEqual(expect.objectContaining({ role: 'assistant', content: 'Hi there!' }));
  });

  it('handles non-JSON strings gracefully (returns null)', () => {
    const result = normalizeConversation('not a json string');
    expect(result).toBeNull();
  });
});
