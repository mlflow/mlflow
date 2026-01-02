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

  it('normalizes OTEL messages nested under gen_ai.input.messages', () => {
    const input = [
      {
        role: 'system',
        parts: [{ type: 'text', content: 'You are helpful.' }],
      },
      {
        role: 'user',
        parts: [{ type: 'text', content: 'Hello there' }],
      },
    ];

    const resultFromString = normalizeConversation({ 'gen_ai.input.messages': JSON.stringify(input) });
    expect(resultFromString).toEqual([
      expect.objectContaining({ role: 'system', content: 'You are helpful.' }),
      expect.objectContaining({ role: 'user', content: 'Hello there' }),
    ]);

    const resultFromArray = normalizeConversation({ 'gen_ai.input.messages': input });
    expect(resultFromArray).toEqual([
      expect.objectContaining({ role: 'system', content: 'You are helpful.' }),
      expect.objectContaining({ role: 'user', content: 'Hello there' }),
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

  it('repairs malformed JSON strings with embedded tool response JSON', () => {
    const brokenPayload =
      '[{' +
      '"role":"assistant","parts":[{"type":"tool_call","id":"call_1","name":"get_random_destination","arguments":"{}"}]' +
      '},{' +
      '"role":"tool","parts":[{"type":"tool_call_response","id":"call_1","response":"{"type":"function_result","call_id":"call_1","result":"Tokyo, Japan"}"}]' +
      '},{' +
      '"role":"assistant","parts":[{"type":"text","content":"done"}]' +
      '}]';

    const result = normalizeConversation(brokenPayload);

    expect(result).toEqual([
      expect.objectContaining({ role: 'assistant', tool_calls: expect.any(Array) }),
      expect.objectContaining({ role: 'tool', tool_call_id: 'call_1', content: expect.stringContaining('Tokyo') }),
      expect.objectContaining({ role: 'assistant', content: expect.stringContaining('done') }),
    ]);
  });
});
