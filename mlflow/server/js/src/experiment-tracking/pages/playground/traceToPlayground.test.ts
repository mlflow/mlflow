import { describe, it, expect, jest, beforeEach } from '@jest/globals';

import { buildPlaygroundPrefillFromTrace } from './traceToPlayground';
import { BLANK_JSON_SCHEMA } from './utils';

// Isolate the mapping logic under test from the shared raw-trace parser: the parser is exercised
// by the model-trace-explorer's own tests, so here we feed it pre-parsed span nodes.
const mockParse = jest.fn();
jest.mock('@databricks/web-shared/model-trace-explorer', () => ({
  ModelSpanType: { LLM: 'LLM', CHAT_MODEL: 'CHAT_MODEL', CHAIN: 'CHAIN', TOOL: 'TOOL' },
  parseModelTraceToTreeWithMultipleRoots: (...args: any[]) => mockParse(...args),
}));

// The helper only reads a handful of fields off each node; cast minimal shapes.
const makeNode = (node: any) => node;

// A stand-in trace object; the parser is mocked so its contents are irrelevant.
const TRACE = { info: {}, data: { spans: [] } } as any;

beforeEach(() => {
  mockParse.mockReset();
});

describe('buildPlaygroundPrefillFromTrace', () => {
  it('prefills messages, endpoint, and params from the requested span', () => {
    mockParse.mockReturnValue([
      makeNode({
        key: 'span-1',
        traceId: 'tr-1',
        type: 'LLM',
        title: 'chat',
        modelName: 'gpt-4o-mini',
        inputs: { messages: [], temperature: 0.4, max_tokens: 256, stop: ['\n\n'] },
        chatMessages: [
          { role: 'system', content: 'You are helpful.' },
          { role: 'user', content: 'Hi there' },
          { role: 'assistant', content: 'Hello!' },
        ],
      }),
    ]);

    const prefill = buildPlaygroundPrefillFromTrace(TRACE, 'span-1');

    expect(prefill).not.toBeNull();
    // Only the captured input is prefilled — the assistant reply is the output being re-generated.
    expect(prefill?.messages).toEqual([
      { role: 'system', content: 'You are helpful.' },
      { role: 'user', content: 'Hi there' },
    ]);
    expect(prefill?.endpointName).toBe('gpt-4o-mini');
    expect(prefill?.params).toEqual({ temperature: 0.4, max_tokens: 256, stop: ['\n\n'] });
    expect(prefill?.spanName).toBe('chat');
    // No tools / tool choice / response format on this span → safe defaults.
    expect(prefill?.tools).toEqual([]);
    expect(prefill?.toolChoice).toBe('auto');
    expect(prefill?.responseFormatType).toBe('text');
    expect(prefill?.responseFormatSchemaText).toBe('');
  });

  it('picks the span with the most complete conversation when no span id is given', () => {
    // An agent run spreads turns across spans; the fullest span carries the final response.
    mockParse.mockReturnValue([
      makeNode({
        key: 'root',
        traceId: 'tr-1',
        type: 'CHAIN',
        title: 'agent',
        chatMessages: undefined,
        children: [
          makeNode({
            key: 'llm-1',
            traceId: 'tr-1',
            type: 'LLM',
            title: 'first-turn',
            modelName: 'claude',
            chatMessages: [
              { role: 'user', content: 'Question?' },
              {
                role: 'assistant',
                content: '',
                tool_calls: [{ id: 'c1', function: { name: 'lookup', arguments: '{}' } }],
              },
            ],
          }),
          makeNode({
            key: 'llm-2',
            traceId: 'tr-1',
            type: 'LLM',
            title: 'final-turn',
            modelName: 'claude',
            chatMessages: [
              { role: 'user', content: 'Question?' },
              {
                role: 'assistant',
                content: '',
                tool_calls: [{ id: 'c1', function: { name: 'lookup', arguments: '{}' } }],
              },
              { role: 'tool', content: '42' },
              { role: 'assistant', content: 'The answer is 42.' },
            ],
          }),
        ],
      }),
    ]);

    const prefill = buildPlaygroundPrefillFromTrace(TRACE);

    expect(prefill?.spanName).toBe('final-turn');
    // Only the input survives: the tool-call round and the final reply are the output.
    expect(prefill?.messages).toEqual([{ role: 'user', content: 'Question?' }]);
  });

  it('maps developer role to system and drops tool/function messages', () => {
    mockParse.mockReturnValue([
      makeNode({
        key: 'span-1',
        traceId: 'tr-1',
        type: 'CHAT_MODEL',
        title: 'chat',
        chatMessages: [
          { role: 'developer', content: 'system-ish' },
          { role: 'tool', content: 'tool output' },
          { role: 'user', content: 'hello' },
        ],
      }),
    ]);

    const prefill = buildPlaygroundPrefillFromTrace(TRACE, 'span-1');

    expect(prefill?.messages).toEqual([
      { role: 'system', content: 'system-ish' },
      { role: 'user', content: 'hello' },
    ]);
  });

  it('returns null when no span carries chat messages', () => {
    mockParse.mockReturnValue([
      makeNode({ key: 'root', traceId: 'tr-1', type: 'CHAIN', title: 'agent', chatMessages: undefined }),
    ]);

    expect(buildPlaygroundPrefillFromTrace(TRACE)).toBeNull();
  });

  it('returns null when the requested span id is not found', () => {
    mockParse.mockReturnValue([
      makeNode({ key: 'span-1', traceId: 'tr-1', type: 'LLM', chatMessages: [{ role: 'user', content: 'hi' }] }),
    ]);

    expect(buildPlaygroundPrefillFromTrace(TRACE, 'missing')).toBeNull();
  });

  it('prefills only the captured input, dropping tool-call rounds and the final reply', () => {
    // An agent-run conversation (e.g. PydanticAI): system + user input, then tool-call rounds
    // with tool results, then the final answer. Only the input is prefilled.
    mockParse.mockReturnValue([
      makeNode({
        key: 'span-1',
        traceId: 'tr-1',
        type: 'LLM',
        chatMessages: [
          { role: 'system', content: 'Be concise. Use the tools.' },
          { role: 'user', content: 'weather in London and Wiltshire?' },
          {
            role: 'assistant',
            content: '',
            tool_calls: [{ id: 't1', function: { name: 'get_lat_lng', arguments: '{"location":"London"}' } }],
          },
          { role: 'tool', content: '{"lat": 51.1, "lng": -0.1}' },
          { role: 'assistant', content: 'Both are sunny at 21°C.' },
        ],
      }),
    ]);

    const prefill = buildPlaygroundPrefillFromTrace(TRACE, 'span-1');
    expect(prefill?.messages).toEqual([
      { role: 'system', content: 'Be concise. Use the tools.' },
      { role: 'user', content: 'weather in London and Wiltshire?' },
    ]);
  });

  it('keeps assistant turns before the last user turn as multi-turn context', () => {
    mockParse.mockReturnValue([
      makeNode({
        key: 'span-1',
        traceId: 'tr-1',
        type: 'CHAT_MODEL',
        chatMessages: [
          { role: 'system', content: 'Be helpful.' },
          { role: 'user', content: 'First question' },
          { role: 'assistant', content: 'First answer' },
          { role: 'user', content: 'Follow-up question' },
          { role: 'assistant', content: 'Final answer' },
        ],
      }),
    ]);

    const prefill = buildPlaygroundPrefillFromTrace(TRACE, 'span-1');
    expect(prefill?.messages).toEqual([
      { role: 'system', content: 'Be helpful.' },
      { role: 'user', content: 'First question' },
      { role: 'assistant', content: 'First answer' },
      { role: 'user', content: 'Follow-up question' },
    ]);
  });

  it('drops tool-result and empty assistant turns, falling back to an empty user message', () => {
    mockParse.mockReturnValue([
      makeNode({
        key: 'span-1',
        traceId: 'tr-1',
        type: 'LLM',
        chatMessages: [
          { role: 'assistant', content: '' },
          { role: 'tool', content: 'tool result' },
        ],
      }),
    ]);

    const prefill = buildPlaygroundPrefillFromTrace(TRACE, 'span-1');
    expect(prefill?.messages).toEqual([{ role: 'user', content: '' }]);
  });

  it('serializes structured (multimodal) content to JSON so nothing is lost', () => {
    mockParse.mockReturnValue([
      makeNode({
        key: 'span-1',
        traceId: 'tr-1',
        type: 'LLM',
        chatMessages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
      }),
    ]);

    const prefill = buildPlaygroundPrefillFromTrace(TRACE, 'span-1');
    expect(prefill?.messages[0].content).toBe(JSON.stringify([{ type: 'text', text: 'hi' }], null, 2));
  });

  it('restores tools, tool choice, and response format from the span', () => {
    const parameters = { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] };
    mockParse.mockReturnValue([
      makeNode({
        key: 'span-1',
        traceId: 'tr-1',
        type: 'CHAT_MODEL',
        chatMessages: [{ role: 'user', content: 'weather?' }],
        chatTools: [{ type: 'function', function: { name: 'get_weather', description: 'Get weather', parameters } }],
        inputs: {
          tool_choice: 'required',
          response_format: { type: 'json_schema', json_schema: { schema: { type: 'object' } } },
        },
      }),
    ]);

    const prefill = buildPlaygroundPrefillFromTrace(TRACE, 'span-1');

    expect(prefill?.tools).toEqual([
      { name: 'get_weather', description: 'Get weather', params: JSON.stringify(parameters, null, 2) },
    ]);
    expect(prefill?.toolChoice).toBe('required');
    expect(prefill?.responseFormatType).toBe('json_schema');
    expect(prefill?.responseFormatSchemaText).toBe(JSON.stringify({ type: 'object' }, null, 2));
  });

  it('falls back to the blank JSON schema for tools without parameters', () => {
    mockParse.mockReturnValue([
      makeNode({
        key: 'span-1',
        traceId: 'tr-1',
        type: 'LLM',
        chatMessages: [{ role: 'user', content: 'hi' }],
        chatTools: [{ type: 'function', function: { name: 'noop' } }],
      }),
    ]);

    const prefill = buildPlaygroundPrefillFromTrace(TRACE, 'span-1');
    expect(prefill?.tools).toEqual([{ name: 'noop', description: '', params: BLANK_JSON_SCHEMA }]);
  });

  it('maps a json_object response format', () => {
    mockParse.mockReturnValue([
      makeNode({
        key: 'span-1',
        traceId: 'tr-1',
        type: 'LLM',
        chatMessages: [{ role: 'user', content: 'hi' }],
        inputs: { response_format: { type: 'json_object' } },
      }),
    ]);

    const prefill = buildPlaygroundPrefillFromTrace(TRACE, 'span-1');
    expect(prefill?.responseFormatType).toBe('json_object');
    expect(prefill?.responseFormatSchemaText).toBe('');
  });

  it('captures tool executions (pairing results to calls by id) so re-runs can auto-answer', () => {
    mockParse.mockReturnValue([
      makeNode({
        key: 'span-1',
        traceId: 'tr-1',
        type: 'LLM',
        chatMessages: [
          { role: 'user', content: 'weather?' },
          {
            role: 'assistant',
            content: '',
            tool_calls: [
              { id: 't1', function: { name: 'get_lat_lng', arguments: '{"location":"London"}' } },
              { id: 't2', function: { name: 'get_lat_lng', arguments: '{"location":"Wiltshire"}' } },
            ],
          },
          // Results arrive out of order — pairing is by tool_call_id.
          { role: 'tool', content: '{"lat":51.5}', tool_call_id: 't2' },
          { role: 'tool', content: '{"lat":51.1}', tool_call_id: 't1' },
        ],
      }),
    ]);

    const prefill = buildPlaygroundPrefillFromTrace(TRACE, 'span-1');
    expect(prefill?.toolResults).toEqual([
      { name: 'get_lat_lng', args: '{"location":"Wiltshire"}', result: '{"lat":51.5}' },
      { name: 'get_lat_lng', args: '{"location":"London"}', result: '{"lat":51.1}' },
    ]);
  });

  it('pairs tool results positionally when ids are missing and de-duplicates repeated history', () => {
    const round = [
      {
        role: 'assistant',
        content: '',
        tool_calls: [{ id: 'x', function: { name: 'get_weather', arguments: '{"city":"Paris"}' } }],
      },
      { role: 'tool', content: 'sunny' },
    ];
    mockParse.mockReturnValue([
      makeNode({ key: 's1', traceId: 'tr-1', type: 'LLM', chatMessages: [{ role: 'user', content: 'hi' }, ...round] }),
      // A later span re-records the same rounds (agent history) — no duplicate entries.
      makeNode({
        key: 's2',
        traceId: 'tr-1',
        type: 'LLM',
        chatMessages: [{ role: 'user', content: 'hi' }, ...round, ...round],
      }),
    ]);

    const prefill = buildPlaygroundPrefillFromTrace(TRACE, 's1');
    expect(prefill?.toolResults).toEqual([{ name: 'get_weather', args: '{"city":"Paris"}', result: 'sunny' }]);
  });

  it('skips tool results with an orphaned tool_call_id instead of mispairing them', () => {
    mockParse.mockReturnValue([
      makeNode({
        key: 'span-1',
        traceId: 'tr-1',
        type: 'LLM',
        chatMessages: [
          { role: 'user', content: 'hi' },
          {
            role: 'assistant',
            content: '',
            tool_calls: [{ id: 't1', function: { name: 'get_weather', arguments: '{"city":"Paris"}' } }],
          },
          // An id matching no pending call must not steal the pending get_weather call.
          { role: 'tool', content: 'orphan result', tool_call_id: 'unknown-id' },
          { role: 'tool', content: 'sunny', tool_call_id: 't1' },
        ],
      }),
    ]);

    const prefill = buildPlaygroundPrefillFromTrace(TRACE, 'span-1');
    expect(prefill?.toolResults).toEqual([{ name: 'get_weather', args: '{"city":"Paris"}', result: 'sunny' }]);
  });

  // NB: per-framework extraction of tool definitions (OpenAI / Anthropic / Gemini / Bedrock /
  // PydanticAI / ...) is normalized into `chatTools` by the shared model-trace-explorer parser
  // and covered by its own tests (chat-utils/tools.test.ts). Here the parser is mocked, so these
  // tests feed normalized `chatTools` and cover the playground-side aggregation and mapping.
  it('aggregates tools across all spans and de-duplicates by name (agent runs)', () => {
    const latLngSchema = {
      type: 'object',
      properties: { location_description: { type: 'string' } },
      required: ['location_description'],
    };
    mockParse.mockReturnValue([
      makeNode({
        key: 'root',
        traceId: 'tr-1',
        type: 'AGENT',
        title: 'Agent.run',
        chatMessages: [{ role: 'user', content: 'weather?' }],
        children: [
          makeNode({
            key: 'llm-1',
            traceId: 'tr-1',
            type: 'LLM',
            chatMessages: [{ role: 'user', content: 'weather?' }],
            chatTools: [
              {
                type: 'function',
                function: {
                  name: 'get_lat_lng',
                  description: 'Get the latitude and longitude of a location.',
                  parameters: latLngSchema,
                },
              },
            ],
          }),
          makeNode({
            key: 'llm-2',
            traceId: 'tr-1',
            type: 'LLM',
            chatMessages: [{ role: 'user', content: 'weather?' }],
            chatTools: [
              { type: 'function', function: { name: 'get_lat_lng', parameters: latLngSchema } },
              { type: 'function', function: { name: 'get_weather' } },
            ],
          }),
        ],
      }),
    ]);

    const prefill = buildPlaygroundPrefillFromTrace(TRACE);
    expect(prefill?.tools).toEqual([
      {
        name: 'get_lat_lng',
        description: 'Get the latitude and longitude of a location.',
        params: JSON.stringify(latLngSchema, null, 2),
      },
      { name: 'get_weather', description: '', params: BLANK_JSON_SCHEMA },
    ]);
  });
});
