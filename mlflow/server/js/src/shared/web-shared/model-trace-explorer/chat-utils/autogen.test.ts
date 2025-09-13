import { normalizeConversation } from '../ModelTraceExplorer.utils';

const MOCK_AUTOGEN_INPUT = {
  messages: [
    {
      content: 'You are a helpful assistant.',
      type: 'SystemMessage',
    },
    {
      content: "Say 'Hello World!'",
      source: 'user',
      type: 'UserMessage',
    },
  ],
  tools: [],
  json_output: null,
  cancellation_token: '<autogen_core._cancellation_token.CancellationToken object at 0x161f644d0>',
};

const MOCK_AUTOGEN_OUTPUT = {
  finish_reason: 'stop',
  content: 'Hello World!',
  usage: {
    prompt_tokens: 23,
    completion_tokens: 3,
  },
  cached: false,
  logprobs: null,
  thought: null,
};

const MOCK_AUTOGEN_COMPLEX_INPUT = {
  messages: [
    {
      content: 'You are a helpful assistant that can use tools to get information.',
      type: 'SystemMessage',
    },
    {
      content: 'What is the weather like in Tokyo today?',
      source: 'user',
      type: 'UserMessage',
    },
    {
      content: [
        {
          id: 'tool_call_1',
          name: 'get_weather',
          arguments: '{"city": "Tokyo"}',
        },
      ],
      source: 'assistant',
      type: 'AssistantMessage',
    },
    {
      content: {
        weather: 'sunny',
        temperature: 25,
        humidity: 60,
      },
      source: 'function',
      type: 'FunctionMessage',
    },
  ],
  tools: [
    {
      name: 'get_weather',
      description: 'Get weather information for a city',
      parameters: {
        type: 'object',
        properties: {
          city: {
            type: 'string',
            description: 'The city name',
          },
        },
        required: ['city'],
      },
    },
  ],
  json_output: null,
  cancellation_token: '<autogen_core._cancellation_token.CancellationToken object at 0x161f644d1>',
};

describe('normalizeConversation', () => {
  it('handles an AutoGen request formats', () => {
    expect(normalizeConversation(MOCK_AUTOGEN_INPUT, 'autogen')).toEqual([
      expect.objectContaining({
        role: 'system',
        content: 'You are a helpful assistant.',
      }),
      expect.objectContaining({
        role: 'user',
        content: "Say 'Hello World!'",
      }),
    ]);
  });

  it('handles an AutoGen response formats', () => {
    expect(normalizeConversation(MOCK_AUTOGEN_OUTPUT, 'autogen')).toEqual([
      expect.objectContaining({
        content: 'Hello World!',
        role: 'assistant',
      }),
    ]);
  });

  it('handles an AutoGen complex input formats', () => {
    const result = normalizeConversation(MOCK_AUTOGEN_COMPLEX_INPUT, 'autogen');
    expect(result).not.toBeNull();
    expect(result).toHaveLength(4);
    expect(result![0]).toEqual(
      expect.objectContaining({
        role: 'system',
        content: 'You are a helpful assistant that can use tools to get information.',
      }),
    );
    expect(result![1]).toEqual(
      expect.objectContaining({
        role: 'user',
        content: 'What is the weather like in Tokyo today?',
      }),
    );
    expect(result![2]).toEqual(
      expect.objectContaining({
        role: 'assistant',
        tool_calls: expect.arrayContaining([
          expect.objectContaining({
            id: 'tool_call_1',
            function: expect.objectContaining({
              name: 'get_weather',
              arguments: expect.stringContaining('"city": "Tokyo"'),
            }),
          }),
        ]),
      }),
    );
    expect(result![3]).toEqual(
      expect.objectContaining({
        role: 'user',
        content: '{"weather":"sunny","temperature":25,"humidity":60}',
      }),
    );
  });
});
