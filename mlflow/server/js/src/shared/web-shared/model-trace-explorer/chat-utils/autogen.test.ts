import { normalizeConversation } from '../ModelTraceExplorer.utils';

export const MOCK_AUTOGEN_INPUT = {
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

export const MOCK_AUTOGEN_OUTPUT = {
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

export const MOCK_AUTOGEN_COMPLEX_INPUT = {
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

export const MOCK_AUTOGEN_COMPLEX_OUTPUT = {
  finish_reason: 'stop',
  content: 'The weather in Tokyo is sunny with a temperature of 25°C and 60% humidity.',
  usage: {
    prompt_tokens: 45,
    completion_tokens: 18,
  },
  cached: false,
  logprobs: null,
  thought: null,
};

export const MOCK_AUTOGEN_MULTIMODAL_INPUT = {
  messages: [
    {
      content: 'You are a helpful assistant that can analyze images and use tools.',
      type: 'SystemMessage',
    },
    {
      content: [
        {
          type: 'text',
          text: 'Please check the weather and then book a flight.',
        },
        {
          type: 'image_url',
          image_url: {
            url: 'https://example.com/weather-map.png',
          },
        },
      ],
      source: 'user',
      type: 'UserMessage',
    },
    {
      content: [
        {
          id: 'weather_tool',
          name: 'get_weather',
          arguments: '{"location": "Tokyo"}',
        },
        {
          id: 'flight_tool',
          name: 'book_flight',
          arguments: '{"destination": "Tokyo", "date": "2024-01-15"}',
        },
      ],
      source: 'assistant',
      type: 'AssistantMessage',
    },
    {
      content: {
        weather: 'sunny',
        temperature: 25,
      },
      source: 'function',
      type: 'FunctionMessage',
    },
    {
      content: {
        booking_id: 'FL12345',
        status: 'confirmed',
      },
      source: 'function',
      type: 'FunctionMessage',
    },
  ],
  tools: [
    {
      name: 'get_weather',
      description: 'Get weather information for a location',
      parameters: {
        type: 'object',
        properties: {
          location: {
            type: 'string',
          },
        },
        required: ['location'],
      },
    },
    {
      name: 'book_flight',
      description: 'Book a flight',
      parameters: {
        type: 'object',
        properties: {
          destination: {
            type: 'string',
          },
          date: {
            type: 'string',
          },
        },
        required: ['destination', 'date'],
      },
    },
  ],
  json_output: null,
  cancellation_token: '<autogen_core._cancellation_token.CancellationToken object at 0x161f644d2>',
};

export const MOCK_AUTOGEN_MULTIMODAL_OUTPUT = {
  finish_reason: 'stop',
  content:
    "I've checked the weather in Tokyo - it's sunny with 25°C. I've also successfully booked your flight for January 15, 2024. Your booking ID is FL12345 and it's confirmed.",
  usage: {
    prompt_tokens: 78,
    completion_tokens: 42,
  },
  cached: false,
  logprobs: null,
  thought: null,
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
