import { normalizeConversation } from '../ModelTraceExplorer.utils';

const MOCK_BEDROCK_INPUT = {
  messages: [
    {
      role: 'user',
      content: [
        {
          text: 'What is the weather like in Tokyo today?',
        },
      ],
    },
  ],
};

const MOCK_BEDROCK_OUTPUT = {
  output: {
    message: {
      role: 'assistant',
      content: [
        {
          text: 'The weather in Tokyo is sunny with a temperature of 25°C.',
        },
      ],
    },
  },
};

const MOCK_BEDROCK_TOOL_USE_INPUT = {
  messages: [
    {
      role: 'user',
      content: [
        {
          text: 'Please check the weather in Tokyo and book a flight.',
        },
      ],
    },
    {
      role: 'assistant',
      content: [
        {
          text: 'I will help you check the weather and book a flight.',
        },
        {
          toolUse: {
            toolUseId: 'weather_tool_1',
            name: 'get_weather',
            input: {
              city: 'Tokyo',
            },
          },
        },
      ],
    },
    {
      role: 'user',
      content: [
        {
          toolResult: {
            toolUseId: 'weather_tool_1',
            content: [
              {
                text: 'The weather in Tokyo is sunny with 25°C.',
              },
            ],
          },
        },
      ],
    },
  ],
};

const MOCK_BEDROCK_TOOL_USE_OUTPUT = {
  output: {
    message: {
      role: 'assistant',
      content: [
        {
          text: 'Based on the weather information, I can help you book a flight to Tokyo.',
        },
        {
          toolUse: {
            toolUseId: 'flight_tool_1',
            name: 'book_flight',
            input: {
              destination: 'Tokyo',
              date: '2024-01-15',
            },
          },
        },
      ],
    },
  },
};

const MOCK_BEDROCK_IMAGE_INPUT = {
  messages: [
    {
      role: 'user',
      content: [
        {
          text: 'Describe this image:',
        },
        {
          image: {
            source: {
              bytes: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
            },
            format: 'jpeg',
          },
        },
      ],
    },
  ],
};

const MOCK_BEDROCK_REASONING_OUTPUT = {
  ResponseMetadata: {
    RequestId: '3be62cb5-d4e6-4af4-985f-3c0f57087dea',
    HTTPStatusCode: 200,
  },
  output: {
    message: {
      role: 'assistant',
      content: [
        {
          reasoningContent: {
            reasoningText: 'The user is asking for the sum of 1955 and 3865. Let me calculate:\n\n1955 + 3865 = 5820',
            signature: {
              algorithm: 'HmacSHA256',
              signature: 'RZ7n5nslCu12b5vQ7yDYfrHR1XhJ9LYRCJvZM1jF3oM=',
            },
          },
        },
        {
          text: 'The sum of 1955 and 3865 is 5820.',
        },
      ],
    },
  },
};

describe('normalizeConversation', () => {
  it('handles a Bedrock input format', () => {
    expect(normalizeConversation(MOCK_BEDROCK_INPUT, 'bedrock')).toEqual([
      expect.objectContaining({
        role: 'user',
        content: 'What is the weather like in Tokyo today?',
      }),
    ]);
  });

  it('handles a Bedrock output format', () => {
    expect(normalizeConversation(MOCK_BEDROCK_OUTPUT, 'bedrock')).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content: 'The weather in Tokyo is sunny with a temperature of 25°C.',
      }),
    ]);
  });

  it('handles a Bedrock tool use input format', () => {
    const result = normalizeConversation(MOCK_BEDROCK_TOOL_USE_INPUT, 'bedrock');
    expect(result).not.toBeNull();
    expect(result).toHaveLength(3);

    expect(result![0]).toEqual(
      expect.objectContaining({
        role: 'user',
        content: 'Please check the weather in Tokyo and book a flight.',
      }),
    );

    expect(result![1]).toEqual(
      expect.objectContaining({
        role: 'assistant',
        content: 'I will help you check the weather and book a flight.',
        tool_calls: expect.arrayContaining([
          expect.objectContaining({
            id: 'weather_tool_1',
            function: expect.objectContaining({
              name: 'get_weather',
              arguments: expect.stringContaining('"city"'),
            }),
          }),
        ]),
      }),
    );

    expect(result![2]).toEqual(
      expect.objectContaining({
        role: 'tool',
        content: 'The weather in Tokyo is sunny with 25°C.',
        tool_call_id: 'weather_tool_1',
      }),
    );
  });

  it('handles a Bedrock tool use output format', () => {
    const result = normalizeConversation(MOCK_BEDROCK_TOOL_USE_OUTPUT, 'bedrock');
    expect(result).not.toBeNull();
    expect(result).toHaveLength(1);

    expect(result![0]).toEqual(
      expect.objectContaining({
        role: 'assistant',
        content: 'Based on the weather information, I can help you book a flight to Tokyo.',
        tool_calls: expect.arrayContaining([
          expect.objectContaining({
            id: 'flight_tool_1',
            function: expect.objectContaining({
              name: 'book_flight',
              arguments: expect.stringContaining('"destination"'),
            }),
          }),
        ]),
      }),
    );
  });

  it('handles a Bedrock image input format', () => {
    const result = normalizeConversation(MOCK_BEDROCK_IMAGE_INPUT, 'bedrock');
    expect(result).not.toBeNull();
    expect(result).toHaveLength(1);

    expect(result![0]).toEqual(
      expect.objectContaining({
        role: 'user',
        content: expect.stringContaining('Describe this image: [Image: data:image/jpeg;base64,'),
      }),
    );
  });

  it('handles a Bedrock reasoning output format', () => {
    const result = normalizeConversation(MOCK_BEDROCK_REASONING_OUTPUT, 'bedrock');
    expect(result).not.toBeNull();
    expect(result).toHaveLength(1);

    expect(result![0]).toEqual(
      expect.objectContaining({
        role: 'assistant',
        content: 'The sum of 1955 and 3865 is 5820.',
      }),
    );
  });
});
