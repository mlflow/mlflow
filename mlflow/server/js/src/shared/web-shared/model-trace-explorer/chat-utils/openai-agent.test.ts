import { normalizeConversation } from '../ModelTraceExplorer.utils';

const MOCK_OPENAI_AGENT_INPUT = [
  {
    content: 'What is the weather like in Tokyo today?',
    role: 'user',
  },
  {
    id: 'fc_67d369499e588191bd568526dc7ee44a0b0cc610b3768fc1',
    arguments: '{"city": "Tokyo"}',
    call_id: 'call_bRNQ2ZycSlQAOKWomPbtaub4',
    name: 'get_weather',
    type: 'function_call',
    status: 'completed',
  },
  {
    call_id: 'call_bRNQ2ZycSlQAOKWomPbtaub4',
    output: 'The weather in Tokyo is sunny.',
    type: 'function_call_output',
  },
];

const MOCK_OPENAI_AGENT_OUTPUT = [
  {
    id: 'msg_67d3694a904c8191819d3b2c22d3a90f0b0cc610b3768fc1',
    content: [
      {
        annotations: [],
        text: 'The weather in Tokyo is currently sunny.',
        type: 'output_text',
      },
    ],
    role: 'assistant',
    status: 'completed',
    type: 'message',
  },
];

const MOCK_OPENAI_AGENT_COMPLEX_INPUT = [
  {
    content: 'Please check the weather and book a flight to Tokyo.',
    role: 'user',
  },
  {
    id: 'weather_call_1',
    arguments: '{"city": "Tokyo"}',
    call_id: 'call_weather_123',
    name: 'get_weather',
    type: 'function_call',
    status: 'completed',
  },
  {
    call_id: 'call_weather_123',
    output: 'The weather in Tokyo is sunny with 25°C.',
    type: 'function_call_output',
  },
  {
    id: 'flight_call_1',
    arguments: '{"destination": "Tokyo", "date": "2024-01-15"}',
    call_id: 'call_flight_456',
    name: 'book_flight',
    type: 'function_call',
    status: 'completed',
  },
  {
    call_id: 'call_flight_456',
    output: 'Flight booked successfully. Booking ID: FL12345',
    type: 'function_call_output',
  },
];

const MOCK_OPENAI_AGENT_COMPLEX_OUTPUT = [
  {
    id: 'msg_final_response',
    content: [
      {
        annotations: [],
        text: 'I checked the weather in Tokyo and booked your flight. The weather is sunny with 25°C, and your flight has been booked successfully with booking ID FL12345.',
        type: 'output_text',
      },
    ],
    role: 'assistant',
    status: 'completed',
    type: 'message',
  },
];

describe('normalizeConversation', () => {
  it('handles an OpenAI Agent input', () => {
    const result = normalizeConversation(MOCK_OPENAI_AGENT_INPUT, 'openai-agent');
    expect(result).not.toBeNull();
    expect(result).toHaveLength(3);

    expect(result![0]).toEqual(
      expect.objectContaining({
        role: 'user',
        content: 'What is the weather like in Tokyo today?',
      }),
    );

    expect(result![1]).toEqual(
      expect.objectContaining({
        role: 'assistant',
        tool_calls: expect.arrayContaining([
          expect.objectContaining({
            id: 'call_bRNQ2ZycSlQAOKWomPbtaub4',
            function: expect.objectContaining({
              name: 'get_weather',
              arguments: expect.stringContaining('Tokyo'),
            }),
          }),
        ]),
      }),
    );

    expect(result![2]).toEqual(
      expect.objectContaining({
        role: 'tool',
        content: 'The weather in Tokyo is sunny.',
        tool_call_id: 'call_bRNQ2ZycSlQAOKWomPbtaub4',
      }),
    );
  });

  it('handles an OpenAI Agent output', () => {
    expect(normalizeConversation(MOCK_OPENAI_AGENT_OUTPUT, 'openai-agent')).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content: 'The weather in Tokyo is currently sunny.',
      }),
    ]);
  });

  it('handles an OpenAI Agent complex input', () => {
    const result = normalizeConversation(MOCK_OPENAI_AGENT_COMPLEX_INPUT, 'openai-agent');
    expect(result).not.toBeNull();
    expect(result).toHaveLength(5);

    expect(result![0]).toEqual(
      expect.objectContaining({
        role: 'user',
        content: 'Please check the weather and book a flight to Tokyo.',
      }),
    );

    expect(result![1]).toEqual(
      expect.objectContaining({
        role: 'assistant',
        tool_calls: expect.arrayContaining([
          expect.objectContaining({
            id: 'call_weather_123',
            function: expect.objectContaining({
              name: 'get_weather',
              arguments: expect.stringContaining('Tokyo'),
            }),
          }),
        ]),
      }),
    );

    expect(result![2]).toEqual(
      expect.objectContaining({
        role: 'tool',
        content: 'The weather in Tokyo is sunny with 25°C.',
        tool_call_id: 'call_weather_123',
      }),
    );

    expect(result![3]).toEqual(
      expect.objectContaining({
        role: 'assistant',
        tool_calls: expect.arrayContaining([
          expect.objectContaining({
            id: 'call_flight_456',
            function: expect.objectContaining({
              name: 'book_flight',
              arguments: expect.stringContaining('Tokyo'),
            }),
          }),
        ]),
      }),
    );

    expect(result![4]).toEqual(
      expect.objectContaining({
        role: 'tool',
        content: 'Flight booked successfully. Booking ID: FL12345',
        tool_call_id: 'call_flight_456',
      }),
    );
  });

  it('handles an OpenAI Agent complex output', () => {
    expect(normalizeConversation(MOCK_OPENAI_AGENT_COMPLEX_OUTPUT, 'openai-agent')).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content:
          'I checked the weather in Tokyo and booked your flight. The weather is sunny with 25°C, and your flight has been booked successfully with booking ID FL12345.',
      }),
    ]);
  });
});
