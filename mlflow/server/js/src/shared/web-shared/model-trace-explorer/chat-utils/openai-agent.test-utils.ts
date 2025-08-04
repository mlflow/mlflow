export const MOCK_OPENAI_AGENT_INPUT = [
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

export const MOCK_OPENAI_AGENT_OUTPUT = [
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

export const MOCK_OPENAI_AGENT_COMPLEX_INPUT = [
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

export const MOCK_OPENAI_AGENT_COMPLEX_OUTPUT = [
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