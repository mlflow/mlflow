import { normalizeConversation } from '../ModelTraceExplorer.utils';

const MOCK_OPENAI_CHAT_INPUT = {
  model: 'gpt-4o-mini',
  messages: [
    {
      role: 'user',
      content: 'tell me a joke in 50 words',
    },
    {
      role: 'assistant',
      tool_calls: [
        {
          id: '1',
          function: {
            arguments: '{"joke_length": 50}',
            name: 'tell_joke',
          },
        },
      ],
    },
    {
      role: 'tool',
      content: 'Why did the scarecrow win an award? Because he was outstanding in his field!',
      tool_call_id: '1',
    },
  ],
  tools: [
    {
      function: {
        name: 'tell_joke',
        description: 'Tell a joke with specified length',
        parameters: {
          type: 'object',
          properties: {
            joke_length: {
              type: 'number',
              description: 'Length of the joke in words',
            },
          },
          required: ['joke_length'],
        },
      },
      type: 'function',
    },
  ],
};

const MOCK_OPENAI_CHAT_OUTPUT = {
  id: 'chatcmpl-A8HdoWt2DsJgtZoxjjAcPdx01jkul',
  choices: [
    {
      finish_reason: 'stop',
      index: 0,
      logprobs: null,
      message: {
        content: 'Why did the scarecrow win an award? Because he was outstanding in his field!',
        refusal: null,
        role: 'assistant',
        function_call: null,
        tool_calls: null,
      },
    },
  ],
  created: 1726537800,
  model: 'gpt-4o-mini-2024-07-18',
  object: 'chat.completion',
  service_tier: null,
  system_fingerprint: 'fp_483d39d857',
  usage: {
    completion_tokens: 17,
    prompt_tokens: 172,
    total_tokens: 189,
    completion_tokens_details: {
      accepted_prediction_tokens: 0,
      audio_tokens: 0,
      reasoning_tokens: 0,
      rejected_prediction_tokens: 0,
    },
    prompt_tokens_details: {
      audio_tokens: 0,
      cached_tokens: 0,
    },
  },
};

const MOCK_OPENAI_RESPONSES_OUTPUT = {
  id: 'resp_68916d4bd2b4819f89d9b238a190feda00db89e39031ce17',
  created_at: 1754361163,
  error: null,
  incomplete_details: null,
  instructions: null,
  metadata: {},
  model: 'o4-mini-2025-04-16',
  object: 'response',
  output: [
    {
      id: 'rs_68916d4c7ae0819f8c2a18395b3a866500db89e39031ce17',
      summary: [],
      type: 'reasoning',
      encrypted_content: null,
      status: null,
    },
    {
      id: 'msg_68916d50a8e8819f99c04a698299cf5a00db89e39031ce17',
      content: [
        {
          annotations: [],
          text: 'The capital of France is Paris.',
          type: 'output_text',
          logprobs: [],
        },
      ],
      role: 'assistant',
      status: 'completed',
      type: 'message',
    },
  ],
  parallel_tool_calls: true,
  temperature: 1,
  tool_choice: 'auto',
  tools: [],
  top_p: 1,
  background: false,
  max_output_tokens: null,
  max_tool_calls: null,
  previous_response_id: null,
  prompt: null,
  reasoning: {
    effort: 'medium',
    generate_summary: null,
    summary: null,
  },
  service_tier: 'default',
  status: 'completed',
  text: {
    format: {
      type: 'text',
    },
  },
  top_logprobs: 0,
  truncation: 'disabled',
  usage: {
    input_tokens: 13,
    input_tokens_details: {
      cached_tokens: 0,
    },
  },
  user: null,
  prompt_cache_key: null,
  safety_identifier: null,
  store: true,
};

const MOCK_OPENAI_RESPONSES_INPUT = {
  model: 'o4-mini',
  input: 'What is the capital of France?',
};

describe('normalizeConversation', () => {
  it('handles an OpenAI chat input', () => {
    expect(normalizeConversation(MOCK_OPENAI_CHAT_INPUT, 'openai')).toEqual([
      expect.objectContaining({
        role: 'user',
        content: 'tell me a joke in 50 words',
      }),
      expect.objectContaining({
        role: 'assistant',
        tool_calls: [
          {
            id: '1',
            function: {
              arguments: expect.stringContaining('joke_length'),
              name: 'tell_joke',
            },
          },
        ],
      }),
      expect.objectContaining({
        role: 'tool',
        content: 'Why did the scarecrow win an award? Because he was outstanding in his field!',
        tool_call_id: '1',
      }),
    ]);
  });

  it('handles an OpenAI chat output', () => {
    expect(normalizeConversation(MOCK_OPENAI_CHAT_OUTPUT, 'openai')).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content: 'Why did the scarecrow win an award? Because he was outstanding in his field!',
      }),
    ]);
  });

  it('handles an OpenAI responses formats', () => {
    expect(normalizeConversation(MOCK_OPENAI_RESPONSES_INPUT, 'openai')).toEqual([
      expect.objectContaining({
        role: 'user',
        content: 'What is the capital of France?',
      }),
    ]);
    expect(normalizeConversation(MOCK_OPENAI_RESPONSES_OUTPUT, 'openai')).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content: 'The capital of France is Paris.',
      }),
    ]);
  });
});
