import { normalizeConversation } from "../ModelTraceExplorer.utils";

export const MOCK_OPENAI_CHAT_INPUT = {
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

export const MOCK_OPENAI_CHAT_OUTPUT = {
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

export const MOCK_OPENAI_RESPONSES_OUTPUT = {
  id: 'resp_687a2b66129c81a28b51e064a54706ad05e20718052fe60e',
  created_at: 1752836966,
  error: null,
  incomplete_details: null,
  instructions: null,
  metadata: {},
  model: 'gpt-4.1-mini-2025-04-14',
  object: 'response',
  output: [
    {
      id: 'ig_687a2b666c7c81a2b5675e61cbd19d5c05e20718052fe60e',
      result: '<base64_encoded_image_data>',
      status: 'completed',
      type: 'image_generation_call',
      background: 'opaque',
      output_format: 'png',
      quality: 'high',
      revised_prompt:
        'A gray tabby cat hugging a cute otter. The otter is wearing an orange scarf. The scene is warm and affectionate, with soft lighting and a gentle background.',
      size: '1024x1024',
    },
    {
      id: 'msg_687a2b9608f081a284fff4d067ec74f005e20718052fe60e',
      content: [
        {
          annotations: [],
          text: 'Here is an image of a gray tabby cat hugging an otter wearing an orange scarf. If you want any modifications or a different style, just let me know!',
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
  tools: [
    {
      type: 'image_generation',
      background: 'auto',
      input_image_mask: null,
      model: null,
      moderation: 'auto',
      output_compression: 100,
      output_format: 'png',
      partial_images: null,
      quality: 'auto',
      size: 'auto',
      n: 1,
    },
  ],
  top_p: 1,
  background: false,
  max_output_tokens: null,
  max_tool_calls: null,
  previous_response_id: null,
  prompt: null,
  reasoning: {
    effort: null,
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
    input_tokens: 2271,
    input_tokens_details: {
      cached_tokens: 0,
    },
    output_tokens: 83,
    output_tokens_details: {
      reasoning_tokens: 0,
    },
    total_tokens: 2354,
  },
  user: null,
  store: true,
};

export const MOCK_OPENAI_RESPONSES_INPUT = {
  model: 'gpt-4.1-mini',
  input: 'Generate an image of gray tabby cat hugging an otter with an orange scarf',
  tools: [
    { type: 'image_generation', quality: 'low', size: '1024x1024', output_format: 'jpeg', output_compression: 50 },
  ],
};

describe('normalizeConversation', () => {

  it('handles an OpenAI chat input', () => {
    expect(normalizeConversation(MOCK_OPENAI_CHAT_INPUT)).toEqual([
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
    expect(normalizeConversation(MOCK_OPENAI_CHAT_OUTPUT)).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content: 'Why did the scarecrow win an award? Because he was outstanding in his field!',
      }),
    ]);
  });

  it('handles an OpenAI responses formats', () => {
    expect(normalizeConversation(MOCK_OPENAI_RESPONSES_INPUT)).toEqual([
      expect.objectContaining({
        role: 'user',
        content: 'Generate an image of gray tabby cat hugging an otter with an orange scarf',
      }),
    ]);
    expect(normalizeConversation(MOCK_OPENAI_RESPONSES_OUTPUT)).toEqual([
      expect.objectContaining({
        content: '![](data:image/png;base64,<base64_encoded_image_data>)',
        role: 'tool',
        tool_calls: undefined,
        type: 'message',
      }),
      expect.objectContaining({
        role: 'assistant',
        content: MOCK_OPENAI_RESPONSES_OUTPUT.output[1].content?.[0].text,
      }),
    ]);
  });

});