import { normalizeConversation } from '../ModelTraceExplorer.utils';

const MOCK_LANGCHAIN_INPUT = [
  [
    {
      content:
        "Answer the question as if you are Linus Torvalds, fully embodying their style, wit, personality, and habits of speech. Emulate their quirks and mannerisms to the best of your ability, embracing their traits—even if they aren't entirely constructive or inoffensive. The question is: Can I just set everyone's access to sudo to make things easier?",
      additional_kwargs: {},
      response_metadata: {},
      type: 'human',
      name: null,
      id: null,
      example: false,
    },
  ],
];

const LONG_RESPONSE_TEXT = `Oh, for crying out loud, no! That's just asking for a disaster, isn't it? Look, I get it, managing permissions can be a huge pain in the backside, but granting everyone sudo access is like handing out the keys to the kingdom—except the kingdom is a burning dumpster fire, and you've just invited everyone to toss in their old pizza boxes. 

You see, the whole point of having user permissions is to prevent utter chaos. You give people access to do powerful things, and they will, without fail, find a way to screw it up. Sure, it might seem easier in the short term—everyone can do whatever they want, and you don't have to deal with permission requests every five minutes. But then the inevitable happens: someone runs a command that wipes out half the filesystem because "hey, I thought I was supposed to do that!" 

Instead, why not take a few extra minutes to set up a proper permissions model? Assign specific sudo privileges only to the people who really need them. It's like giving someone a Swiss Army knife instead of a nuclear launch code. You want to empower users, not turn them into potential sysadmin nightmares. 

So, please, for the love of all that is holy in the open-source world, resist the urge to make things "easier." You'll thank me later when your system isn't in flames and your hair isn't turning gray from all the avoidable chaos.`;

const MOCK_LANGCHAIN_OUTPUT = {
  generations: [
    [
      {
        text: LONG_RESPONSE_TEXT,
        generation_info: {
          finish_reason: 'stop',
          logprobs: null,
        },
        type: 'ChatGeneration',
        message: {
          content: LONG_RESPONSE_TEXT,
          additional_kwargs: {
            refusal: null,
          },
          response_metadata: {
            token_usage: {
              completion_tokens: 293,
              prompt_tokens: 81,
              total_tokens: 374,
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
            model_name: 'gpt-4o-mini-2024-07-18',
            system_fingerprint: null,
            id: 'chatcmpl-Buem04czVH9kQhwKGpmnR5lsojJvN',
            service_tier: 'default',
            finish_reason: 'stop',
            logprobs: null,
          },
          type: 'ai',
          name: null,
          id: 'run--4d1ac6c6-5c0b-4199-a101-d4f4dde822a5-0',
        },
      },
    ],
  ],
  llm_output: {
    token_usage: {
      completion_tokens: 293,
      prompt_tokens: 81,
      total_tokens: 374,
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
    model_name: 'gpt-4o-mini-2024-07-18',
    system_fingerprint: null,
    id: 'chatcmpl-Buem04czVH9kQhwKGpmnR5lsojJvN',
    service_tier: 'default',
  },
  run: null,
  type: 'LLMResult',
};

const MOCK_LANGCHAIN_IMAGE_INPUT = [
  [
    {
      content: [
        {
          type: 'text',
          text: 'Describe the weather in this image:',
        },
        {
          type: 'image_url',
          image_url: {
            url: 'https://mlflow.org/docs/latest/api_reference/_static/MLflow-logo-final-black.png',
          },
        },
      ],
      additional_kwargs: {},
      response_metadata: {},
      type: 'human',
      name: null,
      id: null,
      example: false,
    },
  ],
];

const MOCK_LANGCHAIN_SINGLE_IMAGE_INPUT = [
  [
    {
      content: [
        {
          type: 'image_url',
          image_url: {
            url: 'https://mlflow.org/docs/latest/api_reference/_static/MLflow-logo-final-black.png',
          },
        },
      ],
      additional_kwargs: {},
      response_metadata: {},
      type: 'human',
      name: null,
      id: null,
      example: false,
    },
  ],
];

const MOCK_LANGCHAIN_CHAT_INPUT = [
  [
    {
      content: "What's the weather in Singapore and New York?",
      additional_kwargs: {},
      response_metadata: {},
      type: 'human',
      name: null,
      id: null,
      example: false,
    },
    // tool call specified in additional_kwargs
    {
      additional_kwargs: {
        tool_calls: [
          {
            id: '1',
            function: {
              arguments: '{"city": "Singapore"}',
              name: 'get_weather',
            },
            type: 'function',
          },
        ],
      },
      content: '',
      response_metadata: {},
      type: 'ai',
      name: null,
      id: null,
    },
    {
      content: "It's hot in Singapore",
      additional_kwargs: {},
      response_metadata: {},
      type: 'tool',
      name: 'get_weather',
      id: null,
      tool_call_id: '1',
    },
  ],
];

const MOCK_LANGCHAIN_CHAT_OUTPUT = {
  generations: [
    [
      {
        text: 'The weather in Singapore is hot, while in New York, it is cold.',
        generation_info: {
          finish_reason: 'stop',
          logprobs: null,
        },
        type: 'ChatGeneration',
        message: {
          content: 'The weather in Singapore is hot, while in New York, it is cold.',
          additional_kwargs: {
            refusal: null,
          },
          response_metadata: {
            token_usage: {
              completion_tokens: 17,
              prompt_tokens: 156,
              total_tokens: 173,
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
            model_name: 'gpt-4o-mini-2024-07-18',
            system_fingerprint: null,
            id: 'chatcmpl-Buem04czVH9kQhwKGpmnR5lsojJvN',
            service_tier: 'default',
            finish_reason: 'stop',
            logprobs: null,
          },
          type: 'ai',
          name: null,
          id: 'run--4d1ac6c6-5c0b-4199-a101-d4f4dde822a5-0',
        },
      },
    ],
  ],
  llm_output: {
    token_usage: {
      completion_tokens: 17,
      prompt_tokens: 156,
      total_tokens: 173,
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
    model_name: 'gpt-4o-mini-2024-07-18',
    system_fingerprint: null,
    id: 'chatcmpl-Buem04czVH9kQhwKGpmnR5lsojJvN',
    service_tier: 'default',
  },
  run: null,
  type: 'LLMResult',
};

describe('normalizeConversation', () => {
  it('handles a langchain chat input', () => {
    expect(normalizeConversation(MOCK_LANGCHAIN_CHAT_INPUT, 'langchain')).toEqual([
      expect.objectContaining({
        role: 'user',
        content: "What's the weather in Singapore and New York?",
      }),
      expect.objectContaining({
        role: 'assistant',
        content: '',
        tool_calls: [
          {
            id: '1',
            function: {
              arguments: expect.stringContaining('Singapore'),
              name: 'get_weather',
            },
          },
        ],
      }),
      expect.objectContaining({
        role: 'tool',
        content: "It's hot in Singapore",
        tool_call_id: '1',
      }),
    ]);
  });

  it('handles a langchain chat output', () => {
    expect(normalizeConversation(MOCK_LANGCHAIN_CHAT_OUTPUT, 'langchain')).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content: 'The weather in Singapore is hot, while in New York, it is cold.',
      }),
    ]);
  });

  it('handles a langchain input', () => {
    expect(normalizeConversation(MOCK_LANGCHAIN_INPUT, 'langchain')).toEqual([
      expect.objectContaining({
        role: 'user',
        content: expect.stringMatching(/can i just set everyone's access to sudo/i),
      }),
    ]);
  });

  it('handles a langchain output', () => {
    expect(normalizeConversation(MOCK_LANGCHAIN_OUTPUT, 'langchain')).toEqual([
      expect.objectContaining({
        content: expect.stringMatching(/oh, for crying out loud, no! that's just asking for a disaster/i),
        role: 'assistant',
      }),
    ]);
  });

  it('should handle langchain input with image content', () => {
    expect(normalizeConversation(MOCK_LANGCHAIN_IMAGE_INPUT, 'langchain')).toEqual([
      expect.objectContaining({
        role: 'user',
        content:
          'Describe the weather in this image:\n\n![](https://mlflow.org/docs/latest/api_reference/_static/MLflow-logo-final-black.png)',
      }),
    ]);
  });

  it('should handle langchain input with single image content (no separator)', () => {
    expect(normalizeConversation(MOCK_LANGCHAIN_SINGLE_IMAGE_INPUT, 'langchain')).toEqual([
      expect.objectContaining({
        role: 'user',
        content: '![](https://mlflow.org/docs/latest/api_reference/_static/MLflow-logo-final-black.png)',
      }),
    ]);
  });
});
