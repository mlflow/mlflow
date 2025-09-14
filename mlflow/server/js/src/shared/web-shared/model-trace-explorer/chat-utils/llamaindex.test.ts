import { normalizeConversation } from '../ModelTraceExplorer.utils';

const MOCK_LLAMAINDEX_INPUT = {
  messages: [
    {
      role: 'system',
      additional_kwargs: {},
      blocks: [
        {
          block_type: 'text',
          text: "You are an expert Q&A system that is trusted around the world.\nAlways answer the query using the provided context information, and not prior knowledge.\nSome rules to follow:\n1. Never directly reference the given context in your answer.\n2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.",
        },
      ],
    },
    {
      role: 'user',
      additional_kwargs: {},
      blocks: [
        {
          block_type: 'text',
          text: 'Context information is below.\n---------------------\n<CONTEXT>\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: What was the first program the author wrote?\nAnswer: ',
        },
      ],
    },
  ],
};

const MOCK_LLAMAINDEX_OUTPUT = {
  message: {
    role: 'assistant',
    additional_kwargs: {},
    blocks: [
      {
        block_type: 'text',
        text: 'The first program the author wrote was "start" which runs the command "craco start".',
      },
    ],
  },
  raw: {
    id: 'chatcmpl-Buf4F4nbgPv5tJBMIN2sBqOyZdlXz',
    choices: [
      {
        finish_reason: 'stop',
        index: 0,
        logprobs: null,
        message: {
          content: 'The first program the author wrote was "start" which runs the command "craco start".',
          refusal: null,
          role: 'assistant',
          annotations: [],
          audio: null,
          function_call: null,
          tool_calls: null,
        },
      },
    ],
    created: 1752843931,
    model: 'gpt-3.5-turbo-0125',
    object: 'chat.completion',
    service_tier: 'default',
    system_fingerprint: null,
    usage: {
      completion_tokens: 19,
      prompt_tokens: 2046,
      total_tokens: 2065,
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
  },
  delta: null,
  logprobs: null,
  additional_kwargs: {
    prompt_tokens: 2046,
    completion_tokens: 19,
    total_tokens: 2065,
  },
};

describe('normalizeConversation', () => {
  it('handles a LlamaIndex chat input', () => {
    expect(normalizeConversation(MOCK_LLAMAINDEX_INPUT, 'llamaindex')).toEqual([
      expect.objectContaining({
        role: 'system',
        content: expect.stringMatching(/you are an expert q&a system/i),
      }),
      expect.objectContaining({
        role: 'user',
        content: expect.stringMatching(/what was the first program the author wrote/i),
      }),
    ]);
  });

  it('handles a LlamaIndex chat output', () => {
    expect(normalizeConversation(MOCK_LLAMAINDEX_OUTPUT, 'llamaindex')).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content: expect.stringMatching(/the first program the author wrote was/i),
      }),
    ]);
  });
});
