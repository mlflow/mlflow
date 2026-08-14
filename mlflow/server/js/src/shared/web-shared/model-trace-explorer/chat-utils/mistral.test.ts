import { describe, it, expect } from '@jest/globals';

import { normalizeConversation } from '../ModelTraceExplorer.utils';

// Regular Mistral input (simple string content)
const MOCK_MISTRAL_INPUT = {
  model: 'mistral-small-latest',
  messages: [
    {
      role: 'user',
      content: "How many r's are in the word 'strawberry'?",
    },
  ],
};

// Regular Mistral output (simple string content)
const MOCK_MISTRAL_OUTPUT = {
  id: 'abc123',
  object: 'chat.completion',
  model: 'mistral-small-latest',
  usage: {
    prompt_tokens: 17,
    completion_tokens: 50,
    total_tokens: 67,
  },
  created: 1700000000,
  choices: [
    {
      index: 0,
      message: {
        content: 'There are 3 r\'s in the word "strawberry".',
        tool_calls: null,
        prefix: false,
        role: 'assistant',
      },
      finish_reason: 'stop',
    },
  ],
};

// Magistral (reasoning model) output with ThinkChunk and TextChunk
const MOCK_MAGISTRAL_OUTPUT_WITH_REASONING = {
  id: 'ec078ee3cfda4bdca2abdc6725cd6440',
  object: 'chat.completion',
  model: 'magistral-small-latest',
  usage: {
    prompt_tokens: 17,
    completion_tokens: 394,
    total_tokens: 411,
  },
  created: 1766679098,
  choices: [
    {
      index: 0,
      message: {
        content: [
          {
            thinking: [
              {
                text: "Okay, let's see. The word is \"strawberry.\" I need to count how many times the letter 'r' appears in it.\n\nFirst, let's write out the word: s-t-r-a-w-b-e-r-r-y.\n\nNow, let's go through each letter one by one:\n\n1. s - not an 'r'\n2. t - not an 'r'\n3. r - that's one 'r' (count = 1)\n4. a - not an 'r'\n5. w - not an 'r'\n6. b - not an 'r'\n7. e - not an 'r'\n8. r - that's another 'r' (count = 2)\n9. r - that's another 'r' (count = 3)\n10. y - not an 'r'\n\nSo, in total, there are 3 'r's in the word \"strawberry.\"",
                type: 'text',
              },
            ],
            closed: null,
            type: 'thinking',
          },
          {
            text: "The word \"strawberry\" contains three 'r's. Here's the breakdown:\n\n- s\n- t\n- r (1)\n- a\n- w\n- b\n- e\n- r (2)\n- r (3)\n- y\n\nSo, the answer is three 'r's.",
            type: 'text',
          },
        ],
        tool_calls: null,
        prefix: false,
        role: 'assistant',
      },
      finish_reason: 'stop',
    },
  ],
};

// Multi-turn conversation input
const MOCK_MISTRAL_MULTI_TURN_INPUT = {
  model: 'mistral-small-latest',
  messages: [
    {
      role: 'system',
      content: 'You are a helpful assistant.',
    },
    {
      role: 'user',
      content: 'Hello!',
    },
    {
      role: 'assistant',
      content: 'Hi there! How can I help you today?',
    },
    {
      role: 'user',
      content: 'What is 2 + 2?',
    },
  ],
};

// Tool call output
const MOCK_MISTRAL_TOOL_CALL_OUTPUT = {
  id: 'tool123',
  object: 'chat.completion',
  model: 'mistral-small-latest',
  choices: [
    {
      index: 0,
      message: {
        content: null,
        tool_calls: [
          {
            id: 'call_abc123',
            type: 'function',
            function: {
              name: 'get_weather',
              arguments: '{"location": "San Francisco"}',
            },
          },
        ],
        role: 'assistant',
      },
      finish_reason: 'tool_calls',
    },
  ],
};

// Tool result input
const MOCK_MISTRAL_TOOL_RESULT_INPUT = {
  model: 'mistral-small-latest',
  messages: [
    {
      role: 'user',
      content: 'What is the weather in San Francisco?',
    },
    {
      role: 'assistant',
      content: null,
      tool_calls: [
        {
          id: 'call_abc123',
          type: 'function',
          function: {
            name: 'get_weather',
            arguments: '{"location": "San Francisco"}',
          },
        },
      ],
    },
    {
      role: 'tool',
      tool_call_id: 'call_abc123',
      content: '{"temperature": 72, "condition": "sunny"}',
    },
  ],
};

describe('normalizeConversation for Mistral', () => {
  describe('input normalization', () => {
    it('should handle simple mistral input', () => {
      expect(normalizeConversation(MOCK_MISTRAL_INPUT, 'mistral')).toEqual([
        expect.objectContaining({
          role: 'user',
          content: expect.stringMatching(/how many r's are in the word/i),
        }),
      ]);
    });

    it('should handle multi-turn conversation input', () => {
      const result = normalizeConversation(MOCK_MISTRAL_MULTI_TURN_INPUT, 'mistral');
      expect(result).toHaveLength(4);
      expect(result).toEqual([
        expect.objectContaining({
          role: 'system',
          content: expect.stringMatching(/you are a helpful assistant/i),
        }),
        expect.objectContaining({
          role: 'user',
          content: expect.stringMatching(/hello/i),
        }),
        expect.objectContaining({
          role: 'assistant',
          content: expect.stringMatching(/hi there/i),
        }),
        expect.objectContaining({
          role: 'user',
          content: expect.stringMatching(/what is 2 \+ 2/i),
        }),
      ]);
    });

    it('should handle tool result in input', () => {
      const result = normalizeConversation(MOCK_MISTRAL_TOOL_RESULT_INPUT, 'mistral');
      expect(result).toEqual([
        expect.objectContaining({
          role: 'user',
          content: expect.stringMatching(/weather in san francisco/i),
        }),
        expect.objectContaining({
          role: 'assistant',
          tool_calls: expect.arrayContaining([
            expect.objectContaining({
              function: expect.objectContaining({
                name: 'get_weather',
              }),
            }),
          ]),
        }),
        expect.objectContaining({
          role: 'tool',
          tool_call_id: 'call_abc123',
          content: expect.stringMatching(/temperature.*72/i),
        }),
      ]);
    });
  });

  describe('output normalization', () => {
    it('should handle simple mistral output', () => {
      expect(normalizeConversation(MOCK_MISTRAL_OUTPUT, 'mistral')).toEqual([
        expect.objectContaining({
          role: 'assistant',
          content: expect.stringMatching(/there are 3 r's in the word/i),
        }),
      ]);
    });

    it('should handle magistral output with reasoning content', () => {
      const result = normalizeConversation(MOCK_MAGISTRAL_OUTPUT_WITH_REASONING, 'mistral');
      expect(result).toHaveLength(1);
      expect(result?.[0]).toMatchObject({
        role: 'assistant',
      });
      expect(result?.[0]?.content).toMatch(/the word "strawberry" contains three/i);
      expect(result?.[0]?.reasoning).toMatch(/let's see/i);
      expect(result?.[0]?.reasoning).toMatch(/strawberry/i);
      expect(result?.[0]?.reasoning).toMatch(/count.*3/i);
    });

    it('should handle tool call output', () => {
      const result = normalizeConversation(MOCK_MISTRAL_TOOL_CALL_OUTPUT, 'mistral');
      expect(result).toEqual([
        expect.objectContaining({
          role: 'assistant',
          tool_calls: expect.arrayContaining([
            expect.objectContaining({
              id: 'call_abc123',
              function: expect.objectContaining({
                name: 'get_weather',
                arguments: expect.stringMatching(/san francisco/i),
              }),
            }),
          ]),
        }),
      ]);
    });
  });

  describe('edge cases', () => {
    it('should return null for invalid input', () => {
      expect(normalizeConversation(null, 'mistral')).toBeNull();
      expect(normalizeConversation(undefined, 'mistral')).toBeNull();
      expect(normalizeConversation('string', 'mistral')).toBeNull();
      expect(normalizeConversation(123, 'mistral')).toBeNull();
    });

    it('should return null for empty messages array', () => {
      expect(normalizeConversation({ messages: [] }, 'mistral')).toBeNull();
    });

    it('should return null for empty choices array', () => {
      expect(normalizeConversation({ choices: [] }, 'mistral')).toBeNull();
    });

    it('should handle missing content gracefully', () => {
      const inputWithNullContent = {
        messages: [
          {
            role: 'assistant',
            content: null,
          },
        ],
      };
      const result = normalizeConversation(inputWithNullContent, 'mistral');
      expect(result).toEqual([
        expect.objectContaining({
          role: 'assistant',
        }),
      ]);
    });
  });
});
