import { describe, expect, it } from '@jest/globals';
import { normalizeConversation } from '../ModelTraceExplorer.utils';

// Mock Vercel AI input with messages format
const MOCK_VERCEL_AI_MESSAGES_INPUT = {
  messages: [
    {
      role: 'user',
      content: [
        {
          type: 'text',
          text: 'What is the weather like today?',
        },
      ],
    },
    {
      role: 'assistant',
      content: [
        {
          type: 'text',
          text: 'I need more information. Where are you located?',
        },
      ],
    },
  ],
};

// Mock Vercel AI input with prompt format
const MOCK_VERCEL_AI_PROMPT_INPUT = {
  prompt: 'Tell me a joke',
};

// Mock Vercel AI input with string content
const MOCK_VERCEL_AI_STRING_CONTENT_INPUT = {
  messages: [
    {
      role: 'user',
      content: 'Hello, how are you?',
    },
  ],
};

// Mock Vercel AI input with image content
const MOCK_VERCEL_AI_IMAGE_INPUT = {
  messages: [
    {
      role: 'user',
      content: [
        {
          type: 'text',
          text: 'What is in this image?',
        },
        {
          type: 'image',
          image: 'https://example.com/image.jpg',
        },
      ],
    },
  ],
};

// Mock Vercel AI output with text
const MOCK_VERCEL_AI_TEXT_OUTPUT = {
  text: 'This is the response from the AI',
};

// Mock Vercel AI output with response.text
const MOCK_VERCEL_AI_RESPONSE_TEXT_OUTPUT = {
  response: {
    text: 'This is the response text',
  },
};

// Mock Vercel AI output with messages
const MOCK_VERCEL_AI_MESSAGES_OUTPUT = {
  messages: [
    {
      role: 'assistant',
      content: 'Here is my response',
    },
  ],
};

// Mock Vercel AI output with response.messages
const MOCK_VERCEL_AI_RESPONSE_MESSAGES_OUTPUT = {
  response: {
    messages: [
      {
        role: 'assistant',
        content: [
          {
            type: 'text',
            text: 'This is a detailed response',
          },
        ],
      },
    ],
  },
};

describe('normalizeConversation - Vercel AI', () => {
  describe('Input formats', () => {
    it('should handle Vercel AI messages input with content array', () => {
      const result = normalizeConversation(MOCK_VERCEL_AI_MESSAGES_INPUT, 'vercel_ai');
      expect(result).toEqual([
        expect.objectContaining({
          role: 'user',
          content: expect.stringMatching(/what is the weather like today/i),
        }),
        expect.objectContaining({
          role: 'assistant',
          content: expect.stringMatching(/i need more information/i),
        }),
      ]);
    });

    it('should handle Vercel AI prompt input', () => {
      const result = normalizeConversation(MOCK_VERCEL_AI_PROMPT_INPUT, 'vercel_ai');
      expect(result).toEqual([
        expect.objectContaining({
          role: 'user',
          content: 'Tell me a joke',
        }),
      ]);
    });

    it('should handle Vercel AI input with string content', () => {
      const result = normalizeConversation(MOCK_VERCEL_AI_STRING_CONTENT_INPUT, 'vercel_ai');
      expect(result).toEqual([
        expect.objectContaining({
          role: 'user',
          content: 'Hello, how are you?',
        }),
      ]);
    });

    it('should handle Vercel AI input with image content', () => {
      const result = normalizeConversation(MOCK_VERCEL_AI_IMAGE_INPUT, 'vercel_ai');
      expect(result).toEqual([
        expect.objectContaining({
          role: 'user',
          content: expect.stringMatching(/what is in this image/i),
        }),
      ]);
    });
  });

  describe('Output formats', () => {
    it('should handle Vercel AI output with text', () => {
      const result = normalizeConversation(MOCK_VERCEL_AI_TEXT_OUTPUT, 'vercel_ai');
      expect(result).toEqual([
        expect.objectContaining({
          role: 'assistant',
          content: 'This is the response from the AI',
        }),
      ]);
    });

    it('should handle Vercel AI output with response.text', () => {
      const result = normalizeConversation(MOCK_VERCEL_AI_RESPONSE_TEXT_OUTPUT, 'vercel_ai');
      expect(result).toEqual([
        expect.objectContaining({
          role: 'assistant',
          content: 'This is the response text',
        }),
      ]);
    });

    it('should handle Vercel AI output with messages', () => {
      const result = normalizeConversation(MOCK_VERCEL_AI_MESSAGES_OUTPUT, 'vercel_ai');
      expect(result).toEqual([
        expect.objectContaining({
          role: 'assistant',
          content: 'Here is my response',
        }),
      ]);
    });

    it('should handle Vercel AI output with response.messages', () => {
      const result = normalizeConversation(MOCK_VERCEL_AI_RESPONSE_MESSAGES_OUTPUT, 'vercel_ai');
      expect(result).toEqual([
        expect.objectContaining({
          role: 'assistant',
          content: expect.stringMatching(/this is a detailed response/i),
        }),
      ]);
    });
  });

  describe('Edge cases', () => {
    it('should return null for invalid input', () => {
      expect(normalizeConversation({ invalid: 'data' }, 'vercel_ai')).toBeNull();
    });

    it('should return null for null input', () => {
      expect(normalizeConversation(null, 'vercel_ai')).toBeNull();
    });

    it('should return null for undefined input', () => {
      expect(normalizeConversation(undefined, 'vercel_ai')).toBeNull();
    });
  });
});
