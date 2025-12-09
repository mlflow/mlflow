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

// Mock Vercel AI input with tool call and tool result
const MOCK_VERCEL_AI_TOOL_CALL_INPUT = {
  messages: [
    {
      role: 'user',
      content: [{ type: 'text', text: 'Please check the weather in Tokyo.' }],
    },
    {
      role: 'assistant',
      content: [
        { type: 'text', text: 'Calling weather tool…' },
        {
          type: 'tool-call',
          toolCallId: 'weather_tool_1',
          toolName: 'get_weather',
          input: '{"city":"Tokyo"}',
          providerOptions: {},
        },
      ],
    },
    {
      role: 'tool',
      content: [
        {
          type: 'tool-result',
          toolCallId: 'weather_tool_1',
          toolName: 'get_weather',
          output: { temp: 25, condition: 'Sunny' },
        },
      ],
    },
  ],
};

// Mock Vercel AI output with toolCalls array
const MOCK_VERCEL_AI_TOOL_CALLS_OUTPUT = {
  toolCalls: [
    {
      type: 'tool-call',
      toolCallId: 'calc_1',
      toolName: 'add',
      input: '{"a":2,"b":3}',
      providerOptions: {},
    },
  ],
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

    it('should handle Vercel AI output with toolCalls array', () => {
      const result = normalizeConversation(MOCK_VERCEL_AI_TOOL_CALLS_OUTPUT, 'vercel_ai');
      expect(result).not.toBeNull();
      expect(result).toHaveLength(1);

      const msg = result![0] as any;
      expect(msg.role).toBe('assistant');
      expect(msg.content).toBe('');
      expect(Array.isArray(msg.tool_calls)).toBe(true);
      expect(msg.tool_calls).toHaveLength(1);
      expect(msg.tool_calls[0].id).toBe('calc_1');
      expect(msg.tool_calls[0].function.name).toBe('add');
      expect(typeof msg.tool_calls[0].function.arguments).toBe('string');
      // Arguments are stringified twice in pipeline, so parse twice to verify shape
      const parsedOnce = JSON.parse(msg.tool_calls[0].function.arguments);
      const parsedTwice = JSON.parse(parsedOnce);
      expect(parsedTwice).toEqual({ a: 2, b: 3 });
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

  describe('Tool call cases', () => {
    it('should handle Vercel AI input with tool-call and tool-result parts', () => {
      const result = normalizeConversation(MOCK_VERCEL_AI_TOOL_CALL_INPUT, 'vercel_ai');
      expect(result).not.toBeNull();
      expect(result).toHaveLength(3);

      // User message
      expect(result![0]).toEqual(
        expect.objectContaining({
          role: 'user',
          content: expect.stringContaining('Please check the weather in Tokyo.'),
        }),
      );

      // Assistant tool call message
      const assistantMsg: any = result![1];
      expect(assistantMsg.role).toBe('assistant');
      expect(String(assistantMsg.content)).toContain('Calling weather tool');
      expect(Array.isArray(assistantMsg.tool_calls)).toBe(true);
      expect(assistantMsg.tool_calls).toHaveLength(1);
      expect(assistantMsg.tool_calls[0].id).toBe('weather_tool_1');
      expect(assistantMsg.tool_calls[0].function.name).toBe('get_weather');
      const toolArgsOnce = JSON.parse(assistantMsg.tool_calls[0].function.arguments);
      const toolArgs = JSON.parse(toolArgsOnce);
      expect(toolArgs).toEqual({ city: 'Tokyo' });

      // Tool result message
      expect(result![2]).toEqual(
        expect.objectContaining({
          role: 'tool',
          tool_call_id: 'weather_tool_1',
          content: expect.stringContaining('Sunny'),
        }),
      );
    });
  });
});
