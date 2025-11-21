import { describe, it, expect } from '@jest/globals';

import { normalizeConversation } from '../ModelTraceExplorer.utils';

// Mock PydanticAI messages based on the actual structure from pydantic_ai.messages module
const MOCK_PYDANTIC_AI_REQUEST = {
  kind: 'request',
  parts: [
    {
      part_kind: 'system-prompt',
      content: 'You are a helpful assistant.',
      timestamp: '2025-11-10T03:25:10.268933',
      dynamic_ref: null,
    },
    {
      part_kind: 'user-prompt',
      content: 'Hello, how are you?',
      timestamp: '2025-11-10T03:25:10.268955',
    },
  ],
};

const MOCK_PYDANTIC_AI_RESPONSE = {
  kind: 'response',
  parts: [
    {
      part_kind: 'text',
      content: 'I am doing well, thank you!',
      id: null,
    },
  ],
  usage: {
    input_tokens: 20,
    output_tokens: 10,
    total_tokens: 30,
  },
  model_name: 'claude-3-7-sonnet-latest',
  timestamp: '2025-11-10T03:25:10.268957',
  provider_name: 'anthropic',
  provider_details: null,
  provider_response_id: 'msg_123',
  finish_reason: 'stop',
};

const MOCK_PYDANTIC_AI_REQUEST_WITH_TOOL_RETURN = {
  kind: 'request',
  parts: [
    {
      part_kind: 'user-prompt',
      content: "What's the weather in San Francisco?",
      timestamp: '2025-11-10T03:25:25.690753',
    },
  ],
};

const MOCK_PYDANTIC_AI_RESPONSE_WITH_TOOL_CALL = {
  kind: 'response',
  parts: [
    {
      part_kind: 'tool-call',
      tool_name: 'get_weather',
      args: {
        location: 'San Francisco',
        unit: 'celsius',
      },
      tool_call_id: 'call_abc123',
      id: null,
    },
  ],
  usage: {
    input_tokens: 30,
    output_tokens: 15,
    total_tokens: 45,
  },
  model_name: 'claude-3-7-sonnet-latest',
  timestamp: '2025-11-10T03:25:25.690777',
  provider_name: 'anthropic',
  finish_reason: 'tool_calls',
};

const MOCK_PYDANTIC_AI_REQUEST_WITH_TOOL_RESULT = {
  kind: 'request',
  parts: [
    {
      part_kind: 'tool-return',
      tool_name: 'get_weather',
      content: {
        temperature: 18,
        condition: 'sunny',
      },
      tool_call_id: 'call_abc123',
      metadata: null,
      timestamp: '2025-11-10T03:25:25.690777',
    },
  ],
};

const MOCK_PYDANTIC_AI_RESPONSE_FINAL = {
  kind: 'response',
  parts: [
    {
      part_kind: 'text',
      content: 'The weather in San Francisco is 18°C and sunny.',
      id: null,
    },
  ],
  usage: {
    input_tokens: 45,
    output_tokens: 12,
    total_tokens: 57,
  },
  model_name: 'claude-3-7-sonnet-latest',
  timestamp: '2025-11-10T03:25:25.690779',
  provider_name: 'anthropic',
  finish_reason: 'stop',
};

describe('PydanticAI message normalization', () => {
  describe('normalizeConversation with pydantic_ai format', () => {
    it('should normalize a simple request with system and user prompts', () => {
      const result = normalizeConversation(MOCK_PYDANTIC_AI_REQUEST, 'pydantic_ai');

      expect(result).not.toBeNull();
      expect(result).toHaveLength(2);

      expect(result![0]).toEqual({
        role: 'system',
        content: 'You are a helpful assistant.',
      });

      expect(result![1]).toEqual({
        role: 'user',
        content: 'Hello, how are you?',
      });
    });

    it('should normalize a simple response with text', () => {
      const result = normalizeConversation(MOCK_PYDANTIC_AI_RESPONSE, 'pydantic_ai');

      expect(result).not.toBeNull();
      expect(result).toHaveLength(1);

      expect(result![0]).toEqual({
        role: 'assistant',
        content: 'I am doing well, thank you!',
      });
    });

    it('should normalize an array of messages (request + response)', () => {
      const messages = [MOCK_PYDANTIC_AI_REQUEST, MOCK_PYDANTIC_AI_RESPONSE];
      const result = normalizeConversation(messages, 'pydantic_ai');

      expect(result).not.toBeNull();
      expect(result).toHaveLength(3); // system + user + assistant

      expect(result![0].role).toBe('system');
      expect(result![1].role).toBe('user');
      expect(result![2].role).toBe('assistant');
    });

    it('should normalize a response with tool calls', () => {
      const result = normalizeConversation(MOCK_PYDANTIC_AI_RESPONSE_WITH_TOOL_CALL, 'pydantic_ai');

      expect(result).not.toBeNull();
      expect(result).toHaveLength(1);

      expect(result![0]).toMatchObject({
        role: 'assistant',
        tool_calls: [
          {
            id: 'call_abc123',
            function: {
              name: 'get_weather',
              arguments: expect.stringContaining('San Francisco'),
            },
          },
        ],
      });

      const args = JSON.parse(result![0].tool_calls![0].function.arguments);
      expect(args).toEqual({
        location: 'San Francisco',
        unit: 'celsius',
      });
    });

    it('should normalize a request with tool return', () => {
      const result = normalizeConversation(MOCK_PYDANTIC_AI_REQUEST_WITH_TOOL_RESULT, 'pydantic_ai');

      expect(result).not.toBeNull();
      expect(result).toHaveLength(1);

      expect(result![0]).toMatchObject({
        role: 'tool',
        tool_call_id: 'call_abc123',
        content: expect.stringContaining('temperature'),
      });

      const content = JSON.parse(result![0].content!);
      expect(content).toEqual({
        temperature: 18,
        condition: 'sunny',
      });
    });

    it('should normalize a complete conversation with tool calls', () => {
      const messages = [
        MOCK_PYDANTIC_AI_REQUEST_WITH_TOOL_RETURN,
        MOCK_PYDANTIC_AI_RESPONSE_WITH_TOOL_CALL,
        MOCK_PYDANTIC_AI_REQUEST_WITH_TOOL_RESULT,
        MOCK_PYDANTIC_AI_RESPONSE_FINAL,
      ];

      const result = normalizeConversation(messages, 'pydantic_ai');

      expect(result).not.toBeNull();
      expect(result).toHaveLength(4);

      expect(result![0].role).toBe('user');

      expect(result![1].role).toBe('assistant');
      expect(result![1].tool_calls).toBeDefined();
      expect(result![1].tool_calls).toHaveLength(1);

      expect(result![2].role).toBe('tool');
      expect(result![2].tool_call_id).toBe('call_abc123');

      expect(result![3].role).toBe('assistant');
      expect(result![3].content).toBe('The weather in San Francisco is 18°C and sunny.');
    });

    it('should handle inputs with message_history field', () => {
      const inputWithMessageHistory = {
        user_prompt: 'I am doing well thank you.',
        message_history: [MOCK_PYDANTIC_AI_REQUEST, MOCK_PYDANTIC_AI_RESPONSE],
      };

      const result = normalizeConversation(inputWithMessageHistory, 'pydantic_ai');

      expect(result).not.toBeNull();
      expect(result).toHaveLength(3); // system + user + assistant from message_history

      expect(result![0].role).toBe('system');
      expect(result![1].role).toBe('user');
      expect(result![2].role).toBe('assistant');
    });

    it('should handle response with thinking parts', () => {
      const responseWithThinking = {
        kind: 'response',
        parts: [
          {
            part_kind: 'thinking',
            content: 'Let me think about this...',
            id: null,
            signature: null,
            provider_name: 'anthropic',
          },
          {
            part_kind: 'text',
            content: 'Here is my answer.',
            id: null,
          },
        ],
        usage: {
          input_tokens: 10,
          output_tokens: 20,
        },
        model_name: 'claude-3-7-sonnet-latest',
      };

      const result = normalizeConversation(responseWithThinking, 'pydantic_ai');

      expect(result).not.toBeNull();
      expect(result).toHaveLength(1);

      expect(result![0]).toMatchObject({
        role: 'assistant',
        content: expect.stringContaining('[Thinking]'),
      });
      expect(result![0].content).toContain('Here is my answer.');
    });

    it('should handle builtin tool calls', () => {
      const responseWithBuiltinTool = {
        kind: 'response',
        parts: [
          {
            part_kind: 'builtin-tool-call',
            tool_name: 'web_search',
            args: { query: 'latest news' },
            tool_call_id: 'builtin_123',
            id: null,
            provider_name: 'anthropic',
          },
        ],
        usage: {
          input_tokens: 10,
          output_tokens: 5,
        },
      };

      const result = normalizeConversation(responseWithBuiltinTool, 'pydantic_ai');

      expect(result).not.toBeNull();
      expect(result).toHaveLength(1);

      expect(result![0]).toMatchObject({
        role: 'assistant',
        tool_calls: [
          {
            id: 'builtin_123',
            function: {
              name: 'web_search',
              arguments: expect.stringContaining('latest news'),
            },
          },
        ],
      });
    });

    it('should return null for non-PydanticAI messages', () => {
      const invalidInput = {
        some_field: 'some_value',
      };

      const result = normalizeConversation(invalidInput, 'pydantic_ai');
      expect(result).toBeNull();
    });

    it('should return null for empty array', () => {
      const result = normalizeConversation([], 'pydantic_ai');
      expect(result).toBeNull();
    });

    it('should handle tool call with string arguments', () => {
      const responseWithStringArgs = {
        kind: 'response',
        parts: [
          {
            part_kind: 'tool-call',
            tool_name: 'search',
            args: '{"query": "test"}',
            tool_call_id: 'call_str',
            id: null,
          },
        ],
        usage: {
          input_tokens: 10,
          output_tokens: 5,
        },
      };

      const result = normalizeConversation(responseWithStringArgs, 'pydantic_ai');

      expect(result).not.toBeNull();
      expect(result).toHaveLength(1);

      expect(result![0].role).toBe('assistant');
      expect(result![0].tool_calls).toBeDefined();
      expect(result![0].tool_calls).toHaveLength(1);
      expect(result![0].tool_calls![0].id).toBe('call_str');
      expect(result![0].tool_calls![0].function.name).toBe('search');
      const args = JSON.parse(result![0].tool_calls![0].function.arguments);
      expect(args).toEqual({ query: 'test' });
    });

    it('should handle tool call with null arguments', () => {
      const responseWithNullArgs = {
        kind: 'response',
        parts: [
          {
            part_kind: 'tool-call',
            tool_name: 'no_args_tool',
            args: null,
            tool_call_id: 'call_null',
            id: null,
          },
        ],
        usage: {
          input_tokens: 10,
          output_tokens: 5,
        },
      };

      const result = normalizeConversation(responseWithNullArgs, 'pydantic_ai');

      expect(result).not.toBeNull();
      expect(result).toHaveLength(1);

      expect(result![0]).toMatchObject({
        role: 'assistant',
        tool_calls: [
          {
            id: 'call_null',
            function: {
              name: 'no_args_tool',
              arguments: '{}',
            },
          },
        ],
      });
    });

    it('should handle outputs with _new_messages_serialized field', () => {
      const outputWithSerializedMessages = {
        data: 'Paris',
        output: 'Paris',
        _new_messages_serialized: [MOCK_PYDANTIC_AI_REQUEST, MOCK_PYDANTIC_AI_RESPONSE],
      };

      const result = normalizeConversation(outputWithSerializedMessages, 'pydantic_ai');

      expect(result).not.toBeNull();
      expect(result).toHaveLength(3); // system + user + assistant

      expect(result![0].role).toBe('system');
      expect(result![1].role).toBe('user');
      expect(result![2].role).toBe('assistant');
    });

    it('should handle outputs with _new_messages_serialized for first call without history', () => {
      const firstCallOutput = {
        data: 'I am doing well, thank you!',
        output: 'I am doing well, thank you!',
        _new_messages_serialized: [
          {
            kind: 'request',
            parts: [
              {
                part_kind: 'system-prompt',
                content: 'You are a helpful assistant.',
                timestamp: '2025-11-10T03:25:10.268933',
                dynamic_ref: null,
              },
              {
                part_kind: 'user-prompt',
                content: 'Hello, how are you?',
                timestamp: '2025-11-10T03:25:10.268955',
              },
            ],
          },
          {
            kind: 'response',
            parts: [
              {
                part_kind: 'text',
                content: 'I am doing well, thank you!',
                id: null,
              },
            ],
            usage: {
              input_tokens: 20,
              output_tokens: 10,
            },
            model_name: 'claude-3-7-sonnet-latest',
            timestamp: '2025-11-10T03:25:10.268957',
          },
        ],
      };

      const result = normalizeConversation(firstCallOutput, 'pydantic_ai');

      expect(result).not.toBeNull();
      expect(result).toHaveLength(3); // system + user + assistant

      expect(result![0]).toMatchObject({
        role: 'system',
        content: 'You are a helpful assistant.',
      });

      expect(result![1]).toMatchObject({
        role: 'user',
        content: 'Hello, how are you?',
      });

      expect(result![2]).toMatchObject({
        role: 'assistant',
        content: 'I am doing well, thank you!',
      });
    });
  });
});
