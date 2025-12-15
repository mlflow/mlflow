import { describe, expect, it } from '@jest/globals';

import { normalizeConversation } from '../ModelTraceExplorer.utils';
import {
  normalizeVoltAgentChatInput,
  normalizeVoltAgentChatOutput,
  synthesizeVoltAgentChatMessages,
} from './voltagent';
import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { ModelSpanType } from '../ModelTrace.types';

const MOCK_VOLTAGENT_SIMPLE_MESSAGES = [
  { role: 'user', content: 'Hello, how are you?' },
  { role: 'assistant', content: 'I am doing well, thank you!' },
];

const MOCK_VOLTAGENT_WITH_SYSTEM = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'What can you help me with?' },
  { role: 'assistant', content: 'I can help you with many things!' },
];

const MOCK_VOLTAGENT_TEXT_CONTENT_PARTS = [
  {
    role: 'user',
    content: [
      { type: 'text', text: 'First part' },
      { type: 'text', text: 'Second part' },
    ],
  },
];

const MOCK_VOLTAGENT_TOOL_CALLS = [
  { role: 'user', content: 'What is the weather in San Francisco?' },
  {
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: 'call_123',
        toolName: 'weather',
        input: { location: 'San Francisco' },
      },
    ],
  },
];

const MOCK_VOLTAGENT_TOOL_RESULTS = [
  { role: 'user', content: 'Get the weather' },
  {
    role: 'tool',
    content: [
      {
        type: 'tool-result',
        toolCallId: 'call_456',
        toolName: 'weather',
        output: { type: 'json', value: { temperature: 72, conditions: 'sunny' } },
      },
    ],
  },
];

const MOCK_VOLTAGENT_MIXED_CONTENT = [
  { role: 'user', content: 'Tell me the weather and explain it' },
  {
    role: 'assistant',
    content: [
      { type: 'text', text: 'Let me check the weather for you.' },
      {
        type: 'tool-call',
        toolCallId: 'call_789',
        toolName: 'weather',
        input: { location: 'New York' },
      },
    ],
  },
];

const MOCK_VOLTAGENT_FULL_TOOL_CYCLE = [
  { role: 'user', content: 'What is the weather in London?' },
  {
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: 'call_abc',
        toolName: 'get_weather',
        input: { city: 'London' },
      },
    ],
  },
  {
    role: 'tool',
    content: [
      {
        type: 'tool-result',
        toolCallId: 'call_abc',
        toolName: 'get_weather',
        output: { temperature: 15, conditions: 'cloudy' },
      },
    ],
  },
  {
    role: 'assistant',
    content: 'The weather in London is currently 15°C and cloudy.',
  },
];

describe('VoltAgent Chat Normalization', () => {
  describe('normalizeVoltAgentChatInput', () => {
    it('should normalize simple text messages', () => {
      const result = normalizeVoltAgentChatInput(MOCK_VOLTAGENT_SIMPLE_MESSAGES);

      expect(result).not.toBeNull();
      expect(result).toHaveLength(2);
      expect(result![0]).toEqual(expect.objectContaining({ role: 'user', content: 'Hello, how are you?' }));
      expect(result![1]).toEqual(
        expect.objectContaining({ role: 'assistant', content: 'I am doing well, thank you!' }),
      );
    });

    it('should normalize messages with system prompt', () => {
      const result = normalizeVoltAgentChatInput(MOCK_VOLTAGENT_WITH_SYSTEM);

      expect(result).not.toBeNull();
      expect(result).toHaveLength(3);
      expect(result![0]).toEqual(expect.objectContaining({ role: 'system', content: 'You are a helpful assistant.' }));
    });

    it('should normalize messages with text content parts', () => {
      const result = normalizeVoltAgentChatInput(MOCK_VOLTAGENT_TEXT_CONTENT_PARTS);

      expect(result).not.toBeNull();
      expect(result).toHaveLength(1);
      expect(result![0]).toEqual(
        expect.objectContaining({ role: 'user', content: expect.stringContaining('First part') }),
      );
    });

    it('should normalize messages with tool calls', () => {
      const result = normalizeVoltAgentChatInput(MOCK_VOLTAGENT_TOOL_CALLS);

      expect(result).not.toBeNull();
      expect(result).toHaveLength(2);

      const assistantMessage = result![1];
      expect(assistantMessage.role).toBe('assistant');
      expect(assistantMessage.tool_calls).toBeDefined();
      expect(assistantMessage.tool_calls).toHaveLength(1);
      expect(assistantMessage.tool_calls![0].id).toBe('call_123');
      expect(assistantMessage.tool_calls![0].function.name).toBe('weather');
    });

    it('should normalize messages with tool results', () => {
      const result = normalizeVoltAgentChatInput(MOCK_VOLTAGENT_TOOL_RESULTS);

      expect(result).not.toBeNull();
      expect(result).toHaveLength(2);

      const toolMessage = result![1];
      expect(toolMessage.role).toBe('tool');
      expect(toolMessage.tool_call_id).toBe('call_456');
      expect(toolMessage.content).toContain('temperature');
    });

    it('should normalize messages with mixed content', () => {
      const result = normalizeVoltAgentChatInput(MOCK_VOLTAGENT_MIXED_CONTENT);

      expect(result).not.toBeNull();
      expect(result).toHaveLength(2);

      const assistantMessage = result![1];
      expect(assistantMessage.role).toBe('assistant');
      expect(assistantMessage.content).toContain('Let me check the weather');
      expect(assistantMessage.tool_calls).toBeDefined();
      expect(assistantMessage.tool_calls).toHaveLength(1);
    });

    it('should normalize full tool usage cycle', () => {
      const result = normalizeVoltAgentChatInput(MOCK_VOLTAGENT_FULL_TOOL_CYCLE);

      expect(result).not.toBeNull();
      expect(result).toHaveLength(4);
      expect(result![0].role).toBe('user');
      expect(result![1].role).toBe('assistant');
      expect(result![1].tool_calls).toBeDefined();
      expect(result![2].role).toBe('tool');
      expect(result![3].role).toBe('assistant');
      expect(result![3].content).toContain('15°C');
    });

    it('should handle JSON string input', () => {
      const jsonString = JSON.stringify(MOCK_VOLTAGENT_SIMPLE_MESSAGES);
      const result = normalizeVoltAgentChatInput(jsonString);

      expect(result).not.toBeNull();
      expect(result).toHaveLength(2);
    });

    it('should return null for invalid input', () => {
      expect(normalizeVoltAgentChatInput({ invalid: 'data' })).toBeNull();
      expect(normalizeVoltAgentChatInput(null)).toBeNull();
      expect(normalizeVoltAgentChatInput(undefined)).toBeNull();
      expect(normalizeVoltAgentChatInput([])).toBeNull();
      expect(normalizeVoltAgentChatInput('invalid json')).toBeNull();
    });

    it('should return null for messages with invalid roles', () => {
      const invalidRoles = [{ role: 'invalid_role', content: 'Hello' }];
      expect(normalizeVoltAgentChatInput(invalidRoles)).toBeNull();
    });
  });

  describe('normalizeVoltAgentChatOutput', () => {
    it('should normalize string output to assistant message', () => {
      const result = normalizeVoltAgentChatOutput('This is the AI response');

      expect(result).not.toBeNull();
      expect(result).toHaveLength(1);
      expect(result![0]).toEqual(expect.objectContaining({ role: 'assistant', content: 'This is the AI response' }));
    });

    it('should return null for empty string', () => {
      expect(normalizeVoltAgentChatOutput('')).toBeNull();
      expect(normalizeVoltAgentChatOutput('   ')).toBeNull();
    });

    it('should return null for non-string outputs', () => {
      expect(normalizeVoltAgentChatOutput({ text: 'hello' })).toBeNull();
      expect(normalizeVoltAgentChatOutput(['hello'])).toBeNull();
      expect(normalizeVoltAgentChatOutput(null)).toBeNull();
      expect(normalizeVoltAgentChatOutput(undefined)).toBeNull();
    });
  });

  describe('synthesizeVoltAgentChatMessages', () => {
    const createMockToolSpan = (id: string, name: string, inputs: any, outputs: any): ModelTraceSpanNode => ({
      key: 'span_' + id,
      title: name,
      icon: null,
      start: 0,
      end: 100,
      type: ModelSpanType.TOOL,
      traceId: 'trace_123',
      inputs: inputs,
      outputs: outputs,
      attributes: {
        'span.type': 'tool',
        'tool.call.id': id,
        'tool.name': name,
      },
      events: [],
      children: [],
      assessments: [],
    });

    it('should synthesize messages with tool spans', () => {
      const inputs = [{ role: 'user', content: 'Get the weather' }];
      const outputs = 'The weather is sunny.';
      const children = [createMockToolSpan('tool_1', 'get_weather', { location: 'NYC' }, { temp: 72 })];

      const result = synthesizeVoltAgentChatMessages(inputs, outputs, children);

      expect(result).not.toBeNull();
      expect(result!.length).toBeGreaterThan(1);

      expect(result![0].role).toBe('user');

      const assistantWithToolCalls = result!.find((m) => m.role === 'assistant' && m.tool_calls);
      expect(assistantWithToolCalls).toBeDefined();
      expect(assistantWithToolCalls!.tool_calls![0].id).toBe('tool_1');

      const toolResult = result!.find((m) => m.role === 'tool');
      expect(toolResult).toBeDefined();

      const lastMessage = result![result!.length - 1];
      expect(lastMessage.role).toBe('assistant');
      expect(lastMessage.content).toBe('The weather is sunny.');
    });

    it('should return null when no valid input messages', () => {
      const result = synthesizeVoltAgentChatMessages({ invalid: 'data' }, 'output', []);
      expect(result).toBeNull();
    });

    it('should handle multiple tool spans', () => {
      const inputs = [{ role: 'user', content: 'Get weather for multiple cities' }];
      const outputs = 'Here are the weather reports.';
      const children = [
        createMockToolSpan('tool_1', 'get_weather', { location: 'NYC' }, { temp: 72 }),
        createMockToolSpan('tool_2', 'get_weather', { location: 'LA' }, { temp: 85 }),
      ];

      const result = synthesizeVoltAgentChatMessages(inputs, outputs, children);

      expect(result).not.toBeNull();

      const assistantWithToolCalls = result!.find((m) => m.role === 'assistant' && m.tool_calls);
      expect(assistantWithToolCalls!.tool_calls).toHaveLength(2);

      const toolMessages = result!.filter((m) => m.role === 'tool');
      expect(toolMessages).toHaveLength(2);
    });
  });

  describe('normalizeConversation with voltagent format', () => {
    it('should handle voltagent message format', () => {
      const result = normalizeConversation(MOCK_VOLTAGENT_SIMPLE_MESSAGES, 'voltagent');

      expect(result).not.toBeNull();
      expect(result).toHaveLength(2);
      expect(result![0]).toEqual(expect.objectContaining({ role: 'user' }));
      expect(result![1]).toEqual(expect.objectContaining({ role: 'assistant' }));
    });

    it('should handle JSON string input with voltagent format', () => {
      const jsonString = JSON.stringify(MOCK_VOLTAGENT_WITH_SYSTEM);
      const result = normalizeConversation(jsonString, 'voltagent');

      expect(result).not.toBeNull();
      expect(result).toHaveLength(3);
      expect(result![0]).toEqual(expect.objectContaining({ role: 'system' }));
    });

    it('should handle tool call messages', () => {
      const result = normalizeConversation(MOCK_VOLTAGENT_TOOL_CALLS, 'voltagent');

      expect(result).not.toBeNull();
      expect(result![1].tool_calls).toBeDefined();
      expect(result![1].tool_calls![0].function.name).toBe('weather');
    });

    it('should return null for invalid voltagent data', () => {
      expect(normalizeConversation({ invalid: 'data' }, 'voltagent')).toBeNull();
    });

    it('should return null for null input', () => {
      expect(normalizeConversation(null, 'voltagent')).toBeNull();
    });

    it('should return null for undefined input', () => {
      expect(normalizeConversation(undefined, 'voltagent')).toBeNull();
    });

    it('should normalize string output with voltagent format', () => {
      const result = normalizeConversation('AI response text', 'voltagent');

      expect(result).not.toBeNull();
      expect(result).toHaveLength(1);
      expect(result![0].role).toBe('assistant');
      expect(result![0].content).toBe('AI response text');
    });
  });

  describe('Edge cases', () => {
    it('should handle messages with empty content array', () => {
      const messagesWithEmptyContent = [{ role: 'user', content: [] }];
      const result = normalizeVoltAgentChatInput(messagesWithEmptyContent);

      if (result !== null) {
        expect(result).toHaveLength(1);
        expect(result[0].role).toBe('user');
      }
    });

    it('should handle nested JSON value in tool result output', () => {
      const messagesWithNestedOutput = [
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'call_nested',
              toolName: 'complex_tool',
              output: {
                type: 'json',
                value: {
                  nested: {
                    deeply: {
                      value: 'found',
                    },
                  },
                },
              },
            },
          ],
        },
      ];

      const result = normalizeVoltAgentChatInput(messagesWithNestedOutput);
      expect(result).not.toBeNull();
      expect(result![0].content).toContain('found');
    });

    it('should handle tool call with providerExecuted flag', () => {
      const messagesWithProviderExecuted = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'call_provider',
              toolName: 'builtin_tool',
              input: { param: 'value' },
              providerExecuted: true,
            },
          ],
        },
      ];

      const result = normalizeVoltAgentChatInput(messagesWithProviderExecuted);
      expect(result).not.toBeNull();
      expect(result![0].tool_calls).toBeDefined();
      expect(result![0].tool_calls![0].id).toBe('call_provider');
    });
  });
});
