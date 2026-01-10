import { describe, expect, it } from '@jest/globals';

import { normalizeConversation } from '../ModelTraceExplorer.utils';
import { normalizeAgnoChatInput, normalizeAgnoChatOutput, synthesizeAgnoChatMessages } from './agno';
import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { ModelSpanType } from '../ModelTrace.types';

// Mock Agno LLM span input: { messages: [...] }
const MOCK_AGNO_LLM_INPUT = {
  messages: [
    {
      id: '1',
      role: 'system',
      content: 'You are a helpful assistant.',
      from_history: false,
      created_at: 1767285202,
    },
    {
      id: '2',
      role: 'user',
      content: 'What is the stock price of Apple?',
      from_history: false,
      created_at: 1767285202,
    },
  ],
};

// Mock Agno LLM input with tool calls from assistant
const MOCK_AGNO_LLM_INPUT_WITH_TOOL_CALLS = {
  messages: [
    { role: 'system', content: 'Be helpful.' },
    { role: 'user', content: 'Get the weather' },
    {
      role: 'assistant',
      tool_calls: [
        {
          id: 'call_123',
          type: 'function',
          function: { name: 'get_weather', arguments: '{"location": "NYC"}' },
        },
      ],
    },
  ],
};

// Mock Agno combined tool message format
const MOCK_AGNO_LLM_INPUT_WITH_TOOL_RESULTS = {
  messages: [
    { role: 'system', content: 'Be helpful.' },
    { role: 'user', content: 'Get the weather for multiple cities' },
    {
      role: 'assistant',
      tool_calls: [
        { id: 'call_1', type: 'function', function: { name: 'get_weather', arguments: '{"city": "NYC"}' } },
        { id: 'call_2', type: 'function', function: { name: 'get_weather', arguments: '{"city": "LA"}' } },
      ],
    },
    {
      role: 'tool',
      content: ['72°F sunny', '85°F clear'],
      tool_name: 'get_weather, get_weather',
      tool_calls: [
        { tool_call_id: 'call_1', tool_name: 'get_weather', content: '72°F sunny' },
        { tool_call_id: 'call_2', tool_name: 'get_weather', content: '85°F clear' },
      ],
    },
  ],
};

// Mock Agno LLM output: JSON string of messages
const MOCK_AGNO_LLM_OUTPUT_WITH_TOOL_CALLS =
  '[{"role": "assistant", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{\\"location\\": \\"NYC\\"}"}}]}]';

const MOCK_AGNO_LLM_OUTPUT_WITH_CONTENT = '[{"role": "assistant", "content": "The weather in NYC is 72°F and sunny."}]';

// Mock Agno AGENT span input/output (plain strings)
const MOCK_AGNO_AGENT_INPUT = 'What is the stock price of Apple?';
const MOCK_AGNO_AGENT_OUTPUT = '| Company | Symbol | Price |\n|---------|--------|-------|\n| Apple | AAPL | 271.86 |';

describe('Agno Chat Normalization', () => {
  describe('normalizeAgnoChatInput', () => {
    it('should normalize LLM span input with messages array', () => {
      const result = normalizeAgnoChatInput(MOCK_AGNO_LLM_INPUT);

      expect(result).not.toBeNull();
      expect(result).toHaveLength(2);
      expect(result![0]).toEqual(expect.objectContaining({ role: 'system', content: 'You are a helpful assistant.' }));
      expect(result![1]).toEqual(
        expect.objectContaining({ role: 'user', content: 'What is the stock price of Apple?' }),
      );
    });

    it('should normalize AGENT span input as user message', () => {
      const result = normalizeAgnoChatInput(MOCK_AGNO_AGENT_INPUT);

      expect(result).not.toBeNull();
      expect(result).toHaveLength(1);
      expect(result![0]).toEqual(
        expect.objectContaining({ role: 'user', content: 'What is the stock price of Apple?' }),
      );
    });

    it('should normalize messages with tool calls', () => {
      const result = normalizeAgnoChatInput(MOCK_AGNO_LLM_INPUT_WITH_TOOL_CALLS);

      expect(result).not.toBeNull();
      expect(result).toHaveLength(3);

      const assistantMessage = result![2];
      expect(assistantMessage.role).toBe('assistant');
      expect(assistantMessage.tool_calls).toBeDefined();
      expect(assistantMessage.tool_calls).toHaveLength(1);
      expect(assistantMessage.tool_calls![0].id).toBe('call_123');
      expect(assistantMessage.tool_calls![0].function.name).toBe('get_weather');
    });

    it('should normalize combined tool message with individual tool results', () => {
      const result = normalizeAgnoChatInput(MOCK_AGNO_LLM_INPUT_WITH_TOOL_RESULTS);

      expect(result).not.toBeNull();

      // Should have: system, user, assistant (with tool_calls), tool (NYC), tool (LA)
      const toolMessages = result!.filter((m) => m.role === 'tool');
      expect(toolMessages).toHaveLength(2);
      expect(toolMessages[0].content).toBe('72°F sunny');
      expect(toolMessages[0].tool_call_id).toBe('call_1');
      expect(toolMessages[1].content).toBe('85°F clear');
      expect(toolMessages[1].tool_call_id).toBe('call_2');
    });

    it('should return null for invalid input', () => {
      expect(normalizeAgnoChatInput(null)).toBeNull();
      expect(normalizeAgnoChatInput(undefined)).toBeNull();
      expect(normalizeAgnoChatInput({ invalid: 'data' })).toBeNull();
      expect(normalizeAgnoChatInput({ messages: [] })).toBeNull();
    });

    it('should not treat JSON strings as user messages', () => {
      // JSON strings should be handled by normalizeAgnoChatOutput, not input
      const jsonString = '[{"role": "assistant", "content": "Hello"}]';
      const result = normalizeAgnoChatInput(jsonString);
      expect(result).toBeNull();
    });
  });

  describe('normalizeAgnoChatOutput', () => {
    it('should normalize LLM output JSON string with tool calls', () => {
      const result = normalizeAgnoChatOutput(MOCK_AGNO_LLM_OUTPUT_WITH_TOOL_CALLS);

      expect(result).not.toBeNull();
      expect(result).toHaveLength(1);

      const assistantMessage = result![0];
      expect(assistantMessage.role).toBe('assistant');
      expect(assistantMessage.tool_calls).toBeDefined();
      expect(assistantMessage.tool_calls).toHaveLength(1);
      expect(assistantMessage.tool_calls![0].function.name).toBe('get_weather');
    });

    it('should normalize LLM output JSON string with content', () => {
      const result = normalizeAgnoChatOutput(MOCK_AGNO_LLM_OUTPUT_WITH_CONTENT);

      expect(result).not.toBeNull();
      expect(result).toHaveLength(1);
      expect(result![0]).toEqual(
        expect.objectContaining({ role: 'assistant', content: 'The weather in NYC is 72°F and sunny.' }),
      );
    });

    it('should normalize AGENT output as assistant message', () => {
      const result = normalizeAgnoChatOutput(MOCK_AGNO_AGENT_OUTPUT);

      expect(result).not.toBeNull();
      expect(result).toHaveLength(1);
      expect(result![0].role).toBe('assistant');
      expect(result![0].content).toContain('Apple');
      expect(result![0].content).toContain('271.86');
    });

    it('should return null for empty or invalid output', () => {
      expect(normalizeAgnoChatOutput(null)).toBeNull();
      expect(normalizeAgnoChatOutput(undefined)).toBeNull();
      expect(normalizeAgnoChatOutput('')).toBeNull();
      expect(normalizeAgnoChatOutput({ invalid: 'data' })).toBeNull();
    });
  });

  describe('synthesizeAgnoChatMessages', () => {
    const createMockLLMSpan = (outputs: string, inputs?: unknown): ModelTraceSpanNode => ({
      key: 'llm_span',
      title: 'OpenAIResponses.invoke',
      icon: null,
      start: 0,
      end: 100,
      type: ModelSpanType.LLM,
      traceId: 'trace_123',
      inputs: inputs ?? {},
      outputs: outputs,
      attributes: { 'mlflow.spanType': 'LLM' },
      events: [],
      children: [],
      assessments: [],
    });

    const MOCK_LLM_INPUTS_WITH_SYSTEM = {
      messages: [
        { role: 'system', content: 'You are a helpful financial assistant.' },
        { role: 'user', content: 'What is the stock price of Apple?' },
      ],
    };

    const createMockToolSpan = (name: string, outputs: unknown): ModelTraceSpanNode => ({
      key: 'tool_span',
      title: name,
      icon: null,
      start: 0,
      end: 100,
      type: ModelSpanType.TOOL,
      traceId: 'trace_123',
      inputs: {},
      outputs: outputs,
      attributes: { 'mlflow.spanType': 'TOOL', 'tool.name': name },
      events: [],
      children: [],
      assessments: [],
    });

    it('should synthesize messages from AGENT inputs, LLM outputs, and TOOL spans', () => {
      const children = [
        createMockLLMSpan(
          '[{"role": "assistant", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "get_price", "arguments": "{}"}}]}]',
        ),
        createMockToolSpan('get_price', '271.86'),
        createMockLLMSpan('[{"role": "assistant", "content": "The price is $271.86"}]'),
      ];

      const result = synthesizeAgnoChatMessages(MOCK_AGNO_AGENT_INPUT, MOCK_AGNO_AGENT_OUTPUT, children);

      expect(result).not.toBeNull();
      expect(result!.length).toBeGreaterThan(1);

      // First message should be user input
      expect(result![0].role).toBe('user');
      expect(result![0].content).toBe('What is the stock price of Apple?');

      // Should have assistant messages from LLM spans
      const assistantMessages = result!.filter((m) => m.role === 'assistant');
      expect(assistantMessages.length).toBeGreaterThanOrEqual(1);

      // Should have tool result
      const toolMessages = result!.filter((m) => m.role === 'tool');
      expect(toolMessages).toHaveLength(1);
      expect(toolMessages[0].content).toContain('271.86');
    });

    it('should maintain chronological order: system -> user -> LLM (tool call) -> TOOL (result) -> LLM (final answer)', () => {
      const children = [
        createMockLLMSpan(
          '[{"role": "assistant", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "get_price", "arguments": "{}"}}]}]',
          MOCK_LLM_INPUTS_WITH_SYSTEM,
        ),
        createMockToolSpan('get_price', '271.86'),
        createMockLLMSpan('[{"role": "assistant", "content": "The price is $271.86"}]'),
      ];

      const result = synthesizeAgnoChatMessages(MOCK_AGNO_AGENT_INPUT, MOCK_AGNO_AGENT_OUTPUT, children);

      expect(result).not.toBeNull();
      expect(result!.length).toBe(5);

      // Verify exact order:
      // 1. System prompt (extracted from first LLM span inputs)
      expect(result![0].role).toBe('system');
      expect(result![0].content).toContain('financial assistant');
      // 2. User input
      expect(result![1].role).toBe('user');
      // 3. Assistant with tool call (from first LLM span)
      expect(result![2].role).toBe('assistant');
      expect(result![2].tool_calls).toBeDefined();
      // 4. Tool result (from TOOL span)
      expect(result![3].role).toBe('tool');
      expect(result![3].content).toContain('271.86');
      // 5. Assistant with final answer (from second LLM span)
      expect(result![4].role).toBe('assistant');
      expect(result![4].content).toContain('$271.86');
    });

    it('should extract system prompt from first LLM span inputs', () => {
      const children = [createMockLLMSpan('[{"role": "assistant", "content": "Hello!"}]', MOCK_LLM_INPUTS_WITH_SYSTEM)];

      const result = synthesizeAgnoChatMessages(MOCK_AGNO_AGENT_INPUT, MOCK_AGNO_AGENT_OUTPUT, children);

      expect(result).not.toBeNull();
      expect(result!.length).toBe(3);
      expect(result![0].role).toBe('system');
      expect(result![0].content).toBe('You are a helpful financial assistant.');
      expect(result![1].role).toBe('user');
      expect(result![2].role).toBe('assistant');
    });

    it('should return null when no valid input', () => {
      const result = synthesizeAgnoChatMessages({ invalid: 'data' }, 'output', []);
      expect(result).toBeNull();
    });

    it('should return only user input when no children', () => {
      const result = synthesizeAgnoChatMessages(MOCK_AGNO_AGENT_INPUT, MOCK_AGNO_AGENT_OUTPUT, []);

      expect(result).not.toBeNull();
      expect(result!.length).toBe(1);
      expect(result![0].role).toBe('user');
    });
  });

  describe('normalizeConversation with agno format', () => {
    it('should handle agno message format for LLM inputs', () => {
      const result = normalizeConversation(MOCK_AGNO_LLM_INPUT, 'agno');

      expect(result).not.toBeNull();
      expect(result).toHaveLength(2);
      expect(result![0]).toEqual(expect.objectContaining({ role: 'system' }));
      expect(result![1]).toEqual(expect.objectContaining({ role: 'user' }));
    });

    it('should handle agno message format for LLM outputs', () => {
      const result = normalizeConversation(MOCK_AGNO_LLM_OUTPUT_WITH_CONTENT, 'agno');

      expect(result).not.toBeNull();
      expect(result).toHaveLength(1);
      expect(result![0]).toEqual(expect.objectContaining({ role: 'assistant' }));
    });

    it('should handle agno format with tool calls', () => {
      const result = normalizeConversation(MOCK_AGNO_LLM_INPUT_WITH_TOOL_CALLS, 'agno');

      expect(result).not.toBeNull();
      expect(result![2].tool_calls).toBeDefined();
      expect(result![2].tool_calls![0].function.name).toBe('get_weather');
    });

    it('should return null for invalid agno data', () => {
      expect(normalizeConversation({ invalid: 'data' }, 'agno')).toBeNull();
    });

    it('should return null for null input', () => {
      expect(normalizeConversation(null, 'agno')).toBeNull();
    });

    it('should return null for undefined input', () => {
      expect(normalizeConversation(undefined, 'agno')).toBeNull();
    });

    it('should normalize LLM JSON output with agno format', () => {
      // JSON array outputs are correctly parsed as assistant messages
      const result = normalizeConversation(MOCK_AGNO_LLM_OUTPUT_WITH_CONTENT, 'agno');

      expect(result).not.toBeNull();
      expect(result).toHaveLength(1);
      expect(result![0].role).toBe('assistant');
      expect(result![0].content).toContain('weather');
    });

    it('should treat plain string as user message in normalizeConversation', () => {
      // Plain strings are ambiguous - normalizeConversation tries input first
      // For actual output handling, use normalizeAgnoChatOutput directly
      const result = normalizeConversation(MOCK_AGNO_AGENT_OUTPUT, 'agno');

      expect(result).not.toBeNull();
      expect(result).toHaveLength(1);
      // Plain strings are treated as user messages by normalizeAgnoChatInput
      expect(result![0].role).toBe('user');
    });
  });

  describe('Edge cases', () => {
    it('should handle messages with empty content', () => {
      const messagesWithEmptyContent = { messages: [{ role: 'user', content: '' }] };
      const result = normalizeAgnoChatInput(messagesWithEmptyContent);

      expect(result).not.toBeNull();
      expect(result).toHaveLength(1);
      expect(result![0].role).toBe('user');
    });

    it('should handle single tool result (not combined format)', () => {
      const singleToolResult = {
        messages: [{ role: 'tool', content: 'Tool result', tool_call_id: 'call_1', tool_name: 'my_tool' }],
      };
      const result = normalizeAgnoChatInput(singleToolResult);

      expect(result).not.toBeNull();
      expect(result![0].role).toBe('tool');
    });

    it('should handle content as array (fallback for tool messages)', () => {
      const toolWithContentArray = {
        messages: [{ role: 'tool', content: ['Result 1', 'Result 2'] }],
      };
      const result = normalizeAgnoChatInput(toolWithContentArray);

      expect(result).not.toBeNull();
      expect(result!.length).toBeGreaterThanOrEqual(1);
    });

    it('should handle malformed JSON output gracefully', () => {
      const malformedJson = '[{invalid json}]';
      const result = normalizeAgnoChatOutput(malformedJson);

      // Should fall back to treating as plain string
      expect(result).not.toBeNull();
      expect(result![0].role).toBe('assistant');
    });
  });
});
