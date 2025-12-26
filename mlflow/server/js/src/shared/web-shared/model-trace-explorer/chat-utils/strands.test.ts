import { describe, it, expect } from '@jest/globals';

import { normalizeStrandsChatInput, normalizeStrandsChatOutput, synthesizeStrandsChatMessages } from './strands';
import { ModelSpanType, type ModelTraceSpanNode } from '../ModelTrace.types';

const MOCK_STRANDS_USER_INPUT = [
  {
    role: 'user',
    content: [{ text: 'What is 2+2' }],
  },
];

const MOCK_STRANDS_TOOL_CALL_OUTPUT = [
  {
    toolUse: {
      toolUseId: 'call_PHijhFE1QjCO6ZuW1F9yYAXo',
      name: 'calculator',
      input: { expression: '2+2' },
    },
  },
];

const MOCK_STRANDS_TEXT_OUTPUT = [{ text: 'The result of (2 + 2) is 4.' }];

const MOCK_STRANDS_PLAIN_STRING_OUTPUT = 'The result of (2 + 2) is 4.';

const MOCK_STRANDS_REASONING_OUTPUT = [
  {
    reasoningContent: {
      reasoningText: {
        text: 'Let me think about this step by step. 2+2=4.',
        signature: 'some_signature_here',
      },
    },
  },
  {
    toolUse: {
      toolUseId: 'call_PHijhFE1QjCO6ZuW1F9yYAXo',
      name: 'calculator',
      input: { expression: '2+2' },
    },
  },
];

const MOCK_STRANDS_REASONING_WITH_TEXT_OUTPUT = [
  {
    reasoningContent: {
      reasoningText: {
        text: 'Thinking about the math problem...',
        signature: 'sig123',
      },
    },
  },
  { text: 'The answer is 4.' },
];

const MOCK_TOOL_SPAN: Partial<ModelTraceSpanNode> = {
  title: 'execute_tool calculator',
  type: ModelSpanType.TOOL,
  outputs: [{ text: 'Result: 4' }],
  attributes: {
    'gen_ai.tool.call.id': 'call_PHijhFE1QjCO6ZuW1F9yYAXo',
  },
};

const MOCK_CHAT_SPAN_WITH_TOOL: Partial<ModelTraceSpanNode> = {
  title: 'chat',
  type: ModelSpanType.CHAT_MODEL,
  inputs: MOCK_STRANDS_USER_INPUT,
  outputs: MOCK_STRANDS_TOOL_CALL_OUTPUT,
};

const MOCK_CHAT_SPAN_WITH_TEXT: Partial<ModelTraceSpanNode> = {
  title: 'chat',
  type: ModelSpanType.CHAT_MODEL,
  inputs: MOCK_STRANDS_USER_INPUT,
  outputs: MOCK_STRANDS_TEXT_OUTPUT,
};

describe('normalizeStrandsChatInput', () => {
  it('should normalize user input with text content', () => {
    const result = normalizeStrandsChatInput(MOCK_STRANDS_USER_INPUT);

    expect(result).toEqual([
      expect.objectContaining({
        role: 'user',
        content: 'What is 2+2',
      }),
    ]);
  });

  it('should handle system messages', () => {
    const input = [{ role: 'system', content: [{ text: 'You are helpful' }] }];
    const result = normalizeStrandsChatInput(input);

    expect(result).toEqual([
      expect.objectContaining({
        role: 'system',
        content: 'You are helpful',
      }),
    ]);
  });

  it('should return null for empty input', () => {
    expect(normalizeStrandsChatInput([])).toBeNull();
    expect(normalizeStrandsChatInput(null)).toBeNull();
    expect(normalizeStrandsChatInput(undefined)).toBeNull();
  });

  it('should return null for invalid input format', () => {
    expect(normalizeStrandsChatInput([{ invalid: 'data' }])).toBeNull();
    expect(normalizeStrandsChatInput('not an array')).toBeNull();
  });
});

describe('normalizeStrandsChatOutput', () => {
  it('should normalize tool call output', () => {
    const result = normalizeStrandsChatOutput(MOCK_STRANDS_TOOL_CALL_OUTPUT);

    expect(result).toEqual([
      expect.objectContaining({
        role: 'assistant',
        tool_calls: [
          expect.objectContaining({
            id: 'call_PHijhFE1QjCO6ZuW1F9yYAXo',
            function: expect.objectContaining({
              name: 'calculator',
              arguments: expect.stringContaining('expression'),
            }),
          }),
        ],
      }),
    ]);
  });

  it('should normalize text output', () => {
    const result = normalizeStrandsChatOutput(MOCK_STRANDS_TEXT_OUTPUT);

    expect(result).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content: 'The result of (2 + 2) is 4.',
      }),
    ]);
  });

  it('should normalize plain string output', () => {
    const result = normalizeStrandsChatOutput(MOCK_STRANDS_PLAIN_STRING_OUTPUT);

    expect(result).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content: 'The result of (2 + 2) is 4.',
      }),
    ]);
  });

  it('should return null for empty output', () => {
    expect(normalizeStrandsChatOutput([])).toBeNull();
    expect(normalizeStrandsChatOutput(null)).toBeNull();
    expect(normalizeStrandsChatOutput('')).toBeNull();
  });

  it('should normalize output with reasoning content and tool calls', () => {
    const result = normalizeStrandsChatOutput(MOCK_STRANDS_REASONING_OUTPUT);

    expect(result).toEqual([
      expect.objectContaining({
        role: 'assistant',
        reasoning: 'Let me think about this step by step. 2+2=4.',
        tool_calls: [
          expect.objectContaining({
            id: 'call_PHijhFE1QjCO6ZuW1F9yYAXo',
            function: expect.objectContaining({
              name: 'calculator',
            }),
          }),
        ],
      }),
    ]);
  });

  it('should normalize output with reasoning content and text', () => {
    const result = normalizeStrandsChatOutput(MOCK_STRANDS_REASONING_WITH_TEXT_OUTPUT);

    expect(result).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content: 'The answer is 4.',
        reasoning: 'Thinking about the math problem...',
      }),
    ]);
  });

  it('should handle reasoning content only (no text or tool calls)', () => {
    const reasoningOnly = [
      {
        reasoningContent: {
          reasoningText: {
            text: 'Just thinking...',
            signature: 'sig',
          },
        },
      },
    ];
    const result = normalizeStrandsChatOutput(reasoningOnly);

    expect(result).toEqual([
      expect.objectContaining({
        role: 'assistant',
        reasoning: 'Just thinking...',
      }),
    ]);
  });
});

describe('synthesizeStrandsChatMessages', () => {
  it('should synthesize complete conversation with tool calls', () => {
    const children = [
      {
        ...MOCK_CHAT_SPAN_WITH_TOOL,
        children: [MOCK_TOOL_SPAN as ModelTraceSpanNode],
      } as ModelTraceSpanNode,
    ];

    const result = synthesizeStrandsChatMessages(
      MOCK_STRANDS_USER_INPUT,
      MOCK_STRANDS_PLAIN_STRING_OUTPUT,
      children,
    );

    expect(result).toHaveLength(4);

    // User message
    expect(result?.[0]).toEqual(
      expect.objectContaining({
        role: 'user',
        content: 'What is 2+2',
      }),
    );

    // Assistant with tool calls
    expect(result?.[1]).toEqual(
      expect.objectContaining({
        role: 'assistant',
        tool_calls: expect.arrayContaining([
          expect.objectContaining({
            id: 'call_PHijhFE1QjCO6ZuW1F9yYAXo',
            function: expect.objectContaining({
              name: 'calculator',
            }),
          }),
        ]),
      }),
    );

    // Tool result
    expect(result?.[2]).toEqual(
      expect.objectContaining({
        role: 'tool',
        content: 'Result: 4',
        tool_call_id: 'call_PHijhFE1QjCO6ZuW1F9yYAXo',
      }),
    );

    // Final assistant response
    expect(result?.[3]).toEqual(
      expect.objectContaining({
        role: 'assistant',
        content: 'The result of (2 + 2) is 4.',
      }),
    );
  });

  it('should handle simple response without tool calls', () => {
    const children = [MOCK_CHAT_SPAN_WITH_TEXT as ModelTraceSpanNode];

    const result = synthesizeStrandsChatMessages(MOCK_STRANDS_USER_INPUT, 'Simple answer', children);

    expect(result).toHaveLength(2);
    expect(result?.[0]).toEqual(expect.objectContaining({ role: 'user' }));
    expect(result?.[1]).toEqual(expect.objectContaining({ role: 'assistant', content: 'Simple answer' }));
  });

  it('should return null for empty inputs and children', () => {
    const result = synthesizeStrandsChatMessages([], null, []);
    expect(result).toBeNull();
  });

  it('should find tool spans nested in children', () => {
    // Simulate the actual trace structure with nested spans
    const eventLoopSpan: Partial<ModelTraceSpanNode> = {
      title: 'execute_event_loop_cycle',
      type: undefined,
      children: [MOCK_CHAT_SPAN_WITH_TOOL as ModelTraceSpanNode, MOCK_TOOL_SPAN as ModelTraceSpanNode],
    };

    const result = synthesizeStrandsChatMessages(
      MOCK_STRANDS_USER_INPUT,
      MOCK_STRANDS_PLAIN_STRING_OUTPUT,
      [eventLoopSpan as ModelTraceSpanNode],
    );

    expect(result).toHaveLength(4);
    expect(result?.[1]?.tool_calls).toBeDefined();
    expect(result?.[2]?.role).toBe('tool');
  });

  it('should synthesize conversation with reasoning content', () => {
    const chatSpanWithReasoning: Partial<ModelTraceSpanNode> = {
      title: 'chat',
      type: ModelSpanType.CHAT_MODEL,
      inputs: MOCK_STRANDS_USER_INPUT,
      outputs: MOCK_STRANDS_REASONING_OUTPUT,
    };

    const children = [
      {
        ...chatSpanWithReasoning,
        children: [MOCK_TOOL_SPAN as ModelTraceSpanNode],
      } as ModelTraceSpanNode,
    ];

    const result = synthesizeStrandsChatMessages(
      MOCK_STRANDS_USER_INPUT,
      MOCK_STRANDS_PLAIN_STRING_OUTPUT,
      children,
    );

    // Should have: user, assistant with reasoning + tool calls, tool result, final assistant
    expect(result).toHaveLength(4);

    // User message
    expect(result?.[0]).toEqual(
      expect.objectContaining({
        role: 'user',
      }),
    );

    // Assistant with reasoning and tool calls
    expect(result?.[1]).toEqual(
      expect.objectContaining({
        role: 'assistant',
        reasoning: 'Let me think about this step by step. 2+2=4.',
        tool_calls: expect.arrayContaining([
          expect.objectContaining({
            id: 'call_PHijhFE1QjCO6ZuW1F9yYAXo',
          }),
        ]),
      }),
    );

    // Tool result
    expect(result?.[2]).toEqual(
      expect.objectContaining({
        role: 'tool',
      }),
    );

    // Final response
    expect(result?.[3]).toEqual(
      expect.objectContaining({
        role: 'assistant',
        content: 'The result of (2 + 2) is 4.',
      }),
    );
  });

  it('should not duplicate response when reasoning present without tool calls', () => {
    // Simulate a simple query with reasoning but no tools
    const reasoningOnlyOutput = [
      {
        reasoningContent: {
          reasoningText: {
            text: 'Let me calculate 2+2. That equals 4.',
            signature: 'sig123',
          },
        },
      },
      { text: '2 + 2 = 4' },
    ];

    const chatSpanWithReasoningNoTools: Partial<ModelTraceSpanNode> = {
      title: 'chat',
      type: ModelSpanType.CHAT_MODEL,
      inputs: MOCK_STRANDS_USER_INPUT,
      outputs: reasoningOnlyOutput,
    };

    const children = [chatSpanWithReasoningNoTools as ModelTraceSpanNode];

    // Root output is same as CHAT_MODEL text output
    const result = synthesizeStrandsChatMessages(MOCK_STRANDS_USER_INPUT, '2 + 2 = 4', children);

    // Should have only 2 messages: user + assistant (with reasoning), NOT a duplicate assistant
    expect(result).toHaveLength(2);
    expect(result?.[0]).toEqual(expect.objectContaining({ role: 'user' }));
    expect(result?.[1]).toEqual(
      expect.objectContaining({
        role: 'assistant',
        content: '2 + 2 = 4',
        reasoning: 'Let me calculate 2+2. That equals 4.',
      }),
    );
  });
});

