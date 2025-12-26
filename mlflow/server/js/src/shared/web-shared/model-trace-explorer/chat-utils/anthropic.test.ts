import { describe, it, expect } from '@jest/globals';

import { normalizeConversation } from '../ModelTraceExplorer.utils';

const MOCK_ANTHROPIC_INPUT = {
  max_tokens: 1024,
  messages: [
    {
      role: 'user',
      content: [
        {
          type: 'image',
          source: {
            type: 'url',
            url: 'https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg',
          },
        },
        {
          type: 'text',
          text: 'Describe this image.',
        },
      ],
    },
  ],
  model: 'claude-sonnet-4-20250514',
};

const MOCK_ANTHROPIC_OUTPUT = {
  id: 'msg_01QU6qxHJ25f73LPeRMF94vd',
  content: [
    {
      citations: null,
      text: "This is a close-up macro photograph of an ant captured in a defensive or alert posture. The ant appears to be rearing up on its hind legs with its front legs raised, creating a dramatic silhouette against the blurred background. The ant has a dark, segmented body with the characteristic three main body parts (head, thorax, and abdomen) clearly visible. Its long, thin antennae are extended forward, and its legs are positioned in what appears to be a threatening or defensive stance.\n\nThe photograph is taken at ground level, showing the ant on what looks like a concrete or stone surface with a shallow depth of field that creates a soft, warm-toned bokeh effect in the background. The lighting emphasizes the ant's form and creates nice contrast, making the subject stand out prominently. This type of behavior is typical of certain ant species when they feel threatened or are defending their territory.",
      type: 'text',
    },
  ],
  model: 'claude-sonnet-4-20250514',
  role: 'assistant',
  stop_reason: 'end_turn',
  stop_sequence: null,
  type: 'message',
  usage: {
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
    input_tokens: 1552,
    output_tokens: 194,
    server_tool_use: null,
    service_tier: 'standard',
  },
};

// Output with extended thinking
const MOCK_ANTHROPIC_OUTPUT_WITH_THINKING = {
  id: 'msg_01XYZ',
  content: [
    {
      type: 'thinking',
      thinking:
        'Let me count the letters in "strawberry": s-t-r-a-w-b-e-r-r-y. I see r at position 3, 8, and 9. So there are 3 r\'s.',
    },
    {
      type: 'text',
      text: 'There are 3 r\'s in the word "strawberry".',
    },
  ],
  model: 'claude-sonnet-4-20250514',
  role: 'assistant',
  stop_reason: 'end_turn',
  type: 'message',
};

// Output with thinking but no text
const MOCK_ANTHROPIC_OUTPUT_THINKING_ONLY = {
  id: 'msg_01ABC',
  content: [
    {
      type: 'thinking',
      thinking: 'Processing the request...',
    },
  ],
  model: 'claude-sonnet-4-20250514',
  role: 'assistant',
  stop_reason: 'end_turn',
  type: 'message',
};

// Output with tool use and thinking
const MOCK_ANTHROPIC_OUTPUT_WITH_THINKING_AND_TOOL = {
  id: 'msg_01DEF',
  content: [
    {
      type: 'thinking',
      thinking: 'I need to use the calculator tool to multiply 6 and 7.',
    },
    {
      type: 'tool_use',
      id: 'tool_001',
      name: 'calculator',
      input: { operation: 'multiply', a: 6, b: 7 },
    },
  ],
  model: 'claude-sonnet-4-20250514',
  role: 'assistant',
  stop_reason: 'tool_use',
  type: 'message',
};

describe('normalizeConversation', () => {
  it('should handle anthropic input', () => {
    expect(normalizeConversation(MOCK_ANTHROPIC_INPUT, 'anthropic')).toEqual([
      expect.objectContaining({
        role: 'user',
        content: expect.stringMatching(/describe this image/i),
      }),
    ]);
  });

  it('should handle anthropic output', () => {
    expect(normalizeConversation(MOCK_ANTHROPIC_OUTPUT, 'anthropic')).toEqual([
      expect.objectContaining({
        content: expect.stringMatching(/this is a close-up macro photograph of an ant/i),
        role: 'assistant',
      }),
    ]);
  });

  describe('Anthropic extended thinking support', () => {
    it('handles output with thinking and text', () => {
      const result = normalizeConversation(MOCK_ANTHROPIC_OUTPUT_WITH_THINKING, 'anthropic');
      expect(result).toEqual([
        expect.objectContaining({
          role: 'assistant',
          content: 'There are 3 r\'s in the word "strawberry".',
          reasoning: expect.stringContaining('s-t-r-a-w-b-e-r-r-y'),
        }),
      ]);
    });

    it('handles output with thinking and tool use', () => {
      const result = normalizeConversation(MOCK_ANTHROPIC_OUTPUT_WITH_THINKING_AND_TOOL, 'anthropic');
      expect(result).toEqual([
        expect.objectContaining({
          role: 'assistant',
          reasoning: expect.stringContaining('calculator tool'),
          tool_calls: [
            expect.objectContaining({
              id: 'tool_001',
              function: {
                name: 'calculator',
                arguments: expect.stringContaining('multiply'),
              },
            }),
          ],
        }),
      ]);
    });

    it('handles output without thinking (backward compatibility)', () => {
      const result = normalizeConversation(MOCK_ANTHROPIC_OUTPUT, 'anthropic');
      expect(result?.[0]).not.toHaveProperty('reasoning');
    });
  });
});
