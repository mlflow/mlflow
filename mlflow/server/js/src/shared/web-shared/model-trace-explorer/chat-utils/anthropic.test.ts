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
});
