import { normalizeConversation } from '../ModelTraceExplorer.utils';

const MOCK_DSPY_INPUT = {
  messages: [
    {
      role: 'system',
      content:
        'Your input fields are:\n1. `passage` (str): a passage to summarize\n\nYour output fields are:\n1. `reasoning` (str)\n2. `summary` (str): a one-line summary of the passage\n\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## passage ## ]]\n{passage}\n\n[[ ## reasoning ## ]]\n{reasoning}\n\n[[ ## summary ## ]]\n{summary}\n\n[[ ## completed ## ]]\n\nIn adhering to this structure, your objective is: \n        Given a passage, generate a summary.',
    },
    {
      role: 'user',
      content:
        "[[ ## passage ## ]]\nMLflow Tracing is a feature that enhances LLM observability in your Generative AI (GenAI) applications by capturing detailed information about the execution of your application's services. Tracing provides a way to record the inputs, outputs, and metadata associated with each intermediate step of a request, enabling you to easily pinpoint the source of bugs and unexpected behaviors.\n\nRespond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## summary ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.",
    },
  ],
  prompt: null,
};

const MOCK_DSPY_OUTPUT = [
  '[[ ## reasoning ## ]]\nThe passage explains the functionality of MLflow Tracing in the context of Generative AI applications. It highlights how this feature improves observability by recording detailed information about the execution process, which aids in identifying bugs and unexpected behaviors. The emphasis is on the benefits of capturing inputs, outputs, and metadata for each step of a request.\n\n[[ ## summary ## ]]\nMLflow Tracing enhances observability in Generative AI applications by recording detailed execution information, helping to identify bugs and unexpected behaviors.\n\n[[ ## completed ## ]]',
];

describe('normalizeConversation', () => {
  it('should handle dspy input', () => {
    const conv = normalizeConversation(MOCK_DSPY_INPUT, 'dspy');
    expect(conv).toEqual([
      expect.objectContaining({
        role: 'system',
        content: expect.stringContaining('Your input fields are:'),
      }),
      expect.objectContaining({
        role: 'user',
        content: expect.stringContaining('[[ ## passage ## ]]'),
      }),
    ]);
    // Ensure single newlines are converted to hard breaks for markdown rendering
    expect(conv?.[0].content).toContain('  \n');
  });

  it('should handle dspy output', () => {
    const conv = normalizeConversation(MOCK_DSPY_OUTPUT, 'dspy');
    expect(conv).toEqual([
      expect.objectContaining({
        content: expect.stringContaining('[[ ## reasoning ## ]]'),
        role: 'assistant',
      }),
    ]);
    expect(conv?.[0].content).toContain('  \n');
  });
});
