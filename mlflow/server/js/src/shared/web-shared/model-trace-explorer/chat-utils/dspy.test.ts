import { describe, it, expect } from '@jest/globals';

import { normalizeConversation } from '../ModelTraceExplorer.utils';
import { formatDspySections } from './dspy';

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
        content: expect.stringContaining('#### Passage'),
      }),
    ]);
    // Standalone markers should be converted to headings (first one has no rule above)
    expect(conv?.[0].content).toContain('#### Passage');
    expect(conv?.[0].content).toContain('#### Reasoning');
    expect(conv?.[0].content).toContain('#### Summary');
    // The completed marker should be removed
    expect(conv?.[0].content).not.toContain('[[ ## completed ## ]]');
    // Inline references inside backticks should be preserved
    expect(conv?.[1].content).toContain('`[[ ## reasoning ## ]]`');
    expect(conv?.[1].content).toContain('`[[ ## summary ## ]]`');
    expect(conv?.[1].content).toContain('`[[ ## completed ## ]]`');
    // Ensure single newlines are converted to hard breaks for markdown rendering
    expect(conv?.[0].content).toContain('  \n');
  });

  it('should handle dspy output', () => {
    const conv = normalizeConversation(MOCK_DSPY_OUTPUT, 'dspy');
    expect(conv).toEqual([
      expect.objectContaining({
        content: expect.stringContaining('#### Reasoning'),
        role: 'assistant',
      }),
    ]);
    expect(conv?.[0].content).toContain('#### Summary');
    // completed marker should be removed
    expect(conv?.[0].content).not.toContain('[[ ## completed ## ]]');
    expect(conv?.[0].content).toContain('  \n');
  });

  it('should render JSON output as a formatted code block', () => {
    const jsonOutput = ['{ "summary": "MLflow is great." }'];
    const conv = normalizeConversation(jsonOutput, 'dspy');
    expect(conv).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content: '```json\n{\n  "summary": "MLflow is great."\n}\n```',
      }),
    ]);
  });
});

describe('formatDspySections', () => {
  it('should convert standalone markers to markdown headings', () => {
    const input = '[[ ## reasoning ## ]]\nSome reasoning text';
    expect(formatDspySections(input)).toBe('#### Reasoning\nSome reasoning text');
  });

  it('should handle markers with leading/trailing whitespace on the line', () => {
    const input = '  [[ ## passage ## ]]  \nSome text';
    expect(formatDspySections(input)).toBe('#### Passage\nSome text');
  });

  it('should remove the completed marker entirely', () => {
    const input = 'Some text\n\n[[ ## completed ## ]]';
    expect(formatDspySections(input)).toBe('Some text\n\n');
  });

  it('should leave inline references unchanged', () => {
    const input = 'Start with `[[ ## reasoning ## ]]` then `[[ ## summary ## ]]`.';
    expect(formatDspySections(input)).toBe(input);
  });

  it('should title-case snake_case variable names', () => {
    const input = '[[ ## tool_name_0 ## ]]';
    expect(formatDspySections(input)).toBe('#### Tool Name 0');
  });

  it('should handle multiple sections', () => {
    const input = '[[ ## reasoning ## ]]\nThinking...\n\n[[ ## summary ## ]]\nDone.\n\n[[ ## completed ## ]]';
    const result = formatDspySections(input);
    expect(result).toContain('#### Reasoning');
    expect(result).toContain('#### Summary');
    expect(result).not.toContain('[[ ## completed ## ]]');
    expect(result).not.toContain('#### Completed');
  });

  it('should return text unchanged when there are no markers', () => {
    const input = 'Just regular text with no markers.';
    expect(formatDspySections(input)).toBe(input);
  });

  it('should return empty string unchanged', () => {
    expect(formatDspySections('')).toBe('');
  });
});
