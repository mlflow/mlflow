import { normalizeConversation } from "../ModelTraceExplorer.utils";

export const MOCK_DSPY_INPUT = {
  question: 'What is the capital of France?',
};

export const MOCK_DSPY_OUTPUT = {
  answer: 'The capital of France is Paris.',
};

describe('normalizeConversation', () => {

  it('should handle dspy input', () => {
    expect(normalizeConversation(MOCK_DSPY_INPUT)).toEqual([
      expect.objectContaining({
        role: 'user',
        content: expect.stringMatching(/what is the capital of france/i),
      }),
    ]);
  });

  it('should handle dspy output', () => {
    expect(normalizeConversation(MOCK_DSPY_OUTPUT)).toEqual([
      expect.objectContaining({
        content: expect.stringMatching(/capital of france is paris/i),
        role: 'assistant',
      }),
    ]);
  });

});