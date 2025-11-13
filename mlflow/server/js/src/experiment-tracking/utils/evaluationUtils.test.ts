import { jest, describe, it, expect } from '@jest/globals';
import { buildSystemPrompt, buildUserPrompt, extractTemplateVariables } from './evaluationUtils';

jest.mock('./TraceUtils', () => ({
  extractInputs: jest.fn(),
  extractOutputs: jest.fn(),
  extractExpectations: jest.fn(),
}));

describe('evaluationUtils', () => {
  describe('extractTemplateVariables', () => {
    describe('Golden Path - Successful Operations', () => {
      it('should extract template variables in order they appear', () => {
        // Arrange
        const instructions = 'Evaluate if {{outputs}} correctly answers {{inputs}}';

        // Act
        const result = extractTemplateVariables(instructions);

        // Assert
        expect(result).toEqual(['outputs', 'inputs']);
      });

      it('should extract all three reserved variables', () => {
        // Arrange
        const instructions = 'Check {{inputs}}, {{outputs}}, and {{expectations}}';

        // Act
        const result = extractTemplateVariables(instructions);

        // Assert
        expect(result).toEqual(['inputs', 'outputs', 'expectations']);
      });

      it('should handle only one variable', () => {
        // Arrange
        const instructions = 'Review the {{outputs}}';

        // Act
        const result = extractTemplateVariables(instructions);

        // Assert
        expect(result).toEqual(['outputs']);
      });

      it('should extract trace variable', () => {
        // Arrange
        const instructions = 'Evaluate the {{trace}}';

        // Act
        const result = extractTemplateVariables(instructions);

        // Assert
        expect(result).toEqual(['trace']);
      });
    });

    describe('Edge Cases', () => {
      it('should handle duplicate variables', () => {
        // Arrange
        const instructions = 'Compare {{inputs}} with {{outputs}} and verify {{inputs}} again';

        // Act
        const result = extractTemplateVariables(instructions);

        // Assert - should only include each variable once
        expect(result).toEqual(['inputs', 'outputs']);
      });

      it('should ignore non-reserved variables', () => {
        // Arrange
        const instructions = 'Use {{inputs}} and {{custom_var}} to evaluate {{outputs}}';

        // Act
        const result = extractTemplateVariables(instructions);

        // Assert - should only include reserved variables
        expect(result).toEqual(['inputs', 'outputs']);
      });

      it('should handle variables with whitespace', () => {
        // Arrange
        const instructions = 'Check {{ inputs }} and {{  outputs  }}';

        // Act
        const result = extractTemplateVariables(instructions);

        // Assert
        expect(result).toEqual(['inputs', 'outputs']);
      });

      it('should return empty array when no variables found', () => {
        // Arrange
        const instructions = 'No variables here';

        // Act
        const result = extractTemplateVariables(instructions);

        // Assert
        expect(result).toEqual([]);
      });

      it('should handle empty string', () => {
        // Arrange
        const instructions = '';

        // Act
        const result = extractTemplateVariables(instructions);

        // Assert
        expect(result).toEqual([]);
      });
    });
  });

  describe('buildSystemPrompt', () => {
    describe('Golden Path - Successful Operations', () => {
      it('should successfully build system prompt with instructions', () => {
        // Arrange
        const instructions = 'Rate the response on a scale of 1-5 for accuracy and relevance';

        // Act
        const result = buildSystemPrompt(instructions);

        // Assert
        expect(result).toEqual(
          `You are an expert judge tasked with evaluating the performance of an AI
agent on a particular query. You will be given instructions that describe the criteria and
methodology for evaluating the agent's performance on the query.

Your task: Rate the response on a scale of 1-5 for accuracy and relevance.

Please provide your assessment in the following JSON format only (no markdown):

{
    "result": "The evaluation rating/result",
    "rationale": "Detailed explanation for the evaluation"
}`,
        );
      });
    });
  });

  describe('buildUserPrompt', () => {
    describe('Golden Path - Successful Operations', () => {
      it('should successfully build user prompt with all fields present when all template variables included', () => {
        // Arrange
        const inputs = '{"query": "What is AI?"}';
        const outputs = '{"response": "AI stands for Artificial Intelligence"}';
        const expectations = { accuracy: 0.9, completeness: 'high' };
        const templateVariables = ['inputs', 'outputs', 'expectations'];

        // Act
        const result = buildUserPrompt(inputs, outputs, expectations, templateVariables);

        // Assert
        expect(result).toEqual(
          `inputs: {"query": "What is AI?"}
outputs: {"response": "AI stands for Artificial Intelligence"}
expectations: {
  "accuracy": 0.9,
  "completeness": "high"
}`,
        );
      });

      it('should only include fields referenced in template variables', () => {
        // Arrange - only inputs and outputs in template, but expectations exist
        const inputs = 'test input';
        const outputs = 'test output';
        const expectations = { score: 5 };
        const templateVariables = ['inputs', 'outputs'];

        // Act
        const result = buildUserPrompt(inputs, outputs, expectations, templateVariables);

        // Assert - expectations should NOT be included
        expect(result).toEqual('inputs: test input\noutputs: test output');
      });

      it('should build prompt with partial fields based on template variables', () => {
        // Arrange - only inputs in template
        let result = buildUserPrompt('test input', 'test output', {}, ['inputs']);
        expect(result).toEqual('inputs: test input');

        // Arrange - only outputs in template
        result = buildUserPrompt('test input', 'test output', {}, ['outputs']);
        expect(result).toEqual('outputs: test output');

        // Arrange - only expectations in template
        result = buildUserPrompt('test input', null, { score: 5 }, ['expectations']);
        expect(result).toEqual(`expectations: {
  "score": 5
}`);

        // Arrange - inputs and outputs in template
        result = buildUserPrompt('question', 'answer', { score: 5 }, ['inputs', 'outputs']);
        expect(result).toEqual('inputs: question\noutputs: answer');
      });

      it('should preserve order of template variables', () => {
        // Arrange - outputs before inputs in template
        const inputs = 'test input';
        const outputs = 'test output';
        const templateVariables = ['outputs', 'inputs'];

        // Act
        const result = buildUserPrompt(inputs, outputs, {}, templateVariables);

        // Assert - outputs should appear first
        expect(result).toEqual('outputs: test output\ninputs: test input');
      });
    });

    describe('Edge Cases', () => {
      it('should return fallback message when all fields are null/empty', () => {
        // Arrange - all null with template variables
        let result = buildUserPrompt(null, null, {}, ['inputs', 'outputs']);
        expect(result).toEqual('Follow the instructions from the first message');

        // Arrange - all undefined
        result = buildUserPrompt(undefined as any, undefined as any, {}, ['inputs', 'outputs']);
        expect(result).toEqual('Follow the instructions from the first message');

        // Arrange - empty strings
        result = buildUserPrompt('', '', {}, ['inputs', 'outputs']);
        expect(result).toEqual('Follow the instructions from the first message');
      });

      it('should return fallback message when template variables is empty', () => {
        // Arrange - empty template variables array
        const result = buildUserPrompt('test input', 'test output', { score: 5 }, []);
        expect(result).toEqual('Follow the instructions from the first message');
      });

      it('should return fallback message when no matching data for template variables', () => {
        // Arrange - template asks for expectations but none exist
        const result = buildUserPrompt(null, null, {}, ['expectations']);
        expect(result).toEqual('Follow the instructions from the first message');
      });

      it('should handle complex data types and special formatting', () => {
        // Arrange - complex nested expectations
        const expectations = {
          nested: { level1: { level2: 'value' } },
          array: [1, 2, 3],
          nullValue: null,
          boolValue: true,
        };
        let result = buildUserPrompt('test', null, expectations, ['inputs', 'expectations']);
        expect(result).toEqual(`inputs: test
expectations: {
  "nested": {
    "level1": {
      "level2": "value"
    }
  },
  "array": [
    1,
    2,
    3
  ],
  "nullValue": null,
  "boolValue": true
}`);

        // Arrange - newlines in inputs/outputs
        result = buildUserPrompt('Line 1\nLine 2', 'Output line 1\nOutput line 2', {}, ['inputs', 'outputs']);
        expect(result).toEqual('inputs: Line 1\nLine 2\noutputs: Output line 1\nOutput line 2');

        // Arrange - JSON string inputs
        result = buildUserPrompt('{"key": "value"}', null, {}, ['inputs']);
        expect(result).toEqual('inputs: {"key": "value"}');
      });
    });

    describe('Error Conditions', () => {
      it('should handle null/undefined expectations object gracefully', () => {
        // Arrange - null expectations
        let result = buildUserPrompt('test', null, null as any, ['inputs']);
        expect(result).toEqual('inputs: test');

        // Arrange - undefined expectations
        result = buildUserPrompt(null, 'test', undefined as any, ['outputs']);
        expect(result).toEqual('outputs: test');
      });

      it('should skip variables not in data even if in template', () => {
        // Arrange - template wants all three but only inputs provided
        const result = buildUserPrompt('test input', null, {}, ['inputs', 'outputs', 'expectations']);
        expect(result).toEqual('inputs: test input');
      });

      it('should ignore trace variable since it is not in dataMap', () => {
        // Arrange - template includes trace but we don't have trace data in dataMap
        const result = buildUserPrompt('test input', 'test output', {}, ['inputs', 'trace', 'outputs']);
        // Assert - trace is skipped, only inputs and outputs included
        expect(result).toEqual('inputs: test input\noutputs: test output');
      });
    });
  });
});
