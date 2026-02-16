import { describe, it, expect } from '@jest/globals';
import { getFormValuesFromScorer } from './scorerCardUtils';
import type { LLMScorer, CustomCodeScorer } from './types';
import { TEMPLATE_INSTRUCTIONS_MAP } from './prompts';
import { ScorerEvaluationScope } from './constants';

describe('scorerCardUtils', () => {
  describe('getFormValuesFromScorer', () => {
    describe('Golden Path - Successful Operations', () => {
      it('should successfully convert LLM scorer with all properties to form data', () => {
        // Arrange
        const llmScorer: LLMScorer = {
          name: 'Test LLM Scorer',
          type: 'llm',
          sampleRate: 75,
          filterString: 'status == "success"',
          llmTemplate: 'Safety',
          guidelines: ['Be objective', 'Consider context', 'Rate consistently'],
        };

        // Act
        const result = getFormValuesFromScorer(llmScorer);

        // Assert
        expect(result).toEqual({
          llmTemplate: 'Safety',
          name: 'Test LLM Scorer',
          sampleRate: 75,
          code: '',
          scorerType: 'llm',
          guidelines: 'Be objective\nConsider context\nRate consistently',
          instructions: TEMPLATE_INSTRUCTIONS_MAP['Safety'],
          filterString: 'status == "success"',
          model: '',
          disableMonitoring: undefined,
          isInstructionsJudge: undefined,
          evaluationScope: ScorerEvaluationScope.TRACES,
          outputTypeKind: 'default',
        });
      });

      it('should successfully convert custom code scorer with all properties to form data', () => {
        // Arrange
        const customCodeScorer: CustomCodeScorer = {
          name: 'Test Custom Scorer',
          type: 'custom-code',
          sampleRate: 50,
          filterString: 'model_name == "gpt-4"',
          code: 'def evaluate(input, output):\n    return {"score": 1.0}',
          callSignature: '',
          originalFuncName: '',
        };

        // Act
        const result = getFormValuesFromScorer(customCodeScorer);

        // Assert
        expect(result).toEqual({
          llmTemplate: '',
          name: 'Test Custom Scorer',
          sampleRate: 50,
          code: 'def evaluate(input, output):\n    return {"score": 1.0}',
          scorerType: 'custom-code',
          guidelines: '',
          instructions: '',
          filterString: 'model_name == "gpt-4"',
          model: '',
          evaluationScope: ScorerEvaluationScope.TRACES,
        });
      });

      it('should convert LLM scorer with minimal properties to form data', () => {
        // Arrange
        const llmScorer: LLMScorer = {
          name: 'Minimal LLM Scorer',
          type: 'llm',
        };

        // Act
        const result = getFormValuesFromScorer(llmScorer);

        // Assert
        expect(result).toEqual({
          llmTemplate: '',
          name: 'Minimal LLM Scorer',
          sampleRate: 0,
          code: '',
          scorerType: 'llm',
          guidelines: '',
          instructions: '',
          filterString: '',
          model: '',
          evaluationScope: ScorerEvaluationScope.TRACES,
          disableMonitoring: undefined,
          isInstructionsJudge: undefined,
          outputTypeKind: 'default',
        });
      });

      it('should convert custom code scorer with minimal properties to form data', () => {
        // Arrange
        const customCodeScorer: CustomCodeScorer = {
          name: 'Minimal Custom Scorer',
          type: 'custom-code',
          code: 'return {"score": 0.5}',
          callSignature: '',
          originalFuncName: '',
        };

        // Act
        const result = getFormValuesFromScorer(customCodeScorer);

        // Assert
        expect(result).toEqual({
          llmTemplate: '',
          name: 'Minimal Custom Scorer',
          sampleRate: 0,
          code: 'return {"score": 0.5}',
          scorerType: 'custom-code',
          guidelines: '',
          instructions: '',
          filterString: '',
          model: '',
          evaluationScope: ScorerEvaluationScope.TRACES,
        });
      });
    });

    describe('Edge Cases', () => {
      it('should handle LLM scorer with empty name', () => {
        // Arrange
        const llmScorer: LLMScorer = {
          name: '',
          type: 'llm',
          sampleRate: 25,
        };

        // Act
        const result = getFormValuesFromScorer(llmScorer);

        // Assert
        expect(result.name).toBe('');
        expect(result.scorerType).toBe('llm');
        expect(result.sampleRate).toBe(25);
      });

      it('should handle custom code scorer with empty code', () => {
        // Arrange
        const customCodeScorer: CustomCodeScorer = {
          name: 'Empty Code Scorer',
          type: 'custom-code',
          code: '',
          callSignature: '',
          originalFuncName: '',
        };

        // Act
        const result = getFormValuesFromScorer(customCodeScorer);

        // Assert
        expect((result as any).code).toBe('');
        expect(result.name).toBe('Empty Code Scorer');
        expect(result.scorerType).toBe('custom-code');
      });

      it('should handle LLM scorer with empty guidelines array', () => {
        // Arrange
        const llmScorer: LLMScorer = {
          name: 'No Guidelines Scorer',
          type: 'llm',
          guidelines: [],
        };

        // Act
        const result = getFormValuesFromScorer(llmScorer);

        // Assert
        expect((result as any).guidelines).toBe('');
        expect(result.scorerType).toBe('llm');
      });

      it('should handle LLM scorer with single guideline', () => {
        // Arrange
        const llmScorer: LLMScorer = {
          name: 'Single Guideline Scorer',
          type: 'llm',
          guidelines: ['Single guideline'],
        };

        // Act
        const result = getFormValuesFromScorer(llmScorer);

        // Assert
        expect((result as any).guidelines).toBe('Single guideline');
      });

      it('should handle scorer with zero sample rate', () => {
        // Arrange
        const llmScorer: LLMScorer = {
          name: 'Zero Sample Rate Scorer',
          type: 'llm',
          sampleRate: 0,
        };

        // Act
        const result = getFormValuesFromScorer(llmScorer);

        // Assert
        expect(result.sampleRate).toBe(0);
      });

      it('should handle scorer with undefined sample rate', () => {
        // Arrange
        const llmScorer: LLMScorer = {
          name: 'Undefined Sample Rate Scorer',
          type: 'llm',
        };

        // Act
        const result = getFormValuesFromScorer(llmScorer);

        // Assert
        expect(result.sampleRate).toBe(0);
      });

      it('should handle scorer with undefined filter string', () => {
        // Arrange
        const llmScorer: LLMScorer = {
          name: 'No Filter Scorer',
          type: 'llm',
        };

        // Act
        const result = getFormValuesFromScorer(llmScorer);

        // Assert
        expect(result.filterString).toBe('');
      });

      it('should handle LLM scorer with undefined built-in scorer class', () => {
        // Arrange
        const llmScorer: LLMScorer = {
          name: 'No Built-in Class Scorer',
          type: 'llm',
        };

        // Act
        const result = getFormValuesFromScorer(llmScorer);

        // Assert
        expect((result as any).llmTemplate).toBe('');
      });

      it('should handle guidelines with newline characters properly', () => {
        // Arrange
        const llmScorer: LLMScorer = {
          name: 'Multiline Guidelines Scorer',
          type: 'llm',
          guidelines: ['First line\nwith newline', 'Second line', 'Third\nmultiline\nguideline'],
        };

        // Act
        const result = getFormValuesFromScorer(llmScorer);

        // Assert
        expect((result as any).guidelines).toBe('First line\nwith newline\nSecond line\nThird\nmultiline\nguideline');
      });

      it('should handle negative sample rate', () => {
        // Arrange
        const llmScorer: LLMScorer = {
          name: 'Negative Sample Rate',
          type: 'llm',
          sampleRate: -10,
        };

        // Act
        const result = getFormValuesFromScorer(llmScorer);

        // Assert
        expect(result.sampleRate).toBe(-10);
      });

      it('should handle sample rate above 100', () => {
        // Arrange
        const llmScorer: LLMScorer = {
          name: 'High Sample Rate',
          type: 'llm',
          sampleRate: 150,
        };

        // Act
        const result = getFormValuesFromScorer(llmScorer);

        // Assert
        expect(result.sampleRate).toBe(150);
      });

      it('should handle non-integer sample rate', () => {
        // Arrange
        const llmScorer: LLMScorer = {
          name: 'Decimal Sample Rate',
          type: 'llm',
          sampleRate: 23.5,
        };

        // Act
        const result = getFormValuesFromScorer(llmScorer);

        // Assert
        expect(result.sampleRate).toBe(23.5);
      });
    });

    describe('Error Conditions', () => {
      it('should handle null name gracefully', () => {
        // Arrange
        const llmScorer = {
          name: null as any,
          type: 'llm' as const,
        };

        // Act
        const result = getFormValuesFromScorer(llmScorer);

        // Assert
        expect(result.name).toBe('');
      });

      it('should handle undefined name gracefully', () => {
        // Arrange
        const llmScorer = {
          name: undefined as any,
          type: 'llm' as const,
        };

        // Act
        const result = getFormValuesFromScorer(llmScorer);

        // Assert
        expect(result.name).toBe('');
      });

      it('should handle null guidelines array gracefully', () => {
        // Arrange
        const llmScorer: LLMScorer = {
          name: 'Null Guidelines Scorer',
          type: 'llm',
          guidelines: null as any,
        };

        // Act
        const result = getFormValuesFromScorer(llmScorer);

        // Assert
        expect((result as any).guidelines).toBe('');
      });

      it('should handle undefined guidelines array gracefully', () => {
        // Arrange
        const llmScorer: LLMScorer = {
          name: 'Undefined Guidelines Scorer',
          type: 'llm',
          guidelines: undefined,
        };

        // Act
        const result = getFormValuesFromScorer(llmScorer);

        // Assert
        expect((result as any).guidelines).toBe('');
      });

      it('should handle null filter string gracefully', () => {
        // Arrange
        const llmScorer = {
          name: 'Null Filter Scorer',
          type: 'llm' as const,
          filterString: null as any,
        };

        // Act
        const result = getFormValuesFromScorer(llmScorer);

        // Assert
        expect(result.filterString).toBe('');
      });

      it('should handle null built-in scorer class gracefully', () => {
        // Arrange
        const llmScorer: LLMScorer = {
          name: 'Null Built-in Class Scorer',
          type: 'llm',
          llmTemplate: null as any,
        };

        // Act
        const result = getFormValuesFromScorer(llmScorer);

        // Assert
        expect((result as any).llmTemplate).toBe('');
      });

      it('should handle null code for custom scorer gracefully', () => {
        // Arrange
        const customCodeScorer = {
          name: 'Null Code Scorer',
          type: 'custom-code' as const,
          code: null as any,
          callSignature: '',
          originalFuncName: '',
        };

        // Act
        const result = getFormValuesFromScorer(customCodeScorer);

        // Assert
        expect((result as any).code).toBe('');
      });

      it('should handle undefined code for custom scorer gracefully', () => {
        // Arrange
        const customCodeScorer = {
          name: 'Undefined Code Scorer',
          type: 'custom-code' as const,
          code: undefined as any,
          callSignature: '',
          originalFuncName: '',
        };

        // Act
        const result = getFormValuesFromScorer(customCodeScorer);

        // Assert
        expect((result as any).code).toBe('');
      });

      it('should handle null sample rate gracefully', () => {
        // Arrange
        const llmScorer = {
          name: 'Null Sample Rate Scorer',
          type: 'llm' as const,
          sampleRate: null as any,
        };

        // Act
        const result = getFormValuesFromScorer(llmScorer);

        // Assert
        expect(result.sampleRate).toBe(0);
      });
    });
  });
});
