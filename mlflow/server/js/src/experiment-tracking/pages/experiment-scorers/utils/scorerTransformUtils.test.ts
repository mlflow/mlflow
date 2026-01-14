import { describe, it, expect, jest } from '@jest/globals';
import {
  ScorerTransformationError,
  transformScorerConfig,
  transformScheduledScorer,
  convertFormDataToScheduledScorer,
  convertRegisterScorerResponseToConfig,
} from './scorerTransformUtils';
import type { RegisterScorerResponse } from '../api';
import type { ScorerConfig, LLMScorer, CustomCodeScorer } from '../types';
import type { LLMScorerFormData } from '../LLMScorerFormRenderer';
import type { CustomCodeScorerFormData } from '../CustomCodeScorerFormRenderer';

jest.mock('../../../../common/utils/FeatureUtils', () => ({
  isEvaluatingSessionsInScorersEnabled: () => true,
}));

describe('transformScorerConfig', () => {
  describe('Custom template (instructions-based) LLM scorer', () => {
    it('should transform instructions-based scorer with model', () => {
      const config: ScorerConfig = {
        name: 'Test Instructions Scorer',
        sample_rate: 0.8,
        filter_string: 'status="completed"',
        serialized_scorer: JSON.stringify({
          instructions_judge_pydantic_data: {
            instructions: 'Evaluate the response quality',
            model: 'databricks:/databricks-gpt-5',
          },
        }),
        custom: {},
      };

      const result = transformScorerConfig(config);

      expect(result).toEqual({
        name: 'Test Instructions Scorer',
        sampleRate: 80,
        filterString: 'status="completed"',
        type: 'llm',
        llmTemplate: 'Custom',
        instructions: 'Evaluate the response quality',
        model: 'databricks:/databricks-gpt-5',
        disableMonitoring: false,
        is_instructions_judge: true,
      });
    });

    it('should transform instructions-based scorer without model', () => {
      const config: ScorerConfig = {
        name: 'Test Instructions Scorer',
        serialized_scorer: JSON.stringify({
          instructions_judge_pydantic_data: {
            instructions: 'Evaluate the response quality',
          },
        }),
        custom: {},
      };

      const result = transformScorerConfig(config);

      expect(result).toEqual({
        name: 'Test Instructions Scorer',
        type: 'llm',
        llmTemplate: 'Custom',
        instructions: 'Evaluate the response quality',
        model: undefined,
        disableMonitoring: false,
        is_instructions_judge: true,
      });
    });
  });

  describe('Guidelines LLM scorer', () => {
    it('should transform Guidelines scorer with array guidelines', () => {
      const config: ScorerConfig = {
        name: 'Test Guidelines Scorer',
        sample_rate: 0.75,
        filter_string: 'status="completed"',
        serialized_scorer: JSON.stringify({
          builtin_scorer_class: 'Guidelines',
          builtin_scorer_pydantic_data: {
            guidelines: ['Guideline 1', 'Guideline 2'],
          },
        }),
        builtin: { name: 'Test Guidelines Scorer' },
      };

      const result = transformScorerConfig(config);

      expect(result).toEqual({
        name: 'Test Guidelines Scorer',
        sampleRate: 75,
        filterString: 'status="completed"',
        type: 'llm',
        llmTemplate: 'Guidelines',
        guidelines: ['Guideline 1', 'Guideline 2'],
        disableMonitoring: false,
        is_instructions_judge: false,
        model: undefined,
      });
    });

    it('should transform Guidelines scorer with string guideline to array', () => {
      const config: ScorerConfig = {
        name: 'Test Guidelines Scorer',
        serialized_scorer: JSON.stringify({
          builtin_scorer_class: 'Guidelines',
          builtin_scorer_pydantic_data: {
            guidelines: 'Single guideline',
          },
        }),
        builtin: { name: 'Test Guidelines Scorer' },
      };

      const result = transformScorerConfig(config);

      expect(result).toEqual({
        name: 'Test Guidelines Scorer',
        type: 'llm',
        llmTemplate: 'Guidelines',
        guidelines: ['Single guideline'],
        disableMonitoring: false,
        is_instructions_judge: false,
        model: undefined,
      });
    });

    it('should handle missing guidelines', () => {
      const config: ScorerConfig = {
        name: 'Test Guidelines Scorer',
        serialized_scorer: JSON.stringify({
          builtin_scorer_class: 'Guidelines',
          builtin_scorer_pydantic_data: {},
        }),
        builtin: { name: 'Test Guidelines Scorer' },
      };

      const result = transformScorerConfig(config);

      expect(result).toEqual({
        name: 'Test Guidelines Scorer',
        type: 'llm',
        llmTemplate: 'Guidelines',
        guidelines: [],
        disableMonitoring: false,
        is_instructions_judge: false,
        model: undefined,
      });
    });
  });

  describe('Built-in LLM scorers', () => {
    it('should transform built-in LLM scorer', () => {
      const config: ScorerConfig = {
        name: 'Test Builtin Scorer',
        sample_rate: 0.5,
        serialized_scorer: JSON.stringify({
          builtin_scorer_class: 'Safety',
        }),
        builtin: { name: 'Test Builtin Scorer' },
      };

      const result = transformScorerConfig(config);

      expect(result).toEqual({
        name: 'Test Builtin Scorer',
        sampleRate: 50,
        type: 'llm',
        llmTemplate: 'Safety',
        disableMonitoring: false,
        is_instructions_judge: false,
        model: undefined,
      });
    });
  });

  describe('Custom code scorers', () => {
    it('should transform custom scorer with function definition', () => {
      const config: ScorerConfig = {
        name: 'Test Custom Scorer',
        sample_rate: 1.0,
        filter_string: 'experiment_id="123"',
        serialized_scorer: JSON.stringify({
          call_source: 'return len(inputs["text"]) > 10',
          original_func_name: 'my_scorer',
          call_signature: '(inputs, outputs, metadata)',
        }),
        custom: {},
      };

      const result = transformScorerConfig(config);

      expect(result).toEqual({
        name: 'Test Custom Scorer',
        sampleRate: 100,
        filterString: 'experiment_id="123"',
        type: 'custom-code',
        code: 'def my_scorer(inputs, outputs, metadata):\n    return len(inputs["text"]) > 10',
        callSignature: '(inputs, outputs, metadata)',
        originalFuncName: 'my_scorer',
        disableMonitoring: false,
      });
    });

    it('should transform custom scorer without function definition', () => {
      const config: ScorerConfig = {
        name: 'Test Custom Scorer',
        serialized_scorer: JSON.stringify({
          call_source: 'def evaluate(inputs, outputs):\n    return True',
        }),
        custom: {},
      };

      const result = transformScorerConfig(config);

      expect(result).toEqual({
        name: 'Test Custom Scorer',
        type: 'custom-code',
        code: 'def evaluate(inputs, outputs):\n    return True',
        disableMonitoring: false,
        originalFuncName: undefined,
        callSignature: undefined,
      });
    });

    it('should handle missing call_source', () => {
      const config: ScorerConfig = {
        name: 'Test Custom Scorer',
        serialized_scorer: JSON.stringify({}),
        custom: {},
      };

      const result = transformScorerConfig(config);

      expect(result).toEqual({
        name: 'Test Custom Scorer',
        type: 'custom-code',
        code: '',
        disableMonitoring: false,
        originalFuncName: undefined,
        callSignature: undefined,
      });
    });
  });

  describe('Edge cases and error handling', () => {
    it('should handle undefined sample_rate', () => {
      const config: ScorerConfig = {
        name: 'Test Scorer',
        serialized_scorer: JSON.stringify({
          builtin_scorer_class: 'Safety',
        }),
        builtin: { name: 'Test Scorer' },
      };

      const result = transformScorerConfig(config);

      expect(result.sampleRate).toBeUndefined();
    });

    it('should handle falsy filter_string values', () => {
      const configWithUndefined: ScorerConfig = {
        name: 'Test Scorer',
        serialized_scorer: JSON.stringify({
          builtin_scorer_class: 'Safety',
        }),
        builtin: { name: 'Test Scorer' },
      };

      const configWithEmpty: ScorerConfig = {
        name: 'Test Scorer',
        filter_string: '',
        serialized_scorer: JSON.stringify({
          builtin_scorer_class: 'Safety',
        }),
        builtin: { name: 'Test Scorer' },
      };

      expect(transformScorerConfig(configWithUndefined)).not.toHaveProperty('filterString');
      expect(transformScorerConfig(configWithEmpty)).not.toHaveProperty('filterString');
    });

    it('should throw ScorerTransformationError for invalid JSON', () => {
      const config: ScorerConfig = {
        name: 'Invalid Scorer',
        serialized_scorer: 'invalid json',
      };

      expect(() => transformScorerConfig(config)).toThrow(ScorerTransformationError);
      expect(() => transformScorerConfig(config)).toThrow('Failed to parse scorer configuration:');
    });
  });
});

describe('transformScheduledScorer', () => {
  describe('LLM scorers', () => {
    it('should transform Custom template scorer with model', () => {
      const scorer: LLMScorer = {
        name: 'Test Instructions Scorer',
        sampleRate: 80,
        filterString: 'status="completed"',
        type: 'llm',
        llmTemplate: 'Custom',
        instructions: 'Evaluate the response quality',
        model: 'openai:/gpt-4o-mini',
        is_instructions_judge: true,
      };

      const result = transformScheduledScorer(scorer);

      const serialized = JSON.parse(result.serialized_scorer);
      expect(serialized.instructions_judge_pydantic_data).toEqual({
        instructions: 'Evaluate the response quality',
        model: 'openai:/gpt-4o-mini',
      });
      expect(result.custom).toEqual({});
    });

    it('should transform Custom template scorer without model', () => {
      const scorer: LLMScorer = {
        name: 'Test Instructions Scorer',
        sampleRate: 80,
        type: 'llm',
        llmTemplate: 'Custom',
        instructions: 'Evaluate the response quality',
        is_instructions_judge: true,
      };

      const result = transformScheduledScorer(scorer);

      const serialized = JSON.parse(result.serialized_scorer);
      expect(serialized.instructions_judge_pydantic_data).toEqual({
        instructions: 'Evaluate the response quality',
      });
      expect(serialized.instructions_judge_pydantic_data).not.toHaveProperty('model');
    });

    it('should transform Guidelines LLM scorer', () => {
      const scorer: LLMScorer = {
        name: 'Test Guidelines Scorer',
        sampleRate: 75,
        filterString: 'status="completed"',
        type: 'llm',
        llmTemplate: 'Guidelines',
        guidelines: ['Guideline 1', 'Guideline 2'],
        isSessionLevelScorer: false,
      };

      const result = transformScheduledScorer(scorer);

      expect(result).toEqual({
        name: 'Test Guidelines Scorer',
        sample_rate: 0.75,
        filter_string: 'status="completed"',
        serialized_scorer: JSON.stringify({
          mlflow_version: '3.3.2+ui',
          serialization_version: 1,
          is_session_level_scorer: false,
          name: 'Test Guidelines Scorer',
          builtin_scorer_class: 'Guidelines',
          builtin_scorer_pydantic_data: {
            name: 'Test Guidelines Scorer',
            required_columns: ['outputs', 'inputs'],
            guidelines: ['Guideline 1', 'Guideline 2'],
          },
        }),
        builtin: {
          name: 'Test Guidelines Scorer',
        },
      });
    });

    it('should transform built-in LLM scorer without instructions as built-in', () => {
      const scorer: LLMScorer = {
        name: 'Test Toxicity Scorer',
        sampleRate: 50,
        type: 'llm',
        llmTemplate: 'Safety',
        isSessionLevelScorer: false,
      };

      const result = transformScheduledScorer(scorer);

      expect(result).toEqual({
        name: 'Test Toxicity Scorer',
        sample_rate: 0.5,
        serialized_scorer: JSON.stringify({
          mlflow_version: '3.3.2+ui',
          serialization_version: 1,
          is_session_level_scorer: false,
          name: 'Test Toxicity Scorer',
          builtin_scorer_class: 'Safety',
          builtin_scorer_pydantic_data: {
            name: 'Test Toxicity Scorer',
            required_columns: ['outputs', 'inputs'],
          },
        }),
        builtin: {
          name: 'Test Toxicity Scorer',
        },
      });
    });

    it('should transform Safety template with instructions as instructions judge', () => {
      const scorer: LLMScorer = {
        name: 'Safety Judge',
        sampleRate: 80,
        type: 'llm',
        llmTemplate: 'Safety',
        instructions: 'Evaluate if the response is safe and appropriate.',
        model: 'openai:/gpt-4o-mini',
        is_instructions_judge: true,
      };

      const result = transformScheduledScorer(scorer);

      const serialized = JSON.parse(result.serialized_scorer);
      expect(serialized.instructions_judge_pydantic_data).toEqual({
        instructions: 'Evaluate if the response is safe and appropriate.',
        model: 'openai:/gpt-4o-mini',
      });
      expect(serialized.builtin_scorer_class).toBeNull();
      expect(result.custom).toEqual({});
      expect(result).not.toHaveProperty('builtin');
    });

    it('should transform RelevanceToQuery template with instructions as instructions judge', () => {
      const scorer: LLMScorer = {
        name: 'Relevance Judge',
        sampleRate: 75,
        type: 'llm',
        llmTemplate: 'RelevanceToQuery',
        instructions: 'Evaluate if the response is relevant to the query.',
        is_instructions_judge: true,
      };

      const result = transformScheduledScorer(scorer);

      const serialized = JSON.parse(result.serialized_scorer);
      expect(serialized.instructions_judge_pydantic_data).toEqual({
        instructions: 'Evaluate if the response is relevant to the query.',
      });
      expect(serialized.builtin_scorer_class).toBeNull();
      expect(result.custom).toEqual({});
    });

    it('should transform non-editable template (Correctness) as built-in even with instructions', () => {
      const scorer: LLMScorer = {
        name: 'Correctness Scorer',
        sampleRate: 60,
        type: 'llm',
        llmTemplate: 'Correctness',
        instructions: 'Some instructions that should be ignored',
      };

      const result = transformScheduledScorer(scorer);

      const serialized = JSON.parse(result.serialized_scorer);
      expect(serialized.builtin_scorer_class).toBe('Correctness');
      expect(serialized.instructions_judge_pydantic_data).toBeUndefined();
      expect(result.builtin).toEqual({ name: 'Correctness Scorer' });
    });

    it('should handle LLM scorer with undefined sampleRate', () => {
      const scorer: LLMScorer = {
        name: 'Test Scorer',
        type: 'llm',
        llmTemplate: 'Safety',
        isSessionLevelScorer: false,
      };

      const result = transformScheduledScorer(scorer);

      expect(result).not.toHaveProperty('sample_rate');
    });

    it('should handle LLM scorer with empty filterString', () => {
      const scorer: LLMScorer = {
        name: 'Test Scorer',
        filterString: '',
        type: 'llm',
        llmTemplate: 'Safety',
        isSessionLevelScorer: false,
      };

      const result = transformScheduledScorer(scorer);

      expect(result).not.toHaveProperty('filter_string');
    });

    it('should handle LLM scorer without llmTemplate', () => {
      const scorer = {
        name: 'Test Scorer',
        type: 'llm' as const,
        isSessionLevelScorer: false,
      };

      const result = transformScheduledScorer(scorer);

      expect(result.serialized_scorer).toBe('');
      expect(result).not.toHaveProperty('builtin');
    });
  });

  describe('Custom code scorers', () => {
    it('should transform custom code scorer', () => {
      const scorer: CustomCodeScorer = {
        name: 'Test Custom Scorer',
        sampleRate: 100,
        filterString: 'experiment_id="123"',
        type: 'custom-code',
        code: 'def my_scorer(inputs, outputs):\n    return True',
        callSignature: '',
        originalFuncName: '',
        isSessionLevelScorer: false,
      };

      const result = transformScheduledScorer(scorer);

      expect(result).toEqual({
        name: 'Test Custom Scorer',
        sample_rate: 1.0,
        filter_string: 'experiment_id="123"',
        serialized_scorer: JSON.stringify({
          mlflow_version: '3.3.2+ui',
          serialization_version: 1,
          is_session_level_scorer: false,
          name: 'Test Custom Scorer',
          call_source: 'def my_scorer(inputs, outputs):\n    return True',
          call_signature: '',
          original_func_name: '',
        }),
        custom: {},
      });
    });

    it('should handle custom code scorer with undefined sampleRate', () => {
      const scorer: CustomCodeScorer = {
        name: 'Test Custom Scorer',
        type: 'custom-code',
        code: 'return True',
        callSignature: '',
        originalFuncName: '',
        isSessionLevelScorer: false,
      };

      const result = transformScheduledScorer(scorer);

      expect(result).not.toHaveProperty('sample_rate');
    });
  });
});

describe('convertFormDataToScheduledScorer', () => {
  describe('Creating new scorers', () => {
    it('should create new Custom template scorer with model', () => {
      const formData: LLMScorerFormData & { scorerType: 'llm' } = {
        name: 'New Instructions Scorer',
        sampleRate: 80,
        filterString: 'status="success"',
        scorerType: 'llm',
        llmTemplate: 'Custom',
        instructions: 'Evaluate the response',
        model: 'openai:/gpt-4o-mini',
        isInstructionsJudge: true,
      };

      const result = convertFormDataToScheduledScorer(formData);

      expect(result).toEqual({
        name: 'New Instructions Scorer',
        sampleRate: 80,
        filterString: 'status="success"',
        type: 'llm',
        llmTemplate: 'Custom',
        guidelines: undefined,
        instructions: 'Evaluate the response',
        model: 'openai:/gpt-4o-mini',
        is_instructions_judge: true,
        isSessionLevelScorer: false,
      });
    });

    it('should create new Custom template scorer without model', () => {
      const formData: LLMScorerFormData & { scorerType: 'llm' } = {
        name: 'New Instructions Scorer',
        sampleRate: 80,
        scorerType: 'llm',
        llmTemplate: 'Custom',
        instructions: 'Evaluate the response',
        isInstructionsJudge: true,
        model: '',
      };

      const result = convertFormDataToScheduledScorer(formData);

      expect(result).toEqual({
        name: 'New Instructions Scorer',
        sampleRate: 80,
        filterString: '',
        type: 'llm',
        llmTemplate: 'Custom',
        guidelines: undefined,
        instructions: 'Evaluate the response',
        model: undefined,
        is_instructions_judge: true,
        isSessionLevelScorer: false,
      });
    });

    it('should create new Safety scorer with instructions as instructions judge', () => {
      const formData: LLMScorerFormData & { scorerType: 'llm' } = {
        name: 'New Safety Scorer',
        sampleRate: 80,
        filterString: 'status="success"',
        scorerType: 'llm',
        llmTemplate: 'Safety',
        instructions: 'Evaluate if the response is safe.',
        model: 'openai:/gpt-4o-mini',
        isInstructionsJudge: true,
      };

      const result = convertFormDataToScheduledScorer(formData);

      expect(result).toEqual({
        name: 'New Safety Scorer',
        sampleRate: 80,
        filterString: 'status="success"',
        type: 'llm',
        llmTemplate: 'Safety',
        guidelines: undefined,
        instructions: 'Evaluate if the response is safe.',
        model: 'openai:/gpt-4o-mini',
        is_instructions_judge: true,
        isSessionLevelScorer: false,
      });
    });

    it('should create new RelevanceToQuery scorer with instructions as instructions judge', () => {
      const formData: LLMScorerFormData & { scorerType: 'llm' } = {
        name: 'New Relevance Scorer',
        sampleRate: 75,
        scorerType: 'llm',
        llmTemplate: 'RelevanceToQuery',
        instructions: 'Evaluate if the response is relevant.',
        isInstructionsJudge: true,
        model: '',
      };

      const result = convertFormDataToScheduledScorer(formData);

      expect(result).toEqual({
        name: 'New Relevance Scorer',
        sampleRate: 75,
        filterString: '',
        type: 'llm',
        llmTemplate: 'RelevanceToQuery',
        guidelines: undefined,
        instructions: 'Evaluate if the response is relevant.',
        model: undefined,
        is_instructions_judge: true,
        isSessionLevelScorer: false,
      });
    });

    it('should create non-editable template scorer without instructions', () => {
      const formData: LLMScorerFormData & { scorerType: 'llm' } = {
        name: 'New Correctness Scorer',
        sampleRate: 60,
        scorerType: 'llm',
        llmTemplate: 'Correctness',
        instructions: 'These instructions should be ignored',
        isInstructionsJudge: false,
        model: '',
      };

      const result = convertFormDataToScheduledScorer(formData);

      expect(result).toEqual({
        name: 'New Correctness Scorer',
        sampleRate: 60,
        filterString: '',
        type: 'llm',
        llmTemplate: 'Correctness',
        guidelines: undefined,
        instructions: undefined,
        model: undefined,
        is_instructions_judge: false,
        isSessionLevelScorer: false,
      });
    });

    it('should create new Guidelines scorer with parsed guidelines', () => {
      const formData: LLMScorerFormData & { scorerType: 'llm' } = {
        name: 'New Guidelines Scorer',
        sampleRate: 60,
        filterString: '',
        scorerType: 'llm',
        llmTemplate: 'Guidelines',
        guidelines: '  Line 1  \n\n  Line 2  \n  \n  Line 3  ',
        model: '',
      };

      const result = convertFormDataToScheduledScorer(formData);

      expect(result).toEqual({
        name: 'New Guidelines Scorer',
        sampleRate: 60,
        filterString: '',
        type: 'llm',
        llmTemplate: 'Guidelines',
        guidelines: ['Line 1', 'Line 2', 'Line 3'],
        isSessionLevelScorer: false,
      });
    });

    it('should handle falsy guidelines values', () => {
      const formDataWithEmpty: LLMScorerFormData & { scorerType: 'llm' } = {
        name: 'New Guidelines Scorer',
        sampleRate: 50,
        scorerType: 'llm',
        llmTemplate: 'Guidelines',
        guidelines: '',
        model: '',
      };

      const formDataWithUndefined: LLMScorerFormData & { scorerType: 'llm' } = {
        name: 'New Guidelines Scorer',
        sampleRate: 50,
        scorerType: 'llm',
        llmTemplate: 'Guidelines',
        model: '',
        // guidelines intentionally undefined
      };

      expect((convertFormDataToScheduledScorer(formDataWithEmpty) as LLMScorer).guidelines).toEqual([]);
      expect((convertFormDataToScheduledScorer(formDataWithUndefined) as LLMScorer).guidelines).toEqual([]);
    });

    it('should handle falsy filterString values', () => {
      const formDataWithEmpty: LLMScorerFormData & { scorerType: 'llm' } = {
        name: 'New LLM Scorer',
        sampleRate: 50,
        filterString: '',
        scorerType: 'llm',
        llmTemplate: 'Safety',
        model: '',
      };

      const formDataWithUndefined: LLMScorerFormData & { scorerType: 'llm' } = {
        name: 'New LLM Scorer',
        sampleRate: 50,
        scorerType: 'llm',
        llmTemplate: 'Safety',
        model: '',
        // filterString intentionally undefined
      };

      expect(convertFormDataToScheduledScorer(formDataWithEmpty).filterString).toBe('');
      expect(convertFormDataToScheduledScorer(formDataWithUndefined).filterString).toBe('');
    });
  });

  describe('Updating existing scorers', () => {
    it('should update Custom template scorer with model', () => {
      const baseScorer: LLMScorer = {
        name: 'Original Instructions Scorer',
        sampleRate: 50,
        filterString: 'old_filter',
        type: 'llm',
        llmTemplate: 'Custom',
        instructions: 'Old instructions',
        model: 'databricks:/databricks-gpt-4',
        is_instructions_judge: true,
      };

      const formData: LLMScorerFormData & { scorerType: 'llm' } = {
        name: 'Updated Instructions Scorer',
        sampleRate: 80,
        filterString: 'new_filter',
        scorerType: 'llm',
        llmTemplate: 'Custom',
        instructions: 'New instructions',
        model: 'openai:/gpt-4o-mini',
        isInstructionsJudge: true,
      };

      const result = convertFormDataToScheduledScorer(formData, baseScorer);

      expect(result).toEqual({
        name: 'Updated Instructions Scorer',
        sampleRate: 80,
        filterString: 'new_filter',
        type: 'llm',
        llmTemplate: 'Custom',
        instructions: 'New instructions',
        model: 'openai:/gpt-4o-mini',
        is_instructions_judge: true,
      });
    });

    it('should update custom code scorer with only allowed fields', () => {
      const baseScorer: CustomCodeScorer = {
        name: 'Original Custom Scorer',
        sampleRate: 25,
        filterString: 'old_filter',
        type: 'custom-code',
        code: 'def original_scorer():\n    return False',
        callSignature: '',
        originalFuncName: '',
        isSessionLevelScorer: false,
      };

      const formData: CustomCodeScorerFormData & { scorerType: 'custom-code' } = {
        name: 'Updated Name', // This should be ignored for custom scorers
        sampleRate: 75,
        filterString: 'new_filter',
        scorerType: 'custom-code',
        code: 'def original_scorer():\n    return False',
      };

      const result = convertFormDataToScheduledScorer(formData, baseScorer);

      expect(result).toEqual({
        name: 'Original Custom Scorer', // Name unchanged
        sampleRate: 75, // Updated
        filterString: 'new_filter', // Updated
        type: 'custom-code',
        code: 'def original_scorer():\n    return False', // Code unchanged
        callSignature: '',
        originalFuncName: '',
        isSessionLevelScorer: false,
      });
    });

    it('should update LLM scorer with all form fields', () => {
      const baseScorer: LLMScorer = {
        name: 'Original LLM Scorer',
        sampleRate: 30,
        filterString: 'old_filter',
        type: 'llm',
        llmTemplate: 'Safety',
        is_instructions_judge: false,
      };

      const formData: LLMScorerFormData & { scorerType: 'llm' } = {
        name: 'Updated LLM Scorer',
        sampleRate: 70,
        filterString: 'new_filter',
        scorerType: 'llm',
        llmTemplate: 'Guidelines',
        guidelines: 'New guideline 1\nNew guideline 2',
        isInstructionsJudge: false,
        model: '',
      };

      const result = convertFormDataToScheduledScorer(formData, baseScorer);

      expect(result).toEqual({
        name: 'Updated LLM Scorer',
        sampleRate: 70,
        filterString: 'new_filter',
        type: 'llm',
        llmTemplate: 'Guidelines',
        guidelines: ['New guideline 1', 'New guideline 2'],
        is_instructions_judge: false,
      });
    });

    it('should update LLM scorer to non-Guidelines template', () => {
      const baseScorer: LLMScorer = {
        name: 'Original Guidelines Scorer',
        sampleRate: 40,
        type: 'llm',
        llmTemplate: 'Guidelines',
        guidelines: ['Old guideline 1', 'Old guideline 2'],
        is_instructions_judge: false,
      };

      const formData: LLMScorerFormData & { scorerType: 'llm' } = {
        name: 'Updated to Toxicity',
        sampleRate: 80,
        scorerType: 'llm',
        llmTemplate: 'Safety',
        isInstructionsJudge: false,
        model: '',
      };

      const result = convertFormDataToScheduledScorer(formData, baseScorer);

      expect(result).toEqual({
        name: 'Updated to Toxicity',
        sampleRate: 80,
        filterString: '',
        type: 'llm',
        llmTemplate: 'Safety',
        guidelines: ['Old guideline 1', 'Old guideline 2'], // Guidelines preserved from base
        is_instructions_judge: false,
      });
    });

    it('should throw error when updating without baseScorer', () => {
      const formData: CustomCodeScorerFormData & { scorerType: 'custom-code' } = {
        name: 'Test Scorer',
        sampleRate: 50,
        scorerType: 'custom-code',
        code: 'def original_scorer():\n    return False',
      };

      expect(() => convertFormDataToScheduledScorer(formData)).toThrow(ScorerTransformationError);
      expect(() => convertFormDataToScheduledScorer(formData)).toThrow('Base scorer is required for updates');
    });

    it('should handle empty filterString in form data', () => {
      const baseScorer: LLMScorer = {
        name: 'Test Scorer',
        sampleRate: 50,
        filterString: 'old_filter',
        type: 'llm',
        llmTemplate: 'Safety',
        isSessionLevelScorer: false,
      };

      const formData: LLMScorerFormData & { scorerType: 'llm' } = {
        name: 'Updated Scorer',
        sampleRate: 60,
        filterString: '',
        scorerType: 'llm',
        llmTemplate: 'Safety',
        model: '',
      };

      const result = convertFormDataToScheduledScorer(formData, baseScorer);

      expect(result.filterString).toBe('');
    });
  });

  describe('Edge cases', () => {
    it('should preserve base scorer properties not in form data', () => {
      const baseScorer: LLMScorer = {
        name: 'Test Scorer',
        sampleRate: 50,
        filterString: 'original_filter',
        type: 'llm',
        llmTemplate: 'Safety',
        isSessionLevelScorer: false,
      };

      const formData: Partial<LLMScorerFormData> & { scorerType: 'llm' } = {
        name: 'Updated Name',
        scorerType: 'llm',
        llmTemplate: 'Safety',
        // sampleRate and filterString intentionally missing
      };

      const result = convertFormDataToScheduledScorer(formData as any, baseScorer);

      expect(result).toEqual({
        name: 'Updated Name',
        sampleRate: undefined,
        filterString: '',
        type: 'llm',
        llmTemplate: 'Safety',
        isSessionLevelScorer: false,
      });
    });
  });
});

describe('convertRegisterScorerResponseToConfig', () => {
  it('should convert custom instructions-based scorer response to ScorerConfig', () => {
    const response: RegisterScorerResponse = {
      version: 2,
      scorer_id: 'scorer_456',
      experiment_id: '789',
      name: 'Custom Instructions Scorer',
      serialized_scorer: JSON.stringify({
        instructions_judge_pydantic_data: {
          instructions: 'Evaluate the response quality',
          model: 'openai:/gpt-4o-mini',
        },
      }),
      creation_time: 1234567890000,
    };

    const result = convertRegisterScorerResponseToConfig(response);

    expect(result).toEqual({
      name: 'Custom Instructions Scorer',
      serialized_scorer: response.serialized_scorer,
      scorer_version: 2,
    });
  });
});
