import { describe, test, expect } from '@jest/globals';
import { formDataToModelConfig, modelConfigToFormData, validateModelConfig, getModelConfigFromTags } from './utils';
import type { PromptModelConfig, PromptModelConfigFormData } from './types';

describe('Model Config Utils', () => {
  describe('formDataToModelConfig', () => {
    test('converts all fields correctly', () => {
      const formData: PromptModelConfigFormData = {
        provider: 'openai',
        modelName: 'gpt-4',
        temperature: '0.7',
        maxTokens: '2048',
        topP: '0.9',
        topK: '40',
        frequencyPenalty: '0.5',
        presencePenalty: '0.3',
        stopSequences: '\\n\\n, END, ###',
      };

      const result = formDataToModelConfig(formData);

      expect(result).toEqual({
        provider: 'openai',
        model_name: 'gpt-4',
        temperature: 0.7,
        max_tokens: 2048,
        top_p: 0.9,
        top_k: 40,
        frequency_penalty: 0.5,
        presence_penalty: 0.3,
        stop_sequences: ['\\n\\n', 'END', '###'],
      });
    });

    test('returns undefined for empty form data', () => {
      const formData: PromptModelConfigFormData = {};
      expect(formDataToModelConfig(formData)).toBeUndefined();
    });

    test('handles partial data', () => {
      const formData: PromptModelConfigFormData = {
        modelName: 'claude-3',
        temperature: '1.0',
      };

      const result = formDataToModelConfig(formData);

      expect(result).toEqual({
        model_name: 'claude-3',
        temperature: 1.0,
      });
    });

    test('filters out invalid number strings', () => {
      const formData: PromptModelConfigFormData = {
        temperature: 'abc',
        maxTokens: 'xyz',
      };

      expect(formDataToModelConfig(formData)).toBeUndefined();
    });

    test('trims whitespace from strings', () => {
      const formData: PromptModelConfigFormData = {
        modelName: '  gpt-4  ',
        stopSequences: '  \\n\\n  ,  END  ',
      };

      const result = formDataToModelConfig(formData);

      expect(result).toEqual({
        model_name: 'gpt-4',
        stop_sequences: ['\\n\\n', 'END'],
      });
    });
  });

  describe('modelConfigToFormData', () => {
    test('converts all fields correctly', () => {
      const config: PromptModelConfig = {
        provider: 'openai',
        model_name: 'gpt-4',
        temperature: 0.7,
        max_tokens: 2048,
        top_p: 0.9,
        top_k: 40,
        frequency_penalty: 0.5,
        presence_penalty: 0.3,
        stop_sequences: ['\\n\\n', 'END', '###'],
      };

      const result = modelConfigToFormData(config);

      expect(result).toEqual({
        provider: 'openai',
        modelName: 'gpt-4',
        temperature: '0.7',
        maxTokens: '2048',
        topP: '0.9',
        topK: '40',
        frequencyPenalty: '0.5',
        presencePenalty: '0.3',
        stopSequences: '\\n\\n, END, ###',
      });
    });

    test('returns empty object for undefined config', () => {
      expect(modelConfigToFormData(undefined)).toEqual({});
    });

    test('handles partial data', () => {
      const config: PromptModelConfig = {
        model_name: 'claude-3',
        temperature: 1.0,
      };

      const result = modelConfigToFormData(config);

      expect(result).toEqual({
        provider: '',
        modelName: 'claude-3',
        temperature: '1',
        maxTokens: '',
        topP: '',
        topK: '',
        frequencyPenalty: '',
        presencePenalty: '',
        stopSequences: '',
      });
    });
  });

  describe('validateModelConfig', () => {
    test('returns no errors for valid data', () => {
      const formData: PromptModelConfigFormData = {
        temperature: '0.7',
        maxTokens: '2048',
        topP: '0.9',
        topK: '40',
        frequencyPenalty: '0.5',
        presencePenalty: '0.3',
      };

      expect(validateModelConfig(formData)).toEqual({});
    });

    test('validates temperature range', () => {
      expect(validateModelConfig({ temperature: '-1' })).toHaveProperty('temperature');
      expect(validateModelConfig({ temperature: 'abc' })).toHaveProperty('temperature');
      expect(validateModelConfig({ temperature: '0' })).toEqual({});
    });

    test('validates maxTokens range', () => {
      expect(validateModelConfig({ maxTokens: '0' })).toHaveProperty('maxTokens');
      expect(validateModelConfig({ maxTokens: '-1' })).toHaveProperty('maxTokens');
      expect(validateModelConfig({ maxTokens: 'abc' })).toHaveProperty('maxTokens');
      expect(validateModelConfig({ maxTokens: '1' })).toEqual({});
    });

    test('validates topP range', () => {
      expect(validateModelConfig({ topP: '-0.1' })).toHaveProperty('topP');
      expect(validateModelConfig({ topP: '1.1' })).toHaveProperty('topP');
      expect(validateModelConfig({ topP: '0.5' })).toEqual({});
    });

    test('validates topK range', () => {
      expect(validateModelConfig({ topK: '0' })).toHaveProperty('topK');
      expect(validateModelConfig({ topK: '-1' })).toHaveProperty('topK');
      expect(validateModelConfig({ topK: '1' })).toEqual({});
    });

    test('validates frequency and presence penalty range', () => {
      expect(validateModelConfig({ frequencyPenalty: '-2.1' })).toHaveProperty('frequencyPenalty');
      expect(validateModelConfig({ frequencyPenalty: '2.1' })).toHaveProperty('frequencyPenalty');
      expect(validateModelConfig({ presencePenalty: '-2.1' })).toHaveProperty('presencePenalty');
      expect(validateModelConfig({ presencePenalty: '2.1' })).toHaveProperty('presencePenalty');
      expect(validateModelConfig({ frequencyPenalty: '0' })).toEqual({});
    });
  });

  describe('getModelConfigFromTags', () => {
    test('parses valid JSON tag', () => {
      const tags = [{ key: '_mlflow_prompt_model_config', value: '{"model_name":"gpt-4","temperature":0.7}' }];

      const result = getModelConfigFromTags(tags);

      expect(result).toEqual({
        model_name: 'gpt-4',
        temperature: 0.7,
      });
    });

    test('returns undefined for missing tag', () => {
      const tags = [{ key: 'other.tag', value: 'value' }];
      expect(getModelConfigFromTags(tags)).toBeUndefined();
    });

    test('returns undefined for invalid JSON', () => {
      const tags = [{ key: '_mlflow_prompt_model_config', value: 'not-json' }];
      expect(getModelConfigFromTags(tags)).toBeUndefined();
    });

    test('returns undefined for empty tags array', () => {
      expect(getModelConfigFromTags([])).toBeUndefined();
    });

    test('returns undefined for undefined tags', () => {
      expect(getModelConfigFromTags(undefined)).toBeUndefined();
    });
  });
});
