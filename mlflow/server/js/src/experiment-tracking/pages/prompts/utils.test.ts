import { describe, test, expect } from '@jest/globals';
import {
  formDataToModelConfig,
  modelConfigToFormData,
  validateModelConfig,
  getModelConfigFromTags,
  getResponseFormatFromTags,
  validateResponseFormatJson,
  jsonSchemaToProperties,
  propertiesToJsonSchema,
} from './utils';
import type { PromptModelConfig, PromptModelConfigFormData, SchemaProperty } from './types';

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

describe('Response format (structured output) utils', () => {
  describe('getResponseFormatFromTags', () => {
    test('returns raw string value for existing tag', () => {
      const rawValue = '{"type":"object","properties":{"result":{"type":"string"}},"additionalProperties":false}';
      const tags = [{ key: '_mlflow_prompt_response_format', value: rawValue }];

      expect(getResponseFormatFromTags(tags)).toBe(rawValue);
    });

    test('returns undefined for missing tag', () => {
      const tags = [{ key: 'other.tag', value: 'value' }];
      expect(getResponseFormatFromTags(tags)).toBeUndefined();
    });

    test('returns raw string even for invalid JSON (no parsing)', () => {
      const tags = [{ key: '_mlflow_prompt_response_format', value: 'not-json' }];
      expect(getResponseFormatFromTags(tags)).toBe('not-json');
    });

    test('returns undefined for empty tags array', () => {
      expect(getResponseFormatFromTags([])).toBeUndefined();
    });

    test('returns undefined for undefined tags', () => {
      expect(getResponseFormatFromTags(undefined)).toBeUndefined();
    });
  });

  describe('validateResponseFormatJson', () => {
    test('returns valid for empty or whitespace-only string', () => {
      expect(validateResponseFormatJson('')).toEqual({ valid: true });
      expect(validateResponseFormatJson('   ')).toEqual({ valid: true });
    });

    test('returns valid for valid JSON object', () => {
      expect(validateResponseFormatJson('{"type":"object","properties":{},"additionalProperties":false}')).toEqual({
        valid: true,
      });
    });

    test('returns invalid for invalid JSON', () => {
      const result = validateResponseFormatJson('{ invalid }');
      expect(result.valid).toBe(false);
      expect(result.error).toBeDefined();
    });

    test('returns invalid when JSON parses to non-object', () => {
      expect(validateResponseFormatJson('[]')).toEqual({
        valid: false,
        error: 'Structured output must be a JSON object (e.g. a JSON schema).',
      });
      expect(validateResponseFormatJson('123')).toEqual({
        valid: false,
        error: 'Structured output must be a JSON object (e.g. a JSON schema).',
      });
    });
  });
});

describe('Schema builder utils', () => {
  const schema = {
    type: 'object',
    properties: {
      name: { type: 'string' },
      tags: { type: 'array', items: { type: 'string' } },
      priority: { type: 'string', enum: ['low', 'medium', 'high'] },
    },
    required: ['name'],
  };

  describe('jsonSchemaToProperties', () => {
    test('returns an empty array for an empty string', () => {
      expect(jsonSchemaToProperties('')).toEqual([]);
    });

    test('returns an empty array for invalid JSON without throwing', () => {
      expect(() => jsonSchemaToProperties('{ broken')).not.toThrow();
      expect(jsonSchemaToProperties('{ broken')).toEqual([]);
    });

    test('preserves property ordering from propertyOrdering when present, for backwards compatibility', () => {
      const schemaWithCustomOrdering = {
        type: 'object',
        properties: {
          name: { type: 'string' },
          tags: { type: 'array', items: { type: 'string' } },
          priority: { type: 'string', enum: ['low', 'medium', 'high'] },
        },
        propertyOrdering: ['priority', 'name', 'tags'],
        required: ['name'],
      };

      const properties = jsonSchemaToProperties(JSON.stringify(schemaWithCustomOrdering));
      expect(properties.map((p) => p.name)).toEqual(['priority', 'name', 'tags']);
    });

    test('ignores `required` entries with no matching property', () => {
      const withGhost = JSON.stringify({
        type: 'object',
        properties: { name: { type: 'string' } },
        required: ['name', 'ghost'],
      });

      const result = JSON.parse(propertiesToJsonSchema(jsonSchemaToProperties(withGhost)));

      expect(result.required).toEqual(['name']);
    });

    test('falls back to string for types the visual editor does not support', () => {
      const withUnsupportedType = JSON.stringify({
        type: 'object',
        properties: { meta: { type: 'object' }, count: { type: 'integer' } },
      });

      const properties = jsonSchemaToProperties(withUnsupportedType);

      expect(properties.map((p) => [p.name, p.type])).toEqual([
        ['meta', 'string'],
        ['count', 'integer'],
      ]);
    });
  });

  describe('propertiesToJsonSchema', () => {
    test('skips properties with an empty name', () => {
      const properties: SchemaProperty[] = [
        { id: '1', name: '', type: 'string', isArray: false, required: false, enumValues: [] },
        { id: '2', name: 'valid', type: 'string', isArray: false, required: false, enumValues: [] },
      ];

      const result = JSON.parse(propertiesToJsonSchema(properties));

      expect(result.properties).toEqual({ valid: { type: 'string' } });
    });

    test('filters out empty enum values', () => {
      const properties: SchemaProperty[] = [
        { id: '1', name: 'priority', type: 'enum', isArray: false, required: false, enumValues: ['low', 'medium', ''] },
      ];

      const result = JSON.parse(propertiesToJsonSchema(properties));

      expect(result.properties.priority).toEqual({ type: 'string', enum: ['low', 'medium'] });
    });

    test('deduplicates enum values while preserving order', () => {
      const properties: SchemaProperty[] = [
        {
          id: '1',
          name: 'priority',
          type: 'enum',
          isArray: false,
          required: false,
          enumValues: ['low', 'medium', 'low', 'high', 'medium'],
        },
      ];

      const result = JSON.parse(propertiesToJsonSchema(properties));

      expect(result.properties.priority).toEqual({ type: 'string', enum: ['low', 'medium', 'high'] });
    });

    test('omits the enum keyword when no non-empty values remain', () => {
      const properties: SchemaProperty[] = [
        { id: '1', name: 'priority', type: 'enum', isArray: false, required: false, enumValues: ['', '  '] },
      ];

      const result = JSON.parse(propertiesToJsonSchema(properties));

      expect(result.properties.priority).toEqual({ type: 'string' });
    });

    test('emits an array of enum values as an array with enum on the items', () => {
      const properties: SchemaProperty[] = [
        {
          id: '1',
          name: 'tags',
          type: 'enum',
          isArray: true,
          required: false,
          enumValues: ['low', 'medium', 'high'],
        },
      ];

      const result = JSON.parse(propertiesToJsonSchema(properties));

      expect(result.properties.tags).toEqual({
        type: 'array',
        items: { type: 'string', enum: ['low', 'medium', 'high'] },
      });
    });

    test('omits the required key when no property is marked required', () => {
      const properties: SchemaProperty[] = [
        { id: '1', name: 'name', type: 'string', isArray: false, required: false, enumValues: [] },
      ];

      const result = JSON.parse(propertiesToJsonSchema(properties));

      expect(result).not.toHaveProperty('required');
    });
  });

  describe('jsonSchemaToProperties / propertiesToJsonSchema round-trip', () => {
    test('produces an equivalent schema', () => {
      const properties = jsonSchemaToProperties(JSON.stringify(schema));
      const result = JSON.parse(propertiesToJsonSchema(properties));

      expect(result).toEqual(schema);
    });

    test('preserves field order', () => {
      const properties = jsonSchemaToProperties(JSON.stringify(schema));
      const result = JSON.parse(propertiesToJsonSchema(properties));

      expect(Object.keys(result.properties)).toEqual(['name', 'tags', 'priority']);
    });
  });
});
