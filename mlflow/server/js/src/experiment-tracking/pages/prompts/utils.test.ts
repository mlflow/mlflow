import { describe, test, expect } from '@jest/globals';
import {
  formDataToModelConfig,
  modelConfigToFormData,
  validateModelConfig,
  getModelConfigFromTags,
  getResponseFormatFromTags,
  validateResponseFormatJson,
  canEditSchemaVisually,
  jsonSchemaToProperties,
  propertiesToJsonSchema,
} from './utils';
import type { EditableSchemaProperty, PromptModelConfig, PromptModelConfigFormData, SchemaProperty } from './types';

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

    test('keeps properties that a partial propertyOrdering leaves out, sorted after the listed ones', () => {
      const partiallyOrdered = JSON.stringify({
        type: 'object',
        properties: { name: { type: 'string' }, tags: { type: 'string' }, priority: { type: 'string' } },
        propertyOrdering: ['priority'],
      });

      expect(jsonSchemaToProperties(partiallyOrdered).map((p) => p.name)).toEqual(['priority', 'name', 'tags']);
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

    test('marks definitions the visual editor cannot represent as read-only rows', () => {
      const withUnsupportedTypes = JSON.stringify({
        type: 'object',
        properties: {
          meta: { type: 'object', properties: { id: { type: 'string' } } },
          ref: { $ref: '#/$defs/Address' },
          either: { oneOf: [{ type: 'string' }, { type: 'number' }] },
          objects: { type: 'array', items: { type: 'object' } },
          untyped: { type: 'array' },
          count: { type: 'integer' },
        },
      });

      const properties = jsonSchemaToProperties(withUnsupportedTypes);

      expect(properties.map((p) => [p.name, p.kind === 'readOnly' ? p.label : undefined])).toEqual([
        ['meta', 'object'],
        ['ref', '$ref'],
        ['either', 'oneOf'],
        ['objects', 'array'],
        ['untyped', 'array'],
        ['count', undefined],
      ]);
    });

    test('records the JSON type of enum values so they are not coerced to strings', () => {
      const withTypedEnums = JSON.stringify({
        type: 'object',
        properties: {
          level: { type: 'integer', enum: [1, 2, 3] },
          flag: { enum: [true, false] },
          name: { type: 'string', enum: ['a', 'b'] },
        },
      });

      const properties = jsonSchemaToProperties(withTypedEnums) as EditableSchemaProperty[];

      expect(properties.map((p) => [p.name, p.type, p.enumValueType, p.enumValues])).toEqual([
        ['level', 'enum', 'integer', ['1', '2', '3']],
        ['flag', 'enum', 'boolean', ['true', 'false']],
        ['name', 'enum', 'string', ['a', 'b']],
      ]);
    });
  });

  describe('propertiesToJsonSchema', () => {
    test('skips properties with an empty name', () => {
      const properties: SchemaProperty[] = [
        { kind: 'editable', id: '1', name: '', type: 'string', isArray: false, required: false, enumValues: [] },
        { kind: 'editable', id: '2', name: 'valid', type: 'string', isArray: false, required: false, enumValues: [] },
      ];

      const result = JSON.parse(propertiesToJsonSchema(properties));

      expect(result.properties).toEqual({ valid: { type: 'string' } });
    });

    test('filters out empty enum values', () => {
      const properties: SchemaProperty[] = [
        {
          kind: 'editable',
          id: '1',
          name: 'priority',
          type: 'enum',
          isArray: false,
          required: false,
          enumValues: ['low', 'medium', ''],
        },
      ];

      const result = JSON.parse(propertiesToJsonSchema(properties));

      expect(result.properties.priority).toEqual({ type: 'string', enum: ['low', 'medium'] });
    });

    test('deduplicates enum values while preserving order', () => {
      const properties: SchemaProperty[] = [
        {
          kind: 'editable',
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
        {
          kind: 'editable',
          id: '1',
          name: 'priority',
          type: 'enum',
          isArray: false,
          required: false,
          enumValues: ['', '  '],
        },
      ];

      const result = JSON.parse(propertiesToJsonSchema(properties));

      expect(result.properties.priority).toEqual({ type: 'string' });
    });

    test('emits an array of enum values as an array with enum on the items', () => {
      const properties: SchemaProperty[] = [
        {
          kind: 'editable',
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

    test('does not repeat a name in `required` when two rows share it', () => {
      const properties: SchemaProperty[] = [
        { kind: 'editable', id: '1', name: 'name', type: 'string', isArray: false, required: true, enumValues: [] },
        { kind: 'editable', id: '2', name: 'name', type: 'integer', isArray: false, required: true, enumValues: [] },
      ];

      const result = JSON.parse(propertiesToJsonSchema(properties));

      expect(result.properties).toEqual({ name: { type: 'integer' } });
      expect(result.required).toEqual(['name']);
    });

    test('omits the required key when no property is marked required', () => {
      const properties: SchemaProperty[] = [
        { kind: 'editable', id: '1', name: 'name', type: 'string', isArray: false, required: false, enumValues: [] },
      ];

      const result = JSON.parse(propertiesToJsonSchema(properties));

      expect(result).not.toHaveProperty('required');
    });
  });

  describe('propertiesToJsonSchema merged onto the source schema', () => {
    const richSchema = {
      $schema: 'https://json-schema.org/draft/2020-12/schema',
      type: 'object',
      title: 'Ticket',
      additionalProperties: false,
      properties: {
        summary: { type: 'string', description: 'One line summary', maxLength: 120 },
        level: { type: 'integer', enum: [1, 2, 3], description: 'Severity' },
        owner: { $ref: '#/$defs/Person' },
        meta: { type: 'object', properties: { source: { type: 'string' } } },
      },
      required: ['summary'],
      $defs: { Person: { type: 'object', properties: { name: { type: 'string' } } } },
    };

    // What the editor emits after a no-op pass over the source schema, i.e. what a single edit to
    // any one field does to all the others.
    const reserialize = (
      schema: Record<string, unknown>,
      edit: (properties: SchemaProperty[]) => SchemaProperty[] = (p) => p,
    ) => JSON.parse(propertiesToJsonSchema(edit(jsonSchemaToProperties(JSON.stringify(schema))), schema));

    test('preserves root-level keywords the editor does not model', () => {
      const result = reserialize(richSchema);

      expect(result.$schema).toEqual(richSchema.$schema);
      expect(result.title).toEqual('Ticket');
      expect(result.additionalProperties).toBe(false);
      expect(result.$defs).toEqual(richSchema.$defs);
    });

    test('preserves per-property keywords the editor does not model', () => {
      const result = reserialize(richSchema);

      expect(result.properties.summary).toEqual({ type: 'string', description: 'One line summary', maxLength: 120 });
    });

    test('preserves the JSON type of a numeric enum instead of stringifying it', () => {
      const result = reserialize(richSchema);

      expect(result.properties.level).toEqual({ type: 'integer', enum: [1, 2, 3], description: 'Severity' });
    });

    test('emits definitions the editor cannot represent verbatim', () => {
      const result = reserialize(richSchema);

      expect(result.properties.owner).toEqual({ $ref: '#/$defs/Person' });
      expect(result.properties.meta).toEqual(richSchema.properties.meta);
    });

    test('keeps unmodeled keywords when a property is renamed', () => {
      const result = reserialize(richSchema, (properties) =>
        properties.map((p) => (p.name === 'summary' ? { ...p, name: 'headline' } : p)),
      );

      expect(result.properties).not.toHaveProperty('summary');
      expect(result.properties.headline).toEqual({ type: 'string', description: 'One line summary', maxLength: 120 });
      expect(result.required).toEqual(['headline']);
    });

    test('does not resurrect the keywords of a deleted property when a new one reuses its name', () => {
      const result = reserialize(richSchema, (properties) => [
        ...properties.filter((p) => p.name !== 'summary'),
        {
          kind: 'editable',
          id: 'new',
          name: 'summary',
          type: 'string',
          isArray: false,
          required: false,
          enumValues: [],
        },
      ]);

      expect(result.properties.summary).toEqual({ type: 'string' });
    });

    test.each([
      ['a value is not a number', ['1', 'urgent']],
      ['a value is not written as a JSON number', ['1', '0x10']],
    ])('falls back to a string enum when %s', (_label, enumValues) => {
      const result = reserialize(richSchema, (properties) =>
        properties.map((p) => (p.name === 'level' ? { ...p, enumValues } : p)),
      );

      expect(result.properties.level).toEqual({ type: 'string', enum: enumValues, description: 'Severity' });
    });

    test('moves value constraints into `items` when a scalar becomes an array, keeping annotations', () => {
      const scalarSchema = {
        type: 'object',
        properties: {
          name: {
            type: 'string',
            description: 'Name',
            deprecated: true,
            $comment: 'legacy',
            maxLength: 120,
            default: 'n/a',
          },
        },
      };

      const result = reserialize(scalarSchema, (properties) => properties.map((p) => ({ ...p, isArray: true })));

      expect(result.properties.name).toEqual({
        type: 'array',
        description: 'Name',
        deprecated: true,
        $comment: 'legacy',
        items: { type: 'string', maxLength: 120, default: 'n/a' },
      });
    });

    test('folds the array container keywords back onto the value when an array becomes a scalar', () => {
      const arraySchema = {
        type: 'object',
        properties: {
          tags: {
            type: 'array',
            description: 'Labels',
            minItems: 1,
            default: ['a'],
            items: { type: 'string', pattern: '^[a-z]+$', default: 'a' },
          },
        },
      };

      const result = reserialize(arraySchema, (properties) => properties.map((p) => ({ ...p, isArray: false })));

      // `minItems` and the array-valued `default` only mean something on an array, so they go with it
      expect(result.properties.tags).toEqual({
        type: 'string',
        description: 'Labels',
        pattern: '^[a-z]+$',
        default: 'a',
      });
    });

    test('drops a property removed in the editor, along with its `required` entry', () => {
      const result = reserialize(richSchema, (properties) => properties.filter((p) => p.name !== 'summary'));

      expect(Object.keys(result.properties)).toEqual(['level', 'owner', 'meta']);
      expect(result).not.toHaveProperty('required');
    });

    test('keeps `propertyOrdering` in sync when the source schema uses it, and never adds it', () => {
      const ordered = {
        type: 'object',
        properties: { a: { type: 'string' }, b: { type: 'string' } },
        propertyOrdering: ['b', 'a'],
      };

      const result = reserialize(ordered, (properties) => properties.filter((p) => p.name !== 'a'));

      expect(result.propertyOrdering).toEqual(['b']);
      expect(reserialize(richSchema)).not.toHaveProperty('propertyOrdering');
    });

    test('emits untouched properties verbatim when a sibling is edited', () => {
      const mixed = {
        type: 'object',
        properties: {
          code: { enum: ['a', 1] },
          note: { type: 'string' },
        },
      };

      const result = reserialize(mixed, (properties) =>
        properties.map((p) => (p.name === 'note' ? { ...p, required: true } : p)),
      );

      expect(result.properties.code).toEqual({ enum: ['a', 1] });
      expect(result.required).toEqual(['note']);
    });

    test('drops constraints and values of the old type when a property changes type', () => {
      const typed = {
        type: 'object',
        properties: {
          code: {
            type: 'string',
            description: 'Code',
            pattern: '^[A-Z]+$',
            minLength: 2,
            format: 'uuid',
            default: 'AB',
            examples: ['XY'],
          },
          count: { type: 'integer', minimum: 0, format: 'int64' },
        },
      };

      const result = reserialize(typed, (properties) =>
        properties.map((p) => {
          if (p.kind !== 'editable') return p;
          return { ...p, type: p.name === 'code' ? 'integer' : 'number' };
        }),
      );

      expect(result.properties.code).toEqual({ type: 'integer', description: 'Code' });
      expect(result.properties.count).toEqual({ type: 'number', minimum: 0, format: 'int64' });
    });

    test('drops the array-valued `default` when the item type changes', () => {
      const withDefault = {
        type: 'object',
        properties: { tags: { type: 'array', minItems: 1, default: ['a'], items: { type: 'string' } } },
      };

      const result = reserialize(withDefault, (properties) =>
        properties.map((p) => (p.kind === 'editable' ? { ...p, type: 'integer' } : p)),
      );

      expect(result.properties.tags).toEqual({ type: 'array', minItems: 1, items: { type: 'integer' } });
    });

    test('drops `default` / `examples` that an edited enum no longer allows', () => {
      const withValues = {
        type: 'object',
        properties: {
          level: { type: 'string', enum: ['low', 'high'], default: 'low', examples: ['high'] },
          tags: { type: 'array', default: [1], items: { type: 'integer', enum: [1, 2] } },
        },
      };

      const result = reserialize(withValues, (properties) =>
        properties.map((p) => {
          if (p.kind !== 'editable') return p;
          return { ...p, enumValues: p.name === 'level' ? ['low', 'medium'] : ['2', '3'] };
        }),
      );

      expect(result.properties.level).toEqual({ type: 'string', enum: ['low', 'medium'], default: 'low' });
      expect(result.properties.tags).toEqual({ type: 'array', items: { type: 'integer', enum: [2, 3] } });
    });

    test('ignores a base schema that is not an object', () => {
      const properties: SchemaProperty[] = [
        { kind: 'editable', id: '1', name: 'name', type: 'string', isArray: false, required: false, enumValues: [] },
      ];

      const result = JSON.parse(propertiesToJsonSchema(properties, [1, 2]));

      expect(result).toEqual({ type: 'object', properties: { name: { type: 'string' } } });
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

  describe('canEditSchemaVisually', () => {
    test.each([
      ['an empty value', ''],
      ['an empty object', '{}'],
      ['an object schema with properties', '{"type":"object","properties":{"a":{"type":"string"}}}'],
      ['an object schema without properties yet', '{"type":"object"}'],
      ['a properties map without an explicit type', '{"properties":{"a":{"type":"string"}}}'],
    ])('accepts %s', (_label, schemaText) => {
      expect(canEditSchemaVisually(schemaText)).toBe(true);
    });

    test.each([
      ['invalid JSON', '{ broken'],
      ['a non-object', '[1, 2]'],
      ['an OpenAI response_format wrapper', '{"type":"json_schema","json_schema":{"schema":{"type":"object"}}}'],
      ['an array schema', '{"type":"array","items":{"type":"string"}}'],
      ['a $ref at the root', '{"$ref":"#/$defs/Person"}'],
      ['a combinator at the root', '{"anyOf":[{"type":"object"}]}'],
    ])('rejects %s', (_label, schemaText) => {
      expect(canEditSchemaVisually(schemaText)).toBe(false);
    });
  });
});
