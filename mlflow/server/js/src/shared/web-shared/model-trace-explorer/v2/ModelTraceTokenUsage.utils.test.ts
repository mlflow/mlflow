import { describe, expect, it } from '@jest/globals';

import { getSpanTokenUsage } from './ModelTraceTokenUsage.utils';

describe('getSpanTokenUsage', () => {
  it('returns the precomputed span.tokenUsage as-is when present', () => {
    const tokenUsage = { input_tokens: 1, output_tokens: 2, total_tokens: 3 };

    expect(
      getSpanTokenUsage({
        tokenUsage,
        attributes: { 'gen_ai.usage.cache_read_input_tokens': 7 },
      }),
    ).toEqual(tokenUsage);
  });

  it('returns undefined when no token usage source is present', () => {
    expect(getSpanTokenUsage({ attributes: {} })).toBeUndefined();
  });

  it('fills cache token fields from OTEL attributes when the mlflow blob omits them', () => {
    const span = {
      attributes: {
        'mlflow.chat.tokenUsage': JSON.stringify({ input_tokens: 10, output_tokens: 5, total_tokens: 20 }),
        'gen_ai.usage.input_tokens': 10,
        'gen_ai.usage.output_tokens': 5,
        'gen_ai.usage.cache_read_input_tokens': 7,
        'gen_ai.usage.cache_creation_input_tokens': 3,
      },
    };

    expect(getSpanTokenUsage(span)).toEqual({
      input_tokens: 10,
      output_tokens: 5,
      // total_tokens comes from the blob (20), not input + output from OTEL (15)
      total_tokens: 20,
      cache_read_input_tokens: 7,
      cache_creation_input_tokens: 3,
    });
  });

  it('reads cache token fields from raw OTEL attributes when no mlflow blob is present', () => {
    const span = {
      attributes: {
        'gen_ai.usage.input_tokens': 10,
        'gen_ai.usage.output_tokens': 5,
        'gen_ai.usage.cache_read_input_tokens': 7,
        'gen_ai.usage.cache_creation_input_tokens': 3,
      },
    };

    expect(getSpanTokenUsage(span)).toEqual({
      input_tokens: 10,
      output_tokens: 5,
      total_tokens: 15,
      cache_read_input_tokens: 7,
      cache_creation_input_tokens: 3,
    });
  });

  it('reads cache token values delivered as double_value in V4 array-form attributes', () => {
    const span = {
      attributes: [
        { key: 'gen_ai.usage.input_tokens', value: { int_value: 10 } },
        { key: 'gen_ai.usage.output_tokens', value: { int_value: 5 } },
        { key: 'gen_ai.usage.cache_read_input_tokens', value: { double_value: 7.5 } },
        { key: 'gen_ai.usage.cache_creation_input_tokens', value: { double_value: 3 } },
      ],
    };

    expect(getSpanTokenUsage(span)).toEqual({
      input_tokens: 10,
      output_tokens: 5,
      total_tokens: 15,
      cache_read_input_tokens: 7.5,
      cache_creation_input_tokens: 3,
    });
  });

  it('returns the mlflow blob values unchanged when no OTEL attributes are present', () => {
    const span = {
      attributes: {
        'mlflow.chat.tokenUsage': JSON.stringify({ input_tokens: 10, output_tokens: 5, total_tokens: 15 }),
      },
    };

    expect(getSpanTokenUsage(span)).toEqual({
      input_tokens: 10,
      output_tokens: 5,
      total_tokens: 15,
    });
  });

  it('keeps cache token fields from the mlflow blob when both sources provide them', () => {
    const span = {
      attributes: {
        'mlflow.chat.tokenUsage': JSON.stringify({
          input_tokens: 10,
          output_tokens: 5,
          total_tokens: 15,
          cache_read_input_tokens: 100,
        }),
        'gen_ai.usage.input_tokens': 10,
        'gen_ai.usage.output_tokens': 5,
        'gen_ai.usage.cache_read_input_tokens': 7,
        'gen_ai.usage.cache_creation_input_tokens': 3,
      },
    };

    expect(getSpanTokenUsage(span)).toEqual({
      input_tokens: 10,
      output_tokens: 5,
      total_tokens: 15,
      cache_read_input_tokens: 100,
      cache_creation_input_tokens: 3,
    });
  });

  it('preserves a zero cache token count delivered as double_value in V4 array-form attributes', () => {
    const span = {
      attributes: [
        {
          key: 'mlflow.chat.tokenUsage',
          value: { string_value: JSON.stringify({ input_tokens: 10, output_tokens: 5, total_tokens: 15 }) },
        },
        { key: 'gen_ai.usage.input_tokens', value: { int_value: 10 } },
        { key: 'gen_ai.usage.output_tokens', value: { int_value: 5 } },
        { key: 'gen_ai.usage.cache_read_input_tokens', value: { double_value: 0 } },
      ],
    };

    expect(getSpanTokenUsage(span)).toEqual({
      input_tokens: 10,
      output_tokens: 5,
      total_tokens: 15,
      cache_read_input_tokens: 0,
    });
  });

  it('fills cache token fields from OTEL attributes even when OTEL input/output are absent', () => {
    const span = {
      attributes: {
        'mlflow.chat.tokenUsage': JSON.stringify({ input_tokens: 100, output_tokens: 50, total_tokens: 150 }),
        'gen_ai.usage.cache_read_input_tokens': 30,
        'gen_ai.usage.cache_creation_input_tokens': 20,
      },
    };

    expect(getSpanTokenUsage(span)).toEqual({
      input_tokens: 100,
      output_tokens: 50,
      total_tokens: 150,
      cache_read_input_tokens: 30,
      cache_creation_input_tokens: 20,
    });
  });

  it('fills cache token fields from V4 array-form attributes even when OTEL input/output are absent', () => {
    const span = {
      attributes: [
        {
          key: 'mlflow.chat.tokenUsage',
          value: { string_value: JSON.stringify({ input_tokens: 100, output_tokens: 50, total_tokens: 150 }) },
        },
        { key: 'gen_ai.usage.cache_read_input_tokens', value: { int_value: 30 } },
        { key: 'gen_ai.usage.cache_creation_input_tokens', value: { double_value: 20 } },
      ],
    };

    expect(getSpanTokenUsage(span)).toEqual({
      input_tokens: 100,
      output_tokens: 50,
      total_tokens: 150,
      cache_read_input_tokens: 30,
      cache_creation_input_tokens: 20,
    });
  });

  it('keeps the blob-derived total when the blob omits total_tokens and OTEL implies a different one', () => {
    const span = {
      attributes: {
        'mlflow.chat.tokenUsage': JSON.stringify({ input_tokens: 10, output_tokens: 5 }),
        'gen_ai.usage.input_tokens': 8,
        'gen_ai.usage.output_tokens': 4,
        'gen_ai.usage.cache_read_input_tokens': 7,
      },
    };

    expect(getSpanTokenUsage(span)).toEqual({
      input_tokens: 10,
      output_tokens: 5,
      // derived from the blob (10 + 5), not from the OTEL attributes (8 + 4)
      total_tokens: 15,
      cache_read_input_tokens: 7,
    });
  });

  // @ui-test-skill-generated
  describe('dotted OTEL cache attribute spelling', () => {
    it('reads cache tokens from the dotted keys in record-form attributes', () => {
      const span = {
        attributes: {
          'gen_ai.usage.input_tokens': 10,
          'gen_ai.usage.output_tokens': 5,
          'gen_ai.usage.cache_read.input_tokens': 7,
          'gen_ai.usage.cache_creation.input_tokens': 3,
        },
      };

      expect(getSpanTokenUsage(span)).toEqual({
        input_tokens: 10,
        output_tokens: 5,
        total_tokens: 15,
        cache_read_input_tokens: 7,
        cache_creation_input_tokens: 3,
      });
    });

    it('reads cache tokens from the dotted keys in V4 array-form attributes', () => {
      const span = {
        attributes: [
          { key: 'gen_ai.usage.input_tokens', value: { int_value: 10 } },
          { key: 'gen_ai.usage.output_tokens', value: { int_value: 5 } },
          { key: 'gen_ai.usage.cache_read.input_tokens', value: { int_value: 7 } },
          { key: 'gen_ai.usage.cache_creation.input_tokens', value: { double_value: 3 } },
        ],
      };

      expect(getSpanTokenUsage(span)).toEqual({
        input_tokens: 10,
        output_tokens: 5,
        total_tokens: 15,
        cache_read_input_tokens: 7,
        cache_creation_input_tokens: 3,
      });
    });

    it('prefers the dotted key over the underscore key when both are present', () => {
      const span = {
        attributes: {
          'gen_ai.usage.input_tokens': 10,
          'gen_ai.usage.output_tokens': 5,
          'gen_ai.usage.cache_read.input_tokens': 7,
          'gen_ai.usage.cache_read_input_tokens': 99,
        },
      };

      expect(getSpanTokenUsage(span)?.cache_read_input_tokens).toBe(7);
    });

    it('fills dotted cache tokens around an mlflow blob while the blob keeps winning per field', () => {
      const span = {
        attributes: {
          'mlflow.chat.tokenUsage': JSON.stringify({
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
            cache_read_input_tokens: 100,
          }),
          'gen_ai.usage.cache_read.input_tokens': 7,
          'gen_ai.usage.cache_creation.input_tokens': 3,
        },
      };

      expect(getSpanTokenUsage(span)).toEqual({
        input_tokens: 10,
        output_tokens: 5,
        total_tokens: 15,
        cache_read_input_tokens: 100,
        cache_creation_input_tokens: 3,
      });
    });
  });
});
