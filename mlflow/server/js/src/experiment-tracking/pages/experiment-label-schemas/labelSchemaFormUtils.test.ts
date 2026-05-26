import { describe, it, expect } from '@jest/globals';

import type { LabelSchema } from '../../components/label-schemas/types';
import {
  DEFAULT_FORM_VALUES,
  buildLabelSchemaInputFromForm,
  getFormValuesFromSchema,
  parseCategoricalOptions,
  validateLabelSchemaForm,
  type LabelSchemaFormData,
} from './labelSchemaFormUtils';

const baseValidForm: LabelSchemaFormData = {
  ...DEFAULT_FORM_VALUES,
  name: 'correctness',
  title: 'Is the answer correct?',
  inputKind: 'pass_fail',
  passFailPositiveLabel: 'Correct',
  passFailNegativeLabel: 'Incorrect',
};

describe('parseCategoricalOptions', () => {
  it('splits by newline, trims, dedupes, and drops blanks', () => {
    expect(parseCategoricalOptions('  low\nmedium \n\nhigh\nlow\n')).toEqual(['low', 'medium', 'high']);
  });

  it('returns an empty array for whitespace-only input', () => {
    expect(parseCategoricalOptions('   \n\n   ')).toEqual([]);
  });
});

describe('buildLabelSchemaInputFromForm', () => {
  it('builds pass_fail input from form fields', () => {
    expect(buildLabelSchemaInputFromForm(baseValidForm)).toEqual({
      pass_fail: { positive_label: 'Correct', negative_label: 'Incorrect' },
    });
  });

  it('builds categorical input with polarity and multi_select', () => {
    const form: LabelSchemaFormData = {
      ...baseValidForm,
      inputKind: 'categorical',
      categoricalOptions: 'low\nmedium\nhigh',
      categoricalPolarity: 'ascending',
      categoricalMultiSelect: true,
    };
    expect(buildLabelSchemaInputFromForm(form)).toEqual({
      categorical: {
        options: ['low', 'medium', 'high'],
        semantic_polarity: 'ascending',
        multi_select: true,
      },
    });
  });

  it('omits semantic_polarity when blank', () => {
    const form: LabelSchemaFormData = {
      ...baseValidForm,
      inputKind: 'categorical',
      categoricalOptions: 'a\nb',
      categoricalPolarity: '',
    };
    const input = buildLabelSchemaInputFromForm(form);
    expect(input.categorical).toBeDefined();
    expect(input.categorical?.semantic_polarity).toBeUndefined();
  });

  it('builds numeric input and skips missing bounds', () => {
    const form: LabelSchemaFormData = {
      ...baseValidForm,
      inputKind: 'numeric',
      numericMinValue: '1',
      numericMaxValue: '',
    };
    expect(buildLabelSchemaInputFromForm(form)).toEqual({
      numeric: { min_value: 1 },
    });
  });
});

describe('getFormValuesFromSchema', () => {
  it('round-trips a pass_fail schema through form ↔ wire', () => {
    const schema: LabelSchema = {
      schema_id: 'ls-1',
      experiment_id: '1',
      name: 'correctness',
      type: 'feedback',
      title: 'Is the answer correct?',
      instruction: 'Mark Correct if accurate.',
      enable_comment: true,
      input: { pass_fail: { positive_label: 'Correct', negative_label: 'Incorrect' } },
    };
    const form = getFormValuesFromSchema(schema);
    expect(form.name).toEqual('correctness');
    expect(form.inputKind).toEqual('pass_fail');
    expect(form.passFailPositiveLabel).toEqual('Correct');
    expect(buildLabelSchemaInputFromForm(form)).toEqual(schema.input);
  });

  it('round-trips a categorical schema (options joined with newlines)', () => {
    // multi_select defaults to false; the build path omits it when false
    // so the round-trip target is the schema with multi_select absent.
    const schema: LabelSchema = {
      schema_id: 'ls-2',
      experiment_id: '1',
      name: 'severity',
      type: 'feedback',
      title: 'Severity',
      input: {
        categorical: {
          options: ['low', 'medium', 'high'],
          semantic_polarity: 'ascending',
        },
      },
    };
    const form = getFormValuesFromSchema(schema);
    expect(form.categoricalOptions).toEqual('low\nmedium\nhigh');
    expect(form.categoricalPolarity).toEqual('ascending');
    expect(buildLabelSchemaInputFromForm(form)).toEqual(schema.input);
  });

  it('round-trips a categorical schema with multi_select=true', () => {
    const schema: LabelSchema = {
      schema_id: 'ls-2b',
      experiment_id: '1',
      name: 'tags',
      type: 'feedback',
      title: 'Tags',
      input: {
        categorical: {
          options: ['a', 'b'],
          semantic_polarity: 'ascending',
          multi_select: true,
        },
      },
    };
    const form = getFormValuesFromSchema(schema);
    expect(buildLabelSchemaInputFromForm(form)).toEqual(schema.input);
  });
});

describe('validateLabelSchemaForm', () => {
  it('accepts a valid pass_fail form', () => {
    expect(validateLabelSchemaForm(baseValidForm)).toEqual({});
  });

  it('rejects an invalid name (hyphens)', () => {
    const errors = validateLabelSchemaForm({ ...baseValidForm, name: 'bad-name' });
    expect(errors.name).toMatch(/alphanumeric and underscore/);
  });

  it('rejects a name longer than 150 characters', () => {
    const errors = validateLabelSchemaForm({ ...baseValidForm, name: 'a'.repeat(151) });
    expect(errors.name).toMatch(/at most 150/);
  });

  it('rejects empty title', () => {
    const errors = validateLabelSchemaForm({ ...baseValidForm, title: '' });
    expect(errors.title).toMatch(/required/);
  });

  it('rejects pass-fail labels that match', () => {
    const errors = validateLabelSchemaForm({
      ...baseValidForm,
      passFailPositiveLabel: 'same',
      passFailNegativeLabel: 'same',
    });
    expect(errors.passFailNegativeLabel).toMatch(/distinct/);
  });

  it('requires polarity for feedback-type categorical', () => {
    const errors = validateLabelSchemaForm({
      ...baseValidForm,
      inputKind: 'categorical',
      categoricalOptions: 'low\nhigh',
      categoricalPolarity: '',
    });
    expect(errors.categoricalPolarity).toMatch(/Polarity is required/);
  });

  it('allows expectation-type categorical without polarity', () => {
    const errors = validateLabelSchemaForm({
      ...baseValidForm,
      type: 'expectation',
      inputKind: 'categorical',
      categoricalOptions: 'low\nhigh',
      categoricalPolarity: '',
    });
    expect(errors.categoricalPolarity).toBeUndefined();
  });

  it('requires both bounds for feedback-type numeric', () => {
    const errors = validateLabelSchemaForm({
      ...baseValidForm,
      inputKind: 'numeric',
      numericMinValue: '',
      numericMaxValue: '',
    });
    expect(errors.numericMinValue).toMatch(/require both min and max/);
  });

  it('rejects numeric min >= max', () => {
    const errors = validateLabelSchemaForm({
      ...baseValidForm,
      inputKind: 'numeric',
      numericMinValue: '5',
      numericMaxValue: '5',
    });
    expect(errors.numericMaxValue).toMatch(/strictly greater/);
  });

  it('rejects non-numeric min/max even for expectation-type', () => {
    const errors = validateLabelSchemaForm({
      ...baseValidForm,
      type: 'expectation',
      inputKind: 'numeric',
      numericMinValue: 'abc',
      numericMaxValue: '',
    });
    expect(errors.numericMinValue).toMatch(/must be a number/);
  });

  // Boundary tests: lock in client-side length caps against silent drift
  // from validation.py.
  it.each([
    {
      label: 'title > 256',
      form: { ...baseValidForm, title: 'a'.repeat(257) },
      field: 'title' as const,
      match: /at most 256/,
    },
    {
      label: 'instruction > 1000',
      form: { ...baseValidForm, instruction: 'a'.repeat(1001) },
      field: 'instruction' as const,
      match: /at most 1000/,
    },
    {
      label: 'pass-fail positive label > 64',
      form: { ...baseValidForm, passFailPositiveLabel: 'a'.repeat(65) },
      field: 'passFailPositiveLabel' as const,
      match: /at most 64/,
    },
    {
      label: 'pass-fail negative label > 64',
      form: { ...baseValidForm, passFailNegativeLabel: 'a'.repeat(65) },
      field: 'passFailNegativeLabel' as const,
      match: /at most 64/,
    },
    {
      label: 'categorical options > 100',
      form: {
        ...baseValidForm,
        inputKind: 'categorical' as const,
        categoricalPolarity: 'ascending' as const,
        categoricalOptions: Array.from({ length: 101 }, (_, i) => `o${i}`).join('\n'),
      },
      field: 'categoricalOptions' as const,
      match: /at most 100/i,
    },
    {
      label: 'categorical option > 64 chars',
      form: {
        ...baseValidForm,
        inputKind: 'categorical' as const,
        categoricalPolarity: 'ascending' as const,
        categoricalOptions: 'a'.repeat(65),
      },
      field: 'categoricalOptions' as const,
      match: /at most 64/,
    },
  ])('boundary: $label', ({ form, field, match }) => {
    const errors = validateLabelSchemaForm(form);
    expect(errors[field]).toMatch(match);
  });
});

describe('numeric round-trip', () => {
  it('preserves min_value: 0 (verifies formatNumeric uses == null, not falsy)', () => {
    const schema: LabelSchema = {
      schema_id: 'ls-3',
      experiment_id: '1',
      name: 'rating',
      type: 'expectation',
      title: 'Rating',
      input: { numeric: { min_value: 0, max_value: 10 } },
    };
    const form = getFormValuesFromSchema(schema);
    expect(form.numericMinValue).toEqual('0');
    expect(form.numericMaxValue).toEqual('10');
    expect(buildLabelSchemaInputFromForm(form)).toEqual(schema.input);
  });
});
