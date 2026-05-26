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
          multi_select: false,
        },
      },
    };
    const form = getFormValuesFromSchema(schema);
    expect(form.categoricalOptions).toEqual('low\nmedium\nhigh');
    expect(form.categoricalPolarity).toEqual('ascending');
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
});
