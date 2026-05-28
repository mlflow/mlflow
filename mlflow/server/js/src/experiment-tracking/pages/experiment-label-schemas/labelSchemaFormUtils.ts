/**
 * Form-state shape + bidirectional transforms for the labeling-schema
 * admin page. Mirrors the type hierarchy convention from the
 * `experiment-scorers` CLAUDE.md:
 *
 *   LabelSchema (wire + domain model)
 *     ↓ getFormValuesFromSchema
 *   LabelSchemaFormData (one field per form control)
 *     ↓ buildLabelSchemaInputFromForm + buildLabelSchemaPayloadFromForm
 *   { input: LabelSchemaInput, ...payload }  (back to wire shape)
 *
 * The categorical `options` field is newline-joined in the form
 * (textarea-friendly) and split back into an array on submit; this
 * matches how the scorers form handles `guidelines`.
 */

import type {
  InputCategorical,
  InputNumeric,
  InputPassFail,
  LabelSchema,
  LabelSchemaInput,
  LabelSchemaType,
} from '../../components/label-schemas/types';

/** Discriminator used in the form to switch which input fieldset is shown. */
export type LabelSchemaInputKind = 'pass_fail' | 'categorical' | 'numeric';

/**
 * Flat shape backing the react-hook-form controls. Different input
 * variants share this struct and are discriminated by `inputKind`; the
 * submit handler reads only the fields relevant to the active variant.
 */
export interface LabelSchemaFormData {
  name: string;
  type: LabelSchemaType;
  title: string;
  instruction: string;
  enable_comment: boolean;
  inputKind: LabelSchemaInputKind;

  // pass_fail
  passFailPositiveLabel: string;
  passFailNegativeLabel: string;

  // categorical
  categoricalOptions: string; // newline-joined for textarea
  categoricalPolarity: 'ascending' | 'descending' | '';
  categoricalMultiSelect: boolean;

  // numeric
  numericMinValue: string; // empty-string means "not set"
  numericMaxValue: string;
}

export const DEFAULT_FORM_VALUES: LabelSchemaFormData = {
  name: '',
  type: 'FEEDBACK',
  title: '',
  instruction: '',
  enable_comment: false,
  inputKind: 'pass_fail',
  passFailPositiveLabel: '',
  passFailNegativeLabel: '',
  categoricalOptions: '',
  categoricalPolarity: '',
  categoricalMultiSelect: false,
  numericMinValue: '',
  numericMaxValue: '',
};

const detectInputKind = (input: LabelSchemaInput): LabelSchemaInputKind => {
  if (input.pass_fail) {
    return 'pass_fail';
  }
  if (input.categorical) {
    return 'categorical';
  }
  if (input.numeric) {
    return 'numeric';
  }
  // Should never happen for a server-side-validated schema; fall back so
  // the form at least loads.
  return 'pass_fail';
};

const formatNumeric = (value: number | undefined): string => (value == null ? '' : String(value));

export const getFormValuesFromSchema = (schema: LabelSchema): LabelSchemaFormData => {
  const passFail: InputPassFail | undefined = schema.input.pass_fail;
  const categorical: InputCategorical | undefined = schema.input.categorical;
  const numeric: InputNumeric | undefined = schema.input.numeric;
  return {
    name: schema.name,
    type: schema.type,
    title: schema.title,
    instruction: schema.instruction ?? '',
    enable_comment: schema.enable_comment ?? false,
    inputKind: detectInputKind(schema.input),
    passFailPositiveLabel: passFail?.positive_label ?? '',
    passFailNegativeLabel: passFail?.negative_label ?? '',
    categoricalOptions: (categorical?.options ?? []).join('\n'),
    categoricalPolarity: categorical?.semantic_polarity ?? '',
    categoricalMultiSelect: categorical?.multi_select ?? false,
    numericMinValue: formatNumeric(numeric?.min_value),
    numericMaxValue: formatNumeric(numeric?.max_value),
  };
};

/** Parse the textarea-joined options back into a clean, deduplicated array. */
export const parseCategoricalOptions = (raw: string): string[] => {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const line of raw.split('\n')) {
    const trimmed = line.trim();
    if (trimmed === '' || seen.has(trimmed)) {
      continue;
    }
    seen.add(trimmed);
    out.push(trimmed);
  }
  return out;
};

const parseNumeric = (raw: string): number | undefined => {
  const trimmed = raw.trim();
  if (trimmed === '') {
    return undefined;
  }
  const parsed = Number(trimmed);
  if (Number.isNaN(parsed)) {
    return undefined;
  }
  return parsed;
};

export const buildLabelSchemaInputFromForm = (form: LabelSchemaFormData): LabelSchemaInput => {
  switch (form.inputKind) {
    case 'pass_fail':
      return {
        pass_fail: {
          positive_label: form.passFailPositiveLabel,
          negative_label: form.passFailNegativeLabel,
        },
      };
    case 'categorical': {
      const categorical: InputCategorical = {
        options: parseCategoricalOptions(form.categoricalOptions),
      };
      // Only emit `multi_select` when the user opted into it; absent on the
      // wire is equivalent to false on the server and keeps the round-trip
      // lossless for schemas that were created without `multi_select` set.
      if (form.categoricalMultiSelect) {
        categorical.multi_select = true;
      }
      if (form.categoricalPolarity !== '') {
        categorical.semantic_polarity = form.categoricalPolarity;
      }
      return { categorical };
    }
    case 'numeric': {
      const numeric: InputNumeric = {};
      const min = parseNumeric(form.numericMinValue);
      const max = parseNumeric(form.numericMaxValue);
      if (min !== undefined) {
        numeric.min_value = min;
      }
      if (max !== undefined) {
        numeric.max_value = max;
      }
      return { numeric };
    }
    default: {
      // Exhaustiveness guard: if a fourth LabelSchemaInputKind variant is
      // ever added without updating this switch, TypeScript will refuse the
      // assignment to `never` here.
      const _exhaustive: never = form.inputKind;
      throw new Error(`Unhandled inputKind: ${String(_exhaustive)}`);
    }
  }
};

/**
 * Per-field error map produced by client-side validation. Keys mirror
 * `LabelSchemaFormData` field names. All entries are optional; a missing
 * key means "no error for this field".
 */
export interface LabelSchemaFormErrors {
  name?: string;
  title?: string;
  instruction?: string;
  passFailPositiveLabel?: string;
  passFailNegativeLabel?: string;
  categoricalOptions?: string;
  categoricalPolarity?: string;
  numericMinValue?: string;
  numericMaxValue?: string;
}

/** Client-side mirror of the server-side validation rules in validation.py. */
export const validateLabelSchemaForm = (form: LabelSchemaFormData): LabelSchemaFormErrors => {
  const errors: LabelSchemaFormErrors = {};

  if (!form.name) {
    errors.name = 'Name is required.';
  } else if (form.name.length > 150) {
    errors.name = 'Name must be at most 150 characters.';
  } else if (!/^[a-zA-Z0-9_]+$/.test(form.name)) {
    errors.name = 'Name must be alphanumeric and underscore only.';
  }

  if (!form.title) {
    errors.title = 'Title is required.';
  } else if (form.title.length > 256) {
    errors.title = 'Title must be at most 256 characters.';
  }

  if (form.instruction.length > 1000) {
    errors.instruction = 'Instruction must be at most 1000 characters.';
  }

  if (form.inputKind === 'pass_fail') {
    if (!form.passFailPositiveLabel) {
      errors.passFailPositiveLabel = 'Positive label is required.';
    } else if (form.passFailPositiveLabel.length > 64) {
      errors.passFailPositiveLabel = 'Positive label must be at most 64 characters.';
    }
    if (!form.passFailNegativeLabel) {
      errors.passFailNegativeLabel = 'Negative label is required.';
    } else if (form.passFailNegativeLabel.length > 64) {
      errors.passFailNegativeLabel = 'Negative label must be at most 64 characters.';
    }
    if (
      form.passFailPositiveLabel &&
      form.passFailNegativeLabel &&
      form.passFailPositiveLabel === form.passFailNegativeLabel
    ) {
      errors.passFailNegativeLabel = 'Positive and negative labels must be distinct.';
    }
  }

  if (form.inputKind === 'categorical') {
    // Mirror validation.py: 1-100 options, each 1-64 chars (after dedupe + trim).
    const options = parseCategoricalOptions(form.categoricalOptions);
    if (options.length === 0) {
      errors.categoricalOptions = 'At least one option is required.';
    } else if (options.length > 100) {
      errors.categoricalOptions = 'At most 100 options are allowed.';
    } else if (options.some((o) => o.length > 64)) {
      errors.categoricalOptions = 'Each option must be at most 64 characters.';
    }
    if (form.type === 'FEEDBACK' && form.categoricalPolarity === '') {
      errors.categoricalPolarity = 'Polarity is required for feedback-type categorical schemas.';
    }
  }

  if (form.inputKind === 'numeric') {
    // Reject non-empty strings that don't parse as numbers up front so the
    // user gets a field-level error instead of a vague server-side
    // INVALID_PARAMETER_VALUE for expectation-type schemas where the
    // "require both bounds" rule doesn't apply.
    const minRaw = form.numericMinValue.trim();
    const maxRaw = form.numericMaxValue.trim();
    if (minRaw !== '' && Number.isNaN(Number(minRaw))) {
      errors.numericMinValue = 'Min value must be a number.';
    }
    if (maxRaw !== '' && Number.isNaN(Number(maxRaw))) {
      errors.numericMaxValue = 'Max value must be a number.';
    }
    const min = parseNumeric(form.numericMinValue);
    const max = parseNumeric(form.numericMaxValue);
    if (form.type === 'FEEDBACK' && (min === undefined || max === undefined)) {
      errors.numericMinValue = errors.numericMinValue ?? 'Feedback-type numeric schemas require both min and max.';
    }
    if (min !== undefined && max !== undefined && min >= max) {
      errors.numericMaxValue = 'Max must be strictly greater than min.';
    }
  }

  return errors;
};
