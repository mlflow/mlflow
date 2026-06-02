/**
 * Form-state shape + bidirectional transforms for the labeling-schema
 * admin page. Mirrors the type hierarchy convention from the
 * `experiment-scorers` CLAUDE.md:
 *
 *   LabelSchema (wire + domain model)
 *     ↓ getFormValuesFromSchema
 *   LabelSchemaFormData (one field per form control)
 *     ↓ buildLabelSchemaInputFromForm
 *   { input: LabelSchemaInput }  (back to wire shape)
 *
 * The categorical `options` field is a `string[]` backing the editable
 * options list (one input row per option); it's trimmed, deduped, and
 * stripped of blanks on submit.
 */

import type {
  InputCategorical,
  InputNumeric,
  InputPassFail,
  InputText,
  LabelSchema,
  LabelSchemaInput,
  LabelSchemaType,
} from '../../components/label-schemas/types';

/** Discriminator used in the form to switch which input fieldset is shown. */
export type LabelSchemaInputKind = 'pass_fail' | 'categorical' | 'numeric' | 'text';

/**
 * Flat shape backing the react-hook-form controls. Different input
 * variants share this struct and are discriminated by `inputKind`; the
 * submit handler reads only the fields relevant to the active variant.
 */
export interface LabelSchemaFormData {
  name: string;
  type: LabelSchemaType;
  instruction: string;
  enable_comment: boolean;
  inputKind: LabelSchemaInputKind;

  // pass_fail
  passFailPositiveLabel: string;
  passFailNegativeLabel: string;

  // categorical
  categoricalOptions: string[]; // one entry per editable option row
  categoricalMultiSelect: boolean;

  // numeric
  numericMinValue: string; // empty-string means "not set"
  numericMaxValue: string;

  // text
  textMaxLength: string; // empty-string means "no limit"
}

/**
 * Default labels for the pass/fail positive/negative inputs. Used both as
 * the form's initial values (so an author can create a thumbs-up/down
 * schema with zero typing) and as the preview's blank-field fallback.
 */
export const PASS_FAIL_POSITIVE_DEFAULT = 'Pass';
export const PASS_FAIL_NEGATIVE_DEFAULT = 'Fail';

/** Maximum number of categorical options a schema may define. */
export const MAX_CATEGORICAL_OPTIONS = 10;

export const DEFAULT_FORM_VALUES: LabelSchemaFormData = {
  name: '',
  type: 'FEEDBACK',
  instruction: '',
  enable_comment: false,
  inputKind: 'pass_fail',
  passFailPositiveLabel: PASS_FAIL_POSITIVE_DEFAULT,
  passFailNegativeLabel: PASS_FAIL_NEGATIVE_DEFAULT,
  categoricalOptions: [],
  categoricalMultiSelect: false,
  numericMinValue: '',
  numericMaxValue: '',
  textMaxLength: '',
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
  if (input.text) {
    return 'text';
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
  const text: InputText | undefined = schema.input.text;
  return {
    name: schema.name,
    type: schema.type,
    instruction: schema.instruction ?? '',
    enable_comment: schema.enable_comment ?? false,
    inputKind: detectInputKind(schema.input),
    passFailPositiveLabel: passFail?.positive_label ?? PASS_FAIL_POSITIVE_DEFAULT,
    passFailNegativeLabel: passFail?.negative_label ?? PASS_FAIL_NEGATIVE_DEFAULT,
    categoricalOptions: categorical?.options ?? [],
    categoricalMultiSelect: categorical?.multi_select ?? false,
    numericMinValue: formatNumeric(numeric?.min_value),
    numericMaxValue: formatNumeric(numeric?.max_value),
    textMaxLength: formatNumeric(text?.max_length),
  };
};

/** Trim, drop blanks, and dedupe the option list (order-preserving). */
export const normalizeCategoricalOptions = (options: string[]): string[] => {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const option of options) {
    const trimmed = option.trim();
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
        options: normalizeCategoricalOptions(form.categoricalOptions),
      };
      // Only emit `multi_select` when the user opted into it; absent on the
      // wire is equivalent to false on the server and keeps the round-trip
      // lossless for schemas that were created without `multi_select` set.
      if (form.categoricalMultiSelect) {
        categorical.multi_select = true;
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
    case 'text': {
      const text: InputText = {};
      const maxLength = parseNumeric(form.textMaxLength);
      if (maxLength !== undefined) {
        text.max_length = maxLength;
      }
      return { text };
    }
    default: {
      // Exhaustiveness guard: if a new LabelSchemaInputKind variant is ever
      // added without updating this switch, TypeScript will refuse the
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
  instruction?: string;
  passFailPositiveLabel?: string;
  passFailNegativeLabel?: string;
  categoricalOptions?: string;
  numericMinValue?: string;
  numericMaxValue?: string;
  textMaxLength?: string;
}

/** Client-side mirror of the server-side validation rules in validation.py. */
export const validateLabelSchemaForm = (form: LabelSchemaFormData): LabelSchemaFormErrors => {
  const errors: LabelSchemaFormErrors = {};

  if (!form.name) {
    errors.name = 'Name is required.';
  } else if (form.name.length > 256) {
    errors.name = 'Name must be at most 256 characters.';
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
    // Up to MAX_CATEGORICAL_OPTIONS options, each 1-64 chars (after dedupe + trim).
    const options = normalizeCategoricalOptions(form.categoricalOptions);
    if (options.length === 0) {
      errors.categoricalOptions = 'At least one option is required.';
    } else if (options.length > MAX_CATEGORICAL_OPTIONS) {
      errors.categoricalOptions = `At most ${MAX_CATEGORICAL_OPTIONS} options are allowed.`;
    } else if (options.some((o) => o.length > 64)) {
      errors.categoricalOptions = 'Each option must be at most 64 characters.';
    }
  }

  if (form.inputKind === 'numeric') {
    // Reject non-empty strings that don't parse as numbers up front so
    // the user gets a field-level error instead of a vague server-side
    // INVALID_PARAMETER_VALUE.
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
    if (min !== undefined && max !== undefined && min >= max) {
      errors.numericMaxValue = 'Max must be strictly greater than min.';
    }
  }

  if (form.inputKind === 'text') {
    const raw = form.textMaxLength.trim();
    if (raw !== '') {
      const parsed = Number(raw);
      if (Number.isNaN(parsed) || !Number.isInteger(parsed) || parsed < 1) {
        errors.textMaxLength = 'Max length must be a positive whole number, or blank for no limit.';
      }
    }
  }

  return errors;
};
