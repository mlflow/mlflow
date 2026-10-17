import {
  COST_METADATA_KEY,
  MLFLOW_TRACE_USER_KEY,
  SESSION_ID_METADATA_KEY,
  SOURCE_NAME_METADATA_KEY,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
import { FilterOp, type FilterClause, type FilterFieldDef } from '@databricks/web-shared/traces-table';

/** A field definition together with values observed in the loaded trace rows. */
export interface TraceSearchField {
  id: string;
  field: FilterFieldDef;
  key?: string;
  label: string;
  values: { value: string; label: string }[];
  aliases: string[];
}

interface TraceSearchExample {
  field: TraceSearchField;
  clause: FilterClause;
  label: string;
}

const TRACE_NAME_TAG = 'mlflow.traceName';

/** The order in which field suggestions should be surfaced by the typeahead. */
const FIELD_PRIORITY = new Map<string, number>([
  ['tag', 0],
  ['duration', 1],
  ['cost', 2],
  ['assessment', 3],
  ['rationale', 3],
  ['expectation', 3],
  ['state', 4],
  ['trace_name', 5],
  ['service_name', 6],
  ['source', 7],
  ['user', 8],
  ['session', 9],
]);

const FIELD_ALIASES = new Map<string, string[]>([
  ['duration', ['duration', 'latency']],
  ['cost', ['cost']],
  ['assessment', ['feedback']],
  ['rationale', ['rationale', 'scorer rationale']],
  ['expectation', ['expectation']],
  ['tag', ['tag', 'tags']],
  ['metadata', ['metadata']],
  ['span_attribute', ['attributes']],
  ['session', ['session', 'session id']],
  ['trace_name', ['trace_name', 'trace name']],
  ['state', ['state', 'status']],
  ['service_name', ['service_name', 'service name', 'service']],
  ['source', ['source']],
  ['user', ['user']],
]);

const compareText = (left: string, right: string): number => (left < right ? -1 : left > right ? 1 : 0);

const quoteString = (value: string): string => `'${value.replaceAll('\\', '\\\\').replaceAll("'", "\\'")}'`;

const quotePart = (value: string): string => (/^[A-Za-z0-9_.:/+-]+$/.test(value) ? value : quoteString(value));

/** Format a field and optional key using the trace search language's canonical spelling. */
export const formatTraceSearchFieldName = (fieldId: string, key?: string): string => {
  const fieldName =
    fieldId === 'duration'
      ? 'latency'
      : fieldId === 'assessment'
        ? 'feedback'
        : fieldId === 'span_attribute'
          ? 'attributes'
          : fieldId.toLocaleLowerCase();
  return key === undefined ? fieldName : `${fieldName}.${quotePart(key)}`;
};

/** Format a value for a canonical trace search label while preserving its actual value. */
export const formatTraceSearchValue = (
  field: Pick<FilterFieldDef, 'id' | 'valueInput'> | string,
  value: string,
  displayValue?: string,
): string => {
  const fieldId = typeof field === 'string' ? field : field.id;
  const isNumeric =
    fieldId === 'duration' || fieldId === 'cost' || (typeof field !== 'string' && field.valueInput === 'number');
  return isNumeric ? (displayValue ?? (fieldId === 'duration' ? `${value} ms` : value)) : quoteString(value);
};

// The backend accepts integer milliseconds, while observed durations have finer precision.
// Suggest ranges so rounding does not turn an observed value into an unmatched equality.
const DURATION_OPERATORS: FilterOp[] = [
  FilterOp.GREATER_THAN,
  FilterOp.GREATER_THAN_OR_EQUALS,
  FilterOp.LESS_THAN,
  FilterOp.LESS_THAN_OR_EQUALS,
];

/** Values are emitted into string-valued clauses, so reject values that cannot be serialized reliably. */
const toSuggestionValue = (value: unknown): string | undefined => {
  if (typeof value === 'string') {
    return value.trim() === '' ? undefined : value;
  }
  if (typeof value === 'number') {
    return Number.isFinite(value) ? String(value) : undefined;
  }
  if (typeof value === 'boolean') {
    return String(value);
  }
  if (Array.isArray(value) && value.every((item) => typeof item === 'string')) {
    const serialized = String(value);
    return serialized.trim() === '' ? undefined : serialized;
  }
  return undefined;
};

const toDurationMs = (duration: string | undefined): number | undefined => {
  if (!duration) {
    return undefined;
  }
  const match = /^\s*(\d+(?:\.\d+)?)(ms|s)\s*$/.exec(duration);
  if (!match) {
    return undefined;
  }
  const amount = Number(match[1]);
  const milliseconds = match[2] === 's' ? amount * 1000 : amount;
  if (!Number.isFinite(milliseconds) || milliseconds < 0) {
    return undefined;
  }
  return Math.floor(milliseconds);
};

/**
 * The tag compiler quotes keys containing dots/spaces with backticks, but cannot escape a backtick
 * within a key. Control characters also make a generated clause unsafe.
 */
const isSafeTagKey = (key: string): boolean => /^[A-Za-z0-9_. ]+$/.test(key);

/** Metadata keys are emitted as a bare request_metadata.<key> reference by the compiler. */
const isSafeMetadataKey = (key: string): boolean => /^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*$/.test(key);

/** Assessment names are wrapped in backticks by the compiler, so embedded delimiters are unsafe. */
const isSafeAssessmentKey = (key: string): boolean => !/[`\r\n]/.test(key);

const isSafeKey = (fieldId: string, key: string): boolean => {
  switch (fieldId) {
    case 'tag':
      return isSafeTagKey(key);
    case 'metadata':
      return isSafeMetadataKey(key);
    case 'assessment':
    case 'expectation':
    case 'rationale':
      return isSafeAssessmentKey(key);
    case 'span_attribute':
      return key.trim() !== '' && !/["\\]/.test(key);
    default:
      return true;
  }
};

/** Whether a manually typed key can be passed to the existing MLflow clause compiler safely. */
export const isSafeTraceSearchKey = (fieldId: string, key: string): boolean => isSafeKey(fieldId, key);

const makeAliases = (field: FilterFieldDef): string[] => {
  const aliases = FIELD_ALIASES.get(field.id) ?? [field.id];
  return [...new Set(aliases.filter((alias) => alias.trim() !== '').map((alias) => alias.toLowerCase()))];
};

/** Stable, product-facing aliases used by both the legacy typeahead and the query language. */
export const getTraceSearchFieldAliases = (field: FilterFieldDef): string[] => makeAliases(field);

const makeFieldAliases = (field: FilterFieldDef, key?: string): string[] => [
  ...new Set([...makeAliases(field), ...(key ? [key.toLowerCase()] : [])]),
];

const makeOption = (value: string, label?: string): { value: string; label: string } => ({
  value,
  label: label ?? value,
});

const addObservedValue = (values: Map<string, string>, rawValue: unknown, label?: string): void => {
  const value = toSuggestionValue(rawValue);
  if (value !== undefined && !values.has(value)) {
    values.set(value, label ?? value);
  }
};

const sortValues = (values: Map<string, string>): { value: string; label: string }[] =>
  [...values.entries()]
    .sort(([left], [right]) => compareText(left, right))
    .map(([value, label]) => makeOption(value, label));

const getFieldValue = (fieldId: string, trace: ModelTraceInfoV3): unknown => {
  switch (fieldId) {
    case 'duration':
      return toDurationMs(trace.execution_duration);
    case 'cost': {
      const rawCost = trace.trace_metadata?.[COST_METADATA_KEY];
      if (typeof rawCost !== 'string') {
        return undefined;
      }
      try {
        const parsed: unknown = JSON.parse(rawCost);
        return parsed && typeof parsed === 'object' && 'total_cost' in parsed ? parsed.total_cost : undefined;
      } catch {
        return undefined;
      }
    }
    case 'session':
      return trace.trace_metadata?.[SESSION_ID_METADATA_KEY];
    case 'trace_name':
      return trace.tags?.[TRACE_NAME_TAG];
    case 'state':
      return trace.state === 'STATE_UNSPECIFIED' ? undefined : trace.state;
    case 'source':
      return trace.trace_metadata?.[SOURCE_NAME_METADATA_KEY];
    case 'user':
      return trace.trace_metadata?.[MLFLOW_TRACE_USER_KEY];
    case 'run_name':
      return trace.trace_metadata?.['mlflow.sourceRun'];
    default:
      return undefined;
  }
};

const getFieldOptionLabel = (field: FilterFieldDef, value: string): string | undefined =>
  field.options?.find((option) => option.value === value)?.label;

const collectScalarValues = (field: FilterFieldDef, traces: ModelTraceInfoV3[]): Map<string, string> => {
  const values = new Map<string, string>();
  for (const trace of traces) {
    const value = toSuggestionValue(getFieldValue(field.id, trace));
    if (value === undefined) {
      continue;
    }
    // Select fields (currently State) may have a fixed set of supported values. Do not suggest a
    // value that the field model cannot serialize even when a row contains an unknown enum token.
    if (field.options && !field.options.some((option) => option.value === value)) {
      continue;
    }
    addObservedValue(values, value, field.id === 'duration' ? `${value} ms` : getFieldOptionLabel(field, value));
  }
  return values;
};

const collectKeyedValues = (field: FilterFieldDef, traces: ModelTraceInfoV3[]): Map<string, Map<string, string>> => {
  const valuesByKey = new Map<string, Map<string, string>>();
  // The server compares feedback and expectation values with `=` against their stored JSON text, and
  // a parsed number can't tell a stored `4` from `4.0`. Numeric values are therefore not suggested,
  // while their names still are.
  const numericKeys = new Set<string>();
  const add = (rawKey: unknown, rawValue: unknown): void => {
    const key = toSuggestionValue(rawKey);
    if (key === undefined || !isSafeKey(field.id, key)) {
      return;
    }
    let values = valuesByKey.get(key);
    if (!values) {
      values = new Map<string, string>();
      valuesByKey.set(key, values);
    }
    if (typeof rawValue === 'number') {
      numericKeys.add(key);
    } else {
      addObservedValue(values, rawValue);
    }
    if (values.size === 0 && !numericKeys.has(key)) {
      valuesByKey.delete(key);
    }
  };

  for (const trace of traces) {
    if (field.id === 'assessment' || field.id === 'expectation' || field.id === 'rationale') {
      for (const assessment of trace.assessments ?? []) {
        if (assessment.valid === false || assessment.error) {
          continue;
        }
        // Keep names and values scoped to the namespace that the clause compiles into.
        if (
          field.id === 'assessment' &&
          'feedback' in assessment &&
          assessment.feedback &&
          !assessment.feedback.error
        ) {
          add(assessment.assessment_name, assessment.feedback.value);
        } else if (field.id === 'rationale' && assessment.rationale) {
          add(assessment.assessment_name, assessment.rationale);
        } else if (
          field.id === 'expectation' &&
          'expectation' in assessment &&
          assessment.expectation &&
          'value' in assessment.expectation
        ) {
          add(assessment.assessment_name, assessment.expectation.value);
        }
      }
      continue;
    }

    if (field.id !== 'tag' && field.id !== 'metadata') {
      continue;
    }
    const entries = field.id === 'tag' ? Object.entries(trace.tags ?? {}) : Object.entries(trace.trace_metadata ?? {});
    for (const [key, value] of entries) {
      // mlflow.* tags are implementation metadata rather than user tags. The trace-name tag is
      // offered through its dedicated field above.
      if (field.id === 'tag' && key.startsWith('mlflow.')) {
        continue;
      }
      add(key, value);
    }
  }
  return valuesByKey;
};

/**
 * Builds searchable field instances from the supplied filter model and rows already loaded by the
 * table. Row-backed keyed fields (tags, metadata, assessments) are expanded once per observed safe
 * key. Span attributes retain a namespace-only entry because trace rows do not include span
 * payloads; other keyed fields without row data are omitted. Non-keyed fields remain available with
 * an empty value list when no row data is present.
 */
export const buildTraceSearchFields = (fields: FilterFieldDef[], traces: ModelTraceInfoV3[]): TraceSearchField[] => {
  const result: TraceSearchField[] = [];
  const seenFields = new Set<string>();

  for (const field of fields) {
    if (seenFields.has(field.id)) {
      continue;
    }
    seenFields.add(field.id);
    const searchField =
      field.id === 'duration'
        ? { ...field, operators: field.operators.filter((operator) => DURATION_OPERATORS.includes(operator)) }
        : field;
    if (searchField.requiresKey) {
      const valuesByKey = collectKeyedValues(searchField, traces);
      if (searchField.id === 'span_attribute') {
        result.push({
          id: searchField.id,
          field: searchField,
          label: formatTraceSearchFieldName(searchField.id),
          values: [],
          aliases: makeFieldAliases(searchField),
        });
      }
      for (const key of [...valuesByKey.keys()].sort(compareText)) {
        result.push({
          id: `${searchField.id}:${key}`,
          field: searchField,
          key,
          label: formatTraceSearchFieldName(searchField.id, key),
          values: sortValues(valuesByKey.get(key) ?? new Map()),
          aliases: makeFieldAliases(searchField, key),
        });
      }
      continue;
    }

    result.push({
      id: searchField.id,
      field: searchField,
      label: formatTraceSearchFieldName(searchField.id),
      values: sortValues(collectScalarValues(searchField, traces)),
      aliases: makeFieldAliases(searchField),
    });
  }

  return result;
};

const firstValue = (field: TraceSearchField): { value: string; label: string } | undefined => {
  if (field.values.length === 0) {
    return undefined;
  }

  // Numeric examples use the largest observed value as a useful threshold while retaining the
  // observed value exactly. Invalid or negative values are ignored even for manually constructed
  // search fields supplied by a caller.
  if (field.field.id === 'duration' || field.field.id === 'cost') {
    const numericValues = field.values
      .map((option) => ({ option, number: Number(option.value) }))
      .filter(({ number }) => Number.isFinite(number) && number >= 0)
      .sort((left, right) => left.number - right.number || compareText(left.option.value, right.option.value));
    return numericValues[numericValues.length - 1]?.option;
  }

  return [...field.values].sort((left, right) => compareText(left.value, right.value))[0];
};

const exampleOperator = (field: TraceSearchField): FilterOp | undefined => {
  if (field.field.id === 'duration' && field.field.operators.includes(FilterOp.GREATER_THAN_OR_EQUALS)) {
    return FilterOp.GREATER_THAN_OR_EQUALS;
  }
  if (field.field.id === 'cost' && field.field.operators.includes(FilterOp.GREATER_THAN_OR_EQUALS)) {
    return FilterOp.GREATER_THAN_OR_EQUALS;
  }
  if (field.field.operators.includes(FilterOp.EQUALS)) {
    return FilterOp.EQUALS;
  }
  return field.field.operators[0];
};

const exampleRank = (field: TraceSearchField): [number, string, string] => [
  FIELD_PRIORITY.get(field.field.id) ?? Number.MAX_SAFE_INTEGER,
  field.field.id,
  field.key ?? '',
];

const compareFieldsForExamples = (left: TraceSearchField, right: TraceSearchField): number => {
  const leftRank = exampleRank(left);
  const rightRank = exampleRank(right);
  return leftRank[0] - rightRank[0] || compareText(leftRank[1], rightRank[1]) || compareText(leftRank[2], rightRank[2]);
};

const formatExampleLabel = (field: TraceSearchField, clause: FilterClause, valueLabel: string): string =>
  `${field.label} ${clause.operator} ${formatTraceSearchValue(field.field, clause.value, valueLabel)}`;

/** Returns at most five deterministic, complete clauses backed by observed values. */
export const getTraceSearchExamples = (searchFields: TraceSearchField[]): TraceSearchExample[] => {
  const examples: TraceSearchExample[] = [];
  const seenClauses = new Set<string>();
  const seenFieldCategories = new Set<string>();

  for (const field of [...searchFields].sort(compareFieldsForExamples)) {
    const value = firstValue(field);
    const operator = exampleOperator(field);
    if (
      !value ||
      !operator ||
      seenFieldCategories.has(field.field.id) ||
      (field.key !== undefined && !isSafeKey(field.field.id, field.key))
    ) {
      continue;
    }

    const clause: FilterClause = {
      field: field.field.id,
      operator,
      value: value.value,
      ...(field.key !== undefined ? { key: field.key } : {}),
    };
    const clauseKey = `${clause.field}\u0000${clause.key ?? ''}\u0000${clause.operator}\u0000${clause.value}`;
    if (seenClauses.has(clauseKey)) {
      continue;
    }
    seenClauses.add(clauseKey);
    seenFieldCategories.add(field.field.id);
    examples.push({
      field,
      clause,
      label: formatExampleLabel(field, clause, value.label),
    });
    if (examples.length === 5) {
      break;
    }
  }

  return examples;
};
