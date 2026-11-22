import {
  getAssessmentValue,
  isExpectationAssessment,
  MLFLOW_TRACE_USER_KEY,
  SESSION_ID_METADATA_KEY,
  SOURCE_NAME_METADATA_KEY,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
import { MLFLOW_SOURCE_RUN_KEY } from '@databricks/web-shared/genai-traces-table';
import type { FilterFieldSelectOption } from '@databricks/web-shared/traces-table';
import { isAssessmentColumnEligible } from './assessmentColumns';

/** Fields whose values can be suggested from the trace rows currently loaded in the table. */
export type TraceFilterSuggestionField = 'trace_name' | 'user' | 'session' | 'run_name' | 'source' | 'input' | 'output';

export type TraceFilterSuggestionKeyField = 'tag' | 'metadata' | 'assessment' | 'expectation';

interface KeyValueSuggestionSet {
  keys: FilterFieldSelectOption[];
  valuesByKey: ReadonlyMap<string, FilterFieldSelectOption[]>;
}

export interface TraceFilterSuggestions {
  /** Value suggestions for scalar trace fields. */
  values: Readonly<Record<TraceFilterSuggestionField, FilterFieldSelectOption[]>>;
  /** Assessment names discovered on the loaded rows. */
  assessmentNames: FilterFieldSelectOption[];
  /** Key suggestions and key-dependent values for tags, metadata, and assessments. */
  keyedValues: Readonly<Record<TraceFilterSuggestionKeyField, KeyValueSuggestionSet>>;
}

const TRACE_NAME_TAG = 'mlflow.traceName';

const SUGGESTION_FIELDS: readonly TraceFilterSuggestionField[] = [
  'trace_name',
  'user',
  'session',
  'run_name',
  'source',
  'input',
  'output',
];

const option = (value: string, label = value): FilterFieldSelectOption => ({ value, label });

const addValue = (set: Set<string>, value: string | undefined): void => {
  // Restrict suggestions to non-empty scalar strings because all trace-search fields are serialized
  // as string literals by the MLflow compiler.
  if (typeof value === 'string' && value.trim().length > 0) {
    set.add(value);
  }
};

const toOptions = (values: Set<string>): FilterFieldSelectOption[] =>
  [...values].sort((left, right) => left.localeCompare(right)).map((value) => option(value));

const collectKeyedValues = (entries: Iterable<readonly [string, string]>): KeyValueSuggestionSet => {
  const valuesByKey = new Map<string, Set<string>>();
  for (const [key, value] of entries) {
    if (!key) {
      continue;
    }
    let values = valuesByKey.get(key);
    if (!values) {
      values = new Set<string>();
      valuesByKey.set(key, values);
    }
    addValue(values, value);
  }

  const keys = [...valuesByKey.keys()].sort((left, right) => left.localeCompare(right)).map((value) => option(value));
  const normalizedValues = new Map<string, FilterFieldSelectOption[]>();
  valuesByKey.forEach((values, key) => normalizedValues.set(key, toOptions(values)));
  return { keys, valuesByKey: normalizedValues };
};

/**
 * Builds all filter suggestions from the rows that are already available to the table.
 *
 * Input/output suggestions use the preview text already shown in the table. Previews may be truncated, so users can
 * edit a suggested value before applying a filter; freeform input remains available for values absent from the
 * current page. Run suggestions are the run IDs the traces were logged under.
 */
export const buildTraceFilterSuggestions = (traces: ModelTraceInfoV3[]): TraceFilterSuggestions => {
  const scalarValues = Object.fromEntries(SUGGESTION_FIELDS.map((field) => [field, new Set<string>()])) as Record<
    TraceFilterSuggestionField,
    Set<string>
  >;
  const tagEntries: Array<readonly [string, string]> = [];
  const metadataEntries: Array<readonly [string, string]> = [];
  const assessmentEntries: Array<readonly [string, string]> = [];
  const expectationEntries: Array<readonly [string, string]> = [];

  for (const trace of traces) {
    addValue(scalarValues.trace_name, trace.tags?.[TRACE_NAME_TAG]);
    addValue(scalarValues.user, trace.trace_metadata?.[MLFLOW_TRACE_USER_KEY]);
    addValue(scalarValues.session, trace.trace_metadata?.[SESSION_ID_METADATA_KEY]);
    addValue(scalarValues.run_name, trace.trace_metadata?.[MLFLOW_SOURCE_RUN_KEY]);
    addValue(scalarValues.source, trace.trace_metadata?.[SOURCE_NAME_METADATA_KEY]);

    addValue(scalarValues.input, trace.request);
    addValue(scalarValues.input, trace.request_preview);
    addValue(scalarValues.output, trace.response);
    addValue(scalarValues.output, trace.response_preview);

    for (const [key, value] of Object.entries(trace.tags ?? {})) {
      tagEntries.push([key, value]);
    }
    for (const [key, value] of Object.entries(trace.trace_metadata ?? {})) {
      metadataEntries.push([key, value]);
    }
    for (const assessment of trace.assessments ?? []) {
      if (!isAssessmentColumnEligible(assessment)) {
        continue;
      }
      const value = getAssessmentValue(assessment);
      const entries = isExpectationAssessment(assessment) ? expectationEntries : assessmentEntries;
      // Keep names whose value is missing so they are still offered as keys.
      entries.push([assessment.assessment_name, value === null || value === undefined ? '' : String(value)]);
    }
  }

  const values = Object.fromEntries(
    SUGGESTION_FIELDS.map((field) => [field, toOptions(scalarValues[field])]),
  ) as Record<TraceFilterSuggestionField, FilterFieldSelectOption[]>;

  return {
    values,
    assessmentNames: collectKeyedValues(assessmentEntries).keys,
    keyedValues: {
      tag: collectKeyedValues(tagEntries),
      metadata: collectKeyedValues(metadataEntries),
      assessment: collectKeyedValues(assessmentEntries),
      expectation: collectKeyedValues(expectationEntries),
    },
  };
};

/** Returns key-dependent options while keeping the field's freeform entry path intact. */
export const getTraceFilterValueOptions = (
  suggestions: TraceFilterSuggestions,
  field: TraceFilterSuggestionKeyField,
  key: string,
): FilterFieldSelectOption[] => suggestions.keyedValues[field].valuesByKey.get(key) ?? [];
