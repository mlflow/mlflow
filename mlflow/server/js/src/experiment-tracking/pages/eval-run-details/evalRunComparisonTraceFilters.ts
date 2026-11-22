import type { IntlShape } from 'react-intl';
import { convertTraceInfoV3ToRunEvalEntry } from '@databricks/web-shared/genai-traces-table';
// Not re-exported from the genai-traces-table barrel in OSS — import from its module.
import { ERROR_KEY } from '@databricks/web-shared/genai-traces-table/utils/AggregationUtils';
import {
  getAssessmentValue,
  isExpectationAssessment,
  isSessionLevelAssessment,
  MLFLOW_TRACE_USER_KEY,
  SESSION_ID_METADATA_KEY,
  SOURCE_NAME_METADATA_KEY,
  type Assessment,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
import {
  FilterOp,
  isClauseComplete,
  parseTraceDurationToMs,
  type FilterClause,
  type FilterFieldDef,
  type TracesComparisonRow,
  type TraceFilterModel,
} from '@databricks/web-shared/traces-table';
import { isAssessmentColumnEligible } from '../../components/experiment-page/components/traces-v4/utils/assessmentColumns';
import { isSupportedFilterClause } from '../../components/experiment-page/components/traces-v4/utils/filterModel';
import { getComparisonRunTraces } from './evalRunSessionComparisonRows';

export const COMPARISON_FILTER_FIELDS = new Set([
  'state',
  'duration',
  'trace_name',
  'user',
  'session',
  'source',
  'input',
  'output',
  'tag',
  'metadata',
  'assessment',
  'expectation',
]);

const CANONICAL_NUMERIC_FILTER = /^[+-]?(?:(?:0|[1-9]\d*)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;
const NUMERIC_VALUE = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

export const getComparisonTraceInput = (trace: ModelTraceInfoV3, recordedInputs?: ReadonlyMap<string, string>) =>
  recordedInputs?.get(trace.trace_id) ??
  trace.request ??
  trace.trace_metadata?.['mlflow.traceInputs'] ??
  trace.request_preview;

export const getComparisonTraceOutput = (trace: ModelTraceInfoV3) =>
  trace.response ?? trace.trace_metadata?.['mlflow.traceOutputs'] ?? trace.response_preview;

/** Filter kinds stay separate even when feedback and expectation share a column name. */
export const getComparisonFilterAssessments = (trace: ModelTraceInfoV3): Assessment[] => {
  const latest = new Map<string, Assessment>();
  for (const assessment of trace.assessments ?? []) {
    if (!isAssessmentColumnEligible(assessment) || isSessionLevelAssessment(assessment)) {
      continue;
    }
    const key = JSON.stringify([isExpectationAssessment(assessment), assessment.assessment_name]);
    const previous = latest.get(key);
    if (!previous || (assessment.create_time ?? '') > (previous.create_time ?? '')) {
      latest.set(key, assessment);
    }
  }
  return [...latest.values()];
};

export const validateComparisonFilters = (
  model: TraceFilterModel,
  fields: FilterFieldDef[],
  intl: IntlShape,
): string | undefined => {
  for (const clause of model) {
    if (!isClauseComplete(clause) || !isSupportedFilterClause(fields, clause)) {
      return intl.formatMessage({
        defaultMessage: 'Complete or remove unsupported filters before applying.',
        description: 'Invalid or incomplete comparison trace filter',
      });
    }
    if (
      clause.field === 'duration' &&
      (!NUMERIC_VALUE.test(clause.value.trim()) || !Number.isFinite(Number(clause.value)))
    ) {
      return intl.formatMessage({
        defaultMessage: 'Enter a finite duration in milliseconds.',
        description: 'Invalid comparison duration filter value',
      });
    }
    if (clause.operator === FilterOp.RLIKE) {
      try {
        // Unicode mode rejects identity escapes such as Java's \\A, rather than silently matching "A".
        new RegExp(clause.value, 'u');
      } catch {
        return intl.formatMessage({
          defaultMessage:
            'Enter a valid JavaScript regular expression. Backend-only regex syntax is not supported here.',
          description: 'Invalid comparison input or output regular expression',
        });
      }
    }
  }
  return undefined;
};

const assessmentValueEquals = (value: ReturnType<typeof getAssessmentValue>, filter: string): boolean => {
  if (
    (typeof value === 'number' || (typeof value === 'string' && NUMERIC_VALUE.test(value.trim()))) &&
    CANONICAL_NUMERIC_FILTER.test(filter.trim()) &&
    Number.isFinite(Number(value)) &&
    Number.isFinite(Number(filter))
  ) {
    return Number(value) === Number(filter);
  }
  return value?.toString() === filter;
};

const matchesAssessment = (assessments: Assessment[], clause: FilterClause): boolean => {
  const assessment = assessments.find(
    (candidate) =>
      candidate.assessment_name === clause.key &&
      isExpectationAssessment(candidate) === (clause.field === 'expectation'),
  );
  const hasError = Boolean(
    assessment && (assessment.error ?? ('feedback' in assessment ? assessment.feedback?.error : undefined)),
  );
  const value = assessment && !hasError ? getAssessmentValue(assessment) : undefined;
  switch (clause.operator) {
    case FilterOp.EQUALS:
      return clause.value === ERROR_KEY ? hasError : assessmentValueEquals(value, clause.value);
    case FilterOp.NOT_EQUALS:
      return value !== null && value !== undefined && !assessmentValueEquals(value, clause.value);
    default:
      return false;
  }
};

const getFieldValue = (
  trace: ModelTraceInfoV3,
  runUuid: string,
  clause: FilterClause,
  inputs?: ReadonlyMap<string, string>,
): string | number | null | undefined => {
  switch (clause.field) {
    case 'state':
      return trace.state;
    case 'duration':
      return trace.execution_duration ? parseTraceDurationToMs(trace.execution_duration) : undefined;
    case 'trace_name':
      return trace.tags?.['mlflow.traceName'];
    case 'user':
      return trace.trace_metadata?.[MLFLOW_TRACE_USER_KEY] ?? trace.tags?.[MLFLOW_TRACE_USER_KEY];
    case 'session':
      return trace.trace_metadata?.[SESSION_ID_METADATA_KEY];
    case 'run_name':
      return runUuid;
    case 'source':
      return trace.trace_metadata?.[SOURCE_NAME_METADATA_KEY];
    case 'input':
      return getComparisonTraceInput(trace, inputs);
    case 'output':
      return getComparisonTraceOutput(trace);
    case 'tag':
      return trace.tags?.[clause.key ?? ''];
    case 'metadata':
      return trace.trace_metadata?.[clause.key ?? ''];
    default:
      return undefined;
  }
};

const matchesValue = (value: ReturnType<typeof getFieldValue>, clause: FilterClause, regex?: RegExp): boolean => {
  const missing = value === null || value === undefined;
  if (missing || (typeof value === 'number' && !Number.isFinite(value))) {
    return false;
  }
  const operand = typeof value === 'number' ? Number(clause.value) : clause.value;
  switch (clause.operator) {
    case FilterOp.EQUALS:
      return value === operand;
    case FilterOp.NOT_EQUALS:
      return value !== operand;
    case FilterOp.GREATER_THAN:
      return typeof value === 'number' && value > Number(operand);
    case FilterOp.GREATER_THAN_OR_EQUALS:
      return typeof value === 'number' && value >= Number(operand);
    case FilterOp.LESS_THAN:
      return typeof value === 'number' && value < Number(operand);
    case FilterOp.LESS_THAN_OR_EQUALS:
      return typeof value === 'number' && value <= Number(operand);
    case FilterOp.CONTAINS:
      return String(value).toLowerCase().includes(clause.value.toLowerCase());
    case FilterOp.RLIKE:
      return regex?.test(String(value)) ?? false;
    default:
      return false;
  }
};

export const matchesComparisonTraceSearch = (trace: ModelTraceInfoV3, search: string): boolean => {
  if (!search) {
    return true;
  }
  const value = convertTraceInfoV3ToRunEvalEntry(trace);
  const query = search.toLowerCase();
  return Boolean(
    JSON.stringify(value.inputs).toLowerCase().includes(query) ||
    value.inputsId.toLowerCase() === query ||
    value.evaluationId?.toLowerCase().includes(query) ||
    value.fullTraceId?.toLowerCase().includes(query),
  );
};

/** Evaluate every criterion on the same trace, then retain its complete, already-matched group. */
export const filterEvalRunComparisonRows = <T extends TracesComparisonRow>(
  rows: T[],
  model: TraceFilterModel,
  search: string,
  inputs?: ReadonlyMap<string, string>,
): T[] => {
  let compiled: { clause: FilterClause; regex?: RegExp }[];
  try {
    compiled = model.map((clause) => ({
      clause,
      regex: clause.operator === FilterOp.RLIKE ? new RegExp(clause.value, 'u') : undefined,
    }));
  } catch {
    return [];
  }
  return rows.filter((row) =>
    row.runs.some((run) =>
      getComparisonRunTraces(run).some((trace) => {
        if (!trace || !matchesComparisonTraceSearch(trace, search)) {
          return false;
        }
        const assessments = getComparisonFilterAssessments(trace);
        return compiled.every(({ clause, regex }) =>
          clause.field === 'assessment' || clause.field === 'expectation'
            ? matchesAssessment(assessments, clause)
            : matchesValue(getFieldValue(trace, run.runUuid, clause, inputs), clause, regex),
        );
      }),
    ),
  );
};
