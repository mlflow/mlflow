import type { IntlShape } from '@databricks/i18n';
import { KnownEvaluationResultAssessmentStringValue } from '@databricks/web-shared/genai-traces-table';
import {
  formatCostUSD,
  getAssessmentValue,
  getTraceCost,
  getTraceTokenUsage,
  isFeedbackAssessment,
  isSessionLevelAssessment,
  type AssessmentValue,
  type FeedbackAssessment,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
import {
  formatTraceDurationMs,
  parseTraceDurationToMs,
  type SortDirection,
  type TracesComparisonRow,
  type TraceSortColumnId,
} from '@databricks/web-shared/traces-table';
import {
  assessmentColumnId,
  assessmentNameFromColumnId,
  isAssessmentColumnEligible,
  isSessionAssessmentColumnEligible,
  isAssessmentColumnId,
} from '../../components/experiment-page/components/traces-v4/utils/assessmentColumns';
import {
  getNumericComparisonAggregate,
  type EvalRunComparisonColumnAggregate,
} from './evalRunComparisonColumnAggregates';
import { getComparisonRunTraces } from './evalRunSessionComparisonRows';
import { compareComparisonSortValues } from './evalRunComparisonTraceSort';

export interface SessionComparisonAssessmentInfo {
  name: string;
  scope: 'session' | 'trace';
  dtype: 'numeric' | 'boolean' | 'pass-fail' | 'string';
}

const isFiniteValue = (value: AssessmentValue | undefined): value is number =>
  typeof value === 'number' && Number.isFinite(value);
const averageValues = (values: number[]): number | undefined =>
  values.length ? values.reduce((sum, value) => sum + value / values.length, 0) : undefined;
const getAssessmentError = (assessment: FeedbackAssessment) => assessment.error ?? assessment.feedback.error;

/** The same eligibility/create-time policy as single-run, with a stable ID tie-break for joined runs. */
export const pickComparisonFeedback = (
  traces: ModelTraceInfoV3[],
  name: string,
  scope: SessionComparisonAssessmentInfo['scope'],
): FeedbackAssessment | undefined =>
  traces
    .flatMap((trace) => trace.assessments ?? [])
    .reduce<FeedbackAssessment | undefined>((latest, assessment) => {
      if (
        assessment.assessment_name !== name ||
        !isFeedbackAssessment(assessment) ||
        // OSS keeps trace and session eligibility apart; universe checks one predicate plus the scope.
        !(scope === 'session' ? isSessionAssessmentColumnEligible(assessment) : isAssessmentColumnEligible(assessment))
      ) {
        return latest;
      }
      return !latest ||
        assessment.create_time > latest.create_time ||
        (assessment.create_time === latest.create_time && assessment.assessment_id < latest.assessment_id)
        ? assessment
        : latest;
    }, undefined);

const getSourceAssessments = (
  traces: ModelTraceInfoV3[],
  info: Pick<SessionComparisonAssessmentInfo, 'name' | 'scope'>,
) =>
  (info.scope === 'session' ? [traces] : traces.map((trace) => [trace])).flatMap((members) => {
    const assessment = pickComparisonFeedback(members, info.name, info.scope);
    return assessment ? [assessment] : [];
  });

const getValidAssessmentValue = (assessment: FeedbackAssessment): AssessmentValue | undefined => {
  const value = getAssessmentValue(assessment);
  return getAssessmentError(assessment) ||
    value === null ||
    value === undefined ||
    (typeof value === 'number' && !isFiniteValue(value))
    ? undefined
    : value;
};

/** Scope is stable across filtering. Only the retained source values determine the displayed type. */
export const getSessionComparisonAssessmentInfos = (
  allTraces: ModelTraceInfoV3[],
  rows: TracesComparisonRow[],
): SessionComparisonAssessmentInfo[] => {
  const scopes = new Map<string, SessionComparisonAssessmentInfo['scope']>();
  for (const trace of allTraces) {
    for (const assessment of trace.assessments ?? []) {
      if (
        !isFeedbackAssessment(assessment) ||
        !(isAssessmentColumnEligible(assessment) || isSessionAssessmentColumnEligible(assessment))
      ) {
        continue;
      }
      const name = assessment.assessment_name;
      scopes.set(name, isSessionLevelAssessment(assessment) || scopes.get(name) === 'session' ? 'session' : 'trace');
    }
  }
  return [...scopes]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([name, scope]) => {
      const values = rows.flatMap((row) =>
        row.runs.flatMap((run) =>
          getSourceAssessments(getComparisonRunTraces(run), { name, scope }).flatMap((assessment) => {
            const value = getValidAssessmentValue(assessment);
            return value === undefined ? [] : [value];
          }),
        ),
      );
      const dtype =
        values.length && values.every(isFiniteValue)
          ? 'numeric'
          : values.length && values.every((value) => typeof value === 'boolean')
            ? 'boolean'
            : values.length &&
                values.every(
                  (value) =>
                    value === KnownEvaluationResultAssessmentStringValue.YES ||
                    value === KnownEvaluationResultAssessmentStringValue.NO,
                )
              ? 'pass-fail'
              : 'string';
      return { name, scope, dtype };
    });
};

export interface SessionAssessmentSummary {
  value?: number;
  values: AssessmentValue[];
  assessments: FeedbackAssessment[];
  hasError: boolean;
}

export const getSessionAssessmentSummary = (
  traces: ModelTraceInfoV3[],
  info: SessionComparisonAssessmentInfo,
): SessionAssessmentSummary => {
  const assessments = getSourceAssessments(traces, info);
  const values = assessments.flatMap((assessment) => {
    const value = getValidAssessmentValue(assessment);
    return value === undefined ? [] : [value];
  });
  const samples = values.flatMap((value) => {
    if (info.dtype === 'numeric') {
      return isFiniteValue(value) ? [value] : [];
    }
    if (info.dtype === 'boolean') {
      return typeof value === 'boolean' ? [Number(value)] : [];
    }
    if (info.dtype === 'pass-fail') {
      return value === KnownEvaluationResultAssessmentStringValue.YES
        ? [1]
        : value === KnownEvaluationResultAssessmentStringValue.NO
          ? [0]
          : [];
    }
    return [];
  });
  return {
    value: averageValues(samples),
    values,
    assessments,
    hasError: assessments.some((assessment) => Boolean(getAssessmentError(assessment))),
  };
};

export interface SessionComparisonRunSummary {
  start_time?: number;
  duration?: number;
  tokens?: number;
  cost?: number;
  state?: string;
  assessments: Record<string, SessionAssessmentSummary>;
}

export const getSessionComparisonSummaries = (
  rows: TracesComparisonRow[],
  infos: SessionComparisonAssessmentInfo[],
): Map<string, SessionComparisonRunSummary[]> =>
  new Map(
    rows.map((row) => [
      row.id,
      row.runs.map((run) => {
        const traces = getComparisonRunTraces(run);
        const start = traces[0] ? Date.parse(traces[0].request_time) : undefined;
        const durations = traces.map((trace) =>
          trace.execution_duration ? parseTraceDurationToMs(trace.execution_duration) : undefined,
        );
        const sum = (values: (number | undefined)[]) => {
          const valid = values.filter(isFiniteValue);
          return valid.length ? valid.reduce((total, value) => total + value, 0) : undefined;
        };
        return {
          start_time: isFiniteValue(start) ? start : undefined,
          duration:
            durations.length && durations.every(isFiniteValue)
              ? durations.reduce((total, value) => total + value, 0)
              : undefined,
          tokens: sum(traces.map((trace) => getTraceTokenUsage(trace)?.total_tokens)),
          cost: sum(traces.map((trace) => getTraceCost(trace)?.total_cost)),
          state: traces.at(-1)?.state,
          assessments: Object.fromEntries(infos.map((info) => [info.name, getSessionAssessmentSummary(traces, info)])),
        };
      }),
    ]),
  );

export const getEvalRunSessionComparisonColumnAggregates = (
  rows: TracesComparisonRow[],
  summaries: ReadonlyMap<string, SessionComparisonRunSummary[]>,
  infos: SessionComparisonAssessmentInfo[],
  intl: IntlShape,
  numRuns: number,
): Record<string, EvalRunComparisonColumnAggregate> => {
  const matrix = (getValue: (summary: SessionComparisonRunSummary) => number | undefined) =>
    rows.map((row) =>
      Array.from({ length: numRuns }, (_, index) => {
        const summary = summaries.get(row.id)?.[index];
        return summary ? getValue(summary) : undefined;
      }),
    );
  const number = (value: number) => intl.formatNumber(value, { maximumFractionDigits: 2 });
  const percent = (value: number) => intl.formatNumber(value, { style: 'percent', maximumFractionDigits: 0 });
  const rateDelta = (value: number) => intl.formatNumber(value * 100, { maximumFractionDigits: 0 });
  const numericAggregate = (
    getValue: (summary: SessionComparisonRunSummary) => number | undefined,
    format: (value: number) => string,
    options?: Parameters<typeof getNumericComparisonAggregate>[3],
  ) =>
    getNumericComparisonAggregate(matrix(getValue), numRuns, format, {
      ...options,
      includePercentDelta: true,
      // Different turn counts can produce a few ULPs of noise in equal session averages/totals.
      areValuesEqual: (left, right) =>
        Math.abs(left - right) <= 8 * Number.EPSILON * Math.max(Math.abs(left), Math.abs(right)),
    });
  return {
    ...Object.fromEntries(
      infos.map((info) => {
        const isRate = info.dtype === 'boolean' || info.dtype === 'pass-fail';
        const aggregate: EvalRunComparisonColumnAggregate =
          info.dtype === 'string'
            ? {
                values: Array.from({ length: numRuns }, (_, index) => {
                  const values = new Set(
                    rows
                      .flatMap((row) => summaries.get(row.id)?.[index]?.assessments[info.name]?.values ?? [])
                      .map((value) => JSON.stringify(value)),
                  );
                  return {
                    displayValue: values.size
                      ? intl.formatMessage(
                          {
                            defaultMessage: '{count} unique',
                            description: 'Distinct assessment values in session comparison summaries',
                          },
                          { count: values.size },
                        )
                      : '—',
                  };
                }),
              }
            : numericAggregate((summary) => summary.assessments[info.name]?.value, isRate ? percent : number, {
                higherIsBetter: true,
                label: info.dtype === 'numeric' ? 'AVG' : info.dtype === 'boolean' ? 'TRUE' : 'PASS',
                formatDelta: isRate ? rateDelta : number,
                isRoundedZero: (value) => Math.round(Math.abs(value) * 100) === 0,
              });
        return [assessmentColumnId(info.name), aggregate];
      }),
    ),
    duration: numericAggregate((summary) => summary.duration, formatTraceDurationMs, {
      formatDelta: (value) => formatTraceDurationMs(value).replace(/[a-z]+$/, ''),
    }),
    tokens: numericAggregate((summary) => summary.tokens, number),
    cost: numericAggregate((summary) => summary.cost, formatCostUSD, {
      formatDelta: (value) => formatCostUSD(value).replace('$', ''),
    }),
    state: numericAggregate(
      (summary) => (summary.state === 'ERROR' ? 1 : summary.state === 'OK' ? 0 : undefined),
      percent,
      { label: 'ERROR RATE', formatDelta: rateDelta },
    ),
  };
};

export const sortSessionComparisonRows = (
  rows: TracesComparisonRow[],
  summaries: ReadonlyMap<string, SessionComparisonRunSummary[]>,
  column: TraceSortColumnId,
  dir: SortDirection,
  infos: SessionComparisonAssessmentInfo[] = [],
): TracesComparisonRow[] => {
  const sortValue = (summary: SessionComparisonRunSummary | undefined) => {
    if (!summary) {
      return undefined;
    }
    if (isAssessmentColumnId(column)) {
      const name = assessmentNameFromColumnId(column);
      const score = summary.assessments[name];
      if (
        score?.values.length &&
        infos.some((info) => info.name === name && info.scope === 'session' && info.dtype === 'string')
      ) {
        const value = score.values[0];
        return typeof value === 'string' ? value : JSON.stringify(value);
      }
      return (
        score?.value ??
        (score?.values.length ? new Set(score.values.map((value) => JSON.stringify(value))).size : undefined)
      );
    }
    switch (column) {
      case 'start_time':
        return summary.start_time;
      case 'duration':
        return summary.duration;
      case 'tokens':
        return summary.tokens;
      case 'cost':
        return summary.cost;
      case 'state':
        return summary.state;
      default:
        return undefined;
    }
  };
  return [...rows].sort((a, b) => {
    for (let index = 0; index < Math.max(a.runs.length, b.runs.length); index++) {
      const difference = compareComparisonSortValues(
        sortValue(summaries.get(a.id)?.[index]),
        sortValue(summaries.get(b.id)?.[index]),
        dir,
      );
      if (difference) {
        return difference;
      }
    }
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
};
