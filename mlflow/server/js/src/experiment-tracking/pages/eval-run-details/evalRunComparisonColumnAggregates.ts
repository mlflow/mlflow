import type { IntlShape } from '@databricks/i18n';
import {
  convertTraceInfoV3ToRunEvalEntry,
  KnownEvaluationResultAssessmentStringValue,
  type AssessmentInfo,
  type EvalTraceComparisonEntry,
  type RunEvaluationResultAssessment,
} from '@databricks/web-shared/genai-traces-table';
// Not re-exported from the genai-traces-table barrel in OSS — import from their modules.
import {
  getAssessmentAggregateOverallFraction,
  getAssessmentAggregatesMulti,
} from '@databricks/web-shared/genai-traces-table/utils/AggregationUtils';
import { getDisplayScoreAndChangeForRun } from '@databricks/web-shared/genai-traces-table/utils/DisplayUtils';
import { getEvaluationResultAssessmentValue } from '@databricks/web-shared/genai-traces-table/components/GenAiEvaluationTracesReview.utils';
import {
  ASSESSMENT_SESSION_METADATA_KEY,
  formatCostUSD,
  getTraceCost,
  getTraceTokenUsage,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
import {
  formatTraceDurationMs,
  parseTraceDurationToMs,
  type TraceComparisonRow,
} from '@databricks/web-shared/traces-table';
import {
  assessmentColumnId,
  assessmentNameFromColumnId,
  isAssessmentColumnId,
} from '../../components/experiment-page/components/traces-v4/utils/assessmentColumns';
import { getEvalRunComparisonRelativeChangePercent } from './EvalRunComparisonDelta';

export type NumericAggregateMethod = 'mean' | 'median' | 'min' | 'max';
export type NumericAggregateMethods = Readonly<Partial<Record<string, NumericAggregateMethod>>>;
export type EvalRunComparisonOutcome = 'regression' | 'improvement';

export interface EvalRunComparisonColumnAggregate {
  label?: 'AVG' | 'PASS' | 'TRUE' | 'ERROR RATE' | 'MEDIAN' | 'MIN' | 'MAX' | 'WIN / LOSE';
  values: {
    displayValue: string;
    displayDelta?: string;
    delta?: number;
    percentDelta?: number;
    changeCounts?: { improved: number; regressed: number };
  }[];
}

const isFiniteValue = (value: number | null | undefined): value is number =>
  typeof value === 'number' && Number.isFinite(value);

export const getPercentDelta = (value: number | undefined, baseline: number | undefined): number | undefined => {
  if (!isFiniteValue(value) || !isFiniteValue(baseline)) return undefined;
  const percentDelta = getEvalRunComparisonRelativeChangePercent(value, baseline);
  if (!isFiniteValue(percentDelta)) return undefined;
  const roundedPercentDelta = Number(percentDelta.toFixed(2));
  return Object.is(roundedPercentDelta, -0) ? 0 : roundedPercentDelta;
};

const average = (values: number[]): number | undefined =>
  values.length ? values.reduce((sum, value) => sum + value / values.length, 0) : undefined;

const comparisonEntryCache = new WeakMap<ModelTraceInfoV3, ReturnType<typeof convertTraceInfoV3ToRunEvalEntry>>();
const getComparisonEntry = (trace: ModelTraceInfoV3) => {
  let entry = comparisonEntryCache.get(trace);
  if (!entry) {
    entry = convertTraceInfoV3ToRunEvalEntry(trace);
    comparisonEntryCache.set(trace, entry);
  }
  return entry;
};

export const getNumericAggregateValue = (values: number[], method: NumericAggregateMethod): number | undefined => {
  if (!values.length) {
    return undefined;
  }
  if (method === 'mean') {
    return average(values);
  }
  const sorted = [...values].sort((left, right) => left - right);
  if (method === 'min') {
    return sorted[0];
  }
  if (method === 'max') {
    return sorted[sorted.length - 1];
  }
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

const getNumericAggregateMethod = (methods: NumericAggregateMethods | undefined, columnId: string) =>
  methods?.[columnId] ?? 'mean';

const numericAggregateLabels: Record<NumericAggregateMethod, NonNullable<EvalRunComparisonColumnAggregate['label']>> = {
  mean: 'AVG',
  median: 'MEDIAN',
  min: 'MIN',
  max: 'MAX',
};

const getNumericAggregateLabel = (method: NumericAggregateMethod): EvalRunComparisonColumnAggregate['label'] =>
  numericAggregateLabels[method];

const getComparisonChangeCounts = (
  valuesByRow: (number | undefined)[][],
  numRuns: number,
  higherIsBetter: boolean,
  areValuesEqual?: (left: number, right: number) => boolean,
): EvalRunComparisonColumnAggregate['values'][number]['changeCounts'][] =>
  Array.from({ length: numRuns }, (_, runIndex) => {
    if (runIndex === 0) {
      return undefined;
    }
    const counts = { improved: 0, regressed: 0 };
    for (const values of valuesByRow) {
      const baseline = values[0];
      const candidate = values[runIndex];
      if (
        !isFiniteValue(baseline) ||
        !isFiniteValue(candidate) ||
        candidate === baseline ||
        areValuesEqual?.(baseline, candidate)
      ) {
        continue;
      }
      const increased = candidate > baseline;
      if (increased === higherIsBetter) {
        counts.improved++;
      } else {
        counts.regressed++;
      }
    }
    return counts.improved || counts.regressed ? counts : undefined;
  });

const isValidAssessment = (assessment: RunEvaluationResultAssessment, info: AssessmentInfo): boolean => {
  if (assessment.metadata?.[ASSESSMENT_SESSION_METADATA_KEY] || assessment.errorCode || assessment.errorMessage) {
    return false;
  }
  const value = getEvaluationResultAssessmentValue(assessment);
  if (value === undefined || (typeof value === 'number' && !Number.isFinite(value))) {
    return false;
  }
  switch (info.dtype) {
    case 'numeric':
      return typeof value === 'number';
    case 'boolean':
      return typeof value === 'boolean';
    case 'pass-fail':
      return (
        value === KnownEvaluationResultAssessmentStringValue.YES ||
        value === KnownEvaluationResultAssessmentStringValue.NO
      );
    case 'string':
      return true;
    default:
      return false;
  }
};

const getAssessmentRowValue = (assessments: RunEvaluationResultAssessment[], info: AssessmentInfo) => {
  const values = assessments
    .filter((assessment) => isValidAssessment(assessment, info))
    .map((assessment) => {
      const value = getEvaluationResultAssessmentValue(assessment);
      if (typeof value === 'number') {
        return value;
      }
      return value === true || value === KnownEvaluationResultAssessmentStringValue.YES ? 1 : 0;
    });
  return average(values);
};

const getMetricComparisonValue = (trace: ModelTraceInfoV3, columnId: string): number | undefined => {
  const value =
    columnId === 'state'
      ? trace.state === 'ERROR'
        ? 1
        : trace.state === 'OK'
          ? 0
          : undefined
      : columnId === 'duration'
        ? trace.execution_duration
          ? parseTraceDurationToMs(trace.execution_duration)
          : undefined
        : columnId === 'tokens'
          ? getTraceTokenUsage(trace)?.total_tokens
          : columnId === 'cost'
            ? getTraceCost(trace)?.total_cost
            : undefined;
  return isFiniteValue(value) && value >= 0 ? value : undefined;
};

const getMetricAggregate = (
  rows: TraceComparisonRow[],
  numRuns: number,
  getValue: (trace: ModelTraceInfoV3) => number | null | undefined,
  format: (value: number) => string,
  options?: {
    label?: EvalRunComparisonColumnAggregate['label'];
    formatDelta?: (value: number) => string;
    includePercentDelta?: boolean;
  },
): EvalRunComparisonColumnAggregate => {
  const valuesByRow = rows.map((row) =>
    Array.from({ length: numRuns }, (_, runIndex) => {
      const trace = row.runs[runIndex]?.trace;
      const value = trace ? getValue(trace) : undefined;
      return isFiniteValue(value) && value >= 0 ? value : undefined;
    }),
  );
  return getNumericComparisonAggregate(valuesByRow, numRuns, format, options);
};

/** Each matrix row is one comparison group, regardless of the number of turns it contains. */
export const getNumericComparisonAggregate = (
  valuesByRow: (number | undefined)[][],
  numRuns: number,
  format: (value: number) => string,
  options?: {
    label?: EvalRunComparisonColumnAggregate['label'];
    formatDelta?: (value: number) => string;
    higherIsBetter?: boolean;
    isRoundedZero?: (value: number) => boolean;
    areValuesEqual?: (left: number, right: number) => boolean;
    method?: NumericAggregateMethod;
    includePercentDelta?: boolean;
  },
): EvalRunComparisonColumnAggregate => {
  const method = options?.method ?? 'mean';
  const aggregateValues = Array.from({ length: numRuns }, (_, runIndex) =>
    getNumericAggregateValue(valuesByRow.map((values) => values[runIndex]).filter(isFiniteValue), method),
  );
  const changeCounts = getComparisonChangeCounts(
    valuesByRow,
    numRuns,
    options?.higherIsBetter ?? false,
    options?.areValuesEqual,
  );
  const baseline = aggregateValues[0];
  const formatDelta = options?.formatDelta ?? format;
  return {
    label: method === 'mean' ? (options?.label ?? 'AVG') : getNumericAggregateLabel(method),
    values: aggregateValues.map((value, index) => {
      const delta = index > 0 && isFiniteValue(value) && isFiniteValue(baseline) ? value - baseline : undefined;
      const deltaMagnitude = delta ? formatDelta(Math.abs(delta)) : undefined;
      const isRoundedZero =
        delta !== undefined &&
        (options?.isRoundedZero ? options.isRoundedZero(delta) : deltaMagnitude === formatDelta(0));
      return {
        displayValue: isFiniteValue(value) ? format(value) : '—',
        displayDelta: delta && !isRoundedZero ? `${delta > 0 ? '+' : '-'}${deltaMagnitude}` : undefined,
        ...(options?.includePercentDelta && delta !== undefined ? { delta: isRoundedZero ? 0 : delta } : {}),
        ...(options?.includePercentDelta && index > 0 && isFiniteValue(value) && isFiniteValue(baseline)
          ? { percentDelta: isRoundedZero ? 0 : getPercentDelta(value, baseline) }
          : {}),
        changeCounts: changeCounts[index],
      };
    }),
  };
};

type ComparisonEntry = ReturnType<typeof getComparisonEntry>;

/** A trace's entry holding only the scorer's valid assessments, as the shared aggregators read it. */
const getValidScorerEntry = (entry: ComparisonEntry, info: AssessmentInfo): ComparisonEntry => ({
  ...entry,
  overallAssessments: entry.overallAssessments.filter((assessment) => isValidAssessment(assessment, info)),
  responseAssessmentsByName: {
    [info.name]: (entry.responseAssessmentsByName[info.name] ?? []).filter((assessment) =>
      isValidAssessment(assessment, info),
    ),
  },
});

const getScorerAssessments = (entry: ComparisonEntry | undefined, info: AssessmentInfo) =>
  (info.isOverall ? entry?.overallAssessments : entry?.responseAssessmentsByName[info.name]) ?? [];

/** Rows are the complete search-retained join, never the paginated or trace-ID-deduplicated view. */
export const getEvalRunComparisonColumnAggregates = (
  rows: TraceComparisonRow[],
  assessmentInfos: AssessmentInfo[],
  intl: IntlShape,
  numRuns: number,
  aggregateMethods?: NumericAggregateMethods,
  includePercentDelta = false,
): Record<string, EvalRunComparisonColumnAggregate> => {
  const entries = rows.map((row) => row.runs.map(({ trace }) => (trace ? getComparisonEntry(trace) : undefined)));
  const assessmentAggregates = assessmentInfos.map((info) => {
    // Copies keep invalid samples out of this consumer's summaries without changing shared caches/helpers.
    const sanitized: EvalTraceComparisonEntry[] = entries.map((runValues) => ({
      runValues: runValues.map((entry) => (entry ? getValidScorerEntry(entry, info) : undefined)),
    }));
    // Each run's valid values for this scorer, extracted once for the unique values, numeric aggregates
    // and change counts below.
    const valuesByRun = Array.from({ length: numRuns }, (_, runIndex) =>
      sanitized.flatMap(({ runValues }) =>
        getScorerAssessments(runValues?.[runIndex], info).map(getEvaluationResultAssessmentValue),
      ),
    );
    const uniqueValuesByRun = valuesByRun.map((values) => new Set(values));
    const sanitizedInfo = { ...info, uniqueValues: new Set(valuesByRun.flat()) };
    const aggregates = getAssessmentAggregatesMulti(sanitizedInfo, sanitized, [], numRuns);
    const method = getNumericAggregateMethod(aggregateMethods, assessmentColumnId(info.name));
    const aggregateValuesByRun =
      info.dtype === 'numeric'
        ? valuesByRun.map((values) =>
            getNumericAggregateValue(
              values.filter((value): value is number => typeof value === 'number' && Number.isFinite(value)),
              method,
            ),
          )
        : info.dtype === 'boolean' || info.dtype === 'pass-fail'
          ? aggregates.countsByRun?.map((counts) => getAssessmentAggregateOverallFraction(info, counts))
          : undefined;
    const baselineValue = aggregateValuesByRun?.[0];
    const isScored = info.dtype === 'numeric' || info.dtype === 'boolean' || info.dtype === 'pass-fail';
    const changeCounts = getComparisonChangeCounts(
      sanitized.map(({ runValues }) =>
        // Each matched trace contributes once, even when several valid assessments share the name.
        (runValues ?? []).map((entry) =>
          entry && isScored ? getAssessmentRowValue(getScorerAssessments(entry, info), info) : undefined,
        ),
      ),
      numRuns,
      true,
    );
    const values = uniqueValuesByRun.map((uniqueValues, index) => {
      if (!uniqueValues.size) {
        return { displayValue: '—' };
      }
      const value = aggregateValuesByRun?.[index];
      const delta =
        index > 0 && uniqueValuesByRun[0]?.size && isFiniteValue(value) && isFiniteValue(baselineValue)
          ? value - baselineValue
          : 0;
      // Match the shared formatters' precision without depending on their display strings.
      const roundedDeltaMagnitude =
        info.dtype === 'numeric' ? Math.round(Math.abs(delta) * 100) / 100 : Number((Math.abs(delta) * 100).toFixed(0));
      const displayAggregates =
        info.dtype === 'numeric' ? { ...aggregates, numericAverageByRun: aggregateValuesByRun } : aggregates;
      const display = getDisplayScoreAndChangeForRun(
        intl,
        { ...sanitizedInfo, uniqueValues },
        displayAggregates,
        index,
      );
      return {
        displayValue: display.displayScore,
        displayDelta: roundedDeltaMagnitude ? display.displayScoreChange?.replace('%', '') || undefined : undefined,
        ...(includePercentDelta && index > 0 && isFiniteValue(value) && isFiniteValue(baselineValue)
          ? { delta: roundedDeltaMagnitude ? delta : 0 }
          : {}),
        ...(includePercentDelta && index > 0 && isFiniteValue(value) && isFiniteValue(baselineValue)
          ? { percentDelta: roundedDeltaMagnitude ? getPercentDelta(value, baselineValue) : 0 }
          : {}),
        changeCounts: changeCounts[index],
      };
    });
    const aggregate: EvalRunComparisonColumnAggregate = {
      label:
        info.dtype === 'numeric'
          ? getNumericAggregateLabel(method)
          : info.dtype === 'pass-fail'
            ? 'PASS'
            : info.dtype === 'boolean'
              ? 'TRUE'
              : undefined,
      values,
    };
    return [assessmentColumnId(info.name), aggregate];
  });
  return {
    ...Object.fromEntries(assessmentAggregates),
    state: getMetricAggregate(
      rows,
      numRuns,
      (trace) => getMetricComparisonValue(trace, 'state'),
      (value) => intl.formatNumber(value, { style: 'percent', maximumFractionDigits: 0 }),
      {
        label: 'ERROR RATE',
        formatDelta: (value) => intl.formatNumber(value * 100, { maximumFractionDigits: 0 }),
        includePercentDelta,
      },
    ),
    duration: getMetricAggregate(
      rows,
      numRuns,
      (trace) => getMetricComparisonValue(trace, 'duration'),
      formatTraceDurationMs,
      { formatDelta: (value) => formatTraceDurationMs(value).replace(/[a-z]+$/, ''), includePercentDelta },
    ),
    tokens: getMetricAggregate(
      rows,
      numRuns,
      (trace) => getMetricComparisonValue(trace, 'tokens'),
      (value) => intl.formatNumber(value, { maximumFractionDigits: 2 }),
      { includePercentDelta },
    ),
    cost: getMetricAggregate(rows, numRuns, (trace) => getMetricComparisonValue(trace, 'cost'), formatCostUSD, {
      formatDelta: (value) => formatCostUSD(value).replace('$', ''),
      includePercentDelta,
    }),
  };
};

const getAssessmentComparisonValue = (trace: ModelTraceInfoV3, info: AssessmentInfo): number | undefined => {
  if (info.dtype !== 'numeric' && info.dtype !== 'boolean' && info.dtype !== 'pass-fail') {
    return undefined;
  }
  const entry = getComparisonEntry(trace);
  const assessments = info.isOverall ? entry.overallAssessments : entry.responseAssessmentsByName[info.name];
  return getAssessmentRowValue(assessments ?? [], info);
};

const getComparisonColumnValue = (
  trace: ModelTraceInfoV3 | undefined,
  columnId: string,
  assessmentInfos: AssessmentInfo[],
): number | undefined => {
  if (!trace) {
    return undefined;
  }
  if (isAssessmentColumnId(columnId)) {
    const info = assessmentInfos.find((candidate) => candidate.name === assessmentNameFromColumnId(columnId));
    return info ? getAssessmentComparisonValue(trace, info) : undefined;
  }
  return getMetricComparisonValue(trace, columnId);
};

/** Returns the baseline-relative outcome used by comparison-row outcome filters. */
export const getEvalRunComparisonOutcome = (
  row: TraceComparisonRow,
  columnId: string,
  runUuid: string,
  assessmentInfos: AssessmentInfo[],
): EvalRunComparisonOutcome | undefined => {
  const candidateIndex = row.runs.findIndex((run) => run.runUuid === runUuid);
  if (candidateIndex <= 0) {
    return undefined;
  }
  const baselineValue = getComparisonColumnValue(row.runs[0]?.trace, columnId, assessmentInfos);
  const candidateValue = getComparisonColumnValue(row.runs[candidateIndex]?.trace, columnId, assessmentInfos);
  if (!isFiniteValue(baselineValue) || !isFiniteValue(candidateValue) || candidateValue === baselineValue) {
    return undefined;
  }
  const higherIsBetter = isAssessmentColumnId(columnId);
  const candidateIsHigher = candidateValue > baselineValue;
  return candidateIsHigher === higherIsBetter ? 'improvement' : 'regression';
};
