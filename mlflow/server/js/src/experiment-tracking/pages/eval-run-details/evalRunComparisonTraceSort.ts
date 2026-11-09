import {
  getAssessmentValue,
  getTraceCost,
  getTraceTokenUsage,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
import {
  parseTraceDurationToMs,
  type SortDirection,
  type TraceComparisonRow,
  type TraceSortColumnId,
} from '@databricks/web-shared/traces-table';
import {
  assessmentNameFromColumnId,
  isAssessmentColumnId,
  pickCellAssessment,
} from '../../components/experiment-page/components/traces-v4/utils/assessmentColumns';

type SortValue = number | boolean | string | undefined;

const getSortValue = (trace: ModelTraceInfoV3 | undefined, column: TraceSortColumnId): SortValue => {
  if (!trace) {
    return undefined;
  }
  let value: unknown;
  if (isAssessmentColumnId(column)) {
    const assessment = pickCellAssessment(trace, assessmentNameFromColumnId(column));
    const error =
      assessment && (assessment.error ?? ('feedback' in assessment ? assessment.feedback?.error : undefined));
    value = assessment && !error ? getAssessmentValue(assessment) : undefined;
  } else {
    switch (column) {
      case 'start_time':
        value = Date.parse(trace.request_time);
        break;
      case 'duration':
        value = trace.execution_duration ? parseTraceDurationToMs(trace.execution_duration) : undefined;
        break;
      case 'state':
        value = trace.state;
        break;
      case 'tokens':
        value = getTraceTokenUsage(trace)?.total_tokens;
        break;
      case 'cost':
        value = getTraceCost(trace)?.total_cost;
        break;
      default:
        return undefined;
    }
  }
  return (typeof value === 'number' && Number.isFinite(value)) ||
    typeof value === 'boolean' ||
    typeof value === 'string'
    ? value
    : undefined;
};

export const compareComparisonSortValues = (left: SortValue, right: SortValue, dir: SortDirection): number => {
  if (left === undefined) {
    return right === undefined ? 0 : 1;
  }
  if (right === undefined) {
    return -1;
  }
  const rank = (value: SortValue) => (typeof value === 'number' ? 0 : typeof value === 'boolean' ? 1 : 2);
  const typeDifference = rank(left) - rank(right);
  const difference = typeDifference || (left < right ? -1 : left > right ? 1 : 0);
  return dir === 'asc' ? difference : -difference;
};

/** Baseline sorts first; subsequent slots break ties without ever reordering traces within a group. */
export const sortEvalRunComparisonRows = (
  rows: TraceComparisonRow[],
  column: TraceSortColumnId,
  dir: SortDirection,
): TraceComparisonRow[] => {
  const keyed = rows.map((row) => ({ row, values: row.runs.map(({ trace }) => getSortValue(trace, column)) }));
  keyed.sort((left, right) => {
    for (let index = 0; index < Math.max(left.values.length, right.values.length); index++) {
      const comparison = compareComparisonSortValues(left.values[index], right.values[index], dir);
      if (comparison) {
        return comparison;
      }
    }
    return left.row.id < right.row.id ? -1 : left.row.id > right.row.id ? 1 : 0;
  });
  return keyed.map(({ row }) => row);
};
