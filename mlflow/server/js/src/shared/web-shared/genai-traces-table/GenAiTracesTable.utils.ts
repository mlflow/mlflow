import type { ColumnDef } from '@tanstack/react-table';
import { isNil } from 'lodash';

import { KnownEvaluationResultAssessmentName } from './enum';
import {
  REQUEST_TIME_COLUMN_ID,
  STATE_COLUMN_ID,
  SOURCE_COLUMN_ID,
  TAGS_COLUMN_ID,
  EXECUTION_DURATION_COLUMN_ID,
  TRACE_NAME_COLUMN_ID,
  SORTABLE_INFO_COLUMNS,
  TRACE_ID_COLUMN_ID,
  SESSION_COLUMN_ID,
  INPUTS_COLUMN_ID,
  RESPONSE_COLUMN_ID,
  USER_COLUMN_ID,
  LOGGED_MODEL_COLUMN_ID,
  TOKENS_COLUMN_ID,
} from './hooks/useTableColumns';
import { TracesTableColumnGroup, TracesTableColumnType } from './types';
import type { TracesTableColumn, EvalTraceComparisonEntry, RunEvaluationTracesDataEntry, TraceInfoV3 } from './types';
import { getTraceInfoInputs, shouldUseTraceInfoV3 } from './utils/TraceUtils';

const GROUP_PRIORITY = [
  TracesTableColumnGroup.INFO,
  TracesTableColumnGroup.ASSESSMENT,
  TracesTableColumnGroup.EXPECTATION,
  TracesTableColumnGroup.TAG,
] as const;

/** Preferred order *within* the INFO group by column ID */
const INFO_COLUMN_PRIORITY = [
  TRACE_ID_COLUMN_ID,
  INPUTS_COLUMN_ID,
  RESPONSE_COLUMN_ID,
  SESSION_COLUMN_ID,
  USER_COLUMN_ID,
  TRACE_NAME_COLUMN_ID,
  LOGGED_MODEL_COLUMN_ID,
  TOKENS_COLUMN_ID,
] as const;

/** Preferred order *within* the ASSESSMENT group by column ID */
const ASSESSMENT_COLUMN_PRIORITY = [KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT] as const;

const groupRank: Record<TracesTableColumnGroup, number> = Object.fromEntries(
  GROUP_PRIORITY.map((grp, idx) => [grp, idx]),
) as any;

const infoColumnRank: Record<string, number> = Object.fromEntries(INFO_COLUMN_PRIORITY.map((id, idx) => [id, idx]));

const assessmentColumnRank: Record<string, number> = Object.fromEntries(
  ASSESSMENT_COLUMN_PRIORITY.map((id, idx) => [id, idx]),
);

export function sortGroupedColumns(columns: TracesTableColumn[], isComparing?: boolean): TracesTableColumn[] {
  return [...columns].sort((colA, colB) => {
    // If comparing, always put request time column first
    if (isComparing) {
      if (colA.id === INPUTS_COLUMN_ID) return -1;
      if (colB.id === INPUTS_COLUMN_ID) return 1;
    }

    // 1) Compare their groups by precomputed rank
    const groupA = colA.group ?? TracesTableColumnGroup.INFO;
    const groupB = colB.group ?? TracesTableColumnGroup.INFO;
    const groupComparison = groupRank[groupA] - groupRank[groupB];
    if (groupComparison !== 0) return groupComparison;

    // 2) Same group: INFO
    if (groupA === TracesTableColumnGroup.INFO) {
      const rankA = infoColumnRank[colA.id] ?? Infinity;
      const rankB = infoColumnRank[colB.id] ?? Infinity;
      if (rankA !== rankB) return rankA - rankB;
      return colA.label.localeCompare(colB.label);
    }

    // 3) Same group: ASSESSMENT
    if (groupA === TracesTableColumnGroup.ASSESSMENT) {
      const rankA = assessmentColumnRank[colA.id] ?? Infinity;
      const rankB = assessmentColumnRank[colB.id] ?? Infinity;
      if (rankA !== rankB) return rankA - rankB;
      return colA.label.localeCompare(colB.label);
    }

    // 4) Same group: EXPECTATION
    if (groupA === TracesTableColumnGroup.EXPECTATION) {
      return colA.label.localeCompare(colB.label);
    }

    // 5) Same group: TAG (or any other fallback)
    return colA.label.localeCompare(colB.label);
  });
}

export const sortColumns = (columns: ColumnDef<EvalTraceComparisonEntry>[], selectedColumns: TracesTableColumn[]) => {
  return columns.sort((a, b) => {
    const getPriority = (col: typeof a) => {
      const colType = selectedColumns.find((c) => c.id === col.id)?.type;

      if (colType === TracesTableColumnType.INPUT) return 1;
      if (col.id === TRACE_NAME_COLUMN_ID) return 2;
      if (colType === TracesTableColumnType.TRACE_INFO) return 3;
      if (colType === TracesTableColumnType.INTERNAL_MONITOR_REQUEST_TIME) return 4;
      if (colType === TracesTableColumnType.ASSESSMENT) return 5;
      return 999; // keep any other columns after the known ones
    };

    // primary sort key: our priority number
    const diff = getPriority(a) - getPriority(b);
    if (diff !== 0) return diff;

    // secondary key: for assessment columns, prioritize 'Overall' and then sort alphabetically by label
    const aCol = selectedColumns.find((c) => c.id === a.id);
    const bCol = selectedColumns.find((c) => c.id === b.id);
    if (aCol?.type === TracesTableColumnType.ASSESSMENT && bCol?.type === TracesTableColumnType.ASSESSMENT) {
      if (aCol.id === KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT) return -1;
      if (bCol.id === KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT) return 1;
      return (aCol.label || '').localeCompare(bCol.label || '');
    }

    // tertiary key: original array order (stable sort fallback)
    return 0;
  });
};

export const traceInfoSortingFn = (
  traceInfoA: TraceInfoV3 | undefined,
  traceInfoB: TraceInfoV3 | undefined,
  colId: string,
) => {
  // only support sorting by request time for now
  if (!SORTABLE_INFO_COLUMNS.includes(colId)) {
    return 0;
  }

  const aVal = String(getTraceInfoValueWithColId(traceInfoA as TraceInfoV3, colId) ?? '');
  const bVal = String(getTraceInfoValueWithColId(traceInfoB as TraceInfoV3, colId) ?? '');

  return aVal.localeCompare(bVal, undefined, { numeric: true });
};

export const getTraceInfoValueWithColId = (traceInfo: TraceInfoV3, colId: string) => {
  switch (colId) {
    case REQUEST_TIME_COLUMN_ID:
    case EXECUTION_DURATION_COLUMN_ID:
    case TAGS_COLUMN_ID:
    case STATE_COLUMN_ID:
      return traceInfo[colId];
    case SOURCE_COLUMN_ID:
      return traceInfo.tags;
    case TRACE_ID_COLUMN_ID:
      return traceInfo.trace_id;
    case SESSION_COLUMN_ID:
      return traceInfo.tags?.['mlflow.trace.session'];
    default:
      throw new Error(`Unknown column id: ${colId}`);
  }
};

function getUniqueInputRequests(
  evaluationResults: RunEvaluationTracesDataEntry[],
): Map<string, RunEvaluationTracesDataEntry> {
  const resultMap = new Map<string, RunEvaluationTracesDataEntry>();
  // If there are duplicate input ids, we need to append a count to the key to ensure uniqueness.
  const duplicateIndexMap = new Map<string, number>();

  evaluationResults?.forEach((entry) => {
    let key = shouldUseTraceInfoV3([entry]) ? getTraceInfoInputs(entry.traceInfo as TraceInfoV3) : entry.inputsId;
    if (resultMap.has(key)) {
      const currentCount = duplicateIndexMap.get(entry.inputsId) || 0;
      const newCount = currentCount + 1;
      duplicateIndexMap.set(entry.inputsId, newCount);
      key = `${entry.inputsId}_${newCount}`;
    }
    resultMap.set(key, entry);
  });
  return resultMap;
}

export function computeEvaluationsComparison(
  currentRunEvalResults: RunEvaluationTracesDataEntry[],
  otherRunEvalResults?: RunEvaluationTracesDataEntry[],
): EvalTraceComparisonEntry[] {
  if (isNil(otherRunEvalResults)) {
    return currentRunEvalResults.map((entry) => ({ currentRunValue: entry }));
  }

  // TODO(nsthorat): This logic does not work when a single eval run contains the same input ids, e.g. there is multiple evals with the same
  // input id. This is a bug in the current implementation.

  // Merge the two eval results by joining on inputsId. There may be results that are only present in one of the two.
  const otherRunEvalResultsMap = getUniqueInputRequests(otherRunEvalResults);

  const currentRunEvalResultsMap = getUniqueInputRequests(currentRunEvalResults);
  const allRequestIds = new Set([...currentRunEvalResultsMap.keys(), ...otherRunEvalResultsMap.keys()]);

  return Array.from(allRequestIds).map((inputsId) => {
    return {
      currentRunValue: currentRunEvalResultsMap.get(inputsId),
      otherRunValue: otherRunEvalResultsMap.get(inputsId),
    };
  });
}
