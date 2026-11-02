import type { IntlShape } from '@databricks/i18n';
import {
  getAssessmentValue,
  type Assessment,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
import {
  getTextColumnMaxSize,
  type TraceComparisonRow,
  type TraceTableColumn,
} from '@databricks/web-shared/traces-table';
import {
  assessmentNameFromColumnId,
  getAssessmentColumnType,
  isAssessmentColumnId,
  pickCellAssessment,
} from '../../components/experiment-page/components/traces-v4/utils/assessmentColumns';
import { PAIRWISE_PREFERENCE_ASSESSMENT_NAME } from '../../components/experiment-page/components/traces-v4/utils/pairwiseReview';
import {
  EvalRunComparisonAssessmentCell,
  formatEvalRunComparisonAssessmentDelta,
  type EvalRunComparisonAssessmentDelta,
} from './EvalRunComparisonAssessmentCell';

// The width of TraceAssessmentCell's score bar track.
const SCORE_BAR_TRACK_WIDTH = 48;
// Score bar, the two inline gaps, and the delta tag's padding and trailing margin.
const DELTA_CELL_CHROME_PX = SCORE_BAR_TRACK_WIDTH + 2 * 4 + 24;

const getAssessmentError = (assessment: Assessment) =>
  assessment.error ?? ('feedback' in assessment ? assessment.feedback?.error : undefined);

const getNumericCellValue = (trace: ModelTraceInfoV3 | undefined, assessmentName: string): number | undefined => {
  const assessment = trace ? pickCellAssessment(trace, assessmentName) : undefined;
  if (!assessment || getAssessmentError(assessment)) {
    return undefined;
  }
  const value = getAssessmentValue(assessment);
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
};

/** The difference from the baseline score a cell shows, or none when either score is unavailable or it rounds to 0. */
export const getEvalRunComparisonAssessmentDelta = (
  trace: ModelTraceInfoV3 | undefined,
  baselineTrace: ModelTraceInfoV3 | undefined,
  assessmentName: string,
): EvalRunComparisonAssessmentDelta | undefined => {
  const value = getNumericCellValue(trace, assessmentName);
  const baselineValue = getNumericCellValue(baselineTrace, assessmentName);
  if (value === undefined || baselineValue === undefined) {
    return undefined;
  }
  const difference = value - baselineValue;
  // The pill shows at most two decimals, so a smaller difference would read as no change.
  return Math.round(Math.abs(difference) * 100) === 0 ? undefined : { value, baselineValue, difference };
};

/**
 * Copies the numeric scorer columns so each non-baseline comparison cell shows its change from run slot 0, the
 * baseline. `traces` must be the set the assessment columns were typed from, so pills appear only beside score bars.
 */
export const withEvalRunComparisonAssessmentDeltas = (
  columnDefs: TraceTableColumn[],
  traces: ModelTraceInfoV3[],
  rows: TraceComparisonRow[],
  intl: IntlShape,
): TraceTableColumn[] =>
  columnDefs.map((column) => {
    if (!column.id || !isAssessmentColumnId(column.id)) {
      return column;
    }
    const assessmentName = assessmentNameFromColumnId(column.id);
    if (
      assessmentName === PAIRWISE_PREFERENCE_ASSESSMENT_NAME ||
      getAssessmentColumnType(traces, assessmentName) !== 'numeric'
    ) {
      return column;
    }
    const deltaTexts = rows.flatMap((row) =>
      row.runs.slice(1).flatMap(({ trace }) => {
        const delta = getEvalRunComparisonAssessmentDelta(trace, row.runs[0]?.trace, assessmentName);
        return delta
          ? [`${delta.value.toFixed(2)} ${formatEvalRunComparisonAssessmentDelta(intl, delta.difference)}`]
          : [];
      }),
    );
    const deltaCellWidth = deltaTexts.length ? getTextColumnMaxSize(deltaTexts, 0, DELTA_CELL_CHROME_PX) : undefined;
    return {
      ...column,
      ...(deltaCellWidth !== undefined && {
        size: Math.max(column.size ?? 0, deltaCellWidth),
        maxSize: Math.max(column.maxSize ?? 0, deltaCellWidth),
      }),
      renderComparisonCell: ({ trace, row, runIndex }) => (
        <EvalRunComparisonAssessmentCell
          trace={trace}
          assessmentName={assessmentName}
          delta={
            runIndex > 0 ? getEvalRunComparisonAssessmentDelta(trace, row.runs[0]?.trace, assessmentName) : undefined
          }
        />
      ),
    };
  });
