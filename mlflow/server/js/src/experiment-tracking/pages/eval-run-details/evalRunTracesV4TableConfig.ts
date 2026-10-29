import { isSessionLevelAssessment, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import type { TraceColumnId } from '@databricks/web-shared/traces-table';
import { assessmentNameFromColumnId } from '../../components/experiment-page/components/traces-v4/utils/assessmentColumns';

export const EVAL_RUN_DEFAULT_COLUMN_SIZING = { input: 240, output: 240 } as const;

export const buildEvalRunTableColumnOrder = (
  standardColumnOrder: readonly TraceColumnId[],
  assessmentColumnIds: readonly string[],
  traces: readonly ModelTraceInfoV3[],
  knownFeedbackAssessmentNames: Iterable<string> = [],
): string[] => {
  const feedbackNames = new Set(knownFeedbackAssessmentNames);
  const expectationNames = new Set<string>();
  for (const trace of traces) {
    for (const assessment of trace.assessments ?? []) {
      if (isSessionLevelAssessment(assessment)) {
        continue;
      }
      if ('feedback' in assessment) {
        feedbackNames.add(assessment.assessment_name);
      } else if ('expectation' in assessment) {
        expectationNames.add(assessment.assessment_name);
      }
    }
  }

  const feedbackIds: string[] = [];
  const expectationIds: string[] = [];
  const otherIds: string[] = [];
  for (const id of assessmentColumnIds) {
    const name = assessmentNameFromColumnId(id);
    if (feedbackNames.has(name)) {
      feedbackIds.push(id);
    } else if (expectationNames.has(name)) {
      expectationIds.push(id);
    } else {
      otherIds.push(id);
    }
  }

  return [
    ...new Set([
      'input',
      'output',
      'start_time',
      'duration',
      'tokens',
      ...feedbackIds,
      ...expectationIds,
      ...otherIds,
      ...standardColumnOrder,
    ]),
  ];
};
