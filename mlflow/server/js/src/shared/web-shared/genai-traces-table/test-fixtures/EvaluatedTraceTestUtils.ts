import { KnownEvaluationResultAssessmentName } from '../enum';
import type {
  RunEvaluationResultAssessment,
  RunEvaluationTracesDataEntry,
  TraceInfoV3,
  AssessmentInfo,
  AssessmentDType,
  TracesTableColumn,
} from '../types';
import { TracesTableColumnType, TracesTableColumnGroup } from '../types';

export function createTestTrace(entry: {
  requestId: string;
  request: string;
  assessments: {
    name: string;
    value: string | number | boolean | null | undefined;
    dtype: 'pass-fail' | 'float' | 'boolean' | 'string';
  }[];
}): RunEvaluationTracesDataEntry {
  const assessments: RunEvaluationResultAssessment[] = entry.assessments.map((assessment) => {
    let numericValue = null;
    let booleanValue = null;
    let stringValue = null;

    switch (assessment.dtype) {
      case 'pass-fail':
      case 'string':
        stringValue = typeof assessment.value === 'string' ? assessment.value : null;
        break;
      case 'float':
        numericValue = typeof assessment.value === 'number' ? assessment.value : null;
        break;
      case 'boolean':
        booleanValue = typeof assessment.value === 'boolean' ? assessment.value : null;
        break;
    }

    return {
      name: assessment.name,
      stringValue,
      numericValue,
      booleanValue,
    };
  });

  const responseAssessmentsByName: Record<string, RunEvaluationResultAssessment[]> = {};
  for (const assessment of assessments) {
    if (!responseAssessmentsByName[assessment.name]) {
      responseAssessmentsByName[assessment.name] = [];
    }
    responseAssessmentsByName[assessment.name].push(assessment);
  }

  return {
    evaluationId: `eval_${entry.requestId}`,
    requestId: entry.requestId,
    inputs: { request: entry.request },
    inputsId: `input_${entry.requestId}`,
    outputs: {},
    targets: {},
    overallAssessments: assessments.filter((a) => a.name === KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT),
    responseAssessmentsByName,
    metrics: {},
  };
}

export function createTestTraces(
  entries: {
    requestId: string;
    request: string;
    assessments: {
      name: string;
      value: string | number | boolean | null | undefined;
      dtype: 'pass-fail' | 'float' | 'boolean' | 'string';
    }[];
  }[],
): RunEvaluationTracesDataEntry[] {
  return entries.map(createTestTrace);
}

/**
 * Helper function to create test trace info V3
 */
export const createTestTraceInfoV3 = (
  traceId: string,
  requestId: string,
  request: string,
  assessments: Array<{
    name: string;
    value: string | number | boolean;
    dtype: 'pass-fail' | 'numeric' | 'boolean' | 'string';
  }> = [],
  experimentId = 'test-experiment-id',
): TraceInfoV3 => ({
  trace_id: traceId,
  client_request_id: requestId,
  trace_location: {
    type: 'MLFLOW_EXPERIMENT',
    mlflow_experiment: { experiment_id: experimentId },
  },
  request,
  request_preview: request,
  response: 'Test response',
  response_preview: 'Test response',
  request_time: '2024-01-15T10:00:00Z',
  execution_duration: '1000',
  state: 'OK',
  trace_metadata: {},
  tags: {},
  assessments: assessments.map((assessment) => ({
    assessment_id: `${traceId}_${assessment.name}`,
    assessment_name: assessment.name,
    trace_id: traceId,
    create_time: '2024-01-15T10:00:00Z',
    last_update_time: '2024-01-15T10:00:00Z',
    feedback: {
      value: assessment.value,
    },
    source: {
      source_type: 'LLM_JUDGE',
      source_id: 'test-judge',
    },
  })),
});

/**
 * Helper function to create test assessment info
 */
export const createTestAssessmentInfo = (
  name: string,
  displayName: string,
  dtype: AssessmentDType = 'string',
): AssessmentInfo => ({
  name,
  displayName,
  isKnown: true,
  isOverall: false,
  metricName: name,
  isCustomMetric: false,
  isEditable: false,
  isRetrievalAssessment: false,
  dtype,
  uniqueValues: new Set(['test-value']),
  docsLink: '',
  missingTooltip: '',
  description: '',
});

/**
 * Helper function to create test columns
 */
export const createTestColumns = (assessmentInfos: AssessmentInfo[]): TracesTableColumn[] => {
  const columns: TracesTableColumn[] = [
    {
      id: 'request',
      label: 'Request',
      type: TracesTableColumnType.INPUT,
      group: TracesTableColumnGroup.INFO,
    },
    {
      id: 'trace_id',
      label: 'Trace ID',
      type: TracesTableColumnType.TRACE_INFO,
      group: TracesTableColumnGroup.INFO,
    },
  ];

  // Add assessment columns
  assessmentInfos.forEach((assessmentInfo) => {
    columns.push({
      id: `assessment_${assessmentInfo.name}`,
      label: assessmentInfo.displayName,
      type: TracesTableColumnType.ASSESSMENT,
      group: TracesTableColumnGroup.ASSESSMENT,
      assessmentInfo,
    });
  });

  return columns;
};
