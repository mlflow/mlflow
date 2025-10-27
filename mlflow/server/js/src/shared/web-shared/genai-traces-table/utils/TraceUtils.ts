import { isNil, uniq } from 'lodash';

import { ModelTraceSpanType } from '@databricks/web-shared/model-trace-explorer';
import type { ModelTrace, ModelTraceInfo, RetrieverDocument } from '@databricks/web-shared/model-trace-explorer';

import { stringifyValue } from '../components/GenAiEvaluationTracesReview.utils';
import { KnownEvaluationResultAssessmentName } from '../enum';
import { CUSTOM_METADATA_COLUMN_ID, TAGS_COLUMN_ID } from '../hooks/useTableColumns';
import type {
  AssessmentType,
  AssessmentV3,
  RunEvaluationResultAssessment,
  RunEvaluationResultAssessmentSource,
  RunEvaluationTracesDataEntry,
  RunEvaluationTracesRetrievalChunk,
  TraceInfoV3,
} from '../types';

// This is the key used by the eval harness to record
// which chunk a given retrieval assessment corresponds to.
const MLFLOW_SPAN_OUTPUT_KEY = 'span_output_key';

const MLFLOW_ASSESSMENT_ROOT_CAUSE_ASSESSMENT = 'root_cause_assessment';
const MLFLOW_ASSESSMENT_ROOT_CAUSE_RATIONALE = 'root_cause_rationale';
const MLFLOW_ASSESSMENT_SUGGESTED_ACTION = 'suggested_action';
export const MLFLOW_SOURCE_RUN_KEY = 'mlflow.sourceRun';

export const MLFLOW_INTERNAL_PREFIX = 'mlflow.';

export const getRowIdFromEvaluation = (evaluation?: RunEvaluationTracesDataEntry) => {
  return evaluation?.evaluationId || '';
};

export const getRowIdFromTrace = (trace?: TraceInfoV3) => {
  return trace?.trace_id || '';
};

export const getTagKeyFromColumnId = (columnId: string) => {
  return columnId.split(':').pop();
};

export const getCustomMetadataKeyFromColumnId = (columnId: string) => {
  return columnId.split(':').pop();
};

export const createTagColumnId = (tagKey: string) => {
  return `${TAGS_COLUMN_ID}:${tagKey}`;
};

export const createCustomMetadataColumnId = (metadataKey: string) => {
  return `${CUSTOM_METADATA_COLUMN_ID}:${metadataKey}`;
};

export const getTracesTagKeys = (traces: TraceInfoV3[]): string[] => {
  return uniq(
    traces
      .map((result) => {
        return Object.keys(result.tags || {}).filter((key) => key && !key.startsWith(MLFLOW_INTERNAL_PREFIX));
      })
      .flat(),
  );
};

/**
 * This is currently only used to support the edit tags flow which only requires request id + tags
 */
export const convertTraceInfoV3ToModelTraceInfo = (trace: TraceInfoV3) => {
  const kvArray = (obj?: Record<string, string>): { key: string; value: string }[] | undefined =>
    obj ? Object.entries(obj).map(([key, value]) => ({ key, value })) : undefined;

  const model: ModelTraceInfo = {
    request_id: trace.client_request_id ?? trace.trace_id,
    tags: kvArray(trace.tags),
  };

  return model;
};

// This function checks if the traceInfo field is present in the first entry of the evalResults array.
// We assume that all entries in evalResults will either contain traceInfo or not.
export const shouldUseTraceInfoV3 = (evalResults: RunEvaluationTracesDataEntry[]): boolean => {
  return evalResults.length > 0 && Boolean(evalResults[0].traceInfo);
};

const safelyParseValue = <T>(val: string): string | T => {
  try {
    return JSON.parse(val);
  } catch {
    return val;
  }
};

export const getTraceInfoInputs = (traceInfo: TraceInfoV3) => {
  return traceInfo.request_preview || traceInfo.request || traceInfo.trace_metadata?.['mlflow.traceInputs'] || '';
};

export const getTraceInfoOutputs = (traceInfo: TraceInfoV3) => {
  return traceInfo.response_preview || traceInfo.response || traceInfo.trace_metadata?.['mlflow.traceOutputs'] || '';
};

const isExpectationAssessment = (assessment: AssessmentV3): boolean => {
  return Boolean(assessment.expectation);
};

const LIST_TRACES_IGNORE_ASSESSMENTS = ['agent/latency_seconds'];

function processExpectationAssessment(assessment: AssessmentV3, targets: Record<string, any>): void {
  const assessmentName = assessment.assessment_name;
  const assessmentValue = assessment.expectation?.value || assessment.expectation?.serialized_value?.value;

  if (Array.isArray(assessmentValue) && assessmentValue.length > 0) {
    targets[assessmentName] = assessmentValue.map((val) => {
      return safelyParseValue(val);
    });
  } else if (typeof assessmentValue === 'string') {
    targets[assessmentName] = safelyParseValue(assessmentValue);
  } else {
    targets[assessmentName] = assessmentValue;
  }
}

function processFeedbackAssessment(
  assessment: AssessmentV3,
  overallAssessments: RunEvaluationResultAssessment[],
  responseAssessmentsByName: Record<string, RunEvaluationResultAssessment[]>,
): void {
  const assessmentName = assessment.assessment_name;
  const evalResultAssessment = convertFeedbackAssessmentToRunEvalAssessment(assessment);

  if (assessmentName === KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT) {
    overallAssessments.push(evalResultAssessment);
  }
  if (!responseAssessmentsByName[assessmentName]) {
    responseAssessmentsByName[assessmentName] = [];
  }

  responseAssessmentsByName[assessmentName].push(evalResultAssessment);
}

const convertAssessmentV3Source = (assessment: AssessmentV3): RunEvaluationResultAssessmentSource | undefined => {
  if (!assessment.source?.source_type) {
    return undefined;
  }
  const sourceType = assessment.source?.source_type;

  let runEvalSourceType: AssessmentType;
  if (sourceType === 'LLM_JUDGE') {
    runEvalSourceType = 'AI_JUDGE';
  } else {
    runEvalSourceType = sourceType;
  }

  return {
    sourceType: runEvalSourceType,
    sourceId: assessment.source?.source_id || '',
    metadata: {},
  };
};

const convertFeedbackAssessmentToRunEvalAssessment = (assessment: AssessmentV3): RunEvaluationResultAssessment => {
  const assessmentValue = assessment.feedback?.value;
  const isOverallAssessment = assessment.assessment_name === KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT;
  const source = convertAssessmentV3Source(assessment);
  const error = assessment.feedback?.error || assessment.error;
  return {
    name: assessment.assessment_name,
    stringValue: typeof assessmentValue === 'string' ? assessmentValue : undefined,
    booleanValue: typeof assessmentValue === 'boolean' ? assessmentValue : undefined,
    numericValue: typeof assessmentValue === 'number' ? assessmentValue : undefined,
    errorCode: error?.error_code,
    errorMessage: error?.error_message,
    rationale: assessment.metadata?.[MLFLOW_ASSESSMENT_ROOT_CAUSE_RATIONALE] || assessment.rationale,
    source,
    rootCauseAssessment: isOverallAssessment
      ? {
          assessmentName: assessment.metadata?.[MLFLOW_ASSESSMENT_ROOT_CAUSE_ASSESSMENT] || '',
          suggestedActions: assessment.metadata?.[MLFLOW_ASSESSMENT_SUGGESTED_ACTION],
        }
      : undefined,
    metadata: assessment.metadata,
  };
};

export const convertTraceInfoV3ToRunEvalEntry = (traceInfo: TraceInfoV3): RunEvaluationTracesDataEntry => {
  const evaluationId = getRowIdFromTrace(traceInfo);

  // Prepare containers for our assessments.
  const overallAssessments: RunEvaluationResultAssessment[] = [];
  const responseAssessmentsByName: Record<string, RunEvaluationResultAssessment[]> = {};
  const targets: Record<string, any> = {};

  traceInfo.assessments?.forEach((assessment) => {
    const assessmentName = assessment.assessment_name;
    if (LIST_TRACES_IGNORE_ASSESSMENTS.includes(assessmentName)) {
      return;
    }
    if (isExpectationAssessment(assessment)) {
      processExpectationAssessment(assessment, targets);
    } else {
      processFeedbackAssessment(assessment, overallAssessments, responseAssessmentsByName);
    }
  });

  // trace server has input/output in request/response field, and mlflow tracking server has it in the metadata
  const rawInputs = getTraceInfoInputs(traceInfo);
  const rawOutputs = getTraceInfoOutputs(traceInfo);

  let inputsTitle = rawInputs;
  let inputs: Record<string, any> = {};
  let outputs: Record<string, any> = {};
  try {
    inputs = JSON.parse(rawInputs);

    // Try to parse OpenAI messages
    const messages = inputs['messages'];
    if (Array.isArray(messages) && !isNil(messages[0]?.content)) {
      inputsTitle = messages[messages.length - 1]?.content;
    } else {
      inputsTitle = stringifyValue(inputs);
    }
  } catch {
    inputs = {
      request: rawInputs,
    };
  }

  try {
    outputs = { response: JSON.parse(rawOutputs) };
  } catch {
    outputs = { response: rawOutputs };
  }
  return {
    evaluationId,
    requestId: traceInfo.client_request_id || evaluationId,
    inputsId: evaluationId,
    inputsTitle,
    inputs,
    outputs,
    targets,
    overallAssessments,
    responseAssessmentsByName,
    metrics: {},
    traceInfo,
  };
};

export const applyTraceInfoV3ToEvalEntry = (
  evalResults: RunEvaluationTracesDataEntry[],
): RunEvaluationTracesDataEntry[] => {
  if (!shouldUseTraceInfoV3(evalResults)) {
    return evalResults;
  }
  return evalResults.map((result) => {
    if (!result.traceInfo) {
      return result;
    }
    // Convert the single TraceInfo to a single RunEvaluationTracesDataEntry
    const converted = convertTraceInfoV3ToRunEvalEntry(result.traceInfo);
    // Merge the newly converted fields with the existing data
    return {
      ...result,
      ...converted,
    };
  });
};

export const isTraceExportable = (entry: RunEvaluationTracesDataEntry) => {
  let responseJson;
  try {
    responseJson = JSON.parse(entry.outputs['response']);
  } catch {
    if (!entry.outputs['response']) {
      return false;
    }
    // entry.outputs.response may already be parsed in case of external monitors
    // so try using it directly here.
    responseJson = entry.outputs['response'];
  }
  if (isNil(responseJson)) {
    return false;
  }

  const responseIsChatCompletion =
    (Array.isArray(responseJson['messages']) && !isNil(responseJson['messages']?.[0]?.['content'])) ||
    (Array.isArray(responseJson['choices']) && !isNil(responseJson['choices']?.[0]?.['message']?.['content']));

  return responseIsChatCompletion;
};

export function getRetrievedContextFromTrace(
  responseAssessmentsByName: Record<string, RunEvaluationResultAssessment[]>,
  trace: ModelTrace | undefined,
): RunEvaluationTracesRetrievalChunk[] | undefined {
  if (isNil(trace)) {
    return undefined;
  }
  let docUriKey = 'doc_uri';
  const tags = trace.info.tags as Record<string, string> | undefined;
  if (tags?.['retrievers']) {
    const retrieverInfos = safelyParseValue<{ doc_uri: string; chunk_id: string }[]>(tags['retrievers']);
    if (typeof retrieverInfos === 'object' && retrieverInfos.length > 0) {
      docUriKey = retrieverInfos[0].doc_uri;
    }
  }

  const retrievalSpans = trace.data.spans.filter(
    (span) =>
      span.attributes?.['mlflow.spanType'] &&
      safelyParseValue(span.attributes?.['mlflow.spanType']) === ModelTraceSpanType.RETRIEVER,
  );
  if (retrievalSpans.length === 0) {
    return [];
  }

  // Return the last retrieval span chronologically since it is the one analyzed by our judges.
  const spanOutputs = retrievalSpans.at(-1)?.attributes?.['mlflow.spanOutputs'];
  if (!spanOutputs) {
    return [];
  }

  const outputs = safelyParseValue(spanOutputs) as RetrieverDocument[];
  if (!Array.isArray(outputs)) {
    return [];
  }

  const retrievalChunks = outputs.map((doc, index) => {
    return {
      docUrl: doc.metadata?.[docUriKey],
      content: doc.page_content,
      retrievalAssessmentsByName: getRetrievalAssessmentsByName(responseAssessmentsByName, index),
    };
  });

  return retrievalChunks;
}

const getRetrievalAssessmentsByName = (
  responseAssessmentsByName: Record<string, RunEvaluationResultAssessment[]>,
  chunkIndex: number,
): Record<string, RunEvaluationResultAssessment[]> => {
  const filteredResponseAssessmentsByName = Object.fromEntries(
    Object.entries(responseAssessmentsByName)
      .map(([key, assessments]) => [
        key,
        assessments.filter((assessment) => Number(assessment?.metadata?.[MLFLOW_SPAN_OUTPUT_KEY]) === chunkIndex),
      ])
      .filter(([key, filteredAssessments]) => filteredAssessments.length > 0),
  );

  return filteredResponseAssessmentsByName;
};
