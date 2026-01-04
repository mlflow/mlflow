import type { FeedbackAssessment, ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import { fetchOrFail, getAjaxUrl } from '../../../common/utils/FetchUtils';

/**
 * Fetches a single full trace from the MLflow API.
 * TODO: move and/or reuse it in shared/model-trace-explorer/api.ts
 */
export const getMlflowTraceV3ForEvaluation = async (requestId: string): Promise<ModelTrace> => {
  const [traceInfoResponse, traceDataResponse] = await Promise.all([
    fetchOrFail(getAjaxUrl(`ajax-api/3.0/mlflow/traces/${requestId}`)),
    fetchOrFail(getAjaxUrl(`ajax-api/3.0/mlflow/get-trace-artifact?request_id=${requestId}`)),
  ]);
  const [traceInfo, traceData] = await Promise.all([traceInfoResponse.json(), traceDataResponse.json()]);

  return {
    info: traceInfo?.trace?.trace_info || {},
    data: traceData,
  } as ModelTrace;
};

/**
 * Result from evaluating a judge on a single trace.
 * Always returns results as an array, even for single-result assessments.
 * This simplifies rendering logic as components can always iterate over results.
 */
interface JudgeSimplifiedAssessmentResult {
  assessment_id?: string;
  result: string | null;
  rationale: string | null;
  error: string | null;
  span_name?: string;
}

export interface JudgeEvaluationResult {
  trace: ModelTrace | null;
  results: JudgeSimplifiedAssessmentResult[]; // Always an array, even for single-result assessments
  error: string | null;
}
