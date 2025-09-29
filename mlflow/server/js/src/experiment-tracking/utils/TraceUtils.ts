import { MlflowService } from '../sdk/MlflowService';
import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';

/**
 * Fetches trace information and data for a given trace ID.
 *
 * @param traceId - The ID of the trace to fetch
 * @returns Promise resolving to ModelTrace object or undefined if trace cannot be fetched
 */
export async function getTrace(traceId?: string): Promise<ModelTrace | undefined> {
  if (!traceId) {
    return undefined;
  }

  const [traceInfoResponse, traceData] = await Promise.all([
    MlflowService.getExperimentTraceInfoV3(traceId),
    MlflowService.getExperimentTraceData(traceId),
  ]);

  return traceData
    ? {
        info: traceInfoResponse?.trace?.trace_info || {},
        data: traceData,
      }
    : undefined;
}

/**
 * Fetches trace information and data for a given trace ID using the legacy API.
 *
 * @param requestId - The ID of the request to fetch
 * @returns Promise resolving to ModelTrace object or undefined if trace cannot be fetched
 */
export async function getTraceLegacy(requestId?: string): Promise<ModelTrace | undefined> {
  if (!requestId) {
    return undefined;
  }

  const [traceInfo, traceData] = await Promise.all([
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    MlflowService.getExperimentTraceInfo(requestId!).then((response) => response.trace_info || {}),
    // get-trace-artifact is only currently supported in mlflow 2.0 apis
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    MlflowService.getExperimentTraceData(requestId!),
  ]);
  return traceData
    ? {
        info: traceInfo,
        data: traceData,
      }
    : undefined;
}
