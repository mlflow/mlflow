import { useState } from 'react';
import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import { createTraceLocationForExperiment } from '@databricks/web-shared/genai-traces-table';
import { useInvokeIssueDetection } from '../../experiment-page/components/traces-v3/hooks/useInvokeIssueDetection';
import type { IssueCategory } from '../../experiment-page/components/traces-v3/IssueDetectionCategories';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';

interface RetryIssueDetectionParams {
  experimentId: string;
  runUuid: string;
  /** endpoint_name tag value, present when a gateway endpoint was used */
  endpointName?: string;
  /** model tag value, e.g. "openai:/gpt-4", used when no endpoint_name */
  model?: string;
  categories: IssueCategory[];
}

interface RetryIssueDetectionResult {
  job_id: string;
  run_id: string;
}

async function fetchTraceIdsForRun(experimentId: string, runUuid: string): Promise<string[]> {
  const location = createTraceLocationForExperiment(experimentId);
  const filter = `run_id = '${runUuid}'`;
  const traceIds: string[] = [];
  let pageToken: string | undefined;

  do {
    const payload: Record<string, unknown> = {
      locations: [location],
      filter,
      ...(pageToken ? { page_token: pageToken } : {}),
    };

    const json = (await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/traces/search'), {
      method: 'POST',
      body: payload,
    })) as { traces?: ModelTraceInfoV3[]; next_page_token?: string };

    for (const trace of json.traces ?? []) {
      if (trace.trace_id) {
        traceIds.push(trace.trace_id);
      }
    }
    pageToken = json.next_page_token;
  } while (pageToken);

  return traceIds;
}

function resolveProviderParams(
  endpointName?: string,
  model?: string,
): { provider: string; model: string; endpoint_name?: string } {
  if (endpointName) {
    return { provider: '', model: '', endpoint_name: endpointName };
  }
  const separatorIndex = (model ?? '').indexOf(':/');
  if (separatorIndex === -1) {
    return { provider: model ?? '', model: '' };
  }
  return { provider: model!.slice(0, separatorIndex), model: model!.slice(separatorIndex + 2) };
}

export const useRetryIssueDetection = () => {
  const [isRetrying, setIsRetrying] = useState(false);
  const { mutateAsync: invokeIssueDetection } = useInvokeIssueDetection();

  const retryIssueDetection = async (params: RetryIssueDetectionParams): Promise<RetryIssueDetectionResult> => {
    setIsRetrying(true);
    try {
      const traceIds = await fetchTraceIdsForRun(params.experimentId, params.runUuid);
      if (traceIds.length === 0) {
        throw new Error('No traces found for this run');
      }
      const { provider, model, endpoint_name } = resolveProviderParams(params.endpointName, params.model);
      return await invokeIssueDetection({
        experimentId: params.experimentId,
        traceIds,
        categories: params.categories,
        provider,
        model,
        endpoint_name,
      });
    } finally {
      setIsRetrying(false);
    }
  };

  return { retryIssueDetection, isRetrying };
};
