import { useMutation } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';

interface InvokeGenAIEvaluationParams {
  experimentId: string;
  traceIds: string[];
  serializedScorers: string[];
}

interface InvokeGenAIEvaluationResponse {
  job_id: string;
  run_id: string;
}

export const useInvokeGenAIEvaluation = () =>
  useMutation<InvokeGenAIEvaluationResponse, Error, InvokeGenAIEvaluationParams>({
    mutationFn: async ({ experimentId, traceIds, serializedScorers }) => {
      const response = await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/genai/evaluate/invoke'), {
        method: 'POST',
        body: {
          experiment_id: experimentId,
          trace_ids: traceIds,
          serialized_scorers: serializedScorers,
        },
      });
      return response as InvokeGenAIEvaluationResponse;
    },
  });
