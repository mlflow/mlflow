import { useMutation } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '../../../../../../common/utils/FetchUtils';
import type { IssueCategory } from '../IssueDetectionCategories';

interface InvokeIssueDetectionParams {
  experimentId: string;
  traceIds: string[];
  categories: IssueCategory[];
  provider: string;
  model: string;
  secret_id: string;
}

interface InvokeIssueDetectionResponse {
  job_id: string;
  run_id: string;
}

export const useInvokeIssueDetection = () => {
  return useMutation<InvokeIssueDetectionResponse, Error, InvokeIssueDetectionParams>({
    mutationFn: async (params) => {
      const response = await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/issues/invoke'), {
        method: 'POST',
        body: {
          experiment_id: params.experimentId,
          trace_ids: params.traceIds,
          categories: params.categories,
          provider: params.provider,
          model: params.model,
          secret_id: params.secret_id,
        },
      });
      return response as InvokeIssueDetectionResponse;
    },
  });
};
