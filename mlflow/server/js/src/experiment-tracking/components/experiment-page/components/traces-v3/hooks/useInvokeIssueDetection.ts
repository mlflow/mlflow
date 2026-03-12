import { useMutation } from '@tanstack/react-query';
import { fetchAPI, getAjaxUrl } from '../../../../../../common/utils/FetchUtils';
import type { IssueCategory } from '../IssueDetectionCategories';

interface InvokeIssueDetectionParams {
  experimentId: string;
  traceIds: string[];
  categories: IssueCategory[];
  provider: string;
  model: string;
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
        },
      });
      return response as InvokeIssueDetectionResponse;
    },
  });
};
