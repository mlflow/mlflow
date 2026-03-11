import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { MlflowService } from '../../../sdk/MlflowService';
import { FETCH_ISSUE_JOB_STATUS_QUERY_KEY } from './useFetchIssueJobStatus';

interface CancelJobParams {
  jobId: string;
  runUuid?: string;
}

interface CancelJobResponse {
  status: string;
}

export const useCancelJob = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error } = useMutation<CancelJobResponse, Error, CancelJobParams>({
    mutationFn: async ({ jobId, runUuid }) => {
      const response = (await fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/jobs/cancel/${jobId}`), {
        method: 'PATCH',
      })) as CancelJobResponse;

      // Terminate the underlying MLflow run since the job process is killed independently
      if (runUuid) {
        await MlflowService.updateRun({
          run_id: runUuid,
          status: 'KILLED',
        });
      }

      return response;
    },
    onSuccess: (_data, { jobId }) => {
      queryClient.invalidateQueries([FETCH_ISSUE_JOB_STATUS_QUERY_KEY, jobId]);
    },
  });

  return {
    cancelJob: mutate,
    cancelJobAsync: mutateAsync,
    isCancelling: isLoading,
    error,
  };
};
