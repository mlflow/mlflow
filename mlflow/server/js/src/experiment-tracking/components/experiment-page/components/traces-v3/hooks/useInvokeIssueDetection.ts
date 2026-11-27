import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '../../../../../../common/utils/FetchUtils';
import { MLFLOW_ISSUE_DETECTION_JOB_ID_TAG } from '../../../../../constants';
import { LATEST_ISSUE_DETECTION_QUERY_KEY } from '../../../../../pages/experiment-issues/useIssueDetection';

interface InvokeIssueDetectionParams {
  experimentId: string;
  traceIds: string[];
  provider: string;
  model: string;
  secret_id?: string;
  endpoint_name?: string;
}

interface InvokeIssueDetectionResponse {
  /** Unset when the server started the run but could not record its job on it. */
  job_id?: string;
  run_id: string;
}

interface DetectIssuesResponse {
  issue_detection_run?: { run_id?: string };
}

interface GetRunResponse {
  run?: { data?: { tags?: { key: string; value: string }[] } };
}

/**
 * Starts a detection run through the public `DetectIssues` API.
 *
 * The run uses the categories and instructions saved on the experiment, which the server reads
 * itself. Its job ID, which the run's progress and cancellation are tracked by, is read back from
 * the run's tags: the server records it there before it responds.
 */
export const useInvokeIssueDetection = () => {
  const queryClient = useQueryClient();
  return useMutation<InvokeIssueDetectionResponse, Error, InvokeIssueDetectionParams>({
    mutationFn: async (params) => {
      const model = params.endpoint_name ? `gateway:/${params.endpoint_name}` : `${params.provider}:/${params.model}`;
      const response = (await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/issues/detect'), {
        method: 'POST',
        body: {
          experiment_id: params.experimentId,
          trace_ids: params.traceIds,
          model,
          secret_id: params.endpoint_name ? undefined : params.secret_id,
        },
      })) as DetectIssuesResponse;
      const runId = response.issue_detection_run?.run_id;
      if (!runId) {
        throw new Error('Issue detection started without returning its run.');
      }
      const query = new URLSearchParams({ run_id: runId });
      const run = (await fetchAPI(getAjaxUrl(`ajax-api/2.0/mlflow/runs/get?${query.toString()}`))) as GetRunResponse;
      const jobId = run.run?.data?.tags?.find((tag) => tag.key === MLFLOW_ISSUE_DETECTION_JOB_ID_TAG)?.value;
      return { job_id: jobId, run_id: runId };
    },
    onSuccess: ({ job_id }) => {
      // A run without a recorded job can't be followed through its job, but it did start, so
      // the surfaces showing the latest run are refreshed to pick it up.
      if (!job_id) {
        queryClient.invalidateQueries([LATEST_ISSUE_DETECTION_QUERY_KEY]);
      }
    },
  });
};
