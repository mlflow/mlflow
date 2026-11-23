import { useMutation, useQueryClient, useIsMutating, type QueryClient } from '@databricks/web-shared/query-client';
import { useActiveWorkspace } from '../../../../../../workspaces/utils/WorkspaceUtils';
import { fetchAPI, getAjaxUrl } from '../../../../../../common/utils/FetchUtils';
import type { IssueCategory } from '../IssueDetectionCategories';

interface InvokeIssueDetectionParams {
  experimentId: string;
  traceIds: string[];
  /** Built-in ids plus any custom focus areas saved in the experiment's detection settings. */
  categories: (IssueCategory | string)[];
  /** The experiment's saved detection priorities, if it has any. */
  instructions?: string;
  provider: string;
  model: string;
  secret_id?: string;
  endpoint_name?: string;
}

interface InvokeIssueDetectionResponse {
  job_id: string;
  run_id: string;
}

const pendingLaunches = new WeakMap<QueryClient, Map<string, Promise<InvokeIssueDetectionResponse>>>();

export const useInvokeIssueDetection = (experimentId?: string) => {
  const client = useQueryClient();
  const workspace = useActiveWorkspace();
  const mutationKey = ['ISSUE_DETECTION_LAUNCH', workspace, experimentId];
  const mutation = useMutation<InvokeIssueDetectionResponse, Error, InvokeIssueDetectionParams>({
    mutationKey,
    mutationFn: (params) => {
      let launches = pendingLaunches.get(client);
      if (!launches) {
        launches = new Map();
        pendingLaunches.set(client, launches);
      }
      const key = JSON.stringify([
        workspace,
        params.experimentId,
        [...params.traceIds].sort(),
        [...params.categories].sort(),
        params.instructions ?? '',
        params.provider,
        params.model,
        params.secret_id,
        params.endpoint_name,
      ]);
      const pending = launches.get(key);
      if (pending) return pending;
      const request = fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/issues/invoke'), {
        method: 'POST',
        body: {
          experiment_id: params.experimentId,
          trace_ids: params.traceIds,
          categories: params.categories,
          instructions: params.instructions,
          provider: params.provider,
          model: params.model,
          secret_id: params.secret_id,
          endpoint_name: params.endpoint_name,
        },
      })
        .then((response) => response as InvokeIssueDetectionResponse)
        .finally(() => launches.delete(key));
      launches.set(key, request);
      return request;
    },
  });
  return { ...mutation, isLoading: useIsMutating({ mutationKey }) > 0 };
};
