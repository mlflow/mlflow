import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { SEARCH_ISSUES_QUERY_KEY, type Issue, type IssueStatus } from './useSearchIssuesQuery';

export interface UpdateIssueParams {
  issueId: string;
  status?: IssueStatus;
  name?: string;
  description?: string;
  severity?: string;
}

interface UpdateIssueResponse {
  issue: Issue;
}

export const useUpdateIssue = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error } = useMutation<UpdateIssueResponse, Error, UpdateIssueParams>({
    mutationFn: async ({ issueId, ...updates }) => {
      return (await fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/issues/${issueId}`), {
        method: 'PATCH',
        body: { issue_id: issueId, ...updates },
      })) as UpdateIssueResponse;
    },
    onSuccess: () => {
      queryClient.invalidateQueries([SEARCH_ISSUES_QUERY_KEY]);
    },
  });

  return {
    updateIssue: mutate,
    updateIssueAsync: mutateAsync,
    isUpdating: isLoading,
    error,
  };
};
