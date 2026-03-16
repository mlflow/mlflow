import { useQuery } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import type { Issue } from './useSearchIssuesQuery';

export const GET_ISSUE_QUERY_KEY = 'GET_ISSUE';

type GetIssueResponse = {
  issue: Issue;
};

export const useGetIssueQuery = ({ issueId, enabled = true }: { issueId: string; enabled?: boolean }) => {
  const { data, isLoading, isFetching, refetch, error } = useQuery<GetIssueResponse, Error>({
    queryKey: [GET_ISSUE_QUERY_KEY, issueId],
    queryFn: async () => {
      return (await fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/issues/${encodeURIComponent(issueId)}`), {
        method: 'GET',
      })) as GetIssueResponse;
    },
    cacheTime: 0,
    refetchOnWindowFocus: false,
    retry: false,
    enabled: enabled && Boolean(issueId),
  });

  return {
    issue: data?.issue,
    isLoading,
    isFetching,
    refetch,
    error,
  };
};

export const getIssue = async (issueId: string): Promise<Issue> => {
  const response = (await fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/issues/${encodeURIComponent(issueId)}`), {
    method: 'GET',
  })) as GetIssueResponse;
  return response.issue;
};
