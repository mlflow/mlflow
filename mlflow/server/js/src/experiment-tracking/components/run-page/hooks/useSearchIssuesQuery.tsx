import { useQuery } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';

export const SEARCH_ISSUES_QUERY_KEY = 'SEARCH_ISSUES';

export type IssueStatus = 'pending' | 'accepted' | 'rejected';

export interface Issue {
  issue_id: string;
  experiment_id: string;
  name: string;
  description?: string;
  status: IssueStatus;
  source_run_id?: string;
  created_by?: string;
  created_timestamp: number;
  last_updated_timestamp: number;
}

type SearchIssuesResponse = {
  issues?: Issue[];
  next_page_token?: string;
};

export const useSearchIssuesQuery = ({
  experimentId,
  sourceRunId,
  enabled = true,
}: {
  experimentId: string;
  sourceRunId: string;
  enabled?: boolean;
}) => {
  const { data, isLoading, isFetching, refetch, error } = useQuery<SearchIssuesResponse, Error>({
    queryKey: [SEARCH_ISSUES_QUERY_KEY, experimentId, sourceRunId],
    queryFn: async () => {
      const filterString = `source_run_id = '${sourceRunId}'`;
      const requestBody = {
        experiment_id: experimentId,
        filter_string: filterString,
      };

      return (await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/issues/search'), {
        method: 'POST',
        body: requestBody,
      })) as SearchIssuesResponse;
    },
    cacheTime: 0,
    refetchOnWindowFocus: false,
    retry: false,
    enabled,
  });

  return {
    issues: data?.issues ?? [],
    isLoading,
    isFetching,
    refetch,
    error,
  };
};
