import { useQuery } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';

export const SEARCH_ISSUES_QUERY_KEY = 'SEARCH_ISSUES';

export type IssueStatus = 'pending' | 'accepted' | 'rejected' | 'resolved';

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

const POLLING_INTERVAL_MS = 3000;

export const useSearchIssuesQuery = ({
  experimentId,
  sourceRunId,
  status,
  enabled = true,
  pollingEnabled = false,
}: {
  experimentId: string;
  sourceRunId: string;
  /** Filter by status. If undefined, returns all statuses. */
  status?: IssueStatus;
  enabled?: boolean;
  /** Whether to poll for updates (e.g., while job is running) */
  pollingEnabled?: boolean;
}) => {
  const { data, isLoading, isFetching, refetch, error } = useQuery<SearchIssuesResponse, Error>({
    queryKey: [SEARCH_ISSUES_QUERY_KEY, experimentId, sourceRunId, status],
    queryFn: async () => {
      const filters = [`source_run_id = '${sourceRunId}'`];
      if (status) {
        filters.push(`status = '${status}'`);
      }
      const filterString = filters.join(' AND ');
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
    refetchInterval: (_data, query) => {
      if (!pollingEnabled || query.state.error) {
        return false;
      }
      return POLLING_INTERVAL_MS;
    },
  });

  return {
    issues: data?.issues ?? [],
    isLoading,
    isFetching,
    refetch,
    error,
  };
};
