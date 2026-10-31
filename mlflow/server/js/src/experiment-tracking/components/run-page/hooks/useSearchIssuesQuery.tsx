import { useQuery } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';

export const SEARCH_ISSUES_QUERY_KEY = 'SEARCH_ISSUES';

export type IssueStatus = 'pending' | 'rejected' | 'resolved';

export type IssueSeverity = 'not_an_issue' | 'low' | 'medium' | 'high';

export interface Issue {
  issue_id: string;
  experiment_id: string;
  name: string;
  description?: string;
  severity?: IssueSeverity;
  status: IssueStatus;
  source_run_id?: string;
  created_by?: string;
  created_timestamp: number;
  last_updated_timestamp: number;
  categories?: string[];
  trace_count?: number;
  last_seen_timestamp?: number;
  last_updated_by?: string;
  metadata?: Record<string, string>;
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
        include_trace_count: true,
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

/** The server defaults to 100 issues per page. Past the cap the overflow is simply not reported. */
const SEARCH_ISSUE_IDS_PAGE_SIZE = 1000;
const SEARCH_ISSUE_IDS_MAX_PAGES = 5;

/**
 * The IDs of an experiment's issues with the given status.
 *
 * Search is the only issues read that answers for many issues at once -- `GetIssue` is one request
 * per ID -- so a caller needing the status of every issue on screen asks for the IDs in the status
 * it cares about and tests membership.
 */
export const searchIssueIdsByStatus = async (experimentId: string, status: IssueStatus): Promise<string[]> => {
  const issueIds: string[] = [];
  let pageToken: string | undefined;

  for (let page = 0; page < SEARCH_ISSUE_IDS_MAX_PAGES; page++) {
    const response = (await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/issues/search'), {
      method: 'POST',
      body: {
        experiment_id: experimentId,
        statuses: [status],
        // Only the IDs are read, so skip the trace counts and the long-form fields.
        include_trace_count: false,
        view: 'basic',
        max_results: SEARCH_ISSUE_IDS_PAGE_SIZE,
        ...(pageToken ? { page_token: pageToken } : {}),
      },
    })) as SearchIssuesResponse;

    issueIds.push(...(response.issues ?? []).map(({ issue_id }) => issue_id));
    pageToken = response.next_page_token;
    if (!pageToken) {
      break;
    }
  }

  return issueIds;
};
