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

/**
 * The dismissed set has to be complete: an ID missing from it reads as not dismissed, which is the
 * bug the lookup exists to fix. So every page is drained rather than stopping at a fixed count.
 *
 * The page size is the most the Databricks issues backend accepts -- `DatabricksRestStore`
 * rejects `max_results` above 100 outright -- so the set is assembled from many small pages there
 * as well as here.
 *
 * The page bound and the repeated-token check only guarantee termination; the bound covers the
 * same 100,000 issues it did at ten times the page size. A server still handing back pages after
 * that many, or handing back a token it already gave, is broken rather than large, and throwing
 * leaves the caller hiding nothing.
 */
const SEARCH_ISSUE_IDS_PAGE_SIZE = 100;
const SEARCH_ISSUE_IDS_MAX_PAGES = 1000;

/**
 * The IDs of an experiment's issues with the given status.
 *
 * Search is the only issues read that answers for many issues at once -- `GetIssue` is one request
 * per ID -- so a caller needing the status of every issue on screen asks for the IDs in the status
 * it cares about and tests membership.
 *
 * `signal` comes from the caller's query function, so a lookup still paging when the last surface
 * reading it unmounts stops instead of walking the rest of the experiment.
 */
export const searchIssueIdsByStatus = async (
  experimentId: string,
  status: IssueStatus,
  signal?: AbortSignal,
): Promise<string[]> => {
  const issueIds: string[] = [];
  const seenPageTokens = new Set<string>();
  let pageToken: string | undefined;

  for (let page = 0; page < SEARCH_ISSUE_IDS_MAX_PAGES; page++) {
    const response = (await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/issues/search'), {
      method: 'POST',
      signal,
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

    // The server sends an empty token on the last page.
    pageToken = response.next_page_token || undefined;
    if (!pageToken) {
      return issueIds;
    }
    if (seenPageTokens.has(pageToken)) {
      throw new Error('Issue search returned a repeated page token');
    }
    seenPageTokens.add(pageToken);
  }

  throw new Error(`Issue search did not reach its last page within ${SEARCH_ISSUE_IDS_MAX_PAGES} pages`);
};
