/**
 * One page of the issue detection runs an experiment has recorded, newest first.
 *
 * Runs are MLflow runs, so the history is a run search narrowed to detection runs. The issue
 * counts come from the issues themselves rather than from the run: an issue records the run that
 * first filed it, and a run that re-finds a known issue refreshes that issue instead of filing
 * another, so this counts what each run found that was new.
 */

import { useCallback, useMemo, useState } from 'react';

import { useQueries, useQuery } from '@databricks/web-shared/query-client';

import { fetchAPI, getAjaxUrl } from '../../../common/utils/FetchUtils';
import { useActiveWorkspace } from '../../../workspaces/utils/WorkspaceUtils';
import { MlflowService } from '../../sdk/MlflowService';
import type { SearchRunsApiResponse } from '../../types';
import {
  DETECTION_RUN_FILTER,
  type DetectionRunRow,
  isDetectionRunSettled,
  toDetectionRunRow,
} from './detectionRunHistory';

export const DETECTION_RUNS_QUERY_KEY = 'DETECTION_RUNS';
export const DETECTION_RUN_NEW_ISSUES_QUERY_KEY = 'DETECTION_RUN_NEW_ISSUES';

/** Runs per page, and so the number of issue-count requests one page can cost. */
export const DETECTION_RUNS_PAGE_SIZE = 25;

const DETECTION_RUN_ORDER_BY = ['attributes.start_time DESC'];

/**
 * How often an unfinished run is checked.
 *
 * A detection run takes minutes, so this is about the page not going stale while someone watches
 * it rather than about catching the moment it lands. Only ever on while a run on the page is
 * still going.
 */
export const RUNNING_POLL_INTERVAL_MS = 10_000;

interface IssueCountResponse {
  total_count?: number;
}

/**
 * How many issues this run filed that were new.
 *
 * Asks for the count alone -- one issue at most comes back -- because the page shows a number and
 * the Issues tab is where the issues themselves are read.
 */
const fetchNewIssueCount = async (experimentId: string, runId: string, signal?: AbortSignal) => {
  const page = (await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/issues/search'), {
    method: 'POST',
    signal,
    body: {
      experiment_id: experimentId,
      filter_string: `source_run_id = '${runId}'`,
      max_results: 1,
      include_total_count: true,
      view: 'basic',
    },
  })) as IssueCountResponse;
  return page.total_count ?? 0;
};

/**
 * Each run's new-issue count, one query per run so a run finishing re-reads its own count alone.
 *
 * One request per run because the issue filter grammar compares a single `source_run_id` at a
 * time and has no `IN`; bounded by the page size rather than by a cap of its own. A run still
 * going is not asked about at all: it has not finished filing, so its count so far would read as
 * a result.
 */
const useNewIssueCounts = (experimentId: string, rows: DetectionRunRow[], enabled: boolean) => {
  const workspace = useActiveWorkspace();
  const results = useQueries({
    queries: rows.map((row) => ({
      // The status is part of the key, so a count read while the run was still filing is read
      // again once it finishes rather than staying on screen.
      queryKey: [DETECTION_RUN_NEW_ISSUES_QUERY_KEY, workspace, experimentId, row.runId, row.status],
      queryFn: ({ signal }: { signal?: AbortSignal }) => fetchNewIssueCount(experimentId, row.runId, signal),
      enabled: enabled && isDetectionRunSettled(row.status),
      retry: false,
      refetchOnWindowFocus: false,
    })),
  });
  const counts: Record<string, number> = {};
  rows.forEach((row, index) => {
    const count = results[index]?.data;
    // A count that cannot be read leaves that one cell blank; the rest of the row, and every
    // other row, is still worth showing.
    if (typeof count === 'number') {
      counts[row.runId] = count;
    }
  });
  return counts;
};

export const useDetectionRuns = (experimentId: string, { enabled = true } = {}) => {
  const workspace = useActiveWorkspace();
  // Page cursors, newest page last. `[undefined]` is the first page. Tokens stay in component
  // state rather than in the URL, the way the datasets and prompts lists hold theirs: they are
  // opaque and they expire.
  const [tokenHistory, setTokenHistory] = useState<(string | undefined)[]>([undefined]);
  const pageToken = tokenHistory[tokenHistory.length - 1];

  const { data, isLoading, isFetching, error, refetch } = useQuery<SearchRunsApiResponse, Error>({
    queryKey: [DETECTION_RUNS_QUERY_KEY, workspace, experimentId, pageToken],
    queryFn: () =>
      MlflowService.searchRuns({
        experiment_ids: [experimentId],
        order_by: DETECTION_RUN_ORDER_BY,
        run_view_type: 'ACTIVE_ONLY',
        filter: DETECTION_RUN_FILTER,
        max_results: DETECTION_RUNS_PAGE_SIZE,
        page_token: pageToken,
      }),
    // A run on screen while it works would otherwise read `Running` for good: the search neither
    // polls nor refetches on focus, so its status, cost and failure reason would not arrive until
    // the page was left and reopened. Asked again after each fetch, so polling stops by itself
    // once every run on the page has finished.
    refetchInterval: (page) =>
      (page?.runs ?? []).some((run) => {
        const row = toDetectionRunRow(run);
        return row !== undefined && !isDetectionRunSettled(row.status);
      })
        ? RUNNING_POLL_INTERVAL_MS
        : false,
    // Paging forward and back keeps the rows that are up while the next page is read, so the
    // table does not collapse to a skeleton on every click.
    keepPreviousData: true,
    refetchOnWindowFocus: false,
    retry: false,
    enabled,
  });

  const rows = useMemo(
    () => (data?.runs ?? []).map(toDetectionRunRow).filter((row): row is DetectionRunRow => row !== undefined),
    [data?.runs],
  );
  const newIssueCounts = useNewIssueCounts(experimentId, rows, enabled);

  const nextPageToken = data?.next_page_token;
  const goToNextPage = useCallback(() => {
    if (nextPageToken) {
      setTokenHistory((previous) => [...previous, nextPageToken]);
    }
  }, [nextPageToken]);
  const goToPreviousPage = useCallback(
    () => setTokenHistory((previous) => (previous.length > 1 ? previous.slice(0, -1) : previous)),
    [],
  );

  return {
    rows,
    newIssueCounts,
    hasNextPage: Boolean(nextPageToken),
    hasPreviousPage: tokenHistory.length > 1,
    goToNextPage,
    goToPreviousPage,
    isLoading,
    isFetching,
    error,
    refetch,
  };
};
