/**
 * The issue detection runs an experiment has recorded, newest first.
 *
 * Runs come from the same paged search the Evaluation runs tab uses, narrowed to detection runs.
 * The issue counts come from the issues themselves rather than from the run: an issue records the
 * run that first filed it, and a run that re-finds a known issue refreshes that issue instead of
 * filing another, so this counts what each run found that was new.
 */

import { useMemo } from 'react';

import { useQuery } from '@databricks/web-shared/query-client';

import { fetchAPI, getAjaxUrl } from '../../../common/utils/FetchUtils';
import { useActiveWorkspace } from '../../../workspaces/utils/WorkspaceUtils';
import { useExperimentEvaluationRunsData } from '../../components/experiment-page/hooks/useExperimentEvaluationRunsData';
import {
  DETECTION_RUN_FILTER,
  type DetectionRunRow,
  isDetectionRunSettled,
  toDetectionRunRow,
} from './detectionRunHistory';

export const DETECTION_RUN_NEW_ISSUES_QUERY_KEY = 'DETECTION_RUN_NEW_ISSUES';

/**
 * How many runs' issue counts are fetched.
 *
 * One request per run, because the issue filter grammar compares a single `source_run_id` at a
 * time and has no `IN`. Bounded so that scrolling a long history cannot turn into hundreds of
 * requests; past it the count is left blank rather than guessed at.
 */
export const MAX_COUNTED_RUNS = 50;

/**
 * How often an unfinished run is checked.
 *
 * A detection run takes minutes, so this is about the page not going stale while someone watches
 * it rather than about catching the moment it lands. Only ever on while a loaded run is still
 * going.
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

const useNewIssueCounts = (experimentId: string, runKeys: string[], enabled: boolean) => {
  const workspace = useActiveWorkspace();
  // One query keyed by the whole set rather than one per row: a hook cannot be called in a loop,
  // and the counts are only ever read together.
  //
  // The keys carry each run's status as well as its id, so a run finishing changes the key and its
  // count is read again. Keyed on ids alone, a count fetched while a run was still filing issues
  // would stay on screen after it finished.
  const counted = runKeys.slice(0, MAX_COUNTED_RUNS);
  const runIds = counted.map((key) => key.split(':')[0]);
  const { data } = useQuery<Record<string, number>, Error>({
    queryKey: [DETECTION_RUN_NEW_ISSUES_QUERY_KEY, workspace, experimentId, counted],
    queryFn: async ({ signal }) => {
      const counts = await Promise.all(
        runIds.map(async (runId) => {
          try {
            return [runId, await fetchNewIssueCount(experimentId, runId, signal)] as const;
          } catch {
            // A count that cannot be read leaves that one cell blank; the rest of the row, and
            // every other row, is still worth showing.
            return [runId, undefined] as const;
          }
        }),
      );
      return Object.fromEntries(counts.filter(([, count]) => count !== undefined)) as Record<string, number>;
    },
    enabled: enabled && counted.length > 0,
    retry: false,
    refetchOnWindowFocus: false,
  });
  return data;
};

export const useDetectionRuns = (experimentId: string, { enabled = true } = {}) => {
  const {
    data: runs,
    hasNextPage,
    fetchNextPage,
    isLoading,
    isFetching,
    error,
    refetch,
  } = useExperimentEvaluationRunsData({
    experimentId,
    enabled,
    filter: DETECTION_RUN_FILTER,
    // A run on screen while it works would otherwise read `Running` for good: the run search
    // neither polls nor refetches on focus, so its status, cost and failure reason would not
    // arrive until the page was left and reopened. Asked again after each fetch, so polling stops
    // by itself once every loaded run has finished.
    refetchInterval: (data) =>
      (data?.pages ?? []).some((page) =>
        (page.runs ?? []).some((run) => {
          const row = toDetectionRunRow(run);
          return row !== undefined && !isDetectionRunSettled(row.status);
        }),
      )
        ? RUNNING_POLL_INTERVAL_MS
        : false,
  });

  const rows = useMemo(
    () => (runs ?? []).map(toDetectionRunRow).filter((row): row is DetectionRunRow => row !== undefined),
    [runs],
  );
  // Carries each run's status, so a count read while one was still filing is read again once it
  // finishes.
  const runKeys = useMemo(() => rows.map((row) => `${row.runId}:${row.status}`), [rows]);
  const newIssueCounts = useNewIssueCounts(experimentId, runKeys, enabled && rows.length > 0);

  return {
    rows,
    newIssueCounts,
    hasNextPage,
    fetchNextPage,
    isLoading,
    isFetching,
    error,
    refetch,
  };
};
