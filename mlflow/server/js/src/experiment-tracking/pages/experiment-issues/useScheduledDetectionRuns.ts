/**
 * The scheduled detection occurrences an experiment has recorded, newest first.
 *
 * Runs come from the same paged search the Evaluation runs tab uses, narrowed to the occurrences
 * a schedule fired. The issue counts come from the issues themselves rather than from the run:
 * an issue records the run that first filed it, and a run that re-finds a known issue refreshes
 * that issue instead of filing another, so this counts what each occurrence found that was new.
 */

import { useMemo } from 'react';

import { useQuery } from '@databricks/web-shared/query-client';

import { fetchAPI, getAjaxUrl } from '../../../common/utils/FetchUtils';
import { useActiveWorkspace } from '../../../workspaces/utils/WorkspaceUtils';
import { useExperimentEvaluationRunsData } from '../../components/experiment-page/hooks/useExperimentEvaluationRunsData';
import {
  SCHEDULED_DETECTION_RUN_FILTER,
  type ScheduledRunRow,
  summarizeScheduledRuns,
  toScheduledRunRow,
} from './scheduledRunHistory';

export const SCHEDULED_RUN_NEW_ISSUES_QUERY_KEY = 'SCHEDULED_RUN_NEW_ISSUES';

/**
 * How many occurrences' issue counts are fetched.
 *
 * One request per occurrence, because the issue filter grammar compares a single `source_run_id`
 * at a time and has no `IN`. Bounded so that scrolling a long history cannot turn into hundreds
 * of requests; past it the count is left blank rather than guessed at.
 */
export const MAX_COUNTED_OCCURRENCES = 50;

interface IssueCountResponse {
  total_count?: number;
}

/**
 * How many issues this occurrence filed that were new.
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

const useNewIssueCounts = (experimentId: string, runIds: string[], enabled: boolean) => {
  const workspace = useActiveWorkspace();
  // One query keyed by the whole set rather than one per row: a hook cannot be called in a loop,
  // and the counts are only ever read together.
  const counted = runIds.slice(0, MAX_COUNTED_OCCURRENCES);
  const { data } = useQuery<Record<string, number>, Error>({
    queryKey: [SCHEDULED_RUN_NEW_ISSUES_QUERY_KEY, workspace, experimentId, counted],
    queryFn: async ({ signal }) => {
      const counts = await Promise.all(
        counted.map(async (runId) => {
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

export const useScheduledDetectionRuns = (experimentId: string, { enabled = true } = {}) => {
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
    filter: SCHEDULED_DETECTION_RUN_FILTER,
  });

  const rows = useMemo(
    () => (runs ?? []).map(toScheduledRunRow).filter((row): row is ScheduledRunRow => row !== undefined),
    [runs],
  );
  const runIds = useMemo(() => rows.map((row) => row.runId), [rows]);
  const newIssueCounts = useNewIssueCounts(experimentId, runIds, enabled && rows.length > 0);
  // `Date.now()` is read once per render rather than per row, so every duration and the summary
  // window are measured from the same instant.
  const summary = useMemo(() => summarizeScheduledRuns(rows, Date.now()), [rows]);

  return {
    rows,
    newIssueCounts,
    summary,
    hasNextPage,
    fetchNextPage,
    isLoading,
    isFetching,
    error,
    refetch,
  };
};
