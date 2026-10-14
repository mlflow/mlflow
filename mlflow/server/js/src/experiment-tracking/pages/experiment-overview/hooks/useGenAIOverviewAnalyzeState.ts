import { useCallback, useMemo, useState } from 'react';
import { useQueries, useQuery, useQueryClient } from '@databricks/web-shared/query-client';
import { ErrorWrapper } from '../../../../common/utils/ErrorWrapper';
import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import type { RunEntity, SearchRunsApiResponse } from '../../../types';
import { MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION } from '../../../constants';
import {
  SEARCH_RUNS_QUERY_KEY,
  useExperimentEvaluationRunsData,
} from '../../../components/experiment-page/hooks/useExperimentEvaluationRunsData';
import type { Issue } from '../../../components/run-page/hooks/useSearchIssuesQuery';
import { SEARCH_ISSUES_QUERY_KEY } from '../../../components/run-page/hooks/useSearchIssuesQuery';
import { useMonitoringConfig } from '../../../hooks/useMonitoringConfig';
import { MlflowService } from '../../../sdk/MlflowService';
import { generateTimeBuckets } from '../utils/chartUtils';
import { createGenAIOverviewTimeRange } from '../utils/timeUtils';

const DAY_IN_SECONDS = 24 * 60 * 60;
const ISSUE_SEARCH_PAGE_SIZE = 1000;
const COMPLETED_ISSUE_DETECTION_RUN_FILTER =
  `tags.\`${MLFLOW_RUN_TYPE_TAG}\` = '${MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION}'` +
  ` AND attributes.status = 'FINISHED'`;
const LATEST_READABLE_ISSUE_DETECTION_RUN_QUERY_KEY = 'LATEST_READABLE_ISSUE_DETECTION_RUN';
const OVERVIEW_ISSUES_QUERY_KEY = 'GENAI_OVERVIEW_ISSUES';

interface SearchIssuesResponse {
  issues?: Issue[];
  next_page_token?: string;
}

const getIssueDetectionRunCompletionTime = (run: RunEntity) => run.info.endTime || run.info.startTime;

const isMissingIssueDetectionResult = (error: unknown) =>
  error instanceof ErrorWrapper && Number(error.getStatus()) === 404;

export const fetchIssueDetectionIssues = async (experimentId: string, runUuid: string): Promise<Issue[]> => {
  const issues: Issue[] = [];
  let pageToken: string | undefined;
  do {
    const response = (await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/issues/search'), {
      method: 'POST',
      body: {
        experiment_id: experimentId,
        filter_string: `source_run_id = '${runUuid}'`,
        include_trace_count: true,
        max_results: ISSUE_SEARCH_PAGE_SIZE,
        page_token: pageToken,
      },
    })) as SearchIssuesResponse;
    issues.push(...(response.issues ?? []));
    pageToken = response.next_page_token;
  } while (pageToken);
  return issues;
};

export interface GenAIOverviewLatestReadableIssueDetectionRun {
  runUuid: string;
  completedAtMs: number;
}

export interface LatestReadableIssueDetectionRunDependencies {
  searchRuns: (experimentId: string, pageToken?: string) => Promise<SearchRunsApiResponse>;
  fetchIssues: (experimentId: string, runUuid: string) => Promise<Issue[]>;
}

const latestReadableIssueDetectionRunDependencies: LatestReadableIssueDetectionRunDependencies = {
  searchRuns: (experimentId, pageToken) =>
    MlflowService.searchRuns({
      experiment_ids: [experimentId],
      filter: COMPLETED_ISSUE_DETECTION_RUN_FILTER,
      run_view_type: 'ACTIVE_ONLY',
      order_by: ['attributes.start_time DESC'],
      max_results: 1000,
      page_token: pageToken,
    }),
  fetchIssues: fetchIssueDetectionIssues,
};

export const findLatestReadableIssueDetectionRun = async (
  experimentId: string,
  dependencies: LatestReadableIssueDetectionRunDependencies = latestReadableIssueDetectionRunDependencies,
): Promise<GenAIOverviewLatestReadableIssueDetectionRun | undefined> => {
  const completedRuns: RunEntity[] = [];
  let pageToken: string | undefined;
  do {
    const response = await dependencies.searchRuns(experimentId, pageToken);
    completedRuns.push(...(response.runs ?? []).filter((run) => run.info.status === 'FINISHED'));
    pageToken = response.next_page_token;
  } while (pageToken);

  completedRuns.sort(
    (left, right) =>
      getIssueDetectionRunCompletionTime(right) - getIssueDetectionRunCompletionTime(left) ||
      right.info.startTime - left.info.startTime,
  );
  for (const run of completedRuns) {
    try {
      await dependencies.fetchIssues(experimentId, run.info.runUuid);
      return {
        runUuid: run.info.runUuid,
        completedAtMs: getIssueDetectionRunCompletionTime(run),
      };
    } catch (error) {
      if (!isMissingIssueDetectionResult(error)) {
        throw error;
      }
    }
  }
  return undefined;
};

export interface GenAIOverviewIssueActivityPoint {
  timestampMs: number;
  count: number;
  high: number;
  medium: number;
  low: number;
}

export type GenAIOverviewAnalyzeState =
  | { status: 'loading' }
  | { status: 'never-run' }
  | {
      status: 'completed';
      runUuid: string;
      latestRunCompletedAtMs: number;
      issueCount: number;
      issuesCreatedInLastSevenDays: number;
      activity: GenAIOverviewIssueActivityPoint[];
    }
  | { status: 'unavailable'; runUuid?: string }
  | { status: 'error' };

const getUtcDayStart = (timestampMs: number) => {
  const date = new Date(timestampMs);
  return Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate());
};

export const resolveGenAIOverviewAnalyzeState = ({
  runs,
  isLoading,
  error,
  issuesByRunUuid,
  areIssuesLoading,
  startTimeMs,
  endTimeMs,
  recentIssueStartTimeMs,
  latestReadableRun,
  isLatestReadableRunLoading,
  latestReadableRunError,
}: {
  runs: RunEntity[];
  isLoading: boolean;
  error: unknown;
  issuesByRunUuid: ReadonlyMap<string, Issue[]>;
  areIssuesLoading: boolean;
  startTimeMs: number;
  endTimeMs: number;
  recentIssueStartTimeMs: number;
  latestReadableRun?: GenAIOverviewLatestReadableIssueDetectionRun;
  isLatestReadableRunLoading: boolean;
  latestReadableRunError: unknown;
}): GenAIOverviewAnalyzeState => {
  if (isLoading || isLatestReadableRunLoading || areIssuesLoading) return { status: 'loading' };
  if (error) return { status: 'error' };

  const completedRuns = runs.filter((run) => run.info.status === 'FINISHED');
  if (!latestReadableRun) {
    if (latestReadableRunError) return { status: 'error' };
    return completedRuns.length > 0
      ? { status: 'unavailable', runUuid: completedRuns[0].info.runUuid }
      : { status: 'never-run' };
  }

  const activityByTimestamp = new Map<number, GenAIOverviewIssueActivityPoint>(
    generateTimeBuckets(startTimeMs, endTimeMs, DAY_IN_SECONDS).map((timestampMs) => [
      timestampMs,
      { timestampMs, count: 0, high: 0, medium: 0, low: 0 },
    ]),
  );
  let issueCount = 0;
  let issuesCreatedInLastSevenDays = 0;
  for (const run of completedRuns) {
    const activityPoint = activityByTimestamp.get(getUtcDayStart(run.info.startTime));
    const issues = issuesByRunUuid.get(run.info.runUuid);
    if (!activityPoint || !issues) continue;
    const visibleIssues = issues.filter((issue) => issue.status !== 'rejected' && issue.severity !== 'not_an_issue');
    if (run.info.startTime >= recentIssueStartTimeMs) {
      issuesCreatedInLastSevenDays += visibleIssues.length;
    }
    for (const issue of visibleIssues) {
      const severity = issue.severity === 'high' || issue.severity === 'low' ? issue.severity : 'medium';
      activityPoint[severity] += 1;
      activityPoint.count += 1;
      issueCount += 1;
    }
  }

  return {
    status: 'completed',
    runUuid: latestReadableRun.runUuid,
    latestRunCompletedAtMs: latestReadableRun.completedAtMs,
    issueCount,
    issuesCreatedInLastSevenDays,
    activity: [...activityByTimestamp.values()],
  };
};

export const useGenAIOverviewAnalyzeState = (experimentId: string, useRollingSevenDayRange = true) => {
  const queryClient = useQueryClient();
  const { dateNow } = useMonitoringConfig();
  const [legacyTimeRange] = useState(() => createGenAIOverviewTimeRange(new Date(), false));
  const rollingTimeRange = useMemo(() => createGenAIOverviewTimeRange(dateNow, true), [dateNow]);
  const timeRange = useRollingSevenDayRange ? rollingTimeRange : legacyTimeRange;
  const issueDetectionFilter = `${COMPLETED_ISSUE_DETECTION_RUN_FILTER} AND attributes.start_time >= ${timeRange.startTimeMs}`;
  const { data, isLoading, error } = useExperimentEvaluationRunsData({
    experimentId,
    enabled: Boolean(experimentId),
    filter: issueDetectionFilter,
    fetchAllPages: true,
  });
  const {
    data: latestReadableRun,
    isLoading: isLatestReadableRunLoading,
    error: latestReadableRunError,
  } = useQuery<GenAIOverviewLatestReadableIssueDetectionRun | undefined, Error>({
    queryKey: [LATEST_READABLE_ISSUE_DETECTION_RUN_QUERY_KEY, experimentId],
    queryFn: () => findLatestReadableIssueDetectionRun(experimentId),
    enabled: Boolean(experimentId),
    refetchOnWindowFocus: false,
    retry: false,
  });
  const completedRuns = useMemo(() => data.filter((run) => run.info.status === 'FINISHED'), [data]);
  const issueResults = useQueries({
    queries: completedRuns.map((run) => ({
      queryKey: [OVERVIEW_ISSUES_QUERY_KEY, experimentId, run.info.runUuid],
      queryFn: () => fetchIssueDetectionIssues(experimentId, run.info.runUuid),
      refetchOnWindowFocus: false,
      retry: false,
    })),
  });
  const { issuesByRunUuid, areIssuesLoading, issuesError } = useMemo(
    () => ({
      issuesByRunUuid: new Map(
        issueResults.flatMap((result, index) =>
          result.data ? [[completedRuns[index].info.runUuid, result.data] as const] : [],
        ),
      ),
      areIssuesLoading: issueResults.some((result) => result.isLoading),
      issuesError: issueResults.find((result) => result.error)?.error,
    }),
    [completedRuns, issueResults],
  );

  const state = useMemo(
    () =>
      resolveGenAIOverviewAnalyzeState({
        runs: data,
        isLoading,
        error: error ?? issuesError,
        issuesByRunUuid,
        areIssuesLoading,
        startTimeMs: timeRange.startTimeMs,
        endTimeMs: timeRange.endTimeMs,
        recentIssueStartTimeMs: rollingTimeRange.startTimeMs,
        latestReadableRun,
        isLatestReadableRunLoading,
        latestReadableRunError,
      }),
    [
      areIssuesLoading,
      data,
      error,
      isLatestReadableRunLoading,
      isLoading,
      issuesByRunUuid,
      issuesError,
      latestReadableRun,
      latestReadableRunError,
      rollingTimeRange.startTimeMs,
      timeRange.endTimeMs,
      timeRange.startTimeMs,
    ],
  );

  const refetchState = useCallback(async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: [SEARCH_RUNS_QUERY_KEY, experimentId] }),
      queryClient.invalidateQueries({ queryKey: [SEARCH_ISSUES_QUERY_KEY] }),
      queryClient.invalidateQueries({ queryKey: [OVERVIEW_ISSUES_QUERY_KEY, experimentId] }),
      queryClient.invalidateQueries({ queryKey: [LATEST_READABLE_ISSUE_DETECTION_RUN_QUERY_KEY, experimentId] }),
    ]);
  }, [experimentId, queryClient]);

  return { state, refetch: refetchState };
};
