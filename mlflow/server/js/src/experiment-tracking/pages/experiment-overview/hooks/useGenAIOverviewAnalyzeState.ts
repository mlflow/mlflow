import { useCallback, useMemo, useState } from 'react';
import { useQuery } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import { useActiveWorkspace } from '../../../../workspaces/utils/WorkspaceUtils';
import type { RunEntity } from '../../../types';
import { MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION } from '../../../constants';
import { useExperimentEvaluationRunsData } from '../../../components/experiment-page/hooks/useExperimentEvaluationRunsData';
import type { Issue } from '../../../components/run-page/hooks/useSearchIssuesQuery';
import { SEARCH_ISSUES_QUERY_KEY } from '../../../components/run-page/hooks/useSearchIssuesQuery';
import { useMonitoringConfig } from '../../../hooks/useMonitoringConfig';
import { useServerInfo } from '../../../hooks/useServerInfo';
import { generateTimeBuckets } from '../utils/chartUtils';
import { createGenAIOverviewTimeRange } from '../utils/timeUtils';

const DAY_IN_SECONDS = 24 * 60 * 60;
const ISSUE_SEARCH_PAGE_SIZE = 1000;
const COMPLETED_ISSUE_DETECTION_RUN_FILTER =
  `tags.\`${MLFLOW_RUN_TYPE_TAG}\` = '${MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION}'` +
  ` AND attributes.status = 'FINISHED'`;

interface SearchIssuesResponse {
  issues?: Issue[];
}

interface GenAIOverviewIssuesResult {
  issues: Issue[];
}

export const fetchGenAIOverviewIssues = async (
  experimentId: string,
  signal?: AbortSignal,
): Promise<GenAIOverviewIssuesResult> => {
  const response = (await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/issues/search'), {
    method: 'POST',
    signal,
    body: {
      experiment_id: experimentId,
      statuses: ['pending', 'resolved'],
      severities: ['low', 'medium', 'high'],
      order_by: ['created_timestamp DESC'],
      max_results: ISSUE_SEARCH_PAGE_SIZE,
      view: 'basic',
    },
  })) as SearchIssuesResponse;

  return {
    issues: response.issues ?? [],
  };
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
  | { status: 'error' };

const getUtcDayStart = (timestampMs: number) => {
  const date = new Date(timestampMs);
  return Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate());
};

const getIssueDetectionRunCompletionTime = (run: RunEntity) => run.info.endTime || run.info.startTime;

export const resolveGenAIOverviewAnalyzeState = ({
  latestCompletedRun,
  isLoading,
  error,
  issues,
  startTimeMs,
  endTimeMs,
  recentIssueStartTimeMs,
}: {
  latestCompletedRun?: RunEntity;
  isLoading: boolean;
  error: unknown;
  issues: Issue[];
  startTimeMs: number;
  endTimeMs: number;
  recentIssueStartTimeMs: number;
}): GenAIOverviewAnalyzeState => {
  if (isLoading) return { status: 'loading' };
  if (error) return { status: 'error' };
  if (!latestCompletedRun) return { status: 'never-run' };

  const activityByTimestamp = new Map<number, GenAIOverviewIssueActivityPoint>(
    generateTimeBuckets(startTimeMs, endTimeMs, DAY_IN_SECONDS).map((timestampMs) => [
      timestampMs,
      { timestampMs, count: 0, high: 0, medium: 0, low: 0 },
    ]),
  );
  let issuesCreatedInLastSevenDays = 0;
  for (const issue of issues) {
    if (issue.status === 'rejected' || issue.severity === 'not_an_issue') continue;
    if (issue.created_timestamp >= recentIssueStartTimeMs) issuesCreatedInLastSevenDays += 1;
    const activityPoint = activityByTimestamp.get(getUtcDayStart(issue.created_timestamp));
    if (!activityPoint) continue;
    const severity = issue.severity === 'high' || issue.severity === 'low' ? issue.severity : 'medium';
    activityPoint[severity] += 1;
    activityPoint.count += 1;
  }
  const activity = [...activityByTimestamp.values()];

  return {
    status: 'completed',
    runUuid: latestCompletedRun.info.runUuid,
    latestRunCompletedAtMs: getIssueDetectionRunCompletionTime(latestCompletedRun),
    issueCount: activity.reduce((count, point) => count + point.count, 0),
    issuesCreatedInLastSevenDays,
    activity,
  };
};

export const useGenAIOverviewAnalyzeState = (experimentId: string, useRollingSevenDayRange = true) => {
  const workspace = useActiveWorkspace();
  const { dateNow } = useMonitoringConfig();
  const { data: serverInfo, isLoading: isLoadingServerInfo } = useServerInfo();
  const [legacyTimeRange] = useState(() => createGenAIOverviewTimeRange(new Date(), false));
  const rollingTimeRange = useMemo(() => createGenAIOverviewTimeRange(dateNow, true), [dateNow]);
  const timeRange = useRollingSevenDayRange ? rollingTimeRange : legacyTimeRange;
  const canSearchIssues = serverInfo?.store_type === 'SqlStore';
  const latestCompletedRun = useExperimentEvaluationRunsData({
    experimentId,
    enabled: Boolean(experimentId),
    filter: COMPLETED_ISSUE_DETECTION_RUN_FILTER,
  });
  const overviewIssues = useQuery<GenAIOverviewIssuesResult, Error>({
    queryKey: [SEARCH_ISSUES_QUERY_KEY, 'overview', workspace, experimentId],
    queryFn: ({ signal }) => fetchGenAIOverviewIssues(experimentId, signal),
    enabled: Boolean(experimentId) && canSearchIssues,
    refetchOnWindowFocus: false,
    retry: false,
  });

  const state = useMemo(
    () =>
      resolveGenAIOverviewAnalyzeState({
        latestCompletedRun: latestCompletedRun.data[0],
        isLoading: latestCompletedRun.isLoading || isLoadingServerInfo || (canSearchIssues && overviewIssues.isLoading),
        error: latestCompletedRun.error ?? (canSearchIssues ? overviewIssues.error : null),
        issues: canSearchIssues ? (overviewIssues.data?.issues ?? []) : [],
        startTimeMs: timeRange.startTimeMs,
        endTimeMs: timeRange.endTimeMs,
        recentIssueStartTimeMs: rollingTimeRange.startTimeMs,
      }),
    [
      canSearchIssues,
      isLoadingServerInfo,
      latestCompletedRun.data,
      latestCompletedRun.error,
      latestCompletedRun.isLoading,
      overviewIssues.data,
      overviewIssues.error,
      overviewIssues.isLoading,
      rollingTimeRange.startTimeMs,
      timeRange.endTimeMs,
      timeRange.startTimeMs,
    ],
  );

  const { refetch: refetchLatestCompletedRun } = latestCompletedRun;
  const { refetch: refetchOverviewIssues } = overviewIssues;
  const refetchState = useCallback(async () => {
    if (canSearchIssues) {
      await Promise.all([refetchLatestCompletedRun(), refetchOverviewIssues()]);
    } else {
      await refetchLatestCompletedRun();
    }
  }, [canSearchIssues, refetchLatestCompletedRun, refetchOverviewIssues]);

  return { state, refetch: refetchState };
};
