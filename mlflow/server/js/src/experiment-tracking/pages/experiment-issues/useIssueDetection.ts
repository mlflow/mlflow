import { useCallback, useEffect, useRef } from 'react';
import { useQuery, useQueryClient } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '../../../common/utils/FetchUtils';
import { useActiveWorkspace } from '../../../workspaces/utils/WorkspaceUtils';
import { searchMlflowTracesQueryFn } from '../../../shared/web-shared/genai-traces-table/hooks/useMlflowTraces';
import {
  ISSUE_DETECTION_JOB_COMPLETED_EVENT,
  ISSUE_DETECTION_JOB_SUBMITTED_EVENT,
  type SubmittedIssueDetectionJob,
} from '../../components/experiment-page/components/traces-v3/IssueDetectionJobNotifications';
import { SEARCH_ISSUES_QUERY_KEY } from '../../components/run-page/hooks/useSearchIssuesQuery';
import { JobStatus, isJobComplete, useFetchJobStatus } from '../../components/run-page/hooks/useFetchJobStatus';
import {
  MLFLOW_ISSUE_DETECTION_JOB_ID_TAG,
  MLFLOW_ISSUE_DETECTION_RESULT_ISSUES_TAG,
  MLFLOW_ISSUE_DETECTION_RESULT_TOTAL_TRACES_TAG,
  MLFLOW_RUN_TYPE_TAG,
  MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION,
} from '../../constants';

export const LATEST_ISSUE_DETECTION_QUERY_KEY = 'LATEST_ISSUE_DETECTION';

interface DetectionRun {
  info: { run_id: string; status: string; start_time: number };
  data?: { tags?: { key: string; value: string }[] };
}

export const useLatestIssueDetection = (experimentId: string) => {
  const workspace = useActiveWorkspace();
  const queryClient = useQueryClient();
  const completedRuns = useRef(new Set<string>());
  const refreshCompletedRun = useCallback(
    (runId: string) => {
      if (!completedRuns.current.has(runId)) {
        completedRuns.current.add(runId);
        queryClient.invalidateQueries([SEARCH_ISSUES_QUERY_KEY]);
        queryClient.invalidateQueries([LATEST_ISSUE_DETECTION_QUERY_KEY]);
      }
    },
    [queryClient],
  );
  const latest = useQuery<{ runs?: DetectionRun[] }, Error>({
    queryKey: [LATEST_ISSUE_DETECTION_QUERY_KEY, workspace, experimentId],
    queryFn: ({ signal }) =>
      fetchAPI(getAjaxUrl('ajax-api/2.0/mlflow/runs/search'), {
        method: 'POST',
        signal,
        body: {
          experiment_ids: [experimentId],
          filter: `tags.\`${MLFLOW_RUN_TYPE_TAG}\` = '${MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION}'`,
          max_results: 1,
          order_by: ['attributes.start_time DESC'],
        },
      }),
    retry: false,
    refetchInterval: (data) =>
      data?.runs?.[0]?.info.status === 'RUNNING' && !completedRuns.current.has(data.runs[0].info.run_id) ? 3000 : false,
  });
  const run = latest.data?.runs?.[0];
  const jobId = run?.data?.tags?.find((tag) => tag.key === MLFLOW_ISSUE_DETECTION_JOB_ID_TAG)?.value;
  const job = useFetchJobStatus({ jobId });
  const runStatus = (
    { FINISHED: JobStatus.SUCCEEDED, FAILED: JobStatus.FAILED, KILLED: JobStatus.CANCELED } as Record<string, JobStatus>
  )[run?.info.status ?? ''];
  const status = isJobComplete(job.status) ? job.status : (runStatus ?? job.status);
  useEffect(() => {
    if (run && isJobComplete(status)) {
      refreshCompletedRun(run.info.run_id);
    }
  }, [status, refreshCompletedRun, run]);
  useEffect(() => {
    // Global notifications also observe older concurrent runs, beyond the latest run polled here.
    const onCompleted = (event: Event) => {
      const completed = (event as CustomEvent<SubmittedIssueDetectionJob>).detail;
      if (completed?.experimentId === experimentId) refreshCompletedRun(completed.runId);
    };
    window.addEventListener(ISSUE_DETECTION_JOB_COMPLETED_EVENT, onCompleted);
    return () => window.removeEventListener(ISSUE_DETECTION_JOB_COMPLETED_EVENT, onCompleted);
  }, [experimentId, refreshCompletedRun]);
  useEffect(() => {
    const refresh = () => queryClient.invalidateQueries([LATEST_ISSUE_DETECTION_QUERY_KEY]);
    window.addEventListener(ISSUE_DETECTION_JOB_SUBMITTED_EVENT, refresh);
    return () => window.removeEventListener(ISSUE_DETECTION_JOB_SUBMITTED_EVENT, refresh);
  }, [queryClient]);
  const readCount = (key: string) => {
    const value = run?.data?.tags?.find((tag) => tag.key === key)?.value;
    const count = value === undefined ? undefined : Number(value);
    return count !== undefined && Number.isInteger(count) && count >= 0 ? count : undefined;
  };
  const result =
    typeof job.result === 'object' && job.result !== null
      ? (job.result as { issues?: number; total_traces_analyzed?: number })
      : !jobId
        ? {
            issues: readCount(MLFLOW_ISSUE_DETECTION_RESULT_ISSUES_TAG),
            total_traces_analyzed: readCount(MLFLOW_ISSUE_DETECTION_RESULT_TOTAL_TRACES_TAG),
          }
        : undefined;
  return { run, jobId, job, status, result, isLoading: latest.isLoading, error: latest.error };
};

export const useDetectionTraceIds = (experimentId: string, enabled: boolean) => {
  const workspace = useActiveWorkspace();
  return useQuery<string[], Error>({
    queryKey: ['ISSUE_DETECTION_DEFAULT_TRACES', workspace, experimentId],
    queryFn: async ({ signal }) => {
      const traces = await searchMlflowTracesQueryFn({
        signal,
        locations: [{ type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: experimentId } }],
        pageSize: 50,
        limit: 50,
        orderBy: ['timestamp DESC'],
      });
      return traces.map((trace) => trace.trace_id);
    },
    enabled,
    retry: false,
    cacheTime: 0,
  });
};
