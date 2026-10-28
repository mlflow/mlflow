import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '../../../common/utils/FetchUtils';
import { useActiveWorkspace } from '../../../workspaces/utils/WorkspaceUtils';
import { searchMlflowTracesQueryFn } from '../../../shared/web-shared/genai-traces-table/hooks/useMlflowTraces';
import {
  getSubmittedIssueDetectionJobs,
  clearSubmittedIssueDetectionJob,
  ISSUE_DETECTION_JOB_COMPLETED_EVENT,
  ISSUE_DETECTION_JOB_SUBMITTED_EVENT,
  type SubmittedIssueDetectionJob,
} from '../../components/experiment-page/components/traces-v3/IssueDetectionJobNotifications';
import { SEARCH_ISSUES_QUERY_KEY } from '../../components/run-page/hooks/useSearchIssuesQuery';
import { GET_ISSUE_QUERY_KEY } from '../../components/run-page/hooks/useGetIssueQuery';
import { ISSUE_EVIDENCE_TRACES_QUERY_KEY } from './useIssueEvidenceTraces';
import { JobStatus, isJobComplete, useFetchJobStatus } from '../../components/run-page/hooks/useFetchJobStatus';
import {
  MLFLOW_ISSUE_DETECTION_JOB_ID_TAG,
  MLFLOW_ISSUE_DETECTION_RESULT_ISSUES_TAG,
  MLFLOW_ISSUE_DETECTION_RESULT_TOTAL_TRACES_TAG,
  MLFLOW_RUN_TYPE_TAG,
  MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION,
} from '../../constants';

export const LATEST_ISSUE_DETECTION_QUERY_KEY = 'LATEST_ISSUE_DETECTION';

/**
 * How long a RUNNING detection run may carry no job ID before it is treated as never started.
 *
 * The invoke handler creates the run and then tags it with the job ID in the same request, so
 * in practice the gap is milliseconds. A run that is still untagged much later means job
 * submission failed after the run existed -- and because every surface derives "a detection is
 * in flight" from the run alone, that left Detect disabled with nothing to cancel, on the
 * Traces tab, the Issues tab and the GenAI overview alike, until someone edited the database.
 *
 * Generous enough that a slow-but-successful submission is never mistaken for a failed one.
 */
const STALLED_SUBMISSION_MS = 2 * 60 * 1000;

interface DetectionRun {
  info: { run_id: string; status: string; start_time: number };
  data?: { tags?: { key: string; value: string }[] };
}

export const useLatestIssueDetection = (experimentId: string) => {
  const workspace = useActiveWorkspace();
  const queryClient = useQueryClient();
  const [submitted, setSubmitted] = useState(
    () =>
      getSubmittedIssueDetectionJobs()
        .filter((job) => job.experimentId === experimentId && (job.workspace ?? null) === workspace)
        .sort((a, b) => b.submittedAtMs - a.submittedAtMs)[0],
  );
  const completedRuns = useRef(new Set<string>());
  const refreshCompletedRun = useCallback(
    (runId: string) => {
      if (!completedRuns.current.has(runId)) {
        completedRuns.current.add(runId);
        queryClient.invalidateQueries([SEARCH_ISSUES_QUERY_KEY]);
        queryClient.invalidateQueries([LATEST_ISSUE_DETECTION_QUERY_KEY]);
        // The list alone is not enough once a run can match its findings to an existing issue:
        // that issue's row count grows while the open detail pane and its evidence stay cached,
        // so the pane shows one affected trace next to a count of two.
        queryClient.invalidateQueries([GET_ISSUE_QUERY_KEY]);
        queryClient.invalidateQueries([ISSUE_EVIDENCE_TRACES_QUERY_KEY]);
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
      (submitted && !completedRuns.current.has(submitted.runId)) ||
      (data?.runs?.[0]?.info.status === 'RUNNING' && !completedRuns.current.has(data.runs[0].info.run_id))
        ? 3000
        : false,
  });
  const latestRun = latest.data?.runs?.[0];
  // History can still contain an older terminal run immediately after submission.
  const run = useMemo(
    () =>
      submitted &&
      latestRun?.info.run_id !== submitted.runId &&
      (!latestRun || latestRun.info.start_time <= submitted.submittedAtMs)
        ? {
            info: { run_id: submitted.runId, status: 'RUNNING', start_time: submitted.submittedAtMs },
            data: { tags: [{ key: MLFLOW_ISSUE_DETECTION_JOB_ID_TAG, value: submitted.jobId }] },
          }
        : latestRun,
    [submitted, latestRun],
  );
  const jobId = run?.data?.tags?.find((tag) => tag.key === MLFLOW_ISSUE_DETECTION_JOB_ID_TAG)?.value;
  // A run whose submission failed is reported as no run at all, so every consumer's
  // `Boolean(run) && !isJobComplete(status)` stops claiming a detection is in flight. Done here
  // rather than in each consumer because all three compute that the same way, and two of them
  // would otherwise keep blocking.
  const submissionStalled =
    !jobId && run?.info.status === 'RUNNING' && Date.now() - (run?.info.start_time ?? 0) > STALLED_SUBMISSION_MS;
  const job = useFetchJobStatus({ jobId });
  const runStatus = (
    { FINISHED: JobStatus.SUCCEEDED, FAILED: JobStatus.FAILED, KILLED: JobStatus.CANCELED } as Record<string, JobStatus>
  )[run?.info.status ?? ''];
  const status = isJobComplete(job.status) ? job.status : (runStatus ?? job.status);
  useEffect(() => {
    if (run && isJobComplete(status)) {
      refreshCompletedRun(run.info.run_id);
      if (submitted?.runId === run.info.run_id) clearSubmittedIssueDetectionJob(submitted.jobId);
    }
  }, [status, refreshCompletedRun, run, submitted]);
  useEffect(() => {
    // Global notifications also observe older concurrent runs, beyond the latest run polled here.
    const onCompleted = (event: Event) => {
      const completed = (event as CustomEvent<SubmittedIssueDetectionJob>).detail;
      if (completed?.experimentId === experimentId && (completed.workspace ?? null) === workspace) {
        refreshCompletedRun(completed.runId);
      }
    };
    window.addEventListener(ISSUE_DETECTION_JOB_COMPLETED_EVENT, onCompleted);
    return () => window.removeEventListener(ISSUE_DETECTION_JOB_COMPLETED_EVENT, onCompleted);
  }, [experimentId, workspace, refreshCompletedRun]);
  useEffect(() => {
    const onSubmitted = (event: Event) => {
      const job = (event as CustomEvent<SubmittedIssueDetectionJob>).detail;
      if (job?.experimentId !== experimentId || (job.workspace ?? null) !== workspace) return;
      setSubmitted(job);
      queryClient.invalidateQueries([LATEST_ISSUE_DETECTION_QUERY_KEY, workspace, experimentId]);
    };
    window.addEventListener(ISSUE_DETECTION_JOB_SUBMITTED_EVENT, onSubmitted);
    return () => window.removeEventListener(ISSUE_DETECTION_JOB_SUBMITTED_EVENT, onSubmitted);
  }, [queryClient, workspace, experimentId]);
  const readCount = (key: string) => {
    const value = run?.data?.tags?.find((tag) => tag.key === key)?.value;
    const count = value === undefined ? undefined : Number(value);
    return count !== undefined && Number.isInteger(count) && count >= 0 ? count : undefined;
  };
  const jobResult =
    typeof job.result === 'object' && job.result !== null
      ? (job.result as { issues?: number; total_traces_analyzed?: number })
      : undefined;
  const result = {
    issues: jobResult?.issues ?? readCount(MLFLOW_ISSUE_DETECTION_RESULT_ISSUES_TAG),
    total_traces_analyzed:
      jobResult?.total_traces_analyzed ?? readCount(MLFLOW_ISSUE_DETECTION_RESULT_TOTAL_TRACES_TAG),
  };
  return {
    run: submissionStalled ? undefined : run,
    jobId,
    job,
    status,
    result,
    // Reported separately from `run` so a surface that wants to say "the last attempt never
    // started" can, rather than having to infer it from a run that is no longer reported.
    submissionStalled,
    isLoading: latest.isLoading,
    isFetching: latest.isFetching,
    error: latest.error,
    refetch: latest.refetch,
  };
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
