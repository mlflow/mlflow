import { useQuery } from '@databricks/web-shared/query-client';
import { MlflowService } from '../../../../../sdk/MlflowService';
import {
  MLFLOW_RUN_TYPE_TAG,
  MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION,
  MLFLOW_ISSUE_DETECTION_JOB_ID_TAG,
} from '../../../../../constants';

const ACTIVE_RUNS_POLL_INTERVAL_MS = 5000;

export interface ActiveIssueDetectionRun {
  runId: string;
  jobId?: string;
}

/**
 * Polls for the most recent issue detection run still in RUNNING state in the
 * experiment, so background jobs are discovered even after a page reload (or
 * when started by someone else). The run's job id comes from the
 * mlflow.issueDetection.jobId tag set by the invoke handler.
 */
export const useActiveIssueDetectionRun = ({
  experimentId,
  enabled = true,
}: {
  experimentId?: string;
  enabled?: boolean;
}): { activeRun: ActiveIssueDetectionRun | undefined } => {
  const { data } = useQuery({
    queryKey: ['ACTIVE_ISSUE_DETECTION_RUNS', experimentId],
    queryFn: async () =>
      MlflowService.searchRuns({
        experiment_ids: [experimentId],
        filter: `tags.\`${MLFLOW_RUN_TYPE_TAG}\` = '${MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION}' AND attributes.status = 'RUNNING'`,
        run_view_type: 'ACTIVE_ONLY',
        order_by: ['attributes.start_time DESC'],
        max_results: 1,
      }),
    refetchInterval: ACTIVE_RUNS_POLL_INTERVAL_MS,
    refetchOnWindowFocus: false,
    retry: false,
    enabled: enabled && Boolean(experimentId),
  });

  const run = data?.runs?.[0];
  const runId = run?.info?.runUuid;
  const jobId = run?.data?.tags?.find((tag: { key: string }) => tag.key === MLFLOW_ISSUE_DETECTION_JOB_ID_TAG)?.value;

  return { activeRun: runId ? { runId, jobId } : undefined };
};
