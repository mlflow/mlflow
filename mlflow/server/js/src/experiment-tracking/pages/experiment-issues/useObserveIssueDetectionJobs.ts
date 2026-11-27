import { useEffect, useRef } from 'react';
import { useQuery } from '@databricks/web-shared/query-client';
import { useActiveWorkspace } from '../../../workspaces/utils/WorkspaceUtils';
import { shouldEnableIssueDetection } from '../../../common/utils/FeatureUtils';
import { MlflowService } from '../../sdk/MlflowService';
import type { RunEntity } from '../../types';
import { MLFLOW_ISSUE_DETECTION_JOB_ID_TAG } from '../../constants';
import { ISSUE_DETECTION_JOB_OBSERVED_EVENT } from '../../components/experiment-page/components/traces-v3/IssueDetectionJobNotifications';
import { DETECTION_RUN_FILTER, RUNNING_POLL_INTERVAL_MS } from './detectionRunHistory';

export const useObserveIssueDetectionJobs = (experimentId: string) => {
  const workspace = useActiveWorkspace();
  const observed = useRef(new Set<string>());
  const { data } = useQuery({
    queryKey: ['ACTIVE_ISSUE_DETECTION_JOBS', workspace, experimentId],
    queryFn: async () => {
      const runs: RunEntity[] = [];
      let pageToken: string | undefined;
      do {
        const page = await MlflowService.searchRuns({
          experiment_ids: [experimentId],
          filter: `${DETECTION_RUN_FILTER} AND attributes.status = 'RUNNING'`,
          run_view_type: 'ACTIVE_ONLY',
          max_results: 1000,
          page_token: pageToken,
        });
        runs.push(...(page.runs ?? []));
        pageToken = page.next_page_token;
      } while (pageToken);
      return runs;
    },
    retry: false,
    enabled: shouldEnableIssueDetection(),
    refetchInterval: RUNNING_POLL_INTERVAL_MS,
  });

  useEffect(() => {
    for (const run of data ?? []) {
      const jobId = run.data?.tags?.find((tag) => tag.key === MLFLOW_ISSUE_DETECTION_JOB_ID_TAG)?.value;
      const key = JSON.stringify([workspace, experimentId, jobId]);
      if (!jobId || !run.info?.runUuid || observed.current.has(key)) continue;
      observed.current.add(key);
      window.dispatchEvent(
        new CustomEvent(ISSUE_DETECTION_JOB_OBSERVED_EVENT, {
          detail: {
            experimentId,
            workspace,
            jobId,
            runId: run.info.runUuid,
            traceCount: 0,
            submittedAtMs: Date.now(),
            startTimeMs: run.info.startTime ?? 0,
          },
        }),
      );
    }
  }, [data, experimentId, workspace]);
};
