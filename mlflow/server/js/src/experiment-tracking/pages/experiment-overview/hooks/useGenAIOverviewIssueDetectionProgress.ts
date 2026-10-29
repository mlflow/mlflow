import { useCallback } from 'react';
import { useCancelJob } from '../../../components/run-page/hooks/useCancelJob';
import { isJobComplete, JobStatus } from '../../../components/run-page/hooks/useFetchJobStatus';
import { useLatestIssueDetection } from '../../experiment-issues/useIssueDetection';

export type GenAIOverviewIssueDetectionPhase = 'idle' | 'starting' | 'running';

export const useGenAIOverviewIssueDetectionProgress = (experimentId: string) => {
  const detection = useLatestIssueDetection(experimentId);
  const { cancelJobAsync, isCancelling, error: cancelError } = useCancelJob();
  const isRunning = Boolean(detection.run) && !isJobComplete(detection.status);
  const phase: GenAIOverviewIssueDetectionPhase = !isRunning
    ? 'idle'
    : detection.status === undefined || detection.status === JobStatus.PENDING
      ? 'starting'
      : 'running';

  const cancelDetection = useCallback(async () => {
    if (!detection.jobId || !detection.run) return;
    await cancelJobAsync({ jobId: detection.jobId, runUuid: detection.run.info.run_id });
  }, [cancelJobAsync, detection.jobId, detection.run]);

  return {
    phase,
    canStartDetection: !isRunning,
    cancelDetection: isRunning && detection.jobId ? cancelDetection : undefined,
    isCancelling,
    cancelError,
  };
};
