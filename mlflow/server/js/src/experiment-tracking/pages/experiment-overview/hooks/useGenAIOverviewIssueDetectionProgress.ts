import { useCallback, useEffect, useState } from 'react';
import {
  clearSubmittedIssueDetectionJob,
  getSubmittedIssueDetectionJobs,
  type SubmittedIssueDetectionJob,
} from '../../../components/experiment-page/components/traces-v3/IssueDetectionJobNotifications';
import { useCancelJob } from '../../../components/run-page/hooks/useCancelJob';
import { isJobComplete, JobStatus, useFetchJobStatus } from '../../../components/run-page/hooks/useFetchJobStatus';

export type GenAIOverviewIssueDetectionPhase = 'idle' | 'starting' | 'running';

const getTrackedJob = (experimentId: string) =>
  getSubmittedIssueDetectionJobs().find((job) => job.experimentId === experimentId);

export const useGenAIOverviewIssueDetectionProgress = (experimentId: string) => {
  const [job, setJob] = useState<SubmittedIssueDetectionJob | undefined>(() => getTrackedJob(experimentId));
  const { status } = useFetchJobStatus({ jobId: job?.jobId, enabled: Boolean(job) });
  const { cancelJobAsync, isCancelling } = useCancelJob();

  useEffect(() => {
    setJob(getTrackedJob(experimentId));
  }, [experimentId]);

  useEffect(() => {
    if (!job || !isJobComplete(status)) return;
    clearSubmittedIssueDetectionJob(job.jobId);
    setJob((currentJob) => (currentJob?.jobId === job.jobId ? undefined : currentJob));
  }, [job, status]);

  const registerJob = useCallback((submittedJob: SubmittedIssueDetectionJob) => {
    setJob(submittedJob);
  }, []);

  const cancelDetection = useCallback(async () => {
    if (!job) return;
    await cancelJobAsync({ jobId: job.jobId, runUuid: job.runId });
    clearSubmittedIssueDetectionJob(job.jobId);
    setJob(undefined);
  }, [cancelJobAsync, job]);

  const isRunning =
    Boolean(job) && (status === undefined || status === JobStatus.PENDING || status === JobStatus.RUNNING);

  return {
    phase: (isRunning ? 'running' : 'idle') as GenAIOverviewIssueDetectionPhase,
    canStartDetection: !isRunning,
    cancelDetection: isRunning ? cancelDetection : undefined,
    isCancelling,
    registerJob,
  };
};
