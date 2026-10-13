import { JobStatus } from '../components/run-page/hooks/useFetchJobStatus';
import type { RunInfoEntity } from '../types';

/**
 * Maps MLflow run status to job status.
 * Used when displaying job-like progress for runs that don't have an associated job.
 *
 * @param runStatus - The status of the MLflow run
 * @returns The corresponding job status, or undefined if the run status doesn't map to a job status
 */
export const runStatusToJobStatus = (runStatus: RunInfoEntity['status']): JobStatus | undefined => {
  switch (runStatus) {
    case 'FINISHED':
      return JobStatus.SUCCEEDED;
    case 'FAILED':
      return JobStatus.FAILED;
    case 'KILLED':
      return JobStatus.CANCELED;
    case 'RUNNING':
      return JobStatus.RUNNING;
    default:
      return undefined;
  }
};
