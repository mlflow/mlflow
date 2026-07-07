export enum TrackingJobStatus {
  RUNNING = 'RUNNING',
  PENDING = 'PENDING',
  SUCCEEDED = 'SUCCEEDED',
  FAILED = 'FAILED',
  CANCELLED = 'CANCELLED',
  TIMEOUT = 'TIMEOUT',
}

export type TrackingJobQueryResult<ResultType> = (
  | {
      status:
        | TrackingJobStatus.RUNNING
        | TrackingJobStatus.PENDING
        | TrackingJobStatus.CANCELLED
        | TrackingJobStatus.TIMEOUT;
    }
  | {
      status: TrackingJobStatus.SUCCEEDED;
      // Actual result is present in the payload if the job succeeded
      result: ResultType;
    }
  | {
      status: TrackingJobStatus.FAILED;
      // In case of failure, the result is the error message
      result: string;
    }
) & {
  jobId: string;
};
