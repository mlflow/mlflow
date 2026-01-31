/**
 * Job status string values as returned by the backend API.
 * The backend returns these as string enum names from the protobuf definition.
 */
export const JobStatus = {
  UNSPECIFIED: 'JOB_STATUS_UNSPECIFIED',
  PENDING: 'JOB_STATUS_PENDING',
  IN_PROGRESS: 'JOB_STATUS_IN_PROGRESS',
  COMPLETED: 'JOB_STATUS_COMPLETED',
  FAILED: 'JOB_STATUS_FAILED',
  CANCELED: 'JOB_STATUS_CANCELED',
} as const;

export type JobStatusType = (typeof JobStatus)[keyof typeof JobStatus];

/**
 * Optimizer type string values as returned by the backend API.
 * The backend returns these as string enum names from the protobuf definition.
 */
export const OptimizerType = {
  UNSPECIFIED: 'OPTIMIZER_TYPE_UNSPECIFIED',
  GEPA: 'OPTIMIZER_TYPE_GEPA',
  METAPROMPT: 'OPTIMIZER_TYPE_METAPROMPT',
} as const;

export type OptimizerTypeValue = (typeof OptimizerType)[keyof typeof OptimizerType];

/**
 * Job state containing status and optional error message
 */
export interface JobState {
  status?: JobStatusType;
  error_message?: string;
  metadata?: Record<string, string>;
}

/**
 * Configuration for a prompt optimization job
 */
export interface PromptOptimizationJobConfig {
  optimizer_type?: OptimizerTypeValue;
  dataset_id?: string;
  scorers?: string[];
  optimizer_config_json?: string;
}

/**
 * Tag for a prompt optimization job
 */
export interface PromptOptimizationJobTag {
  key?: string;
  value?: string;
}

/**
 * Prompt optimization job entity
 */
export interface PromptOptimizationJob {
  job_id?: string;
  run_id?: string;
  state?: JobState;
  experiment_id?: string;
  source_prompt_uri?: string;
  optimized_prompt_uri?: string;
  config?: PromptOptimizationJobConfig;
  creation_timestamp_ms?: number;
  completion_timestamp_ms?: number;
  tags?: PromptOptimizationJobTag[];
  initial_eval_scores?: Record<string, number>;
  final_eval_scores?: Record<string, number>;
  progress?: number;
}

/**
 * Payload for creating a new optimization job
 */
export interface CreateOptimizationJobPayload {
  experiment_id: string;
  source_prompt_uri: string;
  config: {
    optimizer_type: OptimizerTypeValue;
    dataset_id?: string;
    scorers: string[];
    optimizer_config_json?: string;
  };
  tags?: PromptOptimizationJobTag[];
}

/**
 * Response from create job API
 */
export interface CreateOptimizationJobResponse {
  job?: PromptOptimizationJob;
}

/**
 * Response from get job API
 */
export interface GetOptimizationJobResponse {
  job?: PromptOptimizationJob;
}

/**
 * Response from search jobs API
 */
export interface SearchOptimizationJobsResponse {
  jobs?: PromptOptimizationJob[];
}

/**
 * Response from cancel job API
 */
export interface CancelOptimizationJobResponse {
  job?: PromptOptimizationJob;
}

/**
 * Helper function to get human-readable optimizer type name
 */
export const getOptimizerTypeName = (type?: OptimizerTypeValue): string => {
  switch (type) {
    case OptimizerType.GEPA:
      return 'GEPA';
    case OptimizerType.METAPROMPT:
      return 'MetaPrompt';
    default:
      return 'Unknown';
  }
};

/**
 * Helper function to get human-readable job status name
 */
export const getJobStatusName = (status?: JobStatusType): string => {
  switch (status) {
    case JobStatus.PENDING:
      return 'Pending';
    case JobStatus.IN_PROGRESS:
      return 'Running';
    case JobStatus.COMPLETED:
      return 'Completed';
    case JobStatus.FAILED:
      return 'Failed';
    case JobStatus.CANCELED:
      return 'Canceled';
    default:
      return 'Unknown';
  }
};

/**
 * Check if a job is in a running state (can be cancelled)
 */
export const isJobRunning = (status?: JobStatusType): boolean => {
  return status === JobStatus.PENDING || status === JobStatus.IN_PROGRESS;
};

/**
 * Check if a job is finalized (no more updates expected)
 */
export const isJobFinalized = (status?: JobStatusType): boolean => {
  return status === JobStatus.COMPLETED || status === JobStatus.FAILED || status === JobStatus.CANCELED;
};

/**
 * Get progress from job state metadata (stored as string "0.XX")
 */
export const getJobProgress = (job: PromptOptimizationJob): number | undefined => {
  // First check top-level progress field (if API returns it directly)
  if (job.progress !== undefined) {
    return job.progress;
  }
  // Fall back to metadata (backend stores progress as string in state.metadata)
  const progressStr = job.state?.metadata?.['progress'];
  if (progressStr) {
    const parsed = parseFloat(progressStr);
    return isNaN(parsed) ? undefined : parsed;
  }
  return undefined;
};
