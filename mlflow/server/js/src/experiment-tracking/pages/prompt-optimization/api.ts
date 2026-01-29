import { matchPredefinedError, UnknownError } from '@databricks/web-shared/errors';
import { fetchEndpoint } from '../../../common/utils/FetchUtils';
import type {
  CreateOptimizationJobPayload,
  CreateOptimizationJobResponse,
  GetOptimizationJobResponse,
  SearchOptimizationJobsResponse,
  CancelOptimizationJobResponse,
} from './types';

const defaultErrorHandler = async ({
  reject,
  response,
  err: originalError,
}: {
  reject: (cause: any) => void;
  response: Response;
  err: Error;
}) => {
  const predefinedError = matchPredefinedError(response);
  const error = predefinedError instanceof UnknownError ? originalError : predefinedError;
  if (response) {
    try {
      const messageFromResponse = (await response.json())?.message;
      if (messageFromResponse) {
        error.message = messageFromResponse;
      }
    } catch {
      // Keep original error message if we fail to extract
    }
  }
  reject(error);
};

export const PromptOptimizationApi = {
  /**
   * Create a new prompt optimization job
   */
  createJob: (payload: CreateOptimizationJobPayload) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/prompt-optimization/jobs',
      method: 'POST',
      body: JSON.stringify(payload),
      error: defaultErrorHandler,
    }) as Promise<CreateOptimizationJobResponse>;
  },

  /**
   * Get details of an optimization job
   */
  getJob: (jobId: string) => {
    return fetchEndpoint({
      relativeUrl: `ajax-api/3.0/mlflow/prompt-optimization/jobs/${encodeURIComponent(jobId)}`,
      method: 'GET',
      error: defaultErrorHandler,
    }) as Promise<GetOptimizationJobResponse>;
  },

  /**
   * Search for optimization jobs in an experiment
   */
  searchJobs: (experimentId: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/prompt-optimization/jobs/search',
      method: 'POST',
      body: JSON.stringify({ experiment_id: experimentId }),
      error: defaultErrorHandler,
    }) as Promise<SearchOptimizationJobsResponse>;
  },

  /**
   * Cancel an in-progress optimization job
   */
  cancelJob: (jobId: string) => {
    return fetchEndpoint({
      relativeUrl: `ajax-api/3.0/mlflow/prompt-optimization/jobs/${encodeURIComponent(jobId)}/cancel`,
      method: 'POST',
      error: defaultErrorHandler,
    }) as Promise<CancelOptimizationJobResponse>;
  },

  /**
   * Delete an optimization job
   */
  deleteJob: (jobId: string) => {
    return fetchEndpoint({
      relativeUrl: `ajax-api/3.0/mlflow/prompt-optimization/jobs/${encodeURIComponent(jobId)}`,
      method: 'DELETE',
      error: defaultErrorHandler,
    }) as Promise<void>;
  },
};
