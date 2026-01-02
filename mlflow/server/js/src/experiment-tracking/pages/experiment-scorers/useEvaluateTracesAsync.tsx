import { fetchAPI, getAjaxUrl } from '../../../common/utils/FetchUtils';
import { useMutation, useQueryClient } from '../../../common/utils/reactQueryHooks';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FeedbackAssessment, isV3ModelTraceInfo, ModelTrace } from '../../../shared/web-shared/model-trace-explorer';
import { EvaluateTracesParams } from './types';
import { useGetTraceIdsForEvaluation } from './useGetTracesForEvaluation';
import { getMlflowTraceV3ForEvaluation, JudgeEvaluationResult } from './useEvaluateTraces.common';
import {
  TrackingJobQueryResult,
  TrackingJobStatus,
  useGetTrackingServerJobStatus,
} from '../../../common/hooks/useGetTrackingServerJobStatus';
import { compact, zipObject } from 'lodash';

type EvaluateTracesAsyncJobResult = {
  [traceId: string]: {
    assessments: FeedbackAssessment[];
    failures?: {
      error_code: string;
      error_message: string;
    }[];
  };
};

const JOB_POLLING_INTERVAL = 1500;

const isJobRunning = (jobData?: TrackingJobQueryResult<EvaluateTracesAsyncJobResult>): boolean => {
  return jobData?.status === TrackingJobStatus.RUNNING || jobData?.status === TrackingJobStatus.PENDING;
};

type StartEvaluationJobParams = { evaluateParams: EvaluateTracesParams; traceIds: string[] };
type StartEvaluationJobResponse = { jobs: { job_id: string }[] };
export const useEvaluateTracesAsync = ({ onScorerFinished }: { onScorerFinished?: () => void }) => {
  const queryClient = useQueryClient();
  const [currentJobsId, setCurrentJobsId] = useState<string[] | undefined>(undefined);
  const getTraceIdsForEvaluation = useGetTraceIdsForEvaluation();

  const [tracesData, setTracesData] = useState<Record<string, ModelTrace> | undefined>();
  const lastFinishedJobsIdRef = useRef<string[] | undefined>(undefined);

  const { jobResults, areJobsRunning } = useGetTrackingServerJobStatus<EvaluateTracesAsyncJobResult>(currentJobsId, {
    enabled: Boolean(currentJobsId),
    refetchInterval: (data) => {
      if (data?.some((job) => isJobRunning(job))) {
        return JOB_POLLING_INTERVAL;
      }
      return false;
    },
    refetchOnWindowFocus: false,
    refetchOnMount: false,
  });

  // Call onScorerFinished when the job successfully completes (once per job)
  useEffect(() => {
    if (!areJobsRunning && currentJobsId && lastFinishedJobsIdRef.current !== currentJobsId) {
      lastFinishedJobsIdRef.current = currentJobsId;
      onScorerFinished?.();
    }
  }, [areJobsRunning, currentJobsId, onScorerFinished]);

  const {
    mutate: startEvaluationJob,
    isLoading: isJobStarting,
    error: startEvaluationJobError,
  } = useMutation<StartEvaluationJobResponse, Error, StartEvaluationJobParams>({
    mutationFn: async ({ evaluateParams, traceIds }) => {
      const responseData = await fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/scorer/invoke`), 'POST', {
        experiment_id: evaluateParams.experimentId,
        serialized_scorer: evaluateParams.serializedScorer,
        trace_ids: traceIds,
      });
      return responseData;
    },
    onSuccess: (data) => {
      setCurrentJobsId(data.jobs.map((job) => job.job_id));
    },
  });

  const isLoading = areJobsRunning || isJobStarting;

  const evaluateTracesAsync = useCallback(
    async (params: EvaluateTracesParams) => {
      const traceIds = await getTraceIdsForEvaluation(params);

      const serializedScorer = params.serializedScorer;

      if (!serializedScorer) {
        throw new Error('The serialized scorer is malformed');
      }

      // Start the evaluation job
      startEvaluationJob({ evaluateParams: params, traceIds });

      // After the job is started, fetch all the traces in parallel
      const traces = await Promise.all(
        traceIds.map(async (traceId) => {
          return await queryClient.fetchQuery({
            queryKey: ['GetMlflowTraceV3', traceId],
            queryFn: () => getMlflowTraceV3ForEvaluation(traceId),
            staleTime: Infinity,
            cacheTime: Infinity,
          });
        }),
      );
      setTracesData(zipObject(traceIds, traces));
    },
    [startEvaluationJob, getTraceIdsForEvaluation, queryClient],
  );

  const reset = useCallback(() => {
    setCurrentJobsId(undefined);
    setTracesData(undefined);
    lastFinishedJobsIdRef.current = undefined;
  }, []);

  const error = useMemo(() => {
    if (startEvaluationJobError) {
      return startEvaluationJobError;
    }
    if (areJobsRunning) {
      return null;
    }
    for (const job of Object.values(jobResults ?? {})) {
      if (job.status === TrackingJobStatus.FAILED) {
        return new Error(job.result);
      }
    }
    return null;
  }, [jobResults, areJobsRunning, startEvaluationJobError]);

  // Combine the traces data and the evaluation job data to get the results
  const data = useMemo<JudgeEvaluationResult[] | null>(() => {
    if (!tracesData || isLoading || error) {
      return null;
    }
    const aggregatedJobResults: EvaluateTracesAsyncJobResult = {};
    // Get data from all successful jobs
    for (const job of Object.values(jobResults ?? {})) {
      if (job.status === TrackingJobStatus.SUCCEEDED) {
        for (const traceId of Object.keys(job.result)) {
          aggregatedJobResults[traceId] = {
            assessments: job.result[traceId]?.assessments ?? [],
            failures: job.result[traceId]?.failures ?? [],
          };
        }
      }
    }

    const evaluationResults = Object.entries(tracesData).map(([traceId, trace]) => {
      const results = aggregatedJobResults[traceId]?.assessments ?? [];
      const failureString = aggregatedJobResults[traceId]?.failures?.map((failure) => failure.error_message).join(', ');
      return {
        trace,
        results,
        error: failureString || null,
      };
    });

    return compact(evaluationResults);
  }, [jobResults, isLoading, tracesData, error]);

  return [evaluateTracesAsync, { data, isLoading, error, reset }] as const;
};
