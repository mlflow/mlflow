import { fetchAPI, getAjaxUrl } from '../../../common/utils/FetchUtils';
import { useMutation, useQueryClient } from '../../../common/utils/reactQueryHooks';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FeedbackAssessment, isV3ModelTraceInfo, ModelTrace } from '../../../shared/web-shared/model-trace-explorer';
import { EvaluateTracesParams } from './types';
import { useGetTraceIdsForEvaluation } from './useGetTracesForEvaluation';
import { getMlflowTraceV3ForEvaluation, JudgeEvaluationResult } from './useEvaluateTraces.common';
import {
  TrackingJobQueryResult,
  useGetTrackingServerJobStatus,
} from '../../../common/hooks/useGetTrackingServerJobStatus';
import { compact } from 'lodash';

type EvaluateTracesAsyncJobResult = {
  [traceId: string]: {
    assessments: FeedbackAssessment[];
  };
};

const JOB_POLLING_INTERVAL = 2500;

const isJobRunning = (jobData?: TrackingJobQueryResult<EvaluateTracesAsyncJobResult>): boolean => {
  return jobData?.status === 'RUNNING' || jobData?.status === 'PENDING';
};

type StartEvaluationJobParams = { evaluateParams: EvaluateTracesParams; traceIds: string[]; serializedScorer: string };
type StartEvaluationJobResponse = { jobs: { job_id: string }[] };
export const useEvaluateTracesAsync = ({
  onScorerFinished,
  getSerializedScorer,
}: {
  onScorerFinished?: () => void;
  getSerializedScorer?: () => string;
}) => {
  const queryClient = useQueryClient();
  const [currentJobsId, setCurrentJobsId] = useState<string[] | undefined>(undefined);
  const getTraceIdsForEvaluation = useGetTraceIdsForEvaluation();

  const [tracesData, setTracesData] = useState<ModelTrace[] | undefined>();
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
    mutationFn: async ({ evaluateParams, traceIds, serializedScorer }) => {
      const responseData = await fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/scorer/invoke`), 'POST', {
        experiment_id: evaluateParams.experimentId,
        serialized_scorer: serializedScorer,
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

      const serializedScorer = getSerializedScorer?.();

      if (!serializedScorer) {
        throw new Error('Cannot build serialized scorer');
      }

      // Start the evaluation job
      startEvaluationJob({ evaluateParams: params, traceIds, serializedScorer });

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
      setTracesData(traces);
    },
    [startEvaluationJob, getTraceIdsForEvaluation, queryClient, getSerializedScorer],
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
      if (job.status === 'FAILED') {
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
    for (const job of Object.values(jobResults ?? {})) {
      if (job.status === 'SUCCEEDED') {
        for (const traceId of Object.keys(job.result)) {
          aggregatedJobResults[traceId] = { assessments: job.result[traceId]?.assessments ?? [] };
        }
      }
    }

    const evaluationResults = tracesData.map((trace: ModelTrace) => {
      const traceInfo = isV3ModelTraceInfo(trace.info) ? trace.info : null;
      if (!traceInfo) {
        return null;
      }
      const traceId = traceInfo.trace_id;

      const results = aggregatedJobResults[traceId]?.assessments ?? [];
      return {
        trace,
        results,
        error: null,
      };
    });

    return compact(evaluationResults);
  }, [jobResults, isLoading, tracesData, error]);

  return [evaluateTracesAsync, { data, isLoading, error, reset }] as const;
};
