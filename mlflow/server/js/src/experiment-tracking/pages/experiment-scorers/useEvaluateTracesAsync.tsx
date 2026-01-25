import { fetchAPI, getAjaxUrl } from '../../../common/utils/FetchUtils';
import { useMutation, useQueryClient } from '../../../common/utils/reactQueryHooks';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FeedbackAssessment, ModelTrace } from '../../../shared/web-shared/model-trace-explorer';
import { EvaluateTracesParams } from './types';
import { useGetTraceIdsForEvaluation } from './useGetTracesForEvaluation';
import {
  getMlflowTraceV3ForEvaluation,
  JudgeEvaluationResult,
  SessionJudgeEvaluationResult,
} from './useEvaluateTraces.common';
import {
  TrackingJobQueryResult,
  TrackingJobStatus,
  useGetTrackingServerJobStatus,
} from '../../../common/hooks/useGetTrackingServerJobStatus';
import { compact, uniq, zipObject } from 'lodash';
import { SessionForEvaluation, useGetSessionsForEvaluation } from './useGetSessionsForEvaluation';
import { ScorerEvaluationScope } from './constants';

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

type StartEvaluationJobParams = { evaluateParams: EvaluateTracesParams; traceIds: string[]; requestKey: string };
type StartEvaluationJobResponse = { jobs: { job_id: string }[] };

/**
 * Represents a single evaluation request with its state
 */
export interface EvaluationRequest {
  requestKey: string;
  jobIds: string[];
  status: TrackingJobStatus;
  tracesData?: Record<string, ModelTrace>;
  sessionsData?: SessionForEvaluation[];
  results?: JudgeEvaluationResult[];
  error?: Error | null;
  startedAt: number;
}

/**
 * Event payload for scorer state updates.
 * Fired when an evaluation's status changes (PENDING -> RUNNING -> SUCCEEDED/FAILED).
 */
export interface ScorerUpdateEvent {
  /** Unique identifier for the evaluation request */
  requestKey: string;
  /** Current status of the evaluation */
  status: TrackingJobStatus;
  /** Evaluation results (only present when status is SUCCEEDED) */
  results?: JudgeEvaluationResult[];
  /** Error details (only present when status is FAILED) */
  error?: Error | null;
}

/**
 * Generate a unique request key
 */
const generateRequestKey = () => `eval-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

export const useEvaluateTracesAsync = ({
  onScorerUpdate,
}: {
  /** Callback fired when an evaluation's status changes */
  onScorerUpdate?: (event: ScorerUpdateEvent) => void;
}) => {
  const queryClient = useQueryClient();
  const getTraceIdsForEvaluation = useGetTraceIdsForEvaluation();
  const getSessionsForEvaluation = useGetSessionsForEvaluation();

  // Multi-request state management
  const [evaluations, setEvaluations] = useState<Record<string, EvaluationRequest>>({});
  const [latestRequestKey, setLatestRequestKey] = useState<string | null>(null);

  // Track which evaluations have already triggered onScorerUpdate for terminal states
  const finishedEvaluationsRef = useRef<Set<string>>(new Set());

  // State update helpers
  const startEvaluation = useCallback(
    (requestKey: string) => {
      setLatestRequestKey(requestKey);
      setEvaluations((prev) => ({
        ...prev,
        [requestKey]: {
          requestKey,
          jobIds: [],
          status: TrackingJobStatus.PENDING,
          startedAt: Date.now(),
          error: null,
        },
      }));
      onScorerUpdate?.({ requestKey, status: TrackingJobStatus.PENDING });
    },
    [onScorerUpdate],
  );

  const setJobIds = useCallback((requestKey: string, jobIds: string[]) => {
    setEvaluations((prev) => ({
      ...prev,
      [requestKey]: {
        ...prev[requestKey],
        jobIds,
      },
    }));
  }, []);

  const updateStatus = useCallback(
    (requestKey: string, status: TrackingJobStatus) => {
      setEvaluations((prev) => ({
        ...prev,
        [requestKey]: {
          ...prev[requestKey],
          status,
        },
      }));
      onScorerUpdate?.({ requestKey, status });
    },
    [onScorerUpdate],
  );

  const setResults = useCallback(
    (requestKey: string, results: JudgeEvaluationResult[]) => {
      setEvaluations((prev) => ({
        ...prev,
        [requestKey]: {
          ...prev[requestKey],
          results,
          status: TrackingJobStatus.SUCCEEDED,
        },
      }));
      // Only fire callback once per evaluation for terminal states
      if (!finishedEvaluationsRef.current.has(requestKey)) {
        finishedEvaluationsRef.current.add(requestKey);
        onScorerUpdate?.({ requestKey, status: TrackingJobStatus.SUCCEEDED, results });
      }
    },
    [onScorerUpdate],
  );

  const setError = useCallback(
    (requestKey: string, error: Error) => {
      setEvaluations((prev) => ({
        ...prev,
        [requestKey]: {
          ...prev[requestKey],
          error,
          status: TrackingJobStatus.FAILED,
        },
      }));
      // Only fire callback once per evaluation for terminal states
      if (!finishedEvaluationsRef.current.has(requestKey)) {
        finishedEvaluationsRef.current.add(requestKey);
        onScorerUpdate?.({ requestKey, status: TrackingJobStatus.FAILED, error });
      }
    },
    [onScorerUpdate],
  );

  const setTracesDataForRequest = useCallback((requestKey: string, tracesData: Record<string, ModelTrace>) => {
    setEvaluations((prev) => ({
      ...prev,
      [requestKey]: {
        ...prev[requestKey],
        tracesData,
      },
    }));
  }, []);

  const setSessionsDataForRequest = useCallback((requestKey: string, sessionsData: SessionForEvaluation[]) => {
    setEvaluations((prev) => ({
      ...prev,
      [requestKey]: {
        ...prev[requestKey],
        sessionsData,
      },
    }));
  }, []);

  // Compute active job IDs from evaluations (only PENDING/RUNNING)
  const activeJobIds = useMemo(() => {
    return Object.values(evaluations)
      .filter((e) => e.status === TrackingJobStatus.PENDING || e.status === TrackingJobStatus.RUNNING)
      .flatMap((e) => e.jobIds);
  }, [evaluations]);

  const { jobResults, areJobsRunning, jobStatuses } = useGetTrackingServerJobStatus<EvaluateTracesAsyncJobResult>(
    activeJobIds,
    {
      enabled: activeJobIds.length > 0,
      refetchInterval: (data) => {
        if (data?.some((job) => isJobRunning(job))) {
          return JOB_POLLING_INTERVAL;
        }
        return false;
      },
      refetchOnWindowFocus: false,
      refetchOnMount: false,
    },
  );

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
        log_assessments: evaluateParams.saveAssessment,
      });
      return responseData;
    },
    onSuccess: (data, variables) => {
      setJobIds(
        variables.requestKey,
        data.jobs.map((job) => job.job_id),
      );
    },
    onError: (error, variables) => {
      setError(variables.requestKey, error);
    },
  });

  // Create a map from jobId -> requestKey for quick lookup
  const jobToRequestMap = useMemo(() => {
    const map = new Map<string, string>();
    Object.values(evaluations).forEach((evaluation) => {
      evaluation.jobIds.forEach((jobId) => map.set(jobId, evaluation.requestKey));
    });
    return map;
  }, [evaluations]);

  // Use a ref to track evaluations state for use in effect without causing re-runs
  const evaluationsRef = useRef(evaluations);
  evaluationsRef.current = evaluations;

  // Helper function to compute results from job result and evaluation data
  const computeResults = useCallback(
    (
      jobResult: EvaluateTracesAsyncJobResult,
      evaluation: EvaluationRequest,
    ): JudgeEvaluationResult[] | null => {
      const { tracesData, sessionsData } = evaluation;
      if (!tracesData) {
        return null;
      }

      const aggregatedJobResults: EvaluateTracesAsyncJobResult = {};
      for (const traceId of Object.keys(jobResult)) {
        aggregatedJobResults[traceId] = {
          assessments: jobResult[traceId]?.assessments ?? [],
          failures: jobResult[traceId]?.failures ?? [],
        };
      }

      if (sessionsData) {
        return sessionsData.map<SessionJudgeEvaluationResult>((session) => {
          const sessionResults: { assessments: FeedbackAssessment[]; errors: string[] } = {
            assessments: [],
            errors: [],
          };
          for (const traceInfo of session.traceInfos) {
            const traceId = traceInfo.trace_id;
            const traceResults = aggregatedJobResults[traceId];
            sessionResults.assessments.push(...(traceResults?.assessments ?? []));
            sessionResults.errors.push(...(traceResults?.failures?.map((failure) => failure.error_message) ?? []));
          }

          return {
            sessionId: session.sessionId ?? '',
            results: sessionResults.assessments,
            traces: compact(session.traceInfos.map((traceInfo) => tracesData[traceInfo.trace_id] ?? null)),
            error: sessionResults.errors.join(', ') || null,
          };
        });
      }

      const traceEvaluationResults = Object.entries(tracesData).map(([traceId, trace]) => {
        const results = aggregatedJobResults[traceId]?.assessments ?? [];
        const failureString = aggregatedJobResults[traceId]?.failures?.map((failure) => failure.error_message).join(', ');
        return {
          trace,
          results,
          error: failureString || null,
        };
      });

      return compact(traceEvaluationResults);
    },
    [],
  );

  // Sync job results to state when polling returns new data
  useEffect(() => {
    if (!jobStatuses || !jobResults) return;

    jobStatuses.forEach(({ jobId, status }) => {
      const requestKey = jobToRequestMap.get(jobId);
      if (!requestKey) return;

      // Use ref to get current evaluation state without adding evaluations to deps
      const evaluation = evaluationsRef.current[requestKey];
      if (!evaluation) return;

      // Skip if this evaluation is already completed or status hasn't changed
      if (
        evaluation.status === TrackingJobStatus.SUCCEEDED ||
        evaluation.status === TrackingJobStatus.FAILED ||
        evaluation.status === status
      ) {
        return;
      }

      if (status === TrackingJobStatus.RUNNING) {
        updateStatus(requestKey, TrackingJobStatus.RUNNING);
      } else if (status === TrackingJobStatus.SUCCEEDED) {
        const result = jobResults[jobId];
        if (result?.status === TrackingJobStatus.SUCCEEDED && evaluation.tracesData) {
          const computedResults = computeResults(result.result, evaluation);
          if (computedResults) {
            // setResults will fire onScorerUpdate callback
            setResults(requestKey, computedResults);
          }
        }
      } else if (status === TrackingJobStatus.FAILED) {
        const result = jobResults[jobId];
        if (result?.status === TrackingJobStatus.FAILED) {
          // setError will fire onScorerUpdate callback
          setError(requestKey, new Error(result.result));
        }
      }
    });
  }, [jobStatuses, jobResults, jobToRequestMap, updateStatus, setResults, setError, computeResults]);

  const evaluateTracesAsync = useCallback(
    async (params: EvaluateTracesParams, requestKey?: string) => {
      const key = requestKey ?? generateRequestKey();
      startEvaluation(key);

      let traceIds: string[];
      if (params.evaluationScope === ScorerEvaluationScope.SESSIONS) {
        const sessions = await getSessionsForEvaluation(params);
        setSessionsDataForRequest(key, sessions);
        traceIds = uniq(sessions.flatMap((session) => session.traceInfos.map((traceInfo) => traceInfo.trace_id)));
      } else {
        traceIds = await getTraceIdsForEvaluation(params);
      }

      const serializedScorer = params.serializedScorer;

      if (!serializedScorer) {
        setError(key, new Error('The serialized scorer is malformed'));
        return key;
      }

      // Start the evaluation job
      startEvaluationJob({ evaluateParams: params, traceIds, requestKey: key });

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
      setTracesDataForRequest(key, zipObject(traceIds, traces));

      return key;
    },
    [
      startEvaluation,
      startEvaluationJob,
      getTraceIdsForEvaluation,
      queryClient,
      getSessionsForEvaluation,
      setSessionsDataForRequest,
      setTracesDataForRequest,
      setError,
    ],
  );

  const reset = useCallback(() => {
    // Cancel any running jobs on the backend (fire-and-forget)
    // Use ref to access current state without adding to dependencies
    Object.values(evaluationsRef.current)
      .filter((e) => e.status === TrackingJobStatus.PENDING || e.status === TrackingJobStatus.RUNNING)
      .flatMap((e) => e.jobIds)
      .forEach((jobId) => {
        fetchAPI(getAjaxUrl(`ajax-api/3.0/jobs/cancel/${jobId}`), 'PATCH').catch(() => {
          // Ignore errors - job may have already completed
        });
      });

    setEvaluations({});
    setLatestRequestKey(null);
    finishedEvaluationsRef.current = new Set();
  }, []);

  // TODO: Implement cleanup on unmount - cancel running jobs when component unmounts
  // useEffect(() => {
  //   return () => {
  //     Object.values(evaluations)
  //       .filter(
  //         (e) =>
  //           e.status === TrackingJobStatus.PENDING ||
  //           e.status === TrackingJobStatus.RUNNING
  //       )
  //       .flatMap((e) => e.jobIds)
  //       .forEach((jobId) => {
  //         fetchAPI(getAjaxUrl(`ajax-api/3.0/jobs/cancel/${jobId}`), 'PATCH').catch(
  //           () => {}
  //         );
  //       });
  //   };
  // }, []);

  // Derive latest evaluation for backward compatibility
  const latestEvaluation = latestRequestKey ? evaluations[latestRequestKey] : undefined;

  // Helper to get specific evaluation
  const getEvaluation = useCallback((key: string) => evaluations[key], [evaluations]);

  // Compute backward-compatible data/error from latest evaluation
  const data = latestEvaluation?.results ?? null;
  const isLoading =
    isJobStarting ||
    (latestEvaluation?.status === TrackingJobStatus.PENDING ||
      latestEvaluation?.status === TrackingJobStatus.RUNNING);
  const error = startEvaluationJobError ?? latestEvaluation?.error ?? null;

  // Backward-compatible tuple return (existing API)
  return [
    evaluateTracesAsync,
    {
      data,
      isLoading,
      error,
      reset,
      // Extended API for multi-request consumers
      getEvaluation,
      allEvaluations: evaluations,
    },
  ] as const;
};
