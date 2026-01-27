import { fetchAPI, getAjaxUrl } from '../../../common/utils/FetchUtils';
import { useMutation, useQueryClient } from '../../../common/utils/reactQueryHooks';
import { useCallback, useEffect, useRef, useState } from 'react';
import { FeedbackAssessment, ModelTrace } from '../../../shared/web-shared/model-trace-explorer';
import { EvaluateTracesParams } from './types';
import { useGetTraceIdsForEvaluation } from './useGetTracesForEvaluation';
import {
  getMlflowTraceV3ForEvaluation,
  JudgeEvaluationResult,
  SessionJudgeEvaluationResult,
} from './useEvaluateTraces.common';
import { TrackingJobStatus } from '../../../common/hooks/useGetTrackingServerJobStatus';
import { compact, uniq, zipObject } from 'lodash';
import { SessionForEvaluation, useGetSessionsForEvaluation } from './useGetSessionsForEvaluation';
import { ScorerEvaluationScope } from './constants';
import { parseJSONSafe } from '../../../common/utils/TagUtils';

const JOB_POLLING_INTERVAL = 1500;

type JobStatusMap = Record<string, { status: TrackingJobStatus; result?: unknown }>;

type EvaluateTracesAsyncJobResult = Record<
  string,
  { assessments: FeedbackAssessment[]; failures?: { error_code: string; error_message: string }[] }
>;

/** Check if any job is still running or pending */
const isJobsLoading = (jobStatuses: JobStatusMap, jobIds: string[]): boolean => {
  if (!jobIds.length) return true; // No jobs yet = still loading
  return jobIds.some((id) => {
    const status = jobStatuses[id]?.status;
    return !status || status === TrackingJobStatus.RUNNING || status === TrackingJobStatus.PENDING;
  });
};

/**
 * Represents a single evaluation request with its state
 */
export interface ScorerEvaluation {
  requestKey: string;
  label: string;
  jobIds: string[];
  jobStatuses: JobStatusMap;
  /** Whether any job is still running or pending */
  isLoading: boolean;
  tracesData?: Record<string, ModelTrace>;
  sessionsData?: SessionForEvaluation[];
  results?: JudgeEvaluationResult[];
  error?: Error | null;
}

/** Event payload for scorer state updates */
export interface ScorerFinishedEvent {
  requestKey: string;
  status: TrackingJobStatus;
  results?: JudgeEvaluationResult[];
  error?: Error | null;
}

// Generates a unique request key for an evaluation
const generateRequestKey = () => `eval-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

/**
 * Derive aggregated status of entire evaluation request.
 */
const deriveRequestStatus = (evaluationRequest: ScorerEvaluation): TrackingJobStatus => {
  const { jobIds, jobStatuses } = evaluationRequest;
  if (!jobIds.length) return TrackingJobStatus.PENDING;
  const statuses = jobIds.map((id) => jobStatuses[id]?.status);
  if (
    statuses.some((status) => !status || status === TrackingJobStatus.RUNNING || status === TrackingJobStatus.PENDING)
  ) {
    return statuses.some((status) => status === TrackingJobStatus.RUNNING)
      ? TrackingJobStatus.RUNNING
      : TrackingJobStatus.PENDING;
  }
  // All jobs are complete - check if at least one succeeded
  return statuses.includes(TrackingJobStatus.SUCCEEDED) ? TrackingJobStatus.SUCCEEDED : TrackingJobStatus.FAILED;
};

/**
 * Build evaluation results from job data and traces (handles partial success)
 */
const buildResults = (evaluationRequest: ScorerEvaluation): JudgeEvaluationResult[] | null => {
  const { tracesData, sessionsData, jobIds, jobStatuses } = evaluationRequest;
  if (!tracesData) return null;

  // Aggregate from all completed jobs - process successful jobs first, then failed (to prioritize successful results)
  const aggregated: EvaluateTracesAsyncJobResult = {};
  let failedJobError: string | null = null;

  for (const jobId of jobIds) {
    const job = jobStatuses[jobId];
    if (job?.status !== TrackingJobStatus.SUCCEEDED || typeof job.result !== 'object' || !job.result) {
      continue;
    }
    Object.assign(aggregated, job.result as EvaluateTracesAsyncJobResult);
  }

  for (const jobId of jobIds) {
    const job = jobStatuses[jobId];
    if (job?.status !== TrackingJobStatus.FAILED) {
      continue;
    }
    if (!failedJobError && typeof job.result === 'string') {
      failedJobError = job.result;
    }
    if (typeof job.result === 'object' && job.result) {
      for (const [traceId, data] of Object.entries(job.result as EvaluateTracesAsyncJobResult)) {
        if (!aggregated[traceId]) aggregated[traceId] = data;
      }
    }
  }

  // Default error for traces without results when some jobs failed
  const hasFailedJobs = jobIds.some((id) => jobStatuses[id]?.status === TrackingJobStatus.FAILED);
  const defaultError = hasFailedJobs ? failedJobError || 'Evaluation job failed' : null;

  if (sessionsData) {
    return sessionsData.map<SessionJudgeEvaluationResult>((session) => {
      const assessments: FeedbackAssessment[] = [];
      const errors: string[] = [];
      let hasMissingTraces = false;

      for (const { trace_id } of session.traceInfos) {
        if (aggregated[trace_id]) {
          assessments.push(...(aggregated[trace_id].assessments ?? []));
          errors.push(...(aggregated[trace_id].failures?.map((f) => f.error_message) ?? []));
        } else {
          hasMissingTraces = true;
        }
      }

      // Add default error if some traces are missing and we had failed jobs
      if (hasMissingTraces && defaultError) {
        errors.push(defaultError);
      }

      return {
        sessionId: session.sessionId ?? '',
        results: assessments,
        traces: compact(session.traceInfos.map((t) => tracesData[t.trace_id] ?? null)),
        error: errors.join(', ') || null,
      };
    });
  }

  return compact(
    Object.entries(tracesData).map(([traceId, trace]) => {
      const traceData = aggregated[traceId];
      const traceErrors = traceData?.failures?.map((f) => f.error_message).join(', ');
      // If trace has no data and there were failed jobs, use the default error
      const error = traceErrors || (!traceData && defaultError) || null;
      return {
        trace,
        results: traceData?.assessments ?? [],
        error,
      };
    }),
  );
};

/** Extract error message from first failed job */
const extractError = (jobStatuses: JobStatusMap, jobIds: string[]): Error => {
  for (const jobId of jobIds) {
    const job = jobStatuses[jobId];
    if (job?.status === TrackingJobStatus.FAILED) {
      return new Error(typeof job.result === 'string' ? job.result : 'Job failed');
    }
  }
  return new Error('Unknown error');
};

export const useEvaluateTracesAsync = ({
  onScorerFinished,
}: {
  onScorerFinished?: (event: ScorerFinishedEvent) => void;
}) => {
  const queryClient = useQueryClient();
  const getTraceIdsForEvaluation = useGetTraceIdsForEvaluation();
  const getSessionsForEvaluation = useGetSessionsForEvaluation();

  const [evaluations, setEvaluations] = useState<Record<string, ScorerEvaluation>>({});
  const [latestRequestKey, setLatestRequestKey] = useState<string | null>(null);

  // Refs for stable access in polling interval (avoids stale closure issues)
  const evaluationsRef = useRef(evaluations);
  evaluationsRef.current = evaluations;
  const finishedCallbackRef = useRef(onScorerFinished);
  finishedCallbackRef.current = onScorerFinished;

  // Track finalized evaluations (to fire callbacks only once)
  const finalizedRef = useRef<Set<string>>(new Set());

  // Always-on polling interval that checks refs for current state
  useEffect(() => {
    const poll = async () => {
      const currentEvals = evaluationsRef.current;

      // Find jobs that need polling
      const jobsToPoll: string[] = [];
      for (const ev of Object.values(currentEvals)) {
        if (finalizedRef.current.has(ev.requestKey)) {
          continue;
        }
        for (const jobId of ev.jobIds) {
          const status = ev.jobStatuses[jobId]?.status;
          if (status !== TrackingJobStatus.SUCCEEDED && status !== TrackingJobStatus.FAILED) {
            jobsToPoll.push(jobId);
          }
        }
      }

      if (jobsToPoll.length === 0) {
        return;
      }

      const results = await Promise.all(
        jobsToPoll.map(async (jobId) => {
          try {
            const res = await fetchAPI(getAjaxUrl(`ajax-api/3.0/jobs/${jobId}`));
            return { jobId, status: res.status as TrackingJobStatus, result: res.result };
          } catch {
            return { jobId, status: TrackingJobStatus.FAILED, result: 'Failed to fetch job status' };
          }
        }),
      );

      setEvaluations((prev) => {
        const next = { ...prev };
        // Update job statuses for each evaluation request
        for (const { jobId, status, result } of results) {
          for (const evaluationRequest of Object.values(next)) {
            if (evaluationRequest.jobIds.includes(jobId)) {
              const newJobStatuses = { ...evaluationRequest.jobStatuses, [jobId]: { status, result } };
              next[evaluationRequest.requestKey] = {
                ...evaluationRequest,
                jobStatuses: newJobStatuses,
                isLoading: isJobsLoading(newJobStatuses, evaluationRequest.jobIds),
              };
            }
          }
        }
        // Check for completions (deriveRequestStatus returns SUCCEEDED/FAILED only when all jobs complete)
        for (const evaluationRequest of Object.values(next)) {
          if (finalizedRef.current.has(evaluationRequest.requestKey)) {
            continue;
          }
          const status = deriveRequestStatus(evaluationRequest);
          if (status === TrackingJobStatus.SUCCEEDED) {
            const evalResults = buildResults(evaluationRequest);
            if (evalResults) {
              finalizedRef.current.add(evaluationRequest.requestKey);
              next[evaluationRequest.requestKey] = {
                ...next[evaluationRequest.requestKey],
                results: evalResults,
                isLoading: false,
              };
              finishedCallbackRef.current?.({ requestKey: evaluationRequest.requestKey, status, results: evalResults });
            }
          } else if (status === TrackingJobStatus.FAILED) {
            const error = extractError(evaluationRequest.jobStatuses, evaluationRequest.jobIds);
            finalizedRef.current.add(evaluationRequest.requestKey);
            next[evaluationRequest.requestKey] = { ...next[evaluationRequest.requestKey], error, isLoading: false };
          }
        }
        return next;
      });
    };

    const intervalId = setInterval(poll, JOB_POLLING_INTERVAL);
    return () => clearInterval(intervalId);
  }, []);

  const {
    mutate: startEvaluationJob,
    isLoading: isJobStarting,
    error: startEvaluationJobError,
  } = useMutation<
    { jobs: { job_id: string }[] },
    Error,
    { evaluateParams: EvaluateTracesParams; traceIds: string[]; requestKey: string }
  >({
    mutationFn: async ({ evaluateParams, traceIds }) =>
      fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/scorer/invoke'), 'POST', {
        experiment_id: evaluateParams.experimentId,
        serialized_scorer: evaluateParams.serializedScorer,
        trace_ids: traceIds,
        log_assessments: evaluateParams.saveAssessment,
      }),
    onSuccess: (data, { requestKey }) => {
      const jobIds = data.jobs.map((j) => j.job_id);
      setEvaluations((prev) => ({
        ...prev,
        [requestKey]: {
          ...prev[requestKey],
          jobIds,
          isLoading: isJobsLoading(prev[requestKey]?.jobStatuses ?? {}, jobIds),
        },
      }));
    },
    onError: (error, { requestKey }) => {
      finalizedRef.current.add(requestKey);
      setEvaluations((prev) => ({ ...prev, [requestKey]: { ...prev[requestKey], error, isLoading: false } }));
    },
  });

  const evaluateTracesAsync = useCallback(
    async (params: EvaluateTracesParams, requestKey?: string) => {
      const key = requestKey ?? generateRequestKey();
      setLatestRequestKey(key);

      // Use the scorer name to be saved as a user-facing label
      const evaluationRequestLabel = params.serializedScorer
        ? parseJSONSafe(params.serializedScorer)?.name
        : 'Feedback';

      // Initialize evaluation
      setEvaluations((prev) => ({
        ...prev,
        [key]: {
          requestKey: key,
          jobIds: [],
          jobStatuses: {},
          isLoading: true,
          label: evaluationRequestLabel,
        },
      }));

      let traceIds: string[];
      let sessionsData: SessionForEvaluation[] | undefined;

      if (params.evaluationScope === ScorerEvaluationScope.SESSIONS) {
        const sessions = await getSessionsForEvaluation(params);
        sessionsData = sessions;
        traceIds = uniq(sessions.flatMap((s) => s.traceInfos.map((t) => t.trace_id)));
      } else {
        traceIds = await getTraceIdsForEvaluation(params);
      }

      if (!params.serializedScorer) {
        const error = new Error('The serialized scorer is malformed');
        finalizedRef.current.add(key);
        setEvaluations((prev) => ({ ...prev, [key]: { ...prev[key], error, isLoading: false } }));
        return key;
      }

      startEvaluationJob({ evaluateParams: params, traceIds, requestKey: key });

      // Fetch traces in parallel
      const traces = await Promise.all(
        traceIds.map((traceId) =>
          queryClient.fetchQuery({
            queryKey: ['GetMlflowTraceV3', traceId],
            queryFn: () => getMlflowTraceV3ForEvaluation(traceId),
            staleTime: Infinity,
            cacheTime: Infinity,
          }),
        ),
      );

      setEvaluations((prev) => ({
        ...prev,
        [key]: { ...prev[key], tracesData: zipObject(traceIds, traces), sessionsData },
      }));

      return key;
    },
    [startEvaluationJob, getTraceIdsForEvaluation, queryClient, getSessionsForEvaluation],
  );

  const reset = useCallback(() => {
    // Cancel running jobs (fire-and-forget)
    for (const ev of Object.values(evaluationsRef.current)) {
      if (!finalizedRef.current.has(ev.requestKey)) {
        for (const jobId of ev.jobIds) {
          fetchAPI(getAjaxUrl(`ajax-api/3.0/jobs/cancel/${jobId}`), 'PATCH').catch(() => {});
        }
      }
    }
    setEvaluations({});
    setLatestRequestKey(null);
    finalizedRef.current = new Set();
  }, []);

  // Get the latest evaluation request from the evaluations map
  const latestEvaluation = latestRequestKey ? evaluations[latestRequestKey] : undefined;

  return [
    evaluateTracesAsync,
    {
      data: latestEvaluation?.results ?? null,
      isLoading: isJobStarting || (latestEvaluation && !latestEvaluation.results && !latestEvaluation.error) || false,
      error: startEvaluationJobError ?? latestEvaluation?.error ?? null,
      reset,
      evaluations,
    },
  ] as const;
};
