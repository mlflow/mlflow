import { useCallback, useEffect, useRef } from 'react';
import { useIntl } from 'react-intl';

import { NetworkRequestError } from '@databricks/web-shared/errors';
import { getUser } from '@databricks/web-shared/global-settings/getUser';
import {
  doesTraceSupportV4API,
  getTraceV4QueryKey,
  SEARCH_MLFLOW_TRACES_QUERY_KEY,
} from '@databricks/web-shared/genai-traces-table';
import {
  displayErrorNotification,
  isV3ModelTraceInfo,
  shouldUseTracesV4API,
  TracesServiceV4,
  type Assessment,
  type FeedbackAssessment,
  type ModelTrace,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
import {
  createAssessment as createAssessmentV3,
  fetchTraceInfoV3,
  type FetchTraceInfoV3Response,
} from '@databricks/web-shared/model-trace-explorer/api';
import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';
import { useCurrentUserQuery } from '@mlflow/mlflow/src/account/hooks';

import {
  findPairwiseScoreAssessment,
  PAIRWISE_PREFERENCE_ASSESSMENT_NAME,
  PAIRWISE_SCORE_OPPONENT_TRACE_ID_METADATA_KEY,
  type PairwiseScoreValue,
} from '../utils/pairwiseReview';

interface PairwiseReviewAssessmentInput {
  traceInfo: ModelTraceInfoV3;
  opponentTraceId: string;
  score: PairwiseScoreValue;
}

export interface SavePairwiseReviewParams {
  baseline: PairwiseReviewAssessmentInput;
  candidate: PairwiseReviewAssessmentInput;
  rationale: string;
}

export interface SavedPairwiseReview {
  baselineAssessment: Assessment;
  candidateAssessment: Assessment;
}

const PAIRWISE_ASSESSMENT_MAX_ATTEMPTS = 3;
const PAIRWISE_ASSESSMENT_RETRY_DELAY_MS = 200;
const CURRENT_USER_RETRY_DELAY_MS = 2000;

const isTransientAssessmentError = (error: unknown): boolean => {
  if (error instanceof TypeError) {
    return true;
  }
  if (!(error instanceof NetworkRequestError)) {
    return false;
  }
  return error.status === undefined || error.status === 429 || (error.status >= 500 && error.status < 600);
};

const waitForAssessmentRetry = (attempt: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, PAIRWISE_ASSESSMENT_RETRY_DELAY_MS * 2 ** (attempt - 1)));

export const retryPairwiseAssessmentWrite = async <T>(write: () => Promise<T>): Promise<T> => {
  for (let attempt = 1; attempt < PAIRWISE_ASSESSMENT_MAX_ATTEMPTS; attempt += 1) {
    try {
      return await write();
    } catch (error) {
      if (!isTransientAssessmentError(error)) {
        throw error;
      }
      await waitForAssessmentRetry(attempt);
    }
  }
  return write();
};

const appendAssessment = (assessments: Assessment[] | undefined, assessment: Assessment): Assessment[] => [
  ...(assessments ?? []).filter((cachedAssessment) => cachedAssessment.assessment_id !== assessment.assessment_id),
  assessment,
];

const addAssessmentToCachedTrace = (
  trace: ModelTrace | null | undefined,
  assessment: Assessment,
): ModelTrace | null | undefined => {
  if (!trace || !isV3ModelTraceInfo(trace.info) || trace.info.trace_id !== assessment.trace_id) {
    return trace;
  }
  return {
    ...trace,
    info: {
      ...trace.info,
      assessments: appendAssessment(trace.info.assessments, assessment),
    },
  };
};

const isCachedTraceInfo = (cached: unknown): cached is ModelTraceInfoV3 =>
  typeof cached === 'object' && cached !== null && 'trace_id' in cached && 'trace_location' in cached;

const addAssessmentToSearchCache = (cached: unknown, assessment: Assessment): unknown => {
  if (Array.isArray(cached)) {
    return cached.map((trace) =>
      isCachedTraceInfo(trace) && trace.trace_id === assessment.trace_id
        ? { ...trace, assessments: appendAssessment(trace.assessments, assessment) }
        : trace,
    );
  }
  if (cached instanceof Map) {
    const trace = cached.get(assessment.trace_id);
    if (!trace || typeof trace !== 'object') {
      return cached;
    }
    const updated = new Map(cached);
    const cachedTrace = trace as Pick<ModelTraceInfoV3, 'assessments'>;
    updated.set(assessment.trace_id, {
      ...trace,
      assessments: appendAssessment(cachedTrace.assessments, assessment),
    });
    return updated;
  }
  return cached;
};

// The trace as stored now. Each write overrides the preference that is current on the server, not one that a
// stale cache, an overlapping save or a write whose response was lost has already replaced.
const fetchLatestTraceInfo = async (traceInfo: ModelTraceInfoV3): Promise<ModelTraceInfoV3> => {
  if (shouldUseTracesV4API() && doesTraceSupportV4API(traceInfo)) {
    return TracesServiceV4.getTraceInfoV4({ traceId: traceInfo.trace_id, traceLocation: traceInfo.trace_location });
  }
  const response: FetchTraceInfoV3Response = await fetchTraceInfoV3({ traceId: traceInfo.trace_id });
  const latestTraceInfo = response.trace?.trace_info;
  return latestTraceInfo && isV3ModelTraceInfo(latestTraceInfo) ? latestTraceInfo : traceInfo;
};

export const usePairwiseReviewAssessmentMutation = () => {
  const intl = useIntl();
  const queryClient = useQueryClient();
  const {
    data: currentUser,
    error: currentUserError,
    errorUpdatedAt: currentUserErrorUpdatedAt,
    refetch: refetchCurrentUser,
  } = useCurrentUserQuery();
  // A server with auth enabled knows the signed-in user. Without auth it has no `/users/current` (404), and the
  // review is attributed to the same placeholder user as the trace explorer's other assessment writes. Any other
  // failure keeps voting locked and retries, so a signed-in reviewer is never saved as the shared placeholder.
  const isAuthUnavailable = currentUserError instanceof NetworkRequestError && currentUserError.status === 404;
  const isReviewerResolved = Boolean(currentUser) || isAuthUnavailable;
  const sourceId = currentUser?.user?.username || (isReviewerResolved ? getUser() : '') || '';
  useEffect(() => {
    if (!currentUserError || isAuthUnavailable) {
      return undefined;
    }
    const timer = setTimeout(() => void refetchCurrentUser(), CURRENT_USER_RETRY_DELAY_MS);
    return () => clearTimeout(timer);
  }, [currentUserError, currentUserErrorUpdatedAt, isAuthUnavailable, refetchCurrentUser]);

  const cacheAssessment = (assessment: Assessment) => {
    queryClient.setQueriesData<ModelTrace | null | undefined>(
      { queryKey: getTraceV4QueryKey(assessment.trace_id).slice(0, 2) },
      (trace) => addAssessmentToCachedTrace(trace, assessment),
    );
    queryClient.setQueriesData<ModelTrace | null | undefined>({ queryKey: ['getTrace'] }, (trace) =>
      addAssessmentToCachedTrace(trace, assessment),
    );
    queryClient.setQueriesData<unknown>({ queryKey: [SEARCH_MLFLOW_TRACES_QUERY_KEY] }, (cached: unknown) =>
      addAssessmentToSearchCache(cached, assessment),
    );
  };

  const mutation = useMutation<SavedPairwiseReview, Error, SavePairwiseReviewParams>({
    mutationFn: async ({ baseline, candidate, rationale }) => {
      const createAssessment = async ({ traceInfo, opponentTraceId, score }: PairwiseReviewAssessmentInput) => {
        const latestTraceInfo = await fetchLatestTraceInfo(traceInfo);
        const previousAssessment = findPairwiseScoreAssessment({ trace: latestTraceInfo, opponentTraceId, sourceId });
        const assessment: Omit<FeedbackAssessment, 'assessment_id' | 'create_time' | 'last_update_time'> = {
          assessment_name: PAIRWISE_PREFERENCE_ASSESSMENT_NAME,
          trace_id: traceInfo.trace_id,
          source: { source_type: 'HUMAN', source_id: sourceId },
          feedback: { value: score },
          metadata: {
            [PAIRWISE_SCORE_OPPONENT_TRACE_ID_METADATA_KEY]: opponentTraceId,
          },
          ...(previousAssessment ? { overrides: previousAssessment.assessment_id } : {}),
          ...(rationale.trim() ? { rationale: rationale.trim() } : {}),
        };
        // Same API choice as the trace explorer's own assessment writes: V4 for V4-backed traces, else V3.
        if (shouldUseTracesV4API() && doesTraceSupportV4API(traceInfo)) {
          return TracesServiceV4.createAssessmentV4({
            payload: { assessment },
            traceLocation: traceInfo.trace_location,
          });
        }
        return createAssessmentV3({ payload: { assessment } }).then((response) => response.assessment);
      };

      const [baselineResult, candidateResult] = await Promise.allSettled([
        retryPairwiseAssessmentWrite(() => createAssessment(baseline)),
        retryPairwiseAssessmentWrite(() => createAssessment(candidate)),
      ]);
      // Show whichever side was saved even if the other failed, so the review reflects what is stored.
      for (const result of [baselineResult, candidateResult]) {
        if (result.status === 'fulfilled') {
          cacheAssessment(result.value);
        }
      }
      if (baselineResult.status === 'rejected') {
        throw baselineResult.reason;
      }
      if (candidateResult.status === 'rejected') {
        throw candidateResult.reason;
      }
      return { baselineAssessment: baselineResult.value, candidateAssessment: candidateResult.value };
    },
    onSettled: () => {
      // The traces table caches each page as a `{ trace_infos }` response, which `cacheAssessment` doesn't
      // touch, so refetch it like the trace explorer's other assessment writes do.
      void queryClient.invalidateQueries({ queryKey: [SEARCH_MLFLOW_TRACES_QUERY_KEY] });
    },
    onError: (error) => {
      displayErrorNotification(
        intl.formatMessage(
          {
            defaultMessage: 'Failed to save pairwise review. Error: {error}',
            description: 'Error shown when saving a pairwise trace review fails',
          },
          { error: error.message },
        ),
      );
    },
  });

  // Saves run one at a time, so a comment saved on blur and a winner chosen by the same click don't both
  // override the same preference.
  const { mutateAsync } = mutation;
  const saveQueueRef = useRef<Promise<unknown>>(Promise.resolve());
  const savePairwiseReview = useCallback(
    (params: SavePairwiseReviewParams) => {
      const save = saveQueueRef.current.then(() => mutateAsync(params));
      saveQueueRef.current = save.catch(() => undefined);
      return save;
    },
    [mutateAsync],
  );

  return {
    savePairwiseReview,
    isSavingPairwiseReview: mutation.isLoading,
    reviewerSourceId: sourceId,
    isReviewerResolved,
  };
};
