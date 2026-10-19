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
import { createAssessment as createAssessmentV3 } from '@databricks/web-shared/model-trace-explorer/api';
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

export const usePairwiseReviewAssessmentMutation = () => {
  const intl = useIntl();
  const queryClient = useQueryClient();
  const { data: currentUser, isLoading: isLoadingCurrentUser } = useCurrentUserQuery();
  // A server with auth enabled knows the signed-in user. Without auth, `/users/current` fails and the
  // review is attributed to the same placeholder user as the trace explorer's other assessment writes.
  const sourceId = currentUser?.user?.username || getUser() || '';

  const mutation = useMutation<SavedPairwiseReview, Error, SavePairwiseReviewParams>({
    mutationFn: async ({ baseline, candidate, rationale }) => {
      const createAssessment = ({ traceInfo, opponentTraceId, score }: PairwiseReviewAssessmentInput) => {
        const previousAssessment = findPairwiseScoreAssessment({ trace: traceInfo, opponentTraceId, sourceId });
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

      const [baselineAssessment, candidateAssessment] = await Promise.all([
        retryPairwiseAssessmentWrite(() => createAssessment(baseline)),
        retryPairwiseAssessmentWrite(() => createAssessment(candidate)),
      ]);
      return { baselineAssessment, candidateAssessment };
    },
    onSuccess: ({ baselineAssessment, candidateAssessment }) => {
      for (const assessment of [baselineAssessment, candidateAssessment]) {
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
      }
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

  return {
    savePairwiseReview: mutation.mutateAsync,
    isSavingPairwiseReview: mutation.isLoading,
    reviewerSourceId: sourceId,
    // Saving before `/users/current` settles would attribute the review to the placeholder user.
    isReviewerResolved: !isLoadingCurrentUser,
  };
};
