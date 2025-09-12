import { useMemo } from 'react';

import type { QueryFunctionContext } from '@databricks/web-shared/query-client';
import { useQueries } from '@databricks/web-shared/query-client';

import { GenAiTraceEvaluationArtifactFile } from '../enum';
import type {
  EvaluationArtifactTableEntryAssessment,
  EvaluationArtifactTableEntryEvaluation,
  EvaluationArtifactTableEntryMetric,
  RawGenaiEvaluationArtifactResponse,
} from '../types';
import { mergeMetricsAndAssessmentsWithEvaluations, parseRawTableArtifact } from '../utils/EvaluationDataParseUtils';
import { getAjaxUrl, makeRequest } from '../utils/FetchUtils';

type UseGetTraceEvaluationArtifactQueryKey = [
  'GET_TRACE_EVALUATION_ARTIFACT',
  { runUuid: string; artifactFile: GenAiTraceEvaluationArtifactFile },
];

const getQueryKey = (
  runUuid: string,
  artifactFile: GenAiTraceEvaluationArtifactFile,
): UseGetTraceEvaluationArtifactQueryKey => ['GET_TRACE_EVALUATION_ARTIFACT', { runUuid, artifactFile }];

const queryFn = async ({
  queryKey: [, { runUuid, artifactFile }],
}: QueryFunctionContext<UseGetTraceEvaluationArtifactQueryKey>): Promise<RawGenaiEvaluationArtifactResponse> => {
  const queryParams = new URLSearchParams({ run_uuid: runUuid, path: artifactFile });
  const url = [getAjaxUrl('ajax-api/2.0/mlflow/get-artifact'), queryParams].join('?');
  return makeRequest(url, 'GET').then((data) => ({
    ...data,
    filename: artifactFile,
  }));
};

const allArtifactFiles = [
  GenAiTraceEvaluationArtifactFile.Assessments,
  GenAiTraceEvaluationArtifactFile.Evaluations,
  GenAiTraceEvaluationArtifactFile.Metrics,
];

/**
 * Fetches evaluation trace artifacts for a given run.
 * @param runUuid - The run UUID for which to fetch evaluation artifacts.
 * @param artifacts - The list of artifact files to fetch. By default, all artifacts are fetched.
 */
export const useGenAiTraceEvaluationArtifacts = (
  { runUuid, artifacts = allArtifactFiles }: { runUuid: string; artifacts?: GenAiTraceEvaluationArtifactFile[] },
  { disabled = false }: { disabled?: boolean } = {},
) => {
  const isAnyArtifactRetrievalEnabled =
    !disabled && allArtifactFiles.some((artifactFile) => artifacts.includes(artifactFile));

  const queriesResult = useQueries({
    queries: allArtifactFiles.map((artifactFile) => ({
      queryFn,
      queryKey: getQueryKey(runUuid, artifactFile),
      enabled: !disabled && artifacts.includes(artifactFile),
      refetchOnWindowFocus: false,
    })),
  });

  const isLoading = queriesResult.some((query) => query.isLoading);
  const isFetching = queriesResult.some((query) => query.isFetching);
  const error = queriesResult.find((query) => query.error);

  const [assessments, evaluations, metrics] = queriesResult.map((query) => query.data);

  const parsedAssessments = useMemo(
    () => parseRawTableArtifact<EvaluationArtifactTableEntryAssessment[]>(assessments),
    [assessments],
  );
  const parsedEvaluations = useMemo(
    () => parseRawTableArtifact<EvaluationArtifactTableEntryEvaluation[]>(evaluations),
    [evaluations],
  );
  const parsedMetrics = useMemo(() => parseRawTableArtifact<EvaluationArtifactTableEntryMetric[]>(metrics), [metrics]);

  const mergedData = useMemo(() => {
    if (!parsedEvaluations) {
      return undefined;
    }
    return mergeMetricsAndAssessmentsWithEvaluations(parsedEvaluations, parsedMetrics, parsedAssessments);
  }, [parsedAssessments, parsedEvaluations, parsedMetrics]);

  return {
    requestError: error,
    isLoading: isLoading && isAnyArtifactRetrievalEnabled,
    isFetching: isFetching && isAnyArtifactRetrievalEnabled,
    data: mergedData,
  };
};
