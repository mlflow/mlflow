import { isEmpty, keyBy } from 'lodash';
import { useEffect, useMemo } from 'react';
import { useRunDetailsPageDataLegacy } from '../useRunDetailsPageDataLegacy';
import type {
  UseGetRunQueryDataApiError,
  UseGetRunQueryResponseDataMetrics,
  UseGetRunQueryResponseDatasetInputs,
  UseGetRunQueryResponseRunInfo,
} from './useGetRunQuery';
import {
  type UseGetRunQueryResponseExperiment,
  useGetRunQuery,
  type UseGetRunQueryResponseInputs,
  type UseGetRunQueryResponseOutputs,
} from './useGetRunQuery';
import type { RunDatasetWithTags } from '../../../types';
import {
  type ExperimentEntity,
  type MetricEntitiesByName,
  type MetricEntity,
  type RunInfoEntity,
} from '../../../types';
import type { KeyValueEntity } from '../../../../common/types';
import {
  shouldEnableGraphQLModelVersionsForRunDetails,
  shouldEnableGraphQLRunDetailsPage,
} from '../../../../common/utils/FeatureUtils';
import type { ThunkDispatch } from '../../../../redux-types';
import { useDispatch } from 'react-redux';
import { searchModelVersionsApi } from '../../../../model-registry/actions';
import type { ApolloError } from '@mlflow/mlflow/src/common/utils/graphQLHooks';
import type { ErrorWrapper } from '../../../../common/utils/ErrorWrapper';
import { pickBy } from 'lodash';
import {
  type RunPageModelVersionSummary,
  useUnifiedRegisteredModelVersionsSummariesForRun,
} from './useUnifiedRegisteredModelVersionsSummariesForRun';

// Internal util: transforms an array of objects into a keyed object by the `key` field
const transformToKeyedObject = <Output, Input = any>(inputArray: Input[]) =>
  // TODO: fix this type error
  // @ts-expect-error: Conversion of type 'Dictionary<Input>' to type 'Record<string, Output>' may be a mistake because neither type sufficiently overlaps with the other. If this was intentional, convert the expression to 'unknown' first.
  keyBy(inputArray, 'key') as Record<string, Output>;

// Internal util: transforms an array of metric values into an array of MetricEntity objects
// GraphQL uses strings for steps and timestamp so we cast them to numbers
const transformMetricValues = (inputArray: UseGetRunQueryResponseDataMetrics): MetricEntity[] =>
  inputArray
    .filter(({ key, value, step, timestamp }) => key !== null && value !== null && step !== null && timestamp !== null)
    .map(({ key, value, step, timestamp }: any) => ({
      key,
      value,
      step: Number(step),
      timestamp: Number(timestamp),
    }));

// Internal util: transforms an array of dataset inputs into an array of RunDatasetWithTags objects
export const transformDatasets = (inputArray?: UseGetRunQueryResponseDatasetInputs): RunDatasetWithTags[] | undefined =>
  inputArray?.map((datasetInput) => ({
    dataset: {
      digest: datasetInput.dataset?.digest ?? '',
      name: datasetInput.dataset?.name ?? '',
      profile: datasetInput.dataset?.profile ?? '',
      schema: datasetInput.dataset?.schema ?? '',
      source: datasetInput.dataset?.source ?? '',
      sourceType: datasetInput.dataset?.sourceType ?? '',
    },
    tags:
      datasetInput.tags
        ?.map((tag) => ({
          key: tag.key ?? '',
          value: tag.value ?? '',
        }))
        .filter((tag) => !isEmpty(tag.key)) ?? [],
  }));

interface UseRunDetailsPageDataResult {
  experiment?: ExperimentEntity | UseGetRunQueryResponseExperiment;
  error: Error | ErrorWrapper | undefined | ApolloError;

  latestMetrics: MetricEntitiesByName;
  loading: boolean;
  params: Record<string, KeyValueEntity>;
  refetchRun: any;
  runInfo?: RunInfoEntity | UseGetRunQueryResponseRunInfo;
  tags: Record<string, KeyValueEntity>;
  datasets?: RunDatasetWithTags[];
  runInputs?: UseGetRunQueryResponseInputs;
  runOutputs?: UseGetRunQueryResponseOutputs;

  // Only present in legacy implementation
  runFetchError?: Error | ErrorWrapper | undefined;
  experimentFetchError?: Error | ErrorWrapper | undefined;

  registeredModelVersionSummaries: RunPageModelVersionSummary[];

  // Only present in graphQL implementation
  apiError?: UseGetRunQueryDataApiError;
}

/**
 * An updated version of the `useRunDetailsPageData` hook that either uses the REST API-based implementation
 * or the GraphQL-based implementation to fetch run details, based on the `shouldEnableGraphQLRunDetailsPage` flag.
 */
export const useRunDetailsPageData = ({
  runUuid,
  experimentId,
}: {
  runUuid: string;
  experimentId: string;
}): UseRunDetailsPageDataResult => {
  const usingGraphQL = shouldEnableGraphQLRunDetailsPage();
  const dispatch = useDispatch<ThunkDispatch>();

  const enableWorkspaceModelsRegistryCall = true;

  // If GraphQL flag is enabled, use the graphQL query to fetch the run data.
  // We can safely disable the eslint rule since feature flag evaluation is stable
  /* eslint-disable react-hooks/rules-of-hooks */
  if (usingGraphQL) {
    const graphQLQuery = () =>
      useGetRunQuery({
        runUuid,
      });

    const detailsPageGraphqlResponse = graphQLQuery();

    // If model versions are colocated in the GraphQL response, we don't need to make an additional API call
    useEffect(() => {
      if (shouldEnableGraphQLModelVersionsForRunDetails()) {
        return;
      }
      if (enableWorkspaceModelsRegistryCall) {
        dispatch(searchModelVersionsApi({ run_id: runUuid }));
      }
    }, [dispatch, runUuid, enableWorkspaceModelsRegistryCall]);

    const { latestMetrics, tags, params, datasets } = useMemo(() => {
      // Filter out tags, metrics, and params that are entirely whitespace
      return {
        latestMetrics: pickBy(
          transformToKeyedObject<MetricEntity>(
            transformMetricValues(detailsPageGraphqlResponse.data?.data?.metrics ?? []),
          ),
          (metric) => metric.key.trim().length > 0,
        ),
        tags: pickBy(
          transformToKeyedObject<KeyValueEntity>(detailsPageGraphqlResponse.data?.data?.tags ?? []),
          (tag) => tag.key.trim().length > 0,
        ),
        params: pickBy(
          transformToKeyedObject<KeyValueEntity>(detailsPageGraphqlResponse.data?.data?.params ?? []),
          (param) => param.key.trim().length > 0,
        ),
        datasets: transformDatasets(detailsPageGraphqlResponse.data?.inputs?.datasetInputs),
      };
    }, [detailsPageGraphqlResponse.data]);

    const registeredModelVersionSummaries = useUnifiedRegisteredModelVersionsSummariesForRun({
      runUuid,
      queryResult: detailsPageGraphqlResponse,
    });

    return {
      runInfo: detailsPageGraphqlResponse.data?.info ?? undefined,
      experiment: detailsPageGraphqlResponse.data?.experiment ?? undefined,
      loading: detailsPageGraphqlResponse.loading,
      error: detailsPageGraphqlResponse.apolloError,
      apiError: detailsPageGraphqlResponse.apiError,
      refetchRun: detailsPageGraphqlResponse.refetchRun,
      runInputs: detailsPageGraphqlResponse.data?.inputs,
      runOutputs: detailsPageGraphqlResponse.data?.outputs,
      registeredModelVersionSummaries,
      datasets,
      latestMetrics,
      tags,
      params,
    };
  }

  // If GraphQL flag is disabled, use the legacy implementation to fetch the run data.
  const detailsPageResponse = useRunDetailsPageDataLegacy(runUuid, experimentId, enableWorkspaceModelsRegistryCall);
  const error = detailsPageResponse.errors.runFetchError || detailsPageResponse.errors.experimentFetchError;

  const registeredModelVersionSummaries = useUnifiedRegisteredModelVersionsSummariesForRun({
    runUuid,
  });

  return {
    runInfo: detailsPageResponse.data?.runInfo,
    latestMetrics: detailsPageResponse.data?.latestMetrics,
    tags: detailsPageResponse.data?.tags,
    experiment: detailsPageResponse.data?.experiment,
    params: detailsPageResponse.data?.params,
    datasets: detailsPageResponse.data?.datasets,
    loading: detailsPageResponse.loading,
    error,
    runFetchError: detailsPageResponse.errors.runFetchError,
    experimentFetchError: detailsPageResponse.errors.experimentFetchError,
    refetchRun: detailsPageResponse.refetchRun,
    registeredModelVersionSummaries,
  };
};
