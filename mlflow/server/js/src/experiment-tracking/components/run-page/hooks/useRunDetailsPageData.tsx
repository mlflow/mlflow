import { isEmpty, keyBy } from 'lodash';
import { useEffect, useMemo } from 'react';
import { useRunDetailsPageDataLegacy } from '../useRunDetailsPageDataLegacy';
import {
  type UseGetRunQueryResponseExperiment,
  useGetRunQuery,
  UseGetRunQueryDataApiError,
  UseGetRunQueryResponseDataMetrics,
  UseGetRunQueryResponseDatasetInputs,
  UseGetRunQueryResponseRunInfo,
} from './useGetRunQuery';
import {
  KeyValueEntity,
  RunDatasetWithTags,
  type ExperimentEntity,
  type MetricEntitiesByName,
  type MetricEntity,
  type RunInfoEntity,
} from '../../../types';
import { shouldEnableGraphQLRunDetailsPage } from '../../../../common/utils/FeatureUtils';
import { ThunkDispatch } from '../../../../redux-types';
import { useDispatch } from 'react-redux';
import { searchModelVersionsApi } from '../../../../model-registry/actions';
import { ApolloError } from '@apollo/client';
import { ErrorWrapper } from '../../../../common/utils/ErrorWrapper';
import { pickBy } from 'lodash';

// Internal util: transforms an array of objects into a keyed object by the `key` field
const transformToKeyedObject = <Output, Input = any>(inputArray: Input[]) =>
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
const transformDatasets = (inputArray?: UseGetRunQueryResponseDatasetInputs): RunDatasetWithTags[] | undefined =>
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

  // Only present in legacy implementation
  runFetchError?: Error | ErrorWrapper | undefined;
  experimentFetchError?: Error | ErrorWrapper | undefined;

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

  // If GraphQL flag is enabled, use the graphQL query to fetch the run data.
  // We can safely disable the eslint rule since feature flag evaluation is stable
  /* eslint-disable react-hooks/rules-of-hooks */
  if (usingGraphQL) {
    const detailsPageGraphqlResponse = useGetRunQuery({
      runUuid,
    });

    // Model versions are not fully supported by GraphQL yet, so we need to fetch them separately
    useEffect(() => {
      dispatch(searchModelVersionsApi({ run_id: runUuid }));
    }, [dispatch, runUuid]);

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

    return {
      runInfo: detailsPageGraphqlResponse.data?.info ?? undefined,
      experiment: detailsPageGraphqlResponse.data?.experiment ?? undefined,
      loading: detailsPageGraphqlResponse.loading,
      error: detailsPageGraphqlResponse.apolloError,
      apiError: detailsPageGraphqlResponse.apiError,
      refetchRun: detailsPageGraphqlResponse.refetchRun,
      datasets,
      latestMetrics,
      tags,
      params,
    };
  }

  // If GraphQL flag is disabled, use the legacy implementation to fetch the run data.
  const detailsPageResponse = useRunDetailsPageDataLegacy(runUuid, experimentId);
  const error = detailsPageResponse.errors.runFetchError || detailsPageResponse.errors.experimentFetchError;

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
  };
};
