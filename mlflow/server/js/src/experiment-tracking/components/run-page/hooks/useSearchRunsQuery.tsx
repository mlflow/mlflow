import { type ApolloError, type ApolloQueryResult, gql } from '@mlflow/mlflow/src/common/utils/graphQLHooks';
import { useQuery } from '@mlflow/mlflow/src/common/utils/graphQLHooks';
import { first } from 'lodash';
import type { SearchRuns } from '../../../../graphql/__generated__/graphql';

const SEARCH_RUNS_QUERY = gql`
  query SearchRuns($data: MlflowSearchRunsInput!) {
    mlflowSearchRuns(input: $data) {
      apiError {
        helpUrl
        code
        message
      }
      runs {
        info {
          runName
          status
          runUuid
          experimentId
          artifactUri
          endTime
          lifecycleStage
          startTime
          userId
        }
        experiment {
          experimentId
          name
          tags {
            key
            value
          }
          artifactLocation
          lifecycleStage
          lastUpdateTime
        }
        data {
          metrics {
            key
            value
            step
            timestamp
          }
          params {
            key
            value
          }
          tags {
            key
            value
          }
        }
        inputs {
          datasetInputs {
            dataset {
              digest
              name
              profile
              schema
              source
              sourceType
            }
            tags {
              key
              value
            }
          }
          modelInputs {
            modelId
          }
        }
        outputs {
          modelOutputs {
            modelId
            step
          }
        }
        modelVersions {
          version
          name
          creationTimestamp
          status
          source
        }
      }
    }
  }
`;

export type UseSearchRunsQueryResponseDataMetrics = NonNullable<
  NonNullable<NonNullable<UseSearchRunsQueryDataResponseSingleRun>['data']>['metrics']
>;
export type UseSearchRunsQueryDataResponseSingleRun = NonNullable<
  NonNullable<SearchRuns['mlflowSearchRuns']>['runs']
>[0];
export type UseSearchRunsQueryDataApiError = NonNullable<SearchRuns['mlflowSearchRuns']>['apiError'];
export type UseSearchRunsQueryResponse = {
  data?: UseSearchRunsQueryDataResponseSingleRun;
  loading: boolean;
  apolloError?: ApolloError;
  apiError?: UseSearchRunsQueryDataApiError;
  refetchRun: () => Promise<ApolloQueryResult<SearchRuns>>;
};

export const useSearchRunsQuery = ({
  filter,
  experimentIds,
  disabled = false,
}: {
  filter?: string;
  experimentIds: string[];
  disabled?: boolean;
}): UseSearchRunsQueryResponse => {
  const {
    data,
    loading,
    error: apolloError,
    refetch,
  } = useQuery<any, any>(SEARCH_RUNS_QUERY, {
    variables: {
      data: {
        filter,
        experimentIds,
      },
    },
    skip: disabled,
  });

  return {
    loading,
    data: first(data?.mlflowSearchRuns?.runs ?? []),
    refetchRun: refetch,
    apolloError,
    apiError: data?.mlflowSearchRuns?.apiError,
  } as const;
};
