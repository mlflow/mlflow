import { type ApolloError, type ApolloQueryResult, gql } from '@mlflow/mlflow/src/common/utils/graphQLHooks';
import type { GetRun, GetRunVariables } from '../../../../graphql/__generated__/graphql';
import { useQuery, useLazyQuery } from '@mlflow/mlflow/src/common/utils/graphQLHooks';

const GET_RUN_QUERY = gql`
  query GetRun($data: MlflowGetRunInput!) @component(name: "MLflow.ExperimentRunTracking") {
    mlflowGetRun(input: $data) {
      apiError {
        helpUrl
        code
        message
      }
      run {
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
        modelVersions {
          status
          version
          name
          source
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
      }
    }
  }
`;

export type UseGetRunQueryResponseRunInfo = NonNullable<NonNullable<UseGetRunQueryDataResponse>['info']>;
export type UseGetRunQueryResponseDatasetInputs = NonNullable<
  NonNullable<UseGetRunQueryDataResponse>['inputs']
>['datasetInputs'];
export type UseGetRunQueryResponseInputs = NonNullable<UseGetRunQueryDataResponse>['inputs'];
export type UseGetRunQueryResponseOutputs = NonNullable<UseGetRunQueryDataResponse>['outputs'];
export type UseGetRunQueryResponseExperiment = NonNullable<NonNullable<UseGetRunQueryDataResponse>['experiment']>;
export type UseGetRunQueryResponseDataMetrics = NonNullable<
  NonNullable<NonNullable<UseGetRunQueryDataResponse>['data']>['metrics']
>;

export type UseGetRunQueryDataResponse = NonNullable<GetRun['mlflowGetRun']>['run'];
export type UseGetRunQueryDataApiError = NonNullable<GetRun['mlflowGetRun']>['apiError'];
export type UseGetRunQueryResponse = {
  data?: UseGetRunQueryDataResponse;
  loading: boolean;
  apolloError?: ApolloError;
  apiError?: UseGetRunQueryDataApiError;
  refetchRun: () => Promise<ApolloQueryResult<GetRun>>;
};

export const useGetRunQuery = ({
  runUuid,
  disabled = false,
}: {
  runUuid: string;
  disabled?: boolean;
}): UseGetRunQueryResponse => {
  const {
    data,
    loading,
    error: apolloError,
    refetch,
  } = useQuery<GetRun, GetRunVariables>(GET_RUN_QUERY, {
    variables: {
      data: {
        runId: runUuid,
      },
    },
    skip: disabled,
  });

  return {
    loading,
    data: data?.mlflowGetRun?.run,
    refetchRun: refetch,
    apolloError,
    apiError: data?.mlflowGetRun?.apiError,
  } as const;
};

export const useLazyGetRunQuery = () => useLazyQuery<GetRun, GetRunVariables>(GET_RUN_QUERY);
