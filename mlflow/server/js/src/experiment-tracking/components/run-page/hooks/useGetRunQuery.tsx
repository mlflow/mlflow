import { type ApolloError, type ApolloQueryResult, gql } from '@apollo/client';
import type { GetRun, GetRunVariables } from '../../../../graphql/__generated__/graphql';
import { useQuery } from '@mlflow/mlflow/src/common/utils/graphQLHooks';

const GET_RUN_QUERY = gql`
  query GetRun($data: MlflowGetRunInput!) @component(name: "MLflow.ExperimentRunTracking") {
    mlflowGetRun(input: $data) {
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
            }
          }
        }
      }
    }
  }
`;

export type UseGetRunQueryResponseRunInfo = NonNullable<NonNullable<UseGetRunQueryDataResponse>['info']>;
export type UseGetRunQueryResponseExperiment = NonNullable<NonNullable<UseGetRunQueryDataResponse>['experiment']>;

export type UseGetRunQueryDataResponse = NonNullable<GetRun['mlflowGetRun']>['run'];
export type UseGetRunQueryResponse = {
  data?: UseGetRunQueryDataResponse;
  loading: boolean;
  error?: ApolloError;
  refetchRun: () => Promise<ApolloQueryResult<GetRun>>;
};

export const useGetRunQuery = ({
  runUuid,
  disabled = false,
}: {
  runUuid: string;
  disabled?: boolean;
}): UseGetRunQueryResponse => {
  const { data, loading, error, refetch } = useQuery<GetRun, GetRunVariables>(GET_RUN_QUERY, {
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
    error,
  } as const;
};
