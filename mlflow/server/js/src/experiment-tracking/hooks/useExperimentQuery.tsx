import type { QueryHookOptions } from '@mlflow/mlflow/src/common/utils/graphQLHooks';
import { gql } from '@mlflow/mlflow/src/common/utils/graphQLHooks';
import { useQuery } from '@mlflow/mlflow/src/common/utils/graphQLHooks';
import type { MlflowGetExperimentQuery, MlflowGetExperimentQueryVariables } from '../../graphql/__generated__/graphql';
import { isArray } from 'lodash';
import { NotFoundError } from '@databricks/web-shared/errors';

const GET_EXPERIMENT_QUERY = gql`
  query MlflowGetExperimentQuery($input: MlflowGetExperimentInput!) @component(name: "MLflow.ExperimentRunTracking") {
    mlflowGetExperiment(input: $input) {
      apiError {
        code
        message
      }
      experiment {
        artifactLocation
        creationTime
        experimentId
        lastUpdateTime
        lifecycleStage
        name
        tags {
          key
          value
        }
      }
    }
  }
`;

export type UseGetExperimentQueryResultExperiment = NonNullable<
  MlflowGetExperimentQuery['mlflowGetExperiment']
>['experiment'];

/* eslint-disable react-hooks/rules-of-hooks */
export const useGetExperimentQuery = ({
  experimentId,
  options = {},
}: {
  experimentId?: string;
  options?: QueryHookOptions<MlflowGetExperimentQuery, MlflowGetExperimentQueryVariables>;
}) => {
  const {
    data,
    loading,
    error: apolloError,
    refetch,
  } = useQuery<MlflowGetExperimentQuery, MlflowGetExperimentQueryVariables>(GET_EXPERIMENT_QUERY, {
    variables: {
      input: {
        experimentId,
      },
    },
    skip: !experimentId,
    ...options,
  });

  // Extract the single experiment entity from the response
  const experimentEntity: UseGetExperimentQueryResultExperiment | undefined = data?.mlflowGetExperiment?.experiment;

  const getApiError = () => {
    return data?.mlflowGetExperiment?.apiError;
  };

  return {
    loading,
    data: experimentEntity,
    refetch,
    apolloError: apolloError,
    apiError: getApiError(),
  } as const;
};
