/**
 * React Query hook for fetching assistant configuration.
 */

import type { QueryFunctionContext } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { getConfig } from '../AssistantService';
import type { AssistantConfig } from '../types';

type AssistantConfigQueryKey = ['assistant_config'];

const queryFn = async ({}: QueryFunctionContext<AssistantConfigQueryKey>) => {
  return getConfig();
};

export const useAssistantConfigQuery = () => {
  const queryResult = useQuery<AssistantConfig, Error, AssistantConfig, AssistantConfigQueryKey>(['assistant_config'], {
    queryFn,
    retry: false,
  });

  return {
    config: queryResult.data,
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    refetch: queryResult.refetch,
  };
};
