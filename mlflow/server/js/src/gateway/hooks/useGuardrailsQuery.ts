import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { ListGuardrailsResponse } from '../types';
import { GatewayQueryKeys } from './queryKeys';

export const useGuardrailsQuery = (endpointName?: string) => {
  const queryResult = useQuery<ListGuardrailsResponse, Error>([GatewayQueryKeys.guardrails, endpointName], {
    queryFn: () => GatewayApi.listGuardrails(endpointName),
    retry: false,
  });

  return {
    data: queryResult.data?.guardrails ?? [],
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    refetch: queryResult.refetch,
  };
};
