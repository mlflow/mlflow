import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { ListEndpointGuardrailConfigsResponse } from '../types';
import { GatewayQueryKeys } from './queryKeys';

export const useGuardrailsQuery = (endpointId?: string) => {
  const queryResult = useQuery<ListEndpointGuardrailConfigsResponse, Error>([...GatewayQueryKeys.guardrails, endpointId], {
    queryFn: () => GatewayApi.listEndpointGuardrailConfigs(endpointId!),
    retry: false,
    enabled: !!endpointId,
  });

  return {
    data: queryResult.data?.configs ?? [],
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    refetch: queryResult.refetch,
  };
};
