import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { ListEndpointBindingsResponse } from '../types';

export const useBindingsQuery = () => {
  const queryResult = useQuery<ListEndpointBindingsResponse, Error>(['gateway_bindings'], {
    queryFn: () => GatewayApi.listEndpointBindings(),
    retry: false,
  });

  return {
    data: queryResult.data?.bindings ?? [],
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    refetch: queryResult.refetch,
  };
};
