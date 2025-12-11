import type { QueryFunctionContext } from '../../common/utils/reactQueryHooks';
import { useQuery } from '../../common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { ProvidersResponse } from '../types';

const queryFn = () => {
  return GatewayApi.listProviders();
};

type ProvidersQueryKey = ['gateway_providers'];

export const useProvidersQuery = () => {
  const queryResult = useQuery<ProvidersResponse, Error, ProvidersResponse, ProvidersQueryKey>(['gateway_providers'], {
    queryFn,
    retry: false,
  });

  return {
    data: queryResult.data?.providers ?? [],
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    refetch: queryResult.refetch,
  };
};
