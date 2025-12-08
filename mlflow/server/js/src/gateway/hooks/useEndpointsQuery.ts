import type { QueryFunctionContext } from '../../common/utils/reactQueryHooks';
import { useQuery } from '../../common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { ListEndpointsResponse } from '../types';

const queryFn = ({ queryKey }: QueryFunctionContext<EndpointsQueryKey>) => {
  const [, { provider }] = queryKey;
  return GatewayApi.listEndpoints(provider);
};

type EndpointsQueryKey = ['gateway_endpoints', { provider?: string }];

export const useEndpointsQuery = ({ provider }: { provider?: string } = {}) => {
  const queryResult = useQuery<ListEndpointsResponse, Error, ListEndpointsResponse, EndpointsQueryKey>(
    ['gateway_endpoints', { provider }],
    {
      queryFn,
      retry: false,
    },
  );

  return {
    data: queryResult.data?.endpoints,
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    refetch: queryResult.refetch,
  };
};
