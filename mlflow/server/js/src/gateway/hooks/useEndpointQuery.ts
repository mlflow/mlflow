import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';

export const useEndpointQuery = (endpointId: string) => {
  return useQuery(['gateway_endpoint', endpointId], {
    queryFn: () => GatewayApi.getEndpoint(endpointId),
    retry: false,
    enabled: Boolean(endpointId),
  });
};
