import { useQuery } from '@databricks/web-shared/query-client';
import { GatewayApi } from '../api';
import { GatewayQueryKeys } from './queryKeys';

export const useSecretsConfigQuery = () => {
  return useQuery({
    queryKey: GatewayQueryKeys.secretsConfig,
    queryFn: () => GatewayApi.getSecretsConfig(),
    retry: false,
    staleTime: 30000,
  });
};
