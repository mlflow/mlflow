import { useQuery } from '../../common/utils/reactQueryHooks';
import { GatewayApi } from '../api';

export const useSecretQuery = (secretId: string | undefined) => {
  return useQuery(['gateway_secret', secretId], {
    queryFn: () => GatewayApi.getSecret(secretId!),
    enabled: Boolean(secretId),
  });
};
