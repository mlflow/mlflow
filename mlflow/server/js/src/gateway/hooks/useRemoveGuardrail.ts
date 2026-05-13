import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { RemoveGuardrailFromEndpointRequest } from '../types';
import { GatewayQueryKeys } from './queryKeys';

export const useRemoveGuardrail = () => {
  const queryClient = useQueryClient();

  return useMutation<void, Error, RemoveGuardrailFromEndpointRequest>({
    mutationFn: async (request): Promise<void> => {
      await GatewayApi.removeGuardrailFromEndpoint(request);
    },
    onSuccess: () => {
      queryClient.invalidateQueries(GatewayQueryKeys.guardrails);
    },
  });
};
