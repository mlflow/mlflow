import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import { GatewayQueryKeys } from './queryKeys';

export const useRemoveGuardrail = () => {
  const queryClient = useQueryClient();

  return useMutation<void, Error, string>({
    mutationFn: (guardrailId) => GatewayApi.removeGuardrail(guardrailId) as Promise<void>,
    onSuccess: () => {
      queryClient.invalidateQueries(GatewayQueryKeys.guardrails);
    },
  });
};
