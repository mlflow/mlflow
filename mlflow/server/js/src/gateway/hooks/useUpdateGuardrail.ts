import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import { GatewayQueryKeys } from './queryKeys';
import type { AddGuardrailResponse, UpdateGuardrailRequest } from '../types';

export const useUpdateGuardrail = () => {
  const queryClient = useQueryClient();

  return useMutation<AddGuardrailResponse, Error, UpdateGuardrailRequest>({
    mutationFn: (request) => GatewayApi.updateGuardrail(request),
    onSuccess: () => {
      queryClient.invalidateQueries(GatewayQueryKeys.guardrails);
    },
  });
};
