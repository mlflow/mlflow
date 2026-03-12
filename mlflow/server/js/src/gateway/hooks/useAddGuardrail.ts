import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { AddGuardrailRequest, AddGuardrailResponse } from '../types';
import { GatewayQueryKeys } from './queryKeys';

export const useAddGuardrail = () => {
  const queryClient = useQueryClient();

  return useMutation<AddGuardrailResponse, Error, AddGuardrailRequest>({
    mutationFn: (request) => GatewayApi.addGuardrail(request),
    onSuccess: () => {
      queryClient.invalidateQueries(GatewayQueryKeys.guardrails);
    },
  });
};
