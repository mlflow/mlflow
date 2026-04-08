import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { CreateGatewayGuardrailRequest, CreateGatewayGuardrailResponse } from '../types';
import { GatewayQueryKeys } from './queryKeys';

export const useAddGuardrail = () => {
  const queryClient = useQueryClient();

  return useMutation<CreateGatewayGuardrailResponse, Error, CreateGatewayGuardrailRequest>({
    mutationFn: (request) => GatewayApi.createGuardrail(request),
    onSuccess: () => {
      queryClient.invalidateQueries(GatewayQueryKeys.guardrails);
    },
  });
};
