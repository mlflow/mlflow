import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import { GatewayQueryKeys } from './queryKeys';
import type { UpdateEndpointGuardrailConfigRequest, UpdateEndpointGuardrailConfigResponse } from '../types';

export const useUpdateGuardrail = () => {
  const queryClient = useQueryClient();

  return useMutation<UpdateEndpointGuardrailConfigResponse, Error, UpdateEndpointGuardrailConfigRequest>({
    mutationFn: (request) => GatewayApi.updateEndpointGuardrailConfig(request),
    onSuccess: () => {
      queryClient.invalidateQueries(GatewayQueryKeys.guardrails);
    },
  });
};
