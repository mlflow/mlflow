import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { GatewayEndpointModelConfig } from '../types';

export const useUpdateEndpointMutation = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: {
      endpointId: string;
      name?: string;
      model_configs?: GatewayEndpointModelConfig[];
      routing_strategy?: string;
      fallback_config?: { strategy: string; max_attempts: number };
      usage_tracking?: boolean;
      experiment_id?: string;
    }) =>
      GatewayApi.updateEndpoint({
        endpoint_id: data.endpointId,
        name: data.name,
        model_configs: data.model_configs,
        routing_strategy: data.routing_strategy,
        fallback_config: data.fallback_config,
        usage_tracking: data.usage_tracking,
        experiment_id: data.experiment_id,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries(['gateway_endpoints']);
      queryClient.invalidateQueries(['gateway_endpoint']);
    },
  });
};
