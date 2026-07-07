import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { EndpointModelMapping } from '../types';

export const useDeleteEndpoint = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      endpointId,
      modelMappings,
    }: {
      endpointId: string;
      modelMappings: EndpointModelMapping[];
    }) => {
      await GatewayApi.deleteEndpoint(endpointId);

      const modelDefinitionIds = modelMappings.map((m) => m.model_definition_id);
      await Promise.allSettled(modelDefinitionIds.map((id) => GatewayApi.deleteModelDefinition(id)));
    },
    onSuccess: () => {
      queryClient.invalidateQueries(['gateway_endpoints']);
      queryClient.invalidateQueries(['gateway_bindings']);
      queryClient.invalidateQueries(['gateway_model_definitions']);
    },
  });
};
