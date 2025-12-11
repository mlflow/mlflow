import { useMutation, useQueryClient } from '../../common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { CreateModelDefinitionRequest } from '../types';

export const useCreateModelDefinitionMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: CreateModelDefinitionRequest) => GatewayApi.createModelDefinition(request),
    onSuccess: () => {
      queryClient.invalidateQueries(['gateway_model_definitions']);
    },
  });
};
