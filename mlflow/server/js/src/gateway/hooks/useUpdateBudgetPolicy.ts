import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { UpdateBudgetPolicyRequest, UpdateBudgetPolicyResponse } from '../types';

export const useUpdateBudgetPolicy = () => {
  const queryClient = useQueryClient();

  return useMutation<UpdateBudgetPolicyResponse, Error, UpdateBudgetPolicyRequest>({
    mutationFn: (request) => GatewayApi.updateBudgetPolicy(request),
    onSuccess: () => {
      queryClient.invalidateQueries(['gateway_budget_policies']);
    },
  });
};
