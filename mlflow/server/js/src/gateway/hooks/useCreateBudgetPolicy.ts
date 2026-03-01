import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { CreateBudgetPolicyRequest, CreateBudgetPolicyResponse } from '../types';

export const useCreateBudgetPolicy = () => {
  const queryClient = useQueryClient();

  return useMutation<CreateBudgetPolicyResponse, Error, CreateBudgetPolicyRequest>({
    mutationFn: (request) => GatewayApi.createBudgetPolicy(request),
    onSuccess: () => {
      queryClient.invalidateQueries(['gateway_budget_policies']);
    },
  });
};
