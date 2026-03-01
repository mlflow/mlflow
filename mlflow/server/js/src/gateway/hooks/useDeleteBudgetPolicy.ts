import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';

export const useDeleteBudgetPolicy = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (budgetPolicyId: string) => GatewayApi.deleteBudgetPolicy(budgetPolicyId),
    onSuccess: () => {
      queryClient.invalidateQueries(['gateway_budget_policies']);
    },
  });
};
