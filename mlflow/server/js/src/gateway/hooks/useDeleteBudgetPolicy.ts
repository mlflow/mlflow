import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import { GatewayQueryKeys } from './queryKeys';

export const useDeleteBudgetPolicy = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (budgetPolicyId: string) => GatewayApi.deleteBudgetPolicy(budgetPolicyId),
    onSuccess: () => {
      queryClient.invalidateQueries(GatewayQueryKeys.budgetPolicies);
      queryClient.invalidateQueries(GatewayQueryKeys.budgetWindows);
    },
  });
};
