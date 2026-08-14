import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { BudgetPolicyWindow, ListBudgetWindowsResponse } from '../types';
import { GatewayQueryKeys } from './queryKeys';

export const useBudgetWindowsQuery = () => {
  const queryResult = useQuery<
    ListBudgetWindowsResponse,
    Error,
    ListBudgetWindowsResponse,
    typeof GatewayQueryKeys.budgetWindows
  >(GatewayQueryKeys.budgetWindows, {
    queryFn: () => GatewayApi.getBudgetWindows(),
    retry: false,
  });

  const windowsByPolicyId: Record<string, BudgetPolicyWindow> = {};
  for (const w of queryResult.data?.windows ?? []) {
    windowsByPolicyId[w.budget_policy_id] = w;
  }

  return {
    data: windowsByPolicyId,
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
  };
};
