import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { BudgetPolicyWindow, ListBudgetWindowsResponse } from '../types';

export const useBudgetWindowsQuery = () => {
  const queryResult = useQuery<ListBudgetWindowsResponse, Error, ListBudgetWindowsResponse, ['gateway_budget_windows']>(
    ['gateway_budget_windows'],
    {
      queryFn: () => GatewayApi.getBudgetWindows(),
      retry: false,
    },
  );

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
