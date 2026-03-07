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

  return {
    data: queryResult.data?.windows ?? ({} as Record<string, BudgetPolicyWindow>),
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
  };
};
