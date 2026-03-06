import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { ListBudgetPoliciesResponse } from '../types';

type BudgetPoliciesQueryKey = ['gateway_budget_policies'];

export const useBudgetPoliciesQuery = () => {
  const queryResult = useQuery<ListBudgetPoliciesResponse, Error, ListBudgetPoliciesResponse, BudgetPoliciesQueryKey>(
    ['gateway_budget_policies'],
    {
      queryFn: () => GatewayApi.listBudgetPolicies(),
      retry: false,
    },
  );

  return {
    data: queryResult.data?.budget_policies ?? [],
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    refetch: queryResult.refetch,
  };
};
