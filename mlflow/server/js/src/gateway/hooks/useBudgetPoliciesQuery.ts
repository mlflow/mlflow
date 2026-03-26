import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { ListBudgetPoliciesResponse } from '../types';

const DEFAULT_MAX_RESULTS = 10;

type BudgetPoliciesQueryKey = ['gateway_budget_policies', number, string | undefined];

export const useBudgetPoliciesQuery = (maxResults: number = DEFAULT_MAX_RESULTS, pageToken?: string) => {
  const queryResult = useQuery<ListBudgetPoliciesResponse, Error, ListBudgetPoliciesResponse, BudgetPoliciesQueryKey>(
    ['gateway_budget_policies', maxResults, pageToken],
    {
      queryFn: () => GatewayApi.listBudgetPolicies(maxResults, pageToken),
      retry: false,
    },
  );

  return {
    data: queryResult.data?.budget_policies ?? [],
    nextPageToken: queryResult.data?.next_page_token,
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    refetch: queryResult.refetch,
  };
};
