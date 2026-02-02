import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MlflowService } from '../../experiment-tracking/sdk/MlflowService';
import type { SearchExperimentsApiResponse } from '../../experiment-tracking/types';

const QUERY_KEY = 'experiments_for_select';

/**
 * Hook to fetch experiments for use in select components.
 * Returns a list of active experiments sorted by name.
 */
export const useExperimentsForSelect = () => {
  const { data, isLoading, error } = useQuery<SearchExperimentsApiResponse, Error>(
    [QUERY_KEY],
    () =>
      MlflowService.searchExperiments([
        ['max_results', '1000'],
        ['order_by', 'name ASC'],
        ['view_type', 'ACTIVE_ONLY'],
      ]),
    {
      staleTime: 30000, // Cache for 30 seconds
    },
  );

  return {
    experiments: data?.experiments ?? [],
    isLoading,
    error,
  };
};
