import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { endpointsApi } from '../api/endpointsApi';
import type { ListEndpointsResponse } from '../types';
import { LIST_ROUTES_QUERY_KEY } from '../constants';

export const useListEndpoints = ({ enabled = true }: { enabled?: boolean } = {}) => {
  const { data, isLoading, error, refetch } = useQuery<ListEndpointsResponse, Error>({
    queryKey: [LIST_ROUTES_QUERY_KEY],
    queryFn: async () => {
      // The backend returns Endpoint[] with models already populated
      // No transformation needed - just return the response directly
      return endpointsApi.listEndpoints();
    },
    refetchOnWindowFocus: false,
    retry: false,
    enabled,
  });

  return {
    endpoints: data?.routes ?? [],
    isLoading,
    error,
    refetch,
  };
};
