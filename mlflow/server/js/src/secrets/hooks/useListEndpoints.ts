import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { endpointsApi_Legacy } from '../api/routesApi';
import type { ListEndpointsResponse_Legacy } from '../types';
import { LIST_ROUTES_QUERY_KEY } from '../constants';

export const useListEndpoints = ({ enabled = true }: { enabled?: boolean } = {}) => {
  const { data, isLoading, error, refetch } = useQuery<ListEndpointsResponse_Legacy, Error>({
    queryKey: [LIST_ROUTES_QUERY_KEY],
    queryFn: async () => {
      const response = await endpointsApi_Legacy.listEndpoints();
      // Backend returns {} instead of {routes: []} when empty due to protobuf serialization
      // Normalize to always have routes array
      return { routes: response.routes || [] };
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
