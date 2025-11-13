import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { routesApi } from '../api/routesApi';
import type { ListRoutesResponse } from '../types';

const LIST_ROUTES_QUERY_KEY = 'listRoutes';

export const useListRoutes = ({ enabled = true }: { enabled?: boolean } = {}) => {
  const { data, isLoading, error, refetch } = useQuery<ListRoutesResponse, Error>({
    queryKey: [LIST_ROUTES_QUERY_KEY],
    queryFn: async () => {
      const response = await routesApi.listRoutes();
      // Backend returns {} instead of {routes: []} when empty due to protobuf serialization
      // Normalize to always have routes array
      return { routes: response.routes || [] };
    },
    refetchOnWindowFocus: false,
    retry: false,
    enabled,
  });

  return {
    routes: data?.routes ?? [],
    isLoading,
    error,
    refetch,
  };
};
