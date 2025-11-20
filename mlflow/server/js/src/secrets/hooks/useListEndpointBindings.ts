import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { endpointsApi } from '../api/endpointsApi';
import type { ListBindingsResponse } from '../types';

export const useListEndpointBindings = ({
  endpointId,
  enabled = true,
}: {
  endpointId: string;
  enabled?: boolean;
}) => {
  const { data, isLoading, error, refetch } = useQuery<ListBindingsResponse, Error>({
    queryKey: ['endpoint-bindings', endpointId],
    queryFn: async () => {
      return await endpointsApi.listEndpointBindings({ endpoint_id: endpointId });
    },
    refetchOnWindowFocus: false,
    retry: false,
    enabled: enabled && !!endpointId,
  });

  return {
    bindings: data?.bindings ?? [],
    isLoading,
    error,
    refetch,
  };
};
