import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MCPRegistryApi } from '../api';
import type { CreateMCPAccessEndpointRequest, MCPAccessEndpoint, UpdateMCPAccessEndpointRequest } from '../types';
import { MCP_QUERY_KEYS } from '../utils';

const useInvalidateEndpointQueries = () => {
  const queryClient = useQueryClient();
  return (serverName: string) => {
    queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVER_ENDPOINTS, serverName]);
    queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVER, serverName]);
    queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVERS_LIST]);
  };
};

export const useCreateAccessEndpointMutation = () => {
  const invalidate = useInvalidateEndpointQueries();

  return useMutation<MCPAccessEndpoint, Error, { serverName: string; request: CreateMCPAccessEndpointRequest }>({
    mutationFn: ({ serverName, request }) => MCPRegistryApi.createMCPAccessEndpoint(serverName, request),
    onSuccess: (_data, { serverName }) => invalidate(serverName),
  });
};

export const useUpdateAccessEndpointMutation = () => {
  const invalidate = useInvalidateEndpointQueries();

  return useMutation<
    MCPAccessEndpoint,
    Error,
    { serverName: string; endpointId: string; request: UpdateMCPAccessEndpointRequest }
  >({
    mutationFn: ({ serverName, endpointId, request }) =>
      MCPRegistryApi.updateMCPAccessEndpoint(serverName, endpointId, request),
    onSuccess: (_data, { serverName }) => invalidate(serverName),
  });
};

export const useDeleteAccessEndpointMutation = () => {
  const invalidate = useInvalidateEndpointQueries();

  return useMutation<unknown, Error, { serverName: string; endpointId: string }>({
    mutationFn: ({ serverName, endpointId }) => MCPRegistryApi.deleteMCPAccessEndpoint(serverName, endpointId),
    onSuccess: (_data, { serverName }) => invalidate(serverName),
  });
};
