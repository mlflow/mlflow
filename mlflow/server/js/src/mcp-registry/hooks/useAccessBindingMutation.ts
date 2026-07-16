import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MCPRegistryApi } from '../api';
import type { CreateMCPAccessBindingRequest, MCPAccessBinding, UpdateMCPAccessBindingRequest } from '../types';
import { MCP_QUERY_KEYS } from '../utils';

const useInvalidateBindingQueries = () => {
  const queryClient = useQueryClient();
  return (serverName: string) => {
    queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVER_BINDINGS, serverName]);
    queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVER, serverName]);
    queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVERS_LIST]);
  };
};

export const useCreateAccessBindingMutation = () => {
  const invalidate = useInvalidateBindingQueries();

  return useMutation<MCPAccessBinding, Error, { serverName: string; request: CreateMCPAccessBindingRequest }>({
    mutationFn: ({ serverName, request }) => MCPRegistryApi.createMCPAccessBinding(serverName, request),
    onSuccess: (_data, { serverName }) => invalidate(serverName),
  });
};

export const useUpdateAccessBindingMutation = () => {
  const invalidate = useInvalidateBindingQueries();

  return useMutation<
    MCPAccessBinding,
    Error,
    { serverName: string; bindingId: number; request: UpdateMCPAccessBindingRequest }
  >({
    mutationFn: ({ serverName, bindingId, request }) =>
      MCPRegistryApi.updateMCPAccessBinding(serverName, bindingId, request),
    onSuccess: (_data, { serverName }) => invalidate(serverName),
  });
};

export const useDeleteAccessBindingMutation = () => {
  const invalidate = useInvalidateBindingQueries();

  return useMutation<unknown, Error, { serverName: string; bindingId: number }>({
    mutationFn: ({ serverName, bindingId }) => MCPRegistryApi.deleteMCPAccessBinding(serverName, bindingId),
    onSuccess: (_data, { serverName }) => invalidate(serverName),
  });
};
