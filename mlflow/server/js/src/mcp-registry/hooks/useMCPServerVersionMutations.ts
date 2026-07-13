import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MCPRegistryApi } from '../api';
import type { MCPStatus, MCPTool } from '../types';
import { MCP_QUERY_KEYS } from '../utils';

const useInvalidateServerQueries = () => {
  const queryClient = useQueryClient();
  return (serverName: string) => {
    queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVER, serverName]);
    queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVER_VERSIONS, serverName]);
    queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVER_LATEST_VERSION, serverName]);
    queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVERS_LIST]);
  };
};

type UpdateMCPServerVersionPayload = {
  version: string;
  displayName?: string;
  status?: MCPStatus;
  tools?: MCPTool[] | null;
  hiddenConnectOptions?: string[] | null;
  aliases?: { add: string[]; remove: string[] };
};

export const useUpdateMCPServerVersion = (serverName: string) => {
  const invalidate = useInvalidateServerQueries();

  return useMutation<unknown, Error, UpdateMCPServerVersionPayload>({
    mutationFn: async ({ version, displayName, status, tools, hiddenConnectOptions, aliases }) => {
      const versionUpdate: Partial<{
        display_name: string | null;
        status: MCPStatus;
        tools: MCPTool[] | null;
        hidden_connect_options: string[] | null;
      }> = {};
      if (displayName !== undefined) {
        versionUpdate['display_name'] = displayName || null;
      }
      if (status !== undefined) {
        versionUpdate['status'] = status;
      }
      if (tools !== undefined) {
        versionUpdate['tools'] = tools;
      }
      if (hiddenConnectOptions !== undefined) {
        versionUpdate['hidden_connect_options'] = hiddenConnectOptions;
      }

      const promises: Promise<unknown>[] = [];

      if (Object.keys(versionUpdate).length > 0) {
        promises.push(MCPRegistryApi.updateMCPServerVersion(serverName, version, versionUpdate));
      }

      if (aliases) {
        promises.push(
          ...aliases.add.map((alias) => MCPRegistryApi.setMCPServerAlias(serverName, { alias, version })),
          ...aliases.remove.map((alias) => MCPRegistryApi.deleteMCPServerAlias(serverName, alias)),
        );
      }

      await Promise.all(promises);
    },
    onSuccess: () => invalidate(serverName),
  });
};

export const useDeleteMCPServerVersion = (serverName: string) => {
  const invalidate = useInvalidateServerQueries();

  return useMutation<unknown, Error, string>({
    mutationFn: (version) => MCPRegistryApi.deleteMCPServerVersion(serverName, version),
    onSuccess: () => invalidate(serverName),
  });
};

export const useUpdateMCPServerDisplayName = (serverName: string) => {
  const invalidate = useInvalidateServerQueries();

  return useMutation<unknown, Error, string | null>({
    mutationFn: (displayName: string | null) =>
      MCPRegistryApi.updateMCPServer(serverName, { display_name: displayName }),
    onSuccess: () => invalidate(serverName),
  });
};

export const useDeleteMCPServer = () => {
  const queryClient = useQueryClient();

  return useMutation<unknown, Error, string>({
    mutationFn: (name) => MCPRegistryApi.deleteMCPServer(name),
    onSuccess: () => {
      queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVERS_LIST]);
    },
  });
};
