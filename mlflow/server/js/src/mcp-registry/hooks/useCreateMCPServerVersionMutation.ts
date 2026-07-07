import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MCPRegistryApi } from '../api';
import type { MCPServerVersion, MCPStatus, MCPTool, ServerJSONPayload } from '../types';
import { MCP_QUERY_KEYS } from '../utils';

type CreateMCPServerVersionPayload = {
  serverJson: ServerJSONPayload;
  displayName?: string;
  isNewServer?: boolean;
  status?: MCPStatus;
  source?: string;
  tools?: MCPTool[];
  tags?: Record<string, string>;
};

export const useCreateMCPServerVersionMutation = () => {
  const queryClient = useQueryClient();

  return useMutation<MCPServerVersion, Error, CreateMCPServerVersionPayload>({
    mutationFn: async ({ serverJson, displayName, isNewServer, status, source, tools, tags }) => {
      const name = serverJson.name;
      const version = await MCPRegistryApi.createMCPServerVersion(name, {
        server_json: serverJson,
        display_name: displayName || undefined,
        status,
        source,
        tools,
      });

      try {
        if (isNewServer) {
          const serverDisplayName = displayName || serverJson.title;
          if (serverDisplayName || serverJson.description) {
            await MCPRegistryApi.updateMCPServer(name, {
              display_name: serverDisplayName || undefined,
              description: serverJson.description || undefined,
            });
          }
        }

        if (tags) {
          const setTag = isNewServer
            ? (key: string, value: string) => MCPRegistryApi.setMCPServerTag(name, { key, value })
            : (key: string, value: string) =>
                MCPRegistryApi.setMCPServerVersionTag(name, version.version, { key, value });
          await Promise.all(Object.entries(tags).map(([key, value]) => setTag(key, value)));
        }
      } catch {
        // Version was created successfully; metadata/tag failures are non-fatal
      }

      return version;
    },
    onSuccess: (_data, { serverJson }) => {
      const name = serverJson.name;
      queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVERS_LIST]);
      queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVER, name]);
      queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVER_VERSIONS, name]);
      queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVER_LATEST_VERSION, name]);
    },
  });
};
