import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MCPRegistryApi } from '../api';
import type { ConnectOptionsMap, MCPIcon, MCPServerVersion, MCPStatus, MCPTool, ServerJSONPayload } from '../types';
import { MCP_QUERY_KEYS } from '../utils';
import Utils from '../../common/utils/Utils';

type CreateMCPServerVersionPayload = {
  serverJson: ServerJSONPayload;
  displayName?: string;
  isNewServer?: boolean;
  status?: MCPStatus;
  source?: string;
  icons?: MCPIcon[] | null;
  tools?: MCPTool[];
  tags?: Record<string, string>;
  connectOptions?: ConnectOptionsMap;
};

export const useCreateMCPServerVersionMutation = () => {
  const queryClient = useQueryClient();

  return useMutation<MCPServerVersion, Error, CreateMCPServerVersionPayload>({
    mutationFn: async ({
      serverJson,
      displayName,
      isNewServer,
      status,
      source,
      icons,
      tools,
      tags,
      connectOptions,
    }) => {
      const name = serverJson.name;
      const version = await MCPRegistryApi.createMCPServerVersion(name, {
        server_json: serverJson,
        display_name: isNewServer ? undefined : displayName || undefined,
        status,
        source,
        tools,
        connect_options: connectOptions,
      });

      if (isNewServer) {
        const serverDisplayName = displayName || serverJson.title;
        if (serverDisplayName || serverJson.description) {
          try {
            await MCPRegistryApi.updateMCPServer(name, {
              display_name: serverDisplayName || undefined,
              description: serverJson.description || undefined,
            });
          } catch {
            Utils.displayGlobalErrorNotification('Display name and description could not be saved');
          }
        }
      }

      if (icons !== undefined) {
        try {
          await MCPRegistryApi.updateMCPServer(name, { icons });
        } catch {
          Utils.displayGlobalErrorNotification('Icons could not be saved');
        }
      }

      try {
        if (tags) {
          const setTag = isNewServer
            ? (key: string, value: string) => MCPRegistryApi.setMCPServerTag(name, { key, value })
            : (key: string, value: string) =>
                MCPRegistryApi.setMCPServerVersionTag(name, version.version, { key, value });
          await Promise.all(Object.entries(tags).map(([key, value]) => setTag(key, value)));
        }
      } catch {
        Utils.displayGlobalErrorNotification('Tags could not be saved');
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
