import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useIntl } from 'react-intl';
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
  const intl = useIntl();

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
            Utils.displayGlobalErrorNotification(
              intl.formatMessage({
                defaultMessage: 'Display name and description could not be saved',
                description: 'Error notification when updating MCP server display name fails',
              }),
            );
          }
        }
      }

      if (icons !== undefined) {
        try {
          await MCPRegistryApi.updateMCPServer(name, { icons });
        } catch {
          Utils.displayGlobalErrorNotification(
            intl.formatMessage({
              defaultMessage: 'Icons could not be saved',
              description: 'Error notification when updating MCP server icons fails',
            }),
          );
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
        Utils.displayGlobalErrorNotification(
          intl.formatMessage({
            defaultMessage: 'Tags could not be saved',
            description: 'Error notification when saving MCP server tags fails',
          }),
        );
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
