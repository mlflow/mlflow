import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MCPRegistryApi } from '../api';
import type {
  MCPServer,
  MCPServerVersion,
  SearchMCPServerVersionsResponse,
  SearchMCPAccessEndpointsResponse,
} from '../types';
import { MCP_QUERY_KEYS } from '../utils';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import { ErrorCodes } from '../../common/constants';

export const useMCPServerQuery = (name: string) => {
  return useQuery<MCPServer, Error>([MCP_QUERY_KEYS.SERVER, name], {
    queryFn: () => MCPRegistryApi.getMCPServer(name),
    retry: false,
    enabled: Boolean(name),
  });
};

export const useMCPServerVersionsQuery = (name: string) => {
  const queryResult = useQuery<SearchMCPServerVersionsResponse, Error>([MCP_QUERY_KEYS.SERVER_VERSIONS, name], {
    queryFn: () => MCPRegistryApi.searchMCPServerVersions(name, { order_by: ['`version` DESC'], max_results: 100 }),
    retry: false,
    enabled: Boolean(name),
  });

  return {
    ...queryResult,
    data: queryResult.data?.mcp_server_versions,
    hasMoreVersions: Boolean(queryResult.data?.next_page_token),
  };
};

export const useLatestMCPServerVersionQuery = (name: string, enabled = true) => {
  return useQuery<MCPServerVersion | undefined, Error>([MCP_QUERY_KEYS.SERVER_LATEST_VERSION, name], {
    queryFn: async () => {
      try {
        return await MCPRegistryApi.getLatestMCPServerVersion(name);
      } catch (e: unknown) {
        if (
          e instanceof ErrorWrapper &&
          (e.getStatus() === 404 || e.getErrorCode() === ErrorCodes.RESOURCE_DOES_NOT_EXIST)
        ) {
          return undefined;
        }
        throw e;
      }
    },
    retry: false,
    enabled: Boolean(name) && enabled,
  });
};

export const useMCPAccessEndpointsQuery = (name: string) => {
  const queryResult = useQuery<SearchMCPAccessEndpointsResponse, Error>([MCP_QUERY_KEYS.SERVER_ENDPOINTS, name], {
    queryFn: () => MCPRegistryApi.searchMCPAccessEndpoints(name),
    retry: false,
    enabled: Boolean(name),
  });

  return {
    ...queryResult,
    data: queryResult.data?.mcp_access_endpoints,
  };
};
