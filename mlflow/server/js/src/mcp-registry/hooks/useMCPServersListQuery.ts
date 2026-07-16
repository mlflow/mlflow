import { MCPRegistryApi } from '../api';
import type { SearchMCPServersResponse } from '../types';
import { MCP_QUERY_KEYS } from '../utils';
import { buildSearchFilterClause } from '../../common/utils/SearchUtils';
import { useCursorPaginatedQuery } from './useCursorPaginatedQuery';

const AVAILABLE_FILTER = "status = 'active' AND has_access_bindings = 'true'";

export const useMCPServersListQuery = ({
  searchFilter,
  availableOnly = false,
  enabled = true,
}: { searchFilter?: string; availableOnly?: boolean; enabled?: boolean } = {}) => {
  return useCursorPaginatedQuery<SearchMCPServersResponse, SearchMCPServersResponse['mcp_servers']>({
    queryKeyPrefix: MCP_QUERY_KEYS.SERVERS_LIST,
    searchFilter,
    extraQueryKeys: { availableOnly },
    storageKey: 'mcp_registry.page_size',
    queryFn: ({ searchFilter: filter, pageToken, pageSize }) => {
      const nameClause = buildSearchFilterClause(filter);
      const clauses = [nameClause, availableOnly ? AVAILABLE_FILTER : undefined].filter(Boolean);
      return MCPRegistryApi.searchMCPServers({
        filter_string: clauses.length ? clauses.join(' AND ') : undefined,
        page_token: pageToken,
        max_results: pageSize,
      });
    },
    extractData: (response) => response.mcp_servers,
    enabled,
  });
};
