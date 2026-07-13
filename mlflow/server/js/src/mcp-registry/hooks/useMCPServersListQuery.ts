import { MCPRegistryApi } from '../api';
import type { SearchMCPServersResponse } from '../types';
import { MCP_QUERY_KEYS } from '../utils';
import { buildSearchFilterClause } from '../../common/utils/SearchUtils';
import { useCursorPaginatedQuery } from './useCursorPaginatedQuery';

export const useMCPServersListQuery = ({
  searchFilter,
  enabled = true,
}: { searchFilter?: string; enabled?: boolean } = {}) => {
  return useCursorPaginatedQuery<SearchMCPServersResponse, SearchMCPServersResponse['mcp_servers']>({
    queryKeyPrefix: MCP_QUERY_KEYS.SERVERS_LIST,
    searchFilter,
    storageKey: 'mcp_registry.page_size',
    queryFn: ({ searchFilter: filter, pageToken, pageSize }) =>
      MCPRegistryApi.searchMCPServers({
        filter_string: buildSearchFilterClause(filter),
        page_token: pageToken,
        max_results: pageSize,
      }),
    extractData: (response) => response.mcp_servers,
    enabled,
  });
};
