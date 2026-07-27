import { MCPRegistryApi } from '../api';
import type { SearchMCPServersResponse } from '../types';
import { MCP_QUERY_KEYS } from '../utils';
import { buildSearchFilterClause } from '../../common/utils/SearchUtils';
import { useCursorPaginatedQuery } from './useCursorPaginatedQuery';

const STATUS_ACTIVE_FILTER = "status = 'active'";
const HAS_ENDPOINTS_FILTER = "has_access_endpoints = 'true'";

export const useMCPServersListQuery = ({
  searchFilter,
  filterActive = true,
  filterHasEndpoints = true,
  enabled = true,
}: {
  searchFilter?: string;
  filterActive?: boolean;
  filterHasEndpoints?: boolean;
  enabled?: boolean;
} = {}) => {
  return useCursorPaginatedQuery<SearchMCPServersResponse, SearchMCPServersResponse['mcp_servers']>({
    queryKeyPrefix: MCP_QUERY_KEYS.SERVERS_LIST,
    searchFilter,
    extraQueryKeys: { filterActive, filterHasEndpoints },
    storageKey: 'mcp_registry.page_size',
    queryFn: ({ searchFilter: filter, pageToken, pageSize }) => {
      const nameClause = buildSearchFilterClause(filter);
      const clauses = [
        nameClause,
        filterActive ? STATUS_ACTIVE_FILTER : undefined,
        filterHasEndpoints ? HAS_ENDPOINTS_FILTER : undefined,
      ].filter(Boolean);
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
