import { fetchAPI, getAjaxUrl, HTTPMethods } from '../common/utils/FetchUtils';
import type {
  MCPServer,
  MCPServerVersion,
  MCPAccessBinding,
  CreateMCPServerRequest,
  UpdateMCPServerRequest,
  CreateMCPServerVersionRequest,
  UpdateMCPServerVersionRequest,
  CreateMCPAccessBindingRequest,
  UpdateMCPAccessBindingRequest,
  SetMCPServerTagRequest,
  SetMCPServerAliasRequest,
  SearchMCPServersParams,
  SearchMCPServerVersionsParams,
  SearchMCPAccessBindingsParams,
  SearchMCPServersResponse,
  SearchMCPServerVersionsResponse,
  SearchMCPAccessBindingsResponse,
} from './types';

const BASE_URL = 'ajax-api/3.0/mlflow/mcp-servers';

function buildSearchParams(params: Record<string, string | number | string[] | undefined>): string {
  const searchParams = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined) {
      continue;
    }
    if (Array.isArray(value)) {
      for (const item of value) {
        searchParams.append(key, item);
      }
    } else {
      searchParams.append(key, String(value));
    }
  }
  const queryString = searchParams.toString();
  return queryString ? `?${queryString}` : '';
}

// MCP Server endpoints

export const MCPRegistryApi = {
  createMCPServer: (request: CreateMCPServerRequest): Promise<MCPServer> => {
    return fetchAPI(getAjaxUrl(BASE_URL), {
      method: HTTPMethods.POST,
      body: request,
    }) as Promise<MCPServer>;
  },

  searchMCPServers: (params: SearchMCPServersParams = {}): Promise<SearchMCPServersResponse> => {
    const query = buildSearchParams({
      filter_string: params.filter_string,
      max_results: params.max_results,
      order_by: params.order_by,
      page_token: params.page_token,
    });
    return fetchAPI(getAjaxUrl(`${BASE_URL}${query}`)) as Promise<SearchMCPServersResponse>;
  },

  getMCPServer: (name: string): Promise<MCPServer> => {
    return fetchAPI(getAjaxUrl(`${BASE_URL}/${encodeURIComponent(name)}`)) as Promise<MCPServer>;
  },

  updateMCPServer: (name: string, request: UpdateMCPServerRequest): Promise<MCPServer> => {
    return fetchAPI(getAjaxUrl(`${BASE_URL}/${encodeURIComponent(name)}`), {
      method: HTTPMethods.PATCH,
      body: request,
    }) as Promise<MCPServer>;
  },

  deleteMCPServer: (name: string): Promise<void> => {
    return fetchAPI(getAjaxUrl(`${BASE_URL}/${encodeURIComponent(name)}`), {
      method: HTTPMethods.DELETE,
    }) as Promise<void>;
  },

  // MCP Server Version endpoints

  createMCPServerVersion: (name: string, request: CreateMCPServerVersionRequest): Promise<MCPServerVersion> => {
    return fetchAPI(getAjaxUrl(`${BASE_URL}/${encodeURIComponent(name)}/versions`), {
      method: HTTPMethods.POST,
      body: request,
    }) as Promise<MCPServerVersion>;
  },

  searchMCPServerVersions: (
    name: string,
    params: SearchMCPServerVersionsParams = {},
  ): Promise<SearchMCPServerVersionsResponse> => {
    const query = buildSearchParams({
      filter_string: params.filter_string,
      max_results: params.max_results,
      order_by: params.order_by,
      page_token: params.page_token,
    });
    return fetchAPI(
      getAjaxUrl(`${BASE_URL}/${encodeURIComponent(name)}/versions${query}`),
    ) as Promise<SearchMCPServerVersionsResponse>;
  },

  getMCPServerVersion: (name: string, version: string): Promise<MCPServerVersion> => {
    return fetchAPI(
      getAjaxUrl(`${BASE_URL}/${encodeURIComponent(name)}/versions/${encodeURIComponent(version)}`),
    ) as Promise<MCPServerVersion>;
  },

  updateMCPServerVersion: (
    name: string,
    version: string,
    request: UpdateMCPServerVersionRequest,
  ): Promise<MCPServerVersion> => {
    return fetchAPI(getAjaxUrl(`${BASE_URL}/${encodeURIComponent(name)}/versions/${encodeURIComponent(version)}`), {
      method: HTTPMethods.PATCH,
      body: request,
    }) as Promise<MCPServerVersion>;
  },

  deleteMCPServerVersion: (name: string, version: string): Promise<void> => {
    return fetchAPI(getAjaxUrl(`${BASE_URL}/${encodeURIComponent(name)}/versions/${encodeURIComponent(version)}`), {
      method: HTTPMethods.DELETE,
    }) as Promise<void>;
  },

  getLatestMCPServerVersion: (name: string): Promise<MCPServerVersion> => {
    return fetchAPI(getAjaxUrl(`${BASE_URL}/${encodeURIComponent(name)}/versions/latest`)) as Promise<MCPServerVersion>;
  },

  // MCP Access Binding endpoints

  createMCPAccessBinding: (name: string, request: CreateMCPAccessBindingRequest): Promise<MCPAccessBinding> => {
    return fetchAPI(getAjaxUrl(`${BASE_URL}/${encodeURIComponent(name)}/bindings`), {
      method: HTTPMethods.POST,
      body: request,
    }) as Promise<MCPAccessBinding>;
  },

  searchMCPAccessBindingsAll: (
    params: SearchMCPAccessBindingsParams = {},
  ): Promise<SearchMCPAccessBindingsResponse> => {
    const query = buildSearchParams({
      server_version: params.server_version,
      server_alias: params.server_alias,
      filter_string: params.filter_string,
      max_results: params.max_results,
      order_by: params.order_by,
      page_token: params.page_token,
    });
    return fetchAPI(getAjaxUrl(`${BASE_URL}/bindings${query}`)) as Promise<SearchMCPAccessBindingsResponse>;
  },

  searchMCPAccessBindings: (
    name: string,
    params: SearchMCPAccessBindingsParams = {},
  ): Promise<SearchMCPAccessBindingsResponse> => {
    const query = buildSearchParams({
      server_version: params.server_version,
      server_alias: params.server_alias,
      filter_string: params.filter_string,
      max_results: params.max_results,
      order_by: params.order_by,
      page_token: params.page_token,
    });
    return fetchAPI(
      getAjaxUrl(`${BASE_URL}/${encodeURIComponent(name)}/bindings${query}`),
    ) as Promise<SearchMCPAccessBindingsResponse>;
  },

  getMCPAccessBinding: (name: string, bindingId: number): Promise<MCPAccessBinding> => {
    return fetchAPI(
      getAjaxUrl(`${BASE_URL}/${encodeURIComponent(name)}/bindings/${bindingId}`),
    ) as Promise<MCPAccessBinding>;
  },

  updateMCPAccessBinding: (
    name: string,
    bindingId: number,
    request: UpdateMCPAccessBindingRequest,
  ): Promise<MCPAccessBinding> => {
    return fetchAPI(getAjaxUrl(`${BASE_URL}/${encodeURIComponent(name)}/bindings/${bindingId}`), {
      method: HTTPMethods.PATCH,
      body: request,
    }) as Promise<MCPAccessBinding>;
  },

  deleteMCPAccessBinding: (name: string, bindingId: number): Promise<void> => {
    return fetchAPI(getAjaxUrl(`${BASE_URL}/${encodeURIComponent(name)}/bindings/${bindingId}`), {
      method: HTTPMethods.DELETE,
    }) as Promise<void>;
  },

  // MCP Server Tag endpoints

  setMCPServerTag: (name: string, request: SetMCPServerTagRequest): Promise<void> => {
    return fetchAPI(getAjaxUrl(`${BASE_URL}/${encodeURIComponent(name)}/tags`), {
      method: HTTPMethods.POST,
      body: request,
    }) as Promise<void>;
  },

  deleteMCPServerTag: (name: string, key: string): Promise<void> => {
    return fetchAPI(getAjaxUrl(`${BASE_URL}/${encodeURIComponent(name)}/tags/${encodeURIComponent(key)}`), {
      method: HTTPMethods.DELETE,
    }) as Promise<void>;
  },

  // MCP Server Version Tag endpoints

  setMCPServerVersionTag: (name: string, version: string, request: SetMCPServerTagRequest): Promise<void> => {
    return fetchAPI(
      getAjaxUrl(`${BASE_URL}/${encodeURIComponent(name)}/versions/${encodeURIComponent(version)}/tags`),
      {
        method: HTTPMethods.POST,
        body: request,
      },
    ) as Promise<void>;
  },

  deleteMCPServerVersionTag: (name: string, version: string, key: string): Promise<void> => {
    return fetchAPI(
      getAjaxUrl(
        `${BASE_URL}/${encodeURIComponent(name)}/versions/${encodeURIComponent(version)}/tags/${encodeURIComponent(key)}`,
      ),
      {
        method: HTTPMethods.DELETE,
      },
    ) as Promise<void>;
  },

  // MCP Server Alias endpoints

  setMCPServerAlias: (name: string, request: SetMCPServerAliasRequest): Promise<void> => {
    return fetchAPI(getAjaxUrl(`${BASE_URL}/${encodeURIComponent(name)}/aliases`), {
      method: HTTPMethods.POST,
      body: request,
    }) as Promise<void>;
  },

  getMCPServerVersionByAlias: (name: string, alias: string): Promise<MCPServerVersion> => {
    return fetchAPI(
      getAjaxUrl(`${BASE_URL}/${encodeURIComponent(name)}/aliases/${encodeURIComponent(alias)}`),
    ) as Promise<MCPServerVersion>;
  },

  deleteMCPServerAlias: (name: string, alias: string): Promise<void> => {
    return fetchAPI(getAjaxUrl(`${BASE_URL}/${encodeURIComponent(name)}/aliases/${encodeURIComponent(alias)}`), {
      method: HTTPMethods.DELETE,
    }) as Promise<void>;
  },
};
