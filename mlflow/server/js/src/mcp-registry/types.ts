export type MCPStatus = 'draft' | 'active' | 'deprecated' | 'deleted';

export type MCPRemoteTransportType = 'streamable-http' | 'sse';

// Entity types

export interface MCPTool {
  name: string;
  title?: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
  outputSchema?: Record<string, unknown>;
  annotations?: Record<string, unknown>;
  icons?: MCPIcon[];
  execution?: Record<string, unknown>;
}

export interface MCPIcon {
  src: string;
  sizes?: string[];
  mimeType?: string;
  theme?: string;
}

export interface MCPAccessBindingSummary {
  binding_id: number;
  server_name: string;
  endpoint_url: string;
  transport_type: MCPRemoteTransportType;
  server_version?: string;
  server_alias?: string;
  resolved_version?: MCPServerVersion;
}

export interface MCPServerAlias {
  alias: string;
  version: string;
}

export interface MCPServer {
  name: string;
  display_name?: string;
  description?: string;
  icons?: MCPIcon[];
  status?: MCPStatus;
  access_bindings: MCPAccessBindingSummary[];
  latest_version?: string;
  aliases: MCPServerAlias[];
  tags: Record<string, string>;
  created_by?: string;
  last_updated_by?: string;
  creation_timestamp?: number;
  last_updated_timestamp?: number;
}

export interface MCPServerVersion {
  name: string;
  version: string;
  server_json: ServerJSONPayload;
  display_name?: string;
  status: MCPStatus;
  tools?: MCPTool[];
  aliases: string[];
  tags: Record<string, string>;
  source?: string;
  created_by?: string;
  last_updated_by?: string;
  creation_timestamp?: number;
  last_updated_timestamp?: number;
}

export interface MCPAccessBinding {
  binding_id: number;
  server_name: string;
  endpoint_url: string;
  transport_type: MCPRemoteTransportType;
  tools?: MCPTool[];
  server_version?: string;
  server_alias?: string;
  resolved_version?: MCPServerVersion;
  created_by?: string;
  last_updated_by?: string;
  creation_timestamp?: number;
  last_updated_timestamp?: number;
}

// ServerJSON payload types use camelCase field names to match the MCP server.json specification

export interface ServerJSONEnvironmentVariable {
  name: string;
  description?: string;
  isRequired?: boolean;
  isSecret?: boolean;
}

export interface ServerJSONPackage {
  registryType: string;
  identifier: string;
  transport: string;
  registryBaseUrl?: string;
  version?: string;
  environmentVariables?: ServerJSONEnvironmentVariable[];
  [key: string]: unknown;
}

export interface ServerJSONRemote {
  type: string;
  url: string;
  [key: string]: unknown;
}

export interface ServerJSONRepository {
  url: string;
  source: string;
  id?: string;
  subfolder?: string;
}

export interface ServerJSONPayload {
  name: string;
  version: string;
  title?: string;
  description?: string;
  packages?: ServerJSONPackage[];
  remotes?: ServerJSONRemote[];
  repository?: ServerJSONRepository;
  websiteUrl?: string;
  _meta?: Record<string, unknown>;
  [key: string]: unknown;
}

// Request types

export interface CreateMCPServerRequest {
  name: string;
  description?: string;
  icons?: MCPIcon[];
}

export interface UpdateMCPServerRequest {
  display_name?: string | null;
  description?: string | null;
  icons?: MCPIcon[] | null;
}

export interface CreateMCPServerVersionRequest {
  server_json: ServerJSONPayload;
  display_name?: string;
  status?: MCPStatus;
  source?: string;
  tools?: MCPTool[];
}

export interface UpdateMCPServerVersionRequest {
  display_name?: string | null;
  status?: MCPStatus | null;
  tools?: MCPTool[] | null;
}

export interface CreateMCPAccessBindingRequest {
  server_version?: string;
  server_alias?: string;
  endpoint_url: string;
  transport_type?: MCPRemoteTransportType;
}

export interface UpdateMCPAccessBindingRequest {
  server_version?: string | null;
  server_alias?: string | null;
  endpoint_url?: string | null;
  transport_type?: MCPRemoteTransportType | null;
}

export interface SetMCPServerTagRequest {
  key: string;
  value: string;
}

export interface SetMCPServerAliasRequest {
  alias: string;
  version: string;
}

// Search parameters

export interface SearchMCPServersParams {
  filter_string?: string;
  max_results?: number;
  order_by?: string[];
  page_token?: string;
}

export interface SearchMCPServerVersionsParams {
  filter_string?: string;
  max_results?: number;
  order_by?: string[];
  page_token?: string;
}

export interface SearchMCPAccessBindingsParams {
  server_version?: string;
  server_alias?: string;
  filter_string?: string;
  max_results?: number;
  order_by?: string[];
  page_token?: string;
}

// Response types

export interface SearchMCPServersResponse {
  mcp_servers: MCPServer[];
  next_page_token?: string;
}

export interface SearchMCPServerVersionsResponse {
  mcp_server_versions: MCPServerVersion[];
  next_page_token?: string;
}

export interface SearchMCPAccessBindingsResponse {
  mcp_access_bindings: MCPAccessBinding[];
  next_page_token?: string;
}
