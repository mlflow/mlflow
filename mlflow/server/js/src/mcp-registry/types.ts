export type MCPStatus = 'draft' | 'active' | 'deprecated' | 'deleted';

export enum TransportType {
  STDIO = 'stdio',
  STREAMABLE_HTTP = 'streamable-http',
  SSE = 'sse',
}

export enum ConnectionSource {
  PACKAGE = 'package',
  REMOTE = 'remote',
}

export enum ConnectionFormat {
  CLAUDE_CODE = 'claude-code',
  MCP_JSON = 'mcp-json',
}

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
  hidden_connect_options?: string[] | null;
  source?: string;
  created_by?: string;
  last_updated_by?: string;
  creation_timestamp?: number;
  last_updated_timestamp?: number;
}

// ServerJSON payload types use camelCase field names to match the MCP server.json specification

export interface ServerJSONInput {
  value?: string;
  default?: string;
  choices?: string[];
  placeholder?: string;
  valueHint?: string;
  format?: string;
  isRequired?: boolean;
  isSecret?: boolean;
  isRepeated?: boolean;
  description?: string;
  variables?: Record<string, ServerJSONInput>;
}

export interface ServerJSONEnvironmentVariable extends ServerJSONInput {
  name: string;
}

export interface ServerJSONNamedArgument extends ServerJSONInput {
  name: string;
}

export type ServerJSONPositionalArgument = ServerJSONInput;

export type ServerJSONArgument = ServerJSONNamedArgument | ServerJSONPositionalArgument;

export interface ServerJSONTransport {
  type: TransportType;
  url?: string;
  headers?: ServerJSONEnvironmentVariable[];
  variables?: Record<string, ServerJSONInput>;
}

export interface ServerJSONPackage {
  registryType: string;
  identifier: string;
  transport: ServerJSONTransport;
  registryBaseUrl?: string;
  version?: string;
  environmentVariables?: ServerJSONEnvironmentVariable[];
  runtimeHint?: string;
  runtimeArguments?: ServerJSONArgument[];
  packageArguments?: ServerJSONArgument[];
  fileSha256?: string;
  websiteUrl?: string;
  [key: string]: unknown;
}

export interface ServerJSONRepository {
  url: string;
  source?: string;
  id?: string;
  subfolder?: string;
}

export interface ServerJSONPayload {
  $schema?: string;
  name: string;
  version: string;
  title?: string;
  description?: string;
  packages?: ServerJSONPackage[];
  remotes?: ServerJSONTransport[];
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
  latest_version?: string | null;
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
  hidden_connect_options?: string[] | null;
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

// Response types

export interface SearchMCPServersResponse {
  mcp_servers: MCPServer[];
  next_page_token?: string;
}

export interface SearchMCPServerVersionsResponse {
  mcp_server_versions: MCPServerVersion[];
  next_page_token?: string;
}
