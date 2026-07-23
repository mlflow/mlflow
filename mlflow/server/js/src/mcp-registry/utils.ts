import type { TagProps } from '@databricks/design-system';
import type {
  ConnectOptionKey,
  MCPAccessEndpoint,
  MCPServer,
  MCPRemoteTransportType,
  MCPTool,
  PackageConnectOptionKey,
  RemoteConnectOptionKey,
  ServerJSONPayload,
} from './types';
import { MCPStatus, MCPServerAction } from './types';

export const sanitizeHref = (url: string | undefined): string | undefined => {
  if (!url) return undefined;
  try {
    const parsed = new URL(url);
    if (parsed.protocol === 'http:' || parsed.protocol === 'https:') {
      return url;
    }
  } catch {
    // malformed URL
  }
  return undefined;
};

export const STATUS_TAG_COLOR: Record<MCPStatus, TagProps['color']> = {
  [MCPStatus.DRAFT]: 'charcoal',
  [MCPStatus.ACTIVE]: 'lime',
  [MCPStatus.DEPRECATED]: 'lemon',
  [MCPStatus.DELETED]: 'coral',
};

export const STATUS_TRANSITIONS: Record<MCPStatus, MCPStatus[]> = {
  [MCPStatus.DRAFT]: [MCPStatus.ACTIVE],
  [MCPStatus.ACTIVE]: [MCPStatus.DRAFT, MCPStatus.DEPRECATED],
  [MCPStatus.DEPRECATED]: [MCPStatus.ACTIVE],
  [MCPStatus.DELETED]: [],
};

export const LATEST_ALIAS = 'latest';

export const RESERVED_ALIASES = [LATEST_ALIAS];

export const MCP_QUERY_KEYS = {
  SERVERS_LIST: 'mcp_servers_list',
  SERVER: 'mcp_server',
  SERVER_VERSIONS: 'mcp_server_versions',
  SERVER_LATEST_VERSION: 'mcp_server_latest_version',
  SERVER_ENDPOINTS: 'mcp_server_endpoints',
} as const;

export const DEFAULT_PAGE_SIZE = 25;
export const PAGE_SIZE_OPTIONS = [10, 25, 50, 100];

export const resolveDisplayName = (server: { display_name?: string; name: string }): string => {
  return server.display_name || server.name;
};

const resolveVersionDisplayName = (
  version: { display_name?: string; server_json?: { title?: string } } | null | undefined,
  fallback: string,
): string => {
  return version?.display_name || version?.server_json?.title || fallback;
};

export const resolveEndpointDisplayName = (endpoint: {
  server_name: string;
  resolved_version?: { display_name?: string; server_json?: { title?: string } } | null;
}): string => {
  return resolveVersionDisplayName(endpoint.resolved_version, endpoint.server_name);
};

const TRANSPORT_LABELS: Record<MCPRemoteTransportType, string> = {
  'streamable-http': 'streamable-http',
  sse: 'sse',
};

export const formatTransportType = (transport: MCPRemoteTransportType): string => {
  return TRANSPORT_LABELS[transport] || transport;
};

export const isValidEndpointUrl = (url: string): boolean => {
  const trimmed = url.trim();
  if (!/^https?:\/\//.test(trimmed)) return false;
  try {
    return Boolean(new URL(trimmed).hostname);
  } catch {
    return false;
  }
};

export const tagsRecordToArray = (tags: Record<string, string> = {}): { key: string; value: string }[] =>
  Object.entries(tags).map(([key, value]) => ({ key, value }));

export const findLatestEndpoint = (server: MCPServer): MCPAccessEndpoint | undefined =>
  server.latest_version
    ? (server.access_endpoints ?? []).find((e) => e.resolved_version?.version === server.latest_version)
    : server.access_endpoints?.[0];

export const isServerDimmed = (server: MCPServer): boolean => server.status !== MCPStatus.ACTIVE;

export const formatEndpointTarget = (endpoint: Pick<MCPAccessEndpoint, 'server_alias' | 'server_version'>): string =>
  endpoint.server_alias ? `@${endpoint.server_alias}` : endpoint.server_version || '—';

const hasAction = (actions: MCPServerAction[] | undefined, action: MCPServerAction) =>
  actions === undefined || actions.includes(action);

export const getServerPermissions = (server?: MCPServer) => {
  if (!server) {
    return { canUpdate: false, canDelete: false, canManage: false };
  }
  const actions = server.allowed_actions;
  return {
    canUpdate: hasAction(actions, MCPServerAction.UPDATE),
    canDelete: hasAction(actions, MCPServerAction.DELETE),
    canManage: hasAction(actions, MCPServerAction.MANAGE),
  };
};

interface ServerJsonValidationResult {
  valid: boolean;
  error?: string;
  parsed?: ServerJSONPayload;
}

export const validateServerJson = (value: string): ServerJsonValidationResult => {
  const trimmed = value?.trim();
  if (!trimmed) {
    return { valid: false, error: 'Server definition is required' };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return { valid: false, error: 'Invalid JSON format in server configuration' };
  }

  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    return { valid: false, error: 'Server configuration must be a JSON object' };
  }

  const obj = parsed as Record<string, unknown>;

  if (!obj['name'] || typeof obj['name'] !== 'string') {
    return { valid: false, error: 'Server configuration must include a "name" field' };
  }

  if (!obj['version'] || typeof obj['version'] !== 'string') {
    return { valid: false, error: 'Server configuration must include a "version" field' };
  }

  return { valid: true, parsed: obj as ServerJSONPayload };
};

export const validateToolsJson = (value: string): { valid: boolean; error?: string; parsed?: MCPTool[] } => {
  const trimmed = value?.trim();
  if (!trimmed) {
    return { valid: true };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return { valid: false, error: 'Invalid JSON format in tools configuration' };
  }

  if (!Array.isArray(parsed)) {
    return { valid: false, error: 'Tools must be a JSON array' };
  }

  for (let i = 0; i < parsed.length; i++) {
    const tool = parsed[i];
    if (typeof tool !== 'object' || tool === null || Array.isArray(tool)) {
      return { valid: false, error: `Tool at index ${i} must be a JSON object` };
    }
    if (!(tool as Record<string, unknown>)['name'] || typeof (tool as Record<string, unknown>)['name'] !== 'string') {
      return { valid: false, error: `Tool at index ${i} must have a "name" field` };
    }
  }

  return { valid: true, parsed: parsed as MCPTool[] };
};

export const buildPackageConnectOptionKey = (pkg: {
  registryType: string;
  identifier: string;
}): PackageConnectOptionKey => `pkg:${pkg.registryType}:${pkg.identifier}`;

export const buildRemoteConnectOptionKey = (remote: { url?: string; type: string }): RemoteConnectOptionKey =>
  `remote:${remote.type}:${remote.url ?? ''}`;

export const deriveConnectOptionKeys = (serverJson: ServerJSONPayload): Set<ConnectOptionKey> => {
  const keys = new Set<ConnectOptionKey>();
  for (const pkg of serverJson.packages ?? []) {
    keys.add(buildPackageConnectOptionKey(pkg));
  }
  for (const remote of serverJson.remotes ?? []) {
    keys.add(buildRemoteConnectOptionKey(remote));
  }
  return keys;
};
