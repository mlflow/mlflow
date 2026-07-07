import type { TagProps } from '@databricks/design-system';
import type { MCPStatus, MCPTool, ServerJSONPayload } from './types';

export const STATUS_TAG_COLOR: Record<MCPStatus, TagProps['color']> = {
  draft: 'charcoal',
  active: 'lime',
  deprecated: 'lemon',
  deleted: 'coral',
};

export const STATUS_TRANSITIONS: Record<MCPStatus, MCPStatus[]> = {
  draft: ['active'],
  active: ['draft', 'deprecated'],
  deprecated: ['active'],
  deleted: [],
};

export const LATEST_ALIAS = 'latest';

export const RESERVED_ALIASES = [LATEST_ALIAS];

export const emptyCenterStyles = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  height: '100%',
  minHeight: 400,
  width: '100%',
  '& > div': {
    height: '100%',
    display: 'flex',
    flexDirection: 'column' as const,
    justifyContent: 'center',
    alignItems: 'center',
  },
};

export const MCP_QUERY_KEYS = {
  SERVERS_LIST: 'mcp_servers_list',
  SERVER: 'mcp_server',
  SERVER_VERSIONS: 'mcp_server_versions',
  SERVER_LATEST_VERSION: 'mcp_server_latest_version',
} as const;

export const DEFAULT_PAGE_SIZE = 25;
export const PAGE_SIZE_OPTIONS = [10, 25, 50, 100];

export const resolveDisplayName = (server: { display_name?: string; name: string }): string => {
  return server.display_name || server.name;
};

export const tagsRecordToArray = (tags: Record<string, string> = {}): { key: string; value: string }[] =>
  Object.entries(tags).map(([key, value]) => ({ key, value }));

export const resolveVersionDisplayName = (
  version: { display_name?: string; server_json?: { title?: string } } | null | undefined,
  fallback: string,
): string => {
  return version?.display_name || version?.server_json?.title || fallback;
};

export const buildSearchFilterClause = (searchFilter: string | undefined, field: string): string | undefined => {
  if (!searchFilter) {
    return undefined;
  }
  const sqlKeywordPattern = /(\s+(ILIKE|LIKE|IN|IS)\s+)|=|!=|<=|>=|<|>/i;
  if (sqlKeywordPattern.test(searchFilter)) {
    return searchFilter;
  }
  return `${field} LIKE '%${searchFilter.replace(/'/g, "''")}%'`;
};

export interface ServerJsonValidationResult {
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
