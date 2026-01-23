import { getWorkspacesEnabledSync } from './ServerFeaturesContext';

const WORKSPACE_STORAGE_KEY = 'mlflow.activeWorkspace';

const getStoredWorkspace = () => {
  if (typeof window === 'undefined') {
    return null;
  }
  try {
    return window.localStorage.getItem(WORKSPACE_STORAGE_KEY);
  } catch {
    return null;
  }
};

let activeWorkspace: string | null = getStoredWorkspace();
let availableWorkspaces: string[] = [];

const WORKSPACE_PREFIX = '/workspaces/';

const listeners = new Set<(workspace: string | null) => void>();

/**
 * Get the currently active workspace name.
 * Returns null if workspaces feature is not enabled or no workspace is selected.
 */
export const getActiveWorkspace = () => {
  // Only return the active workspace if the workspaces feature is enabled
  if (!getWorkspacesEnabledSync()) {
    return null;
  }
  return activeWorkspace;
};

export const setActiveWorkspace = (workspace: string | null) => {
  activeWorkspace = workspace;
  if (typeof window !== 'undefined') {
    try {
      if (workspace) {
        window.localStorage.setItem(WORKSPACE_STORAGE_KEY, workspace);
      } else {
        window.localStorage.removeItem(WORKSPACE_STORAGE_KEY);
      }
    } catch {
      // no-op: localStorage might be unavailable (e.g., private browsing)
    }
  }
  listeners.forEach((listener) => listener(activeWorkspace));
};

// Workspace name validation constants (must match backend: mlflow/store/workspace/abstract_store.py)
export const WORKSPACE_NAME_PATTERN = /^(?!.*--)[a-z0-9]([-a-z0-9]*[a-z0-9])?$/;
export const WORKSPACE_NAME_MIN_LENGTH = 2;
export const WORKSPACE_NAME_MAX_LENGTH = 63;

export type WorkspaceValidationResult = {
  valid: boolean;
  error?: string;
};

/**
 * Validates a workspace name against backend rules.
 * Returns { valid: true } if valid, or { valid: false, error: "message" } if invalid.
 * Note: Reserved names are validated by the API server.
 */
export const validateWorkspaceName = (name: string): WorkspaceValidationResult => {
  if (typeof name !== 'string') {
    return { valid: false, error: 'Workspace name must be a string.' };
  }

  if (name.length < WORKSPACE_NAME_MIN_LENGTH || name.length > WORKSPACE_NAME_MAX_LENGTH) {
    return {
      valid: false,
      error: `Workspace name must be between ${WORKSPACE_NAME_MIN_LENGTH} and ${WORKSPACE_NAME_MAX_LENGTH} characters.`,
    };
  }

  if (!WORKSPACE_NAME_PATTERN.test(name)) {
    return {
      valid: false,
      error: 'Workspace name must be lowercase alphanumeric with optional single hyphens (no consecutive hyphens).',
    };
  }

  return { valid: true };
};

export const extractWorkspaceFromPathname = (pathname: string): string | null => {
  if (!pathname || !pathname.startsWith(WORKSPACE_PREFIX)) {
    return null;
  }
  const segments = pathname.split('/');
  if (segments.length < 3 || !segments[2]) {
    return null;
  }

  let workspaceName: string;
  try {
    workspaceName = decodeURIComponent(segments[2]);
  } catch {
    // Malformed percent-encoding in URL - treat as no workspace
    return null;
  }

  // Validate workspace name format
  if (!WORKSPACE_NAME_PATTERN.test(workspaceName)) {
    return null;
  }

  return workspaceName;
};

export const subscribeToWorkspaceChanges = (listener: (workspace: string | null) => void) => {
  listeners.add(listener);
  listener(activeWorkspace);
  return () => {
    listeners.delete(listener);
  };
};

export const setAvailableWorkspaces = (workspaces: string[]) => {
  availableWorkspaces = workspaces;
};

export const getAvailableWorkspaces = () => availableWorkspaces;

const isAbsoluteUrl = (value: string) => /^[a-zA-Z][a-zA-Z\d+\-.]*:/.test(value);

/**
 * Helper to build a workspace path segment: /workspaces/{encoded-workspace-name}
 */
const buildWorkspacePath = (workspace: string) => `/workspaces/${encodeURIComponent(workspace)}`;

/**
 * Core logic to prefix a normalized pathname with the active workspace.
 * Returns the pathname unchanged if already a workspace path or no workspace is active.
 */
const applyWorkspacePrefix = (pathname: string): string => {
  // Don't prefix if already a workspace path or navigating to the workspace landing page
  if (pathname.startsWith('/workspaces/') || pathname === '/workspaces') {
    return pathname;
  }

  const workspace = getActiveWorkspace();
  if (!workspace) {
    return pathname;
  }

  // Root path becomes just the workspace path
  if (pathname === '/' || pathname === '') {
    return buildWorkspacePath(workspace);
  }

  return `${buildWorkspacePath(workspace)}${pathname}`;
};

export const prefixRouteWithWorkspace = (to: string) => {
  if (typeof to !== 'string' || to.length === 0) {
    return to;
  }

  if (!getWorkspacesEnabledSync() || isAbsoluteUrl(to)) {
    return to;
  }

  const prefix = to.startsWith('#') ? '#' : '';
  const valueWithoutPrefix = prefix ? to.slice(1) : to;

  const isAbsoluteNavigation = prefix !== '' || valueWithoutPrefix.startsWith('/');
  if (!isAbsoluteNavigation) {
    return to;
  }

  if (!valueWithoutPrefix) {
    const workspace = getActiveWorkspace();
    if (!workspace) {
      return to;
    }
    return `${prefix}${buildWorkspacePath(workspace)}`;
  }

  // Separate hash fragment (if any)
  let pathWithQuery = valueWithoutPrefix;
  let hashFragment = '';
  const hashIndex = pathWithQuery.indexOf('#');
  if (hashIndex >= 0) {
    hashFragment = pathWithQuery.slice(hashIndex);
    pathWithQuery = pathWithQuery.slice(0, hashIndex);
  }

  // Separate query string (if any)
  let queryString = '';
  const queryIndex = pathWithQuery.indexOf('?');
  if (queryIndex >= 0) {
    queryString = pathWithQuery.slice(queryIndex);
    pathWithQuery = pathWithQuery.slice(0, queryIndex);
  }

  // Inline sanitize: ensure path starts with /
  const normalizedPath = pathWithQuery ? (pathWithQuery.startsWith('/') ? pathWithQuery : `/${pathWithQuery}`) : '';

  // Apply workspace prefix to the normalized path
  const finalPath = applyWorkspacePrefix(normalizedPath);

  return `${prefix}${finalPath}${queryString}${hashFragment}`;
};

export const prefixPathnameWithWorkspace = (pathname: string | undefined) => {
  if (!pathname) {
    const workspace = getActiveWorkspace();
    return workspace ? buildWorkspacePath(workspace) : pathname;
  }
  if (!getWorkspacesEnabledSync() || isAbsoluteUrl(pathname) || !pathname.startsWith('/')) {
    return pathname;
  }
  // Inline sanitize: ensure path starts with /
  const sanitized = pathname.startsWith('/') ? pathname : `/${pathname}`;

  // Apply workspace prefix using shared logic
  return applyWorkspacePrefix(sanitized);
};
