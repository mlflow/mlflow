import { getWorkspacesEnabledSync } from '../../experiment-tracking/hooks/useServerInfo';

const WORKSPACE_STORAGE_KEY = 'mlflow.activeWorkspace';
export const WORKSPACE_QUERY_PARAM = 'workspace';

let activeWorkspace: string | null = null;

export const getActiveWorkspace = () => activeWorkspace;

export const setActiveWorkspace = (workspace: string | null) => {
  activeWorkspace = workspace;
  if (workspace) {
    setLastUsedWorkspace(workspace);
  }
};

/**
 * Get the last used workspace from localStorage.
 * Used for UI hints like the "Last used" badge.
 */
export const getLastUsedWorkspace = (): string | null => {
  if (typeof window === 'undefined') {
    return null;
  }
  try {
    return window.localStorage.getItem(WORKSPACE_STORAGE_KEY);
  } catch {
    return null;
  }
};

/**
 * Set the active workspace in localStorage and notify listeners.
 * Used for syncing state with URL and for UI hints via getLastUsedWorkspace().
 */
export const setLastUsedWorkspace = (workspace: string | null) => {
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
};

// Workspace name validation constants (must match backend: mlflow/store/workspace/abstract_store.py)
export const WORKSPACE_NAME_PATTERN = /^(?!.*--)[a-z0-9]([-a-z0-9]*[a-z0-9])?$/;
export const WORKSPACE_NAME_MIN_LENGTH = 2;
export const WORKSPACE_NAME_MAX_LENGTH = 63;

export type WorkspaceValidationResult = {
  valid: boolean;
  error?: string;
};

/** Validates workspace name against backend rules. Reserved names checked by API. */
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

/** Extract and validate workspace from URL search params. */
export const extractWorkspaceFromSearchParams = (search: string | URLSearchParams): string | null => {
  const params = typeof search === 'string' ? new URLSearchParams(search) : search;
  const workspaceName = params.get(WORKSPACE_QUERY_PARAM);

  if (!workspaceName || !WORKSPACE_NAME_PATTERN.test(workspaceName)) {
    return null;
  }

  return workspaceName;
};

const isAbsoluteUrl = (value: string) => /^[a-zA-Z][a-zA-Z\d+\-.]*:/.test(value);

/**
 * Routes that never have workspace context. Root '/' is contextual.
 * Kept as an extension point for any future workspace-agnostic routes.
 */
const ALWAYS_GLOBAL_ROUTES: string[] = [];

/** Check if pathname is always global (workspace-agnostic). */
export const isGlobalRoute = (pathname: string): boolean => {
  const normalizedPath = pathname.split('?')[0].split('#')[0];
  return ALWAYS_GLOBAL_ROUTES.some((route) => normalizedPath === route || normalizedPath.startsWith(route + '/'));
};

/** Add workspace query param to URL, preserving existing params and hash. */
const addWorkspaceQueryParam = (url: string, workspace: string): string => {
  const hashPrefix = url.startsWith('#') ? '#' : '';
  const urlWithoutHashPrefix = hashPrefix ? url.slice(1) : url;

  let pathname = urlWithoutHashPrefix;
  let existingQuery = '';
  let hashFragment = '';

  const hashIndex = pathname.indexOf('#');
  if (hashIndex >= 0) {
    hashFragment = pathname.slice(hashIndex);
    pathname = pathname.slice(0, hashIndex);
  }

  const queryIndex = pathname.indexOf('?');
  if (queryIndex >= 0) {
    existingQuery = pathname.slice(queryIndex + 1);
    pathname = pathname.slice(0, queryIndex);
  }

  const params = new URLSearchParams(existingQuery);
  params.set(WORKSPACE_QUERY_PARAM, workspace);

  return `${hashPrefix}${pathname}?${params.toString()}${hashFragment}`;
};

/** Remove workspace query param from URL. */
export const removeWorkspaceQueryParam = (url: string): string => {
  const hashPrefix = url.startsWith('#') ? '#' : '';
  const urlWithoutHashPrefix = hashPrefix ? url.slice(1) : url;

  let pathname = urlWithoutHashPrefix;
  let existingQuery = '';
  let hashFragment = '';

  const hashIndex = pathname.indexOf('#');
  if (hashIndex >= 0) {
    hashFragment = pathname.slice(hashIndex);
    pathname = pathname.slice(0, hashIndex);
  }

  const queryIndex = pathname.indexOf('?');
  if (queryIndex >= 0) {
    existingQuery = pathname.slice(queryIndex + 1);
    pathname = pathname.slice(0, queryIndex);
  }

  const params = new URLSearchParams(existingQuery);
  params.delete(WORKSPACE_QUERY_PARAM);

  const queryString = params.toString();
  return `${hashPrefix}${pathname}${queryString ? '?' + queryString : ''}${hashFragment}`;
};

/**
 * Prefix route with workspace query param. Removes workspace from global routes.
 * Relative paths unchanged. Root '/' is contextual based on workspace param.
 */
export const prefixRouteWithWorkspace = (to: string): string => {
  if (typeof to !== 'string' || to.length === 0) {
    return to;
  }

  if (!getWorkspacesEnabledSync() || isAbsoluteUrl(to)) {
    return to;
  }

  const hashPrefix = to.startsWith('#') ? '#' : '';
  const urlWithoutHashPrefix = hashPrefix ? to.slice(1) : to;

  const isAbsoluteNavigation = hashPrefix !== '' || urlWithoutHashPrefix.startsWith('/');
  if (!isAbsoluteNavigation) {
    return to;
  }

  let pathname = urlWithoutHashPrefix;
  let existingQuery = '';
  const hashIndex = pathname.indexOf('#');
  if (hashIndex >= 0) {
    pathname = pathname.slice(0, hashIndex);
  }
  const queryIndex = pathname.indexOf('?');
  if (queryIndex >= 0) {
    existingQuery = pathname.slice(queryIndex + 1);
    pathname = pathname.slice(0, queryIndex);
  }

  const existingParams = new URLSearchParams(existingQuery);
  const hasExplicitWorkspace = existingParams.has(WORKSPACE_QUERY_PARAM);

  if (isGlobalRoute(pathname || '/')) {
    return removeWorkspaceQueryParam(to);
  }

  if (hasExplicitWorkspace) {
    return to;
  }

  const workspace = getActiveWorkspace();
  if (!workspace) {
    return to;
  }

  return addWorkspaceQueryParam(to, workspace);
};

/** Prefix pathname with workspace query param. Similar to prefixRouteWithWorkspace. */
export const appendWorkspaceSearchParams = (pathname: string | undefined): string | undefined => {
  if (!pathname) {
    return pathname;
  }
  if (!getWorkspacesEnabledSync() || isAbsoluteUrl(pathname)) {
    return pathname;
  }

  if (isGlobalRoute(pathname)) {
    return pathname;
  }

  const workspace = getActiveWorkspace();
  if (!workspace) {
    return pathname;
  }

  return `${pathname}?${WORKSPACE_QUERY_PARAM}=${encodeURIComponent(workspace)}`;
};
