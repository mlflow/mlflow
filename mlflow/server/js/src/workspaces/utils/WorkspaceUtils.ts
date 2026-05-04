import { useSyncExternalStore } from 'use-sync-external-store/shim';
import { getWorkspacesEnabledSync } from '../../experiment-tracking/hooks/useServerInfo';

const WORKSPACE_STORAGE_KEY = 'mlflow.activeWorkspace';
export const WORKSPACE_QUERY_PARAM = 'workspace';

let activeWorkspace: string | null = null;

// Subscribers for reactive consumption via ``useActiveWorkspace``. The
// in-memory active workspace is set both by ``WorkspaceRouterSync`` (URL
// changes) and by ``WorkspaceSelector`` (workspace switches on global
// routes that don't navigate). React components reading the value need
// to re-render on either path - a non-reactive ``getActiveWorkspace``
// call covers the URL-change case (re-render is triggered by the
// router), but not the in-place selector switch.
const activeWorkspaceListeners = new Set<() => void>();

const subscribeToActiveWorkspace = (listener: () => void): (() => void) => {
  activeWorkspaceListeners.add(listener);
  return () => {
    activeWorkspaceListeners.delete(listener);
  };
};

export const getActiveWorkspace = () => activeWorkspace;

export const setActiveWorkspace = (workspace: string | null) => {
  if (workspace === activeWorkspace) {
    return;
  }
  activeWorkspace = workspace;
  if (workspace) {
    setLastUsedWorkspace(workspace);
  }
  activeWorkspaceListeners.forEach((listener) => listener());
};

/**
 * Reactive read of the in-memory active workspace. Re-renders the
 * caller whenever ``setActiveWorkspace`` mutates the value, including
 * the global-route fast-path in ``WorkspaceSelector`` that doesn't do
 * a URL navigation.
 */
export const useActiveWorkspace = (): string | null =>
  useSyncExternalStore(subscribeToActiveWorkspace, getActiveWorkspace, getActiveWorkspace);

/**
 * Get the last used workspace from localStorage.
 * Used for UI hints like the "Last used" badge.
 */
export const getLastUsedWorkspace = (): string | null => {
  if (typeof window === 'undefined') {
    return null;
  }
  try {
    // eslint-disable-next-line @databricks/no-direct-storage -- OSS only use-case
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
        // eslint-disable-next-line @databricks/no-direct-storage -- OSS only use-case
        window.localStorage.setItem(WORKSPACE_STORAGE_KEY, workspace);
      } else {
        // eslint-disable-next-line @databricks/no-direct-storage -- OSS only use-case
        window.localStorage.removeItem(WORKSPACE_STORAGE_KEY);
      }
    } catch {
      // no-op: localStorage might be unavailable (e.g., private browsing)
    }
  }
};

// Workspace name validation constants (must match backend: mlflow/store/workspace/abstract_store.py)
const WORKSPACE_NAME_PATTERN = /^(?!.*--)[a-z0-9]([-a-z0-9]*[a-z0-9])?$/;
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

  if (!workspaceName || !validateWorkspaceName(workspaceName).valid) {
    return null;
  }

  return workspaceName;
};

const isAbsoluteUrl = (value: string) => /^[a-zA-Z][a-zA-Z\d+\-.]*:/.test(value);

/**
 * Routes that never have workspace context. Root '/' is contextual.
 * Kept as an extension point for any future workspace-agnostic routes.
 */
const ALWAYS_GLOBAL_ROUTES: string[] = ['/account'];

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

  if (isAbsoluteUrl(to)) {
    return to;
  }
  // Keep workspace prefixing when an active workspace is already known, even if
  // the async server feature flags have not resolved yet. This avoids dropping
  // workspace context during early navigation on initial load.
  if (!getWorkspacesEnabledSync() && !getActiveWorkspace()) {
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
  if (isAbsoluteUrl(pathname)) {
    return pathname;
  }
  if (!getWorkspacesEnabledSync() && !getActiveWorkspace()) {
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
