import { shouldEnableWorkspaces as importedShouldEnableWorkspaces } from './FeatureUtils';

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

export const DEFAULT_WORKSPACE_NAME = 'default';

const listeners = new Set<(workspace: string | null) => void>();

export const getActiveWorkspace = () => activeWorkspace;

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

export const extractWorkspaceFromPathname = (pathname: string): string | null => {
  if (!pathname || !pathname.startsWith(WORKSPACE_PREFIX)) {
    return null;
  }
  const segments = pathname.split('/');
  if (segments.length < 3 || !segments[2]) {
    return null;
  }
  return decodeURIComponent(segments[2]);
};

export const buildWorkspacePath = (workspace: string) => `workspaces/${encodeURIComponent(workspace)}`;

export const subscribeToWorkspaceChanges = (listener: (workspace: string | null) => void) => {
  listeners.add(listener);
  listener(activeWorkspace);
  return () => {
    listeners.delete(listener);
  };
};

export const getCurrentWorkspace = () => activeWorkspace;

export const setAvailableWorkspaces = (workspaces: string[]) => {
  availableWorkspaces = workspaces;
};

export const getAvailableWorkspaces = () => availableWorkspaces;

export const hasWorkspaceAccess = (workspace: string | null): boolean => {
  if (!workspace) return true;
  if (availableWorkspaces.length === 0) return true; // Not loaded yet, assume access
  return availableWorkspaces.includes(workspace);
};

const isAbsoluteUrl = (value: string) => /^[a-zA-Z][a-zA-Z\d+\-.]*:/.test(value);

const isWorkspacesFeatureEnabled = () => {
  if (typeof importedShouldEnableWorkspaces === 'function') {
    return importedShouldEnableWorkspaces();
  }
  return true;
};

const sanitizePath = (path: string) => {
  if (!path) {
    return '';
  }
  return path.startsWith('/') ? path : `/${path}`;
};

export const getWorkspaceOrDefault = () => {
  if (!isWorkspacesFeatureEnabled()) {
    return null;
  }
  // When navigating directly to a workspace-specific route, the router might render
  // before `setActiveWorkspace` is called. In that case, fall back to the workspace
  // encoded in the current pathname so links remain workspace-aware on first render.
  const workspaceFromPath =
    typeof window !== 'undefined' ? extractWorkspaceFromPathname(window.location.pathname) : null;
  return activeWorkspace || workspaceFromPath || DEFAULT_WORKSPACE_NAME;
};

export const prefixRouteWithWorkspace = (to: string) => {
  if (typeof to !== 'string' || to.length === 0) {
    return to;
  }

  if (!isWorkspacesFeatureEnabled() || isAbsoluteUrl(to)) {
    return to;
  }

  const prefix = to.startsWith('#') ? '#' : '';
  const valueWithoutPrefix = prefix ? to.slice(1) : to;

  const isAbsoluteNavigation = prefix !== '' || valueWithoutPrefix.startsWith('/');
  if (!isAbsoluteNavigation) {
    return to;
  }

  if (!valueWithoutPrefix) {
    const workspace = getWorkspaceOrDefault();
    if (!workspace) {
      return to;
    }
    return `${prefix}/workspaces/${encodeURIComponent(workspace)}`;
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

  const normalizedPath = sanitizePath(pathWithQuery);

  if (normalizedPath.startsWith('/workspaces/')) {
    return `${prefix}${normalizedPath}${queryString}${hashFragment}`;
  }

  const workspace = getWorkspaceOrDefault();
  if (!workspace) {
    return `${prefix}${normalizedPath}${queryString}${hashFragment}`;
  }

  const workspacePath = `/workspaces/${encodeURIComponent(workspace)}`;
  const finalPath = normalizedPath === '/' ? workspacePath : `${workspacePath}${normalizedPath}`;

  return `${prefix}${finalPath}${queryString}${hashFragment}`;
};

export const prefixPathnameWithWorkspace = (pathname: string | undefined) => {
  if (!pathname) {
    const workspace = getWorkspaceOrDefault();
    return workspace ? `/workspaces/${encodeURIComponent(workspace)}` : pathname;
  }
  if (!isWorkspacesFeatureEnabled() || isAbsoluteUrl(pathname) || !pathname.startsWith('/')) {
    return pathname;
  }
  const sanitized = sanitizePath(pathname);
  if (sanitized.startsWith('/workspaces/')) {
    return sanitized;
  }
  const workspace = getWorkspaceOrDefault();
  if (!workspace) {
    return sanitized;
  }
  if (sanitized === '/' || sanitized === '') {
    return `/workspaces/${encodeURIComponent(workspace)}`;
  }
  return `/workspaces/${encodeURIComponent(workspace)}${sanitized}`;
};
