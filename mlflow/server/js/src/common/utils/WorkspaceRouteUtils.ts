import { matchPath, type PathMatch } from './RoutingUtils';

const WORKSPACE_ROUTE_PREFIX = '/workspaces/:workspaceName';

const normalizeRoutePath = (path: string) => {
  if (path === '*') {
    return '/workspaces/:workspaceName/*';
  }
  if (path === '/') {
    return WORKSPACE_ROUTE_PREFIX;
  }
  return path.startsWith('/') ? path : `/${path}`;
};

export const prefixRoutePathWithWorkspace = (path?: string) => {
  if (!path) {
    return path;
  }

  const normalized = normalizeRoutePath(path);
  if (normalized === WORKSPACE_ROUTE_PREFIX) {
    return WORKSPACE_ROUTE_PREFIX;
  }

  if (normalized.startsWith(WORKSPACE_ROUTE_PREFIX)) {
    return normalized;
  }

  return `${WORKSPACE_ROUTE_PREFIX}${normalized}`;
};

export const matchPathWithWorkspace = <ParamKey extends string = string>(
  routePath: string,
  pathname: string,
): PathMatch<ParamKey> | null => {
  const directMatch = matchPath<ParamKey, string>(routePath, pathname);
  if (directMatch) {
    return directMatch;
  }

  const workspaceAwareRoutePath = prefixRoutePathWithWorkspace(routePath);
  if (!workspaceAwareRoutePath || workspaceAwareRoutePath === routePath) {
    return null;
  }

  return matchPath<ParamKey, string>(workspaceAwareRoutePath, pathname);
};
