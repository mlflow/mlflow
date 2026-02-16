/**
 * This file is the only one that should directly import from 'react-router-dom' module
 */
/* eslint-disable no-restricted-imports */

/**
 * Import React Router V6 parts
 */
import {
  BrowserRouter,
  MemoryRouter,
  HashRouter,
  matchPath,
  generatePath,
  Route,
  UNSAFE_NavigationContext,
  NavLink as NavLinkDirect,
  Outlet as OutletDirect,
  Link as LinkDirect,
  useNavigate as useNavigateDirect,
  useLocation as useLocationDirect,
  useParams as useParamsDirect,
  useSearchParams as useSearchParamsDirect,
  useMatches as useMatchesDirect,
  createHashRouter,
  RouterProvider,
  Routes,
  type To,
  type NavigateOptions,
  type Location,
  type NavigateFunction,
  type Params,
  type PathMatch,
} from 'react-router-dom';

/**
 * Import React Router V5 parts
 */
import { HashRouter as HashRouterV5, Link as LinkV5, NavLink as NavLinkV5 } from 'react-router-dom';
import type { ComponentProps } from 'react';
import React, { useCallback } from 'react';

import {
  prefixRouteWithWorkspace,
  getActiveWorkspace,
  isGlobalRoute,
  WORKSPACE_QUERY_PARAM,
} from '../../workspaces/utils/WorkspaceUtils';
import { getWorkspacesEnabledSync } from '../../experiment-tracking/hooks/useServerInfo';

const useLocation = useLocationDirect;

const useSearchParams = useSearchParamsDirect;

const useParams = useParamsDirect;

type UseNavigateOptions = {
  bypassWorkspacePrefix?: boolean;
};

const useNavigate = (options: UseNavigateOptions = { bypassWorkspacePrefix: false }): NavigateFunction => {
  const { bypassWorkspacePrefix } = options;
  const navigate = useNavigateDirect();
  const wrappedNavigate = useCallback(
    (to: To | number, options?: NavigateOptions) => {
      if (typeof to === 'number') {
        navigate(to);
        return;
      }
      navigate(bypassWorkspacePrefix ? to : prefixRouteWithWorkspaceForTo(to), options);
    },
    [navigate, bypassWorkspacePrefix],
  );

  return wrappedNavigate as NavigateFunction;
};

const useMatches = useMatchesDirect;

const Outlet = OutletDirect;

/**
 * Add workspace query param to a To object (used by Link/NavLink/navigate).
 * Handles both string and object forms of To.
 */
const prefixRouteWithWorkspaceForTo = (to: To): To => {
  // String form - delegate to prefixRouteWithWorkspace
  if (typeof to === 'string') {
    return prefixRouteWithWorkspace(to);
  }

  // Object form - need to handle pathname and search separately
  if (typeof to === 'object' && to !== null) {
    const pathname = 'pathname' in to ? to.pathname : undefined;

    // Skip if workspaces not enabled or pathname not provided
    if (!getWorkspacesEnabledSync() || typeof pathname !== 'string') {
      return to;
    }

    // Skip workspace param for global routes
    if (isGlobalRoute(pathname)) {
      // Remove workspace param if present in existing search
      if (to.search) {
        const params = new URLSearchParams(to.search);
        params.delete(WORKSPACE_QUERY_PARAM);
        const newSearch = params.toString();
        return {
          ...to,
          search: newSearch ? `?${newSearch}` : undefined,
        };
      }
      return to;
    }

    // Add workspace query param
    const workspace = getActiveWorkspace();
    if (!workspace) {
      return to;
    }

    // Merge workspace into existing search params
    const existingSearch = to.search || '';
    const params = new URLSearchParams(existingSearch.startsWith('?') ? existingSearch.slice(1) : existingSearch);
    params.set(WORKSPACE_QUERY_PARAM, workspace);

    return {
      ...to,
      search: `?${params.toString()}`,
    };
  }

  return to;
};

const Link = React.forwardRef<
  HTMLAnchorElement,
  ComponentProps<typeof LinkDirect> & { disableWorkspacePrefix?: boolean }
>(function Link(props, ref) {
  const { to, disableWorkspacePrefix, ...rest } = props;
  const finalTo = disableWorkspacePrefix ? to : prefixRouteWithWorkspaceForTo(to);
  return <LinkDirect ref={ref} to={finalTo} {...rest} />;
});

const NavLink = React.forwardRef<
  HTMLAnchorElement,
  ComponentProps<typeof NavLinkDirect> & { disableWorkspacePrefix?: boolean }
>(function NavLink(props, ref) {
  const { to, disableWorkspacePrefix, ...rest } = props;
  const finalTo = disableWorkspacePrefix ? to : prefixRouteWithWorkspaceForTo(to);
  return <NavLinkDirect ref={ref} to={finalTo} {...rest} />;
});

export const createMLflowRoutePath = (routePath: string) => {
  return routePath;
};

export {
  // React Router V6 API exports
  BrowserRouter,
  MemoryRouter,
  HashRouter,
  Link,
  NavLink,
  useNavigate,
  useLocation,
  useParams,
  useSearchParams,
  useMatches,
  generatePath,
  matchPath,
  Route,
  Routes,
  Outlet,

  // Exports used to build hash-based data router
  createHashRouter,
  RouterProvider,

  // Unsafe navigation context, will be improved after full migration to react-router v6
  UNSAFE_NavigationContext,
};

export const createLazyRouteElement = (
  // Load the module's default export and turn it into React Element
  componentLoader: () => Promise<{ default: React.ComponentType<React.PropsWithChildren<any>> }>,
) => React.createElement(React.lazy(componentLoader));
export const createRouteElement = (component: React.ComponentType<React.PropsWithChildren<any>>) =>
  React.createElement(component);

/**
 * Handle for route definitions that can be used to set the document title.
 */
export interface DocumentTitleHandle {
  getPageTitle: (params: Params<string>) => string;
}

/**
 * Handle for route definitions that provides context-aware assistant prompts.
 */
export interface AssistantPromptsHandle {
  getAssistantPrompts?: () => string[];
}

/**
 * Combined route handle interface for routes that support both page titles and assistant prompts.
 */
export type RouteHandle = DocumentTitleHandle & AssistantPromptsHandle;

export const usePageTitle = () => {
  const matches = useMatches();
  if (matches.length === 0) {
    return;
  }

  const lastMatch = matches[matches.length - 1];
  const handle = lastMatch.handle as DocumentTitleHandle | undefined;
  const title = handle?.getPageTitle(lastMatch.params);

  return title;
};

export const DEFAULT_ASSISTANT_PROMPTS = [
  'How do I get started with MLflow?',
  'What should I know about MLflow?',
  'Explain how MLflow Tracing works.',
];

/**
 * Hook to get context-aware assistant prompts based on the current route.
 * Falls back to default prompts if the route doesn't define any.
 */
export const useAssistantPrompts = (): string[] => {
  const matches = useMatches();

  // Find the most specific route that defines assistant prompts (search from end)
  for (let i = matches.length - 1; i >= 0; i--) {
    const handle = matches[i].handle as AssistantPromptsHandle | undefined;
    if (handle?.getAssistantPrompts) {
      return handle.getAssistantPrompts();
    }
  }

  return DEFAULT_ASSISTANT_PROMPTS;
};

export type { Location, NavigateFunction, Params, To, NavigateOptions, PathMatch };
