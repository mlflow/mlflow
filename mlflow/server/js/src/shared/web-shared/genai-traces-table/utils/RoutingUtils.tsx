/**
 * This file re-exports the appropriate package for routing based on the environment.
 *
 * Duplicate of mlflow/server/js/src/common/utils/RoutingUtils.tsx, because unfortunately
 * this package is located in web-shared and we can't directly access it from here.
 */
/* eslint-disable no-restricted-imports */
import type { ComponentProps } from 'react';
import React, { useCallback } from 'react';
/**
 * Import React Router V6 parts
 */
import {
  BrowserRouter,
  MemoryRouter,
  HashRouter,
  matchPath,
  generatePath,
  Navigate,
  Route,
  NavLink as NavLinkDirect,
  Outlet as OutletDirect,
  Link as LinkDirect,
  useNavigate as useNavigateDirect,
  useLocation as useLocationDirect,
  useParams as useParamsDirect,
  useSearchParams as useSearchParamsDirect,
  createHashRouter,
  RouterProvider,
  Routes,
  type To,
  type NavigateOptions,
  type Location,
  type NavigateFunction,
  type Params,
} from 'react-router-dom';

/**
 * Workspace utilities — minimal self-contained copies for web-shared.
 * These mirror the functions in mlflow/server/js/src/workspaces/utils/WorkspaceUtils.ts
 * and mlflow/server/js/src/common/utils/ServerFeaturesContext.tsx but cannot be imported
 * directly due to the web-shared package boundary.
 */

const WORKSPACE_STORAGE_KEY = 'mlflow.activeWorkspace';
const WORKSPACE_QUERY_PARAM = 'workspace';

/**
 * Get the currently active workspace from localStorage.
 * Mirrors getActiveWorkspace() from WorkspaceUtils.ts — uses the same storage key
 * that the main app writes to, so it reads the correct value at runtime.
 */
const getActiveWorkspace = (): string | null => {
  if (typeof window === 'undefined') {
    return null;
  }
  try {
    return window.localStorage.getItem(WORKSPACE_STORAGE_KEY);
  } catch {
    return null;
  }
};

const isAbsoluteUrl = (value: string) => /^[a-zA-Z][a-zA-Z\d+\-.]*:/.test(value);

/** Routes that never have workspace context (e.g., /settings). Root '/' is contextual. */
const ALWAYS_GLOBAL_ROUTES = ['/settings'];

/** Check if pathname is always global (workspace-agnostic). */
const isGlobalRoute = (pathname: string): boolean => {
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
const removeWorkspaceQueryParam = (url: string): string => {
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
const prefixRouteWithWorkspace = (to: string): string => {
  if (typeof to !== 'string' || to.length === 0) {
    return to;
  }

  if (isAbsoluteUrl(to)) {
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

    if (typeof pathname !== 'string') {
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

const Outlet = OutletDirect;

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
  generatePath,
  matchPath,
  Navigate,
  Route,
  Routes,
  Outlet,

  // Exports used to build hash-based data router
  createHashRouter,
  RouterProvider,
};

export const createLazyRouteElement = (
  // Load the module's default export and turn it into React Element
  componentLoader: () => Promise<{ default: React.ComponentType<React.PropsWithChildren<any>> }>,
) => React.createElement(React.lazy(componentLoader));
export const createRouteElement = (component: React.ComponentType<React.PropsWithChildren<any>>) =>
  React.createElement(component);

export type { Location, NavigateFunction, Params, To, NavigateOptions };
