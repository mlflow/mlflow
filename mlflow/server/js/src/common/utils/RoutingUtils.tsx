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

import { prefixPathnameWithWorkspace, prefixRouteWithWorkspace } from './WorkspaceUtils';

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

const prefixRouteWithWorkspaceForTo = (to: To): To => {
  if (typeof to === 'string') {
    return prefixRouteWithWorkspace(to);
  }
  if (typeof to === 'object' && to !== null) {
    const pathname = 'pathname' in to ? to.pathname : undefined;
    if (typeof pathname === 'string') {
      return {
        ...to,
        pathname: prefixPathnameWithWorkspace(pathname),
      };
    }
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

export type { Location, NavigateFunction, Params, To, NavigateOptions, PathMatch };
