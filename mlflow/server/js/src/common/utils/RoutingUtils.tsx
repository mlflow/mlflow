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
  Navigate,
  Route,
  UNSAFE_NavigationContext,
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

const useNavigate = (): NavigateFunction => {
  const navigate = useNavigateDirect();
  const wrappedNavigate = useCallback(
    (to: To | number, options?: NavigateOptions) => {
      if (typeof to === 'number') {
        navigate(to);
        return;
      }
      navigate(prefixRouteWithWorkspaceForTo(to), options);
    },
    [navigate],
  );

  return wrappedNavigate as NavigateFunction;
};

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

const Link = React.forwardRef<HTMLAnchorElement, ComponentProps<typeof LinkDirect>>((props, ref) => {
  const { to, ...rest } = props;
  return <LinkDirect ref={ref} to={prefixRouteWithWorkspaceForTo(to)} {...rest} />;
});

Link.displayName = 'Link';

const NavLink = React.forwardRef<HTMLAnchorElement, ComponentProps<typeof NavLinkDirect>>((props, ref) => {
  const { to, ...rest } = props;
  return <NavLinkDirect ref={ref} to={prefixRouteWithWorkspaceForTo(to)} {...rest} />;
});

NavLink.displayName = 'NavLink';

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
  generatePath,
  matchPath,
  Navigate,
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

export type { Location, NavigateFunction, Params, To, NavigateOptions, PathMatch };
