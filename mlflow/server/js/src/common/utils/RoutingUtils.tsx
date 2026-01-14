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
  NavLink,
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
} from 'react-router-dom';

/**
 * Import React Router V5 parts
 */
import { HashRouter as HashRouterV5, Link as LinkV5, NavLink as NavLinkV5 } from 'react-router-dom';
import type { ComponentProps } from 'react';
import React from 'react';

const useLocation = useLocationDirect;

const useSearchParams = useSearchParamsDirect;

const useParams = useParamsDirect;

const useNavigate = useNavigateDirect;

const useMatches = useMatchesDirect;

const Outlet = OutletDirect;

const Link = LinkDirect;

export const createMLflowRoutePath = (routePath: string) => {
  return routePath;
};

export {
  // React Router V6 API exports
  BrowserRouter,
  MemoryRouter,
  HashRouter,
  Link,
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

export const useGetRouteTitle = () => {
  const matches = useMatches();
  if (matches.length === 0) {
    return;
  }

  const lastMatch = matches[matches.length - 1];
  const handle = lastMatch.handle as DocumentTitleHandle | undefined;
  const title = handle?.getPageTitle(lastMatch.params);

  return title;
};

export type { Location, NavigateFunction, Params, To, NavigateOptions };
