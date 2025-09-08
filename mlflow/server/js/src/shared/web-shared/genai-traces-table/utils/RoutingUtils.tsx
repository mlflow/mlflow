/**
 * This file re-exports the appropriate package for routing based on the environment.
 *
 * Duplicate of mlflow/web/js/src/common/utils/RoutingUtils.tsx, because unfortunately
 * this package is located in web-shared and we can't directly access it from here.
 */
/* eslint-disable no-restricted-imports */
import React from 'react';
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

const useLocation = useLocationDirect;

const useSearchParams = useSearchParamsDirect;

const useParams = useParamsDirect;

const useNavigate = useNavigateDirect;

const Outlet = OutletDirect;

const Link = LinkDirect;

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
