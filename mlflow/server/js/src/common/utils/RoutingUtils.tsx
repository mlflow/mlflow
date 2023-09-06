/**
 * shouldUsePathRouting() evaluation is stable so we omit
 * `react-hooks/rules-of-hooks` rule for hooks in this file
 */
/* eslint-disable react-hooks/rules-of-hooks */

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
  StaticRouter,
  HashRouter,
  CompatRouter,
  matchPath,
  Navigate,
  Route,
  UNSAFE_NavigationContext,
  NavLink,
  Link as LinkDirect,
  useNavigate as useNavigateDirect,
  useLocation as useLocationDirect,
  useParams as useParamsDirect,
  Routes,
  type To,
  type NavigateOptions,
  type Location,
  type NavigateFunction,
  type Params,
} from 'react-router-dom-v5-compat';

/**
 * Import React Router V5 parts
 */
import { HashRouter as HashRouterV5, Link as LinkV5, NavLink as NavLinkV5 } from 'react-router-dom';
import { shouldUsePathRouting } from './FeatureUtils';
import React, { ComponentProps } from 'react';

const useLocation = useLocationDirect;

const useParams = useParamsDirect;

const useNavigate = useNavigateDirect;

const Link = LinkDirect;

export const createMLflowRoutePath = (routePath: string) => {
  return routePath;
};

export {
  // React Router V6 API exports
  BrowserRouter,
  MemoryRouter,
  StaticRouter,
  HashRouter,
  CompatRouter,
  Link,
  NavLink,
  useNavigate,
  useLocation,
  useParams,
  matchPath,
  Navigate,
  Route,
  Routes,

  // Unsafe navigation context, will be improved after full migration to react-router v6
  UNSAFE_NavigationContext,

  // React Router V5 API exports
  HashRouterV5,
  LinkV5,
  NavLinkV5,
};

export const createLazyRouteElement = (
  // Load the module's default export and turn it into React Element
  componentLoader: () => Promise<{ default: React.ComponentType<any> }>,
) => React.createElement(React.lazy(componentLoader));
export const createRouteElement = (component: React.ComponentType<any>) =>
  React.createElement(component);

export type { Location, NavigateFunction, Params, To, NavigateOptions };
