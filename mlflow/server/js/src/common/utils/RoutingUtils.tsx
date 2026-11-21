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
  matchPath,
  generatePath,
  Route,
  UNSAFE_NavigationContext,
  UNSAFE_DataRouterContext,
  UNSAFE_LocationContext,
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
  type SetURLSearchParams,
  createSearchParams,
  NavigationType,
} from 'react-router-dom';

/**
 * Import React Router V5 parts
 */
import { HashRouter as HashRouterV5, Link as LinkV5, NavLink as NavLinkV5 } from 'react-router-dom';
import type { ComponentProps } from 'react';
import React, { createContext, useCallback, useContext, useMemo, useRef } from 'react';
import { useSyncExternalStoreWithSelector } from 'use-sync-external-store/shim/with-selector';
import { isEqual } from 'lodash';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
} from '@databricks/design-system';

import {
  prefixRouteWithWorkspace,
  getActiveWorkspace,
  isGlobalRoute,
  WORKSPACE_QUERY_PARAM,
} from '../../workspaces/utils/WorkspaceUtils';
import { getWorkspacesEnabledSync } from '../../experiment-tracking/hooks/useServerInfo';

const useLocation = useLocationDirect;

const useSearchParams = useSearchParamsDirect;

const UnusedLocationContext = createContext<React.ContextType<typeof UNSAFE_LocationContext>>({
  location: { pathname: '/', search: '', hash: '', state: null, key: 'default' },
  navigationType: NavigationType.Pop,
});
const subscribeToNothing = () => () => {};

/** Read selected location state without subscribing to unrelated query-string changes. */
export const useLocationSelector = <T,>(selector: (location: Location) => T): T => {
  const dataRouter = useContext(UNSAFE_DataRouterContext);
  const router = dataRouter?.router;
  const basename = dataRouter?.basename ?? '/';
  // Non-data routers (for example MemoryRouter) expose their location through context instead.
  const fallback = useContext(router ? UnusedLocationContext : UNSAFE_LocationContext);
  const getSnapshot = () => {
    const location = router?.state.location ?? fallback?.location;
    if (!location) throw new Error('Location selectors require a router.');
    return location;
  };
  return useSyncExternalStoreWithSelector(
    router?.subscribe ?? subscribeToNothing,
    getSnapshot,
    getSnapshot,
    (current) =>
      selector(
        router && basename !== '/'
          ? { ...current, pathname: current.pathname.slice(basename.replace(/\/$/, '').length) || '/' }
          : current,
      ),
    isEqual,
  );
};

/**
 * Select query state without unrelated data-router renders. The setter has React Router's call signature,
 * but functional updates receive fresh params and navigation retains the current path, hash and state.
 * Explicit options override retained state, including `state: undefined`. Non-functional updates replace
 * the query. Only data routers compose same-tick writes across hook instances; fallback routers use their
 * committed location plus writes made through this setter.
 */
export const useSearchParamSelector = <T,>(selector: (params: URLSearchParams) => T): [T, SetURLSearchParams] => {
  const router = useContext(UNSAFE_DataRouterContext)?.router;
  const { navigator, basename } = useContext(UNSAFE_NavigationContext);
  // Only non-data routers need a location subscription to keep the setter's fallback current.
  const location = useLocationSelector((current) => (router ? null : current));
  const locationRef = useRef(location);
  locationRef.current = location;
  const selected = useLocationSelector((current) => selector(new URLSearchParams(current.search)));
  const setParams = useCallback<SetURLSearchParams>(
    (next, options) => {
      const current = router?.state.navigation.location ?? router?.state.location ?? locationRef.current;
      if (!current) throw new Error('Search parameter setters require a router.');
      const params = createSearchParams(typeof next === 'function' ? next(new URLSearchParams(current.search)) : next);
      const pathname =
        router && basename !== '/'
          ? current.pathname.slice(basename.replace(/\/$/, '').length) || '/'
          : current.pathname;
      const target = { pathname, search: `?${params}`, hash: current.hash };
      const nextOptions = { state: current.state, ...options };
      if (router) {
        void router.navigate(target, nextOptions);
      } else {
        locationRef.current = { ...current, ...target, state: nextOptions.state };
        navigator[options?.replace ? 'replace' : 'push'](
          { ...target, pathname: `${basename === '/' ? '' : basename.replace(/\/$/, '')}${target.pathname}` },
          nextOptions.state,
          options,
        );
      }
    },
    [router, navigator, basename],
  );
  return [selected, setParams];
};

const useParams = useParamsDirect;

const useMatches = useMatchesDirect;

type UseNavigateOptions = {
  bypassWorkspacePrefix?: boolean;
};

const useNavigate = (options: UseNavigateOptions = { bypassWorkspacePrefix: false }): NavigateFunction => {
  const { bypassWorkspacePrefix } = options;
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const navigate = useNavigateDirect();
  // eslint-disable-next-line react-hooks/rules-of-hooks
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

    // Keep workspace prefixing when an active workspace is already known, even
    // if the async server feature flags have not resolved yet.
    if ((!getWorkspacesEnabledSync() && !getActiveWorkspace()) || typeof pathname !== 'string') {
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
  ComponentProps<typeof LinkDirect> & { disableWorkspacePrefix?: boolean; componentId: string }
>(function Link(props, ref) {
  const { to, disableWorkspacePrefix, componentId, onClick, ...rest } = props;
  const finalTo = disableWorkspacePrefix ? to : prefixRouteWithWorkspaceForTo(to);
  const events = useMemo(() => [DesignSystemEventProviderAnalyticsEventTypes.OnClick], []);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Button,
    componentId,
    analyticsEvents: events,
  });

  const handleClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
    eventContext?.onClick(e);
    onClick?.(e);
  };

  return <LinkDirect ref={ref} to={finalTo} onClick={handleClick} {...rest} />;
});

export const createMLflowRoutePath = (routePath: string) => {
  return routePath;
};

export {
  // React Router V6 API exports
  BrowserRouter,
  MemoryRouter,
  Link,
  useNavigate,
  useLocation,
  useParams,
  useSearchParams,
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
  // eslint-disable-next-line @databricks/react-lazy-only-at-top-level
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
