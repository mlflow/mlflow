import React, { useEffect, useMemo, useState } from 'react';
import { LegacySkeleton, useDesignSystemTheme } from '@databricks/design-system';

import ErrorModal from './experiment-tracking/components/modals/ErrorModal';
import AppErrorBoundary from './common/components/error-boundaries/AppErrorBoundary';
import {
  createHashRouter,
  RouterProvider,
  Outlet,
  createLazyRouteElement,
  useLocation,
  useNavigate,
  useParams,
} from './common/utils/RoutingUtils';
import { MlflowHeader } from './common/components/MlflowHeader';
import { useDarkThemeContext } from './common/contexts/DarkThemeContext';
import { useWorkspacesEnabled } from './common/utils/ServerFeaturesContext';

// Route definition imports:
import { getRouteDefs as getExperimentTrackingRouteDefs } from './experiment-tracking/route-defs';
import { getRouteDefs as getModelRegistryRouteDefs } from './model-registry/route-defs';
import { getRouteDefs as getCommonRouteDefs } from './common/route-defs';
import { useInitializeExperimentRunColors } from './experiment-tracking/components/experiment-page/hooks/useExperimentRunColor';
import { MlflowSidebar } from './common/components/MlflowSidebar';
import {
  DEFAULT_WORKSPACE_NAME,
  extractWorkspaceFromPathname,
  setActiveWorkspace,
  subscribeToWorkspaceChanges,
  getCurrentWorkspace,
} from './common/utils/WorkspaceUtils';
import { prefixRoutePathWithWorkspace } from './common/utils/WorkspaceRouteUtils';

/**
 * This is the MLflow default entry/landing route.
 */
const landingRoute = {
  path: '/',
  element: createLazyRouteElement(() => import('./experiment-tracking/components/HomePage')),
  pageId: 'mlflow.experiments.list',
};

type MlflowRouteDef = {
  path?: string;
  element?: React.ReactNode;
  pageId?: string;
  children?: MlflowRouteDef[];
  [key: string]: unknown;
};

/**
 * This is root element for MLflow routes, containing app header.
 */
const MlflowRootRoute = () => {
  useInitializeExperimentRunColors();

  const [showSidebar, setShowSidebar] = useState(true);
  const { theme } = useDesignSystemTheme();
  const { experimentId } = useParams();
  const { setIsDarkTheme } = useDarkThemeContext();
  const isDarkTheme = theme.isDarkMode;

  // Hide sidebar if we are in a single experiment page
  const isSingleExperimentPage = Boolean(experimentId);
  useEffect(() => {
    setShowSidebar(!isSingleExperimentPage);
  }, [isSingleExperimentPage]);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <ErrorModal />
      <AppErrorBoundary>
        <MlflowHeader
          isDarkTheme={isDarkTheme}
          setIsDarkTheme={setIsDarkTheme}
          sidebarOpen={showSidebar}
          toggleSidebar={() => setShowSidebar((isOpen) => !isOpen)}
        />
        <div
          css={{
            backgroundColor: theme.colors.backgroundSecondary,
            display: 'flex',
            flexDirection: 'row',
            flexGrow: 1,
            minHeight: 0,
          }}
        >
          {showSidebar && <MlflowSidebar />}
          <main
            css={{
              width: '100%',
              backgroundColor: theme.colors.backgroundPrimary,
              margin: theme.spacing.sm,
              borderRadius: theme.borders.borderRadiusMd,
              boxShadow: theme.shadows.md,
              overflowX: 'auto',
            }}
          >
            <React.Suspense fallback={<LegacySkeleton />}>
              <Outlet />
            </React.Suspense>
          </main>
        </div>
      </AppErrorBoundary>
    </div>
  );
};

const WorkspaceRouterSync = ({ workspacesEnabled }: { workspacesEnabled: boolean }) => {
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    if (!workspacesEnabled) {
      setActiveWorkspace(null);
      return;
    }

    const workspace = extractWorkspaceFromPathname(location.pathname);
    const activeWorkspace = getCurrentWorkspace();
    if (!workspace) {
      const fallbackWorkspace = activeWorkspace || DEFAULT_WORKSPACE_NAME;
      if (activeWorkspace !== fallbackWorkspace) {
        setActiveWorkspace(fallbackWorkspace);
      }
      const suffix = location.pathname === '/' ? '' : location.pathname;
      const search = location.search ?? '';
      const targetPath = `/workspaces/${encodeURIComponent(fallbackWorkspace)}${suffix === '/' ? '' : suffix}`;
      if (location.pathname !== targetPath) {
        navigate(`${targetPath}${search}`, { replace: true });
      }
      return;
    }

    if (workspace !== activeWorkspace) {
      setActiveWorkspace(workspace);
    }
  }, [location, navigate, workspacesEnabled]);

  return null;
};

const WorkspaceAwareRootRoute = ({ workspacesEnabled }: { workspacesEnabled: boolean }) => (
  <>
    <WorkspaceRouterSync workspacesEnabled={workspacesEnabled} />
    <MlflowRootRoute />
  </>
);

const prependWorkspaceToRoutes = (routeDefs: MlflowRouteDef[]): MlflowRouteDef[] =>
  routeDefs.map((route) => {
    const children = route.children ? prependWorkspaceToRoutes(route.children) : undefined;

    return {
      ...route,
      path: prefixRoutePathWithWorkspace(route.path),
      ...(children ? { children } : {}),
    };
  });

export const MlflowRouter = () => {
  const { workspacesEnabled, loading: featuresLoading } = useWorkspacesEnabled();

  // eslint-disable-next-line react-hooks/rules-of-hooks
  const routes = useMemo<MlflowRouteDef[]>(
    () => [...getExperimentTrackingRouteDefs(), ...getModelRegistryRouteDefs(), landingRoute, ...getCommonRouteDefs()],
    [],
  );
  const [workspaceKey, setWorkspaceKey] = useState(() => getCurrentWorkspace() ?? DEFAULT_WORKSPACE_NAME);

  useEffect(() => {
    return subscribeToWorkspaceChanges((workspace) => {
      setWorkspaceKey(workspace ?? DEFAULT_WORKSPACE_NAME);
    });
  }, []);

  const workspaceRoutes = useMemo(
    () => (workspacesEnabled ? prependWorkspaceToRoutes(routes) : []),
    [routes, workspacesEnabled],
  );
  const combinedRoutes = useMemo(
    () => (workspacesEnabled ? [...routes, ...workspaceRoutes] : routes),
    [routes, workspaceRoutes, workspacesEnabled],
  );

  // eslint-disable-next-line react-hooks/rules-of-hooks
  const hashRouter = useMemo(
    () =>
      // Don't create router while still loading features
      featuresLoading
        ? null
        : createHashRouter([
            {
              path: '/',
              element: <WorkspaceAwareRootRoute key={workspaceKey} workspacesEnabled={workspacesEnabled} />,
              children: combinedRoutes,
            },
          ]),
    [combinedRoutes, workspaceKey, workspacesEnabled, featuresLoading],
  );

  // Show loading skeleton while determining if workspaces are enabled
  if (featuresLoading || !hashRouter) {
    return <LegacySkeleton />;
  }

  return (
    <React.Suspense fallback={<LegacySkeleton />}>
      <RouterProvider router={hashRouter} />
    </React.Suspense>
  );
};
