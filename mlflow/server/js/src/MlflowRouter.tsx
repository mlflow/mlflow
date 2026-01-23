import React, { useEffect, useMemo, useState } from 'react';
import { LegacySkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { useDocumentTitle } from '@databricks/web-shared/routing';

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
  usePageTitle,
} from './common/utils/RoutingUtils';
import { MlflowHeader } from './common/components/MlflowHeader';
import { useDarkThemeContext } from './common/contexts/DarkThemeContext';
import { useWorkspacesEnabled } from './common/utils/ServerFeaturesContext';

// Route definition imports:
import { getRouteDefs as getExperimentTrackingRouteDefs } from './experiment-tracking/route-defs';
import { getRouteDefs as getModelRegistryRouteDefs } from './model-registry/route-defs';
import { getRouteDefs as getCommonRouteDefs } from './common/route-defs';
import { getGatewayRouteDefs } from './gateway/route-defs';
import { useInitializeExperimentRunColors } from './experiment-tracking/components/experiment-page/hooks/useExperimentRunColor';
import { MlflowSidebar } from './common/components/MlflowSidebar';
import { AssistantProvider, AssistantRouteContextProvider } from './assistant';
import { RootAssistantLayout } from './common/components/RootAssistantLayout';
import { extractWorkspaceFromPathname, setActiveWorkspace, getActiveWorkspace } from './common/utils/WorkspaceUtils';
import { prefixRoutePathWithWorkspace } from './common/utils/WorkspaceRouteUtils';
import { useWorkspaces } from './common/hooks/useWorkspaces';

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

  const routeTitle = usePageTitle();
  useDocumentTitle({ title: routeTitle });

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
    <AssistantProvider>
      <AssistantRouteContextProvider />
      <div css={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        <ErrorModal />
        <AppErrorBoundary>
          <MlflowHeader
            isDarkTheme={isDarkTheme}
            setIsDarkTheme={setIsDarkTheme}
            sidebarOpen={showSidebar}
            toggleSidebar={() => setShowSidebar((isOpen) => !isOpen)}
          />
          <RootAssistantLayout>
            <div
              css={{
                backgroundColor: theme.colors.backgroundSecondary,
                display: 'flex',
                flexDirection: 'row',
                width: '100%',
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
          </RootAssistantLayout>
        </AppErrorBoundary>
      </div>
    </AssistantProvider>
  );
};

const WorkspaceRouterSync = ({ workspacesEnabled }: { workspacesEnabled: boolean }) => {
  const location = useLocation();
  const navigate = useNavigate({ bypassWorkspacePrefix: true });
  const { workspaces, isLoading } = useWorkspaces(workspacesEnabled);

  useEffect(() => {
    if (!workspacesEnabled) {
      setActiveWorkspace(null);
      return;
    }

    const workspaceFromPath = extractWorkspaceFromPathname(location.pathname);
    const activeWorkspace = getActiveWorkspace();

    if (isLoading) {
      return;
    }

    // Validate workspace from path
    if (workspaceFromPath) {
      const isValid = workspaces.some((w) => w.name === workspaceFromPath);
      if (!isValid) {
        setActiveWorkspace(null);
        navigate('/', { replace: true });
        return;
      }
      if (activeWorkspace !== workspaceFromPath) {
        setActiveWorkspace(workspaceFromPath);
      }
      return;
    }

    // If not in a workspace path and not on root, redirect to selector
    const isRootPath = location.pathname === '/' || location.pathname === '';
    if (!isRootPath) {
      setActiveWorkspace(null);
      navigate('/', { replace: true });
      return;
    }
  }, [location, navigate, workspacesEnabled, workspaces, isLoading]);

  return null;
};

const WorkspaceAwareRootRoute = ({ workspacesEnabled }: { workspacesEnabled: boolean }) => (
  <>
    <WorkspaceRouterSync workspacesEnabled={workspacesEnabled} />
    <MlflowRootRoute />
  </>
);

const prependWorkspaceToRoutes = (routeDefs: MlflowRouteDef[], isChild = false): MlflowRouteDef[] =>
  routeDefs.map((route) => {
    // Only prepend workspace to child routes if they're absolute paths starting with /
    // Otherwise keep them relative to preserve React Router's nested routing
    const shouldPrependToPath = !isChild || (route.path && route.path.startsWith('/'));
    const children = route.children ? prependWorkspaceToRoutes(route.children, true) : undefined;

    return {
      ...route,
      path: shouldPrependToPath ? prefixRoutePathWithWorkspace(route.path) : route.path,
      ...(children ? { children } : {}),
    };
  });

export const MlflowRouter = () => {
  const { workspacesEnabled, loading: featuresLoading } = useWorkspacesEnabled();

  // eslint-disable-next-line react-hooks/rules-of-hooks
  const routes = useMemo<MlflowRouteDef[]>(
    () => [
      ...getExperimentTrackingRouteDefs(),
      ...getModelRegistryRouteDefs(),
      ...getGatewayRouteDefs(),
      ...getCommonRouteDefs(),
    ],
    [],
  );

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
              element: <WorkspaceAwareRootRoute workspacesEnabled={workspacesEnabled} />,
              children: combinedRoutes,
            },
          ]),
    [combinedRoutes, workspacesEnabled, featuresLoading],
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
