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
  useSearchParams,
} from './common/utils/RoutingUtils';
import { MlflowHeader } from './common/components/MlflowHeader';
import { useDarkThemeContext } from './common/contexts/DarkThemeContext';
import { WorkflowTypeProvider } from './common/contexts/WorkflowTypeContext';
import { shouldEnableWorkflowBasedNavigation } from './common/utils/FeatureUtils';
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
import {
  extractWorkspaceFromSearchParams,
  getActiveWorkspace,
  isGlobalRoute,
  setActiveWorkspace,
} from './workspaces/utils/WorkspaceUtils';
import { useWorkspaces } from './workspaces/hooks/useWorkspaces';

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
  const enableWorkflowBasedNavigation = shouldEnableWorkflowBasedNavigation();

  // Hide sidebar if we are in a single experiment page (only when feature flag is disabled)
  // When feature flag is enabled, sidebar should always be visible
  const isSingleExperimentPage = Boolean(experimentId);
  useEffect(() => {
    setShowSidebar(enableWorkflowBasedNavigation || !isSingleExperimentPage);
  }, [isSingleExperimentPage, enableWorkflowBasedNavigation]);

  return (
    <AssistantProvider>
      <AssistantRouteContextProvider />
      <WorkflowTypeProvider>
        <div css={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          <ErrorModal />
          <AppErrorBoundary>
            <RootAssistantLayout>
              <div
                css={{
                  backgroundColor: theme.colors.backgroundSecondary,
                  display: 'flex',
                  flexDirection: 'row',
                  width: '100%',
                }}
              >
                <MlflowSidebar showSidebar={showSidebar} setShowSidebar={setShowSidebar} />
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
      </WorkflowTypeProvider>
    </AssistantProvider>
  );
};

const WorkspaceRouterSync = ({ workspacesEnabled }: { workspacesEnabled: boolean }) => {
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate({ bypassWorkspacePrefix: true });
  const { workspaces, isLoading } = useWorkspaces(workspacesEnabled);

  useEffect(() => {
    if (!workspacesEnabled) {
      setActiveWorkspace(null);
      return;
    }

    // Extract workspace from query param
    const workspaceFromQuery = extractWorkspaceFromSearchParams(searchParams);
    const activeWorkspace = getActiveWorkspace();
    const isRootPath = location.pathname === '/' || location.pathname === '';

    // If workspace is in query param, validate it once workspaces are loaded
    if (workspaceFromQuery) {
      if (isLoading) {
        // Still loading workspaces, optimistically set the workspace
        // (will validate once loaded)
        if (activeWorkspace !== workspaceFromQuery) {
          setActiveWorkspace(workspaceFromQuery);
        }
        return;
      }

      // Workspaces loaded - validate
      const workspaceExists = workspaces.some((ws) => ws.name === workspaceFromQuery);
      if (!workspaceExists) {
        // Invalid workspace - clear it and redirect to workspace selector
        setActiveWorkspace(null);
        navigate('/', { replace: true });
        return;
      }

      // Valid workspace - sync to active state
      if (activeWorkspace !== workspaceFromQuery) {
        setActiveWorkspace(workspaceFromQuery);
      }
      return;
    }

    // No workspace query param - check if this is a global route
    const isOnGlobalRoute = isRootPath || isGlobalRoute(location.pathname);

    if (isOnGlobalRoute) {
      // Clear active workspace on global routes (workspace selector, settings)
      if (activeWorkspace) {
        setActiveWorkspace(null);
      }
      return;
    }

    // No workspace query param on a workspace-scoped route - redirect to selector
    setActiveWorkspace(null);
    navigate('/', { replace: true });
  }, [location, navigate, workspacesEnabled, searchParams, workspaces, isLoading]);

  return null;
};

const WorkspaceAwareRootRoute = ({ workspacesEnabled }: { workspacesEnabled: boolean }) => (
  <>
    <WorkspaceRouterSync workspacesEnabled={workspacesEnabled} />
    <MlflowRootRoute />
  </>
);

export const MlflowRouter = () => {
  const { workspacesEnabled, loading: featuresLoading } = useWorkspacesEnabled();

  // Routes are the same regardless of workspace mode - workspace context comes from query param
  const routes = useMemo<MlflowRouteDef[]>(
    () => [
      ...getExperimentTrackingRouteDefs(),
      ...getModelRegistryRouteDefs(),
      ...getGatewayRouteDefs(),
      ...getCommonRouteDefs(),
    ],
    [],
  );

  const hashRouter = useMemo(
    () =>
      // Don't create router while still loading features
      featuresLoading
        ? null
        : createHashRouter([
            {
              path: '/',
              element: <WorkspaceAwareRootRoute workspacesEnabled={workspacesEnabled} />,
              children: routes,
            },
          ]),
    [routes, workspacesEnabled, featuresLoading],
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
