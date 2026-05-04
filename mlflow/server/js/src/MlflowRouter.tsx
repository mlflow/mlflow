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
import { useWorkflowType, WorkflowType, WorkflowTypeProvider } from './common/contexts/WorkflowTypeContext';
import { shouldEnableWorkflowBasedNavigation } from './common/utils/FeatureUtils';
import { useWorkspacesEnabled } from './experiment-tracking/hooks/useServerInfo';

// Route definition imports:
import { getRouteDefs as getExperimentTrackingRouteDefs } from './experiment-tracking/route-defs';
import { getRouteDefs as getModelRegistryRouteDefs } from './model-registry/route-defs';
import { getRouteDefs as getCommonRouteDefs } from './common/route-defs';
import { getGatewayRouteDefs } from './gateway/route-defs';
import { getAccountRouteDefs } from './account/route-defs';
import { useInitializeExperimentRunColors } from './experiment-tracking/components/experiment-page/hooks/useExperimentRunColor';
import { MlflowSidebar } from './common/components/MlflowSidebar';
import { AssistantProvider, AssistantRouteContextProvider } from './assistant';
import { RootAssistantLayout } from './common/components/RootAssistantLayout';
import {
  extractWorkspaceFromSearchParams,
  getActiveWorkspace,
  getLastUsedWorkspace,
  isGlobalRoute,
  setActiveWorkspace,
  setLastUsedWorkspace,
  WORKSPACE_QUERY_PARAM,
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
 * Inner layout component that has access to WorkflowType context
 */
const MlflowRootLayout = ({
  showSidebar,
  setShowSidebar,
}: {
  showSidebar: boolean;
  setShowSidebar: (showSidebar: boolean) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const { workflowType } = useWorkflowType();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <ErrorModal />
      <AppErrorBoundary>
        <RootAssistantLayout>
          <div
            css={{
              display: 'flex',
              flexDirection: 'row',
              width: '100%',
              background:
                workflowType === WorkflowType.GENAI
                  ? `linear-gradient(163deg, rgba(66, 153, 224, 0.06) 20%, rgba(202, 66, 224, 0.06) 35%, rgba(255, 95, 70, 0.06) 50%, transparent 80%), ${theme.colors.backgroundSecondary}`
                  : theme.colors.backgroundSecondary,
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
  );
};

/**
 * This is root element for MLflow routes, containing app header.
 */
const MlflowRootRoute = () => {
  useInitializeExperimentRunColors();

  const routeTitle = usePageTitle();
  useDocumentTitle({ title: routeTitle });

  const [showSidebar, setShowSidebar] = useState(true);
  const { experimentId } = useParams();
  const enableWorkflowBasedNavigation = shouldEnableWorkflowBasedNavigation();

  // Hide sidebar if we are in a single experiment page (only when feature flag is disabled)
  const isSingleExperimentPage = Boolean(experimentId);
  useEffect(() => {
    // When feature flag is enabled, sidebar should always retain its current state
    if (enableWorkflowBasedNavigation) {
      return;
    }
    setShowSidebar(enableWorkflowBasedNavigation || !isSingleExperimentPage);
  }, [isSingleExperimentPage, enableWorkflowBasedNavigation]);

  return (
    <AssistantProvider>
      <AssistantRouteContextProvider />
      <WorkflowTypeProvider>
        <MlflowRootLayout showSidebar={showSidebar} setShowSidebar={setShowSidebar} />
      </WorkflowTypeProvider>
    </AssistantProvider>
  );
};

export const WorkspaceRouterSync = ({ workspacesEnabled }: { workspacesEnabled: boolean }) => {
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate({ bypassWorkspacePrefix: true });
  useWorkspaces(workspacesEnabled);

  useEffect(() => {
    if (!workspacesEnabled) {
      setActiveWorkspace(null);
      // Clear localStorage so stale workspace values don't leak into
      // requests (e.g. X-MLFLOW-WORKSPACE header) when the server
      // no longer supports workspaces.
      setLastUsedWorkspace(null);
      return;
    }

    // Extract workspace from query param
    const workspaceFromQuery = extractWorkspaceFromSearchParams(searchParams);
    const activeWorkspace = getActiveWorkspace();
    const isRootPath = location.pathname === '/' || location.pathname === '';

    // If workspace is in query param, keep it active. The workspace list is not
    // authoritative because users may still have access to individual resources
    // inside a workspace even when the workspace itself is filtered out from
    // listWorkspaces.
    if (workspaceFromQuery) {
      if (activeWorkspace !== workspaceFromQuery) {
        setActiveWorkspace(workspaceFromQuery);
      }
      return;
    }

    // No workspace query param - check if this is a global route
    const isOnGlobalRoute = isRootPath || isGlobalRoute(location.pathname);

    if (isOnGlobalRoute) {
      // The workspace selector (root '/') clears the in-memory active
      // workspace so the user is in selector mode. Other global routes
      // (e.g. /account) leave the active workspace alone so navigating
      // back to a workspace-scoped page resumes in the same workspace.
      if (isRootPath && activeWorkspace) {
        setActiveWorkspace(null);
      }
      return;
    }

    // No workspace query param or local storage on a workspace-scoped route - redirect to selector
    const lastUsedWorkspace = getLastUsedWorkspace();
    if (!lastUsedWorkspace) {
      setActiveWorkspace(null);
      navigate('/', { replace: true });
      return;
    } else {
      navigate(location.pathname + '?' + WORKSPACE_QUERY_PARAM + '=' + lastUsedWorkspace, { replace: true });
    }
  }, [location, navigate, workspacesEnabled, searchParams]);

  return null;
};

const WorkspaceAwareRootRoute = ({ workspacesEnabled }: { workspacesEnabled: boolean }) => (
  <>
    <WorkspaceRouterSync workspacesEnabled={workspacesEnabled} />
    <MlflowRootRoute />
  </>
);

export const MlflowRouter = () => {
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const { workspacesEnabled, loading: featuresLoading } = useWorkspacesEnabled();

  // Routes are the same regardless of workspace mode - workspace context comes from query param
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const routes = useMemo<MlflowRouteDef[]>(
    () => [
      ...getExperimentTrackingRouteDefs(),
      ...getModelRegistryRouteDefs(),
      ...getGatewayRouteDefs(),
      ...getAccountRouteDefs(),
      ...getCommonRouteDefs(),
    ],
    [],
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
