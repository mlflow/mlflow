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
  useParams,
  usePageTitle,
} from './common/utils/RoutingUtils';
import { MlflowHeader } from './common/components/MlflowHeader';
import { useDarkThemeContext } from './common/contexts/DarkThemeContext';
import { WorkflowTypeProvider, useWorkflowType, WorkflowType } from './common/contexts/WorkflowTypeContext';
import { shouldEnableWorkflowBasedNavigation } from './common/utils/FeatureUtils';

// Route definition imports:
import { getRouteDefs as getExperimentTrackingRouteDefs } from './experiment-tracking/route-defs';
import { getRouteDefs as getModelRegistryRouteDefs } from './model-registry/route-defs';
import { getRouteDefs as getCommonRouteDefs } from './common/route-defs';
import { getGatewayRouteDefs } from './gateway/route-defs';
import { useInitializeExperimentRunColors } from './experiment-tracking/components/experiment-page/hooks/useExperimentRunColor';
import { MlflowSidebar } from './common/components/MlflowSidebar';
import { AssistantProvider, AssistantRouteContextProvider } from './assistant';
import { RootAssistantLayout } from './common/components/RootAssistantLayout';

/**
 * Inner layout component that has access to WorkflowType context.
 */
const MlflowRootLayout = ({
  showSidebar,
  setShowSidebar,
}: {
  showSidebar: boolean;
  setShowSidebar: React.Dispatch<React.SetStateAction<boolean>>;
}) => {
  const { theme } = useDesignSystemTheme();
  const { setIsDarkTheme } = useDarkThemeContext();
  const isDarkTheme = theme.isDarkMode;
  const { workflowType } = useWorkflowType();

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
        <RootAssistantLayout>
          <div
            css={{
              backgroundColor: theme.colors.backgroundSecondary,
              display: 'flex',
              flexDirection: 'row',
              width: '100%',
              background:
                workflowType === WorkflowType.GENAI
                  ? `radial-gradient(ellipse 200% 100% at bottom right, ${theme.colors.actionDangerDefaultBackgroundPress}, transparent 70%)`
                  : `radial-gradient(ellipse 200% 100% at bottom right, ${theme.colors.actionDefaultBackgroundPress}, transparent 70%)`,
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
  // When feature flag is enabled, sidebar should always be visible
  const isSingleExperimentPage = Boolean(experimentId);
  useEffect(() => {
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
export const MlflowRouter = () => {
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const routes = useMemo(
    () => [
      ...getExperimentTrackingRouteDefs(),
      ...getModelRegistryRouteDefs(),
      ...getGatewayRouteDefs(),
      ...getCommonRouteDefs(),
    ],
    [],
  );
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const hashRouter = useMemo(
    () =>
      createHashRouter([
        {
          path: '/',
          element: <MlflowRootRoute />,
          children: routes,
        },
      ]),
    [routes],
  );

  return (
    <React.Suspense fallback={<LegacySkeleton />}>
      <RouterProvider router={hashRouter} />
    </React.Suspense>
  );
};
