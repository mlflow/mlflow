import React, { useMemo, useState } from 'react';
import { LegacySkeleton, useDesignSystemTheme } from '@databricks/design-system';

import ErrorModal from './experiment-tracking/components/modals/ErrorModal';
import AppErrorBoundary from './common/components/error-boundaries/AppErrorBoundary';
import {
  HashRouter,
  createHashRouter,
  RouterProvider,
  Outlet,
  Route,
  Routes,
  createLazyRouteElement,
} from './common/utils/RoutingUtils';
import { MlflowHeader } from './common/components/MlflowHeader';

// Route definition imports:
import { getRouteDefs as getExperimentTrackingRouteDefs } from './experiment-tracking/route-defs';
import { getRouteDefs as getModelRegistryRouteDefs } from './model-registry/route-defs';
import { getRouteDefs as getCommonRouteDefs } from './common/route-defs';
import { useInitializeExperimentRunColors } from './experiment-tracking/components/experiment-page/hooks/useExperimentRunColor';
import { shouldEnableExperimentPageChildRoutes } from './common/utils/FeatureUtils';
import { MlflowSidebar } from './common/components/MlflowSidebar';

/**
 * This is the MLflow default entry/landing route.
 */
const landingRoute = {
  path: '/',
  element: createLazyRouteElement(() => import('./experiment-tracking/components/HomePage')),
  pageId: 'mlflow.experiments.list',
};

/**
 * This is root element for MLflow routes, containing app header.
 */
const MlflowRootRoute = ({
  isDarkTheme,
  setIsDarkTheme,
  useChildRoutesOutlet = false,
  routes,
}: {
  isDarkTheme?: boolean;
  setIsDarkTheme?: (isDarkTheme: boolean) => void;
  useChildRoutesOutlet?: boolean;
  routes?: any[];
}) => {
  useInitializeExperimentRunColors();

  const [showSidebar, setShowSidebar] = useState(true);
  const { theme } = useDesignSystemTheme();

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
              {useChildRoutesOutlet ? (
                <Outlet />
              ) : (
                <Routes>
                  {routes?.map(({ element, pageId, path }) => (
                    <Route key={pageId} path={path} element={element} />
                  ))}
                </Routes>
              )}
            </React.Suspense>
          </main>
        </div>
      </AppErrorBoundary>
    </div>
  );
};
export const MlflowRouter = ({
  isDarkTheme,
  setIsDarkTheme,
}: {
  isDarkTheme?: boolean;
  setIsDarkTheme?: (isDarkTheme: boolean) => void;
}) => {
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const routes = useMemo(
    () => [...getExperimentTrackingRouteDefs(), ...getModelRegistryRouteDefs(), landingRoute, ...getCommonRouteDefs()],
    [],
  );
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const hashRouter = useMemo(
    () => {
      if (!shouldEnableExperimentPageChildRoutes()) {
        return null;
      }
      return createHashRouter([
        {
          path: '/',
          element: <MlflowRootRoute isDarkTheme={isDarkTheme} setIsDarkTheme={setIsDarkTheme} useChildRoutesOutlet />,
          children: routes,
        },
      ]);
    },
    [routes, isDarkTheme, setIsDarkTheme] /* eslint-disable-line react-hooks/exhaustive-deps */,
  );

  if (hashRouter && shouldEnableExperimentPageChildRoutes()) {
    return (
      <React.Suspense fallback={<LegacySkeleton />}>
        <RouterProvider router={hashRouter} />
      </React.Suspense>
    );
  }

  return (
    <HashRouter>
      <MlflowRootRoute routes={routes} isDarkTheme={isDarkTheme} setIsDarkTheme={setIsDarkTheme} />
    </HashRouter>
  );
};
