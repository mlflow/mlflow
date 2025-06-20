import React, { useMemo } from 'react';
import { LegacySkeleton, useDesignSystemTheme } from '@databricks/design-system';

import ErrorModal from './experiment-tracking/components/modals/ErrorModal';
import AppErrorBoundary from './common/components/error-boundaries/AppErrorBoundary';
import { HashRouter, Route, Routes, createLazyRouteElement } from './common/utils/RoutingUtils';
import { MlflowHeader } from './common/components/MlflowHeader';

// Route definition imports:
import { getRouteDefs as getExperimentTrackingRouteDefs } from './experiment-tracking/route-defs';
import { getRouteDefs as getModelRegistryRouteDefs } from './model-registry/route-defs';
import { getRouteDefs as getCommonRouteDefs } from './common/route-defs';
import { useInitializeExperimentRunColors } from './experiment-tracking/components/experiment-page/hooks/useExperimentRunColor';
import { MlflowSidebar } from './common/components/MlflowSidebar';

/**
 * This is the MLflow default entry/landing route.
 */
const landingRoute = {
  path: '/',
  element: createLazyRouteElement(() => import('./experiment-tracking/components/HomePage')),
  pageId: 'mlflow.experiments.list',
};

export const MlflowRouter = ({
  isDarkTheme,
  setIsDarkTheme,
}: {
  isDarkTheme?: boolean;
  setIsDarkTheme?: (isDarkTheme: boolean) => void;
}) => {
  // eslint-disable-next-line react-hooks/rules-of-hooks
  useInitializeExperimentRunColors();

  // eslint-disable-next-line react-hooks/rules-of-hooks
  const routes = useMemo(
    () => [...getExperimentTrackingRouteDefs(), ...getModelRegistryRouteDefs(), landingRoute, ...getCommonRouteDefs()],
    [],
  );

  const { theme } = useDesignSystemTheme();

  return (
    <>
      <ErrorModal />
      <HashRouter>
        <AppErrorBoundary>
          <MlflowHeader isDarkTheme={isDarkTheme} setIsDarkTheme={setIsDarkTheme} />
          <div
            css={{
              backgroundColor: theme.colors.backgroundSecondary,
              display: 'flex',
              flexDirection: 'row',
              height: '100%',
              justifyContent: 'stretch',
            }}
          >
            <MlflowSidebar />
            <main
              css={{
                width: '100%',
                backgroundColor: theme.colors.backgroundPrimary,
                margin: theme.spacing.sm,
                borderRadius: theme.borders.borderRadiusMd,
                boxShadow: theme.shadows.md,
              }}
            >
              <React.Suspense fallback={<LegacySkeleton />}>
                <Routes>
                  {routes.map(({ element, pageId, path }) => (
                    <Route key={pageId} path={path} element={element} />
                  ))}
                </Routes>
              </React.Suspense>
            </main>
          </div>
        </AppErrorBoundary>
      </HashRouter>
    </>
  );
};
