import React, { useEffect, useMemo, useRef, useState } from 'react';
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
  useLocation,
  matchPath,
} from './common/utils/RoutingUtils';
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
  const location = useLocation();

  // Determine current experiment context
  const experimentMatch =
    matchPath('/experiments/:experimentId/*', location.pathname) ||
    matchPath('/experiments/:experimentId', location.pathname);
  const currentExperimentId = (experimentMatch && (experimentMatch as any).params?.experimentId) as
    | string
    | undefined;
  const isSingleExperimentContext = Boolean(currentExperimentId);

  // Keep previous context to default-hide only on transitions into a single experiment
  const prevCtxRef = useRef<{ isSingle: boolean; experimentId?: string } | null>(null);
  useEffect(() => {
    const prev = prevCtxRef.current;
    if (!prev) {
      // Initial mount: default-hide if we start on an experiment page
      setShowSidebar(!isSingleExperimentContext);
      prevCtxRef.current = { isSingle: isSingleExperimentContext, experimentId: currentExperimentId };
      return;
    }

    // Entering a single-experiment context (from non-experiment or switching experiments)
    if (
      isSingleExperimentContext &&
      (!prev.isSingle || prev.experimentId !== currentExperimentId)
    ) {
      setShowSidebar(false);
    }
    // Leaving experiment context back to top-level pages
    else if (!isSingleExperimentContext && prev.isSingle) {
      setShowSidebar(true);
    }

    prevCtxRef.current = { isSingle: isSingleExperimentContext, experimentId: currentExperimentId };
  }, [isSingleExperimentContext, currentExperimentId, location.pathname]);

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

  if (hashRouter) {
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
