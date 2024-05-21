import { createLazyRouteElement } from '../common/utils/RoutingUtils';

import { RoutePaths } from './routes';
export const getRouteDefs = () => [
  {
    path: RoutePaths.experimentPage,
    element: createLazyRouteElement(() => import('./components/HomePage')),
    pageId: 'mlflow.experiment.details',
  },
  {
    path: RoutePaths.experimentPageSearch,
    element: createLazyRouteElement(() => import('./components/HomePage')),
    pageId: 'mlflow.experiment.details.search',
  },
  {
    path: RoutePaths.compareExperimentsSearch,
    element: createLazyRouteElement(() => import('./components/HomePage')),
    pageId: 'mlflow.experiment.compare',
  },
  {
    path: RoutePaths.runPageWithTab,
    element: createLazyRouteElement(() => import('./components/run-page/RunPage')),
    pageId: 'mlflow.experiment.run.details',
  },
  {
    path: RoutePaths.runPageDirect,
    element: createLazyRouteElement(() => import('./components/DirectRunPage')),
    pageId: 'mlflow.experiment.run.details.direct',
  },
  {
    path: RoutePaths.compareRuns,
    element: createLazyRouteElement(() => import('./components/CompareRunPage')),
    pageId: 'mlflow.experiment.run.compare',
  },
  {
    path: RoutePaths.metricPage,
    element: createLazyRouteElement(() => import('./components/MetricPage')),
    pageId: 'mlflow.metric.details',
  },
];
