import { createLazyRouteElement } from '../common/utils/RoutingUtils';

import { RoutePaths } from './routes';

const getPromptPagesRouteDefs = () => {
  return [
    {
      path: RoutePaths.promptsPage,
      element: createLazyRouteElement(() => import('./pages/prompts/PromptsPage')),
      pageId: 'mlflow.prompts',
    },
    {
      path: RoutePaths.promptDetailsPage,
      element: createLazyRouteElement(() => import('./pages/prompts/PromptsDetailsPage')),
      pageId: 'mlflow.prompts.details',
    },
  ];
};

export const getRouteDefs = () => [
  {
    path: RoutePaths.experimentPageTabbed,
    element: createLazyRouteElement(() => {
      return import(/* webpackChunkName: "experimentPage" */ './components/HomePage');
    }),
    pageId: 'mlflow.experiment.details.tab',
  },
  {
    path: RoutePaths.experimentLoggedModelDetailsPageTab,
    element: createLazyRouteElement(() => import('./pages/experiment-logged-models/ExperimentLoggedModelDetailsPage')),
    pageId: 'mlflow.logged-model.details.tab',
  },
  {
    path: RoutePaths.experimentLoggedModelDetailsPage,
    element: createLazyRouteElement(() => import('./pages/experiment-logged-models/ExperimentLoggedModelDetailsPage')),
    pageId: 'mlflow.logged-model.details',
  },
  {
    path: RoutePaths.experimentPage,
    element: createLazyRouteElement(() => import(/* webpackChunkName: "experimentPage" */ './components/HomePage')),
    pageId: 'mlflow.experiment.details',
  },
  {
    path: RoutePaths.experimentPageSearch,
    element: createLazyRouteElement(() => import(/* webpackChunkName: "experimentPage" */ './components/HomePage')),
    pageId: 'mlflow.experiment.details.search',
  },
  {
    path: RoutePaths.compareExperimentsSearch,
    element: createLazyRouteElement(() => import(/* webpackChunkName: "experimentPage" */ './components/HomePage')),
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
  ...getPromptPagesRouteDefs(),
];
