import { createLazyRouteElement } from '../common/utils/RoutingUtils';

import { PageId, RoutePaths } from './routes';
import { shouldEnableExperimentPageChildRoutes } from '../common/utils/FeatureUtils';

const getPromptPagesRouteDefs = () => {
  return [
    {
      path: RoutePaths.promptsPage,
      element: createLazyRouteElement(() => import('./pages/prompts/PromptsPage')),
      pageId: PageId.promptsPage,
    },
    {
      path: RoutePaths.promptDetailsPage,
      element: createLazyRouteElement(() => import('./pages/prompts/PromptsDetailsPage')),
      pageId: PageId.promptDetailsPage,
    },
  ];
};

const getExperimentPageRouteDefs = () => {
  if (shouldEnableExperimentPageChildRoutes()) {
    // When child routes flag is enabled, we use a single parent route with an <Outlet>
    // and define child routes for each tab.
    return [
      {
        path: RoutePaths.experimentObservatory,
        element: createLazyRouteElement(() => {
          return import('./components/ExperimentListView');
        }),
        pageId: 'mlflow.experiment.list',
      },
      {
        path: RoutePaths.experimentPage,
        element: createLazyRouteElement(() => {
          return import('./pages/experiment-page-tabs/ExperimentPageTabs');
        }),
        pageId: PageId.experimentPage,
        children: [
          {
            path: RoutePaths.experimentPageTabRuns,
            pageId: PageId.experimentPageTabRuns,
            element: createLazyRouteElement(() => import('./pages/experiment-runs/ExperimentRunsPage')),
          },
          {
            path: RoutePaths.experimentPageTabTraces,
            pageId: PageId.experimentPageTabTraces,
            element: createLazyRouteElement(() => import('./pages/experiment-traces/ExperimentTracesPage')),
          },
          {
            path: RoutePaths.experimentPageTabModels,
            pageId: PageId.experimentPageTabModels,
            element: createLazyRouteElement(
              () => import('./pages/experiment-logged-models/ExperimentLoggedModelListPage'),
            ),
          },
        ],
      },
    ];
  }
  return [
    {
      path: RoutePaths.experimentObservatory,
      element: createLazyRouteElement(() => {
        return import('./components/ExperimentListView');
      }),
      pageId: 'mlflow.experiment.list',
    },
    {
      path: RoutePaths.experimentPageTabbed,
      element: createLazyRouteElement(() => {
        return import(/* webpackChunkName: "experimentPage" */ './pages/experiment-page-tabs/ExperimentPageTabs');
      }),
      pageId: PageId.experimentPageTabbed,
    },
    {
      path: RoutePaths.experimentPage,
      element: createLazyRouteElement(
        () => import(/* webpackChunkName: "experimentPage" */ './components/experiment-page/ExperimentPage'),
      ),
      pageId: PageId.experimentPage,
    },
    {
      path: RoutePaths.experimentPageSearch,
      element: createLazyRouteElement(
        () => import(/* webpackChunkName: "experimentPage" */ './components/experiment-page/ExperimentPage'),
      ),
      pageId: PageId.experimentPageSearch,
    },
  ];
};

export const getRouteDefs = () => [
  {
    path: RoutePaths.rootRoute,
    element: createLazyRouteElement(() => import('../home/HomePage')),
    pageId: PageId.home,
  },
  ...getExperimentPageRouteDefs(),
  {
    path: RoutePaths.experimentLoggedModelDetailsPageTab,
    element: createLazyRouteElement(() => import('./pages/experiment-logged-models/ExperimentLoggedModelDetailsPage')),
    pageId: PageId.experimentLoggedModelDetailsPageTab,
  },
  {
    path: RoutePaths.experimentLoggedModelDetailsPage,
    element: createLazyRouteElement(() => import('./pages/experiment-logged-models/ExperimentLoggedModelDetailsPage')),
    pageId: PageId.experimentLoggedModelDetailsPage,
  },
  {
    path: RoutePaths.compareExperimentsSearch,
    element: createLazyRouteElement(
      () => import(/* webpackChunkName: "experimentPage" */ './components/experiment-page/ExperimentPage'),
    ),
    pageId: PageId.compareExperimentsSearch,
  },
  {
    path: RoutePaths.runPageWithTab,
    element: createLazyRouteElement(() => import('./components/run-page/RunPage')),
    pageId: PageId.runPageWithTab,
  },
  {
    path: RoutePaths.runPageDirect,
    element: createLazyRouteElement(() => import('./components/DirectRunPage')),
    pageId: PageId.runPageDirect,
  },
  {
    path: RoutePaths.compareRuns,
    element: createLazyRouteElement(() => import('./components/CompareRunPage')),
    pageId: PageId.compareRuns,
  },
  {
    path: RoutePaths.metricPage,
    element: createLazyRouteElement(() => import('./components/MetricPage')),
    pageId: PageId.metricPage,
  },
  ...getPromptPagesRouteDefs(),
];
