import { createLazyRouteElement } from '../common/utils/RoutingUtils';

import { PageId, RoutePaths } from './routes';

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
          path: RoutePaths.experimentPageTabChatSessions,
          pageId: PageId.experimentPageTabChatSessions,
          element: createLazyRouteElement(() => import('./pages/experiment-chat-sessions/ExperimentChatSessionsPage')),
        },
        {
          path: RoutePaths.experimentPageTabSingleChatSession,
          pageId: PageId.experimentPageTabSingleChatSession,
          element: createLazyRouteElement(
            () => import('./pages/experiment-chat-sessions/single-chat-view/ExperimentSingleChatSessionPage'),
          ),
        },
        {
          path: RoutePaths.experimentPageTabModels,
          pageId: PageId.experimentPageTabModels,
          element: createLazyRouteElement(
            () => import('./pages/experiment-logged-models/ExperimentLoggedModelListPage'),
          ),
        },
        {
          path: RoutePaths.experimentPageTabEvaluationRuns,
          pageId: PageId.experimentPageTabEvaluationRuns,
          element: createLazyRouteElement(
            () => import('./pages/experiment-evaluation-runs/ExperimentEvaluationRunsPage'),
          ),
        },
        {
          path: RoutePaths.experimentPageTabScorers,
          pageId: PageId.experimentPageTabScorers,
          element: createLazyRouteElement(() => import('./pages/experiment-scorers/ExperimentScorersPage')),
        },
        {
          path: RoutePaths.experimentPageTabDatasets,
          pageId: PageId.experimentPageTabDatasets,
          element: createLazyRouteElement(() => {
            return import('./pages/experiment-evaluation-datasets/ExperimentEvaluationDatasetsPage');
          }),
        },
        {
          path: RoutePaths.experimentPageTabPrompts,
          pageId: PageId.experimentPageTabPrompts,
          element: createLazyRouteElement(() => import('./pages/prompts/ExperimentPromptsPage')),
        },
        {
          path: RoutePaths.experimentPageTabPromptDetails,
          pageId: PageId.experimentPageTabPromptDetails,
          element: createLazyRouteElement(() => import('./pages/prompts/ExperimentPromptDetailsPage')),
        },
      ],
    },
  ];
};

export const getRouteDefs = () => [
  {
    path: RoutePaths.rootRoute,
    element: createLazyRouteElement(() => import('../home/HomePage')),
    pageId: PageId.home,
  },
  {
    path: RoutePaths.settingsPage,
    element: createLazyRouteElement(() => import('../settings/SettingsPage')),
    pageId: PageId.settingsPage,
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
