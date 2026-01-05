import { createLazyRouteElement } from '../common/utils/RoutingUtils';

import { PageId, RoutePaths } from './routes';

const getPromptPagesRouteDefs = () => {
  return [
    {
      path: RoutePaths.promptsPage,
      element: createLazyRouteElement(() => import('./pages/prompts/PromptsPage')),
      pageId: PageId.promptsPage,
      handle: { title: 'Prompts' },
    },
    {
      path: RoutePaths.promptDetailsPage,
      element: createLazyRouteElement(() => import('./pages/prompts/PromptsDetailsPage')),
      pageId: PageId.promptDetailsPage,
      handle: { title: 'Prompt Details' },
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
      handle: { title: 'Experiments' },
    },
    {
      path: RoutePaths.experimentPage,
      element: createLazyRouteElement(() => {
        return import('./pages/experiment-page-tabs/ExperimentPageTabs');
      }),
      pageId: PageId.experimentPage,
      handle: { title: 'Experiment' },
      children: [
        {
          path: RoutePaths.experimentPageTabOverview,
          pageId: PageId.experimentPageTabOverview,
          element: createLazyRouteElement(() => import('./pages/experiment-overview/ExperimentGenAIOverviewPage')),
          handle: { title: 'Experiment Overview' },
        },
        {
          path: RoutePaths.experimentPageTabRuns,
          pageId: PageId.experimentPageTabRuns,
          element: createLazyRouteElement(() => import('./pages/experiment-runs/ExperimentRunsPage')),
          handle: { title: 'Experiment Runs' },
        },
        {
          path: RoutePaths.experimentPageTabTraces,
          pageId: PageId.experimentPageTabTraces,
          element: createLazyRouteElement(() => import('./pages/experiment-traces/ExperimentTracesPage')),
          handle: { title: 'Experiment Traces' },
        },
        {
          path: RoutePaths.experimentPageTabChatSessions,
          pageId: PageId.experimentPageTabChatSessions,
          element: createLazyRouteElement(() => import('./pages/experiment-chat-sessions/ExperimentChatSessionsPage')),
          handle: { title: 'Chat Sessions' },
        },
        {
          path: RoutePaths.experimentPageTabSingleChatSession,
          pageId: PageId.experimentPageTabSingleChatSession,
          element: createLazyRouteElement(
            () => import('./pages/experiment-chat-sessions/single-chat-view/ExperimentSingleChatSessionPage'),
          ),
          handle: { title: 'Chat Session' },
        },
        {
          path: RoutePaths.experimentPageTabModels,
          pageId: PageId.experimentPageTabModels,
          element: createLazyRouteElement(
            () => import('./pages/experiment-logged-models/ExperimentLoggedModelListPage'),
          ),
          handle: { title: 'Logged Models' },
        },
        {
          path: RoutePaths.experimentPageTabEvaluationRuns,
          pageId: PageId.experimentPageTabEvaluationRuns,
          element: createLazyRouteElement(
            () => import('./pages/experiment-evaluation-runs/ExperimentEvaluationRunsPage'),
          ),
          handle: { title: 'Evaluation Runs' },
        },
        {
          path: RoutePaths.experimentPageTabScorers,
          pageId: PageId.experimentPageTabScorers,
          element: createLazyRouteElement(() => import('./pages/experiment-scorers/ExperimentScorersPage')),
          handle: { title: 'Scorers' },
        },
        {
          path: RoutePaths.experimentPageTabDatasets,
          pageId: PageId.experimentPageTabDatasets,
          element: createLazyRouteElement(() => {
            return import('./pages/experiment-evaluation-datasets/ExperimentEvaluationDatasetsPage');
          }),
          handle: { title: 'Datasets' },
        },
        {
          path: RoutePaths.experimentPageTabPrompts,
          pageId: PageId.experimentPageTabPrompts,
          element: createLazyRouteElement(() => import('./pages/prompts/ExperimentPromptsPage')),
          handle: { title: 'Experiment Prompts' },
        },
        {
          path: RoutePaths.experimentPageTabPromptDetails,
          pageId: PageId.experimentPageTabPromptDetails,
          element: createLazyRouteElement(() => import('./pages/prompts/ExperimentPromptDetailsPage')),
          handle: { title: 'Prompt Details' },
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
    handle: { title: 'Home' },
  },
  {
    path: RoutePaths.settingsPage,
    element: createLazyRouteElement(() => import('../settings/SettingsPage')),
    pageId: PageId.settingsPage,
    handle: { title: 'Settings' },
  },
  ...getExperimentPageRouteDefs(),
  {
    path: RoutePaths.experimentLoggedModelDetailsPageTab,
    element: createLazyRouteElement(() => import('./pages/experiment-logged-models/ExperimentLoggedModelDetailsPage')),
    pageId: PageId.experimentLoggedModelDetailsPageTab,
    handle: { title: 'Model Details' },
  },
  {
    path: RoutePaths.experimentLoggedModelDetailsPage,
    element: createLazyRouteElement(() => import('./pages/experiment-logged-models/ExperimentLoggedModelDetailsPage')),
    pageId: PageId.experimentLoggedModelDetailsPage,
    handle: { title: 'Model Details' },
  },
  {
    path: RoutePaths.compareExperimentsSearch,
    element: createLazyRouteElement(
      () => import(/* webpackChunkName: "experimentPage" */ './components/experiment-page/ExperimentPage'),
    ),
    pageId: PageId.compareExperimentsSearch,
    handle: { title: 'Compare Experiments' },
  },
  {
    path: RoutePaths.runPageWithTab,
    element: createLazyRouteElement(() => import('./components/run-page/RunPage')),
    pageId: PageId.runPageWithTab,
    handle: { title: 'Run Details' },
  },
  {
    path: RoutePaths.runPageDirect,
    element: createLazyRouteElement(() => import('./components/DirectRunPage')),
    pageId: PageId.runPageDirect,
    handle: { title: 'Run Details' },
  },
  {
    path: RoutePaths.compareRuns,
    element: createLazyRouteElement(() => import('./components/CompareRunPage')),
    pageId: PageId.compareRuns,
    handle: { title: 'Compare Runs' },
  },
  {
    path: RoutePaths.metricPage,
    element: createLazyRouteElement(() => import('./components/MetricPage')),
    pageId: PageId.metricPage,
    handle: { title: 'Metric Details' },
  },
  ...getPromptPagesRouteDefs(),
];
