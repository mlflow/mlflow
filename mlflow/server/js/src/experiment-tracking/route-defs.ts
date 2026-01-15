import { createLazyRouteElement, DocumentTitleHandle } from '../common/utils/RoutingUtils';

import { PageId, RoutePaths } from './routes';

const getPromptPagesRouteDefs = () => {
  return [
    {
      path: RoutePaths.promptsPage,
      element: createLazyRouteElement(() => import('./pages/prompts/PromptsPage')),
      pageId: PageId.promptsPage,
      handle: { getPageTitle: () => 'Prompts' } satisfies DocumentTitleHandle,
    },
    {
      path: RoutePaths.promptDetailsPage,
      element: createLazyRouteElement(() => import('./pages/prompts/PromptsDetailsPage')),
      pageId: PageId.promptDetailsPage,
      handle: { getPageTitle: (params) => `Prompt: ${params['promptName']}` } satisfies DocumentTitleHandle,
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
      handle: { getPageTitle: () => 'Experiments' } satisfies DocumentTitleHandle,
    },
    {
      path: RoutePaths.experimentPage,
      element: createLazyRouteElement(() => {
        return import('./pages/experiment-page-tabs/ExperimentPageTabs');
      }),
      pageId: PageId.experimentPage,
      handle: { getPageTitle: (params) => `Experiment ${params['experimentId']}` } satisfies DocumentTitleHandle,
      children: [
        {
          path: RoutePaths.experimentPageTabOverview,
          pageId: PageId.experimentPageTabOverview,
          element: createLazyRouteElement(() => import('./pages/experiment-overview/ExperimentGenAIOverviewPage')),
          handle: {
            getPageTitle: (params) => `Overview - Experiment ${params['experimentId']}`,
          } satisfies DocumentTitleHandle,
        },
        {
          path: RoutePaths.experimentPageTabRuns,
          pageId: PageId.experimentPageTabRuns,
          element: createLazyRouteElement(() => import('./pages/experiment-runs/ExperimentRunsPage')),
          handle: {
            getPageTitle: (params) => `Runs - Experiment ${params['experimentId']}`,
          } satisfies DocumentTitleHandle,
        },
        {
          path: RoutePaths.experimentPageTabTraces,
          pageId: PageId.experimentPageTabTraces,
          element: createLazyRouteElement(() => import('./pages/experiment-traces/ExperimentTracesPage')),
          handle: {
            getPageTitle: (params) => `Traces - Experiment ${params['experimentId']}`,
          } satisfies DocumentTitleHandle,
        },
        {
          path: RoutePaths.experimentPageTabChatSessions,
          pageId: PageId.experimentPageTabChatSessions,
          element: createLazyRouteElement(() => import('./pages/experiment-chat-sessions/ExperimentChatSessionsPage')),
          handle: {
            getPageTitle: (params) => `Chat Sessions - Experiment ${params['experimentId']}`,
          } satisfies DocumentTitleHandle,
        },
        {
          path: RoutePaths.experimentPageTabSingleChatSession,
          pageId: PageId.experimentPageTabSingleChatSession,
          element: createLazyRouteElement(
            () => import('./pages/experiment-chat-sessions/single-chat-view/ExperimentSingleChatSessionPage'),
          ),
          handle: { getPageTitle: (params) => `Chat Session ${params['sessionId']}` } satisfies DocumentTitleHandle,
        },
        {
          path: RoutePaths.experimentPageTabModels,
          pageId: PageId.experimentPageTabModels,
          element: createLazyRouteElement(
            () => import('./pages/experiment-logged-models/ExperimentLoggedModelListPage'),
          ),
          handle: {
            getPageTitle: (params) => `Logged Models - Experiment ${params['experimentId']}`,
          } satisfies DocumentTitleHandle,
        },
        {
          path: RoutePaths.experimentPageTabEvaluationRuns,
          pageId: PageId.experimentPageTabEvaluationRuns,
          element: createLazyRouteElement(
            () => import('./pages/experiment-evaluation-runs/ExperimentEvaluationRunsPage'),
          ),
          handle: {
            getPageTitle: (params) => `Evaluation Runs - Experiment ${params['experimentId']}`,
          } satisfies DocumentTitleHandle,
        },
        {
          path: RoutePaths.experimentPageTabScorers,
          pageId: PageId.experimentPageTabScorers,
          element: createLazyRouteElement(() => import('./pages/experiment-scorers/ExperimentScorersPage')),
          handle: {
            getPageTitle: (params) => `Scorers - Experiment ${params['experimentId']}`,
          } satisfies DocumentTitleHandle,
        },
        {
          path: RoutePaths.experimentPageTabDatasets,
          pageId: PageId.experimentPageTabDatasets,
          element: createLazyRouteElement(() => {
            return import('./pages/experiment-evaluation-datasets/ExperimentEvaluationDatasetsPage');
          }),
          handle: {
            getPageTitle: (params) => `Datasets - Experiment ${params['experimentId']}`,
          } satisfies DocumentTitleHandle,
        },
        {
          path: RoutePaths.experimentPageTabPrompts,
          pageId: PageId.experimentPageTabPrompts,
          element: createLazyRouteElement(() => import('./pages/prompts/ExperimentPromptsPage')),
          handle: {
            getPageTitle: (params) => `Prompts - Experiment ${params['experimentId']}`,
          } satisfies DocumentTitleHandle,
        },
        {
          path: RoutePaths.experimentPageTabPromptDetails,
          pageId: PageId.experimentPageTabPromptDetails,
          element: createLazyRouteElement(() => import('./pages/prompts/ExperimentPromptDetailsPage')),
          handle: { getPageTitle: (params) => `Prompt: ${params['promptName']}` } satisfies DocumentTitleHandle,
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
    handle: { getPageTitle: () => 'Home' } satisfies DocumentTitleHandle,
  },
  {
    path: RoutePaths.settingsPage,
    element: createLazyRouteElement(() => import('../settings/SettingsPage')),
    pageId: PageId.settingsPage,
    handle: { getPageTitle: () => 'Settings' } satisfies DocumentTitleHandle,
  },
  ...getExperimentPageRouteDefs(),
  {
    path: RoutePaths.experimentLoggedModelDetailsPageTab,
    element: createLazyRouteElement(() => import('./pages/experiment-logged-models/ExperimentLoggedModelDetailsPage')),
    pageId: PageId.experimentLoggedModelDetailsPageTab,
    handle: { getPageTitle: (params) => `Model ${params['loggedModelId']}` } satisfies DocumentTitleHandle,
  },
  {
    path: RoutePaths.experimentLoggedModelDetailsPage,
    element: createLazyRouteElement(() => import('./pages/experiment-logged-models/ExperimentLoggedModelDetailsPage')),
    pageId: PageId.experimentLoggedModelDetailsPage,
    handle: { getPageTitle: (params) => `Model ${params['loggedModelId']}` } satisfies DocumentTitleHandle,
  },
  {
    path: RoutePaths.compareExperimentsSearch,
    element: createLazyRouteElement(
      () => import(/* webpackChunkName: "experimentPage" */ './components/experiment-page/ExperimentPage'),
    ),
    pageId: PageId.compareExperimentsSearch,
    handle: { getPageTitle: () => 'Compare Experiments' } satisfies DocumentTitleHandle,
  },
  {
    path: RoutePaths.runPageWithTab,
    element: createLazyRouteElement(() => import('./components/run-page/RunPage')),
    pageId: PageId.runPageWithTab,
    handle: { getPageTitle: (params) => `Run ${params['runUuid']}` } satisfies DocumentTitleHandle,
  },
  {
    path: RoutePaths.runPageDirect,
    element: createLazyRouteElement(() => import('./components/DirectRunPage')),
    pageId: PageId.runPageDirect,
    handle: { getPageTitle: (params) => `Run ${params['runUuid']}` } satisfies DocumentTitleHandle,
  },
  {
    path: RoutePaths.compareRuns,
    element: createLazyRouteElement(() => import('./components/CompareRunPage')),
    pageId: PageId.compareRuns,
    handle: { getPageTitle: () => 'Compare Runs' } satisfies DocumentTitleHandle,
  },
  {
    path: RoutePaths.metricPage,
    element: createLazyRouteElement(() => import('./components/MetricPage')),
    pageId: PageId.metricPage,
    handle: { getPageTitle: () => 'Metric Details' } satisfies DocumentTitleHandle,
  },
  ...getPromptPagesRouteDefs(),
];
