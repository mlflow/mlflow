import { createLazyRouteElement, RouteHandle, DEFAULT_ASSISTANT_PROMPTS } from '../common/utils/RoutingUtils';

import { PageId, RoutePaths } from './routes';

const getPromptPagesRouteDefs = () => {
  return [
    {
      path: RoutePaths.promptsPage,
      element: createLazyRouteElement(() => import('./pages/prompts/PromptsPage')),
      pageId: PageId.promptsPage,
      handle: {
        getPageTitle: () => 'Prompts',
        getAssistantPrompts: () => [
          'How do I create a new prompt?',
          'How do I version my prompts?',
          'How do I use prompts with my LLM application?',
        ],
      } satisfies RouteHandle,
    },
    {
      path: RoutePaths.promptDetailsPage,
      element: createLazyRouteElement(() => import('./pages/prompts/PromptsDetailsPage')),
      pageId: PageId.promptDetailsPage,
      handle: {
        getPageTitle: (params) => `Prompt: ${params['promptName']}`,
        getAssistantPrompts: () => [
          'Explain what this prompt does.',
          'How can I improve this prompt?',
          'Show me how to use this prompt in code.',
        ],
      } satisfies RouteHandle,
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
      handle: {
        getPageTitle: () => 'Experiments',
        getAssistantPrompts: () => [
          'How do I create a new experiment?',
          'How do I organize my experiments?',
          'Show me my recent experiments.',
        ],
      } satisfies RouteHandle,
    },
    {
      path: RoutePaths.experimentPage,
      element: createLazyRouteElement(() => {
        return import('./pages/experiment-page-tabs/ExperimentPageTabs');
      }),
      pageId: PageId.experimentPage,
      handle: { getPageTitle: (params) => `Experiment ${params['experimentId']}` } satisfies RouteHandle,
      children: [
        {
          path: RoutePaths.experimentPageTabOverview,
          pageId: PageId.experimentPageTabOverview,
          element: createLazyRouteElement(() => import('./pages/experiment-overview/ExperimentGenAIOverviewPage')),
          handle: {
            getPageTitle: (params) => `Overview - Experiment ${params['experimentId']}`,
            getAssistantPrompts: () => [
              'How do I get started with MLflow GenAI?',
              'What is the trend of token usage?',
              'Why did the error rate spike?',
            ],
          } satisfies RouteHandle,
        },
        {
          path: RoutePaths.experimentPageTabRuns,
          pageId: PageId.experimentPageTabRuns,
          element: createLazyRouteElement(() => import('./pages/experiment-runs/ExperimentRunsPage')),
          handle: {
            getPageTitle: (params) => `Runs - Experiment ${params['experimentId']}`,
            getAssistantPrompts: () => [
              'Compare the selected runs.',
              'Which run performed best?',
              'What parameters differ between runs?',
            ],
          } satisfies RouteHandle,
        },
        {
          path: RoutePaths.experimentPageTabTraces,
          pageId: PageId.experimentPageTabTraces,
          element: createLazyRouteElement(() => import('./pages/experiment-traces/ExperimentTracesPage')),
          handle: {
            getPageTitle: (params) => `Traces - Experiment ${params['experimentId']}`,
            getAssistantPrompts: () => [
              'Debug the error in this trace.',
              'What is the performance bottleneck in this trace?',
              'How to measure the quality of my agent with traces?',
            ],
          } satisfies RouteHandle,
        },
        {
          path: RoutePaths.experimentPageTabChatSessions,
          pageId: PageId.experimentPageTabChatSessions,
          element: createLazyRouteElement(() => import('./pages/experiment-chat-sessions/ExperimentChatSessionsPage')),
          handle: {
            getPageTitle: (params) => `Chat Sessions - Experiment ${params['experimentId']}`,
            getAssistantPrompts: () => [
              'How to group multiple traces by session?',
              'What topics were discussed in this session?',
              'Were there any issues in this conversation?',
            ],
          } satisfies RouteHandle,
        },
        {
          path: RoutePaths.experimentPageTabSingleChatSession,
          pageId: PageId.experimentPageTabSingleChatSession,
          element: createLazyRouteElement(
            () => import('./pages/experiment-chat-sessions/single-chat-view/ExperimentSingleChatSessionPage'),
          ),
          handle: {
            getPageTitle: (params) => `Chat Session ${params['sessionId']}`,
            getAssistantPrompts: () => [
              'Summarize this chat session.',
              'What was the outcome of this conversation?',
              'Identify any issues in this chat.',
            ],
          } satisfies RouteHandle,
        },
        {
          path: RoutePaths.experimentPageTabModels,
          pageId: PageId.experimentPageTabModels,
          element: createLazyRouteElement(
            () => import('./pages/experiment-logged-models/ExperimentLoggedModelListPage'),
          ),
          handle: {
            getPageTitle: (params) => `Logged Models - Experiment ${params['experimentId']}`,
            getAssistantPrompts: () => [
              'Which model should I use for production?',
              'Compare these models.',
              'What are the differences between model versions?',
            ],
          } satisfies RouteHandle,
        },
        {
          path: RoutePaths.experimentPageTabEvaluationRuns,
          pageId: PageId.experimentPageTabEvaluationRuns,
          element: createLazyRouteElement(
            () => import('./pages/experiment-evaluation-runs/ExperimentEvaluationRunsPage'),
          ),
          handle: {
            getPageTitle: (params) => `Evaluation Runs - Experiment ${params['experimentId']}`,
            getAssistantPrompts: () => [
              'Summarize the evaluation results.',
              'Which model performed best in evaluation?',
              'What metrics should I focus on?',
            ],
          } satisfies RouteHandle,
        },
        {
          path: RoutePaths.experimentPageTabScorers,
          pageId: PageId.experimentPageTabScorers,
          element: createLazyRouteElement(() => import('./pages/experiment-scorers/ExperimentScorersPage')),
          handle: {
            getPageTitle: (params) => `Scorers - Experiment ${params['experimentId']}`,
            getAssistantPrompts: () => [
              'How do I create LLM judge for testing the quality of my agent?',
              'Which built-in LLM judges should I use for my project?',
              'How to run my judges on traces?',
            ],
          } satisfies RouteHandle,
        },
        {
          path: RoutePaths.experimentPageTabDatasets,
          pageId: PageId.experimentPageTabDatasets,
          element: createLazyRouteElement(() => {
            return import('./pages/experiment-evaluation-datasets/ExperimentEvaluationDatasetsPage');
          }),
          handle: {
            getPageTitle: (params) => `Datasets - Experiment ${params['experimentId']}`,
            getAssistantPrompts: () => [
              'How to add a new record to a dataset?',
              'How do I use this dataset for evaluation?',
              'What format should my dataset be in?',
            ],
          } satisfies RouteHandle,
        },
        {
          path: RoutePaths.experimentPageTabPrompts,
          pageId: PageId.experimentPageTabPrompts,
          element: createLazyRouteElement(() => import('./pages/prompts/ExperimentPromptsPage')),
          handle: {
            getPageTitle: (params) => `Prompts - Experiment ${params['experimentId']}`,
            getAssistantPrompts: () => [
              'How do I create a new prompt?',
              'Which prompt performs best for my project?',
              'How do I use registered prompts with my LLM application?',
            ],
          } satisfies RouteHandle,
        },
        {
          path: RoutePaths.experimentPageTabPromptDetails,
          pageId: PageId.experimentPageTabPromptDetails,
          element: createLazyRouteElement(() => import('./pages/prompts/ExperimentPromptDetailsPage')),
          handle: {
            getPageTitle: (params) => `Prompt: ${params['promptName']}`,
            getAssistantPrompts: () => [
              'Explain what this prompt does.',
              'How can I improve this prompt?',
              'Show me how to use this prompt in code.',
            ],
          } satisfies RouteHandle,
        },
        {
          path: RoutePaths.experimentPageTabPromptOptimization,
          pageId: PageId.experimentPageTabPromptOptimization,
          element: createLazyRouteElement(() => import('./pages/prompt-optimization/ExperimentPromptOptimizationPage')),
          handle: {
            getPageTitle: (params) => `Prompt Optimization - Experiment ${params['experimentId']}`,
          } satisfies DocumentTitleHandle,
        },
        {
          path: RoutePaths.experimentPageTabPromptOptimizationDetails,
          pageId: PageId.experimentPageTabPromptOptimizationDetails,
          element: createLazyRouteElement(
            () => import('./pages/prompt-optimization/ExperimentPromptOptimizationDetailsPage'),
          ),
          handle: {
            getPageTitle: (params) => `Optimization Job ${params['jobId']}`,
          } satisfies DocumentTitleHandle,
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
    handle: {
      getPageTitle: () => 'Home',
      getAssistantPrompts: () => DEFAULT_ASSISTANT_PROMPTS,
    } satisfies RouteHandle,
  },
  {
    path: RoutePaths.settingsPage,
    element: createLazyRouteElement(() => import('../settings/SettingsPage')),
    pageId: PageId.settingsPage,
    handle: { getPageTitle: () => 'Settings' } satisfies RouteHandle,
  },
  ...getExperimentPageRouteDefs(),
  {
    path: RoutePaths.experimentLoggedModelDetailsPageTab,
    element: createLazyRouteElement(() => import('./pages/experiment-logged-models/ExperimentLoggedModelDetailsPage')),
    pageId: PageId.experimentLoggedModelDetailsPageTab,
    handle: {
      getPageTitle: (params) => `Model ${params['loggedModelId']}`,
      getAssistantPrompts: () => [
        'Explain this model.',
        'How do I deploy this model?',
        'What are the model inputs and outputs?',
      ],
    } satisfies RouteHandle,
  },
  {
    path: RoutePaths.experimentLoggedModelDetailsPage,
    element: createLazyRouteElement(() => import('./pages/experiment-logged-models/ExperimentLoggedModelDetailsPage')),
    pageId: PageId.experimentLoggedModelDetailsPage,
    handle: {
      getPageTitle: (params) => `Model ${params['loggedModelId']}`,
      getAssistantPrompts: () => [
        'Explain this model.',
        'How do I deploy this model?',
        'What are the model inputs and outputs?',
      ],
    } satisfies RouteHandle,
  },
  {
    path: RoutePaths.compareExperimentsSearch,
    element: createLazyRouteElement(
      () => import(/* webpackChunkName: "experimentPage" */ './components/experiment-page/ExperimentPage'),
    ),
    pageId: PageId.compareExperimentsSearch,
    handle: {
      getPageTitle: () => 'Compare Experiments',
      getAssistantPrompts: () => [
        'Compare these experiments.',
        'Which experiment performed best?',
        'What are the key differences?',
      ],
    } satisfies RouteHandle,
  },
  {
    path: RoutePaths.runPageWithTab,
    element: createLazyRouteElement(() => import('./components/run-page/RunPage')),
    pageId: PageId.runPageWithTab,
    handle: {
      getPageTitle: (params) => `Run ${params['runUuid']}`,
      getAssistantPrompts: () => [
        'Summarize this run.',
        'What parameters were used?',
        'How does this run compare to others?',
      ],
    } satisfies RouteHandle,
  },
  {
    path: RoutePaths.runPageDirect,
    element: createLazyRouteElement(() => import('./components/DirectRunPage')),
    pageId: PageId.runPageDirect,
    handle: {
      getPageTitle: (params) => `Run ${params['runUuid']}`,
      getAssistantPrompts: () => [
        'Summarize this run.',
        'What parameters were used?',
        'How does this run compare to others?',
      ],
    } satisfies RouteHandle,
  },
  {
    path: RoutePaths.compareRuns,
    element: createLazyRouteElement(() => import('./components/CompareRunPage')),
    pageId: PageId.compareRuns,
    handle: {
      getPageTitle: () => 'Compare Runs',
      getAssistantPrompts: () => [
        'Compare these runs.',
        'Which run performed best?',
        'What parameters differ between runs?',
      ],
    } satisfies RouteHandle,
  },
  {
    path: RoutePaths.metricPage,
    element: createLazyRouteElement(() => import('./components/MetricPage')),
    pageId: PageId.metricPage,
    handle: {
      getPageTitle: () => 'Metric Details',
      getAssistantPrompts: () => ['Explain this metric.', 'What trends do you see?', 'How can I improve this metric?'],
    } satisfies RouteHandle,
  },
  ...getPromptPagesRouteDefs(),
];
