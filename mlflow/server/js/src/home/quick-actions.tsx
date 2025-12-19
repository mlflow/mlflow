import { BeakerIcon, ModelsIcon, NotebookIcon, WorkflowsIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import type { HomeQuickActionDefinition } from './types';

export const homeQuickActions: HomeQuickActionDefinition[] = [
  {
    id: 'log-traces',
    icon: WorkflowsIcon,
    componentId: 'mlflow.home.quick_action.log_traces',
    title: (
      <FormattedMessage defaultMessage="Log traces" description="Home page quick action title for logging traces" />
    ),
    description: (
      <FormattedMessage
        defaultMessage="Trace LLM applications for debugging and monitoring."
        description="Home page quick action description for logging traces"
      />
    ),
    link: 'https://mlflow.org/docs/latest/llms/tracing/index.html',
  },
  {
    id: 'run-evaluation',
    icon: BeakerIcon,
    componentId: 'mlflow.home.quick_action.run_evaluation',
    title: (
      <FormattedMessage
        defaultMessage="Run evaluation"
        description="Home page quick action title for running evaluations"
      />
    ),
    description: (
      <FormattedMessage
        defaultMessage="Iterate on quality with offline evaluations and comparisons."
        description="Home page quick action description for running evaluations"
      />
    ),
    link: 'https://mlflow.org/docs/latest/llms/llm-evaluate/index.html',
  },
  {
    id: 'train-models',
    icon: NotebookIcon,
    componentId: 'mlflow.home.quick_action.train_models',
    title: (
      <FormattedMessage defaultMessage="Train models" description="Home page quick action title for training models" />
    ),
    description: (
      <FormattedMessage
        defaultMessage="Track experiments, parameters, and metrics throughout training."
        description="Home page quick action description for training models"
      />
    ),
    link: 'https://mlflow.org/docs/latest/ml/tracking/quickstart/',
  },
  {
    id: 'register-prompts',
    icon: ModelsIcon,
    componentId: 'mlflow.home.quick_action.register_prompts',
    title: (
      <FormattedMessage
        defaultMessage="Register prompts"
        description="Home page quick action title for registering prompts"
      />
    ),
    description: (
      <FormattedMessage
        defaultMessage="Manage prompt updates and collaborate across teams."
        description="Home page quick action description for registering prompts"
      />
    ),
    link: 'https://mlflow.org/docs/latest/genai/prompt-registry/',
  },
];
