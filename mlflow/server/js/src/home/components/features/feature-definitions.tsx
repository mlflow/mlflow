import { BeakerIcon, ForkHorizontalIcon, ModelsIcon, NotebookIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import type { ComponentType, ReactNode } from 'react';

export interface FeatureDefinition {
  id: string;
  icon: ComponentType<{ className?: string; css?: any }>;
  title: ReactNode;
  summary: ReactNode;
  docsLink: string;
  navigationPath: string;
  demoFeatureId?: 'traces' | 'evaluation' | 'prompts';
}

export const featureDefinitions: FeatureDefinition[] = [
  {
    id: 'tracing',
    icon: ForkHorizontalIcon,
    title: <FormattedMessage defaultMessage="Tracing" description="Feature card title for tracing" />,
    summary: (
      <FormattedMessage
        defaultMessage="Capture and analyze LLM interactions, from simple prompts to complex multi-step agents. Debug issues, monitor performance, and understand your AI application's behavior."
        description="Feature card summary for tracing"
      />
    ),
    docsLink: 'https://mlflow.org/docs/latest/llms/tracing/index.html',
    navigationPath: '/experiments',
    demoFeatureId: 'traces',
  },
  {
    id: 'evaluation',
    icon: BeakerIcon,
    title: <FormattedMessage defaultMessage="Evaluation" description="Feature card title for evaluation" />,
    summary: (
      <FormattedMessage
        defaultMessage="Measure and compare LLM quality using built-in and custom scorers. Run offline evaluations to iterate on prompts, models, and retrieval strategies."
        description="Feature card summary for evaluation"
      />
    ),
    docsLink: 'https://mlflow.org/docs/latest/llms/llm-evaluate/index.html',
    navigationPath: '/experiments',
    demoFeatureId: 'evaluation',
  },
  {
    id: 'prompts',
    icon: ModelsIcon,
    title: <FormattedMessage defaultMessage="Prompts" description="Feature card title for prompts" />,
    summary: (
      <FormattedMessage
        defaultMessage="Version control your prompts with full history tracking. Manage aliases, collaborate across teams, and deploy prompt updates without code changes."
        description="Feature card summary for prompts"
      />
    ),
    docsLink: 'https://mlflow.org/docs/latest/genai/prompt-registry/',
    navigationPath: '/prompts',
    demoFeatureId: 'prompts',
  },
  {
    id: 'experiments',
    icon: NotebookIcon,
    title: <FormattedMessage defaultMessage="Model Training" description="Feature card title for model training" />,
    summary: (
      <FormattedMessage
        defaultMessage="Track ML experiments with parameters, metrics, and artifacts. Compare runs, visualize results, and reproduce any model training."
        description="Feature card summary for experiments"
      />
    ),
    docsLink: 'https://mlflow.org/docs/latest/ml/tracking/quickstart/',
    navigationPath: '/experiments',
  },
];
