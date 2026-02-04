import { BeakerIcon, CloudModelIcon, ForkHorizontalIcon, ModelsIcon, NotebookIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import type { ComponentType, ReactNode } from 'react';

export interface FeatureDefinition {
  id: string;
  icon: ComponentType<{ className?: string; css?: any }>;
  title: ReactNode;
  summary: ReactNode;
  docsLink: string;
  hasDrawer?: boolean;
}

export const featureDefinitions: FeatureDefinition[] = [
  {
    id: 'tracing',
    icon: ForkHorizontalIcon,
    title: <FormattedMessage defaultMessage="Tracing" description="Feature card title for tracing" />,
    summary: (
      <FormattedMessage
        defaultMessage="Capture and debug LLM interactions and agent workflows."
        description="Feature card summary for tracing"
      />
    ),
    docsLink: 'https://mlflow.org/docs/latest/llms/tracing/index.html',
    hasDrawer: true,
  },
  {
    id: 'evaluation',
    icon: BeakerIcon,
    title: <FormattedMessage defaultMessage="Evaluation" description="Feature card title for evaluation" />,
    summary: (
      <FormattedMessage
        defaultMessage="Measure and compare LLM quality with built-in and custom scorers."
        description="Feature card summary for evaluation"
      />
    ),
    docsLink: 'https://mlflow.org/docs/latest/llms/llm-evaluate/index.html',
  },
  {
    id: 'prompts',
    icon: ModelsIcon,
    title: <FormattedMessage defaultMessage="Prompts" description="Feature card title for prompts" />,
    summary: (
      <FormattedMessage
        defaultMessage="Version control and manage prompts with aliases across teams."
        description="Feature card summary for prompts"
      />
    ),
    docsLink: 'https://mlflow.org/docs/latest/genai/prompt-registry/',
  },
  {
    id: 'ai-gateway',
    icon: CloudModelIcon,
    title: <FormattedMessage defaultMessage="AI Gateway" description="Feature card title for AI Gateway" />,
    summary: (
      <FormattedMessage
        defaultMessage="Unified API for multiple LLM providers with rate limiting."
        description="Feature card summary for AI Gateway"
      />
    ),
    docsLink: 'https://mlflow.org/docs/latest/llms/gateway/index.html',
  },
  {
    id: 'experiments',
    icon: NotebookIcon,
    title: <FormattedMessage defaultMessage="Model Training" description="Feature card title for model training" />,
    summary: (
      <FormattedMessage
        defaultMessage="Track experiments with parameters, metrics, and artifacts."
        description="Feature card summary for experiments"
      />
    ),
    docsLink: 'https://mlflow.org/docs/latest/ml/tracking/quickstart/',
  },
];
