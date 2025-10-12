import { FormattedMessage } from 'react-intl';
import { AssistantIcon, NotebookIcon, RobotIcon, DatabaseIcon } from '@databricks/design-system';
import type { HomeNewsItemDefinition } from './types';

export const homeNewsItems: HomeNewsItemDefinition[] = [
  {
    id: 'mlflow-mcp-server',
    title: <FormattedMessage defaultMessage="MLflow MCP server" description="Home page news card title one" />,
    description:
      'Connect your coding assistants and AI applications to MLflow and automatically analyze your experiments and traces.',
    link: 'https://mlflow.org/docs/latest/genai/mcp/index.html',
    componentId: 'mlflow.home.news.auto_tune_llm_judge',
    thumbnail: {
      gradient: 'linear-gradient(135deg, #E9F2FF 0%, #F4EBFF 100%)',
      icon: AssistantIcon,
    },
  },
  {
    id: 'optimize-prompts',
    title: <FormattedMessage defaultMessage="Optimize prompts" description="Home page news card title two" />,
    description:
      'Access the state-of-the-art prompt optimization algorithms such as MIPROv2, GEPA, through MLflow Prompt Registry.',
    link: 'https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts',
    componentId: 'mlflow.home.news.optimize_prompts',
    thumbnail: {
      gradient: 'linear-gradient(135deg, #E8F7F2 0%, #D5E8FF 100%)',
      icon: NotebookIcon,
    },
  },
  {
    id: 'agents-as-a-judge',
    title: <FormattedMessage defaultMessage="Agents-as-a-judge" description="Home page news card title three" />,
    description: 'Leverage agents as a judge to perform deep trace analysis and improve your evaluation accuracy.',
    link: 'https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/agentic-overview/index.html',
    componentId: 'mlflow.home.news.agents_as_a_judge',
    thumbnail: {
      gradient: 'linear-gradient(135deg, #FFF5E1 0%, #FFE2F2 100%)',
      icon: RobotIcon,
    },
  },
  {
    id: 'dataset-tracking',
    title: <FormattedMessage defaultMessage="Dataset tracking" description="Home page news card title four" />,
    description: 'Track dataset lineage and versions and effectively drive the quality improvement loop.',
    link: 'https://mlflow.org/docs/latest/genai/datasets/',
    componentId: 'mlflow.home.news.dataset_tracking',
    thumbnail: {
      gradient: 'linear-gradient(135deg, #E6F3FF 0%, #E0F1F6 100%)',
      icon: DatabaseIcon,
    },
  },
];
