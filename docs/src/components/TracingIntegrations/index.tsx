import React from 'react';
import { CardGroup, SmallLogoCard } from '../Card';
import useBaseUrl from '@docusaurus/useBaseUrl';

interface TracingIntegration {
  id: string;
  name: string;
  logoPath: string;
  link: string;
  category: string;
}

interface TracingIntegrationsProps {
  cardGroupProps?: {
    isSmall?: boolean;
    cols?: number;
    noGap?: boolean;
  };
  category?: Category;
}

type Category =
  | 'OpenTelemetry'
  | 'Agent Frameworks (Python)'
  | 'Agent Frameworks (TypeScript)'
  | 'Model Providers'
  | 'Tools';

const CATEGORY_ORDER: Category[] = [
  'OpenTelemetry',
  'Agent Frameworks (Python)',
  'Agent Frameworks (TypeScript)',
  'Model Providers',
  'Tools',
];

// Centralized integration definitions with categories
const TRACING_INTEGRATIONS: TracingIntegration[] = [
  // OpenTelemetry
  {
    id: 'opentelemetry',
    name: 'OpenTelemetry',
    logoPath: '/images/logos/opentelemetry-logo-only.png',
    link: '/genai/tracing/app-instrumentation/opentelemetry',
    category: 'OpenTelemetry',
  },
  // Agent Frameworks (Python)
  {
    id: 'langchain',
    name: 'LangChain',
    logoPath: '/images/logos/langchain-logo-only.png',
    link: '/genai/tracing/integrations/listing/langchain',
    category: 'Agent Frameworks (Python)',
  },
  {
    id: 'langgraph',
    name: 'LangGraph',
    logoPath: '/images/logos/langgraph-logo-only.png',
    link: '/genai/tracing/integrations/listing/langgraph',
    category: 'Agent Frameworks (Python)',
  },
  {
    id: 'openai-agent',
    name: 'OpenAI Agent',
    logoPath: '/images/logos/openai-logo-only.png',
    link: '/genai/tracing/integrations/listing/openai-agent',
    category: 'Agent Frameworks (Python)',
  },
  {
    id: 'dspy',
    name: 'DSPy',
    logoPath: '/images/logos/dspy-logo.png',
    link: '/genai/tracing/integrations/listing/dspy',
    category: 'Agent Frameworks (Python)',
  },
  {
    id: 'pydantic_ai',
    name: 'PydanticAI',
    logoPath: '/images/logos/pydantic-ai-logo-only.png',
    link: '/genai/tracing/integrations/listing/pydantic_ai',
    category: 'Agent Frameworks (Python)',
  },
  {
    id: 'google-adk',
    name: 'Google ADK',
    logoPath: '/images/logos/google-adk-logo.png',
    link: '/genai/tracing/integrations/listing/google-adk',
    category: 'Agent Frameworks (Python)',
  },
  {
    id: 'microsoft-agent-framework',
    name: 'Microsoft Agent Framework',
    logoPath: '/images/logos/microsoft-agent-framework-logo.png',
    link: '/genai/tracing/integrations/listing/microsoft-agent-framework',
    category: 'Agent Frameworks (Python)',
  },
  {
    id: 'crewai',
    name: 'CrewAI',
    logoPath: '/images/logos/crewai-logo.svg',
    link: '/genai/tracing/integrations/listing/crewai',
    category: 'Agent Frameworks (Python)',
  },
  {
    id: 'llama_index',
    name: 'LlamaIndex',
    logoPath: '/images/logos/llamaindex-logo.svg',
    link: '/genai/tracing/integrations/listing/llama_index',
    category: 'Agent Frameworks (Python)',
  },
  {
    id: 'autogen',
    name: 'AutoGen',
    logoPath: '/images/logos/autogen-logo.png',
    link: '/genai/tracing/integrations/listing/autogen',
    category: 'Agent Frameworks (Python)',
  },
  {
    id: 'strands',
    name: 'Strands Agent SDK',
    logoPath: '/images/logos/strands-logo.png',
    link: '/genai/tracing/integrations/listing/strands',
    category: 'Agent Frameworks (Python)',
  },
  {
    id: 'agno',
    name: 'Agno',
    logoPath: '/images/logos/agno-logo.png',
    link: '/genai/tracing/integrations/listing/agno',
    category: 'Agent Frameworks (Python)',
  },
  {
    id: 'smolagents',
    name: 'Smolagents',
    logoPath: '/images/logos/smolagents-logo.png',
    link: '/genai/tracing/integrations/listing/smolagents',
    category: 'Agent Frameworks (Python)',
  },

  {
    id: 'semantic_kernel',
    name: 'Semantic Kernel',
    logoPath: '/images/logos/semantic-kernel-logo.png',
    link: '/genai/tracing/integrations/listing/semantic_kernel',
    category: 'Agent Frameworks (Python)',
  },
  {
    id: 'ag2',
    name: 'AG2',
    logoPath: '/images/logos/ag2-logo.png',
    link: '/genai/tracing/integrations/listing/ag2',
    category: 'Agent Frameworks (Python)',
  },
  {
    id: 'haystack',
    name: 'Haystack',
    logoPath: '/images/logos/haystack-logo.png',
    link: '/genai/tracing/integrations/listing/haystack',
    category: 'Agent Frameworks (Python)',
  },
  {
    id: 'spring-ai',
    name: 'Spring AI',
    logoPath: '/images/logos/spring-ai-logo.png',
    link: '/genai/tracing/integrations/listing/spring-ai',
    category: 'Agent Frameworks (Java)',
  },
  {
    id: 'txtai',
    name: 'txtai',
    logoPath: '/images/logos/txtai-logo.png',
    link: '/genai/tracing/integrations/listing/txtai',
    category: 'Agent Frameworks (Python)',
  },
  // Agent Frameworks (TypeScript)
  {
    id: 'langchain-ts',
    name: 'LangChain',
    logoPath: '/images/logos/langchain-logo-only.png',
    link: '/genai/tracing/integrations/listing/langchain',
    category: 'Agent Frameworks (TypeScript)',
  },
  {
    id: 'langgraph-ts',
    name: 'LangGraph',
    logoPath: '/images/logos/langgraph-logo-only.png',
    link: '/genai/tracing/integrations/listing/langgraph',
    category: 'Agent Frameworks (TypeScript)',
  },
  {
    id: 'vercelai',
    name: 'Vercel AI SDK',
    logoPath: '/images/logos/vercel-logo.svg',
    link: '/genai/tracing/integrations/listing/vercelai',
    category: 'Agent Frameworks (TypeScript)',
  },
  {
    id: 'mastra',
    name: 'Mastra',
    logoPath: '/images/logos/mastra-logo.png',
    link: '/genai/tracing/integrations/listing/mastra',
    category: 'Agent Frameworks (TypeScript)',
  },
  {
    id: 'voltagent',
    name: 'VoltAgent',
    logoPath: '/images/logos/voltagent-logo.png',
    link: '/genai/tracing/integrations/listing/voltagent',
    category: 'Agent Frameworks (TypeScript)',
  },
  // Model Providers
  {
    id: 'openai',
    name: 'OpenAI',
    logoPath: '/images/logos/openai-logo-only.png',
    link: '/genai/tracing/integrations/listing/openai',
    category: 'Model Providers',
  },
  {
    id: 'anthropic',
    name: 'Anthropic',
    logoPath: '/images/logos/anthropic-logo.png',
    link: '/genai/tracing/integrations/listing/anthropic',
    category: 'Model Providers',
  },
  {
    id: 'bedrock',
    name: 'Amazon Bedrock',
    logoPath: '/images/logos/bedrock-logo.png',
    link: '/genai/tracing/integrations/listing/bedrock',
    category: 'Model Providers',
  },
  {
    id: 'gemini',
    name: 'Gemini',
    logoPath: '/images/logos/google-gemini-logo.svg',
    link: '/genai/tracing/integrations/listing/gemini',
    category: 'Model Providers',
  },
  {
    id: 'litellm',
    name: 'LiteLLM',
    logoPath: '/images/logos/litellm-logo.png',
    link: '/genai/tracing/integrations/listing/litellm',
    category: 'Model Providers',
  },
  {
    id: 'mistral',
    name: 'Mistral',
    logoPath: '/images/logos/mistral-ai-logo.svg',
    link: '/genai/tracing/integrations/listing/mistral',
    category: 'Model Providers',
  },
  {
    id: 'xai-grok',
    name: 'xAI / Grok',
    logoPath: '/images/logos/grok-logo.png',
    link: '/genai/tracing/integrations/listing/xai-grok',
    category: 'Model Providers',
  },
  {
    id: 'databricks',
    name: 'Databricks',
    logoPath: '/images/logos/databricks-logo.png',
    link: '/genai/tracing/integrations/listing/databricks',
    category: 'Model Providers',
  },
  {
    id: 'ollama',
    name: 'Ollama',
    logoPath: '/images/logos/ollama-logo.png',
    link: '/genai/tracing/integrations/listing/ollama',
    category: 'Model Providers',
  },
  {
    id: 'groq',
    name: 'Groq',
    logoPath: '/images/logos/groq-logo.svg',
    link: '/genai/tracing/integrations/listing/groq',
    category: 'Model Providers',
  },
  {
    id: 'deepseek',
    name: 'DeepSeek',
    logoPath: '/images/logos/deepseek-logo.png',
    link: '/genai/tracing/integrations/listing/deepseek',
    category: 'Model Providers',
  },
  {
    id: 'qwen',
    name: 'Qwen',
    logoPath: '/images/logos/qwen-logo.jpg',
    link: '/genai/tracing/integrations/listing/qwen',
    category: 'Model Providers',
  },
  {
    id: 'moonshot',
    name: 'Moonshot AI',
    logoPath: '/images/logos/kimi-logo.png',
    link: '/genai/tracing/integrations/listing/moonshot',
    category: 'Model Providers',
  },
  {
    id: 'cohere',
    name: 'Cohere',
    logoPath: '/images/logos/cohere-logo.png',
    link: '/genai/tracing/integrations/listing/cohere',
    category: 'Model Providers',
  },
  {
    id: 'byteplus',
    name: 'BytePlus',
    logoPath: '/images/logos/byteplus-logo.png',
    link: '/genai/tracing/integrations/listing/byteplus',
    category: 'Model Providers',
  },
  {
    id: 'novitaai',
    name: 'Novita AI',
    logoPath: '/images/logos/novitaai-logo.jpg',
    link: '/genai/tracing/integrations/listing/novitaai',
    category: 'Model Providers',
  },
  {
    id: 'fireworksai',
    name: 'FireworksAI',
    logoPath: '/images/logos/fireworks-ai-logo.png',
    link: '/genai/tracing/integrations/listing/fireworksai',
    category: 'Model Providers',
  },
  {
    id: 'togetherai',
    name: 'Together AI',
    logoPath: '/images/logos/together-ai-logo.png',
    link: '/genai/tracing/integrations/listing/togetherai',
    category: 'Model Providers',
  },
  // Tools
  {
    id: 'instructor',
    name: 'Instructor',
    logoPath: '/images/logos/instructor-logo.svg',
    link: '/genai/tracing/integrations/listing/instructor',
    category: 'Tools',
  },
  {
    id: 'claude_code',
    name: 'Claude Code',
    logoPath: '/images/logos/claude-code-logo.png',
    link: '/genai/tracing/integrations/listing/claude_code',
    category: 'Tools',
  },
];

const IntegrationSection: React.FC<{
  title: string;
  integrations: TracingIntegration[];
  cardGroupProps?: TracingIntegrationsProps['cardGroupProps'];
}> = ({ integrations, cardGroupProps = {} }) => {
  if (integrations.length === 0) return null;

  return (
    <div style={{ marginBottom: '2rem' }}>
      <CardGroup isSmall={cardGroupProps.isSmall} cols={cardGroupProps.cols} noGap={cardGroupProps.noGap}>
        {integrations.map((integration) => (
          <SmallLogoCard key={integration.id} link={integration.link} title={integration.name}>
            <span>
              <img src={useBaseUrl(integration.logoPath)} alt={`${integration.name} Logo`} />
            </span>
          </SmallLogoCard>
        ))}
      </CardGroup>
    </div>
  );
};

const getIntegrations = (predicate: (integration: TracingIntegration) => boolean) =>
  TRACING_INTEGRATIONS.filter(predicate);

export const TracingIntegrationsSection: React.FC<TracingIntegrationsProps> = ({ category, cardGroupProps = {} }) => {
  const targetCategories = category ? [category] : CATEGORY_ORDER;
  const sections = targetCategories
    .map((cat) => ({
      title: cat,
      integrations: getIntegrations((integration) => integration.category === cat),
    }))
    .filter(({ integrations }) => integrations.length > 0);

  return (
    <>
      {sections.map(({ title, integrations }) => (
        <IntegrationSection key={title} title={title} integrations={integrations} cardGroupProps={cardGroupProps} />
      ))}
    </>
  );
};

export const TracingIntegrations: React.FC<TracingIntegrationsProps> = ({ cardGroupProps = {}, category }) => (
  <TracingIntegrationsSection category={category} cardGroupProps={cardGroupProps} />
);

export default TracingIntegrations;
