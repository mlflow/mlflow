import React from 'react';
import { CardGroup, SmallLogoCard } from '../Card';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import TabsWrapper from '../TabsWrapper';

interface TracingIntegration {
  id: string;
  name: string;
  logoPath: string;
  link: string;
  category: string;
  languages: ('python' | 'typescript')[];
}

// Centralized integration definitions with categories
const TRACING_INTEGRATIONS: TracingIntegration[] = [
  // Agent Frameworks
  {
    id: 'langchain',
    name: 'LangChain',
    logoPath: '/images/logos/langchain-logo.png',
    link: '/genai/tracing/integrations/listing/langchain',
    category: 'Agent Frameworks',
    languages: ['python', 'typescript'],
  },
  {
    id: 'langgraph',
    name: 'LangGraph',
    logoPath: '/images/logos/langgraph-logo.png',
    link: '/genai/tracing/integrations/listing/langgraph',
    category: 'Agent Frameworks',
    languages: ['python', 'typescript'],
  },
  {
    id: 'vercelai',
    name: 'Vercel AI SDK',
    logoPath: '/images/logos/vercel-logo.svg',
    link: '/genai/tracing/integrations/listing/vercelai',
    category: 'Agent Frameworks',
    languages: ['typescript'],
  },
  {
    id: 'openai-agent',
    name: 'OpenAI Agent',
    logoPath: '/images/logos/openai-agent-logo.png',
    link: '/genai/tracing/integrations/listing/openai-agent',
    category: 'Agent Frameworks',
    languages: ['python'],
  },
  {
    id: 'dspy',
    name: 'DSPy',
    logoPath: '/images/logos/dspy-logo.png',
    link: '/genai/tracing/integrations/listing/dspy',
    category: 'Agent Frameworks',
    languages: ['python'],
  },
  {
    id: 'pydantic_ai',
    name: 'PydanticAI',
    logoPath: '/images/logos/pydanticai-logo.png',
    link: '/genai/tracing/integrations/listing/pydantic_ai',
    category: 'Agent Frameworks',
    languages: ['python'],
  },
  {
    id: 'google-adk',
    name: 'Google ADK',
    logoPath: '/images/logos/google-adk-logo.png',
    link: '/genai/tracing/integrations/listing/google-adk',
    category: 'Agent Frameworks',
    languages: ['python'],
  },
  {
    id: 'microsoft-agent-framework',
    name: 'Microsoft Agent Framework',
    logoPath: '/images/logos/microsoft-agent-framework-logo.jpg',
    link: '/genai/tracing/integrations/listing/microsoft-agent-framework',
    category: 'Agent Frameworks',
    languages: ['python'],
  },
  {
    id: 'crewai',
    name: 'CrewAI',
    logoPath: '/images/logos/crewai-logo.png',
    link: '/genai/tracing/integrations/listing/crewai',
    category: 'Agent Frameworks',
    languages: ['python'],
  },
  {
    id: 'llama_index',
    name: 'LlamaIndex',
    logoPath: '/images/logos/llamaindex-logo.svg',
    link: '/genai/tracing/integrations/listing/llama_index',
    category: 'Agent Frameworks',
    languages: ['python'],
  },
  {
    id: 'autogen',
    name: 'AutoGen',
    logoPath: '/images/logos/autogen-logo.png',
    link: '/genai/tracing/integrations/listing/autogen',
    category: 'Agent Frameworks',
    languages: ['python'],
  },
  {
    id: 'strands',
    name: 'Strands Agent SDK',
    logoPath: '/images/logos/strands-logo.png',
    link: '/genai/tracing/integrations/listing/strands',
    category: 'Agent Frameworks',
    languages: ['python'],
  },
  {
    id: 'mastra',
    name: 'Mastra',
    logoPath: '/images/logos/mastra-logo.png',
    link: '/genai/tracing/integrations/listing/mastra',
    category: 'Agent Frameworks',
    languages: ['typescript'],
  },
  {
    id: 'voltagent',
    name: 'VoltAgent',
    logoPath: '/images/logos/voltagent-logo.png',
    link: '/genai/tracing/integrations/listing/voltagent',
    category: 'Agent Frameworks',
    languages: ['typescript'],
  },
  {
    id: 'agno',
    name: 'Agno',
    logoPath: '/images/logos/agno-logo.png',
    link: '/genai/tracing/integrations/listing/agno',
    category: 'Agent Frameworks',
    languages: ['python'],
  },
  {
    id: 'smolagents',
    name: 'Smolagents',
    logoPath: '/images/logos/smolagents-logo.png',
    link: '/genai/tracing/integrations/listing/smolagents',
    category: 'Agent Frameworks',
    languages: ['python'],
  },

  {
    id: 'semantic_kernel',
    name: 'Semantic Kernel',
    logoPath: '/images/logos/semantic-kernel-logo.png',
    link: '/genai/tracing/integrations/listing/semantic_kernel',
    category: 'Agent Frameworks',
    languages: ['python'],
  },
  {
    id: 'ag2',
    name: 'AG2',
    logoPath: '/images/logos/ag2-logo.png',
    link: '/genai/tracing/integrations/listing/ag2',
    category: 'Agent Frameworks',
    languages: ['python'],
  },
  {
    id: 'haystack',
    name: 'Haystack',
    logoPath: '/images/logos/haystack-logo.png',
    link: '/genai/tracing/integrations/listing/haystack',
    category: 'Agent Frameworks',
    languages: ['python'],
  },
  {
    id: 'instructor',
    name: 'Instructor',
    logoPath: '/images/logos/instructor-logo.svg',
    link: '/genai/tracing/integrations/listing/instructor',
    category: 'Tools',
    languages: ['python'],
  },
  {
    id: 'txtai',
    name: 'txtai',
    logoPath: '/images/logos/txtai-logo.png',
    link: '/genai/tracing/integrations/listing/txtai',
    category: 'Agent Frameworks',
    languages: ['python'],
  },
  // Model Providers
  {
    id: 'openai',
    name: 'OpenAI',
    logoPath: '/images/logos/openai-logo.png',
    link: '/genai/tracing/integrations/listing/openai',
    category: 'Model Providers',
    languages: ['python', 'typescript'],
  },
  {
    id: 'anthropic',
    name: 'Anthropic',
    logoPath: '/images/logos/anthropic-logo.svg',
    link: '/genai/tracing/integrations/listing/anthropic',
    category: 'Model Providers',
    languages: ['python', 'typescript'],
  },
  {
    id: 'bedrock',
    name: 'Bedrock',
    logoPath: '/images/logos/bedrock-logo.png',
    link: '/genai/tracing/integrations/listing/bedrock',
    category: 'Model Providers',
    languages: ['python'],
  },
  {
    id: 'gemini',
    name: 'Gemini',
    logoPath: '/images/logos/google-gemini-logo.svg',
    link: '/genai/tracing/integrations/listing/gemini',
    category: 'Model Providers',
    languages: ['python', 'typescript'],
  },
  {
    id: 'ollama',
    name: 'Ollama',
    logoPath: '/images/logos/ollama-logo.png',
    link: '/genai/tracing/integrations/listing/ollama',
    category: 'Model Providers',
    languages: ['python'],
  },
  {
    id: 'groq',
    name: 'Groq',
    logoPath: '/images/logos/groq-logo.svg',
    link: '/genai/tracing/integrations/listing/groq',
    category: 'Model Providers',
    languages: ['python'],
  },
  {
    id: 'mistral',
    name: 'Mistral',
    logoPath: '/images/logos/mistral-ai-logo.svg',
    link: '/genai/tracing/integrations/listing/mistral',
    category: 'Model Providers',
    languages: ['python'],
  },
  {
    id: 'fireworksai',
    name: 'FireworksAI',
    logoPath: '/images/logos/fireworks-ai-logo.svg',
    link: '/genai/tracing/integrations/listing/fireworksai',
    category: 'Model Providers',
    languages: ['python'],
  },
  {
    id: 'deepseek',
    name: 'DeepSeek',
    logoPath: '/images/logos/deepseek-logo.png',
    link: '/genai/tracing/integrations/listing/deepseek',
    category: 'Model Providers',
    languages: ['python'],
  },
  {
    id: 'litellm',
    name: 'LiteLLM',
    logoPath: '/images/logos/litellm-logo.jpg',
    link: '/genai/tracing/integrations/listing/litellm',
    category: 'Model Providers',
    languages: ['python'],
  },
  // Tools
  {
    id: 'claude_code',
    name: 'Claude Code',
    logoPath: '/images/logos/claude-code-logo.svg',
    link: '/genai/tracing/integrations/listing/claude_code',
    category: 'Tools',
    languages: ['python'],
  },
];

interface TracingIntegrationsProps {
  cardGroupProps?: {
    isSmall?: boolean;
    cols?: number;
    noGap?: boolean;
  };
  categorized?: boolean;
}

const IntegrationContent: React.FC<{
  integrations: TracingIntegration[];
  cardGroupProps: TracingIntegrationsProps['cardGroupProps'];
  categorized: boolean;
}> = ({ integrations, cardGroupProps = {}, categorized }) => {
  if (categorized) {
    // Group integrations by category
    const categories = integrations.reduce(
      (acc, integration) => {
        if (!acc[integration.category]) {
          acc[integration.category] = [];
        }
        acc[integration.category].push(integration);
        return acc;
      },
      {} as Record<string, TracingIntegration[]>,
    );

    // Define category order
    const categoryOrder = ['Agent Frameworks', 'Model Providers', 'Tools'];

    return (
      <>
        {categoryOrder.map((category) => {
          const categoryIntegrations = categories[category] || [];
          if (categoryIntegrations.length === 0) return null;

          return (
            <div key={category} style={{ marginBottom: '2rem' }}>
              <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>{category}</h2>
              <CardGroup isSmall={cardGroupProps.isSmall} cols={cardGroupProps.cols} noGap={cardGroupProps.noGap}>
                {categoryIntegrations.map((integration) => (
                  <SmallLogoCard key={integration.id} link={integration.link}>
                    <span>
                      <img src={useBaseUrl(integration.logoPath)} alt={`${integration.name} Logo`} />
                    </span>
                  </SmallLogoCard>
                ))}
              </CardGroup>
            </div>
          );
        })}
      </>
    );
  }

  return (
    <CardGroup isSmall={cardGroupProps.isSmall} cols={cardGroupProps.cols} noGap={cardGroupProps.noGap}>
      {integrations.map((integration) => (
        <SmallLogoCard key={integration.id} link={integration.link}>
          <span>
            <img src={useBaseUrl(integration.logoPath)} alt={`${integration.name} Logo`} />
          </span>
        </SmallLogoCard>
      ))}
    </CardGroup>
  );
};

export const TracingIntegrations: React.FC<TracingIntegrationsProps> = ({
  cardGroupProps = {},
  categorized = false,
}) => {
  const pythonIntegrations = TRACING_INTEGRATIONS.filter((integration) => integration.languages.includes('python'));
  const typescriptIntegrations = TRACING_INTEGRATIONS.filter((integration) =>
    integration.languages.includes('typescript'),
  );

  return (
    <TabsWrapper>
      <Tabs groupId="programming-language">
        <TabItem value="python" label="Python" default>
          <IntegrationContent
            integrations={pythonIntegrations}
            cardGroupProps={cardGroupProps}
            categorized={categorized}
          />
        </TabItem>
        <TabItem value="typescript" label="TypeScript">
          <IntegrationContent
            integrations={typescriptIntegrations}
            cardGroupProps={cardGroupProps}
            categorized={categorized}
          />
        </TabItem>
      </Tabs>
    </TabsWrapper>
  );
};

export default TracingIntegrations;
