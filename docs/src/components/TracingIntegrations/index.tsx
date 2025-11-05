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

// Centralized integration definitions with categories
const TRACING_INTEGRATIONS: TracingIntegration[] = [
  // Agent Frameworks
  {
    id: 'langchain',
    name: 'LangChain',
    logoPath: '/images/logos/langchain-logo.png',
    link: '/genai/tracing/integrations/listing/langchain',
    category: 'Agent Frameworks',
  },
  {
    id: 'langgraph',
    name: 'LangGraph',
    logoPath: '/images/logos/langgraph-logo.png',
    link: '/genai/tracing/integrations/listing/langgraph',
    category: 'Agent Frameworks',
  },
  {
    id: 'crewai',
    name: 'CrewAI',
    logoPath: '/images/logos/crewai-logo.png',
    link: '/genai/tracing/integrations/listing/crewai',
    category: 'Agent Frameworks',
  },
  {
    id: 'autogen',
    name: 'AutoGen',
    logoPath: '/images/logos/autogen-logo.png',
    link: '/genai/tracing/integrations/listing/autogen',
    category: 'Agent Frameworks',
  },
  {
    id: 'ag2',
    name: 'AG2',
    logoPath: '/images/logos/ag2-logo.png',
    link: '/genai/tracing/integrations/listing/ag2',
    category: 'Agent Frameworks',
  },
  {
    id: 'agno',
    name: 'Agno',
    logoPath: '/images/logos/agno-logo.png',
    link: '/genai/tracing/integrations/listing/agno',
    category: 'Agent Frameworks',
  },
  {
    id: 'pydantic_ai',
    name: 'PydanticAI',
    logoPath: '/images/logos/pydanticai-logo.png',
    link: '/genai/tracing/integrations/listing/pydantic_ai',
    category: 'Agent Frameworks',
  },
  {
    id: 'smolagents',
    name: 'Smolagents',
    logoPath: '/images/logos/smolagents-logo.png',
    link: '/genai/tracing/integrations/listing/smolagents',
    category: 'Agent Frameworks',
  },
  {
    id: 'openai-agent',
    name: 'OpenAI Agent',
    logoPath: '/images/logos/openai-agent-logo.png',
    link: '/genai/tracing/integrations/listing/openai-agent',
    category: 'Agent Frameworks',
  },
  {
    id: 'strands',
    name: 'Strands Agent SDK',
    logoPath: '/images/logos/strands-logo.png',
    link: '/genai/tracing/integrations/listing/strands',
    category: 'Agent Frameworks',
  },
  {
    id: 'semantic_kernel',
    name: 'Semantic Kernel',
    logoPath: '/images/logos/semantic-kernel-logo.png',
    link: '/genai/tracing/integrations/listing/semantic_kernel',
    category: 'Agent Frameworks',
  },
  {
    id: 'haystack',
    name: 'Haystack',
    logoPath: '/images/logos/haystack-logo.png',
    link: '/genai/tracing/integrations/listing/haystack',
    category: 'Agent Frameworks',
  },
  {
    id: 'llama_index',
    name: 'LlamaIndex',
    logoPath: '/images/logos/llamaindex-logo.svg',
    link: '/genai/tracing/integrations/listing/llama_index',
    category: 'Agent Frameworks',
  },
  {
    id: 'dspy',
    name: 'DSPy',
    logoPath: '/images/logos/dspy-logo.png',
    link: '/genai/tracing/integrations/listing/dspy',
    category: 'Agent Frameworks',
  },
  {
    id: 'instructor',
    name: 'Instructor',
    logoPath: '/images/logos/instructor-logo.svg',
    link: '/genai/tracing/integrations/listing/instructor',
    category: 'Tools',
  },
  {
    id: 'txtai',
    name: 'txtai',
    logoPath: '/images/logos/txtai-logo.png',
    link: '/genai/tracing/integrations/listing/txtai',
    category: 'Agent Frameworks',
  },
  // Model Providers
  {
    id: 'openai',
    name: 'OpenAI',
    logoPath: '/images/logos/openai-logo.png',
    link: '/genai/tracing/integrations/listing/openai',
    category: 'Model Providers',
  },
  {
    id: 'anthropic',
    name: 'Anthropic',
    logoPath: '/images/logos/anthropic-logo.svg',
    link: '/genai/tracing/integrations/listing/anthropic',
    category: 'Model Providers',
  },
  {
    id: 'bedrock',
    name: 'Bedrock',
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
    id: 'mistral',
    name: 'Mistral',
    logoPath: '/images/logos/mistral-ai-logo.svg',
    link: '/genai/tracing/integrations/listing/mistral',
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
    id: 'litellm',
    name: 'LiteLLM',
    logoPath: '/images/logos/litellm-logo.jpg',
    link: '/genai/tracing/integrations/listing/litellm',
    category: 'Model Providers',
  },
  // Tools
  {
    id: 'claude_code',
    name: 'Claude Code',
    logoPath: '/images/logos/claude-code-logo.svg',
    link: '/genai/tracing/integrations/listing/claude_code',
    category: 'Tools',
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

export const TracingIntegrations: React.FC<TracingIntegrationsProps> = ({
  cardGroupProps = {},
  categorized = false,
}) => {
  if (categorized) {
    // Group integrations by category
    const categories = TRACING_INTEGRATIONS.reduce(
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
          const integrations = categories[category] || [];
          if (integrations.length === 0) return null;

          return (
            <div key={category} style={{ marginBottom: '2rem' }}>
              <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>{category}</h2>
              <CardGroup {...cardGroupProps}>
                {integrations.map((integration) => (
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
    <CardGroup {...cardGroupProps}>
      {TRACING_INTEGRATIONS.map((integration) => (
        <SmallLogoCard key={integration.id} link={integration.link}>
          <span>
            <img src={useBaseUrl(integration.logoPath)} alt={`${integration.name} Logo`} />
          </span>
        </SmallLogoCard>
      ))}
    </CardGroup>
  );
};

export default TracingIntegrations;
