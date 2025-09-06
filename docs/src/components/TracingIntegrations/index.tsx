import React from 'react';
import { CardGroup, SmallLogoCard } from '../Card';
import useBaseUrl from '@docusaurus/useBaseUrl';

interface TracingIntegration {
  id: string;
  name: string;
  logoPath: string;
  link: string;
}

// Centralized integration definitions
const TRACING_INTEGRATIONS: TracingIntegration[] = [
  {
    id: 'openai',
    name: 'OpenAI',
    logoPath: '/images/logos/openai-logo.png',
    link: '/genai/tracing/integrations/listing/openai',
  },
  {
    id: 'langchain',
    name: 'LangChain',
    logoPath: '/images/logos/langchain-logo.png',
    link: '/genai/tracing/integrations/listing/langchain',
  },
  {
    id: 'langgraph',
    name: 'LangGraph',
    logoPath: '/images/logos/langgraph-logo.png',
    link: '/genai/tracing/integrations/listing/langgraph',
  },
  {
    id: 'llama_index',
    name: 'LlamaIndex',
    logoPath: '/images/logos/llamaindex-logo.svg',
    link: '/genai/tracing/integrations/listing/llama_index',
  },
  {
    id: 'anthropic',
    name: 'Anthropic',
    logoPath: '/images/logos/anthropic-logo.svg',
    link: '/genai/tracing/integrations/listing/anthropic',
  },
  {
    id: 'dspy',
    name: 'DSPy',
    logoPath: '/images/logos/dspy-logo.png',
    link: '/genai/tracing/integrations/listing/dspy',
  },
  {
    id: 'bedrock',
    name: 'Bedrock',
    logoPath: '/images/logos/bedrock-logo.png',
    link: '/genai/tracing/integrations/listing/bedrock',
  },
  {
    id: 'semantic_kernel',
    name: 'Semantic Kernel',
    logoPath: '/images/logos/semantic-kernel-logo.png',
    link: '/genai/tracing/integrations/listing/semantic_kernel',
  },
  {
    id: 'autogen',
    name: 'AutoGen',
    logoPath: '/images/logos/autogen-logo.png',
    link: '/genai/tracing/integrations/listing/autogen',
  },
  {
    id: 'ag2',
    name: 'AG2',
    logoPath: '/images/logos/ag2-logo.png',
    link: '/genai/tracing/integrations/listing/ag2',
  },
  {
    id: 'gemini',
    name: 'Gemini',
    logoPath: '/images/logos/google-gemini-logo.svg',
    link: '/genai/tracing/integrations/listing/gemini',
  },
  {
    id: 'litellm',
    name: 'LiteLLM',
    logoPath: '/images/logos/litellm-logo.jpg',
    link: '/genai/tracing/integrations/listing/litellm',
  },
  {
    id: 'crewai',
    name: 'CrewAI',
    logoPath: '/images/logos/crewai-logo.png',
    link: '/genai/tracing/integrations/listing/crewai',
  },
  {
    id: 'openai-agent',
    name: 'OpenAI Agent',
    logoPath: '/images/logos/openai-agent-logo.png',
    link: '/genai/tracing/integrations/listing/openai-agent',
  },
  {
    id: 'ollama',
    name: 'Ollama',
    logoPath: '/images/logos/ollama-logo.png',
    link: '/genai/tracing/integrations/listing/ollama',
  },
  {
    id: 'pydantic_ai',
    name: 'PydanticAI',
    logoPath: '/images/logos/pydanticai-logo.png',
    link: '/genai/tracing/integrations/listing/pydantic_ai',
  },
  {
    id: 'agno',
    name: 'Agno',
    logoPath: '/images/logos/agno-logo.png',
    link: '/genai/tracing/integrations/listing/agno',
  },
  {
    id: 'smolagents',
    name: 'Smolagents',
    logoPath: '/images/logos/smolagents-logo.png',
    link: '/genai/tracing/integrations/listing/smolagents',
  },
  {
    id: 'groq',
    name: 'Groq',
    logoPath: '/images/logos/groq-logo.svg',
    link: '/genai/tracing/integrations/listing/groq',
  },
  {
    id: 'mistral',
    name: 'Mistral',
    logoPath: '/images/logos/mistral-ai-logo.svg',
    link: '/genai/tracing/integrations/listing/mistral',
  },
  {
    id: 'instructor',
    name: 'Instructor',
    logoPath: '/images/logos/instructor-logo.svg',
    link: '/genai/tracing/integrations/listing/instructor',
  },
  {
    id: 'strands',
    name: 'Strands Agent SDK',
    logoPath: '/images/logos/strands-logo.png',
    link: '/genai/tracing/integrations/listing/strands',
  },
  {
    id: 'deepseek',
    name: 'DeepSeek',
    logoPath: '/images/logos/deepseek-logo.png',
    link: '/genai/tracing/integrations/listing/deepseek',
  },
  {
    id: 'txtai',
    name: 'txtai',
    logoPath: '/images/logos/txtai-logo.png',
    link: '/genai/tracing/integrations/listing/txtai',
  },
  {
    id: 'haystack',
    name: 'Haystack',
    logoPath: '/images/logos/haystack-logo.png',
    link: '/genai/tracing/integrations/listing/haystack',
  },
  {
    id: 'claude_code',
    name: 'Claude Code',
    logoPath: '/images/logos/claude-code-logo.svg',
    link: '/genai/tracing/integrations/listing/claude_code',
  },
];

interface TracingIntegrationsProps {
  cardGroupProps?: {
    isSmall?: boolean;
    cols?: number;
    noGap?: boolean;
  };
}

export const TracingIntegrations: React.FC<TracingIntegrationsProps> = ({ cardGroupProps = {} }) => {
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
