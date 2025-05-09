import { defineMessages } from 'react-intl';
import type { QUICKSTART_FLAVOR } from './TraceTableQuickstart.utils';

export interface MessageDescriptor {
  defaultMessage: string;
  description: string;
}

const raw = defineMessages({
  openai: {
    id: 'traces.quickstart.tab.openai',
    defaultMessage: 'OpenAI',
    description: 'Header for OpenAI tab in the MLflow Tracing quickstart guide',
  },
  langchain: {
    id: 'traces.quickstart.tab.langchain',
    defaultMessage: 'LangChain / LangGraph',
    description: 'Header for LangChain / LangGraph tab in the MLflow Tracing quickstart guide',
  },
  llama_index: {
    id: 'traces.quickstart.tab.llamaIndex',
    defaultMessage: 'LlamaIndex',
    description: 'Header for LlamaIndex tab in the MLflow Tracing quickstart guide',
  },
  dspy: {
    id: 'traces.quickstart.tab.dspy',
    defaultMessage: 'DSPy',
    description: 'Header for DSPy tab in the MLflow Tracing quickstart guide',
  },
  crewai: {
    id: 'traces.quickstart.tab.crewai',
    defaultMessage: 'CrewAI',
    description: 'Header for CrewAI tab in the MLflow Tracing quickstart guide',
  },
  autogen: {
    id: 'traces.quickstart.tab.autogen',
    defaultMessage: 'AutoGen',
    description: 'Header for AutoGen tab in the MLflow Tracing quickstart guide',
  },
  anthropic: {
    id: 'traces.quickstart.tab.anthropic',
    defaultMessage: 'Anthropic',
    description: 'Header for Anthropic tab in the MLflow Tracing quickstart guide',
  },
  bedrock: {
    id: 'traces.quickstart.tab.bedrock',
    defaultMessage: 'Bedrock',
    description: 'Header for Bedrock tab in the MLflow Tracing quickstart guide',
  },
  litellm: {
    id: 'traces.quickstart.tab.litellm',
    defaultMessage: 'LiteLLM',
    description: 'Header for LiteLLM tab in the MLflow Tracing quickstart guide',
  },
  gemini: {
    id: 'traces.quickstart.tab.gemini',
    defaultMessage: 'Gemini',
    description: 'Header for Gemini tab in the MLflow Tracing quickstart guide',
  },
  custom: {
    id: 'traces.quickstart.tab.custom',
    defaultMessage: 'Custom',
    description: 'Header for custom tracing tab in the MLflow Tracing quickstart guide',
  },
});

export const QUICKSTART_TAB_MESSAGES: Record<QUICKSTART_FLAVOR, MessageDescriptor> = raw;
