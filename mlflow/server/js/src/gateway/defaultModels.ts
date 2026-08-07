export type MlflowDefaultLLMModelTier = 'frontier' | 'everyday' | 'cheap';

export interface MlflowDefaultLLMModel {
  model: string;
  tier: MlflowDefaultLLMModelTier;
}

export interface MlflowDefaultLLMProvider {
  provider: string;
  defaultModel: string;
  models: MlflowDefaultLLMModel[];
}

export const MLFLOW_DEFAULT_LLM_PROVIDERS: MlflowDefaultLLMProvider[] = [
  {
    provider: 'openai',
    defaultModel: 'gpt-5.6-sol',
    models: [
      { model: 'gpt-5.6-sol', tier: 'frontier' },
      { model: 'gpt-5', tier: 'everyday' },
      { model: 'gpt-5-mini', tier: 'cheap' },
    ],
  },
  {
    provider: 'anthropic',
    defaultModel: 'claude-opus-4-8',
    models: [
      { model: 'claude-opus-4-8', tier: 'frontier' },
      { model: 'claude-sonnet-4-6', tier: 'everyday' },
      { model: 'claude-haiku-4-5', tier: 'cheap' },
    ],
  },
  {
    provider: 'gemini',
    defaultModel: 'gemini-3.6-flash',
    models: [
      { model: 'gemini-3.1-pro-preview', tier: 'frontier' },
      { model: 'gemini-3.6-flash', tier: 'everyday' },
      { model: 'gemini-3.1-flash-lite', tier: 'cheap' },
    ],
  },
];

export const getDefaultLLMProvider = (provider: string) =>
  MLFLOW_DEFAULT_LLM_PROVIDERS.find((option) => option.provider === provider);
