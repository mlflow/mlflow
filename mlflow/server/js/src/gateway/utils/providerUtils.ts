export const COMMON_PROVIDERS = ['openai', 'anthropic', 'bedrock', 'gemini', 'azure', 'groq', 'databricks'] as const;

export function groupProviders(providers: string[]): {
  common: string[];
  other: string[];
} {
  const commonSet = new Set<string>(COMMON_PROVIDERS);
  const common: string[] = [];
  const other: string[] = [];

  for (const provider of providers) {
    if (commonSet.has(provider)) {
      common.push(provider);
    } else {
      other.push(provider);
    }
  }

  common.sort((a, b) => COMMON_PROVIDERS.indexOf(a as any) - COMMON_PROVIDERS.indexOf(b as any));
  other.sort((a, b) => a.localeCompare(b));

  return { common, other };
}

export function formatProviderName(provider: string): string {
  const formatMap: Record<string, string> = {
    openai: 'OpenAI',
    anthropic: 'Anthropic',
    bedrock: 'Amazon Bedrock',
    gemini: 'Google Gemini',
    vertex_ai: 'Google Vertex AI',
    azure: 'Azure OpenAI',
    groq: 'Groq',
    databricks: 'Databricks',
    cohere: 'Cohere',
    mistral: 'Mistral AI',
    together_ai: 'Together AI',
    fireworks_ai: 'Fireworks AI',
    replicate: 'Replicate',
    huggingface: 'Hugging Face',
    ai21: 'AI21',
    perplexity: 'Perplexity',
    deepinfra: 'DeepInfra',
    nvidia_nim: 'NVIDIA NIM',
    cerebras: 'Cerebras',
  };
  return formatMap[provider] ?? provider.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}
