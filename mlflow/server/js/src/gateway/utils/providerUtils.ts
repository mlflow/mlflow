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
    xai: 'xAI',
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
