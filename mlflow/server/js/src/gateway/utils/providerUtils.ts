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

/**
 * Formats an auth method name for display.
 * Converts snake_case identifiers to readable titles.
 */
export function formatAuthMethodName(authMethod: string): string {
  const formatMap: Record<string, string> = {
    auth_token: 'Auth Token',
    api_key: 'API Key',
    access_key: 'Access Key',
    sts: 'STS (Assume Role)',
    oauth: 'OAuth',
    oauth2: 'OAuth 2.0',
    service_account: 'Service Account',
    bearer_token: 'Bearer Token',
    basic_auth: 'Basic Auth',
    pat: 'Personal Access Token',
  };
  return formatMap[authMethod] ?? authMethod.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * Formats a secret or auth config field name for display.
 * Maps backend field names to human-readable labels.
 */
export function formatSecretFieldName(fieldName: string): string {
  const formatMap: Record<string, string> = {
    // Common secret fields
    api_key: 'API Key',
    // AWS/Bedrock secret fields
    aws_access_key_id: 'Access Key ID',
    aws_secret_access_key: 'Secret Access Key',
    // AWS/Bedrock auth config fields (STS)
    aws_role_arn: 'Role ARN',
    aws_role_name: 'Role Name',
    aws_session_name: 'Session Name',
    aws_external_id: 'External ID',
    aws_region: 'Region',
  };
  return formatMap[fieldName] ?? fieldName.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}
