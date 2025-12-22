export const COMMON_PROVIDERS = [
  'openai',
  'anthropic',
  'databricks',
  'bedrock',
  'gemini',
  'vertex_ai',
  'azure',
  'xai',
] as const;

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

export function formatCredentialFieldName(fieldName: string): string {
  const formatMap: Record<string, string> = {
    api_key: 'API Key',
    aws_access_key_id: 'AWS Access Key ID',
    aws_secret_access_key: 'AWS Secret Access Key',
    aws_session_token: 'AWS Session Token',
    aws_region_name: 'AWS Region',
    aws_role_name: 'AWS Role Name',
    aws_session_name: 'AWS Session Name',
    client_secret: 'Client Secret',
    client_id: 'Client ID',
    tenant_id: 'Tenant ID',
    api_base: 'API Base URL',
    api_version: 'API Version',
    vertex_credentials: 'Service Account Credentials',
    vertex_project: 'Project ID',
    vertex_location: 'Location',
    databricks_token: 'Databricks Token',
    databricks_host: 'Databricks Host',
  };
  return formatMap[fieldName] ?? fieldName.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}
