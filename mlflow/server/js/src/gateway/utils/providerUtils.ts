export const COMMON_PROVIDERS = [
  'openai',
  'azure',
  'anthropic',
  'databricks',
  'bedrock',
  'gemini',
  'vertex_ai',
  'xai',
  'mistral',
  'groq',
  'deepseek',
  'portkey',
  'openrouter',
  'ollama',
  'together_ai',
] as const;

const PROVIDER_DISPLAY_NAMES = {
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  bedrock: 'Amazon Bedrock',
  gemini: 'Google Gemini',
  vertex_ai: 'Google Vertex AI',
  azure: 'Azure OpenAI',
  portkey: 'Portkey',
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
  deepseek: 'DeepSeek',
  openrouter: 'OpenRouter',
  ollama: 'Ollama',
} satisfies Record<string, string>;

export function formatProviderName(provider: string): string {
  if (provider in PROVIDER_DISPLAY_NAMES) {
    return PROVIDER_DISPLAY_NAMES[provider as keyof typeof PROVIDER_DISPLAY_NAMES];
  }

  return provider.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function formatAuthMethodName(authMethod: string): string {
  const formatMap = {
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
  } satisfies Record<string, string>;

  if (authMethod in formatMap) {
    return formatMap[authMethod as keyof typeof formatMap];
  }

  return authMethod.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function formatCredentialFieldName(fieldName: string): string {
  const formatMap = {
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
  } satisfies Record<string, string>;

  if (fieldName in formatMap) {
    return formatMap[fieldName as keyof typeof formatMap];
  }

  return fieldName.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

const PROVIDER_FIELD_ORDER = {
  databricks: ['client_id', 'client_secret', 'api_base'],
} satisfies Record<string, string[]>;

export function sortFieldsByProvider<T extends { name: string }>(fields: T[], provider: string): T[] {
  if (!(provider in PROVIDER_FIELD_ORDER)) {
    return fields;
  }

  const fieldOrder = PROVIDER_FIELD_ORDER[provider as keyof typeof PROVIDER_FIELD_ORDER];
  if (!fieldOrder) {
    return fields;
  }

  return [...fields].sort((a, b) => {
    const aIndex = fieldOrder.indexOf(a.name);
    const bIndex = fieldOrder.indexOf(b.name);

    if (aIndex === -1 && bIndex === -1) return 0;
    if (aIndex === -1) return 1;
    if (bIndex === -1) return -1;

    return aIndex - bIndex;
  });
}
