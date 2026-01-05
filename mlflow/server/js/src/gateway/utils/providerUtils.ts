export const COMMON_PROVIDERS = ['openai', 'anthropic', 'databricks', 'bedrock', 'gemini', 'vertex_ai', 'xai'] as const;

export interface ProviderGroup {
  groupId: string;
  displayName: string;
  defaultProvider: string;
  providers: string[];
}

export const PROVIDER_GROUPS: Record<string, Omit<ProviderGroup, 'providers'>> = {
  openai_azure: {
    groupId: 'openai_azure',
    displayName: 'OpenAI / Azure OpenAI',
    defaultProvider: 'openai',
  },
  vertex_ai: {
    groupId: 'vertex_ai',
    displayName: 'Google Vertex AI',
    defaultProvider: 'vertex_ai',
  },
};

export function getProviderGroupId(provider: string): string | null {
  if (provider === 'openai' || provider === 'azure') {
    return 'openai_azure';
  }
  if (provider === 'vertex_ai' || provider.startsWith('vertex_ai-')) {
    return 'vertex_ai';
  }
  return null;
}

export function buildProviderGroups(providers: string[]): {
  groups: ProviderGroup[];
  ungroupedProviders: string[];
} {
  const groupedProviders = new Map<keyof typeof PROVIDER_GROUPS, string[]>();
  const ungroupedProviders: string[] = [];

  for (const provider of providers) {
    const groupId = getProviderGroupId(provider);
    if (groupId) {
      const existing = groupedProviders.get(groupId) ?? [];
      existing.push(provider);
      groupedProviders.set(groupId, existing);
    } else {
      ungroupedProviders.push(provider);
    }
  }

  const groups: ProviderGroup[] = [];

  const openaiAzureProviders = groupedProviders.get('openai_azure');
  if (openaiAzureProviders && openaiAzureProviders.length > 0) {
    const preferredOrder = ['openai', 'azure'];
    openaiAzureProviders.sort((a, b) => {
      const aIndex = preferredOrder.indexOf(a);
      const bIndex = preferredOrder.indexOf(b);
      if (aIndex !== -1 && bIndex !== -1) return aIndex - bIndex;
      if (aIndex !== -1) return -1;
      if (bIndex !== -1) return 1;
      return a.localeCompare(b);
    });
    groups.push({
      ...PROVIDER_GROUPS['openai_azure'],
      providers: openaiAzureProviders,
    });
  }

  const vertexProviders = groupedProviders.get('vertex_ai');
  if (vertexProviders && vertexProviders.length > 0) {
    vertexProviders.sort((a, b) => {
      if (a === 'vertex_ai') return -1;
      if (b === 'vertex_ai') return 1;
      return a.localeCompare(b);
    });
    groups.push({
      ...PROVIDER_GROUPS['vertex_ai'],
      providers: vertexProviders,
    });
  }

  return { groups, ungroupedProviders };
}

const PROVIDER_DISPLAY_NAMES: Record<string, string> = {
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

const VERTEX_AI_VARIANT_NAMES: Record<string, string> = {
  'vertex_ai-anthropic': 'Vertex AI (Anthropic)',
  'vertex_ai-llama3': 'Vertex AI (Llama 3)',
  'vertex_ai-mistral': 'Vertex AI (Mistral)',
  'vertex_ai-ai21': 'Vertex AI (AI21)',
  'vertex_ai-codey': 'Vertex AI (Codey)',
  'vertex_ai-image-models': 'Vertex AI (Image)',
  'vertex_ai-code-text-models': 'Vertex AI (Code)',
  'vertex_ai-text-models': 'Vertex AI (Text)',
  'vertex_ai-chat-models': 'Vertex AI (Chat)',
  'vertex_ai-embedding-models': 'Vertex AI (Embedding)',
  'vertex_ai-vision-models': 'Vertex AI (Vision)',
};

export function formatProviderName(provider: string): string {
  if (PROVIDER_DISPLAY_NAMES[provider]) {
    return PROVIDER_DISPLAY_NAMES[provider];
  }

  if (VERTEX_AI_VARIANT_NAMES[provider]) {
    return VERTEX_AI_VARIANT_NAMES[provider];
  }

  if (provider.startsWith('vertex_ai-')) {
    const variant = provider.replace('vertex_ai-', '');
    const formattedVariant = variant.replace(/[-_]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
    return `Vertex AI (${formattedVariant})`;
  }

  return provider.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
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

const PROVIDER_FIELD_ORDER: Record<string, string[]> = {
  databricks: ['client_id', 'client_secret', 'api_base'],
};

export function sortFieldsByProvider<T extends { name: string }>(fields: T[], provider: string): T[] {
  const fieldOrder = PROVIDER_FIELD_ORDER[provider];
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
