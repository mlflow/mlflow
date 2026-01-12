export interface OpenAICompatibleProvider {
  id: string;
  name: string;
  displayName?: string; // e.g., "Kimi (Moonshot AI)" - if different from name
  codeRefName?: string; // e.g., "Kimi" - for code comments, defaults to name
  baseUrl: string;
  tsBaseUrl?: string; // if different from Python
  apiKeyPlaceholder: string;
  sampleModel: string;
  tsSampleModel?: string; // if different in TS example
}

export const OPENAI_COMPATIBLE_PROVIDERS: OpenAICompatibleProvider[] = [
  {
    id: 'byteplus',
    name: 'BytePlus',
    baseUrl: 'https://ark.ap-southeast.bytepluses.com/api/v3/',
    apiKeyPlaceholder: '<your_byteplus_api_key>',
    sampleModel: '<your_byteplus_model>',
  },
  {
    id: 'cohere',
    name: 'Cohere',
    baseUrl: 'https://api.cohere.ai/compatibility/v1',
    tsBaseUrl: 'https://api.cohere.com/v1',
    apiKeyPlaceholder: '<your_cohere_api_key>',
    sampleModel: 'command-a-03-2025',
    tsSampleModel: '<your_cohere_model>',
  },
  {
    id: 'deepseek',
    name: 'DeepSeek',
    baseUrl: 'https://api.deepseek.com',
    apiKeyPlaceholder: '<your_deepseek_api_key>',
    sampleModel: 'deepseek-chat',
  },
  {
    id: 'moonshot',
    name: 'Moonshot AI',
    displayName: 'Kimi (Moonshot AI)',
    codeRefName: 'Kimi',
    baseUrl: 'https://api.moonshot.ai/v1',
    apiKeyPlaceholder: '<MOONSHOT_API_KEY>',
    sampleModel: 'moonshot-v1-8k',
  },
  {
    id: 'novitaai',
    name: 'Novita AI',
    baseUrl: 'https://api.novita.ai/openai',
    apiKeyPlaceholder: '<your_novita_api_key>',
    sampleModel: 'deepseek/deepseek-r1',
  },
  {
    id: 'qwen',
    name: 'Qwen',
    displayName: 'Qwen (DashScope)',
    baseUrl: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
    apiKeyPlaceholder: '<DASHSCOPE_API_KEY>',
    sampleModel: 'qwen-plus',
  },
  {
    id: 'togetherai',
    name: 'Together AI',
    baseUrl: 'https://api.together.xyz/v1',
    apiKeyPlaceholder: '<your_together_api_key>',
    sampleModel: 'openai/gpt-oss-20b',
  },
  {
    id: 'xai-grok',
    name: 'xAI',
    displayName: 'xAI / Grok',
    codeRefName: 'Grok',
    baseUrl: 'https://api.x.ai/v1',
    apiKeyPlaceholder: '<your_grok_api_key>',
    sampleModel: 'grok-4',
  },
];

export function getProvider(id: string): OpenAICompatibleProvider | undefined {
  return OPENAI_COMPATIBLE_PROVIDERS.find((p) => p.id === id);
}
