export interface Model {
  id: string;
  name: string;
  created?: number;
}

export interface AuthConfigField {
  name: string;
  label: string;
  placeholder?: string;
  required?: boolean;
  sensitive?: boolean;
  multiline?: boolean;
  helpText?: string;
}

export interface Provider {
  value: string;
  label: string;
  supportsModelFetch?: boolean;
  default_key_name?: string;
  commonModels: Model[];
  authConfigFields?: AuthConfigField[];
}

export const PROVIDERS: Provider[] = [
  {
    value: 'openai',
    label: 'OpenAI',
    supportsModelFetch: true,
    default_key_name: 'OPENAI_API_KEY',
    commonModels: [
      { id: 'gpt-5-nano', name: 'gpt-5-nano' },
      { id: 'gpt-5-mini', name: 'gpt-5-mini' },
      { id: 'gpt-4o', name: 'gpt-4o' },
      { id: 'gpt-4o-mini', name: 'gpt-4o-mini' },
      { id: 'gpt-4-turbo', name: 'gpt-4-turbo' },
      { id: 'gpt-4', name: 'gpt-4' },
      { id: 'gpt-3.5-turbo', name: 'gpt-3.5-turbo' },
      { id: 'o1', name: 'o1' },
      { id: 'o1-mini', name: 'o1-mini' },
    ],
  },
  {
    value: 'anthropic',
    label: 'Anthropic',
    supportsModelFetch: false,
    default_key_name: 'ANTHROPIC_API_KEY',
    commonModels: [
      { id: 'claude-sonnet-4-5-20250929', name: 'claude-sonnet-4-5-20250929' },
      { id: 'claude-3-7-sonnet-20250219', name: 'claude-3-7-sonnet-20250219' },
      { id: 'claude-3-5-sonnet-20241022', name: 'claude-3-5-sonnet-20241022' },
      { id: 'claude-3-opus-20240229', name: 'claude-3-opus-20240229' },
      { id: 'claude-3-haiku-20240307', name: 'claude-3-haiku-20240307' },
    ],
  },
  {
    value: 'bedrock',
    label: 'AWS Bedrock',
    supportsModelFetch: false,
    default_key_name: 'AWS_SECRET_ACCESS_KEY',
    commonModels: [],
    authConfigFields: [
      {
        name: 'aws_region',
        label: 'AWS Region',
        placeholder: 'e.g., us-east-1',
        required: true,
        helpText: 'The AWS region where your Bedrock service is located',
      },
      {
        name: 'aws_access_key_id',
        label: 'AWS Access Key ID',
        placeholder: 'AKIAIOSFODNN7EXAMPLE', // gitleaks:allow - AWS documentation example placeholder
        required: true,
        sensitive: true,
      },
    ],
  },
  {
    value: 'vertex_ai',
    label: 'Google Vertex AI',
    supportsModelFetch: false,
    default_key_name: 'GOOGLE_API_KEY',
    commonModels: [
      { id: 'gemini-2.0-flash-exp', name: 'gemini-2.0-flash-exp' },
      { id: 'gemini-1.5-pro', name: 'gemini-1.5-pro' },
      { id: 'gemini-1.5-flash', name: 'gemini-1.5-flash' },
    ],
    authConfigFields: [
      {
        name: 'project_id',
        label: 'GCP Project ID',
        placeholder: 'my-project-12345',
        required: true,
        helpText: 'Your Google Cloud Platform project ID',
      },
      {
        name: 'location',
        label: 'Location',
        placeholder: 'us-central1',
        required: true,
        helpText: 'The GCP region (e.g., us-central1, europe-west1)',
      },
      {
        name: 'service_account_json',
        label: 'Service Account JSON',
        placeholder: '{"type": "service_account", ...}',
        required: true,
        sensitive: true,
        multiline: true,
        helpText: 'The full JSON content of your service account key file',
      },
    ],
  },
  {
    value: 'azure',
    label: 'Azure OpenAI',
    supportsModelFetch: false,
    default_key_name: 'AZURE_OPENAI_API_KEY',
    commonModels: [],
    authConfigFields: [
      {
        name: 'azure_endpoint',
        label: 'Azure Endpoint',
        placeholder: 'https://your-resource.openai.azure.com',
        required: true,
        helpText: 'Your Azure OpenAI resource endpoint URL',
      },
      {
        name: 'api_version',
        label: 'API Version',
        placeholder: '2024-02-15-preview',
        required: true,
        helpText: 'The Azure OpenAI API version to use',
      },
      {
        name: 'deployment_name',
        label: 'Deployment Name',
        placeholder: 'my-gpt4-deployment',
        required: true,
        helpText: 'The name of your Azure OpenAI deployment',
      },
    ],
  },
  {
    value: 'databricks',
    label: 'Databricks',
    supportsModelFetch: false,
    default_key_name: 'DATABRICKS_TOKEN',
    commonModels: [],
    authConfigFields: [
      {
        name: 'workspace_url',
        label: 'Workspace URL',
        placeholder: 'https://my-workspace.cloud.databricks.com',
        required: true,
        helpText: 'Your Databricks workspace URL',
      },
    ],
  },
  {
    value: 'custom',
    label: 'Custom Provider',
    supportsModelFetch: false,
    commonModels: [],
  },
];

export const EMPTY_MODEL_ARRAY: Model[] = [];
