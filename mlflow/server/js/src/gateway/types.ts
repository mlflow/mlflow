export interface Provider {
  name: string;
}

export interface ProviderModel {
  model: string;
  provider: string;
  supports_function_calling: boolean;
  supports_vision?: boolean;
  supports_reasoning?: boolean;
  supports_prompt_caching?: boolean;
  supports_response_schema?: boolean;
  max_input_tokens?: number;
  max_output_tokens?: number;
  input_cost_per_token?: number;
  output_cost_per_token?: number;
  deprecation_date?: string;
}

export interface SecretField {
  name: string;
  type: string;
  description?: string;
  required: boolean;
}

export interface ConfigField {
  name: string;
  type: string;
  description?: string;
  required: boolean;
}

export interface AuthMode {
  mode: string;
  display_name: string;
  description?: string;
  secret_fields: SecretField[];
  config_fields: ConfigField[];
}

export interface ProviderConfig {
  auth_modes: AuthMode[];
  default_mode: string;
}

export interface ProvidersResponse {
  providers: string[];
}

export interface ModelsResponse {
  models: ProviderModel[];
}

export interface SecretInfo {
  secret_id: string;
  secret_name: string;
  masked_values: Record<string, string>;
  provider?: string;
  auth_config?: Record<string, string>;
  created_at: number;
  last_updated_at: number;
  created_by?: string;
  last_updated_by?: string;
}

export interface CreateSecretRequest {
  secret_name: string;
  secret_value: Record<string, string>;
  provider?: string;
  auth_config?: Record<string, string>;
  created_by?: string;
}

export interface CreateSecretInfoResponse {
  secret: SecretInfo;
}

export interface GetSecretInfoResponse {
  secret: SecretInfo;
}

export interface UpdateSecretRequest {
  secret_id: string;
  secret_value: Record<string, string>;
  auth_config?: Record<string, string>;
  updated_by?: string;
}

export interface UpdateSecretInfoResponse {
  secret: SecretInfo;
}

export interface ListSecretInfosResponse {
  secrets: SecretInfo[];
}

export interface ModelDefinition {
  model_definition_id: string;
  name: string;
  secret_id: string;
  secret_name: string;
  provider: string;
  model_name: string;
  created_at: number;
  last_updated_at: number;
  created_by?: string;
  last_updated_by?: string;
  endpoint_count: number;
}

export interface EndpointModelMapping {
  mapping_id: string;
  endpoint_id: string;
  model_definition_id: string;
  model_definition?: ModelDefinition;
  weight: number;
  linkage_type?: string;
  fallback_order?: number;
  created_at: number;
  created_by?: string;
}

export interface GatewayEndpointModelConfig {
  model_definition_id: string;
  linkage_type: string;
  weight?: number;
  fallback_order?: number;
}

export interface Endpoint {
  endpoint_id: string;
  name: string;
  model_mappings: EndpointModelMapping[];
  created_at: number;
  last_updated_at: number;
  created_by?: string;
  last_updated_by?: string;
  routing_strategy?: string;
  fallback_config?: {
    strategy: string;
    max_attempts: number;
  };
}

export interface CreateEndpointRequest {
  name?: string;
  model_configs: GatewayEndpointModelConfig[];
  created_by?: string;
  routing_strategy?: string;
  fallback_config?: {
    strategy: string;
    max_attempts: number;
  };
}

export interface CreateEndpointResponse {
  endpoint: Endpoint;
}

export interface GetEndpointResponse {
  endpoint: Endpoint;
}

export interface UpdateEndpointRequest {
  endpoint_id: string;
  name?: string;
  updated_by?: string;
  routing_strategy?: string;
  fallback_config?: {
    strategy: string;
    max_attempts: number;
  };
  model_configs?: GatewayEndpointModelConfig[];
}

export interface UpdateEndpointResponse {
  endpoint: Endpoint;
}

export interface ListEndpointsResponse {
  endpoints: Endpoint[];
}

// Model Definition CRUD
export interface CreateModelDefinitionRequest {
  name: string;
  secret_id: string;
  provider: string;
  model_name: string;
  created_by?: string;
}

export interface CreateModelDefinitionResponse {
  model_definition: ModelDefinition;
}

export interface GetModelDefinitionResponse {
  model_definition: ModelDefinition;
}

export interface ListModelDefinitionsResponse {
  model_definitions: ModelDefinition[];
}

export interface UpdateModelDefinitionRequest {
  model_definition_id: string;
  name?: string;
  secret_id?: string;
  provider?: string;
  model_name?: string;
  last_updated_by?: string;
}

export interface UpdateModelDefinitionResponse {
  model_definition: ModelDefinition;
}

// Attach/Detach Model to Endpoint
export interface AttachModelToEndpointRequest {
  endpoint_id: string;
  model_config: GatewayEndpointModelConfig;
  created_by?: string;
}

export interface AttachModelToEndpointResponse {
  mapping: EndpointModelMapping;
}

export interface DetachModelFromEndpointRequest {
  endpoint_id: string;
  model_definition_id: string;
}

export type ResourceType = 'scorer_job';

export interface EndpointBinding {
  endpoint_id: string;
  resource_type: ResourceType;
  resource_id: string;
  created_at: number;
  last_updated_at?: number;
  created_by?: string;
  last_updated_by?: string;
}

export interface CreateEndpointBindingRequest {
  endpoint_id: string;
  resource_type: ResourceType;
  resource_id: string;
  created_by?: string;
}

export interface CreateEndpointBindingResponse {
  binding: EndpointBinding;
}

export interface ListEndpointBindingsResponse {
  bindings: EndpointBinding[];
}

export interface SecretsConfigResponse {
  secrets_available: boolean;
  using_default_passphrase: boolean;
}

// Usage Tracking Types

export interface UsageMetricsEntry {
  endpoint_id: string;
  time_bucket: number;
  bucket_size: number; // Size in seconds
  total_invocations: number;
  successful_invocations: number;
  failed_invocations: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  total_tokens: number;
  total_cost: number;
  avg_latency_ms: number;
  // Error/success rates included in the response for convenience
  success_rate: number;
  error_rate: number;
}

export interface UsageMetricsResponse {
  metrics: UsageMetricsEntry[];
}
