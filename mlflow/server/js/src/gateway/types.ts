export interface Provider {
  name: string;
}

export interface Model {
  model: string;
  provider: string;
  supports_function_calling: boolean;
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
  models: Model[];
}

export interface Secret {
  secret_id: string;
  secret_name: string;
  provider?: string;
  auth_config?: Record<string, any>;
  created_at: number;
  updated_at: number;
  created_by?: string;
  updated_by?: string;
}

export interface CreateSecretRequest {
  secret_name: string;
  secret_value: string;
  provider?: string;
  auth_config_json?: string;
  created_by?: string;
}

export interface CreateSecretResponse {
  secret: Secret;
}

export interface GetSecretResponse {
  secret: Secret;
}

export interface UpdateSecretRequest {
  secret_id: string;
  secret_value: string;
  auth_config_json?: string;
  updated_by?: string;
}

export interface UpdateSecretResponse {
  secret: Secret;
}

export interface ListSecretsResponse {
  secrets: Secret[];
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
  created_at: number;
  created_by?: string;
}

export interface Endpoint {
  endpoint_id: string;
  name: string;
  model_mappings: EndpointModelMapping[];
  created_at: number;
  last_updated_at: number;
  created_by?: string;
  last_updated_by?: string;
}

export interface CreateEndpointRequest {
  name?: string;
  model_definition_ids: string[];
  created_by?: string;
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
  model_definition_id: string;
  weight?: number;
  created_by?: string;
}

export interface AttachModelToEndpointResponse {
  mapping: EndpointModelMapping;
}

export interface DetachModelFromEndpointRequest {
  endpoint_id: string;
  model_definition_id: string;
}

export interface EndpointBinding {
  binding_id: string;
  endpoint_id: string;
  experiment_id: string;
  created_at: number;
}

export interface CreateEndpointBindingRequest {
  endpoint_id: string;
  experiment_id: string;
}

export interface CreateEndpointBindingResponse {
  binding: EndpointBinding;
}

export interface ListEndpointBindingsResponse {
  bindings: EndpointBinding[];
}
